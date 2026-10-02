import asyncio
import contextlib
from datetime import UTC, datetime

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from tests.fake_mcp import FakeRuns, build_fake_app, serve
from tests.fake_talk import FakeTalk
from tests.fakes import (
    FailingChatModel,
    GatedChatModel,
    HangingChatModel,
    RecordingChatModel,
    ScriptedChatModel,
    StaticGroups,
)
from tests.test_chat import Decider
from uriel.agent.mcp_tools import McpToolbox
from uriel.gateway import runner as runner_module
from uriel.gateway.app import Coworker, run_coworker
from uriel.gateway.chat import EMPTY_ANSWER, ChatService, collect
from uriel.gateway.runner import (
    FAILED,
    GATEWAY_ERROR,
    MAX_RUNS_PER_TICK,
    NOT_A_MEMBER,
    TIMED_OUT,
    ClaimedRun,
    ScheduledRunner,
    local_due,
    run_input,
)
from uriel.gateway.talk_api import Room, TalkClient
from uriel.principal import INTERNAL_GROUP, Principal

RUNS = FakeRuns()
GROUPS = {"kid": {"family"}, "dad": {"admins", "family"}}


@pytest.fixture(scope="module")
def mcp_url():
    with serve(build_fake_app("test-key", RUNS)) as url:
        yield url


@pytest.fixture(autouse=True)
def fresh_runs():
    RUNS.due.clear()
    RUNS.finished.clear()
    RUNS.claimed_by.clear()


def due(**overrides):
    return {
        "run_id": 1,
        "owner": "kid",
        "owner_sub": "sub-kid",
        "title": "F1 digest",
        "prompt": "news about Formula 1",
        "tools": ["web_search"],
        "due_at": "2026-10-01T13:00:00Z",
        "tz": "America/Toronto",
    } | overrides


def make(mcp_url, replies=(), *, decider="tools", groups=None, model=None, chat_model=None):
    fake = FakeTalk()
    talk = TalkClient(
        "http://talk.test", "uriel", "talk-password", transport=httpx.ASGITransport(app=fake.app())
    )
    model = model or RecordingChatModel(messages=iter(replies))
    toolbox = McpToolbox(mcp_url, "test-key")
    chat = ChatService(
        chat_model=chat_model or ScriptedChatModel(messages=iter([])),
        background_model=model,
        decider=Decider(decider),
        toolbox=toolbox,
        checkpointer=InMemorySaver(),
        conversations=None,
        decision_log=None,
        recursion_limit=10,
        max_message_chars=8000,
    )
    directory = StaticGroups(GROUPS if groups is None else groups)
    runner = ScheduledRunner(toolbox=toolbox, chat=chat, directory=directory, talk=talk)
    return fake, runner, model, directory, chat


def test_local_due_time_follows_dst_and_falls_back_to_utc():
    assert local_due(datetime(2026, 10, 1, 13, tzinfo=UTC), "America/Toronto") == (
        "Thu 1 Oct 2026, 09:00 (America/Toronto)"
    )
    # Toronto leaves daylight saving time on 1 November 2026.
    assert local_due(datetime(2026, 11, 2, 14, tzinfo=UTC), "America/Toronto") == (
        "Mon 2 Nov 2026, 09:00 (America/Toronto)"
    )
    assert local_due(datetime(2026, 10, 1, 13, tzinfo=UTC), "Mars/Olympus") == "Thu 1 Oct 2026, 13:00 (UTC)"


def test_the_run_input_names_the_task_and_its_local_time():
    assert run_input(ClaimedRun.from_tool(due())) == (
        'Scheduled task "F1 digest" (Thu 1 Oct 2026, 09:00 (America/Toronto)): news about Formula 1'
    )


async def test_a_due_run_is_answered_in_the_owners_new_dm_and_finished_ok(mcp_url):
    search = {"name": "web_search", "args": {"query": "Formula 1 news"}, "id": "c1"}
    answer = "Verstappen won. https://example.com/f1"
    fake, runner, model, directory, _ = make(mcp_url, [AIMessage("", tool_calls=[search]), AIMessage(answer)])
    RUNS.due.append(due())
    assert await runner.run_once() == 1
    [post] = fake.posted("dm-kid")
    assert post["message"] == answer
    assert RUNS.finished == [
        {"run_id": 1, "status": "ok", "summary": answer, "error": None, "talk_message_id": post["id"]}
    ]
    claim = RUNS.claimed_by[0]
    assert (claim["user"], claim["groups"]) == ("uriel-gateway", [INTERNAL_GROUP])
    assert model.seen[0][-1].content == run_input(ClaimedRun.from_tool(due()))
    assert directory.calls == [("kid", True)]


async def test_an_existing_dm_is_reused(mcp_url):
    fake, runner, *_ = make(mcp_url, [AIMessage("Nothing new today.")], decider="direct")
    fake.add_room("kid-and-uriel", Room.ONE_TO_ONE, name="kid")
    RUNS.due.append(due())
    await runner.run_once()
    assert fake.posted("kid-and-uriel")[0]["message"].startswith("Nothing new today.")
    assert "dm-kid" not in fake.rooms


async def test_the_model_gets_the_schedules_unattended_tools_the_owner_may_use(mcp_url):
    _, runner, model, *_ = make(mcp_url, [AIMessage("Done.")])
    RUNS.due.append(due(owner="dad", tools=["web_search", "search_documents"]))
    await runner.run_once()
    assert model.bound == [["search_documents", "web_search"]]
    assert RUNS.finished[0]["status"] == "ok"


async def test_a_tool_that_cant_run_unattended_fails_the_run_and_says_so(mcp_url):
    fake, runner, model, *_ = make(mcp_url)
    RUNS.due.append(due(tools=["web_search", "remember"]))
    await runner.run_once()
    [finished] = RUNS.finished
    assert finished["status"] == "failed" and "remember" in finished["error"]
    assert [p["message"] for p in fake.posted("dm-kid")] == [FAILED.format(title="F1 digest")]
    assert model.seen == []


@pytest.mark.parametrize("tools", [["web_search"], []])
async def test_an_owner_removed_from_family_gets_a_failed_run_without_running_or_posting(mcp_url, tools):
    fake, runner, model, directory, _ = make(mcp_url, [AIMessage("unused")], groups={"kid": {"friends"}})
    RUNS.due.append(due(tools=tools))
    await runner.run_once()
    assert [(f["status"], f["error"]) for f in RUNS.finished] == [("failed", NOT_A_MEMBER)]
    assert directory.calls == [("kid", True)]
    assert model.seen == [] and fake.rooms == {}


async def test_an_owner_no_longer_in_lldap_fails_without_a_dm(mcp_url):
    fake, runner, *_ = make(mcp_url, groups={})
    RUNS.due.append(due())
    await runner.run_once()
    assert (RUNS.finished[0]["status"], RUNS.finished[0]["error"]) == ("failed", NOT_A_MEMBER)
    assert fake.rooms == {}


async def test_talk_down_leaves_the_answer_undelivered_in_a_fallback_thread(mcp_url):
    fake, runner, _, _, chat = make(mcp_url, [AIMessage("Digest.")], decider="direct")
    fake.broken = {"POST /ocs": 503}
    RUNS.due.append(due())
    await runner.run_once()
    assert RUNS.finished[0]["status"] == "undelivered"
    assert RUNS.finished[0]["summary"] == "Digest."
    assert await chat._checkpointer.aget_tuple({"configurable": {"thread_id": "kid:schedules"}})


async def test_a_failed_post_leaves_the_answer_undelivered_in_the_dm_thread(mcp_url):
    fake, runner, _, _, chat = make(mcp_url, [AIMessage("Digest.")], decider="direct")
    fake.broken = {"POST /ocs/v2.php/apps/spreed/api/v1/chat": 503}
    RUNS.due.append(due())
    await runner.run_once()
    assert RUNS.finished[0]["status"] == "undelivered"
    assert await chat._checkpointer.aget_tuple({"configurable": {"thread_id": "kid:talk-dm-kid"}})


async def test_an_agent_failure_finishes_failed_and_tells_the_owner(mcp_url):
    fake, runner, *_ = make(mcp_url, model=FailingChatModel(messages=iter([])))
    RUNS.due.append(due())
    await runner.run_once()
    assert RUNS.finished[0]["status"] == "failed" and "unavailable" in RUNS.finished[0]["error"]
    assert [p["message"] for p in fake.posted("dm-kid")] == [FAILED.format(title="F1 digest")]


async def test_nothing_due_claims_nothing(mcp_url):
    _, runner, *_ = make(mcp_url)
    assert await runner.run_once() == 0
    assert RUNS.finished == []


async def test_several_due_runs_are_claimed_one_at_a_time(mcp_url):
    _, runner, *_ = make(
        mcp_url, [AIMessage("One."), AIMessage("Two."), AIMessage("Three.")], decider="direct"
    )
    RUNS.due.extend(due(run_id=i) for i in (1, 2, 3))
    assert await runner.run_once() == 3
    assert [f["run_id"] for f in RUNS.finished] == [1, 2, 3]
    assert len(RUNS.claimed_by) == 4  # three runs, then the empty claim that ends the loop


async def test_a_run_that_had_already_expired_is_logged_not_raised(mcp_url, caplog):
    _, runner, *_ = make(mcp_url, [AIMessage("Late.")], decider="direct")
    RUNS.due.append(due())
    RUNS.finished.append({"run_id": 1, "status": "failed"})
    with caplog.at_level("WARNING"):
        assert await runner.run_once() == 1
    assert "lease" in caplog.text


async def test_a_scheduled_turn_uses_the_background_model_and_only_the_schedules_tools(mcp_url):
    _, runner, *_ = make(mcp_url)
    turns = []
    real = runner._chat.thread_turn
    runner._chat.thread_turn = lambda *a, **kw: turns.append(kw) or real(*a, **kw)
    RUNS.due.append(due(tools=["web_search"]))
    await runner.run_once()
    assert turns[0]["background"] is True and turns[0]["tools"] == frozenset({"web_search"})


async def test_a_run_handed_back_twice_is_executed_once_and_the_tick_ends(mcp_url, caplog):
    _, runner, model, *_ = make(mcp_url, [AIMessage("One."), AIMessage("Two.")], decider="direct")

    RUNS.due.append(due())
    original = runner._internal

    async def claim_again(name, args):
        result = await original(name, args)
        if name == "claim_due_runs" and result["runs"]:
            RUNS.due.append(due())
        return result

    runner._internal = claim_again
    with caplog.at_level("WARNING"):
        assert await runner.run_once() == 1
    assert len(RUNS.finished) == 1 and len(model.seen) == 1
    RUNS.due.clear()
    assert "again" in caplog.text


async def test_a_tick_handles_at_most_the_per_tick_cap(mcp_url):
    _, runner, *_ = make(mcp_url, [AIMessage("Ok.")] * (MAX_RUNS_PER_TICK + 1), decider="direct")
    RUNS.due.extend(due(run_id=i) for i in range(1, MAX_RUNS_PER_TICK + 6))
    assert await runner.run_once() == MAX_RUNS_PER_TICK
    assert len(RUNS.due) == 5


async def test_a_malformed_claim_is_failed_when_it_has_an_id_and_skipped_otherwise(mcp_url, caplog):
    _, runner, *_ = make(mcp_url, [AIMessage("Fine.")], decider="direct")
    RUNS.due.extend([{"run_id": 7, "owner": "kid"}, {"owner": "kid"}, due(run_id=9)])
    with caplog.at_level("WARNING"):
        assert await runner.run_once() == 2
    assert [(f["run_id"], f["status"], f["error"]) for f in RUNS.finished] == [
        (7, "failed", GATEWAY_ERROR),
        (9, "ok", None),
    ]
    assert "malformed" in caplog.text and "KeyError" in caplog.text


async def test_an_lldap_outage_finishes_the_run_failed_with_a_generic_error(mcp_url, caplog):
    _, runner, _, directory, _ = make(mcp_url)
    directory.down = True
    RUNS.due.append(due())
    with caplog.at_level("WARNING"):
        await runner.run_once()
    # finish_run's error is mirrored into the family's calendar: the details go to the log only.
    assert (RUNS.finished[0]["status"], RUNS.finished[0]["error"]) == ("failed", GATEWAY_ERROR)
    assert "lldap unreachable" in caplog.text


async def test_the_loop_keeps_going_after_a_tick_raises():
    runner = ScheduledRunner(toolbox=None, chat=None, directory=None, talk=None, seconds=7)
    ticks, sleeps = [], []

    async def run_once():
        ticks.append(1)
        if len(ticks) == 1:
            raise RuntimeError("uriel-tools down")

    async def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise asyncio.CancelledError

    runner.run_once = run_once
    with pytest.raises(asyncio.CancelledError):
        await runner.run(sleep)
    assert len(ticks) == 2 and sleeps == [7, 7]


async def test_id_less_claims_count_toward_the_per_tick_cap(mcp_url):
    _, runner, *_ = make(mcp_url)
    RUNS.due.extend({"owner": "kid"} for _ in range(MAX_RUNS_PER_TICK + 5))
    await runner.run_once()
    assert len(RUNS.due) == 5 and RUNS.finished == []


async def test_a_scheduled_run_with_tools_skips_the_route_decider(mcp_url):
    _, runner, model, *_ = make(mcp_url, [AIMessage("Nothing new.")], decider="direct")
    runner._chat._decider = None  # a run with tools must not ask it
    RUNS.due.append(due())
    await runner.run_once()
    assert model.bound == [["web_search"]]
    assert RUNS.finished[0]["status"] == "ok"


async def test_a_run_that_hangs_times_out_and_the_owners_next_dm_turn_still_works(mcp_url, monkeypatch):
    monkeypatch.setattr(runner_module, "RUN_TIMEOUT_S", 0.3)
    chat_model = RecordingChatModel(messages=iter([AIMessage("I'm here.")]))
    fake, runner, _, _, chat = make(mcp_url, model=HangingChatModel(messages=iter([])), chat_model=chat_model)
    RUNS.due.append(due())
    await asyncio.wait_for(runner.run_once(), 5)
    assert [(f["status"], f["error"]) for f in RUNS.finished] == [("failed", TIMED_OUT)]
    assert [p["message"] for p in fake.posted("dm-kid")] == [FAILED.format(title="F1 digest")]
    kid = Principal("kid", frozenset({"family"}), "human")
    turn = chat.thread_turn(kid, "kid:talk-dm-kid", "are you there?")
    result = await asyncio.wait_for(collect(chat.stream(turn)), 5)
    assert (result.error, result.answer) == (None, "I'm here.")


async def test_shutdown_lets_the_run_in_hand_finish_and_claims_no_more(mcp_url):
    model = GatedChatModel(
        messages=iter([AIMessage("Digest.")]), started=asyncio.Event(), gate=asyncio.Event()
    )
    fake, runner, *_ = make(mcp_url, model=model, decider="direct")
    RUNS.due.extend([due(run_id=1), due(run_id=2)])
    async with contextlib.AsyncExitStack() as stack:
        run_coworker(stack, Coworker(runner._talk, None, runner))
        await asyncio.wait_for(model.started.wait(), 5)
        asyncio.get_running_loop().call_later(0.1, model.gate.set)
    assert [(f["run_id"], f["status"]) for f in RUNS.finished] == [(1, "ok")]
    assert [r["run_id"] for r in RUNS.due] == [2]
    [post] = fake.posted("dm-kid")
    assert post["message"].startswith("Digest.")


async def test_an_empty_answer_is_delivered_as_a_short_line(mcp_url):
    fake, runner, *_ = make(mcp_url, [AIMessage("")])
    RUNS.due.append(due(tools=[]))
    await runner.run_once()
    [post] = fake.posted("dm-kid")
    assert post["message"] == EMPTY_ANSWER
    assert (RUNS.finished[0]["status"], RUNS.finished[0]["summary"]) == ("ok", EMPTY_ANSWER)
