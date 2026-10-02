import asyncio
import re
import uuid

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.checkpoint.memory import InMemorySaver

from tests.fakes import HIDDEN, FailingChatModel, RecordingChatModel, ScriptedChatModel
from uriel.agent.decider import Decision
from uriel.gateway.chat import (
    EMPTY_ANSWER,
    MISSING_TOOLS,
    NOT_LOOKED_UP,
    ChatService,
    ConversationNotFound,
    InvalidMessage,
    TurnResult,
    collect,
)
from uriel.principal import Principal

DAD = Principal("dad", frozenset({"admins"}), "human")
KID = Principal("kid", frozenset({"family"}), "human")


@tool
def homelab_status() -> str:
    """Homelab."""
    return "all good"


class Decider:
    def __init__(self, choice="direct"):
        self.choice = choice

    async def choose(self, point, context, options):
        return Decision(point, self.choice, 0.9, "llm", "fake", 1)


class MemoryConversations:
    def __init__(self):
        self.owners = {}

    async def claim(self, cid, user_id, title):
        return self.owners.setdefault(cid, user_id) == user_id

    async def list_for(self, user_id, limit=50):
        return [{"id": c, "title": "t"} for c, u in self.owners.items() if u == user_id]


class Toolbox:
    def __init__(self, tools):
        self.tools = tools

    async def enter(self, stack, principal, request_id):
        return self.tools


def service(
    model, *, tools=(), toolbox=None, decider=None, max_chars=100, tool_model=None, background_model=None
):
    return ChatService(
        chat_model=model,
        tool_model=tool_model,
        background_model=background_model,
        decider=decider or Decider(),
        toolbox=toolbox or Toolbox(list(tools)),
        checkpointer=InMemorySaver(),
        conversations=MemoryConversations(),
        decision_log=None,
        recursion_limit=10,
        max_message_chars=max_chars,
    )


async def run(svc, principal, text, cid=None):
    turn = await svc.prepare(principal, cid, text)
    return turn, [e async for e in svc.stream(turn)]


@pytest.mark.parametrize("text", ["", "   ", "x" * 101])
async def test_invalid_messages_are_rejected_before_streaming(text):
    with pytest.raises(InvalidMessage):
        await service(ScriptedChatModel(messages=iter([]))).prepare(DAD, None, text)


async def test_other_users_conversation_is_not_found():
    svc = service(ScriptedChatModel(messages=iter([AIMessage("hi")])))
    turn, _ = await run(svc, DAD, "hello")
    with pytest.raises(ConversationNotFound):
        await svc.prepare(KID, turn.conversation_id, "let me in")
    with pytest.raises(ConversationNotFound):
        await svc.history(KID, turn.conversation_id)


async def test_llm_failure_becomes_error_event():
    _, events = await run(service(FailingChatModel(messages=iter([]))), DAD, "hi")
    assert events[-1].kind == "error"
    assert "unavailable" in events[-1].data


async def test_mcp_down_answers_without_tools_and_says_so():
    class Down:
        async def enter(self, stack, principal, request_id):
            return None

    svc = service(ScriptedChatModel(messages=iter([AIMessage("I can still chat.")])), toolbox=Down())
    _, events = await run(svc, DAD, "homelab?")
    assert events[0].kind == "notice"
    assert (events[-1].kind, events[-1].data) == ("final", "I can still chat.")


async def test_recursion_limit_becomes_a_friendly_final_reply_not_an_error():
    # The spec wants a clean "couldn't finish" reply, not "unavailable": the recursion abort is
    # a normal conversational outcome, so it must not surface as a 503 through the API.
    call = {"name": "homelab_status", "args": {}}
    loop = [AIMessage("", tool_calls=[call | {"id": f"c{i}"}]) for i in range(30)]
    svc = service(ScriptedChatModel(messages=iter(loop)), tools=[homelab_status], decider=Decider("tools"))
    svc._recursion_limit = 4
    _, events = await run(svc, DAD, "loop")
    assert events[-1].kind == "final"
    assert "couldn't finish" in events[-1].data


async def test_concurrent_turns_on_one_conversation_are_serialized():
    svc = service(ScriptedChatModel(messages=iter([AIMessage("A."), AIMessage("B.")])))
    cid = uuid.uuid4()
    t1 = await svc.prepare(DAD, cid, "one")
    t2 = await svc.prepare(DAD, cid, "two")

    async def drain(t):
        return [e async for e in svc.stream(t)]

    await asyncio.gather(drain(t1), drain(t2))
    roles = [(m["role"], m["content"]) for m in await svc.history(DAD, cid)]
    assert len(roles) == 4
    assert roles[0][0] == "user" and roles[1][0] == "assistant"
    assert roles[2][0] == "user" and roles[3][0] == "assistant"


async def test_answer_without_a_lookup_says_so_when_tools_were_available():
    svc = service(ScriptedChatModel(messages=iter([AIMessage("Hello!")])), tools=[homelab_status])
    _, events = await run(svc, DAD, "hi")
    assert (events[-1].kind, events[-1].data) == ("notice", NOT_LOOKED_UP)


async def test_answer_with_a_lookup_is_not_flagged_and_history_keeps_the_tool():
    call = {"name": "homelab_status", "args": {}, "id": "c1"}
    script = iter([AIMessage("", tool_calls=[call]), AIMessage("All good.")])
    svc = service(ScriptedChatModel(messages=script), tools=[homelab_status], decider=Decider("tools"))
    turn, events = await run(svc, DAD, "homelab?")
    assert NOT_LOOKED_UP not in [e.data for e in events if e.kind == "notice"]
    history = await svc.history(DAD, turn.conversation_id)
    assert [m["role"] for m in history] == ["user", "tool", "assistant"]
    assert history[1]["content"] == "homelab_status"


async def test_tool_turns_use_the_tool_model_and_direct_turns_the_chat_model():
    call = {"name": "homelab_status", "args": {}, "id": "c1"}
    tool_model = ScriptedChatModel(messages=iter([AIMessage("", tool_calls=[call]), AIMessage("All good.")]))
    chat_model = ScriptedChatModel(messages=iter([AIMessage("Hello!")]))
    svc = service(chat_model, tools=[homelab_status], tool_model=tool_model, decider=Decider("tools"))
    _, events = await run(svc, DAD, "homelab?")
    assert [e.data["name"] for e in events if e.kind == "tool_call"] == ["homelab_status"]
    svc._decider = Decider("direct")
    _, events = await run(svc, DAD, "hi")
    assert ("final", "Hello!") in [(e.kind, e.data) for e in events]


async def test_memory_reaches_the_prompt_and_is_not_a_model_tool():
    @tool
    def memory_context() -> str:
        """Memory."""
        return (
            '{"user": "# About Tony\\n## Name\\n- Tony", "soul": "", '
            '"tz": "America/Toronto", "now_local": "Tue 29 Sep 2026, 22:41"}'
        )

    memory_context.metadata = HIDDEN

    model = RecordingChatModel(messages=iter([AIMessage("Hi Tony.")]))
    _, events = await run(service(model, tools=[memory_context]), KID, "hi")
    system = str(model.seen[0][0].content)
    assert "What you know about this person:\n# About Tony" in system
    # Probe 2026-09-29: without today's date, qwen3:8b scheduled "remind me" for 2023.
    assert "It is now Tue 29 Sep 2026, 22:41 (America/Toronto)." in system
    assert "How this person wants you to talk" not in system
    # Loading memory is not a lookup, and with no other tools there is nothing to say about lookups.
    assert all(e.data != NOT_LOOKED_UP for e in events)


async def test_a_tool_marked_hidden_is_never_bound_even_by_name_unknown_to_the_gateway():
    @tool
    def claim_due_runs() -> str:
        """Runner only."""
        return "[]"

    claim_due_runs.metadata = HIDDEN
    model = RecordingChatModel(messages=iter([AIMessage("Hi.")]))
    await run(service(model, tools=[homelab_status, claim_due_runs], decider=Decider("tools")), DAD, "hi")
    assert model.bound == [["homelab_status"]]


def meta_tool(name, **flags):
    """A tool as McpToolbox returns it, with uriel-tools' `_meta.uriel` flags (hidden, unattended)."""

    async def run() -> str:
        return f"{name} ran"

    metadata = {"_meta": {"uriel": flags}}
    return StructuredTool.from_function(coroutine=run, name=name, description=name, metadata=metadata)


async def test_a_thread_turn_keeps_its_own_thread_and_can_skip_memory():
    @tool
    def memory_context() -> str:
        """Memory."""
        raise AssertionError("a shared room must not load personal memory")

    memory_context.metadata = HIDDEN
    model = RecordingChatModel(messages=iter([AIMessage("Hi all.")]))
    svc = service(model, tools=[memory_context])
    turn = svc.thread_turn(DAD, "room:talk-family", " hi ", personal=False, note="Everyone can read this.")
    result = await collect(svc.stream(turn))
    assert (turn.text, turn.thread_id, result.answer) == ("hi", "room:talk-family", "Hi all.")
    assert "Everyone can read this." in str(model.seen[0][0].content)
    assert re.search(
        r"It is now \w{3} \d{1,2} \w{3} \d{4}, \d{2}:\d{2} \(UTC\)\.", str(model.seen[0][0].content)
    )
    assert await svc._checkpointer.aget_tuple({"configurable": {"thread_id": "room:talk-family"}})


@pytest.mark.parametrize("text", ["", "   ", "x" * 101])
def test_thread_turns_are_validated_like_web_turns(text):
    with pytest.raises(InvalidMessage):
        service(ScriptedChatModel(messages=iter([]))).thread_turn(DAD, "dad:talk-x", text)


async def test_a_scheduled_turn_offers_only_its_unattended_tools_on_the_background_model():
    background = RecordingChatModel(messages=iter([AIMessage("Digest.")]))
    tools = [
        meta_tool("web_search", unattended=True),
        meta_tool("search_documents", unattended=True),
        meta_tool("remember"),
    ]
    svc = service(
        ScriptedChatModel(messages=iter([])),
        tools=tools,
        decider=Decider("tools"),
        background_model=background,
    )
    turn = svc.thread_turn(
        KID, "kid:talk-dm", "Scheduled task", tools=frozenset({"web_search"}), background=True
    )
    result = await collect(svc.stream(turn))
    assert result.answer == "Digest."
    assert background.bound == [["web_search"]]


async def test_a_scheduled_turn_fails_when_a_tool_cant_run_unattended():
    background = RecordingChatModel(messages=iter([]))
    tools = [meta_tool("web_search", unattended=True), meta_tool("remember")]
    svc = service(ScriptedChatModel(messages=iter([])), tools=tools, background_model=background)
    wanted = frozenset({"web_search", "remember", "gone"})
    result = await collect(
        svc.stream(svc.thread_turn(KID, "kid:t", "Scheduled task", tools=wanted, background=True))
    )
    assert result.error == MISSING_TOOLS.format(names="gone, remember", user="kid")
    assert background.seen == []


def test_a_turn_result_reads_as_one_chat_message():
    assert TurnResult("Answer.", [NOT_LOOKED_UP]).message() == f"Answer.\n\n_{NOT_LOOKED_UP}_"
    assert (
        TurnResult(" ", ["Tools are unavailable."]).message() == f"{EMPTY_ANSWER}\n\n_Tools are unavailable._"
    )


async def test_an_unflagged_memory_context_is_loaded_but_never_bound():
    @tool
    def memory_context() -> str:
        """Memory, as uriel-tools before 0.8.0 lists it: no hidden flag."""
        return '{"user": "# About Tony", "soul": ""}'

    model = RecordingChatModel(messages=iter([AIMessage("Hi.")]))
    await run(service(model, tools=[homelab_status, memory_context], decider=Decider("tools")), KID, "hi")
    assert model.bound == [["homelab_status"]]
    assert "# About Tony" in str(model.seen[0][0].content)


async def test_a_turn_cancelled_mid_tool_call_leaves_the_next_turn_a_clean_history():
    ran = []

    @tool
    async def slow_search(query: str) -> str:
        """Search, slowly."""
        ran.append(query)
        await asyncio.Event().wait()
        return ""

    call = {"name": "slow_search", "args": {"query": "x"}, "id": "c1"}
    model = RecordingChatModel(messages=iter([AIMessage("", tool_calls=[call]), AIMessage("Back.")]))
    svc = service(model, tools=[slow_search], decider=Decider("tools"))
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.3):
            await collect(svc.stream(svc.thread_turn(KID, "kid:talk-dm", "search x")))
    result = await asyncio.wait_for(collect(svc.stream(svc.thread_turn(KID, "kid:talk-dm", "hello?"))), 5)
    assert (result.error, result.answer) == (None, "Back.")
    assert ran == ["x"]  # the abandoned call isn't run again
    seen = model.seen[-1]
    assert not any(getattr(m, "tool_calls", None) or isinstance(m, ToolMessage) for m in seen)
    assert [m.content for m in seen[1:]] == ["search x", "hello?"]
