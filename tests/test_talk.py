import asyncio
import contextlib

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from tests.fake_talk import URIEL_MENTION, FakeTalk
from tests.fakes import (
    HIDDEN,
    FailingChatModel,
    GatedChatModel,
    MemoryCursors,
    RecordingChatModel,
    StaticGroups,
)
from tests.test_chat import Decider, Toolbox, service
from uriel.gateway.app import Coworker, run_coworker
from uriel.gateway.chat import EMPTY_ANSWER
from uriel.gateway.talk import EMPTY_MENTION, TURN_FAILED, UNKNOWN_SENDER, TalkChannel, next_delay
from uriel.gateway.talk_api import EYES, Room, TalkClient, TalkUnauthorized, TalkUnavailable
from uriel.principal import INTERNAL_GROUP

GROUPS = {"dad": {"admins", "family", INTERNAL_GROUP}, "kid": {"family"}, "neighbour": {"friends"}}


class Clock:
    def __init__(self, now: float = 1_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def make(
    replies=(),
    *,
    fake=None,
    model=None,
    toolbox=None,
    tools=(),
    directory=None,
    cursors=None,
    clock=None,
    max_chars=200,
    decider=None,
):
    fake = fake or FakeTalk()
    client = TalkClient(
        "http://talk.test", "uriel", "talk-password", transport=httpx.ASGITransport(app=fake.app())
    )
    model = model or RecordingChatModel(messages=iter(replies))
    channel = TalkChannel(
        client=client,
        chat=service(model, tools=tools, toolbox=toolbox, max_chars=max_chars, decider=decider),
        directory=directory or StaticGroups(GROUPS),
        cursors=cursors if cursors is not None else MemoryCursors(),
        poll_seconds=3,
        history_turns=20,
        clock=clock or Clock(),
    )
    return fake, channel, model


async def tick(channel):
    await channel.poll_once()
    await channel.drain()


def said(fake, token):
    return [p["message"] for p in fake.posted(token)]


async def test_the_first_sight_of_a_room_does_not_answer_history():
    cursors = MemoryCursors()
    fake, channel, model = make([AIMessage("unused")], cursors=cursors)
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    old = fake.say("dm-dad", "dad", "are you there?")
    await tick(channel)
    assert fake.posted() == [] and model.seen == []
    assert cursors.rows == {"dm-dad": old}


async def test_a_dm_is_answered_in_the_persons_own_thread():
    fake, channel, _ = make([AIMessage("Hi Dad.")])
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(channel)
    fake.say("dm-dad", "dad", "hello")
    await tick(channel)
    assert said(fake, "dm-dad") == ["Hi Dad."]
    assert await channel._chat._checkpointer.aget_tuple({"configurable": {"thread_id": "dad:talk-dm-dad"}})


async def test_the_cursor_survives_a_restart_without_replaying_or_skipping():
    cursors = MemoryCursors()
    fake, first, _ = make([AIMessage("One.")], cursors=cursors)
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(first)
    fake.say("dm-dad", "dad", "one")
    await tick(first)
    fake.say("dm-dad", "dad", "two")  # sent while the gateway restarts
    _, second, _ = make([AIMessage("Two.")], fake=fake, cursors=cursors)
    await tick(second)
    assert said(fake, "dm-dad") == ["One.", "Two."]


async def test_a_dm_opened_after_startup_is_answered_from_its_first_message():
    fake, channel, _ = make([AIMessage("Hello!")])
    await tick(channel)
    fake.add_room("dm-kid", Room.ONE_TO_ONE, name="kid")
    fake.say("dm-kid", "kid", "sent before the gateway started", timestamp=500)
    fake.say("dm-kid", "kid", "hi")
    await tick(channel)
    assert said(fake, "dm-kid") == ["Hello!"]


async def test_messages_that_arrive_together_are_answered_in_order():
    fake, channel, model = make([AIMessage("A."), AIMessage("B.")])
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(channel)
    fake.say("dm-dad", "dad", "first")
    fake.say("dm-dad", "dad", "second")
    await tick(channel)
    assert said(fake, "dm-dad") == ["A.", "B."]
    assert model.seen[1][-1].content == "second"


async def test_own_system_guest_bridged_and_deleted_messages_are_ignored():
    cursors = MemoryCursors()
    fake, channel, model = make([], cursors=cursors)
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(channel)
    fake.say("dm-dad", "uriel", "my own earlier answer")
    fake.say("dm-dad", "dad", "You created the conversation", system="conversation_created")
    fake.say("dm-dad", "guest-1", "hi", actor_type="guests")
    fake.say("dm-dad", "matrix-bob", "hi", actor_type="bridged")
    last = fake.say("dm-dad", "dad", "Message deleted by you", message_type="comment_deleted")
    await tick(channel)
    assert said(fake, "dm-dad") == ["my own earlier answer"] and model.seen == []
    assert cursors.rows["dm-dad"] == last


async def test_an_unknown_sender_gets_one_refusal_a_day_and_no_turn():
    clock = Clock()
    fake, channel, model = make([], clock=clock)
    fake.add_room("dm-x", Room.ONE_TO_ONE, name="authelia-19e1429e")
    await tick(channel)
    fake.say("dm-x", "authelia-19e1429e", "hi")
    fake.say("dm-x", "authelia-19e1429e", "hello?")
    await tick(channel)
    assert said(fake, "dm-x") == [UNKNOWN_SENDER] and model.seen == []
    clock.now += 86_400
    fake.say("dm-x", "authelia-19e1429e", "anyone?")
    await tick(channel)
    assert said(fake, "dm-x") == [UNKNOWN_SENDER, UNKNOWN_SENDER]


async def test_someone_in_lldap_but_outside_the_member_groups_is_an_unknown_sender():
    fake, channel, model = make([AIMessage("unused")])
    fake.add_room("dm-n", Room.ONE_TO_ONE, name="neighbour")
    await family_room(fake, channel)
    await tick(channel)
    fake.say("dm-n", "neighbour", "hi")
    asked = fake.say("family", "neighbour", "{mention-user1} what's the wifi password?", params=URIEL_MENTION)
    await tick(channel)
    assert said(fake, "dm-n") == [UNKNOWN_SENDER]
    assert [(p["message"], p["reply_to"]) for p in fake.posted("family")] == [(UNKNOWN_SENDER, asked)]
    assert model.seen == []


async def test_no_person_is_given_the_internal_group():
    class Recording(Toolbox):
        def __init__(self):
            super().__init__([])
            self.principals = []

        async def enter(self, stack, principal, request_id):
            self.principals.append(principal)
            return self.tools

    toolbox = Recording()
    fake, channel, _ = make([AIMessage("Hi.")], toolbox=toolbox)
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(channel)
    fake.say("dm-dad", "dad", "hi")
    await tick(channel)
    assert toolbox.principals[0].groups == frozenset({"admins", "family"})


async def test_a_message_over_the_limit_is_answered_with_why():
    fake, channel, model = make([], max_chars=20)
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(channel)
    fake.say("dm-dad", "dad", "x" * 50)
    await tick(channel)
    [answer] = said(fake, "dm-dad")
    assert "longer than 20 characters" in answer and model.seen == []


def test_backoff_doubles_to_a_minute_and_waits_five_minutes_after_a_401():
    assert next_delay(3, 3, None) == 3
    assert next_delay(3, 3, TalkUnavailable("503")) == 6
    assert next_delay(3, 48, TalkUnavailable("503")) == 60
    assert next_delay(3, 3, TalkUnauthorized("401")) == 300


async def test_the_poll_loop_backs_off_and_recovers():
    fake, channel, _ = make([])
    fake.fail = [503, 503, 401]
    slept = []

    async def sleep(seconds):
        slept.append(seconds)
        if len(slept) == 4:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await channel.run(sleep=sleep)
    assert slept == [6, 12, 300, 3]


@tool
def memory_context() -> str:
    """Memory."""
    return '{"user": "# About Dad\\n- Likes jazz", "soul": ""}'


memory_context.metadata = HIDDEN  # as uriel-tools 0.8.0 lists it


async def family_room(fake, channel):
    fake.add_room("family", Room.GROUP, display_name="Family")
    await tick(channel)


async def test_a_group_room_answers_only_mentions_as_a_reply_without_personal_memory():
    fake, channel, model = make([AIMessage("Pasta.")], tools=[memory_context])
    await family_room(fake, channel)
    fake.say("family", "kid", "what's for dinner?")
    asked = fake.say("family", "dad", "{mention-user1} what's for dinner?", params=URIEL_MENTION)
    await tick(channel)
    assert [(p["message"], p["reply_to"]) for p in fake.posted("family")] == [("Pasta.", asked)]
    system = str(model.seen[0][0].content)
    assert 'shared Talk room "Family"' in system and "What you know" not in system
    assert model.seen[0][-1].content == "what's for dinner?"
    assert await channel._chat._checkpointer.aget_tuple({"configurable": {"thread_id": "room:talk-family"}})


async def test_a_dm_brings_the_persons_memory_and_no_audience_line():
    fake, channel, model = make([AIMessage("Hi.")], tools=[memory_context])
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(channel)
    fake.say("dm-dad", "dad", "hi")
    await tick(channel)
    system = str(model.seen[0][0].content)
    assert "What you know about this person" in system and "shared Talk room" not in system


async def test_a_bare_mention_is_asked_what_it_can_do():
    fake, channel, model = make([])
    await family_room(fake, channel)
    asked = fake.say("family", "dad", "{mention-user1}", params=URIEL_MENTION)
    await tick(channel)
    assert [(p["message"], p["reply_to"]) for p in fake.posted("family")] == [(EMPTY_MENTION, asked)]
    assert model.seen == []


async def test_mentioning_everyone_is_not_mentioning_uriel():
    fake, channel, model = make([])
    await family_room(fake, channel)
    everyone = {"mention-call1": {"type": "call", "id": "family", "name": "Family"}}
    fake.say("family", "dad", "{mention-call1} dinner is ready", params=everyone)
    await tick(channel)
    assert fake.posted("family") == [] and model.seen == []


async def test_eyes_mark_pickup_and_are_removed_when_the_turn_fails():
    fake, channel, _ = make([AIMessage("Hi.")])
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(channel)
    ok = fake.say("dm-dad", "dad", "hi")
    await tick(channel)
    assert fake.reactions[("dm-dad", ok)] == {EYES}

    broken, failing, _ = make(model=FailingChatModel(messages=iter([])))
    broken.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(failing)
    bad = broken.say("dm-dad", "dad", "hi")
    await tick(failing)
    assert broken.reactions[("dm-dad", bad)] == set()
    assert said(broken, "dm-dad") == [TURN_FAILED]


async def test_a_failed_group_room_turn_removes_the_eyes_and_replies_with_the_apology():
    fake, channel, _ = make(model=FailingChatModel(messages=iter([])))
    await family_room(fake, channel)
    asked = fake.say("family", "dad", "{mention-user1} what's for dinner?", params=URIEL_MENTION)
    await tick(channel)
    assert fake.reactions[("family", asked)] == set()
    assert [(p["message"], p["reply_to"]) for p in fake.posted("family")] == [(TURN_FAILED, asked)]


async def test_an_empty_answer_posts_a_short_line_instead_of_nothing():
    fake, channel, _ = make([AIMessage("")])
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(channel)
    fake.say("dm-dad", "dad", "hmm")
    await tick(channel)
    assert said(fake, "dm-dad") == [EMPTY_ANSWER]


async def test_a_long_answer_is_split_and_only_its_first_part_replies():
    long = "\n\n".join(["a" * 3000, "b" * 3000, "c" * 900])
    fake, channel, _ = make([AIMessage(long)])
    await family_room(fake, channel)
    asked = fake.say("family", "dad", "{mention-user1} tell me everything", params=URIEL_MENTION)
    await tick(channel)
    assert [(len(p["message"]), p["reply_to"]) for p in fake.posted("family")] == [
        (3000, asked),
        (3902, None),
    ]


async def test_a_shared_file_reads_as_its_name():
    fake, channel, model = make([AIMessage("Got it.")])
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(channel)
    fake.say("dm-dad", "dad", "{file}", params={"file": {"type": "file", "id": "42", "name": "report.pdf"}})
    await tick(channel)
    assert model.seen[0][-1].content == "report.pdf"


async def test_an_lldap_outage_leaves_the_message_for_the_next_poll(caplog):
    directory, cursors = StaticGroups(GROUPS), MemoryCursors()
    fake, channel, _ = make([AIMessage("Hi.")], directory=directory, cursors=cursors)
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(channel)
    directory.down = True
    sent = fake.say("dm-dad", "dad", "hi")
    await tick(channel)
    assert fake.posted() == [] and cursors.rows["dm-dad"] < sent
    assert [r for r in caplog.records if r.exc_info] == []
    directory.down = False
    await tick(channel)
    assert said(fake, "dm-dad") == ["Hi."]


async def test_stop_with_a_room_task_that_never_started_lets_drain_return():
    fake, channel, _ = make([AIMessage("Hi.")])
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(channel)
    fake.say("dm-dad", "dad", "hi")
    await channel.poll_once()  # the room task is created but hasn't run yet
    channel.stop()
    await asyncio.wait_for(channel.drain(), timeout=1)


async def test_a_shared_room_never_replays_one_persons_tool_results_to_the_next():
    @tool
    def search_documents(query: str) -> str:
        """Search the family's documents."""
        return "dad-only-bank-statement.pdf: balance 12,345"

    search = {"name": "search_documents", "args": {"query": "bank"}, "id": "c1"}
    replies = [AIMessage("", tool_calls=[search]), AIMessage("I found it."), AIMessage("Hi kid.")]
    fake, channel, model = make(replies, tools=[search_documents], decider=Decider("tools"))
    await family_room(fake, channel)
    fake.say("family", "dad", "{mention-user1} find my bank statement", params=URIEL_MENTION)
    await tick(channel)
    fake.say("family", "kid", "{mention-user1} what did you find?", params=URIEL_MENTION)
    await tick(channel)
    kid_turn = model.seen[-1]
    assert "12,345" not in str([m.content for m in kid_turn])
    assert not any(isinstance(m, ToolMessage) or getattr(m, "tool_calls", None) for m in kid_turn)
    # What the room already read stays: the earlier question and the answer posted to it.
    assert [m.content for m in kid_turn[1:]] == [
        "find my bank statement",
        "I found it.",
        "what did you find?",
    ]


async def test_shutdown_lets_a_turn_in_flight_finish_and_post_its_answer():
    model = GatedChatModel(messages=iter([AIMessage("Done.")]), started=asyncio.Event(), gate=asyncio.Event())
    fake, channel, _ = make(model=model)
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    await tick(channel)
    fake.say("dm-dad", "dad", "hi")
    fake.say("dm-dad", "dad", "and another thing")
    async with contextlib.AsyncExitStack() as stack:
        run_coworker(stack, Coworker(channel._client, channel, None))
        await asyncio.wait_for(model.started.wait(), 5)
        asyncio.get_running_loop().call_later(0.1, model.gate.set)
    # The turn in flight is answered; the message after it waits for the next start.
    assert said(fake, "dm-dad") == ["Done."]
