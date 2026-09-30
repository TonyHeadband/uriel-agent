import asyncio
import uuid

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from tests.fakes import FailingChatModel, RecordingChatModel, ScriptedChatModel
from uriel.agent.decider import Decision
from uriel.gateway.chat import NOT_LOOKED_UP, ChatService, ConversationNotFound, InvalidMessage
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


def service(model, *, tools=(), toolbox=None, decider=None, max_chars=100, tool_model=None):
    return ChatService(
        chat_model=model,
        tool_model=tool_model,
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
        return '{"user": "# About Tony\\n## Name\\n- Tony", "soul": ""}'

    model = RecordingChatModel(messages=iter([AIMessage("Hi Tony.")]))
    _, events = await run(service(model, tools=[memory_context]), KID, "hi")
    system = str(model.seen[0][0].content)
    assert "What you know about this person:\n# About Tony" in system
    assert "How this person wants you to talk" not in system
    # Loading memory is not a lookup, and with no other tools there is nothing to say about lookups.
    assert all(e.data != NOT_LOOKED_UP for e in events)
