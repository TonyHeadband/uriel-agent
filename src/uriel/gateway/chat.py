import asyncio
import logging
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from uriel.agent.events import ChatEvent, stream_graph
from uriel.agent.graph import RunContext, build_graph
from uriel.agent.mcp_tools import unattended, uriel_meta, visible
from uriel.agent.memory import Memory, load_memory
from uriel.principal import Principal

log = logging.getLogger(__name__)
NOT_LOOKED_UP = "Nothing was looked up for this answer."
TOOLS_UNAVAILABLE = "Tools are unavailable right now; answering without them."
UNFINISHED = "I couldn't finish that in a reasonable number of steps. Try rephrasing?"
# Posted when the model's answer is empty, so a Talk message or a scheduled run never gets silence.
EMPTY_ANSWER = "I didn't come up with an answer to that. Could you ask another way?"


def tagged(event: ChatEvent, tools: dict[str, dict[str, Any]]) -> ChatEvent:
    """A tool event with its category and companion_action from uriel-tools, for the desktop companion."""
    if event.kind not in ("tool_call", "tool_result"):
        return event
    flags = tools.get(event.data["name"], {})
    return ChatEvent(
        event.kind,
        {**event.data, "category": flags.get("category"), "companion_action": flags.get("companion_action")},
    )


class InvalidMessage(ValueError):
    pass


class ConversationNotFound(LookupError):
    pass


MISSING_TOOLS = (
    "{names} can't run unattended for {user}: uriel-tools doesn't offer them to this person, or doesn't mark "
    "them unattended."
)


@dataclass(frozen=True)
class Turn:
    principal: Principal
    conversation_id: uuid.UUID | None
    text: str
    request_id: str
    # Talk rooms and scheduled runs name their thread; a web turn's comes from its conversation.
    thread: str | None = None
    personal: bool = True  # load the person's memory; off in a shared room
    shared: bool = False  # several people take turns on this thread (a Talk group room)
    note: str = ""
    history_turns: int | None = None
    # A scheduled run: only these tools, each marked unattended, on the background model.
    tools: frozenset[str] | None = None
    background: bool = False

    @property
    def thread_id(self) -> str:
        return self.thread or thread_id_for(self.principal, self.conversation_id)


@dataclass
class TurnResult:
    answer: str = ""
    notices: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def reply(self) -> str:
        return self.answer.strip() or EMPTY_ANSWER

    def message(self) -> str:
        """The answer with its notices in italics below it, as one chat message."""
        return "\n\n".join([self.reply, *(f"_{n}_" for n in self.notices)])


async def collect(events: AsyncIterator[ChatEvent]) -> TurnResult:
    """Drain a turn in the calling task: stream() holds the thread lock and the MCP session in `async with`
    blocks that anyio requires closing from the task that opened them."""
    result = TurnResult()
    async for event in events:
        match event.kind:
            case "final":
                result.answer = str(event.data)
            case "tool_call":
                result.tools.append(event.data["name"])
            case "notice":
                result.notices.append(event.data)
            case "error":
                result.error = result.error or event.data
    return result


def thread_id_for(principal: Principal, conversation_id: uuid.UUID | None) -> str:
    return f"{principal.user_id}:{conversation_id}"


class ChatService:
    def __init__(
        self,
        *,
        chat_model,
        tool_model=None,
        background_model=None,
        decider,
        toolbox,
        checkpointer,
        conversations,
        decision_log,
        recursion_limit: int,
        coverage: float | None = None,
        max_message_chars: int,
    ):
        self._chat_model = chat_model
        self._tool_model = tool_model or chat_model
        self._background = background_model or chat_model
        self._decider = decider
        self._toolbox = toolbox
        self._checkpointer = checkpointer
        self._conversations = conversations
        self._decision_log = decision_log
        self._recursion_limit = recursion_limit
        self._coverage = coverage
        self._max_chars = max_message_chars
        # Single gateway replica: an in-process lock per thread stops two tabs racing the checkpointer.
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _clean(self, text: str) -> str:
        text = text.strip()
        if not text:
            raise InvalidMessage("message is empty")
        if len(text) > self._max_chars:
            raise InvalidMessage(f"message is longer than {self._max_chars} characters")
        return text

    async def prepare(self, principal: Principal, conversation_id: uuid.UUID | None, text: str) -> Turn:
        text = self._clean(text)
        cid = conversation_id or uuid.uuid4()
        if not await self._conversations.claim(cid, principal.user_id, text[:80]):
            raise ConversationNotFound(str(cid))
        return Turn(principal, cid, text, uuid.uuid4().hex)

    def thread_turn(self, principal: Principal, thread_id: str, text: str, **options: Any) -> Turn:
        """A turn on a named thread (a Talk room, a scheduled run): no web conversation to claim."""
        return Turn(principal, None, self._clean(text), uuid.uuid4().hex, thread=thread_id, **options)

    async def stream(self, turn: Turn) -> AsyncIterator[ChatEvent]:
        thread_id = turn.thread_id
        async with self._locks[thread_id], AsyncExitStack() as stack:
            # The MCP session is entered and exited inside this generator: anyio requires the same task.
            tools = await self._toolbox.enter(stack, turn.principal, turn.request_id)
            if tools is None:
                yield ChatEvent("notice", TOOLS_UNAVAILABLE)
            tools = tools or []
            memory = await load_memory(tools) if turn.personal else Memory()
            tools = visible(tools)
            if turn.tools is not None:
                tools, missing = unattended(tools, turn.tools)
                if missing:
                    user = turn.principal.user_id
                    yield ChatEvent("error", MISSING_TOOLS.format(names=", ".join(missing), user=user))
                    return
            chat_model, tool_model = (
                (self._background, self._background)
                if turn.background
                else (self._chat_model, self._tool_model)
            )
            graph = build_graph(
                chat_model=chat_model,
                tool_model=tool_model,
                decider=self._decider,
                tools=tools,
                checkpointer=self._checkpointer,
                decision_log=self._decision_log,
                coverage=self._coverage,
            )
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": self._recursion_limit}
            context = RunContext(
                turn.principal,
                thread_id,
                turn.request_id,
                memory,
                turn.note,
                turn.history_turns,
                shared=turn.shared,
                always_tools=bool(turn.tools),
            )
            looked_up = False
            flags = {t.name: uriel_meta(t) for t in tools}
            try:
                async for event in stream_graph(graph, turn.text, config=config, context=context):
                    looked_up = looked_up or event.kind == "tool_call"
                    yield tagged(event, flags)
                # The family should be able to tell a grounded answer from the model's own knowledge.
                if tools and not looked_up:
                    yield ChatEvent("notice", NOT_LOOKED_UP)
            except GraphRecursionError:
                # A normal conversational outcome, not an outage: "final" keeps the API at 200 and
                # the web UI shows it like any other assistant reply, instead of a 503.
                yield ChatEvent("final", UNFINISHED)
            except Exception:
                log.exception("chat turn failed (request_id=%s)", turn.request_id)
                yield ChatEvent("error", "The assistant is unavailable right now. Please try again shortly.")

    async def history(self, principal: Principal, conversation_id: uuid.UUID) -> list[dict[str, Any]]:
        owned = {c["id"] for c in await self._conversations.list_for(principal.user_id, limit=1000)}
        if conversation_id not in owned:
            raise ConversationNotFound(str(conversation_id))
        tup = await self._checkpointer.aget_tuple(
            {"configurable": {"thread_id": thread_id_for(principal, conversation_id)}}
        )
        messages = tup.checkpoint["channel_values"].get("messages", []) if tup else []
        out = []
        for m in messages:
            if isinstance(m, HumanMessage):
                out.append({"role": "user", "content": str(m.content)})
            elif isinstance(m, AIMessage) and m.tool_calls:
                out.extend({"role": "tool", "content": tc["name"]} for tc in m.tool_calls)
            elif isinstance(m, AIMessage) and m.content:
                out.append({"role": "assistant", "content": str(m.content)})
        return out

    async def conversations(self, principal: Principal) -> list[dict[str, Any]]:
        return await self._conversations.list_for(principal.user_id)
