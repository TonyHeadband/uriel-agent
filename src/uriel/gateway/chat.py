import asyncio
import logging
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from uriel.agent.events import ChatEvent, stream_graph
from uriel.agent.graph import RunContext, build_graph
from uriel.agent.memory import split_memory
from uriel.principal import Principal

log = logging.getLogger(__name__)
NOT_LOOKED_UP = "Nothing was looked up for this answer."


class InvalidMessage(ValueError):
    pass


class ConversationNotFound(LookupError):
    pass


@dataclass(frozen=True)
class Turn:
    principal: Principal
    conversation_id: uuid.UUID
    text: str
    request_id: str


def thread_id_for(principal: Principal, conversation_id: uuid.UUID) -> str:
    return f"{principal.user_id}:{conversation_id}"


class ChatService:
    def __init__(
        self,
        *,
        chat_model,
        tool_model=None,
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

    async def prepare(self, principal: Principal, conversation_id: uuid.UUID | None, text: str) -> Turn:
        text = text.strip()
        if not text:
            raise InvalidMessage("message is empty")
        if len(text) > self._max_chars:
            raise InvalidMessage(f"message is longer than {self._max_chars} characters")
        cid = conversation_id or uuid.uuid4()
        if not await self._conversations.claim(cid, principal.user_id, text[:80]):
            raise ConversationNotFound(str(cid))
        return Turn(principal, cid, text, uuid.uuid4().hex)

    async def stream(self, turn: Turn) -> AsyncIterator[ChatEvent]:
        thread_id = thread_id_for(turn.principal, turn.conversation_id)
        async with self._locks[thread_id], AsyncExitStack() as stack:
            # The MCP session is entered and exited inside this generator: anyio requires the same task.
            tools = await self._toolbox.enter(stack, turn.principal, turn.request_id)
            if tools is None:
                yield ChatEvent("notice", "Tools are unavailable right now; answering without them.")
            tools, memory = await split_memory(tools or [])
            graph = build_graph(
                chat_model=self._chat_model,
                tool_model=self._tool_model,
                decider=self._decider,
                tools=tools,
                checkpointer=self._checkpointer,
                decision_log=self._decision_log,
                coverage=self._coverage,
            )
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": self._recursion_limit}
            context = RunContext(turn.principal, thread_id, turn.request_id, memory)
            looked_up = False
            try:
                async for event in stream_graph(graph, turn.text, config=config, context=context):
                    looked_up = looked_up or event.kind == "tool_call"
                    yield event
                # The family should be able to tell a grounded answer from the model's own knowledge.
                if tools and not looked_up:
                    yield ChatEvent("notice", NOT_LOOKED_UP)
            except GraphRecursionError:
                # A normal conversational outcome, not an outage: "final" keeps the API at 200 and
                # the web UI shows it like any other assistant reply, instead of a 503.
                yield ChatEvent(
                    "final", "I couldn't finish that in a reasonable number of steps. Try rephrasing?"
                )
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
