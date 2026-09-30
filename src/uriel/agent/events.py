from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

EventKind = Literal["token", "tool_call", "tool_result", "final", "notice", "error"]
ANSWER_NODES = {"agent", "respond"}  # the router's structured output streams too; keep it out of chat


@dataclass(frozen=True)
class ChatEvent:
    kind: EventKind
    data: Any


async def stream_graph(graph, text: str, *, config: dict, context: Any) -> AsyncIterator[ChatEvent]:
    async for mode, chunk in graph.astream(
        {"messages": [HumanMessage(text)]}, config, context=context, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            msg, meta = chunk
            if isinstance(msg, AIMessageChunk) and msg.content and meta.get("langgraph_node") in ANSWER_NODES:
                yield ChatEvent("token", msg.content)
            continue
        for update in chunk.values():
            for m in (update or {}).get("messages", []):
                if isinstance(m, AIMessage) and m.tool_calls:
                    for tc in m.tool_calls:
                        yield ChatEvent("tool_call", {"name": tc["name"], "args": tc["args"], "id": tc["id"]})
                elif isinstance(m, ToolMessage):
                    yield ChatEvent("tool_result", {"name": m.name, "status": m.status, "content": m.content})
                elif isinstance(m, AIMessage):
                    yield ChatEvent("final", m.content)
