import socket
import threading
import time
from contextlib import AsyncExitStack

import pytest
import uvicorn
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tests.fake_mcp import build_fake_app
from uriel.agent.mcp_tools import McpToolbox
from uriel.agent.memory import Memory, split_memory
from uriel.principal import Principal

DAD = Principal("dad", frozenset({"admins", "family"}), "human")
KID = Principal("kid", frozenset({"family"}), "human")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mcp_url():
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(build_fake_app("test-key"), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("MCP test server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(5)


async def test_tools_are_filtered_per_principal(mcp_url):
    box = McpToolbox(mcp_url, "test-key")
    async with box.open(DAD, "r1") as tools:
        assert sorted(t.name for t in tools) == ["homelab_status", "memory_context", "search_documents"]
    async with box.open(KID, "r2") as tools:
        assert sorted(t.name for t in tools) == ["memory_context", "search_documents"]


async def test_tool_call_carries_identity(mcp_url):
    async with McpToolbox(mcp_url, "test-key").open(DAD, "r1") as tools:
        # ToolNode.ainvoke needs a graph runtime to resolve config outside a compiled graph
        # (probe 2026-09-27, langgraph 1.2.12), so wrap it in a one-node graph.
        graph = StateGraph(MessagesState)
        graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
        graph.set_entry_point("tools")
        graph.set_finish_point("tools")
        compiled = graph.compile()
        call = AIMessage("", tool_calls=[{"name": "homelab_status", "args": {}, "id": "c1"}])
        out = await compiled.ainvoke({"messages": [call]})
    msg = out["messages"][-1]
    assert msg.status == "success"
    assert '"received_meta":{"user":"dad","groups":["admins","family"],"request_id":"r1"}' in str(msg.content)


async def test_enter_returns_none_when_mcp_is_down():
    async with AsyncExitStack() as stack:
        tools = await McpToolbox(f"http://127.0.0.1:{free_port()}/mcp", "k").enter(stack, DAD, "r1")
    assert tools is None


async def test_wrong_api_key_behaves_like_unreachable(mcp_url):
    async with AsyncExitStack() as stack:
        assert await McpToolbox(mcp_url, "wrong").enter(stack, DAD, "r1") is None


async def test_memory_is_loaded_with_identity_and_hidden_from_the_model(mcp_url):
    async with McpToolbox(mcp_url, "test-key").open(KID, "r1") as tools:
        rest, memory = await split_memory(tools)
    assert [t.name for t in rest] == ["search_documents"]
    assert memory == Memory("# About kid\n## Name\n- Tony", "## Length\n- Short")
