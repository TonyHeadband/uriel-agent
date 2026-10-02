from contextlib import AsyncExitStack

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tests.fake_mcp import FakeRuns, build_fake_app, free_port, serve
from uriel.agent.mcp_tools import (
    McpToolbox,
    call_json,
    is_hidden,
    is_unattended,
    unattended,
    uriel_meta,
    visible,
)
from uriel.agent.memory import Memory, load_memory
from uriel.principal import INTERNAL_GROUP, Principal

DAD = Principal("dad", frozenset({"admins", "family"}), "human")
KID = Principal("kid", frozenset({"family"}), "human")
INTERNAL = Principal("uriel-gateway", frozenset({INTERNAL_GROUP}), "service")
RUNS = FakeRuns()


@pytest.fixture(scope="module")
def mcp_url():
    with serve(build_fake_app("test-key", RUNS)) as url:
        yield url


async def test_tools_are_filtered_per_principal(mcp_url):
    box = McpToolbox(mcp_url, "test-key")
    async with box.open(DAD, "r1") as tools:
        assert sorted(t.name for t in tools) == [
            "homelab_status",
            "memory_context",
            "remember",
            "search_documents",
            "web_search",
        ]
    async with box.open(KID, "r2") as tools:
        assert sorted(t.name for t in tools) == [
            "memory_context",
            "remember",
            "search_documents",
            "web_search",
        ]


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
        memory = await load_memory(tools)
        offered = visible(tools)
    assert sorted(t.name for t in offered) == ["remember", "search_documents", "web_search"]
    assert memory == Memory(
        "# About kid\n## Name\n- Tony", "## Length\n- Short", "Tue 29 Sep 2026, 22:41 (America/Toronto)"
    )


async def test_uriel_tools_marks_hidden_and_unattended_tools(mcp_url):
    box = McpToolbox(mcp_url, "test-key")
    async with box.open(INTERNAL, "r1") as tools:
        assert sorted(t.name for t in tools) == ["claim_due_runs", "finish_run"]
        assert all(is_hidden(t) for t in tools) and visible(tools) == []
    async with box.open(DAD, "r2") as tools:
        assert sorted(t.name for t in tools if is_unattended(t)) == ["search_documents", "web_search"]
        # No flags at all (homelab_status, remember) means neither hidden nor unattended.
        assert sorted(t.name for t in visible(tools)) == [
            "homelab_status",
            "remember",
            "search_documents",
            "web_search",
        ]


async def test_unattended_keeps_only_requested_unattended_tools(mcp_url):
    async with McpToolbox(mcp_url, "test-key").open(KID, "r1") as tools:
        offered, missing = unattended(tools, {"web_search", "remember", "homelab_status"})
    assert [t.name for t in offered] == ["web_search"]
    assert missing == ["homelab_status", "remember"]


async def test_call_json_reads_a_tools_json_result(mcp_url):
    run = {"run_id": 1, "owner": "kid", "title": "t"}
    RUNS.due.append(run)
    async with McpToolbox(mcp_url, "test-key").open(INTERNAL, "r1") as tools:
        claim = next(t for t in tools if t.name == "claim_due_runs")
        assert await call_json(claim, {"limit": 5}) == {"runs": [run]}
        assert await call_json(claim, {"limit": 5}) == {"runs": []}
    assert RUNS.claimed_by[0]["groups"] == [INTERNAL_GROUP]


async def test_finish_run_reports_whether_it_changed_anything(mcp_url):
    async with McpToolbox(mcp_url, "test-key").open(INTERNAL, "r1") as tools:
        finish = next(t for t in tools if t.name == "finish_run")
        first = await call_json(finish, {"run_id": 7, "status": "ok"})
        again = await call_json(finish, {"run_id": 7, "status": "failed"})
    assert first == {"run_id": 7, "status": "ok", "finished": True}
    assert again == {"run_id": 7, "status": "ok", "finished": False}
    assert [r["run_id"] for r in RUNS.finished] == [7]


def flagged(meta):
    @tool
    def some_tool() -> str:
        """Some tool."""
        return ""

    some_tool.metadata = meta
    return some_tool


@pytest.mark.parametrize(
    "meta", [None, {"_meta": "oops"}, {"_meta": ["hidden"]}, {"_meta": {"uriel": "hidden"}}, {"_meta": None}]
)
def test_malformed_meta_is_neither_hidden_nor_unattended(meta):
    t = flagged(meta)
    assert uriel_meta(t) == {} and not is_hidden(t) and not is_unattended(t)


def test_memory_context_is_never_offered_even_without_the_hidden_flag():
    @tool
    def memory_context() -> str:
        """Memory, as uriel-tools before 0.8.0 lists it: no flags."""
        return "{}"

    assert visible([memory_context]) == []
