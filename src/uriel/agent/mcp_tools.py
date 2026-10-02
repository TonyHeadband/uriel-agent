import json
import logging
from collections.abc import AsyncIterator, Collection, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp.types import PaginatedRequestParams

from uriel.principal import Principal

log = logging.getLogger(__name__)
SERVER = "core"
MEMORY_TOOL = "memory_context"


class McpToolbox:
    """One MCP session per chat turn.

    Identity goes in MCP _meta, set here from the authenticated principal and never from model
    output. Header overrides are silently dropped on a reused session (probe 2026-09-27), so the
    interceptor calls the session directly with meta.
    """

    def __init__(self, url: str, api_key: str):
        self._client = MultiServerMCPClient(
            {SERVER: {"transport": "streamable_http", "url": url, "headers": {"X-API-Key": api_key}}}
        )

    @asynccontextmanager
    async def open(self, principal: Principal, request_id: str) -> AsyncIterator[list[BaseTool]]:
        meta = principal.mcp_meta() | {"request_id": request_id}
        async with self._client.session(SERVER) as session:

            async def with_identity(request: MCPToolCallRequest, handler):
                return await session.call_tool(request.name, request.args, meta=meta)

            listed = await session.list_tools(params=PaginatedRequestParams(_meta=meta))
            yield [
                convert_mcp_tool_to_langchain_tool(
                    session, tool, tool_interceptors=[with_identity], server_name=SERVER
                )
                for tool in listed.tools
            ]

    async def enter(
        self, stack: AsyncExitStack, principal: Principal, request_id: str
    ) -> list[BaseTool] | None:
        try:
            return await stack.enter_async_context(self.open(principal, request_id))
        except Exception:  # includes ExceptionGroup from the anyio transport
            log.warning("MCP unavailable; continuing without tools", exc_info=True)
            return None


def uriel_meta(tool: BaseTool) -> dict[str, Any]:
    """The tool's `_meta.uriel` from uriel-tools (category, hidden, unattended); langchain-mcp-adapters keeps
    the MCP `_meta` in the tool's metadata."""
    meta = (tool.metadata or {}).get("_meta")
    flags = meta.get("uriel") if isinstance(meta, dict) else None
    return flags if isinstance(flags, dict) else {}


def is_hidden(tool: BaseTool) -> bool:
    return bool(uriel_meta(tool).get("hidden", False))


def is_unattended(tool: BaseTool) -> bool:
    return bool(uriel_meta(tool).get("unattended", False))


def visible(tools: Sequence[BaseTool]) -> list[BaseTool]:
    """The tools a model may be offered: never one uriel-tools marks hidden, which only the gateway calls."""
    # uriel-tools before 0.8.0 lists memory_context without the hidden flag, and in a shared Talk room the
    # model could pull the mentioner's USER.md and SOUL.md into it. The name check can go once every
    # deployment runs uriel-tools >= 0.8.0.
    return [t for t in tools if not is_hidden(t) and t.name != MEMORY_TOOL]


def unattended(tools: Sequence[BaseTool], wanted: Collection[str]) -> tuple[list[BaseTool], list[str]]:
    """For a scheduled run: the wanted tools that are listed for this person and marked unattended, and the
    names of those that aren't."""
    offered = [t for t in tools if t.name in wanted and is_unattended(t)]
    missing = sorted(set(wanted) - {t.name for t in offered})
    return offered, missing


async def call_json(tool: BaseTool, args: dict[str, Any]) -> Any:
    """Call a tool the gateway uses itself and decode its JSON; None when it returned no content."""
    content = await tool.ainvoke(args)
    # MCP tools return content blocks (probe 2026-09-28); plain langchain tools return the string.
    text = content if isinstance(content, str) else "".join(b.get("text", "") for b in content)
    return json.loads(text) if text.strip() else None
