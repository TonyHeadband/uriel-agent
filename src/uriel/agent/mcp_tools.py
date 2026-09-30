import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp.types import PaginatedRequestParams

from uriel.principal import Principal

log = logging.getLogger(__name__)
SERVER = "core"


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
