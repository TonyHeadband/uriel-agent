"""A stand-in for uriel-tools that speaks its wire contract (uriel-tools docs/contract.md).

It checks the x-api-key header, lists only the tools the caller's `_meta` groups allow, and echoes the
`_meta` it received, so the agent's tests prove what the gateway sends without the real tool server.
"""

import hmac

from fastmcp import Context, FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from starlette.middleware import Middleware as ASGIMiddleware
from starlette.responses import JSONResponse

TOOL_GROUPS = {"homelab_status": "admins", "search_documents": "family", "memory_context": "family"}


def _meta(ctx: Context | None) -> dict:
    meta = ctx.request_context.meta if ctx and ctx.request_context else None
    return meta.model_dump(exclude_none=True) if meta else {}


class _GroupFilter(Middleware):
    async def on_list_tools(self, context: MiddlewareContext, call_next):
        groups = set(_meta(context.fastmcp_context).get("groups", []))
        return [t for t in await call_next(context) if TOOL_GROUPS[t.name] in groups]


class _ApiKey:
    def __init__(self, app, key: str):
        self.app, self.key = app, key.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not hmac.compare_digest(
            dict(scope["headers"]).get(b"x-api-key", b""), self.key
        ):
            await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def build_fake_app(api_key: str):
    mcp = FastMCP("fake-uriel-tools", middleware=[_GroupFilter()])

    @mcp.tool
    def homelab_status(ctx: Context) -> dict:
        """Echo the caller identity the gateway sent."""
        return {"received_meta": _meta(ctx)}

    @mcp.tool
    def search_documents(query: str, ctx: Context) -> dict:
        """Echo the caller identity the gateway sent."""
        return {"query": query, "received_meta": _meta(ctx)}

    @mcp.tool
    def memory_context(ctx: Context) -> dict:
        """The caller's USER.md and SOUL.md, keyed to the user the gateway sent."""
        return {"user": f"# About {_meta(ctx).get('user')}\n## Name\n- Tony", "soul": "## Length\n- Short"}

    return mcp.http_app(middleware=[ASGIMiddleware(_ApiKey, key=api_key)])
