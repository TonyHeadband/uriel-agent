"""A stand-in for uriel-tools that speaks its wire contract (uriel-tools docs/contract.md, 0.8.0).

It checks the x-api-key header, lists only the tools the caller's `_meta` groups allow, each with its
`_meta.uriel` flags (hidden, unattended), and echoes the `_meta` it received, so the agent's tests prove what
the gateway sends without the real tool server. `FakeRuns` stands in for the schedules database behind the
runner's tools.
"""

import hmac
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import uvicorn
from fastmcp import Context, FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from starlette.middleware import Middleware as ASGIMiddleware
from starlette.responses import JSONResponse

from uriel.principal import INTERNAL_GROUP

TOOL_GROUPS = {
    "homelab_status": "admins",
    "search_documents": "family",
    "memory_context": "family",
    "web_search": "family",
    "remember": "family",
    "claim_due_runs": INTERNAL_GROUP,
    "finish_run": INTERNAL_GROUP,
}


def uriel(**flags) -> dict:
    """Tool meta as uriel-tools lists it: its own flags under `uriel`, next to `category` since 0.7.0."""
    return {"uriel": flags}


@dataclass
class FakeRuns:
    due: list[dict] = field(default_factory=list)
    finished: list[dict] = field(default_factory=list)
    claimed_by: list[dict] = field(default_factory=list)


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


def build_fake_app(api_key: str, runs: FakeRuns | None = None):
    runs = runs if runs is not None else FakeRuns()
    mcp = FastMCP("fake-uriel-tools", middleware=[_GroupFilter()])

    @mcp.tool
    def homelab_status(ctx: Context) -> dict:
        """Echo the caller identity the gateway sent."""
        return {"received_meta": _meta(ctx)}

    @mcp.tool(meta=uriel(unattended=True))
    def search_documents(query: str, ctx: Context) -> dict:
        """Echo the caller identity the gateway sent."""
        return {"query": query, "received_meta": _meta(ctx)}

    @mcp.tool(meta=uriel(hidden=True))
    def memory_context(ctx: Context) -> dict:
        """The caller's USER.md and SOUL.md, keyed to the user the gateway sent."""
        return {
            "user": f"# About {_meta(ctx).get('user')}\n## Name\n- Tony",
            "soul": "## Length\n- Short",
            "tz": "America/Toronto",
            "now_local": "Tue 29 Sep 2026, 22:41",
        }

    @mcp.tool(meta=uriel(unattended=True))
    def web_search(query: str, ctx: Context) -> dict:
        """Search the web."""
        return {"query": query, "results": [{"title": "F1 news", "url": "https://example.com/f1"}]}

    @mcp.tool
    def remember(section: str, text: str, ctx: Context) -> dict:
        """Save a fact about the person."""
        return {"status": "saved", "section": section}

    @mcp.tool(meta=uriel(hidden=True))
    def claim_due_runs(ctx: Context, limit: int = 5) -> dict:
        """Claim due scheduled runs (the gateway's runner only)."""
        runs.claimed_by.append(_meta(ctx))
        taken, runs.due[:] = runs.due[:limit], runs.due[limit:]
        return {"runs": taken}

    @mcp.tool(meta=uriel(hidden=True))
    def finish_run(
        run_id: int,
        status: str,
        ctx: Context,
        summary: str | None = None,
        error: str | None = None,
        talk_message_id: int | None = None,
    ) -> dict:
        """Record a run's outcome (the gateway's runner only)."""
        for done in runs.finished:
            if done["run_id"] == run_id:
                # Finishing twice changes nothing, as in uriel-tools.
                return {"run_id": run_id, "status": done["status"], "finished": False}
        runs.finished.append(
            {
                "run_id": run_id,
                "status": status,
                "summary": summary,
                "error": error,
                "talk_message_id": talk_message_id,
            }
        )
        return {"run_id": run_id, "status": status, "finished": True}

    return mcp.http_app(middleware=[ASGIMiddleware(_ApiKey, key=api_key)])


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def serve(app) -> Iterator[str]:
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("MCP test server did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(5)
