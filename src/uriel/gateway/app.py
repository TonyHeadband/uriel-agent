import asyncio
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from starlette.middleware.sessions import SessionMiddleware

from uriel.agent.decider import GuardedDecider, decider_for
from uriel.agent.mcp_tools import McpToolbox
from uriel.agent.models import build_chat_model
from uriel.config import GatewaySettings, ModelsConfig, load_yaml, models_outside_the_house
from uriel.gateway import api, oidc, web
from uriel.gateway.auth import AuthError, JwtVerifier, ServiceKeys
from uriel.gateway.chat import ChatService
from uriel.gateway.db import apply_migrations, make_pool
from uriel.gateway.stores import ConversationStore, DecisionLog
from uriel.principal import Principal

log = logging.getLogger(__name__)


@dataclass
class Services:
    chat: ChatService
    keys: ServiceKeys
    verifier: JwtVerifier | None
    oauth: Any
    pool: Any = None


async def require_principal(request: Request) -> Principal:
    services: Services = request.app.state.services
    if key := request.headers.get("x-api-key"):
        if principal := services.keys.lookup(key):
            return principal
        raise HTTPException(401, "invalid API key")
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        # Fail closed: a bearer token with no verifier configured must never fall through to the
        # session cookie, or a stolen/forged header could ride on someone else's browser session.
        if services.verifier is None:
            raise HTTPException(401, "bearer authentication is not configured")
        try:
            return await asyncio.to_thread(services.verifier.verify, auth[7:])
        except AuthError as exc:
            raise HTTPException(401, str(exc)) from exc
    if data := request.session.get("principal"):
        return Principal.from_session(data)
    raise HTTPException(401, "authentication required")


async def _build_services(settings: GatewaySettings, stack: contextlib.AsyncExitStack) -> Services:
    models = load_yaml(ModelsConfig, settings.models_file)
    if outside := models_outside_the_house(models):
        raise RuntimeError(
            f"{settings.models_file} has models outside the house network ({', '.join(outside)}); hosted "
            "models are for evals only and the gateway won't start with one"
        )
    pool = make_pool(settings.database_url)
    await pool.open()
    stack.push_async_callback(pool.close)
    await apply_migrations(pool)
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    decisions = DecisionLog(pool)
    route_spec = models.decider("route")
    chat_model = build_chat_model(models.role("interactive"))
    tool_model = build_chat_model(models.role("tools"))
    decider = GuardedDecider(decider_for(models.models[route_spec.model]), route_spec, fallback="tools")
    chat = ChatService(
        chat_model=chat_model,
        tool_model=tool_model,
        decider=decider,
        toolbox=McpToolbox(settings.mcp_url, settings.mcp_api_key),
        checkpointer=checkpointer,
        conversations=ConversationStore(pool),
        decision_log=decisions,
        recursion_limit=settings.recursion_limit,
        coverage=route_spec.coverage,
        max_message_chars=settings.max_message_chars,
    )
    keys = (
        ServiceKeys.from_file(settings.service_keys_file)
        if settings.service_keys_file
        else ServiceKeys.empty()
    )
    verifier = JwtVerifier.for_issuer(settings.oidc_issuer, settings.oidc_client_id)

    async def purge_daily():
        while True:
            try:
                removed = await decisions.purge_older_than(settings.decisions_retention_days)
                log.info("purged %d old decisions", removed)
            except Exception:
                log.warning("decision purge failed", exc_info=True)
            await asyncio.sleep(24 * 3600)

    task = asyncio.create_task(purge_daily())
    stack.callback(task.cancel)
    return Services(chat=chat, keys=keys, verifier=verifier, oauth=oidc.register_oauth(settings), pool=pool)


def create_app(settings: GatewaySettings, services: Services | None = None) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async with contextlib.AsyncExitStack() as stack:
            if services is None:
                app.state.services = await _build_services(settings, stack)
            yield

    # Family members are all on the same VPN as everyone else; the interactive API docs would be
    # reachable by anyone on it, not just the gateway's own clients.
    app = FastAPI(title="uriel-gateway", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    # Injected services are known synchronously, so tests that never enter TestClient as a
    # context manager (and thus never run ASGI lifespan events) still see app.state.services.
    if services is not None:
        app.state.services = services
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        session_cookie="uriel_session",
        max_age=settings.session_max_age_s,
        same_site="lax",
        https_only=settings.cookie_secure,
    )
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    app.include_router(oidc.build_router(settings))
    app.include_router(api.build_router(require_principal))
    app.include_router(web.build_router())

    @app.get("/livez")
    async def livez():
        return {"status": "ok"}

    @app.get("/health")
    async def health(request: Request):
        # Deliberately ignores MCP: a tool outage must not take chat down.
        pool = request.app.state.services.pool
        if pool is not None:
            try:
                async with pool.connection() as conn:
                    await conn.execute("SELECT 1")
            except Exception:
                log.warning("health check: database unavailable", exc_info=True)
                return JSONResponse({"status": "db unavailable"}, status_code=503)
        return {"status": "ok"}

    return app


def create_app_from_env() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return create_app(GatewaySettings())
