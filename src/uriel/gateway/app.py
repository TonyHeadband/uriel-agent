import asyncio
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
from uriel.gateway.directory import LdapGroupDirectory
from uriel.gateway.runner import ScheduledRunner
from uriel.gateway.stores import ConversationStore, DecisionLog, TalkCursors
from uriel.gateway.talk import TalkChannel
from uriel.gateway.talk_api import TALK_USER, TalkClient
from uriel.principal import Principal

log = logging.getLogger(__name__)
# How long a shutdown waits for Talk turns and a scheduled run in flight; Kubernetes' default grace is 30 s.
SHUTDOWN_GRACE_S = 20


@dataclass
class Services:
    chat: ChatService
    keys: ServiceKeys
    verifier: JwtVerifier | None
    oauth: Any
    pool: Any = None


@dataclass
class Coworker:
    """Uriel as a Nextcloud coworker: the Talk channel and the scheduled-run runner share one Talk client."""

    talk: TalkClient
    channel: TalkChannel | None
    runner: ScheduledRunner | None


def build_coworker(settings: GatewaySettings, *, chat, pool, toolbox) -> Coworker | None:
    if not (settings.talk_enabled or settings.runner_enabled):
        return None
    talk = TalkClient(settings.nc_url, TALK_USER, settings.nc_app_password)
    directory = LdapGroupDirectory(
        settings.ldap_url, settings.ldap_bind_dn, settings.ldap_password, settings.ldap_base_dn
    )
    channel = (
        TalkChannel(
            client=talk,
            chat=chat,
            directory=directory,
            cursors=TalkCursors(pool),
            poll_seconds=settings.talk_poll_seconds,
            history_turns=settings.talk_history_turns,
            member_groups=settings.member_groups,
        )
        if settings.talk_enabled
        else None
    )
    runner = (
        ScheduledRunner(
            toolbox=toolbox,
            chat=chat,
            directory=directory,
            talk=talk,
            seconds=settings.runner_seconds,
            history_turns=settings.talk_history_turns,
            member_groups=settings.member_groups,
        )
        if settings.runner_enabled
        else None
    )
    return Coworker(talk, channel, runner)


def run_coworker(stack: contextlib.AsyncExitStack, coworker: Coworker) -> None:
    """Start the coworker's loops. Leaving the stack stops polling Talk and claiming runs, gives the Talk
    turns and the scheduled run in flight SHUTDOWN_GRACE_S to finish and post, cancels what's left, then
    closes Talk.

    A cancelled Talk turn leaves its 👀 and is answered again after the restart, re-running its tools, so
    turns in flight are let finish rather than cancelled outright.
    """
    channel, runner = coworker.channel, coworker.runner
    stack.push_async_callback(coworker.talk.aclose)
    poller = asyncio.create_task(channel.run()) if channel is not None else None
    running = asyncio.create_task(runner.run()) if runner is not None else None

    async def shutdown() -> None:
        finishing = []
        if channel is not None:
            poller.cancel()
            channel.close()
            await asyncio.gather(poller, return_exceptions=True)
            finishing.append(asyncio.wait_for(channel.drain(), SHUTDOWN_GRACE_S))
        if runner is not None:
            runner.stop()
            finishing.append(asyncio.wait_for(running, SHUTDOWN_GRACE_S))
        # wait_for cancels what outlives the grace: drain's gather cancels the room turns with it.
        for outcome in await asyncio.gather(*finishing, return_exceptions=True):
            if isinstance(outcome, TimeoutError):
                log.warning(
                    "shutdown: work in flight didn't finish in %s s and was cancelled", SHUTDOWN_GRACE_S
                )
        if channel is not None:
            channel.stop()
            await channel.drain()

    stack.push_async_callback(shutdown)


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
    toolbox = McpToolbox(settings.mcp_url, settings.mcp_api_key)
    chat = ChatService(
        chat_model=chat_model,
        background_model=build_chat_model(models.role("background")),
        tool_model=tool_model,
        decider=decider,
        toolbox=toolbox,
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
    verifier = JwtVerifier.for_issuer(
        settings.oidc_issuer, [settings.oidc_client_id, *settings.oidc_extra_audiences]
    )

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
    if coworker := build_coworker(settings, chat=chat, pool=pool, toolbox=toolbox):
        run_coworker(stack, coworker)
        log.info("Talk channel %s, scheduled runs %s", settings.talk_enabled, settings.runner_enabled)
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
    if settings.cors_origins:
        # The companion authenticates with Bearer tokens; credentials stay off so the session cookie can never
        # be used from another origin.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["authorization", "content-type", "accept", "x-api-key"],
            allow_credentials=False,
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
