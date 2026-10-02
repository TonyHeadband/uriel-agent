import asyncio
import contextlib
import hashlib
import json
import uuid

import pytest
from authlib.integrations.base_client.errors import OAuthError
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from tests.fakes import ScriptedChatModel
from tests.test_chat import MemoryConversations, Toolbox
from uriel.agent.decider import Decision
from uriel.config import GatewaySettings
from uriel.gateway import app as app_module
from uriel.gateway.app import Coworker, Services, build_coworker, create_app, run_coworker
from uriel.gateway.auth import ServiceKeys, _ServiceKey
from uriel.gateway.chat import ChatService
from uriel.principal import Principal

ADMIN_KEY, FAMILY_KEY = "admin-key", "family-key"
ADMIN = Principal("dad", frozenset({"admins", "family"}), "service")
KID = Principal("kid", frozenset({"family"}), "service")


@tool
def homelab_status() -> str:
    """Homelab."""
    return "homelab ready"


homelab_status.metadata = {"_meta": {"uriel": {"category": "homelab", "companion_action": "lookup"}}}


class Decider:
    async def choose(self, point, context, options):
        return Decision(point, "tools", 0.9, "llm", "fake", 1)


class PerUserToolbox(Toolbox):
    async def enter(self, stack, principal, request_id):
        return self.tools if "admins" in principal.groups else []


class TrackingToolbox(Toolbox):
    """Proves which task closes the MCP `async with` session on the stack ChatService.stream owns.

    anyio requires an `async with` to be entered and exited by the same task; asyncio.Lock alone
    can't detect a violation of that (its release isn't task-bound), so a real bug here would slip
    past an assertion that only checks the lock. `stack.callback` mirrors how McpToolbox.enter
    registers the session's exit on the same AsyncExitStack (`agent/mcp_tools.py`).
    """

    def __init__(self, tools):
        super().__init__(tools)
        self.enter_task = None
        self.exit_task = None
        self.closed = False

    async def enter(self, stack, principal, request_id):
        self.enter_task = asyncio.current_task()

        def record_exit():
            self.exit_task = asyncio.current_task()
            self.closed = True

        stack.callback(record_exit)
        return self.tools


class FlakyChatModel(ScriptedChatModel):
    """Fails the first invocation only, then answers normally.

    Proves ChatService.stream releases its per-thread lock (and closes its MCP `async with`) on
    the failing turn, so the next turn on the same conversation still goes through.
    """

    fail_first: bool = True

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.fail_first:
            self.fail_first = False
            raise ConnectionError("ollama unreachable")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class FakeOAuthClient:
    def __init__(self, claims):
        self.claims = claims

    async def authorize_redirect(self, request, redirect_uri):
        from starlette.responses import RedirectResponse

        return RedirectResponse(
            f"https://auth.example/authorize?redirect_uri={redirect_uri}&code_challenge=x"
        )

    async def authorize_access_token(self, request):
        return {"userinfo": self.claims}


class RaisingOAuthClient(FakeOAuthClient):
    async def authorize_access_token(self, request):
        raise OAuthError(error="mismatching_state", description="CSRF Warning! State mismatch.")


class FakeOAuth:
    def __init__(self, claims):
        self.authelia = FakeOAuthClient(claims)


SETTINGS = GatewaySettings(
    database_url="postgresql://unused",
    mcp_url="http://unused",
    mcp_api_key="k",
    oidc_issuer="https://auth.example",
    oidc_client_id="uriel",
    oidc_client_secret="s",
    session_secret="s" * 32,
    public_url="http://testserver",
    cookie_secure=False,
    max_message_chars=50,
)


def make_client(replies, claims=None, settings=None):
    chat = ChatService(
        chat_model=ScriptedChatModel(messages=iter(replies)),
        decider=Decider(),
        toolbox=PerUserToolbox([homelab_status]),
        checkpointer=InMemorySaver(),
        conversations=MemoryConversations(),
        decision_log=None,
        recursion_limit=10,
        max_message_chars=50,
    )
    keys = ServiceKeys(
        [
            _ServiceKey(hashlib.sha256(ADMIN_KEY.encode()).hexdigest(), ADMIN),
            _ServiceKey(hashlib.sha256(FAMILY_KEY.encode()).hexdigest(), KID),
        ]
    )
    services = Services(chat=chat, keys=keys, verifier=None, oauth=FakeOAuth(claims or {}))
    return TestClient(create_app(settings or SETTINGS, services))


def test_liveness_and_unauthenticated_api():
    c = make_client([])
    assert c.get("/livez").json() == {"status": "ok"}
    assert c.post("/v1/chat", json={"message": "hi"}).status_code == 401


def test_admin_gets_tool_backed_answer_and_history():
    c = make_client(
        [
            AIMessage("", tool_calls=[{"name": "homelab_status", "args": {}, "id": "c1"}]),
            AIMessage("Homelab is ready."),
        ]
    )
    r = c.post("/v1/chat", json={"message": "homelab?"}, headers={"X-API-Key": ADMIN_KEY})
    body = r.json()
    assert r.status_code == 200
    assert (body["reply"], body["tools_used"]) == ("Homelab is ready.", ["homelab_status"])
    msgs = c.get(f"/v1/conversations/{body['conversation_id']}/messages", headers={"X-API-Key": ADMIN_KEY})
    assert [m["role"] for m in msgs.json()] == ["user", "tool", "assistant"]


def test_family_member_cannot_read_admins_conversation():
    c = make_client([AIMessage("hello")])
    cid = c.post("/v1/chat", json={"message": "hi"}, headers={"X-API-Key": ADMIN_KEY}).json()[
        "conversation_id"
    ]
    r = c.post("/v1/chat", json={"message": "hi", "conversation_id": cid}, headers={"X-API-Key": FAMILY_KEY})
    assert r.status_code == 404
    assert c.get(f"/v1/conversations/{cid}/messages", headers={"X-API-Key": FAMILY_KEY}).status_code == 404


@pytest.mark.parametrize("message", ["", "   ", "x" * 51])
def test_invalid_message_is_422(message):
    r = make_client([]).post("/v1/chat", json={"message": message}, headers={"X-API-Key": ADMIN_KEY})
    assert r.status_code == 422


def test_web_page_redirects_to_login_when_anonymous():
    r = make_client([]).get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/auth/login"


def test_login_callback_without_groups_is_refused():
    c = make_client([], claims={"preferred_username": "dad"})
    r = c.get("/auth/callback", follow_redirects=False)
    assert r.status_code == 403
    assert "groups" in r.text
    assert c.get("/", follow_redirects=False).status_code == 303


def test_login_then_chat_over_sse():
    c = make_client([AIMessage("Hi there.")], claims={"preferred_username": "dad", "groups": ["family"]})
    assert c.get("/auth/callback", follow_redirects=False).status_code == 303
    page = c.get("/")
    assert page.status_code == 200 and "dad" in page.text
    cid = str(uuid.uuid4())
    frag = c.post("/chat", data={"conversation_id": cid, "message": "hello"})
    assert frag.status_code == 200 and "sse-connect" in frag.text
    turn_id = frag.text.split('sse-connect="/chat/stream/')[1].split('"')[0]
    stream = c.get(f"/chat/stream/{turn_id}")
    assert "event: token" in stream.text and "Hi" in stream.text and "event: done" in stream.text
    # a turn streams only once; the second GET must not 404 (htmx/sse.js retries a 4xx forever)
    replay = c.get(f"/chat/stream/{turn_id}")
    assert replay.status_code == 200
    assert "event: error" in replay.text and "event: done" in replay.text
    assert "no longer available" in replay.text.lower()


def test_chat_error_is_503_and_releases_the_lock_for_the_next_turn():
    toolbox = TrackingToolbox([homelab_status])
    chat = ChatService(
        chat_model=FlakyChatModel(messages=iter([AIMessage("Back online.")])),
        decider=Decider(),
        toolbox=toolbox,
        checkpointer=InMemorySaver(),
        conversations=MemoryConversations(),
        decision_log=None,
        recursion_limit=10,
        max_message_chars=50,
    )
    keys = ServiceKeys([_ServiceKey(hashlib.sha256(ADMIN_KEY.encode()).hexdigest(), ADMIN)])
    services = Services(chat=chat, keys=keys, verifier=None, oauth=FakeOAuth({}))
    c = TestClient(create_app(SETTINGS, services))
    cid = str(uuid.uuid4())

    r = c.post("/v1/chat", json={"message": "hi", "conversation_id": cid}, headers={"X-API-Key": ADMIN_KEY})
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"]
    # The toolbox's stack.callback must run in the same task that entered it, not via
    # GC-driven asyncgen finalization on some other task.
    assert toolbox.closed is True
    assert toolbox.exit_task is toolbox.enter_task

    r2 = c.post(
        "/v1/chat", json={"message": "hi again", "conversation_id": cid}, headers={"X-API-Key": ADMIN_KEY}
    )
    assert r2.status_code == 200
    assert r2.json()["reply"] == "Back online."


def test_callback_oauth_error_becomes_a_generic_400():
    c = make_client([])
    c.app.state.services.oauth.authelia = RaisingOAuthClient({})
    r = c.get("/auth/callback", follow_redirects=False)
    assert r.status_code == 400
    assert "mismatching_state" not in r.text
    assert "try again" in r.text.lower()
    assert c.get("/", follow_redirects=False).status_code == 303  # session was cleared


def test_bearer_header_with_no_verifier_fails_closed_even_with_a_valid_session():
    c = make_client([], claims={"preferred_username": "dad", "groups": ["family"]})
    assert c.get("/auth/callback", follow_redirects=False).status_code == 303
    r = c.post("/v1/chat", json={"message": "hi"}, headers={"Authorization": "Bearer whatever"})
    assert r.status_code == 401


def test_invalid_api_key_fails_closed_even_with_a_valid_session():
    c = make_client([], claims={"preferred_username": "dad", "groups": ["family"]})
    assert c.get("/auth/callback", follow_redirects=False).status_code == 303
    r = c.post("/v1/chat", json={"message": "hi"}, headers={"X-API-Key": "not-a-real-key"})
    assert r.status_code == 401


def _looping_chat_service(recursion_limit, said=""):
    call = {"name": "homelab_status", "args": {}}
    loop = [AIMessage(said, tool_calls=[call | {"id": f"c{i}"}]) for i in range(30)]
    return ChatService(
        chat_model=ScriptedChatModel(messages=iter(loop)),
        decider=Decider(),
        toolbox=PerUserToolbox([homelab_status]),
        checkpointer=InMemorySaver(),
        conversations=MemoryConversations(),
        decision_log=None,
        recursion_limit=recursion_limit,
        max_message_chars=50,
    )


def test_recursion_limit_hit_is_a_clean_200_not_a_503():
    keys = ServiceKeys([_ServiceKey(hashlib.sha256(ADMIN_KEY.encode()).hexdigest(), ADMIN)])
    services = Services(chat=_looping_chat_service(4), keys=keys, verifier=None, oauth=FakeOAuth({}))
    c = TestClient(create_app(SETTINGS, services))
    r = c.post("/v1/chat", json={"message": "loop"}, headers={"X-API-Key": ADMIN_KEY})
    assert r.status_code == 200
    assert "couldn't finish" in r.json()["reply"]


def test_recursion_limit_hit_in_web_chat_shows_a_friendly_message():
    keys = ServiceKeys([_ServiceKey(hashlib.sha256(ADMIN_KEY.encode()).hexdigest(), ADMIN)])
    claims = {"preferred_username": "dad", "groups": ["admins", "family"]}
    services = Services(chat=_looping_chat_service(4), keys=keys, verifier=None, oauth=FakeOAuth(claims))
    c = TestClient(create_app(SETTINGS, services))
    assert c.get("/auth/callback", follow_redirects=False).status_code == 303
    cid = str(uuid.uuid4())
    frag = c.post("/chat", data={"conversation_id": cid, "message": "loop"})
    turn_id = frag.text.split('sse-connect="/chat/stream/')[1].split('"')[0]
    stream = c.get(f"/chat/stream/{turn_id}")
    assert "event: token" in stream.text
    assert "reasonable number of steps" in stream.text
    assert "event: done" in stream.text


class BrokenPool:
    """A pool whose connection() always fails, for exercising /health's DB-down path."""

    def connection(self):
        return self

    async def __aenter__(self):
        raise ConnectionError("db unreachable")

    async def __aexit__(self, *exc_info):
        return False


def test_health_logs_the_exception_before_returning_503(caplog):
    c = make_client([])
    c.app.state.services.pool = BrokenPool()
    with caplog.at_level("WARNING"):
        r = c.get("/health")
    assert r.status_code == 503
    assert any("database unavailable" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_docs_are_not_exposed_on_the_vpn(path):
    c = make_client([])
    assert c.get(path).status_code == 404


def test_post_chat_without_a_session_gets_an_hx_redirect_not_a_bare_401():
    c = make_client([])
    r = c.post("/chat", data={"conversation_id": str(uuid.uuid4()), "message": "hi"})
    assert r.status_code == 204
    assert r.headers["hx-redirect"] == "/auth/login"


def test_unknown_turn_id_gets_a_generic_error_stream_not_a_404():
    c = make_client([], claims={"preferred_username": "dad", "groups": ["family"]})
    assert c.get("/auth/callback", follow_redirects=False).status_code == 303
    r = c.get("/chat/stream/does-not-exist")
    assert r.status_code == 200
    assert "event: error" in r.text and "event: done" in r.text
    assert "no longer available" in r.text.lower()


def test_another_users_turn_id_gets_a_generic_error_and_the_owner_can_still_stream_it():
    c1 = make_client([AIMessage("For dad only.")], claims={"preferred_username": "dad", "groups": ["family"]})
    app = c1.app
    assert c1.get("/auth/callback", follow_redirects=False).status_code == 303
    cid = str(uuid.uuid4())
    frag = c1.post("/chat", data={"conversation_id": cid, "message": "hello"})
    turn_id = frag.text.split('sse-connect="/chat/stream/')[1].split('"')[0]

    c2 = TestClient(app)
    app.state.services.oauth.authelia = FakeOAuthClient({"preferred_username": "kid", "groups": ["family"]})
    assert c2.get("/auth/callback", follow_redirects=False).status_code == 303

    foreign = c2.get(f"/chat/stream/{turn_id}")
    assert foreign.status_code == 200
    assert "event: error" in foreign.text and "event: done" in foreign.text
    assert "dad" not in foreign.text

    owner = c1.get(f"/chat/stream/{turn_id}")
    assert "event: token" in owner.text and "dad" in owner.text


def test_logout_shows_a_signed_out_page_instead_of_bouncing_back_into_sso():
    c = make_client([], claims={"preferred_username": "dad", "groups": ["family"]})
    assert c.get("/auth/callback", follow_redirects=False).status_code == 303
    r = c.post("/auth/logout", follow_redirects=False)
    assert r.status_code == 200
    assert "sign in again" in r.text.lower()
    assert "/auth/login" in r.text
    # the session was actually cleared, not just displayed as cleared
    assert c.get("/", follow_redirects=False).headers["location"] == "/auth/login"


def test_logout_page_links_to_the_authelia_portal_logout_when_configured():
    settings = SETTINGS.model_copy(
        update={"oidc_logout_url": "https://auth.example/logout", "public_url": "https://uriel.example"}
    )
    c = make_client([], claims={"preferred_username": "dad", "groups": ["family"]}, settings=settings)
    assert c.get("/auth/callback", follow_redirects=False).status_code == 303
    r = c.post("/auth/logout", follow_redirects=False)
    assert r.status_code == 200
    assert "sign out of all family services" in r.text.lower()
    assert "https://auth.example/logout?rd=https%3A%2F%2Furiel.example" in r.text


def test_logout_page_has_no_authelia_link_when_not_configured():
    c = make_client([], claims={"preferred_username": "dad", "groups": ["family"]})
    assert c.get("/auth/callback", follow_redirects=False).status_code == 303
    r = c.post("/auth/logout", follow_redirects=False)
    assert "family services" not in r.text.lower()


async def test_gateway_refuses_a_hosted_model_before_touching_anything(tmp_path, monkeypatch):
    import contextlib
    import socket

    from uriel.gateway.app import _build_services

    models = tmp_path / "models.yaml"
    models.write_text(
        "models:\n"
        "  gemini: { provider: openai_compat, base_url: 'https://generativelanguage.googleapis.com/v1beta/openai/',"
        " model: gemini, api_key_env: GEMINI_API_KEY }\n"
        "roles: { interactive: gemini, background: gemini }\n"
        "deciders: { route: { adapter: llm, model: gemini } }\n"
    )
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda host, port, *a, **kw: [(2, 1, 6, "", ("142.250.184.10", 0))]
    )
    settings = SETTINGS.model_copy(update={"models_file": models})
    async with contextlib.AsyncExitStack() as stack:
        with pytest.raises(RuntimeError, match="outside the house network \\(gemini\\)"):
            await _build_services(settings, stack)  # database_url is unusable: it must fail before the pool


COWORKER = {
    "nc_url": "https://cloud.example",
    "nc_app_password": "p",
    "ldap_url": "ldap://lldap:3890",
    "ldap_bind_dn": "uid=uriel-gateway,ou=people,dc=example,dc=com",
    "ldap_password": "p",
    "ldap_base_dn": "dc=example,dc=com",
}


def test_the_coworker_is_off_unless_enabled():
    assert build_coworker(SETTINGS, chat=None, pool=None, toolbox=None) is None


@pytest.mark.parametrize(("talk", "runner"), [(True, False), (False, True), (True, True)])
async def test_the_coworker_starts_only_what_is_enabled(talk, runner):
    settings = SETTINGS.model_copy(update=COWORKER | {"talk_enabled": talk, "runner_enabled": runner})
    coworker = build_coworker(settings, chat=object(), pool=object(), toolbox=object())
    try:
        assert isinstance(coworker, Coworker)
        assert (coworker.channel is not None, coworker.runner is not None) == (talk, runner)
    finally:
        await coworker.talk.aclose()


class Polling:
    """A Talk channel whose poll loop runs until cancelled and whose room turns finish on drain."""

    def __init__(self, events):
        self.events = events
        self.cancelled = False

    async def run(self):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            self.events.append("poll cancelled")
            raise

    def close(self):
        self.events.append("close")

    def stop(self):
        self.events.append("stop")

    async def drain(self):
        self.events.append("drain")


class Running:
    """A runner in the middle of a run: it returns once stopped and its run is done."""

    def __init__(self, events):
        self.events = events
        self.stopped = asyncio.Event()

    async def run(self):
        await self.stopped.wait()
        self.events.append("run finished")

    def stop(self):
        self.events.append("runner stop")
        self.stopped.set()


class ClosingTalk:
    def __init__(self, events):
        self.events = events

    async def aclose(self):
        self.events.append("talk closed")


async def test_shutdown_stops_taking_work_then_lets_turns_in_flight_finish_and_closes_talk():
    events = []
    channel, runner = Polling(events), Running(events)
    coworker = Coworker(ClosingTalk(events), channel, runner)
    async with contextlib.AsyncExitStack() as stack:
        run_coworker(stack, coworker)
        await asyncio.sleep(0)
    assert channel.cancelled
    assert events.index("close") < events.index("drain")
    assert events.index("runner stop") < events.index("run finished")
    assert events[-1] == "talk closed"


async def test_shutdown_cancels_what_is_still_running_after_the_grace(monkeypatch):
    monkeypatch.setattr(app_module, "SHUTDOWN_GRACE_S", 0.05)

    class Stuck(Running):
        def stop(self):
            self.events.append("runner stop")  # and never finishes

    events = []
    runner = Stuck(events)

    async def shut_down():
        async with contextlib.AsyncExitStack() as stack:
            run_coworker(stack, Coworker(ClosingTalk(events), None, runner))
            await asyncio.sleep(0)

    await asyncio.wait_for(shut_down(), 2)
    assert "run finished" not in events and events[-1] == "talk closed"


async def test_the_coworker_serves_only_the_configured_member_groups():
    settings = SETTINGS.model_copy(
        update=COWORKER | {"talk_enabled": True, "runner_enabled": True, "member_groups": ["family", "gran"]}
    )
    coworker = build_coworker(settings, chat=object(), pool=object(), toolbox=object())
    try:
        assert coworker.channel._members == coworker.runner._members == {"family", "gran"}
    finally:
        await coworker.talk.aclose()


def sse_events(text):
    """(event, data) pairs from a raw SSE body; sse-starlette separates lines with CRLF."""
    out = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        lines = [ln for ln in block.split("\n") if ln and not ln.startswith(":")]
        if not lines:
            continue
        event = next(ln[len("event: ") :] for ln in lines if ln.startswith("event: "))
        data = "\n".join(ln[len("data: ") :] for ln in lines if ln.startswith("data: "))
        out.append((event, json.loads(data)))
    return out


def stream(c, body, key=ADMIN_KEY):
    return c.post("/v1/chat/stream", json=body, headers={"X-API-Key": key})


def test_stream_tool_turn_sends_start_tool_tokens_done():
    c = make_client(
        [
            AIMessage("", tool_calls=[{"name": "homelab_status", "args": {}, "id": "c1"}]),
            AIMessage("Homelab is ready."),
        ]
    )
    r = stream(c, {"message": "homelab?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    names = [e for e, _ in events]
    assert names[0] == "start" and names[-1] == "done"
    tags = {"name": "homelab_status", "category": "homelab", "companion_action": "lookup"}
    assert ("tool_call", tags) in events
    assert ("tool_result", tags | {"status": "success", "awaiting": False}) in events
    assert names.index("tool_call") < names.index("tool_result") < names.index("token")
    assert "".join(d["text"] for e, d in events if e == "token") == "Homelab is ready."
    assert "final" not in names  # tokens already carried the text
    cid = events[0][1]["conversation_id"]
    msgs = c.get(f"/v1/conversations/{cid}/messages", headers={"X-API-Key": ADMIN_KEY}).json()
    assert [m["role"] for m in msgs] == ["user", "tool", "assistant"]


def test_stream_continues_an_existing_conversation():
    c = make_client([AIMessage("one"), AIMessage("two")])
    cid = sse_events(stream(c, {"message": "a"}).text)[0][1]["conversation_id"]
    second = sse_events(stream(c, {"message": "b", "conversation_id": cid}).text)
    assert second[0] == ("start", {"conversation_id": cid})


def test_stream_requires_auth_and_validates_like_chat():
    c = make_client([AIMessage("hello")])
    assert c.post("/v1/chat/stream", json={"message": "hi"}).status_code == 401
    assert stream(c, {"message": "  "}).status_code == 422
    first = c.post("/v1/chat", json={"message": "hi"}, headers={"X-API-Key": ADMIN_KEY})
    cid = first.json()["conversation_id"]
    assert stream(c, {"message": "hi", "conversation_id": cid}, key=FAMILY_KEY).status_code == 404


def test_stream_error_ends_with_done_and_releases_the_session_in_its_own_task():
    toolbox = TrackingToolbox([homelab_status])
    chat = ChatService(
        chat_model=FlakyChatModel(messages=iter([AIMessage("Back online.")])),
        decider=Decider(),
        toolbox=toolbox,
        checkpointer=InMemorySaver(),
        conversations=MemoryConversations(),
        decision_log=None,
        recursion_limit=10,
        max_message_chars=50,
    )
    keys = ServiceKeys([_ServiceKey(hashlib.sha256(ADMIN_KEY.encode()).hexdigest(), ADMIN)])
    c = TestClient(create_app(SETTINGS, Services(chat=chat, keys=keys, verifier=None, oauth=FakeOAuth({}))))
    cid = str(uuid.uuid4())
    events = sse_events(stream(c, {"message": "hi", "conversation_id": cid}).text)
    assert [e for e, _ in events][-2:] == ["error", "done"]
    assert "unavailable" in events[-2][1]["message"]
    assert toolbox.closed is True and toolbox.exit_task is toolbox.enter_task
    again = sse_events(stream(c, {"message": "hi again", "conversation_id": cid}).text)
    assert "".join(d["text"] for e, d in again if e == "token") == "Back online."


def test_stream_sends_final_when_no_tokens_were_streamed():
    keys = ServiceKeys([_ServiceKey(hashlib.sha256(ADMIN_KEY.encode()).hexdigest(), ADMIN)])
    services = Services(chat=_looping_chat_service(4), keys=keys, verifier=None, oauth=FakeOAuth({}))
    events = sse_events(stream(TestClient(create_app(SETTINGS, services)), {"message": "loop"}).text)
    finals = [d["text"] for e, d in events if e == "final"]
    assert len(finals) == 1 and "couldn't finish" in finals[0]
    assert "".join(d["text"] for e, d in events if e == "token") == ""
    assert events[-1] == ("done", {})


def test_stream_notices_carry_a_code():
    c = make_client([AIMessage("Hi.")])
    events = sse_events(stream(c, {"message": "hello"}).text)
    assert ("notice", {"text": "Nothing was looked up for this answer.", "code": "not_looked_up"}) in events


def test_stream_still_says_it_couldnt_finish_after_streamed_tokens():
    keys = ServiceKeys([_ServiceKey(hashlib.sha256(ADMIN_KEY.encode()).hexdigest(), ADMIN)])
    services = Services(
        chat=_looping_chat_service(4, "Checking."), keys=keys, verifier=None, oauth=FakeOAuth({})
    )
    events = sse_events(stream(TestClient(create_app(SETTINGS, services)), {"message": "loop"}).text)
    said = "".join(d["text"] for e, d in events if e == "token")
    assert said.startswith("Checking.") and said.endswith(
        "\n\nI couldn't finish that in a reasonable number of steps. Try rephrasing?"
    )
    assert "final" not in [e for e, _ in events]


def test_web_chat_still_says_it_couldnt_finish_after_streamed_tokens():
    keys = ServiceKeys([_ServiceKey(hashlib.sha256(ADMIN_KEY.encode()).hexdigest(), ADMIN)])
    claims = {"preferred_username": "dad", "groups": ["admins", "family"]}
    services = Services(
        chat=_looping_chat_service(4, "Checking."), keys=keys, verifier=None, oauth=FakeOAuth(claims)
    )
    c = TestClient(create_app(SETTINGS, services))
    assert c.get("/auth/callback", follow_redirects=False).status_code == 303
    frag = c.post("/chat", data={"conversation_id": str(uuid.uuid4()), "message": "loop"})
    turn_id = frag.text.split('sse-connect="/chat/stream/')[1].split('"')[0]
    body = c.get(f"/chat/stream/{turn_id}").text
    assert "Checking." in body and "reasonable number of steps" in body


def test_stream_data_is_not_html_escaped():
    c = make_client([AIMessage("1 < 2 & <b>bold</b>")])
    events = sse_events(stream(c, {"message": "x"}).text)
    assert "".join(d["text"] for e, d in events if e == "token") == "1 < 2 & <b>bold</b>"


CORS_SETTINGS = SETTINGS.model_copy(update={"cors_origins": ["tauri://localhost"]})


def test_cors_preflight_is_allowed_for_the_companion_origin_only():
    c = make_client([], settings=CORS_SETTINGS)
    pre = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    }
    ok = c.options("/v1/chat/stream", headers={"Origin": "tauri://localhost", **pre})
    assert ok.headers.get("access-control-allow-origin") == "tauri://localhost"
    # Bearer tokens only: the session cookie must never be usable cross-origin.
    assert "access-control-allow-credentials" not in ok.headers
    other = c.options("/v1/chat/stream", headers={"Origin": "https://evil.example", **pre})
    assert "access-control-allow-origin" not in other.headers


def test_no_cors_headers_without_configured_origins():
    r = make_client([]).get("/livez", headers={"Origin": "tauri://localhost"})
    assert "access-control-allow-origin" not in r.headers
