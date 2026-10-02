import html
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from uriel.gateway.chat import UNFINISHED, ConversationNotFound, InvalidMessage, Turn
from uriel.principal import Principal

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Generic on purpose: an unknown, already-consumed or another user's turn_id all look the same
# from the outside, so the message must never hint at which one it was (no leak, no enumeration).
_STREAM_UNAVAILABLE = "This reply is no longer available."


async def _unavailable_stream():
    yield {"event": "error", "data": f'<span class="error">{html.escape(_STREAM_UNAVAILABLE)}</span>'}
    yield {"event": "done", "data": ""}


def session_principal(request: Request) -> Principal | None:
    data = request.session.get("principal")
    return Principal.from_session(data) if data else None


def build_router() -> APIRouter:
    router = APIRouter()
    # A POST registers a turn; the SSE GET streams it exactly once. Single replica, so memory is fine.
    pending: dict[str, Turn] = {}

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request, c: uuid.UUID | None = None):
        principal = session_principal(request)
        if principal is None:
            return RedirectResponse("/auth/login", status_code=303)
        svc = request.app.state.services.chat
        history = []
        if c is not None:
            try:
                history = await svc.history(principal, c)
            except ConversationNotFound:
                return RedirectResponse("/", status_code=303)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "user": principal.user_id,
                "conversations": await svc.conversations(principal),
                "conversation_id": c or uuid.uuid4(),
                "history": history,
            },
        )

    @router.post("/chat", response_class=HTMLResponse)
    async def chat(request: Request, conversation_id: uuid.UUID = Form(...), message: str = Form(...)):
        principal = session_principal(request)
        if principal is None:
            # htmx doesn't swap 4xx responses in, so a bare 401 would leave "Send" doing nothing;
            # a 204 with HX-Redirect sends the browser to a full-page login instead.
            return HTMLResponse(status_code=204, headers={"HX-Redirect": "/auth/login"})
        try:
            turn = await request.app.state.services.chat.prepare(principal, conversation_id, message)
        except InvalidMessage as exc:
            return HTMLResponse(f'<div class="msg error">{html.escape(str(exc))}</div>')
        except ConversationNotFound as exc:
            raise HTTPException(404) from exc
        turn_id = uuid.uuid4().hex
        pending[turn_id] = turn
        return templates.TemplateResponse(request, "_turn.html", {"text": turn.text, "turn_id": turn_id})

    @router.get("/chat/stream/{turn_id}")
    async def stream(turn_id: str, request: Request):
        principal = session_principal(request)
        turn = pending.get(turn_id)
        if principal is None or turn is None or turn.principal != principal:
            # Don't pop: a foreign or stale request must not consume someone else's still-pending
            # turn, or the real owner would find it gone when they come to stream it.
            return EventSourceResponse(_unavailable_stream())
        del pending[turn_id]
        svc = request.app.state.services.chat

        async def events():
            # A successful turn already streamed its answer as "token" chunks, so its trailing
            # "final" (the same text in full) is redundant and must not be rendered twice. A
            # recursion-abort "final" is new text, whether or not tokens came before it.
            streamed_tokens = False

            async for e in svc.stream(turn):
                match e.kind:
                    case "token":
                        streamed_tokens = True
                        yield {"event": "token", "data": html.escape(str(e.data))}
                    case "final":
                        if not streamed_tokens:
                            yield {"event": "token", "data": html.escape(str(e.data))}
                        elif e.data == UNFINISHED:
                            # The turn stopped after part of an answer: say so after it, not instead of it.
                            yield {"event": "token", "data": html.escape(f"\n\n{UNFINISHED}")}
                    case "tool_call":
                        name = html.escape(e.data["name"])
                        yield {"event": "tool", "data": f'<span class="status">used {name}</span>'}
                    case "notice":
                        yield {
                            "event": "notice",
                            "data": f'<span class="status">{html.escape(e.data)}</span>',
                        }
                    case "error":
                        yield {"event": "error", "data": f'<span class="error">{html.escape(e.data)}</span>'}
            yield {"event": "done", "data": ""}

        return EventSourceResponse(events())

    return router
