import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from uriel.gateway.chat import (
    NOT_LOOKED_UP,
    TOOLS_UNAVAILABLE,
    UNFINISHED,
    ConversationNotFound,
    InvalidMessage,
)
from uriel.principal import Principal


class ChatRequest(BaseModel):
    message: str
    conversation_id: uuid.UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    reply: str
    tools_used: list[str]
    notices: list[str]


# Lets the companion react to a notice without matching its wording; other notices go out with code null.
NOTICE_CODES = {NOT_LOOKED_UP: "not_looked_up", TOOLS_UNAVAILABLE: "tools_unavailable"}
TOOL_FIELDS = ("name", "category", "companion_action")


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data)}


def build_router(require_principal) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.post("/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest, request: Request, principal: Principal = Depends(require_principal)):
        svc = request.app.state.services.chat
        try:
            turn = await svc.prepare(principal, body.conversation_id, body.message)
        except InvalidMessage as exc:
            raise HTTPException(422, str(exc)) from exc
        except ConversationNotFound as exc:
            raise HTTPException(404, "conversation not found") from exc
        reply, tools, notices, error = "", [], [], None
        # Drain the generator fully in this task before raising: svc.stream holds the per-thread
        # lock and the MCP session in an `async with` that anyio requires closing from this task,
        # not from asyncgen finalization on another one.
        async for event in svc.stream(turn):
            match event.kind:
                case "final":
                    reply = str(event.data)
                case "tool_call":
                    tools.append(event.data["name"])
                case "notice":
                    notices.append(event.data)
                case "error":
                    error = error or event.data
        if error is not None:
            raise HTTPException(503, error)
        return ChatResponse(
            conversation_id=turn.conversation_id, reply=reply, tools_used=tools, notices=notices
        )

    @router.post("/chat/stream")
    async def chat_stream(
        body: ChatRequest, request: Request, principal: Principal = Depends(require_principal)
    ):
        svc = request.app.state.services.chat
        try:
            turn = await svc.prepare(principal, body.conversation_id, body.message)
        except InvalidMessage as exc:
            raise HTTPException(422, str(exc)) from exc
        except ConversationNotFound as exc:
            raise HTTPException(404, "conversation not found") from exc

        async def events():
            # svc.stream is iterated inside this generator, so the per-thread lock and the MCP session are
            # entered and exited by the response task, as /v1/chat requires. A turn that streamed tokens also
            # ends with a "final" carrying the same text; only a token-less turn sends it, and a
            # recursion abort comes after the tokens as one more.
            streamed_tokens = False
            yield _sse("start", {"conversation_id": str(turn.conversation_id)})
            async for e in svc.stream(turn):
                match e.kind:
                    case "token":
                        streamed_tokens = True
                        yield _sse("token", {"text": str(e.data)})
                    case "final" if not streamed_tokens:
                        yield _sse("final", {"text": str(e.data)})
                    case "final" if e.data == UNFINISHED:
                        # The turn stopped after part of an answer: say so after it, not instead of it.
                        yield _sse("token", {"text": f"\n\n{UNFINISHED}"})
                    case "tool_call":
                        yield _sse("tool_call", {k: e.data.get(k) for k in TOOL_FIELDS})
                    case "tool_result":
                        # Never the content: it can be long, and it is the family's data, not the character's.
                        fields = {k: e.data.get(k) for k in TOOL_FIELDS}
                        yield _sse(
                            "tool_result",
                            fields | {"status": e.data["status"], "awaiting": e.data["awaiting"]},
                        )
                    case "notice":
                        yield _sse("notice", {"text": str(e.data), "code": NOTICE_CODES.get(str(e.data))})
                    case "error":
                        yield _sse("error", {"message": str(e.data)})
            yield _sse("done", {})

        return EventSourceResponse(events())

    @router.get("/conversations")
    async def conversations(request: Request, principal: Principal = Depends(require_principal)):
        return await request.app.state.services.chat.conversations(principal)

    @router.get("/conversations/{conversation_id}/messages")
    async def messages(
        conversation_id: uuid.UUID, request: Request, principal: Principal = Depends(require_principal)
    ):
        try:
            return await request.app.state.services.chat.history(principal, conversation_id)
        except ConversationNotFound as exc:
            raise HTTPException(404, "conversation not found") from exc

    return router
