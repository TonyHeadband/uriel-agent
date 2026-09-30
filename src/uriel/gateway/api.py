import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from uriel.gateway.chat import ConversationNotFound, InvalidMessage
from uriel.principal import Principal


class ChatRequest(BaseModel):
    message: str
    conversation_id: uuid.UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    reply: str
    tools_used: list[str]
    notices: list[str]


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
