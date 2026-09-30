"""Deterministic stand-in for Ollama's OpenAI-compatible API, used only by the compose e2e test."""

import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
KEYWORDS = {"homelab": "homelab_status", "camera": "door_camera_last_event", "door": "door_camera_last_event"}


def last(messages, role):
    return next((m for m in reversed(messages) if m["role"] == role), None)


def text_of(message) -> str:
    content = message.get("content") or ""
    return content if isinstance(content, str) else " ".join(p.get("text", "") for p in content)


def plan(body: dict) -> tuple[str, list[dict]]:
    """Returns (content, tool_calls) for this request."""
    messages = body["messages"]
    user = text_of(last(messages, "user") or {}).lower()
    wanted = next((tool for word, tool in KEYWORDS.items() if word in user), None)
    if "response_format" in body:
        choice = "tools" if wanted else "direct"
        return json.dumps({"choice": choice, "confidence": 0.95}), []
    if messages[-1]["role"] == "tool":
        return f"Here is what I found: {text_of(messages[-1])[:200]}", []
    offered = {t["function"]["name"] for t in body.get("tools", [])}
    if wanted and wanted in offered:
        return "", [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": wanted, "arguments": "{}"},
            }
        ]
    if wanted:
        return "I can't access that for you.", []
    return "Hello from the stub.", []


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    content, tool_calls = plan(body)
    base = {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "created": int(time.time()), "model": body["model"]}
    finish = "tool_calls" if tool_calls else "stop"
    if not body.get("stream"):
        message = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return JSONResponse(
            base
            | {
                "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    def chunk(delta, finish_reason=None):
        return (
            "data: "
            + json.dumps(
                base
                | {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }
            )
            + "\n\n"
        )

    def events():
        yield chunk({"role": "assistant", "content": ""})
        if tool_calls:
            yield chunk({"tool_calls": [dict(tc, index=i) for i, tc in enumerate(tool_calls)]})
        else:
            for word in content.split(" "):
                yield chunk({"content": word + " "})
        yield chunk({}, finish)
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
