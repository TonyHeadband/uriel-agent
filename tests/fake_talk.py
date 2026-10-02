"""A stand-in for Nextcloud Talk's OCS API (spreed 25), as the gateway uses it.

Only what the channel and the runner call: list rooms, read and post chat messages, reactions, and opening a
one-to-one room. Unit tests arrange and inspect `FakeTalk` directly through httpx's ASGI transport; the
compose e2e runs it as a container (`e2e_app`) and drives it through the `/_test` routes.
"""

import base64
import itertools
import os
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

OCS = "/ocs/v2.php/apps/spreed/api"
URIEL_MENTION = {"mention-user1": {"type": "user", "id": "uriel", "name": "Uriel"}}


def ocs(data: Any, status: int = 200) -> JSONResponse:
    meta = {"status": "ok", "statuscode": status, "message": "OK"}
    return JSONResponse({"ocs": {"meta": meta, "data": data}}, status_code=status)


@dataclass
class FakeTalk:
    user: str = "uriel"
    password: str = "talk-password"
    rooms: dict[str, dict] = field(default_factory=dict)
    messages: dict[str, list[dict]] = field(default_factory=dict)
    reactions: dict[tuple[str, int], set[str]] = field(default_factory=dict)
    fail: list[int] = field(default_factory=list)  # statuses for the next API requests, in order
    broken: dict[str, int] = field(default_factory=dict)  # "METHOD /path-prefix" -> status, until removed
    ids: Any = field(default_factory=lambda: itertools.count(1))

    def add_room(self, token: str, type_: int, *, name: str = "", display_name: str = "") -> str:
        self.rooms[token] = {
            "token": token,
            "type": type_,
            "name": name,
            "displayName": display_name or name or token,
        }
        self.messages.setdefault(token, [])
        return token

    def dm(self, uid: str) -> str:
        for room in self.rooms.values():
            if room["type"] == 1 and room["name"] == uid:
                return room["token"]
        return self.add_room(f"dm-{uid}", 1, name=uid)

    def say(
        self,
        token: str,
        actor: str,
        text: str,
        *,
        actor_type: str = "users",
        params: dict | None = None,
        system: str = "",
        message_type: str = "comment",
        timestamp: int | None = None,
        reply_to: int | None = None,
    ) -> int:
        message = {
            "id": next(self.ids),
            "token": token,
            "actorType": actor_type,
            "actorId": actor,
            "actorDisplayName": actor,
            "timestamp": int(time.time()) if timestamp is None else timestamp,
            "message": text,
            # Talk sends [] (a PHP empty array) when a message has no parameters.
            "messageParameters": params or [],
            "systemMessage": system,
            "messageType": "system" if system else message_type,
        }
        if reply_to:
            message["parent"] = {"id": reply_to}
        self.messages[token].append(message)
        return message["id"]

    def posted(self, token: str | None = None) -> list[dict]:
        return [
            {"token": t, "id": m["id"], "message": m["message"], "reply_to": m.get("parent", {}).get("id")}
            for t, msgs in self.messages.items()
            if token in (None, t)
            for m in msgs
            if m["actorId"] == self.user
        ]

    def _view(self, room: dict) -> dict:
        msgs = self.messages[room["token"]]
        return room | {"lastMessage": msgs[-1] if msgs else []}

    def _authorized(self, request: Request) -> bool:
        expected = "Basic " + base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
        return request.headers.get("authorization") == expected

    def app(self) -> FastAPI:
        api = FastAPI()

        @api.middleware("http")
        async def faults(request: Request, call_next):
            if request.url.path.startswith("/_test"):
                return await call_next(request)
            if self.fail:
                return Response(status_code=self.fail.pop(0))
            route = f"{request.method} {request.url.path}"
            for prefix, status in self.broken.items():
                if route.startswith(prefix):
                    return Response(status_code=status)
            if not self._authorized(request):
                return Response(status_code=401)
            return await call_next(request)

        @api.get(f"{OCS}/v4/room")
        async def list_rooms():
            return ocs([self._view(r) for r in self.rooms.values()])

        @api.post(f"{OCS}/v4/room")
        async def create_room(request: Request):
            uid = (await request.json())["invite"]
            existed = any(r["type"] == 1 and r["name"] == uid for r in self.rooms.values())
            return ocs(self._view(self.rooms[self.dm(uid)]), 200 if existed else 201)

        @api.get(f"{OCS}/v1/chat/{{token}}")
        async def read(token: str, lastKnownMessageId: int = 0, limit: int = 100):
            newer = [m for m in self.messages[token] if m["id"] > lastKnownMessageId][:limit]
            return ocs(newer) if newer else Response(status_code=304)

        @api.post(f"{OCS}/v1/chat/{{token}}")
        async def post(token: str, request: Request):
            body = await request.json()
            self.say(token, self.user, body["message"], reply_to=body.get("replyTo"))
            return ocs(self.messages[token][-1], 201)

        @api.post(f"{OCS}/v1/reaction/{{token}}/{{message_id}}")
        async def react(token: str, message_id: int, request: Request):
            self.reactions.setdefault((token, message_id), set()).add((await request.json())["reaction"])
            return ocs({}, 201)

        @api.delete(f"{OCS}/v1/reaction/{{token}}/{{message_id}}")
        async def unreact(token: str, message_id: int, reaction: str):
            self.reactions.setdefault((token, message_id), set()).discard(reaction)
            return ocs({})

        @api.get("/_test/rooms")
        async def test_rooms():
            return [{"token": r["token"], "type": r["type"], "name": r["name"]} for r in self.rooms.values()]

        @api.post("/_test/rooms")
        async def test_add_room(request: Request):
            body = await request.json()
            if body["type"] == 1:
                return {"token": self.dm(body["name"])}
            token = body.get("token") or f"room-{next(self.ids)}"
            return {"token": self.add_room(token, body["type"], name=body.get("name", ""))}

        @api.post("/_test/say")
        async def test_say(request: Request):
            body = await request.json()
            return {"id": self.say(body["token"], body["actor"], body["message"], params=body.get("params"))}

        @api.get("/_test/posted")
        async def test_posted():
            return self.posted()

        return api


def e2e_app() -> FastAPI:
    return FakeTalk(password=os.environ.get("FAKE_TALK_PASSWORD", "talk-password")).app()
