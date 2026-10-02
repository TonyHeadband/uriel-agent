"""Nextcloud Talk (spreed) OCS API as the `uriel` user.

From the Talk API reference; verified live in Task 13.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import httpx

OCS = "/ocs/v2.php/apps/spreed/api"
TALK_USER = "uriel"
EYES = "👀"
# Talk accepts 32 000 characters; shorter messages read better on a phone.
MESSAGE_LIMIT = 4000


class TalkError(Exception):
    """Talk answered with something the gateway can't use."""


class TalkUnavailable(TalkError):
    """Nextcloud is unreachable or failing (5xx): back off and retry."""


class TalkUnauthorized(TalkError):
    """401: the uriel app password is wrong or revoked."""


@dataclass(frozen=True)
class Room:
    ONE_TO_ONE: ClassVar[int] = 1
    GROUP: ClassVar[int] = 2
    PUBLIC: ClassVar[int] = 3

    token: str
    type: int
    name: str  # for a one-to-one room, the other person's user id
    display_name: str
    last_message_id: int

    @classmethod
    def from_ocs(cls, data: dict[str, Any]) -> "Room":
        last = data.get("lastMessage")
        # An empty room's lastMessage is [] (a PHP empty array), not null.
        last_id = int(last["id"]) if isinstance(last, dict) and "id" in last else 0
        return cls(
            data["token"], int(data["type"]), data.get("name", ""), data.get("displayName", ""), last_id
        )


@dataclass(frozen=True)
class Message:
    id: int
    actor_type: str
    actor_id: str
    text: str
    parameters: dict[str, Any]
    system: bool
    message_type: str
    timestamp: int  # Unix seconds, so UTC

    @classmethod
    def from_ocs(cls, data: dict[str, Any]) -> "Message":
        params = data.get("messageParameters")
        return cls(
            int(data["id"]),
            data.get("actorType", ""),
            data.get("actorId", ""),
            data.get("message", ""),
            params if isinstance(params, dict) else {},
            bool(data.get("systemMessage")),
            data.get("messageType", "comment"),
            int(data.get("timestamp", 0)),
        )


def _pack(chunks: list[str], sep: str, limit: int) -> list[str]:
    out, current = [], ""
    for chunk in chunks:
        if current and len(current) + len(sep) + len(chunk) > limit:
            out.append(current)
            current = chunk
        else:
            current = f"{current}{sep}{chunk}" if current else chunk
    if current:
        out.append(current)
    return out


def split_message(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Messages of at most `limit` characters, cut between paragraphs, else between lines, else anywhere."""
    paragraphs = []
    for para in text.strip().split("\n\n"):
        if len(para) <= limit:
            paragraphs.append(para)
            continue
        lines = [line[i : i + limit] for line in para.split("\n") for i in range(0, max(len(line), 1), limit)]
        paragraphs.extend(_pack(lines, "\n", limit))
    return _pack(paragraphs, "\n\n", limit)


class TalkClient:
    def __init__(
        self,
        base_url: str,
        user: str,
        password: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 15.0,
    ):
        self.user = user
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            auth=(user, password),
            headers={"OCS-APIRequest": "true", "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, *, ok=(200, 201), **kwargs) -> httpx.Response:
        try:
            response = await self._http.request(method, f"{OCS}{path}", **kwargs)
        except httpx.TransportError as exc:
            raise TalkUnavailable(f"{method} {path}: {exc}") from exc
        if response.status_code == 401:
            raise TalkUnauthorized("Nextcloud refused the uriel app password")
        # 429 is Nextcloud's brute-force throttling: back off like an outage.
        if response.status_code == 429 or response.status_code >= 500:
            raise TalkUnavailable(f"{method} {path}: {response.status_code}")
        if response.status_code not in ok:
            # Not the body: a 4xx can echo the message the family sent, and this ends up in the logs.
            raise TalkError(f"{method} {path}: {response.status_code}{TalkClient._ocs_code(response)}")
        return response

    @staticmethod
    def _ocs_code(response: httpx.Response) -> str:
        try:
            return f" (OCS {int(response.json()['ocs']['meta']['statuscode'])})"
        except (ValueError, KeyError, TypeError):
            return ""

    @staticmethod
    def _data(response: httpx.Response) -> Any:
        # A proxy or SSO login page can answer 200 with something that isn't OCS: treat it as an outage.
        try:
            ocs = response.json()["ocs"]
            data, code = ocs["data"], ocs.get("meta", {}).get("statuscode", 200)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise TalkUnavailable(f"Talk answered {response.status_code} without an OCS payload") from exc
        if code not in (100, 200, 201):
            raise TalkError(f"Talk OCS status {code}")
        return data

    @staticmethod
    def _parse(build, items: list[Any]) -> list[Any]:
        try:
            return [build(item) for item in items]
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise TalkUnavailable("Talk answered with an unexpected payload shape") from exc

    async def rooms(self) -> list[Room]:
        response = await self._request("GET", "/v4/room", params={"noStatusUpdate": 1})
        return self._parse(Room.from_ocs, self._data(response))

    async def messages_after(self, token: str, last_id: int, limit: int = 100) -> list[Message]:
        params = {"lookIntoFuture": 1, "lastKnownMessageId": last_id, "limit": limit, "timeout": 0}
        response = await self._request("GET", f"/v1/chat/{token}", ok=(200, 304), params=params)
        if response.status_code == 304:
            return []
        return self._parse(Message.from_ocs, self._data(response))

    async def post(self, token: str, text: str, reply_to: int | None = None) -> int:
        body: dict[str, Any] = {"message": text}
        if reply_to:
            body["replyTo"] = reply_to
        response = await self._request("POST", f"/v1/chat/{token}", json=body)
        return self._parse(lambda data: int(data["id"]), [self._data(response)])[0]

    async def post_answer(self, token: str, text: str, reply_to: int | None = None) -> list[int]:
        ids = []
        for i, part in enumerate(split_message(text)):
            ids.append(await self.post(token, part, reply_to if i == 0 else None))
        return ids

    async def react(self, token: str, message_id: int, emoji: str = EYES) -> None:
        await self._request("POST", f"/v1/reaction/{token}/{message_id}", json={"reaction": emoji})

    async def unreact(self, token: str, message_id: int, emoji: str = EYES) -> None:
        await self._request("DELETE", f"/v1/reaction/{token}/{message_id}", params={"reaction": emoji})

    async def open_dm(self, uid: str) -> str:
        # Talk returns the existing one-to-one room (200) rather than a second one (201).
        response = await self._request("POST", "/v4/room", json={"roomType": 1, "invite": uid})
        return self._parse(lambda data: data["token"], [self._data(response)])[0]
