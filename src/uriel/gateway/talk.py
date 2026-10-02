"""Uriel's Talk channel: polls Nextcloud Talk as the `uriel` user and answers through ChatService.

Single gateway replica: the cursors and ChatService's per-thread lock assume one poller.
"""

import asyncio
import logging
import re
import time
from collections.abc import Callable, Collection
from datetime import UTC, datetime

from uriel.gateway.chat import ChatService, InvalidMessage, collect
from uriel.gateway.directory import MEMBER_GROUPS, GroupDirectory, GroupLookupError, is_member
from uriel.gateway.talk_api import EYES, Message, Room, TalkClient, TalkError, TalkUnauthorized
from uriel.principal import Principal

log = logging.getLogger(__name__)
CHAT_ROOMS = {Room.ONE_TO_ONE, Room.GROUP, Room.PUBLIC}  # not changelog or note-to-self rooms
MAX_BACKOFF_S = 60.0
# Repeated 401s would trip Nextcloud's brute-force protection and lock the uriel account out.
UNAUTHORIZED_BACKOFF_S = 300.0
UNKNOWN_SENDER = "I only work for family members signed in through Homelab"
TURN_FAILED = (
    "Sorry, something went wrong on my side and I couldn't answer. If it keeps happening, ask me to report "
    "it to Anthony."
)
EMPTY_MENTION = "Yes? What can I do for you?"
GROUP_ROOMS = {Room.GROUP, Room.PUBLIC}


def next_delay(poll_s: float, previous_s: float, error: Exception | None) -> float:
    if error is None:
        return poll_s
    if isinstance(error, TalkUnauthorized):
        return UNAUTHORIZED_BACKOFF_S
    return min(max(previous_s * 2, poll_s * 2), MAX_BACKOFF_S)


def plain_text(message: Message, user: str) -> str:
    """The text with Uriel's own mention removed and other placeholders ({mention-user1}, {file})
    spelled out."""
    text = message.text
    for key, param in message.parameters.items():
        if not isinstance(param, dict):
            continue
        if param.get("type") == "user" and param.get("id") == user:
            value = ""
        else:
            value = f"{'@' if key.startswith('mention-') else ''}{param.get('name') or param.get('id', '')}"
        text = text.replace(f"{{{key}}}", value)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def mentions(message: Message, user: str) -> bool:
    """Whether the message @mentions `user` itself; @all is a `call` mention and doesn't count."""
    return any(
        isinstance(p, dict) and p.get("type") == "user" and p.get("id") == user
        for p in message.parameters.values()
    )


def audience_note(room: Room) -> str:
    return (
        f'You are answering in the shared Talk room "{room.display_name}", where everyone in it can read '
        "your answer: don't bring up this person's private documents, notes or what you know about them "
        "unless they ask for it here."
    )


def dm_thread(user_id: str, token: str) -> str:
    return f"{user_id}:talk-{token}"


def room_thread(token: str) -> str:
    return f"room:talk-{token}"


class TalkChannel:
    def __init__(
        self,
        *,
        client: TalkClient,
        chat: ChatService,
        directory: GroupDirectory,
        cursors,
        poll_seconds: float = 3.0,
        history_turns: int = 20,
        member_groups: Collection[str] = MEMBER_GROUPS,
        clock: Callable[[], float] = time.time,
    ):
        self._client, self._chat, self._directory, self._cursors = client, chat, directory, cursors
        self._members = frozenset(member_groups)
        self._poll, self._history, self._clock = poll_seconds, history_turns, clock
        self._known: dict[str, int] | None = None
        # A room first seen after startup (a DM someone just opened) is read from its start, or its first
        # message could become the cursor unanswered; only messages older than the gateway count as history.
        self._since: dict[str, float] = {}
        self._busy: dict[str, asyncio.Task] = {}
        self._refused: set[tuple[str, str]] = set()
        self._started = clock()
        self._closing = False

    async def run(self, sleep=asyncio.sleep) -> None:
        delay = self._poll
        while True:
            error = None
            try:
                await self.poll_once()
            except TalkUnauthorized as exc:
                log.error("Nextcloud refused uriel's app password; polling Talk again in 5 minutes: %s", exc)
                error = exc
            except Exception as exc:
                log.warning("Talk poll failed; backing off", exc_info=True)
                error = exc
            delay = next_delay(self._poll, delay, error)
            await sleep(delay)

    async def poll_once(self) -> None:
        rooms = await self._client.rooms()
        first = self._known is None
        if first:
            self._known = await self._cursors.load()
        for room in rooms:
            if room.type not in CHAT_ROOMS:
                continue
            if room.token not in self._known:
                if first:
                    # Uriel doesn't answer history: a room that exists at startup starts at its latest
                    # message.
                    await self._advance(room.token, room.last_message_id)
                    continue
                self._known[room.token] = 0
                self._since[room.token] = self._started
            if room.last_message_id > self._known[room.token] and room.token not in self._busy:
                task = asyncio.create_task(self._drain(room))
                self._busy[room.token] = task
                # Not in _drain's finally: a task cancelled before it starts never runs that, and drain()
                # would wait on it forever.
                task.add_done_callback(lambda _t, token=room.token: self._busy.pop(token, None))

    async def drain(self) -> None:
        """Wait until the rooms being handled are done."""
        while self._busy:
            await asyncio.gather(*self._busy.values(), return_exceptions=True)

    def close(self) -> None:
        """Take no more messages: a room being handled stops after the message in hand."""
        self._closing = True

    def stop(self) -> None:
        for task in self._busy.values():
            task.cancel()

    async def _drain(self, room: Room) -> None:
        try:
            for message in await self._client.messages_after(room.token, self._known[room.token]):
                if self._closing:
                    return
                await self._handle(room, message)
                await self._advance(room.token, message.id)
        except TalkError as exc:
            log.warning("Talk room %s: %s; retrying at the next poll", room.token, exc)
        except GroupLookupError as exc:
            log.warning("Talk room %s: %s; retrying at the next poll", room.token, exc)
        except Exception:
            log.exception(
                "Talk room %s: a message couldn't be handled; retrying at the next poll", room.token
            )

    async def _advance(self, token: str, message_id: int) -> None:
        await self._cursors.advance(token, message_id)
        self._known[token] = max(self._known.get(token, 0), message_id)

    def _from_a_person(self, room: Room, message: Message) -> bool:
        return (
            message.message_type == "comment"
            and not message.system
            and message.actor_type == "users"
            and message.actor_id != self._client.user
            and message.timestamp >= self._since.get(room.token, 0)
        )

    async def _handle(self, room: Room, message: Message) -> None:
        if not self._from_a_person(room, message):
            return
        shared = room.type in GROUP_ROOMS
        if shared and not mentions(message, self._client.user):
            return
        reply_to = message.id if shared else None
        principal = await self._principal(room, message, reply_to)
        if principal is None:
            return
        text = plain_text(message, self._client.user)
        thread = room_thread(room.token) if shared else dm_thread(principal.user_id, room.token)
        try:
            turn = self._chat.thread_turn(
                principal,
                thread,
                text,
                personal=not shared,
                shared=shared,
                note=audience_note(room) if shared else "",
                history_turns=self._history,
            )
        except InvalidMessage as exc:
            await self._reply(
                room.token, f"I couldn't take that: {exc}." if text else EMPTY_MENTION, reply_to
            )
            return
        await self._quietly(self._client.react(room.token, message.id, EYES))
        result = await collect(self._chat.stream(turn))
        if result.error:
            await self._quietly(self._client.unreact(room.token, message.id, EYES))
            await self._reply(room.token, TURN_FAILED, reply_to)
            return
        await self._reply(room.token, result.message(), reply_to)

    @staticmethod
    async def _quietly(call) -> None:
        try:
            await call
        except TalkError as exc:
            log.warning("Talk reaction failed: %s", exc)

    async def _principal(self, room: Room, message: Message, reply_to: int | None) -> Principal | None:
        groups = await self._directory.groups_of(message.actor_id)
        if not is_member(groups, self._members):
            await self._refuse(room, reply_to)
            return None
        return Principal(message.actor_id, groups, "human")

    async def _refuse(self, room: Room, reply_to: int | None) -> None:
        day = datetime.fromtimestamp(self._clock(), UTC).date().isoformat()
        if (room.token, day) in self._refused:
            return
        self._refused.add((room.token, day))
        await self._reply(room.token, UNKNOWN_SENDER, reply_to)

    async def _reply(self, token: str, text: str, reply_to: int | None) -> None:
        try:
            await self._client.post_answer(token, text, reply_to)
        except TalkError as exc:
            # The turn already ran and may have changed something, so it isn't run again: the cursor moves
            # on and the answer stays in the thread.
            log.warning("could not post to Talk room %s: %s", token, exc)
