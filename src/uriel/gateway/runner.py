"""Scheduled runs: claim due runs from uriel-tools, answer each as its owner, post it in their Talk DM."""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from uriel.agent.mcp_tools import McpToolbox, call_json
from uriel.gateway.chat import ChatService, InvalidMessage, collect
from uriel.gateway.directory import MEMBER_GROUPS, GroupDirectory, is_member
from uriel.gateway.talk import dm_thread
from uriel.gateway.talk_api import TalkClient, TalkError
from uriel.principal import INTERNAL_GROUP, Principal

log = logging.getLogger(__name__)
# Only the runner holds uriel-internal, the group uriel-tools gates claim_due_runs and finish_run to.
INTERNAL = Principal("uriel-gateway", frozenset({INTERNAL_GROUP}), "service")
MAX_RUNS_PER_TICK = 20  # a bound on one tick; what's left waits for the next
# Below uriel-tools' 10-minute lease, so a stuck run is finished here, and told to its owner, before the lease
# expires and uriel-tools fails it on its own.
RUN_TIMEOUT_S = 8 * 60
TIMED_OUT = "timed out after 8 minutes"
# finish_run's error is mirrored into the family's calendar; what went wrong in the gateway goes to the log.
GATEWAY_ERROR = "the gateway couldn't run this task"
SUMMARY_CHARS = 280  # the calendar mirror's done-task description
NOT_A_MEMBER = "owner is no longer a family member"
FAILED = 'Your scheduled task "{title}" failed. If it keeps happening, ask me to report it to Anthony.'


@dataclass(frozen=True)
class ClaimedRun:
    run_id: int
    owner: str
    owner_sub: str | None
    title: str
    prompt: str
    tools: frozenset[str]
    due_at: datetime
    tz: str

    @classmethod
    def from_tool(cls, data: dict[str, Any]) -> "ClaimedRun":
        return cls(
            int(data["run_id"]),
            data["owner"],
            data.get("owner_sub"),
            data["title"],
            data["prompt"],
            frozenset(data.get("tools") or []),
            datetime.fromisoformat(data["due_at"]),
            data.get("tz") or "UTC",
        )


def local_due(due_at: datetime, tz: str) -> str:
    """The due time on the owner's clock: the only time the gateway renders itself."""
    try:
        zone = ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        zone, tz = ZoneInfo("UTC"), "UTC"
    local = due_at.astimezone(zone)
    return f"{local:%a} {local.day} {local:%b %Y, %H:%M} ({tz})"


def run_input(run: ClaimedRun) -> str:
    return f'Scheduled task "{run.title}" ({local_due(run.due_at, run.tz)}): {run.prompt}'


@dataclass
class Delivery:
    """Where a run's answer goes, kept outside the run so a timeout can still tell the owner."""

    room: str | None = None


class ScheduledRunner:
    def __init__(
        self,
        *,
        toolbox: McpToolbox,
        chat: ChatService,
        directory: GroupDirectory,
        talk: TalkClient,
        seconds: float = 30.0,
        history_turns: int | None = None,
        member_groups: Collection[str] = MEMBER_GROUPS,
    ):
        self._toolbox, self._chat, self._directory, self._talk = toolbox, chat, directory, talk
        self._members = frozenset(member_groups)
        self._seconds, self._history = seconds, history_turns
        self._stopping = asyncio.Event()

    def stop(self) -> None:
        """Claim nothing more: the run in hand carries on, then run() returns."""
        self._stopping.set()

    async def run(self, sleep=None) -> None:
        sleep = sleep or self._pause
        while not self._stopping.is_set():
            try:
                await self.run_once()
            except Exception:
                log.warning("scheduled runs: couldn't claim due runs", exc_info=True)
            await sleep(self._seconds)

    async def _pause(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stopping.wait(), seconds)

    async def run_once(self) -> int:
        """Claim and run due runs one at a time until none is left; returns how many runs were finished."""
        claimed, handled, seen = 0, 0, set()
        # One claim per run: uriel-tools fails a run whose lease (10 minutes) expires, so a batch run in
        # sequence could have its later runs failed there while we still posted their answers.
        while claimed < MAX_RUNS_PER_TICK and not self._stopping.is_set():
            runs = ((await self._internal("claim_due_runs", {"limit": 1})) or {}).get("runs", [])
            if not runs:
                break
            for data in runs:
                # Malformed payloads count too, or a stream of them would keep the tick claiming.
                claimed += 1
                run_id = data.get("run_id")
                if run_id is not None and run_id in seen:
                    log.warning("scheduled run %s was handed out again; ending this tick", run_id)
                    return handled
                seen.add(run_id)
                handled += await self._handle(data)
        return handled

    async def _handle(self, data: dict[str, Any]) -> bool:
        """Run one claimed payload and report it; False when it was too malformed to name a run."""
        run_id = data.get("run_id")
        delivery = Delivery()
        try:
            run = ClaimedRun.from_tool(data)
            try:
                async with asyncio.timeout(RUN_TIMEOUT_S) as deadline:
                    outcome = await self._execute(run, delivery)
            except TimeoutError:
                if not deadline.expired():
                    raise
                log.warning("scheduled run %s timed out after %s s", run.run_id, RUN_TIMEOUT_S)
                outcome = await self._failed(run, delivery.room, TIMED_OUT)
        except Exception:
            log.warning("scheduled run %s failed in the gateway (payload: %r)", run_id, data, exc_info=True)
            outcome = {"status": "failed", "error": GATEWAY_ERROR}
        if run_id is None:
            log.warning("skipping a malformed claimed run without an id: %r", data)
            return False
        await self._finish(int(run_id), outcome)
        return True

    async def _execute(self, run: ClaimedRun, delivery: "Delivery") -> dict[str, Any]:
        # Looked up now, not from the cache: someone removed from family must stop getting their jobs run,
        # including a schedule with no tools for uriel-tools to refuse.
        groups = await self._directory.groups_of(run.owner, fresh=True)
        if not is_member(groups, self._members):
            log.warning(
                "scheduled run %s: %s isn't in lldap or in %s", run.run_id, run.owner, sorted(self._members)
            )
            return {"status": "failed", "error": NOT_A_MEMBER}
        principal = Principal(run.owner, groups, "human", run.owner_sub)
        room = delivery.room = await self._dm(run.owner)
        # The DM thread, so "tell me more about the third link" is a normal follow-up there.
        thread = dm_thread(run.owner, room) if room else f"{run.owner}:schedules"
        try:
            turn = self._chat.thread_turn(
                principal,
                thread,
                run_input(run),
                tools=run.tools,
                background=True,
                history_turns=self._history,
            )
        except InvalidMessage as exc:
            return await self._failed(run, room, str(exc))
        result = await collect(self._chat.stream(turn))
        if result.error:
            return await self._failed(run, room, result.error)
        summary = result.reply[:SUMMARY_CHARS]
        if room is None:
            return {"status": "undelivered", "summary": summary}
        try:
            ids = await self._talk.post_answer(room, result.message())
        except TalkError as exc:
            log.warning("scheduled run %s: couldn't post to Talk: %s", run.run_id, exc)
            return {"status": "undelivered", "summary": summary}
        return {"status": "ok", "summary": summary, "talk_message_id": ids[0] if ids else None}

    async def _dm(self, owner: str) -> str | None:
        try:
            return await self._talk.open_dm(owner)
        except TalkError as exc:
            log.warning("couldn't open %s's Talk DM: %s", owner, exc)
            return None

    async def _failed(self, run: ClaimedRun, room: str | None, error: str) -> dict[str, Any]:
        if room is not None:
            try:
                await self._talk.post(room, FAILED.format(title=run.title))
            except TalkError as exc:
                log.warning("scheduled run %s: couldn't tell the owner it failed: %s", run.run_id, exc)
        return {"status": "failed", "error": error}

    async def _finish(self, run_id: int, outcome: dict[str, Any]) -> None:
        try:
            result = await self._internal("finish_run", {"run_id": run_id, **outcome})
            if result and not result.get("finished"):
                log.warning("scheduled run %s wasn't finished: its lease had already expired", run_id)
        except Exception:
            log.warning(
                "finish_run failed for run %s; uriel-tools will expire its lease", run_id, exc_info=True
            )

    async def _internal(self, name: str, args: dict[str, Any]) -> Any:
        async with self._toolbox.open(INTERNAL, f"runner-{uuid.uuid4().hex[:12]}") as tools:
            tool = next((t for t in tools if t.name == name), None)
            if tool is None:
                raise LookupError(
                    f"uriel-tools doesn't offer {name}: it needs 0.8.0 and the {INTERNAL_GROUP} group"
                )
            return await call_json(tool, args)
