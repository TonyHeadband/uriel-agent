import uuid
from collections.abc import Sequence

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from uriel.agent.decider import Decision


class DecisionLog:
    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool

    async def record(
        self,
        d: Decision,
        *,
        thread_id: str,
        user_id: str,
        context: str,
        options: Sequence[str],
        pool: Sequence[str] | None = None,
    ) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO decisions (id, thread_id, user_id, point, context, options, choice, confidence,"
                " adapter, model, latency_ms, probabilities, pool)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    uuid.uuid4(),
                    thread_id,
                    user_id,
                    d.point,
                    context,
                    Jsonb(list(options)),
                    d.choice,
                    d.confidence,
                    d.adapter,
                    d.model,
                    d.latency_ms,
                    Jsonb(d.probabilities) if d.probabilities is not None else None,
                    Jsonb(list(pool)) if pool is not None else None,
                ),
            )

    async def purge_older_than(self, days: int) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM decisions WHERE created_at < now() - make_interval(days => %s)", (days,)
            )
            return cur.rowcount


class ConversationStore:
    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool

    async def claim(self, conversation_id: uuid.UUID, user_id: str, title: str) -> bool:
        """Create or touch a conversation. Returns False if another user owns it."""
        async with self._pool.connection() as conn:
            row = await (
                await conn.execute(
                    "INSERT INTO conversations (id, user_id, title) VALUES (%s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET updated_at = now() "
                    "WHERE conversations.user_id = EXCLUDED.user_id RETURNING id",
                    (conversation_id, user_id, title[:80]),
                )
            ).fetchone()
            return row is not None

    async def list_for(self, user_id: str, limit: int = 50) -> list[dict]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id, title, updated_at FROM conversations WHERE user_id = %s "
                "ORDER BY updated_at DESC LIMIT %s",
                (user_id, limit),
            )
            return await cur.fetchall()


class TalkCursors:
    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool

    async def load(self) -> dict[str, int]:
        async with self._pool.connection() as conn:
            rows = await (
                await conn.execute("SELECT room_token, last_message_id FROM talk_cursors")
            ).fetchall()
            return {r["room_token"]: r["last_message_id"] for r in rows}

    async def advance(self, token: str, message_id: int) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO talk_cursors (room_token, last_message_id) VALUES (%s, %s) "
                "ON CONFLICT (room_token) DO UPDATE SET updated_at = now(), "
                "last_message_id = GREATEST(talk_cursors.last_message_id, EXCLUDED.last_message_id)",
                (token, message_id),
            )
