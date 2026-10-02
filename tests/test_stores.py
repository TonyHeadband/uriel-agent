import uuid

import pytest

from uriel.agent.decider import Decision
from uriel.gateway.db import apply_migrations, make_pool
from uriel.gateway.stores import ConversationStore, DecisionLog, TalkCursors

pytestmark = pytest.mark.db


@pytest.fixture
async def pool(pg_dsn):
    p = make_pool(pg_dsn)
    await p.open()
    await apply_migrations(p)
    yield p
    await p.close()


async def test_migrations_are_idempotent(pool):
    assert await apply_migrations(pool) == []


async def test_decision_log_records_and_purges(pool):
    log = DecisionLog(pool)
    d = Decision("route", "tools", 0.9, "llm", "qwen3:8b", 120)
    await log.record(d, thread_id="dad:1", user_id="dad", context="homelab?", options=["direct", "tools"])
    async with pool.connection() as conn:
        await conn.execute("UPDATE decisions SET created_at = now() - interval '200 days'")
    await log.record(d, thread_id="dad:2", user_id="dad", context="new", options=["direct", "tools"])
    assert await log.purge_older_than(180) == 1
    async with pool.connection() as conn:
        row = await (await conn.execute("SELECT context, options, choice FROM decisions")).fetchone()
    assert row == {"context": "new", "options": ["direct", "tools"], "choice": "tools"}


async def test_decision_log_keeps_probabilities_and_pool(pool):
    d = Decision("category", "memory", 0.9, "systemone", "tev", 5, {"memory": 0.9, "none": 0.1})
    await DecisionLog(pool).record(
        d, thread_id="dad:1", user_id="dad", context="hi", options=["none", "memory"], pool=["memory"]
    )
    async with pool.connection() as conn:
        row = await (await conn.execute("SELECT probabilities, pool FROM decisions")).fetchone()
    assert row == {"probabilities": {"memory": 0.9, "none": 0.1}, "pool": ["memory"]}


async def test_conversation_is_owned_by_first_claimer(pool):
    store = ConversationStore(pool)
    cid = uuid.uuid4()
    assert await store.claim(cid, "dad", "homelab")
    assert await store.claim(cid, "dad", "ignored on update")
    assert not await store.claim(cid, "kid", "hijack")
    assert [c["title"] for c in await store.list_for("dad")] == ["homelab"]
    assert await store.list_for("kid") == []


async def test_migrations_handle_dollar_quoted_functions(pool, tmp_path):
    """Verify migrations work with plpgsql functions containing semicolons in $$ ... $$."""
    migration_file = tmp_path / "0002_test_function.sql"
    migration_file.write_text(
        """
CREATE FUNCTION test_fn() RETURNS text AS $$
BEGIN
  RETURN 'semicolon inside; this should work';
END;
$$ LANGUAGE plpgsql;

CREATE TABLE test_table (id serial PRIMARY KEY);
"""
    )
    applied = await apply_migrations(pool, tmp_path)
    assert "0002_test_function" in applied

    # Verify function was created
    async with pool.connection() as conn:
        result = await (await conn.execute("SELECT test_fn() AS result")).fetchone()
    assert result["result"] == "semicolon inside; this should work"

    # Verify table was created
    async with pool.connection() as conn:
        result = await (
            await conn.execute(
                "SELECT EXISTS(SELECT FROM information_schema.tables WHERE table_name = %s) AS exists",
                ("test_table",),
            )
        ).fetchone()
    assert result["exists"]


async def test_talk_cursors_only_move_forward(pool):
    cursors = TalkCursors(pool)
    assert await cursors.load() == {}
    await cursors.advance("dm-dad", 5)
    await cursors.advance("dm-dad", 3)
    await cursors.advance("family", 0)
    assert await cursors.load() == {"dm-dad": 5, "family": 0}
