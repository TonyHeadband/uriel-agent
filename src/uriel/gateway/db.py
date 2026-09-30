from pathlib import Path

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

MIGRATIONS = Path(__file__).parent / "migrations"
_MIGRATION_LOCK = 7_272_741  # arbitrary advisory-lock id; serialises concurrent gateway starts


def make_pool(dsn: str, max_size: int = 10) -> AsyncConnectionPool:
    # autocommit is required: AsyncPostgresSaver.setup() runs CREATE INDEX CONCURRENTLY.
    return AsyncConnectionPool(
        dsn,
        max_size=max_size,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )


def _load_migrations(directory: Path) -> list[tuple[str, str]]:
    """Load migration files from disk: (stem, sql_text) tuples."""
    return [(p.stem, p.read_text()) for p in sorted(directory.glob("*.sql"))]


async def apply_migrations(pool: AsyncConnectionPool, directory: Path = MIGRATIONS) -> list[str]:
    applied: list[str] = []
    migration_files = _load_migrations(directory)
    async with pool.connection() as conn:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        await conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK,))
        try:
            rows = await (await conn.execute("SELECT version FROM schema_migrations")).fetchall()
            done = {r["version"] for r in rows}
            for stem, sql_text in migration_files:
                if stem in done:
                    continue
                async with conn.transaction():
                    # prepare=False uses the simple query protocol, allowing multiple statements
                    await conn.execute(sql_text, prepare=False)
                    await conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (stem,))
                applied.append(stem)
        finally:
            await conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK,))
    return applied
