import os
import uuid

import psycopg
import pytest


@pytest.fixture
async def pg_dsn():
    """A throwaway database per test; skipped when no Postgres is configured."""
    admin = os.environ.get("URIEL_TEST_DATABASE_URL")
    if not admin:
        pytest.skip("URIEL_TEST_DATABASE_URL not set")
    name = f"uriel_test_{uuid.uuid4().hex[:12]}"
    async with await psycopg.AsyncConnection.connect(admin, autocommit=True) as conn:
        await conn.execute(f'CREATE DATABASE "{name}"')
    base, _, _ = admin.rpartition("/")
    yield f"{base}/{name}"
    async with await psycopg.AsyncConnection.connect(admin, autocommit=True) as conn:
        await conn.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
