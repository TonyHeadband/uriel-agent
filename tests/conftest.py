import os
import uuid
from datetime import UTC, datetime

import psycopg
import pytest

# With URIEL_RESULTS_DATABASE_URL set, every session's outcomes go to the results database for Grafana
# (evals/publish.py). Publishing never fails the session: a down database only prints a warning.
_results: dict[str, dict] = {}
_session: dict = {}
KINDS = {"db", "e2e", "live"}  # the markers that say what a test needs, from pyproject.toml


def pytest_sessionstart(session):
    _session["started"] = datetime.now(UTC)


def pytest_runtest_logreport(report):
    row = _results.setdefault(
        report.nodeid,
        {
            "outcome": "passed",
            "duration": 0.0,
            "message": None,
            "markers": KINDS & set(report.keywords),
        },
    )
    row["duration"] += report.duration
    if report.when == "call":
        row["outcome"] = report.outcome
    elif report.skipped:
        row["outcome"] = "skipped"
    elif report.failed:  # a fixture failed, before or after the test
        row["outcome"] = "error"
    if report.skipped and isinstance(report.longrepr, tuple):
        row["message"] = report.longrepr[2]  # the skip reason
    elif report.failed:
        row["message"] = report.longreprtext[-4000:]


def pytest_sessionfinish(session, exitstatus):
    url = os.environ.get("URIEL_RESULTS_DATABASE_URL")
    if not url or not _results or session.config.getoption("collectonly"):
        return
    from evals import publish

    rows = [
        publish.TestRow(nodeid, r["outcome"], sorted(r["markers"]), r["duration"], r["message"])
        for nodeid, r in _results.items()
    ]
    info = publish.RunInfo("test", session.config.getoption("markexpr") or "all", _session["started"])
    info.finished_at = datetime.now(UTC)
    try:
        publish.publish_tests(url, rows, publish.here_and_now(info))
    except Exception as e:
        print(f"\ncould not publish test results ({type(e).__name__}: {e})")


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
