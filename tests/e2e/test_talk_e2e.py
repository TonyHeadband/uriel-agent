import os
import time
import uuid

import httpx
import psycopg
import pytest

pytestmark = pytest.mark.e2e
TALK = os.environ.get("URIEL_E2E_TALK_URL", "")
TOOLS_DB = os.environ.get("URIEL_E2E_TOOLS_DB_URL", "")


@pytest.fixture(scope="module")
def talk():
    if not TALK:
        pytest.skip("URIEL_E2E_TALK_URL not set")
    with httpx.Client(base_url=TALK, timeout=10) as client:
        yield client


def wait_for(check, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if found := check():
            return found
        time.sleep(1)
    raise AssertionError("timed out waiting")


def uriel_said(talk, token, needle):
    return [m for m in talk.get("/_test/posted").json() if m["token"] == token and needle in m["message"]]


def test_a_dm_is_answered(talk):
    token = talk.post("/_test/rooms", json={"type": 1, "name": "kid"}).json()["token"]
    talk.post("/_test/say", json={"token": token, "actor": "kid", "message": "hello"})
    wait_for(lambda: uriel_said(talk, token, "Hello from the stub"))


def test_a_due_run_lands_in_the_owners_dm(talk):
    if not TOOLS_DB:
        pytest.skip("URIEL_E2E_TOOLS_DB_URL not set")
    # The stub LLM can't draft a schedule, so a due one goes straight into uriel-tools' table
    # (0.8.0's schedules migration). No tools: the check is the delivery path, not a search.
    with psycopg.connect(TOOLS_DB, autocommit=True) as conn:
        schedule_id = conn.execute(
            "INSERT INTO schedules "
            "(owner, title, prompt, rrule, dtstart, tz, tools, next_run_at, caldav_uid) "
            "VALUES ('dad', 'E2E hello', 'say hello', 'FREQ=DAILY;COUNT=1', "
            "(now() AT TIME ZONE 'America/Toronto')::timestamp, 'America/Toronto', '{}', "
            "now() - interval '5 seconds', %s) RETURNING id",
            (str(uuid.uuid4()),),
        ).fetchone()[0]

    def delivered():
        dm = next((r for r in talk.get("/_test/rooms").json() if r["type"] == 1 and r["name"] == "dad"), None)
        return dm and uriel_said(talk, dm["token"], "Hello from the stub")

    wait_for(delivered)
    with psycopg.connect(TOOLS_DB, autocommit=True) as conn:

        def finished():
            row = conn.execute(
                "SELECT status FROM schedule_runs WHERE schedule_id = %s", (schedule_id,)
            ).fetchone()
            return row if row and row[0] != "claimed" else None

        assert wait_for(finished, timeout=30)[0] == "ok"
