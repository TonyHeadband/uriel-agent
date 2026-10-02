import json
from dataclasses import asdict
from datetime import UTC, datetime

import psycopg
import pytest

from evals import publish
from evals.harness import Check, Run, TurnRun

pytestmark = pytest.mark.db
WHEN = datetime(2026, 10, 1, 21, 0, tzinfo=UTC)


def run(scenario, *checks, error=None):
    turn = TurnRun(say="hi", calls=[{"name": "remember", "args": {"file": "user"}}], reply="ok", seconds=2.0)
    turn.error = error
    return Run(scenario, [turn, TurnRun(say="and?", reply="yes", seconds=4.0)], list(checks), "qwen3:14b")


def rows(dsn, sql):
    with psycopg.connect(dsn) as conn:
        return conn.execute(sql).fetchall()


def on_the_workstation(monkeypatch):
    for name in ("CI", "GITHUB_ACTIONS", "GITHUB_SHA", "GITHUB_REF_NAME"):
        monkeypatch.delenv(name, raising=False)


async def test_an_eval_run_is_stored_once_with_a_flag_per_checked_category(pg_dsn, monkeypatch):
    on_the_workstation(monkeypatch)
    publish.migrate(pg_dsn)
    publish.migrate(pg_dsn)  # repeatable
    runs = [
        run("memory_fact", Check(0, "calls", True, ""), Check(0, "state", False, "show_memory lacks 'sir'")),
        run("memory_fact", Check(0, "calls", True, ""), Check(0, "state", True, "")),
        run("small_talk", Check(0, "route", False, "tools"), error="TimeoutError: slow"),
    ]
    info = publish.RunInfo(
        "eval", "live", WHEN, label="grp-on-14b", model="qwen3:14b", cost_usd=0.01234, source_key="eval:x"
    )
    first = publish.publish_evals(pg_dsn, runs, publish.here_and_now(info), {"memory_fact": ["memory"]})
    assert first is not None
    assert publish.publish_evals(pg_dsn, runs, info, {}) is None

    [(source, cost, branch, label, model)] = rows(
        pg_dsn, "SELECT source, cost_usd, branch, label, model FROM runs"
    )
    assert source == "workstation" and float(cost) == 0.0123 and branch
    assert (label, model) == ("grp-on-14b", "qwen3:14b")
    stored = rows(
        pg_dsn,
        "SELECT scenario, tags, attempt, passed, route_ok, calls_ok, state_ok, failures, error, seconds_p50,"
        " turns FROM eval_results ORDER BY id",
    )
    assert [r[:7] for r in stored] == [
        ("memory_fact", ["memory"], 1, False, None, True, False),
        ("memory_fact", ["memory"], 2, True, None, True, True),
        ("small_talk", [], 1, False, False, None, None),
    ]
    assert stored[0][7] == ["turn 1 state: show_memory lacks 'sir'"]
    assert stored[2][8] == "TimeoutError: slow" and stored[2][9] == 4.0
    assert stored[0][10][0]["calls"] == [{"name": "remember", "args": {"file": "user"}}]


async def test_test_outcomes_are_stored_with_their_file_and_markers(pg_dsn):
    publish.migrate(pg_dsn)
    tests = [
        publish.TestRow("tests/test_a.py::test_ok", "passed", [], 0.1),
        publish.TestRow(
            "tests/live/test_b.py::test_nc", "skipped", ["live"], 0.0, "URIEL_LIVE_NC_URL not set"
        ),
        publish.TestRow("tests/test_c.py::test_db", "failed", ["db"], 1.5, "assert 1 == 2"),
    ]
    info = publish.RunInfo("test", "not e2e and not live", WHEN, source="ci")
    assert publish.publish_tests(pg_dsn, tests, info)
    assert rows(pg_dsn, "SELECT file, markers, outcome, message FROM test_results ORDER BY id") == [
        ("tests/test_a.py", [], "passed", None),
        ("tests/live/test_b.py", ["live"], "skipped", "URIEL_LIVE_NC_URL not set"),
        ("tests/test_c.py", ["db"], "failed", "assert 1 == 2"),
    ]
    assert rows(pg_dsn, "SELECT kind, source FROM runs") == [("test", "ci")]


async def test_backfill_adds_each_saved_file_once(pg_dsn, tmp_path):
    publish.migrate(pg_dsn)
    saved = [asdict(run("live_memory_style", Check(0, "state", True, ""))) | {"passed": True}]
    (tmp_path / "20261001-230340.json").write_text(json.dumps(saved))
    publish.backfill(pg_dsn, list(tmp_path.glob("*.json")))
    publish.backfill(pg_dsn, list(tmp_path.glob("*.json")))
    [(suite, started, key, tags)] = rows(
        pg_dsn,
        "SELECT r.suite, r.started_at, r.source_key, e.tags"
        " FROM runs r JOIN eval_results e ON e.run_id = r.id",
    )
    assert (suite, key) == ("live", "eval:20261001-230340")
    assert started == datetime(2026, 10, 1, 23, 3, 40, tzinfo=UTC)
    assert tags == ["memory"]  # from the scenario file of that name


def test_a_ci_run_says_so_with_its_commit_and_branch(monkeypatch):
    on_the_workstation(monkeypatch)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_REF_NAME", "feature")
    info = publish.here_and_now(publish.RunInfo("test", "all", WHEN))
    assert (info.source, info.git_sha, info.branch) == ("ci", "abc123", "feature")


def test_nothing_is_published_without_the_url(monkeypatch):
    monkeypatch.delenv(publish.ENV, raising=False)
    with pytest.raises(SystemExit, match=publish.ENV):
        monkeypatch.setattr("sys.argv", ["publish"])
        publish.main()
