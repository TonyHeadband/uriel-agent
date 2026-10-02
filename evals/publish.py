"""Send eval and test results to the results database that Homelab's Grafana reads ("Uriel" folder).

    uv run python -m evals.publish --migrate        # create the tables, as the database's owner
    uv run python -m evals.publish                  # backfill every saved eval run (safe to repeat)

evals.run and the pytest hook in tests/conftest.py publish on their own when URIEL_RESULTS_DATABASE_URL is
set; without it nothing is sent. The URL is the results_writer role, which can add rows but not change them.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from evals.harness import CATEGORIES, Run, outcome

HERE = Path(__file__).parent
ENV = "URIEL_RESULTS_DATABASE_URL"


@dataclass
class RunInfo:
    kind: str  # eval | test
    suite: str
    started_at: datetime
    finished_at: datetime | None = None
    label: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    source_key: str | None = None
    # Where it ran; filled from the environment unless given (a backfill knows none of it).
    source: str | None = None
    host: str | None = None
    git_sha: str | None = None
    branch: str | None = None
    dirty: bool | None = None


@dataclass
class TestRow:
    nodeid: str
    outcome: str  # passed | failed | skipped | error
    markers: list[str]
    duration_s: float
    message: str | None = None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=HERE, capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def here_and_now(info: RunInfo) -> RunInfo:
    """Fill in where the run happened: CI (Forgejo Actions sets the GITHUB_* variables) or this machine."""
    ci = os.environ.get("GITHUB_ACTIONS") == "true" or bool(os.environ.get("CI"))
    info.source = info.source or ("ci" if ci else "workstation")
    info.host = info.host or socket.gethostname()
    info.git_sha = info.git_sha or os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    branch = os.environ.get("GITHUB_REF_NAME") or _git("rev-parse", "--abbrev-ref", "HEAD")
    info.branch = info.branch or branch
    if info.dirty is None and (status := _git("status", "--porcelain", "--untracked-files=no")) is not None:
        info.dirty = bool(status)
    return info


def migrate(url: str) -> None:
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute((HERE / "results.sql").read_text())


def _insert_run(conn: psycopg.Connection, info: RunInfo) -> int | None:
    """The new run's id; None when a run with this source_key is already there."""
    fields = asdict(info)
    if fields["cost_usd"] is not None:
        fields["cost_usd"] = Decimal(str(round(fields["cost_usd"], 4)))
    cols = ", ".join(fields)
    row = conn.execute(
        f"INSERT INTO runs ({cols}) VALUES ({', '.join(['%s'] * len(fields))}) "
        "ON CONFLICT (source_key) DO NOTHING RETURNING id",
        list(fields.values()),
    ).fetchone()
    return row[0] if row else None


def _seconds(run: Run) -> tuple[float, float | None]:
    secs = sorted(t.seconds for t in run.turns)
    return sum(secs), (secs[len(secs) // 2] if secs else None)


def publish_evals(url: str, runs: list[Run], info: RunInfo, tags: dict[str, list[str]]) -> int | None:
    """One runs row and one eval_results row per scenario run; the run's id, or None if already there."""
    attempts: Counter[str] = Counter()
    with psycopg.connect(url) as conn, conn.transaction():
        run_id = _insert_run(conn, info)
        if run_id is None:
            return None
        for run in runs:
            attempts[run.scenario] += 1
            total, p50 = _seconds(run)
            flags = [outcome(run, cat) for cat in CATEGORIES]
            conn.execute(
                "INSERT INTO eval_results (run_id, scenario, tags, attempt, passed, route_ok, calls_ok,"
                " args_ok, reply_ok, state_ok, failures, error, seconds_total, seconds_p50, turns)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    run_id,
                    run.scenario,
                    tags.get(run.scenario, []),
                    attempts[run.scenario],
                    run.passed,
                    *flags,
                    run.failures(),
                    next((t.error for t in run.turns if t.error), None),
                    total,
                    p50,
                    Jsonb([asdict(t) for t in run.turns]),
                ],
            )
    return run_id


def publish_tests(url: str, rows: list[TestRow], info: RunInfo) -> int | None:
    with psycopg.connect(url) as conn, conn.transaction():
        run_id = _insert_run(conn, info)
        if run_id is None:
            return None
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO test_results (run_id, nodeid, file, markers, outcome, duration_s, message)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [
                    (run_id, r.nodeid, r.nodeid.split("::")[0], r.markers, r.outcome, r.duration_s, r.message)
                    for r in rows
                ],
            )
    return run_id


def backfill(url: str, files: list[Path]) -> None:
    from evals.report import load_run
    from evals.run import load_scenarios

    tags = {s.name: s.tags for live in (False, True) for s in load_scenarios([], live=live)}
    for path in sorted(files):
        runs = [load_run(r) for r in json.loads(path.read_text())]
        if not runs:
            continue
        # Saved files are named for when the run finished, in UTC; they record nothing about the machine.
        when = datetime.strptime(path.stem, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
        suite = "live" if all(r.scenario.startswith("live_") for r in runs) else "scripted"
        info = RunInfo("eval", suite, when, when, label=runs[0].label, source_key=f"eval:{path.stem}")
        info.source, info.host = "workstation", socket.gethostname()
        print(f"{path.name}: {'added' if publish_evals(url, runs, info, tags) else 'already there'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="saved eval runs; default: all of evals/results")
    ap.add_argument("--url", default=os.environ.get(ENV), help=f"default: ${ENV}")
    ap.add_argument("--migrate", action="store_true", help="create the tables (needs the owner's URL)")
    args = ap.parse_args()
    if not args.url:
        sys.exit(f"set {ENV} or pass --url")
    if args.migrate:
        migrate(args.url)
        print("tables are up to date")
        return
    backfill(args.url, args.files or list((HERE / "results").glob("*.json")))


if __name__ == "__main__":
    main()
