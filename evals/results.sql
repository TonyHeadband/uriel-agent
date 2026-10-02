-- Eval and test results, for the Grafana dashboards. Applied by `python -m evals.publish --migrate` as the
-- database's owner; the writer only ever adds rows, so every change here must be additive.

CREATE TABLE IF NOT EXISTS runs (
    id          bigserial PRIMARY KEY,
    kind        text NOT NULL CHECK (kind IN ('eval', 'test')),
    suite       text NOT NULL,  -- eval: scripted | live; test: the -m expression it ran with
    started_at  timestamptz NOT NULL,
    finished_at timestamptz,
    source      text NOT NULL,  -- workstation | ci
    host        text,
    git_sha     text,
    branch      text,
    dirty       boolean,
    label       text NOT NULL DEFAULT '',  -- eval: the model ids
    tokens_in   bigint,
    tokens_out  bigint,
    cost_usd    numeric(10, 4),
    source_key  text UNIQUE  -- a run published twice (a re-run backfill) is kept once
);

CREATE TABLE IF NOT EXISTS eval_results (
    id            bigserial PRIMARY KEY,
    run_id        bigint NOT NULL REFERENCES runs (id),
    scenario      text NOT NULL,
    tags          text[] NOT NULL DEFAULT '{}',
    attempt       int NOT NULL,
    passed        boolean NOT NULL,
    -- null when the scenario has no check of that category
    route_ok      boolean,
    calls_ok      boolean,
    args_ok       boolean,
    reply_ok      boolean,
    state_ok      boolean,
    failures      text[] NOT NULL DEFAULT '{}',
    error         text,
    seconds_total real,
    seconds_p50   real,
    turns         jsonb NOT NULL  -- what was said, the calls made and the reply, per turn
);
CREATE INDEX IF NOT EXISTS eval_results_run ON eval_results (run_id);
CREATE INDEX IF NOT EXISTS eval_results_scenario ON eval_results (scenario);

CREATE TABLE IF NOT EXISTS test_results (
    id         bigserial PRIMARY KEY,
    run_id     bigint NOT NULL REFERENCES runs (id),
    nodeid     text NOT NULL,
    file       text NOT NULL,
    markers    text[] NOT NULL DEFAULT '{}',
    outcome    text NOT NULL CHECK (outcome IN ('passed', 'failed', 'skipped', 'error')),
    duration_s real,
    message    text
);
CREATE INDEX IF NOT EXISTS test_results_run ON test_results (run_id);
CREATE INDEX IF NOT EXISTS test_results_nodeid ON test_results (nodeid);
