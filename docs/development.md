# Developing Uriel

## Prerequisites
uv, Docker, and the LAN or VPN (dev models use Homelab's Ollama at `192.168.1.10:31434`).

## The tool server
The MCP tools live in a separate repo, `uriel-tools`, with its own releases. Its `docs/contract.md` is the whole
interface: `x-api-key`, and the caller's `user`, `groups` and `request_id` in MCP `_meta`. The unit tests here use
`tests/fake_mcp.py`, which follows that contract. The compose stack runs the uriel-tools image pinned in
`deploy/compose.yaml`, pulled from the Forgejo registry (`docker login git.example.com` once).
To run the stack against a local checkout instead, build it and override the image:
```bash
docker build -t uriel-tools:dev -f ../uriel-tools/deploy/Dockerfile ../uriel-tools
export URIEL_TOOLS_IMAGE=uriel-tools:dev   # applies to every compose command below
```

`config/models*.yaml` is the one models file for the whole deployment. The gateway reads the `interactive` and
`background` roles; uriel-tools reads `embedding` and `ocr` from the same file (mounted at `/app/shared`).

## Tests
```bash
uv sync
uv run pytest                       # unit and graph tests; Postgres tests skip without a database
docker run -d --name uriel-test-pg -e POSTGRES_PASSWORD=test -p 55432:5432 pgvector/pgvector:pg17
URIEL_TEST_DATABASE_URL=postgresql://postgres:test@localhost:55432/postgres uv run pytest -m "db or not db"
```

## Results in Grafana
Eval runs and pytest sessions can go to a results database (schema in `evals/results.sql`). Two Grafana dashboards
read it (`deploy/grafana/dashboards/`):
- **Uriel evals** shows:
  - pass rate per run and per area over time;
  - the latest result per scenario and model, with the change from the run before;
  - checks passed by category;
  - turn time;
  - every failure, with its conversation.
- **Uriel tests** shows:
  - outcomes per session;
  - failures, flaky tests and slow tests;
  - when each live or e2e test last actually ran, rather than being skipped.

Nothing is sent unless `URIEL_RESULTS_DATABASE_URL` is set. When it is set:
- `evals.run` publishes at the end of each run.
- The hook in `tests/conftest.py` publishes each pytest session.

Each run records where it ran (workstation or CI), the branch, the commit and whether the tree was dirty. A
publishing error only prints a warning: the eval run is still saved as JSON, and the tests still pass or fail
on their own.

**On Homelab:** https://grafana.example.com, folder "Uriel". This is the place to look.
- **Database:** `uriel-results-db` in the `monitoring` namespace (homelab-apps
  `services/monitoring/uriel-results/`), reachable on the LAN at `192.168.1.10:30543`, database `results`.
- **Roles:**
  - `results_writer` can only add rows. Its URL goes in `URIEL_RESULTS_DATABASE_URL`, in `.env` here and as the
    Forgejo CI secret of the same name.
  - `results` is the owner, used only for `--migrate`, through `URIEL_RESULTS_OWNER_URL`.
  - `grafana_reader` is Grafana's.
- **Passwords:** all three are in the sealed `uriel-results-secret`.
- **Changing a dashboard:** homelab-apps keeps a copy of `deploy/grafana/` in
  `services/monitoring/monitoring/grafana/uriel-configmaps.yaml`, so copy the change there too.

To view them on this machine instead (http://localhost:3000, no login):
```bash
docker compose -f deploy/compose.yaml --profile results up -d --wait results-db results-grafana
export URIEL_RESULTS_DATABASE_URL=postgresql://results:results@127.0.0.1:55444/results
uv run python -m evals.publish --migrate   # create the tables, as the owner; repeat after a schema change
uv run python -m evals.publish             # backfill evals/results/*.json; files already sent are skipped
```
- **Changing the schema:** keep every change additive. On Homelab the writer role can only add rows, so the
  owner runs `--migrate`.
- **Editing the dashboards:** change them in Grafana, then export the JSON over the file in
  `deploy/grafana/dashboards/`.

## End-to-end (stub LLM, no GPU)
```bash
compose="docker compose --env-file tests/e2e/e2e.env -f deploy/compose.yaml --profile e2e"
$compose up -d --build --wait
$compose exec -T lldap /app/bootstrap.sh   # seeds dad (admins, family) and kid (family)
URIEL_E2E_URL=http://localhost:8000 URIEL_E2E_TALK_URL=http://localhost:9100 \
  URIEL_E2E_TOOLS_DB_URL=postgresql://uriel_tools:uriel_tools@localhost:55433/uriel_tools \
  uv run pytest -m e2e
$compose down -v
```
The e2e stack also runs a fake Talk server (`tests/fake_talk.py`) and a test lldap. `tests/e2e/e2e.env` turns the Talk channel and the runner on against them.
Talk turns and scheduled runs are only for people in one of the lldap groups in `URIEL_MEMBER_GROUPS` (a JSON list,
default `["family"]`); anyone else in lldap is answered like an unknown sender.

## Tool-use evals (real model, scripted tools)
`evals/` measures how the agent picks tools and fills their arguments. It runs the gateway's own graph, prompt
and router against a real model.
- The tool names, descriptions and schemas are listed live from a uriel-tools server, so a docstring change there is
  measured.
- Every call is answered from the scenario's scripted results, so nothing is filed, edited or signed.
- A scenario (`evals/scenarios/*.yaml`) is a list of turns, with checks on the route, the tools called or avoided,
  their arguments, and the reply text. Its `tags` name the area it covers (`reporting`, `memory`, `routing`, `schedule`, `search`, `signing`).
- Treat evals like tests: while changing something, run only its area (`--tag`, or `--only` a scenario). Run the
  whole suite only as the release check, so a hosted model's cost grows with the change, not with the suite.

**Design work runs on qwen3:14b** on the workstation's RTX 4080 (`evals/models.4080-qwen3-14b.yaml`, the default).
It is the same family, template and params as Homelab's qwen3:8b, so reading the two side by side tells a design
problem (fails on both) from a model-size gap (fails only on the 8B). The workstation's ollama needs the full bundle
with its CUDA libraries, and `OLLAMA_CONTEXT_LENGTH=8192` to match Homelab. **Gemini** (`evals/models.gemini.yaml`,
Gemini 2.5 Flash) is opt-in, for a ceiling check: the scenarios are synthetic, so no family data leaves the house,
but it costs money. **Before a release, run the same suite on the local model** the deployment uses:
```bash
# any uriel-tools MCP reachable from here (the compose mcp service doesn't publish a port by default)
export URIEL_EVAL_MCP_URL=http://localhost:8011/mcp
uv run python -m evals.run --tag memory [--only report_declined]          # qwen3:14b on the workstation
GEMINI_API_KEY=… uv run python -m evals.run --tag memory --models evals/models.gemini.yaml [--budget-usd 0.50]
uv run python -m evals.run --n 3 --models config/models.dev.yaml   # release check: everything, on qwen
uv run python -m evals.report   # evals/results/report.html: pass rates over time, and every failing run's turns
```
- **Cost:**
  - A hosted model's price per million tokens goes in its models file under `eval_prices`, which the gateway
    ignores. Without a price, the runner refuses to use that model.
  - A run prints its tokens and cost, and stops once it reaches `--budget-usd` (default $0.50). A full suite on
    Gemini 2.5 Flash costs about $0.12.
  - It also stops after 3 rate-limited turns in a row, which usually means the quota is used up.
  - Keep a budget alert on the Google Cloud project as a backstop.
- **Keys:** the key comes from the environment (`api_key_env`), never from the file. Keep hosted models files in
  `evals/`, not in `config/`, which is mounted into the containers.
- **Model choice:** not `gemini-flash-latest` (3.x). It requires each tool call's `thought_signature` back, and
  LangChain's OpenAI client drops it, so every tool turn fails.
- **Labels:** each result records the model ids (`--label` overrides them), and the report shows them per run.

The gateway refuses to start if any model in its models file resolves to an address outside private networks,
or doesn't resolve at all. This covers every model in the file, including uriel-tools' embedding and OCR roles,
since the whole deployment shares that file. There is no switch to allow a hosted model in the gateway.

The runs happen one conversation at a time on purpose: parallel chats on one GPU slow each other down and skew
the results. Every run is saved as JSON in `evals/results/` (git-ignored); the report reads all of them.

### Live evals (real uriel-tools)
A scenario with `live: true` runs against a real uriel-tools instead of scripted results. That is the only way
to check that a turn really saved something, and that the next turn sees it through `memory_context`.
- **Isolation:** each run is a new throwaway user (`eval-<user>-<hex>`), so it starts with nothing remembered,
  runs repeat cleanly, and no real person's memory is touched.
- **Scenario format:**
  - `setup` lists tool calls that seed the throwaway user's state before turn 1.
  - `after` lists tool calls made once the turns are done. Each result's JSON is matched against its `expect`
    rule, and the outcome shows in the `state` column.
  - A live scenario can't have `tool_results` or `memory`.
- **Selection:** live scenarios run only with `--live` (the `--tag` and `--only` filters still apply), and
  never mix with scripted ones.
- **Coverage today:** memory only (`live_memory_*`), which needs just the tools database. Document search would
  need a dedicated Nextcloud account, so the local indexer never sees family folders.
- **Leftovers:** the notes stay in the stack's database. `compose down -v` clears them.
```bash
# a throwaway uriel-tools with its own database, published on :8011 (the override adds the port)
printf 'services:\n  mcp:\n    ports: ["8011:8001"]\n  db:\n    ports: !reset []\n' > /tmp/live.yaml
docker compose -p uriel-live-evals -f deploy/compose.yaml -f /tmp/live.yaml up -d --wait db mcp
uv run python -m evals.run --live [--n 3]
docker compose -p uriel-live-evals -f deploy/compose.yaml -f /tmp/live.yaml down -v
```

## Decision models on the house Ollama
The route decider (`tev1-4b` in the models files) asks `tev1-4b-cpu`, which is `tev1:4b-q4_K_M` pinned to the CPU.
- On the 8 GB GPU, tev1:4b would evict qwen3:8b, which then takes ~26 s to reload. `/v1/systemone` ignores a
  per-request `num_gpu`, so the pin lives in a Modelfile variant.
- On the CPU, qwen3:8b stays on the GPU, but a category decision takes ~4.6 s: its ~385-token prompt is evaluated
  afresh for each new message. Only an exact repeat, served from the prompt cache, answers in ~200 ms. See the
  spec's "Measured on Homelab"; the route decider stays on qwen3:8b until this is solved.
- `params.keep_alive` keeps it loaded between turns, since a cold load on the CPU takes ~6 s.

After an Ollama reinstall or a model update, recreate it (pulls the base model first):
```bash
scripts/create-decision-models.sh
uv run python -m evals.deciders.bench --models config/models.dev.yaml --model tev1-4b --point category \
  --set evals/deciders/route_holdout.jsonl --coverage 0.8 0.9 0.95 0.98 --n 3
```

## Local stack with the real model and browser login
1. Add `127.0.0.1 mock-oidc` to `/etc/hosts` (the browser and the gateway must see the same issuer host).
2. `docker compose -f deploy/compose.yaml --profile dev up -d --build --wait`
3. Open http://localhost:8000. On the mock login page, set the user name and paste claims such as
   `{"preferred_username": "dad", "groups": ["admins", "family"]}`.
4. Live checks: `URIEL_LIVE_URL=http://localhost:8000 uv run pytest -m live`.

## Local stack with the real Authelia login
The `uriel` client in Authelia also accepts `http://localhost:8000/auth/callback`, so a local gateway can log in
with your real account and groups (two-factor, like the other family services).
1. Put these in `.env` at the repo root (git-ignored; keep a copy of the client secret in Vaultwarden as `uriel: oidc-client-secret`):
   ```
   URIEL_OIDC_ISSUER=https://auth.example.com
   URIEL_OIDC_CLIENT_SECRET=<plaintext client secret>
   ```
   The compose file already allows the desktop companion's webview origins (`URIEL_CORS_ORIGINS`).
   `URIEL_OIDC_EXTRA_AUDIENCES` (a JSON list, empty by default) adds accepted JWT audiences, for when the
   companion's access tokens carry an audience other than the gateway's client id.
2. `docker compose --env-file .env -f deploy/compose.yaml up -d --build --wait` (no `dev` profile, since the mock issuer isn't needed).
   `--env-file` is required: compose otherwise looks for `.env` next to the compose file, in `deploy/`.
3. Open http://localhost:8000. You land on Authelia; the first login shows a one-time consent screen.

## Document search against your real Nextcloud
The `rag` profile adds uriel-tools' indexer (and its webhook intake, which Nextcloud can't reach on a laptop).
Folders are opted in from Nextcloud by sharing them with the `uriel` account; see uriel-tools' `docs/guide.md`.
The indexer lists those shares at startup and every 5 minutes, reads them over WebDAV, and re-checks them nightly.
Scans wait for the 01:00–06:00 OCR window.
1. Add `URIEL_NC_APP_PASSWORD=<app password>` to `.env`. It's in Vaultwarden, `Homelab` / "Nextcloud: uriel", in
   the "app password (uriel-tools indexer)" field.
2. `docker compose --env-file .env -f deploy/compose.yaml --profile rag up -d --build --wait`
3. Progress: `docker compose -f deploy/compose.yaml exec db psql -U uriel_tools -c "select status, count(*) from documents group by 1"`.

The tools database is created by `deploy/initdb/` on a fresh volume only; after pulling this change, recreate the
stack once with `down -v`. Sign out of Uriel and back in once, too: the gateway now forwards your OIDC subject
(`sub`), which is how Nextcloud accounts created by sociallogin (`authelia-<sub>`) are matched to you.

## Talk and scheduled runs
The gateway can answer in Nextcloud Talk as the `uriel` user and run the family's scheduled tasks
(docs/specs/2026-09-29-coworker-talk-scheduler-design.md). Both are off by default:
- `URIEL_TALK_ENABLED` polls Talk every `URIEL_TALK_POLL_SECONDS` (3).
  - A DM is always a turn. In a group room, only an @mention is.
  - The model sees the last `URIEL_TALK_HISTORY_TURNS` (20) turns of a room.
- `URIEL_RUNNER_ENABLED` claims due runs from uriel-tools every `URIEL_RUNNER_SECONDS` (30) and posts
  each answer in the owner's DM.
- Both need:
  - `URIEL_NC_URL` and `URIEL_NC_APP_PASSWORD` (the `uriel` app password, as for uriel-tools);
  - `URIEL_LDAP_URL`, `URIEL_LDAP_BIND_DN`, `URIEL_LDAP_PASSWORD` and `URIEL_LDAP_BASE_DN` (the read-only
    `uriel-gateway` bind user). Groups are read from lldap's `memberOf` and cached for 5 minutes.
- `URIEL_MEMBER_GROUPS` (a JSON list, default `["family"]`) names the lldap groups whose members may use Talk
  and scheduled runs. Anyone else is answered like an unknown sender.

**Only one gateway may poll a Nextcloud.** Leave both off in a local stack while the cluster's gateway
runs them, or every message is answered twice and every run can be claimed by either.

Deploy notes:
- Every person must be in the lowercase lldap group `family`, or they get the unknown-sender reply.
- Keep uriel-tools' `URIEL_MIRROR_ENABLED=false` in the cluster until uriel-tools' own Nextcloud live test passes.

Live checks, from a machine that can reach Homelab:
```bash
kubectl -n authelia port-forward svc/lldap 3890:3890 &   # lldap is ClusterIP-only
URIEL_LDAP_URL=ldap://127.0.0.1:3890 uv run --env-file .env pytest -m live tests/live/test_live_ldap.py
URIEL_LIVE_NC_URL=https://cloud.example.com uv run --env-file .env pytest -m live tests/live/test_live_talk.py
```
The DM round trip also needs `URIEL_LIVE_TALK_USER` and `URIEL_LIVE_TALK_PASSWORD`: an lldap test user in
`family` with a Nextcloud app password. It also needs a gateway with Talk enabled polling that Nextcloud.

## Observed latency (qwen3:8b on the shared 3070)
Single observed run against Homelab's Ollama (`192.168.1.10:31434`, qwen3:8b, `reasoning_effort: none`),
via `docker compose -f deploy/compose.yaml up -d --build --wait` and `uv run pytest -m live -v --durations=10`
on 2026-09-27: tool turn (`homelab_status` routed and executed) 5.00 s; direct turn (small talk, no tools)
0.76 s. The GPU is shared with other workloads, so treat these as one sample, not a stable baseline;
re-measure if the routing or chat model config changes.
