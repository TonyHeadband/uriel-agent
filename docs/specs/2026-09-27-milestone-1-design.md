# Uriel — Milestone 1 design: agent harness walking skeleton

Status: draft, awaiting review · Date: 2026-09-27 · Brief: [docs/idea.md](../idea.md)

## Goal

Uriel is a fully local family assistant. Milestone 1 builds a harness that runs from end to end on Homelab: a family member logs in through Authelia and chats on a minimal web page. Their message then goes through a LangGraph agent loop that makes at least one group-gated MCP tool call. History survives restarts and belongs to that user.

Success means:

- a logged-in `admins` user asks about the homelab and gets an answer backed by `homelab_status`;
- a `family`-only user is never offered that tool, and calling it directly is refused;
- both work on the cluster (Argo-deployed) and in local dev (compose).

## Context: where this runs

- **Homelab** is a single-node k3s cluster with an RTX 3070 (8 GB). The GPU is time-sliced between Ollama, Frigate, immich-ml and Jellyfin, so VRAM is contended.
- **Ollama** already runs in the `ai` namespace (`ollama.ai.svc.cluster.local:11434`, NodePort `31434`), with `qwen3:8b` and `nomic-embed-text` pulled.
- **GitOps:** Argo CD app-of-apps from `anthony-headband/homelab-apps` (`apps/<name>.yaml` → `services/<ns>/<name>/`). `epp` is the template for custom apps.
- **Images** are built by the reusable `ci-workflows/.forgejo/workflows/docker-publish.yaml` on a tag push and stored in the `git.example.com` registry.
- **Network trust:** the network is reachable only over VPN, and the VPN is shared by family **and friends**. Authelia accounts are family-only. Authelia's `access_control` bypasses auth for `192.168.0.0/16`, so network location is never a basis for authorization.

## Decisions

| # | Decision | Rejected alternatives |
|---|---|---|
| D1 | Human identity via **Authelia OIDC**; the gateway validates JWTs itself | Forward-auth headers (bypassed on LAN; forgeable by any pod); oauth2-proxy (same header-trust flaw) |
| D2 | Services and agents authenticate with **API keys** mapped to fixed principals | OIDC for machines (no browser; device flow unverified) |
| D3 | Gateway→MCP trust via **shared API key + NetworkPolicy** | Forwarding the user's JWT to MCP (a second mechanism, useless for services); NetworkPolicy alone |
| D4 | **Gateway and agent in one service**; MCP separate | Three services; LangGraph Server (Redis, licensing, heavier) |
| D5 | **Minimal htmx page** served by the gateway | Open WebUI front end (own history, no groups); API only |
| D6 | **RAG stubbed**; the real pipeline is a later spec | Real or minimal RAG now |
| D7 | **`Decider` interface** with an LLM-backed adapter; a Jev-style decision model in spec 2 | Deploying a decision model now |
| D8 | **Argo/epp deployment pattern**; image digest bumped **manually** after each release | CI commits to GitOps; Argo Image Updater; round-house `kubectl` rollout (fights selfHeal) |

## Architecture

```
 browser ──OIDC──► Authelia
    │ session cookie
    ▼
┌──────────── uriel-gateway (FastAPI) ─────────────┐
│ auth/     OIDC code+PKCE → session; Bearer JWT   │
│           or X-API-Key → Principal(user, groups) │
│ web/      htmx page: GET / , POST /chat (SSE)    │
│ api/      POST /v1/chat  (JSON, for agents)      │
│ agent/    LangGraph graph, no FastAPI imports    │
│   route ─► Decider ─► respond | agent⇄tools      │
└───────┬───────────────────────┬──────────────────┘
        │ X-API-Key + {user,groups}        │ checkpoints, decisions
        ▼                                  ▼
  uriel-mcp (FastMCP)               uriel-db (Postgres + pgvector)
  gate(tool, groups) → run | ToolException
        │
  Ollama (ai ns, cluster DNS)
```

The repo is a `uv` workspace with one package per service (`gateway`, `mcp`) plus shared config models. `agent/` must not import FastAPI, so it could be split into its own service later.

## Identity

Every request resolves to a single `Principal(user_id, groups, kind: human | service)` before it touches the agent, or gets a 401.

- **Browser.** OIDC authorization code with PKCE against Authelia, on client `uriel` (confidential, scopes `openid profile email groups`, `authorization_policy: one_factor`). The gateway validates the ID token (signature against Authelia's JWKS, `iss`, `aud`, `exp`) and stores the principal in a signed, `HttpOnly`, `Secure` session cookie.
- **API bearer.** `Authorization: Bearer <JWT>` is validated the same way. It's the path for scripted tests and future clients.
- **Service key.** `X-API-Key` is looked up in a keys map (a sealed secret). Each key maps to a fixed principal, for example `satellite-kitchen → {user: kitchen, groups: [family]}`. Keys are compared in constant time.
- **No bypass mode exists.** Dev uses a mock OIDC issuer; only the issuer URL differs between environments.

**Threads.** `thread_id = "{user_id}:{conversation_id}"`. The gateway refuses any thread whose prefix isn't the caller's `user_id`.

## Agent graph

State: `messages`, `principal`, `route`.

1. **`route`**: calls `Decider.choose(point="route", context=<last user message>, options=["direct","tools"])`.
2. **`respond`** (`direct`): the chat model answers without tools.
3. **`agent`** (`tools`): the chat model with the principal's MCP tools bound, looping with a `ToolNode` until there are no tool calls. `recursion_limit` defaults to 10.

Persistence: `AsyncPostgresSaver` over a psycopg3 async pool (autocommit, `dict_row`). `setup()` runs at startup.

### Decider

```python
class Decider(Protocol):
    async def choose(self, point: str, context: str, options: list[str]) -> Decision: ...
    # Decision(choice, confidence, adapter, model, latency_ms)
```

- **M1 adapter, `LLMDecider`:** the chat model with structured output constrained to `options`. Confidence comes from the model's self-report and is treated as advisory.
- **Fallback:** on an error, or confidence below the threshold set in config, the decision is `tools`, which is always safe. It's logged with `adapter=fallback`.
- **Log:** every decision is written to the `decisions` table. That data is used for evaluation and fine-tuning in spec 2.
- **Spec 2:** a Jev-style decider. The preferred candidate is the OpenJev/SemIf logit-reading approach (no training needed); the alternative is a `/v1/systemone` adapter (Laya, Kev). The graph doesn't change; only the adapter is selected in config.

## MCP server (`uriel-mcp`)

- **Transport:** FastMCP over streamable HTTP at `/mcp`.
- **Caller authentication:** middleware requires a valid `X-API-Key` from a list, so keys can be rotated without downtime. Otherwise it returns 401 before any MCP handling.
- **Identity:** the gateway sends `{user, groups}` with each call, taken from graph state, never from model output. The brief's mechanism is request `meta` (`ctx.request_context.meta`). If `langchain-mcp-adapters` can't set per-call meta, the fallback is per-request headers `X-Uriel-User` / `X-Uriel-Groups`. Either is trusted only because the API key is valid.
- **Gating**, driven by config and denying by default:

  ```yaml
  # tools.yaml
  tools:
    homelab_status:         { groups: [admins] }
    search_documents:       { groups: [family] }
    door_camera_last_event: { groups: [family] }
  ```

  1. The tool list is filtered for each principal, so the model only sees tools the user may call.
  2. Every call runs a `@gated` check and raises `ToolException("not permitted for <user>")` on denial. A tool missing from config is denied to everyone.
- **Stub tools** have typed arguments and realistic canned output:
  - `homelab_status()`: node and pod summary;
  - `search_documents(query, k=3)`: hits with `source` and `snippet`;
  - `door_camera_last_event()`: a Frigate-shaped event.
- **Audit:** one JSON log line per call, `{request_id, user, tool, allowed, duration_ms}`, collected by the existing Alloy→Loki pipeline.

## Data

**Postgres** (`pgvector/pgvector:pg17`): database `uriel`, with Deployment `uriel-db` (`Recreate`, local-path PVC).

- The LangGraph checkpoint tables are owned by the library.
- Our tables are managed by numbered SQL files in `migrations/`, applied at startup and tracked in `schema_migrations`.

```sql
CREATE TABLE decisions (
  id               uuid PRIMARY KEY,
  created_at       timestamptz NOT NULL DEFAULT now(),
  thread_id        text NOT NULL,
  user_id          text NOT NULL,
  point            text NOT NULL,
  context          text NOT NULL,
  options          jsonb NOT NULL,
  choice           text NOT NULL,
  confidence       real,
  adapter          text NOT NULL,
  model            text,
  latency_ms       integer,
  corrected_choice text
);
```

- **Retention:** rows older than `decisions.retention_days` (default 180) are deleted at gateway startup and daily after that.
- **Backups:** add `try pgdump uriel uriel-db uriel` to `backup-restic/prep-cronjob.yaml`.

## Model routing

`config/models.yaml` is mounted from a ConfigMap and validated at startup (pydantic). An unknown role or model reference fails startup.

```yaml
models:
  qwen3-8b:
    provider: openai_compat
    base_url: http://ollama.ai.svc.cluster.local:11434/v1
    model: qwen3:8b
    params: { temperature: 0.3, reasoning: false }
roles:
  interactive: qwen3-8b
  background:  qwen3-8b       # defined; unused in M1
deciders:
  route: { adapter: llm, model: qwen3-8b, min_confidence: 0.6 }
```

- Chat models go through `ChatOpenAI(base_url=...)`, so vLLM or llama.cpp can be swapped in later through config alone. Moving to Qwen3.6-35B-A3B means adding one model entry and changing `roles.interactive`.
- Ollama's context length and keep-alive are server settings on the **shared** `ollama` Deployment. M1 measures first; any change to the Ollama manifest is proposed separately and not bundled into M1.

## Web UI

The gateway serves one page with htmx and SSE. It's deliberately plain; the real UI is a later spec.

- `GET /` redirects to login when unauthenticated. Otherwise it shows the conversation list and the current thread.
- `POST /chat` streams tokens over SSE and shows tool calls inline as simple status lines (for example "used homelab_status").
- `GET /auth/login`, `GET /auth/callback` and `POST /auth/logout` handle the session.

## Deployment

**This repo** contains `gateway/`, `migrations/`, `config/`, `deploy/compose.yaml`, `scripts/smoke.sh`, and `.forgejo/workflows/{ci,release}.yaml`.

**`homelab-apps`** gets `apps/uriel.yaml` → `services/uriel/uriel/`, with Argo automated prune and selfHeal:

```
namespace.yaml
registry-pull-sealedsecret.yaml
secrets-sealedsecret.yaml   # OIDC client secret, session key, MCP API keys, PG password, service keys map
gateway/  deployment, service, configmap (models.yaml), ingressroute
mcp/      deployment, service, configmap (tools.yaml)
postgres/ deployment, service, pvc
networkpolicies.yaml
```

- **Gateway:** `/health` for readiness (DB pool up, migrations applied; **not** MCP) and `/livez` for liveness. Its IngressRoute serves `uriel.example.com` through `chain-local` with `certResolver: dns-cloudflare`. There's no forward-auth middleware.
- **MCP and Postgres** get ClusterIP Services only.
- **NetworkPolicies** use default-deny for ingress in the namespace. The gateway accepts traffic from Traefik only, MCP from the gateway only, and Postgres from the gateway only (`backup-prep` uses `kubectl exec`; verify). Egress stays open in M1.
- **Authelia:** add client `uriel` to the Authelia `values.yaml` with a pbkdf2-sha512 digest and redirect URI `https://uriel.example.com/auth/callback`. The plaintext secret goes to the uriel SealedSecret and the vault.
- **Images and releases:** images are pinned by digest. After each `*.*.*` tag, `release.yaml` publishes `uriel-gateway` (and uriel-tools' own release publishes `uriel-tools`, the MCP server; split out 2026-09-27), and the digest bump in `homelab-apps` is a manual one-line commit.

## Failure handling

| Failure | Behaviour |
|---|---|
| Ollama down or slow | Configurable timeout (60 s) and 1 retry, then "assistant unavailable" in the UI and 503 on the API |
| Decider error or low confidence | Route `tools`; logged with `adapter=fallback` |
| MCP down | Agent runs without tools and says so; gateway stays ready |
| Tool denied or bad arguments | `ToolException` becomes a tool message, and the model explains |
| Recursion limit hit | A clean "couldn't finish" reply |
| Postgres down | Readiness fails; requests get 503 |
| Auth expired or invalid | Web redirects to login; the API returns 401 |

A request ID is generated at the gateway, passed to MCP, and included in every log line.

## Testing

1. **Unit tests:**
   - JWT validation: wrong issuer or audience, expired, bad signature, missing `groups`;
   - key → principal mapping and thread ownership;
   - config validation;
   - MCP gating: 401 without a key, list filtering, denial raising `ToolException`, unlisted tool denied.
2. **Graph tests**, with a scripted fake chat model and the in-memory FastMCP client:
   - route → agent → tool → answer;
   - denial handled gracefully;
   - history persisted across two turns;
   - decider fallback.
3. **Compose end to end (CI):** Postgres, MCP, gateway, a mock OIDC server (`navikt/mock-oauth2-server`) and a stub OpenAI-compatible LLM that returns scripted tool calls. It covers the admin-allowed and family-denied scenarios over HTTP.
4. **`@pytest.mark.live`:** the same scenarios against real `qwen3:8b` on Ollama. It runs manually, not in CI.
5. **`scripts/smoke.sh`:** after a deploy, `POST /v1/chat` with a service key and check for a tool-backed answer.

## Out of scope (later specs)

- Real RAG: ingest, embeddings (`nomic-embed-text`), rerank, pgvector search.
- The Jev-style decider (spec 2).
- Background-role work and scheduler.
- The real UI.
- Voice satellites, vision and image generation.
- Restricting egress.
- Automatic digest bumps.

## To verify first in the implementation plan

1. Whether `langchain-mcp-adapters` can set per-call MCP `meta`; otherwise use the header fallback.
2. How the tool list gets filtered per principal: per-principal MCP session/client, or filtering on the gateway side while still enforcing on the MCP side.
3. Whether Authelia 4.39 issues a `groups` claim in the ID token for client `uriel`, or only via userinfo.
4. Whether `backup-prep`'s `pg_dump` via `kubectl exec` needs any NetworkPolicy allowance.
5. Qwen3-8B tool-calling reliability through Ollama's OpenAI endpoint with reasoning off.
