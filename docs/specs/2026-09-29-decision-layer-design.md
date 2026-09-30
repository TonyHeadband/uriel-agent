# Uriel — Decision layer design: System One engine and tool categories

Status: built; route decider not switched (see Measured on Homelab) · Date: 2026-09-29 · Follows: [milestone 1](2026-09-27-milestone-1-design.md) (D7)

## Goal

This spec gives Uriel a second, small model that helps qwen3:8b with what it lacks: knowing when it is unsure.
The small model doesn't replace the chat model. It would do two jobs:

- narrow the tools qwen3:8b chooses from, on every turn;
- in spec 4, check tool results and confirmations.

On the house hardware, tev1:4b failed the bar for the first job, so it is not switched on. See "Measured on
Homelab" below.

Pain points it addresses:

- misroutes;
- wrong tool picks from all 17 tools bound at once;
- a route confidence that can't be trusted;
- bad tool results that go unchecked (spec 4).

## Findings (2026-09-29)

**The ecosystem.**
- Jev (TypeSafe, 2026-09-15) is a hosted "System One" model that answers typed questions instead of generating
  text.
- Its HTTP format, `POST /v1/systemone`, is now shared by:
  - Laya (a 421M encoder);
  - Kev (Qwen3.5 LoRA);
  - OpenJev (vLLM);
  - **Ollama ≥ 0.35**, which serves it natively for the Tev1 and Nimble models.
- SemIf (formerly OpenJev), the logit-reading approach M1 named, is now a reference implementation only.
- "Layla" is an unrelated phone app.

**The wire format, as Ollama 0.35 accepts it:**

```json
{"model": "tev1:4b-q4_K_M", "state": "<context>",
 "questions": {"route": {"type": "choice", "instructions": "<question>",
                         "criteria": {"tools": "<description>", "direct": "<description>"}}}}
```

It answers with `{"answers": {"route": {"choice", "probabilities": {option: p}, "confidence"}}}`.

- `confidence` is 1 minus the normalised entropy of the probabilities, so it is low on a torn two-way choice even at
  p = 0.69.
- Ollama rejects list-valued criteria (TypeSafe's format allows them), so each option's description must be a single
  string.

**Benchmark.** 40 held-out route decisions (`evals/deciders/route_holdout.jsonl`): Italian, indirect phrasing,
follow-ups, and near-misses such as "What is a W-2 form?". The Tev runs were on CPU.

| Decider | Accuracy | Confidence | p50 |
|---|---|---|---|
| qwen3:8b, structured output (M1) | 88.8% | Misses reported 0.95–1.00: the guard catches none | 263 ms (GPU) |
| tev1:0.8b | 72.5% | Weak | 550 ms (CPU) |
| tev1:4b-q4_K_M | 80.0% (route question; the category pool measured lower, see Measured on Homelab) | **12/12 right at ≥ 0.5; every miss ≤ 0.21** | 2.4 s (CPU; ~4.6 s on Homelab, see Measured on Homelab) |

qwen3:8b understands better, but it doesn't know when it's wrong. tev1:4b knows when it's unsure. So each model gets
the job its strength suits:

- **tev1:4b** would decide how wide the tool pool is;
- **qwen3:8b** reasons, picks a tool and fills its arguments.

That was the plan. On Homelab's hardware tev1:4b failed the latency and recall bars, so qwen3:8b still routes.
See "Measured on Homelab" below.

### Measured on Homelab (2026-09-29): the bar is not met, so the decider is not switched

**Placement.**
- On the GPU, tev1:4b answers in ~71 ms but evicts qwen3:8b (8 GB card: qwen 5.76 GiB, other workloads ~0.55 GiB),
  whose reload takes ~26 s.
- `/v1/systemone` ignores a per-request `options.num_gpu`, though it honours `keep_alive`.
- So the models files name `tev1-4b-cpu`, a Modelfile variant with `PARAMETER num_gpu 0`, recreated by
  `scripts/create-decision-models.sh`.
- `/api/ps` after the benches: qwen3:8b fully in VRAM (6,186,378,198 B); tev1-4b-cpu at `size_vram 0`, expiring
  30 min after its last use (`keep_alive: 30m`). **Residency holds.**

**Latency: fails.** A category decision on the CPU takes **~4.6 s** (p50 4616 ms, p95 4835 ms on `route.jsonl`,
first pass). Its ~385-token prompt (usage `input_tokens` 384–390) is evaluated afresh for every new message:
the unchanged question is not reused from the prompt cache, presumably because the server's template puts the
state first. Only an exact repeat, served from the prompt cache, answers in ~190–280 ms.
That is where the earlier ~175 ms "warm" figure came from, and why `--n 3` shows p50 278 ms but p95 4764 ms. A
`timeout_s: 2` decider would time out on every real turn and fall back to all tools.

**Recall: fails.** `category` point, `--coverage 0.8 0.9 0.95 0.98`:

| Set · model · wording | Accuracy | Recall 0.80 / 0.90 / 0.95 / 0.98 | Mean pool (categories) | Mean tools | p50 / p95 |
|---|---|---|---|---|---|
| holdout (40) · tev1-4b-cpu · shipped, ×3 | 67.5% | 75.0 / 70.0 / 65.0 / 60.0% | 0.80 / 1.05 / 1.32 / 1.75 | 5.4 / 6.9 / 7.9 / 9.3 | 278 / 4764 ms (cache) |
| holdout (40) · tev1-4b-cpu · reworded, ×1 | 82.5% | 82.5 / 77.5 / 75.0 / 65.0% | 0.72 / 0.93 / 1.05 / 1.57 | 4.8 / 6.1 / 6.7 / 9.1 | 5697 / 6058 ms |
| route (44) · tev1-4b-cpu · shipped, ×1 | 84.1% | 95.5 / 90.9 / 81.8 / 77.3% | 0.84 / 0.95 / 1.20 / 1.57 | 5.3 / 5.8 / 6.9 / 8.4 | 4616 / 4835 ms |
| holdout (40) · tev1:0.8b (loads on CPU) · shipped, ×1 | 67.5% | 60.0 / 60.0 / 57.5 / 55.0% | 1.18 / 1.80 / 2.50 / 3.25 | 6.7 / 8.5 / 9.6 / 12.0 | 230 / 293 ms |

- tev1:4b is deterministic, so the three holdout passes gave identical decisions.
- **Recall falls as coverage rises.** A `none` case counts only when the pool is empty, which needs p(none) ≥
  coverage. 16 of the 40 holdout cases are `none`, and tev1 rarely puts ≥ 0.9 on `none` (a poem: 0.895; "thanks"
  after a tool: 0.888). At coverage ≥ 0.9, every holdout miss is a `none` case that gets a pool. That error is
  cheap, since qwen answers with tools it doesn't need, but it is what the bar counts.
- The near-miss general questions ("What is a W-2 form?", "server vs NAS", "renew a passport") go to their topic's
  category at p 0.59–0.77: tev1:4b doesn't separate "about the family's own X" from "about X".
- One wording round (each category as "the family's own ... right now", `none` spelled out as advice, how-to and
  writing) raised accuracy from 67.5% to 82.5% and the best recall from 75.0% to 82.5%, still far from 97%. The
  longer prompt also cost ~1 s. It was not kept.
- tev1:0.8b is fast enough, but no better than chance at the pool.

**Evals** (`--tag memory reporting routing --n 3`, qwen3:8b, uriel-tools 0.7.0). Both use the `llm` route decider,
so this checks Tasks 1–5 and uriel-tools 0.7.0 for regressions, not the category pool.

| Scenario | Branch pass | Branch p50 | `main` pass | `main` p50 |
|---|---|---|---|---|
| feature_request | 3/3 | 8.0 s | 3/3 | 8.1 s |
| general_knowledge | 3/3 | 0.9 s | 3/3 | 0.9 s |
| memory_call_me_sir | 3/3 | 5.7 s | 2/3 | 5.1 s |
| memory_family_fact | 3/3 | 11.6 s | 3/3 | 8.9 s |
| memory_forget_birthday | 3/3 | 5.9 s | 3/3 | 6.0 s |
| memory_personal_fact | 3/3 | 6.4 s | 3/3 | 6.4 s |
| memory_questionnaire | 3/3 | 11.7 s | 3/3 | 6.7 s |
| memory_style_by_example | 3/3 | 8.9 s | 3/3 | 6.6 s |
| memory_talk_like_character | 3/3 | 9.0 s | 3/3 | 8.6 s |
| memory_uses_what_it_knows | 3/3 | 4.1 s | 3/3 | 3.9 s |
| offer_after_tool_failure | 3/3 | 11.8 s | 3/3 | 9.3 s |
| report_bug_explicit | 2/3 | 11.4 s | 3/3 | 12.2 s |
| report_bug_grounded | 2/3 | 8.4 s | 3/3 | 9.1 s |
| report_declined | 3/3 | 8.7 s | 3/3 | 10.5 s |
| small_talk | 3/3 | 0.8 s | 3/3 | 0.8 s |
| **Total** | **43/45** | | **44/45** | |

- Branch failures: report_bug_explicit (the reply didn't cite `#12`); report_bug_grounded (the description quoted
  "let anthony know"). `main` failure: memory_call_me_sir (`remember` into `soul` instead of `user`).
- A re-run of the two branch failures passed 6/6, so the one-run gap is sampling noise (temperature 0.3).

**Chosen coverage: none.** `deciders.route` stays `llm` on qwen3-8b; `tev1-4b` stays defined but unused. Before
revisiting:
- a GPU placement that doesn't evict qwen3:8b (a smaller quant, or a bigger card);
- or a prompt layout that lets Ollama cache the question, with the state last;
- and a way past the `none` recall ceiling (for example, count `none` as met when qwen answers without a tool
  call), or a better decision model.

### Grouping with qwen as the picker (2026-09-29)

With tev1 out, this tests the grouping idea itself.
- **How it works:** the route call asks qwen3:8b for a category instead of yes/no (`LLMDecider` with
  `rubric("category", …)`, the same wording tev had). Its single choice is the pool: `none` means direct, and a
  fallback means all tools. It is switched on by setting `coverage` on an llm decider.

**Bench** (`--point category`, qwen3-8b):

| Set | Category accuracy (= pool recall) | Binary route | p50 / p95 | Tools bound |
|---|---|---|---|---|
| `route.jsonl` | 100% | 100% | 253 / 273 ms | 4.6 of 17 |
| `route_holdout.jsonl` (×3) | 90.0% | 88.8% | 254 / 280 ms | 3.5 of 17 |

- Every miss is the same tools-versus-direct confusion the binary route has. There were no wrong-category picks
  among requests that needed tools.
- The memory criterion needed the route rubric's examples. Without them, "My favourite band is Radiohead" went to
  `none`: 90.9% in-distribution before the fix.

**Evals** (memory, reporting, routing; n = 3; uriel-tools 0.7.0; grouping on):
- 43/45, against 43/45 on this branch with grouping off and 44/45 on `main`.
- The only failure was report_bug_grounded (twice, on the same wording check it failed with grouping off).
- The mean run was 9.1 s against 9.5 s on `main`, which is within noise.

**Conclusion:** grouping is neutral at 17 tools, because qwen already picks well from all of them. It is merged
switched off (`coverage` unset). Re-measure when the schedules and web tools land (about 26 tools), which is where
a smaller pool should start to pay.

Known wart: pydantic prints a serializer warning on each `none` choice from the `Literal` schema. It's harmless;
fix it before switching grouping on.

## Decisions

| # | Decision | Rejected alternatives |
|---|---|---|
| E1 | **Single inference point:** decision models run in the house Ollama (≥ 0.35) via `/v1/systemone` | Laya or Kev as a separate service; SemIf wrapper |
| E2 | **Zero training:** off-the-shelf Tev1; fine-tune only if evals demand it | Training a classifier now |
| E3 | The **model's provider picks the adapter** (`provider: systemone`) | An adapter flag separate from the model |
| C1 | Tool **categories are declared in uriel-tools' `tools.yaml`** and sent as tool `_meta` | Map in agent config (drifts); tev scoring every tool (costly, unpredictable) |
| C2 | The **pool is sized by probability mass**, not by the single winning category | Top-1 category (a wrong pick hides the right tool) |
| C3 | **`reporting` is always in a non-empty pool** | Letting the decider drop it (REPORT_HINT offers a report after any failure) |

## Engine (built)

- `SystemOneDecider` (`src/uriel/agent/decider.py`) asks one `choice` question.
  - `QUESTIONS[point]` holds the instructions and a list of cases per option; the adapter joins them into one string.
  - `Decision.confidence` is the server's confidence.
- `decider_for(spec)` picks `SystemOneDecider` or `LLMDecider` by `provider`.
  - The gateway, `evals/run.py` and the bench all use it.
- `GuardedDecider` is unchanged: errors, timeouts and low confidence fall back safely.
- `ModelsConfig` rejects:
  - a `systemone` model in a chat role;
  - a decider whose adapter doesn't match its model's provider.
- `evals/deciders/bench.py` reports, for any decider:
  - accuracy;
  - a calibration table: kept, fallback and accuracy per `min_confidence`;
  - p50/p95 latency.

## Tool categories (spec 3)

### Declaring them (uriel-tools 0.7.0)

`ToolRule` gains a required `category`:

```yaml
tools:
  remember: { groups: [family], category: memory }
```

A tool without one fails config validation at startup. `GroupGate.on_list_tools` sets
`_meta: {"uriel": {"category": ...}}` on each tool it lists. langchain-mcp-adapters passes that on as
`tool.metadata["_meta"]`.

Initial categories, for the 17 tools the model sees (`memory_context` is loaded by the gateway, not the model):

| Category | Tools |
|---|---|
| `documents` | search_documents, ocr_now, inspect_document, fill_form, edit_text, sign_document, confirm_edit |
| `memory` | remember, forget, show_memory, next_question, answer_question, set_style |
| `reporting` | draft_issue, file_issue |
| `homelab` | homelab_status |
| `camera` | door_camera_last_event |

### The decision

- `route` becomes one `choice` over `none` plus the categories present in this principal's tools.
  - Options follow the tools actually listed, so a `family` user is never offered `homelab`.
  - Each category's description lives in `QUESTIONS["category"]` in the agent. A tool whose category has no
    description is bound as uncategorised and never offered to the decider.
- `Decision` gains `probabilities: dict[str, float] | None`; `LLMDecider` leaves it `None`.
- **Pool:** sort categories by probability and take the smallest prefix whose sum is ≥ `coverage`. `coverage` is set
  per point in `models.yaml`. It is unset by default (`None` keeps today's binary route); 0.9 was the planned
  value, not the code default.
  - The prefix is only `none` → `respond` (today's `direct`, with `DIRECT_NOTE`).
  - Otherwise → `agent`, bound to the tools of every category in the prefix, plus `reporting` (C3). A `none` in the
    prefix is dropped.
  - Decider error, timeout, or no probabilities (`LLMDecider`) → `agent` with all tools. That is today's behaviour
    and the fallback.
- The pool is fixed for the turn: the `agent ⇄ tools` loop binds the same tools each step.
- `State.route` stays for the event stream. The pool goes in a new `State.pool: list[str]`.

### Logging

Migration `0002` adds `probabilities jsonb` and `pool jsonb` to `decisions`, written by `DecisionLog.record`.
`pool` records the categories picked (or all categories on the binary route); `reporting` and uncategorised
tools are bound implicitly and are not listed.

### Measuring

- `route_holdout.jsonl` gains `expected_category` per case, and the bench learns category points. It reports:
  - **pool recall**: the expected category is in the pool, with `none` counting only as `respond`;
  - **mean pool size**, in tools;
  - p50/p95 latency.
- **Bar:**
  - pool recall ≥ 97% on the held-out set at the chosen `coverage`;
  - mean pool size well below 17;
  - p95 ≤ 300 ms on the Homelab GPU.
- Then run the `memory`, `reporting` and `routing` eval areas on local qwen. Pass rates must be at least `main`'s,
  and the tool-pick (`calls`) checks are the ones expected to improve.

### Error handling

- The decider fails → all tools. A turn never fails because of the decider.
- An unknown category in `_meta`, or a tool without one, goes to an `uncategorised` group that is always bound, with
  a warning logged. A half-upgraded uriel-tools degrades to today's behaviour instead of hiding tools.

## Rollout

1. Ollama 0.35.0 on Homelab (`homelab-apps` commit `8593266`, awaiting push). Pull `tev1:4b-q4_K_M` there,
   then measure GPU latency, and check with `/api/ps` that qwen3:8b isn't evicted.
2. uriel-tools 0.7.0: `category` in `tools.yaml` plus `_meta`.
3. uriel-agent: categories, pool, migration, bench, built. Then `deciders.route` → `tev1-4b` in both model
   files: not done, failed the pass bar; see Measured on Homelab.

## Spec 4 (next)

- An `effect` tag per tool in `tools.yaml`: `read | reversible | confirm`.
- Only `confirm` tools become two-part (draft → commit, like `draft_issue`/`file_issue` and the edit_ copies →
  `confirm_edit`).
- The person's reply to a draft becomes a tev `choice` (`confirm | decline | amend`), and the graph, not the
  prompt, allows the commit.
- After a tool returns, a tev yes/no question checks that the result answers the request, before qwen replies.
