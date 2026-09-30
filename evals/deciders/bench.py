"""Measure a decider on labelled decisions: accuracy, calibration and latency.

    uv run python -m evals.deciders.bench --models config/models.dev.yaml
    uv run python -m evals.deciders.bench --models config/models.dev.yaml --model tev1-0.8b --n 3
    uv run python -m evals.deciders.bench --models config/models.dev.yaml --point category
        --set evals/deciders/route.jsonl --coverage 0.8 0.9 0.95

The category point also reports, per coverage, how often the tool pool still holds the right category and
how many categories and tools it leaves the chat model.

The decider runs unguarded, so the table shows what each min_confidence would cost: how often the guard
would fall back to tools, and how accurate the decisions that pass it are. Local models only; the set is
synthetic, but a decider is meant to run in the house.
"""

import argparse
import asyncio
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from uriel.agent.decider import ROUTE_OPTIONS, Decider, Decision, decider_for
from uriel.agent.pool import NONE, pool_for
from uriel.config import ModelsConfig, load_yaml, models_outside_the_house

HERE = Path(__file__).parent
THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
CATEGORY_OPTIONS = [NONE, "camera", "documents", "homelab", "memory", "reporting"]
COVERAGES = (0.8, 0.9, 0.95)
TOOLS_PER_CATEGORY = {"camera": 1, "documents": 7, "homelab": 1, "memory": 6, "reporting": 2}
# The pool always adds reporting for a non-empty pool (pool.ALWAYS), so it costs its tools even unpicked.
ALWAYS_TOOLS = TOOLS_PER_CATEGORY["reporting"]


@dataclass
class Result:
    id: str
    expected: str
    choice: str | None
    confidence: float | None
    latency_ms: int
    error: str | None = None
    probabilities: dict[str, float] | None = None

    @property
    def correct(self) -> bool:
        return self.choice == self.expected


def load_set(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def point_setup(point: str, row: dict | None = None) -> tuple[list[str], str]:
    """The options a point decides among and the label field that holds the right one."""
    if point == "category":
        return CATEGORY_OPTIONS, "expected_category"
    return (row or {}).get("options", ROUTE_OPTIONS), "expected"


async def run(decider: Decider, rows: list[dict], point: str) -> list[Result]:
    out = []
    for row in rows:
        options, label = point_setup(point, row)
        expected = row[label]  # a row without its label is a broken set, not a skip
        try:
            d = await decider.choose(point, row["context"], options)
            out.append(
                Result(row["id"], expected, d.choice, d.confidence, d.latency_ms, None, d.probabilities)
            )
        except Exception as e:  # a failure is a fallback in production; count it, don't stop the run
            out.append(Result(row["id"], expected, None, None, 0, repr(e)))
    return out


@dataclass
class PoolStats:
    recall: float
    mean_pool: float
    mean_tools: float
    misses: list[Result]


def pool_stats(results: list[Result], categories: list[str], coverage: float) -> PoolStats:
    """What the tool pool would hold at this coverage. A failed decision has no probabilities, so its pool
    is every category, as in production."""
    misses, sizes, tools = [], [], []
    for r in results:
        decision = Decision("category", r.choice or "", r.confidence, "", None, r.latency_ms, r.probabilities)
        pool = pool_for(decision, categories, coverage)
        if not (pool == [] if r.expected == NONE else r.expected in pool):
            misses.append(r)
        sizes.append(len(pool))
        held = sum(TOOLS_PER_CATEGORY[c] for c in pool)
        tools.append(held + (ALWAYS_TOOLS if pool and "reporting" not in pool else 0))
    n = len(results)
    return PoolStats(1 - len(misses) / n, sum(sizes) / n, sum(tools) / n, misses)


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def report(results: list[Result]) -> None:
    n = len(results)
    ok = [r for r in results if r.error is None]
    latencies = [r.latency_ms for r in ok]
    print(f"\naccuracy {sum(r.correct for r in results)}/{n} = {sum(r.correct for r in results) / n:.1%}")
    if latencies:
        print(
            f"latency p50 {percentile(latencies, 0.5)} ms, p95 {percentile(latencies, 0.95)} ms, "
            f"mean {statistics.mean(latencies):.0f} ms"
        )
    print(f"errors {n - len(ok)}")
    print("\nmin_conf  kept  fallback  acc_kept  acc_with_fallback")
    for t in THRESHOLDS:
        kept = [r for r in ok if r.confidence is not None and r.confidence >= t]
        # The guard turns every dropped decision into tools, so score those as tools.
        with_fallback = sum(r.correct if r in kept else r.expected == "tools" for r in results)
        acc_kept = sum(r.correct for r in kept) / len(kept) if kept else 0
        print(f"{t:>8.1f}  {len(kept):>4}  {n - len(kept):>8}  {acc_kept:>8.1%}  {with_fallback / n:>17.1%}")
    misses = [r for r in results if not r.correct]
    if misses:
        print("\nmisses:")
        for r in misses:
            conf = f"{r.confidence:.2f}" if r.confidence is not None else "-"
            print(
                f"  {r.id}: expected {r.expected}, got {r.choice} ({conf}){' ' + r.error if r.error else ''}"
            )


def report_pool(results: list[Result], coverages: list[float]) -> None:
    categories = [o for o in CATEGORY_OPTIONS if o != NONE]
    print("\ncoverage  recall  mean_pool  mean_tools")
    all_stats = [(c, pool_stats(results, categories, c)) for c in coverages]
    for c, s in all_stats:
        print(f"{c:>8.2f}  {s.recall:>6.1%}  {s.mean_pool:>9.2f}  {s.mean_tools:>10.1f}")
    for c, s in all_stats:
        if s.misses:
            print(f"\npool misses at coverage {c}:")
            for r in s.misses:
                print(f"  {r.id}: expected {r.expected}, decided {r.choice}, probabilities {r.probabilities}")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="config/models.dev.yaml")
    ap.add_argument("--point", default="route")
    ap.add_argument("--model", help="models: entry to decide with; default: the point's configured decider")
    ap.add_argument("--set", default=None, help="labelled JSONL; default: evals/deciders/<point>.jsonl")
    ap.add_argument("--coverage", type=float, nargs="+", default=list(COVERAGES), help="category point only")
    ap.add_argument("--n", type=int, default=1, help="passes over the set")
    args = ap.parse_args()

    models = load_yaml(ModelsConfig, args.models)
    name = args.model or models.decider(args.point).model
    if name in models_outside_the_house(models):
        raise SystemExit(f"{name} is outside the house network; deciders are benchmarked locally")
    rows = load_set(Path(args.set) if args.set else HERE / f"{args.point}.jsonl")
    decider = decider_for(models.models[name])
    await decider.choose(
        args.point, "warm-up", point_setup(args.point)[0]
    )  # keep model loading out of the latencies

    results = []
    for _ in range(args.n):
        results += await run(decider, rows, args.point)
    print(f"{name} ({models.models[name].model}) on {len(rows)} {args.point} decisions x{args.n}")
    report(results)
    if args.point == "category":
        report_pool(results, args.coverage)


if __name__ == "__main__":
    asyncio.run(main())
