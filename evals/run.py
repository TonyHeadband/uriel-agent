"""Run the tool-use scenarios against a model and print pass rates.

    GEMINI_API_KEY=... uv run python -m evals.run --tag memory [--n 3] [--budget-usd 0.50]
    uv run python -m evals.run --models config/models.dev.yaml     # everything on local qwen: release check

Like tests: while working on something, run only its area (--tag, or --only a scenario); the whole suite
is the release check.

Design work runs on Gemini by default (synthetic scenarios only), within a dollar budget per run. Needs a
uriel-tools server to list the tools; nothing is called on it.
"""

import argparse
import asyncio
import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from evals.cost import Meter, load_prices, missing_prices, stop_reason
from evals.harness import Scenario, run_scenario, summarise
from uriel.agent.decider import GuardedDecider, LLMDecider, decider_for
from uriel.agent.mcp_tools import McpToolbox
from uriel.agent.models import build_chat_model
from uriel.config import ModelsConfig, load_yaml, models_outside_the_house
from uriel.principal import Principal

HERE = Path(__file__).parent


def load_scenarios(only: list[str], tags: list[str] | None = None) -> list[Scenario]:
    """Scenarios named in only, plus those carrying any of tags; everything when neither is given."""
    found = [
        Scenario.model_validate(yaml.safe_load(p.read_text())) for p in sorted(HERE.glob("scenarios/*.yaml"))
    ]
    if not only and not tags:
        return found
    return [s for s in found if s.name in only or set(s.tags) & set(tags or [])]


async def templates_for(toolbox: McpToolbox, scenario: Scenario):
    principal = Principal(scenario.user.name, frozenset(scenario.user.groups), "human")
    async with toolbox.open(principal, "eval-list-tools") as tools:
        return list(tools)  # only names, descriptions and schemas are used, after the session closes


def print_table(rows: list[dict]) -> None:
    cols = ["scenario", "pass", "route", "calls", "args", "reply", "p50_s"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def save(runs, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps([asdict(r) | {"passed": r.passed} for r in runs], indent=2, default=str))
    return path


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="runs per scenario")
    ap.add_argument(
        "--tag", nargs="*", default=[], help="run the scenarios for these areas (reporting, memory…)"
    )
    ap.add_argument("--only", nargs="*", default=[], help="scenario names")
    ap.add_argument("--models", default=os.environ.get("URIEL_EVAL_MODELS", str(HERE / "models.gemini.yaml")))
    ap.add_argument(
        "--budget-usd", type=float, default=0.50, help="stop the suite once hosted usage costs this"
    )
    ap.add_argument("--mcp-url", default=os.environ.get("URIEL_EVAL_MCP_URL", "http://localhost:8011/mcp"))
    ap.add_argument("--mcp-key", default=os.environ.get("URIEL_EVAL_MCP_KEY", "dev-mcp-key"))
    ap.add_argument("--label", default="", help="name for these runs in the report; default: the model ids")
    ap.add_argument("--out", default=str(HERE / "results"), help="directory for the JSON of every run")
    args = ap.parse_args()

    models = load_yaml(ModelsConfig, args.models)
    prices = load_prices(Path(args.models))
    meter = Meter(prices)
    if outside := models_outside_the_house(models):
        if missing := missing_prices(outside, prices):
            raise SystemExit(f"add eval_prices for {', '.join(missing)} to {args.models}: cost must be known")
        # Allowed here only: the scenarios are synthetic. The gateway refuses these models outright.
        print(f"note: sending scenario text to hosted model(s) {', '.join(outside)}", flush=True)

    def metered(name: str):
        model = build_chat_model(models.models[name])
        model.callbacks = [meter.handler(name)]
        model.stream_usage = True  # without it, streamed answers report no token usage to price
        return model

    route = models.decider("route")
    route_model = models.models[route.model]
    inner = (
        decider_for(route_model)
        if route_model.provider == "systemone"  # always local, so nothing to price
        else LLMDecider(metered(route.model), route_model.model)
    )
    decider = GuardedDecider(inner, route, "tools")
    interactive = models.roles["interactive"]
    chat_model, tool_model = metered(interactive), metered(models.roles.get("tools", interactive))
    toolbox = McpToolbox(args.mcp_url, args.mcp_key)
    used = [models.role("interactive"), models.role("tools"), models.models[route.model]]
    label = args.label or " + ".join(sorted({m.model for m in used}))

    runs, stopped = [], None
    for scenario in load_scenarios(args.only, args.tag):
        templates = await templates_for(toolbox, scenario)
        for i in range(args.n):
            run = await run_scenario(
                scenario,
                chat_model=chat_model,
                tool_model=tool_model,
                decider=decider,
                templates=templates,
                coverage=route.coverage,
            )
            run.label = label
            print(f"{scenario.name} #{i + 1}: {'pass' if run.passed else 'FAIL'}", flush=True)
            runs.append(run)
            if stopped := stop_reason(runs, spent=meter.usd, budget=args.budget_usd):
                break
        if stopped:
            print(f"\nstopped early: {stopped}")
            break

    print()
    print_table(summarise(runs))
    failures = Counter((r.scenario, f) for r in runs for f in r.failures())
    if failures:
        print("\nfailures (count):")
        for (name, why), n in failures.most_common():
            print(f"  {n}x {name}: {why}")

    for name, (tin, tout) in sorted(meter.tokens.items()):
        print(f"\n{name}: {tin:,} tokens in, {tout:,} out")
    print(f"cost: ${meter.usd:.3f} (budget ${args.budget_usd:.2f})")
    print(f"\nevery run: {save(runs, Path(args.out))}")


if __name__ == "__main__":
    asyncio.run(main())
