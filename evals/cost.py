"""Keep hosted-model evals cheap: price the usage each model reports, and stop a suite that is over budget or
only hitting rate limits."""

from pathlib import Path

import yaml
from langchain_core.callbacks import BaseCallbackHandler

RATE_LIMITED = ("429", "RateLimit", "RESOURCE_EXHAUSTED")
RATE_LIMIT_STREAK = 3  # consecutive turns; one could be a burst, three is a quota


class Meter:
    """Adds up token usage per model name; models without a price (the local ones) cost nothing."""

    def __init__(self, prices: dict[str, dict[str, float]]):
        self._prices = prices
        self.tokens: dict[str, tuple[int, int]] = {}

    def handler(self, model: str) -> BaseCallbackHandler:
        meter = self

        class _Usage(BaseCallbackHandler):
            def on_llm_end(self, response, **kwargs):
                for generation in (g for batch in response.generations for g in batch):
                    usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                    tin, tout = meter.tokens.get(model, (0, 0))
                    meter.tokens[model] = (
                        tin + usage.get("input_tokens", 0),
                        tout + usage.get("output_tokens", 0),
                    )

        return _Usage()

    @property
    def usd(self) -> float:
        total = 0.0
        for model, (tin, tout) in self.tokens.items():
            price = self._prices.get(model)
            if price:
                total += tin / 1e6 * price["usd_per_mtok_in"] + tout / 1e6 * price["usd_per_mtok_out"]
        return total


def load_prices(models_file: Path) -> dict[str, dict[str, float]]:
    """Prices live beside the models under a key the gateway's config ignores."""
    return (yaml.safe_load(models_file.read_text()) or {}).get("eval_prices", {})


def missing_prices(hosted: list[str], prices: dict) -> list[str]:
    return [m for m in hosted if m not in prices]


def stop_reason(runs, *, spent: float, budget: float) -> str | None:
    if spent >= budget:
        return f"budget reached: ${spent:.3f} of ${budget:.2f}"
    streak = 0
    for turn in (t for run in runs for t in run.turns):
        limited = bool(turn.error) and any(s in turn.error for s in RATE_LIMITED)
        streak = streak + 1 if limited else 0
    if streak >= RATE_LIMIT_STREAK:
        return f"the last {streak} turns were rate-limited; the quota is probably used up"
    return None
