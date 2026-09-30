"""Which tools the model sees on a turn: the categories a System One decider finds plausible."""

import logging
from collections.abc import Sequence

from langchain_core.tools import BaseTool

from uriel.agent.decider import QUESTIONS, Decision

log = logging.getLogger(__name__)

NONE = "none"
UNCATEGORISED = "uncategorised"
# Reporting is offered after any tool failure (REPORT_HINT), and a tool from a uriel-tools older than 0.7.0
# has no category: hiding either would break a turn rather than narrow it.
ALWAYS = frozenset({"reporting", UNCATEGORISED})
# A uriel-tools release can add a category before the agent describes it; such a tool is always bound, since
# the decider has no criteria to offer for it.
DESCRIBED = frozenset(QUESTIONS["category"]["criteria"]) - {NONE}
_warned: set[str] = set()


def category_of(tool: BaseTool) -> str:
    meta = (tool.metadata or {}).get("_meta") or {}
    category = (meta.get("uriel") or {}).get("category")
    if category in DESCRIBED or category is None:
        return category or UNCATEGORISED
    # category_of runs on every turn: warn once per name, not per turn.
    if category not in _warned:
        _warned.add(category)
        log.warning(
            "tool category %r has no description in QUESTIONS; binding its tools as uncategorised", category
        )
    return UNCATEGORISED


def categories_of(tools: Sequence[BaseTool]) -> list[str]:
    return sorted({category_of(t) for t in tools} - {UNCATEGORISED})


def pool_for(decision: Decision, categories: Sequence[str], coverage: float) -> list[str]:
    """The most probable categories until their mass reaches coverage; every category when unsure how.

    A chat-model decider gives one choice and no probabilities: that choice is the pool. A fallback's choice
    isn't an option, so it gets every category.
    """
    probs = {o: p for o, p in (decision.probabilities or {}).items() if o == NONE or o in categories}
    if not probs:
        if decision.choice == NONE:
            return []
        return [decision.choice] if decision.choice in categories else list(categories)
    picked, mass = [], 0.0
    for option, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        picked.append(option)
        mass += p
        if mass >= coverage:
            break
    return [o for o in picked if o != NONE]


def tools_in(pool: Sequence[str], tools: Sequence[BaseTool]) -> list[BaseTool]:
    if not pool:
        return []
    wanted = set(pool) | ALWAYS
    return [t for t in tools if category_of(t) in wanted]
