"""Measure how the agent picks tools and fills their arguments, against the real model.

The graph, prompt and router are the gateway's own. Tool names, descriptions and schemas are the ones
uriel-tools lists, so a docstring change there is measured here; but every call is answered from the
scenario's scripted results, so nothing is filed, edited or signed and runs are repeatable.
"""

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from uriel.agent.events import stream_graph
from uriel.agent.graph import RunContext, build_graph
from uriel.agent.memory import MEMORY_TOOL, Memory
from uriel.principal import Principal

# A rule is a literal (compared for equality) or a dict of string checks, all case-insensitive.
Rule = Any


class User(BaseModel):
    name: str
    groups: list[str]


class ScriptedResult(BaseModel):
    # Every listed arg must equal this for the result to apply.
    when: dict[str, Any] = Field(default_factory=dict)
    # Used up by its first match, so a later entry answers the next call (a questionnaire's next question).
    once: bool = False
    result: Any = None
    error: str | None = None  # the tool fails with this message instead


class Expect(BaseModel):
    route: str | None = None
    calls: list[str] = Field(default_factory=list)  # each must be called at least once
    min_calls: dict[str, int] = Field(default_factory=dict)  # e.g. several remember calls for one request
    no_calls: bool = False
    not_calls: list[str | dict[str, dict[str, Rule]]] = Field(default_factory=list)  # a name, or {name: args}
    args: dict[str, dict[str, Rule]] = Field(default_factory=dict)  # checked on the tool's last call
    reply: dict[str, Rule] = Field(default_factory=dict)


class Turn(BaseModel):
    say: str
    expect: Expect = Field(default_factory=Expect)


class Scenario(BaseModel):
    name: str
    tags: list[str] = Field(default_factory=list)  # the area it covers: run only what a change touches
    user: User
    # The person's USER.md and SOUL.md, as the gateway loads them: {"user": ..., "soul": ...}.
    memory: dict[str, str] = Field(default_factory=dict)
    tool_results: dict[str, list[ScriptedResult]] = Field(default_factory=dict)
    turns: list[Turn]


@dataclass
class Check:
    turn: int
    category: str  # route, calls, args, reply, error
    ok: bool
    detail: str


@dataclass
class TurnRun:
    say: str = ""
    route: str | None = None
    calls: list[dict] = field(default_factory=list)
    reply: str = ""
    error: str | None = None
    seconds: float = 0.0


@dataclass
class Run:
    scenario: str
    turns: list[TurnRun]
    checks: list[Check]
    label: str = ""  # which models answered, so runs of different models can be told apart

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)

    def failures(self) -> list[str]:
        return [f"turn {c.turn + 1} {c.category}: {c.detail}" for c in self.checks if not c.ok]


def _text(value: Any) -> str:
    return "" if value is None else str(value).casefold()


def match(value: Any, rule: Rule) -> str | None:
    """None if value satisfies rule, else why not."""
    if not isinstance(rule, dict):
        return None if value == rule else f"is {value!r}, expected {rule!r}"
    text = _text(value)
    if rule.get("absent") and value is not None:
        return f"is {value!r}, expected it absent"
    for needle in rule.get("contains", []):
        if needle.casefold() not in text:
            return f"lacks {needle!r}"
    for needle in rule.get("not_contains", []):
        if needle.casefold() in text:
            return f"contains {needle!r}"
    if (anyof := rule.get("contains_any")) and not any(n.casefold() in text for n in anyof):
        return f"has none of {anyof!r}"
    if (pattern := rule.get("matches")) and not re.search(pattern, str(value or ""), re.IGNORECASE):
        return f"doesn't match {pattern!r}"
    return None


def _args_match(args: dict, rules: dict[str, Rule]) -> str | None:
    for key, rule in rules.items():
        if (why := match(args.get(key), rule)) is not None:
            return f"{key} {why}"
    return None


def scripted_tools(templates: list[BaseTool], results: dict[str, list[ScriptedResult]], calls: list[dict]):
    """Tools with the templates' names, descriptions and schemas that record each call and answer it from
    the first scripted result whose `when` matches the call's arguments."""

    used: set[int] = set()

    def make(t: BaseTool) -> BaseTool:
        async def answer(**args):
            calls.append({"name": t.name, "args": args})
            for r in results.get(t.name, []):
                if all(args.get(k) == v for k, v in r.when.items()) and id(r) not in used:
                    if r.once:
                        used.add(id(r))
                    if r.error:
                        raise ToolException(r.error)
                    return json.dumps(r.result)
            raise ToolException(f"{t.name} is not scripted for these arguments in this eval")

        return StructuredTool(
            name=t.name,
            description=t.description,
            args_schema=t.args_schema,
            coroutine=answer,
            metadata=t.metadata,
        )

    return [make(t) for t in templates]


def _check(i: int, expect: Expect, turn: TurnRun) -> list[Check]:
    out = []
    if turn.error:
        out.append(Check(i, "error", False, turn.error))
    if expect.route:
        why = match(turn.route, expect.route)
        out.append(Check(i, "route", why is None, why or turn.route))
    names = [c["name"] for c in turn.calls]
    for name in expect.calls:
        out.append(Check(i, "calls", name in names, f"{name} {'called' if name in names else 'not called'}"))
    for name, least in expect.min_calls.items():
        n = names.count(name)
        out.append(Check(i, "calls", n >= least, f"{name} called {n} time(s), expected at least {least}"))
    if expect.no_calls:
        out.append(Check(i, "calls", not names, f"called {names}" if names else "no calls"))
    for banned in expect.not_calls:
        name, rules = (banned, {}) if isinstance(banned, str) else next(iter(banned.items()))
        hits = [c for c in turn.calls if c["name"] == name and _args_match(c["args"], rules) is None]
        out.append(
            Check(i, "calls", not hits, f"forbidden {name} {rules or ''} {'called' if hits else 'avoided'}")
        )
    for name, rules in expect.args.items():
        made = [c for c in turn.calls if c["name"] == name]
        why = _args_match(made[-1]["args"], rules) if made else f"{name} not called"
        out.append(Check(i, "args", why is None, why or f"{name} args ok"))
    for key, rule in expect.reply.items():
        why = match(turn.reply, {key: rule})
        out.append(Check(i, "reply", why is None, why or "reply ok"))
    return out


async def run_scenario(
    scenario: Scenario,
    *,
    chat_model,
    decider,
    templates: list[BaseTool],
    tool_model=None,
    recursion_limit=10,
    coverage: float | None = None,
) -> Run:
    principal = Principal(scenario.user.name, frozenset(scenario.user.groups), "human")
    thread = f"{principal.user_id}:eval-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread}, "recursion_limit": recursion_limit}
    calls: list[dict] = []
    # As the gateway does (split_memory): memory goes into the prompt, never to the model as a tool.
    templates = [t for t in templates if t.name != MEMORY_TOOL]
    memory = Memory(scenario.memory.get("user", ""), scenario.memory.get("soul", ""))
    graph = build_graph(
        chat_model=chat_model,
        tool_model=tool_model,
        decider=decider,
        coverage=coverage,
        tools=scripted_tools(templates, scenario.tool_results, calls),
        checkpointer=InMemorySaver(),
    )
    turns, checks = [], []
    for i, turn in enumerate(scenario.turns):
        run, start, before = TurnRun(say=turn.say), time.monotonic(), len(calls)
        try:
            ctx = RunContext(principal, thread, f"eval-{i}", memory)
            async for event in stream_graph(graph, turn.say, config=config, context=ctx):
                if event.kind == "final":
                    run.reply = str(event.data)
        except Exception as e:  # a timeout or a crash is a result to report, not a reason to stop the suite
            run.error = f"{type(e).__name__}: {e}"
        run.seconds = time.monotonic() - start
        run.calls = calls[before:]
        state = await graph.aget_state(config)
        run.route = state.values.get("route") if state else None
        turns.append(run)
        checks.extend(_check(i, turn.expect, run))
    return Run(scenario.name, turns, checks)


def summarise(runs: list[Run]) -> list[dict]:
    """One row per scenario: runs that passed overall and per check category, and the median turn time."""
    rows: dict[str, dict] = {}
    for run in runs:
        row = rows.setdefault(run.scenario, {"scenario": run.scenario, "_runs": [], "_secs": []})
        row["_runs"].append(run)
        row["_secs"].extend(t.seconds for t in run.turns)
    out = []
    for row in rows.values():
        runs_ = row.pop("_runs")
        secs = sorted(row.pop("_secs"))
        n = len(runs_)
        row["pass"] = f"{sum(r.passed for r in runs_)}/{n}"
        for cat in ("route", "calls", "args", "reply"):
            relevant = [r for r in runs_ if any(c.category == cat for c in r.checks)]
            if relevant:
                ok = sum(all(c.ok for c in r.checks if c.category == cat) for r in relevant)
                row[cat] = f"{ok}/{len(relevant)}"
            else:
                row[cat] = "-"
        row["p50_s"] = round(secs[len(secs) // 2], 1) if secs else None
        out.append(row)
    return out
