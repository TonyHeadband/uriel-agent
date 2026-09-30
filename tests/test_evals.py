import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from evals.deciders.bench import Result, pool_stats
from evals.harness import Scenario, match, run_scenario, scripted_tools, summarise
from tests.fakes import ScriptedChatModel
from uriel.agent.decider import Decision
from uriel.agent.pool import category_of


class FixedDecider:
    def __init__(self, choice):
        self.choice = choice

    async def choose(self, point, context, options):
        return Decision(point, self.choice, 0.9, "llm", "fake", 1)


REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "kind": {"type": "string", "enum": ["bug", "feature"]},
        "description": {"type": "string"},
        "confirmed": {"type": "boolean", "default": False},
    },
    "required": ["title", "kind", "description"],
}


def template(name, schema=REPORT_SCHEMA):
    """What McpToolbox returns: a tool with the server's name, description and JSON schema."""

    async def unused(**kwargs):
        raise AssertionError("templates are never called")

    return StructuredTool(name=name, description=f"{name} tool", args_schema=schema, coroutine=unused)


def call(name, args, id_="c1"):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": id_}])


REPORT = Scenario.model_validate(
    {
        "name": "report_bug",
        "user": {"name": "kid", "groups": ["family"]},
        "tool_results": {
            "report_issue": [
                {"when": {"confirmed": True}, "result": {"filed": True, "number": 7, "url": "https://x/7"}},
                {"result": {"filed": False, "next": "Nothing has been sent yet."}},
            ]
        },
        "turns": [
            {
                "say": "The signature is in the wrong box, tell Anthony",
                "expect": {
                    "route": "tools",
                    "calls": ["report_issue"],
                    "not_calls": ["sign_document", {"report_issue": {"confirmed": True}}],
                    "args": {
                        "report_issue": {"kind": "bug", "description": {"not_contains": ["tell anthony"]}}
                    },
                    "reply": {"not_contains": ["has been sent"]},
                },
            },
            {"say": "Yes", "expect": {"args": {"report_issue": {"confirmed": True}}}},
        ],
    }
)


@pytest.mark.parametrize(
    ("value", "rule", "ok"),
    [
        ("bug", "bug", True),
        ("feature", "bug", False),
        (True, True, True),
        (None, False, False),
        ("Signature in the WRONG box", {"contains": ["wrong box"]}, True),
        ("Please tell Anthony", {"not_contains": ["tell anthony"]}, False),
        ("I've filed it as #7", {"contains_any": ["#7", "issue 7"]}, True),
        ("done", {"matches": r"#\d+"}, False),
        (None, {"absent": True}, True),
        ("x", {"absent": True}, False),
    ],
)
def test_match(value, rule, ok):
    assert (match(value, rule) is None) == ok


async def test_scripted_tools_record_calls_and_pick_results_by_args():
    calls = []
    [tool] = scripted_tools([template("report_issue")], REPORT.tool_results, calls)
    draft = await tool.ainvoke({"title": "t", "kind": "bug", "description": "d"})
    filed = await tool.ainvoke({"title": "t", "kind": "bug", "description": "d", "confirmed": True})
    assert json.loads(draft)["filed"] is False
    assert json.loads(filed)["number"] == 7
    assert [c["args"].get("confirmed") for c in calls] == [None, True]


async def test_unscripted_tool_is_an_error_result():
    calls = []
    [tool] = scripted_tools([template("sign_document")], {}, calls)
    with pytest.raises(Exception, match="not scripted"):
        await tool.ainvoke({"title": "t", "kind": "bug", "description": "d"})
    assert calls[0]["name"] == "sign_document"


async def test_a_good_run_passes_every_check():
    model = ScriptedChatModel(
        messages=iter(
            [
                call(
                    "report_issue", {"title": "Wrong box", "kind": "bug", "description": "Signed wrong box"}
                ),
                AIMessage("Here is the draft. Shall I send it?"),
                call(
                    "report_issue",
                    {"title": "Wrong box", "kind": "bug", "description": "x", "confirmed": True},
                ),
                AIMessage("Sent as #7: https://x/7"),
            ]
        )
    )
    run = await run_scenario(
        REPORT, chat_model=model, decider=FixedDecider("tools"), templates=[template("report_issue")]
    )
    assert run.passed, run.failures()
    assert [t.route for t in run.turns] == ["tools", "tools"]
    assert run.turns[1].calls[0]["args"]["confirmed"] is True


async def test_a_false_claim_and_a_premature_filing_fail():
    model = ScriptedChatModel(
        messages=iter(
            [
                call(
                    "report_issue",
                    {"title": "Wrong box", "kind": "bug", "description": "please tell Anthony"},
                ),
                AIMessage("Your report has been sent!"),
                AIMessage("Done."),
            ]
        )
    )
    run = await run_scenario(
        REPORT, chat_model=model, decider=FixedDecider("tools"), templates=[template("report_issue")]
    )
    assert not run.passed
    failed = {(f.turn, f.category) for f in run.checks if not f.ok}
    assert failed == {(0, "args"), (0, "reply"), (1, "args")}


async def test_a_crashing_turn_is_recorded_not_raised():
    class Boom(FixedDecider):
        async def choose(self, point, context, options):
            raise TimeoutError("ollama timed out")

    run = await run_scenario(
        REPORT,
        chat_model=ScriptedChatModel(messages=iter([])),
        decider=Boom("tools"),
        templates=[template("report_issue")],
    )
    assert not run.passed
    assert "ollama timed out" in run.turns[0].error


async def test_summary_counts_passes_per_scenario_and_category():
    model = ScriptedChatModel(
        messages=iter(
            [
                call("report_issue", {"title": "t", "kind": "bug", "description": "d"}),
                AIMessage("Draft ready. Send it?"),
                call("report_issue", {"title": "t", "kind": "bug", "description": "d", "confirmed": True}),
                AIMessage("Sent as #7."),
            ]
        )
    )
    good = await run_scenario(
        REPORT, chat_model=model, decider=FixedDecider("tools"), templates=[template("report_issue")]
    )
    bad = await run_scenario(
        REPORT, chat_model=ScriptedChatModel(messages=iter([])), decider=FixedDecider("direct"), templates=[]
    )
    [row] = summarise([good, bad])
    assert row["scenario"] == "report_bug"
    assert row["pass"] == "1/2"
    assert row["route"] == "1/2"


def test_every_scenario_file_loads():
    from evals.run import load_scenarios

    names = [s.name for s in load_scenarios([])]
    assert "report_bug_explicit" in names and "small_talk" in names
    assert len(names) == len(set(names))


def test_report_embeds_every_result_file(tmp_path):
    from evals.report import build, render

    good = {
        "scenario": "small_talk",
        "turns": [
            {"say": "hi", "route": "direct", "calls": [], "reply": "Hi!", "error": None, "seconds": 0.5}
        ],
        "checks": [{"turn": 0, "category": "route", "ok": True, "detail": "direct"}],
        "passed": True,
    }
    # Results saved before turns recorded what was said have no "say".
    old = good | {"turns": [{k: v for k, v in good["turns"][0].items() if k != "say"}]}
    bad = good | {
        "checks": [{"turn": 0, "category": "route", "ok": False, "detail": "tools"}],
        "passed": False,
        "turns": [good["turns"][0] | {"reply": "</script><b>x</b>"}],
    }
    (tmp_path / "20260929-010000.json").write_text(json.dumps([old]))
    (tmp_path / "20260929-020000.json").write_text(json.dumps([good | {"label": "qwen3:8b"}, bad]))

    data = build(tmp_path)
    assert [f["id"] for f in data["files"]] == ["20260929-010000", "20260929-020000"]
    assert data["files"][1]["rows"][0]["pass"] == "1/2"
    assert [f["label"] for f in data["files"]] == ["", "qwen3:8b"]  # older files have no label
    assert data["files"][1]["runs"][1]["failures"] == ["turn 1 route: tools"]
    html = render(data)
    assert "/*REPORT_DATA*/" not in html
    assert "</script><b>" not in html  # a reply can't close the inline script


def test_meter_prices_reported_usage():
    from langchain_core.messages import AIMessage as Msg
    from langchain_core.outputs import ChatGeneration, LLMResult

    from evals.cost import Meter

    meter = Meter({"gemini": {"usd_per_mtok_in": 0.30, "usd_per_mtok_out": 2.50}})
    handler = meter.handler("gemini")
    usage = {"input_tokens": 1_000_000, "output_tokens": 200_000, "total_tokens": 1_200_000}
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=Msg("x", usage_metadata=usage))]]))
    meter.handler("qwen").on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=Msg("y", usage_metadata=usage))]])
    )
    assert round(meter.usd, 2) == 0.80  # 0.30 + 0.2 * 2.50; the local model costs nothing
    assert meter.tokens == {"gemini": (1_000_000, 200_000), "qwen": (1_000_000, 200_000)}


def test_hosted_models_need_a_price():
    from evals.cost import missing_prices

    assert missing_prices(["gemini"], {}) == ["gemini"]
    assert missing_prices(["gemini"], {"gemini": {"usd_per_mtok_in": 0.3, "usd_per_mtok_out": 2.5}}) == []


def test_stop_on_budget_or_repeated_rate_limits():
    from evals.cost import stop_reason
    from evals.harness import Run, TurnRun

    limited = TurnRun(error="OpenAIRateLimitError: Error code: 429 - RESOURCE_EXHAUSTED")
    fine = TurnRun(reply="ok")
    assert stop_reason([Run("a", [fine, limited], [])], spent=0.1, budget=0.5) is None
    assert "budget" in stop_reason([Run("a", [fine], [])], spent=0.5, budget=0.5)
    runs = [Run("a", [limited, limited], []), Run("b", [limited], [])]
    assert "rate" in stop_reason(runs, spent=0, budget=1)
    assert stop_reason([Run("a", [limited, limited, fine], [])], spent=0, budget=1) is None


async def test_memory_reaches_the_prompt_and_its_loader_stays_hidden():
    seen = {}

    class Recording(ScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen["system"] = messages[0].content
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    scenario = Scenario.model_validate(
        {
            "name": "memory",
            "user": {"name": "kid", "groups": ["family"]},
            "memory": {"user": "## Name\n- Sir"},
            "turns": [{"say": "what's my name?", "expect": {"reply": {"contains": ["sir"]}}}],
        }
    )
    model = Recording(messages=iter([AIMessage("You're Sir.")]))
    templates = [template("remember"), template("memory_context")]
    run = await run_scenario(scenario, chat_model=model, decider=FixedDecider("tools"), templates=templates)
    assert run.passed, run.failures()
    assert "- Sir" in seen["system"]


async def test_once_results_are_used_in_order_and_min_calls_are_counted():
    scenario = Scenario.model_validate(
        {
            "name": "questions",
            "user": {"name": "kid", "groups": ["family"]},
            "tool_results": {
                "next_question": [
                    {"once": True, "result": {"ask": "What should I call you?"}},
                    {"result": {"ask": "When's your birthday?"}},
                ]
            },
            "turns": [{"say": "set up my profile", "expect": {"min_calls": {"next_question": 2}}}],
        }
    )
    calls = []
    [tool] = scripted_tools([template("next_question")], scenario.tool_results, calls)
    first, second, third = [
        json.loads(await tool.ainvoke({"title": "t", "kind": "bug", "description": "d"})) for _ in range(3)
    ]
    assert (first["ask"], second["ask"], third["ask"]) == (
        "What should I call you?",
        "When's your birthday?",
        "When's your birthday?",
    )

    model = ScriptedChatModel(
        messages=iter(
            [
                call("next_question", {"title": "t", "kind": "bug", "description": "d"}),
                AIMessage("What should I call you?"),
            ]
        )
    )
    run = await run_scenario(
        scenario, chat_model=model, decider=FixedDecider("tools"), templates=[template("next_question")]
    )
    assert [c.detail for c in run.checks if not c.ok] == [
        "next_question called 1 time(s), expected at least 2"
    ]


def test_scenarios_are_selected_by_tag_or_name():
    from evals.run import load_scenarios

    everything = load_scenarios([])
    assert all(s.tags for s in everything), [s.name for s in everything if not s.tags]
    memory = load_scenarios([], tags=["memory"])
    assert memory and all("memory" in s.tags for s in memory)
    both = {s.name for s in load_scenarios(["small_talk"], tags=["memory"])}
    assert both == {s.name for s in memory} | {"small_talk"}


def test_scripted_tools_keep_the_templates_category():
    template = StructuredTool.from_function(
        lambda: "", name="remember", description="r", metadata={"_meta": {"uriel": {"category": "memory"}}}
    )
    [t] = scripted_tools([template], {}, [])
    assert category_of(t) == "memory"


def _result(id, expected, probabilities):
    return Result(
        id, expected, max(probabilities, key=probabilities.get), 0.9, 1, probabilities=probabilities
    )


CATEGORIES = ["camera", "documents", "homelab", "memory", "reporting"]


def test_pool_stats_count_recall_size_and_tools():
    results = [
        _result("chat", "none", {"none": 0.97, "memory": 0.03}),
        _result("fact", "memory", {"memory": 0.6, "documents": 0.3, "none": 0.1}),
        _result("bill", "documents", {"memory": 0.7, "documents": 0.2, "none": 0.1}),
    ]
    tight = pool_stats(results, CATEGORIES, 0.5)
    assert tight.recall == pytest.approx(2 / 3)
    assert tight.mean_pool == pytest.approx(2 / 3)
    # chat: no tools; fact: memory 6 + reporting 2; bill: memory 6 + reporting 2
    assert tight.mean_tools == pytest.approx(16 / 3)
    assert [r.id for r in tight.misses] == ["bill"]

    wide = pool_stats(results, CATEGORIES, 0.95)
    assert wide.recall == 1
    assert [r.id for r in wide.misses] == []
    # the 0.95 mass is reached only after none, so fact and bill each keep memory and documents
    assert wide.mean_pool == pytest.approx((0 + 2 + 2) / 3)


def test_pool_stats_reporting_in_pool_is_not_counted_twice():
    [r] = [_result("bug", "reporting", {"reporting": 0.9, "none": 0.1})]
    assert pool_stats([r], CATEGORIES, 0.8).mean_tools == 2


def test_a_failed_decision_falls_back_to_every_category():
    failed = Result("boom", "camera", None, None, 0, "boom")
    stats = pool_stats([failed], CATEGORIES, 0.9)
    assert stats.recall == 1
    assert stats.mean_pool == 5
