import logging

from langchain_core.tools import StructuredTool

from uriel.agent.decider import QUESTIONS, Decision
from uriel.agent.pool import categories_of, category_of, pool_for, tools_in

CATS = ["camera", "documents", "homelab", "memory", "reporting"]


def tool(name, category=None):
    meta = {"_meta": {"uriel": {"category": category}}} if category else None
    return StructuredTool.from_function(lambda: "", name=name, description=name, metadata=meta)


def decided(**p):
    return Decision("category", max(p, key=p.get), 0.9, "systemone", "tev", 1, p)


def test_category_question_describes_exactly_these_categories():
    assert set(QUESTIONS["category"]["criteria"]) == {
        "none",
        "documents",
        "memory",
        "reporting",
        "homelab",
        "camera",
        "calendar",
        "schedules",
        "web",
    }


def test_confident_pick_is_one_category():
    assert pool_for(decided(memory=0.95, none=0.03, reporting=0.02), CATS, 0.9) == ["memory"]


def test_torn_decision_widens_the_pool():
    assert pool_for(decided(memory=0.5, documents=0.45, none=0.05), CATS, 0.9) == ["memory", "documents"]


def test_none_alone_means_no_tools_but_none_with_mass_elsewhere_does_not():
    assert pool_for(decided(none=0.96, memory=0.04), CATS, 0.9) == []
    assert pool_for(decided(none=0.6, memory=0.4), CATS, 0.9) == ["memory"]


def test_reporting_alone_is_a_pool():
    assert pool_for(decided(reporting=0.97, none=0.03), CATS, 0.9) == ["reporting"]


def test_no_probabilities_means_every_category():
    fallback = Decision("category", "tools", None, "fallback", "tev", 1)
    assert pool_for(fallback, CATS, 0.9) == CATS


def test_a_single_choice_without_probabilities_is_its_own_pool():
    assert pool_for(Decision("category", "memory", 0.9, "llm", "qwen", 1), CATS, 0.9) == ["memory"]
    assert pool_for(Decision("category", "none", 0.9, "llm", "qwen", 1), CATS, 0.9) == []


def test_an_option_we_did_not_ask_is_ignored():
    assert pool_for(decided(banana=0.9, memory=0.1), CATS, 0.9) == ["memory"]


def test_tools_in_adds_reporting_and_uncategorised_tools():
    tools = [
        tool("remember", "memory"),
        tool("homelab_status", "homelab"),
        tool("draft_issue", "reporting"),
        tool("legacy"),
    ]
    assert [t.name for t in tools_in(["memory"], tools)] == ["remember", "draft_issue", "legacy"]
    assert tools_in([], tools) == []


def test_old_uriel_tools_without_categories_keeps_every_tool():
    tools = [tool("a"), tool("b")]
    assert categories_of(tools) == []
    assert [t.name for t in tools_in(["memory"], tools)] == ["a", "b"]


def test_a_category_the_agent_cannot_describe_is_always_bound():
    tools = [tool("remember", "memory"), tool("open_garage", "garage")]
    assert categories_of(tools) == ["memory"]
    assert [t.name for t in tools_in(["memory"], tools)] == ["remember", "open_garage"]


def test_an_undescribed_category_is_warned_about_once(caplog):
    with caplog.at_level(logging.WARNING, logger="uriel.agent.pool"):
        assert category_of(tool("a", "warn-once-cat")) == "uncategorised"
        assert category_of(tool("b", "warn-once-cat")) == "uncategorised"
    assert len(caplog.records) == 1
    assert "warn-once-cat" in caplog.records[0].getMessage()
