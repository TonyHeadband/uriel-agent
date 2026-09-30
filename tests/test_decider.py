import json
from types import SimpleNamespace

import httpx
import pytest

from tests.fakes import FailingChatModel, ScriptedChatModel
from uriel.agent.decider import (
    QUESTIONS,
    ROUTE_OPTIONS,
    GuardedDecider,
    LLMDecider,
    SystemOneDecider,
    decider_for,
    rubric,
)
from uriel.agent.models import build_chat_model
from uriel.config import DeciderSpec, ModelSpec

SPEC = DeciderSpec(adapter="llm", model="m", min_confidence=0.6)


def llm_answering(choice, confidence):
    return ScriptedChatModel(
        messages=iter([]),
        structured=[SimpleNamespace(choice=choice, confidence=confidence)],
    )


async def test_llm_decider_returns_model_choice():
    d = await LLMDecider(llm_answering("tools", 0.9), "m").choose(
        "route", "how is the homelab?", ROUTE_OPTIONS
    )
    assert (d.choice, d.confidence, d.adapter, d.model) == ("tools", 0.9, "llm", "m")


async def test_guarded_decider_passes_confident_answers_through():
    d = await GuardedDecider(LLMDecider(llm_answering("direct", 0.8), "m"), SPEC, "tools").choose(
        "route", "hi", ROUTE_OPTIONS
    )
    assert (d.choice, d.adapter) == ("direct", "llm")


async def test_low_confidence_falls_back():
    d = await GuardedDecider(LLMDecider(llm_answering("direct", 0.3), "m"), SPEC, "tools").choose(
        "route", "hmm", ROUTE_OPTIONS
    )
    assert (d.choice, d.adapter, d.confidence) == ("tools", "fallback", 0.3)


async def test_error_falls_back():
    d = await GuardedDecider(LLMDecider(FailingChatModel(messages=iter([])), "m"), SPEC, "tools").choose(
        "route", "hi", ROUTE_OPTIONS
    )
    assert (d.choice, d.adapter, d.confidence) == ("tools", "fallback", None)


async def test_choice_outside_options_falls_back():
    d = await GuardedDecider(LLMDecider(llm_answering("banana", 0.99), "m"), SPEC, "tools").choose(
        "route", "hi", ROUTE_OPTIONS
    )
    assert (d.choice, d.adapter) == ("tools", "fallback")


def test_chat_model_factory_applies_spec():
    spec = ModelSpec(
        provider="openai_compat",
        base_url="http://ollama:11434/v1",
        model="qwen3:8b",
        params={"temperature": 0.3, "reasoning_effort": "none"},
    )
    m = build_chat_model(spec)
    assert (m.model_name, m.reasoning_effort, m.max_retries) == ("qwen3:8b", "none", 1)


SYSTEMONE = ModelSpec(provider="systemone", base_url="http://ollama:11434/v1", model="tev1:0.8b")


def systemone_answering(answer: dict | None = None, *, status: int = 200, seen: list | None = None):
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append((str(request.url), json.loads(request.content)))
        body = {"model": "tev1:0.8b", "answers": {"route": answer}} if answer else {"error": "boom"}
        return httpx.Response(status, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def choice(chosen: str, confidence: float, p_tools: float) -> dict:
    probabilities = {"tools": p_tools, "direct": 1 - p_tools}
    return {"type": "choice", "choice": chosen, "probabilities": probabilities, "confidence": confidence}


async def test_systemone_decider_asks_one_choice_question_with_criteria_per_option():
    seen = []
    client = systemone_answering(choice("tools", 0.8, 0.97), seen=seen)

    d = await SystemOneDecider(SYSTEMONE, client=client).choose("route", "who rang the bell?", ROUTE_OPTIONS)

    assert (d.choice, d.confidence, d.adapter, d.model) == ("tools", 0.8, "systemone", "tev1:0.8b")
    assert d.probabilities == {"tools": 0.97, "direct": pytest.approx(0.03)}
    url, body = seen[0]
    assert url == "http://ollama:11434/v1/systemone"
    assert (body["model"], body["state"]) == ("tev1:0.8b", "who rang the bell?")
    question = body["questions"]["route"]
    assert question["type"] == "choice" and question["instructions"]
    assert set(question["criteria"]) == set(ROUTE_OPTIONS)
    # Ollama 0.35 refuses list-valued criteria (probe 2026-09-29), though TypeSafe's format allows them.
    assert all(isinstance(v, str) for v in question["criteria"].values())


async def test_systemone_http_error_falls_back():
    inner = SystemOneDecider(SYSTEMONE, client=systemone_answering(status=500))
    d = await GuardedDecider(inner, SPEC, "tools").choose("route", "hi", ROUTE_OPTIONS)
    assert (d.choice, d.adapter) == ("tools", "fallback")


async def test_systemone_low_confidence_falls_back():
    inner = SystemOneDecider(SYSTEMONE, client=systemone_answering(choice("direct", 0.11, 0.31)))
    d = await GuardedDecider(inner, SPEC, "tools").choose("route", "8 april 1992", ROUTE_OPTIONS)
    assert (d.choice, d.adapter, d.confidence) == ("tools", "fallback", 0.11)


async def test_systemone_category_question_sends_only_the_options_asked():
    seen = []
    answer = {
        "type": "choice",
        "choice": "memory",
        "confidence": 0.9,
        "probabilities": {"none": 0.02, "memory": 0.95, "reporting": 0.03},
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: (
                seen.append(json.loads(r.content)),
                httpx.Response(200, json={"answers": {"category": answer}}),
            )[1]
        )
    )
    d = await SystemOneDecider(SYSTEMONE, client=client).choose(
        "category", "call me Toni", ["none", "memory", "reporting"]
    )
    assert set(seen[0]["questions"]["category"]["criteria"]) == {"none", "memory", "reporting"}
    assert d.probabilities["memory"] == 0.95


async def test_systemone_decider_sends_keep_alive_from_params():
    seen = []
    spec = SYSTEMONE.model_copy(update={"params": {"keep_alive": "30m"}})
    client = systemone_answering(choice("tools", 0.8, 0.97), seen=seen)
    await SystemOneDecider(spec, client=client).choose("route", "hi", ROUTE_OPTIONS)
    await SystemOneDecider(SYSTEMONE, client=client).choose("route", "hi", ROUTE_OPTIONS)
    assert seen[0][1]["keep_alive"] == "30m"
    assert "keep_alive" not in seen[1][1]


async def test_llm_decider_answers_the_category_question():
    d = await LLMDecider(llm_answering("memory", 0.9), "m").choose(
        "category", "call me Toni", ["none", "memory", "reporting"]
    )
    assert (d.choice, d.adapter) == ("memory", "llm")


def test_a_chat_model_rubric_describes_only_the_options_offered():
    text = rubric("category", ["none", "memory"])
    assert QUESTIONS["category"]["instructions"] in text
    assert "- memory: " in text and "- none: " in text
    assert "homelab" not in text


def test_the_model_provider_picks_the_adapter():
    chat = ModelSpec(provider="openai_compat", base_url="http://x/v1", model="qwen3:8b")
    assert isinstance(decider_for(SYSTEMONE), SystemOneDecider)
    assert isinstance(decider_for(chat), LLMDecider)
