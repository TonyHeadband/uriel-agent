import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field, create_model

from uriel.agent.models import build_chat_model
from uriel.config import DeciderSpec, ModelSpec

log = logging.getLogger(__name__)

ROUTE_OPTIONS = ("direct", "tools")

# Qwen3-8B misroutes without an explicit rubric (probe 2026-09-27).
INSTRUCTIONS = {
    "route": (
        "You route messages for a household assistant.\n"
        "- tools: the user asks about live household state or stored information: the homelab or servers, "
        "household documents, the door camera, or anything that needs a lookup. This includes any question "
        "about a specific paper the family has (a form, questionnaire, contract, bill, receipt, certificate, "
        'letter or file), even when it doesn\'t say "document", and any "do I have" or "what does my/the '
        '... say" question, and any request to fill in, change, correct, annotate, sign or approve one of '
        "the family's documents.\n"
        "- tools also: anything the user tells about themselves to keep (what to call them, their name, "
        "birthday, likes, music, hobbies, food), any request to remember or forget something, and setting up "
        "or showing their profile.\n"
        "- tools also: the user tells the assistant how to talk to them, with anyone as the model: "
        '"talk to me like <anyone>", "be <a character or role> with me", "sound like <someone>", '
        '"from now on answer like <someone>", "be shorter / warmer / in <a language>". These are tools '
        "whoever is named.\n"
        "- tools also: the message may follow the assistant's previous reply; a message that answers a "
        "question the assistant asked (a profile question such as a name or birthday, which box to sign, "
        "what to fill in) is tools, however short.\n"
        # Probe 2026-09-29: appended to the bullet above, "Uriel is slow, please tell Anthony" routed direct.
        "- tools: the user reports a problem with Uriel or wants something passed on to Anthony. These are "
        "reported to Anthony with a tool.\n"
        "- direct: greetings, chit-chat, or general knowledge that doesn't depend on the family's own "
        "records.\n"
        "Return the route and your confidence from 0 to 1."
    )
}


# The same rubric as INSTRUCTIONS, as a System One choice: one criterion per option, a list of cases each.
QUESTIONS = {
    "route": {
        "instructions": (
            "A household assistant has tools for the family's own records and settings. Does it need them to "
            "handle the user's newest message? The assistant's previous reply, when given, is context only."
        ),
        "criteria": {
            "tools": [
                "live household state or stored information: the homelab or servers, the door camera, "
                "household documents (a form, questionnaire, contract, bill, receipt, certificate, letter or "
                'file), any "do I have" or "what does my/the ... say" question',
                "filling in, changing, correcting, annotating, signing or approving one of the family's "
                "documents",
                "a fact the user tells about themselves or their family to keep (name, what to call them, "
                "birthday, likes, music, hobbies, food, pets), remembering or forgetting something, or "
                "setting up or showing their profile",
                "how the assistant should talk to them: like a character or person, shorter, warmer, more "
                "formal, in a language",
                "a short answer to a question the assistant just asked (a profile question, which box to "
                "sign, what to fill in)",
                "a problem with Uriel, or something to pass on or report to Anthony",
            ],
            "direct": [
                "greetings, thanks and chit-chat",
                "general knowledge that doesn't depend on the family's own records",
            ],
        },
    },
    "category": {
        "instructions": (
            "A household assistant has tools grouped by what they serve. Which group does it need to handle "
            "the user's newest message? The assistant's previous reply, when given, is context only."
        ),
        "criteria": {
            "none": [
                "greetings, thanks and chit-chat",
                "general knowledge, even about a topic a tool covers (what a form is, how servers work)",
            ],
            "documents": [
                "the family's own documents: a form, questionnaire, contract, bill, receipt, "
                "certificate, letter or file; what one says, whether they have it",
                "filling in, changing, signing or approving one, or answering which box or field",
            ],
            "memory": [
                # Wording from the route rubric: without the examples, qwen3:8b took "My favourite band is
                # Radiohead" for chit-chat (bench 2026-09-29).
                "anything the user tells about themselves or their family, even in passing (name, what to "
                "call them, birthday, likes, music, hobbies, food, pets), or asks you to recall",
                "any request to remember or forget something, or to set up or show their profile",
                "how the assistant should talk to them: a character, a person, shorter, warmer, "
                "formal, a language",
                "a short answer to a profile question the assistant just asked",
            ],
            "reporting": [
                "a problem with Uriel, a wish for a new feature, or anything to pass on to Anthony",
                "confirming or declining a report the assistant drafted",
            ],
            "homelab": ["the homelab, servers, media server or network: whether something is up or down"],
            "camera": ["the door camera or doorbell: who came, what it saw"],
        },
    },
}


@dataclass(frozen=True)
class Decision:
    point: str
    choice: str
    confidence: float | None
    adapter: str
    model: str | None
    latency_ms: int
    probabilities: dict[str, float] | None = None


class Decider(Protocol):
    async def choose(self, point: str, context: str, options: Sequence[str]) -> Decision: ...


def rubric(point: str, options: Sequence[str]) -> str:
    """A System One question as a chat model's instructions, so both kinds of decider share one wording."""
    question = QUESTIONS[point]
    lines = [f"- {o}: {'; '.join(question['criteria'][o])}" for o in options]
    return "\n".join([question["instructions"], *lines, "Return the option and your confidence from 0 to 1."])


def _choice_schema(options: Sequence[str]):
    return create_model(
        "Choice",
        choice=(Literal[tuple(options)], ...),
        confidence=(float, Field(ge=0, le=1)),
    )


class LLMDecider:
    def __init__(self, llm: BaseChatModel, model_name: str):
        self._llm = llm
        self._model_name = model_name

    async def choose(self, point: str, context: str, options: Sequence[str]) -> Decision:
        start = time.perf_counter()
        structured = self._llm.with_structured_output(_choice_schema(options), method="json_schema")
        text = INSTRUCTIONS.get(point) or rubric(point, options)
        out = await structured.ainvoke([SystemMessage(text), HumanMessage(context)])
        latency = round((time.perf_counter() - start) * 1000)
        return Decision(point, out.choice, out.confidence, "llm", self._model_name, latency)


class SystemOneDecider:
    """Asks a System One model one choice question (TypeSafe's /v1/systemone format; Ollama >= 0.35).

    The confidence is the server's, 1 minus the normalised entropy of the option probabilities, so unlike a
    chat model's self-report it drops when the model is torn between options.
    """

    def __init__(self, spec: ModelSpec, client: httpx.AsyncClient | None = None):
        self._spec = spec
        self._client = client or httpx.AsyncClient(timeout=spec.timeout_s)

    async def choose(self, point: str, context: str, options: Sequence[str]) -> Decision:
        start = time.perf_counter()
        question = QUESTIONS[point]
        # Ollama 0.35 wants one string per option, though TypeSafe's format also allows a list.
        criteria = {o: "; ".join(question["criteria"][o]) for o in options}
        body = {
            "model": self._spec.model,
            "state": context,
            "questions": {
                point: {"type": "choice", "instructions": question["instructions"], "criteria": criteria}
            },
        }
        # Ollama unloads an idle model after 5 minutes, and a cold load on CPU costs seconds per turn.
        if "keep_alive" in self._spec.params:
            body["keep_alive"] = self._spec.params["keep_alive"]
        response = await self._client.post(f"{self._spec.base_url.rstrip('/')}/systemone", json=body)
        response.raise_for_status()
        answer = response.json()["answers"][point]
        latency = round((time.perf_counter() - start) * 1000)
        return Decision(
            point,
            answer["choice"],
            answer["confidence"],
            "systemone",
            self._spec.model,
            latency,
            probabilities=answer.get("probabilities"),
        )


def decider_for(spec: ModelSpec) -> Decider:
    """An unguarded decider on this model; its provider picks the adapter."""
    if spec.provider == "systemone":
        return SystemOneDecider(spec)
    return LLMDecider(build_chat_model(spec), spec.model)


class GuardedDecider:
    """Never lets a decider failure break a turn: errors and low confidence become the safe fallback."""

    def __init__(self, inner: Decider, spec: DeciderSpec, fallback: str):
        self._inner = inner
        self._spec = spec
        self._fallback = fallback

    async def choose(self, point: str, context: str, options: Sequence[str]) -> Decision:
        start = time.perf_counter()
        try:
            d = await self._inner.choose(point, context, options)
        except Exception:
            log.warning("decider %s failed; using fallback", point, exc_info=True)
            return Decision(point, self._fallback, None, "fallback", self._spec.model, self._ms(start))
        if d.choice not in options or d.confidence is None or d.confidence < self._spec.min_confidence:
            return Decision(point, self._fallback, d.confidence, "fallback", d.model, d.latency_ms)
        return d

    @staticmethod
    def _ms(start: float) -> int:
        return round((time.perf_counter() - start) * 1000)
