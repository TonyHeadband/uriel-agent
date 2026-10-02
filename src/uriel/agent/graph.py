import json
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime

from uriel.agent.decider import ROUTE_OPTIONS, Decider
from uriel.agent.events import has_draft
from uriel.agent.memory import Memory
from uriel.agent.pool import NONE, categories_of, pool_for, tools_in
from uriel.principal import Principal

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Uriel, a private assistant for one family, running entirely on their own hardware. "
    "Be brief and warm. Use tools when they help; if a tool reports you are not permitted, say so plainly "
    "and do not guess the answer. When a tool can answer, call it now rather than saying you will. "
    # Probe 2026-09-28: without this, qwen3:8b described a family form it had never searched.
    "Never describe the contents of the family's documents, records or household state unless a tool "
    "returned them in this conversation; if you haven't looked, say so and offer to search. When you use a "
    "document, name it and give its link. "
    "Never say a document was filled, changed or signed unless an editing tool returned `saved`; edits "
    "are new edit_ copies next to the original, so give their link and ask the person to review them. "
    "To sign a PDF, preview_signature returns a draft_id and saves nothing: tell the person where the "
    "signature will go and call place_signature only after their yes. "
    "When the person tells you a lasting fact about themselves or their family, save it with remember and "
    "say so in a few words. "
    # Probe 2026-09-29: without this, "talk to me like Jarvis" got role-play and a promise, and nothing saved.
    "When they describe how you should talk, even by example (a character, a person, a style), save it right "
    "away with set_style rather than just saying you will adjust. Never claim to remember something unless a "
    "tool returned saved or unchanged. To set up someone's profile, use next_question and pass each reply to "
    "answer_question. "
    "If a result is wrong or the person wishes you could do something you can't, offer to report it to "
    "Anthony: draft_issue prepares the report, show it to them, and send it with file_issue only after their "
    "yes. Never say a report was sent unless file_issue returned `filed`. "
    "To remind the person of something or put an appointment in their calendar, use add_reminder or "
    "add_event: tell them what you added, where and when, and offer to undo. When either returns a draft_id "
    "(a family calendar), nothing is added yet: show them the change and call confirm_calendar_change only "
    "after their yes. Work you do for them on a schedule is different: prepare it with draft_schedule, show "
    "them when it will run and what it will do, and call create_schedule only after their yes; if they "
    'change anything, draft it again. When they name something you may be doing or keeping for them ("the '
    'news one"), look it up with list_schedules or calendar_agenda before asking what they mean. Move or '
    "delete calendar items with draft_calendar_change and "
    "confirm_calendar_change only after their yes. Never say something is scheduled, added or changed unless "
    "a tool returned it. "
    # Mirrors the reporting rule: nothing is scheduled that the person didn't see.
    "When you search the web, answer from the results and give the links you used; if nothing useful came "
    "back, say so rather than guessing."
)
# Probe 2026-09-29: without it a direct reply to "call me Toni" said it had saved the name.
DIRECT_NOTE = "No tools ran for this reply: nothing was looked up, saved or changed, so never say it was."
ROUTE_CONTEXT_CHARS = 500  # of the previous reply; a question to the person usually ends it
REPORTING_TOOLS = {"draft_issue", "file_issue"}
# Evals 2026-09-29: as a system-prompt rule, neither qwen3:8b nor Gemini reliably offered a report after a
# tool failed; next to the error itself, where every model looks before replying, it is hard to miss.
REPORT_HINT = "\n\nIf this stops you helping the person, offer to report it to Anthony with draft_issue."
ROUTE_TO_NODE = {"direct": "respond", "tools": "agent"}
# Live 2026-09-29: the router sent a bare "yes" to a draft down the direct route, which then said "scheduled".
AWAITING_YES_TURNS = 3
# Live 2026-09-29: qwen3:8b on Ollama sometimes writes its call as this text instead of making it.
TEXT_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def clock(memory: Memory) -> str:
    """Today's date, which the model can't know otherwise (probe 2026-09-29: qwen3:8b drafted "remind me"
    for 2023).

    The person's local time comes with their memory; a turn without it (a shared room) gets UTC.
    """
    now = memory.now or f"{datetime.now(UTC):%a %-d %b %Y, %H:%M} (UTC)"
    try:
        today = datetime.strptime(now.split(",")[0], "%a %d %b %Y").date()
    except ValueError:
        return f"It is now {now}."
    # Evals 2026-10-01: qwen3 miscounted weekdays ("Thursday" from a Tuesday became Wednesday), so list them;
    # one per line read better than "Thu 2026-10-01, ..." on qwen3:14b.
    week = "".join(f"\n- {d:%A}: {d:%Y-%m-%d}" for d in (today + timedelta(days=i) for i in range(1, 8)))
    return f"It is now {now}. Today is {today:%A %Y-%m-%d}; use these dates for the coming days:{week}"


def report_hint(tools: Sequence[BaseTool]):
    """A ToolNode wrapper that appends REPORT_HINT to a failed tool's result, when the person may report."""
    if "draft_issue" not in {t.name for t in tools}:
        return None

    async def wrap(request, execute):
        result = await execute(request)
        failed = isinstance(result, ToolMessage) and result.status == "error"
        # A denial is policy, not a fault; a failed report is already the reporting path.
        if failed and result.name not in REPORTING_TOOLS and "not permitted" not in str(result.content):
            result.content = f"{result.content}{REPORT_HINT}"
        return result

    return wrap


def awaiting_yes(messages: Sequence[AnyMessage]) -> bool:
    """Whether a draft (a tool result with a draft_id) from the last few turns may be waiting for a yes: only
    the tools route can act on it, however short the reply."""
    starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    since = starts[-AWAITING_YES_TURNS - 1] if len(starts) > AWAITING_YES_TURNS else 0
    return any(isinstance(m, ToolMessage) and has_draft(m) for m in messages[since:])


def made_calls(message: AIMessage, names: set[str]) -> AIMessage:
    """The message with the calls it wrote as text made real, when it is nothing but calls to known tools.

    Anything else stays as it was: text around the calls may be the model talking about a call.
    """
    if message.tool_calls or not isinstance(message.content, str):
        return message
    found = TEXT_TOOL_CALL.findall(message.content)
    if not found or TEXT_TOOL_CALL.sub("", message.content).strip():
        return message
    calls = []
    for raw in found:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return message
        if not isinstance(data, dict) or data.get("name") not in names:
            return message
        args = data.get("arguments") or {}
        if not isinstance(args, dict):
            return message
        calls.append(
            {"name": data["name"], "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}
        )
    log.info("made %d tool call(s) the model wrote as text", len(calls))
    return AIMessage("", tool_calls=calls, id=message.id, response_metadata=message.response_metadata)


def only_text_calls(message: AnyMessage) -> bool:
    """A reply that is nothing but calls written as text: never made, and misleading to read back."""
    if not isinstance(message, AIMessage) or message.tool_calls or not isinstance(message.content, str):
        return False
    return (
        bool(TEXT_TOOL_CALL.search(message.content)) and not TEXT_TOOL_CALL.sub("", message.content).strip()
    )


def route_context(messages: Sequence[AnyMessage]) -> str:
    """The new message, after the assistant's previous reply when there is one: a bare answer ("8 april 1992")
    only reads as something to act on next to the question it answers."""
    text = str(messages[-1].content)
    said = (m for m in reversed(messages[:-1]) if isinstance(m, AIMessage) and m.content)
    previous = next((m for m in said if not only_text_calls(m)), None)
    if previous is None:
        return text
    return f"Assistant: {str(previous.content)[-ROUTE_CONTEXT_CHARS:]}\nUser: {text}"


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    pool: list[str]


@dataclass(frozen=True)
class RunContext:
    """Per-run data kept out of checkpoints and out of the model's reach."""

    principal: Principal
    thread_id: str
    request_id: str
    memory: Memory = Memory()
    note: str = ""  # one more system line, such as who else reads a shared Talk room
    history_turns: int | None = None  # how many of the latest turns the model sees; None: all
    shared: bool = False  # a thread several people take turns on, each with their own tool gates
    # A scheduled run's tools were chosen with the schedule, so its turn never takes the direct route.
    always_tools: bool = False


def said_aloud(message: AnyMessage) -> bool:
    """What everyone in a shared room read: the questions and the answers posted, not the lookups behind
    them."""
    if isinstance(message, HumanMessage):
        return True
    return isinstance(message, AIMessage) and not message.tool_calls and bool(message.content)


def answered(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """The messages without tool calls that never got their results, nor results without their call, nor
    calls that were only written as text.

    A turn cancelled while its tools ran (a scheduled run's timeout, a shutdown) leaves the tool call in the
    checkpoint with no result, and OpenAI-compatible APIs refuse a history like that.
    """
    messages = [m for m in messages if not only_text_calls(m)]
    results = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    kept = [
        m
        for m in messages
        if not (isinstance(m, AIMessage) and any(tc["id"] not in results for tc in m.tool_calls))
    ]
    calls = {tc["id"] for m in kept if isinstance(m, AIMessage) for tc in m.tool_calls}
    return [m for m in kept if not isinstance(m, ToolMessage) or m.tool_call_id in calls]


def recent(messages: Sequence[AnyMessage], turns: int | None, *, shared: bool = False) -> list[AnyMessage]:
    """The last `turns` turns, each from its human message on, so a tool call never loses its result.

    In a shared thread the earlier turns keep only what was said aloud: another person's tool calls and
    results were gated by their groups, not by those of whoever is asking now.
    """
    starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if turns is not None and len(starts) > turns:
        messages, starts = messages[starts[-turns] :], [i - starts[-turns] for i in starts[-turns:]]
    if shared and starts:
        current = starts[-1]
        messages = [m for m in messages[:current] if said_aloud(m)] + list(messages[current:])
    return answered(messages)


def build_graph(
    *,
    chat_model: BaseChatModel,
    tool_model: BaseChatModel | None = None,
    decider: Decider,
    tools: Sequence[BaseTool],
    checkpointer: Any,
    decision_log: Any | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    coverage: float | None = None,
):
    tools = list(tools)
    # What a turn gets when it skips the decider: a yes to a draft, a scheduled run, a rerouted reply.
    every_category = categories_of(tools) or ["all"]

    def prompt(runtime: Runtime[RunContext], note: str = "") -> SystemMessage:
        text = f"{system_prompt}{note}\nYou are talking to {runtime.context.principal.user_id}. "
        text += clock(runtime.context.memory)
        if runtime.context.note:
            text = f"{text}\n{runtime.context.note}"
        memory = runtime.context.memory.prompt()
        return SystemMessage(f"{text}\n\n{memory}" if memory else text)

    async def route(state: State, runtime: Runtime[RunContext]) -> dict:
        if not tools:
            return {"route": "direct", "pool": []}
        if runtime.context.always_tools or awaiting_yes(state["messages"]):
            return {"route": "tools", "pool": every_category}
        text = route_context(state["messages"])
        categories = categories_of(tools)
        if coverage is None or not categories:
            point, options = "route", list(ROUTE_OPTIONS)
            decision = await decider.choose(point, text, options)
            pool = every_category if decision.choice == "tools" else []
        else:
            point, options = "category", [NONE, *categories]
            decision = await decider.choose(point, text, options)
            pool = pool_for(decision, categories, coverage)
        if decision_log is not None:
            try:
                await decision_log.record(
                    decision,
                    thread_id=runtime.context.thread_id,
                    user_id=runtime.context.principal.user_id,
                    context=text,
                    options=options,
                    pool=pool,
                )
            except Exception:
                log.warning("failed to record decision", exc_info=True)
        return {"route": "tools" if pool else "direct", "pool": pool}

    async def respond(state: State, runtime: Runtime[RunContext]) -> dict:
        note = f" {DIRECT_NOTE}"
        history = recent(state["messages"], runtime.context.history_turns, shared=runtime.context.shared)
        answer = await chat_model.ainvoke([prompt(runtime, note), *history])
        # A misrouted turn: the model reached for a tool it wasn't given. The tools route asks again with
        # the tools bound, so the arguments come from the real schema and the call never shows as a reply.
        if made_calls(answer, {t.name for t in tools}).tool_calls:
            return {"route": "tools", "pool": every_category}
        return {"messages": [answer]}

    async def agent(state: State, runtime: Runtime[RunContext]) -> dict:
        bound = (tool_model or chat_model).bind_tools(tools_in(state["pool"], tools))
        history = recent(state["messages"], runtime.context.history_turns, shared=runtime.context.shared)
        answer = await bound.ainvoke([prompt(runtime), *history])
        return {"messages": [made_calls(answer, {t.name for t in tools})]}

    g = StateGraph(State, context_schema=RunContext)
    g.add_node("route", route)
    g.add_node("respond", respond)
    g.add_edge(START, "route")
    if tools:
        rerouted = {"agent": "agent", "respond": END}
        g.add_conditional_edges("respond", lambda s: ROUTE_TO_NODE[s["route"]], rerouted)
        g.add_node("agent", agent)
        g.add_node("tools", ToolNode(tools, handle_tool_errors=True, awrap_tool_call=report_hint(tools)))
        g.add_conditional_edges("route", lambda s: ROUTE_TO_NODE[s["route"]], ["respond", "agent"])
        g.add_conditional_edges("agent", tools_condition)
        g.add_edge("tools", "agent")
    else:
        g.add_edge("route", "respond")
        g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)
