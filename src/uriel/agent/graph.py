import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime

from uriel.agent.decider import ROUTE_OPTIONS, Decider
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
    "Before placing a signature, tell the person where it will go and wait for their yes. "
    "When the person tells you a lasting fact about themselves, save it with remember and say so in a few "
    "words. "
    # Probe 2026-09-29: without this, "talk to me like Jarvis" got role-play and a promise, and nothing saved.
    "When they describe how you should talk, even by example (a character, a person, a style), save it right "
    "away with set_style rather than just saying you will adjust. Never claim to remember something unless a "
    "tool returned saved or unchanged. To set up someone's profile, use next_question and pass each reply to "
    "answer_question. "
    "If a result is wrong or the person wishes you could do something you can't, offer to report it to "
    "Anthony: draft_issue prepares the report, show it to them, and send it with file_issue only after their "
    "yes. Never say a report was sent unless file_issue returned `filed`."
)
# Probe 2026-09-29: without it a direct reply to "call me Toni" said it had saved the name.
DIRECT_NOTE = "No tools ran for this reply: nothing was looked up, saved or changed, so never say it was."
ROUTE_CONTEXT_CHARS = 500  # of the previous reply; a question to the person usually ends it
REPORTING_TOOLS = {"draft_issue", "file_issue"}
# Evals 2026-09-29: as a system-prompt rule, neither qwen3:8b nor Gemini reliably offered a report after a
# tool failed; next to the error itself, where every model looks before replying, it is hard to miss.
REPORT_HINT = "\n\nIf this stops you helping the person, offer to report it to Anthony with draft_issue."
ROUTE_TO_NODE = {"direct": "respond", "tools": "agent"}


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


def route_context(messages: Sequence[AnyMessage]) -> str:
    """The new message, after the assistant's previous reply when there is one: a bare answer ("8 april 1992")
    only reads as something to act on next to the question it answers."""
    text = str(messages[-1].content)
    previous = next((m for m in reversed(messages[:-1]) if isinstance(m, AIMessage) and m.content), None)
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

    def prompt(runtime: Runtime[RunContext], note: str = "") -> SystemMessage:
        text = f"{system_prompt}{note}\nYou are talking to {runtime.context.principal.user_id}."
        memory = runtime.context.memory.prompt()
        return SystemMessage(f"{text}\n\n{memory}" if memory else text)

    async def route(state: State, runtime: Runtime[RunContext]) -> dict:
        if not tools:
            return {"route": "direct", "pool": []}
        text = route_context(state["messages"])
        categories = categories_of(tools)
        if coverage is None or not categories:
            point, options = "route", list(ROUTE_OPTIONS)
            decision = await decider.choose(point, text, options)
            pool = (categories or ["all"]) if decision.choice == "tools" else []
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
        return {"messages": [await chat_model.ainvoke([prompt(runtime, note), *state["messages"]])]}

    async def agent(state: State, runtime: Runtime[RunContext]) -> dict:
        bound = (tool_model or chat_model).bind_tools(tools_in(state["pool"], tools))
        return {"messages": [await bound.ainvoke([prompt(runtime), *state["messages"]])]}

    g = StateGraph(State, context_schema=RunContext)
    g.add_node("route", route)
    g.add_node("respond", respond)
    g.add_edge(START, "route")
    g.add_edge("respond", END)
    if tools:
        g.add_node("agent", agent)
        g.add_node("tools", ToolNode(tools, handle_tool_errors=True, awrap_tool_call=report_hint(tools)))
        g.add_conditional_edges("route", lambda s: ROUTE_TO_NODE[s["route"]], ["respond", "agent"])
        g.add_conditional_edges("agent", tools_condition)
        g.add_edge("tools", "agent")
    else:
        g.add_edge("route", "respond")
    return g.compile(checkpointer=checkpointer)
