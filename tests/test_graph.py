import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, ToolException, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from tests.fakes import RecordingChatModel, ScriptedChatModel
from uriel.agent.decider import INSTRUCTIONS, Decision
from uriel.agent.events import stream_graph
from uriel.agent.graph import (
    SYSTEM_PROMPT,
    RunContext,
    awaiting_yes,
    build_graph,
    clock,
    recent,
    route_context,
)
from uriel.agent.memory import Memory
from uriel.principal import Principal

DAD = Principal("dad", frozenset({"admins"}), "human")


@tool
def homelab_status() -> dict:
    """Homelab summary."""
    return {"nodes": [{"name": "homelab", "ready": True}]}


@tool
def door_camera_last_event() -> str:
    """Camera."""
    raise ToolException("door_camera_last_event is not permitted for dad")


class FixedDecider:
    def __init__(self, choice):
        self.choice = choice
        self.calls = 0
        self.contexts = []

    async def choose(self, point, context, options):
        self.calls += 1
        self.contexts.append(context)
        return Decision(point, self.choice, 0.9, "llm", "fake", 1)


class CategoryDecider:
    def __init__(self, **probabilities):
        self.probabilities = probabilities
        self.options = None

    async def choose(self, point, context, options):
        self.options = list(options)
        return Decision(
            point,
            max(self.probabilities, key=self.probabilities.get),
            0.9,
            "systemone",
            "tev",
            1,
            self.probabilities,
        )


def categorised(t, category):
    return StructuredTool.from_function(
        func=t.func,
        coroutine=t.coroutine,
        name=t.name,
        description=t.description,
        args_schema=t.args_schema,
        metadata={"_meta": {"uriel": {"category": category}}},
    )


class RecordingLog:
    def __init__(self):
        self.rows = []

    async def record(self, d, **kw):
        self.rows.append((d.choice, kw["thread_id"], kw["context"], kw["options"], kw["pool"]))


class FailingLog:
    async def record(self, d, **kw):
        raise RuntimeError("decision log unreachable")


def cfg(thread="dad:1", limit=10):
    return {"configurable": {"thread_id": thread}, "recursion_limit": limit}


async def collect(graph, text, thread="dad:1", limit=10):
    ctx = RunContext(DAD, thread, "req-1")
    return [e async for e in stream_graph(graph, text, config=cfg(thread, limit), context=ctx)]


async def test_tools_route_calls_tool_then_answers():
    model = ScriptedChatModel(
        messages=iter(
            [
                AIMessage("", tool_calls=[{"name": "homelab_status", "args": {}, "id": "c1"}]),
                AIMessage("Homelab is healthy."),
            ]
        )
    )
    log = RecordingLog()
    graph = build_graph(
        chat_model=model,
        decider=FixedDecider("tools"),
        tools=[homelab_status],
        checkpointer=InMemorySaver(),
        decision_log=log,
    )
    events = await collect(graph, "how is the homelab?")
    kinds = [e.kind for e in events]
    assert kinds[0] == "tool_call" and events[0].data["name"] == "homelab_status"
    assert "tool_result" in kinds
    assert events[-1].kind == "final" and events[-1].data == "Homelab is healthy."
    assert "".join(e.data for e in events if e.kind == "token") == "Homelab is healthy."
    assert [row[:3] for row in log.rows] == [("tools", "dad:1", "how is the homelab?")]


async def test_turn_completes_when_decision_log_write_fails():
    model = ScriptedChatModel(
        messages=iter(
            [
                AIMessage("", tool_calls=[{"name": "homelab_status", "args": {}, "id": "c1"}]),
                AIMessage("Homelab is healthy."),
            ]
        )
    )
    graph = build_graph(
        chat_model=model,
        decider=FixedDecider("tools"),
        tools=[homelab_status],
        checkpointer=InMemorySaver(),
        decision_log=FailingLog(),
    )
    events = await collect(graph, "how is the homelab?")
    assert events[-1].kind == "final" and events[-1].data == "Homelab is healthy."
    assert any(e.kind == "tool_call" for e in events)


async def test_direct_route_skips_tools():
    graph = build_graph(
        chat_model=ScriptedChatModel(messages=iter([AIMessage("Hello!")])),
        decider=FixedDecider("direct"),
        tools=[homelab_status],
        checkpointer=InMemorySaver(),
    )
    events = await collect(graph, "hi")
    assert [e.kind for e in events if e.kind != "token"] == ["final"]


async def test_no_tools_means_no_decider_call():
    decider = FixedDecider("tools")
    graph = build_graph(
        chat_model=ScriptedChatModel(messages=iter([AIMessage("Hi")])),
        decider=decider,
        tools=[],
        checkpointer=InMemorySaver(),
    )
    await collect(graph, "hi")
    assert decider.calls == 0


async def test_denied_tool_becomes_error_result_and_model_explains():
    model = ScriptedChatModel(
        messages=iter(
            [
                AIMessage("", tool_calls=[{"name": "door_camera_last_event", "args": {}, "id": "c1"}]),
                AIMessage("I'm not allowed to check the camera for you."),
            ]
        )
    )
    graph = build_graph(
        chat_model=model,
        decider=FixedDecider("tools"),
        tools=[door_camera_last_event],
        checkpointer=InMemorySaver(),
    )
    events = await collect(graph, "who's at the door?")
    result = next(e for e in events if e.kind == "tool_result")
    assert result.data["status"] == "error"
    assert events[-1].kind == "final"


async def test_history_persists_across_turns():
    saver = InMemorySaver()
    model = ScriptedChatModel(messages=iter([AIMessage("One."), AIMessage("Two.")]))
    graph = build_graph(chat_model=model, decider=FixedDecider("direct"), tools=[], checkpointer=saver)
    await collect(graph, "first")
    await collect(graph, "second")
    state = await graph.aget_state(cfg())
    assert [m.content for m in state.values["messages"]] == ["first", "One.", "second", "Two."]


async def test_endless_tool_loop_hits_recursion_limit():
    loop = [
        AIMessage("", tool_calls=[{"name": "homelab_status", "args": {}, "id": f"c{i}"}]) for i in range(20)
    ]
    graph = build_graph(
        chat_model=ScriptedChatModel(messages=iter(loop)),
        decider=FixedDecider("tools"),
        tools=[homelab_status],
        checkpointer=InMemorySaver(),
    )
    with pytest.raises(GraphRecursionError):
        await collect(graph, "loop", limit=4)


def test_prompt_and_rubric_cover_memory():
    from uriel.agent.decider import INSTRUCTIONS
    from uriel.agent.graph import SYSTEM_PROMPT

    assert "tool returned saved" in SYSTEM_PROMPT and "answer_question" in SYSTEM_PROMPT
    assert "even by example" in SYSTEM_PROMPT and "set_style" in SYSTEM_PROMPT
    assert "remember or forget" in INSTRUCTIONS["route"] and "profile" in INSTRUCTIONS["route"]
    assert "answers a question the assistant asked" in INSTRUCTIONS["route"]
    assert "talk to me like <anyone>" in INSTRUCTIONS["route"]


async def test_the_router_sees_the_previous_reply_too():
    # A bare answer ("8 april 1992") only reads as something to save next to the question it answers.
    decider = FixedDecider("direct")
    graph = build_graph(
        chat_model=ScriptedChatModel(
            messages=iter([AIMessage("When's your birthday?"), AIMessage("Noted.")])
        ),
        decider=decider,
        tools=[homelab_status],
        checkpointer=InMemorySaver(),
    )
    await collect(graph, "let's set up my profile")
    await collect(graph, "8 april 1992")
    assert decider.contexts == [
        "let's set up my profile",
        "Assistant: When's your birthday?\nUser: 8 april 1992",
    ]


async def test_direct_replies_are_told_nothing_ran():
    model = RecordingChatModel(messages=iter([AIMessage("Hello!")]))
    graph = build_graph(
        chat_model=model, decider=FixedDecider("direct"), tools=[homelab_status], checkpointer=InMemorySaver()
    )
    await collect(graph, "call me Toni")
    assert "No tools ran for this reply" in str(model.seen[0][0].content)


async def test_direct_turns_keep_asking_the_decider():
    decider = FixedDecider("direct")
    graph = build_graph(
        chat_model=ScriptedChatModel(messages=iter([AIMessage("hi"), AIMessage("hi again")])),
        decider=decider,
        tools=[homelab_status],
        checkpointer=InMemorySaver(),
    )
    await collect(graph, "hello")
    await collect(graph, "hello again")
    assert decider.calls == 2


@tool
def search_documents(query: str) -> str:
    """Search."""
    raise ToolException("document search is unavailable right now")


@tool
def draft_issue(title: str) -> str:
    """Draft a report."""
    return "drafted"


@tool
def file_issue(draft_id: str) -> str:
    """File a report."""
    raise ToolException("no such draft (it may have expired): call draft_issue again")


ARGS = {"search_documents": {"query": "trip"}, "file_issue": {"draft_id": "d1"}, "door_camera_last_event": {}}


async def tool_result_for(failing, tools):
    call = {"name": failing, "args": ARGS[failing], "id": "c1"}
    model = ScriptedChatModel(messages=iter([AIMessage("", tool_calls=[call]), AIMessage("ok")]))
    graph = build_graph(
        chat_model=model, decider=FixedDecider("tools"), tools=tools, checkpointer=InMemorySaver()
    )
    events = await collect(graph, "go")
    result = next(e for e in events if e.kind == "tool_result")
    assert result.data["status"] == "error"
    return result.data["content"]


async def test_a_failed_tool_suggests_reporting_it():
    content = await tool_result_for("search_documents", [search_documents, draft_issue, file_issue])
    assert "unavailable right now" in content
    assert "draft_issue" in content


async def test_no_report_hint_without_the_reporting_tool():
    assert "draft_issue" not in await tool_result_for("search_documents", [search_documents])


async def test_no_report_hint_for_a_denial_or_a_reporting_failure():
    denied = await tool_result_for(
        "door_camera_last_event", [door_camera_last_event, draft_issue, file_issue]
    )
    assert "Anthony" not in denied
    failed_report = await tool_result_for("file_issue", [draft_issue, file_issue])
    assert "offer to report" not in failed_report


async def test_the_model_sees_only_the_latest_turns():
    model = RecordingChatModel(messages=iter([AIMessage("1"), AIMessage("2"), AIMessage("3")]))
    graph = build_graph(
        chat_model=model, decider=FixedDecider("direct"), tools=[], checkpointer=InMemorySaver()
    )
    ctx = RunContext(DAD, "dad:1", "r", history_turns=2)
    for text in ["one", "two", "three"]:
        [e async for e in stream_graph(graph, text, config=cfg(), context=ctx)]
    assert [m.content for m in model.seen[-1] if isinstance(m, HumanMessage)] == ["two", "three"]


async def test_a_run_note_reaches_the_system_prompt():
    model = RecordingChatModel(messages=iter([AIMessage("Hi all.")]))
    graph = build_graph(
        chat_model=model, decider=FixedDecider("direct"), tools=[], checkpointer=InMemorySaver()
    )
    ctx = RunContext(DAD, "room:talk-family", "r", note="Everyone in this room can read your answer.")
    [e async for e in stream_graph(graph, "hi", config=cfg("room:talk-family"), context=ctx)]
    system = str(model.seen[0][0].content)
    assert "You are talking to dad. It is now " in system
    assert "(UTC). Today is " in system
    assert system.endswith("\nEveryone in this room can read your answer.")


def test_prompt_and_rubric_cover_schedules_and_search():
    assert "draft_schedule" in SYSTEM_PROMPT and "create_schedule only after their yes" in SYSTEM_PROMPT
    assert "give the links you used" in SYSTEM_PROMPT
    assert "news" in INSTRUCTIONS["route"] and '"Scheduled task"' in INSTRUCTIONS["route"]


def test_prompt_separates_reminders_from_schedules():
    assert "add_reminder" in SYSTEM_PROMPT and "add_event" in SYSTEM_PROMPT
    assert "confirm_calendar_change" in SYSTEM_PROMPT and "draft_schedule" in SYSTEM_PROMPT
    assert "When either returns a draft_id (a family calendar)" in SYSTEM_PROMPT


def test_a_shared_thread_keeps_earlier_turns_as_said_aloud_and_the_current_one_whole():
    call = {"name": "search_documents", "args": {}, "id": "c1"}
    earlier = [HumanMessage("q1"), AIMessage("", tool_calls=[call]), ToolMessage("secret", tool_call_id="c1")]
    earlier.append(AIMessage("a1"))
    current = [HumanMessage("q2"), AIMessage("", tool_calls=[call | {"id": "c2"}])]
    current.append(ToolMessage("mine", tool_call_id="c2"))
    assert recent(earlier + current, None, shared=True) == [earlier[0], earlier[3], *current]
    assert recent(earlier + current, 1, shared=True) == current
    assert recent(earlier + current, None) == earlier + current


def test_a_tool_call_left_without_its_result_is_not_shown_to_the_model():
    call = {"name": "search_documents", "args": {}, "id": "c1"}
    dangling = [HumanMessage("q1"), AIMessage("", tool_calls=[call]), HumanMessage("q2")]
    orphan = [HumanMessage("q1"), ToolMessage("stray", tool_call_id="gone"), HumanMessage("q2")]
    assert recent(dangling, None) == [dangling[0], dangling[2]]
    assert recent(orphan, None) == [orphan[0], orphan[2]]


TEXT_CALL = '<tool_call>\n{"name": "homelab_status", "arguments": {}}\n</tool_call>'


def test_a_call_that_was_posted_as_text_is_not_shown_to_the_model_or_the_router():
    # Live 2026-09-30: with two such replies in the thread, the model answered the third "remind me" by
    # saying it had set the reminder, calling nothing.
    stuck = [HumanMessage("how is the homelab?"), AIMessage(TEXT_CALL), HumanMessage("how is the homelab?")]
    assert recent(stuck, None) == [stuck[0], stuck[2]]
    assert route_context(stuck) == "how is the homelab?"


async def test_a_tool_call_written_as_text_is_made():
    # Live 2026-09-29: qwen3:8b on Ollama answered "y" with this text, and nothing was created.
    model = ScriptedChatModel(messages=iter([AIMessage(TEXT_CALL), AIMessage("Homelab is healthy.")]))
    graph = build_graph(
        chat_model=model, decider=FixedDecider("tools"), tools=[homelab_status], checkpointer=InMemorySaver()
    )
    events = await collect(graph, "how is the homelab?")
    assert [e.data["name"] for e in events if e.kind == "tool_call"] == ["homelab_status"]
    assert [e.data for e in events if e.kind == "final"] == ["Homelab is healthy."]


async def test_a_tool_call_written_on_the_direct_route_hands_the_turn_to_the_tools():
    # Live 2026-09-30: a misrouted "remind me..." was answered with the call as text, which was posted as the
    # reply; the router then read that reply as the job done and sent the retry direct too.
    model = ScriptedChatModel(
        messages=iter(
            [
                AIMessage(TEXT_CALL),
                AIMessage("", tool_calls=[{"name": "homelab_status", "args": {}, "id": "c1"}]),
                AIMessage("Homelab is healthy."),
            ]
        )
    )
    saver = InMemorySaver()
    graph = build_graph(
        chat_model=model, decider=FixedDecider("direct"), tools=[homelab_status], checkpointer=saver
    )
    events = await collect(graph, "how is the homelab?")
    assert [e.data["name"] for e in events if e.kind == "tool_call"] == ["homelab_status"]
    assert [e.data for e in events if e.kind == "final"] == ["Homelab is healthy."]
    kept = (await graph.aget_state(cfg()))[0]["messages"]
    assert not [m for m in kept if "<tool_call>" in str(m.content)]


@pytest.mark.parametrize(
    "text",
    [
        TEXT_CALL.replace("homelab_status", "format_disk"),  # not a tool it was given
        "Here is how I would call it: " + TEXT_CALL,  # talking about a call is not making one
        "<tool_call>not json</tool_call>",
    ],
)
async def test_text_that_is_not_just_a_known_call_stays_text(text):
    model = ScriptedChatModel(messages=iter([AIMessage(text)]))
    graph = build_graph(
        chat_model=model, decider=FixedDecider("tools"), tools=[homelab_status], checkpointer=InMemorySaver()
    )
    events = await collect(graph, "go")
    assert not [e for e in events if e.kind == "tool_call"]
    assert [e.data for e in events if e.kind == "final"] == [text]


@tool
def draft_schedule(title: str) -> dict:
    """Draft."""
    return {"draft_id": "d1", "title": title}


@tool
def create_schedule(draft_id: str) -> dict:
    """Create."""
    return {"created": True, "schedule_id": 1}


async def test_a_yes_to_a_draft_always_reaches_the_tools():
    # Live 2026-09-29: the router sent "yes" to the direct route, which said "scheduled" with nothing created.
    draft = {"name": "draft_schedule", "args": {"title": "Plants"}, "id": "c1"}
    create = {"name": "create_schedule", "args": {"draft_id": "d1"}, "id": "c2"}
    model = ScriptedChatModel(
        messages=iter(
            [
                AIMessage("", tool_calls=[draft]),
                AIMessage("Tonight at 20:00. Schedule it?"),
                AIMessage("", tool_calls=[create]),
                AIMessage("Scheduled."),
            ]
        )
    )
    decider = FixedDecider("tools")
    graph = build_graph(
        chat_model=model,
        decider=decider,
        tools=[draft_schedule, create_schedule],
        checkpointer=InMemorySaver(),
    )
    drafted = await collect(graph, "remind me at 8pm")
    assert [e.data["awaiting"] for e in drafted if e.kind == "tool_result"] == [True]
    decider.choice = "direct"
    events = await collect(graph, "yes")
    assert [e.data["name"] for e in events if e.kind == "tool_call"] == ["create_schedule"]
    assert [e.data["awaiting"] for e in events if e.kind == "tool_result"] == [False]
    assert decider.calls == 1


def test_a_draft_waits_for_its_yes_for_three_turns():
    drafted = [
        HumanMessage("remind me"),
        AIMessage("", tool_calls=[{"name": "draft_schedule", "args": {}, "id": "c1"}]),
        ToolMessage('{"draft_id": "d1"}', tool_call_id="c1"),
        AIMessage("Schedule it?"),
    ]
    chat = [HumanMessage("hm"), AIMessage("Take your time.")]
    assert awaiting_yes([*drafted, HumanMessage("yes")])
    assert awaiting_yes([*drafted, *chat, *chat, HumanMessage("yes")])
    assert not awaiting_yes([*drafted, *chat, *chat, *chat, HumanMessage("yes")])
    assert not awaiting_yes([HumanMessage("hi"), AIMessage("Hello."), HumanMessage("yes")])


async def test_category_route_binds_only_the_picked_category():
    model = RecordingChatModel(messages=iter([AIMessage("Homelab is healthy.")]))
    decider = CategoryDecider(homelab=0.97, none=0.03)
    graph = build_graph(
        chat_model=model,
        decider=decider,
        checkpointer=InMemorySaver(),
        coverage=0.9,
        tools=[categorised(homelab_status, "homelab"), categorised(door_camera_last_event, "camera")],
    )
    await collect(graph, "is the server up?")
    assert decider.options == ["none", "camera", "homelab"]
    assert model.bound_tool_names == ["homelab_status"]


async def test_category_none_goes_direct():
    decider = CategoryDecider(none=0.99, homelab=0.01)
    graph = build_graph(
        chat_model=ScriptedChatModel(messages=iter([AIMessage("Hi!")])),
        decider=decider,
        checkpointer=InMemorySaver(),
        coverage=0.9,
        tools=[categorised(homelab_status, "homelab")],
    )
    events = await collect(graph, "hi")
    assert [e.kind for e in events if e.kind != "token"] == ["final"]


async def test_binary_route_without_coverage_binds_every_tool_in_order():
    def plain(name):
        return StructuredTool.from_function(lambda: "", name=name, description=name)

    tools = [
        categorised(homelab_status, "homelab"),
        categorised(draft_issue, "reporting"),
        categorised(plain("remember"), "memory"),
        categorised(plain("web_search"), "web"),
        plain("legacy"),
    ]
    model = RecordingChatModel(messages=iter([AIMessage("Homelab is healthy.")]))
    log = RecordingLog()
    graph = build_graph(
        chat_model=model,
        decider=FixedDecider("tools"),
        tools=tools,
        checkpointer=InMemorySaver(),
        decision_log=log,
    )
    await collect(graph, "is the server up?")
    assert model.bound_tool_names == [t.name for t in tools]
    (_, _, _, options, pool) = log.rows[0]
    assert options == ["direct", "tools"]
    assert pool == ["homelab", "memory", "reporting", "web"]


def test_clock_lists_the_coming_week_so_the_model_needs_no_weekday_arithmetic():
    # Evals 2026-10-01: from "Tue 29 Sep", qwen3:14b put "Thursday" on 09-30 and "Friday" on 10-03.
    text = clock(Memory(now="Tue 29 Sep 2026, 19:06 (America/Toronto)"))

    assert text.startswith("It is now Tue 29 Sep 2026, 19:06 (America/Toronto). Today is Tuesday 2026-09-29")
    assert "\n- Wednesday: 2026-09-30\n- Thursday: 2026-10-01\n- Friday: 2026-10-02" in text
    assert "- Tuesday: 2026-10-06" in text and "2026-10-07" not in text


def test_clock_keeps_a_time_it_cannot_read_without_the_week():
    assert clock(Memory(now="sometime soon")) == "It is now sometime soon."
