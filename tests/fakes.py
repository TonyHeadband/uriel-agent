import asyncio
import json
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableLambda

from uriel.gateway.directory import GroupLookupError


class ScriptedChatModel(GenericFakeChatModel):
    """GenericFakeChatModel can't bind tools and drops tool_calls when streaming (verified 2026-09-27)."""

    structured: list[Any] = []

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        queue = iter(self.structured)
        return RunnableLambda(lambda _: next(queue))

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._generate(messages, stop=stop, **kwargs).generations[0].message
        if msg.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=msg.content,
                    id=msg.id,
                    chunk_position="last",
                    tool_call_chunks=[
                        {"name": t["name"], "args": json.dumps(t["args"]), "id": t["id"], "index": i}
                        for i, t in enumerate(msg.tool_calls)
                    ],
                )
            )
            return
        words = str(msg.content).split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token, id=msg.id))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk


# Metadata for a plain langchain tool standing in for one uriel-tools marks hidden (memory_context).
HIDDEN = {"_meta": {"uriel": {"hidden": True}}}


class RecordingChatModel(ScriptedChatModel):
    """Keeps the messages of every call and the tools bound, so tests can read what the model got."""

    seen: list[list[BaseMessage]] = []
    bound: list[list[str]] = []  # every binding, names sorted
    bound_tool_names: list[str] = []  # the latest binding, in the order given

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_tool_names = [t.name for t in tools]
        self.bound.append(sorted(self.bound_tool_names))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class HangingChatModel(ScriptedChatModel):
    """A model that never answers, like Ollama stuck behind a busy GPU."""

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        await asyncio.Event().wait()

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        await asyncio.Event().wait()
        yield


class GatedChatModel(ScriptedChatModel):
    """Sets `started` when asked, and answers only once `gate` is set: a turn caught in flight."""

    started: Any = None
    gate: Any = None

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.started.set()
        await self.gate.wait()
        for chunk in self._stream(messages, stop=stop, **kwargs):
            yield chunk


class FailingChatModel(ScriptedChatModel):
    """Simulates Ollama being down."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise ConnectionError("ollama unreachable")

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        def boom(_):
            raise ConnectionError("ollama unreachable")

        return RunnableLambda(boom)


class StaticGroups:
    """A GroupDirectory from a dict, returning groups raw as lldap would; `down` simulates an outage."""

    def __init__(self, groups: dict[str, set[str]]):
        self.groups = groups
        self.calls: list[tuple[str, bool]] = []
        self.down = False

    async def groups_of(self, uid, *, fresh=False):
        self.calls.append((uid, fresh))
        if self.down:
            raise GroupLookupError("lldap unreachable")
        found = self.groups.get(uid)
        return None if found is None else frozenset(found)


class MemoryCursors:
    """TalkCursors without Postgres."""

    def __init__(self, initial: dict[str, int] | None = None):
        self.rows = dict(initial or {})

    async def load(self):
        return dict(self.rows)

    async def advance(self, token, message_id):
        self.rows[token] = max(self.rows.get(token, message_id), message_id)
