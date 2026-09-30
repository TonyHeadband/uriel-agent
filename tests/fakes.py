import json
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableLambda


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


class RecordingChatModel(ScriptedChatModel):
    """Keeps the messages of every call, so tests can read the system prompt the model got."""

    seen: list[list[BaseMessage]] = []
    bound_tool_names: list[str] = []

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_tool_names = [t.name for t in tools]
        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class FailingChatModel(ScriptedChatModel):
    """Simulates Ollama being down."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise ConnectionError("ollama unreachable")

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        def boom(_):
            raise ConnectionError("ollama unreachable")

        return RunnableLambda(boom)
