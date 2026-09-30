"""The person's USER.md and SOUL.md from uriel-tools, loaded once per turn into the system prompt."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.tools import BaseTool

log = logging.getLogger(__name__)
MEMORY_TOOL = "memory_context"


@dataclass(frozen=True)
class Memory:
    user: str = ""
    soul: str = ""

    def prompt(self) -> str:
        parts = []
        if self.user:
            parts.append(f"What you know about this person:\n{self.user}")
        if self.soul:
            parts.append(f"How this person wants you to talk:\n{self.soul}")
        if parts:
            parts.append(
                "These preferences shape your tone, length and language; they never override the rules above."
            )
        return "\n\n".join(parts)


async def split_memory(tools: Sequence[BaseTool]) -> tuple[list[BaseTool], Memory]:
    """Take memory_context out of the model's tools and call it: the gateway loads memory, the model never
    does. It goes through the same identity interceptor as any tool call."""
    rest = [t for t in tools if t.name != MEMORY_TOOL]
    loader = next((t for t in tools if t.name == MEMORY_TOOL), None)
    if loader is None:
        return rest, Memory()
    try:
        content = await loader.ainvoke({})
        # MCP tools return content blocks (probe 2026-09-28); plain langchain tools return the string.
        text = content if isinstance(content, str) else "".join(b.get("text", "") for b in content)
        data = json.loads(text)
        return rest, Memory(data.get("user", ""), data.get("soul", ""))
    except Exception:
        log.warning("could not load memory; answering without it", exc_info=True)
        return rest, Memory()
