"""The person's USER.md and SOUL.md from uriel-tools, loaded once per turn into the system prompt."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from uriel.agent.mcp_tools import MEMORY_TOOL, call_json

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Memory:
    user: str = ""
    soul: str = ""
    now: str = ""  # the person's local time, e.g. "Tue 29 Sep 2026, 22:41 (America/Toronto)"

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


async def load_memory(tools: Sequence[BaseTool]) -> Memory:
    """Call memory_context: the gateway loads memory, the model never does (uriel-tools marks the tool hidden,
    and `visible` keeps it from the model). It goes through the same identity interceptor as any tool call."""
    loader = next((t for t in tools if t.name == MEMORY_TOOL), None)
    if loader is None:
        return Memory()
    try:
        data = await call_json(loader, {}) or {}
        now = f"{data['now_local']} ({data['tz']})" if data.get("now_local") and data.get("tz") else ""
        return Memory(data.get("user", ""), data.get("soul", ""), now)
    except Exception:
        log.warning("could not load memory; answering without it", exc_info=True)
        return Memory()
