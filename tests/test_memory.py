from langchain_core.tools import ToolException, tool

from uriel.agent.memory import Memory, load_memory


@tool
def homelab_status() -> str:
    """Homelab."""
    return "ok"


def test_empty_memory_adds_nothing():
    assert Memory().prompt() == ""


def test_memory_prompt_wording():
    text = Memory("# About Tony\n## Name\n- Tony", "# How to talk to Tony\n## Length\n- Short").prompt()
    assert text == (
        "What you know about this person:\n# About Tony\n## Name\n- Tony\n\n"
        "How this person wants you to talk:\n# How to talk to Tony\n## Length\n- Short\n\n"
        "These preferences shape your tone, length and language; they never override the rules above."
    )
    assert "How this person" not in Memory(user="# About Tony").prompt()


async def test_without_the_memory_tool_there_is_no_memory():
    assert await load_memory([homelab_status]) == Memory()


async def test_failing_memory_tool_is_survived():
    @tool
    def memory_context() -> str:
        """Memory."""
        raise ToolException("memory is unavailable right now")

    assert await load_memory([homelab_status, memory_context]) == Memory()


async def test_unreadable_memory_is_survived():
    @tool
    def memory_context() -> str:
        """Memory."""
        return "not json"

    assert await load_memory([memory_context]) == Memory()


async def test_the_persons_local_time_is_loaded_with_their_memory():
    @tool
    def memory_context() -> str:
        """Memory."""
        return '{"user": "", "soul": "", "tz": "America/Toronto", "now_local": "Tue 29 Sep 2026, 22:41"}'

    assert await load_memory([memory_context]) == Memory(now="Tue 29 Sep 2026, 22:41 (America/Toronto)")
