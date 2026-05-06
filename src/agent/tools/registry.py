from __future__ import annotations

import time
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from agent.tools.files import edit_file, list_dir, read_file, write_file
from agent.tools.git import (
    git_add,
    git_checkpoint,
    git_commit,
    git_diff,
    git_log,
    git_rollback,
    git_status,
)
from agent.tools.search import search_code
from agent.tools.shell import run_shell
from agent.tools.tests import run_tests


class ToolResult(BaseModel):
    ok: bool
    output: str | dict[str, Any]
    error: str | None = None
    duration_ms: int


def all_tools() -> list[BaseTool]:
    """Return every registered tool as a LangChain BaseTool, ready for bind_tools()."""
    return [
        read_file,  # type: ignore[list-item]
        write_file,  # type: ignore[list-item]
        edit_file,  # type: ignore[list-item]
        list_dir,  # type: ignore[list-item]
        run_shell,  # type: ignore[list-item]
        search_code,  # type: ignore[list-item]
        run_tests,  # type: ignore[list-item]
        git_status,  # type: ignore[list-item]
        git_diff,  # type: ignore[list-item]
        git_add,  # type: ignore[list-item]
        git_commit,  # type: ignore[list-item]
        git_log,  # type: ignore[list-item]
        git_checkpoint,  # type: ignore[list-item]
        git_rollback,  # type: ignore[list-item]
    ]


def dispatch(tool_call: dict[str, Any]) -> ToolResult:
    """Execute a tool by name with the provided args dict.

    Returns a ToolResult wrapping ok/output/error/duration_ms.
    Unknown tool names or argument validation failures are returned as
    ToolResult(ok=False, ...) rather than raised.
    """
    tool_name: str = tool_call.get("name", "")
    tool_args: dict[str, Any] = tool_call.get("args", {}) or {}

    tool_map: dict[str, BaseTool] = {t.name: t for t in all_tools()}

    if tool_name not in tool_map:
        return ToolResult(
            ok=False,
            output="",
            error=f"Unknown tool: {tool_name!r}",
            duration_ms=0,
        )

    selected = tool_map[tool_name]
    start = time.monotonic()
    try:
        result = selected.invoke(tool_args)
        duration_ms = int((time.monotonic() - start) * 1000)
        return ToolResult(ok=True, output=result, duration_ms=duration_ms)
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return ToolResult(ok=False, output="", error=str(exc), duration_ms=duration_ms)
