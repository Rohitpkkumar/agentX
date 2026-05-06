"""Local ReAct loop — LLM on remote Ollama, file ops on local machine."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agentx.history import History
from agentx.tools import ALL_TOOLS

_MAX_ITER = 40

_SYSTEM = """\
You are an expert AI coding assistant running in a local terminal.
You help with any software engineering task: reading, writing, refactoring, debugging, and testing code.

WORKSPACE: {workspace}

## Tools available
- read_file(path)              → read any file
- write_file(path, content)    → create or overwrite a file
- edit_file(path, old_string, new_string) → replace exact string in a file
- list_dir(path)               → list directory contents
- run_shell(command)           → run any shell command
- search_code(query)           → search across all project files

## Rules
- ALWAYS read a file before editing it.
- Use write_file to CREATE new files. Use edit_file to MODIFY existing files.
- Use run_shell for installs, tests, builds — NOT for writing files.
- After making changes, verify they work.
- Only work within the workspace directory.
- Produce working, runnable code.
"""


def _build_model():
    from langchain_ollama import ChatOllama
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("CHAT_MODEL", "qwen3-coder:30b")
    extra: dict[str, Any] = {}
    if "qwen3" in model.lower():
        extra["think"] = False
    return ChatOllama(base_url=url, model=model, temperature=0, **extra)


async def run_turn(
    user_message: str,
    *,
    workspace: Path,
    session_id: str,
    history: History,
    on_content: Callable[[str], None] | None = None,
    on_tool_start: Callable[[str, dict], None] | None = None,
    on_tool_end: Callable[[str, str, bool], None] | None = None,
) -> list[str]:
    """Run one turn. Returns list of changed file paths."""
    os.environ["AGENT_PROJECT_ROOT"] = str(workspace)

    sys_msg = SystemMessage(content=_SYSTEM.format(workspace=workspace))
    prior_data = history.load(session_id)
    prior: list[BaseMessage] = []
    for m in prior_data:
        if m["role"] == "human":
            prior.append(HumanMessage(content=m["content"]))
        elif m["role"] == "ai":
            prior.append(AIMessage(content=m["content"], tool_calls=m["extra"].get("tool_calls", [])))
        elif m["role"] == "tool":
            prior.append(ToolMessage(content=m["content"], tool_call_id=m["extra"].get("tool_call_id", ""), name=m["extra"].get("name", "")))

    human_msg = HumanMessage(content=user_message)
    messages: list[BaseMessage] = [sys_msg, *prior, human_msg]
    history.append(session_id, "human", user_message)

    model = _build_model().bind_tools(ALL_TOOLS)
    tool_map = {t.name: t for t in ALL_TOOLS}

    files_changed: list[str] = []

    for _ in range(_MAX_ITER):
        response: AIMessage = await model.ainvoke(messages)  # type: ignore[assignment]
        messages.append(response)

        text = response.content if isinstance(response.content, str) else ""
        if text and on_content:
            on_content(text)

        if not response.tool_calls:
            history.append(session_id, "ai", text)
            break

        history.append(session_id, "ai", text, {"tool_calls": response.tool_calls})

        for tc in response.tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("args") or {}
            call_id = tc.get("id", "")

            if on_tool_start:
                on_tool_start(name, args)

            if name in tool_map:
                try:
                    output = str(tool_map[name].invoke(args))
                    ok = True
                except Exception as e:
                    output = f"Error: {e}"
                    ok = False
            else:
                output = f"Unknown tool: {name}"
                ok = False

            if on_tool_end:
                on_tool_end(name, output, ok)

            if name in {"write_file", "edit_file"} and ok:
                path_arg = args.get("path", "")
                if path_arg and path_arg not in files_changed:
                    files_changed.append(path_arg)

            tm = ToolMessage(content=output, tool_call_id=call_id, name=name)
            messages.append(tm)
            history.append(session_id, "tool", output, {"tool_call_id": call_id, "name": name})

    return files_changed
