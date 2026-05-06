"""Core ReAct agent loop — the engine that powers every conversation turn.

Architecture (same as Claude Code):
  1. System prompt + full conversation history + new user message → LLM
  2. LLM responds with text and/or tool calls
  3. Tool calls are executed immediately; results appended to conversation
  4. Repeat from step 1 until LLM returns a response with no tool calls
  5. All messages are persisted to SQLite so the next turn has full context

This replaces the LangGraph plan→act→verify pipeline with a simple while loop
that is more reliable, faster to iterate on, and closer to how Claude Code works.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agent.core.history import ConversationHistory
from agent.llm.chat import chat_model
from agent.memory.project import ProjectStore
from agent.safety.policy import TrustMode
from agent.tools.registry import all_tools, dispatch

_MAX_ITERATIONS = 40  # absolute safety cap


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert AI coding assistant running in a local terminal — similar to Claude Code.
You help with any software engineering task: reading, writing, refactoring, debugging, and testing code.

WORKSPACE: {project_root}
TRUST MODE: {trust_mode}

## Your tools
- read_file(path)          → read any file in the workspace
- write_file(path, content) → create or overwrite a file (creates parent dirs automatically)
- edit_file(path, old_string, new_string) → replace an exact string in a file
- list_dir(path)           → list directory contents
- run_shell(command)       → run any shell command (pip install, pytest, npm, git, etc.)
- search_code(query)       → ripgrep search across all files
- run_tests(path)          → run test suite for a file or directory
- git_status / git_diff / git_add / git_commit / git_log → git operations

## How to work
1. ALWAYS read a file before editing it — use read_file first.
2. Use write_file to CREATE new files with full content.
3. Use edit_file to MODIFY an existing file — read it first to get exact strings.
4. Use run_shell for package installs, running code, tests, builds.
5. After making changes, verify they work: run tests or run the code.
6. Be concise. Show actual code, not pseudocode.
7. When creating an app, create ALL necessary files so it can run immediately.

## Rules
- Only work within {project_root} — never access paths outside it.
- Do not ask for permission before using a tool — just use it.
- If a tool fails, read the error and fix it. Do not give up.
- Prefer editing existing files over rewriting them entirely.
- Always produce working, runnable code.

## Project conventions
{conventions}
"""


def _build_system(workspace: Path, trust: TrustMode) -> SystemMessage:
    agent_dir = workspace / ".agent"
    try:
        facts = ProjectStore(agent_dir / "state.db").all()
        conventions = "\n".join(f"  {f.key}: {f.value}" for f in facts) or "  none detected"
    except Exception:
        conventions = "  none detected"
    return SystemMessage(content=_SYSTEM.format(
        project_root=str(workspace),
        trust_mode=trust,
        conventions=conventions,
    ))


# ---------------------------------------------------------------------------
# Turn result
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    text: str
    files_changed: list[str] = field(default_factory=list)
    tool_calls_made: int = 0
    iterations: int = 0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_turn(
    user_message: str,
    *,
    workspace: Path,
    session_id: str,
    history: ConversationHistory,
    trust: TrustMode = "trusted",
    on_content: Callable[[str], None] | None = None,
    on_tool_start: Callable[[str, dict[str, Any]], None] | None = None,
    on_tool_end: Callable[[str, str, bool], None] | None = None,
) -> TurnResult:
    """Run one user turn of the ReAct loop.

    Persists every message (user, AI, tool results) to history so the
    next call to run_turn has full context.
    """
    workspace = workspace.resolve()
    os.environ["AGENT_PROJECT_ROOT"] = str(workspace)
    os.environ["AGENT_TRUST_MODE"] = trust

    # Build full message list for this turn
    sys_msg = _build_system(workspace, trust)
    prior: list[BaseMessage] = history.load(session_id)
    human_msg = HumanMessage(content=user_message)

    messages: list[BaseMessage] = [sys_msg, *prior, human_msg]

    # Persist user message
    history.append(session_id, human_msg)

    tools = all_tools()
    model = chat_model(temperature=0).bind_tools(tools)

    files_changed: list[str] = []
    tool_calls_made = 0
    final_text = ""
    iterations = 0

    for iterations in range(_MAX_ITERATIONS):
        response: AIMessage = await model.ainvoke(messages)  # type: ignore[assignment]
        messages.append(response)

        # Surface text content immediately
        text = response.content if isinstance(response.content, str) else ""
        if text:
            final_text = text
            if on_content:
                on_content(text)

        # No tool calls → model is done with this turn
        if not response.tool_calls:
            history.append(session_id, response)
            break

        # Execute every tool call in this response
        tool_msgs: list[ToolMessage] = []
        for tc in response.tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("args") or {}

            if on_tool_start:
                on_tool_start(name, args)

            result = dispatch(tc)
            ok = result.ok
            content = str(result.output) if ok else f"Error: {result.error}"

            if on_tool_end:
                on_tool_end(name, content, ok)

            tool_calls_made += 1

            # Track file writes
            if name in {"write_file", "edit_file"} and ok:
                path_arg = args.get("path", "")
                if path_arg and path_arg not in files_changed:
                    files_changed.append(path_arg)

            tm = ToolMessage(
                content=content,
                tool_call_id=tc.get("id", ""),
                name=name,
            )
            tool_msgs.append(tm)
            messages.append(tm)

        # Persist AI response + all tool results together
        history.append(session_id, response)
        history.append_many(session_id, tool_msgs)  # type: ignore[arg-type]

    return TurnResult(
        text=final_text,
        files_changed=files_changed,
        tool_calls_made=tool_calls_made,
        iterations=iterations + 1,
    )
