"""agentX — coding agent thin client (remote) or local agent (local).

Two modes:
  AGENT_SERVER set  → thin client, agent runs on server, server files
  OLLAMA_URL set    → local agent, runs on THIS machine, local files, LLM on server

Install:
    pip install "git+https://github.com/Rohitpkkumar/agentX.git#subdirectory=client"

Usage:
    agentX              → start chat
    agentX sessions     → list past sessions
    agentX resume <id>  → resume a session
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Run: pip install 'git+https://github.com/Rohitpkkumar/agentX.git#subdirectory=client'")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    console = Console()

    def render(text: str) -> None:
        if text.strip():
            console.print()
            try:
                console.print(Markdown(text))
            except Exception:
                console.print(text)
            console.print()
except ImportError:
    class _FakeConsole:  # type: ignore[no-redef]
        def print(self, *a, **k): print(*a)
    console = _FakeConsole()  # type: ignore[assignment]

    def render(text: str) -> None:
        print(text)

# ── config ────────────────────────────────────────────────────────────────────
_SERVER = os.environ.get("AGENT_SERVER", "").rstrip("/")
_OLLAMA_URL = os.environ.get("OLLAMA_URL", "")
_SESSION_FILE = Path.home() / ".agentX" / "session"
_LOCAL_DB = Path.home() / ".agentX" / "history.db"

_TOOL_ICONS = {
    "write_file": "✎", "edit_file": "✎", "read_file": "📄",
    "list_dir": "📁", "run_shell": "⚡", "search_code": "🔍",
    "run_tests": "🧪",
}


def _mode() -> str:
    if _SERVER:
        return "remote"
    if _OLLAMA_URL:
        return "local"
    return "none"


def _check_config() -> None:
    if _mode() == "none":
        console.print("[red]Error:[/] Neither AGENT_SERVER nor OLLAMA_URL is set.\n")
        console.print("  Remote mode (agent on server, server files):")
        console.print("    export AGENT_SERVER=\"http://server-ip:8080\"\n")
        console.print("  Local mode (agent on THIS machine, local files, LLM on server):")
        console.print("    export OLLAMA_URL=\"http://server-ip:11434\"")
        sys.exit(1)


def _load_session() -> str | None:
    try:
        return _SESSION_FILE.read_text().strip() or None
    except FileNotFoundError:
        return None


def _save_session(sid: str) -> None:
    _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SESSION_FILE.write_text(sid)


def _tool_line(name: str, args: dict) -> str:
    icon = _TOOL_ICONS.get(name, "🔧")
    hint = args.get("command") or args.get("path") or args.get("query") or ""
    if len(hint) > 70:
        hint = hint[:67] + "..."
    return f"  {icon} {name}  [dim]{hint}[/]" if hint else f"  {icon} {name}"


# ── remote mode ───────────────────────────────────────────────────────────────

def _remote_chat(message: str, session_id: str | None) -> str:
    url = f"{_SERVER}/chat"
    payload: dict = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    returned_sid = session_id or ""
    try:
        with httpx.stream("POST", url, json=payload, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "session":
                    returned_sid = event["session_id"]
                elif etype == "content":
                    render(event["data"])
                elif etype == "tool_start":
                    d = event["data"]
                    console.print(_tool_line(d["name"], d.get("args", {})))
                elif etype == "tool_end":
                    d = event["data"]
                    if not d.get("ok"):
                        console.print(f"    [red]✗[/] {(d.get('output') or '')[:200]}")
                    elif d["name"] == "run_shell":
                        lines = [l for l in (d.get("output") or "").splitlines() if l.strip()]
                        if lines:
                            console.print(f"    [dim]↳ {lines[-1][:100]}[/]")
                elif etype == "done":
                    returned_sid = event.get("session_id", returned_sid)
                    changed = event.get("files_changed", [])
                    if changed:
                        console.print("\n[dim]Files changed:[/]")
                        for f in changed:
                            console.print(f"  [green]✓[/] {f}")
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to {_SERVER}[/]")
        sys.exit(1)
    return returned_sid


def _remote_sessions() -> list[dict]:
    resp = httpx.get(f"{_SERVER}/sessions", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── local mode ────────────────────────────────────────────────────────────────

def _local_chat(message: str, session_id: str, workspace: Path) -> list[str]:
    from agentx.history import History
    from agentx.loop import run_turn

    history = History(_LOCAL_DB)

    def on_content(text: str) -> None:
        render(text)

    def on_tool_start(name: str, args: dict) -> None:
        console.print(_tool_line(name, args))

    def on_tool_end(name: str, output: str, ok: bool) -> None:
        if not ok:
            console.print(f"    [red]✗[/] {output[:200]}")
        elif name == "run_shell":
            lines = [l for l in output.splitlines() if l.strip()]
            if lines:
                console.print(f"    [dim]↳ {lines[-1][:100]}[/]")

    return asyncio.run(run_turn(
        message,
        workspace=workspace,
        session_id=session_id,
        history=history,
        on_content=on_content,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
    ))


def _local_sessions() -> list[dict]:
    from agentx.history import History
    return History(_LOCAL_DB).list_sessions()


# ── shared UI ─────────────────────────────────────────────────────────────────

def _print_sessions(sessions: list[dict]) -> None:
    if not sessions:
        console.print("[dim]No sessions yet.[/]")
        return
    table = Table(title="Sessions", show_header=True, border_style="dim")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Title", max_width=55)
    table.add_column("Updated", width=18)
    for s in sessions:
        sid = (s.get("session_id") or s.get("id") or "")[:8]
        title = s.get("title") or "[dim](untitled)[/]"
        updated = (s.get("updated_at") or "")[:16].replace("T", " ")
        table.add_row(sid, title, updated)
    console.print(table)
    console.print("[dim]Resume: agentX resume <id>[/]")


def _cmd_sessions() -> None:
    if _mode() == "remote":
        _print_sessions(_remote_sessions())
    else:
        _print_sessions(_local_sessions())


def _cmd_resume(sid_prefix: str) -> None:
    if _mode() == "remote":
        sessions = _remote_sessions()
        key = "session_id"
    else:
        sessions = _local_sessions()
        key = "id"
    matches = [s for s in sessions if (s.get(key) or "").startswith(sid_prefix)]
    if not matches:
        console.print(f"[red]No session starting with {sid_prefix!r}[/]")
        sys.exit(1)
    _cmd_chat(session_id=matches[0][key])


def _cmd_chat(session_id: str | None = None) -> None:
    _check_config()
    mode = _mode()
    sid = session_id or _load_session()
    resumed = sid is not None
    workspace = Path.cwd().resolve()

    if mode == "remote":
        try:
            health = httpx.get(f"{_SERVER}/health", timeout=5).json()
            ws_label = health.get("workspace", "server")
        except Exception:
            ws_label = "server (unknown)"
        mode_label = f"Remote  [dim]{_SERVER}[/]"
    else:
        ws_label = str(workspace)
        mode_label = f"Local   [dim]Ollama: {_OLLAMA_URL}[/]"

    if mode == "local" and not sid:
        from agentx.history import History
        history = History(_LOCAL_DB)
        sid = history.create_session()

    console.print(Panel(
        f"[bold]agentX[/]   [dim]({'Resumed' if resumed else 'New session'})[/]\n"
        f"Mode      : {mode_label}\n"
        f"Workspace : [cyan]{ws_label}[/]\n"
        f"Session   : [dim]{sid or 'new'}[/]\n\n"
        "Type your task. [dim]Type [bold]exit[/] to quit.[/]",
        title="agentX",
        border_style="dim",
    ))

    while True:
        try:
            user_input = input("[you] > ").strip()
        except (EOFError, KeyboardInterrupt):
            if sid:
                console.print(f"\n[dim]Resume: agentX resume {sid[:8]}[/]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            if sid:
                console.print(f"[dim]Resume: agentX resume {sid[:8]}[/]")
            break
        if user_input.lower() == "sessions":
            _cmd_sessions()
            continue

        # Auto-title from first message
        if mode == "local" and sid:
            from agentx.history import History
            h = History(_LOCAL_DB)
            if h.message_count(sid) == 0:
                h.update_title(sid, user_input[:60])

        console.print(Rule(style="dim"))

        if mode == "remote":
            sid = _remote_chat(user_input, sid)
        else:
            changed = _local_chat(user_input, sid, workspace)
            if changed:
                console.print("\n[dim]Files changed:[/]")
                for f in changed:
                    console.print(f"  [green]✓[/] {f}")

        if sid:
            _save_session(sid)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] == "chat":
        _check_config()
        _cmd_chat()
    elif args[0] == "sessions":
        _check_config()
        _cmd_sessions()
    elif args[0] == "resume" and len(args) >= 2:
        _check_config()
        _cmd_resume(args[1])
    else:
        console.print("[bold]agentX[/] — AI coding agent\n")
        console.print("Usage:")
        console.print("  agentX              → start chat")
        console.print("  agentX sessions     → list sessions")
        console.print("  agentX resume <id>  → resume a session")
        console.print()
        console.print("Set one of:")
        console.print("  [bold]AGENT_SERVER[/]=http://server-ip:8080   (remote mode, server files)")
        console.print("  [bold]OLLAMA_URL[/]=http://server-ip:11434     (local mode, YOUR files, LLM on server)")
        sys.exit(1)


if __name__ == "__main__":
    main()
