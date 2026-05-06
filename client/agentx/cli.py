"""agentX — thin client for the remote coding agent server."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Run: pip install agentx-client  or  pip install httpx rich")
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
_SESSION_FILE = Path.home() / ".agentX" / "session"

_TOOL_ICONS = {
    "write_file": "✎", "edit_file": "✎", "read_file": "📄",
    "list_dir": "📁", "run_shell": "⚡", "search_code": "🔍",
    "run_tests": "🧪",
}


def _check_server() -> None:
    if not _SERVER:
        console.print("[red]Error:[/] AGENT_SERVER is not set.")
        console.print("")
        console.print("  Mac/Linux — add to ~/.zshrc or ~/.bashrc:")
        console.print("    export AGENT_SERVER=\"http://your-server-ip:8080\"")
        console.print("")
        console.print("  Windows — set in PowerShell or System Environment Variables:")
        console.print("    $env:AGENT_SERVER = \"http://your-server-ip:8080\"")
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


def _chat(message: str, session_id: str | None) -> str:
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
        console.print("Make sure the server is running:  docker compose up agent-server -d")
        sys.exit(1)

    return returned_sid


def _cmd_sessions() -> None:
    try:
        resp = httpx.get(f"{_SERVER}/sessions", timeout=10)
        resp.raise_for_status()
        sessions = resp.json()
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)

    if not sessions:
        console.print("[dim]No sessions yet.[/]")
        return

    table = Table(title="Sessions", show_header=True, border_style="dim")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Title", max_width=55)
    table.add_column("Updated", width=18)
    for s in sessions:
        sid = s["session_id"][:8]
        title = s.get("title") or "[dim](untitled)[/]"
        updated = (s.get("updated_at") or "")[:16].replace("T", " ")
        table.add_row(sid, title, updated)
    console.print(table)
    console.print("[dim]Resume: agentX resume <id>[/]")


def _cmd_resume(sid_prefix: str) -> None:
    try:
        resp = httpx.get(f"{_SERVER}/sessions", timeout=10)
        resp.raise_for_status()
        sessions = resp.json()
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)

    matches = [s for s in sessions if s["session_id"].startswith(sid_prefix)]
    if not matches:
        console.print(f"[red]No session starting with {sid_prefix!r}[/]")
        sys.exit(1)

    _cmd_chat(session_id=matches[0]["session_id"])


def _cmd_chat(session_id: str | None = None) -> None:
    _check_server()
    sid = session_id or _load_session()
    resumed = sid is not None

    try:
        health = httpx.get(f"{_SERVER}/health", timeout=5).json()
        workspace = health.get("workspace", "unknown")
    except Exception:
        workspace = "unknown"

    console.print(Panel(
        f"[bold]Local coding agent[/]   [dim]({'Resumed' if resumed else 'New session'})[/]\n"
        f"Server    : [cyan]{_SERVER}[/]\n"
        f"Workspace : [cyan]{workspace}[/]\n"
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

        console.print(Rule(style="dim"))
        sid = _chat(user_input, sid)
        if sid:
            _save_session(sid)


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] == "chat":
        _check_server()
        _cmd_chat()
    elif args[0] == "sessions":
        _check_server()
        _cmd_sessions()
    elif args[0] == "resume" and len(args) >= 2:
        _check_server()
        _cmd_resume(args[1])
    else:
        console.print("[bold]agentX[/] — coding agent thin client\n")
        console.print("Usage:")
        console.print("  agentX              → start chat")
        console.print("  agentX sessions     → list sessions")
        console.print("  agentX resume <id>  → resume a session")
        console.print()
        console.print("Set [bold]AGENT_SERVER[/]=http://your-server-ip:8080")
        sys.exit(1)


if __name__ == "__main__":
    main()
