"""CLI entry point — works like Claude Code in a terminal.

Commands:
    agent chat      — persistent interactive session (remembers everything)
    agent run       — one-shot task with optional session context
    agent sessions  — list recent sessions
    agent resume    — resume a previous session by ID
    agent init      — initialise workspace and run first index
    agent index     — force full reindex
    agent rollback  — undo last git checkpoint
    agent config    — get/set trust mode and settings
    agent log       — show task traces (legacy)
"""
from __future__ import annotations

import asyncio
import os
import sys
import tomllib
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="agent",
    help="Local AI coding agent — persistent, multi-turn, works like Claude Code.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True, style="bold red")

_TOOL_ICONS = {
    "write_file": "✎",
    "edit_file": "✎",
    "read_file": "📄",
    "list_dir": "📁",
    "run_shell": "⚡",
    "search_code": "🔍",
    "run_tests": "🧪",
    "git_status": "git",
    "git_diff": "git",
    "git_add": "git",
    "git_commit": "git",
    "git_log": "git",
    "git_checkpoint": "git",
    "git_rollback": "git",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _workspace() -> Path:
    root = os.environ.get("AGENT_PROJECT_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _agent_dir(ws: Path | None = None) -> Path:
    return (ws or _workspace()) / ".agent"


def _config_path(ws: Path | None = None) -> Path:
    return _agent_dir(ws) / "config.toml"


def _load_config(ws: Path | None = None) -> dict[str, Any]:
    p = _config_path(ws)
    return tomllib.loads(p.read_text()) if p.exists() else {}


def _save_config(cfg: dict[str, Any], ws: Path | None = None) -> None:
    import tomli_w  # type: ignore[import]
    p = _config_path(ws)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(tomli_w.dumps(cfg).encode())


def _get_history(ws: Path) -> Any:
    from agent.core.history import ConversationHistory
    _agent_dir(ws).mkdir(parents=True, exist_ok=True)
    return ConversationHistory(_agent_dir(ws) / "state.db")


def _get_logger(ws: Path) -> Any:
    from agent.observability.logger import EventLogger
    return EventLogger(_agent_dir(ws) / "logs.db")


def _short_id(sid: str) -> str:
    return sid[:8]


# ---------------------------------------------------------------------------
# Rich display helpers
# ---------------------------------------------------------------------------

def _print_tool_start(name: str, args: dict[str, Any]) -> None:
    icon = _TOOL_ICONS.get(name, "🔧")
    hint = (
        args.get("command")
        or args.get("path")
        or args.get("query")
        or ""
    )
    if len(hint) > 72:
        hint = hint[:69] + "..."
    label = f"{icon} {name}"
    if hint:
        label += f"  [dim]{hint}[/]"
    console.print(f"  {label}")


def _print_tool_end(name: str, output: str, ok: bool) -> None:
    if not ok:
        preview = output[:200]
        console.print(f"    [red]✗ Error:[/] {preview}")
    else:
        # Show a compact preview for shell output
        lines = [l for l in output.splitlines() if l.strip()]
        if lines and name == "run_shell":
            preview = lines[-1][:120] if lines else ""
            if preview:
                console.print(f"    [dim]↳ {preview}[/]")


def _print_response(text: str) -> None:
    """Render the model's text response as markdown."""
    if text.strip():
        console.print()
        try:
            console.print(Markdown(text))
        except Exception:
            console.print(text)
        console.print()


def _session_banner(ws: Path, trust: str, session_id: str, resumed: bool = False) -> None:
    action = "Resumed" if resumed else "New session"
    console.print(Panel(
        f"[bold]Local coding agent[/]   [dim]({action})[/]\n"
        f"Workspace : [cyan]{ws}[/]\n"
        f"Trust     : [cyan]{trust}[/]\n"
        f"Session   : [dim]{session_id}[/]\n\n"
        "Type your task. [dim]Type [bold]exit[/] to quit, [bold]sessions[/] to list history.[/]",
        title="agent chat",
        border_style="dim",
    ))


# ---------------------------------------------------------------------------
# Core async run
# ---------------------------------------------------------------------------

async def _do_turn(
    user_message: str,
    ws: Path,
    session_id: str,
    trust: str,
    history: Any,
) -> Any:
    from agent.core.loop import run_turn

    result = await run_turn(
        user_message,
        workspace=ws,
        session_id=session_id,
        history=history,
        trust=trust,  # type: ignore[arg-type]
        on_content=_print_response,
        on_tool_start=_print_tool_start,
        on_tool_end=_print_tool_end,
    )
    return result


# ---------------------------------------------------------------------------
# agent chat
# ---------------------------------------------------------------------------

@app.command()
def chat(
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
    trust: Annotated[Optional[str], typer.Option("--trust")] = None,
    session: Annotated[Optional[str], typer.Option("--session", "-s", help="Resume a session ID")] = None,
) -> None:
    """Persistent interactive session — remembers everything across turns."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    cfg = _load_config(ws)
    trust_mode = trust or cfg.get("trust_mode", "trusted")
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    os.environ["AGENT_TRUST_MODE"] = trust_mode

    history = _get_history(ws)
    resumed = False

    try:
        if session:
            info = history.get_session(session)
            if not info:
                err_console.print(f"Session {session!r} not found.")
                raise typer.Exit(1)
            session_id = session
            resumed = True
        else:
            session_id = history.create_session(str(ws))

        _session_banner(ws, trust_mode, session_id, resumed)

        if resumed:
            prior = history.load(session_id)
            msg_count = len([m for m in prior if m.__class__.__name__ == "HumanMessage"])
            console.print(f"[dim]Loaded {msg_count} previous message(s).[/]\n")

        while True:
            try:
                user_input = typer.prompt("", prompt_suffix="[you] > ")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Session saved. Resume with:[/]")
                console.print(f"  agent chat --session {session_id}")
                break

            text = user_input.strip()
            if not text:
                continue
            if text.lower() in {"exit", "quit", "bye"}:
                console.print(f"\n[dim]Session saved. Resume with:[/]")
                console.print(f"  agent chat --session {session_id}")
                break
            if text.lower() == "sessions":
                _print_sessions(history)
                continue

            # Auto-title session from first message
            if history.message_count(session_id) == 0:
                title = text[:60]
                history.update_title(session_id, title)

            console.print(Rule(style="dim"))
            try:
                result = asyncio.run(_do_turn(text, ws, session_id, trust_mode, history))
                if result.files_changed:
                    console.print("[dim]Files changed:[/]")
                    for f in result.files_changed:
                        console.print(f"  [green]✓[/] {f}")
                    console.print()
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/]")
            except Exception as exc:
                err_console.print(f"Error: {exc}")
    finally:
        history.close()


# ---------------------------------------------------------------------------
# agent run (one-shot with session context)
# ---------------------------------------------------------------------------

@app.command()
def run(
    request: Annotated[str, typer.Argument(help="Coding task to execute")],
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
    trust: Annotated[Optional[str], typer.Option("--trust")] = None,
    session: Annotated[Optional[str], typer.Option("--session", "-s", help="Use existing session for context")] = None,
) -> None:
    """Execute a one-shot coding task."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    cfg = _load_config(ws)
    trust_mode = trust or cfg.get("trust_mode", "trusted")
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    os.environ["AGENT_TRUST_MODE"] = trust_mode

    history = _get_history(ws)
    try:
        if session:
            info = history.get_session(session)
            if not info:
                err_console.print(f"Session {session!r} not found.")
                raise typer.Exit(1)
            session_id = session
        else:
            session_id = history.create_session(str(ws), title=request[:60])

        console.print(Panel(f"[bold]{request}[/]", title="Task", border_style="dim"))
        console.print()

        try:
            result = asyncio.run(_do_turn(request, ws, session_id, trust_mode, history))

            if result.files_changed:
                console.print("[bold]Files changed:[/]")
                for f in result.files_changed:
                    console.print(f"  [green]✓[/] {f}")
                console.print()

            console.print(f"[dim]Session ID: {session_id}[/]")
            console.print(f"[dim]Continue with: agent chat --session {session_id}[/]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
            console.print(f"[dim]Resume: agent chat --session {session_id}[/]")
            raise typer.Exit(130)
        except Exception as exc:
            err_console.print(f"Task failed: {exc}")
            raise typer.Exit(1)
    finally:
        history.close()


# ---------------------------------------------------------------------------
# agent sessions
# ---------------------------------------------------------------------------

@app.command()
def sessions(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
) -> None:
    """List recent conversation sessions."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    history = _get_history(ws)
    try:
        _print_sessions(history, limit)
    finally:
        history.close()


def _print_sessions(history: Any, limit: int = 20) -> None:
    rows = history.list_sessions(limit)
    if not rows:
        console.print("[dim]No sessions yet.[/]")
        return
    table = Table(title="Recent sessions", show_header=True, border_style="dim")
    table.add_column("ID", style="dim", no_wrap=True, width=10)
    table.add_column("Title", max_width=55)
    table.add_column("Updated", width=20)
    for r in rows:
        sid = r.get("id", "?")
        title = r.get("title") or "[dim](untitled)[/]"
        updated = (r.get("updated_at") or "")[:16].replace("T", " ")
        table.add_row(_short_id(sid), title, updated)
    console.print(table)
    console.print("[dim]Resume a session: agent chat --session <ID>[/]")


# ---------------------------------------------------------------------------
# agent resume (alias for chat --session)
# ---------------------------------------------------------------------------

@app.command()
def resume(
    session_id: Annotated[str, typer.Argument(help="Session ID to resume (8+ chars)")],
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
    trust: Annotated[Optional[str], typer.Option("--trust")] = None,
) -> None:
    """Resume a previous session by its ID."""
    # Find full ID if short prefix given
    ws = Path(workspace).resolve() if workspace else _workspace()
    history = _get_history(ws)
    try:
        all_sess = history.list_sessions(100)
        matches = [s for s in all_sess if s["id"].startswith(session_id)]
        if not matches:
            err_console.print(f"No session starting with {session_id!r}")
            raise typer.Exit(1)
        full_id = matches[0]["id"]
    finally:
        history.close()

    # Delegate to chat with --session
    ctx = typer.get_current_context()
    ctx.invoke(
        chat,
        workspace=workspace,
        trust=trust,
        session=full_id,
    )


# ---------------------------------------------------------------------------
# agent init
# ---------------------------------------------------------------------------

@app.command()
def init(
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
    trust: Annotated[str, typer.Option("--trust")] = "trusted",
) -> None:
    """Initialise .agent/ and run first index."""
    ws = Path(workspace).resolve() if workspace else Path.cwd().resolve()
    agent_dir = _agent_dir(ws)
    agent_dir.mkdir(parents=True, exist_ok=True)

    cfg: dict[str, Any] = {"trust_mode": trust, "project_root": str(ws)}
    try:
        _save_config(cfg, ws)
    except ImportError:
        _config_path(ws).write_text(f'trust_mode = "{trust}"\nproject_root = "{ws}"\n')

    console.print(Panel(f"[green]Initialised[/] .agent/ in {ws}", title="agent init"))

    try:
        from agent.memory.project import ProjectStore
        store = ProjectStore(agent_dir / "state.db")
        facts = store.detect_and_persist(ws)
        if facts:
            console.print(f"Detected {len(facts)} project convention(s):")
            for f in facts:
                console.print(f"  [cyan]{f.key}[/]: {f.value}")
    except Exception as exc:
        console.print(f"[yellow]Convention detection skipped:[/] {exc}")

    console.print("Indexing project files…")
    try:
        os.environ["AGENT_PROJECT_ROOT"] = str(ws)
        asyncio.run(_do_index(ws, agent_dir))
        console.print("[green]Index complete.[/]")
    except Exception as exc:
        console.print(f"[yellow]Index skipped:[/] {exc}")

    console.print("\nRun [bold]agent chat[/] to start a session.")


async def _do_index(workspace: Path, agent_dir: Path) -> None:
    from agent.code_index import index_project
    await index_project(workspace, agent_dir)


# ---------------------------------------------------------------------------
# agent index
# ---------------------------------------------------------------------------

@app.command()
def index(
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
) -> None:
    """Force a full reindex of the project."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    agent_dir = _agent_dir(ws)
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    console.print("Reindexing…")
    try:
        asyncio.run(_do_index(ws, agent_dir))
        console.print("[green]Done.[/]")
    except Exception as exc:
        err_console.print(f"Index failed: {exc}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# agent rollback
# ---------------------------------------------------------------------------

@app.command()
def rollback(
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
) -> None:
    """Undo last task via git checkpoint."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    from agent.tools.git import git_rollback
    result = git_rollback.invoke({})
    if "error" in str(result).lower() or "fail" in str(result).lower():
        err_console.print(str(result))
        raise typer.Exit(1)
    console.print(f"[green]Rolled back:[/] {result}")


# ---------------------------------------------------------------------------
# agent log (legacy)
# ---------------------------------------------------------------------------

@app.command(name="log")
def log_cmd(
    task: Annotated[Optional[str], typer.Option("--task", "-t")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
) -> None:
    """Show recent task traces (legacy observability log)."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    logger = _get_logger(ws)
    try:
        if task:
            task_data = logger.get_task(task)
            if task_data is None:
                err_console.print(f"Task {task!r} not found.")
                raise typer.Exit(1)
            events = logger.get_events(task)
            from agent.observability.tracer import format_trace
            console.print(format_trace(task_data, events))
        else:
            tasks = logger.list_tasks(limit=limit)
            if not tasks:
                console.print("No tasks logged yet.")
                return
            table = Table(title="Recent tasks", show_header=True)
            table.add_column("Task ID", style="dim", no_wrap=True)
            table.add_column("Request", max_width=50)
            table.add_column("Outcome")
            for t in tasks:
                colour = {"success": "green", "failure": "red", "partial": "yellow"}.get(
                    t.get("outcome", ""), "white"
                )
                table.add_row(
                    (t.get("task_id") or "?")[:12],
                    (t.get("request") or "")[:50],
                    f"[{colour}]{t.get('outcome', '?')}[/{colour}]",
                )
            console.print(table)
    finally:
        logger.close()


# ---------------------------------------------------------------------------
# agent config
# ---------------------------------------------------------------------------

@app.command(name="config")
def config_cmd(
    key: Annotated[Optional[str], typer.Argument()] = None,
    value: Annotated[Optional[str], typer.Argument()] = None,
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
) -> None:
    """Get or set agent configuration values."""
    ws = Path(workspace).resolve() if workspace else _workspace()
    cfg = _load_config(ws)

    if key is None:
        if not cfg:
            console.print("No configuration found. Run [bold]agent init[/] first.")
            return
        table = Table(title=f"Config ({_config_path(ws)})")
        table.add_column("Key")
        table.add_column("Value")
        for k, v in cfg.items():
            table.add_row(k, str(v))
        console.print(table)
        return

    if value is None:
        if key not in cfg:
            err_console.print(f"Key {key!r} not found.")
            raise typer.Exit(1)
        console.print(cfg[key])
        return

    cfg[key] = value
    try:
        _save_config(cfg, ws)
    except ImportError:
        lines = "\n".join(f'{k} = "{v}"' for k, v in cfg.items())
        _config_path(ws).write_text(lines + "\n")
    console.print(f"[green]Set[/] {key} = {value}")


# ---------------------------------------------------------------------------
# agent serve — HTTP API server for remote thin-client access
# ---------------------------------------------------------------------------

@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p")] = 8080,
    workspace: Annotated[Optional[str], typer.Option("--workspace", "-w")] = None,
    trust: Annotated[str, typer.Option("--trust")] = "trusted",
) -> None:
    """Start the agent as an HTTP API server (for remote thin-client access)."""
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        err_console.print(
            "fastapi/uvicorn not installed. Run: pip install 'local-coding-agent[serve]'"
        )
        raise typer.Exit(1)

    ws = Path(workspace).resolve() if workspace else _workspace()
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    os.environ["AGENT_TRUST_MODE"] = trust

    console.print(Panel(
        f"[bold]Agent API server[/]\n"
        f"Workspace : [cyan]{ws}[/]\n"
        f"Trust     : [cyan]{trust}[/]\n"
        f"Listening : [cyan]http://{host}:{port}[/]\n\n"
        "Connect from any machine on the same network with [bold]agentX[/].",
        title="agent serve",
        border_style="dim",
    ))

    import uvicorn
    uvicorn.run("agent.server:app", host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# agentX entry point — type `agentX` in any directory to start coding
# ---------------------------------------------------------------------------

def agentx_main() -> None:
    """Entry point for the `agentX` command.

    Usage:
        agentX                    → interactive chat in current directory
        agentX "fix the bug"      → one-shot task
        agentX sessions           → list saved sessions
        agentX resume <id>        → resume a previous session
    """
    import sys

    args = sys.argv[1:]
    ws = _workspace()

    # Ensure .agent/ exists silently (no init ceremony needed)
    _agent_dir(ws).mkdir(parents=True, exist_ok=True)

    if not args:
        # Just `agentX` → start interactive chat
        _agentx_chat(ws)
    elif args[0] == "sessions":
        history = _get_history(ws)
        try:
            _print_sessions(history)
        finally:
            history.close()
    elif args[0] == "resume" and len(args) >= 2:
        _agentx_chat(ws, session_id=args[1])
    elif args[0] in {"help", "--help", "-h"}:
        console.print(Panel(
            "[bold]agentX[/] — AI coding agent\n\n"
            "  [bold]agentX[/]                   start interactive session\n"
            "  [bold]agentX[/] [italic]\"do something\"[/]   one-shot task\n"
            "  [bold]agentX sessions[/]           list saved sessions\n"
            "  [bold]agentX resume[/] [italic]<id>[/]       resume a session\n\n"
            "LLM config:\n"
            "  OLLAMA_URL=http://server:11434   (default, Ollama)\n"
            "  LLM_PROVIDER=groq GROQ_API_KEY=sk-...   (Groq cloud)",
            title="agentX help",
            border_style="dim",
        ))
    else:
        # Treat args as a one-shot task
        task = " ".join(args)
        _agentx_run(ws, task)


def _agentx_chat(ws: Path, session_id: str | None = None) -> None:
    """Interactive multi-turn session — the core agentX experience."""
    cfg = _load_config(ws)
    trust_mode = cfg.get("trust_mode", "trusted")
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    os.environ["AGENT_TRUST_MODE"] = trust_mode

    history = _get_history(ws)
    resumed = False

    try:
        if session_id:
            # Expand short prefix to full ID
            all_sess = history.list_sessions(100)
            matches = [s for s in all_sess if s["id"].startswith(session_id)]
            if not matches:
                err_console.print(f"No session starting with {session_id!r}")
                return
            session_id = matches[0]["id"]
            resumed = True
        else:
            session_id = history.create_session(str(ws))

        _session_banner(ws, trust_mode, session_id, resumed)

        if resumed:
            prior = history.load(session_id)
            msg_count = len([m for m in prior if m.__class__.__name__ == "HumanMessage"])
            console.print(f"[dim]Loaded {msg_count} previous message(s).[/]\n")

        while True:
            try:
                user_input = input("[you] > ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print(f"\n[dim]Resume with: agentX resume {session_id[:8]}[/]")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "bye"}:
                console.print(f"[dim]Resume with: agentX resume {session_id[:8]}[/]")
                break
            if user_input.lower() == "sessions":
                _print_sessions(history)
                continue

            if history.message_count(session_id) == 0:
                history.update_title(session_id, user_input[:60])

            console.print(Rule(style="dim"))
            try:
                result = asyncio.run(_do_turn(user_input, ws, session_id, trust_mode, history))
                if result.files_changed:
                    console.print("[dim]Files changed:[/]")
                    for f in result.files_changed:
                        console.print(f"  [green]✓[/] {f}")
                    console.print()
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/]")
            except Exception as exc:
                err_console.print(f"Error: {exc}")
    finally:
        history.close()


def _agentx_run(ws: Path, task: str) -> None:
    """One-shot task — run a single request and exit."""
    cfg = _load_config(ws)
    trust_mode = cfg.get("trust_mode", "trusted")
    os.environ["AGENT_PROJECT_ROOT"] = str(ws)
    os.environ["AGENT_TRUST_MODE"] = trust_mode

    history = _get_history(ws)
    try:
        session_id = history.create_session(str(ws), title=task[:60])
        console.print(Panel(f"[bold]{task}[/]", title="agentX", border_style="dim"))
        console.print()
        try:
            result = asyncio.run(_do_turn(task, ws, session_id, trust_mode, history))
            if result.files_changed:
                console.print("[bold]Files changed:[/]")
                for f in result.files_changed:
                    console.print(f"  [green]✓[/] {f}")
                console.print()
            console.print(f"[dim]Continue: agentX resume {session_id[:8]}[/]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/]")
        except Exception as exc:
            err_console.print(f"Task failed: {exc}")
    finally:
        history.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
