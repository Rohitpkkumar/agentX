"""Basic file and shell tools for local agent mode."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool


def _ws() -> Path:
    return Path(os.environ.get("AGENT_PROJECT_ROOT", os.getcwd())).resolve()


def _safe(path: str) -> Path:
    ws = _ws()
    resolved = (ws / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not str(resolved).startswith(str(ws)):
        raise ValueError(f"Path outside workspace: {path}")
    return resolved


@tool
def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        return _safe(path).read_text(errors="replace")
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file with the given content."""
    try:
        p = _safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written: {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace an exact string in a file."""
    try:
        p = _safe(path)
        text = p.read_text(errors="replace")
        if old_string not in text:
            return f"Error: string not found in {path}"
        p.write_text(text.replace(old_string, new_string, 1))
        return f"Edited: {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def list_dir(path: str = ".") -> str:
    """List files and directories at a path."""
    try:
        p = _safe(path)
        items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        return "\n".join(
            f"{'  ' if item.is_file() else '/'}{item.name}" for item in items
        )
    except Exception as e:
        return f"Error: {e}"


@tool
def run_shell(command: str) -> str:
    """Run a shell command in the workspace directory."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(_ws()),
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = result.stdout + result.stderr
        return out[:8000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120s"
    except Exception as e:
        return f"Error: {e}"


@tool
def search_code(query: str) -> str:
    """Search for a string across all files in the workspace using grep."""
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts",
             "--include=*.go", "--include=*.java", "--include=*.cpp", "--include=*.c",
             "--include=*.rs", "--include=*.yaml", "--include=*.yml", "--include=*.json",
             "-l", query, "."],
            cwd=str(_ws()),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout[:4000] or "No matches found."
    except Exception as e:
        return f"Error: {e}"


ALL_TOOLS = [read_file, write_file, edit_file, list_dir, run_shell, search_code]
