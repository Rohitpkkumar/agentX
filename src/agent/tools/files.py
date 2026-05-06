from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool

from agent.safety.policy import is_path_allowed


def _project_root() -> Path:
    root = os.environ.get("AGENT_PROJECT_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _validated_path(path: str) -> Path:
    """Resolve path and verify it lives inside the project root."""
    allowed, reason = is_path_allowed(path, _project_root())
    if not allowed:
        raise PermissionError(reason)
    return Path(path).resolve()


@tool  # type: ignore[misc]
def read_file(path: str) -> str:
    """Read the complete UTF-8 contents of a file.

    The path must be within the project root. Raises FileNotFoundError if the
    file does not exist and IsADirectoryError if the path points to a directory.
    """
    p = _validated_path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path!r}")
    if not p.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {path!r}")
    return p.read_text(encoding="utf-8")


@tool  # type: ignore[misc]
def write_file(path: str, content: str) -> str:
    """Write UTF-8 content to a file, creating parent directories as needed.

    Overwrites the file if it already exists.
    The path must be within the project root.
    """
    p = _validated_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path!r}"


@tool  # type: ignore[misc]
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace exactly one occurrence of old_string with new_string in a file.

    Raises ValueError if old_string is not found or appears more than once —
    provide a longer, more distinctive search string in the latter case.
    The path must be within the project root.
    """
    p = _validated_path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path!r}")

    content = p.read_text(encoding="utf-8")
    count = content.count(old_string)

    if count == 0:
        raise ValueError(
            f"Search string not found in {path!r}. No changes made.\n"
            f"Search string was: {old_string!r}"
        )
    if count > 1:
        raise ValueError(
            f"Search string appears {count} times in {path!r}. "
            "Provide a longer, more distinctive search string."
        )

    new_content = content.replace(old_string, new_string, 1)
    p.write_text(new_content, encoding="utf-8")
    return f"Replaced 1 occurrence in {path!r}"


@tool  # type: ignore[misc]
def list_dir(path: str) -> str:
    """List the entries in a directory.

    Returns one line per entry, prefixed with [F] for files and [D] for
    directories. The path must be within the project root.
    """
    p = _validated_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path!r}")
    if not p.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path!r}")

    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
    if not entries:
        return "(empty directory)"

    lines = [f"[{'F' if e.is_file() else 'D'}] {e.name}" for e in entries]
    return "\n".join(lines)
