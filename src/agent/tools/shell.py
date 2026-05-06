from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import cast

from langchain_core.tools import tool

from agent.safety.policy import (
    NETWORK_COMMANDS,
    TrustMode,
    is_network_allowed,
    is_shell_allowed,
)

_DEFAULT_TIMEOUT = 120


def _get_trust() -> TrustMode:
    mode = os.environ.get("AGENT_TRUST_MODE", "trusted")
    if mode not in ("readonly", "trusted", "yolo"):
        return "trusted"
    return cast(TrustMode, mode)


def _get_project_root() -> Path:
    root = os.environ.get("AGENT_PROJECT_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _check_network(command: str, trust: TrustMode) -> None:
    """Raise PermissionError if the command invokes a network tool and trust != yolo."""
    allowed, reason = is_network_allowed(trust)
    if allowed:
        return
    first_token = command.strip().split()[0] if command.strip() else ""
    binary_name = Path(first_token).name
    if binary_name in NETWORK_COMMANDS:
        raise PermissionError(f"Network command {binary_name!r} blocked. {reason}")


@tool  # type: ignore[misc]
def run_shell(command: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Run a shell command in the project root and return combined stdout+stderr.

    Permanently blocked (all trust modes): rm -rf, dd, mkfs, shred, fdisk, parted,
    writes to /dev/.
    Blocked in readonly mode: any write-pattern command.
    Network commands (curl, wget, ssh, …) are blocked unless trust mode is yolo.
    Commands time out after `timeout` seconds (default 120).
    """
    trust = _get_trust()

    allowed, reason = is_shell_allowed(command, trust)
    if not allowed:
        raise PermissionError(f"Command blocked by policy: {reason}")

    _check_network(command, trust)

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_get_project_root()),
    )

    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(result.stderr)

    combined = "\n".join(parts).strip()

    if result.returncode != 0:
        return f"Exit code {result.returncode}:\n{combined}"
    return combined or "(no output)"
