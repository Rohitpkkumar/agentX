from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

TrustMode = Literal["readonly", "trusted", "yolo"]

# Word-boundary patterns for commands that are permanently denied in all trust modes.
_ALWAYS_DENY: list[tuple[re.Pattern[str], str]] = [
    # rm with recursive+force flag combination (-rf, -fr, -Rf, -rvf, etc.)
    (re.compile(r"\brm\b"), "rm_base"),
    (re.compile(r"\bdd\b"), "dd"),
    (re.compile(r"\bmkfs\b"), "mkfs"),
    (re.compile(r"\bshred\b"), "shred"),
    (re.compile(r"\bfdisk\b"), "fdisk"),
    (re.compile(r"\bparted\b"), "parted"),
    (re.compile(r"\bwipefs\b"), "wipefs"),
    # Writing directly to block/char devices
    (re.compile(r">\s*/dev/[^\s]"), "write_to_dev"),
]

_RF_FLAGS = re.compile(
    r"-[a-zA-Z]*[rR][a-zA-Z]*[fF]"  # -rf, -Rf, -rvf, ...
    r"|-[a-zA-Z]*[fF][a-zA-Z]*[rR]"  # -fr, -Fr, -fvr, ...
    r"|--recursive\b.*--force\b"
    r"|--force\b.*--recursive\b"
)

# Patterns that indicate write operations — blocked in readonly mode.
_WRITE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\becho\b.*>"),
    re.compile(r"\btee\b"),
    re.compile(r"\bsed\b.*-i"),
    re.compile(r"\bcp\b"),
    re.compile(r"\bmv\b"),
    re.compile(r"\bmkdir\b"),
    re.compile(r"\btouch\b"),
    re.compile(r"\bchmod\b"),
    re.compile(r"\bchown\b"),
    re.compile(r"\bgit\s+(commit|push|merge|rebase|reset|clean)\b"),
    re.compile(r">[^>]"),  # any single-arrow redirect (write/create)
]

# Network commands blocked unless trust == "yolo".
NETWORK_COMMANDS: frozenset[str] = frozenset(
    [
        "curl",
        "wget",
        "nc",
        "netcat",
        "ssh",
        "scp",
        "rsync",
        "ftp",
        "sftp",
        "telnet",
        "nmap",
        "ping",
        "traceroute",
        "dig",
        "nslookup",
        "host",
        "http",
        "https",
    ]
)


def _is_dangerous_rm(command: str) -> bool:
    """Return True if the command is a recursive+force delete."""
    return bool(re.search(r"\brm\b", command) and _RF_FLAGS.search(command))


def is_shell_allowed(command: str, trust: TrustMode) -> tuple[bool, str]:
    """Return (allowed, reason). reason is empty when allowed.

    Always-deny list applies in every trust mode, including yolo.
    Readonly mode also blocks write-pattern commands.
    """
    if _is_dangerous_rm(command):
        return False, "rm with recursive+force flags is permanently denied"

    for pattern, label in _ALWAYS_DENY:
        if label == "rm_base":
            continue  # already handled above
        if pattern.search(command):
            return False, f"Command matches permanently-denied pattern: {label!r}"

    if trust == "readonly":
        for pat in _WRITE_PATTERNS:
            if pat.search(command):
                return False, f"Write operation blocked in readonly mode (pattern: {pat.pattern!r})"

    return True, ""


def is_path_allowed(path: str | Path, project_root: Path) -> tuple[bool, str]:
    """Return (allowed, reason). reason is empty when the path is within project_root."""
    try:
        resolved = Path(path).resolve()
        resolved.relative_to(project_root.resolve())
        return True, ""
    except ValueError:
        return False, f"Path {str(path)!r} is outside project root {str(project_root)!r}"


def is_network_allowed(trust: TrustMode) -> tuple[bool, str]:
    """Outbound network from tools is blocked unless trust mode is yolo."""
    if trust == "yolo":
        return True, ""
    return False, "Outbound network is disabled. Set trust=yolo to enable."
