"""Fail-closed environment projection for optional MCP child processes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final

# Only operating-system values needed to locate and launch a local executable are
# inherited.  Application configuration and credentials are intentionally absent.
_SAFE_CHILD_ENVIRONMENT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "SHELL",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
    }
)

_SENSITIVE_NAME_MARKERS: Final[tuple[str, ...]] = (
    "API_KEY",
    "CREDENTIAL",
    "MODEL",
    "PASSWORD",
    "PROJECT",
    "SECRET",
    "TOKEN",
)


def _is_sensitive_name(name: str) -> bool:
    normalized = name.upper()
    return (
        normalized == "KEY"
        or normalized.endswith("_KEY")
        or normalized.startswith("OCI_")
        or any(marker in normalized for marker in _SENSITIVE_NAME_MARKERS)
    )


def build_child_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the minimum environment safe for a local MCP child.

    OCI configuration, project/model selection, keys, tokens and other application
    values are excluded even if a future edit accidentally adds one to the safe-name
    allowlist.  Matching is case-insensitive for Windows compatibility.
    """

    candidate = os.environ if source is None else source
    result: dict[str, str] = {}
    for name, value in candidate.items():
        normalized = name.upper()
        if normalized not in _SAFE_CHILD_ENVIRONMENT_NAMES or _is_sensitive_name(normalized):
            continue
        result[name] = value
    return result
