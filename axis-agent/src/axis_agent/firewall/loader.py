"""Safe loading and construction helpers for firewall policy snapshots."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID, uuid5

from pydantic import ValidationError

from .models import (
    ALL_PURPOSES,
    FirewallPolicy,
    FirewallRule,
    RuleEffect,
    UrlPurpose,
    UrlScheme,
    canonicalize_rule_host,
)

MAX_POLICY_BYTES = 1024 * 1024
_RULE_ID_NAMESPACE = UUID("88b8441c-9a6c-4d8b-8924-fd6431202f10")


class FirewallPolicyLoadError(ValueError):
    """Raised when a policy file cannot be read or strictly validated."""


def load_policy_file(path: str | Path, *, max_bytes: int = MAX_POLICY_BYTES) -> FirewallPolicy:
    """Read a complete UTF-8 JSON file and validate it as one policy snapshot.

    This function never mutates a running service. Callers can load and fully
    validate the returned immutable policy before replacing their active
    service reference, so a malformed file cannot partially apply rules.
    """

    policy_path = Path(path)
    try:
        raw = policy_path.read_bytes()
    except OSError as exc:
        raise FirewallPolicyLoadError(f"unable to read firewall policy: {policy_path}") from exc
    if not raw:
        raise FirewallPolicyLoadError("firewall policy file is empty")
    if max_bytes < 1 or len(raw) > max_bytes:
        raise FirewallPolicyLoadError(f"firewall policy exceeds {max_bytes} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FirewallPolicyLoadError("firewall policy must be UTF-8") from exc
    try:
        # model_validate_json keeps JSON parsing and strict Pydantic validation
        # in one operation; no unvalidated dictionary is exposed to callers.
        return FirewallPolicy.model_validate_json(text, strict=True)
    except (ValidationError, ValueError) as exc:
        raise FirewallPolicyLoadError("firewall policy JSON is invalid") from exc


def policy_from_hosts(
    allow_hosts: Sequence[str],
    deny_hosts: Sequence[str] = (),
    *,
    include_subdomains: bool = False,
    schemes: frozenset[UrlScheme] = frozenset({UrlScheme.HTTP, UrlScheme.HTTPS}),
    ports: frozenset[int] | None = None,
    purposes: frozenset[UrlPurpose] = ALL_PURPOSES,
) -> FirewallPolicy:
    """Build a strict policy from CLI-friendly allow and deny host lists.

    Rule IDs are deterministic UUIDv5 values, which keeps policy hashes stable
    across process restarts. Duplicate hosts remain validation errors rather
    than being silently discarded.
    """

    rules: list[FirewallRule] = []
    for effect, hosts in ((RuleEffect.ALLOW, allow_hosts), (RuleEffect.DENY, deny_hosts)):
        for raw_host in hosts:
            host = canonicalize_rule_host(raw_host)
            identity = json.dumps(
                {
                    "effect": effect.value,
                    "host": host,
                    "include_subdomains": include_subdomains,
                    "schemes": sorted(item.value for item in schemes),
                    "ports": None if ports is None else sorted(ports),
                    "purposes": sorted(item.value for item in purposes),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            rules.append(
                FirewallRule(
                    id=uuid5(_RULE_ID_NAMESPACE, identity),
                    effect=effect,
                    host=host,
                    include_subdomains=include_subdomains,
                    schemes=schemes,
                    ports=ports,
                    purposes=purposes,
                )
            )
    return FirewallPolicy(rules=tuple(rules))
