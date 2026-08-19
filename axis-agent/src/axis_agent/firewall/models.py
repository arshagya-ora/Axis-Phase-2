"""Strict data contracts for AXIS URL firewall decisions."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_CONTROL_CATEGORIES = {"Cc", "Cf"}
POLICY_VERSION = "axis-firewall-v1"


class RuleEffect(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class UrlScheme(StrEnum):
    HTTP = "http"
    HTTPS = "https"


class UrlPurpose(StrEnum):
    NAVIGATION = "navigation"
    REDIRECT = "redirect"
    POPUP = "popup"
    IFRAME = "iframe"
    SUBRESOURCE = "subresource"
    DOWNLOAD = "download"
    API = "api"


ALL_PURPOSES = frozenset(UrlPurpose)


class ReasonCode(StrEnum):
    ALLOW_RULE_MATCH = "ALLOW_RULE_MATCH"
    ALLOW_INTERNAL_BLANK = "ALLOW_INTERNAL_BLANK"
    DENY_INVALID_URL = "DENY_INVALID_URL"
    DENY_CONTROL_CHARACTER = "DENY_CONTROL_CHARACTER"
    DENY_BACKSLASH = "DENY_BACKSLASH"
    DENY_INVALID_IDNA = "DENY_INVALID_IDNA"
    DENY_HARD_SCHEME = "DENY_HARD_SCHEME"
    DENY_UNSUPPORTED_SCHEME = "DENY_UNSUPPORTED_SCHEME"
    DENY_INTERNAL_ONLY_URL = "DENY_INTERNAL_ONLY_URL"
    DENY_HARD_HOST = "DENY_HARD_HOST"
    DENY_USERINFO = "DENY_USERINFO"
    DENY_PRIVATE_ADDRESS = "DENY_PRIVATE_ADDRESS"
    DENY_DNS_FAILURE = "DENY_DNS_FAILURE"
    DENY_RULE_MATCH = "DENY_RULE_MATCH"
    DENY_NO_ALLOW_MATCH = "DENY_NO_ALLOW_MATCH"


class FirewallRule(BaseModel):
    """A structured host rule.

    ``ports=None`` means only the scheme's default port. Non-default ports must
    be explicitly granted. Regexes and wildcard strings are intentionally not
    supported; subdomains are enabled through ``include_subdomains``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: UUID
    effect: RuleEffect
    host: str
    include_subdomains: bool = False
    schemes: frozenset[UrlScheme] = frozenset({UrlScheme.HTTP, UrlScheme.HTTPS})
    ports: frozenset[int] | None = None
    purposes: frozenset[UrlPurpose] = ALL_PURPOSES
    enabled: bool = True

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        return canonicalize_rule_host(value)

    @field_validator("schemes")
    @classmethod
    def validate_schemes(cls, value: frozenset[UrlScheme]) -> frozenset[UrlScheme]:
        if not value:
            raise ValueError("schemes must not be empty")
        return value

    @field_validator("ports")
    @classmethod
    def validate_ports(cls, value: frozenset[int] | None) -> frozenset[int] | None:
        if value is not None and not value:
            raise ValueError("ports must be null or contain at least one port")
        if value is not None and any(port < 1 or port > 65535 for port in value):
            raise ValueError("ports must be between 1 and 65535")
        return value

    @field_validator("purposes")
    @classmethod
    def validate_purposes(cls, value: frozenset[UrlPurpose]) -> frozenset[UrlPurpose]:
        if not value:
            raise ValueError("purposes must not be empty")
        return value

    @model_validator(mode="after")
    def validate_subdomain_mode(self) -> Self:
        try:
            ipaddress.ip_address(self.host)
        except ValueError:
            return self
        if self.include_subdomains:
            raise ValueError("IP rules cannot include subdomains")
        return self

    def canonical_signature(self) -> tuple[object, ...]:
        return (
            self.effect.value,
            self.host,
            self.include_subdomains,
            tuple(sorted(item.value for item in self.schemes)),
            None if self.ports is None else tuple(sorted(self.ports)),
            tuple(sorted(item.value for item in self.purposes)),
            self.enabled,
        )


class FirewallPolicy(BaseModel):
    """An immutable, hash-addressed policy snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    rules: tuple[FirewallRule, ...] = ()

    @model_validator(mode="after")
    def validate_unique_rules(self) -> Self:
        ids: set[UUID] = set()
        signatures: set[tuple[object, ...]] = set()
        for rule in self.rules:
            if rule.id in ids:
                raise ValueError(f"duplicate firewall rule id: {rule.id}")
            signature = rule.canonical_signature()
            if signature in signatures:
                raise ValueError("duplicate equivalent firewall rule")
            ids.add(rule.id)
            signatures.add(signature)
        return self

    @property
    def policy_hash(self) -> str:
        rules = [
            {
                "id": str(rule.id),
                "effect": rule.effect.value,
                "host": rule.host,
                "include_subdomains": rule.include_subdomains,
                "schemes": sorted(item.value for item in rule.schemes),
                "ports": None if rule.ports is None else sorted(rule.ports),
                "purposes": sorted(item.value for item in rule.purposes),
                "enabled": rule.enabled,
            }
            for rule in sorted(self.rules, key=lambda item: str(item.id))
        ]
        payload = json.dumps(
            {"version": POLICY_VERSION, "rules": rules},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class FirewallDecision(BaseModel):
    """Result returned for every URL policy evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    decision_id: UUID = Field(default_factory=uuid4)
    allowed: bool
    reason: ReasonCode
    purpose: UrlPurpose
    original_url_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_url: str | None = None
    matched_rule_id: UUID | None = None
    resolved_ips: tuple[str, ...] = ()
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class NavigationPermit(BaseModel):
    """Short-lived, action-bound authorization consumed by a browser adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    permit_id: UUID = Field(default_factory=uuid4)
    decision_id: UUID
    action_id: UUID
    canonical_url: str
    purpose: UrlPurpose
    resolved_ips: tuple[str, ...]
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    issued_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime


def canonicalize_rule_host(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("host must be non-empty and must not contain surrounding whitespace")
    if any(unicodedata.category(char) in _CONTROL_CATEGORIES for char in value):
        raise ValueError("host contains a control or formatting character")
    if any(token in value for token in ("\\", "/", "@", "?", "#", "*")) or "://" in value:
        raise ValueError("host must be a domain or IP address, not a URL, wildcard, or regex")

    host = value.lower().rstrip(".")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        pass

    try:
        ascii_host = host.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("host is not valid IDNA") from exc
    if not ascii_host or len(ascii_host) > 253:
        raise ValueError("host is not valid IDNA")
    labels = ascii_host.split(".")
    if any(not _DNS_LABEL.fullmatch(label) for label in labels):
        raise ValueError("host is not valid IDNA")
    return ascii_host
