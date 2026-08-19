"""Default-deny URL policy enforcement for AXIS browser automation."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import socket
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import SplitResult, urlsplit, urlunsplit
from uuid import UUID

from .models import (
    FirewallDecision,
    FirewallPolicy,
    FirewallRule,
    NavigationPermit,
    ReasonCode,
    RuleEffect,
    UrlPurpose,
    UrlScheme,
    canonicalize_rule_host,
)

Resolver = Callable[[str, int], Awaitable[Sequence[str]]]

_HARD_SCHEMES = frozenset(
    {"chrome", "chrome-extension", "javascript", "data", "vbscript", "file", "ws", "wss"}
)
_HARD_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata",
        "metadata.google.internal",
        "metadata.azure.internal",
        "instance-data",
        "instance-data.ec2.internal",
    }
)
_METADATA_IPS = frozenset(
    {
        "100.100.100.200",  # Alibaba Cloud
        "169.254.169.254",  # AWS, Azure, GCP, Oracle and others
        "169.254.170.2",  # AWS ECS task metadata
        "fd00:ec2::254",  # AWS IPv6 metadata
    }
)
_CONTROL_CATEGORIES = {"Cc", "Cf"}
_DNS_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class _NormalizedUrl:
    canonical: str
    scheme: UrlScheme
    host: str
    port: int
    path: str
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None


class _UrlError(ValueError):
    def __init__(self, reason: ReasonCode) -> None:
        self.reason = reason
        super().__init__(reason.value)


async def system_resolver(host: str, port: int) -> tuple[str, ...]:
    """Resolve a host without blocking the agent event loop."""

    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(
        host,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    return tuple(str(record[4][0]) for record in records)


class FirewallService:
    """Evaluate browser destinations against an immutable policy snapshot."""

    def __init__(self, policy: FirewallPolicy, resolver: Resolver = system_resolver) -> None:
        self._policy = policy
        self._resolver = resolver

    @property
    def policy(self) -> FirewallPolicy:
        return self._policy

    async def evaluate(
        self,
        url: str,
        *,
        purpose: UrlPurpose = UrlPurpose.NAVIGATION,
        internal: bool = False,
    ) -> FirewallDecision:
        raw_hash = hashlib.sha256(url.encode("utf-8", errors="surrogatepass")).hexdigest()
        # Inspect the original input before trimming. urllib.parse intentionally
        # strips some leading C0 controls, which must not bypass our rejection.
        if any(unicodedata.category(char) in _CONTROL_CATEGORIES for char in url):
            return self._decision(
                allowed=False,
                reason=ReasonCode.DENY_CONTROL_CHARACTER,
                purpose=purpose,
                raw_hash=raw_hash,
            )
        stripped = url.strip()

        if stripped.lower() == "about:blank":
            return self._decision(
                allowed=internal,
                reason=(
                    ReasonCode.ALLOW_INTERNAL_BLANK
                    if internal
                    else ReasonCode.DENY_INTERNAL_ONLY_URL
                ),
                purpose=purpose,
                raw_hash=raw_hash,
                canonical_url="about:blank",
            )

        try:
            normalized = _normalize_url(stripped)
        except _UrlError as exc:
            return self._decision(
                allowed=False,
                reason=exc.reason,
                purpose=purpose,
                raw_hash=raw_hash,
            )

        if _is_hard_host(normalized.host) or _is_chrome_web_store(normalized):
            return self._decision(
                allowed=False,
                reason=ReasonCode.DENY_HARD_HOST,
                purpose=purpose,
                raw_hash=raw_hash,
                canonical_url=normalized.canonical,
            )

        if normalized.ip is not None and not _is_public_address(normalized.ip):
            return self._decision(
                allowed=False,
                reason=ReasonCode.DENY_PRIVATE_ADDRESS,
                purpose=purpose,
                raw_hash=raw_hash,
                canonical_url=normalized.canonical,
                resolved_ips=(normalized.ip.compressed,),
            )

        deny_rule = self._first_matching_rule(normalized, purpose, RuleEffect.DENY)
        if deny_rule is not None:
            return self._decision(
                allowed=False,
                reason=ReasonCode.DENY_RULE_MATCH,
                purpose=purpose,
                raw_hash=raw_hash,
                canonical_url=normalized.canonical,
                matched_rule_id=deny_rule.id,
            )

        allow_rule = self._first_matching_rule(normalized, purpose, RuleEffect.ALLOW)
        if allow_rule is None:
            return self._decision(
                allowed=False,
                reason=ReasonCode.DENY_NO_ALLOW_MATCH,
                purpose=purpose,
                raw_hash=raw_hash,
                canonical_url=normalized.canonical,
            )

        try:
            resolved_ips = await self._resolve_public_addresses(normalized)
        except (OSError, ValueError):
            return self._decision(
                allowed=False,
                reason=ReasonCode.DENY_DNS_FAILURE,
                purpose=purpose,
                raw_hash=raw_hash,
                canonical_url=normalized.canonical,
                matched_rule_id=allow_rule.id,
            )

        if any(not _is_public_address(ipaddress.ip_address(item)) for item in resolved_ips):
            return self._decision(
                allowed=False,
                reason=ReasonCode.DENY_PRIVATE_ADDRESS,
                purpose=purpose,
                raw_hash=raw_hash,
                canonical_url=normalized.canonical,
                matched_rule_id=allow_rule.id,
                resolved_ips=resolved_ips,
            )

        return self._decision(
            allowed=True,
            reason=ReasonCode.ALLOW_RULE_MATCH,
            purpose=purpose,
            raw_hash=raw_hash,
            canonical_url=normalized.canonical,
            matched_rule_id=allow_rule.id,
            resolved_ips=resolved_ips,
        )

    def _first_matching_rule(
        self,
        url: _NormalizedUrl,
        purpose: UrlPurpose,
        effect: RuleEffect,
    ) -> FirewallRule | None:
        for rule in sorted(self._policy.rules, key=lambda item: str(item.id)):
            if rule.enabled and rule.effect is effect and _rule_matches(rule, url, purpose):
                return rule
        return None

    async def _resolve_public_addresses(self, url: _NormalizedUrl) -> tuple[str, ...]:
        if url.ip is not None:
            return (url.ip.compressed,)
        values = await asyncio.wait_for(
            self._resolver(url.host, url.port),
            timeout=_DNS_TIMEOUT_SECONDS,
        )
        if not values:
            raise ValueError("resolver returned no addresses")
        addresses: set[str] = set()
        for value in values:
            addresses.add(ipaddress.ip_address(value).compressed)
        return tuple(sorted(addresses))

    def _decision(
        self,
        *,
        allowed: bool,
        reason: ReasonCode,
        purpose: UrlPurpose,
        raw_hash: str,
        canonical_url: str | None = None,
        matched_rule_id: UUID | None = None,
        resolved_ips: tuple[str, ...] = (),
    ) -> FirewallDecision:
        return FirewallDecision(
            allowed=allowed,
            reason=reason,
            purpose=purpose,
            original_url_sha256=raw_hash,
            canonical_url=canonical_url,
            matched_rule_id=matched_rule_id,
            resolved_ips=resolved_ips,
            policy_hash=self._policy.policy_hash,
        )


class NavigationPermitStore:
    """Issue and consume process-local, single-use navigation permits."""

    def __init__(self) -> None:
        self._permits: dict[UUID, NavigationPermit] = {}

    def issue(
        self,
        decision: FirewallDecision,
        *,
        action_id: UUID,
        ttl_seconds: int = 5,
    ) -> NavigationPermit:
        if not decision.allowed or decision.canonical_url is None:
            raise ValueError("cannot issue a permit for a blocked decision")
        if ttl_seconds < 1 or ttl_seconds > 30:
            raise ValueError("permit TTL must be between 1 and 30 seconds")
        now = datetime.now(UTC)
        self._purge_expired(now)
        permit = NavigationPermit(
            decision_id=decision.decision_id,
            action_id=action_id,
            canonical_url=decision.canonical_url,
            purpose=decision.purpose,
            resolved_ips=decision.resolved_ips,
            policy_hash=decision.policy_hash,
            issued_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        self._permits[permit.permit_id] = permit
        return permit

    def revoke(self, permit_id: UUID) -> None:
        self._permits.pop(permit_id, None)

    def consume(
        self,
        permit: NavigationPermit,
        *,
        action_id: UUID,
        canonical_url: str,
        purpose: UrlPurpose,
        policy_hash: str,
    ) -> bool:
        expected = self._permits.pop(permit.permit_id, None)
        if expected is None or expected != permit:
            return False
        return (
            datetime.now(UTC) <= permit.expires_at
            and permit.action_id == action_id
            and permit.canonical_url == canonical_url
            and permit.purpose is purpose
            and permit.policy_hash == policy_hash
        )

    def _purge_expired(self, now: datetime) -> None:
        expired = [
            permit_id for permit_id, permit in self._permits.items() if permit.expires_at < now
        ]
        for permit_id in expired:
            self._permits.pop(permit_id, None)


def _normalize_url(url: str) -> _NormalizedUrl:
    if not url:
        raise _UrlError(ReasonCode.DENY_INVALID_URL)
    if any(unicodedata.category(char) in _CONTROL_CATEGORIES for char in url):
        raise _UrlError(ReasonCode.DENY_CONTROL_CHARACTER)
    if "\\" in url:
        raise _UrlError(ReasonCode.DENY_BACKSLASH)

    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise _UrlError(ReasonCode.DENY_INVALID_URL) from exc
    scheme_text = parsed.scheme.lower()
    if scheme_text in _HARD_SCHEMES:
        raise _UrlError(ReasonCode.DENY_HARD_SCHEME)
    if scheme_text not in {UrlScheme.HTTP.value, UrlScheme.HTTPS.value}:
        raise _UrlError(ReasonCode.DENY_UNSUPPORTED_SCHEME)
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise _UrlError(ReasonCode.DENY_USERINFO)

    try:
        host_text = parsed.hostname
        explicit_port = parsed.port
    except ValueError as exc:
        raise _UrlError(ReasonCode.DENY_INVALID_URL) from exc
    if not host_text or not parsed.netloc:
        raise _UrlError(ReasonCode.DENY_INVALID_URL)
    authority = parsed.netloc
    if authority.endswith(":") or authority.endswith("]:"):
        raise _UrlError(ReasonCode.DENY_INVALID_URL)

    try:
        host = canonicalize_rule_host(host_text)
    except ValueError as exc:
        raise _UrlError(ReasonCode.DENY_INVALID_IDNA) from exc

    scheme = UrlScheme(scheme_text)
    default_port = 80 if scheme is UrlScheme.HTTP else 443
    port = default_port if explicit_port is None else explicit_port
    if port < 1 or port > 65535:
        raise _UrlError(ReasonCode.DENY_INVALID_URL)

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    display_host = f"[{host}]" if isinstance(ip, ipaddress.IPv6Address) else host
    display_port = "" if port == default_port else f":{port}"
    path = parsed.path or "/"
    canonical = urlunsplit(
        SplitResult(scheme.value, f"{display_host}{display_port}", path, parsed.query, "")
    )
    return _NormalizedUrl(
        canonical=canonical,
        scheme=scheme,
        host=host,
        port=port,
        path=path,
        ip=ip,
    )


def _is_hard_host(host: str) -> bool:
    return (
        host in _HARD_HOSTS
        or host.endswith(".localhost")
        or host.endswith(".localhost.localdomain")
        or host.endswith(".metadata.google.internal")
        or host.endswith(".instance-data.ec2.internal")
    )


def _is_chrome_web_store(url: _NormalizedUrl) -> bool:
    if url.host == "chromewebstore.google.com" or url.host.endswith(".chromewebstore.google.com"):
        return True
    return url.host == "chrome.google.com" and (
        url.path == "/webstore" or url.path.startswith("/webstore/")
    )


def _is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if address.compressed in _METADATA_IPS:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        mapped = address.ipv4_mapped
        return mapped.compressed not in _METADATA_IPS and _is_public_address(mapped)
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or not address.is_global
    )


def _rule_matches(rule: FirewallRule, url: _NormalizedUrl, purpose: UrlPurpose) -> bool:
    if purpose not in rule.purposes or url.scheme not in rule.schemes:
        return False
    default_port = 80 if url.scheme is UrlScheme.HTTP else 443
    if rule.ports is None:
        if url.port != default_port:
            return False
    elif url.port not in rule.ports:
        return False

    if url.host == rule.host:
        return True
    return rule.include_subdomains and url.ip is None and url.host.endswith(f".{rule.host}")
