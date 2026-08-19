from __future__ import annotations

import asyncio
import ipaddress
import json
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from axis_agent.firewall import (
    FirewallPolicy,
    FirewallPolicyLoadError,
    FirewallRule,
    FirewallService,
    NavigationPermitStore,
    ReasonCode,
    RuleEffect,
    UrlPurpose,
    UrlScheme,
    load_policy_file,
    policy_from_hosts,
)

PUBLIC_V4 = "93.184.216.34"
PUBLIC_V6 = "2606:4700:4700::1111"


class FakeResolver:
    def __init__(self, answers: dict[str, Sequence[str] | Exception] | None = None) -> None:
        self.answers = answers or {}
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, host: str, port: int) -> Sequence[str]:
        self.calls.append((host, port))
        answer = self.answers.get(host, OSError(f"unexpected DNS lookup for {host}"))
        if isinstance(answer, Exception):
            raise answer
        return answer


def make_rule(
    number: int,
    effect: RuleEffect,
    host: str,
    *,
    include_subdomains: bool = False,
    schemes: frozenset[UrlScheme] = frozenset({UrlScheme.HTTP, UrlScheme.HTTPS}),
    ports: frozenset[int] | None = None,
    purposes: frozenset[UrlPurpose] = frozenset(UrlPurpose),
    enabled: bool = True,
) -> FirewallRule:
    return FirewallRule(
        id=UUID(int=number),
        effect=effect,
        host=host,
        include_subdomains=include_subdomains,
        schemes=schemes,
        ports=ports,
        purposes=purposes,
        enabled=enabled,
    )


def service(
    *rules: FirewallRule,
    answers: dict[str, Sequence[str] | Exception] | None = None,
) -> tuple[FirewallService, FakeResolver]:
    resolver = FakeResolver(answers)
    return FirewallService(FirewallPolicy(rules=tuple(rules)), resolver), resolver


def evaluate(
    firewall: FirewallService,
    url: str,
    *,
    purpose: UrlPurpose = UrlPurpose.NAVIGATION,
    internal: bool = False,
):
    return asyncio.run(firewall.evaluate(url, purpose=purpose, internal=internal))


def test_empty_policy_is_default_deny_without_dns_lookup() -> None:
    firewall, resolver = service()

    decision = evaluate(firewall, "https://example.com")

    assert decision.allowed is False
    assert decision.reason is ReasonCode.DENY_NO_ALLOW_MATCH
    assert resolver.calls == []


def test_matching_allow_rule_resolves_and_allows_public_destination() -> None:
    allow = make_rule(1, RuleEffect.ALLOW, "example.com")
    firewall, resolver = service(allow, answers={"example.com": (PUBLIC_V4, PUBLIC_V6)})

    decision = evaluate(firewall, "https://example.com/path")

    assert decision.allowed is True
    assert decision.reason is ReasonCode.ALLOW_RULE_MATCH
    assert decision.matched_rule_id == allow.id
    assert decision.resolved_ips == (PUBLIC_V6, PUBLIC_V4)
    assert resolver.calls == [("example.com", 443)]


def test_navigation_permit_is_action_bound_and_single_use() -> None:
    allow = make_rule(1, RuleEffect.ALLOW, "example.com")
    firewall, _ = service(allow, answers={"example.com": (PUBLIC_V4,)})
    decision = evaluate(firewall, "https://example.com/path")
    action_id = UUID(int=200)
    permits = NavigationPermitStore()
    permit = permits.issue(decision, action_id=action_id)

    assert permits.consume(
        permit,
        action_id=action_id,
        canonical_url="https://example.com/path",
        purpose=UrlPurpose.NAVIGATION,
        policy_hash=decision.policy_hash,
    )
    assert not permits.consume(
        permit,
        action_id=action_id,
        canonical_url="https://example.com/path",
        purpose=UrlPurpose.NAVIGATION,
        policy_hash=decision.policy_hash,
    )


def test_navigation_permit_rejects_wrong_action_and_is_burned() -> None:
    allow = make_rule(1, RuleEffect.ALLOW, "example.com")
    firewall, _ = service(allow, answers={"example.com": (PUBLIC_V4,)})
    decision = evaluate(firewall, "https://example.com")
    permits = NavigationPermitStore()
    permit = permits.issue(decision, action_id=UUID(int=200))

    assert not permits.consume(
        permit,
        action_id=UUID(int=201),
        canonical_url="https://example.com/",
        purpose=UrlPurpose.NAVIGATION,
        policy_hash=decision.policy_hash,
    )
    assert not permits.consume(
        permit,
        action_id=UUID(int=200),
        canonical_url="https://example.com/",
        purpose=UrlPurpose.NAVIGATION,
        policy_hash=decision.policy_hash,
    )


def test_deny_rule_overrides_allow_rule_and_skips_dns() -> None:
    allow = make_rule(1, RuleEffect.ALLOW, "example.com", include_subdomains=True)
    deny = make_rule(2, RuleEffect.DENY, "private.example.com")
    firewall, resolver = service(allow, deny)

    decision = evaluate(firewall, "https://private.example.com")

    assert decision.allowed is False
    assert decision.reason is ReasonCode.DENY_RULE_MATCH
    assert decision.matched_rule_id == deny.id
    assert resolver.calls == []


@pytest.mark.parametrize(
    "url",
    [
        "chrome://extensions",
        "chrome-extension://abc/options.html",
        "javascript:alert(1)",
        "data:text/html,hello",
        "vbscript:msgbox(1)",
        "file:///etc/passwd",
        "ws://example.com/socket",
        "wss://example.com/socket",
    ],
)
def test_dangerous_schemes_are_hard_blocked(url: str) -> None:
    firewall, resolver = service()

    decision = evaluate(firewall, url)

    assert decision.reason is ReasonCode.DENY_HARD_SCHEME
    assert resolver.calls == []


@pytest.mark.parametrize(
    "url", ["ftp://example.com/file", "mailto:user@example.com", "about:newtab"]
)
def test_non_http_schemes_are_rejected(url: str) -> None:
    firewall, _ = service()
    assert evaluate(firewall, url).reason is ReasonCode.DENY_UNSUPPORTED_SCHEME


def test_about_blank_is_only_allowed_for_explicit_internal_use() -> None:
    firewall, resolver = service()

    external = evaluate(firewall, "about:blank")
    internal = evaluate(firewall, "  ABOUT:blank  ", internal=True)

    assert external.reason is ReasonCode.DENY_INTERNAL_ONLY_URL
    assert internal.allowed is True
    assert internal.reason is ReasonCode.ALLOW_INTERNAL_BLANK
    assert internal.canonical_url == "about:blank"
    assert resolver.calls == []


@pytest.mark.parametrize(
    "url",
    [
        "https://chromewebstore.google.com/detail/extension/id",
        "https://sub.chromewebstore.google.com/anything",
        "https://chrome.google.com/webstore/detail/extension/id",
    ],
)
def test_chrome_web_store_is_hard_blocked(url: str) -> None:
    firewall, resolver = service()
    assert evaluate(firewall, url).reason is ReasonCode.DENY_HARD_HOST
    assert resolver.calls == []


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost",
        "http://api.localhost",
        "http://localhost.localdomain",
        "http://metadata",
        "http://metadata.google.internal/latest",
        "http://instance-data.ec2.internal/latest",
    ],
)
def test_localhost_and_metadata_names_are_hard_blocked(url: str) -> None:
    firewall, resolver = service()
    assert evaluate(firewall, url).reason is ReasonCode.DENY_HARD_HOST
    assert resolver.calls == []


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1",
        "http://10.0.0.1",
        "http://172.16.0.1",
        "http://192.168.1.1",
        "http://169.254.169.254/latest/meta-data",
        "http://169.254.170.2",
        "http://100.64.0.1",
        "http://100.100.100.200",
        "http://0.0.0.0",
        "http://224.0.0.1",
        "http://192.0.2.1",
        "http://[::1]",
        "http://[fe80::1]",
        "http://[fc00::1]",
        "http://[ff00::1]",
        "http://[::ffff:127.0.0.1]",
        "http://[fd00:ec2::254]",
    ],
)
def test_direct_non_public_addresses_are_blocked_before_rules(url: str) -> None:
    firewall, resolver = service()
    decision = evaluate(firewall, url)
    assert decision.reason is ReasonCode.DENY_PRIVATE_ADDRESS
    assert decision.resolved_ips
    assert resolver.calls == []


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("https://user:secret@example.com", ReasonCode.DENY_USERINFO),
        ("https://@example.com", ReasonCode.DENY_USERINFO),
        ("https://example.com\\@evil.com", ReasonCode.DENY_BACKSLASH),
        ("https://example.com/\npath", ReasonCode.DENY_CONTROL_CHARACTER),
        ("\nhttps://example.com", ReasonCode.DENY_CONTROL_CHARACTER),
        ("https://example.com:abc", ReasonCode.DENY_INVALID_URL),
        ("https://example.com:", ReasonCode.DENY_INVALID_URL),
        ("https://[::1", ReasonCode.DENY_INVALID_URL),
        ("https://exa_mple.com", ReasonCode.DENY_INVALID_IDNA),
        ("https:///missing-host", ReasonCode.DENY_INVALID_URL),
    ],
)
def test_ambiguous_or_malformed_urls_fail_closed(url: str, reason: ReasonCode) -> None:
    firewall, resolver = service()
    assert evaluate(firewall, url).reason is reason
    assert resolver.calls == []


def test_idna_case_trailing_dot_default_port_and_fragment_are_canonicalized() -> None:
    allow = make_rule(1, RuleEffect.ALLOW, "bücher.example")
    firewall, resolver = service(allow, answers={"xn--bcher-kva.example": (PUBLIC_V4,)})

    decision = evaluate(firewall, "  HTTPS://BÜCHER.Example.:443/a?x=1#not-sent  ")

    assert decision.allowed is True
    assert decision.canonical_url == "https://xn--bcher-kva.example/a?x=1"
    assert resolver.calls == [("xn--bcher-kva.example", 443)]


def test_exact_host_does_not_implicitly_allow_subdomains() -> None:
    allow = make_rule(1, RuleEffect.ALLOW, "example.com")
    firewall, resolver = service(allow)
    assert evaluate(firewall, "https://www.example.com").reason is ReasonCode.DENY_NO_ALLOW_MATCH
    assert resolver.calls == []


def test_explicit_subdomain_rule_uses_dns_label_boundary() -> None:
    allow = make_rule(1, RuleEffect.ALLOW, "example.com", include_subdomains=True)
    firewall, resolver = service(allow, answers={"api.example.com": (PUBLIC_V4,)})

    assert evaluate(firewall, "https://api.example.com").allowed is True
    assert evaluate(firewall, "https://evil-example.com").reason is ReasonCode.DENY_NO_ALLOW_MATCH
    assert resolver.calls == [("api.example.com", 443)]


def test_non_default_port_requires_an_explicit_port_grant() -> None:
    default_only = make_rule(1, RuleEffect.ALLOW, "example.com")
    explicit = make_rule(2, RuleEffect.ALLOW, "api.example.com", ports=frozenset({8443}))
    firewall, resolver = service(explicit, default_only, answers={"api.example.com": (PUBLIC_V4,)})

    blocked = evaluate(firewall, "https://example.com:8443")
    allowed = evaluate(firewall, "https://api.example.com:8443/v1")

    assert blocked.reason is ReasonCode.DENY_NO_ALLOW_MATCH
    assert allowed.allowed is True
    assert allowed.canonical_url == "https://api.example.com:8443/v1"
    assert resolver.calls == [("api.example.com", 8443)]


def test_rule_is_scoped_to_explicit_purpose() -> None:
    allow = make_rule(
        1,
        RuleEffect.ALLOW,
        "example.com",
        purposes=frozenset({UrlPurpose.NAVIGATION}),
    )
    firewall, resolver = service(allow, answers={"example.com": (PUBLIC_V4,)})

    assert evaluate(firewall, "https://example.com").allowed is True
    assert (
        evaluate(firewall, "https://example.com", purpose=UrlPurpose.REDIRECT).reason
        is ReasonCode.DENY_NO_ALLOW_MATCH
    )
    assert resolver.calls == [("example.com", 443)]


def test_disabled_rules_are_ignored() -> None:
    allow = make_rule(1, RuleEffect.ALLOW, "example.com", enabled=False)
    firewall, resolver = service(allow)
    assert evaluate(firewall, "https://example.com").reason is ReasonCode.DENY_NO_ALLOW_MATCH
    assert resolver.calls == []


@pytest.mark.parametrize(
    "answers",
    [
        (PUBLIC_V4, "127.0.0.1"),
        (PUBLIC_V4, "10.0.0.1"),
        (PUBLIC_V4, "100.64.0.1"),
        (PUBLIC_V6, "::ffff:192.168.1.1"),
        ("100.100.100.200",),
    ],
)
def test_any_non_public_dns_answer_blocks_the_destination(answers: Sequence[str]) -> None:
    allow = make_rule(1, RuleEffect.ALLOW, "example.com")
    firewall, _ = service(allow, answers={"example.com": answers})
    decision = evaluate(firewall, "https://example.com")
    assert decision.reason is ReasonCode.DENY_PRIVATE_ADDRESS
    assert {ipaddress.ip_address(item) for item in decision.resolved_ips} == {
        ipaddress.ip_address(item) for item in answers
    }


@pytest.mark.parametrize("answer", [(), ("not-an-ip",), OSError("NXDOMAIN")])
def test_dns_errors_and_invalid_answers_fail_closed(answer: Sequence[str] | Exception) -> None:
    allow = make_rule(1, RuleEffect.ALLOW, "example.com")
    firewall, _ = service(allow, answers={"example.com": answer})
    assert evaluate(firewall, "https://example.com").reason is ReasonCode.DENY_DNS_FAILURE


def test_public_ip_literal_can_be_explicitly_allowed_without_dns() -> None:
    allow = make_rule(1, RuleEffect.ALLOW, PUBLIC_V4)
    firewall, resolver = service(allow)
    decision = evaluate(firewall, f"https://{PUBLIC_V4}")
    assert decision.allowed is True
    assert decision.resolved_ips == (PUBLIC_V4,)
    assert resolver.calls == []


def test_policy_hash_is_sha256_order_independent_and_change_sensitive() -> None:
    one = make_rule(1, RuleEffect.ALLOW, "example.com")
    two = make_rule(2, RuleEffect.DENY, "blocked.example.com")
    changed = make_rule(2, RuleEffect.DENY, "other.example.com")

    first = FirewallPolicy(rules=(one, two)).policy_hash
    reordered = FirewallPolicy(rules=(two, one)).policy_hash
    different = FirewallPolicy(rules=(one, changed)).policy_hash

    assert len(first) == 64
    assert first == reordered
    assert first != different


def test_decision_contains_stable_hashes_and_does_not_echo_raw_url() -> None:
    firewall, _ = service()
    secret_url = "https://example.com/path?token=top-secret#fragment"
    decision = evaluate(firewall, secret_url)

    assert len(decision.original_url_sha256) == 64
    assert len(decision.policy_hash) == 64
    assert "fragment" not in (decision.canonical_url or "")
    assert "top-secret" not in repr(decision.original_url_sha256)


def test_strict_models_reject_extra_fields_and_type_coercion() -> None:
    with pytest.raises(ValidationError):
        FirewallRule(
            id=UUID(int=1),
            effect=RuleEffect.ALLOW,
            host="example.com",
            include_subdomains=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        FirewallRule(
            id=UUID(int=1),
            effect=RuleEffect.ALLOW,
            host="example.com",
            unexpected=True,  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    "host",
    [
        "*.example.com",
        "example.*",
        "https://example.com",
        "example.com/path",
        "user@example.com",
        "exa_mple.com",
        " example.com",
        "example.com ",
    ],
)
def test_rule_hosts_reject_wildcards_regex_urls_and_ambiguous_syntax(host: str) -> None:
    with pytest.raises(ValidationError):
        make_rule(1, RuleEffect.ALLOW, host)


def test_rule_host_is_canonicalized_to_idna() -> None:
    rule = make_rule(1, RuleEffect.ALLOW, "BÜCHER.Example.")
    assert rule.host == "xn--bcher-kva.example"


def test_policy_rejects_duplicate_ids_and_equivalent_rules() -> None:
    first = make_rule(1, RuleEffect.ALLOW, "example.com")
    duplicate_id = make_rule(1, RuleEffect.DENY, "blocked.example.com")
    equivalent = make_rule(2, RuleEffect.ALLOW, "example.com")

    with pytest.raises(ValidationError, match="duplicate firewall rule id"):
        FirewallPolicy(rules=(first, duplicate_id))
    with pytest.raises(ValidationError, match="duplicate equivalent firewall rule"):
        FirewallPolicy(rules=(first, equivalent))


def test_ip_rule_cannot_claim_subdomains() -> None:
    with pytest.raises(ValidationError, match="IP rules cannot include subdomains"):
        make_rule(1, RuleEffect.ALLOW, PUBLIC_V4, include_subdomains=True)


@pytest.mark.parametrize("ports", [frozenset(), frozenset({0}), frozenset({65536})])
def test_rule_rejects_empty_or_out_of_range_port_sets(ports: frozenset[int]) -> None:
    with pytest.raises(ValidationError):
        make_rule(1, RuleEffect.ALLOW, "example.com", ports=ports)


def test_load_policy_file_strictly_validates_complete_json(tmp_path: Path) -> None:
    path = tmp_path / "firewall.json"
    path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "id": "00000000-0000-0000-0000-000000000001",
                        "effect": "allow",
                        "host": "example.com",
                        "include_subdomains": True,
                        "schemes": ["https"],
                        "ports": None,
                        "purposes": ["navigation", "redirect"],
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    policy = load_policy_file(path)

    assert len(policy.rules) == 1
    assert policy.rules[0].host == "example.com"
    assert policy.rules[0].schemes == frozenset({UrlScheme.HTTPS})


@pytest.mark.parametrize(
    "payload",
    [
        "{not-json",
        json.dumps({"rules": [], "unknown": True}),
        json.dumps(
            {
                "rules": [
                    {
                        "id": "00000000-0000-0000-0000-000000000001",
                        "effect": "allow",
                        "host": "*.example.com",
                    }
                ]
            }
        ),
        json.dumps(
            {
                "rules": [
                    {
                        "id": "00000000-0000-0000-0000-000000000001",
                        "effect": "allow",
                        "host": "example.com",
                    },
                    {
                        "id": "00000000-0000-0000-0000-000000000001",
                        "effect": "deny",
                        "host": "blocked.example.com",
                    },
                ]
            }
        ),
    ],
)
def test_load_policy_file_rejects_malformed_unknown_and_duplicate_rules(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "invalid-firewall.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(FirewallPolicyLoadError) as raised:
        load_policy_file(path)

    assert "invalid" in str(raised.value)


def test_load_policy_file_rejects_empty_non_utf8_and_oversized_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    with pytest.raises(FirewallPolicyLoadError, match="empty"):
        load_policy_file(empty)

    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe")
    with pytest.raises(FirewallPolicyLoadError, match="UTF-8"):
        load_policy_file(binary)

    large = tmp_path / "large.json"
    large.write_bytes(b"{}")
    with pytest.raises(FirewallPolicyLoadError, match="exceeds"):
        load_policy_file(large, max_bytes=1)


def test_policy_from_hosts_has_deterministic_ids_and_strict_default_deny() -> None:
    first = policy_from_hosts(["example.com"], ["blocked.example.com"], include_subdomains=True)
    second = policy_from_hosts(["example.com"], ["blocked.example.com"], include_subdomains=True)

    assert first == second
    assert first.policy_hash == second.policy_hash
    assert {rule.effect for rule in first.rules} == {RuleEffect.ALLOW, RuleEffect.DENY}


def test_policy_from_hosts_rejects_duplicates_instead_of_silently_deduplicating() -> None:
    with pytest.raises(ValidationError, match="duplicate firewall rule id"):
        policy_from_hosts(["example.com", "example.com"])
