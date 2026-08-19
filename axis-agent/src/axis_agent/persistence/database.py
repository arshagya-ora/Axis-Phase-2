"""Auditable SQLite storage used by the standalone AXIS agent.

The database deliberately stores execution metadata rather than raw prompts,
DOM snapshots, screenshots, secrets, or unrestricted model reasoning.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import SplitResult, urlsplit, urlunsplit
from uuid import UUID, uuid4

import aiosqlite

from axis_agent.contracts.actions import PRODUCTION_ACTION_TYPES, ProductionActionType
from axis_agent.contracts.agents import ExecutionPlan
from axis_agent.contracts.approvals import (
    ApprovalDecision,
    ApprovalRequest,
)

SessionStatus = Literal["created", "running", "paused", "succeeded", "failed", "cancelled"]
StepStatus = Literal["created", "running", "succeeded", "failed", "cancelled"]
ActionState = Literal[
    "created",
    "validated",
    "allowed",
    "blocked",
    "dispatched",
    "succeeded",
    "failed",
    "timed_out",
    "unknown",
]

_SENSITIVE_FRAGMENTS: Final[tuple[str, ...]] = (
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)
_UNRESTRICTED_CONTENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "audio",
        "content",
        "details",
        "dom",
        "html",
        "image",
        "input",
        "intent",
        "keys",
        "message",
        "output",
        "prompt",
        "query",
        "screenshot",
        "text",
        "transcript",
        "value",
    }
)
_MAX_JSON_BYTES: Final[int] = 64 * 1024
_HASHED_TASK_SUMMARY: Final[str] = "[HASHED]"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _migration_statements(sql: str) -> tuple[str, ...]:
    """Split trusted migration SQL without breaking trigger bodies."""

    statements: list[str] = []
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer.strip()):
            statements.append(buffer.strip())
            buffer = ""
    if buffer.strip():  # pragma: no cover - static migrations are validated in tests
        raise RuntimeError("migration contains an incomplete SQL statement")
    return tuple(statements)


def _canonical_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if len(encoded.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError(f"audit payload exceeds {_MAX_JSON_BYTES} bytes")
    return encoded


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_uuid(value: UUID | str | None, field: str) -> str:
    if value is None:
        return str(uuid4())
    try:
        return str(UUID(str(value)))
    except ValueError as exc:
        raise ValueError(f"{field} must be a UUID") from exc


def _is_sensitive_key(value: str) -> bool:
    normalized = value.lower().replace("_", "").replace("-", "")
    return any(fragment in normalized for fragment in _SENSITIVE_FRAGMENTS)


def redact_for_audit(value: Any) -> Any:
    """Return a JSON-compatible value with common secret fields removed."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text) or key_text.lower() in _UNRESTRICTED_CONTENT_KEYS:
                result[key_text] = "[REDACTED]"
            elif key_text.lower() in {"url", "target_url", "targeturl"} and isinstance(item, str):
                result[key_text] = sanitize_url_for_audit(item)[0]
            else:
                result[key_text] = redact_for_audit(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_for_audit(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sanitize_url_for_audit(raw_url: str) -> tuple[str, str]:
    """Return a query-free URL and a correlation hash of the original URL."""

    raw_hash = _sha256_text(raw_url)
    try:
        parsed = urlsplit(raw_url)
        hostname = parsed.hostname
        if not hostname:
            return "[invalid-url]", raw_hash

        try:
            host = ipaddress.ip_address(hostname).compressed
        except ValueError:
            host = hostname.encode("idna").decode("ascii").lower().rstrip(".")

        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = parsed.port
        default_port = (parsed.scheme.lower() == "http" and port == 80) or (
            parsed.scheme.lower() == "https" and port == 443
        )
        netloc = host if port is None or default_port else f"{host}:{port}"
        safe = SplitResult(parsed.scheme.lower(), netloc, parsed.path or "/", "", "")
        return urlunsplit(safe), raw_hash
    except (UnicodeError, ValueError):
        return "[invalid-url]", raw_hash


@dataclass(frozen=True, slots=True)
class DatabaseHealth:
    connected: bool
    journal_mode: str
    foreign_keys: bool
    migration_version: int


@dataclass(frozen=True, slots=True)
class StoredAction:
    action_id: str
    state: str
    action_type: str
    result: dict[str, Any] | None
    error_code: str | None


@dataclass(frozen=True, slots=True)
class StoredPlanMetadata:
    plan_id: str
    session_id: str
    task_id: str
    revision: int
    step_count: int
    plan_sha256: str


@dataclass(frozen=True, slots=True)
class StoredApproval:
    approval_id: str
    action_id: str
    action_type: str
    action_sha256: str
    decision: str | None
    decision_actor: str | None


@dataclass(frozen=True, slots=True)
class _ActionIdentity:
    session_id: str
    step_id: str
    ordinal: int
    action_type: str
    request_json: str
    request_hash: str
    idempotency_class: str
    expected_observation_id: str | None
    expected_page_id: str | None
    expected_origin: str | None


@dataclass(frozen=True, slots=True)
class _FirewallIdentity:
    sanitized_url: str
    raw_url_hash: str
    purpose: str
    allowed: bool
    reason_code: str
    matched_rule_id: str | None
    resolved_ips_json: str
    policy_hash: str


_MIGRATION_001 = r"""
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    client_request_id TEXT UNIQUE,
    task_summary TEXT NOT NULL,
    status TEXT NOT NULL CHECK(
        status IN ('created','running','paused','succeeded','failed','cancelled')
    ),
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE steps (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    step_number INTEGER NOT NULL CHECK(step_number >= 0),
    status TEXT NOT NULL CHECK(status IN ('created','running','succeeded','failed','cancelled')),
    decision_summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, step_number)
);

CREATE TABLE actions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    step_id TEXT NOT NULL REFERENCES steps(id) ON DELETE RESTRICT,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    action_type TEXT NOT NULL,
    state TEXT NOT NULL CHECK(
        state IN (
            'created','validated','allowed','blocked','dispatched',
            'succeeded','failed','timed_out','unknown'
        )
    ),
    request_json TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    idempotency_class TEXT NOT NULL,
    expected_page_id TEXT,
    expected_origin TEXT,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
    result_json TEXT,
    error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    dispatched_at TEXT,
    completed_at TEXT,
    UNIQUE(step_id, ordinal)
);

CREATE TABLE action_attempts (
    action_id TEXT NOT NULL REFERENCES actions(id) ON DELETE RESTRICT,
    attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
    outcome TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    PRIMARY KEY(action_id, attempt_number)
);

CREATE TABLE firewall_decisions (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id) ON DELETE RESTRICT,
    step_id TEXT REFERENCES steps(id) ON DELETE RESTRICT,
    action_id TEXT REFERENCES actions(id) ON DELETE RESTRICT,
    sanitized_url TEXT NOT NULL,
    raw_url_hash TEXT NOT NULL,
    purpose TEXT NOT NULL,
    allowed INTEGER NOT NULL CHECK(allowed IN (0,1)),
    reason_code TEXT NOT NULL,
    matched_rule_id TEXT,
    resolved_ips_json TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE audit_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    session_id TEXT REFERENCES sessions(id) ON DELETE RESTRICT,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('debug','info','warning','error','critical')),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE model_calls (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id) ON DELETE RESTRICT,
    step_id TEXT REFERENCES steps(id) ON DELETE RESTRICT,
    agent_name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency_ms INTEGER,
    error_code TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE firewall_rules (
    policy_hash TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    effect TEXT NOT NULL CHECK(effect IN ('allow','deny')),
    host TEXT NOT NULL,
    rule_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(policy_hash, rule_id)
);

CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_steps_session ON steps(session_id, step_number);
CREATE INDEX idx_actions_session ON actions(session_id, created_at);
CREATE INDEX idx_actions_step ON actions(step_id, ordinal);
CREATE INDEX idx_firewall_created ON firewall_decisions(created_at);
CREATE INDEX idx_firewall_action ON firewall_decisions(action_id);
CREATE INDEX idx_audit_session_sequence ON audit_events(session_id, sequence);
CREATE INDEX idx_model_calls_session ON model_calls(session_id, created_at);
CREATE INDEX idx_firewall_rules_policy ON firewall_rules(policy_hash);

CREATE TRIGGER firewall_decisions_no_update
BEFORE UPDATE ON firewall_decisions BEGIN
    SELECT RAISE(ABORT, 'firewall_decisions are immutable');
END;
CREATE TRIGGER firewall_decisions_no_delete
BEFORE DELETE ON firewall_decisions BEGIN
    SELECT RAISE(ABORT, 'firewall_decisions are immutable');
END;
CREATE TRIGGER audit_events_no_update
BEFORE UPDATE ON audit_events BEGIN
    SELECT RAISE(ABORT, 'audit_events are immutable');
END;
CREATE TRIGGER audit_events_no_delete
BEFORE DELETE ON audit_events BEGIN
    SELECT RAISE(ABORT, 'audit_events are immutable');
END;
CREATE TRIGGER firewall_rules_no_update
BEFORE UPDATE ON firewall_rules BEGIN
    SELECT RAISE(ABORT, 'firewall_rules are immutable');
END;
CREATE TRIGGER firewall_rules_no_delete
BEFORE DELETE ON firewall_rules BEGIN
    SELECT RAISE(ABORT, 'firewall_rules are immutable');
END;
"""

_MIGRATION_002 = r"""
ALTER TABLE sessions ADD COLUMN task_summary_sha256 TEXT;
ALTER TABLE actions ADD COLUMN expected_observation_id TEXT;

CREATE TABLE execution_plans (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK(revision BETWEEN 1 AND 3),
    step_count INTEGER NOT NULL CHECK(step_count BETWEEN 1 AND 100),
    plan_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, revision)
);

CREATE TABLE execution_plan_steps (
    plan_id TEXT NOT NULL REFERENCES execution_plans(id) ON DELETE RESTRICT,
    step_key TEXT NOT NULL,
    step_order INTEGER NOT NULL CHECK(step_order BETWEEN 1 AND 100),
    dependency_keys_json TEXT NOT NULL,
    allowed_action_types_json TEXT NOT NULL,
    max_attempts INTEGER NOT NULL CHECK(max_attempts BETWEEN 1 AND 10),
    created_at TEXT NOT NULL,
    PRIMARY KEY(plan_id, step_key),
    UNIQUE(plan_id, step_order)
);

CREATE TABLE approval_requests (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    task_id TEXT NOT NULL,
    plan_id TEXT NOT NULL REFERENCES execution_plans(id) ON DELETE RESTRICT,
    step_key TEXT NOT NULL,
    action_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    action_sha256 TEXT NOT NULL,
    risk TEXT NOT NULL CHECK(risk IN ('interaction','data_entry','external_side_effect')),
    reason_code TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE approval_decisions (
    id TEXT PRIMARY KEY,
    approval_id TEXT NOT NULL UNIQUE REFERENCES approval_requests(id) ON DELETE RESTRICT,
    decision TEXT NOT NULL CHECK(decision IN ('approved','rejected','expired','cancelled')),
    actor TEXT NOT NULL CHECK(actor IN ('user','system')),
    reason_code TEXT NOT NULL,
    decided_at TEXT NOT NULL
);

CREATE INDEX idx_execution_plans_session ON execution_plans(session_id, revision);
CREATE INDEX idx_plan_steps_order ON execution_plan_steps(plan_id, step_order);
CREATE INDEX idx_approval_requests_session ON approval_requests(session_id, requested_at);
CREATE INDEX idx_approval_requests_action ON approval_requests(action_id);

CREATE TRIGGER sessions_require_hashed_summary_insert
BEFORE INSERT ON sessions
WHEN NEW.task_summary != '[HASHED]' OR length(NEW.task_summary_sha256) != 64 BEGIN
    SELECT RAISE(ABORT, 'sessions require a hashed task summary');
END;
CREATE TRIGGER sessions_require_hashed_summary_update
BEFORE UPDATE OF task_summary, task_summary_sha256 ON sessions
WHEN NEW.task_summary != '[HASHED]' OR length(NEW.task_summary_sha256) != 64 BEGIN
    SELECT RAISE(ABORT, 'sessions require a hashed task summary');
END;
CREATE TRIGGER execution_plans_no_update
BEFORE UPDATE ON execution_plans BEGIN
    SELECT RAISE(ABORT, 'execution_plans are immutable');
END;
CREATE TRIGGER execution_plans_no_delete
BEFORE DELETE ON execution_plans BEGIN
    SELECT RAISE(ABORT, 'execution_plans are immutable');
END;
CREATE TRIGGER execution_plan_steps_no_update
BEFORE UPDATE ON execution_plan_steps BEGIN
    SELECT RAISE(ABORT, 'execution_plan_steps are immutable');
END;
CREATE TRIGGER execution_plan_steps_no_delete
BEFORE DELETE ON execution_plan_steps BEGIN
    SELECT RAISE(ABORT, 'execution_plan_steps are immutable');
END;
CREATE TRIGGER approval_requests_no_update
BEFORE UPDATE ON approval_requests BEGIN
    SELECT RAISE(ABORT, 'approval_requests are immutable');
END;
CREATE TRIGGER approval_requests_no_delete
BEFORE DELETE ON approval_requests BEGIN
    SELECT RAISE(ABORT, 'approval_requests are immutable');
END;
CREATE TRIGGER approval_decisions_no_update
BEFORE UPDATE ON approval_decisions BEGIN
    SELECT RAISE(ABORT, 'approval_decisions are immutable');
END;
CREATE TRIGGER approval_decisions_no_delete
BEFORE DELETE ON approval_decisions BEGIN
    SELECT RAISE(ABORT, 'approval_decisions are immutable');
END;
CREATE TRIGGER model_calls_no_update
BEFORE UPDATE ON model_calls BEGIN
    SELECT RAISE(ABORT, 'model_calls are immutable');
END;
CREATE TRIGGER model_calls_no_delete
BEFORE DELETE ON model_calls BEGIN
    SELECT RAISE(ABORT, 'model_calls are immutable');
END;
"""

_MIGRATIONS: Final[tuple[tuple[int, str], ...]] = (
    (1, _MIGRATION_001),
    (2, _MIGRATION_002),
)


class AxisDatabase:
    """A single-connection async SQLite store with serialized writes."""

    def __init__(
        self,
        path: str | Path,
        *,
        trusted_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path)
        self._connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._trusted_clock = trusted_clock or (lambda: datetime.now(UTC))

    async def __aenter__(self) -> AxisDatabase:
        await self.connect()
        await self.initialize()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("database is not connected")
        return self._connection

    def _trusted_now(self) -> datetime:
        now = self._trusted_clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise RuntimeError("trusted clock must return a timezone-aware datetime")
        return now.astimezone(UTC)

    async def connect(self) -> None:
        if self._connection is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            self.path.parent.chmod(0o700)
        self._connection = await aiosqlite.connect(self.path, timeout=5.0)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA foreign_keys=ON")
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA synchronous=FULL")
        await self._connection.execute("PRAGMA busy_timeout=5000")
        await self._connection.commit()
        if os.name != "nt" and self.path.exists():
            self.path.chmod(0o600)

    async def close(self) -> None:
        if self._connection is None:
            return
        await self._connection.close()
        self._connection = None

    async def initialize(self) -> None:
        connection = self.connection
        try:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    checksum TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

        for version, sql in _MIGRATIONS:
            checksum = _sha256_text(sql)
            async with self._write_lock:
                try:
                    await connection.execute("BEGIN IMMEDIATE")
                    row = await self._fetchone(
                        "SELECT checksum FROM schema_migrations WHERE version = ?", (version,)
                    )
                    if row is not None:
                        if row["checksum"] != checksum:
                            raise RuntimeError(f"migration {version} checksum mismatch")
                    else:
                        for statement in _migration_statements(sql):
                            await connection.execute(statement)
                        await connection.execute(
                            """
                            INSERT INTO schema_migrations(version, checksum, applied_at)
                            VALUES (?, ?, ?)
                            """,
                            (version, checksum, _utc_now()),
                        )
                    await connection.commit()
                except BaseException:
                    await connection.rollback()
                    raise

        await self._backfill_task_summary_hashes()

    async def _backfill_task_summary_hashes(self) -> None:
        """Remove any raw summaries created by schema version 1."""

        rows_cursor = await self.connection.execute(
            "SELECT id, task_summary FROM sessions WHERE task_summary_sha256 IS NULL"
        )
        try:
            rows = await rows_cursor.fetchall()
        finally:
            await rows_cursor.close()
        if not rows:
            return

        async with self._write_lock:
            try:
                await self.connection.execute("BEGIN IMMEDIATE")
                for row in rows:
                    await self.connection.execute(
                        """
                        UPDATE sessions
                        SET task_summary=?, task_summary_sha256=?, updated_at=?
                        WHERE id=? AND task_summary_sha256 IS NULL
                        """,
                        (
                            _HASHED_TASK_SUMMARY,
                            _sha256_text(str(row["task_summary"])),
                            _utc_now(),
                            row["id"],
                        ),
                    )
                await self.connection.commit()
            except BaseException:
                await self.connection.rollback()
                raise

    async def healthcheck(self) -> DatabaseHealth:
        await self.connection.execute("SELECT 1")
        journal = await self._fetchone("PRAGMA journal_mode")
        foreign_keys = await self._fetchone("PRAGMA foreign_keys")
        version = await self._fetchone(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
        )
        return DatabaseHealth(
            connected=True,
            journal_mode=str(journal[0]).lower() if journal else "unknown",
            foreign_keys=bool(foreign_keys[0]) if foreign_keys else False,
            migration_version=int(version["version"]) if version else 0,
        )

    async def create_session(
        self,
        *,
        task_summary: str,
        config_hash: str,
        task_id: UUID | str | None = None,
        session_id: UUID | str | None = None,
        client_request_id: str | None = None,
    ) -> str:
        if not isinstance(task_summary, str) or not task_summary.strip():
            raise ValueError("task_summary must be a non-empty string")
        now = _utc_now()
        generated_session_id = _canonical_uuid(session_id, "session_id")
        generated_task_id = _canonical_uuid(task_id, "task_id")
        summary_hash = _sha256_text(task_summary)
        async with self._write_lock:
            await self.connection.execute(
                """
                INSERT INTO sessions(
                    id, task_id, client_request_id, task_summary, task_summary_sha256,
                    status, config_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'created', ?, ?, ?)
                """,
                (
                    generated_session_id,
                    generated_task_id,
                    client_request_id,
                    _HASHED_TASK_SUMMARY,
                    summary_hash,
                    config_hash,
                    now,
                    now,
                ),
            )
            await self.connection.commit()
        return generated_session_id

    async def create_step(
        self,
        *,
        session_id: UUID | str,
        step_number: int,
        step_id: UUID | str | None = None,
    ) -> str:
        if step_number < 0:
            raise ValueError("step_number must be non-negative")
        now = _utc_now()
        generated_id = _canonical_uuid(step_id, "step_id")
        canonical_session_id = _canonical_uuid(session_id, "session_id")
        async with self._write_lock:
            await self.connection.execute(
                """
                INSERT INTO steps(id, session_id, step_number, status, created_at, updated_at)
                VALUES (?, ?, ?, 'created', ?, ?)
                """,
                (generated_id, canonical_session_id, step_number, now, now),
            )
            await self.connection.commit()
        return generated_id

    async def transition_session_status(
        self,
        session_id: UUID | str,
        status: SessionStatus,
    ) -> bool:
        """Apply one validated, idempotent session lifecycle transition."""

        canonical_session_id = _canonical_uuid(session_id, "session_id")
        allowed: dict[str, frozenset[str]] = {
            "created": frozenset({"running", "failed", "cancelled"}),
            "running": frozenset({"paused", "succeeded", "failed", "cancelled"}),
            "paused": frozenset({"running", "failed", "cancelled"}),
            "succeeded": frozenset(),
            "failed": frozenset(),
            "cancelled": frozenset(),
        }
        async with self._write_lock:
            row = await self._fetchone(
                "SELECT status FROM sessions WHERE id=?",
                (canonical_session_id,),
            )
            if row is None:
                raise ValueError("session status transition references an unknown session")
            current = str(row["status"])
            if current == status:
                return False
            if current not in allowed or status not in allowed[current]:
                raise ValueError(f"invalid session status transition: {current} -> {status}")
            now = _utc_now()
            cursor = await self.connection.execute(
                "UPDATE sessions SET status=?, updated_at=? WHERE id=? AND status=?",
                (status, now, canonical_session_id, current),
            )
            if cursor.rowcount != 1:
                await self.connection.rollback()
                raise RuntimeError("session status changed concurrently")
            await self.connection.execute(
                """
                INSERT INTO audit_events(
                    id, session_id, event_type, severity, payload_json, created_at
                ) VALUES (?, ?, 'session.status', 'info', ?, ?)
                """,
                (
                    str(uuid4()),
                    canonical_session_id,
                    _canonical_json({"from": current, "to": status}),
                    now,
                ),
            )
            await self.connection.commit()
        return True

    async def transition_step_status(
        self,
        step_id: UUID | str,
        status: StepStatus,
    ) -> bool:
        """Apply one validated, idempotent step lifecycle transition."""

        canonical_step_id = _canonical_uuid(step_id, "step_id")
        allowed: dict[str, frozenset[str]] = {
            "created": frozenset({"running", "failed", "cancelled"}),
            "running": frozenset({"succeeded", "failed", "cancelled"}),
            "succeeded": frozenset(),
            "failed": frozenset(),
            "cancelled": frozenset(),
        }
        async with self._write_lock:
            row = await self._fetchone(
                "SELECT session_id, status FROM steps WHERE id=?",
                (canonical_step_id,),
            )
            if row is None:
                raise ValueError("step status transition references an unknown step")
            current = str(row["status"])
            if current == status:
                return False
            if current not in allowed or status not in allowed[current]:
                raise ValueError(f"invalid step status transition: {current} -> {status}")
            session_id = str(row["session_id"])
            now = _utc_now()
            cursor = await self.connection.execute(
                "UPDATE steps SET status=?, updated_at=? WHERE id=? AND status=?",
                (status, now, canonical_step_id, current),
            )
            if cursor.rowcount != 1:
                await self.connection.rollback()
                raise RuntimeError("step status changed concurrently")
            await self.connection.execute(
                """
                INSERT INTO audit_events(
                    id, session_id, event_type, severity, payload_json, created_at
                ) VALUES (?, ?, 'step.status', 'info', ?, ?)
                """,
                (
                    str(uuid4()),
                    session_id,
                    _canonical_json({"stepId": canonical_step_id, "from": current, "to": status}),
                    now,
                ),
            )
            await self.connection.commit()
        return True

    async def record_plan_metadata(self, plan: ExecutionPlan) -> StoredPlanMetadata:
        """Atomically append a plan revision without storing plan prose."""

        session_id = str(plan.session_id)
        task_id = str(plan.task_id)
        session = await self._fetchone(
            "SELECT task_id FROM sessions WHERE id=?",
            (session_id,),
        )
        if session is None:
            raise ValueError("execution plan references an unknown session")
        if str(session["task_id"]) != task_id:
            raise ValueError("execution plan task_id does not match its session")

        serialized = json.dumps(
            plan.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            separators=(",", ":"),
        )
        plan_hash = _sha256_text(serialized)
        existing = await self.get_plan_metadata(plan.plan_id)
        if existing is not None:
            if existing.plan_sha256 != plan_hash or existing.session_id != session_id:
                raise ValueError("plan_id is already associated with different metadata")
            return existing

        created_at = plan.created_at.astimezone(UTC).isoformat()
        async with self._write_lock:
            try:
                await self.connection.execute("BEGIN IMMEDIATE")
                await self.connection.execute(
                    """
                    INSERT INTO execution_plans(
                        id, session_id, task_id, revision, step_count, plan_sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(plan.plan_id),
                        session_id,
                        task_id,
                        plan.revision,
                        len(plan.steps),
                        plan_hash,
                        created_at,
                    ),
                )
                for step in plan.steps:
                    await self.connection.execute(
                        """
                        INSERT INTO execution_plan_steps(
                            plan_id, step_key, step_order, dependency_keys_json,
                            allowed_action_types_json, max_attempts, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(plan.plan_id),
                            step.key,
                            step.order,
                            _canonical_json(list(step.depends_on)),
                            _canonical_json(list(step.allowed_action_types)),
                            step.max_attempts,
                            created_at,
                        ),
                    )
                await self.connection.commit()
            except aiosqlite.IntegrityError:
                await self.connection.rollback()
                existing = await self.get_plan_metadata(plan.plan_id)
                if existing is not None and existing.plan_sha256 == plan_hash:
                    return existing
                raise
            except Exception:
                await self.connection.rollback()
                raise

        stored = await self.get_plan_metadata(plan.plan_id)
        if stored is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("persisted plan metadata could not be reloaded")
        return stored

    async def get_plan_metadata(self, plan_id: UUID | str) -> StoredPlanMetadata | None:
        canonical_plan_id = _canonical_uuid(plan_id, "plan_id")
        row = await self._fetchone(
            """
            SELECT id, session_id, task_id, revision, step_count, plan_sha256
            FROM execution_plans WHERE id=?
            """,
            (canonical_plan_id,),
        )
        if row is None:
            return None
        return StoredPlanMetadata(
            plan_id=str(row["id"]),
            session_id=str(row["session_id"]),
            task_id=str(row["task_id"]),
            revision=int(row["revision"]),
            step_count=int(row["step_count"]),
            plan_sha256=str(row["plan_sha256"]),
        )

    async def record_model_call(
        self,
        *,
        session_id: UUID | str,
        agent_name: Literal["planner", "navigator"],
        provider: str,
        model: str,
        status: Literal["succeeded", "failed", "cancelled", "timed_out"],
        step_id: UUID | str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: int | None = None,
        error_code: str | None = None,
        call_id: UUID | str | None = None,
    ) -> str:
        """Append non-sensitive provider/model usage metadata."""

        if agent_name not in {"planner", "navigator"}:
            raise ValueError("agent_name must be planner or navigator")
        if status not in {"succeeded", "failed", "cancelled", "timed_out"}:
            raise ValueError("invalid model call status")
        for name, value, maximum in (
            ("provider", provider, 128),
            ("model", model, 256),
        ):
            if not value or value != value.strip() or len(value) > maximum:
                raise ValueError(f"{name} must be non-empty, trimmed, and bounded")
        for metric_name, metric_value in (
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
            ("latency_ms", latency_ms),
        ):
            if metric_value is not None and (type(metric_value) is not int or metric_value < 0):
                raise ValueError(f"{metric_name} must be a non-negative integer")
        if error_code is not None and (
            not error_code or error_code != error_code.strip() or len(error_code) > 128
        ):
            raise ValueError("error_code must be non-empty, trimmed, and bounded")
        if status == "succeeded" and error_code is not None:
            raise ValueError("successful model calls cannot include error_code")

        generated_id = _canonical_uuid(call_id, "call_id")
        canonical_session_id = _canonical_uuid(session_id, "session_id")
        canonical_step_id = _canonical_uuid(step_id, "step_id") if step_id else None
        async with self._write_lock:
            try:
                await self.connection.execute(
                    """
                    INSERT INTO model_calls(
                        id, session_id, step_id, agent_name, provider, model, status,
                        input_tokens, output_tokens, latency_ms, error_code, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generated_id,
                        canonical_session_id,
                        canonical_step_id,
                        agent_name,
                        provider,
                        model,
                        status,
                        input_tokens,
                        output_tokens,
                        latency_ms,
                        error_code,
                        _utc_now(),
                    ),
                )
                await self.connection.commit()
            except Exception:
                await self.connection.rollback()
                raise
        return generated_id

    async def record_approval_request(self, request: ApprovalRequest) -> StoredApproval:
        """Append an action digest and approval metadata, never the raw action."""

        plan = await self.get_plan_metadata(request.plan_id)
        if (
            plan is None
            or plan.session_id != str(request.session_id)
            or plan.task_id != str(request.task_id)
        ):
            raise ValueError("approval request references an unknown plan/session")
        step = await self._fetchone(
            "SELECT 1 FROM execution_plan_steps WHERE plan_id=? AND step_key=?",
            (str(request.plan_id), request.step_key),
        )
        if step is None:
            raise ValueError("approval request references an unknown plan step")

        existing = await self.get_approval(request.approval_id)
        if existing is not None:
            if (
                existing.action_id != str(request.action_id)
                or existing.action_type != request.action_type
                or existing.action_sha256 != request.action_sha256
            ):
                raise ValueError("approval_id is already associated with a different action")
            return existing

        async with self._write_lock:
            try:
                await self.connection.execute(
                    """
                    INSERT INTO approval_requests(
                        id, session_id, task_id, plan_id, step_key, action_id,
                        observation_id, action_type, action_sha256, risk, reason_code,
                        requested_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(request.approval_id),
                        str(request.session_id),
                        str(request.task_id),
                        str(request.plan_id),
                        request.step_key,
                        str(request.action_id),
                        request.observation_id,
                        request.action_type,
                        request.action_sha256,
                        request.risk.value,
                        request.reason_code,
                        request.requested_at.astimezone(UTC).isoformat(),
                        request.expires_at.astimezone(UTC).isoformat(),
                    ),
                )
                await self.connection.commit()
            except aiosqlite.IntegrityError:
                await self.connection.rollback()
                existing = await self.get_approval(request.approval_id)
                if (
                    existing is not None
                    and existing.action_id == str(request.action_id)
                    and existing.action_sha256 == request.action_sha256
                ):
                    return existing
                raise
            except Exception:
                await self.connection.rollback()
                raise

        stored = await self.get_approval(request.approval_id)
        if stored is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("persisted approval request could not be reloaded")
        return stored

    async def record_approval_decision(self, decision: ApprovalDecision) -> StoredApproval:
        """Append exactly one terminal decision for an approval request."""

        decided_at = decision.decided_at.astimezone(UTC)
        async with self._write_lock:
            try:
                await self.connection.execute("BEGIN IMMEDIATE")
                request_row = await self._fetchone(
                    "SELECT requested_at, expires_at FROM approval_requests WHERE id=?",
                    (str(decision.approval_id),),
                )
                if request_row is None:
                    raise ValueError("approval decision references an unknown request")

                existing = await self.get_approval(decision.approval_id)
                if existing is not None and existing.decision is not None:
                    if existing.decision != decision.decision.value:
                        raise ValueError("approval request already has a different decision")
                    await self.connection.commit()
                    return existing

                requested_at = datetime.fromisoformat(str(request_row["requested_at"])).astimezone(
                    UTC
                )
                expires_at = datetime.fromisoformat(str(request_row["expires_at"])).astimezone(UTC)
                if decided_at < requested_at:
                    raise ValueError("approval decision predates its request")
                if decision.decision.value == "approved" and self._trusted_now() >= expires_at:
                    raise ValueError("expired approval request cannot be approved")

                await self.connection.execute(
                    """
                    INSERT INTO approval_decisions(
                        id, approval_id, decision, actor, reason_code, decided_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(decision.decision_id),
                        str(decision.approval_id),
                        decision.decision.value,
                        decision.actor,
                        decision.reason_code,
                        decided_at.isoformat(),
                    ),
                )
                await self.connection.commit()
            except BaseException:
                await self.connection.rollback()
                raise

        stored = await self.get_approval(decision.approval_id)
        if stored is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("persisted approval decision could not be reloaded")
        return stored

    async def get_approval(self, approval_id: UUID | str) -> StoredApproval | None:
        canonical_approval_id = _canonical_uuid(approval_id, "approval_id")
        row = await self._fetchone(
            """
            SELECT request.id, request.action_id, request.action_type, request.action_sha256,
                   decision.decision, decision.actor
            FROM approval_requests AS request
            LEFT JOIN approval_decisions AS decision ON decision.approval_id=request.id
            WHERE request.id=?
            """,
            (canonical_approval_id,),
        )
        if row is None:
            return None
        return StoredApproval(
            approval_id=str(row["id"]),
            action_id=str(row["action_id"]),
            action_type=str(row["action_type"]),
            action_sha256=str(row["action_sha256"]),
            decision=str(row["decision"]) if row["decision"] is not None else None,
            decision_actor=str(row["actor"]) if row["actor"] is not None else None,
        )

    async def has_valid_action_approval(
        self,
        *,
        session_id: UUID | str,
        task_id: UUID | str,
        action_id: UUID | str,
        observation_id: str | None,
        action_type: ProductionActionType,
        action_payload: Mapping[str, Any],
    ) -> bool:
        """Verify one current user decision against the exact action digest."""

        if observation_id is None:
            return False
        canonical_session_id = _canonical_uuid(session_id, "session_id")
        canonical_task_id = _canonical_uuid(task_id, "task_id")
        canonical_action_id = _canonical_uuid(action_id, "action_id")
        action_sha256 = _sha256_text(
            json.dumps(action_payload, sort_keys=True, separators=(",", ":"))
        )
        rows = tuple(
            await self.connection.execute_fetchall(
                """
                SELECT request.session_id, request.task_id, request.observation_id,
                       request.action_type, request.action_sha256, request.requested_at,
                       request.expires_at, decision.decision, decision.actor,
                       decision.decided_at
                FROM approval_requests AS request
                JOIN approval_decisions AS decision ON decision.approval_id=request.id
                WHERE request.action_id=?
                """,
                (canonical_action_id,),
            )
        )
        if len(rows) != 1:
            return False
        row = rows[0]
        if (
            str(row["session_id"]) != canonical_session_id
            or str(row["task_id"]) != canonical_task_id
            or str(row["observation_id"]) != observation_id
            or str(row["action_type"]) != action_type
            or str(row["action_sha256"]) != action_sha256
            or str(row["decision"]) != "approved"
            or str(row["actor"]) != "user"
        ):
            return False
        try:
            requested_at = datetime.fromisoformat(str(row["requested_at"])).astimezone(UTC)
            expires_at = datetime.fromisoformat(str(row["expires_at"])).astimezone(UTC)
            decided_at = datetime.fromisoformat(str(row["decided_at"])).astimezone(UTC)
        except ValueError:
            return False
        return requested_at <= decided_at < expires_at and self._trusted_now() < expires_at

    async def _require_step_session_ownership(self, *, step_id: str, session_id: str) -> None:
        row = await self._fetchone(
            """
            SELECT sessions.task_id
            FROM steps
            JOIN sessions ON sessions.id=steps.session_id
            WHERE steps.id=? AND steps.session_id=?
            """,
            (step_id, session_id),
        )
        if row is None:
            raise ValueError("step_id does not belong to session_id")

    async def validate_execution_context(
        self,
        *,
        session_id: UUID | str,
        task_id: UUID | str,
        step_id: UUID | str,
    ) -> None:
        """Verify the runtime-owned task/session/step relationship before dispatch."""

        canonical_session_id = _canonical_uuid(session_id, "session_id")
        canonical_task_id = _canonical_uuid(task_id, "task_id")
        canonical_step_id = _canonical_uuid(step_id, "step_id")
        row = await self._fetchone(
            """
            SELECT 1
            FROM steps
            JOIN sessions ON sessions.id=steps.session_id
            WHERE steps.id=? AND steps.session_id=? AND sessions.task_id=?
            """,
            (canonical_step_id, canonical_session_id, canonical_task_id),
        )
        if row is None:
            raise ValueError("task_id, session_id, and step_id do not share one execution context")

    async def _get_matching_action(
        self,
        *,
        action_id: str,
        identity: _ActionIdentity,
        firewall_identity: _FirewallIdentity | None = None,
    ) -> StoredAction | None:
        row = await self._fetchone(
            """
            SELECT id, session_id, step_id, ordinal, action_type, state, request_json,
                   request_hash, idempotency_class, expected_observation_id,
                   expected_page_id, expected_origin, result_json, error_code
            FROM actions
            WHERE id=?
            """,
            (action_id,),
        )
        if row is None:
            return None

        stored_identity = _ActionIdentity(
            session_id=str(row["session_id"]),
            step_id=str(row["step_id"]),
            ordinal=int(row["ordinal"]),
            action_type=str(row["action_type"]),
            request_json=str(row["request_json"]),
            request_hash=str(row["request_hash"]),
            idempotency_class=str(row["idempotency_class"]),
            expected_observation_id=(
                str(row["expected_observation_id"])
                if row["expected_observation_id"] is not None
                else None
            ),
            expected_page_id=(
                str(row["expected_page_id"]) if row["expected_page_id"] is not None else None
            ),
            expected_origin=(
                str(row["expected_origin"]) if row["expected_origin"] is not None else None
            ),
        )
        if stored_identity != identity:
            raise ValueError(
                "action_id is already associated with different context, payload, or preconditions"
            )

        if firewall_identity is not None:
            firewall_row = await self._fetchone(
                """
                SELECT sanitized_url, raw_url_hash, purpose, allowed, reason_code,
                       matched_rule_id, resolved_ips_json, policy_hash
                FROM firewall_decisions
                WHERE action_id=?
                """,
                (action_id,),
            )
            if firewall_row is None:
                raise ValueError("action_id is already associated with a missing firewall decision")
            stored_firewall_identity = _FirewallIdentity(
                sanitized_url=str(firewall_row["sanitized_url"]),
                raw_url_hash=str(firewall_row["raw_url_hash"]),
                purpose=str(firewall_row["purpose"]),
                allowed=bool(firewall_row["allowed"]),
                reason_code=str(firewall_row["reason_code"]),
                matched_rule_id=(
                    str(firewall_row["matched_rule_id"])
                    if firewall_row["matched_rule_id"] is not None
                    else None
                ),
                resolved_ips_json=str(firewall_row["resolved_ips_json"]),
                policy_hash=str(firewall_row["policy_hash"]),
            )
            if stored_firewall_identity != firewall_identity:
                raise ValueError(
                    "action_id is already associated with a different firewall decision"
                )

        return StoredAction(
            action_id=str(row["id"]),
            state=str(row["state"]),
            action_type=str(row["action_type"]),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error_code=str(row["error_code"]) if row["error_code"] is not None else None,
        )

    async def record_action(
        self,
        *,
        session_id: UUID | str,
        step_id: UUID | str,
        action_id: UUID | str,
        ordinal: int,
        action_type: ProductionActionType,
        action_payload: Mapping[str, Any],
        idempotency_class: str,
        expected_observation_id: str | None = None,
        expected_page_id: str | None = None,
        expected_origin: str | None = None,
    ) -> StoredAction:
        """Persist one non-URL production action without a firewall decision.

        URL-bearing actions must use ``record_action_with_firewall_decision``.
        The action is placed in ``allowed`` state so the dispatch transition is
        identical after either persistence path.
        """

        if action_type not in PRODUCTION_ACTION_TYPES or action_type in {
            "go_to_url",
            "open_tab",
        }:
            raise ValueError("record_action accepts only non-URL production actions")
        if ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if not idempotency_class or idempotency_class != idempotency_class.strip():
            raise ValueError("idempotency_class must be non-empty and trimmed")
        if len(idempotency_class) > 64:
            raise ValueError("idempotency_class must be at most 64 characters")
        self._validate_observation_id(expected_observation_id)

        canonical_session_id = _canonical_uuid(session_id, "session_id")
        canonical_step_id = _canonical_uuid(step_id, "step_id")
        canonical_action_id = _canonical_uuid(action_id, "action_id")
        request_json = _canonical_json(redact_for_audit(action_payload))
        request_hash = _sha256_text(
            json.dumps(action_payload, sort_keys=True, separators=(",", ":"), default=str)
        )
        identity = _ActionIdentity(
            session_id=canonical_session_id,
            step_id=canonical_step_id,
            ordinal=ordinal,
            action_type=action_type,
            request_json=request_json,
            request_hash=request_hash,
            idempotency_class=idempotency_class,
            expected_observation_id=expected_observation_id,
            expected_page_id=expected_page_id,
            expected_origin=expected_origin,
        )
        existing = await self._get_matching_action(
            action_id=canonical_action_id,
            identity=identity,
        )
        if existing is not None:
            return existing

        now = _utc_now()
        async with self._write_lock:
            try:
                await self.connection.execute("BEGIN IMMEDIATE")
                await self._require_step_session_ownership(
                    step_id=canonical_step_id,
                    session_id=canonical_session_id,
                )
                existing = await self._get_matching_action(
                    action_id=canonical_action_id,
                    identity=identity,
                )
                if existing is not None:
                    await self.connection.commit()
                    return existing
                await self.connection.execute(
                    """
                    INSERT INTO actions(
                        id, session_id, step_id, ordinal, action_type, state, request_json,
                        request_hash, idempotency_class, expected_observation_id,
                        expected_page_id, expected_origin, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'allowed', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        canonical_action_id,
                        canonical_session_id,
                        canonical_step_id,
                        ordinal,
                        action_type,
                        request_json,
                        request_hash,
                        idempotency_class,
                        expected_observation_id,
                        expected_page_id,
                        expected_origin,
                        now,
                        now,
                    ),
                )
                await self.connection.execute(
                    """
                    INSERT INTO audit_events(
                        id, session_id, event_type, severity, payload_json, created_at
                    ) VALUES (?, ?, 'action.allowed', 'info', ?, ?)
                    """,
                    (
                        str(uuid4()),
                        canonical_session_id,
                        _canonical_json(
                            {
                                "actionId": canonical_action_id,
                                "actionType": action_type,
                            }
                        ),
                        now,
                    ),
                )
                await self.connection.commit()
            except BaseException:
                await self.connection.rollback()
                raise

        stored = await self.get_action(canonical_action_id)
        if stored is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("persisted action could not be reloaded")
        return stored

    async def record_action_with_firewall_decision(
        self,
        *,
        session_id: UUID | str,
        step_id: UUID | str,
        action_id: UUID | str,
        ordinal: int,
        action_type: str,
        action_payload: Mapping[str, Any],
        idempotency_class: str,
        raw_url: str,
        purpose: str,
        allowed: bool,
        reason_code: str,
        policy_hash: str,
        matched_rule_id: UUID | str | None = None,
        resolved_ips: Sequence[str] = (),
        expected_observation_id: str | None = None,
        expected_page_id: str | None = None,
        expected_origin: str | None = None,
        decision_id: UUID | str | None = None,
    ) -> StoredAction:
        """Atomically persist an action and the policy decision made for it.

        Re-delivery of an existing action ID returns the original row and never
        overwrites the recorded request or policy decision.
        """

        canonical_session_id = _canonical_uuid(session_id, "session_id")
        canonical_step_id = _canonical_uuid(step_id, "step_id")
        canonical_action_id = _canonical_uuid(action_id, "action_id")
        self._validate_observation_id(expected_observation_id)
        redacted_payload = redact_for_audit(action_payload)
        request_json = _canonical_json(redacted_payload)
        request_hash = _sha256_text(
            json.dumps(action_payload, sort_keys=True, separators=(",", ":"), default=str)
        )
        sanitized_url, raw_url_hash = sanitize_url_for_audit(raw_url)
        now = _utc_now()
        state: ActionState = "allowed" if allowed else "blocked"
        generated_decision_id = _canonical_uuid(decision_id, "decision_id")
        canonical_matched_rule_id = str(matched_rule_id) if matched_rule_id else None
        resolved_ips_json = _canonical_json(list(resolved_ips))
        identity = _ActionIdentity(
            session_id=canonical_session_id,
            step_id=canonical_step_id,
            ordinal=ordinal,
            action_type=action_type,
            request_json=request_json,
            request_hash=request_hash,
            idempotency_class=idempotency_class,
            expected_observation_id=expected_observation_id,
            expected_page_id=expected_page_id,
            expected_origin=expected_origin,
        )
        firewall_identity = _FirewallIdentity(
            sanitized_url=sanitized_url,
            raw_url_hash=raw_url_hash,
            purpose=purpose,
            allowed=allowed,
            reason_code=reason_code,
            matched_rule_id=canonical_matched_rule_id,
            resolved_ips_json=resolved_ips_json,
            policy_hash=policy_hash,
        )
        existing = await self._get_matching_action(
            action_id=canonical_action_id,
            identity=identity,
            firewall_identity=firewall_identity,
        )
        if existing is not None:
            return existing

        async with self._write_lock:
            try:
                await self.connection.execute("BEGIN IMMEDIATE")
                await self._require_step_session_ownership(
                    step_id=canonical_step_id,
                    session_id=canonical_session_id,
                )
                existing = await self._get_matching_action(
                    action_id=canonical_action_id,
                    identity=identity,
                    firewall_identity=firewall_identity,
                )
                if existing is not None:
                    await self.connection.commit()
                    return existing
                await self.connection.execute(
                    """
                    INSERT INTO actions(
                        id, session_id, step_id, ordinal, action_type, state, request_json,
                        request_hash, idempotency_class, expected_observation_id,
                        expected_page_id, expected_origin, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        canonical_action_id,
                        canonical_session_id,
                        canonical_step_id,
                        ordinal,
                        action_type,
                        state,
                        request_json,
                        request_hash,
                        idempotency_class,
                        expected_observation_id,
                        expected_page_id,
                        expected_origin,
                        now,
                        now,
                    ),
                )
                await self.connection.execute(
                    """
                    INSERT INTO firewall_decisions(
                        id, session_id, step_id, action_id, sanitized_url, raw_url_hash,
                        purpose, allowed, reason_code, matched_rule_id, resolved_ips_json,
                        policy_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generated_decision_id,
                        canonical_session_id,
                        canonical_step_id,
                        canonical_action_id,
                        sanitized_url,
                        raw_url_hash,
                        purpose,
                        int(allowed),
                        reason_code,
                        canonical_matched_rule_id,
                        resolved_ips_json,
                        policy_hash,
                        now,
                    ),
                )
                await self.connection.execute(
                    """
                    INSERT INTO audit_events(
                        id, session_id, event_type, severity, payload_json, created_at
                    )
                    VALUES (?, ?, 'firewall.decision', ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        canonical_session_id,
                        "info" if allowed else "warning",
                        _canonical_json(
                            {
                                "actionId": canonical_action_id,
                                "allowed": allowed,
                                "reasonCode": reason_code,
                                "target": sanitized_url,
                            }
                        ),
                        now,
                    ),
                )
                await self.connection.commit()
            except BaseException:
                await self.connection.rollback()
                raise

        stored = await self.get_action(canonical_action_id)
        if stored is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("persisted action could not be reloaded")
        return stored

    async def mark_dispatched(self, action_id: UUID | str) -> bool:
        """Persist dispatch before browser invocation.

        Returns ``False`` for duplicates or for blocked/non-dispatchable actions.
        """

        now = _utc_now()
        canonical_action_id = _canonical_uuid(action_id, "action_id")
        async with self._write_lock:
            cursor = await self.connection.execute(
                """
                UPDATE actions
                SET state='dispatched', attempts=attempts+1, dispatched_at=?, updated_at=?
                WHERE id=? AND state='allowed'
                """,
                (now, now, canonical_action_id),
            )
            if cursor.rowcount != 1:
                await self.connection.rollback()
                return False
            row = await self._fetchone(
                "SELECT attempts FROM actions WHERE id=?", (canonical_action_id,)
            )
            if row is None:  # pragma: no cover - protected by the successful update above
                await self.connection.rollback()
                raise RuntimeError("dispatched action disappeared before attempt persistence")
            await self.connection.execute(
                """
                INSERT INTO action_attempts(action_id, attempt_number, outcome, started_at)
                VALUES (?, ?, 'dispatched', ?)
                """,
                (canonical_action_id, int(row["attempts"]), now),
            )
            await self.connection.commit()
            return True

    async def complete_action(
        self,
        action_id: UUID | str,
        *,
        state: Literal["succeeded", "failed", "timed_out"],
        result: Mapping[str, Any] | None = None,
        error_code: str | None = None,
    ) -> bool:
        now = _utc_now()
        canonical_action_id = _canonical_uuid(action_id, "action_id")
        result_json = _canonical_json(redact_for_audit(result)) if result is not None else None
        async with self._write_lock:
            cursor = await self.connection.execute(
                """
                UPDATE actions
                SET state=?, result_json=?, error_code=?, completed_at=?, updated_at=?
                WHERE id=? AND state='dispatched'
                """,
                (state, result_json, error_code, now, now, canonical_action_id),
            )
            if cursor.rowcount != 1:
                await self.connection.rollback()
                return False
            await self.connection.execute(
                """
                UPDATE action_attempts
                SET outcome=?, completed_at=?
                WHERE action_id=? AND attempt_number=(SELECT attempts FROM actions WHERE id=?)
                """,
                (state, now, canonical_action_id, canonical_action_id),
            )
            await self.connection.commit()
            return True

    async def recover_dispatched_actions(self) -> int:
        """Mark crash-interrupted actions unknown instead of replaying them."""

        now = _utc_now()
        async with self._write_lock:
            cursor = await self.connection.execute(
                "UPDATE actions SET state='unknown', updated_at=? WHERE state='dispatched'",
                (now,),
            )
            count = cursor.rowcount
            await self.connection.commit()
        return count

    async def record_audit_event(
        self,
        *,
        event_type: str,
        payload: Mapping[str, Any],
        severity: Literal["debug", "info", "warning", "error", "critical"] = "info",
        session_id: UUID | str | None = None,
    ) -> str:
        event_id = str(uuid4())
        safe_payload = _canonical_json(redact_for_audit(payload))
        async with self._write_lock:
            await self.connection.execute(
                """
                INSERT INTO audit_events(
                    id, session_id, event_type, severity, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    _canonical_uuid(session_id, "session_id") if session_id else None,
                    event_type,
                    severity,
                    safe_payload,
                    _utc_now(),
                ),
            )
            await self.connection.commit()
        return event_id

    async def store_firewall_policy(
        self,
        *,
        policy_hash: str,
        rules: Sequence[Mapping[str, Any]],
    ) -> int:
        """Append an immutable, non-secret firewall policy snapshot."""

        now = _utc_now()
        inserted = 0
        async with self._write_lock:
            try:
                await self.connection.execute("BEGIN IMMEDIATE")
                for rule in rules:
                    safe_rule = redact_for_audit(rule)
                    rule_id = str(safe_rule.get("id", ""))
                    effect = str(safe_rule.get("effect", ""))
                    host = str(safe_rule.get("host", ""))
                    if not rule_id or effect not in {"allow", "deny"} or not host:
                        raise ValueError("firewall rule snapshot is missing id, effect, or host")
                    cursor = await self.connection.execute(
                        """
                        INSERT OR IGNORE INTO firewall_rules(
                            policy_hash, rule_id, effect, host, rule_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (policy_hash, rule_id, effect, host, _canonical_json(safe_rule), now),
                    )
                    inserted += cursor.rowcount
                await self.connection.commit()
            except Exception:
                await self.connection.rollback()
                raise
        return inserted

    async def get_action(self, action_id: UUID | str) -> StoredAction | None:
        canonical_action_id = _canonical_uuid(action_id, "action_id")
        row = await self._fetchone(
            "SELECT id, state, action_type, result_json, error_code FROM actions WHERE id=?",
            (canonical_action_id,),
        )
        if row is None:
            return None
        return StoredAction(
            action_id=row["id"],
            state=row["state"],
            action_type=row["action_type"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error_code=row["error_code"],
        )

    async def _fetchone(self, query: str, parameters: Sequence[Any] = ()) -> aiosqlite.Row | None:
        cursor = await self.connection.execute(query, parameters)
        try:
            return await cursor.fetchone()
        finally:
            await cursor.close()

    @staticmethod
    def _validate_observation_id(value: str | None) -> None:
        if value is not None and (
            len(value) != 32 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("expected_observation_id must be 32 lowercase hexadecimal characters")
