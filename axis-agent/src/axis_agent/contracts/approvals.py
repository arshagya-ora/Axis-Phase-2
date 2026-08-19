"""Human-approval contracts containing metadata but no raw browser inputs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from axis_agent.contracts.actions import (
    ProductionActionType,
    parse_production_action,
    production_action_payload,
)
from axis_agent.contracts.base import ContractModel


class ApprovalRisk(StrEnum):
    INTERACTION = "interaction"
    DATA_ENTRY = "data_entry"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"


class ApprovalDecisionValue(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalRequest(ContractModel):
    """One short-lived, action-specific approval request."""

    approval_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    task_id: UUID
    plan_id: UUID
    step_key: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    action_id: UUID
    observation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    action_type: ProductionActionType
    action_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    risk: ApprovalRisk
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    requested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime

    @field_validator("requested_at", "expires_at")
    @classmethod
    def require_aware_timestamps(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approval timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_expiry(self) -> Self:
        if self.expires_at <= self.requested_at:
            raise ValueError("approval expiry must follow request time")
        if self.expires_at - self.requested_at > timedelta(minutes=5):
            raise ValueError("approval requests cannot remain valid for more than five minutes")
        return self

    @classmethod
    def for_action(
        cls,
        *,
        session_id: UUID,
        task_id: UUID,
        plan_id: UUID,
        step_key: str,
        action_id: UUID,
        observation_id: str,
        action: object,
        risk: ApprovalRisk,
        reason_code: str,
        ttl_seconds: int = 300,
    ) -> ApprovalRequest:
        if ttl_seconds < 1 or ttl_seconds > 300:
            raise ValueError("ttl_seconds must be between 1 and 300")
        production_action = parse_production_action(action)
        payload = production_action_payload(production_action)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        requested_at = datetime.now(UTC)
        return cls(
            session_id=session_id,
            task_id=task_id,
            plan_id=plan_id,
            step_key=step_key,
            action_id=action_id,
            observation_id=observation_id,
            action_type=production_action.type,
            action_sha256=hashlib.sha256(encoded).hexdigest(),
            risk=risk,
            reason_code=reason_code,
            requested_at=requested_at,
            expires_at=requested_at + timedelta(seconds=ttl_seconds),
        )


class ApprovalDecision(ContractModel):
    """Append-only human or runtime decision for one approval request."""

    decision_id: UUID = Field(default_factory=uuid4)
    approval_id: UUID
    decision: ApprovalDecisionValue
    actor: Literal["user", "system"]
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("decided_at")
    @classmethod
    def require_aware_decided_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decided_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_actor(self) -> Self:
        if self.decision is ApprovalDecisionValue.EXPIRED and self.actor != "system":
            raise ValueError("only the runtime can mark an approval expired")
        if self.decision is ApprovalDecisionValue.APPROVED and self.actor != "user":
            raise ValueError("only a user can approve an action")
        return self
