from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field

from axis_agent.contracts.base import ContractModel


class TaskStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskRequest(ContractModel):
    task_id: UUID = Field(default_factory=uuid4)
    prompt: str = Field(min_length=1, max_length=32_000)
    client_request_id: str | None = Field(default=None, max_length=128)
    page_id: str | None = None
    origin: str | None = None
