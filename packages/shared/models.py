from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum, StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class Action(IntEnum):
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    CLIMB = 4
    DESCEND = 5
    HOVER = 6
    PICK_UP = 7
    DELIVER = 8
    MOVE_TO_CHARGER = 9
    CHARGE = 10
    RETURN_TO_BASE = 11
    EMERGENCY_LAND = 12


class Priority(StrEnum):
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MissionStatus(StrEnum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MissionCreate(BaseModel):
    description: str = Field(default="", max_length=500)
    pickup: tuple[float, float] = (1.0, 1.0)
    destination: tuple[float, float]
    package_weight: float = Field(gt=0, le=10)
    deadline_steps: int = Field(default=120, ge=10, le=2000)
    priority: Priority = Priority.NORMAL

    @field_validator("description")
    @classmethod
    def strip_description(cls, value: str) -> str:
        return value.strip()


class Mission(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    status: MissionStatus = MissionStatus.PENDING
    created_at: datetime = Field(default_factory=utcnow)
    request: MissionCreate


class DroneBroadcast(BaseModel):
    drone_id: str
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    next_waypoint: tuple[float, float, float]
    planned_altitude: float
    battery: float = Field(ge=0, le=1)
    package_priority: Priority
    mission_status: MissionStatus
    timestamp: datetime = Field(default_factory=utcnow)


class DeliveredMessage(BaseModel):
    sender: str
    receiver: str
    payload: DroneBroadcast
    latency_ms: int
    dropped: bool
    stale: bool = False
    conflict_acknowledgement: bool = False


class SafetyDecision(BaseModel):
    allowed: bool
    original_action: str
    replacement_action: str | None = None
    violated_rules: list[str] = Field(default_factory=list)
    reason: str = ""
    rule_version: str = "1.0.0"
    collision: dict[str, float] | None = None


class DecisionEvent(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    simulation_id: UUID
    episode: int
    step: int
    timestamp: datetime = Field(default_factory=utcnow)
    drone_id: str
    position: tuple[float, float, float]
    observation_summary: dict[str, Any]
    received_messages: list[dict[str, Any]]
    available_actions: list[str]
    policy_q_values: dict[str, float]
    policy_action: str
    safety_result: SafetyDecision
    executed_action: str
    reward: float
    reward_breakdown: dict[str, float]
    model_version: str


class ApprovalRequest(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    action: str
    payload: dict[str, Any]
    status: str = "PENDING"
    created_at: datetime = Field(default_factory=utcnow)
    decided_at: datetime | None = None


class TrainingRequest(BaseModel):
    algorithm: str = Field(pattern="^(qmix|idqn)$")
    preset: str = Field(default="quick", pattern="^(quick|standard|full)$")
    seed: int = 42


class AgentRequest(BaseModel):
    request: str = Field(min_length=1, max_length=1000)

