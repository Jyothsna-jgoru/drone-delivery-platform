from __future__ import annotations

import os
from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/drone_platform.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class RecordMixin:
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class Drone(RecordMixin, Base):
    __tablename__ = "drones"
    name: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(32), default="IDLE")


class Package(RecordMixin, Base):
    __tablename__ = "packages"
    weight: Mapped[float] = mapped_column(Float)
    priority: Mapped[str] = mapped_column(String(16))
    destination: Mapped[dict] = mapped_column(JSON)


class MissionRecord(RecordMixin, Base):
    __tablename__ = "missions"
    status: Mapped[str] = mapped_column(String(32))
    request: Mapped[dict] = mapped_column(JSON)


def event_model(name: str, table: str):
    return type(
        name,
        (RecordMixin, Base),
        {
            "__tablename__": table,
            "__annotations__": {"payload": Mapped[dict]},
            "payload": mapped_column(JSON),
        },
    )


SimulationRun = event_model("SimulationRun", "simulation_runs")
SimulationEpisode = event_model("SimulationEpisode", "simulation_episodes")
DecisionEventRecord = event_model("DecisionEventRecord", "decision_events")
DroneMessageRecord = event_model("DroneMessageRecord", "drone_messages")
CollisionRiskEvent = event_model("CollisionRiskEvent", "collision_risk_events")
SafetyIntervention = event_model("SafetyIntervention", "safety_interventions")
TelemetrySample = event_model("TelemetrySample", "telemetry_samples")
AgentExecution = event_model("AgentExecution", "agent_executions")
ToolCall = event_model("ToolCall", "tool_calls")
ApprovalRequestRecord = event_model("ApprovalRequestRecord", "approval_requests")
TrainingRun = event_model("TrainingRun", "training_runs")
ModelVersion = event_model("ModelVersion", "model_versions")
EvaluationResult = event_model("EvaluationResult", "evaluation_results")
Incident = event_model("Incident", "incidents")
IncidentTimelineEvent = event_model("IncidentTimelineEvent", "incident_timeline_events")


def create_schema() -> None:
    Base.metadata.create_all(engine)

