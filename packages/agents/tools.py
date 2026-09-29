from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field

from packages.shared.models import ApprovalRequest


class DroneQuery(BaseModel):
    drone_id: str


class MissionEnergyRequest(BaseModel):
    distance: float = Field(ge=0)
    package_weight: float = Field(ge=0, le=10)
    wind_speed: float = Field(ge=0)


class ProtectedAction(BaseModel):
    mission_id: str
    reason: str = Field(min_length=3, max_length=300)


class MissionValidationRequest(BaseModel):
    destination: tuple[float, float]
    package_weight: float = Field(gt=0)
    deadline_steps: int = Field(gt=0)


class RouteComparisonRequest(BaseModel):
    routes: list[list[tuple[float, float]]]


class RunQuery(BaseModel):
    run_id: str


@dataclass
class ToolDefinition:
    name: str
    category: str
    input_model: type[BaseModel]
    function: Callable


class ToolContext:
    def __init__(self, state):
        self.state = state

    def get_fleet_state(self, _: BaseModel | None = None):
        return self.state.current_snapshot or {"drones": {}}

    def get_drone_state(self, request: DroneQuery):
        return (self.state.current_snapshot or {}).get("drones", {}).get(request.drone_id)

    def get_drone_health(self, request: DroneQuery):
        drone = self.get_drone_state(request)
        if not drone:
            return {"error": "drone not found"}
        return {k: drone.get(k) for k in ("battery", "battery_temperature", "motor_temperature", "vibration", "gps_healthy")}

    def get_active_missions(self, _: BaseModel | None = None):
        return [m.model_dump(mode="json") for m in self.state.missions.values() if m.status == "ACTIVE"]

    def get_weather(self, _: BaseModel | None = None):
        wind = (self.state.current_snapshot or {}).get("wind", [0, 0])
        return {"wind_vector": wind, "wind_speed": float(np.linalg.norm(wind)), "source": "simulator"}

    def query_geofences(self, _: BaseModel | None = None):
        return (self.state.current_snapshot or {}).get("restricted_zones", [])

    def estimate_mission_energy(self, request: MissionEnergyRequest):
        fraction = request.distance * (0.0025 + request.package_weight * 0.0005) * (1 + request.wind_speed / 20)
        return {"estimated_battery_fraction": round(fraction, 4), "required_with_reserve": round(fraction + 0.15, 4)}

    def validate_mission(self, request: MissionValidationRequest):
        x, y = request.destination
        restricted = 8 <= x <= 12 and 8 <= y <= 12
        distance = abs(x - 1) + abs(y - 1)
        errors = []
        if request.package_weight > 5:
            errors.append("MAXIMUM_PAYLOAD")
        if restricted:
            errors.append("RESTRICTED_ZONE")
        if request.deadline_steps < distance + 5:
            errors.append("DEADLINE_INFEASIBLE")
        return {"valid": not errors, "errors": errors, "distance": distance}

    def simulate_mission(self, request: MissionValidationRequest):
        validation = self.validate_mission(request)
        energy = self.estimate_mission_energy(MissionEnergyRequest(distance=validation["distance"], package_weight=request.package_weight, wind_speed=0))
        return {**validation, "energy": energy, "feasible": validation["valid"] and energy["required_with_reserve"] <= 1}

    def compare_candidate_routes(self, request: RouteComparisonRequest):
        def length(route):
            return sum(float(np.linalg.norm(np.asarray(b) - np.asarray(a))) for a, b in zip(route, route[1:]))
        measured = [{"index": i, "distance": length(route)} for i, route in enumerate(request.routes)]
        return {"routes": measured, "recommended_index": min(measured, key=lambda x: x["distance"])["index"] if measured else None}

    def get_training_run(self, request: RunQuery):
        return self.state.training_runs.get(request.run_id, {"error": "training run not found"})

    def compare_models(self, _: BaseModel | None = None):
        return {"evaluations": list(self.state.evaluations.values()), "note": "Only completed measured evaluations are returned"}

    def get_incident_timeline(self, request: RunQuery):
        return self.state.incidents.get(request.run_id, {}).get("timeline", [])

    def create_incident(self, request: ProtectedAction):
        return self._approval("create_incident", request)

    def _approval(self, action: str, request: ProtectedAction):
        approval = ApprovalRequest(action=action, payload=request.model_dump())
        self.state.approvals[str(approval.id)] = approval
        return {"approval_required": True, "approval": approval.model_dump(mode="json")}

    def pause_mission(self, request: ProtectedAction):
        return self._approval("pause_mission", request)

    def request_return_to_base(self, request: ProtectedAction):
        return self._approval("return_to_base", request)

    def request_model_approval(self, request: ProtectedAction):
        return self._approval("promote_model", request)

    def definitions(self) -> dict[str, ToolDefinition]:
        empty = BaseModel
        return {
            "get_fleet_state": ToolDefinition("get_fleet_state", "read", empty, self.get_fleet_state),
            "get_drone_state": ToolDefinition("get_drone_state", "read", DroneQuery, self.get_drone_state),
            "get_drone_health": ToolDefinition("get_drone_health", "read", DroneQuery, self.get_drone_health),
            "get_active_missions": ToolDefinition("get_active_missions", "read", empty, self.get_active_missions),
            "get_weather": ToolDefinition("get_weather", "read", empty, self.get_weather),
            "query_geofences": ToolDefinition("query_geofences", "read", empty, self.query_geofences),
            "estimate_mission_energy": ToolDefinition("estimate_mission_energy", "simulation", MissionEnergyRequest, self.estimate_mission_energy),
            "validate_mission": ToolDefinition("validate_mission", "simulation", MissionValidationRequest, self.validate_mission),
            "simulate_mission": ToolDefinition("simulate_mission", "simulation", MissionValidationRequest, self.simulate_mission),
            "compare_candidate_routes": ToolDefinition("compare_candidate_routes", "simulation", RouteComparisonRequest, self.compare_candidate_routes),
            "get_training_run": ToolDefinition("get_training_run", "read", RunQuery, self.get_training_run),
            "compare_models": ToolDefinition("compare_models", "read", empty, self.compare_models),
            "get_incident_timeline": ToolDefinition("get_incident_timeline", "read", RunQuery, self.get_incident_timeline),
            "create_incident": ToolDefinition("create_incident", "protected", ProtectedAction, self.create_incident),
            "pause_mission": ToolDefinition("pause_mission", "protected", ProtectedAction, self.pause_mission),
            "request_return_to_base": ToolDefinition("request_return_to_base", "protected", ProtectedAction, self.request_return_to_base),
            "request_model_approval": ToolDefinition("request_model_approval", "protected", ProtectedAction, self.request_model_approval),
        }

