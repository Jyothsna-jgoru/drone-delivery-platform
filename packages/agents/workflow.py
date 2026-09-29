from __future__ import annotations

import re
from typing import Any, TypedDict

from packages.agents.providers import select_provider
from packages.agents.tools import MissionEnergyRequest, ToolContext


class WorkflowState(TypedDict, total=False):
    request: str
    category: str
    evidence: dict[str, Any]
    response: dict[str, Any]
    errors: list[str]


class OperationalWorkflow:
    """LangGraph-compatible operational workflow with deterministic execution."""

    ROUTES = {
        "mission_planning": "Mission Planning Agent",
        "fleet_operations": "Fleet Operations Agent",
        "maintenance": "Maintenance Agent",
        "incident_analysis": "Incident Analysis Agent",
        "model_evaluation": "Model Evaluation Agent",
    }

    def __init__(self, app_state):
        self.provider = select_provider()
        self.tools = ToolContext(app_state)
        self.graph = None
        try:
            from langgraph.graph import END, StateGraph
            builder = StateGraph(WorkflowState)
            builder.add_node("supervisor", self._classify)
            builder.add_node("specialist", self._operate)
            builder.set_entry_point("supervisor")
            builder.add_edge("supervisor", "specialist")
            builder.add_edge("specialist", END)
            self.graph = builder.compile()
        except ImportError:
            self.graph = None

    def invoke(self, operator_request: str) -> dict:
        initial: WorkflowState = {"request": operator_request, "errors": []}
        final = self.graph.invoke(initial) if self.graph else self._operate(self._classify(initial))
        return final["response"]

    def _classify(self, state: WorkflowState) -> WorkflowState:
        operator_request = state["request"]
        classification = self.provider.structured(
            operator_request,
            {"type": "object", "properties": {"category": {"type": "string"}, "reason": {"type": "string"}}},
        )
        category = classification.get("category", "mission_planning")
        return {**state, "category": category, "evidence": {"classification": classification}}

    def _operate(self, state: WorkflowState) -> WorkflowState:
        operator_request = state["request"]
        category = state["category"]
        classification = state["evidence"]["classification"]
        evidence: dict[str, Any] = {}
        recommendations: list[str] = []
        if category == "mission_planning":
            numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", operator_request)]
            weight = next((n for n in numbers if n <= 10), 1.0)
            weather = self.tools.get_weather()
            energy = self.tools.estimate_mission_energy(MissionEnergyRequest(distance=25, package_weight=weight, wind_speed=weather["wind_speed"]))
            evidence = {"weather": weather, "geofences": self.tools.query_geofences(), "energy": energy}
            recommendations.append("Mission is feasible" if energy["required_with_reserve"] < 1 else "Mission exceeds energy budget")
        elif category == "fleet_operations":
            evidence = self.tools.get_fleet_state()
            low = [key for key, value in evidence.get("drones", {}).items() if value.get("battery", 1) < 0.2]
            recommendations.append(f"Review low-battery drones: {', '.join(low)}" if low else "No low-battery drones detected")
        elif category == "maintenance":
            evidence = {key: self.tools.get_drone_health(type("Q", (), {"drone_id": key})()) for key in (self.tools.get_fleet_state().get("drones") or {})}
            rows = [[v.get("battery", 1), v.get("motor_temperature", 0) or 0, v.get("vibration", 0) or 0] for v in evidence.values() if v and "error" not in v]
            if len(rows) >= 3:
                try:
                    from sklearn.ensemble import IsolationForest
                    scores = IsolationForest(random_state=42, contamination="auto").fit(rows).decision_function(rows)
                    for (drone_id, values), score in zip(evidence.items(), scores):
                        values["anomaly_score"] = float(score)
                        values["anomalous"] = bool(score < 0)
                except ImportError:
                    pass
            recommendations.append("Inspect any drone above deterministic temperature or vibration thresholds")
        elif category == "incident_analysis":
            evidence = {"decision_events": app_events(self.tools.state, "decision")[-25:], "safety_events": app_events(self.tools.state, "safety")[-25:]}
            recommendations.append("Reproduce using the recorded simulation seed and event sequence")
        else:
            evidence = {"training_runs": list(self.tools.state.training_runs.values())}
            recommendations.append("Promotion requires measured evaluation and human approval")
        response = {
            "supervisor": "Supervisor Agent",
            "routed_to": self.ROUTES.get(category, "Mission Planning Agent"),
            "provider": self.provider.name,
            "classification": classification,
            "evidence": evidence,
            "recommendations": recommendations,
            "protected_action_executed": False,
        }
        return {**state, "evidence": evidence, "response": response}


def app_events(state, stream: str):
    return state.event_store.list(stream)
