from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from packages.shared.models import Action, SafetyDecision


@dataclass(frozen=True)
class CollisionPrediction:
    current_separation: float
    time_to_closest_approach: float
    closest_separation: float
    horizontal_separation: float
    vertical_separation: float
    unsafe: bool


class SafetyEngine:
    """Deterministic safety shield applied after every policy decision."""

    ACTION_VELOCITY = {
        Action.MOVE_NORTH: np.array([0.0, 1.0, 0.0]),
        Action.MOVE_SOUTH: np.array([0.0, -1.0, 0.0]),
        Action.MOVE_EAST: np.array([1.0, 0.0, 0.0]),
        Action.MOVE_WEST: np.array([-1.0, 0.0, 0.0]),
        Action.CLIMB: np.array([0.0, 0.0, 1.0]),
        Action.DESCEND: np.array([0.0, 0.0, -1.0]),
        Action.HOVER: np.zeros(3),
    }

    def __init__(self, config_path: str | Path):
        with Path(config_path).open(encoding="utf-8") as handle:
            self.rules = yaml.safe_load(handle)

    def predict_collision(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        other_position: np.ndarray,
        other_velocity: np.ndarray,
    ) -> CollisionPrediction:
        relative_position = other_position - position
        relative_velocity = other_velocity - velocity
        speed_squared = float(np.dot(relative_velocity, relative_velocity))
        horizon = float(self.rules["collision_prediction_horizon_s"])
        if speed_squared < 1e-9:
            t_closest = 0.0
        else:
            t_closest = float(np.clip(-np.dot(relative_position, relative_velocity) / speed_squared, 0, horizon))
        closest_vector = relative_position + relative_velocity * t_closest
        horizontal = float(np.linalg.norm(closest_vector[:2]))
        vertical = float(abs(closest_vector[2]))
        unsafe = (
            horizontal < float(self.rules["minimum_horizontal_separation_m"])
            and vertical < float(self.rules["minimum_vertical_separation_m"])
        )
        return CollisionPrediction(
            current_separation=float(np.linalg.norm(relative_position)),
            time_to_closest_approach=t_closest,
            closest_separation=float(np.linalg.norm(closest_vector)),
            horizontal_separation=horizontal,
            vertical_separation=vertical,
            unsafe=unsafe,
        )

    def _in_restricted_zone(self, position: np.ndarray) -> bool:
        for zone in self.rules["restricted_zones"]:
            if (
                zone["min_x"] <= position[0] <= zone["max_x"]
                and zone["min_y"] <= position[1] <= zone["max_y"]
                and zone["min_altitude"] <= position[2] <= zone["max_altitude"]
            ):
                return True
        return False

    def evaluate(
        self,
        action: Action,
        drone: dict,
        peers: list[dict],
        station_capacity_available: bool = True,
    ) -> SafetyDecision:
        violations: list[str] = []
        replacement = Action.HOVER
        position = np.asarray(drone["position"], dtype=float)
        velocity = self.ACTION_VELOCITY.get(action, np.zeros(3))
        proposed = position + velocity
        collision_data = None

        if proposed[2] < self.rules["minimum_altitude_m"] and action == Action.DESCEND:
            violations.append("MINIMUM_ALTITUDE")
        if proposed[2] > self.rules["maximum_altitude_m"]:
            violations.append("MAXIMUM_ALTITUDE")
        if self._in_restricted_zone(proposed):
            violations.append("RESTRICTED_ZONE")
        if drone["battery"] <= self.rules["emergency_battery_level"] and action != Action.EMERGENCY_LAND:
            violations.append("EMERGENCY_BATTERY")
            replacement = Action.EMERGENCY_LAND
        elif drone["battery"] <= self.rules["minimum_battery_reserve"] and action not in (
            Action.RETURN_TO_BASE, Action.MOVE_TO_CHARGER, Action.CHARGE, Action.EMERGENCY_LAND
        ):
            violations.append("MINIMUM_BATTERY_RESERVE")
            replacement = Action.RETURN_TO_BASE
        if action == Action.MOVE_TO_CHARGER and not station_capacity_available:
            violations.append("CHARGER_FULL")

        for peer in peers:
            prediction = self.predict_collision(
                position,
                velocity,
                np.asarray(peer["position"], dtype=float),
                np.asarray(peer["velocity"], dtype=float),
            )
            if prediction.unsafe:
                violations.append("MINIMUM_SEPARATION")
                collision_data = {
                    "current_separation": prediction.current_separation,
                    "time_to_closest_approach": prediction.time_to_closest_approach,
                    "closest_separation": prediction.closest_separation,
                    "horizontal_separation": prediction.horizontal_separation,
                    "vertical_separation": prediction.vertical_separation,
                }
                if position[2] + 1 <= self.rules["maximum_altitude_m"] and position[2] <= peer["position"][2]:
                    replacement = Action.CLIMB
                break

        if not violations:
            return SafetyDecision(allowed=True, original_action=action.name)
        return SafetyDecision(
            allowed=False,
            original_action=action.name,
            replacement_action=replacement.name,
            violated_rules=sorted(set(violations)),
            reason="; ".join(sorted(set(violations))).replace("_", " ").title(),
            rule_version=str(self.rules["version"]),
            collision=collision_data,
        )

