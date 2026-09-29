from pathlib import Path

import numpy as np

from packages.safety.engine import SafetyEngine
from packages.shared.models import Action


ROOT = Path(__file__).resolve().parents[2]


def engine():
    return SafetyEngine(ROOT / "configs/safety_rules.yaml")


def test_collision_prediction_uses_closest_approach():
    prediction = engine().predict_collision(
        np.array([0.0, 0.0, 2.0]), np.array([1.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 2.0]), np.array([-1.0, 0.0, 0.0]),
    )
    assert prediction.unsafe
    assert prediction.time_to_closest_approach == 2.0
    assert prediction.closest_separation == 0.0


def test_safety_replaces_collision_course_and_restricted_zone():
    shield = engine()
    drone = {"position": np.array([7.0, 9.0, 2.0]), "battery": 1.0}
    restricted = shield.evaluate(Action.MOVE_EAST, drone, [])
    assert not restricted.allowed
    assert "RESTRICTED_ZONE" in restricted.violated_rules
    drone = {"position": np.array([0.0, 0.0, 2.0]), "battery": 1.0}
    peer = {"position": np.array([2.0, 0.0, 2.0]), "velocity": np.array([-1.0, 0.0, 0.0])}
    collision = shield.evaluate(Action.MOVE_EAST, drone, [peer])
    assert not collision.allowed
    assert "MINIMUM_SEPARATION" in collision.violated_rules
    assert collision.replacement_action in {"CLIMB", "HOVER"}


def test_low_battery_forces_return():
    decision = engine().evaluate(Action.MOVE_NORTH, {"position": np.array([1.0, 1.0, 2.0]), "battery": 0.1}, [])
    assert decision.replacement_action == "RETURN_TO_BASE"

