import numpy as np

from packages.shared.models import Action
from packages.simulation.environment import DroneDeliveryEnv


def test_reset_is_seeded_and_observation_is_local_and_normalized():
    a = DroneDeliveryEnv(seed=7)
    obs_a, _ = a.reset(seed=7)
    b = DroneDeliveryEnv(seed=7)
    obs_b, _ = b.reset(seed=7)
    assert obs_a["D1"]["observation"].shape == (32,)
    assert np.allclose(obs_a["D1"]["observation"], obs_b["D1"]["observation"])
    assert np.max(np.abs(obs_a["D1"]["observation"])) <= 1
    assert obs_a["D1"]["action_mask"].shape == (len(Action),)


def test_pickup_delivery_battery_and_reward_breakdown():
    env = DroneDeliveryEnv(num_drones=1, seed=3)
    env.reset(seed=3)
    _, rewards, *_ = env.step({"D1": Action.PICK_UP})
    assert env.drones["D1"]["package_id"] == "P1"
    assert rewards["D1"] > 0
    battery = env.drones["D1"]["battery"]
    env.step({"D1": Action.MOVE_NORTH})
    assert env.drones["D1"]["battery"] < battery
    package = env.packages["P1"]
    env.drones["D1"]["position"] = package["destination"].copy()
    _, rewards, *_ = env.step({"D1": Action.DELIVER})
    assert package["delivered"]
    assert rewards["D1"] >= 20
    assert env.decision_events[-1].reward_breakdown["delivery"] == 20


def test_charging_and_deadline_handling():
    env = DroneDeliveryEnv(num_drones=1, seed=4)
    env.reset(seed=4)
    env.drones["D1"]["position"] = env.chargers[0].copy()
    env.drones["D1"]["battery"] = 0.5
    env.step({"D1": Action.CHARGE})
    assert env.drones["D1"]["battery"] > 0.5
    env.packages["P1"]["picked_up"] = True
    env.drones["D1"]["package_id"] = "P1"
    env.drones["D1"]["position"] = env.packages["P1"]["destination"].copy()
    env.step_count = env.packages["P1"]["deadline"] + 1
    _, _, _, _, infos = env.step({"D1": Action.DELIVER})
    assert "on_time" not in infos["D1"]["decision_event"]["reward_breakdown"]


def test_invalid_delivery_is_masked_and_replaced():
    env = DroneDeliveryEnv(num_drones=1)
    env.reset()
    assert not env.action_mask("D1")[Action.DELIVER]
    env.step({"D1": Action.DELIVER})
    event = env.decision_events[-1]
    assert not event.safety_result.allowed
    assert "ACTION_MASK" in event.safety_result.violated_rules
    assert event.executed_action == "HOVER"


def test_environment_records_collision_safety_override():
    env = DroneDeliveryEnv(num_drones=2, seed=8)
    env.reset(seed=8)
    env.drones["D1"]["position"] = np.array([2.0, 2.0, 2.0])
    env.drones["D2"]["position"] = np.array([4.0, 2.0, 2.0])
    env.drones["D2"]["velocity"] = np.array([-1.0, 0.0, 0.0])
    env.step({"D1": Action.MOVE_EAST, "D2": Action.MOVE_WEST})
    event = next(e for e in env.decision_events if e.drone_id == "D1")
    assert "MINIMUM_SEPARATION" in event.safety_result.violated_rules
    assert event.policy_action != event.executed_action


def test_drones_acknowledge_conflict_and_lower_priority_drone_yields():
    env = DroneDeliveryEnv(num_drones=2, seed=8, message_loss=0, message_delay_steps=0)
    env.reset(seed=8)
    env.drones["D1"]["position"] = np.array([2.0, 2.0, 2.0])
    env.drones["D2"]["position"] = np.array([4.0, 2.0, 2.0])
    env.step({"D1": Action.MOVE_EAST, "D2": Action.MOVE_WEST})
    assert env.conflict_events
    assert all(event["acknowledged"] for event in env.conflict_events)
    assert any(
        "COMMUNICATION_CONFLICT" in event.safety_result.violated_rules
        for event in env.decision_events
    )
    assert any(
        message["conflict_acknowledgement"] for message in env.communication_events
    )

