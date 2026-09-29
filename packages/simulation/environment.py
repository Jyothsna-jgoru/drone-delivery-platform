from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from packages.communication.network import CommunicationNetwork, resolve_conflict
from packages.safety.engine import SafetyEngine
from packages.shared.models import (
    Action,
    DecisionEvent,
    DroneBroadcast,
    MissionStatus,
    Priority,
    SafetyDecision,
)


ROOT = Path(__file__).resolve().parents[2]


class DroneDeliveryEnv(ParallelEnv):
    metadata = {"name": "drone_delivery_v0", "is_parallelizable": True}
    OBS_SIZE = 32
    DELIVERY_RADIUS = 0.75

    def __init__(
        self,
        num_drones: int = 3,
        grid_size: int = 20,
        max_steps: int = 150,
        seed: int = 42,
        communication_range: float = 8.0,
        message_loss: float = 0.05,
        message_delay_steps: int = 2,
    ):
        self.num_drones = num_drones
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.initial_seed = seed
        self.possible_agents = [f"D{i + 1}" for i in range(num_drones)]
        self.agents = self.possible_agents[:]
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(-1.0, 1.0, (self.OBS_SIZE,), dtype=np.float32),
                    "action_mask": spaces.MultiBinary(len(Action)),
                }
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {agent: spaces.Discrete(len(Action)) for agent in self.possible_agents}
        self.safety = SafetyEngine(ROOT / "configs" / "safety_rules.yaml")
        self.network_options = (communication_range, message_loss, message_delay_steps)
        self.simulation_id: UUID = uuid4()
        self.episode = 0
        self.step_count = 0
        self.decision_events: list[DecisionEvent] = []
        self.communication_events: list[dict] = []
        self.safety_events: list[dict] = []
        self.conflict_events: list[dict] = []
        self.reset(seed=seed)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        seed = self.initial_seed if seed is None else seed
        self.rng = np.random.default_rng(seed)
        self.episode += 1
        self.step_count = 0
        self.agents = self.possible_agents[:]
        self.simulation_id = uuid4()
        self.decision_events = []
        self.communication_events = []
        self.safety_events = []
        self.conflict_events = []
        self.wind = self.rng.uniform(-0.25, 0.25, size=2)
        self.depot = np.array([1.0, 1.0, 1.0])
        self.chargers = [np.array([2.0, 2.0, 1.0]), np.array([18.0, 2.0, 1.0])]
        self.charger_capacity = 1
        self.obstacles = [
            {"x": 6.0, "y": 5.0, "width": 2.0, "height": 4.0},
            {"x": 13.0, "y": 11.0, "width": 2.0, "height": 3.0},
        ]
        self.packages = {
            f"P{i + 1}": {
                "id": f"P{i + 1}",
                "pickup": np.array([1.0, 1.0 + i * 1.8, 1.0]),
                "destination": np.array([17.0 - i * 2, 16.0 - i, 2.0 + i]),
                "weight": 1.0 + i * 0.5,
                "priority": Priority.CRITICAL if i == 0 else Priority.NORMAL,
                "deadline": 80 + i * 15,
                "assigned_to": self.possible_agents[i % self.num_drones],
                "active": True,
                "picked_up": False,
                "delivered": False,
                "delivered_step": None,
            }
            for i in range(self.num_drones)
        }
        self.drones = {}
        for i, agent in enumerate(self.possible_agents):
            self.drones[agent] = {
                "id": agent,
                "position": np.array([1.0, 1.0 + i * 1.8, 1.0]),
                "velocity": np.zeros(3),
                "battery": 1.0,
                "battery_temperature": 24.0,
                "motor_temperature": 28.0,
                "vibration": 0.05,
                "gps_healthy": True,
                "payload_weight": 0.0,
                "package_id": None,
                "assigned_package": f"P{i + 1}",
                "previous_action": Action.HOVER,
                "distance_travelled": 0.0,
                "energy_consumed": 0.0,
                "status": "IDLE",
                "communication_healthy": True,
                "path": [],
            }
        self.network = CommunicationNetwork(*self.network_options, seed=seed)
        observations = {agent: self._observe(agent) for agent in self.agents}
        return observations, {agent: {"seed": seed} for agent in self.agents}

    def _assigned(self, agent: str) -> dict:
        return self.packages[self.drones[agent]["assigned_package"]]

    def _distance_to_destination(self, agent: str) -> float:
        drone, package = self.drones[agent], self._assigned(agent)
        target = package["destination"] if package["picked_up"] else package["pickup"]
        return float(np.linalg.norm(target - drone["position"]))

    def _nearby(self, agent: str, distance: float = 8.0) -> list[dict]:
        position = self.drones[agent]["position"]
        return [
            peer
            for peer_id, peer in self.drones.items()
            if peer_id != agent and np.linalg.norm(peer["position"] - position) <= distance
        ]

    def action_mask(self, agent: str) -> np.ndarray:
        drone, package = self.drones[agent], self._assigned(agent)
        mask = np.ones(len(Action), dtype=np.int8)
        pos = drone["position"]
        mask[Action.MOVE_NORTH] = pos[1] < self.grid_size
        mask[Action.MOVE_SOUTH] = pos[1] > 0
        mask[Action.MOVE_EAST] = pos[0] < self.grid_size
        mask[Action.MOVE_WEST] = pos[0] > 0
        mask[Action.CLIMB] = pos[2] < self.safety.rules["maximum_altitude_m"]
        mask[Action.DESCEND] = pos[2] > self.safety.rules["minimum_altitude_m"]
        for action in (Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST):
            proposed = pos + self.safety.ACTION_VELOCITY[action]
            if any(o["x"] <= proposed[0] <= o["x"] + o["width"] and o["y"] <= proposed[1] <= o["y"] + o["height"] for o in self.obstacles):
                mask[action] = 0
        at_pickup = np.linalg.norm(pos - package["pickup"]) <= 1.1
        mask[Action.PICK_UP] = int(at_pickup and not package["picked_up"] and drone["package_id"] is None)
        at_destination = np.linalg.norm(pos - package["destination"]) <= self.DELIVERY_RADIUS
        mask[Action.DELIVER] = int(at_destination and drone["package_id"] == package["id"])
        at_charger = any(np.linalg.norm(pos - charger) <= 1.1 for charger in self.chargers)
        mask[Action.CHARGE] = int(at_charger and drone["battery"] < 0.999)
        return mask

    def _observe(self, agent: str) -> dict[str, np.ndarray]:
        drone, package = self.drones[agent], self._assigned(agent)
        pos = drone["position"] / np.array([self.grid_size, self.grid_size, 12.0])
        vel = drone["velocity"]
        target = package["destination"] if package["picked_up"] else package["pickup"]
        delta = (target - drone["position"]) / self.grid_size
        nearby = self._nearby(agent)
        closest = min((np.linalg.norm(p["position"] - drone["position"]) for p in nearby), default=self.grid_size)
        nearest_charger = min(np.linalg.norm(c - drone["position"]) for c in self.chargers)
        deadline_remaining = max(0, package["deadline"] - self.step_count) / max(1, package["deadline"])
        obstacle_distances = []
        for obs in self.obstacles[:2]:
            center = np.array([obs["x"] + obs["width"] / 2, obs["y"] + obs["height"] / 2])
            obstacle_distances.append(min(1.0, np.linalg.norm(center - drone["position"][:2]) / self.grid_size))
        vector = np.array(
            [
                *pos,
                *vel,
                drone["battery"] * 2 - 1,
                (drone["battery_temperature"] - 50) / 50,
                drone["payload_weight"] / 5,
                int(package["priority"] == Priority.CRITICAL),
                deadline_remaining * 2 - 1,
                *delta,
                *obstacle_distances,
                closest / self.grid_size,
                len(nearby) / max(1, self.num_drones - 1),
                nearest_charger / self.grid_size,
                *self.wind,
                float(drone["communication_healthy"]),
                int(drone["previous_action"]) / (len(Action) - 1),
                float(package["picked_up"]),
                float(package["delivered"]),
                self.step_count / self.max_steps,
            ],
            dtype=np.float32,
        )
        vector = np.pad(vector, (0, max(0, self.OBS_SIZE - vector.size)))[: self.OBS_SIZE]
        return {"observation": np.clip(vector, -1, 1), "action_mask": self.action_mask(agent)}

    def state(self) -> np.ndarray:
        values = []
        for agent in self.possible_agents:
            values.extend(self._observe(agent)["observation"])
        return np.asarray(values, dtype=np.float32)

    def _broadcast(self, agent: str, intended: Action) -> DroneBroadcast:
        drone, package = self.drones[agent], self._assigned(agent)
        waypoint = drone["position"] + self.safety.ACTION_VELOCITY.get(intended, np.zeros(3))
        payload = DroneBroadcast(
            drone_id=agent,
            position=tuple(drone["position"]),
            velocity=tuple(drone["velocity"]),
            next_waypoint=tuple(waypoint),
            planned_altitude=float(waypoint[2]),
            battery=drone["battery"],
            package_priority=package["priority"],
            mission_status=MissionStatus.ACTIVE,
        )
        positions = {key: tuple(value["position"]) for key, value in self.drones.items()}
        self.network.broadcast(payload, positions, self.step_count)
        return payload

    def _execute(self, agent: str, action: Action, breakdown: dict[str, float]) -> None:
        drone, package = self.drones[agent], self._assigned(agent)
        old = drone["position"].copy()
        velocity = self.safety.ACTION_VELOCITY.get(action, np.zeros(3)).copy()
        if action in (Action.MOVE_TO_CHARGER, Action.RETURN_TO_BASE):
            target = min(self.chargers, key=lambda c: np.linalg.norm(c - old)) if action == Action.MOVE_TO_CHARGER else self.depot
            delta = target - old
            velocity = delta / max(1.0, np.linalg.norm(delta))
        if action == Action.EMERGENCY_LAND:
            velocity = np.array([0.0, 0.0, -min(1.0, old[2])])
            drone["status"] = "EMERGENCY_LANDING"
        if action in self.safety.ACTION_VELOCITY or action in (Action.MOVE_TO_CHARGER, Action.RETURN_TO_BASE, Action.EMERGENCY_LAND):
            drone["position"] = np.clip(old + velocity, [0, 0, 0], [self.grid_size, self.grid_size, 12])
            drone["velocity"] = velocity
            travelled = float(np.linalg.norm(drone["position"] - old))
            cost = travelled * (0.0025 + drone["payload_weight"] * 0.0005) + (0.0005 if action == Action.HOVER else 0)
            drone["distance_travelled"] += travelled
            drone["battery"] = max(0.0, drone["battery"] - cost)
            drone["energy_consumed"] += cost
            breakdown["energy"] = -cost
        else:
            drone["velocity"] = np.zeros(3)
        if action == Action.PICK_UP and self.action_mask(agent)[Action.PICK_UP]:
            package["picked_up"] = True
            drone["package_id"] = package["id"]
            drone["payload_weight"] = package["weight"]
            drone["status"] = "DELIVERING"
            breakdown["pickup"] = 1.0
        if action == Action.DELIVER and self.action_mask(agent)[Action.DELIVER]:
            package["delivered"] = True
            package["delivered_step"] = self.step_count
            drone["package_id"] = None
            drone["payload_weight"] = 0.0
            drone["status"] = "DELIVERED"
            breakdown["delivery"] = 20.0
            if self.step_count <= package["deadline"]:
                breakdown["on_time"] = 5.0
            if package["priority"] == Priority.CRITICAL:
                breakdown["priority"] = 5.0
        if action == Action.CHARGE and self.action_mask(agent)[Action.CHARGE]:
            before = drone["battery"]
            drone["battery"] = min(1.0, before + 0.08)
            breakdown["charge"] = 0.2
        drone["path"].append(tuple(float(x) for x in drone["position"]))
        drone["previous_action"] = action
        drone["battery_temperature"] = 24 + 20 * (1 - drone["battery"])

    def step(self, actions: dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        self.step_count += 1
        broadcasts = {
            agent: self._broadcast(agent, Action(actions.get(agent, Action.HOVER)))
            for agent in self.agents
        }

        rewards, infos = {}, {}
        for agent in self.agents:
            proposed = Action(actions.get(agent, Action.HOVER))
            mask = self.action_mask(agent)
            invalid = not bool(mask[proposed])
            peers = self._nearby(agent)
            messages = self.network.receive(agent, self.step_count)
            self.drones[agent]["communication_healthy"] = not any(m.dropped for m in messages)
            communication_replacement = None
            conflict_reason = ""
            own_broadcast = broadcasts[agent]
            own_waypoint = np.asarray(own_broadcast.next_waypoint)
            for message in messages:
                if message.dropped:
                    continue
                other_waypoint = np.asarray(message.payload.next_waypoint)
                horizontal = float(np.linalg.norm(own_waypoint[:2] - other_waypoint[:2]))
                vertical = float(abs(own_waypoint[2] - other_waypoint[2]))
                if (
                    horizontal < self.safety.rules["minimum_horizontal_separation_m"]
                    and vertical < self.safety.rules["minimum_vertical_separation_m"]
                ):
                    winner, reason = resolve_conflict(own_broadcast, message.payload)
                    message.conflict_acknowledgement = True
                    conflict = {
                        "step": self.step_count,
                        "drone_id": agent,
                        "peer_id": message.sender,
                        "winner": winner,
                        "reason": reason,
                        "horizontal_waypoint_separation": horizontal,
                        "vertical_waypoint_separation": vertical,
                        "acknowledged": True,
                    }
                    self.conflict_events.append(conflict)
                    if winner != agent:
                        candidate = Action.CLIMB if mask[Action.CLIMB] else Action.HOVER
                        candidate_check = self.safety.evaluate(candidate, self.drones[agent], peers)
                        communication_replacement = candidate if candidate_check.allowed else Action.HOVER
                        conflict_reason = f"Yielding to {winner}: {reason}"
                    break
            safety = self.safety.evaluate(proposed, self.drones[agent], peers)
            if communication_replacement is not None:
                safety = SafetyDecision(
                    allowed=False,
                    original_action=proposed.name,
                    replacement_action=communication_replacement.name,
                    violated_rules=["COMMUNICATION_CONFLICT"],
                    reason=conflict_reason,
                    rule_version=str(self.safety.rules["version"]),
                )
            if invalid:
                safety.allowed = False
                safety.replacement_action = Action.HOVER.name
                safety.violated_rules.append("ACTION_MASK")
                safety.reason = "Action is not available in current state"
            executed = proposed if safety.allowed else Action[safety.replacement_action or "HOVER"]
            serialized_messages = [m.model_dump(mode="json") for m in messages]
            self.communication_events.extend(serialized_messages)
            breakdown = {"step": -0.01}
            if not safety.allowed:
                breakdown["unsafe_proposal"] = -2.0
                if "MINIMUM_SEPARATION" in safety.violated_rules:
                    breakdown["collision_avoidance"] = 1.0
                self.safety_events.append(
                    {"step": self.step_count, "drone_id": agent, **safety.model_dump(mode="json")}
                )
            self._execute(agent, executed, breakdown)
            reward = float(sum(breakdown.values()))
            rewards[agent] = reward
            package = self._assigned(agent)
            event = DecisionEvent(
                simulation_id=self.simulation_id,
                episode=self.episode,
                step=self.step_count,
                drone_id=agent,
                position=tuple(self.drones[agent]["position"]),
                observation_summary={
                    "battery": round(self.drones[agent]["battery"], 4),
                    "destination_distance": round(self._distance_to_destination(agent), 3),
                    "nearest_drone_distance": round(min((np.linalg.norm(p["position"] - self.drones[agent]["position"]) for p in peers), default=99.0), 3),
                    "communication_healthy": self.drones[agent]["communication_healthy"],
                },
                received_messages=serialized_messages,
                available_actions=[Action(i).name for i, allowed in enumerate(mask) if allowed],
                policy_q_values={},
                policy_action=proposed.name,
                safety_result=safety,
                executed_action=executed.name,
                reward=reward,
                reward_breakdown=breakdown,
                model_version="rule-based-v1",
            )
            self.decision_events.append(event)
            infos[agent] = {"decision_event": event.model_dump(mode="json"), "package": deepcopy(package)}

        done = all(package["delivered"] for package in self.packages.values() if package.get("active", True))
        truncated = self.step_count >= self.max_steps
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        observations = {agent: self._observe(agent) for agent in self.agents}
        if done or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def snapshot(self) -> dict:
        def serial(value):
            if isinstance(value, np.ndarray):
                return [float(x) for x in value]
            if isinstance(value, Priority):
                return value.value
            return value

        drones = {key: {k: serial(v) for k, v in drone.items()} for key, drone in self.drones.items()}
        packages = {key: {k: serial(v) for k, v in package.items()} for key, package in self.packages.items()}
        return {
            "simulation_id": str(self.simulation_id),
            "step": self.step_count,
            "grid_size": self.grid_size,
            "drones": drones,
            "packages": packages,
            "depot": serial(self.depot),
            "chargers": [serial(c) for c in self.chargers],
            "obstacles": self.obstacles,
            "restricted_zones": self.safety.rules["restricted_zones"],
            "conflicts": self.conflict_events[-20:],
            "wind": serial(self.wind),
        }


def make_env(**kwargs) -> DroneDeliveryEnv:
    return DroneDeliveryEnv(**kwargs)

