from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from packages.shared.models import DeliveredMessage, DroneBroadcast, Priority


@dataclass
class QueuedMessage:
    deliver_step: int
    message: DeliveredMessage


class CommunicationNetwork:
    """Seeded local topic-like message channel with latency and loss."""

    def __init__(self, communication_range: float, loss_probability: float, max_delay_steps: int, seed: int):
        self.range = communication_range
        self.loss_probability = loss_probability
        self.max_delay_steps = max_delay_steps
        self.rng = np.random.default_rng(seed)
        self.queue: list[QueuedMessage] = []

    def broadcast(self, payload: DroneBroadcast, recipients: dict[str, tuple[float, float, float]], step: int) -> None:
        origin = np.asarray(payload.position)
        for receiver, location in recipients.items():
            if receiver == payload.drone_id or np.linalg.norm(np.asarray(location) - origin) > self.range:
                continue
            delay = int(self.rng.integers(0, self.max_delay_steps + 1))
            dropped = bool(self.rng.random() < self.loss_probability)
            message = DeliveredMessage(
                sender=payload.drone_id,
                receiver=receiver,
                payload=payload,
                latency_ms=delay * 100,
                dropped=dropped,
            )
            if dropped:
                self.queue.append(QueuedMessage(step, message))
            else:
                self.queue.append(QueuedMessage(step + delay, message))

    def receive(self, receiver: str, step: int) -> list[DeliveredMessage]:
        ready = [q for q in self.queue if q.deliver_step <= step and q.message.receiver == receiver]
        self.queue = [q for q in self.queue if q not in ready]
        return [q.message for q in ready]


def resolve_conflict(a: DroneBroadcast, b: DroneBroadcast) -> tuple[str, str]:
    """Return winner and deterministic reason using priority, deadline proxy, battery, then id."""
    rank = {Priority.NORMAL: 0, Priority.HIGH: 1, Priority.CRITICAL: 2}
    if rank[a.package_priority] != rank[b.package_priority]:
        winner = a if rank[a.package_priority] > rank[b.package_priority] else b
        return winner.drone_id, "package_priority"
    if abs(a.battery - b.battery) > 0.05:
        winner = a if a.battery < b.battery else b
        return winner.drone_id, "lower_battery_gets_shorter_maneuver"
    return min(a.drone_id, b.drone_id), "stable_drone_id_tiebreak"
