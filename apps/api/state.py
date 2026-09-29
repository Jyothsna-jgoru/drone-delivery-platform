from __future__ import annotations

import asyncio
from pathlib import Path

from packages.telemetry.store import EventStore
from packages.telemetry.runtime_state import RuntimeStateBackend


ROOT = Path(__file__).resolve().parents[2]


class AppState:
    def __init__(self):
        self.missions = {}
        self.simulations = {}
        self.simulation_tasks: dict[str, asyncio.Task] = {}
        self.simulation_controls = {}
        self.current_snapshot = None
        self.approvals = {}
        self.incidents = {}
        self.training_runs = {}
        self.evaluations = {}
        self.comparison_runs = {}
        self.event_store = EventStore(ROOT / "data" / "events")
        self.runtime = RuntimeStateBackend()


state = AppState()

