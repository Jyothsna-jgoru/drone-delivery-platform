from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class EventStore:
    """Durable JSONL event log plus in-process WebSocket fanout fallback."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.subscribers: set[asyncio.Queue] = set()

    async def append(self, stream: str, event: dict[str, Any]) -> None:
        clean = json.loads(json.dumps(event, default=str))
        self.events[stream].append(clean)
        with (self.data_dir / f"{stream}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean, separators=(",", ":")) + "\n")
        envelope = {"type": stream, "data": clean}
        for queue in list(self.subscribers):
            if not queue.full():
                queue.put_nowait(envelope)

    def list(self, stream: str) -> list[dict[str, Any]]:
        return list(self.events[stream])

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)

