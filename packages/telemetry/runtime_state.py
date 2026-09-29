from __future__ import annotations

import json
import os


class RuntimeStateBackend:
    """Redis state and fanout with explicit in-memory development fallback."""

    def __init__(self):
        self.redis = None
        self.memory: dict[str, dict] = {}
        self.mode = "in-memory-fallback"

    async def connect(self):
        try:
            from redis.asyncio import from_url
            client = from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), socket_connect_timeout=0.3)
            await client.ping()
            self.redis = client
            self.mode = "redis"
        except Exception:
            self.redis = None
            self.mode = "in-memory-fallback"
        return self.mode

    async def set(self, key: str, value: dict, ttl: int = 3600):
        if self.redis:
            await self.redis.set(key, json.dumps(value, default=str), ex=ttl)
        else:
            self.memory[key] = value

    async def get(self, key: str):
        if self.redis:
            value = await self.redis.get(key)
            return json.loads(value) if value else None
        return self.memory.get(key)
