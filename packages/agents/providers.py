from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from urllib.request import Request, urlopen


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def structured(self, prompt: str, schema: dict) -> dict:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    def available(self) -> bool:
        try:
            with urlopen(f"{self.base_url}/api/tags", timeout=0.3) as response:
                return response.status == 200
        except Exception:
            return False

    def structured(self, prompt: str, schema: dict) -> dict:
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps({"model": self.model, "prompt": prompt, "format": schema, "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(json.loads(response.read())["response"])


class DeterministicProvider(LLMProvider):
    name = "deterministic-fallback"

    def structured(self, prompt: str, schema: dict) -> dict:
        text = prompt.lower()
        category = "mission_planning"
        if any(word in text for word in ("maintenance", "battery health", "vibration", "temperature")):
            category = "maintenance"
        elif any(word in text for word in ("incident", "failure", "why did")):
            category = "incident_analysis"
        elif any(word in text for word in ("model", "qmix", "dqn", "evaluation")):
            category = "model_evaluation"
        elif any(word in text for word in ("fleet", "delay", "charger", "low battery")):
            category = "fleet_operations"
        return {"category": category, "reason": "matched deterministic operational vocabulary"}


def select_provider() -> LLMProvider:
    ollama = OllamaProvider()
    return ollama if ollama.available() else DeterministicProvider()

