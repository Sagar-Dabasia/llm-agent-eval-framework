from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional
import requests

from .config import Settings

@dataclass
class LLMResult:
    text: str
    latency_s: float

class OllamaLLMClient:
    def __init__(self, settings: Settings):
        self.host = settings.ollama_host.rstrip("/")
        self.model = settings.ollama_model
        self.temperature = settings.temperature

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResult:
        system_msg = system or "You are a precise assistant. Follow instructions carefully."
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": self.temperature},
            "stream": False,
        }

        t0 = time.time()
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")

        data = r.json()
        text = data.get("message", {}).get("content", "")
        return LLMResult(text=text.strip(), latency_s=round(time.time() - t0, 4))
