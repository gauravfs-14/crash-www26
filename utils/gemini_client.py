import os
import time
from typing import List, Optional, Any, Type

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel


DEFAULT_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
]

RATE_LIMIT_ERRORS = (
    "429", "rate limit", "quota", "Resource has been exhausted", "exceeded"
)


class RotatingGeminiClient:
    """Rotate across Gemini models and API keys on rate/quota limits."""

    def __init__(self, models: Optional[List[str]] = None, temperature: float = 0.2, api_keys: Optional[List[str]] = None):
        self.models = models or DEFAULT_MODELS
        self.temperature = temperature
        # Discover API keys: GOOGLE_API_KEY, GOOGLE_API_KEY_1.._10
        if api_keys is not None:
            discovered = api_keys
        else:
            keys: List[str] = []
            base = os.environ.get("GOOGLE_API_KEY")
            if base:
                keys.append(base)
            for i in range(1, 11):
                k = os.environ.get(f"GOOGLE_API_KEY_{i}")
                if k:
                    keys.append(k)
            # Deduplicate while preserving order
            seen = set()
            discovered = []
            for k in keys:
                if k and k not in seen:
                    seen.add(k)
                    discovered.append(k)
        if not discovered:
            raise RuntimeError("No Google API keys found. Set GOOGLE_API_KEY or GOOGLE_API_KEY_1..N.")
        self.api_keys = discovered
        self.key_index = 0
        self.model_index = 0
        self._client = self._build_client(self.models[self.model_index], self.api_keys[self.key_index])

    def _build_client(self, model: str, api_key: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(model=model, temperature=self.temperature, google_api_key=api_key)

    def _rotate_model(self):
        self.model_index = (self.model_index + 1) % len(self.models)
        self._client = self._build_client(self.models[self.model_index], self.api_keys[self.key_index])

    def _rotate_key(self):
        self.key_index = (self.key_index + 1) % len(self.api_keys)
        # After switching key, reset model to first for fairness
        self.model_index = 0
        self._client = self._build_client(self.models[self.model_index], self.api_keys[self.key_index])

    def with_structured_output(self, schema: Type[BaseModel]):
        return self._client.with_structured_output(schema)

    def run_structured(self, prompt, schema: Type[BaseModel], values: dict, retries: int = 6, backoff: float = 2.0) -> Any:
        """Run prompt → structured schema with rotation across models and keys on rate limits.
        Rebuilds the chain on each attempt to apply current client.
        """
        last_err: Optional[Exception] = None
        rotations_on_key = 0
        for attempt in range(retries):
            try:
                structured = self._client.with_structured_output(schema)
                chain = prompt | structured
                return chain.invoke(values)
            except Exception as e:
                msg = str(e).lower()
                last_err = e
                if any(tok in msg for tok in RATE_LIMIT_ERRORS):
                    # Rotate model; after a full cycle, rotate key
                    prev_model_index = self.model_index
                    self._rotate_model()
                    rotations_on_key += 1
                    if rotations_on_key >= len(self.models):
                        self._rotate_key()
                        rotations_on_key = 0
                    time.sleep(backoff * (attempt + 1))
                    continue
                raise
        if last_err:
            raise last_err
        raise RuntimeError("Unknown error invoking structured chain")
