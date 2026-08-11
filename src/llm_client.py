"""
LLM client with request tracking.

The client is intentionally configuration-driven:
- provider
- model
- API key
- base URL
- temperature
- timeout

are supplied externally.

No API key or model name is hard-coded here.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class LLMResponse:
    """Normalized response returned by the LLM client."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    raw_response: Dict[str, Any]

    @property
    def usage(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "raw_response": self.raw_response,
        }


class LLMClient:
    """
    Configuration-driven LLM client.

    Currently supports OpenAI-compatible chat-completions APIs.

    The provider/model are NOT hard-coded. They are passed to the client.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float = 0.0,
        timeout: int = 60,
    ) -> None:

        if not api_key:
            raise ValueError("API key is required.")

        if not model:
            raise ValueError("Model name is required.")

        if not base_url:
            raise ValueError("Base URL is required.")

        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                "Temperature must be between 0.0 and 2.0."
            )

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send a prompt to the configured LLM.

        Returns a normalized LLMResponse containing:
        - generated text
        - model
        - token usage
        - latency
        - raw response
        """

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        endpoint = f"{self.base_url}/chat/completions"

        start_time = time.perf_counter()

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.RequestException as exc:
            raise RuntimeError(
                f"LLM API request failed: {exc}"
            ) from exc

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "LLM API returned invalid JSON."
            ) from exc

        text = self._extract_text(data)

        usage = data.get("usage", {})

        input_tokens = int(
            usage.get(
                "prompt_tokens",
                usage.get("input_tokens", 0),
            )
            or 0
        )

        output_tokens = int(
            usage.get(
                "completion_tokens",
                usage.get("output_tokens", 0),
            )
            or 0
        )

        total_tokens = int(
            usage.get(
                "total_tokens",
                input_tokens + output_tokens,
            )
            or 0
        )

        self.call_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total_tokens

        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            raw_response=data,
        )

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        """Extract generated text from an OpenAI-compatible response."""

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Could not extract generated text from LLM response."
            ) from exc

        if not isinstance(text, str):
            raise RuntimeError(
                "LLM response content is not a string."
            )

        return text

    def stats(self) -> Dict[str, int]:
        """Return cumulative client usage statistics."""

        return {
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
        }