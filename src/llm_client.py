"""
LLM client with request tracking and transient-error retry handling.

The client is intentionally configuration-driven:

- provider
- model
- API key
- base URL
- temperature
- timeout
- maximum output tokens

are supplied externally.

No API key or model name is hard-coded here.
"""

from __future__ import annotations

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
        max_output_tokens: int = 1500,
        max_request_retries: int = 3,
        retry_base_delay: float = 2.0,
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

        if max_output_tokens <= 0:
            raise ValueError(
                "max_output_tokens must be greater than 0."
            )

        if max_request_retries < 0:
            raise ValueError(
                "max_request_retries cannot be negative."
            )

        if retry_base_delay < 0:
            raise ValueError(
                "retry_base_delay cannot be negative."
            )

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.max_request_retries = max_request_retries
        self.retry_base_delay = retry_base_delay

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

        Retries transient HTTP failures such as:
        - 429 Too Many Requests
        - 500
        - 502
        - 503
        - 504

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
            "max_tokens": self.max_output_tokens,

            # Ask the provider/model for a JSON object.
            # OpenRouter supports this for compatible models.
            "response_format": {
                "type": "json_object"
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        endpoint = f"{self.base_url}/chat/completions"

        start_time = time.perf_counter()

        response = self._post_with_retries(
            endpoint=endpoint,
            headers=headers,
            payload=payload,
        )

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

    def _post_with_retries(
        self,
        *,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> requests.Response:
        """
        Perform the HTTP request with controlled retry handling.

        Handles:
        - 429 rate limiting
        - 500
        - 502
        - 503
        - 504
        - transient network failures

        For HTTP 429, the server's Retry-After header is preferred.
        If it is unavailable, exponential backoff is used.
        """

        transient_statuses = {
            429,
            500,
            502,
            503,
            504,
        }

        for attempt in range(self.max_request_retries + 1):

            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

            except requests.RequestException as exc:

                if attempt >= self.max_request_retries:
                    raise RuntimeError(
                        f"LLM API request failed after "
                        f"{self.max_request_retries} retries: {exc}"
                    ) from exc

                delay = self._calculate_retry_delay(
                    attempt=attempt,
                    response=None,
                )

                print(
                    f"    Transient network error. "
                    f"Retrying in {delay:.1f}s "
                    f"(attempt {attempt + 2}/"
                    f"{self.max_request_retries + 1})...",
                    flush=True,
                )

                time.sleep(delay)
                continue

            # -----------------------------------------------------------
            # Successful response
            # -----------------------------------------------------------

            if 200 <= response.status_code < 300:
                return response

            # -----------------------------------------------------------
            # HTTP 429 - rate limiting
            # -----------------------------------------------------------

            if response.status_code == 429:

                error_body = response.text[:2000].strip()

                if attempt >= self.max_request_retries:
                    raise RuntimeError(
                        "LLM API rate limit exceeded "
                        f"(HTTP 429) after "
                        f"{self.max_request_retries} retries.\n"
                        f"Response: {error_body}"
                    )

                delay = self._calculate_retry_delay(
                    attempt=attempt,
                    response=response,
                )

                print(
                    f"    HTTP 429 rate limit. "
                    f"Retrying in {delay:.1f}s "
                    f"(attempt {attempt + 2}/"
                    f"{self.max_request_retries + 1})...",
                    flush=True,
                )

                time.sleep(delay)
                continue

            # -----------------------------------------------------------
            # Other transient server errors
            # -----------------------------------------------------------

            if response.status_code in transient_statuses:

                error_body = response.text[:2000].strip()

                if attempt >= self.max_request_retries:
                    raise RuntimeError(
                        f"LLM API request failed with HTTP "
                        f"{response.status_code} after "
                        f"{self.max_request_retries} retries.\n"
                        f"Response: {error_body}"
                    )

                delay = self._calculate_retry_delay(
                    attempt=attempt,
                    response=response,
                )

                print(
                    f"    HTTP {response.status_code} from LLM API. "
                    f"Retrying in {delay:.1f}s "
                    f"(attempt {attempt + 2}/"
                    f"{self.max_request_retries + 1})...",
                    flush=True,
                )

                time.sleep(delay)
                continue

            # -----------------------------------------------------------
            # Non-transient API error
            # -----------------------------------------------------------

            error_body = response.text[:2000].strip()

            raise RuntimeError(
                f"LLM API request failed with HTTP "
                f"{response.status_code}.\n"
                f"Response: {error_body}"
            )

        raise RuntimeError(
            "LLM API request failed unexpectedly."
        ) 
    
    def _calculate_retry_delay(
        self,
        *,
        attempt: int,
        response: Optional[requests.Response],
    ) -> float:
        """
        Calculate retry delay.

        Prefer Retry-After when supplied by the server.
        Otherwise use exponential backoff.
        """

        if response is not None:
            retry_after = response.headers.get("Retry-After")

            if retry_after:
                try:
                    retry_after_seconds = float(retry_after)

                    if retry_after_seconds >= 0:
                        return retry_after_seconds
                except ValueError:
                    pass

        return self.retry_base_delay * (2 ** attempt)

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