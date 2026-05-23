from __future__ import annotations

"""
Ollama HTTP Client — clean abstraction over the Ollama local API.

Uses httpx (sync) for simplicity. No streaming — we need the full response
before building the answer.

Implements:
- Configurable model selection (default: qwen2.5:3b from settings)
- Timeout handling (configurable, default 120s)
- Retry with exponential backoff (3 attempts: 1s, 2s, 4s)
- Clean OllamaConnectionError for all failure modes
- Model availability check
- No dependency on the reasoning engine — pure transport layer
"""

import json
import logging
import time
from typing import Optional

import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when Ollama is unreachable or returns an error after all retries."""
    pass


class OllamaClient:
    """
    HTTP client for the Ollama local inference API.

    Args:
        host: Ollama API host (default from settings).
        model: Default model name (default from settings).
        timeout: Request timeout in seconds.
        max_retries: Number of retry attempts.
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.host = (host or settings.OLLAMA_HOST).rstrip("/")
        self.default_model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT
        self.max_retries = max_retries or settings.OLLAMA_MAX_RETRIES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate a response from Ollama.

        Args:
            prompt: The user prompt (contains assembled code context + question).
            system: System prompt setting the role and constraints.
            model: Override model name (defaults to self.default_model).
            temperature: Sampling temperature (0.1 = deterministic, grounded).

        Returns:
            Generated response text.

        Raises:
            OllamaConnectionError: If Ollama is unreachable or errors after retries.
        """
        selected_model = model or self.default_model

        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
            },
        }
        if system:
            payload["system"] = system

        logger.debug(
            "Ollama generate — model=%s prompt_len=%d",
            selected_model,
            len(prompt),
        )

        raw = self._request_with_retry("POST", "/api/generate", payload)

        response_text = raw.get("response", "").strip()
        if not response_text:
            logger.warning("Ollama returned empty response for model=%s", selected_model)
            response_text = "(No response generated)"

        logger.debug("Ollama response — %d chars", len(response_text))
        return response_text

    def check_available(self, model: str | None = None) -> bool:
        """
        Check if Ollama is running and the specified model is available.

        Returns:
            True if Ollama is reachable and model exists, False otherwise.
        """
        selected_model = model or self.default_model
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                # Ollama model names may include tags: "qwen2.5:3b"
                # Check both exact match and prefix match
                available = any(
                    m == selected_model or m.startswith(selected_model.split(":")[0])
                    for m in models
                )
                if not available:
                    logger.warning(
                        "Model '%s' not found in Ollama. Available: %s",
                        selected_model,
                        models,
                    )
                return available
        except Exception as exc:
            logger.warning("Ollama not reachable at %s: %s", self.host, exc)
            return False

    def list_models(self) -> list[str]:
        """Return list of model names available in Ollama."""
        try:
            raw = self._request_with_retry("GET", "/api/tags", {})
            return [m["name"] for m in raw.get("models", [])]
        except OllamaConnectionError:
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        payload: dict,
    ) -> dict:
        """
        Execute an HTTP request with exponential backoff retry.

        Retry delays: 1s, 2s, 4s (doubles each attempt).

        Raises:
            OllamaConnectionError: After all retries are exhausted.
        """
        url = f"{self.host}{endpoint}"
        last_exc: Optional[Exception] = None
        delay = 1.0

        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    if method == "GET":
                        response = client.get(url)
                    else:
                        response = client.post(url, json=payload)

                    # Ollama returns 200 for errors too sometimes —
                    # check status explicitly
                    if response.status_code >= 400:
                        body = response.text[:500]
                        raise OllamaConnectionError(
                            f"Ollama HTTP {response.status_code}: {body}"
                        )

                    return response.json()

            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning(
                    "Ollama request timed out (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
            except httpx.ConnectError as exc:
                last_exc = exc
                logger.warning(
                    "Ollama connection refused (attempt %d/%d). Is Ollama running at %s?",
                    attempt,
                    self.max_retries,
                    self.host,
                )
            except json.JSONDecodeError as exc:
                last_exc = exc
                logger.error("Ollama returned non-JSON response: %s", exc)
                raise OllamaConnectionError(f"Invalid JSON from Ollama: {exc}") from exc
            except OllamaConnectionError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.error("Unexpected Ollama error (attempt %d/%d): %s", attempt, self.max_retries, exc)

            if attempt < self.max_retries:
                logger.debug("Retrying in %.1fs...", delay)
                time.sleep(delay)
                delay *= 2

        raise OllamaConnectionError(
            f"Ollama unreachable after {self.max_retries} attempts. Last error: {last_exc}"
        )
