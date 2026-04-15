"""
models/base_llm.py
------------------
Abstract interface for any LLM backend + concrete implementations.

Backends
--------
MockLLM     — deterministic stubs, no server required (tests / CI)
OllamaLLM   — local Ollama server via /api/generate (production)
AnthropicLLM — stub ready to wire up (cloud)
HuggingFaceLLM — stub ready to wire up (local transformers)

Multi-model design
------------------
Each pipeline component receives its *own* LLM instance, so generator,
critic, and refiner can run different models simultaneously:

    generator_llm = OllamaLLM(config, model="mistral")
    critic_llm    = OllamaLLM(config, model="llama3")
    refiner_llm   = OllamaLLM(config, model="mistral")

The ``model`` parameter on ``OllamaLLM.__init__`` overrides
``config.model_name`` for that instance only — shared sampling params
(temperature, max_tokens, timeout) still come from ``config``.
"""

from __future__ import annotations

import json
import socket
import textwrap
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional

from config import LLMConfig


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    """
    Minimal interface every LLM backend must satisfy.

    Parameters
    ----------
    config:
        LLMConfig with model name, token limit, and sampling params.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send *prompt* to the model and return the raw text completion.

        Parameters
        ----------
        prompt:
            The full user-facing prompt text.
        system_prompt:
            Optional system / instruction prefix (used by chat models).

        Returns
        -------
        str
            Model-generated text, stripped of leading/trailing whitespace.
        """


# ---------------------------------------------------------------------------
# Mock implementation (no API keys, fully deterministic)
# ---------------------------------------------------------------------------

class MockLLM(BaseLLM):
    """
    Deterministic stub that returns scripted responses.

    Useful for:
    - Unit tests that must not hit an external API.
    - CI pipelines.
    - Local development without credentials.

    The mock inspects the prompt for keywords and returns a matching
    canned response.  The critic branch returns valid JSON so downstream
    parsing never breaks.
    """

    # Canned critic response — valid JSON matching CriticFeedback schema.
    # Score is below the default 7.0 threshold so the loop iterates in demos.
    _CRITIC_TEMPLATE = {
        "factual_errors": [
            "Claim about the mechanism is oversimplified and partially incorrect."
        ],
        "hallucinations": [
            "Referenced a study that does not exist in the provided context."
        ],
        "missing_concepts": [
            "Key concept of gradient flow is not mentioned.",
            "No discussion of edge cases or failure modes.",
        ],
        "logical_flaws": [
            "Conclusion does not follow from the stated premises."
        ],
        "improvement_actions": [
            "Remove or verify the referenced study — it appears fabricated.",
            "Correct the mechanism description with accurate details.",
            "Explain gradient flow explicitly.",
            "Restructure the conclusion to follow logically from the evidence.",
        ],
        "score": 4.5,
        "confidence": 0.78,
        "verdict": "poor",
    }

    # Canned "good enough" response — returned on even-numbered calls.
    _CRITIC_GOOD = {
        "factual_errors": [],
        "hallucinations": [],
        "missing_concepts": [
            "Could briefly note computational complexity trade-offs."
        ],
        "logical_flaws": [],
        "improvement_actions": [
            "Optionally add a sentence on complexity trade-offs for completeness."
        ],
        "score": 8.2,
        "confidence": 0.91,
        "verdict": "good",
    }

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._call_count: int = 0

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Return a canned response based on the role detected in *prompt*.

        The mock cycles through responses so that multi-iteration loops
        behave realistically: first critique flags issues, second passes.
        """
        self._call_count += 1
        prompt_lower = prompt.lower()

        # Detect the critic prompt by any of its structural markers.
        # "answer under review" is unique to the new research-grade critic prompt.
        # "critique"/"evaluate" kept for backwards compatibility with tests.
        _CRITIC_MARKERS = (
            "critique", "evaluate", "answer under review", "evaluation rules",
        )
        if any(marker in prompt_lower for marker in _CRITIC_MARKERS):
            template = (
                self._CRITIC_GOOD
                if self._call_count % 2 == 0
                else self._CRITIC_TEMPLATE
            )
            return json.dumps(template, indent=2)

        if "refine" in prompt_lower or "improve" in prompt_lower:
            return textwrap.dedent(
                f"""
                [Refined — pass {self._call_count}]

                Here is an improved answer that addresses the feedback:

                1. **Concrete example added**: Consider the case where X leads to Y
                   because of Z mechanism.
                2. **Structured explanation**: The process works as follows:
                   - Step A: initialise the context.
                   - Step B: apply the transformation.
                   - Step C: validate the result.
                3. **Summary**: In short, the answer is well-grounded in both
                   theory and practice.
                """
            ).strip()

        # Default: initial generation
        return textwrap.dedent(
            """
            This is a generated answer for the given query.

            The topic involves several interconnected concepts that are worth
            exploring carefully.  At a high level, the mechanism works by
            combining inputs in a way that produces a coherent output.

            Further details depend on the specific context provided.
            """
        ).strip()


# ---------------------------------------------------------------------------
# Ollama backend (local inference — no API key required)
# ---------------------------------------------------------------------------

class OllamaError(RuntimeError):
    """Raised when the Ollama server returns an error or is unreachable."""


class OllamaLLM(BaseLLM):
    """
    LLM backend backed by a local Ollama server.

    Calls ``POST {base_url}/api/generate`` with ``stream=false`` and
    returns the completed text.  No third-party libraries required —
    only stdlib ``urllib``.

    Parameters
    ----------
    config:
        Shared ``LLMConfig`` supplying temperature, max_tokens, base_url,
        and timeout.  ``config.model_name`` is used as the default model.
    model:
        Optional override.  When supplied, this model is used instead of
        ``config.model_name``.  Allows each pipeline component to run a
        different model while sharing one ``LLMConfig``.

    Raises
    ------
    OllamaError
        If Ollama is not running, the model is not found, or the request
        times out.  The message is human-readable so the operator knows
        exactly what to fix.

    Examples
    --------
    >>> llm = OllamaLLM(config, model="mistral")
    >>> llm.complete("What is backpropagation?")
    'Backpropagation is ...'
    """

    _ENDPOINT = "/api/generate"

    def __init__(self, config: LLMConfig, model: Optional[str] = None) -> None:
        super().__init__(config)
        self._model = model or config.model_name
        self._url = config.base_url.rstrip("/") + self._ENDPOINT
        self._timeout = config.timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """The Ollama model this instance is configured to use."""
        return self._model

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send *prompt* to Ollama and return the text completion.

        Parameters
        ----------
        prompt:
            User-facing prompt text.
        system_prompt:
            Optional instruction passed as the ``system`` field in the
            Ollama request body.

        Returns
        -------
        str
            Model output, stripped of leading/trailing whitespace.

        Raises
        ------
        OllamaError
            On connection failure, timeout, or a non-200 HTTP response.
        """
        payload = self._build_payload(prompt, system_prompt)
        raw_bytes = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=self._url,
            data=raw_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            self._raise_connection_error(exc)
        except socket.timeout:
            raise OllamaError(
                f"Ollama request timed out after {self._timeout}s. "
                f"Model '{self._model}' may be loading — try increasing "
                f"LLMConfig.timeout or running `ollama pull {self._model}` first."
            )

        return self._parse_response(body)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self, prompt: str, system_prompt: Optional[str]
    ) -> dict:
        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    @staticmethod
    def _parse_response(body: str) -> str:
        """Extract the ``response`` field from Ollama's JSON reply."""
        try:
            data = json.loads(body)
            return data["response"].strip()
        except (json.JSONDecodeError, KeyError) as exc:
            raise OllamaError(
                f"Unexpected response format from Ollama: {exc}\nBody: {body[:300]}"
            ) from exc

    def _raise_connection_error(self, exc: urllib.error.URLError) -> None:
        """Convert urllib errors into actionable OllamaError messages."""
        reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)

        if "refused" in reason.lower() or "connection" in reason.lower():
            raise OllamaError(
                f"Cannot connect to Ollama at '{self._url}'. "
                f"Make sure Ollama is running: `ollama serve`"
            ) from exc

        if isinstance(exc, urllib.error.HTTPError):
            if exc.code == 404:
                raise OllamaError(
                    f"Model '{self._model}' not found in Ollama. "
                    f"Pull it first: `ollama pull {self._model}`"
                ) from exc
            raise OllamaError(
                f"Ollama returned HTTP {exc.code}: {exc.reason}"
            ) from exc

        raise OllamaError(f"Ollama request failed: {reason}") from exc


# ---------------------------------------------------------------------------
# Cloud / HuggingFace stubs (implement to use instead of OllamaLLM)
# ---------------------------------------------------------------------------

class AnthropicLLM(BaseLLM):
    """
    Stub for Anthropic Claude via the ``anthropic`` SDK.

    Install: pip install anthropic
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        # import anthropic
        # self._client = anthropic.Anthropic()

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError(
            "Wire up the Anthropic client and uncomment the import above."
        )


class HuggingFaceLLM(BaseLLM):
    """
    Stub for a HuggingFace ``transformers`` pipeline.

    Install: pip install transformers torch
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        # from transformers import pipeline
        # self._pipe = pipeline("text-generation", model=config.model_name)

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError(
            "Instantiate the transformers pipeline and uncomment the import above."
        )
