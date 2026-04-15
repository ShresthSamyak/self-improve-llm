"""
core/critic.py
--------------
Critic module: evaluates an answer and returns structured feedback.

Responsibility
--------------
Given a (query, answer) pair, the Critic asks the LLM to act as a
rigorous reviewer.  The LLM must return valid JSON matching
``CriticFeedback``.  A robust JSON parser recovers from minor LLM
formatting slips (markdown fences, trailing commas).

The Critic is stateless — the same instance can be reused across loop
iterations without side effects.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List

from config import LLMConfig
from models.base_llm import BaseLLM
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class CriticFeedback:
    """
    Structured output from one critic pass.

    Fields
    ------
    score:
        Overall quality, 0–10.
    confidence:
        Critic's self-assessed reliability, 0–1.
    issues:
        List of specific problems found.
    suggestions:
        Actionable improvements the refiner should apply.
    verdict:
        ``"acceptable"`` | ``"needs_improvement"`` | ``"unacceptable"``
    raw_response:
        Original LLM text, preserved for debugging / audit.
    """
    score: float
    confidence: float
    issues: List[str]
    suggestions: List[str]
    verdict: str
    raw_response: str = field(repr=False)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic:
    """
    Uses an LLM to critique a (query, answer) pair.

    The response is parsed into ``CriticFeedback``.  If parsing fails,
    a safe fallback is returned so the pipeline never crashes on a
    malformed LLM response.

    Parameters
    ----------
    llm:
        Any ``BaseLLM`` implementation.
    config:
        LLMConfig (passed through for reference; sampling lives in LLM).
    """

    _SYSTEM_PROMPT = (
        "You are a strict but fair answer quality reviewer. "
        "Your job is to identify weaknesses and suggest concrete improvements. "
        "Always respond with valid JSON only — no markdown, no prose."
    )

    _RESPONSE_SCHEMA = """
{
  "score": <float 0-10>,
  "confidence": <float 0-1>,
  "issues": ["<issue 1>", "..."],
  "suggestions": ["<suggestion 1>", "..."],
  "verdict": "<acceptable | needs_improvement | unacceptable>"
}"""

    def __init__(self, llm: BaseLLM, config: LLMConfig) -> None:
        self._llm = llm
        self._config = config

    def critique(self, query: str, answer: str) -> CriticFeedback:
        """
        Evaluate *answer* in the context of *query*.

        Parameters
        ----------
        query:
            The original user question.
        answer:
            The answer to be evaluated (initial or refined).

        Returns
        -------
        CriticFeedback
            Parsed structured feedback.  Falls back to a safe default on
            JSON parse failure.
        """
        prompt = self._build_prompt(query, answer)
        logger.info("Critic: evaluating answer.")

        raw = self._llm.complete(prompt, system_prompt=self._SYSTEM_PROMPT)
        logger.debug("Critic raw response:\n%s", raw)

        feedback = self._parse(raw)
        logger.info(
            "Critic: score=%.1f  confidence=%.2f  verdict=%s",
            feedback.score,
            feedback.confidence,
            feedback.verdict,
        )
        return feedback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, answer: str) -> str:
        return (
            f"Critique and evaluate the following answer.\n\n"
            f"QUERY:\n{query}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"Respond ONLY with a JSON object matching this schema:\n"
            f"{self._RESPONSE_SCHEMA}"
        )

    def _parse(self, raw: str) -> CriticFeedback:
        """
        Parse LLM output into ``CriticFeedback``.

        Strips markdown code fences before parsing, since LLMs often
        wrap JSON in ```json ... ``` even when instructed not to.
        """
        cleaned = self._strip_fences(raw)
        try:
            data = json.loads(cleaned)
            return CriticFeedback(
                score=float(data.get("score", 5.0)),
                confidence=float(data.get("confidence", 0.5)),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                verdict=data.get("verdict", "needs_improvement"),
                raw_response=raw,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Critic: JSON parse failed (%s). Using fallback.", exc)
            return self._fallback_feedback(raw)

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove ```json ... ``` or ``` ... ``` wrappers."""
        return re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text).strip()

    @staticmethod
    def _fallback_feedback(raw: str) -> CriticFeedback:
        """Return a conservative fallback when parsing fails."""
        return CriticFeedback(
            score=4.0,
            confidence=0.3,
            issues=["Could not parse structured feedback from critic."],
            suggestions=["Re-attempt with a clearer, more detailed answer."],
            verdict="needs_improvement",
            raw_response=raw,
        )
