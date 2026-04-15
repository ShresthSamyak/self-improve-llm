"""
core/refiner.py
---------------
Refiner module: improves an answer given structured critic feedback.

Responsibility
--------------
The Refiner receives the original query, the current answer, and a
``CriticFeedback`` object.  It constructs a targeted improvement prompt
that includes the specific issues and suggestions from the critic, then
asks the LLM to produce a better version.

The Refiner is stateless.  It does *not* decide whether refinement is
necessary — that is the loop's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import LLMConfig
from core.critic import CriticFeedback
from models.base_llm import BaseLLM
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class RefinerOutput:
    """Structured result returned by the Refiner."""
    query: str
    previous_answer: str
    refined_answer: str
    feedback_applied: CriticFeedback
    iteration: int


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------

class Refiner:
    """
    Takes a (query, answer, feedback) triple and produces an improved answer.

    Parameters
    ----------
    llm:
        Any ``BaseLLM`` implementation.
    config:
        LLMConfig (passed through for reference).
    """

    _SYSTEM_PROMPT = (
        "You are an expert editor and domain specialist. "
        "You will receive an answer and specific feedback. "
        "Rewrite the answer to fully address every issue and suggestion listed. "
        "Do NOT acknowledge the feedback in your reply — just return the improved answer."
    )

    def __init__(self, llm: BaseLLM, config: LLMConfig) -> None:
        self._llm = llm
        self._config = config

    def refine(
        self,
        query: str,
        answer: str,
        feedback: CriticFeedback,
        iteration: int = 1,
    ) -> RefinerOutput:
        """
        Produce a refined answer for *query* by addressing *feedback*.

        Parameters
        ----------
        query:
            The original user question.
        answer:
            The current (unrefined) answer.
        feedback:
            Structured critic output containing issues and suggestions.
        iteration:
            Current loop iteration number (used for logging only).

        Returns
        -------
        RefinerOutput
            The refined answer alongside its provenance.
        """
        prompt = self._build_prompt(query, answer, feedback)
        logger.info("Refiner: applying feedback (iteration %d).", iteration)
        logger.debug("Refiner prompt:\n%s", prompt)

        refined = self._llm.complete(prompt, system_prompt=self._SYSTEM_PROMPT)

        logger.info(
            "Refiner: done. Answer length %d -> %d chars.",
            len(answer),
            len(refined),
        )
        return RefinerOutput(
            query=query,
            previous_answer=answer,
            refined_answer=refined,
            feedback_applied=feedback,
            iteration=iteration,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self, query: str, answer: str, feedback: CriticFeedback
    ) -> str:
        issues_text = "\n".join(f"  - {i}" for i in feedback.issues) or "  (none)"
        suggestions_text = (
            "\n".join(f"  - {s}" for s in feedback.suggestions) or "  (none)"
        )

        return (
            f"Improve the following answer based on the critic feedback below.\n\n"
            f"ORIGINAL QUESTION:\n{query}\n\n"
            f"CURRENT ANSWER:\n{answer}\n\n"
            f"ISSUES IDENTIFIED (score {feedback.score}/10):\n{issues_text}\n\n"
            f"SUGGESTIONS:\n{suggestions_text}\n\n"
            f"Write an improved version of the answer that addresses all issues "
            f"and incorporates all suggestions. Return only the refined answer."
        )
