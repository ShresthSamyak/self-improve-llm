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
        def _section(title: str, items: list, marker: str = "-") -> str:
            if not items:
                return f"{title}: (none)"
            lines = "\n".join(f"  {marker} {item}" for item in items)
            return f"{title}:\n{lines}"

        # Hallucinations and factual errors get the most prominent placement
        # so the Refiner addresses them first.
        sections = "\n\n".join([
            f"ORIGINAL QUESTION:\n{query}",
            (
                f"CURRENT ANSWER  "
                f"[critic score: {feedback.score}/10  verdict: {feedback.verdict}]:\n"
                f"{answer}"
            ),
            _section(
                "[CRITICAL] HALLUCINATIONS — remove or replace with verified facts",
                feedback.hallucinations,
                marker="[!]",
            ),
            _section(
                "[CRITICAL] FACTUAL ERRORS — correct each one explicitly",
                feedback.factual_errors,
                marker="[x]",
            ),
            _section(
                "LOGICAL FLAWS — fix the reasoning",
                feedback.logical_flaws,
                marker="[~]",
            ),
            _section(
                "MISSING CONCEPTS — add where appropriate",
                feedback.missing_concepts,
                marker="[ ]",
            ),
            _section(
                "IMPROVEMENT ACTIONS — apply in order",
                feedback.improvement_actions,
                marker=">>",
            ),
        ])

        return (
            "You are rewriting an answer to fix all identified issues.\n"
            "Priorities: (1) remove hallucinations, (2) correct facts, "
            "(3) fix logic, (4) add missing concepts.\n"
            "Do NOT mention the feedback in your answer. "
            "Return ONLY the improved answer.\n\n"
            + sections
        )
