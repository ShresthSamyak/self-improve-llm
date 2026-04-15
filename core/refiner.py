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
    strict_mode: bool = False


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

    # Escalated prompt used when hallucinations persisted across iterations.
    # Prioritises removing uncertain claims over completeness.
    _SYSTEM_PROMPT_STRICT = (
        "You are a rigorous fact-checker and editor operating in STRICT MODE. "
        "Hallucinations were detected and survived the previous refinement pass. "
        "Your absolute priority is factual accuracy — above completeness, above fluency. "
        "RULES: "
        "(1) Remove every claim flagged as a hallucination with no exceptions. "
        "(2) Do NOT introduce any new claim you cannot verify from the question context. "
        "(3) If a fact is uncertain, say so explicitly rather than asserting it. "
        "(4) A shorter, accurate answer is always better than a longer, fabricated one. "
        "Return ONLY the corrected answer — no commentary, no preamble."
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
        strict_mode: bool = False,
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
        strict_mode:
            When True, the escalated system prompt is used.  The loop sets
            this automatically when hallucinations persist across consecutive
            iterations — do not set it manually unless you know why.

        Returns
        -------
        RefinerOutput
            The refined answer alongside its provenance.
        """
        system_prompt = (
            self._SYSTEM_PROMPT_STRICT if strict_mode else self._SYSTEM_PROMPT
        )
        prompt = self._build_prompt(query, answer, feedback, strict_mode=strict_mode)

        logger.info(
            "Refiner: iteration=%d  strict_mode=%s", iteration, strict_mode
        )
        logger.debug("Refiner prompt:\n%s", prompt)

        refined = self._llm.complete(prompt, system_prompt=system_prompt)

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
            strict_mode=strict_mode,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self, query: str, answer: str, feedback: CriticFeedback,
        strict_mode: bool = False,
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

        header = (
            "*** STRICT MODE ACTIVE — hallucinations persisted. "
            "Accuracy over completeness. Remove all unverifiable claims. ***\n\n"
            if strict_mode else ""
        )
        return (
            f"{header}"
            "You are rewriting an answer to fix all identified issues.\n"
            "Priorities: (1) remove hallucinations, (2) correct facts, "
            "(3) fix logic, (4) add missing concepts.\n"
            "Do NOT mention the feedback in your answer. "
            "Return ONLY the improved answer.\n\n"
            + sections
        )
