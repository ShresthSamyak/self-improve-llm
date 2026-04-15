"""
core/loop.py
------------
Pipeline orchestrator: runs the generate → critique → refine loop.

Responsibility
--------------
``RefinementLoop`` owns the iterative control flow.  It:

1. Calls ``Generator`` once to produce an initial answer.
2. Calls ``Critic`` to score the current answer.
3. If the answer is good enough (score ≥ threshold OR verdict is
   "acceptable" OR max iterations reached), exits and returns the
   final result.
4. Otherwise calls ``Refiner`` and repeats from step 2.

This module deliberately contains *no* LLM logic — it is pure
orchestration.  Swapping any component only requires passing a
different object to the constructor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from config import LoopConfig
from core.critic import Critic, CriticFeedback
from core.generator import Generator, GeneratorOutput
from core.refiner import Refiner, RefinerOutput
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """Snapshot of one critic-refiner pass."""
    iteration: int
    answer: str
    feedback: CriticFeedback


@dataclass
class PipelineResult:
    """
    Full result returned after the loop finishes.

    Fields
    ------
    query:
        The original user query.
    initial_answer:
        Raw output from the Generator before any refinement.
    final_answer:
        Best answer after all loop iterations.
    iterations:
        Ordered list of every (answer, feedback) pair produced.
    total_iterations:
        Number of critic-refiner cycles that ran.
    converged:
        True if the loop exited because quality met the threshold
        (not because max_iterations was exhausted).
    """
    query: str
    initial_answer: str
    final_answer: str
    iterations: List[IterationRecord] = field(default_factory=list)
    total_iterations: int = 0
    converged: bool = False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class RefinementLoop:
    """
    Orchestrates the Generator → Critic → Refiner pipeline.

    Parameters
    ----------
    generator:
        Produces the initial candidate answer.
    critic:
        Scores any (query, answer) pair.
    refiner:
        Improves an answer given structured feedback.
    config:
        ``LoopConfig`` controlling iteration limits and thresholds.
    """

    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        refiner: Refiner,
        config: LoopConfig,
    ) -> None:
        self._generator = generator
        self._critic = critic
        self._refiner = refiner
        self._config = config

    def run(self, query: str) -> PipelineResult:
        """
        Execute the full self-correcting pipeline for *query*.

        Parameters
        ----------
        query:
            Raw user question or task description.

        Returns
        -------
        PipelineResult
            Complete record including initial answer, every iteration,
            the final answer, and convergence status.
        """
        logger.info("=" * 60)
        logger.info("Pipeline start | query: %r", query[:80])
        logger.info("=" * 60)

        # --- Step 1: Generate initial answer ---
        gen_output: GeneratorOutput = self._generator.generate(query)
        current_answer = gen_output.answer

        result = PipelineResult(
            query=query,
            initial_answer=current_answer,
            final_answer=current_answer,
        )

        # --- Step 2–N: Critique → (maybe) Refine ---
        for iteration in range(1, self._config.max_iterations + 1):
            logger.info("--- Iteration %d / %d ---", iteration, self._config.max_iterations)

            feedback = self._critic.critique(query, current_answer)
            result.iterations.append(
                IterationRecord(
                    iteration=iteration,
                    answer=current_answer,
                    feedback=feedback,
                )
            )
            result.total_iterations = iteration

            if self._should_stop(feedback, iteration):
                result.converged = self._quality_met(feedback)
                result.final_answer = current_answer
                logger.info(
                    "Loop exiting | converged=%s | score=%.1f | verdict=%s | "
                    "hallucinations=%d | factual_errors=%d",
                    result.converged,
                    feedback.score,
                    feedback.verdict,
                    len(feedback.hallucinations),
                    len(feedback.factual_errors),
                )
                break

            refiner_output: RefinerOutput = self._refiner.refine(
                query=query,
                answer=current_answer,
                feedback=feedback,
                iteration=iteration,
            )
            current_answer = refiner_output.refined_answer

        else:
            # Exhausted all iterations — use whatever we have.
            result.final_answer = current_answer
            result.converged = False
            last_feedback = result.iterations[-1].feedback if result.iterations else None
            if last_feedback and last_feedback.has_hallucinations:
                logger.warning(
                    "Loop exhausted but final answer still contains %d hallucination(s): %s. "
                    "Consider increasing max_iterations or using a stronger model.",
                    len(last_feedback.hallucinations),
                    last_feedback.hallucinations,
                )
            elif last_feedback and last_feedback.factual_errors:
                logger.warning(
                    "Loop exhausted but final answer still has %d factual error(s). "
                    "Manual review recommended.",
                    len(last_feedback.factual_errors),
                )
            logger.info(
                "Loop exhausted max iterations (%d). Final score=%.1f verdict=%s.",
                self._config.max_iterations,
                last_feedback.score if last_feedback else 0.0,
                last_feedback.verdict if last_feedback else "unknown",
            )

        logger.info("Pipeline done | total_iterations=%d", result.total_iterations)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quality_met(self, feedback: CriticFeedback) -> bool:
        """
        Return True only when the answer is genuinely acceptable.

        Rules (all must hold):
        - Zero hallucinations.  A single hallucination is an automatic fail.
        - Zero factual errors.  Incorrect facts cannot be "good enough".
        - Penalised score >= configured minimum (default 7.0).
        - Verdict is "good" or "excellent" — "acceptable" and "poor"
          always require another pass.
        """
        if feedback.has_hallucinations:
            logger.debug("quality_met=False: hallucinations present.")
            return False
        if feedback.factual_errors:
            logger.debug("quality_met=False: factual errors present.")
            return False
        score_ok   = feedback.score >= self._config.min_quality_score
        verdict_ok = feedback.verdict in ("good", "excellent")
        return score_ok and verdict_ok

    def _should_stop(self, feedback: CriticFeedback, iteration: int) -> bool:
        """Return True if the loop should exit (with or without converging)."""
        if self._quality_met(feedback):
            return True
        if iteration >= self._config.max_iterations:
            return True
        return False
