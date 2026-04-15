"""
core/loop.py
------------
Adaptive pipeline orchestrator: generate → critique → refine loop.

Behaviour overview
------------------
The loop is no longer a fixed counter — it acts as an intelligent
optimizer that monitors progress and adjusts strategy:

  1. Generate an initial answer (once).
  2. Critique the current answer.
  3. Evaluate three exit conditions in priority order:
       a. Quality met   → exit as "converged"
       b. Stagnation    → exit as "stagnated"  (score not improving)
       c. Max iters hit → exit as "exhausted"
  4. Determine refinement strategy:
       - If hallucinations persisted from the previous iteration,
         escalate the Refiner to strict_mode=True.
  5. Refine and loop back to step 2.

Stagnation detection
--------------------
A ``StagnationTracker`` records per-iteration score deltas.  When the
last ``stagnation_patience`` consecutive deltas are all below
``min_improvement_delta`` (default 0.3), further refinement is
unlikely to help — the loop exits early rather than burning iterations.

Adaptive strict mode
--------------------
If hallucinations survive one refiner pass, the next pass uses an
escalated system prompt that prioritises accuracy over completeness.
This is reset automatically when hallucinations clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from config import LoopConfig
from core.critic import Critic, CriticFeedback
from core.generator import Generator, GeneratorOutput
from core.refiner import Refiner, RefinerOutput
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stagnation tracker
# ---------------------------------------------------------------------------

class StagnationTracker:
    """
    Detects when score improvement flatlines across iterations.

    Maintains a rolling window of the last ``patience`` score deltas.
    Stagnation is declared when every delta in the window is below
    ``min_delta`` — meaning the refiner is no longer making meaningful
    progress.

    Parameters
    ----------
    patience:
        Number of consecutive low-delta iterations required before
        declaring stagnation.
    min_delta:
        Score improvement below this value is treated as no progress.
    """

    def __init__(self, patience: int, min_delta: float) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._deltas: List[float] = []

    def record(self, delta: float) -> None:
        """Record the score change from one iteration to the next."""
        self._deltas.append(delta)
        if len(self._deltas) > self._patience:
            self._deltas.pop(0)

    def is_stagnated(self) -> bool:
        """
        Return True if the last ``patience`` iterations all improved by
        less than ``min_delta``.

        Returns False until enough history has accumulated.
        """
        if len(self._deltas) < self._patience:
            return False
        return all(d < self._min_delta for d in self._deltas)

    @property
    def recent_deltas(self) -> List[float]:
        """Read-only view of the current rolling window."""
        return list(self._deltas)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """
    Snapshot of one full critic-refiner cycle.

    Fields
    ------
    iteration:
        1-based cycle number.
    answer:
        The answer that was critiqued in this iteration (before refinement).
    feedback:
        Full structured critic output.
    improvement_delta:
        Score change vs. the previous iteration.  0.0 for the first pass
        (no baseline yet).  Negative means the answer got worse.
    issue_count:
        Total number of issues across all categories (hallucinations +
        factual errors + logical flaws + missing concepts).
    hallucination_count:
        len(feedback.hallucinations).  Surfaced separately because
        hallucinations trigger the strictest exit/escalation rules.
    strict_mode_used:
        Whether the Refiner was called in strict mode after this critique.
        False when the loop exits without refining (final iteration).
    """
    iteration: int
    answer: str
    feedback: CriticFeedback
    improvement_delta: float
    issue_count: int
    hallucination_count: int
    strict_mode_used: bool = False


@dataclass
class PipelineResult:
    """
    Full result returned after the adaptive loop finishes.

    Fields
    ------
    query:
        Original user query.
    initial_answer:
        Raw Generator output before any refinement.
    final_answer:
        Best answer produced — the one from the iteration that triggered
        exit, or the last refined answer if the loop was exhausted.
    iterations:
        Ordered list of every critic-refiner cycle.
    total_iterations:
        Number of critique passes that ran.
    converged:
        True iff ``exit_reason == "converged"``.
    exit_reason:
        ``"converged"``  — quality thresholds met, answer is good.
        ``"stagnated"``  — score stopped improving; further passes unlikely
                           to help.
        ``"exhausted"``  — max_iterations reached without meeting quality.
    score_history:
        Critic score at each iteration in order.  Useful for plotting
        progress curves or post-run analysis.
    """
    query: str
    initial_answer: str
    final_answer: str
    iterations: List[IterationRecord] = field(default_factory=list)
    total_iterations: int = 0
    converged: bool = False
    exit_reason: str = "pending"
    score_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class RefinementLoop:
    """
    Adaptive Generator → Critic → Refiner orchestrator.

    Parameters
    ----------
    generator:
        Produces the initial candidate answer.
    critic:
        Scores any (query, answer) pair.
    refiner:
        Improves an answer given structured feedback.
    config:
        ``LoopConfig`` controlling iteration limits, quality thresholds,
        and stagnation detection parameters.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> PipelineResult:
        """
        Execute the adaptive self-correcting pipeline for *query*.

        Returns
        -------
        PipelineResult
            Complete record including all iteration snapshots, score
            history, and the reason the loop exited.
        """
        logger.info("=" * 60)
        logger.info("Pipeline start | query: %r", query[:80])
        logger.info(
            "Config: max_iter=%d  min_score=%.1f  "
            "stagnation_patience=%d  min_delta=%.2f",
            self._config.max_iterations,
            self._config.min_quality_score,
            self._config.stagnation_patience,
            self._config.min_improvement_delta,
        )
        logger.info("=" * 60)

        # --- Step 1: Generate initial answer ---
        gen_output: GeneratorOutput = self._generator.generate(query)
        current_answer = gen_output.answer

        result = PipelineResult(
            query=query,
            initial_answer=current_answer,
            final_answer=current_answer,
        )

        stagnation = StagnationTracker(
            patience=self._config.stagnation_patience,
            min_delta=self._config.min_improvement_delta,
        )
        previous_score: Optional[float] = None

        # --- Steps 2–N: Critique → Decide → (maybe) Refine ---
        for iteration in range(1, self._config.max_iterations + 1):
            logger.info("--- Iteration %d / %d ---", iteration, self._config.max_iterations)

            # 2a. Critique
            feedback = self._critic.critique(query, current_answer)

            # 2b. Compute per-iteration metadata
            delta = (
                round(feedback.score - previous_score, 3)
                if previous_score is not None
                else 0.0
            )
            if previous_score is not None:
                stagnation.record(delta)

            record = IterationRecord(
                iteration=iteration,
                answer=current_answer,
                feedback=feedback,
                improvement_delta=delta,
                issue_count=feedback.total_issue_count,
                hallucination_count=len(feedback.hallucinations),
            )
            result.iterations.append(record)
            result.total_iterations = iteration
            result.score_history.append(feedback.score)

            self._log_iteration_summary(record, stagnation)

            # 2c. Check exit conditions (quality, stagnation, max iters)
            exit_reason = self._check_exit(feedback, stagnation, iteration)
            if exit_reason:
                result.converged = exit_reason == "converged"
                result.exit_reason = exit_reason
                result.final_answer = current_answer
                self._log_exit(result)
                break

            # 2d. Decide refinement strategy
            strict_mode = self._should_use_strict_mode(feedback, result.iterations)
            record.strict_mode_used = strict_mode

            if strict_mode:
                logger.warning(
                    "Strict mode ACTIVATED — hallucinations persisted "
                    "from previous iteration."
                )

            # 2e. Refine
            refiner_output: RefinerOutput = self._refiner.refine(
                query=query,
                answer=current_answer,
                feedback=feedback,
                iteration=iteration,
                strict_mode=strict_mode,
            )
            current_answer = refiner_output.refined_answer
            previous_score = feedback.score

        else:
            # Loop ran to completion without an explicit break.
            result.final_answer = current_answer
            result.exit_reason = "exhausted"
            result.converged = False
            self._log_exit(result)

        logger.info(
            "Pipeline done | iterations=%d | exit=%s | final_score=%.1f",
            result.total_iterations,
            result.exit_reason,
            result.score_history[-1] if result.score_history else 0.0,
        )
        return result

    # ------------------------------------------------------------------
    # Exit condition logic
    # ------------------------------------------------------------------

    def _check_exit(
        self,
        feedback: CriticFeedback,
        stagnation: StagnationTracker,
        iteration: int,
    ) -> Optional[str]:
        """
        Evaluate all exit conditions in priority order.

        Returns the exit reason string if the loop should stop, or
        None to continue.

        Priority
        --------
        1. Quality met   → "converged"  (positive exit)
        2. Stagnation    → "stagnated"  (give up gracefully)
        3. Max iters hit → "exhausted"  (hard stop)
        """
        if self._quality_met(feedback):
            return "converged"
        if stagnation.is_stagnated():
            logger.warning(
                "Stagnation detected — last %d deltas %s all below %.2f threshold. "
                "Stopping early.",
                self._config.stagnation_patience,
                stagnation.recent_deltas,
                self._config.min_improvement_delta,
            )
            return "stagnated"
        if iteration >= self._config.max_iterations:
            return "exhausted"
        return None

    def _quality_met(self, feedback: CriticFeedback) -> bool:
        """
        Return True only when the answer is genuinely acceptable.

        Hard blocks (any one fails the check):
        - Any hallucinations present.
        - Any factual errors present.

        Soft requirements (both must hold):
        - Penalised score >= min_quality_score.
        - Verdict is "good" or "excellent".
        """
        if feedback.has_hallucinations:
            return False
        if feedback.factual_errors:
            return False
        return (
            feedback.score >= self._config.min_quality_score
            and feedback.verdict in ("good", "excellent")
        )

    # ------------------------------------------------------------------
    # Adaptive strategy
    # ------------------------------------------------------------------

    @staticmethod
    def _should_use_strict_mode(
        feedback: CriticFeedback,
        history: List[IterationRecord],
    ) -> bool:
        """
        Return True if the Refiner should run in strict mode.

        Condition: hallucinations are present in the CURRENT critique
        AND were also present in the PREVIOUS iteration's critique.
        This indicates the normal refiner failed to remove them.
        """
        if not feedback.has_hallucinations:
            return False
        if len(history) < 2:
            return False
        return history[-2].feedback.has_hallucinations

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_iteration_summary(
        record: IterationRecord,
        stagnation: StagnationTracker,
    ) -> None:
        delta_str = (
            f"{record.improvement_delta:+.2f}"
            if record.iteration > 1
            else "  n/a"
        )
        progress = "improving" if record.improvement_delta >= 0.3 else (
            "stagnant"  if record.improvement_delta < 0.3 and record.iteration > 1
            else "first"
        )
        logger.info(
            "  score=%.2f  delta=%s  verdict=%s  "
            "hallucinations=%d  issues=%d  progress=%s",
            record.feedback.score,
            delta_str,
            record.feedback.verdict,
            record.hallucination_count,
            record.issue_count,
            progress,
        )

    @staticmethod
    def _log_exit(result: PipelineResult) -> None:
        last_score = result.score_history[-1] if result.score_history else 0.0
        first_score = result.score_history[0] if result.score_history else 0.0
        total_gain = round(last_score - first_score, 2)

        logger.info(
            "Loop exit | reason=%s | score %s -> %s (gain=%s) | iterations=%d",
            result.exit_reason.upper(),
            f"{first_score:.2f}",
            f"{last_score:.2f}",
            f"{total_gain:+.2f}",
            result.total_iterations,
        )

        if result.exit_reason == "stagnated":
            logger.warning(
                "Stagnated exit — answer did not improve enough to converge. "
                "Consider a stronger model or more max_iterations."
            )
        elif result.exit_reason == "exhausted":
            last_fb = result.iterations[-1].feedback if result.iterations else None
            if last_fb and last_fb.has_hallucinations:
                logger.warning(
                    "Exhausted with %d hallucination(s) still present: %s",
                    len(last_fb.hallucinations),
                    last_fb.hallucinations,
                )
            elif last_fb and last_fb.factual_errors:
                logger.warning(
                    "Exhausted with %d factual error(s). Manual review recommended.",
                    len(last_fb.factual_errors),
                )
