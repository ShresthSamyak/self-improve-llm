"""
evaluation/metrics.py
---------------------
Lightweight output quality metrics for the self-correcting pipeline.

Responsibility
--------------
``MetricsEvaluator`` computes observable, heuristic-based signals
about answer quality.  These are intentionally model-free — no LLM
call, no external API — so they run instantly and are fully
reproducible.

They are meant to complement, not replace, the Critic.  Typical uses:
- Logging quality trends across iterations.
- Early-exit heuristics in the loop.
- Offline dataset evaluation.
- Regression testing when building a training corpus.

To add a learned metric (BLEU, BERTScore, etc.) subclass
``MetricsEvaluator`` and override ``evaluate`` or add new methods.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from core.loop import PipelineResult
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class AnswerMetrics:
    """
    Per-answer quality signals.

    All fields are on [0, 1] where 1 is best, *except* where noted.

    Fields
    ------
    char_count:
        Raw character count (not normalised).
    word_count:
        Raw word count (not normalised).
    sentence_count:
        Approximate sentence count.
    has_structure:
        1.0 if the answer uses numbered lists, bullets, or headers.
    avg_sentence_length:
        Mean words-per-sentence.  A heuristic for density.
    lexical_diversity:
        Unique tokens / total tokens.  Higher → less repetition.
    coverage_score:
        Fraction of query keywords present in the answer.
    composite_score:
        Weighted composite of the above signals, 0–10.
    """
    char_count: int
    word_count: int
    sentence_count: int
    has_structure: float
    avg_sentence_length: float
    lexical_diversity: float
    coverage_score: float
    composite_score: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PipelineMetrics:
    """Aggregated metrics across a full pipeline run."""
    query: str
    initial_metrics: AnswerMetrics
    final_metrics: AnswerMetrics
    score_delta: float          # final composite − initial composite
    iterations_run: int
    converged: bool
    iteration_scores: List[float]   # composite score at each iteration


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class MetricsEvaluator:
    """
    Computes heuristic quality metrics for pipeline outputs.

    Parameters
    ----------
    min_word_count:
        Minimum expected answer length in words.  Answers below this
        are penalised in the composite score.
    max_word_count:
        Upper bound for length normalisation.
    """

    def __init__(
        self,
        min_word_count: int = 30,
        max_word_count: int = 300,
    ) -> None:
        self._min_words = min_word_count
        self._max_words = max_word_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_answer(
        self, answer: str, query: Optional[str] = None
    ) -> AnswerMetrics:
        """
        Compute quality signals for a single *answer*.

        Parameters
        ----------
        answer:
            Text to evaluate.
        query:
            Optional original query; used for keyword coverage scoring.

        Returns
        -------
        AnswerMetrics
        """
        words = self._tokenize(answer)
        sentences = self._split_sentences(answer)
        word_count = len(words)
        sentence_count = max(len(sentences), 1)

        has_structure = self._detect_structure(answer)
        avg_sent_len = word_count / sentence_count
        lexical_diversity = self._lexical_diversity(words)
        coverage = self._coverage_score(words, query) if query else 0.5

        composite = self._composite(
            word_count=word_count,
            has_structure=has_structure,
            lexical_diversity=lexical_diversity,
            coverage=coverage,
            avg_sent_len=avg_sent_len,
        )

        return AnswerMetrics(
            char_count=len(answer),
            word_count=word_count,
            sentence_count=sentence_count,
            has_structure=has_structure,
            avg_sentence_length=round(avg_sent_len, 2),
            lexical_diversity=round(lexical_diversity, 3),
            coverage_score=round(coverage, 3),
            composite_score=round(composite, 2),
        )

    def evaluate_pipeline(self, result: PipelineResult) -> PipelineMetrics:
        """
        Compute metrics across a full ``PipelineResult``.

        Parameters
        ----------
        result:
            Completed pipeline run (from ``RefinementLoop.run``).

        Returns
        -------
        PipelineMetrics
        """
        initial = self.evaluate_answer(result.initial_answer, result.query)
        final = self.evaluate_answer(result.final_answer, result.query)

        iteration_scores = [
            self.evaluate_answer(it.answer, result.query).composite_score
            for it in result.iterations
        ]

        pm = PipelineMetrics(
            query=result.query,
            initial_metrics=initial,
            final_metrics=final,
            score_delta=round(final.composite_score - initial.composite_score, 2),
            iterations_run=result.total_iterations,
            converged=result.converged,
            iteration_scores=iteration_scores,
        )

        logger.info(
            "Metrics | initial=%.2f  final=%.2f  delta=%+.2f  converged=%s",
            initial.composite_score,
            final.composite_score,
            pm.score_delta,
            pm.converged,
        )
        return pm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _composite(
        self,
        word_count: int,
        has_structure: float,
        lexical_diversity: float,
        coverage: float,
        avg_sent_len: float,
    ) -> float:
        """
        Weighted composite score in [0, 10].

        Weights (tunable):
        - Length adequacy   : 25%
        - Lexical diversity : 25%
        - Keyword coverage  : 30%
        - Structural clarity: 20%
        """
        length_score = min(word_count / self._max_words, 1.0)
        if word_count < self._min_words:
            length_score *= word_count / self._min_words  # penalise short answers

        # Prefer sentences of 10–20 words; penalise extremes.
        density_score = 1.0 - min(abs(avg_sent_len - 15) / 15, 1.0)

        raw = (
            0.25 * length_score
            + 0.25 * lexical_diversity
            + 0.30 * coverage
            + 0.20 * has_structure
        ) * density_score

        return round(raw * 10, 2)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    @staticmethod
    def _detect_structure(text: str) -> float:
        """Return 1.0 if the text contains lists, bullets, or headers."""
        patterns = [
            r"^\s*[-*•]\s",         # bullet points
            r"^\s*\d+[.)]\s",       # numbered lists
            r"^#+\s",               # markdown headers
            r"\*\*.+?\*\*",         # bold text
        ]
        for pat in patterns:
            if re.search(pat, text, re.MULTILINE):
                return 1.0
        return 0.0

    @staticmethod
    def _lexical_diversity(words: List[str]) -> float:
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    @staticmethod
    def _coverage_score(words: List[str], query: Optional[str]) -> float:
        if not query:
            return 0.5
        query_keywords = set(re.findall(r"\b\w{4,}\b", query.lower()))
        if not query_keywords:
            return 0.5
        word_set = set(words)
        return len(query_keywords & word_set) / len(query_keywords)