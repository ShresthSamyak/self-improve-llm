"""
core/critic.py
--------------
Research-grade critic: evaluates answers with zero leniency.

Design philosophy
-----------------
This is NOT a helpful reviewer — it is an adversarial auditor.
Every claim in the answer is treated as suspicious until verified.
The scoring model penalises hallucinations most heavily, then factual
errors, then logical flaws, then missing concepts.

Penalty model (applied on top of LLM's raw score)
--------------------------------------------------
  hallucinations   : -2.0 per item,  cap -5.0
  factual_errors   : -1.5 per item,  cap -4.0
  logical_flaws    : -1.0 per item,  cap -3.0
  missing_concepts : -0.5 per item,  cap -2.0

The penalised score is what the loop sees — the LLM's self-reported
score is preserved as ``raw_llm_score`` for audit purposes only.

Verdict scale
-------------
  "poor"        score < 4.0   — do not use without heavy revision
  "acceptable"  score < 6.5   — usable but meaningful gaps remain
  "good"        score < 8.5   — correct and complete, minor polish only
  "excellent"   score >= 8.5  — publication-ready
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
    Structured, research-grade output from one critic pass.

    All list fields default to empty — never None — so callers can
    iterate without null-checks.

    Fields
    ------
    factual_errors:
        Specific claims in the answer that are demonstrably wrong.
    hallucinations:
        Claims that are unverifiable, fabricated, or not grounded in
        the query context.  Even *plausible-sounding* unverified claims
        belong here.
    missing_concepts:
        Important concepts the query requires that the answer omits.
    logical_flaws:
        Reasoning errors — non-sequiturs, circular arguments, invalid
        generalisations, unsupported conclusions.
    improvement_actions:
        Concrete, ordered steps the Refiner must take.  Each action
        should map directly to one or more issues above.
    score:
        Penalised quality score, 0–10.  This is the LLM's self-assessed
        score MINUS computed penalties.  Lower is always worse.
    raw_llm_score:
        The score the LLM reported before penalties.  Preserved for
        debugging and drift analysis.
    confidence:
        Critic's self-assessed reliability, 0–1.  Low confidence means
        the critic itself is uncertain — treat the feedback cautiously.
    verdict:
        Qualitative summary: "poor" | "acceptable" | "good" | "excellent"
    raw_response:
        Verbatim LLM output.  Never shown to end users; used for
        debugging and prompt-engineering.
    """
    factual_errors: List[str]
    hallucinations: List[str]
    missing_concepts: List[str]
    logical_flaws: List[str]
    improvement_actions: List[str]
    score: float
    confidence: float
    verdict: str
    raw_llm_score: float = field(repr=False)
    raw_response: str = field(repr=False)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def has_hallucinations(self) -> bool:
        """True if at least one hallucination was identified."""
        return len(self.hallucinations) > 0

    @property
    def has_critical_issues(self) -> bool:
        """True if hallucinations OR factual errors are present."""
        return self.has_hallucinations or len(self.factual_errors) > 0

    @property
    def total_issue_count(self) -> int:
        """Total number of discrete issues across all categories."""
        return (
            len(self.factual_errors)
            + len(self.hallucinations)
            + len(self.logical_flaws)
            + len(self.missing_concepts)
        )


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class Critic:
    """
    Strict, adversarial LLM-based answer evaluator.

    Produces a ``CriticFeedback`` with penalty-adjusted scoring.
    Parsing never raises — a conservative fallback is returned on any
    JSON error so the pipeline always continues.

    Parameters
    ----------
    llm:
        Any ``BaseLLM`` implementation.  For best results use a capable
        model (llama3, mistral-large, gpt-4) that can detect subtle
        hallucinations.
    config:
        LLMConfig (sampling params; model selection lives in the LLM).
    """

    # ------------------------------------------------------------------
    # Penalty weights — tune these to change strictness
    # ------------------------------------------------------------------
    _HALLUCINATION_PENALTY: float = 2.0
    _HALLUCINATION_CAP: float = 5.0
    _FACTUAL_ERROR_PENALTY: float = 1.5
    _FACTUAL_ERROR_CAP: float = 4.0
    _LOGICAL_FLAW_PENALTY: float = 1.0
    _LOGICAL_FLAW_CAP: float = 3.0
    _MISSING_CONCEPT_PENALTY: float = 0.5
    _MISSING_CONCEPT_CAP: float = 2.0

    # ------------------------------------------------------------------
    # Verdict thresholds
    # ------------------------------------------------------------------
    _VERDICT_THRESHOLDS = [
        (8.5, "excellent"),
        (6.5, "good"),
        (4.0, "acceptable"),
        (0.0, "poor"),
    ]

    _VALID_VERDICTS = {"poor", "acceptable", "good", "excellent"}

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    _SYSTEM_PROMPT = (
        "You are a strict academic peer reviewer conducting a rigorous evaluation. "
        "Your only goal is accuracy — NOT encouragement, NOT politeness. "
        "Assume errors exist until you can prove otherwise. "
        "Return ONLY valid JSON. No markdown, no prose, no explanation outside the JSON."
    )

    _EVALUATION_RULES = """\
EVALUATION RULES — follow exactly:
1. CORRECTNESS OVER EVERYTHING. A wrong confident answer is worse than silence.
2. If a claim cannot be verified from the question context alone, mark it as a hallucination.
   Do NOT give benefit of the doubt.
3. Assume the answer author is trying to sound knowledgeable but may be wrong.
   Scrutinise every factual claim individually.
4. Verbosity is NOT quality. Extra words that add no value are a flaw, not a feature.
5. A score of 7 or higher means the answer is GENUINELY correct and complete.
   Do NOT score above 6 if any factual error or hallucination exists.
6. If you are unsure whether a claim is correct — mark it as a potential hallucination.
7. Be critical, not helpful. Your job is to find problems, not to validate the answer."""

    _SCORING_GUIDE = """\
SCORING GUIDE:
 0–2  : Fundamentally wrong, dangerous misinformation, or largely fabricated.
 3–4  : Major factual errors or significant hallucinations present.
 5–6  : Partially correct; notable gaps or unverified claims remain.
 7–8  : Mostly correct; only minor issues that do not mislead.
 9–10 : Exceptional — fully accurate, complete, well-structured, no detectable errors."""

    _RESPONSE_SCHEMA = """\
Respond with ONLY this JSON object — no other text:
{
  "factual_errors":      ["<specific factual claim that is wrong>"],
  "hallucinations":      ["<unverifiable or fabricated claim>"],
  "missing_concepts":    ["<important concept required but absent>"],
  "logical_flaws":       ["<specific reasoning error>"],
  "improvement_actions": ["<concrete action the writer must take>"],
  "score":               <float 0-10>,
  "confidence":          <float 0-1, your certainty in this evaluation>,
  "verdict":             "<poor | acceptable | good | excellent>"
}
Empty lists are valid and expected when no issues of that type exist."""

    def __init__(self, llm: BaseLLM, config: LLMConfig) -> None:
        self._llm = llm
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def critique(self, query: str, answer: str) -> CriticFeedback:
        """
        Rigorously evaluate *answer* in the context of *query*.

        Parameters
        ----------
        query:
            The original user question.
        answer:
            The answer to be evaluated — initial or refined.

        Returns
        -------
        CriticFeedback
            Parsed, penalty-adjusted feedback.  Never raises.
        """
        prompt = self._build_prompt(query, answer)
        logger.info("Critic: starting evaluation.")

        raw = self._llm.complete(prompt, system_prompt=self._SYSTEM_PROMPT)
        logger.debug("Critic raw response:\n%s", raw)

        feedback = self._parse(raw)

        logger.info(
            "Critic: score=%.1f (raw=%.1f)  verdict=%s  "
            "hallucinations=%d  factual_errors=%d  "
            "logical_flaws=%d  missing_concepts=%d",
            feedback.score,
            feedback.raw_llm_score,
            feedback.verdict,
            len(feedback.hallucinations),
            len(feedback.factual_errors),
            len(feedback.logical_flaws),
            len(feedback.missing_concepts),
        )
        if feedback.has_hallucinations:
            logger.warning(
                "Critic: HALLUCINATIONS DETECTED: %s", feedback.hallucinations
            )
        return feedback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, answer: str) -> str:
        return (
            f"{self._EVALUATION_RULES}\n\n"
            f"{self._SCORING_GUIDE}\n\n"
            f"QUERY TO ANSWER:\n{query}\n\n"
            f"ANSWER UNDER REVIEW:\n{answer}\n\n"
            f"{self._RESPONSE_SCHEMA}"
        )

    def _parse(self, raw: str) -> CriticFeedback:
        """
        Parse LLM output into ``CriticFeedback`` with full normalisation.

        Steps:
        1. Strip markdown fences.
        2. Extract the first JSON object found (handles leading prose).
        3. Coerce all fields to expected types.
        4. Apply penalty scoring.
        5. Derive/normalise verdict.
        6. Fall back safely if any step fails.
        """
        cleaned = self._strip_fences(raw)
        extracted = self._extract_json_object(cleaned)

        try:
            data = json.loads(extracted)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Critic: JSON parse failed (%s). Using fallback.", exc)
            return self._fallback_feedback(raw)

        factual_errors   = self._safe_list(data.get("factual_errors"))
        hallucinations   = self._safe_list(data.get("hallucinations"))
        missing_concepts = self._safe_list(data.get("missing_concepts"))
        logical_flaws    = self._safe_list(data.get("logical_flaws"))
        improvement_actions = self._safe_list(data.get("improvement_actions"))

        raw_llm_score = self._safe_float(data.get("score", 5.0), lo=0.0, hi=10.0)
        confidence    = self._safe_float(data.get("confidence", 0.5), lo=0.0, hi=1.0)

        penalised_score = self._apply_penalties(
            raw_llm_score, hallucinations, factual_errors,
            logical_flaws, missing_concepts,
        )

        raw_verdict = str(data.get("verdict", "")).lower().strip()
        verdict = self._normalize_verdict(raw_verdict, penalised_score)

        return CriticFeedback(
            factual_errors=factual_errors,
            hallucinations=hallucinations,
            missing_concepts=missing_concepts,
            logical_flaws=logical_flaws,
            improvement_actions=improvement_actions,
            score=round(penalised_score, 2),
            raw_llm_score=round(raw_llm_score, 2),
            confidence=round(confidence, 3),
            verdict=verdict,
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Penalty scoring
    # ------------------------------------------------------------------

    def _apply_penalties(
        self,
        raw_score: float,
        hallucinations: List[str],
        factual_errors: List[str],
        logical_flaws: List[str],
        missing_concepts: List[str],
    ) -> float:
        """
        Subtract category-specific penalties from the LLM's raw score.

        Penalties are capped per category so a single very wrong answer
        cannot dominate everything else — but the overall score can still
        reach 0.
        """
        penalty = 0.0
        penalty += min(len(hallucinations)   * self._HALLUCINATION_PENALTY,   self._HALLUCINATION_CAP)
        penalty += min(len(factual_errors)   * self._FACTUAL_ERROR_PENALTY,   self._FACTUAL_ERROR_CAP)
        penalty += min(len(logical_flaws)    * self._LOGICAL_FLAW_PENALTY,    self._LOGICAL_FLAW_CAP)
        penalty += min(len(missing_concepts) * self._MISSING_CONCEPT_PENALTY, self._MISSING_CONCEPT_CAP)
        return max(0.0, raw_score - penalty)

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _normalize_verdict(self, verdict: str, score: float) -> str:
        """
        Return a valid verdict string.

        If the LLM returns a known string, use it — but cross-check
        against the score.  If the LLM claims "excellent" but the
        penalised score is 3.0, override to "poor".  This prevents the
        model from self-reporting a verdict that contradicts the score.
        """
        # Map legacy / variant strings to canonical names
        _LEGACY_MAP = {
            "needs_improvement": "poor",
            "needs improvement": "poor",
            "unacceptable":      "poor",
            "bad":               "poor",
            "great":             "good",
            "perfect":           "excellent",
        }
        if verdict in _LEGACY_MAP:
            verdict = _LEGACY_MAP[verdict]

        if verdict not in self._VALID_VERDICTS:
            # Derive purely from score
            return self._verdict_from_score(score)

        # Cross-check: score and verdict must be consistent
        score_derived = self._verdict_from_score(score)
        _RANK = {"poor": 0, "acceptable": 1, "good": 2, "excellent": 3}
        if abs(_RANK[verdict] - _RANK[score_derived]) > 1:
            # More than one tier apart — trust the score
            logger.debug(
                "Critic: verdict '%s' inconsistent with score %.1f — overriding to '%s'.",
                verdict, score, score_derived,
            )
            return score_derived

        return verdict

    @classmethod
    def _verdict_from_score(cls, score: float) -> str:
        for threshold, label in cls._VERDICT_THRESHOLDS:
            if score >= threshold:
                return label
        return "poor"

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove ```json ... ``` or ``` ... ``` wrappers."""
        return re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text).strip()

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """
        Return the first {...} block found in *text*.

        Handles cases where the LLM prepends a sentence before the JSON
        despite being told not to.
        """
        match = re.search(r"\{[\s\S]*\}", text)
        return match.group(0) if match else text

    @staticmethod
    def _safe_list(value: object) -> List[str]:
        """
        Coerce *value* to a list of non-empty strings.

        Handles: None, str, list of mixed types, nested lists.
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    result.append(item.strip())
                elif isinstance(item, (int, float)):
                    result.append(str(item))
            return result
        return []

    @staticmethod
    def _safe_float(value: object, lo: float = 0.0, hi: float = 10.0) -> float:
        """Parse *value* as float and clamp to [lo, hi]."""
        try:
            return max(lo, min(hi, float(value)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return (lo + hi) / 2.0

    @staticmethod
    def _fallback_feedback(raw: str) -> CriticFeedback:
        """
        Conservative fallback returned when JSON parsing fails entirely.

        Scores the answer as poor-but-unknown so the loop refines once
        before giving up.
        """
        return CriticFeedback(
            factual_errors=[],
            hallucinations=[],
            missing_concepts=[],
            logical_flaws=[],
            improvement_actions=[
                "Critic failed to parse structured feedback — rewrite answer "
                "with clearer, more precise language."
            ],
            score=3.0,
            raw_llm_score=3.0,
            confidence=0.2,
            verdict="poor",
            raw_response=raw,
        )
