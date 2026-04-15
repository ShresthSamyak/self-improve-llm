"""
tests/test_critic.py
--------------------
Unit tests for core/critic.py — covers the upgraded research-grade schema.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import LLMConfig
from core.critic import Critic, CriticFeedback
from models.base_llm import BaseLLM, MockLLM


# ---------------------------------------------------------------------------
# LLM stubs
# ---------------------------------------------------------------------------

def _make_llm_response(**kwargs) -> str:
    """Build a minimal valid critic JSON string."""
    defaults = {
        "factual_errors": [],
        "hallucinations": [],
        "missing_concepts": [],
        "logical_flaws": [],
        "improvement_actions": [],
        "score": 8.0,
        "confidence": 0.90,
        "verdict": "good",
    }
    defaults.update(kwargs)
    return json.dumps(defaults)


class _StaticLLM(BaseLLM):
    """Returns a fixed JSON string on every call."""
    def __init__(self, response: str):
        super().__init__(LLMConfig())
        self._response = response

    def complete(self, prompt, system_prompt=None):
        return self._response


class _MalformedLLM(BaseLLM):
    def complete(self, prompt, system_prompt=None):
        return "This is not JSON at all ¯\\_(ツ)_/¯"


# ---------------------------------------------------------------------------
# Schema and parsing tests
# ---------------------------------------------------------------------------

def test_feedback_parses_all_fields():
    response = _make_llm_response(
        factual_errors=["X is wrong"],
        hallucinations=["Y was fabricated"],
        missing_concepts=["Z not covered"],
        logical_flaws=["Non-sequitur in para 2"],
        improvement_actions=["Fix X", "Remove Y"],
        score=6.0,
        confidence=0.80,
        verdict="acceptable",
    )
    critic = Critic(_StaticLLM(response), LLMConfig())
    fb = critic.critique("Q?", "A")

    assert fb.factual_errors   == ["X is wrong"]
    assert fb.hallucinations   == ["Y was fabricated"]
    assert fb.missing_concepts == ["Z not covered"]
    assert fb.logical_flaws    == ["Non-sequitur in para 2"]
    assert fb.improvement_actions == ["Fix X", "Remove Y"]
    assert fb.confidence == 0.80
    assert isinstance(fb.raw_response, str)


def test_fallback_on_malformed_json():
    critic = Critic(_MalformedLLM(LLMConfig()), LLMConfig())
    fb = critic.critique("Q?", "A")

    assert isinstance(fb, CriticFeedback)
    assert fb.verdict == "poor"
    assert fb.score < 5.0
    assert fb.confidence < 0.3
    assert len(fb.improvement_actions) > 0  # fallback always has an action


def test_markdown_fences_stripped():
    fenced = '```json\n' + _make_llm_response(score=7.5, verdict="good") + '\n```'
    critic = Critic(_StaticLLM(fenced), LLMConfig())
    fb = critic.critique("Q?", "A")

    assert fb.score > 0
    assert fb.verdict in ("poor", "acceptable", "good", "excellent")


def test_json_with_leading_prose():
    """LLMs sometimes add a sentence before the JSON block."""
    prose_prefix = "Here is my evaluation:\n"
    response = prose_prefix + _make_llm_response(score=7.0, verdict="good")
    critic = Critic(_StaticLLM(response), LLMConfig())
    fb = critic.critique("Q?", "A")

    assert fb.verdict in ("good", "excellent")


# ---------------------------------------------------------------------------
# Penalty model tests
# ---------------------------------------------------------------------------

def test_hallucination_applies_heavy_penalty():
    # 2 hallucinations → 2 * 2.0 = 4.0 penalty on raw score of 8.0
    response = _make_llm_response(
        hallucinations=["H1", "H2"],
        score=8.0,
        verdict="good",
    )
    critic = Critic(_StaticLLM(response), LLMConfig())
    fb = critic.critique("Q?", "A")

    assert fb.raw_llm_score == 8.0
    assert fb.score == pytest_approx(4.0, abs=0.05)
    assert fb.has_hallucinations is True
    assert fb.has_critical_issues is True


def test_factual_error_penalty():
    # 2 factual errors → 2 * 1.5 = 3.0 penalty on raw 7.5
    response = _make_llm_response(factual_errors=["F1", "F2"], score=7.5)
    critic = Critic(_StaticLLM(response), LLMConfig())
    fb = critic.critique("Q?", "A")

    assert fb.raw_llm_score == 7.5
    assert fb.score == pytest_approx(4.5, abs=0.05)


def test_penalties_cannot_go_below_zero():
    # Extreme case: many issues
    response = _make_llm_response(
        hallucinations=["H1", "H2", "H3", "H4"],
        factual_errors=["F1", "F2", "F3"],
        logical_flaws=["L1", "L2", "L3"],
        missing_concepts=["M1", "M2", "M3", "M4"],
        score=3.0,
    )
    critic = Critic(_StaticLLM(response), LLMConfig())
    fb = critic.critique("Q?", "A")

    assert fb.score >= 0.0


def test_no_issues_no_penalty():
    response = _make_llm_response(score=9.0, verdict="excellent")
    critic = Critic(_StaticLLM(response), LLMConfig())
    fb = critic.critique("Q?", "A")

    assert fb.score == fb.raw_llm_score == 9.0


# ---------------------------------------------------------------------------
# Verdict normalisation tests
# ---------------------------------------------------------------------------

def test_verdict_derived_from_score_when_unknown():
    response = _make_llm_response(score=8.0, verdict="UNKNOWN_VERDICT")
    critic = Critic(_StaticLLM(response), LLMConfig())
    fb = critic.critique("Q?", "A")

    assert fb.verdict in ("poor", "acceptable", "good", "excellent")


def test_legacy_verdict_mapped():
    """Old 'needs_improvement' verdict must map to 'poor'."""
    response = _make_llm_response(score=3.5, verdict="needs_improvement")
    critic = Critic(_StaticLLM(response), LLMConfig())
    fb = critic.critique("Q?", "A")

    assert fb.verdict == "poor"


def test_inconsistent_verdict_overridden_by_score():
    # LLM claims "excellent" but score is 2.5 → override to "poor"
    response = _make_llm_response(score=2.5, verdict="excellent")
    critic = Critic(_StaticLLM(response), LLMConfig())
    fb = critic.critique("Q?", "A")

    assert fb.verdict == "poor"


# ---------------------------------------------------------------------------
# CriticFeedback properties tests
# ---------------------------------------------------------------------------

def test_has_hallucinations_property():
    response = _make_llm_response(hallucinations=["fabricated claim"])
    critic = Critic(_StaticLLM(response), LLMConfig())
    fb = critic.critique("Q?", "A")
    assert fb.has_hallucinations is True


def test_has_no_hallucinations_when_empty():
    response = _make_llm_response(hallucinations=[])
    critic = Critic(_StaticLLM(response), LLMConfig())
    fb = critic.critique("Q?", "A")
    assert fb.has_hallucinations is False


def test_total_issue_count():
    response = _make_llm_response(
        factual_errors=["F1"],
        hallucinations=["H1", "H2"],
        logical_flaws=["L1"],
        missing_concepts=["M1", "M2", "M3"],
    )
    critic = Critic(_StaticLLM(response), LLMConfig())
    fb = critic.critique("Q?", "A")
    assert fb.total_issue_count == 7


# ---------------------------------------------------------------------------
# MockLLM integration test (verifies canned responses match new schema)
# ---------------------------------------------------------------------------

def test_mock_llm_alternates_verdicts():
    llm = MockLLM(LLMConfig())
    critic = Critic(llm, LLMConfig())

    fb1 = critic.critique("Q?", "A")   # call 1 → poor (with hallucination)
    fb2 = critic.critique("Q?", "A")   # call 2 → good (clean)

    assert fb1.verdict == "poor"
    assert fb1.has_hallucinations is True

    assert fb2.verdict == "good"
    assert fb2.has_hallucinations is False


def test_mock_llm_score_clamped():
    llm = MockLLM(LLMConfig())
    critic = Critic(llm, LLMConfig())
    fb = critic.critique("Q?", "A")

    assert 0.0 <= fb.score <= 10.0
    assert 0.0 <= fb.confidence <= 1.0


# ---------------------------------------------------------------------------
# Minimal pytest-approx shim (no pytest dependency required)
# ---------------------------------------------------------------------------

class pytest_approx:
    def __init__(self, expected, abs=0.01):
        self.expected = expected
        self.abs = abs

    def __eq__(self, other):
        return abs(other - self.expected) <= self.abs

    def __repr__(self):
        return f"approx({self.expected} ± {self.abs})"


if __name__ == "__main__":
    test_feedback_parses_all_fields()
    test_fallback_on_malformed_json()
    test_markdown_fences_stripped()
    test_json_with_leading_prose()
    test_hallucination_applies_heavy_penalty()
    test_factual_error_penalty()
    test_penalties_cannot_go_below_zero()
    test_no_issues_no_penalty()
    test_verdict_derived_from_score_when_unknown()
    test_legacy_verdict_mapped()
    test_inconsistent_verdict_overridden_by_score()
    test_has_hallucinations_property()
    test_has_no_hallucinations_when_empty()
    test_total_issue_count()
    test_mock_llm_alternates_verdicts()
    test_mock_llm_score_clamped()
    print("All critic tests passed.")
