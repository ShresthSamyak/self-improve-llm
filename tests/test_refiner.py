"""
tests/test_refiner.py
---------------------
Unit tests for core/refiner.py.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import LLMConfig
from core.critic import CriticFeedback
from core.refiner import Refiner, RefinerOutput
from models.base_llm import MockLLM


def _make_feedback(score: float = 5.0) -> CriticFeedback:
    return CriticFeedback(
        factual_errors=["The mechanism described is inaccurate."],
        hallucinations=["Referenced a non-existent theorem."],
        missing_concepts=["Does not address edge cases."],
        logical_flaws=["Conclusion does not follow from the premise."],
        improvement_actions=[
            "Remove or verify the theorem reference.",
            "Correct the mechanism description.",
            "Add a section on edge cases.",
        ],
        score=score,
        raw_llm_score=score + 1.5,   # penalty was already applied
        confidence=0.72,
        verdict="poor",
        raw_response="{}",
    )


def test_refiner_returns_output():
    llm = MockLLM(LLMConfig())
    refiner = Refiner(llm, LLMConfig())
    output = refiner.refine(
        query="What is ML?",
        answer="ML is machine learning.",
        feedback=_make_feedback(),
        iteration=1,
    )

    assert isinstance(output, RefinerOutput)
    assert isinstance(output.refined_answer, str)
    assert len(output.refined_answer) > 0


def test_refiner_preserves_provenance():
    llm = MockLLM(LLMConfig())
    refiner = Refiner(llm, LLMConfig())
    original = "ML is machine learning."
    feedback = _make_feedback()

    output = refiner.refine(
        query="What is ML?",
        answer=original,
        feedback=feedback,
        iteration=2,
    )

    assert output.previous_answer == original
    assert output.feedback_applied is feedback
    assert output.iteration == 2


def test_refiner_prompt_includes_all_issue_categories():
    """Verify the prompt surfaces hallucinations, factual errors, and actions."""
    captured_prompts = []

    class _CaptureLLM(MockLLM):
        def complete(self, prompt, system_prompt=None):
            captured_prompts.append(prompt)
            return super().complete(prompt, system_prompt)

    llm = _CaptureLLM(LLMConfig())
    refiner = Refiner(llm, LLMConfig())
    feedback = _make_feedback()

    refiner.refine("Q?", "A", feedback)

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]

    # Hallucinations must appear prominently (they're highest priority)
    assert "Referenced a non-existent theorem." in prompt
    # Factual errors must appear
    assert "The mechanism described is inaccurate." in prompt
    # Improvement actions must appear
    assert "Remove or verify the theorem reference." in prompt


def test_refiner_prompt_includes_score_and_verdict():
    """Score and verdict must appear so the refiner knows severity."""
    captured_prompts = []

    class _CaptureLLM(MockLLM):
        def complete(self, prompt, system_prompt=None):
            captured_prompts.append(prompt)
            return super().complete(prompt, system_prompt)

    llm = _CaptureLLM(LLMConfig())
    refiner = Refiner(llm, LLMConfig())
    refiner.refine("Q?", "A", _make_feedback(score=3.5))

    prompt = captured_prompts[0]
    assert "3.5" in prompt
    assert "poor" in prompt


if __name__ == "__main__":
    test_refiner_returns_output()
    test_refiner_preserves_provenance()
    test_refiner_prompt_includes_all_issue_categories()
    test_refiner_prompt_includes_score_and_verdict()
    print("All refiner tests passed.")
