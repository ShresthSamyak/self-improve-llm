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
        score=score,
        confidence=0.6,
        issues=["Too vague.", "No examples."],
        suggestions=["Add a concrete example.", "Use numbered steps."],
        verdict="needs_improvement",
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


def test_refiner_prompt_includes_issues():
    """Verify the prompt forwards critic issues to the LLM."""
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
    assert "Too vague." in captured_prompts[0]
    assert "Add a concrete example." in captured_prompts[0]


if __name__ == "__main__":
    test_refiner_returns_output()
    test_refiner_preserves_provenance()
    test_refiner_prompt_includes_issues()
    print("All refiner tests passed.")
