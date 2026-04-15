"""
tests/test_critic.py
--------------------
Unit tests for core/critic.py.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import LLMConfig
from core.critic import Critic, CriticFeedback
from models.base_llm import BaseLLM, MockLLM


class _AlwaysGoodLLM(BaseLLM):
    """Stub that returns a perfect score every time."""
    def complete(self, prompt, system_prompt=None):
        return json.dumps({
            "score": 9.5,
            "confidence": 0.97,
            "issues": [],
            "suggestions": [],
            "verdict": "acceptable",
        })


class _MalformedLLM(BaseLLM):
    """Stub that returns broken JSON."""
    def complete(self, prompt, system_prompt=None):
        return "This is not JSON at all ¯\\_(ツ)_/¯"


def test_critic_parses_valid_json():
    llm = _AlwaysGoodLLM(LLMConfig())
    critic = Critic(llm, LLMConfig())
    feedback = critic.critique("What is AI?", "AI is artificial intelligence.")

    assert isinstance(feedback, CriticFeedback)
    assert feedback.score == 9.5
    assert feedback.verdict == "acceptable"
    assert feedback.issues == []


def test_critic_fallback_on_bad_json():
    llm = _MalformedLLM(LLMConfig())
    critic = Critic(llm, LLMConfig())
    feedback = critic.critique("What is AI?", "Some answer.")

    # Should not raise — fallback is returned instead.
    assert isinstance(feedback, CriticFeedback)
    assert feedback.verdict == "needs_improvement"
    assert feedback.score < 5.0


def test_critic_strips_markdown_fences():
    """LLMs often wrap JSON in ```json``` — ensure we handle it."""

    class _FencedLLM(BaseLLM):
        def complete(self, prompt, system_prompt=None):
            return '```json\n{"score": 7.0, "confidence": 0.8, "issues": [], "suggestions": [], "verdict": "acceptable"}\n```'

    llm = _FencedLLM(LLMConfig())
    critic = Critic(llm, LLMConfig())
    feedback = critic.critique("Test?", "Test answer.")

    assert feedback.score == 7.0
    assert feedback.verdict == "acceptable"


def test_mock_llm_critic_alternates():
    llm = MockLLM(LLMConfig())
    critic = Critic(llm, LLMConfig())

    fb1 = critic.critique("Q?", "A")   # call 1 → needs_improvement
    fb2 = critic.critique("Q?", "A")   # call 2 → acceptable

    assert fb1.verdict == "needs_improvement"
    assert fb2.verdict == "acceptable"


if __name__ == "__main__":
    test_critic_parses_valid_json()
    test_critic_fallback_on_bad_json()
    test_critic_strips_markdown_fences()
    test_mock_llm_critic_alternates()
    print("All critic tests passed.")
