"""
tests/test_loop.py
------------------
Unit tests for the adaptive RefinementLoop.

Covers: stagnation detection, strict mode escalation, exit reasons,
score history, and the IterationRecord metadata fields.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import LLMConfig, LoopConfig
from core.critic import Critic
from core.generator import Generator
from core.loop import RefinementLoop, StagnationTracker
from core.refiner import Refiner
from models.base_llm import BaseLLM, MockLLM


# ---------------------------------------------------------------------------
# Minimal LLM stubs
# ---------------------------------------------------------------------------

class _ScriptedLLM(BaseLLM):
    """
    Returns a sequence of scripted responses in order.
    Repeats the last response indefinitely once the script is exhausted.
    """
    def __init__(self, responses: list):
        super().__init__(LLMConfig())
        self._responses = responses
        self._idx = 0

    def complete(self, prompt, system_prompt=None):
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return resp


def _critic_json(score: float, hallucinations=None, factual_errors=None,
                 verdict=None) -> str:
    if verdict is None:
        verdict = "good" if score >= 7.0 else ("acceptable" if score >= 4.0 else "poor")
    return json.dumps({
        "factual_errors":      factual_errors or [],
        "hallucinations":      hallucinations or [],
        "missing_concepts":    [],
        "logical_flaws":       [],
        "improvement_actions": [],
        "score":               score,
        "confidence":          0.85,
        "verdict":             verdict,
    })


def _make_loop(
    generator_responses: list,
    critic_responses: list,
    refiner_responses: list,
    loop_config: LoopConfig = None,
) -> RefinementLoop:
    if loop_config is None:
        loop_config = LoopConfig(
            max_iterations=5,
            min_quality_score=7.0,
            stagnation_patience=2,
            min_improvement_delta=0.3,
        )
    generator_llm = _ScriptedLLM(generator_responses)
    critic_llm    = _ScriptedLLM(critic_responses)
    refiner_llm   = _ScriptedLLM(refiner_responses)

    return RefinementLoop(
        generator=Generator(generator_llm, LLMConfig()),
        critic=Critic(critic_llm, LLMConfig()),
        refiner=Refiner(refiner_llm, LLMConfig()),
        config=loop_config,
    )


# ---------------------------------------------------------------------------
# StagnationTracker unit tests
# ---------------------------------------------------------------------------

def test_stagnation_not_triggered_before_patience():
    t = StagnationTracker(patience=2, min_delta=0.3)
    t.record(0.1)           # only 1 delta — not enough
    assert t.is_stagnated() is False


def test_stagnation_triggered_after_patience():
    t = StagnationTracker(patience=2, min_delta=0.3)
    t.record(0.1)
    t.record(0.1)
    assert t.is_stagnated() is True


def test_stagnation_reset_by_good_improvement():
    t = StagnationTracker(patience=2, min_delta=0.3)
    t.record(0.1)
    t.record(0.5)           # 0.5 >= 0.3 → window now [0.1, 0.5]
    assert t.is_stagnated() is False


def test_stagnation_exact_threshold_not_triggered():
    # delta == min_delta is NOT stagnation (condition is strict <)
    t = StagnationTracker(patience=2, min_delta=0.3)
    t.record(0.3)
    t.record(0.3)
    assert t.is_stagnated() is False


def test_stagnation_negative_delta_triggers():
    t = StagnationTracker(patience=2, min_delta=0.3)
    t.record(-0.5)          # score went DOWN
    t.record(-0.2)
    assert t.is_stagnated() is True


# ---------------------------------------------------------------------------
# Loop exit reason tests
# ---------------------------------------------------------------------------

def test_exit_converged_when_quality_met():
    loop = _make_loop(
        generator_responses=["Initial answer."],
        critic_responses=[_critic_json(8.5, verdict="good")],
        refiner_responses=["Refined."],
    )
    result = loop.run("What is ML?")
    assert result.exit_reason == "converged"
    assert result.converged is True


def test_exit_stagnated_when_no_improvement():
    # Score stays flat (0.0 delta twice → stagnation with patience=2)
    loop = _make_loop(
        generator_responses=["Answer."],
        critic_responses=[
            _critic_json(5.0),   # iter 1
            _critic_json(5.0),   # iter 2  delta=0.0
            _critic_json(5.0),   # iter 3  delta=0.0 → stagnated
        ],
        refiner_responses=["Refined."],
        loop_config=LoopConfig(
            max_iterations=5,
            min_quality_score=7.0,
            stagnation_patience=2,
            min_improvement_delta=0.3,
        ),
    )
    result = loop.run("Q?")
    assert result.exit_reason == "stagnated"
    assert result.converged is False


def test_exit_exhausted_when_max_iters_hit():
    # Score always improves slightly so no stagnation, never reaches quality
    loop = _make_loop(
        generator_responses=["Answer."],
        critic_responses=[_critic_json(s) for s in [4.0, 4.5, 5.0]],
        refiner_responses=["Refined."],
        loop_config=LoopConfig(max_iterations=3, stagnation_patience=5),
    )
    result = loop.run("Q?")
    assert result.exit_reason == "exhausted"
    assert result.total_iterations == 3


# ---------------------------------------------------------------------------
# IterationRecord metadata tests
# ---------------------------------------------------------------------------

def test_iteration_record_delta_is_zero_for_first():
    loop = _make_loop(
        generator_responses=["Answer."],
        critic_responses=[_critic_json(5.0), _critic_json(8.5, verdict="good")],
        refiner_responses=["Refined."],
    )
    result = loop.run("Q?")
    assert result.iterations[0].improvement_delta == 0.0


def test_iteration_record_delta_computed_correctly():
    # iter1 score=4.0, iter2 score=6.5  → delta=+2.5
    loop = _make_loop(
        generator_responses=["Answer."],
        critic_responses=[
            _critic_json(4.0),
            _critic_json(6.5),
            _critic_json(8.5, verdict="good"),
        ],
        refiner_responses=["R1", "R2"],
    )
    result = loop.run("Q?")
    assert result.iterations[1].improvement_delta == pytest_approx(2.5, abs=0.01)


def test_score_history_populated():
    loop = _make_loop(
        generator_responses=["Answer."],
        critic_responses=[_critic_json(4.0), _critic_json(8.5, verdict="good")],
        refiner_responses=["Refined."],
    )
    result = loop.run("Q?")
    assert len(result.score_history) == result.total_iterations
    assert result.score_history[0] == pytest_approx(4.0, abs=0.5)


def test_iteration_record_issue_count():
    critic_resp = json.dumps({
        "factual_errors":      ["E1"],
        "hallucinations":      ["H1", "H2"],
        "missing_concepts":    ["M1"],
        "logical_flaws":       [],
        "improvement_actions": [],
        "score":               4.0,
        "confidence":          0.7,
        "verdict":             "poor",
    })
    loop = _make_loop(
        generator_responses=["Answer."],
        # first pass has issues, second is clean
        critic_responses=[critic_resp, _critic_json(8.5, verdict="good")],
        refiner_responses=["Refined."],
    )
    result = loop.run("Q?")
    assert result.iterations[0].issue_count == 4   # 1 + 2 + 1 + 0
    assert result.iterations[0].hallucination_count == 2


# ---------------------------------------------------------------------------
# Strict mode escalation tests
# ---------------------------------------------------------------------------

def test_strict_mode_not_used_on_first_iteration():
    loop = _make_loop(
        generator_responses=["Answer."],
        critic_responses=[
            _critic_json(4.0, hallucinations=["H1"]),
            _critic_json(8.5, verdict="good"),
        ],
        refiner_responses=["Refined."],
    )
    result = loop.run("Q?")
    # First iteration: hallucination present but no prior history → no strict mode
    assert result.iterations[0].strict_mode_used is False


def test_strict_mode_activated_when_hallucinations_persist():
    """
    If hallucinations are present in iteration 1 AND iteration 2,
    the refiner call after iteration 2 should use strict mode.
    """
    loop = _make_loop(
        generator_responses=["Answer."],
        critic_responses=[
            _critic_json(3.0, hallucinations=["H1"]),   # iter 1 — hallucination
            _critic_json(3.5, hallucinations=["H1"]),   # iter 2 — still there
            _critic_json(8.5, verdict="good"),           # iter 3 — clean
        ],
        refiner_responses=["R1", "R2 strict"],
    )
    result = loop.run("Q?")
    # After iteration 2 the strict mode flag should be set
    assert result.iterations[1].strict_mode_used is True


def test_strict_mode_cleared_when_hallucinations_resolved():
    """Once hallucinations clear, the refiner goes back to normal mode."""
    loop = _make_loop(
        generator_responses=["Answer."],
        critic_responses=[
            _critic_json(3.0, hallucinations=["H1"]),   # iter 1 — hallucination
            _critic_json(6.5),                           # iter 2 — clean (no halluc)
            _critic_json(8.5, verdict="good"),           # iter 3 — exit
        ],
        refiner_responses=["R1", "R2"],
    )
    result = loop.run("Q?")
    # Iteration 1 had hallucinations but iter 2 did not → no strict mode after iter 2
    assert result.iterations[0].strict_mode_used is False   # no prior history
    assert result.iterations[1].strict_mode_used is False   # hallucination cleared


# ---------------------------------------------------------------------------
# MockLLM integration (end-to-end smoke test)
# ---------------------------------------------------------------------------

def test_mock_llm_end_to_end():
    """Full pipeline run — just confirm it completes without exceptions."""
    from models.base_llm import MockLLM

    llm = MockLLM(LLMConfig())
    loop = RefinementLoop(
        generator=Generator(llm, LLMConfig()),
        critic=Critic(llm, LLMConfig()),
        refiner=Refiner(llm, LLMConfig()),
        config=LoopConfig(max_iterations=3),
    )
    result = loop.run("What is backpropagation?")

    assert result.exit_reason in ("converged", "stagnated", "exhausted")
    assert len(result.score_history) == result.total_iterations
    assert result.final_answer
    assert len(result.iterations) == result.total_iterations


# ---------------------------------------------------------------------------
# pytest_approx shim
# ---------------------------------------------------------------------------

class pytest_approx:
    def __init__(self, expected, abs=0.01):
        self.expected = expected
        self.abs = abs

    def __eq__(self, other):
        return abs(other - self.expected) <= self.abs

    def __repr__(self):
        return f"approx({self.expected} +/- {self.abs})"


if __name__ == "__main__":
    # StagnationTracker
    test_stagnation_not_triggered_before_patience()
    test_stagnation_triggered_after_patience()
    test_stagnation_reset_by_good_improvement()
    test_stagnation_exact_threshold_not_triggered()
    test_stagnation_negative_delta_triggers()
    # Exit reasons
    test_exit_converged_when_quality_met()
    test_exit_stagnated_when_no_improvement()
    test_exit_exhausted_when_max_iters_hit()
    # Metadata
    test_iteration_record_delta_is_zero_for_first()
    test_iteration_record_delta_computed_correctly()
    test_score_history_populated()
    test_iteration_record_issue_count()
    # Strict mode
    test_strict_mode_not_used_on_first_iteration()
    test_strict_mode_activated_when_hallucinations_persist()
    test_strict_mode_cleared_when_hallucinations_resolved()
    # Integration
    test_mock_llm_end_to_end()
    print("All loop tests passed.")
