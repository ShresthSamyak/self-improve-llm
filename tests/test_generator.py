"""
tests/test_generator.py
-----------------------
Unit tests for core/generator.py.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import LLMConfig
from core.generator import Generator
from models.base_llm import MockLLM


def test_generator_returns_output():
    llm = MockLLM(LLMConfig())
    gen = Generator(llm, LLMConfig())
    output = gen.generate("What is machine learning?")

    assert output.query == "What is machine learning?"
    assert isinstance(output.answer, str)
    assert len(output.answer) > 0


def test_generator_prompt_contains_query():
    llm = MockLLM(LLMConfig())
    gen = Generator(llm, LLMConfig())
    output = gen.generate("Explain transformers.")

    assert "Explain transformers." in output.prompt_used


def test_generator_preserves_query():
    llm = MockLLM(LLMConfig())
    gen = Generator(llm, LLMConfig())
    query = "How does reinforcement learning work?"
    output = gen.generate(query)

    assert output.query == query


if __name__ == "__main__":
    test_generator_returns_output()
    test_generator_prompt_contains_query()
    test_generator_preserves_query()
    print("All generator tests passed.")