"""
core/generator.py
-----------------
Generator module: produces an initial answer from a user query.

Responsibility
--------------
Given a raw user query, construct a generation prompt and return the
LLM's first-pass answer as a plain string.

The Generator deliberately knows nothing about critique or refinement —
it has one job: produce an initial candidate answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import LLMConfig
from models.base_llm import BaseLLM
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class GeneratorOutput:
    """Structured result returned by the Generator."""
    query: str
    answer: str
    prompt_used: str


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    """
    Wraps an LLM to produce an initial answer for a user query.

    Parameters
    ----------
    llm:
        Any ``BaseLLM`` implementation.
    config:
        LLMConfig for sampling parameters (informational; the LLM owns
        the actual call).
    """

    _SYSTEM_PROMPT = (
        "You are a knowledgeable assistant. Answer the user's question "
        "clearly, accurately, and concisely. Prefer structured responses "
        "when the topic benefits from it."
    )

    def __init__(self, llm: BaseLLM, config: LLMConfig) -> None:
        self._llm = llm
        self._config = config

    def generate(self, query: str) -> GeneratorOutput:
        """
        Produce an initial answer for *query*.

        Parameters
        ----------
        query:
            Raw user question or task description.

        Returns
        -------
        GeneratorOutput
            Contains the original query, the generated answer, and the
            exact prompt that was sent to the LLM (useful for debugging).
        """
        prompt = self._build_prompt(query)
        logger.info("Generator: producing initial answer.")
        logger.debug("Generator prompt:\n%s", prompt)

        answer = self._llm.complete(prompt, system_prompt=self._SYSTEM_PROMPT)

        logger.info("Generator: answer produced (%d chars).", len(answer))
        return GeneratorOutput(query=query, answer=answer, prompt_used=prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str) -> str:
        return f"Question: {query}\n\nAnswer:"
