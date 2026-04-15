"""
config.py
---------
Central configuration for the Self-Correcting LLM pipeline.

All tuneable knobs live here. Import `PipelineConfig` and override
fields as needed — no magic env vars, no hidden globals.
"""

from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """
    Shared sampling parameters forwarded to every LLM call.

    These act as pipeline-wide defaults.  Individual components can
    override ``model_name`` by passing a different value to their LLM
    constructor — see ``OllamaLLM(config, model=...)``.
    """
    model_name: str = "mistral:7b"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    # Ollama server location — override if running on a remote host.
    base_url: str = "http://localhost:11434"
    # Per-request timeout in seconds.  Raise if you use large models.
    timeout: int = 120


@dataclass
class LoopConfig:
    """
    Controls the adaptive critic-refiner iteration loop.

    Quality gate
    ------------
    The loop exits with ``exit_reason="converged"`` only when ALL of:
      - zero hallucinations
      - zero factual errors
      - score >= min_quality_score
      - verdict is "good" or "excellent"

    Stagnation detection
    --------------------
    If the score improvement over the last ``stagnation_patience``
    consecutive iterations is all below ``min_improvement_delta``,
    the loop exits early with ``exit_reason="stagnated"`` rather than
    wasting iterations on a plateaued answer.

    Adaptive strict mode
    --------------------
    When hallucinations persist across two consecutive iterations, the
    Refiner is automatically escalated to strict mode for the next pass.
    """
    max_iterations: int = 3
    # Minimum score the critic must assign before the loop exits.
    min_quality_score: float = 7.0           # out of 10
    # Stagnation: how many consecutive low-delta iterations trigger early stop.
    stagnation_patience: int = 2
    # Minimum score improvement to count as meaningful progress.
    min_improvement_delta: float = 0.3


@dataclass
class PipelineConfig:
    """
    Top-level config passed through the entire pipeline.

    ``llm`` holds shared sampling parameters (temperature, max_tokens,
    base_url, timeout).  The three model-name fields let you route each
    pipeline stage to a different Ollama model without duplicating the
    rest of the config.
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    # Per-component model selection — each can be any model available in Ollama.
    generator_model: str = "mistral:7b"
    critic_model: str = "llama3.1:8b"
    refiner_model: str = "mistral:7b"
    verbose: bool = True


def get_default_config() -> PipelineConfig:
    """Return a PipelineConfig populated with sensible defaults."""
    return PipelineConfig()