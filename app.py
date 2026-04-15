"""
app.py
------
Entry point for the Self-Correcting LLM pipeline.

Usage
-----
    # Real Ollama models (Ollama must be running):
    python app.py --query "Explain gradient descent"

    # Override which model each stage uses:
    python app.py --generator-model mistral --critic-model llama3 --refiner-model mistral

    # No Ollama? Use deterministic mocks for development:
    python app.py --mock

Architecture
------------
    User Query
        |
        v
    Generator  (--generator-model, default: mistral)
        |
        v
    Critic     (--critic-model,    default: llama3)
        |
        +-- quality OK? --> exit loop --> Final Answer
        |
        v
    Refiner    (--refiner-model,   default: mistral)
        |
        +-----------------------> back to Critic (up to --max-iterations)

    MetricsEvaluator  -- post-run quality report (no LLM calls)

Composition root
----------------
This is the *only* file that knows which LLM backend is in use.
Nothing inside core/ or evaluation/ imports a concrete LLM class.
Swapping backends means changing this file alone.
"""

from __future__ import annotations

import argparse
import sys

from config import LLMConfig, LoopConfig, PipelineConfig
from core.critic import Critic
from core.generator import Generator
from core.loop import RefinementLoop
from core.refiner import Refiner
from evaluation.metrics import MetricsEvaluator
from models.base_llm import MockLLM, OllamaLLM, OllamaError
from utils.logger import get_logger

logger = get_logger(__name__)


def build_pipeline(config: PipelineConfig, use_mock: bool = False) -> RefinementLoop:
    """
    Construct and wire all pipeline components.

    Each stage gets its own LLM instance, allowing generator, critic,
    and refiner to run completely different models simultaneously.

    Parameters
    ----------
    config:
        Full pipeline configuration.  ``config.generator_model``,
        ``config.critic_model``, and ``config.refiner_model`` control
        which Ollama model each stage calls.
    use_mock:
        When True, all stages use ``MockLLM`` (no Ollama required).
        Useful for development and CI.

    Returns
    -------
    RefinementLoop
        Ready-to-run orchestrator.
    """
    if use_mock:
        logger.info("build_pipeline: using MockLLM for all stages.")
        generator_llm = MockLLM(config.llm)
        critic_llm    = MockLLM(config.llm)
        refiner_llm   = MockLLM(config.llm)
    else:
        logger.info(
            "build_pipeline: generator=%s  critic=%s  refiner=%s",
            config.generator_model,
            config.critic_model,
            config.refiner_model,
        )
        generator_llm = OllamaLLM(config.llm, model=config.generator_model)
        critic_llm    = OllamaLLM(config.llm, model=config.critic_model)
        refiner_llm   = OllamaLLM(config.llm, model=config.refiner_model)

    generator = Generator(generator_llm, config.llm)
    critic    = Critic(critic_llm,    config.llm)
    refiner   = Refiner(refiner_llm,  config.llm)

    return RefinementLoop(
        generator=generator,
        critic=critic,
        refiner=refiner,
        config=config.loop,
    )


def run(query: str, config: PipelineConfig, use_mock: bool = False) -> None:
    """
    Run the full pipeline for *query* and print the results.

    Parameters
    ----------
    query:
        User question or task.
    config:
        Pipeline configuration.
    use_mock:
        Route all stages to MockLLM instead of Ollama.
    """
    pipeline = build_pipeline(config, use_mock=use_mock)
    evaluator = MetricsEvaluator()

    # --- Run the loop ---
    try:
        result = pipeline.run(query)
    except OllamaError as exc:
        logger.error("Ollama error: %s", exc)
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        print(
            "\nHint: start Ollama with `ollama serve`, or use --mock for offline testing.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Compute metrics ---
    metrics = evaluator.evaluate_pipeline(result)

    # --- Display results ---
    separator = "-" * 60

    print(f"\n{separator}")
    print("QUERY")
    print(separator)
    print(result.query)

    print(f"\n{separator}")
    print("INITIAL ANSWER")
    print(separator)
    print(result.initial_answer)

    if result.total_iterations > 0:
        print(f"\n{separator}")
        print(f"REFINEMENT ITERATIONS  (ran {result.total_iterations})")
        print(separator)
        for record in result.iterations:
            fb = record.feedback
            delta_str = (
                f"{record.improvement_delta:+.2f}"
                if record.iteration > 1 else "n/a (first)"
            )
            strict_str = " [STRICT MODE]" if record.strict_mode_used else ""
            print(
                f"\n[Iteration {record.iteration}]  "
                f"score={fb.score:.2f}/10 (raw={fb.raw_llm_score:.2f})  "
                f"delta={delta_str}  verdict={fb.verdict}  "
                f"issues={record.issue_count}  "
                f"hallucinations={record.hallucination_count}"
                f"{strict_str}"
            )
            if fb.hallucinations:
                print("  Hallucinations [!]:")
                for h in fb.hallucinations:
                    print(f"    [!] {h}")
            if fb.factual_errors:
                print("  Factual errors [x]:")
                for e in fb.factual_errors:
                    print(f"    [x] {e}")
            if fb.logical_flaws:
                print("  Logical flaws [~]:")
                for f_ in fb.logical_flaws:
                    print(f"    [~] {f_}")
            if fb.missing_concepts:
                print("  Missing concepts [ ]:")
                for m in fb.missing_concepts:
                    print(f"    [ ] {m}")
            if fb.improvement_actions:
                print("  Actions >>:")
                for a in fb.improvement_actions:
                    print(f"    >> {a}")

    print(f"\n{separator}")
    print(
        f"FINAL ANSWER  "
        f"[exit={result.exit_reason.upper()}  converged={result.converged}]"
    )
    print(separator)
    print(result.final_answer)

    print(f"\n{separator}")
    print("METRICS")
    print(separator)
    print(
        f"  Initial composite score : {metrics.initial_metrics.composite_score:.2f} / 10"
    )
    print(
        f"  Final composite score   : {metrics.final_metrics.composite_score:.2f} / 10"
    )
    print(f"  Score delta             : {metrics.score_delta:+.2f}")
    print(f"  Score history (critic)  : {result.score_history}")
    print(f"  Exit reason             : {result.exit_reason}")
    print(f"  Converged               : {result.converged}")
    print(separator)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-Correcting LLM — multi-model critic-refiner pipeline"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Explain how neural networks learn using backpropagation.",
        help="The question to answer.",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="mistral",
        help="Ollama model used by the Generator stage.",
    )
    parser.add_argument(
        "--critic-model",
        type=str,
        default="llama3",
        help="Ollama model used by the Critic stage.",
    )
    parser.add_argument(
        "--refiner-model",
        type=str,
        default="mistral",
        help="Ollama model used by the Refiner stage.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server base URL.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum critic-refiner cycles.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=7.0,
        help="Minimum critic score to accept an answer (0-10).",
    )
    parser.add_argument(
        "--stagnation-patience",
        type=int,
        default=2,
        help="Consecutive low-improvement iterations before early stop.",
    )
    parser.add_argument(
        "--min-improvement-delta",
        type=float,
        default=0.3,
        help="Minimum score gain per iteration to count as progress.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockLLM instead of Ollama (no server required).",
    )
    args = parser.parse_args()

    config = PipelineConfig(
        llm=LLMConfig(
            base_url=args.base_url,
            timeout=args.timeout,
        ),
        loop=LoopConfig(
            max_iterations=args.max_iterations,
            min_quality_score=args.min_score,
            stagnation_patience=args.stagnation_patience,
            min_improvement_delta=args.min_improvement_delta,
        ),
        generator_model=args.generator_model,
        critic_model=args.critic_model,
        refiner_model=args.refiner_model,
    )

    run(args.query, config, use_mock=args.mock)


if __name__ == "__main__":
    main()