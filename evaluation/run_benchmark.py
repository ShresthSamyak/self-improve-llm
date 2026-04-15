"""
evaluation/run_benchmark.py
----------------------------
Benchmarks three pipeline configurations head-to-head:

  System A — Baseline
    Generator only.  No critic, no refinement.  Raw first-pass answer.

  System B — LLM Critic Loop
    Generator + full LLM-based Critic (research-grade adversarial prompt)
    + Refiner + adaptive loop with stagnation detection.

  System C — Learned Critic Loop
    Generator + LearnedCritic (fine-tuned MiniLM, fast local inference)
    with LLM fallback for answers below the confidence threshold
    + same Refiner + adaptive loop.

Metrics (per query, aggregated across the dataset)
----------------------------------------------------
  factual_accuracy   : LLM-as-judge score (0–10) on the final answer.
                       Uses a dedicated judge prompt focused only on facts.
  hallucination_rate : Fraction of queries whose final answer has >= 1
                       hallucination as reported by the judge.
  avg_critic_score   : Mean penalised critic score at final iteration
                       (from the pipeline's own internal critic).
  avg_iterations     : Mean refinement cycles consumed.
  avg_latency_s      : Mean wall-clock seconds per query (end-to-end).

Output
------
  Console   : Formatted comparison table with absolute values and
              % improvement of B/C vs A (baseline).
  JSON file : Full per-query results + summaries saved to
              evaluation/results/benchmark_<timestamp>.json

Usage
-----
  # offline / CI (no Ollama required):
  python evaluation/run_benchmark.py --mock

  # real Ollama models:
  python evaluation/run_benchmark.py \\
      --generator-model mistral \\
      --critic-model llama3 \\
      --refiner-model mistral

  # specify held-out query file:
  python evaluation/run_benchmark.py --queries-file data/eval_queries.txt --mock

  # skip Learned Critic (no weights / no torch):
  python evaluation/run_benchmark.py --skip-learned --mock
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Make project root importable when run as a script
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import LLMConfig, LoopConfig, PipelineConfig
from core.critic import Critic, CriticFeedback
from core.generator import Generator
from core.loop import RefinementLoop
from core.refiner import Refiner
from models.base_llm import BaseLLM, MockLLM, OllamaLLM, OllamaError
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Held-out evaluation queries
# ---------------------------------------------------------------------------
# These are intentionally different from the training queries used in
# generate_dataset.py (backprop, supervised/unsupervised, CNNs, transformers).
# Sourced to cover breadth: optimisation, regularisation, architecture,
# generative models, evaluation, and transfer learning.

HELD_OUT_QUERIES: List[str] = [
    "Explain the vanishing gradient problem and how residual connections solve it.",
    "What is the difference between L1 and L2 regularisation, and when should you use each?",
    "How does the Adam optimiser differ from SGD with momentum?",
    "Describe how a Variational Autoencoder (VAE) differs from a standard Autoencoder.",
    "What is the purpose of batch normalisation and where should it be placed in a network?",
    "Explain the bias-variance trade-off in machine learning.",
    "How does RLHF (Reinforcement Learning from Human Feedback) improve language models?",
    "What is knowledge distillation and why is it used in model compression?",
    "Describe the difference between fine-tuning and prompt engineering for LLM adaptation.",
    "How does contrastive learning work, and what is its role in self-supervised learning?",
]


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """
    Complete record for one (system, query) pair.

    Fields
    ------
    system_name:
        Identifier for the system that produced this result.
    query:
        The evaluation query.
    initial_answer:
        Raw Generator output before any refinement.
    final_answer:
        Answer returned to the user after the pipeline finished.
    iterations_used:
        Number of critic-refiner cycles that ran (0 for Baseline).
    latency_s:
        Wall-clock seconds from pipeline entry to final answer.
    pipeline_critic_score:
        Final penalised score reported by the pipeline's internal critic.
        For Baseline this is the judge score (no internal critic runs).
    hallucination_count:
        Number of hallucinations detected in the final answer.
    factual_accuracy:
        LLM-as-judge factual accuracy score (0–10).  Same judge used
        for all three systems, enabling fair comparison.
    exit_reason:
        Loop exit reason: "converged" | "stagnated" | "exhausted" | "baseline".
    judge_raw_response:
        Verbatim judge output (for auditing).
    """
    system_name: str
    query: str
    initial_answer: str
    final_answer: str
    iterations_used: int
    latency_s: float
    pipeline_critic_score: float
    hallucination_count: int
    factual_accuracy: float
    exit_reason: str
    judge_raw_response: str = field(repr=False)


@dataclass
class SystemSummary:
    """Aggregated metrics for one system across all evaluated queries."""
    system_name: str
    query_count: int
    avg_factual_accuracy: float      # mean judge score (0–10)
    hallucination_rate: float        # fraction of queries with halluc > 0
    avg_critic_score: float          # mean internal critic score
    avg_iterations: float            # mean refinement cycles
    avg_latency_s: float             # mean wall-clock time
    convergence_rate: float          # fraction of queries that converged


@dataclass
class BenchmarkReport:
    """Full benchmark results including per-query data and summaries."""
    timestamp: str
    queries: List[str]
    per_query_results: Dict[str, List[QueryResult]]   # system_name → results
    summaries: Dict[str, SystemSummary]               # system_name → summary
    improvements: Dict[str, Dict[str, float]]         # "B_vs_A" → {metric → pct}


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

class FactualJudge:
    """
    Evaluates the factual accuracy of a final answer using an LLM.

    Deliberately focused: the judge is asked ONLY about factual accuracy
    and hallucinations, not about writing quality or completeness.  This
    ensures a fair comparison across systems that may produce answers of
    different lengths or styles.

    The judge re-uses the same ``BaseLLM`` interface as the rest of the
    pipeline — swap MockLLM for OllamaLLM for real evaluation.

    Parameters
    ----------
    llm:
        Any BaseLLM implementation.
    """

    _SYSTEM_PROMPT = (
        "You are a factual accuracy evaluator. "
        "Your only job is to assess whether the stated facts in an answer are correct. "
        "Ignore writing style, length, structure, and completeness. "
        "Focus exclusively on: Are the facts correct? Are there hallucinations? "
        "Return ONLY a JSON object with two fields. No prose."
    )

    _PROMPT_TEMPLATE = """\
Evaluate the factual accuracy of the following answer.

QUESTION: {query}

ANSWER: {answer}

Judge ONLY factual correctness. Ignore completeness, style, and length.

Respond with ONLY this JSON:
{{
  "factual_accuracy_score": <float 0-10, where 10=fully accurate, 0=fabricated/wrong>,
  "hallucinations_detected": <true | false>,
  "reasoning": "<one sentence explaining the score>"
}}"""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def judge(self, query: str, answer: str) -> tuple[float, bool, str]:
        """
        Score the factual accuracy of *answer* for *query*.

        Returns
        -------
        (factual_accuracy_score, hallucinations_detected, raw_response)
        """
        prompt = self._PROMPT_TEMPLATE.format(query=query, answer=answer)
        raw = self._llm.complete(prompt, system_prompt=self._SYSTEM_PROMPT)
        return self._parse(raw)

    def _parse(self, raw: str) -> tuple[float, bool, str]:
        import re
        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", raw).strip()
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                data = json.loads(match.group(0))
                score = max(0.0, min(10.0, float(data.get("factual_accuracy_score", 5.0))))
                halluc = bool(data.get("hallucinations_detected", False))
                return score, halluc, raw
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        logger.warning("FactualJudge: could not parse response. Using fallback score.")
        return 5.0, False, raw


# ---------------------------------------------------------------------------
# System runners
# ---------------------------------------------------------------------------

class SystemRunner(ABC):
    """Abstract base for a benchmarkable pipeline configuration."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable system identifier."""

    @abstractmethod
    def run_query(self, query: str, judge: FactualJudge) -> QueryResult:
        """
        Run the system on *query* and return a fully populated QueryResult.

        The implementation must:
        - Measure latency with ``time.perf_counter()``.
        - Call ``judge.judge()`` on the final answer.
        - Never raise (catch errors, return a result with score=0).
        """


class BaselineRunner(SystemRunner):
    """
    System A: Generator only — no critic, no refinement.

    Represents the naive baseline.  The Generator produces one answer
    and we immediately judge it without any iterative improvement.
    """

    @property
    def name(self) -> str:
        return "A_Baseline"

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    def run_query(self, query: str, judge: FactualJudge) -> QueryResult:
        t0 = time.perf_counter()
        try:
            gen_out = self._generator.generate(query)
            final_answer = gen_out.answer
            initial_answer = gen_out.answer
        except Exception as exc:
            logger.error("BaselineRunner: generation failed: %s", exc)
            final_answer = initial_answer = ""

        latency = round(time.perf_counter() - t0, 3)

        # Judge the raw answer
        factual_score, has_halluc, judge_raw = judge.judge(query, final_answer)

        return QueryResult(
            system_name=self.name,
            query=query,
            initial_answer=initial_answer,
            final_answer=final_answer,
            iterations_used=0,
            latency_s=latency,
            pipeline_critic_score=factual_score,  # no internal critic — use judge score
            hallucination_count=1 if has_halluc else 0,
            factual_accuracy=factual_score,
            exit_reason="baseline",
            judge_raw_response=judge_raw,
        )


class LLMCriticRunner(SystemRunner):
    """
    System B: Generator + LLM Critic (research-grade) + adaptive Refiner loop.

    Uses the full ``RefinementLoop`` with the adversarial LLM-based Critic
    that penalises hallucinations, factual errors, and logical flaws.
    """

    @property
    def name(self) -> str:
        return "B_LLM_Critic"

    def __init__(self, loop: RefinementLoop) -> None:
        self._loop = loop

    def run_query(self, query: str, judge: FactualJudge) -> QueryResult:
        t0 = time.perf_counter()
        try:
            result = self._loop.run(query)
        except (OllamaError, Exception) as exc:
            logger.error("LLMCriticRunner: pipeline failed: %s", exc)
            # Return a zero-scored result rather than crashing the benchmark
            return _error_result(self.name, query, str(exc))

        latency = round(time.perf_counter() - t0, 3)
        factual_score, has_halluc, judge_raw = judge.judge(query, result.final_answer)

        last_feedback = result.iterations[-1].feedback if result.iterations else None
        halluc_count = len(last_feedback.hallucinations) if last_feedback else (1 if has_halluc else 0)
        critic_score = last_feedback.score if last_feedback else 0.0

        return QueryResult(
            system_name=self.name,
            query=query,
            initial_answer=result.initial_answer,
            final_answer=result.final_answer,
            iterations_used=result.total_iterations,
            latency_s=latency,
            pipeline_critic_score=critic_score,
            hallucination_count=halluc_count,
            factual_accuracy=factual_score,
            exit_reason=result.exit_reason,
            judge_raw_response=judge_raw,
        )


class LearnedCriticRunner(SystemRunner):
    """
    System C: Generator + LearnedCritic (fast MiniLM) with LLM fallback
    + adaptive Refiner loop.

    Uses the same loop architecture as System B but replaces the LLM
    critic with the trained lightweight model.  Falls back to the LLM
    critic when the answer is below the model's confidence threshold.

    If torch/transformers are not installed or model weights are missing,
    the runner falls back to System B behaviour and logs a warning.
    """

    @property
    def name(self) -> str:
        return "C_Learned_Critic"

    def __init__(self, loop: RefinementLoop) -> None:
        self._loop = loop

    def run_query(self, query: str, judge: FactualJudge) -> QueryResult:
        t0 = time.perf_counter()
        try:
            result = self._loop.run(query)
        except (OllamaError, Exception) as exc:
            logger.error("LearnedCriticRunner: pipeline failed: %s", exc)
            return _error_result(self.name, query, str(exc))

        latency = round(time.perf_counter() - t0, 3)
        factual_score, has_halluc, judge_raw = judge.judge(query, result.final_answer)

        last_feedback = result.iterations[-1].feedback if result.iterations else None
        halluc_count = len(last_feedback.hallucinations) if last_feedback else (1 if has_halluc else 0)
        critic_score = last_feedback.score if last_feedback else 0.0

        return QueryResult(
            system_name=self.name,
            query=query,
            initial_answer=result.initial_answer,
            final_answer=result.final_answer,
            iterations_used=result.total_iterations,
            latency_s=latency,
            pipeline_critic_score=critic_score,
            hallucination_count=halluc_count,
            factual_accuracy=factual_score,
            exit_reason=result.exit_reason,
            judge_raw_response=judge_raw,
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _error_result(system_name: str, query: str, error: str) -> QueryResult:
    """Return a zero-scored result when a pipeline raises an unrecoverable error."""
    return QueryResult(
        system_name=system_name,
        query=query,
        initial_answer="",
        final_answer=f"[ERROR: {error}]",
        iterations_used=0,
        latency_s=0.0,
        pipeline_critic_score=0.0,
        hallucination_count=0,
        factual_accuracy=0.0,
        exit_reason="error",
        judge_raw_response="",
    )


def build_systems(
    llm_config: LLMConfig,
    loop_config: LoopConfig,
    pipeline_config: PipelineConfig,
    use_mock: bool,
    model_dir: str,
    skip_learned: bool,
) -> tuple[List[SystemRunner], FactualJudge]:
    """
    Instantiate all three system runners and the factual judge.

    Parameters
    ----------
    llm_config:
        Shared sampling parameters.
    loop_config:
        Loop configuration (max_iterations, stagnation, etc.).
    pipeline_config:
        Model routing (generator/critic/refiner model names).
    use_mock:
        Use MockLLM for all components (no Ollama required).
    model_dir:
        Path to trained LearnedCritic weights.
    skip_learned:
        If True, skip System C entirely.

    Returns
    -------
    (runners, judge)
    """
    def _llm(model: str) -> BaseLLM:
        if use_mock:
            return MockLLM(llm_config)
        return OllamaLLM(llm_config, model=model)

    gen_llm  = _llm(pipeline_config.generator_model)
    crit_llm = _llm(pipeline_config.critic_model)
    ref_llm  = _llm(pipeline_config.refiner_model)
    judge_llm = _llm(pipeline_config.critic_model)  # judge uses same model as critic

    generator = Generator(gen_llm, llm_config)
    critic    = Critic(crit_llm, llm_config)
    refiner   = Refiner(ref_llm, llm_config)
    judge     = FactualJudge(judge_llm)

    # --- System A: Baseline ---
    baseline = BaselineRunner(generator=generator)

    # --- System B: LLM Critic ---
    llm_loop = RefinementLoop(
        generator=generator,
        critic=critic,
        refiner=refiner,
        config=loop_config,
    )
    llm_runner = LLMCriticRunner(loop=llm_loop)

    runners: List[SystemRunner] = [baseline, llm_runner]

    # --- System C: Learned Critic ---
    if not skip_learned:
        learned_critic = _try_build_learned_critic(
            model_dir=model_dir,
            llm_fallback=crit_llm,
            llm_config=llm_config,
        )
        if learned_critic is not None:
            learned_loop = RefinementLoop(
                generator=generator,
                critic=learned_critic,   # duck-typed: same .critique() interface
                refiner=refiner,
                config=loop_config,
            )
            runners.append(LearnedCriticRunner(loop=learned_loop))
        else:
            logger.warning(
                "System C (Learned Critic) skipped — torch/transformers "
                "unavailable or weights not found at '%s'.", model_dir
            )

    return runners, judge


def _try_build_learned_critic(
    model_dir: str, llm_fallback: BaseLLM, llm_config: LLMConfig
):
    """
    Attempt to build a LearnedCritic.  Returns None on any import or
    initialisation failure so the benchmark degrades gracefully.
    """
    try:
        from core.learned_critic import LearnedCritic
        return LearnedCritic(
            model_dir=model_dir,
            llm_fallback=llm_fallback,
            config=llm_config,
        )
    except ImportError as exc:
        logger.warning("LearnedCritic import failed (missing dependency): %s", exc)
        return None
    except Exception as exc:
        logger.warning("LearnedCritic initialisation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------

class Benchmark:
    """
    Runs all system runners against the query set and produces a report.

    Parameters
    ----------
    runners:
        List of SystemRunner instances (typically A, B, C).
    judge:
        FactualJudge used consistently across all systems.
    queries:
        Held-out evaluation queries.
    output_dir:
        Directory where JSON results are saved.
    """

    def __init__(
        self,
        runners: List[SystemRunner],
        judge: FactualJudge,
        queries: List[str],
        output_dir: str = "evaluation/results",
    ) -> None:
        self._runners = runners
        self._judge = judge
        self._queries = queries
        self._output_dir = Path(output_dir)

    def run(self) -> BenchmarkReport:
        """
        Execute all systems on all queries and return the full report.
        """
        logger.info("=" * 60)
        logger.info(
            "Benchmark start | systems=%d  queries=%d",
            len(self._runners), len(self._queries),
        )
        logger.info("=" * 60)

        per_query: Dict[str, List[QueryResult]] = {r.name: [] for r in self._runners}

        total = len(self._runners) * len(self._queries)
        done = 0

        for query in self._queries:
            logger.info("Query: %r", query[:60])
            for runner in self._runners:
                logger.info("  Running system: %s", runner.name)
                result = runner.run_query(query, self._judge)
                per_query[runner.name].append(result)
                done += 1
                logger.info(
                    "  [%d/%d] %s | factual=%.2f  halluc=%d  iters=%d  latency=%.2fs",
                    done, total,
                    runner.name,
                    result.factual_accuracy,
                    result.hallucination_count,
                    result.iterations_used,
                    result.latency_s,
                )

        summaries = {
            name: self._compute_summary(name, results)
            for name, results in per_query.items()
        }
        improvements = self._compute_improvements(summaries)

        report = BenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            queries=self._queries,
            per_query_results=per_query,
            summaries=summaries,
            improvements=improvements,
        )

        self._print_table(report)
        saved_path = self._save_json(report)
        logger.info("Results saved to: %s", saved_path)

        return report

    # ------------------------------------------------------------------
    # Summary computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_summary(name: str, results: List[QueryResult]) -> SystemSummary:
        n = len(results)
        if n == 0:
            return SystemSummary(
                system_name=name, query_count=0,
                avg_factual_accuracy=0.0, hallucination_rate=0.0,
                avg_critic_score=0.0, avg_iterations=0.0,
                avg_latency_s=0.0, convergence_rate=0.0,
            )
        return SystemSummary(
            system_name=name,
            query_count=n,
            avg_factual_accuracy=round(
                sum(r.factual_accuracy for r in results) / n, 3
            ),
            hallucination_rate=round(
                sum(1 for r in results if r.hallucination_count > 0) / n, 3
            ),
            avg_critic_score=round(
                sum(r.pipeline_critic_score for r in results) / n, 3
            ),
            avg_iterations=round(
                sum(r.iterations_used for r in results) / n, 2
            ),
            avg_latency_s=round(
                sum(r.latency_s for r in results) / n, 3
            ),
            convergence_rate=round(
                sum(1 for r in results if r.exit_reason == "converged") / n, 3
            ),
        )

    @staticmethod
    def _compute_improvements(
        summaries: Dict[str, SystemSummary],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute % improvement of each system vs the Baseline (System A).

        For metrics where higher is better (factual_accuracy, critic_score,
        convergence_rate): pct = (improved - baseline) / max(baseline, 0.01) * 100
        For metrics where lower is better (hallucination_rate, latency):
        pct = (baseline - improved) / max(baseline, 0.01) * 100
        """
        baseline_name = "A_Baseline"
        if baseline_name not in summaries:
            return {}

        base = summaries[baseline_name]
        improvements: Dict[str, Dict[str, float]] = {}

        for name, s in summaries.items():
            if name == baseline_name:
                continue
            tag = f"{name}_vs_{baseline_name}"

            def _pct_up(improved: float, baseline: float) -> float:
                """Higher-is-better metric."""
                return round((improved - baseline) / max(abs(baseline), 0.01) * 100, 1)

            def _pct_down(improved: float, baseline: float) -> float:
                """Lower-is-better metric."""
                return round((baseline - improved) / max(abs(baseline), 0.01) * 100, 1)

            improvements[tag] = {
                "factual_accuracy_pct":    _pct_up(s.avg_factual_accuracy,   base.avg_factual_accuracy),
                "hallucination_rate_pct":  _pct_down(s.hallucination_rate,   base.hallucination_rate),
                "critic_score_pct":        _pct_up(s.avg_critic_score,       base.avg_critic_score),
                "iterations_pct":          _pct_down(s.avg_iterations,       base.avg_iterations),
                "latency_pct":             _pct_down(s.avg_latency_s,        base.avg_latency_s),
                "convergence_rate_pct":    _pct_up(s.convergence_rate,       base.convergence_rate),
            }
        return improvements

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _print_table(self, report: BenchmarkReport) -> None:
        """Print a formatted comparison table to stdout."""
        sep = "-" * 80
        print(f"\n{sep}")
        print("BENCHMARK RESULTS")
        print(f"Timestamp : {report.timestamp}")
        print(f"Queries   : {len(report.queries)}")
        print(f"Systems   : {list(report.summaries.keys())}")
        print(sep)

        metrics = [
            ("Factual Accuracy (0-10)", "avg_factual_accuracy",  True,  ".2f"),
            ("Hallucination Rate",       "hallucination_rate",    False, ".1%"),
            ("Avg Critic Score (0-10)",  "avg_critic_score",      True,  ".2f"),
            ("Avg Iterations",           "avg_iterations",        False, ".2f"),
            ("Avg Latency (s)",          "avg_latency_s",         True,  ".3f"),
            ("Convergence Rate",         "convergence_rate",      True,  ".1%"),
        ]

        # Header
        systems = list(report.summaries.keys())
        col_w = 22
        hdr = f"{'Metric':<30}" + "".join(f"{s:<{col_w}}" for s in systems)
        print(hdr)
        print("-" * len(hdr))

        # Rows
        for label, attr, higher_better, fmt in metrics:
            row = f"{label:<30}"
            values = [getattr(report.summaries[s], attr) for s in systems]
            best_val = max(values) if higher_better else min(values)

            for i, (s, v) in enumerate(zip(systems, values)):
                cell = format(v, fmt)
                is_best = abs(v - best_val) < 1e-9
                marker = " *" if is_best and len(systems) > 1 else "  "
                row += f"{cell + marker:<{col_w}}"
            print(row)

        print(sep)

        # % Improvement vs Baseline
        if report.improvements:
            print("\n% IMPROVEMENT vs BASELINE (A_Baseline)")
            print("-" * 60)
            improvement_labels = {
                "factual_accuracy_pct":   "Factual Accuracy",
                "hallucination_rate_pct": "Hallucination Rate (lower)",
                "critic_score_pct":       "Critic Score",
                "convergence_rate_pct":   "Convergence Rate",
                "latency_pct":            "Latency (lower)",
            }
            for tag, deltas in report.improvements.items():
                system_label = tag.replace("_vs_A_Baseline", "")
                print(f"\n  {system_label}:")
                for key, label in improvement_labels.items():
                    pct = deltas.get(key, 0.0)
                    direction = "+" if pct >= 0 else ""
                    print(f"    {label:<30} {direction}{pct:.1f}%")

        print(sep)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_json(self, report: BenchmarkReport) -> str:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._output_dir / f"benchmark_{ts}.json"

        # Serialize per_query_results (QueryResult → dict)
        per_query_serialized = {
            name: [asdict(r) for r in results]
            for name, results in report.per_query_results.items()
        }
        summaries_serialized = {
            name: asdict(s) for name, s in report.summaries.items()
        }

        payload = {
            "timestamp":         report.timestamp,
            "queries":           report.queries,
            "summaries":         summaries_serialized,
            "improvements":      report.improvements,
            "per_query_results": per_query_serialized,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return str(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark: Baseline vs LLM Critic vs Learned Critic"
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="Path to a text file with one query per line. "
             "Defaults to built-in held-out set.",
    )
    parser.add_argument(
        "--generator-model", type=str, default="mistral:7b",
        help="Ollama model for the Generator."
    )
    parser.add_argument(
        "--critic-model", type=str, default="llama3.1:8b",
        help="Ollama model for the LLM Critic and judge."
    )
    parser.add_argument(
        "--refiner-model", type=str, default="mistral:7b",
        help="Ollama model for the Refiner."
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:11434",
        help="Ollama server base URL."
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Per-request timeout in seconds."
    )
    parser.add_argument(
        "--max-iterations", type=int, default=3,
        help="Maximum refinement cycles per query."
    )
    parser.add_argument(
        "--min-score", type=float, default=7.0,
        help="Minimum critic score to accept answer (0-10)."
    )
    parser.add_argument(
        "--stagnation-patience", type=int, default=2,
        help="Consecutive flat iterations before early stop."
    )
    parser.add_argument(
        "--min-improvement-delta", type=float, default=0.3,
        help="Minimum per-iteration score gain to count as progress."
    )
    parser.add_argument(
        "--model-dir", type=str, default="trained_models/learned_critic",
        help="Directory containing trained LearnedCritic weights."
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation/results",
        help="Directory where JSON benchmark results are saved."
    )
    parser.add_argument(
        "--skip-learned", action="store_true",
        help="Skip System C (Learned Critic). Useful if torch is not installed."
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use MockLLM for all components (no Ollama required)."
    )
    args = parser.parse_args()

    # Load queries
    if args.queries_file and Path(args.queries_file).exists():
        with open(args.queries_file, encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
        logger.info("Loaded %d queries from %s", len(queries), args.queries_file)
    else:
        queries = HELD_OUT_QUERIES
        logger.info("Using built-in held-out query set (%d queries).", len(queries))

    llm_config = LLMConfig(
        base_url=args.base_url,
        timeout=args.timeout,
    )
    loop_config = LoopConfig(
        max_iterations=args.max_iterations,
        min_quality_score=args.min_score,
        stagnation_patience=args.stagnation_patience,
        min_improvement_delta=args.min_improvement_delta,
    )
    pipeline_config = PipelineConfig(
        llm=llm_config,
        loop=loop_config,
        generator_model=args.generator_model,
        critic_model=args.critic_model,
        refiner_model=args.refiner_model,
    )

    runners, judge = build_systems(
        llm_config=llm_config,
        loop_config=loop_config,
        pipeline_config=pipeline_config,
        use_mock=args.mock,
        model_dir=args.model_dir,
        skip_learned=args.skip_learned,
    )

    benchmark = Benchmark(
        runners=runners,
        judge=judge,
        queries=queries,
        output_dir=args.output_dir,
    )
    benchmark.run()


if __name__ == "__main__":
    main()