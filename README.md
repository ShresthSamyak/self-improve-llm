# Self-Correcting LLM with Critic–Refiner Loop and Tool-Augmented Grounding

> A modular framework for studying self-correction, learned evaluation, and tool-augmented reasoning in LLM systems.

## Abstract

This project explores iterative self-improvement in large language models (LLMs) through a critic–refiner feedback loop, augmented with learned evaluation and optional tool-based grounding.

The system combines:
- LLM-based critique
- A learned critic model (MiniLM)
- Parameter-efficient fine-tuning (QLoRA)
- Optional browser-based evidence retrieval

We benchmark multiple configurations to evaluate how self-reflection, learned evaluation, and external grounding affect answer quality, hallucination rates, and convergence behavior.

---

## System Overview

The core pipeline follows an iterative refinement loop:

```text
            ┌──────────────┐
            │   User Query │
            └──────┬───────┘
                   ↓
            ┌──────────────┐
            │  Generator   │
            │   (LLM)      │
            └──────┬───────┘
                   ↓
            ┌──────────────┐
            │    Critic    │
            │ (LLM / ML)   │
            └──────┬───────┘
                   ↓
    ┌──────────────┴──────────────┐
    │                             │
    ↓                             ↓
┌──────────────┐          ┌──────────────┐
│   Browser    │          │   Refiner    │
│  (Optional)  │          │    (LLM)     │
└──────┬───────┘          └──────┬───────┘
       │                         ↓
       └──────────→──────────→───┘
                                 ↓
                          ┌──────────────┐
                          │  Loop Exit   │
                          └──────────────┘
```

Each iteration:
- Evaluates the current answer
- Identifies hallucinations, factual errors, and gaps
- Refines the response accordingly

---

## Key Features

### 1. Critic–Refiner Loop
- Iterative self-correction mechanism
- Tracks score, delta, and convergence behavior
- Supports strict mode when hallucinations persist

### 2. Learned Critic (MiniLM)
- Multi-task model:
  - Quality score prediction (regression)
  - Verdict classification
- Reduces reliance on expensive LLM critiques
- Falls back to LLM critic for detailed feedback

### 3. QLoRA Fine-Tuned Critic
- Fine-tuned Mistral-7B using LoRA adapters
- Efficient training with low memory footprint
- Synthetic dataset generated via corruption strategies

### 4. Browser-Augmented Grounding (Optional)
- CLI-based browsing (DuckDuckGo + w3m)
- Triggered only when:
  - hallucinations detected
  - factual errors present
- Injects retrieved evidence into refinement loop

---

## Training Pipeline

### Dataset Generation
- Synthetic corruption of correct answers:
  - factual distortions
  - logical inconsistencies
  - missing concepts
- Enables supervised learning of critique behavior

### MiniLM Critic Training
- Input: (query, answer)
- Outputs:
  - quality score
  - verdict label
- Loss:
  - regression + classification

### QLoRA Fine-Tuning
- Base model: Mistral-7B
- LoRA adapters for efficient updates
- Fine-tuned as a critic model

---

## Browser-Augmented System

The system includes an optional tool-use module for grounding.

### Behavior
- Activated when critic detects errors
- Performs:
  - search → fetch → inject evidence
- Evidence used in subsequent refinement

### Observed Limitation

Retrieval success is inconsistent:

- In multiple runs, browser triggered correctly  
- However, **0 pages were retrieved in some cases**

Example:
```text
Browser grounding triggered
pages_retrieved = 0
```

This highlights a key limitation:

> The effectiveness of tool-augmented LLMs is highly dependent on retrieval reliability.

---

## Benchmark Setup

We evaluate four systems:

| System | Description |
|------|------------|
| A | Baseline (no refinement) |
| B | LLM Critic |
| C | Learned Critic |
| D | Browser-Augmented |

Metrics:
- Factual accuracy
- Hallucination count
- Iterations to convergence
- Latency

---

## Benchmark Summary

| System | Behavior | Strength | Limitation |
|--------|--------|----------|-----------|
| Baseline | Single-pass generation | Fast | No correction |
| LLM Critic | Iterative refinement | Reduces errors | Can stagnate |
| Learned Critic | Faster evaluation | Efficient | Misalignment with true quality |
| Browser-Augmented | Grounded refinement | Improves factuality (when retrieval works) | Retrieval-dependent |

### Key Observation

- Self-correction improves quality, but is **not guaranteed to converge**
- Tool augmentation improves results **only when retrieval succeeds**
- Learned critics introduce speed but reduce reliability

---

## Empirical Observations

### 1. Iterative Refinement is Non-Monotonic

Performance does not always improve across iterations.

Example:
- score: 0.70 → 0.00  
- delta: negative  
- loop exit: exhausted

Indicates:
- refinement can degrade outputs
- critic feedback is not always constructive

---

### 2. Hallucination Persistence

Even after multiple iterations:

- hallucinations remain
- strict mode activates
- convergence often fails

---

### 3. Learned Critic Limitations

- Predicts reasonable scores (~6.5)
- But actual evaluation may drop to 0

This shows:
> Learned evaluation does not fully align with true correctness.

---

### 4. Tool-Augmented Gains (Conditional)

When retrieval succeeds:
- hallucinations decrease
- scores improve (e.g., 0 → 2.8)

When retrieval fails:
- Browser module triggers correctly
- However, retrieval may fail (0 pages returned)
- no improvement, system behaves like standard loop

Implication:

> Effectiveness of tool-augmented systems depends heavily on retrieval reliability

---

### 5. Stagnation Behavior

- System detects low improvement (delta ≈ 0)
- Exits early to avoid wasted iterations

---

## Failure Analysis

This system exhibits several important failure modes:

### 1. Non-Monotonic Refinement
- Iterative updates can degrade answer quality
- Example:
  - score: 0.70 → 0.00
  - loop exit: exhausted

### 2. Hallucination Persistence
- Some hallucinations persist across multiple iterations
- Even under strict refinement mode

### 3. Retrieval Failure
- Browser module may return zero results
- Leads to no improvement in grounded refinement

### 4. Critic–Reality Misalignment
- Learned critic predicts "good"
- LLM critic assigns score ≈ 0

### 5. Stagnation
- System halts when improvements fall below threshold
- Indicates limitations in refinement strategy

These failure modes highlight that:

> Self-improving LLM systems require not just feedback, but reliable evaluation and grounding mechanisms.

---

## Why This Matters

Modern LLM systems increasingly rely on:

- Self-reflection
- Tool usage
- Iterative reasoning

This project explores all three in a unified framework.

Key implications:

- Self-correction alone is insufficient for reliability
- Learned evaluation introduces efficiency–accuracy tradeoffs
- Tool augmentation is powerful but brittle
- Reliable AI systems require coordination between reasoning, evaluation, and grounding

This work serves as a prototype for:

- Agentic AI systems
- Autonomous reasoning loops
- Reliable LLM deployment pipelines

---

## Limitations

- Synthetic dataset does not fully represent real LLM failure modes
- Learned critic shows weak alignment with ground truth evaluation
- Browser retrieval is inconsistent
- No human preference data (unlike RLHF-style training)
- Quantitative improvements are limited and context-dependent

---

## Related Work

This project builds on existing paradigms:

- Self-Refine (Madaan et al., 2023)
- Reflexion (Shinn et al., 2023)

The focus here is on:
- system integration
- evaluation behavior
- practical experimentation

---

## Design Philosophy

- **Modularity**: generator, critic, and refiner are decoupled
- **Conditional complexity**: expensive components (LLM critic, browser) activate only when needed
- **Measurability**: each iteration tracks score and convergence signals

---

## Installation

```bash
git clone <repo-url>
cd self-correcting-llm
pip install -r requirements.txt
```

## Usage

Run benchmark:

```bash
python evaluation/run_benchmark.py
```

With browser:

```bash
python evaluation/run_benchmark.py --use-browser
```

Run pipeline:

```bash
python app.py
```

## Project Structure
```text
core/
  generator.py
  critic.py
  learned_critic.py
  refiner.py
  loop.py

tools/
  browser.py

scripts/
  generate_dataset.py
  train_critic.py
  finetune_critic_qlora.py

evaluation/
  run_benchmark.py
```
