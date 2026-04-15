"""
scripts/finetune_critic_qlora.py
---------------------------------
QLoRA fine-tuning of Mistral-7B-Instruct to act as a structured critic.

The trained model learns to produce structured JSON feedback — identical
to the schema used by the pipeline's LLM Critic — directly from a
(query, answer) pair, without an external API call.

Architecture
------------
  Base model : mistralai/Mistral-7B-Instruct-v0.2
  Quantisation: 4-bit NF4 via bitsandbytes
  Adapter    : LoRA (r=16, alpha=32) on q_proj + v_proj
  Training   : HuggingFace Trainer, FP16 mixed precision
  Output     : LoRA adapter weights only (< 100 MB)

Dataset format (JSONL, one object per line)
-------------------------------------------
{
  "query": "...",
  "answer": "...",
  "feedback": {
    "factual_errors":   [...],
    "hallucinations":   [...],
    "missing_concepts": [...],
    "logical_flaws":    [...],
    "score":            7.5,
    "verdict":          "good"
  }
}

Instruction format used for fine-tuning
----------------------------------------
### Instruction:
Evaluate the quality of the answer and provide structured critique.

### Input:
Query: {query}
Answer: {answer}

### Response:
{
  "hallucinations": [...],
  "factual_errors": [...],
  "missing_concepts": [...],
  "logical_flaws": [...],
  "score": 7.5,
  "verdict": "good"
}

Usage
-----
  # Basic run
  python scripts/finetune_critic_qlora.py \\
      --data-path data/critic_train.jsonl \\
      --output-dir trained_models/critic_qlora

  # Custom hyperparameters
  python scripts/finetune_critic_qlora.py \\
      --data-path data/critic_train.jsonl \\
      --output-dir trained_models/critic_qlora \\
      --base-model mistralai/Mistral-7B-Instruct-v0.2 \\
      --epochs 2 \\
      --batch-size 2 \\
      --grad-accum 8 \\
      --lora-r 16 \\
      --lora-alpha 32 \\
      --max-length 1024

  # Dry-run on 32 samples (smoke-test without GPU)
  python scripts/finetune_critic_qlora.py \\
      --data-path data/critic_train.jsonl \\
      --output-dir trained_models/critic_qlora_test \\
      --max-samples 32 --epochs 1

Dependencies
------------
  pip install torch transformers peft bitsandbytes accelerate datasets

Notes
-----
- Requires a CUDA-capable GPU with >= 12 GB VRAM for Mistral-7B in 4-bit.
- On CPU-only machines, omit --load-in-4bit (training will be very slow).
- The adapter is saved with ``model.save_pretrained(output_dir)``.
  Load it at inference with ``PeftModel.from_pretrained(base, adapter_dir)``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging setup (before heavy imports so failures are visible)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heavy imports — fail early with helpful messages
# ---------------------------------------------------------------------------
def _require(pkg: str, install: str) -> None:
    try:
        __import__(pkg)
    except ImportError:
        logger.error(
            "Missing dependency '%s'. Install it with:\n    pip install %s",
            pkg, install,
        )
        sys.exit(1)


_require("torch",          "torch")
_require("transformers",   "transformers")
_require("peft",           "peft")
_require("bitsandbytes",   "bitsandbytes")
_require("accelerate",     "accelerate")
_require("datasets",       "datasets")

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

INSTRUCTION = (
    "Evaluate the quality of the answer and provide structured critique. "
    "Return ONLY valid JSON matching the schema shown."
)

PROMPT_TEMPLATE = """\
### Instruction:
{instruction}

### Input:
Query: {query}
Answer: {answer}

### Response:
"""

RESPONSE_SCHEMA_COMMENT = """\
{{
  "hallucinations":   ["<unverifiable or fabricated claim>"],
  "factual_errors":   ["<specific wrong fact>"],
  "missing_concepts": ["<important concept not covered>"],
  "logical_flaws":    ["<reasoning error>"],
  "score":            <float 0-10>,
  "verdict":          "<poor | acceptable | good | excellent>"
}}"""


def build_prompt(query: str, answer: str) -> str:
    """Return the instruction-formatted prompt (no response appended)."""
    return PROMPT_TEMPLATE.format(
        instruction=INSTRUCTION,
        query=query,
        answer=answer,
    )


def build_feedback_json(feedback: Dict[str, Any]) -> str:
    """
    Serialise the feedback dict to a compact, deterministic JSON string.

    Only the six fields the model is trained to produce are included.
    Extra fields (raw_llm_score, raw_response, etc.) are stripped so
    the model learns a clean target.
    """
    clean = {
        "hallucinations":   feedback.get("hallucinations",   []),
        "factual_errors":   feedback.get("factual_errors",   []),
        "missing_concepts": feedback.get("missing_concepts", []),
        "logical_flaws":    feedback.get("logical_flaws",    []),
        "score":            round(float(feedback.get("score", 5.0)), 2),
        "verdict":          feedback.get("verdict", "acceptable"),
    }
    return json.dumps(clean, indent=2)


# ---------------------------------------------------------------------------
# Dataset loading & formatting
# ---------------------------------------------------------------------------

def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load a JSONL or JSON file. Supports:
    - One JSON object per line (JSONL).
    - A JSON array (legacy format from generate_dataset.py).
    """
    data_path = Path(path)
    if not data_path.exists():
        logger.error("Dataset not found: %s", path)
        sys.exit(1)

    raw = data_path.read_text(encoding="utf-8").strip()
    records: List[Dict] = []

    # Try JSONL first, fall back to JSON array
    if raw.startswith("["):
        records = json.loads(raw)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    logger.info("Loaded %d samples from %s", len(records), path)
    return records


def format_sample(record: Dict) -> Dict[str, str]:
    """
    Convert a dataset record into a single training string.

    Format:
        <prompt><response><eos>

    The loss is computed on the full string by default.  For
    instruction-tuning best practice, pass ``DataCollatorForSeq2Seq``
    with ``label_pad_token_id=-100`` to mask the prompt tokens.
    """
    prompt   = build_prompt(record["query"], record["answer"])
    response = build_feedback_json(record["feedback"])
    return {"text": prompt + response}


def prepare_dataset(
    records: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """
    Tokenise the dataset and return a HuggingFace Dataset.

    Each sample is tokenised as:
        input_ids  = tokenise(prompt + response + eos)
        labels     = -100 for prompt tokens, token_id for response tokens

    Masking the prompt prevents the model from wasting capacity
    learning to reproduce the fixed instruction template.
    """
    formatted = [format_sample(r) for r in records]

    def tokenise(batch: Dict[str, List[str]]) -> Dict:
        results = {"input_ids": [], "attention_mask": [], "labels": []}

        for text in batch["text"]:
            # Tokenise full text (prompt + response)
            full = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            input_ids      = full["input_ids"]
            attention_mask = full["attention_mask"]

            # Find where the response starts so we can mask the prompt
            prompt_only = text.split("### Response:\n")[0] + "### Response:\n"
            prompt_ids  = tokenizer(
                prompt_only,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )["input_ids"]

            prompt_len = len(prompt_ids)
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            # Append EOS token if the sequence wasn't truncated
            if len(input_ids) < max_length:
                eos = tokenizer.eos_token_id
                input_ids.append(eos)
                attention_mask.append(1)
                labels.append(eos)

            results["input_ids"].append(input_ids)
            results["attention_mask"].append(attention_mask)
            results["labels"].append(labels)

        return results

    raw_ds = Dataset.from_list(formatted)
    tokenised = raw_ds.map(
        tokenise,
        batched=True,
        remove_columns=["text"],
        desc="Tokenising",
    )
    return tokenised


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(
    model_name: str,
    load_in_4bit: bool = True,
) -> AutoModelForCausalLM:
    """
    Load the base causal LM, optionally in 4-bit NF4 quantisation.

    NF4 + double quantisation cuts the memory footprint of Mistral-7B
    from ~14 GB (FP16) to ~5 GB, making it runnable on a 12 GB GPU.
    """
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        logger.info("Loading %s in 4-bit NF4 quantisation.", model_name)
    else:
        bnb_config = None
        logger.info("Loading %s in full precision (no quantisation).", model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        torch_dtype=torch.float16 if not load_in_4bit else None,
    )
    model.config.use_cache = False          # Required for gradient checkpointing
    model.config.pretraining_tp = 1         # Disable tensor parallelism during training
    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)

    # Mistral uses </s> as EOS but has no explicit pad token — set it to EOS.
    # This is standard practice for Mistral fine-tuning.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"   # Left padding causes issues with causal LMs
    return tokenizer


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

def apply_lora(
    model: AutoModelForCausalLM,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> AutoModelForCausalLM:
    """
    Wrap the quantised base model with LoRA adapters.

    Target modules: q_proj and v_proj are the attention weight matrices
    that benefit most from task-specific adaptation.  Adding k_proj and
    o_proj can improve quality at the cost of more trainable parameters.

    Trainable parameters after LoRA injection ≈ r * 2 * num_layers * 2
    For Mistral-7B (32 layers, r=16): ~4M trainable out of 7B total (~0.06%).
    """
    # prepare_model_for_kbit_training casts LayerNorm weights to FP32
    # and enables gradient checkpointing for memory efficiency.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        inference_mode=False,
    )
    # autocast_adapter_dtype=False disables float8 autocast which requires
    # torch >= 2.1 with float8 support. Pass it to get_peft_model, not LoraConfig.
    import inspect as _inspect
    _gpeft_kwargs = {}
    if "autocast_adapter_dtype" in _inspect.signature(get_peft_model).parameters:
        _gpeft_kwargs["autocast_adapter_dtype"] = False
    model = get_peft_model(model, lora_config, **_gpeft_kwargs)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training callback — logs per-step metrics cleanly
# ---------------------------------------------------------------------------

class LoggingCallback(TrainerCallback):
    """Emit a clean log line every N steps."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step  = state.global_step
        total = state.max_steps
        parts = [f"step={step}/{total}"]
        for key in ("loss", "learning_rate", "epoch"):
            if key in logs:
                val = logs[key]
                parts.append(f"{key}={val:.4g}" if isinstance(val, float) else f"{key}={val}")
        logger.info("  ".join(parts))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("QLoRA Fine-tuning: Mistral-7B → Critic")
    logger.info("=" * 60)
    logger.info("Config:")
    logger.info("  base_model    : %s", args.base_model)
    logger.info("  data_path     : %s", args.data_path)
    logger.info("  output_dir    : %s", args.output_dir)
    logger.info("  epochs        : %d", args.epochs)
    logger.info("  batch_size    : %d", args.batch_size)
    logger.info("  grad_accum    : %d", args.grad_accum)
    logger.info("  lora_r        : %d", args.lora_r)
    logger.info("  lora_alpha    : %d", args.lora_alpha)
    logger.info("  max_length    : %d", args.max_length)
    logger.info("  load_in_4bit  : %s", args.load_in_4bit)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    records = load_jsonl(args.data_path, max_samples=args.max_samples)
    if len(records) == 0:
        logger.error("Dataset is empty. Aborting.")
        sys.exit(1)

    # Train / validation split (90 / 10)
    split_idx   = max(1, int(0.9 * len(records)))
    train_records = records[:split_idx]
    val_records   = records[split_idx:]
    logger.info(
        "Split: train=%d  val=%d", len(train_records), len(val_records)
    )

    # ------------------------------------------------------------------
    # 2. Load tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer: %s", args.base_model)
    tokenizer = load_tokenizer(args.base_model)

    # ------------------------------------------------------------------
    # 3. Tokenise datasets
    # ------------------------------------------------------------------
    logger.info("Tokenising training set...")
    train_dataset = prepare_dataset(train_records, tokenizer, args.max_length)

    eval_dataset = None
    if val_records:
        logger.info("Tokenising validation set...")
        eval_dataset = prepare_dataset(val_records, tokenizer, args.max_length)

    logger.info("Training samples  : %d", len(train_dataset))
    if eval_dataset:
        logger.info("Validation samples: %d", len(eval_dataset))

    # ------------------------------------------------------------------
    # 4. Load base model
    # ------------------------------------------------------------------
    model = load_base_model(args.base_model, load_in_4bit=args.load_in_4bit)

    # ------------------------------------------------------------------
    # 5. Apply LoRA
    # ------------------------------------------------------------------
    model = apply_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # ------------------------------------------------------------------
    # 6. Training arguments
    # ------------------------------------------------------------------
    effective_batch = args.batch_size * args.grad_accum
    logger.info(
        "Effective batch size: %d × %d = %d",
        args.batch_size, args.grad_accum, effective_batch,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",      # memory-efficient AdamW for QLoRA
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,                     # disabled: FP16 scaler conflicts with 4-bit quant
        bf16=False,                     # set True if you have an Ampere+ GPU (RTX 30xx/40xx)
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        save_total_limit=2,             # keep only the 2 best checkpoints
        load_best_model_at_end=False,          # disabled: load_adapter triggers float8 bug on this PEFT version
        report_to="none",               # disable wandb / tensorboard by default
        dataloader_num_workers=0,
        remove_unused_columns=False,
        group_by_length=True,           # pack similar-length sequences → less padding
    )

    # ------------------------------------------------------------------
    # 7. Data collator
    # ------------------------------------------------------------------
    # DataCollatorForSeq2Seq pads variable-length sequences in each batch
    # and respects label=-100 masking set during tokenisation.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # ------------------------------------------------------------------
    # 8. Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LoggingCallback()],
    )

    # ------------------------------------------------------------------
    # 9. Train
    # ------------------------------------------------------------------
    logger.info("Starting training...")
    train_result = trainer.train()

    logger.info("Training complete.")
    logger.info("  train_loss   : %.4f", train_result.training_loss)
    logger.info("  train_samples: %d",   train_result.metrics.get("train_samples", 0))
    logger.info("  train_runtime: %.1fs", train_result.metrics.get("train_runtime", 0))

    # ------------------------------------------------------------------
    # 10. Save LoRA adapters (not the full model)
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving LoRA adapters to: %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training config alongside the weights for reproducibility
    config_path = out_dir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Training config saved to: %s", config_path)

    logger.info("Done. Adapter files written:")
    for p in sorted(out_dir.iterdir()):
        logger.info("  %s  (%d bytes)", p.name, p.stat().st_size)


# ---------------------------------------------------------------------------
# Inference helper (loaded after training, used at serving time)
# ---------------------------------------------------------------------------

def load_for_inference(
    base_model_name: str,
    adapter_dir: str,
    load_in_4bit: bool = True,
) -> tuple:
    """
    Load the base model + LoRA adapter for inference.

    Returns
    -------
    (model, tokenizer)

    Example
    -------
    model, tokenizer = load_for_inference(
        "mistralai/Mistral-7B-Instruct-v0.2",
        "trained_models/critic_qlora",
    )
    critique = generate_critique(model, tokenizer, query, answer)
    """
    logger.info("Loading base model for inference: %s", base_model_name)
    tokenizer = load_tokenizer(base_model_name)
    base      = load_base_model(base_model_name, load_in_4bit=load_in_4bit)

    logger.info("Merging LoRA adapter from: %s", adapter_dir)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_critique(
    model,
    tokenizer,
    query: str,
    answer: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Run the fine-tuned critic on a (query, answer) pair.

    Parameters
    ----------
    model, tokenizer:
        From ``load_for_inference()``.
    query:
        The user's original question.
    answer:
        The answer to be evaluated.
    max_new_tokens:
        Maximum tokens in the generated critique JSON.
    temperature:
        Low temperature (0.1) produces deterministic, structured output.
        Increase to ~0.3 for more varied feedback.

    Returns
    -------
    dict
        Parsed JSON critique.  Falls back to {"raw": text} on parse error.

    Example
    -------
    critique = generate_critique(
        model, tokenizer,
        query="What is backpropagation?",
        answer="Backpropagation uses gradient descent to...",
    )
    print(critique["score"])   # 7.5
    print(critique["verdict"]) # "good"
    """
    prompt = build_prompt(query, answer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,     # slight penalty prevents JSON repetition loops
        )

    # Decode only the newly generated tokens (strip the input prompt)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    raw_text   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse the JSON output
    try:
        # Extract the first {...} block in case there is extra text
        match = __import__("re").search(r"\{[\s\S]*\}", raw_text)
        return json.loads(match.group(0)) if match else {"raw": raw_text}
    except (json.JSONDecodeError, AttributeError):
        logger.warning("generate_critique: could not parse JSON. Returning raw text.")
        return {"raw": raw_text}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QLoRA fine-tuning of Mistral-7B as a structured critic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument(
        "--data-path", type=str, default="data/critic_train.jsonl",
        help="Path to JSONL (or JSON array) training file.",
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap training samples (useful for smoke-tests).",
    )

    # Model
    p.add_argument(
        "--base-model", type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace model ID for the base causal LM.",
    )
    p.add_argument(
        "--no-4bit", dest="load_in_4bit", action="store_false", default=True,
        help="Disable 4-bit quantisation (needed on CPU-only machines).",
    )

    # LoRA
    p.add_argument("--lora-r",       type=int,   default=16,   help="LoRA rank.")
    p.add_argument("--lora-alpha",   type=int,   default=32,   help="LoRA alpha scaling.")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate.")

    # Training
    p.add_argument("--epochs",        type=int,   default=2,    help="Training epochs.")
    p.add_argument("--batch-size",    type=int,   default=2,    help="Per-device batch size.")
    p.add_argument("--grad-accum",    type=int,   default=8,    help="Gradient accumulation steps.")
    p.add_argument("--learning-rate", type=float, default=2e-4, help="Peak learning rate.")
    p.add_argument("--max-length",    type=int,   default=1024, help="Max tokenised sequence length.")

    # Output
    p.add_argument(
        "--output-dir", type=str, default="trained_models/critic_qlora",
        help="Directory where LoRA adapter weights are saved.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)


# ---------------------------------------------------------------------------
# Quick-start inference example (run as a module after training)
# ---------------------------------------------------------------------------
# To test inference after training:
#
#   python - << 'EOF'
#   from scripts.finetune_critic_qlora import load_for_inference, generate_critique
#
#   model, tokenizer = load_for_inference(
#       base_model_name="mistralai/Mistral-7B-Instruct-v0.2",
#       adapter_dir="trained_models/critic_qlora",
#   )
#
#   result = generate_critique(
#       model, tokenizer,
#       query="What is backpropagation?",
#       answer=(
#           "Backpropagation is an algorithm used to train neural networks. "
#           "It computes gradients by applying the chain rule backwards "
#           "through the network layers, starting from the loss."
#       ),
#   )
#   import json
#   print(json.dumps(result, indent=2))
#   EOF
