import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm

from config import get_default_config
from core.generator import Generator
from core.critic import Critic
from models.base_llm import OllamaLLM

def load_queries(file_path):
    # Depending on the use case, this could load from a text file, CSV or JSON
    # For now, we supply some default queries if path is not provided.
    return [
        "Explain the process of backpropagation in neural networks.",
        "What is the difference between supervised and unsupervised learning?",
        "Describe the self-attention mechanism in Transformers.",
        "How does a convolutional neural network (CNN) work?"
    ]

def get_corruption_prompt(original_answer):
    prompts = [
        f"Rewrite the following answer but subtly introduce a factual error about dates or inventors:\n\n{original_answer}",
        f"Rewrite the following answer but remove the most crucial part of the explanation, leaving it incomplete:\n\n{original_answer}",
        f"Rewrite the following answer and add a hallucinated reference to a non-existent scientific paper (e.g. 'Smith et al. 2019'):\n\n{original_answer}",
        f"Rewrite the following answer but make the conclusion logically contradict the premises you stated:\n\n{original_answer}",
    ]
    return random.choice(prompts)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data to train the Critic.")
    parser.add_argument("--queries_file", type=str, default=None, help="Path to text file containing queries (one per line).")
    parser.add_argument("--output_file", type=str, default="data/critic_train.json")
    parser.add_argument("--num_corruptions", type=int, default=1, help="Number of corrupted versions per query.")
    parser.add_argument("--model", type=str, default=None, help="Ollama model to use (overrides config default).")
    args = parser.parse_args()

    config = get_default_config()
    if args.model:
        config.llm.model_name = args.model
        config.generator_model = args.model
        config.critic_model = args.model
        config.refiner_model = args.model
    # We use the existing local LLM to do the heavy lifting
    llm = OllamaLLM(config.llm)

    generator = Generator(llm, config.llm)
    critic = Critic(llm, config.llm)

    # 1. Load Queries
    if args.queries_file and Path(args.queries_file).exists():
        with open(args.queries_file, "r") as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        queries = load_queries(args.queries_file)

    dataset = []

    print(f"Generating dataset for {len(queries)} queries...")
    for query in tqdm(queries):
        # 2. Generate a "Good" Answer
        good_answer = generator.generate(query)
        
        # 3. Label the "Good" Answer with Critic
        good_feedback = critic.critique(query, good_answer)
        
        dataset.append({
            "query": query,
            "answer": good_answer,
            "feedback": {
                "factual_errors": good_feedback.factual_errors,
                "hallucinations": good_feedback.hallucinations,
                "missing_concepts": good_feedback.missing_concepts,
                "logical_flaws": good_feedback.logical_flaws,
                "score": good_feedback.score,
                "verdict": good_feedback.verdict
            }
        })

        # 4. Generate Corrupted Answers (Negative Sampling)
        for _ in range(args.num_corruptions):
            corruption_instruction = get_corruption_prompt(good_answer)
            # Use the LLM simply as an editor to apply the corruption instruction
            corrupted_answer = llm.complete(corruption_instruction)
            
            # Label the corrupted answer
            corrupt_feedback = critic.critique(query, corrupted_answer)
            dataset.append({
                "query": query,
                "answer": corrupted_answer,
                "feedback": {
                    "factual_errors": corrupt_feedback.factual_errors,
                    "hallucinations": corrupt_feedback.hallucinations,
                    "missing_concepts": corrupt_feedback.missing_concepts,
                    "logical_flaws": corrupt_feedback.logical_flaws,
                    "score": corrupt_feedback.score,
                    "verdict": corrupt_feedback.verdict
                }
            })

    # Ensure output dir exists
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Dataset generated with {len(dataset)} examples at {args.output_file}")

if __name__ == "__main__":
    main()
