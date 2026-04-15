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
    return [
        # AI / ML
        "Explain backpropagation in neural networks",
        "What is overfitting and how to prevent it",
        "Difference between supervised and unsupervised learning",
        "What is gradient descent",
        "Explain the transformer architecture",
        "What is the self-attention mechanism in transformers",
        "How does a convolutional neural network work",
        "What is batch normalization and why is it used",
        "Explain the bias-variance tradeoff",
        "What is reinforcement learning from human feedback (RLHF)",
        "Explain knowledge distillation in deep learning",
        "What is a variational autoencoder (VAE)",
        "Difference between L1 and L2 regularization",
        "What is the vanishing gradient problem",
        "How does the Adam optimizer work",
        "What is contrastive learning",
        "Explain fine-tuning vs prompting in large language models",
        "What is dropout and how does it prevent overfitting",
        "How does a recurrent neural network differ from a feedforward network",
        "What is transfer learning",

        # Computer Science
        "What is a hash table and how does it handle collisions",
        "Explain the time complexity of quicksort",
        "Difference between a process and a thread",
        "What is a REST API",
        "Explain how DNS works",
        "What is a binary search tree",
        "Explain the CAP theorem",
        "What is the difference between TCP and UDP",
        "How does garbage collection work in Python",
        "What is a deadlock and how can it be prevented",
        "Explain the concept of virtual memory",
        "What is a microservices architecture",
        "Difference between SQL and NoSQL databases",
        "What is a Bloom filter",
        "Explain consistent hashing",

        # General Science
        "Why is the sky blue",
        "How does photosynthesis work",
        "What causes earthquakes",
        "Explain the theory of relativity simply",
        "What is quantum computing",
        "How does CRISPR gene editing work",
        "What is the difference between fission and fusion",
        "How does the immune system fight viruses",
        "What is entropy in thermodynamics",
        "Explain how vaccines work",

        # Programming / Coding
        "Write a Python function for binary search",
        "Explain recursion with an example",
        "What is dynamic programming",
        "Explain the four principles of object-oriented programming",
        "What is a decorator in Python",
        "Explain the difference between mutable and immutable objects",
        "What is a closure in programming",
        "How does async/await work in Python",
        "What is the difference between a stack and a queue",
        "Explain big-O notation with examples",
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
        good_output = generator.generate(query)
        good_answer = good_output.answer

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
