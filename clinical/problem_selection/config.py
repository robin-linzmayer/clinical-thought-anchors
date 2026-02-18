import argparse
from pathlib import Path

# Model configuration (matching paper Section 2.1)
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
NUM_TRIALS = 10
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_NEW_TOKENS = 8192
SEED = 44

# Selection band
MIN_ACCURACY = 0.25
MAX_ACCURACY = 0.75

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "med_qa_diagnostic_reasoning" / "train.parquet"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "problem_selection"

# Prompt template (aligned with generate_rollouts.py:447 style)
PROMPT_TEMPLATE = (
    "Solve this diagnostic problem step by step. You MUST put your final diagnosis "
    "in \\boxed{{}}. Problem: {question} Diagnosis: \n<think>\n"
)


def get_parser(description="Problem selection pipeline"):
    """Create argument parser with common options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--num_trials", type=int, default=NUM_TRIALS, help="Trials per question")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=TOP_P, help="Top-p sampling")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="Max new tokens")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--dataset_path", type=str, default=str(DATASET_PATH), help="Path to parquet file")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--min_accuracy", type=float, default=MIN_ACCURACY, help="Min accuracy for selection")
    parser.add_argument("--max_accuracy", type=float, default=MAX_ACCURACY, help="Max accuracy for selection")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--include_problems", type=str, default=None, help="Comma-separated problem indices to include")
    return parser
