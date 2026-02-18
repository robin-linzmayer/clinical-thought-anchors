"""Evaluate inference results, visualize accuracy distribution, and select problems."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import get_parser


def load_results(inference_dir: Path):
    """Load all per-problem results from inference directory."""
    problems = []
    for problem_dir in sorted(inference_dir.iterdir()):
        if not problem_dir.is_dir() or not problem_dir.name.startswith("problem_"):
            continue
        results_file = problem_dir / "results.json"
        if not results_file.exists():
            continue
        with open(results_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        problems.append({
            "problem_idx": saved["problem_idx"],
            "question": saved["question"],
            "gt_answer": saved["gt_answer"],
            "trials": saved["trials"],
        })
    return problems


def compute_metrics(problems):
    """Compute per-problem accuracy and token stats."""
    metrics = []
    for p in problems:
        trials = p["trials"]
        num_correct = sum(1 for t in trials if t["is_correct"])
        accuracy = num_correct / len(trials) if trials else 0.0
        mean_tokens = np.mean([t["num_tokens"] for t in trials]) if trials else 0.0

        correct_answers = list(set(
            t["extracted_answer"] for t in trials if t["is_correct"] and t["extracted_answer"]
        ))
        incorrect_answers = list(set(
            t["extracted_answer"] for t in trials if not t["is_correct"] and t["extracted_answer"]
        ))

        metrics.append({
            "problem_idx": p["problem_idx"],
            "question": p["question"],
            "gt_answer": p["gt_answer"],
            "accuracy": accuracy,
            "num_correct": num_correct,
            "num_trials": len(trials),
            "mean_tokens": float(mean_tokens),
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
        })
    return metrics


def plot_accuracy_distribution(metrics, output_dir, min_acc, max_acc):
    """Histogram of accuracy with selection band highlighted."""
    accuracies = [m["accuracy"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(accuracies, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvspan(min_acc, max_acc, alpha=0.2, color="green", label=f"Selection band [{min_acc}, {max_acc}]")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Number of problems")
    ax.set_title("Accuracy Distribution Across Medical QA Questions")
    ax.legend()

    path = output_dir / "accuracy_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_accuracy_vs_tokens(metrics, output_dir, min_acc, max_acc):
    """Scatter plot of accuracy vs mean tokens, colored by selection band."""
    accuracies = [m["accuracy"] for m in metrics]
    tokens = [m["mean_tokens"] for m in metrics]
    in_band = [min_acc <= m["accuracy"] <= max_acc for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (a, t, ib) in enumerate(zip(accuracies, tokens, in_band)):
        color = "green" if ib else "gray"
        ax.scatter(a, t, c=color, alpha=0.6, s=20)

    # Legend
    ax.scatter([], [], c="green", alpha=0.6, s=20, label="In selection band")
    ax.scatter([], [], c="gray", alpha=0.6, s=20, label="Outside band")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Mean tokens per trial")
    ax.set_title("Accuracy vs Token Count")
    ax.legend()

    path = output_dir / "accuracy_vs_tokens.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def select_problems(metrics, args):
    """Select problems within accuracy band, with fallback widening."""
    min_acc = args.min_accuracy
    max_acc = args.max_accuracy

    # Filter to band
    selected = [m for m in metrics if min_acc <= m["accuracy"] <= max_acc]
    print(f"Found {len(selected)} problems in [{min_acc}, {max_acc}] band")

    # Widen band if too few
    while len(selected) < args.min_problems and (min_acc > 0 or max_acc < 1):
        min_acc = max(0.0, min_acc - 0.05)
        max_acc = min(1.0, max_acc + 0.05)
        selected = [m for m in metrics if min_acc <= m["accuracy"] <= max_acc]
        print(f"Widened band to [{min_acc:.2f}, {max_acc:.2f}]: {len(selected)} problems")

    # If too many, prefer problems closest to 50%
    if len(selected) > args.max_problems:
        selected.sort(key=lambda m: abs(m["accuracy"] - 0.5))
        selected = selected[:args.max_problems]
        print(f"Trimmed to {len(selected)} problems closest to 50% accuracy")

    # Build output in format compatible with existing selected_problems.json
    output = []
    for m in sorted(selected, key=lambda x: x["problem_idx"]):
        output.append({
            "problem_idx": f"problem_{m['problem_idx']}",
            "problem": m["question"],
            "gt_answer": m["gt_answer"],
            "accuracy": m["accuracy"],
            "approximate_mean_tokens": m["mean_tokens"],
            "incorrect_answers": m["incorrect_answers"],
            "correct_answers": m["correct_answers"],
        })

    return output


def main():
    parser = get_parser("Evaluate and select problems")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Find inference results
    model_name = args.model.split("/")[-1]
    inference_dir = output_dir / "inference" / model_name
    if not inference_dir.exists():
        print(f"Inference directory not found: {inference_dir}. Run run_inference.py first.")
        return

    # Load and compute metrics
    problems = load_results(inference_dir)
    print(f"Loaded results for {len(problems)} problems")

    metrics = compute_metrics(problems)

    # Print summary
    accuracies = [m["accuracy"] for m in metrics]
    print(f"Accuracy: mean={np.mean(accuracies):.3f}, "
          f"median={np.median(accuracies):.3f}, "
          f"std={np.std(accuracies):.3f}")
    in_band = sum(1 for a in accuracies if args.min_accuracy <= a <= args.max_accuracy)
    print(f"Problems in [{args.min_accuracy}, {args.max_accuracy}] band: {in_band}")

    # Generate plots
    plot_accuracy_distribution(metrics, output_dir, args.min_accuracy, args.max_accuracy)
    plot_accuracy_vs_tokens(metrics, output_dir, args.min_accuracy, args.max_accuracy)

    # Select problems
    selected = select_problems(metrics, args)

    # Save
    selected_path = output_dir / "selected_problems.json"
    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(selected)} selected problems to {selected_path}")

    # Also save full metrics for reference
    metrics_path = output_dir / "all_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved all metrics to {metrics_path}")


if __name__ == "__main__":
    main()
