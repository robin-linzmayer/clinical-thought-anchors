"""Evaluate inference results using GPT-4o-mini as a judge.

Reads results.json files from inference directories, sends each
(question, gt_answer, extracted_answer) triple to GPT-4o-mini for
correctness grading, and writes is_correct back into the results files.

Usage:
    python evaluate_answers.py
    python evaluate_answers.py --include_problems 0,1,2,3,4
    python evaluate_answers.py --force  # re-evaluate already-graded trials
"""

import json
import os
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from config import get_parser

JUDGE_MODEL = "gpt-4o-mini"

JUDGE_PROMPT = """\
You are a medical expert evaluating whether a predicted diagnosis is correct.

**Question:** {question}

**Ground truth diagnosis:** {gt_answer}

**Predicted diagnosis:** {predicted}

Is the predicted diagnosis semantically equivalent to the ground truth? \
The predicted answer does not need to match word-for-word — it is correct if it \
refers to the same medical condition, even if phrased differently, abbreviated, \
or includes additional qualifiers.

Respond with ONLY "correct" or "incorrect"."""


def judge_answer(client, question, gt_answer, predicted):
    """Call GPT-4o-mini to judge whether predicted matches gt_answer."""
    if not predicted or not predicted.strip():
        return False

    prompt = JUDGE_PROMPT.format(
        question=question,
        gt_answer=gt_answer,
        predicted=predicted,
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            verdict = response.choices[0].message.content.strip().lower()
            return "correct" in verdict and "incorrect" not in verdict
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/3): {e}")
            time.sleep(2 ** attempt)

    return False


def main():
    parser = get_parser("Evaluate answers with GPT-4o-mini")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate trials that already have is_correct")
    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    client = OpenAI(api_key=api_key)

    output_dir = Path(args.output_dir)
    model_name = args.model.split("/")[-1]
    inference_dir = output_dir / "inference" / model_name

    if not inference_dir.exists():
        print(f"Inference directory not found: {inference_dir}. Run run_inference.py first.")
        return

    # Collect all problem directories
    problem_dirs = sorted(
        [d for d in inference_dir.iterdir() if d.is_dir() and d.name.startswith("problem_")]
    )

    # Filter if --include_problems specified
    if args.include_problems:
        include_set = set(int(x) for x in args.include_problems.split(","))
        problem_dirs = [d for d in problem_dirs if int(d.name.split("_")[1]) in include_set]

    print(f"Evaluating {len(problem_dirs)} problems with {JUDGE_MODEL}")

    total_trials = 0
    total_correct = 0
    total_evaluated = 0

    for problem_dir in tqdm(problem_dirs, desc="Problems"):
        results_file = problem_dir / "results.json"
        if not results_file.exists():
            continue

        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        question = data["question"]
        gt_answer = data["gt_answer"]
        trials = data["trials"]
        changed = False

        for trial in trials:
            total_trials += 1
            # Skip if already evaluated (unless --force)
            if "is_correct" in trial and not args.force:
                if trial["is_correct"]:
                    total_correct += 1
                continue

            total_evaluated += 1
            extracted = trial.get("extracted_answer", "")
            is_correct = judge_answer(client, question, gt_answer, extracted)
            trial["is_correct"] = is_correct
            changed = True

            if is_correct:
                total_correct += 1

        if changed:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        # Print per-problem summary
        n_correct = sum(1 for t in trials if t.get("is_correct"))
        problem_idx = data["problem_idx"]
        print(f"  Problem {problem_idx}: {n_correct}/{len(trials)} correct | gt: {gt_answer}")

    print(f"\nDone. Evaluated {total_evaluated} new trials across {len(problem_dirs)} problems.")
    print(f"Overall: {total_correct}/{total_trials} correct ({total_correct/total_trials:.1%})" if total_trials else "No trials found.")


if __name__ == "__main__":
    main()
