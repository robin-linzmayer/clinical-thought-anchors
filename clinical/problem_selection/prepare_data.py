"""Load med_qa_diagnostic_reasoning parquet and prepare questions for inference."""

import json
import pandas as pd
from pathlib import Path
from config import get_parser


def main():
    parser = get_parser("Prepare medical QA questions")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load parquet
    print(f"Loading dataset from {args.dataset_path}")
    df = pd.read_parquet(args.dataset_path)
    print(f"Loaded {len(df)} rows")

    # Drop rows with null answers
    before = len(df)
    df = df.dropna(subset=["answer"])
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows with null answers")

    # Build question list
    questions = []
    for idx, row in df.iterrows():
        # Clean answer: strip whitespace, newlines, and stray quotes
        answer = row["answer"].strip().strip('"').strip()
        questions.append({
            "problem_idx": int(idx),
            "question": row["question"],
            "gt_answer": answer,
        })

    # Save
    output_path = output_dir / "prepared_questions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(questions)} questions to {output_path}")


if __name__ == "__main__":
    main()
