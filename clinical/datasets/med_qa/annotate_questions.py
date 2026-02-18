"""
Script to annotate MedQA questions with question type categories.

Categories:
- diagnosis: Questions asking for most likely diagnosis
- cause: Questions asking for etiology/cause
- next_step: Questions asking for next step in management
- expected_findings: Questions asking what findings to expect
- other: All other question types
"""

import json
import os
from pathlib import Path
from collections import Counter


def categorize_question(question_text: str) -> str:
    """
    Categorize a question based on its content.

    Args:
        question_text: The full question text

    Returns:
        Category string: 'diagnosis', 'cause', 'next_step', 'expected_findings', or 'other'
    """
    # Extract the question sentence (the part with ?)
    question_part = question_text.lower()
    if '?' in question_text:
        parts = question_text.split('.')
        for part in reversed(parts):
            if '?' in part:
                question_part = part.strip().lower()
                break

    # Diagnosis patterns
    diagnosis_patterns = [
        'most likely diagnosis',
        'diagnosis is most likely',
        'likely diagnosis',
        'probable diagnosis',
        'best diagnosis',
        'diagnosis of this patient',
        'diagnose this patient',
        'condition does this patient',
        'disorder does this patient',
        'disease does this patient',
        'syndrome does this patient',
        'suffering from',
        'suffer from',
    ]

    # Cause/Etiology patterns
    cause_patterns = [
        'most likely cause',
        'cause of this patient',
        'cause of the patient',
        'cause of her',
        'cause of his',
        'etiology',
        'responsible for this patient',
        'responsible for the patient',
        'underlying cause',
        'primary cause',
        'reason for this patient',
        'explains this patient',
        'explanation for this patient',
        'explanation for the patient',
        'led to this patient',
        'resulted in this patient',
        'accounts for',
    ]

    # Next step patterns
    next_step_patterns = [
        'next step',
        'next best step',
        'best next step',
        'next appropriate',
        'best initial step',
        'initial step',
        'first step',
        'immediate step',
        'next in management',
        'appropriate management',
        'best management',
        'should be done next',
        'should be performed next',
        'should be ordered next',
        'appropriate next',
        'most appropriate action',
        'best course of action',
        'best approach',
        'should be treated first',
        'treat first',
        'priority',
    ]

    # Expected findings patterns
    expected_findings_patterns = [
        'most likely to show',
        'most likely to reveal',
        'most likely to find',
        'most likely to demonstrate',
        'most likely finding',
        'expected to show',
        'expected to reveal',
        'expected to find',
        'expect to see',
        'expect to find',
        'would show',
        'would reveal',
        'would demonstrate',
        'will show',
        'will reveal',
        'likely to reveal',
        'likely to show',
        'likely finding',
        'biopsy would',
        'biopsy will',
        'biopsy of',
        'examination will show',
        'examination would show',
        'further evaluation',
        'laboratory finding',
        'lab finding',
        'physical examination finding',
        'characteristic finding',
        'most likely to be seen',
        'most likely seen',
        'histological',
        'histopathological',
        'microscopic examination',
    ]

    # Check patterns in order of specificity
    for pattern in diagnosis_patterns:
        if pattern in question_part:
            return 'diagnosis'

    for pattern in cause_patterns:
        if pattern in question_part:
            return 'cause'

    for pattern in next_step_patterns:
        if pattern in question_part:
            return 'next_step'

    for pattern in expected_findings_patterns:
        if pattern in question_part:
            return 'expected_findings'

    return 'other'


def annotate_dataset(input_path: str, origin: str) -> list:
    """
    Read and annotate a dataset file.

    Args:
        input_path: Path to the input JSONL file
        origin: Origin label ('train' or 'dev')

    Returns:
        List of annotated question dictionaries
    """
    annotated = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())

            # Add annotations
            data['origin'] = origin
            data['question_type'] = categorize_question(data['question'])

            annotated.append(data)

    return annotated


def main():
    # Setup paths
    base_dir = Path(__file__).parent
    output_dir = base_dir / 'annotated'
    output_dir.mkdir(exist_ok=True)

    # Process dev dataset
    print("Processing dev dataset...")
    dev_data = annotate_dataset(base_dir / 'dev.jsonl', 'dev')

    # Process train dataset
    print("Processing train dataset...")
    train_data = annotate_dataset(base_dir / 'train.jsonl', 'train')

    # Write annotated dev dataset
    dev_output_path = output_dir / 'dev.jsonl'
    with open(dev_output_path, 'w', encoding='utf-8') as f:
        for item in dev_data:
            f.write(json.dumps(item) + '\n')
    print(f"Wrote {len(dev_data)} questions to {dev_output_path}")

    # Write annotated train dataset
    train_output_path = output_dir / 'train.jsonl'
    with open(train_output_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    print(f"Wrote {len(train_data)} questions to {train_output_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("ANNOTATION STATISTICS")
    print("=" * 60)

    for name, data in [('Dev', dev_data), ('Train', train_data)]:
        print(f"\n{name} Dataset (n={len(data)}):")
        print("-" * 40)

        type_counts = Counter(item['question_type'] for item in data)
        for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = count / len(data) * 100
            print(f"  {qtype:20} {count:5} ({pct:5.1f}%)")

    # Combined statistics
    all_data = dev_data + train_data
    print(f"\nCombined Dataset (n={len(all_data)}):")
    print("-" * 40)

    type_counts = Counter(item['question_type'] for item in all_data)
    for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_data) * 100
        print(f"  {qtype:20} {count:5} ({pct:5.1f}%)")


if __name__ == '__main__':
    main()
