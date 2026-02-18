"""Answer extraction and evaluation utilities for medical QA."""

import re
import sys
from pathlib import Path

# Add project root to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import extract_boxed_answers


def extract_answer(text: str) -> str:
    """Extract the predicted diagnosis from model output.

    Primary: look for \\boxed{...} content.
    Fallback: text after </think> tag (last substantive line).
    """
    # Primary: boxed answer
    boxed = extract_boxed_answers(text)
    if boxed and boxed[0].strip():
        return boxed[0].strip()

    # Fallback: text after </think>
    if "</think>" in text:
        after_think = text.split("</think>")[-1].strip()
        # Take last non-empty line
        lines = [l.strip() for l in after_think.split("\n") if l.strip()]
        if lines:
            return lines[-1]

    # Last resort: last non-empty line of the whole output
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else ""


def normalize_answer(answer: str) -> str:
    """Normalize a medical diagnosis answer for comparison."""
    if not answer:
        return ""

    s = answer.lower().strip()

    # Remove common prefixes
    prefixes = [
        "the most likely diagnosis is",
        "the diagnosis is",
        "most likely diagnosis:",
        "diagnosis:",
        "the patient has",
        "the patient most likely has",
        "this is consistent with",
        "this is",
        "the answer is",
        "final diagnosis:",
        "final answer:",
    ]
    for prefix in prefixes:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()

    # Remove leading articles
    for article in ["a ", "an ", "the "]:
        if s.startswith(article):
            s = s[len(article):]

    # Remove parenthetical content like "(CIN)" or "(type 2)"
    s = re.sub(r"\s*\([^)]*\)", "", s)

    # Remove trailing punctuation
    s = s.rstrip(".,;:!?")

    # Remove quotation marks and asterisks (markdown bold)
    s = s.replace('"', "").replace("'", "").replace("*", "")

    # Remove \\boxed{} wrapper if still present
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def check_answer(predicted: str, ground_truth: str) -> bool:
    """Check if predicted diagnosis matches ground truth using tiered matching.

    1. Exact match after normalization
    2. Substring containment (either direction)
    3. Word overlap (Jaccard > 0.7 on content words)
    """
    norm_pred = normalize_answer(predicted)
    norm_gt = normalize_answer(ground_truth)

    if not norm_pred or not norm_gt:
        return False

    # Tier 1: Exact match
    if norm_pred == norm_gt:
        return True

    # Tier 2: Substring containment
    if norm_pred in norm_gt or norm_gt in norm_pred:
        return True

    # Tier 3: Word overlap (Jaccard similarity)
    stop_words = {"of", "the", "a", "an", "in", "on", "and", "or", "with", "to", "by", "for", "is", "at", "from"}
    pred_words = set(norm_pred.split()) - stop_words
    gt_words = set(norm_gt.split()) - stop_words

    if not pred_words or not gt_words:
        return False

    intersection = pred_words & gt_words
    union = pred_words | gt_words
    jaccard = len(intersection) / len(union)

    return jaccard > 0.7
