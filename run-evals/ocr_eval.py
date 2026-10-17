#!/usr/bin/env python3

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

from sacrebleu.metrics import CHRF

# ============================================================
# Text normalization
# ============================================================


def normalize_text(text: str) -> str:
    """
    Normalization used for normalized OCR metrics.

    Current behavior:
    - Unicode NFKC normalization
    - Lowercase
    - Normalize whitespace
    - Strip leading/trailing whitespace

    You can optionally remove punctuation here if your OCR
    evaluation should ignore punctuation differences.
    """
    if text is None:
        return ""

    text = str(text)

    # Normalize Unicode representations
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Collapse all whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# Levenshtein distance
# ============================================================


def levenshtein_distance(reference, hypothesis):
    """
    Generic Levenshtein distance.

    Works on:
    - strings -> character distance
    - lists   -> word/token distance
    """
    n = len(reference)
    m = len(hypothesis)

    # Save memory by keeping only two rows.
    previous = list(range(m + 1))

    for i in range(1, n + 1):
        current = [i] + [0] * m

        for j in range(1, m + 1):
            substitution_cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1

            current[j] = min(
                previous[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                previous[j - 1] + substitution_cost,
            )

        previous = current

    return previous[m]


# ============================================================
# OCR metrics
# ============================================================


def calculate_cer(truth: str, predicted: str) -> float:
    """
    Character Error Rate:
        CER = character edit distance / number of reference characters
    """
    if len(truth) == 0:
        return 0.0 if len(predicted) == 0 else 1.0

    distance = levenshtein_distance(truth, predicted)
    return distance / len(truth)


def calculate_wer(truth: str, predicted: str) -> float:
    """
    Word Error Rate:
        WER = word-level edit distance / number of reference words
    """
    truth_words = truth.split()
    predicted_words = predicted.split()

    if len(truth_words) == 0:
        return 0.0 if len(predicted_words) == 0 else 1.0

    distance = levenshtein_distance(truth_words, predicted_words)
    return distance / len(truth_words)


def normalized_edit_similarity(truth: str, predicted: str) -> float:
    """
    Similarity based on Levenshtein distance.

    1.0 = identical
    0.0 = maximally different under this normalization
    """
    max_len = max(len(truth), len(predicted))

    if max_len == 0:
        return 1.0

    distance = levenshtein_distance(truth, predicted)

    return max(0.0, 1.0 - (distance / max_len))


def word_precision_recall_f1(truth: str, predicted: str):
    """
    Bag-of-words precision, recall and F1.

    Repeated words are handled using Counter intersection.
    """
    truth_words = truth.split()
    predicted_words = predicted.split()

    truth_counter = Counter(truth_words)
    pred_counter = Counter(predicted_words)

    matching = sum((truth_counter & pred_counter).values())

    if len(predicted_words) == 0:
        precision = 1.0 if len(truth_words) == 0 else 0.0
    else:
        precision = matching / len(predicted_words)

    if len(truth_words) == 0:
        recall = 1.0 if len(predicted_words) == 0 else 0.0
    else:
        recall = matching / len(truth_words)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


# ============================================================
# Input parsing
# ============================================================


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1:
    # {
    #   "truth": "...",
    #   "predicted": "..."
    # }
    if isinstance(data, dict) and "truth" in data and "predicted" in data:
        return [data]

    # Case 2:
    # [
    #   {"truth": "...", "predicted": "..."},
    #   ...
    # ]
    if isinstance(data, list):
        return data

    # Case 3:
    # {
    #   "data": [
    #       {"truth": "...", "predicted": "..."}
    #   ]
    # }
    if isinstance(data, dict) and "data" in data:
        if isinstance(data["data"], list):
            return data["data"]

    raise ValueError(
        "JSON must contain either:\n"
        '  {"truth": "...", "predicted": "..."}\n'
        "or\n"
        '  [{"truth": "...", "predicted": "..."}, ...]\n'
        "or\n"
        '  {"data": [{"truth": "...", "predicted": "..."}, ...]}'
    )


# ============================================================
# Dataset evaluation
# ============================================================


def evaluate(records):
    chrf_metric = CHRF()

    total_char_distance = 0
    total_truth_chars = 0

    total_word_distance = 0
    total_truth_words = 0

    total_norm_char_distance = 0
    total_norm_truth_chars = 0

    total_norm_word_distance = 0
    total_norm_truth_words = 0

    exact_matches = 0
    normalized_exact_matches = 0

    total_chrf = 0.0
    total_edit_similarity = 0.0

    total_word_precision = 0.0
    total_word_recall = 0.0
    total_word_f1 = 0.0

    total_pred_chars = 0
    total_pred_words = 0

    valid_records = 0

    for index, item in enumerate(records):
        if "truth" not in item or "predicted" not in item:
            raise ValueError(f"Record {index} is missing 'truth' or 'predicted'.")

        truth = str(item["truth"] if item["truth"] is not None else "")
        predicted = str(item["predicted"] if item["predicted"] is not None else "")

        norm_truth = normalize_text(truth)
        norm_predicted = normalize_text(predicted)

        # ----------------------------------------------------
        # Strict character-level
        # ----------------------------------------------------

        char_distance = levenshtein_distance(truth, predicted)

        total_char_distance += char_distance
        total_truth_chars += len(truth)
        total_pred_chars += len(predicted)

        # ----------------------------------------------------
        # Strict word-level
        # ----------------------------------------------------

        truth_words = truth.split()
        pred_words = predicted.split()

        word_distance = levenshtein_distance(truth_words, pred_words)

        total_word_distance += word_distance
        total_truth_words += len(truth_words)
        total_pred_words += len(pred_words)

        # ----------------------------------------------------
        # Normalized metrics
        # ----------------------------------------------------

        normalized_char_distance = levenshtein_distance(norm_truth, norm_predicted)

        total_norm_char_distance += normalized_char_distance
        total_norm_truth_chars += len(norm_truth)

        normalized_word_distance = levenshtein_distance(
            norm_truth.split(), norm_predicted.split()
        )

        total_norm_word_distance += normalized_word_distance
        total_norm_truth_words += len(norm_truth.split())

        # ----------------------------------------------------
        # Exact match
        # ----------------------------------------------------

        if truth == predicted:
            exact_matches += 1

        if norm_truth == norm_predicted:
            normalized_exact_matches += 1

        # ----------------------------------------------------
        # ChrF
        #
        # SacreBLEU returns score in 0-100.
        # ----------------------------------------------------

        chrf_score = chrf_metric.sentence_score(predicted, [truth]).score

        total_chrf += chrf_score

        # ----------------------------------------------------
        # Edit similarity
        # ----------------------------------------------------

        total_edit_similarity += normalized_edit_similarity(norm_truth, norm_predicted)

        # ----------------------------------------------------
        # Token metrics
        # ----------------------------------------------------

        precision, recall, f1 = word_precision_recall_f1(norm_truth, norm_predicted)

        total_word_precision += precision
        total_word_recall += recall
        total_word_f1 += f1

        valid_records += 1

    if valid_records == 0:
        raise ValueError("No records found in input JSON.")

    # ========================================================
    # Corpus-level metrics
    # ========================================================

    cer = total_char_distance / total_truth_chars if total_truth_chars > 0 else 0.0

    wer = total_word_distance / total_truth_words if total_truth_words > 0 else 0.0

    normalized_cer = (
        total_norm_char_distance / total_norm_truth_chars
        if total_norm_truth_chars > 0
        else 0.0
    )

    normalized_wer = (
        total_norm_word_distance / total_norm_truth_words
        if total_norm_truth_words > 0
        else 0.0
    )

    result = {
        "samples": valid_records,
        "truth_characters": total_truth_chars,
        "predicted_characters": total_pred_chars,
        "truth_words": total_truth_words,
        "predicted_words": total_pred_words,
        "character_edit_distance": total_char_distance,
        "word_edit_distance": total_word_distance,
        "cer": cer,
        "character_accuracy": max(0.0, 1.0 - cer),
        "wer": wer,
        "word_accuracy": max(0.0, 1.0 - wer),
        "normalized_cer": normalized_cer,
        "normalized_character_accuracy": max(0.0, 1.0 - normalized_cer),
        "normalized_wer": normalized_wer,
        "normalized_word_accuracy": max(0.0, 1.0 - normalized_wer),
        "exact_match": exact_matches / valid_records,
        "normalized_exact_match": normalized_exact_matches / valid_records,
        "chrf": total_chrf / valid_records,
        "normalized_edit_similarity": total_edit_similarity / valid_records,
        "word_precision": total_word_precision / valid_records,
        "word_recall": total_word_recall / valid_records,
        "word_f1": total_word_f1 / valid_records,
    }

    return result


# ============================================================
# CSV
# ============================================================


def append_to_csv(csv_path, model_name, input_file, metrics):
    csv_path = Path(csv_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "input_file": str(input_file),
        **metrics,
    }

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists or csv_path.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)

    return row


# ============================================================
# Pretty printing
# ============================================================


def print_results(model_name, result):
    print("\n" + "=" * 60)
    print(f"OCR Evaluation: {model_name}")
    print("=" * 60)

    print(f"Samples                 : {result['samples']}")
    print()

    print(f"CER                     : {result['cer']:.4f}")
    print(f"Character Accuracy      : " f"{result['character_accuracy']:.4f}")

    print(f"WER                     : {result['wer']:.4f}")
    print(f"Word Accuracy           : " f"{result['word_accuracy']:.4f}")

    print()

    print(f"Normalized CER          : " f"{result['normalized_cer']:.4f}")

    print(f"Normalized WER          : " f"{result['normalized_wer']:.4f}")

    print()

    print(f"Exact Match             : " f"{result['exact_match']:.4f}")

    print(f"Normalized Exact Match  : " f"{result['normalized_exact_match']:.4f}")

    print(f"ChrF                    : " f"{result['chrf']:.2f}")

    print(f"Edit Similarity         : " f"{result['normalized_edit_similarity']:.4f}")

    print()

    print(f"Word Precision          : " f"{result['word_precision']:.4f}")

    print(f"Word Recall             : " f"{result['word_recall']:.4f}")

    print(f"Word F1                 : " f"{result['word_f1']:.4f}")

    print("=" * 60)


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR output against ground truth."
    )

    parser.add_argument("--input", required=True, help="Path to JSON evaluation file.")

    parser.add_argument(
        "--model-name", required=True, help="Name of the OCR/model being evaluated."
    )

    parser.add_argument(
        "--csv",
        default="ocr_results.csv",
        help=(
            "Parent CSV where each evaluation run is appended. "
            "Default: ocr_results.csv"
        ),
    )

    args = parser.parse_args()

    records = load_json(args.input)

    result = evaluate(records)

    print_results(args.model_name, result)

    append_to_csv(
        csv_path=args.csv,
        model_name=args.model_name,
        input_file=args.input,
        metrics=result,
    )

    print(f"\nResults appended to: {args.csv}")


if __name__ == "__main__":
    main()
