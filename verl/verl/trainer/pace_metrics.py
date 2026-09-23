"""Classification metrics for multimodal sarcasm detection."""

from __future__ import annotations

import math
from typing import Any


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(
    gold: list[str],
    predicted: list[str],
    *,
    invalid: int,
    invalid_gold: list[str] | None = None,
) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    if invalid_gold is None:
        invalid_gold = []
    if invalid_gold and len(invalid_gold) != invalid:
        raise ValueError("invalid_gold must contain one label per invalid prediction")
    labels = ("non-sarcastic", "sarcastic")
    invalid_by_gold = {
        label: sum(value == label for value in invalid_gold) for label in labels
    }
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        fn += invalid_by_gold[label]
        precision = _divide(tp, tp + fp)
        recall = _divide(tp, tp + fn)
        f1 = _divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(g == label for g in gold) + invalid_by_gold[label],
            "predicted": sum(p == label for p in predicted),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }

    correct = sum(g == p for g, p in zip(gold, predicted))
    total = len(gold) + invalid
    valid = len(gold)
    supports = sum(int(per_class[label]["support"]) for label in labels)
    macro_precision = sum(float(per_class[label]["precision"]) for label in labels) / len(labels)
    macro_recall = sum(float(per_class[label]["recall"]) for label in labels) / len(labels)
    macro_f1 = sum(float(per_class[label]["f1"]) for label in labels) / len(labels)
    weighted_precision = _divide(
        sum(float(per_class[label]["precision"]) * int(per_class[label]["support"]) for label in labels),
        supports,
    )
    weighted_recall = _divide(
        sum(float(per_class[label]["recall"]) * int(per_class[label]["support"]) for label in labels),
        supports,
    )
    weighted_f1 = _divide(
        sum(float(per_class[label]["f1"]) * int(per_class[label]["support"]) for label in labels),
        supports,
    )

    tp = sum(g == "sarcastic" and p == "sarcastic" for g, p in zip(gold, predicted))
    tn = sum(g == "non-sarcastic" and p == "non-sarcastic" for g, p in zip(gold, predicted))
    fp = sum(g == "non-sarcastic" and p == "sarcastic" for g, p in zip(gold, predicted))
    fn_valid = sum(g == "sarcastic" and p == "non-sarcastic" for g, p in zip(gold, predicted))
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn_valid) * (tn + fp) * (tn + fn_valid))
    mcc = (tp * tn - fp * fn_valid) / mcc_denominator if mcc_denominator else 0.0

    valid_accuracy = _divide(correct, valid)
    gold_positive = sum(g == "sarcastic" for g in gold)
    predicted_positive = sum(p == "sarcastic" for p in predicted)
    expected_agreement = 0.0
    if valid:
        expected_agreement = (
            gold_positive * predicted_positive
            + (valid - gold_positive) * (valid - predicted_positive)
        ) / (valid * valid)
    kappa = _divide(valid_accuracy - expected_agreement, 1.0 - expected_agreement)

    # Wrong valid predictions contribute a micro FP and FN. Invalid JSON is an
    # abstention, so it contributes only a micro FN and lowers coverage/recall.
    wrong_valid = valid - correct
    micro_precision = _divide(correct, correct + wrong_valid)
    micro_recall = _divide(correct, correct + wrong_valid + invalid)
    micro_f1 = _divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    sarcastic = per_class["sarcastic"]
    non_sarcastic = per_class["non-sarcastic"]
    return {
        "samples": total,
        "valid_predictions": valid,
        "invalid_json": invalid,
        "prediction_coverage": _divide(valid, total),
        "accuracy": _divide(correct, total),
        "valid_accuracy": valid_accuracy,
        "precision": sarcastic["precision"],
        "recall": sarcastic["recall"],
        "f1": sarcastic["f1"],
        "specificity": non_sarcastic["recall"],
        "negative_predictive_value": non_sarcastic["precision"],
        "balanced_accuracy": macro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "matthews_correlation_coefficient_valid": mcc,
        "cohen_kappa_valid": kappa,
        "confusion_matrix_valid": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn_valid,
            "true_positive": tp,
        },
        "invalid_by_gold": invalid_by_gold,
        "per_class": per_class,
        "metric_policy": {
            "invalid_json_counts_as_incorrect": True,
            "mcc_and_kappa_scope": "valid_predictions_only",
        },
    }
