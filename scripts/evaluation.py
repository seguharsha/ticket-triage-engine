"""Metric functions used by the eval scripts (pure; unit-tested)."""

from collections import Counter
from math import comb


def pair_metrics(predicted: list, truth: list) -> dict:
    """Pairwise precision/recall: a pair is positive when both tickets share a group.

    precision = of the pairs we grouped together, how many are true duplicates
    recall    = of the true duplicate pairs, how many we grouped together
    """
    together = sum(comb(n, 2) for n in Counter(predicted).values())
    true_pairs = sum(comb(n, 2) for n in Counter(truth).values())
    correct = sum(comb(n, 2) for n in Counter(zip(predicted, truth)).values())
    precision = correct / together if together else 1.0
    recall = correct / true_pairs if true_pairs else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "predicted_pairs": together, "true_pairs": true_pairs}


def pick_dedup_threshold(sweep: list[dict], min_precision: float) -> dict:
    """Highest recall among thresholds meeting the precision target; else best F1."""
    ok = [s for s in sweep if s["precision"] >= min_precision]
    if ok:
        return max(ok, key=lambda s: (s["recall"], -s["threshold"]))
    return max(sweep, key=lambda s: s["f1"])


def routing_sweep(confidences: list[float], correct: list[bool], thresholds: list[float]) -> list[dict]:
    """For each threshold: share auto-routed, and accuracy among auto-routed tickets."""
    out = []
    n = len(confidences)
    for t in thresholds:
        auto = [c for conf, c in zip(confidences, correct) if conf >= t]
        out.append({
            "threshold": round(t, 3),
            "auto_share": len(auto) / n if n else 0.0,
            "auto_accuracy": sum(auto) / len(auto) if auto else 1.0,
            "auto_count": len(auto),
        })
    return out


def pick_route_threshold(sweep: list[dict], min_accuracy: float) -> dict:
    """Largest auto-routed share whose accuracy meets the target; else the most accurate."""
    ok = [s for s in sweep if s["auto_accuracy"] >= min_accuracy and s["auto_count"] > 0]
    if ok:
        return max(ok, key=lambda s: (s["auto_share"], -s["threshold"]))
    return max(sweep, key=lambda s: s["auto_accuracy"])
