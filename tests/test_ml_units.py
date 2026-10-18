"""Embeddings, dedup grouping rule, classifier, metrics and eval helpers (no database)."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from app.metrics import format_metrics, quantile_from_histogram
from app.services.classifier import EmbeddingClassifier
from app.services.dedup import group_online
from app.services.embeddings import HashingEmbedder
from scripts.evaluation import pair_metrics, pick_dedup_threshold, pick_route_threshold, routing_sweep

T0 = datetime(2026, 10, 20, 10, 0)


def test_hashing_embeddings_are_unit_length_and_similarity_tracks_overlap():
    e = HashingEmbedder().embed([
        "sso login loops back to the sign in page",
        "SSO login keeps looping back to sign in page",
        "please send me a VAT invoice for october",
    ])
    assert e.shape == (3, 384)
    assert np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-5)
    assert e[0] @ e[1] > 0.5 > e[0] @ e[2]


def unit(*vecs):
    a = np.asarray(vecs, dtype=float)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_group_online_joins_nearest_above_threshold():
    E = unit([1, 0], [0.99, 0.1], [0, 1])
    labels = group_online(E, [T0, T0 + timedelta(minutes=1), T0 + timedelta(minutes=2)], 0.9, timedelta(hours=72))
    assert labels[0] == labels[1] != labels[2]


def test_group_online_respects_time_window():
    E = unit([1, 0], [1, 0])
    labels = group_online(E, [T0, T0 + timedelta(hours=100)], 0.9, timedelta(hours=72))
    assert labels[0] != labels[1]


def test_group_online_processes_in_time_order_and_chains():
    # b is similar to a, c to b but not to a: single-link chains them through b.
    E = unit([1, 0], [0.8, 0.6], [0.28, 0.96])
    times = [T0, T0 + timedelta(minutes=1), T0 + timedelta(minutes=2)]
    assert len(set(group_online(E, times, 0.79, timedelta(hours=1)))) == 1
    # Shuffled input order gives the same grouping, since time decides processing order.
    assert len(set(group_online(E[[2, 0, 1]], [times[2], times[0], times[1]], 0.79, timedelta(hours=1)))) == 1


def test_pair_metrics():
    m = pair_metrics(predicted=[0, 0, 0, 1], truth=["a", "a", "b", "b"])
    # predicted pairs: (0,1) (0,2) (1,2) -> only (0,1) true; true pairs: (0,1) (2,3)
    assert m["precision"] == pytest.approx(1 / 3) and m["recall"] == pytest.approx(1 / 2)


def test_pair_metrics_perfect_and_empty():
    assert pair_metrics([0, 0, 1], ["x", "x", "y"])["f1"] == 1.0
    assert pair_metrics([0, 1], ["x", "y"])["precision"] == 1.0


def test_pick_dedup_threshold_prefers_recall_within_precision_target():
    sweep = [
        {"threshold": 0.6, "precision": 0.80, "recall": 0.95, "f1": 0.87},
        {"threshold": 0.7, "precision": 0.91, "recall": 0.85, "f1": 0.88},
        {"threshold": 0.8, "precision": 0.97, "recall": 0.70, "f1": 0.81},
    ]
    assert pick_dedup_threshold(sweep, 0.9)["threshold"] == 0.7
    assert pick_dedup_threshold(sweep, 0.99)["threshold"] == 0.7  # none qualify -> best F1


def test_routing_sweep_and_pick():
    sweep = routing_sweep([0.9, 0.8, 0.6, 0.4], [True, True, False, True], [0.5, 0.7])
    assert sweep[0] == {"threshold": 0.5, "auto_share": 0.75, "auto_accuracy": 2 / 3, "auto_count": 3}
    assert sweep[1]["auto_accuracy"] == 1.0
    assert pick_route_threshold(sweep, 0.95)["threshold"] == 0.7


def test_classifier_train_predict_save_load(tmp_path):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(1, 0.1, (20, 384)), rng.normal(-1, 0.1, (20, 384))]).astype(np.float32)
    labels = {"category": ["bug"] * 20 + ["billing"] * 20,
              "priority": ["P2"] * 20 + ["P3"] * 20,
              "team": ["Product"] * 20 + ["Payments"] * 20}
    clf = EmbeddingClassifier.train(X, labels, "test-v1")
    path = tmp_path / "clf.joblib"
    clf.save(path)
    loaded = EmbeddingClassifier.load(path)
    pred = loaded.predict(X[0])
    assert loaded.version == "test-v1"
    assert pred["category"].value == "bug" and pred["team"].value == "Product"
    assert 0.5 < pred["category"].confidence <= 1.0
    assert sum(pred["category"].probabilities.values()) == pytest.approx(1.0)


def test_format_metrics_and_quantile():
    from app.metrics import BUCKETS

    values = [0.4, 0.9, 0.9, 1.5]
    stored = {repr(float(b)): sum(v <= b for v in values) for b in BUCKETS}
    stored.update({"+Inf": 4, "sum": sum(values), "count": 4})

    text = format_metrics({"triage_processing_seconds": stored}, {b"triage_tickets_processed_total": b"4"},
                          {"triage_queue_depth": 2})
    assert 'triage_processing_seconds_bucket{le="1"} 3' in text
    assert 'triage_processing_seconds_bucket{le="+Inf"} 4' in text
    assert "triage_processing_seconds_count 4" in text
    assert "triage_tickets_processed_total 4.0" in text
    assert "triage_queue_depth 2" in text
    assert quantile_from_histogram(stored, 0.5) == 1
    assert quantile_from_histogram(stored, 0.95) == 2
    assert np.isnan(quantile_from_histogram({}, 0.5))