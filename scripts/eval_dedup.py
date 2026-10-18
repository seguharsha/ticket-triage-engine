"""Duplicate-detection evaluation: precision/recall across thresholds.

Runs the service's exact grouping rule (app.services.dedup.group_online) over each split in time
order. The threshold is chosen on TRAIN (highest recall with precision >= target), then reported on
TEST, which is never used for tuning.

  python -m scripts.eval_dedup
Outputs reports/dedup_eval.json and reports/dedup_pr_curve.png; set DEDUP_THRESHOLD in .env to the pick.
"""

import argparse
from datetime import timedelta

import numpy as np

from app.config import settings
from app.services.dedup import group_online
from app.services.embeddings import get_embedder
from scripts.common import REPORTS, embed_rows, load_tickets, parse_time, try_pyplot, write_json
from scripts.evaluation import pair_metrics, pick_dedup_threshold


def sweep(rows: list[dict], thresholds: np.ndarray, window: timedelta) -> list[dict]:
    E = embed_rows(rows)
    times = [parse_time(r["created_at"]) for r in rows]
    truth = [r["duplicate_group_id"] or r["id"] for r in rows]  # standalone tickets are their own group
    results = []
    for t in thresholds:
        pred = group_online(E, times, float(t), window)
        results.append({"threshold": round(float(t), 3), **pair_metrics(pred, truth),
                        "incidents": len(set(pred))})
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-precision", type=float, default=0.90)
    p.add_argument("--window-hours", type=float, default=settings.dedup_window_hours)
    p.add_argument("--start", type=float, default=0.40)
    p.add_argument("--stop", type=float, default=0.96)
    p.add_argument("--step", type=float, default=0.02)
    args = p.parse_args()

    thresholds = np.arange(args.start, args.stop, args.step)
    window = timedelta(hours=args.window_hours)
    train = sweep(load_tickets("train"), thresholds, window)
    test = sweep(load_tickets("test"), thresholds, window)
    chosen = pick_dedup_threshold(train, args.min_precision)
    at_test = next(s for s in test if s["threshold"] == chosen["threshold"])

    print(f"{'thr':>6} {'P(train)':>9} {'R(train)':>9} {'P(test)':>8} {'R(test)':>8}")
    for a, b in zip(train, test):
        mark = "  <- chosen" if a["threshold"] == chosen["threshold"] else ""
        print(f"{a['threshold']:6.2f} {a['precision']:9.3f} {a['recall']:9.3f} "
              f"{b['precision']:8.3f} {b['recall']:8.3f}{mark}")
    print(f"\nchosen threshold {chosen['threshold']} (train P={chosen['precision']:.3f} R={chosen['recall']:.3f})")
    print(f"TEST: precision {at_test['precision']:.3f}  recall {at_test['recall']:.3f}  F1 {at_test['f1']:.3f}")
    print(f"Set DEDUP_THRESHOLD={chosen['threshold']} in .env")

    write_json(REPORTS / "dedup_eval.json", {
        "embedding_model": get_embedder().version, "window_hours": args.window_hours,
        "min_precision_target": args.min_precision, "chosen_threshold": chosen["threshold"],
        "test_at_chosen": at_test, "train_sweep": train, "test_sweep": test,
    })

    plt = try_pyplot()
    if plt:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for name, data, style in (("train", train, "--"), ("test", test, "-")):
            ax.plot([s["recall"] for s in data], [s["precision"] for s in data], style, marker=".", label=name)
        ax.scatter([at_test["recall"]], [at_test["precision"]], color="black", zorder=3,
                   label=f"chosen {chosen['threshold']} (test)")
        ax.axhline(args.min_precision, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("recall (duplicate pairs found)")
        ax.set_ylabel("precision (grouped pairs that are true duplicates)")
        ax.set_title(f"Duplicate detection: precision vs recall\n{get_embedder().version}, threshold sweep")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(REPORTS / "dedup_pr_curve.png", dpi=150)
        print(f"chart -> {REPORTS / 'dedup_pr_curve.png'}")


if __name__ == "__main__":
    main()
