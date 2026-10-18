"""Classification + routing evaluation on the held-out TEST split.

Reports accuracy and macro-F1 per field, then the routing trade-off: for each confidence threshold,
the share of tickets auto-routed and the accuracy of those (all routing fields correct). The
threshold is chosen as the largest auto-routed share with accuracy >= target.

  python -m scripts.eval_classifier
  python -m scripts.eval_classifier --llm 200   # also run the LLM fallback on up to 200 uncertain tickets

Note: tuning the routing threshold on TEST would leak; it is chosen on a slice of TRAIN held out from
fitting (--calib-fraction) and then reported on TEST.
"""

import argparse
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from app.config import settings
from app.services.classifier import FIELDS, EmbeddingClassifier, get_classifier
from app.services.embeddings import get_embedder
from app.services.llm import extract_json, get_llm_client
from app.services.routing import classification_prompt, decide, parse_llm_labels, uncertain_fields
from scripts.common import REPORTS, embed_rows, load_tickets, texts, try_pyplot, write_json
from scripts.evaluation import pick_route_threshold, routing_sweep


def routing_confidence(pred: dict, fields: list[str]) -> float:
    return min(pred[f].confidence for f in fields)


def routing_correct(pred: dict, row: dict, fields: list[str]) -> bool:
    return all(pred[f].value == row[f] for f in fields)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-accuracy", type=float, default=0.95)
    p.add_argument("--calib-fraction", type=float, default=0.2)
    p.add_argument("--llm", type=int, default=0, help="max uncertain test tickets to send to the LLM")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    fields = settings.routing_fields

    # Fit on most of TRAIN, choose the routing threshold on the rest (calibration), report on TEST.
    train = load_tickets("train")
    random.Random(args.seed).shuffle(train)
    n_cal = int(len(train) * args.calib_fraction)
    calib, fit = train[:n_cal], train[n_cal:]
    X_fit = embed_rows(fit)
    clf = EmbeddingClassifier.train(X_fit, {f: [r[f] for r in fit] for f in FIELDS}, "eval-fit")
    deployed = get_classifier()

    thresholds = [round(t, 2) for t in np.arange(0.3, 0.99, 0.02)]
    cal_pred = clf.predict_many(embed_rows(calib))
    cal_sweep = routing_sweep([routing_confidence(p, fields) for p in cal_pred],
                              [routing_correct(p, r, fields) for p, r in zip(cal_pred, calib)], thresholds)
    chosen = pick_route_threshold(cal_sweep, args.min_accuracy)

    test = load_tickets("test")
    X_test = embed_rows(test)
    preds = clf.predict_many(X_test)

    per_field = {}
    for f in FIELDS:
        y, yhat = [r[f] for r in test], [p[f].value for p in preds]
        per_field[f] = {"accuracy": accuracy_score(y, yhat), "macro_f1": f1_score(y, yhat, average="macro")}
    test_sweep = routing_sweep([routing_confidence(p, fields) for p in preds],
                               [routing_correct(p, r, fields) for p, r in zip(preds, test)], thresholds)
    at_test = next(s for s in test_sweep if s["threshold"] == chosen["threshold"])

    print(f"classifier on TEST ({len(test)} tickets, {get_embedder().version} + logistic regression)")
    for f, m in per_field.items():
        print(f"  {f:9s} accuracy {m['accuracy']:.3f}   macro-F1 {m['macro_f1']:.3f}")
    print(f"\nrouting on fields {fields}: threshold {chosen['threshold']} chosen on calibration slice")
    print(f"  TEST: {at_test['auto_share']:.1%} auto-routed at {at_test['auto_accuracy']:.1%} accuracy "
          f"(target >= 70% at >= {args.min_accuracy:.0%})")
    print(f"Set ROUTE_THRESHOLD={chosen['threshold']} in .env")
    if deployed is None:
        print("(no deployed model yet: run python -m scripts.train_classifier)")

    report = {
        "embedding_model": get_embedder().version, "routing_fields": fields, "test_size": len(test),
        "per_field": per_field, "chosen_threshold": chosen["threshold"], "test_at_chosen": at_test,
        "calibration_sweep": cal_sweep, "test_sweep": test_sweep,
    }

    if args.llm:
        llm = get_llm_client()
        if llm is None:
            raise SystemExit("LLM_PROVIDER is none; set groq/ollama to evaluate the fallback")
        test_texts = texts(test)
        unsure = set([i for i, p in enumerate(preds) if uncertain_fields(p, fields, chosen["threshold"])][: args.llm])
        auto, correct, cost, calls = 0, 0, 0.0, 0
        for i, p in enumerate(preds):
            llm_labels = None
            if i in unsure:
                result = llm.complete(classification_prompt(test_texts[i]), max_tokens=100, json_mode=True)
                llm_labels, cost, calls = parse_llm_labels(extract_json(result.text)), cost + result.cost_usd, calls + 1
            elif uncertain_fields(p, fields, chosen["threshold"]):
                continue  # beyond --llm budget: skip from this end-to-end estimate
            d = decide(p, llm_labels, fields, chosen["threshold"])
            if d.routing == "auto":
                auto += 1
                correct += all(d.labels[f].value == test[i][f] for f in fields)
        evaluated = sum(1 for i, p in enumerate(preds)
                        if i in unsure or not uncertain_fields(p, fields, chosen["threshold"]))
        report["with_llm"] = {
            "evaluated": evaluated, "llm_calls": calls, "auto_share": auto / evaluated,
            "auto_accuracy": correct / auto if auto else None, "llm_cost_usd": cost, "model": llm.model,
        }
        print(f"\nwith LLM fallback ({llm.model}, {calls} calls): {auto / evaluated:.1%} auto-routed at "
              f"{(correct / auto if auto else 0):.1%} accuracy; cost ${cost:.4f}")

    write_json(REPORTS / "classification_eval.json", report)

    plt = try_pyplot()
    if plt:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot([s["threshold"] for s in test_sweep], [s["auto_share"] for s in test_sweep], label="auto-routed share")
        ax.plot([s["threshold"] for s in test_sweep], [s["auto_accuracy"] for s in test_sweep],
                label="accuracy of auto-routed")
        ax.axvline(chosen["threshold"], color="grey", ls=":", label=f"chosen {chosen['threshold']}")
        ax.set_xlabel("confidence threshold (min over routing fields)")
        ax.set_ylim(0, 1.02)
        ax.set_title("Routing trade-off on TEST")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(REPORTS / "routing_tradeoff.png", dpi=150)
        print(f"chart -> {REPORTS / 'routing_tradeoff.png'}")


if __name__ == "__main__":
    main()
