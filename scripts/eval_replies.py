"""Reply-suggestion evaluation: generate drafts for 30 test tickets for manual rating.

  python -m scripts.eval_replies generate     # writes reports/reply_review.md
  (open it, rate each WRITTEN draft: replace "Usable: ?" with yes or no; "no suggestion" entries are
   scored automatically from the subtopic labels)
  python -m scripts.eval_replies score        # usable rate, coverage, correct vs missed declines

Uses the same retrieval threshold, prompt and citation parsing as the worker.
"""

import argparse
import random
import re

import numpy as np

from app.config import settings
from app.services.llm import get_llm_client
from app.services.replies import Source, parse_draft, relevant, reply_prompt
from scripts.common import DATA, REPORTS, embed_rows, load_jsonl, load_tickets, texts, write_json

REVIEW_MD = REPORTS / "reply_review.md"


def generate(n: int, seed: int) -> None:
    llm = get_llm_client()
    if llm is None:
        raise SystemExit("Set LLM_PROVIDER to groq or ollama in .env")
    resolved = load_jsonl(DATA / "resolved.jsonl")
    R = embed_rows(resolved)
    sample = random.Random(seed).sample(load_tickets("test"), n)
    Q = embed_rows(sample)
    sample_texts = texts(sample)

    lines = [f"# Reply review: {n} test tickets", "",
             f"Model: {llm.model}. Retrieval: top {settings.reply_top_k}, min similarity "
             f"{settings.reply_min_similarity}.", "",
             "Rate each draft: would an agent send it with light edits? Replace `?` with yes or no. "
             "A correct \"no suggestion\" counts as usable when no source fits.", ""]
    results, cost = [], 0.0
    for i, (row, q) in enumerate(zip(sample, Q), 1):
        sims = R @ q
        top = np.argsort(-sims)[: settings.reply_top_k]
        sources = [Source(resolved[j]["id"], resolved[j]["subject"], resolved[j]["reply"], float(sims[j])) for j in top]
        usable = relevant(sources, settings.reply_min_similarity)
        if usable:
            result = llm.complete(reply_prompt(sample_texts[i - 1], usable), max_tokens=400)
            cost += result.cost_usd
            draft = parse_draft(result.text, len(usable))
        else:
            draft = None
        status = draft.status if draft else "no_suggestion"
        results.append({"ticket": row["id"], "status": status, "cited": draft.cited if draft else []})
        lines += [
            f"## {i}. {row['id']} Â· {row['category']} Â· {row['subtopic']}",
            f"**{row['subject']}**", "", "```text", row["body"][:1200], "```", "",
            "Sources: " + "; ".join(f"[S{k}] {s.ticket_id} ({s.similarity:.2f}) {s.subject}"
                                    for k, s in enumerate(usable, 1)) if usable else "Sources: none above threshold",
            "",
            f"Draft ({status}):", "", "```text", (draft.text if draft and draft.text else "(no confident suggestion)"),
            "```", "", "Usable: ?", "",
        ]
    REPORTS.mkdir(exist_ok=True)
    REVIEW_MD.write_text("\n".join(lines), encoding="utf-8")
    write_json(REPORTS / "reply_drafts.json", {"model": llm.model, "llm_cost_usd": cost, "results": results})
    print(f"wrote {REVIEW_MD}; {sum(r['status'] == 'ready' for r in results)}/{n} drafts, cost ${cost:.4f}")


def score() -> None:
    """Usable rate over all 30 tickets: a written draft counts if the reviewer rated it usable; a
    "no suggestion" counts if declining was correct, i.e. none of the retrieved sources has the
    ticket's subtopic (objective from the labels, so declines need no human rating)."""
    subtopic = {r["id"]: r["subtopic"] for r in load_tickets()}
    source_subtopic = {r["id"]: r["subtopic"] for r in load_jsonl(DATA / "resolved.jsonl")}

    rows = []
    for entry in re.split(r"(?m)^## (?=\d+\. )", REVIEW_MD.read_text(encoding="utf-8"))[1:]:
        number = int(entry.split(".", 1)[0])
        ticket = re.search(r"T-\d+", entry).group(0)
        status = re.search(r"Draft \((\w+)\)", entry).group(1)
        sources = re.findall(r"\[S\d+\] (R-\d+)", re.search(r"(?m)^Sources: ([^\r\n]*)", entry).group(1))
        rating_match = re.search(r"(?m)^Usable:[ \t]*(\S*)", entry)
        rating = rating_match.group(1).lower() if rating_match else ""
        matching = [s for s in sources if source_subtopic.get(s) == subtopic.get(ticket)]
        rows.append({"n": number, "ticket": ticket, "status": status, "rating": rating, "matching_sources": matching})

    drafts = [r for r in rows if r["status"] == "ready"]
    declines = [r for r in rows if r["status"] == "no_suggestion"]
    unrated = [r["n"] for r in drafts if r["rating"] not in ("yes", "no")]
    if unrated:
        raise SystemExit(f"Written drafts not rated yet: #{', #'.join(map(str, unrated))} in {REVIEW_MD}")

    usable_drafts = sum(r["rating"] == "yes" for r in drafts)
    correct_declines = [r for r in declines if not r["matching_sources"]]
    missed = [r for r in declines if r["matching_sources"]]
    usable = usable_drafts + len(correct_declines)
    total = len(rows)

    print(f"usable overall: {usable}/{total} = {usable / total:.0%}   (spec metric, target >= 60%)")
    print(f"  written drafts rated usable:     {usable_drafts}/{len(drafts)}")
    print(f"  coverage (draft written):        {len(drafts)}/{total} = {len(drafts) / total:.0%}")
    print(f"  no suggestion, correctly:        {len(correct_declines)}/{len(declines)} (no same-subtopic source retrieved)")
    print(f"  no suggestion, missed answer:    {len(missed)}/{len(declines)} "
          f"(#{', #'.join(str(r['n']) for r in missed)}: a same-subtopic source was retrieved)")
    disagree = [r["n"] for r in declines if r["rating"] in ("yes", "no") and (r["rating"] == "yes") != (not r["matching_sources"])]
    if disagree:
        print(f"  note: the manual rating on #{', #'.join(map(str, disagree))} differs from the subtopic rule; the rule is used")
    write_json(REPORTS / "reply_eval.json", {
        "tickets": total, "usable": usable, "usable_share": usable / total,
        "drafts": {"count": len(drafts), "rated_usable": usable_drafts},
        "coverage": len(drafts) / total,
        "no_suggestion": {"count": len(declines), "correct": len(correct_declines),
                          "missed": [{"n": r["n"], "ticket": r["ticket"], "matching_sources": r["matching_sources"]} for r in missed]},
        "rule": "decline is correct iff no retrieved source shares the ticket's subtopic",
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["generate", "score"])
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate(args.n, args.seed) if args.command == "generate" else score()

if __name__ == "__main__":
    main()
