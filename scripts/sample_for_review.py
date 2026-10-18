"""Write a random sample of generated tickets to a Markdown file for hand review.

Read each one: does the text match its labels? Is it realistic? Fix or delete bad rows in
data/tickets.jsonl, and if many share a flaw, change the prompt and regenerate that batch.

  python -m scripts.sample_for_review --n 50
"""

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/tickets.jsonl"))
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/review_sample.md"))
    args = p.parse_args()

    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    lines = [f"# Review sample: {len(sample)} of {len(rows)} tickets\n",
             "Mark each: OK / FIX / DROP. Check that the text fits category, priority and team.\n"]
    for r in sample:
        group = f" · group {r['duplicate_group_id']}" if r["duplicate_group_id"] else ""
        lines += [
            f"## {r['id']} · {r['category']} · {r['priority']} · {r['team']} · {r['subtopic']}{group}",
            f"**{r['subject']}**\n",
            "```text", r["body"], "```",
            "Verdict: \n",
        ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
