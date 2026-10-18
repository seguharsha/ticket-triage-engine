"""Generate the labeled TaskFlow ticket dataset with an LLM (Gemini or a local Ollama model).

Labels are chosen first (subtopic -> category, team; priority; style), then the LLM writes text
that fits them, so every ticket is labeled by construction.

Outputs (in --out, default data/):
  tickets.jsonl   incoming tickets: normal + duplicate groups + incident bursts, sorted by created_at,
                  each with split = train | test (whole duplicate groups go to one side)
  groups.jsonl    one row per duplicate group with its invented root problem
  resolved.jsonl  past resolved tickets with the agent reply, for grounded reply suggestions

Every API response is cached in <out>/raw/, so an interrupted run resumes where it stopped.

  python -m scripts.generate_dataset --dry-run          # show plan + one prompt, no API calls
  python -m scripts.generate_dataset --normal 40 --groups 4 --incidents 1 --resolved 10   # small trial
  python -m scripts.generate_dataset                    # full dataset (Gemini)
  python -m scripts.generate_dataset --provider ollama  # local model; slow on CPU
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from app.taxonomy import PRIORITY_MEANING, SUBTOPICS, SUBTOPICS_BY_KEY, Subtopic

WEEK_START = datetime(2026, 10, 19)  # incoming tickets span this Monday + 7 days
TEST_FRACTION = 0.2

TONES = ["polite", "frustrated", "angry", "terse", "confused and non-technical", "technical and detailed",
         "rambling", "apologetic"]
LENGTHS = ["one short line", "two or three sentences", "a detailed paragraph"]
TYPOS = ["no typos", "a few typos", "many typos, little punctuation, lowercase"]
FORMATS = {
    "plain text": 0.72,
    "email ending with a signature block (name, title, company, phone)": 0.12,
    "email reply: new text on top, then a quoted earlier support message (e.g. 'On ... wrote:' and '>' lines)": 0.10,
    "HTML email body (with <p>, <br>, maybe <b>)": 0.06,
}

PRODUCT_BRIEF = (
    "TaskFlow is a fictional project-management SaaS: workspaces, projects, tasks, Kanban boards, comments, "
    "file attachments, notifications, web + iOS/Android apps, SSO for business plans, Free/Pro/Business plans, "
    "CSV export and a public API."
)
RULES = (
    "Rules: invent concrete, varied details (pages, browsers, devices, error messages, numbers, dates) so no two "
    "tickets share specifics. Reflect the urgency in the wording but never write a priority code, category or team "
    "name. Use only fake names and example.com emails. Do not sign as TaskFlow. Write in the customer's own "
    "words: never copy the wording of the problem description into the subject or body."
)


def pick_weighted(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def style(rng: random.Random) -> dict:
    return {
        "tone": rng.choice(TONES),
        "length": rng.choices(LENGTHS, weights=[0.25, 0.45, 0.30])[0],
        "typos": rng.choices(TYPOS, weights=[0.5, 0.35, 0.15])[0],
        "format": pick_weighted(rng, FORMATS),
    }


def business_hours_time(rng: random.Random, start: datetime, days: float) -> datetime:
    # Most tickets arrive in working hours; keep some at night/weekend.
    while True:
        t = start + timedelta(seconds=rng.uniform(0, days * 86400))
        if (8 <= t.hour < 20 and t.weekday() < 5) or rng.random() < 0.25:
            return t


def build_plan(args, rng: random.Random) -> dict:
    normal = []
    for i in range(args.normal):
        sub = rng.choice(SUBTOPICS)
        normal.append({"idx": i, "subtopic": sub.key, "priority": pick_weighted(rng, sub.priorities), **style(rng)})

    groups = []
    incident_subs = [s for s in SUBTOPICS if s.incident_prone]
    for g in range(args.groups + args.incidents):
        is_incident = g >= args.groups
        if is_incident:
            sub = rng.choice(incident_subs)
            priority, size = "P1", rng.randint(10, 16)
        else:
            sub = rng.choice(SUBTOPICS)
            priority, size = pick_weighted(rng, sub.priorities), rng.randint(3, 6)
        groups.append({
            "group": f"G-{g + 1:03d}",
            "subtopic": sub.key,
            "priority": priority,
            "incident": is_incident,
            "tickets": [{"n": n + 1, **style(rng)} for n in range(size)],
        })

    resolved = []
    for i in range(args.resolved):
        sub = SUBTOPICS[i % len(SUBTOPICS)]  # even coverage so every subtopic has reply evidence
        resolved.append({"idx": i, "subtopic": sub.key, "priority": pick_weighted(rng, sub.priorities), **style(rng)})
    return {"normal": normal, "groups": groups, "resolved": resolved}


def describe(sub: Subtopic, priority: str) -> dict:
    return {"problem": sub.description, "urgency": PRIORITY_MEANING[priority]}


def normal_prompt(batch: list[dict]) -> str:
    specs = [{"idx": s["idx"], **describe(SUBTOPICS_BY_KEY[s["subtopic"]], s["priority"]),
              **{k: s[k] for k in ("tone", "length", "typos", "format")}} for s in batch]
    return (
        f"You write realistic customer support tickets. {PRODUCT_BRIEF}\n\n"
        "Write one ticket per spec below; each comes from a different customer. Follow each spec's problem, "
        f"urgency, tone, length, typos and format.\n{RULES}\n\n"
        f"Specs:\n{json.dumps(specs, indent=1)}\n\n"
        'Return a JSON array with one object per spec: {"idx": <spec idx>, "subject": "...", "body": "..."}'
    )


def groups_prompt(batch: list[dict]) -> str:
    specs = []
    for g in batch:
        sub = SUBTOPICS_BY_KEY[g["subtopic"]]
        spec = {"group": g["group"], **describe(sub, g["priority"]), "tickets": g["tickets"]}
        if g["incident"]:
            spec["situation"] = "a live outage hitting many customers at the same time, reported within minutes"
        specs.append(spec)
    return (
        f"You write realistic customer support tickets. {PRODUCT_BRIEF}\n\n"
        "For each group below, invent ONE concrete, specific root problem matching the group's problem "
        "(a specific page, action, browser/device, error message...). Then write one ticket per entry in "
        "\"tickets\", each from a different customer who hit that same root problem. They must read like "
        "independent people: different vocabulary, different symptoms noticed, different guesses at the cause, "
        "some non-technical. Do not reuse distinctive phrases between tickets of a group. Each ticket follows its "
        f"own tone, length, typos and format.\n{RULES}\n\n"
        f"Groups:\n{json.dumps(specs, indent=1)}\n\n"
        'Return a JSON array: [{"group": "<group id>", "root_problem": "<one sentence>", '
        '"tickets": [{"n": <ticket n>, "subject": "...", "body": "..."}]}]'
    )


def resolved_prompt(batch: list[dict]) -> str:
    specs = [{"idx": s["idx"], **describe(SUBTOPICS_BY_KEY[s["subtopic"]], s["priority"]),
              "tone": s["tone"], "length": s["length"]} for s in batch]
    return (
        f"You write realistic, already-resolved customer support conversations. {PRODUCT_BRIEF}\n\n"
        "For each spec, write the customer's ticket and the support agent's reply that resolved it. The reply is "
        "friendly and concrete: the cause if known, numbered steps or the action taken (refund issued, setting to "
        "change, workaround, fix deployed), and what to do if it still fails. 60-180 words. Sign replies as "
        f"'TaskFlow Support'.\n{RULES}\n\n"
        f"Specs:\n{json.dumps(specs, indent=1)}\n\n"
        'Return a JSON array: [{"idx": <spec idx>, "subject": "...", "body": "...", "reply": "..."}]'
    )


def unwrap_items(parsed):
    """Prompts ask for a JSON array; JSON modes that force an object return {"items": [...]}."""
    if isinstance(parsed, dict):
        lists = [v for v in parsed.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    return parsed


class CachedJsonLLM:
    """Shared caching/pacing/retry loop. Subclasses implement _request(prompt) -> text or None (retry)."""

    model = "unknown"
    min_interval = 0.0

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.last_call = 0.0
        self.calls = 0

    def _request(self, name: str, prompt: str, attempt: int) -> str | None:
        raise NotImplementedError

    def generate_json(self, name: str, prompt: str):
        path = self.cache_dir / f"{name}.json"
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached["prompt_sha256"] == digest:
                return cached["response"]

        for attempt in range(6):
            wait = self.min_interval - (time.monotonic() - self.last_call)
            if wait > 0:
                time.sleep(wait)
            self.last_call = time.monotonic()
            self.calls += 1
            text = self._request(name, prompt, attempt)
            if text is None:
                continue
            try:
                parsed = unwrap_items(json.loads(text))
            except json.JSONDecodeError:
                print(f"  {name}: invalid JSON, retrying", file=sys.stderr)
                continue
            if not isinstance(parsed, list):
                print(f"  {name}: expected a list, retrying", file=sys.stderr)
                continue
            path.write_text(json.dumps({"prompt_sha256": digest, "model": self.model, "response": parsed},
                                       ensure_ascii=False), encoding="utf-8")
            return parsed
        print(f"  {name}: giving up after retries; batch skipped", file=sys.stderr)
        return []


class Gemini(CachedJsonLLM):
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str, rpm: float, cache_dir: Path):
        super().__init__(cache_dir)
        self.api_key, self.model = api_key, model
        self.min_interval = 60.0 / rpm
        self.http = httpx.Client(timeout=180)

    def _request(self, name: str, prompt: str, attempt: int) -> str | None:
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0},
        }
        try:
            resp = self.http.post(self.URL.format(model=self.model), json=body, headers={"x-goog-api-key": self.api_key})
        except httpx.HTTPError as exc:
            print(f"  {name}: network error {exc!r}, retrying", file=sys.stderr)
            time.sleep(5 * 2 ** attempt)
            return None
        if resp.status_code == 429 or resp.status_code >= 500:
            if resp.status_code == 429 and "PerDay" in resp.text:
                sys.exit("Daily Gemini quota exhausted. Re-run tomorrow; finished batches are cached.")
            delay = min(120, 10 * 2 ** attempt)
            print(f"  {name}: HTTP {resp.status_code}, waiting {delay}s", file=sys.stderr)
            time.sleep(delay)
            return None
        if resp.status_code != 200:
            sys.exit(f"Gemini error {resp.status_code}: {resp.text[:500]}")
        try:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            print(f"  {name}: unusable response ({exc.__class__.__name__}), retrying", file=sys.stderr)
            return None


class OllamaGen(CachedJsonLLM):
    """Local generation. JSON mode forces an object, so the prompt asks for {"items": [...]}."""

    def __init__(self, base_url: str, model: str, cache_dir: Path, num_predict: int = 3000):
        super().__init__(cache_dir)
        self.model = f"ollama/{model}"
        self._model, self.num_predict = model, num_predict
        self.http = httpx.Client(base_url=base_url, timeout=1800)

    def _request(self, name: str, prompt: str, attempt: int) -> str | None:
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt + '\n\nWrap the array in an object: {"items": [...]}'}],
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {"temperature": 0.9, "num_ctx": 8192, "num_predict": self.num_predict, "seed": attempt},
        }
        try:
            resp = self.http.post("/api/chat", json=body)
        except httpx.ConnectError:
            sys.exit(f"Cannot reach Ollama at {self.http.base_url}. Is it running?")
        except httpx.HTTPError as exc:
            print(f"  {name}: {exc!r}, retrying", file=sys.stderr)
            return None
        if resp.status_code == 404:
            sys.exit(f"Model not found. Run: ollama pull {self._model}")
        if resp.status_code != 200:
            print(f"  {name}: HTTP {resp.status_code} {resp.text[:200]}, retrying", file=sys.stderr)
            return None
        data = resp.json()
        if data.get("eval_count") and data.get("eval_duration"):
            print(f"  {name}: {data['eval_count']} tokens at "
                  f"{data['eval_count'] / (data['eval_duration'] / 1e9):.1f} tok/s")
        return data["message"]["content"]

def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield i // size, items[i:i + size]


def valid_text(obj: dict, *fields: str) -> bool:
    return all(isinstance(obj.get(f), str) and len(obj[f].strip()) >= 3 for f in fields)


def labels(sub_key: str, priority: str) -> dict:
    sub = SUBTOPICS_BY_KEY[sub_key]
    return {"category": sub.category, "priority": priority, "team": sub.team, "subtopic": sub.key}


def generate(args, plan: dict, llm: CachedJsonLLM, rng: random.Random):
    normal_batch = getattr(args, "normal_batch", 20)
    group_batch = getattr(args, "group_batch", 4)
    resolved_batch = getattr(args, "resolved_batch", 10)
    tickets, groups_out, resolved_out = [], [], []

    batches = list(chunks(plan["normal"], normal_batch))
    for b, batch in batches:
        print(f"normal batch {b + 1}/{len(batches)}")
        by_idx = {o.get("idx"): o for o in llm.generate_json(f"normal_{b:03d}", normal_prompt(batch))
                  if isinstance(o, dict)}
        for spec in batch:
            out = by_idx.get(spec["idx"])
            if out and valid_text(out, "subject", "body"):
                tickets.append({"subject": out["subject"].strip(), "body": out["body"].strip(),
                                **labels(spec["subtopic"], spec["priority"]), "duplicate_group_id": None,
                                "is_incident": False,
                                "created_at": business_hours_time(rng, WEEK_START, 7)})

    regular = [g for g in plan["groups"] if not g["incident"]]
    incidents = [g for g in plan["groups"] if g["incident"]]
    group_batches = [(f"groups_{b:03d}", batch) for b, batch in chunks(regular, group_batch)]
    group_batches += [(f"incident_{g['group']}", [g]) for g in incidents]
    for n, (name, batch) in enumerate(group_batches):
        print(f"group batch {n + 1}/{len(group_batches)}")
        by_id = {o.get("group"): o for o in llm.generate_json(name, groups_prompt(batch)) if isinstance(o, dict)}
        for g in batch:
            out = by_id.get(g["group"])
            texts = [t for t in (out or {}).get("tickets", []) if isinstance(t, dict) and valid_text(t, "subject", "body")]
            if len(texts) < 2:
                print(f"  {g['group']}: fewer than 2 usable tickets, dropped", file=sys.stderr)
                continue
            if g["incident"]:
                anchor = business_hours_time(rng, WEEK_START, 6.9)
                times = [anchor] + [anchor + timedelta(minutes=rng.uniform(0, 30)) for _ in texts[1:]]
            else:
                anchor = business_hours_time(rng, WEEK_START, 5)
                times = [anchor] + [anchor + timedelta(hours=rng.uniform(0, 48)) for _ in texts[1:]]
            for t, created_at in zip(texts, sorted(times)):
                tickets.append({"subject": t["subject"].strip(), "body": t["body"].strip(),
                                **labels(g["subtopic"], g["priority"]), "duplicate_group_id": g["group"],
                                "is_incident": g["incident"], "created_at": created_at})
            groups_out.append({"duplicate_group_id": g["group"], "root_problem": out.get("root_problem", ""),
                               "is_incident": g["incident"], "size": len(texts),
                               **labels(g["subtopic"], g["priority"])})

    batches = list(chunks(plan["resolved"], resolved_batch))
    for b, batch in batches:
        print(f"resolved batch {b + 1}/{len(batches)}")
        by_idx = {o.get("idx"): o for o in llm.generate_json(f"resolved_{b:03d}", resolved_prompt(batch))
                  if isinstance(o, dict)}
        for spec in batch:
            out = by_idx.get(spec["idx"])
            if out and valid_text(out, "subject", "body", "reply"):
                resolved_out.append({"subject": out["subject"].strip(), "body": out["body"].strip(),
                                     "reply": out["reply"].strip(), **labels(spec["subtopic"], spec["priority"]),
                                     # resolved history predates the incoming week
                                     "created_at": WEEK_START - timedelta(days=rng.uniform(7, 180))})
    return tickets, groups_out, resolved_out


def assign_splits(tickets: list[dict], groups: list[dict], rng: random.Random) -> None:
    # Split by unit (whole duplicate group, or a single normal ticket) so no group leaks across splits.
    test_groups = set()
    for is_incident in (False, True):
        ids = sorted(g["duplicate_group_id"] for g in groups if g["is_incident"] == is_incident)
        rng.shuffle(ids)
        test_groups.update(ids[: round(len(ids) * TEST_FRACTION)])
    singles = [t for t in tickets if t["duplicate_group_id"] is None]
    rng.shuffle(singles)
    test_singles = {id(t) for t in singles[: round(len(singles) * TEST_FRACTION)]}
    for t in tickets:
        in_test = t["duplicate_group_id"] in test_groups if t["duplicate_group_id"] else id(t) in test_singles
        t["split"] = "test" if in_test else "train"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=["gemini", "ollama"], default="gemini",
                   help="gemini (hosted, fast) or ollama (local, slow on CPU)")
    p.add_argument("--normal", type=int, default=2000, help="standalone tickets")
    p.add_argument("--groups", type=int, default=88, help="regular duplicate groups (3-6 tickets each)")
    p.add_argument("--incidents", type=int, default=12, help="incident groups (10-16 tickets within 30 minutes)")
    p.add_argument("--resolved", type=int, default=300, help="resolved tickets with agent replies")
    p.add_argument("--normal-batch", type=int, help="tickets per call (default 20 gemini, 5 ollama)")
    p.add_argument("--group-batch", type=int, help="groups per call (default 4 gemini, 1 ollama)")
    p.add_argument("--resolved-batch", type=int, help="resolved tickets per call (default 10 gemini, 3 ollama)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rpm", type=float, default=9, help="max Gemini requests per minute (free tier ~10)")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--dry-run", action="store_true", help="print the plan and one prompt; no API calls")
    args = p.parse_args()
    # Small local models are unreliable with long JSON arrays: smaller batches by default.
    local = args.provider == "ollama"
    args.normal_batch = args.normal_batch or (5 if local else 20)
    args.group_batch = args.group_batch or (1 if local else 4)
    args.resolved_batch = args.resolved_batch or (3 if local else 10)

    rng = random.Random(args.seed)
    plan = build_plan(args, rng)
    n_group_tickets = sum(len(g["tickets"]) for g in plan["groups"])
    n_calls = (-(-args.normal // args.normal_batch) + -(-args.groups // args.group_batch) + args.incidents
               + -(-args.resolved // args.resolved_batch))
    print(f"plan: {args.normal} normal + {n_group_tickets} in {len(plan['groups'])} duplicate groups "
          f"({args.incidents} incidents) + {args.resolved} resolved; ~{n_calls} LLM calls via {args.provider}")
    if args.dry_run:
        print("\n--- first group prompt ---\n" + groups_prompt(plan["groups"][:1]))
        return

    load_env(Path(".env"))
    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    if local:
        model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        llm = OllamaGen(os.environ.get("OLLAMA_URL", "http://localhost:11434"), model, args.out / "raw")
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit("Set GEMINI_API_KEY in .env (free key: https://aistudio.google.com/apikey), "
                     "or use --provider ollama")
        llm = Gemini(api_key, os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), args.rpm, args.out / "raw")

    # Separate RNG for timestamps/splits so they don't shift when the plan args change.
    started = time.monotonic()
    tickets, groups, resolved = generate(args, plan, llm, random.Random(args.seed + 1))

    tickets.sort(key=lambda t: t["created_at"])
    for i, t in enumerate(tickets, 1):
        t["id"] = f"T-{i:05d}"
        t["source"] = "generated"
    assign_splits(tickets, groups, random.Random(args.seed + 2))
    for i, r in enumerate(sorted(resolved, key=lambda r: r["created_at"]), 1):
        r["id"] = f"R-{i:04d}"

    key_order = ["id", "subject", "body", "category", "priority", "team", "subtopic", "duplicate_group_id",
                 "is_incident", "created_at", "split", "source"]
    write_jsonl(args.out / "tickets.jsonl", [{k: t[k] for k in key_order} for t in tickets])
    write_jsonl(args.out / "groups.jsonl", groups)
    write_jsonl(args.out / "resolved.jsonl", sorted(resolved, key=lambda r: r["id"]))

    n_test = sum(t["split"] == "test" for t in tickets)
    meta = {
        "generator_model": llm.model, "provider": args.provider, "seed": args.seed,
        "requested": {"normal": args.normal, "groups": args.groups, "incidents": args.incidents,
                      "resolved": args.resolved},
        "tickets": len(tickets), "test_tickets": n_test, "train_tickets": len(tickets) - n_test,
        "standalone_tickets": sum(t["duplicate_group_id"] is None for t in tickets),
        "duplicate_groups": sum(not g["is_incident"] for g in groups),
        "incident_groups": sum(g["is_incident"] for g in groups),
        "tickets_in_groups": sum(t["duplicate_group_id"] is not None for t in tickets),
        "resolved_tickets": len(resolved),
    }
    (args.out / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {len(tickets)} tickets ({n_test} test), {len(groups)} groups, {len(resolved)} resolved "
          f"to {args.out}/ using {llm.model}; {llm.calls} LLM calls this run, "
          f"{(time.monotonic() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()