"""Exercise plan building and assembly with a fake LLM (no API calls)."""

import argparse
import random
import re
from datetime import timedelta

from app.taxonomy import SUBTOPICS_BY_KEY
from scripts import generate_dataset as gd


class FakeLLM:
    calls = 0

    def generate_json(self, name, prompt):
        self.calls += 1
        if name.startswith("normal"):
            return [{"idx": int(i), "subject": f"subject {i}", "body": f"body {i}"}
                    for i in re.findall(r'"idx": (\d+)', prompt)]
        if name.startswith("resolved"):
            return [{"idx": int(i), "subject": "subject", "body": "body", "reply": "do this"}
                    for i in re.findall(r'"idx": (\d+)', prompt)]
        # groups: echo every group id with as many tickets as requested
        out = []
        for gid in re.findall(r'"group": "(G-\d+)"', prompt):
            block = prompt.split(f'"group": "{gid}"', 1)[1].split('"group":', 1)[0]
            n = len(re.findall(r'"n": \d+', block))
            out.append({"group": gid, "root_problem": "rp",
                        "tickets": [{"n": k + 1, "subject": "dup", "body": f"dup {k}"} for k in range(n)]})
        return out


def run(**overrides):
    args = argparse.Namespace(normal=50, groups=10, incidents=3, resolved=30, seed=1, **overrides)
    plan = gd.build_plan(args, random.Random(args.seed))
    tickets, groups, resolved = gd.generate(args, plan, FakeLLM(), random.Random(2))
    gd.assign_splits(tickets, groups, random.Random(3))
    return plan, tickets, groups, resolved


def test_plan_is_deterministic():
    args = argparse.Namespace(normal=30, groups=5, incidents=2, resolved=10, seed=4)
    assert gd.build_plan(args, random.Random(4)) == gd.build_plan(args, random.Random(4))


def test_counts_and_labels_follow_subtopic():
    plan, tickets, groups, resolved = run()
    group_tickets = sum(len(g["tickets"]) for g in plan["groups"])
    assert len(tickets) == 50 + group_tickets
    assert len(groups) == 13 and len(resolved) == 30
    for t in tickets:
        sub = SUBTOPICS_BY_KEY[t["subtopic"]]
        assert (t["category"], t["team"]) == (sub.category, sub.team)


def test_group_sizes_and_incident_bursts():
    _, tickets, groups, _ = run()
    for g in groups:
        members = [t for t in tickets if t["duplicate_group_id"] == g["duplicate_group_id"]]
        times = [t["created_at"] for t in members]
        if g["is_incident"]:
            assert 10 <= len(members) <= 16
            assert max(times) - min(times) <= timedelta(minutes=30)
            assert all(t["priority"] == "P1" for t in members)
        else:
            assert 3 <= len(members) <= 6
            assert max(times) - min(times) <= timedelta(hours=48)


def test_split_keeps_groups_together_and_is_about_20_percent():
    _, tickets, _, _ = run()
    for gid in {t["duplicate_group_id"] for t in tickets if t["duplicate_group_id"]}:
        assert len({t["split"] for t in tickets if t["duplicate_group_id"] == gid}) == 1
    singles = [t for t in tickets if t["duplicate_group_id"] is None]
    assert sum(t["split"] == "test" for t in singles) == 10


def test_resolved_tickets_cover_every_subtopic_and_predate_the_week():
    _, _, _, resolved = run()
    assert {r["subtopic"] for r in resolved} == set(SUBTOPICS_BY_KEY)
    assert all(r["created_at"] < gd.WEEK_START for r in resolved)
