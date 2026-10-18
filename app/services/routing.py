"""Decide final labels and whether a ticket is auto-routed or goes to human review.

Rule:
  1. Every routing field (default: category, team) confident >= threshold  -> auto.
  2. Otherwise ask the LLM (if enabled). For each uncertain routing field, if the LLM agrees with the
     classifier's top label, two independent models agree -> treat as confident.
  3. Any remaining disagreement -> review queue, showing the LLM's label as the suggestion.

LLM self-reported confidence is unreliable, so agreement between two different models is used as
the signal instead.
"""

from dataclasses import dataclass

from app.services.classifier import FIELDS, FieldPrediction
from app.taxonomy import CATEGORIES, PRIORITIES, TEAMS, SUBTOPICS, PRIORITY_MEANING

ALLOWED = {"category": CATEGORIES, "priority": PRIORITIES, "team": TEAMS}


@dataclass
class Decision:
    labels: dict[str, FieldPrediction]
    routing: str  # auto | review
    reason: str
    used_llm: bool


def uncertain_fields(primary: dict[str, FieldPrediction], routing_fields: list[str], threshold: float) -> list[str]:
    return [f for f in routing_fields if f not in primary or primary[f].confidence < threshold]


def decide(
    primary: dict[str, FieldPrediction],
    llm_labels: dict[str, str] | None,
    routing_fields: list[str],
    threshold: float,
) -> Decision:
    unsure = uncertain_fields(primary, routing_fields, threshold)
    if not unsure:
        return Decision(dict(primary), "auto", "classifier confident", used_llm=False)
    if not llm_labels:
        return Decision(dict(primary), "review", f"low confidence on {', '.join(unsure)}; no LLM", used_llm=False)

    labels = dict(primary)
    disagreements = []
    for name in FIELDS:
        llm_value = llm_labels.get(name)
        current = primary.get(name)
        if current is None:
            # No classifier output: the LLM label is shown, but one model alone can't auto-route.
            if llm_value:
                labels[name] = FieldPrediction(llm_value, 0.0)
            if name in routing_fields:
                disagreements.append(name)
        elif name in unsure:
            if llm_value is None:
                disagreements.append(name)
            elif llm_value != current.value:
                labels[name] = FieldPrediction(
                    llm_value, current.probabilities.get(llm_value, 0.0), current.probabilities
                )
                disagreements.append(name)
            # else: agreement, keep the classifier's label

    if disagreements:
        return Decision(labels, "review", f"classifier and LLM disagree on {', '.join(disagreements)}", used_llm=True)
    return Decision(labels, "auto", "LLM confirmed uncertain fields", used_llm=True)


def classification_prompt(ticket_text: str) -> str:
    categories = sorted({s.category for s in SUBTOPICS})
    examples = {c: [s.description for s in SUBTOPICS if s.category == c] for c in categories}
    owners = {t: sorted({s.description for s in SUBTOPICS if s.team == t}) for t in TEAMS}
    lines = [
        "You triage support tickets for TaskFlow, a project-management SaaS.",
        "Classify the ticket. Use exactly one allowed value per field.",
        "",
        "category (with typical topics):",
        *[f"- {c}: {'; '.join(examples[c])}" for c in categories],
        "",
        "team (owns these topics):",
        *[f"- {t}: {'; '.join(owners[t])}" for t in TEAMS],
        "",
        "priority:",
        *[f"- {p}: {PRIORITY_MEANING[p]}" for p in PRIORITIES],
        "",
        "Ticket:",
        '"""',
        ticket_text[:4000],
        '"""',
        "",
        'Answer with JSON only: {"category": "...", "priority": "...", "team": "..."}',
    ]
    return "\n".join(lines)


def parse_llm_labels(data: dict | None) -> dict[str, str]:
    if not data:
        return {}
    out = {}
    for name, allowed in ALLOWED.items():
        value = data.get(name)
        if isinstance(value, str):
            value = value.strip()
            match = next((a for a in allowed if a.lower() == value.lower()), None)
            if match:
                out[name] = match
    return out
