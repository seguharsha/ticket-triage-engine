"""Grounded reply drafts from similar resolved tickets.

1. Retrieve the top-k resolved tickets by embedding similarity.
2. Nothing above reply_min_similarity -> "no confident suggestion" without calling the LLM.
3. Otherwise the LLM sees only those tickets, labelled [S1]..[Sk], must cite them, and may answer
   NO_SUGGESTION. A draft that cites nothing valid is discarded rather than shown.
"""

import re
import uuid
from dataclasses import dataclass

from sqlalchemy import Select, select

from app.models import Ticket, TicketStatus

NO_SUGGESTION = "NO_SUGGESTION"
_CITATION = re.compile(r"\[S(\d+)\]")


@dataclass
class Source:
    ticket_id: uuid.UUID
    subject: str
    resolution: str
    similarity: float


@dataclass
class Draft:
    status: str  # ready | no_suggestion
    text: str | None
    cited: list[int]  # 1-based source numbers
    reason: str


def similar_resolved_stmt(embedding, k: int) -> Select:
    distance = Ticket.embedding.cosine_distance(embedding)
    return (
        select(Ticket.id, Ticket.subject, Ticket.resolution, distance.label("distance"))
        .where(Ticket.status == TicketStatus.RESOLVED, Ticket.embedding.is_not(None))
        .order_by(distance)
        .limit(k)
    )


def to_sources(rows) -> list[Source]:
    return [Source(r.id, r.subject, r.resolution or "", 1 - float(r.distance)) for r in rows]


def relevant(sources: list[Source], min_similarity: float) -> list[Source]:
    return [s for s in sources if s.similarity >= min_similarity]


def reply_prompt(ticket_text: str, sources: list[Source]) -> str:
    blocks = [
        f"[S{i}] Customer: {s.subject}\nAgent reply that resolved it:\n{s.resolution[:1500]}"
        for i, s in enumerate(sources, 1)
    ]
    return "\n".join([
        "You draft a reply for a TaskFlow support agent to review before sending.",
        "Use ONLY the resolved tickets below as your knowledge. Do not invent steps, settings, policies or "
        "timelines that are not in them.",
        "Every sentence that gives a cause or a step MUST end with the source it came from, like [S1] or [S2]. "
        "Example: \"Go to Admin > Security and upload the new certificate [S1].\"",
        "If a step is not in the sources, leave it out.",
        f"If none of the sources address this customer's problem, answer with exactly {NO_SUGGESTION}.",
        "Keep it under 150 words, friendly and specific. Sign as 'TaskFlow Support'.",
        "",
        "Resolved tickets:",
        "\n\n".join(blocks),
        "",
        "New ticket:",
        '"""',
        ticket_text[:3000],
        '"""',
        "",
        "Draft reply:",
    ])


def parse_draft(text: str, n_sources: int) -> Draft:
    stripped = text.strip()
    if not stripped or stripped.upper().startswith(NO_SUGGESTION) or stripped.upper() == NO_SUGGESTION:
        return Draft("no_suggestion", None, [], "model found no relevant source")
    cited = sorted({int(n) for n in _CITATION.findall(stripped) if 1 <= int(n) <= n_sources})
    if not cited:
        return Draft("no_suggestion", None, [], "draft cited no retrieved source")
    # Drop citations to sources that were never provided ([S9] when there are 3).
    cleaned = _CITATION.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", stripped)
    return Draft("ready", cleaned, cited, f"grounded in {len(cited)} source(s)")


def sources_json(sources: list[Source], cited: list[int]) -> list[dict]:
    return [
        {
            "ticket_id": str(s.ticket_id),
            "subject": s.subject,
            "similarity": round(s.similarity, 4),
            "cited": i in cited,
            "url": f"/tickets/{s.ticket_id}",
        }
        for i, s in enumerate(sources, 1)
    ]
