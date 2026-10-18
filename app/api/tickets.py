import uuid
from datetime import UTC, datetime
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import get_session
from app.models import Correction, ReplySuggestion, Routing, Ticket, TicketStatus
from app.queue import enqueue_ticket
from app.schemas import (
    ApproveReply,
    CorrectionIn,
    CorrectionOut,
    SimilarTicket,
    SuggestedReply,
    TicketAccepted,
    TicketOut,
    TicketSummary,
)
from app.services.attachments import read_uploads
from app.services.intake import IntakeResult, NewTicket, create_ticket
from app.services.replies import similar_resolved_stmt, to_sources

router = APIRouter(tags=["tickets"])


async def accept(result: IntakeResult, response: Response) -> TicketAccepted:
    if result.created:
        await enqueue_ticket(result.id)
    else:
        response.status_code = status.HTTP_200_OK
        # The original request may have stored the ticket but failed to enqueue it.
        # The retry heals that; the worker's per-ticket lock makes a double enqueue harmless.
        if result.status == TicketStatus.RECEIVED:
            await enqueue_ticket(result.id)
    return TicketAccepted(id=result.id, status=result.status, duplicate_submission=not result.created)


async def load_ticket(session: AsyncSession, ticket_id: uuid.UUID) -> Ticket:
    ticket = await session.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    return ticket


@router.post(
    "/tickets",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TicketAccepted,
    responses={200: {"model": TicketAccepted, "description": "Idempotency key already used"}},
)
async def submit_ticket(
    response: Response,
    idempotency_key: Annotated[str, Header(min_length=1, max_length=255)],
    subject: Annotated[str, Form(min_length=1, max_length=500)],
    body: Annotated[str, Form(min_length=1, max_length=50_000)],
    customer_email: Annotated[str | None, Form(max_length=320)] = None,
    created_at: Annotated[
        datetime | None, Form(description="Override the received time; used by the dataset replay script.")
    ] = None,
    attachments: Annotated[list[UploadFile] | None, File()] = None,
    session: AsyncSession = Depends(get_session),
) -> TicketAccepted:
    files = await read_uploads(attachments or [])
    result = await create_ticket(
        session,
        NewTicket(
            idempotency_key=idempotency_key,
            subject=subject,
            body=body,
            customer_email=customer_email,
            created_at=created_at,
        ),
        files,
    )
    return await accept(result, response)


@router.get("/tickets/{ticket_id}", response_model=TicketOut)
async def get_ticket(ticket_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> Ticket:
    ticket = await session.scalar(
        select(Ticket)
        .options(selectinload(Ticket.attachments), selectinload(Ticket.predictions))
        .where(Ticket.id == ticket_id)
    )
    if ticket is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    return ticket


@router.get("/tickets/{ticket_id}/similar", response_model=list[SimilarTicket])
async def similar_resolved(
    ticket_id: uuid.UUID,
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    session: AsyncSession = Depends(get_session),
) -> list[SimilarTicket]:
    ticket = await load_ticket(session, ticket_id)
    if ticket.embedding is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Ticket not embedded yet; retry shortly")
    rows = (await session.execute(similar_resolved_stmt(np.asarray(ticket.embedding).tolist(), k))).all()
    return [
        SimilarTicket(ticket_id=s.ticket_id, subject=s.subject, resolution=s.resolution,
                      similarity=round(s.similarity, 4), url=f"/tickets/{s.ticket_id}")
        for s in to_sources(rows)
    ]


def _reply_out(ticket_id: uuid.UUID, s: ReplySuggestion | None) -> SuggestedReply:
    if s is None:
        return SuggestedReply(ticket_id=ticket_id, status="pending")
    return SuggestedReply(
        ticket_id=ticket_id, status=s.status, draft=s.draft, reason=s.reason, sources=s.sources,
        model_version=s.model_version, approved_by=s.approved_by, approved_at=s.approved_at,
    )


@router.get("/tickets/{ticket_id}/suggested-reply", response_model=SuggestedReply)
async def suggested_reply(ticket_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> SuggestedReply:
    await load_ticket(session, ticket_id)
    suggestion = await session.scalar(select(ReplySuggestion).where(ReplySuggestion.ticket_id == ticket_id))
    return _reply_out(ticket_id, suggestion)


@router.post("/tickets/{ticket_id}/suggested-reply/approve", response_model=SuggestedReply)
async def approve_reply(
    ticket_id: uuid.UUID, body: ApproveReply, session: AsyncSession = Depends(get_session)
) -> SuggestedReply:
    """Record that an agent approved the draft. Sending stays with the agent's helpdesk."""
    suggestion = await session.scalar(select(ReplySuggestion).where(ReplySuggestion.ticket_id == ticket_id))
    if suggestion is None or suggestion.status != "ready":
        raise HTTPException(status.HTTP_409_CONFLICT, "No ready draft to approve")
    suggestion.approved_by = body.approved_by
    suggestion.approved_at = datetime.now(UTC)
    await session.commit()
    return _reply_out(ticket_id, suggestion)


@router.post("/tickets/{ticket_id}/corrections", response_model=list[CorrectionOut])
async def correct_ticket(
    ticket_id: uuid.UUID, body: CorrectionIn, session: AsyncSession = Depends(get_session)
) -> list[Correction]:
    """Fix any of category / priority / team. Sending no fields approves the prediction as-is.

    Either way the ticket leaves the review queue, and changes are stored as corrections (FR10).
    """
    ticket = await session.get(Ticket, ticket_id, with_for_update=True)
    if ticket is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found")
    corrections = []
    for name in ("category", "priority", "team"):
        new_value = getattr(body, name)
        if new_value is not None and new_value != getattr(ticket, name):
            corrections.append(Correction(ticket_id=ticket.id, field=name, old_value=getattr(ticket, name),
                                          new_value=new_value, corrected_by=body.corrected_by))
            setattr(ticket, name, new_value)
    session.add_all(corrections)
    ticket.reviewed_by = body.corrected_by
    ticket.reviewed_at = datetime.now(UTC)
    await session.commit()
    return corrections


@router.get("/review-queue", response_model=list[TicketSummary], tags=["review"])
async def review_queue(
    limit: Annotated[int, Query(ge=1, le=500)] = 50, session: AsyncSession = Depends(get_session)
) -> list[Ticket]:
    """Low-confidence tickets awaiting a human, oldest first."""
    return (
        await session.scalars(
            select(Ticket)
            .where(Ticket.routing == Routing.REVIEW, Ticket.reviewed_at.is_(None))
            .order_by(Ticket.priority.asc().nulls_last(), Ticket.created_at)
            .limit(limit)
        )
    ).all()
