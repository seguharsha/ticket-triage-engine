import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import get_session
from app.models import Incident
from app.schemas import IncidentDetail, IncidentOut, MergeIn, SplitIn
from app.services import incidents as service

router = APIRouter(prefix="/incidents", tags=["incidents"])


@router.get("", response_model=list[IncidentOut])
async def list_incidents(
    status_: Annotated[str, Query(alias="status")] = "open",
    min_tickets: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    session: AsyncSession = Depends(get_session),
) -> list[Incident]:
    """Incidents sorted by size, then priority (P1 first)."""
    return (
        await session.scalars(
            select(Incident)
            .where(Incident.status == status_, Incident.ticket_count >= min_tickets)
            .order_by(Incident.ticket_count.desc(), Incident.priority.asc().nulls_last(), Incident.created_at)
            .limit(limit)
        )
    ).all()


@router.get("/{incident_id}", response_model=IncidentDetail)
async def get_incident(incident_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> Incident:
    incident = await session.scalar(
        select(Incident)
        .options(selectinload(Incident.tickets))
        .where(Incident.id == incident_id)
    )
    if incident is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Incident not found")
    return incident


@router.post("/{incident_id}/merge", response_model=IncidentOut)
async def merge_incident(
    incident_id: uuid.UUID, body: MergeIn, session: AsyncSession = Depends(get_session)
) -> Incident:
    """Move every ticket of `source_incident_id` into this incident; the source is marked merged."""
    try:
        return await service.merge(session, incident_id, body.source_incident_id, body.corrected_by)
    except service.IncidentError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc


@router.post("/{incident_id}/split", response_model=IncidentOut, status_code=status.HTTP_201_CREATED)
async def split_incident(
    incident_id: uuid.UUID, body: SplitIn, session: AsyncSession = Depends(get_session)
) -> Incident:
    """Move the given tickets out of this incident into a new one."""
    try:
        return await service.split(session, incident_id, body.ticket_ids, body.corrected_by, body.title)
    except service.IncidentError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc
