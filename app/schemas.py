import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.taxonomy import CATEGORIES, PRIORITIES, TEAMS

Category = Literal[tuple(CATEGORIES)]
Priority = Literal[tuple(PRIORITIES)]
Team = Literal[tuple(TEAMS)]


class TicketAccepted(BaseModel):
    id: uuid.UUID
    status: str
    duplicate_submission: bool = Field(
        False, description="True when this idempotency key was already used; the original ticket is returned."
    )


class AttachmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    field: str
    predicted_value: str
    confidence: float | None
    model_version: str


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source: str
    subject: str
    body: str
    clean_text: str | None
    status: str
    error: str | None
    incident_id: uuid.UUID | None
    dedup_similarity: float | None
    category: str | None
    priority: str | None
    team: str | None
    routing: str | None
    reviewed_by: str | None
    reviewed_at: datetime | None
    created_at: datetime
    received_at: datetime
    processed_at: datetime | None
    attachments: list[AttachmentOut]
    predictions: list[PredictionOut]


class TicketSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    subject: str
    status: str
    category: str | None
    priority: str | None
    team: str | None
    routing: str | None
    incident_id: uuid.UUID | None
    created_at: datetime


class WebhookTicket(BaseModel):
    external_id: str = Field(min_length=1, max_length=255)
    subject: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=50_000)
    customer_email: str | None = Field(None, max_length=320)


class SimilarTicket(BaseModel):
    ticket_id: uuid.UUID
    subject: str
    resolution: str
    similarity: float
    url: str


class ReplySource(BaseModel):
    ticket_id: uuid.UUID
    subject: str
    similarity: float
    cited: bool
    url: str


class SuggestedReply(BaseModel):
    ticket_id: uuid.UUID
    status: Literal["pending", "ready", "no_suggestion"]
    draft: str | None = None
    reason: str | None = None
    sources: list[ReplySource] = []
    model_version: str | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    sent_automatically: Literal[False] = Field(False, description="Drafts are never sent by this service.")


class ApproveReply(BaseModel):
    approved_by: str = Field(min_length=1, max_length=255)


class CorrectionIn(BaseModel):
    corrected_by: str = Field(min_length=1, max_length=255)
    category: Category | None = None
    priority: Priority | None = None
    team: Team | None = None


class CorrectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    field: str
    old_value: str | None
    new_value: str
    corrected_by: str
    created_at: datetime


class IncidentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    category: str | None
    priority: str | None
    team: str | None
    ticket_count: int
    status: str
    merged_into: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


class IncidentDetail(IncidentOut):
    tickets: list[TicketSummary]


class MergeIn(BaseModel):
    source_incident_id: uuid.UUID
    corrected_by: str = Field(min_length=1, max_length=255)


class SplitIn(BaseModel):
    ticket_ids: list[uuid.UUID] = Field(min_length=1)
    corrected_by: str = Field(min_length=1, max_length=255)
    title: str | None = Field(None, max_length=200)
