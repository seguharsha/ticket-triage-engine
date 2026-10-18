import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Index, Integer, Sequence, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class Base(DeclarativeBase):
    pass


# Numbers each incident assignment in lock order; dedup uses it to find assignments made while it searched.
DEDUP_SEQUENCE = Sequence("ticket_dedup_seq", metadata=Base.metadata)


class TicketStatus:
    RECEIVED = "received"      # stored, job queued
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"          # retries exhausted, sent to dead letters
    RESOLVED = "resolved"      # historical ticket with an agent reply; source for reply suggestions


class Routing:
    AUTO = "auto"
    REVIEW = "review"


class IncidentStatus:
    OPEN = "open"
    RESOLVED = "resolved"
    MERGED = "merged"


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(64))
    priority: Mapped[str | None] = mapped_column(String(8))
    team: Mapped[str | None] = mapped_column(String(64))
    ticket_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default=IncidentStatus.OPEN, index=True)
    merged_into: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("incidents.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    tickets: Mapped[list["Ticket"]] = relationship(back_populates="incident", foreign_keys="Ticket.incident_id")


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True)
    source: Mapped[str] = mapped_column(String(32), default="api")
    external_id: Mapped[str | None] = mapped_column(String(255))
    customer_email: Mapped[str | None] = mapped_column(String(320))
    subject: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    clean_text: Mapped[str | None] = mapped_column(Text)
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default=TicketStatus.RECEIVED, index=True)
    error: Mapped[str | None] = mapped_column(Text)

    # Dedup result
    incident_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("incidents.id"), index=True)
    dedup_similarity: Mapped[float | None] = mapped_column(Float)
    dedup_seq: Mapped[int | None] = mapped_column(BigInteger, index=True)

    # Final labels (classifier, LLM fallback, or human correction) and routing decision
    category: Mapped[str | None] = mapped_column(String(64))
    priority: Mapped[str | None] = mapped_column(String(8))
    team: Mapped[str | None] = mapped_column(String(64))
    routing: Mapped[str | None] = mapped_column(String(16), index=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(255))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # For historical resolved tickets: the agent reply that resolved it
    resolution: Mapped[str | None] = mapped_column(Text)

    # created_at is when the customer wrote it (can be set on replay); received_at is server time.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    incident: Mapped[Incident | None] = relationship(back_populates="tickets", foreign_keys=[incident_id])
    attachments: Mapped[list["Attachment"]] = relationship(back_populates="ticket", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="ticket", cascade="all, delete-orphan")

    __table_args__ = (
        Index(
            "ix_tickets_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # Reply retrieval only ever searches resolved tickets; a partial index keeps that search small.
        Index(
            "ix_tickets_resolved_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_where=text("status = 'resolved'"),
        ),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickets.id", ondelete="CASCADE"), index=True)
    field: Mapped[str] = mapped_column(String(32))  # category | priority | team
    predicted_value: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float | None] = mapped_column(Float)  # None for LLM predictions (no calibrated score)
    model_version: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    ticket: Mapped[Ticket] = relationship(back_populates="predictions")


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickets.id", ondelete="CASCADE"), index=True)
    field: Mapped[str] = mapped_column(String(32))  # category | priority | team | incident
    old_value: Mapped[str | None] = mapped_column(String(255))
    new_value: Mapped[str] = mapped_column(String(255))
    corrected_by: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Attachment(Base):
    __tablename__ = "attachments"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    ticket_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickets.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(Integer)
    storage_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    ticket: Mapped[Ticket] = relationship(back_populates="attachments")


class ReplySuggestion(Base):
    __tablename__ = "reply_suggestions"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickets.id", ondelete="CASCADE"), unique=True)
    status: Mapped[str] = mapped_column(String(32))  # ready | no_suggestion
    reason: Mapped[str | None] = mapped_column(Text)
    draft: Mapped[str | None] = mapped_column(Text)
    # [{ticket_id, subject, similarity, cited}] for every retrieved source
    sources: Mapped[list] = mapped_column(JSONB, default=list)
    model_version: Mapped[str | None] = mapped_column(String(128))
    # Drafts are never sent automatically; a human approves (and sends from their helpdesk).
    approved_by: Mapped[str | None] = mapped_column(String(255))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LLMUsage(Base):
    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tickets.id", ondelete="SET NULL"), index=True)
    purpose: Mapped[str] = mapped_column(String(32))  # classify | draft
    model: Mapped[str] = mapped_column(String(128))
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
