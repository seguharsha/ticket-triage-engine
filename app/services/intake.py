"""Store an incoming ticket exactly once per idempotency key."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.models import Attachment, Ticket, TicketStatus
from app.services.attachments import IncomingFile, storage


@dataclass
class NewTicket:
    idempotency_key: str
    subject: str
    body: str
    source: str = "api"
    external_id: str | None = None
    customer_email: str | None = None
    created_at: datetime | None = None


@dataclass
class IntakeResult:
    id: uuid.UUID
    status: str
    created: bool


async def create_ticket(
    session: AsyncSession, new: NewTicket, files: list[IncomingFile] = ()
) -> IntakeResult:
    values = {
        "id": uuid.uuid4(),
        "idempotency_key": new.idempotency_key,
        "subject": new.subject,
        "body": new.body,
        "source": new.source,
        "external_id": new.external_id,
        "customer_email": new.customer_email,
        "status": TicketStatus.RECEIVED,
    }
    if new.created_at is not None:
        created_at = new.created_at
        values["created_at"] = created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)

    # ON CONFLICT DO NOTHING makes the unique index the arbiter: if two identical webhooks
    # race, the second insert waits for the first to commit and then inserts nothing.
    stmt = (
        pg_insert(Ticket)
        .values(**values)
        .on_conflict_do_nothing(index_elements=["idempotency_key"])
        .returning(Ticket.id)
    )
    ticket_id = (await session.execute(stmt)).scalar_one_or_none()

    if ticket_id is None:
        await session.rollback()
        row = (
            await session.execute(
                select(Ticket.id, Ticket.status).where(Ticket.idempotency_key == new.idempotency_key)
            )
        ).one()
        return IntakeResult(row.id, row.status, created=False)

    for file in files:
        attachment_id = uuid.uuid4()
        path = await run_in_threadpool(storage.save, ticket_id, attachment_id, file.filename, file.data)
        session.add(
            Attachment(
                id=attachment_id,
                ticket_id=ticket_id,
                filename=file.filename,
                content_type=file.content_type,
                size_bytes=len(file.data),
                storage_path=path,
            )
        )
    await session.commit()
    return IntakeResult(ticket_id, TicketStatus.RECEIVED, created=True)
