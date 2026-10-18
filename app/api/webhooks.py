from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.tickets import accept
from app.config import settings
from app.db import get_session
from app.schemas import TicketAccepted, WebhookTicket
from app.services import signatures
from app.services.intake import NewTicket, create_ticket

router = APIRouter(tags=["webhooks"])


@router.post(
    "/webhooks/tickets",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TicketAccepted,
    openapi_extra={
        "requestBody": {"content": {"application/json": {"schema": WebhookTicket.model_json_schema()}}}
    },
)
async def receive_webhook(
    request: Request,
    response: Response,
    x_timestamp: Annotated[str, Header(description="Unix seconds; part of the signed payload")],
    x_signature: Annotated[str, Header(description='"sha256=" + hex HMAC of "<timestamp>.<raw body>"')],
    idempotency_key: Annotated[str | None, Header(max_length=255)] = None,
    session: AsyncSession = Depends(get_session),
) -> TicketAccepted:
    # Verify against the exact raw bytes; re-serialized JSON would not match the signature.
    raw = await request.body()
    try:
        signatures.verify(
            settings.webhook_secret, x_timestamp, raw, x_signature, settings.webhook_tolerance_seconds
        )
    except signatures.InvalidSignature as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid signature: {exc}") from exc

    try:
        payload = WebhookTicket.model_validate_json(raw)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    # Helpdesks resend the same ticket on timeout; their ticket id is a natural idempotency key.
    key = idempotency_key or f"webhook:{payload.external_id}"
    result = await create_ticket(
        session,
        NewTicket(
            idempotency_key=key,
            subject=payload.subject,
            body=payload.body,
            source="webhook",
            external_id=payload.external_id,
            customer_email=payload.customer_email,
        ),
    )
    return await accept(result, response)
