"""Attachment validation and storage. In v1 files are stored, never read."""

import re
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from fastapi import HTTPException, UploadFile, status

from app.config import settings


@dataclass
class IncomingFile:
    filename: str
    content_type: str
    data: bytes


def safe_filename(name: str | None) -> str:
    # Strip any client-supplied directory (either separator style), then whitelist characters.
    base = PureWindowsPath(PurePosixPath(name or "").name).name
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    return base[:200] or "file"


async def read_uploads(files: list[UploadFile]) -> list[IncomingFile]:
    """Validate count, type and size before anything is written to the database."""
    if len(files) > settings.max_attachments_per_ticket:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"At most {settings.max_attachments_per_ticket} attachments per ticket",
        )
    result = []
    for upload in files:
        content_type = (upload.content_type or "").split(";")[0].strip().lower()
        if content_type not in settings.allowed_attachment_types:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"Attachment type {content_type or 'unknown'!r} not allowed",
            )
        data = await upload.read(settings.max_attachment_bytes + 1)
        if len(data) > settings.max_attachment_bytes:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"Attachment {upload.filename!r} exceeds {settings.max_attachment_bytes} bytes",
            )
        result.append(IncomingFile(safe_filename(upload.filename), content_type, data))
    return result


class LocalStorage:
    """Files on local disk (a Docker volume). Swap for an S3 implementation with the same save()."""

    def __init__(self, root: str):
        self.root = Path(root)

    def save(self, ticket_id: uuid.UUID, attachment_id: uuid.UUID, filename: str, data: bytes) -> str:
        directory = self.root / str(ticket_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{attachment_id}_{filename}"
        path.write_bytes(data)
        return str(path)


storage = LocalStorage(settings.attachment_dir)
