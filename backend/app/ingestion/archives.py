"""Safety checks for archive-backed formats.

DOCX, PPTX and XLSX are ZIP files containing XML. That makes them a
decompression-bomb vector: a few hundred kilobytes of highly compressible XML
can expand to gigabytes and exhaust memory before any size limit written in
terms of *upload* size gets a chance to fire.

The upload limit bounds what arrives. This bounds what it becomes.
"""

import io
import zipfile

from app.core.logging import get_logger
from app.ingestion.base import UnparsableDocumentError

logger = get_logger(__name__)

#: Ceiling on total uncompressed size across all entries.
MAX_UNCOMPRESSED_BYTES = 400 * 1024 * 1024

#: Ceiling on the expansion factor. Real Office files sit well under 50x;
#: a crafted bomb reaches thousands.
MAX_COMPRESSION_RATIO = 200


def guard_ooxml(data: bytes, *, filename: str) -> None:
    """Reject archives that would expand unreasonably.

    Reads only the central directory, not the entries themselves, so the check
    itself cannot be the thing that exhausts memory.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            uncompressed = sum(entry.file_size for entry in entries)
    except zipfile.BadZipFile as exc:
        raise UnparsableDocumentError(
            f"{filename} is not a valid Office file (its container is corrupt)."
        ) from exc

    if uncompressed > MAX_UNCOMPRESSED_BYTES:
        logger.warning(
            "archive_too_large",
            extra={"source_name": filename, "uncompressed": uncompressed},
        )
        raise UnparsableDocumentError(f"{filename} expands to more than 400 MB of content.")

    ratio = uncompressed / max(len(data), 1)
    if ratio > MAX_COMPRESSION_RATIO:
        logger.warning(
            "archive_ratio_suspicious",
            extra={"source_name": filename, "ratio": round(ratio, 1)},
        )
        raise UnparsableDocumentError(
            f"{filename} has an implausible compression ratio and was rejected."
        )
