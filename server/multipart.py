"""Minimal ``multipart/form-data`` reader with a hard byte cap.

Only what this backend needs: one file part per request, counted while reading so a
forged ``Content-Length`` cannot bypass the limit. No third-party packages.
"""

from email import policy
from email.parser import BytesParser


class UploadError(Exception):
    """A rejected upload with an HTTP status and a stable code."""

    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def boundary_from_content_type(content_type):
    """Return the multipart boundary, or raise a precise UploadError."""
    if not content_type:
        raise UploadError("missing_content_type", "Content-Type must be multipart/form-data")
    pieces = content_type.split(";")
    if pieces[0].strip().lower() != "multipart/form-data":
        raise UploadError("unsupported_content_type",
                          "Send the replay as multipart/form-data with one file field")
    for piece in pieces[1:]:
        key, _, value = piece.strip().partition("=")
        if key.strip().lower() == "boundary" and value.strip():
            boundary = value.strip()
            if len(boundary) >= 2 and boundary[0] == '"' and boundary[-1] == '"':
                boundary = boundary[1:-1]
            return boundary
    raise UploadError("missing_boundary", "multipart Content-Type has no boundary")


def read_body(rfile, content_length, envelope_limit):
    """Read exactly ``content_length`` bytes, stopping as soon as the cap is crossed."""
    if content_length <= 0:
        raise UploadError("empty_body", "The request body is empty")
    remaining = content_length
    total = 0
    chunks = []
    while remaining > 0:
        chunk = rfile.read(min(256 * 1024, remaining))
        if not chunk:
            raise UploadError("incomplete_body",
                              "The connection closed before the declared Content-Length")
        total += len(chunk)
        if total > envelope_limit:
            raise UploadError("upload_too_large",
                              f"Upload exceeds the {envelope_limit} byte limit", 413)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def parse_file_part(content_type, body):
    """Return ``{"field", "filename", "data"}`` for the single file part."""
    boundary = boundary_from_content_type(content_type)
    header = (f"Content-Type: multipart/form-data; boundary={boundary}\r\n"
              "MIME-Version: 1.0\r\n\r\n").encode("latin-1")
    message = BytesParser(policy=policy.default).parsebytes(header + body)
    if not message.is_multipart():
        raise UploadError("invalid_multipart", "The body is not valid multipart data")
    parts = []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition") or ""
        filename = part.get_filename()
        if filename is not None or name == "file":
            parts.append({"field": name, "filename": filename or "",
                          "data": part.get_payload(decode=True) or b""})
    if not parts:
        raise UploadError("missing_file",
                          "No file part found; use form field 'file' with the replay")
    if len(parts) > 1:
        raise UploadError("too_many_files", "Send exactly one replay per request")
    return parts[0]


def drain(rfile, count):
    """Read and discard up to ``count`` bytes so an early rejection is not a reset.

    A client that is still writing when the server closes usually sees
    ``ConnectionAbortedError`` instead of the 413 body. Draining what it already sent
    keeps the response readable; the cap bounds the work.
    """
    remaining = max(0, int(count))
    while remaining > 0:
        chunk = rfile.read(min(256 * 1024, remaining))
        if not chunk:
            break
        remaining -= len(chunk)
