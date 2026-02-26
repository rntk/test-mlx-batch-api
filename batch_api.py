#!/usr/bin/env python3
"""
Batch API Server with Files API

Implements a subset of the OpenAI Files API and Batch API using only the
Python standard library.

Files API
---------
  POST   /files                   – Upload a file (multipart/form-data or raw body)
  GET    /files                   – List files
  GET    /files/{file_id}         – Retrieve file metadata
  GET    /files/{file_id}/content – Download file content
  DELETE /files/{file_id}         – Delete a file

Batch API
---------
  POST   /batches                 – Create a batch (JSON body: input_file_id, endpoint, …)
  GET    /batches                 – List all batches
  GET    /batches/{batch_id}      – Get batch status
  POST   /batches/{batch_id}/cancel – Cancel a batch
  DELETE /batches/{batch_id}      – Delete a batch

Typical workflow
~~~~~~~~~~~~~~~~
1.  Upload your .jsonl file:
        POST /files  (multipart/form-data, purpose=batch)
    → returns FileObject with an ``id`` such as ``file-<hex>``

2.  Create a batch:
        POST /batches
        {"input_file_id": "file-<hex>", "endpoint": "/v1/chat/completions",
         "completion_window": "24h"}
    → returns Batch object with status ``in_progress``

3.  Poll until complete:
        GET /batches/{batch_id}
    → status transitions: validating → in_progress → finalizing → completed

4.  Download results:
        GET /files/{output_file_id}/content
    (output_file_id comes from the completed Batch object)

5.  Optionally cancel:
        POST /batches/{batch_id}/cancel
"""

import email
import json
import os
import shutil
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOST = "0.0.0.0"
PORT = 8788
BATCHES_DIR = Path("./batches")
FILES_DIR = Path("./files")

BATCHES_DIR.mkdir(parents=True, exist_ok=True)
FILES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# File-store helpers
# ---------------------------------------------------------------------------

def _file_meta_path(file_id: str) -> Path:
    return FILES_DIR / f"{file_id}.meta.json"


def _file_content_path(file_id: str) -> Path:
    return FILES_DIR / file_id


def _read_file_meta(file_id: str) -> dict | None:
    path = _file_meta_path(file_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _list_all_file_metas() -> list[dict]:
    files = []
    for p in FILES_DIR.glob("*.meta.json"):
        with open(p) as f:
            files.append(json.load(f))
    files.sort(key=lambda x: x.get("created_at", 0))
    return files


def _store_file(data: bytes, purpose: str, filename: str) -> dict:
    """Write *data* into the file store and return its FileObject metadata."""
    file_id = f"file-{uuid.uuid4().hex}"
    _file_content_path(file_id).write_bytes(data)
    meta = {
        "id": file_id,
        "object": "file",
        "bytes": len(data),
        "created_at": int(time.time()),
        "filename": filename,
        "purpose": purpose,
        "status": "processed",
    }
    with open(_file_meta_path(file_id), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# ---------------------------------------------------------------------------
# Multipart form-data parser (stdlib only)
# ---------------------------------------------------------------------------

def _parse_multipart(headers, body: bytes) -> dict[str, bytes]:
    """Return {field_name: raw_bytes} for each part in a multipart/form-data body."""
    content_type = headers.get("Content-Type", "")
    # Prepend a minimal header so email.message_from_bytes can parse it.
    raw = f"Content-Type: {content_type}\r\n\r\n".encode() + body
    msg = email.message_from_bytes(raw)
    fields: dict[str, bytes] = {}
    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        cd = part.get("Content-Disposition", "")
        if not cd:
            continue
        name: str | None = None
        for token in cd.split(";"):
            token = token.strip()
            if token.startswith("name="):
                name = token[5:].strip('"')
                break
        if name is None:
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            payload = (part.get_payload() or "").encode()
        fields[name] = payload
    return fields


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class BatchAPIHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):  # noqa: A002
        print(f"[{self.log_date_time_string()}] {format % args}")

    # ------------------------------------------------------------------
    # Low-level response helpers
    # ------------------------------------------------------------------

    def _send_json(self, status: int, data):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, content_type: str, data: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str):
        self._send_json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self) -> list[str]:
        """Return path segments with empty strings removed."""
        return [s for s in urlparse(self.path).path.split("/") if s]

    def do_GET(self):
        parts = self._route()
        match parts:
            case ["files"]:
                self._files_list()
            case ["files", file_id]:
                self._files_retrieve(file_id)
            case ["files", file_id, "content"]:
                self._files_content(file_id)
            case ["batches"]:
                self._batches_list()
            case ["batches", batch_id]:
                self._batches_retrieve(batch_id)
            case _:
                self._error(404, "Not found")

    def do_POST(self):
        parts = self._route()
        match parts:
            case ["files"]:
                self._files_create()
            case ["batches"]:
                self._batches_create()
            case ["batches", batch_id, "cancel"]:
                self._batches_cancel(batch_id)
            case _:
                self._error(404, "Not found")

    def do_DELETE(self):
        parts = self._route()
        match parts:
            case ["files", file_id]:
                self._files_delete(file_id)
            case ["batches", batch_id]:
                self._batches_delete(batch_id)
            case _:
                self._error(404, "Not found")

    # ==================================================================
    # Files API
    # ==================================================================

    def _files_create(self):
        """POST /files — upload a file."""
        body = self._read_body()
        ct = self.headers.get("Content-Type", "")

        if "multipart/form-data" in ct:
            fields = _parse_multipart(self.headers, body)
            file_bytes: bytes = fields.get("file", b"")
            purpose: str = (fields.get("purpose") or b"batch").decode().strip()
            filename: str = "upload.jsonl"
        else:
            # Fallback: treat the raw body as the file content.
            file_bytes = body
            purpose = "batch"
            filename = "upload.jsonl"

        if not file_bytes:
            self._error(400, "No file content provided")
            return

        meta = _store_file(file_bytes, purpose, filename)
        self._send_json(200, meta)

    def _files_list(self):
        """GET /files — list all files."""
        files = _list_all_file_metas()
        self._send_json(200, {
            "object": "list",
            "data": files,
            "first_id": files[0]["id"] if files else None,
            "last_id": files[-1]["id"] if files else None,
            "has_more": False,
        })

    def _files_retrieve(self, file_id: str):
        """GET /files/{file_id} — return file metadata."""
        meta = _read_file_meta(file_id)
        if meta is None:
            self._error(404, f"No such file: {file_id}")
            return
        self._send_json(200, meta)

    def _files_content(self, file_id: str):
        """GET /files/{file_id}/content — return raw file bytes."""
        meta = _read_file_meta(file_id)
        if meta is None:
            self._error(404, f"No such file: {file_id}")
            return
        content_path = _file_content_path(file_id)
        if not content_path.exists():
            self._error(404, f"Content for file {file_id} not found")
            return
        self._send_bytes(200, "application/jsonl", content_path.read_bytes())

    def _files_delete(self, file_id: str):
        """DELETE /files/{file_id} — delete a file."""
        if _read_file_meta(file_id) is None:
            self._error(404, f"No such file: {file_id}")
            return
        _file_content_path(file_id).unlink(missing_ok=True)
        _file_meta_path(file_id).unlink(missing_ok=True)
        self._send_json(200, {"id": file_id, "object": "file", "deleted": True})

    # ==================================================================
    # Batches API
    # ==================================================================

    def _batches_create(self):
        """POST /batches — create a batch job referencing an uploaded file."""
        body = self._read_body()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._error(400, "Request body must be valid JSON")
            return

        input_file_id: str | None = data.get("input_file_id")
        endpoint: str = data.get("endpoint", "/v1/chat/completions")
        completion_window: str = data.get("completion_window", "24h")
        metadata = data.get("metadata")

        if not input_file_id:
            self._error(400, "input_file_id is required")
            return

        file_meta = _read_file_meta(input_file_id)
        if file_meta is None:
            self._error(404, f"No such file: {input_file_id}")
            return

        # Count non-empty lines in the input file.
        content_path = _file_content_path(input_file_id)
        total_requests = 0
        if content_path.exists():
            total_requests = sum(
                1 for line in content_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

        batch_id = f"batch_{uuid.uuid4().hex}"
        now = int(time.time())
        batch_dir = BATCHES_DIR / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        batch_obj = {
            "id": batch_id,
            "object": "batch",
            "endpoint": endpoint,
            "errors": None,
            "input_file_id": input_file_id,
            "completion_window": completion_window,
            "status": "in_progress",
            "output_file_id": None,
            "error_file_id": None,
            "created_at": now,
            "in_progress_at": now,
            "expires_at": now + 86400,
            "finalizing_at": None,
            "completed_at": None,
            "failed_at": None,
            "expired_at": None,
            "cancelling_at": None,
            "cancelled_at": None,
            "request_counts": {
                "total": total_requests,
                "completed": 0,
                "failed": 0,
            },
            "metadata": metadata,
        }

        with open(batch_dir / "metadata.json", "w") as f:
            json.dump(batch_obj, f, indent=2)

        self._send_json(200, batch_obj)

    def _batches_list(self):
        """GET /batches — list all batches."""
        batches = []
        for batch_dir in BATCHES_DIR.iterdir():
            if not batch_dir.is_dir():
                continue
            meta_path = batch_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    batches.append(json.load(f))
        batches.sort(key=lambda x: x.get("created_at", 0))
        self._send_json(200, {
            "object": "list",
            "data": batches,
            "first_id": batches[0]["id"] if batches else None,
            "last_id": batches[-1]["id"] if batches else None,
            "has_more": False,
        })

    def _batches_retrieve(self, batch_id: str):
        """GET /batches/{batch_id} — return batch status."""
        meta_path = BATCHES_DIR / batch_id / "metadata.json"
        if not meta_path.exists():
            self._error(404, f"No such batch: {batch_id}")
            return
        with open(meta_path) as f:
            self._send_json(200, json.load(f))

    def _batches_cancel(self, batch_id: str):
        """POST /batches/{batch_id}/cancel — cancel an in-progress batch."""
        meta_path = BATCHES_DIR / batch_id / "metadata.json"
        if not meta_path.exists():
            self._error(404, f"No such batch: {batch_id}")
            return
        with open(meta_path) as f:
            batch = json.load(f)
        terminal = {"completed", "cancelled", "expired", "failed"}
        if batch["status"] in terminal:
            self._error(400, f"Cannot cancel batch with status '{batch['status']}'")
            return
        now = int(time.time())
        batch["status"] = "cancelling"
        batch["cancelling_at"] = now
        with open(meta_path, "w") as f:
            json.dump(batch, f, indent=2)
        self._send_json(200, batch)

    def _batches_delete(self, batch_id: str):
        """DELETE /batches/{batch_id} — permanently remove a batch."""
        batch_dir = BATCHES_DIR / batch_id
        if not batch_dir.exists():
            self._error(404, f"No such batch: {batch_id}")
            return
        shutil.rmtree(batch_dir)
        self._send_json(200, {"id": batch_id, "object": "batch", "deleted": True})


# ---------------------------------------------------------------------------
# Worker helpers (called by background processors, not by the HTTP layer)
# ---------------------------------------------------------------------------

def get_pending_batches() -> list[dict]:
    """Return all batches whose status is ``in_progress``, ``validating``, or
    ``processing``.

    Each entry is a dict with:

    * ``metadata``       – the full Batch object dict
    * ``batch_dir``      – :class:`~pathlib.Path` to the batch directory
    * ``input_file_path``– :class:`~pathlib.Path` to the raw input JSONL, or
                           ``None`` if the file is missing
    """
    pending = []
    for batch_dir in BATCHES_DIR.iterdir():
        if not batch_dir.is_dir():
            continue
        meta_path = batch_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            metadata = json.load(f)
        if metadata.get("status") not in {"in_progress", "validating", "processing"}:
            continue
        input_file_id: str | None = metadata.get("input_file_id")
        input_file_path: Path | None = None
        if input_file_id:
            p = _file_content_path(input_file_id)
            input_file_path = p if p.exists() else None
        pending.append({
            "metadata": metadata,
            "batch_dir": batch_dir,
            "input_file_path": input_file_path,
        })
    return pending


def get_batch_input_file(batch_id: str) -> Path | None:
    """Return the :class:`~pathlib.Path` to the input JSONL for *batch_id*,
    or ``None`` if not found."""
    meta_path = BATCHES_DIR / batch_id / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        metadata = json.load(f)
    input_file_id: str | None = metadata.get("input_file_id")
    if not input_file_id:
        return None
    p = _file_content_path(input_file_id)
    return p if p.exists() else None


def mark_batch_complete(
    batch_id: str,
    results: list,
    errors: list | None = None,
) -> None:
    """Save *results* (and optionally *errors*) as output files in the Files
    store, then mark the batch as ``completed``.

    Each element of *results* / *errors* is serialised as one JSONL line.
    The ``output_file_id`` (and ``error_file_id``) fields on the Batch object
    are populated so callers can retrieve them via
    ``GET /files/{output_file_id}/content``.
    """
    batch_dir = BATCHES_DIR / batch_id
    if not batch_dir.exists():
        raise ValueError(f"Batch '{batch_id}' not found")

    meta_path = batch_dir / "metadata.json"
    with open(meta_path) as f:
        batch = json.load(f)

    now = int(time.time())

    if results:
        out_meta = _store_file(
            data=("\n".join(json.dumps(r) for r in results) + "\n").encode(),
            purpose="batch_output",
            filename=f"{batch_id}_output.jsonl",
        )
        batch["output_file_id"] = out_meta["id"]
        batch["request_counts"]["completed"] = len(results)

    if errors:
        err_meta = _store_file(
            data=("\n".join(json.dumps(e) for e in errors) + "\n").encode(),
            purpose="batch_output",
            filename=f"{batch_id}_errors.jsonl",
        )
        batch["error_file_id"] = err_meta["id"]
        batch["request_counts"]["failed"] = len(errors)

    batch["status"] = "completed"
    batch["finalizing_at"] = now
    batch["completed_at"] = now
    batch["request_counts"]["total"] = (
        batch["request_counts"]["completed"] + batch["request_counts"]["failed"]
    )

    with open(meta_path, "w") as f:
        json.dump(batch, f, indent=2)


# Backward-compatible alias
def mark_batch_ready(batch_id: str, results: list) -> None:
    """Alias for :func:`mark_batch_complete` (backward compatibility)."""
    mark_batch_complete(batch_id, results)


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def run_server(host: str = HOST, port: int = PORT):
    """Start the HTTP server."""
    httpd = HTTPServer((host, port), BatchAPIHandler)
    print(f"Batch API Server  http://{host}:{port}")
    print(f"  Files store : {FILES_DIR.absolute()}")
    print(f"  Batches dir : {BATCHES_DIR.absolute()}")
    print()
    print("Files API:")
    print("  POST   /files                    Upload a file (multipart/form-data, purpose=batch)")
    print("  GET    /files                    List files")
    print("  GET    /files/{id}               Retrieve file metadata")
    print("  GET    /files/{id}/content       Download file content")
    print("  DELETE /files/{id}               Delete a file")
    print()
    print("Batch API:")
    print("  POST   /batches                  Create a batch (JSON: input_file_id, endpoint, …)")
    print("  GET    /batches                  List batches")
    print("  GET    /batches/{id}             Get batch status")
    print("  POST   /batches/{id}/cancel      Cancel a batch")
    print("  DELETE /batches/{id}             Delete a batch")
    print()
    print("Press Ctrl+C to stop.")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
