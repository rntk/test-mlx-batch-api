#!/usr/bin/env python3
"""
Script to send a batch request following the OpenAI Batch API workflow:

  1. Build a JSONL file from the source text (one chat-completion request per line).
  2. Upload the JSONL to POST /files  (multipart/form-data, purpose=batch).
  3. Create a batch via POST /batches referencing the uploaded file ID.
  4. Poll GET /batches/{batch_id} until the batch reaches a terminal state.
  5. Download results from GET /files/{output_file_id}/content.
"""

import email.mime.multipart
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FILE_PATH = "/Users/rnt/dev/python/mlx/all_posts_raw_1771928104.txt"
BASE_URL = "http://localhost:8788"
MODEL = "gpt-3.5-turbo-0125"
POLL_INTERVAL = 5  # seconds between status checks


# ---------------------------------------------------------------------------
# Step 1 – Build JSONL
# ---------------------------------------------------------------------------

def build_jsonl(file_path: str, num_texts: int = 4) -> bytes:
    """Read *num_texts* non-empty lines from *file_path* and return a JSONL
    payload compatible with the /v1/chat/completions batch endpoint."""
    with open(file_path, encoding="utf-8") as fh:
        lines = [l.strip() for l in fh if l.strip()]

    if len(lines) < num_texts:
        raise ValueError(
            f"Expected at least {num_texts} non-empty lines in {file_path}, "
            f"found {len(lines)}."
        )

    records = []
    for idx, text in enumerate(lines[:num_texts], start=1):
        records.append(json.dumps({
            "custom_id": f"request-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Summarize the following text:\n\n{text}"},
                ],
                "max_tokens": 256,
            },
        }))

    return ("\n".join(records) + "\n").encode()


# ---------------------------------------------------------------------------
# Step 2 – Upload file to /files
# ---------------------------------------------------------------------------

def upload_file(jsonl_bytes: bytes, filename: str = "batchinput.jsonl") -> dict:
    """Upload *jsonl_bytes* via multipart/form-data to POST /files and return
    the FileObject dict."""
    boundary = "----BatchBoundary" + os.urandom(8).hex()
    body_parts: list[bytes] = []

    def _part(name: str, value: bytes, extra_headers: str = "") -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"{extra_headers}\r\n'
            f"\r\n"
        ).encode() + value + b"\r\n"

    body_parts.append(_part("purpose", b"batch"))
    body_parts.append(_part(
        "file",
        jsonl_bytes,
        f'; filename="{filename}"\r\nContent-Type: application/jsonl',
    ))
    body_parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(body_parts)
    req = urllib.request.Request(
        f"{BASE_URL}/files",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"File upload failed [{exc.code}]: {exc.read().decode()}") from exc


# ---------------------------------------------------------------------------
# Step 3 – Create batch
# ---------------------------------------------------------------------------

def create_batch(input_file_id: str, endpoint: str = "/v1/chat/completions") -> dict:
    """Create a batch job referencing *input_file_id* and return the Batch object."""
    payload = json.dumps({
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": "24h",
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/batches",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Batch creation failed [{exc.code}]: {exc.read().decode()}") from exc


# ---------------------------------------------------------------------------
# Step 4 – Poll batch status
# ---------------------------------------------------------------------------

TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}


def poll_batch(batch_id: str) -> dict:
    """Poll GET /batches/{batch_id} until a terminal state is reached."""
    url = f"{BASE_URL}/batches/{batch_id}"
    while True:
        try:
            with urllib.request.urlopen(url) as resp:
                batch = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Status check failed [{exc.code}]: {exc.read().decode()}") from exc

        status = batch.get("status", "unknown")
        counts = batch.get("request_counts", {})
        print(
            f"  status={status}  "
            f"completed={counts.get('completed', 0)}  "
            f"failed={counts.get('failed', 0)}  "
            f"total={counts.get('total', 0)}"
        )

        if status in TERMINAL_STATES:
            return batch

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Step 5 – Download results
# ---------------------------------------------------------------------------

def download_file_content(file_id: str) -> bytes:
    """Fetch raw bytes from GET /files/{file_id}/content."""
    url = f"{BASE_URL}/files/{file_id}/content"
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Download failed [{exc.code}]: {exc.read().decode()}") from exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        sys.exit(1)

    # 1. Build JSONL
    print("Building JSONL payload …")
    jsonl_bytes = build_jsonl(FILE_PATH)
    print(f"  {len(jsonl_bytes)} bytes, {jsonl_bytes.count(b'\n')} requests")

    # 2. Upload to /files
    print("\nUploading input file to /files …")
    file_obj = upload_file(jsonl_bytes)
    file_id = file_obj["id"]
    print(f"  file_id  = {file_id}")
    print(f"  filename = {file_obj.get('filename')}")
    print(f"  bytes    = {file_obj.get('bytes')}")

    # 3. Create batch
    print("\nCreating batch on /batches …")
    batch = create_batch(file_id)
    batch_id = batch["id"]
    print(f"  batch_id = {batch_id}")
    print(f"  status   = {batch['status']}")

    # Save initial batch metadata
    batch_meta_file = f"batch_response_{batch_id}.json"
    with open(batch_meta_file, "w") as fh:
        json.dump(batch, fh, indent=2)
    print(f"  metadata saved → {batch_meta_file}")

    # 4. Poll until done
    print("\nPolling batch status …")
    batch = poll_batch(batch_id)
    final_status = batch["status"]
    print(f"\nBatch finished with status: {final_status}")

    # Persist final metadata
    with open(batch_meta_file, "w") as fh:
        json.dump(batch, fh, indent=2)

    if final_status != "completed":
        print(f"Batch did not complete successfully (status={final_status}). Exiting.")
        sys.exit(1)

    # 5. Download results
    output_file_id = batch.get("output_file_id")
    if output_file_id:
        print(f"\nDownloading results from /files/{output_file_id}/content …")
        content = download_file_content(output_file_id)
        results_file = f"results_{batch_id}.jsonl"
        with open(results_file, "wb") as fh:
            fh.write(content)
        print(f"  Results saved → {results_file}")

    error_file_id = batch.get("error_file_id")
    if error_file_id:
        print(f"\nDownloading error file from /files/{error_file_id}/content …")
        content = download_file_content(error_file_id)
        errors_file = f"errors_{batch_id}.jsonl"
        with open(errors_file, "wb") as fh:
            fh.write(content)
        print(f"  Errors saved → {errors_file}")


if __name__ == "__main__":
    main()