#!/usr/bin/env python3
"""
Simple Batch API Server

A pure Python HTTP server for batch file upload and download.
- Upload batch files via POST /batches
- Download results via GET /batches/{batch_id}/results
- Results are only available if a .ready file exists
"""

import json
import os
import uuid
import hashlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from pathlib import Path

# Configuration
HOST = "0.0.0.0"
PORT = 8788
BATCHES_DIR = Path("./batches")

# Ensure batches directory exists
BATCHES_DIR.mkdir(parents=True, exist_ok=True)


class BatchAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Batch API."""

    def log_message(self, format, *args):
        """Override to use custom logging format."""
        print(f"[{self.log_date_time_string()}] {args[0]}")

    def _send_json_response(self, status_code: int, data: dict):
        """Send a JSON response."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _send_file(self, file_path: Path):
        """Send a file as response."""
        self.send_response(200)
        self.send_header("Content-Type", "application/jsonl")
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()
        with open(file_path, "rb") as f:
            self.wfile.write(f.read())

    def _send_error(self, status_code: int, message: str):
        """Send an error response."""
        self._send_json_response(status_code, {"error": message})

    def _get_content_length(self) -> int:
        """Get Content-Length from headers."""
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            return 0
        return int(content_length)

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # GET /batches - List all batches
        if path == "/batches":
            self._list_batches()
            return

        # GET /batches/{batch_id} - Get batch status
        if path.startswith("/batches/") and path.count("/") == 2:
            batch_id = path.split("/")[2]
            self._get_batch_status(batch_id)
            return

        # GET /batches/{batch_id}/results - Download results
        if path.endswith("/results") and path.startswith("/batches/"):
            parts = path.split("/")
            if len(parts) == 4:
                batch_id = parts[2]
                self._download_results(batch_id)
                return

        self._send_error(404, "Not found")

    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # POST /batches - Upload batch file
        if path == "/batches":
            self._upload_batch()
            return

        # POST /batches/{batch_id}/cancel - Cancel a batch
        if path.endswith("/cancel") and path.startswith("/batches/"):
            parts = path.split("/")
            if len(parts) == 4:
                batch_id = parts[2]
                self._cancel_batch(batch_id)
                return

        self._send_error(404, "Not found")

    def do_DELETE(self):
        """Handle DELETE requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # DELETE /batches/{batch_id} - Delete a batch
        if path.startswith("/batches/") and path.count("/") == 2:
            batch_id = path.split("/")[2]
            self._delete_batch(batch_id)
            return

        self._send_error(404, "Not found")

    def _upload_batch(self):
        """Handle batch file upload."""
        content_length = self._get_content_length()
        if content_length == 0:
            self._send_error(400, "No file content provided")
            return

        # Check content type
        content_type = self.headers.get("Content-Type", "")
        if "jsonl" not in content_type and "application/octet-stream" not in content_type:
            # Accept various content types for flexibility
            pass

        # Generate batch ID
        batch_id = f"batch_{uuid.uuid4().hex}"
        batch_dir = BATCHES_DIR / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Generate random input file name
        input_file_name = f"{uuid.uuid4().hex}.jsonl"
        input_file_path = batch_dir / input_file_name
        content = self.rfile.read(content_length)

        # Validate JSONL format (each line should be valid JSON)
        try:
            lines = content.decode("utf-8").strip().split("\n")
            for line in lines:
                if line.strip():
                    json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # Clean up on validation failure
            input_file_path.unlink(missing_ok=True)
            batch_dir.rmdir()
            self._send_error(400, f"Invalid JSONL format: {str(e)}")
            return

        with open(input_file_path, "wb") as f:
            f.write(content)

        # Create batch metadata
        metadata = {
            "id": batch_id,
            "status": "validating",
            "input_file": input_file_name,
            "output_file": None,
            "error_file": None,
            "created_at": int(os.path.getctime(input_file_path)),
            "request_counts": {
                "total": len(lines),
                "completed": 0,
                "failed": 0
            }
        }

        # Save metadata
        metadata_path = batch_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update status to in_progress (simulating validation)
        metadata["status"] = "in_progress"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self._send_json_response(200, metadata)

    def _list_batches(self):
        """List all batches."""
        batches = []
        for batch_dir in BATCHES_DIR.iterdir():
            if batch_dir.is_dir():
                metadata_path = batch_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        batches.append(metadata)

        self._send_json_response(200, {"data": batches, "object": "list"})

    def _get_batch_status(self, batch_id: str):
        """Get batch status."""
        batch_dir = BATCHES_DIR / batch_id
        metadata_path = batch_dir / "metadata.json"

        if not metadata_path.exists():
            self._send_error(404, f"Batch '{batch_id}' not found")
            return

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self._send_json_response(200, metadata)

    def _download_results(self, batch_id: str):
        """Download batch results."""
        batch_dir = BATCHES_DIR / batch_id

        if not batch_dir.exists():
            self._send_error(404, f"Batch '{batch_id}' not found")
            return

        # Load metadata
        metadata_path = batch_dir / "metadata.json"
        if not metadata_path.exists():
            self._send_error(404, f"Batch '{batch_id}' metadata not found")
            return

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        output_file_name = metadata.get("output_file")
        if not output_file_name:
            self._send_error(500, "Output file not specified in metadata")
            return

        # Check if results are ready (.ready file exists)
        ready_file = batch_dir / f"{output_file_name}.ready"
        output_file = batch_dir / output_file_name

        if not ready_file.exists():
            self._send_json_response(200, {
                "status": "not_ready",
                "message": "Results are not ready yet. Batch is still processing."
            })
            return

        if not output_file.exists():
            self._send_error(500, "Ready flag exists but output file not found")
            return

        self._send_file(output_file)

    def _cancel_batch(self, batch_id: str):
        """Cancel a batch."""
        batch_dir = BATCHES_DIR / batch_id
        metadata_path = batch_dir / "metadata.json"

        if not metadata_path.exists():
            self._send_error(404, f"Batch '{batch_id}' not found")
            return

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        if metadata["status"] in ["completed", "cancelled", "expired", "failed"]:
            self._send_error(400, f"Cannot cancel batch with status '{metadata['status']}'")
            return

        metadata["status"] = "cancelled"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self._send_json_response(200, metadata)

    def _delete_batch(self, batch_id: str):
        """Delete a batch."""
        batch_dir = BATCHES_DIR / batch_id

        if not batch_dir.exists():
            self._send_error(404, f"Batch '{batch_id}' not found")
            return

        import shutil
        shutil.rmtree(batch_dir)

        self._send_json_response(200, {"deleted": True, "id": batch_id})


def mark_batch_ready(batch_id: str, results: list):
    """
    Helper function to mark a batch as complete with results.
    This would typically be called by a background worker.
    """
    batch_dir = BATCHES_DIR / batch_id
    if not batch_dir.exists():
        raise ValueError(f"Batch '{batch_id}' not found")

    # Generate random output file name
    output_file_name = f"{uuid.uuid4().hex}.jsonl"
    output_file = batch_dir / output_file_name
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Create .ready flag file
    ready_file = batch_dir / f"{output_file_name}.ready"
    ready_file.touch()

    # Update metadata
    metadata_path = batch_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    metadata["status"] = "completed"
    metadata["output_file"] = output_file_name
    metadata["completed_at"] = int(os.path.getctime(output_file))
    metadata["request_counts"]["completed"] = len(results)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def get_pending_batches():
    """
    Get list of batches that are not yet completed.
    Returns list of batch metadata for batches with status 'in_progress' or 'validating'.
    """
    pending = []
    for batch_dir in BATCHES_DIR.iterdir():
        if batch_dir.is_dir():
            metadata_path = batch_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    if metadata.get("status") in ["in_progress", "validating", "processing"]:
                        input_file_name = metadata.get("input_file", "input.jsonl")
                        pending.append({
                            "metadata": metadata,
                            "batch_dir": batch_dir,
                            "input_file_path": batch_dir / input_file_name
                        })
    return pending


def get_batch_input_file(batch_id: str) -> Path | None:
    """
    Get the input file path for a batch.
    Returns None if batch doesn't exist or input file not found.
    """
    batch_dir = BATCHES_DIR / batch_id
    if not batch_dir.exists():
        return None
    
    metadata_path = batch_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    input_file_name = metadata.get("input_file")
    if not input_file_name:
        return None

    input_file = batch_dir / input_file_name
    if input_file.exists():
        return input_file
    return None


def mark_batch_complete(batch_id: str, results: list):
    """
    Mark a batch as complete with results.
    This is the main entry point for workers to save results and mark batch as ready.
    """
    mark_batch_ready(batch_id, results)


def run_server(host: str = HOST, port: int = PORT):
    """Start the HTTP server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, BatchAPIHandler)
    print(f"Batch API Server running at http://{host}:{port}")
    print(f"Batch files stored in: {BATCHES_DIR.absolute()}")
    print("\nEndpoints:")
    print(f"  POST   /batches              - Upload batch file")
    print(f"  GET    /batches              - List all batches")
    print(f"  GET    /batches/{{id}}         - Get batch status")
    print(f"  GET    /batches/{{id}}/results - Download results")
    print(f"  POST   /batches/{{id}}/cancel  - Cancel batch")
    print(f"  DELETE /batches/{{id}}         - Delete batch")
    print("\nPress Ctrl+C to stop the server")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
