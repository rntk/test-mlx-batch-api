#!/usr/bin/env python3
"""
Test script for Batch API.
Demonstrates upload, status check, and download functionality.
"""

import http.client
import json
import time
from pathlib import Path

HOST = "localhost"
PORT = 8788


def create_sample_batch():
    """Create a sample batch input file."""
    requests = [
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello world!"}
                ],
                "max_tokens": 100
            }
        },
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "How are you?"}
                ],
                "max_tokens": 100
            }
        }
    ]
    return "\n".join(json.dumps(r) for r in requests)


def test_upload():
    """Test uploading a batch file."""
    print("\n=== Test: Upload Batch ===")
    conn = http.client.HTTPConnection(HOST, PORT)
    
    batch_content = create_sample_batch()
    headers = {"Content-Type": "application/jsonl"}
    
    conn.request("POST", "/batches", body=batch_content, headers=headers)
    response = conn.getresponse()
    
    print(f"Status: {response.status}")
    data = json.loads(response.read().decode())
    print(f"Batch ID: {data['id']}")
    print(f"Status: {data['status']}")
    print(f"Total requests: {data['request_counts']['total']}")
    
    conn.close()
    return data["id"]


def test_list_batches():
    """Test listing all batches."""
    print("\n=== Test: List Batches ===")
    conn = http.client.HTTPConnection(HOST, PORT)
    
    conn.request("GET", "/batches")
    response = conn.getresponse()
    
    print(f"Status: {response.status}")
    data = json.loads(response.read().decode())
    print(f"Total batches: {len(data['data'])}")
    
    conn.close()
    return data["data"]


def test_get_status(batch_id):
    """Test getting batch status."""
    print(f"\n=== Test: Get Status for {batch_id} ===")
    conn = http.client.HTTPConnection(HOST, PORT)
    
    conn.request("GET", f"/batches/{batch_id}")
    response = conn.getresponse()
    
    print(f"Status: {response.status}")
    data = json.loads(response.read().decode())
    print(f"Batch status: {data['status']}")
    
    conn.close()
    return data


def test_download_results_not_ready(batch_id):
    """Test downloading results when not ready."""
    print(f"\n=== Test: Download Results (Not Ready) ===")
    conn = http.client.HTTPConnection(HOST, PORT)
    
    conn.request("GET", f"/batches/{batch_id}/results")
    response = conn.getresponse()
    
    print(f"Status: {response.status}")
    data = json.loads(response.read().decode())
    print(f"Response: {data}")
    
    conn.close()
    return data


def test_download_results_ready(batch_id):
    """Test downloading results when ready."""
    print(f"\n=== Test: Download Results (Ready) ===")
    conn = http.client.HTTPConnection(HOST, PORT)
    
    conn.request("GET", f"/batches/{batch_id}/results")
    response = conn.getresponse()
    
    print(f"Status: {response.status}")
    
    if response.status == 200:
        content = response.read().decode()
        # Check if it's JSONL (results) or JSON (not ready message)
        lines = content.strip().split("\n")
        if lines and lines[0].startswith("{"):
            try:
                first_line = json.loads(lines[0])
                if "status" in first_line and first_line.get("status") == "not_ready":
                    print(f"Response: {first_line}")
                else:
                    print(f"Results ({len(lines)} lines):")
                    for line in lines[:3]:  # Show first 3 lines
                        print(f"  {line[:100]}...")
            except json.JSONDecodeError:
                print(f"Content: {content[:200]}...")
    else:
        data = json.loads(response.read().decode())
        print(f"Error: {data}")
    
    conn.close()


def test_cancel_batch(batch_id):
    """Test cancelling a batch."""
    print(f"\n=== Test: Cancel Batch ===")
    conn = http.client.HTTPConnection(HOST, PORT)
    
    conn.request("POST", f"/batches/{batch_id}/cancel")
    response = conn.getresponse()
    
    print(f"Status: {response.status}")
    data = json.loads(response.read().decode())
    print(f"New status: {data.get('status', data)}")
    
    conn.close()


def test_delete_batch(batch_id):
    """Test deleting a batch."""
    print(f"\n=== Test: Delete Batch ===")
    conn = http.client.HTTPConnection(HOST, PORT)
    
    conn.request("DELETE", f"/batches/{batch_id}")
    response = conn.getresponse()
    
    print(f"Status: {response.status}")
    data = json.loads(response.read().decode())
    print(f"Response: {data}")
    
    conn.close()


def mark_batch_as_ready_manually(batch_id):
    """Helper to mark a batch as ready (simulates worker completion)."""
    from batch_api import mark_batch_ready
    
    results = [
        {
            "id": "batch_req_1",
            "custom_id": "request-1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": "Hello! How can I help you?"}}]
                }
            },
            "error": None
        },
        {
            "id": "batch_req_2",
            "custom_id": "request-2",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": "I'm doing well, thank you!"}}]
                }
            },
            "error": None
        }
    ]
    
    mark_batch_ready(batch_id, results)
    print(f"Marked batch {batch_id} as ready")


def main():
    """Run all tests."""
    print("Batch API Test Suite")
    print("=" * 50)
    
    try:
        # Test 1: Upload a batch
        batch_id = test_upload()
        
        # Test 2: List batches
        test_list_batches()
        
        # Test 3: Get batch status
        test_get_status(batch_id)
        
        # Test 4: Try to download results (should be not ready)
        test_download_results_not_ready(batch_id)
        
        # Simulate batch completion
        print("\n>>> Simulating batch completion...")
        mark_batch_as_ready_manually(batch_id)
        
        # Test 5: Get status after completion
        test_get_status(batch_id)
        
        # Test 6: Download results (should be ready now)
        test_download_results_ready(batch_id)
        
        # Test 7: Delete batch
        test_delete_batch(batch_id)
        
        # Verify deletion
        print("\n=== Verify Deletion ===")
        batches = test_list_batches()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        
    except ConnectionRefusedError:
        print(f"\nError: Could not connect to server at {HOST}:{PORT}")
        print("Make sure the server is running: python batch_api.py")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
