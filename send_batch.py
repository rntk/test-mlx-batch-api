#!/usr/bin/env python3
"""
Script to send a batch request to the batch API with four text summaries.
"""

import json
import urllib.request
import urllib.error
import sys
import os
import time

# Path to the file
FILE_PATH = "/Users/rnt/dev/python/mlx/all_posts_raw_1771928104.txt"

# Batch API URL
BATCH_API_URL = "http://localhost:8788/batches"

def read_file_and_get_four_texts(file_path):
    """Read the file and extract four text snippets."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Take the first four lines, strip them
    return [line.strip() for line in lines[:4] if line.strip()]

def create_jsonl(texts):
    """Create JSONL content with prompts."""
    jsonl_lines = []
    for text in texts:
        prompt = f"Summarize the following text: {text}"
        json_obj = {"prompt": prompt}
        jsonl_lines.append(json.dumps(json_obj))
    return '\n'.join(jsonl_lines)

def send_batch_request(jsonl_content):
    """Send the batch request to the API."""
    data = jsonl_content.encode('utf-8')
    req = urllib.request.Request(BATCH_API_URL, data=data, headers={'Content-Type': 'application/jsonl'})
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.read().decode('utf-8')}")
        return None
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        return None

def get_batch_status(batch_id):
    """Get batch status."""
    url = f"{BATCH_API_URL}/{batch_id}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result
    except urllib.error.HTTPError as e:
        print(f"HTTP Error getting status: {e.code} - {e.read().decode('utf-8')}")
        return None
    except urllib.error.URLError as e:
        print(f"URL Error getting status: {e.reason}")
        return None

def download_results(batch_id):
    """Download batch results."""
    url = f"{BATCH_API_URL}/{batch_id}/results"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8')
            return content
    except urllib.error.HTTPError as e:
        if e.code == 200:
            return e.read().decode('utf-8')
        else:
            print(f"HTTP Error downloading results: {e.code} - {e.read().decode('utf-8')}")
            return None
    except urllib.error.URLError as e:
        print(f"URL Error downloading results: {e.reason}")
        return None

def main():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        sys.exit(1)
    
    texts = read_file_and_get_four_texts(FILE_PATH)
    if len(texts) < 4:
        print("Not enough texts found in the file.")
        sys.exit(1)
    
    jsonl_content = create_jsonl(texts)
    print("Sending batch request...")
    response = send_batch_request(jsonl_content)
    if response is None:
        sys.exit(1)
    
    print("Batch uploaded successfully!")
    print(f"Batch ID: {response['id']}")
    print(f"Status: {response['status']}")
    
    # Save response to file
    response_file = f"batch_response_{response['id']}.json"
    with open(response_file, 'w') as f:
        json.dump(response, f, indent=2)
    print(f"API response saved to: {response_file}")
    
    # Poll for results
    batch_id = response['id']
    while True:
        print("Checking batch status...")
        status_response = get_batch_status(batch_id)
        if status_response is None:
            print("Failed to get status.")
            break
        status = status_response.get('status')
        print(f"Status: {status}")
        if status == 'completed':
            print("Batch completed. Downloading results...")
            results = download_results(batch_id)
            if results is not None:
                results_file = f"results_{batch_id}.jsonl"
                with open(results_file, 'w') as f:
                    f.write(results)
                print(f"Results saved to: {results_file}")
            break
        elif status in ['failed', 'cancelled', 'expired']:
            print(f"Batch {status}. Stopping.")
            break
        else:
            print("Batch still processing. Waiting 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()