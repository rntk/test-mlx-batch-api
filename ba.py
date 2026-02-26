import mlx.core as mx
import sys
import os
import time
import json
import argparse
import re
import uuid
from pathlib import Path

from mlx_lm import load
from mlx_lm.generate import BatchGenerator, wired_limit, generation_stream
from mlx_lm.models import cache as cache_module
from mlx_lm.models.cache import KVCache, save_prompt_cache, load_prompt_cache

import batch_api

MODEL_PATH = "/Users/rnt/dev/models/openai/gpt-oss-20b-mlx-Q8"
DEFAULT_BATCHES_DIR = "./batches"


def get_cache_file_path(batch_dir: Path) -> str:
    """Return the prompt cache file path stored inside the batch directory."""
    return str(batch_dir / "input.jsonl.promptcache")


def resolve_cache_file_path(base_cache_file: str) -> str:
    """Return the actual cache file path if it already exists."""
    if os.path.exists(base_cache_file):
        return base_cache_file
    safetensors_path = base_cache_file + ".safetensors"
    if os.path.exists(safetensors_path):
        return safetensors_path
    return base_cache_file


def find_common_prefix_length(token_lists):
    """Find the longest common token prefix across all tokenized prompts."""
    if len(token_lists) < 2:
        return 0
    ref = token_lists[0]
    common_len = len(ref)
    for tokens in token_lists[1:]:
        common_len = min(common_len, len(tokens))
        for i in range(common_len):
            if tokens[i] != ref[i]:
                common_len = i
                break
    return common_len


def build_and_save_prefix_cache(model, tokenizer, full_prompts_tokens, cache_file: str):
    """
    Compute the shared-prefix KV cache from the first 2 prompts,
    prefill it using model() directly, and save to disk.
    Returns (prefix_cache, common_len).
    """
    # Use up to 2 prompts to find the common prefix
    sample = full_prompts_tokens[:2]
    common_len = find_common_prefix_length(sample)

    print(f"--- Building Shared Prefix Cache ---")
    print(f"  Common token prefix : {common_len} tokens")
    print(f"  Example total prompt: {len(full_prompts_tokens[0])} tokens")

    if common_len == 0:
        print("  Warning: no common prefix found; cache will be empty.")

    prefix_tokens = full_prompts_tokens[0][:common_len]
    prefix_cache = cache_module.make_prompt_cache(model)

    t0 = time.perf_counter()
    remaining = prefix_tokens
    with mx.stream(generation_stream):
        while len(remaining) > 0:
            n = min(4096, len(remaining))
            model(mx.array(remaining[:n])[None], cache=prefix_cache)
            mx.eval([c.state for c in prefix_cache])
            remaining = remaining[n:]
    t1 = time.perf_counter()

    n_tokens = len(prefix_tokens)
    print(f"  Prefix prefill: {n_tokens} tokens in {t1 - t0:.2f}s "
          f"({n_tokens / (t1 - t0):.0f} tok/s)")

    print(f"  Saving cache to {cache_file} ...")
    save_prompt_cache(cache_file, prefix_cache, {"model": MODEL_PATH})
    print(f"  Cache saved.")

    return prefix_cache, common_len


def load_prefix_cache(cache_file: str):
    """Load a previously saved prefix cache from disk. Returns (cache, metadata)."""
    cache, metadata = load_prompt_cache(cache_file, return_metadata=True)
    return cache, metadata


def clone_cache(prompt_cache):
    """Deep-clone a single-sequence KV cache so each prompt gets its own copy."""
    cloned = []
    for c in prompt_cache:
        new_c = KVCache()
        new_c.keys = c.keys[..., :c.offset, :] if c.keys is not None else None
        new_c.values = c.values[..., :c.offset, :] if c.values is not None else None
        new_c.offset = c.offset
        cloned.append(new_c)
    return cloned


def _mark_batch_complete(batch_dir: Path, results: list):
    """Store results in the Files store and mark the batch as completed via batch_api."""
    batch_api.mark_batch_complete(batch_dir.name, results)


def process_batch(model, tokenizer, batch_item: dict):
    """Process a single batch directory (batch_api.py format)."""
    batch_dir = batch_item['batch_dir']
    input_file = batch_item['input_file_path']

    # Guard: re-read metadata in case another process already completed it
    metadata_path = batch_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as _f:
            _meta = json.load(_f)
        if _meta.get("status") not in {"in_progress", "validating", "processing"}:
            print(f"Skipping {batch_dir.name}: status={_meta.get('status')}")
            return

    # Read all lines from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = [l for l in f.readlines() if l.strip()]

    if not all_lines:
        print(f"No lines to process in {batch_dir.name}")
        return

    print(f"Processing {len(all_lines)} lines from batch {batch_dir.name}")
    
    # Build prompts — each line is a JSON object with a "prompt" key
    prompts_text = []
    line_data = []
    for line in all_lines:
        data = json.loads(line)
        url = data.get("url", "")
        body = data.get("body", {})
        # Support Responses API format: {"url": "/v1/responses", "body": {"input": ..., "instructions": ...}}
        if url == "/v1/responses" or ("input" in body and "messages" not in body):
            raw_input = body.get("input", "")
            if isinstance(raw_input, str):
                prompt = raw_input
            elif isinstance(raw_input, list):
                # Extract text from ResponseInputItem list
                parts = []
                for item in raw_input:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        role = item.get("role", "")
                        content = item.get("content", "")
                        if isinstance(content, str):
                            parts.append(content)
                        elif isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "input_text":
                                    parts.append(c.get("text", ""))
                                elif isinstance(c, str):
                                    parts.append(c)
                prompt = " ".join(parts)
            else:
                prompt = ""
        # Support Chat Completions Batch API format: {"body": {"messages": [...]}}
        elif "body" in data and "messages" in body:
            user_content = next(
                (m["content"] for m in reversed(body["messages"]) if m["role"] == "user"),
                "",
            )
            prompt = user_content
        else:
            prompt = data.get("prompt", "")
        prompts_text.append(prompt)
        line_data.append(data)

    if not prompts_text:
        print(f"No valid prompts in batch {batch_dir.name}")
        return {'prompts': 0, 'sentences': 0, 'tokens': 0, 'time': 0.0}
    
    total_sentences = sum(len(re.findall(r'[.!?]+', p)) for p in prompts_text)
    
    # Tokenize all prompts
    full_prompts_tokens = []
    for idx, p in enumerate(prompts_text):
        orig = line_data[idx]
        url = orig.get("url", "")
        body = orig.get("body", {})
        # Responses API format: build messages from input + optional instructions
        if url == "/v1/responses" or ("input" in body and "messages" not in body):
            raw_input = body.get("input", "")
            instructions = body.get("instructions", None)
            msgs = []
            if instructions:
                msgs.append({"role": "system", "content": instructions})
            if isinstance(raw_input, list):
                # ResponseInputItem list — may already be message-shaped
                for item in raw_input:
                    if isinstance(item, dict) and "role" in item:
                        content = item.get("content", "")
                        if isinstance(content, list):
                            # Flatten content parts to a single string
                            text_parts = []
                            for c in content:
                                if isinstance(c, dict):
                                    text_parts.append(c.get("text", c.get("input_text", "")))
                                elif isinstance(c, str):
                                    text_parts.append(c)
                            msgs.append({"role": item["role"], "content": " ".join(text_parts)})
                        else:
                            msgs.append({"role": item["role"], "content": content})
                    else:
                        # Plain string item — treat as user message
                        msgs.append({"role": "user", "content": str(item)})
            else:
                msgs.append({"role": "user", "content": str(raw_input)})
        # Chat Completions format
        elif "messages" in body:
            msgs = body["messages"]
        else:
            msgs = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        full_prompts_tokens.append(tokenizer.encode(formatted))
    
    # Handle KV cache prefix optimization — cache is kept in the batch directory
    base_cache_file = get_cache_file_path(batch_dir)
    cache_file = resolve_cache_file_path(base_cache_file)
    
    if os.path.exists(cache_file):
        print(f"--- Loading Shared Prefix Cache from {cache_file} ---")
        prefix_cache, meta = load_prefix_cache(cache_file)
        common_len = prefix_cache[0].offset if prefix_cache else 0
    else:
        prefix_cache, common_len = build_and_save_prefix_cache(
            model, tokenizer, full_prompts_tokens, base_cache_file
        )
    
    if common_len > 0 and len(full_prompts_tokens) > 1:
        suffix_prompts = [toks[common_len:] for toks in full_prompts_tokens]
        mx.eval([c.state for c in prefix_cache])
        caches = [clone_cache(prefix_cache) for _ in suffix_prompts]
    else:
        suffix_prompts = full_prompts_tokens
        caches = None
    
    # Batch inference
    gen = BatchGenerator(
        model,
        stop_tokens=set(tokenizer.eos_token_ids),
        completion_batch_size=48,
        prefill_batch_size=12,
        prefill_step_size=4096,
    )
    
    total_tokens = 0
    t_start = time.perf_counter()
    results_dict = {}
    
    with wired_limit(model, [generation_stream]):
        uids = gen.insert(suffix_prompts, max_tokens=24000, caches=caches)
        results_dict = {uid: [] for uid in uids}
        
        while responses := gen.next():
            for r in responses:
                if r.finish_reason is None:
                    results_dict[r.uid].append(r.token)
                    total_tokens += 1
    
    t_elapsed = time.perf_counter() - t_start
    tps = total_tokens / t_elapsed if t_elapsed > 0 else 0.0

    print(f"  Batch {batch_dir.name}: {len(prompts_text)} prompts, {total_sentences} sentences, "
          f"{total_tokens} tokens, {t_elapsed:.2f}s, {tps:.1f} tok/s")

    # Build output records and mark batch complete
    results = []
    for idx, (uid, tokens) in enumerate(results_dict.items()):
        decoded = tokenizer.decode(tokens)
        orig = line_data[idx]
        url = orig.get("url", "")
        body = orig.get("body", {})
        custom_id = orig.get("custom_id", str(idx))
        now = int(time.time())

        if url == "/v1/responses" or ("input" in body and "messages" not in body):
            # Responses API output format
            model_id = body.get("model", MODEL_PATH)
            response_body = {
                "id": f"resp_{uuid.uuid4().hex}",
                "object": "response",
                "created_at": now,
                "model": model_id,
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex}",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": decoded, "annotations": []}
                        ],
                        "status": "completed",
                    }
                ],
                "status": "completed",
                "usage": {
                    "input_tokens": len(full_prompts_tokens[idx]),
                    "output_tokens": len(tokens),
                    "total_tokens": len(full_prompts_tokens[idx]) + len(tokens),
                },
            }
            record = {
                "id": f"batch_req_{uuid.uuid4().hex}",
                "custom_id": custom_id,
                "response": {"status_code": 200, "request_id": f"req_{uuid.uuid4().hex}", "body": response_body},
                "error": None,
            }
        elif "messages" in body or url in ("/v1/chat/completions", ""):
            # Chat Completions output format (if it was a batch API request)
            if "custom_id" in orig:
                model_id = body.get("model", MODEL_PATH)
                response_body = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": now,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": decoded},
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(full_prompts_tokens[idx]),
                        "completion_tokens": len(tokens),
                        "total_tokens": len(full_prompts_tokens[idx]) + len(tokens),
                    },
                }
                record = {
                    "id": f"batch_req_{uuid.uuid4().hex}",
                    "custom_id": custom_id,
                    "response": {"status_code": 200, "request_id": f"req_{uuid.uuid4().hex}", "body": response_body},
                    "error": None,
                }
            else:
                # Legacy format — keep backward compatible
                record = dict(orig)
                record["result"] = decoded
        else:
            # Legacy format
            record = dict(orig)
            record["result"] = decoded
        results.append(record)

    _mark_batch_complete(batch_dir, results)

    return {'prompts': len(prompts_text), 'sentences': total_sentences, 'tokens': total_tokens, 'time': t_elapsed}


def get_pending_batches(batches_dir: Path) -> list:
    """Return batch subdirectories that are pending based on metadata status.

    Resolves the input file through the batch_api Files store (input_file_id),
    with a fallback to a legacy input.jsonl inside the batch directory.
    """
    pending = []
    if not batches_dir.is_dir():
        return pending
    for batch_dir in sorted(batches_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        metadata_path = batch_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        with open(metadata_path) as f:
            metadata = json.load(f)
        if metadata.get("status") not in {"in_progress", "validating", "processing"}:
            continue

        # Resolve input file: new API uses input_file_id → Files store
        input_file_id = metadata.get("input_file_id")
        if input_file_id:
            input_file_path = batch_api._file_content_path(input_file_id)
            if not input_file_path.exists():
                print(f"Warning: input file {input_file_id} not found for batch "
                      f"{batch_dir.name} — skipping")
                continue
        else:
            # Legacy fallback: input.jsonl lives directly in the batch directory
            input_file_name = metadata.get("input_file", "input.jsonl")
            input_file_path = batch_dir / input_file_name
            if not input_file_path.exists():
                print(f"Warning: input file {input_file_path} not found for batch "
                      f"{batch_dir.name} — skipping")
                continue

        pending.append({
            "metadata": metadata,
            "batch_dir": batch_dir,
            "input_file_path": input_file_path,
        })
    return pending


def main():
    parser = argparse.ArgumentParser(
        description="Process pending batch_api batches from the batches directory"
    )
    parser.add_argument(
        "batches_dir",
        nargs="?",
        default=DEFAULT_BATCHES_DIR,
        help=f"Path to the batches directory (default: {DEFAULT_BATCHES_DIR})",
    )
    args = parser.parse_args()

    batches_dir = Path(args.batches_dir)
    if not batches_dir.is_dir():
        print(f"Error: {batches_dir} is not a directory")
        sys.exit(1)

    # Load model once
    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    print("Model loaded.\n")

    # Initialize metrics
    total_batches = 0
    total_prompts = 0
    total_sentences = 0
    total_tokens = 0
    total_time = 0.0

    try:
        while True:
            pending = get_pending_batches(batches_dir)
            if pending:
                print(f"Found {len(pending)} pending batch(es) in {batches_dir}\n")
                for item in pending:
                    batch_dir = item['batch_dir']
                    print(f"--- Processing batch {batch_dir.name} ---")
                    metrics = process_batch(model, tokenizer, item)
                    total_batches += 1
                    total_prompts += metrics['prompts']
                    total_sentences += metrics['sentences']
                    total_tokens += metrics['tokens']
                    total_time += metrics['time']
            else:
                print(f"No pending batches in {batches_dir}. Checking again in 10 seconds...")
                time.sleep(10)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")

    # Print overall metrics
    avg_tps = total_tokens / total_time if total_time > 0 else 0.0
    print(f"\n--- Overall Metrics ---")
    print(f"Total batches processed: {total_batches}")
    print(f"Total prompts processed: {total_prompts}")
    print(f"Total sentences processed: {total_sentences}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total inference time: {total_time:.2f}s")
    print(f"Average tokens per second: {avg_tps:.1f} tok/s")


if __name__ == "__main__":
    main()