import mlx.core as mx
import sys
import os
import time
import json
import argparse
import re
from pathlib import Path

from mlx_lm import load
from mlx_lm.generate import BatchGenerator, wired_limit, generation_stream
from mlx_lm.models import cache as cache_module
from mlx_lm.models.cache import KVCache, save_prompt_cache, load_prompt_cache

MODEL_PATH = "/Users/rnt/dev/models/openai/gpt-oss-20b-mlx-Q8"
DEFAULT_BATCHES_DIR = "./batches"


def get_cache_file_path(input_file: str) -> str:
    """Return the prompt cache file path for a given input file."""
    return str(Path(input_file).parent / "input.jsonl.promptcache")


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


def mark_batch_complete(batch_dir: Path, results: list):
    """Write output.jsonl, create the .ready flag, and update metadata.json."""
    output_file = batch_dir / "output.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    (batch_dir / "output.jsonl.ready").touch()

    metadata_path = batch_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata["status"] = "completed"
        metadata["output_file"] = "output.jsonl"
        metadata["completed_at"] = int(os.path.getctime(output_file))
        metadata["request_counts"]["completed"] = len(results)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def process_batch(model, tokenizer, batch_item: dict):
    """Process a single batch directory (batch_api.py format)."""
    batch_dir = batch_item['batch_dir']
    input_file = batch_item['input_file_path']
    ready_file = batch_dir / "output.jsonl.ready"

    if ready_file.exists():
        print(f"Skipping {batch_dir.name}: already completed")
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
        prompt = data["prompt"]
        prompts_text.append(prompt)
        line_data.append(data)

    if not prompts_text:
        print(f"No valid prompts in batch {batch_dir.name}")
        return {'prompts': 0, 'sentences': 0, 'tokens': 0, 'time': 0.0}
    
    total_sentences = sum(len(re.findall(r'[.!?]+', p)) for p in prompts_text)
    
    # Tokenize all prompts
    full_prompts_tokens = []
    for p in prompts_text:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        full_prompts_tokens.append(tokenizer.encode(formatted))
    
    # Handle KV cache prefix optimization
    base_cache_file = get_cache_file_path(str(input_file))
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
        record = dict(line_data[idx])  # copy original fields
        record["result"] = decoded
        results.append(record)

    mark_batch_complete(batch_dir, results)

    return {'prompts': len(prompts_text), 'sentences': total_sentences, 'tokens': total_tokens, 'time': t_elapsed}


def get_pending_batches(batches_dir: Path) -> list:
    """Return batch subdirectories that are pending based on metadata status."""
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
        if metadata.get("status") in ["in_progress", "validating", "processing"]:
            input_file_name = metadata.get("input_file", "input.jsonl")
            pending.append({
                "metadata": metadata,
                "batch_dir": batch_dir,
                "input_file_path": batch_dir / input_file_name
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

    pending = get_pending_batches(batches_dir)
    if not pending:
        print(f"No pending batches in {batches_dir}")
        return

    print(f"Found {len(pending)} pending batch(es) in {batches_dir}\n")

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

    for item in pending:
        batch_dir = item['batch_dir']
        print(f"--- Processing batch {batch_dir.name} ---")
        metrics = process_batch(model, tokenizer, item)
        total_batches += 1
        total_prompts += metrics['prompts']
        total_sentences += metrics['sentences']
        total_tokens += metrics['tokens']
        total_time += metrics['time']

    # Print overall metrics
    avg_tps = total_tokens / total_time if total_time > 0 else 0.0
    print(f"\n--- Overall Metrics ---")
    print(f"Total batches processed: {total_batches}")
    print(f"Total prompts processed: {total_prompts}")
    print(f"Total sentences processed: {total_sentences}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total inference time: {total_time:.2f}s")
    print(f"Average tokens per second: {avg_tps:.1f} tok/s")

    print("\nAll pending batches processed.")


if __name__ == "__main__":
    main()