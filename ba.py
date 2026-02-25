from mlx_lm import load
from mlx_lm.generate import BatchGenerator, wired_limit, generation_stream
from mlx_lm.models import cache as cache_module
from mlx_lm.models.cache import KVCache, save_prompt_cache, load_prompt_cache
import mlx.core as mx
import sys
import os
import time
import json
import argparse
from pathlib import Path
sys.path.insert(0, '/Users/rnt/dev/python/mlx/txt-splitt/src')

from txt_splitt import BatchPipeline, RegexSentenceSplitter, BracketMarker, StrictGapHandler, SizeBasedChunker
from txt_splitt.types import SentenceGroup, SentenceRange
from txt_splitt.llm import _build_topic_ranges_prompt

# Total character budget for the full prompt sent to the model.
# The chunker receives content_max_chars = PROMPT_MAX_CHARS - instruction_overhead
# so that tagged_text + instructions always fits within PROMPT_MAX_CHARS.
PROMPT_MAX_CHARS = 17_000
_INSTRUCTION_OVERHEAD = len(_build_topic_ranges_prompt(""))  # ~5164 chars
_CONTENT_MAX_CHARS = PROMPT_MAX_CHARS - _INSTRUCTION_OVERHEAD

MODEL_PATH = "/Users/rnt/dev/models/openai/gpt-oss-20b-mlx-Q8"

# Batch API configuration
BATCHES_DIR = Path("./batches")


class DummyResponseParser:
    """Dummy parser for benchmarking - just returns empty groups."""
    def parse(self, response: str, sentence_count: int):
        return []


def get_cache_file_path(input_file: str) -> str:
    """Return the prompt cache file path for a given input file."""
    return input_file + ".promptcache"


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


def mark_batch_ready(batch_id: str, results: list, metadata: dict):
    """
    Mark a batch as complete with results.
    Writes output.jsonl and creates .ready flag file.
    """
    batch_dir = BATCHES_DIR / batch_id
    if not batch_dir.exists():
        batch_dir.mkdir(parents=True, exist_ok=True)

    # Write output file
    output_file = batch_dir / "output.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Create .ready flag file
    ready_file = batch_dir / "output.jsonl.ready"
    ready_file.touch()

    # Update metadata
    metadata_path = batch_dir / "metadata.json"
    metadata["status"] = "completed"
    metadata["output_file"] = "output.jsonl"
    metadata["completed_at"] = int(os.path.getctime(output_file))
    metadata["request_counts"]["completed"] = len(results)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Batch {batch_id} marked as ready with {len(results)} results")


def process_batch(model, tokenizer, text_content: str, batch_id: str, chunk_index: int):
    """Process a single batch of text and return results."""
    # Initialize BatchPipeline with components
    pipeline = BatchPipeline(
        splitter=RegexSentenceSplitter(),
        marker=BracketMarker(),
        parser=DummyResponseParser(),
        gap_handler=StrictGapHandler(),
        chunker=SizeBasedChunker(max_chars=_CONTENT_MAX_CHARS),
    )

    # Prepare document - splits text, marks sentences, chunks
    prepared = pipeline.prepare(text_content)

    # Build full prompts: instructions + tagged content
    prompts_text = []
    for chunk in prepared.chunks:
        prompts_text.append(_build_topic_ranges_prompt(chunk.tagged_text))

    if not prompts_text:
        return []

    # Tokenize all prompts fully (with chat template)
    full_prompts_tokens = []
    for p in prompts_text:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        full_prompts_tokens.append(tokenizer.encode(formatted))

    # ── KV cache prefix optimization ──────────────────────────────────────
    base_cache_file = f"batch_{batch_id}.promptcache"
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

    # ── Batch inference ───────────────────────────────────────────────────
    gen = BatchGenerator(
        model,
        stop_tokens=set(tokenizer.eos_token_ids),
        completion_batch_size=48,
        prefill_batch_size=12,
        prefill_step_size=4096,
    )

    total_tokens = 0
    t_start = time.perf_counter()
    results = []
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

    print(f"  Batch {batch_id} [{chunk_index}]: {total_tokens} tokens, "
          f"{t_elapsed:.2f}s, {tps:.1f} tok/s")

    # Convert results to JSONL format
    for uid, tokens in results_dict.items():
        decoded = tokenizer.decode(tokens)
        results.append({
            "uid": uid,
            "chunk_index": chunk_index,
            "response": decoded,
            "tokens": tokens
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch process text files")
    parser.add_argument("input_file", help="Input text file to process")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of lines per batch (default: 100)")
    parser.add_argument("--batch-api", action="store_true",
                        help="Use batch API mode - save results to batches directory")
    parser.add_argument("--batch-id", type=str, default=None,
                        help="Batch ID for batch API mode")
    args = parser.parse_args()

    input_file = args.input_file

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)

    # Read ALL lines from file
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    if not all_lines:
        print("No content to process")
        sys.exit(1)

    print(f"Loaded {len(all_lines)} lines from {input_file}")

    # Load model once for all batches
    print("\nLoading model...")
    model, tokenizer = load(MODEL_PATH)
    print("Model loaded.\n")

    batch_size = args.batch_size
    all_results = []

    # Process file in batches
    total_batches = (len(all_lines) + batch_size - 1) // batch_size
    print(f"Processing {len(all_lines)} lines in {total_batches} batches (batch size: {batch_size})\n")

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_lines))
        batch_lines = all_lines[start_idx:end_idx]
        text_content = "".join(batch_lines)

        batch_id = args.batch_id if args.batch_id else f"file_{os.path.basename(input_file)}_{batch_idx}"
        chunk_index = batch_idx

        print(f"\n--- Processing Batch {batch_idx + 1}/{total_batches} (lines {start_idx}-{end_idx}) ---")

        if args.batch_api:
            # Batch API mode: create batch directory and save results there
            batch_dir = BATCHES_DIR / batch_id
            batch_dir.mkdir(parents=True, exist_ok=True)

            # Create initial metadata
            metadata = {
                "id": batch_id,
                "status": "processing",
                "input_file": input_file,
                "chunk_index": chunk_index,
                "lines_range": [start_idx, end_idx],
                "created_at": int(time.time()),
                "request_counts": {
                    "total": len(batch_lines),
                    "completed": 0,
                    "failed": 0
                }
            }

            # Save initial metadata
            metadata_path = batch_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # Process the batch
            results = process_batch(model, tokenizer, text_content, batch_id, chunk_index)

            # Mark batch as ready
            mark_batch_ready(batch_id, results, metadata)
            all_results.extend(results)
        else:
            # Standard mode: process and collect results
            results = process_batch(model, tokenizer, text_content, batch_id, chunk_index)
            all_results.extend(results)

    # In non-batch-API mode, save all results to a single file
    if not args.batch_api:
        output_file = "results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(f"--- UID {result['uid']} (Chunk {result['chunk_index']}) ---\n")
                f.write(f"{result['response']}\n\n")

        print(f"\n--- All Results ---")
        print(f"Total batches processed: {total_batches}")
        print(f"Total results: {len(all_results)}")
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()