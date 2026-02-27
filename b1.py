from mlx_lm import load
from mlx_lm.generate import BatchGenerator, wired_limit, generation_stream
from mlx_lm.models import cache as cache_module
from mlx_lm.models.cache import KVCache, save_prompt_cache, load_prompt_cache
import mlx.core as mx
import sys
import os
import time
import json

MODEL_PATH = "/Users/rnt/dev/models/openai/gpt-oss-20b-mlx-Q8"


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


def extract_prompt_from_line(data: dict) -> str:
    """Extract prompt text from a JSONL line supporting multiple formats."""
    # Support Responses API format: {"body": {"input": [...]}}
    body = data.get("body", {})
    input_data = body.get("input", None)
    
    if input_data is not None:
        # input is a list of message objects
        if isinstance(input_data, list):
            parts = []
            for item in input_data:
                if isinstance(item, dict):
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
            return " ".join(parts)
        # input is a string
        elif isinstance(input_data, str):
            return input_data
    
    # Fallback to "prompt" field
    return data.get("prompt", "")


def main():
    if len(sys.argv) < 2:
        print("Usage: python b1.py <input_file> [num_lines]")
        print("Example: python b1.py files/file-xxx/input.jsonl 8")
        sys.exit(1)

    input_file = sys.argv[1]
    num_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)

    # Read JSONL file - extract prompts from first N lines
    prompts_text = []
    line_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Extract prompt from JSONL - supports multiple formats
            prompt = extract_prompt_from_line(data)
            if prompt:
                prompts_text.append(prompt)
                line_data.append(data)

    if not prompts_text:
        print("No prompts found in input file")
        sys.exit(1)

    print(f"Loaded {len(prompts_text)} prompts from {input_file}")

    # Save prompts to file
    prompts_file = "prompts.txt"
    with open(prompts_file, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(prompts_text, 1):
            f.write(f"--- Prompt {i} ---\n{prompt}\n\n")

    print(f"Prompts saved to {prompts_file}\n")

    model, tokenizer = load(MODEL_PATH)

    # Tokenize all prompts fully (with chat template)
    full_prompts_tokens = []
    for p in prompts_text:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        full_prompts_tokens.append(tokenizer.encode(formatted))

    # ── KV cache prefix optimization ──────────────────────────────────────
    # All prompts share an identical instruction prefix.
    # The cache is generated once per input file and saved to disk.
    # On subsequent runs the cache is loaded directly, skipping prefill.
    base_cache_file = get_cache_file_path(input_file)
    cache_file = resolve_cache_file_path(base_cache_file)

    if os.path.exists(cache_file):
        print(f"--- Loading Shared Prefix Cache from {cache_file} ---")
        prefix_cache, meta = load_prefix_cache(cache_file)
        # Determine common_len from the loaded cache offset (first layer)
        common_len = prefix_cache[0].offset if prefix_cache else 0
        print(f"  Loaded cache offset : {common_len} tokens")
    else:
        prefix_cache, common_len = build_and_save_prefix_cache(
            model, tokenizer, full_prompts_tokens, base_cache_file
        )

    print(f"\n--- Shared Prefix KV Cache ---")
    print(f"  Common token prefix : {common_len} tokens")
    print(f"  Example total prompt: {len(full_prompts_tokens[0])} tokens")
    print(f"  Tokens saved/chunk  : {common_len}")

    if common_len > 0 and len(full_prompts_tokens) > 1:
        # Build per-chunk suffix tokens, with shared prefix cache for all
        suffix_prompts = [toks[common_len:] for toks in full_prompts_tokens]
        # Evaluate lazy cache updates once before cloning (not per-clone)
        mx.eval([c.state for c in prefix_cache])
        # Clone the prefix cache for each prompt to avoid state corruption during merge
        caches = [clone_cache(prefix_cache) for _ in suffix_prompts]
        print(f"  Suffix tokens/chunk : {[len(s) for s in suffix_prompts]}")
    else:
        # Only one prompt or no common prefix – fall back to normal path
        suffix_prompts = full_prompts_tokens
        caches = None
        print("  (no prefix sharing – using standard prefill)")

    # ── Batch inference ───────────────────────────────────────────────────
    # Configure for M3 Ultra
    gen = BatchGenerator(
        model,
        stop_tokens=set(tokenizer.eos_token_ids),
        completion_batch_size=48,     # High for M3 Ultra's 80 GPU cores
        prefill_batch_size=12,        # Parallel prefill
        prefill_step_size=4096,       # Large chunks for 800GB/s bandwidth
    )

    print(f"\nRunning batch inference...")
    total_tokens = 0
    t_start = time.perf_counter()
    with wired_limit(model, [generation_stream]):
        uids = gen.insert(suffix_prompts, max_tokens=24000, caches=caches)
        results = {uid: [] for uid in uids}
        prompt_tokens = {
            uid: len(full_prompts_tokens[i]) for i, uid in enumerate(uids)
        }

        while responses := gen.next():
            for r in responses:
                if r.finish_reason is None:
                    results[r.uid].append(r.token)
                    total_tokens += 1
    t_elapsed = time.perf_counter() - t_start

    # Print generation metrics
    tps = total_tokens / t_elapsed if t_elapsed > 0 else 0.0
    print(f"\n--- Generation Metrics ---")
    print(f"  Elapsed time      : {t_elapsed:.2f}s")
    print(f"  Total tokens      : {total_tokens}")
    print(f"  Throughput        : {tps:.1f} tok/s")
    print(f"  Prompts           : {len(uids)}")
    print(f"  Prefix tokens saved: {common_len} × {len(uids)} = "
          f"{common_len * len(uids)} tokens (prefilled once)")
    for uid, tokens in results.items():
        gen_toks = len(tokens)
        prompt_toks = prompt_tokens.get(uid, 0)
        print(f"  UID {uid:3d}: prompt={prompt_toks} gen={gen_toks} "
              f"total={prompt_toks + gen_toks}")

    # Save results to file
    output_file = "results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for uid, tokens in results.items():
            decoded = tokenizer.decode(tokens)
            f.write(f"--- UID {uid} ---\n{decoded}\n\n")

    print(f"\nResults saved to {output_file}")

    # Also print to console
    for uid, tokens in results.items():
        print(f"UID {uid}: {tokenizer.decode(tokens)[:120]}...")


if __name__ == "__main__":
    main()
