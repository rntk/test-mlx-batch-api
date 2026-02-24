from mlx_lm import load
from mlx_lm.generate import BatchGenerator, wired_limit, generation_stream, generate_step
from mlx_lm.models import cache as cache_module
from mlx_lm.models.cache import KVCache, RotatingKVCache, save_prompt_cache, load_prompt_cache
import mlx.core as mx
import sys
import os
import time
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
    prefill it using generate_step(max_tokens=0), and save to disk.
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

    prefix_tokens = mx.array(full_prompts_tokens[0][:common_len])
    prefix_cache = cache_module.make_prompt_cache(model)

    t0 = time.perf_counter()
    for _ in generate_step(
        prefix_tokens,
        model,
        max_tokens=0,
        prompt_cache=prefix_cache,
        prefill_step_size=4096,
    ):
        pass
    t1 = time.perf_counter()

    print(f"  Prefill: {common_len} tokens in {t1 - t0:.2f}s "
          f"({common_len / (t1 - t0):.0f} tok/s)" if (t1 - t0) > 0 else
          f"  Prefill: {common_len} tokens")

    print(f"  Saving cache to {cache_file} ...")
    save_prompt_cache(cache_file, prefix_cache, {"model": MODEL_PATH})
    print(f"  Cache saved.")

    return prefix_cache, common_len


def load_prefix_cache(cache_file: str):
    """Load a previously saved prefix cache from disk. Returns (cache, metadata)."""
    cache, metadata = load_prompt_cache(cache_file, return_metadata=True)
    return cache, metadata


def clone_cache(prompt_cache):
    """
    Deep-clone a per-prompt KV cache list preserving the original cache types.

    KVCache layers are cloned by slicing to the filled portion (offset).
    RotatingKVCache layers (sliding-window attention) are put into temporal
    (chronological) order first and cloned so that _idx == keys.shape[2].
    BatchRotatingKVCache.merge requires this invariant to copy data correctly.

    NOTE: The caller must call mx.eval([c.state for c in prompt_cache]) once
    before the cloning loop to force evaluation of lazy cache updates.
    """
    cloned = []
    for c in prompt_cache:
        if isinstance(c, RotatingKVCache):
            # The circular buffer may be rotated; put it in chronological order
            # so that _idx == keys.shape[2] after the clone.
            if c.keys is not None:
                temporal_keys = mx.contiguous(c._temporal_order(c.keys))
                temporal_values = mx.contiguous(c._temporal_order(c.values))
                new_c = RotatingKVCache(max_size=c.max_size, keep=c.keep)
                new_c.keys = temporal_keys
                new_c.values = temporal_values
                new_c.offset = c.offset  # absolute position (used for RoPE)
                new_c._idx = temporal_keys.shape[2]  # = size() tokens, in order
            else:
                new_c = RotatingKVCache(max_size=c.max_size, keep=c.keep)
        else:
            # KVCache: slice to exactly offset tokens (zero-copy view).
            # BatchKVCache.merge() will copy into its own batch buffer,
            # so contiguous copies here are redundant.
            new_c = KVCache()
            if c.keys is not None:
                new_c.keys = c.keys[..., :c.offset, :]
                new_c.values = c.values[..., :c.offset, :]
            new_c.offset = c.offset
        cloned.append(new_c)
    return cloned


def main():
    if len(sys.argv) < 2:
        print("Usage: python b.py <input_file> [num_lines]")
        print("Example: python b.py data.txt 10")
        sys.exit(1)

    input_file = sys.argv[1]
    num_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)

    # Read text from file
    text_content = ""
    line_count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            text_content += line
            line_count += 1

    if not text_content.strip():
        print("No content to process")
        sys.exit(1)

    print(f"Loaded {line_count} lines from {input_file}")

    # Initialize BatchPipeline with components
    pipeline = BatchPipeline(
        splitter=RegexSentenceSplitter(),
        marker=BracketMarker(),
        parser=DummyResponseParser(),
        gap_handler=StrictGapHandler(),
        chunker=SizeBasedChunker(max_chars=_CONTENT_MAX_CHARS),
    )
    print(f"  Instruction overhead: {_INSTRUCTION_OVERHEAD} chars")
    print(f"  Prompt budget: {PROMPT_MAX_CHARS} chars (content budget: {_CONTENT_MAX_CHARS} chars)")

    # Prepare document - splits text, marks sentences, chunks
    print("Preparing document with BatchPipeline...")
    prepared = pipeline.prepare(text_content)

    print(f"  Sentences: {len(prepared.sentences)}")
    print(f"  Chunks: {len(prepared.chunks)}")

    # Build full prompts: instructions + tagged content
    prompts_text = []
    for chunk in prepared.chunks:
        prompts_text.append(_build_topic_ranges_prompt(chunk.tagged_text))

    if not prompts_text:
        print("No prompts generated from pipeline")
        sys.exit(1)

    print(f"Generated {len(prompts_text)} prompts from pipeline chunks\n")

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