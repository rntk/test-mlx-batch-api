from mlx_lm import load
from mlx_lm.generate import BatchGenerator, wired_limit, generation_stream
from mlx_lm.models import cache as cache_module
from mlx_lm.models.cache import KVCache
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


class DummyResponseParser:
    """Dummy parser for benchmarking - just returns empty groups."""
    def parse(self, response: str, sentence_count: int):
        return []


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


def prefill_shared_prefix(model, prefix_tokens, step_size=4096):
    """Run model prefill on the shared prefix tokens and return the KV cache."""
    prompt_cache = cache_module.make_prompt_cache(model)
    remaining = prefix_tokens

    t0 = time.perf_counter()
    with mx.stream(generation_stream):
        while len(remaining) > 0:
            n = min(step_size, len(remaining))
            model(remaining[:n][None], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            remaining = remaining[n:]
    t1 = time.perf_counter()

    n_tokens = len(prefix_tokens)
    print(f"  Prefix prefill: {n_tokens} tokens in {t1 - t0:.2f}s "
          f"({n_tokens / (t1 - t0):.0f} tok/s)")
    return prompt_cache


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

    model, tokenizer = load("/Users/rnt/dev/models/openai/gpt-oss-20b-mlx-Q8")

    # Tokenize all prompts fully (with chat template)
    full_prompts_tokens = []
    for p in prompts_text:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        full_prompts_tokens.append(tokenizer.encode(formatted))

    # ── KV cache prefix optimization ──────────────────────────────────────
    # All prompts share an identical instruction prefix (~5 k chars).
    # Find the longest common token prefix, prefill it once, then pass
    # only the per-chunk suffix tokens + cloned caches to BatchGenerator.
    common_len = find_common_prefix_length(full_prompts_tokens)

    print(f"--- Shared Prefix KV Cache ---")
    print(f"  Common token prefix : {common_len} tokens")
    print(f"  Example total prompt: {len(full_prompts_tokens[0])} tokens")
    print(f"  Tokens saved/chunk  : {common_len}")

    if common_len > 0:
        common_prefix_text = tokenizer.decode(full_prompts_tokens[0][:common_len])
        print(f"\n  Common prefix text:")
        print(f"  {common_prefix_text}...")
        print()

    if common_len > 0 and len(full_prompts_tokens) > 1:
        # 1. Prefill the shared prefix once
        shared_prefix = mx.array(full_prompts_tokens[0][:common_len])
        prefix_cache = prefill_shared_prefix(model, shared_prefix, step_size=4096)

        # 2. Build per-chunk suffix tokens and cloned caches
        suffix_prompts = [toks[common_len:] for toks in full_prompts_tokens]
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