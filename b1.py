from mlx_lm import load
from mlx_lm.generate import BatchGenerator, wired_limit, generation_stream
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
    
    # Tokenize multiple prompts
    prompts = []
    for p in prompts_text:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(tokenizer.encode(formatted))
    
    # Configure for M3 Ultra
    gen = BatchGenerator(
        model,
        stop_tokens=set(tokenizer.eos_token_ids),
        completion_batch_size=64,     # High for M3 Ultra's 80 GPU cores
        prefill_batch_size=16,        # Parallel prefill
        prefill_step_size=4096,       # Large chunks for 800GB/s bandwidth
    )
    
    # Run batch inference
    print("Running batch inference...")
    total_tokens = 0
    t_start = time.perf_counter()
    with wired_limit(model, [generation_stream]):
        uids = gen.insert(prompts, max_tokens=24000)
        results = {uid: [] for uid in uids}
        prompt_tokens = {uid: len(prompts[i]) for i, uid in enumerate(uids)}

        while responses := gen.next():
            for r in responses:
                if r.finish_reason is None:
                    results[r.uid].append(r.token)
                    total_tokens += 1
    t_elapsed = time.perf_counter() - t_start

    # Print generation metrics
    tps = total_tokens / t_elapsed if t_elapsed > 0 else 0.0
    print(f"\n--- Generation Metrics ---")
    print(f"  Elapsed time   : {t_elapsed:.2f}s")
    print(f"  Total tokens   : {total_tokens}")
    print(f"  Throughput     : {tps:.1f} tok/s")
    print(f"  Prompts        : {len(uids)}")
    for uid, tokens in results.items():
        gen_toks = len(tokens)
        prompt_toks = prompt_tokens.get(uid, 0)
        print(f"  UID {uid:3d}: prompt={prompt_toks} gen={gen_toks} total={prompt_toks+gen_toks}")
    
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