#pip install mlx-lm tiktoken sentencepiece
from mlx_lm import load, generate

model, tokenizer = load("/Users/rnt/dev/models/openai/gpt-oss-20b-mlx-Q8")

with open("prompts.log") as f:
    prompt = f.read()

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=100000)
