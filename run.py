#pip install mlx-lm tiktoken sentencepiece fastapi uvicorn pydantic
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import time
import uuid
import uvicorn

app = FastAPI()

# Load model once at startup
model, tokenizer = load("/Users/rnt/dev/models/openai/gpt-oss-20b-mlx-Q8")

MODEL_ID = "gpt-oss-20b-mlx-Q8"

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[Message]
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: Optional[float] = None

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }]
    }

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    # Prepare the prompt
    if tokenizer.chat_template is not None:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        full_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        full_prompt = "\n".join([m.content for m in request.messages])

    # Write prompt to file
    with open("prompts.log", "a") as f:
        f.write(f"--- Prompt at {time.time()} ---\n{full_prompt}\n\n")

    # Build generate kwargs
    #sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
    sampler = make_sampler(temp=0.0)
    gen_kwargs = dict(
        #max_tokens=request.max_tokens,
        sampler=sampler,
        verbose=True,
    )
    # Generate response
    response = generate(model, tokenizer, prompt=full_prompt)#, **gen_kwargs)
    print(response)

    completion = response

    # Calculate token counts
    prompt_tokens = len(tokenizer.encode(full_prompt))
    completion_tokens = len(tokenizer.encode(completion))
    total_tokens = prompt_tokens + completion_tokens

    # Format OpenAI-compatible chat completion response
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": completion,
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8787)
