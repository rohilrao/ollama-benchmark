import asyncio
from openai import AsyncOpenAI

# Point client to your vLLM container endpoint
client = AsyncOpenAI(
    base_url="http://localhost:8555/v1",
    api_key="EMPTY"  # vLLM does not require an API key by default
)

MODEL_NAME = "/models/deepseek-r1"

async def stream_request(req_id: int, prompt: str):
    """Sends a single streaming request and prints tokens with R<req>_T<token> prefix."""
    token_idx = 1
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=60,  # Adjust token length as needed
            temperature=0.7
        )

        async for chunk in response:
            # Safely check if choices exist (the final chunk can sometimes be empty)
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Safely extract both reasoning and standard content (vLLM can return None)
            reasoning = getattr(delta, "reasoning_content", None)
            content = getattr(delta, "content", None)

            # Combine them in case a single chunk contains the transition from reasoning to content
            token_text = ""
            if reasoning:
                token_text += reasoning
            if content:
                token_text += content

            if token_text:
                print(f"R{req_id}_T{token_idx}: {repr(token_text)}", flush=True)
                token_idx += 1

    except Exception as e:
        print(f"R{req_id}: Error -> {e}", flush=True)

async def main():
    prompts = [
        f"Give me 3 unique words starting with the letter '{chr(65 + i)}' and define them briefly."
        for i in range(10)
    ]

    print("Starting 10 parallel streaming requests...\n")
    tasks = [stream_request(req_id=i + 1, prompt=prompt) for i, prompt in enumerate(prompts)]
    
    await asyncio.gather(*tasks)
    print("\nAll 10 requests completed.")

if __name__ == "__main__":
    asyncio.run(main())
