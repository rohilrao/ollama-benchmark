python -c '
import asyncio
from ollama import AsyncClient

async def run_single_request(client, req_id):
    prompt = f"Count from 1 to 5. (Request {req_id})"
    token_count = 0
    
    async for chunk in await client.chat(
        model="mistral-small3.2:24b", 
        messages=[{"role": "user", "content": prompt}], 
        stream=True
    ):
        content = chunk.get("message", {}).get("content", "")
        if content:  # Only count/print if there is actual text
            token_count += 1
            # Format: [ReqID - TokenNum: |text|]
            print(f"[R{req_id}-T{token_count}:|{content}|] ", end="", flush=True)

async def main():
    client = AsyncClient(host="http://localhost:11440")
    print("Launching 5 concurrent streams...\n")
    
    # 1. Create a list of 5 paused tasks
    tasks = [run_single_request(client, i) for i in range(1, 6)]
    
    # 2. Fire them all off to the Ollama server at the exact same time
    await asyncio.gather(*tasks)
        
    print("\n\n[All Streams Complete]")

asyncio.run(main())
'
