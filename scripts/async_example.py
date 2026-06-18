python -c '
import asyncio
from ollama import AsyncClient

async def main():
    client = AsyncClient(host="http://localhost:11440")
    print("Connecting and waiting for stream...\n")
    
    async for chunk in await client.chat(
        model="mistral-small3.2:24b-32k", 
        messages=[{"role": "user", "content": "Count from 1 to 5."}], 
        stream=True
    ):
        # Extract the content and wrap it in brackets to visualize the chunks
        content = chunk.get("message", {}).get("content", "")
        print(f"|{content}|", end="", flush=True)
        
    print("\n\n[Stream Complete]")

asyncio.run(main())
'
