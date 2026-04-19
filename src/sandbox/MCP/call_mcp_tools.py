import asyncio
import json

import aiohttp

MCP_URL = "https://docs.langchain.com/mcp"


async def parse_sse(resp: aiohttp.ClientResponse) -> dict | None:

    async for raw_line in resp.content:
        line = raw_line.decode("utf-8").strip()
        print(f'{line=}')
        if not line or line.startswith("event:"):
            continue
        if line.startswith("data:"):
            line = line[len("data:"):].strip()
        return json.loads(line)

    return None


async def call_mcp_tool(query: str) -> dict:
 
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search_docs_by_lang_chain",
            "arguments": {"query": query},
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(MCP_URL, json=payload, headers=headers, ssl=False) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                return await parse_sse(resp)
            return await resp.json()


async def main():
    result = await call_mcp_tool("how to create a chat agent")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())