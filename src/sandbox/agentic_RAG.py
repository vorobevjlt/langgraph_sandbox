
import asyncio

import requests
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from settings import settings


@tool
def fetch_documentation(url: str) -> str:
    """Retrieve and transform documentation from URL"""

    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    return response.text

system_prompt = """
You are an expert Agentic RAG engeneer and a technical assistant.
Your main role is to help users with questions about Anthropic.

To answer user questions, you have access to documentation links — use them, as they contain the most up-to-date information:
https://docs.langchain.com/oss/python/langchain/agents
Use fetch_documentation to retrieve information.
"""

tools = [fetch_documentation]
model = ChatOpenAI(
    model="gpt-5.2",
    temperature=0.1,
    api_key=settings.OPENAI_API_KEY,
    base_url="https://api.proxyapi.ru/openai/v1",
    max_retries=2,
)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    name="Agentic RAG",
)

async def main():
    user_message = HumanMessage('Give me most important core Agent components. Answer short and percist')

    async for chunk in agent.astream({"messages": [user_message]}):
        print(chunk)


if __name__ == '__main__':
    asyncio.run(main())
