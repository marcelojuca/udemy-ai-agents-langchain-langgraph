import asyncio
import os

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage


load_dotenv()

stdio_server_params = StdioServerParameters(
    command="python",
    args=["servers/math_server.py"],
)

llm = ChatOpenAI(model="gpt-4o-mini")


async def main() -> None:
    # print("Hello from langchain-course!")
    async with stdio_client(stdio_server_params) as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            # print("session initialized")
            tools = await load_mcp_tools(session)
            # print(tools)
            # print("--------------------------------\n")
            agent = create_react_agent(model=llm, tools=tools)
            result = await agent.ainvoke({"messages": [HumanMessage(content="What is 54 + 2 * 3?")]})
            print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
