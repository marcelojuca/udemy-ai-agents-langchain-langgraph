from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
tools = [TavilySearch()]
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm = llm, 
    tools = tools, 
    prompt=react_prompt
)
agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True,
    handle_parsing_errors=True,
    max_iterations=5
)
chain = agent_executor

def main():
    result = chain.invoke(
        input = {
            "input": "search for 3 job postings for an ai engineer using langchain in Calgary on LinkedIn and list their details"
        }
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
