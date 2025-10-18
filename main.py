# use of predefined react function from langchain
# instead of developing our own react agent from scratch

from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

# print("-"*50)
# print(output_parser.get_format_instructions())
# print("-"*50)

# https://smith.langchain.com/hub/hwchase17/react
react_prompt = hub.pull("hwchase17/react")

final_answer_format_instructions = """
\n\nIMPORTANT: Final Answer: the final answer to the original input question formatted 
according to format_instructions: {format_instructions}
"""
# print("-"*50)
# print(react_prompt.template)
# print("-"*50)

react_prompt_with_format_instructions = PromptTemplate(
    input_variables=["tools", "input", "agent_scratchpad", "tool_names"],
    template=react_prompt.template + final_answer_format_instructions,
).partial(format_instructions=output_parser.get_format_instructions())

# The {format_instructions} originates from the PydanticOutputParser’s get_format_instructions() method, 
# which is based on the AgentResponse Pydantic model. It’s injected into the PromptTemplate via the 
# .partial() method to ensure the language model’s output is structured correctly for parsing. 
# This mechanism allows the agent to produce a standardized, parseable response that aligns with 
# the expected schema.


# react_prompt_with_format_instructions = PromptTemplate(
#     input_variables=["input", "agent_scratchpad", "tool_names"],
#     template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
# ).partial(format_instructions=output_parser.get_format_instructions())


agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | parse_output


def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
        }
    )
    print("Job Positions Found:")
    print("=" * 50)
    print(result.answer)
    print("\n" + "=" * 50)
    print("Sources:")
    for i, source in enumerate(result.sources, 1):
        print(f"{i}. {source.url}")


if __name__ == "__main__":
    main()