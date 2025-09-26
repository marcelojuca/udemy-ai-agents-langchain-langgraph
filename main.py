from typing import List, Union

from dotenv import load_dotenv
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, tool
from langchain_core.tools.render import render_text_description
from langchain_openai import ChatOpenAI

from callbacks import AgentCallbackHandler
from log import format_log

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text"""
    print(f"Getting the length of the text: {text}")

    # stripping away non-printable characters
    text = text.strip("'\n").strip('"')

    return len(text)


def find_tool_by_name(tools: List[Tool], name: str) -> Tool:
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool with name {name} not found")


def main():
    # print("Hello from langchain-course!")
    # print(get_text_length("Dog"))
    # print(get_text_length.invoke(input={"text": "Dog"}))
    tools = [get_text_length]
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    # "partial" populates the variables in the template
    prompt = PromptTemplate.from_template(template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # lesson 27 01:01
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        stop="Observation:",
        callbacks=[AgentCallbackHandler()],
    )

    # history of intermediate steps
    intermediate_steps = []

    # the "|" takes the output of the previous step and passes it as input to the next step
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of the text 'Dog'?",
                "agent_scratchpad": intermediate_steps,
            }
        )

        if isinstance(agent_step, AgentAction):
            tool = find_tool_by_name(tools, agent_step.tool)
            observation = tool.invoke(agent_step.tool_input)
            intermediate_steps.append((agent_step, str(observation)))


if __name__ == "__main__":
    main()
