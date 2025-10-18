# implementation of the react agent using the langchain framework
# instead of using "from langchain.agents.react.agent import create_react_agent"

from typing import List, Union

from dotenv import load_dotenv
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, tool
from langchain_core.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain import hub

from callbacks import AgentCallbackHandler
from log import my_format_log

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


def my_react_agent(input: str) -> str:
    # print("Hello from langchain-course!")
    # print(get_text_length("Dog"))
    # print(get_text_length.invoke(input={"text": "Dog"}))
    tools = [get_text_length]

    # https://smith.langchain.com/hub/hwchase17/react
    react_prompt = hub.pull("hwchase17/react")

    prompt = PromptTemplate(
        input_variables=["tools", "input", "agent_scratchpad", "tool_names"],
        template=react_prompt.template,
    ).partial(
        tools=render_text_description(tools),  # render the tools to a text description
        tool_names=", ".join([t.name for t in tools]),
    )
        
    # lesson 27 01:01
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        stop="Observation:",  # stop the llm when it sees the word "Observation:"
        callbacks=[AgentCallbackHandler()],
    )

    # the "|" takes the output of the previous step and passes it as input to the next step
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: my_format_log(x["agent_scratchpad"]),  # format the agent scratchpad to a log
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser() # format the output of the llm to the ReAct format
    )

    intermediate_steps = []   # history of intermediate steps
    agent_step = ""

    # loop until the agent finishes
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                # "input": "What is the length of the text 'Dog'?",
                "input": input,
                "agent_scratchpad": intermediate_steps,
            }
        )

        if isinstance(agent_step, AgentAction):
            tool = find_tool_by_name(tools, agent_step.tool)
            observation = tool.invoke(agent_step.tool_input)
            intermediate_steps.append((agent_step, str(observation)))

    return agent_step.return_values["output"]

def main():
    result = my_react_agent(input="What is the length of the text 'Dog'?")
    print(result)

if __name__ == "__main__":
    main()
