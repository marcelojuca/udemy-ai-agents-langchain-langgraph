from dotenv import load_dotenv
from langchain import hub
from langchain_openai import OpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents import Tool
from typing import Any, Mapping

load_dotenv()


def main():
    print("Start...")

    base_prompt = hub.pull("langchain-ai/react-agent-template")

    ####################################################### ReAct Agent #######################################################
    python_agent_instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    You have qrcode package installed
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    python_agent_prompt = base_prompt.partial(instructions=python_agent_instructions)
    python_agent_tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        tools=python_agent_tools,
        prompt=python_agent_prompt,
    )
    python_agent_executor = AgentExecutor(
        agent=python_agent,
        tools=python_agent_tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    # input_python_agent_executor = """Generate five qr codes that points to the url https://www.udemy.com/course/langchain 
    # then save them to a subdirectory called qr_codes. In case this folder does not exist, create it. 
    # You have qrcode package installed.
    # """

    # result = python_agent_executor.invoke({"input": input_python_agent_executor})
    # print(result)

    ####################################################### CSV Agent #######################################################
    # create_csv_agent uses pandas library to read the csv file and use the llm to answer the question
    csv_agent = create_csv_agent(
        # llm=ChatOpenAI(temperature=0, model="gpt-5-mini"),    
            # this model does not support stop : LangChain’s agent is sending a stop parameter to OpenAI’s Chat Completions API while streaming. 
            # The specific model you configured rejects stop, so OpenAI returns a 400 BadRequest: “Unsupported parameter: 'stop' is not supported with this model.”

        # llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),    # this model does not support 
            # Because gpt-4o-mini is a much smaller model, it’s less reliable at following the strict ReAct/tool-usage patterns LangChain agents expect. 
            # In this setup, the CSV agent depends on the model to write and execute Python correctly (via the REPL) and to adhere to the agent’s output format.
            # Use a more capable model for agents that must execute code: gpt-4, gpt-4o, or gpt-4.1. 
            # https://platform.openai.com/docs/models  - official list, capabilities, context windows, tool/function-calling support, and deprecations.
            # https://python.langchain.com/docs/how_to/#agents
            # 

        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    # input_csv_agent = """How many columns are there in the file episode_info.csv?"""
    
    # input_csv_agent = """Which writer wrote the most episodes? how many episodes did the writer write? 
    # Take into account that some episodes might have multiple writers so you will need to identify the writer 
    # from the list of writers and also count the number of episodes. 
    # The final answer should be 58 episodes. Then, create a better prompt I should have used to get the answer.
    # """

    # input_csv_agent = """Which writer wrote the most episodes? how many episodes did the writer write? 
    # Considering that some episodes might have multiple writers, identify the writer who wrote the most episodes 
    # and count the number of episodes they wrote. 
    # Clean the data to ensure that each writer is listed only once.
    # """

    # input_csv_agent = """
    #     Print the seasons by ascending order of the number of episodes they have.
    # """

    # result = csv_agent.invoke({"input": input_csv_agent})
    # print(result)

    ####################################################### Router Agent #######################################################
    def run_python_agent_via_tool(tool_input: Any) -> Any:
        """Normalize tool input and invoke the python agent.

        Accepts either a natural language instruction (preferred) or a dict
        with keys like 'input' or 'code'. If 'code' is provided, it wraps the
        code into an instruction for execution.
        """
        if isinstance(tool_input, Mapping):
            maybe_text = tool_input.get("input")
            if isinstance(maybe_text, str):
                return python_agent_executor.invoke({"input": maybe_text})
            maybe_code = tool_input.get("code")
            if isinstance(maybe_code, str):
                instruction = (
                    "Execute the following Python code and return the outputs. "
                    "If files are created, save them to the requested paths.\n" + maybe_code
                )
                return python_agent_executor.invoke({"input": instruction})
            # Fallback for other mapping inputs
            return python_agent_executor.invoke({"input": str(tool_input)})
        # Non-dict inputs treated as plain instruction
        return python_agent_executor.invoke({"input": str(tool_input)})

    router_agent_tools = [
        Tool(
            name="python_agent",
            func=run_python_agent_via_tool,
            description="""useful to write and execute Python. Accepts natural language instructions
                         or an object with 'input' or 'code' (string). Returns execution results.""",
        ),
        Tool(
            name="csv_agent",
            func=csv_agent.invoke,
            description="""useful when you need to answer question over episode_info.csv file,
                         takes an input the entire question and returns the answer after running pandas calculations""",
        ),
    ]
    router_agent_prompt = base_prompt.partial(instructions="")
    router_agent = create_react_agent(
        prompt=router_agent_prompt,
        tools=router_agent_tools,
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
    )
    router_agent_executor = AgentExecutor(
        agent=router_agent,
        tools=router_agent_tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    input_router_agent = """
        Which season has the least episodes?
        Next, generate the same amount of qr codes. These qr codes points to the url https://www.udemy.com/course/langchain. 
        Then save these qr codes to a subdirectory called 'qr_codes'. In case this subdirectory does not exist, create it. 
        You have qrcode package installed.
    """
    result = router_agent_executor.invoke({"input": input_router_agent})
    print(result)

if __name__ == "__main__":
    main()
