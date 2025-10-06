from dotenv import load_dotenv
from langchain import hub
from langchain_openai import OpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_csv_agent

load_dotenv()


def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    You have qrcode package installed
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)
    tools = [PythonREPLTool()]
    agent = create_react_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    input_agent_executor = """Generate and save in current working directory five qr codes that points to the url https://www.udemy.com/course/langchain. 
    You have qrcode package installed.
    """
    # result = agent_executor.invoke({"input": input_agent_executor})
    # print(result)

    csv_agent = create_csv_agent(
        # llm=ChatOpenAI(temperature=0, model="gpt-5-mini"),    
            # this model does not support stop : LangChain’s agent is sending a stop parameter to OpenAI’s Chat Completions API while streaming. 
            # The specific model you configured rejects stop, so OpenAI returns a 400 BadRequest: “Unsupported parameter: 'stop' is not supported with this model.”
        # llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),    # this model does not support 
            # Because gpt-4o-mini is a much smaller model, it’s less reliable at following the strict ReAct/tool-usage patterns LangChain agents expect. 
            # In this setup, the CSV agent depends on the model to write and execute Python correctly (via the REPL) and to adhere to the agent’s output format.
            # Use a more capable model for agents that must execute code: gpt-4, gpt-4o, or gpt-4.1. 
            # https://platform.openai.com/docs/models  - official list, capabilities, context windows, tool/function-calling support, and deprecations.
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

    input_csv_agent = """
        Print the seasons by ascending order of the number of episodes they have.
    """

    result = csv_agent.invoke({"input": input_csv_agent})
    print(result)


if __name__ == "__main__":
    main()
