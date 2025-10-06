import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tools.tools import get_profile_url_tavily

load_dotenv()

def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.environ["OPENAI_API_KEY"])
    template = """Given the full name {name_of_person}, search for the LinkedIn profile URL on the internet. 
    You must use the available tool to search for the person's LinkedIn profile.
    Return only the plain LinkedIn profile URL (e.g., https://www.linkedin.com/in/username), no markdown formatting or additional text."""
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 LinkedIn profile",
            func=get_profile_url_tavily,
            description="useful when you need to find the LinkedIn profile URL from a person's name. Use this tool to search for the person's LinkedIn profile.",
        ),
    ]
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools_for_agent, 
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=True
    )
    result = agent_executor.invoke(
        {"input": prompt_template.format_prompt(name_of_person=name)}
    )
    linkedin_profile_url = result["output"]
    return linkedin_profile_url


if __name__ == "__main__":
    print(lookup("Eden Marco"))