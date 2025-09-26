from typing import List

from dotenv import load_dotenv
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI

from callbacks import AgentCallbackHandler

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
    tools = [get_text_length]
    
    # Create LLM and bind tools to it
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    ).bind_tools(tools)

    # Simple conversation with tool calling
    response = llm.invoke("What is the length of the text 'Dog'?")
    
    print("Response:", response.content)
    
    # Check if the model wants to call any tools
    if response.tool_calls:
        print(f"\nModel wants to call {len(response.tool_calls)} tool(s):")
        for tool_call in response.tool_calls:
            print(f"- Tool: {tool_call['name']}")
            print(f"  Args: {tool_call['args']}")
            
            # Execute the tool
            tool = find_tool_by_name(tools, tool_call['name'])
            result = tool.invoke(tool_call['args'])
            print(f"  Result: {result}")
            
            # Get final response with tool result
            final_response = llm.invoke([
                {"role": "user", "content": "What is the length of the text 'Dog'?"},
                {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls},
                {"role": "tool", "content": str(result), "tool_call_id": tool_call['id']}
            ])
            print(f"\nFinal answer: {final_response.content}")
    else:
        print("No tools were called.")


if __name__ == "__main__":
    main()
