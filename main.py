"""
🎯 **SOLUTION: Tool Calling Exercise - LangChain Modern Approach**
 
This is the complete solution for transitioning from ReAct prompting to modern tool calling.
Students should implement the "implement_" functions in their exercise.py file.
 
Note: This demonstrates modern LangChain tool calling patterns using simulated classes.
The API is identical to real LangChain: from langchain_openai import ChatOpenAI
"""
 
import os
from typing import List, Any, Dict
 
# LangChain-compatible classes (simulated for exercise)
class Tool:
    """Tool class simulating LangChain's @tool decorator."""
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func
    
    def invoke(self, input_data):
        return self.func(input_data)
 
class AIMessage:
    """AI message response with tool calling capabilities."""
    def __init__(self, content: str = "", tool_calls: List[Dict] = None):
        self.content = content
        self.tool_calls = tool_calls or []
 
class ChatOpenAI:
    """ChatOpenAI class with tool calling capabilities."""
    def __init__(self, temperature=0, model="gpt-3.5-turbo"):
        self.temperature = temperature
        self.model = model
        self.tools = []
    
    def bind_tools(self, tools):
        """Bind tools to the model for tool calling."""
        self.tools = tools
        return self
    
    def invoke(self, messages):
        """Mock invoke that simulates tool calling behavior."""
        if isinstance(messages, str):
            user_input = messages
        elif isinstance(messages, list) and len(messages) > 0:
            user_input = messages[-1].get('content', '') if isinstance(messages[-1], dict) else str(messages[-1])
        else:
            user_input = ""
        
        # Simulate tool calling decision based on input
        if "length" in user_input.lower() and "dog" in user_input.lower():
            return AIMessage(
                content="",  # Empty content when making tool calls (like real OpenAI API)
                tool_calls=[{
                    'name': 'get_text_length',
                    'args': {'text': 'DOG'},
                    'id': 'call_123',
                    'type': 'tool_call'
                }]
            )
        elif "length" in user_input.lower():
            # Extract text from user input for length calculation
            import re
            # Try multiple patterns to extract the word
            patterns = [
                r'length.*?(?:of|for).*?["\']([^"\']+)["\']',  # quoted text
                r'length.*?(?:of|for).*?(?:word|text).*?:\s*([A-Za-z]+)',  # word: WORD  
                r'length.*?(?:of|for).*?(?:word|text)\s+([A-Za-z]+)',  # word WORD
                r'(?:word|text)\s*:\s*([A-Za-z]+)',  # word: WORD
                r'length.*?:\s*([A-Za-z]+)',  # length: WORD
                r'\b([A-Z]+)\b',  # any uppercase word (fallback for DOG)
            ]
            
            text = "unknown"
            for pattern in patterns:
                text_match = re.search(pattern, user_input, re.IGNORECASE)
                if text_match:
                    text = text_match.group(1)
                    break
            
            return AIMessage(
                content="",  # Empty content when making tool calls (like real OpenAI API)
                tool_calls=[{
                    'name': 'get_text_length',
                    'args': {'text': text},
                    'id': 'call_124',
                    'type': 'tool_call'
                }]
            )
 
        else:
            return AIMessage(content="I can help you with text length calculations!")
 
# Tool definition (same as original code)
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"🔍 get_text_length called with text: {text}")
    text = text.strip("'\n").strip('"')  # Clean the text
    return len(text)
 
# Create tool instance
text_length_tool = Tool(
    name="get_text_length",
    description="Returns the length of a text by characters",
    func=get_text_length
)
 
# ===============================================
# 💡 SOLUTION FUNCTIONS WITH DETAILED HINTS
# ===============================================
# These functions demonstrate the complete implementation
# with step-by-step hints to help students understand
# the transition from ReAct prompting to tool calling
#
# 🔄 KEY DIFFERENCES FROM REACT:
# ❌ Old: Complex prompt template + text parsing
# ✅ New: Simple .bind_tools() + structured responses
# ❌ Old: Manual tool selection in prompts  
# ✅ New: LLM vendor optimized tool calling
# ❌ Old: Fragile text format requirements
# ✅ New: Reliable JSON tool call responses
 
def implement_set_api_key(api_key: str):
    """
    SOLUTION: Set the OPENAI_API_KEY environment variable.
    
    Args:
        api_key (str): Your OpenAI API key
        
    💡 HINT: Use os.environ to set environment variables
    Example: os.environ["VARIABLE_NAME"] = value
    """
    os.environ["OPENAI_API_KEY"] = api_key
    print("✓ API key has been set in environment variables!")
    pass

def implement_create_model_with_tools(tools: List[Tool]) -> ChatOpenAI:
    """
    SOLUTION: Create a ChatOpenAI model and bind tools to it.
    This replaces the ReAct prompt template approach!
    
    Args:
        tools (List[Tool]): List of tools to bind to the model
        
    Returns:
        ChatOpenAI: Model with tools bound for tool calling
        
    💡 HINT: 
    1. Create a ChatOpenAI instance with temperature=0
    2. Use the .bind_tools() method to attach the tools
    Example: model = ChatOpenAI(temperature=0)
             model_with_tools = model.bind_tools(tools)
    """
 
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )
    
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools
    
    pass
 
def implement_check_for_tool_calls(response: AIMessage) -> bool:
    """
    SOLUTION: Check if the model response contains any tool calls.
    
    Args:
        response (AIMessage): The model's response
        
    Returns:
        bool: True if there are tool calls, False otherwise
        
    💡 HINT: Check the tool_calls attribute of the response
    Example: return bool(response.tool_calls)
    """
    return bool(response.tool_calls)


def implement_execute_tool_call(tool_call: Dict, available_tools: List[Tool]) -> str:
    """
    SOLUTION: Execute a single tool call and return the result.
    
    Args:
        tool_call (Dict): Tool call information with 'name' and 'args'
        available_tools (List[Tool]): List of available tools
        
    Returns:
        str: The result of the tool execution
        
    💡 HINT:
    1. Find the tool by name from available_tools
    2. Extract the arguments from tool_call['args']
    3. Call the tool with the arguments
    Example: tool_name = tool_call['name']
             tool_args = tool_call['args']
             # Find tool where tool.name == tool_name
             # Execute: result = tool.func(tool_args['text'])
    """
    tool_name = tool_call['name']
    tool_args = tool_call['args']
    
    tool_to_use = None
    
    for tool in available_tools:
        if tool.name == tool_name:
            tool_to_use = tool
        break
    
    if tool_to_use is None:
        raise ValueError(f"Tool with name {tool_call['name']} not found!")
    
    if 'text' in tool_args:
        result = tool_to_use.func(tool_args['text'])
    else:
        first_arg = next(iter(tool_args.values()))
        result = tool_to_use.func(first_arg)

    return str(result)

def implement_run_agent_with_tool_calling(model_with_tools: ChatOpenAI, 
                                        user_input: str, 
                                        available_tools: List[Tool]) -> str:
    """
    SOLUTION: Run the modern tool calling agent.
    This replaces the complex ReAct while loop with a cleaner approach!
    
    Args:
        model_with_tools (ChatOpenAI): Model with tools bound
        user_input (str): The user's question
        available_tools (List[Tool]): Available tools for execution
        
    Returns:
        str: The final answer
        
    💡 HINT: Use the implement_ functions you've already created!
    Algorithm:
    1. Send user_input to the model
    2. Check if response has tool calls (use implement_check_for_tool_calls)
    3. If yes: execute the tool call (use implement_execute_tool_call) and return result
    4. If no tool calls: return the model's content as final answer
    Example: response = model_with_tools.invoke(user_input)
             if implement_check_for_tool_calls(response):
                 # Execute tools and return result
    """

    print(f"🚀 Starting tool calling agent with input: {user_input}")
    response = model_with_tools.invoke(user_input)
    
    if response.content:
        print(f"📨 Model response: {response.content}")
    else:
        print("📨 Model response: [Making tool calls - no content]")
    
    if implement_check_for_tool_calls(response):
        print(f"🔧 Found {len(response.tool_calls)} tool call(s)")
        for tool_call in response.tool_calls:
            print(f"⚡ Executing tool: {tool_call['name']} with args: {tool_call['args']}")
            result = implement_execute_tool_call(tool_call, available_tools)
            print(f"✅ Tool result: {result}")
            return f"The length is {result} characters."
        else:
            print("💬 No tool calls needed, returning direct response")
            return response.content
    pass

# Helper function (same as original)
def check_api_key():
    """Check if OpenAI API key is set."""
    if "OPENAI_API_KEY" not in os.environ:
        raise Exception("❌ OPENAI_API_KEY environment variable is required!")
    print("✅ API key is set!")
 
print("🎯 Tool Calling Solution - Complete Implementation")
print("=" * 55)
print("✨ Demonstrating modern LangChain tool calling patterns!")
print("🔄 This replaces the old ReAct prompting approach")
print()
 
try:
    # Set up API key
    print("🔑 Setting up API key...")
    implement_set_api_key("demo_openai_key_12345")
    check_api_key()
    
    # Create model with tools (this replaces ReAct prompt!)
    print("\n🔧 Creating model with tool calling capabilities...")
    tools = [text_length_tool]
    model_with_tools = implement_create_model_with_tools(tools)
    print("✓ Model created and tools bound successfully!")
    
    # Test the modern agent
    print("\n🤖 Testing modern tool calling agent...")
    user_question = "What is the length of the word: DOG"
    result = implement_run_agent_with_tool_calling(model_with_tools, user_question, tools)
    
    print(f"\n📊 Final Result: {result}")
    
    print("\n🎉 Tool calling solution working perfectly!")
    print("✅ Key improvements over ReAct prompting:")
    print("  - No complex prompt template needed")
    print("  - No text parsing with ReActSingleInputOutputParser")
    print("  - Direct tool call handling from model response")
    print("  - Vendor-optimized tool selection")
    print("  - Cleaner, more reliable agent loop")
    
    print("\n🚀 For real LangChain projects:")
    print("  1. Use: from langchain_openai import ChatOpenAI")
    print("  2. Create tools with @tool decorator")
    print("  3. Bind tools: model.bind_tools([my_tool])")
    print("  4. Handle response.tool_calls directly")
    print("  5. Execute tools and get structured results")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("This shouldn't happen in the complete solution!") 