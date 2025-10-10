from typing import List
from pathlib import Path

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import revise_chain, first_responder_chain
from tool_executor import execute_tools

MAX_ITERATIONS = 2
builder = MessageGraph()
builder.add_node("draft", first_responder_chain)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_chain)
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop, {END:END, "execute_tools":"execute_tools"})
builder.set_entry_point("draft")
graph = builder.compile()

# Save the graph as PNG
graph.get_graph().draw_mermaid_png(output_file_path=Path("graph.png"))
print("Graph saved as graph.png")


res = graph.invoke(
    "Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital."
)

print("\n" + "="*50)
print("FINAL RESULT:")
print("="*50)
for i, message in enumerate(res):
    print(f"Message {i}: {type(message).__name__}")
    print(f"Content: {message.content}")
    if hasattr(message, 'tool_calls') and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    print("-" * 30)