from typing import Annotated, TypedDict

from chains import generation_chain, reflection_chain
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv("../.env")


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def generation_node(state: State) -> State:
    res = generation_chain.invoke({"messages": state["messages"]})
    return {"messages": [res]}


def reflection_nodes(state: State) -> State:
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


REFLECT = "reflect"
GENERATE = "generate"

builder = StateGraph(state_schema=State)
builder.add_node(REFLECT, reflection_nodes)
builder.add_node(GENERATE, generation_node)
builder.set_entry_point(GENERATE)


def should_continue(state: State) -> str:
    if len(state["messages"]) > 3:
        return END
    return REFLECT


builder.add_conditional_edges(
    GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT}
)
builder.add_edge(REFLECT, GENERATE)
graph = builder.compile()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [HumanMessage(content="""
                Make this tweet better:"
                @LangChainAI — newly Tool Calling feature is seriously underrated.
                After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.
                Made a video covering their newest blog post

                """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)
