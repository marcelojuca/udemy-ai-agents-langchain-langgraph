from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated, TypedDict

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage
from chains_2 import generate, reflect

#######
query = "Write an essay on why the little prince is relevant in modern childhood"
request = HumanMessage(content=query)
########################## Generate Essay ##########################
essay = ""
for chunk in generate.stream({"messages": [request]}):
    print(chunk.content, end="")
    essay += chunk.content

########################## Reflect on Essay ##########################
reflection = ""
for chunk in reflect.stream({"messages": [request, HumanMessage(content=essay)]}):
    print(chunk.content, end="")
    reflection += chunk.content


for chunk in generate.stream({"messages": [request, AIMessage(content=essay), HumanMessage(content=reflection)]}
):
    print(chunk.content, end="")

############################################################

class State(TypedDict):
    messages: Annotated[list, add_messages]


async def generation_node(state: State) -> State:
    return {"messages": [await generate.ainvoke(state["messages"])]}


async def reflection_node(state: State) -> State:
    # Other messages we need to adjust
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    res = await reflect.ainvoke(translated)
    # We treat the output of this as human feedback for the generator
    return {"messages": [HumanMessage(content=res.content)]}


REFLECT = "reflect"
GENERATE = "generate"

builder = StateGraph(State)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
# builder.add_edge(START, "generate")
builder.set_entry_point(GENERATE)


def should_continue(state: State):
    if len(state["messages"]) > 3:
        # End after 3 iterations
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)