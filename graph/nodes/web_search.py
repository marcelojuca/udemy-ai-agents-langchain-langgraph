from typing import Any, Dict

from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Performs a web search for the given question
    """
    print("---PERFORMING WEB SEARCH---")
    question = state["question"]
    # handle cases where documents doesn't exist in the state 
    documents = state.get("documents", [])
    docs = web_search_tool.invoke(question)
    web_results = "\n".join(docs)
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}
