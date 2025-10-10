from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=5)


def run_search_queries(search_query: list[str], **kwargs):
    """Run search queries and return the results"""
    return tavily_tool.batch([{"query": query} for query in search_query])


def run_references(references: list[str], **kwargs):
    """Process references (no search needed)"""
    return [f"Reference processed: {ref}" for ref in references]


execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_search_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_references, name=ReviseAnswer.__name__),
    ]
)
