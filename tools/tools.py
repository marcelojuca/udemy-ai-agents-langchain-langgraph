from langchain_tavily import TavilySearch

def get_profile_url_tavily(name: str) -> str:
    """
    Search for a person's LinkedIn/Twitter profile URL using Tavily.
    """
    search = TavilySearch()
    res = search.run(f"{name}")
    return res