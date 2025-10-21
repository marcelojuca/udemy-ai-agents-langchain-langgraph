def should_include_sources(response_text: str, query: str, source_documents: list) -> bool:
    """
    Determine if sources should be included based on:
    1. Whether the agent indicates it cannot answer
    2. Whether the retrieved documents are relevant to the query
    3. Whether the answer appears to be based on the retrieved documents
    """
    if not response_text or not response_text.strip():
        return False
    
    # Common phrases that indicate the agent cannot answer
    no_answer_phrases = [
        "i'm unable to provide",
        "i cannot provide",
        "i don't have",
        "i do not have",
        "i'm not able to",
        "i am not able to",
        "i don't know",
        "i do not know",
        "i cannot find",
        "i can't find",
        "no information available",
        "no relevant information",
        "unable to find",
        "not available",
        "my training only includes",
        "as my training only includes",
        "i don't have access to",
        "i do not have access to",
        "i cannot access",
        "i can't access",
        "i don't have that information",
        "i do not have that information",
        "i cannot answer",
        "i can't answer",
        "i am unable to answer",
        "i'm unable to answer",
        "i don't have enough information",
        "i do not have enough information",
        "i cannot help with that",
        "i can't help with that",
        "i don't have the information",
        "i do not have the information",
        "i'm not sure",
        "i am not sure",
        "i cannot determine",
        "i can't determine",
        "i don't have data",
        "i do not have data",
        "no data available",
        "insufficient information",
        "i don't have access",
        "i do not have access"
    ]
    
    response_lower = response_text.lower().strip()
    query_lower = query.lower().strip()
    
    # Check if any no-answer phrase is present
    for phrase in no_answer_phrases:
        if phrase in response_lower:
            return False
    
    # Check if response is very short (likely a "no answer" response)
    if len(response_text.strip()) < 30:
        return False
    
    # Check if response starts with common "no answer" patterns
    if response_lower.startswith(("i'm unable", "i cannot", "i don't have", "i do not have", "i'm not able", "i am not able")):
        return False
    
    # Check if no documents were retrieved
    if not source_documents or len(source_documents) == 0:
        return False
    
    # Check if the query is asking for real-time information that wouldn't be in documents
    real_time_queries = [
        "what date is today",
        "what time is it",
        "current date",
        "current time",
        "today's date",
        "what's the weather",
        "current weather",
        "what day is it",
        "what year is it",
        "current year",
        "current month",
        "what month is it"
    ]
    
    for real_time_query in real_time_queries:
        if real_time_query in query_lower:
            return False
    
    # Check if the answer seems to be based on general knowledge rather than documents
    # This is a heuristic - if the answer is very specific and doesn't reference document content
    general_knowledge_indicators = [
        "the current date is",
        "today is",
        "it is currently",
        "as of my last update",
        "based on my knowledge",
        "according to my training",
        "my training data",
        "my knowledge cutoff"
    ]
    
    for indicator in general_knowledge_indicators:
        if indicator in response_lower:
            return False
    
    return True