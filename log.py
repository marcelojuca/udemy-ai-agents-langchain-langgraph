from typing import List, Tuple

from langchain_core.agents import AgentAction


def format_log(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "Observation:",
    llm_prefix: str = "Thought:",
) -> str:
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += "\n" + observation_prefix + observation
        thoughts += "\n" + llm_prefix
    return thoughts
