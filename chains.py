import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (JsonOutputToolsParser,
                                                        PydanticToolsParser)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
        Current time: {time}
        1. {first_instruction}
        2. Reflect and critique your answer. Be severe to maximize improvement.
        3. Recommend search queries to research information and improve your answer.
        
        You MUST use the AnswerQuestion tool and provide ALL required fields:
        - answer: Your detailed answer (around 250 words)
        - reflection: Criticize what is missing and what is superfluous
        - search_query: 1-3 search queries for researching improvements""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the AnswerQuestion tool with all required fields.",
        ),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_instructions = "Provide a detailed ~250 word answer."
revise_instructions = """
    Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
    - You MUST include numerical citations in your revised answer to ensure it can be verified.
    - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
        - [1] https://example.com
        - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
    
    IMPORTANT: For the references field, provide actual URLs (like https://www.company.com) that support your answer, not just numbers.
    """

first_responder_chain = actor_prompt_template.partial(
    first_instruction=first_responder_instructions
) | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

revise_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == "__main__":

    # Test first response chain
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."
    )

    first_response = first_responder_chain | parser_pydantic
    res_first_response = first_response.invoke({"messages": [human_message]})
    print("--------------------------------")
    print("First response - answer:", res_first_response[0].answer)
    print("--------------------------------")
    print(
        "First response - reflection - missing:",
        res_first_response[0].reflection.missing,
    )
    print("--------------------------------")
    print(
        "First response - reflection - superfluous:",
        res_first_response[0].reflection.superfluous,
    )
    print("--------------------------------")
    print("First response - search_query:", res_first_response[0].search_query)
    print("--------------------------------")
    # Test revise chain
    revise_response = revise_chain | PydanticToolsParser(tools=[ReviseAnswer])
    # Create a message that includes the original question and the first response for review
    from langchain_core.messages import AIMessage

    review_message = HumanMessage(
        content=f"Original question: {human_message.content}\n\nFirst response to review: {res_first_response[0].answer}"
    )

    res_revise_response = revise_response.invoke({"messages": [review_message]})
    print("--------------------------------")
    print("Revise response - references:", res_revise_response[0].references)
    print("--------------------------------")
