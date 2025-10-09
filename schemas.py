from typing import List

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Criticize what is missing.")
    superfluous: str = Field(description="Criticize what is superfluous.")


class AnswerQuestion(BaseModel):
    """Answer a question."""

    answer: str = Field(description="250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Reflect on the initial answer.")
    search_query: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ReviseAnswer(BaseModel):
    """Revise your original answer based on the critique and search queries."""

    references: List[str] = Field(
        description="URL citations motivating your updated answer. Must be actual URLs like https://www.example.com"
    )
