from typing import List, Dict, Any
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Summary(BaseModel):
    summary: str = Field(description="The summary of the person")
    facts: List[str] = Field(description="interesting facts about the person")

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "facts": self.facts}

    # def to_json(self) -> str:
    #     return json.dumps(self.to_dict())

    # def to_markdown(self) -> str:
    #     return f"**Summary:** {self.summary}\n\n**Facts:** {'\n'.join(self.facts)}"


summary_parser = PydanticOutputParser(pydantic_object=Summary)