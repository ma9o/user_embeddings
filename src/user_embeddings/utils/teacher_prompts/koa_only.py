from typing import List

from pydantic import BaseModel, Field

from .utils import load_prompt


class PromptOutput(BaseModel):
    """Pydantic model for validating the output of the KOA extraction prompt."""

    koa: List[str] = Field(
        ...,
        description="List of extracted Knowledge, Opinion, or Attribute statements.",
    )


PROMPT = load_prompt("koa_only")
