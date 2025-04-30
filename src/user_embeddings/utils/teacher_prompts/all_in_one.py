from typing import List

from pydantic import BaseModel, Field

from .utils import load_prompt

PROMPT, VERSION = load_prompt("all_in_one")


class PromptOutput(BaseModel):
    """Pydantic model for validating the output of the All-in-One prompt."""

    koa: List[str] = Field(..., description="List of distilled KOA statements.")
    intents: List[str] = Field(..., description="List of distilled intents.")
