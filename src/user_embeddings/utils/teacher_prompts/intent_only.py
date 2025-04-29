from typing import List

from pydantic import BaseModel, Field

from .utils import load_prompt


class PromptOutput(BaseModel):
    """Pydantic model for validating the output of the Intent extraction prompt."""

    intents: List[str] = Field(..., description="List of extracted Intent statements.")


PROMPT, VERSION = load_prompt("intent_only")
