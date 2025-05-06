from textwrap import dedent
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .utils import load_prompt


class PromptOutput(BaseModel):
    """Pydantic model for validating the output of the Intent extraction prompt."""

    intents: List[str] = Field(..., description="List of extracted Intent statements.")
    rationale: Optional[str] = Field(
        None, description="Rationale for the intent decision"
    )


PROMPT, VERSION = load_prompt("intent_only")


def format_intent_only_input(inputs: Dict[str, Any]) -> str:
    # Return as a compact JSON string
    return dedent(f"""
    Raw input:
    {inputs.get("__RAW_INPUT__")}

    Inference output:
    {inputs.get("inference").model_dump()}
    """).strip()
