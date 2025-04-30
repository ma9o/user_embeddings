from pydantic import BaseModel, Field

from .utils import load_prompt

PROMPT, VERSION = load_prompt("inference")


class PromptOutput(BaseModel):
    """Pydantic model for validating the output of the Inference prompt."""

    inference_output: str = Field(..., description="The output of the inference task.")
