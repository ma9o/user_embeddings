from pydantic import BaseModel

from ..utils import load_prompt


class InferenceOutput(BaseModel):
    context: str
    actions: list[list[str]]


PROMPT, VERSION = load_prompt("inference")
