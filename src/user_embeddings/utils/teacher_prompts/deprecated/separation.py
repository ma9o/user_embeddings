from typing import Union

from pydantic import BaseModel

from ..utils import load_prompt


class SeparationOutput(BaseModel):
    context: str
    actions: list[Union[str, "SeparationOutput"]]


PROMPT, VERSION = load_prompt("separation")
