import json
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel

PROMPT_DIR = Path(__file__).parent.parent.parent.parent.parent / "prompts"


def load_prompt(prompt_name: str, version: str = "latest") -> tuple[str, str]:
    if version != "latest":
        prompt_path = PROMPT_DIR / prompt_name / f"{version}.txt"
    else:
        # Sort files alphabetically and take the latest
        # Fix: Use .glob() to get an iterable list of paths
        prompt_files = list((PROMPT_DIR / prompt_name).glob("*.txt"))
        if not prompt_files:
            raise FileNotFoundError(
                f"No prompt files found in {PROMPT_DIR / prompt_name}"
            )
        prompt_path = sorted(prompt_files)[-1]

    version_str = prompt_path.stem.split(".")[0]

    print(f"Loading prompt {prompt_name} from {prompt_path}")

    with open(prompt_path, "r") as f:
        return f.read(), version_str


def get_prompt(prompt: str, user_context_raw: str) -> str:
    try:
        json.loads(user_context_raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in user_context_raw: {e}") from e
    return f"{prompt}\n{user_context_raw}\n```"


def parse_output(last_json_str: str, pydantic_model: Type[BaseModel]) -> Any:
    """Parses the LLM output to extract the hierarchical summaries."""
    parsed_data = _extract_last_json(last_json_str)
    if parsed_data is None:
        raise ValueError("No valid JSON object found in the inference output.")

    return pydantic_model.model_validate(parsed_data).model_dump()


def _extract_last_json(text: str) -> dict | None:
    """
    Finds and parses the last valid top-level JSON object enclosed in {}
    within a string, correctly handling nested structures.

    Args:
        text: The string potentially containing JSON objects and other text.

    Returns:
        The parsed JSON object as a dictionary, or None if no valid JSON object
        is found.
    """
    end_index = len(text) - 1
    while True:
        # Find the last '}' at or before end_index
        end_index = text.rfind("}", 0, end_index + 1)
        if end_index == -1:
            return None  # No more '}' found

        # Try to find the matching '{' for this '}' by tracking brace levels
        start_index = -1
        brace_level = 0
        for i in range(end_index, -1, -1):
            char = text[i]
            if char == "}":
                brace_level += 1
            elif char == "{":
                brace_level -= 1
                if brace_level == 0:
                    start_index = i
                    break  # Found the matching '{'

        if start_index != -1:
            # Extract the potential JSON substring
            potential_json_str = text[start_index : end_index + 1]
            try:
                # Attempt to parse the substring as JSON
                # Consider using repair_json if slightly malformed JSON is expected
                parsed_json = json.loads(potential_json_str)
                # Ensure it's a dictionary (object), not just an array or primitive
                if isinstance(parsed_json, dict):
                    return parsed_json  # Return the first valid JSON object found from the end
            except json.JSONDecodeError:
                # If parsing fails, this segment wasn't a valid JSON object.
                # Continue searching from before this '}' in the next iteration.
                pass

        # Move search backward: look for a '}' before the one we just processed
        end_index -= 1
        if end_index < 0:
            return None  # Reached beginning of string without finding valid JSON
