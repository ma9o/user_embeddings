import json
import re
from typing import Any, Optional

from json_repair import repair_json


def parse_llm_json_output(
    llm_output: Optional[str],
    expect_type: type = dict,  # Default to expecting a dictionary
    return_on_error: Optional[Any] = None,
    verbose_errors: bool = False,  # Add flag to control error printing
) -> Optional[Any]:
    """
    Robustly parses JSON output from an LLM string.

    Handles:
        - None or empty input.
        - Extracting JSON from markdown code blocks (```json ... ```).
        - Using json_repair for robustness.
        - Checking if the parsed type matches expect_type (dict or list).

    Args:
        llm_output: The raw string output from the LLM.
        expect_type: The expected Python type after parsing (dict or list). Defaults to dict.
        return_on_error: Value to return if parsing fails or type mismatch occurs. Defaults to None.
        verbose_errors: If True, print detailed errors to console. Defaults to False.

    Returns:
        The parsed JSON object (dict or list) if successful and type matches,
        otherwise returns the value specified by return_on_error.
    """
    if not llm_output:
        return return_on_error

    json_str = llm_output.strip()
    repaired_json_str = ""

    # 1. Extract from markdown code block if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_str, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    # Else, assume the whole string might be JSON (or needs repair)

    # 2. Attempt repair and parsing
    try:
        # Use repair_json first
        repaired_json_str = repair_json(json_str)
        parsed_json = json.loads(repaired_json_str)

        # 3. Validate type
        if isinstance(parsed_json, expect_type):
            return parsed_json
        else:
            if verbose_errors:
                print(
                    f"Warning: Parsed JSON type ({type(parsed_json).__name__}) does not match expected type ({expect_type.__name__}). Output: {llm_output[:100]}..."
                )
            return return_on_error

    except (json.JSONDecodeError, TypeError, Exception) as e:
        # Catch potential errors during repair or final loads
        if verbose_errors:
            print(
                f"Error parsing LLM JSON output: {e}\nRepaired string snippet: {repaired_json_str[:100]}...\nRaw output snippet: {llm_output[:100]}..."
            )
        return return_on_error
