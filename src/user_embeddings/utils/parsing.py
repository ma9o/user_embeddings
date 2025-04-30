import json
import re
from typing import Any, Optional

from json_repair import repair_json


def _extract_last_json(text: Optional[str]) -> Optional[str]:
    """
    Finds the last valid top-level JSON object string enclosed in {}
    within a string, correctly handling nested structures.
    Returns the string segment corresponding to the JSON object, or None.
    """
    if not text:
        return None

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
            # Basic check: Does it start with { and end with }?
            # More robust validation happens during parsing.
            if potential_json_str.startswith("{") and potential_json_str.endswith("}"):
                # Perform a quick parse check to increase confidence
                try:
                    json.loads(potential_json_str)
                    return potential_json_str  # Return the string segment
                except json.JSONDecodeError:
                    pass  # Not valid JSON, continue search

        # Move search backward: look for a '}' before the one we just processed
        end_index -= 1
        if end_index < 0:
            return None  # Reached beginning of string without finding valid JSON


def parse_llm_json_output(
    llm_output: Optional[str],
    expect_type: type = dict,  # Default to expecting a dictionary
    return_on_error: Optional[Any] = None,
    verbose_errors: bool = True,  # Default to True for better debugging during dev
) -> Optional[Any]:
    """
    Robustly parses JSON output from an LLM string.

    Handles:
        - None or empty input.
        - Extracting the *last* JSON object string using _extract_last_json.
        - Extracting JSON from markdown code blocks (```json ... ```) as a fallback.
        - Using json_repair for robustness on the extracted/selected string.
        - Checking if the parsed type matches expect_type (dict or list).

    Args:
        llm_output: The raw string output from the LLM.
        expect_type: The expected Python type after parsing (dict or list). Defaults to dict.
        return_on_error: Value to return if parsing fails or type mismatch occurs. Defaults to None.
        verbose_errors: If True, print detailed errors to console. Defaults to True.

    Returns:
        The parsed JSON object (dict or list) if successful and type matches,
        otherwise returns the value specified by return_on_error.
    """
    if not llm_output:
        return return_on_error

    extracted_json_str: Optional[str] = None
    repaired_json_str = ""

    # 1. Try extracting the last JSON object directly
    extracted_json_str = _extract_last_json(llm_output)

    # 2. If direct extraction fails, try extracting from markdown code block
    if not extracted_json_str:
        match = re.search(
            r"```(?:json)?\s*([\s\S]*?)\s*```", llm_output.strip(), re.DOTALL
        )
        if match:
            extracted_json_str = match.group(1).strip()

    # 3. If no JSON string could be extracted, return error
    if not extracted_json_str:
        if verbose_errors:
            print(
                f"Warning: Could not extract JSON object or markdown block from: {llm_output[:100]}..."
            )
        return return_on_error

    # 4. Attempt repair and parsing on the extracted string
    try:
        # Use repair_json first
        repaired_json_str = repair_json(extracted_json_str)
        parsed_json = json.loads(repaired_json_str)

        # 5. Validate type
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
