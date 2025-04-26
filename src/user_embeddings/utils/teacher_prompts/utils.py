import json


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
