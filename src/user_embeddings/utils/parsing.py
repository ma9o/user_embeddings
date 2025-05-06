import json
import logging
import re
from typing import Any, Optional

from json_repair import repair_json
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def _extract_result_from_xml_tag(text: Optional[str]) -> Optional[str]:
    """
    Finds the content within the last <result>...</result> tag.
    Returns the content string, or None if not found.
    """
    if not text:
        return None

    # Simple regex to find the last occurrence of <result>...</result>
    # This is a basic implementation and might need refinement for complex/nested XML scenarios
    # or if the <result> tag can have attributes.
    matches = list(re.finditer(r"<result>(.*?)</result>", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()  # Return the content of the last match
    return None


def parse_llm_tagged_output(
    llm_output: Optional[str],
    task_id: str,  # Added task_id for non-Pydantic model output
    expect_model: Optional[
        type[BaseModel]
    ] = None,  # Renamed from expect_type and made Optional
    return_on_error: Optional[Any] = None,
    verbose_errors: bool = True,
) -> Optional[Any]:
    """
    Robustly parses tagged output from an LLM string.

    Handles:
        - None or empty input.
        - Extracting content from the last <result>...</result> tag.
        - If expect_model is provided:
            - Using json_repair for robustness on the extracted string.
            - Parsing as JSON.
            - Validating against the Pydantic expect_model.
        - If expect_model is None:
            - Returns a dictionary {f"{task_id}_output": extracted_content}.

    Args:
        llm_output: The raw string output from the LLM.
        task_id: The ID of the current task, used for creating the output dictionary
                 when no Pydantic model is specified.
        expect_model: The expected Pydantic model for validation. If None,
                      the extracted string is returned in a specific dict format.
        return_on_error: Value to return if parsing/validation fails. Defaults to None.
        verbose_errors: If True, print detailed errors to console. Defaults to True.

    Returns:
        The parsed and validated Pydantic object if expect_model is provided and successful,
        a dictionary {f"{task_id}_output": extracted_content} if expect_model is None,
        otherwise returns the value specified by return_on_error.
    """
    if not llm_output:
        return return_on_error

    extracted_content: Optional[str] = None
    repaired_json_str = ""

    # 1. Extract content from <result>...</result> tag
    extracted_content = _extract_result_from_xml_tag(llm_output)

    if (
        extracted_content is None
    ):  # Changed from "not extracted_content" to handle empty string explicitly if needed
        if verbose_errors:
            logger.warning(
                f"Could not extract content from <result> tags in: {llm_output[:100]}..."
            )
        return return_on_error

    # 2. Conditional processing based on expect_model
    if expect_model:
        # Attempt repair and JSON parsing if a Pydantic model is expected
        try:
            repaired_json_str = repair_json(extracted_content)
            parsed_json = json.loads(repaired_json_str)

            # Validate against Pydantic model (which expects a dict typically)
            if not isinstance(
                parsed_json, dict
            ):  # Pydantic model_validate usually expects a dict
                if verbose_errors:
                    logger.warning(
                        f"Repaired JSON content is not a dictionary for Pydantic validation. Task ID: {task_id}. Repaired: {repaired_json_str[:100]}..."
                    )
                return return_on_error

            validated_data = expect_model.model_validate(parsed_json)
            return validated_data

        except (json.JSONDecodeError, TypeError, ValidationError, Exception) as e:
            if verbose_errors:
                error_type = type(e).__name__
                logger.error(
                    f"Error during JSON parsing or Pydantic validation for task '{task_id}': {error_type} - {e}\nExtracted content snippet: {extracted_content[:100]}...\nRepaired JSON snippet: {repaired_json_str[:100]}...\nRaw output snippet: {llm_output[:100]}..."
                )
            return return_on_error
    else:
        # No Pydantic model, return the extracted content in a specific dictionary format
        return {f"{task_id}_output": extracted_content}
