import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import utility shared or needed by these funcs
from user_embeddings.utils.parsing import parse_llm_json_output

# Read the base judge prompt
_JUDGE_BASE_PROMPT_TEXT = ""
_PROJECT_ROOT = (
    Path(__file__).resolve().parents[4]
)  # src/user_embeddings/utils/judge_prompts -> user_embeddings (project root)
_JUDGE_BASE_TXT_PATH = _PROJECT_ROOT / "prompts" / "judge_base.txt"

try:
    with open(_JUDGE_BASE_TXT_PATH, "r", encoding="utf-8") as f:
        _JUDGE_BASE_PROMPT_TEXT = f.read().strip()
except FileNotFoundError:
    print(
        f"Warning: Base judge prompt file not found at {_JUDGE_BASE_TXT_PATH}. Proceeding without it."
    )
    _JUDGE_BASE_PROMPT_TEXT = "IMPORTANT GENERAL INSTRUCTION: Default to prioritizing precision over recall in your evaluations."  # Fallback
except Exception as e:
    print(
        f"Error reading base judge prompt file at {_JUDGE_BASE_TXT_PATH}: {e}. Proceeding without it."
    )
    _JUDGE_BASE_PROMPT_TEXT = "IMPORTANT GENERAL INSTRUCTION: Default to prioritizing precision over recall in your evaluations."  # Fallback


# --- Helper for Masking ---
def _create_masked_data(
    outputs: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], List[str]]:
    """Shuffles outputs and creates masked names and mappings."""
    original_items = list(outputs.items())
    random.shuffle(original_items)

    masked_outputs: Dict[str, str] = {}
    mask_to_original_map: Dict[str, str] = {}
    original_to_mask_map: Dict[str, str] = {}
    masked_model_names: List[str] = []

    for i, (original_name, output) in enumerate(original_items):
        masked_name = f"MODEL_{chr(ord('A') + i)}"
        masked_outputs[masked_name] = output
        mask_to_original_map[masked_name] = original_name
        original_to_mask_map[original_name] = masked_name
        masked_model_names.append(masked_name)

    return (
        masked_outputs,
        mask_to_original_map,
        original_to_mask_map,
        masked_model_names,
    )


# --- Ranking Specific Helpers ---


def create_judge_prompt(
    instruction_prompt: str, input_data: str, outputs: Dict[str, str]
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """Creates a blinded prompt for the ranking judge model using f-strings."""

    masked_outputs, mask_to_original_map, original_to_mask_map, masked_model_names = (
        _create_masked_data(outputs)
    )

    llm_outputs_section = ""
    for masked_name, output_text in masked_outputs.items():
        llm_outputs_section += f"\nOutput ({masked_name}):\n{output_text}\n---"

    # Ensure masked_model_names is a string representation of a list for the prompt
    # e.g., "['MODEL_A', 'MODEL_B']"
    masked_model_names_str = str(masked_model_names)

    # Prepend the base prompt text if it exists
    base_prompt_section = (
        f"{_JUDGE_BASE_PROMPT_TEXT}\n\n" if _JUDGE_BASE_PROMPT_TEXT else ""
    )

    prompt = f"""{base_prompt_section}You are an expert evaluator tasked with ranking the quality of different Large Language Model (LLM) outputs based on a given instruction and input.

INSTRUCTION PROMPT GIVEN TO MODELS:
---
{instruction_prompt}
---

INPUT DATA GIVEN TO MODELS:
---
{input_data}
---

LLM OUTPUTS TO EVALUATE (Models have been anonymized):
---{llm_outputs_section}

TASK:
1. Evaluate the outputs based *only* on how well they follow the INSTRUCTION PROMPT for the given INPUT DATA. Consider clarity, structure, adherence to format, and accuracy of the generated summary/actions based *solely* on the provided input context.
2. Identify *all* anonymized model outputs that correctly and completely fulfilled the INSTRUCTION PROMPT.

RANKING AND CORRECTNESS FORMAT:
Provide your evaluation as a JSON object containing three keys: 'ranking' (a list of anonymized model names, ordered from best to worst), 'rationale' (a brief explanation for your ranking decisions), and 'correct_models' (a list containing the anonymized names of *only* the models whose output was correct and complete. If no models were correct, provide an empty list `[]`). Use the anonymized model names provided (e.g., MODEL_A, MODEL_B). For example:
```json
{{
  "ranking": ["MODEL_A", "MODEL_C", "MODEL_B"],
  "rationale": "MODEL_A was best because..., MODEL_C was okay..., MODEL_B failed...",
  "correct_models": ["MODEL_A", "MODEL_C"]
}}
```

IMPORTANT: In your 'rationale', make sure to refer to the models using their anonymized names (e.g., MODEL_A, MODEL_B).

The available anonymized model names are: {masked_model_names_str}. Use these exact names (e.g., MODEL_A, MODEL_B) in the 'ranking' list and, if applicable, in the 'correct_models' list. Return ONLY the JSON object and nothing else.
"""
    return prompt, mask_to_original_map, original_to_mask_map


def parse_judge_output(
    judge_response: str,
) -> Tuple[Optional[List[str]], Optional[str], Optional[List[str]]]:
    """Parses the ranking judge's JSON response, expecting ranking, rationale, and a list of correct models."""
    parsed_json = parse_llm_json_output(judge_response, expect_type=dict)

    if parsed_json is None:
        print(f"Error parsing judge output. Raw output:\n{judge_response}")
        return None, None, None

    ranking = parsed_json.get("ranking")
    rationale = parsed_json.get("rationale")
    correct_models = parsed_json.get("correct_models")

    if not isinstance(ranking, list) or not all(
        isinstance(item, str) for item in ranking
    ):
        print(
            f"Warning: Judge output 'ranking' key is not a list of strings: {ranking}"
        )
        ranking = None
    if not isinstance(rationale, str):
        print(f"Warning: Judge output 'rationale' key is not a string: {rationale}")
        rationale = None
    if not isinstance(correct_models, list) or not all(
        isinstance(item, str) for item in correct_models
    ):
        print(
            f"Warning: Judge output 'correct_models' key is not a list of strings: {correct_models}"
        )
        correct_models = None

    return ranking, rationale, correct_models
