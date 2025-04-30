import asyncio  # Added back
import json
from typing import Any, Dict, List, Optional

# from tqdm.asyncio import tqdm_asyncio # Removed
# Import the async LLM runner
from user_embeddings.utils.llm.get_text_completion import (
    get_text_completion,  # Direct import
)
from user_embeddings.utils.parsing import parse_llm_json_output

# --- Constraint Specific Helpers ---


def create_constraint_judge_prompt(
    constraints_prompt: str,  # Specific prompt detailing constraints
    input_data: str,
    model_output: str,
) -> str:
    """Creates a prompt for a judge model to identify constraint violations."""
    # ... (Copy implementation from evaluation_utils.py) ...
    prompt = "You are an expert evaluator tasked with identifying violations of specific constraints in a Large Language Model (LLM) output based on a given input, and a set of constraints.\n\n"
    prompt += f"INPUT DATA GIVEN TO THE MODEL:\n---\n{input_data}\n---\n\n"
    prompt += f"MODEL OUTPUT TO EVALUATE:\n---\n{model_output}\n---\n\n"
    prompt += f"CONSTRAINTS TO CHECK:\n---\n{constraints_prompt}\n---\n\n"
    prompt += "TASK:\n1. Carefully review the MODEL OUTPUT.\n2. Compare it against the CONSTRAINTS TO CHECK, considering the INPUT DATA.\n3. Identify *all* constraints that the MODEL OUTPUT failed to meet.\n\n"
    prompt += "OUTPUT FORMAT:\nProvide your evaluation as a JSON object where each key is a unique identifier string for the violated constraint and the value is a brief string explaining the violation.\n"
    prompt += "The key MUST follow the format `CATEGORY.MainSection.SubSection` (e.g., `OUTPUT_FORMATTING.2.1`, `SEMANTIC_DISTILLATION.3.4`), referencing the corresponding section and subsection numbers from the 'CONSTRAINTS TO CHECK'. Use the ALL_CAPS category name and at most two numerical parts (e.g., `OUTPUT_FORMATTING.2` or `OUTPUT_FORMATTING.2.3` are valid, but `OUTPUT_FORMATTING.2.3.1` is NOT).\n"
    prompt += "If no constraints were violated, return an empty JSON object (`{}`).\n\n"
    prompt += "Example (Constraints violated):\n"
    prompt += (
        "```json\n"
        "{\n"
        # Using example IDs derived from all_in_one.py
        '  "OUTPUT_FORMATTING.2.3": "Explain in detail where the violation happened.",\n'
        '  "ATOMICITY.4.1": "Explain in detail where the violation happened.",\n'
        '  "SEMANTIC_DISTILLATION.3.4.2": "Explain in detail where the violation happened."\n'
        "}\n"
        "```\n\n"
    )
    prompt += "Example (No constraints violated):\n"
    prompt += "```json\n{}\n```\n\n"
    prompt += "Return ONLY the JSON object and nothing else."
    return prompt


def parse_constraint_judge_output(judge_response: str) -> Optional[Dict[str, str]]:
    """Parses the constraint judge's dictionary response using the utility function."""
    # ... (Copy implementation from evaluation_utils.py) ...
    parsed_json = parse_llm_json_output(judge_response, expect_type=dict)

    if parsed_json is None:
        print(f"Error parsing constraint judge output. Raw output:\n{judge_response}")
        return None

    # Validate the structure: Dict[str, str]
    if not isinstance(parsed_json, dict):
        # This check might be redundant if parse_llm_json_output already ensures dict
        print(f"Warning: Constraint judge output is not a dictionary: {parsed_json}")
        return None

    violations_dict = {}
    valid = True
    for key, value in parsed_json.items():
        if not isinstance(key, str) or not isinstance(value, str):
            print(
                f"Warning: Constraint judge dictionary contains non-string key or value: ({type(key)}) {key}: ({type(value)}) {value}"
            )
            valid = False
            break  # Stop validation on first error
        violations_dict[key] = value

    if not valid:
        return None  # Treat malformed dictionary as error

    return violations_dict  # Return the dictionary (possibly empty)


# Internal async helper (same as in ranking_utils)
async def _run_single_judge_llm_async(model_name: str, prompt: str) -> str:
    """Async helper to run a single judge LLM call."""
    try:
        result = await get_text_completion(model_name, prompt)
        return result
    except Exception as e:
        print(f"Error running judge model {model_name}: {e}")
        return f"ERROR: Judge model execution failed - {e}"


# Refactored to be synchronous
def run_constraint_judge_evaluation(
    # Expects the structure prepared *after* dask compute & judge prep
    # List[Dict{'input_data': str, 'judge_inputs': {model: str}, 'detailed_model_outputs': ...}]
    sample_data_list: List[Dict[str, Any]],
    model_to_evaluate: str,  # The single model being judged
    judge_model: str,
    judge_constraints_prompt_text: str,
) -> Dict[int, str]:  # Map sample index to raw judge response string
    """Runs the constraint judge model synchronously for each sample."""
    judge_prompts_with_meta = []  # Store (index, prompt) tuples
    print(f"Preparing constraint judge tasks for model '{model_to_evaluate}'...")

    for i, sample_data in enumerate(sample_data_list):
        model_output_for_judge = sample_data.get("judge_inputs", {}).get(
            model_to_evaluate
        )
        if model_output_for_judge is None or model_output_for_judge.startswith(
            "ERROR:"
        ):
            print(
                f"Skipping constraint judge for sample {i}: No valid prepared output found..."
            )
            continue

        judge_prompt = create_constraint_judge_prompt(
            constraints_prompt=judge_constraints_prompt_text,
            input_data=sample_data["input_data"],
            model_output=model_output_for_judge,
        )
        judge_prompts_with_meta.append((i, judge_prompt))

    judge_response_map: Dict[int, str] = {}
    if judge_prompts_with_meta:
        print(
            f"Running {len(judge_prompts_with_meta)} constraint judge tasks for model '{model_to_evaluate}'..."
        )
        # Run tasks one by one using asyncio.run
        # This is simpler than managing dask compute for these helpers
        for index, prompt in judge_prompts_with_meta:
            try:
                # Call the async helper using asyncio.run
                raw_response = asyncio.run(
                    _run_single_judge_llm_async(judge_model, prompt)
                )
                judge_response_map[index] = raw_response
            except Exception as e:
                print(f"Error running judge task for sample index {index}: {e}")
                judge_response_map[index] = f"ERROR: Judge execution failed - {e}"
        print("Constraint judge tasks complete.")
    else:
        print("No constraint judge tasks to run.")

    return judge_response_map


# aggregate_constraint_results needs adjustment to accept the new input format
def aggregate_constraint_results(
    # Expects the structure prepared *after* dask compute & judge prep
    # List[Dict{'input_data': str, 'judge_inputs': {model: str}, 'detailed_model_outputs': {model: {task_id: TaskResult}}}]
    processed_results_list: List[Dict[str, Any]],
    judge_response_map: Dict[int, str],
    model_to_evaluate: str,
    effective_seed: int,
    workflow_name: str,
    judge_prompt_name: str,
    # workflow: List[Dict[str, Any]], # Not strictly needed
    # available_prompts: Dict[str, Tuple[str, str]], # Not needed
) -> List[Dict[str, Any]]:
    """Aggregates results for the constraint violation evaluation."""
    print("Aggregating constraint evaluation results...")
    results_data = []
    # constraints_prompt_version = available_prompts.get(judge_prompt_name, ("", "N/A"))[1] # Removed dependency

    for i, sample_data in enumerate(processed_results_list):
        input_context = sample_data.get("input_data", "ERROR: Input not found")
        input_length = len(input_context) if isinstance(input_context, str) else -1

        # Get the prepared output string that was sent to the judge
        judged_output_str = sample_data.get("judge_inputs", {}).get(
            model_to_evaluate, "ERROR: Judged output missing"
        )

        judge_raw_response = judge_response_map.get(i)
        violated_constraints_dict: Optional[Dict[str, str]] = None
        if judge_raw_response:
            violated_constraints_dict = parse_constraint_judge_output(
                judge_raw_response
            )

        result_row: Dict[str, Any] = {
            "input": input_context,
            "input_length": input_length,
            "model_output_judged": judged_output_str,  # Store what was actually judged
            "judge_raw_output": judge_raw_response
            if judge_raw_response
            else "Judge Skipped/Failed",
            "violated_constraints": json.dumps(violated_constraints_dict)
            if violated_constraints_dict is not None
            else (
                "ERROR: Parse Failed" if judge_raw_response else "Judge Skipped/Failed"
            ),
            "violation_count": len(violated_constraints_dict)
            if violated_constraints_dict is not None
            else -1,
            "seed": effective_seed,
            "workflow_name": workflow_name,
            "model_evaluated": model_to_evaluate,
            "judge_prompt_name": judge_prompt_name,
            # "constraints_prompt_version": constraints_prompt_version, # Removed
        }

        # Add detailed task outputs if needed
        model_detailed_outputs = sample_data.get("detailed_model_outputs", {}).get(
            model_to_evaluate, {}
        )
        # Consider which specific task outputs are valuable to log here.
        # For now, let's skip adding all intermediate task outputs unless requested.
        # for task_id, task_result in model_detailed_outputs.items():
        #     col_name_raw = f"raw_output_{task_id}_{model_to_evaluate}"
        #     result_row[col_name_raw] = task_result.get("raw_output", "N/A")
        #     # Add parsed or error if needed

        results_data.append(result_row)

    return results_data
