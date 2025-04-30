import asyncio  # Added back
import json
from typing import Any, Dict, List, Optional

import dask  # Added
from dask import delayed  # Added

# from tqdm.asyncio import tqdm_asyncio # Removed
# Import the async LLM runner
from user_embeddings.utils.llm.get_text_completion import (
    get_text_completion,  # Direct import
    initialize_openrouter_client,  # Correct path
)
from user_embeddings.utils.parsing import parse_llm_json_output

# --- Constraint Specific Helpers ---


# Step 1 (Delayed): Create the constraint judge prompt
@delayed(pure=True)
def _create_constraint_judge_prompt_delayed(
    constraints_prompt: str,
    input_data: str,
    model_output: str,  # Expects resolved string output
) -> str:
    """Delayed function to create the prompt for the constraint judge."""
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


# Step 2 (Delayed): Run the judge LLM (using the async helper)
# Internal async helper (copied from ranking_utils for now)
# TODO: Consolidate this helper if appropriate
async def _run_single_judge_llm_async(model_name: str, prompt: str) -> str:
    """Async helper to run a single judge LLM call."""
    try:
        # Ensure client is initialized here too
        _ = initialize_openrouter_client()
        result = await get_text_completion(model_name, prompt)
        return result
    except Exception as e:
        print(f"Error running judge model {model_name}: {e}")
        return f"ERROR: Judge model execution failed - {e}"


# Dask wrapper (copied from ranking_utils for now)
@delayed(pure=True)
def _run_judge_llm_dask_wrapper(judge_model: str, judge_prompt: str) -> str:
    """Dask wrapper to run the async judge LLM call using asyncio.run."""
    try:
        # Need asyncio import if not top-level
        return asyncio.run(_run_single_judge_llm_async(judge_model, judge_prompt))
    except Exception as e:
        print(f"Error in Dask wrapper for judge model {judge_model}: {e}")
        return f"ERROR: Judge Dask wrapper failed - {e}"


# Step 3 (Delayed): Parse the judge's output
@delayed(pure=True)
def _parse_constraint_judge_output_delayed(
    judge_response: str,
) -> Optional[Dict[str, str]]:
    """Delayed function to parse the constraint judge's dictionary response."""
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


# --- New Graph Building Function ---
def build_constraint_judge_graph(
    # Delayed object representing the specific model output string to judge
    model_output_delayed: dask.delayed,
    # Delayed object representing the input data string
    input_data_delayed: dask.delayed,
    judge_model: str,
    judge_constraints_prompt_text: str,  # Static text
) -> Dict[str, dask.delayed]:
    """Builds the Dask graph segment for constraint judge evaluation.

    Returns a dictionary containing delayed objects for:
        - 'judge_raw_output': Raw string output from the judge LLM.
        - 'judge_parsed_violations': Parsed dictionary of violations (or None).
    """
    # Step 1: Create the prompt (depends on resolved input_data and model_output)
    judge_prompt_delayed = _create_constraint_judge_prompt_delayed(
        constraints_prompt=judge_constraints_prompt_text,
        input_data=input_data_delayed,
        model_output=model_output_delayed,
    )

    # Step 2: Run the judge LLM (depends on the delayed prompt)
    judge_raw_output_delayed = _run_judge_llm_dask_wrapper(
        judge_model=judge_model, judge_prompt=judge_prompt_delayed
    )

    # Step 3: Parse the output (depends on the delayed raw output)
    parsed_violations_delayed = _parse_constraint_judge_output_delayed(
        judge_response=judge_raw_output_delayed
    )

    # Return a dictionary of the relevant delayed objects
    return {
        "judge_raw_output": judge_raw_output_delayed,
        "judge_parsed_violations": parsed_violations_delayed,
    }


# --- Deprecated Synchronous Function ---
def run_constraint_judge_evaluation(*args, **kwargs):
    raise DeprecationWarning(
        "run_constraint_judge_evaluation is deprecated. Use build_constraint_judge_graph and dask.compute."
    )


# --- Aggregation Function --- (Needs updates to use delayed results)
def aggregate_constraint_results(
    # Operates on *computed* results
    # Structure per sample: {'input_data': str, 'workflow_outputs': {model: {task: TaskResult}}, 'judge_results': {model: {key: computed_value}}}
    computed_results_list: List[Dict[str, Any]],
    model_to_evaluate: str,
    effective_seed: int,
    workflow_name: str,
    judge_prompt_name: str,
) -> List[Dict[str, Any]]:
    """Aggregates constraint results from the computed Dask graph output."""
    print("Aggregating constraint evaluation results...")
    results_data = []

    for i, computed_sample in enumerate(computed_results_list):
        input_context = computed_sample.get("input_data", "ERROR: Input missing")
        workflow_outputs = computed_sample.get("workflow_outputs", {})
        # Judge results are now nested per model being evaluated
        judge_results_for_model = computed_sample.get("judge_results", {}).get(
            model_to_evaluate, {}
        )

        input_length = len(input_context) if isinstance(input_context, str) else -1

        # Get the specific model output that was *intended* for judging
        # This relies on the main graph construction providing this info if needed,
        # or reconstructing it. Let's assume it's not directly available here easily.
        # We will log the judge's raw output and parsed violations.

        judge_raw_response = judge_results_for_model.get(
            "judge_raw_output", "Judge Skipped/Failed"
        )
        violated_constraints_dict = judge_results_for_model.get(
            "judge_parsed_violations"
        )

        result_row: Dict[str, Any] = {
            "input": input_context,
            "input_length": input_length,
            # "model_output_judged": judged_output_str, # Hard to get reliably here
            "judge_raw_output": judge_raw_response,
            "violated_constraints": json.dumps(violated_constraints_dict)
            if violated_constraints_dict is not None
            else (
                "ERROR: Parse Failed"
                if judge_raw_response != "Judge Skipped/Failed"
                else "Judge Skipped/Failed"
            ),
            "violation_count": len(violated_constraints_dict)
            if violated_constraints_dict is not None
            else -1,
            "seed": effective_seed,
            "workflow_name": workflow_name,
            "model_evaluated": model_to_evaluate,
            "judge_prompt_name": judge_prompt_name,
        }

        # Optionally add detailed task outputs for the evaluated model
        model_detailed_outputs = workflow_outputs.get(model_to_evaluate, {})
        # for task_id, task_result in model_detailed_outputs.items():
        #     result_row[f"raw_output_{task_id}_{model_to_evaluate}"] = task_result.get("raw_output", "N/A")

        results_data.append(result_row)

    return results_data
