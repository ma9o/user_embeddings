import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from tqdm.asyncio import tqdm_asyncio

from user_embeddings.utils.llm.workflow_executor import _run_single_prompt

# Import utility shared or needed by these funcs
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


async def run_constraint_judge_evaluation(
    sample_workflow_results: List[
        Dict[str, Any]
    ],  # Output from run_and_parse_test_models
    model_to_evaluate: str,  # The single model being judged
    judge_model: str,
    judge_constraints_prompt_text: str,  # Renamed to clarify it's text only
) -> Dict[int, str]:  # Map sample index to raw judge response string
    """Runs the constraint judge model for each sample."""
    # ... (Copy implementation from evaluation_utils.py) ...
    judge_tasks = []
    judge_task_metadata = []  # Store sample index
    print(f"Preparing constraint judge tasks for model '{model_to_evaluate}'...")

    for i, sample_data in enumerate(sample_workflow_results):
        model_final_merged_json = sample_data.get("final_merged_json", {}).get(
            model_to_evaluate
        )

        if not isinstance(model_final_merged_json, dict) or not model_final_merged_json:
            model_final_output_fallback = sample_data.get(
                "final_parsed_outputs", {}
            ).get(model_to_evaluate)
            if (
                model_final_output_fallback is None
                or model_final_output_fallback.startswith("ERROR:")
            ):
                print(
                    f"Skipping constraint judge for sample {i}: Neither merged JSON nor valid parsed output found for model '{model_to_evaluate}'."
                )
                continue
            else:
                print(
                    f"Warning: Using fallback parsed output for sample {i}, model '{model_to_evaluate}' as merged JSON was invalid/missing."
                )
                model_output_for_judge = model_final_output_fallback
        else:
            try:
                model_output_for_judge = json.dumps(
                    model_final_merged_json, separators=(",", ":"), ensure_ascii=False
                )
            except TypeError as e:
                print(
                    f"Error serializing merged JSON for sample {i}, model '{model_to_evaluate}': {e}. Skipping."
                )
                continue

        judge_prompt = create_constraint_judge_prompt(
            constraints_prompt=judge_constraints_prompt_text,
            input_data=sample_data["input_data"],
            model_output=model_output_for_judge,
        )

        task = asyncio.create_task(_run_single_prompt(judge_model, judge_prompt))
        judge_tasks.append(task)
        judge_task_metadata.append({"sample_index": i})

    judge_responses_raw = []
    if judge_tasks:
        print(
            f"Running {len(judge_tasks)} constraint judge tasks concurrently for model '{model_to_evaluate}'..."
        )
        judge_responses_raw = await tqdm_asyncio.gather(
            *judge_tasks, desc=f"Running Constraint Judge ({judge_model})", unit="task"
        )
    else:
        print("No constraint judge tasks to run.")

    judge_response_map: Dict[int, str] = {}  # Map sample_index -> raw judge response
    for i, raw_response in enumerate(judge_responses_raw):
        meta = judge_task_metadata[i]
        sample_index = meta["sample_index"]
        judge_response_map[sample_index] = raw_response

    return judge_response_map


def aggregate_constraint_results(
    sample_workflow_results: List[
        Dict[str, Any]
    ],  # Output from run_and_parse_test_models
    judge_response_map: Dict[int, str],  # Output from run_constraint_judge_evaluation
    model_to_evaluate: str,
    effective_seed: int,
    workflow_name: str,
    judge_prompt_name: str,  # Name of the constraints prompt module
    workflow: List[Dict[str, Any]],
    available_prompts: Dict[str, Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Aggregates results for the constraint violation evaluation including versions."""
    # ... (Copy implementation from evaluation_utils.py) ...
    print("Aggregating constraint evaluation results (including versions)...")
    results_data = []
    all_task_ids_in_workflow = set(p for stage in workflow for p in stage["prompts"])

    constraints_prompt_version = available_prompts.get(judge_prompt_name, ("", "N/A"))[
        1
    ]

    for i, sample_data in enumerate(sample_workflow_results):
        input_context = sample_data.get("input_data", "ERROR: Input not found")
        input_length = len(input_context) if isinstance(input_context, str) else -1

        final_parsed_output = sample_data.get("final_parsed_outputs", {}).get(
            model_to_evaluate, "N/A"
        )
        final_merged_json = sample_data.get("final_merged_json", {}).get(
            model_to_evaluate, {}
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
            "judge_raw_output": judge_raw_response
            if judge_raw_response
            else "Judge Skipped/Failed",
            "violated_constraints": json.dumps(violated_constraints_dict)
            if violated_constraints_dict is not None
            else "ERROR: Parse Failed"
            if judge_raw_response
            else "Judge Skipped/Failed",
            "violation_count": len(violated_constraints_dict)
            if violated_constraints_dict is not None
            else -1,
            "seed": effective_seed,
            "workflow_name": workflow_name,
            "model_evaluated": model_to_evaluate,
            "judge_prompt_name": judge_prompt_name,
            "constraints_prompt_version": constraints_prompt_version,
        }

        model_outputs_all_tasks = sample_data.get("model_outputs", {}).get(
            model_to_evaluate, {}
        )
        for task_id in all_task_ids_in_workflow:
            col_name = f"output_{task_id}_{model_to_evaluate}"
            result_row[col_name] = model_outputs_all_tasks.get(task_id, "N/A")
            task_version = available_prompts.get(task_id, ("", "N/A"))[1]
            result_row[f"version_{task_id}"] = task_version

        if isinstance(final_merged_json, dict):
            try:
                result_row[f"final_merged_output_{model_to_evaluate}"] = json.dumps(
                    final_merged_json, separators=(",", ":"), ensure_ascii=False
                )
            except TypeError:
                result_row[f"final_merged_output_{model_to_evaluate}"] = (
                    "ERROR: Failed to serialize merged JSON"
                )

            for key, value in final_merged_json.items():
                col_name = f"final_{key}_{model_to_evaluate}"
                try:
                    result_row[col_name] = (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else str(value)
                    )
                except TypeError:
                    result_row[col_name] = "ERROR: Failed to serialize value"
        else:
            result_row[f"final_merged_output_{model_to_evaluate}"] = (
                "ERROR: Merged JSON not available or not a dict"
            )
            result_row[f"final_merged_json_error_{model_to_evaluate}"] = (
                "ERROR: Merged JSON not available or not a dict"
            )

        results_data.append(result_row)

    return results_data
