import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

# Import new judge prompt functions with standardized names
from src.user_embeddings.utils.judge_prompts.prompt_adherence_judge import (
    create_judge_prompt,
    parse_judge_output,
)
from user_embeddings.utils.llm.workflow_executor import (
    WorkflowStage,
    _run_single_prompt,
)

# --- Constraint Specific Helpers ---


async def run_constraint_judge_evaluation(
    sample_workflow_results: List[
        Dict[str, Any]
    ],  # Output from run_and_parse_test_models
    model_to_evaluate: str,  # The single model being judged
    judge_model: str,
    judge_directive_text: str,  # Renamed from judge_constraints_prompt_text
) -> Dict[int, str]:  # Map sample index to raw judge response string
    """Runs the constraint judge model for each sample using its final judge input."""
    judge_tasks = []
    judge_task_metadata = []  # Store sample index
    print(f"Preparing constraint judge tasks for model '{model_to_evaluate}'...")

    for i, sample_data in enumerate(sample_workflow_results):
        model_output_for_judge = sample_data.get("final_judge_inputs", {}).get(
            model_to_evaluate, None
        )

        if (
            not model_output_for_judge
            or not isinstance(model_output_for_judge, str)
            or model_output_for_judge.startswith("ERROR:")
        ):
            print(
                f"Skipping constraint judge for sample {i}, model '{model_to_evaluate}': Invalid or missing final judge input ('{str(model_output_for_judge)[:50]}...')."
            )
            continue
        # Use imported create_judge_prompt (formerly create_constraint_judge_prompt)
        judge_prompt = create_judge_prompt(
            constraints_prompt=judge_directive_text,
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
    workflow: List[WorkflowStage],
    available_prompts: Dict[str, Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Aggregates results for the constraint violation evaluation including versions."""
    print("Aggregating constraint evaluation results (including versions)...")
    results_data = []
    all_task_ids_in_workflow = set(
        task["prompt"] for stage in workflow for task in stage.get("tasks", [])
    )

    constraints_prompt_version = available_prompts.get(judge_prompt_name, ("", "N/A"))[
        1
    ]

    for i, sample_data in enumerate(sample_workflow_results):
        input_context = sample_data.get("input_data", "ERROR: Input not found")
        input_length = len(input_context) if isinstance(input_context, str) else -1

        final_judge_input = sample_data.get("final_judge_inputs", {}).get(
            model_to_evaluate, "N/A"
        )

        judge_raw_response = judge_response_map.get(i)
        violated_constraints_dict: Optional[Dict[str, str]] = None
        if judge_raw_response:
            # Use imported parse_judge_output (formerly parse_constraint_judge_output)
            violated_constraints_dict = parse_judge_output(judge_raw_response)

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
            "final_judge_input": final_judge_input,
            "judge_prompt_name": judge_prompt_name,
            "constraints_prompt_version": constraints_prompt_version,
        }

        model_outputs_all_tasks = sample_data.get("model_outputs", {}).get(
            model_to_evaluate, {}
        )
        model_raw_outputs = model_outputs_all_tasks.get("raw_outputs", {})
        model_validated_outputs = model_outputs_all_tasks.get("validated_outputs", {})

        for task_id in all_task_ids_in_workflow:
            # Raw output column
            raw_col_name = f"raw_output_{task_id}_{model_to_evaluate}"
            result_row[raw_col_name] = model_raw_outputs.get(task_id, "N/A")

            # Validated/Processed output column (serialized)
            validated_col_name = f"validated_output_{task_id}_{model_to_evaluate}"
            validated_data = model_validated_outputs.get(task_id)
            if isinstance(validated_data, BaseModel):
                try:
                    result_row[validated_col_name] = validated_data.model_dump_json(
                        indent=2
                    )
                except Exception:
                    result_row[validated_col_name] = (
                        "ERROR: Failed to dump Pydantic model"
                    )
            elif isinstance(validated_data, (dict, list)):
                try:
                    result_row[validated_col_name] = json.dumps(
                        validated_data, indent=2, ensure_ascii=False
                    )
                except Exception:
                    result_row[validated_col_name] = (
                        "ERROR: Failed to serialize dict/list"
                    )
            elif isinstance(validated_data, str):
                result_row[validated_col_name] = (
                    validated_data  # Store raw string or error string directly
                )
            elif validated_data is None:
                result_row[validated_col_name] = (
                    "N/A"  # Task might not have run or produced output
                )
            else:
                result_row[validated_col_name] = (
                    f"ERROR: Unexpected data type {type(validated_data)}"
                )

            task_version = available_prompts.get(task_id, ("", "N/A"))[1]
            result_row[f"version_{task_id}"] = task_version

        results_data.append(result_row)

    return results_data
