# import asyncio # No longer needed
import asyncio  # Re-add asyncio for asyncio.run
import json
from typing import Any, Dict, List, Optional, Tuple, Type, TypedDict, Union

import dask
import polars as pl
from dask import delayed
from pydantic import BaseModel, ValidationError

# from tqdm.asyncio import tqdm_asyncio # No longer needed
from user_embeddings.utils.parsing import parse_llm_json_output

# Assuming get_text_completion is available via relative import or PYTHONPATH
# It MUST be awaitable if we want to use it with dask's async scheduling
from .get_text_completion import get_text_completion, initialize_openrouter_client


# --- Type Definitions ---
class TaskDefinition(TypedDict):
    """Defines a single task within a workflow stage."""

    task_id: str  # Unique identifier for the task's output (e.g., "extract_keywords")
    prompt: str  # Name of the prompt module to use
    input_from: List[str]  # List of task_ids whose outputs are needed as input


class WorkflowStage(TypedDict):
    """Defines the structure for a single stage in a workflow."""

    stage: int
    tasks: List[TaskDefinition]  # List of tasks to run in this stage


class TaskResult(TypedDict):
    """Stores the result of a single task execution."""

    raw_output: str
    parsed_output: Optional[Union[Dict, List, str, int, float, bool]]
    error: Optional[str]


# --- Constants ---
# Key used in the results dict to store the parsed output
PARSED_OUTPUT_KEY = "parsed_output"
RAW_OUTPUT_KEY = "raw_output"
ERROR_KEY = "error"


# --- Workflow Validation (Simplified - Focuses on Task Structure) ---
def validate_workflow(
    workflow_name: str,
    workflow_definition: List[WorkflowStage],
    available_prompts: Dict[str, Tuple[str, str]],
    # available_output_models is needed for more robust validation,
    # but keeping it simpler for now to focus on structure.
) -> bool:
    """
    Validates the structure and basic dependencies of a workflow definition.

    Checks for:
        - Existence of all specified prompt modules.
        - Correct dependencies (input_from tasks must exist in previous stages).
        - Unique task_ids across the entire workflow.

    Returns:
        True if the workflow is valid, False otherwise. Prints errors to console.
    """
    is_valid = True
    defined_task_ids = set()
    all_task_ids_in_workflow = set()

    # First pass: check prompt existence and collect all task_ids
    for stage_def in workflow_definition:
        stage_num = stage_def["stage"]
        if "tasks" not in stage_def or not isinstance(stage_def["tasks"], list):
            print(
                f"Error: Stage {stage_num} in workflow '{workflow_name}' is missing a valid 'tasks' list."
            )
            return False  # Cannot proceed without tasks list

        for task_def in stage_def["tasks"]:
            task_id = task_def.get("task_id")
            prompt_name = task_def.get("prompt")

            if not task_id:
                print(
                    f"Error: Task in stage {stage_num} of workflow '{workflow_name}' is missing 'task_id'."
                )
                is_valid = False
                continue  # Skip further checks for this invalid task

            if task_id in all_task_ids_in_workflow:
                print(
                    f"Error: Duplicate task_id '{task_id}' found in workflow '{workflow_name}'. Task IDs must be unique."
                )
                is_valid = False
            all_task_ids_in_workflow.add(task_id)

            if not prompt_name:
                print(
                    f"Error: Task '{task_id}' in stage {stage_num} of workflow '{workflow_name}' is missing 'prompt'."
                )
                is_valid = False
            elif prompt_name not in available_prompts:
                print(
                    f"Error: Prompt module '{prompt_name}' for task '{task_id}' in stage {stage_num} "
                    f"of workflow '{workflow_name}' not found. "
                    f"Available prompts: {list(available_prompts.keys())}"
                )
                is_valid = False

            if "input_from" not in task_def or not isinstance(
                task_def["input_from"], list
            ):
                print(
                    f"Error: Task '{task_id}' in stage {stage_num} of workflow '{workflow_name}' is missing a valid 'input_from' list (can be empty [])."
                )
                is_valid = False

    # Second pass: check dependencies now that we have all task IDs
    sorted_workflow = sorted(workflow_definition, key=lambda x: x["stage"])
    for stage_def in sorted_workflow:
        stage_num = stage_def["stage"]
        current_stage_tasks = set()
        for task_def in stage_def["tasks"]:
            task_id = task_def["task_id"]
            input_from_tasks = task_def.get("input_from", [])  # Default to empty list

            # Validate input_from dependencies against previously defined tasks
            for required_input_task in input_from_tasks:
                if required_input_task not in defined_task_ids:
                    print(
                        f"Error: Task '{task_id}' in stage {stage_num} requires input from '{required_input_task}', "
                        f"but it's not defined in any previous stage of workflow '{workflow_name}'. "
                        f"Available previous tasks: {list(defined_task_ids)}"
                    )
                    is_valid = False
            current_stage_tasks.add(task_id)

        # Add outputs of this stage to the set for next stage validation
        defined_task_ids.update(current_stage_tasks)

    return is_valid


# --- Dask-Compatible Core Execution Logic ---


# This function MUST remain async if get_text_completion is async
# Dask's distributed scheduler can handle invoking async functions.
async def _run_llm_async(model_name: str, prompt_text: str, input_context: str) -> str:
    """Async helper to format prompt and run a single model call."""
    # Ensure client is initialized in the context where this runs (Dask worker)
    _ = initialize_openrouter_client()  # Call initializer here
    # Basic formatting: Instruction + Input Context
    model_prompt = f"{prompt_text}\n\nINPUT DATA:\n---\n{input_context}\n---"
    result = await get_text_completion(model_name, model_prompt)
    return result


# Wrapper for Dask: Handles potential input errors and calls the async LLM function
# This wrapper itself is synchronous from Dask's perspective.
@delayed(pure=True)
def _run_single_prompt_dask_wrapper(
    prepared_input: Union[
        str, Tuple[None, str]
    ],  # (serialized_context, None) or (None, error_msg)
    model_name: str,
    prompt_text: str,
    task_id: str,  # For logging
) -> str:
    """Dask-delayed wrapper. Checks prepared input and calls async LLM execution."""
    serialized_context, error_msg = prepared_input
    if error_msg:
        print(f"Task '{task_id}' skipping LLM call due to input error: {error_msg}")
        return f"ERROR: Input preparation failed - {error_msg}"

    try:
        # Use asyncio.run() to execute the async function from this sync context.
        # This creates a new event loop for each call, which is usually fine for
        # isolated async operations called from sync code.
        result = asyncio.run(
            _run_llm_async(model_name, prompt_text, serialized_context)
        )
        # result = dask.local.get_async(_run_llm_async(model_name, prompt_text, serialized_context)) # Incorrect usage
        return result
    except Exception as e:
        print(f"Error running model {model_name} for task '{task_id}': {e}")
        return f"ERROR: Model execution failed - {e}"


@delayed(pure=True)
def _prepare_task_input_dask(
    task_id: str,
    input_from_tasks: List[str],
    # Pass resolved results, NOT futures here. Dask handles dependency.
    results_so_far: Dict[str, TaskResult],
    available_output_models: Dict[str, Type[BaseModel]],
    initial_input: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Delayed function to gather, parse, and merge inputs for a task.
    This runs *after* its dependencies (previous TaskResults) are computed.
    """
    if not input_from_tasks:
        if initial_input is not None:
            return json.dumps({"initial_input": initial_input}), None
        else:
            return json.dumps({}), None

    merged_inputs = {}
    for required_task_id in input_from_tasks:
        # results_so_far contains the *actual* computed results of dependencies
        if required_task_id not in results_so_far:
            # This case implies a structural error in the graph or workflow def
            return (
                None,
                f"FATAL: Missing expected dependency result for '{required_task_id}'",
            )

        prev_result = results_so_far[required_task_id]
        if prev_result.get(ERROR_KEY):
            return (
                None,
                f"Dependency '{required_task_id}' failed: {prev_result[ERROR_KEY]}",
            )

        input_data_to_use = prev_result.get(PARSED_OUTPUT_KEY)
        input_model = available_output_models.get(required_task_id)

        if input_model and input_data_to_use is None and not prev_result.get(ERROR_KEY):
            # If an input model was expected, but parsing failed upstream (and wasn't fatal error)
            return (
                None,
                f"Dependency '{required_task_id}' (model '{input_model.__name__}') has no parsed output.",
            )

        if input_data_to_use is None:
            input_data_to_use = prev_result[RAW_OUTPUT_KEY]  # Fallback to raw

        merged_inputs[required_task_id] = input_data_to_use

    try:
        serialized_context = json.dumps(merged_inputs, indent=2, ensure_ascii=False)
        return serialized_context, None
    except TypeError as e:
        return None, f"Failed to serialize merged inputs to JSON: {e}"


@delayed(pure=True)
def _process_output_dask(
    raw_llm_output: str,  # Result from _run_single_prompt_dask_wrapper
    output_model: Optional[Type[BaseModel]],
    task_id: str,  # For logging
    model_name: str,  # For logging
) -> TaskResult:
    """
    Delayed function to parse, validate, and structure the final task result.
    Runs *after* the LLM call for the task is complete.
    """
    parsed_output = None
    error_message = None

    if raw_llm_output.startswith("ERROR:"):
        error_message = raw_llm_output
    elif output_model:
        parsed_dict_or_list = parse_llm_json_output(
            raw_llm_output, expect_type=(dict, list)
        )
        if parsed_dict_or_list is not None:
            try:
                validated_data = output_model.model_validate(parsed_dict_or_list)
                parsed_output = validated_data.model_dump(mode="json")
            except ValidationError as ve:
                error_message = f"Pydantic validation failed: {ve}"
                print(
                    f"Warning: Pydantic validation failed for task '{task_id}', model '{model_name}': {ve}. Raw output: {raw_llm_output[:100]}..."
                )
            except Exception as e:
                error_message = (
                    f"Output processing/dumping failed after validation: {e}"
                )
                print(
                    f"Error: Output processing failed for task '{task_id}' after validation: {e}"
                )
        else:
            # JSON parsing failed
            error_message = "JSON parsing failed."
            # Warning printed by parse_llm_json_output

    return {
        RAW_OUTPUT_KEY: raw_llm_output,
        PARSED_OUTPUT_KEY: parsed_output,
        ERROR_KEY: error_message,
    }


def build_dask_workflow_graph(
    model_name: str,
    initial_input: str,
    workflow: List[WorkflowStage],
    available_prompts: Dict[str, Tuple[str, str]],
    available_output_models: Dict[str, Type[BaseModel]],
) -> Dict[str, dask.delayed]:
    """
    Builds the Dask computation graph for a single workflow execution.
    Does NOT execute the workflow, only defines the delayed tasks.

    Returns:
        A dictionary mapping task_id to its corresponding Dask delayed object
        representing the final TaskResult for that task.
    """
    # Stores the delayed object representing the TaskResult for each task_id
    delayed_task_results: Dict[str, dask.delayed] = {}
    # Stores delayed objects representing the *raw inputs* needed for _prepare_task_input_dask
    # (This seems overly complex, let's simplify)
    # We need to pass the *actual* delayed TaskResult objects as dependencies.

    # Sort by stage to process tasks in a somewhat logical order for graph construction
    sorted_workflow = sorted(workflow, key=lambda x: x["stage"])

    for stage_def in sorted_workflow:
        # Ensure tasks exist before proceeding
        if "tasks" not in stage_def or not isinstance(stage_def["tasks"], list):
            print(f"Warning: Skipping invalid stage definition: {stage_def}")
            continue

        for task_def in stage_def["tasks"]:
            task_id = task_def["task_id"]
            prompt_name = task_def["prompt"]
            input_from = task_def["input_from"]

            if task_id in delayed_task_results:
                # Avoid redefining task if already processed (e.g., due to duplicate task_id)
                continue

            # 1. Get Delayed objects for all direct dependencies (previous TaskResults)
            dependency_futures = {
                dep_id: delayed_task_results[dep_id]
                for dep_id in input_from
                if dep_id in delayed_task_results
            }
            # Check if all dependencies were found (validation should catch this, but double-check)
            if len(dependency_futures) != len(input_from):
                missing_deps = set(input_from) - set(dependency_futures.keys())
                # Create a dummy delayed object that immediately returns an error
                err_msg = f"FATAL: Missing dependencies {list(missing_deps)} for task '{task_id}' during graph construction."
                print(err_msg)
                prepared_input_delayed = delayed(
                    (None, err_msg)
                )  # Directly create error tuple
            else:
                # 2. Create Delayed object for preparing input
                # Pass the dictionary of dependency futures. Dask passes the *results*.
                prepared_input_delayed = _prepare_task_input_dask(
                    task_id=task_id,
                    input_from_tasks=input_from,
                    results_so_far=dependency_futures,  # Dask resolves these futures before calling
                    available_output_models=available_output_models,
                    initial_input=initial_input if not input_from else None,
                )

            # Get prompt text, handle missing prompt error
            prompt_info = available_prompts.get(prompt_name)
            if not prompt_info:
                err_msg = f"ERROR: Prompt module '{prompt_name}' not found for task '{task_id}'."
                print(err_msg)
                # Create dummy delayed objects representing the error flow
                raw_output_delayed = delayed(err_msg)
                output_model = None  # No model if prompt missing
            else:
                prompt_text, _ = prompt_info
                output_model = available_output_models.get(task_id)

                # 3. Create Delayed object for running the LLM
                raw_output_delayed = _run_single_prompt_dask_wrapper(
                    prepared_input=prepared_input_delayed,  # Pass the delayed input prep result
                    model_name=model_name,
                    prompt_text=prompt_text,
                    task_id=task_id,
                )

            # 4. Create Delayed object for processing the output
            final_result_delayed = _process_output_dask(
                raw_llm_output=raw_output_delayed,  # Pass the delayed raw output
                output_model=output_model,
                task_id=task_id,
                model_name=model_name,
            )

            # 5. Store the final delayed object for this task_id
            delayed_task_results[task_id] = final_result_delayed

    return delayed_task_results


# Renamed: orchestrates graph building over samples/models
def build_full_dask_graph(
    sample_df: pl.DataFrame,
    models_to_test: List[str],
    workflow: List[WorkflowStage],
    available_prompts: Dict[str, Tuple[str, str]],
    available_output_models: Dict[str, Type[BaseModel]],
    input_column: str = "formatted_context",
) -> List[Dict[str, Any]]:
    """
    Builds the complete Dask graph for running a workflow across multiple samples and models.
    Returns a structure containing Dask delayed objects, NOT computed results.

    Args:
        sample_df: Polars DataFrame with input data.
        models_to_test: List of model names to run.
        workflow: The workflow definition (list of stages with tasks).
        available_prompts: Map of prompt names to (prompt text, version).
        available_output_models: Map of task_ids to their Pydantic output model.
        input_column: The column name in sample_df containing the initial input.

    Returns:
        A list of dictionaries, one per sample. Each dictionary contains:
            'input_data': The initial input string for the sample.
            'model_outputs': A dict mapping model_name -> delayed_results_dict, where
                             delayed_results_dict maps task_id: dask.delayed[TaskResult].
    """
    results_structure: List[Dict[str, Any]] = []
    print("Building Dask computation graph for all samples and models...")

    if input_column not in sample_df.columns:
        raise ValueError(f"Input column '{input_column}' not found in DataFrame.")

    # Validate the workflow definition once before starting
    if not validate_workflow("Main Workflow", workflow, available_prompts):
        raise ValueError("Workflow validation failed. Please check errors above.")

    for i, row in enumerate(sample_df.iter_rows(named=True)):
        initial_input_data = row[input_column]
        sample_output = {"input_data": initial_input_data, "model_outputs": {}}
        for model in models_to_test:
            # Build the graph for this specific sample and model
            delayed_results_dict: Dict[str, dask.delayed] = build_dask_workflow_graph(
                model_name=model,
                initial_input=initial_input_data,
                workflow=workflow,
                available_prompts=available_prompts,
                available_output_models=available_output_models,
            )
            sample_output["model_outputs"][model] = delayed_results_dict

        results_structure.append(sample_output)

    print(
        f"Dask graph built for {len(sample_df)} samples and {len(models_to_test)} models."
    )
    return results_structure


# Note: The caller (e.g., prompt_refinement.py) will now be responsible for:
# 1. Initializing a Dask client (e.g., Client())
# 2. Calling build_full_dask_graph to get the structure of delayed objects.
# 3. Calling client.compute(results_structure) or dask.compute(results_structure)
#    to trigger execution and get the actual results back in the same structure.
