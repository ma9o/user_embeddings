import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl
from pydantic import BaseModel

# Import utility
# Import shared configurations to use as defaults
from evaluation.config import (
    AVAILABLE_INPUT_FORMATTERS,
    AVAILABLE_OUTPUT_MODELS,
    AVAILABLE_PROMPTS,
)

# Import the NEW orchestrator and the single prompt runner
from user_embeddings.utils.llm.workflow_executor import (
    WorkflowStage,
    run_workflow_on_samples,
)

# --- Shared Evaluation Helpers ---


def load_and_sample_data(
    input_source_path: Path, num_samples: int, seed: Optional[int]
) -> Optional[pl.DataFrame]:
    # ...
    full_df = None
    required_column = "formatted_context"
    if input_source_path.is_file():
        print(f"Loading data from file: {input_source_path}...")
        try:
            full_df = pl.read_csv(input_source_path)
            if required_column not in full_df.columns:
                print(
                    f"Error: '{required_column}' column not found in {input_source_path}"
                )
                return None
        except Exception as e:
            print(f"Error reading CSV file {input_source_path}: {e}")
            return None
    elif input_source_path.is_dir():
        print(f"Loading data from directory: {input_source_path}...")
        glob_pattern = "test_output_*.csv"
        all_files = list(input_source_path.glob(glob_pattern))
        if not all_files:
            print(f"No files matching '{glob_pattern}' found in {input_source_path}")
            return None
        print(f"Found {len(all_files)} files matching pattern.")
        df_list = []
        for f in all_files:
            print(f"  Reading {f.name}...")
            try:
                df = pl.read_csv(f)
                if required_column not in df.columns:
                    print(
                        f"  Warning: '{required_column}' column not found in {f.name}. Skipping file."
                    )
                    continue
                df_list.append(df)
            except Exception as e:
                print(f"  Error reading file {f.name}: {e}. Skipping file.")
                continue
        if not df_list:
            print(f"No valid CSV files with '{required_column}' column could be read.")
            return None
        try:
            full_df = pl.concat(df_list, how="vertical_relaxed")
        except Exception as e:
            print(f"Error concatenating DataFrames: {e}")
            return None
    else:
        print(
            f"Error: Input source path is neither a file nor a directory: {input_source_path}"
        )
        return None
    if full_df is None or len(full_df) == 0:
        print("No data loaded after processing input source.")
        return None
    print(f"Total rows loaded: {len(full_df)}")
    if required_column not in full_df.columns:
        print(f"Error: '{required_column}' column is missing after processing files.")
        return None
    if len(full_df) < num_samples:
        print(
            f"Warning: Not enough data ({len(full_df)} rows) for {num_samples} samples. Using all available data."
        )
        sample_df = full_df
    else:
        print(f"Using seed {seed} for sampling.")
        sample_df = full_df.sample(n=num_samples, shuffle=True, seed=seed)
    print(f"Selected {len(sample_df)} rows for evaluation.")
    return sample_df


async def run_and_parse_test_models(
    sample_df: pl.DataFrame,
    models_to_test: List[str],
    workflow: List[WorkflowStage],
    available_prompts: Dict[str, Tuple[str, str]] = AVAILABLE_PROMPTS,
    available_output_models: Dict[str, type[BaseModel]] = AVAILABLE_OUTPUT_MODELS,
    available_input_formatters: Dict[
        str, Callable[[Dict[str, Any]], str]
    ] = AVAILABLE_INPUT_FORMATTERS,
) -> List[Dict[str, Any]]:
    """
    Runs test models using the updated workflow orchestrator.
    and prepares a string representation for the judge.
    """
    # Call the orchestrator from the utils module
    # It now returns validated outputs (or errors) per task
    execution_results_by_sample = await run_workflow_on_samples(
        sample_df=sample_df,
        models_to_test=models_to_test,
        workflow=workflow,
        available_prompts=available_prompts,
        available_output_models=available_output_models,
        input_column="formatted_context",
        available_input_formatters=available_input_formatters,
    )

    # --- Process results to create judge input string from final stage outputs ---
    print("Processing workflow execution results and preparing judge inputs...")
    processed_results: List[Dict[str, Any]] = []

    # Determine task IDs from the final stage
    final_stage_tasks: List[Dict[str, Any]] = workflow[-1]["tasks"]
    final_stage_task_ids: List[str] = [task["prompt"] for task in final_stage_tasks]

    for exec_result_sample in execution_results_by_sample:
        # Initialize the structure expected by downstream functions
        processed_sample_data = {
            "input_data": exec_result_sample["input_data"],
            "model_outputs": exec_result_sample[
                "model_outputs"
            ],  # Keep full {model: {validated: {}, raw: {}}}
            "final_judge_inputs": {},  # String representation for judge
            "final_validated_outputs": {},  # Dict of final validated Pydantic objects/raw strings/errors
        }

        model_execution_results = exec_result_sample.get("model_outputs", {})

        for model, execution_result in model_execution_results.items():
            # execution_result = {"validated_outputs": {...}, "raw_outputs": {...}}
            validated_outputs = execution_result.get("validated_outputs", {})

            # --- Prepare Judge Input String ---
            # Serialize the *validated* outputs of the final stage tasks
            final_outputs_for_judge = {}
            parsing_or_validation_error = False
            for task_id in final_stage_task_ids:
                output_data = validated_outputs.get(task_id)
                if output_data is None:
                    final_outputs_for_judge[task_id] = (
                        f"ERROR: Output missing for task '{task_id}'"
                    )
                    parsing_or_validation_error = True
                elif isinstance(output_data, str) and output_data.startswith("ERROR:"):
                    final_outputs_for_judge[task_id] = (
                        output_data  # Propagate error string
                    )
                    parsing_or_validation_error = True
                elif isinstance(output_data, BaseModel):
                    # Serialize Pydantic model to dict, then to JSON string for judge
                    try:
                        # Use model_dump for validated object
                        final_outputs_for_judge[task_id] = output_data.model_dump()
                    except Exception as e:
                        final_outputs_for_judge[task_id] = (
                            f"ERROR: Failed to dump Pydantic model for task '{task_id}': {e}"
                        )
                        parsing_or_validation_error = True
                else:
                    # Assume it's a raw string (task without Pydantic model) or already a dict/list
                    final_outputs_for_judge[task_id] = output_data

            # Create the single string representation for the judge
            judge_input_string: str
            if parsing_or_validation_error:
                # Option 1: Include details of errors
                judge_input_string = (
                    "ERROR: One or more final stage outputs failed validation or parsing.\nDetails:\n"
                    + json.dumps(final_outputs_for_judge, indent=2)
                )
                # Option 2: Simpler error message
                # judge_input_string = "ERROR: One or more final stage outputs failed validation or parsing."
            elif len(final_outputs_for_judge) == 1:
                # Single final task: Serialize its output (dict or raw string)
                single_output = list(final_outputs_for_judge.values())[0]
                if isinstance(
                    single_output, (dict, list)
                ):  # Already dict from Pydantic or list?
                    try:
                        judge_input_string = json.dumps(
                            single_output, indent=2, ensure_ascii=False
                        )
                    except TypeError:
                        judge_input_string = f"ERROR: Failed to serialize final output for judge: {single_output}"
                elif isinstance(single_output, str):
                    judge_input_string = single_output  # Use raw string directly
                else:
                    judge_input_string = f"ERROR: Unexpected type for single final output: {type(single_output)}"
            elif len(final_outputs_for_judge) > 1:
                # Multiple final tasks: Serialize the dict containing all final outputs
                try:
                    judge_input_string = json.dumps(
                        final_outputs_for_judge, indent=2, ensure_ascii=False
                    )
                except TypeError:
                    judge_input_string = f"ERROR: Failed to serialize multiple final outputs for judge: {final_outputs_for_judge}"
            else:  # No final stage tasks? Should be caught by validation.
                judge_input_string = "ERROR: No final stage tasks found in workflow."

            # Store the judge input string and the validated final outputs
            processed_sample_data["final_judge_inputs"][model] = judge_input_string
            processed_sample_data["final_validated_outputs"][
                model
            ] = {  # Store the actual data
                tid: validated_outputs.get(tid) for tid in final_stage_task_ids
            }

        processed_results.append(processed_sample_data)

    return processed_results


def save_results(results_df: pl.DataFrame, output_file: Path):
    """Saves evaluation results DataFrame to a CSV file."""
    # ... (implementation unchanged) ...
    print(f"Saving evaluation results to {output_file}...")
    # Ensure output directory exists before writing
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df = results_df.select(sorted(results_df.columns))
    results_df.write_csv(output_file)


# --- Ranking Specific Helpers (Moved to ranking_utils.py) ---
# create_judge_prompt
# parse_judge_output
# run_judge_evaluation
# aggregate_ranking_results
# calculate_and_print_leaderboard

# --- Constraint Specific Helpers (Moved to constraint_utils.py) ---
# create_constraint_judge_prompt
# parse_constraint_judge_output
# run_constraint_judge_evaluation
# aggregate_constraint_results

# --- Refiner Specific Helpers (Moved to refiner_utils.py) ---
# save_single_row_results_appending
