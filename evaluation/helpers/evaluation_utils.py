import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from pydantic import BaseModel

# Import the refactored orchestrator and types
from user_embeddings.utils.llm.workflow_executor import (
    ERROR_KEY,
    PARSED_OUTPUT_KEY,
    RAW_OUTPUT_KEY,
    TaskResult,  # Import new type
    WorkflowStage,  # Import new type
    # run_workflow_on_samples, # Removed
)

# Import utility
# from user_embeddings.utils.parsing import parse_llm_json_output

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


# This function is likely NO LONGER NEEDED in the Dask flow,
# as the main scripts now compute the graph and prepare judge inputs directly.
# Keep it for now but comment out its body and mark as deprecated/to be removed.
async def run_and_prepare_judge_inputs(
    sample_df: pl.DataFrame,
    models_to_test: List[str],
    workflow: List[WorkflowStage],
    available_prompts: Dict[str, Tuple[str, str]],
    available_output_models: Dict[str, type[BaseModel]],
    processed_results_override: Optional[
        List[Dict[str, Any]]
    ] = None,  # Added for potential reuse
) -> List[Dict[str, Any]]:
    """
    DEPRECATED in Dask flow. Prepares judge input from computed results.
    Original: Runs test models using the refactored workflow executor and prepares
    the final output string representation needed for the judge evaluation.
    """
    print("WARNING: run_and_prepare_judge_inputs is deprecated in Dask flow.")

    if processed_results_override is None:
        # This path should not be taken in the Dask flow.
        print("ERROR: Dask flow should provide pre-computed results.")
        # Placeholder: Original call path (now broken due to async/dask changes)
        # processed_results_by_sample = await run_workflow_on_samples(
        #     sample_df=sample_df,
        #     models_to_test=models_to_test,
        #     workflow=workflow,
        #     available_prompts=available_prompts,
        #     available_output_models=available_output_models,
        #     input_column="formatted_context",
        # )
        return []  # Return empty list to avoid downstream errors
    else:
        # Use the pre-computed results passed in
        processed_results_by_sample = processed_results_override

    print("Preparing judge inputs from processed workflow results...")
    judge_ready_results: List[Dict[str, Any]] = []

    if not workflow:
        print("Warning: Empty workflow definition provided.")
        return []

    final_stage_num = max(stage["stage"] for stage in workflow)
    final_task_ids: List[str] = []
    for stage in workflow:
        if stage["stage"] == final_stage_num:
            final_task_ids.extend([task["task_id"] for task in stage["tasks"]])

    if not final_task_ids:
        print(
            f"Warning: Could not identify any tasks in the final stage ({final_stage_num})."
        )

    for sample_result_bundle in processed_results_by_sample:
        judge_ready_sample_data = {
            "input_data": sample_result_bundle["input_data"],
            "detailed_model_outputs": sample_result_bundle["model_outputs"],
            "judge_inputs": {},
        }

        model_task_results = sample_result_bundle.get("model_outputs", {})
        for model, task_results_dict in model_task_results.items():
            final_outputs_for_judge = {}
            any_error_in_final = False

            for task_id in final_task_ids:
                result: Optional[TaskResult] = task_results_dict.get(task_id)
                if result is None:
                    final_outputs_for_judge[task_id] = (
                        f"ERROR: Result missing for final task '{task_id}'"
                    )
                    any_error_in_final = True
                elif result.get(ERROR_KEY):
                    final_outputs_for_judge[task_id] = (
                        f"ERROR in '{task_id}': {result[ERROR_KEY]}"
                    )
                    any_error_in_final = True
                elif result.get(PARSED_OUTPUT_KEY) is not None:
                    try:
                        final_outputs_for_judge[task_id] = json.dumps(
                            result[PARSED_OUTPUT_KEY], indent=2, ensure_ascii=False
                        )
                    except TypeError:
                        final_outputs_for_judge[task_id] = (
                            f"ERROR: Failed to serialize parsed output for '{task_id}'"
                        )
                        any_error_in_final = True
                else:
                    final_outputs_for_judge[task_id] = result[RAW_OUTPUT_KEY].strip()

            judge_input_string: str
            if any_error_in_final:
                judge_input_string = (
                    "ERROR: One or more final tasks failed or produced errors.\nDetails:\n"
                    + "\n".join(
                        [
                            f"- {tid}: {out[:200]}..."
                            if len(out) > 200
                            else f"- {tid}: {out}"
                            for tid, out in sorted(final_outputs_for_judge.items())
                        ]
                    )
                )
            elif len(final_outputs_for_judge) == 1:
                judge_input_string = list(final_outputs_for_judge.values())[0]
            elif len(final_outputs_for_judge) > 1:
                judge_input_string = "\n---\n".join(
                    f"Output from {tid}:\n{out}"
                    for tid, out in sorted(final_outputs_for_judge.items())
                )
            else:
                judge_input_string = (
                    "ERROR: No valid final task outputs could be prepared."
                )

            judge_ready_sample_data["judge_inputs"][model] = judge_input_string

        judge_ready_results.append(judge_ready_sample_data)

    return judge_ready_results


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
