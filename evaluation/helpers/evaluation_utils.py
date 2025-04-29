import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl
from pydantic import BaseModel, ValidationError

# Import the NEW orchestrator and the single prompt runner
from user_embeddings.utils.llm.workflow_executor import (
    FINAL_MERGED_OUTPUT_KEY,
    run_workflow_on_samples,
)

# Import utility
from user_embeddings.utils.parsing import parse_llm_json_output

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
    workflow: List[Dict[str, Any]],
    available_prompts: Dict[str, Tuple[str, str]],
    input_formatters: Dict[str, Callable[[Dict[str, str]], str]],
    available_output_models: Dict[str, type[BaseModel]],
) -> List[Dict[str, Any]]:
    """
    Runs test models using the workflow orchestrator.
    Extracts the final merged JSON output (if available from the executor)
    and prepares a string representation for the judge.
    Optionally validates individual task outputs using Pydantic.
    """
    # Call the orchestrator from the utils module
    # It now potentially returns pre-merged final JSON output
    raw_results_by_sample = await run_workflow_on_samples(
        sample_df=sample_df,
        models_to_test=models_to_test,
        workflow=workflow,
        available_prompts=available_prompts,
        input_formatters=input_formatters,
        input_column="formatted_context",
    )

    # --- Process results to extract merged JSON and create judge input string ---
    print("Processing workflow results and preparing judge inputs...")
    parsed_results: List[Dict[str, Any]] = []
    final_stage_task_ids: List[str] = workflow[-1]["prompts"]

    for sample_data in raw_results_by_sample:
        # Initialize the structure expected by downstream functions
        parsed_sample_data = {
            "input_data": sample_data["input_data"],
            "model_outputs": sample_data[
                "model_outputs"
            ],  # Keep raw outputs {model: {task: output}}
            "final_parsed_outputs": {},  # String representation for judge
            "final_merged_json": {},  # Merged JSON dictionary
        }

        model_run_outputs = sample_data.get("model_outputs", {})
        for model, workflow_result_dict in model_run_outputs.items():
            # workflow_result_dict contains {task_id: str_output, ..., _final_merged_output: dict}

            # 1. Extract the merged JSON if the executor produced it
            merged_json_output = workflow_result_dict.get(FINAL_MERGED_OUTPUT_KEY)

            if isinstance(merged_json_output, dict):
                # Successfully merged by executor
                parsed_sample_data["final_merged_json"][model] = merged_json_output
                # Create string representation for the judge from the merged dict
                try:
                    parsed_final_output_str = json.dumps(
                        merged_json_output, indent=2, ensure_ascii=False
                    )
                except TypeError:
                    parsed_final_output_str = "ERROR: Failed to serialize merged JSON"
                    # Keep the dict in final_merged_json even if serialization fails

            else:
                # No merged output from executor (e.g., single final task, merge failed, or error)
                # Fallback: Create judge input string by joining individual final task outputs
                # (This retains previous behavior for non-merge cases)
                final_outputs_for_judge = {}
                parsing_error = False
                for task_id in final_stage_task_ids:
                    output = workflow_result_dict.get(task_id)
                    if output is None:
                        final_outputs_for_judge[task_id] = "ERROR: Output missing"
                        parsing_error = True
                    elif isinstance(output, str) and output.startswith("ERROR:"):
                        final_outputs_for_judge[task_id] = output
                        parsing_error = True
                    elif isinstance(output, str):
                        final_outputs_for_judge[task_id] = output.strip()
                    else:
                        # Handle unexpected output types (e.g., if merge failed and left non-string)
                        final_outputs_for_judge[task_id] = (
                            f"ERROR: Unexpected output type for {task_id}"
                        )
                        parsing_error = True

                # Combine final outputs for the judge (simple join)
                if parsing_error:
                    parsed_final_output_str = (
                        "ERROR: One or more final stage outputs failed or missing."
                    )
                elif len(final_outputs_for_judge) == 1:
                    parsed_final_output_str = list(final_outputs_for_judge.values())[0]
                elif len(final_outputs_for_judge) > 1:
                    # Default: Simple join for judge input when multiple non-merged outputs
                    parsed_final_output_str = "\n---\n".join(
                        f"Output from {tid}:\n{out}"
                        for tid, out in sorted(
                            final_outputs_for_judge.items()
                        )  # Sort for consistency
                    )
                else:  # No valid final outputs found at all
                    parsed_final_output_str = "ERROR: No valid final outputs found."

                # Since merging didn't happen/failed in executor, merged_json is empty/error
                parsed_sample_data["final_merged_json"][model] = {
                    "error": "Automatic merge did not occur or failed in executor"
                }
                # Optional: Could add more details if the executor provided them

            # Store the string representation (either serialized JSON or fallback join)
            parsed_sample_data["final_parsed_outputs"][model] = parsed_final_output_str

            # --- Optional: Individual Task Validation (using Pydantic & Utility Parser) ---
            validated_individual_outputs = {}
            for task_id in final_stage_task_ids:
                raw_output = workflow_result_dict.get(task_id)
                output_model = available_output_models.get(task_id)

                if output_model:
                    # Use the utility parser first
                    parsed_dict = parse_llm_json_output(raw_output, expect_type=dict)

                    if parsed_dict is not None:
                        try:
                            # Validate the already parsed dict with Pydantic
                            # Note: model_validate expects a dict, not a JSON string
                            validated_data = output_model.model_validate(parsed_dict)
                            validated_individual_outputs[task_id] = (
                                validated_data.model_dump()
                            )
                            # Store if needed...
                        except ValidationError as ve:
                            print(
                                f"Warning: Pydantic validation failed for task '{task_id}', model '{model}': {ve}. Parsed dict: {parsed_dict}"
                            )
                    # else: Parsing itself failed, handled by parse_llm_json_output (prints warning/returns None)

            # End Optional Validation Block

        parsed_results.append(parsed_sample_data)

    return parsed_results


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
