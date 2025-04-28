import asyncio
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl
from pydantic import BaseModel, ValidationError
from tqdm.asyncio import tqdm_asyncio

# Import the NEW orchestrator and the single prompt runner
from user_embeddings.utils.llm.workflow_executor import (
    FINAL_MERGED_OUTPUT_KEY,
    _run_single_prompt,
    run_workflow_on_samples,
)

# Import utility
from user_embeddings.utils.parsing import parse_llm_json_output


# ... (create_judge_prompt, parse_judge_output, load_and_sample_data are unchanged) ...
def create_judge_prompt(
    instruction_prompt: str, input_data: str, outputs: Dict[str, str]
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    # ...
    original_items = list(outputs.items())
    random.shuffle(original_items)
    masked_outputs = {}
    mask_to_original_map = {}
    original_to_mask_map = {}
    masked_model_names = []
    for i, (original_name, output) in enumerate(original_items):
        masked_name = f"MODEL_{chr(ord('A') + i)}"
        masked_outputs[masked_name] = output
        mask_to_original_map[masked_name] = original_name
        original_to_mask_map[original_name] = masked_name
        masked_model_names.append(masked_name)
    prompt = "You are an expert evaluator tasked with ranking the quality of different Large Language Model (LLM) outputs based on a given instruction and input.\n\n"
    prompt += f"INSTRUCTION PROMPT GIVEN TO MODELS:\n---\n{instruction_prompt}\n---\n\n"
    prompt += f"INPUT DATA GIVEN TO MODELS:\n---\n{input_data}\n---\n\n"
    prompt += 'LLM OUTPUTS TO EVALUATE (Models have been anonymized):\n---"'
    for masked_name, output in masked_outputs.items():
        prompt += f"\nOutput ({masked_name}):\n{output}\n---"
    prompt += "\n\nTASK:\n1. Evaluate the outputs based *only* on how well they follow the INSTRUCTION PROMPT for the given INPUT DATA. Consider clarity, structure, adherence to format, and accuracy of the generated summary/actions based *solely* on the provided input context.\n"
    prompt += "2. Determine if *at least one* of the provided outputs correctly and completely fulfilled the INSTRUCTION PROMPT.\n\n"
    prompt += "RANKING AND CORRECTNESS FORMAT:\nProvide your evaluation as a JSON object containing three keys: 'ranking' (a list of anonymized model names, ordered from best to worst), 'rationale' (a brief explanation for your ranking decisions), and 'any_correct' (a boolean value - `true` if at least one model was correct, `false` otherwise). Use the anonymized model names provided (e.g., Model A, Model B). For example:\n"
    prompt += (
        "```json\n"
        "{\n"
        '  "ranking": ["MODEL_A", "MODEL_C", "MODEL_B"],\n'
        '  "rationale": "MODEL_A was best because..., MODEL_C was okay..., MODEL_B failed...",\n'
        '  "any_correct": true\n'
        "}\n"
        "```\n"
        "\nIMPORTANT: In your 'rationale', make sure to refer to the models using their anonymized names (e.g., MODEL_A, MODEL_B).\n"
    )
    prompt += f"The available anonymized model names are: {masked_model_names}. Use these exact names (e.g., MODEL_A, MODEL_B) in your response. Return ONLY the JSON object and nothing else."
    return prompt, mask_to_original_map, original_to_mask_map


def parse_judge_output(
    judge_response: str,
) -> Tuple[Optional[List[str]], Optional[str], Optional[bool]]:
    """Parses the ranking judge's JSON response using the utility function."""
    parsed_json = parse_llm_json_output(judge_response, expect_type=dict)

    if parsed_json is None:
        print(f"Error parsing judge output. Raw output:\n{judge_response}")
        return None, None, None

    # Extract fields with type checking
    ranking = parsed_json.get("ranking")
    rationale = parsed_json.get("rationale")
    any_correct = parsed_json.get("any_correct")

    # Validate types
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
    if not isinstance(any_correct, bool):
        print(
            f"Warning: Judge output 'any_correct' key is not a boolean: {any_correct}"
        )
        any_correct = None

    return ranking, rationale, any_correct


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
    available_prompts: Dict[str, str],
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


# run_judge_evaluation uses _run_single_prompt, internal logic unchanged
async def run_judge_evaluation(
    sample_workflow_results: List[Dict[str, Any]],
    judge_model: str,
    judge_instruction_prompt: str,
) -> Dict[int, Tuple[str, Dict[str, str], Dict[str, str]]]:
    # ... (implementation remains the same as previous version) ...
    judge_tasks = []
    judge_task_metadata = []
    print("Preparing judge tasks based on final workflow outputs...")

    for i, sample_data in enumerate(sample_workflow_results):
        final_outputs_for_sample = sample_data.get("final_parsed_outputs", {})
        valid_outputs_for_judge = {
            model: output
            for model, output in final_outputs_for_sample.items()
            if not output.startswith("ERROR:")
        }

        if len(valid_outputs_for_judge) > 1:
            judge_prompt, mask_map, original_map = create_judge_prompt(
                judge_instruction_prompt,
                sample_data["input_data"],
                valid_outputs_for_judge,
            )
            task = asyncio.create_task(
                _run_single_prompt(judge_model, judge_prompt)
            )  # Uses helper
            judge_tasks.append(task)
            judge_task_metadata.append(
                {"sample_index": i, "mask_map": mask_map, "original_map": original_map}
            )
        else:
            print(
                f"Skipping judge task for sample {i} due to insufficient valid final outputs ({len(valid_outputs_for_judge)} found)."
            )

    judge_responses_raw = []
    if judge_tasks:
        print(f"Running {len(judge_tasks)} judge tasks concurrently...")
        judge_responses_raw = await tqdm_asyncio.gather(
            *judge_tasks, desc="Running Judge Models", unit="task"
        )
    else:
        print("No judge tasks to run.")

    judge_response_map: Dict[int, Tuple[str, Dict[str, str], Dict[str, str]]] = {}
    for i, raw_response in enumerate(judge_responses_raw):
        meta = judge_task_metadata[i]
        sample_index = meta["sample_index"]
        mask_map = meta["mask_map"]
        original_map = meta["original_map"]
        judge_response_map[sample_index] = (raw_response, mask_map, original_map)

    return judge_response_map


def aggregate_ranking_results(
    sample_workflow_results: List[Dict[str, Any]],
    judge_response_map: Dict[int, Tuple[str, Dict[str, str], Dict[str, str]]],
    models_to_test: List[str],
    effective_seed: int,
    workflow_name: str,
    judge_prompt_name: str,
    workflow: List[Dict[str, Any]],
    debug: bool = False,  # Add debug flag
) -> List[Dict[str, Any]]:
    """Aggregates results using prompt module names as task IDs."""
    print("Processing judge results and aggregating final data...")
    results_data = []
    # final_stage_task_ids are the prompt module names from the last stage
    final_stage_task_ids: List[str] = workflow[-1]["prompts"]

    for i, sample_data in enumerate(sample_workflow_results):
        # ... (Judge data processing, ranking translation, input context, rationale unmasking - unchanged) ...
        judge_data = judge_response_map.get(i)
        ranking_masked, rationale, any_correct = (None, None, None)
        mask_to_original_map = None
        original_to_mask_map = None
        judge_raw_response = None
        if judge_data:
            judge_raw_response, mask_to_original_map, original_to_mask_map = judge_data
            ranking_masked, rationale, any_correct = parse_judge_output(
                judge_raw_response
            )
        else:
            print(
                f"No judge response data found for sample {i}, likely skipped or failed."
            )
        ranking_original = None
        expected_models_in_judge_input = set()
        if mask_to_original_map:
            expected_models_in_judge_input = set(mask_to_original_map.values())
            if ranking_masked:
                try:
                    ranking_original = [
                        mask_to_original_map[masked]
                        for masked in ranking_masked
                        if masked in mask_to_original_map
                    ]
                    if len(ranking_original) != len(expected_models_in_judge_input):
                        print(f"Warning: Judge ranking length mismatch for sample {i}.")
                except KeyError as e:
                    print(f"Error translating ranking for sample {i}: {e}")
                    ranking_original = None
        input_context = sample_data.get("input_data", "ERROR: Input not found")
        input_length = len(input_context) if isinstance(input_context, str) else -1
        unmasked_rationale = rationale
        if rationale and original_to_mask_map:
            # Add debug print here
            if debug:
                print(
                    f"DEBUG: original_to_mask_map for sample {i}: {original_to_mask_map}"
                )

            # Sort map items by length of the masked name (value) in descending order
            sorted_map_items = sorted(
                original_to_mask_map.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
            for original_name, masked_name in sorted_map_items:
                # Use simple string replace
                new_rationale = unmasked_rationale.replace(masked_name, original_name)

                unmasked_rationale = (
                    new_rationale  # Update rationale for next iteration
                )
        else:
            # Handle case where rationale or map is missing, maybe print if debugging
            if debug and not original_to_mask_map and judge_data:
                print(
                    f"DEBUG: original_to_mask_map is missing for sample {i}, although judge_data exists."
                )
            if debug and not rationale and judge_data:
                print(
                    f"DEBUG: Rationale is missing for sample {i}, although judge_data exists."
                )

        result_row: Dict[str, Any] = {
            "judge_raw_output": judge_raw_response
            if judge_raw_response
            else (
                "ERROR: Judge response not parsed"
                if judge_data
                else "Judge Skipped/Failed"
            ),
            "input": input_context,
            "input_length": input_length,
            "judge_rationale": unmasked_rationale
            if unmasked_rationale
            else (
                "ERROR: Rationale not parsed" if judge_data else "Judge Skipped/Failed"
            ),
            "judge_any_correct": any_correct if any_correct is not None else "ERROR",
            "seed": effective_seed,
            "workflow_name": workflow_name,
            "judge_prompt_name": judge_prompt_name,
        }

        # Calculate ranks (logic unchanged)
        rank_map = {model: -1 for model in models_to_test}
        if ranking_original:
            for rank, model in enumerate(ranking_original):
                if model in rank_map:
                    rank_map[model] = rank + 1

        # Store all intermediate & final outputs per model
        model_outputs_all_tasks = sample_data.get(
            "model_outputs", {}
        )  # {model: {prompt_name: output}}
        final_parsed_outputs = sample_data.get(
            "final_parsed_outputs", {}
        )  # {model: final_str}

        # Get the merged JSON output produced by run_and_parse_test_models
        final_merged_json = sample_data.get(
            "final_merged_json",
            {},  # {model: {merged_dict}}
        )

        for model in models_to_test:
            result_row[f"final_parsed_output_{model}"] = final_parsed_outputs.get(
                model, "N/A"
            )
            result_row[f"rank_{model}"] = rank_map.get(model, -1)

            # Store individual task outputs (raw), using prompt_module_name in column name
            model_task_results = model_outputs_all_tasks.get(model, {})
            # Iterate through workflow definition to get all possible task IDs (prompt names)
            all_task_ids_in_workflow = set(
                p for stage in workflow for p in stage["prompts"]
            )  # Get all prompt names
            for task_id in all_task_ids_in_workflow:  # task_id is prompt_module_name
                col_name = f"output_{task_id}_{model}"  # e.g., output_separation_gemma
                result_row[col_name] = model_task_results.get(task_id, "N/A")

            # Extract specific keys (like 'intents', 'koa') from the merged final JSON
            merged_dict_for_model = final_merged_json.get(model, {})
            if isinstance(merged_dict_for_model, dict):
                # Iterate directly over the keys present in the merged dictionary
                for key, value in merged_dict_for_model.items():
                    col_name = f"final_{key}_{model}"  # e.g., final_intents_gemma, final_koa_gemma
                    if isinstance(value, list):
                        # Store lists as JSON strings
                        result_row[col_name] = json.dumps(value)
                    else:
                        # Store other types as is (or handle as errors if needed)
                        # If you expect only lists, you could mark this as an error
                        result_row[col_name] = str(
                            value
                        )  # Store string representation for safety
            else:
                # Handle cases where final_merged_json wasn't populated correctly (e.g., parsing errors)
                # Add a generic error column if the merged dict itself is invalid
                result_row[f"final_merged_json_error_{model}"] = (
                    "ERROR: Merged JSON not available or not a dict"
                )

        results_data.append(result_row)

    return results_data


# save_results remains the same
def save_results(results_df: pl.DataFrame, output_file: Path):
    # ... (implementation unchanged) ...
    print(f"Saving evaluation results to {output_file}...")
    results_df = results_df.select(sorted(results_df.columns))
    results_df.write_csv(output_file)


# calculate_and_print_leaderboard remains the same
def calculate_and_print_leaderboard(
    results_df: pl.DataFrame, models_to_test: List[str]
):
    """Calculates and prints the final leaderboard based on average ranks, correctness, and input length bins."""
    # ... (Overall leaderboard logic - unchanged) ...
    # ... (Correctness logic - unchanged) ...

    # --- Dynamically Calculate and Print Leaderboards per Input Length Bin ---
    print("\n--- Leaderboards by Dynamic Input Length (Terciles) ---")

    # Check column validity
    if (
        "input_length" not in results_df.columns
        or results_df["input_length"].is_null().all()
        or results_df["input_length"].is_not_null().sum() == 0
        or len(results_df.drop_nulls("input_length")) < 3
    ):
        print(
            "Could not calculate dynamic bins: 'input_length' column missing, empty, or too few non-null values."
        )
        return

    # --- Proceed with quantile calculation if check passes ---
    try:
        input_lengths = results_df["input_length"].drop_nulls()

        q1_val = input_lengths.quantile(0.333)
        q2_val = input_lengths.quantile(0.667)
        if q1_val is None or q2_val is None:
            raise ValueError("Quantile calculation returned None")

        q1 = int(q1_val)
        q2 = int(q2_val)
        min_len_val = input_lengths.min()
        max_len_val = input_lengths.max()

        if min_len_val is None or max_len_val is None:
            raise ValueError("Min/Max calculation returned None")

    except Exception as e:
        print(f"Error calculating quantiles for input length bins: {e}")
        return

    # Define dynamic bins based on terciles
    bins = {
        f"Shortest ~33% (<= {q1})": (pl.col("input_length") <= q1),
        f"Middle ~33% ({q1} < L <= {q2})": (pl.col("input_length") > q1)
        & (pl.col("input_length") <= q2),
        f"Longest ~33% (> {q2})": (pl.col("input_length") > q2),
    }

    if q1 == q2:
        print(f"Note: Input length terciles are equal ({q1}). Adjusting binning.")
        bins = {}
        if min_len_val < q1:
            bins[f"Short (< {q1})"] = pl.col("input_length") < q1
        bins[f"Equal to {q1}"] = pl.col("input_length") == q1
        if max_len_val > q1:
            bins[f"Long (> {q1})"] = pl.col("input_length") > q1

    for bin_name, filter_condition in bins.items():
        bin_df = results_df.filter(
            pl.col("input_length").is_not_null() & filter_condition
        )
        bin_total_samples = len(bin_df)

        if bin_total_samples == 0:
            print(f"\n--- {bin_name}: (No samples in this range) ---")
            continue

        # ... (Rest of leaderboard calculation for the bin - unchanged) ...
        bin_leaderboard = []
        for model in models_to_test:
            rank_col = f"rank_{model}"
            if rank_col in bin_df.columns:
                valid_ranks_df = bin_df.filter(pl.col(rank_col) > 0)
                num_valid = len(valid_ranks_df)
                if num_valid > 0:
                    avg_rank = valid_ranks_df[rank_col].mean()
                    bin_leaderboard.append((model, avg_rank, num_valid))
                else:
                    bin_leaderboard.append((model, float("inf"), 0))
            else:
                print(
                    f"Warning: Rank column '{rank_col}' not found for bin '{bin_name}'. Skipping model {model}."
                )
                bin_leaderboard.append((model, float("inf"), 0))
        bin_leaderboard.sort(key=lambda x: x[1])
        bin_header_line = f"\n--- {bin_name} ({bin_total_samples} Samples) ---"
        print(f"{bin_header_line}")
        print("-" * len(bin_header_line))
        for i, (model, avg_rank, num_valid) in enumerate(bin_leaderboard):
            rank_str = f"{avg_rank:.2f}" if num_valid > 0 else "N/A"
            print(
                f"{i + 1}. {model:<40} Avg Rank = {rank_str:<6} ({num_valid:>3}/{bin_total_samples} ranked samples)"
            )
        print("-" * len(bin_header_line))


# --- New functions for Constraint Violation Evaluation ---


def create_constraint_judge_prompt(
    constraints_prompt: str,  # Specific prompt detailing constraints
    input_data: str,
    model_output: str,
) -> str:
    """Creates a prompt for a judge model to identify constraint violations."""
    prompt = "You are an expert evaluator tasked with identifying violations of specific constraints in a Large Language Model (LLM) output based on a given input, and a set of constraints.\n\n"
    prompt += f"INPUT DATA GIVEN TO THE MODEL:\n---\n{input_data}\n---\n\n"
    prompt += f"MODEL OUTPUT TO EVALUATE:\n---\n{model_output}\n---\n\n"
    prompt += f"CONSTRAINTS TO CHECK:\n---\n{constraints_prompt}\n---\n\n"
    prompt += "TASK:\n1. Carefully review the MODEL OUTPUT.\n2. Compare it against the CONSTRAINTS TO CHECK, considering the INPUT DATA.\n3. Identify *all* constraints that the MODEL OUTPUT failed to meet.\n\n"
    prompt += "OUTPUT FORMAT:\nProvide your evaluation as a JSON object where each key is a unique identifier string for the violated constraint and the value is a brief string explaining the violation.\n"
    prompt += "The key MUST follow the format `CATEGORY.MainSection.SubSection` (e.g., `OUTPUT_FORMATTING.2.1`, `SEMANTIC_DISTILLATION.3.4`), referencing the corresponding section and subsection numbers from the 'CONSTRAINTS TO CHECK'. Use the ALL_CAPS category name and at most two numerical parts (e.g., `OUTPUT_FORMATTING.2` or `OUTPUT_FORMATTING.2.3` are valid, but `OUTPUT_FORMATTING.2.3.1` is NOT).\\n"
    prompt += (
        "If no constraints were violated, return an empty JSON object (`{}`).\\n\\n"
    )
    prompt += "Example (Constraints violated):\n"
    prompt += (
        "```json\\n"
        "{\\n"
        # Using example IDs derived from all_in_one.py
        '  "OUTPUT_FORMATTING.2.3": "Explain in detail where the violation happened.",\\n'
        '  "ATOMICITY.4.1": "Explain in detail where the violation happened.",\\n'
        '  "SEMANTIC_DISTILLATION.3.4.2": "Explain in detail where the violation happened."\\n'
        "}\\n"
        "```\\n\\n"
    )
    prompt += "Example (No constraints violated):\n"
    prompt += "```json\\n{}\\n```\\n\\n"
    prompt += "Return ONLY the JSON object and nothing else."
    return prompt


def parse_constraint_judge_output(judge_response: str) -> Optional[Dict[str, str]]:
    """Parses the constraint judge's dictionary response using the utility function."""
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
    judge_constraints_prompt: str,  # The prompt defining constraints
) -> Dict[int, str]:  # Map sample index to raw judge response string
    """Runs the constraint judge model for each sample."""
    judge_tasks = []
    judge_task_metadata = []  # Store sample index
    print(f"Preparing constraint judge tasks for model '{model_to_evaluate}'...")

    for i, sample_data in enumerate(sample_workflow_results):
        # Get the FINAL MERGED JSON output for the specific model being evaluated
        model_final_merged_json = sample_data.get("final_merged_json", {}).get(
            model_to_evaluate
        )

        # Check if the merged JSON exists and is a dictionary
        if not isinstance(model_final_merged_json, dict) or not model_final_merged_json:
            # Attempt to retrieve the parsed output as a fallback, if merged failed
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
                # Use the fallback if merged JSON is invalid but parsed output exists
                print(
                    f"Warning: Using fallback parsed output for sample {i}, model '{model_to_evaluate}' as merged JSON was invalid/missing."
                )
                model_output_for_judge = model_final_output_fallback
        else:
            # Serialize the valid merged JSON dictionary to a string for the judge
            try:
                # Use compact separators to minimize whitespace, ensure_ascii=False for broader char support
                model_output_for_judge = json.dumps(
                    model_final_merged_json, separators=(",", ":"), ensure_ascii=False
                )
            except TypeError as e:
                print(
                    f"Error serializing merged JSON for sample {i}, model '{model_to_evaluate}': {e}. Skipping."
                )
                continue

        # Create the specific prompt for the constraint judge using the selected output
        judge_prompt = create_constraint_judge_prompt(
            constraints_prompt=judge_constraints_prompt,
            input_data=sample_data["input_data"],
            model_output=model_output_for_judge,  # Use the serialized JSON or fallback
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
    judge_prompt_name: str,  # Renamed back
    workflow: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregates results for the constraint violation evaluation."""
    print("Aggregating constraint evaluation results...")
    results_data = []
    final_stage_task_ids: List[str] = workflow[-1][
        "prompts"
    ]  # Needed to get original instruction prompt?

    # Determine the instruction prompt used for the final stage (needed for context in results)
    # This assumes the judge needs context about the *last* instruction given
    # to the model being evaluated. If the constraints apply regardless of the final
    # instruction, this might be simplified.
    # For simplicity, if the final stage had multiple prompts, we might need
    # a specific way to decide which instruction matters most for the constraint check,
    # or combine them. Let's assume for now the judge prompt already contains sufficient context
    # or we primarily care about the constraints themselves. We'll store the constraints prompt name.

    for i, sample_data in enumerate(sample_workflow_results):
        input_context = sample_data.get("input_data", "ERROR: Input not found")
        input_length = len(input_context) if isinstance(input_context, str) else -1

        # Get the final parsed output and merged JSON for the evaluated model
        final_parsed_output = sample_data.get("final_parsed_outputs", {}).get(
            model_to_evaluate, "N/A"
        )
        final_merged_json = sample_data.get("final_merged_json", {}).get(
            model_to_evaluate, {}
        )

        # Get judge response and parse it
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
            # Store parsed violations dict as a JSON string or handle None/Error
            "violated_constraints": json.dumps(violated_constraints_dict)
            if violated_constraints_dict is not None
            else "ERROR: Parse Failed"
            if judge_raw_response
            else "Judge Skipped/Failed",
            "violation_count": len(violated_constraints_dict)
            if violated_constraints_dict is not None
            else -1,  # Count keys if dict exists, else -1
            "seed": effective_seed,
            "workflow_name": workflow_name,
            "model_evaluated": model_to_evaluate,
            "judge_prompt_name": judge_prompt_name,  # Renamed back
        }

        # Optionally, add intermediate task outputs for the evaluated model
        model_outputs_all_tasks = sample_data.get("model_outputs", {}).get(
            model_to_evaluate, {}
        )
        all_task_ids_in_workflow = set(
            p for stage in workflow for p in stage["prompts"]
        )
        for task_id in all_task_ids_in_workflow:
            col_name = f"output_{task_id}_{model_to_evaluate}"
            result_row[col_name] = model_outputs_all_tasks.get(task_id, "N/A")

        # Add the final merged JSON output (serialized) and individual fields
        if isinstance(final_merged_json, dict):
            try:
                # Store the complete merged JSON as a string
                result_row[f"final_merged_output_{model_to_evaluate}"] = json.dumps(
                    final_merged_json, separators=(",", ":"), ensure_ascii=False
                )
            except TypeError:
                result_row[f"final_merged_output_{model_to_evaluate}"] = (
                    "ERROR: Failed to serialize merged JSON"
                )

            # Also store individual keys from the merged JSON
            for key, value in final_merged_json.items():
                col_name = f"final_{key}_{model_to_evaluate}"
                try:
                    # Store lists/dicts as JSON strings, others as strings
                    result_row[col_name] = (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else str(value)
                    )
                except TypeError:
                    result_row[col_name] = "ERROR: Failed to serialize value"
        else:
            # Indicate error if final_merged_json wasn't a dict
            result_row[f"final_merged_output_{model_to_evaluate}"] = (
                "ERROR: Merged JSON not available or not a dict"
            )
            result_row[f"final_merged_json_error_{model_to_evaluate}"] = (
                "ERROR: Merged JSON not available or not a dict"
            )

        results_data.append(result_row)

    return results_data
