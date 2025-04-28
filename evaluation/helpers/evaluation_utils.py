import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl
from pydantic import BaseModel, ValidationError
from tqdm.asyncio import tqdm_asyncio

# Import the NEW orchestrator and the single prompt runner
from user_embeddings.utils.workflow_executor import (
    _run_single_prompt,
    run_workflow_on_samples,
)


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
    # ...
    try:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", judge_response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = judge_response.strip()
        parsed_json = json.loads(json_str)
        if not isinstance(parsed_json, dict):
            print(f"Error: Judge output is not a JSON object: {parsed_json}")
            return None, None, None
        ranking = parsed_json.get("ranking")
        rationale = parsed_json.get("rationale")
        any_correct = parsed_json.get("any_correct")
        if not isinstance(ranking, list) or not all(
            isinstance(item, str) for item in ranking
        ):
            print(f"Error: 'ranking' key is not a list of strings: {ranking}")
            ranking = None
        if not isinstance(rationale, str):
            print(f"Error: 'rationale' key is not a string: {rationale}")
            rationale = None
        if not isinstance(any_correct, bool):
            print(f"Error: 'any_correct' key is not a boolean: {any_correct}")
            any_correct = None
        return ranking, rationale, any_correct
    except (json.JSONDecodeError, IndexError, TypeError) as e:
        print(f"Error parsing judge output: {e}\nRaw output:\n{judge_response}")
        return None, None, None


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
    Runs test models using the workflow orchestrator and parses final outputs for the judge.
    Also validates and merges JSON outputs from final stage tasks using Pydantic models.
    """
    # Call the orchestrator from the utils module
    raw_results_by_sample = await run_workflow_on_samples(
        sample_df=sample_df,
        models_to_test=models_to_test,
        workflow=workflow,
        available_prompts=available_prompts,
        input_formatters=input_formatters,
        input_column="formatted_context",  # Assuming this is the standard input column
    )

    # --- Parse final outputs for the judge ---
    print("Parsing final outputs for judge...")
    parsed_results: List[Dict[str, Any]] = []
    final_stage_task_ids: List[str] = workflow[-1]["prompts"]

    for sample_data in raw_results_by_sample:
        # Initialize the structure expected by downstream functions
        parsed_sample_data = {
            "input_data": sample_data["input_data"],
            "model_outputs": sample_data["model_outputs"],  # Keep raw outputs
            "final_parsed_outputs": {},  # Populate this
            "final_merged_json": {},  # Add a new field to store the MERGED JSON from final stage tasks
        }

        # Add a new field to store the MERGED JSON from final stage tasks
        parsed_sample_data["final_merged_json"] = {}

        model_outputs = sample_data.get("model_outputs", {})
        for model, task_outputs in model_outputs.items():
            final_outputs_for_judge = {}
            final_merged_json_for_model = {}  # Store merged JSON per model
            parsing_error = False
            for task_id in final_stage_task_ids:  # task_id is prompt_module_name
                output = task_outputs.get(task_id)
                if output is None:
                    print(
                        f"Warning: Final task ID '{task_id}' missing for model {model}. Marking as error."
                    )
                    final_outputs_for_judge[task_id] = "ERROR: Output missing"
                    parsing_error = True
                elif output.startswith("ERROR:"):
                    final_outputs_for_judge[task_id] = output
                    parsing_error = True
                else:
                    final_outputs_for_judge[task_id] = output.strip()
                    # Attempt to parse the output as JSON and merge
                    try:
                        # Basic JSON block extraction (like in judge parsing)
                        match = re.search(
                            r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", output, re.DOTALL
                        )
                        if match:
                            json_str = match.group(1)
                        else:
                            json_str = output.strip()

                        # Find the Pydantic model for this task ID
                        output_model = available_output_models.get(task_id)

                        if output_model:
                            try:
                                # Parse and validate using the Pydantic model
                                parsed_data = output_model.model_validate_json(json_str)
                                # Merge the validated data (as a dict) into the model's merged JSON
                                final_merged_json_for_model.update(
                                    parsed_data.model_dump()
                                )
                            except ValidationError as ve:
                                print(
                                    f"Warning: Pydantic validation failed for task '{task_id}', model '{model}': {ve}. Skipping merge."
                                )
                        else:
                            # If no Pydantic model is defined for this final stage task ID
                            print(
                                f"Warning: No Pydantic output model found in mapping for final task '{task_id}'. Cannot validate or merge its JSON output."
                            )

                    except (json.JSONDecodeError, TypeError) as e:
                        # This might catch errors during the initial JSON block extraction/stripping
                        print(
                            f"Warning: Could not parse JSON output for task '{task_id}', model '{model}': {e}. Raw output: {output[:100]}..."
                        )
                        # Mark merge as failed?

            # Combine final outputs for the judge (simple join)
            if parsing_error:
                parsed_final_output_str = (
                    "ERROR: One or more final stage outputs failed."
                )
            elif len(final_outputs_for_judge) == 1:
                parsed_final_output_str = list(final_outputs_for_judge.values())[0]
            elif len(final_outputs_for_judge) > 1:
                # Default: Simple join for judge input when multiple outputs
                parsed_final_output_str = "\n---\n".join(
                    f"Output from {tid}:\n{out}"
                    for tid, out in final_outputs_for_judge.items()
                )
            else:  # No valid final outputs
                parsed_final_output_str = "ERROR: No valid final outputs found."

            parsed_sample_data["final_parsed_outputs"][model] = parsed_final_output_str
            parsed_sample_data["final_merged_json"][model] = final_merged_json_for_model

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


def aggregate_results(
    sample_workflow_results: List[Dict[str, Any]],
    judge_response_map: Dict[int, Tuple[str, Dict[str, str], Dict[str, str]]],
    models_to_test: List[str],
    effective_seed: int,
    workflow_name: str,
    judge_prompt_module_name: str,
    workflow: List[Dict[str, Any]],
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
            sorted_map_items = sorted(
                original_to_mask_map.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
            for original_name, masked_name in sorted_map_items:
                # Use regex to replace based on leading word boundary and sorting by length
                unmasked_rationale = re.sub(
                    rf"\\b{re.escape(masked_name)}",  # Removed trailing \\b
                    original_name,
                    unmasked_rationale,
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
            "judge_prompt_name": judge_prompt_module_name,
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
