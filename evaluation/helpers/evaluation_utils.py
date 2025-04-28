import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from tqdm.asyncio import tqdm_asyncio

# Assuming these are accessible from the new location
# Adjust the import path if necessary based on your project structure
from user_embeddings.utils.get_text_completion import get_text_completion


async def run_model(model_name: str, prompt: str) -> str:
    """Runs a single model and returns its output."""
    try:
        result = await get_text_completion(model_name, prompt)
        return result
    except Exception as e:
        print(f"Error running model {model_name}: {e}")
        return f"ERROR: {e}"


def create_judge_prompt(
    instruction_prompt: str, input_data: str, outputs: Dict[str, str]
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """Creates the prompt for the judge LLM, shuffling and masking model names."""
    original_items = list(outputs.items())
    random.shuffle(original_items)

    masked_outputs = {}
    mask_to_original_map = {}
    original_to_mask_map = {}
    masked_model_names = []

    for i, (original_name, output) in enumerate(original_items):
        masked_name = f"Model {chr(ord('A') + i)}"
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
        '  "ranking": ["Model A", "Model C", "Model B"],\n'
        '  "rationale": "Model A was best because..., Model C was okay..., Model B failed...",\n'
        '  "any_correct": true\n'
        "}\n"
        "```\n"
    )
    prompt += f"The available anonymized model names are: {masked_model_names}. Return ONLY the JSON object and nothing else."

    return prompt, mask_to_original_map, original_to_mask_map


def parse_judge_output(
    judge_response: str,
) -> Tuple[Optional[List[str]], Optional[str], Optional[bool]]:
    """Parses the JSON ranking, rationale, and correctness from the judge's response."""
    try:
        # Use regex to find JSON block, handles optional ```json and ``` markers
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", judge_response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback: assume the whole response might be JSON if no ``` markers
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
    """Loads data from a specific CSV file or all test_output_*.csv files in a directory and samples it."""
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
            full_df = pl.concat(
                df_list, how="vertical_relaxed"
            )  # Use vertical_relaxed for schema flexibility
        except Exception as e:
            print(f"Error concatenating DataFrames: {e}")
            return None

    else:
        print(
            f"Error: Input source path is neither a file nor a directory: {input_source_path}"
        )
        return None

    # --- Sampling Logic (applies to both single file and combined data) ---
    if full_df is None or len(full_df) == 0:
        print("No data loaded after processing input source.")
        return None

    print(f"Total rows loaded: {len(full_df)}")

    # Ensure 'input_data' column exists after potential concatenation
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
    instruction_prompt: str,
) -> List[Dict[str, Any]]:
    """Runs test models concurrently and parses their outputs using the provided instruction prompt."""
    test_model_tasks = []
    task_metadata = []
    print("Preparing test model tasks...")
    for i, row in enumerate(sample_df.iter_rows(named=True)):
        # Use the 'formatted_context' column
        input_data = row["formatted_context"]
        # Removed call to get_teacher_prompt
        # instruction_prompt = get_teacher_prompt(input_context)

        # Create the full prompt for the model
        # (You might want to format this differently depending on prompt structure)
        model_prompt = f"{instruction_prompt}\n\nINPUT DATA:\n---\n{input_data}\n---"

        for model in models_to_test:
            # Pass the combined prompt to run_model
            task = asyncio.create_task(run_model(model, model_prompt))
            test_model_tasks.append(task)
            task_metadata.append(
                {
                    "sample_index": i,
                    "model": model,
                    "input_data": input_data,  # Store input_data (which is formatted_context)
                    # "instruction_prompt": instruction_prompt, # No need to store static prompt here
                }
            )

    print(f"Running {len(test_model_tasks)} test model tasks concurrently...")
    test_model_results_raw = await tqdm_asyncio.gather(
        *test_model_tasks, desc="Running Test Models", unit="task"
    )

    print("Organizing and parsing test results...")
    sample_intermediate_results: List[Dict[str, Any]] = [
        {} for _ in range(len(sample_df))
    ]
    for i, raw_output in enumerate(test_model_results_raw):
        meta = task_metadata[i]
        sample_index = meta["sample_index"]
        model = meta["model"]

        if not sample_intermediate_results[sample_index]:
            sample_intermediate_results[sample_index] = {
                "input_data": meta["input_data"],
                # Store instruction_prompt at sample level if needed later (e.g., for judge)
                # "instruction_prompt": instruction_prompt,
                "raw_outputs": {},
                "parsed_outputs": {},
            }

        sample_intermediate_results[sample_index]["raw_outputs"][model] = raw_output

        # Placeholder parsing - replace with actual logic if needed
        if raw_output.startswith("ERROR:"):
            parsed_output_str = raw_output
        else:
            # Removed call to parse_teacher_prompt_output
            # try:
            #     parsed_output = parse_teacher_prompt_output(raw_output)
            #     parsed_output_str = str(parsed_output)
            # except Exception as parse_error:
            #     print(
            #         f"Error parsing output from {model} for sample {sample_index}: {parse_error}"
            #     )
            #     parsed_output_str = f"ERROR PARSING OUTPUT: {parse_error}\\nRAW OUTPUT:\\n{raw_output}"

            # Simple placeholder: just use the raw output as parsed for now
            # You might want to implement JSON parsing or regex here based on expected output format
            parsed_output_str = raw_output.strip()

        sample_intermediate_results[sample_index]["parsed_outputs"][model] = (
            parsed_output_str
        )

    return sample_intermediate_results


async def run_judge_evaluation(
    sample_intermediate_results: List[Dict[str, Any]],
    judge_model: str,
    instruction_prompt: str,
) -> Dict[int, Tuple[str, Dict[str, str], Dict[str, str]]]:
    """Runs the judge model for each sample, returning responses and name mappings."""
    judge_tasks = []
    judge_task_metadata = []
    print("Preparing judge tasks...")
    for i, intermediate_data in enumerate(sample_intermediate_results):
        if (
            "parsed_outputs" in intermediate_data
            and intermediate_data["parsed_outputs"]
            and len(intermediate_data["parsed_outputs"]) > 1
        ):
            # Pass the static instruction_prompt to the judge prompt creation
            # Capture all three return values from create_judge_prompt
            judge_prompt, mask_map, original_map = create_judge_prompt(
                instruction_prompt,
                intermediate_data["input_data"],
                intermediate_data["parsed_outputs"],
            )
            task = asyncio.create_task(run_model(judge_model, judge_prompt))
            judge_tasks.append(task)
            # Store both maps in metadata
            judge_task_metadata.append(
                {"sample_index": i, "mask_map": mask_map, "original_map": original_map}
            )
        else:
            print(
                f"Skipping judge task for sample {i} due to missing/empty or single parsed outputs."
            )

    judge_responses_raw = []
    if judge_tasks:
        print(f"Running {len(judge_tasks)} judge tasks concurrently...")
        judge_responses_raw = await tqdm_asyncio.gather(
            *judge_tasks, desc="Running Judge Models", unit="task"
        )
    else:
        print("No judge tasks to run.")

    # Update the type hint for judge_response_map to reflect the stored tuple
    judge_response_map: Dict[int, Tuple[str, Dict[str, str], Dict[str, str]]] = {}
    for i, raw_response in enumerate(judge_responses_raw):
        meta = judge_task_metadata[i]
        sample_index = meta["sample_index"]
        mask_map = meta["mask_map"]
        original_map = meta["original_map"]  # Retrieve the original map
        # Store the tuple (raw_response, mask_to_original_map, original_to_mask_map)
        judge_response_map[sample_index] = (raw_response, mask_map, original_map)

    return judge_response_map


def aggregate_results(
    sample_intermediate_results: List[Dict[str, Any]],
    judge_response_map: Dict[int, Tuple[str, Dict[str, str], Dict[str, str]]],
    models_to_test: List[str],
    effective_seed: int,
    prompt_module_name: str,
) -> List[Dict[str, Any]]:
    """Aggregates results including inputs, outputs, ranks (translated), rationale (unmasked), correctness, seed, and prompt name."""
    print("Processing judge results and aggregating final data...")
    results_data = []

    for i, intermediate_data in enumerate(sample_intermediate_results):
        judge_data = judge_response_map.get(i)
        ranking_masked, rationale, any_correct = (None, None, None)
        mask_to_original_map = None
        original_to_mask_map = None  # Initialize original_to_mask_map

        if judge_data:
            # Unpack the original_to_mask_map as well
            judge_response, mask_to_original_map, original_to_mask_map = judge_data
            ranking_masked, rationale, any_correct = parse_judge_output(judge_response)
        else:
            print(
                f"No judge response data found for sample {i}, likely skipped or failed."
            )

        ranking_original = None
        if ranking_masked and mask_to_original_map:
            try:
                ranking_original = [
                    mask_to_original_map[masked]
                    for masked in ranking_masked
                    if masked in mask_to_original_map
                ]

                if len(ranking_original) != len(mask_to_original_map):
                    print(
                        f"Warning: Judge ranking for sample {i} ({ranking_masked}) after translation "
                        f"({ranking_original}) did not contain all expected models from map "
                        f"({list(mask_to_original_map.values())}). Some ranks might be missing or judge hallucinated names."
                    )
                    # If the judge hallucinated names not in the map, ranking_original will be shorter.
                    # If the judge missed ranking some models, ranking_original will also be shorter.
                    # We might want to invalidate the ranking here, or proceed carefully.
                    # For now, we let it proceed, and the check below against expected_models_in_sample handles inconsistencies.

            except KeyError as e:
                print(
                    f"Error translating ranking for sample {i}: Masked name {e} not found in map {mask_to_original_map}. Raw masked ranking: {ranking_masked}"
                )
                ranking_original = None  # Invalidate if translation fails

        # Use input_data which should be stored in intermediate results
        input_context = intermediate_data.get("input_data", "ERROR: Input not found")
        input_length = len(input_context) if isinstance(input_context, str) else -1

        # Unmask the rationale before storing
        unmasked_rationale = rationale
        if rationale and original_to_mask_map:
            unmasked_rationale = rationale
            # Sort by length of masked name descending to replace longer names first if needed (e.g. Model AB before Model A)
            # Though current naming scheme ('Model A', 'Model B') makes this less critical
            sorted_map_items = sorted(
                original_to_mask_map.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
            for original_name, masked_name in sorted_map_items:
                # Use word boundaries to avoid partial replacements (e.g., replacing 'Model A' in 'Model AB')
                # Need to escape potential regex characters in model names if they exist
                # For simplicity, we'll assume 'Model X' format doesn't need escaping
                unmasked_rationale = re.sub(
                    rf"\b{re.escape(masked_name)}\b", original_name, unmasked_rationale
                )

        result_row = {
            "input": input_context,
            "input_length": input_length,
            # Store the unmasked rationale
            "judge_rationale": unmasked_rationale
            if unmasked_rationale
            else (
                "ERROR: Rationale not parsed" if judge_data else "Judge Skipped/Failed"
            ),
            "judge_any_correct": any_correct if any_correct is not None else "ERROR",
            "seed": effective_seed,
            "prompt_name": prompt_module_name,
        }

        rank_map = {}
        if ranking_original:
            expected_models_in_sample = set(
                intermediate_data.get("parsed_outputs", {}).keys()
            )
            ranked_models_in_sample = set(ranking_original)

            if not ranked_models_in_sample.issubset(expected_models_in_sample):
                print(
                    f"Warning: Translated judge ranking for sample {i} ({ranking_original}) "
                    f"contains models not expected for this sample ({list(expected_models_in_sample)}). "
                    "Assigning default ranks (-1)."
                )
                rank_map = {model: -1 for model in models_to_test}
            elif len(ranked_models_in_sample) != len(expected_models_in_sample):
                print(
                    f"Warning: Translated judge ranking for sample {i} ({ranking_original}) "
                    f"is missing some expected models ({list(expected_models_in_sample - ranked_models_in_sample)}). "
                    "Assigning partial ranks based on available data, others get -1."
                )
                rank_map = {
                    model: rank + 1 for rank, model in enumerate(ranking_original)
                }
                for model in expected_models_in_sample - ranked_models_in_sample:
                    rank_map[model] = -1
            else:
                rank_map = {
                    model: rank + 1 for rank, model in enumerate(ranking_original)
                }

        else:
            rank_map = {model: -1 for model in models_to_test}

        for model in models_to_test:
            result_row[f"raw_output_{model}"] = intermediate_data.get(
                "raw_outputs", {}
            ).get(model, "N/A")
            result_row[f"parsed_output_{model}"] = intermediate_data.get(
                "parsed_outputs", {}
            ).get(model, "N/A")
            result_row[f"rank_{model}"] = rank_map.get(model, -1)

        results_data.append(result_row)

    return results_data


def save_results(results_df: pl.DataFrame, output_file: Path):
    """Saves the evaluation results DataFrame to a CSV file."""
    print(f"Saving evaluation results to {output_file}...")
    # Order the columns alphabetically before saving
    results_df = results_df.select(sorted(results_df.columns))
    results_df.write_csv(output_file)


def calculate_and_print_leaderboard(
    results_df: pl.DataFrame, models_to_test: List[str]
):
    """Calculates and prints the final leaderboard based on average ranks, correctness, and input length bins."""
    print("--- Overall Leaderboard (Average Rank) ---")
    total_samples = len(results_df)

    # --- Calculate Overall Leaderboard ---
    overall_leaderboard = []
    for model in models_to_test:
        rank_col = f"rank_{model}"
        if rank_col in results_df.columns:
            # Filter out invalid ranks (-1)
            valid_ranks_df = results_df.filter(pl.col(rank_col) > 0)
            num_valid = len(valid_ranks_df)
            if num_valid > 0:
                avg_rank = valid_ranks_df[rank_col].mean()
                overall_leaderboard.append((model, avg_rank, num_valid))
            else:
                overall_leaderboard.append((model, float("inf"), 0))
        else:
            print(
                f"Warning: Rank column '{rank_col}' not found. Skipping model {model} for overall."
            )
            overall_leaderboard.append((model, float("inf"), 0))

    overall_leaderboard.sort(key=lambda x: x[1])

    # --- Print Overall Leaderboard ---
    header_line = (
        f" Models Tested: {len(models_to_test)} | Total Samples: {total_samples} "
    )
    print("-" * len(header_line))
    print(header_line)
    print("-" * len(header_line))
    for i, (model, avg_rank, num_valid) in enumerate(overall_leaderboard):
        rank_str = f"{avg_rank:.2f}" if num_valid > 0 else "N/A"
        print(
            f"{i + 1}. {model:<40} Avg Rank = {rank_str:<6} ({num_valid:>3}/{total_samples} valid runs)"
        )
    print("-" * len(header_line))

    # --- Calculate and Print Overall Correctness ---
    if "judge_any_correct" in results_df.columns:
        try:
            # Ensure the column is boolean before filtering
            correct_df = results_df.filter(pl.col("judge_any_correct") == True)
            num_correct_samples = len(correct_df)
            if total_samples > 0:
                correct_percentage = (num_correct_samples / total_samples) * 100
                print(
                    f"\nJudge Assessment: {num_correct_samples}/{total_samples} ({correct_percentage:.1f}%) samples had at least one correct output."
                )
            else:
                print("\nJudge Assessment: No samples to assess correctness.")
        except (
            Exception
        ) as e:  # Catch potential errors if column isn't boolean as expected
            print(
                f"\nCould not calculate correctness percentage due to error: {e}. Check 'judge_any_correct' column type."
            )
    else:
        print(
            "\n'judge_any_correct' column not found in results. Skipping correctness calculation."
        )

    # --- Dynamically Calculate and Print Leaderboards per Input Length Bin ---
    print("\n--- Leaderboards by Dynamic Input Length (Terciles) ---")

    if (
        "input_length" not in results_df.columns
        or results_df["input_length"].is_null().all()
        or len(results_df.drop_nulls("input_length")) < 3
    ):
        print(
            "Could not calculate dynamic bins: 'input_length' column missing, empty, or too few values."
        )
        return

    # Calculate Terciles (33.3rd and 66.7th percentiles)
    # Ensure we drop nulls and handle potential errors
    try:
        # Call quantile separately for each percentile
        q1_val = results_df["input_length"].drop_nulls().quantile(0.333)
        q2_val = results_df["input_length"].drop_nulls().quantile(0.667)
        if q1_val is None or q2_val is None:
            raise ValueError("Quantile calculation returned None")

        q1 = int(q1_val)  # Lower tercile boundary
        q2 = int(q2_val)  # Upper tercile boundary
        min_len_val = results_df["input_length"].min()
        max_len_val = results_df["input_length"].max()
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

    # Handle edge case where quantiles might be equal (low variance in lengths)
    if q1 == q2:
        print(f"Note: Input length quantiles are equal ({q1}). Adjusting binning.")
        bins = {
            f"Short (< {q1})": (pl.col("input_length") < q1),
            f"Equal to {q1}": (pl.col("input_length") == q1),
            f"Long (> {q1})": (pl.col("input_length") > q1),
        }
        # Remove empty bins if min/max are also the same
        if min_len_val == q1:
            bins.pop(f"Short (< {q1})", None)
        if max_len_val == q1:
            bins.pop(f"Long (> {q1})", None)

    for bin_name, filter_condition in bins.items():
        bin_df = results_df.filter(filter_condition)
        bin_total_samples = len(bin_df)

        if bin_total_samples == 0:
            print(f"--- {bin_name}: (No samples in this range) ---")
            continue

        bin_leaderboard = []
        for model in models_to_test:
            rank_col = f"rank_{model}"
            if rank_col in bin_df.columns:
                # Filter out invalid ranks (-1)
                valid_ranks_df = bin_df.filter(pl.col(rank_col) > 0)
                num_valid = len(valid_ranks_df)
                if num_valid > 0:
                    avg_rank = valid_ranks_df[rank_col].mean()
                    bin_leaderboard.append((model, avg_rank, num_valid))
                else:
                    bin_leaderboard.append((model, float("inf"), 0))
            else:
                # This case should ideally not happen if overall check passed, but good to have
                print(
                    f"Warning: Rank column '{rank_col}' not found for bin '{bin_name}'. Skipping model {model}."
                )
                bin_leaderboard.append((model, float("inf"), 0))

        bin_leaderboard.sort(key=lambda x: x[1])

        # --- Print Bin Leaderboard ---
        bin_header_line = f"--- {bin_name} ({bin_total_samples} Samples) ---"
        print(f"{bin_header_line}")
        print("-" * len(bin_header_line))
        for i, (model, avg_rank, num_valid) in enumerate(bin_leaderboard):
            rank_str = f"{avg_rank:.2f}" if num_valid > 0 else "N/A"
            print(
                f"{i + 1}. {model:<40} Avg Rank = {rank_str:<6} ({num_valid:>3}/{bin_total_samples} valid runs)"
            )
        print("-" * len(bin_header_line))
