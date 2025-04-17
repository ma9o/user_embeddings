import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from tqdm.asyncio import tqdm_asyncio

# Assuming these are accessible from the new location
# Adjust the import path if necessary based on your project structure
from user_embeddings.utils.get_text_completion import get_text_completion
from user_embeddings.utils.teacher_prompt import (
    get_teacher_prompt,
    parse_teacher_prompt_output,
)


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
) -> str:
    """Creates the prompt for the judge LLM."""
    prompt = "You are an expert evaluator tasked with ranking the quality of different Large Language Model (LLM) outputs based on a given instruction and input.\\\\n\\\\n"
    prompt += f"INSTRUCTION PROMPT GIVEN TO MODELS:\\\\n---\\\\n{instruction_prompt}\\\\n---\\\\n\\\\n"
    prompt += f"INPUT DATA GIVEN TO MODELS:\\\\n---\\\\n{input_data}\\\\n---\\\\n\\\\n"
    prompt += 'LLM OUTPUTS TO EVALUATE:\\\\n---\\"'
    for i, (model_name, output) in enumerate(outputs.items()):
        prompt += f"\\\\nOutput {i + 1} (Model: {model_name}):\\\\n{output}\\\\n---"

    prompt += "\\\\n\\\\nTASK:\\\\nEvaluate the outputs based *only* on how well they follow the INSTRUCTION PROMPT for the given INPUT DATA. Consider clarity, structure, adherence to format, and accuracy of the generated summary/actions based *solely* on the provided input context.\\\\n\\\\n"
    prompt += "RANKING FORMAT:\\\\nProvide your ranking as a JSON object containing two keys: 'ranking' (a list of model names, ordered from best to worst) and 'rationale' (a brief explanation for your ranking decisions). For example:\\\\n"
    prompt += (
        "```json\\\\n"
        "{\\\\n"
        '  "ranking": ["model_name_best", "model_name_middle", "model_name_worst"],\\\\n'
        '  "rationale": "Model A was best because... Model B struggled with... Model C failed to..."\\\\n'
        "}\\\\n"
        "```\\\\n"
    )
    prompt += f"The available model names are: {list(outputs.keys())}. Return ONLY the JSON object and nothing else."

    return prompt


def parse_judge_output(
    judge_response: str,
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Parses the JSON ranking and rationale from the judge\'s response."""
    try:
        # Extract JSON block if necessary
        if "```json" in judge_response:
            json_str = judge_response.split("```json\\\\n")[1].split("\\\\n```")[0]
        else:
            json_str = judge_response

        parsed_json = json.loads(json_str)

        if not isinstance(parsed_json, dict):
            print(f"Error: Judge output is not a JSON object: {parsed_json}")
            return None, None

        ranking = parsed_json.get("ranking")
        rationale = parsed_json.get("rationale")

        if not isinstance(ranking, list) or not all(
            isinstance(item, str) for item in ranking
        ):
            print(f"Error: 'ranking' key is not a list of strings: {ranking}")
            ranking = None  # Set ranking to None if invalid

        if not isinstance(rationale, str):
            print(f"Error: 'rationale' key is not a string: {rationale}")
            rationale = None  # Set rationale to None if invalid

        return ranking, rationale

    except (json.JSONDecodeError, IndexError, TypeError) as e:
        print(f"Error parsing judge output: {e}\\\\nRaw output:\\\\n{judge_response}")
        return None, None


def load_and_sample_data(
    input_dir: Path, num_samples: int, seed: Optional[int]
) -> Optional[pl.DataFrame]:
    """Loads data from CSV files in the input directory and samples it."""
    print(f"Loading data from {input_dir}...")
    all_files = list(input_dir.glob("test_output_*.csv"))
    if not all_files:
        print(f"No CSV files found in {input_dir}")
        return None

    df_list = [pl.read_csv(f) for f in all_files]
    full_df = pl.concat(df_list)
    print(f"Total rows loaded: {len(full_df)}")

    if len(full_df) < num_samples:
        print(
            f"Warning: Not enough data ({len(full_df)} rows) for {num_samples} samples. Using all available data."
        )
        sample_df = full_df
    else:
        sample_df = full_df.sample(n=num_samples, shuffle=True, seed=seed)

    print(f"Selected {len(sample_df)} rows for evaluation.")
    return sample_df


async def run_and_parse_test_models(
    sample_df: pl.DataFrame, models_to_test: List[str]
) -> List[Dict[str, Any]]:
    """Runs test models concurrently and parses their outputs."""
    test_model_tasks = []
    task_metadata = []
    print("Preparing test model tasks...")
    for i, row in enumerate(sample_df.iter_rows(named=True)):
        input_context = row["formatted_context"]
        instruction_prompt = get_teacher_prompt(input_context)
        for model in models_to_test:
            task = asyncio.create_task(run_model(model, instruction_prompt))
            test_model_tasks.append(task)
            task_metadata.append(
                {
                    "sample_index": i,
                    "model": model,
                    "input_context": input_context,
                    "instruction_prompt": instruction_prompt,
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
                "input_context": meta["input_context"],
                "instruction_prompt": meta["instruction_prompt"],
                "raw_outputs": {},
                "parsed_outputs": {},
            }

        sample_intermediate_results[sample_index]["raw_outputs"][model] = raw_output

        if raw_output.startswith("ERROR:"):
            parsed_output_str = raw_output
        else:
            try:
                parsed_output = parse_teacher_prompt_output(raw_output)
                parsed_output_str = str(parsed_output)
            except Exception as parse_error:
                print(
                    f"Error parsing output from {model} for sample {sample_index}: {parse_error}"
                )
                parsed_output_str = f"ERROR PARSING OUTPUT: {parse_error}\\\\nRAW OUTPUT:\\\\n{raw_output}"
        sample_intermediate_results[sample_index]["parsed_outputs"][model] = (
            parsed_output_str
        )

    return sample_intermediate_results


async def run_judge_evaluation(
    sample_intermediate_results: List[Dict[str, Any]], judge_model: str
) -> Dict[int, str]:
    """Runs the judge model for each sample with valid test model outputs."""
    judge_tasks = []
    judge_task_sample_indices = []
    print("Preparing judge tasks...")
    for i, intermediate_data in enumerate(sample_intermediate_results):
        if (
            "parsed_outputs" in intermediate_data
            and intermediate_data["parsed_outputs"]
        ):
            judge_prompt = create_judge_prompt(
                intermediate_data["instruction_prompt"],
                intermediate_data["input_context"],
                intermediate_data["parsed_outputs"],
            )
            task = asyncio.create_task(run_model(judge_model, judge_prompt))
            judge_tasks.append(task)
            judge_task_sample_indices.append(i)
        else:
            print(
                f"Skipping judge task for sample {i} due to missing/empty parsed outputs."
            )

    judge_responses = []
    if judge_tasks:
        print(f"Running {len(judge_tasks)} judge tasks concurrently...")
        judge_responses = await tqdm_asyncio.gather(
            *judge_tasks, desc="Running Judge Models", unit="task"
        )
    else:
        print("No judge tasks to run.")

    return dict(zip(judge_task_sample_indices, judge_responses))


def aggregate_results(
    sample_intermediate_results: List[Dict[str, Any]],
    judge_response_map: Dict[int, str],
    models_to_test: List[str],
) -> List[Dict[str, Any]]:
    """Aggregates results including inputs, outputs, ranks, and rationale."""
    print("Processing judge results and aggregating final data...")
    results_data = []

    for i, intermediate_data in enumerate(sample_intermediate_results):
        judge_response = judge_response_map.get(i)
        ranking, rationale = (
            parse_judge_output(judge_response) if judge_response else (None, None)
        )

        if not judge_response:
            print(f"No judge response found for sample {i}, likely skipped or failed.")

        input_context = intermediate_data.get("input_context", "ERROR: Input not found")
        input_length = (
            len(input_context) if isinstance(input_context, str) else -1
        )  # Calculate length

        result_row = {
            "input": input_context,
            "input_length": input_length,  # Add input length here
            "judge_rationale": rationale
            if rationale
            else "ERROR: Rationale not parsed or judge skipped",
        }

        rank_map = {}
        if ranking:
            expected_models = set(models_to_test)
            actual_models = set(ranking)
            if actual_models == expected_models:
                rank_map = {model: rank + 1 for rank, model in enumerate(ranking)}
            else:
                print(
                    f"Warning: Judge ranking for sample {i} ({ranking}) "
                    f"does not match/contain all MODELS_TO_TEST ({list(expected_models)}). "
                    "Assigning default ranks (-1)."
                )
                rank_map = {model: -1 for model in models_to_test}
        else:
            rank_map = {model: -1 for model in models_to_test}

        for model in models_to_test:
            result_row[f"raw_output_{model}"] = intermediate_data.get(
                "raw_outputs", {}
            ).get(model, "ERROR: Model raw output not found")
            result_row[f"parsed_output_{model}"] = intermediate_data.get(
                "parsed_outputs", {}
            ).get(model, "ERROR: Model parsed output not found")
            result_row[f"rank_{model}"] = rank_map.get(model, -1)

        results_data.append(result_row)

    return results_data


def save_results(results_df: pl.DataFrame, output_file: Path):
    """Saves the evaluation results DataFrame to a CSV file."""
    print(f"\\\\nSaving evaluation results to {output_file}...")
    # Order the columns alphabetically before saving
    results_df = results_df.select(sorted(results_df.columns))
    results_df.write_csv(output_file)


def calculate_and_print_leaderboard(
    results_df: pl.DataFrame, models_to_test: List[str]
):
    """Calculates and prints the final leaderboard based on average ranks,
    including breakdowns by dynamic input length bins."""
    print("\\n--- Overall Leaderboard (Average Rank) ---")
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

    # --- Dynamically Calculate and Print Leaderboards per Input Length Bin ---
    print("\\n--- Leaderboards by Dynamic Input Length (Terciles) ---")

    if (
        "input_length" not in results_df.columns
        or results_df["input_length"].is_null().all()
        or len(results_df.drop_nulls("input_length")) < 3
    ):
        print(
            "\\nCould not calculate dynamic bins: 'input_length' column missing, empty, or too few values."
        )
        return  # Exit if we can't calculate bins

    # Calculate Terciles (33.3rd and 66.7th percentiles)
    # Ensure we drop nulls and handle potential errors
    try:
        quantiles = (
            results_df["input_length"].drop_nulls().quantile([0.333, 0.667]).to_list()
        )
        q1 = int(quantiles[0])  # Lower tercile boundary
        q2 = int(quantiles[1])  # Upper tercile boundary
        min_len_val = results_df["input_length"].min()
        max_len_val = results_df["input_length"].max()
    except Exception as e:
        print(f"\\nError calculating quantiles for input length bins: {e}")
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
        print(f"\\nNote: Input length quantiles are equal ({q1}). Adjusting binning.")
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
            print(f"\\n--- {bin_name}: (No samples in this range) ---")
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

        # --- Print Bin Leaderboard ---\
        bin_header_line = f"--- {bin_name} ({bin_total_samples} Samples) ---"
        print(f"\\n{bin_header_line}")
        print("-" * len(bin_header_line))
        for i, (model, avg_rank, num_valid) in enumerate(bin_leaderboard):
            rank_str = f"{avg_rank:.2f}" if num_valid > 0 else "N/A"
            print(
                f"{i + 1}. {model:<40} Avg Rank = {rank_str:<6} ({num_valid:>3}/{bin_total_samples} valid runs)"
            )
        print("-" * len(bin_header_line))
