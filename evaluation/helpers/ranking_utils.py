# import asyncio # Removed
import random
from typing import Any, Dict, List, Optional, Tuple

import dask  # Added
import polars as pl

# from tqdm.asyncio import tqdm_asyncio # Removed
# Import the async LLM runner, assuming it's still needed for the judge
# If workflow_executor._run_single_prompt is removed/changed, adapt this import
# We might need direct access to get_text_completion if _run_single_prompt changes
from user_embeddings.utils.llm.get_text_completion import (
    get_text_completion,  # Direct import might be safer
)
from user_embeddings.utils.parsing import parse_llm_json_output

# --- Ranking Specific Helpers ---


def create_judge_prompt(
    instruction_prompt: str, input_data: str, outputs: Dict[str, str]
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """Creates a blinded prompt for the ranking judge model."""
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


# Internal async helper for judge call (used by synchronous wrapper)
async def _run_single_judge_llm_async(model_name: str, prompt: str) -> str:
    """Async helper to run a single judge LLM call."""
    try:
        # Assuming get_text_completion handles initialization or it's done globally
        result = await get_text_completion(model_name, prompt)
        return result
    except Exception as e:
        print(f"Error running judge model {model_name}: {e}")
        return f"ERROR: Judge model execution failed - {e}"


# Refactored to be synchronous
def run_judge_evaluation(
    # Expects the structure prepared *after* dask compute
    # List[Dict{'input_data': str, 'judge_inputs': {model: str}, 'detailed_model_outputs': ...}]
    judge_ready_results: List[Dict[str, Any]],
    judge_model: str,
    judge_instruction_prompt_text: str,
) -> Dict[
    int, Tuple[str, Dict[str, str], Dict[str, str]]
]:  # Map sample index -> (raw_resp, mask_map, orig_map)
    """Runs the ranking judge model synchronously for each sample."""
    judge_tasks_to_compute = []
    judge_task_metadata = []
    print("Preparing judge tasks (will run synchronously or via Dask compute)...")

    for i, sample_data in enumerate(judge_ready_results):
        # judge_ready_results contains 'judge_inputs' map: {model_name: judge_input_string}
        # For ranking, we need all model outputs for *one* sample to be judged together.
        # The structure passed should reflect this. Assuming judge_inputs contains *all* model outputs for the judge.

        # Let's rethink the input structure. We need the outputs for different models for the *same* sample input.
        # The current `judge_ready_results` structure seems correct for this.
        # `sample_data` here is one element of that list.

        valid_outputs_for_judge = sample_data.get(
            "judge_inputs"
        )  # This should map model->output_string
        if (
            not isinstance(valid_outputs_for_judge, dict)
            or len(valid_outputs_for_judge) < 2
        ):
            print(
                f"Skipping judge task for sample {i} due to insufficient valid final outputs ({len(valid_outputs_for_judge or {})} found in judge_inputs)."
            )
            continue

        # Create the blinded prompt
        judge_prompt, mask_map, original_map = create_judge_prompt(
            judge_instruction_prompt_text,
            sample_data["input_data"],
            valid_outputs_for_judge,  # Pass the prepared {model: output_str} map
        )

        # Use dask.delayed to represent the async call
        # This allows Dask to manage the async execution if desired later,
        # but we'll compute it synchronously here for simplicity.
        delayed_call = dask.delayed(_run_single_judge_llm_async)(
            judge_model, judge_prompt
        )
        judge_tasks_to_compute.append(delayed_call)

        judge_task_metadata.append(
            {"sample_index": i, "mask_map": mask_map, "original_map": original_map}
        )

    judge_responses_raw = []
    if judge_tasks_to_compute:
        print(f"Running {len(judge_tasks_to_compute)} judge tasks...")
        # Compute the delayed tasks synchronously using dask.compute
        # We could use the distributed client here too if preferred.
        judge_responses_raw = dask.compute(
            *judge_tasks_to_compute, scheduler="sync"
        )  # Use sync scheduler for local run
        print("Judge tasks complete.")
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


# aggregate_ranking_results needs adjustment to accept the new input format
def aggregate_ranking_results(
    # Expects the structure prepared *after* dask compute & judge input prep
    # List[Dict{'input_data': str, 'judge_inputs': {model: str}, 'detailed_model_outputs': {model: {task_id: TaskResult}}}]
    processed_results_list: List[Dict[str, Any]],
    judge_response_map: Dict[int, Tuple[str, Dict[str, str], Dict[str, str]]],
    models: List[str],
    seed: int,
    workflow_name: str,
    judge_prompt_name: str,
    # workflow: List[Dict[str, Any]], # Not strictly needed if using detailed_model_outputs
    # available_prompts: Dict[str, Tuple[str, str]], # Not needed if detailed_model_outputs has versions?
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Aggregates ranking results, including prompt versions and unmasked rationale."""
    print("Processing judge results and aggregating final data...")
    results_data = []
    # Assume prompt versions are not easily accessible here, need to reconsider if needed
    # judge_prompt_version = available_prompts.get(judge_prompt_name, ("", "N/A"))[1]

    for i, sample_data in enumerate(processed_results_list):
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
        if mask_to_original_map and ranking_masked:
            try:
                ranking_original = [
                    mask_to_original_map[masked]
                    for masked in ranking_masked
                    if masked in mask_to_original_map
                ]
                expected_models_in_judge_input = set(mask_to_original_map.values())
                if len(ranking_original) != len(expected_models_in_judge_input):
                    print(
                        f"Warning: Judge ranking length mismatch for sample {i}. Judge saw {len(expected_models_in_judge_input)} models, ranked {len(ranking_original)}."
                    )
            except KeyError as e:
                print(
                    f"Error translating ranking for sample {i}: Mask '{e}' not found in map."
                )
                ranking_original = None

        input_context = sample_data.get("input_data", "ERROR: Input not found")
        input_length = len(input_context) if isinstance(input_context, str) else -1
        unmasked_rationale = rationale
        if rationale and original_to_mask_map:
            if debug:
                print(
                    f"DEBUG: original_to_mask_map for sample {i}: {original_to_mask_map}"
                )
            sorted_map_items = sorted(
                original_to_mask_map.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
            for original_name, masked_name in sorted_map_items:
                unmasked_rationale = unmasked_rationale.replace(
                    masked_name, original_name
                )
        else:
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
            else "Judge Skipped/Failed",
            "input": input_context,
            "input_length": input_length,
            "judge_rationale": unmasked_rationale
            if unmasked_rationale
            else (
                "ERROR: Rationale missing/unmask failed"
                if judge_data
                else "Judge Skipped/Failed"
            ),
            "judge_any_correct": any_correct
            if any_correct is not None
            else ("ERROR: Parse failed" if judge_data else "Judge Skipped/Failed"),
            "seed": seed,
            "workflow_name": workflow_name,
            "judge_prompt_name": judge_prompt_name,
            # "judge_prompt_version": judge_prompt_version, # Reconsider how to get this if needed
        }

        rank_map = {model: -1 for model in models}
        if ranking_original:
            for rank, model in enumerate(ranking_original):
                if model in rank_map:
                    rank_map[model] = rank + 1

        # Add individual model ranks and raw/parsed outputs if available
        detailed_outputs = sample_data.get("detailed_model_outputs", {})
        for model in models:
            result_row[f"rank_{model}"] = rank_map.get(
                model, -1
            )  # Rank or -1 if not ranked/present
            model_results = detailed_outputs.get(model, {})  # {task_id: TaskResult}

            # Extract final task output(s) for this model (assuming logic similar to judge prep)
            # This part is tricky without the workflow definition readily available
            # Let's just store the prepared judge input string for now
            result_row[f"output_{model}"] = sample_data.get("judge_inputs", {}).get(
                model, "N/A"
            )
            # TODO: Revisit if detailed outputs per task are needed in final CSV

        results_data.append(result_row)

    return results_data


def calculate_and_print_leaderboard(
    results_df: pl.DataFrame, models_to_test: List[str]
):
    """Calculates and prints the final leaderboard based on average ranks, correctness, and input length bins."""
    print("\n--- Overall Leaderboard ---")
    overall_leaderboard = []
    total_samples = len(results_df)
    for model in models_to_test:
        rank_col = f"rank_{model}"
        if rank_col in results_df.columns:
            valid_ranks_df = results_df.filter(pl.col(rank_col) > 0)
            num_valid = len(valid_ranks_df)
            if num_valid > 0:
                avg_rank = valid_ranks_df[rank_col].mean()
                overall_leaderboard.append((model, avg_rank, num_valid))
            else:
                overall_leaderboard.append((model, float("inf"), 0))
        else:
            print(
                f"Warning: Rank column '{rank_col}' not found. Skipping model {model} in overall leaderboard."
            )
            overall_leaderboard.append((model, float("inf"), 0))
    overall_leaderboard.sort(key=lambda x: x[1])
    header_line = f"--- Overall Leaderboard ({total_samples} Samples) ---"
    print(f"\n{header_line}")
    print("-" * len(header_line))
    for i, (model, avg_rank, num_valid) in enumerate(overall_leaderboard):
        rank_str = f"{avg_rank:.2f}" if num_valid > 0 else "N/A"
        print(
            f"{i + 1}. {model:<40} Avg Rank = {rank_str:<6} ({num_valid:>3}/{total_samples} ranked samples)"
        )
    print("-" * len(header_line))

    print("\n--- Judge Correctness Assessment ---")
    if "judge_any_correct" in results_df.columns:
        correct_counts = results_df["judge_any_correct"].value_counts()
        print(correct_counts)
        num_judged = results_df.filter(pl.col("judge_any_correct") != "ERROR").height
        if num_judged > 0:
            try:
                true_count = correct_counts.filter(pl.col("judge_any_correct") == True)[
                    "count"
                ].sum()
            except pl.ColumnNotFoundError:
                true_count = 0  # Handle case where 'true' count might be missing
            percent_correct = (true_count / num_judged) * 100
            print(
                f"Percentage of samples where judge found AT LEAST ONE correct output: {percent_correct:.2f}%"
            )
        else:
            print("No samples were successfully judged for correctness.")
    else:
        print("'judge_any_correct' column not found in results.")

    print("\n--- Leaderboards by Dynamic Input Length (Terciles) ---")
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
