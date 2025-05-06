import asyncio
import json

# import random # Removed as create_judge_prompt is moved
from typing import Any, Dict, List, Tuple

import polars as pl
from pydantic import BaseModel  # Keep if other functions use it, otherwise remove
from tqdm.asyncio import tqdm_asyncio

# Removed: from user_embeddings.utils.parsing import parse_llm_json_output
# Import new judge prompt functions
from src.user_embeddings.utils.judge_prompts.rank_benchmark_judge import (
    create_judge_prompt,
    parse_judge_output,
)
from user_embeddings.utils.llm.workflow_executor import (
    WorkflowStage,
    _run_single_prompt,
)

# --- Ranking Specific Helpers ---

# // ... existing code ...
# create_judge_prompt function was here, now removed

# // ... existing code ...
# parse_judge_output function was here, now removed


async def run_judge_evaluation(
    sample_workflow_results: List[Dict[str, Any]],
    judge_model: str,
    judge_directive_text: str,
) -> Dict[int, Tuple[str, Dict[str, str], Dict[str, str]]]:
    """Runs the ranking judge model for each sample where multiple valid outputs exist."""
    judge_tasks = []
    judge_task_metadata = []
    print("Preparing judge tasks based on final workflow outputs...")

    for i, sample_data in enumerate(sample_workflow_results):
        # Access the pre-serialized judge input strings from the new key
        final_judge_inputs_for_sample = sample_data.get("final_judge_inputs", {})
        # Filter out any models whose final output resulted in an error string
        valid_outputs_for_judge = {
            model: output
            for model, output in final_judge_inputs_for_sample.items()
            if isinstance(output, str) and not output.startswith("ERROR:")
        }

        if len(valid_outputs_for_judge) > 1:
            # Use the imported create_judge_prompt
            judge_prompt, mask_map, original_map = create_judge_prompt(
                instruction_prompt=judge_directive_text,
                input_data=sample_data["input_data"],
                outputs=valid_outputs_for_judge,
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
    workflow: List[WorkflowStage],
    available_prompts: Dict[str, Tuple[str, str]],
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Aggregates ranking results, including prompt versions and unmasked rationale."""
    print("Processing judge results and aggregating final data (including versions)...")
    results_data = []
    all_task_ids_in_workflow = set(
        task["prompt"] for stage in workflow for task in stage.get("tasks", [])
    )
    judge_prompt_version = available_prompts.get(judge_prompt_name, ("N/A",))[1]

    for i, sample_data in enumerate(sample_workflow_results):
        judge_data = judge_response_map.get(i)
        ranking_masked, rationale, correct_models_masked = (None, None, None)
        mask_to_original_map = None
        original_to_mask_map = None
        judge_raw_response = None
        if judge_data:
            judge_raw_response, mask_to_original_map, original_to_mask_map = judge_data
            # Use the imported parse_judge_output
            ranking_masked, rationale, correct_models_masked = parse_judge_output(
                judge_raw_response
            )
        else:
            print(
                f"No judge response data found for sample {i}, likely skipped or failed."
            )

        ranking_original = None
        correct_models_original_set = None
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
            if correct_models_masked is not None:
                try:
                    correct_models_original_set = {
                        mask_to_original_map[masked]
                        for masked in correct_models_masked
                        if masked in mask_to_original_map
                    }
                except KeyError as e:
                    print(f"Error translating correct_models list for sample {i}: {e}")
                    correct_models_original_set = None
            elif judge_data:
                correct_models_original_set = None

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
            else "ERROR: Judge response not parsed"
            if judge_data
            else "Judge Skipped/Failed",
            "input": input_context,
            "input_length": input_length,
            "judge_rationale": unmasked_rationale
            if unmasked_rationale
            else "ERROR: Rationale not parsed"
            if judge_data
            else "Judge Skipped/Failed",
            "seed": effective_seed,
            "workflow_name": workflow_name,
            "judge_prompt_name": judge_prompt_name,
            "judge_prompt_version": judge_prompt_version,
        }

        rank_map = {model: -1 for model in models_to_test}
        if ranking_original:
            for rank, model in enumerate(ranking_original):
                if model in rank_map:
                    rank_map[model] = rank + 1

        model_outputs_all_tasks = sample_data.get("model_outputs", {})
        final_judge_inputs = sample_data.get("final_judge_inputs", {})

        for model in models_to_test:
            result_row[f"final_judge_input_{model}"] = final_judge_inputs.get(
                model, "N/A"
            )
            result_row[f"rank_{model}"] = rank_map.get(model, -1)

            model_correctness_flag: bool | str
            if model not in expected_models_in_judge_input:
                model_correctness_flag = "Not Judged (Input Error)"
            elif correct_models_original_set is not None:
                model_correctness_flag = model in correct_models_original_set
            elif judge_data:
                model_correctness_flag = "ERROR: Judge Parse Failed"
            else:
                model_correctness_flag = "ERROR: Judge Skipped/Failed"

            result_row[f"correct_{model}"] = model_correctness_flag

            model_exec_result = model_outputs_all_tasks.get(model, {})
            model_raw_outputs = model_exec_result.get("raw_outputs", {})
            model_validated_outputs = model_exec_result.get("validated_outputs", {})

            for task_id in all_task_ids_in_workflow:
                raw_col_name = f"raw_output_{task_id}_{model}"
                result_row[raw_col_name] = model_raw_outputs.get(task_id, "N/A")

                validated_col_name = f"validated_output_{task_id}_{model}"
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
                    result_row[validated_col_name] = validated_data
                elif validated_data is None:
                    result_row[validated_col_name] = "N/A"
                else:
                    result_row[validated_col_name] = (
                        f"ERROR: Unexpected data type {type(validated_data)}"
                    )

                task_version = available_prompts.get(task_id, ("", "N/A"))[1]
                result_row[f"version_{task_id}"] = task_version

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
        correct_col = f"correct_{model}"
        avg_rank = float("inf")
        num_valid_rank = 0
        correct_percent = 0.0
        num_judged_correctness = 0

        # Calculate Rank Stats
        if rank_col in results_df.columns:
            valid_ranks_df = results_df.filter(pl.col(rank_col) > 0)
            num_valid_rank = len(valid_ranks_df)
            if num_valid_rank > 0:
                avg_rank = valid_ranks_df[rank_col].mean()
        else:
            print(
                f"Warning: Rank column '{rank_col}' not found. Skipping rank calculation for {model}."
            )

        # Calculate Correctness Stats
        if correct_col in results_df.columns:
            # Filter out rows where judge was skipped, failed, or model wasn't in the judge input
            # Select only rows where the correctness value is explicitly True or False
            judged_correctness_df = results_df.filter(
                (pl.col(correct_col) == True) | (pl.col(correct_col) == False)
            )
            num_judged_correctness = len(judged_correctness_df)
            if num_judged_correctness > 0:
                true_count = judged_correctness_df.filter(
                    pl.col(correct_col) == True
                ).height
                correct_percent = (true_count / num_judged_correctness) * 100
        else:
            print(
                f"Warning: Correctness column '{correct_col}' not found. Skipping correctness calculation for {model}."
            )

        overall_leaderboard.append(
            (
                model,
                avg_rank,
                num_valid_rank,
                correct_percent,
                num_judged_correctness,
            )
        )

    overall_leaderboard.sort(key=lambda x: x[1])  # Sort by average rank

    header_line = f"--- Overall Leaderboard ({total_samples} Samples) ---"
    print(f"\n{header_line}")
    print("-" * len(header_line))
    print(
        f"{'Model':<40} {'Avg Rank':<10} {'Correct (%)':<12} {'# Ranked':<10} {'# Judged Correct'}"
    )
    print("-" * len(header_line))
    for i, (
        model,
        avg_rank,
        num_valid_rank,
        correct_percent,
        num_judged_correctness,
    ) in enumerate(overall_leaderboard):
        rank_str = f"{avg_rank:.2f}" if num_valid_rank > 0 else "N/A"
        correct_str = f"{correct_percent:.1f}%" if num_judged_correctness > 0 else "N/A"
        rank_count_str = f"{num_valid_rank}/{total_samples}"
        correct_count_str = f"{num_judged_correctness}/{total_samples}"
        print(
            f"{i + 1}. {model:<37} {rank_str:<10} {correct_str:<12} {rank_count_str:<10} {correct_count_str}"
        )
    print("-" * len(header_line))

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
