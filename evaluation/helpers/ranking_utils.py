# import asyncio # Removed
import random
from typing import Any, Dict, List, Optional, Tuple

import dask  # Added
import polars as pl
from dask import delayed  # Added

# from tqdm.asyncio import tqdm_asyncio # Removed
# Import the async LLM runner, assuming it's still needed for the judge
# If workflow_executor._run_single_prompt is removed/changed, adapt this import
# We might need direct access to get_text_completion if _run_single_prompt changes
from user_embeddings.utils.llm.get_text_completion import (
    get_text_completion,  # Direct import might be safer
    initialize_openrouter_client,  # Correct path
)
from user_embeddings.utils.parsing import parse_llm_json_output

# --- Ranking Specific Helpers ---


# Step 1 (Delayed): Create the blinded prompt and mapping info
@delayed(pure=True)
def _create_judge_prompt_delayed(
    instruction_prompt: str,
    input_data: str,
    # Expects resolved outputs {model_name: output_string}
    outputs: Dict[str, str],
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """Delayed function to create a blinded prompt for the ranking judge."""
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


# Step 2 (Delayed): Run the judge LLM (using the async helper)
# This internal helper remains async
async def _run_single_judge_llm_async(model_name: str, prompt: str) -> str:
    """Async helper to run a single judge LLM call."""
    try:
        # Assuming get_text_completion handles initialization or it's done globally
        # Ensure client is initialized here too
        _ = initialize_openrouter_client()
        result = await get_text_completion(model_name, prompt)
        return result
    except Exception as e:
        print(f"Error running judge model {model_name}: {e}")
        return f"ERROR: Judge model execution failed - {e}"


# Dask wrapper for the async judge call
@delayed(pure=True)
def _run_judge_llm_dask_wrapper(judge_model: str, judge_prompt: str) -> str:
    """Dask wrapper to run the async judge LLM call using asyncio.run."""
    import asyncio  # Import locally if not top-level

    try:
        return asyncio.run(_run_single_judge_llm_async(judge_model, judge_prompt))
    except Exception as e:
        print(f"Error in Dask wrapper for judge model {judge_model}: {e}")
        return f"ERROR: Judge Dask wrapper failed - {e}"


# Step 3 (Delayed): Parse the judge's output
@delayed(pure=True)
def _parse_judge_output_delayed(
    judge_response: str,
) -> Tuple[Optional[List[str]], Optional[str], Optional[bool]]:
    """Delayed function to parse the ranking judge's JSON response."""
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


# --- New Graph Building Function ---
# This function defines the judge part of the Dask graph for one sample.
# It takes delayed objects for workflow outputs as input.
def build_ranking_judge_graph(
    # Delayed object representing the input string
    input_data_delayed: dask.delayed,
    # Dict mapping model_name -> delayed object representing the final output string for judge
    model_outputs_delayed: Dict[str, dask.delayed],
    judge_model: str,
    judge_instruction_prompt_text: str,  # This is static text, not delayed
) -> Dict[str, dask.delayed]:
    """Builds the Dask graph segment for ranking judge evaluation for a single sample.

    Returns a dictionary containing delayed objects for:
        - 'judge_raw_output': Raw string output from the judge LLM.
        - 'judge_parsed_ranking': Parsed list of masked model ranks.
        - 'judge_parsed_rationale': Parsed rationale string.
        - 'judge_parsed_any_correct': Parsed boolean correctness flag.
        - 'judge_mask_map': Dictionary mapping masked name -> original name.
        - 'judge_original_map': Dictionary mapping original name -> masked name.
    """
    print(f"DEBUG: Building judge graph. Judge model: {judge_model}")
    print(f"DEBUG: Judge instruction: {judge_instruction_prompt_text[:50]}...")
    print(f"DEBUG: Input data delayed: {input_data_delayed}")
    print(f"DEBUG: Model outputs delayed: {model_outputs_delayed}")

    if not model_outputs_delayed or len(model_outputs_delayed) < 2:
        print("DEBUG: Skipping judge graph build - insufficient models/outputs.")
        # Return delayed objects representing a skipped state
        return {
            "judge_raw_output": delayed("Judge Skipped - Insufficient valid outputs"),
            "judge_parsed_ranking": delayed(None),
            "judge_parsed_rationale": delayed(None),
            "judge_parsed_any_correct": delayed(None),
            "judge_mask_map": delayed({}),
            "judge_original_map": delayed({}),
        }

    # Step 1: Create the prompt (depends on resolved input_data and model_outputs)
    # Dask automatically passes the resolved values of the delayed objects here.
    prompt_details_delayed = _create_judge_prompt_delayed(
        instruction_prompt=judge_instruction_prompt_text,
        input_data=input_data_delayed,
        outputs=model_outputs_delayed,  # Dask resolves the dict values
    )

    # Extract the individual delayed elements from the tuple returned by _create_judge_prompt_delayed
    # We need helper functions or direct tuple indexing within delayed calls if possible,
    # or compute the tuple and then use its elements.
    # Let's use dask's built-in itemgetter for delayed objects.
    judge_prompt_delayed = delayed(lambda x: x[0])(prompt_details_delayed)
    mask_map_delayed = delayed(lambda x: x[1])(prompt_details_delayed)
    original_map_delayed = delayed(lambda x: x[2])(prompt_details_delayed)

    # Step 2: Run the judge LLM (depends on the delayed prompt)
    judge_raw_output_delayed = _run_judge_llm_dask_wrapper(
        judge_model=judge_model, judge_prompt=judge_prompt_delayed
    )

    # Step 3: Parse the output (depends on the delayed raw output)
    parsed_results_delayed = _parse_judge_output_delayed(
        judge_response=judge_raw_output_delayed
    )

    # Extract individual parsed elements
    parsed_ranking_delayed = delayed(lambda x: x[0])(parsed_results_delayed)
    parsed_rationale_delayed = delayed(lambda x: x[1])(parsed_results_delayed)
    parsed_any_correct_delayed = delayed(lambda x: x[2])(parsed_results_delayed)

    # Return a dictionary of all the relevant delayed objects for this judge task
    return {
        "judge_raw_output": judge_raw_output_delayed,
        "judge_parsed_ranking": parsed_ranking_delayed,
        "judge_parsed_rationale": parsed_rationale_delayed,
        "judge_parsed_any_correct": parsed_any_correct_delayed,
        "judge_mask_map": mask_map_delayed,
        "judge_original_map": original_map_delayed,
    }


# --- Deprecated Synchronous Function ---
def run_judge_evaluation(*args, **kwargs):
    # This function is now replaced by building the graph segment
    # and computing it as part of the main Dask graph.
    raise DeprecationWarning(
        "run_judge_evaluation is deprecated. Use build_ranking_judge_graph and dask.compute."
    )


# --- Aggregation Function --- (Needs updates to use delayed results)
def aggregate_ranking_results(
    # This function now operates on the *computed* results after dask.compute
    # Structure per sample: {'input_data': str, 'workflow_outputs': {model: {task: TaskResult}}, 'judge_results': {key: computed_value}}
    computed_results_list: List[Dict[str, Any]],
    models: List[str],
    seed: int,
    workflow_name: str,
    judge_prompt_name: str,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Aggregates ranking results from the computed Dask graph output."""
    print("Aggregating final results including judge outputs...")
    results_data = []

    for i, computed_sample in enumerate(computed_results_list):
        input_context = computed_sample.get("input_data", "ERROR: Input missing")
        workflow_outputs = computed_sample.get("workflow_outputs", {})
        judge_results = computed_sample.get(
            "judge_results", {}
        )  # Contains computed judge outputs

        # Extract computed judge results
        judge_raw_response = judge_results.get(
            "judge_raw_output", "Judge Skipped/Failed"
        )
        ranking_masked = judge_results.get("judge_parsed_ranking")
        rationale = judge_results.get("judge_parsed_rationale")
        any_correct = judge_results.get("judge_parsed_any_correct")
        mask_to_original_map = judge_results.get("judge_mask_map", {})
        original_to_mask_map = judge_results.get("judge_original_map", {})

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
            if (
                debug
                and not original_to_mask_map
                and judge_raw_response != "Judge Skipped/Failed"
            ):
                print(
                    f"DEBUG: original_to_mask_map is missing for sample {i}, although judge_raw_response exists."
                )
            if debug and not rationale and judge_raw_response != "Judge Skipped/Failed":
                print(
                    f"DEBUG: Rationale is missing for sample {i}, although judge_raw_response exists."
                )

        result_row: Dict[str, Any] = {
            "judge_raw_output": judge_raw_response,
            "input": input_context,
            "input_length": input_length,
            "judge_rationale": unmasked_rationale
            if unmasked_rationale
            else (
                "ERROR: Rationale missing/unmask failed"
                if judge_raw_response != "Judge Skipped/Failed"
                else "Judge Skipped/Failed"
            ),
            "judge_any_correct": any_correct
            if any_correct is not None
            else (
                "ERROR: Parse failed"
                if judge_raw_response != "Judge Skipped/Failed"
                else "Judge Skipped/Failed"
            ),
            "seed": seed,
            "workflow_name": workflow_name,
            "judge_prompt_name": judge_prompt_name,
        }

        rank_map = {model: -1 for model in models}
        if ranking_original:
            for rank, model in enumerate(ranking_original):
                if model in rank_map:
                    rank_map[model] = rank + 1

        # Add individual model ranks and potentially workflow outputs
        for model in models:
            result_row[f"rank_{model}"] = rank_map.get(model, -1)
            # Extract final workflow output string for this model (assuming single final task)
            # This part needs refinement based on how workflow outputs are structured
            model_workflow_output = "N/A"  # Placeholder
            model_tasks = workflow_outputs.get(model, {})
            # Simple approach: find the last task based on name or structure? Needs better way.
            # Or maybe the caller prepares the judge_inputs string beforehand?
            # For now, let's rely on judge_inputs being prepared elsewhere or store raw.
            # Get the prepared judge input string for this model if available?
            # This relies on judge_inputs being part of computed_sample, which it isn't currently.
            # Let's just log rank for now.
            # result_row[f"output_{model}"] = computed_sample.get("judge_inputs", {}).get(model, "N/A")

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
