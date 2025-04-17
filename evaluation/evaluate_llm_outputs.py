import asyncio
import json
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from user_embeddings.utils.get_text_completion import (
    get_text_completion,
    initialize_openrouter_client,
)
from user_embeddings.utils.teacher_prompt import (
    get_teacher_prompt,
    parse_teacher_prompt_output,
)

load_dotenv()

# --- Configuration ---
MODELS_TO_TEST = [
    "google/gemma-3-27b-it",
    "deepseek/deepseek-r1-distill-llama-70b",
    "meta-llama/llama-4-maverick",
]
JUDGE_MODEL = "google/gemini-2.5-pro-preview-03-25"
NUM_SAMPLES = 5
INPUT_DATA_DIR = Path("./data/test_results")
OUTPUT_FILE = Path("./data/test_results/llm_evaluation_results.csv")
SEED = None  # Set to None for random sampling


# --- Helper Functions ---
async def run_model(model_name: str, prompt: str):
    """Runs a single model and returns its output."""
    try:
        result = await get_text_completion(model_name, prompt)
        return result
    except Exception as e:
        print(f"Error running model {model_name}: {e}")
        return f"ERROR: {e}"


def create_judge_prompt(
    instruction_prompt: str, input_data: str, outputs: dict[str, str]
) -> str:
    """Creates the prompt for the judge LLM."""
    prompt = "You are an expert evaluator tasked with ranking the quality of different Large Language Model (LLM) outputs based on a given instruction and input.\n\n"
    prompt += f"INSTRUCTION PROMPT GIVEN TO MODELS:\n---\n{instruction_prompt}\n---\n\n"
    prompt += f"INPUT DATA GIVEN TO MODELS:\n---\n{input_data}\n---\n\n"
    prompt += 'LLM OUTPUTS TO EVALUATE:\n---"'
    for i, (model_name, output) in enumerate(outputs.items()):
        prompt += f"\nOutput {i + 1} (Model: {model_name}):\n{output}\n---"

    prompt += "\n\nTASK:\nEvaluate the outputs based *only* on how well they follow the INSTRUCTION PROMPT for the given INPUT DATA. Consider clarity, structure, adherence to format, and accuracy of the generated summary/actions based *solely* on the provided input context.\n\n"
    prompt += "RANKING FORMAT:\nProvide your ranking as a JSON object containing two keys: 'ranking' (a list of model names, ordered from best to worst) and 'rationale' (a brief explanation for your ranking decisions). For example:\n"
    prompt += (
        "```json\n"
        "{\n"
        '  "ranking": ["model_name_best", "model_name_middle", "model_name_worst"],\n'
        '  "rationale": "Model A was best because... Model B struggled with... Model C failed to..."\n'
        "}\n"
        "```\n"
    )
    prompt += f"The available model names are: {list(outputs.keys())}. Return ONLY the JSON object and nothing else."

    return prompt


def parse_judge_output(judge_response: str) -> tuple[list[str] | None, str | None]:
    """Parses the JSON ranking and rationale from the judge's response."""
    try:
        # Extract JSON block if necessary
        if "```json" in judge_response:
            json_str = judge_response.split("```json\n")[1].split("\n```")[0]
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
        print(f"Error parsing judge output: {e}\nRaw output:\n{judge_response}")
        return None, None


async def main():
    initialize_openrouter_client()
    all_files = list(INPUT_DATA_DIR.glob("test_output_*.csv"))
    if not all_files:
        print(f"No CSV files found in {INPUT_DATA_DIR}")
        return

    # --- Load and Sample Data ---
    print(f"Loading data from {INPUT_DATA_DIR}...")
    df_list = [pl.read_csv(f) for f in all_files]
    full_df = pl.concat(df_list)
    print(f"Total rows loaded: {len(full_df)}")

    if len(full_df) < NUM_SAMPLES:
        print(
            f"Warning: Not enough data ({len(full_df)} rows) for {NUM_SAMPLES} samples. Using all available data."
        )
        sample_df = full_df
    else:
        sample_df = full_df.sample(n=NUM_SAMPLES, shuffle=True, seed=SEED)

    print(f"Selected {len(sample_df)} rows for evaluation.")

    # --- Prepare Test Model Tasks ---
    test_model_tasks = []
    # Store sample info along with task for later mapping
    task_metadata = []  # Stores dicts: {sample_index, model, input_context, instruction_prompt}
    print("Preparing test model tasks...")
    for i, row in enumerate(sample_df.iter_rows(named=True)):
        input_context = row["formatted_context"]
        # Generate instruction prompt once per sample
        instruction_prompt = get_teacher_prompt(input_context)
        for model in MODELS_TO_TEST:
            # Create task for each model x sample combination
            task = asyncio.create_task(run_model(model, instruction_prompt))
            test_model_tasks.append(task)
            task_metadata.append(
                {
                    "sample_index": i,
                    "model": model,
                    "input_context": input_context,  # Store for later use
                    "instruction_prompt": instruction_prompt,  # Store for later use
                }
            )

    # --- Run Test Models Concurrently ---
    print(f"Running {len(test_model_tasks)} test model tasks concurrently...")
    test_model_results_raw = await tqdm_asyncio.gather(
        *test_model_tasks, desc="Running Test Models", unit="task"
    )

    # --- Organize and Parse Test Model Results & Prepare Judge Prompts ---
    print("Organizing test results and preparing judge prompts...")
    # Store intermediate results indexed by sample_index
    # Each element will store: {input_context, instruction_prompt, raw_outputs, parsed_outputs}
    sample_intermediate_results = [{} for _ in range(len(sample_df))]

    for i, raw_output in enumerate(test_model_results_raw):
        meta = task_metadata[i]
        sample_index = meta["sample_index"]
        model = meta["model"]

        # Initialize sample data if first time seeing this index
        if not sample_intermediate_results[sample_index]:
            sample_intermediate_results[sample_index] = {
                "input_context": meta["input_context"],
                "instruction_prompt": meta["instruction_prompt"],
                "raw_outputs": {},
                "parsed_outputs": {},
            }

        # Store raw output
        sample_intermediate_results[sample_index]["raw_outputs"][model] = raw_output

        # Parse output
        if raw_output.startswith("ERROR:"):
            parsed_output_str = raw_output
        else:
            try:
                parsed_output = parse_teacher_prompt_output(raw_output)
                parsed_output_str = str(parsed_output)
            except Exception as parse_error:
                # Log error without stopping execution for other models/samples
                print(
                    f"Error parsing output from {model} for sample {sample_index}: {parse_error}"
                )
                parsed_output_str = (
                    f"ERROR PARSING OUTPUT: {parse_error}\nRAW OUTPUT:\n{raw_output}"
                )
        sample_intermediate_results[sample_index]["parsed_outputs"][model] = (
            parsed_output_str
        )

    # --- Prepare Judge Tasks ---
    judge_tasks = []
    judge_task_sample_indices = []  # Keep track of which sample index maps to which task
    print("Preparing judge tasks...")
    for i, intermediate_data in enumerate(sample_intermediate_results):
        # Check if we have parsed outputs (might be missing if all models failed for a sample)
        if (
            "parsed_outputs" in intermediate_data
            and intermediate_data["parsed_outputs"]
        ):
            judge_prompt = create_judge_prompt(
                intermediate_data["instruction_prompt"],
                intermediate_data["input_context"],
                intermediate_data["parsed_outputs"],
            )
            task = asyncio.create_task(run_model(JUDGE_MODEL, judge_prompt))
            judge_tasks.append(task)
            judge_task_sample_indices.append(
                i
            )  # Map task back to original sample index
        else:
            print(
                f"Skipping judge task for sample {i} due to missing/empty parsed outputs."
            )
            # We still need to handle this sample later when aggregating results

    # --- Run Judge Models Concurrently ---
    judge_responses = []
    if judge_tasks:
        print(f"Running {len(judge_tasks)} judge tasks concurrently...")
        judge_responses = await tqdm_asyncio.gather(
            *judge_tasks, desc="Running Judge Models", unit="task"
        )
    else:
        print("No judge tasks to run.")

    # --- Process Judge Results and Aggregate Final Data ---
    print("Processing judge results and aggregating final data...")
    results_data = []
    judge_response_map = dict(zip(judge_task_sample_indices, judge_responses))

    for i, intermediate_data in enumerate(sample_intermediate_results):
        judge_response = judge_response_map.get(
            i
        )  # Get response using original sample index
        ranking = None
        rationale = None

        if judge_response:
            ranking, rationale = parse_judge_output(judge_response)
        else:
            # Handle cases where judge task was skipped or failed implicitly
            print(f"No judge response found for sample {i}, likely skipped or failed.")

        # Prepare final result row for this sample
        result_row = {
            "input": intermediate_data["input_context"],
            "judge_rationale": rationale
            if rationale
            else "ERROR: Rationale not parsed or judge skipped",
        }

        rank_map = {}
        if ranking:
            # Validate ranking contains expected models
            expected_models = set(MODELS_TO_TEST)
            actual_models = set(ranking)
            if actual_models == expected_models:
                rank_map = {model: rank + 1 for rank, model in enumerate(ranking)}
            else:
                print(
                    f"Warning: Judge ranking for sample {i} ({ranking}) \
                    does not match/contain all MODELS_TO_TEST ({list(expected_models)}). \
                    Assigning default ranks (-1)."
                )
                # Assign -1 to all expected models, even if some were ranked
                rank_map = {model: -1 for model in MODELS_TO_TEST}
        else:
            # Assign default rank if ranking is None (parsing failed or judge skipped)
            # print(f"Could not parse judge ranking for sample {i}. Assigning default ranks (-1).") # Can be verbose
            rank_map = {model: -1 for model in MODELS_TO_TEST}

        # Add model-specific outputs and ranks
        for model in MODELS_TO_TEST:
            result_row[f"raw_output_{model}"] = intermediate_data.get(
                "raw_outputs", {}
            ).get(model, "ERROR: Model raw output not found")
            result_row[f"parsed_output_{model}"] = intermediate_data.get(
                "parsed_outputs", {}
            ).get(model, "ERROR: Model parsed output not found")
            # Use the rank_map determined above
            result_row[f"rank_{model}"] = rank_map.get(model, -1)

        results_data.append(result_row)

    # --- Save Results ---
    results_df = pl.DataFrame(results_data)
    print(f"\nSaving evaluation results to {OUTPUT_FILE}...")

    # Order the columns alphabetically and save
    results_df = results_df.select(sorted(results_df.columns))
    results_df.write_csv(OUTPUT_FILE)

    # --- Calculate and Print Leaderboard ---
    print("\n--- Final Leaderboard (Average Rank) ---")
    leaderboard = []
    total_samples = len(results_df)
    for model in MODELS_TO_TEST:
        rank_col = f"rank_{model}"
        # Ensure the rank column exists before trying to access it
        if rank_col in results_df.columns:
            valid_ranks = results_df.filter(pl.col(rank_col) != -1)[rank_col]
            num_valid = len(valid_ranks)
            if num_valid > 0:
                avg_rank = valid_ranks.mean()
                leaderboard.append((model, avg_rank, num_valid))
            else:
                leaderboard.append(
                    (model, float("inf"), 0)
                )  # Assign infinity rank if no valid runs
        else:
            print(
                f"Warning: Rank column '{rank_col}' not found in results. Skipping model {model} for leaderboard."
            )
            leaderboard.append((model, float("inf"), 0))  # Treat as if no valid runs

    # Sort by average rank (ascending), lower is better
    leaderboard.sort(key=lambda x: x[1])

    header_line = "--- Final Leaderboard (Average Rank) ---"
    print("-" * len(header_line))  # Match header length for separator

    for i, (model, avg_rank, num_valid) in enumerate(leaderboard):
        rank_str = f"{avg_rank:.2f}" if num_valid > 0 else "N/A"
        print(
            f"{i + 1}. {model}: Avg Rank = {rank_str} ({num_valid}/{total_samples} valid runs)"
        )

    print("-" * len(header_line))  # Footer separator
    print("Evaluation complete.")


if __name__ == "__main__":
    asyncio.run(main())
