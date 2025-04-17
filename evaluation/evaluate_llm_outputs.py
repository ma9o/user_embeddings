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
        sample_df = full_df.sample(n=NUM_SAMPLES, shuffle=True, seed=42)

    print(f"Selected {len(sample_df)} rows for evaluation.")

    results_data = []

    # --- Run Evaluation ---
    for row in tqdm_asyncio(
        sample_df.iter_rows(named=True), total=len(sample_df), desc="Evaluating Samples"
    ):
        input_context = row["formatted_context"]
        instruction_prompt = get_teacher_prompt(
            input_context
        )  # Assuming this generates the prompt

        # Run models in parallel
        tasks = [run_model(model, instruction_prompt) for model in MODELS_TO_TEST]
        model_outputs_raw = await asyncio.gather(*tasks)
        raw_outputs_dict = dict(zip(MODELS_TO_TEST, model_outputs_raw))

        # Parse the final output from each model
        parsed_outputs_dict = {}
        for model, raw_output in raw_outputs_dict.items():
            if raw_output.startswith("ERROR:"):
                parsed_outputs_dict[model] = raw_output  # Keep error message
            else:
                try:
                    # Attempt to parse the structured output
                    # Assuming parse_teacher_prompt_output returns the relevant parsed string/object
                    # Convert to string for consistency in the judge prompt
                    parsed_output = parse_teacher_prompt_output(raw_output)
                    parsed_outputs_dict[model] = str(
                        parsed_output
                    )  # Ensure it's a string
                except Exception as parse_error:
                    print(f"Error parsing output from {model}: {parse_error}")
                    # Provide the raw output to the judge if parsing fails, clearly marked
                    parsed_outputs_dict[model] = (
                        f"ERROR PARSING OUTPUT: {parse_error}\nRAW OUTPUT:\n{raw_output}"
                    )

        # Prepare for judge using PARSED outputs
        judge_prompt = create_judge_prompt(
            instruction_prompt, input_context, parsed_outputs_dict
        )

        # Run judge model
        print(f"\nAsking judge ({JUDGE_MODEL}) for ranking...")
        judge_response = await run_model(JUDGE_MODEL, judge_prompt)
        ranking, rationale = parse_judge_output(judge_response)

        # Store results
        result_row = {
            "input": input_context,
            "judge_rationale": rationale
            if rationale
            else "ERROR: Rationale not parsed",
        }
        rank_map = {}
        if ranking:
            print(f"Judge ranking: {ranking}")
            if rationale:
                print(f"Judge rationale: {rationale}")
            else:
                print("Judge rationale: Not provided or parsing failed.")
            rank_map = {model: rank + 1 for rank, model in enumerate(ranking)}
        else:
            print("Could not parse judge ranking for sample. Assigning default ranks.")
            rank_map = {
                model: -1 for model in MODELS_TO_TEST
            }  # Use -1 to indicate error

        for model in MODELS_TO_TEST:
            # Store the RAW output for reference/debugging
            result_row[f"{model}_raw_output"] = raw_outputs_dict.get(
                model, "ERROR: Model raw output not found"
            )
            # Store the PARSED output that was judged
            result_row[f"{model}_parsed_output"] = parsed_outputs_dict.get(
                model, "ERROR: Model parsed output not found"
            )
            result_row[f"{model}_rank"] = rank_map.get(
                model, -1
            )  # Default to -1 if model missing from ranking

        results_data.append(result_row)

    # --- Save Results ---
    results_df = pl.DataFrame(results_data)
    print(f"\nSaving evaluation results to {OUTPUT_FILE}...")
    results_df.write_csv(OUTPUT_FILE)

    # --- Calculate and Print Leaderboard ---
    print("\n--- Final Leaderboard (Average Rank) ---")
    leaderboard = []
    total_samples = len(results_df)
    for model in MODELS_TO_TEST:
        rank_col = f"{model}_rank"
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
