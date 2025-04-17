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
from user_embeddings.utils.teacher_prompt import get_teacher_prompt

load_dotenv()

# --- Configuration ---
MODELS_TO_TEST = [
    "google/gemma-3-27b-it",
    "deepseek/deepseek-r1-distill-llama-70b",
    "meta-llama/llama-4-maverick",
    "google/gemini-2.5-pro-preview-03-25",
]
JUDGE_MODEL = "google/gemini-2.5-pro-preview-03-25"
NUM_SAMPLES = 10
INPUT_DATA_DIR = Path("../data/test_results")
OUTPUT_FILE = Path("../data/test_results/llm_evaluation_results.csv")


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
    prompt += "RANKING FORMAT:\nProvide your ranking as a JSON list of model names, ordered from best to worst. For example:\n"
    prompt += (
        '```json\n["model_name_best", "model_name_middle", "model_name_worst"]\n```\n'
    )
    prompt += f"The available model names are: {list(outputs.keys())}. Return ONLY the JSON list and nothing else."

    return prompt


def parse_judge_output(judge_response: str) -> list[str] | None:
    """Parses the JSON ranking list from the judge's response."""
    try:
        # Extract JSON block if necessary
        if "```json" in judge_response:
            json_str = judge_response.split("```json\n")[1].split("\n```")[0]
        else:
            json_str = judge_response

        ranking = json.loads(json_str)
        if isinstance(ranking, list) and all(isinstance(item, str) for item in ranking):
            return ranking
        else:
            print(f"Error: Judge output is not a list of strings: {ranking}")
            return None
    except (json.JSONDecodeError, IndexError, TypeError) as e:
        print(f"Error parsing judge output: {e}\nRaw output:\n{judge_response}")
        return None


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
        model_outputs = await asyncio.gather(*tasks)
        outputs_dict = dict(zip(MODELS_TO_TEST, model_outputs))

        # Prepare for judge
        judge_prompt = create_judge_prompt(
            instruction_prompt, input_context, outputs_dict
        )

        # Run judge model
        print(f"\nAsking judge ({JUDGE_MODEL}) for ranking...")
        judge_response = await run_model(JUDGE_MODEL, judge_prompt)
        ranking = parse_judge_output(judge_response)

        # Store results
        result_row = {"input": input_context}
        rank_map = {}
        if ranking:
            print(f"Judge ranking: {ranking}")
            rank_map = {model: rank + 1 for rank, model in enumerate(ranking)}
        else:
            print("Could not parse judge ranking for sample. Assigning default ranks.")
            rank_map = {
                model: -1 for model in MODELS_TO_TEST
            }  # Use -1 to indicate error

        for model in MODELS_TO_TEST:
            result_row[f"{model}_output"] = outputs_dict.get(
                model, "ERROR: Model not found"
            )
            result_row[f"{model}_rank"] = rank_map.get(
                model, -1
            )  # Default to -1 if model missing from ranking

        results_data.append(result_row)

    # --- Save Results ---
    results_df = pl.DataFrame(results_data)
    print(f"\nSaving evaluation results to {OUTPUT_FILE}...")
    results_df.write_csv(OUTPUT_FILE)
    print("Evaluation complete.")


if __name__ == "__main__":
    asyncio.run(main())
