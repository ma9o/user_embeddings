import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
from dotenv import load_dotenv

from user_embeddings.utils.get_text_completion import (
    get_text_completion,
    initialize_openrouter_client,
)

# Import helper functions from the new module
from .helpers.evaluation_utils import (
    aggregate_results,
    calculate_and_print_leaderboard,
    load_and_sample_data,
    run_and_parse_test_models,
    run_judge_evaluation,
    save_results,
)

load_dotenv()

# --- Configuration ---
MODELS_TO_TEST = [
    "google/gemma-3-27b-it",
    "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-chat-v3-0324",
    # "meta-llama/llama-4-maverick",
    # "google/gemini-2.5-pro-preview-03-25",
]
JUDGE_MODEL = "google/gemini-2.5-pro-preview-03-25"
NUM_SAMPLES = 10
INPUT_DATA_DIR = Path("./data/test_results")
OUTPUT_FILE = Path("./data/test_results/llm_evaluation_results.csv")
SEED = None  # Set to None for random sampling


# --- Helper Functions ---
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
    prompt = "You are an expert evaluator tasked with ranking the quality of different Large Language Model (LLM) outputs based on a given instruction and input.\\n\\n"
    prompt += (
        f"INSTRUCTION PROMPT GIVEN TO MODELS:\\n---\\n{instruction_prompt}\\n---\\n\\n"
    )
    prompt += f"INPUT DATA GIVEN TO MODELS:\\n---\\n{input_data}\\n---\\n\\n"
    prompt += 'LLM OUTPUTS TO EVALUATE:\\n---"'
    for i, (model_name, output) in enumerate(outputs.items()):
        prompt += f"\\nOutput {i + 1} (Model: {model_name}):\\n{output}\\n---"

    prompt += "\\n\\nTASK:\\nEvaluate the outputs based *only* on how well they follow the INSTRUCTION PROMPT for the given INPUT DATA. Consider clarity, structure, adherence to format, and accuracy of the generated summary/actions based *solely* on the provided input context.\\n\\n"
    prompt += "RANKING FORMAT:\\nProvide your ranking as a JSON object containing two keys: 'ranking' (a list of model names, ordered from best to worst) and 'rationale' (a brief explanation for your ranking decisions). For example:\\n"
    prompt += (
        "```json\\n"
        "{\\n"
        '  "ranking": ["model_name_best", "model_name_middle", "model_name_worst"],\\n'
        '  "rationale": "Model A was best because... Model B struggled with... Model C failed to..."\\n'
        "}\\n"
        "```\\n"
    )
    prompt += f"The available model names are: {list(outputs.keys())}. Return ONLY the JSON object and nothing else."

    return prompt


def parse_judge_output(
    judge_response: str,
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Parses the JSON ranking and rationale from the judge's response."""
    try:
        # Extract JSON block if necessary
        if "```json" in judge_response:
            json_str = judge_response.split("```json\\n")[1].split("\\n```")[0]
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
        print(f"Error parsing judge output: {e}\\nRaw output:\\n{judge_response}")
        return None, None


async def main():
    """Main function to orchestrate the LLM evaluation pipeline."""
    c = initialize_openrouter_client()

    # 1. Load and Sample Data
    sample_df = load_and_sample_data(INPUT_DATA_DIR, NUM_SAMPLES, SEED)
    if sample_df is None:
        return  # Exit if no data

    # 2. Run Test Models and Parse Outputs
    sample_intermediate_results = await run_and_parse_test_models(
        sample_df, MODELS_TO_TEST
    )

    # 3. Run Judge Model Evaluation
    judge_response_map = await run_judge_evaluation(
        sample_intermediate_results, JUDGE_MODEL
    )

    # 4. Aggregate Final Results
    results_data = aggregate_results(
        sample_intermediate_results, judge_response_map, MODELS_TO_TEST
    )
    results_df = pl.DataFrame(results_data)

    # 5. Save Results
    save_results(results_df, OUTPUT_FILE)

    # 6. Calculate and Print Leaderboard
    calculate_and_print_leaderboard(results_df, MODELS_TO_TEST)

    print("Evaluation complete.")

    await c.aclose()


if __name__ == "__main__":
    asyncio.run(main())
