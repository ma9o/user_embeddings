import argparse
import asyncio

# import importlib.util # Removed
import json

# Keep sys.path modification for robustness when running script from different locations
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# import sys # Keep sys if sys.path modification is kept
import polars as pl
from dotenv import load_dotenv

# Import helper functions from the new module
from helpers.evaluation_utils import (
    aggregate_results,
    calculate_and_print_leaderboard,
    load_and_sample_data,
    run_and_parse_test_models,
    run_judge_evaluation,
    save_results,
)

from user_embeddings.utils.get_text_completion import (
    get_text_completion,
    initialize_openrouter_client,
)

# Import teacher prompts directly
from user_embeddings.utils.teacher_prompts import (
    all_in_one,
    inference,
    separation,
)

load_dotenv()

project_root = Path(__file__).resolve().parent.parent


# --- Prompt Mapping ---
# Assumes each prompt module has an INSTRUCTION_PROMPT variable
AVAILABLE_PROMPTS = {
    "all_in_one": all_in_one.PROMPT,
    "inference": inference.PROMPT,
    "separation": separation.PROMPT,
    # Add other prompts here
}

# --- Default Configuration ---
DEFAULT_MODELS_TO_TEST = [
    "google/gemma-3-27b-it",
    "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-chat-v3-0324",
    # "meta-llama/llama-4-maverick",
    # "google/gemini-2.5-pro-preview-03-25",
]
DEFAULT_JUDGE_MODEL = "google/gemini-2.5-pro-preview-03-25"
DEFAULT_NUM_SAMPLES = 10
DEFAULT_OUTPUT_DIR = Path("./data/test_results")
DEFAULT_SEED = None

# --- Argument Parser ---
parser = argparse.ArgumentParser(
    description="Evaluate LLM outputs based on a specific prompt."
)
parser.add_argument(
    "--prompt-module",
    type=str,
    required=True,
    # Update help text to reflect available keys
    help=f"Name of the prompt configuration to use. Available: {list(AVAILABLE_PROMPTS.keys())}",
)
parser.add_argument(
    "--input-data-file",
    type=Path,
    default=None,
    help="Path to a specific input data CSV file (must contain a 'formatted_context' column). If not provided, samples from all test_output_*.csv files in --input-data-dir.",
)
parser.add_argument(
    "--input-data-dir",
    type=Path,
    default=Path("./data/test_results"),
    help="Directory containing input data files (test_output_*.csv) which must contain a 'formatted_context' column, used when --input-data-file is not specified.",
)
parser.add_argument(
    "--models",
    nargs="+",
    default=DEFAULT_MODELS_TO_TEST,
    help="List of models to test.",
)
parser.add_argument(
    "--judge-model",
    type=str,
    default=DEFAULT_JUDGE_MODEL,
    help="Model to use for judging.",
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=DEFAULT_NUM_SAMPLES,
    help="Number of samples to evaluate from the input data.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="Directory to save the evaluation results.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help="Random seed for sampling. Defaults to current time if None.",
)


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
        print(f"Error parsing judge output: {e}\nRaw output:\n{judge_response}")
        return None, None


async def main():
    """Main function to orchestrate the LLM evaluation pipeline."""
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Get Instruction Prompt from mapping ---
    prompt_module_name = args.prompt_module
    if prompt_module_name not in AVAILABLE_PROMPTS:
        print(
            f"Error: Prompt module '{prompt_module_name}' not found. "
            f"Available prompts: {list(AVAILABLE_PROMPTS.keys())}"
        )
        return

    instruction_prompt = AVAILABLE_PROMPTS[prompt_module_name]
    print(f"Using instruction prompt: '{prompt_module_name}'")

    c = initialize_openrouter_client()

    # 1. Load and Sample Data
    effective_seed = args.seed if args.seed is not None else int(time.time())
    print(f"Using seed: {effective_seed}")

    # Determine input source and construct output filename
    if args.input_data_file:
        input_source_path = args.input_data_file
        if not input_source_path.is_file():
            print(f"Error: Specified input data file not found: {input_source_path}")
            await c.aclose()
            return
        input_data_stem = input_source_path.stem
        print(f"Using specific input file: {input_source_path}")
    else:
        input_source_path = args.input_data_dir
        if not input_source_path.is_dir():
            print(
                f"Error: Specified input data directory not found: {input_source_path}"
            )
            await c.aclose()
            return
        input_data_stem = (
            f"combined_{input_source_path.name}"  # Use dir name for combined output
        )
        print(f"Sampling from CSV files in directory: {input_source_path}")

    # Construct output filename with prompt name, input data name/source, and seed
    output_filename = (
        f"llm_eval_{prompt_module_name}_{input_data_stem}_seed_{effective_seed}.csv"
    )
    output_file_path = args.output_dir / output_filename

    # Modify load_and_sample_data call to use the determined source path
    # Assuming the helper function is updated to handle a file or directory path
    sample_df = load_and_sample_data(
        input_source_path, args.num_samples, effective_seed
    )
    if sample_df is None:
        await c.aclose()  # Close client if exiting early
        return  # Exit if no data

    # 2. Run Test Models and Parse Outputs
    # Pass instruction_prompt to the helper function
    sample_intermediate_results = await run_and_parse_test_models(
        sample_df,
        args.models,
        instruction_prompt,  # Pass instruction_prompt
    )

    # 3. Run Judge Model Evaluation
    # Pass instruction_prompt to the helper function
    judge_response_map = await run_judge_evaluation(
        sample_intermediate_results,
        args.judge_model,
        instruction_prompt,  # Pass instruction_prompt
    )

    # 4. Aggregate Final Results
    # Pass necessary args like seed, prompt name etc. if needed by the helper
    results_data = aggregate_results(
        sample_intermediate_results,
        judge_response_map,
        args.models,
        effective_seed,
        prompt_module_name,
    )
    results_df = pl.DataFrame(results_data)

    # 5. Save Results
    save_results(results_df, output_file_path)

    # 6. Calculate and Print Leaderboard
    calculate_and_print_leaderboard(results_df, args.models)

    print(f"Evaluation complete. Results saved to {output_file_path}")

    await c.aclose()


if __name__ == "__main__":
    asyncio.run(main())
