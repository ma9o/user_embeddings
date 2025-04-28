import argparse
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import polars as pl
from dotenv import load_dotenv

# Import helpers
from helpers.evaluation_utils import (
    aggregate_results,
    calculate_and_print_leaderboard,
    load_and_sample_data,
    run_and_parse_test_models,
    run_judge_evaluation,
    save_results,
)

from user_embeddings.utils.get_text_completion import initialize_openrouter_client

# Import teacher prompts
from user_embeddings.utils.teacher_prompts import (
    all_in_one,
    inference,
    intent_only,
    koa_only,
    separation,
)
from user_embeddings.utils.teacher_prompts import intent_only as intent_only_module

# Import Pydantic models for output validation
from user_embeddings.utils.teacher_prompts import koa_only as koa_only_module

# Import workflow utilities
from user_embeddings.utils.workflow_executor import (
    DEFAULT_INPUT_FORMATTERS,  # Import default formatters
    PromptStage,  # Import type if needed, though WORKFLOWS uses it implicitly
    validate_workflow,  # Import validator
)

load_dotenv()
project_root = Path(__file__).resolve().parent.parent

# --- Prompt Mapping ---
AVAILABLE_PROMPTS = {
    "all_in_one": all_in_one.PROMPT,
    "inference": inference.PROMPT,
    "separation": separation.PROMPT,
    "intent_only": intent_only.PROMPT,
    "koa_only": koa_only.PROMPT,
}

# --- Pydantic Output Model Mapping --- Map prompt names to their validation models
AVAILABLE_OUTPUT_MODELS = {
    "koa_only": koa_only_module.PromptOutput,
    "intent_only": intent_only_module.PromptOutput,
    # Add other models here if defined
    # "separation": separation_module.PromptOutput,
    # "inference": inference_module.PromptOutput,
    # "all_in_one": all_in_one_module.PromptOutput,
}

# --- Workflow Definition ---
WORKFLOWS: Dict[str, List[PromptStage]] = {
    "serial_separation_inference": [
        {
            "stage": 1,
            "prompts": ["separation"],
            "input_from": None,
            "input_formatter": None,
        },
        {
            "stage": 2,
            "prompts": ["inference"],
            "input_from": ["separation"],
            "input_formatter": "format_single_input",
        },
    ],
    "concurrent_intent_koa": [
        {
            "stage": 1,
            "prompts": ["intent_only", "koa_only"],
            "input_from": None,
            "input_formatter": None,
        },
    ],
}

# --- Default Configuration ---
DEFAULT_MODELS_TO_TEST = [
    # "deepseek/deepseek-r1-distill-llama-70b",
    # "deepseek/deepseek-chat-v3-0324",
    "google/gemma-3-27b-it",
    "google/gemini-2.5-flash-preview",
]
DEFAULT_JUDGE_MODEL = "google/gemini-2.5-pro-preview-03-25"
DEFAULT_NUM_SAMPLES = 10
DEFAULT_OUTPUT_DIR = Path("./data/test_results")
DEFAULT_SEED = None

# --- Argument Parser ---
parser = argparse.ArgumentParser(
    description="Evaluate LLM outputs based on a defined workflow."
)
parser.add_argument(
    "--workflow",
    type=str,
    required=True,
    choices=list(WORKFLOWS.keys()),
    help="Name of the evaluation workflow to run.",
)
parser.add_argument(
    "--judge-prompt-module",
    type=str,
    default=None,
    help=f"Name of the prompt configuration for the judge model. If not provided and only one test prompt is given, uses the test prompt. Available: {list(AVAILABLE_PROMPTS.keys())}",
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
parser.add_argument(
    "--debug",
    action="store_true",  # Makes it a flag, default is False
    help="Enable debug printing for steps like rationale unmasking.",
)


async def main():
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_workflow_name = args.workflow
    selected_workflow = WORKFLOWS[selected_workflow_name]
    print(f"Using workflow: '{selected_workflow_name}'")

    # --- Validate Workflow using the imported function ---
    # Use DEFAULT_INPUT_FORMATTERS imported from workflow_executor
    is_valid = validate_workflow(
        workflow_name=selected_workflow_name,
        workflow_definition=selected_workflow,
        available_prompts=AVAILABLE_PROMPTS,
        available_formatters=DEFAULT_INPUT_FORMATTERS,  # Use imported default formatters
        # Pass the output models for validation if needed within the validator (optional extension)
    )
    if not is_valid:
        print("Workflow validation failed. Exiting.")
        return

    # --- Determine Judge Prompt ---
    judge_prompt_module_name = args.judge_prompt_module
    if not judge_prompt_module_name:
        last_stage = selected_workflow[-1]
        if len(last_stage["prompts"]) == 1:
            judge_prompt_module_name = last_stage["prompts"][0]
            print(
                f"Judge prompt not specified, defaulting to: '{judge_prompt_module_name}'"
            )
        else:
            print(
                "Error: --judge-prompt-module is required when the workflow's final stage has multiple prompts."
            )
            return
    if judge_prompt_module_name not in AVAILABLE_PROMPTS:
        print(f"Error: Judge prompt module '{judge_prompt_module_name}' not found.")
        return
    judge_instruction_prompt = AVAILABLE_PROMPTS[judge_prompt_module_name]
    print(f"Using judge prompt module: '{judge_prompt_module_name}'")

    c = initialize_openrouter_client()

    # 1. Load and Sample Data
    effective_seed = args.seed if args.seed is not None else int(time.time())
    print(f"Using seed: {effective_seed}")

    # Determine input source and construct output filename (Input source logic remains same)
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

    # Construct output filename with workflow name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"llm_eval_judge-{judge_prompt_module_name}_workflow-{selected_workflow_name}_{input_data_stem}_seed_{effective_seed}_{timestamp}.csv"
    output_file_path = args.output_dir / output_filename

    # Load data (using existing helper)
    sample_df = load_and_sample_data(
        input_source_path, args.num_samples, effective_seed
    )
    if sample_df is None:
        await c.aclose()
        return

    # 2. Run Test Models according to Workflow
    # Pass the workflow definition, available prompts, and formatters
    sample_workflow_results = await run_and_parse_test_models(
        sample_df,
        args.models,
        selected_workflow,
        AVAILABLE_PROMPTS,
        DEFAULT_INPUT_FORMATTERS,  # Pass the default formatters
        AVAILABLE_OUTPUT_MODELS,  # Pass the output model mapping
    )

    # 3. Run Judge Model Evaluation
    # Pass the single judge instruction prompt
    judge_response_map = await run_judge_evaluation(
        sample_workflow_results,  # Updated results structure from workflow run
        args.judge_model,
        judge_instruction_prompt,
    )

    # 4. Aggregate Final Results
    results_data = aggregate_results(
        sample_workflow_results,
        judge_response_map,
        args.models,
        effective_seed,
        selected_workflow_name,  # Pass workflow name
        judge_prompt_module_name,
        selected_workflow,  # Pass workflow definition for context if needed
        debug=args.debug,  # Pass the debug flag
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
