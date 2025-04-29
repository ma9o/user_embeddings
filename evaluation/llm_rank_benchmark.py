import argparse
import asyncio
import time
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

# Import shared configurations
from evaluation.config import (
    AVAILABLE_OUTPUT_MODELS,
    AVAILABLE_PROMPTS,
    WORKFLOWS,
)
from evaluation.helpers.common_args import add_common_eval_args

# Import shared and specific helpers
from evaluation.helpers.evaluation_utils import (
    load_and_sample_data,
    run_and_parse_test_models,
    save_results,  # Still used here
)
from evaluation.helpers.filename_utils import generate_eval_filename
from evaluation.helpers.ranking_utils import (
    aggregate_ranking_results,
    calculate_and_print_leaderboard,
    run_judge_evaluation,
)
from user_embeddings.utils.llm.get_text_completion import initialize_openrouter_client
from user_embeddings.utils.llm.workflow_executor import (
    DEFAULT_INPUT_FORMATTERS,
    validate_workflow,
)

# No longer need direct imports for prompts/models used only in config
# from user_embeddings.utils.teacher_prompts import ...
# from user_embeddings.utils.teacher_prompts import ... as ..._module

load_dotenv()
project_root = Path(__file__).resolve().parent.parent

# Remove shared config definitions
# AVAILABLE_PROMPTS = { ... }
# AVAILABLE_OUTPUT_MODELS = { ... }
# WORKFLOWS = { ... }

# --- Script-Specific Default Configuration ---
DEFAULT_MODELS_TO_TEST = [
    # "deepseek/deepseek-r1-distill-llama-70b",
    # "deepseek/deepseek-chat-v3-0324",
    "google/gemma-3-27b-it",
    "google/gemini-2.5-flash-preview",
]
# DEFAULT_JUDGE_MODEL, DEFAULT_NUM_SAMPLES, DEFAULT_SEED are now in config.py
DEFAULT_RANK_OUTPUT_SUBDIR = "llm_rank_results"  # Specific subdir for this script

# --- Argument Parser ---
parser = argparse.ArgumentParser(
    description="Evaluate LLM outputs based on a defined workflow."
)

# Add common arguments using the helper function
add_common_eval_args(parser, default_output_subdir=DEFAULT_RANK_OUTPUT_SUBDIR)

parser.add_argument(
    "--models",
    nargs="+",
    default=DEFAULT_MODELS_TO_TEST,
    help="List of models to test.",
)
parser.add_argument(
    "--debug",
    action="store_true",  # Makes it a flag, default is False
    help="Enable debug printing for steps like rationale unmasking.",
)

# Remove redundant common arguments
# parser.add_argument("--workflow", ...)
# parser.add_argument("--input-data-file", ...)
# parser.add_argument("--input-data-dir", ...)
# parser.add_argument("--judge-model", ...)
# parser.add_argument("--num-samples", ...)
# parser.add_argument("--output-dir", ...)
# parser.add_argument("--seed", ...)


async def main():
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use imported WORKFLOWS
    selected_workflow_name = args.workflow
    selected_workflow = WORKFLOWS[selected_workflow_name]
    print(f"Using workflow: '{selected_workflow_name}'")

    # --- Validate Workflow using the imported function ---
    # Use imported AVAILABLE_PROMPTS
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
    # Use imported AVAILABLE_PROMPTS
    if judge_prompt_module_name not in AVAILABLE_PROMPTS:
        print(f"Error: Judge prompt module '{judge_prompt_module_name}' not found.")
        return
    # Extract only the prompt text for the judge
    judge_instruction_prompt_text = AVAILABLE_PROMPTS[judge_prompt_module_name][0]
    print(
        f"Using judge prompt module: '{judge_prompt_module_name}' (Version: {AVAILABLE_PROMPTS[judge_prompt_module_name][1]})"
    )

    c = initialize_openrouter_client()

    # 1. Load and Sample Data
    # Use imported DEFAULT_SEED logic from common_args via args.seed
    effective_seed = args.seed if args.seed is not None else int(time.time())
    print(f"Using seed: {effective_seed}")

    # Determine input source and construct output filename (Input source logic remains same)
    # Uses args.input_data_file, args.input_data_dir from common_args
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

    # Construct output filename using the utility
    try:
        # Generate the filename path - append=False requires seed and adds timestamp
        output_file_path = generate_eval_filename(
            output_dir=args.output_dir,
            prefix=Path(__file__).stem,
            workflow_name=selected_workflow_name,
            judge_model=args.judge_model,
            judge_prompt_module_name=judge_prompt_module_name,
            input_data_stem=input_data_stem,
            seed=effective_seed,
            append=False,
        )
        print(f"Output will be saved to: {output_file_path}")
    except ValueError as e:
        print(f"Error generating filename: {e}")
        await c.aclose()
        return

    # Load data (using existing helper)
    # Use args.num_samples from common_args
    sample_df = load_and_sample_data(
        input_source_path, args.num_samples, effective_seed
    )
    if sample_df is None:
        await c.aclose()
        return

    # 2. Run Test Models according to Workflow
    # Pass the imported workflow definition, available prompts, and formatters
    # Use imported AVAILABLE_PROMPTS, AVAILABLE_OUTPUT_MODELS
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
    # Use args.judge_model from common_args
    judge_response_map = await run_judge_evaluation(
        sample_workflow_results,  # Updated results structure from workflow run
        args.judge_model,
        judge_instruction_prompt_text,  # Pass only the text
    )

    # 4. Aggregate Final Results
    results_data = aggregate_ranking_results(
        sample_workflow_results,
        judge_response_map,
        args.models,
        effective_seed,
        selected_workflow_name,  # Pass workflow name
        judge_prompt_module_name,
        selected_workflow,  # Pass workflow definition for context if needed
        AVAILABLE_PROMPTS,  # Pass available prompts for version lookup
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
