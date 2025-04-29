import argparse
import asyncio
import json  # For parsing violations
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

# Import shared configurations
from evaluation.config import (
    AVAILABLE_OUTPUT_MODELS,
    AVAILABLE_PROMPTS,
    WORKFLOWS,  # For help text
)
from evaluation.helpers.common_args import (  # Import common args helper
    add_common_eval_args,
)
from evaluation.helpers.constraint_utils import (
    aggregate_constraint_results,
    run_constraint_judge_evaluation,
)

# Import shared and specific helpers
from evaluation.helpers.evaluation_utils import (  # Import relevant helpers
    load_and_sample_data,
    run_and_parse_test_models,
    # save_results is not used here
)
from evaluation.helpers.refiner_utils import (
    save_single_row_results_appending,  # Import the specific appending helper
)
from user_embeddings.utils.llm.get_text_completion import initialize_openrouter_client
from user_embeddings.utils.llm.workflow_executor import (
    DEFAULT_INPUT_FORMATTERS,  # Shared
    validate_workflow,  # Shared
)

load_dotenv()
project_root = Path(__file__).resolve().parent.parent

# --- Script-Specific Default Configuration ---
DEFAULT_EVAL_MODEL = "google/gemma-3-27b-it"
DEFAULT_REFINER_OUTPUT_SUBDIR = "single_sample_refinement_results"  # Specific subdir

# --- Argument Parser ---
parser = argparse.ArgumentParser(
    description="Evaluate a single LLM's output against constraints for a *single* sample, appending results."
)

# Add common arguments (will add --seed, --num-samples etc.)
add_common_eval_args(parser, default_output_subdir=DEFAULT_REFINER_OUTPUT_SUBDIR)

# Add script-specific arguments
parser.add_argument(
    "--model-to-evaluate",
    type=str,
    default=DEFAULT_EVAL_MODEL,
    help="The specific model whose output will be judged.",
)

# Override/Adjust common arguments
parser.set_defaults(num_samples=1)  # Force num_samples to 1 for this script
# Make seed required
for action in parser._actions:
    if action.dest == "seed":
        action.required = True
        action.help += " (Required for reproducibility in this script)"
    if action.dest == "num_samples":
        action.help = "Number of samples to evaluate (Fixed to 1 for this script)."


async def main():
    args = parser.parse_args()

    # --- Basic Validations ---
    if args.num_samples != 1:
        print("Error: This script is designed to run with exactly 1 sample.")
        print("Please use --num-samples 1 (or omit it as it defaults to 1).")
        return

    if args.seed is None:
        # This should not happen due to the 'required=True' change, but belt-and-suspenders
        print("Error: A --seed must be provided for reproducible sample selection.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Workflow Setup ---
    selected_workflow_name = args.workflow
    selected_workflow = WORKFLOWS[selected_workflow_name]
    print(
        f"Using workflow '{selected_workflow_name}' for model '{args.model_to_evaluate}'"
    )

    is_valid = validate_workflow(
        workflow_name=selected_workflow_name,
        workflow_definition=selected_workflow,
        available_prompts=AVAILABLE_PROMPTS,
        available_formatters=DEFAULT_INPUT_FORMATTERS,
    )
    if not is_valid:
        print("Workflow validation failed. Exiting.")
        return

    # --- Judge Constraints Prompt Setup ---
    judge_prompt_module_name = args.judge_prompt_module
    if judge_prompt_module_name not in AVAILABLE_PROMPTS:
        print(
            f"Error: Judge constraints prompt module '{judge_prompt_module_name}' not found."
        )
        return
    # Extract only the prompt text
    judge_constraints_prompt_text = AVAILABLE_PROMPTS[judge_prompt_module_name][0]
    print(
        f"Using judge constraints prompt: '{judge_prompt_module_name}' (Version: {AVAILABLE_PROMPTS[judge_prompt_module_name][1]})"
    )

    c = initialize_openrouter_client()

    # --- Load Single Sample ---
    effective_seed = args.seed  # Seed is required and validated
    print(f"Using seed: {effective_seed} to select sample")

    # Determine input source
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
        input_data_stem = f"combined_{input_source_path.name}"
        print(f"Sampling from CSV files in directory: {input_source_path}")

    # Load the single sample
    sample_df = load_and_sample_data(
        input_source_path, 1, effective_seed
    )  # num_samples=1
    if sample_df is None or sample_df.is_empty():
        print("Error: Failed to load the single sample.")
        await c.aclose()
        return

    # --- Construct Output Filename (No Timestamp) ---
    sanitized_model = args.model_to_evaluate.replace("/", "_")
    sanitized_judge = args.judge_model.replace("/", "_")
    output_filename = (
        f"refine_results_model-{sanitized_model}_"
        f"workflow-{selected_workflow_name}_"
        f"judge-{sanitized_judge}_"
        f"prompt-{judge_prompt_module_name}_"
        f"input-{input_data_stem}.csv"  # Consistent filename for appending
    )
    output_file_path = args.output_dir / output_filename
    print(f"Results will be appended to: {output_file_path}")

    # --- Run Model Workflow ---
    print(
        f"Running workflow '{selected_workflow_name}' for model '{args.model_to_evaluate}' on the single sample..."
    )
    sample_workflow_results = await run_and_parse_test_models(
        sample_df,
        [args.model_to_evaluate],  # Single model
        selected_workflow,
        AVAILABLE_PROMPTS,
        DEFAULT_INPUT_FORMATTERS,
        AVAILABLE_OUTPUT_MODELS,
    )

    # --- Run Constraint Judge ---
    judge_response_map = await run_constraint_judge_evaluation(
        sample_workflow_results=sample_workflow_results,  # List containing one sample result
        model_to_evaluate=args.model_to_evaluate,
        judge_model=args.judge_model,
        judge_constraints_prompt_text=judge_constraints_prompt_text,  # Pass only text
    )

    # --- Aggregate Results ---
    results_data = aggregate_constraint_results(
        sample_workflow_results=sample_workflow_results,
        judge_response_map=judge_response_map,
        model_to_evaluate=args.model_to_evaluate,
        effective_seed=effective_seed,
        workflow_name=selected_workflow_name,
        judge_prompt_name=judge_prompt_module_name,
        workflow=selected_workflow,
        available_prompts=AVAILABLE_PROMPTS,  # Pass available prompts for version lookup
    )

    if not results_data:
        print("Error: Failed to aggregate results.")
        await c.aclose()
        return

    # --- Print Violations to Console ---
    single_result_dict = results_data[0]
    violations_json_str = single_result_dict.get("violated_constraints", "{}")
    violation_count = single_result_dict.get("violation_count", -1)

    print("\n--- Constraint Violation Check ---")
    if violation_count > 0 and violations_json_str != "ERROR: Parse Failed":
        try:
            violations_dict = json.loads(violations_json_str)
            if violations_dict:  # Check if dict is not empty
                print(f"Found {violation_count} violation(s):")
                for constraint_id, explanation in violations_dict.items():
                    print(f"  - ID: {constraint_id}")
                    print(f"    Explanation: {explanation}")
            else:
                # This case might happen if count > 0 but parsing yielded empty dict somehow
                print(
                    "No violations found (parsed dictionary was empty despite count > 0)."
                )
        except json.JSONDecodeError:
            print(
                f"Error: Could not parse violations JSON string: {violations_json_str}"
            )
        except Exception as e:
            print(f"An unexpected error occurred while processing violations: {e}")
    elif violation_count == 0:
        print("No violations found. Good job!")
    else:  # violation_count == -1 or ERROR string
        print("Could not determine violations (Judge failed or parse error).")
        print(f"Raw judge output: {single_result_dict.get('judge_raw_output', 'N/A')}")

    # --- Append Result to CSV ---
    results_df = pl.DataFrame(results_data)  # DataFrame with single row
    save_single_row_results_appending(results_df, output_file_path)

    print(f"\nSingle sample constraint evaluation complete for seed {effective_seed}.")
    print(f"Results appended to {output_file_path}")

    await c.aclose()


if __name__ == "__main__":
    asyncio.run(main())
