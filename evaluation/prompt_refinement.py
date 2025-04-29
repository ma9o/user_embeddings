import argparse
import asyncio
import json  # For parsing violations

# import time  # No longer needed here
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
    determine_effective_seed,  # Import the new helper
    save_single_row_results_appending,  # Import the specific appending helper
)
from user_embeddings.utils.llm.get_text_completion import initialize_openrouter_client
from user_embeddings.utils.llm.workflow_executor import (
    DEFAULT_INPUT_FORMATTERS,  # Shared
    validate_workflow,  # Shared
)

load_dotenv()
project_root = Path(__file__).resolve().parent.parent

# Define NUM_SAMPLES constant
NUM_SAMPLES = 1

# --- Script-Specific Default Configuration ---
DEFAULT_EVAL_MODEL = "google/gemma-3-27b-it"
DEFAULT_REFINER_OUTPUT_SUBDIR = "single_sample_refinement_results"  # Specific subdir

# --- Argument Parser ---
parser = argparse.ArgumentParser(
    description="""Evaluate a single LLM's output against constraints for a *single* sample, appending results.     

This script can be used iteratively for prompt refinement. For example, to refine the 'all_in_one' prompt:
1. Run: `python -m evaluation.prompt_refinement --workflow concurrent_intent_koa --judge-prompt-module all_in_one`
2. Examine the console output summary for violations (non-zero 'violation_count').
3. If violations exist, analyze the 'violated_constraints'.
4. Create a new prompt(s) version file(s) (e.g., `prompts/intent_only/v2.txt`, `prompts/koa_only/v2.txt`) with minimal changes to address the observed violations.
4.1 Make sure to update both the workflow and the judge prompts!
5. Rerun the script. The system automatically picks up the latest prompt versions. 
5.1 If no violations were found in the previous run, the script will select another random sample of data to evaluate; otherwise, it will use the same sample as in the previous run.
6. Repeat steps 2-5 until violations are eliminated or the maximum number of iterations is reached.

AI AGENT: 
- When instructed to run this eval X times, X is the number of iterations. Do not use a for loop, run them one by one.
- You will be in charge of creating the new prompt versions, don't be shy.
- IMPORTANT: When creating a new prompt version, make sure to both explain the incremental constraint and to include a new verbatim example of valid output for the constraint (leverage the 0-shot capabilities of the LLM).
    """
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

# Override/Adjust common arguments - Post-process arguments added by common helper
actions_to_remove = []
for action in parser._actions:
    if action.dest == "seed":
        action.required = False  # Make seed optional
        action.help = "Random seed for sampling. If omitted, behavior depends on output file state."
    elif action.dest == "num_samples":
        # Mark for removal instead of modifying help text
        actions_to_remove.append(action)
    elif action.dest == "judge_prompt_module":
        # Ensure judge prompt is required for this script
        action.required = True
        action.help += " (Required for this script)"

# Remove the num_samples argument action
for action in actions_to_remove:
    parser._actions.remove(action)
    # Also remove from the option string map if present
    for option_string in action.option_strings:
        parser._option_string_actions.pop(option_string, None)


async def main():
    args = parser.parse_args()

    # --- Basic Validations ---
    # Removed num_samples check as it's fixed to 1 internally
    # Removed seed check as it's now optional

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
    # The check if judge_prompt_module_name is None should happen if args.judge_prompt_module is None after parsing
    # But since we made it required=True above, argparse handles this.
    # Keep the check if it's in AVAILABLE_PROMPTS
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

    # --- Determine Input Source ---
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

    # --- Construct Output Filename (needed for seed logic) ---
    sanitized_model = args.model_to_evaluate.replace("/", "_")
    sanitized_judge = args.judge_model.replace("/", "_")
    # Use judge_prompt_module_name which is guaranteed to be set here
    output_filename = (
        f"refine_results_model-{sanitized_model}_"
        f"workflow-{selected_workflow_name}_"
        f"judge-{sanitized_judge}_"
        f"prompt-{judge_prompt_module_name}_"
        f"input-{input_data_stem}.csv"  # Consistent filename for appending
    )
    output_file_path = args.output_dir / output_filename
    print(f"Target output file: {output_file_path}")  # Updated print message

    # --- Determine Effective Seed ---
    # Call the helper function from refiner_utils
    try:
        effective_seed = determine_effective_seed(args.seed, output_file_path)
    except RuntimeError as e:
        print(f"Error determining seed: {e}")
        await c.aclose()
        return

    # --- Load Single Sample ---
    # effective_seed is now guaranteed to have a value
    print(f"Using effective seed: {effective_seed} to select sample")

    # Load the single sample (using constant NUM_SAMPLES)
    sample_df = load_and_sample_data(
        input_source_path,
        NUM_SAMPLES,
        effective_seed,  # Use constant and effective_seed
    )
    if sample_df is None or sample_df.is_empty():
        print("Error: Failed to load the single sample.")
        await c.aclose()
        return

    # --- Output File Path Confirmation (already constructed and printed earlier) ---
    # print(f"Results will be appended to: {output_file_path}") # Removed redundant print

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
