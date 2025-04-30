import argparse

# import asyncio # No longer needed
import json
from pathlib import Path

import dask  # Import dask
import polars as pl
from dask.distributed import Client  # Import dask client and progress
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
)

# Import shared and specific helpers
from evaluation.helpers.evaluation_utils import (  # Import relevant helpers
    load_and_sample_data,
    # save_results is not used here
)
from evaluation.helpers.filename_utils import generate_eval_filename
from evaluation.helpers.refiner_utils import (
    determine_effective_seed,  # Import the new helper
    save_single_row_results_appending,  # Import the specific appending helper
)
from user_embeddings.utils.llm.get_text_completion import initialize_openrouter_client
from user_embeddings.utils.llm.workflow_executor import (
    # DEFAULT_INPUT_FORMATTERS, # Removed
    build_full_evaluation_graph,  # Use this now
    validate_workflow,
)

load_dotenv()
project_root = Path(__file__).resolve().parent.parent

# Define NUM_SAMPLES constant
NUM_SAMPLES = 1

# --- Script-Specific Default Configuration ---
# DEFAULT_EVAL_MODEL = "google/gemma-3-27b-it"
# DEFAULT_EVAL_MODEL = "google/gemini-2.5-flash-preview"
DEFAULT_EVAL_MODEL = "x-ai/grok-3-mini-beta"

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
add_common_eval_args(parser, default_output_subdir=Path(__file__).stem)

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


# --- Main Execution Logic (Synchronous with Dask) ---
def main():
    args = parser.parse_args()

    # --- Basic Validations ---
    # Removed num_samples check as it's fixed to 1 internally
    # Removed seed check as it's now optional

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize Dask Client ---
    print("Initializing Dask client...")
    client = Client()
    print(f"Dask dashboard link: {client.dashboard_link}")

    try:
        # --- Initialize LLM Client (Potentially needed by workers) ---
        # Ensure OpenRouter (or other LLM client) is initialized
        # This might need adjustment based on how Dask workers inherit state
        # or using client.run() to initialize on workers.
        print("Initializing LLM client...")
        # Assuming initialize_openrouter_client() sets up necessary global state
        # or returns a client object that can be passed/recreated.
        _ = initialize_openrouter_client()

        # --- Workflow Setup ---
        selected_workflow_name = args.workflow
        selected_workflow = WORKFLOWS[selected_workflow_name]
        print(
            f"Using workflow '{selected_workflow_name}' for model '{args.model_to_evaluate}'"
        )

        # Validate workflow structure before building graph
        if not validate_workflow("Main Workflow", selected_workflow, AVAILABLE_PROMPTS):
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

        # --- Determine Input Source ---
        if args.input_data_file:
            input_source_path = args.input_data_file
            if not input_source_path.is_file():
                print(
                    f"Error: Specified input data file not found: {input_source_path}"
                )
                return
            input_data_stem = input_source_path.stem
            print(f"Using specific input file: {input_source_path}")
        else:
            input_source_path = args.input_data_dir
            if not input_source_path.is_dir():
                print(
                    f"Error: Specified input data directory not found: {input_source_path}"
                )
                return
            input_data_stem = f"combined_{input_source_path.name}"
            print(f"Sampling from CSV files in directory: {input_source_path}")

        # --- Construct Output Filename (needed for seed logic) ---
        try:
            # Generate the filename path - append=True means no seed/timestamp in name
            output_file_path = generate_eval_filename(
                output_dir=args.output_dir,
                prefix=Path(__file__).stem,
                workflow_name=selected_workflow_name,
                judge_model=args.judge_model,
                judge_prompt_module_name=judge_prompt_module_name,
                input_data_stem=input_data_stem,
                append=True,
                seed=None,  # Pass None for seed when appending
            )
            print(f"Target output file: {output_file_path}")
        except ValueError as e:
            print(f"Error generating filename: {e}")
            return

        # --- Determine Effective Seed ---
        # Call the helper function from refiner_utils
        try:
            effective_seed = determine_effective_seed(args.seed, output_file_path)
        except RuntimeError as e:
            print(f"Error determining seed: {e}")
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
            return

        # --- Build Full Dask Graph (Workflow + Judge) --- (Does not execute yet)
        print(
            f"Building FULL Dask graph for workflow '{selected_workflow_name}', model '{args.model_to_evaluate}' ..."
        )
        # Call the new function, including judge parameters
        graph_structure = build_full_evaluation_graph(
            sample_df=sample_df,  # Single sample DataFrame
            models_to_test=[args.model_to_evaluate],
            workflow=selected_workflow,
            available_prompts=AVAILABLE_PROMPTS,
            available_output_models=AVAILABLE_OUTPUT_MODELS,
            input_column="formatted_context",
            # Judge specific args for constraint evaluation
            judge_type="constraint",
            judge_model=args.judge_model,
            judge_prompt_text=judge_constraints_prompt_text,
            constraint_model_to_evaluate=args.model_to_evaluate,
        )

        # --- Execute Dask Graph ---
        print("Computing Dask graph (including judge tasks)...")
        # Structure: List[Dict{'input_data': delayed, 'workflow_outputs': ..., 'judge_results': ...}]
        computed_result_tuple = dask.compute(*graph_structure, scheduler="distributed")
        if not computed_result_tuple or not isinstance(computed_result_tuple[0], dict):
            print(
                "Error: Dask computation did not return the expected results structure (dict for single sample)."
            )
            print(f"Received: {computed_result_tuple}")
            return
        # Assign the computed result for the single sample
        computed_sample_result = computed_result_tuple[0]
        print("Dask computation complete.")

        # --- Prepare Judge Inputs (No longer needed inline) ---
        # The judge results are now computed as part of the graph.

        # --- Run Constraint Judge (No longer needed separately) ---

        # --- Aggregate Results --- (Operates on the fully computed result)
        print("Aggregating results...")
        results_data = aggregate_constraint_results(
            # Pass the list containing the single computed result dict
            computed_results_list=[computed_sample_result],
            # judge_response_map is no longer needed as input, judge results are inside computed_sample_result
            model_to_evaluate=args.model_to_evaluate,
            effective_seed=effective_seed,
            workflow_name=selected_workflow_name,
            judge_prompt_name=judge_prompt_module_name,
        )

        if not results_data:
            print("Error: Failed to aggregate results.")
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
            print(
                f"Raw judge output: {single_result_dict.get('judge_raw_output', 'N/A')}"
            )

        # --- Append Result to CSV ---
        results_df = pl.DataFrame(results_data)  # DataFrame with single row
        save_single_row_results_appending(results_df, output_file_path)

        print(
            f"\nSingle sample constraint evaluation complete for seed {effective_seed}."
        )
        print(f"Results appended to {output_file_path}")

    finally:
        # Ensure Dask client is closed
        print("Closing Dask client...")
        client.close()
        print("Dask client closed.")


if __name__ == "__main__":
    # Run the synchronous main function
    main()
