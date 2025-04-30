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
    run_constraint_judge_evaluation,
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
    build_full_dask_graph,  # New graph builder
    validate_workflow,  # Validation still useful
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
    # Using a local client. Adjust if using a distributed cluster.
    print("Initializing Dask client...")
    # Use try-finally to ensure client closes
    client = Client()  # threads_per_worker=1 might be useful if tasks are GIL-bound
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
        is_valid = validate_workflow(
            workflow_name=selected_workflow_name,
            workflow_definition=selected_workflow,
            available_prompts=AVAILABLE_PROMPTS,
            # available_formatters removed
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

        # --- Build Dask Graph --- (Does not execute yet)
        print(
            f"Building Dask graph for workflow '{selected_workflow_name}', model '{args.model_to_evaluate}' ..."
        )
        # build_full_dask_graph returns the structure with delayed objects
        # Structure: List[Dict{'input_data': str, 'model_outputs': {model: {task_id: dask.delayed}}}]
        graph_structure = build_full_dask_graph(
            sample_df=sample_df,  # Single sample DataFrame
            models_to_test=[args.model_to_evaluate],
            workflow=selected_workflow,
            available_prompts=AVAILABLE_PROMPTS,
            available_output_models=AVAILABLE_OUTPUT_MODELS,
            input_column="formatted_context",  # Assuming this column exists
        )

        # --- Execute Dask Graph ---
        print("Computing Dask graph...")
        # dask.compute executes the graph based on dependencies
        # It returns a tuple containing results for each argument passed via *.
        # Since graph_structure is a list containing one dict (for one sample),
        # the result will be a tuple containing one element: that dict.
        computed_result_tuple = dask.compute(*graph_structure, scheduler="distributed")
        # Check if the result tuple is empty or does not contain the expected dictionary
        if not computed_result_tuple or not isinstance(computed_result_tuple[0], dict):
            print(
                "Error: Dask computation did not return the expected results structure (dict for single sample)."
            )
            print(f"Received: {computed_result_tuple}")
            return
        # Extract the single sample result dictionary
        single_computed_sample = computed_result_tuple[0]

        # --- Prepare Judge Inputs (from computed results) ---
        # TODO: Refactor or verify run_and_prepare_judge_inputs to accept pre-computed results.
        # For now, assume it's adapted or we process inline:
        # Remove the check here as we have the single sample already
        # if not computed_results:
        #     print("Error: Dask computation returned no results.")
        #     return
        # single_computed_sample = computed_results[0] # Already assigned

        # Manually prepare judge input here for simplicity for now:
        # This replicates logic from run_and_prepare_judge_inputs
        final_stage_num = max(stage["stage"] for stage in selected_workflow)
        final_task_ids = []
        for stage in selected_workflow:
            if stage["stage"] == final_stage_num:
                final_task_ids.extend([task["task_id"] for task in stage["tasks"]])

        model_computed_results = single_computed_sample["model_outputs"].get(
            args.model_to_evaluate, {}
        )
        final_outputs_for_judge = {}
        any_error_in_final = False
        for task_id in final_task_ids:
            result = model_computed_results.get(task_id)
            # ... (rest of the judge input preparation logic from evaluation_utils) ...
            if result is None:
                error_detail = "Result missing"
                any_error_in_final = True
            elif result.get("error"):
                error_detail = result["error"]
                any_error_in_final = True
            elif result.get("parsed_output") is not None:
                try:
                    error_detail = json.dumps(
                        result["parsed_output"], indent=2, ensure_ascii=False
                    )
                except TypeError:
                    error_detail = "ERROR: Failed to serialize parsed output"
                    any_error_in_final = True
            else:
                error_detail = result.get("raw_output", "").strip()
            final_outputs_for_judge[task_id] = error_detail
        # ... (logic to combine into judge_input_string) ...
        if any_error_in_final:
            judge_input_string = "ERROR: Final task(s) failed."
        elif len(final_outputs_for_judge) == 1:
            judge_input_string = list(final_outputs_for_judge.values())[0]
        else:
            judge_input_string = "\n---\n".join(
                f"Output from {tid}:\n{out}"
                for tid, out in sorted(final_outputs_for_judge.items())
            )

        # Create the structure expected by run_constraint_judge_evaluation
        judge_eval_input = [
            {
                "input_data": single_computed_sample["input_data"],
                "judge_inputs": {args.model_to_evaluate: judge_input_string},
                # Include detailed outputs if needed by downstream aggregation
                "detailed_model_outputs": single_computed_sample["model_outputs"],
            }
        ]

        # --- Run Constraint Judge ---
        # This function likely needs to be adapted to handle the Dask computed results structure
        # and potentially become synchronous or wrapped for Dask if it performs long computation.
        # Assuming it remains async for now, but called synchronously after dask compute.
        print("Running constraint judge...")
        # We need an event loop to run this if it's still async
        # judge_response_map = asyncio.run(run_constraint_judge_evaluation(...))
        # OR refactor run_constraint_judge_evaluation to be synchronous.
        # Let's assume it's refactored to be synchronous or wrapped.
        judge_response_map = run_constraint_judge_evaluation(
            sample_data_list=judge_eval_input,  # Pass prepared input
            model_to_evaluate=args.model_to_evaluate,
            judge_model=args.judge_model,
            judge_constraints_prompt_text=judge_constraints_prompt_text,
        )
        # TODO: Adapt run_constraint_judge_evaluation if it was async.

        # --- Aggregate Results ---
        print("Aggregating results...")
        results_data = aggregate_constraint_results(
            processed_results_list=judge_eval_input,  # Pass the same structure used for judging
            judge_response_map=judge_response_map,
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
