import argparse
import json

# import asyncio # Removed
import time
from pathlib import Path

import dask  # Added
import polars as pl
from dask.distributed import Client  # Added
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
    # run_and_parse_test_models, # Replaced by Dask execution + preparation
    save_results,
)
from evaluation.helpers.filename_utils import generate_eval_filename
from evaluation.helpers.ranking_utils import (
    aggregate_ranking_results,
    calculate_and_print_leaderboard,
    run_judge_evaluation,  # Needs judge logic
)

# Initialize LLM client
from user_embeddings.utils.llm.get_text_completion import initialize_openrouter_client

# Import Dask-based workflow executor
from user_embeddings.utils.llm.workflow_executor import (
    # DEFAULT_INPUT_FORMATTERS, # Removed
    build_full_dask_graph,  # New graph builder
    validate_workflow,
)

load_dotenv()
project_root = Path(__file__).resolve().parent.parent

# Default models (Unchanged)
DEFAULT_MODELS_TO_TEST = [
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1",
    "google/gemma-3-27b-it",
    "google/gemini-2.5-flash-preview",
    "x-ai/grok-3-mini-beta",
]
DEFAULT_RANK_OUTPUT_SUBDIR = "llm_rank_results"

# --- Argument Parser (Unchanged) ---
parser = argparse.ArgumentParser(
    description="Evaluate LLM outputs based on a defined workflow (Dask version)."
)
add_common_eval_args(parser, default_output_subdir=DEFAULT_RANK_OUTPUT_SUBDIR)
parser.add_argument(
    "--models",
    nargs="+",
    default=DEFAULT_MODELS_TO_TEST,
    help="List of models to test.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug printing for steps like rationale unmasking.",
)


# --- Main Execution Logic (Synchronous with Dask) ---
def main():
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing Dask client...")
    client = Client()  # Initialize Dask client
    print(f"Dask dashboard link: {client.dashboard_link}")

    try:
        # Initialize LLM Client
        print("Initializing LLM client...")
        _ = initialize_openrouter_client()

        # Use imported WORKFLOWS
        selected_workflow_name = args.workflow
        selected_workflow = WORKFLOWS[selected_workflow_name]
        print(f"Using workflow: '{selected_workflow_name}'")

        # --- Validate Workflow ---
        is_valid = validate_workflow(
            workflow_name=selected_workflow_name,
            workflow_definition=selected_workflow,
            available_prompts=AVAILABLE_PROMPTS,
            # available_formatters removed
        )
        if not is_valid:
            print("Workflow validation failed. Exiting.")
            return

        # --- Determine Judge Prompt ---
        judge_prompt_module_name = args.judge_prompt_module
        if not judge_prompt_module_name:
            last_stage_num = max(s["stage"] for s in selected_workflow)
            final_tasks = []
            for s in selected_workflow:
                if s["stage"] == last_stage_num:
                    final_tasks.extend(t["prompt"] for t in s["tasks"])

            if len(final_tasks) == 1:
                judge_prompt_module_name = final_tasks[0]
                print(
                    f"Judge prompt not specified, defaulting to: '{judge_prompt_module_name}'"
                )
            else:
                print(
                    "Error: --judge-prompt-module is required when workflow final stage has multiple tasks."
                )
                return

        if judge_prompt_module_name not in AVAILABLE_PROMPTS:
            print(f"Error: Judge prompt module '{judge_prompt_module_name}' not found.")
            return
        judge_instruction_prompt_text = AVAILABLE_PROMPTS[judge_prompt_module_name][0]
        print(
            f"Using judge prompt module: '{judge_prompt_module_name}' (Version: {AVAILABLE_PROMPTS[judge_prompt_module_name][1]})"
        )

        # 1. Load and Sample Data
        effective_seed = args.seed if args.seed is not None else int(time.time())
        print(f"Using seed: {effective_seed}")

        # Determine input source and construct output filename
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

        try:
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
            return

        sample_df = load_and_sample_data(
            input_source_path, args.num_samples, effective_seed
        )
        if sample_df is None:
            return

        # 2. Build Dask Graph for Test Models
        print(
            f"Building Dask graph for {len(args.models)} models and {args.num_samples} samples..."
        )
        graph_structure = build_full_dask_graph(
            sample_df=sample_df,
            models_to_test=args.models,
            workflow=selected_workflow,
            available_prompts=AVAILABLE_PROMPTS,
            available_output_models=AVAILABLE_OUTPUT_MODELS,
            input_column="formatted_context",
        )

        # 3. Execute Dask Graph
        print("Computing Dask graph...")
        # Structure: List[Dict{'input_data': str, 'model_outputs': {model: {task_id: TaskResult}}}]
        computed_results = dask.compute(*graph_structure, scheduler="distributed")[0]
        # futures = client.compute(graph_structure); progress(futures); computed_results = client.gather(futures)
        print("Dask computation complete.")

        # 4. Prepare Judge Inputs (from computed results)
        # This step might need adaptation in run_and_prepare_judge_inputs or inline processing
        # Assume run_and_prepare_judge_inputs handles the computed_results structure
        print("Preparing judge inputs from computed results...")
        # TODO: Verify/adapt run_and_prepare_judge_inputs if necessary
        # It needs to iterate through the computed_results list
        judge_ready_results = []
        for computed_sample in computed_results:
            # Simplified inline preparation for now
            final_stage_num = max(stage["stage"] for stage in selected_workflow)
            final_task_ids = []
            for stage in selected_workflow:
                if stage["stage"] == final_stage_num:
                    final_task_ids.extend([task["task_id"] for task in stage["tasks"]])

            judge_inputs_for_sample = {}
            for model in args.models:
                model_computed_results = computed_sample["model_outputs"].get(model, {})
                final_outputs_for_judge = {}
                any_error_in_final = False
                for task_id in final_task_ids:
                    result = model_computed_results.get(task_id)
                    # ... (judge input preparation logic as in prompt_refinement) ...
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
                judge_inputs_for_sample[model] = judge_input_string

            judge_ready_results.append(
                {
                    "input_data": computed_sample["input_data"],
                    "judge_inputs": judge_inputs_for_sample,
                    "detailed_model_outputs": computed_sample["model_outputs"],
                }
            )

        # 5. Run Judge Model Evaluation
        # This helper might also need adaptation if it was async.
        # Assuming it's synchronous or wrapped.
        print("Running judge model evaluation...")
        judge_response_map = run_judge_evaluation(
            judge_ready_results,  # Pass the prepared results
            args.judge_model,
            judge_instruction_prompt_text,
        )
        # TODO: Adapt run_judge_evaluation if it was async.

        # 6. Aggregate Final Results
        print("Aggregating final results...")
        results_data = aggregate_ranking_results(
            processed_results_list=judge_ready_results,  # Pass prepared judge results
            judge_response_map=judge_response_map,
            models=args.models,
            seed=effective_seed,
            workflow_name=selected_workflow_name,
            judge_prompt_name=judge_prompt_module_name,
            # workflow=selected_workflow, # May not be needed
            # available_prompts=AVAILABLE_PROMPTS, # May not be needed
            debug=args.debug,
        )
        results_df = pl.DataFrame(results_data)

        # 7. Save Results
        print("Saving results...")
        save_results(results_df, output_file_path)

        # 8. Calculate and Print Leaderboard
        print("\n--- Leaderboard ---")
        calculate_and_print_leaderboard(results_df, args.models)

        print(f"\nEvaluation complete. Results saved to {output_file_path}")

    finally:
        # Ensure Dask client is closed
        print("Closing Dask client...")
        client.close()
        print("Dask client closed.")


if __name__ == "__main__":
    # Run synchronous main function
    main()
