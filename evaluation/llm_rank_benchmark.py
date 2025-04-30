import argparse

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
    calculate_and_print_leaderboard,  # Needs judge logic
)

# Initialize LLM client
from user_embeddings.utils.llm.get_text_completion import initialize_openrouter_client

# Import Dask-based workflow executor
from user_embeddings.utils.llm.workflow_executor import (
    # DEFAULT_INPUT_FORMATTERS, # Removed
    build_full_evaluation_graph,  # Use this now
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

        # 2. Build Full Dask Graph (Workflow + Ranking Judge)
        print(
            f"Building FULL Dask graph for {len(args.models)} models and {args.num_samples} samples..."
        )
        graph_structure = build_full_evaluation_graph(
            sample_df=sample_df,
            models_to_test=args.models,
            workflow=selected_workflow,
            available_prompts=AVAILABLE_PROMPTS,
            available_output_models=AVAILABLE_OUTPUT_MODELS,
            input_column="formatted_context",
            # Judge specific args for ranking evaluation
            judge_type="ranking",
            judge_model=args.judge_model,
            judge_prompt_text=judge_instruction_prompt_text,
            # constraint_model_to_evaluate is not needed for ranking
        )

        # 3. Execute Dask Graph
        print("Computing Dask graph (including ranking judge tasks)...")
        # Structure: List[Dict{'input_data': delayed, 'workflow_outputs': ..., 'judge_results': ...}]
        computed_result_tuple = dask.compute(*graph_structure, scheduler="distributed")
        # Check result structure (expecting a tuple containing one list)
        if not computed_result_tuple or not isinstance(computed_result_tuple[0], list):
            print(
                "Error: Dask computation did not return the expected results structure (list of samples)."
            )
            print(f"Received: {computed_result_tuple}")
            return
        computed_results_list = computed_result_tuple[0]
        print("Dask computation complete.")

        # 4. Prepare Judge Inputs (No longer needed inline) ---

        # 5. Run Judge Model Evaluation (No longer needed separately) ---

        # 6. Aggregate Final Results (Operates on fully computed results)
        print("Aggregating final results...")
        results_data = aggregate_ranking_results(
            # Pass the list of computed result dicts
            computed_results_list=computed_results_list,
            # judge_response_map is no longer needed here
            models=args.models,
            seed=effective_seed,
            workflow_name=selected_workflow_name,
            judge_prompt_name=judge_prompt_module_name,
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
