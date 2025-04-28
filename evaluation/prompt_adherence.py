import argparse
import asyncio
import time
from datetime import datetime
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

# Import shared configurations
from evaluation.config import (
    AVAILABLE_OUTPUT_MODELS,
    AVAILABLE_PROMPTS,
    WORKFLOWS,  # For help text
)
from evaluation.helpers.common_args import (  # Import new helper
    add_common_eval_args,
)

# Import helpers (using newly added/renamed functions)
from evaluation.helpers.evaluation_utils import (
    aggregate_constraint_results,  # New aggregator
    load_and_sample_data,
    run_and_parse_test_models,  # Still used to get model output
    run_constraint_judge_evaluation,  # New judge runner
    save_results,
)
from user_embeddings.utils.llm.get_text_completion import initialize_openrouter_client

# Import workflow utilities
from user_embeddings.utils.llm.workflow_executor import (
    DEFAULT_INPUT_FORMATTERS,  # Shared
    validate_workflow,  # Shared
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
DEFAULT_EVAL_MODEL = "google/gemma-3-27b-it"
# DEFAULT_JUDGE_MODEL, DEFAULT_NUM_SAMPLES, DEFAULT_SEED are now in config.py
DEFAULT_ADHERENCE_OUTPUT_SUBDIR = "constraint_test_results"  # Specific subdir

# --- Argument Parser --- (Adapted for constraint evaluation)
parser = argparse.ArgumentParser(
    description="Evaluate a single LLM's output against a set of constraints."
)

# Add common arguments
add_common_eval_args(parser, default_output_subdir=DEFAULT_ADHERENCE_OUTPUT_SUBDIR)

# Add script-specific arguments
parser.add_argument(
    "--model-to-evaluate",
    type=str,
    default=DEFAULT_EVAL_MODEL,
    help="The specific model whose output will be judged.",
)


async def main():
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use imported WORKFLOWS
    selected_workflow_name = args.workflow
    selected_workflow = WORKFLOWS[selected_workflow_name]
    print(
        f"Using workflow '{selected_workflow_name}' to generate outputs for model '{args.model_to_evaluate}'"
    )

    # --- Validate Workflow --- (Reused)
    # Use imported AVAILABLE_PROMPTS
    is_valid = validate_workflow(
        workflow_name=selected_workflow_name,
        workflow_definition=selected_workflow,
        available_prompts=AVAILABLE_PROMPTS,
        available_formatters=DEFAULT_INPUT_FORMATTERS,
    )
    if not is_valid:
        print("Workflow validation failed. Exiting.")
        return

    # --- Load Judge Constraints Prompt ---
    judge_prompt_module_name = args.judge_prompt_module  # This is for constraints
    # Use imported AVAILABLE_PROMPTS
    if judge_prompt_module_name not in AVAILABLE_PROMPTS:
        print(
            f"Error: Judge constraints prompt module '{judge_prompt_module_name}' not found."
        )
        return
    judge_constraints_prompt = AVAILABLE_PROMPTS[judge_prompt_module_name]
    print(f"Using judge constraints prompt: '{judge_prompt_module_name}'")

    c = initialize_openrouter_client()

    # 1. Load and Sample Data (Reused)
    # Use imported DEFAULT_SEED logic from common_args via args.seed
    effective_seed = args.seed if args.seed is not None else int(time.time())
    print(f"Using seed: {effective_seed}")

    # Determine input source (Reused logic)
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
        input_data_stem = f"combined_{input_source_path.name}"
        print(f"Sampling from CSV files in directory: {input_source_path}")

    # Construct output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = (
        f"constraint_eval_model-{args.model_to_evaluate.replace('/', '_')}_"  # Sanitize model name
        f"judge-{args.judge_model.replace('/', '_')}_"  # Sanitize judge model name
        f"prompt-{judge_prompt_module_name}_"  # Constraints used
        f"workflow-{selected_workflow_name}_{input_data_stem}_seed_{effective_seed}_{timestamp}.csv"
    )
    # Use args.output_dir from common_args
    output_file_path = args.output_dir / output_filename

    # Load data (Reused helper)
    # Use args.num_samples from common_args
    sample_df = load_and_sample_data(
        input_source_path, args.num_samples, effective_seed
    )
    if sample_df is None:
        await c.aclose()
        return

    # 2. Run Test Model (single model) according to Workflow (Reused helper)
    # Pass only the model to evaluate in the list
    print(
        f"Running workflow '{selected_workflow_name}' for model '{args.model_to_evaluate}'..."
    )
    # Use imported AVAILABLE_PROMPTS, AVAILABLE_OUTPUT_MODELS
    sample_workflow_results = await run_and_parse_test_models(
        sample_df,
        [args.model_to_evaluate],  # IMPORTANT: Pass only the single model here
        selected_workflow,
        AVAILABLE_PROMPTS,
        DEFAULT_INPUT_FORMATTERS,
        AVAILABLE_OUTPUT_MODELS,
    )

    # 3. Run Constraint Judge Model Evaluation (New helper)
    # Use args.judge_model from common_args
    judge_response_map = await run_constraint_judge_evaluation(
        sample_workflow_results=sample_workflow_results,
        model_to_evaluate=args.model_to_evaluate,
        judge_model=args.judge_model,
        judge_constraints_prompt=judge_constraints_prompt,  # Constraints definition
    )

    # 4. Aggregate Final Results (New helper)
    results_data = aggregate_constraint_results(
        sample_workflow_results=sample_workflow_results,
        judge_response_map=judge_response_map,
        model_to_evaluate=args.model_to_evaluate,
        effective_seed=effective_seed,
        workflow_name=selected_workflow_name,
        judge_prompt_name=judge_prompt_module_name,  # Renamed back
        workflow=selected_workflow,
    )
    results_df = pl.DataFrame(results_data)

    # 5. Save Results (Reused helper)
    save_results(results_df, output_file_path)

    # 6. Print Summary (Optional: Add more detailed summary)
    if not results_df.is_empty() and "violation_count" in results_df.columns:
        total_violations = results_df.filter(pl.col("violation_count") > 0)[
            "violation_count"
        ].sum()
        avg_violations = results_df.filter(pl.col("violation_count") >= 0)[
            "violation_count"
        ].mean()
        num_samples_judged = results_df.filter(pl.col("violation_count") >= 0).height
        print("--- Constraint Violation Summary ---")
        print(f"Model Evaluated: {args.model_to_evaluate}")
        print(f"Prompt Definition: {judge_prompt_module_name}")
        print(f"Total Samples Judged: {num_samples_judged}")
        print(f"Total Violations Found: {total_violations}")
        if num_samples_judged > 0:
            print(f"Average Violations per Sample: {avg_violations:.2f}")

        # Optional: Print top N most common violations if needed
        if "violated_constraints" in results_df.columns and num_samples_judged > 0:
            try:
                all_violations = (
                    results_df.filter(
                        # Consider adding a more robust check for valid JSON strings here if needed
                        pl.col("violated_constraints").str.starts_with("[")
                    )
                    .select(
                        # Use the more efficient native JSON decoding expression
                        pl.col("violated_constraints").str.json_decode()
                    )
                    .explode("violated_constraints")
                )
                if not all_violations.is_empty():
                    violation_counts = (
                        all_violations["violated_constraints"]
                        .value_counts()
                        .sort("count", descending=True)
                    )
                    print("\nMost Common Violations:")
                    print(violation_counts.head(10))
                else:
                    print("No violations recorded in results.")

            except Exception as e:
                print(f"Could not analyze violation details: {e}")

    print(f"Constraint evaluation complete. Results saved to {output_file_path}")

    await c.aclose()


if __name__ == "__main__":
    asyncio.run(main())
