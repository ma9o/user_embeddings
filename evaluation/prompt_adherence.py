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
    WORKFLOWS,  # For help text
)
from evaluation.helpers.common_args import (  # Import new helper
    add_common_eval_args,
)
from evaluation.helpers.constraint_utils import (
    aggregate_constraint_results,
    run_constraint_judge_evaluation,
)

# Import shared and specific helpers
from evaluation.helpers.evaluation_utils import (
    load_and_sample_data,
    run_and_parse_test_models,
    save_results,  # Shared save function
)
from evaluation.helpers.filename_utils import generate_eval_filename
from user_embeddings.utils.llm.get_text_completion import initialize_openrouter_client

# Import workflow utilities
from user_embeddings.utils.llm.workflow_executor import (
    # DEFAULT_INPUT_FORMATTERS no longer needed
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
    # Also pass available_output_models for validation
    is_valid = validate_workflow(
        workflow_name=selected_workflow_name,
        workflow_definition=selected_workflow,
        available_prompts=AVAILABLE_PROMPTS,
        available_output_models=AVAILABLE_OUTPUT_MODELS,
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
    # Extract only the prompt text
    judge_constraints_prompt_text = AVAILABLE_PROMPTS[judge_prompt_module_name][0]
    print(
        f"Using judge constraints prompt: '{judge_prompt_module_name}' (Version: {AVAILABLE_PROMPTS[judge_prompt_module_name][1]})"
    )

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
    # Note: run_and_parse_test_models now passes AVAILABLE_OUTPUT_MODELS to the executor
    # Use imported AVAILABLE_PROMPTS, AVAILABLE_OUTPUT_MODELS
    sample_workflow_results = await run_and_parse_test_models(
        sample_df,
        [args.model_to_evaluate],  # IMPORTANT: Pass only the single model here
        selected_workflow,
        AVAILABLE_PROMPTS,
        AVAILABLE_OUTPUT_MODELS,
    )

    # 3. Run Constraint Judge Model Evaluation (New helper)
    # Note: run_constraint_judge_evaluation now receives results containing
    # 'final_judge_inputs' which holds the serialized output of the model to evaluate.
    judge_response_map = await run_constraint_judge_evaluation(
        sample_workflow_results=sample_workflow_results,
        model_to_evaluate=args.model_to_evaluate,
        judge_model=args.judge_model,
        judge_constraints_prompt_text=judge_constraints_prompt_text,  # Pass only text
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
        available_prompts=AVAILABLE_PROMPTS,  # Pass available prompts for version lookup
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
        print(
            f"Prompt Definition: {judge_prompt_module_name} (Version: {AVAILABLE_PROMPTS.get(judge_prompt_module_name, ('', 'N/A'))[1]})"
        )
        print(f"Total Samples Judged: {num_samples_judged}")
        print(f"Total Violations Found: {total_violations}")
        if num_samples_judged > 0:
            print(f"Average Violations per Sample: {avg_violations:.2f}")

        # Optional: Print top N most common violations if needed
        if "violated_constraints" in results_df.columns and num_samples_judged > 0:
            try:
                # Filter for rows where violations were found and successfully parsed
                violation_dicts_df = results_df.filter(
                    (pl.col("violation_count") > 0)
                    & (
                        pl.col("violated_constraints").str.starts_with("{")
                    )  # Check for dict start
                ).select(
                    pl.col("violated_constraints")
                    .str.json_decode()  # Decode JSON string to struct
                    .alias("violations_dict")
                )

                # Check if the column is indeed a struct after decoding
                if not isinstance(
                    violation_dicts_df["violations_dict"].dtype, pl.Struct
                ):
                    print(
                        "Warning: Column 'violations_dict' is not a Struct after JSON decoding. Skipping violation analysis."
                    )
                    all_violation_ids = pl.DataFrame(
                        {"violation_id": []}, schema={"violation_id": pl.String}
                    )  # Empty DF with correct schema
                elif violation_dicts_df.is_empty():
                    print("No rows with valid violation dictionaries found.")
                    all_violation_ids = pl.DataFrame(
                        {"violation_id": []}, schema={"violation_id": pl.String}
                    )
                else:
                    # Extract keys (violation IDs) from each dictionary/struct using map_elements
                    all_violation_ids = (
                        violation_dicts_df.select(
                            pl.col("violations_dict")
                            # Apply lambda to get keys, handle potential non-dict elements gracefully
                            .map_elements(
                                lambda d: list(d.keys()) if isinstance(d, dict) else [],
                                return_dtype=pl.List(pl.String),
                            )
                            .alias("violation_id_list")
                        )
                        .explode("violation_id_list")
                        .rename({"violation_id_list": "violation_id"})
                    )

                # Proceed only if all_violation_ids DataFrame is not empty and has the column
                if (
                    not all_violation_ids.is_empty()
                    and "violation_id" in all_violation_ids.columns
                ):
                    violation_counts = (
                        all_violation_ids.filter(pl.col("violation_id").is_not_null())[
                            "violation_id"
                        ]  # Filter nulls just in case
                        .value_counts()
                        .sort("count", descending=True)
                    )
                    print("\nMost Common Violations:")
                    print(violation_counts.head(10))
                else:
                    # This case covers initial empty df, parsing failures, or empty key lists
                    print(
                        "No specific violation details could be extracted or parsed from results."
                    )

            except Exception as e:
                # Add more specific error context if possible
                print(f"Could not analyze violation details due to an error: {e}")
                import traceback

                traceback.print_exc()  # Print full traceback for debugging

    print(f"Constraint evaluation complete. Results saved to {output_file_path}")

    await c.aclose()


if __name__ == "__main__":
    asyncio.run(main())
