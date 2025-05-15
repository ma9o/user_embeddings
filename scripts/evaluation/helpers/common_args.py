import argparse
from pathlib import Path

# Import shared configs needed for choices/defaults in args
from evaluation.config import (
    AVAILABLE_PROMPTS,
    DEFAULT_BASE_DATA_DIR,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEED,
    WORKFLOWS,
)


def add_common_eval_args(parser: argparse.ArgumentParser, default_output_subdir: str):
    """Adds common arguments used by evaluation scripts to the parser."""

    # Workflow Selection
    parser.add_argument(
        "--workflow",
        type=str,
        required=True,
        choices=list(WORKFLOWS.keys()),
        help="Name of the evaluation workflow to run.",
    )

    # Judge Model
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="Model to use for judging.",
    )

    # Input Data Specification
    parser.add_argument(
        "--input-data-file",
        type=Path,
        default=None,
        help="Path to a specific input data CSV file (must contain a 'formatted_context' column). If not provided, samples from all test_output_*.csv files in --input-data-dir.",
    )
    parser.add_argument(
        "--input-data-dir",
        type=Path,
        default=DEFAULT_BASE_DATA_DIR
        / "test_results",  # Default relative to base data dir
        help="Directory containing input data files (test_output_*.csv) which must contain a 'formatted_context' column, used when --input-data-file is not specified.",
    )

    # Sampling and Output
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of samples to evaluate from the input data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BASE_DATA_DIR
        / default_output_subdir,  # Default relative to base, script specifies subdir
        help="Directory to save the evaluation results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for sampling. Defaults to current time if None.",
    )

    # Note: We don't add script-specific args like --models here.
    # Those should be added in the calling script.
    # Prompt module args should use the add_prompt_module_arg helper.

    # Add judge prompt module here as it's used by both, but keep optional
    add_prompt_module_arg(
        parser,
        arg_name="--judge-prompt-module",
        help_text="Name of the prompt module defining the judge's primary task (e.g., ranking criteria or constraints list).",
        required=False,  # Make optional here; specific scripts can enforce requirement
        default=None,
    )


def add_prompt_module_arg(
    parser: argparse.ArgumentParser,
    arg_name: str,
    help_text: str,
    required: bool = False,
    default: str | None = None,
):
    """Adds an argument to select a prompt module name from AVAILABLE_PROMPTS."""
    # Ensure AVAILABLE_PROMPTS is imported where this function is defined.
    parser.add_argument(
        arg_name,
        type=str,
        required=required,
        default=default,
        choices=list(AVAILABLE_PROMPTS.keys()),  # Dynamically set choices
        help=f"{help_text} Available: {list(AVAILABLE_PROMPTS.keys())}",  # Add available choices to help
    )
