"""Utility functions for generating standardized evaluation result filenames."""

from datetime import datetime
from pathlib import Path


def sanitize_name(name: str) -> str:
    """Replaces '/' with '_' for filesystem compatibility."""
    return name.replace("/", "_")


def generate_eval_filename(
    output_dir: Path,
    prefix: str,  # Use the script stem directly as prefix
    workflow_name: str,
    judge_model: str,
    judge_prompt_module_name: str,
    input_data_stem: str,
    seed: int | None = None,
    append: bool = False,
) -> Path:
    """Generates a standardized filename Path using the script stem as prefix.

    Args:
        output_dir: The directory where the file will be saved.
        prefix: The prefix for the filename (intended to be the script's stem).
        workflow_name: Name of the workflow used.
        judge_model: Name of the judge model used (will be sanitized).
        judge_prompt_module_name: Name of the judge prompt module used.
        input_data_stem: Stem name derived from the input data source.
        seed: The random seed used for sampling (required if append=False).
        append: If True, omits seed and timestamp from the filename.

    Returns:
        The full Path object for the output file.

    Raises:
        ValueError: If seed is None when append is False.
    """
    sanitized_judge = sanitize_name(judge_model)

    # Construct core part
    core_filename = (
        f"{prefix}_"
        f"workflow-{workflow_name}_"
        f"judge-{sanitized_judge}_"
        f"prompt-{judge_prompt_module_name}_"
        f"input-{input_data_stem}"
    )

    if append:
        # Filename for appending - no seed or timestamp
        filename_str = f"{core_filename}.csv"
    else:
        # New file logic: requires seed and adds timestamp
        if seed is None:
            raise ValueError("Seed cannot be None when append is False.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_str = f"{core_filename}_seed_{seed}_{timestamp}.csv"

    return output_dir / filename_str
