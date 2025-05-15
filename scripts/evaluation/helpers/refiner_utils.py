import logging
import time  # Added for timestamp seeding
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# --- Refiner Specific Helpers ---


def determine_effective_seed(provided_seed: int | None, output_file_path: Path) -> int:
    """
    Determines the effective seed to use for the refinement run.

    If a seed is provided, it's used directly.
    If no seed is provided, it checks the last run's results in the output file:
        - If the last run had 0 violations, a new seed based on the current time is generated.
        - If the last run had violations, the seed from the last run is reused.
        - If the output file doesn't exist or has issues, a new seed is generated.

    Args:
        provided_seed: The seed explicitly provided via arguments (or None).
        output_file_path: Path to the CSV file containing results from previous runs.

    Returns:
        The integer seed to use for the current run.

    Raises:
        RuntimeError: If the output file exists but cannot be read or parsed correctly
                      to determine the seed from the previous run.
    """
    if provided_seed is not None:
        logger.info(f"Using provided seed: {provided_seed}")
        return provided_seed

    logger.info("Seed not provided, attempting to determine from previous run...")
    if output_file_path.exists():
        try:
            lazy_df = pl.scan_csv(
                output_file_path,
                has_header=True,
            )
            last_row_df = lazy_df.select(["seed", "violation_count"]).tail(1).collect()

            if not last_row_df.is_empty():
                last_seed = last_row_df.item(0, "seed")
                last_violation_count_str = last_row_df.item(0, "violation_count")

                try:
                    last_violation_count = int(last_violation_count_str)
                except (ValueError, TypeError) as e:
                    raise RuntimeError(
                        f"Error interpreting 'violation_count' ('{last_violation_count_str}') from last row of '{output_file_path}'. Expected an integer. Cannot proceed."
                    ) from e

                if last_violation_count == 0:
                    effective_seed = int(time.time())
                    logger.info(
                        f"Last run (seed {last_seed}) had 0 violations. Generating new seed: {effective_seed}"
                    )
                    return effective_seed
                else:
                    logger.info(
                        f"Last run (seed {last_seed}) had {last_violation_count} violation(s). Reusing seed: {last_seed}"
                    )
                    return last_seed  # Return the reused seed
            else:
                raise RuntimeError(
                    f"Output file '{output_file_path}' exists but is empty or contains no data rows. Cannot determine seed from previous run. Please provide a seed or ensure the file has valid data."
                )

        except pl.exceptions.NoDataError as e:
            raise RuntimeError(
                f"Output file '{output_file_path}' exists but Polars found no data. Cannot determine seed. Error: {e}"
            ) from e
        except (
            pl.exceptions.SchemaError,
            pl.exceptions.ComputeError,
            KeyError,
            IndexError,
        ) as e:
            raise RuntimeError(
                f"Error reading schema or required columns ('seed', 'violation_count') from '{output_file_path}'. Cannot determine seed. Error: {e}"
            ) from e
        except (
            FileNotFoundError
        ) as e:  # Should not happen due to .exists() but defensive
            raise RuntimeError(
                f"File '{output_file_path}' not found during read, despite existing initially. Error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error reading or processing '{output_file_path}'. Error: {e}"
            ) from e
    else:
        # Output file does not exist - generate initial seed
        effective_seed = int(time.time())
        logger.info(
            f"Output file '{output_file_path}' not found. Generating initial seed: {effective_seed}"
        )
        return effective_seed


def save_single_row_results_appending(single_row_df: pl.DataFrame, output_file: Path):
    """Appends a single-row DataFrame to a CSV file, adding headers if the file doesn't exist."""
    file_exists = output_file.exists()
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Appending result row to {output_file}...")
    try:
        # Open file in append mode and pass file object to write_csv
        with open(output_file, "a") as f:
            single_row_df.write_csv(
                f,  # Pass the file object
                include_header=not file_exists,  # Only write header if file was newly created
            )
        logger.info("Append successful.")

    except Exception as e:
        logger.error(f"Error appending to CSV file {output_file}: {e}")
        logger.exception("Exception details:")  # This includes the traceback
