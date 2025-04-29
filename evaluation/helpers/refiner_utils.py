import traceback
from pathlib import Path

import polars as pl

# --- Refiner Specific Helpers ---


def save_single_row_results_appending(single_row_df: pl.DataFrame, output_file: Path):
    """Appends a single-row DataFrame to a CSV file, adding headers if the file doesn't exist."""
    file_exists = output_file.exists()
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Appending result row to {output_file}...")
    try:
        # Open file in append mode and pass file object to write_csv
        with open(output_file, "a") as f:
            single_row_df.write_csv(
                f,  # Pass the file object
                include_header=not file_exists,  # Only write header if file was newly created
            )
        print("Append successful.")

    except Exception as e:
        print(f"Error appending to CSV file {output_file}: {e}")
        traceback.print_exc()
