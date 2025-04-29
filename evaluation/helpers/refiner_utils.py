import traceback
from pathlib import Path

import polars as pl

# --- Refiner Specific Helpers ---


def save_single_row_results_appending(single_row_df: pl.DataFrame, output_file: Path):
    """Appends a single-row DataFrame to a CSV file, adding headers if the file doesn't exist."""
    # ... (Copy implementation from evaluation_utils.py) ...
    file_exists = output_file.exists()
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Appending result row to {output_file}...")
    try:
        # Polars write_csv doesn't reliably append across versions/OS.
        # Convert to CSV string and append manually.
        # Get string representation, include header only if file is new
        csv_string = single_row_df.write_csv(include_header=not file_exists)
        # Remove the header line from the string if the file already exists
        if file_exists:
            csv_string_lines = csv_string.splitlines()
            if len(csv_string_lines) > 1:
                csv_string = (
                    "\n".join(csv_string_lines[1:]) + "\n"
                )  # Keep newline at end
            else:
                # Handle case where dataframe might produce only header or empty string
                csv_string = ""  # Don't write anything if only header was present

        # Only write if there's content to write
        if csv_string:
            with open(output_file, "a") as f:
                f.write(csv_string)
            print("Append successful.")
        else:
            print("No data to append (empty or header-only DataFrame).")

    except Exception as e:
        print(f"Error appending to CSV file {output_file}: {e}")
        traceback.print_exc()
