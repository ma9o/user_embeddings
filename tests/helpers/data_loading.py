import pytest
import dask.dataframe as dd
import pandas as pd
import os
import glob
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import traceback
from typing import Optional
import time

# Ensure the utils directory is in the Python path for import
# This assumes data_loading.py is in the tests/helpers/ directory
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Import from utils AFTER potentially modifying sys.path
try:
    from user_embeddings.utils.zst_io import (
        read_single_zst_ndjson_chunked,
        DEFAULT_CHUNK_SIZE,
    )
except ImportError as e:
    print(
        f"ERROR: Failed to import from user_embeddings.utils.zst_io. Check sys.path and file existence. Error: {e}"
    )
    # Optionally re-raise or handle appropriately
    raise


def _clean_cast_validate_chunk(
    chunk_df: pd.DataFrame, meta_df: pd.DataFrame, chunk_index: int, data_type_name: str
) -> Optional[pd.DataFrame]:
    """Cleans, casts, and validates a single chunk against the meta DataFrame."""
    # Skip empty chunks immediately
    if chunk_df.empty:
        return None

    # Select and add missing columns
    columns_to_keep = [col for col in meta_df.columns if col in chunk_df.columns]
    chunk_df = chunk_df[
        columns_to_keep
    ].copy()  # Use copy to avoid SettingWithCopyWarning

    for col in meta_df.columns:
        if col not in chunk_df:
            dtype = meta_df[col].dtype
            # Use appropriate NA type based on Pandas dtype
            if pd.api.types.is_integer_dtype(dtype) or isinstance(dtype, pd.Int64Dtype):
                fill_val = pd.NA
            elif pd.api.types.is_bool_dtype(dtype) or isinstance(
                dtype, pd.BooleanDtype
            ):
                fill_val = pd.NA
            elif isinstance(dtype, pd.StringDtype):
                fill_val = pd.NA  # Or None, consistent with StringDtype preference?
            elif pd.api.types.is_float_dtype(dtype):
                fill_val = pd.NA
            else:
                fill_val = None  # Default for object, datetime etc.
            chunk_df[col] = fill_val

    chunk_df = chunk_df[list(meta_df.columns)]  # Reorder

    # Cast dtypes carefully
    for col, dtype in meta_df.dtypes.items():
        if chunk_df[col].dtype != dtype:
            try:
                # Specific handling for Pandas extension types
                if isinstance(dtype, pd.BooleanDtype):
                    chunk_df[col] = chunk_df[col].astype("boolean")
                elif isinstance(dtype, pd.StringDtype):
                    chunk_df[col] = chunk_df[col].astype(pd.StringDtype())
                elif isinstance(dtype, pd.Int64Dtype):
                    # Coerce to numeric, then cast to Int64
                    chunk_df[col] = pd.to_numeric(chunk_df[col], errors="coerce")
                    chunk_df[col] = chunk_df[col].astype(pd.Int64Dtype())
                else:
                    # Standard numpy dtypes
                    chunk_df[col] = chunk_df[col].astype(dtype)
            except (ValueError, TypeError, OverflowError) as e:
                print(
                    f"WARNING: Could not cast column '{col}' (dtype: {chunk_df[col].dtype}) to {dtype} in {data_type_name} chunk {chunk_index}. Error: {e}. Filling with NA/None."
                )
                # Determine fill value again based on target type
                if (
                    pd.api.types.is_integer_dtype(dtype)
                    or isinstance(dtype, pd.Int64Dtype)
                    or pd.api.types.is_bool_dtype(dtype)
                    or isinstance(dtype, pd.BooleanDtype)
                    or pd.api.types.is_float_dtype(dtype)
                    or isinstance(dtype, pd.StringDtype)
                ):
                    fill_val = pd.NA
                else:
                    fill_val = None
                chunk_df[col] = fill_val
                # Retry casting after filling
                try:
                    if isinstance(
                        dtype, pd.Int64Dtype
                    ):  # Special case needs numeric conversion first
                        chunk_df[col] = pd.to_numeric(
                            chunk_df[col], errors="coerce"
                        ).astype(dtype)
                    # Handle StringDtype explicitly here too
                    elif isinstance(dtype, pd.StringDtype):
                        chunk_df[col] = chunk_df[col].astype(pd.StringDtype())
                    else:
                        chunk_df[col] = chunk_df[col].astype(dtype)
                except Exception as e2:
                    print(
                        f"ERROR: Failed second cast attempt for '{col}' to {dtype} after filling. Error: {e2}. Leaving as object."
                    )
                    chunk_df[col] = chunk_df[col].astype("object")  # Fallback

    # Specific handling for 'edited' column AFTER main casting attempts
    # Convert to nullable string to handle mixed types (bool/int/None) reliably for Arrow
    if "edited" in chunk_df.columns:
        try:
            # Ensure it's StringDtype before passing to Arrow
            if chunk_df["edited"].dtype != pd.StringDtype():
                chunk_df["edited"] = chunk_df["edited"].astype(pd.StringDtype())
        except Exception as e:
            print(
                f"ERROR: Failed to cast 'edited' column to StringDtype in chunk {chunk_index}. Error: {e}"
            )
            # If conversion fails even to string, maybe drop or return None?
            # Returning None to skip the chunk is safer.
            return None

    # Final check if resulting dtypes match meta
    current_dtypes = chunk_df.dtypes
    mismatched_cols = []
    for col, target_dtype in meta_df.dtypes.items():
        if current_dtypes[col] != target_dtype:
            mismatched_cols.append((col, current_dtypes[col], target_dtype))

    if mismatched_cols:
        print(
            f"ERROR: Dtype mismatch after casting in chunk {chunk_index} for {data_type_name}."
        )
        for col, current, target in mismatched_cols:
            print(f"  Column '{col}': Expected {target}, Got {current}")
        # Decide: return None to skip chunk, or raise error?
        # Let's return None for now to allow processing to continue.
        return None

    return chunk_df


def _process_zst_to_parquet(
    zst_file_path: str,
    cache_filepath: str,
    meta_df: pd.DataFrame,
    pa_schema: pa.Schema,
    data_type_name: str,
) -> int:
    """
    Processes a single ZST file chunk by chunk, cleans/casts according to meta_df,
    converts to Arrow table using pa_schema, and writes to a Parquet file.
    Returns the total number of rows written.
    """
    print(
        f"Processing {data_type_name} from {os.path.basename(zst_file_path)} using schema and caching to {cache_filepath}..."
    )
    writer = None
    total_rows_written = 0
    chunk_generator = read_single_zst_ndjson_chunked(
        zst_file_path, chunk_size=DEFAULT_CHUNK_SIZE
    )

    try:
        # Use the provided pa_schema to initialize the writer
        print(
            f"Creating Parquet writer for: {cache_filepath} with provided schema:\n{pa_schema}"
        )
        try:
            writer = pq.ParquetWriter(cache_filepath, pa_schema)
        except Exception as e:
            print(
                f"ERROR: Failed to create Parquet writer for {cache_filepath} with provided schema: {e}"
            )
            raise  # Cannot proceed without writer

        for i, chunk_df in enumerate(chunk_generator):
            # Clean, cast, and validate chunk against meta_df
            # This ensures chunk conforms to the schema before converting to Arrow
            processed_chunk = _clean_cast_validate_chunk(
                chunk_df, meta_df, i, data_type_name
            )

            if (
                processed_chunk is None
            ):  # Skip empty or problematic chunks after cleaning
                print(
                    f"Skipping invalid or empty processed chunk {i} for {data_type_name}."
                )
                continue

            # Convert processed chunk to PyArrow Table using the provided schema
            try:
                # Ensure index is not included; use provided schema
                table = pa.Table.from_pandas(
                    processed_chunk, schema=pa_schema, preserve_index=False
                )
            except Exception as e:
                print(
                    f"ERROR: Error converting processed {data_type_name} chunk {i} to Arrow Table using provided schema: {e}"
                )
                print(f"Processed chunk {i} dtypes:\n{processed_chunk.dtypes}")
                print(f"Expected Arrow schema:\n{pa_schema}")
                # processed_chunk.to_csv(f"problem_chunk_{data_type_name}_{i}.csv") # Optional debug output
                continue  # Skip this chunk

            # Write table (schema should inherently match writer's schema now)
            try:
                writer.write_table(table)
                total_rows_written += len(processed_chunk)
            except Exception as e:
                print(
                    f"ERROR: Failed to write chunk {i} to Parquet file {cache_filepath}: {e}"
                )
                # Decide: continue or raise? Let's continue but log error.

    finally:
        if writer:
            try:
                writer.close()
                print(
                    f"Closed Parquet writer for {cache_filepath}. Total rows written: {total_rows_written}"
                )
            except Exception as e:
                print(
                    f"ERROR: Failed to close Parquet writer for {cache_filepath}: {e}"
                )

    return total_rows_written


def _validate_and_cast_ddf(
    ddf: dd.DataFrame, meta_df: pd.DataFrame, pa_schema: pa.Schema, data_type_name: str
) -> dd.DataFrame:
    """Validates schema, adds missing columns, and casts types for a loaded Dask DataFrame."""
    print(f"Validating and casting loaded Dask DataFrame for {data_type_name}...")
    # Convert meta dtypes to a dictionary suitable for astype (handle Pandas specific types)
    meta_dtypes_dict = {}
    for col, dtype in meta_df.dtypes.items():
        if isinstance(dtype, pd.StringDtype):
            # Check the corresponding field in the PyArrow schema
            try:
                arrow_type = pa_schema.field(col).type
                is_arrow_string = pa.types.is_string(
                    arrow_type
                ) or pa.types.is_large_string(arrow_type)
            except KeyError:
                print(
                    f"Warning: Column '{col}' found in meta_df but not in pa_schema. Assuming basic string."
                )
                is_arrow_string = False
            meta_dtypes_dict[col] = "string[pyarrow]" if is_arrow_string else "string"
        elif isinstance(dtype, pd.BooleanDtype):
            meta_dtypes_dict[col] = "boolean"
        elif isinstance(dtype, pd.Int64Dtype):
            meta_dtypes_dict[col] = "Int64"  # Dask uses 'Int64' (capital I)
        else:
            meta_dtypes_dict[col] = dtype  # Keep numpy dtypes as is

    # Ensure all meta columns exist and add missing ones
    cols_to_add = {
        col: dtype for col, dtype in meta_dtypes_dict.items() if col not in ddf.columns
    }
    if cols_to_add:
        print(
            f"WARNING: Columns missing from loaded Dask DF for {data_type_name}: {list(cols_to_add.keys())}. Adding them with NA/None."
        )
        for col, dtype in cols_to_add.items():
            # Determine fill value based on target Dask/Pandas type
            fill_val = pd.NA  # Dask generally handles pd.NA correctly across types
            ddf[col] = fill_val
            try:
                # Apply correct type to new column
                ddf[col] = ddf[col].astype(dtype)
            except Exception as e:
                print(
                    f"ERROR: Failed to cast newly added column '{col}' to {dtype}: {e}. Setting as object."
                )
                ddf[col] = ddf[col].astype("object")  # Fallback?

    # Reorder columns according to meta
    ddf = ddf[list(meta_df.columns)]

    # Cast existing columns if their types don't match the target meta types
    try:
        current_dtypes = ddf.dtypes.to_dict()
        types_to_cast = {}
        for col, target_dtype in meta_dtypes_dict.items():
            current_dtype_str = str(current_dtypes.get(col, "N/A"))
            target_dtype_str = str(target_dtype)

            # Explicitly check type compatibility before casting
            is_match = False
            try:
                current_pd_dtype = pd.api.types.pandas_dtype(current_dtype_str)
                target_pd_dtype = pd.api.types.pandas_dtype(target_dtype_str)
            except TypeError:
                # If conversion fails, assume they don't match
                is_match = False
            else:
                if current_pd_dtype == target_pd_dtype:
                    is_match = True
                # Add specific compatible mappings (e.g., Dask reads Parquet bool as numpy bool)
                elif isinstance(
                    target_pd_dtype, pd.BooleanDtype
                ) and pd.api.types.is_bool_dtype(current_pd_dtype):
                    is_match = True
                # Allow casting from object if target is string
                elif isinstance(
                    target_pd_dtype, pd.StringDtype
                ) and pd.api.types.is_object_dtype(current_pd_dtype):
                    is_match = False  # Needs cast
                # Allow casting from float64 to Int64 (will handle NaNs)
                elif isinstance(
                    target_pd_dtype, pd.Int64Dtype
                ) and pd.api.types.is_float_dtype(current_pd_dtype):
                    is_match = False  # Needs cast
                # Allow casting from int64 to Int64
                elif isinstance(
                    target_pd_dtype, pd.Int64Dtype
                ) and pd.api.types.is_integer_dtype(current_pd_dtype):
                    is_match = False  # Needs cast for nullability

            if col in current_dtypes and not is_match:
                types_to_cast[col] = target_dtype
            elif col not in current_dtypes:
                print(
                    f"ERROR: Column '{col}' still missing after attempted add for {data_type_name}."
                )

        if types_to_cast:
            print(f"Casting {data_type_name} columns to match meta: {types_to_cast}")
            # Apply casts - Dask handles this lazily
            ddf = ddf.astype(types_to_cast)

    except Exception as e:
        print(
            f"ERROR: Failed during dtype comparison/casting preparation for {data_type_name}: {e}"
        )
        print(f"Dask dtypes: {ddf.dtypes}")
        print(f"Meta dtypes dict: {meta_dtypes_dict}")
        pytest.fail(f"Dtype casting setup failed for {data_type_name}: {e}")

    # Final check on columns vs meta after all operations
    final_columns = list(ddf.columns)
    meta_columns = list(meta_df.columns)
    if final_columns != meta_columns:
        mismatch_msg = (
            f"Loaded {data_type_name} final columns {final_columns} "
            f"do not match meta {meta_columns} after processing."
        )
        print(f"ERROR: {mismatch_msg}")
        print(f"ERROR: Dask dtypes: {ddf.dtypes}")
        print(f"ERROR: Meta dtypes: {meta_df.dtypes}")
        print(f"Columns in DDF but not meta: {set(final_columns) - set(meta_columns)}")
        print(f"Columns in meta but not DDF: {set(meta_columns) - set(final_columns)}")
        pytest.fail(mismatch_msg)  # Use pytest.fail in test helpers

    print(
        f"Successfully validated and cast Dask DataFrame for {data_type_name}. Final Columns: {list(ddf.columns)}, Final Dtypes: {ddf.dtypes.to_dict()}"
    )
    return ddf


def load_or_create_cached_ddf(
    data_path: str,
    file_pattern: str,
    cache_dir: str,
    meta_df: pd.DataFrame,
    pa_schema: pa.Schema,
    data_type_name: str,
) -> dd.DataFrame:
    """
    Loads a Dask DataFrame from a cached Parquet file if it exists,
    otherwise processes the first matching ZST file using an explicit schema
    to create the cache. The cache filename matches the source ZST filename.

    Args:
        data_path: Directory containing the source ZST files.
        file_pattern: Glob pattern for the ZST files (e.g., "RC_*.zst").
        cache_dir: Directory where the Parquet cache should be stored.
        meta_df: Pandas DataFrame defining the target schema and types.
        pa_schema: PyArrow schema corresponding to meta_df for Parquet writing.
        data_type_name: Name for the dataset type (e.g., "comments", "submissions").

    Returns:
        A Dask DataFrame representing the loaded or created data, conforming to the meta.

    Raises:
        FileNotFoundError: If the data_path does not exist.
        ValueError: If no ZST files are found.
        AssertionError: If final DataFrame schema doesn't match meta.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_type_name.capitalize()} data directory not found: {data_path}"
        )

    # Find source ZST files
    zst_files = glob.glob(os.path.join(data_path, file_pattern))
    if not zst_files:
        raise ValueError(f"No ZST files matching '{file_pattern}' found in {data_path}")

    # Select the first file to process (consistent with previous logic)
    zst_file_to_process = zst_files[0]
    print(
        f"Target ZST file for caching/loading: {os.path.basename(zst_file_to_process)}"
    )

    # Determine cache filename based on the source ZST filename
    base_filename = os.path.splitext(os.path.basename(zst_file_to_process))[0]
    cache_filepath = os.path.join(cache_dir, f"{base_filename}.parquet")
    print(f"Expected cache file path: {cache_filepath}")

    ddf = None  # Initialize ddf

    if os.path.exists(cache_filepath):
        print(f"Loading {data_type_name} from cached Parquet file: {cache_filepath}")
        try:
            start_time = time.time()
            # Load from cache
            ddf_loaded = dd.read_parquet(cache_filepath)
            end_time = time.time()
            print(
                f"Loaded {data_type_name} cache in {end_time - start_time:.2f}s. Initial Columns: {ddf_loaded.columns}"
            )

            # Validate and cast the loaded DataFrame against the expected meta schema
            ddf = _validate_and_cast_ddf(ddf_loaded, meta_df, pa_schema, data_type_name)
            print(f"Validated and cast loaded cache for {data_type_name}.")
            # return ddf # Return validated ddf

        except Exception as e:
            print(
                f"ERROR: Failed to load or validate cached Parquet file {cache_filepath}: {e}. Attempting to rebuild."
            )
            ddf = None  # Force rebuild
            # Optionally remove the corrupted cache file
            try:
                os.remove(cache_filepath)
            except OSError as remove_err:
                print(
                    f"Warning: Failed to remove corrupted cache file {cache_filepath}: {remove_err}"
                )
    else:
        print(f"Cached Parquet file not found: {cache_filepath}")
        ddf = None  # Signal to create cache

    # --- Cache not found, failed to load/validate, or rebuild forced ---
    if ddf is None:
        # Already selected the file: zst_file_to_process
        # zst_files = glob.glob(os.path.join(data_path, file_pattern))
        # if not zst_files:
        #     raise ValueError(f"No ZST files matching '{file_pattern}' found in {data_path}")

        print(
            f"Processing {data_type_name} from ZST file: {os.path.basename(zst_file_to_process)} to create cache: {cache_filepath}"
        )
        # Process only the first file for caching for speed in testing -- Redundant comment
        # zst_file_to_process = zst_files[0]
        # print(f"Processing ONLY the first file for caching: {os.path.basename(zst_file_to_process)}")

        start_time = time.time()
        total_rows = _process_zst_to_parquet(
            zst_file_path=zst_file_to_process,
            cache_filepath=cache_filepath,
            meta_df=meta_df,  # Pass meta
            pa_schema=pa_schema,  # Pass schema
            data_type_name=data_type_name,
        )
        end_time = time.time()

        if total_rows > 0 and os.path.exists(cache_filepath):
            print(
                f"Successfully created Parquet cache for {data_type_name} ({total_rows} rows) at {cache_filepath} in {end_time - start_time:.2f}s"
            )
            # Load the newly created cache file
            print(f"Loading newly created {data_type_name} cache...")
            try:
                ddf_loaded = dd.read_parquet(cache_filepath)
                # Validate the newly created cache file against meta
                ddf = _validate_and_cast_ddf(
                    ddf_loaded, meta_df, pa_schema, data_type_name
                )
                print(f"Validated newly created cache. Final Columns: {ddf.columns}")
                # return ddf
            except Exception as e:
                print(
                    f"ERROR: Failed to load or validate newly created cache file {cache_filepath}: {e}. Handling error."
                )
                ddf = None  # Reset ddf on load/validation failure
        else:
            print(
                f"Warning: Parquet cache creation for {data_type_name} resulted in 0 rows or file missing. Check source/processing."
            )
            ddf = None  # Ensure empty ddf is created

    # If ddf is still None (creation/loading/validation failed), create an empty one matching meta
    if ddf is None:
        print(
            f"Creating empty Dask DataFrame for {data_type_name} matching meta schema."
        )
        # Create empty Pandas DataFrame with correct columns and types from meta_df
        empty_pdf = pd.DataFrame(columns=meta_df.columns).astype(
            meta_df.dtypes.to_dict()
        )
        ddf = dd.from_pandas(empty_pdf, npartitions=1)
        print(
            f"Created empty DataFrame with columns: {ddf.columns} and types: {ddf.dtypes.to_dict()}"
        )

    # Final check: Ensure the returned ddf is not None
    if ddf is None:
        raise RuntimeError(
            f"Failed to load or create a valid Dask DataFrame for {data_type_name}"
        )

    return ddf
