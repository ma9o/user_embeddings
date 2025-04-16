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

# Ensure the utils directory is in the Python path for import
# This assumes data_loading.py is in the tests/helpers/ directory
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Import from utils AFTER potentially modifying sys.path
try:
    from user_embeddings.utils.zst_io import read_single_zst_ndjson_chunked, DEFAULT_CHUNK_SIZE 
except ImportError as e:
    print(f"ERROR: Failed to import from user_embeddings.utils.zst_io. Check sys.path and file existence. Error: {e}")
    # Optionally re-raise or handle appropriately
    raise


def _clean_cast_validate_chunk(
    chunk_df: pd.DataFrame, 
    meta_df: pd.DataFrame, 
    chunk_index: int, 
    data_type_name: str
) -> Optional[pd.DataFrame]:
    """Cleans, casts, and validates a single chunk against the meta DataFrame."""
    # Skip empty chunks immediately
    if chunk_df.empty:
        return None

    # Select and add missing columns
    columns_to_keep = [col for col in meta_df.columns if col in chunk_df.columns]
    chunk_df = chunk_df[columns_to_keep].copy() # Use copy to avoid SettingWithCopyWarning
    
    for col in meta_df.columns:
        if col not in chunk_df:
             dtype = meta_df[col].dtype
             # Use appropriate NA type based on Pandas dtype
             if pd.api.types.is_integer_dtype(dtype) or isinstance(dtype, pd.Int64Dtype): 
                 fill_val = pd.NA
             elif pd.api.types.is_bool_dtype(dtype) or isinstance(dtype, pd.BooleanDtype):
                  fill_val = pd.NA
             elif isinstance(dtype, pd.StringDtype):
                  fill_val = pd.NA # Or None, consistent with StringDtype preference?
             elif pd.api.types.is_float_dtype(dtype):
                  fill_val = pd.NA
             else:
                  fill_val = None # Default for object, datetime etc.
             chunk_df[col] = fill_val
                 
    chunk_df = chunk_df[list(meta_df.columns)] # Reorder

    # Cast dtypes carefully
    for col, dtype in meta_df.dtypes.items():
        if chunk_df[col].dtype != dtype:
            try:
                # Specific handling for Pandas extension types
                if isinstance(dtype, pd.BooleanDtype):
                     chunk_df[col] = chunk_df[col].astype('boolean')
                elif isinstance(dtype, pd.StringDtype):
                     chunk_df[col] = chunk_df[col].astype(pd.StringDtype())
                elif isinstance(dtype, pd.Int64Dtype):
                     # Coerce to numeric, then cast to Int64
                     chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')
                     chunk_df[col] = chunk_df[col].astype(pd.Int64Dtype())
                else:
                     # Standard numpy dtypes
                     chunk_df[col] = chunk_df[col].astype(dtype)
            except (ValueError, TypeError, OverflowError) as e:
                print(f"WARNING: Could not cast column '{col}' (dtype: {chunk_df[col].dtype}) to {dtype} in {data_type_name} chunk {chunk_index}. Error: {e}. Filling with NA/None.")
                # Determine fill value again based on target type
                if pd.api.types.is_integer_dtype(dtype) or isinstance(dtype, pd.Int64Dtype) or \
                   pd.api.types.is_bool_dtype(dtype) or isinstance(dtype, pd.BooleanDtype) or \
                   pd.api.types.is_float_dtype(dtype) or isinstance(dtype, pd.StringDtype):
                   fill_val = pd.NA
                else:
                   fill_val = None
                chunk_df[col] = fill_val
                # Retry casting after filling
                try: 
                     if isinstance(dtype, pd.Int64Dtype): # Special case needs numeric conversion first
                         chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').astype(dtype)
                     else:
                         chunk_df[col] = chunk_df[col].astype(dtype) 
                except Exception as e2:
                     print(f"ERROR: Failed second cast attempt for '{col}' to {dtype} after filling. Error: {e2}. Leaving as object.")
                     chunk_df[col] = chunk_df[col].astype('object') # Fallback
                     
    # Final check if resulting dtypes match meta
    current_dtypes = chunk_df.dtypes
    mismatched_cols = []
    for col, target_dtype in meta_df.dtypes.items():
        if current_dtypes[col] != target_dtype:
             mismatched_cols.append((col, current_dtypes[col], target_dtype))
             
    if mismatched_cols:
         print(f"ERROR: Dtype mismatch after casting in chunk {chunk_index} for {data_type_name}.")
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
    data_type_name: str
) -> int:
    """
    Processes a single ZST file chunk by chunk, cleans/casts data according 
    to meta_df, converts to Arrow table using pa_schema, and writes to a 
    Parquet file. Returns the total number of rows written.
    """
    print(f"Processing {data_type_name} from {os.path.basename(zst_file_path)} chunked and caching to {cache_filepath}...")
    writer = None
    total_rows_written = 0
    chunk_generator = read_single_zst_ndjson_chunked(zst_file_path, chunk_size=DEFAULT_CHUNK_SIZE)
    
    try:
        for i, chunk_df in enumerate(chunk_generator):
            
            processed_chunk = _clean_cast_validate_chunk(chunk_df, meta_df, i, data_type_name)
            
            if processed_chunk is None: # Skip empty or problematic chunks
                continue

            # Convert chunk to PyArrow Table
            try:
                 # Ensure index is not included in the table
                 table = pa.Table.from_pandas(processed_chunk, schema=pa_schema, preserve_index=False) 
            except Exception as e:
                 print(f"ERROR: Error converting {data_type_name} chunk {i} to Arrow Table: {e}")
                 print("Problematic chunk dtypes after cleaning/casting:\n", processed_chunk.dtypes)
                 print("Expected Arrow schema:\n", pa_schema)
                 # chunk_df.to_csv(f"problem_chunk_{data_type_name}_{i}.csv")
                 continue # Skip this chunk

            # Initialize writer with the first valid table's schema
            if writer is None:
                print(f"Creating Parquet writer for: {cache_filepath} with schema:\n{table.schema}")
                # Use the schema from the first successfully converted table
                try:
                    writer = pq.ParquetWriter(cache_filepath, table.schema)
                except Exception as e:
                    print(f"ERROR: Failed to create Parquet writer for {cache_filepath}: {e}")
                    raise # Cannot proceed without writer
            
            # Write table - Ensure schema matches writer's schema
            if writer is not None and table.schema.equals(writer.schema):
                 try:
                     writer.write_table(table)
                     total_rows_written += len(processed_chunk)
                 except Exception as e:
                      print(f"ERROR: Failed to write chunk {i} to Parquet file {cache_filepath}: {e}")
                      # Decide: continue or raise? Let's continue but log error.
            elif writer is not None:
                 print(f"ERROR: Schema mismatch in chunk {i}. Expected:\n{writer.schema}\nGot:\n{table.schema}\nSkipping chunk.")

    finally:
        if writer:
            try:
                writer.close()
                print(f"Closed Parquet writer for {cache_filepath}. Total rows written: {total_rows_written}")
            except Exception as e:
                print(f"ERROR: Failed to close Parquet writer for {cache_filepath}: {e}")
                
    return total_rows_written

def _validate_and_cast_ddf(ddf: dd.DataFrame, meta_df: pd.DataFrame, pa_schema: pa.Schema, data_type_name: str) -> dd.DataFrame:
    """Validates schema, adds missing columns, and casts types for a loaded Dask DataFrame."""
    print(f"Validating and casting loaded Dask DataFrame for {data_type_name}...")
    # Convert meta dtypes to a dictionary suitable for astype (handle Pandas specific types)
    meta_dtypes_dict = {}
    for col, dtype in meta_df.dtypes.items():
        if isinstance(dtype, pd.StringDtype):
            # Check the corresponding field in the PyArrow schema
            try:
                arrow_type = pa_schema.field(col).type
                is_arrow_string = pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type)
            except KeyError:
                print(f"Warning: Column '{col}' found in meta_df but not in pa_schema. Assuming basic string.")
                is_arrow_string = False
            meta_dtypes_dict[col] = 'string[pyarrow]' if is_arrow_string else 'string'
        elif isinstance(dtype, pd.BooleanDtype):
            meta_dtypes_dict[col] = 'boolean'
        elif isinstance(dtype, pd.Int64Dtype):
            meta_dtypes_dict[col] = 'Int64' # Dask uses 'Int64' (capital I)
        else:
             meta_dtypes_dict[col] = dtype # Keep numpy dtypes as is

    # Ensure all meta columns exist and add missing ones
    cols_to_add = {col: dtype for col, dtype in meta_dtypes_dict.items() if col not in ddf.columns}
    if cols_to_add:
         print(f"WARNING: Columns missing from loaded Dask DF for {data_type_name}: {list(cols_to_add.keys())}. Adding them with NA/None.")
         for col, dtype in cols_to_add.items():
             # Determine fill value based on target Dask/Pandas type
             fill_val = pd.NA # Dask generally handles pd.NA correctly across types
             ddf[col] = fill_val
             try:
                 # Apply correct type to new column
                 ddf[col] = ddf[col].astype(dtype) 
             except Exception as e:
                 print(f"ERROR: Failed to cast newly added column '{col}' to {dtype}: {e}. Setting as object.")
                 ddf[col] = ddf[col].astype('object') # Fallback?

    # Reorder columns according to meta
    ddf = ddf[list(meta_df.columns)] 
    
    # Cast existing columns if their types don't match the target meta types
    try:
         current_dtypes = ddf.dtypes.to_dict()
         types_to_cast = {}
         for col, target_dtype in meta_dtypes_dict.items():
             current_dtype_str = str(current_dtypes.get(col, 'N/A'))
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
                 elif isinstance(target_pd_dtype, pd.BooleanDtype) and pd.api.types.is_bool_dtype(current_pd_dtype):
                      is_match = True
                 # Allow casting from object if target is string
                 elif isinstance(target_pd_dtype, pd.StringDtype) and pd.api.types.is_object_dtype(current_pd_dtype):
                      is_match = False # Needs cast
                 # Allow casting from float64 to Int64 (will handle NaNs)
                 elif isinstance(target_pd_dtype, pd.Int64Dtype) and pd.api.types.is_float_dtype(current_pd_dtype):
                      is_match = False # Needs cast
                 # Allow casting from int64 to Int64
                 elif isinstance(target_pd_dtype, pd.Int64Dtype) and pd.api.types.is_integer_dtype(current_pd_dtype):
                      is_match = False # Needs cast for nullability

             if col in current_dtypes and not is_match:
                  types_to_cast[col] = target_dtype
             elif col not in current_dtypes:
                  print(f"ERROR: Column '{col}' still missing after attempted add for {data_type_name}.")

         if types_to_cast:
              print(f"Casting {data_type_name} columns to match meta: {types_to_cast}")
              # Apply casts - Dask handles this lazily
              ddf = ddf.astype(types_to_cast)

    except Exception as e:
         print(f"ERROR: Failed during dtype comparison/casting preparation for {data_type_name}: {e}")
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
        pytest.fail(mismatch_msg) # Use pytest.fail in test helpers

    print(f"Successfully validated and cast Dask DataFrame for {data_type_name}. Final Columns: {list(ddf.columns)}, Final Dtypes: {ddf.dtypes.to_dict()}")
    return ddf

def load_or_create_cached_ddf(
    data_path: str, 
    file_pattern: str, 
    cache_dir: str, 
    meta_df: pd.DataFrame, 
    pa_schema: pa.Schema,
    data_type_name: str # e.g., "comments", "submissions"
) -> dd.DataFrame:
    """
    Loads a Dask DataFrame from a Parquet cache if it exists, otherwise
    processes the first matching .zst file chunk by chunk using helper functions,
    creates the cache, and then loads it. Ensures schema consistency.
    Public wrapper for the helper functions.
    """
    if not os.path.exists(data_path):
         msg = f"{data_type_name.capitalize()} directory not found: {data_path}"
         print(f"ERROR: {msg}")
         pytest.fail(msg)

    try:
        filepaths = glob.glob(os.path.join(data_path, file_pattern))
        if not filepaths:
            raise FileNotFoundError(f"No files matching '{file_pattern}' found in {data_path}")
        
        first_file_path = filepaths[0]
        base_filename = os.path.splitext(os.path.basename(first_file_path))[0]
        cache_filename = f"{base_filename}_{data_type_name}.parquet"
        cache_filepath = os.path.join(cache_dir, cache_filename)
        
        n_partitions = os.cpu_count() or 4 # Calculate once
        ddf = None # Initialize ddf

        if os.path.exists(cache_filepath):
            print(f"Loading {data_type_name} from cache: {cache_filepath}")
            if os.path.getsize(cache_filepath) > 0:
                try:
                    ddf = dd.read_parquet(cache_filepath, engine='pyarrow')
                    print(f"Loaded {data_type_name} from cache with {ddf.npartitions} partitions.")
                except Exception as e:
                    print(f"ERROR: Failed to load Parquet cache {cache_filepath}: {e}")
                    print("Attempting to regenerate cache...")
                    ddf = None # Force cache regeneration
                    try: 
                        os.remove(cache_filepath)
                        print(f"Removed corrupted cache file: {cache_filepath}")
                    except OSError as rm_err:
                        print(f"ERROR: Failed to remove corrupted cache file {cache_filepath}: {rm_err}")
                        pytest.fail(f"Failed to remove corrupted cache file: {rm_err}")
            else:
                print(f"WARNING: Cache file {cache_filepath} is empty. Will regenerate.")
                ddf = None # Ensure cache is regenerated
        else: # Cache does not exist
             print(f"Cache not found for {os.path.basename(first_file_path)}. Will create.")
             ddf = None # Ensure cache is created

        
        if ddf is None: # Cache needs regeneration or creation
            total_rows_written = _process_zst_to_parquet(
                zst_file_path=first_file_path,
                cache_filepath=cache_filepath,
                meta_df=meta_df,
                pa_schema=pa_schema,
                data_type_name=data_type_name
            )
                    
            # Load Dask DataFrame from the newly created cache (if rows were written)
            if total_rows_written > 0 and os.path.exists(cache_filepath) and os.path.getsize(cache_filepath) > 0:
                 print(f"Loading the newly created {data_type_name} cache: {cache_filepath}")
                 try:
                     ddf = dd.read_parquet(cache_filepath, engine='pyarrow')
                     print(f"Loaded newly created {data_type_name} cache with {ddf.npartitions} partitions.")
                 except Exception as e:
                     print(f"ERROR: Failed to load newly created Parquet cache {cache_filepath}: {e}")
                     pytest.fail(f"Failed to load newly created cache: {e}")
            else: # Handle case where cache file wasn't created or is empty after processing
                 print(f"WARNING: No valid rows written or cache file missing/empty after processing for {cache_filepath}. Creating empty DataFrame.")
                 ddf = None # Will be handled below

        # Create empty DataFrame if loading failed or no data was processed/written
        if ddf is None:
            pandas_df = pd.DataFrame(columns=meta_df.columns).astype(meta_df.dtypes.to_dict())
            print(f"Creating empty Dask DataFrame for {data_type_name} with {n_partitions} partitions and correct meta.")
            ddf = dd.from_pandas(pandas_df, npartitions=n_partitions)

        # --- Repartitioning --- 
        if ddf.npartitions < n_partitions:
            print(f"Repartitioning Dask DataFrame from {ddf.npartitions} to {n_partitions} partitions...")
            ddf = ddf.repartition(npartitions=n_partitions)
            
        # --- Final Schema Validation and Casting --- 
        ddf = _validate_and_cast_ddf(ddf, meta_df, pa_schema, data_type_name)

        return ddf
        
    except FileNotFoundError as e:
        print(f"ERROR: {str(e)}") 
        pytest.fail(str(e))
    except Exception as e:
        print(f"ERROR: Failed to load/cache {data_type_name}: {e}") 
        traceback.print_exc()
        pytest.fail(f"Failed to load/cache {data_type_name}: {e}") 