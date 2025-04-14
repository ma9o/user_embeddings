import pytest
import dask.dataframe as dd
import pandas as pd
import os
import glob 
import pyarrow as pa 
import pyarrow.parquet as pq 
import sys
import traceback

# Ensure the utils directory is in the Python path for import
# This assumes test_helpers.py is in the tests/ directory
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Import from utils AFTER potentially modifying sys.path
try:
    from utils.zst_io import read_single_zst_ndjson_chunked, DEFAULT_CHUNK_SIZE 
except ImportError as e:
    print(f"ERROR: Failed to import from utils.zst_io. Check sys.path and file existence. Error: {e}")
    # Optionally re-raise or handle appropriately
    raise

def _load_or_create_cached_ddf(
    data_path: str, 
    file_pattern: str, 
    cache_dir: str, 
    meta_df: pd.DataFrame, 
    pa_schema: pa.Schema,
    data_type_name: str # e.g., "comments", "submissions"
) -> dd.DataFrame:
    """
    Loads a Dask DataFrame from a Parquet cache if it exists, otherwise
    processes the first matching .zst file chunk by chunk, creates the cache, 
    and then loads it. Ensures schema consistency against the provided meta.
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

        if os.path.exists(cache_filepath):
            print(f"Loading {data_type_name} from cache: {cache_filepath}")
            print(f"Loading directly into Dask DataFrame from {cache_filepath} with {n_partitions} partitions...")
            # Ensure cache isn't empty before reading
            if os.path.getsize(cache_filepath) > 0:
                ddf = dd.read_parquet(cache_filepath, engine='pyarrow')
            else:
                print(f"WARNING: Cache file {cache_filepath} is empty. Creating empty Dask DataFrame.")
                ddf = None # Flag to create empty DF later

            if ddf is not None and ddf.npartitions < n_partitions:
                print(f"Repartitioning Dask DataFrame from {ddf.npartitions} to {n_partitions} partitions...")
                ddf = ddf.repartition(npartitions=n_partitions)
            elif ddf is None: # Create empty DF if cache was empty
                 pandas_df = pd.DataFrame(columns=meta_df.columns).astype(meta_df.dtypes.to_dict())
                 print(f"Creating empty Dask DataFrame with {n_partitions} partitions and correct meta.")
                 ddf = dd.from_pandas(pandas_df, npartitions=n_partitions)
                 # No need to cast again here, from_pandas respects the astype
        
        else:
            print(f"Cache not found for {os.path.basename(first_file_path)}. Processing {data_type_name} chunked and caching...")
            writer = None
            total_rows_written = 0
            chunk_generator = read_single_zst_ndjson_chunked(first_file_path, chunk_size=DEFAULT_CHUNK_SIZE)
            
            try:
                for i, chunk_df in enumerate(chunk_generator):
                    # Skip empty chunks immediately
                    if chunk_df.empty:
                        continue

                    columns_to_keep = [col for col in meta_df.columns if col in chunk_df.columns]
                    chunk_df = chunk_df[columns_to_keep]
                    
                    for col in meta_df.columns:
                        if col not in chunk_df:
                             dtype = meta_df[col].dtype
                             if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
                                 chunk_df[col] = pd.NA
                             else:
                                 chunk_df[col] = None
                                 
                    chunk_df = chunk_df[list(meta_df.columns)] # Reorder

                    # Cast dtypes carefully
                    for col, dtype in meta_df.dtypes.items():
                        if chunk_df[col].dtype != dtype:
                            try:
                                # Handle specific casts like Int64 to avoid object fallback if possible
                                if isinstance(dtype, pd.BooleanDtype):
                                     chunk_df[col] = chunk_df[col].astype('boolean')
                                elif isinstance(dtype, pd.StringDtype):
                                     chunk_df[col] = chunk_df[col].astype(pd.StringDtype())
                                elif isinstance(dtype, pd.Int64Dtype):
                                     # Convert to float first if needed for Int64 conversion from non-int
                                     if not pd.api.types.is_integer_dtype(chunk_df[col].dtype):
                                          chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')
                                     chunk_df[col] = chunk_df[col].astype(pd.Int64Dtype())
                                else:
                                     chunk_df[col] = chunk_df[col].astype(dtype)
                            except (ValueError, TypeError, OverflowError) as e:
                                print(f"WARNING: Could not cast column '{col}' (dtype: {chunk_df[col].dtype}) to {dtype} in {data_type_name} chunk {i}. Error: {e}. Filling with NA/None.")
                                fill_val = pd.NA if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) else None
                                chunk_df[col] = fill_val
                                # Try casting the filled column
                                try: 
                                     chunk_df[col] = chunk_df[col].astype(dtype) 
                                except: 
                                     print(f"ERROR: Failed second cast attempt for '{col}' to {dtype}. Leaving as NA/None.")
                                     # Ensure it has *some* dtype to avoid issues later, fallback object
                                     if fill_val is pd.NA and not pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                                         chunk_df[col] = chunk_df[col].astype('object') # Fallback if NA fill + non-numeric/bool target

                    # Convert chunk to PyArrow Table
                    try:
                         # Ensure index is not included in the table
                         table = pa.Table.from_pandas(chunk_df, schema=pa_schema, preserve_index=False) 
                    except Exception as e:
                         print(f"ERROR: Error converting {data_type_name} chunk {i} to Arrow Table: {e}")
                         print("Problematic chunk dtypes:\n", chunk_df.dtypes)
                         print("Expected Arrow schema:\n", pa_schema)
                         # Consider saving problematic chunk for inspection
                         # chunk_df.to_csv(f"problem_chunk_{data_type_name}_{i}.csv")
                         continue # Skip this chunk

                    # Initialize writer with the first valid table's schema
                    if writer is None:
                        print(f"Creating Parquet writer for: {cache_filepath} with schema:\n{table.schema}")
                        # Use the schema from the first successfully converted table
                        writer = pq.ParquetWriter(cache_filepath, table.schema) 
                    
                    # Write table - Ensure schema matches writer's schema
                    if table.schema.equals(writer.schema):
                         writer.write_table(table)
                         total_rows_written += len(chunk_df)
                    else:
                         print(f"ERROR: Schema mismatch in chunk {i}. Expected:\n{writer.schema}\nGot:\n{table.schema}\nSkipping chunk.")


            finally:
                if writer:
                    writer.close()
                    print(f"Closed Parquet writer. Total rows written: {total_rows_written}")
                    
            # Load Dask DataFrame from the newly created cache
            if total_rows_written > 0 and os.path.exists(cache_filepath) and os.path.getsize(cache_filepath) > 0:
                 print(f"Loading the newly created {data_type_name} cache: {cache_filepath}")
                 ddf = dd.read_parquet(cache_filepath, engine='pyarrow')
                 if ddf.npartitions < n_partitions:
                     print(f"Repartitioning newly cached Dask DataFrame from {ddf.npartitions} to {n_partitions} partitions...")
                     ddf = ddf.repartition(npartitions=n_partitions)
            else: # Handle case where cache file wasn't created or is empty
                 print(f"WARNING: No valid rows written or cache file missing/empty for {cache_filepath}. Creating empty DataFrame.")
                 pandas_df = pd.DataFrame(columns=meta_df.columns).astype(meta_df.dtypes.to_dict())
                 print(f"Creating empty Dask DataFrame with {n_partitions} partitions and correct meta.")
                 ddf = dd.from_pandas(pandas_df, npartitions=n_partitions)
                 # Ensure meta applied to empty ddf created via pandas
                 ddf = ddf.astype(meta_df.dtypes.to_dict())


        # --- Schema Validation and Casting for loaded Dask DataFrame ---
        # Convert meta dtypes to a dictionary suitable for astype (handle Pandas specific types)
        meta_dtypes_dict = {}
        for col, dtype in meta_df.dtypes.items():
            if isinstance(dtype, pd.StringDtype):
                meta_dtypes_dict[col] = 'string[pyarrow]' 
            elif isinstance(dtype, pd.BooleanDtype):
                meta_dtypes_dict[col] = 'boolean'
            # Add handling for Int64Dtype if you use it in meta
            elif isinstance(dtype, pd.Int64Dtype):
                meta_dtypes_dict[col] = 'Int64' # Dask uses 'Int64' (capital I)
            else:
                 meta_dtypes_dict[col] = dtype # Keep numpy dtypes as is

        # Ensure all meta columns exist and add missing ones
        cols_to_add = {col: dtype for col, dtype in meta_dtypes_dict.items() if col not in ddf.columns}
        if cols_to_add:
             print(f"WARNING: Columns missing from cached Parquet file or loaded Dask DF for {data_type_name}: {list(cols_to_add.keys())}. Adding them.")
             for col, dtype in cols_to_add.items():
                 # Determine fill value based on target Dask/Pandas type
                 if dtype == 'boolean':
                     fill_val = pd.NA 
                 elif dtype == 'Int64':
                      fill_val = pd.NA
                 elif pd.api.types.is_numeric_dtype(dtype): # Handle numpy int/float
                      fill_val = pd.NA # Usually safe for numeric
                 else: # string, object etc.
                      fill_val = None 
                 ddf[col] = fill_val
                 try:
                     # Apply correct type to new column
                     ddf[col] = ddf[col].astype(dtype) 
                 except Exception as e:
                     print(f"ERROR: Failed to cast newly added column '{col}' to {dtype}: {e}")
                     ddf[col] = ddf[col].astype('object') # Fallback?

        # Reorder columns according to meta
        ddf = ddf[list(meta_df.columns)] 
        
        # Cast existing columns if their types don't match the target meta types
        try:
             current_dtypes = ddf.dtypes.to_dict()
             types_to_cast = {}
             for col, target_dtype in meta_dtypes_dict.items():
                 # Need careful comparison as Dask dtypes might differ slightly (e.g., numpy vs pandas extension)
                 current_dtype_str = str(current_dtypes.get(col, 'N/A')) # Get string representation
                 target_dtype_str = str(target_dtype)
                 
                 # Handle known equivalent types explicitly if needed
                 is_match = False
                 if current_dtype_str == target_dtype_str:
                      is_match = True
                 # Add specific checks if direct string comparison isn't enough
                 # e.g., if target is 'string[pyarrow]' and current is 'string' sometimes
                 elif target_dtype_str == 'string[pyarrow]' and current_dtype_str == 'string':
                      # Depending on dask/pyarrow versions, 'string' might be pyarrow backed already
                      # Let's assume we still want to cast to be explicit
                      # is_match = True # Or force cast below
                      pass # Force cast for now
                 elif target_dtype_str == 'boolean' and current_dtype_str == 'bool':
                      is_match = True # Dask often reads bool as numpy bool
                      
                 if col in current_dtypes and not is_match:
                      types_to_cast[col] = target_dtype
                 elif col not in current_dtypes:
                      # This case should be handled by 'cols_to_add' above, but log error if it happens
                      print(f"ERROR: Column '{col}' still missing after attempted add for {data_type_name}.")

             if types_to_cast:
                  print(f"Casting {data_type_name} columns to match meta: {types_to_cast}")
                  # Apply casts one by one for better error isolation? Or all at once?
                  try:
                     ddf = ddf.astype(types_to_cast)
                  except Exception as cast_err:
                     print(f"ERROR during bulk casting for {data_type_name}: {cast_err}")
                     print("Attempting column-by-column casting...")
                     for col_to_cast, type_to_cast in types_to_cast.items():
                         try:
                             ddf[col_to_cast] = ddf[col_to_cast].astype(type_to_cast)
                             print(f"  Successfully cast '{col_to_cast}' to {type_to_cast}")
                         except Exception as single_cast_err:
                             print(f"  ERROR casting column '{col_to_cast}' to {type_to_cast}: {single_cast_err}")
                             print(f"  Column '{col_to_cast}' dtype before error: {ddf[col_to_cast].dtype}")
                             # Option: Fallback to object? Or raise? Let's raise for now.
                             pytest.fail(f"Failed single column cast for '{col_to_cast}' in {data_type_name}")

        except Exception as e:
             print(f"ERROR: Failed during dtype comparison/casting preparation for {data_type_name}: {e}")
             print(f"Dask dtypes: {ddf.dtypes}")
             print(f"Meta dtypes dict: {meta_dtypes_dict}")
             pytest.fail(f"Dtype casting setup failed for {data_type_name}: {e}")

        # Final check on columns vs meta after all operations
        final_columns = list(ddf.columns)
        meta_columns = list(meta_df.columns)
        if final_columns != meta_columns:
            mismatch_msg = (f"Loaded {data_type_name} columns {final_columns} "
                            f"do not match meta {meta_columns} after processing.")
            print(f"ERROR: {mismatch_msg}")
            print(f"ERROR: Dask dtypes: {ddf.dtypes}")
            print(f"ERROR: Meta dtypes: {meta_df.dtypes}")
            # Try to show difference
            print(f"Columns in DDF but not meta: {set(final_columns) - set(meta_columns)}")
            print(f"Columns in meta but not DDF: {set(meta_columns) - set(final_columns)}")
            assert False, mismatch_msg

        print(f"Successfully created Dask DataFrame for {data_type_name}. Final Columns: {list(ddf.columns)}, Final Dtypes: {ddf.dtypes.to_dict()}")
        return ddf
        
    except FileNotFoundError as e:
        print(f"ERROR: {str(e)}") 
        pytest.fail(str(e))
    except Exception as e:
        print(f"ERROR: Failed to load/cache {data_type_name}: {e}") 
        traceback.print_exc()
        pytest.fail(f"Failed to load/cache {data_type_name}: {e}")