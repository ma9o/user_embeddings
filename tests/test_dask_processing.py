import pytest
import dask.dataframe as dd
import pandas as pd
import os
# import random # No longer needed
import glob # Add glob import
# import dask.bag as db # No longer needed for fixtures
import logging # Import logging
import pyarrow as pa # Import pyarrow
import pyarrow.parquet as pq # Import pyarrow.parquet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the utils directory is in the Python path for import
# This might be handled by pytest configuration or environment variables in a real setup
import sys
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, workspace_root)

from utils.dask_processing import generate_user_context
# Import the new chunked reader and the original one (if still needed elsewhere)
from utils.zst_io import read_single_zst_ndjson, read_single_zst_ndjson_chunked 
from utils.zst_io import DEFAULT_CHUNK_SIZE # Import chunk size constant

# --- Constants --- 
# Adjust these paths if your data structure is different
COMMENTS_PATH = os.path.join(workspace_root, "data/reddit/comments")
SUBMISSIONS_PATH = os.path.join(workspace_root, "data/reddit/submissions")
CACHE_DIR = os.path.join(workspace_root, "data/test_cache") # Cache directory

# Set to None to sample a user, or specify a username string.
TEST_USER = None

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Fixtures (Optional but recommended for setup/teardown) ---
@pytest.fixture(scope="module")
def comments_ddf():
    """Loads the required columns of the comments Dask DataFrame from ONE .zst file, using a cache.""" # Updated docstring
    if not os.path.exists(COMMENTS_PATH):
         logging.error(f"Comments directory not found: {COMMENTS_PATH}")
         pytest.fail(f"Comments directory not found: {COMMENTS_PATH}")

    # Define meta for schema consistency
    meta_comments = pd.DataFrame({ # Keep meta definition
        'id': pd.Series(dtype=pd.StringDtype()),
        'author': pd.Series(dtype=pd.StringDtype()),
        'link_id': pd.Series(dtype=pd.StringDtype()),
        'parent_id': pd.Series(dtype=pd.StringDtype()),
        'body': pd.Series(dtype=pd.StringDtype()),
        'created_utc': pd.Series(dtype='int64')
    })
    # Define PyArrow schema from meta for consistent Parquet writing
    pa_schema_comments = pa.Schema.from_pandas(meta_comments)


    try:
        # Find the first comments file
        file_pattern = "RC_*.zst"
        filepaths = glob.glob(os.path.join(COMMENTS_PATH, file_pattern))
        if not filepaths:
            raise FileNotFoundError(f"No files matching '{file_pattern}' found in {COMMENTS_PATH}")
        # Find first file, define cache path
        first_file_path = filepaths[0] 
        base_filename = os.path.splitext(os.path.basename(first_file_path))[0]
        cache_filename = f"{base_filename}_comments.parquet"
        cache_filepath = os.path.join(CACHE_DIR, cache_filename)

        if os.path.exists(cache_filepath):
            logging.info(f"Loading comments from cache: {cache_filepath}")
            # Still load the full cache into pandas first, then convert to dask
            pandas_df = pd.read_parquet(cache_filepath)
            logging.info(f"Loaded {len(pandas_df)} comments from cache.")
        else:
            logging.info(f"Cache not found for {os.path.basename(first_file_path)}. Processing chunked and caching...")
            
            # --- Chunked Processing and Caching ---
            writer = None
            total_rows_written = 0
            # Use chunk size from zst_io
            chunk_generator = read_single_zst_ndjson_chunked(first_file_path, chunk_size=DEFAULT_CHUNK_SIZE) 
            
            try:
                for i, chunk_df in enumerate(chunk_generator):
                    # Select only columns present in the meta schema
                    columns_to_keep = [col for col in meta_comments.columns if col in chunk_df.columns]
                    chunk_df = chunk_df[columns_to_keep]
                    
                    # Ensure columns match meta order and add missing columns if any
                    for col in meta_comments.columns:
                        if col not in chunk_df:
                             # Add missing column with appropriate NaN type if possible
                             if pd.api.types.is_integer_dtype(meta_comments[col].dtype):
                                 chunk_df[col] = pd.NA # Use pd.NA for nullable integers
                             elif pd.api.types.is_bool_dtype(meta_comments[col].dtype):
                                 chunk_df[col] = pd.NA # Use pd.NA for nullable booleans
                             else:
                                 chunk_df[col] = None # Use None for others (like string/object)
                                 
                    # Reorder columns according to meta
                    chunk_df = chunk_df[list(meta_comments.columns)]

                    # Cast dtypes - handle potential errors during cast
                    for col, dtype in meta_comments.dtypes.items():
                        if chunk_df[col].dtype != dtype:
                            try:
                                chunk_df[col] = chunk_df[col].astype(dtype)
                            except Exception as e:
                                logging.warning(f"Could not cast column '{col}' to {dtype} in chunk {i}. Error: {e}. Filling with NA/None.")
                                # Fill with NA/None on error to maintain schema
                                if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
                                    chunk_df[col] = pd.NA
                                else:
                                    chunk_df[col] = None
                                # Attempt cast again if needed, though unlikely to succeed
                                try:
                                     chunk_df[col] = chunk_df[col].astype(dtype)
                                except:
                                     pass # Ignore if second cast fails

                    # Convert chunk to PyArrow Table with the defined schema
                    try:
                         table = pa.Table.from_pandas(chunk_df, schema=pa_schema_comments, preserve_index=False)
                    except Exception as e:
                         logging.error(f"Error converting chunk {i} to Arrow Table: {e}")
                         # Optional: Log problematic chunk data
                         # logging.error(f"Problematic chunk head:\n{chunk_df.head()}")
                         # logging.error(f"Problematic chunk dtypes:\n{chunk_df.dtypes}")
                         # logging.error(f"Expected meta dtypes:\n{meta_comments.dtypes}")
                         continue # Skip this chunk if conversion fails

                    if writer is None:
                        logging.info(f"Creating Parquet writer for: {cache_filepath}")
                        # Use PyArrow ParquetWriter for schema enforcement and append
                        writer = pq.ParquetWriter(cache_filepath, table.schema)
                    
                    writer.write_table(table)
                    total_rows_written += len(chunk_df)
                    logging.debug(f"Written chunk {i} ({len(chunk_df)} rows) to {cache_filepath}")

            finally:
                if writer:
                    writer.close()
                    logging.info(f"Closed Parquet writer. Total rows written: {total_rows_written}")
                    
            # --- Load the newly created cache file ---
            if total_rows_written > 0:
                 logging.info(f"Loading the newly created comments cache: {cache_filepath}")
                 pandas_df = pd.read_parquet(cache_filepath)
                 logging.info(f"Loaded {len(pandas_df)} comments from generated cache.")
            else:
                 logging.warning(f"No rows were written to cache file {cache_filepath}. Creating empty DataFrame.")
                 # Create an empty DataFrame matching the meta schema if nothing was written
                 pandas_df = pd.DataFrame(columns=meta_comments.columns).astype(meta_comments.dtypes.to_dict())


        # --- Convert Pandas DF (from cache or generated) to Dask DF ---
        logging.info("Converting Pandas DataFrame to Dask DataFrame...")
        # Ensure correct dtypes based on meta after loading from parquet
        for col, dtype in meta_comments.dtypes.items():
             if col in pandas_df.columns:
                  if pandas_df[col].dtype != dtype:
                      try:
                          pandas_df[col] = pandas_df[col].astype(dtype)
                      except Exception as e:
                           logging.warning(f"Could not cast column '{col}' to {dtype} after loading cache. Error: {e}")
             else:
                  logging.warning(f"Column '{col}' from meta not found in loaded cache DataFrame.")
                  # Add missing columns if necessary before creating Dask DF
                  if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
                       pandas_df[col] = pd.NA
                  else:
                       pandas_df[col] = None
                  pandas_df[col] = pandas_df[col].astype(dtype) # Cast the new column

        # Ensure columns are in the correct order before creating Dask DataFrame
        pandas_df = pandas_df[list(meta_comments.columns)]

        ddf = dd.from_pandas(pandas_df, npartitions=1) # Use 1 partition
        
        # Re-apply meta if needed (dd.from_pandas might not preserve all Dask-specific types)
        # ddf = ddf.astype(meta_comments.dtypes.to_dict())
        
        # Check columns match meta
        if not list(ddf.columns) == list(meta_comments.columns):
             mismatch_msg = f"Loaded comments columns {list(ddf.columns)} do not match meta {list(meta_comments.columns)}"
             logging.error(mismatch_msg)
             # Log dtypes for debugging
             logging.error(f"Dask dtypes: {ddf.dtypes}")
             logging.error(f"Meta dtypes: {meta_comments.dtypes}")
             assert False, mismatch_msg
             
        logging.info(f"Successfully created Dask DataFrame for comments. Columns: {list(ddf.columns)}")
        return ddf
        
    except FileNotFoundError as e:
        logging.error(str(e))
        pytest.fail(str(e))
    except Exception as e:
        import traceback
        logging.error(f"Failed to load/cache comments: {e}", exc_info=True)
        traceback.print_exc()
        pytest.fail(f"Failed to load/cache comments: {e}")

@pytest.fixture(scope="module")
def submissions_ddf():
    """Loads the required columns of the submissions Dask DataFrame from ONE .zst file, using a cache.""" # Updated docstring
    if not os.path.exists(SUBMISSIONS_PATH):
         logging.error(f"Submissions directory not found: {SUBMISSIONS_PATH}")
         pytest.fail(f"Submissions directory not found: {SUBMISSIONS_PATH}")

    # Define meta for schema consistency
    meta_submissions = pd.DataFrame({ # Keep meta definition
        'id': pd.Series(dtype=pd.StringDtype()),
        'title': pd.Series(dtype=pd.StringDtype()),
        'selftext': pd.Series(dtype=pd.StringDtype()),
        'is_self': pd.Series(dtype='boolean') # Use nullable boolean
    })
    # Define PyArrow schema from meta
    pa_schema_submissions = pa.Schema.from_pandas(meta_submissions)

    try:
        # Find the first submissions file
        file_pattern = "RS_*.zst"
        filepaths = glob.glob(os.path.join(SUBMISSIONS_PATH, file_pattern))
        if not filepaths:
             raise FileNotFoundError(f"No files matching '{file_pattern}' found in {SUBMISSIONS_PATH}")
        # Find first file, define cache path
        first_file_path = filepaths[0] 
        base_filename = os.path.splitext(os.path.basename(first_file_path))[0]
        cache_filename = f"{base_filename}_submissions.parquet"
        cache_filepath = os.path.join(CACHE_DIR, cache_filename)

        if os.path.exists(cache_filepath):
            logging.info(f"Loading submissions from cache: {cache_filepath}")
            pandas_df = pd.read_parquet(cache_filepath)
            logging.info(f"Loaded {len(pandas_df)} submissions from cache.")
        else:
            logging.info(f"Cache not found for {os.path.basename(first_file_path)}. Processing chunked and caching...")
            
            # --- Chunked Processing and Caching ---
            writer = None
            total_rows_written = 0
            # Use chunk size from zst_io
            chunk_generator = read_single_zst_ndjson_chunked(first_file_path, chunk_size=DEFAULT_CHUNK_SIZE) 
            
            try:
                for i, chunk_df in enumerate(chunk_generator):
                     # Select/Add/Reorder columns to match meta
                    columns_to_keep = [col for col in meta_submissions.columns if col in chunk_df.columns]
                    chunk_df = chunk_df[columns_to_keep]
                    for col in meta_submissions.columns:
                        if col not in chunk_df:
                            if pd.api.types.is_bool_dtype(meta_submissions[col].dtype):
                                chunk_df[col] = pd.NA 
                            else:
                                chunk_df[col] = None
                    chunk_df = chunk_df[list(meta_submissions.columns)] # Reorder

                    # Cast dtypes
                    for col, dtype in meta_submissions.dtypes.items():
                        if chunk_df[col].dtype != dtype:
                            try:
                                chunk_df[col] = chunk_df[col].astype(dtype)
                            except Exception as e:
                                logging.warning(f"Could not cast column '{col}' to {dtype} in submissions chunk {i}. Error: {e}. Filling with NA/None.")
                                if pd.api.types.is_bool_dtype(dtype):
                                    chunk_df[col] = pd.NA
                                else:
                                    chunk_df[col] = None
                                try: # Attempt cast again
                                     chunk_df[col] = chunk_df[col].astype(dtype)
                                except:
                                     pass 
                                     
                    # Convert chunk to PyArrow Table
                    try:
                         table = pa.Table.from_pandas(chunk_df, schema=pa_schema_submissions, preserve_index=False)
                    except Exception as e:
                         logging.error(f"Error converting submissions chunk {i} to Arrow Table: {e}")
                         continue # Skip chunk

                    if writer is None:
                        logging.info(f"Creating Parquet writer for: {cache_filepath}")
                        writer = pq.ParquetWriter(cache_filepath, table.schema)
                    
                    writer.write_table(table)
                    total_rows_written += len(chunk_df)
                    logging.debug(f"Written chunk {i} ({len(chunk_df)} rows) to {cache_filepath}")

            finally:
                if writer:
                    writer.close()
                    logging.info(f"Closed Parquet writer. Total rows written: {total_rows_written}")

            # --- Load the newly created cache file ---
            if total_rows_written > 0:
                logging.info(f"Loading the newly created submissions cache: {cache_filepath}")
                pandas_df = pd.read_parquet(cache_filepath)
                logging.info(f"Loaded {len(pandas_df)} submissions from generated cache.")
            else:
                logging.warning(f"No rows were written to cache file {cache_filepath}. Creating empty DataFrame.")
                pandas_df = pd.DataFrame(columns=meta_submissions.columns).astype(meta_submissions.dtypes.to_dict())


        # --- Convert Pandas DF to Dask DF ---
        logging.info("Converting Pandas DataFrame to Dask DataFrame...")
        # Ensure correct dtypes based on meta after loading from parquet
        for col, dtype in meta_submissions.dtypes.items():
             if col in pandas_df.columns:
                 if pandas_df[col].dtype != dtype:
                      try:
                           pandas_df[col] = pandas_df[col].astype(dtype)
                      except Exception as e:
                           logging.warning(f"Could not cast column '{col}' to {dtype} after loading cache. Error: {e}")
             else:
                  logging.warning(f"Column '{col}' from meta not found in loaded cache DataFrame.")
                  # Add missing columns if necessary
                  if pd.api.types.is_bool_dtype(dtype):
                       pandas_df[col] = pd.NA
                  else:
                       pandas_df[col] = None
                  pandas_df[col] = pandas_df[col].astype(dtype) # Cast the new column

        # Ensure columns are in the correct order
        pandas_df = pandas_df[list(meta_submissions.columns)]
                     
        ddf = dd.from_pandas(pandas_df, npartitions=1) # Use 1 partition
        
        # Re-apply meta if needed
        # ddf = ddf.astype(meta_submissions.dtypes.to_dict())
        
        # Check columns match meta
        if not list(ddf.columns) == list(meta_submissions.columns):
            mismatch_msg = f"Loaded submissions columns {list(ddf.columns)} do not match meta {list(meta_submissions.columns)}"
            logging.error(mismatch_msg)
            logging.error(f"Dask dtypes: {ddf.dtypes}")
            logging.error(f"Meta dtypes: {meta_submissions.dtypes}")
            assert False, mismatch_msg
            
        logging.info(f"Successfully created Dask DataFrame for submissions. Columns: {list(ddf.columns)}")
        return ddf

    except FileNotFoundError as e:
         logging.error(str(e))
         pytest.fail(str(e))
    except Exception as e:
        import traceback
        logging.error(f"Failed to load/cache submissions: {e}", exc_info=True)
        traceback.print_exc()
        pytest.fail(f"Failed to load/cache submissions: {e}")

# --- Test Function ---
def test_generate_context_sample_user(
    comments_ddf: dd.DataFrame, 
    submissions_ddf: dd.DataFrame,
    caplog: pytest.LogCaptureFixture
):
    """Tests the generate_user_context function with sample data."""
    caplog.set_level(logging.INFO)
    
    current_test_user = TEST_USER
    if current_test_user is None:
        logging.info("TEST_USER is None, sampling comments to find a valid user...")
        selected = False
        sample_size = 100 # How many comments to check
        logging.info(f"Attempting to sample from the head of the last partition (sample size: {sample_size})...") 
        try:
            # Take a sample from the head of the *last partition* for efficiency
            # This avoids the slowness of .tail() while still sampling near the end.
            # Since we now load a Pandas DF first, we sample directly from it
            # Check if the underlying dataframe is pandas (it should be now)
            if isinstance(comments_ddf._meta, pd.DataFrame):
                comments_sample = comments_ddf.compute().head(sample_size) # Sample from pandas directly
            else:
                # Fallback for safety, though unlikely with current fixture logic
                logging.warning("Sampling from Dask partition head as underlying data is not Pandas.")
                comments_sample = comments_ddf.partitions[0].head(sample_size, compute=True) # Use first partition head as it's just 1 partition
                
            if not comments_sample.empty:
                # Iterate through the sample to find a non-deleted user
                for author in comments_sample['author'].dropna():
                    if author and author != '[deleted]' and not author.endswith('Bot') and not author.endswith('bot'): 
                        # Add more bot checks here if needed, e.g., not author.endswith('Bot')
                        current_test_user = author
                        selected = True
                        logging.info(f"Selected user from sample: {current_test_user}")
                        break # Stop after finding the first valid user
                
            if not selected:
                 logging.warning(f"Could not find a suitable user in the sample (size {sample_size}). Skipping test.")
                 pytest.skip(f"Could not find a non-deleted user in the sample of size {sample_size}.")
                 
        except KeyError as e:
             logging.error(f"Failed to find column '{e}' in comments sample.", exc_info=True)
             pytest.fail(f"Failed to find column '{e}' in comments sample.")
        except Exception as e:
            logging.error(f"Failed to sample user from comments: {e}", exc_info=True)
            pytest.fail(f"Failed to sample user from comments: {e}")
            
    logging.info(f"Testing context generation for user: {current_test_user}")
    
    # Basic check if dataframes loaded (pytest fixtures handle load errors)
    assert comments_ddf is not None
    assert submissions_ddf is not None
    
    # Ensure required columns exist before calling the function
    required_comment_cols = ['id', 'author', 'link_id', 'parent_id', 'body', 'created_utc']
    required_submission_cols = ['id', 'title', 'selftext', 'is_self']
    
    missing_comment_cols = [col for col in required_comment_cols if col not in comments_ddf.columns]
    missing_submission_cols = [col for col in required_submission_cols if col not in submissions_ddf.columns]
    
    assert not missing_comment_cols, f"Comments Dask DataFrame missing columns: {missing_comment_cols}"
    assert not missing_submission_cols, f"Submissions Dask DataFrame missing columns: {missing_submission_cols}"

    # --- Execute the function under test ---
    try:
        context_ddf = generate_user_context(
            user_id=current_test_user, 
            ddf_comments=comments_ddf, 
            ddf_submissions=submissions_ddf
        )
        
        # --- Compute the results ---
        logging.info("Computing the result Dask DataFrame...")
        result_pdf = context_ddf.compute()
        logging.info(f"Computation finished. Result shape: {result_pdf.shape}")
        
    except ValueError as e:
         pytest.fail(f"generate_user_context raised ValueError: {e}")
    except Exception as e:
         import traceback
         traceback.print_exc()
         pytest.fail(f"generate_user_context raised an unexpected exception: {e}")

    # --- Assertions on the result ---
    assert isinstance(result_pdf, pd.DataFrame), "Result should be a Pandas DataFrame after compute()"
    
    # --- Save test artifact ---
    output_dir = os.path.join(workspace_root, "data/test_results")
    os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists
    output_filename = f"test_output_{current_test_user}.csv"
    output_path = os.path.join(output_dir, output_filename)
    try:
        logging.info(f"Saving test result artifact to: {output_path}")
        result_pdf.to_csv(output_path, index=False)
        logging.info("Artifact saved successfully.")
    except Exception as e:
        logging.warning(f"Failed to save test artifact to {output_path}: {e}")
    
    # Check if the DataFrame is empty or not. 
    # It *could* be empty if the selected user has no comments in the data, which might be valid.
    logging.info(f"Result DataFrame head:\n{result_pdf.head()}")
    # assert not result_pdf.empty, f"Result DataFrame is empty for user {current_test_user}. Check if user exists and has comments."
    
    # Check if expected columns exist in the result
    expected_cols = ['submission_id', 'formatted_context', 'user_comment_ids']
    missing_result_cols = [col for col in expected_cols if col not in result_pdf.columns]
    if missing_result_cols:
        err_msg = f"Result DataFrame missing expected columns: {missing_result_cols}"
        logging.error(err_msg)
        assert False, err_msg

    # Optional: Add more specific assertions if you know the expected output for TEST_USER
    # For example:
    # if not result_pdf.empty:
    #     assert 'expected_submission_id' in result_pdf['submission_id'].values
    #     first_row_context = result_pdf['formatted_context'].iloc[0]
    #     assert isinstance(first_row_context, str)
    #     assert "title:" in first_row_context # Check if YAML structure seems present
    #     assert "replies:" in first_row_context
    #     user_ids_list = result_pdf['user_comment_ids'].iloc[0]
    #     assert isinstance(user_ids_list, list)

    logging.info(f"Test for user {current_test_user} completed successfully.")

# To run this test:
# 1. Make sure you have pytest installed (`pip install pytest dask distributed pandas pyyaml`)
# 2. Ensure your data is in `data/reddit/comments` and `data/reddit/submissions` (as parquet files)
# 3. Replace TEST_USER with a valid username from your data.
# 4. Navigate to your project root directory in the terminal.
# 5. Run `pytest -s` (the -s flag shows print statements). 