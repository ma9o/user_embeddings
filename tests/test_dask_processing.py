import pytest
import dask.dataframe as dd
import pandas as pd
import os
# import random # No longer needed
import glob # Add glob import
# import dask.bag as db # No longer needed for fixtures
# import logging # Import logging -> No longer needed
import pyarrow as pa # Import pyarrow
# No longer need pq here if helper handles writing
# import pyarrow.parquet as pq # Import pyarrow.parquet

# Configure logging -> No longer needed
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the utils directory is in the Python path for import
# This might be handled by pytest configuration or environment variables in a real setup
import sys
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, workspace_root)

from utils.dask_processing import generate_user_context
# No longer need zst_io imports here if helper handles reading/chunking
# from utils.zst_io import read_single_zst_ndjson_chunked
# from utils.zst_io import DEFAULT_CHUNK_SIZE 

# Import the helper function
from tests.test_helpers import _load_or_create_cached_ddf
from dask.diagnostics import ProgressBar

pbar = ProgressBar()
pbar.register()

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
         print(f"Comments directory not found: {COMMENTS_PATH}") # Replaced logging.error
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

    # Call the helper function from the other file
    return _load_or_create_cached_ddf(
        data_path=COMMENTS_PATH,
        file_pattern="RC_*.zst",
        cache_dir=CACHE_DIR,
        meta_df=meta_comments,
        pa_schema=pa_schema_comments,
        data_type_name="comments"
    )

@pytest.fixture(scope="module")
def submissions_ddf():
    """Loads the required columns of the submissions Dask DataFrame from ONE .zst file, using a cache.""" # Updated docstring
    if not os.path.exists(SUBMISSIONS_PATH):
         print(f"Submissions directory not found: {SUBMISSIONS_PATH}") # Replaced logging.error
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

    # Call the helper function from the other file
    return _load_or_create_cached_ddf(
        data_path=SUBMISSIONS_PATH,
        file_pattern="RS_*.zst",
        cache_dir=CACHE_DIR,
        meta_df=meta_submissions,
        pa_schema=pa_schema_submissions,
        data_type_name="submissions"
    )

# --- Test Function ---
def test_generate_context_sample_user(
    comments_ddf: dd.DataFrame, 
    submissions_ddf: dd.DataFrame,
    #caplog: pytest.LogCaptureFixture -> No longer needed
):
    """Tests the generate_user_context function with sample data."""
    # caplog.set_level(logging.INFO) -> No longer needed
    
    current_test_user = TEST_USER
    if current_test_user is None:
        print("TEST_USER is None, sampling comments to find a valid user...") # Replaced logging.info
        selected = False
        sample_size = 100 # How many comments to check
        print(f"Attempting to sample from the head of the last partition (sample size: {sample_size})...") # Replaced logging.info
        try:
            # Take a sample from the head of the *last partition* for efficiency
            # This avoids the slowness of .tail() while still sampling near the end.
            # Since we now load a Pandas DF first, we sample directly from it
            # Check if the underlying dataframe is pandas (it should be now)
            if isinstance(comments_ddf._meta, pd.DataFrame):
                # Sample directly from the Dask DataFrame head without full compute
                comments_sample = comments_ddf.head(sample_size, compute=True) # Sample from Dask head directly
            else:
                # Fallback for safety, though unlikely with current fixture logic
                print("WARNING: Sampling from Dask partition head as underlying data is not Pandas.") # Replaced logging.warning
                comments_sample = comments_ddf.partitions[0].head(sample_size, compute=True) # Use first partition head as it's just 1 partition
                
            if not comments_sample.empty:
                # Iterate through the sample to find a non-deleted user
                for author in comments_sample['author'].dropna():
                    if author and author != '[deleted]' and not author.endswith('Bot') and not author.endswith('bot'): 
                        # Add more bot checks here if needed, e.g., not author.endswith('Bot')
                        current_test_user = author
                        selected = True
                        print(f"Selected user from sample: {current_test_user}") # Replaced logging.info
                        break # Stop after finding the first valid user
                
            if not selected:
                 print(f"WARNING: Could not find a suitable user in the sample (size {sample_size}). Skipping test.") # Replaced logging.warning
                 pytest.skip(f"Could not find a non-deleted user in the sample of size {sample_size}.")
                 
        except KeyError as e:
             print(f"ERROR: Failed to find column '{e}' in comments sample.") # Replaced logging.error
             pytest.fail(f"Failed to find column '{e}' in comments sample.")
        except Exception as e:
            print(f"ERROR: Failed to sample user from comments: {e}") # Replaced logging.error
            pytest.fail(f"Failed to sample user from comments: {e}")
            
    print(f"Testing context generation for user: {current_test_user}") # Replaced logging.info
    
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
        print("Computing the result Dask DataFrame...") # Replaced logging.info
        result_pdf = context_ddf.compute()
        print(f"Computation finished. Result shape: {result_pdf.shape}") # Replaced logging.info
        
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
        print(f"Saving test result artifact to: {output_path}") # Replaced logging.info
        result_pdf.to_csv(output_path, index=False)
        print("Artifact saved successfully.") # Replaced logging.info
    except Exception as e:
        print(f"WARNING: Failed to save test artifact to {output_path}: {e}") # Replaced logging.warning
    
    # Check if the DataFrame is empty or not. 
    # It *could* be empty if the selected user has no comments in the data, which might be valid.
    print(f"Result DataFrame head:\n{result_pdf.head()}") # Replaced logging.info
    
    # Check if expected columns exist in the result
    expected_cols = ['submission_id', 'formatted_context', 'user_comment_ids']
    missing_result_cols = [col for col in expected_cols if col not in result_pdf.columns]
    if missing_result_cols:
        err_msg = f"Result DataFrame missing expected columns: {missing_result_cols}"
        print(f"ERROR: {err_msg}") # Replaced logging.error
        assert False, err_msg

    print(f"Test for user {current_test_user} completed successfully.") # Replaced logging.info

# To run this test:
# 1. Make sure you have pytest installed (`pip install pytest dask distributed pandas pyyaml`)
# 2. Ensure your data is in `data/reddit/comments` and `data/reddit/submissions` (as parquet files)
# 3. Replace TEST_USER with a valid username from your data.
# 4. Navigate to your project root directory in the terminal.
# 5. Run `pytest -s` (the -s flag shows print statements). 