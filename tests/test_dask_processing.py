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
import time # Add time import
from dask.distributed import Client, LocalCluster, progress
from typing import List # Add List import

# Configure logging -> No longer needed
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the utils directory is in the Python path for import
# This might be handled by pytest configuration or environment variables in a real setup
import sys
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if workspace_root not in sys.path: # Ensure root is in path for src import
    sys.path.insert(0, workspace_root)

from user_embeddings.utils.dask_processing import generate_user_context
# No longer need zst_io imports here if helper handles reading/chunking
# from utils.zst_io import read_single_zst_ndjson_chunked
# from utils.zst_io import DEFAULT_CHUNK_SIZE

# Import the helper function from its new location
from .helpers.data_loading import load_or_create_cached_ddf

# --- Constants ---
# Adjust these paths if your data structure is different
COMMENTS_PATH = os.path.join(workspace_root, "data/reddit/comments")
SUBMISSIONS_PATH = os.path.join(workspace_root, "data/reddit/submissions")
CACHE_DIR = os.path.join(workspace_root, "data/test_cache") # Cache directory

# Set to None to sample users, or a list of specific usernames.
TEST_USER = None
NUM_TEST_USERS = 10 # Define how many users to sample and test

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Dask Cluster Setup ---
# Setup local cluster before fixtures might use it implicitly
print("Setting up local Dask cluster...")
# Use slightly fewer workers than cores initially, distribute memory
cluster = LocalCluster(n_workers=5, threads_per_worker=2, memory_limit='20GB')
client = Client(cluster)
print(f"Dask dashboard link: {client.dashboard_link}")

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
    return load_or_create_cached_ddf(
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
        'author': pd.Series(dtype=pd.StringDtype()),
        'created_utc': pd.Series(dtype='int64'),
        'subreddit': pd.Series(dtype=pd.StringDtype()),
        'title': pd.Series(dtype=pd.StringDtype()),
        'selftext': pd.Series(dtype=pd.StringDtype()),
        'is_self': pd.Series(dtype='boolean') # Use nullable boolean
    })
    # Define PyArrow schema from meta
    pa_schema_submissions = pa.Schema.from_pandas(meta_submissions)

    # Call the helper function from the other file
    return load_or_create_cached_ddf(
        data_path=SUBMISSIONS_PATH,
        file_pattern="RS_*.zst",
        cache_dir=CACHE_DIR,
        meta_df=meta_submissions,
        pa_schema=pa_schema_submissions,
        data_type_name="submissions"
    )

def _sample_test_users(comments_ddf: dd.DataFrame, num_users_to_find: int) -> List[str]:
    """Samples the comments DataFrame to find a list of valid user authors."""
    print(f"Sampling comments to find {num_users_to_find} valid users...")
    sampled_users = []
    # Increase sample size significantly to find multiple users
    sample_size = 1000 # How many comments to check
    print(f"Attempting to sample from the head (sample size: {sample_size})...")
    try:
        # Sample directly from the Dask DataFrame head
        # Compute is needed here to get actual author names
        comments_sample = comments_ddf.head(sample_size, compute=True)

        if not comments_sample.empty:
            sampled_authors = comments_sample['author'].dropna().unique()
            found_users = 0
            for author in sampled_authors:
                # Filter out deleted users, bots, etc.
                if author and author != '[deleted]' and not author.endswith('Bot') and not author.endswith('bot'):
                    sampled_users.append(author)
                    found_users += 1
                    if found_users >= num_users_to_find:
                        break # Stop once we have enough users

            print(f"Found {len(sampled_users)} valid users from sample: {sampled_users}")
            if len(sampled_users) < num_users_to_find:
                 print(f"WARNING: Found only {len(sampled_users)}/{num_users_to_find} distinct valid users in the sample of size {sample_size}.")

        if not sampled_users:
             print(f"ERROR: Could not find any suitable users in the sample (size {sample_size}).")
             # Let the caller decide whether to skip or fail

    except KeyError as e:
         print(f"ERROR: Failed to find column '{e}' in comments sample.")
         raise # Re-raise the error to be caught by the test
    except Exception as e:
        print(f"ERROR: Failed to sample users from comments: {e}")
        import traceback
        traceback.print_exc()
        raise # Re-raise the error
        
    return sampled_users

def _run_and_validate_user_test(
    user_id: str, 
    comments_ddf: dd.DataFrame, 
    submissions_ddf: dd.DataFrame, 
    output_dir: str
) -> bool:
    """Runs the context generation for a single user, validates, and saves artifact.
    Returns True if successful, False otherwise.
    """
    print(f"\n--- Processing User: {user_id} --- --- --- --- --- --- --- --- --- --- --- --- ---")
    success = True
    try:
        # --- Execute the function under test ---
        context_ddf = generate_user_context(
            user_id=user_id,
            ddf_comments=comments_ddf,
            ddf_submissions=submissions_ddf
        )

        # --- Compute the results ---
        print(f"Computing the result Dask DataFrame for user {user_id}...")
        start_time = time.time() # Start timing
        # Persist before compute can sometimes help with complex graphs
        future = context_ddf.persist()
        progress(future) # Display progress bar
        result_pdf = future.compute()
        end_time = time.time() # End timing
        print(f"Computation finished for user {user_id}. Result shape: {result_pdf.shape}. Time: {end_time - start_time:.2f}s")

        # --- Assertions on the result ---
        assert isinstance(result_pdf, pd.DataFrame), f"Result for user {user_id} should be a Pandas DataFrame after compute()"

        # Check if expected columns exist in the result
        expected_cols = ['submission_id', 'formatted_context', 'user_comment_ids']
        missing_result_cols = [col for col in expected_cols if col not in result_pdf.columns]
        if missing_result_cols:
            err_msg = f"Result DataFrame for user {user_id} missing expected columns: {missing_result_cols}"
            print(f"ERROR: {err_msg}")
            assert False, err_msg # Fail the specific user test
            
        print(f"Result DataFrame head for user {user_id}:\n{result_pdf.head()}")

        # --- Save test artifact ---
        os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists
        # Sanitize username for filename
        safe_username = "".join(c if c.isalnum() else "_" for c in user_id)
        output_filename = f"test_output_{safe_username}.csv" 
        output_path = os.path.join(output_dir, output_filename)
        try:
            print(f"Saving test result artifact for {user_id} to: {output_path}")
            result_pdf.to_csv(output_path, index=False)
            print("Artifact saved successfully.")
        except Exception as e:
            print(f"WARNING: Failed to save test artifact for {user_id} to {output_path}: {e}")
            success = False # Mark test as partially failed if save fails

        print(f"--- Test for user {user_id} completed successfully. ---")

    except ValueError as e:
         print(f"ERROR: generate_user_context raised ValueError for user {user_id}: {e}")
         success = False 
    except AssertionError as e: # Catch assertion errors explicitly
         print(f"ERROR: Assertion failed for user {user_id}: {e}")
         success = False
    except Exception as e:
         print(f"ERROR: generate_user_context raised an unexpected exception for user {user_id}: {e}")
         import traceback
         traceback.print_exc()
         success = False
         
    return success

# --- Test Function ---
def test_generate_context_sample_user(
    comments_ddf: dd.DataFrame,
    submissions_ddf: dd.DataFrame,
    #caplog: pytest.LogCaptureFixture -> No longer needed
):
    """
    Tests the generate_user_context function for multiple sampled users.
    """
    # caplog.set_level(logging.INFO) -> No longer needed

    test_users_to_process = []
    if TEST_USER is None:
        try:
            test_users_to_process = _sample_test_users(comments_ddf, NUM_TEST_USERS)
            if not test_users_to_process:
                pytest.skip(f"Could not find any non-deleted users in the sample. Skipping test.")
        except Exception as e:
             pytest.fail(f"Failed during user sampling: {e}")

    elif isinstance(TEST_USER, list):
         print(f"Using predefined list of test users: {TEST_USER}")
         test_users_to_process = TEST_USER
    elif isinstance(TEST_USER, str):
         print(f"Using predefined single test user: {TEST_USER}")
         test_users_to_process = [TEST_USER]
    else:
         pytest.fail(f"Invalid TEST_USER type: {type(TEST_USER)}. Should be None, list, or str.")


    # Basic check if dataframes loaded (pytest fixtures handle load errors)
    assert comments_ddf is not None
    assert submissions_ddf is not None

    # Ensure required columns exist before calling the function
    required_comment_cols = ['id', 'author', 'link_id', 'parent_id', 'body', 'created_utc']
    required_submission_cols = ['id', 'subreddit', 'title', 'selftext', 'is_self']

    missing_comment_cols = [col for col in required_comment_cols if col not in comments_ddf.columns]
    missing_submission_cols = [col for col in required_submission_cols if col not in submissions_ddf.columns]

    assert not missing_comment_cols, f"Comments Dask DataFrame missing columns: {missing_comment_cols}"
    assert not missing_submission_cols, f"Submissions Dask DataFrame missing columns: {missing_submission_cols}"

    # --- Loop through users and execute the function ---
    all_tests_passed = True
    test_output_dir = os.path.join(workspace_root, "data/test_results") # Define once

    for user_index, current_test_user in enumerate(test_users_to_process):
        # Call the helper function for the single user test
        user_test_passed = _run_and_validate_user_test(
            user_id=current_test_user,
            comments_ddf=comments_ddf,
            submissions_ddf=submissions_ddf,
            output_dir=test_output_dir
        )
        if not user_test_passed:
            all_tests_passed = False
            # Decide whether to continue testing other users or stop
            # For now, let's continue to see all failures
            # To stop on first failure: break

    # --- Dask Cluster Teardown (after all users) ---
    print("\nShutting down Dask client and cluster...")
    client.close()
    cluster.close()
    print("Dask client and cluster shut down.")

    # Final assertion based on whether all user tests passed
    assert all_tests_passed, "One or more user context generation tests failed."


# To run this test:
# 1. Make sure you have pytest installed (`pip install pytest dask distributed pandas pyyaml pyarrow`)
# 2. Ensure distributed is installed: `pip install distributed`
# 3. Ensure your data is in `data/reddit/comments` and `data/reddit/submissions` (as parquet files generated by the cache mechanism)
# 4. Leave TEST_USER as None to sample NUM_TEST_USERS, or provide a specific list/string.
# 5. Navigate to your project root directory in the terminal.
# 6. Run `pytest -s tests/test_dask_processing.py`. You should see the dashboard link printed and processing for multiple users. 