import pytest
import dask.dataframe as dd
import pandas as pd
import os
# import random # No longer needed
import glob # Add glob import
import dask.bag as db # Add dask.bag import

# Ensure the utils directory is in the Python path for import
# This might be handled by pytest configuration or environment variables in a real setup
import sys
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, workspace_root)

from utils.dask_processing import generate_user_context
# Import the specific reader function and the main one
from utils.zst_io import read_single_zst_ndjson, read_zst_ndjson_files 

# --- Constants --- 
# Adjust these paths if your data structure is different
COMMENTS_PATH = os.path.join(workspace_root, "data/reddit/comments")
SUBMISSIONS_PATH = os.path.join(workspace_root, "data/reddit/submissions")
# Set to None to sample a user, or specify a username string.
TEST_USER = None

# --- Fixtures (Optional but recommended for setup/teardown) ---
@pytest.fixture(scope="module")
def comments_ddf():
    """Loads the required columns of the comments Dask DataFrame from ONE .zst file.""" # Updated docstring
    if not os.path.exists(COMMENTS_PATH):
         pytest.fail(f"Comments directory not found: {COMMENTS_PATH}")

    # Define required columns and their expected dtypes for meta
    # required_cols = ['id', 'author', 'link_id', 'parent_id', 'body', 'created_utc'] # Keep meta definition
    # Use pd.StringDtype() for potentially large string columns
    meta_comments = pd.DataFrame({
        'id': pd.Series(dtype=pd.StringDtype()),
        'author': pd.Series(dtype=pd.StringDtype()),
        'link_id': pd.Series(dtype=pd.StringDtype()),
        'parent_id': pd.Series(dtype=pd.StringDtype()),
        'body': pd.Series(dtype=pd.StringDtype()),
        'created_utc': pd.Series(dtype='int64') # Assuming UTC stored as integer
    })
    # Set index for meta if the output dataframe is expected to have one
    # meta_comments = meta_comments.set_index('id') 

    try:
        # Find files matching the pattern
        file_pattern = "RC_*.zst"
        print(f"Searching for comments files in: {COMMENTS_PATH} (pattern: {file_pattern})")
        filepaths = glob.glob(os.path.join(COMMENTS_PATH, file_pattern))
        
        if not filepaths:
            raise FileNotFoundError(f"No files matching '{file_pattern}' found in directory: {COMMENTS_PATH}")

        # Select only the first file
        first_file_path = filepaths[0]
        print(f"Using only the first found comments file: {os.path.basename(first_file_path)}")

        # Create a Dask Bag from the single file path
        bag = db.from_sequence([first_file_path]).map_partitions(
            lambda paths: [record for path in paths for record in read_single_zst_ndjson(path)]
        )

        # Convert the bag to a Dask DataFrame using the meta
        ddf = bag.to_dataframe(meta=meta_comments)

        # Basic check: Ensure it's a Dask DataFrame
        assert isinstance(ddf, dd.DataFrame), "Conversion from bag did not return a Dask DataFrame"
        # Check columns match meta
        assert list(ddf.columns) == list(meta_comments.columns), \
               f"Loaded comments columns {list(ddf.columns)} do not match meta {list(meta_comments.columns)}"
        print(f"Successfully created Dask DataFrame for comments from single file. Columns: {list(ddf.columns)}")
        return ddf
    except FileNotFoundError as e:
        pytest.fail(str(e)) # Pass the error message from the check above
    except Exception as e:
        import traceback
        traceback.print_exc()
        pytest.fail(f"Failed to load comments using read_single_zst_ndjson from {first_file_path}: {e}")

@pytest.fixture(scope="module")
def submissions_ddf():
    """Loads the required columns of the submissions Dask DataFrame from ONE .zst file.""" # Updated docstring
    if not os.path.exists(SUBMISSIONS_PATH):
         pytest.fail(f"Submissions directory not found: {SUBMISSIONS_PATH}")

    # Define required columns and their expected dtypes for meta
    # required_cols = ['id', 'title', 'selftext', 'is_self'] # Keep meta definition
    meta_submissions = pd.DataFrame({
        'id': pd.Series(dtype=pd.StringDtype()),
        'title': pd.Series(dtype=pd.StringDtype()),
        'selftext': pd.Series(dtype=pd.StringDtype()),
        'is_self': pd.Series(dtype='boolean') # Use nullable boolean
    })
    # meta_submissions = meta_submissions.set_index('id')

    try:
        # Find files matching the pattern
        file_pattern = "RS_*.zst"
        print(f"Searching for submissions files in: {SUBMISSIONS_PATH} (pattern: {file_pattern})")
        filepaths = glob.glob(os.path.join(SUBMISSIONS_PATH, file_pattern))

        if not filepaths:
            raise FileNotFoundError(f"No files matching '{file_pattern}' found in directory: {SUBMISSIONS_PATH}")

        # Select only the first file
        first_file_path = filepaths[0]
        print(f"Using only the first found submissions file: {os.path.basename(first_file_path)}")

        # Create a Dask Bag from the single file path
        bag = db.from_sequence([first_file_path]).map_partitions(
            lambda paths: [record for path in paths for record in read_single_zst_ndjson(path)]
        )

        # Convert the bag to a Dask DataFrame using the meta
        ddf = bag.to_dataframe(meta=meta_submissions)

        assert isinstance(ddf, dd.DataFrame), "Conversion from bag did not return a Dask DataFrame"
        # Check columns match meta
        assert list(ddf.columns) == list(meta_submissions.columns), \
               f"Loaded submissions columns {list(ddf.columns)} do not match meta {list(meta_submissions.columns)}"
        print(f"Successfully created Dask DataFrame for submissions from single file. Columns: {list(ddf.columns)}")
        return ddf
    except FileNotFoundError as e:
         pytest.fail(str(e)) # Pass the error message from the check above
    except Exception as e:
        import traceback
        traceback.print_exc()
        pytest.fail(f"Failed to load submissions using read_single_zst_ndjson from {first_file_path}: {e}")

# --- Test Function ---
def test_generate_context_sample_user(
    comments_ddf: dd.DataFrame, 
    submissions_ddf: dd.DataFrame
):
    """Tests the generate_user_context function with sample data."""
    
    current_test_user = TEST_USER
    if current_test_user is None:
        print("\nTEST_USER is placeholder, sampling comments to find a valid user...")
        selected = False
        sample_size = 100 # How many comments to check
        try:
            # Take a small sample from the beginning (cheaper than full unique() or sample())
            comments_sample = comments_ddf.tail(sample_size, compute=True) 
            if not comments_sample.empty:
                # Iterate through the sample to find a non-deleted user
                for author in comments_sample['author'].dropna():
                    if author and author != '[deleted]' and not author.endswith('Bot') and not author.endswith('bot'): 
                        # Add more bot checks here if needed, e.g., not author.endswith('Bot')
                        current_test_user = author
                        selected = True
                        print(f"Selected user from sample: {current_test_user}")
                        break # Stop after finding the first valid user
                
            if not selected:
                 pytest.skip(f"Could not find a non-deleted user in the last {sample_size} comments.")
                 
        except KeyError:
             pytest.fail("Failed to find 'author' column in comments sample.")
        except Exception as e:
            pytest.fail(f"Failed to sample user from comments: {e}")
            
    print(f"\nTesting context generation for user: {current_test_user}")
    
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
        print("Computing the result...")
        result_pdf = context_ddf.compute()
        print(f"Computation finished. Result shape: {result_pdf.shape}")
        
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
        print(f"Saving test result artifact to: {output_path}")
        result_pdf.to_csv(output_path, index=False)
        print("Artifact saved successfully.")
    except Exception as e:
        print(f"Warning: Failed to save test artifact to {output_path}: {e}")
    
    # Check if the DataFrame is empty or not. 
    # It *could* be empty if the selected user has no comments in the data, which might be valid.
    print(f"Result DataFrame head:\n{result_pdf.head()}")
    # assert not result_pdf.empty, f"Result DataFrame is empty for user {current_test_user}. Check if user exists and has comments."
    
    # Check if expected columns exist in the result
    expected_cols = ['submission_id', 'formatted_context', 'user_comment_ids']
    missing_result_cols = [col for col in expected_cols if col not in result_pdf.columns]
    assert not missing_result_cols, f"Result DataFrame missing expected columns: {missing_result_cols}"

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

    print(f"Test for user {current_test_user} completed successfully.")

# To run this test:
# 1. Make sure you have pytest installed (`pip install pytest dask distributed pandas pyyaml`)
# 2. Ensure your data is in `data/reddit/comments` and `data/reddit/submissions` (as parquet files)
# 3. Replace TEST_USER with a valid username from your data.
# 4. Navigate to your project root directory in the terminal.
# 5. Run `pytest -s` (the -s flag shows print statements). 