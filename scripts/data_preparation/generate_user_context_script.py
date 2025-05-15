import logging
import os
import sys
import time
from typing import List

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa

# import pytest # Removed pytest
from dask.distributed import Client, LocalCluster, progress

from user_embeddings.utils.data_loading.dask_processing import generate_user_context

# Import the helper function from its new location
from .helpers.data_loading import load_or_create_cached_ddf

logger = logging.getLogger(__name__)

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:  # Ensure root is in path for src import
    sys.path.insert(0, workspace_root)


# --- Constants ---
# Adjust these paths if your data structure is different
COMMENTS_PATH = os.path.join(workspace_root, "data/reddit/comments")
SUBMISSIONS_PATH = os.path.join(workspace_root, "data/reddit/submissions")
CACHE_DIR = os.path.join(
    workspace_root, "data/script_cache"
)  # Cache directory for script

# Set to None to sample users, or a list of specific usernames.
TARGET_USER = None  # Renamed from TEST_USER
NUM_TARGET_USERS = 10  # Renamed from NUM_TEST_USERS

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Dask Cluster Setup ---
# Setup local cluster
logger.info("Setting up local Dask cluster...")
# Use slightly fewer workers than cores initially, distribute memory
cluster = LocalCluster(n_workers=5, threads_per_worker=2, memory_limit="20GB")
client = Client(cluster)
logger.info(f"Dask dashboard link: {client.dashboard_link}")


# --- Data Loading Functions (formerly fixtures) ---
def load_comments_ddf():
    """Loads the comments Dask DataFrame using a cached Parquet file with full schema."""
    if not os.path.exists(COMMENTS_PATH):
        logger.error(f"Comments directory not found: {COMMENTS_PATH}")
        raise FileNotFoundError(f"Comments directory not found: {COMMENTS_PATH}")

    # Define full meta based on docs/schema.md (Comments)
    meta_comments = pd.DataFrame(
        {
            "id": pd.Series(dtype=pd.StringDtype()),
            "author": pd.Series(dtype=pd.StringDtype()),
            "link_id": pd.Series(dtype=pd.StringDtype()),
            "parent_id": pd.Series(dtype=pd.StringDtype()),
            "created_utc": pd.Series(dtype="int64"),
            "subreddit": pd.Series(dtype=pd.StringDtype()),
            "subreddit_id": pd.Series(dtype=pd.StringDtype()),
            "body": pd.Series(dtype=pd.StringDtype()),
            "score": pd.Series(dtype=pd.Int64Dtype()),
            "distinguished": pd.Series(dtype=pd.StringDtype()),  # String / Null
            "edited": pd.Series(dtype=pd.StringDtype()),  # Boolean / Integer -> String
            "stickied": pd.Series(dtype="boolean"),
            "retrieved_on": pd.Series(dtype="int64"),
            "gilded": pd.Series(dtype=pd.Int64Dtype()),
            "controversiality": pd.Series(dtype=pd.Int64Dtype()),
            "author_flair_css_class": pd.Series(
                dtype=pd.StringDtype()
            ),  # String / Null
            "author_flair_text": pd.Series(dtype=pd.StringDtype()),  # String / Null
            # Add other fields from schema.md if needed, e.g., 'score_hidden' if present
        }
    )
    # Define PyArrow schema from meta
    pa_schema_comments = pa.Schema.from_pandas(meta_comments)

    # Call the helper function with the full schema
    return load_or_create_cached_ddf(
        data_path=COMMENTS_PATH,
        file_pattern="RC_*.zst",
        cache_dir=CACHE_DIR,
        meta_df=meta_comments,  # Pass full meta
        pa_schema=pa_schema_comments,  # Pass full schema
        data_type_name="comments",
    )


def load_submissions_ddf():
    """Loads the submissions Dask DataFrame using a cached Parquet file with full schema."""
    if not os.path.exists(SUBMISSIONS_PATH):
        logger.error(f"Submissions directory not found: {SUBMISSIONS_PATH}")
        raise FileNotFoundError(f"Submissions directory not found: {SUBMISSIONS_PATH}")

    # Define full meta based on docs/schema.md (Submissions)
    meta_submissions = pd.DataFrame(
        {
            "id": pd.Series(dtype=pd.StringDtype()),
            "url": pd.Series(dtype=pd.StringDtype()),
            "permalink": pd.Series(dtype=pd.StringDtype()),
            "author": pd.Series(dtype=pd.StringDtype()),
            "created_utc": pd.Series(dtype="int64"),
            "subreddit": pd.Series(dtype=pd.StringDtype()),
            "subreddit_id": pd.Series(dtype=pd.StringDtype()),
            "selftext": pd.Series(dtype=pd.StringDtype()),
            "title": pd.Series(dtype=pd.StringDtype()),
            "num_comments": pd.Series(dtype=pd.Int64Dtype()),
            "score": pd.Series(dtype=pd.Int64Dtype()),
            "is_self": pd.Series(dtype="boolean"),
            "over_18": pd.Series(dtype="boolean"),  # NSFW flag
            "distinguished": pd.Series(dtype=pd.StringDtype()),  # String / Null
            "edited": pd.Series(dtype=pd.StringDtype()),  # Boolean / Integer -> String
            "domain": pd.Series(dtype=pd.StringDtype()),
            "stickied": pd.Series(dtype="boolean"),
            "locked": pd.Series(dtype="boolean"),
            "quarantine": pd.Series(dtype="boolean"),
            # 'hidden_score' in schema.md, but 'score_hidden' often used in Pushshift? Let's use 'score_hidden'
            "score_hidden": pd.Series(
                dtype="boolean"
            ),  # Typically 'score_hidden' in data
            "retrieved_on": pd.Series(dtype="int64"),
            "author_flair_css_class": pd.Series(
                dtype=pd.StringDtype()
            ),  # String / Null
            "author_flair_text": pd.Series(dtype=pd.StringDtype()),  # String / Null
            # Add other fields like 'gilded' if needed and present
        }
    )
    # Define PyArrow schema from meta
    pa_schema_submissions = pa.Schema.from_pandas(meta_submissions)

    # Call the helper function with the full schema
    return load_or_create_cached_ddf(
        data_path=SUBMISSIONS_PATH,
        file_pattern="RS_*.zst",
        cache_dir=CACHE_DIR,
        meta_df=meta_submissions,  # Pass full meta
        pa_schema=pa_schema_submissions,  # Pass full schema
        data_type_name="submissions",
    )


def _sample_target_users(
    comments_ddf: dd.DataFrame, num_users_to_find: int
) -> List[str]:
    """Samples the comments DataFrame to find a list of valid user authors."""
    logger.info(f"Sampling comments to find {num_users_to_find} valid users...")
    sampled_users = []
    # Increase sample size significantly to find multiple users
    sample_size = 1000  # How many comments to check
    logger.info(f"Attempting to sample from the head (sample size: {sample_size})...")
    try:
        # Sample directly from the Dask DataFrame head
        # Compute is needed here to get actual author names
        comments_sample = comments_ddf.head(sample_size, compute=True)

        if not comments_sample.empty:
            sampled_authors = comments_sample["author"].dropna().unique()
            found_users = 0
            for author in sampled_authors:
                # Filter out deleted users, bots, etc.
                if (
                    author
                    and author != "[deleted]"
                    and not author.endswith("Bot")
                    and not author.endswith("bot")
                ):
                    sampled_users.append(author)
                    found_users += 1
                    if found_users >= num_users_to_find:
                        break  # Stop once we have enough users

            logger.info(
                f"Found {len(sampled_users)} valid users from sample: {sampled_users}"
            )
            if len(sampled_users) < num_users_to_find:
                logger.warning(
                    f"Found only {len(sampled_users)}/{num_users_to_find} distinct valid users in the sample of size {sample_size}."
                )

        if not sampled_users:
            logger.error(
                f"Could not find any suitable users in the sample (size {sample_size})."
            )
            # Let the caller decide whether to skip or fail

    except KeyError as e:
        logger.error(f"Failed to find column '{e}' in comments sample.")
        raise  # Re-raise the error to be caught by the script
    except Exception as e:
        logger.error(f"Failed to sample users from comments: {e}")
        logger.exception("An error occurred while sampling users")
        raise  # Re-raise the error

    return sampled_users


def _run_and_validate_user_processing(
    user_id: str,
    comments_ddf: dd.DataFrame,
    submissions_ddf: dd.DataFrame,
    output_dir: str,
) -> bool:
    """Runs the context generation for a single user, validates, and saves artifact.
    Returns True if successful, False otherwise.
    """
    logger.info(
        f"\n--- Processing User: {user_id} --- --- --- --- --- --- --- --- --- --- --- --- ---"
    )
    success = True
    try:
        # --- Execute the function under test ---
        context_ddf = generate_user_context(
            user_id=user_id, ddf_comments=comments_ddf, ddf_submissions=submissions_ddf
        )

        # --- Compute the results ---
        logger.info(f"Computing the result Dask DataFrame for user {user_id}...")
        start_time = time.time()  # Start timing
        # Persist before compute can sometimes help with complex graphs
        future = context_ddf.persist()
        progress(future)  # Display progress bar
        result_pdf = future.compute()
        end_time = time.time()  # End timing
        logger.info(
            f"Computation finished for user {user_id}. Result shape: {result_pdf.shape}. Time: {end_time - start_time:.2f}s"
        )

        # --- Validations on the result ---
        if not isinstance(result_pdf, pd.DataFrame):
            logger.error(
                f"Result for user {user_id} is not a Pandas DataFrame after compute(). Type: {type(result_pdf)}"
            )
            return False

        # Check if expected columns exist in the result
        expected_cols = ["submission_id", "formatted_context", "user_comment_ids"]
        missing_result_cols = [
            col for col in expected_cols if col not in result_pdf.columns
        ]
        if missing_result_cols:
            err_msg = f"Result DataFrame for user {user_id} missing expected columns: {missing_result_cols}"
            logger.error(f"{err_msg}")
            return False

        logger.info(f"Result DataFrame head for user {user_id}:\n{result_pdf.head()}")

        # --- Save script artifact ---
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        # Sanitize username for filename
        safe_username = "".join(c if c.isalnum() else "_" for c in user_id)
        output_filename = f"user_context_output_{safe_username}.csv"
        output_path = os.path.join(output_dir, output_filename)
        try:
            logger.info(
                f"Saving script result artifact for {user_id} to: {output_path}"
            )
            result_pdf.to_csv(output_path, index=False)
            logger.info("Artifact saved successfully.")
        except Exception as e:
            logger.warning(
                f"Failed to save script artifact for {user_id} to {output_path}: {e}"
            )
            success = False  # Mark as partially failed if save fails

        logger.info(f"--- Processing for user {user_id} completed successfully. ---")

    except ValueError as e:
        logger.error(f"generate_user_context raised ValueError for user {user_id}: {e}")
        success = False
    except Exception as e:
        logger.error(
            f"generate_user_context raised an unexpected exception for user {user_id}: {e}"
        )
        logger.exception("An error occurred during user context generation")
        success = False

    return success


# --- Main Script Function ---
def main():
    """
    Main function to generate user context for multiple sampled users.
    """
    users_to_process = []
    if TARGET_USER is None:
        try:
            # Load comments_ddf just for sampling if TARGET_USER is None
            comments_for_sampling = load_comments_ddf()
            if comments_for_sampling is None:
                logger.error("Failed to load comments for sampling users. Exiting.")
                return
            users_to_process = _sample_target_users(
                comments_for_sampling, NUM_TARGET_USERS
            )
            if not users_to_process:
                logger.warning(
                    "Could not find any non-deleted users in the sample. No users to process."
                )
                return  # Exit if no users found
        except Exception as e:
            logger.error(f"Failed during user sampling: {e}")
            return  # Exit on sampling failure

    elif isinstance(TARGET_USER, list):
        logger.info(f"Using predefined list of target users: {TARGET_USER}")
        users_to_process = TARGET_USER
    elif isinstance(TARGET_USER, str):
        logger.info(f"Using predefined single target user: {TARGET_USER}")
        users_to_process = [TARGET_USER]
    else:
        logger.error(
            f"Invalid TARGET_USER type: {type(TARGET_USER)}. Should be None, list, or str."
        )
        return

    # Load main Dask DataFrames
    comments_ddf = load_comments_ddf()
    submissions_ddf = load_submissions_ddf()

    if comments_ddf is None or submissions_ddf is None:
        logger.error(
            "Failed to load comments or submissions Dask DataFrames. Exiting script."
        )
        return

    # Ensure required columns exist before calling the function
    required_comment_cols = list(comments_ddf.columns)
    required_submission_cols = list(submissions_ddf.columns)

    actual_comment_cols = comments_ddf.columns
    actual_submission_cols = submissions_ddf.columns

    logger.info(f"Expected Comment Columns (from Dask DF): {required_comment_cols}")
    logger.info(
        f"Expected Submission Columns (from Dask DF): {required_submission_cols}"
    )

    missing_comment_cols = [
        col for col in required_comment_cols if col not in actual_comment_cols
    ]
    missing_submission_cols = [
        col for col in required_submission_cols if col not in actual_submission_cols
    ]

    if missing_comment_cols:
        logger.error(
            f"Comments Dask DataFrame missing expected columns: {missing_comment_cols}. Exiting."
        )
        return
    if missing_submission_cols:
        logger.error(
            f"Submissions Dask DataFrame missing expected columns: {missing_submission_cols}. Exiting."
        )
        return

    # --- Loop through users and execute the function ---
    all_processing_successful = True
    script_output_dir = os.path.join(
        workspace_root, "data/script_results"
    )  # Define once

    for user_index, current_target_user in enumerate(users_to_process):
        user_processing_passed = _run_and_validate_user_processing(
            user_id=current_target_user,
            comments_ddf=comments_ddf,
            submissions_ddf=submissions_ddf,
            output_dir=script_output_dir,
        )
        if not user_processing_passed:
            all_processing_successful = False
            # Decide whether to continue processing other users or stop
            # For now, let's continue to see all failures

    # --- Dask Cluster Teardown (after all users) ---
    logger.info("\nShutting down Dask client and cluster...")
    client.close()
    cluster.close()
    logger.info("Dask client and cluster shut down.")

    if all_processing_successful:
        logger.info("All user context generation tasks completed successfully.")
    else:
        logger.warning(
            "One or more user context generation tasks failed or had issues."
        )


if __name__ == "__main__":
    main()

# To run this script:
# 1. Make sure you have the necessary libraries installed (pandas, dask, distributed, pyarrow).
# 2. Ensure your data is in `data/reddit/comments` and `data/reddit/submissions`.
# 3. Configure TARGET_USER (None to sample, or a specific list/string) and NUM_TARGET_USERS.
# 4. Navigate to your project root directory in the terminal.
# 5. Run `python scripts/data_preparation/generate_user_context_script.py`.
# You should see the dashboard link printed and processing for multiple users.
