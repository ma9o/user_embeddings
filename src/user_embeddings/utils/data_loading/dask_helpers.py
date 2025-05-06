import logging
from typing import List, Tuple

import dask.dataframe as dd

logger = logging.getLogger(__name__)


def _validate_input_dataframes(
    ddf_comments: dd.DataFrame, ddf_submissions: dd.DataFrame
):
    """Validates that input DataFrames have the required columns."""
    required_comment_cols = [
        "id",
        "author",
        "link_id",
        "parent_id",
        "body",
        "created_utc",
    ]
    missing_comment_cols = [
        col for col in required_comment_cols if col not in ddf_comments.columns
    ]
    if missing_comment_cols:
        raise ValueError(
            f"ddf_comments is missing required columns: {missing_comment_cols}"
        )

    required_submission_cols = [
        "id",
        "subreddit",
        "title",
        "selftext",
        "is_self",
        "author",
        "created_utc",
    ]
    missing_submission_cols = [
        col for col in required_submission_cols if col not in ddf_submissions.columns
    ]
    if missing_submission_cols:
        raise ValueError(
            f"ddf_submissions is missing required columns: {missing_submission_cols}"
        )


def _find_relevant_link_ids(ddf_comments: dd.DataFrame, user_id: str) -> List[str]:
    """Finds unique link_ids for submissions the target user commented on."""
    logger.info(f"Filtering comments for user '{user_id}' to find relevant submissions...")
    # Select only the link_id column early
    user_comments_links = ddf_comments[ddf_comments["author"] == user_id][["link_id"]]
    # Drop duplicates and compute
    relevant_link_ids = (
        user_comments_links.dropna().drop_duplicates()["link_id"].compute().tolist()
    )
    logger.info(
        f"Found {len(relevant_link_ids)} submissions with comments from user '{user_id}'."
    )
    return relevant_link_ids


def _filter_dataframes_by_links(
    ddf_comments: dd.DataFrame,
    ddf_submissions: dd.DataFrame,
    relevant_link_ids: List[str],
) -> Tuple[dd.DataFrame, dd.DataFrame]:
    """Filters comments and submissions DataFrames to include only relevant entries."""
    if not relevant_link_ids:
        # Return empty Dask DataFrames matching original structure
        # Use known divisions/meta if possible, otherwise empty slices
        return ddf_comments.map_partitions(
            lambda pdf: pdf.iloc[0:0], meta=ddf_comments._meta
        ), ddf_submissions.map_partitions(
            lambda pdf: pdf.iloc[0:0], meta=ddf_submissions._meta
        )

    logger.info("Filtering comments and submissions based on relevant link_ids...")
    # Filter comments based on the computed list
    ddf_comments_filtered = ddf_comments[
        ddf_comments["link_id"].isin(relevant_link_ids)
    ].copy()

    # Filter submissions based on the computed list
    relevant_submission_ids_short = [
        link_id[3:]
        for link_id in relevant_link_ids
        if link_id and link_id.startswith("t3_")
    ]
    ddf_submissions_filtered = ddf_submissions[
        ddf_submissions["id"].isin(relevant_submission_ids_short)
    ].copy()

    logger.info(f"Filtered comments partitions: {ddf_comments_filtered.npartitions}")
    logger.info(f"Filtered submissions partitions: {ddf_submissions_filtered.npartitions}")
    return ddf_comments_filtered, ddf_submissions_filtered


def _prepare_and_merge_data(
    ddf_comments_filtered: dd.DataFrame, ddf_submissions_filtered: dd.DataFrame
) -> dd.DataFrame:
    """Prepares filtered DataFrames and merges them."""
    logger.info("Preparing filtered comments and submissions for merge...")

    # Define required columns locally (could be passed as args)
    required_comment_cols = [
        "id",
        "author",
        "link_id",
        "parent_id",
        "body",
        "created_utc",
    ]
    required_submission_cols = [
        "id",
        "subreddit",
        "title",
        "selftext",
        "is_self",
        "author",
        "created_utc",
    ]

    # --- Prepare Comments DataFrame ---
    # Check if empty using partitions/divisions if possible, fallback to compute head(1)
    # Assuming non-empty based on filtering logic, proceed with preparation
    # Select necessary columns
    ddf_comments_prep = ddf_comments_filtered[required_comment_cols].copy()

    # Ensure 'link_id' is usable and extract short submission ID
    # Dropna on link_id - should be redundant if filtering worked, but safe
    ddf_comments_prep = ddf_comments_prep.dropna(subset=["link_id"])
    # Ensure it starts with t3_
    ddf_comments_prep = ddf_comments_prep[
        ddf_comments_prep["link_id"].str.startswith("t3_")
    ]

    # Create the merge key
    ddf_comments_prep["submission_id_short"] = ddf_comments_prep["link_id"].str.slice(
        start=3
    )

    # Ensure correct type for merge key
    ddf_comments_prep = ddf_comments_prep.astype({"submission_id_short": "string"})

    # --- Prepare Submissions DataFrame ---
    # Select necessary columns
    ddf_submissions_prep = ddf_submissions_filtered[required_submission_cols].copy()

    # Ensure correct type for merge key
    ddf_submissions_prep = ddf_submissions_prep.astype({"id": "string"})

    # --- Perform Dask Merge ---
    logger.info(
        "Merging prepared comments with prepared submissions (constructing Dask graph)..."
    )
    ddf_merged = dd.merge(
        ddf_comments_prep,
        ddf_submissions_prep,
        left_on="submission_id_short",
        right_on="id",
        how="left",  # Keep all comments from relevant submissions
    )
    logger.info("Merge graph constructed.")
    return ddf_merged
