import json
from typing import Any, Dict, List, Optional, Set, Tuple

import dask.dataframe as dd
import pandas as pd

from .dask_helpers import (
    _filter_dataframes_by_links,
    _find_relevant_link_ids,
    _prepare_and_merge_data,
    _validate_input_dataframes,
)

# Import helpers from the new modules
from .reddit_helpers import (
    _get_all_ancestors_optimized,
    build_nested_thread,
    format_submission_context,
)


def _get_ancestors(target_comment_id: str, comment_map: pd.DataFrame) -> Set[str]:
    """Finds all ancestor comment IDs for a single target comment within the group."""
    ancestors = set()
    current_id = target_comment_id
    # Limit depth to prevent infinite loops in case of data inconsistency
    max_depth = 100
    depth = 0

    while depth < max_depth:
        if current_id not in comment_map.index:
            # Current comment ID not found in the group (shouldn't happen if starting from user comment)
            break

        comment = comment_map.loc[current_id]
        parent_id_full = comment.get("parent_id")

        if not parent_id_full or not isinstance(parent_id_full, str):
            # No parent information or invalid format
            break

        try:
            parent_type, parent_id_short = parent_id_full.split("_", 1)
        except ValueError:
            # parent_id format is unexpected
            break

        if parent_type == "t3":  # Parent is the submission itself
            break
        elif parent_type == "t1":  # Parent is another comment
            # Check if the parent comment exists within this submission group
            if parent_id_short in comment_map.index:
                ancestors.add(parent_id_short)
                current_id = parent_id_short  # Continue traversing up
            else:
                # Parent comment not in this submission's group (e.g., deleted or different partition scope)
                break
        else:
            # Unexpected parent type (e.g., t2 for accounts, t4 for messages)
            break
        depth += 1

    return ancestors


def _initialize_thread_build(
    relevant_comment_ids: Set[str], comment_map: pd.DataFrame
) -> Tuple[Dict[str, pd.Series], List[str], Dict[str, List[str]], List[str]]:
    """
    Initializes data structures needed for building the nested thread.
    Returns a tuple containing:
        - comment_data_map: Dict mapping relevant comment IDs to their data.
        - root_comment_ids: List of comment IDs that are top-level replies.
        - parent_to_child_ids: Dict mapping parent IDs (t1_ or t3_) to child IDs.
        - sorted_relevant_ids: List of relevant comment IDs sorted by creation time.
    """
    comment_data_map = {}
    valid_relevant_ids = set()
    for cid in relevant_comment_ids:
        if cid in comment_map.index:
            comment_data_map[cid] = comment_map.loc[cid]
            valid_relevant_ids.add(cid)
        # Else: A required ancestor comment was not found in the group data, skip it.

    if not valid_relevant_ids:
        return {}, [], {}, []  # Return empty structures

    parent_to_child_ids = {}
    for cid in valid_relevant_ids:
        comment = comment_data_map[cid]
        parent_id_full = comment.get("parent_id")
        if parent_id_full and isinstance(parent_id_full, str):
            parent_key = parent_id_full
            if parent_key not in parent_to_child_ids:
                parent_to_child_ids[parent_key] = []
            parent_to_child_ids[parent_key].append(cid)

    # Sort relevant IDs chronologically
    sorted_relevant_ids = sorted(
        list(valid_relevant_ids), key=lambda cid: comment_data_map[cid]["created_utc"]
    )

    root_comment_ids = []
    for cid in sorted_relevant_ids:
        comment = comment_data_map[cid]
        parent_id_full = comment.get("parent_id")
        # Treat comments replying to submission (t3_) or having no/invalid parent as roots
        if (
            parent_id_full
            and isinstance(parent_id_full, str)
            and parent_id_full.startswith("t3_")
        ) or (not parent_id_full or not isinstance(parent_id_full, str)):
            root_comment_ids.append(cid)

    return comment_data_map, root_comment_ids, parent_to_child_ids, sorted_relevant_ids


def _build_tree_structure(
    comment_data_map: Dict[str, pd.Series],
    root_comment_ids: List[str],
    parent_to_child_ids: Dict[str, List[str]],
    sorted_relevant_ids: List[str],
) -> List[Dict[str, Any]]:
    """
    Builds the nested comment thread structure iteratively using pre-initialized data.
    """
    if not comment_data_map:  # If initialization returned empty map
        return []

    # Create basic dicts for all relevant comments first
    comments_processed = {}
    for cid in sorted_relevant_ids:
        comment = comment_data_map[cid]
        comments_processed[cid] = {
            # "id": cid, # Optional debug info
            "author": comment.get("author", "[unknown]"),
            "body": comment.get("body", "[unavailable]"),
            "replies": [],
        }

    final_nested_structure = []
    processed_for_tree = set()
    queue = []  # Use a queue for BFS-like processing

    # Add roots to the final structure and the queue
    for root_id in root_comment_ids:
        if root_id in comments_processed:
            final_nested_structure.append(comments_processed[root_id])
            processed_for_tree.add(root_id)
            queue.append(root_id)

    # Iteratively process children
    head = 0
    while head < len(queue):
        parent_id = queue[head]
        head += 1

        parent_node = comments_processed.get(parent_id)
        if not parent_node:
            continue

        # Find children of this parent (use t1_ prefix for lookup)
        parent_lookup_key = f"t1_{parent_id}"
        child_ids = parent_to_child_ids.get(parent_lookup_key, [])

        # Sort children by creation time before adding
        sorted_child_ids = sorted(
            [
                cid for cid in child_ids if cid in comments_processed
            ],  # Ensure child exists
            key=lambda cid: comment_data_map[cid]["created_utc"],
        )

        for child_id in sorted_child_ids:
            if child_id not in processed_for_tree:
                child_node = comments_processed[child_id]
                parent_node["replies"].append(child_node)
                processed_for_tree.add(child_id)
                queue.append(child_id)  # Add child to queue

    # Handle comments whose parents were not in the relevant set (orphans)
    for cid in sorted_relevant_ids:
        if cid not in processed_for_tree:
            # Add it as a top-level item, preserving some order via sorted_relevant_ids
            final_nested_structure.append(comments_processed[cid])
            processed_for_tree.add(cid)

    # Optional: Sort final top-level list by original timestamp?
    # Current order is roots first (by timestamp), then orphans (by timestamp).
    # This seems reasonable.

    return final_nested_structure


def _build_nested_thread(
    relevant_comment_ids: Set[str], comment_map: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Builds the nested comment thread structure containing only relevant comments,
    ordered chronologically where possible and structured hierarchically.
    """
    if not relevant_comment_ids:
        return []

    # 1. Initialize data structures
    comment_data_map, root_ids, parent_to_child_map, sorted_ids = (
        _initialize_thread_build(relevant_comment_ids, comment_map)
    )

    # 2. Build the tree structure
    nested_structure = _build_tree_structure(
        comment_data_map, root_ids, parent_to_child_map, sorted_ids
    )

    return nested_structure


def process_submission_group(
    group: pd.DataFrame,
    target_user: str,
) -> Optional[pd.DataFrame]:
    """
    Processes comments for a single submission (group) to find user comments
    and build the minimal conversation context leading to them.
    Assumes the input group DataFrame contains merged submission data
    (title, selftext, is_self) associated with the comments.

    Args:
        group: Pandas DataFrame containing all comments for a single submission_id,
               merged with submission data. Expected columns include comment fields
               ('id_x', 'author', 'link_id', 'parent_id', 'body', 'created_utc') and
               submission fields ('subreddit', 'title', 'selftext', 'is_self').
        target_user: The username to filter comments for.

    Returns:
        A Pandas DataFrame with a single row containing the formatted context
        and user comment IDs for this submission, or None if the user didn't comment
        or required data is missing.
    """
    # Filter for comments by the target user within this submission group
    # Use 'author_x' if merge added suffix, otherwise 'author'
    author_col = "author_x" if "author_x" in group.columns else "author"
    user_comments = group[group[author_col] == target_user]
    if user_comments.empty:
        return None

    # Get the submission ID (link_id) and details from the first row
    first_row = group.iloc[0]
    submission_id_full = first_row.get("link_id")

    if not isinstance(submission_id_full, str) or not submission_id_full.startswith(
        "t3_"
    ):
        print(f"Warning: Invalid link_id found in group: {submission_id_full}")
        return None

    try:
        submission_id_short = submission_id_full.split("_", 1)[1]
    except IndexError:
        print(f"Warning: Could not extract short ID from link_id: {submission_id_full}")
        return None

    # Extract submission data directly from the first row
    # Use suffixed names if they exist after merge (e.g., 'title_y')
    submission_data = pd.Series(
        {
            "subreddit": first_row.get(
                "subreddit_y", first_row.get("subreddit", "N/A")
            ),  # Prefer _y suffix
            "title": first_row.get(
                "title_y", first_row.get("title", "N/A")
            ),  # Prefer _y suffix
            "selftext": first_row.get(
                "selftext_y", first_row.get("selftext", "")
            ),  # Prefer _y suffix
            "is_self": first_row.get(
                "is_self_y", first_row.get("is_self", False)
            ),  # Prefer _y suffix
            # Correctly extract author and created_utc using _y suffix from submission
            "author": first_row.get(
                "author_y", first_row.get("author", "[unknown_author_fallback]")
            ),
            "created_utc": first_row.get(
                "created_utc_y", first_row.get("created_utc", None)
            ),
        }
    )

    # Check if submission data seems valid (at least title is not N/A)
    if submission_data["title"] == "N/A":
        print(
            f"Warning: Submission data (title) appears missing for group {submission_id_short}. Skipping."
        )
        return None

    # Create comment map using the comment ID column ('id_x' after merge)
    comment_id_col = "id_x" if "id_x" in group.columns else "id"
    comment_map = group.drop_duplicates(subset=[comment_id_col]).set_index(
        comment_id_col, drop=False
    )

    # Get user comment IDs from the correct column
    user_comment_ids_in_submission = list(user_comments[comment_id_col].unique())
    user_comment_ids_set = set(user_comment_ids_in_submission)

    # Find ancestors using helper from reddit_helpers
    all_ancestor_ids = _get_all_ancestors_optimized(user_comment_ids_set, comment_map)
    relevant_comment_ids = all_ancestor_ids.union(user_comment_ids_set)

    # Build nested thread using helper from reddit_helpers
    # Requires 'created_utc' column - check existence
    created_utc_col = (
        "created_utc_x" if "created_utc_x" in comment_map.columns else "created_utc"
    )
    if created_utc_col not in comment_map.columns:
        print(
            f"Warning: '{created_utc_col}' column missing in comment data for submission {submission_id_short}. Thread structure might be incorrect."
        )
        # Add a dummy column if needed for the helper function to run
        comment_map[created_utc_col] = 0

    # Ensure the map passed to build_nested_thread uses the base column names expected by it
    # Rename columns like 'id_x' to 'id', 'author_x' to 'author' etc. if needed
    rename_map = {}
    base_cols = ["id", "author", "link_id", "parent_id", "body", "created_utc"]
    for base_col in base_cols:
        suffixed_col = f"{base_col}_x"
        if suffixed_col in comment_map.columns and base_col not in comment_map.columns:
            rename_map[suffixed_col] = base_col

    if rename_map:
        comment_map_for_build = comment_map.rename(columns=rename_map)
    else:
        comment_map_for_build = comment_map

    # Ensure the required base columns exist in the map before building thread
    missing_base_cols = [
        col for col in base_cols if col not in comment_map_for_build.columns
    ]
    if missing_base_cols:
        print(
            f"Error: Cannot build thread for {submission_id_short}. Missing essential columns after potential rename: {missing_base_cols}"
        )
        return None

    # Build the nested comment thread (excluding submission)
    nested_thread = build_nested_thread(
        relevant_comment_ids, comment_map_for_build, target_user
    )

    # Combine submission data and nested comments into the final structure
    full_thread_structure = format_submission_context(
        submission_data, nested_thread, target_user
    )

    # Convert the final structure to a JSON string
    json_context = json.dumps(full_thread_structure, indent=2)

    # Create the resulting DataFrame row
    result_df = pd.DataFrame(
        {
            "submission_id": [submission_id_short],
            "formatted_context": [json_context],
            "user_comment_ids": [user_comment_ids_in_submission],
        }
    )
    return result_df


def generate_user_context(
    user_id: str, ddf_comments: dd.DataFrame, ddf_submissions: dd.DataFrame
) -> dd.DataFrame:
    """
    Generates a Dask DataFrame containing the conversation context for each submission
    a specific user commented on.

    Args:
        user_id: The Reddit username (author) to generate context for.
        ddf_comments: Dask DataFrame of comments.
                      Requires columns: 'id', 'author', 'link_id', 'parent_id', 'body', 'created_utc'.
        ddf_submissions: Dask DataFrame of submissions.
                         Requires columns: 'id', 'subreddit', 'title', 'selftext', 'is_self'.

    Returns:
        A Dask DataFrame with columns:
        - submission_id: The ID of the submission (without 't3_' prefix).
        - formatted_context: A JSON string representing the submission post
                             (title, subreddit, body) and the comment thread including
                             the user's comments and their ancestors.
        - user_comment_ids: A list of comment IDs made by the target user in that submission.

    Raises:
        ValueError: If required columns are missing in the input DataFrames.
    """

    # --- Input Validation (using dask_helpers) ---
    _validate_input_dataframes(ddf_comments, ddf_submissions)

    # --- Find relevant submissions (using dask_helpers) ---
    relevant_link_ids = _find_relevant_link_ids(ddf_comments, user_id)

    if not relevant_link_ids:
        print(f"No comments found for user '{user_id}'. Returning empty DataFrame.")
        meta_empty = pd.DataFrame(
            {
                "submission_id": pd.Series(dtype="string"),
                "formatted_context": pd.Series(dtype="string"),
                "user_comment_ids": pd.Series(dtype="object"),
            }
        )
        return dd.from_pandas(meta_empty, npartitions=1)

    # --- Filter DataFrames (using dask_helpers) ---
    ddf_comments_filtered, ddf_submissions_filtered = _filter_dataframes_by_links(
        ddf_comments, ddf_submissions, relevant_link_ids
    )

    # --- Merge Filtered Data (using dask_helpers) ---
    ddf_merged = _prepare_and_merge_data(
        ddf_comments_filtered, ddf_submissions_filtered
    )

    # --- Define Output Metadata for Apply ---
    meta_apply = pd.DataFrame(
        {
            "submission_id": pd.Series(dtype="string"),
            "formatted_context": pd.Series(dtype="string"),
            "user_comment_ids": pd.Series(dtype="object"),
        }
    )

    # --- Group Merged Data and Apply Context Generation ---
    # Group by link_id. Need to ensure link_id exists after merge.
    # If merge adds suffixes, use 'link_id_x' or similar.
    link_id_col_merged = "link_id_x" if "link_id_x" in ddf_merged.columns else "link_id"
    print(f"Grouping filtered merged data by '{link_id_col_merged}'...")
    grouped_merged_comments = ddf_merged.groupby(link_id_col_merged)

    print(
        "Applying context generation logic to filtered groups (constructing Dask graph)..."
    )
    user_context_ddf = grouped_merged_comments.apply(
        process_submission_group,  # This function remains in this file
        target_user=user_id,
        meta=meta_apply,
    ).reset_index(drop=True)

    print("Dask graph for context generation constructed successfully.")
    return user_context_ddf


# --- Example Usage ---
# (Example usage remains the same, potentially remove if not needed)
# if __name__ == "__main__":
#     from dask.distributed import Client
#     import dask.dataframe as dd
#     import time
#
#     # --- Setup Dask Client ---
#     # client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')
#     # print(f"Dask dashboard link: {client.dashboard_link}")
#
#     # --- Load Sample Data (Replace with your actual data loading) ---
#     print("Creating sample Dask DataFrames...")
#     comments_data = {
#         'id': [f'c{i}' for i in range(10)] + ['c10', 'c11'],
#         'author': ['user_a', 'user_b', 'user_a', 'user_c', 'user_b', 'user_a', 'user_d', 'user_a', 'user_b', 'user_c', 'user_a', 'user_b'],
#         'link_id': ['t3_s1'] * 5 + ['t3_s2'] * 5 + ['t3_s1', 't3_s2'], # Comments spread across 2 submissions
#         'parent_id': ['t3_s1', 't1_c0', 't1_c1', 't3_s1', 't1_c3', 't3_s2', 't1_c5', 't1_c6', 't3_s2', 't1_c8', 't1_c2', 't1_c7'], # Sample hierarchy
#         'body': [f'Comment body {i}' for i in range(12)],
#         'created_utc': [1678886400 + i * 10 for i in range(12)] # Timestamps
#     }
#     submissions_data = {
#         'id': ['s1', 's2', 's3'], # s3 has no comments in sample
#         'title': ['Submission Title 1', 'Submission Title 2', 'Submission Title 3'],
#         'selftext': ['Body of submission 1.', 'Link post content here.', ''],
#         'is_self': [True, True, False] # s3 is a link post
#     }
#
#     comments_pdf = pd.DataFrame(comments_data)
#     submissions_pdf = pd.DataFrame(submissions_data)
#
#     # Create Dask DataFrames (e.g., 2 partitions)
#     comments_ddf = dd.from_pandas(comments_pdf, npartitions=2)
#     submissions_ddf = dd.from_pandas(submissions_pdf, npartitions=1)
#     print("Sample DataFrames created.")
#
#     # --- Define Target User ---
#     target_user = "user_a"
#
#     # --- Generate Context ---
#     start_time = time.time()
#     print(f"Generating context for user: {target_user}")
#     try:
#         context_ddf = generate_user_context(target_user, comments_ddf, submissions_ddf)
#
#         # --- Compute and Display Results ---
#         print("Computing the result...")
#         result_pdf = context_ddf.compute()
#         end_time = time.time()
#
#         print("\n--- Generated Context ---")
#         if not result_pdf.empty:
#             for index, row in result_pdf.iterrows():
#                 print(f"Submission ID: {row['submission_id']}")
#                 print(f"User Comment IDs: {row['user_comment_ids']}")
#                 print(f"Formatted Context:\n{row['formatted_context']}")
#                 print("-" * 20)
#         else:
#             print(f"No comments found for user '{target_user}' or context could not be generated.")
#
#         print(f"Total execution time: {end_time - start_time:.2f} seconds")
#
#     except ValueError as e:
#         print(f"Execution Error: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#     finally:
#         pass
