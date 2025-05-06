from datetime import datetime
import logging
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# --- Ancestor Finding ---


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


def _get_all_ancestors_optimized(
    start_comment_ids: Set[str], comment_map: pd.DataFrame
) -> Set[str]:
    """Finds all unique ancestor comment IDs for a set of starting comments within the group."""
    all_ancestors = set()
    # Initialize queue with valid starting IDs present in the map
    queue = {cid for cid in start_comment_ids if cid in comment_map.index}
    # Keep track of visited nodes (including starts) to prevent cycles and redundant work
    visited_for_ancestors = set(queue)
    # Limit total iterations as a safety measure against unexpected data issues
    max_iterations = len(comment_map) * 2  # Heuristic limit
    iterations = 0

    while queue and iterations < max_iterations:
        current_id = queue.pop()
        iterations += 1

        # Should always be in map if it was added to the queue
        if current_id not in comment_map.index:
            continue

        comment = comment_map.loc[current_id]
        parent_id_full = comment.get("parent_id")

        if not parent_id_full or not isinstance(parent_id_full, str):
            continue

        try:
            parent_type, parent_id_short = parent_id_full.split("_", 1)
        except ValueError:
            continue  # Skip malformed parent IDs

        # We only care about comment parents (t1)
        if parent_type == "t1":
            # Check if the parent comment exists within this submission group
            if parent_id_short in comment_map.index:
                # If we haven't processed this parent yet
                if parent_id_short not in visited_for_ancestors:
                    all_ancestors.add(parent_id_short)
                    visited_for_ancestors.add(parent_id_short)
                    queue.add(
                        parent_id_short
                    )  # Add parent to the queue to find its ancestors
            # else: Parent comment not in this submission's group - stop traversal up this path

    if iterations >= max_iterations:
        logger.warning(
            f"Ancestor search reached max iterations ({max_iterations}). Might be incomplete."
        )

    return all_ancestors


# --- Nested Thread Building ---


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
    # Ensure 'created_utc' exists before sorting
    if "created_utc" not in comment_map.columns:
        logger.warning(
            "'created_utc' column missing in comment map for thread building. Using arbitrary order."
        )
        sorted_relevant_ids = list(
            valid_relevant_ids
        )  # Keep original order or arbitrary order
    else:
        sorted_relevant_ids = sorted(
            list(valid_relevant_ids),
            key=lambda cid: comment_data_map[cid]["created_utc"],
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
    target_username: str,  # Added for masking
) -> List[Dict[str, Any]]:
    """
    Builds the generic nested thread structure (user, time, content, replies) iteratively.
    """
    if not comment_data_map:  # If initialization returned empty map
        return []

    # Check if 'created_utc' exists for timestamp formatting and sorting
    has_created_utc = False
    if sorted_relevant_ids:
        first_id = sorted_relevant_ids[0]
        if first_id in comment_data_map and "created_utc" in comment_data_map[first_id]:
            has_created_utc = True

    # Create basic dicts for all relevant comments first
    comments_processed = {}
    for cid in sorted_relevant_ids:
        comment = comment_data_map[cid]
        author = comment.get("author", "[unknown]")
        user = "SUBJECT" if author == target_username else author

        timestamp_str = "[unknown_time]"
        if has_created_utc and pd.notna(comment["created_utc"]):
            try:
                # Convert Unix timestamp to datetime object
                dt_object = datetime.utcfromtimestamp(comment["created_utc"])
                # Format datetime object
                timestamp_str = dt_object.strftime("%d-%m-%Y %H:%M")
            except (ValueError, TypeError):
                timestamp_str = "[invalid_time]"  # Handle potential conversion errors

        comments_processed[cid] = {
            "user": user,
            "time": timestamp_str,
            "content": comment.get("body", "[unavailable]"),
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

        # Sort children by creation time before adding, if possible
        child_ids_in_map = [cid for cid in child_ids if cid in comments_processed]
        if has_created_utc:
            sorted_child_ids = sorted(
                child_ids_in_map, key=lambda cid: comment_data_map[cid]["created_utc"]
            )
        else:
            sorted_child_ids = child_ids_in_map  # Keep arbitrary order if no timestamp

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


def build_nested_thread(
    relevant_comment_ids: Set[str],
    comment_map: pd.DataFrame,
    target_username: str,  # Added for masking
) -> List[Dict[str, Any]]:
    """
    Builds the nested comment thread structure containing only relevant comments,
    ordered chronologically where possible and structured hierarchically.
    (Public wrapper for the helper functions)
    """
    if not relevant_comment_ids:
        return []

    # 1. Initialize data structures
    comment_data_map, root_ids, parent_to_child_map, sorted_ids = (
        _initialize_thread_build(relevant_comment_ids, comment_map)
    )

    # Handle case where created_utc might be missing
    if "created_utc" not in comment_map.columns:
        # If missing, we might need a fallback for sorting or time formatting
        # For now, _build_tree_structure handles missing timestamps gracefully
        pass

    # 2. Build the tree
    nested_thread = _build_tree_structure(
        comment_data_map,
        root_ids,
        parent_to_child_map,
        sorted_ids,
        target_username,  # Pass target_username
    )

    return nested_thread


def format_submission_context(
    submission_data: pd.Series,
    nested_thread: List[Dict[str, Any]],
    target_username: str,  # Added for masking
) -> List[Dict[str, Any]]:
    """
    Formats the initial submission post and prepends it to the nested comment thread.
    Returns the complete thread structure starting with the submission.
    """
    author = submission_data.get("author", "[unknown]")
    user = "SUBJECT" if author == target_username else author

    timestamp_str = "[unknown_time]"
    if pd.notna(submission_data.get("created_utc")):
        try:
            dt_object = datetime.utcfromtimestamp(submission_data["created_utc"])
            timestamp_str = dt_object.strftime("%d-%m-%Y %H:%M")
        except (ValueError, TypeError):
            timestamp_str = "[invalid_time]"

    submission_node = {
        "user": user,
        "time": timestamp_str,
        "content": f"""Title: {submission_data.get("title", "[no title]")}
Subreddit: {submission_data.get("subreddit", "[no subreddit]")}
Body: {submission_data.get("selftext", "[no body]")}""",
        "replies": nested_thread,  # The built comment thread becomes replies to the submission
    }
    # The final structure is a list containing the submission node
    return [submission_node]


# Example Usage Placeholder (Illustrative)
# This part would typically be in the main processing script (e.g., dask_processing.py)
#
# submission_data = ... # pd.Series for the submission
# comment_map = ... # pd.DataFrame of all comments in the submission
# user_comment_ids = ... # Set of comment IDs by the target user
# target_username = ... # The username to mask
#
# # 1. Find all relevant comments (user's comments + their ancestors)
# ancestor_ids = _get_all_ancestors_optimized(user_comment_ids, comment_map)
# relevant_ids = user_comment_ids.union(ancestor_ids)
#
# # 2. Build the nested comment thread
# nested_comments = build_nested_thread(relevant_ids, comment_map, target_username)
#
# # 3. Format with submission context
# full_thread_structure = format_submission_context(submission_data, nested_comments, target_username)
#
# # 4. Convert to JSON (likely done elsewhere)
# # import json
# # json_output = json.dumps(full_thread_structure, indent=2)
