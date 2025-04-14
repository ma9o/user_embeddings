import dask.dataframe as dd
import pandas as pd
import yaml
from typing import List, Dict, Any, Optional, Set

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

        if parent_type == "t3": # Parent is the submission itself
            break 
        elif parent_type == "t1": # Parent is another comment
            # Check if the parent comment exists within this submission group
            if parent_id_short in comment_map.index: 
                 ancestors.add(parent_id_short)
                 current_id = parent_id_short # Continue traversing up
            else: 
                 # Parent comment not in this submission's group (e.g., deleted or different partition scope)
                 break
        else:
            # Unexpected parent type (e.g., t2 for accounts, t4 for messages)
            break
        depth += 1
            
    return ancestors

def _get_all_ancestors_optimized(start_comment_ids: Set[str], comment_map: pd.DataFrame) -> Set[str]:
    """Finds all unique ancestor comment IDs for a set of starting comments within the group."""
    all_ancestors = set()
    # Initialize queue with valid starting IDs present in the map
    queue = {cid for cid in start_comment_ids if cid in comment_map.index}
    # Keep track of visited nodes (including starts) to prevent cycles and redundant work
    visited_for_ancestors = set(queue)
    # Limit total iterations as a safety measure against unexpected data issues
    max_iterations = len(comment_map) * 2 # Heuristic limit
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
            continue # Skip malformed parent IDs

        # We only care about comment parents (t1)
        if parent_type == "t1":
            # Check if the parent comment exists within this submission group
            if parent_id_short in comment_map.index: 
                # If we haven't processed this parent yet
                if parent_id_short not in visited_for_ancestors:
                    all_ancestors.add(parent_id_short)
                    visited_for_ancestors.add(parent_id_short)
                    queue.add(parent_id_short) # Add parent to the queue to find its ancestors
            # else: Parent comment not in this submission's group - stop traversal up this path
            
    if iterations >= max_iterations:
        print(f"Warning: Ancestor search reached max iterations ({max_iterations}). Might be incomplete.")
        
    return all_ancestors

def _build_nested_thread(relevant_comment_ids: Set[str], comment_map: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Builds the nested comment thread structure containing only relevant comments,
    ordered chronologically where possible and structured hierarchically.
    """
    if not relevant_comment_ids:
        return []
        
    # Create lookup map for relevant comment data only
    comment_data_map = {}
    valid_relevant_ids = set()
    for cid in relevant_comment_ids:
         if cid in comment_map.index:
              comment_data_map[cid] = comment_map.loc[cid]
              valid_relevant_ids.add(cid)
         # Else: A required ancestor comment was not found in the group data, skip it.

    if not valid_relevant_ids:
         return []

    # Temporary structure holding processed comment dicts, keyed by comment ID
    comments_processed = {} 
    # Map parent_id (short 't1_' or full 't3_') to list of child comment IDs
    parent_to_child_ids = {} 
    
    # Populate parent_to_child_ids map
    for cid in valid_relevant_ids:
        comment = comment_data_map[cid]
        parent_id_full = comment.get("parent_id")
        if parent_id_full and isinstance(parent_id_full, str):
             parent_key = parent_id_full # Use t3_... or t1_... as key initially
             if parent_key not in parent_to_child_ids:
                  parent_to_child_ids[parent_key] = []
             parent_to_child_ids[parent_key].append(cid)

    # Sort comments by creation time to help build threads mostly top-down
    # Note: UTC timestamps ensure correct chronological order
    sorted_ids = sorted(list(valid_relevant_ids), key=lambda cid: comment_data_map[cid]['created_utc'])

    root_comment_ids = [] # IDs of comments directly replying to the submission (t3_...)

    # First pass: create basic dicts and identify roots
    for cid in sorted_ids:
        comment = comment_data_map[cid]
        comment_dict = {
            # "id": cid, # Optional: include for debugging
            "author": comment.get("author", "[unknown]"),
            "body": comment.get("body", "[unavailable]"),
            "replies": [] # Initialize replies list
        }
        comments_processed[cid] = comment_dict # Store processed dict

        parent_id_full = comment.get("parent_id")
        if parent_id_full and isinstance(parent_id_full, str) and parent_id_full.startswith("t3_"):
             root_comment_ids.append(cid)
        elif not parent_id_full or not isinstance(parent_id_full, str):
             # If no parent or invalid parent, treat as root? Or discard? Let's treat as root for now.
             root_comment_ids.append(cid)


    # Second pass: build the tree structure recursively (or iteratively)
    # We can use a stack-based approach for iterative tree building
    
    final_nested_structure = []
    processed_for_tree = set()

    # Process roots first
    queue = root_comment_ids[:] # Start with identified roots

    # Add roots to the final structure first
    for root_id in root_comment_ids:
         if root_id in comments_processed:
              final_nested_structure.append(comments_processed[root_id])
              processed_for_tree.add(root_id)

    # Iteratively process children
    head = 0
    while head < len(queue):
         parent_id = queue[head]
         head += 1
         
         parent_node = comments_processed.get(parent_id)
         if not parent_node: continue # Should not happen if in queue

         # Find children of this parent (use t1_ prefix for lookup)
         parent_lookup_key = f"t1_{parent_id}"
         child_ids = parent_to_child_ids.get(parent_lookup_key, [])
         
         # Sort children by creation time before adding
         sorted_child_ids = sorted(
              [cid for cid in child_ids if cid in comments_processed], # Ensure child exists
              key=lambda cid: comment_data_map[cid]['created_utc']
         )

         for child_id in sorted_child_ids:
              if child_id not in processed_for_tree:
                   child_node = comments_processed[child_id]
                   parent_node["replies"].append(child_node)
                   processed_for_tree.add(child_id)
                   queue.append(child_id) # Add child to queue for its children processing
                   
    # Handle comments whose parents were not in the relevant set (orphans relative to this structure)
    # These might not have been added to the final_nested_structure yet.
    for cid in sorted_ids:
         if cid not in processed_for_tree:
             # This comment is relevant, but its parent wasn't, or it wasn't reachable from roots.
             # Add it as a top-level item.
             final_nested_structure.append(comments_processed[cid])
             processed_for_tree.add(cid) # Mark as added

    # Sort the final top-level list by original timestamp again?
    # The structure is built, order within replies is chronological.
    # Let's keep the order derived from root identification and BFS/DFS traversal for now.

    return final_nested_structure


def _format_context(submission: pd.Series, nested_thread: List[Dict[str, Any]]) -> str:
    """Formats the submission and nested comment thread into a YAML string."""
    data = {
        # Use .get() with defaults for safety
        "title": submission.get("title", "N/A"),
        # Check 'is_self' field from submission data
        "submission_body": submission.get("selftext", "N/A") if submission.get("is_self", False) else "[Link Post]", 
        "replies": nested_thread
    }
    try:
        # Use safe_dump, allow unicode, don't use aliases/anchors, standard indent
        return yaml.safe_dump(
            data, 
            allow_unicode=True, 
            default_flow_style=False, 
            indent=2, # Use 2 spaces for YAML indent
            sort_keys=False # Preserve insertion order where possible
        )
    except yaml.YAMLError as e:
        # Fallback or logging if YAML formatting fails
        print(f"Warning: YAML formatting failed for submission {submission.get('id', 'UNKNOWN')}: {e}")
        # Return a basic string representation as fallback
        return f"Title: {data['title']}\nSubmission Body: {data['submission_body']}\nReplies: {str(data['replies'])}"


def process_submission_group(
    group: pd.DataFrame, 
    target_user: str, 
    # submissions_lookup: Dict[str, pd.Series] # Removed lookup
) -> Optional[pd.DataFrame]:
    """
    Processes comments for a single submission (group) to find user comments 
    and build the minimal conversation context leading to them. 
    Assumes the input group DataFrame contains merged submission data 
    (title, selftext, is_self) associated with the comments.

    Args:
        group: Pandas DataFrame containing all comments for a single submission_id, 
               merged with submission data. Expected columns include comment fields 
               ('id', 'author', 'link_id', 'parent_id', 'body', 'created_utc') and 
               submission fields ('title', 'selftext', 'is_self').
        target_user: The username to filter comments for.
        # submissions_lookup: Removed.

    Returns:
        A Pandas DataFrame with a single row containing the formatted context 
        and user comment IDs for this submission, or None if the user didn't comment 
        or required data is missing.
    """
    # Filter for comments by the target user within this submission group
    user_comments = group[group['author'] == target_user]
    if user_comments.empty:
        # User did not comment in this submission
        return None

    # Get the submission ID (link_id) associated with this group
    # Assuming link_id is consistent within the group
    # Also extract submission details from the first row (should be identical for all rows in group)
    first_row = group.iloc[0]
    submission_id_full = first_row.get('link_id') # Get link_id from data
    
    if not isinstance(submission_id_full, str) or not submission_id_full.startswith("t3_"):
        # Invalid or missing link_id format
        print(f"Warning: Invalid link_id found in group: {submission_id_full}")
        return None 
        
    try:
        submission_id_short = submission_id_full.split("_", 1)[1]
    except IndexError:
        print(f"Warning: Could not extract short ID from link_id: {submission_id_full}")
        return None

    # Extract submission data directly from the first row of the group
    # Use .get() for safety in case merge failed or columns are missing
    submission_data = pd.Series({
        'title': first_row.get('title', 'N/A'),
        'selftext': first_row.get('selftext', ''), # Default to empty string if missing
        'is_self': first_row.get('is_self', False) # Default to False if missing
    })
    
    # Check if submission data seems valid (at least title is not N/A)
    # This handles cases where the left merge might not have found a matching submission
    if submission_data['title'] == 'N/A' and pd.isna(first_row.get('title')): 
         print(f"Warning: Submission data (title) appears missing for group {submission_id_short}. Skipping.")
         return None

    # Create a map of comment_id -> comment_row for efficient lookup within this group
    # Drop duplicates just in case, keeping the first occurrence
    # Use 'id_x' which is the comment ID after the merge
    comment_map = group.drop_duplicates(subset=['id_x']).set_index('id_x', drop=False) 
    
    # Use 'id_x' to get user comment IDs
    user_comment_ids_in_submission = list(user_comments['id_x'].unique())
    user_comment_ids_set = set(user_comment_ids_in_submission) # Use a set for the function

    # Find all direct ancestors for *all* user comments in one pass
    # print(f"DEBUG: Finding ancestors for {len(user_comment_ids_set)} user comments in submission {submission_id_short}")
    all_ancestor_ids = _get_all_ancestors_optimized(user_comment_ids_set, comment_map)
    # print(f"DEBUG: Found {len(all_ancestor_ids)} unique ancestors.")

    # The set of relevant comments includes the user's comments plus all their unique ancestors
    relevant_comment_ids = all_ancestor_ids.union(user_comment_ids_set)

    # Build the nested thread structure using only the relevant comments found within this group
    # Requires 'created_utc' column to be present in the group df
    if 'created_utc' not in comment_map.columns:
         print(f"Warning: 'created_utc' column missing in comment data for submission {submission_id_short}. Thread structure might be incorrect.")
         # Add a dummy column if missing to avoid error, but structure will be arbitrary
         comment_map['created_utc'] = 0 
         
    nested_thread = _build_nested_thread(relevant_comment_ids, comment_map)

    # Format the context string using submission data and the nested thread
    formatted_context = _format_context(submission_data, nested_thread)

    # Create the resulting DataFrame row
    result_df = pd.DataFrame({
        'submission_id': [submission_id_short],
        'formatted_context': [formatted_context],
        # Store the list of the target user's comment IDs for this submission
        'user_comment_ids': [user_comment_ids_in_submission] 
    })
    return result_df


def generate_user_context(
    user_id: str, 
    ddf_comments: dd.DataFrame, 
    ddf_submissions: dd.DataFrame
) -> dd.DataFrame:
    """
    Generates a Dask DataFrame containing the conversation context for each submission 
    a specific user commented on.

    Args:
        user_id: The Reddit username (author) to generate context for.
        ddf_comments: Dask DataFrame of comments. 
                      Requires columns: 'id', 'author', 'link_id', 'parent_id', 'body', 'created_utc'.
        ddf_submissions: Dask DataFrame of submissions.
                         Requires columns: 'id', 'title', 'selftext', 'is_self'.

    Returns:
        A Dask DataFrame with columns:
        - submission_id: The ID of the submission (without 't3_' prefix).
        - formatted_context: A string (YAML format) representing the submission title, 
                             body, and the comment thread leading to the user's comments.
        - user_comment_ids: A list of comment IDs made by the target user in that submission.
                             
    Raises:
        ValueError: If required columns are missing in the input DataFrames.
    """
    
    # --- Input Validation ---
    required_comment_cols = ['id', 'author', 'link_id', 'parent_id', 'body', 'created_utc']
    missing_comment_cols = [col for col in required_comment_cols if col not in ddf_comments.columns]
    if missing_comment_cols:
         raise ValueError(f"ddf_comments is missing required columns: {missing_comment_cols}")

    required_submission_cols = ['id', 'title', 'selftext', 'is_self']
    missing_submission_cols = [col for col in required_submission_cols if col not in ddf_submissions.columns]
    if missing_submission_cols:
         raise ValueError(f"ddf_submissions is missing required columns: {missing_submission_cols}")
         
    # Select only necessary columns early to potentially reduce data shuffling
    ddf_comments = ddf_comments[required_comment_cols].copy() # Use copy to avoid SettingWithCopyWarning
    ddf_submissions = ddf_submissions[required_submission_cols].copy()

    # --- Prepare for Merge ---
    print("Preparing comments and submissions for merge...")
    # Ensure 'link_id' is usable and extract short submission ID
    ddf_comments = ddf_comments.dropna(subset=['link_id'])
    ddf_comments = ddf_comments[ddf_comments['link_id'].str.startswith('t3_')]
    # Create 'submission_id_short' column for merging
    ddf_comments['submission_id_short'] = ddf_comments['link_id'].str.slice(start=3)
    
    # Ensure submission 'id' is string type for consistent merging
    ddf_submissions = ddf_submissions.astype({'id': 'string'}) 
    ddf_comments = ddf_comments.astype({'submission_id_short': 'string'}) # Ensure merge keys match type
    
    # --- Perform Dask Merge ---
    # Merge comments with submission data. Use left merge to keep all comments.
    print("Merging comments with submissions (constructing Dask graph)...")
    # Rename submission 'id' to avoid conflict if needed, although merge handles suffixes
    # ddf_submissions = ddf_submissions.rename(columns={'id': 'submission_id'})
    ddf_merged = dd.merge(
        ddf_comments, 
        ddf_submissions, 
        left_on='submission_id_short', 
        right_on='id', 
        how='left'
    )
    print("Merge graph constructed.")

    # --- Define Output Metadata for Apply --- 
    # This meta reflects the output of process_submission_group
    meta_apply = pd.DataFrame({
        'submission_id': pd.Series(dtype='string'), 
        'formatted_context': pd.Series(dtype='string'),
        'user_comment_ids': pd.Series(dtype='object') # Keep as object for lists
    }).set_index('submission_id') # Index helps Dask understand output structure for apply

    # --- Filter and Group Merged Data ---
    # Now group the *merged* dataframe by the original link_id
    print(f"Grouping merged data by 'link_id' for user '{user_id}'...")
    grouped_merged_comments = ddf_merged.groupby('link_id')

    # --- Apply Context Generation Logic --- 
    print("Applying context generation logic to merged groups (constructing Dask graph)...")
    # Apply the processing function to each group (submission)
    # Note: submissions_lookup is no longer passed
    user_context_ddf = grouped_merged_comments.apply(
        process_submission_group,
        target_user=user_id,
        # submissions_lookup=submissions_lookup, # REMOVED
        meta=meta_apply # Provide output metadata for the apply function
    ).reset_index(drop=True) # Reset index to get submission_id back as a column

    print("Dask graph for context generation constructed successfully.")
    
    # Final check on columns - should match meta_apply
    # print(f"Final Dask DataFrame columns: {user_context_ddf.columns}")
    
    # The result is a Dask DataFrame, computation is lazy.
    return user_context_ddf

# --- Example Usage --- 
# (Requires Dask cluster/client setup and actual Dask DataFrames)
# if __name__ == "__main__":
#     from dask.distributed import Client
#     import dask.dataframe as dd
#     import time

#     # --- Setup Dask Client ---
#     # client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB') 
#     # print(f"Dask dashboard link: {client.dashboard_link}")

#     # --- Load Sample Data (Replace with your actual data loading) ---
#     # Create dummy data for demonstration
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
    
#     comments_pdf = pd.DataFrame(comments_data)
#     submissions_pdf = pd.DataFrame(submissions_data)
    
#     # Create Dask DataFrames (e.g., 2 partitions)
#     comments_ddf = dd.from_pandas(comments_pdf, npartitions=2)
#     submissions_ddf = dd.from_pandas(submissions_pdf, npartitions=1)
#     print("Sample DataFrames created.")

#     # --- Define Target User ---
#     target_user = "user_a" 
    
#     # --- Generate Context ---
#     start_time = time.time()
#     print(f"Generating context for user: {target_user}")
#     try:
#         context_ddf = generate_user_context(target_user, comments_ddf, submissions_ddf)
        
#         # --- Compute and Display Results ---
#         print("Computing the result...")
#         result_pdf = context_ddf.compute()
#         end_time = time.time()
        
#         print("\n--- Generated Context ---")
#         if not result_pdf.empty:
#             for index, row in result_pdf.iterrows():
#                 print(f"Submission ID: {row['submission_id']}")
#                 print(f"User Comment IDs: {row['user_comment_ids']}")
#                 print(f"Formatted Context:\n{row['formatted_context']}")
#                 print("-" * 20)
#         else:
#             print(f"No comments found for user '{target_user}' or context could not be generated.")
            
#         print(f"Total execution time: {end_time - start_time:.2f} seconds")
        
#         # --- Optional: Save Output ---
#         # print("Saving results to Parquet...")
#         # context_ddf.to_parquet("user_a_context_output") 
#         # print("Results saved.")

#     except ValueError as e:
#         print(f"Execution Error: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         # In a real scenario, log the full traceback
#     finally:
#         # --- Shutdown Dask Client ---
#         # print("Shutting down Dask client...")
#         # client.close()
#         pass 