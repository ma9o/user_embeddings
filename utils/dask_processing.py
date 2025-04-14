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
    submissions_lookup: Dict[str, pd.Series]
) -> Optional[pd.DataFrame]:
    """
    Processes comments for a single submission (group) to find user comments 
    and build the minimal conversation context leading to them.

    Args:
        group: Pandas DataFrame containing all comments for a single submission_id (link_id).
        target_user: The username to filter comments for.
        submissions_lookup: A pre-computed dictionary mapping submission_id (without prefix) 
                          to submission data (as Pandas Series).

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
    submission_id_full = group['link_id'].iloc[0] 
    if not isinstance(submission_id_full, str) or not submission_id_full.startswith("t3_"):
        # Invalid or missing link_id format
        print(f"Warning: Invalid link_id found in group: {submission_id_full}")
        return None 
        
    try:
        submission_id_short = submission_id_full.split("_", 1)[1]
    except IndexError:
        print(f"Warning: Could not extract short ID from link_id: {submission_id_full}")
        return None

    # Look up submission details using the pre-computed map
    submission_data = submissions_lookup.get(submission_id_short)
    if submission_data is None:
        # Submission data not found (might be in a different partition or missing)
        # As per requirement, skip this group for now.
        # print(f"Debug: Submission data not found for ID: {submission_id_short}")
        return None 

    # Create a map of comment_id -> comment_row for efficient lookup within this group
    # Drop duplicates just in case, keeping the first occurrence
    comment_map = group.drop_duplicates(subset=['id']).set_index('id', drop=False) 
    
    user_comment_ids_in_submission = list(user_comments['id'].unique())

    # Find all direct ancestors for *all* of the user's comments in this submission
    all_ancestor_ids = set()
    for comment_id in user_comment_ids_in_submission:
         if comment_id in comment_map.index: # Ensure the user comment itself exists in the map
             ancestors = _get_ancestors(comment_id, comment_map)
             all_ancestor_ids.update(ancestors)
         else:
             print(f"Warning: User comment {comment_id} not found in comment_map for submission {submission_id_short}")


    # The set of relevant comments includes the user's comments plus all their unique ancestors
    relevant_comment_ids = all_ancestor_ids.union(set(user_comment_ids_in_submission))

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
         
    # Select only necessary columns to potentially reduce data shuffling
    ddf_comments = ddf_comments[required_comment_cols]
    ddf_submissions = ddf_submissions[required_submission_cols]

    # --- Prepare Submissions Lookup ---
    # This computes the submissions DataFrame and collects it to the client, then broadcasts.
    # WARNING: This can consume significant memory on the client and workers if ddf_submissions is large.
    # For larger datasets, consider Dask joins (e.g., ddf_comments.merge(ddf_submissions_indexed)) 
    # or broadcasting the lookup table more efficiently if using Dask Distributed.
    print("Computing submissions lookup table (may take time and memory)...")
    try:
        # Ensure submission 'id' is string type for consistent indexing
        ddf_submissions = ddf_submissions.astype({'id': 'string'}) 
        # Compute the submission data into a Pandas DataFrame
        submissions_pdf = ddf_submissions.set_index('id', sorted=True).compute() 
        # Convert to dictionary of Pandas Series for faster lookups within apply
        submissions_lookup = {index: row for index, row in submissions_pdf.iterrows()}
        print(f"Submissions lookup table created with {len(submissions_lookup)} entries.")
        del submissions_pdf # Free memory if possible
    except Exception as e:
        print(f"Error computing submissions lookup table: {e}")
        raise

    # --- Define Output Metadata ---
    # Metadata for the output of the groupby().apply() operation
    meta = pd.DataFrame({
        'submission_id': pd.Series(dtype='string'), # Use Arrow string type
        'formatted_context': pd.Series(dtype='string'),
        'user_comment_ids': pd.Series(dtype='object') # Keep as object for lists
    }).set_index('submission_id') # Set index temporarily for apply, will reset later if needed

    # --- Filter and Group Comments ---
    # Ensure 'link_id' is usable. Drop rows where it's null or not in the expected format.
    print("Filtering comments for valid link_id...")
    ddf_comments = ddf_comments.dropna(subset=['link_id'])
    ddf_comments = ddf_comments[ddf_comments['link_id'].str.startswith('t3_')]
    
    # Optional: Filter comments by target_user *before* grouping if the user is very common.
    # However, grouping all comments is necessary to reconstruct the full ancestor thread.
    # ddf_comments_filtered = ddf_comments[ddf_comments['author'] == user_id] # Might filter too much

    print(f"Grouping comments by submission ('link_id') for user '{user_id}'...")
    # Group all comments by the submission they belong to.
    grouped_comments = ddf_comments.groupby('link_id')

    # --- Apply Context Generation Logic ---
    print("Applying context generation logic to comment groups (constructing Dask graph)...")
    # Apply the processing function to each group (each submission)
    user_context_ddf = grouped_comments.apply(
        process_submission_group,
        target_user=user_id,
        submissions_lookup=submissions_lookup, # Broadcast the lookup map
        meta=meta # Provide output metadata
    ).reset_index(drop=True) # Reset index to get submission_id back as a column

    print("Dask graph for context generation constructed successfully.")
    
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