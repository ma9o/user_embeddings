# Learning Universal User Embeddings via Teacher-Guided Synthesis

## Overview

This project explores a novel approach to learning comprehensive user embeddings applicable across various downstream tasks, including item recommendation and user similarity analysis. Traditional methods often specialize (e.g., Collaborative Filtering for recommendation) or struggle with the scale and noise of raw user activity logs (e.g., directly modeling long sequences).

Our approach draws inspiration from knowledge distillation techniques (like Gecko) and self-supervised learning (like SoMeR), but introduces a unique data synthesis step guided by a powerful Large Language Model (LLM) acting as a 'teacher'. The core idea is to train a 'student' embedding model not on raw activity logs, but on semantically meaningful, context-aware examples synthesized by the teacher LLM. 

## Core Assumptions

See [assumptions.md](docs/assumptions.md)

## Approach

The process involves two main stages:

1.  **Teacher-Guided Data Synthesis:** Generating a structured training dataset from raw user histories using an LLM teacher.
2.  **Student Model Training:** Training the final user/item embedding model (the 'student') on the synthesized dataset.

### Step 1: Teacher-Guided Data Synthesis 

This stage transforms raw user activity histories (`H`) into structured textual training triplets `(Context_History_Representation, Positive_Summary, Negative_Summary)`. It leverages a Large Language Model (LLM) as an intelligent "teacher". The pipeline consists of two main phases: Positive Summary Extraction and Negative Summary Selection.

**Phase 1: Positive Summary Extraction (Iterative)**

This phase aims to identify and summarize distinct clusters of related activities within each user's history.

1.  **Initialization:** Start with a user's activity history segment (e.g., recent N activities or a sliding window).
2.  **Pivot Identification:** Identify the most recent activity in the current segment as the 'pivot'.
3.  **LLM Prompting (Cluster & Summarize):** Prompt the LLM teacher with the history segment. Instruct it to:
    *   Find activities strongly related to the pivot (semantically or contextually) within the segment, forming a 'cluster'.
    *   Synthesize a concise, descriptive textual summary (`I_pos_summary`) capturing the core theme/interest/opinion of the cluster.
    *   Return the unique indices of all activities within the identified cluster.
4.  **Store Positive Data:** If a cluster is found, store the tuple `(Full_History_Segment, Cluster_Indices, I_pos_summary)`.
5.  **Iteration:** Remove the activities corresponding to the `Cluster_Indices` from the history segment. Repeat steps 2-4 on the remaining segment until no more meaningful clusters can be identified or a desired number of positives per user is extracted.
6.  **Output:** A dataset containing `(User_ID, Full_History_Segment, Cluster_Indices, I_pos_summary)` for all identified positive instances across all users. This also implicitly defines the `Context_History (H_context)` for each positive instance (as `Full_History_Segment` minus activities at `Cluster_Indices`). A global pool of all generated `I_pos_summary` values is also maintained.

**Phase 2: Negative Summary Selection (Per Positive Instance)**

This phase generates a corresponding negative summary (`I_neg_summary`) for each positive summary (`I_pos_summary`) identified in Phase 1, using an LLM-guided reranking approach inspired by Gecko.

1.  **Input:** For each positive instance `(User_ID, H_context, I_pos_summary)`:
    *   The specific `User_ID`.
    *   The user's context history representation (`H_context`, which might be the remaining text or a summary of it).
    *   The target positive summary (`I_pos_summary`).
2.  **Candidate Negative Sampling:** Randomly sample `N` candidate negative summaries (`I_neg_cand_1` to `I_neg_cand_N`) from the global pool of positive summaries generated in Phase 1, ensuring they are not identical to the current `I_pos_summary` and ideally filtering out other known positives for the same `User_ID`.
3.  **LLM Prompting (Rerank Candidates):** Prompt the *same* LLM teacher. Provide it with:
    *   The user's context (`H_context`).
    *   The true positive summary (`I_pos_summary`).
    *   The `N` sampled candidate negative summaries (`I_neg_cand_1`...`I_neg_cand_N`).
    *   Instruct the LLM to **rank** all `N+1` summaries based on their likelihood or relevance *specifically for the provided user context `H_context`*.
4.  **Hard Negative Selection (`I_neg_summary`):** Select the negative summary (`I_neg_cand_k`) that the LLM ranked highest among the `N` candidates (i.e., the one deemed most plausible or confusable with `I_pos` for this user context). This becomes the final `I_neg_summary`. *(Alternative: A mix of hard and easier negatives could be selected based on the ranking distribution).*
5.  **Final Triplet Formation:** Create the final textual training triplet: `(H_context_representation, I_pos_summary, I_neg_summary)`. *[Note: `H_context_representation` might be the text, a summary, or omitted depending on student model needs.]*
6.  **Output:** A large dataset of these final textual triplets ready for training the student embedding model.

**(Alternative for Phase 2):** An alternative is to prompt the LLM in step 3 to *generate* synthetic hard negatives directly based on `H_context` and `I_pos`, instead of reranking sampled ones. This avoids the need for the global pool but increases reliance on the LLM's generative capabilities and risks synthetic bias. The reranking approach is generally preferred for grounding in real data while still using LLM judgment for hardness.

### Step 2: Benchmarking Against Standard Q&A Embedders

Before proceeding extensively with training the student model, it is **essential** to validate that this data synthesis process provides a meaningful learning challenge beyond simple semantic retrieval.

**Objective:** Verify that the relationship between the `Context_History (H_context)` and the `Positive_Summary (I_pos_summary)` is complex enough that a standard, pre-trained Q&A or semantic retrieval embedding model cannot easily identify `I_pos_summary` as the correct "answer" given `H_context` as the "query" when compared against the `Negative_Summary (I_neg_summary)` and potentially other distractors.

**Procedure:**

1.  **Select Validation Set:** Take a representative subset of the generated `(H_context, I_pos_summary, I_neg_summary)` triplets.
2.  **Choose Baseline Embedder:** Select a strong, publicly available text embedding model fine-tuned for semantic similarity or retrieval (e.g., `text-embedding-large-exp-03-07`).
3.  **Encode:** Use the baseline embedder to generate embeddings for:
    *   The `H_context` representation (treating it as a query/context document).
    *   The `I_pos_summary` (treating it as the target document).
    *   The `I_neg_summary` (treating it as a distractor document).
    *   (Optional: Include more distractors sampled from the global `I_pos` pool).
4.  **Calculate Similarity:** For each triplet, compute the similarity score between the `H_context` embedding and the embeddings of `I_pos_summary`, `I_neg_summary`, and any other distractors.
5.  **Evaluate Retrieval:** Measure standard retrieval metrics (e.g., Recall@1, Mean Reciprocal Rank - MRR). Does the `I_pos_summary` consistently achieve the highest similarity score compared to `I_neg_summary` and other distractors?
6.  **Analysis:**
    *   **If Baseline Performs Poorly:** This is **good news**. It indicates that the relationship captured by our synthesis process (linking context to specific summarized actions/interests) requires more nuanced understanding than simple semantic matching provided by standard embedders. The approach is adding value.
    *   **If Baseline Performs Very Well:** This is a **red flag**. It suggests that the `I_pos_summary` might be too easily identifiable from the `H_context` based on simple keyword overlap or semantic closeness. The task might be too easy, or the negative selection might not be effective enough. The complex teacher synthesis might not be creating a sufficiently challenging learning signal beyond what standard semantic embedders already capture. In this case, reconsider the teacher's summarization strategy (is it too literal?) or the negative selection method (need harder negatives?).

This benchmarking step is critical to ensure the effort invested in the teacher-guided synthesis translates into a genuinely advanced representation capability in the student model.

### Step 3: Full training run

TODO

---