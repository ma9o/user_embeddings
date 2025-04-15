# Implementation Considerations

This document outlines various design choices, trade-offs, and potential future considerations related to the project's implementation.

## Parallelism Strategy for Multi-Entity Processing

**Context:** When processing data for multiple independent entities (e.g., users, submissions, groups), a common pattern involves applying a core Dask computation to each entity.

**Approach 1: Sequential Orchestration of Parallel Tasks:**
Iterate through the entities using a standard Python loop. Within the loop, trigger the Dask computation graph for the current entity (e.g., using `.compute()` or `.persist().compute()`).
*   **Parallelism:** Dask effectively parallelizes the computation *within* each entity's task graph, distributing work across the cluster.
*   **Orchestration:** The Python loop manages the sequence, processing one entity's parallel workload after the previous one completes.

**Approach 2: Parallel Orchestration (`dask.delayed`):**
Wrap the per-entity processing logic (including triggering Dask computation and handling results/side effects) in a Python function decorated with `@dask.delayed`. Create a list of these delayed tasks for all entities and execute them concurrently using `dask.compute()`.
*   **Parallelism:** Dask parallelizes both the computation *within* each entity's task graph AND the *execution* of potentially overlapping task graphs for different entities.
*   **Orchestration:** Dask's scheduler manages the concurrent execution of the delayed tasks.

**Decision Rationale:**
While Approach 2 (`dask.delayed`) offers potentially higher concurrency by overlapping the processing of different entities, its benefits depend on cluster saturation. If the Dask computation for a single entity is complex enough to fully utilize the cluster resources, Approach 1 (Sequential Orchestration) is often sufficient and preferable due to:
*   **Simplicity:** Easier to implement, read, and debug.
*   **Resource Management:** Avoids potentially overwhelming the scheduler or memory with too many concurrent large graphs.
*   **Side Effects:** Simpler to manage operations like saving results for each entity.

Approach 1 is generally the recommended starting point. Consider Approach 2 if profiling reveals that the cluster is significantly underutilized during the sequential processing of entities.

---
*(Add more sections as the project evolves)* 