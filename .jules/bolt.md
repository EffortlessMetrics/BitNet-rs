## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2025-01-28 - Optimize typical and min_p logits filtering
**Learning:** In hot paths like logits filtering, separating mapping and entropy calculations into separate passes creates redundant overhead for expensive operations like `.ln()` and intermediate array allocations. Also, iterating and writing to raw, unfiltered sparse probability arrays without conditional checks like `p > 0.0` leads to unnecessary cache invalidations.
**Action:** When working on array mappings or aggregations over probability structures, compute all aggregated metrics (like entropy) and related mapped outputs (like surprises) within a single pass. For sparse collections, condition sparse mutations on non-zero thresholds instead of overwriting elements.
