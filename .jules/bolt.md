## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2026-03-05 - Sparse Probability Optimization in Sampling Filters
**Learning:** In top-k and top-p sampling logic, iterating and allocating memory proportional to the full vocabulary size (e.g. `vec![false; logits.len()]` or a `sanitized` copy vector) is extremely wasteful because many logits are already `f32::NEG_INFINITY` or `NaN` from prior sampling stages.
**Action:** When filtering logit or probability slices, pre-filter to only keep valid/finite values (`!is_nan() && > f32::NEG_INFINITY`) *before* sorting or allocating helper arrays, effectively reducing the active working set from O(N) to O(k). Mask the original slice directly rather than keeping boolean bitmasks.
