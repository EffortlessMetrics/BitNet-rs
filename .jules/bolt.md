## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2026-03-09 - Fusing computations in sorting operations
**Learning:** When sorting a sparse collection based on a complex mathematical computation (e.g. `surprise = -p.ln()` and `deviation = (surprise - entropy).abs()`), calculating those transcendental functions on-the-fly during a `.sort_by` comparator is disastrous for performance. They evaluate O(N log N) times.
**Action:** When implementing or optimizing custom sorts over mathematical transformations, perform an initial O(N) pass to pre-calculate the values and fuse them into the element tuples/structs (e.g. caching the deviation value in-place) before sorting the collection.
