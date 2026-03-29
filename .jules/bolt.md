## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2026-03-07 - Fusing Logits Operations
**Learning:** Mathematical operations like `.ln()` are expensive. In operations like `apply_typical` filtering, computing the initial `entropy` sum and then re-computing `.ln()` a second time for each probability in the deviation mapping results in severe duplicate computational overhead. We also found that separating the `.filter(|p| p > 0.0).collect()` and the map causes extra iterations and allocations.
**Action:** Fuse iteration loops and cache expensive mathematical operations. For probability slices, iterate once to compute sums/metrics, cache the intermediate values (e.g. `log_p`), and then use the cached values in the second pass. This avoids intermediate collections and redundant math operations.
