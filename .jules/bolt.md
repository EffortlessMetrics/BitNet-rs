## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.
## 2026-05-06 - Costly power computation in repetition penalty
**Learning:** `f32::powi` performs additional checks and branches when used for simple integer powers, creating unnecessary latency overhead in hot loops. In `RepetitionPenaltyConfig::apply`, computing multiplicative count penalties iteratively is more efficient. Furthermore, division inside the loop can be replaced by taking the inverse `1.0 / penalty` and multiplying.
**Action:** When computing integer powers (especially dynamically based on token counts) in sampling logic, use simple multiplication loops instead of `powi`, and avoid division in hot paths by precomputing reciprocals.
