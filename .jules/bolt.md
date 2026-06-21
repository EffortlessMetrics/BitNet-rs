## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2026-03-05 - Repetition Penalty Power Calculation Optimization
**Learning:** Calculating `powi` unconditionally before an `if/else` block introduces significant overhead by performing work for branches that aren't taken. Furthermore, executing division (e.g., `logit /= penalty`) in a hot loop is computationally expensive and can be safely eliminated by calculating the inverse first (`1.0 / penalty`) and multiplying with the inverse `inv_penalty.powi(count)`.
**Action:** When working on math operations in conditional branches inside hot loops, pre-calculate values if possible outside the loop and lazily calculate values inside the branches that actually use them to avoid wasted cycles and unnecessary expensive operations like division.
