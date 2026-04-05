## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2026-03-05 - Avoid .powi() and Division in Repetition Penalty
**Learning:** Computing repetition penalties using `f32::powi()` and float division (`/= penalty`) inside the token counting hot loop is slower than simple iterative multiplication with a pre-calculated inverse penalty (`*= inv_p`). Tests show iterative multiplication is 10-15% faster for typical counts (1-5).
**Action:** When applying multiplicative penalties per token occurrence, replace `.powi()` and division with an iterative multiplication of pre-calculated inverse values to avoid instruction overhead and pipeline stalls.
