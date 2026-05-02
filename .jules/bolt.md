## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.
## 2026-05-15 - Fast Pow for Repetition Penalty
**Learning:** Using `f32::powi(count as i32)` in a hot loop for repetition penalty calculation is slow because `powi` involves complex branching and edge cases. In text generation, `count` is usually very small (1-5).
**Action:** Replace `powi` with a simple iterative loop (`let mut penalty = 1.0; for _ in 0..count { penalty *= base; }`) when applying count-based penalties. Initializing to `1.0` and looping `0..count` ensures the mathematical correctness even if `count` is `0`, preventing unintended penalization of all tokens.
