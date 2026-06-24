## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2024-05-18 - Eliminating Division by Power Overhead
**Learning:** In conditional hot-path logic involving divisions by a constant power, unconditionally computing the power (e.g., `.powi(count)`) before the branch introduces unnecessary overhead. Furthermore, executing divisions inside the loop is slower than multiplying by a pre-calculated inverse.
**Action:** Pre-calculate the inverse of the base outside the loop (`inv_base = 1.0 / base`), compute the power lazily inside its respective branch, and use multiplication (`*=`) instead of division to safely eliminate the division overhead.
