## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2026-03-12 - Token History Sliding Window
**Learning:** Naively using `clear()` on a token history buffer (e.g., `if len > 1000 { clear() }`) causes abrupt resets in repetition penalty semantics, leading to unexpected behavioral changes in long generation sequences.
**Action:** When bounding memory growth for token histories, always use a sliding window approach (e.g., allowing the buffer to reach 2X capacity and draining the oldest half) to preserve recent context while maintaining amortized O(1) performance.
