## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2026-04-10 - Sparse Vector Allocation Optimization
**Learning:** In probability filtering layers (like Top-P and Typical), explicitly pre-counting elements to pre-allocate vector capacity `Vec::with_capacity(count)` requires a full O(N) pass over the logits. Since distributions are often extremely sparse (e.g., heavily masked by prior Top-K or Min-P steps), the cost of this extra O(N) scan drastically outweighs the minor cost of dynamic allocations in a single pass.
**Action:** When filtering sparse probability distributions, avoid two-pass "count then allocate" patterns. Use a single pass with dynamic `Vec` allocation to eliminate O(N) iteration overhead.
