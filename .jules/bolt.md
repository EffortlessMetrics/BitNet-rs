## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2026-03-22 - Fused Sorting Operations
**Learning:** When optimizing operations that involve sorting (e.g., `apply_typical` in logits filtering), placing expensive computations like transcendental functions (`ln()`) inside the mapping pass results in redundant work if those calculated values aren't fully utilized or if they're calculated on elements that can be mutated in-place later. Fusing passes (e.g. calculating entropy and storing the intermediate `.ln()` value for later modification instead of recomputing it) eliminates intermediate allocations and expensive redundant math ops.
**Action:** Identify and fuse chained iterator operations that perform redundant passes or compute expensive transcendental math ops like logarithms twice. Calculate the intermediate value in the first pass and mutate the collection directly.
