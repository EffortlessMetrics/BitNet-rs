## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2026-03-05 - Fusing Mathematical Passes in Logit Filters
**Learning:** When applying complex mathematical filters like typical sampling (`apply_typical`), intermediate iterator chains (e.g. `.filter().map().collect()`) allocate unnecessary vectors and force redundant iterations over the array. Fusing operations like probability filtering, entropy calculation, and deviation calculation into a single loop, and then mutating the collected array in-place, drastically reduces allocations and provides a significant (~15-20%) speedup without sacrificing readability.
**Action:** When filtering or transforming sparse probability arrays, manually fuse mathematical operations into a single pass and mutate arrays in-place where possible, instead of chaining iterator methods that implicitly allocate.
