## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2026-03-06 - Fusing Loops and Avoiding Transcendental Re-computation in Logits Filters
**Learning:** When calculating values that depend on aggregates (like entropy) in logits filters, first generating a filtered list and then performing multiple functional passes (`map`, `sum`, `collect`) causes redundant iteration, multiple intermediate `Vec` allocations, and re-computation of expensive transcendental functions like `ln()`.
**Action:** In mathematical hot paths like logits filters, fuse the calculation of aggregates (e.g. entropy) and intermediate values (e.g. surprise) into a single initial pass that simultaneously populates the final array structure. Use an in-place mutation pass for the final deviation calculation to avoid redundant mathematical operations and intermediate collections.
