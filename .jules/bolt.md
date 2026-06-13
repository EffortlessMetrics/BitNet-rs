## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2026-03-05 - Optimization of Floating-Point Division by a Power
**Learning:** Division is an inherently more expensive operation than multiplication, especially when applied iteratively inside the hot path. In repetition penalty calculations involving a base penalty and an exponent, using `logits /= base.powi(count)` triggers an expensive division per token. Because the base is constant for the penalty configuration, its inverse can be pre-calculated outside the loop.
**Action:** When a calculation involves repeated division by a constant or an exponentiated constant, pre-calculate the inverse (`inv = 1.0 / base`) outside the loop and use multiplication by the exponentiated inverse (`logits *= inv.powi(count)`) to convert O(log N) division to O(log N) multiplication, safely eliminating the division overhead.
