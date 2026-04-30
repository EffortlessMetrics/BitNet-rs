## 2026-02-27 - Sparse Top-P Optimization
**Learning:** Top-P sampling often follows Top-K or Softmax, resulting in many zero or near-zero probabilities. Sorting the entire vocabulary (O(N log N)) is wasteful. Filtering out zero probabilities first reduces complexity to O(k log k).
**Action:** When optimizing probability operations, always check if the distribution is sparse (e.g. from Top-K) and leverage it to skip processing zeros.

## 2026-03-05 - Hot Loop Allocations in Token Sampling
**Learning:** Allocating memory in the hot path of LLM token generation (e.g., `logits.to_vec()` or creating `HashMap`s per token) significantly degrades performance due to repeated allocation overhead of vocabulary-sized vectors (often 128K+ elements). Additionally, mathematically equivalent iterative multiplication (`logit *= inv_penalty`) can replace `HashMap` counting and `.powi(count)`, completely eliminating O(N) memory allocations per token.
**Action:** When working on generation loops, use buffer pooling (e.g. storing a `Vec` in the generator state and using `std::mem::take` to bypass borrow checker limitations) and avoid `HashMap` allocations for simple counting if an iterative scalar approach is mathematically equivalent.

## 2026-03-06 - Replacing Intermediate Tensors with In-Place Buffers
**Learning:** During inference token generation, mapping logits to `CandleTensor` just to call `.flatten_all()?.to_vec1::<f32>()?` causes repeated O(N) vocabulary-size allocations. Using `std::mem::take` to swap in a persistent `Vec<f32>` buffer for pure-Rust operations (like repetition penalties and logit filtering) effectively avoids these allocations.
**Action:** When working on generation loops involving Candle, extract slices into a pre-allocated, reused `Vec` and utilize in-place mutation methods from specialized logic crates (like `bitnet_logits`) instead of leaning on tensor operations that trigger internal allocations.
