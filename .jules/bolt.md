## 2024-03-24 - Sampling Optimization: Tensor Overhead vs. Native Vector Operations

**Learning:**
For LLM inference sampling (specifically with vocabularies ~32k), avoiding repeated `Tensor` allocations and kernel launches is critical for performance.
The original implementation converted `CandleTensor` to `Vec<f32>` and back multiple times (for Repetition Penalty, Top-K, Top-P), causing significant overhead (900µs per sample).
Refactoring the pipeline to extract logits once and perform all sampling logic (Softmax, Top-K, Top-P) on a single `Vec<f32>` reduced sampling time by ~40% (to ~536µs) while maintaining correctness.

**Action:**
When optimizing sequential, scalar-heavy logic in inference pipelines (like sampling), prefer extracting data to native Rust types (`Vec`, slices) early and processing on CPU, rather than chaining multiple small tensor operations that incur allocation and launch overhead. Verify performance with micro-benchmarks.
