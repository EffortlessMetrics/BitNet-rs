# Per-Position Logits Comparison

This document describes the per-position logits comparison functionality in the crossval crate.

## Overview

The per-position logits comparison feature allows you to compare logits between BitNet.rs (Rust) and bitnet.cpp (C++) implementations at each token position during generation. This is useful for:

1. **Debugging divergences**: Identifying the exact token position where outputs start to differ
2. **Quality assurance**: Ensuring numerical parity across implementations
3. **Performance analysis**: Understanding where computational differences emerge

## API

### Core Module: `logits_compare`

Located in `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs`

#### `LogitsDivergence` Struct

```rust
pub struct LogitsDivergence {
    /// First token position where logits diverged (None if all match)
    pub first_divergence_token: Option<usize>,

    /// Cosine similarity for each token position
    pub per_token_cosine_sim: Vec<f32>,

    /// L2 distance for each token position
    pub per_token_l2_dist: Vec<f32>,

    /// Maximum absolute difference across all positions and logits
    pub max_absolute_diff: f32,
}
```

#### `compare_per_position_logits` Function

```rust
pub fn compare_per_position_logits(
    rs_logits: &[Vec<f32>],
    cpp_logits: &[Vec<f32>],
) -> LogitsDivergence
```

**Parameters:**
- `rs_logits`: Logits from Rust implementation (outer vec = token positions, inner vec = vocabulary)
- `cpp_logits`: Logits from C++ implementation (same structure)

**Returns:**
- A `LogitsDivergence` struct with per-position metrics

**Tolerance:**
- Cosine similarity threshold: `1e-4` (matches existing parity threshold)
- Divergence is detected when `(1.0 - cosine_sim) > 1e-4`

## Usage Example

```rust
use bitnet_crossval::logits_compare::compare_per_position_logits;
use bitnet_inference::eval_logits_once;
use bitnet_sys::wrapper::Session as CppSession;

// Initialize C++ backend
bitnet_sys::wrapper::init_backend();
let _guard = scopeguard::guard((), |_| bitnet_sys::wrapper::free_backend());

// Create C++ session
let mut cpp_session = CppSession::load_deterministic("model.gguf")?;

// Tokenize prompt
let tokens = cpp_session.tokenize("The capital of France is")?;

// Generate multiple tokens and collect logits
let mut rust_all_logits = Vec::new();
let mut cpp_all_logits = Vec::new();

for step in 0..5 {
    // Get logits from both implementations
    let cpp_logits = cpp_session.eval_and_get_logits(&tokens, 0)?;
    let rust_logits = eval_logits_once("model.gguf", &tokens)?;

    rust_all_logits.push(rust_logits.clone());
    cpp_all_logits.push(cpp_logits.clone());

    // Sample and append next token
    let next_token = cpp_session.context.sample_greedy(&cpp_logits);
    tokens.push(next_token);
}

// Compare all positions
let divergence = compare_per_position_logits(&rust_all_logits, &cpp_all_logits);

// Check for divergence
if let Some(div_pos) = divergence.first_divergence_token {
    println!("Divergence detected at position {}", div_pos);
    println!("Cosine similarity: {:.6}", divergence.per_token_cosine_sim[div_pos]);
    println!("L2 distance: {:.6e}", divergence.per_token_l2_dist[div_pos]);
} else {
    println!("No divergence detected!");
}
```

## Tests

The implementation includes comprehensive tests:

### Unit Tests (in `logits_compare.rs`)

- `test_cosine_similarity_identical`: Validates cosine similarity calculation for identical vectors
- `test_cosine_similarity_orthogonal`: Validates orthogonal vectors produce 0.0 similarity
- `test_l2_distance_identical`: Validates L2 distance for identical vectors
- `test_l2_distance_simple`: Validates L2 distance calculation (3-4-5 triangle)
- `test_compare_per_position_logits_no_divergence`: Tests with matching logits
- `test_compare_per_position_logits_with_divergence`: Tests with divergent logits
- `test_compare_per_position_logits_size_mismatch`: Handles size mismatches gracefully
- `test_compare_per_position_logits_empty`: Handles empty inputs

Run with:
```bash
cargo test -p bitnet-crossval --lib --no-default-features --features crossval logits_compare
```

### Integration Tests (in `tests/per_position_logits.rs`)

**Note:** These tests require `BITNET_CPP_DIR` to be set and the C++ reference implementation to be available.

- `test_single_token_logits_parity`: Compares single-token generation
- `test_multi_token_generation_divergence`: Compares multi-token generation step-by-step
- `test_prefill_decode_logits_comparison`: Compares prefill vs decode phases
- `test_logits_compare_module`: Unit test for the comparison function (no FFI)

Run with:
```bash
export BITNET_CPP_DIR=$HOME/.cache/bitnet_cpp
export CROSSVAL_GGUF=/path/to/test/model.gguf
cargo test -p bitnet-crossval --test per_position_logits --features crossval,integration-tests -- --nocapture
```

## Metrics Explanation

### Cosine Similarity

- **Range**: -1.0 to 1.0
- **Interpretation**:
  - 1.0 = Identical distributions (perfect match)
  - 0.0 = Orthogonal (no correlation)
  - -1.0 = Opposite distributions
- **Used for**: Detecting when probability distributions diverge

### L2 Distance (Euclidean Distance)

- **Range**: 0.0 to ∞
- **Interpretation**:
  - 0.0 = Identical vectors
  - Larger values = Greater difference
- **Used for**: Quantifying the magnitude of divergence

### Max Absolute Difference

- **Range**: 0.0 to ∞
- **Interpretation**: The largest single element-wise difference across all positions
- **Used for**: Identifying worst-case numerical errors

## Implementation Notes

1. **Memory Efficiency**: The comparison operates on borrowed slices, avoiding unnecessary copies
2. **Graceful Degradation**: Size mismatches are handled by setting cosine similarity to 0.0 and L2 distance to ∞
3. **Numerical Stability**: Handles zero-norm vectors gracefully (returns 0.0 similarity)
4. **Performance**: O(n*m) where n = number of positions, m = vocabulary size

## Future Enhancements

Potential improvements:
- Per-token divergence visualization
- Configurable tolerance thresholds
- Support for batch comparisons
- Statistical significance testing
- Integration with receipt verification system

## Related Files

- `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs` - Core implementation
- `/home/steven/code/Rust/BitNet-rs/crossval/src/lib.rs` - Module export
- `/home/steven/code/Rust/BitNet-rs/crossval/tests/per_position_logits.rs` - Integration tests
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs` - Rust-side logits evaluation
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-sys/src/wrapper.rs` - C++ FFI wrapper

## References

- Existing parity tests: `/home/steven/code/Rust/BitNet-rs/crossval/tests/parity.rs`
- FFI session management: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/ffi_session.rs`
