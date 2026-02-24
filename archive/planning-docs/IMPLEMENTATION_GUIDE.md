# Sprint 1 Implementation Guide - Exact Code Locations & Sections

## Overview
This guide provides exact line-by-line references for implementing all-positions logits extraction.

---

## File 1: `crates/bitnet-inference/src/parity.rs`

### Section A: Template for `eval_logits_all_positions()`
**Insert After**: Line 128 (after `eval_logits_once_for_parity()`)  
**Size**: ~120 lines  
**Template Source**: Copy from lines 30-111, modify as shown below

```rust
/// Perform a forward pass and return logits for ALL token positions (PRODUCTION)
///
/// This function processes the entire token sequence at once and returns
/// logits for each position, enabling per-position cross-validation and debugging.
///
/// This function supports all BitNet quantization formats including:
/// - BitNet I2_S (32-element blocks with inline F16 scales)
/// - GGML I2_S (QK256 - 256-element blocks, pure Rust kernel)
/// - Other standard quantization formats
///
/// # Arguments
/// * `model_path` - Path to the GGUF model file
/// * `tokens` - Token IDs (all tokens to process)
///
/// # Returns
/// * Logits vectors for all token positions (seq_len × vocab_size matrix)
///   where seq_len matches the number of input tokens
///
/// # Errors
/// * Fails if model file cannot be loaded or if unsupported format is detected
pub fn eval_logits_all_positions(model_path: &str, tokens: &[i32]) -> Result<Vec<Vec<f32>>> {
    // COPY BLOCK: Lines 31-89 from eval_logits_once()
    // Load model tensors with Rust GGUF loader (fail-closed, no FFI routing)
    let (config, model) = match load_gguf_full(
        Path::new(model_path),
        Device::Cpu,
        bitnet_models::GGUFLoaderConfig::default(),
    ) {
        Ok(result) => {
            eprintln!(
                "DEBUG parity: load_gguf_full returned config: hidden={}, n_heads={}, n_kv_heads={}",
                result.config.model.hidden_size,
                result.config.model.num_heads,
                result.config.model.num_key_value_heads
            );

            // Convert i2s_qk256 map to raw_tensors map with key remapping
            // QK256 tensors are stored as raw bytes in I2SQk256NoScale, we need to convert them to Candle tensors
            // and remap GGUF keys (e.g., "blk.0.attn_q.weight") to model keys (e.g., "layers.0.attention.q_proj.weight")
            let mut raw_tensors_unmapped = std::collections::HashMap::new();
            for (key, qk256_tensor) in result.i2s_qk256.iter() {
                // Create a U8 tensor from the raw bytes with shape [rows, row_stride_bytes]
                let raw_bytes_tensor = candle_core::Tensor::from_raw_buffer(
                    &qk256_tensor.qs,
                    candle_core::DType::U8,
                    &[qk256_tensor.rows, qk256_tensor.row_stride_bytes],
                    &candle_core::Device::Cpu,
                )
                .map_err(|e| anyhow::anyhow!("Failed to create raw tensor for '{}': {}", key, e))?;

                // Store with .qk256_qs suffix (GGUF key format)
                let qk256_key = format!("{}.qk256_qs", key);
                raw_tensors_unmapped.insert(qk256_key, raw_bytes_tensor);
            }

            eprintln!(
                "DEBUG parity: Converted {} QK256 tensors to raw_tensors (pre-remap)",
                raw_tensors_unmapped.len()
            );

            // Remap keys from GGUF format to model format
            let raw_tensors = bitnet_models::weight_mapper::remap_gguf_weights_with_options(
                &raw_tensors_unmapped,
                false, // non-strict
            )?;

            eprintln!("DEBUG parity: Remapped raw_tensors keys ({} tensors)", raw_tensors.len());

            let model = BitNetModel::from_gguf(
                result.config.clone(),
                result.tensors,
                raw_tensors,
                Device::Cpu,
            )?;
            (result.config, model)
        }
        Err(e) => {
            // Propagate the error with context (fail-closed for ggml I2_S)
            anyhow::bail!("Failed to load GGUF model: {}", e);
        }
    };

    // Convert i32 tokens to u32
    let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();

    // Get embeddings for ALL tokens (not just incremental single token)
    let embedded = model.embed(&tokens_u32)?;

    // Run forward pass for full sequence (NO KV CACHE - process all at once)
    // Use forward_full() instead of forward()
    let output = model.forward_full(&embedded)?;

    // Get logits from the output
    let logits = model.logits(&output)?;

    // Extract logits for ALL token positions
    let logits = extract_all_position_logits(logits)?;

    Ok(logits)
}
```

### Section B: Template for `extract_all_position_logits()`
**Insert After**: Line 261 (after `extract_last_token_logits()`)  
**Size**: ~70 lines  
**Template Source**: Adapt from lines 223-261

```rust
/// Extract logits for all token positions from the model output
///
/// This function takes the 3D logits tensor [B, T, V] and converts it to
/// a Vec<Vec<f32>> where each inner vector contains logits for one token position.
fn extract_all_position_logits(logits: bitnet_common::ConcreteTensor) -> Result<Vec<Vec<f32>>> {
    use bitnet_common::ConcreteTensor;

    match logits {
        ConcreteTensor::BitNet(tensor) => {
            // Get the underlying Candle tensor
            let candle_tensor = tensor.as_candle();

            // Shape should be [batch, seq_len, vocab_size]
            let dims = candle_tensor.dims();
            if dims.len() != 3 {
                anyhow::bail!("Expected 3D logits tensor [B, T, V], got {:?}", dims);
            }

            let (batch_size, seq_len, vocab_size) = (dims[0], dims[1], dims[2]);

            // Ensure we have batch_size=1
            if batch_size != 1 {
                anyhow::bail!("Expected batch_size=1, got {}", batch_size);
            }

            let mut all_logits = Vec::with_capacity(seq_len);

            // Extract logits for each position
            for pos in 0..seq_len {
                // Extract logits for this position: narrow to single position in seq dimension
                let pos_logits = candle_tensor
                    .narrow(1, pos, 1)?      // Get single position from seq dim
                    .squeeze(1)?              // Remove seq dimension to get [B, V]
                    .i(0)?;                   // Get first (and only) batch to get [V]

                // Convert to F32 if needed
                let pos_logits = if pos_logits.dtype() != DType::F32 {
                    pos_logits.to_dtype(DType::F32)?
                } else {
                    pos_logits.clone()
                };

                // Convert to Vec<f32>
                let logits_vec = pos_logits.to_vec1::<f32>()?;
                all_logits.push(logits_vec);
            }

            Ok(all_logits)
        }
        ConcreteTensor::Mock(mock) => {
            // For mock tensors, return mock logits for each position
            let seq_len = mock.shape()[1];
            let vocab_size = mock.shape()[2];
            let mock_logits = vec![0.0f32; vocab_size];
            Ok(vec![mock_logits; seq_len])
        }
    }
}
```

### Section C: Add to test module (lines 283-308)
**Insert Before**: Line 308 (before closing brace of tests module)

```rust
    #[test]
    fn test_eval_logits_all_positions() {
        // Test with non-existent model file (should fail gracefully)
        let tokens = vec![1, 2, 3, 4];

        // This should fail since the file doesn't exist
        let result = eval_logits_all_positions("nonexistent_test.gguf", &tokens);

        // Should fail with proper error message (fail-closed behavior)
        assert!(result.is_err(), "Non-existent model should fail to load");

        // Verify it's a proper file error, not a ggml I2_S error
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(
            err_str.contains("Failed to open GGUF file") || err_str.contains("No such file"),
            "Expected file not found error, got: {}",
            err_str
        );
    }
```

---

## File 2: `crates/bitnet-inference/src/lib.rs`

### Section: Add to public exports
**Modify**: Lines 44-46 (pub use parity block)

**Current**:
```rust
pub use parity::{
    eval_logits_incremental, eval_logits_once, eval_logits_once_for_parity, get_model_config,
    get_model_vocab_size,
};
```

**New**:
```rust
pub use parity::{
    eval_logits_all_positions, eval_logits_incremental, eval_logits_once, eval_logits_once_for_parity,
    get_model_config, get_model_vocab_size,
};
```

---

## File 3: Reference - No Changes Needed

### `crates/bitnet-models/src/transformer.rs`

These functions already exist and support what we need:

- **`forward_full()` at line 1416**: Processes full sequences `[B, T, H] → [B, T, H]`
- **`logits()` at line 1548**: Handles 3D tensors via rank-3 path (lines 1635-1687)
- **`embed()`: Processes all tokens when given full token sequence**

### `crossval/src/logits_compare.rs`

Already has everything we need (lines 49-102):

- **`compare_per_position_logits()`**: Compares `Vec<Vec<f32>>` from both implementations
- **`cosine_similarity()`**: Per-position comparison metric
- **`l2_distance()`**: Per-position distance calculation

---

## Verification Checklist

After making changes:

### Compilation
- [ ] `cargo build --no-default-features --features cpu` succeeds
- [ ] No compiler warnings
- [ ] No clippy warnings: `cargo clippy --all-targets`

### Unit Tests
- [ ] `cargo test parity::tests::test_mock_eval` passes
- [ ] `cargo test parity::tests::test_eval_logits_all_positions` passes

### Integration Tests
- [ ] `cargo test -p bitnet-inference --no-default-features --features cpu` passes
- [ ] All existing tests still pass

### With C++ Comparison (if crossval available)
- [ ] `cargo test -p bitnet-inference --features crossval` passes
- [ ] Per-position cosine similarity > 0.9999

---

## Implementation Steps (In Order)

1. **Open file**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs`

2. **Add `eval_logits_all_positions()` function**:
   - Position: After line 128
   - Copy-paste from Section A above
   - Make sure to change lines 102-108 from current flow to new flow

3. **Add `extract_all_position_logits()` function**:
   - Position: After line 261 (after `extract_last_token_logits()` closes)
   - Copy-paste from Section B above
   - Notice the loop over all positions instead of just last

4. **Add test case**:
   - Position: Before line 308 (before closing brace)
   - Copy-paste from Section C above

5. **Update `lib.rs` exports**:
   - Position: Lines 44-46
   - Add `eval_logits_all_positions` to the pub use list

6. **Verify compilation**:
   ```bash
   cd /home/steven/code/Rust/BitNet-rs
   cargo build --no-default-features --features cpu
   ```

7. **Run tests**:
   ```bash
   cargo test -p bitnet-inference --no-default-features --features cpu
   ```

8. **Optional: Test with C++**:
   ```bash
   cargo test -p bitnet-inference --features crossval
   ```

---

## Key Differences from `eval_logits_once()`

| Line | Current (`eval_logits_once()`) | New (`eval_logits_all_positions()`) |
|------|------|-----|
| ~95 | `let cache = KVCache::new(...)` | ✗ DELETE - not needed |
| ~99 | `let embedded = model.embed(...)` | ✓ Same (will be full sequence) |
| ~102 | `let output = model.forward(...)` | Change to: `model.forward_full(...)` |
| ~105 | `let logits = model.logits(...)` | ✓ Same (handles 3D) |
| ~108 | `extract_last_token_logits(...)` | Change to: `extract_all_position_logits(...)` |
| Return | `Vec<f32>` | Change to: `Vec<Vec<f32>>` |

---

## Code Sections to Study First

Before implementing, read these exact sections to understand the flow:

1. **Current single-token flow**:
   - `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs` lines 30-111

2. **Current extraction logic**:
   - `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs` lines 223-261

3. **Model full-sequence support**:
   - `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/transformer.rs` lines 1416-1430

4. **Logits 3D path**:
   - `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/transformer.rs` lines 1635-1687

5. **Comparison infrastructure**:
   - `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs` lines 49-102

---

## Common Issues & Solutions

### Issue: "Expected 3D logits tensor"
**Cause**: Calling with wrong input path  
**Solution**: Ensure using `forward_full()` not `forward()`

### Issue: Shape mismatch error
**Cause**: Wrong tensor operations  
**Solution**: Check dimensions at each step:
- After embed: should be `[1, T, H]`
- After forward_full: should be `[1, T, H]`
- After logits: should be `[1, T, V]`

### Issue: Tests fail with OOM
**Cause**: Too many tokens  
**Solution**: Use smaller sequences for testing (e.g., 4-8 tokens)

### Issue: Compilation error about `forward_full`
**Cause**: Function not found  
**Solution**: Verify it exists at `transformer.rs:1416`

---

## Performance Notes

- Processing all tokens at once is generally **faster** than calling `eval_logits_once()` multiple times
- Memory usage is O(seq_len × vocab_size) - acceptable for typical sequence lengths
- GPU version will have even better performance (but not needed for MVP)

---

## Files to Review

Essential:
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/parity.rs`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/transformer.rs`

Reference:
- `/home/steven/code/Rust/BitNet-rs/LOGITS_EXTRACTION_ANALYSIS.md` (full analysis)
- `/home/steven/code/Rust/BitNet-rs/SPRINT1_QUICK_REFERENCE.md` (quick overview)
- `/home/steven/code/Rust/BitNet-rs/crossval/src/logits_compare.rs` (comparison utilities)

