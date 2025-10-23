# Quick Reference: QK256 Dual-Map Bug Fix

## The Problem in 30 Seconds

**File**: `crates/bitnet-models/tests/gguf_weight_loading_tests.rs`
**Test**: `test_ac3_tensor_shape_validation_cpu` (lines 378-437)
**Bug**: Line 401 accesses the WRONG map for QK256 tensors

```rust
// WRONG (current):
if let Some(qk256_tensor) = load_result.tensors.get("tok_embeddings.weight") {

// RIGHT (should be):
if let Some(qk256_tensor) = load_result.i2s_qk256.get("tok_embeddings.weight") {
```

## Why It's Wrong

The GGUF loader's **dual-map architecture**:
- `.tensors` → Dequantized tensors (F32, F16, F64)
- `.i2s_qk256` → Packed QK256 tensors (raw 2-bit data, NOT dequantized)

The test fixture generates a QK256 tensor, which is stored in `.i2s_qk256`, not `.tensors`.

## The 3-Line Fix

**File**: `crates/bitnet-models/tests/gguf_weight_loading_tests.rs`

```diff
-            if let Some(qk256_tensor) = load_result.tensors.get("tok_embeddings.weight") {
+            if let Some(qk256_tensor) = load_result.i2s_qk256.get("tok_embeddings.weight") {
                 assert_eq!(qk256_tensor.rows, 4, ...);
                 assert_eq!(qk256_tensor.cols, 256, ...);
             } else {
-                anyhow::bail!("Missing tok_embeddings.weight in tensors map");
+                anyhow::bail!("Missing tok_embeddings.weight in i2s_qk256 map");
             }
```

## Test Verification

```bash
# Before fix (fails):
cargo test -p bitnet-models --test gguf_weight_loading_tests \
  test_ac3_tensor_shape_validation_cpu --no-default-features --features cpu

# After fix (passes):
cargo test -p bitnet-models --test gguf_weight_loading_tests \
  test_ac3_tensor_shape_validation_cpu --no-default-features --features cpu
```

## Key Concepts

| Concept | Details |
|---------|---------|
| **GgufLoadResult** | Contains: `config`, `tensors`, `i2s_qk256` |
| **QK256 Tensors** | 2-bit packed format, stored in `.i2s_qk256` map |
| **Regular Tensors** | F32/F16/F64 dequantized format, stored in `.tensors` map |
| **Field Access** | I2SQk256NoScale: direct (`.rows`, `.cols`), CandleTensor: methods (`.shape().dims()`) |
| **Why Separate?** | Memory efficiency (2-bit vs 32-bit), kernel dispatch, avoid dequantization overhead |

## Architecture Diagram

```
GGUF File
  ├─ I2_S Quantized Tensor
  │  ├─ Size calculation: rows × ceil(cols/256) × 64 bytes
  │  ├─ Match detected?
  │  │  ├─ YES → QK256 format
  │  │  │       └─ Store in load_result.i2s_qk256
  │  │  │           └─ Type: I2SQk256NoScale
  │  │  │               └─ Access: .rows, .cols, .row_stride_bytes
  │  │  │
  │  │  └─ NO → BitNet32 format
  │  │       └─ Dequantize to F32
  │  │       └─ Store in load_result.tensors
  │  │           └─ Type: CandleTensor
  │  │               └─ Access: .shape().dims()
```

## Related Files

| File | Role |
|------|------|
| `gguf_simple.rs` | Loader implementation, decision tree (lines 273-301) |
| `i2s_qk256.rs` | I2SQk256NoScale structure definition (lines 65-71) |
| `qk256_fixtures.rs` | QK256 4×256 fixture generator |
| `gguf_weight_loading_tests.rs` | The failing test (lines 378-437) |

## Full Solution

**See**: `/home/steven/code/Rust/BitNet-rs/ci/solutions/gguf_shape_validation_fix.md` (514 lines)

Includes:
- Complete architecture explanation
- Fixture analysis
- Field access patterns
- Code diffs
- Verification procedures
- Root cause analysis
