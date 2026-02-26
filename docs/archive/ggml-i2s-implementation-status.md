# GGML I2_S (QK=256) Pure-Rust Implementation Status

**Date**: 2025-10-17
**Goal**: Add full Rust support for GGML's I2_S quantization format (QK=256, 64B/block, no embedded scales)

## Overview

This document tracks the implementation of pure-Rust GGML I2_S support for BitNet-rs. The goal is to eliminate the FFI dependency for GGML-format models while keeping C++ only in the parity harness.

## What's Implemented ✅

### 1. Scalar QK256 Kernel (Complete)

**File**: `crates/bitnet-models/src/quant/i2s_qk256.rs`

**Features**:
- ✅ Block unpacking: 64 bytes → 256 2-bit codes
- ✅ Code→float LUT (placeholder, needs C++ verification)
- ✅ Single-row GEMV: `gemv_qk256_row()`
- ✅ Multi-row GEMV: `gemv_qk256()`
- ✅ Tail handling for non-256-aligned dimensions
- ✅ Comprehensive unit tests (6 test cases)

**Key Constants**:
```rust
pub const QK256_BLOCK: usize = 256;
pub const QK256_PACKED_BYTES: usize = 64;
```

**Current LUT** (needs verification):
```rust
const LUT: [f32; 4] = [-3.0, -1.0, 1.0, 3.0];
```

### 2. C++ LUT Verification Helper (Complete)

**File**: `crates/bitnet-sys/csrc/i2s_qk256_dumper.cc`

**Purpose**: Confirm the exact code→float mapping from llama.cpp

**Usage**:
```bash
# Compile (adjust paths for your llama.cpp location)
g++ -std=c++17 -O2 -I/path/to/llama.cpp i2s_qk256_dumper.cc -o i2s_qk256_dumper

# Run
./i2s_qk256_dumper path/to/model.gguf
```

**See**: `crates/bitnet-sys/csrc/I2S_QK256_LUT_VERIFICATION.md` for detailed instructions

### 3. Flavor Detection (Already Complete)

**File**: `crates/bitnet-models/src/formats/gguf/types.rs`

The `I2SFlavor::GgmlQk256NoScale` variant and `detect_i2s_flavor()` function already exist and correctly identify GGML format.

## What's Remaining 🚧

### Critical Path Items

#### 1. Verify LUT Mapping (PRIORITY 1)

**Action Required**:
1. Compile and run `i2s_qk256_dumper` against llama.cpp
2. Compare output with Rust LUT in `i2s_qk256.rs`
3. Update `code_to_f32()` if values differ
4. Check for global/per-tensor scale factors

**Estimate**: 1-2 hours (depends on llama.cpp setup)

#### 2. GGUF Loader Integration (PRIORITY 2)

**File to Modify**: `crates/bitnet-models/src/gguf_simple.rs`

**Current Behavior** (line ~550):
```rust
if available.abs_diff(ggml_need) <= tolerance {
    return Err(BitNetError::Validation(format!(
        "GGML/llama.cpp format detected ... not yet supported"
    )));
}
```

**Needed Changes**:
- Instead of returning error, store raw QK256 bytes
- Options for storage:
  1. Store as Candle u8 tensor with metadata
  2. Create side map: `HashMap<String, RawQuantData>`
  3. Use custom Candle dtype (if supported)

**Recommendation**: Start with option #2 (side map) for quickest prototype, then refactor to #1 or #3.

**Estimate**: 2-3 hours

#### 3. Linear Layer Dispatch (PRIORITY 3)

**Files to Modify**:
- Find where `BitNetLinear` or matmul is dispatched
- Add QK256 detection and kernel routing

**Pseudocode**:
```rust
match weight_format {
    I2SFormat::InlineF16_32x10 => { /* existing path */ }
    I2SFormat::GgmlQk256NoScale => {
        let qk256_data = /* extract from storage */;
        let y = i2s_qk256::gemv_qk256(qk256_data, x, ...)?;
    }
}
```

**Estimate**: 2-4 hours (depends on architecture familiarity)

#### 4. Integration Tests (PRIORITY 4)

**File**: Create `crates/bitnet-models/tests/i2s_qk256_integration.rs`

**Test Cases**:
- Load GGML I2_S model
- Run single-layer inference
- Compare output with C++ parity harness
- Verify cosine similarity ≥ 0.99

**Estimate**: 1-2 hours

### Optional Enhancements

#### 5. Feature Flag (Safe Landing)

**File**: `crates/bitnet-models/Cargo.toml`

```toml
[features]
default = []
i2s-qk256-kernel = []  # Enable pure-Rust GGML I2_S
```

Allows safe rollout:
- Default: fail-closed on GGML I2_S
- Nightly: feature enabled for testing

**Estimate**: 30 minutes

#### 6. SIMD Fast Path

**After scalar kernel is validated**:
- AVX2 vectorized unpack + dot
- NEON vectorized unpack + dot
- Keep scalar as fallback

**Estimate**: 1-2 days (per platform)

## Architecture Decisions

### Storage Strategy

Three options for storing QK256 quantized data:

| Option | Pros | Cons | Effort |
|--------|------|------|--------|
| **Side Map** | Quick, no API changes | Extra indirection | Low |
| **U8 Tensor + Metadata** | Clean, uses Candle | Metadata tracking | Medium |
| **Custom Dtype** | Proper abstraction | Requires Candle fork? | High |

**Recommendation**: Start with Side Map, migrate to U8 Tensor later if needed.

### Dispatch Strategy

```rust
pub enum WeightFormat {
    F32Dense,
    I2S_InlineF16_32x10 { data: Vec<u8>, scales: Vec<f16> },
    I2S_GgmlQk256NoScale { data: Vec<u8>, row_stride: usize },
    // ... other formats
}

impl BitNetLinear {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        match &self.weight_format {
            WeightFormat::I2S_GgmlQk256NoScale { data, row_stride } => {
                i2s_qk256::gemv_qk256(data, x, self.out_dim, self.in_dim, *row_stride)
            }
            // ... other cases
        }
    }
}
```

## Testing Strategy

### Phase 1: Kernel Validation
- ✅ Unit tests (scalar logic)
- 🚧 C++ LUT verification
- 🚧 Single-block parity

### Phase 2: Integration
- 🚧 Load GGML I2_S GGUF
- 🚧 Single-layer forward pass
- 🚧 Multi-layer inference
- 🚧 Compare with C++ harness

### Phase 3: End-to-End
- 🚧 Full model inference (Rust only)
- 🚧 Logits parity with C++ (≥0.99 cosine)
- 🚧 Receipt validation with proper kernel IDs

## Acceptance Criteria

### For Rust Production Path
- [ ] LUT verified against llama.cpp
- [ ] GGUF loader stores QK256 data (no dequant)
- [ ] Linear layers dispatch to QK256 kernel
- [ ] All tests pass with `--features i2s-qk256-kernel`
- [ ] Inference receipt shows `backend: rust`

### For Parity Harness
- [ ] C++ harness runs GGML I2_S without crashes
- [ ] One FFI session per test (no re-init)
- [ ] No `munmap_chunk()` errors
- [ ] Receipt shows `validation.backend: cpp`

## Current Risks

### 1. LUT Mismatch (High Impact, Easy to Fix)
If C++ LUT differs from placeholder, outputs will be wrong.
**Mitigation**: Run helper first before proceeding.

### 2. Scale Factor Mystery (Medium Impact)
GGML may apply hidden per-tensor/global scales.
**Mitigation**: Dump real model blocks, compare magnitudes.

### 3. Candle Integration (Medium Impact)
Current arch dequantizes everything; quantized path needs new storage.
**Mitigation**: Use side map initially, refactor later.

## Next Steps (Recommended Order)

1. **Verify LUT** (30 min - 2 hours)
   - Compile C++ helper
   - Run against GGML model
   - Update Rust LUT if needed

2. **Storage Prototype** (2-3 hours)
   - Add side map for QK256 raw bytes
   - Modify GGUF loader to populate map
   - Keep existing error path as fallback

3. **Dispatch Wiring** (2-4 hours)
   - Find matmul dispatch point
   - Add QK256 branch
   - Wire to `gemv_qk256()`

4. **Integration Test** (1-2 hours)
   - Load small GGML model
   - Run inference
   - Compare with C++ output

5. **Feature Flag** (30 min)
   - Add `i2s-qk256-kernel` flag
   - Update CI to test both paths

6. **Documentation** (1 hour)
   - Update CLAUDE.md
   - Add howto guide
   - Document kernel selection

**Total Estimate**: 6-12 hours for MVP (scalar kernel, no SIMD)

## Success Metrics

- [ ] GGML I2_S models load without FFI fallback
- [ ] Inference runs 100% Rust (no C++ except parity)
- [ ] Logits parity ≥ 0.99 cosine similarity
- [ ] No performance regression vs C++ (scalar)
- [ ] Receipts show proper kernel IDs

## References

- GGML I2_S spec: `docs/reference/ggml-quants.md` (TODO: create)
- Parity harness: `crossval/tests/parity_bitnetcpp.rs`
- Flavor detection tests: `crates/bitnet-models/tests/i2s_flavor_detection.rs`
- BitNet I2_S (32-block): `crates/bitnet-models/src/quant/i2s.rs`
