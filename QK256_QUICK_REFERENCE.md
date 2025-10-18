# QK256 Integration - Quick Reference Card

## Document Index

1. **QK256_INTEGRATION_ANALYSIS.md** (560 lines)
   - Comprehensive overview of all components
   - Tensor storage patterns and naming conventions
   - GGUF loader implementation details
   - Parity harness configuration
   - File paths and line number references
   - Environmental controls

2. **QK256_TECHNICAL_DETAILS.md** (426 lines)
   - Memory layout diagrams
   - Unpacking and GEMV computation
   - Flavor detection algorithm
   - Error handling & fallback flows
   - Kernel integration patterns
   - Test examples
   - Performance optimization notes

3. **This document**: Quick lookup reference

---

## The 4 Critical Integration Points

### 1. Tensor Loading (gguf_simple.rs:783-791)
**Problem**: Currently returns error for GgmlQk256NoScale
**Solution**: Store in I2SQk256NoScale HashMap instead
**Impact**: Enables pure-Rust QK256 support

```rust
// Current: return Err("GGML/llama.cpp format detected...")
// Action: Create I2SQk256NoScale and store in dedicated map
```

### 2. Layer Forward Pass (quantized_linear.rs)
**Problem**: No QK256 flavor branching
**Solution**: Add flavor detection → backend selection
**Impact**: Routes QK256 ops to correct kernel

```rust
// Pseudocode pattern:
match detect_i2s_flavor() {
    BitNet32F16 => self.gemv_i2s_32(),
    Split32WithSibling => self.gemv_i2s_32_split(),
    GgmlQk256NoScale => {
        match select_qk256_backend() {
            Rust => self.gemv_qk256_rust(),
            Ffi => self.gemv_qk256_ffi(),
        }
    }
}
```

### 3. Backend Selection (backend.rs)
**Problem**: No I2sQk256Backend abstraction
**Solution**: Follow Iq2sBackend pattern
**Impact**: Environment control via BITNET_I2S_QK256_IMPL

### 4. Kernel Recording (kernel_recorder.rs)
**Problem**: No QK256 kernel tracking
**Solution**: Add "i2s_qk256_gemv_row" entries
**Impact**: Honest parity receipts

---

## Kernel API Quick Reference

### I2SQk256NoScale Structure
```rust
pub struct I2SQk256NoScale {
    pub rows: usize,
    pub cols: usize,
    pub row_stride_bytes: usize,  // ceil(cols/256)*64
    pub qs: Vec<u8>,              // Row-major packed data
}
```

### Core Functions
```rust
// Single row: y = A[row] · x
pub fn gemv_qk256_row(qs_row: &[u8], x: &[f32], cols: usize) -> f32

// All rows: y = Ax
pub fn gemv_qk256(
    qs_data: &[u8],
    x: &[f32],
    y_out: &mut [f32],
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
) -> Result<()>

// Code → float: 0→-2.0, 1→-1.0, 2→+1.0, 3→+2.0
pub fn code_to_f32(code: u8) -> f32

// Unpack 64B block → 256 2-bit codes
pub fn unpack_qk256_block(
    qs64: &[u8; 64],
    out_codes256: &mut [u8; 256]
)
```

---

## I2S Flavor Detection

| Flavor | Block Size | Scales | Bytes/Block | Detection |
|--------|-----------|--------|-------------|-----------|
| BitNet32F16 | 32 | Inline F16 | 10 | blocks_32 * 10 ± 128 |
| Split32WithSibling | 32 | Separate | 8 | blocks_32 * 8 ± 128 + has_sibling |
| GgmlQk256NoScale | 256 | None (LUT) | 64 | blocks_256 * 64 ± 128 |

**Priority**: GgmlQk256 > Split32 > BitNet32F16

---

## Tensor Naming Patterns

### BitNet Models (current standard)
```
token_embd.weight         [vocab, hidden]
output.weight             [hidden, vocab]
blk.{i}.attn_q.weight    [hidden, hidden]
blk.{i}.ffn_gate.weight  [inter, hidden]
```

### Storage in HashMap
```rust
HashMap<String, CandleTensor>        // Most tensors
HashMap<String, I2SQk256NoScale>     // QK256-specific (when integrated)
```

### Access by Layer
```rust
let weights = tensor_map.get("blk.0.attn_q.weight")?;
```

---

## Error Handling Pattern

```
load_tensor_from_gguf()
├─ BitNet32F16 detected → Dequantize, return CandleTensor ✓
├─ Split32WithSibling detected → Dequantize, return CandleTensor ✓
└─ GgmlQk256NoScale detected → Return Error
   └─ Parity harness catches
      └─ eval_logits_once_for_parity() routes to C++ FFI
         └─ Receipt shows "backend": "cpp"
```

---

## Parity Harness Flow

```
CROSSVAL_GGUF=/path/to/qk256.gguf \
BITNET_CPP_DIR=/path/to/bitnet-cpp \
cargo test -p crossval --features ffi,integration-tests
    ↓
parity_bitnetcpp() detects GgmlQk256
    ↓
eval_logits_once_for_parity() tries Rust path
    ↓
Rust path fails with "GGML format" error
    ↓
FFI routing activates (if BITNET_CPP_DIR set)
    ↓
C++ eval_logits_via_ffi_session() succeeds
    ↓
Cosine similarity ≥ 0.99 → "cosine_ok"
    ↓
Receipt: {
    "validation": { "backend": "cpp", "compute": "cpp" },
    "parity": { "cosine_ok": true, "exact_match_rate": 1.0 }
}
```

---

## Memory Layout Example

**Model**: 2048×4096 weights in QK256 format

```
Total rows = 2048
Total cols = 4096
Blocks per row = ceil(4096 / 256) = 16
Bytes per block = 64
Row stride = 16 * 64 = 1024 bytes
Total storage = 2048 * 1024 = 2,097,152 bytes (2 MB)

Row 0:   [0..1024)      → 16 blocks of 256 elements
Row 1:   [1024..2048)   → 16 blocks of 256 elements
...
Row 2047: [2096128..2097152)
```

---

## Environment Variables

```bash
# Model & C++ backend
export CROSSVAL_GGUF=/path/to/model.gguf
export BITNET_CPP_DIR=/path/to/bitnet-cpp

# Backend selection (when integrated)
export BITNET_I2S_QK256_IMPL=rust    # Force pure-Rust
export BITNET_I2S_QK256_IMPL=ffi     # Force C++ FFI

# Determinism
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
```

---

## Test Validation Checklist

- [ ] Flavor detection identifies QK256 correctly
- [ ] Pure-Rust `gemv_qk256_row()` computes correctly
- [ ] Pure-Rust `gemv_qk256()` multi-row works
- [ ] Backend selection environment variables work
- [ ] FFI routing activates for QK256 models
- [ ] Cosine similarity ≥ 0.99 for parity
- [ ] Exact match rate tracking accurate
- [ ] Receipt generation includes kernel IDs
- [ ] Documentation updated
- [ ] CI/CD pipeline passes

---

## Performance Baseline

**Expectations** (scalar Rust implementation):
- QK256 GEMV (2048×4096): ~10-50 ms on CPU
- vs BitNet32F16: Similar (both scalar)
- vs C++ FFI: May be faster for small ops, slower for large

---

## Key Files (Absolute Paths)

**To modify**:
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/gguf_simple.rs` (lines 760-814)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/layers/quantized_linear.rs`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/quant/backend.rs`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/kernel_recorder.rs`

**Already ready**:
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/quant/i2s_qk256.rs` (kernel API)
- `/home/steven/code/Rust/BitNet-rs/crossval/tests/parity_bitnetcpp.rs` (harness)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/i2s_flavor_detection.rs` (tests)

---

## Quick Build & Test

```bash
# Build with CPU support
cd /home/steven/code/Rust/BitNet-rs
cargo build --no-default-features --features cpu

# Build with FFI support (requires bitnet-cpp)
cargo build --no-default-features --features cpu,ffi

# Test flavor detection
cargo test -p bitnet-models i2s_flavor_detection --no-default-features --features cpu

# Test parity (requires GGUF model)
CROSSVAL_GGUF=/models/qk256.gguf \
cargo test -p crossval --no-default-features --features cpu,integration-tests

# Test parity with C++ (requires bitnet-cpp)
CROSSVAL_GGUF=/models/qk256.gguf \
BITNET_CPP_DIR=/path/to/bitnet-cpp \
cargo test -p crossval --features cpu,ffi,integration-tests
```

---

## Integration Status

```
Phase 1: Load & Store              [COMPLETE] ✓
  - QK256 detection             [DONE]
  - I2SQk256NoScale structure   [DONE]
  - gemv_qk256 kernels          [DONE]

Phase 2: Layer Integration          [IN PROGRESS]
  - Add flavor branching        [TODO]
  - Create backend selector     [TODO]
  - Route to kernels            [TODO]

Phase 3: Testing & Validation       [READY]
  - Flavor detection tests      [DONE]
  - Parity harness              [DONE]
  - Error propagation tests     [DONE]

Phase 4: Deployment                 [PENDING]
  - CI/CD integration           [TODO]
  - Documentation               [TODO]
  - Release                     [TODO]
```
