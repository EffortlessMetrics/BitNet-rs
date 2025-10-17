# BitNet.rs QK256 Support Integration - Codebase Summary

## Executive Summary

This document provides a comprehensive analysis of the BitNet.rs codebase structure for integrating QK256 (GGML I2_S with 256-element blocks) support. The analysis covers tensor storage patterns, layer implementations, kernel APIs, and the parity harness that will validate the integration.

**Key Finding**: The codebase has a **dual-flavor approach** for I2_S quantization:
1. **BitNet32F16** (pure Rust, 32-element blocks with inline F16 scales)
2. **GgmlQk256NoScale** (GGML format, 256-element blocks, no per-block scales)

The QK256 kernel API already exists and is ready for integration into the layer forward pass.

---

## 1. GGUF Loader Implementation

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/gguf_simple.rs`

### Key Patterns

#### 1.1 Tensor Storage & Naming Conventions

**Tensor Map Structure** (lines 13-14):
```rust
pub struct GgufLoadResult {
    pub config: bitnet_common::BitNetConfig,
    pub tensors: HashMap<String, CandleTensor>,
    pub i2s_qk256: HashMap<String, I2SQk256NoScale>,
}
```

**Tensor Naming Conventions** (validated in `normalize_vendor_key` at weight_mapper.rs):
- **BitNet style**: `blk.{layer}.attn_q.weight`, `blk.{layer}.ffn_gate.weight`
- **LLaMA style**: `layers.{layer}.self_attn.wq.weight`, `layers.{layer}.mlp.w1.weight`
- **Transformer outputs**: `output.weight` or `lm_head.weight`

#### 1.2 I2_S Quantization Detection

**Location**: `load_tensor_from_gguf()` at lines 702-971

**Layout Detection Logic** (lines 760-814):
```
For I2_S type tensors:

1. Calculate expected bytes for different layouts:
   - BitNet32F16 (inline F16): blocks_32 * 10 bytes/block
   - Split32WithSibling: blocks_32 * 8 bytes/block (scales separate)
   - GgmlQk256NoScale: blocks_256 * 64 bytes/block

2. Use tolerance ±128 bytes to match available bytes

3. Detect layout by available byte count:
   - If matches GgmlQk256NoScale: blocks_256 = (cols + 255) / 256
   - If matches BitNet32F16: blocks_32 = (cols + 31) / 32
   - Prefer GgmlQk256 when qk256_need matches (CRITICAL)
```

**Key Detection Code** (lines 783-791):
```rust
// Check if available bytes match ggml I2_S (QK_K=256, 64B/block)
if available.abs_diff(ggml_need) <= tolerance {
    return Err(BitNetError::Validation(format!(
        "I2_S '{}': GGML/llama.cpp format detected (QK_K=256, 64B/block, no scale tensor)...
        Please use the FFI backend..."
    )));
}
```

**Current Status**: QK256 detection returns an error, suggesting use of FFI backend. This is where kernel integration happens.

---

## 2. QK256 Kernel API

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/quant/i2s_qk256.rs`

### 2.1 Core Data Structure

**I2SQk256NoScale** (lines 65-128):
```rust
pub struct I2SQk256NoScale {
    pub rows: usize,           // Weight matrix rows
    pub cols: usize,           // Weight matrix columns
    pub row_stride_bytes: usize, // Bytes per row = ceil(cols/256) * 64
    pub qs: Vec<u8>,           // Packed quantized data (contiguous row-major)
}
```

**Layout Example** (from lines 57-64):
```
For 512×1024 weight matrix:
- rows = 512
- cols = 1024
- blocks_per_row = ceil(1024/256) = 4
- row_stride_bytes = 4 * 64 = 256 bytes
- qs.len() = 512 * 256 = 131,072 bytes
```

### 2.2 Kernel Functions

**gemv_qk256_row()** (lines 186-233):
```rust
pub fn gemv_qk256_row(qs_row: &[u8], x: &[f32], cols: usize) -> f32
```
- Single row × vector dot product
- Input: packed bytes, dense vector
- Output: scalar result (dot product)
- **Unpacking**: Uses LUT [code_to_f32()] with 2-bit code extraction per byte

**gemv_qk256()** (lines 249-289):
```rust
pub fn gemv_qk256(
    qs_data: &[u8],        // All rows packed
    x: &[f32],             // Dense input vector
    y_out: &mut [f32],     // Output vector
    rows: usize,
    cols: usize,
    row_stride_bytes: usize,
) -> Result<()>
```
- Multi-row GEMV: y = Ax where A is quantized
- Calls `gemv_qk256_row()` for each row

### 2.3 Code-to-Float Mapping (GGML Reference)

**LUT** (lines 139-146):
```rust
pub fn code_to_f32(code: u8) -> f32 {
    const LUT: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];
    LUT[code as usize]
}
```

**Verified** against GGML (ggml-quants.c:62):
- Code 0 → -2.0
- Code 1 → -1.0
- Code 2 → +1.0
- Code 3 → +2.0

### 2.4 Unpacking Example

**unpack_qk256_block()** (lines 159-168):
```
Each byte packs 4 elements (2 bits each):
byte = elem0 | (elem1 << 2) | (elem2 << 4) | (elem3 << 6)

Extraction:
- elem[i] = (byte >> (2*i)) & 0x03
```

---

## 3. Tensor Storage & Accessing Weights

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/weight_mapper.rs` (lines 44-150)

### 3.1 Tensor Shape Handling

**Canonical Format** (lines 92-96):
```
Attention: layers.{i}.attention.{q_proj|k_proj|v_proj|o_proj}.weight
           [out_dim, in_dim] = [hidden, hidden]

Feed-forward: layers.{i}.feed_forward.{gate_proj|up_proj|down_proj}.weight
              [out_dim, in_dim] = [intermediate, hidden] or [hidden, intermediate]

Norms: layers.{i}.attention_norm.weight, layers.{i}.ffn_norm.weight
       [hidden]
```

### 3.2 Tensor Access Pattern in GGUF Loader

**Location**: `gguf_simple.rs` lines 149-161

```rust
let mut tensor_map = HashMap::with_capacity(tensor_count);

for i in 0..tensor_count {
    let info = gguf_reader.get_tensor_info(i)?;
    let tensor = load_tensor_from_gguf(gguf_reader, i, info, &cdevice)?;
    tensor_map.insert(info.name.clone(), tensor);
}
```

**Retrieval in Model** (`bitnet.rs` lines 36-69):
```rust
pub fn from_gguf(
    config: BitNetConfig,
    tensors: HashMap<String, CandleTensor>,
    device: Device,
) -> Result<Self> {
    // Validate required tensors present
    // Build transformer using tensors
    let transformer = Self::build_transformer(&config, &tensors, &device)?;
}
```

---

## 4. Linear Layer Forward Pass Implementation

**Location**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/layers/quantized_linear.rs`

### 4.1 QuantizedLinear Structure

**Lines 143-168**:
```rust
pub struct QuantizedLinear {
    pub weights: QuantizedTensor,
    pub bias: Option<BitNetTensor>,
    pub qtype: QuantizationType,
    pub in_features: usize,
    pub out_features: usize,
    pub device: Device,
    pub kernel_manager: Arc<KernelManager>,
}
```

**Creation** (lines 172-194):
```rust
pub fn new_i2s(weights: QuantizedTensor, device: Device) -> Result<Self>
```

### 4.2 Kernel Selection Architecture

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/quant/backend.rs`

**Backend Abstraction** (lines 165-172):
```rust
pub enum Iq2sBackend {
    Rust,  // Pure Rust (always available)
    Ffi,   // Optional ggml FFI (requires `iq2s-ffi` feature)
}

impl Iq2sBackend {
    pub fn selected() -> Self {
        match std::env::var("BITNET_IQ2S_IMPL").ok().as_deref() {
            Some("ffi") => Iq2sBackend::Ffi,
            _ => Iq2sBackend::Rust,
        }
    }
}
```

**Pattern for QK256**: Will likely need similar backend selection:
```rust
pub enum I2sQk256Backend {
    Rust,   // Pure Rust (crates/bitnet-models/src/quant/i2s_qk256.rs)
    Ffi,    // Optional ggml FFI through bitnet-sys
}
```

---

## 5. Parity Harness Configuration

**File**: `/home/steven/code/Rust/BitNet-rs/crossval/tests/parity_bitnetcpp.rs`

### 5.1 Backend Detection & Selection

**Location**: Lines 376-380
```rust
let cpp_available = env::var("BITNET_CPP_DIR").is_ok();
if !cpp_available {
    eprintln!("BITNET_CPP_DIR not set; running Rust-only validation");
}
```

### 5.2 Parity Validation Flow

**Lines 375-412**:
1. Check `CROSSVAL_GGUF` environment variable
2. Load GGUF model with Rust loader
3. Detect model format (QK256, BitNet32F16, etc.)
4. If QK256 detected and `BITNET_CPP_DIR` set → route to C++ FFI
5. If Rust-only → validate with pure Rust kernels
6. Compare outputs with C++ if available

### 5.3 Token Sources & Routing

**Lines 444-448**:
```rust
// Use pure Rust tokenization for parity (no C++ fallback)
let tokens_for_parity = rust_ids.clone();
let tokenizer_source = "rust";
eprintln!("parity.tokenizer_source=rust");
```

**Kernel Routing** (Lines 450-451):
```rust
let rust_logits = rust_eval_last_logits(&gguf_path, &tokens_for_parity, vocab_size).await?;
```

**Parity Function** (`parity.rs` lines 75-99):
```rust
pub fn eval_logits_once_for_parity(model_path: &str, tokens: &[i32]) -> Result<Vec<f32>> {
    // Try pure-Rust path first
    let rust_result = eval_logits_once(model_path, tokens);
    
    // If ggml I2_S error AND FFI available → route to C++
    #[cfg(feature = "ffi")]
    if is_ggml_i2s_error && std::env::var("BITNET_CPP_DIR").is_ok() {
        eprintln!("parity: ggml I2_S detected -> routing compute to C++ FFI");
        return eval_logits_via_ffi_session(model_path, tokens);
    }
}
```

### 5.4 Receipt Generation

**Lines 552-600**:
```json
{
    "validation": {
        "backend": validation_backend,  // "rust" or "cpp"
        "tokenizer": "rust",            // Always Rust
        "compute": validation_backend
    },
    "parity": {
        "cpp_available": cpp_loaded,
        "cosine_similarity": cosine_similarity,
        "cosine_ok": cosine_ok,
        "exact_match_rate": exact_match_rate,
        "first_divergence_step": first_divergence
    }
}
```

---

## 6. I2S Flavor Detection

**File**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/i2s_flavor_detection.rs`

### 6.1 Flavor Enum

**Location**: `formats/gguf/types.rs` (inferred from test usage)

```rust
pub enum I2SFlavor {
    BitNet32F16,          // 32-elem blocks, inline F16 scales (10B/block)
    Split32WithSibling,   // 32-elem blocks, separate scale tensor (8B/block)
    GgmlQk256NoScale,     // 256-elem blocks, no per-block scales (64B/block)
}
```

### 6.2 Detection Logic

**Test Coverage** (lines 23-139):
- BitNet32F16: 2 blocks × 10B = 20 bytes (line 28)
- Split32WithSibling: 2 blocks × 8B = 16 bytes (line 43)
- GgmlQk256NoScale: 4 blocks × 64B = 256 bytes (line 106)
- Tolerance: ±8 bytes for alignment (line 121)

**Priority**: QK256 preferred when byte count matches both split32 and qk256
(Due to fundamental coincidence at certain element counts)

---

## 7. Quantization Infrastructure Integration

### 7.1 Quantization Types

**Location**: `bitnet_common` crate (referenced throughout)

```rust
pub enum QuantizationType {
    I2S,    // Pure Rust (BitNet32F16)
    TL1,    // Lookup table (ARM optimized)
    TL2,    // Lookup table (x86 optimized)
    // Note: QK256 is a variant of I2S, needs backend selection
}
```

### 7.2 Cross-Validation Test Infrastructure

**Files**:
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/i2s_flavor_detection.rs`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/ggml_i2s_error_propagation.rs`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/logits_greedy_smoke.rs`

**Pattern**: 
1. Load GGUF with real model
2. Detect I2S variant (BitNet32 vs QK256)
3. If QK256: validate error propagation and FFI routing
4. Compare outputs with C++ reference

---

## 8. Key Integration Points for QK256 Support

### 8.1 Tensor Loading Pipeline

**Status**: ✅ Complete for QK256 storage

```
GGUF File
  ↓
GgufReader::get_tensor_info() → Detect QK256 (256-elem blocks, no scales)
  ↓
load_tensor_from_gguf() → Current: Returns error for QK256
  ↓
[INTEGRATION POINT]: Create I2SQk256NoScale storage instead of error
```

### 8.2 Layer Forward Pass

**Status**: ⚠️  Needs integration

```
Input: [batch, seq_len, hidden] → [1, 1, 2048]
  ↓
QuantizedLinear::forward()
  ↓
[INTEGRATION POINT]: 
  - Detect qtype == I2S
  - Select backend (Rust or FFI)
  - If Rust: call gemv_qk256_row() for each output row
  - If FFI: delegate to C++ through bitnet-sys
  ↓
Output: [batch, seq_len, out_features]
```

### 8.3 Parity Harness Routing

**Status**: ✅ Ready for QK256

```
parity_bitnetcpp() 
  ↓
Load GGUF → detect I2S variant
  ↓
eval_logits_once_for_parity()
  ↓
[INTEGRATION POINT]:
  - If BitNet32F16: Use Rust kernels
  - If QK256: Route to C++ if BITNET_CPP_DIR set
  - Compare outputs
```

---

## 9. File Paths & Line Number References

### Core Files

| Component | File | Key Lines |
|-----------|------|-----------|
| GGUF Loader | `crates/bitnet-models/src/gguf_simple.rs` | 760-814 (I2S detection) |
| QK256 Kernels | `crates/bitnet-models/src/quant/i2s_qk256.rs` | 186-289 (GEMV ops) |
| Weight Mapper | `crates/bitnet-models/src/weight_mapper.rs` | 92-150 (tensor names) |
| Quantized Layer | `crates/bitnet-inference/src/layers/quantized_linear.rs` | 143-250 (structure) |
| Backend Selection | `crates/bitnet-models/src/quant/backend.rs` | 165-172 (enum) |
| Parity Harness | `crossval/tests/parity_bitnetcpp.rs` | 375-412 (flow) |
| Parity Function | `crates/bitnet-inference/src/parity.rs` | 75-99 (FFI routing) |

### Test Files

| Test | File | Validation |
|------|------|-----------|
| Flavor Detection | `crates/bitnet-models/tests/i2s_flavor_detection.rs` | 22-139 |
| Error Propagation | `crates/bitnet-models/tests/ggml_i2s_error_propagation.rs` | (location TBD) |
| Smoke Tests | `crates/bitnet-inference/tests/logits_greedy_smoke.rs` | (location TBD) |

---

## 10. Environmental Controls

### 10.1 Detection & Selection

```bash
# Force Rust backends
BITNET_IQ2S_IMPL=rust cargo test

# Enable FFI backends (if compiled with iq2s-ffi feature)
BITNET_IQ2S_IMPL=ffi cargo test

# Enable C++ parity validation
BITNET_CPP_DIR=/path/to/bitnet-cpp cargo test --features ffi

# Deterministic testing
BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1 cargo test
```

### 10.2 Parity Harness Environment

```bash
# Enable crossval with GGUF model
CROSSVAL_GGUF=/path/to/model.gguf \
BITNET_CPP_DIR=/path/to/bitnet-cpp \
cargo test -p crossval --features ffi,integration-tests

# Rust-only validation (no C++ required)
CROSSVAL_GGUF=/path/to/model.gguf \
cargo test -p crossval --features integration-tests
```

---

## 11. Summary: QK256 Integration Roadmap

### Phase 1: Load & Store (COMPLETE)
- [x] Detect QK256 format in GGUF loader
- [x] Create `I2SQk256NoScale` storage structure
- [x] Verify gemv_qk256 kernels compile

### Phase 2: Layer Integration (CURRENT)
- [ ] Modify `QuantizedLinear::forward()` to handle QK256
- [ ] Add backend selection for I2S variant
- [ ] Route pure-Rust QK256 through `gemv_qk256()` 
- [ ] Route FFI QK256 through C++ bindings
- [ ] Add kernel_recorder entries for QK256 kernels

### Phase 3: Testing & Validation
- [ ] Add I2S flavor detection to layer initialization
- [ ] Add smoke tests for QK256 forward pass
- [ ] Validate parity with C++ reference
- [ ] Benchmark pure-Rust vs FFI performance
- [ ] Generate baseline receipts

### Phase 4: Deployment
- [ ] Documentation updates
- [ ] CI/CD integration
- [ ] Release notes

---

## Appendix: Tensor Naming Examples

### BitNet Models
```
token_embd.weight         [vocab, hidden]
output.weight             [hidden, vocab]
blk.0.attn_q.weight      [hidden, hidden]
blk.0.attn_k.weight      [hidden, hidden]
blk.0.attn_v.weight      [hidden, hidden]
blk.0.attn_output.weight [hidden, hidden]
blk.0.attn_norm.weight   [hidden]
blk.0.ffn_gate.weight    [inter, hidden]
blk.0.ffn_up.weight      [inter, hidden]
blk.0.ffn_down.weight    [hidden, inter]
blk.0.ffn_norm.weight    [hidden]
output_norm.weight       [hidden]
```

### LLaMA Models (for reference)
```
model.embed_tokens.weight            [vocab, hidden]
lm_head.weight                       [vocab, hidden]
model.layers.0.self_attn.wq.weight  [hidden, hidden]
model.layers.0.self_attn.wk.weight  [hidden, hidden]
model.layers.0.self_attn.wv.weight  [hidden, hidden]
model.layers.0.self_attn.wo.weight  [hidden, hidden]
model.layers.0.mlp.w1.weight        [inter, hidden]
model.layers.0.mlp.w3.weight        [inter, hidden]
model.layers.0.mlp.w2.weight        [hidden, inter]
model.layers.0.input_layernorm.weight [hidden]
model.layers.0.post_attention_layernorm.weight [hidden]
```

