# QK256 Integration - Technical Deep Dive

## Memory Layout & Data Flow

### QK256 Tensor Storage in GGUF

```
File: model.gguf
├── Header (24 bytes)
├── Metadata KV pairs
├── Tensor metadata (names, shapes, offsets, types)
└── Tensor data (row-major, QK256-packed)
    └── For "blk.0.attn_q.weight" [2048, 4096]:
        ├── Row 0: ceil(4096/256)*64 = 16*64 = 1024 bytes
        ├── Row 1: 1024 bytes
        ├── ...
        └── Row 2047: 1024 bytes
        Total: 2048 * 1024 = 2,097,152 bytes
```

### Unpacking QK256 Block (256 elements → 64 bytes)

```
Input: 64-byte block (packed 2-bit codes)
[byte0, byte1, ..., byte63]

Process each byte:
  byte0 = c0 | (c1<<2) | (c2<<4) | (c3<<6)
  byte1 = c4 | (c5<<2) | (c6<<4) | (c7<<6)
  ...
  byte63 = c252 | (c253<<2) | (c254<<4) | (c255<<6)

Extraction:
  c0 = (byte0 >> 0) & 0b11  →  LUT[0] → f32
  c1 = (byte0 >> 2) & 0b11  →  LUT[1] → f32
  c2 = (byte0 >> 4) & 0b11  →  LUT[2] → f32
  c3 = (byte0 >> 6) & 0b11  →  LUT[3] → f32
  ...

Output: 256 floats [-2.0, -1.0, +1.0, +2.0]
```

### GEMV Computation: y = Ax

```
Input:
  A: quantized weight matrix [out_features, in_features] = [2048, 4096]
     Row-major, QK256-packed (1024 bytes per row)
  x: dense input vector [in_features] = [4096]

For each output row i in 0..2048:

  1. Get row bytes from A:
     row_start = i * 1024
     row_end = (i+1) * 1024
     qs_row = &A.qs[row_start..row_end]

  2. Compute dot product:
     acc = 0.0
     for j in 0..16:  // 16 blocks per row (4096 / 256)
       block_start = j * 64
       block = &qs_row[block_start..block_start+64]
       codes = unpack_block(block)  // 256 codes

       for k in 0..256:
         w = code_to_f32(codes[k])
         acc += w * x[j*256 + k]

     y[i] = acc

Output: y [out_features] = [2048]
```

## Flavor Detection Algorithm

### I2S Layout Classification

```
Given:
  nelems = product of tensor shape
  available = bytes in GGUF tensor data
  has_scale_sibling = bool (separate scale tensor exists)

Calculate expected bytes for each layout:
  blocks_32 = (nelems + 31) / 32
  blocks_256 = (nelems + 255) / 256

  bitnet_inline_need = blocks_32 * 10
  split32_need = blocks_32 * 8
  ggml_qk256_need = blocks_256 * 64

Tolerance = 128 bytes (alignment padding)

Detection priority:
  1. If matches ggml_qk256_need (±128):
     → GgmlQk256NoScale (256-block, no scales)

  2. Else if has_scale_sibling && matches split32_need (±128):
     → Split32WithSibling (32-block, separate scales)

  3. Else if matches bitnet_inline_need (±128):
     → BitNet32F16 (32-block, inline F16 scales)

  4. Else:
     → Error: unsupported layout
```

### Flavor Characteristic Table

| Flavor | Block Size | Scales | Format | Bytes/Block |
|--------|-----------|--------|--------|------------|
| BitNet32F16 | 32 | Inline F16 | 8B data + 2B f16 | 10 |
| Split32WithSibling | 32 | Separate tensor | 8B data only | 8 |
| GgmlQk256NoScale | 256 | None (LUT direct) | 64B packed | 64 |

## Error Handling & Fallback

### GGUF Loader Error Flow

```
load_tensor_from_gguf(tensor_index, info, device)
  │
  └─ Match info.tensor_type:
     │
     ├─ F32: Direct load → Candle F32 tensor ✓
     ├─ F16: Convert to F32 → Candle tensor ✓
     │
     ├─ I2_S: Detect layout (3 branches)
     │  │
     │  ├─ GgmlQk256NoScale detected:
     │  │  └─ Return Error("GGML format detected...")
     │  │     └─ Parity harness catches → routes to C++ FFI
     │  │
     │  ├─ Split32WithSibling detected:
     │  │  ├─ Find sibling scale tensor (via find_sibling_scale)
     │  │  ├─ Load scales with cast_scales_to_f32
     │  │  ├─ Pack data + scales
     │  │  ├─ Dequantize via BitNet I2S quantizer
     │  │  └─ Return Candle tensor ✓
     │  │
     │  └─ BitNet32F16 detected:
     │     ├─ Unpack inline F16 scales
     │     ├─ Dequantize via BitNet I2S quantizer
     │     └─ Return Candle tensor ✓
     │
     ├─ Q4_0 → TL1 dequant → Candle tensor ✓
     └─ Q8_K → TL2 dequant → Candle tensor ✓

Notes:
  - GgmlQk256NoScale error is INTENTIONAL (fail-closed)
  - Parity harness detects and routes to C++ FFI
  - Pure-Rust path only supports BitNet32F16 and Split32
```

## Parity Harness Integration Points

### eval_logits_once_for_parity Flow

```
eval_logits_once_for_parity(model_path, tokens)
  │
  ├─ Try pure-Rust path:
  │  eval_logits_once(model_path, tokens)
  │  │
  │  └─ load_gguf(path, CPU)
  │     └─ Detects I2S flavor
  │        ├─ BitNet32F16: Load successfully ✓
  │        ├─ Split32: Load successfully ✓
  │        └─ GgmlQk256NoScale: Return Error("GGML format...")
  │
  ├─ Check error for "GGML/llama.cpp format detected":
  │  │
  │  ├─ If NO: Return Rust error (fail-closed)
  │  │
  │  └─ If YES and BITNET_CPP_DIR set:
  │     │
  │     └─ Route to C++ FFI:
  │        eval_logits_via_ffi_session(model_path, tokens)
  │        ├─ Load/reuse global FFI session
  │        ├─ Prefill with tokens
  │        ├─ Eval last logits
  │        └─ Return logits from C++ ✓
  │
  └─ Compare Rust vs C++ logits:
     ├─ Cosine similarity ≥ 0.99 → "cosine_ok"
     ├─ Exact match rate (for discrete decode)
     └─ Write parity receipt

Receipt fields:
  "validation": {
    "backend": "cpp" or "rust",  // Which path actually ran
    "compute": "cpp" or "rust"   // Where compute happened
  },
  "parity": {
    "cosine_similarity": f32,
    "cosine_ok": bool,
    "exact_match_rate": f32,
    "first_divergence_step": Option<usize>
  }
```

## Kernel Integration Points

### QuantizedLinear::forward() Integration

```
pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
  // Current structure (hypothetical)

  match self.qtype {
    QuantizationType::I2S => {
      // INTEGRATION POINT 1: Detect flavor
      let flavor = detect_i2s_flavor(&self.weights)?;

      match flavor {
        I2SFlavor::BitNet32F16 => {
          // Already works - use BitNet dequantizer
          self.gemv_i2s_32(input)
        }
        I2SFlavor::Split32WithSibling => {
          // Already works - use BitNet dequantizer
          self.gemv_i2s_32_split(input)
        }
        I2SFlavor::GgmlQk256NoScale => {
          // INTEGRATION POINT 2: Route to backend
          match select_qk256_backend() {
            I2sQk256Backend::Rust => {
              // Call pure-Rust gemv_qk256
              self.gemv_qk256_rust(input)
            }
            I2sQk256Backend::Ffi => {
              // Call C++ FFI through bitnet-sys
              self.gemv_qk256_ffi(input)
            }
          }
        }
      }
    }
    _ => { /* TL1, TL2, etc. */ }
  }
}

fn gemv_qk256_rust(&self, input: &Tensor) -> Result<Tensor> {
  let input_vec = input.flatten_all()?.to_vec1::<f32>()?;
  let mut output = vec![0.0f32; self.out_features];

  // Call i2s_qk256::gemv_qk256()
  i2s_qk256::gemv_qk256(
    &self.weights.data,          // Raw packed bytes
    &input_vec,
    &mut output,
    self.out_features,
    self.in_features,
    self.weight_row_stride,
  )?;

  // Add bias if present
  if let Some(bias) = &self.bias {
    for i in 0..self.out_features {
      output[i] += bias_vec[i];
    }
  }

  Ok(Tensor::from_vec(output, &[1, 1, self.out_features], device)?)
}
```

## Kernel Manager Integration

### Kernel Recording for QK256

```
When recording kernels executed during forward pass:

For QK256 operations, record:
  {
    "kernel_id": "i2s_qk256_gemv_row",
    "layer": "blk.0.attn_q",
    "operation": "matrix-vector multiply",
    "rows": 2048,
    "cols": 4096,
    "blocks_per_row": 16,
    "bytes_per_block": 64,
    "backend": "rust",  // or "ffi"
    "compute_type": "quantized",
    "quantization_type": "i2s_qk256",
    "device": "cpu",  // or "cuda:0"
    "timestamp_us": 12345,
    "duration_us": 678
  }

For parity receipts, aggregate:
  "kernels_executed": [
    "i2s_qk256_gemv_row",
    "i2s_qk256_gemv_row",
    ...  // one per output row
  ],
  "kernel_ids": [
    "i2s_qk256_gemv_row",  // unique
  ],
  "backend_used": "rust"  // or "ffi"
```

## Test Coverage Examples

### Smoke Test: Pure-Rust QK256 Forward

```rust
#[test]
fn qk256_forward_pass_rust() {
  // Create mock QK256 quantized weights
  let weights = I2SQk256NoScale {
    rows: 64,
    cols: 256,
    row_stride_bytes: 64,
    qs: vec![0xAAu8; 64 * 64],  // All codes = 2 (→ +1.0)
  };

  // Create layer
  let layer = QuantizedLinear::new_i2s(weights, Device::Cpu)?;

  // Create input (256 elements)
  let input = vec![0.5f32; 256];

  // Forward pass
  let output = layer.forward(&Tensor::from_vec(input, &[1, 256]))?;

  // Verify: output[i] ≈ sum(input) * 1.0 = 128.0 (all codes are +1.0)
  let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
  assert!(output_vec.len() == 64);
  for &val in &output_vec {
    assert!((val - 128.0).abs() < 0.01);
  }
}
```

### Parity Test: QK256 with C++ Reference

```rust
#[tokio::test]
#[cfg_attr(not(feature = "ffi"), ignore)]
async fn qk256_parity_with_cpp() {
  // Require C++ backend
  if env::var("BITNET_CPP_DIR").is_err() {
    return;  // Skip if C++ not available
  }

  // Load real QK256 model
  let gguf_path = env::var("CROSSVAL_GGUF").expect("CROSSVAL_GGUF not set");
  let tokens = vec![1, 2, 3, 4, 5];

  // Get logits from both backends
  let rust_logits = eval_logits_once(&gguf_path, &tokens)?;
  let cpp_logits = eval_logits_once_for_parity(&gguf_path, &tokens)?;

  // Compute cosine similarity
  let cosine = cosine_similarity(&rust_logits, &cpp_logits);

  // Verify parity (should be very close)
  assert!(cosine >= 0.9999, "QK256 parity: cosine={}", cosine);
}
```

## Build Configuration

### Feature Flags for QK256

```toml
[features]
default = []
cpu = []                    # Pure-Rust kernels (always available)
gpu = ["cuda", "bitnet-kernels/cuda"]
ffi = ["bitnet-sys/ffi"]   # C++ FFI (optional)
crossval = []              # Cross-validation harness
integration-tests = []     # Real-model tests

[dependencies.bitnet-models]
features = ["cpu"]         # Load from GGUF

[dependencies.bitnet-inference]
features = ["cpu"]         # Inference engine

[dependencies.bitnet-sys]
optional = true
features = ["ffi"]         # C++ bindings (optional)
```

### Compilation for Different Backends

```bash
# Pure-Rust (always works)
cargo build --no-default-features --features cpu

# With C++ FFI support
cargo build --no-default-features --features cpu,ffi

# With GPU support
cargo build --no-default-features --features gpu,cpu

# Full testing setup
cargo test --no-default-features --features cpu,ffi,crossval,integration-tests
```

## Performance Optimization Notes

### QK256 Kernel Optimization Opportunities

1. **SIMD Vectorization**:
   - Current: Scalar loop (one output per iteration)
   - Opportunity: AVX2/AVX-512 to process 8-16 rows in parallel

2. **Cache Optimization**:
   - Row stride = ceil(cols/256)*64 bytes
   - For 4096 cols: stride = 1024 bytes (fits in L1 cache for small blocks)
   - Process row-by-row to maximize cache reuse

3. **FFI Overhead**:
   - C++ FFI has function call overhead (~1-2 μs)
   - For small operations (< 1K elements), Rust kernel may be faster
   - For large operations (> 100K), FFI amortizes overhead

4. **Memory Layout**:
   - QK256 uses row-major storage (good for GEMV, bad for GEMM)
   - No transpose needed for forward pass
   - Consider cache-tiling for very large matrices
