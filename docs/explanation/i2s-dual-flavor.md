# I2_S Dual-Flavor Quantization Support

**Status:** Draft Specification
**Authors:** BitNet.rs Architecture Team
**Created:** 2025-10-17
**Version:** 1.0

## Executive Summary

This specification defines architectural support for two I2_S quantization flavors in BitNet.rs:

1. **BitNet32F16** (existing): 32-element blocks with inline f16 scales (10 bytes/block)
2. **GgmlQk256NoScale** (new): 256-element blocks with separate scale tensors (64 bytes/block)

The architecture follows a **fail-closed production** model with **parity-only FFI routing**:
production code rejects unsupported formats, while the parity harness can route ggml I2_S
validation to C++ when BITNET_CPP_DIR is set.

## Architecture Decision Record

### Context

BitNet.rs currently supports BitNet's native I2_S format (32-element blocks, 8B packed + 2B f16 scale).
GGML/llama.cpp uses a different I2_S format with 256-element blocks and separate scale tensors.
We need to support both formats for model compatibility while maintaining production quality standards.

### Decision

We adopt a **phased implementation** approach:

- **Phase 1**: FFI session wrapper for C++ validation-only compute (parity harness)
- **Phase 2**: Pure-Rust kernel implementation with scalar + SIMD paths (production)

Production code **fail-closes** on ggml I2_S format until Phase 2 kernels land.
Parity harness routes compute to C++ when FFI is available and BITNET_CPP_DIR is set,
maintaining Rust tokenizer and I/O.

### Consequences

**Benefits:**

- Incremental validation against C++ reference before committing to Rust implementation
- Session-based FFI prevents munmap_chunk() crashes from repeated model load/free
- Opaque representation avoids premature f32 dequantization at load time
- Receipt tracking enables exact parity validation of both implementations

**Risks:**

- Phase 1 validation requires C++ dependency (BITNET_CPP_DIR + ffi feature)
- 2-bit mapping must exactly match ggml signed symmetric codes
- Block size difference (32 vs 256) impacts SIMD vectorization strategies

## Component Specifications

### 1. I2SFlavor Enum and Detection Logic

#### API Contract

```rust
/// I2_S quantization flavor (block layout and scale storage)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I2SFlavor {
    /// BitNet native: 32 elems/block, 8B packed + 2B f16 scale = 10B/block
    BitNet32F16,

    /// GGML/llama.cpp: 256 elems/block, 64B packed data, scales in separate tensor
    /// QK_K = 256 per ggml conventions
    GgmlQk256NoScale,
}

impl I2SFlavor {
    /// Detect I2_S flavor from tensor metadata
    ///
    /// # Arguments
    /// * `info.size` - Available bytes in tensor data
    /// * `total_elements` - Total number of elements (from shape: product of dims)
    ///
    /// # Detection Logic
    /// ```text
    /// For N total elements:
    ///   BitNet32F16:      num_blocks = ceil(N/32),  expected = num_blocks * 10
    ///   GgmlQk256NoScale: num_blocks = ceil(N/256), expected = num_blocks * 64
    ///
    /// If info.size matches BitNet32F16 expected → BitNet32F16
    /// If info.size matches GgmlQk256NoScale expected → GgmlQk256NoScale
    /// If both match (ambiguous) → prefer BitNet32F16 (existing format)
    /// If neither match → Err(UnsupportedLayout)
    /// ```
    pub fn detect(total_elements: usize, available_bytes: usize) -> Result<Self> {
        let blocks_32 = total_elements.div_ceil(32);
        let expected_bitnet = blocks_32 * 10; // 8B packed + 2B f16 scale

        let blocks_256 = total_elements.div_ceil(256);
        let expected_ggml = blocks_256 * 64; // 64B packed data only

        match (available_bytes == expected_bitnet, available_bytes == expected_ggml) {
            (true, false) => Ok(I2SFlavor::BitNet32F16),
            (false, true) => Ok(I2SFlavor::GgmlQk256NoScale),
            (true, true) => {
                // Ambiguous: prefer existing format
                tracing::warn!(
                    "I2_S size {} matches both flavors (N={} → BitNet={}, GGML={}), using BitNet32F16",
                    available_bytes, total_elements, expected_bitnet, expected_ggml
                );
                Ok(I2SFlavor::BitNet32F16)
            }
            (false, false) => Err(BitNetError::Model(ModelError::InvalidFormat {
                format: format!(
                    "I2_S tensor size {} doesn't match BitNet32F16 ({}) or GgmlQk256NoScale ({})",
                    available_bytes, expected_bitnet, expected_ggml
                ),
            })),
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            I2SFlavor::BitNet32F16 => 32,
            I2SFlavor::GgmlQk256NoScale => 256, // QK_K in ggml
        }
    }

    pub fn data_bytes_per_block(&self) -> usize {
        match self {
            I2SFlavor::BitNet32F16 => 8,  // 32 * 2 bits / 8 = 8 bytes
            I2SFlavor::GgmlQk256NoScale => 64, // 256 * 2 bits / 8 = 64 bytes
        }
    }

    pub fn has_inline_scales(&self) -> bool {
        matches!(self, I2SFlavor::BitNet32F16)
    }

    pub fn requires_scale_tensor(&self) -> bool {
        matches!(self, I2SFlavor::GgmlQk256NoScale)
    }
}
```

#### Integration with GGUF Reader

```rust
// In bitnet-models/src/formats/gguf/reader.rs

impl TensorInfo {
    /// Detect I2_S flavor for quantized tensors
    pub fn detect_i2s_flavor(&self) -> Result<I2SFlavor> {
        if self.tensor_type != GgufTensorType::I2_S {
            return Err(BitNetError::Model(ModelError::InvalidFormat {
                format: format!("detect_i2s_flavor called on non-I2_S tensor: {:?}", self.tensor_type),
            }));
        }

        let total_elements: usize = self.shape.iter().product();
        I2SFlavor::detect(total_elements, self.size as usize)
    }
}
```

### 2. QuantTensor Opaque Representation

#### API Contract (QuantTensor)

```rust
/// Opaque quantized tensor with flavor-specific storage
///
/// For GgmlQk256NoScale: stores raw packed bytes without dequantization.
/// Scales loaded separately from companion scale tensor (if present).
#[derive(Debug, Clone)]
pub struct QuantTensor {
    /// Raw packed quantized data (opaque bytes)
    pub data: Vec<u8>,

    /// Scale factors (optional - inline for BitNet32F16, loaded separately for ggml)
    pub scales: Option<Vec<f32>>,

    /// Original tensor shape [d1, d2, ...]
    pub shape: Vec<usize>,

    /// I2_S flavor determines layout and scale handling
    pub flavor: I2SFlavor,

    /// Quantization type (always I2_S for this spec)
    pub qtype: QuantizationType,
}

impl QuantTensor {
    /// Create from GGUF I2_S tensor with flavor detection
    pub fn from_i2s_gguf(
        tensor_info: &TensorInfo,
        raw_data: &[u8],
        scale_tensor: Option<&[f32]>, // Companion scale tensor (f32/f16/f64)
    ) -> Result<Self> {
        let flavor = tensor_info.detect_i2s_flavor()?;
        let total_elements: usize = tensor_info.shape.iter().product();

        match flavor {
            I2SFlavor::BitNet32F16 => {
                // Parse inline f16 scales from 10-byte blocks
                let num_blocks = total_elements.div_ceil(32);
                let mut scales = Vec::with_capacity(num_blocks);
                let mut data = Vec::with_capacity(num_blocks * 8);

                for block_idx in 0..num_blocks {
                    let offset = block_idx * 10;

                    // Extract 8 bytes packed data
                    data.extend_from_slice(&raw_data[offset..offset + 8]);

                    // Extract 2 bytes f16 scale
                    let scale_bytes = [raw_data[offset + 8], raw_data[offset + 9]];
                    let scale_f16 = half::f16::from_le_bytes(scale_bytes);
                    scales.push(scale_f16.to_f32());
                }

                Ok(Self {
                    data,
                    scales: Some(scales),
                    shape: tensor_info.shape.clone(),
                    flavor,
                    qtype: QuantizationType::I2S,
                })
            }

            I2SFlavor::GgmlQk256NoScale => {
                // Store opaque packed bytes (no dequant at load)
                let num_blocks = total_elements.div_ceil(256);
                let expected_bytes = num_blocks * 64;

                if raw_data.len() != expected_bytes {
                    return Err(BitNetError::Model(ModelError::InvalidFormat {
                        format: format!(
                            "GgmlQk256NoScale data size {} != expected {}",
                            raw_data.len(), expected_bytes
                        ),
                    }));
                }

                // Scales must come from companion tensor (if provided)
                let scales = if let Some(scale_data) = scale_tensor {
                    if scale_data.len() != num_blocks {
                        return Err(BitNetError::Model(ModelError::InvalidFormat {
                            format: format!(
                                "Scale tensor length {} != num_blocks {}",
                                scale_data.len(), num_blocks
                            ),
                        }));
                    }
                    Some(scale_data.to_vec())
                } else {
                    // Production code MUST have scales for inference
                    // Parity validation may defer to C++ FFI
                    None
                };

                Ok(Self {
                    data: raw_data.to_vec(),
                    scales,
                    shape: tensor_info.shape.clone(),
                    flavor,
                    qtype: QuantizationType::I2S,
                })
            }
        }
    }

    /// Check if tensor is ready for inference (has scales)
    pub fn has_scales(&self) -> bool {
        self.scales.is_some()
    }

    /// Production inference requirement: fail if no scales
    pub fn require_scales(&self) -> Result<&[f32]> {
        self.scales.as_deref().ok_or_else(|| {
            BitNetError::Model(ModelError::InvalidFormat {
                format: format!(
                    "I2_S tensor {:?} missing scales (required for inference)",
                    self.flavor
                ),
            })
        })
    }
}
```

### 3. FFI Session Wrapper Design (Phase 1)

#### Architecture

Session-based FFI wrapper for **validation-only** compute, preventing munmap_chunk() crashes
from repeated model load/free cycles.

#### API Contract (FFI Session)

```rust
/// FFI session for ggml I2_S validation (PARITY ONLY)
///
/// Reusable context prevents repeated model load/free crashes.
/// Only available when `ffi` feature enabled and BITNET_CPP_DIR set.
#[cfg(feature = "ffi")]
pub struct I2SFfiSession {
    /// C++ model handle (owned, dropped on session cleanup)
    cpp_model: bitnet_sys::BitnetModel,

    /// C++ inference context (owned)
    cpp_ctx: bitnet_sys::BitnetContext,

    /// Model path for logging
    model_path: String,

    /// Session creation timestamp
    created_at: std::time::Instant,
}

#[cfg(feature = "ffi")]
impl I2SFfiSession {
    /// Create FFI session for ggml I2_S validation
    ///
    /// # Errors
    /// - C++ model load failure
    /// - Context creation failure
    /// - BITNET_CPP_DIR not set
    pub fn new(model_path: &str) -> Result<Self> {
        // Verify BITNET_CPP_DIR for parity validation
        if std::env::var("BITNET_CPP_DIR").is_err() {
            return Err(BitNetError::Configuration {
                key: "BITNET_CPP_DIR".to_string(),
                reason: "Required for ggml I2_S FFI validation".to_string(),
            });
        }

        let cpp_model = bitnet_sys::BitnetModel::from_file(model_path)?;
        let cpp_ctx = bitnet_sys::BitnetContext::new(
            &cpp_model,
            4096, // max_tokens
            1,    // batch_size
            0,    // threads (auto)
        )?;

        tracing::info!(
            "Created I2_S FFI session for ggml validation: {}",
            model_path
        );

        Ok(Self {
            cpp_model,
            cpp_ctx,
            model_path: model_path.to_string(),
            created_at: std::time::Instant::now(),
        })
    }

    /// Evaluate tokens via C++ FFI (validation only)
    ///
    /// Returns logits for last token position.
    pub fn eval_logits(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        use bitnet_sys::{bitnet_eval_tokens, bitnet_prefill, cpp_vocab_size};

        let vocab_size = cpp_vocab_size(&self.cpp_ctx)?;

        // Prefill context with tokens
        bitnet_prefill(&self.cpp_ctx, tokens)?;

        // Evaluate to get logits
        let logits = bitnet_eval_tokens(&self.cpp_ctx, tokens, vocab_size)?;

        // Sanity check: logits should be non-zero
        let sum_abs: f32 = logits.iter().map(|x| x.abs()).sum();
        if sum_abs < 1e-6 {
            return Err(BitNetError::Validation(format!(
                "C++ FFI logits near zero (sum_abs={:.2e}), KV/logits wiring issue",
                sum_abs
            )));
        }

        Ok(logits)
    }

    /// Get vocab size from C++ model
    pub fn vocab_size(&self) -> Result<usize> {
        bitnet_sys::cpp_vocab_size(&self.cpp_ctx).map(|v| v as usize)
    }
}

#[cfg(feature = "ffi")]
impl Drop for I2SFfiSession {
    fn drop(&mut self) {
        let duration = self.created_at.elapsed();
        tracing::debug!(
            "Dropping I2_S FFI session after {:.2}s: {}",
            duration.as_secs_f32(),
            self.model_path
        );
        // Rust Drop trait handles C++ cleanup automatically
        // Manual bitnet_free_backend() caused "free(): invalid pointer" errors
    }
}
```

#### Integration with Parity Harness

```rust
// In bitnet-inference/src/parity.rs

/// Evaluate logits for PARITY VALIDATION (can route to C++ FFI)
///
/// Production code uses `eval_logits_once` (fail-closed).
/// Parity validation uses this function for automatic FFI routing.
pub fn eval_logits_once_for_parity(model_path: &str, tokens: &[i32]) -> Result<Vec<f32>> {
    // Try pure-Rust path first
    let rust_result = eval_logits_once(model_path, tokens);

    // Check if error is due to ggml I2_S format
    let is_ggml_i2s_error = if let Err(ref e) = rust_result {
        let err_str = format!("{:?}", e);
        err_str.contains("GgmlQk256NoScale") && err_str.contains("no scale tensor")
    } else {
        return rust_result; // Success with Rust - return early
    };

    // Only parity may route to C++, only when FFI and BITNET_CPP_DIR present
    #[cfg(feature = "ffi")]
    if is_ggml_i2s_error && std::env::var("BITNET_CPP_DIR").is_ok() {
        tracing::warn!(
            "parity: ggml I2_S detected → routing compute to C++ FFI (tokenizer still Rust)"
        );

        // Create session (reusable across calls in real parity tests)
        let session = I2SFfiSession::new(model_path)?;
        return session.eval_logits(tokens);
    }

    // Fail-closed: propagate original Rust error
    rust_result
}
```

### 4. Pure-Rust Kernel API (Phase 2)

#### Kernel Specification

```rust
/// I2_S ggml flavor kernel: 256-element blocks, 2-bit signed symmetric quantization
///
/// # Layout
/// - Block size: QK_K = 256 elements
/// - Data: 64 bytes packed (256 * 2 bits / 8)
/// - Scales: f32 array (separate tensor, 1 scale per block)
///
/// # 2-bit Mapping (signed symmetric)
/// Must match ggml exactly:
/// - 00 → -2
/// - 01 → -1
/// - 10 → +1
/// - 11 → +2
///
/// # Implementation Variants
/// - Scalar: reference implementation
/// - AVX2: 256-bit SIMD (process 32 bytes = 128 elements per iteration)
/// - AVX-512: 512-bit SIMD (process 64 bytes = 256 elements = 1 block per iteration)
/// - NEON: 128-bit SIMD (process 16 bytes = 64 elements per iteration)
pub trait I2SGgmlKernel {
    /// Unpack 2-bit signed values from ggml I2_S format
    ///
    /// # Arguments
    /// * `packed` - Packed 2-bit data (64 bytes per block)
    /// * `num_blocks` - Number of 256-element blocks
    ///
    /// # Returns
    /// Unpacked i8 values in range [-2, -1, +1, +2]
    fn unpack_ggml_i2s(&self, packed: &[u8], num_blocks: usize) -> Vec<i8>;

    /// Dequantize ggml I2_S to f32
    ///
    /// # Arguments
    /// * `packed` - Packed 2-bit data (64 bytes per 256-element block)
    /// * `scales` - f32 scales (1 per block)
    ///
    /// # Returns
    /// Dequantized f32 values
    fn dequantize_ggml_i2s(&self, packed: &[u8], scales: &[f32]) -> Result<Vec<f32>>;

    /// Matrix multiplication: A (f32) @ B (ggml I2_S) → C (f32)
    ///
    /// # Arguments
    /// * `a` - Input matrix [M, K] (f32)
    /// * `b_packed` - Weight matrix [K, N] packed as ggml I2_S
    /// * `b_scales` - Scales for B (num_blocks_k * num_blocks_n)
    /// * `c` - Output matrix [M, N] (f32, pre-allocated)
    /// * `m, n, k` - Matrix dimensions
    fn matmul_i2s_ggml(
        &self,
        a: &[f32],
        b_packed: &[u8],
        b_scales: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;
}
```

#### Scalar Reference Implementation

```rust
pub struct I2SGgmlScalarKernel;

impl I2SGgmlKernel for I2SGgmlScalarKernel {
    fn unpack_ggml_i2s(&self, packed: &[u8], num_blocks: usize) -> Vec<i8> {
        const QK_K: usize = 256;
        let mut unpacked = Vec::with_capacity(num_blocks * QK_K);

        for block_idx in 0..num_blocks {
            let block_start = block_idx * 64; // 64 bytes per block

            for byte_idx in 0..64 {
                let byte = packed[block_start + byte_idx];

                // Each byte contains 4 2-bit values (little-endian)
                for shift in (0..8).step_by(2) {
                    let bits = (byte >> shift) & 0b11;

                    // GGML signed symmetric mapping: 00→-2, 01→-1, 10→+1, 11→+2
                    let value = match bits {
                        0b00 => -2i8,
                        0b01 => -1i8,
                        0b10 => 1i8,
                        0b11 => 2i8,
                        _ => unreachable!(),
                    };

                    unpacked.push(value);
                }
            }
        }

        unpacked
    }

    fn dequantize_ggml_i2s(&self, packed: &[u8], scales: &[f32]) -> Result<Vec<f32>> {
        const QK_K: usize = 256;
        let num_blocks = scales.len();
        let unpacked = self.unpack_ggml_i2s(packed, num_blocks);

        let mut dequantized = Vec::with_capacity(num_blocks * QK_K);
        for block_idx in 0..num_blocks {
            let scale = scales[block_idx];
            let block_start = block_idx * QK_K;

            for elem_idx in 0..QK_K {
                let quantized = unpacked[block_start + elem_idx];
                dequantized.push((quantized as f32) * scale);
            }
        }

        Ok(dequantized)
    }

    fn matmul_i2s_ggml(
        &self,
        a: &[f32],
        b_packed: &[u8],
        b_scales: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        const QK_K: usize = 256;

        // Validate dimensions
        if a.len() != m * k {
            return Err(BitNetError::Kernel(KernelError::InvalidDimensions {
                reason: format!("A matrix size {} != m*k ({}*{})", a.len(), m, k),
            }));
        }

        let num_blocks_k = k.div_ceil(QK_K);
        let num_blocks_n = n.div_ceil(QK_K);

        if b_packed.len() != num_blocks_k * num_blocks_n * 64 {
            return Err(BitNetError::Kernel(KernelError::InvalidDimensions {
                reason: format!(
                    "B packed size {} != expected {}",
                    b_packed.len(),
                    num_blocks_k * num_blocks_n * 64
                ),
            }));
        }

        if b_scales.len() != num_blocks_k * num_blocks_n {
            return Err(BitNetError::Kernel(KernelError::InvalidDimensions {
                reason: format!(
                    "B scales length {} != num_blocks {}",
                    b_scales.len(),
                    num_blocks_k * num_blocks_n
                ),
            }));
        }

        // Dequantize B for reference implementation
        // (SIMD implementations would fuse unpack + scale + accumulate)
        let b_dequant = self.dequantize_ggml_i2s(b_packed, b_scales)?;

        // Standard matmul: C = A @ B
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b_dequant[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }

        Ok(())
    }
}
```

#### SIMD Optimizations (Phase 2.1)

```rust
/// AVX2 implementation (256-bit SIMD)
///
/// Process 32 bytes (128 elements) per iteration:
/// - Load 32 bytes packed data
/// - Unpack to 128 i8 values (4 values per byte)
/// - Convert to 128 f32 values (4x32 = 128)
/// - Multiply by broadcasted scale
/// - Accumulate into result
#[cfg(target_arch = "x86_64")]
pub struct I2SGgmlAvx2Kernel;

/// AVX-512 implementation (512-bit SIMD)
///
/// Process 64 bytes (256 elements = 1 block) per iteration:
/// - Perfect fit for QK_K block size
/// - Single pass per block
#[cfg(target_arch = "x86_64")]
pub struct I2SGgmlAvx512Kernel;

/// NEON implementation (128-bit SIMD)
///
/// Process 16 bytes (64 elements) per iteration:
/// - 4 iterations per 256-element block
#[cfg(target_arch = "aarch64")]
pub struct I2SGgmlNeonKernel;
```

### 5. Transformer Integration and Matmul Op Registration

#### QuantizedLinear Layer Extension

```rust
// In bitnet-inference/src/layers/quantized_linear.rs

impl QuantizedLinear {
    /// Detect flavor and route to appropriate kernel
    pub fn forward_i2s(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.weight.as_ref().ok_or_else(|| {
            BitNetError::Inference(InferenceError::MissingWeight {
                layer: "quantized_linear".to_string(),
            })
        })?;

        match weight {
            QuantTensor { flavor: I2SFlavor::BitNet32F16, .. } => {
                // Use existing I2_S kernel
                self.forward_i2s_bitnet(input, weight)
            }

            QuantTensor { flavor: I2SFlavor::GgmlQk256NoScale, scales: Some(_), .. } => {
                // Phase 2: Use ggml I2_S kernel
                self.forward_i2s_ggml(input, weight)
            }

            QuantTensor { flavor: I2SFlavor::GgmlQk256NoScale, scales: None, .. } => {
                // Production fail-closed: no scales, can't infer
                Err(BitNetError::Model(ModelError::InvalidFormat {
                    format: "GgmlQk256NoScale requires scale tensor for inference".to_string(),
                }))
            }
        }
    }

    /// BitNet native I2_S forward pass
    fn forward_i2s_bitnet(&self, input: &Tensor, weight: &QuantTensor) -> Result<Tensor> {
        // Existing implementation unchanged
        todo!("Use existing I2SQuantizer and bitnet_kernels matmul_i2s")
    }

    /// GGML I2_S forward pass (Phase 2)
    fn forward_i2s_ggml(&self, input: &Tensor, weight: &QuantTensor) -> Result<Tensor> {
        #[cfg(not(feature = "ggml-i2s-kernel"))]
        {
            // Fail-closed until Phase 2 kernel lands
            return Err(BitNetError::Kernel(KernelError::Unsupported {
                operation: "GgmlQk256NoScale matmul".to_string(),
                reason: "Phase 2 kernel not yet implemented (enable 'ggml-i2s-kernel' feature)".to_string(),
            }));
        }

        #[cfg(feature = "ggml-i2s-kernel")]
        {
            use bitnet_kernels::i2s_ggml::{I2SGgmlKernel, select_kernel};

            let kernel = select_kernel()?; // Device-aware selection

            // Extract input as f32 slice
            let input_f32 = input.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let [m, k] = input.dims2()?;
            let [k2, n] = weight.shape[..];

            if k != k2 {
                return Err(BitNetError::Kernel(KernelError::InvalidDimensions {
                    reason: format!("Input K={} != Weight K={}", k, k2),
                }));
            }

            // Allocate output
            let mut output = vec![0.0f32; m * n];

            // Matmul via ggml I2_S kernel
            kernel.matmul_i2s_ggml(
                &input_f32,
                &weight.data,
                weight.require_scales()?,
                &mut output,
                m, n, k,
            )?;

            // Convert back to Tensor
            Tensor::from_slice(&output, (m, n), input.device())
                .map_err(|e| BitNetError::Kernel(KernelError::InternalError {
                    source: e.to_string(),
                }))
        }
    }
}
```

### 6. Receipt Tracking for Dual-Flavor Validation

#### Receipt Schema Extension

```json
{
  "receipt_version": "1.1.0",
  "validation": {
    "backend": "rust | cpp",
    "crossval_source": "rust | cpp_ffi",
    "i2s_flavor_detected": "BitNet32F16 | GgmlQk256NoScale | mixed",
    "scale_tensor_present": true,
    "tensors": [
      {
        "name": "layers.0.attention.q_proj.weight",
        "qtype": "I2_S",
        "flavor": "GgmlQk256NoScale",
        "blocks": 256,
        "block_size": 256,
        "has_scales": true,
        "kernel_id": "i2s_ggml_avx512 | i2s_ggml_scalar | cpp_ffi"
      }
    ]
  },
  "compute_path": "real",
  "kernels": {
    "matmul_i2s_ggml": "i2s_ggml_avx512",
    "matmul_i2s_bitnet": "i2s_avx2"
  }
}
```

#### Receipt Generation

```rust
// In bitnet-inference/src/kernel_recorder.rs

impl KernelRecorder {
    /// Record I2_S kernel invocation with flavor tracking
    pub fn record_i2s_matmul(
        &mut self,
        flavor: I2SFlavor,
        kernel_id: &str,
        tensor_name: &str,
    ) {
        self.kernels.entry(format!("matmul_i2s_{:?}", flavor))
            .or_insert_with(Vec::new)
            .push(KernelInvocation {
                kernel_id: kernel_id.to_string(),
                tensor_name: tensor_name.to_string(),
                timestamp: std::time::Instant::now(),
            });
    }

    /// Finalize receipt with I2_S flavor metadata
    pub fn finalize_receipt(&self, model: &BitNetModel) -> InferenceReceipt {
        let i2s_tensors: Vec<_> = model.tensors.iter()
            .filter_map(|(name, tensor)| {
                if let QuantTensor { flavor, qtype: QuantizationType::I2S, .. } = tensor {
                    Some((name.clone(), *flavor))
                } else {
                    None
                }
            })
            .collect();

        let mixed_flavors = i2s_tensors.iter()
            .map(|(_, f)| f)
            .collect::<std::collections::HashSet<_>>()
            .len() > 1;

        let flavor_summary = if mixed_flavors {
            "mixed".to_string()
        } else if let Some((_, first_flavor)) = i2s_tensors.first() {
            format!("{:?}", first_flavor)
        } else {
            "none".to_string()
        };

        InferenceReceipt {
            receipt_version: "1.1.0".to_string(),
            validation: ValidationMetadata {
                backend: if cfg!(feature = "ffi") { "cpp" } else { "rust" },
                i2s_flavor_detected: flavor_summary,
                scale_tensor_present: i2s_tensors.iter()
                    .all(|(name, _)| model.has_scale_tensor(name)),
                tensors: i2s_tensors.into_iter()
                    .map(|(name, flavor)| TensorMetadata {
                        name,
                        flavor: format!("{:?}", flavor),
                        // ... other fields
                    })
                    .collect(),
            },
            // ... rest of receipt
        }
    }
}
```

## Testing Requirements

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flavor_detection_bitnet32f16() {
        // 32 elements → 1 block × 10 bytes
        assert_eq!(
            I2SFlavor::detect(32, 10).unwrap(),
            I2SFlavor::BitNet32F16
        );

        // 64 elements → 2 blocks × 10 bytes
        assert_eq!(
            I2SFlavor::detect(64, 20).unwrap(),
            I2SFlavor::BitNet32F16
        );
    }

    #[test]
    fn test_flavor_detection_ggml_qk256() {
        // 256 elements → 1 block × 64 bytes
        assert_eq!(
            I2SFlavor::detect(256, 64).unwrap(),
            I2SFlavor::GgmlQk256NoScale
        );

        // 512 elements → 2 blocks × 64 bytes
        assert_eq!(
            I2SFlavor::detect(512, 128).unwrap(),
            I2SFlavor::GgmlQk256NoScale
        );
    }

    #[test]
    fn test_flavor_detection_ambiguous() {
        // 32 elements: 10B (bitnet) vs 64B (ggml) → prefer BitNet32F16
        // This case is unlikely in practice
        // (would need total_elements where both formulas yield same size)
    }

    #[test]
    fn test_ggml_i2s_unpack_mapping() {
        let kernel = I2SGgmlScalarKernel;

        // Test 2-bit mapping: 00→-2, 01→-1, 10→+1, 11→+2
        let packed = [
            0b11_10_01_00, // First 4 values: -2, -1, +1, +2
            // ... fill rest to complete block
        ];

        let unpacked = kernel.unpack_ggml_i2s(&packed, 1);
        assert_eq!(unpacked[0], -2);
        assert_eq!(unpacked[1], -1);
        assert_eq!(unpacked[2], 1);
        assert_eq!(unpacked[3], 2);
    }

    #[test]
    fn test_ggml_i2s_dequantize() {
        let kernel = I2SGgmlScalarKernel;

        // Create packed data with known pattern
        let mut packed = vec![0u8; 64]; // 1 block
        packed[0] = 0b11_10_01_00; // -2, -1, +1, +2

        let scales = vec![2.0f32]; // Scale = 2.0
        let dequant = kernel.dequantize_ggml_i2s(&packed, &scales).unwrap();

        assert_eq!(dequant[0], -4.0); // -2 * 2.0
        assert_eq!(dequant[1], -2.0); // -1 * 2.0
        assert_eq!(dequant[2], 2.0);  // +1 * 2.0
        assert_eq!(dequant[3], 4.0);  // +2 * 2.0
    }
}
```

### Integration Tests (Phase 1)

```rust
#[cfg(all(test, feature = "ffi"))]
mod ffi_integration_tests {
    use super::*;

    #[test]
    fn test_ffi_session_lifecycle() {
        let model_path = env!("BITNET_GGUF"); // Requires test model

        // Create session
        let session = I2SFfiSession::new(model_path).unwrap();

        // Evaluate multiple times (reuse context)
        let tokens1 = vec![1, 2, 3];
        let logits1 = session.eval_logits(&tokens1).unwrap();

        let tokens2 = vec![4, 5, 6];
        let logits2 = session.eval_logits(&tokens2).unwrap();

        // Different tokens should produce different logits
        assert_ne!(logits1, logits2);

        // Drop session (should not crash)
        drop(session);
    }

    #[test]
    fn test_parity_routing_ggml_i2s() {
        let model_path = "tests/fixtures/ggml_i2s_model.gguf";
        let tokens = vec![1, 2, 3, 4];

        // Should route to FFI when BITNET_CPP_DIR set
        std::env::set_var("BITNET_CPP_DIR", "/path/to/bitnet.cpp");

        let logits = eval_logits_once_for_parity(model_path, &tokens).unwrap();

        // Verify logits are reasonable
        assert!(logits.len() > 0);
        assert!(logits.iter().any(|&x| x.abs() > 1e-6));
    }
}
```

### Cross-Validation Tests (Phase 2)

```rust
#[cfg(all(test, feature = "crossval"))]
mod crossval_tests {
    use super::*;

    #[test]
    fn test_ggml_i2s_vs_cpp_reference() {
        let model_path = "tests/fixtures/ggml_i2s_model.gguf";
        let tokens = vec![1, 2, 3, 4];

        // Get C++ reference logits
        let cpp_session = I2SFfiSession::new(model_path).unwrap();
        let cpp_logits = cpp_session.eval_logits(&tokens).unwrap();

        // Get Rust kernel logits (Phase 2)
        let rust_logits = eval_logits_once(model_path, &tokens).unwrap();

        // Compare with tight tolerance (exact 2-bit mapping)
        let max_diff = cpp_logits.iter()
            .zip(&rust_logits)
            .map(|(c, r)| (c - r).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "Rust vs C++ divergence: max_diff={:.6e}",
            max_diff
        );
    }
}
```

## Implementation Roadmap

### Phase 1: FFI Session Wrapper (Week 1-2)

**Tasks:**

1. Implement `I2SFlavor` enum and detection logic
2. Create `QuantTensor` with opaque storage for GgmlQk256NoScale
3. Implement `I2SFfiSession` wrapper with session lifecycle
4. Integrate FFI routing in `eval_logits_once_for_parity`
5. Add receipt tracking for FFI compute path
6. Write integration tests for FFI session reuse

**Acceptance Criteria:**

- AC1: Flavor detection correctly identifies BitNet32F16 vs GgmlQk256NoScale from size
- AC2: FFI session prevents munmap_chunk() crashes across multiple evaluations
- AC3: Parity harness routes ggml I2_S to C++ when BITNET_CPP_DIR set
- AC4: Receipts track `backend=cpp_ffi` for FFI-routed inference
- AC5: Production code fail-closes on ggml I2_S (no accidental FFI routing)

### Phase 2: Pure-Rust Kernel Implementation (Week 3-6)

**Tasks:**

1. Implement scalar reference kernel (`I2SGgmlScalarKernel`)
2. Validate 2-bit mapping against C++ reference with unit tests
3. Implement AVX2 SIMD kernel (32-byte chunks, 128 elements/iteration)
4. Implement AVX-512 SIMD kernel (64-byte chunks, 256 elements/iteration)
5. Implement NEON SIMD kernel (16-byte chunks, 64 elements/iteration)
6. Add device-aware kernel selection (similar to existing I2_S)
7. Integrate `forward_i2s_ggml` into `QuantizedLinear`
8. Cross-validate Rust kernels vs C++ FFI with tight tolerance (<1e-4)

**Acceptance Criteria:**

- AC6: Scalar kernel matches C++ reference exactly (bit-level parity)
- AC7: SIMD kernels achieve 2-4x speedup over scalar baseline
- AC8: Cross-validation tests pass with max_diff < 1e-4
- AC9: Receipts track Rust kernel IDs (e.g., `i2s_ggml_avx512`)
- AC10: Production inference works with ggml I2_S models (no FFI required)

### Phase 3: Performance Optimization (Week 7-8)

**Tasks:**

1. Fused unpack-dequant-matmul kernel (avoid intermediate buffers)
2. Block-level parallelization with Rayon
3. Benchmark against C++ reference (latency, throughput, memory)
4. Profile and optimize hot paths
5. Add performance baselines to CI

**Acceptance Criteria:**

- AC11: Rust kernel latency within 10% of C++ reference
- AC12: Memory usage comparable to C++ (no excess allocations)
- AC13: Throughput baselines established and tracked in CI

## Risk Mitigation

### Risk 1: 2-bit Mapping Mismatch

**Mitigation:**

- Unit tests verify exact mapping: 00→-2, 01→-1, 10→+1, 11→+2
- Cross-validation against C++ reference with tight tolerance (<1e-4)
- Golden test fixtures with known bit patterns

### Risk 2: FFI Memory Leaks

**Mitigation:**

- Session-based design prevents repeated model load/free
- Rely on Rust Drop trait for C++ cleanup (avoid manual free)
- Valgrind testing for FFI integration tests

### Risk 3: SIMD Correctness

**Mitigation:**

- Scalar reference implementation as ground truth
- Property-based tests (QuickCheck) for SIMD vs scalar equivalence
- Cross-validation against C++ on random inputs

### Risk 4: Production Accidentally Uses FFI

**Mitigation:**

- `eval_logits_once` (production) fail-closes on ggml I2_S
- Only `eval_logits_once_for_parity` (parity harness) routes to FFI
- Explicit BITNET_CPP_DIR requirement for FFI (env var check)
- CI validates production builds reject ggml I2_S models

## Success Metrics

### Phase 1 Success

- FFI session reuse prevents crashes (100% pass rate on 10+ consecutive evaluations)
- Parity harness correctly routes ggml I2_S to C++ (80% code coverage)
- Receipts accurately track FFI compute path (100% receipt validation)

### Phase 2 Success

- Rust kernel achieves bit-level parity with C++ (max_diff < 1e-4)
- Production inference works with ggml I2_S models (no FFI dependency)
- SIMD kernels provide measurable speedup (2-4x over scalar)

### Phase 3 Success

- Rust kernel latency within 10% of C++ reference
- Memory usage comparable to C++ (no excess allocations)
- Performance baselines established and stable in CI

## References

### GGML Implementation

- [ggml/src/ggml-quants.c](https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.c) - Reference quantization code
- QK_K = 256 (block size constant)
- iq2_xxs/iq2_xs/iq2_s quantization variants

### BitNet.cpp Reference

- [BitNet.cpp repository](https://github.com/microsoft/BitNet) - C++ implementation
- I2_S format with split scales (external tensor)

### BitNet.rs Existing Patterns

- `bitnet-quantization/src/i2s.rs` - BitNet32F16 implementation
- `bitnet-kernels/src/cpu/` - SIMD kernel patterns (AVX2, NEON)
- `bitnet-inference/src/parity.rs` - Parity harness FFI routing

## Appendix: Size Calculation Examples

### BitNet32F16

```text
Elements: 1024
Block size: 32
Num blocks: ceil(1024/32) = 32
Bytes per block: 8 (packed) + 2 (f16 scale) = 10
Total size: 32 * 10 = 320 bytes
```

### GgmlQk256NoScale

```text
Elements: 1024
Block size: 256 (QK_K)
Num blocks: ceil(1024/256) = 4
Bytes per block: 64 (packed data only)
Total size: 4 * 64 = 256 bytes
Scales: 4 f32 values in separate tensor (16 bytes)
```

### Edge Case: Ambiguous Size (32 elements)

```text
Elements: 32

BitNet32F16:
  Blocks: ceil(32/32) = 1
  Size: 1 * 10 = 10 bytes

GgmlQk256NoScale:
  Blocks: ceil(32/256) = 1
  Size: 1 * 64 = 64 bytes

No ambiguity: different sizes (10 vs 64)
```

## Appendix: Kernel ID Naming Conventions

Following ADR-012 kernel ID naming conventions:

```text
Rust Kernels:
- i2s_ggml_scalar       (scalar reference)
- i2s_ggml_avx2         (AVX2 SIMD)
- i2s_ggml_avx512       (AVX-512 SIMD)
- i2s_ggml_neon         (NEON SIMD)

FFI Kernels:
- cpp_ffi               (C++ via bitnet-sys)

Existing BitNet Kernels:
- i2s_avx2              (BitNet32F16 AVX2)
- i2s_neon              (BitNet32F16 NEON)
- i2s_scalar            (BitNet32F16 scalar)
```

---

**Document Control:**

- Review Status: Draft
- Next Review Date: 2025-10-24
- Owner: BitNet.rs Architecture Team
