//! Optimized quantized GEMM for 1-bit and 2-bit weight matrices.
//!
//! # Overview
//!
//! This module provides a comprehensive quantized General Matrix Multiply
//! (GEMM) implementation optimized for BitNet inference.  Unlike
//! [`super::quantized_matmul`] which focuses on I2_S fused
//! dequantize-matmul, this module adds:
//!
//! - **Multiple precision**: INT2 (ternary), INT4, INT8 weight formats
//! - **Tiled GEMM**: Shared-memory tiling for optimal GPU occupancy
//! - **Split-K / Stream-K**: Parallel reduction along K for tall-skinny
//!   matrices
//! - **Tensor-core awareness**: Config hints for WMMA tile sizes
//! - **Mixed-precision accumulation**: Low-precision inputs with FP32
//!   accumulator
//! - **Batched GEMM**: Multi-head attention and batch inference
//! - **Auto-tuning**: Heuristic tile/split selection per problem shape
//!
//! # CPU fallback
//!
//! Every entry point has a pure-Rust CPU fallback so that tests pass
//! without GPU hardware.  The unified [`quantized_gemm`] dispatcher
//! tries the GPU path first and falls back transparently.
//!
//! # Feature gate
//!
//! GPU launch stubs and CUDA kernel sources are behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Quantization type enum ────────────────────────────────────────────

/// Supported weight quantization bit-widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantType {
    /// 2-bit ternary: {-1, 0, +1} — BitNet native.
    INT2,
    /// 4-bit signed integer: [-8, +7].
    INT4,
    /// 8-bit signed integer: [-128, +127].
    INT8,
}

impl QuantType {
    /// Number of bits per weight element.
    pub fn bits(self) -> u32 {
        match self {
            Self::INT2 => 2,
            Self::INT4 => 4,
            Self::INT8 => 8,
        }
    }

    /// Number of weight elements packed per byte.
    pub fn elems_per_byte(self) -> usize {
        (8 / self.bits()) as usize
    }
}

// ── Accumulation type ─────────────────────────────────────────────────

/// Accumulation precision for mixed-precision GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccumulationType {
    /// 32-bit floating point accumulator.
    F32,
    /// 16-bit floating point accumulator (lower precision, higher
    /// throughput on tensor cores).
    F16,
}

// ── QuantGemmConfig ───────────────────────────────────────────────────

/// Configuration for a quantized GEMM operation.
///
/// Describes the matrix dimensions, quantization format, tiling
/// strategy, and optional tensor-core hints.
#[derive(Debug, Clone)]
pub struct QuantGemmConfig {
    /// Output rows (batch × sequence length).
    pub m: usize,
    /// Output columns (output hidden dimension).
    pub n: usize,
    /// Inner (reduction) dimension.
    pub k: usize,
    /// Weight quantization type.
    pub quant_type: QuantType,
    /// Tile size along M for shared-memory blocking.
    pub tile_m: u32,
    /// Tile size along N.
    pub tile_n: u32,
    /// Tile size along K.
    pub tile_k: u32,
    /// Accumulation data type.
    pub accumulation_type: AccumulationType,
    /// Whether to prefer tensor-core (WMMA) tile sizes.
    pub use_tensor_cores: bool,
    /// CUDA threads per block.
    pub threads_per_block: u32,
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
    /// Batch count (1 for non-batched).
    pub batch_size: usize,
    /// Scalar multiplier for the product (α).
    pub alpha: f32,
    /// Scalar multiplier for existing output (β).
    pub beta: f32,
    /// Number of K-partitions for split-K (1 = disabled).
    pub split_k: u32,
}

impl Default for QuantGemmConfig {
    fn default() -> Self {
        Self {
            m: 1,
            n: 1,
            k: 1,
            quant_type: QuantType::INT2,
            tile_m: 32,
            tile_n: 32,
            tile_k: 32,
            accumulation_type: AccumulationType::F32,
            use_tensor_cores: false,
            threads_per_block: 256,
            shared_mem_bytes: 8192,
            batch_size: 1,
            alpha: 1.0,
            beta: 0.0,
            split_k: 1,
        }
    }
}

impl QuantGemmConfig {
    /// Create a config for the given dimensions and quantization type.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn for_shape(m: usize, n: usize, k: usize, quant_type: QuantType) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "quantized GEMM dimensions must be non-zero: \
                     m={m}, n={n}, k={k}"
                ),
            }
            .into());
        }
        let (tile_m, tile_n, tile_k) = default_tile_for_quant(quant_type);
        let shared = estimate_shared_mem(tile_m, tile_n, tile_k, quant_type);
        Ok(Self {
            m,
            n,
            k,
            quant_type,
            tile_m,
            tile_n,
            tile_k,
            shared_mem_bytes: shared,
            ..Self::default()
        })
    }

    /// Enable tensor-core tile sizes (16×16×16 for Volta/Ampere WMMA).
    pub fn with_tensor_cores(mut self, enable: bool) -> Self {
        self.use_tensor_cores = enable;
        if enable {
            self.tile_m = 16;
            self.tile_n = 16;
            self.tile_k = 16;
            self.shared_mem_bytes = estimate_shared_mem(16, 16, 16, self.quant_type);
        }
        self
    }

    /// Set accumulation type.
    pub fn with_accumulation(mut self, acc: AccumulationType) -> Self {
        self.accumulation_type = acc;
        self
    }

    /// Set batch size.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch_size` is zero.
    pub fn with_batch_size(mut self, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "batch_size must be > 0".into() }.into()
            );
        }
        self.batch_size = batch_size;
        Ok(self)
    }

    /// Set alpha / beta scalars.
    pub fn with_alpha_beta(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Set split-K factor.
    ///
    /// # Errors
    ///
    /// Returns an error if `splits` is zero.
    pub fn with_split_k(mut self, splits: u32) -> Result<Self> {
        if splits == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "split_k must be > 0".into() }.into()
            );
        }
        self.split_k = splits;
        Ok(self)
    }

    /// Set custom tile sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if any tile dimension is zero.
    pub fn with_tiles(mut self, tile_m: u32, tile_n: u32, tile_k: u32) -> Result<Self> {
        if tile_m == 0 || tile_n == 0 || tile_k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "tile dimensions must be non-zero: \
                     tile_m={tile_m}, tile_n={tile_n}, tile_k={tile_k}"
                ),
            }
            .into());
        }
        self.tile_m = tile_m;
        self.tile_n = tile_n;
        self.tile_k = tile_k;
        self.shared_mem_bytes = estimate_shared_mem(tile_m, tile_n, tile_k, self.quant_type);
        Ok(self)
    }

    /// Compute the CUDA grid dimensions.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let grid_x = (self.n as u32).div_ceil(self.tile_n);
        let grid_y = (self.m as u32).div_ceil(self.tile_m);
        (grid_x, grid_y, self.batch_size as u32)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }

    /// Packed weight bytes needed for the K dimension of one column.
    pub fn packed_k_bytes(&self) -> usize {
        self.k.div_ceil(self.quant_type.elems_per_byte())
    }

    /// Total packed weight buffer size in bytes (all columns).
    pub fn weight_buffer_size(&self) -> usize {
        self.packed_k_bytes() * self.n
    }

    /// Estimated GFLOPS for this configuration at a given duration.
    pub fn gflops(&self, duration_secs: f64) -> f64 {
        let ops = 2.0 * self.m as f64 * self.n as f64 * self.k as f64 * self.batch_size as f64;
        ops / (duration_secs * 1e9)
    }
}

// ── Tile heuristics ───────────────────────────────────────────────────

fn default_tile_for_quant(qt: QuantType) -> (u32, u32, u32) {
    match qt {
        QuantType::INT2 => (32, 64, 32),
        QuantType::INT4 => (32, 32, 32),
        QuantType::INT8 => (32, 32, 32),
    }
}

fn estimate_shared_mem(tile_m: u32, tile_n: u32, tile_k: u32, _qt: QuantType) -> u32 {
    // Activation tile (f32) + weight tile (f32 after dequant)
    let act_bytes = tile_m * tile_k * 4;
    let wt_bytes = tile_k * tile_n * 4;
    (act_bytes + wt_bytes).max(4096)
}

// ── Packing / unpacking helpers ───────────────────────────────────────

/// Pack `elems` signed values into bytes for a given [`QuantType`].
///
/// For INT2: 4 values per byte, LSB-first, codes: 0→0b00, +1→0b01,
/// -1→0b11.
/// For INT4: 2 values per byte, LSB-first, two's complement nibble.
/// For INT8: 1 value per byte.
pub fn pack_weights(values: &[i8], quant_type: QuantType) -> Vec<u8> {
    match quant_type {
        QuantType::INT2 => {
            let packed_len = values.len().div_ceil(4);
            let mut out = vec![0u8; packed_len];
            for (i, &v) in values.iter().enumerate() {
                let code: u8 = match v {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                out[i / 4] |= code << ((i % 4) * 2);
            }
            out
        }
        QuantType::INT4 => {
            let packed_len = values.len().div_ceil(2);
            let mut out = vec![0u8; packed_len];
            for (i, &v) in values.iter().enumerate() {
                let nibble = (v as u8) & 0x0F;
                out[i / 2] |= nibble << ((i % 2) * 4);
            }
            out
        }
        QuantType::INT8 => values.iter().map(|&v| v as u8).collect(),
    }
}

/// Unpack one element from a packed weight byte slice.
#[inline(always)]
fn unpack_weight(packed: &[u8], index: usize, qt: QuantType) -> f32 {
    match qt {
        QuantType::INT2 => {
            let byte = packed[index / 4];
            let shift = (index % 4) * 2;
            let bits = (byte >> shift) & 0x03;
            match bits {
                0b01 => 1.0,
                0b11 => -1.0,
                _ => 0.0,
            }
        }
        QuantType::INT4 => {
            let byte = packed[index / 2];
            let shift = (index % 2) * 4;
            let nibble = (byte >> shift) & 0x0F;
            // Sign-extend from 4 bits
            let signed = if nibble & 0x08 != 0 { nibble as i8 | !0x0Fi8 } else { nibble as i8 };
            signed as f32
        }
        QuantType::INT8 => packed[index] as i8 as f32,
    }
}

// ── Validation ────────────────────────────────────────────────────────

fn validate_buffers(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &[f32],
    cfg: &QuantGemmConfig,
) -> Result<()> {
    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;
    let batch = cfg.batch_size;
    let packed_k = cfg.packed_k_bytes();

    if activations.len() < batch * m * k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "activations too small: expected {}, got {}",
                batch * m * k,
                activations.len()
            ),
        }));
    }
    if weights_packed.len() < batch * packed_k * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "weights_packed too small: expected {}, got {}",
                batch * packed_k * n,
                weights_packed.len()
            ),
        }));
    }
    if scales.len() < batch * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("scales too small: expected >= {}, got {}", batch * n, scales.len()),
        }));
    }
    if out.len() < batch * m * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("output too small: expected {}, got {}", batch * m * n, out.len()),
        }));
    }
    Ok(())
}

// ── Core CPU GEMM (generic over quant type) ───────────────────────────

/// CPU reference implementation for quantized GEMM.
///
/// Computes `C = α · A · dequant(W_packed, scales) + β · C` per batch.
///
/// # Layout
/// - `activations`: row-major `[batch, m, k]` f32
/// - `weights_packed`: packed weights `[batch, packed_k, n]`
///   column-major per column
/// - `scales`: per-column scale `[batch, n]`
/// - `out`: row-major `[batch, m, n]` f32
fn gemm_cpu_inner(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    cfg: &QuantGemmConfig,
) -> Result<()> {
    validate_buffers(activations, weights_packed, scales, out, cfg)?;

    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;
    let qt = cfg.quant_type;
    let packed_k = cfg.packed_k_bytes();
    let alpha = cfg.alpha;
    let beta = cfg.beta;
    let batch = cfg.batch_size;

    let a_stride = m * k;
    let w_stride = packed_k * n;
    let s_stride = n;
    let o_stride = m * n;

    for b in 0..batch {
        let a_off = b * a_stride;
        let w_off = b * w_stride;
        let s_off = b * s_stride;
        let o_off = b * o_stride;

        for row in 0..m {
            for col in 0..n {
                let scale = scales[s_off + col];
                let mut acc = 0.0f32;
                for l in 0..k {
                    let a_val = activations[a_off + row * k + l];
                    let w_idx = col * packed_k;
                    let w_val = unpack_weight(&weights_packed[w_off + w_idx..], l, qt);
                    acc += a_val * w_val;
                }
                let idx = o_off + row * n + col;
                let val = alpha * acc * scale;
                out[idx] = if beta == 0.0 { val } else { val + beta * out[idx] };
            }
        }
    }
    Ok(())
}

// ── Public CPU entry points ───────────────────────────────────────────

/// INT2 (ternary) GEMM — CPU fallback.
pub fn int2_gemm(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    cfg: &QuantGemmConfig,
) -> Result<()> {
    if cfg.quant_type != QuantType::INT2 {
        return Err(KernelError::InvalidArguments {
            reason: format!("int2_gemm requires INT2 quant type, got {:?}", cfg.quant_type),
        }
        .into());
    }
    gemm_cpu_inner(activations, weights_packed, scales, out, cfg)
}

/// INT4 GEMM — CPU fallback.
pub fn int4_gemm(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    cfg: &QuantGemmConfig,
) -> Result<()> {
    if cfg.quant_type != QuantType::INT4 {
        return Err(KernelError::InvalidArguments {
            reason: format!("int4_gemm requires INT4 quant type, got {:?}", cfg.quant_type),
        }
        .into());
    }
    gemm_cpu_inner(activations, weights_packed, scales, out, cfg)
}

/// INT8 GEMM — CPU fallback.
pub fn int8_gemm(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    cfg: &QuantGemmConfig,
) -> Result<()> {
    if cfg.quant_type != QuantType::INT8 {
        return Err(KernelError::InvalidArguments {
            reason: format!("int8_gemm requires INT8 quant type, got {:?}", cfg.quant_type),
        }
        .into());
    }
    gemm_cpu_inner(activations, weights_packed, scales, out, cfg)
}

/// Mixed-precision GEMM with configurable accumulation type.
///
/// On CPU this always accumulates in f32 regardless of the
/// [`AccumulationType`] hint (the hint is used by the GPU path).
pub fn mixed_precision_gemm(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    cfg: &QuantGemmConfig,
) -> Result<()> {
    gemm_cpu_inner(activations, weights_packed, scales, out, cfg)
}

/// Batched quantized GEMM for multi-head attention.
///
/// Each batch element has independent weight / activation / scale
/// buffers laid out contiguously.
pub fn batched_quantized_gemm(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    cfg: &QuantGemmConfig,
) -> Result<()> {
    if cfg.batch_size < 1 {
        return Err(KernelError::InvalidArguments {
            reason: "batched_quantized_gemm requires batch_size >= 1".into(),
        }
        .into());
    }
    gemm_cpu_inner(activations, weights_packed, scales, out, cfg)
}

/// Tiled GEMM — CPU implementation with explicit tiling for cache
/// locality.
pub fn tiled_gemm(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    cfg: &QuantGemmConfig,
) -> Result<()> {
    validate_buffers(activations, weights_packed, scales, out, cfg)?;

    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;
    let qt = cfg.quant_type;
    let packed_k = cfg.packed_k_bytes();
    let alpha = cfg.alpha;
    let beta = cfg.beta;
    let tile_m = cfg.tile_m as usize;
    let tile_n = cfg.tile_n as usize;
    let tile_k = cfg.tile_k as usize;

    // Apply beta
    if beta == 0.0 {
        out[..m * n].fill(0.0);
    } else if (beta - 1.0).abs() > f32::EPSILON {
        for v in out[..m * n].iter_mut() {
            *v *= beta;
        }
    }

    let mut i0 = 0;
    while i0 < m {
        let i_end = (i0 + tile_m).min(m);
        let mut j0 = 0;
        while j0 < n {
            let j_end = (j0 + tile_n).min(n);
            let mut l0 = 0;
            while l0 < k {
                let l_end = (l0 + tile_k).min(k);
                for i in i0..i_end {
                    for j in j0..j_end {
                        let scale = scales[j];
                        let mut acc = 0.0f32;
                        for l in l0..l_end {
                            let a_val = activations[i * k + l];
                            let w_idx = j * packed_k;
                            let w_val = unpack_weight(&weights_packed[w_idx..], l, qt);
                            acc += a_val * w_val;
                        }
                        out[i * n + j] += alpha * acc * scale;
                    }
                }
                l0 += tile_k;
            }
            j0 += tile_n;
        }
        i0 += tile_m;
    }
    Ok(())
}

/// Split-K parallel GEMM for tall/skinny matrices.
///
/// Partitions the K dimension into `split_k` slices, computes partial
/// sums, then reduces.  On CPU this is serial but mirrors the GPU
/// algorithm for testing.
pub fn split_k_gemm(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    cfg: &QuantGemmConfig,
) -> Result<()> {
    validate_buffers(activations, weights_packed, scales, out, cfg)?;

    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;
    let qt = cfg.quant_type;
    let packed_k = cfg.packed_k_bytes();
    let alpha = cfg.alpha;
    let beta = cfg.beta;
    let splits = cfg.split_k.max(1) as usize;
    let k_per_split = k.div_ceil(splits);

    // Partial sums: [splits][m * n]
    let mut partials = vec![0.0f32; splits * m * n];

    for s in 0..splits {
        let k_start = s * k_per_split;
        let k_end = ((s + 1) * k_per_split).min(k);
        if k_start >= k {
            break;
        }
        for row in 0..m {
            for col in 0..n {
                let scale = scales[col];
                let mut acc = 0.0f32;
                for l in k_start..k_end {
                    let a_val = activations[row * k + l];
                    let w_idx = col * packed_k;
                    let w_val = unpack_weight(&weights_packed[w_idx..], l, qt);
                    acc += a_val * w_val;
                }
                partials[s * m * n + row * n + col] = acc * scale;
            }
        }
    }

    // Reduce partials
    for row in 0..m {
        for col in 0..n {
            let mut total = 0.0f32;
            for s in 0..splits {
                total += partials[s * m * n + row * n + col];
            }
            let idx = row * n + col;
            let val = alpha * total;
            out[idx] = if beta == 0.0 { val } else { val + beta * out[idx] };
        }
    }
    Ok(())
}

/// Stream-K scheduling for load-balanced GEMM.
///
/// Distributes work across a fixed number of "virtual processors",
/// each handling a contiguous range of output tiles.  On CPU this is
/// equivalent to the tiled path but exercises the stream-K partitioning
/// logic for correctness testing.
pub fn stream_k_gemm(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    cfg: &QuantGemmConfig,
) -> Result<()> {
    validate_buffers(activations, weights_packed, scales, out, cfg)?;

    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;
    let qt = cfg.quant_type;
    let packed_k = cfg.packed_k_bytes();
    let alpha = cfg.alpha;
    let beta = cfg.beta;
    let tile_m = cfg.tile_m as usize;
    let tile_n = cfg.tile_n as usize;

    let num_tiles_m = m.div_ceil(tile_m);
    let num_tiles_n = n.div_ceil(tile_n);
    let total_tiles = num_tiles_m * num_tiles_n;

    // Apply beta
    if beta == 0.0 {
        out[..m * n].fill(0.0);
    } else if (beta - 1.0).abs() > f32::EPSILON {
        for v in out[..m * n].iter_mut() {
            *v *= beta;
        }
    }

    // Stream-K: iterate tiles linearly (models stream scheduling)
    for tile_idx in 0..total_tiles {
        let tile_row = tile_idx / num_tiles_n;
        let tile_col = tile_idx % num_tiles_n;
        let i0 = tile_row * tile_m;
        let j0 = tile_col * tile_n;
        let i_end = (i0 + tile_m).min(m);
        let j_end = (j0 + tile_n).min(n);

        for i in i0..i_end {
            for j in j0..j_end {
                let scale = scales[j];
                let mut acc = 0.0f32;
                for l in 0..k {
                    let a_val = activations[i * k + l];
                    let w_idx = j * packed_k;
                    let w_val = unpack_weight(&weights_packed[w_idx..], l, qt);
                    acc += a_val * w_val;
                }
                out[i * n + j] += alpha * acc * scale;
            }
        }
    }
    Ok(())
}

// ── GemmWorkspace ─────────────────────────────────────────────────────

/// Pre-allocated workspace for GEMM operations.
///
/// Reusing a workspace avoids repeated heap allocation in the
/// inference hot-loop.  The workspace holds temporary buffers for
/// split-K partial sums and stream-K tile accumulation.
#[derive(Debug, Clone)]
pub struct GemmWorkspace {
    /// Temporary buffer for split-K partial sums.
    pub partials: Vec<f32>,
    /// Capacity in number of f32 elements.
    capacity: usize,
}

impl GemmWorkspace {
    /// Create a new workspace with capacity for the given config.
    pub fn new(cfg: &QuantGemmConfig) -> Self {
        let cap = cfg.split_k.max(1) as usize * cfg.m * cfg.n;
        Self { partials: vec![0.0; cap], capacity: cap }
    }

    /// Ensure the workspace has enough capacity for a new config.
    pub fn ensure_capacity(&mut self, cfg: &QuantGemmConfig) {
        let needed = cfg.split_k.max(1) as usize * cfg.m * cfg.n;
        if needed > self.capacity {
            self.partials.resize(needed, 0.0);
            self.capacity = needed;
        }
    }

    /// Current capacity in f32 elements.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Reset all values to zero.
    pub fn reset(&mut self) {
        self.partials.fill(0.0);
    }
}

// ── GemmAutoTuner ─────────────────────────────────────────────────────

/// Heuristic autotuner for quantized GEMM parameters.
///
/// Selects tile sizes, split-K factor, and tensor-core hints based on
/// problem shape and quantization type.
#[derive(Debug, Clone)]
pub struct GemmAutoTuner {
    /// Maximum shared memory per block (bytes).
    pub max_shared_mem: u32,
    /// Number of SMs on the target device.
    pub num_sms: u32,
}

impl Default for GemmAutoTuner {
    fn default() -> Self {
        // Conservative defaults (Ampere A100-like)
        Self { max_shared_mem: 49152, num_sms: 108 }
    }
}

impl GemmAutoTuner {
    /// Create a tuner with known device parameters.
    pub fn new(max_shared_mem: u32, num_sms: u32) -> Self {
        Self { max_shared_mem, num_sms }
    }

    /// Produce an optimised [`QuantGemmConfig`] for the given problem.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimensions are zero.
    pub fn tune(
        &self,
        m: usize,
        n: usize,
        k: usize,
        quant_type: QuantType,
    ) -> Result<QuantGemmConfig> {
        let mut cfg = QuantGemmConfig::for_shape(m, n, k, quant_type)?;

        // Use tensor cores for INT8 when dimensions are multiples of 16
        if quant_type == QuantType::INT8
            && m.is_multiple_of(16)
            && n.is_multiple_of(16)
            && k.is_multiple_of(16)
        {
            cfg = cfg.with_tensor_cores(true);
        }

        // Split-K for tall-skinny (M >> N) or deep-K matrices
        if k >= 2048 && m * n < 4096 {
            let splits = (k as u32 / 512).clamp(2, 16);
            cfg = cfg.with_split_k(splits)?;
        }

        // Clamp shared memory
        if cfg.shared_mem_bytes > self.max_shared_mem {
            let smaller = self.max_shared_mem / 2;
            let side = (smaller / 8).max(16); // rough
            cfg.tile_m = side.min(cfg.tile_m);
            cfg.tile_n = side.min(cfg.tile_n);
            cfg.tile_k = side.min(cfg.tile_k);
            cfg.shared_mem_bytes =
                estimate_shared_mem(cfg.tile_m, cfg.tile_n, cfg.tile_k, quant_type);
        }

        Ok(cfg)
    }
}

// ── Benchmark helper ──────────────────────────────────────────────────

/// Measure GFLOPS for a given quantized GEMM configuration.
///
/// Runs `iterations` CPU GEMM invocations and returns average GFLOPS.
pub fn benchmark_gemm(cfg: &QuantGemmConfig, iterations: usize) -> f64 {
    let m = cfg.m;
    let n = cfg.n;
    let k = cfg.k;
    let packed_k = cfg.packed_k_bytes();

    let activations = vec![1.0f32; m * k];
    let weights_packed = vec![0u8; packed_k * n];
    let scales = vec![1.0f32; n];
    let mut out = vec![0.0f32; m * n];

    let iters = iterations.max(1);
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = gemm_cpu_inner(&activations, &weights_packed, &scales, &mut out, cfg);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let avg_secs = elapsed / iters as f64;
    cfg.gflops(avg_secs)
}

// ── Unified dispatch ──────────────────────────────────────────────────

/// CUDA kernel source for quantized GEMM (INT2/INT4/INT8).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const QUANTIZED_GEMM_KERNEL_SRC: &str = r#"
extern "C" __global__ void quantized_gemm_f32(
    const float* __restrict__ activations,
    const unsigned char* __restrict__ weights_packed,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int M, int N, int K, int quant_bits, float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int elems_per_byte = 8 / quant_bits;
    int packed_k = (K + elems_per_byte - 1) / elems_per_byte;
    float scale = scales[col];
    float acc = 0.0f;

    for (int idx = 0; idx < K; idx++) {
        int byte_idx = col * packed_k + idx / elems_per_byte;
        int bit_off  = (idx % elems_per_byte) * quant_bits;
        int mask     = (1 << quant_bits) - 1;
        unsigned char bits = (weights_packed[byte_idx] >> bit_off) & mask;

        float w;
        if (quant_bits == 2) {
            if      (bits == 0x01) w =  1.0f;
            else if (bits == 0x03) w = -1.0f;
            else                   w =  0.0f;
        } else if (quant_bits == 4) {
            int s = (int)bits;
            if (s & 0x08) s |= ~0x0F;
            w = (float)s;
        } else {
            w = (float)((char)bits);
        }

        acc += activations[row * K + idx] * w;
    }
    output[row * N + col] = alpha * acc * scale + beta * output[row * N + col];
}
"#;

/// Launch stub for the quantized GEMM CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_quantized_gemm(
    _activations: &[f32],
    _weights_packed: &[u8],
    _scales: &[f32],
    _output: &mut [f32],
    config: &QuantGemmConfig,
) -> Result<()> {
    log::debug!(
        "quantized GEMM CUDA stub: m={}, n={}, k={}, qt={:?}, grid={:?}",
        config.m,
        config.n,
        config.k,
        config.quant_type,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "quantized GEMM CUDA kernel not yet compiled \
                 — scaffold only"
            .into(),
    }
    .into())
}

/// Main GEMM entry point: dispatches by quant type, GPU-first with CPU
/// fallback.
pub fn quantized_gemm(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    output: &mut [f32],
    config: &QuantGemmConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && launch_quantized_gemm(activations, weights_packed, scales, output, config).is_ok()
        {
            return Ok(());
        }
    }
    gemm_cpu_inner(activations, weights_packed, scales, output, config)
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    /// Build packed col-major weight buffer + uniform scale=1.0.
    fn pack_col_major(weights: &[i8], k: usize, n: usize, qt: QuantType) -> (Vec<u8>, Vec<f32>) {
        let epb = qt.elems_per_byte();
        let packed_k = k.div_ceil(epb);
        let mut packed = vec![0u8; packed_k * n];
        for col in 0..n {
            for row in 0..k {
                let v = weights[row * n + col];
                let byte_idx = col * packed_k + row / epb;
                let shift = (row % epb) * qt.bits() as usize;
                let code: u8 = match qt {
                    QuantType::INT2 => match v {
                        1 => 0b01,
                        -1 => 0b11,
                        _ => 0b00,
                    },
                    QuantType::INT4 => (v as u8) & 0x0F,
                    QuantType::INT8 => v as u8,
                };
                packed[byte_idx] |= code << shift;
            }
        }
        let scales = vec![1.0f32; n];
        (packed, scales)
    }

    /// Naive f32 matmul reference.
    fn naive_matmul(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..k {
                    s += a[i * k + l] * w[l * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn run_all_paths(
        act: &[f32],
        packed: &[u8],
        scales: &[f32],
        cfg: &QuantGemmConfig,
        expected: &[f32],
        tol: f32,
    ) {
        let len = cfg.m * cfg.n * cfg.batch_size;

        // gemm_cpu_inner
        let mut out = vec![0.0f32; len];
        gemm_cpu_inner(act, packed, scales, &mut out, cfg).unwrap();
        assert_close(&out[..expected.len()], expected, tol);

        // unified dispatch
        let mut out2 = vec![0.0f32; len];
        quantized_gemm(act, packed, scales, &mut out2, cfg).unwrap();
        assert_close(&out2[..expected.len()], expected, tol);
    }

    // ── QuantType tests ───────────────────────────────────────────

    #[test]
    fn test_quant_type_bits() {
        assert_eq!(QuantType::INT2.bits(), 2);
        assert_eq!(QuantType::INT4.bits(), 4);
        assert_eq!(QuantType::INT8.bits(), 8);
    }

    #[test]
    fn test_quant_type_elems_per_byte() {
        assert_eq!(QuantType::INT2.elems_per_byte(), 4);
        assert_eq!(QuantType::INT4.elems_per_byte(), 2);
        assert_eq!(QuantType::INT8.elems_per_byte(), 1);
    }

    // ── QuantGemmConfig tests ─────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = QuantGemmConfig::default();
        assert_eq!(cfg.quant_type, QuantType::INT2);
        assert_eq!(cfg.accumulation_type, AccumulationType::F32);
        assert!(!cfg.use_tensor_cores);
        assert_eq!(cfg.batch_size, 1);
        assert_eq!(cfg.alpha, 1.0);
        assert_eq!(cfg.beta, 0.0);
        assert_eq!(cfg.split_k, 1);
    }

    #[test]
    fn test_config_for_shape() {
        let cfg = QuantGemmConfig::for_shape(4, 64, 128, QuantType::INT2).unwrap();
        assert_eq!(cfg.m, 4);
        assert_eq!(cfg.n, 64);
        assert_eq!(cfg.k, 128);
    }

    #[test]
    fn test_config_rejects_zero_dims() {
        assert!(QuantGemmConfig::for_shape(0, 8, 8, QuantType::INT2).is_err());
        assert!(QuantGemmConfig::for_shape(8, 0, 8, QuantType::INT2).is_err());
        assert!(QuantGemmConfig::for_shape(8, 8, 0, QuantType::INT2).is_err());
    }

    #[test]
    fn test_config_tensor_cores() {
        let cfg = QuantGemmConfig::for_shape(16, 16, 16, QuantType::INT8).unwrap();
        let cfg = cfg.with_tensor_cores(true);
        assert!(cfg.use_tensor_cores);
        assert_eq!(cfg.tile_m, 16);
        assert_eq!(cfg.tile_n, 16);
        assert_eq!(cfg.tile_k, 16);
    }

    #[test]
    fn test_config_batch_size() {
        let cfg = QuantGemmConfig::for_shape(4, 4, 4, QuantType::INT2).unwrap();
        let cfg = cfg.with_batch_size(8).unwrap();
        assert_eq!(cfg.batch_size, 8);
    }

    #[test]
    fn test_config_rejects_zero_batch() {
        let cfg = QuantGemmConfig::for_shape(4, 4, 4, QuantType::INT2).unwrap();
        assert!(cfg.with_batch_size(0).is_err());
    }

    #[test]
    fn test_config_split_k() {
        let cfg = QuantGemmConfig::for_shape(4, 4, 64, QuantType::INT2).unwrap();
        let cfg = cfg.with_split_k(4).unwrap();
        assert_eq!(cfg.split_k, 4);
    }

    #[test]
    fn test_config_rejects_zero_split_k() {
        let cfg = QuantGemmConfig::for_shape(4, 4, 4, QuantType::INT2).unwrap();
        assert!(cfg.with_split_k(0).is_err());
    }

    #[test]
    fn test_config_alpha_beta() {
        let cfg = QuantGemmConfig::for_shape(4, 4, 4, QuantType::INT2).unwrap();
        let cfg = cfg.with_alpha_beta(2.0, 0.5);
        assert_eq!(cfg.alpha, 2.0);
        assert_eq!(cfg.beta, 0.5);
    }

    #[test]
    fn test_config_tiles() {
        let cfg = QuantGemmConfig::for_shape(4, 4, 4, QuantType::INT2).unwrap();
        let cfg = cfg.with_tiles(16, 16, 8).unwrap();
        assert_eq!(cfg.tile_m, 16);
        assert_eq!(cfg.tile_n, 16);
        assert_eq!(cfg.tile_k, 8);
    }

    #[test]
    fn test_config_rejects_zero_tiles() {
        let cfg = QuantGemmConfig::for_shape(4, 4, 4, QuantType::INT2).unwrap();
        assert!(cfg.with_tiles(0, 16, 16).is_err());
        let cfg = QuantGemmConfig::for_shape(4, 4, 4, QuantType::INT2).unwrap();
        assert!(cfg.with_tiles(16, 0, 16).is_err());
        let cfg = QuantGemmConfig::for_shape(4, 4, 4, QuantType::INT2).unwrap();
        assert!(cfg.with_tiles(16, 16, 0).is_err());
    }

    #[test]
    fn test_config_grid_dim() {
        let cfg = QuantGemmConfig::for_shape(128, 256, 64, QuantType::INT2).unwrap();
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, (256u32).div_ceil(cfg.tile_n));
        assert_eq!(gy, (128u32).div_ceil(cfg.tile_m));
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_config_accumulation() {
        let cfg = QuantGemmConfig::for_shape(4, 4, 4, QuantType::INT2).unwrap();
        let cfg = cfg.with_accumulation(AccumulationType::F16);
        assert_eq!(cfg.accumulation_type, AccumulationType::F16);
    }

    #[test]
    fn test_config_packed_k_bytes() {
        // INT2: k=8 → 8/4 = 2 bytes
        let cfg = QuantGemmConfig::for_shape(1, 1, 8, QuantType::INT2).unwrap();
        assert_eq!(cfg.packed_k_bytes(), 2);
        // INT4: k=8 → 8/2 = 4 bytes
        let cfg = QuantGemmConfig::for_shape(1, 1, 8, QuantType::INT4).unwrap();
        assert_eq!(cfg.packed_k_bytes(), 4);
        // INT8: k=8 → 8 bytes
        let cfg = QuantGemmConfig::for_shape(1, 1, 8, QuantType::INT8).unwrap();
        assert_eq!(cfg.packed_k_bytes(), 8);
    }

    #[test]
    fn test_config_weight_buffer_size() {
        let cfg = QuantGemmConfig::for_shape(1, 4, 8, QuantType::INT2).unwrap();
        // packed_k = 2, n = 4 → 8
        assert_eq!(cfg.weight_buffer_size(), 8);
    }

    #[test]
    fn test_config_gflops() {
        let cfg = QuantGemmConfig::for_shape(64, 64, 64, QuantType::INT2).unwrap();
        let gf = cfg.gflops(1.0); // 1 second
        let expected = 2.0 * 64.0 * 64.0 * 64.0 / 1e9;
        assert!((gf - expected).abs() < 1e-6);
    }

    // ── pack_weights round-trip ───────────────────────────────────

    #[test]
    fn test_pack_int2_roundtrip() {
        let vals: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 1, 1];
        let packed = pack_weights(&vals, QuantType::INT2);
        for (i, &expected) in vals.iter().enumerate() {
            let got = unpack_weight(&packed, i, QuantType::INT2);
            assert_eq!(got, expected as f32, "INT2 mismatch at {i}");
        }
    }

    #[test]
    fn test_pack_int4_roundtrip() {
        let vals: Vec<i8> = vec![0, 1, -1, 7, -8, 3, -4, 5];
        let packed = pack_weights(&vals, QuantType::INT4);
        for (i, &expected) in vals.iter().enumerate() {
            let got = unpack_weight(&packed, i, QuantType::INT4);
            assert_eq!(got, expected as f32, "INT4 mismatch at {i}");
        }
    }

    #[test]
    fn test_pack_int8_roundtrip() {
        let vals: Vec<i8> = vec![0, 1, -1, 127, -128, 42, -42, 100];
        let packed = pack_weights(&vals, QuantType::INT8);
        for (i, &expected) in vals.iter().enumerate() {
            let got = unpack_weight(&packed, i, QuantType::INT8);
            assert_eq!(got, expected as f32, "INT8 mismatch at {i}");
        }
    }

    // ── INT2 GEMM correctness ─────────────────────────────────────

    #[test]
    fn test_int2_identity_2x2() {
        let (m, n, k) = (2, 2, 2);
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![3.0f32, -2.0, 5.0, 7.0];
        let expected = naive_matmul(&act, &[1.0, 0.0, 0.0, 1.0], m, n, k);
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();
        run_all_paths(&act, &packed, &scales, &cfg, &expected, 1e-6);
    }

    #[test]
    fn test_int2_all_ones_4x4() {
        let (m, n, k) = (4, 4, 4);
        let w = vec![1i8; k * n];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();
        run_all_paths(&act, &packed, &scales, &cfg, &expected, 1e-5);
    }

    #[test]
    fn test_int2_all_neg_ones() {
        let (m, n, k) = (3, 3, 4);
        let w = vec![-1i8; k * n];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![1.0f32; m * k];
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();
        run_all_paths(&act, &packed, &scales, &cfg, &expected, 1e-5);
    }

    #[test]
    fn test_int2_mixed_ternary() {
        let (m, n, k) = (4, 4, 8);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();
        run_all_paths(&act, &packed, &scales, &cfg, &expected, 1e-4);
    }

    #[test]
    fn test_int2_zero_weights() {
        let (m, n, k) = (4, 4, 8);
        let w = vec![0i8; k * n];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![42.0f32; m * k];
        let expected = vec![0.0f32; m * n];
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();
        run_all_paths(&act, &packed, &scales, &cfg, &expected, 1e-6);
    }

    #[test]
    fn test_int2_gemm_type_mismatch() {
        let cfg = QuantGemmConfig::for_shape(2, 2, 2, QuantType::INT4).unwrap();
        let mut out = vec![0.0f32; 4];
        assert!(int2_gemm(&[1.0; 4], &[0; 2], &[1.0; 2], &mut out, &cfg).is_err());
    }

    #[test]
    fn test_int2_1x1() {
        let cfg = QuantGemmConfig::for_shape(1, 1, 1, QuantType::INT2).unwrap();
        let w: Vec<i8> = vec![1];
        let (packed, scales) = pack_col_major(&w, 1, 1, QuantType::INT2);
        let act = vec![7.5f32];
        let mut out = vec![0.0f32; 1];
        quantized_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        assert_close(&out, &[7.5], 1e-6);
    }

    // ── INT4 GEMM correctness ─────────────────────────────────────

    #[test]
    fn test_int4_identity_2x2() {
        let (m, n, k) = (2, 2, 2);
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT4);
        let act = vec![3.0f32, -2.0, 5.0, 7.0];
        let expected = naive_matmul(&act, &[1.0, 0.0, 0.0, 1.0], m, n, k);
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT4).unwrap();
        run_all_paths(&act, &packed, &scales, &cfg, &expected, 1e-6);
    }

    #[test]
    fn test_int4_range_values() {
        let (m, n, k) = (2, 2, 4);
        let w: Vec<i8> = vec![7, -8, 3, -4, 1, -1, 0, 5];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT4);
        let act = vec![1.0f32; m * k];
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT4).unwrap();
        run_all_paths(&act, &packed, &scales, &cfg, &expected, 1e-5);
    }

    #[test]
    fn test_int4_gemm_type_mismatch() {
        let cfg = QuantGemmConfig::for_shape(2, 2, 2, QuantType::INT2).unwrap();
        let mut out = vec![0.0f32; 4];
        assert!(int4_gemm(&[1.0; 4], &[0; 2], &[1.0; 2], &mut out, &cfg).is_err());
    }

    // ── INT8 GEMM correctness ─────────────────────────────────────

    #[test]
    fn test_int8_identity_2x2() {
        let (m, n, k) = (2, 2, 2);
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT8);
        let act = vec![3.0f32, -2.0, 5.0, 7.0];
        let expected = naive_matmul(&act, &[1.0, 0.0, 0.0, 1.0], m, n, k);
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT8).unwrap();
        run_all_paths(&act, &packed, &scales, &cfg, &expected, 1e-6);
    }

    #[test]
    fn test_int8_full_range() {
        let (m, n, k) = (2, 2, 4);
        let w: Vec<i8> = vec![127, -128, 42, -42, 1, -1, 0, 100];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT8);
        let act = vec![1.0f32; m * k];
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT8).unwrap();
        run_all_paths(&act, &packed, &scales, &cfg, &expected, 1e-4);
    }

    #[test]
    fn test_int8_gemm_type_mismatch() {
        let cfg = QuantGemmConfig::for_shape(2, 2, 2, QuantType::INT2).unwrap();
        let mut out = vec![0.0f32; 4];
        assert!(int8_gemm(&[1.0; 4], &[0; 2], &[1.0; 2], &mut out, &cfg).is_err());
    }

    // ── mixed_precision_gemm ──────────────────────────────────────

    #[test]
    fn test_mixed_precision_f32() {
        let (m, n, k) = (2, 2, 4);
        let w: Vec<i8> = vec![1; k * n];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![1.0f32; m * k];
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2)
            .unwrap()
            .with_accumulation(AccumulationType::F32);
        let mut out = vec![0.0f32; m * n];
        mixed_precision_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        // Each element: sum of k ones = 4.0
        assert_close(&out, &[4.0, 4.0, 4.0, 4.0], 1e-6);
    }

    #[test]
    fn test_mixed_precision_f16_hint() {
        let (m, n, k) = (2, 2, 4);
        let w: Vec<i8> = vec![1; k * n];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![1.0f32; m * k];
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2)
            .unwrap()
            .with_accumulation(AccumulationType::F16);
        let mut out = vec![0.0f32; m * n];
        mixed_precision_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        assert_close(&out, &[4.0, 4.0, 4.0, 4.0], 1e-6);
    }

    // ── batched_quantized_gemm ────────────────────────────────────

    #[test]
    fn test_batched_2_batches() {
        let (m, n, k): (usize, usize, usize) = (2, 2, 2);
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let qt = QuantType::INT2;
        let epb = qt.elems_per_byte();
        let packed_k = k.div_ceil(epb);

        // Pack weights for 2 batches (same weights)
        let (p1, _) = pack_col_major(&w, k, n, qt);
        let mut packed = Vec::new();
        packed.extend_from_slice(&p1);
        packed.extend_from_slice(&p1);
        let scales = vec![1.0f32; 2 * n];

        let act = vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ];
        let cfg = QuantGemmConfig::for_shape(m, n, k, qt).unwrap().with_batch_size(2).unwrap();

        let mut out = vec![0.0f32; 2 * m * n];
        batched_quantized_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();

        // Identity weights → output = input
        assert_close(&out[0..4], &[1.0, 2.0, 3.0, 4.0], 1e-6);
        assert_close(&out[4..8], &[5.0, 6.0, 7.0, 8.0], 1e-6);

        // Suppress unused variable warning
        let _ = packed_k;
    }

    #[test]
    fn test_batched_rejects_zero_batch() {
        let cfg = QuantGemmConfig {
            batch_size: 0,
            ..QuantGemmConfig::for_shape(2, 2, 2, QuantType::INT2).unwrap()
        };
        let mut out = vec![0.0f32; 4];
        assert!(batched_quantized_gemm(&[1.0; 4], &[0; 2], &[1.0; 2], &mut out, &cfg,).is_err());
    }

    // ── tiled_gemm ────────────────────────────────────────────────

    #[test]
    fn test_tiled_identity_4x4() {
        let (m, n, k) = (4, 4, 4);
        let w: Vec<i8> = vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; m * n];
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();
        tiled_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        assert_close(&out, &act, 1e-6);
    }

    #[test]
    fn test_tiled_matches_reference() {
        let (m, n, k) = (8, 6, 12);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);

        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();
        let mut out_tiled = vec![0.0f32; m * n];
        tiled_gemm(&act, &packed, &scales, &mut out_tiled, &cfg).unwrap();

        let mut out_ref = vec![0.0f32; m * n];
        gemm_cpu_inner(&act, &packed, &scales, &mut out_ref, &cfg).unwrap();

        assert_close(&out_tiled, &expected, 1e-3);
        assert_close(&out_tiled, &out_ref, 1e-5);
    }

    #[test]
    fn test_tiled_custom_tile_size() {
        let (m, n, k) = (16, 16, 32);
        let w: Vec<i8> = vec![1; k * n];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![1.0f32; m * k];
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2)
            .unwrap()
            .with_tiles(8, 8, 8)
            .unwrap();
        let mut out = vec![0.0f32; m * n];
        tiled_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        // Each element: sum of k ones = 32
        for &v in &out {
            assert!((v - 32.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_tiled_alpha_beta() {
        let (m, n, k) = (2, 2, 2);
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![1.0f32; m * k];
        let cfg =
            QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap().with_alpha_beta(2.0, 1.0);
        let mut out = vec![10.0, 20.0, 30.0, 40.0];
        tiled_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        // C = 2*(A×W) + 1*old = 2*[1,1,1,1]+[10,20,30,40]
        assert_close(&out, &[12.0, 22.0, 32.0, 42.0], 1e-5);
    }

    // ── split_k_gemm ──────────────────────────────────────────────

    #[test]
    fn test_split_k_matches_reference() {
        let (m, n, k) = (4, 4, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();

        let cfg_ref = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();
        let mut out_ref = vec![0.0f32; m * n];
        gemm_cpu_inner(&act, &packed, &scales, &mut out_ref, &cfg_ref).unwrap();

        let cfg_split =
            QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap().with_split_k(4).unwrap();
        let mut out_split = vec![0.0f32; m * n];
        split_k_gemm(&act, &packed, &scales, &mut out_split, &cfg_split).unwrap();

        assert_close(&out_split, &out_ref, 1e-4);
    }

    #[test]
    fn test_split_k_single_partition() {
        let (m, n, k) = (2, 2, 4);
        let w: Vec<i8> = vec![1; k * n];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![1.0f32; m * k];
        let cfg =
            QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap().with_split_k(1).unwrap();
        let mut out = vec![0.0f32; m * n];
        split_k_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        assert_close(&out, &[4.0, 4.0, 4.0, 4.0], 1e-6);
    }

    #[test]
    fn test_split_k_many_partitions() {
        let (m, n, k) = (2, 2, 16);
        let w: Vec<i8> = vec![1; k * n];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![1.0f32; m * k];
        let cfg =
            QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap().with_split_k(8).unwrap();
        let mut out = vec![0.0f32; m * n];
        split_k_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        assert_close(&out, &[16.0, 16.0, 16.0, 16.0], 1e-5);
    }

    #[test]
    fn test_split_k_alpha() {
        let (m, n, k) = (2, 2, 8);
        let w: Vec<i8> = vec![1; k * n];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![1.0f32; m * k];
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2)
            .unwrap()
            .with_split_k(2)
            .unwrap()
            .with_alpha_beta(0.5, 0.0);
        let mut out = vec![0.0f32; m * n];
        split_k_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        // 0.5 * 8 = 4.0
        assert_close(&out, &[4.0, 4.0, 4.0, 4.0], 1e-5);
    }

    // ── stream_k_gemm ─────────────────────────────────────────────

    #[test]
    fn test_stream_k_matches_reference() {
        let (m, n, k) = (8, 6, 12);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();

        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();
        let mut out_ref = vec![0.0f32; m * n];
        gemm_cpu_inner(&act, &packed, &scales, &mut out_ref, &cfg).unwrap();

        let mut out_sk = vec![0.0f32; m * n];
        stream_k_gemm(&act, &packed, &scales, &mut out_sk, &cfg).unwrap();

        assert_close(&out_sk, &out_ref, 1e-5);
    }

    #[test]
    fn test_stream_k_alpha_beta() {
        let (m, n, k) = (2, 2, 2);
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![1.0f32; m * k];
        let cfg =
            QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap().with_alpha_beta(3.0, 1.0);
        let mut out = vec![5.0, 10.0, 15.0, 20.0];
        stream_k_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        // C = 3*(A×W) + 1*old = 3*[1,1,1,1]+[5,10,15,20]
        assert_close(&out, &[8.0, 13.0, 18.0, 23.0], 1e-5);
    }

    #[test]
    fn test_stream_k_1x1() {
        let cfg = QuantGemmConfig::for_shape(1, 1, 1, QuantType::INT2).unwrap();
        let w: Vec<i8> = vec![1];
        let (packed, scales) = pack_col_major(&w, 1, 1, QuantType::INT2);
        let act = vec![5.0f32];
        let mut out = vec![0.0f32; 1];
        stream_k_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        assert_close(&out, &[5.0], 1e-6);
    }

    // ── GemmWorkspace ─────────────────────────────────────────────

    #[test]
    fn test_workspace_new() {
        let cfg =
            QuantGemmConfig::for_shape(4, 4, 4, QuantType::INT2).unwrap().with_split_k(4).unwrap();
        let ws = GemmWorkspace::new(&cfg);
        assert_eq!(ws.capacity(), 4 * 4 * 4);
    }

    #[test]
    fn test_workspace_ensure_capacity() {
        let cfg1 = QuantGemmConfig::for_shape(2, 2, 2, QuantType::INT2).unwrap();
        let mut ws = GemmWorkspace::new(&cfg1);
        assert_eq!(ws.capacity(), 4); // 1*2*2

        let cfg2 =
            QuantGemmConfig::for_shape(8, 8, 8, QuantType::INT2).unwrap().with_split_k(4).unwrap();
        ws.ensure_capacity(&cfg2);
        assert!(ws.capacity() >= 4 * 8 * 8);
    }

    #[test]
    fn test_workspace_reset() {
        let cfg = QuantGemmConfig::for_shape(2, 2, 2, QuantType::INT2).unwrap();
        let mut ws = GemmWorkspace::new(&cfg);
        ws.partials[0] = 42.0;
        ws.reset();
        assert_eq!(ws.partials[0], 0.0);
    }

    // ── GemmAutoTuner ─────────────────────────────────────────────

    #[test]
    fn test_autotuner_default() {
        let tuner = GemmAutoTuner::default();
        assert_eq!(tuner.max_shared_mem, 49152);
        assert_eq!(tuner.num_sms, 108);
    }

    #[test]
    fn test_autotuner_basic_tune() {
        let tuner = GemmAutoTuner::default();
        let cfg = tuner.tune(64, 64, 64, QuantType::INT2).unwrap();
        assert_eq!(cfg.m, 64);
        assert_eq!(cfg.n, 64);
        assert_eq!(cfg.k, 64);
        assert_eq!(cfg.quant_type, QuantType::INT2);
    }

    #[test]
    fn test_autotuner_tensor_cores_for_int8() {
        let tuner = GemmAutoTuner::default();
        let cfg = tuner.tune(16, 16, 16, QuantType::INT8).unwrap();
        assert!(cfg.use_tensor_cores);
        assert_eq!(cfg.tile_m, 16);
    }

    #[test]
    fn test_autotuner_no_tensor_cores_for_int2() {
        let tuner = GemmAutoTuner::default();
        let cfg = tuner.tune(16, 16, 16, QuantType::INT2).unwrap();
        assert!(!cfg.use_tensor_cores);
    }

    #[test]
    fn test_autotuner_split_k_for_tall_skinny() {
        let tuner = GemmAutoTuner::default();
        let cfg = tuner.tune(4, 4, 4096, QuantType::INT2).unwrap();
        assert!(cfg.split_k > 1, "should enable split-K for deep K");
    }

    #[test]
    fn test_autotuner_rejects_zero_dims() {
        let tuner = GemmAutoTuner::default();
        assert!(tuner.tune(0, 4, 4, QuantType::INT2).is_err());
    }

    #[test]
    fn test_autotuner_custom_device() {
        let tuner = GemmAutoTuner::new(16384, 40);
        let cfg = tuner.tune(32, 32, 32, QuantType::INT4).unwrap();
        assert!(cfg.shared_mem_bytes <= 16384);
    }

    // ── benchmark_gemm ────────────────────────────────────────────

    #[test]
    fn test_benchmark_returns_positive() {
        let cfg = QuantGemmConfig::for_shape(4, 4, 4, QuantType::INT2).unwrap();
        let gflops = benchmark_gemm(&cfg, 2);
        assert!(gflops > 0.0);
    }

    #[test]
    fn test_benchmark_single_iteration() {
        let cfg = QuantGemmConfig::for_shape(2, 2, 2, QuantType::INT2).unwrap();
        let gflops = benchmark_gemm(&cfg, 1);
        assert!(gflops > 0.0);
    }

    // ── Validation / buffer errors ────────────────────────────────

    #[test]
    fn test_activation_buffer_too_small() {
        let cfg = QuantGemmConfig::for_shape(4, 4, 8, QuantType::INT2).unwrap();
        let act = vec![1.0f32; 2]; // too small
        let packed = vec![0u8; cfg.weight_buffer_size()];
        let scales = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 16];
        assert!(quantized_gemm(&act, &packed, &scales, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_weight_buffer_too_small() {
        let cfg = QuantGemmConfig::for_shape(4, 4, 8, QuantType::INT2).unwrap();
        let act = vec![1.0f32; 32];
        let packed = vec![0u8; 1]; // too small
        let scales = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 16];
        assert!(quantized_gemm(&act, &packed, &scales, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_scales_buffer_too_small() {
        let cfg = QuantGemmConfig::for_shape(4, 4, 8, QuantType::INT2).unwrap();
        let act = vec![1.0f32; 32];
        let packed = vec![0u8; cfg.weight_buffer_size()];
        let scales = vec![1.0f32; 1]; // too small
        let mut out = vec![0.0f32; 16];
        assert!(quantized_gemm(&act, &packed, &scales, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_output_buffer_too_small() {
        let cfg = QuantGemmConfig::for_shape(4, 4, 8, QuantType::INT2).unwrap();
        let act = vec![1.0f32; 32];
        let packed = vec![0u8; cfg.weight_buffer_size()];
        let scales = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 2]; // too small
        assert!(quantized_gemm(&act, &packed, &scales, &mut out, &cfg).is_err());
    }

    // ── Non-unit scale tests ──────────────────────────────────────

    #[test]
    fn test_non_unit_scales() {
        let (m, n, k) = (2, 2, 4);
        let w: Vec<i8> = vec![1; k * n];
        let qt = QuantType::INT2;
        let (packed, _) = pack_col_major(&w, k, n, qt);
        let scales = vec![2.0f32, 0.5];
        let act = vec![1.0f32; m * k];
        let cfg = QuantGemmConfig::for_shape(m, n, k, qt).unwrap();
        let mut out = vec![0.0f32; m * n];
        quantized_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        // col0: 4*2.0=8, col1: 4*0.5=2
        assert_close(&out, &[8.0, 2.0, 8.0, 2.0], 1e-5);
    }

    #[test]
    fn test_zero_scale_produces_zero() {
        let (m, n, k) = (3, 3, 4);
        let w: Vec<i8> = vec![1; k * n];
        let (packed, _) = pack_col_major(&w, k, n, QuantType::INT2);
        let scales = vec![0.0f32; n];
        let act = vec![42.0f32; m * k];
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();
        let mut out = vec![f32::NAN; m * n];
        quantized_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        for &v in &out {
            assert!((v - 0.0).abs() < 1e-7, "expected 0, got {v}");
        }
    }

    // ── Larger matrix tests ───────────────────────────────────────

    #[test]
    fn test_large_16x8_int2() {
        let (m, n, k) = (16, 8, 48);
        let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1, 1, -1][i % 5]).collect();
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).sin()).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();
        run_all_paths(&act, &packed, &scales, &cfg, &expected, 1e-3);
    }

    #[test]
    fn test_large_32x32_int4() {
        let (m, n, k) = (32, 32, 16);
        let w: Vec<i8> = (0..k * n).map(|i| (i % 15) as i8 - 7).collect();
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT4);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT4).unwrap();
        run_all_paths(&act, &packed, &scales, &cfg, &expected, 1e-2);
    }

    // ── Alpha scaling ─────────────────────────────────────────────

    #[test]
    fn test_alpha_scales_output() {
        let (m, n, k) = (2, 2, 2);
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![1.0f32; m * k];
        let cfg =
            QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap().with_alpha_beta(3.0, 0.0);
        let mut out = vec![0.0f32; m * n];
        quantized_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        // 3 * (A×W) where A=all-ones, W=identity → all-ones
        // 3 * [1,1,1,1] = [3,3,3,3]
        assert_close(&out, &[3.0, 3.0, 3.0, 3.0], 1e-6);
    }

    #[test]
    fn test_beta_accumulate() {
        let (m, n, k) = (2, 2, 2);
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![1.0f32; m * k];
        let cfg =
            QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap().with_alpha_beta(1.0, 1.0);
        let mut out = vec![10.0, 20.0, 30.0, 40.0];
        quantized_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        // C = (A×W) + old = [1,1,1,1] + [10,20,30,40]
        assert_close(&out, &[11.0, 21.0, 31.0, 41.0], 1e-6);
    }

    // ── K not multiple of elems_per_byte ──────────────────────────

    #[test]
    fn test_k_not_multiple_of_4_int2() {
        let (m, n, k) = (3, 2, 5);
        let w: Vec<i8> = vec![1, 0, -1, 1, 0, 1, -1, 0, 1, -1];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32 + 0.5).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();
        run_all_paths(&act, &packed, &scales, &cfg, &expected, 1e-5);
    }

    #[test]
    fn test_k_odd_int4() {
        let (m, n, k) = (2, 2, 3);
        let w: Vec<i8> = vec![1, -1, 3, -3, 2, -2];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT4);
        let act = vec![1.0f32; m * k];
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_matmul(&act, &w_f32, m, n, k);
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT4).unwrap();
        run_all_paths(&act, &packed, &scales, &cfg, &expected, 1e-5);
    }

    // ── CUDA launch / kernel source ───────────────────────────────

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn test_kernel_src_not_empty() {
        assert!(!QUANTIZED_GEMM_KERNEL_SRC.is_empty());
        assert!(QUANTIZED_GEMM_KERNEL_SRC.contains("quantized_gemm_f32"));
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu \
                on GPU hardware"]
    fn test_cuda_quantized_gemm_launch() {
        let cfg = QuantGemmConfig::for_shape(4, 2048, 2048, QuantType::INT2).unwrap();
        let act = vec![1.0f32; 4 * 2048];
        let packed = vec![0u8; cfg.weight_buffer_size()];
        let scales = vec![1.0f32; 2048];
        let mut out = vec![0.0f32; 4 * 2048];
        let result = quantized_gemm(&act, &packed, &scales, &mut out, &cfg);
        assert!(result.is_ok(), "quantized GEMM launch failed: {result:?}");
    }

    // ── Cross-path consistency ────────────────────────────────────

    #[test]
    fn test_all_paths_agree_int2() {
        let (m, n, k) = (4, 6, 8);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0][i % 3]).collect();
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.05).collect();
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();

        let mut out_ref = vec![0.0f32; m * n];
        gemm_cpu_inner(&act, &packed, &scales, &mut out_ref, &cfg).unwrap();

        let mut out_tiled = vec![0.0f32; m * n];
        tiled_gemm(&act, &packed, &scales, &mut out_tiled, &cfg).unwrap();

        let mut out_sk = vec![0.0f32; m * n];
        stream_k_gemm(&act, &packed, &scales, &mut out_sk, &cfg).unwrap();

        let cfg_split =
            QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap().with_split_k(2).unwrap();
        let mut out_split = vec![0.0f32; m * n];
        split_k_gemm(&act, &packed, &scales, &mut out_split, &cfg_split).unwrap();

        let mut out_dispatch = vec![0.0f32; m * n];
        quantized_gemm(&act, &packed, &scales, &mut out_dispatch, &cfg).unwrap();

        assert_close(&out_tiled, &out_ref, 1e-5);
        assert_close(&out_sk, &out_ref, 1e-5);
        assert_close(&out_split, &out_ref, 1e-4);
        assert_close(&out_dispatch, &out_ref, 1e-6);
    }

    #[test]
    fn test_all_paths_agree_int4() {
        let (m, n, k) = (4, 4, 8);
        let w: Vec<i8> = (0..k * n).map(|i| (i % 9) as i8 - 4).collect();
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT4);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT4).unwrap();

        let mut out_ref = vec![0.0f32; m * n];
        gemm_cpu_inner(&act, &packed, &scales, &mut out_ref, &cfg).unwrap();

        let mut out_tiled = vec![0.0f32; m * n];
        tiled_gemm(&act, &packed, &scales, &mut out_tiled, &cfg).unwrap();

        let mut out_dispatch = vec![0.0f32; m * n];
        quantized_gemm(&act, &packed, &scales, &mut out_dispatch, &cfg).unwrap();

        assert_close(&out_tiled, &out_ref, 1e-4);
        assert_close(&out_dispatch, &out_ref, 1e-6);
    }

    #[test]
    fn test_all_paths_agree_int8() {
        let (m, n, k) = (4, 4, 8);
        let w: Vec<i8> = (0..k * n).map(|i| (i % 7) as i8 - 3).collect();
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT8);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT8).unwrap();

        let mut out_ref = vec![0.0f32; m * n];
        gemm_cpu_inner(&act, &packed, &scales, &mut out_ref, &cfg).unwrap();

        let mut out_tiled = vec![0.0f32; m * n];
        tiled_gemm(&act, &packed, &scales, &mut out_tiled, &cfg).unwrap();

        assert_close(&out_tiled, &out_ref, 1e-4);
    }

    // ── property-like tests ───────────────────────────────────────

    #[test]
    fn test_zero_activations_produce_zero() {
        for qt in [QuantType::INT2, QuantType::INT4, QuantType::INT8] {
            let (m, n, k) = (4, 4, 8);
            let w: Vec<i8> = vec![1; k * n];
            let (packed, scales) = pack_col_major(&w, k, n, qt);
            let act = vec![0.0f32; m * k];
            let cfg = QuantGemmConfig::for_shape(m, n, k, qt).unwrap();
            let mut out = vec![f32::NAN; m * n];
            quantized_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
            for &v in &out {
                assert!(v.abs() < 1e-7, "expected 0 for {qt:?}, got {v}");
            }
        }
    }

    #[test]
    fn test_output_bounded_by_k() {
        let (m, n, k) = (4, 4, 16);
        let w: Vec<i8> = vec![1; k * n];
        let (packed, scales) = pack_col_major(&w, k, n, QuantType::INT2);
        let act = vec![1.0f32; m * k];
        let cfg = QuantGemmConfig::for_shape(m, n, k, QuantType::INT2).unwrap();
        let mut out = vec![0.0f32; m * n];
        quantized_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
        let bound = k as f32 + 1e-4;
        for &v in &out {
            assert!(v.abs() <= bound, "output {v} exceeds bound {bound}");
        }
    }

    // ── proptest ──────────────────────────────────────────────────

    mod prop {
        use super::*;
        use proptest::prelude::*;

        fn dim_range() -> impl Strategy<Value = usize> {
            1..=12usize
        }

        fn k_range() -> impl Strategy<Value = usize> {
            1..=24usize
        }

        proptest! {
            #[test]
            fn output_shape_matches_config(
                m in dim_range(),
                n in dim_range(),
                k in k_range(),
            ) {
                let qt = QuantType::INT2;
                let cfg = QuantGemmConfig::for_shape(m, n, k, qt).unwrap();
                let epb = qt.elems_per_byte();
                let packed_k = k.div_ceil(epb);
                let act = vec![1.0f32; m * k];
                let packed = vec![0u8; packed_k * n];
                let scales = vec![1.0f32; n];
                let mut out = vec![0.0f32; m * n];
                quantized_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
                prop_assert_eq!(out.len(), m * n);
            }

            #[test]
            fn zero_weights_zero_output(
                m in dim_range(),
                n in dim_range(),
                k in k_range(),
            ) {
                let qt = QuantType::INT2;
                let cfg = QuantGemmConfig::for_shape(m, n, k, qt).unwrap();
                let packed_k = k.div_ceil(qt.elems_per_byte());
                let act: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.37).collect();
                let packed = vec![0u8; packed_k * n];
                let scales = vec![1.0f32; n];
                let mut out = vec![f32::NAN; m * n];
                quantized_gemm(&act, &packed, &scales, &mut out, &cfg).unwrap();
                for &v in &out {
                    prop_assert!((v - 0.0).abs() < 1e-7, "expected 0, got {v}");
                }
            }

            #[test]
            fn tiled_matches_reference(
                m in dim_range(),
                n in dim_range(),
                k in k_range(),
            ) {
                let qt = QuantType::INT2;
                let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0][i % 3]).collect();
                let (packed, scales) = pack_col_major(&w, k, n, qt);
                let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
                let cfg = QuantGemmConfig::for_shape(m, n, k, qt).unwrap();

                let mut out_ref = vec![0.0f32; m * n];
                gemm_cpu_inner(&act, &packed, &scales, &mut out_ref, &cfg).unwrap();

                let mut out_tiled = vec![0.0f32; m * n];
                tiled_gemm(&act, &packed, &scales, &mut out_tiled, &cfg).unwrap();

                for (i, (&a, &b)) in out_ref.iter().zip(out_tiled.iter()).enumerate() {
                    prop_assert!(
                        (a - b).abs() < 1e-4,
                        "mismatch at {i}: ref={a}, tiled={b}"
                    );
                }
            }

            #[test]
            fn split_k_matches_reference(
                m in dim_range(),
                n in dim_range(),
                k in 4..=24usize,
            ) {
                let qt = QuantType::INT2;
                let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0][i % 3]).collect();
                let (packed, scales) = pack_col_major(&w, k, n, qt);
                let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();

                let cfg_ref = QuantGemmConfig::for_shape(m, n, k, qt).unwrap();
                let mut out_ref = vec![0.0f32; m * n];
                gemm_cpu_inner(&act, &packed, &scales, &mut out_ref, &cfg_ref).unwrap();

                let cfg_split = QuantGemmConfig::for_shape(m, n, k, qt)
                    .unwrap()
                    .with_split_k(2)
                    .unwrap();
                let mut out_split = vec![0.0f32; m * n];
                split_k_gemm(&act, &packed, &scales, &mut out_split, &cfg_split).unwrap();

                for (i, (&a, &b)) in out_ref.iter().zip(out_split.iter()).enumerate() {
                    prop_assert!(
                        (a - b).abs() < 1e-4,
                        "mismatch at {i}: ref={a}, split={b}"
                    );
                }
            }
        }
    }
}
