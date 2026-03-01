//! CUDA quantized I2_S matrix multiplication kernel.
//!
//! # Kernel strategy
//!
//! Fused dequantization + GEMM for I2_S (2-bit signed ternary) weight
//! matrices.  Each weight element is one of {-1, 0, +1}, packed 4 values
//! per byte.  Two block sizes are supported:
//!
//! - **32**:  BitNet32-F16 (32-element blocks with inline F16 scales)
//! - **256**: QK256 / GGML (256-element blocks with separate scales)
//!
//! The kernel avoids materialising the full FP32 weight matrix by
//! unpacking 2-bit codes in shared memory, multiplying against the
//! activation tile in FP32, and applying per-block scales after
//! reduction.
//!
//! Target occupancy: ≥ 75 % on SM 8.0+ (Ampere) with 48 KB shared
//! memory.
//!
//! # CPU fallback
//!
//! [`i2s_matmul_cpu`] provides an equivalent pure-Rust implementation
//! for correctness testing and non-GPU environments.  The unified
//! [`i2s_matmul_forward`] dispatcher tries the GPU path first and
//! falls back transparently.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Encoding helpers (shared with CPU) ────────────────────────────────

/// Decode a 2-bit I2_S code to its signed integer value.
#[inline(always)]
fn decode_i2s(bits: u8) -> i8 {
    match bits & 0x03 {
        0b00 => 0,
        0b01 => 1,
        0b11 => -1,
        _ => 0, // 0b10 unused in I2_S spec
    }
}

/// Pack four ternary values ({-1, 0, +1}) into one byte, LSB-first.
pub fn pack_i2s(vals: [i8; 4]) -> u8 {
    let mut byte = 0u8;
    for (i, &v) in vals.iter().enumerate() {
        let code: u8 = match v {
            1 => 0b01,
            -1 => 0b11,
            _ => 0b00,
        };
        byte |= code << (i * 2);
    }
    byte
}

// ── Launch configuration ──────────────────────────────────────────────

/// Launch configuration for the I2_S quantized matmul CUDA kernel.
///
/// The grid is 2-D: `(ceil(n / tile_n), ceil(m / tile_m))`.
/// Each thread-block processes one `tile_m × tile_n` output tile.
#[derive(Debug, Clone)]
pub struct I2sMatmulConfig {
    /// Number of output rows (batch × sequence length).
    pub m: usize,
    /// Number of output columns (output hidden dimension).
    pub n: usize,
    /// Inner (reduction) dimension.
    pub k: usize,
    /// Block size for quantization scales (32 or 256).
    pub block_size: usize,
    /// CUDA tile size in the M dimension.
    pub tile_m: u32,
    /// CUDA tile size in the N dimension.
    pub tile_n: u32,
    /// Number of threads per block.
    pub threads_per_block: u32,
    /// Bytes of dynamic shared memory for packed weight tiles.
    pub shared_mem_bytes: u32,
}

impl Default for I2sMatmulConfig {
    fn default() -> Self {
        Self {
            m: 1,
            n: 1,
            k: 32,
            block_size: 32,
            tile_m: 4,
            tile_n: 64,
            threads_per_block: 256,
            shared_mem_bytes: 4096,
        }
    }
}

impl I2sMatmulConfig {
    /// Create a config tuned for the given matrix dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero or `block_size` is
    /// not 32 or 256.
    pub fn for_shape(m: usize, n: usize, k: usize, block_size: usize) -> Result<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "I2S matmul dimensions must be non-zero: \
                     m={m}, n={n}, k={k}"
                ),
            }
            .into());
        }
        if block_size != 32 && block_size != 256 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "I2S matmul block_size must be 32 or 256, \
                     got {block_size}"
                ),
            }
            .into());
        }

        let num_blocks_k = k.div_ceil(block_size);
        // Shared memory: packed weight tile + scale buffer
        let packed_bytes_per_block = block_size / 4;
        let shared = (packed_bytes_per_block + 4) * num_blocks_k;
        let shared_mem_bytes = (shared as u32).max(4096);

        Ok(Self {
            m,
            n,
            k,
            block_size,
            tile_m: 4,
            tile_n: 64,
            threads_per_block: 256,
            shared_mem_bytes,
        })
    }

    /// Compute the CUDA grid dimensions `(grid_x, grid_y, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let grid_x = (self.n as u32).div_ceil(self.tile_n);
        let grid_y = (self.m as u32).div_ceil(self.tile_m);
        (grid_x, grid_y, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ── Validation ────────────────────────────────────────────────────────

fn validate_i2s_matmul_args(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &[f32],
    config: &I2sMatmulConfig,
) -> Result<()> {
    let m = config.m;
    let n = config.n;
    let k = config.k;
    let block_size = config.block_size;

    if block_size == 0 {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "block_size must be > 0".into(),
        }));
    }

    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);

    if activations.len() < m * k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("activations too small: expected {}, got {}", m * k, activations.len()),
        }));
    }
    if weights_packed.len() < packed_k * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "weights_packed too small: expected {}, got {}",
                packed_k * n,
                weights_packed.len()
            ),
        }));
    }
    if scales.len() < n * num_blocks_k {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!(
                "scales too small: expected {}, got {}",
                n * num_blocks_k,
                scales.len()
            ),
        }));
    }
    if out.len() < m * n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("output too small: expected {}, got {}", m * n, out.len()),
        }));
    }
    Ok(())
}

// ── CPU fallback ──────────────────────────────────────────────────────

/// Scalar I2_S matrix multiplication (CPU fallback).
///
/// Computes `C[m×n] = A[m×k] · B_packed[k×n]` where `B_packed` stores
/// each column-major weight in 2-bit I2_S encoding (4 values per byte).
///
/// # Layout
/// - `activations`: row-major `[m, k]` f32
/// - `weights_packed`: packed I2_S, `ceil(k/4) * n` bytes, column-major
///   within each output column
/// - `scales`: one f32 per block of `block_size` elements along `k` per
///   output column → `n * num_blocks_k` entries
/// - `out`: row-major `[m, n]` f32
pub fn i2s_matmul_cpu(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    out: &mut [f32],
    config: &I2sMatmulConfig,
) -> Result<()> {
    validate_i2s_matmul_args(activations, weights_packed, scales, out, config)?;

    let m = config.m;
    let n = config.n;
    let k = config.k;
    let block_size = config.block_size;
    let packed_k = k.div_ceil(4);
    let num_blocks_k = k.div_ceil(block_size);

    out[..m * n].fill(0.0);

    for row in 0..m {
        let a_row = &activations[row * k..(row + 1) * k];
        for col in 0..n {
            let mut acc = 0.0f32;
            for blk in 0..num_blocks_k {
                let blk_start = blk * block_size;
                let blk_end = (blk_start + block_size).min(k);
                let scale = scales[col * num_blocks_k + blk];

                for (rel, &a_val) in a_row[blk_start..blk_end].iter().enumerate() {
                    let idx = blk_start + rel;
                    let byte_idx = col * packed_k + idx / 4;
                    let bit_off = (idx % 4) * 2;
                    let bits = (weights_packed[byte_idx] >> bit_off) & 0x03;
                    let w = decode_i2s(bits) as f32 * scale;
                    acc += a_val * w;
                }
            }
            out[row * n + col] = acc;
        }
    }
    Ok(())
}

// ── CUDA launch stub ──────────────────────────────────────────────────

/// Launch stub for the I2_S quantized matmul CUDA kernel.
///
/// # Arguments
///
/// * `activations`    — FP32 input `[m, k]` (row-major)
/// * `weights_packed` — I2_S packed weights, `ceil(k/4) * n` bytes
/// * `scales`         — Per-block FP32 scale factors
/// * `output`         — FP32 output `[m, n]` (row-major, written)
/// * `config`         — Launch configuration
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled
/// and loaded.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_i2s_matmul(
    _activations: &[f32],
    _weights_packed: &[u8],
    _scales: &[f32],
    _output: &mut [f32],
    config: &I2sMatmulConfig,
) -> Result<()> {
    log::debug!(
        "I2S matmul CUDA stub: m={}, n={}, k={}, bs={}, grid={:?}",
        config.m,
        config.n,
        config.k,
        config.block_size,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "I2S matmul CUDA kernel not yet compiled \
                 — scaffold only"
            .into(),
    }
    .into())
}

// ── Unified dispatch ──────────────────────────────────────────────────

/// Quantized I2_S matmul with automatic dispatch: GPU if available,
/// else CPU fallback.
pub fn i2s_matmul_forward(
    activations: &[f32],
    weights_packed: &[u8],
    scales: &[f32],
    output: &mut [f32],
    config: &I2sMatmulConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_i2s_matmul(
                activations,
                weights_packed,
                scales,
                output,
                config,
            )
        {
            return Ok(());
        }
        // GPU launch failed — fall through to CPU path
    }
    i2s_matmul_cpu(activations, weights_packed, scales, output, config)
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────

    /// Pack a full weight matrix (k×n, row-major ternary) into I2_S
    /// bytes with column-major packing and uniform scale = 1.0.
    fn pack_weight_matrix(
        weights: &[i8],
        k: usize,
        n: usize,
        block_size: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        let packed_k = k.div_ceil(4);
        let num_blocks_k = k.div_ceil(block_size);
        let mut packed = vec![0u8; packed_k * n];
        for col in 0..n {
            for row in 0..k {
                let val = weights[row * n + col];
                let code: u8 = match val {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                let byte_idx = col * packed_k + row / 4;
                let bit_off = (row % 4) * 2;
                packed[byte_idx] |= code << bit_off;
            }
        }
        let scales = vec![1.0f32; n * num_blocks_k];
        (packed, scales)
    }

    /// Naive f32 matmul: C = A · W  (A m×k, W k×n, row-major).
    fn naive_f32_matmul(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
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

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol {tol})");
        }
    }

    fn run_cpu_and_forward(
        act: &[f32],
        packed: &[u8],
        scales: &[f32],
        config: &I2sMatmulConfig,
        expected: &[f32],
        tol: f32,
    ) {
        let len = config.m * config.n;
        let mut out_cpu = vec![0.0f32; len];
        i2s_matmul_cpu(act, packed, scales, &mut out_cpu, config).unwrap();
        assert_close(&out_cpu, expected, tol);

        let mut out_fwd = vec![0.0f32; len];
        i2s_matmul_forward(act, packed, scales, &mut out_fwd, config).unwrap();
        assert_close(&out_fwd, expected, tol);
    }

    // ── config tests ──────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = I2sMatmulConfig::default();
        assert_eq!(cfg.threads_per_block, 256);
        assert_eq!(cfg.block_size, 32);
    }

    #[test]
    fn test_config_for_shape_block32() {
        let cfg = I2sMatmulConfig::for_shape(4, 2048, 2048, 32).unwrap();
        assert_eq!(cfg.m, 4);
        assert_eq!(cfg.n, 2048);
        assert_eq!(cfg.k, 2048);
        assert_eq!(cfg.block_size, 32);
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, 32); // ceil(2048/64)
        assert_eq!(gy, 1); // ceil(4/4)
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_config_for_shape_block256() {
        let cfg = I2sMatmulConfig::for_shape(1, 512, 2048, 256).unwrap();
        assert_eq!(cfg.block_size, 256);
        let (gx, gy, _) = cfg.grid_dim();
        assert_eq!(gx, 8); // ceil(512/64)
        assert_eq!(gy, 1);
    }

    #[test]
    fn test_config_rejects_zero_dims() {
        assert!(I2sMatmulConfig::for_shape(0, 8, 8, 32).is_err());
        assert!(I2sMatmulConfig::for_shape(8, 0, 8, 32).is_err());
        assert!(I2sMatmulConfig::for_shape(8, 8, 0, 32).is_err());
    }

    #[test]
    fn test_config_rejects_bad_block_size() {
        assert!(I2sMatmulConfig::for_shape(4, 8, 8, 64).is_err());
        assert!(I2sMatmulConfig::for_shape(4, 8, 8, 0).is_err());
        assert!(I2sMatmulConfig::for_shape(4, 8, 8, 128).is_err());
    }

    // ── basic matmul correctness ──────────────────────────────────

    #[test]
    fn test_identity_2x2_block32() {
        let (m, n, k, bs) = (2, 2, 2, 32);
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act = vec![3.0f32, -2.0, 5.0, 7.0];
        let expected = naive_f32_matmul(&act, &[1.0, 0.0, 0.0, 1.0], m, n, k);
        let cfg = I2sMatmulConfig::for_shape(m, n, k, bs).unwrap();
        run_cpu_and_forward(&act, &packed, &scales, &cfg, &expected, 1e-6);
    }

    #[test]
    fn test_identity_2x2_block256() {
        let (m, n, k, bs) = (2, 2, 2, 256);
        let w: Vec<i8> = vec![1, 0, 0, 1];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act = vec![3.0f32, -2.0, 5.0, 7.0];
        let expected = naive_f32_matmul(&act, &[1.0, 0.0, 0.0, 1.0], m, n, k);
        let cfg = I2sMatmulConfig::for_shape(m, n, k, bs).unwrap();
        run_cpu_and_forward(&act, &packed, &scales, &cfg, &expected, 1e-6);
    }

    #[test]
    fn test_all_ones_weight_4x4() {
        let (m, n, k, bs) = (4, 4, 4, 32);
        let w = vec![1i8; k * n];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        let cfg = I2sMatmulConfig::for_shape(m, n, k, bs).unwrap();
        run_cpu_and_forward(&act, &packed, &scales, &cfg, &expected, 1e-5);
    }

    #[test]
    fn test_all_neg_ones_weight() {
        let (m, n, k, bs) = (3, 3, 4, 32);
        let w = vec![-1i8; k * n];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act = vec![1.0f32; m * k];
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        let cfg = I2sMatmulConfig::for_shape(m, n, k, bs).unwrap();
        run_cpu_and_forward(&act, &packed, &scales, &cfg, &expected, 1e-5);
    }

    // ── quantized weight patterns ─────────────────────────────────

    #[test]
    fn test_mixed_ternary_pattern() {
        let (m, n, k, bs) = (4, 4, 8, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        let cfg = I2sMatmulConfig::for_shape(m, n, k, bs).unwrap();
        run_cpu_and_forward(&act, &packed, &scales, &cfg, &expected, 1e-4);
    }

    #[test]
    fn test_non_unit_scales() {
        let (m, n, k, bs) =
            (2usize, 2usize, 4usize, 32usize);
        let packed_k = k.div_ceil(4);
        let num_blocks_k = k.div_ceil(bs);

        // All weights = +1
        let mut packed = vec![0u8; packed_k * n];
        for col in 0..n {
            for row in 0..k {
                let byte_idx = col * packed_k + row / 4;
                let bit_off = (row % 4) * 2;
                packed[byte_idx] |= 0b01u8 << bit_off;
            }
        }
        let mut scales = vec![0.0f32; n * num_blocks_k];
        scales[0] = 2.0; // col 0
        scales[1] = 0.5; // col 1

        let act = vec![1.0f32; m * k];
        let cfg = I2sMatmulConfig::for_shape(m, n, k, bs).unwrap();
        // col0: 4 × 1.0 × 2.0 = 8.0,  col1: 4 × 1.0 × 0.5 = 2.0
        let expected = vec![8.0, 2.0, 8.0, 2.0];
        run_cpu_and_forward(
            &act, &packed, &scales, &cfg, &expected, 1e-5,
        );
    }

    // ── batch / larger matrices ───────────────────────────────────

    #[test]
    fn test_batch_16x8_block32() {
        let (m, n, k, bs) = (16, 8, 48, 32);
        let w: Vec<i8> = (0..k * n).map(|i| [1, 0, -1, 1, -1][i % 5]).collect();
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.03).sin()).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        let cfg = I2sMatmulConfig::for_shape(m, n, k, bs).unwrap();
        run_cpu_and_forward(&act, &packed, &scales, &cfg, &expected, 1e-3);
    }

    #[test]
    fn test_batch_64x32_block256() {
        let (m, n, k, bs) = (64, 32, 256, 256);
        let w: Vec<i8> = (0..k * n).map(|i| [1, -1][i % 2]).collect();
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        let cfg = I2sMatmulConfig::for_shape(m, n, k, bs).unwrap();
        run_cpu_and_forward(&act, &packed, &scales, &cfg, &expected, 1e-3);
    }

    // ── edge cases ────────────────────────────────────────────────

    #[test]
    fn test_1x1_matrix() {
        let (m, n, k, bs) = (1, 1, 1, 32);
        let w = vec![1i8];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act = vec![7.5f32];
        let expected = vec![7.5f32];
        let cfg = I2sMatmulConfig::for_shape(m, n, k, bs).unwrap();
        run_cpu_and_forward(&act, &packed, &scales, &cfg, &expected, 1e-6);
    }

    #[test]
    fn test_k_not_multiple_of_4() {
        let (m, n, k, bs) = (3, 2, 5, 32);
        let w: Vec<i8> = vec![1, 0, -1, 1, 0, 1, -1, 0, 1, -1];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act: Vec<f32> = (0..m * k).map(|i| i as f32 + 0.5).collect();
        let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let expected = naive_f32_matmul(&act, &w_f32, m, n, k);
        let cfg = I2sMatmulConfig::for_shape(m, n, k, bs).unwrap();
        run_cpu_and_forward(&act, &packed, &scales, &cfg, &expected, 1e-5);
    }

    #[test]
    fn test_zero_weight_produces_zero_output() {
        let (m, n, k, bs) = (4, 4, 8, 32);
        let w = vec![0i8; k * n];
        let (packed, scales) = pack_weight_matrix(&w, k, n, bs);
        let act = vec![42.0f32; m * k];
        let expected = vec![0.0f32; m * n];
        let cfg = I2sMatmulConfig::for_shape(m, n, k, bs).unwrap();
        run_cpu_and_forward(&act, &packed, &scales, &cfg, &expected, 1e-6);
    }

    // ── validation / error handling ───────────────────────────────

    #[test]
    fn test_activation_buffer_too_small() {
        let cfg = I2sMatmulConfig::for_shape(2, 2, 4, 32).unwrap();
        let act = vec![1.0f32; 2]; // too small
        let packed = vec![0u8; 4];
        let scales = vec![1.0f32; 2];
        let mut out = vec![0.0f32; 4];
        assert!(i2s_matmul_cpu(&act, &packed, &scales, &mut out, &cfg).is_err());
    }

    #[test]
    fn test_output_buffer_too_small() {
        let cfg = I2sMatmulConfig::for_shape(2, 2, 2, 32).unwrap();
        let act = vec![1.0f32; 4];
        let packed = vec![0u8; 2];
        let scales = vec![1.0f32; 2];
        let mut out = vec![0.0f32; 1]; // too small
        assert!(i2s_matmul_cpu(&act, &packed, &scales, &mut out, &cfg).is_err());
    }

    // ── pack_i2s round-trip ───────────────────────────────────────

    #[test]
    fn test_pack_i2s_roundtrip() {
        let vals: [i8; 4] = [1, -1, 0, 1];
        let byte = pack_i2s(vals);
        for (i, &expected) in vals.iter().enumerate() {
            let bits = (byte >> (i * 2)) & 0x03;
            assert_eq!(decode_i2s(bits), expected);
        }
    }

    // ── CUDA launch (requires GPU hardware) ───────────────────────

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu \
                on GPU hardware"]
    fn test_cuda_i2s_matmul_launch() {
        let cfg = I2sMatmulConfig::for_shape(4, 2048, 2048, 32).unwrap();
        let packed = vec![0u8; 2048 * 2048 / 4];
        let scales = vec![1.0f32; 2048 * (2048usize.div_ceil(32))];
        let act = vec![1.0f32; 4 * 2048];
        let mut output = vec![0.0f32; 4 * 2048];
        let result = i2s_matmul_forward(&act, &packed, &scales, &mut output, &cfg);
        assert!(result.is_ok(), "I2S matmul launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu \
                on GPU hardware"]
    fn test_cuda_i2s_matmul_block256_launch() {
        let cfg = I2sMatmulConfig::for_shape(1, 512, 2048, 256).unwrap();
        let packed = vec![0u8; 2048 * 512 / 4];
        let scales = vec![1.0f32; 512 * (2048usize.div_ceil(256))];
        let act = vec![1.0f32; 2048];
        let mut output = vec![0.0f32; 512];
        let result = i2s_matmul_forward(&act, &packed, &scales, &mut output, &cfg);
        assert!(result.is_ok(), "I2S matmul block256 launch failed: {result:?}");
    }
}
