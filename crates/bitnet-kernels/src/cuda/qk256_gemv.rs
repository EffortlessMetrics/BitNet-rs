//! QK256 dequantization + GEMV fused CUDA kernel.
//!
//! # Kernel strategy
//!
//! QK256 packs 256 ternary weights ({-1, 0, +1}) into 64 bytes (2 bits each)
//! with a separate per-block `f16` scale factor. The fused dequant+GEMV kernel
//! avoids materialising the full FP32 weight matrix by:
//!
//! 1. Loading a 256-element packed block into shared memory (64 B).
//! 2. Each warp unpacks 2-bit → i8 via bit-shift/mask (no LUT on SM ≥ 7.0).
//! 3. Dot-product of unpacked i8 weights with the FP16 activation tile is
//!    accumulated in FP32 using `__dp4a` (int8 dot) or FMA.
//! 4. The per-block scale is applied after reduction, writing one FP32 output
//!    element per thread-block row.
//!
//! Target occupancy: ≥ 75 % on SM 8.0+ (Ampere) with 48 KB shared memory.

use bitnet_common::{KernelError, Result};

/// Launch configuration for the QK256 dequant+GEMV kernel.
///
/// The grid is 2-D: `(ceil(n_out / tile_n), ceil(seq_len / tile_m))`.
/// Each thread-block processes one `tile_m × tile_n` output tile.
#[derive(Debug, Clone)]
pub struct Qk256GemvConfig {
    /// CUDA block size in the M (sequence) dimension.
    pub block_m: u32,
    /// CUDA block size in the N (output-channel) dimension.
    pub block_n: u32,
    /// Number of threads per block (typically `block_m * block_n` capped at 256).
    pub threads_per_block: u32,
    /// Bytes of dynamic shared memory per block for packed weight tiles.
    pub shared_mem_bytes: u32,
    /// Number of output rows (sequence length).
    pub seq_len: usize,
    /// Number of output columns (hidden dimension).
    pub n_out: usize,
    /// Inner dimension (input hidden dimension, must be multiple of 256).
    pub k: usize,
}

impl Default for Qk256GemvConfig {
    fn default() -> Self {
        Self {
            block_m: 1,
            block_n: 256,
            threads_per_block: 256,
            shared_mem_bytes: 4096,
            seq_len: 1,
            n_out: 1,
            k: 256,
        }
    }
}

impl Qk256GemvConfig {
    /// Create a config tuned for the given matrix dimensions.
    ///
    /// `k` must be a multiple of 256 (the QK256 block size).
    pub fn for_shape(seq_len: usize, n_out: usize, k: usize) -> Result<Self> {
        if k == 0 || k % 256 != 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "QK256 GEMV inner dimension k={k} must be a positive multiple of 256"
                ),
            }
            .into());
        }
        if seq_len == 0 || n_out == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "QK256 GEMV dimensions must be non-zero: seq_len={seq_len}, n_out={n_out}"
                ),
            }
            .into());
        }

        // 64 bytes per packed QK256 block + 2 bytes for f16 scale
        let blocks_per_row = k / 256;
        let shared_mem_bytes = (blocks_per_row * (64 + 2)) as u32;

        Ok(Self {
            block_m: 1,
            block_n: 256,
            threads_per_block: 256,
            shared_mem_bytes: shared_mem_bytes.max(4096),
            seq_len,
            n_out,
            k,
        })
    }

    /// Compute the CUDA grid dimensions `(grid_x, grid_y, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let grid_x = (self.n_out as u32).div_ceil(self.block_n);
        let grid_y = (self.seq_len as u32).div_ceil(self.block_m);
        (grid_x, grid_y, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

/// Launch stub for the QK256 dequant+GEMV kernel.
///
/// # Arguments
///
/// * `packed_weights` — QK256-packed weight matrix (2-bit ternary, 64 B per 256-elem block)
/// * `scales`         — Per-block f16 scale factors
/// * `input`          — FP32 input activations `[seq_len, k]`
/// * `output`         — FP32 output buffer `[seq_len, n_out]` (written by kernel)
/// * `config`         — Launch configuration
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled and loaded.
pub fn launch_qk256_gemv(
    _packed_weights: &[u8],
    _scales: &[u8],
    _input: &[f32],
    _output: &mut [f32],
    config: &Qk256GemvConfig,
) -> Result<()> {
    log::debug!(
        "QK256 GEMV stub: seq_len={}, n_out={}, k={}, grid={:?}",
        config.seq_len,
        config.n_out,
        config.k,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "QK256 GEMV CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qk256_gemv_config_defaults() {
        let cfg = Qk256GemvConfig::default();
        assert_eq!(cfg.threads_per_block, 256);
        assert_eq!(cfg.k, 256);
    }

    #[test]
    fn test_qk256_gemv_config_for_shape() {
        let cfg = Qk256GemvConfig::for_shape(1, 2048, 2048).unwrap();
        assert_eq!(cfg.seq_len, 1);
        assert_eq!(cfg.n_out, 2048);
        assert_eq!(cfg.k, 2048);
        let (gx, gy, gz) = cfg.grid_dim();
        assert_eq!(gx, 8); // 2048 / 256
        assert_eq!(gy, 1);
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_qk256_gemv_config_rejects_non_multiple_k() {
        let err = Qk256GemvConfig::for_shape(1, 2048, 100);
        assert!(err.is_err());
    }

    #[test]
    fn test_qk256_gemv_config_rejects_zero_dims() {
        assert!(Qk256GemvConfig::for_shape(0, 2048, 256).is_err());
        assert!(Qk256GemvConfig::for_shape(1, 0, 256).is_err());
    }

    #[test]
    fn test_qk256_gemv_grid_dim_rounding() {
        let cfg = Qk256GemvConfig::for_shape(3, 500, 512).unwrap();
        let (gx, gy, _) = cfg.grid_dim();
        assert_eq!(gx, 2); // ceil(500/256)
        assert_eq!(gy, 3); // ceil(3/1)
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_qk256_gemv_launch() {
        let cfg = Qk256GemvConfig::for_shape(1, 2048, 2048).unwrap();
        let packed = vec![0u8; 2048 * 2048 / 4]; // 2 bits per weight
        let scales = vec![0u8; (2048 * 2048 / 256) * 2]; // f16 per block
        let input = vec![1.0f32; 2048];
        let mut output = vec![0.0f32; 2048];
        // When a real kernel is loaded this should succeed
        let result = launch_qk256_gemv(&packed, &scales, &input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA QK256 GEMV launch failed: {result:?}");
    }
}
