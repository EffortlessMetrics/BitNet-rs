//! QK256 2-bit GEMV kernel for ROCm/HIP.
//!
//! Mirrors the CUDA matmul path but targets AMD GPUs via the HIP runtime.
//!
//! # HIP mapping notes
//!
//! | CUDA concept | HIP equivalent |
//! |-------------|----------------|
//! | `cudaMalloc` | `hipMalloc` |
//! | `cudaMemcpy` | `hipMemcpy` |
//! | `cudaStream_t` | `hipStream_t` |
//! | `__shared__` | `__shared__` (identical) |
//! | Thread block | Work-group |
//! | Grid | NDRange |
//! | `blockIdx` / `threadIdx` | `hipBlockIdx_x` / `hipThreadIdx_x` |
//!
//! The QK256 GEMV kernel processes 256-element quantized blocks, each
//! containing 2-bit signed weights packed 4 per byte.  Scale factors are
//! stored separately (one `f32` per block).
//!
//! # Launch configuration
//!
//! ```text
//! work-group size : 256 threads (4 wavefronts of 64 on GCN/CDNA)
//! grid            : ceil(N / 256) work-groups
//! LDS (shared)    : 1 KiB for partial-sum reduction
//! ```

use bitnet_common::{KernelError, Result};

/// Launch configuration for the QK256 HIP GEMV kernel.
#[derive(Debug, Clone, Copy)]
pub struct Qk256GemvConfig {
    /// Work-group (block) size — must be a multiple of the wavefront size (64).
    pub workgroup_size: u32,
    /// Bytes of LDS (shared memory) reserved per work-group.
    pub shared_mem_bytes: u32,
}

impl Default for Qk256GemvConfig {
    fn default() -> Self {
        Self { workgroup_size: 256, shared_mem_bytes: 1024 }
    }
}

impl Qk256GemvConfig {
    /// Create a config tuned for the given K dimension.
    pub fn for_k(k: usize) -> Result<Self> {
        if k == 0 || k % 256 != 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "QK256 GEMV inner dimension k={k} must be a \
                     positive multiple of 256"
                ),
            }
            .into());
        }
        let blocks_per_row = k / 256;
        // 64 bytes packed data + 4 bytes f32 scale per block
        let shared = ((blocks_per_row * (64 + 4)) as u32).max(1024);
        Ok(Self { workgroup_size: 256, shared_mem_bytes: shared })
    }

    /// Compute the HIP grid dimensions `(grid_x, 1, 1)`.
    pub fn grid_dim(&self, n: usize) -> (u32, u32, u32) {
        let grid_x = (n as u32).div_ceil(self.workgroup_size);
        (grid_x.max(1), 1, 1)
    }
}

// ── CPU fallback ─────────────────────────────────────────────────────

/// CPU scalar fallback for QK256 2-bit GEMV.
///
/// `weights` is packed 2-bit ternary (4 values per byte, 64 bytes per
/// 256-element block).  `scales` holds one `f32` per block.
///
/// Output: `output[row] = Σ_block ( scale[block] * dot(unpack(w), x) )`
pub fn qk256_gemv_cpu(
    weights: &[u8],
    scales: &[f32],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    _config: &Qk256GemvConfig,
) -> Result<()> {
    if k == 0 || k % 256 != 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("QK256 GEMV: k={k} must be a positive multiple of 256"),
        }
        .into());
    }
    if m == 0 || n == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("QK256 GEMV: m={m}, n={n} must be non-zero"),
        }
        .into());
    }

    let blocks_per_row = k / 256;
    let bytes_per_block = 64; // 256 values * 2 bits / 8

    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for blk in 0..blocks_per_row {
                let block_idx = row * n * blocks_per_row + col * blocks_per_row + blk;
                let scale = scales.get(block_idx).copied().unwrap_or(1.0);
                let w_offset = block_idx * bytes_per_block;

                let mut dot = 0.0f32;
                for elem in 0..256 {
                    let byte_in_block = elem / 4;
                    let bit_pos = (elem % 4) * 2;
                    let w_byte = weights.get(w_offset + byte_in_block).copied().unwrap_or(0);
                    let raw = (w_byte >> bit_pos) & 0x03;
                    let w_val: f32 = match raw {
                        0 => -1.0,
                        2 => 1.0,
                        _ => 0.0,
                    };
                    let x_idx = blk * 256 + elem;
                    let x_val = input.get(x_idx).copied().unwrap_or(0.0);
                    dot += w_val * x_val;
                }
                acc += scale * dot;
            }
            output[row * n + col] = acc;
        }
    }

    Ok(())
}

// ── HIP dispatch ─────────────────────────────────────────────────────

/// Execute a QK256 2-bit GEMV, dispatching to HIP when available.
pub fn qk256_gemv_hip(
    weights: &[u8],
    scales: &[f32],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &Qk256GemvConfig,
) -> Result<()> {
    #[cfg(feature = "rocm")]
    {
        if super::is_rocm_available() {
            return qk256_gemv_hip_device(weights, scales, input, output, m, n, k, config);
        }
    }

    qk256_gemv_cpu(weights, scales, input, output, m, n, k, config)
}

/// HIP device-side QK256 GEMV launch.
#[cfg(feature = "rocm")]
fn qk256_gemv_hip_device(
    weights: &[u8],
    scales: &[f32],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    config: &Qk256GemvConfig,
) -> Result<()> {
    use super::hip_ffi;

    let blocks_per_row = k / 256;
    let total_blocks = m * n * blocks_per_row;
    let w_bytes = total_blocks * 64;
    let s_bytes = total_blocks * std::mem::size_of::<f32>();
    let in_bytes = k * std::mem::size_of::<f32>();
    let out_bytes = m * n * std::mem::size_of::<f32>();

    unsafe {
        let stream = hip_ffi::current_stream()?;

        let d_w = hip_ffi::device_malloc(w_bytes)?;
        let d_s = hip_ffi::device_malloc(s_bytes)?;
        let d_in = hip_ffi::device_malloc(in_bytes)?;
        let d_out = hip_ffi::device_malloc(out_bytes)?;

        hip_ffi::memcpy_h2d(d_w, weights.as_ptr().cast(), w_bytes, stream)?;
        hip_ffi::memcpy_h2d(d_s, scales.as_ptr().cast(), s_bytes, stream)?;
        hip_ffi::memcpy_h2d(d_in, input.as_ptr().cast(), in_bytes, stream)?;

        let (grid_x, _, _) = config.grid_dim(n);
        hip_ffi::launch_qk256_gemv(
            d_w.cast(),
            d_s.cast(),
            d_in.cast(),
            d_out.cast(),
            m as u32,
            n as u32,
            k as u32,
            config.workgroup_size,
            grid_x * (m as u32),
            stream,
        )?;

        hip_ffi::memcpy_d2h(output.as_mut_ptr().cast(), d_out, out_bytes, stream)?;
        hip_ffi::stream_synchronize(stream)?;

        hip_ffi::device_free(d_w)?;
        hip_ffi::device_free(d_s)?;
        hip_ffi::device_free(d_in)?;
        hip_ffi::device_free(d_out)?;
    }

    Ok(())
}

/// A single batch GEMV item: (weights, scales, input, output, M, N, K).
pub type GemvBatchItem<'a> = (&'a [u8], &'a [f32], &'a [f32], &'a mut [f32], usize, usize, usize);

/// Batch QK256 GEMV — processes multiple operations sequentially.
pub fn qk256_gemv_hip_batch(
    batches: &mut [GemvBatchItem<'_>],
    config: &Qk256GemvConfig,
) -> Result<()> {
    for (weights, scales, input, output, m, n, k) in batches.iter_mut() {
        qk256_gemv_hip(weights, scales, input, output, *m, *n, *k, config)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = Qk256GemvConfig::default();
        assert_eq!(cfg.workgroup_size, 256);
        assert_eq!(cfg.shared_mem_bytes, 1024);
    }

    #[test]
    fn config_for_k_valid() {
        let cfg = Qk256GemvConfig::for_k(512).unwrap();
        assert_eq!(cfg.workgroup_size, 256);
        assert!(cfg.shared_mem_bytes >= 1024);
    }

    #[test]
    fn config_for_k_rejects_non_multiple() {
        assert!(Qk256GemvConfig::for_k(100).is_err());
        assert!(Qk256GemvConfig::for_k(0).is_err());
    }

    #[test]
    fn grid_dim_rounding() {
        let cfg = Qk256GemvConfig::default();
        let (gx, _, _) = cfg.grid_dim(500);
        assert_eq!(gx, 2); // ceil(500/256)
    }

    #[test]
    fn qk256_gemv_cpu_basic() {
        // 1 row, 1 output, k=256. All weights = +1 (code 0b10).
        let cfg = Qk256GemvConfig::default();
        let weights = vec![0b10_10_10_10u8; 64]; // all +1
        let scales = vec![1.0f32];
        let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; 1];

        qk256_gemv_cpu(&weights, &scales, &input, &mut output, 1, 1, 256, &cfg).unwrap();

        // dot(all_ones, 0..255) = sum(0..255) = 255*256/2 = 32640
        let expected: f32 = (0..256).map(|i| i as f32).sum();
        assert!((output[0] - expected).abs() < 1e-2, "got {} expected {}", output[0], expected,);
    }

    #[test]
    fn qk256_gemv_cpu_with_scale() {
        let cfg = Qk256GemvConfig::default();
        let weights = vec![0b10_10_10_10u8; 64]; // all +1
        let scales = vec![2.0f32]; // scale = 2
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];

        qk256_gemv_cpu(&weights, &scales, &input, &mut output, 1, 1, 256, &cfg).unwrap();

        // dot(all_ones, all_ones) * 2.0 = 256 * 2 = 512
        assert!((output[0] - 512.0).abs() < 1e-2, "got {} expected 512", output[0],);
    }

    #[test]
    fn qk256_gemv_cpu_rejects_bad_k() {
        let cfg = Qk256GemvConfig::default();
        let w = vec![0u8; 64];
        let s = vec![1.0f32; 1];
        let input = vec![1.0f32; 100];
        let mut output = vec![0.0f32; 1];
        assert!(qk256_gemv_cpu(&w, &s, &input, &mut output, 1, 1, 100, &cfg).is_err());
    }

    #[test]
    fn qk256_gemv_hip_falls_back_to_cpu() {
        let cfg = Qk256GemvConfig::default();
        let weights = vec![0b10_10_10_10u8; 64];
        let scales = vec![1.0f32];
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];

        qk256_gemv_hip(&weights, &scales, &input, &mut output, 1, 1, 256, &cfg).unwrap();

        assert!((output[0] - 256.0).abs() < 1e-2, "got {} expected 256", output[0],);
    }

    #[test]
    #[ignore = "requires ROCm/HIP runtime — run on AMD GPU hardware"]
    fn qk256_gemv_hip_device_dispatch() {
        let cfg = Qk256GemvConfig::for_k(256).unwrap();
        let weights = vec![0b10_10_10_10u8; 64];
        let scales = vec![1.0f32];
        let input = vec![1.0f32; 256];
        let mut output = vec![0.0f32; 1];

        qk256_gemv_hip(&weights, &scales, &input, &mut output, 1, 1, 256, &cfg).unwrap();

        assert!((output[0] - 256.0).abs() < 1.0, "HIP GEMV: got {} expected ~256", output[0],);
    }
}
