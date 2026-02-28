//! RMSNorm forward-pass kernel for ROCm/HIP.
//!
//! Root Mean Square Layer Normalization is used in LLaMA-family
//! architectures (and by extension BitNet) instead of classic LayerNorm.
//!
//! ```text
//! rms  = sqrt( (1/d) * Σ x_i² + ε )
//! out_i = (x_i / rms) * γ_i
//! ```
//!
//! # HIP implementation
//!
//! 1. Each work-group handles one row (token position).
//! 2. Parallel reduction across the hidden dimension computes Σ x².
//! 3. A single-thread normalisation pass writes the output.
//!
//! Wavefront-level intrinsics (`__shfl_xor`, width 64) are used for the
//! reduction step, followed by LDS if the hidden dim exceeds one wavefront.

use bitnet_common::{KernelError, Result};

/// Configuration for the HIP RMSNorm kernel.
#[derive(Debug, Clone, Copy)]
pub struct HipRmsNormConfig {
    /// Hidden dimension (number of elements per row to normalise).
    pub hidden_dim: usize,
    /// Epsilon added inside the square-root for numerical stability.
    pub eps: f32,
}

impl HipRmsNormConfig {
    /// Create a config with default epsilon (1e-6).
    pub fn new(hidden_dim: usize) -> Self {
        Self { hidden_dim, eps: 1e-6 }
    }

    /// Override the epsilon value.
    #[must_use]
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

// ── CPU fallback ─────────────────────────────────────────────────────

/// RMSNorm computed on CPU (fallback when HIP is unavailable).
///
/// `input` is `[num_rows, hidden_dim]` in row-major order.
/// `gamma` is `[hidden_dim]`.
/// `output` must have the same length as `input`.
pub fn rmsnorm_cpu(
    input: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    num_rows: usize,
    config: &HipRmsNormConfig,
) -> Result<()> {
    let d = config.hidden_dim;
    let eps = config.eps;

    if d == 0 || num_rows == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "RMSNorm dimensions must be non-zero: \
                 num_rows={num_rows}, hidden_dim={d}"
            ),
        }
        .into());
    }

    let expected_len = num_rows * d;
    if input.len() < expected_len || output.len() < expected_len {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "RMSNorm buffer too small: need {expected_len}, \
                 got input={} output={}",
                input.len(),
                output.len()
            ),
        }
        .into());
    }
    if gamma.len() < d {
        return Err(KernelError::InvalidArguments {
            reason: format!("RMSNorm gamma too small: need {d}, got {}", gamma.len()),
        }
        .into());
    }

    let inv_d = 1.0 / d as f32;

    for row in 0..num_rows {
        let start = row * d;
        let end = start + d;
        let row_slice = &input[start..end];

        // Compute mean of squares
        let sq_sum: f32 = row_slice.iter().map(|x| x * x).sum();
        let rms = (sq_sum * inv_d + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Normalise and scale
        for (i, &x) in row_slice.iter().enumerate() {
            output[start + i] = (x * inv_rms) * gamma[i];
        }
    }

    Ok(())
}

// ── HIP dispatch ─────────────────────────────────────────────────────

/// Execute RMSNorm, dispatching to HIP when available.
pub fn rmsnorm_hip(
    input: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    num_rows: usize,
    config: &HipRmsNormConfig,
) -> Result<()> {
    #[cfg(feature = "rocm")]
    {
        if super::is_rocm_available() {
            return rmsnorm_hip_device(input, gamma, output, num_rows, config);
        }
    }

    rmsnorm_cpu(input, gamma, output, num_rows, config)
}

/// HIP device-side RMSNorm launch.
#[cfg(feature = "rocm")]
fn rmsnorm_hip_device(
    input: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    num_rows: usize,
    config: &HipRmsNormConfig,
) -> Result<()> {
    use super::hip_ffi;

    let d = config.hidden_dim;
    let total = num_rows * d;
    let total_bytes = total * std::mem::size_of::<f32>();
    let gamma_bytes = d * std::mem::size_of::<f32>();

    unsafe {
        let stream = hip_ffi::current_stream()?;

        let d_input = hip_ffi::device_malloc(total_bytes)?;
        let d_gamma = hip_ffi::device_malloc(gamma_bytes)?;
        let d_output = hip_ffi::device_malloc(total_bytes)?;

        hip_ffi::memcpy_h2d(d_input, input.as_ptr().cast(), total_bytes, stream)?;
        hip_ffi::memcpy_h2d(d_gamma, gamma.as_ptr().cast(), gamma_bytes, stream)?;

        let threads = (d as u32).min(1024);
        let blocks = num_rows as u32;

        hip_ffi::launch_rmsnorm(
            d_input.cast(),
            d_gamma.cast(),
            d_output.cast(),
            d as u32,
            num_rows as u32,
            config.eps,
            threads,
            blocks,
            stream,
        )?;

        hip_ffi::memcpy_d2h(output.as_mut_ptr().cast(), d_output, total_bytes, stream)?;
        hip_ffi::stream_synchronize(stream)?;

        hip_ffi::device_free(d_input)?;
        hip_ffi::device_free(d_gamma)?;
        hip_ffi::device_free(d_output)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_cpu_identity_gamma() {
        let cfg = HipRmsNormConfig::new(4);
        let input = [1.0f32, 1.0, 1.0, 1.0];
        let gamma = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        rmsnorm_cpu(&input, &gamma, &mut output, 1, &cfg).unwrap();

        // rms(1,1,1,1) = sqrt(1 + eps) ≈ 1, out ≈ 1
        for &v in &output {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn rmsnorm_cpu_with_gamma() {
        let cfg = HipRmsNormConfig::new(4);
        let input = [2.0f32; 4];
        let gamma = [0.5f32; 4];
        let mut output = [0.0f32; 4];
        rmsnorm_cpu(&input, &gamma, &mut output, 1, &cfg).unwrap();

        // rms = sqrt(4 + eps) ≈ 2, out ≈ (2/2)*0.5 = 0.5
        for &v in &output {
            assert!((v - 0.5).abs() < 1e-3, "expected ~0.5, got {v}");
        }
    }

    #[test]
    fn rmsnorm_cpu_multi_row() {
        let cfg = HipRmsNormConfig::new(2);
        let input = [1.0, 0.0, 0.0, 1.0];
        let gamma = [1.0, 1.0];
        let mut output = [0.0f32; 4];
        rmsnorm_cpu(&input, &gamma, &mut output, 2, &cfg).unwrap();

        assert!(output[0].is_finite());
        assert!(output[3].is_finite());
    }

    #[test]
    fn rmsnorm_cpu_rejects_zero_dims() {
        let cfg = HipRmsNormConfig::new(0);
        let input = [1.0f32; 4];
        let gamma = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        assert!(rmsnorm_cpu(&input, &gamma, &mut output, 1, &cfg).is_err());
    }

    #[test]
    fn rmsnorm_cpu_rejects_short_buffer() {
        let cfg = HipRmsNormConfig::new(4);
        let input = [1.0f32; 2]; // too short
        let gamma = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        assert!(rmsnorm_cpu(&input, &gamma, &mut output, 1, &cfg).is_err());
    }

    #[test]
    fn default_eps() {
        let cfg = HipRmsNormConfig::new(128);
        assert!((cfg.eps - 1e-6).abs() < 1e-10);
        assert_eq!(cfg.hidden_dim, 128);
    }

    #[test]
    fn with_eps_override() {
        let cfg = HipRmsNormConfig::new(64).with_eps(1e-5);
        assert!((cfg.eps - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn rmsnorm_hip_falls_back_to_cpu() {
        let cfg = HipRmsNormConfig::new(4);
        let input = [1.0f32; 4];
        let gamma = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        rmsnorm_hip(&input, &gamma, &mut output, 1, &cfg).unwrap();

        for &v in &output {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    #[ignore = "requires ROCm/HIP runtime — run on AMD GPU hardware"]
    fn rmsnorm_hip_device_dispatch() {
        let cfg = HipRmsNormConfig::new(2048);
        let input = vec![1.0f32; 2048];
        let gamma = vec![1.0f32; 2048];
        let mut output = vec![0.0f32; 2048];
        rmsnorm_hip(&input, &gamma, &mut output, 1, &cfg).unwrap();

        for &v in &output {
            assert!((v - 1.0).abs() < 1e-2, "HIP RMSNorm: got {v}");
        }
    }
}
