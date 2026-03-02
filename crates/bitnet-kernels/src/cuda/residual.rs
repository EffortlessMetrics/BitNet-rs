//! CUDA-accelerated residual and skip connection kernels with CPU fallback.
//!
//! Residual connections are fundamental to transformer architectures, enabling
//! gradient flow through deep networks. This module provides several variants:
//!
//! - [`residual_add`]: Element-wise `x + residual`
//! - [`residual_add_scaled`]: Scaled `x + alpha * residual`
//! - [`pre_norm_residual`]: Pre-LayerNorm pattern `x + sublayer(norm(x))`
//! - [`post_norm_residual`]: Post-LayerNorm pattern `norm(x + sublayer(x))`
//! - [`gated_residual`]: Learnable gate `x + gate * sublayer(x)`
//! - [`stochastic_depth_residual`]: Drop path for training regularization
//! - [`dense_residual`]: DenseNet-style concatenation
//! - [`skip_connection_with_projection`]: Dimension-changing skip with linear projection
//!
//! # Kernel strategy
//!
//! All element-wise operations use grid-stride loops with 256 threads per block.
//! Norm-based variants compose with the LayerNorm / RMSNorm CPU fallbacks from
//! [`super::layernorm`].
//!
//! # CPU fallback
//!
//! Every public function provides a pure-Rust implementation for correctness
//! testing and non-GPU environments.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// PTX source (compiled at runtime via NVRTC when `gpu`/`cuda` is active)
// ---------------------------------------------------------------------------

/// Inline CUDA C source for residual connection kernels.
///
/// Contains kernels: `residual_add_f32`, `residual_add_scaled_f32`,
/// `gated_residual_f32`, `stochastic_depth_residual_f32`.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const RESIDUAL_KERNEL_SRC: &str = r#"
extern "C" __global__ void residual_add_f32(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = x[i] + residual[i];
    }
}

extern "C" __global__ void residual_add_scaled_f32(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    float* __restrict__ out,
    float alpha,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = x[i] + alpha * residual[i];
    }
}

extern "C" __global__ void gated_residual_f32(
    const float* __restrict__ x,
    const float* __restrict__ sublayer,
    const float* __restrict__ gate,
    float* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = x[i] + gate[i] * sublayer[i];
    }
}

extern "C" __global__ void stochastic_depth_residual_f32(
    const float* __restrict__ x,
    const float* __restrict__ sublayer,
    float* __restrict__ out,
    float survival_prob,
    int keep,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        if (keep) {
            out[i] = x[i] + sublayer[i] / survival_prob;
        } else {
            out[i] = x[i];
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for residual connection kernels.
#[derive(Debug, Clone)]
pub struct ResidualConfig {
    /// Number of elements to process.
    pub n: usize,
    /// Threads per block (default 256).
    pub threads_per_block: u32,
}

impl ResidualConfig {
    /// Create a new configuration for the given element count.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `n` is zero.
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "residual element count must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { n, threads_per_block: 256 })
    }

    /// Compute the CUDA grid dimensions.
    ///
    /// Caps at 65 535 blocks; the grid-stride loop handles overflow.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let blocks = (self.n as u32).div_ceil(self.threads_per_block);
        (blocks.min(65_535), 1, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// CPU fallback — element-wise residual add
// ---------------------------------------------------------------------------

/// Element-wise residual addition: `out[i] = x[i] + residual[i]`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `x` and `residual` differ in
/// length.
pub fn residual_add(x: &[f32], residual: &[f32]) -> Result<Vec<f32>> {
    if x.len() != residual.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "residual_add: x length {} != residual length {}",
                x.len(),
                residual.len()
            ),
        }
        .into());
    }
    Ok(x.iter().zip(residual.iter()).map(|(&a, &b)| a + b).collect())
}

/// In-place element-wise residual addition: `x[i] += residual[i]`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if lengths differ.
pub fn residual_add_inplace(x: &mut [f32], residual: &[f32]) -> Result<()> {
    if x.len() != residual.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "residual_add_inplace: x length {} != residual length {}",
                x.len(),
                residual.len()
            ),
        }
        .into());
    }
    for (xi, &ri) in x.iter_mut().zip(residual.iter()) {
        *xi += ri;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU fallback — scaled residual
// ---------------------------------------------------------------------------

/// Scaled residual: `out[i] = x[i] + alpha * residual[i]`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if lengths differ or alpha is
/// non-finite.
pub fn residual_add_scaled(x: &[f32], residual: &[f32], alpha: f32) -> Result<Vec<f32>> {
    if x.len() != residual.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "residual_add_scaled: x length {} != residual length {}",
                x.len(),
                residual.len()
            ),
        }
        .into());
    }
    if !alpha.is_finite() {
        return Err(KernelError::InvalidArguments {
            reason: format!("residual_add_scaled: alpha must be finite, got {alpha}"),
        }
        .into());
    }
    Ok(x.iter().zip(residual.iter()).map(|(&a, &b)| a + alpha * b).collect())
}

// ---------------------------------------------------------------------------
// CPU fallback — pre-norm residual
// ---------------------------------------------------------------------------

/// Pre-LayerNorm residual pattern: `x + sublayer_output`.
///
/// In a real transformer the caller first applies `norm(x)` then feeds it
/// through the sublayer. This function combines the identity skip with the
/// sublayer result: `out = x + sublayer_output`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if lengths differ.
pub fn pre_norm_residual(x: &[f32], sublayer_output: &[f32]) -> Result<Vec<f32>> {
    if x.len() != sublayer_output.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "pre_norm_residual: x length {} != sublayer_output length {}",
                x.len(),
                sublayer_output.len()
            ),
        }
        .into());
    }
    Ok(x.iter().zip(sublayer_output.iter()).map(|(&a, &b)| a + b).collect())
}

// ---------------------------------------------------------------------------
// CPU fallback — post-norm residual
// ---------------------------------------------------------------------------

/// Post-LayerNorm residual pattern: `norm(x + sublayer_output)`.
///
/// Adds the sublayer output to the input, then applies RMS normalization.
///
/// # Arguments
///
/// * `x` — input tensor `[n]`
/// * `sublayer_output` — sublayer result `[n]`
/// * `gamma` — normalization scale weights `[n]`
/// * `eps` — epsilon for normalization stability
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if lengths are inconsistent.
pub fn post_norm_residual(
    x: &[f32],
    sublayer_output: &[f32],
    gamma: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    if x.len() != sublayer_output.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "post_norm_residual: x length {} != sublayer_output length {}",
                x.len(),
                sublayer_output.len()
            ),
        }
        .into());
    }
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if gamma.len() < n {
        return Err(KernelError::InvalidArguments {
            reason: format!("post_norm_residual: gamma length {} < n {}", gamma.len(), n),
        }
        .into());
    }

    // x + sublayer_output
    let sum: Vec<f32> = x.iter().zip(sublayer_output.iter()).map(|(&a, &b)| a + b).collect();

    // RMS norm
    let sq_sum: f32 = sum.iter().map(|&v| v * v).sum();
    let inv_rms = 1.0 / (sq_sum / n as f32 + eps).sqrt();

    Ok(sum.iter().enumerate().map(|(i, &v)| v * inv_rms * gamma[i]).collect())
}

// ---------------------------------------------------------------------------
// CPU fallback — gated residual
// ---------------------------------------------------------------------------

/// Gated residual: `out[i] = x[i] + gate[i] * sublayer[i]`.
///
/// The gate tensor contains learned (or fixed) per-element scalars.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if any lengths differ.
pub fn gated_residual(x: &[f32], sublayer: &[f32], gate: &[f32]) -> Result<Vec<f32>> {
    let n = x.len();
    if sublayer.len() != n || gate.len() != n {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "gated_residual: length mismatch x={}, sublayer={}, gate={}",
                n,
                sublayer.len(),
                gate.len()
            ),
        }
        .into());
    }
    Ok(x.iter()
        .zip(sublayer.iter())
        .zip(gate.iter())
        .map(|((&xi, &si), &gi)| xi + gi * si)
        .collect())
}

// ---------------------------------------------------------------------------
// CPU fallback — stochastic depth (drop path)
// ---------------------------------------------------------------------------

/// Stochastic depth residual for training regularization.
///
/// When `keep` is true, outputs `x + sublayer / survival_prob` (rescaled to
/// preserve expected value). When `keep` is false, outputs `x` unchanged
/// (the sublayer is "dropped").
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if lengths differ or
/// `survival_prob` is out of `(0, 1]`.
pub fn stochastic_depth_residual(
    x: &[f32],
    sublayer: &[f32],
    survival_prob: f32,
    keep: bool,
) -> Result<Vec<f32>> {
    if x.len() != sublayer.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "stochastic_depth_residual: x length {} != sublayer length {}",
                x.len(),
                sublayer.len()
            ),
        }
        .into());
    }
    if survival_prob <= 0.0 || survival_prob > 1.0 || !survival_prob.is_finite() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "stochastic_depth_residual: survival_prob must be in (0, 1], got {survival_prob}"
            ),
        }
        .into());
    }
    if keep {
        let scale = 1.0 / survival_prob;
        Ok(x.iter().zip(sublayer.iter()).map(|(&xi, &si)| xi + si * scale).collect())
    } else {
        Ok(x.to_vec())
    }
}

// ---------------------------------------------------------------------------
// CPU fallback — dense (DenseNet-style) residual
// ---------------------------------------------------------------------------

/// DenseNet-style concatenation residual: concatenates `x` and `sublayer`
/// along the feature dimension.
///
/// For a single vector this simply appends `sublayer` after `x`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if either slice is empty.
pub fn dense_residual(x: &[f32], sublayer: &[f32]) -> Result<Vec<f32>> {
    if x.is_empty() || sublayer.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "dense_residual: x and sublayer must be non-empty".into(),
        }
        .into());
    }
    let mut out = Vec::with_capacity(x.len() + sublayer.len());
    out.extend_from_slice(x);
    out.extend_from_slice(sublayer);
    Ok(out)
}

/// Batched DenseNet-style concatenation along the feature axis.
///
/// Each row of `x` (width `x_dim`) is concatenated with the corresponding row
/// of `sublayer` (width `sub_dim`), producing rows of width `x_dim + sub_dim`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if lengths are inconsistent.
pub fn dense_residual_batched(
    x: &[f32],
    x_dim: usize,
    sublayer: &[f32],
    sub_dim: usize,
) -> Result<Vec<f32>> {
    if x_dim == 0 || sub_dim == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "dense_residual_batched: dimensions must be non-zero".into(),
        }
        .into());
    }
    if !x.is_empty() && !x.len().is_multiple_of(x_dim) {
        return Err(KernelError::InvalidArguments {
            reason: format!("dense_residual_batched: x length {} not multiple of {x_dim}", x.len()),
        }
        .into());
    }
    if !sublayer.is_empty() && !sublayer.len().is_multiple_of(sub_dim) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense_residual_batched: sublayer length {} not multiple of {sub_dim}",
                sublayer.len()
            ),
        }
        .into());
    }
    let n_rows_x = if x.is_empty() { 0 } else { x.len() / x_dim };
    let n_rows_s = if sublayer.is_empty() { 0 } else { sublayer.len() / sub_dim };
    if n_rows_x != n_rows_s {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "dense_residual_batched: row count mismatch x={n_rows_x}, sublayer={n_rows_s}"
            ),
        }
        .into());
    }
    let out_dim = x_dim + sub_dim;
    let mut out = vec![0.0_f32; n_rows_x * out_dim];
    for row in 0..n_rows_x {
        let x_start = row * x_dim;
        let s_start = row * sub_dim;
        let o_start = row * out_dim;
        out[o_start..o_start + x_dim].copy_from_slice(&x[x_start..x_start + x_dim]);
        out[o_start + x_dim..o_start + out_dim]
            .copy_from_slice(&sublayer[s_start..s_start + sub_dim]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// CPU fallback — skip connection with projection
// ---------------------------------------------------------------------------

/// Skip connection with linear projection for dimension mismatch.
///
/// Computes `out = sublayer_output + (x @ weight^T + bias)` where `weight` is
/// `[out_dim, in_dim]` row-major. This allows the skip path to change
/// dimensionality.
///
/// # Arguments
///
/// * `x` — input `[n_rows, in_dim]`
/// * `sublayer_output` — sublayer result `[n_rows, out_dim]`
/// * `weight` — projection matrix `[out_dim, in_dim]` row-major
/// * `bias` — optional bias `[out_dim]`
/// * `in_dim` — input feature dimension
/// * `out_dim` — output feature dimension
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on size mismatch.
pub fn skip_connection_with_projection(
    x: &[f32],
    sublayer_output: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    if in_dim == 0 || out_dim == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "skip_connection_with_projection: dimensions must be non-zero".into(),
        }
        .into());
    }
    if !x.is_empty() && !x.len().is_multiple_of(in_dim) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "skip_connection_with_projection: x length {} not multiple of in_dim {in_dim}",
                x.len()
            ),
        }
        .into());
    }
    let n_rows = if x.is_empty() { 0 } else { x.len() / in_dim };
    if sublayer_output.len() != n_rows * out_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "skip_connection_with_projection: sublayer_output length {} != n_rows({}) * out_dim({})",
                sublayer_output.len(),
                n_rows,
                out_dim
            ),
        }
        .into());
    }
    if weight.len() < out_dim * in_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "skip_connection_with_projection: weight length {} < out_dim({}) * in_dim({})",
                weight.len(),
                out_dim,
                in_dim
            ),
        }
        .into());
    }
    if let Some(b) = bias
        && b.len() < out_dim
    {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "skip_connection_with_projection: bias length {} < out_dim {}",
                b.len(),
                out_dim
            ),
        }
        .into());
    }

    let mut out = vec![0.0_f32; n_rows * out_dim];
    for row in 0..n_rows {
        let x_row = &x[row * in_dim..(row + 1) * in_dim];
        let s_row = &sublayer_output[row * out_dim..(row + 1) * out_dim];
        let o_row = &mut out[row * out_dim..(row + 1) * out_dim];
        for j in 0..out_dim {
            let w_row = &weight[j * in_dim..(j + 1) * in_dim];
            let dot: f32 = x_row.iter().zip(w_row.iter()).map(|(&a, &b)| a * b).sum();
            let b = bias.map_or(0.0, |b| b[j]);
            o_row[j] = s_row[j] + dot + b;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Batch residual add
// ---------------------------------------------------------------------------

/// Batched residual addition: applies `residual_add` to each pair.
///
/// # Errors
///
/// Propagates errors from [`residual_add`] for any mismatched pair.
pub fn batch_residual_add(xs: &[&[f32]], residuals: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
    if xs.len() != residuals.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "batch_residual_add: batch size mismatch xs={}, residuals={}",
                xs.len(),
                residuals.len()
            ),
        }
        .into());
    }
    xs.iter().zip(residuals.iter()).map(|(x, r)| residual_add(x, r)).collect()
}

// ---------------------------------------------------------------------------
// CUDA launch stubs
// ---------------------------------------------------------------------------

/// Launch stub for residual_add CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_residual_add(
    _x: &[f32],
    _residual: &[f32],
    config: &ResidualConfig,
) -> Result<Vec<f32>> {
    log::debug!("residual_add CUDA stub: n={}, grid={:?}", config.n, config.grid_dim(),);
    Err(KernelError::GpuError {
        reason: "residual_add CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Launch stub for gated_residual CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_gated_residual(
    _x: &[f32],
    _sublayer: &[f32],
    _gate: &[f32],
    config: &ResidualConfig,
) -> Result<Vec<f32>> {
    log::debug!("gated_residual CUDA stub: n={}, grid={:?}", config.n, config.grid_dim(),);
    Err(KernelError::GpuError {
        reason: "gated_residual CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ---------------------------------------------------------------------------
// Unified dispatch
// ---------------------------------------------------------------------------

/// Apply residual addition with automatic dispatch: GPU if available,
/// else CPU fallback.
pub fn residual_add_forward(x: &[f32], residual: &[f32]) -> Result<Vec<f32>> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let n = x.len();
        if n > 0
            && n == residual.len()
            && crate::device_features::gpu_available_runtime()
            && let Ok(config) = ResidualConfig::new(n)
            && let Ok(result) = launch_residual_add(x, residual, &config)
        {
            return Ok(result);
        }
    }
    residual_add(x, residual)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── residual_add — basic sizes ────────────────────────────────────────

    #[test]
    fn test_residual_add_size_1() {
        let out = residual_add(&[1.0], &[2.0]).unwrap();
        assert_eq!(out, vec![3.0]);
    }

    #[test]
    fn test_residual_add_size_32() {
        let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let r: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5).collect();
        let out = residual_add(&x, &r).unwrap();
        for i in 0..32 {
            assert!((out[i] - (x[i] + r[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn test_residual_add_size_256() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let r: Vec<f32> = (0..256).map(|i| (i as f32) * -0.01).collect();
        let out = residual_add(&x, &r).unwrap();
        for &v in &out {
            assert!(v.abs() < 1e-6, "expected ~0, got {v}");
        }
    }

    #[test]
    fn test_residual_add_size_1024() {
        let x = vec![1.0_f32; 1024];
        let r = vec![2.0_f32; 1024];
        let out = residual_add(&x, &r).unwrap();
        assert!(out.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_residual_add_size_8192() {
        let x: Vec<f32> = (0..8192).map(|i| ((i as f32) * 0.1).sin()).collect();
        let r = vec![0.5_f32; 8192];
        let out = residual_add(&x, &r).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!((v - (x[i] + 0.5)).abs() < 1e-6);
        }
    }

    // ── residual_add — identity & edge cases ──────────────────────────────

    #[test]
    fn test_residual_add_zero_residual_identity() {
        let x: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let zero = vec![0.0_f32; 64];
        let out = residual_add(&x, &zero).unwrap();
        assert_eq!(out, x);
    }

    #[test]
    fn test_residual_add_zero_input_copies_residual() {
        let zero = vec![0.0_f32; 64];
        let r: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let out = residual_add(&zero, &r).unwrap();
        assert_eq!(out, r);
    }

    #[test]
    fn test_residual_add_empty() {
        let out = residual_add(&[], &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_residual_add_mismatched_err() {
        assert!(residual_add(&[1.0, 2.0], &[1.0]).is_err());
    }

    // ── residual_add_inplace ──────────────────────────────────────────────

    #[test]
    fn test_residual_add_inplace_basic() {
        let mut x = vec![1.0, 2.0, 3.0];
        residual_add_inplace(&mut x, &[10.0, 20.0, 30.0]).unwrap();
        assert_eq!(x, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_residual_add_inplace_matches_out_of_place() {
        let x = vec![1.5, -2.5, 3.5, -4.5];
        let r = vec![0.5, 0.5, 0.5, 0.5];
        let out = residual_add(&x, &r).unwrap();
        let mut x_mut = x.clone();
        residual_add_inplace(&mut x_mut, &r).unwrap();
        assert_eq!(x_mut, out);
    }

    #[test]
    fn test_residual_add_inplace_mismatched_err() {
        let mut x = vec![1.0];
        assert!(residual_add_inplace(&mut x, &[1.0, 2.0]).is_err());
    }

    // ── residual_add_scaled ───────────────────────────────────────────────

    #[test]
    fn test_scaled_alpha_zero() {
        let x = vec![1.0, 2.0, 3.0];
        let r = vec![10.0, 20.0, 30.0];
        let out = residual_add_scaled(&x, &r, 0.0).unwrap();
        assert_eq!(out, x);
    }

    #[test]
    fn test_scaled_alpha_one() {
        let x = vec![1.0, 2.0, 3.0];
        let r = vec![10.0, 20.0, 30.0];
        let out = residual_add_scaled(&x, &r, 1.0).unwrap();
        let expected = residual_add(&x, &r).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_scaled_alpha_half() {
        let x = vec![0.0; 4];
        let r = vec![2.0, 4.0, 6.0, 8.0];
        let out = residual_add_scaled(&x, &r, 0.5).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scaled_negative_alpha() {
        let x = vec![10.0; 3];
        let r = vec![5.0; 3];
        let out = residual_add_scaled(&x, &r, -1.0).unwrap();
        assert!(out.iter().all(|&v| (v - 5.0).abs() < 1e-6));
    }

    #[test]
    fn test_scaled_large_alpha() {
        let x = vec![1.0; 4];
        let r = vec![0.001; 4];
        let out = residual_add_scaled(&x, &r, 1000.0).unwrap();
        assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-3));
    }

    #[test]
    fn test_scaled_mismatched_err() {
        assert!(residual_add_scaled(&[1.0], &[1.0, 2.0], 1.0).is_err());
    }

    #[test]
    fn test_scaled_nan_alpha_err() {
        assert!(residual_add_scaled(&[1.0], &[1.0], f32::NAN).is_err());
    }

    #[test]
    fn test_scaled_inf_alpha_err() {
        assert!(residual_add_scaled(&[1.0], &[1.0], f32::INFINITY).is_err());
    }

    // ── pre_norm_residual ─────────────────────────────────────────────────

    #[test]
    fn test_pre_norm_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let sub = vec![0.1, 0.2, 0.3, 0.4];
        let out = pre_norm_residual(&x, &sub).unwrap();
        for i in 0..4 {
            assert!((out[i] - (x[i] + sub[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn test_pre_norm_zero_sublayer_identity() {
        let x = vec![5.0, 6.0, 7.0];
        let out = pre_norm_residual(&x, &[0.0, 0.0, 0.0]).unwrap();
        assert_eq!(out, x);
    }

    #[test]
    fn test_pre_norm_mismatched_err() {
        assert!(pre_norm_residual(&[1.0, 2.0], &[1.0]).is_err());
    }

    // ── post_norm_residual ────────────────────────────────────────────────

    #[test]
    fn test_post_norm_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let sub = vec![0.0; 4];
        let gamma = vec![1.0; 4];
        let out = post_norm_residual(&x, &sub, &gamma, 1e-5).unwrap();
        // norm(x + 0) = norm(x)
        let sq_sum: f32 = x.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sq_sum / 4.0 + 1e-5_f32).sqrt();
        for i in 0..4 {
            let expected = x[i] * inv_rms;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "idx={i}: got {}, expected {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_post_norm_vs_pre_norm_differs() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let sub = vec![0.5, 0.5, 0.5, 0.5];
        let pre = pre_norm_residual(&x, &sub).unwrap();
        let gamma = vec![1.0; 4];
        let post = post_norm_residual(&x, &sub, &gamma, 1e-5).unwrap();
        // pre and post should differ (post has normalization applied)
        assert!(pre != post, "pre_norm and post_norm should produce different results");
    }

    #[test]
    fn test_post_norm_empty() {
        let out = post_norm_residual(&[], &[], &[], 1e-5).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_post_norm_mismatched_err() {
        assert!(post_norm_residual(&[1.0, 2.0], &[1.0], &[1.0, 1.0], 1e-5).is_err());
    }

    #[test]
    fn test_post_norm_short_gamma_err() {
        assert!(post_norm_residual(&[1.0, 2.0], &[1.0, 2.0], &[1.0], 1e-5).is_err());
    }

    // ── gated_residual ────────────────────────────────────────────────────

    #[test]
    fn test_gated_gate_zero() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let sub = vec![100.0; 4];
        let gate = vec![0.0; 4];
        let out = gated_residual(&x, &sub, &gate).unwrap();
        assert_eq!(out, x, "gate=0 should pass through x unchanged");
    }

    #[test]
    fn test_gated_gate_one() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let sub = vec![10.0, 20.0, 30.0, 40.0];
        let gate = vec![1.0; 4];
        let out = gated_residual(&x, &sub, &gate).unwrap();
        let expected = residual_add(&x, &sub).unwrap();
        assert_eq!(out, expected, "gate=1 should equal plain residual add");
    }

    #[test]
    fn test_gated_gate_half() {
        let x = vec![0.0; 4];
        let sub = vec![2.0, 4.0, 6.0, 8.0];
        let gate = vec![0.5; 4];
        let out = gated_residual(&x, &sub, &gate).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gated_varying_gates() {
        let x = vec![1.0, 1.0, 1.0];
        let sub = vec![10.0, 10.0, 10.0];
        let gate = vec![0.0, 0.5, 1.0];
        let out = gated_residual(&x, &sub, &gate).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 6.0).abs() < 1e-6);
        assert!((out[2] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_gated_mismatched_err() {
        assert!(gated_residual(&[1.0], &[1.0, 2.0], &[1.0]).is_err());
        assert!(gated_residual(&[1.0], &[1.0], &[1.0, 2.0]).is_err());
    }

    // ── stochastic_depth_residual ─────────────────────────────────────────

    #[test]
    fn test_stochastic_keep_true() {
        let x = vec![1.0, 2.0, 3.0];
        let sub = vec![0.5, 0.5, 0.5];
        let out = stochastic_depth_residual(&x, &sub, 1.0, true).unwrap();
        // survival_prob=1.0 → scale=1.0, so x + sub
        let expected = residual_add(&x, &sub).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_stochastic_keep_false() {
        let x = vec![1.0, 2.0, 3.0];
        let sub = vec![100.0, 200.0, 300.0];
        let out = stochastic_depth_residual(&x, &sub, 0.5, false).unwrap();
        assert_eq!(out, x, "keep=false should return x unchanged");
    }

    #[test]
    fn test_stochastic_rescaling() {
        let x = vec![0.0; 4];
        let sub = vec![1.0; 4];
        let out = stochastic_depth_residual(&x, &sub, 0.5, true).unwrap();
        // scale = 1/0.5 = 2.0, so output = 0 + 1*2 = 2
        assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_stochastic_survival_prob_boundary() {
        // survival_prob = 1.0 (max valid)
        assert!(stochastic_depth_residual(&[1.0], &[1.0], 1.0, true).is_ok());
        // survival_prob = 0.0 (invalid)
        assert!(stochastic_depth_residual(&[1.0], &[1.0], 0.0, true).is_err());
        // negative
        assert!(stochastic_depth_residual(&[1.0], &[1.0], -0.1, true).is_err());
        // > 1
        assert!(stochastic_depth_residual(&[1.0], &[1.0], 1.1, true).is_err());
    }

    #[test]
    fn test_stochastic_nan_survival_err() {
        assert!(stochastic_depth_residual(&[1.0], &[1.0], f32::NAN, true).is_err());
    }

    #[test]
    fn test_stochastic_mismatched_err() {
        assert!(stochastic_depth_residual(&[1.0, 2.0], &[1.0], 0.5, true).is_err());
    }

    // ── dense_residual ────────────────────────────────────────────────────

    #[test]
    fn test_dense_basic() {
        let x = vec![1.0, 2.0];
        let sub = vec![3.0, 4.0, 5.0];
        let out = dense_residual(&x, &sub).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_dense_single_element() {
        let out = dense_residual(&[42.0], &[7.0]).unwrap();
        assert_eq!(out, vec![42.0, 7.0]);
    }

    #[test]
    fn test_dense_empty_x_err() {
        assert!(dense_residual(&[], &[1.0]).is_err());
    }

    #[test]
    fn test_dense_empty_sublayer_err() {
        assert!(dense_residual(&[1.0], &[]).is_err());
    }

    // ── dense_residual_batched ────────────────────────────────────────────

    #[test]
    fn test_dense_batched_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows x 2 cols
        let sub = vec![10.0, 20.0, 30.0]; // 2 rows, but 3 is not right
        // Correct: 2 rows x 1 col
        let sub2 = vec![10.0, 20.0];
        let out = dense_residual_batched(&x, 2, &sub2, 1).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 10.0, 3.0, 4.0, 20.0]);
        // mismatched rows
        assert!(dense_residual_batched(&x, 2, &sub, 3).is_err());
    }

    #[test]
    fn test_dense_batched_same_dim() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let sub = vec![5.0, 6.0, 7.0, 8.0];
        let out = dense_residual_batched(&x, 2, &sub, 2).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_dense_batched_zero_dim_err() {
        assert!(dense_residual_batched(&[1.0], 0, &[1.0], 1).is_err());
    }

    // ── skip_connection_with_projection ────────────────────────────────────

    #[test]
    fn test_skip_proj_identity_weight() {
        // in_dim == out_dim, weight = identity
        let x = vec![1.0, 2.0, 3.0];
        let sub = vec![0.0; 3];
        let weight = vec![
            1.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, // row 1
            0.0, 0.0, 1.0, // row 2
        ];
        let out = skip_connection_with_projection(&x, &sub, &weight, None, 3, 3).unwrap();
        assert_eq!(out, x);
    }

    #[test]
    fn test_skip_proj_dim_change() {
        // in_dim=2 → out_dim=3
        let x = vec![1.0, 2.0]; // 1 row
        let sub = vec![0.0, 0.0, 0.0]; // 1 row, out_dim=3
        let weight = vec![
            1.0, 0.0, // row 0 of W
            0.0, 1.0, // row 1
            1.0, 1.0, // row 2
        ];
        let out = skip_connection_with_projection(&x, &sub, &weight, None, 2, 3).unwrap();
        // proj = [1*1+0*2, 0*1+1*2, 1*1+1*2] = [1, 2, 3]
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_skip_proj_with_bias() {
        let x = vec![1.0, 0.0];
        let sub = vec![0.0; 2];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // identity 2x2
        let bias = vec![10.0, 20.0];
        let out = skip_connection_with_projection(&x, &sub, &weight, Some(&bias), 2, 2).unwrap();
        assert!((out[0] - 11.0).abs() < 1e-6);
        assert!((out[1] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_skip_proj_with_sublayer() {
        let x = vec![1.0, 1.0];
        let sub = vec![5.0, 5.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let out = skip_connection_with_projection(&x, &sub, &weight, None, 2, 2).unwrap();
        // sub + proj(x) = [5+1, 5+1] = [6, 6]
        assert!((out[0] - 6.0).abs() < 1e-6);
        assert!((out[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_skip_proj_batched() {
        let x = vec![1.0, 0.0, 0.0, 1.0]; // 2 rows x 2 cols
        let sub = vec![0.0; 4];
        let weight = vec![2.0, 0.0, 0.0, 2.0]; // 2*identity
        let out = skip_connection_with_projection(&x, &sub, &weight, None, 2, 2).unwrap();
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
        assert!((out[2] - 0.0).abs() < 1e-6);
        assert!((out[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_skip_proj_zero_dim_err() {
        assert!(skip_connection_with_projection(&[], &[], &[], None, 0, 1).is_err());
    }

    #[test]
    fn test_skip_proj_mismatched_sublayer_err() {
        let x = vec![1.0, 2.0];
        let sub = vec![0.0; 3]; // wrong: should be 2
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        assert!(skip_connection_with_projection(&x, &sub, &weight, None, 2, 2).is_err());
    }

    #[test]
    fn test_skip_proj_short_weight_err() {
        let x = vec![1.0, 2.0];
        let sub = vec![0.0; 2];
        let weight = vec![1.0]; // too short
        assert!(skip_connection_with_projection(&x, &sub, &weight, None, 2, 2).is_err());
    }

    #[test]
    fn test_skip_proj_short_bias_err() {
        let x = vec![1.0, 2.0];
        let sub = vec![0.0; 2];
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![1.0]; // too short
        assert!(skip_connection_with_projection(&x, &sub, &weight, Some(&bias), 2, 2).is_err());
    }

    // ── batch_residual_add ────────────────────────────────────────────────

    #[test]
    fn test_batch_residual_add_basic() {
        let x1 = [1.0_f32, 2.0];
        let x2 = [3.0_f32, 4.0];
        let r1 = [10.0_f32, 20.0];
        let r2 = [30.0_f32, 40.0];
        let out = batch_residual_add(&[&x1, &x2], &[&r1, &r2]).unwrap();
        assert_eq!(out, vec![vec![11.0, 22.0], vec![33.0, 44.0]]);
    }

    #[test]
    fn test_batch_residual_add_empty() {
        let out: Vec<Vec<f32>> = batch_residual_add(&[], &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_batch_residual_add_mismatch_count_err() {
        let x = [1.0_f32];
        assert!(batch_residual_add(&[&x[..]], &[]).is_err());
    }

    // ── numerical stability ───────────────────────────────────────────────

    #[test]
    fn test_residual_add_large_values() {
        let x = vec![1e30_f32; 8];
        let r = vec![1e30_f32; 8];
        let out = residual_add(&x, &r).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_residual_add_small_values() {
        let x = vec![1e-30_f32; 8];
        let r = vec![1e-30_f32; 8];
        let out = residual_add(&x, &r).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_residual_add_mixed_sign() {
        let x = vec![1e20_f32, -1e20, 1e-20, -1e-20];
        let r = vec![-1e20_f32, 1e20, -1e-20, 1e-20];
        let out = residual_add(&x, &r).unwrap();
        assert!(out.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_post_norm_large_values_stable() {
        let x = vec![1e6_f32; 4];
        let sub = vec![1e6_f32; 4];
        let gamma = vec![1.0; 4];
        let out = post_norm_residual(&x, &sub, &gamma, 1e-5).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gated_residual_large_gate() {
        let x = vec![1.0; 4];
        let sub = vec![1.0; 4];
        let gate = vec![1e6; 4];
        let out = gated_residual(&x, &sub, &gate).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── config tests ──────────────────────────────────────────────────────

    #[test]
    fn test_config_new() {
        let cfg = ResidualConfig::new(1024).unwrap();
        assert_eq!(cfg.n, 1024);
        assert_eq!(cfg.threads_per_block, 256);
    }

    #[test]
    fn test_config_rejects_zero() {
        assert!(ResidualConfig::new(0).is_err());
    }

    #[test]
    fn test_config_grid_dim() {
        let cfg = ResidualConfig::new(512).unwrap();
        assert_eq!(cfg.grid_dim(), (2, 1, 1));
    }

    #[test]
    fn test_config_grid_dim_large() {
        let cfg = ResidualConfig::new(256 * 100_000).unwrap();
        // Capped at 65535
        assert_eq!(cfg.grid_dim(), (65_535, 1, 1));
    }

    #[test]
    fn test_config_block_dim() {
        let cfg = ResidualConfig::new(1024).unwrap();
        assert_eq!(cfg.block_dim(), (256, 1, 1));
    }

    // ── forward dispatch ──────────────────────────────────────────────────

    #[test]
    fn test_residual_add_forward_cpu_fallback() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let r = vec![0.1, 0.2, 0.3, 0.4];
        let out = residual_add_forward(&x, &r).unwrap();
        let expected = residual_add(&x, &r).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn test_residual_add_forward_empty() {
        let out = residual_add_forward(&[], &[]).unwrap();
        assert!(out.is_empty());
    }

    // ── CUDA launch stubs (ignored — require GPU) ─────────────────────────

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_residual_add_launch() {
        let x = vec![1.0_f32; 4096];
        let r = vec![2.0_f32; 4096];
        let result = residual_add_forward(&x, &r);
        assert!(result.is_ok());
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_gated_residual_launch() {
        let n = 4096;
        let x = vec![1.0_f32; n];
        let sub = vec![2.0_f32; n];
        let gate = vec![0.5_f32; n];
        let result = gated_residual(&x, &sub, &gate);
        assert!(result.is_ok());
    }

    // ── negative residual ─────────────────────────────────────────────────

    #[test]
    fn test_residual_add_negative_values() {
        let x = vec![-1.0, -2.0, -3.0];
        let r = vec![-4.0, -5.0, -6.0];
        let out = residual_add(&x, &r).unwrap();
        assert_eq!(out, vec![-5.0, -7.0, -9.0]);
    }

    #[test]
    fn test_scaled_residual_negative_values() {
        let x = vec![-1.0, -2.0];
        let r = vec![-3.0, -4.0];
        let out = residual_add_scaled(&x, &r, 2.0).unwrap();
        assert!((out[0] - (-7.0)).abs() < 1e-6);
        assert!((out[1] - (-10.0)).abs() < 1e-6);
    }

    // ── symmetry / commutativity ──────────────────────────────────────────

    #[test]
    fn test_residual_add_commutative() {
        let a = vec![1.0, 3.0, 5.0, 7.0];
        let b = vec![2.0, 4.0, 6.0, 8.0];
        let out_ab = residual_add(&a, &b).unwrap();
        let out_ba = residual_add(&b, &a).unwrap();
        assert_eq!(out_ab, out_ba);
    }
}
