//! Rotary Position Embedding (RoPE) CUDA kernel.
//!
//! # Kernel strategy
//!
//! RoPE encodes absolute position information into query and key vectors by
//! rotating pairs of dimensions using sinusoidal functions:
//!
//!   For each pair `(x[2i], x[2i+1])` at position `pos`:
//!     `theta_i = base^(-2i / head_dim)`
//!     `angle   = pos * theta_i`
//!     `y[2i]   = x[2i]   * cos(angle) - x[2i+1] * sin(angle)`
//!     `y[2i+1] = x[2i]   * sin(angle) + x[2i+1] * cos(angle)`
//!
//! The default rotation base is `10000.0` (following the original RoPE paper).
//!
//! The CPU fallback precomputes sin/cos tables for all `(position, dim_pair)`
//! combinations, then applies the rotation in a single pass.
//!
//! The GPU kernel assigns one thread per `(position, dim_pair)`, reading the
//! precomputed sin/cos from constant memory or computing them on-the-fly via
//! `__sincosf`.
//!
//! Target: one thread-block per token position, threads covering head_dim/2
//! pairs. Grid size equals `seq_len × n_heads`.

use bitnet_common::{KernelError, Result};

// ── CUDA kernel source strings ───────────────────────────────────────

/// CUDA kernel source for RoPE forward pass (FP32).
///
/// Each thread handles one `(position, dim_pair)` element. The kernel
/// computes angles on-the-fly via `__sincosf` to avoid constant-memory
/// pressure for large sequence lengths.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ROPE_FORWARD_KERNEL_SRC: &str = r#"
extern "C" __global__ void rope_forward_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int head_dim,
    const int seq_len,
    const int n_heads,
    const int position_offset,
    const float theta_base,
    const float scaling_factor,
    const int interleaved
) {
    const int pos = blockIdx.x;
    const int head = blockIdx.y;
    const int half_dim = head_dim / 2;
    const int i = threadIdx.x;
    if (i >= half_dim) return;

    float exponent = -2.0f * (float)i / (float)head_dim;
    float inv_freq = powf(theta_base, exponent);
    float angle = (float)(pos + position_offset) * inv_freq * scaling_factor;

    float cos_val, sin_val;
    __sincosf(angle, &sin_val, &cos_val);

    int row_start = (head * seq_len + pos) * head_dim;

    float x0, x1;
    int idx0, idx1;
    if (interleaved) {
        idx0 = row_start + i;
        idx1 = row_start + i + half_dim;
    } else {
        idx0 = row_start + 2 * i;
        idx1 = row_start + 2 * i + 1;
    }
    x0 = input[idx0];
    x1 = input[idx1];

    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}
"#;

/// CUDA kernel source for RoPE backward pass (FP32).
///
/// The backward pass applies the transpose of the rotation matrix:
///   `dx[2i]   =  dy[2i] * cos + dy[2i+1] * sin`
///   `dx[2i+1] = -dy[2i] * sin + dy[2i+1] * cos`
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ROPE_BACKWARD_KERNEL_SRC: &str = r#"
extern "C" __global__ void rope_backward_f32(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    const int head_dim,
    const int seq_len,
    const int n_heads,
    const int position_offset,
    const float theta_base,
    const float scaling_factor,
    const int interleaved
) {
    const int pos = blockIdx.x;
    const int head = blockIdx.y;
    const int half_dim = head_dim / 2;
    const int i = threadIdx.x;
    if (i >= half_dim) return;

    float exponent = -2.0f * (float)i / (float)head_dim;
    float inv_freq = powf(theta_base, exponent);
    float angle = (float)(pos + position_offset) * inv_freq * scaling_factor;

    float cos_val, sin_val;
    __sincosf(angle, &sin_val, &cos_val);

    int row_start = (head * seq_len + pos) * head_dim;

    float dy0, dy1;
    int idx0, idx1;
    if (interleaved) {
        idx0 = row_start + i;
        idx1 = row_start + i + half_dim;
    } else {
        idx0 = row_start + 2 * i;
        idx1 = row_start + 2 * i + 1;
    }
    dy0 = grad_output[idx0];
    dy1 = grad_output[idx1];

    grad_input[idx0] =  dy0 * cos_val + dy1 * sin_val;
    grad_input[idx1] = -dy0 * sin_val + dy1 * cos_val;
}
"#;

/// Launch configuration for the RoPE kernel.
#[derive(Debug, Clone)]
pub struct RopeConfig {
    /// Per-head embedding dimension (must be even).
    pub head_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Sequence length (number of token positions).
    pub seq_len: usize,
    /// Position offset for KV-cache decode (added to the position index).
    pub position_offset: usize,
    /// Rotation base frequency (default `10_000.0`).
    pub base: f32,
    /// Frequency scaling factor (default `1.0`). Applied as a multiplier on
    /// the inverse-frequency table, allowing NTK-aware RoPE and YaRN-style
    /// frequency interpolation.
    pub scaling_factor: f32,
    /// Maximum sequence length for pre-computed sin/cos tables (defaults to
    /// `seq_len`). Set this larger when you plan to cache a table covering
    /// positions beyond the current batch.
    pub max_seq_len: usize,
    /// Threads per block — typically `head_dim / 2`, capped at 1024.
    pub threads_per_block: u32,
    /// When `true`, use the GPT-NeoX interleaved layout where pairs are at
    /// `(i, i + head_dim/2)` instead of `(2*i, 2*i+1)`.
    pub interleaved: bool,
}

impl RopeConfig {
    /// Create a configuration for the given shape.
    pub fn for_shape(head_dim: usize, n_heads: usize, seq_len: usize) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(KernelError::InvalidArguments {
                reason: format!("RoPE head_dim must be even and non-zero, got {head_dim}"),
            }
            .into());
        }
        if n_heads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "RoPE n_heads must be non-zero".into(),
            }
            .into());
        }
        if seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "RoPE seq_len must be non-zero".into(),
            }
            .into());
        }

        let half_dim = head_dim / 2;
        let threads_per_block = (half_dim as u32).min(1024);

        Ok(Self {
            head_dim,
            n_heads,
            seq_len,
            position_offset: 0,
            base: 10_000.0,
            scaling_factor: 1.0,
            max_seq_len: seq_len,
            threads_per_block,
            interleaved: false,
        })
    }

    /// Override the rotation base frequency (default `10_000.0`).
    #[must_use]
    pub fn with_base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }

    /// Set a position offset for KV-cache continuation.
    #[must_use]
    pub fn with_position_offset(mut self, offset: usize) -> Self {
        self.position_offset = offset;
        self
    }

    /// Override the frequency scaling factor (default `1.0`).
    #[must_use]
    pub fn with_scaling_factor(mut self, factor: f32) -> Self {
        self.scaling_factor = factor;
        self
    }

    /// Override `max_seq_len` for pre-computed sin/cos tables.
    #[must_use]
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Use the GPT-NeoX interleaved layout `(i, i + head_dim/2)`.
    #[must_use]
    pub fn with_interleaved(mut self, interleaved: bool) -> Self {
        self.interleaved = interleaved;
        self
    }

    /// Compute the CUDA grid dimensions `(seq_len, n_heads, 1)`.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        (self.seq_len as u32, self.n_heads as u32, 1)
    }

    /// Compute the CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

/// Compute the inverse-frequency table for RoPE.
///
/// Returns `head_dim / 2` values: `base^(-2i / head_dim)` for `i` in
/// `0..head_dim/2`.
fn compute_inv_freq(head_dim: usize, base: f32) -> Vec<f32> {
    let half = head_dim / 2;
    (0..half)
        .map(|i| {
            let exponent = -(2.0 * i as f32) / head_dim as f32;
            base.powf(exponent)
        })
        .collect()
}

/// Pre-compute an interleaved `[cos, sin, cos, sin, …]` table for
/// `max_seq_len` positions × `head_dim/2` dimension pairs.
///
/// Total length: `max_seq_len × head_dim`.
///
/// This is the CUDA-style analogue of `cpu::rope::compute_frequencies` and
/// uses the same interleaved layout so that the two modules can share tables
/// when needed.
pub fn compute_sincos_table(config: &RopeConfig) -> Vec<f32> {
    let inv_freq = compute_inv_freq(config.head_dim, config.base);
    let mut table = Vec::with_capacity(config.max_seq_len * config.head_dim);

    for pos in 0..config.max_seq_len {
        for &freq in &inv_freq {
            let angle = (pos as f32) * freq * config.scaling_factor;
            table.push(angle.cos());
            table.push(angle.sin());
        }
    }
    table
}

/// Apply RoPE on the CPU (fallback path).
///
/// Rotates `input[n_heads, seq_len, head_dim]` in-place (written to `output`)
/// using precomputed sin/cos tables.
pub fn rope_forward_cpu(input: &[f32], output: &mut [f32], config: &RopeConfig) -> Result<()> {
    let expected_len = config.n_heads * config.seq_len * config.head_dim;
    if input.len() != expected_len || output.len() != expected_len {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "RoPE buffer length mismatch: expected {expected_len}, \
                 got input={}, output={}",
                input.len(),
                output.len(),
            ),
        }
        .into());
    }

    let inv_freq = compute_inv_freq(config.head_dim, config.base);
    let half_dim = config.head_dim / 2;

    for head in 0..config.n_heads {
        for pos in 0..config.seq_len {
            let actual_pos = (pos + config.position_offset) as f32;
            let row_start = head * config.seq_len * config.head_dim + pos * config.head_dim;

            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = actual_pos * freq * config.scaling_factor;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let (idx0, idx1) = if config.interleaved {
                    (row_start + i, row_start + i + half_dim)
                } else {
                    (row_start + 2 * i, row_start + 2 * i + 1)
                };

                let x0 = input[idx0];
                let x1 = input[idx1];

                output[idx0] = x0 * cos_val - x1 * sin_val;
                output[idx1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    Ok(())
}

/// Launch stub for the RoPE CUDA kernel.
///
/// # Arguments
///
/// * `input`  — Input tensor `[n_heads, seq_len, head_dim]` (FP32)
/// * `output` — Output buffer `[n_heads, seq_len, head_dim]` (FP32, written)
/// * `config` — Launch configuration
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled and
/// loaded.
pub fn launch_rope(_input: &[f32], _output: &mut [f32], config: &RopeConfig) -> Result<()> {
    log::debug!(
        "RoPE stub: head_dim={}, n_heads={}, seq_len={}, offset={}, grid={:?}",
        config.head_dim,
        config.n_heads,
        config.seq_len,
        config.position_offset,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "RoPE CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// Apply RoPE with automatic dispatch: GPU if available, else CPU fallback.
///
/// # Arguments
///
/// * `input`  — Input tensor `[n_heads, seq_len, head_dim]` (FP32)
/// * `output` — Output buffer `[n_heads, seq_len, head_dim]` (FP32, written)
/// * `config` — Launch configuration
pub fn rope_forward(input: &[f32], output: &mut [f32], config: &RopeConfig) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_rope(input, output, config)
        {
            return Ok(());
        }
    }
    rope_forward_cpu(input, output, config)
}

// ── Precomputed frequency table ──────────────────────────────────────

/// Build a flat frequency table `[cos, sin, cos, sin, …]` for all
/// `(position, dim_pair)` combinations.
///
/// Returns `max_seq_len × head_dim` floats. This is a convenience wrapper
/// around [`compute_sincos_table`] for callers who only need `(head_dim,
/// max_seq_len, theta)` without constructing a full [`RopeConfig`].
pub fn build_rope_freqs(head_dim: usize, max_seq_len: usize, theta: f32) -> Vec<f32> {
    let inv_freq = compute_inv_freq(head_dim, theta);
    let mut table = Vec::with_capacity(max_seq_len * head_dim);

    for pos in 0..max_seq_len {
        for &freq in &inv_freq {
            let angle = (pos as f32) * freq;
            table.push(angle.cos());
            table.push(angle.sin());
        }
    }
    table
}

// ── Explicit-position API ────────────────────────────────────────────

/// Apply RoPE with an explicit position array (CPU fallback, GPU dispatch).
///
/// Unlike [`rope_forward`] which uses sequential positions
/// `[0..seq_len)`, this function takes arbitrary per-token positions,
/// enabling KV-cache-friendly non-contiguous position assignment.
///
/// # Arguments
///
/// * `input`     — `[n_heads × len(positions) × head_dim]` FP32
/// * `positions` — one position per token (length = `seq_len` in config)
/// * `config`    — launch configuration (`seq_len` must equal
///   `positions.len()`)
///
/// Returns the rotated output as a new `Vec<f32>`.
pub fn apply_rope(input: &[f32], positions: &[u32], config: &RopeConfig) -> Vec<f32> {
    let n = config.n_heads * positions.len() * config.head_dim;
    let mut output = vec![0.0f32; n];

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_rope(input, &mut output, config)
        {
            return output;
        }
    }

    let inv_freq = compute_inv_freq(config.head_dim, config.base);
    let half_dim = config.head_dim / 2;

    for head in 0..config.n_heads {
        for (seq_idx, &pos) in positions.iter().enumerate() {
            let actual_pos = pos as f32;
            let row_start = head * positions.len() * config.head_dim + seq_idx * config.head_dim;

            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = actual_pos * freq * config.scaling_factor;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let (idx0, idx1) = if config.interleaved {
                    (row_start + i, row_start + i + half_dim)
                } else {
                    (row_start + 2 * i, row_start + 2 * i + 1)
                };

                let x0 = input[idx0];
                let x1 = input[idx1];

                output[idx0] = x0 * cos_val - x1 * sin_val;
                output[idx1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
    output
}

/// Apply RoPE to a batched tensor with sequential positions per batch.
///
/// Input shape: `[batch_size × n_heads × seq_len × head_dim]`.
/// Each batch element uses positions `[0..seq_len)`.
///
/// Returns the rotated output as a new `Vec<f32>`.
pub fn apply_rope_batched(
    input: &[f32],
    batch_size: usize,
    seq_len: usize,
    config: &RopeConfig,
) -> Vec<f32> {
    let per_batch = config.n_heads * seq_len * config.head_dim;
    let total = batch_size * per_batch;
    let mut output = vec![0.0f32; total];

    let batch_cfg =
        RopeConfig { seq_len, max_seq_len: seq_len.max(config.max_seq_len), ..config.clone() };

    for b in 0..batch_size {
        let start = b * per_batch;
        let end = start + per_batch;
        let mut batch_out = vec![0.0f32; per_batch];

        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            if crate::device_features::gpu_available_runtime()
                && let Ok(()) = launch_rope(&input[start..end], &mut batch_out, &batch_cfg)
            {
                output[start..end].copy_from_slice(&batch_out);
                continue;
            }
        }

        let _ = rope_forward_cpu(&input[start..end], &mut batch_out, &batch_cfg);
        output[start..end].copy_from_slice(&batch_out);
    }
    output
}

// ── Backward pass ────────────────────────────────────────────────────

/// RoPE backward pass on the CPU.
///
/// Computes `grad_input` from `grad_output` by applying the transpose of
/// the rotation matrix:
///   `dx[2i]   =  dy[2i] * cos(θ) + dy[2i+1] * sin(θ)`
///   `dx[2i+1] = -dy[2i] * sin(θ) + dy[2i+1] * cos(θ)`
pub fn rope_backward_cpu(
    grad_output: &[f32],
    grad_input: &mut [f32],
    config: &RopeConfig,
) -> Result<()> {
    let expected_len = config.n_heads * config.seq_len * config.head_dim;
    if grad_output.len() != expected_len || grad_input.len() != expected_len {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "RoPE backward buffer length mismatch: expected {expected_len}, \
                 got grad_output={}, grad_input={}",
                grad_output.len(),
                grad_input.len(),
            ),
        }
        .into());
    }

    let inv_freq = compute_inv_freq(config.head_dim, config.base);
    let half_dim = config.head_dim / 2;

    for head in 0..config.n_heads {
        for pos in 0..config.seq_len {
            let actual_pos = (pos + config.position_offset) as f32;
            let row_start = head * config.seq_len * config.head_dim + pos * config.head_dim;

            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = actual_pos * freq * config.scaling_factor;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let (idx0, idx1) = if config.interleaved {
                    (row_start + i, row_start + i + half_dim)
                } else {
                    (row_start + 2 * i, row_start + 2 * i + 1)
                };

                let dy0 = grad_output[idx0];
                let dy1 = grad_output[idx1];

                grad_input[idx0] = dy0 * cos_val + dy1 * sin_val;
                grad_input[idx1] = -dy0 * sin_val + dy1 * cos_val;
            }
        }
    }

    Ok(())
}

/// Launch stub for the RoPE backward CUDA kernel.
///
/// # Errors
///
/// Returns `KernelError::GpuError` until a real PTX kernel is compiled.
pub fn launch_rope_backward(
    _grad_output: &[f32],
    _grad_input: &mut [f32],
    config: &RopeConfig,
) -> Result<()> {
    log::debug!(
        "RoPE backward stub: head_dim={}, n_heads={}, seq_len={}, grid={:?}",
        config.head_dim,
        config.n_heads,
        config.seq_len,
        config.grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "RoPE backward CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

/// RoPE backward with automatic dispatch: GPU if available, else CPU.
pub fn rope_backward(
    grad_output: &[f32],
    grad_input: &mut [f32],
    config: &RopeConfig,
) -> Result<()> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if crate::device_features::gpu_available_runtime()
            && let Ok(()) = launch_rope_backward(grad_output, grad_input, config)
        {
            return Ok(());
        }
    }
    rope_backward_cpu(grad_output, grad_input, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config construction ──────────────────────────────────────────

    #[test]
    fn test_rope_config_for_shape() {
        let cfg = RopeConfig::for_shape(128, 32, 512).unwrap();
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.seq_len, 512);
        assert_eq!(cfg.position_offset, 0);
        assert!((cfg.base - 10_000.0).abs() < 1e-3);
        assert!((cfg.scaling_factor - 1.0).abs() < 1e-6);
        assert_eq!(cfg.max_seq_len, 512);
        assert_eq!(cfg.threads_per_block, 64); // 128/2 = 64
    }

    #[test]
    fn test_rope_config_threads_capped() {
        // head_dim = 4096 → half = 2048, capped at 1024
        let cfg = RopeConfig::for_shape(4096, 1, 1).unwrap();
        assert_eq!(cfg.threads_per_block, 1024);
    }

    #[test]
    fn test_rope_config_rejects_zero_head_dim() {
        assert!(RopeConfig::for_shape(0, 8, 1).is_err());
    }

    #[test]
    fn test_rope_config_rejects_odd_head_dim() {
        assert!(RopeConfig::for_shape(63, 8, 1).is_err());
    }

    #[test]
    fn test_rope_config_rejects_zero_heads() {
        assert!(RopeConfig::for_shape(64, 0, 1).is_err());
    }

    #[test]
    fn test_rope_config_rejects_zero_seq() {
        assert!(RopeConfig::for_shape(64, 8, 0).is_err());
    }

    #[test]
    fn test_rope_config_with_base() {
        let cfg = RopeConfig::for_shape(64, 1, 1).unwrap().with_base(500_000.0);
        assert!((cfg.base - 500_000.0).abs() < 1.0);
    }

    #[test]
    fn test_rope_config_with_position_offset() {
        let cfg = RopeConfig::for_shape(64, 1, 1).unwrap().with_position_offset(42);
        assert_eq!(cfg.position_offset, 42);
    }

    #[test]
    fn test_rope_config_grid_dim() {
        let cfg = RopeConfig::for_shape(128, 8, 64).unwrap();
        assert_eq!(cfg.grid_dim(), (64, 8, 1));
        assert_eq!(cfg.block_dim(), (64, 1, 1)); // 128/2
    }

    #[test]
    fn test_rope_config_with_scaling_factor() {
        let cfg = RopeConfig::for_shape(64, 1, 1).unwrap().with_scaling_factor(2.0);
        assert!((cfg.scaling_factor - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_config_with_max_seq_len() {
        let cfg = RopeConfig::for_shape(64, 1, 4).unwrap().with_max_seq_len(8192);
        assert_eq!(cfg.max_seq_len, 8192);
        assert_eq!(cfg.seq_len, 4);
    }

    // ── Sin/cos table generation ─────────────────────────────────────

    #[test]
    fn test_sincos_table_length() {
        let cfg = RopeConfig::for_shape(8, 1, 16).unwrap().with_max_seq_len(16);
        let table = compute_sincos_table(&cfg);
        assert_eq!(table.len(), 16 * 8);
    }

    #[test]
    fn test_sincos_table_position_zero() {
        let cfg = RopeConfig::for_shape(4, 1, 2).unwrap().with_max_seq_len(2);
        let table = compute_sincos_table(&cfg);
        assert!((table[0] - 1.0).abs() < 1e-6, "cos(0) should be 1");
        assert!(table[1].abs() < 1e-6, "sin(0) should be 0");
        assert!((table[2] - 1.0).abs() < 1e-6, "cos(0) should be 1");
        assert!(table[3].abs() < 1e-6, "sin(0) should be 0");
    }

    #[test]
    fn test_sincos_table_known_values() {
        let cfg = RopeConfig::for_shape(2, 1, 2).unwrap().with_max_seq_len(2);
        let table = compute_sincos_table(&cfg);
        let expected_cos = 1.0f32.cos();
        let expected_sin = 1.0f32.sin();
        assert!(
            (table[2] - expected_cos).abs() < 1e-5,
            "cos(1): got {}, expected {expected_cos}",
            table[2]
        );
        assert!(
            (table[3] - expected_sin).abs() < 1e-5,
            "sin(1): got {}, expected {expected_sin}",
            table[3]
        );
    }

    #[test]
    fn test_sincos_table_with_scaling_factor() {
        let cfg1 = RopeConfig::for_shape(4, 1, 4).unwrap().with_max_seq_len(4);
        let cfg2 =
            RopeConfig::for_shape(4, 1, 4).unwrap().with_max_seq_len(4).with_scaling_factor(2.0);
        let t1 = compute_sincos_table(&cfg1);
        let t2 = compute_sincos_table(&cfg2);
        for i in 0..4 {
            assert!((t1[i] - t2[i]).abs() < 1e-6);
        }
        let any_diff = (4..8).any(|i| (t1[i] - t2[i]).abs() > 1e-4);
        assert!(any_diff, "scaling_factor should change table values");
    }

    #[test]
    fn test_sincos_table_matches_forward() {
        let head_dim = 4;
        let cfg = RopeConfig::for_shape(head_dim, 1, 2).unwrap().with_max_seq_len(2);
        let table = compute_sincos_table(&cfg);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0f32; 8];
        rope_forward_cpu(&input, &mut output, &cfg).unwrap();

        let pos = 1;
        let half = head_dim / 2;
        for i in 0..half {
            let cos_val = table[pos * head_dim + 2 * i];
            let sin_val = table[pos * head_dim + 2 * i + 1];
            let x0 = input[pos * head_dim + 2 * i];
            let x1 = input[pos * head_dim + 2 * i + 1];
            let expected0 = x0 * cos_val - x1 * sin_val;
            let expected1 = x0 * sin_val + x1 * cos_val;
            assert!(
                (output[pos * head_dim + 2 * i] - expected0).abs() < 1e-5,
                "table/forward mismatch at pair {i}"
            );
            assert!(
                (output[pos * head_dim + 2 * i + 1] - expected1).abs() < 1e-5,
                "table/forward mismatch at pair {i}"
            );
        }
    }

    // ── CPU forward correctness ──────────────────────────────────────

    #[test]
    fn test_rope_cpu_identity_at_position_zero() {
        // At position 0 all angles are 0 → cos=1, sin=0 → output == input
        let head_dim = 4;
        let cfg = RopeConfig::for_shape(head_dim, 1, 1).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];

        rope_forward_cpu(&input, &mut output, &cfg).unwrap();

        for (o, i) in output.iter().zip(input.iter()) {
            assert!((o - i).abs() < 1e-6, "position 0 should be identity: got {o}, expected {i}");
        }
    }

    #[test]
    fn test_rope_cpu_preserves_norm() {
        // RoPE is a rotation → vector norm is preserved
        let head_dim = 8;
        let n_heads = 2;
        let seq_len = 4;
        let cfg = RopeConfig::for_shape(head_dim, n_heads, seq_len).unwrap();
        let total = n_heads * seq_len * head_dim;
        let input: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut output = vec![0.0f32; total];

        rope_forward_cpu(&input, &mut output, &cfg).unwrap();

        // Check per-position norm preservation
        for head in 0..n_heads {
            for pos in 0..seq_len {
                let start = head * seq_len * head_dim + pos * head_dim;
                let in_norm: f32 = input[start..start + head_dim].iter().map(|x| x * x).sum();
                let out_norm: f32 = output[start..start + head_dim].iter().map(|x| x * x).sum();
                assert!(
                    (in_norm.sqrt() - out_norm.sqrt()).abs() < 1e-4,
                    "norm not preserved: head={head}, pos={pos}, \
                     in={}, out={}",
                    in_norm.sqrt(),
                    out_norm.sqrt(),
                );
            }
        }
    }

    #[test]
    fn test_rope_cpu_basic_rotation() {
        // Single head, single position (pos=1), head_dim=2 → one pair
        // theta_0 = 10000^0 = 1.0, angle = 1.0 * 1.0 = 1.0
        let cfg = RopeConfig::for_shape(2, 1, 2).unwrap();
        let input = vec![
            1.0, 0.0, // pos 0 → angle = 0
            1.0, 0.0, // pos 1 → angle = 1
        ];
        let mut output = vec![0.0f32; 4];

        rope_forward_cpu(&input, &mut output, &cfg).unwrap();

        // pos 0: cos(0)=1, sin(0)=0 → (1, 0)
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 0.0).abs() < 1e-6);

        // pos 1: (cos(1), sin(1)) ≈ (0.5403, 0.8415)
        let expected_cos = 1.0f32.cos();
        let expected_sin = 1.0f32.sin();
        assert!(
            (output[2] - expected_cos).abs() < 1e-5,
            "got {}, expected {expected_cos}",
            output[2]
        );
        assert!(
            (output[3] - expected_sin).abs() < 1e-5,
            "got {}, expected {expected_sin}",
            output[3]
        );
    }

    #[test]
    fn test_rope_cpu_position_offset() {
        let head_dim = 4;
        // Without offset at pos=1
        let cfg_no_offset = RopeConfig::for_shape(head_dim, 1, 2).unwrap();
        let input = vec![1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5];
        let mut out_no_offset = vec![0.0f32; 8];
        rope_forward_cpu(&input, &mut out_no_offset, &cfg_no_offset).unwrap();

        // With offset=1 at pos=0 should match no-offset pos=1
        let cfg_offset = RopeConfig::for_shape(head_dim, 1, 1).unwrap().with_position_offset(1);
        let single_input = vec![1.0, 0.0, 0.5, 0.5];
        let mut out_offset = vec![0.0f32; 4];
        rope_forward_cpu(&single_input, &mut out_offset, &cfg_offset).unwrap();

        // pos=1 from no-offset should equal pos=0 from offset=1
        for i in 0..head_dim {
            assert!(
                (out_no_offset[head_dim + i] - out_offset[i]).abs() < 1e-6,
                "offset mismatch at dim {i}: {} vs {}",
                out_no_offset[head_dim + i],
                out_offset[i],
            );
        }
    }

    #[test]
    fn test_rope_cpu_multi_head() {
        // Verify each head is processed independently with same rotation
        let head_dim = 4;
        let n_heads = 3;
        let seq_len = 2;
        let cfg = RopeConfig::for_shape(head_dim, n_heads, seq_len).unwrap();

        let total = n_heads * seq_len * head_dim;
        // All heads get the same data
        let pattern = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input: Vec<f32> = pattern.iter().copied().cycle().take(total).collect();
        let mut output = vec![0.0f32; total];

        rope_forward_cpu(&input, &mut output, &cfg).unwrap();

        // All heads should produce identical outputs (same input + same pos)
        let stride = seq_len * head_dim;
        for pos in 0..seq_len {
            for d in 0..head_dim {
                let ref_val = output[0 * stride + pos * head_dim + d];
                for head in 1..n_heads {
                    let val = output[head * stride + pos * head_dim + d];
                    assert!(
                        (ref_val - val).abs() < 1e-6,
                        "head {head} diverges from head 0 at pos={pos}, d={d}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_rope_cpu_different_seq_lengths() {
        let head_dim = 8;
        let n_heads = 2;

        for seq_len in [1, 2, 7, 16, 128] {
            let cfg = RopeConfig::for_shape(head_dim, n_heads, seq_len).unwrap();
            let total = n_heads * seq_len * head_dim;
            let input = vec![1.0f32; total];
            let mut output = vec![0.0f32; total];

            rope_forward_cpu(&input, &mut output, &cfg).unwrap();

            // Just verify no panic and output is finite
            assert!(output.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_rope_cpu_different_head_dims() {
        for head_dim in [2, 4, 8, 32, 64, 128, 256] {
            let cfg = RopeConfig::for_shape(head_dim, 1, 4).unwrap();
            let total = 1 * 4 * head_dim;
            let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
            let mut output = vec![0.0f32; total];

            rope_forward_cpu(&input, &mut output, &cfg).unwrap();
            assert!(output.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_rope_cpu_buffer_length_mismatch() {
        let cfg = RopeConfig::for_shape(4, 1, 1).unwrap();
        let input = vec![1.0f32; 4];
        let mut output_short = vec![0.0f32; 2]; // too short
        assert!(rope_forward_cpu(&input, &mut output_short, &cfg).is_err());

        let input_short = vec![1.0f32; 2]; // too short
        let mut output = vec![0.0f32; 4];
        assert!(rope_forward_cpu(&input_short, &mut output, &cfg).is_err());
    }

    #[test]
    fn test_rope_cpu_custom_base() {
        let head_dim = 4;
        let cfg_default = RopeConfig::for_shape(head_dim, 1, 2).unwrap();
        let cfg_custom = RopeConfig::for_shape(head_dim, 1, 2).unwrap().with_base(500_000.0);

        let input = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let mut out_default = vec![0.0f32; 8];
        let mut out_custom = vec![0.0f32; 8];

        rope_forward_cpu(&input, &mut out_default, &cfg_default).unwrap();
        rope_forward_cpu(&input, &mut out_custom, &cfg_custom).unwrap();

        // pos 0 identical (angle = 0 regardless of base)
        for i in 0..head_dim {
            assert!((out_default[i] - out_custom[i]).abs() < 1e-6);
        }
        // pos 1 should differ (different base → different angle)
        let any_diff = (0..head_dim)
            .any(|i| (out_default[head_dim + i] - out_custom[head_dim + i]).abs() > 1e-4);
        assert!(any_diff, "different base should produce different rotations");
    }

    #[test]
    fn test_rope_cpu_scaling_factor_identity() {
        let head_dim = 4;
        let cfg = RopeConfig::for_shape(head_dim, 1, 2).unwrap();
        let cfg_explicit = RopeConfig::for_shape(head_dim, 1, 2).unwrap().with_scaling_factor(1.0);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out1 = vec![0.0f32; 8];
        let mut out2 = vec![0.0f32; 8];
        rope_forward_cpu(&input, &mut out1, &cfg).unwrap();
        rope_forward_cpu(&input, &mut out2, &cfg_explicit).unwrap();
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-6, "scaling_factor=1.0 should match default");
        }
    }

    #[test]
    fn test_rope_cpu_scaling_factor_changes_output() {
        let head_dim = 4;
        let cfg1 = RopeConfig::for_shape(head_dim, 1, 2).unwrap();
        let cfg2 = RopeConfig::for_shape(head_dim, 1, 2).unwrap().with_scaling_factor(0.5);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out1 = vec![0.0f32; 8];
        let mut out2 = vec![0.0f32; 8];
        rope_forward_cpu(&input, &mut out1, &cfg1).unwrap();
        rope_forward_cpu(&input, &mut out2, &cfg2).unwrap();
        for i in 0..head_dim {
            assert!((out1[i] - out2[i]).abs() < 1e-6);
        }
        let any_diff =
            (0..head_dim).any(|i| (out1[head_dim + i] - out2[head_dim + i]).abs() > 1e-4);
        assert!(any_diff, "different scaling_factor should change rotations");
    }

    // ── Property tests ───────────────────────────────────────────────

    #[test]
    fn test_rope_zero_input_preserved() {
        for head_dim in [2, 4, 8, 64] {
            let cfg = RopeConfig::for_shape(head_dim, 1, 4).unwrap();
            let total = 4 * head_dim;
            let input = vec![0.0f32; total];
            let mut output = vec![1.0f32; total];
            rope_forward_cpu(&input, &mut output, &cfg).unwrap();
            for (i, val) in output.iter().enumerate() {
                assert!(val.abs() < 1e-10, "zero input not preserved at index {i}");
            }
        }
    }

    #[test]
    fn test_rope_different_positions_different_rotations() {
        let head_dim = 4;
        let cfg = RopeConfig::for_shape(head_dim, 1, 4).unwrap();
        let total = 4 * head_dim;
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0].into_iter().cycle().take(total).collect();
        let mut output = vec![0.0f32; total];
        rope_forward_cpu(&input, &mut output, &cfg).unwrap();
        for p in 0..3 {
            let start_a = p * head_dim;
            let start_b = (p + 1) * head_dim;
            let any_diff =
                (0..head_dim).any(|d| (output[start_a + d] - output[start_b + d]).abs() > 1e-6);
            assert!(any_diff, "pos {p} and {} should differ", p + 1);
        }
    }

    #[test]
    fn test_rope_norm_preservation_various_inputs() {
        let head_dim = 16;
        let cfg = RopeConfig::for_shape(head_dim, 1, 8).unwrap();
        let total = 8 * head_dim;
        let input: Vec<f32> = (0..total).map(|i| ((i * 37 + 13) as f32).sin() * 3.0).collect();
        let mut output = vec![0.0f32; total];
        rope_forward_cpu(&input, &mut output, &cfg).unwrap();
        for pos in 0..8 {
            let start = pos * head_dim;
            let in_norm: f32 =
                input[start..start + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            let out_norm: f32 =
                output[start..start + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (in_norm - out_norm).abs() < 1e-3,
                "norm not preserved at pos={pos}: {in_norm} vs {out_norm}"
            );
        }
    }

    // ── Inverse-frequency table ──────────────────────────────────────

    #[test]
    fn test_inv_freq_table() {
        let inv = compute_inv_freq(8, 10_000.0);
        assert_eq!(inv.len(), 4);
        // First entry: 10000^(-0/8) = 10000^0 = 1.0
        assert!((inv[0] - 1.0).abs() < 1e-6);
        // Second entry: 10000^(-2/8) = 10000^(-0.25) ≈ 0.1
        let expected = 10_000.0f32.powf(-0.25);
        assert!((inv[1] - expected).abs() < 1e-6, "got {}, expected {expected}", inv[1]);
        // Monotonically decreasing
        for w in inv.windows(2) {
            assert!(w[0] > w[1], "inv_freq should be decreasing");
        }
    }

    // ── Runtime dispatch ─────────────────────────────────────────────

    #[test]
    fn test_rope_forward_dispatches_cpu() {
        // On CPU-only builds, rope_forward should succeed via the CPU path
        let cfg = RopeConfig::for_shape(4, 1, 1).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];

        let result = rope_forward(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CPU dispatch should succeed: {result:?}");
    }

    // ── GPU launch stub ──────────────────────────────────────────────

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_rope_launch() {
        let cfg = RopeConfig::for_shape(128, 32, 64).unwrap();
        let total = 32 * 64 * 128;
        let input = vec![1.0f32; total];
        let mut output = vec![0.0f32; total];
        let result = launch_rope(&input, &mut output, &cfg);
        assert!(result.is_ok(), "CUDA RoPE launch failed: {result:?}");
    }

    #[test]
    #[ignore = "requires CUDA runtime — run with --features gpu on GPU hardware"]
    fn test_cuda_rope_backward_launch() {
        let cfg = RopeConfig::for_shape(128, 32, 64).unwrap();
        let total = 32 * 64 * 128;
        let grad_out = vec![1.0f32; total];
        let mut grad_in = vec![0.0f32; total];
        let result = launch_rope_backward(&grad_out, &mut grad_in, &cfg);
        assert!(result.is_ok(), "CUDA RoPE backward launch failed: {result:?}");
    }

    // ── Interleaved layout ───────────────────────────────────────────

    #[test]
    fn test_rope_config_interleaved_default_false() {
        let cfg = RopeConfig::for_shape(64, 1, 1).unwrap();
        assert!(!cfg.interleaved);
    }

    #[test]
    fn test_rope_config_with_interleaved() {
        let cfg = RopeConfig::for_shape(64, 1, 1).unwrap().with_interleaved(true);
        assert!(cfg.interleaved);
    }

    #[test]
    fn test_rope_interleaved_identity_at_pos_zero() {
        let head_dim = 4;
        let cfg = RopeConfig::for_shape(head_dim, 1, 1).unwrap().with_interleaved(true);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        rope_forward_cpu(&input, &mut output, &cfg).unwrap();
        for (o, i) in output.iter().zip(input.iter()) {
            assert!((o - i).abs() < 1e-6, "interleaved pos-0 identity: {o} vs {i}");
        }
    }

    #[test]
    fn test_rope_interleaved_preserves_norm() {
        let head_dim = 8;
        let cfg = RopeConfig::for_shape(head_dim, 2, 4).unwrap().with_interleaved(true);
        let total = 2 * 4 * head_dim;
        let input: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut output = vec![0.0f32; total];
        rope_forward_cpu(&input, &mut output, &cfg).unwrap();

        for head in 0..2 {
            for pos in 0..4 {
                let start = head * 4 * head_dim + pos * head_dim;
                let in_norm: f32 =
                    input[start..start + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                let out_norm: f32 =
                    output[start..start + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(
                    (in_norm - out_norm).abs() < 1e-4,
                    "interleaved norm: head={head}, pos={pos}"
                );
            }
        }
    }

    #[test]
    fn test_rope_interleaved_differs_from_default() {
        let head_dim = 8;
        let cfg_default = RopeConfig::for_shape(head_dim, 1, 2).unwrap();
        let cfg_interleaved = RopeConfig::for_shape(head_dim, 1, 2).unwrap().with_interleaved(true);
        let input =
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out_default = vec![0.0f32; 16];
        let mut out_interleaved = vec![0.0f32; 16];
        rope_forward_cpu(&input, &mut out_default, &cfg_default).unwrap();
        rope_forward_cpu(&input, &mut out_interleaved, &cfg_interleaved).unwrap();
        // pos 0 is identity for both, pos 1 should differ
        let any_diff =
            (head_dim..2 * head_dim).any(|i| (out_default[i] - out_interleaved[i]).abs() > 1e-6);
        assert!(any_diff, "interleaved should produce different rotation at pos>0");
    }

    // ── build_rope_freqs ─────────────────────────────────────────────

    #[test]
    fn test_build_rope_freqs_length() {
        let table = build_rope_freqs(8, 16, 10_000.0);
        assert_eq!(table.len(), 16 * 8);
    }

    #[test]
    fn test_build_rope_freqs_matches_sincos_table() {
        let cfg = RopeConfig::for_shape(8, 1, 16).unwrap().with_max_seq_len(16);
        let via_config = compute_sincos_table(&cfg);
        let via_standalone = build_rope_freqs(8, 16, 10_000.0);
        assert_eq!(via_config.len(), via_standalone.len());
        for (a, b) in via_config.iter().zip(via_standalone.iter()) {
            assert!((a - b).abs() < 1e-6, "tables should match: {a} vs {b}");
        }
    }

    #[test]
    fn test_build_rope_freqs_custom_theta() {
        let t1 = build_rope_freqs(4, 4, 10_000.0);
        let t2 = build_rope_freqs(4, 4, 500_000.0);
        // pos 0 should be identical (angle=0)
        for i in 0..4 {
            assert!((t1[i] - t2[i]).abs() < 1e-6);
        }
        // pos 1 should differ
        let any_diff = (4..8).any(|i| (t1[i] - t2[i]).abs() > 1e-6);
        assert!(any_diff, "different theta should produce different freqs");
    }

    #[test]
    fn test_build_rope_freqs_position_zero_is_cos1_sin0() {
        let table = build_rope_freqs(4, 1, 10_000.0);
        // At position 0, angle=0 for all dims → cos(0)=1, sin(0)=0
        assert!((table[0] - 1.0).abs() < 1e-6);
        assert!(table[1].abs() < 1e-6);
        assert!((table[2] - 1.0).abs() < 1e-6);
        assert!(table[3].abs() < 1e-6);
    }

    // ── apply_rope (explicit positions) ──────────────────────────────

    #[test]
    fn test_apply_rope_sequential_matches_forward() {
        let head_dim = 4;
        let cfg = RopeConfig::for_shape(head_dim, 1, 3).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let positions = [0u32, 1, 2];
        let mut expected = vec![0.0f32; 12];
        rope_forward_cpu(&input, &mut expected, &cfg).unwrap();

        let result = apply_rope(&input, &positions, &cfg);
        for (i, (a, b)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_apply_rope_noncontiguous_positions() {
        let head_dim = 4;
        let cfg = RopeConfig::for_shape(head_dim, 1, 2).unwrap();
        let input = vec![1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5];
        // positions [0, 5] — non-contiguous
        let result = apply_rope(&input, &[0, 5], &cfg);
        // Position 0 is identity
        for i in 0..head_dim {
            assert!((result[i] - input[i]).abs() < 1e-6);
        }
        // Position 5 should differ from position 1
        let cfg1 = RopeConfig::for_shape(head_dim, 1, 2).unwrap();
        let mut out_seq = vec![0.0f32; 8];
        rope_forward_cpu(&input, &mut out_seq, &cfg1).unwrap();
        let any_diff =
            (0..head_dim).any(|i| (result[head_dim + i] - out_seq[head_dim + i]).abs() > 1e-4);
        assert!(any_diff, "position 5 should differ from position 1");
    }

    #[test]
    fn test_apply_rope_single_position() {
        let head_dim = 4;
        let cfg = RopeConfig::for_shape(head_dim, 1, 1).unwrap();
        let input = vec![3.0, 4.0, 5.0, 6.0];
        let result = apply_rope(&input, &[0], &cfg);
        // Position 0 → identity
        for (a, b) in result.iter().zip(input.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_rope_multi_head() {
        let head_dim = 4;
        let n_heads = 2;
        let cfg = RopeConfig::for_shape(head_dim, n_heads, 2).unwrap();
        let total = n_heads * 2 * head_dim;
        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let result = apply_rope(&input, &[0, 1], &cfg);
        assert_eq!(result.len(), total);
        assert!(result.iter().all(|x| x.is_finite()));
    }

    // ── apply_rope_batched ───────────────────────────────────────────

    #[test]
    fn test_apply_rope_batched_single_batch() {
        let head_dim = 4;
        let seq_len = 3;
        let cfg = RopeConfig::for_shape(head_dim, 1, seq_len).unwrap();
        let total = seq_len * head_dim;
        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();

        let batched = apply_rope_batched(&input, 1, seq_len, &cfg);
        let mut expected = vec![0.0f32; total];
        rope_forward_cpu(&input, &mut expected, &cfg).unwrap();

        for (i, (a, b)) in batched.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "batch mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_apply_rope_batched_multi_batch() {
        let head_dim = 4;
        let seq_len = 2;
        let n_heads = 2;
        let batch_size = 3;
        let cfg = RopeConfig::for_shape(head_dim, n_heads, seq_len).unwrap();
        let per_batch = n_heads * seq_len * head_dim;
        let total = batch_size * per_batch;
        let input: Vec<f32> = (0..total).map(|i| ((i * 7 + 3) as f32).sin()).collect();

        let batched = apply_rope_batched(&input, batch_size, seq_len, &cfg);
        assert_eq!(batched.len(), total);

        // Each batch should match independent forward calls
        for b in 0..batch_size {
            let start = b * per_batch;
            let end = start + per_batch;
            let mut expected = vec![0.0f32; per_batch];
            rope_forward_cpu(&input[start..end], &mut expected, &cfg).unwrap();
            for i in 0..per_batch {
                assert!(
                    (batched[start + i] - expected[i]).abs() < 1e-5,
                    "batch {b} idx {i}: {} vs {}",
                    batched[start + i],
                    expected[i],
                );
            }
        }
    }

    #[test]
    fn test_apply_rope_batched_preserves_norm() {
        let head_dim = 8;
        let seq_len = 4;
        let batch_size = 2;
        let cfg = RopeConfig::for_shape(head_dim, 1, seq_len).unwrap();
        let per_batch = seq_len * head_dim;
        let total = batch_size * per_batch;
        let input: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let output = apply_rope_batched(&input, batch_size, seq_len, &cfg);
        for b in 0..batch_size {
            for pos in 0..seq_len {
                let start = b * per_batch + pos * head_dim;
                let in_n: f32 =
                    input[start..start + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                let out_n: f32 =
                    output[start..start + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((in_n - out_n).abs() < 1e-3, "batch norm: b={b} pos={pos}");
            }
        }
    }

    // ── Backward pass ────────────────────────────────────────────────

    #[test]
    fn test_rope_backward_identity_at_pos_zero() {
        let head_dim = 4;
        let cfg = RopeConfig::for_shape(head_dim, 1, 1).unwrap();
        let grad_out = vec![1.0, 2.0, 3.0, 4.0];
        let mut grad_in = vec![0.0f32; 4];
        rope_backward_cpu(&grad_out, &mut grad_in, &cfg).unwrap();
        // pos 0 → angle=0, transpose of identity rotation = identity
        for (o, i) in grad_in.iter().zip(grad_out.iter()) {
            assert!((o - i).abs() < 1e-6, "backward pos-0 identity: {o} vs {i}");
        }
    }

    #[test]
    fn test_rope_backward_is_inverse_of_forward() {
        // forward(backward(x)) ≈ x for any input
        let head_dim = 8;
        let cfg = RopeConfig::for_shape(head_dim, 2, 4).unwrap();
        let total = 2 * 4 * head_dim;
        let original: Vec<f32> = (0..total).map(|i| ((i * 13 + 7) as f32).sin()).collect();

        let mut forward_out = vec![0.0f32; total];
        rope_forward_cpu(&original, &mut forward_out, &cfg).unwrap();

        let mut roundtrip = vec![0.0f32; total];
        rope_backward_cpu(&forward_out, &mut roundtrip, &cfg).unwrap();

        for (i, (a, b)) in roundtrip.iter().zip(original.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "roundtrip mismatch at {i}: {a} vs {b}",);
        }
    }

    #[test]
    fn test_rope_backward_buffer_mismatch() {
        let cfg = RopeConfig::for_shape(4, 1, 1).unwrap();
        let grad_out = vec![1.0f32; 4];
        let mut grad_in_short = vec![0.0f32; 2];
        assert!(rope_backward_cpu(&grad_out, &mut grad_in_short, &cfg).is_err());
    }

    #[test]
    fn test_rope_backward_dispatch() {
        let cfg = RopeConfig::for_shape(4, 1, 1).unwrap();
        let grad_out = vec![1.0, 2.0, 3.0, 4.0];
        let mut grad_in = vec![0.0f32; 4];
        let result = rope_backward(&grad_out, &mut grad_in, &cfg);
        assert!(result.is_ok(), "backward dispatch should succeed: {result:?}");
    }

    #[test]
    fn test_rope_backward_zero_grad_preserved() {
        let head_dim = 8;
        let cfg = RopeConfig::for_shape(head_dim, 1, 4).unwrap();
        let total = 4 * head_dim;
        let grad_out = vec![0.0f32; total];
        let mut grad_in = vec![1.0f32; total];
        rope_backward_cpu(&grad_out, &mut grad_in, &cfg).unwrap();
        for val in &grad_in {
            assert!(val.abs() < 1e-10, "zero grad not preserved");
        }
    }

    #[test]
    fn test_rope_backward_interleaved() {
        let head_dim = 8;
        let cfg = RopeConfig::for_shape(head_dim, 1, 2).unwrap().with_interleaved(true);
        let total = 2 * head_dim;
        let original: Vec<f32> = (0..total).map(|i| (i as f32) * 0.3 + 1.0).collect();

        let mut forward_out = vec![0.0f32; total];
        rope_forward_cpu(&original, &mut forward_out, &cfg).unwrap();

        let mut roundtrip = vec![0.0f32; total];
        rope_backward_cpu(&forward_out, &mut roundtrip, &cfg).unwrap();

        for (i, (a, b)) in roundtrip.iter().zip(original.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "interleaved roundtrip at {i}: {a} vs {b}",);
        }
    }

    #[test]
    fn test_rope_backward_norm_preservation() {
        let head_dim = 16;
        let cfg = RopeConfig::for_shape(head_dim, 1, 4).unwrap();
        let total = 4 * head_dim;
        let grad_out: Vec<f32> = (0..total).map(|i| ((i * 11 + 5) as f32).cos()).collect();
        let mut grad_in = vec![0.0f32; total];
        rope_backward_cpu(&grad_out, &mut grad_in, &cfg).unwrap();

        for pos in 0..4 {
            let start = pos * head_dim;
            let in_norm: f32 =
                grad_out[start..start + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            let out_norm: f32 =
                grad_in[start..start + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (in_norm - out_norm).abs() < 1e-3,
                "backward norm at pos={pos}: {in_norm} vs {out_norm}"
            );
        }
    }

    // ── CUDA kernel source availability ──────────────────────────────

    #[test]
    #[ignore = "requires CUDA runtime — compile-check for kernel source strings"]
    fn test_cuda_kernel_sources_compile() {
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            assert!(!ROPE_FORWARD_KERNEL_SRC.is_empty());
            assert!(ROPE_FORWARD_KERNEL_SRC.contains("rope_forward_f32"));
            assert!(!ROPE_BACKWARD_KERNEL_SRC.is_empty());
            assert!(ROPE_BACKWARD_KERNEL_SRC.contains("rope_backward_f32"));
        }
    }
}
