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
    /// Threads per block — typically `head_dim / 2`, capped at 1024.
    pub threads_per_block: u32,
}

impl RopeConfig {
    /// Create a configuration for the given shape.
    pub fn for_shape(head_dim: usize, n_heads: usize, seq_len: usize) -> Result<Self> {
        if head_dim == 0 || head_dim % 2 != 0 {
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
            threads_per_block,
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

            for i in 0..half_dim {
                let angle = actual_pos * inv_freq[i];
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let x0 = input[row_start + 2 * i];
                let x1 = input[row_start + 2 * i + 1];

                output[row_start + 2 * i] = x0 * cos_val - x1 * sin_val;
                output[row_start + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
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
        if crate::device_features::gpu_available_runtime() {
            if let Ok(()) = launch_rope(input, output, config) {
                return Ok(());
            }
            // GPU launch failed — fall through to CPU path
        }
    }
    rope_forward_cpu(input, output, config)
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
}
