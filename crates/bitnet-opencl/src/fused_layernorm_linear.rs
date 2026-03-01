//! Fused LayerNorm + Linear projection kernel.
//!
//! Combines layer normalisation with a linear projection into a single pass,
//! eliminating the intermediate global-memory write that a two-kernel approach
//! would require.  The fusion is configurable at runtime: when disabled, the
//! kernel acts as a plain linear projection.
//!
//! ```text
//! x ─→ [LayerNorm: mean/var → normalise → γ·x̂+β] ─→ [W·x + b] ─→ y
//!       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//!       skipped when fused == false
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Runtime configuration for the fused kernel.
#[derive(Debug, Clone)]
pub struct FusedLayerNormLinearConfig {
    /// Input feature dimension.
    pub hidden_dim: usize,
    /// Output feature dimension.
    pub out_dim: usize,
    /// LayerNorm epsilon. Default: 1e-5.
    pub eps: f32,
    /// Enable/disable the LayerNorm fusion at runtime.
    pub fused: bool,
}

impl Default for FusedLayerNormLinearConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 0,
            out_dim: 0,
            eps: 1e-5,
            fused: true,
        }
    }
}

impl FusedLayerNormLinearConfig {
    pub fn new(
        hidden_dim: usize,
        out_dim: usize,
    ) -> Result<Self, FusedLnLinearError> {
        if hidden_dim == 0 {
            return Err(FusedLnLinearError::InvalidDim(
                "hidden_dim must be > 0",
            ));
        }
        if out_dim == 0 {
            return Err(FusedLnLinearError::InvalidDim(
                "out_dim must be > 0",
            ));
        }
        Ok(Self {
            hidden_dim,
            out_dim,
            eps: 1e-5,
            fused: true,
        })
    }

    /// Builder: set epsilon.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Builder: enable/disable fusion.
    pub fn with_fused(mut self, fused: bool) -> Self {
        self.fused = fused;
        self
    }

    /// Return the OpenCL kernel source.
    pub fn kernel_source() -> &'static str {
        include_str!("../kernels/fused_layernorm_linear.cl")
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum FusedLnLinearError {
    InvalidDim(&'static str),
    ShapeMismatch { context: String },
}

impl fmt::Display for FusedLnLinearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDim(msg) => write!(f, "{msg}"),
            Self::ShapeMismatch { context } => {
                write!(f, "shape mismatch: {context}")
            }
        }
    }
}

impl std::error::Error for FusedLnLinearError {}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

/// CPU reference for the fused LayerNorm + Linear kernel.
pub struct FusedLayerNormLinear {
    cfg: FusedLayerNormLinearConfig,
}

impl FusedLayerNormLinear {
    pub fn new(cfg: FusedLayerNormLinearConfig) -> Self {
        Self { cfg }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `input`  — `[batch, hidden_dim]` row-major
    /// * `gamma`  — `[hidden_dim]` (LayerNorm scale)
    /// * `beta`   — `[hidden_dim]` (LayerNorm bias)
    /// * `weight` — `[out_dim, hidden_dim]` (linear weight)
    /// * `bias_linear` — `[out_dim]` or empty (no bias)
    /// * `batch`  — batch size
    pub fn forward(
        &self,
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        weight: &[f32],
        bias_linear: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>, FusedLnLinearError> {
        let hd = self.cfg.hidden_dim;
        let od = self.cfg.out_dim;
        let eps = self.cfg.eps;

        if input.len() != batch * hd {
            return Err(FusedLnLinearError::ShapeMismatch {
                context: format!(
                    "input: expected {}×{hd}={}, got {}",
                    batch,
                    batch * hd,
                    input.len()
                ),
            });
        }
        if gamma.len() != hd || beta.len() != hd {
            return Err(FusedLnLinearError::ShapeMismatch {
                context: "gamma/beta length != hidden_dim".into(),
            });
        }
        if weight.len() != od * hd {
            return Err(FusedLnLinearError::ShapeMismatch {
                context: format!(
                    "weight: expected {od}×{hd}={}, got {}",
                    od * hd,
                    weight.len()
                ),
            });
        }

        let has_bias = !bias_linear.is_empty();
        if has_bias && bias_linear.len() != od {
            return Err(FusedLnLinearError::ShapeMismatch {
                context: "bias_linear length != out_dim".into(),
            });
        }

        let mut output = vec![0.0f32; batch * od];

        for b in 0..batch {
            let row = &input[b * hd..(b + 1) * hd];

            // Normalised (or pass-through) hidden state.
            let normed: Vec<f32> = if self.cfg.fused {
                let mean =
                    row.iter().sum::<f32>() / hd as f32;
                let var = row
                    .iter()
                    .map(|x| (x - mean) * (x - mean))
                    .sum::<f32>()
                    / hd as f32;
                let inv_std = 1.0 / (var + eps).sqrt();

                row.iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        (x - mean) * inv_std * gamma[i] + beta[i]
                    })
                    .collect()
            } else {
                row.to_vec()
            };

            // Linear projection.
            for o in 0..od {
                let mut acc = 0.0f32;
                for i in 0..hd {
                    acc += weight[o * hd + i] * normed[i];
                }
                if has_bias {
                    acc += bias_linear[o];
                }
                output[b * od + o] = acc;
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Separate reference implementations
// ---------------------------------------------------------------------------

/// Standalone LayerNorm (for correctness comparison).
pub fn layer_norm_cpu(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch: usize,
    hidden_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * hidden_dim];
    for b in 0..batch {
        let row = &input[b * hidden_dim..(b + 1) * hidden_dim];
        let mean = row.iter().sum::<f32>() / hidden_dim as f32;
        let var = row
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f32>()
            / hidden_dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..hidden_dim {
            out[b * hidden_dim + i] =
                (row[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
    out
}

/// Standalone linear projection (for correctness comparison).
pub fn linear_cpu(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    hidden_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_dim];
    for b in 0..batch {
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for i in 0..hidden_dim {
                acc +=
                    weight[o * hidden_dim + i]
                        * input[b * hidden_dim + i];
            }
            if !bias.is_empty() {
                acc += bias[o];
            }
            out[b * out_dim + o] = acc;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(batch: usize, dim: usize) -> Vec<f32> {
        (0..batch * dim)
            .map(|i| ((i * 7 + 3) % 19) as f32 * 0.1 - 0.9)
            .collect()
    }

    fn make_weight(out_dim: usize, hidden_dim: usize) -> Vec<f32> {
        (0..out_dim * hidden_dim)
            .map(|i| ((i * 13 + 5) % 23) as f32 * 0.05 - 0.55)
            .collect()
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    // ---- Fused matches separate LN+Linear -------------------------

    #[test]
    fn fused_matches_separate() {
        let (batch, hd, od) = (4, 16, 8);
        let input = make_input(batch, hd);
        let gamma = vec![1.0f32; hd];
        let beta = vec![0.0f32; hd];
        let weight = make_weight(od, hd);
        let bias = vec![0.1f32; od];

        let cfg = FusedLayerNormLinearConfig::new(hd, od).unwrap();
        let fused = FusedLayerNormLinear::new(cfg);
        let fused_out = fused
            .forward(&input, &gamma, &beta, &weight, &bias, batch)
            .unwrap();

        let ln_out =
            layer_norm_cpu(&input, &gamma, &beta, batch, hd, 1e-5);
        let sep_out =
            linear_cpu(&ln_out, &weight, &bias, batch, hd, od);

        assert!(
            approx_eq(&fused_out, &sep_out, 1e-4),
            "max delta = {}",
            max_delta(&fused_out, &sep_out)
        );
    }

    // ---- Fusion disabled: linear only -----------------------------

    #[test]
    fn unfused_is_linear_only() {
        let (batch, hd, od) = (2, 8, 4);
        let input = make_input(batch, hd);
        let gamma = vec![2.0f32; hd]; // should be ignored
        let beta = vec![3.0f32; hd]; // should be ignored
        let weight = make_weight(od, hd);
        let bias: Vec<f32> = vec![];

        let cfg = FusedLayerNormLinearConfig::new(hd, od)
            .unwrap()
            .with_fused(false);
        let fused = FusedLayerNormLinear::new(cfg);
        let fused_out = fused
            .forward(&input, &gamma, &beta, &weight, &bias, batch)
            .unwrap();

        let linear_out =
            linear_cpu(&input, &weight, &bias, batch, hd, od);

        assert!(approx_eq(&fused_out, &linear_out, 1e-5));
    }

    // ---- Non-trivial gamma/beta -----------------------------------

    #[test]
    fn non_trivial_affine_params() {
        let (batch, hd, od) = (3, 12, 6);
        let input = make_input(batch, hd);
        let gamma: Vec<f32> =
            (0..hd).map(|i| 0.5 + i as f32 * 0.1).collect();
        let beta: Vec<f32> =
            (0..hd).map(|i| -0.3 + i as f32 * 0.05).collect();
        let weight = make_weight(od, hd);
        let bias = vec![0.0f32; od];

        let cfg = FusedLayerNormLinearConfig::new(hd, od).unwrap();
        let fused = FusedLayerNormLinear::new(cfg);
        let fused_out = fused
            .forward(&input, &gamma, &beta, &weight, &bias, batch)
            .unwrap();

        let ln_out = layer_norm_cpu(
            &input, &gamma, &beta, batch, hd, 1e-5,
        );
        let sep_out =
            linear_cpu(&ln_out, &weight, &bias, batch, hd, od);

        assert!(
            approx_eq(&fused_out, &sep_out, 1e-4),
            "max delta = {}",
            max_delta(&fused_out, &sep_out)
        );
    }

    // ---- Single-token batch (batch=1) -----------------------------

    #[test]
    fn single_token_batch() {
        let (batch, hd, od) = (1, 32, 16);
        let input = make_input(batch, hd);
        let gamma = vec![1.0f32; hd];
        let beta = vec![0.0f32; hd];
        let weight = make_weight(od, hd);
        let bias = vec![0.0f32; od];

        let cfg = FusedLayerNormLinearConfig::new(hd, od).unwrap();
        let fused = FusedLayerNormLinear::new(cfg);
        let fused_out = fused
            .forward(&input, &gamma, &beta, &weight, &bias, batch)
            .unwrap();

        let ln_out =
            layer_norm_cpu(&input, &gamma, &beta, batch, hd, 1e-5);
        let sep_out =
            linear_cpu(&ln_out, &weight, &bias, batch, hd, od);

        assert!(approx_eq(&fused_out, &sep_out, 1e-4));
    }

    // ---- Config validation ----------------------------------------

    #[test]
    fn zero_hidden_dim_rejected() {
        assert!(FusedLayerNormLinearConfig::new(0, 8).is_err());
    }

    #[test]
    fn zero_out_dim_rejected() {
        assert!(FusedLayerNormLinearConfig::new(8, 0).is_err());
    }

    // ---- Shape mismatch detection ---------------------------------

    #[test]
    fn shape_mismatch_input() {
        let cfg =
            FusedLayerNormLinearConfig::new(8, 4).unwrap();
        let fused = FusedLayerNormLinear::new(cfg);
        let bad_input = vec![0.0f32; 5]; // wrong size
        let gamma = vec![1.0f32; 8];
        let beta = vec![0.0f32; 8];
        let weight = vec![0.0f32; 32];
        assert!(fused
            .forward(&bad_input, &gamma, &beta, &weight, &[], 1)
            .is_err());
    }

    // ---- Kernel source loads --------------------------------------

    #[test]
    fn kernel_source_loads() {
        let src = FusedLayerNormLinearConfig::kernel_source();
        assert!(src.contains("fused_layernorm_linear"));
        assert!(src.contains("WG_SIZE"));
    }
}
