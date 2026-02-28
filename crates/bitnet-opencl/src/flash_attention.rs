//! Flash Attention for OpenCL — memory-efficient tiled attention.
//!
//! Implements the Flash Attention algorithm (Dao et al., 2022) adapted for
//! OpenCL.  Instead of materialising the full N×N attention matrix in global
//! memory, the kernel processes Q/KV in tiles of `block_size` so intermediate
//! scores stay in local (shared) memory, reducing memory usage from O(N²) to
//! O(N).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │  Host (FlashAttentionConfig / Orchestrator)  │
//! │  ─ splits Q into row-blocks of BLOCK_SIZE    │
//! │  ─ iterates over KV blocks                   │
//! │  ─ launches flash_attention kernel            │
//! └──────────────┬──────────────────────────────┘
//!                │  OpenCL enqueue
//!                ▼
//! ┌─────────────────────────────────────────────┐
//! │  flash_attention.cl  (device kernel)         │
//! │  ─ tiled QKV dot products in local memory    │
//! │  ─ online softmax (running max + exp sum)    │
//! │  ─ incremental output accumulation           │
//! └─────────────────────────────────────────────┘
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configurable parameters for the flash-attention kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashAttentionConfig {
    /// Tile / block size for the Q and KV dimensions.
    /// Must be a power of two. Default: 64.
    pub block_size: usize,
    /// Whether to apply a causal mask (each query can only attend to
    /// earlier positions).
    pub causal: bool,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of each head (d_k).
    pub head_dim: usize,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            causal: false,
            num_heads: 1,
            head_dim: 64,
        }
    }
}

impl FlashAttentionConfig {
    /// Create a new config, validating invariants.
    pub fn new(
        block_size: usize,
        causal: bool,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self, FlashAttentionError> {
        if block_size == 0 || (block_size & (block_size - 1)) != 0 {
            return Err(FlashAttentionError::InvalidBlockSize(block_size));
        }
        if num_heads == 0 {
            return Err(FlashAttentionError::InvalidConfig(
                "num_heads must be > 0".into(),
            ));
        }
        if head_dim == 0 {
            return Err(FlashAttentionError::InvalidConfig(
                "head_dim must be > 0".into(),
            ));
        }
        Ok(Self {
            block_size,
            causal,
            num_heads,
            head_dim,
        })
    }

    /// Scaling factor: `1 / sqrt(head_dim)`.
    #[inline]
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Return the kernel source with `BLOCK_SIZE` baked in.
    pub fn kernel_source(&self) -> String {
        let raw = include_str!("../kernels/flash_attention.cl");
        raw.replace(
            "#define BLOCK_SIZE 64",
            &format!("#define BLOCK_SIZE {}", self.block_size),
        )
    }

    /// Name of the kernel entry-point to use.
    pub fn kernel_name(&self) -> &'static str {
        if self.causal {
            "flash_attention_causal"
        } else {
            "flash_attention_forward"
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors specific to flash-attention operations.
#[derive(Debug, Clone)]
pub enum FlashAttentionError {
    /// Block size must be a power of two.
    InvalidBlockSize(usize),
    /// Tensor shape does not match the expected layout.
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Generic configuration error.
    InvalidConfig(String),
}

impl fmt::Display for FlashAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBlockSize(bs) => {
                write!(f, "block_size must be a power of two, got {bs}")
            }
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "shape mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::InvalidConfig(msg) => {
                write!(f, "invalid config: {msg}")
            }
        }
    }
}

impl std::error::Error for FlashAttentionError {}

// ---------------------------------------------------------------------------
// Host-side orchestration (CPU reference)
// ---------------------------------------------------------------------------

/// Host-side orchestration of flash attention.
///
/// This is the reference CPU implementation that mirrors the OpenCL kernel.
/// Used for correctness testing and platforms without an OpenCL runtime.
pub struct FlashAttentionOrchestrator {
    config: FlashAttentionConfig,
}

impl FlashAttentionOrchestrator {
    pub fn new(config: FlashAttentionConfig) -> Self {
        Self { config }
    }

    /// Run flash attention on CPU, returning the output tensor.
    ///
    /// # Arguments
    /// * `q` — `[num_heads, seq_len_q, head_dim]` row-major
    /// * `k` — `[num_heads, seq_len_kv, head_dim]`
    /// * `v` — `[num_heads, seq_len_kv, head_dim]`
    /// * `seq_len_q`  — query sequence length
    /// * `seq_len_kv` — key/value sequence length
    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len_q: usize,
        seq_len_kv: usize,
    ) -> Result<Vec<f32>, FlashAttentionError> {
        let cfg = &self.config;
        let expected_q = cfg.num_heads * seq_len_q * cfg.head_dim;
        let expected_kv = cfg.num_heads * seq_len_kv * cfg.head_dim;

        if q.len() != expected_q {
            return Err(FlashAttentionError::ShapeMismatch {
                expected: vec![cfg.num_heads, seq_len_q, cfg.head_dim],
                actual: vec![q.len()],
            });
        }
        if k.len() != expected_kv {
            return Err(FlashAttentionError::ShapeMismatch {
                expected: vec![cfg.num_heads, seq_len_kv, cfg.head_dim],
                actual: vec![k.len()],
            });
        }
        if v.len() != expected_kv {
            return Err(FlashAttentionError::ShapeMismatch {
                expected: vec![cfg.num_heads, seq_len_kv, cfg.head_dim],
                actual: vec![v.len()],
            });
        }

        let mut output = vec![0.0f32; expected_q];
        let scale = cfg.scale();
        let block_size = cfg.block_size;

        for head in 0..cfg.num_heads {
            let q_off = head * seq_len_q * cfg.head_dim;
            let kv_off = head * seq_len_kv * cfg.head_dim;

            for q_row in 0..seq_len_q {
                let causal_limit = if cfg.causal {
                    (q_row + 1).min(seq_len_kv)
                } else {
                    seq_len_kv
                };
                let num_kv_blocks =
                    (causal_limit + block_size - 1) / block_size;

                let mut m_i: f32 = f32::NEG_INFINITY;
                let mut l_i: f32 = 0.0;
                let o_base = q_off + q_row * cfg.head_dim;

                for kv_block in 0..num_kv_blocks {
                    let kv_start = kv_block * block_size;
                    let kv_end =
                        (kv_start + block_size).min(causal_limit);

                    // Pass 1: scores + block max.
                    let mut m_ij: f32 = f32::NEG_INFINITY;
                    let mut scores =
                        Vec::with_capacity(kv_end - kv_start);

                    for kv in kv_start..kv_end {
                        let mut s = 0.0f32;
                        for d in 0..cfg.head_dim {
                            s += q[q_off + q_row * cfg.head_dim + d]
                                * k[kv_off + kv * cfg.head_dim + d];
                        }
                        s *= scale;
                        if s > m_ij {
                            m_ij = s;
                        }
                        scores.push(s);
                    }

                    // Online softmax update.
                    let m_new = m_i.max(m_ij);
                    let alpha = (m_i - m_new).exp();
                    let mut l_ij = 0.0f32;

                    // Rescale previous accumulator once.
                    for d in 0..cfg.head_dim {
                        output[o_base + d] *= alpha;
                    }

                    // Pass 2: accumulate output.
                    for (idx, kv) in (kv_start..kv_end).enumerate() {
                        let p = (scores[idx] - m_new).exp();
                        l_ij += p;
                        for d in 0..cfg.head_dim {
                            output[o_base + d] +=
                                p * v[kv_off
                                    + kv * cfg.head_dim
                                    + d];
                        }
                    }

                    l_i = l_i * alpha + l_ij;
                    m_i = m_new;
                }

                // Final normalisation.
                if l_i > 0.0 {
                    let inv_l = 1.0 / l_i;
                    for d in 0..cfg.head_dim {
                        output[o_base + d] *= inv_l;
                    }
                }
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Naive reference (for correctness testing)
// ---------------------------------------------------------------------------

/// Standard O(N²) attention for reference/test comparisons.
pub fn naive_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    seq_len_q: usize,
    seq_len_kv: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; num_heads * seq_len_q * head_dim];

    for head in 0..num_heads {
        let q_off = head * seq_len_q * head_dim;
        let kv_off = head * seq_len_kv * head_dim;

        for q_row in 0..seq_len_q {
            let limit = if causal {
                (q_row + 1).min(seq_len_kv)
            } else {
                seq_len_kv
            };

            let mut scores = vec![f32::NEG_INFINITY; seq_len_kv];
            for kv in 0..limit {
                let mut s = 0.0f32;
                for d in 0..head_dim {
                    s += q[q_off + q_row * head_dim + d]
                        * k[kv_off + kv * head_dim + d];
                }
                scores[kv] = s * scale;
            }

            // Softmax.
            let max_s = scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let mut exp_scores = vec![0.0f32; seq_len_kv];
            for kv in 0..seq_len_kv {
                exp_scores[kv] = (scores[kv] - max_s).exp();
                sum += exp_scores[kv];
            }

            let o_base = q_off + q_row * head_dim;
            for kv in 0..seq_len_kv {
                let w = exp_scores[kv] / sum;
                for d in 0..head_dim {
                    output[o_base + d] +=
                        w * v[kv_off + kv * head_dim + d];
                }
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let n = num_heads * seq_len * head_dim;
        (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect()
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

    // ---- Correctness vs naive ------------------------------------------

    #[test]
    fn flash_matches_naive_single_head() {
        let (nh, sq, skv, hd) = (1, 8, 8, 16);
        let q = make_tensor(nh, sq, hd);
        let k = make_tensor(nh, skv, hd);
        let v = make_tensor(nh, skv, hd);

        let cfg =
            FlashAttentionConfig::new(4, false, nh, hd).unwrap();
        let orch = FlashAttentionOrchestrator::new(cfg);
        let flash = orch.forward(&q, &k, &v, sq, skv).unwrap();
        let naive =
            naive_attention(&q, &k, &v, nh, sq, skv, hd, false);

        assert!(
            approx_eq(&flash, &naive, 1e-4),
            "max delta = {}",
            max_delta(&flash, &naive)
        );
    }

    #[test]
    fn flash_matches_naive_multi_head() {
        let (nh, sq, skv, hd) = (4, 6, 6, 8);
        let q = make_tensor(nh, sq, hd);
        let k = make_tensor(nh, skv, hd);
        let v = make_tensor(nh, skv, hd);

        let cfg =
            FlashAttentionConfig::new(4, false, nh, hd).unwrap();
        let orch = FlashAttentionOrchestrator::new(cfg);
        let flash = orch.forward(&q, &k, &v, sq, skv).unwrap();
        let naive =
            naive_attention(&q, &k, &v, nh, sq, skv, hd, false);

        assert!(approx_eq(&flash, &naive, 1e-4));
    }

    #[test]
    fn flash_causal_matches_naive_causal() {
        let (nh, sq, skv, hd) = (2, 8, 8, 16);
        let q = make_tensor(nh, sq, hd);
        let k = make_tensor(nh, skv, hd);
        let v = make_tensor(nh, skv, hd);

        let cfg =
            FlashAttentionConfig::new(4, true, nh, hd).unwrap();
        let orch = FlashAttentionOrchestrator::new(cfg);
        let flash = orch.forward(&q, &k, &v, sq, skv).unwrap();
        let naive =
            naive_attention(&q, &k, &v, nh, sq, skv, hd, true);

        assert!(approx_eq(&flash, &naive, 1e-4));
    }

    // ---- Block sizes ---------------------------------------------------

    #[test]
    fn block_size_2_matches_naive() {
        let (nh, sq, skv, hd) = (1, 5, 5, 4);
        let q = make_tensor(nh, sq, hd);
        let k = make_tensor(nh, skv, hd);
        let v = make_tensor(nh, skv, hd);

        let cfg =
            FlashAttentionConfig::new(2, false, nh, hd).unwrap();
        let orch = FlashAttentionOrchestrator::new(cfg);
        let flash = orch.forward(&q, &k, &v, sq, skv).unwrap();
        let naive =
            naive_attention(&q, &k, &v, nh, sq, skv, hd, false);

        assert!(approx_eq(&flash, &naive, 1e-4));
    }

    #[test]
    fn block_size_128_matches_naive() {
        let (nh, sq, skv, hd) = (1, 10, 10, 8);
        let q = make_tensor(nh, sq, hd);
        let k = make_tensor(nh, skv, hd);
        let v = make_tensor(nh, skv, hd);

        let cfg =
            FlashAttentionConfig::new(128, false, nh, hd).unwrap();
        let orch = FlashAttentionOrchestrator::new(cfg);
        let flash = orch.forward(&q, &k, &v, sq, skv).unwrap();
        let naive =
            naive_attention(&q, &k, &v, nh, sq, skv, hd, false);

        assert!(approx_eq(&flash, &naive, 1e-4));
    }

    // ---- Edge cases ----------------------------------------------------

    #[test]
    fn single_token_query() {
        let (nh, sq, skv, hd) = (1, 1, 8, 4);
        let q = make_tensor(nh, sq, hd);
        let k = make_tensor(nh, skv, hd);
        let v = make_tensor(nh, skv, hd);

        let cfg =
            FlashAttentionConfig::new(4, false, nh, hd).unwrap();
        let orch = FlashAttentionOrchestrator::new(cfg);
        let flash = orch.forward(&q, &k, &v, sq, skv).unwrap();
        let naive =
            naive_attention(&q, &k, &v, nh, sq, skv, hd, false);

        assert!(approx_eq(&flash, &naive, 1e-4));
    }

    #[test]
    fn seq_len_not_multiple_of_block() {
        let (nh, sq, skv, hd) = (1, 7, 7, 4);
        let q = make_tensor(nh, sq, hd);
        let k = make_tensor(nh, skv, hd);
        let v = make_tensor(nh, skv, hd);

        let cfg =
            FlashAttentionConfig::new(4, false, nh, hd).unwrap();
        let orch = FlashAttentionOrchestrator::new(cfg);
        let flash = orch.forward(&q, &k, &v, sq, skv).unwrap();
        let naive =
            naive_attention(&q, &k, &v, nh, sq, skv, hd, false);

        assert!(approx_eq(&flash, &naive, 1e-4));
    }

    #[test]
    fn asymmetric_q_kv_lengths() {
        let (nh, sq, skv, hd) = (1, 3, 10, 4);
        let q = make_tensor(nh, sq, hd);
        let k = make_tensor(nh, skv, hd);
        let v = make_tensor(nh, skv, hd);

        let cfg =
            FlashAttentionConfig::new(4, false, nh, hd).unwrap();
        let orch = FlashAttentionOrchestrator::new(cfg);
        let flash = orch.forward(&q, &k, &v, sq, skv).unwrap();
        let naive =
            naive_attention(&q, &k, &v, nh, sq, skv, hd, false);

        assert!(approx_eq(&flash, &naive, 1e-4));
    }

    // ---- Config validation ---------------------------------------------

    #[test]
    fn invalid_block_size_rejected() {
        assert!(
            FlashAttentionConfig::new(0, false, 1, 64).is_err()
        );
        assert!(
            FlashAttentionConfig::new(3, false, 1, 64).is_err()
        );
        assert!(
            FlashAttentionConfig::new(5, false, 1, 64).is_err()
        );
    }

    #[test]
    fn zero_num_heads_rejected() {
        assert!(
            FlashAttentionConfig::new(64, false, 0, 64).is_err()
        );
    }

    #[test]
    fn zero_head_dim_rejected() {
        assert!(
            FlashAttentionConfig::new(64, false, 1, 0).is_err()
        );
    }

    #[test]
    fn shape_mismatch_detected() {
        let cfg =
            FlashAttentionConfig::new(4, false, 2, 8).unwrap();
        let orch = FlashAttentionOrchestrator::new(cfg);
        let bad_q = vec![0.0f32; 10];
        let k = make_tensor(2, 4, 8);
        let v = make_tensor(2, 4, 8);
        assert!(orch.forward(&bad_q, &k, &v, 4, 4).is_err());
    }

    // ---- Kernel source -------------------------------------------------

    #[test]
    fn kernel_source_contains_block_size() {
        let cfg =
            FlashAttentionConfig::new(32, false, 1, 64).unwrap();
        let src = cfg.kernel_source();
        assert!(src.contains("#define BLOCK_SIZE 32"));
        assert!(!src.contains("#define BLOCK_SIZE 64"));
    }

    #[test]
    fn kernel_name_reflects_causal() {
        let non_causal =
            FlashAttentionConfig::new(64, false, 1, 64).unwrap();
        assert_eq!(
            non_causal.kernel_name(),
            "flash_attention_forward"
        );

        let causal =
            FlashAttentionConfig::new(64, true, 1, 64).unwrap();
        assert_eq!(
            causal.kernel_name(),
            "flash_attention_causal"
        );
    }
}
