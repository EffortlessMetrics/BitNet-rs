//! Dense (non-quantized) transformer forward pass
//!
//! Implements the standard pre-norm transformer block for FP32 inference,
//! enabling support for dense SLM models (Phi-4, Qwen, Gemma, Mistral, LLaMA).
//!
//! Architecture: RMSNorm → Attention → Residual → RMSNorm → SwiGLU FFN → Residual

use anyhow::{Context, Result};
use bitnet_common::{BitNetTensor, Device, Tensor};

// ── RMSNorm ──────────────────────────────────────────────────────────────────

/// Root Mean Square Layer Normalization.
///
/// Normalizes `x` to unit RMS then scales element-wise by `weight`.
/// `rms_norm(x) = (x / sqrt(mean(x²) + eps)) * weight`
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len(), "rms_norm: x.len() != weight.len()");
    let n = x.len() as f32;
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter().zip(weight.iter()).map(|(xi, wi)| xi * inv_rms * wi).collect()
}

// ── SiLU activation ──────────────────────────────────────────────────────────

/// SiLU (Sigmoid Linear Unit): `silu(x) = x * sigmoid(x)`
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ── Dense linear layer ───────────────────────────────────────────────────────

/// Standard FP32 linear layer (no quantization).
///
/// `weight` is stored row-major as `[out_features, in_features]`.
/// Optional `bias` of length `out_features`.
#[derive(Debug, Clone)]
pub struct DenseLinear {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl DenseLinear {
    pub fn new(
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        assert_eq!(
            weight.len(),
            in_features * out_features,
            "weight length must be in_features * out_features"
        );
        if let Some(ref b) = bias {
            assert_eq!(b.len(), out_features, "bias length must be out_features");
        }
        Self { weight, bias, in_features, out_features }
    }

    /// Forward pass: `output = x @ W^T + bias`
    ///
    /// `x` shape: `[..., in_features]` (flattened to 2-D internally).
    /// Returns shape: `[..., out_features]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len() % self.in_features, 0, "input length must be a multiple of in_features");
        let batch = x.len() / self.in_features;
        let mut out = vec![0.0f32; batch * self.out_features];

        for b in 0..batch {
            let x_row = &x[b * self.in_features..(b + 1) * self.in_features];
            for o in 0..self.out_features {
                let w_row = &self.weight[o * self.in_features..(o + 1) * self.in_features];
                let mut acc = 0.0f32;
                for (xi, wi) in x_row.iter().zip(w_row.iter()) {
                    acc += xi * wi;
                }
                if let Some(ref bias) = self.bias {
                    acc += bias[o];
                }
                out[b * self.out_features + o] = acc;
            }
        }
        out
    }
}

// ── SwiGLU FFN ───────────────────────────────────────────────────────────────

/// SwiGLU Feed-Forward Network: `FFN(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down`
#[derive(Debug, Clone)]
pub struct DenseFFN {
    pub gate_proj: DenseLinear,
    pub up_proj: DenseLinear,
    pub down_proj: DenseLinear,
}

impl DenseFFN {
    pub fn new(gate_proj: DenseLinear, up_proj: DenseLinear, down_proj: DenseLinear) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

    /// Forward: `down(silu(gate(x)) * up(x))`
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        // element-wise silu(gate) * up
        let hidden: Vec<f32> = gate.iter().zip(up.iter()).map(|(&g, &u)| silu(g) * u).collect();
        self.down_proj.forward(&hidden)
    }
}

// ── Dense Attention (GQA) ────────────────────────────────────────────────────

/// Configuration for dense attention.
#[derive(Debug, Clone)]
pub struct DenseAttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// Dense (FP32) multi-head attention with Grouped-Query Attention support.
///
/// Simplified single-token causal attention (no KV-cache) for forward-pass
/// correctness validation. Production KV-cache integration is separate.
#[derive(Debug, Clone)]
pub struct DenseAttention {
    pub config: DenseAttentionConfig,
    pub q_proj: DenseLinear,
    pub k_proj: DenseLinear,
    pub v_proj: DenseLinear,
    pub o_proj: DenseLinear,
}

impl DenseAttention {
    /// Forward pass for a single sequence (no batch dim).
    ///
    /// `x` shape: `[seq_len, hidden_size]` (row-major).
    /// Returns: `[seq_len, hidden_size]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let seq_len = x.len() / self.config.hidden_size;
        assert_eq!(x.len(), seq_len * self.config.hidden_size);

        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let kv_group_size = num_heads / num_kv_heads;

        // Project Q, K, V
        let q_flat = self.q_proj.forward(x); // [seq_len, num_heads * head_dim]
        let k_flat = self.k_proj.forward(x); // [seq_len, num_kv_heads * head_dim]
        let v_flat = self.v_proj.forward(x); // [seq_len, num_kv_heads * head_dim]

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute attention per head with causal mask.
        // Output: [seq_len, num_heads * head_dim]
        let mut attn_out = vec![0.0f32; seq_len * num_heads * head_dim];

        for h in 0..num_heads {
            let kv_h = h / kv_group_size; // which KV head this query head maps to

            for i in 0..seq_len {
                // Compute softmax of scores[0..=i] (causal)
                let mut scores = vec![f32::NEG_INFINITY; seq_len];
                for j in 0..=i {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let qi = q_flat[i * num_heads * head_dim + h * head_dim + d];
                        let kj = k_flat[j * num_kv_heads * head_dim + kv_h * head_dim + d];
                        dot += qi * kj;
                    }
                    scores[j] = dot * scale;
                }

                // Numerically-stable softmax over [0..=i]
                let max_score = scores[..=i].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                let mut weights = vec![0.0f32; i + 1];
                for j in 0..=i {
                    let w = (scores[j] - max_score).exp();
                    weights[j] = w;
                    sum_exp += w;
                }
                for w in weights.iter_mut() {
                    *w /= sum_exp;
                }

                // Weighted sum of V
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for j in 0..=i {
                        let vj = v_flat[j * num_kv_heads * head_dim + kv_h * head_dim + d];
                        val += weights[j] * vj;
                    }
                    attn_out[i * num_heads * head_dim + h * head_dim + d] = val;
                }
            }
        }

        // Output projection
        self.o_proj.forward(&attn_out)
    }
}

// ── Dense Transformer Block ──────────────────────────────────────────────────

/// Pre-norm transformer block for dense (FP32) models.
///
/// ```text
/// residual = x
/// x = RMSNorm(x, attn_norm_weight)
/// x = Attention(x)
/// x = x + residual
/// residual = x
/// x = RMSNorm(x, ffn_norm_weight)
/// x = FFN(x)
/// x = x + residual
/// ```
#[derive(Debug, Clone)]
pub struct DenseTransformerBlock {
    pub attn_norm_weight: Vec<f32>,
    pub ffn_norm_weight: Vec<f32>,
    pub attention: DenseAttention,
    pub ffn: DenseFFN,
    pub norm_eps: f32,
    pub hidden_size: usize,
}

impl DenseTransformerBlock {
    /// Forward pass for a single sequence.
    ///
    /// `x`: `[seq_len, hidden_size]` row-major.
    /// Returns: `[seq_len, hidden_size]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let seq_len = x.len() / self.hidden_size;
        assert_eq!(x.len(), seq_len * self.hidden_size);
        let dim = self.hidden_size;

        // ── Attention sub-block ──────────────────────────────────────────
        let mut normed = Vec::with_capacity(x.len());
        for t in 0..seq_len {
            let row = &x[t * dim..(t + 1) * dim];
            normed.extend(rms_norm(row, &self.attn_norm_weight, self.norm_eps));
        }
        let attn_out = self.attention.forward(&normed);

        // Residual
        let mut h: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

        // ── FFN sub-block ────────────────────────────────────────────────
        let mut normed_ffn = Vec::with_capacity(h.len());
        for t in 0..seq_len {
            let row = &h[t * dim..(t + 1) * dim];
            normed_ffn.extend(rms_norm(row, &self.ffn_norm_weight, self.norm_eps));
        }
        let ffn_out = self.ffn.forward(&normed_ffn);

        // Residual
        for (hi, fi) in h.iter_mut().zip(ffn_out.iter()) {
            *hi += fi;
        }

        h
    }
}

// ── Dense Model (multi-layer) ────────────────────────────────────────────────

/// Complete dense transformer model: embedding → N blocks → final norm → lm_head.
#[derive(Debug, Clone)]
pub struct DenseModel {
    pub blocks: Vec<DenseTransformerBlock>,
    pub final_norm_weight: Vec<f32>,
    pub norm_eps: f32,
    pub lm_head: DenseLinear,
    pub tok_embeddings: Vec<f32>,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl DenseModel {
    /// Run full forward pass: tokens → logits.
    pub fn forward(&self, token_ids: &[usize]) -> Vec<f32> {
        let seq_len = token_ids.len();
        let dim = self.hidden_size;

        // Embedding lookup
        let mut hidden = Vec::with_capacity(seq_len * dim);
        for &tid in token_ids {
            assert!(tid < self.vocab_size, "token_id {} >= vocab_size {}", tid, self.vocab_size);
            hidden.extend_from_slice(&self.tok_embeddings[tid * dim..(tid + 1) * dim]);
        }

        // Transformer blocks
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }

        // Final RMSNorm
        let mut normed = Vec::with_capacity(hidden.len());
        for t in 0..seq_len {
            let row = &hidden[t * dim..(t + 1) * dim];
            normed.extend(rms_norm(row, &self.final_norm_weight, self.norm_eps));
        }

        // LM head (only last position for causal LM)
        let last_hidden = &normed[(seq_len - 1) * dim..seq_len * dim];
        self.lm_head.forward(last_hidden)
    }
}

// ── BitNetTensor integration ─────────────────────────────────────────────────

/// Execute a single dense transformer block on `BitNetTensor` inputs.
///
/// This bridges the raw `f32`-slice implementation to the `BitNetTensor`
/// world used by the rest of the inference engine.
pub fn dense_block_forward_tensor(
    block: &DenseTransformerBlock,
    input: &BitNetTensor,
) -> Result<BitNetTensor> {
    let shape = input.shape().to_vec();
    anyhow::ensure!(shape.len() == 2, "expected 2-D input [seq_len, hidden_size], got {:?}", shape);
    let seq_len = shape[0];
    let hidden = shape[1];
    anyhow::ensure!(
        hidden == block.hidden_size,
        "hidden dim mismatch: {} vs {}",
        hidden,
        block.hidden_size
    );

    let x_candle = input.to_candle().context("to_candle")?;
    let x_flat = x_candle.flatten_all()?.to_vec1::<f32>().context("flatten")?;

    let out = block.forward(&x_flat);
    let device = Device::Cpu;
    BitNetTensor::from_slice(&out, &[seq_len, hidden], &device)
        .map_err(|e| anyhow::anyhow!("from_slice: {e}"))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    // ── helpers ──────────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn assert_vec_approx(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch {} vs {}", a.len(), b.len());
        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                approx_eq(*ai, *bi, tol),
                "{msg}: index {i}: {ai} vs {bi} (diff {})",
                (ai - bi).abs()
            );
        }
    }

    /// Build a minimal DenseLinear with identity-like weight (for dim=dim).
    fn identity_linear(dim: usize) -> DenseLinear {
        let mut w = vec![0.0f32; dim * dim];
        for i in 0..dim {
            w[i * dim + i] = 1.0;
        }
        DenseLinear::new(w, None, dim, dim)
    }

    /// Build a small deterministic transformer block for testing.
    fn make_test_block(dim: usize, intermediate: usize, num_heads: usize) -> DenseTransformerBlock {
        let num_kv_heads = num_heads; // MHA for simplicity
        let head_dim = dim / num_heads;

        let attn_cfg = DenseAttentionConfig { hidden_size: dim, num_heads, num_kv_heads, head_dim };

        let attention = DenseAttention {
            config: attn_cfg,
            q_proj: identity_linear(dim),
            k_proj: identity_linear(dim),
            v_proj: identity_linear(dim),
            o_proj: identity_linear(dim),
        };

        // FFN with small deterministic weights
        let gate_w: Vec<f32> =
            (0..intermediate * dim).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let up_w: Vec<f32> =
            (0..intermediate * dim).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
        let down_w: Vec<f32> =
            (0..dim * intermediate).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();

        let ffn = DenseFFN::new(
            DenseLinear::new(gate_w, None, dim, intermediate),
            DenseLinear::new(up_w, None, dim, intermediate),
            DenseLinear::new(down_w, None, intermediate, dim),
        );

        DenseTransformerBlock {
            attn_norm_weight: vec![1.0; dim],
            ffn_norm_weight: vec![1.0; dim],
            attention,
            ffn,
            norm_eps: EPS,
            hidden_size: dim,
        }
    }

    // ── RMSNorm tests ───────────────────────────────────────────────────

    #[test]
    fn test_rms_norm_unit_weight() {
        let x = vec![3.0, 4.0];
        let w = vec![1.0, 1.0];
        let out = rms_norm(&x, &w, EPS);
        // RMS = sqrt((9+16)/2 + eps) ≈ sqrt(12.5) ≈ 3.5355
        let rms = (12.5f32 + EPS).sqrt();
        assert_vec_approx(&out, &[3.0 / rms, 4.0 / rms], 1e-4, "rms_norm unit weight");
    }

    #[test]
    fn test_rms_norm_scaled_weight() {
        let x = vec![1.0, 0.0, -1.0];
        let w = vec![2.0, 3.0, 0.5];
        let out = rms_norm(&x, &w, EPS);
        let rms = (2.0f32 / 3.0 + EPS).sqrt();
        let expected = [1.0 / rms * 2.0, 0.0, -1.0 / rms * 0.5];
        assert_vec_approx(&out, &expected, 1e-4, "rms_norm scaled weight");
    }

    #[test]
    fn test_rms_norm_all_zeros() {
        let x = vec![0.0, 0.0];
        let w = vec![1.0, 1.0];
        let out = rms_norm(&x, &w, EPS);
        // Should produce near-zero output (eps prevents division by zero)
        for v in &out {
            assert!(v.abs() < 1e-2, "expected near-zero, got {v}");
        }
    }

    // ── SiLU tests ──────────────────────────────────────────────────────

    #[test]
    fn test_silu_values() {
        assert!(approx_eq(silu(0.0), 0.0, 1e-6), "silu(0) = 0");
        // silu(1) = 1/(1+e^-1) ≈ 0.7311
        assert!(approx_eq(silu(1.0), 0.7311, 1e-3), "silu(1)");
        // silu(-1) = -1/(1+e^1) ≈ -0.2689
        assert!(approx_eq(silu(-1.0), -0.2689, 1e-3), "silu(-1)");
        // Large positive → x
        assert!(silu(10.0) > 9.99, "silu(10) ≈ 10");
        // Large negative → 0
        assert!(silu(-10.0).abs() < 0.001, "silu(-10) ≈ 0");
    }

    // ── DenseLinear tests ───────────────────────────────────────────────

    #[test]
    fn test_dense_linear_identity() {
        let lin = identity_linear(3);
        let x = vec![1.0, 2.0, 3.0];
        let out = lin.forward(&x);
        assert_vec_approx(&out, &x, 1e-6, "identity linear");
    }

    #[test]
    fn test_dense_linear_with_bias() {
        // W = [[1,0],[0,1]], bias = [0.5, -0.5]
        let lin = DenseLinear::new(vec![1.0, 0.0, 0.0, 1.0], Some(vec![0.5, -0.5]), 2, 2);
        let out = lin.forward(&[3.0, 4.0]);
        assert_vec_approx(&out, &[3.5, 3.5], 1e-6, "linear with bias");
    }

    #[test]
    fn test_dense_linear_batch() {
        let lin = DenseLinear::new(vec![1.0, 2.0, 3.0, 4.0], None, 2, 2);
        // batch of 2: [1,0] and [0,1]
        let out = lin.forward(&[1.0, 0.0, 0.0, 1.0]);
        // row 0: [1*1+0*0, 1*3+0*4] = [1, 3]
        // row 1: [0*1+1*2, 0*3+1*4] = [2, 4]
        assert_vec_approx(&out, &[1.0, 3.0, 2.0, 4.0], 1e-6, "linear batch");
    }

    // ── SwiGLU FFN tests ────────────────────────────────────────────────

    #[test]
    fn test_ffn_silu_matches_manual() {
        let dim = 2;
        let inter = 3;
        // gate, up: identity-ish; down: identity-ish
        let gate_w = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 3×2
        let up_w = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 3×2
        let down_w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2×3

        let ffn = DenseFFN::new(
            DenseLinear::new(gate_w, None, dim, inter),
            DenseLinear::new(up_w, None, dim, inter),
            DenseLinear::new(down_w, None, inter, dim),
        );

        let x = vec![2.0, 3.0];
        let out = ffn.forward(&x);

        // Manual: gate=[2,3,0], up=[2,3,0]
        // silu(gate)*up = [silu(2)*2, silu(3)*3, 0]
        let s2 = silu(2.0);
        let s3 = silu(3.0);
        // down @ [s2*2, s3*3, 0] with down_w = [[1,0,0],[0,1,0]]
        let expected = [s2 * 2.0, s3 * 3.0];
        assert_vec_approx(&out, &expected, 1e-5, "ffn silu manual");
    }

    // ── DenseAttention tests ────────────────────────────────────────────

    #[test]
    fn test_attention_single_token() {
        let dim = 4;
        let num_heads = 2;
        let head_dim = 2;
        let cfg =
            DenseAttentionConfig { hidden_size: dim, num_heads, num_kv_heads: num_heads, head_dim };
        let attn = DenseAttention {
            config: cfg,
            q_proj: identity_linear(dim),
            k_proj: identity_linear(dim),
            v_proj: identity_linear(dim),
            o_proj: identity_linear(dim),
        };

        // Single token: self-attention with identity projections → output ≈ input
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = attn.forward(&x);
        // With identity Q/K/V/O and single token, attention weight is 1.0 on itself
        assert_vec_approx(&out, &x, 1e-5, "attn single token");
    }

    #[test]
    fn test_attention_gqa() {
        let dim = 4;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 1;
        let cfg = DenseAttentionConfig { hidden_size: dim, num_heads, num_kv_heads, head_dim };
        // Use identity projections (dim=4 for Q, dim=2 for K/V)
        let q_proj = identity_linear(dim);
        let k_w: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2×4
        let v_w = k_w.clone();
        let k_proj = DenseLinear::new(k_w, None, dim, num_kv_heads * head_dim);
        let v_proj = DenseLinear::new(v_w, None, dim, num_kv_heads * head_dim);

        let attn =
            DenseAttention { config: cfg, q_proj, k_proj, v_proj, o_proj: identity_linear(dim) };

        // Single token — should not panic and produce valid output
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = attn.forward(&x);
        assert_eq!(out.len(), dim, "GQA output length");
        // All values should be finite
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "GQA output[{i}] not finite: {v}");
        }
    }

    // ── Transformer Block tests ─────────────────────────────────────────

    #[test]
    fn test_block_residual_connection() {
        // With zero-weight FFN gates, FFN output is zero, so block ≈ attn residual.
        let dim = 4;
        let inter = 4;
        let num_heads = 2;
        let head_dim = dim / num_heads;

        let attn_cfg =
            DenseAttentionConfig { hidden_size: dim, num_heads, num_kv_heads: num_heads, head_dim };
        let attention = DenseAttention {
            config: attn_cfg,
            q_proj: identity_linear(dim),
            k_proj: identity_linear(dim),
            v_proj: identity_linear(dim),
            o_proj: identity_linear(dim),
        };

        // Zero gate weights → silu(0)*up = 0 → FFN contributes nothing
        let ffn = DenseFFN::new(
            DenseLinear::new(vec![0.0; inter * dim], None, dim, inter),
            DenseLinear::new(vec![0.0; inter * dim], None, dim, inter),
            DenseLinear::new(vec![0.0; dim * inter], None, inter, dim),
        );

        let block = DenseTransformerBlock {
            attn_norm_weight: vec![1.0; dim],
            ffn_norm_weight: vec![1.0; dim],
            attention,
            ffn,
            norm_eps: EPS,
            hidden_size: dim,
        };

        // Single token: attn(rms_norm(x)) + x, then + ffn(0) = attn(rms_norm(x)) + x
        // With identity projections and single token, attn returns rms_norm(x).
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = block.forward(&x);
        assert_eq!(out.len(), dim);
        // Output should be x + rms_norm(x) + 0  (residual + attn + ffn)
        let rms_x = rms_norm(&x, &vec![1.0; dim], EPS);
        let expected: Vec<f32> = x.iter().zip(rms_x.iter()).map(|(a, b)| a + b).collect();
        assert_vec_approx(&out, &expected, 1e-4, "block residual");
    }

    #[test]
    fn test_multiple_blocks_stack() {
        let dim = 4;
        let inter = 8;
        let num_heads = 2;

        let block1 = make_test_block(dim, inter, num_heads);
        let block2 = make_test_block(dim, inter, num_heads);

        let x = vec![0.5, -0.5, 1.0, -1.0];
        let h1 = block1.forward(&x);
        let h2 = block2.forward(&h1);

        // Should produce finite values and differ from input
        assert_eq!(h2.len(), dim);
        for (i, v) in h2.iter().enumerate() {
            assert!(v.is_finite(), "stacked block output[{i}] not finite: {v}");
        }
        // Output should differ from input (blocks actually do computation)
        let diff: f32 = x.iter().zip(h2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "stacked blocks should change the hidden state");
    }

    // ── Numerical stability tests ───────────────────────────────────────

    #[test]
    fn test_numerical_stability_large_values() {
        let dim = 4;
        let block = make_test_block(dim, 8, 2);
        let x = vec![1000.0, -1000.0, 500.0, -500.0];
        let out = block.forward(&x);
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "large-value output[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_numerical_stability_small_values() {
        let dim = 4;
        let block = make_test_block(dim, 8, 2);
        let x = vec![1e-8, -1e-8, 1e-8, -1e-8];
        let out = block.forward(&x);
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "small-value output[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_rms_norm_stability_uniform_input() {
        // Uniform input → RMS = |c|, so output = sign(c) * weight
        let c = 42.0f32;
        let x = vec![c; 8];
        let w = vec![1.0; 8];
        let out = rms_norm(&x, &w, EPS);
        // Each output ≈ 1.0 (since c/rms(c) = 1 when all elements are the same)
        for (i, v) in out.iter().enumerate() {
            assert!(approx_eq(*v, 1.0, 1e-4), "uniform rms_norm[{i}] = {v}, expected 1.0");
        }
    }

    // ── BitNetTensor integration ────────────────────────────────────────

    #[test]
    fn test_dense_block_tensor_roundtrip() {
        let dim = 4;
        let block = make_test_block(dim, 8, 2);
        let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let device = Device::Cpu;
        let input = BitNetTensor::from_slice(&x_data, &[1, dim], &device).unwrap();
        let output = dense_block_forward_tensor(&block, &input).unwrap();
        let out_shape = output.shape().to_vec();
        assert_eq!(out_shape, vec![1, dim], "tensor output shape");
        // Values should be finite
        let out_candle = output.to_candle().unwrap();
        let out_vec = out_candle.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, v) in out_vec.iter().enumerate() {
            assert!(v.is_finite(), "tensor output[{i}] not finite: {v}");
        }
    }

    // ── Full model forward test ─────────────────────────────────────────

    #[test]
    fn test_dense_model_forward() {
        let dim = 4;
        let vocab = 8;
        let inter = 8;
        let num_heads = 2;

        let block = make_test_block(dim, inter, num_heads);

        // Simple embeddings: each token maps to a distinct vector
        let mut tok_emb = vec![0.0f32; vocab * dim];
        for t in 0..vocab {
            for d in 0..dim {
                tok_emb[t * dim + d] = ((t + d) as f32) * 0.1;
            }
        }

        let model = DenseModel {
            blocks: vec![block],
            final_norm_weight: vec![1.0; dim],
            norm_eps: EPS,
            lm_head: DenseLinear::new(
                (0..vocab * dim).map(|i| (i as f32) * 0.01).collect(),
                None,
                dim,
                vocab,
            ),
            tok_embeddings: tok_emb,
            hidden_size: dim,
            vocab_size: vocab,
        };

        let logits = model.forward(&[1, 3]);
        assert_eq!(logits.len(), vocab, "logits length should be vocab_size");
        for (i, v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_dense_model_deterministic() {
        let dim = 4;
        let vocab = 8;
        let inter = 8;
        let num_heads = 2;

        let block1 = make_test_block(dim, inter, num_heads);
        let block2 = make_test_block(dim, inter, num_heads);

        let tok_emb: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.05).collect();
        let lm_w: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.01).collect();

        let make_model = |blk| DenseModel {
            blocks: vec![blk],
            final_norm_weight: vec![1.0; dim],
            norm_eps: EPS,
            lm_head: DenseLinear::new(lm_w.clone(), None, dim, vocab),
            tok_embeddings: tok_emb.clone(),
            hidden_size: dim,
            vocab_size: vocab,
        };

        let logits1 = make_model(block1).forward(&[0, 2, 5]);
        let logits2 = make_model(block2).forward(&[0, 2, 5]);
        assert_vec_approx(&logits1, &logits2, 1e-6, "deterministic forward");
    }
}
