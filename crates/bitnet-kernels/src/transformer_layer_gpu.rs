//! Complete GPU transformer layer orchestration.
//!
//! Wires all operations (RMSNorm, attention, FFN, residuals) into a single
//! transformer layer that can execute on OpenCL GPU or fall back to CPU
//! reference implementations.

use bitnet_common::{KernelError, Result};

/// Configuration for a single transformer layer.
#[derive(Debug, Clone)]
pub struct LayerConfig {
    /// Hidden dimension (model width).
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// FFN intermediate size (typically 4× hidden_size).
    pub intermediate_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

/// Lightweight KV cache for a single layer on GPU.
#[derive(Debug)]
pub struct GpuKvCache {
    /// Cached key states: shape [seq_len, num_heads * head_dim].
    pub keys: Vec<f32>,
    /// Cached value states: shape [seq_len, num_heads * head_dim].
    pub values: Vec<f32>,
    /// Number of cached positions.
    pub len: usize,
    /// Maximum sequence length.
    pub capacity: usize,
    /// Key/value dimension per entry.
    kv_dim: usize,
}

impl GpuKvCache {
    /// Create a new KV cache with the given capacity.
    pub fn new(capacity: usize, kv_dim: usize) -> Self {
        Self {
            keys: vec![0.0; capacity * kv_dim],
            values: vec![0.0; capacity * kv_dim],
            len: 0,
            capacity,
            kv_dim,
        }
    }

    /// Append a key-value pair at the current position.
    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        if self.len >= self.capacity {
            return Err(KernelError::ExecutionFailed { reason: "KV cache full".to_string() }.into());
        }
        let offset = self.len * self.kv_dim;
        self.keys[offset..offset + self.kv_dim].copy_from_slice(key);
        self.values[offset..offset + self.kv_dim].copy_from_slice(value);
        self.len += 1;
        Ok(())
    }

    /// Get all cached keys up to current length.
    pub fn cached_keys(&self) -> &[f32] {
        &self.keys[..self.len * self.kv_dim]
    }

    /// Get all cached values up to current length.
    pub fn cached_values(&self) -> &[f32] {
        &self.values[..self.len * self.kv_dim]
    }
}

/// Weights for a single transformer layer.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// Attention input RMSNorm weight [hidden_size].
    pub attn_norm: Vec<f32>,
    /// Q projection [hidden_size, num_heads * head_dim].
    pub wq: Vec<f32>,
    /// K projection [hidden_size, num_heads * head_dim].
    pub wk: Vec<f32>,
    /// V projection [hidden_size, num_heads * head_dim].
    pub wv: Vec<f32>,
    /// Output projection [num_heads * head_dim, hidden_size].
    pub wo: Vec<f32>,
    /// FFN input RMSNorm weight [hidden_size].
    pub ffn_norm: Vec<f32>,
    /// Gate projection [hidden_size, intermediate_size].
    pub w_gate: Vec<f32>,
    /// Up projection [hidden_size, intermediate_size].
    pub w_up: Vec<f32>,
    /// Down projection [intermediate_size, hidden_size].
    pub w_down: Vec<f32>,
}

/// A complete transformer layer executed on GPU (with CPU fallback).
///
/// Orchestrates:
/// 1. RMSNorm → QKV projection → RoPE → Attention scores → Softmax
///    → Weighted sum → Output projection → Residual add
/// 2. RMSNorm → Gate projection → Up projection → SiLU gate
///    → Down projection → Residual add
#[derive(Debug)]
pub struct GpuTransformerLayer {
    config: LayerConfig,
    weights: LayerWeights,
}

impl GpuTransformerLayer {
    /// Create a new GPU transformer layer with the given configuration and weights.
    pub fn new(config: LayerConfig, weights: LayerWeights) -> Result<Self> {
        let kv_dim = config.num_heads * config.head_dim;
        // Validate weight shapes
        if weights.attn_norm.len() != config.hidden_size {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "attn_norm length {} != hidden_size {}",
                    weights.attn_norm.len(),
                    config.hidden_size
                ),
            }
            .into());
        }
        if weights.wq.len() != config.hidden_size * kv_dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "wq length {} != hidden_size * kv_dim {}",
                    weights.wq.len(),
                    config.hidden_size * kv_dim
                ),
            }
            .into());
        }
        if weights.w_gate.len() != config.hidden_size * config.intermediate_size {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "w_gate length {} != hidden_size * intermediate_size {}",
                    weights.w_gate.len(),
                    config.hidden_size * config.intermediate_size
                ),
            }
            .into());
        }
        Ok(Self { config, weights })
    }

    /// Execute one transformer layer.
    ///
    /// Uses CPU reference implementations; GPU dispatch will be wired in when
    /// OpenCL kernel compilation is available at runtime.
    pub fn forward(
        &self,
        input: &[f32],
        kv_cache: &mut GpuKvCache,
        pos: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        if input.len() != hidden {
            return Err(KernelError::InvalidArguments {
                reason: format!("input length {} != hidden_size {}", input.len(), hidden),
            }
            .into());
        }
        let kv_dim = self.config.num_heads * self.config.head_dim;

        // === Attention sub-layer ===
        // 1. RMSNorm
        let normed = rms_norm(input, &self.weights.attn_norm, self.config.rms_norm_eps);

        // 2. QKV projections
        let q = matmul_mv(&self.weights.wq, &normed, hidden, kv_dim);
        let k = matmul_mv(&self.weights.wk, &normed, hidden, kv_dim);
        let v = matmul_mv(&self.weights.wv, &normed, hidden, kv_dim);

        // 3. Apply RoPE to Q and K
        let q = apply_rope(&q, pos, self.config.head_dim);
        let k = apply_rope(&k, pos, self.config.head_dim);

        // 4. Update KV cache
        kv_cache.append(&k, &v)?;

        // 5. Multi-head attention
        let attn_out = multi_head_attention(&q, kv_cache, &self.config);

        // 6. Output projection
        let attn_proj = matmul_mv(&self.weights.wo, &attn_out, kv_dim, hidden);

        // 7. Residual connection
        let mut hidden_state = vec![0.0f32; hidden];
        for i in 0..hidden {
            hidden_state[i] = input[i] + attn_proj[i];
        }

        // === FFN sub-layer ===
        // 8. RMSNorm
        let normed_ffn = rms_norm(&hidden_state, &self.weights.ffn_norm, self.config.rms_norm_eps);

        // 9. Gate + Up projections
        let gate =
            matmul_mv(&self.weights.w_gate, &normed_ffn, hidden, self.config.intermediate_size);
        let up = matmul_mv(&self.weights.w_up, &normed_ffn, hidden, self.config.intermediate_size);

        // 10. SiLU(gate) * up
        let mut ffn_hidden = vec![0.0f32; self.config.intermediate_size];
        for i in 0..self.config.intermediate_size {
            let silu_gate = gate[i] * sigmoid(gate[i]);
            ffn_hidden[i] = silu_gate * up[i];
        }

        // 11. Down projection
        let ffn_out =
            matmul_mv(&self.weights.w_down, &ffn_hidden, self.config.intermediate_size, hidden);

        // 12. Residual connection
        let mut output = vec![0.0f32; hidden];
        for i in 0..hidden {
            output[i] = hidden_state[i] + ffn_out[i];
        }

        Ok(output)
    }
}

// --- CPU reference helper functions ---

/// RMSNorm: out[i] = (x[i] / sqrt(mean(x²) + eps)) * weight[i]
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(&xi, &wi)| xi * inv_rms * wi).collect()
}

/// Matrix-vector multiply: y = W * x, where W is [in_dim, out_dim] row-major.
fn matmul_mv(w: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for j in 0..out_dim {
        let mut sum = 0.0f32;
        for i in 0..in_dim {
            sum += w[i * out_dim + j] * x[i];
        }
        out[j] = sum;
    }
    out
}

/// Sigmoid activation.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply Rotary Position Embedding (RoPE) to a vector.
///
/// Operates on pairs of elements within each head, applying rotation
/// based on position `pos`.
fn apply_rope(x: &[f32], pos: usize, head_dim: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    let num_heads = x.len() / head_dim;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / (10000.0f32).powf(i as f32 / head_dim as f32);
            let theta = pos as f32 * freq;
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let x0 = out[base + i];
            let x1 = out[base + i + 1];
            out[base + i] = x0 * cos_t - x1 * sin_t;
            out[base + i + 1] = x0 * sin_t + x1 * cos_t;
        }
    }
    out
}

/// Multi-head attention: Q @ K^T / sqrt(d) → softmax → @ V.
fn multi_head_attention(q: &[f32], kv_cache: &GpuKvCache, config: &LayerConfig) -> Vec<f32> {
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let seq_len = kv_cache.len;
    let kv_dim = num_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let keys = kv_cache.cached_keys();
    let values = kv_cache.cached_values();
    let mut output = vec![0.0f32; kv_dim];

    for h in 0..num_heads {
        let q_offset = h * head_dim;

        // Compute attention scores for this head
        let mut scores = vec![0.0f32; seq_len];
        for s in 0..seq_len {
            let k_offset = s * kv_dim + h * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_offset + d] * keys[k_offset + d];
            }
            scores[s] = dot * scale;
        }

        // Softmax
        softmax_inplace(&mut scores);

        // Weighted sum of values
        for s in 0..seq_len {
            let v_offset = s * kv_dim + h * head_dim;
            for d in 0..head_dim {
                output[q_offset + d] += scores[s] * values[v_offset + d];
            }
        }
    }

    output
}

/// In-place softmax over a slice.
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal config for testing (hidden=8, 2 heads, head_dim=4, ffn=16).
    fn test_config() -> LayerConfig {
        LayerConfig {
            hidden_size: 8,
            num_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            rms_norm_eps: 1e-5,
        }
    }

    /// Build identity-ish weights for the test config so we can trace data flow.
    fn test_weights(config: &LayerConfig) -> LayerWeights {
        let hidden = config.hidden_size;
        let kv_dim = config.num_heads * config.head_dim;
        let inter = config.intermediate_size;

        // Use small random-ish deterministic values
        let make_weight = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols).map(|i| (i as f32 * 0.1).sin() * 0.5).collect()
        };

        LayerWeights {
            attn_norm: vec![1.0; hidden],
            wq: make_weight(hidden, kv_dim),
            wk: make_weight(hidden, kv_dim),
            wv: make_weight(hidden, kv_dim),
            wo: make_weight(kv_dim, hidden),
            ffn_norm: vec![1.0; hidden],
            w_gate: make_weight(hidden, inter),
            w_up: make_weight(hidden, inter),
            w_down: make_weight(inter, hidden),
        }
    }

    #[test]
    fn test_rms_norm_unit() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let out = rms_norm(&x, &w, 1e-5);
        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5)
        let rms = (7.5f32).sqrt();
        for (i, &v) in out.iter().enumerate() {
            let expected = x[i] / rms;
            assert!((v - expected).abs() < 1e-5, "rms_norm[{i}]: got {v}, expected {expected}");
        }
    }

    #[test]
    fn test_softmax() {
        let mut scores = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut scores);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
        // Should be monotonically increasing
        assert!(scores[0] < scores[1]);
        assert!(scores[1] < scores[2]);
    }

    #[test]
    fn test_matmul_mv_identity() {
        // 2×2 identity matrix times [3, 7] = [3, 7]
        let w = vec![1.0, 0.0, 0.0, 1.0]; // row-major identity
        let x = vec![3.0, 7.0];
        let out = matmul_mv(&w, &x, 2, 2);
        assert!((out[0] - 3.0).abs() < 1e-5);
        assert!((out[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_rope_position_zero() {
        // At position 0, RoPE should be identity (cos(0)=1, sin(0)=0)
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = apply_rope(&x, 0, 4);
        for i in 0..4 {
            assert!(
                (out[i] - x[i]).abs() < 1e-5,
                "rope pos=0 [{i}]: got {}, expected {}",
                out[i],
                x[i]
            );
        }
    }

    #[test]
    fn test_kv_cache_append_and_retrieve() {
        let mut cache = GpuKvCache::new(4, 8);
        let k = vec![1.0; 8];
        let v = vec![2.0; 8];
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.len, 1);
        assert_eq!(cache.cached_keys().len(), 8);
        assert_eq!(cache.cached_values().len(), 8);
        assert!((cache.cached_keys()[0] - 1.0).abs() < 1e-5);
        assert!((cache.cached_values()[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_kv_cache_full() {
        let mut cache = GpuKvCache::new(1, 4);
        cache.append(&[1.0; 4], &[2.0; 4]).unwrap();
        let result = cache.append(&[3.0; 4], &[4.0; 4]);
        assert!(result.is_err(), "should fail when cache is full");
    }

    #[test]
    fn test_layer_forward_shape() {
        let config = test_config();
        let weights = test_weights(&config);
        let layer = GpuTransformerLayer::new(config.clone(), weights).unwrap();

        let input = vec![0.5; config.hidden_size];
        let kv_dim = config.num_heads * config.head_dim;
        let mut kv_cache = GpuKvCache::new(16, kv_dim);

        let output = layer.forward(&input, &mut kv_cache, 0).unwrap();
        assert_eq!(output.len(), config.hidden_size, "output should match hidden_size");
    }

    #[test]
    fn test_layer_forward_wrong_input_size() {
        let config = test_config();
        let weights = test_weights(&config);
        let layer = GpuTransformerLayer::new(config.clone(), weights).unwrap();

        let input = vec![0.5; config.hidden_size + 1]; // wrong size
        let kv_dim = config.num_heads * config.head_dim;
        let mut kv_cache = GpuKvCache::new(16, kv_dim);

        let result = layer.forward(&input, &mut kv_cache, 0);
        assert!(result.is_err(), "should reject wrong input size");
    }

    #[test]
    fn test_residual_connection_effect() {
        // Verify that the residual connection contributes to the output.
        // With all-zero weights, projections produce zero, so output == input.
        let config = test_config();
        let hidden = config.hidden_size;
        let kv_dim = config.num_heads * config.head_dim;
        let inter = config.intermediate_size;

        let weights = LayerWeights {
            attn_norm: vec![1.0; hidden],
            wq: vec![0.0; hidden * kv_dim],
            wk: vec![0.0; hidden * kv_dim],
            wv: vec![0.0; hidden * kv_dim],
            wo: vec![0.0; kv_dim * hidden],
            ffn_norm: vec![1.0; hidden],
            w_gate: vec![0.0; hidden * inter],
            w_up: vec![0.0; hidden * inter],
            w_down: vec![0.0; inter * hidden],
        };

        let layer = GpuTransformerLayer::new(config.clone(), weights).unwrap();
        let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut kv_cache = GpuKvCache::new(16, kv_dim);

        let output = layer.forward(&input, &mut kv_cache, 0).unwrap();

        // With zero projection weights, all projections yield zero,
        // so residual connections pass input through both sub-layers.
        for i in 0..hidden {
            assert!(
                (output[i] - input[i]).abs() < 1e-4,
                "residual pass-through [{i}]: got {}, expected {}",
                output[i],
                input[i]
            );
        }
    }

    #[test]
    fn test_multi_position_forward() {
        let config = test_config();
        let weights = test_weights(&config);
        let layer = GpuTransformerLayer::new(config.clone(), weights).unwrap();

        let kv_dim = config.num_heads * config.head_dim;
        let mut kv_cache = GpuKvCache::new(16, kv_dim);
        let input = vec![0.3; config.hidden_size];

        // Run multiple positions
        for pos in 0..4 {
            let output = layer.forward(&input, &mut kv_cache, pos).unwrap();
            assert_eq!(output.len(), config.hidden_size);
        }
        assert_eq!(kv_cache.len, 4, "KV cache should have 4 entries");
    }

    #[test]
    fn test_weight_shape_validation() {
        let config = test_config();
        let mut weights = test_weights(&config);
        weights.attn_norm = vec![1.0; config.hidden_size + 1]; // wrong size

        let result = GpuTransformerLayer::new(config, weights);
        assert!(result.is_err(), "should reject mismatched weight shapes");
    }
}
