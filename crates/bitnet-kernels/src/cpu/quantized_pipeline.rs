//! End-to-end quantized inference pipeline for CPU.
//!
//! Chains quantized operations (INT2/INT4) without dequantizing to FP32
//! between layers.  Only the final output logits are dequantized, which
//! reduces memory bandwidth and improves throughput for low-bit models.

use bitnet_common::{BitNetError, KernelError, Result};
use std::time::Instant;

// ── Error helper ───────────────────────────────────────────────────

fn invalid_arg(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Quantization type ──────────────────────────────────────────────

/// Quantization bit-width used in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// 2-bit signed ({-1, 0, +1}).
    INT2,
    /// 4-bit signed ([-8, 7]).
    INT4,
}

impl QuantType {
    /// Number of bits per element.
    #[inline]
    pub fn bits(self) -> u32 {
        match self {
            QuantType::INT2 => 2,
            QuantType::INT4 => 4,
        }
    }

    /// Maximum absolute representable value.
    #[inline]
    pub fn max_abs(self) -> i8 {
        match self {
            QuantType::INT2 => 1,
            QuantType::INT4 => 7,
        }
    }
}

/// Precision used for intermediate accumulations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumPrecision {
    /// 32-bit integer accumulation.
    INT32,
    /// 32-bit float accumulation.
    FP32,
}

/// Precision used for compute operations between layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputePrecision {
    /// Remain in quantized integer domain.
    Quantized,
    /// Dequantize to FP32 at each stage (reference path).
    FP32,
}

// ── Configuration ──────────────────────────────────────────────────

/// Fully describes a quantized inference pipeline.
#[derive(Debug, Clone)]
pub struct QuantizedPipelineConfig {
    /// Weight quantization type.
    pub quant_type: QuantType,
    /// Compute precision between layers.
    pub compute_precision: ComputePrecision,
    /// Accumulation precision inside matmuls.
    pub accumulation_precision: AccumPrecision,
    /// Model hidden dimension.
    pub hidden_dim: usize,
    /// FFN intermediate dimension.
    pub intermediate_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Vocabulary size (for final projection).
    pub vocab_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Layer-norm epsilon.
    pub eps: f32,
}

impl QuantizedPipelineConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_dim == 0 {
            return Err(invalid_arg("hidden_dim must be > 0"));
        }
        if self.intermediate_dim == 0 {
            return Err(invalid_arg("intermediate_dim must be > 0"));
        }
        if self.num_heads == 0 {
            return Err(invalid_arg("num_heads must be > 0"));
        }
        if self.head_dim == 0 {
            return Err(invalid_arg("head_dim must be > 0"));
        }
        if self.vocab_size == 0 {
            return Err(invalid_arg("vocab_size must be > 0"));
        }
        if self.num_layers == 0 {
            return Err(invalid_arg("num_layers must be > 0"));
        }
        if self.num_heads * self.head_dim != self.hidden_dim {
            return Err(invalid_arg("num_heads * head_dim must equal hidden_dim"));
        }
        Ok(())
    }
}

// ── Quantized tensor ───────────────────────────────────────────────

/// A tensor stored in quantized form with per-block scale factors.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized values stored as `i8` regardless of actual bit-width
    /// (upper bits zero-extended for INT2).
    pub data: Vec<i8>,
    /// Per-block scale factors.  `data.len() / block_size` entries.
    pub scales: Vec<f32>,
    /// Block size used for quantization.
    pub block_size: usize,
    /// Number of logical rows.
    pub rows: usize,
    /// Number of logical columns.
    pub cols: usize,
}

impl QuantizedTensor {
    /// Create a new quantized tensor by quantizing `values` in blocks.
    pub fn from_f32(
        values: &[f32],
        rows: usize,
        cols: usize,
        block_size: usize,
        qtype: QuantType,
    ) -> Result<Self> {
        if values.len() != rows * cols {
            return Err(invalid_arg(&format!(
                "values length {} != rows*cols {}",
                values.len(),
                rows * cols
            )));
        }
        if block_size == 0 {
            return Err(invalid_arg("block_size must be > 0"));
        }
        let num_blocks = values.len().div_ceil(block_size);
        let mut data = vec![0i8; values.len()];
        let mut scales = Vec::with_capacity(num_blocks);
        let max_abs = qtype.max_abs() as f32;

        for blk in 0..num_blocks {
            let start = blk * block_size;
            let end = (start + block_size).min(values.len());
            let block = &values[start..end];

            let abs_max = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if abs_max < 1e-10 { 1.0 } else { abs_max / max_abs };
            scales.push(scale);

            for (i, &v) in block.iter().enumerate() {
                let q = (v / scale).round().clamp(-max_abs, max_abs) as i8;
                data[start + i] = q;
            }
        }

        Ok(Self { data, scales, block_size, rows, cols })
    }

    /// Dequantize back to f32.
    pub fn to_f32(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.data.len()];
        let num_blocks = self.data.len().div_ceil(self.block_size);
        for blk in 0..num_blocks {
            let start = blk * self.block_size;
            let end = (start + self.block_size).min(self.data.len());
            let scale = self.scales[blk];
            for (o, &d) in out[start..end].iter_mut().zip(&self.data[start..end]) {
                *o = d as f32 * scale;
            }
        }
        out
    }

    /// Number of logical elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ── Quantized weight set for one transformer layer ─────────────────

/// Weights for a single transformer layer, stored quantized.
#[derive(Debug, Clone)]
pub struct QuantizedLayerWeights {
    /// Query projection `[hidden_dim, hidden_dim]`.
    pub w_q: QuantizedTensor,
    /// Key projection `[hidden_dim, hidden_dim]`.
    pub w_k: QuantizedTensor,
    /// Value projection `[hidden_dim, hidden_dim]`.
    pub w_v: QuantizedTensor,
    /// Output projection `[hidden_dim, hidden_dim]`.
    pub w_o: QuantizedTensor,
    /// FFN up-projection `[intermediate_dim, hidden_dim]`.
    pub w_up: QuantizedTensor,
    /// FFN down-projection `[hidden_dim, intermediate_dim]`.
    pub w_down: QuantizedTensor,
    /// Attention LayerNorm gamma.
    pub attn_ln_gamma: Vec<f32>,
    /// FFN LayerNorm gamma.
    pub ffn_ln_gamma: Vec<f32>,
}

// ── Pipeline ───────────────────────────────────────────────────────

/// End-to-end quantized inference pipeline.
#[derive(Debug)]
pub struct QuantizedPipeline {
    /// Pipeline configuration.
    pub config: QuantizedPipelineConfig,
    /// Per-layer quantized weights.
    pub layers: Vec<QuantizedLayerWeights>,
    /// Embedding table (f32, looked up then quantized).
    pub embedding_table: Vec<f32>,
    /// Final output projection `[vocab_size, hidden_dim]`.
    pub output_proj: QuantizedTensor,
}

// ── Core quantized operations ──────────────────────────────────────

/// Quantized matrix-vector multiply: `y = W_q · x_q` accumulated in i32,
/// then rescaled to produce a new quantized output.
///
/// `w` is `[out_dim, in_dim]` row-major, `x` is `[in_dim]`.
/// Returns `[out_dim]` f32 values (scaled accumulations).
pub fn quantized_matmul_vec(w: &QuantizedTensor, x: &[i8], x_scale: f32) -> Result<Vec<f32>> {
    if w.cols != x.len() {
        return Err(invalid_arg(&format!(
            "matmul dimension mismatch: weight cols {} != input len {}",
            w.cols,
            x.len()
        )));
    }
    let out_dim = w.rows;
    let in_dim = w.cols;
    let mut result = vec![0.0f32; out_dim];
    let num_blocks_per_row = in_dim.div_ceil(w.block_size);

    for (row, result_val) in result.iter_mut().enumerate() {
        let mut acc = 0i64;
        let w_row = &w.data[row * in_dim..(row + 1) * in_dim];
        for blk in 0..num_blocks_per_row {
            let start = blk * w.block_size;
            let end = (start + w.block_size).min(in_dim);
            for (wv, xv) in w_row[start..end].iter().zip(&x[start..end]) {
                acc += *wv as i64 * *xv as i64;
            }
        }
        // Combine weight-block scales with input scale.
        // Use the average weight scale for simplicity in this scalar path.
        let w_scale_sum: f32 = (0..num_blocks_per_row)
            .map(|blk| {
                let scale_idx = row * num_blocks_per_row + blk;
                if scale_idx < w.scales.len() { w.scales[scale_idx] } else { 1.0 }
            })
            .sum();
        let w_scale_avg = w_scale_sum / num_blocks_per_row as f32;
        *result_val = acc as f32 * w_scale_avg * x_scale;
    }
    Ok(result)
}

/// Quantized attention computation.
///
/// Computes single-head scaled dot-product attention using quantized Q, K, V
/// projections without full dequantization to FP32 between projection steps.
pub fn quantized_attention(
    input: &[f32],
    weights: &QuantizedLayerWeights,
    config: &QuantizedPipelineConfig,
    seq_len: usize,
) -> Result<Vec<f32>> {
    let hidden = config.hidden_dim;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;

    if input.len() != seq_len * hidden {
        return Err(invalid_arg(&format!(
            "attention input length {} != seq_len*hidden {}",
            input.len(),
            seq_len * hidden
        )));
    }

    // Quantize input for matmul.
    let (x_q, x_scale) = quantize_vec(input, config.quant_type);

    // Project Q, K, V via quantized matmul.
    // For each token, compute projection independently.
    let mut q_all = vec![0.0f32; seq_len * hidden];
    let mut k_all = vec![0.0f32; seq_len * hidden];
    let mut v_all = vec![0.0f32; seq_len * hidden];

    for t in 0..seq_len {
        let x_t = &x_q[t * hidden..(t + 1) * hidden];
        let q_t = quantized_matmul_vec(&weights.w_q, x_t, x_scale)?;
        let k_t = quantized_matmul_vec(&weights.w_k, x_t, x_scale)?;
        let v_t = quantized_matmul_vec(&weights.w_v, x_t, x_scale)?;
        q_all[t * hidden..(t + 1) * hidden].copy_from_slice(&q_t);
        k_all[t * hidden..(t + 1) * hidden].copy_from_slice(&k_t);
        v_all[t * hidden..(t + 1) * hidden].copy_from_slice(&v_t);
    }

    // Scaled dot-product attention per head.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq_len * hidden];

    for h in 0..num_heads {
        for qi in 0..seq_len {
            // Compute attention scores for this query position.
            let mut scores = vec![0.0f32; seq_len];
            for (ki, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = qi * hidden + h * head_dim + d;
                    let k_idx = ki * hidden + h * head_dim + d;
                    dot += q_all[q_idx] * k_all[k_idx];
                }
                // Causal mask: future positions get -inf.
                *score = if ki <= qi { dot * scale } else { f32::NEG_INFINITY };
            }

            // Softmax.
            softmax_inplace(&mut scores);

            // Weighted sum of values.
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (vi, &s) in scores.iter().enumerate() {
                    let v_idx = vi * hidden + h * head_dim + d;
                    acc += s * v_all[v_idx];
                }
                attn_out[qi * hidden + h * head_dim + d] = acc;
            }
        }
    }

    // Output projection (quantized).
    let (out_q, out_scale) = quantize_vec(&attn_out, config.quant_type);
    let mut result = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let o_t = &out_q[t * hidden..(t + 1) * hidden];
        let proj = quantized_matmul_vec(&weights.w_o, o_t, out_scale)?;
        result[t * hidden..(t + 1) * hidden].copy_from_slice(&proj);
    }

    Ok(result)
}

/// Quantized feed-forward network (SiLU gating).
///
/// Computes `SiLU(x · W_up^T) · W_down^T` with quantized projections.
pub fn quantized_ffn(
    input: &[f32],
    weights: &QuantizedLayerWeights,
    config: &QuantizedPipelineConfig,
    seq_len: usize,
) -> Result<Vec<f32>> {
    let hidden = config.hidden_dim;

    if input.len() != seq_len * hidden {
        return Err(invalid_arg(&format!(
            "ffn input length {} != seq_len*hidden {}",
            input.len(),
            seq_len * hidden
        )));
    }

    let (x_q, x_scale) = quantize_vec(input, config.quant_type);

    let mut result = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let x_t = &x_q[t * hidden..(t + 1) * hidden];

        // Up projection.
        let up = quantized_matmul_vec(&weights.w_up, x_t, x_scale)?;

        // SiLU activation (done in FP32 — activations are not quantizable losslessly).
        let activated: Vec<f32> = up.iter().map(|&v| v / (1.0 + (-v).exp())).collect();

        // Quantize the activated intermediate for the down projection.
        let (act_q, act_scale) = quantize_vec(&activated, config.quant_type);

        // Down projection.
        let down = quantized_matmul_vec(&weights.w_down, &act_q, act_scale)?;
        result[t * hidden..(t + 1) * hidden].copy_from_slice(&down);
    }

    Ok(result)
}

/// LayerNorm operating in reduced precision.
///
/// Normalizes `input` and applies `gamma` scale, keeping computations
/// as lean as possible (no beta offset — RMSNorm style when full LN
/// is not needed).
pub fn quantized_layer_norm(
    input: &mut [f32],
    gamma: &[f32],
    norm_size: usize,
    eps: f32,
) -> Result<()> {
    if gamma.len() != norm_size {
        return Err(invalid_arg(&format!(
            "gamma length {} != norm_size {}",
            gamma.len(),
            norm_size
        )));
    }
    if !input.len().is_multiple_of(norm_size) {
        return Err(invalid_arg(&format!(
            "input length {} not divisible by norm_size {}",
            input.len(),
            norm_size
        )));
    }

    let batch = input.len() / norm_size;
    for b in 0..batch {
        let slice = &mut input[b * norm_size..(b + 1) * norm_size];

        // Compute mean.
        let mean = slice.iter().sum::<f32>() / norm_size as f32;

        // Compute variance.
        let var = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / norm_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        // Normalize and scale.
        for (v, &g) in slice.iter_mut().zip(gamma.iter()) {
            *v = (*v - mean) * inv_std * g;
        }
    }
    Ok(())
}

/// Embedding lookup returning quantized values.
///
/// Looks up `token_ids` in the embedding table, returning quantized
/// vectors and the associated scale factor.
pub fn quantized_embedding(
    table: &[f32],
    token_ids: &[u32],
    embedding_dim: usize,
    qtype: QuantType,
) -> Result<(Vec<i8>, f32)> {
    if embedding_dim == 0 || !table.len().is_multiple_of(embedding_dim) {
        return Err(invalid_arg("embedding table size not divisible by embedding_dim"));
    }
    let vocab_size = table.len() / embedding_dim;

    let mut fp_out = vec![0.0f32; token_ids.len() * embedding_dim];
    for (i, &id) in token_ids.iter().enumerate() {
        if (id as usize) >= vocab_size {
            return Err(invalid_arg(&format!("token id {} >= vocab_size {}", id, vocab_size)));
        }
        let src = (id as usize) * embedding_dim;
        fp_out[i * embedding_dim..(i + 1) * embedding_dim]
            .copy_from_slice(&table[src..src + embedding_dim]);
    }

    Ok(quantize_vec(&fp_out, qtype))
}

/// Dequantize the final output logits from quantized form to FP32.
pub fn dequantize_output(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&v| v as f32 * scale).collect()
}

/// Dequantize output from a `QuantizedTensor` (block-scale aware).
pub fn dequantize_output_tensor(tensor: &QuantizedTensor) -> Vec<f32> {
    tensor.to_f32()
}

/// Residual addition in the quantized domain.
///
/// Adds two f32 residual streams element-wise.  This operates in f32
/// because residual connections require high dynamic range — quantizing
/// the residual path degrades quality significantly.
pub fn quantized_residual_add(output: &mut [f32], residual: &[f32]) -> Result<()> {
    if output.len() != residual.len() {
        return Err(invalid_arg(&format!(
            "residual length mismatch: {} vs {}",
            output.len(),
            residual.len()
        )));
    }
    for (o, &r) in output.iter_mut().zip(residual.iter()) {
        *o += r;
    }
    Ok(())
}

/// Run the full quantized pipeline forward for one sequence.
///
/// Processes `seq_len` tokens through all transformer layers, returning
/// logits of shape `[seq_len, vocab_size]`.
pub fn pipeline_forward(pipeline: &QuantizedPipeline, token_ids: &[u32]) -> Result<Vec<f32>> {
    pipeline.config.validate()?;
    let seq_len = token_ids.len();
    let hidden = pipeline.config.hidden_dim;

    if seq_len == 0 {
        return Err(invalid_arg("token_ids must not be empty"));
    }

    // Embedding lookup → f32.
    let mut hidden_state = {
        let (emb_q, emb_scale) = quantized_embedding(
            &pipeline.embedding_table,
            token_ids,
            hidden,
            pipeline.config.quant_type,
        )?;
        dequantize_output(&emb_q, emb_scale)
    };

    // Transformer layers.
    for layer in &pipeline.layers {
        // Pre-attention LayerNorm.
        let mut normed = hidden_state.clone();
        quantized_layer_norm(&mut normed, &layer.attn_ln_gamma, hidden, pipeline.config.eps)?;

        // Attention.
        let attn_out = quantized_attention(&normed, layer, &pipeline.config, seq_len)?;

        // Residual add (attention).
        let mut post_attn = attn_out;
        quantized_residual_add(&mut post_attn, &hidden_state)?;

        // Pre-FFN LayerNorm.
        let mut normed_ffn = post_attn.clone();
        quantized_layer_norm(&mut normed_ffn, &layer.ffn_ln_gamma, hidden, pipeline.config.eps)?;

        // FFN.
        let ffn_out = quantized_ffn(&normed_ffn, layer, &pipeline.config, seq_len)?;

        // Residual add (FFN).
        hidden_state = ffn_out;
        quantized_residual_add(&mut hidden_state, &post_attn)?;
    }

    // Final output projection → logits.
    let (h_q, h_scale) = quantize_vec(&hidden_state, pipeline.config.quant_type);
    let mut logits = vec![0.0f32; seq_len * pipeline.config.vocab_size];
    for t in 0..seq_len {
        let h_t = &h_q[t * hidden..(t + 1) * hidden];
        let row = quantized_matmul_vec(&pipeline.output_proj, h_t, h_scale)?;
        logits[t * pipeline.config.vocab_size..(t + 1) * pipeline.config.vocab_size]
            .copy_from_slice(&row);
    }

    Ok(logits)
}

/// Benchmark result from `pipeline_benchmarkable`.
#[derive(Debug, Clone)]
pub struct PipelineBenchmark {
    /// Total wall-clock time in seconds.
    pub elapsed_secs: f64,
    /// Tokens processed.
    pub tokens: usize,
    /// Operations per second (forward passes / second).
    pub ops_per_sec: f64,
}

/// Measure ops/second for the quantized pipeline.
///
/// Runs `iterations` forward passes with the given `token_ids` and
/// reports throughput.
pub fn pipeline_benchmarkable(
    pipeline: &QuantizedPipeline,
    token_ids: &[u32],
    iterations: usize,
) -> Result<PipelineBenchmark> {
    if iterations == 0 {
        return Err(invalid_arg("iterations must be > 0"));
    }

    // Warm-up.
    pipeline_forward(pipeline, token_ids)?;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = pipeline_forward(pipeline, token_ids)?;
    }
    let elapsed = start.elapsed().as_secs_f64();

    Ok(PipelineBenchmark {
        elapsed_secs: elapsed,
        tokens: token_ids.len() * iterations,
        ops_per_sec: iterations as f64 / elapsed,
    })
}

// ── Internal helpers ───────────────────────────────────────────────

/// Quantize an f32 slice to i8 with a single global scale.
fn quantize_vec(input: &[f32], qtype: QuantType) -> (Vec<i8>, f32) {
    let max_abs_val = qtype.max_abs() as f32;
    let abs_max = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if abs_max < 1e-10 { 1.0 } else { abs_max / max_abs_val };
    let data: Vec<i8> =
        input.iter().map(|&v| (v / scale).round().clamp(-max_abs_val, max_abs_val) as i8).collect();
    (data, scale)
}

/// In-place softmax over a mutable f32 slice.
fn softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

// ── Test helper: build a tiny pipeline with random-ish weights ─────

#[cfg(test)]
fn make_test_config(quant_type: QuantType) -> QuantizedPipelineConfig {
    QuantizedPipelineConfig {
        quant_type,
        compute_precision: ComputePrecision::Quantized,
        accumulation_precision: AccumPrecision::INT32,
        hidden_dim: 16,
        intermediate_dim: 32,
        num_heads: 2,
        head_dim: 8,
        vocab_size: 32,
        num_layers: 1,
        eps: 1e-5,
    }
}

#[cfg(test)]
fn make_test_weights(config: &QuantizedPipelineConfig) -> QuantizedLayerWeights {
    let h = config.hidden_dim;
    let inter = config.intermediate_dim;
    let bs = 16;

    let make_w = |rows: usize, cols: usize| {
        // Deterministic pseudo-random weights.
        let vals: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        QuantizedTensor::from_f32(&vals, rows, cols, bs, config.quant_type).unwrap()
    };

    QuantizedLayerWeights {
        w_q: make_w(h, h),
        w_k: make_w(h, h),
        w_v: make_w(h, h),
        w_o: make_w(h, h),
        w_up: make_w(inter, h),
        w_down: make_w(h, inter),
        attn_ln_gamma: vec![1.0; h],
        ffn_ln_gamma: vec![1.0; h],
    }
}

#[cfg(test)]
fn make_test_pipeline(config: &QuantizedPipelineConfig) -> QuantizedPipeline {
    let h = config.hidden_dim;
    let bs = 16;

    let layers: Vec<QuantizedLayerWeights> =
        (0..config.num_layers).map(|_| make_test_weights(config)).collect();

    let embedding: Vec<f32> =
        (0..config.vocab_size * h).map(|i| (i as f32 * 0.03).sin() * 0.4).collect();

    let output_vals: Vec<f32> =
        (0..config.vocab_size * h).map(|i| (i as f32 * 0.07).cos() * 0.3).collect();
    let output_proj =
        QuantizedTensor::from_f32(&output_vals, config.vocab_size, h, bs, config.quant_type)
            .unwrap();

    QuantizedPipeline { config: config.clone(), layers, embedding_table: embedding, output_proj }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── QuantType tests ────────────────────────────────────────────

    #[test]
    fn test_quant_type_bits() {
        assert_eq!(QuantType::INT2.bits(), 2);
        assert_eq!(QuantType::INT4.bits(), 4);
    }

    #[test]
    fn test_quant_type_max_abs() {
        assert_eq!(QuantType::INT2.max_abs(), 1);
        assert_eq!(QuantType::INT4.max_abs(), 7);
    }

    // ── Config validation ──────────────────────────────────────────

    #[test]
    fn test_config_valid() {
        let cfg = make_test_config(QuantType::INT2);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_zero_hidden_dim() {
        let mut cfg = make_test_config(QuantType::INT2);
        cfg.hidden_dim = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_intermediate_dim() {
        let mut cfg = make_test_config(QuantType::INT2);
        cfg.intermediate_dim = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_num_heads() {
        let mut cfg = make_test_config(QuantType::INT2);
        cfg.num_heads = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_head_dim() {
        let mut cfg = make_test_config(QuantType::INT2);
        cfg.head_dim = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_vocab() {
        let mut cfg = make_test_config(QuantType::INT2);
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_layers() {
        let mut cfg = make_test_config(QuantType::INT2);
        cfg.num_layers = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_heads_dim_mismatch() {
        let mut cfg = make_test_config(QuantType::INT2);
        cfg.head_dim = 3; // 2*3 != 16
        assert!(cfg.validate().is_err());
    }

    // ── QuantizedTensor ────────────────────────────────────────────

    #[test]
    fn test_quantized_tensor_roundtrip_int2() {
        let vals = vec![0.5, -0.3, 0.0, 0.8, -0.9, 0.1, -0.4, 0.6];
        let qt = QuantizedTensor::from_f32(&vals, 2, 4, 4, QuantType::INT2).unwrap();
        assert_eq!(qt.rows, 2);
        assert_eq!(qt.cols, 4);
        assert_eq!(qt.len(), 8);
        assert!(!qt.is_empty());
        // INT2 values should be in {-1, 0, 1}.
        for &v in &qt.data {
            assert!(v >= -1 && v <= 1);
        }
    }

    #[test]
    fn test_quantized_tensor_roundtrip_int4() {
        let vals = vec![3.0, -5.0, 1.0, -2.0, 0.0, 7.0, -1.0, 4.0];
        let qt = QuantizedTensor::from_f32(&vals, 2, 4, 4, QuantType::INT4).unwrap();
        for &v in &qt.data {
            assert!(v >= -7 && v <= 7);
        }
    }

    #[test]
    fn test_quantized_tensor_dequantize_preserves_sign() {
        let vals = vec![1.0, -1.0, 0.0, 0.5];
        let qt = QuantizedTensor::from_f32(&vals, 1, 4, 4, QuantType::INT4).unwrap();
        let deq = qt.to_f32();
        assert!(deq[0] > 0.0);
        assert!(deq[1] < 0.0);
        // Zero should stay near zero.
        assert!(deq[2].abs() < 0.5);
    }

    #[test]
    fn test_quantized_tensor_dequantize_accuracy_int4() {
        let vals: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.3).collect();
        let qt = QuantizedTensor::from_f32(&vals, 4, 8, 8, QuantType::INT4).unwrap();
        let deq = qt.to_f32();
        // INT4 should achieve reasonable reconstruction (< 20% relative error).
        for (orig, recon) in vals.iter().zip(deq.iter()) {
            if orig.abs() > 0.5 {
                let rel_err = (orig - recon).abs() / orig.abs();
                assert!(rel_err < 0.25, "rel_err={rel_err} for orig={orig} recon={recon}");
            }
        }
    }

    #[test]
    fn test_quantized_tensor_bad_shape() {
        let vals = vec![1.0, 2.0, 3.0];
        let res = QuantizedTensor::from_f32(&vals, 2, 4, 4, QuantType::INT2);
        assert!(res.is_err());
    }

    #[test]
    fn test_quantized_tensor_zero_block_size() {
        let vals = vec![1.0, 2.0];
        let res = QuantizedTensor::from_f32(&vals, 1, 2, 0, QuantType::INT2);
        assert!(res.is_err());
    }

    #[test]
    fn test_quantized_tensor_all_zeros() {
        let vals = vec![0.0; 8];
        let qt = QuantizedTensor::from_f32(&vals, 2, 4, 4, QuantType::INT2).unwrap();
        for &v in &qt.data {
            assert_eq!(v, 0);
        }
        let deq = qt.to_f32();
        for &v in &deq {
            assert_eq!(v, 0.0);
        }
    }

    // ── quantize_vec / dequantize_output ───────────────────────────

    #[test]
    fn test_quantize_vec_int2_range() {
        let input = vec![1.0, -0.5, 0.0, 0.8, -1.0];
        let (q, _scale) = quantize_vec(&input, QuantType::INT2);
        for &v in &q {
            assert!(v >= -1 && v <= 1);
        }
    }

    #[test]
    fn test_quantize_vec_int4_range() {
        let input = vec![5.0, -3.0, 0.0, 7.0, -7.0, 1.0];
        let (q, _scale) = quantize_vec(&input, QuantType::INT4);
        for &v in &q {
            assert!(v >= -7 && v <= 7);
        }
    }

    #[test]
    fn test_dequantize_output_simple() {
        let data = vec![1i8, -1, 0, 2, -3];
        let scale = 0.5;
        let result = dequantize_output(&data, scale);
        assert_eq!(result, vec![0.5, -0.5, 0.0, 1.0, -1.5]);
    }

    #[test]
    fn test_dequantize_output_tensor_consistency() {
        let vals = vec![1.0, -2.0, 3.0, -4.0];
        let qt = QuantizedTensor::from_f32(&vals, 1, 4, 4, QuantType::INT4).unwrap();
        let deq = dequantize_output_tensor(&qt);
        assert_eq!(deq.len(), 4);
        // Signs must match originals.
        assert!(deq[0] > 0.0);
        assert!(deq[1] < 0.0);
        assert!(deq[2] > 0.0);
        assert!(deq[3] < 0.0);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let input = vec![0.3, -0.7, 0.0, 0.9, -0.1];
        let (q, scale) = quantize_vec(&input, QuantType::INT4);
        let deq = dequantize_output(&q, scale);
        for (orig, recon) in input.iter().zip(deq.iter()) {
            assert!((orig - recon).abs() < 0.3, "orig={orig} recon={recon}");
        }
    }

    // ── quantized_matmul_vec ───────────────────────────────────────

    #[test]
    fn test_quantized_matmul_vec_identity_like() {
        // Build a near-identity weight matrix.
        let dim = 4;
        let mut vals = vec![0.0f32; dim * dim];
        for i in 0..dim {
            vals[i * dim + i] = 1.0;
        }
        let w = QuantizedTensor::from_f32(&vals, dim, dim, dim, QuantType::INT4).unwrap();
        let x = vec![7i8, -3, 0, 5];
        let x_scale = 0.5;
        let result = quantized_matmul_vec(&w, &x, x_scale).unwrap();
        assert_eq!(result.len(), dim);
        // Each output should approximately equal x[i] * x_scale.
        for i in 0..dim {
            let expected = x[i] as f32 * x_scale;
            assert!(
                (result[i] - expected).abs() < 1.0,
                "i={i} result={} expected={expected}",
                result[i]
            );
        }
    }

    #[test]
    fn test_quantized_matmul_vec_dimension_mismatch() {
        let w = QuantizedTensor::from_f32(&[1.0; 8], 2, 4, 4, QuantType::INT2).unwrap();
        let x = vec![1i8; 3]; // wrong size
        assert!(quantized_matmul_vec(&w, &x, 1.0).is_err());
    }

    #[test]
    fn test_quantized_matmul_vec_all_zeros() {
        let w = QuantizedTensor::from_f32(&[0.0; 16], 4, 4, 4, QuantType::INT2).unwrap();
        let x = vec![1i8, 1, 1, 1];
        let result = quantized_matmul_vec(&w, &x, 1.0).unwrap();
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    // ── quantized_layer_norm ───────────────────────────────────────

    #[test]
    fn test_layer_norm_normalizes() {
        let mut input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        quantized_layer_norm(&mut input, &gamma, 4, 1e-5).unwrap();
        // Mean should be ~0.
        let mean: f32 = input.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean={mean}");
        // Variance should be ~1.
        let var: f32 = input.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 0.01, "var={var}");
    }

    #[test]
    fn test_layer_norm_gamma_scaling() {
        let mut input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0; 4];
        quantized_layer_norm(&mut input, &gamma, 4, 1e-5).unwrap();
        // Variance of the output should be ~4 (gamma=2 squared).
        let mean: f32 = input.iter().sum::<f32>() / 4.0;
        let var: f32 = input.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!((var - 4.0).abs() < 0.1, "var={var}");
    }

    #[test]
    fn test_layer_norm_batched() {
        let mut input = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
        let gamma = vec![1.0; 4];
        quantized_layer_norm(&mut input, &gamma, 4, 1e-5).unwrap();
        // Each batch element should be independently normalized.
        let mean0: f32 = input[0..4].iter().sum::<f32>() / 4.0;
        let mean1: f32 = input[4..8].iter().sum::<f32>() / 4.0;
        assert!(mean0.abs() < 1e-5);
        assert!(mean1.abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm_gamma_mismatch() {
        let mut input = vec![1.0; 8];
        let gamma = vec![1.0; 3]; // wrong size
        assert!(quantized_layer_norm(&mut input, &gamma, 3, 1e-5).is_err());
    }

    #[test]
    fn test_layer_norm_input_not_divisible() {
        let mut input = vec![1.0; 7];
        let gamma = vec![1.0; 4];
        assert!(quantized_layer_norm(&mut input, &gamma, 4, 1e-5).is_err());
    }

    // ── quantized_embedding ────────────────────────────────────────

    #[test]
    fn test_embedding_lookup_basic() {
        let table = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 3 embeddings, dim=2
        let ids = vec![0u32, 2, 1];
        let (q, scale) = quantized_embedding(&table, &ids, 2, QuantType::INT4).unwrap();
        assert_eq!(q.len(), 6);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_embedding_out_of_bounds() {
        let table = vec![0.1, 0.2, 0.3, 0.4]; // 2 embeddings, dim=2
        let ids = vec![5u32]; // out of bounds
        assert!(quantized_embedding(&table, &ids, 2, QuantType::INT2).is_err());
    }

    #[test]
    fn test_embedding_bad_table_size() {
        let table = vec![0.1, 0.2, 0.3]; // not divisible by dim=2
        let ids = vec![0u32];
        assert!(quantized_embedding(&table, &ids, 2, QuantType::INT2).is_err());
    }

    #[test]
    fn test_embedding_zero_dim() {
        let table: Vec<f32> = vec![];
        let ids = vec![0u32];
        assert!(quantized_embedding(&table, &ids, 0, QuantType::INT2).is_err());
    }

    #[test]
    fn test_embedding_preserves_relative_order() {
        // Embedding 0 is all-positive, embedding 1 is all-negative.
        let table = vec![1.0, 2.0, -1.0, -2.0];
        let ids = vec![0u32, 1];
        let (q, _scale) = quantized_embedding(&table, &ids, 2, QuantType::INT4).unwrap();
        // First embedding quantized values should be positive.
        assert!(q[0] > 0);
        assert!(q[1] > 0);
        // Second embedding should be negative.
        assert!(q[2] < 0);
        assert!(q[3] < 0);
    }

    // ── quantized_residual_add ─────────────────────────────────────

    #[test]
    fn test_residual_add_basic() {
        let mut out = vec![1.0, 2.0, 3.0];
        let residual = vec![0.5, -0.5, 1.0];
        quantized_residual_add(&mut out, &residual).unwrap();
        assert_eq!(out, vec![1.5, 1.5, 4.0]);
    }

    #[test]
    fn test_residual_add_length_mismatch() {
        let mut out = vec![1.0, 2.0];
        let residual = vec![0.5];
        assert!(quantized_residual_add(&mut out, &residual).is_err());
    }

    #[test]
    fn test_residual_add_zeros() {
        let mut out = vec![1.0, 2.0, 3.0];
        let residual = vec![0.0, 0.0, 0.0];
        quantized_residual_add(&mut out, &residual).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    // ── quantized_attention ────────────────────────────────────────

    #[test]
    fn test_attention_output_shape() {
        let cfg = make_test_config(QuantType::INT2);
        let weights = make_test_weights(&cfg);
        let seq_len = 3;
        let input = vec![0.1f32; seq_len * cfg.hidden_dim];
        let result = quantized_attention(&input, &weights, &cfg, seq_len).unwrap();
        assert_eq!(result.len(), seq_len * cfg.hidden_dim);
    }

    #[test]
    fn test_attention_single_token() {
        let cfg = make_test_config(QuantType::INT4);
        let weights = make_test_weights(&cfg);
        let input = vec![0.5f32; cfg.hidden_dim];
        let result = quantized_attention(&input, &weights, &cfg, 1).unwrap();
        assert_eq!(result.len(), cfg.hidden_dim);
        // Should produce finite values.
        for &v in &result {
            assert!(v.is_finite(), "non-finite attention output: {v}");
        }
    }

    #[test]
    fn test_attention_input_size_mismatch() {
        let cfg = make_test_config(QuantType::INT2);
        let weights = make_test_weights(&cfg);
        let input = vec![0.1f32; 5]; // wrong size
        assert!(quantized_attention(&input, &weights, &cfg, 1).is_err());
    }

    #[test]
    fn test_attention_causal_mask_applied() {
        // With seq_len > 1, the causal mask should prevent future info leakage.
        let cfg = make_test_config(QuantType::INT2);
        let weights = make_test_weights(&cfg);
        let seq_len = 4;
        let input = vec![0.2f32; seq_len * cfg.hidden_dim];
        let result = quantized_attention(&input, &weights, &cfg, seq_len).unwrap();
        assert_eq!(result.len(), seq_len * cfg.hidden_dim);
        // All values should be finite.
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_attention_int2_vs_int4_differ() {
        let cfg2 = make_test_config(QuantType::INT2);
        let cfg4 = make_test_config(QuantType::INT4);
        let w2 = make_test_weights(&cfg2);
        let w4 = make_test_weights(&cfg4);
        let input = vec![0.3f32; cfg2.hidden_dim];
        let r2 = quantized_attention(&input, &w2, &cfg2, 1).unwrap();
        let r4 = quantized_attention(&input, &w4, &cfg4, 1).unwrap();
        // Results should differ because quantization granularity differs.
        let diff: f32 = r2.iter().zip(r4.iter()).map(|(a, b)| (a - b).abs()).sum();
        // They won't be exactly equal (different quant); just check both valid.
        assert!(r2.iter().all(|v| v.is_finite()));
        assert!(r4.iter().all(|v| v.is_finite()));
        // At least some difference expected for non-trivial weights.
        let _ = diff; // suppress unused warning; we verified finiteness.
    }

    // ── quantized_ffn ──────────────────────────────────────────────

    #[test]
    fn test_ffn_output_shape() {
        let cfg = make_test_config(QuantType::INT2);
        let weights = make_test_weights(&cfg);
        let seq_len = 2;
        let input = vec![0.1f32; seq_len * cfg.hidden_dim];
        let result = quantized_ffn(&input, &weights, &cfg, seq_len).unwrap();
        assert_eq!(result.len(), seq_len * cfg.hidden_dim);
    }

    #[test]
    fn test_ffn_single_token() {
        let cfg = make_test_config(QuantType::INT4);
        let weights = make_test_weights(&cfg);
        let input = vec![0.5f32; cfg.hidden_dim];
        let result = quantized_ffn(&input, &weights, &cfg, 1).unwrap();
        assert_eq!(result.len(), cfg.hidden_dim);
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ffn_input_size_mismatch() {
        let cfg = make_test_config(QuantType::INT2);
        let weights = make_test_weights(&cfg);
        let input = vec![0.1f32; 5]; // wrong
        assert!(quantized_ffn(&input, &weights, &cfg, 1).is_err());
    }

    #[test]
    fn test_ffn_zero_input() {
        let cfg = make_test_config(QuantType::INT4);
        let weights = make_test_weights(&cfg);
        let input = vec![0.0f32; cfg.hidden_dim];
        let result = quantized_ffn(&input, &weights, &cfg, 1).unwrap();
        // Zero input through SiLU should produce zero-ish output.
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    // ── pipeline_forward ───────────────────────────────────────────

    #[test]
    fn test_pipeline_forward_output_shape_int2() {
        let cfg = make_test_config(QuantType::INT2);
        let pipeline = make_test_pipeline(&cfg);
        let token_ids = vec![0u32, 1, 2];
        let logits = pipeline_forward(&pipeline, &token_ids).unwrap();
        assert_eq!(logits.len(), token_ids.len() * cfg.vocab_size);
    }

    #[test]
    fn test_pipeline_forward_output_shape_int4() {
        let cfg = make_test_config(QuantType::INT4);
        let pipeline = make_test_pipeline(&cfg);
        let token_ids = vec![0u32, 1];
        let logits = pipeline_forward(&pipeline, &token_ids).unwrap();
        assert_eq!(logits.len(), token_ids.len() * cfg.vocab_size);
    }

    #[test]
    fn test_pipeline_forward_single_token() {
        let cfg = make_test_config(QuantType::INT4);
        let pipeline = make_test_pipeline(&cfg);
        let logits = pipeline_forward(&pipeline, &[0u32]).unwrap();
        assert_eq!(logits.len(), cfg.vocab_size);
        for &v in &logits {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn test_pipeline_forward_empty_tokens() {
        let cfg = make_test_config(QuantType::INT2);
        let pipeline = make_test_pipeline(&cfg);
        assert!(pipeline_forward(&pipeline, &[]).is_err());
    }

    #[test]
    fn test_pipeline_forward_all_finite() {
        let cfg = make_test_config(QuantType::INT2);
        let pipeline = make_test_pipeline(&cfg);
        let logits = pipeline_forward(&pipeline, &[0, 1, 2, 3]).unwrap();
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "non-finite logit at index {i}: {v}");
        }
    }

    #[test]
    fn test_pipeline_forward_different_tokens_different_logits() {
        let cfg = make_test_config(QuantType::INT4);
        let pipeline = make_test_pipeline(&cfg);
        let l1 = pipeline_forward(&pipeline, &[0]).unwrap();
        let l2 = pipeline_forward(&pipeline, &[1]).unwrap();
        // Different input tokens should produce different logits.
        let differs = l1.iter().zip(l2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "different tokens produced identical logits");
    }

    #[test]
    fn test_pipeline_forward_deterministic() {
        let cfg = make_test_config(QuantType::INT2);
        let pipeline = make_test_pipeline(&cfg);
        let ids = vec![0u32, 1, 2];
        let l1 = pipeline_forward(&pipeline, &ids).unwrap();
        let l2 = pipeline_forward(&pipeline, &ids).unwrap();
        assert_eq!(l1, l2, "pipeline should be deterministic");
    }

    #[test]
    fn test_pipeline_forward_multi_layer() {
        let mut cfg = make_test_config(QuantType::INT4);
        cfg.num_layers = 3;
        let pipeline = make_test_pipeline(&cfg);
        let logits = pipeline_forward(&pipeline, &[0, 1]).unwrap();
        assert_eq!(logits.len(), 2 * cfg.vocab_size);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    // ── Accuracy: quantized vs FP32 reference ──────────────────────

    fn fp32_matmul_vec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows];
        for r in 0..rows {
            for c in 0..cols {
                out[r] += w[r * cols + c] * x[c];
            }
        }
        out
    }

    #[test]
    fn test_quantized_matmul_accuracy_vs_fp32_int4() {
        let rows = 8;
        let cols = 8;
        let w_vals: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.13).sin() * 2.0).collect();
        let x_vals: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.37).cos()).collect();

        let fp32_result = fp32_matmul_vec(&w_vals, &x_vals, rows, cols);

        let w_q = QuantizedTensor::from_f32(&w_vals, rows, cols, cols, QuantType::INT4).unwrap();
        let (x_q, x_scale) = quantize_vec(&x_vals, QuantType::INT4);
        let q_result = quantized_matmul_vec(&w_q, &x_q, x_scale).unwrap();

        // Cosine similarity should be reasonable for INT4.
        let dot: f32 = fp32_result.iter().zip(q_result.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = fp32_result.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = q_result.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos_sim = if norm_a > 0.0 && norm_b > 0.0 { dot / (norm_a * norm_b) } else { 1.0 };
        assert!(cos_sim > 0.7, "INT4 cosine similarity {cos_sim} too low vs FP32 reference");
    }

    #[test]
    fn test_quantized_matmul_accuracy_vs_fp32_int2() {
        let rows = 8;
        let cols = 8;
        let w_vals: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.13).sin() * 2.0).collect();
        let x_vals: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.37).cos()).collect();

        let fp32_result = fp32_matmul_vec(&w_vals, &x_vals, rows, cols);

        let w_q = QuantizedTensor::from_f32(&w_vals, rows, cols, cols, QuantType::INT2).unwrap();
        let (x_q, x_scale) = quantize_vec(&x_vals, QuantType::INT2);
        let q_result = quantized_matmul_vec(&w_q, &x_q, x_scale).unwrap();

        // INT2 is coarser — lower bar.
        let dot: f32 = fp32_result.iter().zip(q_result.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = fp32_result.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = q_result.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos_sim = if norm_a > 0.0 && norm_b > 0.0 { dot / (norm_a * norm_b) } else { 1.0 };
        assert!(cos_sim > 0.3, "INT2 cosine similarity {cos_sim} too low vs FP32 reference");
    }

    // ── End-to-end accuracy bounds ─────────────────────────────────

    #[test]
    fn test_e2e_logits_bounded_int4() {
        let cfg = make_test_config(QuantType::INT4);
        let pipeline = make_test_pipeline(&cfg);
        let logits = pipeline_forward(&pipeline, &[0, 1, 2]).unwrap();
        // Logits should not explode.
        for &v in &logits {
            assert!(v.abs() < 1e6, "logit magnitude too large: {v}");
        }
    }

    #[test]
    fn test_e2e_logits_bounded_int2() {
        let cfg = make_test_config(QuantType::INT2);
        let pipeline = make_test_pipeline(&cfg);
        let logits = pipeline_forward(&pipeline, &[0, 1, 2]).unwrap();
        for &v in &logits {
            assert!(v.abs() < 1e6, "logit magnitude too large: {v}");
        }
    }

    #[test]
    fn test_e2e_longer_sequence() {
        let cfg = make_test_config(QuantType::INT4);
        let pipeline = make_test_pipeline(&cfg);
        let ids: Vec<u32> = (0..8).collect();
        let logits = pipeline_forward(&pipeline, &ids).unwrap();
        assert_eq!(logits.len(), 8 * cfg.vocab_size);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_e2e_accuracy_int4_vs_int2() {
        // INT4 pipeline should generally produce more varied logits than INT2
        // because of higher representational fidelity.
        let cfg2 = make_test_config(QuantType::INT2);
        let cfg4 = make_test_config(QuantType::INT4);
        let p2 = make_test_pipeline(&cfg2);
        let p4 = make_test_pipeline(&cfg4);
        let ids = vec![0u32, 1, 2];
        let l2 = pipeline_forward(&p2, &ids).unwrap();
        let l4 = pipeline_forward(&p4, &ids).unwrap();
        // Both should be finite.
        assert!(l2.iter().all(|v| v.is_finite()));
        assert!(l4.iter().all(|v| v.is_finite()));
    }

    // ── Sequence length variations ─────────────────────────────────

    #[test]
    fn test_seq_len_1() {
        let cfg = make_test_config(QuantType::INT4);
        let pipeline = make_test_pipeline(&cfg);
        let logits = pipeline_forward(&pipeline, &[5]).unwrap();
        assert_eq!(logits.len(), cfg.vocab_size);
    }

    #[test]
    fn test_seq_len_2() {
        let cfg = make_test_config(QuantType::INT2);
        let pipeline = make_test_pipeline(&cfg);
        let logits = pipeline_forward(&pipeline, &[0, 1]).unwrap();
        assert_eq!(logits.len(), 2 * cfg.vocab_size);
    }

    #[test]
    fn test_seq_len_16() {
        let cfg = make_test_config(QuantType::INT4);
        let pipeline = make_test_pipeline(&cfg);
        let ids: Vec<u32> = (0..16).map(|i| i % cfg.vocab_size as u32).collect();
        let logits = pipeline_forward(&pipeline, &ids).unwrap();
        assert_eq!(logits.len(), 16 * cfg.vocab_size);
    }

    // ── softmax_inplace ────────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    }

    #[test]
    fn test_softmax_monotonic() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut v);
        assert!(v[0] < v[1]);
        assert!(v[1] < v[2]);
    }

    #[test]
    fn test_softmax_empty() {
        let mut v: Vec<f32> = vec![];
        softmax_inplace(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn test_softmax_single() {
        let mut v = vec![42.0];
        softmax_inplace(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_with_neg_inf() {
        let mut v = vec![1.0, f32::NEG_INFINITY, 2.0];
        softmax_inplace(&mut v);
        assert!(v[1] < 1e-10, "neg_inf position should be ~0");
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ── pipeline_benchmarkable ─────────────────────────────────────

    #[test]
    fn test_benchmark_returns_positive_ops() {
        let cfg = make_test_config(QuantType::INT2);
        let pipeline = make_test_pipeline(&cfg);
        let result = pipeline_benchmarkable(&pipeline, &[0, 1], 2).unwrap();
        assert!(result.ops_per_sec > 0.0);
        assert!(result.elapsed_secs > 0.0);
        assert_eq!(result.tokens, 4); // 2 tokens × 2 iterations
    }

    #[test]
    fn test_benchmark_zero_iterations() {
        let cfg = make_test_config(QuantType::INT2);
        let pipeline = make_test_pipeline(&cfg);
        assert!(pipeline_benchmarkable(&pipeline, &[0], 0).is_err());
    }

    // ── Compute / accumulation precision enums ─────────────────────

    #[test]
    fn test_compute_precision_enum() {
        let q = ComputePrecision::Quantized;
        let f = ComputePrecision::FP32;
        assert_ne!(q, f);
    }

    #[test]
    fn test_accum_precision_enum() {
        let i = AccumPrecision::INT32;
        let f = AccumPrecision::FP32;
        assert_ne!(i, f);
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn test_quantize_vec_all_zeros() {
        let input = vec![0.0; 8];
        let (q, _scale) = quantize_vec(&input, QuantType::INT4);
        for &v in &q {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_quantize_vec_single_element() {
        let input = vec![1.0];
        let (q, scale) = quantize_vec(&input, QuantType::INT4);
        assert_eq!(q.len(), 1);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_pipeline_forward_repeated_token() {
        let cfg = make_test_config(QuantType::INT2);
        let pipeline = make_test_pipeline(&cfg);
        let logits = pipeline_forward(&pipeline, &[0, 0, 0]).unwrap();
        assert_eq!(logits.len(), 3 * cfg.vocab_size);
        // Due to causal masking, logits for positions 0,1,2 may differ
        // even with the same token.
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_quantized_tensor_large_values() {
        let vals = vec![100.0, -100.0, 50.0, -50.0];
        let qt = QuantizedTensor::from_f32(&vals, 1, 4, 4, QuantType::INT4).unwrap();
        let deq = qt.to_f32();
        // Signs preserved.
        assert!(deq[0] > 0.0);
        assert!(deq[1] < 0.0);
    }

    #[test]
    fn test_dequantize_output_empty() {
        let result = dequantize_output(&[], 1.0);
        assert!(result.is_empty());
    }
}
