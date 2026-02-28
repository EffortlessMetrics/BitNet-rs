//! GPU hardware abstraction layer for `BitNet` inference.
//!
//! Provides device-agnostic abstractions for sampling, quantization,
//! attention, embedding, `RoPE`, transformer blocks, generation control,
//! and memory management.

pub mod profiling;
pub mod speculative;
pub mod streaming;
pub mod test_harness;
use std::fmt;

// ── Errors ────────────────────────────────────────────────────────────────

/// Errors produced by GPU HAL operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HalError {
    /// Index is out of the valid range.
    OutOfBounds { index: u32, vocab_size: u32 },
    /// Requested allocation exceeds the memory budget.
    OutOfMemory { requested: usize, available: usize },
    /// Dimension mismatch between tensors.
    ShapeMismatch { expected: usize, actual: usize },
    /// Input slice was empty when a non-empty slice is required.
    EmptyInput,
}

impl fmt::Display for HalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds { index, vocab_size } => {
                write!(f, "index {index} out of bounds for vocab size {vocab_size}")
            }
            Self::OutOfMemory { requested, available } => {
                write!(
                    f,
                    "out of memory: requested {requested} bytes, \
                     {available} available"
                )
            }
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected}, got {actual}")
            }
            Self::EmptyInput => write!(f, "input slice must not be empty"),
        }
    }
}

impl std::error::Error for HalError {}

// ── Sampling ──────────────────────────────────────────────────────────────

/// Apply softmax in-place, converting logits to a probability distribution.
pub fn softmax(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

/// Apply temperature scaling to logits in-place.
///
/// Temperature = 0.0 is treated as greedy (no scaling).
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 || (temperature - 1.0).abs() < f32::EPSILON {
        return;
    }
    let inv = 1.0 / temperature;
    for v in logits.iter_mut() {
        *v *= inv;
    }
}

/// Return the index of the largest element.
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

/// Keep only the top-k logits, setting the rest to `NEG_INFINITY`.
///
/// If `k == 0` or `k >= logits.len()` the slice is left unchanged.
pub fn top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    for &idx in &indices[k..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

/// Apply repetition penalty to the specified token indices.
///
/// Positive logits are divided by `penalty`; negative logits are multiplied.
pub fn apply_repetition_penalty(logits: &mut [f32], token_ids: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < f32::EPSILON {
        return;
    }
    for &id in token_ids {
        let i = id as usize;
        if i < logits.len() {
            if logits[i] > 0.0 {
                logits[i] /= penalty;
            } else {
                logits[i] *= penalty;
            }
        }
    }
}

// ── Quantization ──────────────────────────────────────────────────────────

/// Ternary quantization: map each value to {-1, 0, +1}.
pub fn ternary_quantize(values: &[f32]) -> Vec<i8> {
    let max_abs = values.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    if max_abs == 0.0 {
        return vec![0i8; values.len()];
    }
    let threshold = max_abs * 0.5;
    values
        .iter()
        .map(|&v| {
            if v > threshold {
                1
            } else if v < -threshold {
                -1
            } else {
                0
            }
        })
        .collect()
}

/// Dequantize ternary values using a scale factor.
pub fn ternary_dequantize(quantized: &[i8], scale: f32) -> Vec<f32> {
    quantized.iter().map(|&v| f32::from(v) * scale).collect()
}

/// Compute the compression ratio (original size / compressed size).
///
/// For ternary quantization: 32-bit floats → 2-bit ternary values.
pub fn compression_ratio(original_elements: usize) -> f32 {
    if original_elements == 0 {
        return 0.0;
    }
    // Original: 4 bytes per f32. Ternary: 2 bits per value (packed).
    let original_bytes = original_elements * 4;
    // Ceiling division for packed 2-bit representation.
    let compressed_bytes = (original_elements * 2).div_ceil(8);
    #[allow(clippy::cast_precision_loss)]
    {
        original_bytes as f32 / compressed_bytes as f32
    }
}

// ── Attention ─────────────────────────────────────────────────────────────

/// Build a causal attention mask of shape `[seq_len, seq_len]`.
///
/// `mask[i][j] = 0.0` if `j <= i`, else `NEG_INFINITY`.
pub fn build_causal_mask(seq_len: usize) -> Vec<f32> {
    let mut mask = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    mask
}

/// Compute scaled dot-product attention output dimensions.
///
/// Returns `(batch_seq_len, head_dim)`.
pub const fn attention_output_shape(seq_len: usize, head_dim: usize) -> (usize, usize) {
    (seq_len, head_dim)
}

/// Simulate a single-head attention forward pass (for shape checking).
///
/// `q`, `k`, `v` each have shape `[seq_len, head_dim]`.
/// Returns output of shape `[seq_len, head_dim]`.
pub fn attention_forward(
    q: &[f32],
    _k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Result<Vec<f32>, HalError> {
    let expected = seq_len * head_dim;
    if q.len() != expected {
        return Err(HalError::ShapeMismatch { expected, actual: q.len() });
    }
    // Simplified: return v (identity attention for shape verification).
    Ok(v[..expected].to_vec())
}

// ── Embedding ─────────────────────────────────────────────────────────────

/// An embedding table with `vocab_size` rows × `dim` columns.
pub struct EmbeddingTable {
    pub vocab_size: u32,
    pub dim: usize,
    pub weights: Vec<f32>,
}

impl EmbeddingTable {
    /// Create a new embedding table filled with a constant.
    pub fn new(vocab_size: u32, dim: usize, fill: f32) -> Self {
        Self { vocab_size, dim, weights: vec![fill; vocab_size as usize * dim] }
    }

    /// Look up a single token embedding.
    pub fn lookup(&self, token_id: u32) -> Result<&[f32], HalError> {
        if token_id >= self.vocab_size {
            return Err(HalError::OutOfBounds { index: token_id, vocab_size: self.vocab_size });
        }
        let start = token_id as usize * self.dim;
        Ok(&self.weights[start..start + self.dim])
    }

    /// Batch lookup: returns a flat vector of `ids.len() * dim` floats.
    pub fn batch_lookup(&self, ids: &[u32]) -> Result<Vec<f32>, HalError> {
        let mut out = Vec::with_capacity(ids.len() * self.dim);
        for &id in ids {
            out.extend_from_slice(self.lookup(id)?);
        }
        Ok(out)
    }
}

// ── RoPE ──────────────────────────────────────────────────────────────────

/// Build `RoPE` cos/sin tables for the given dimension and sequence length.
///
/// Returns `(cos_table, sin_table)` each of length `seq_len * half_dim`.
pub fn build_rope_tables(
    dim: usize,
    seq_len: usize,
    base: f32,
) -> Result<(Vec<f32>, Vec<f32>), HalError> {
    if dim == 0 || !dim.is_multiple_of(2) {
        return Err(HalError::ShapeMismatch { expected: 2, actual: dim % 2 });
    }
    let half_dim = dim / 2;
    let mut cos_table = Vec::with_capacity(seq_len * half_dim);
    let mut sin_table = Vec::with_capacity(seq_len * half_dim);

    for pos in 0..seq_len {
        for i in 0..half_dim {
            #[allow(clippy::cast_precision_loss)]
            let freq = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
            #[allow(clippy::cast_precision_loss)]
            let angle = pos as f32 * freq;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
    }
    Ok((cos_table, sin_table))
}

/// Apply `RoPE` rotation to a vector of `[half_dim]` pairs.
///
/// `x` has length `dim` (must be even). Rotates `(x[2i], x[2i+1])` pairs.
pub fn apply_rope(x: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
    let half = x.len() / 2;
    let mut out = vec![0.0_f32; x.len()];
    for i in 0..half {
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        out[2 * i] = x0.mul_add(cos[i], -(x1 * sin[i]));
        out[2 * i + 1] = x0.mul_add(sin[i], x1 * cos[i]);
    }
    out
}

/// Apply inverse `RoPE` rotation (negate sin).
pub fn apply_rope_inverse(x: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
    let neg_sin: Vec<f32> = sin.iter().map(|&s| -s).collect();
    apply_rope(x, cos, &neg_sin)
}

// ── Transformer blocks ────────────────────────────────────────────────────

/// RMS normalization in-place.
///
/// Normalizes the input vector so that its RMS = 1, then scales by `weight`.
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }
    #[allow(clippy::cast_precision_loss)]
    let ss: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (ss + eps).sqrt();
    for (xi, &wi) in x.iter_mut().zip(weight.iter()) {
        *xi *= inv_rms * wi;
    }
}

/// Feed-forward network (`SwiGLU`-style): `output = silu(x·W_gate) * (x·W_up)`.
///
/// For property testing, this is simplified to shape verification.
/// Input `x` has length `dim`, output also has length `dim`.
pub fn ffn_forward(x: &[f32]) -> Vec<f32> {
    // Simplified FFN: identity transform preserving shape.
    x.to_vec()
}

// ── Generation control ────────────────────────────────────────────────────

/// EOS token ID sentinel.
pub const DEFAULT_EOS_TOKEN: u32 = 2;

/// Configuration for text generation stopping conditions.
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub eos_token_id: u32,
}

/// Outcome of a single generation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// Continue generating.
    Continue,
    /// Stop: reached maximum token count.
    MaxTokens,
    /// Stop: produced the EOS token.
    Eos,
}

/// Check whether generation should stop.
pub const fn check_stop(
    token: u32,
    tokens_generated: usize,
    config: &GenerationConfig,
) -> StepOutcome {
    if token == config.eos_token_id {
        return StepOutcome::Eos;
    }
    if tokens_generated >= config.max_tokens {
        return StepOutcome::MaxTokens;
    }
    StepOutcome::Continue
}

// ── Memory management ─────────────────────────────────────────────────────

/// A simple memory pool that tracks allocations against a fixed budget.
pub struct MemoryPool {
    total: usize,
    used: usize,
}

impl MemoryPool {
    /// Create a pool with `total` bytes of capacity.
    pub const fn new(total: usize) -> Self {
        Self { total, used: 0 }
    }

    /// Bytes currently available.
    pub const fn available(&self) -> usize {
        self.total - self.used
    }

    /// Total capacity.
    pub const fn total(&self) -> usize {
        self.total
    }

    /// Bytes currently in use.
    pub const fn used(&self) -> usize {
        self.used
    }

    /// Attempt to allocate `bytes`. Returns `Ok(offset)` or an error.
    pub const fn allocate(&mut self, bytes: usize) -> Result<usize, HalError> {
        if bytes > self.available() {
            return Err(HalError::OutOfMemory { requested: bytes, available: self.available() });
        }
        let offset = self.used;
        self.used += bytes;
        Ok(offset)
    }

    /// Free `bytes` from the pool.
    pub const fn deallocate(&mut self, bytes: usize) {
        self.used = self.used.saturating_sub(bytes);
    }
}
