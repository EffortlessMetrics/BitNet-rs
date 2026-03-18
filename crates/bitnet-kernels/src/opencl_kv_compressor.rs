//! KV cache compression for longer context windows.
//!
//! Provides CPU reference implementations and OpenCL kernel sources for
//! compressing the key-value cache used during autoregressive transformer
//! inference. Compression strategies include:
//!
//! - **Quantization** — INT8 / INT4 with per-head scales
//! - **Eviction** — LRU, FIFO, attention-score, or hybrid policies
//! - **Token merging** — merge similar KV pairs via cosine similarity
//! - **Sink tokens** — retain high-importance initial positions
//! - **Sliding window** — fixed-size window with preserved sink tokens
//!
//! The embedded OpenCL kernel source (`KV_COMPRESS_CL`) is a string constant
//! suitable for GPU dispatch; all logic has a matching CPU reference path.

#![allow(clippy::needless_range_loop)]

use std::fmt;

pub use bitnet_kv_cache_policy_core::{EvictionPolicy, KvEviction};

// ---------------------------------------------------------------------------
// OpenCL kernel source (embedded string — no runtime dependency)
// ---------------------------------------------------------------------------

/// OpenCL C kernel source for KV cache quantization.
///
/// Provides `kv_quantize_int8` and `kv_quantize_int4` kernels that quantize
/// float KV vectors to fixed-point with per-head scale factors.
pub const KV_COMPRESS_CL: &str = r#"
// KV cache quantization kernels for OpenCL
// Each work-item handles one (head, position, dim) element.

__kernel void kv_quantize_int8(
    __global const float* input,     // [num_heads, seq_len, head_dim]
    __global char*        output,    // [num_heads, seq_len, head_dim]
    __global float*       scales,    // [num_heads]
    const int seq_len,
    const int head_dim)
{
    int head = get_global_id(0);
    int total = seq_len * head_dim;
    int base  = head * total;

    // Pass 1: find absmax for this head
    float absmax = 0.0f;
    for (int i = 0; i < total; i++) {
        float v = fabs(input[base + i]);
        absmax = fmax(absmax, v);
    }
    float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
    scales[head] = scale;

    // Pass 2: quantize
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < total; i++) {
        float v = input[base + i] * inv_scale;
        v = fmax(-127.0f, fmin(127.0f, round(v)));
        output[base + i] = (char)v;
    }
}

__kernel void kv_quantize_int4(
    __global const float*  input,    // [num_heads, seq_len, head_dim]
    __global uchar*        output,   // packed: 2 values per byte
    __global float*        scales,   // [num_heads]
    const int seq_len,
    const int head_dim)
{
    int head  = get_global_id(0);
    int total = seq_len * head_dim;
    int base  = head * total;

    // Pass 1: absmax
    float absmax = 0.0f;
    for (int i = 0; i < total; i++) {
        float v = fabs(input[base + i]);
        absmax = fmax(absmax, v);
    }
    float scale = (absmax > 0.0f) ? (absmax / 7.0f) : 1.0f;
    scales[head] = scale;

    // Pass 2: quantize to 4-bit signed, pack pairs into bytes
    float inv_scale = 1.0f / scale;
    int packed_base = head * ((total + 1) / 2);
    for (int i = 0; i < total; i += 2) {
        float v0 = input[base + i] * inv_scale;
        v0 = fmax(-7.0f, fmin(7.0f, round(v0)));
        int q0 = ((int)v0) & 0x0F;

        int q1 = 0;
        if (i + 1 < total) {
            float v1 = input[base + i + 1] * inv_scale;
            v1 = fmax(-7.0f, fmin(7.0f, round(v1)));
            q1 = ((int)v1) & 0x0F;
        }
        output[packed_base + i / 2] = (uchar)(q0 | (q1 << 4));
    }
}
"#;

// ---------------------------------------------------------------------------
// QuantFormat — INT8 / INT4
// ---------------------------------------------------------------------------

/// Quantization bit-width for KV cache compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// 8-bit signed integer, per-head absmax scale.
    Int8,
    /// 4-bit signed integer, per-head absmax scale, 2 values per byte.
    Int4,
}

impl fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int8 => write!(f, "INT8"),
            Self::Int4 => write!(f, "INT4"),
        }
    }
}

// ---------------------------------------------------------------------------
// CompressionMethod
// ---------------------------------------------------------------------------

/// Top-level compression strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionMethod {
    /// Quantize KV values to reduced precision.
    Quantize(QuantFormat),
    /// Evict low-importance entries.
    Evict(EvictionPolicy),
    /// Merge similar KV pairs (token merging).
    Merge,
    /// Sliding window with sink-token preservation.
    SlidingWindow,
}

impl fmt::Display for CompressionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Quantize(qf) => write!(f, "Quantize({qf})"),
            Self::Evict(ep) => write!(f, "Evict({ep})"),
            Self::Merge => write!(f, "TokenMerge"),
            Self::SlidingWindow => write!(f, "SlidingWindow"),
        }
    }
}

// ---------------------------------------------------------------------------
// KvConfig
// ---------------------------------------------------------------------------

/// Configuration for KV cache compression.
#[derive(Debug, Clone)]
pub struct KvConfig {
    /// Maximum number of tokens retained in the compressed cache.
    pub max_cache_tokens: usize,
    /// Primary compression method.
    pub compression_method: CompressionMethod,
    /// Quality threshold in `[0.0, 1.0]`. Higher = less aggressive compression.
    pub quality_threshold: f32,
    /// Number of "sink" tokens to always retain (first N positions).
    pub num_sink_tokens: usize,
    /// Cosine-similarity threshold for token merging (only used with `Merge`).
    pub merge_similarity_threshold: f32,
}

impl KvConfig {
    /// Create a default config for sliding-window compression.
    pub fn sliding_window(max_tokens: usize, num_sink: usize) -> Self {
        Self {
            max_cache_tokens: max_tokens,
            compression_method: CompressionMethod::SlidingWindow,
            quality_threshold: 0.95,
            num_sink_tokens: num_sink,
            merge_similarity_threshold: 0.95,
        }
    }

    /// Create a config for INT8 quantization.
    pub fn int8() -> Self {
        Self {
            max_cache_tokens: usize::MAX,
            compression_method: CompressionMethod::Quantize(QuantFormat::Int8),
            quality_threshold: 0.99,
            num_sink_tokens: 0,
            merge_similarity_threshold: 0.95,
        }
    }

    /// Create a config for INT4 quantization.
    pub fn int4() -> Self {
        Self {
            max_cache_tokens: usize::MAX,
            compression_method: CompressionMethod::Quantize(QuantFormat::Int4),
            quality_threshold: 0.95,
            num_sink_tokens: 0,
            merge_similarity_threshold: 0.95,
        }
    }

    /// Create a config for attention-score-based eviction.
    pub fn eviction(max_tokens: usize, policy: EvictionPolicy) -> Self {
        Self {
            max_cache_tokens: max_tokens,
            compression_method: CompressionMethod::Evict(policy),
            quality_threshold: 0.90,
            num_sink_tokens: 0,
            merge_similarity_threshold: 0.95,
        }
    }

    /// Create a config for token-merge compression.
    pub fn merge(similarity_threshold: f32) -> Self {
        Self {
            max_cache_tokens: usize::MAX,
            compression_method: CompressionMethod::Merge,
            quality_threshold: 0.90,
            num_sink_tokens: 0,
            merge_similarity_threshold: similarity_threshold,
        }
    }
}

// ---------------------------------------------------------------------------
// KvCompressionStats
// ---------------------------------------------------------------------------

/// Statistics from a compression pass.
#[derive(Debug, Clone)]
pub struct KvCompressionStats {
    /// Number of KV entries before compression.
    pub original_entries: usize,
    /// Number of KV entries after compression.
    pub compressed_entries: usize,
    /// Original memory usage in bytes (estimated).
    pub original_bytes: usize,
    /// Compressed memory usage in bytes (estimated).
    pub compressed_bytes: usize,
    /// Estimated quality loss in `[0.0, 1.0]` (0 = lossless).
    pub quality_loss_estimate: f32,
    /// Compression method that was applied.
    pub method: CompressionMethod,
}

impl KvCompressionStats {
    /// Compression ratio: `original / compressed` (≥ 1.0 means smaller).
    pub fn compression_ratio(&self) -> f32 {
        if self.compressed_bytes == 0 {
            return f32::INFINITY;
        }
        self.original_bytes as f32 / self.compressed_bytes as f32
    }

    /// Bytes saved by compression.
    pub fn bytes_saved(&self) -> usize {
        self.original_bytes.saturating_sub(self.compressed_bytes)
    }
}

impl fmt::Display for KvCompressionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} → {} entries ({:.1}× ratio, {:.2}% quality loss, {} bytes saved)",
            self.method,
            self.original_entries,
            self.compressed_entries,
            self.compression_ratio(),
            self.quality_loss_estimate * 100.0,
            self.bytes_saved(),
        )
    }
}

// ---------------------------------------------------------------------------
// KvQuantizer — INT8 / INT4 quantization with per-head scales
// ---------------------------------------------------------------------------

/// Quantized KV cache data produced by [`KvQuantizer`].
#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    /// Quantized key data.
    pub keys: Vec<u8>,
    /// Quantized value data.
    pub values: Vec<u8>,
    /// Per-head scale factors for keys.
    pub key_scales: Vec<f32>,
    /// Per-head scale factors for values.
    pub value_scales: Vec<f32>,
    /// Quantization format used.
    pub format: QuantFormat,
    /// Number of heads.
    pub num_heads: usize,
    /// Sequence length per head.
    pub seq_len: usize,
    /// Dimensionality per head.
    pub head_dim: usize,
}

impl QuantizedKvCache {
    /// Total compressed size in bytes (keys + values + scales).
    pub fn total_bytes(&self) -> usize {
        self.keys.len()
            + self.values.len()
            + (self.key_scales.len() + self.value_scales.len()) * size_of::<f32>()
    }
}

/// Quantizes float KV cache entries to INT8 or INT4 with per-head absmax
/// scale factors. CPU reference implementation (mirrors the OpenCL kernels).
#[derive(Debug, Clone)]
pub struct KvQuantizer {
    /// Target quantization format.
    pub format: QuantFormat,
}

impl KvQuantizer {
    /// Create a new quantizer with the given format.
    pub fn new(format: QuantFormat) -> Self {
        Self { format }
    }

    /// Quantize a float tensor `[num_heads, seq_len, head_dim]` to the
    /// configured format. Returns per-head scale factors alongside the
    /// quantized bytes.
    pub fn quantize_tensor(
        &self,
        data: &[f32],
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        assert_eq!(data.len(), num_heads * seq_len * head_dim, "data length mismatch");
        match self.format {
            QuantFormat::Int8 => self.quantize_int8(data, num_heads, seq_len, head_dim),
            QuantFormat::Int4 => self.quantize_int4(data, num_heads, seq_len, head_dim),
        }
    }

    /// Dequantize back to float using the stored scales.
    pub fn dequantize_tensor(
        &self,
        quantized: &[u8],
        scales: &[f32],
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        match self.format {
            QuantFormat::Int8 => {
                self.dequantize_int8(quantized, scales, num_heads, seq_len, head_dim)
            }
            QuantFormat::Int4 => {
                self.dequantize_int4(quantized, scales, num_heads, seq_len, head_dim)
            }
        }
    }

    /// Quantize the full KV cache (keys + values).
    pub fn quantize_kv(
        &self,
        keys: &[f32],
        values: &[f32],
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> QuantizedKvCache {
        let (qk, ks) = self.quantize_tensor(keys, num_heads, seq_len, head_dim);
        let (qv, vs) = self.quantize_tensor(values, num_heads, seq_len, head_dim);
        QuantizedKvCache {
            keys: qk,
            values: qv,
            key_scales: ks,
            value_scales: vs,
            format: self.format,
            num_heads,
            seq_len,
            head_dim,
        }
    }

    // -- INT8 ---------------------------------------------------------------

    fn quantize_int8(
        &self,
        data: &[f32],
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        let total_per_head = seq_len * head_dim;
        let mut output = vec![0u8; data.len()];
        let mut scales = vec![0.0f32; num_heads];

        for h in 0..num_heads {
            let base = h * total_per_head;
            let head_data = &data[base..base + total_per_head];

            let absmax = head_data.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
            scales[h] = scale;

            let inv_scale = 1.0 / scale;
            for i in 0..total_per_head {
                let v = (head_data[i] * inv_scale).round().clamp(-127.0, 127.0) as i8;
                output[base + i] = v as u8;
            }
        }
        (output, scales)
    }

    fn dequantize_int8(
        &self,
        quantized: &[u8],
        scales: &[f32],
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let total_per_head = seq_len * head_dim;
        let mut output = vec![0.0f32; num_heads * total_per_head];

        for h in 0..num_heads {
            let base = h * total_per_head;
            let scale = scales[h];
            for i in 0..total_per_head {
                let q = quantized[base + i] as i8;
                output[base + i] = q as f32 * scale;
            }
        }
        output
    }

    // -- INT4 ---------------------------------------------------------------

    fn quantize_int4(
        &self,
        data: &[f32],
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        let total_per_head = seq_len * head_dim;
        let packed_per_head = total_per_head.div_ceil(2);
        let mut output = vec![0u8; num_heads * packed_per_head];
        let mut scales = vec![0.0f32; num_heads];

        for h in 0..num_heads {
            let base = h * total_per_head;
            let head_data = &data[base..base + total_per_head];

            let absmax = head_data.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            let scale = if absmax > 0.0 { absmax / 7.0 } else { 1.0 };
            scales[h] = scale;

            let inv_scale = 1.0 / scale;
            let packed_base = h * packed_per_head;
            for i in (0..total_per_head).step_by(2) {
                let v0 = (head_data[i] * inv_scale).round().clamp(-7.0, 7.0) as i8;
                let q0 = (v0 as u8) & 0x0F;

                let q1 = if i + 1 < total_per_head {
                    let v1 = (head_data[i + 1] * inv_scale).round().clamp(-7.0, 7.0) as i8;
                    (v1 as u8) & 0x0F
                } else {
                    0
                };
                output[packed_base + i / 2] = q0 | (q1 << 4);
            }
        }
        (output, scales)
    }

    fn dequantize_int4(
        &self,
        quantized: &[u8],
        scales: &[f32],
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let total_per_head = seq_len * head_dim;
        let packed_per_head = total_per_head.div_ceil(2);
        let mut output = vec![0.0f32; num_heads * total_per_head];

        for h in 0..num_heads {
            let scale = scales[h];
            let packed_base = h * packed_per_head;
            let out_base = h * total_per_head;

            for i in (0..total_per_head).step_by(2) {
                let byte = quantized[packed_base + i / 2];
                let q0 = sign_extend_4bit(byte & 0x0F);
                output[out_base + i] = q0 as f32 * scale;

                if i + 1 < total_per_head {
                    let q1 = sign_extend_4bit((byte >> 4) & 0x0F);
                    output[out_base + i + 1] = q1 as f32 * scale;
                }
            }
        }
        output
    }
}

/// Sign-extend a 4-bit signed value to `i8`.
#[inline]
fn sign_extend_4bit(nibble: u8) -> i8 {
    let val = nibble & 0x0F;
    if val & 0x08 != 0 {
        // Negative: extend sign
        (val | 0xF0) as i8
    } else {
        val as i8
    }
}

// ---------------------------------------------------------------------------
// KvMerger — merge similar KV pairs (token merging)
// ---------------------------------------------------------------------------

/// Merges similar KV pairs to reduce cache size.
///
/// For each pair of entries whose cosine similarity exceeds the threshold,
/// the two are replaced by their mean vector.
#[derive(Debug, Clone)]
pub struct KvMerger {
    /// Cosine-similarity threshold above which two entries are merged.
    pub threshold: f32,
}

impl KvMerger {
    pub fn new(threshold: f32) -> Self {
        Self { threshold: threshold.clamp(0.0, 1.0) }
    }

    /// Merge similar entries in a `[seq_len, dim]` matrix (row-major).
    ///
    /// Returns the merged matrix and the number of entries after merging.
    /// Pairs are scanned greedily; once an entry is merged it is not reused.
    pub fn merge(&self, data: &[f32], seq_len: usize, dim: usize) -> (Vec<f32>, usize) {
        assert_eq!(data.len(), seq_len * dim, "data length mismatch");
        if seq_len == 0 {
            return (Vec::new(), 0);
        }

        let mut merged: Vec<bool> = vec![false; seq_len];
        let mut output: Vec<f32> = Vec::with_capacity(seq_len * dim);
        let mut out_count = 0usize;

        for i in 0..seq_len {
            if merged[i] {
                continue;
            }
            let row_i = &data[i * dim..(i + 1) * dim];
            let mut best_j: Option<usize> = None;
            let mut best_sim = self.threshold;

            // Find best unmerged partner.
            for j in (i + 1)..seq_len {
                if merged[j] {
                    continue;
                }
                let row_j = &data[j * dim..(j + 1) * dim];
                let sim = cosine_similarity(row_i, row_j);
                if sim > best_sim {
                    best_sim = sim;
                    best_j = Some(j);
                }
            }

            if let Some(j) = best_j {
                // Merge: average the two rows.
                let row_j = &data[j * dim..(j + 1) * dim];
                for d in 0..dim {
                    output.push((row_i[d] + row_j[d]) * 0.5);
                }
                merged[j] = true;
            } else {
                // Keep as-is.
                output.extend_from_slice(row_i);
            }
            out_count += 1;
        }
        (output, out_count)
    }

    /// Merge both keys and values together (paired merge).
    /// Returns (merged_keys, merged_values, new_seq_len).
    pub fn merge_kv(
        &self,
        keys: &[f32],
        values: &[f32],
        seq_len: usize,
        dim: usize,
    ) -> (Vec<f32>, Vec<f32>, usize) {
        assert_eq!(keys.len(), seq_len * dim);
        assert_eq!(values.len(), seq_len * dim);
        if seq_len == 0 {
            return (Vec::new(), Vec::new(), 0);
        }

        let mut merged: Vec<bool> = vec![false; seq_len];
        let mut out_keys: Vec<f32> = Vec::with_capacity(seq_len * dim);
        let mut out_vals: Vec<f32> = Vec::with_capacity(seq_len * dim);
        let mut out_count = 0usize;

        for i in 0..seq_len {
            if merged[i] {
                continue;
            }
            let ki = &keys[i * dim..(i + 1) * dim];
            let vi = &values[i * dim..(i + 1) * dim];
            let mut best_j: Option<usize> = None;
            let mut best_sim = self.threshold;

            for j in (i + 1)..seq_len {
                if merged[j] {
                    continue;
                }
                let kj = &keys[j * dim..(j + 1) * dim];
                let sim = cosine_similarity(ki, kj);
                if sim > best_sim {
                    best_sim = sim;
                    best_j = Some(j);
                }
            }

            if let Some(j) = best_j {
                let kj = &keys[j * dim..(j + 1) * dim];
                let vj = &values[j * dim..(j + 1) * dim];
                for d in 0..dim {
                    out_keys.push((ki[d] + kj[d]) * 0.5);
                    out_vals.push((vi[d] + vj[d]) * 0.5);
                }
                merged[j] = true;
            } else {
                out_keys.extend_from_slice(ki);
                out_vals.extend_from_slice(vi);
            }
            out_count += 1;
        }
        (out_keys, out_vals, out_count)
    }
}

/// Cosine similarity between two equal-length slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

// ---------------------------------------------------------------------------
// SinkTokenManager
// ---------------------------------------------------------------------------

/// Manages "sink" tokens — the first N positions in a sequence that tend to
/// accumulate disproportionate attention and should be preserved across
/// compression passes.
#[derive(Debug, Clone)]
pub struct SinkTokenManager {
    /// Number of initial positions to treat as sinks.
    pub num_sinks: usize,
}

impl SinkTokenManager {
    pub fn new(num_sinks: usize) -> Self {
        Self { num_sinks }
    }

    /// Given a set of positions scheduled for eviction, remove any that fall
    /// within the sink range `[0, num_sinks)`.
    pub fn protect_sinks(&self, eviction_candidates: &[usize]) -> Vec<usize> {
        eviction_candidates.iter().copied().filter(|&pos| pos >= self.num_sinks).collect()
    }

    /// Whether a position is a sink token.
    #[inline]
    pub fn is_sink(&self, position: usize) -> bool {
        position < self.num_sinks
    }

    /// Extract sink-token rows from a `[seq_len, dim]` matrix.
    pub fn extract_sinks(&self, data: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
        let n = self.num_sinks.min(seq_len);
        data[..n * dim].to_vec()
    }
}

// ---------------------------------------------------------------------------
// WindowedKv — sliding window with sink preservation
// ---------------------------------------------------------------------------

/// Sliding-window KV cache that preserves sink tokens at the front.
///
/// Layout: `[sink_0 .. sink_{s-1}] [window_start .. window_end]`
///
/// When the total sequence length exceeds `window_size + num_sinks`, the
/// oldest non-sink entries are discarded.
#[derive(Debug, Clone)]
pub struct WindowedKv {
    /// Key cache: `[capacity, head_dim]` row-major.
    pub keys: Vec<f32>,
    /// Value cache: `[capacity, head_dim]` row-major.
    pub values: Vec<f32>,
    /// Dimension per head.
    pub head_dim: usize,
    /// Number of sink tokens (fixed prefix).
    pub num_sinks: usize,
    /// Maximum window size (excluding sinks).
    pub window_size: usize,
    /// Current number of stored entries (sinks + window).
    pub current_len: usize,
    /// Total capacity (`num_sinks + window_size`).
    capacity: usize,
}

impl WindowedKv {
    /// Create a new windowed KV cache.
    pub fn new(head_dim: usize, window_size: usize, num_sinks: usize) -> Self {
        let capacity = num_sinks + window_size;
        Self {
            keys: vec![0.0; capacity * head_dim],
            values: vec![0.0; capacity * head_dim],
            head_dim,
            num_sinks,
            window_size,
            current_len: 0,
            capacity,
        }
    }

    /// Total capacity of the cache (sinks + window).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Append new key/value pairs. If the window overflows, the oldest
    /// non-sink entries are dropped.
    pub fn append(&mut self, new_keys: &[f32], new_values: &[f32]) {
        let num_new = new_keys.len() / self.head_dim;
        assert_eq!(new_keys.len(), num_new * self.head_dim);
        assert_eq!(new_values.len(), num_new * self.head_dim);

        for t in 0..num_new {
            let k_row = &new_keys[t * self.head_dim..(t + 1) * self.head_dim];
            let v_row = &new_values[t * self.head_dim..(t + 1) * self.head_dim];
            self.push_single(k_row, v_row);
        }
    }

    fn push_single(&mut self, key: &[f32], value: &[f32]) {
        if self.current_len < self.capacity {
            // Still filling up.
            let offset = self.current_len * self.head_dim;
            self.keys[offset..offset + self.head_dim].copy_from_slice(key);
            self.values[offset..offset + self.head_dim].copy_from_slice(value);
            self.current_len += 1;
        } else {
            // Window is full — shift the window portion left by one.
            let window_start = self.num_sinks * self.head_dim;
            let window_bytes = (self.window_size - 1) * self.head_dim;
            self.keys.copy_within(
                window_start + self.head_dim..window_start + self.head_dim + window_bytes,
                window_start,
            );
            self.values.copy_within(
                window_start + self.head_dim..window_start + self.head_dim + window_bytes,
                window_start,
            );
            // Write new entry at the end.
            let last_offset = (self.capacity - 1) * self.head_dim;
            self.keys[last_offset..last_offset + self.head_dim].copy_from_slice(key);
            self.values[last_offset..last_offset + self.head_dim].copy_from_slice(value);
        }
    }

    /// Current keys slice `[current_len, head_dim]`.
    pub fn keys(&self) -> &[f32] {
        &self.keys[..self.current_len * self.head_dim]
    }

    /// Current values slice `[current_len, head_dim]`.
    pub fn values(&self) -> &[f32] {
        &self.values[..self.current_len * self.head_dim]
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.current_len = 0;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: max absolute error between two slices
    // -----------------------------------------------------------------------
    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    fn mean_sq_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum / a.len() as f32
    }

    // -----------------------------------------------------------------------
    // OpenCL kernel source
    // -----------------------------------------------------------------------

    #[test]
    fn kernel_source_is_nonempty() {
        assert!(!KV_COMPRESS_CL.is_empty());
        assert!(KV_COMPRESS_CL.contains("kv_quantize_int8"));
        assert!(KV_COMPRESS_CL.contains("kv_quantize_int4"));
    }

    #[test]
    fn kernel_source_contains_scale_computation() {
        assert!(KV_COMPRESS_CL.contains("absmax"));
        assert!(KV_COMPRESS_CL.contains("scales[head]"));
    }

    // -----------------------------------------------------------------------
    // QuantFormat display
    // -----------------------------------------------------------------------

    #[test]
    fn quant_format_display() {
        assert_eq!(format!("{}", QuantFormat::Int8), "INT8");
        assert_eq!(format!("{}", QuantFormat::Int4), "INT4");
    }

    // -----------------------------------------------------------------------
    // INT8 quantization roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn int8_roundtrip_small() {
        let quantizer = KvQuantizer::new(QuantFormat::Int8);
        let data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let (q, scales) = quantizer.quantize_tensor(&data, 1, 4, 4);
        let deq = quantizer.dequantize_tensor(&q, &scales, 1, 4, 4);
        let err = max_abs_error(&data, &deq);
        // INT8 rounding error should be ≤ scale/2 ≈ absmax/254
        assert!(err < 0.01, "INT8 roundtrip error too large: {err}");
    }

    #[test]
    fn int8_roundtrip_zeros() {
        let quantizer = KvQuantizer::new(QuantFormat::Int8);
        let data = [0.0f32; 32];
        let (q, scales) = quantizer.quantize_tensor(&data, 2, 4, 4);
        let deq = quantizer.dequantize_tensor(&q, &scales, 2, 4, 4);
        assert_eq!(deq, data);
    }

    #[test]
    fn int8_roundtrip_large_values() {
        let quantizer = KvQuantizer::new(QuantFormat::Int8);
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 10.0).collect();
        let (q, scales) = quantizer.quantize_tensor(&data, 2, 8, 4);
        let deq = quantizer.dequantize_tensor(&q, &scales, 2, 8, 4);
        let err = max_abs_error(&data, &deq);
        let max_val = data.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        // Relative error should be bounded.
        assert!(err / max_val < 0.01, "INT8 relative error too large: {}", err / max_val);
    }

    #[test]
    fn int8_per_head_scales() {
        let quantizer = KvQuantizer::new(QuantFormat::Int8);
        // Two heads with very different magnitudes.
        let mut data = vec![0.0f32; 2 * 4 * 4];
        for i in 0..16 {
            data[i] = i as f32 * 0.01; // head 0: small
        }
        for i in 0..16 {
            data[16 + i] = i as f32 * 100.0; // head 1: large
        }
        let (_, scales) = quantizer.quantize_tensor(&data, 2, 4, 4);
        assert_eq!(scales.len(), 2);
        assert!(scales[1] > scales[0] * 100.0, "scales should differ by magnitude");
    }

    #[test]
    fn int8_negative_values() {
        let quantizer = KvQuantizer::new(QuantFormat::Int8);
        let data: Vec<f32> = (0..16).map(|i| -(i as f32) * 0.5).collect();
        let (q, scales) = quantizer.quantize_tensor(&data, 1, 4, 4);
        let deq = quantizer.dequantize_tensor(&q, &scales, 1, 4, 4);
        let err = max_abs_error(&data, &deq);
        assert!(err < 0.1, "INT8 negative roundtrip error: {err}");
    }

    #[test]
    fn int8_single_element() {
        let quantizer = KvQuantizer::new(QuantFormat::Int8);
        let data = vec![42.0f32];
        let (q, scales) = quantizer.quantize_tensor(&data, 1, 1, 1);
        let deq = quantizer.dequantize_tensor(&q, &scales, 1, 1, 1);
        let err = max_abs_error(&data, &deq);
        assert!(err < 0.5, "single-element INT8 error: {err}");
    }

    #[test]
    fn int8_multihead_roundtrip() {
        let quantizer = KvQuantizer::new(QuantFormat::Int8);
        let data: Vec<f32> = (0..128).map(|i| (i as f32 * 0.3) - 20.0).collect();
        let (q, scales) = quantizer.quantize_tensor(&data, 4, 8, 4);
        assert_eq!(scales.len(), 4);
        let deq = quantizer.dequantize_tensor(&q, &scales, 4, 8, 4);
        let mse = mean_sq_error(&data, &deq);
        assert!(mse < 0.1, "INT8 multihead MSE too high: {mse}");
    }

    // -----------------------------------------------------------------------
    // INT4 quantization roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn int4_roundtrip_small() {
        let quantizer = KvQuantizer::new(QuantFormat::Int4);
        let data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let (q, scales) = quantizer.quantize_tensor(&data, 1, 4, 4);
        let deq = quantizer.dequantize_tensor(&q, &scales, 1, 4, 4);
        let err = max_abs_error(&data, &deq);
        // INT4 has coarser quantization, but small values should still be close.
        assert!(err < 0.2, "INT4 roundtrip error too large: {err}");
    }

    #[test]
    fn int4_roundtrip_zeros() {
        let quantizer = KvQuantizer::new(QuantFormat::Int4);
        let data = [0.0f32; 32];
        let (q, scales) = quantizer.quantize_tensor(&data, 2, 4, 4);
        let deq = quantizer.dequantize_tensor(&q, &scales, 2, 4, 4);
        assert_eq!(deq, data);
    }

    #[test]
    fn int4_per_head_scales() {
        let quantizer = KvQuantizer::new(QuantFormat::Int4);
        let mut data = vec![0.0f32; 2 * 4 * 4];
        for i in 0..16 {
            data[i] = i as f32 * 0.001;
        }
        for i in 0..16 {
            data[16 + i] = i as f32 * 50.0;
        }
        let (_, scales) = quantizer.quantize_tensor(&data, 2, 4, 4);
        assert_eq!(scales.len(), 2);
        assert!(scales[1] > scales[0] * 100.0);
    }

    #[test]
    fn int4_packed_byte_count() {
        let quantizer = KvQuantizer::new(QuantFormat::Int4);
        // 16 elements → 8 packed bytes per head, 2 heads
        let data = [1.0f32; 32];
        let (q, _) = quantizer.quantize_tensor(&data, 2, 4, 4);
        assert_eq!(q.len(), 2 * 8, "INT4 packed length mismatch");
    }

    #[test]
    fn int4_odd_element_count() {
        let quantizer = KvQuantizer::new(QuantFormat::Int4);
        // 3 elements per head → 2 packed bytes (last byte has padding)
        let data: Vec<f32> = vec![1.0, -1.0, 0.5];
        let (q, scales) = quantizer.quantize_tensor(&data, 1, 3, 1);
        assert_eq!(q.len(), 2); // ceil(3/2) = 2
        let deq = quantizer.dequantize_tensor(&q, &scales, 1, 3, 1);
        assert_eq!(deq.len(), 3);
    }

    #[test]
    fn int4_negative_values() {
        let quantizer = KvQuantizer::new(QuantFormat::Int4);
        let data: Vec<f32> = vec![-7.0, -3.0, 0.0, 3.0, 7.0, -1.0, 1.0, -5.0];
        let (q, scales) = quantizer.quantize_tensor(&data, 1, 8, 1);
        let deq = quantizer.dequantize_tensor(&q, &scales, 1, 8, 1);
        let err = max_abs_error(&data, &deq);
        assert!(err < 1.5, "INT4 negative roundtrip error: {err}");
    }

    #[test]
    fn int4_multihead() {
        let quantizer = KvQuantizer::new(QuantFormat::Int4);
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.5).collect();
        let (q, scales) = quantizer.quantize_tensor(&data, 4, 4, 4);
        assert_eq!(scales.len(), 4);
        let deq = quantizer.dequantize_tensor(&q, &scales, 4, 4, 4);
        assert_eq!(deq.len(), 64);
    }

    // -----------------------------------------------------------------------
    // KvQuantizer::quantize_kv
    // -----------------------------------------------------------------------

    #[test]
    fn quantize_kv_int8() {
        let quantizer = KvQuantizer::new(QuantFormat::Int8);
        let keys: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let values: Vec<f32> = (0..32).map(|i| -(i as f32) * 0.2).collect();
        let qkv = quantizer.quantize_kv(&keys, &values, 2, 4, 4);
        assert_eq!(qkv.key_scales.len(), 2);
        assert_eq!(qkv.value_scales.len(), 2);
        assert_eq!(qkv.format, QuantFormat::Int8);
        assert!(qkv.total_bytes() > 0);
    }

    #[test]
    fn quantize_kv_int4() {
        let quantizer = KvQuantizer::new(QuantFormat::Int4);
        let keys: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let values: Vec<f32> = (0..32).map(|i| -(i as f32) * 0.2).collect();
        let qkv = quantizer.quantize_kv(&keys, &values, 2, 4, 4);
        assert_eq!(qkv.format, QuantFormat::Int4);
        // INT4 should use fewer bytes than INT8.
        let q8 = KvQuantizer::new(QuantFormat::Int8);
        let qkv8 = q8.quantize_kv(&keys, &values, 2, 4, 4);
        assert!(qkv.keys.len() < qkv8.keys.len(), "INT4 should use fewer bytes than INT8");
    }

    // -----------------------------------------------------------------------
    // EvictionPolicy
    // -----------------------------------------------------------------------

    #[test]
    fn eviction_policy_display() {
        assert_eq!(format!("{}", EvictionPolicy::Lru), "LRU");
        assert_eq!(format!("{}", EvictionPolicy::Fifo), "FIFO");
        assert_eq!(format!("{}", EvictionPolicy::AttentionScore), "AttentionScore");
        let h = EvictionPolicy::hybrid(0.7);
        assert!(format!("{h}").contains("70"));
    }

    #[test]
    fn hybrid_weight_clamping() {
        let p = EvictionPolicy::hybrid(1.5);
        assert!((p.attention_weight_f32() - 1.0).abs() < 0.02);
        let p = EvictionPolicy::hybrid(-0.5);
        assert!(p.attention_weight_f32().abs() < 0.02);
    }

    #[test]
    fn policy_attention_weight_values() {
        assert_eq!(EvictionPolicy::Lru.attention_weight_f32(), 0.0);
        assert_eq!(EvictionPolicy::Fifo.attention_weight_f32(), 0.0);
        assert_eq!(EvictionPolicy::AttentionScore.attention_weight_f32(), 1.0);
    }

    // -----------------------------------------------------------------------
    // KvEviction — LRU
    // -----------------------------------------------------------------------

    #[test]
    fn lru_evicts_oldest() {
        let mut ev = KvEviction::new(EvictionPolicy::Lru);
        for i in 0..5 {
            ev.insert(i);
        }
        let evicted = ev.select_evictions(2);
        assert_eq!(evicted.len(), 2);
        // Oldest (positions 0, 1) should be evicted first.
        assert!(evicted.contains(&0));
        assert!(evicted.contains(&1));
    }

    #[test]
    fn lru_evict_all() {
        let mut ev = KvEviction::new(EvictionPolicy::Lru);
        for i in 0..3 {
            ev.insert(i);
        }
        let evicted = ev.select_evictions(10); // more than available
        assert_eq!(evicted.len(), 3);
    }

    #[test]
    fn lru_evict_zero() {
        let mut ev = KvEviction::new(EvictionPolicy::Lru);
        ev.insert(0);
        let evicted = ev.select_evictions(0);
        assert!(evicted.is_empty());
    }

    #[test]
    fn lru_remove_positions() {
        let mut ev = KvEviction::new(EvictionPolicy::Lru);
        for i in 0..5 {
            ev.insert(i);
        }
        ev.remove_positions(&[0, 2]);
        assert_eq!(ev.len(), 3);
        let evicted = ev.select_evictions(1);
        // Position 1 is now the oldest remaining.
        assert_eq!(evicted, vec![1]);
    }

    // -----------------------------------------------------------------------
    // KvEviction — FIFO
    // -----------------------------------------------------------------------

    #[test]
    fn fifo_evicts_oldest() {
        let mut ev = KvEviction::new(EvictionPolicy::Fifo);
        for i in 0..5 {
            ev.insert(i * 10);
        }
        let evicted = ev.select_evictions(2);
        assert!(evicted.contains(&0));
        assert!(evicted.contains(&10));
    }

    // -----------------------------------------------------------------------
    // KvEviction — AttentionScore
    // -----------------------------------------------------------------------

    #[test]
    fn attention_score_evicts_lowest() {
        let mut ev = KvEviction::new(EvictionPolicy::AttentionScore);
        for i in 0..5 {
            ev.insert(i);
        }
        // Give position 2 and 4 high scores, others low.
        ev.update_scores(&[0.1, 0.1, 10.0, 0.1, 10.0]);
        let evicted = ev.select_evictions(2);
        assert_eq!(evicted.len(), 2);
        // Positions 0, 1, 3 have low scores; two of them should be evicted.
        for &pos in &evicted {
            assert!(pos != 2 && pos != 4, "high-score entries should not be evicted");
        }
    }

    #[test]
    fn attention_score_accumulates() {
        let mut ev = KvEviction::new(EvictionPolicy::AttentionScore);
        ev.insert(0);
        ev.insert(1);
        ev.update_scores(&[1.0, 0.5]);
        ev.update_scores(&[1.0, 0.5]);
        // Position 0 has cumulative 2.0, position 1 has 1.0.
        let evicted = ev.select_evictions(1);
        assert_eq!(evicted, vec![1], "lower cumulative score should be evicted");
    }

    #[test]
    fn attention_score_empty() {
        let ev = KvEviction::new(EvictionPolicy::AttentionScore);
        assert!(ev.is_empty());
        let evicted = ev.select_evictions(1);
        assert!(evicted.is_empty());
    }

    // -----------------------------------------------------------------------
    // KvEviction — Hybrid
    // -----------------------------------------------------------------------

    #[test]
    fn hybrid_balances_recency_and_score() {
        let mut ev = KvEviction::new(EvictionPolicy::hybrid(0.5));
        for i in 0..4 {
            ev.insert(i);
        }
        // Entry 0 is oldest but has high attention score.
        // Entry 3 is newest but has low attention score.
        ev.update_scores(&[10.0, 0.0, 0.0, 0.0]);
        let evicted = ev.select_evictions(1);
        // With 50/50 weighting, entry 1 or 2 should be evicted (mid-age, low score).
        assert_ne!(evicted[0], 0, "entry 0 has high attention — should survive");
    }

    // -----------------------------------------------------------------------
    // KvMerger — token merging
    // -----------------------------------------------------------------------

    #[test]
    fn merge_identical_pairs() {
        let merger = KvMerger::new(0.99);
        // Two identical rows → should merge.
        let data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let (merged, count) = merger.merge(&data, 2, 3);
        assert_eq!(count, 1, "identical rows should merge into one");
        assert_eq!(merged.len(), 3);
        for &v in &merged {
            // Average of identical = same.
            assert!(
                (v - data[merged.len() - 3 + merged.iter().position(|&x| x == v).unwrap()]).abs()
                    < 1e-6
            );
        }
    }

    #[test]
    fn merge_dissimilar_kept() {
        let merger = KvMerger::new(0.99);
        // Two orthogonal rows → should not merge.
        let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let (_, count) = merger.merge(&data, 2, 3);
        assert_eq!(count, 2, "dissimilar rows should not merge");
    }

    #[test]
    fn merge_empty() {
        let merger = KvMerger::new(0.5);
        let (merged, count) = merger.merge(&[], 0, 4);
        assert_eq!(count, 0);
        assert!(merged.is_empty());
    }

    #[test]
    fn merge_single_entry() {
        let merger = KvMerger::new(0.5);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let (merged, count) = merger.merge(&data, 1, 4);
        assert_eq!(count, 1);
        assert_eq!(merged, data);
    }

    #[test]
    fn merge_reduces_entry_count() {
        let merger = KvMerger::new(0.90);
        // Several similar rows mixed with one outlier.
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend_from_slice(&[1.0, 1.0, 1.0, 1.0]); // similar
        }
        data.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]); // dissimilar
        let (_, count) = merger.merge(&data, 5, 4);
        assert!(count < 5, "merging should reduce entries: got {count}");
        assert!(count >= 2, "outlier should remain: got {count}");
    }

    #[test]
    fn merge_kv_paired() {
        let merger = KvMerger::new(0.99);
        let keys = vec![1.0, 2.0, 1.0, 2.0]; // 2 identical key rows
        let vals = vec![5.0, 6.0, 7.0, 8.0]; // different values
        let (mk, mv, count) = merger.merge_kv(&keys, &vals, 2, 2);
        assert_eq!(count, 1, "identical keys should merge");
        assert_eq!(mk.len(), 2);
        assert_eq!(mv.len(), 2);
        // Values should be averaged.
        assert!((mv[0] - 6.0).abs() < 1e-6);
        assert!((mv[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn merge_kv_empty() {
        let merger = KvMerger::new(0.5);
        let (mk, mv, count) = merger.merge_kv(&[], &[], 0, 4);
        assert_eq!(count, 0);
        assert!(mk.is_empty());
        assert!(mv.is_empty());
    }

    #[test]
    fn merge_three_similar_pairs() {
        let merger = KvMerger::new(0.99);
        // Three identical rows — greedy: first two merge, third stays.
        let data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let (_, count) = merger.merge(&data, 3, 2);
        assert_eq!(count, 2, "greedy merge: 3 identical → 2 (first pair merges)");
    }

    // -----------------------------------------------------------------------
    // Cosine similarity
    // -----------------------------------------------------------------------

    #[test]
    fn cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // -----------------------------------------------------------------------
    // SinkTokenManager
    // -----------------------------------------------------------------------

    #[test]
    fn sink_protect_filters_sinks() {
        let mgr = SinkTokenManager::new(3);
        let candidates = vec![0, 1, 2, 3, 4, 5];
        let filtered = mgr.protect_sinks(&candidates);
        assert_eq!(filtered, vec![3, 4, 5]);
    }

    #[test]
    fn sink_protect_empty_candidates() {
        let mgr = SinkTokenManager::new(3);
        assert!(mgr.protect_sinks(&[]).is_empty());
    }

    #[test]
    fn sink_protect_all_sinks() {
        let mgr = SinkTokenManager::new(5);
        let candidates = vec![0, 1, 2, 3];
        assert!(mgr.protect_sinks(&candidates).is_empty());
    }

    #[test]
    fn sink_is_sink() {
        let mgr = SinkTokenManager::new(2);
        assert!(mgr.is_sink(0));
        assert!(mgr.is_sink(1));
        assert!(!mgr.is_sink(2));
    }

    #[test]
    fn sink_zero_sinks() {
        let mgr = SinkTokenManager::new(0);
        assert!(!mgr.is_sink(0));
        assert_eq!(mgr.protect_sinks(&[0, 1, 2]), vec![0, 1, 2]);
    }

    #[test]
    fn sink_extract_sinks() {
        let mgr = SinkTokenManager::new(2);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 rows × 2 dim
        let sinks = mgr.extract_sinks(&data, 3, 2);
        assert_eq!(sinks, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn sink_extract_more_than_available() {
        let mgr = SinkTokenManager::new(10);
        let data = vec![1.0, 2.0, 3.0]; // 3 rows × 1 dim
        let sinks = mgr.extract_sinks(&data, 3, 1);
        assert_eq!(sinks.len(), 3); // min(10, 3) = 3
    }

    // -----------------------------------------------------------------------
    // WindowedKv
    // -----------------------------------------------------------------------

    #[test]
    fn windowed_kv_basic() {
        let mut wkv = WindowedKv::new(2, 3, 0); // head_dim=2, window=3, sinks=0
        wkv.append(&[1.0, 2.0], &[10.0, 20.0]);
        wkv.append(&[3.0, 4.0], &[30.0, 40.0]);
        assert_eq!(wkv.current_len, 2);
        assert_eq!(wkv.keys(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn windowed_kv_overflow_no_sinks() {
        let mut wkv = WindowedKv::new(2, 2, 0); // window=2, no sinks
        wkv.append(&[1.0, 2.0], &[10.0, 20.0]); // pos 0
        wkv.append(&[3.0, 4.0], &[30.0, 40.0]); // pos 1 — full
        wkv.append(&[5.0, 6.0], &[50.0, 60.0]); // pos 2 → evicts pos 0
        assert_eq!(wkv.current_len, 2);
        assert_eq!(wkv.keys(), &[3.0, 4.0, 5.0, 6.0]);
        assert_eq!(wkv.values(), &[30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn windowed_kv_with_sinks() {
        let mut wkv = WindowedKv::new(2, 2, 2); // window=2, sinks=2
        // Fill sinks.
        wkv.append(&[1.0, 1.0], &[10.0, 10.0]); // sink 0
        wkv.append(&[2.0, 2.0], &[20.0, 20.0]); // sink 1
        // Fill window.
        wkv.append(&[3.0, 3.0], &[30.0, 30.0]); // window pos 0
        wkv.append(&[4.0, 4.0], &[40.0, 40.0]); // window pos 1 — full
        assert_eq!(wkv.current_len, 4);
        assert_eq!(wkv.capacity(), 4);

        // Overflow: window shifts, sinks preserved.
        wkv.append(&[5.0, 5.0], &[50.0, 50.0]);
        assert_eq!(wkv.current_len, 4);
        // Sinks [1,1] and [2,2] preserved, window now [4,4] and [5,5].
        assert_eq!(wkv.keys(), &[1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 5.0, 5.0]);
    }

    #[test]
    fn windowed_kv_clear() {
        let mut wkv = WindowedKv::new(4, 3, 1);
        wkv.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        wkv.clear();
        assert_eq!(wkv.current_len, 0);
        assert!(wkv.keys().is_empty());
    }

    #[test]
    fn windowed_kv_empty() {
        let wkv = WindowedKv::new(4, 5, 2);
        assert_eq!(wkv.current_len, 0);
        assert!(wkv.keys().is_empty());
        assert!(wkv.values().is_empty());
    }

    #[test]
    fn windowed_kv_capacity() {
        let wkv = WindowedKv::new(4, 10, 3);
        assert_eq!(wkv.capacity(), 13); // 3 sinks + 10 window
    }

    #[test]
    fn windowed_kv_multi_append() {
        let mut wkv = WindowedKv::new(2, 3, 0);
        // Append 2 tokens at once.
        wkv.append(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(wkv.current_len, 2);
        assert_eq!(wkv.keys(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn windowed_kv_sink_preserved_after_many_overwrites() {
        let mut wkv = WindowedKv::new(1, 2, 1); // dim=1, window=2, sinks=1
        wkv.append(&[100.0], &[100.0]); // sink
        wkv.append(&[1.0], &[1.0]); // window slot 0
        wkv.append(&[2.0], &[2.0]); // window slot 1 — full
        // Several overwrites.
        for i in 3..10 {
            wkv.append(&[i as f32], &[i as f32]);
        }
        assert_eq!(wkv.current_len, 3);
        // Sink should still be 100.0.
        assert_eq!(wkv.keys()[0], 100.0);
        assert_eq!(wkv.values()[0], 100.0);
    }

    // -----------------------------------------------------------------------
    // KvConfig constructors
    // -----------------------------------------------------------------------

    #[test]
    fn config_sliding_window() {
        let cfg = KvConfig::sliding_window(512, 4);
        assert_eq!(cfg.max_cache_tokens, 512);
        assert_eq!(cfg.num_sink_tokens, 4);
        assert_eq!(cfg.compression_method, CompressionMethod::SlidingWindow);
    }

    #[test]
    fn config_int8() {
        let cfg = KvConfig::int8();
        assert_eq!(cfg.compression_method, CompressionMethod::Quantize(QuantFormat::Int8));
    }

    #[test]
    fn config_int4() {
        let cfg = KvConfig::int4();
        assert_eq!(cfg.compression_method, CompressionMethod::Quantize(QuantFormat::Int4));
    }

    #[test]
    fn config_eviction() {
        let cfg = KvConfig::eviction(256, EvictionPolicy::AttentionScore);
        assert_eq!(cfg.max_cache_tokens, 256);
    }

    #[test]
    fn config_merge() {
        let cfg = KvConfig::merge(0.8);
        assert_eq!(cfg.compression_method, CompressionMethod::Merge);
        assert!((cfg.merge_similarity_threshold - 0.8).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // KvCompressionStats
    // -----------------------------------------------------------------------

    #[test]
    fn stats_compression_ratio() {
        let stats = KvCompressionStats {
            original_entries: 100,
            compressed_entries: 50,
            original_bytes: 4000,
            compressed_bytes: 1000,
            quality_loss_estimate: 0.01,
            method: CompressionMethod::Quantize(QuantFormat::Int8),
        };
        assert!((stats.compression_ratio() - 4.0).abs() < 1e-6);
        assert_eq!(stats.bytes_saved(), 3000);
    }

    #[test]
    fn stats_zero_compressed_bytes() {
        let stats = KvCompressionStats {
            original_entries: 0,
            compressed_entries: 0,
            original_bytes: 0,
            compressed_bytes: 0,
            quality_loss_estimate: 0.0,
            method: CompressionMethod::Evict(EvictionPolicy::Lru),
        };
        assert!(stats.compression_ratio().is_infinite());
    }

    #[test]
    fn stats_display() {
        let stats = KvCompressionStats {
            original_entries: 10,
            compressed_entries: 5,
            original_bytes: 200,
            compressed_bytes: 100,
            quality_loss_estimate: 0.05,
            method: CompressionMethod::Merge,
        };
        let s = format!("{stats}");
        assert!(s.contains("10"));
        assert!(s.contains("5"));
        assert!(s.contains("saved"));
    }

    #[test]
    fn stats_no_savings() {
        let stats = KvCompressionStats {
            original_entries: 10,
            compressed_entries: 10,
            original_bytes: 100,
            compressed_bytes: 100,
            quality_loss_estimate: 0.0,
            method: CompressionMethod::SlidingWindow,
        };
        assert_eq!(stats.bytes_saved(), 0);
        assert!((stats.compression_ratio() - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // CompressionMethod display
    // -----------------------------------------------------------------------

    #[test]
    fn compression_method_display() {
        assert_eq!(format!("{}", CompressionMethod::Quantize(QuantFormat::Int8)), "Quantize(INT8)");
        assert_eq!(format!("{}", CompressionMethod::Merge), "TokenMerge");
        assert_eq!(format!("{}", CompressionMethod::SlidingWindow), "SlidingWindow");
    }

    // -----------------------------------------------------------------------
    // sign_extend_4bit
    // -----------------------------------------------------------------------

    #[test]
    fn sign_extend_positive() {
        assert_eq!(sign_extend_4bit(0x07), 7);
        assert_eq!(sign_extend_4bit(0x00), 0);
        assert_eq!(sign_extend_4bit(0x01), 1);
    }

    #[test]
    fn sign_extend_negative() {
        // 0xF = -1 in 4-bit signed, 0x9 = -7
        assert_eq!(sign_extend_4bit(0x0F), -1);
        assert_eq!(sign_extend_4bit(0x09), -7);
        assert_eq!(sign_extend_4bit(0x08), -8);
    }

    // -----------------------------------------------------------------------
    // Property: compressed size ≤ original (INT8)
    // -----------------------------------------------------------------------

    #[test]
    fn int8_compressed_size_leq_original() {
        let quantizer = KvQuantizer::new(QuantFormat::Int8);
        for heads in [1, 2, 4] {
            for seq in [1, 8, 32] {
                let dim = 16;
                let data: Vec<f32> = (0..(heads * seq * dim)).map(|i| i as f32 * 0.01).collect();
                let original_bytes = data.len() * size_of::<f32>();
                let (q, scales) = quantizer.quantize_tensor(&data, heads, seq, dim);
                let compressed_bytes = q.len() + scales.len() * size_of::<f32>();
                assert!(
                    compressed_bytes <= original_bytes,
                    "INT8 compressed ({compressed_bytes}) > original ({original_bytes})"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Property: compressed size ≤ original (INT4)
    // -----------------------------------------------------------------------

    #[test]
    fn int4_compressed_size_leq_original() {
        let quantizer = KvQuantizer::new(QuantFormat::Int4);
        for heads in [1, 2, 4] {
            for seq in [1, 8, 32] {
                let dim = 16;
                let data: Vec<f32> = (0..(heads * seq * dim)).map(|i| i as f32 * 0.01).collect();
                let original_bytes = data.len() * size_of::<f32>();
                let (q, scales) = quantizer.quantize_tensor(&data, heads, seq, dim);
                let compressed_bytes = q.len() + scales.len() * size_of::<f32>();
                assert!(
                    compressed_bytes <= original_bytes,
                    "INT4 compressed ({compressed_bytes}) > original ({original_bytes})"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Property: merge never increases entry count
    // -----------------------------------------------------------------------

    #[test]
    fn merge_never_increases_count() {
        let merger = KvMerger::new(0.5);
        for n in [0, 1, 2, 5, 10] {
            let dim = 4;
            let data: Vec<f32> = (0..(n * dim)).map(|i| (i as f32).sin()).collect();
            let (_, count) = merger.merge(&data, n, dim);
            assert!(count <= n, "merge should not increase entry count: {count} > {n}");
        }
    }

    // -----------------------------------------------------------------------
    // QuantizedKvCache
    // -----------------------------------------------------------------------

    #[test]
    fn quantized_kv_cache_total_bytes() {
        let qkv = QuantizedKvCache {
            keys: vec![0u8; 32],
            values: vec![0u8; 32],
            key_scales: vec![0.0; 4],
            value_scales: vec![0.0; 4],
            format: QuantFormat::Int8,
            num_heads: 4,
            seq_len: 8,
            head_dim: 1,
        };
        // 32 + 32 + (4+4)*4 = 96
        assert_eq!(qkv.total_bytes(), 96);
    }

    // -----------------------------------------------------------------------
    // Edge: KvEviction with score updates exceeding entry count
    // -----------------------------------------------------------------------

    #[test]
    fn eviction_partial_score_update() {
        let mut ev = KvEviction::new(EvictionPolicy::AttentionScore);
        ev.insert(0);
        ev.insert(1);
        ev.insert(2);
        // Only update 2 of 3 entries.
        ev.update_scores(&[5.0, 1.0]);
        let evicted = ev.select_evictions(1);
        // Entry 2 has score 0.0 — lowest.
        assert_eq!(evicted, vec![2]);
    }

    // -----------------------------------------------------------------------
    // Edge: merge threshold 0 (merge nothing)
    // -----------------------------------------------------------------------

    #[test]
    fn merge_threshold_zero_merges_almost_all() {
        // threshold 0 means any positive similarity triggers a merge.
        let merger = KvMerger::new(0.0);
        let data = vec![1.0, 0.0, 0.5, 0.5]; // 2 rows, dim=2, cos > 0
        let (_, count) = merger.merge(&data, 2, 2);
        // cos(a, b) ≈ 0.45, which is > 0.0 → should merge.
        assert_eq!(count, 1, "threshold=0 should merge any non-orthogonal pair");
    }

    // -----------------------------------------------------------------------
    // Edge: merge threshold 1 (merge only identical)
    // -----------------------------------------------------------------------

    #[test]
    fn merge_threshold_one_very_strict() {
        let merger = KvMerger::new(1.0);
        // Near-identical but not identical.
        let data = vec![1.0, 2.0, 1.0, 2.001];
        let (_, count) = merger.merge(&data, 2, 2);
        // cos ≈ 0.99999+ but may not exceed 1.0 → depends on precision.
        // With threshold 1.0 it should NOT merge since we use strict >.
        assert_eq!(count, 2, "threshold=1.0 should not merge near-identical");
    }

    // -----------------------------------------------------------------------
    // Integration: eviction + sink protection
    // -----------------------------------------------------------------------

    #[test]
    fn eviction_with_sink_protection() {
        let mut ev = KvEviction::new(EvictionPolicy::Lru);
        for i in 0..6 {
            ev.insert(i);
        }
        let candidates = ev.select_evictions(3);
        let sink_mgr = SinkTokenManager::new(2);
        let filtered = sink_mgr.protect_sinks(&candidates);
        // Original eviction targets oldest: 0, 1, 2. After sink protection
        // positions 0, 1 are removed → only position 2.
        assert!(!filtered.contains(&0));
        assert!(!filtered.contains(&1));
    }
}
