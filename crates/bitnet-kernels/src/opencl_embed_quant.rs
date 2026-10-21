//! Embedding table quantization for memory-efficient inference.
//!
//! Provides INT8, INT4, product-quantized, and binary embedding representations
//! with CPU reference implementations and OpenCL kernel source for GPU dispatch.
//!
//! # Supported quantization modes
//!
//! | Mode   | Bits | Per-row overhead | Typical cosine vs FP32 |
//! |--------|------|------------------|------------------------|
//! | INT8   | 8    | 1× f32 scale     | > 0.99                 |
//! | INT4   | 4    | 1× f32 scale     | > 0.95                 |
//! | PQ     | ~2-4 | codebook table    | > 0.90 (tunable)       |
//! | Binary | 1    | none              | variable               |
//!
//! No OpenCL runtime (`opencl3`) is required — all operations have scalar CPU
//! reference paths.

use bitnet_common::{KernelError, Result};
use std::fmt;

// ── OpenCL kernel source ─────────────────────────────────────────

/// OpenCL kernel source for quantized embedding lookup operations.
pub const EMBED_QUANT_CL: &str = r#"
// Quantized embedding lookup kernels for Intel Arc / OpenCL 3.0
//
// INT8 quantized embedding lookup: each row stored as int8 + f32 scale.
// Dequantization: output[i] = (float)quant[row * dim + i] * scale[row]
__kernel void embedding_lookup_int8(
    __global const char*  quant_weight, // [vocab_size, embedding_dim] int8
    __global const float* scales,       // [vocab_size]
    __global const uint*  token_ids,    // [seq_len]
    __global float*       output,       // [seq_len, embedding_dim]
    const uint            vocab_size,
    const uint            embedding_dim
) {
    uint tid = get_global_id(0); // token index
    uint dim = get_global_id(1); // dimension index
    if (tid >= get_global_size(0) || dim >= embedding_dim) return;

    uint token = token_ids[tid];
    uint out_idx = tid * embedding_dim + dim;

    if (token >= vocab_size) {
        output[out_idx] = 0.0f;
        return;
    }

    float scale = scales[token];
    char  qval  = quant_weight[token * embedding_dim + dim];
    output[out_idx] = (float)qval * scale;
}

// INT4 quantized embedding lookup: two values packed per byte.
// Even index → low nibble, odd index → high nibble.
__kernel void embedding_lookup_int4(
    __global const uchar* packed_weight, // [vocab_size, embedding_dim / 2]
    __global const float* scales,        // [vocab_size]
    __global const uint*  token_ids,     // [seq_len]
    __global float*       output,        // [seq_len, embedding_dim]
    const uint            vocab_size,
    const uint            embedding_dim
) {
    uint tid = get_global_id(0);
    uint dim = get_global_id(1);
    if (tid >= get_global_size(0) || dim >= embedding_dim) return;

    uint token = token_ids[tid];
    uint out_idx = tid * embedding_dim + dim;

    if (token >= vocab_size) {
        output[out_idx] = 0.0f;
        return;
    }

    float scale = scales[token];
    uint byte_idx = token * (embedding_dim / 2) + (dim / 2);
    uchar packed = packed_weight[byte_idx];
    // Low nibble for even indices, high nibble for odd
    int qval = (dim % 2 == 0) ? (int)(packed & 0x0F) - 8
                               : (int)(packed >> 4)    - 8;
    output[out_idx] = (float)qval * scale;
}

// Batch PQ lookup: decode sub-vector codes through codebook table.
__kernel void embedding_lookup_pq(
    __global const uchar* codes,     // [vocab_size, num_subvectors]
    __global const float* codebook,  // [num_subvectors, num_centroids, sub_dim]
    __global const uint*  token_ids, // [seq_len]
    __global float*       output,    // [seq_len, embedding_dim]
    const uint            vocab_size,
    const uint            num_subvectors,
    const uint            num_centroids,
    const uint            sub_dim
) {
    uint tid = get_global_id(0);
    uint sv  = get_global_id(1); // sub-vector index
    if (tid >= get_global_size(0) || sv >= num_subvectors) return;

    uint token = token_ids[tid];
    uint embedding_dim = num_subvectors * sub_dim;
    uint out_base = tid * embedding_dim + sv * sub_dim;

    if (token >= vocab_size) {
        for (uint d = 0; d < sub_dim; d++) output[out_base + d] = 0.0f;
        return;
    }

    uchar code = codes[token * num_subvectors + sv];
    uint cb_base = (sv * num_centroids + (uint)code) * sub_dim;
    for (uint d = 0; d < sub_dim; d++) {
        output[out_base + d] = codebook[cb_base + d];
    }
}
"#;

// ── Quantization precision ───────────────────────────────────────

/// Quantization bit-width for embedding tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantPrecision {
    /// 8-bit signed integer per element.
    Int8,
    /// 4-bit signed integer (two values per byte).
    Int4,
}

impl fmt::Display for QuantPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int8 => write!(f, "INT8"),
            Self::Int4 => write!(f, "INT4"),
        }
    }
}

// ── QuantizedEmbeddingTable ──────────────────────────────────────

/// INT8 or INT4 quantized vocabulary embedding table with per-row scales.
///
/// Layout:
/// - INT8: `data` has `vocab_size * embedding_dim` bytes (one `i8` per element).
/// - INT4: `data` has `vocab_size * (embedding_dim / 2)` bytes (two elements per byte).
/// - `scales`: one `f32` per vocabulary row.
///
/// Dequantization: `float_val = (i8_val as f32) * scale[row]`.
#[derive(Debug, Clone)]
pub struct QuantizedEmbeddingTable {
    /// Packed quantized weights.
    pub data: Vec<u8>,
    /// Per-row dequantization scales.
    pub scales: Vec<f32>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Full embedding dimension.
    pub embedding_dim: usize,
    /// Quantization precision.
    pub precision: QuantPrecision,
}

impl QuantizedEmbeddingTable {
    /// Quantize a full-precision embedding table to INT8 or INT4.
    ///
    /// For INT8: each element mapped to `[-127, 127]` with absmax per row.
    /// For INT4: each element mapped to `[-7, 7]` with absmax per row.
    pub fn quantize(
        weight: &[f32],
        vocab_size: usize,
        embedding_dim: usize,
        precision: QuantPrecision,
    ) -> Result<Self> {
        let expected = vocab_size * embedding_dim;
        if weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "weight length {} != vocab_size({}) * embedding_dim({})",
                    weight.len(),
                    vocab_size,
                    embedding_dim,
                ),
            }
            .into());
        }
        if precision == QuantPrecision::Int4 && !embedding_dim.is_multiple_of(2) {
            return Err(KernelError::InvalidArguments {
                reason: format!("INT4 requires even embedding_dim, got {}", embedding_dim),
            }
            .into());
        }

        let mut scales = vec![0.0f32; vocab_size];
        let data = match precision {
            QuantPrecision::Int8 => {
                let mut buf = vec![0u8; vocab_size * embedding_dim];
                for row in 0..vocab_size {
                    let row_start = row * embedding_dim;
                    let row_slice = &weight[row_start..row_start + embedding_dim];
                    let absmax = row_slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                    scales[row] = scale;
                    for (j, &val) in row_slice.iter().enumerate() {
                        let q = (val / scale).round().clamp(-127.0, 127.0) as i8;
                        buf[row * embedding_dim + j] = q as u8;
                    }
                }
                buf
            }
            QuantPrecision::Int4 => {
                let packed_dim = embedding_dim / 2;
                let mut buf = vec![0u8; vocab_size * packed_dim];
                for row in 0..vocab_size {
                    let row_start = row * embedding_dim;
                    let row_slice = &weight[row_start..row_start + embedding_dim];
                    let absmax = row_slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let scale = if absmax > 0.0 { absmax / 7.0 } else { 1.0 };
                    scales[row] = scale;
                    for j in (0..embedding_dim).step_by(2) {
                        let lo = (row_slice[j] / scale).round().clamp(-7.0, 7.0) as i8;
                        let hi = (row_slice[j + 1] / scale).round().clamp(-7.0, 7.0) as i8;
                        // Pack: low nibble = lo + 8, high nibble = hi + 8
                        let lo_u = (lo + 8) as u8 & 0x0F;
                        let hi_u = ((hi + 8) as u8 & 0x0F) << 4;
                        buf[row * packed_dim + j / 2] = lo_u | hi_u;
                    }
                }
                buf
            }
        };

        Ok(Self { data, scales, vocab_size, embedding_dim, precision })
    }

    /// Dequantize a single row back to f32.
    pub fn dequantize_row(&self, row: usize) -> Result<Vec<f32>> {
        if row >= self.vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!("row {} >= vocab_size {}", row, self.vocab_size),
            }
            .into());
        }

        let scale = self.scales[row];
        let mut out = vec![0.0f32; self.embedding_dim];
        match self.precision {
            QuantPrecision::Int8 => {
                let offset = row * self.embedding_dim;
                for (j, val) in out.iter_mut().enumerate() {
                    let q = self.data[offset + j] as i8;
                    *val = q as f32 * scale;
                }
            }
            QuantPrecision::Int4 => {
                let packed_dim = self.embedding_dim / 2;
                let offset = row * packed_dim;
                for (j, val) in out.iter_mut().enumerate() {
                    let byte = self.data[offset + j / 2];
                    let nibble = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    let q = nibble as i8 - 8;
                    *val = q as f32 * scale;
                }
            }
        }
        Ok(out)
    }

    /// Compute the memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Compute the FP32 equivalent memory footprint.
    pub fn fp32_memory_bytes(&self) -> usize {
        self.vocab_size * self.embedding_dim * 4
    }
}

// ── ProductQuantizer ─────────────────────────────────────────────

/// Product Quantization for extreme embedding compression.
///
/// The embedding vector is split into `num_subvectors` sub-vectors, each
/// independently quantized to the nearest centroid in its codebook.
/// Storage: one `u8` code per sub-vector per vocabulary entry.
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    /// Codebook: `[num_subvectors][num_centroids][sub_dim]` flattened.
    pub codebook: Vec<f32>,
    /// Encoded codes: `[vocab_size][num_subvectors]`.
    pub codes: Vec<u8>,
    /// Number of sub-vector partitions.
    pub num_subvectors: usize,
    /// Number of centroids per sub-vector (max 256 for u8 codes).
    pub num_centroids: usize,
    /// Dimension of each sub-vector.
    pub sub_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
}

impl ProductQuantizer {
    /// Create from pre-trained codebook and codes.
    pub fn new(
        codebook: Vec<f32>,
        codes: Vec<u8>,
        num_subvectors: usize,
        num_centroids: usize,
        sub_dim: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        let expected_cb = num_subvectors * num_centroids * sub_dim;
        if codebook.len() != expected_cb {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "codebook length {} != num_subvectors({}) * num_centroids({}) * sub_dim({})",
                    codebook.len(),
                    num_subvectors,
                    num_centroids,
                    sub_dim,
                ),
            }
            .into());
        }
        let expected_codes = vocab_size * num_subvectors;
        if codes.len() != expected_codes {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "codes length {} != vocab_size({}) * num_subvectors({})",
                    codes.len(),
                    vocab_size,
                    num_subvectors,
                ),
            }
            .into());
        }
        if num_centroids > 256 {
            return Err(KernelError::InvalidArguments {
                reason: format!("num_centroids {} exceeds u8 max 256", num_centroids),
            }
            .into());
        }

        Ok(Self { codebook, codes, num_subvectors, num_centroids, sub_dim, vocab_size })
    }

    /// Full embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.num_subvectors * self.sub_dim
    }

    /// Decode a single row back to f32 by looking up codebook entries.
    pub fn decode_row(&self, row: usize) -> Result<Vec<f32>> {
        if row >= self.vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!("row {} >= vocab_size {}", row, self.vocab_size),
            }
            .into());
        }

        let dim = self.embedding_dim();
        let mut out = vec![0.0f32; dim];
        for sv in 0..self.num_subvectors {
            let code = self.codes[row * self.num_subvectors + sv] as usize;
            let cb_offset = (sv * self.num_centroids + code) * self.sub_dim;
            let out_offset = sv * self.sub_dim;
            out[out_offset..out_offset + self.sub_dim]
                .copy_from_slice(&self.codebook[cb_offset..cb_offset + self.sub_dim]);
        }
        Ok(out)
    }

    /// Encode a single f32 vector to PQ codes (nearest centroid per sub-vector).
    pub fn encode_vector(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let dim = self.embedding_dim();
        if vector.len() != dim {
            return Err(KernelError::InvalidArguments {
                reason: format!("vector length {} != embedding_dim {}", vector.len(), dim),
            }
            .into());
        }

        let mut codes = vec![0u8; self.num_subvectors];
        for (sv, code) in codes.iter_mut().enumerate() {
            let sv_start = sv * self.sub_dim;
            let sub_vec = &vector[sv_start..sv_start + self.sub_dim];
            let mut best_code = 0u8;
            let mut best_dist = f32::MAX;
            for c in 0..self.num_centroids {
                let cb_offset = (sv * self.num_centroids + c) * self.sub_dim;
                let centroid = &self.codebook[cb_offset..cb_offset + self.sub_dim];
                let dist: f32 = sub_vec.iter().zip(centroid).map(|(a, b)| (a - b) * (a - b)).sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_code = c as u8;
                }
            }
            *code = best_code;
        }
        Ok(codes)
    }

    /// Memory footprint: codebook + codes.
    pub fn memory_bytes(&self) -> usize {
        self.codebook.len() * 4 + self.codes.len()
    }
}

// ── BinaryEmbedding ──────────────────────────────────────────────

/// Binary (1-bit) embedding table for hash-based retrieval.
///
/// Each embedding is stored as a packed bit vector. Similarity is
/// computed via Hamming distance (XOR + popcount).
#[derive(Debug, Clone)]
pub struct BinaryEmbedding {
    /// Packed bits: `[vocab_size][ ceil(embedding_dim / 8) ]`.
    pub data: Vec<u8>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Original embedding dimension (in bits).
    pub embedding_dim: usize,
}

impl BinaryEmbedding {
    /// Binarize an FP32 embedding table: positive → 1, non-positive → 0.
    pub fn from_float(weight: &[f32], vocab_size: usize, embedding_dim: usize) -> Result<Self> {
        let expected = vocab_size * embedding_dim;
        if weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "weight length {} != vocab_size({}) * embedding_dim({})",
                    weight.len(),
                    vocab_size,
                    embedding_dim,
                ),
            }
            .into());
        }

        let bytes_per_row = embedding_dim.div_ceil(8);
        let mut data = vec![0u8; vocab_size * bytes_per_row];
        for row in 0..vocab_size {
            for j in 0..embedding_dim {
                if weight[row * embedding_dim + j] > 0.0 {
                    let byte_idx = row * bytes_per_row + j / 8;
                    data[byte_idx] |= 1 << (j % 8);
                }
            }
        }
        Ok(Self { data, vocab_size, embedding_dim })
    }

    /// Get the binary embedding for a row as a byte slice.
    pub fn get_row(&self, row: usize) -> Result<&[u8]> {
        if row >= self.vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!("row {} >= vocab_size {}", row, self.vocab_size),
            }
            .into());
        }
        let bpr = self.bytes_per_row();
        Ok(&self.data[row * bpr..(row + 1) * bpr])
    }

    /// Hamming distance between two rows.
    pub fn hamming_distance(&self, row_a: usize, row_b: usize) -> Result<u32> {
        let a = self.get_row(row_a)?;
        let b = self.get_row(row_b)?;
        Ok(a.iter().zip(b).map(|(&x, &y)| (x ^ y).count_ones()).sum())
    }

    /// Unpack a binary row to `{-1.0, +1.0}` float vector.
    pub fn unpack_row_signed(&self, row: usize) -> Result<Vec<f32>> {
        if row >= self.vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!("row {} >= vocab_size {}", row, self.vocab_size),
            }
            .into());
        }
        let bpr = self.bytes_per_row();
        let mut out = vec![-1.0f32; self.embedding_dim];
        for (j, val) in out.iter_mut().enumerate() {
            let byte = self.data[row * bpr + j / 8];
            if byte & (1 << (j % 8)) != 0 {
                *val = 1.0;
            }
        }
        Ok(out)
    }

    /// Bytes per row.
    fn bytes_per_row(&self) -> usize {
        self.embedding_dim.div_ceil(8)
    }

    /// Memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }
}

// ── AdaptiveQuant ────────────────────────────────────────────────

/// Row-level importance for adaptive mixed-precision quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowImportance {
    /// High frequency / critical token — use INT8.
    High,
    /// Medium importance — use INT4.
    Medium,
    /// Low frequency — use binary.
    Low,
}

/// Adaptive mixed-precision embedding: assigns per-row precision based on
/// token frequency or importance scores.
#[derive(Debug, Clone)]
pub struct AdaptiveQuant {
    /// Per-row importance classification.
    pub importance: Vec<RowImportance>,
    /// INT8 quantized rows (row index → quantized data + scale).
    int8_data: Vec<u8>,
    int8_scales: Vec<f32>,
    /// INT4 quantized rows.
    int4_data: Vec<u8>,
    int4_scales: Vec<f32>,
    /// Binary rows.
    binary_data: Vec<u8>,
    /// Mapping: global row → (tier, local_row_index).
    row_map: Vec<(RowImportance, usize)>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding dimension.
    pub embedding_dim: usize,
}

impl AdaptiveQuant {
    /// Build an adaptive-quantized table from FP32 weights and importance labels.
    pub fn new(
        weight: &[f32],
        vocab_size: usize,
        embedding_dim: usize,
        importance: &[RowImportance],
    ) -> Result<Self> {
        let expected = vocab_size * embedding_dim;
        if weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "weight length {} != vocab_size({}) * embedding_dim({})",
                    weight.len(),
                    vocab_size,
                    embedding_dim,
                ),
            }
            .into());
        }
        if importance.len() != vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "importance length {} != vocab_size {}",
                    importance.len(),
                    vocab_size,
                ),
            }
            .into());
        }
        if !embedding_dim.is_multiple_of(2) {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "embedding_dim must be even for INT4 support, got {}",
                    embedding_dim,
                ),
            }
            .into());
        }

        let mut int8_rows: Vec<(usize, &[f32])> = Vec::new();
        let mut int4_rows: Vec<(usize, &[f32])> = Vec::new();
        let mut bin_rows: Vec<(usize, &[f32])> = Vec::new();

        for (row, &imp) in importance.iter().enumerate() {
            let slice = &weight[row * embedding_dim..(row + 1) * embedding_dim];
            match imp {
                RowImportance::High => int8_rows.push((row, slice)),
                RowImportance::Medium => int4_rows.push((row, slice)),
                RowImportance::Low => bin_rows.push((row, slice)),
            }
        }

        // Build INT8 data
        let mut int8_data = vec![0u8; int8_rows.len() * embedding_dim];
        let mut int8_scales = vec![0.0f32; int8_rows.len()];
        for (local, &(_, row_slice)) in int8_rows.iter().enumerate() {
            let absmax = row_slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
            int8_scales[local] = scale;
            for (j, &val) in row_slice.iter().enumerate() {
                let q = (val / scale).round().clamp(-127.0, 127.0) as i8;
                int8_data[local * embedding_dim + j] = q as u8;
            }
        }

        // Build INT4 data
        let packed_dim = embedding_dim / 2;
        let mut int4_data = vec![0u8; int4_rows.len() * packed_dim];
        let mut int4_scales = vec![0.0f32; int4_rows.len()];
        for (local, &(_, row_slice)) in int4_rows.iter().enumerate() {
            let absmax = row_slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if absmax > 0.0 { absmax / 7.0 } else { 1.0 };
            int4_scales[local] = scale;
            for j in (0..embedding_dim).step_by(2) {
                let lo = (row_slice[j] / scale).round().clamp(-7.0, 7.0) as i8;
                let hi = (row_slice[j + 1] / scale).round().clamp(-7.0, 7.0) as i8;
                let lo_u = (lo + 8) as u8 & 0x0F;
                let hi_u = ((hi + 8) as u8 & 0x0F) << 4;
                int4_data[local * packed_dim + j / 2] = lo_u | hi_u;
            }
        }

        // Build binary data
        let bytes_per_row = embedding_dim.div_ceil(8);
        let mut binary_data = vec![0u8; bin_rows.len() * bytes_per_row];
        for (local, &(_, row_slice)) in bin_rows.iter().enumerate() {
            for (j, &val) in row_slice.iter().enumerate() {
                if val > 0.0 {
                    binary_data[local * bytes_per_row + j / 8] |= 1 << (j % 8);
                }
            }
        }

        // Build row map
        let mut row_map = vec![(RowImportance::High, 0usize); vocab_size];
        let mut int8_idx = 0usize;
        let mut int4_idx = 0usize;
        let mut bin_idx = 0usize;
        for (row, &imp) in importance.iter().enumerate() {
            match imp {
                RowImportance::High => {
                    row_map[row] = (RowImportance::High, int8_idx);
                    int8_idx += 1;
                }
                RowImportance::Medium => {
                    row_map[row] = (RowImportance::Medium, int4_idx);
                    int4_idx += 1;
                }
                RowImportance::Low => {
                    row_map[row] = (RowImportance::Low, bin_idx);
                    bin_idx += 1;
                }
            }
        }

        Ok(Self {
            importance: importance.to_vec(),
            int8_data,
            int8_scales,
            int4_data,
            int4_scales,
            binary_data,
            row_map,
            vocab_size,
            embedding_dim,
        })
    }

    /// Dequantize a single row according to its assigned precision.
    pub fn dequantize_row(&self, row: usize) -> Result<Vec<f32>> {
        if row >= self.vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!("row {} >= vocab_size {}", row, self.vocab_size),
            }
            .into());
        }

        let (tier, local) = self.row_map[row];
        match tier {
            RowImportance::High => {
                let scale = self.int8_scales[local];
                let offset = local * self.embedding_dim;
                Ok((0..self.embedding_dim)
                    .map(|j| (self.int8_data[offset + j] as i8) as f32 * scale)
                    .collect())
            }
            RowImportance::Medium => {
                let scale = self.int4_scales[local];
                let packed_dim = self.embedding_dim / 2;
                let offset = local * packed_dim;
                let mut out = vec![0.0f32; self.embedding_dim];
                for (j, val) in out.iter_mut().enumerate() {
                    let byte = self.int4_data[offset + j / 2];
                    let nibble = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    *val = (nibble as i8 - 8) as f32 * scale;
                }
                Ok(out)
            }
            RowImportance::Low => {
                let bpr = self.embedding_dim.div_ceil(8);
                let offset = local * bpr;
                let mut out = vec![-1.0f32; self.embedding_dim];
                for (j, val) in out.iter_mut().enumerate() {
                    let byte = self.binary_data[offset + j / 8];
                    if byte & (1 << (j % 8)) != 0 {
                        *val = 1.0;
                    }
                }
                Ok(out)
            }
        }
    }

    /// Total memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.int8_data.len()
            + self.int8_scales.len() * 4
            + self.int4_data.len()
            + self.int4_scales.len() * 4
            + self.binary_data.len()
            + self.row_map.len() * std::mem::size_of::<(RowImportance, usize)>()
    }
}

// ── EmbeddingLookup ──────────────────────────────────────────────

/// Batch embedding lookup with on-the-fly dequantization (CPU reference).
pub struct EmbeddingLookup;

impl EmbeddingLookup {
    /// Look up INT8/INT4 quantized embeddings for a batch of token IDs.
    ///
    /// Out-of-vocabulary tokens (`>= vocab_size`) produce zero vectors.
    pub fn lookup_quantized(
        table: &QuantizedEmbeddingTable,
        token_ids: &[u32],
        output: &mut [f32],
    ) -> Result<()> {
        let seq_len = token_ids.len();
        let dim = table.embedding_dim;
        if output.len() < seq_len * dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output length {} < seq_len({}) * embedding_dim({})",
                    output.len(),
                    seq_len,
                    dim,
                ),
            }
            .into());
        }

        for (t, &tok) in token_ids.iter().enumerate() {
            let tid = tok as usize;
            let out_start = t * dim;
            if tid >= table.vocab_size {
                output[out_start..out_start + dim].fill(0.0);
            } else {
                let row = table.dequantize_row(tid)?;
                output[out_start..out_start + dim].copy_from_slice(&row);
            }
        }
        Ok(())
    }

    /// Look up PQ-encoded embeddings for a batch of token IDs.
    pub fn lookup_pq(pq: &ProductQuantizer, token_ids: &[u32], output: &mut [f32]) -> Result<()> {
        let seq_len = token_ids.len();
        let dim = pq.embedding_dim();
        if output.len() < seq_len * dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output length {} < seq_len({}) * embedding_dim({})",
                    output.len(),
                    seq_len,
                    dim,
                ),
            }
            .into());
        }

        for (t, &tok) in token_ids.iter().enumerate() {
            let tid = tok as usize;
            let out_start = t * dim;
            if tid >= pq.vocab_size {
                output[out_start..out_start + dim].fill(0.0);
            } else {
                let row = pq.decode_row(tid)?;
                output[out_start..out_start + dim].copy_from_slice(&row);
            }
        }
        Ok(())
    }

    /// Look up adaptive-quantized embeddings for a batch of token IDs.
    pub fn lookup_adaptive(
        aq: &AdaptiveQuant,
        token_ids: &[u32],
        output: &mut [f32],
    ) -> Result<()> {
        let seq_len = token_ids.len();
        let dim = aq.embedding_dim;
        if output.len() < seq_len * dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output length {} < seq_len({}) * embedding_dim({})",
                    output.len(),
                    seq_len,
                    dim,
                ),
            }
            .into());
        }

        for (t, &tok) in token_ids.iter().enumerate() {
            let tid = tok as usize;
            let out_start = t * dim;
            if tid >= aq.vocab_size {
                output[out_start..out_start + dim].fill(0.0);
            } else {
                let row = aq.dequantize_row(tid)?;
                output[out_start..out_start + dim].copy_from_slice(&row);
            }
        }
        Ok(())
    }
}

// ── CompressionStats ─────────────────────────────────────────────

/// Compression and quality metrics for quantized embeddings.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original FP32 size in bytes.
    pub original_bytes: usize,
    /// Compressed size in bytes.
    pub compressed_bytes: usize,
    /// Compression ratio (original / compressed).
    pub compression_ratio: f32,
    /// Mean cosine similarity of quantized vs original rows.
    pub mean_cosine_similarity: f32,
    /// Minimum cosine similarity across all rows.
    pub min_cosine_similarity: f32,
    /// Mean squared error (averaged over all elements).
    pub mean_squared_error: f32,
}

impl CompressionStats {
    /// Compute stats by comparing a quantized table to the original FP32 weights.
    pub fn compute_quantized(table: &QuantizedEmbeddingTable, original: &[f32]) -> Result<Self> {
        let vocab_size = table.vocab_size;
        let dim = table.embedding_dim;
        if original.len() != vocab_size * dim {
            return Err(KernelError::InvalidArguments {
                reason: "original weight size mismatch".to_string(),
            }
            .into());
        }

        let mut total_cos = 0.0f64;
        let mut min_cos = f64::MAX;
        let mut total_mse = 0.0f64;

        for row in 0..vocab_size {
            let orig = &original[row * dim..(row + 1) * dim];
            let deq = table.dequantize_row(row)?;
            let cos = cosine_similarity(orig, &deq);
            total_cos += cos;
            if cos < min_cos {
                min_cos = cos;
            }
            let mse: f64 = orig
                .iter()
                .zip(deq.iter())
                .map(|(&a, &b)| ((a - b) as f64) * ((a - b) as f64))
                .sum::<f64>()
                / dim as f64;
            total_mse += mse;
        }

        let mean_cos = total_cos / vocab_size as f64;
        let mean_mse = total_mse / vocab_size as f64;

        Ok(Self {
            original_bytes: vocab_size * dim * 4,
            compressed_bytes: table.memory_bytes(),
            compression_ratio: (vocab_size * dim * 4) as f32 / table.memory_bytes() as f32,
            mean_cosine_similarity: mean_cos as f32,
            min_cosine_similarity: min_cos as f32,
            mean_squared_error: mean_mse as f32,
        })
    }
}

impl fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompressionStats {{ ratio: {:.2}x, mean_cos: {:.4}, min_cos: {:.4}, mse: {:.6} }}",
            self.compression_ratio,
            self.mean_cosine_similarity,
            self.min_cosine_similarity,
            self.mean_squared_error,
        )
    }
}

// ── CodebookTrainer ──────────────────────────────────────────────

/// Trains PQ codebooks from embedding data using k-means.
pub struct CodebookTrainer {
    /// Number of sub-vector partitions.
    pub num_subvectors: usize,
    /// Number of centroids per sub-vector.
    pub num_centroids: usize,
    /// Sub-vector dimension.
    pub sub_dim: usize,
    /// Maximum k-means iterations.
    pub max_iters: usize,
}

impl CodebookTrainer {
    /// Create a new trainer.
    pub fn new(
        num_subvectors: usize,
        num_centroids: usize,
        sub_dim: usize,
        max_iters: usize,
    ) -> Result<Self> {
        if num_centroids == 0 || num_centroids > 256 {
            return Err(KernelError::InvalidArguments {
                reason: format!("num_centroids must be 1..=256, got {}", num_centroids),
            }
            .into());
        }
        if num_subvectors == 0 || sub_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_subvectors and sub_dim must be > 0".to_string(),
            }
            .into());
        }
        Ok(Self { num_subvectors, num_centroids, sub_dim, max_iters })
    }

    /// Train codebooks and encode all rows.
    ///
    /// `weight` is `[vocab_size, embedding_dim]` with `embedding_dim = num_subvectors * sub_dim`.
    pub fn train(&self, weight: &[f32], vocab_size: usize) -> Result<ProductQuantizer> {
        let embedding_dim = self.num_subvectors * self.sub_dim;
        if weight.len() != vocab_size * embedding_dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "weight length {} != vocab_size({}) * embedding_dim({})",
                    weight.len(),
                    vocab_size,
                    embedding_dim,
                ),
            }
            .into());
        }

        let mut codebook = vec![0.0f32; self.num_subvectors * self.num_centroids * self.sub_dim];
        let mut codes = vec![0u8; vocab_size * self.num_subvectors];

        for sv in 0..self.num_subvectors {
            // Extract sub-vectors for this partition
            let mut sub_vecs: Vec<Vec<f32>> = Vec::with_capacity(vocab_size);
            for row in 0..vocab_size {
                let start = row * embedding_dim + sv * self.sub_dim;
                sub_vecs.push(weight[start..start + self.sub_dim].to_vec());
            }

            // Initialize centroids from first `num_centroids` data points (or cycle)
            let mut centroids = vec![vec![0.0f32; self.sub_dim]; self.num_centroids];
            for (c, centroid) in centroids.iter_mut().enumerate() {
                let src = &sub_vecs[c % vocab_size];
                centroid.copy_from_slice(src);
            }

            // K-means iterations
            let mut assignments = vec![0usize; vocab_size];
            for _iter in 0..self.max_iters {
                // Assign each sub-vector to nearest centroid
                let mut changed = false;
                for (i, sv_data) in sub_vecs.iter().enumerate() {
                    let mut best_c = 0;
                    let mut best_d = f32::MAX;
                    for (c, centroid) in centroids.iter().enumerate() {
                        let d: f32 =
                            sv_data.iter().zip(centroid).map(|(a, b)| (a - b) * (a - b)).sum();
                        if d < best_d {
                            best_d = d;
                            best_c = c;
                        }
                    }
                    if assignments[i] != best_c {
                        assignments[i] = best_c;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }

                // Update centroids
                let mut counts = vec![0usize; self.num_centroids];
                let mut sums = vec![vec![0.0f32; self.sub_dim]; self.num_centroids];
                for (i, &a) in assignments.iter().enumerate() {
                    counts[a] += 1;
                    for (d, &val) in sub_vecs[i].iter().enumerate() {
                        sums[a][d] += val;
                    }
                }
                for (c, centroid) in centroids.iter_mut().enumerate() {
                    if counts[c] > 0 {
                        for (d, val) in centroid.iter_mut().enumerate() {
                            *val = sums[c][d] / counts[c] as f32;
                        }
                    }
                }
            }

            // Write codebook
            for (c, centroid) in centroids.iter().enumerate() {
                let cb_offset = (sv * self.num_centroids + c) * self.sub_dim;
                codebook[cb_offset..cb_offset + self.sub_dim].copy_from_slice(centroid);
            }

            // Write codes
            for (i, &a) in assignments.iter().enumerate() {
                codes[i * self.num_subvectors + sv] = a as u8;
            }
        }

        ProductQuantizer::new(
            codebook,
            codes,
            self.num_subvectors,
            self.num_centroids,
            self.sub_dim,
            vocab_size,
        )
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Cosine similarity between two f32 slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: generate deterministic embedding weights
    fn make_weight(vocab_size: usize, dim: usize) -> Vec<f32> {
        (0..vocab_size * dim).map(|i| ((i as f32) * 0.01 - 0.5).sin()).collect()
    }

    fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
        cosine_similarity(a, b) as f32
    }

    // ── OpenCL kernel source ─────────────────────────────────

    #[test]
    fn opencl_source_not_empty() {
        assert!(!EMBED_QUANT_CL.is_empty());
    }

    #[test]
    fn opencl_source_has_int8_kernel() {
        assert!(EMBED_QUANT_CL.contains("embedding_lookup_int8"));
    }

    #[test]
    fn opencl_source_has_int4_kernel() {
        assert!(EMBED_QUANT_CL.contains("embedding_lookup_int4"));
    }

    #[test]
    fn opencl_source_has_pq_kernel() {
        assert!(EMBED_QUANT_CL.contains("embedding_lookup_pq"));
    }

    #[test]
    fn opencl_source_has_kernel_keyword() {
        assert!(EMBED_QUANT_CL.contains("__kernel"));
    }

    // ── QuantPrecision ───────────────────────────────────────

    #[test]
    fn quant_precision_display() {
        assert_eq!(format!("{}", QuantPrecision::Int8), "INT8");
        assert_eq!(format!("{}", QuantPrecision::Int4), "INT4");
    }

    // ── QuantizedEmbeddingTable INT8 ─────────────────────────

    #[test]
    fn int8_roundtrip_quality() {
        let weight = make_weight(32, 64);
        let table =
            QuantizedEmbeddingTable::quantize(&weight, 32, 64, QuantPrecision::Int8).unwrap();
        for row in 0..32 {
            let orig = &weight[row * 64..(row + 1) * 64];
            let deq = table.dequantize_row(row).unwrap();
            let cs = cos_sim(orig, &deq);
            assert!(cs > 0.99, "INT8 cosine {cs} for row {row}");
        }
    }

    #[test]
    fn int8_quantize_rejects_wrong_size() {
        assert!(QuantizedEmbeddingTable::quantize(&[0.0; 5], 2, 4, QuantPrecision::Int8).is_err());
    }

    #[test]
    fn int8_dequant_oov_row() {
        let table =
            QuantizedEmbeddingTable::quantize(&[1.0; 8], 2, 4, QuantPrecision::Int8).unwrap();
        assert!(table.dequantize_row(2).is_err());
    }

    #[test]
    fn int8_zero_row() {
        let weight = vec![0.0f32; 8];
        let table = QuantizedEmbeddingTable::quantize(&weight, 2, 4, QuantPrecision::Int8).unwrap();
        let deq = table.dequantize_row(0).unwrap();
        assert!(deq.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn int8_single_element() {
        let weight = vec![42.0f32];
        let table = QuantizedEmbeddingTable::quantize(&weight, 1, 1, QuantPrecision::Int8).unwrap();
        let deq = table.dequantize_row(0).unwrap();
        assert!((deq[0] - 42.0).abs() < 0.5);
    }

    #[test]
    fn int8_negative_values() {
        let weight = vec![-1.0, -0.5, 0.0, 0.5];
        let table = QuantizedEmbeddingTable::quantize(&weight, 1, 4, QuantPrecision::Int8).unwrap();
        let deq = table.dequantize_row(0).unwrap();
        assert!(cos_sim(&weight, &deq) > 0.99);
    }

    #[test]
    fn int8_memory_is_smaller() {
        let table = QuantizedEmbeddingTable::quantize(
            &make_weight(100, 128),
            100,
            128,
            QuantPrecision::Int8,
        )
        .unwrap();
        assert!(table.memory_bytes() < table.fp32_memory_bytes());
    }

    #[test]
    fn int8_compression_ratio() {
        let table = QuantizedEmbeddingTable::quantize(
            &make_weight(100, 128),
            100,
            128,
            QuantPrecision::Int8,
        )
        .unwrap();
        let ratio = table.fp32_memory_bytes() as f32 / table.memory_bytes() as f32;
        // INT8 should achieve ~4x (minus scale overhead)
        assert!(ratio > 3.5, "INT8 ratio {ratio}");
    }

    // ── QuantizedEmbeddingTable INT4 ─────────────────────────

    #[test]
    fn int4_roundtrip_quality() {
        let weight = make_weight(32, 64);
        let table =
            QuantizedEmbeddingTable::quantize(&weight, 32, 64, QuantPrecision::Int4).unwrap();
        for row in 0..32 {
            let orig = &weight[row * 64..(row + 1) * 64];
            let deq = table.dequantize_row(row).unwrap();
            let cs = cos_sim(orig, &deq);
            assert!(cs > 0.95, "INT4 cosine {cs} for row {row}");
        }
    }

    #[test]
    fn int4_rejects_odd_dim() {
        assert!(QuantizedEmbeddingTable::quantize(&[0.0; 3], 1, 3, QuantPrecision::Int4).is_err());
    }

    #[test]
    fn int4_zero_row() {
        let weight = vec![0.0f32; 8];
        let table = QuantizedEmbeddingTable::quantize(&weight, 2, 4, QuantPrecision::Int4).unwrap();
        let deq = table.dequantize_row(0).unwrap();
        assert!(deq.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn int4_memory_is_smaller_than_int8() {
        let weight = make_weight(100, 128);
        let t8 =
            QuantizedEmbeddingTable::quantize(&weight, 100, 128, QuantPrecision::Int8).unwrap();
        let t4 =
            QuantizedEmbeddingTable::quantize(&weight, 100, 128, QuantPrecision::Int4).unwrap();
        assert!(t4.memory_bytes() < t8.memory_bytes());
    }

    #[test]
    fn int4_compression_ratio() {
        let table = QuantizedEmbeddingTable::quantize(
            &make_weight(100, 128),
            100,
            128,
            QuantPrecision::Int4,
        )
        .unwrap();
        let ratio = table.fp32_memory_bytes() as f32 / table.memory_bytes() as f32;
        assert!(ratio > 7.0, "INT4 ratio {ratio}");
    }

    #[test]
    fn int4_negative_values() {
        let weight = vec![-1.0, -0.5, 0.0, 0.5];
        let table = QuantizedEmbeddingTable::quantize(&weight, 1, 4, QuantPrecision::Int4).unwrap();
        let deq = table.dequantize_row(0).unwrap();
        assert!(cos_sim(&weight, &deq) > 0.90);
    }

    // ── ProductQuantizer ─────────────────────────────────────

    #[test]
    fn pq_decode_roundtrip() {
        // 2 sub-vectors, 4 centroids, sub_dim=2, vocab=3
        let codebook = vec![
            // sv0: 4 centroids of dim 2
            1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, // sv1: 4 centroids of dim 2
            0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5,
        ];
        let codes = vec![
            0, 1, // row 0: sv0=centroid0, sv1=centroid1
            2, 0, // row 1
            1, 3, // row 2
        ];
        let pq = ProductQuantizer::new(codebook.clone(), codes, 2, 4, 2, 3).unwrap();

        let row0 = pq.decode_row(0).unwrap();
        assert_eq!(row0, vec![1.0, 0.0, -0.5, 0.5]);

        let row1 = pq.decode_row(1).unwrap();
        assert_eq!(row1, vec![-1.0, 0.0, 0.5, 0.5]);

        let row2 = pq.decode_row(2).unwrap();
        assert_eq!(row2, vec![0.0, 1.0, -0.5, -0.5]);
    }

    #[test]
    fn pq_encode_nearest_centroid() {
        let codebook = vec![
            1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5,
        ];
        let codes = vec![0, 0]; // placeholder
        let pq = ProductQuantizer::new(codebook, codes, 2, 4, 2, 1).unwrap();

        // Vector close to centroid 0 of sv0, centroid 2 of sv1
        let vec = vec![0.9, 0.1, 0.6, -0.4];
        let encoded = pq.encode_vector(&vec).unwrap();
        assert_eq!(encoded[0], 0); // nearest to [1,0]
        assert_eq!(encoded[1], 2); // nearest to [0.5,-0.5]
    }

    #[test]
    fn pq_rejects_bad_codebook_size() {
        assert!(ProductQuantizer::new(vec![0.0; 5], vec![0; 2], 2, 4, 2, 1).is_err());
    }

    #[test]
    fn pq_rejects_bad_codes_size() {
        assert!(ProductQuantizer::new(vec![0.0; 16], vec![0; 3], 2, 4, 2, 1).is_err());
    }

    #[test]
    fn pq_rejects_too_many_centroids() {
        assert!(ProductQuantizer::new(vec![0.0; 257], vec![0; 1], 1, 257, 1, 1).is_err());
    }

    #[test]
    fn pq_oov_row() {
        let pq = ProductQuantizer::new(vec![0.0; 8], vec![0; 2], 2, 2, 2, 1).unwrap();
        assert!(pq.decode_row(1).is_err());
    }

    #[test]
    fn pq_encode_wrong_dim() {
        let pq = ProductQuantizer::new(vec![0.0; 8], vec![0; 2], 2, 2, 2, 1).unwrap();
        assert!(pq.encode_vector(&[1.0]).is_err());
    }

    #[test]
    fn pq_memory_bytes() {
        let pq = ProductQuantizer::new(vec![0.0; 16], vec![0; 6], 2, 4, 2, 3).unwrap();
        assert_eq!(pq.memory_bytes(), 16 * 4 + 6);
    }

    #[test]
    fn pq_embedding_dim() {
        let pq = ProductQuantizer::new(vec![0.0; 16], vec![0; 2], 2, 4, 2, 1).unwrap();
        assert_eq!(pq.embedding_dim(), 4);
    }

    // ── BinaryEmbedding ──────────────────────────────────────

    #[test]
    fn binary_from_float_basic() {
        let weight = vec![1.0, -1.0, 0.5, -0.5]; // vocab=1, dim=4
        let be = BinaryEmbedding::from_float(&weight, 1, 4).unwrap();
        let row = be.get_row(0).unwrap();
        // bit 0 = 1 (1.0>0), bit 1 = 0 (-1.0), bit 2 = 1 (0.5>0), bit 3 = 0 (-0.5)
        assert_eq!(row[0], 0b0000_0101);
    }

    #[test]
    fn binary_unpack_signed() {
        let weight = vec![1.0, -1.0, 0.5, -0.5];
        let be = BinaryEmbedding::from_float(&weight, 1, 4).unwrap();
        let unpacked = be.unpack_row_signed(0).unwrap();
        assert_eq!(unpacked, vec![1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn binary_hamming_same_row() {
        let weight = vec![1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
        let be = BinaryEmbedding::from_float(&weight, 2, 4).unwrap();
        assert_eq!(be.hamming_distance(0, 0).unwrap(), 0);
    }

    #[test]
    fn binary_hamming_opposite_rows() {
        let weight = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        let be = BinaryEmbedding::from_float(&weight, 2, 4).unwrap();
        assert_eq!(be.hamming_distance(0, 1).unwrap(), 4);
    }

    #[test]
    fn binary_rejects_wrong_size() {
        assert!(BinaryEmbedding::from_float(&[0.0; 5], 2, 4).is_err());
    }

    #[test]
    fn binary_oov_row() {
        let be = BinaryEmbedding::from_float(&[1.0; 4], 1, 4).unwrap();
        assert!(be.get_row(1).is_err());
    }

    #[test]
    fn binary_memory_savings() {
        let be = BinaryEmbedding::from_float(&make_weight(100, 128), 100, 128).unwrap();
        let fp32_bytes = 100 * 128 * 4;
        assert!(be.memory_bytes() < fp32_bytes / 20);
    }

    #[test]
    fn binary_non_multiple_of_8_dim() {
        let weight = vec![1.0, -1.0, 0.5]; // dim=3
        let be = BinaryEmbedding::from_float(&weight, 1, 3).unwrap();
        let unpacked = be.unpack_row_signed(0).unwrap();
        assert_eq!(unpacked, vec![1.0, -1.0, 1.0]);
    }

    #[test]
    fn binary_all_positive() {
        let weight = vec![1.0; 8]; // 2 rows of dim 4
        let be = BinaryEmbedding::from_float(&weight, 2, 4).unwrap();
        assert_eq!(be.hamming_distance(0, 1).unwrap(), 0);
    }

    // ── AdaptiveQuant ────────────────────────────────────────

    #[test]
    fn adaptive_high_uses_int8_quality() {
        let weight = make_weight(4, 16);
        let importance = vec![RowImportance::High; 4];
        let aq = AdaptiveQuant::new(&weight, 4, 16, &importance).unwrap();
        for row in 0..4 {
            let orig = &weight[row * 16..(row + 1) * 16];
            let deq = aq.dequantize_row(row).unwrap();
            assert!(cos_sim(orig, &deq) > 0.99);
        }
    }

    #[test]
    fn adaptive_medium_uses_int4_quality() {
        let weight = make_weight(4, 16);
        let importance = vec![RowImportance::Medium; 4];
        let aq = AdaptiveQuant::new(&weight, 4, 16, &importance).unwrap();
        for row in 0..4 {
            let orig = &weight[row * 16..(row + 1) * 16];
            let deq = aq.dequantize_row(row).unwrap();
            assert!(cos_sim(orig, &deq) > 0.90);
        }
    }

    #[test]
    fn adaptive_low_uses_binary() {
        let weight = make_weight(4, 16);
        let importance = vec![RowImportance::Low; 4];
        let aq = AdaptiveQuant::new(&weight, 4, 16, &importance).unwrap();
        let deq = aq.dequantize_row(0).unwrap();
        // Binary: all values should be -1.0 or 1.0
        assert!(deq.iter().all(|&v| v == -1.0 || v == 1.0));
    }

    #[test]
    fn adaptive_mixed_precision() {
        let weight = make_weight(3, 16);
        let importance = vec![RowImportance::High, RowImportance::Medium, RowImportance::Low];
        let aq = AdaptiveQuant::new(&weight, 3, 16, &importance).unwrap();
        // Each row should be valid
        for row in 0..3 {
            let deq = aq.dequantize_row(row).unwrap();
            assert_eq!(deq.len(), 16);
        }
    }

    #[test]
    fn adaptive_rejects_wrong_size() {
        let imp = vec![RowImportance::High; 2];
        assert!(AdaptiveQuant::new(&[0.0; 5], 2, 4, &imp).is_err());
    }

    #[test]
    fn adaptive_rejects_wrong_importance_len() {
        assert!(AdaptiveQuant::new(&[0.0; 8], 2, 4, &[RowImportance::High]).is_err());
    }

    #[test]
    fn adaptive_oov_row() {
        let aq = AdaptiveQuant::new(&[0.0; 8], 2, 4, &[RowImportance::High; 2]).unwrap();
        assert!(aq.dequantize_row(2).is_err());
    }

    // ── EmbeddingLookup ──────────────────────────────────────

    #[test]
    fn lookup_quantized_basic() {
        let weight = make_weight(8, 16);
        let table =
            QuantizedEmbeddingTable::quantize(&weight, 8, 16, QuantPrecision::Int8).unwrap();
        let mut out = vec![0.0; 32]; // 2 tokens
        EmbeddingLookup::lookup_quantized(&table, &[3, 5], &mut out).unwrap();
        // Check row 3
        let orig = &weight[3 * 16..4 * 16];
        assert!(cos_sim(orig, &out[0..16]) > 0.99);
    }

    #[test]
    fn lookup_quantized_oov_zeroed() {
        let table =
            QuantizedEmbeddingTable::quantize(&make_weight(4, 8), 4, 8, QuantPrecision::Int8)
                .unwrap();
        let mut out = vec![99.0; 8];
        EmbeddingLookup::lookup_quantized(&table, &[100], &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn lookup_quantized_rejects_short_output() {
        let table =
            QuantizedEmbeddingTable::quantize(&make_weight(4, 8), 4, 8, QuantPrecision::Int8)
                .unwrap();
        let mut out = vec![0.0; 4]; // too short for dim=8
        assert!(EmbeddingLookup::lookup_quantized(&table, &[0], &mut out).is_err());
    }

    #[test]
    fn lookup_pq_basic() {
        let codebook = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, 0.5];
        let codes = vec![0, 0, 1, 1];
        let pq = ProductQuantizer::new(codebook, codes, 2, 2, 2, 2).unwrap();
        let mut out = vec![0.0; 8]; // 2 tokens × dim 4
        EmbeddingLookup::lookup_pq(&pq, &[0, 1], &mut out).unwrap();
        assert_eq!(&out[0..4], &[1.0, 0.0, 0.5, 0.5]);
        assert_eq!(&out[4..8], &[0.0, 1.0, -0.5, 0.5]);
    }

    #[test]
    fn lookup_pq_oov_zeroed() {
        let pq = ProductQuantizer::new(vec![0.0; 4], vec![0; 1], 1, 2, 2, 1).unwrap();
        let mut out = vec![99.0; 2];
        EmbeddingLookup::lookup_pq(&pq, &[5], &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn lookup_adaptive_basic() {
        let weight = make_weight(4, 16);
        let importance = vec![RowImportance::High; 4];
        let aq = AdaptiveQuant::new(&weight, 4, 16, &importance).unwrap();
        let mut out = vec![0.0; 16];
        EmbeddingLookup::lookup_adaptive(&aq, &[2], &mut out).unwrap();
        let orig = &weight[2 * 16..3 * 16];
        assert!(cos_sim(orig, &out) > 0.99);
    }

    #[test]
    fn lookup_adaptive_oov_zeroed() {
        let aq = AdaptiveQuant::new(&[0.0; 8], 2, 4, &[RowImportance::High; 2]).unwrap();
        let mut out = vec![99.0; 4];
        EmbeddingLookup::lookup_adaptive(&aq, &[10], &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn lookup_empty_batch() {
        let table =
            QuantizedEmbeddingTable::quantize(&make_weight(4, 8), 4, 8, QuantPrecision::Int8)
                .unwrap();
        let mut out = vec![];
        EmbeddingLookup::lookup_quantized(&table, &[], &mut out).unwrap();
    }

    // ── CompressionStats ─────────────────────────────────────

    #[test]
    fn compression_stats_int8() {
        let weight = make_weight(32, 64);
        let table =
            QuantizedEmbeddingTable::quantize(&weight, 32, 64, QuantPrecision::Int8).unwrap();
        let stats = CompressionStats::compute_quantized(&table, &weight).unwrap();
        assert!(stats.mean_cosine_similarity > 0.99);
        assert!(stats.compression_ratio > 3.5);
        assert!(stats.mean_squared_error < 0.01);
    }

    #[test]
    fn compression_stats_int4() {
        let weight = make_weight(32, 64);
        let table =
            QuantizedEmbeddingTable::quantize(&weight, 32, 64, QuantPrecision::Int4).unwrap();
        let stats = CompressionStats::compute_quantized(&table, &weight).unwrap();
        assert!(stats.mean_cosine_similarity > 0.95);
        assert!(stats.compression_ratio > 7.0);
    }

    #[test]
    fn compression_stats_display() {
        let weight = make_weight(4, 16);
        let table =
            QuantizedEmbeddingTable::quantize(&weight, 4, 16, QuantPrecision::Int8).unwrap();
        let stats = CompressionStats::compute_quantized(&table, &weight).unwrap();
        let s = format!("{}", stats);
        assert!(s.contains("ratio"));
        assert!(s.contains("mean_cos"));
    }

    #[test]
    fn compression_stats_rejects_mismatch() {
        let table =
            QuantizedEmbeddingTable::quantize(&[0.0; 8], 2, 4, QuantPrecision::Int8).unwrap();
        assert!(CompressionStats::compute_quantized(&table, &[0.0; 5]).is_err());
    }

    // ── CodebookTrainer ──────────────────────────────────────

    #[test]
    fn codebook_trainer_basic() {
        let weight = make_weight(8, 4); // 8 rows, dim=4, 2 subvectors of dim 2
        let trainer = CodebookTrainer::new(2, 4, 2, 10).unwrap();
        let pq = trainer.train(&weight, 8).unwrap();
        assert_eq!(pq.vocab_size, 8);
        assert_eq!(pq.embedding_dim(), 4);
        // Decode every row and check finite
        for row in 0..8 {
            let decoded = pq.decode_row(row).unwrap();
            assert!(decoded.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn codebook_trainer_reconstruction() {
        let weight = make_weight(16, 8); // 16 rows, dim=8, 4 sv of dim 2
        let trainer = CodebookTrainer::new(4, 8, 2, 20).unwrap();
        let pq = trainer.train(&weight, 16).unwrap();
        // PQ reconstruction should have some resemblance (not perfect)
        let mut total_cos = 0.0f32;
        for row in 0..16 {
            let orig = &weight[row * 8..(row + 1) * 8];
            let decoded = pq.decode_row(row).unwrap();
            total_cos += cos_sim(orig, &decoded);
        }
        let mean_cos = total_cos / 16.0;
        assert!(mean_cos > 0.5, "PQ mean cosine {mean_cos}");
    }

    #[test]
    fn codebook_trainer_rejects_zero_centroids() {
        assert!(CodebookTrainer::new(2, 0, 2, 10).is_err());
    }

    #[test]
    fn codebook_trainer_rejects_too_many_centroids() {
        assert!(CodebookTrainer::new(2, 257, 2, 10).is_err());
    }

    #[test]
    fn codebook_trainer_rejects_weight_mismatch() {
        let trainer = CodebookTrainer::new(2, 4, 2, 10).unwrap();
        assert!(trainer.train(&[0.0; 5], 2).is_err());
    }

    // ── Edge cases ───────────────────────────────────────────

    #[test]
    fn single_embedding_int8() {
        let weight = vec![3.14f32];
        let table = QuantizedEmbeddingTable::quantize(&weight, 1, 1, QuantPrecision::Int8).unwrap();
        let deq = table.dequantize_row(0).unwrap();
        assert!((deq[0] - 3.14).abs() < 0.05);
    }

    #[test]
    fn large_vocab_int8() {
        let weight = make_weight(1000, 32);
        let table =
            QuantizedEmbeddingTable::quantize(&weight, 1000, 32, QuantPrecision::Int8).unwrap();
        // Spot-check a few rows
        for row in [0, 499, 999] {
            let orig = &weight[row * 32..(row + 1) * 32];
            let deq = table.dequantize_row(row).unwrap();
            assert!(cos_sim(orig, &deq) > 0.99);
        }
    }

    #[test]
    fn uniform_weight_int4() {
        // All same value should quantize/dequantize without NaN
        let weight = vec![0.42f32; 64]; // 4 rows × dim 16
        let table =
            QuantizedEmbeddingTable::quantize(&weight, 4, 16, QuantPrecision::Int4).unwrap();
        for row in 0..4 {
            let deq = table.dequantize_row(row).unwrap();
            assert!(deq.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn pq_single_centroid() {
        // With only 1 centroid, everything maps to it
        let codebook = vec![1.0, 2.0]; // 1 sv, 1 centroid, dim=2
        let codes = vec![0, 0, 0]; // 3 rows
        let pq = ProductQuantizer::new(codebook, codes, 1, 1, 2, 3).unwrap();
        for row in 0..3 {
            let decoded = pq.decode_row(row).unwrap();
            assert_eq!(decoded, vec![1.0, 2.0]);
        }
    }

    // ── Property-like tests ──────────────────────────────────

    #[test]
    fn int8_preserves_relative_ordering() {
        // If embedding[a] has larger L2 norm than embedding[b] in FP32,
        // the quantized version should usually preserve that ordering.
        let weight = vec![
            0.1, 0.1, 0.1, 0.1, // row 0: small norm
            1.0, 1.0, 1.0, 1.0, // row 1: large norm
        ];
        let table = QuantizedEmbeddingTable::quantize(&weight, 2, 4, QuantPrecision::Int8).unwrap();
        let d0 = table.dequantize_row(0).unwrap();
        let d1 = table.dequantize_row(1).unwrap();
        let norm0: f32 = d0.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm1: f32 = d1.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm1 > norm0, "norm ordering preserved");
    }

    #[test]
    fn int4_preserves_relative_ordering() {
        let weight = vec![0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0];
        let table = QuantizedEmbeddingTable::quantize(&weight, 2, 4, QuantPrecision::Int4).unwrap();
        let d0 = table.dequantize_row(0).unwrap();
        let d1 = table.dequantize_row(1).unwrap();
        let norm0: f32 = d0.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm1: f32 = d1.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm1 > norm0, "norm ordering preserved");
    }

    #[test]
    fn quantized_lookup_batch_ordering() {
        // Batch lookup should return same results as individual lookups
        let weight = make_weight(8, 16);
        let table =
            QuantizedEmbeddingTable::quantize(&weight, 8, 16, QuantPrecision::Int8).unwrap();
        let ids = vec![5u32, 2, 7, 0];
        let mut batch_out = vec![0.0; 4 * 16];
        EmbeddingLookup::lookup_quantized(&table, &ids, &mut batch_out).unwrap();

        for (t, &id) in ids.iter().enumerate() {
            let mut single_out = vec![0.0; 16];
            EmbeddingLookup::lookup_quantized(&table, &[id], &mut single_out).unwrap();
            assert_eq!(&batch_out[t * 16..(t + 1) * 16], &single_out[..]);
        }
    }

    #[test]
    fn cosine_similarity_self_is_one() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let cs = cosine_similarity(&v, &v);
        assert!((cs - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_orthogonal_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let cs = cosine_similarity(&a, &b);
        assert!(cs.abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_opposite_is_neg_one() {
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        let cs = cosine_similarity(&a, &b);
        assert!((cs + 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        let cs = cosine_similarity(&a, &b);
        assert_eq!(cs, 0.0);
    }

    #[test]
    fn binary_roundtrip_preserves_sign() {
        let weight = make_weight(10, 32);
        let be = BinaryEmbedding::from_float(&weight, 10, 32).unwrap();
        for row in 0..10 {
            let unpacked = be.unpack_row_signed(row).unwrap();
            for (j, &val) in unpacked.iter().enumerate() {
                let orig = weight[row * 32 + j];
                if orig > 0.0 {
                    assert_eq!(val, 1.0, "row {row} dim {j}");
                } else {
                    assert_eq!(val, -1.0, "row {row} dim {j}");
                }
            }
        }
    }

    #[test]
    fn adaptive_memory_less_than_fp32() {
        let weight = make_weight(100, 64);
        let importance: Vec<RowImportance> = (0..100)
            .map(|i| match i % 3 {
                0 => RowImportance::High,
                1 => RowImportance::Medium,
                _ => RowImportance::Low,
            })
            .collect();
        let aq = AdaptiveQuant::new(&weight, 100, 64, &importance).unwrap();
        let fp32_bytes = 100 * 64 * 4;
        assert!(aq.memory_bytes() < fp32_bytes);
    }

    #[test]
    fn int8_all_rows_finite() {
        let weight = make_weight(50, 32);
        let table =
            QuantizedEmbeddingTable::quantize(&weight, 50, 32, QuantPrecision::Int8).unwrap();
        for row in 0..50 {
            let deq = table.dequantize_row(row).unwrap();
            assert!(deq.iter().all(|v| v.is_finite()), "row {row} has non-finite");
        }
    }

    #[test]
    fn int4_all_rows_finite() {
        let weight = make_weight(50, 32);
        let table =
            QuantizedEmbeddingTable::quantize(&weight, 50, 32, QuantPrecision::Int4).unwrap();
        for row in 0..50 {
            let deq = table.dequantize_row(row).unwrap();
            assert!(deq.iter().all(|v| v.is_finite()), "row {row} has non-finite");
        }
    }

    #[test]
    fn lookup_repeated_token_same_result() {
        let table =
            QuantizedEmbeddingTable::quantize(&make_weight(8, 16), 8, 16, QuantPrecision::Int8)
                .unwrap();
        let mut out = vec![0.0; 48]; // 3 tokens
        EmbeddingLookup::lookup_quantized(&table, &[3, 5, 3], &mut out).unwrap();
        assert_eq!(&out[0..16], &out[32..48]);
    }
}
