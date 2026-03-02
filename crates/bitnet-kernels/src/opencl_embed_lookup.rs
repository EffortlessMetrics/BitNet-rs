//! Optimized embedding lookup for Intel Arc A770 GPUs.
//!
//! Provides coalesced-memory-access embedding gather with support for
//! multiple storage formats (F32, F16, I8 quantized, sparse) and
//! optional position / segment embeddings.
//!
//! CPU reference implementations mirror the OpenCL kernel behaviour
//! for correctness testing without hardware.
#![allow(clippy::needless_range_loop)]

use std::fmt;
use std::time::Instant;

// ── OpenCL kernel source ─────────────────────────────────────────

/// OpenCL kernel source for coalesced embedding gather.
pub const EMBED_LOOKUP_CL: &str = r#"
// ────────────────────────────────────────────────────────────────
// embed_lookup.cl — coalesced embedding gather for Intel Arc A770
// ────────────────────────────────────────────────────────────────

/// Vectorised embedding lookup: each work-item copies one f32 element.
/// Grid: (embed_dim, seq_len)  i.e. global_id(0)=dim, global_id(1)=token.
__kernel void embed_gather_coalesced(
    __global const float* restrict weight,   // [vocab_size, embed_dim]
    __global const uint*  restrict token_ids, // [seq_len]
    __global       float* restrict output,    // [seq_len, embed_dim]
    const uint embed_dim,
    const uint vocab_size,
    const int  padding_idx                    // -1 if none
) {
    const uint d = get_global_id(0);
    const uint t = get_global_id(1);
    if (d >= embed_dim) return;

    const uint tok = token_ids[t];
    const uint out_idx = t * embed_dim + d;

    if (tok >= vocab_size || (padding_idx >= 0 && tok == (uint)padding_idx)) {
        output[out_idx] = 0.0f;
    } else {
        output[out_idx] = weight[tok * embed_dim + d];
    }
}

/// Embedding gather with sqrt(embed_dim) scaling.
__kernel void embed_gather_scaled(
    __global const float* restrict weight,
    __global const uint*  restrict token_ids,
    __global       float* restrict output,
    const uint embed_dim,
    const uint vocab_size,
    const int  padding_idx,
    const float scale
) {
    const uint d = get_global_id(0);
    const uint t = get_global_id(1);
    if (d >= embed_dim) return;

    const uint tok = token_ids[t];
    const uint out_idx = t * embed_dim + d;

    if (tok >= vocab_size || (padding_idx >= 0 && tok == (uint)padding_idx)) {
        output[out_idx] = 0.0f;
    } else {
        output[out_idx] = weight[tok * embed_dim + d] * scale;
    }
}

/// Add position embeddings element-wise.
__kernel void add_position_embeddings(
    __global       float* restrict embeddings, // [seq_len, embed_dim]
    __global const float* restrict pos_weight,  // [max_seq_len, embed_dim]
    const uint embed_dim,
    const uint pos_offset
) {
    const uint d = get_global_id(0);
    const uint t = get_global_id(1);
    if (d >= embed_dim) return;
    const uint idx = t * embed_dim + d;
    embeddings[idx] += pos_weight[(pos_offset + t) * embed_dim + d];
}

/// Add segment (token-type) embeddings element-wise.
__kernel void add_segment_embeddings(
    __global       float* restrict embeddings,  // [seq_len, embed_dim]
    __global const float* restrict seg_weight,   // [num_segments, embed_dim]
    __global const uint*  restrict segment_ids,  // [seq_len]
    const uint embed_dim
) {
    const uint d = get_global_id(0);
    const uint t = get_global_id(1);
    if (d >= embed_dim) return;
    const uint seg = segment_ids[t];
    embeddings[t * embed_dim + d] += seg_weight[seg * embed_dim + d];
}
"#;

// ── Embedding format ─────────────────────────────────────────────

/// Storage format for embedding table weights.
#[derive(Debug, Clone)]
pub enum EmbeddingFormat {
    /// Full-precision f32.
    F32,
    /// Half-precision f16, stored as raw u16 bits.
    F16,
    /// 8-bit quantised with a single scale factor: `value = byte * scale`.
    I8Quantized(f32),
    /// Sparse: only non-zero entries stored as `(row_index, col_index, value)`.
    Sparse(Vec<(u32, u32, f32)>),
}

impl fmt::Display for EmbeddingFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::I8Quantized(s) => write!(f, "I8(scale={s})"),
            Self::Sparse(entries) => write!(f, "Sparse({} nnz)", entries.len()),
        }
    }
}

// ── Embedding table ──────────────────────────────────────────────

/// Embedding table with support for multiple storage formats.
///
/// The `data` field always holds a *decoded* f32 view of the table for
/// CPU reference paths; quantised/sparse originals are kept in `dtype`.
#[derive(Debug, Clone)]
pub struct EmbeddingTable {
    /// Weight data in row-major f32: `[vocab_size, embed_dim]`.
    pub data: Vec<f32>,
    /// Vocabulary size (number of rows).
    pub vocab_size: usize,
    /// Embedding dimension (number of columns).
    pub embed_dim: usize,
    /// Original storage format.
    pub dtype: EmbeddingFormat,
}

impl EmbeddingTable {
    /// Create a new F32 embedding table.
    ///
    /// # Errors
    /// Returns an error if `data.len() != vocab_size * embed_dim`.
    pub fn new(
        data: Vec<f32>,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<Self, EmbedLookupError> {
        let expected = vocab_size * embed_dim;
        if data.len() != expected {
            return Err(EmbedLookupError::DimensionMismatch {
                expected,
                actual: data.len(),
                context: "EmbeddingTable::new".into(),
            });
        }
        Ok(Self { data, vocab_size, embed_dim, dtype: EmbeddingFormat::F32 })
    }

    /// Create a table from F16 data (stored as `u16` bit patterns).
    pub fn from_f16(
        raw: &[u16],
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<Self, EmbedLookupError> {
        let expected = vocab_size * embed_dim;
        if raw.len() != expected {
            return Err(EmbedLookupError::DimensionMismatch {
                expected,
                actual: raw.len(),
                context: "EmbeddingTable::from_f16".into(),
            });
        }
        let data: Vec<f32> = raw.iter().map(|&bits| f16_to_f32(bits)).collect();
        Ok(Self { data, vocab_size, embed_dim, dtype: EmbeddingFormat::F16 })
    }

    /// Create a table from I8 quantised data with a single scale factor.
    pub fn from_i8(
        raw: &[i8],
        scale: f32,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<Self, EmbedLookupError> {
        let expected = vocab_size * embed_dim;
        if raw.len() != expected {
            return Err(EmbedLookupError::DimensionMismatch {
                expected,
                actual: raw.len(),
                context: "EmbeddingTable::from_i8".into(),
            });
        }
        let data: Vec<f32> = raw.iter().map(|&v| v as f32 * scale).collect();
        Ok(Self {
            data,
            vocab_size,
            embed_dim,
            dtype: EmbeddingFormat::I8Quantized(scale),
        })
    }

    /// Create a sparse embedding table. Missing entries default to 0.
    pub fn from_sparse(
        entries: Vec<(u32, u32, f32)>,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Self {
        let mut data = vec![0.0f32; vocab_size * embed_dim];
        for &(row, col, val) in &entries {
            let r = row as usize;
            let c = col as usize;
            if r < vocab_size && c < embed_dim {
                data[r * embed_dim + c] = val;
            }
        }
        Self {
            data,
            vocab_size,
            embed_dim,
            dtype: EmbeddingFormat::Sparse(entries),
        }
    }
}

// ── Lookup configuration ─────────────────────────────────────────

/// Configuration for a lookup operation.
#[derive(Debug, Clone)]
pub struct LookupConfig {
    /// Maximum number of sequences in a batch.
    pub batch_size: usize,
    /// Maximum sequence length.
    pub seq_len: usize,
    /// Optional padding index — tokens equal to this produce zero vectors.
    pub padding_idx: Option<u32>,
    /// If `true`, output embeddings are scaled by `sqrt(embed_dim)`.
    pub scale_by_dim: bool,
}

impl LookupConfig {
    /// Create a basic lookup configuration.
    pub fn new(batch_size: usize, seq_len: usize) -> Self {
        Self { batch_size, seq_len, padding_idx: None, scale_by_dim: false }
    }

    /// Set the padding index.
    #[must_use]
    pub fn with_padding_idx(mut self, idx: u32) -> Self {
        self.padding_idx = Some(idx);
        self
    }

    /// Enable sqrt(embed_dim) scaling.
    #[must_use]
    pub fn with_scale_by_dim(mut self) -> Self {
        self.scale_by_dim = true;
        self
    }
}

// ── Embedding output ─────────────────────────────────────────────

/// Result of an embedding lookup operation.
#[derive(Debug, Clone)]
pub struct EmbeddingOutput {
    /// Gathered embeddings in row-major f32: `[token_count, embed_dim]`.
    pub embeddings: Vec<f32>,
    /// Total number of tokens looked up.
    pub token_count: usize,
    /// Bytes of embedding data read.
    pub memory_bytes: usize,
    /// Wall-clock lookup time in microseconds.
    pub lookup_time_us: u64,
}

// ── Embedding lookup ─────────────────────────────────────────────

/// Performs token → embedding gather using CPU reference kernels.
#[derive(Debug)]
pub struct EmbeddingLookup {
    table: EmbeddingTable,
    config: LookupConfig,
}

impl EmbeddingLookup {
    /// Create a new lookup engine.
    pub fn new(table: EmbeddingTable, config: LookupConfig) -> Self {
        Self { table, config }
    }

    /// Look up embeddings for a flat slice of token IDs.
    ///
    /// Returns [`EmbeddingOutput`] with the gathered vectors.
    pub fn lookup(&self, token_ids: &[u32]) -> Result<EmbeddingOutput, EmbedLookupError> {
        let start = Instant::now();
        let n = token_ids.len();
        let d = self.table.embed_dim;
        let mut embeddings = vec![0.0f32; n * d];

        embed_gather_ref(
            token_ids,
            &self.table.data,
            &mut embeddings,
            self.table.vocab_size,
            d,
            self.config.padding_idx,
        )?;

        if self.config.scale_by_dim {
            let scale = (d as f32).sqrt();
            for v in &mut embeddings {
                *v *= scale;
            }
        }

        let elapsed = start.elapsed();
        Ok(EmbeddingOutput {
            embeddings,
            token_count: n,
            memory_bytes: n * d * std::mem::size_of::<f32>(),
            lookup_time_us: elapsed.as_micros() as u64,
        })
    }

    /// Access the underlying table.
    pub fn table(&self) -> &EmbeddingTable {
        &self.table
    }

    /// Access the lookup configuration.
    pub fn config(&self) -> &LookupConfig {
        &self.config
    }
}

// ── Position embedding ───────────────────────────────────────────

/// Learned or fixed position embeddings, added element-wise.
#[derive(Debug, Clone)]
pub struct PositionEmbedding {
    /// Weight matrix: `[max_positions, embed_dim]`.
    pub weight: Vec<f32>,
    /// Maximum number of positions.
    pub max_positions: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
}

impl PositionEmbedding {
    /// Create a new position embedding table.
    pub fn new(
        weight: Vec<f32>,
        max_positions: usize,
        embed_dim: usize,
    ) -> Result<Self, EmbedLookupError> {
        let expected = max_positions * embed_dim;
        if weight.len() != expected {
            return Err(EmbedLookupError::DimensionMismatch {
                expected,
                actual: weight.len(),
                context: "PositionEmbedding::new".into(),
            });
        }
        Ok(Self { weight, max_positions, embed_dim })
    }

    /// Add position embeddings to `embeddings` in-place.
    ///
    /// `embeddings` has shape `[seq_len, embed_dim]`.
    pub fn add_to(
        &self,
        embeddings: &mut [f32],
        seq_len: usize,
        position_offset: usize,
    ) -> Result<(), EmbedLookupError> {
        let d = self.embed_dim;
        if position_offset + seq_len > self.max_positions {
            return Err(EmbedLookupError::PositionOutOfRange {
                requested: position_offset + seq_len,
                max: self.max_positions,
            });
        }
        if embeddings.len() < seq_len * d {
            return Err(EmbedLookupError::DimensionMismatch {
                expected: seq_len * d,
                actual: embeddings.len(),
                context: "PositionEmbedding::add_to".into(),
            });
        }
        for t in 0..seq_len {
            let emb_off = t * d;
            let pos_off = (position_offset + t) * d;
            for i in 0..d {
                embeddings[emb_off + i] += self.weight[pos_off + i];
            }
        }
        Ok(())
    }
}

// ── Segment embedding ────────────────────────────────────────────

/// Token-type (segment) embeddings for BERT-style models.
#[derive(Debug, Clone)]
pub struct SegmentEmbedding {
    /// Weight matrix: `[num_segments, embed_dim]`.
    pub weight: Vec<f32>,
    /// Number of segment types (typically 2).
    pub num_segments: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
}

impl SegmentEmbedding {
    /// Create a new segment embedding table.
    pub fn new(
        weight: Vec<f32>,
        num_segments: usize,
        embed_dim: usize,
    ) -> Result<Self, EmbedLookupError> {
        let expected = num_segments * embed_dim;
        if weight.len() != expected {
            return Err(EmbedLookupError::DimensionMismatch {
                expected,
                actual: weight.len(),
                context: "SegmentEmbedding::new".into(),
            });
        }
        Ok(Self { weight, num_segments, embed_dim })
    }

    /// Add segment embeddings to `embeddings` in-place.
    pub fn add_to(
        &self,
        embeddings: &mut [f32],
        segment_ids: &[u32],
        seq_len: usize,
    ) -> Result<(), EmbedLookupError> {
        let d = self.embed_dim;
        if embeddings.len() < seq_len * d {
            return Err(EmbedLookupError::DimensionMismatch {
                expected: seq_len * d,
                actual: embeddings.len(),
                context: "SegmentEmbedding::add_to embeddings".into(),
            });
        }
        if segment_ids.len() < seq_len {
            return Err(EmbedLookupError::DimensionMismatch {
                expected: seq_len,
                actual: segment_ids.len(),
                context: "SegmentEmbedding::add_to segment_ids".into(),
            });
        }
        for t in 0..seq_len {
            let seg = segment_ids[t] as usize;
            if seg >= self.num_segments {
                return Err(EmbedLookupError::SegmentOutOfRange {
                    segment_id: seg,
                    num_segments: self.num_segments,
                });
            }
            let emb_off = t * d;
            let seg_off = seg * d;
            for i in 0..d {
                embeddings[emb_off + i] += self.weight[seg_off + i];
            }
        }
        Ok(())
    }
}

// ── Error type ───────────────────────────────────────────────────

/// Errors from embedding lookup operations.
#[derive(Debug, Clone)]
pub enum EmbedLookupError {
    /// Array dimension mismatch.
    DimensionMismatch {
        expected: usize,
        actual: usize,
        context: String,
    },
    /// Token ID exceeds vocabulary size.
    OutOfVocabulary {
        token_id: u32,
        vocab_size: usize,
    },
    /// Position exceeds maximum sequence length.
    PositionOutOfRange {
        requested: usize,
        max: usize,
    },
    /// Segment ID out of range.
    SegmentOutOfRange {
        segment_id: usize,
        num_segments: usize,
    },
}

impl fmt::Display for EmbedLookupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual, context } => {
                write!(
                    f,
                    "dimension mismatch in {context}: expected {expected}, got {actual}"
                )
            }
            Self::OutOfVocabulary { token_id, vocab_size } => {
                write!(
                    f,
                    "token {token_id} out of vocabulary (size {vocab_size})"
                )
            }
            Self::PositionOutOfRange { requested, max } => {
                write!(
                    f,
                    "position {requested} exceeds max {max}"
                )
            }
            Self::SegmentOutOfRange { segment_id, num_segments } => {
                write!(
                    f,
                    "segment {segment_id} exceeds num_segments {num_segments}"
                )
            }
        }
    }
}

impl std::error::Error for EmbedLookupError {}

// ── F16 utility ──────────────────────────────────────────────────

/// Convert an IEEE 754 half-precision bit pattern to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero
        let val = (mant as f32) * (1.0 / 16_777_216.0); // 2^-24
        if sign == 1 { -val } else { val }
    } else if exp == 31 {
        // Inf / NaN
        if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        }
    } else {
        let f_exp = (exp as i32) - 15 + 127;
        let f_bits = (sign << 31) | ((f_exp as u32) << 23) | (mant << 13);
        f32::from_bits(f_bits)
    }
}

/// Convert f32 to f16 bit pattern (round-to-nearest-even).
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 255 {
        // Inf / NaN
        let h_mant = if mant != 0 { 0x200 } else { 0 };
        return ((sign << 15) | 0x7C00 | h_mant) as u16;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        return ((sign << 15) | 0x7C00) as u16; // overflow → Inf
    }
    if unbiased < -24 {
        return (sign << 15) as u16; // underflow → 0
    }
    if unbiased < -14 {
        let shift = (-14 - unbiased) as u32;
        let h_mant = ((0x800000 | mant) >> (shift + 13)) as u32;
        return ((sign << 15) | h_mant) as u16;
    }

    let h_exp = (unbiased + 15) as u32;
    let h_mant = mant >> 13;
    ((sign << 15) | (h_exp << 10) | h_mant) as u16
}

// ── CPU reference: coalesced gather ──────────────────────────────

/// CPU reference for the coalesced embedding gather kernel.
///
/// For each token, copies the corresponding row from the weight table.
/// OOV tokens (>= vocab_size) and padding tokens produce zero vectors.
pub fn embed_gather_ref(
    token_ids: &[u32],
    weight: &[f32],
    output: &mut [f32],
    vocab_size: usize,
    embed_dim: usize,
    padding_idx: Option<u32>,
) -> Result<(), EmbedLookupError> {
    let seq_len = token_ids.len();
    if weight.len() < vocab_size * embed_dim {
        return Err(EmbedLookupError::DimensionMismatch {
            expected: vocab_size * embed_dim,
            actual: weight.len(),
            context: "embed_gather_ref weight".into(),
        });
    }
    if output.len() < seq_len * embed_dim {
        return Err(EmbedLookupError::DimensionMismatch {
            expected: seq_len * embed_dim,
            actual: output.len(),
            context: "embed_gather_ref output".into(),
        });
    }

    for (t, &tok) in token_ids.iter().enumerate() {
        let tid = tok as usize;
        let out_start = t * embed_dim;
        let is_pad = padding_idx.is_some_and(|p| tok == p);

        if tid < vocab_size && !is_pad {
            let src = tid * embed_dim;
            output[out_start..out_start + embed_dim]
                .copy_from_slice(&weight[src..src + embed_dim]);
        } else {
            output[out_start..out_start + embed_dim].fill(0.0);
        }
    }
    Ok(())
}

/// CPU reference for the scaled embedding gather kernel.
pub fn embed_gather_scaled_ref(
    token_ids: &[u32],
    weight: &[f32],
    output: &mut [f32],
    vocab_size: usize,
    embed_dim: usize,
    padding_idx: Option<u32>,
    scale: f32,
) -> Result<(), EmbedLookupError> {
    embed_gather_ref(token_ids, weight, output, vocab_size, embed_dim, padding_idx)?;
    let n = token_ids.len() * embed_dim;
    for v in &mut output[..n] {
        *v *= scale;
    }
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: deterministic weight table [vocab_size, embed_dim].
    fn make_weight(vocab_size: usize, embed_dim: usize) -> Vec<f32> {
        (0..vocab_size * embed_dim)
            .map(|i| (i as f32) * 0.01)
            .collect()
    }

    // ── OpenCL kernel source validation ──────────────────────

    #[test]
    fn cl_source_not_empty() {
        assert!(!EMBED_LOOKUP_CL.is_empty());
    }

    #[test]
    fn cl_source_contains_kernel_keyword() {
        assert!(EMBED_LOOKUP_CL.contains("__kernel"));
    }

    #[test]
    fn cl_source_has_coalesced_gather() {
        assert!(EMBED_LOOKUP_CL.contains("embed_gather_coalesced"));
    }

    #[test]
    fn cl_source_has_scaled_gather() {
        assert!(EMBED_LOOKUP_CL.contains("embed_gather_scaled"));
    }

    #[test]
    fn cl_source_has_position_kernel() {
        assert!(EMBED_LOOKUP_CL.contains("add_position_embeddings"));
    }

    #[test]
    fn cl_source_has_segment_kernel() {
        assert!(EMBED_LOOKUP_CL.contains("add_segment_embeddings"));
    }

    // ── EmbeddingTable construction ──────────────────────────

    #[test]
    fn table_f32_basic() {
        let w = make_weight(10, 4);
        let t = EmbeddingTable::new(w.clone(), 10, 4).unwrap();
        assert_eq!(t.vocab_size, 10);
        assert_eq!(t.embed_dim, 4);
        assert_eq!(t.data.len(), 40);
    }

    #[test]
    fn table_f32_dimension_mismatch() {
        let w = vec![0.0; 15]; // wrong length
        assert!(EmbeddingTable::new(w, 10, 4).is_err());
    }

    #[test]
    fn table_f16_round_trip() {
        let values: Vec<f32> = vec![1.0, -1.0, 0.5, 0.0];
        let f16_bits: Vec<u16> = values.iter().map(|&v| f32_to_f16(v)).collect();
        let t = EmbeddingTable::from_f16(&f16_bits, 2, 2).unwrap();
        for (got, &want) in t.data.iter().zip(values.iter()) {
            assert!((got - want).abs() < 1e-3, "{got} != {want}");
        }
        assert!(matches!(t.dtype, EmbeddingFormat::F16));
    }

    #[test]
    fn table_i8_quantized() {
        let raw: Vec<i8> = vec![10, -20, 30, 0];
        let scale = 0.1;
        let t = EmbeddingTable::from_i8(&raw, scale, 2, 2).unwrap();
        assert!((t.data[0] - 1.0).abs() < 1e-6);
        assert!((t.data[1] - (-2.0)).abs() < 1e-6);
        assert!((t.data[2] - 3.0).abs() < 1e-6);
        assert!((t.data[3] - 0.0).abs() < 1e-6);
        assert!(matches!(t.dtype, EmbeddingFormat::I8Quantized(_)));
    }

    #[test]
    fn table_sparse() {
        let entries = vec![(0, 1, 5.0), (1, 0, 3.0)];
        let t = EmbeddingTable::from_sparse(entries, 2, 2);
        assert_eq!(t.data, vec![0.0, 5.0, 3.0, 0.0]);
        assert!(matches!(t.dtype, EmbeddingFormat::Sparse(_)));
    }

    // ── EmbeddingFormat display ──────────────────────────────

    #[test]
    fn format_display_f32() {
        assert_eq!(EmbeddingFormat::F32.to_string(), "F32");
    }

    #[test]
    fn format_display_f16() {
        assert_eq!(EmbeddingFormat::F16.to_string(), "F16");
    }

    #[test]
    fn format_display_i8() {
        let fmt = EmbeddingFormat::I8Quantized(0.5);
        assert!(fmt.to_string().contains("I8"));
    }

    #[test]
    fn format_display_sparse() {
        let fmt = EmbeddingFormat::Sparse(vec![(0, 0, 1.0)]);
        assert!(fmt.to_string().contains("Sparse"));
    }

    // ── Basic token lookup ───────────────────────────────────

    #[test]
    fn gather_single_token() {
        let w = make_weight(4, 3);
        let mut out = vec![0.0; 3];
        embed_gather_ref(&[2], &w, &mut out, 4, 3, None).unwrap();
        // Token 2 → row 2 → [0.06, 0.07, 0.08]
        for i in 0..3 {
            assert!((out[i] - w[2 * 3 + i]).abs() < 1e-6);
        }
    }

    #[test]
    fn gather_multiple_tokens() {
        let w = make_weight(4, 3);
        let ids = [0u32, 3, 1];
        let mut out = vec![0.0; 9];
        embed_gather_ref(&ids, &w, &mut out, 4, 3, None).unwrap();
        for (t, &tok) in ids.iter().enumerate() {
            for d in 0..3 {
                assert!(
                    (out[t * 3 + d] - w[tok as usize * 3 + d]).abs() < 1e-6
                );
            }
        }
    }

    // ── Batch lookup ─────────────────────────────────────────

    #[test]
    fn batch_lookup_two_sequences() {
        let w = make_weight(8, 4);
        let config = LookupConfig::new(2, 3);
        let table = EmbeddingTable::new(w.clone(), 8, 4).unwrap();
        let lookup = EmbeddingLookup::new(table, config);

        // Sequence 1: [1, 3, 5], Sequence 2: [0, 2, 4]
        let ids: Vec<u32> = vec![1, 3, 5, 0, 2, 4];
        let result = lookup.lookup(&ids).unwrap();
        assert_eq!(result.token_count, 6);
        assert_eq!(result.embeddings.len(), 24);
        // Verify first token (id=1)
        for d in 0..4 {
            assert!((result.embeddings[d] - w[1 * 4 + d]).abs() < 1e-6);
        }
    }

    // ── Padding index handling ───────────────────────────────

    #[test]
    fn padding_produces_zeros() {
        let w = make_weight(4, 3);
        let mut out = vec![99.0; 6];
        embed_gather_ref(&[0, 2], &w, &mut out, 4, 3, Some(0)).unwrap();
        // Token 0 is padding → zeros
        assert_eq!(&out[0..3], &[0.0, 0.0, 0.0]);
        // Token 2 → valid
        for d in 0..3 {
            assert!((out[3 + d] - w[2 * 3 + d]).abs() < 1e-6);
        }
    }

    #[test]
    fn padding_mid_sequence() {
        let w = make_weight(5, 2);
        let mut out = vec![0.0; 6];
        embed_gather_ref(&[1, 3, 1], &w, &mut out, 5, 2, Some(3)).unwrap();
        // Index 1 → [t=1 row]; Index 3 → padding; Index 1 → row again
        assert_eq!(&out[2..4], &[0.0, 0.0]); // padding zeroed
    }

    // ── OOV tokens ───────────────────────────────────────────

    #[test]
    fn oov_token_produces_zeros() {
        let w = make_weight(4, 3);
        let mut out = vec![99.0; 3];
        embed_gather_ref(&[100], &w, &mut out, 4, 3, None).unwrap();
        assert_eq!(&out[..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn oov_at_vocab_boundary() {
        let w = make_weight(4, 2);
        let mut out = vec![99.0; 2];
        embed_gather_ref(&[4], &w, &mut out, 4, 2, None).unwrap();
        assert_eq!(&out[..], &[0.0, 0.0]);
    }

    // ── Scale-by-dim ─────────────────────────────────────────

    #[test]
    fn scale_by_dim() {
        let w = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
        let config = LookupConfig::new(1, 1).with_scale_by_dim();
        let table = EmbeddingTable::new(w.clone(), 2, 2).unwrap();
        let lookup = EmbeddingLookup::new(table, config);

        let result = lookup.lookup(&[0]).unwrap();
        let scale = (2.0f32).sqrt();
        assert!((result.embeddings[0] - 1.0 * scale).abs() < 1e-6);
        assert!((result.embeddings[1] - 2.0 * scale).abs() < 1e-6);
    }

    #[test]
    fn scaled_gather_ref() {
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 2];
        embed_gather_scaled_ref(&[1], &w, &mut out, 2, 2, None, 0.5).unwrap();
        assert!((out[0] - 1.5).abs() < 1e-6); // 3.0*0.5
        assert!((out[1] - 2.0).abs() < 1e-6); // 4.0*0.5
    }

    // ── F16 accuracy ─────────────────────────────────────────

    #[test]
    fn f16_accuracy_within_tolerance() {
        let values: Vec<f32> =
            (0..16).map(|i| (i as f32) * 0.1 - 0.8).collect();
        let f16_bits: Vec<u16> = values.iter().map(|&v| f32_to_f16(v)).collect();
        let table = EmbeddingTable::from_f16(&f16_bits, 4, 4).unwrap();
        for (i, &expected) in values.iter().enumerate() {
            let diff = (table.data[i] - expected).abs();
            assert!(diff < 0.01, "f16 error {diff} at index {i}");
        }
    }

    // ── I8 quantized accuracy ────────────────────────────────

    #[test]
    fn i8_quantized_accuracy() {
        let raw: Vec<i8> = vec![127, -128, 0, 64, -64, 1, -1, 50];
        let scale = 0.01;
        let table = EmbeddingTable::from_i8(&raw, scale, 2, 4).unwrap();
        for (i, &byte) in raw.iter().enumerate() {
            let expected = byte as f32 * scale;
            assert!(
                (table.data[i] - expected).abs() < 1e-6,
                "i8 mismatch at {i}"
            );
        }
    }

    #[test]
    fn i8_lookup_through_engine() {
        let raw: Vec<i8> = vec![10, 20, 30, 40];
        let table = EmbeddingTable::from_i8(&raw, 0.5, 2, 2).unwrap();
        let config = LookupConfig::new(1, 1);
        let engine = EmbeddingLookup::new(table, config);
        let result = engine.lookup(&[1]).unwrap();
        assert!((result.embeddings[0] - 15.0).abs() < 1e-6); // 30*0.5
        assert!((result.embeddings[1] - 20.0).abs() < 1e-6); // 40*0.5
    }

    // ── Position embedding ───────────────────────────────────

    #[test]
    fn position_embedding_add() {
        let pos_w = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let pos = PositionEmbedding::new(pos_w.clone(), 3, 2).unwrap();
        let mut emb = vec![1.0, 2.0, 3.0, 4.0]; // seq_len=2, dim=2
        pos.add_to(&mut emb, 2, 0).unwrap();
        assert!((emb[0] - 1.1).abs() < 1e-6);
        assert!((emb[1] - 2.2).abs() < 1e-6);
        assert!((emb[2] - 3.3).abs() < 1e-6);
        assert!((emb[3] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn position_embedding_with_offset() {
        let pos_w = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let pos = PositionEmbedding::new(pos_w, 3, 2).unwrap();
        let mut emb = vec![1.0, 2.0]; // seq_len=1
        pos.add_to(&mut emb, 1, 2).unwrap();
        assert!((emb[0] - 1.5).abs() < 1e-6);
        assert!((emb[1] - 2.6).abs() < 1e-6);
    }

    #[test]
    fn position_embedding_out_of_range() {
        let pos_w = vec![0.1, 0.2];
        let pos = PositionEmbedding::new(pos_w, 1, 2).unwrap();
        let mut emb = vec![1.0, 2.0];
        assert!(pos.add_to(&mut emb, 1, 1).is_err());
    }

    // ── Segment embedding ────────────────────────────────────

    #[test]
    fn segment_embedding_add() {
        let seg_w = vec![0.1, 0.2, 0.3, 0.4]; // 2 segments, dim=2
        let seg = SegmentEmbedding::new(seg_w, 2, 2).unwrap();
        let mut emb = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens
        seg.add_to(&mut emb, &[0, 1], 2).unwrap();
        assert!((emb[0] - 1.1).abs() < 1e-6);
        assert!((emb[1] - 2.2).abs() < 1e-6);
        assert!((emb[2] - 3.3).abs() < 1e-6);
        assert!((emb[3] - 4.4).abs() < 1e-6);
    }

    #[test]
    fn segment_out_of_range_error() {
        let seg_w = vec![0.1, 0.2];
        let seg = SegmentEmbedding::new(seg_w, 1, 2).unwrap();
        let mut emb = vec![1.0, 2.0];
        assert!(seg.add_to(&mut emb, &[1], 1).is_err());
    }

    // ── Edge cases ───────────────────────────────────────────

    #[test]
    fn vocab_size_one() {
        let w = vec![42.0, 99.0];
        let table = EmbeddingTable::new(w, 1, 2).unwrap();
        let config = LookupConfig::new(1, 1);
        let engine = EmbeddingLookup::new(table, config);
        let result = engine.lookup(&[0]).unwrap();
        assert!((result.embeddings[0] - 42.0).abs() < 1e-6);
        assert!((result.embeddings[1] - 99.0).abs() < 1e-6);
    }

    #[test]
    fn embed_dim_one() {
        let w = vec![10.0, 20.0, 30.0];
        let table = EmbeddingTable::new(w, 3, 1).unwrap();
        let config = LookupConfig::new(1, 1);
        let engine = EmbeddingLookup::new(table, config);
        let result = engine.lookup(&[1]).unwrap();
        assert!((result.embeddings[0] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn empty_token_list() {
        let w = make_weight(4, 3);
        let mut out = vec![];
        embed_gather_ref(&[], &w, &mut out, 4, 3, None).unwrap();
        assert!(out.is_empty());
    }

    // ── EmbeddingOutput metadata ─────────────────────────────

    #[test]
    fn output_metadata_correct() {
        let w = make_weight(4, 3);
        let table = EmbeddingTable::new(w, 4, 3).unwrap();
        let config = LookupConfig::new(1, 2);
        let engine = EmbeddingLookup::new(table, config);
        let result = engine.lookup(&[0, 1]).unwrap();
        assert_eq!(result.token_count, 2);
        assert_eq!(result.memory_bytes, 2 * 3 * 4); // 2 tokens * 3 dims * 4 bytes
    }

    // ── LookupConfig builder ─────────────────────────────────

    #[test]
    fn config_builder_defaults() {
        let c = LookupConfig::new(4, 128);
        assert_eq!(c.batch_size, 4);
        assert_eq!(c.seq_len, 128);
        assert!(c.padding_idx.is_none());
        assert!(!c.scale_by_dim);
    }

    #[test]
    fn config_builder_with_padding() {
        let c = LookupConfig::new(1, 1).with_padding_idx(0);
        assert_eq!(c.padding_idx, Some(0));
    }

    #[test]
    fn config_builder_with_scale() {
        let c = LookupConfig::new(1, 1).with_scale_by_dim();
        assert!(c.scale_by_dim);
    }

    // ── Error display ────────────────────────────────────────

    #[test]
    fn error_display_dimension() {
        let e = EmbedLookupError::DimensionMismatch {
            expected: 10,
            actual: 5,
            context: "test".into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn error_display_oov() {
        let e = EmbedLookupError::OutOfVocabulary {
            token_id: 999,
            vocab_size: 100,
        };
        assert!(e.to_string().contains("999"));
    }

    #[test]
    fn error_display_position() {
        let e = EmbedLookupError::PositionOutOfRange {
            requested: 512,
            max: 256,
        };
        assert!(e.to_string().contains("512"));
    }

    #[test]
    fn error_display_segment() {
        let e = EmbedLookupError::SegmentOutOfRange {
            segment_id: 5,
            num_segments: 2,
        };
        assert!(e.to_string().contains("5"));
    }

    // ── Property: dimensionality preserved ───────────────────

    #[test]
    fn property_lookup_preserves_dim() {
        for dim in [1, 2, 4, 8, 16, 64, 128, 256] {
            let vocab = 10;
            let w = make_weight(vocab, dim);
            let table = EmbeddingTable::new(w, vocab, dim).unwrap();
            let config = LookupConfig::new(1, 5);
            let engine = EmbeddingLookup::new(table, config);
            let ids: Vec<u32> = (0..5).collect();
            let result = engine.lookup(&ids).unwrap();
            assert_eq!(result.embeddings.len(), 5 * dim);
        }
    }

    #[test]
    fn property_padding_always_zeros() {
        for dim in [1, 3, 7, 16] {
            let vocab = 8;
            let w: Vec<f32> = (0..vocab * dim).map(|i| (i + 1) as f32).collect();
            let mut out = vec![0.0; 3 * dim];
            embed_gather_ref(
                &[2, 2, 2],
                &w,
                &mut out,
                vocab,
                dim,
                Some(2),
            )
            .unwrap();
            for v in &out {
                assert_eq!(*v, 0.0, "padding should zero with dim={dim}");
            }
        }
    }

    // ── Combined: lookup + position + segment ────────────────

    #[test]
    fn combined_lookup_position_segment() {
        let w = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
        let table = EmbeddingTable::new(w, 2, 2).unwrap();
        let config = LookupConfig::new(1, 2);
        let engine = EmbeddingLookup::new(table, config);
        let mut result = engine.lookup(&[0, 1]).unwrap();

        let pos_w = vec![0.1, 0.1, 0.2, 0.2];
        let pos = PositionEmbedding::new(pos_w, 2, 2).unwrap();
        pos.add_to(&mut result.embeddings, 2, 0).unwrap();

        let seg_w = vec![0.01, 0.01, 0.02, 0.02];
        let seg = SegmentEmbedding::new(seg_w, 2, 2).unwrap();
        seg.add_to(&mut result.embeddings, &[0, 1], 2).unwrap();

        // Token 0: [1.0+0.1+0.01, 2.0+0.1+0.01]
        assert!((result.embeddings[0] - 1.11).abs() < 1e-6);
        assert!((result.embeddings[1] - 2.11).abs() < 1e-6);
        // Token 1: [3.0+0.2+0.02, 4.0+0.2+0.02]
        assert!((result.embeddings[2] - 3.22).abs() < 1e-6);
        assert!((result.embeddings[3] - 4.22).abs() < 1e-6);
    }

    // ── F16 conversion edge cases ────────────────────────────

    #[test]
    fn f16_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn f16_one() {
        let val = f16_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn f16_negative_one() {
        let val = f16_to_f32(0xBC00);
        assert!((val - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn f16_inf() {
        assert!(f16_to_f32(0x7C00).is_infinite());
    }

    #[test]
    fn f16_nan() {
        assert!(f16_to_f32(0x7C01).is_nan());
    }

    // ── Sparse embedding ─────────────────────────────────────

    #[test]
    fn sparse_lookup_correctness() {
        let entries = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)];
        let table = EmbeddingTable::from_sparse(entries, 2, 2);
        let config = LookupConfig::new(1, 2);
        let engine = EmbeddingLookup::new(table, config);
        let result = engine.lookup(&[0, 1]).unwrap();
        assert!((result.embeddings[0] - 1.0).abs() < 1e-6);
        assert!((result.embeddings[1] - 2.0).abs() < 1e-6);
        assert!((result.embeddings[2] - 3.0).abs() < 1e-6);
        assert!((result.embeddings[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sparse_out_of_bounds_entries_ignored() {
        let entries = vec![(99, 99, 1.0)]; // out of range
        let table = EmbeddingTable::from_sparse(entries, 2, 2);
        assert_eq!(table.data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    // ── Accessor tests ───────────────────────────────────────

    #[test]
    fn lookup_table_accessor() {
        let table = EmbeddingTable::new(vec![1.0, 2.0], 1, 2).unwrap();
        let config = LookupConfig::new(1, 1);
        let engine = EmbeddingLookup::new(table, config);
        assert_eq!(engine.table().vocab_size, 1);
        assert_eq!(engine.config().batch_size, 1);
    }
}
