//! Module stub - implementation pending merge from feature branch
//! Weight compression with pruning, low-rank decomposition, and encoding.

// Numerical code inherently requires usize↔f64 and f64→integer casts.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::cmp::Reverse;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the weight-compression pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionError {
    /// Input tensor is empty or has invalid dimensions.
    InvalidInput(String),
    /// Compression quality fell below the configured threshold.
    QualityBelowThreshold { achieved: f64, required: f64 },
    /// The requested algorithm is not applicable to the given weights.
    AlgorithmNotApplicable(String),
    /// An internal numerical error (e.g. SVD did not converge).
    NumericalError(String),
}

impl fmt::Display for CompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::QualityBelowThreshold { achieved, required } => {
                write!(f, "quality {achieved:.4} below threshold {required:.4}")
            }
            Self::AlgorithmNotApplicable(msg) => write!(f, "algorithm not applicable: {msg}"),
            Self::NumericalError(msg) => write!(f, "numerical error: {msg}"),
        }
    }
}

impl std::error::Error for CompressionError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Selects the compression strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionAlgorithm {
    /// Magnitude-based weight pruning (structured or unstructured).
    Pruning,
    /// Knowledge distillation (placeholder – requires teacher model).
    Distillation,
    /// SVD-based low-rank approximation.
    LowRank,
    /// Scalar / vector quantization of weight values.
    Quantization,
    /// Huffman coding of quantized weight symbols.
    Huffman,
}

impl fmt::Display for CompressionAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Pruning => "Pruning",
            Self::Distillation => "Distillation",
            Self::LowRank => "LowRank",
            Self::Quantization => "Quantization",
            Self::Huffman => "Huffman",
        };
        write!(f, "{name}")
    }
}

/// Top-level configuration for a compression run.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Which algorithm to use.
    pub algorithm: CompressionAlgorithm,
    /// Target compression ratio (compressed / original). Lower is more compressed.
    pub target_ratio: f64,
    /// Minimum acceptable quality score in `[0, 1]`.
    pub quality_threshold: f64,
}

impl CompressionConfig {
    pub const fn new(
        algorithm: CompressionAlgorithm,
        target_ratio: f64,
        quality_threshold: f64,
    ) -> Self {
        Self { algorithm, target_ratio, quality_threshold }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Pruning,
            target_ratio: 0.5,
            quality_threshold: 0.95,
        }
    }
}

// ---------------------------------------------------------------------------
// Pruning
// ---------------------------------------------------------------------------

/// Pruning mode selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningMode {
    /// Zero out individual weights below the threshold.
    Unstructured,
    /// Zero out entire rows whose L2-norm is below the threshold.
    Structured,
}

/// Magnitude-based weight pruner.
#[derive(Debug)]
pub struct WeightPruner {
    pub mode: PruningMode,
    pub threshold: f64,
}

impl WeightPruner {
    pub const fn new(mode: PruningMode, threshold: f64) -> Self {
        Self { mode, threshold }
    }

    /// Prune a *flat* weight vector in-place and return the sparsity ratio.
    pub fn prune_flat(&self, weights: &mut [f64]) -> f64 {
        if weights.is_empty() {
            return 0.0;
        }
        let mut zeroed = 0usize;
        for w in weights.iter_mut() {
            if w.abs() < self.threshold {
                *w = 0.0;
                zeroed += 1;
            }
        }
        zeroed as f64 / weights.len() as f64
    }

    /// Prune a 2-D weight matrix (row-major, `rows × cols`).
    ///
    /// In **Structured** mode entire rows with L2-norm below `threshold` are
    /// zeroed; in **Unstructured** mode individual elements are zeroed.
    pub fn prune_matrix(
        &self,
        weights: &mut [f64],
        rows: usize,
        cols: usize,
    ) -> Result<f64, CompressionError> {
        if weights.len() != rows * cols || rows == 0 || cols == 0 {
            return Err(CompressionError::InvalidInput(format!(
                "expected {rows}×{cols}={} elements, got {}",
                rows * cols,
                weights.len()
            )));
        }
        match self.mode {
            PruningMode::Unstructured => Ok(self.prune_flat(weights)),
            PruningMode::Structured => {
                let mut zeroed_rows = 0usize;
                for r in 0..rows {
                    let row = &weights[r * cols..(r + 1) * cols];
                    let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if norm < self.threshold {
                        for v in &mut weights[r * cols..(r + 1) * cols] {
                            *v = 0.0;
                        }
                        zeroed_rows += 1;
                    }
                }
                Ok(zeroed_rows as f64 / rows as f64)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Low-rank decomposition (simplified SVD)
// ---------------------------------------------------------------------------

/// SVD-based low-rank decomposer.
///
/// Approximates a matrix `A ≈ U * S * Vᵀ` keeping only the top-k singular
/// values. This implementation uses a simplified power-iteration approach
/// suitable for demonstration and testing; production code would delegate to
/// an optimised LAPACK/cuSOLVER backend.
#[derive(Debug)]
pub struct LowRankDecomposer {
    /// Number of singular values / vectors to retain.
    pub rank: usize,
    /// Number of power-iteration steps (higher = more accurate).
    pub iterations: usize,
}

impl LowRankDecomposer {
    pub const fn new(rank: usize, iterations: usize) -> Self {
        Self { rank, iterations }
    }

    /// Decompose `matrix` (row-major, `rows × cols`) into `(U, S, Vt)`.
    ///
    /// * `U`  – `rows × rank`
    /// * `S`  – `rank` singular values
    /// * `Vt` – `rank × cols`
    pub fn decompose(
        &self,
        matrix: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<LowRankResult, CompressionError> {
        if matrix.len() != rows * cols || rows == 0 || cols == 0 {
            return Err(CompressionError::InvalidInput(format!(
                "expected {rows}×{cols}={} elements, got {}",
                rows * cols,
                matrix.len()
            )));
        }
        let effective_rank = self.rank.min(rows).min(cols);
        if effective_rank == 0 {
            return Err(CompressionError::InvalidInput("rank must be > 0".into()));
        }

        let mut u_out = vec![0.0f64; rows * effective_rank];
        let mut s_out = vec![0.0f64; effective_rank];
        let mut vt_out = vec![0.0f64; effective_rank * cols];

        // Work on a deflated copy so we can subtract found components.
        let mut residual: Vec<f64> = matrix.to_vec();

        for k in 0..effective_rank {
            // Initialise v as a unit vector.
            let mut v = vec![0.0f64; cols];
            v[k % cols] = 1.0;

            for _ in 0..self.iterations {
                // u = A * v  (rows-vector)
                let mut u = vec![0.0f64; rows];
                for i in 0..rows {
                    let mut dot = 0.0;
                    for j in 0..cols {
                        dot += residual[i * cols + j] * v[j];
                    }
                    u[i] = dot;
                }
                // normalise u
                let u_norm = u.iter().map(|x| x * x).sum::<f64>().sqrt();
                if u_norm < 1e-15 {
                    break;
                }
                for x in &mut u {
                    *x /= u_norm;
                }

                // v = Aᵀ * u  (cols-vector)
                v = vec![0.0f64; cols];
                for j in 0..cols {
                    let mut dot = 0.0;
                    for i in 0..rows {
                        dot += residual[i * cols + j] * u[i];
                    }
                    v[j] = dot;
                }
                let v_norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if v_norm < 1e-15 {
                    break;
                }
                for x in &mut v {
                    *x /= v_norm;
                }
            }

            // Recompute u and sigma properly after power iterations.
            let mut u_final = vec![0.0f64; rows];
            for i in 0..rows {
                let mut dot = 0.0;
                for j in 0..cols {
                    dot += residual[i * cols + j] * v[j];
                }
                u_final[i] = dot;
            }
            let sigma = u_final.iter().map(|x| x * x).sum::<f64>().sqrt();
            if sigma > 1e-15 {
                for x in &mut u_final {
                    *x /= sigma;
                }
            }

            // Store results.
            for i in 0..rows {
                u_out[i * effective_rank + k] = u_final[i];
            }
            s_out[k] = sigma;
            for j in 0..cols {
                vt_out[k * cols + j] = v[j];
            }

            // Deflate: residual -= sigma * u * vᵀ
            for i in 0..rows {
                for j in 0..cols {
                    residual[i * cols + j] -= sigma * u_final[i] * v[j];
                }
            }
        }

        let original_elements = rows * cols;
        let compressed_elements = rows * effective_rank + effective_rank + effective_rank * cols;
        let ratio = compressed_elements as f64 / original_elements as f64;

        Ok(LowRankResult {
            u: u_out,
            s: s_out,
            vt: vt_out,
            rows,
            cols,
            rank: effective_rank,
            ratio,
        })
    }

    /// Reconstruct the approximated matrix from a [`LowRankResult`].
    pub fn reconstruct(result: &LowRankResult) -> Vec<f64> {
        let LowRankResult { u, s, vt, rows, cols, rank, .. } = result;
        let mut out = vec![0.0f64; rows * cols];
        for i in 0..*rows {
            for j in 0..*cols {
                let mut val = 0.0;
                for k in 0..*rank {
                    val += u[i * rank + k] * s[k] * vt[k * cols + j];
                }
                out[i * cols + j] = val;
            }
        }
        out
    }
}

/// Result of a low-rank decomposition.
#[derive(Debug, Clone)]
pub struct LowRankResult {
    /// Left singular vectors (rows × rank, row-major).
    pub u: Vec<f64>,
    /// Singular values (length = rank).
    pub s: Vec<f64>,
    /// Right singular vectors transposed (rank × cols, row-major).
    pub vt: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    /// Storage ratio (compressed / original element count).
    pub ratio: f64,
}

// ---------------------------------------------------------------------------
// Sparse encoding (CSR / CSC)
// ---------------------------------------------------------------------------

/// Sparse matrix storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Compressed Sparse Row.
    Csr,
    /// Compressed Sparse Column.
    Csc,
}

/// Sparse-encoded weight matrix.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    pub format: SparseFormat,
    pub rows: usize,
    pub cols: usize,
    /// Non-zero values.
    pub values: Vec<f64>,
    /// Column (CSR) or row (CSC) indices of each non-zero value.
    pub indices: Vec<usize>,
    /// Row (CSR) or column (CSC) pointer array.
    pub pointers: Vec<usize>,
    /// Number of non-zero elements.
    pub nnz: usize,
}

/// Encoder that converts dense matrices to CSR or CSC sparse representation.
#[derive(Debug)]
pub struct SparseEncoder {
    pub format: SparseFormat,
}

impl SparseEncoder {
    pub const fn new(format: SparseFormat) -> Self {
        Self { format }
    }

    /// Encode a dense row-major matrix into its sparse representation.
    pub fn encode(
        &self,
        data: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<SparseMatrix, CompressionError> {
        if data.len() != rows * cols || rows == 0 || cols == 0 {
            return Err(CompressionError::InvalidInput(format!(
                "expected {rows}×{cols}={} elements, got {}",
                rows * cols,
                data.len()
            )));
        }
        match self.format {
            SparseFormat::Csr => Ok(encode_csr(data, rows, cols)),
            SparseFormat::Csc => Ok(encode_csc(data, rows, cols)),
        }
    }

    /// Decode a sparse matrix back to dense row-major form.
    pub fn decode(sparse: &SparseMatrix) -> Vec<f64> {
        let mut dense = vec![0.0f64; sparse.rows * sparse.cols];
        match sparse.format {
            SparseFormat::Csr => {
                for r in 0..sparse.rows {
                    let start = sparse.pointers[r];
                    let end = sparse.pointers[r + 1];
                    for idx in start..end {
                        let c = sparse.indices[idx];
                        dense[r * sparse.cols + c] = sparse.values[idx];
                    }
                }
            }
            SparseFormat::Csc => {
                for c in 0..sparse.cols {
                    let start = sparse.pointers[c];
                    let end = sparse.pointers[c + 1];
                    for idx in start..end {
                        let r = sparse.indices[idx];
                        dense[r * sparse.cols + c] = sparse.values[idx];
                    }
                }
            }
        }
        dense
    }
}

fn encode_csr(data: &[f64], rows: usize, cols: usize) -> SparseMatrix {
    let mut values = Vec::new();
    let mut indices = Vec::new();
    let mut pointers = vec![0usize];

    for r in 0..rows {
        for c in 0..cols {
            let v = data[r * cols + c];
            if v != 0.0 {
                values.push(v);
                indices.push(c);
            }
        }
        pointers.push(values.len());
    }
    let nnz = values.len();
    SparseMatrix { format: SparseFormat::Csr, rows, cols, values, indices, pointers, nnz }
}

fn encode_csc(data: &[f64], rows: usize, cols: usize) -> SparseMatrix {
    let mut values = Vec::new();
    let mut indices = Vec::new();
    let mut pointers = vec![0usize];

    for c in 0..cols {
        for r in 0..rows {
            let v = data[r * cols + c];
            if v != 0.0 {
                values.push(v);
                indices.push(r);
            }
        }
        pointers.push(values.len());
    }
    let nnz = values.len();
    SparseMatrix { format: SparseFormat::Csc, rows, cols, values, indices, pointers, nnz }
}

// ---------------------------------------------------------------------------
// Huffman encoding
// ---------------------------------------------------------------------------

/// A node in the Huffman tree.
#[derive(Debug, Clone)]
enum HuffmanNode {
    Leaf { symbol: i64, freq: usize },
    Internal { freq: usize, left: Box<Self>, right: Box<Self> },
}

impl HuffmanNode {
    const fn freq(&self) -> usize {
        match self {
            Self::Leaf { freq, .. } | Self::Internal { freq, .. } => *freq,
        }
    }
}

/// Result of Huffman encoding.
#[derive(Debug, Clone)]
pub struct HuffmanEncoded {
    /// Bit-string per value (stored as `Vec<bool>` for simplicity).
    pub bitstream: Vec<bool>,
    /// Codebook mapping symbol → bit pattern.
    pub codebook: HashMap<i64, Vec<bool>>,
    /// Number of encoded symbols.
    pub symbol_count: usize,
    /// Total bits used.
    pub total_bits: usize,
}

/// Huffman encoder for quantized (integer) weight symbols.
#[derive(Debug)]
pub struct HuffmanEncoder;

impl HuffmanEncoder {
    pub const fn new() -> Self {
        Self
    }

    /// Build a codebook and encode `symbols`.
    pub fn encode(&self, symbols: &[i64]) -> Result<HuffmanEncoded, CompressionError> {
        if symbols.is_empty() {
            return Err(CompressionError::InvalidInput("empty symbol stream".into()));
        }

        // Frequency table.
        let mut freq: HashMap<i64, usize> = HashMap::new();
        for &s in symbols {
            *freq.entry(s).or_insert(0) += 1;
        }

        let codebook = Self::build_codebook(&freq);

        let mut bitstream = Vec::new();
        for &s in symbols {
            let bits = codebook.get(&s).ok_or_else(|| {
                CompressionError::NumericalError(format!("missing code for symbol {s}"))
            })?;
            bitstream.extend_from_slice(bits);
        }
        let total_bits = bitstream.len();

        Ok(HuffmanEncoded { bitstream, codebook, symbol_count: symbols.len(), total_bits })
    }

    /// Decode a [`HuffmanEncoded`] back to the original symbol sequence.
    pub fn decode(&self, huff: &HuffmanEncoded) -> Result<Vec<i64>, CompressionError> {
        // Invert codebook.
        let inverse: HashMap<Vec<bool>, i64> =
            huff.codebook.iter().map(|(&sym, bits)| (bits.clone(), sym)).collect();

        let mut result = Vec::with_capacity(huff.symbol_count);
        let mut buf: Vec<bool> = Vec::new();
        for &bit in &huff.bitstream {
            buf.push(bit);
            if let Some(&sym) = inverse.get(&buf) {
                result.push(sym);
                buf.clear();
            }
        }
        if !buf.is_empty() {
            return Err(CompressionError::NumericalError("trailing bits in bitstream".into()));
        }
        Ok(result)
    }

    // -- internal helpers ---------------------------------------------------

    fn build_codebook(freq: &HashMap<i64, usize>) -> HashMap<i64, Vec<bool>> {
        if freq.len() == 1 {
            // Special case: single symbol gets code `[false]`.
            let sym = *freq.keys().next().unwrap();
            let mut cb = HashMap::new();
            cb.insert(sym, vec![false]);
            return cb;
        }

        // Build a min-heap via sorted vec (simple; fine for moderate alphabets).
        let mut nodes: Vec<HuffmanNode> =
            freq.iter().map(|(&symbol, &f)| HuffmanNode::Leaf { symbol, freq: f }).collect();

        while nodes.len() > 1 {
            nodes.sort_by_key(|n| Reverse(n.freq())); // descending so pop from end
            let left = nodes.pop().unwrap();
            let right = nodes.pop().unwrap();
            nodes.push(HuffmanNode::Internal {
                freq: left.freq() + right.freq(),
                left: Box::new(left),
                right: Box::new(right),
            });
        }

        let root = nodes.into_iter().next().unwrap();
        let mut codebook = HashMap::new();
        walk(&root, &mut Vec::new(), &mut codebook);
        codebook
    }
}

fn walk(node: &HuffmanNode, prefix: &mut Vec<bool>, codebook: &mut HashMap<i64, Vec<bool>>) {
    match node {
        HuffmanNode::Leaf { symbol, .. } => {
            codebook.insert(*symbol, prefix.clone());
        }
        HuffmanNode::Internal { left, right, .. } => {
            prefix.push(false);
            walk(left, prefix, codebook);
            prefix.pop();
            prefix.push(true);
            walk(right, prefix, codebook);
            prefix.pop();
        }
    }
}

impl Default for HuffmanEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Compression analysis
// ---------------------------------------------------------------------------

/// Per-layer compressibility analysis.
#[derive(Debug, Clone)]
pub struct LayerAnalysis {
    pub layer_name: String,
    /// Fraction of near-zero weights (|w| < epsilon).
    pub sparsity: f64,
    /// Number of unique quantized symbols.
    pub unique_symbols: usize,
    /// Recommended algorithm.
    pub recommended_algorithm: CompressionAlgorithm,
    /// Estimated achievable compression ratio.
    pub estimated_ratio: f64,
}

/// Analyses per-layer compressibility and recommends settings.
#[derive(Debug)]
pub struct CompressionAnalyzer {
    /// Threshold below which a weight is considered "near-zero".
    pub sparsity_epsilon: f64,
}

impl CompressionAnalyzer {
    pub const fn new(sparsity_epsilon: f64) -> Self {
        Self { sparsity_epsilon }
    }

    /// Analyse a single layer's weights.
    pub fn analyze_layer(&self, name: &str, weights: &[f64]) -> LayerAnalysis {
        let total = weights.len().max(1) as f64;
        let near_zero = weights.iter().filter(|w| w.abs() < self.sparsity_epsilon).count();
        let sparsity = near_zero as f64 / total;

        // Count unique quantised symbols (round to 2 decimal places).
        let mut symbols: Vec<i64> = weights.iter().map(|w| (w * 100.0).round() as i64).collect();
        symbols.sort_unstable();
        symbols.dedup();
        let unique_symbols = symbols.len();

        let (recommended_algorithm, estimated_ratio) = if sparsity > 0.7 {
            (CompressionAlgorithm::Pruning, 1.0 - sparsity * 0.9)
        } else if unique_symbols < 16 {
            (CompressionAlgorithm::Huffman, 0.3)
        } else {
            (CompressionAlgorithm::LowRank, 0.5)
        };

        LayerAnalysis {
            layer_name: name.to_string(),
            sparsity,
            unique_symbols,
            recommended_algorithm,
            estimated_ratio,
        }
    }

    /// Analyse multiple layers and return recommendations for each.
    pub fn analyze_model(&self, layers: &[(&str, &[f64])]) -> Vec<LayerAnalysis> {
        layers.iter().map(|(name, weights)| self.analyze_layer(name, weights)).collect()
    }
}

impl Default for CompressionAnalyzer {
    fn default() -> Self {
        Self::new(1e-6)
    }
}

// ---------------------------------------------------------------------------
// Quality validation
// ---------------------------------------------------------------------------

/// Validates that compressed weights meet a quality threshold.
///
/// Quality is measured as `1 − (RMSE / range)` where `range` is the span of
/// the original weight values.
#[derive(Debug)]
pub struct QualityValidator {
    pub threshold: f64,
}

impl QualityValidator {
    pub const fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Compare `original` and `reconstructed` weights, returning the quality
    /// score. Returns an error if quality is below `threshold`.
    pub fn validate(
        &self,
        original: &[f64],
        reconstructed: &[f64],
    ) -> Result<f64, CompressionError> {
        if original.len() != reconstructed.len() {
            return Err(CompressionError::InvalidInput(format!(
                "length mismatch: {} vs {}",
                original.len(),
                reconstructed.len()
            )));
        }
        if original.is_empty() {
            return Err(CompressionError::InvalidInput("empty weight vectors".into()));
        }

        let mse: f64 =
            original.iter().zip(reconstructed.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
                / original.len() as f64;
        let rmse = mse.sqrt();

        let min = original.iter().copied().fold(f64::INFINITY, f64::min);
        let max = original.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1e-15);

        let quality = (1.0 - rmse / range).max(0.0);

        if quality < self.threshold {
            Err(CompressionError::QualityBelowThreshold {
                achieved: quality,
                required: self.threshold,
            })
        } else {
            Ok(quality)
        }
    }
}

// ---------------------------------------------------------------------------
// Compression report
// ---------------------------------------------------------------------------

/// Summary of a compression run.
#[derive(Debug, Clone)]
pub struct CompressionReport {
    pub algorithm: CompressionAlgorithm,
    /// Size of the original weight data in bytes (estimated as `count × 8`).
    pub original_size: usize,
    /// Size of the compressed representation in bytes.
    pub compressed_size: usize,
    /// `compressed_size / original_size`.
    pub ratio: f64,
    /// Quality score `[0, 1]` (1 = lossless).
    pub quality: f64,
    /// Per-layer reports (if a multi-layer model was compressed).
    pub layer_reports: Vec<LayerReport>,
}

/// Per-layer compression summary.
#[derive(Debug, Clone)]
pub struct LayerReport {
    pub layer_name: String,
    pub original_elements: usize,
    pub compressed_elements: usize,
    pub ratio: f64,
}

impl fmt::Display for CompressionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompressionReport(algo={}, ratio={:.3}, quality={:.4}, layers={})",
            self.algorithm,
            self.ratio,
            self.quality,
            self.layer_reports.len()
        )
    }
}

// ---------------------------------------------------------------------------
// WeightCompressor – orchestrator
// ---------------------------------------------------------------------------

/// End-to-end orchestrator: analyse → select → compress → validate → report.
#[derive(Debug)]
pub struct WeightCompressor {
    pub config: CompressionConfig,
}

impl WeightCompressor {
    pub const fn new(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Compress a flat weight vector according to the configured algorithm.
    pub fn compress(&self, weights: &[f64]) -> Result<CompressionReport, CompressionError> {
        if weights.is_empty() {
            return Err(CompressionError::InvalidInput("empty weight vector".into()));
        }

        match self.config.algorithm {
            CompressionAlgorithm::Pruning => self.compress_pruning(weights),
            CompressionAlgorithm::LowRank => self.compress_low_rank(weights),
            CompressionAlgorithm::Huffman => self.compress_huffman(weights),
            CompressionAlgorithm::Quantization => self.compress_quantization(weights),
            CompressionAlgorithm::Distillation => Err(CompressionError::AlgorithmNotApplicable(
                "distillation requires a teacher model".into(),
            )),
        }
    }

    /// Compress multiple named layers and produce a combined report.
    pub fn compress_model(
        &self,
        layers: &[(&str, &[f64])],
    ) -> Result<CompressionReport, CompressionError> {
        if layers.is_empty() {
            return Err(CompressionError::InvalidInput("no layers provided".into()));
        }

        let mut total_original = 0usize;
        let mut total_compressed = 0usize;
        let mut layer_reports = Vec::new();
        let mut all_original: Vec<f64> = Vec::new();
        let mut all_reconstructed: Vec<f64> = Vec::new();

        for (name, weights) in layers {
            let orig_elem = weights.len();
            let mut w = weights.to_vec();
            let compressed_elem = self.apply_algorithm(&mut w)?;
            total_original += orig_elem;
            total_compressed += compressed_elem;
            all_original.extend_from_slice(weights);
            all_reconstructed.extend_from_slice(&w);

            layer_reports.push(LayerReport {
                layer_name: name.to_string(),
                original_elements: orig_elem,
                compressed_elements: compressed_elem,
                ratio: compressed_elem as f64 / orig_elem.max(1) as f64,
            });
        }

        let ratio = total_compressed as f64 / total_original.max(1) as f64;
        let validator = QualityValidator::new(self.config.quality_threshold);
        let quality = validator.validate(&all_original, &all_reconstructed)?;

        Ok(CompressionReport {
            algorithm: self.config.algorithm,
            original_size: total_original * 8,
            compressed_size: total_compressed * 8,
            ratio,
            quality,
            layer_reports,
        })
    }

    // -- per-algorithm implementations --------------------------------------

    fn compress_pruning(&self, weights: &[f64]) -> Result<CompressionReport, CompressionError> {
        let mut w = weights.to_vec();
        let mut sorted: Vec<f64> = w.iter().map(|v| v.abs()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((1.0 - self.config.target_ratio) * sorted.len() as f64) as usize;
        let threshold = sorted[idx.min(sorted.len() - 1)];

        let pruner = WeightPruner::new(PruningMode::Unstructured, threshold);
        pruner.prune_flat(&mut w);

        let non_zero = w.iter().filter(|v| **v != 0.0).count();
        let original_size = weights.len() * 8;
        let compressed_size = non_zero * 8;

        let validator = QualityValidator::new(self.config.quality_threshold);
        let quality = validator.validate(weights, &w)?;

        Ok(CompressionReport {
            algorithm: CompressionAlgorithm::Pruning,
            original_size,
            compressed_size,
            ratio: compressed_size as f64 / original_size as f64,
            quality,
            layer_reports: vec![],
        })
    }

    fn compress_low_rank(&self, weights: &[f64]) -> Result<CompressionReport, CompressionError> {
        let n = (weights.len() as f64).sqrt().ceil() as usize;
        let rows = n;
        let cols = n;
        let mut padded = weights.to_vec();
        padded.resize(rows * cols, 0.0);

        let rank = ((self.config.target_ratio * n as f64).ceil() as usize).max(1);
        let decomposer = LowRankDecomposer::new(rank, 10);
        let result = decomposer.decompose(&padded, rows, cols)?;
        let reconstructed = LowRankDecomposer::reconstruct(&result);

        let original_size = weights.len() * 8;
        let compressed_elements = rows * rank + rank + rank * cols;
        let compressed_size = compressed_elements * 8;

        let validator = QualityValidator::new(self.config.quality_threshold);
        let quality = validator.validate(&padded, &reconstructed)?;

        Ok(CompressionReport {
            algorithm: CompressionAlgorithm::LowRank,
            original_size,
            compressed_size,
            ratio: compressed_size as f64 / original_size as f64,
            quality,
            layer_reports: vec![],
        })
    }

    fn compress_huffman(&self, weights: &[f64]) -> Result<CompressionReport, CompressionError> {
        let symbols: Vec<i64> = weights.iter().map(|w| (w * 1000.0).round() as i64).collect();
        let huff_enc = HuffmanEncoder::new();
        let huff_result = huff_enc.encode(&symbols)?;

        let original_bits = weights.len() * 64;
        let compressed_bits = huff_result.total_bits;

        let decoded = huff_enc.decode(&huff_result)?;
        let reconstructed: Vec<f64> = decoded.iter().map(|&s| s as f64 / 1000.0).collect();

        let validator = QualityValidator::new(self.config.quality_threshold);
        let quality = validator.validate(weights, &reconstructed)?;

        Ok(CompressionReport {
            algorithm: CompressionAlgorithm::Huffman,
            original_size: original_bits / 8,
            compressed_size: compressed_bits.div_ceil(8),
            ratio: compressed_bits as f64 / original_bits as f64,
            quality,
            layer_reports: vec![],
        })
    }

    fn compress_quantization(
        &self,
        weights: &[f64],
    ) -> Result<CompressionReport, CompressionError> {
        let min = weights.iter().copied().fold(f64::INFINITY, f64::min);
        let max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(1e-15);
        let levels = 256.0;

        let quantized: Vec<u8> =
            weights.iter().map(|w| ((w - min) / range * (levels - 1.0)).round() as u8).collect();
        let reconstructed: Vec<f64> = quantized
            .iter()
            .map(|&q| (f64::from(q) / (levels - 1.0)).mul_add(range, min))
            .collect();

        let original_size = weights.len() * 8;
        let compressed_size = weights.len(); // 1 byte per weight

        let validator = QualityValidator::new(self.config.quality_threshold);
        let quality = validator.validate(weights, &reconstructed)?;

        Ok(CompressionReport {
            algorithm: CompressionAlgorithm::Quantization,
            original_size,
            compressed_size,
            ratio: compressed_size as f64 / original_size as f64,
            quality,
            layer_reports: vec![],
        })
    }

    fn apply_algorithm(&self, weights: &mut [f64]) -> Result<usize, CompressionError> {
        match self.config.algorithm {
            CompressionAlgorithm::Pruning => {
                let mut sorted: Vec<f64> = weights.iter().map(|v| v.abs()).collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((1.0 - self.config.target_ratio) * sorted.len() as f64) as usize;
                let threshold = sorted[idx.min(sorted.len() - 1)];
                let pruner = WeightPruner::new(PruningMode::Unstructured, threshold);
                pruner.prune_flat(weights);
                let non_zero = weights.iter().filter(|v| **v != 0.0).count();
                Ok(non_zero)
            }
            CompressionAlgorithm::Quantization => {
                let min_val = weights.iter().copied().fold(f64::INFINITY, f64::min);
                let max_val = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let range = (max_val - min_val).max(1e-15);
                for w in weights.iter_mut() {
                    let q = ((*w - min_val) / range * 255.0).round();
                    *w = (q / 255.0).mul_add(range, min_val);
                }
                Ok(weights.len() / 8)
            }
            CompressionAlgorithm::LowRank | CompressionAlgorithm::Huffman => Ok(weights.len()),
            CompressionAlgorithm::Distillation => Err(CompressionError::AlgorithmNotApplicable(
                "distillation requires a teacher model".into(),
            )),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- CompressionConfig --------------------------------------------------

    #[test]
    fn test_config_default() {
        let cfg = CompressionConfig::default();
        assert_eq!(cfg.algorithm, CompressionAlgorithm::Pruning);
        assert!((cfg.target_ratio - 0.5).abs() < f64::EPSILON);
        assert!((cfg.quality_threshold - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_new() {
        let cfg = CompressionConfig::new(CompressionAlgorithm::Huffman, 0.3, 0.9);
        assert_eq!(cfg.algorithm, CompressionAlgorithm::Huffman);
        assert!((cfg.target_ratio - 0.3).abs() < f64::EPSILON);
        assert!((cfg.quality_threshold - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_clone() {
        let cfg = CompressionConfig::default();
        let cfg2 = cfg.clone();
        assert_eq!(cfg.algorithm, cfg2.algorithm);
    }

    // -- CompressionAlgorithm -----------------------------------------------

    #[test]
    fn test_algorithm_display() {
        assert_eq!(format!("{}", CompressionAlgorithm::Pruning), "Pruning");
        assert_eq!(format!("{}", CompressionAlgorithm::LowRank), "LowRank");
        assert_eq!(format!("{}", CompressionAlgorithm::Huffman), "Huffman");
        assert_eq!(format!("{}", CompressionAlgorithm::Distillation), "Distillation");
        assert_eq!(format!("{}", CompressionAlgorithm::Quantization), "Quantization");
    }

    #[test]
    fn test_algorithm_eq() {
        assert_eq!(CompressionAlgorithm::Pruning, CompressionAlgorithm::Pruning);
        assert_ne!(CompressionAlgorithm::Pruning, CompressionAlgorithm::Huffman);
    }

    #[test]
    fn test_algorithm_hash() {
        let mut map = HashMap::new();
        map.insert(CompressionAlgorithm::Pruning, "prune");
        map.insert(CompressionAlgorithm::Huffman, "huff");
        assert_eq!(map[&CompressionAlgorithm::Pruning], "prune");
    }

    #[test]
    fn test_algorithm_copy() {
        let a = CompressionAlgorithm::LowRank;
        let b = a;
        assert_eq!(a, b);
    }

    // -- CompressionError ---------------------------------------------------

    #[test]
    fn test_error_display_invalid_input() {
        let e = CompressionError::InvalidInput("bad".into());
        assert!(e.to_string().contains("invalid input"));
    }

    #[test]
    fn test_error_display_quality() {
        let e = CompressionError::QualityBelowThreshold { achieved: 0.8, required: 0.95 };
        let s = e.to_string();
        assert!(s.contains("0.8"));
        assert!(s.contains("0.95"));
    }

    #[test]
    fn test_error_display_not_applicable() {
        let e = CompressionError::AlgorithmNotApplicable("no teacher".into());
        assert!(e.to_string().contains("not applicable"));
    }

    #[test]
    fn test_error_display_numerical() {
        let e = CompressionError::NumericalError("overflow".into());
        assert!(e.to_string().contains("numerical error"));
    }

    #[test]
    fn test_error_clone_eq() {
        let e = CompressionError::InvalidInput("x".into());
        let e2 = e.clone();
        assert_eq!(e, e2);
    }

    // -- WeightPruner -------------------------------------------------------

    #[test]
    fn test_pruner_unstructured_basic() {
        let mut w = vec![0.01, 0.5, -0.02, 0.8, 0.001];
        let pruner = WeightPruner::new(PruningMode::Unstructured, 0.1);
        let sparsity = pruner.prune_flat(&mut w);
        assert!(sparsity > 0.0);
        assert_eq!(w[0], 0.0);
        assert_eq!(w[2], 0.0);
        assert_eq!(w[4], 0.0);
        assert_eq!(w[1], 0.5);
    }

    #[test]
    fn test_pruner_unstructured_no_pruning() {
        let mut w = vec![1.0, 2.0, 3.0];
        let pruner = WeightPruner::new(PruningMode::Unstructured, 0.1);
        let sparsity = pruner.prune_flat(&mut w);
        assert!((sparsity - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pruner_unstructured_all_pruned() {
        let mut w = vec![0.001, 0.002, 0.003];
        let pruner = WeightPruner::new(PruningMode::Unstructured, 1.0);
        let sparsity = pruner.prune_flat(&mut w);
        assert!((sparsity - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pruner_empty() {
        let mut w: Vec<f64> = vec![];
        let pruner = WeightPruner::new(PruningMode::Unstructured, 0.1);
        let sparsity = pruner.prune_flat(&mut w);
        assert!((sparsity - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pruner_structured_row_pruning() {
        let mut w = vec![0.01, 0.02, 0.01, 1.0, 2.0, 3.0];
        let pruner = WeightPruner::new(PruningMode::Structured, 0.1);
        let ratio = pruner.prune_matrix(&mut w, 2, 3).unwrap();
        assert!((ratio - 0.5).abs() < f64::EPSILON);
        assert_eq!(&w[0..3], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_pruner_structured_no_pruning() {
        let mut w = vec![1.0, 2.0, 3.0, 4.0];
        let pruner = WeightPruner::new(PruningMode::Structured, 0.01);
        let ratio = pruner.prune_matrix(&mut w, 2, 2).unwrap();
        assert!((ratio - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pruner_matrix_invalid_dimensions() {
        let mut w = vec![1.0, 2.0, 3.0];
        let pruner = WeightPruner::new(PruningMode::Structured, 0.1);
        let err = pruner.prune_matrix(&mut w, 2, 2).unwrap_err();
        assert!(matches!(err, CompressionError::InvalidInput(_)));
    }

    #[test]
    fn test_pruner_matrix_zero_rows() {
        let mut w: Vec<f64> = vec![];
        let pruner = WeightPruner::new(PruningMode::Structured, 0.1);
        let err = pruner.prune_matrix(&mut w, 0, 3).unwrap_err();
        assert!(matches!(err, CompressionError::InvalidInput(_)));
    }

    #[test]
    fn test_pruner_unstructured_matrix() {
        let mut w = vec![0.01, 1.0, 0.02, 2.0];
        let pruner = WeightPruner::new(PruningMode::Unstructured, 0.1);
        let ratio = pruner.prune_matrix(&mut w, 2, 2).unwrap();
        assert!((ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pruner_negative_weights() {
        let mut w = vec![-0.01, -0.5, 0.02, -0.8];
        let pruner = WeightPruner::new(PruningMode::Unstructured, 0.1);
        pruner.prune_flat(&mut w);
        assert_eq!(w[0], 0.0);
        assert_eq!(w[1], -0.5);
        assert_eq!(w[2], 0.0);
        assert_eq!(w[3], -0.8);
    }

    // -- LowRankDecomposer --------------------------------------------------

    #[test]
    fn test_lowrank_identity_like() {
        let matrix = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let decomposer = LowRankDecomposer::new(3, 20);
        let result = decomposer.decompose(&matrix, 3, 3).unwrap();
        let recon = LowRankDecomposer::reconstruct(&result);
        let max_err: f64 =
            matrix.iter().zip(recon.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(max_err < 0.1, "max error = {max_err}");
    }

    #[test]
    fn test_lowrank_rank1_matrix() {
        let matrix = vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0];
        let decomposer = LowRankDecomposer::new(1, 20);
        let result = decomposer.decompose(&matrix, 3, 2).unwrap();
        assert_eq!(result.rank, 1);
        let recon = LowRankDecomposer::reconstruct(&result);
        let max_err: f64 =
            matrix.iter().zip(recon.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(max_err < 0.5, "max error = {max_err}");
    }

    #[test]
    fn test_lowrank_compression_ratio() {
        let decomposer = LowRankDecomposer::new(2, 10);
        let matrix = vec![1.0; 16]; // 4×4
        let result = decomposer.decompose(&matrix, 4, 4).unwrap();
        assert!(result.ratio > 0.0);
    }

    #[test]
    fn test_lowrank_invalid_dimensions() {
        let decomposer = LowRankDecomposer::new(2, 10);
        let err = decomposer.decompose(&[1.0, 2.0], 3, 3).unwrap_err();
        assert!(matches!(err, CompressionError::InvalidInput(_)));
    }

    #[test]
    fn test_lowrank_zero_dimensions() {
        let decomposer = LowRankDecomposer::new(1, 5);
        let err = decomposer.decompose(&[], 0, 0).unwrap_err();
        assert!(matches!(err, CompressionError::InvalidInput(_)));
    }

    #[test]
    fn test_lowrank_rank_clamped() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let decomposer = LowRankDecomposer::new(10, 10);
        let result = decomposer.decompose(&matrix, 3, 2).unwrap();
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn test_lowrank_singular_values_nonneg() {
        let matrix = vec![3.0, 1.0, 1.0, 3.0];
        let decomposer = LowRankDecomposer::new(2, 15);
        let result = decomposer.decompose(&matrix, 2, 2).unwrap();
        for &s in &result.s {
            assert!(s >= 0.0, "singular value should be non-negative: {s}");
        }
    }

    #[test]
    fn test_lowrank_reconstruct_dimensions() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let decomposer = LowRankDecomposer::new(2, 10);
        let result = decomposer.decompose(&matrix, 2, 3).unwrap();
        let recon = LowRankDecomposer::reconstruct(&result);
        assert_eq!(recon.len(), 6);
    }

    // -- SparseEncoder (CSR) -----------------------------------------------

    #[test]
    fn test_csr_encode_basic() {
        let data = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
        let enc = SparseEncoder::new(SparseFormat::Csr);
        let sparse = enc.encode(&data, 2, 3).unwrap();
        assert_eq!(sparse.nnz, 3);
        assert_eq!(sparse.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(sparse.indices, vec![1, 0, 2]);
    }

    #[test]
    fn test_csr_roundtrip() {
        let data = vec![0.0, 5.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 9.0];
        let enc = SparseEncoder::new(SparseFormat::Csr);
        let sparse = enc.encode(&data, 3, 3).unwrap();
        let decoded = SparseEncoder::decode(&sparse);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_csr_all_zeros() {
        let data = vec![0.0; 6];
        let enc = SparseEncoder::new(SparseFormat::Csr);
        let sparse = enc.encode(&data, 2, 3).unwrap();
        assert_eq!(sparse.nnz, 0);
    }

    #[test]
    fn test_csr_all_nonzero() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let enc = SparseEncoder::new(SparseFormat::Csr);
        let sparse = enc.encode(&data, 2, 2).unwrap();
        assert_eq!(sparse.nnz, 4);
    }

    #[test]
    fn test_csr_invalid_dims() {
        let data = vec![1.0, 2.0, 3.0];
        let enc = SparseEncoder::new(SparseFormat::Csr);
        let err = enc.encode(&data, 2, 2).unwrap_err();
        assert!(matches!(err, CompressionError::InvalidInput(_)));
    }

    #[test]
    fn test_csr_pointers() {
        let data = vec![1.0, 0.0, 0.0, 2.0];
        let enc = SparseEncoder::new(SparseFormat::Csr);
        let sparse = enc.encode(&data, 2, 2).unwrap();
        assert_eq!(sparse.pointers, vec![0, 1, 2]);
    }

    // -- SparseEncoder (CSC) -----------------------------------------------

    #[test]
    fn test_csc_encode_basic() {
        let data = vec![0.0, 1.0, 2.0, 0.0];
        let enc = SparseEncoder::new(SparseFormat::Csc);
        let sparse = enc.encode(&data, 2, 2).unwrap();
        assert_eq!(sparse.nnz, 2);
        assert_eq!(sparse.values, vec![2.0, 1.0]);
        assert_eq!(sparse.indices, vec![1, 0]);
    }

    #[test]
    fn test_csc_roundtrip() {
        let data = vec![1.0, 0.0, 0.0, 3.0, 0.0, 5.0, 7.0, 0.0, 0.0];
        let enc = SparseEncoder::new(SparseFormat::Csc);
        let sparse = enc.encode(&data, 3, 3).unwrap();
        let decoded = SparseEncoder::decode(&sparse);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_csc_all_zeros() {
        let data = vec![0.0; 4];
        let enc = SparseEncoder::new(SparseFormat::Csc);
        let sparse = enc.encode(&data, 2, 2).unwrap();
        assert_eq!(sparse.nnz, 0);
    }

    #[test]
    fn test_csc_pointers() {
        let data = vec![0.0, 1.0, 2.0, 0.0];
        let enc = SparseEncoder::new(SparseFormat::Csc);
        let sparse = enc.encode(&data, 2, 2).unwrap();
        assert_eq!(sparse.pointers, vec![0, 1, 2]);
    }

    #[test]
    fn test_sparse_format_eq() {
        assert_eq!(SparseFormat::Csr, SparseFormat::Csr);
        assert_ne!(SparseFormat::Csr, SparseFormat::Csc);
    }

    // -- HuffmanEncoder -----------------------------------------------------

    #[test]
    fn test_huffman_roundtrip() {
        let symbols = vec![1, 2, 3, 1, 2, 1, 1, 3, 2];
        let enc = HuffmanEncoder::new();
        let result = enc.encode(&symbols).unwrap();
        let decoded = enc.decode(&result).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_huffman_single_symbol() {
        let symbols = vec![42, 42, 42];
        let enc = HuffmanEncoder::new();
        let result = enc.encode(&symbols).unwrap();
        assert_eq!(result.symbol_count, 3);
        let decoded = enc.decode(&result).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_huffman_two_symbols() {
        let symbols = vec![0, 1, 0, 1, 0];
        let enc = HuffmanEncoder::new();
        let result = enc.encode(&symbols).unwrap();
        assert!(result.total_bits <= symbols.len()); // 1 bit per symbol
        let decoded = enc.decode(&result).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_huffman_empty() {
        let enc = HuffmanEncoder::new();
        let err = enc.encode(&[]).unwrap_err();
        assert!(matches!(err, CompressionError::InvalidInput(_)));
    }

    #[test]
    fn test_huffman_compression_ratio() {
        let mut symbols = vec![0i64; 100];
        symbols.extend(vec![1; 10]);
        symbols.extend(vec![2; 5]);
        let enc = HuffmanEncoder::new();
        let result = enc.encode(&symbols).unwrap();
        let avg = result.total_bits as f64 / result.symbol_count as f64;
        assert!(avg < 2.0, "avg bits = {avg}");
    }

    #[test]
    fn test_huffman_codebook_prefix_free() {
        let symbols = vec![1, 2, 3, 4, 5];
        let enc = HuffmanEncoder::new();
        let result = enc.encode(&symbols).unwrap();
        let codes: Vec<&Vec<bool>> = result.codebook.values().collect();
        for (i, a) in codes.iter().enumerate() {
            for (j, b) in codes.iter().enumerate() {
                if i != j {
                    let shorter = a.len().min(b.len());
                    assert!(a[..shorter] != b[..shorter] || a.len() == b.len());
                }
            }
        }
    }

    #[test]
    fn test_huffman_default() {
        let enc = HuffmanEncoder::default();
        let result = enc.encode(&[1, 2, 3]).unwrap();
        assert_eq!(result.symbol_count, 3);
    }

    #[test]
    fn test_huffman_large_alphabet() {
        let symbols: Vec<i64> = (0..256).collect();
        let enc = HuffmanEncoder::new();
        let result = enc.encode(&symbols).unwrap();
        let decoded = enc.decode(&result).unwrap();
        assert_eq!(decoded, symbols);
    }

    // -- CompressionAnalyzer ------------------------------------------------

    #[test]
    fn test_analyzer_sparse_layer() {
        let weights: Vec<f64> = (0..100).map(|i| if i < 80 { 0.0 } else { 1.0 }).collect();
        let analyzer = CompressionAnalyzer::new(1e-6);
        let analysis = analyzer.analyze_layer("sparse_layer", &weights);
        assert!((analysis.sparsity - 0.8).abs() < f64::EPSILON);
        assert_eq!(analysis.recommended_algorithm, CompressionAlgorithm::Pruning);
    }

    #[test]
    fn test_analyzer_few_unique_symbols() {
        let weights = vec![0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.01, 0.02, 0.03, 1.0];
        let analyzer = CompressionAnalyzer::new(1e-6);
        let analysis = analyzer.analyze_layer("quant_layer", &weights);
        assert!(analysis.unique_symbols < 16);
        assert_eq!(analysis.recommended_algorithm, CompressionAlgorithm::Huffman);
    }

    #[test]
    fn test_analyzer_dense_diverse() {
        let weights: Vec<f64> = (0..100).map(|i| (i as f64) * 0.0137).collect();
        let analyzer = CompressionAnalyzer::new(1e-6);
        let analysis = analyzer.analyze_layer("dense_layer", &weights);
        assert_eq!(analysis.recommended_algorithm, CompressionAlgorithm::LowRank);
    }

    #[test]
    fn test_analyzer_model() {
        let sparse: Vec<f64> = (0..50).map(|i| if i < 40 { 0.0 } else { 1.0 }).collect();
        let dense: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
        let analyzer = CompressionAnalyzer::new(1e-6);
        let results = analyzer.analyze_model(&[("l0", &sparse), ("l1", &dense)]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].layer_name, "l0");
        assert_eq!(results[1].layer_name, "l1");
    }

    #[test]
    fn test_analyzer_default() {
        let analyzer = CompressionAnalyzer::default();
        assert!((analyzer.sparsity_epsilon - 1e-6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyzer_empty_weights() {
        let analyzer = CompressionAnalyzer::new(1e-6);
        let analysis = analyzer.analyze_layer("empty", &[]);
        assert!((analysis.sparsity - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyzer_estimated_ratio_range() {
        let weights: Vec<f64> = (0..100).map(|i| (i as f64) * 0.01).collect();
        let analyzer = CompressionAnalyzer::new(1e-6);
        let analysis = analyzer.analyze_layer("test", &weights);
        assert!(analysis.estimated_ratio > 0.0 && analysis.estimated_ratio <= 1.0);
    }

    // -- QualityValidator ---------------------------------------------------

    #[test]
    fn test_quality_perfect() {
        let original = vec![1.0, 2.0, 3.0];
        let validator = QualityValidator::new(0.95);
        let quality = validator.validate(&original, &original).unwrap();
        assert!((quality - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quality_slight_distortion() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let distorted = vec![1.01, 2.01, 3.01, 4.01];
        let validator = QualityValidator::new(0.95);
        let quality = validator.validate(&original, &distorted).unwrap();
        assert!(quality > 0.95);
    }

    #[test]
    fn test_quality_below_threshold() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let bad = vec![10.0, 20.0, 30.0, 40.0];
        let validator = QualityValidator::new(0.95);
        let err = validator.validate(&original, &bad).unwrap_err();
        assert!(matches!(err, CompressionError::QualityBelowThreshold { .. }));
    }

    #[test]
    fn test_quality_length_mismatch() {
        let validator = QualityValidator::new(0.95);
        let err = validator.validate(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(err, CompressionError::InvalidInput(_)));
    }

    #[test]
    fn test_quality_empty() {
        let validator = QualityValidator::new(0.95);
        let err = validator.validate(&[], &[]).unwrap_err();
        assert!(matches!(err, CompressionError::InvalidInput(_)));
    }

    #[test]
    fn test_quality_zero_range() {
        let original = vec![5.0, 5.0, 5.0];
        let distorted = vec![5.0, 5.0, 5.0];
        let validator = QualityValidator::new(0.95);
        let quality = validator.validate(&original, &distorted).unwrap();
        assert!((quality - 1.0).abs() < f64::EPSILON);
    }

    // -- CompressionReport --------------------------------------------------

    #[test]
    fn test_report_display() {
        let report = CompressionReport {
            algorithm: CompressionAlgorithm::Pruning,
            original_size: 1000,
            compressed_size: 500,
            ratio: 0.5,
            quality: 0.98,
            layer_reports: vec![],
        };
        let s = format!("{report}");
        assert!(s.contains("Pruning"));
        assert!(s.contains("0.500"));
    }

    #[test]
    fn test_report_clone() {
        let report = CompressionReport {
            algorithm: CompressionAlgorithm::Huffman,
            original_size: 100,
            compressed_size: 30,
            ratio: 0.3,
            quality: 0.99,
            layer_reports: vec![LayerReport {
                layer_name: "l0".into(),
                original_elements: 100,
                compressed_elements: 30,
                ratio: 0.3,
            }],
        };
        let r2 = report.clone();
        assert_eq!(r2.layer_reports.len(), 1);
    }

    // -- WeightCompressor (orchestrator) ------------------------------------

    #[test]
    fn test_compressor_pruning() {
        let weights: Vec<f64> = (0..100).map(|i| (i as f64) * 0.01).collect();
        let cfg = CompressionConfig::new(CompressionAlgorithm::Pruning, 0.5, 0.5);
        let compressor = WeightCompressor::new(cfg);
        let report = compressor.compress(&weights).unwrap();
        assert_eq!(report.algorithm, CompressionAlgorithm::Pruning);
        assert!(report.ratio <= 1.0);
    }

    #[test]
    fn test_compressor_huffman() {
        let weights = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 2.0];
        let cfg = CompressionConfig::new(CompressionAlgorithm::Huffman, 0.5, 0.5);
        let compressor = WeightCompressor::new(cfg);
        let report = compressor.compress(&weights).unwrap();
        assert_eq!(report.algorithm, CompressionAlgorithm::Huffman);
    }

    #[test]
    fn test_compressor_quantization() {
        let weights: Vec<f64> = (0..100).map(|i| (i as f64) / 100.0).collect();
        let cfg = CompressionConfig::new(CompressionAlgorithm::Quantization, 0.5, 0.9);
        let compressor = WeightCompressor::new(cfg);
        let report = compressor.compress(&weights).unwrap();
        assert_eq!(report.algorithm, CompressionAlgorithm::Quantization);
        assert!(report.ratio < 0.2);
    }

    #[test]
    fn test_compressor_lowrank() {
        let weights: Vec<f64> = (0..16).map(|i| (i as f64) * 0.1).collect();
        let cfg = CompressionConfig::new(CompressionAlgorithm::LowRank, 0.5, 0.3);
        let compressor = WeightCompressor::new(cfg);
        let report = compressor.compress(&weights).unwrap();
        assert_eq!(report.algorithm, CompressionAlgorithm::LowRank);
    }

    #[test]
    fn test_compressor_distillation_error() {
        let weights = vec![1.0, 2.0];
        let cfg = CompressionConfig::new(CompressionAlgorithm::Distillation, 0.5, 0.9);
        let compressor = WeightCompressor::new(cfg);
        let err = compressor.compress(&weights).unwrap_err();
        assert!(matches!(err, CompressionError::AlgorithmNotApplicable(_)));
    }

    #[test]
    fn test_compressor_empty_input() {
        let cfg = CompressionConfig::default();
        let compressor = WeightCompressor::new(cfg);
        let err = compressor.compress(&[]).unwrap_err();
        assert!(matches!(err, CompressionError::InvalidInput(_)));
    }

    #[test]
    fn test_compressor_model_pruning() {
        let l0: Vec<f64> = (0..50).map(|i| (i as f64) * 0.01).collect();
        let l1: Vec<f64> = (0..50).map(|i| (i as f64) * 0.02).collect();
        let cfg = CompressionConfig::new(CompressionAlgorithm::Pruning, 0.5, 0.3);
        let compressor = WeightCompressor::new(cfg);
        let report = compressor.compress_model(&[("layer0", &l0), ("layer1", &l1)]).unwrap();
        assert_eq!(report.layer_reports.len(), 2);
    }

    #[test]
    fn test_compressor_model_empty_layers() {
        let cfg = CompressionConfig::default();
        let compressor = WeightCompressor::new(cfg);
        let err = compressor.compress_model(&[]).unwrap_err();
        assert!(matches!(err, CompressionError::InvalidInput(_)));
    }

    #[test]
    fn test_compressor_model_quantization() {
        let l0: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
        let cfg = CompressionConfig::new(CompressionAlgorithm::Quantization, 0.5, 0.9);
        let compressor = WeightCompressor::new(cfg);
        let report = compressor.compress_model(&[("layer0", &l0)]).unwrap();
        assert_eq!(report.layer_reports.len(), 1);
    }

    // -- Integration: full pipeline round-trips ----------------------------

    #[test]
    fn test_pipeline_prune_then_sparse_encode() {
        let mut weights = vec![0.01, 0.5, 0.02, 0.8, 0.001, 0.9, 0.0, 0.7, 0.03];
        let pruner = WeightPruner::new(PruningMode::Unstructured, 0.1);
        pruner.prune_flat(&mut weights);

        let sparse_enc = SparseEncoder::new(SparseFormat::Csr);
        let sparse = sparse_enc.encode(&weights, 3, 3).unwrap();
        assert!(sparse.nnz < 9);

        let decoded = SparseEncoder::decode(&sparse);
        assert_eq!(decoded, weights);
    }

    #[test]
    fn test_pipeline_prune_then_huffman() {
        let mut weights = vec![0.001, 1.0, 0.002, 2.0, 0.003, 1.0];
        let pruner = WeightPruner::new(PruningMode::Unstructured, 0.1);
        pruner.prune_flat(&mut weights);

        let symbols: Vec<i64> = weights.iter().map(|w| (w * 1000.0).round() as i64).collect();
        let enc = HuffmanEncoder::new();
        let result = enc.encode(&symbols).unwrap();
        let decoded = enc.decode(&result).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_pipeline_analyze_then_compress() {
        let weights: Vec<f64> =
            (0..100).map(|i| if i < 75 { 0.0 } else { (i as f64) * 0.1 }).collect();
        let analyzer = CompressionAnalyzer::new(1e-6);
        let analysis = analyzer.analyze_layer("test", &weights);

        let cfg = CompressionConfig::new(analysis.recommended_algorithm, 0.5, 0.3);
        let compressor = WeightCompressor::new(cfg);
        let report = compressor.compress(&weights).unwrap();
        assert!(report.ratio <= 1.0);
    }

    #[test]
    fn test_pipeline_lowrank_quality_check() {
        let matrix = vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0, 16.0, 20.0, 20.0];
        let decomposer = LowRankDecomposer::new(2, 20);
        let result = decomposer.decompose(&matrix, 3, 3).unwrap();
        let recon = LowRankDecomposer::reconstruct(&result);

        let validator = QualityValidator::new(0.5);
        let quality = validator.validate(&matrix, &recon).unwrap();
        assert!(quality >= 0.5);
    }

    #[test]
    fn test_pipeline_sparse_csc_roundtrip_after_prune() {
        let mut weights = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let pruner = WeightPruner::new(PruningMode::Unstructured, 0.5);
        pruner.prune_flat(&mut weights);

        let enc = SparseEncoder::new(SparseFormat::Csc);
        let sparse = enc.encode(&weights, 3, 3).unwrap();
        let decoded = SparseEncoder::decode(&sparse);
        assert_eq!(decoded, weights);
    }

    #[test]
    fn test_sparse_negative_values() {
        let data = vec![-1.0, 0.0, 0.0, -2.0];
        let enc = SparseEncoder::new(SparseFormat::Csr);
        let sparse = enc.encode(&data, 2, 2).unwrap();
        assert_eq!(sparse.nnz, 2);
        let decoded = SparseEncoder::decode(&sparse);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_huffman_negative_symbols() {
        let symbols = vec![-3, -2, -1, 0, 1, 2, 3, -3, 0, 0];
        let enc = HuffmanEncoder::new();
        let result = enc.encode(&symbols).unwrap();
        let decoded = enc.decode(&result).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn test_lowrank_wide_matrix() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 1×6
        let decomposer = LowRankDecomposer::new(1, 10);
        let result = decomposer.decompose(&matrix, 1, 6).unwrap();
        assert_eq!(result.rank, 1);
    }

    #[test]
    fn test_lowrank_tall_matrix() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 6×1
        let decomposer = LowRankDecomposer::new(1, 10);
        let result = decomposer.decompose(&matrix, 6, 1).unwrap();
        assert_eq!(result.rank, 1);
    }

    #[test]
    fn test_compressor_report_sizes() {
        let weights: Vec<f64> = (0..64).map(|i| (i as f64) * 0.1).collect();
        let cfg = CompressionConfig::new(CompressionAlgorithm::Quantization, 0.5, 0.9);
        let compressor = WeightCompressor::new(cfg);
        let report = compressor.compress(&weights).unwrap();
        assert_eq!(report.original_size, 64 * 8);
        assert_eq!(report.compressed_size, 64);
    }

    #[test]
    fn test_pruning_mode_eq() {
        assert_eq!(PruningMode::Unstructured, PruningMode::Unstructured);
        assert_ne!(PruningMode::Unstructured, PruningMode::Structured);
    }

    #[test]
    fn test_layer_analysis_fields() {
        let analysis = LayerAnalysis {
            layer_name: "test".into(),
            sparsity: 0.5,
            unique_symbols: 10,
            recommended_algorithm: CompressionAlgorithm::LowRank,
            estimated_ratio: 0.5,
        };
        assert_eq!(analysis.layer_name, "test");
        assert_eq!(analysis.unique_symbols, 10);
    }

    #[test]
    fn test_layer_report_fields() {
        let lr = LayerReport {
            layer_name: "attn".into(),
            original_elements: 100,
            compressed_elements: 50,
            ratio: 0.5,
        };
        assert_eq!(lr.layer_name, "attn");
        assert_eq!(lr.original_elements, 100);
    }
}
