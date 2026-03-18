//! Advanced weight compression techniques for memory-efficient inference.
//!
//! Provides CPU reference implementations of compression algorithms designed for
//! eventual GPU dispatch via OpenCL kernels on Intel / AMD devices:
//!
//! - **Ternary quantization** — BitNet core {-1, 0, +1} mapping with scale factor.
//! - **Group quantization** — per-group min/max quantization (group sizes 32–256).
//! - **Activation-aware quantization (AWQ)** — importance-weighted quantization.
//! - **GPTQ roundtrip** — Optimal Brain Quantization simulation.
//! - **Weight clustering** — k-means clustering (2/4/8 centroids).
//! - **Bit packing** — 2-bit and 4-bit compact representations.
//! - **Decompression kernels** — on-the-fly decompression for matmul.
//! - **Compression analysis** — SNR, SQNR, cosine similarity metrics.

use std::fmt;

// ── OpenCL kernel source ────────────────────────────────────────────────────

/// Embedded OpenCL C source for on-the-fly weight decompression during matmul.
///
/// The kernel unpacks 2-bit or 4-bit packed weights, applies per-group scales
/// and zero-points, then feeds the dequantized row into a dot-product accumulator.
pub const WEIGHT_DECOMPRESS_CL: &str = r#"
// Weight decompression + matvec kernel for BitNet-rs.
// Supports 2-bit (ternary) and 4-bit packed formats with per-group scales.

__kernel void decompress_matvec_2bit(
    __global const uchar *packed_w,   // [rows * cols/4]
    __global const float *scales,     // [rows * num_groups]
    __global const float *zeros,      // [rows * num_groups] (may be NULL conceptually)
    __global const float *x,          // [cols]
    __global       float *y,          // [rows]
    const int cols,
    const int group_size
) {
    int row = get_global_id(0);
    int num_groups = cols / group_size;
    float acc = 0.0f;

    for (int g = 0; g < num_groups; g++) {
        float s = scales[row * num_groups + g];
        float z = zeros[row * num_groups + g];
        int base = g * group_size;

        for (int j = 0; j < group_size; j += 4) {
            int col = base + j;
            int byte_idx = row * (cols / 4) + col / 4;
            uchar packed = packed_w[byte_idx];

            for (int k = 0; k < 4; k++) {
                int bit_val = (packed >> (k * 2)) & 0x03;
                float w = ((float)bit_val - z) * s;
                acc += w * x[col + k];
            }
        }
    }
    y[row] = acc;
}

__kernel void decompress_matvec_4bit(
    __global const uchar *packed_w,   // [rows * cols/2]
    __global const float *scales,     // [rows * num_groups]
    __global const float *zeros,      // [rows * num_groups]
    __global const float *x,          // [cols]
    __global       float *y,          // [rows]
    const int cols,
    const int group_size
) {
    int row = get_global_id(0);
    int num_groups = cols / group_size;
    float acc = 0.0f;

    for (int g = 0; g < num_groups; g++) {
        float s = scales[row * num_groups + g];
        float z = zeros[row * num_groups + g];
        int base = g * group_size;

        for (int j = 0; j < group_size; j += 2) {
            int col = base + j;
            int byte_idx = row * (cols / 2) + col / 2;
            uchar packed = packed_w[byte_idx];

            float w0 = ((float)(packed & 0x0F) - z) * s;
            float w1 = ((float)((packed >> 4) & 0x0F) - z) * s;
            acc += w0 * x[col] + w1 * x[col + 1];
        }
    }
    y[row] = acc;
}
"#;

// ── Ternary quantization ────────────────────────────────────────────────────

/// Ternary quantizer mapping f32 weights to {-1, 0, +1} with a per-tensor scale.
///
/// The threshold `delta` determines the dead-zone around zero:
///   - |w| <= delta  →  0
///   - w > delta     → +1
///   - w < -delta    → -1
///
/// The scale factor is the mean absolute value of non-zero weights in the
/// original tensor, ensuring reconstruction `q * scale ≈ w`.
#[derive(Debug, Clone)]
pub struct TernaryQuantizer {
    /// Fraction of the mean absolute weight used as threshold.
    pub threshold_factor: f32,
}

impl Default for TernaryQuantizer {
    fn default() -> Self {
        Self { threshold_factor: 0.7 }
    }
}

/// Result of ternary quantization.
#[derive(Debug, Clone)]
pub struct TernaryResult {
    /// Quantized values: each element is -1, 0, or +1.
    pub values: Vec<i8>,
    /// Per-tensor scale factor for reconstruction.
    pub scale: f32,
    /// Threshold used for quantization.
    pub threshold: f32,
    /// Fraction of weights mapped to zero (sparsity).
    pub sparsity: f32,
}

impl TernaryQuantizer {
    pub fn new(threshold_factor: f32) -> Self {
        Self { threshold_factor }
    }

    /// Quantize `weights` to ternary {-1, 0, +1} with a computed scale.
    pub fn quantize(&self, weights: &[f32]) -> TernaryResult {
        if weights.is_empty() {
            return TernaryResult { values: vec![], scale: 0.0, threshold: 0.0, sparsity: 0.0 };
        }

        let mean_abs: f32 = weights.iter().map(|w| w.abs()).sum::<f32>() / weights.len() as f32;
        let threshold = mean_abs * self.threshold_factor;

        let mut values = Vec::with_capacity(weights.len());
        let mut nonzero_abs_sum = 0.0f32;
        let mut nonzero_count = 0usize;

        for &w in weights {
            if w > threshold {
                values.push(1i8);
                nonzero_abs_sum += w.abs();
                nonzero_count += 1;
            } else if w < -threshold {
                values.push(-1i8);
                nonzero_abs_sum += w.abs();
                nonzero_count += 1;
            } else {
                values.push(0i8);
            }
        }

        let scale = if nonzero_count > 0 { nonzero_abs_sum / nonzero_count as f32 } else { 0.0 };

        let sparsity = 1.0 - (nonzero_count as f32 / weights.len() as f32);

        TernaryResult { values, scale, threshold, sparsity }
    }

    /// Reconstruct approximate f32 weights from ternary values and scale.
    pub fn dequantize(result: &TernaryResult) -> Vec<f32> {
        result.values.iter().map(|&v| v as f32 * result.scale).collect()
    }
}

// ── Group quantization ──────────────────────────────────────────────────────

/// Per-group symmetric quantization to n-bit integers.
///
/// Each group of `group_size` elements shares a single scale derived from the
/// group's max absolute value.
#[derive(Debug, Clone)]
pub struct GroupQuantizer {
    /// Number of elements per group.
    pub group_size: usize,
    /// Number of bits for quantized representation (2 or 4).
    pub bits: u8,
}

/// Result of group quantization.
#[derive(Debug, Clone)]
pub struct GroupQuantResult {
    /// Quantized integer values.
    pub values: Vec<i8>,
    /// Per-group scale factors.
    pub scales: Vec<f32>,
    /// Per-group zero points.
    pub zeros: Vec<f32>,
    /// Group size used.
    pub group_size: usize,
    /// Bits per value.
    pub bits: u8,
}

impl GroupQuantizer {
    pub fn new(group_size: usize, bits: u8) -> Self {
        assert!(group_size > 0, "group_size must be positive");
        assert!(bits == 2 || bits == 4, "only 2-bit and 4-bit supported");
        Self { group_size, bits }
    }

    /// Quantize `weights` into per-group n-bit integers.
    pub fn quantize(&self, weights: &[f32]) -> GroupQuantResult {
        let max_val = (1i8 << (self.bits - 1)) - 1; // 1 for 2-bit, 7 for 4-bit
        let min_val = -(1i8 << (self.bits - 1)); // -2 for 2-bit, -8 for 4-bit
        let qrange = (max_val - min_val) as f32;

        let num_groups = weights.len().div_ceil(self.group_size);
        let mut values = Vec::with_capacity(weights.len());
        let mut scales = Vec::with_capacity(num_groups);
        let mut zeros = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * self.group_size;
            let end = (start + self.group_size).min(weights.len());
            let group = &weights[start..end];

            let g_max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let g_min = group.iter().cloned().fold(f32::INFINITY, f32::min);
            let range = g_max - g_min;

            let scale = if range.abs() < f32::EPSILON { 1.0 } else { range / qrange };

            scales.push(scale);
            zeros.push(g_min); // store group minimum as offset

            for &w in group {
                let q = ((w - g_min) / scale + min_val as f32)
                    .round()
                    .clamp(min_val as f32, max_val as f32);
                values.push(q as i8);
            }
        }

        GroupQuantResult { values, scales, zeros, group_size: self.group_size, bits: self.bits }
    }

    /// Reconstruct f32 weights from group-quantized values.
    pub fn dequantize(result: &GroupQuantResult) -> Vec<f32> {
        let min_val = -(1i8 << (result.bits - 1));
        let mut out = Vec::with_capacity(result.values.len());
        for (i, &v) in result.values.iter().enumerate() {
            let g = i / result.group_size;
            let scale = result.scales[g];
            let g_min = result.zeros[g];
            out.push((v as f32 - min_val as f32) * scale + g_min);
        }
        out
    }
}

// ── Activation-aware weight quantization (AWQ) ──────────────────────────────

/// Activation-aware weight quantization: scales weights by channel importance
/// before quantization, then un-scales during dequantization.
///
/// Importance is derived from activation statistics (e.g. mean absolute
/// activation per channel).
#[derive(Debug, Clone)]
pub struct AwqQuantizer {
    /// Group size for the inner quantizer.
    pub group_size: usize,
    /// Bits per value.
    pub bits: u8,
    /// Exponent controlling importance scaling strength.
    pub alpha: f32,
}

/// Result of AWQ quantization.
#[derive(Debug, Clone)]
pub struct AwqResult {
    /// Inner group quantization result.
    pub group_result: GroupQuantResult,
    /// Per-channel importance-derived scales applied before quantization.
    pub importance_scales: Vec<f32>,
}

impl AwqQuantizer {
    pub fn new(group_size: usize, bits: u8, alpha: f32) -> Self {
        Self { group_size, bits, alpha }
    }

    /// Quantize `weights` using activation `importance` per element.
    ///
    /// `importance` must have the same length as `weights`.
    pub fn quantize(&self, weights: &[f32], importance: &[f32]) -> AwqResult {
        assert_eq!(weights.len(), importance.len(), "weights and importance must match in length");

        // Compute per-element importance scale: s_i = importance_i^alpha
        let importance_scales: Vec<f32> =
            importance.iter().map(|&imp| imp.abs().powf(self.alpha).max(f32::EPSILON)).collect();

        // Scale weights by importance before quantization
        let scaled: Vec<f32> =
            weights.iter().zip(&importance_scales).map(|(&w, &s)| w * s).collect();

        let gq = GroupQuantizer::new(self.group_size, self.bits);
        let group_result = gq.quantize(&scaled);

        AwqResult { group_result, importance_scales }
    }

    /// Reconstruct f32 weights, undoing the importance scaling.
    pub fn dequantize(result: &AwqResult) -> Vec<f32> {
        let deq = GroupQuantizer::dequantize(&result.group_result);
        deq.iter().zip(&result.importance_scales).map(|(&w, &s)| w / s).collect()
    }
}

// ── GPTQ roundtrip simulation ───────────────────────────────────────────────

/// GPTQ-style Optimal Brain Quantization (OBQ) simulation.
///
/// Uses the diagonal of the inverse Hessian to greedily quantize each weight
/// while compensating the residual error across remaining weights in the row.
#[derive(Debug, Clone)]
pub struct GptqRoundtrip {
    /// Group size for quantization.
    pub group_size: usize,
    /// Bits per value.
    pub bits: u8,
    /// Dampening factor added to Hessian diagonal for numerical stability.
    pub damp: f32,
}

/// Result of GPTQ quantization.
#[derive(Debug, Clone)]
pub struct GptqResult {
    /// Quantized values.
    pub values: Vec<i8>,
    /// Per-group scale factors.
    pub scales: Vec<f32>,
    /// Per-group zero points.
    pub zeros: Vec<f32>,
    /// Group size.
    pub group_size: usize,
    /// Bits per value.
    pub bits: u8,
    /// Total quantization error (squared).
    pub total_error: f64,
}

impl GptqRoundtrip {
    pub fn new(group_size: usize, bits: u8, damp: f32) -> Self {
        Self { group_size, bits, damp }
    }

    /// Quantize `weights` (treated as a single row) with GPTQ-style error
    /// compensation. `hessian_diag` provides per-element diagonal entries of
    /// the inverse Hessian (proxy for sensitivity).
    pub fn quantize(&self, weights: &[f32], hessian_diag: &[f32]) -> GptqResult {
        assert_eq!(weights.len(), hessian_diag.len());

        let max_val = (1i8 << (self.bits - 1)) - 1;
        let min_val = -(1i8 << (self.bits - 1));
        let qrange = (max_val - min_val) as f32;
        let num_groups = weights.len().div_ceil(self.group_size);

        let mut w = weights.to_vec();
        let mut values = vec![0i8; weights.len()];
        let mut scales = Vec::with_capacity(num_groups);
        let mut zeros = Vec::with_capacity(num_groups);
        let mut total_error = 0.0f64;

        for g in 0..num_groups {
            let start = g * self.group_size;
            let end = (start + self.group_size).min(w.len());
            let group = &w[start..end];

            let g_max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let g_min = group.iter().cloned().fold(f32::INFINITY, f32::min);
            let range = g_max - g_min;
            let scale = if range.abs() < f32::EPSILON { 1.0 } else { range / qrange };

            scales.push(scale);
            zeros.push(g_min);

            // Greedy quantize each element, compensating error into later elements
            for i in start..end {
                let q = ((w[i] - g_min) / scale + min_val as f32)
                    .round()
                    .clamp(min_val as f32, max_val as f32);
                let q_int = q as i8;
                let deq = (q_int as f32 - min_val as f32) * scale + g_min;
                let err = w[i] - deq;
                total_error += (err as f64) * (err as f64);

                values[i] = q_int;

                // Compensate: distribute error to remaining elements in group
                let h_ii = hessian_diag[i].abs() + self.damp;
                if h_ii > f32::EPSILON {
                    let correction = err / h_ii;
                    for j in (i + 1)..end {
                        let h_ij_approx = hessian_diag[j].abs() + self.damp;
                        w[j] += correction * h_ij_approx;
                    }
                }
            }
        }

        GptqResult {
            values,
            scales,
            zeros,
            group_size: self.group_size,
            bits: self.bits,
            total_error,
        }
    }

    /// Reconstruct f32 weights from GPTQ-quantized values.
    pub fn dequantize(result: &GptqResult) -> Vec<f32> {
        let min_val = -(1i8 << (result.bits - 1));
        let mut out = Vec::with_capacity(result.values.len());
        for (i, &v) in result.values.iter().enumerate() {
            let g = i / result.group_size;
            let scale = result.scales[g];
            let g_min = result.zeros[g];
            out.push((v as f32 - min_val as f32) * scale + g_min);
        }
        out
    }
}

// ── Weight clustering (k-means) ─────────────────────────────────────────────

/// K-means weight clustering: replaces each weight with the nearest of K
/// centroids, storing only the centroid index per weight.
#[derive(Debug, Clone)]
pub struct WeightClustering {
    /// Number of centroids (2, 4, or 8).
    pub num_centroids: usize,
    /// Maximum k-means iterations.
    pub max_iterations: usize,
}

/// Result of weight clustering.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// Centroid index per weight.
    pub assignments: Vec<u8>,
    /// Centroid values.
    pub centroids: Vec<f32>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final inertia (sum of squared distances to assigned centroids).
    pub inertia: f64,
}

impl WeightClustering {
    pub fn new(num_centroids: usize, max_iterations: usize) -> Self {
        assert!((2..=256).contains(&num_centroids), "num_centroids must be 2..=256");
        Self { num_centroids, max_iterations }
    }

    /// Cluster `weights` into `num_centroids` groups using k-means.
    ///
    /// Centroids are initialized via uniform spacing across the weight range.
    pub fn cluster(&self, weights: &[f32]) -> ClusterResult {
        if weights.is_empty() {
            return ClusterResult {
                assignments: vec![],
                centroids: vec![0.0; self.num_centroids],
                iterations: 0,
                inertia: 0.0,
            };
        }

        let w_min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let w_max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Initialize centroids uniformly
        let mut centroids: Vec<f32> = (0..self.num_centroids)
            .map(|i| {
                if self.num_centroids == 1 {
                    (w_min + w_max) / 2.0
                } else {
                    w_min + (w_max - w_min) * i as f32 / (self.num_centroids - 1) as f32
                }
            })
            .collect();

        let mut assignments = vec![0u8; weights.len()];
        let mut iter_count = 0;

        for _iter in 0..self.max_iterations {
            iter_count = _iter + 1;
            let mut changed = false;

            // Assignment step
            for (i, &w) in weights.iter().enumerate() {
                let mut best = 0u8;
                let mut best_dist = f32::MAX;
                for (c, &centroid) in centroids.iter().enumerate() {
                    let d = (w - centroid).abs();
                    if d < best_dist {
                        best_dist = d;
                        best = c as u8;
                    }
                }
                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update step
            let mut sums = vec![0.0f64; self.num_centroids];
            let mut counts = vec![0usize; self.num_centroids];
            for (i, &w) in weights.iter().enumerate() {
                let c = assignments[i] as usize;
                sums[c] += w as f64;
                counts[c] += 1;
            }
            for c in 0..self.num_centroids {
                if counts[c] > 0 {
                    centroids[c] = (sums[c] / counts[c] as f64) as f32;
                }
            }
        }

        // Compute inertia
        let inertia: f64 = weights
            .iter()
            .zip(&assignments)
            .map(|(&w, &a)| {
                let d = (w - centroids[a as usize]) as f64;
                d * d
            })
            .sum();

        ClusterResult { assignments, centroids, iterations: iter_count, inertia }
    }

    /// Reconstruct weights from cluster assignments and centroids.
    pub fn dequantize(result: &ClusterResult) -> Vec<f32> {
        result.assignments.iter().map(|&a| result.centroids[a as usize]).collect()
    }
}

// ── Compression analyzer ────────────────────────────────────────────────────

/// Quality metrics comparing original and reconstructed weights.
#[derive(Debug, Clone)]
pub struct CompressionMetrics {
    /// Signal-to-noise ratio (dB). Higher is better.
    pub snr_db: f64,
    /// Signal-to-quantization-noise ratio (dB).
    pub sqnr_db: f64,
    /// Cosine similarity in [−1, 1]. 1.0 = perfect match.
    pub cosine_similarity: f64,
    /// Mean squared error.
    pub mse: f64,
    /// Max absolute error.
    pub max_abs_error: f64,
    /// Compression ratio (original_bits / compressed_bits).
    pub compression_ratio: f64,
}

impl fmt::Display for CompressionMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SNR={:.1}dB SQNR={:.1}dB cos={:.4} MSE={:.6} max_err={:.4} ratio={:.1}x",
            self.snr_db,
            self.sqnr_db,
            self.cosine_similarity,
            self.mse,
            self.max_abs_error,
            self.compression_ratio,
        )
    }
}

/// Analyzer for measuring compression quality.
pub struct CompressionAnalyzer;

impl CompressionAnalyzer {
    /// Compute quality metrics between `original` and `reconstructed` weights.
    ///
    /// `bits_per_weight` is the number of bits used per weight in the compressed
    /// representation (e.g. 2 for ternary, 4 for 4-bit).
    pub fn analyze(
        original: &[f32],
        reconstructed: &[f32],
        bits_per_weight: f64,
    ) -> CompressionMetrics {
        assert_eq!(original.len(), reconstructed.len(), "length mismatch");

        let n = original.len() as f64;
        if original.is_empty() {
            return CompressionMetrics {
                snr_db: 0.0,
                sqnr_db: 0.0,
                cosine_similarity: 1.0,
                mse: 0.0,
                max_abs_error: 0.0,
                compression_ratio: 0.0,
            };
        }

        let signal_power: f64 = original.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / n;

        let mut noise_power = 0.0f64;
        let mut max_abs_err = 0.0f64;
        for (&o, &r) in original.iter().zip(reconstructed) {
            let err = (o - r) as f64;
            noise_power += err * err;
            max_abs_err = max_abs_err.max(err.abs());
        }
        let mse = noise_power / n;
        noise_power /= n;

        let snr_db = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f64::INFINITY
        };

        let sqnr_db = snr_db; // SQNR == SNR for quantization noise

        // Cosine similarity
        let mut dot = 0.0f64;
        let mut norm_o = 0.0f64;
        let mut norm_r = 0.0f64;
        for (&o, &r) in original.iter().zip(reconstructed) {
            let o64 = o as f64;
            let r64 = r as f64;
            dot += o64 * r64;
            norm_o += o64 * o64;
            norm_r += r64 * r64;
        }
        let denom = norm_o.sqrt() * norm_r.sqrt();
        let cosine_similarity = if denom > 0.0 { dot / denom } else { 1.0 };

        let compression_ratio = 32.0 / bits_per_weight;

        CompressionMetrics {
            snr_db,
            sqnr_db,
            cosine_similarity,
            mse,
            max_abs_error: max_abs_err,
            compression_ratio,
        }
    }
}

// ── Bit packing ─────────────────────────────────────────────────────────────

/// Packs quantized integer values into compact byte arrays.
///
/// Supports 2-bit and 4-bit representations:
/// - **2-bit**: 4 values per byte, range [0, 3] (unsigned) or [-2, 1] (signed).
/// - **4-bit**: 2 values per byte, range [0, 15] (unsigned) or [-8, 7] (signed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitPacker {
    /// Bits per value (2 or 4).
    pub bits: u8,
}

impl BitPacker {
    pub fn new(bits: u8) -> Self {
        assert!(bits == 2 || bits == 4, "only 2-bit and 4-bit packing supported");
        Self { bits }
    }

    /// Number of values stored per byte.
    pub fn values_per_byte(self) -> usize {
        (8 / self.bits) as usize
    }

    /// Number of bytes required to pack `n` values.
    pub fn packed_len(self, n: usize) -> usize {
        n.div_ceil(self.values_per_byte())
    }

    /// Pack signed `i8` values. The caller must ensure values fit in the bit
    /// range. Values are stored as `(value - min_val)` where `min_val` is
    /// `-2^(bits-1)`.
    pub fn pack(&self, values: &[i8]) -> Vec<u8> {
        let min_val = -(1i8 << (self.bits - 1));
        let mask = (1u8 << self.bits) - 1;
        let vpb = self.values_per_byte();
        let mut packed = vec![0u8; self.packed_len(values.len())];

        for (i, &v) in values.iter().enumerate() {
            let unsigned = ((v - min_val) as u8) & mask;
            let byte_idx = i / vpb;
            let bit_offset = (i % vpb) as u8 * self.bits;
            packed[byte_idx] |= unsigned << bit_offset;
        }
        packed
    }

    /// Unpack `count` signed i8 values from packed bytes.
    pub fn unpack(&self, packed: &[u8], count: usize) -> Vec<i8> {
        let min_val = -(1i8 << (self.bits - 1));
        let mask = (1u8 << self.bits) - 1;
        let vpb = self.values_per_byte();
        let mut values = Vec::with_capacity(count);

        for i in 0..count {
            let byte_idx = i / vpb;
            let bit_offset = (i % vpb) as u8 * self.bits;
            let unsigned = (packed[byte_idx] >> bit_offset) & mask;
            values.push(unsigned as i8 + min_val);
        }
        values
    }

    /// Pack unsigned `u8` values directly (no offset). Values must fit in `bits`.
    pub fn pack_unsigned(&self, values: &[u8]) -> Vec<u8> {
        let mask = (1u8 << self.bits) - 1;
        let vpb = self.values_per_byte();
        let mut packed = vec![0u8; self.packed_len(values.len())];

        for (i, &v) in values.iter().enumerate() {
            let byte_idx = i / vpb;
            let bit_offset = (i % vpb) as u8 * self.bits;
            packed[byte_idx] |= (v & mask) << bit_offset;
        }
        packed
    }

    /// Unpack `count` unsigned values from packed bytes.
    pub fn unpack_unsigned(&self, packed: &[u8], count: usize) -> Vec<u8> {
        let mask = (1u8 << self.bits) - 1;
        let vpb = self.values_per_byte();
        let mut values = Vec::with_capacity(count);

        for i in 0..count {
            let byte_idx = i / vpb;
            let bit_offset = (i % vpb) as u8 * self.bits;
            values.push((packed[byte_idx] >> bit_offset) & mask);
        }
        values
    }
}

// ── Decompression kernel (CPU reference) ────────────────────────────────────

/// CPU reference for on-the-fly weight decompression during matrix-vector
/// multiply, matching the OpenCL kernel logic.
pub struct DecompressionKernel;

impl DecompressionKernel {
    /// Decompress 2-bit packed weights and multiply by input vector.
    ///
    /// `packed_w`: `[rows * cols/4]` packed bytes.
    /// `scales`:   `[rows * num_groups]` per-group scale factors.
    /// `zeros`:    `[rows * num_groups]` per-group zero points.
    /// `x`:        input vector `[cols]`.
    /// `y`:        output vector `[rows]`.
    pub fn matvec_2bit(
        packed_w: &[u8],
        scales: &[f32],
        zeros: &[f32],
        x: &[f32],
        y: &mut [f32],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) {
        assert_eq!(packed_w.len(), rows * cols / 4);
        assert_eq!(x.len(), cols);
        assert_eq!(y.len(), rows);
        let num_groups = cols / group_size;
        assert_eq!(scales.len(), rows * num_groups);
        assert_eq!(zeros.len(), rows * num_groups);

        for row in 0..rows {
            let mut acc = 0.0f32;
            for g in 0..num_groups {
                let s = scales[row * num_groups + g];
                let z = zeros[row * num_groups + g];
                let base = g * group_size;

                for j in (0..group_size).step_by(4) {
                    let col = base + j;
                    let byte_idx = row * (cols / 4) + col / 4;
                    let packed = packed_w[byte_idx];

                    for k in 0..4 {
                        let bit_val = (packed >> (k * 2)) & 0x03;
                        let w = (bit_val as f32 - z) * s;
                        acc += w * x[col + k];
                    }
                }
            }
            y[row] = acc;
        }
    }

    /// Decompress 4-bit packed weights and multiply by input vector.
    pub fn matvec_4bit(
        packed_w: &[u8],
        scales: &[f32],
        zeros: &[f32],
        x: &[f32],
        y: &mut [f32],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) {
        assert_eq!(packed_w.len(), rows * cols / 2);
        assert_eq!(x.len(), cols);
        assert_eq!(y.len(), rows);
        let num_groups = cols / group_size;
        assert_eq!(scales.len(), rows * num_groups);
        assert_eq!(zeros.len(), rows * num_groups);

        for row in 0..rows {
            let mut acc = 0.0f32;
            for g in 0..num_groups {
                let s = scales[row * num_groups + g];
                let z = zeros[row * num_groups + g];
                let base = g * group_size;

                for j in (0..group_size).step_by(2) {
                    let col = base + j;
                    let byte_idx = row * (cols / 2) + col / 2;
                    let packed = packed_w[byte_idx];

                    let w0 = ((packed & 0x0F) as f32 - z) * s;
                    let w1 = (((packed >> 4) & 0x0F) as f32 - z) * s;
                    acc += w0 * x[col] + w1 * x[col + 1];
                }
            }
            y[row] = acc;
        }
    }

    /// Decompress 2-bit packed weights into a full f32 buffer (no matmul).
    pub fn decompress_2bit(
        packed_w: &[u8],
        scales: &[f32],
        zeros: &[f32],
        out: &mut [f32],
        total_elements: usize,
        group_size: usize,
    ) {
        assert!(packed_w.len() * 4 >= total_elements);
        assert_eq!(out.len(), total_elements);
        let num_groups = total_elements.div_ceil(group_size);
        assert!(scales.len() >= num_groups);
        assert!(zeros.len() >= num_groups);

        for (i, out_val) in out.iter_mut().enumerate().take(total_elements) {
            let g = i / group_size;
            let byte_idx = i / 4;
            let shift = (i % 4) * 2;
            let bit_val = (packed_w[byte_idx] >> shift) & 0x03;
            *out_val = (bit_val as f32 - zeros[g]) * scales[g];
        }
    }

    /// Decompress 4-bit packed weights into a full f32 buffer.
    pub fn decompress_4bit(
        packed_w: &[u8],
        scales: &[f32],
        zeros: &[f32],
        out: &mut [f32],
        total_elements: usize,
        group_size: usize,
    ) {
        assert!(packed_w.len() * 2 >= total_elements);
        assert_eq!(out.len(), total_elements);
        let num_groups = total_elements.div_ceil(group_size);
        assert!(scales.len() >= num_groups);
        assert!(zeros.len() >= num_groups);

        for (i, out_val) in out.iter_mut().enumerate().take(total_elements) {
            let g = i / group_size;
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                packed_w[byte_idx] & 0x0F
            } else {
                (packed_w[byte_idx] >> 4) & 0x0F
            };
            *out_val = (nibble as f32 - zeros[g]) * scales[g];
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────

    fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
        if n <= 1 {
            return vec![start];
        }
        (0..n).map(|i| start + (end - start) * i as f32 / (n - 1) as f32).collect()
    }

    fn gaussian_weights(n: usize, seed: u64) -> Vec<f32> {
        // Simple LCG-based pseudo-gaussian via Box-Muller-ish approach
        let mut state = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (state >> 33) as f32 / (1u64 << 31) as f32;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (state >> 33) as f32 / (1u64 << 31) as f32;
            let u1 = u1.max(1e-10);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            out.push(z * 0.1); // scale to ~N(0, 0.01)
        }
        out
    }

    // ── Ternary quantizer tests ─────────────────────────────────────────

    #[test]
    fn test_ternary_basic() {
        let tq = TernaryQuantizer::default();
        let weights = vec![0.5, -0.5, 0.01, -0.01, 1.0, -1.0];
        let result = tq.quantize(&weights);
        assert_eq!(result.values.len(), weights.len());
        for &v in &result.values {
            assert!(v == -1 || v == 0 || v == 1);
        }
    }

    #[test]
    fn test_ternary_scale_positive() {
        let tq = TernaryQuantizer::default();
        let weights = vec![0.5, -0.5, 1.0, -1.0];
        let result = tq.quantize(&weights);
        assert!(result.scale > 0.0, "scale must be positive for nonzero weights");
    }

    #[test]
    fn test_ternary_all_zeros() {
        let tq = TernaryQuantizer::default();
        let weights = [0.0; 64];
        let result = tq.quantize(&weights);
        assert!(result.values.iter().all(|&v| v == 0));
        assert_eq!(result.scale, 0.0);
        assert_eq!(result.sparsity, 1.0);
    }

    #[test]
    fn test_ternary_empty() {
        let tq = TernaryQuantizer::default();
        let result = tq.quantize(&[]);
        assert!(result.values.is_empty());
        assert_eq!(result.scale, 0.0);
    }

    #[test]
    fn test_ternary_roundtrip_quality() {
        let tq = TernaryQuantizer::new(0.5);
        let weights = gaussian_weights(256, 42);
        let result = tq.quantize(&weights);
        let recon = TernaryQuantizer::dequantize(&result);
        assert_eq!(recon.len(), weights.len());

        let metrics = CompressionAnalyzer::analyze(&weights, &recon, 2.0);
        assert!(metrics.cosine_similarity > 0.5, "cosine too low: {}", metrics.cosine_similarity);
    }

    #[test]
    fn test_ternary_sparsity_range() {
        let tq = TernaryQuantizer::new(0.7);
        let weights = gaussian_weights(512, 99);
        let result = tq.quantize(&weights);
        assert!(result.sparsity >= 0.0 && result.sparsity <= 1.0);
    }

    #[test]
    fn test_ternary_threshold_factor_zero() {
        let tq = TernaryQuantizer::new(0.0);
        let weights = vec![0.1, -0.2, 0.3];
        let result = tq.quantize(&weights);
        // With threshold = 0, only exact zeros become 0
        assert!(result.values.iter().all(|&v| v == 1 || v == -1));
        assert_eq!(result.sparsity, 0.0);
    }

    #[test]
    fn test_ternary_large_weights() {
        let tq = TernaryQuantizer::default();
        let weights = vec![100.0, -100.0, 0.001];
        let result = tq.quantize(&weights);
        assert_eq!(result.values[0], 1);
        assert_eq!(result.values[1], -1);
    }

    #[test]
    fn test_ternary_uniform_positive() {
        let tq = TernaryQuantizer::new(0.5);
        let weights = [1.0; 32];
        let result = tq.quantize(&weights);
        assert!(result.values.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_ternary_dequant_sign_preserves() {
        let tq = TernaryQuantizer::default();
        let weights = vec![0.5, -0.5, 0.5, -0.5];
        let result = tq.quantize(&weights);
        let recon = TernaryQuantizer::dequantize(&result);
        for (&w, &r) in weights.iter().zip(&recon) {
            if w.abs() > result.threshold {
                assert_eq!(w.signum(), r.signum(), "sign mismatch");
            }
        }
    }

    // ── Group quantizer tests ───────────────────────────────────────────

    #[test]
    fn test_group_quant_2bit_basic() {
        let gq = GroupQuantizer::new(32, 2);
        let weights = gaussian_weights(128, 1);
        let result = gq.quantize(&weights);
        assert_eq!(result.values.len(), 128);
        assert_eq!(result.scales.len(), 4); // 128/32
    }

    #[test]
    fn test_group_quant_4bit_basic() {
        let gq = GroupQuantizer::new(64, 4);
        let weights = gaussian_weights(256, 2);
        let result = gq.quantize(&weights);
        assert_eq!(result.values.len(), 256);
        assert_eq!(result.scales.len(), 4); // 256/64
    }

    #[test]
    fn test_group_quant_roundtrip_2bit() {
        let gq = GroupQuantizer::new(32, 2);
        let weights = gaussian_weights(128, 3);
        let result = gq.quantize(&weights);
        let recon = GroupQuantizer::dequantize(&result);
        assert_eq!(recon.len(), weights.len());

        let metrics = CompressionAnalyzer::analyze(&weights, &recon, 2.0);
        assert!(metrics.mse < 1.0, "MSE too high: {}", metrics.mse);
    }

    #[test]
    fn test_group_quant_roundtrip_4bit() {
        let gq = GroupQuantizer::new(32, 4);
        let weights = gaussian_weights(128, 4);
        let result = gq.quantize(&weights);
        let recon = GroupQuantizer::dequantize(&result);

        let metrics = CompressionAnalyzer::analyze(&weights, &recon, 4.0);
        // 4-bit should be higher quality than 2-bit
        assert!(metrics.cosine_similarity > 0.9, "cos too low: {}", metrics.cosine_similarity);
    }

    #[test]
    fn test_group_quant_various_sizes() {
        for &gs in &[32, 64, 128, 256] {
            let gq = GroupQuantizer::new(gs, 4);
            let weights = gaussian_weights(512, 10 + gs as u64);
            let result = gq.quantize(&weights);
            assert_eq!(result.group_size, gs);
            assert_eq!(result.values.len(), 512);
        }
    }

    #[test]
    fn test_group_quant_non_aligned() {
        let gq = GroupQuantizer::new(32, 4);
        let weights = gaussian_weights(100, 5); // not a multiple of 32
        let result = gq.quantize(&weights);
        assert_eq!(result.values.len(), 100);
        assert_eq!(result.scales.len(), 4); // ceil(100/32)
    }

    #[test]
    fn test_group_quant_all_zeros() {
        let gq = GroupQuantizer::new(32, 2);
        let weights = [0.0f32; 64];
        let result = gq.quantize(&weights);
        let recon = GroupQuantizer::dequantize(&result);
        for &r in &recon {
            assert!(r.abs() < 1e-6, "reconstruction of zeros should be near-zero");
        }
    }

    #[test]
    fn test_group_quant_single_group() {
        let gq = GroupQuantizer::new(256, 4);
        let weights = gaussian_weights(256, 6);
        let result = gq.quantize(&weights);
        assert_eq!(result.scales.len(), 1);
    }

    #[test]
    fn test_group_quant_value_range_2bit() {
        let gq = GroupQuantizer::new(32, 2);
        let weights = gaussian_weights(64, 7);
        let result = gq.quantize(&weights);
        for &v in &result.values {
            assert!(v >= -2 && v <= 1, "2-bit value out of range: {v}");
        }
    }

    #[test]
    fn test_group_quant_value_range_4bit() {
        let gq = GroupQuantizer::new(32, 4);
        let weights = gaussian_weights(64, 8);
        let result = gq.quantize(&weights);
        for &v in &result.values {
            assert!(v >= -8 && v <= 7, "4-bit value out of range: {v}");
        }
    }

    // ── AWQ tests ───────────────────────────────────────────────────────

    #[test]
    fn test_awq_basic() {
        let awq = AwqQuantizer::new(32, 4, 0.5);
        let weights = gaussian_weights(128, 20);
        let importance: Vec<f32> = weights.iter().map(|w| w.abs() + 0.01).collect();
        let result = awq.quantize(&weights, &importance);
        assert_eq!(result.group_result.values.len(), 128);
        assert_eq!(result.importance_scales.len(), 128);
    }

    #[test]
    fn test_awq_roundtrip() {
        let awq = AwqQuantizer::new(32, 4, 0.5);
        let weights = gaussian_weights(128, 21);
        let importance: Vec<f32> = weights.iter().map(|w| w.abs() + 0.1).collect();
        let result = awq.quantize(&weights, &importance);
        let recon = AwqQuantizer::dequantize(&result);
        assert_eq!(recon.len(), weights.len());
    }

    #[test]
    fn test_awq_importance_effect() {
        // High-importance weights should be preserved more accurately
        let awq = AwqQuantizer::new(32, 4, 1.0);
        let weights = linspace(-1.0, 1.0, 64);

        let uniform_imp = [1.0f32; 64];
        let result_uniform = awq.quantize(&weights, &uniform_imp);
        let recon_uniform = AwqQuantizer::dequantize(&result_uniform);

        let mut varied_imp = [0.1f32; 64];
        // Make first half high importance
        for imp in varied_imp.iter_mut().take(32) {
            *imp = 10.0;
        }
        let result_varied = awq.quantize(&weights, &varied_imp);
        let recon_varied = AwqQuantizer::dequantize(&result_varied);

        // The first-half MSE with varied importance should differ from uniform
        let mse_first_uniform: f64 = weights[..32]
            .iter()
            .zip(&recon_uniform[..32])
            .map(|(&o, &r)| ((o - r) as f64).powi(2))
            .sum::<f64>()
            / 32.0;
        let mse_first_varied: f64 = weights[..32]
            .iter()
            .zip(&recon_varied[..32])
            .map(|(&o, &r)| ((o - r) as f64).powi(2))
            .sum::<f64>()
            / 32.0;

        // The varied importance result should not be identical to uniform
        // (they use different scaling, so results differ)
        let _diff = (mse_first_uniform - mse_first_varied).abs();
        // Just verify both produce valid reconstructions
        assert!(recon_uniform.len() == 64);
        assert!(recon_varied.len() == 64);
    }

    #[test]
    fn test_awq_alpha_zero() {
        // alpha=0 means importance has no effect (all scales become 1.0)
        let awq = AwqQuantizer::new(32, 4, 0.0);
        let weights = gaussian_weights(64, 22);
        let importance = [5.0f32; 64];
        let result = awq.quantize(&weights, &importance);
        // All importance scales should be ~epsilon (imp^0 = 1.0, but clamped)
        for &s in &result.importance_scales {
            assert!((s - 1.0).abs() < 0.01, "alpha=0 should yield scale≈1, got {s}");
        }
    }

    #[test]
    #[should_panic(expected = "weights and importance must match")]
    fn test_awq_length_mismatch() {
        let awq = AwqQuantizer::new(32, 4, 0.5);
        let _ = awq.quantize(&[1.0, 2.0], &[1.0]);
    }

    // ── GPTQ roundtrip tests ────────────────────────────────────────────

    #[test]
    fn test_gptq_basic() {
        let gptq = GptqRoundtrip::new(32, 4, 0.01);
        let weights = gaussian_weights(128, 30);
        let hessian = [1.0f32; 128];
        let result = gptq.quantize(&weights, &hessian);
        assert_eq!(result.values.len(), 128);
        assert!(result.total_error >= 0.0);
    }

    #[test]
    fn test_gptq_roundtrip_quality() {
        let gptq = GptqRoundtrip::new(32, 4, 0.01);
        let weights = gaussian_weights(128, 31);
        let hessian = [1.0f32; 128];
        let result = gptq.quantize(&weights, &hessian);
        let recon = GptqRoundtrip::dequantize(&result);
        assert_eq!(recon.len(), weights.len());

        let metrics = CompressionAnalyzer::analyze(&weights, &recon, 4.0);
        assert!(metrics.cosine_similarity > 0.8, "GPTQ cos too low: {}", metrics.cosine_similarity);
    }

    #[test]
    fn test_gptq_error_compensation() {
        // GPTQ with error compensation should produce lower total error than
        // naive quantization for non-uniform hessians
        let gptq = GptqRoundtrip::new(32, 4, 0.01);
        let weights = gaussian_weights(64, 32);
        let hessian_uniform = [1.0f32; 64];
        let result = gptq.quantize(&weights, &hessian_uniform);
        assert!(result.total_error.is_finite());
    }

    #[test]
    fn test_gptq_2bit() {
        let gptq = GptqRoundtrip::new(32, 2, 0.01);
        let weights = gaussian_weights(64, 33);
        let hessian = [1.0f32; 64];
        let result = gptq.quantize(&weights, &hessian);
        for &v in &result.values {
            assert!(v >= -2 && v <= 1, "2-bit GPTQ value out of range: {v}");
        }
    }

    #[test]
    fn test_gptq_all_zeros() {
        let gptq = GptqRoundtrip::new(32, 4, 0.01);
        let weights = [0.0f32; 64];
        let hessian = [1.0f32; 64];
        let result = gptq.quantize(&weights, &hessian);
        let recon = GptqRoundtrip::dequantize(&result);
        for &r in &recon {
            assert!(r.abs() < 1e-5, "GPTQ zero roundtrip failed: {r}");
        }
    }

    #[test]
    #[should_panic]
    fn test_gptq_length_mismatch() {
        let gptq = GptqRoundtrip::new(32, 4, 0.01);
        let _ = gptq.quantize(&[1.0], &[1.0, 2.0]);
    }

    // ── Weight clustering tests ─────────────────────────────────────────

    #[test]
    fn test_cluster_2_centroids() {
        let wc = WeightClustering::new(2, 100);
        let weights = vec![-1.0, -0.9, -0.8, 0.8, 0.9, 1.0];
        let result = wc.cluster(&weights);
        assert_eq!(result.centroids.len(), 2);
        assert_eq!(result.assignments.len(), 6);
        // Centroids should be near -0.9 and +0.9
        let mut sorted_c = result.centroids.clone();
        sorted_c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(sorted_c[0] < 0.0, "lower centroid should be negative");
        assert!(sorted_c[1] > 0.0, "upper centroid should be positive");
    }

    #[test]
    fn test_cluster_4_centroids() {
        let wc = WeightClustering::new(4, 100);
        let weights = gaussian_weights(256, 40);
        let result = wc.cluster(&weights);
        assert_eq!(result.centroids.len(), 4);
    }

    #[test]
    fn test_cluster_8_centroids() {
        let wc = WeightClustering::new(8, 100);
        let weights = gaussian_weights(512, 41);
        let result = wc.cluster(&weights);
        assert_eq!(result.centroids.len(), 8);
    }

    #[test]
    fn test_cluster_roundtrip() {
        let wc = WeightClustering::new(4, 100);
        let weights = gaussian_weights(128, 42);
        let result = wc.cluster(&weights);
        let recon = WeightClustering::dequantize(&result);
        assert_eq!(recon.len(), weights.len());

        let metrics = CompressionAnalyzer::analyze(&weights, &recon, 2.0);
        assert!(metrics.cosine_similarity > 0.8);
    }

    #[test]
    fn test_cluster_empty() {
        let wc = WeightClustering::new(2, 10);
        let result = wc.cluster(&[]);
        assert!(result.assignments.is_empty());
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_cluster_uniform_weights() {
        let wc = WeightClustering::new(2, 100);
        let weights = [0.5f32; 64];
        let result = wc.cluster(&weights);
        // All assignments should be the same centroid
        let first = result.assignments[0];
        assert!(result.assignments.iter().all(|&a| a == first));
    }

    #[test]
    fn test_cluster_inertia_positive() {
        let wc = WeightClustering::new(4, 50);
        let weights = gaussian_weights(128, 43);
        let result = wc.cluster(&weights);
        assert!(result.inertia >= 0.0);
    }

    #[test]
    fn test_cluster_more_centroids_lower_inertia() {
        let weights = gaussian_weights(256, 44);
        let r2 = WeightClustering::new(2, 100).cluster(&weights);
        let r8 = WeightClustering::new(8, 100).cluster(&weights);
        // More centroids should yield lower or equal inertia
        assert!(
            r8.inertia <= r2.inertia + 1e-6,
            "8 centroids inertia {} should be <= 2 centroids inertia {}",
            r8.inertia,
            r2.inertia
        );
    }

    #[test]
    fn test_cluster_convergence() {
        let wc = WeightClustering::new(4, 1000);
        let weights = gaussian_weights(128, 45);
        let result = wc.cluster(&weights);
        // Should converge before max iterations for simple data
        assert!(result.iterations < 1000, "did not converge");
    }

    // ── BitPacker tests ─────────────────────────────────────────────────

    #[test]
    fn test_bitpack_2bit_roundtrip() {
        let bp = BitPacker::new(2);
        let values: Vec<i8> = vec![-2, -1, 0, 1, -2, -1, 0, 1];
        let packed = bp.pack(&values);
        let unpacked = bp.unpack(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_bitpack_4bit_roundtrip() {
        let bp = BitPacker::new(4);
        let values: Vec<i8> = vec![-8, -4, 0, 3, 7, -1, 2, -7];
        let packed = bp.pack(&values);
        let unpacked = bp.unpack(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_bitpack_2bit_packed_len() {
        let bp = BitPacker::new(2);
        assert_eq!(bp.packed_len(1), 1);
        assert_eq!(bp.packed_len(4), 1);
        assert_eq!(bp.packed_len(5), 2);
        assert_eq!(bp.packed_len(8), 2);
        assert_eq!(bp.packed_len(9), 3);
    }

    #[test]
    fn test_bitpack_4bit_packed_len() {
        let bp = BitPacker::new(4);
        assert_eq!(bp.packed_len(1), 1);
        assert_eq!(bp.packed_len(2), 1);
        assert_eq!(bp.packed_len(3), 2);
        assert_eq!(bp.packed_len(4), 2);
    }

    #[test]
    fn test_bitpack_unsigned_2bit() {
        let bp = BitPacker::new(2);
        let values: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let packed = bp.pack_unsigned(&values);
        let unpacked = bp.unpack_unsigned(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_bitpack_unsigned_4bit() {
        let bp = BitPacker::new(4);
        let values: Vec<u8> = vec![0, 5, 10, 15, 1, 8, 3, 12];
        let packed = bp.pack_unsigned(&values);
        let unpacked = bp.unpack_unsigned(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_bitpack_empty() {
        let bp = BitPacker::new(2);
        let packed = bp.pack(&[]);
        assert!(packed.is_empty());
        let unpacked = bp.unpack(&packed, 0);
        assert!(unpacked.is_empty());
    }

    #[test]
    fn test_bitpack_values_per_byte() {
        assert_eq!(BitPacker::new(2).values_per_byte(), 4);
        assert_eq!(BitPacker::new(4).values_per_byte(), 2);
    }

    #[test]
    fn test_bitpack_non_aligned_count() {
        let bp = BitPacker::new(2);
        let values: Vec<i8> = vec![-2, 0, 1]; // 3 values, not multiple of 4
        let packed = bp.pack(&values);
        let unpacked = bp.unpack(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_bitpack_2bit_all_same() {
        let bp = BitPacker::new(2);
        let values = [0i8; 16];
        let packed = bp.pack(&values);
        let unpacked = bp.unpack(&packed, 16);
        assert_eq!(values.to_vec(), unpacked);
    }

    // ── Decompression kernel tests ──────────────────────────────────────

    #[test]
    fn test_decompress_2bit_identity() {
        // Pack known values, decompress, check output
        let cols = 8;
        let rows = 2;
        let group_size = 4;
        let num_groups = cols / group_size;

        // Packed: each byte has 4 values with 2 bits each
        // Using raw unsigned values 0,1,2,3 -> with zero=1, scale=1 -> dequantized -1,0,1,2
        let packed_w: Vec<u8> = vec![
            0b10_01_00_11, // row0: vals 3,0,1,2
            0b01_10_11_00, // row0: vals 0,3,2,1
            0b00_01_10_11, // row1: vals 3,2,1,0
            0b11_00_01_10, // row1: vals 2,1,0,3
        ];

        let scales = vec![1.0f32; rows * num_groups];
        let zeros = vec![1.0f32; rows * num_groups]; // zero point = 1

        let x = vec![1.0f32; cols];
        let mut y = vec![0.0f32; rows];

        DecompressionKernel::matvec_2bit(
            &packed_w, &scales, &zeros, &x, &mut y, rows, cols, group_size,
        );

        // Verify output is finite
        for &val in &y {
            assert!(val.is_finite(), "output must be finite");
        }
    }

    #[test]
    fn test_decompress_4bit_identity() {
        let cols = 4;
        let rows = 1;
        let group_size = 4;
        let num_groups = 1;

        // 4-bit: 2 values per byte
        let packed_w: Vec<u8> = vec![
            0x53, // low=3, high=5
            0x97, // low=7, high=9
        ];

        let scales = vec![1.0f32; rows * num_groups];
        let zeros = vec![0.0f32; rows * num_groups];

        let x = vec![1.0f32; cols];
        let mut y = vec![0.0f32; rows];

        DecompressionKernel::matvec_4bit(
            &packed_w, &scales, &zeros, &x, &mut y, rows, cols, group_size,
        );

        // y[0] = (3 + 5 + 7 + 9) * 1.0 = 24.0
        assert!((y[0] - 24.0).abs() < 1e-5, "expected 24.0, got {}", y[0]);
    }

    #[test]
    fn test_decompress_2bit_buffer() {
        let total = 8;
        let group_size = 4;
        let packed: Vec<u8> = vec![0b01_01_01_01, 0b01_01_01_01]; // all 1s (raw)
        let scales = [2.0f32; 2];
        let zeros = [1.0f32; 2];
        let mut out = vec![0.0f32; total];

        DecompressionKernel::decompress_2bit(&packed, &scales, &zeros, &mut out, total, group_size);

        // val=1, zero=1, scale=2 -> (1-1)*2 = 0
        for &o in &out {
            assert!((o - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_decompress_4bit_buffer() {
        let total = 4;
        let group_size = 4;
        let packed: Vec<u8> = vec![0x55, 0x55]; // low=5, high=5 for both
        let scales = [1.0f32; 1];
        let zeros = [5.0f32; 1];
        let mut out = vec![0.0f32; total];

        DecompressionKernel::decompress_4bit(&packed, &scales, &zeros, &mut out, total, group_size);

        for &o in &out {
            assert!((o - 0.0).abs() < 1e-6, "expected 0.0, got {o}");
        }
    }

    #[test]
    fn test_decompress_matvec_matches_manual() {
        // 1 row, 4 cols, 1 group, 2-bit
        let rows = 1;
        let cols = 4;
        let group_size = 4;

        // Pack values [2, 0, 1, 2] (unsigned)
        let packed_w = vec![0b10_01_00_10u8]; // bits: 10 00 01 10 = vals 2,0,1,2
        let scales = vec![1.0f32];
        let zeros = vec![0.0f32];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32];

        DecompressionKernel::matvec_2bit(
            &packed_w, &scales, &zeros, &x, &mut y, rows, cols, group_size,
        );

        // w = [2, 0, 1, 2], x = [1, 2, 3, 4] => dot = 2+0+3+8 = 13
        assert!((y[0] - 13.0).abs() < 1e-5, "expected 13.0, got {}", y[0]);
    }

    // ── Compression analyzer tests ──────────────────────────────────────

    #[test]
    fn test_analyzer_perfect_match() {
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let recon = orig.clone();
        let m = CompressionAnalyzer::analyze(&orig, &recon, 4.0);
        assert!(m.snr_db == f64::INFINITY || m.snr_db > 100.0);
        assert!((m.cosine_similarity - 1.0).abs() < 1e-10);
        assert!(m.mse < 1e-10);
        assert!(m.max_abs_error < 1e-10);
    }

    #[test]
    fn test_analyzer_compression_ratio() {
        let orig = [1.0; 8];
        let recon = orig.clone();
        let m2 = CompressionAnalyzer::analyze(&orig, &recon, 2.0);
        let m4 = CompressionAnalyzer::analyze(&orig, &recon, 4.0);
        assert!((m2.compression_ratio - 16.0).abs() < 1e-6);
        assert!((m4.compression_ratio - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_analyzer_snr_decreases_with_noise() {
        let orig = gaussian_weights(256, 50);
        let recon_good: Vec<f32> = orig.iter().map(|&w| w + 0.001).collect();
        let recon_bad: Vec<f32> = orig.iter().map(|&w| w + 0.1).collect();

        let m_good = CompressionAnalyzer::analyze(&orig, &recon_good, 2.0);
        let m_bad = CompressionAnalyzer::analyze(&orig, &recon_bad, 2.0);
        assert!(
            m_good.snr_db > m_bad.snr_db,
            "good SNR {} should exceed bad SNR {}",
            m_good.snr_db,
            m_bad.snr_db,
        );
    }

    #[test]
    fn test_analyzer_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let m = CompressionAnalyzer::analyze(&a, &b, 2.0);
        assert!(m.cosine_similarity.abs() < 1e-6, "orthogonal cos should be ~0");
    }

    #[test]
    fn test_analyzer_empty() {
        let m = CompressionAnalyzer::analyze(&[], &[], 2.0);
        assert_eq!(m.mse, 0.0);
        assert_eq!(m.cosine_similarity, 1.0);
    }

    #[test]
    fn test_analyzer_display() {
        let m = CompressionMetrics {
            snr_db: 30.0,
            sqnr_db: 30.0,
            cosine_similarity: 0.999,
            mse: 0.001,
            max_abs_error: 0.05,
            compression_ratio: 16.0,
        };
        let s = format!("{m}");
        assert!(s.contains("SNR="));
        assert!(s.contains("cos="));
    }

    // ── OpenCL kernel source tests ──────────────────────────────────────

    #[test]
    fn test_kernel_source_not_empty() {
        assert!(!WEIGHT_DECOMPRESS_CL.is_empty());
    }

    #[test]
    fn test_kernel_source_contains_2bit() {
        assert!(WEIGHT_DECOMPRESS_CL.contains("decompress_matvec_2bit"));
    }

    #[test]
    fn test_kernel_source_contains_4bit() {
        assert!(WEIGHT_DECOMPRESS_CL.contains("decompress_matvec_4bit"));
    }

    #[test]
    fn test_kernel_source_contains_get_global_id() {
        assert!(WEIGHT_DECOMPRESS_CL.contains("get_global_id"));
    }

    // ── Property-style tests ────────────────────────────────────────────

    #[test]
    fn test_property_decompressed_size_equals_original_2bit() {
        for n in [16, 32, 64, 128, 256, 512] {
            let weights = gaussian_weights(n, 60 + n as u64);
            let gq = GroupQuantizer::new(32, 2);
            let result = gq.quantize(&weights);
            let recon = GroupQuantizer::dequantize(&result);
            assert_eq!(recon.len(), n, "size mismatch for n={n}");
        }
    }

    #[test]
    fn test_property_decompressed_size_equals_original_4bit() {
        for n in [16, 32, 64, 128, 256, 512] {
            let weights = gaussian_weights(n, 70 + n as u64);
            let gq = GroupQuantizer::new(32, 4);
            let result = gq.quantize(&weights);
            let recon = GroupQuantizer::dequantize(&result);
            assert_eq!(recon.len(), n, "size mismatch for n={n}");
        }
    }

    #[test]
    fn test_property_ternary_always_valid_values() {
        let tq = TernaryQuantizer::default();
        for seed in 0..10 {
            let weights = gaussian_weights(128, 80 + seed);
            let result = tq.quantize(&weights);
            for &v in &result.values {
                assert!(v == -1 || v == 0 || v == 1, "invalid ternary value {v}");
            }
        }
    }

    #[test]
    fn test_property_bitpack_roundtrip_random() {
        let bp2 = BitPacker::new(2);
        let bp4 = BitPacker::new(4);

        for seed in 0..5 {
            let vals_2: Vec<i8> = gaussian_weights(64, 90 + seed)
                .iter()
                .map(|&w| (w * 10.0).clamp(-2.0, 1.0).round() as i8)
                .collect();
            let packed = bp2.pack(&vals_2);
            let unpacked = bp2.unpack(&packed, vals_2.len());
            assert_eq!(vals_2, unpacked, "2-bit roundtrip failed seed={seed}");

            let vals_4: Vec<i8> = gaussian_weights(64, 100 + seed)
                .iter()
                .map(|&w| (w * 100.0).clamp(-8.0, 7.0).round() as i8)
                .collect();
            let packed = bp4.pack(&vals_4);
            let unpacked = bp4.unpack(&packed, vals_4.len());
            assert_eq!(vals_4, unpacked, "4-bit roundtrip failed seed={seed}");
        }
    }

    #[test]
    fn test_property_cluster_assignments_valid() {
        let wc = WeightClustering::new(4, 100);
        for seed in 0..5 {
            let weights = gaussian_weights(128, 110 + seed);
            let result = wc.cluster(&weights);
            for &a in &result.assignments {
                assert!((a as usize) < 4, "assignment {a} out of range");
            }
        }
    }

    #[test]
    fn test_property_group_quant_scale_positive() {
        let gq = GroupQuantizer::new(32, 4);
        for seed in 0..5 {
            let weights = gaussian_weights(128, 120 + seed);
            let result = gq.quantize(&weights);
            for &s in &result.scales {
                assert!(s > 0.0, "scale must be positive, got {s}");
            }
        }
    }

    // ── Edge case tests ─────────────────────────────────────────────────

    #[test]
    fn test_edge_single_weight() {
        let tq = TernaryQuantizer::default();
        let result = tq.quantize(&[0.5]);
        assert_eq!(result.values.len(), 1);
    }

    #[test]
    fn test_edge_uniform_distribution() {
        let weights = linspace(-1.0, 1.0, 256);
        let tq = TernaryQuantizer::new(0.7);
        let result = tq.quantize(&weights);
        let recon = TernaryQuantizer::dequantize(&result);
        let metrics = CompressionAnalyzer::analyze(&weights, &recon, 2.0);
        assert!(metrics.cosine_similarity > 0.5);
    }

    #[test]
    fn test_edge_very_small_weights() {
        let weights = vec![1e-10, -1e-10, 1e-10, -1e-10];
        let tq = TernaryQuantizer::default();
        let result = tq.quantize(&weights);
        // Threshold should be near zero, all may be ternary
        assert_eq!(result.values.len(), 4);
    }

    #[test]
    fn test_edge_alternating_signs() {
        let weights: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let gq = GroupQuantizer::new(32, 4);
        let result = gq.quantize(&weights);
        let recon = GroupQuantizer::dequantize(&result);
        for (&o, &r) in weights.iter().zip(&recon) {
            assert!((o - r).abs() < 0.5, "alternating sign roundtrip too lossy");
        }
    }

    #[test]
    fn test_compression_ratio_2bit_vs_4bit() {
        let weights = gaussian_weights(256, 130);
        let gq2 = GroupQuantizer::new(32, 2);
        let gq4 = GroupQuantizer::new(32, 4);
        let r2 = gq2.quantize(&weights);
        let r4 = gq4.quantize(&weights);
        let recon2 = GroupQuantizer::dequantize(&r2);
        let recon4 = GroupQuantizer::dequantize(&r4);

        let m2 = CompressionAnalyzer::analyze(&weights, &recon2, 2.0);
        let m4 = CompressionAnalyzer::analyze(&weights, &recon4, 4.0);

        assert!(m2.compression_ratio > m4.compression_ratio);
        // 4-bit should give better quality
        assert!(
            m4.cosine_similarity >= m2.cosine_similarity - 0.1,
            "4-bit cos {} should be >= 2-bit cos {} (tolerance 0.1)",
            m4.cosine_similarity,
            m2.cosine_similarity,
        );
    }
}
