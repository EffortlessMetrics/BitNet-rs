//! CPU SIMD vector quantization with product quantization (PQ) and residual
//! vector quantization (RVQ).
//!
//! Provides codebook-based weight compression for fast approximate nearest
//! neighbour search and compact storage of high-dimensional vectors.

use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tuning knobs for the PQ / RVQ pipeline.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Number of sub-vector partitions (PQ sub-spaces).
    pub num_subvectors: usize,
    /// Bits per quantization code (codebook size = 2^bits_per_code).
    pub bits_per_code: u32,
    /// K-means iterations used when training codebooks.
    pub training_iterations: usize,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self { num_subvectors: 8, bits_per_code: 8, training_iterations: 20 }
    }
}

impl QuantizationConfig {
    /// Number of centroids per sub-codebook.
    #[inline]
    pub fn num_centroids(&self) -> usize {
        1usize << self.bits_per_code
    }

    /// Validate that the configuration is internally consistent.
    pub fn validate(&self) -> Result<(), VectorQuantError> {
        if self.num_subvectors == 0 {
            return Err(VectorQuantError::InvalidConfig("num_subvectors must be > 0".into()));
        }
        if self.bits_per_code == 0 || self.bits_per_code > 16 {
            return Err(VectorQuantError::InvalidConfig("bits_per_code must be in 1..=16".into()));
        }
        if self.training_iterations == 0 {
            return Err(VectorQuantError::InvalidConfig("training_iterations must be > 0".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the vector-quantization subsystem.
#[derive(Debug)]
pub enum VectorQuantError {
    /// Configuration is invalid.
    InvalidConfig(String),
    /// Input dimensions are incompatible.
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
    /// Codebook has not been trained yet.
    CodebookNotTrained,
    EmptyInput,
}

impl fmt::Display for VectorQuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::CodebookNotTrained => write!(f, "codebook not trained"),
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for VectorQuantError {}

// ---------------------------------------------------------------------------
// Distance metrics (scalar fallback + SIMD)
// ---------------------------------------------------------------------------

/// Distance metric selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    L2,
    InnerProduct,
}

/// Squared L2 distance – scalar fallback.
#[inline]
fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Negative inner product (lower = more similar).
#[inline]
fn ip_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

/// SIMD-accelerated squared L2 distance (x86-64 AVX2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // SAFETY: caller guarantees AVX2+FMA availability via runtime detection.
    unsafe {
        let n = a.len();
        let chunks = n / 8;
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }

        // Horizontal sum of the 8 lanes.
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        let mut total = _mm_cvtss_f32(result);

        // Remainder elements.
        for i in (chunks * 8)..n {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            total += d * d;
        }
        total
    }
}

/// SIMD-accelerated negative inner product (x86-64 AVX2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn ip_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // SAFETY: caller guarantees AVX2+FMA availability via runtime detection.
    unsafe {
        let n = a.len();
        let chunks = n / 8;
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        let mut total = _mm_cvtss_f32(result);

        for i in (chunks * 8)..n {
            total += *a.get_unchecked(i) * *b.get_unchecked(i);
        }
        -total
    }
}

/// Runtime-dispatched distance computation.
#[inline]
pub fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe {
                match metric {
                    DistanceMetric::L2 => l2_distance_avx2(a, b),
                    DistanceMetric::InnerProduct => ip_distance_avx2(a, b),
                }
            };
        }
    }
    match metric {
        DistanceMetric::L2 => l2_distance_scalar(a, b),
        DistanceMetric::InnerProduct => ip_distance_scalar(a, b),
    }
}

// ---------------------------------------------------------------------------
// Codebook
// ---------------------------------------------------------------------------

/// A single sub-codebook: `num_centroids` vectors of `sub_dim` floats stored
/// contiguously in row-major order.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Flat centroid storage: `centroids[i * sub_dim .. (i+1) * sub_dim]`.
    pub centroids: Vec<f32>,
    /// Dimensionality of each centroid.
    pub sub_dim: usize,
    /// Number of centroids (= 2^bits_per_code).
    pub num_centroids: usize,
}

impl Codebook {
    /// Create an uninitialised codebook.
    pub fn new(num_centroids: usize, sub_dim: usize) -> Self {
        Self { centroids: vec![0.0; num_centroids * sub_dim], sub_dim, num_centroids }
    }

    /// Return a slice for centroid `idx`.
    #[inline]
    pub fn centroid(&self, idx: usize) -> &[f32] {
        let start = idx * self.sub_dim;
        &self.centroids[start..start + self.sub_dim]
    }

    /// Return a mutable slice for centroid `idx`.
    #[inline]
    pub fn centroid_mut(&mut self, idx: usize) -> &mut [f32] {
        let start = idx * self.sub_dim;
        &mut self.centroids[start..start + self.sub_dim]
    }

    /// Find nearest centroid to `query` under the given metric.
    pub fn nearest(&self, query: &[f32], metric: DistanceMetric) -> (usize, f32) {
        debug_assert_eq!(query.len(), self.sub_dim);
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;
        for i in 0..self.num_centroids {
            let d = compute_distance(query, self.centroid(i), metric);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        (best_idx, best_dist)
    }

    /// Precompute an asymmetric distance lookup table (ADC) for a single
    /// query sub-vector. `table[c]` = distance(query, centroid_c).
    pub fn precompute_adc_table(&self, query: &[f32], metric: DistanceMetric) -> Vec<f32> {
        debug_assert_eq!(query.len(), self.sub_dim);
        (0..self.num_centroids).map(|c| compute_distance(query, self.centroid(c), metric)).collect()
    }

    /// Train from a set of sub-vectors using k-means.
    pub fn train(
        &mut self,
        sub_vectors: &[f32],
        metric: DistanceMetric,
        iterations: usize,
    ) -> Result<(), VectorQuantError> {
        let n = sub_vectors.len() / self.sub_dim;
        if n == 0 {
            return Err(VectorQuantError::EmptyInput);
        }

        // Initialise centroids with evenly-spaced samples.
        let step = n.max(1) / self.num_centroids.max(1);
        for c in 0..self.num_centroids {
            let src_idx = (c * step).min(n - 1);
            let src = &sub_vectors[src_idx * self.sub_dim..(src_idx + 1) * self.sub_dim];
            self.centroid_mut(c).copy_from_slice(src);
        }

        let mut assignments = vec![0usize; n];

        for _iter in 0..iterations {
            // Assign each vector to its nearest centroid.
            for i in 0..n {
                let vec_slice = &sub_vectors[i * self.sub_dim..(i + 1) * self.sub_dim];
                let (idx, _) = self.nearest(vec_slice, metric);
                assignments[i] = idx;
            }

            // Recompute centroids as mean of assigned vectors.
            let mut sums = vec![0.0f32; self.num_centroids * self.sub_dim];
            let mut counts = vec![0usize; self.num_centroids];

            for (i, assignment) in assignments.iter().enumerate() {
                let c = *assignment;
                counts[c] += 1;
                let base = c * self.sub_dim;
                let vec_base = i * self.sub_dim;
                for d in 0..self.sub_dim {
                    sums[base + d] += sub_vectors[vec_base + d];
                }
            }

            for (c, &count) in counts.iter().enumerate() {
                if count > 0 {
                    let inv = 1.0 / count as f32;
                    let base = c * self.sub_dim;
                    for d in 0..self.sub_dim {
                        self.centroids[base + d] = sums[base + d] * inv;
                    }
                }
            }
        }
        Ok(())
    }

    /// Update a single centroid in-place.
    pub fn update_centroid(&mut self, idx: usize, values: &[f32]) {
        assert_eq!(values.len(), self.sub_dim);
        self.centroid_mut(idx).copy_from_slice(values);
    }
}

// ---------------------------------------------------------------------------
// VectorQuantizer – PQ + RVQ
// ---------------------------------------------------------------------------

/// Product-quantisation + optional residual vector quantisation.
#[derive(Debug, Clone)]
pub struct VectorQuantizer {
    /// One codebook per sub-vector partition.
    pub codebooks: Vec<Codebook>,
    /// Full vector dimensionality.
    pub dim: usize,
    /// Configuration snapshot.
    pub config: QuantizationConfig,
    /// Distance metric used throughout.
    pub metric: DistanceMetric,
    /// Whether the codebooks have been trained.
    pub trained: bool,
    /// Optional residual codebooks (one per RVQ stage).
    pub residual_codebooks: Vec<Vec<Codebook>>,
}

impl VectorQuantizer {
    /// Create a new, untrained quantizer.
    pub fn new(
        dim: usize,
        config: QuantizationConfig,
        metric: DistanceMetric,
    ) -> Result<Self, VectorQuantError> {
        config.validate()?;
        if dim == 0 || !dim.is_multiple_of(config.num_subvectors) {
            return Err(VectorQuantError::InvalidConfig(format!(
                "dim ({dim}) must be a positive multiple of num_subvectors ({})",
                config.num_subvectors,
            )));
        }
        let sub_dim = dim / config.num_subvectors;
        let num_centroids = config.num_centroids();
        let codebooks =
            (0..config.num_subvectors).map(|_| Codebook::new(num_centroids, sub_dim)).collect();
        Ok(Self { codebooks, dim, config, metric, trained: false, residual_codebooks: Vec::new() })
    }

    /// Dimensionality of each sub-vector.
    #[inline]
    pub fn sub_dim(&self) -> usize {
        self.dim / self.config.num_subvectors
    }

    /// Train all sub-codebooks from a flat matrix of training vectors
    /// (row-major, `n_vectors * dim` elements).
    pub fn train(&mut self, data: &[f32]) -> Result<(), VectorQuantError> {
        let n = data.len() / self.dim;
        if n == 0 {
            return Err(VectorQuantError::EmptyInput);
        }
        if !data.len().is_multiple_of(self.dim) {
            return Err(VectorQuantError::DimensionMismatch {
                expected: self.dim,
                got: data.len() % self.dim,
            });
        }

        let sub_dim = self.sub_dim();

        for (m, cb) in self.codebooks.iter_mut().enumerate() {
            // Extract sub-vectors for this partition.
            let mut sub_vecs = Vec::with_capacity(n * sub_dim);
            for i in 0..n {
                let offset = i * self.dim + m * sub_dim;
                sub_vecs.extend_from_slice(&data[offset..offset + sub_dim]);
            }
            cb.train(&sub_vecs, self.metric, self.config.training_iterations)?;
        }
        self.trained = true;
        Ok(())
    }

    /// Encode a single vector into PQ codes (one code per sub-vector).
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u16>, VectorQuantError> {
        if !self.trained {
            return Err(VectorQuantError::CodebookNotTrained);
        }
        if vector.len() != self.dim {
            return Err(VectorQuantError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        let sub_dim = self.sub_dim();
        let codes: Vec<u16> = self
            .codebooks
            .iter()
            .enumerate()
            .map(|(m, cb)| {
                let sub = &vector[m * sub_dim..(m + 1) * sub_dim];
                let (idx, _) = cb.nearest(sub, self.metric);
                idx as u16
            })
            .collect();
        Ok(codes)
    }

    /// Decode PQ codes back to an approximate vector.
    pub fn decode(&self, codes: &[u16]) -> Result<Vec<f32>, VectorQuantError> {
        if !self.trained {
            return Err(VectorQuantError::CodebookNotTrained);
        }
        if codes.len() != self.config.num_subvectors {
            return Err(VectorQuantError::DimensionMismatch {
                expected: self.config.num_subvectors,
                got: codes.len(),
            });
        }
        let mut out = Vec::with_capacity(self.dim);
        for (m, &code) in codes.iter().enumerate() {
            out.extend_from_slice(self.codebooks[m].centroid(code as usize));
        }
        debug_assert_eq!(out.len(), self.dim);
        Ok(out)
    }

    /// Compute the reconstruction error (squared L2) of encoding `vector`.
    pub fn reconstruction_error(&self, vector: &[f32]) -> Result<f32, VectorQuantError> {
        let codes = self.encode(vector)?;
        let recon = self.decode(&codes)?;
        Ok(l2_distance_scalar(vector, &recon))
    }

    /// Asymmetric distance computation using precomputed ADC tables.
    /// Returns distance from `query` to the vector represented by `codes`.
    pub fn asymmetric_distance(&self, adc_tables: &[Vec<f32>], codes: &[u16]) -> f32 {
        debug_assert_eq!(adc_tables.len(), self.config.num_subvectors);
        debug_assert_eq!(codes.len(), self.config.num_subvectors);
        adc_tables.iter().zip(codes.iter()).map(|(table, &c)| table[c as usize]).sum()
    }

    /// Build ADC lookup tables for a query vector.
    pub fn precompute_adc(&self, query: &[f32]) -> Result<Vec<Vec<f32>>, VectorQuantError> {
        if query.len() != self.dim {
            return Err(VectorQuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let sub_dim = self.sub_dim();
        let tables: Vec<Vec<f32>> = self
            .codebooks
            .iter()
            .enumerate()
            .map(|(m, cb)| {
                let sub_query = &query[m * sub_dim..(m + 1) * sub_dim];
                cb.precompute_adc_table(sub_query, self.metric)
            })
            .collect();
        Ok(tables)
    }

    // -----------------------------------------------------------------------
    // Residual Vector Quantization (RVQ)
    // -----------------------------------------------------------------------

    /// Add one RVQ stage: trains codebooks on the residual between the
    /// original data and the current reconstruction.
    pub fn add_residual_stage(&mut self, data: &[f32]) -> Result<(), VectorQuantError> {
        if !self.trained {
            return Err(VectorQuantError::CodebookNotTrained);
        }
        let n = data.len() / self.dim;
        if n == 0 {
            return Err(VectorQuantError::EmptyInput);
        }

        // Compute current residuals.
        let mut residuals = Vec::with_capacity(data.len());
        for i in 0..n {
            let vec = &data[i * self.dim..(i + 1) * self.dim];
            let codes = self.encode(vec)?;
            let recon = self.decode(&codes)?;
            for (v, r) in vec.iter().zip(recon.iter()) {
                residuals.push(v - r);
            }
        }

        // Also subtract any previous residual reconstructions.
        for stage_cbs in &self.residual_codebooks {
            let sub_dim = self.sub_dim();
            for i in 0..n {
                let res_vec = &residuals[i * self.dim..(i + 1) * self.dim].to_vec();
                for (m, cb) in stage_cbs.iter().enumerate() {
                    let sub = &res_vec[m * sub_dim..(m + 1) * sub_dim];
                    let (idx, _) = cb.nearest(sub, self.metric);
                    let centroid = cb.centroid(idx);
                    let base = i * self.dim + m * sub_dim;
                    for d in 0..sub_dim {
                        residuals[base + d] -= centroid[d];
                    }
                }
            }
        }

        // Train a new set of sub-codebooks on the residuals.
        let sub_dim = self.sub_dim();
        let num_centroids = self.config.num_centroids();
        let mut stage_codebooks: Vec<Codebook> = (0..self.config.num_subvectors)
            .map(|_| Codebook::new(num_centroids, sub_dim))
            .collect();

        for (m, cb) in stage_codebooks.iter_mut().enumerate() {
            let mut sub_vecs = Vec::with_capacity(n * sub_dim);
            for i in 0..n {
                let offset = i * self.dim + m * sub_dim;
                sub_vecs.extend_from_slice(&residuals[offset..offset + sub_dim]);
            }
            cb.train(&sub_vecs, self.metric, self.config.training_iterations)?;
        }

        self.residual_codebooks.push(stage_codebooks);
        Ok(())
    }

    /// Encode a single vector through all RVQ stages.
    /// Returns `(pq_codes, vec_of_residual_codes)`.
    pub fn encode_with_residuals(
        &self,
        vector: &[f32],
    ) -> Result<(Vec<u16>, Vec<Vec<u16>>), VectorQuantError> {
        let pq_codes = self.encode(vector)?;
        let mut recon = self.decode(&pq_codes)?;

        let sub_dim = self.sub_dim();
        let mut residual_codes = Vec::with_capacity(self.residual_codebooks.len());

        for stage_cbs in &self.residual_codebooks {
            // Compute residual.
            let residual: Vec<f32> = vector.iter().zip(recon.iter()).map(|(v, r)| v - r).collect();

            let codes: Vec<u16> = stage_cbs
                .iter()
                .enumerate()
                .map(|(m, cb)| {
                    let sub = &residual[m * sub_dim..(m + 1) * sub_dim];
                    let (idx, _) = cb.nearest(sub, self.metric);
                    idx as u16
                })
                .collect();

            // Update reconstruction.
            for (m, &c) in codes.iter().enumerate() {
                let centroid = stage_cbs[m].centroid(c as usize);
                for d in 0..sub_dim {
                    recon[m * sub_dim + d] += centroid[d];
                }
            }
            residual_codes.push(codes);
        }

        Ok((pq_codes, residual_codes))
    }

    /// Decode PQ + RVQ codes back into an approximate vector.
    pub fn decode_with_residuals(
        &self,
        pq_codes: &[u16],
        residual_codes: &[Vec<u16>],
    ) -> Result<Vec<f32>, VectorQuantError> {
        let mut recon = self.decode(pq_codes)?;
        let sub_dim = self.sub_dim();

        for (stage_idx, codes) in residual_codes.iter().enumerate() {
            if stage_idx >= self.residual_codebooks.len() {
                break;
            }
            let stage_cbs = &self.residual_codebooks[stage_idx];
            for (m, &c) in codes.iter().enumerate() {
                let centroid = stage_cbs[m].centroid(c as usize);
                for d in 0..sub_dim {
                    recon[m * sub_dim + d] += centroid[d];
                }
            }
        }
        Ok(recon)
    }

    /// Total reconstruction error with all RVQ stages included.
    pub fn total_reconstruction_error(&self, vector: &[f32]) -> Result<f32, VectorQuantError> {
        let (pq, rvq) = self.encode_with_residuals(vector)?;
        let recon = self.decode_with_residuals(&pq, &rvq)?;
        Ok(l2_distance_scalar(vector, &recon))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(nsub: usize, bits: u32, iters: usize) -> QuantizationConfig {
        QuantizationConfig { num_subvectors: nsub, bits_per_code: bits, training_iterations: iters }
    }

    /// Deterministic training data: `n` vectors of dimension `dim`.
    fn training_data(n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|i| ((i * 7 + 3) % 101) as f32 / 100.0).collect()
    }

    // -- QuantizationConfig ------------------------------------------------

    #[test]
    fn config_default_is_valid() {
        QuantizationConfig::default().validate().unwrap();
    }

    #[test]
    fn config_zero_subvectors_rejected() {
        let c = make_config(0, 8, 10);
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_zero_bits_rejected() {
        let c = make_config(4, 0, 10);
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_too_many_bits_rejected() {
        let c = make_config(4, 17, 10);
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_zero_iterations_rejected() {
        let c = make_config(4, 8, 0);
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_num_centroids() {
        let c = make_config(4, 4, 10);
        assert_eq!(c.num_centroids(), 16);
    }

    // -- Codebook ----------------------------------------------------------

    #[test]
    fn codebook_creation_and_dims() {
        let cb = Codebook::new(16, 4);
        assert_eq!(cb.num_centroids, 16);
        assert_eq!(cb.sub_dim, 4);
        assert_eq!(cb.centroids.len(), 64);
    }

    #[test]
    fn codebook_centroid_access() {
        let mut cb = Codebook::new(4, 2);
        cb.update_centroid(2, &[1.0, 2.0]);
        assert_eq!(cb.centroid(2), &[1.0, 2.0]);
    }

    #[test]
    fn codebook_nearest_l2() {
        let mut cb = Codebook::new(4, 2);
        cb.update_centroid(0, &[0.0, 0.0]);
        cb.update_centroid(1, &[1.0, 0.0]);
        cb.update_centroid(2, &[0.0, 1.0]);
        cb.update_centroid(3, &[1.0, 1.0]);

        let (idx, _) = cb.nearest(&[0.9, 0.1], DistanceMetric::L2);
        assert_eq!(idx, 1);
    }

    #[test]
    fn codebook_nearest_ip() {
        let mut cb = Codebook::new(3, 2);
        cb.update_centroid(0, &[1.0, 0.0]);
        cb.update_centroid(1, &[0.0, 1.0]);
        cb.update_centroid(2, &[0.7, 0.7]);

        // query mostly aligns with centroid 2.
        let (idx, _) = cb.nearest(&[0.6, 0.6], DistanceMetric::InnerProduct);
        assert_eq!(idx, 2);
    }

    #[test]
    fn codebook_adc_table_length() {
        let cb = Codebook::new(8, 3);
        let tbl = cb.precompute_adc_table(&[1.0, 2.0, 3.0], DistanceMetric::L2);
        assert_eq!(tbl.len(), 8);
    }

    #[test]
    fn codebook_train_does_not_panic() {
        let mut cb = Codebook::new(4, 2);
        let data: Vec<f32> = (0..20).map(|i| (i as f32) / 10.0).collect();
        cb.train(&data, DistanceMetric::L2, 5).unwrap();
    }

    // -- Distance functions ------------------------------------------------

    #[test]
    fn l2_distance_zero_for_identical() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert!((compute_distance(&v, &v, DistanceMetric::L2)).abs() < 1e-6);
    }

    #[test]
    fn l2_distance_known_value() {
        let a = vec![0.0; 4];
        let b = vec![1.0; 4];
        let d = compute_distance(&a, &b, DistanceMetric::L2);
        assert!((d - 4.0).abs() < 1e-5);
    }

    #[test]
    fn ip_distance_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        let d = compute_distance(&a, &b, DistanceMetric::InnerProduct);
        assert!(d.abs() < 1e-6); // dot product 0 → distance 0
    }

    #[test]
    fn ip_distance_negative_for_similar() {
        // Both point the same direction → large positive dot → negative
        // distance (lower = more similar).
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let d = compute_distance(&a, &a, DistanceMetric::InnerProduct);
        assert!(d < 0.0);
    }

    #[test]
    fn simd_scalar_parity_l2() {
        let a: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..32).map(|i| (i as f32 + 0.5) * 0.1).collect();
        let scalar = l2_distance_scalar(&a, &b);
        let dispatched = compute_distance(&a, &b, DistanceMetric::L2);
        assert!((scalar - dispatched).abs() < 1e-3, "scalar={scalar} dispatched={dispatched}");
    }

    #[test]
    fn simd_scalar_parity_ip() {
        let a: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..32).map(|i| (i as f32 + 0.5) * 0.1).collect();
        let scalar = ip_distance_scalar(&a, &b);
        let dispatched = compute_distance(&a, &b, DistanceMetric::InnerProduct);
        assert!((scalar - dispatched).abs() < 1e-3, "scalar={scalar} dispatched={dispatched}");
    }

    // -- VectorQuantizer (PQ) ----------------------------------------------

    #[test]
    fn quantizer_creation() {
        let cfg = make_config(4, 4, 5);
        let vq = VectorQuantizer::new(16, cfg, DistanceMetric::L2).unwrap();
        assert_eq!(vq.sub_dim(), 4);
        assert!(!vq.trained);
    }

    #[test]
    fn quantizer_dim_not_multiple_rejected() {
        let cfg = make_config(3, 4, 5);
        assert!(VectorQuantizer::new(10, cfg, DistanceMetric::L2).is_err());
    }

    #[test]
    fn pq_encode_before_train_fails() {
        let cfg = make_config(4, 4, 5);
        let vq = VectorQuantizer::new(16, cfg, DistanceMetric::L2).unwrap();
        let v = vec![0.0; 16];
        assert!(vq.encode(&v).is_err());
    }

    #[test]
    fn pq_train_encode_decode_roundtrip() {
        let dim = 16;
        let cfg = make_config(4, 2, 10);
        let mut vq = VectorQuantizer::new(dim, cfg, DistanceMetric::L2).unwrap();
        let data = training_data(64, dim);
        vq.train(&data).unwrap();

        // Pick one training vector and check reconstruction error is bounded.
        let vec = &data[0..dim];
        let codes = vq.encode(vec).unwrap();
        let recon = vq.decode(&codes).unwrap();
        assert_eq!(recon.len(), dim);

        let err = l2_distance_scalar(vec, &recon);
        // With 4 centroids per sub-codebook, error should be modest.
        assert!(err < 10.0, "reconstruction error too large: {err}");
    }

    #[test]
    fn pq_wrong_dim_rejected() {
        let cfg = make_config(4, 2, 5);
        let mut vq = VectorQuantizer::new(16, cfg, DistanceMetric::L2).unwrap();
        let data = training_data(32, 16);
        vq.train(&data).unwrap();
        assert!(vq.encode(&vec![0.0; 8]).is_err());
    }

    #[test]
    fn pq_asymmetric_distance_consistent() {
        let dim = 16;
        let cfg = make_config(4, 2, 10);
        let mut vq = VectorQuantizer::new(dim, cfg, DistanceMetric::L2).unwrap();
        let data = training_data(64, dim);
        vq.train(&data).unwrap();

        let query = &data[0..dim];
        let codes = vq.encode(&data[dim..2 * dim]).unwrap();
        let tables = vq.precompute_adc(query).unwrap();
        let adc_dist = vq.asymmetric_distance(&tables, &codes);

        // ADC should approximate the symmetric distance.
        let recon = vq.decode(&codes).unwrap();
        let sym_dist = l2_distance_scalar(query, &recon);
        assert!((adc_dist - sym_dist).abs() < 1e-3, "adc={adc_dist} sym={sym_dist}");
    }

    // -- Residual Vector Quantization (RVQ) --------------------------------

    #[test]
    fn rvq_reduces_reconstruction_error() {
        let dim = 16;
        let cfg = make_config(4, 2, 10);
        let mut vq = VectorQuantizer::new(dim, cfg, DistanceMetric::L2).unwrap();
        let data = training_data(64, dim);
        vq.train(&data).unwrap();

        let vec = &data[0..dim];
        let err_pq = vq.reconstruction_error(vec).unwrap();

        vq.add_residual_stage(&data).unwrap();

        let err_rvq = vq.total_reconstruction_error(vec).unwrap();
        assert!(
            err_rvq <= err_pq + 1e-6,
            "RVQ should not increase error: pq={err_pq} rvq={err_rvq}"
        );
    }

    #[test]
    fn rvq_encode_decode_roundtrip() {
        let dim = 16;
        let cfg = make_config(4, 2, 10);
        let mut vq = VectorQuantizer::new(dim, cfg, DistanceMetric::L2).unwrap();
        let data = training_data(64, dim);
        vq.train(&data).unwrap();
        vq.add_residual_stage(&data).unwrap();

        let vec = &data[0..dim];
        let (pq, rvq) = vq.encode_with_residuals(vec).unwrap();
        let recon = vq.decode_with_residuals(&pq, &rvq).unwrap();
        assert_eq!(recon.len(), dim);
    }

    #[test]
    fn rvq_before_train_fails() {
        let cfg = make_config(4, 2, 5);
        let mut vq = VectorQuantizer::new(16, cfg, DistanceMetric::L2).unwrap();
        let data = training_data(32, 16);
        assert!(vq.add_residual_stage(&data).is_err());
    }
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy: a vector of `n` floats in [-1, 1].
    fn float_vec(n: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-1.0f32..1.0, n)
    }

    proptest! {
        #[test]
        fn l2_is_non_negative(a in float_vec(16), b in float_vec(16)) {
            let d = compute_distance(&a, &b, DistanceMetric::L2);
            prop_assert!(d >= -1e-6, "L2 distance negative: {d}");
        }

        #[test]
        fn l2_self_is_zero(v in float_vec(16)) {
            let d = compute_distance(&v, &v, DistanceMetric::L2);
            prop_assert!(d.abs() < 1e-4, "L2 self-distance non-zero: {d}");
        }

        #[test]
        fn l2_symmetric(a in float_vec(16), b in float_vec(16)) {
            let d1 = compute_distance(&a, &b, DistanceMetric::L2);
            let d2 = compute_distance(&b, &a, DistanceMetric::L2);
            prop_assert!((d1 - d2).abs() < 1e-4, "asymmetry: {d1} vs {d2}");
        }

        #[test]
        fn ip_symmetric(a in float_vec(16), b in float_vec(16)) {
            let d1 = compute_distance(&a, &b, DistanceMetric::InnerProduct);
            let d2 = compute_distance(&b, &a, DistanceMetric::InnerProduct);
            prop_assert!((d1 - d2).abs() < 1e-4, "ip asymmetry: {d1} vs {d2}");
        }

        #[test]
        fn pq_code_count_equals_num_subvectors(
            seed in 0u64..1000,
        ) {
            let dim = 16;
            let nsub = 4;
            let cfg = QuantizationConfig {
                num_subvectors: nsub,
                bits_per_code: 2,
                training_iterations: 5,
            };
            let mut vq = VectorQuantizer::new(dim, cfg, DistanceMetric::L2)
                .unwrap();
            let data: Vec<f32> = (0..(64 * dim))
                .map(|i| ((i as u64 * 7 + seed) % 101) as f32 / 100.0)
                .collect();
            vq.train(&data).unwrap();
            let codes = vq.encode(&data[0..dim]).unwrap();
            prop_assert_eq!(codes.len(), nsub);
        }

        #[test]
        fn pq_codes_within_codebook_range(seed in 0u64..1000) {
            let dim = 16;
            let bits = 2u32;
            let cfg = QuantizationConfig {
                num_subvectors: 4,
                bits_per_code: bits,
                training_iterations: 5,
            };
            let mut vq = VectorQuantizer::new(dim, cfg, DistanceMetric::L2)
                .unwrap();
            let data: Vec<f32> = (0..(64 * dim))
                .map(|i| ((i as u64 * 13 + seed) % 97) as f32 / 97.0)
                .collect();
            vq.train(&data).unwrap();
            let codes = vq.encode(&data[0..dim]).unwrap();
            let max_code = (1u16 << bits) - 1;
            for &c in &codes {
                prop_assert!(c <= max_code, "code {c} > max {max_code}");
            }
        }

        #[test]
        fn decode_length_equals_dim(seed in 0u64..1000) {
            let dim = 16;
            let cfg = QuantizationConfig {
                num_subvectors: 4,
                bits_per_code: 2,
                training_iterations: 3,
            };
            let mut vq = VectorQuantizer::new(dim, cfg, DistanceMetric::L2)
                .unwrap();
            let data: Vec<f32> = (0..(32 * dim))
                .map(|i| ((i as u64 + seed) % 53) as f32 / 53.0)
                .collect();
            vq.train(&data).unwrap();
            let codes = vq.encode(&data[0..dim]).unwrap();
            let recon = vq.decode(&codes).unwrap();
            prop_assert_eq!(recon.len(), dim);
        }
    }
}
