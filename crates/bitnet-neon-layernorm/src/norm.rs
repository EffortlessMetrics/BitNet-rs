//! Public normalization operation types.

use half::f16;

use crate::neon::{
    group_norm_f32_dispatch, layer_norm_affine_f32_dispatch, layer_norm_f32_dispatch,
    rms_norm_f32_dispatch, rms_norm_scale_f32_dispatch,
};

// ── LayerNorm ────────────────────────────────────────────────────────

/// Standard Layer Normalization.
///
/// Normalizes a vector to zero-mean, unit-variance, then optionally
/// applies an affine transform (scale `gamma` + shift `beta`).
///
/// ```
/// use bitnet_neon_layernorm::LayerNorm;
///
/// let ln = LayerNorm::new(4, 1e-5);
/// let mut v = [1.0f32, 2.0, 3.0, 4.0];
/// ln.forward(&mut v);
/// // v is now approximately [-1.342, -0.447, 0.447, 1.342]
/// ```
#[derive(Debug, Clone)]
pub struct LayerNorm {
    hidden_size: usize,
    epsilon: f32,
    gamma: Option<Vec<f32>>,
    beta: Option<Vec<f32>>,
}

impl LayerNorm {
    /// Create a new `LayerNorm` without affine parameters.
    #[must_use]
    pub const fn new(hidden_size: usize, epsilon: f32) -> Self {
        Self { hidden_size, epsilon, gamma: None, beta: None }
    }

    /// Create a new `LayerNorm` with affine parameters (scale + shift).
    ///
    /// # Panics
    ///
    /// Panics if `gamma` or `beta` length differs from `hidden_size`.
    #[must_use]
    pub fn with_affine(hidden_size: usize, epsilon: f32, gamma: Vec<f32>, beta: Vec<f32>) -> Self {
        assert_eq!(gamma.len(), hidden_size, "gamma length mismatch");
        assert_eq!(beta.len(), hidden_size, "beta length mismatch");
        Self { hidden_size, epsilon, gamma: Some(gamma), beta: Some(beta) }
    }

    /// Return the expected hidden size.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Return the epsilon value.
    #[must_use]
    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Apply `LayerNorm` in-place on an `f32` slice.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != hidden_size`.
    pub fn forward(&self, data: &mut [f32]) {
        assert_eq!(data.len(), self.hidden_size, "input length mismatch");
        match (&self.gamma, &self.beta) {
            (Some(g), Some(b)) => {
                layer_norm_affine_f32_dispatch(data, g, b, self.epsilon);
            }
            _ => {
                layer_norm_f32_dispatch(data, self.epsilon);
            }
        }
    }

    /// Apply `LayerNorm` on a batch of vectors stored contiguously.
    ///
    /// `data.len()` must be a multiple of `hidden_size`.
    pub fn forward_batch(&self, data: &mut [f32]) {
        assert!(
            data.len().is_multiple_of(self.hidden_size),
            "batch data length must be a multiple of hidden_size"
        );
        for chunk in data.chunks_mut(self.hidden_size) {
            self.forward(chunk);
        }
    }

    /// Apply `LayerNorm` on `f16` input, converting to `f32` internally.
    ///
    /// Results are written back as `f16`.
    pub fn forward_f16(&self, data: &mut [f16]) {
        assert_eq!(data.len(), self.hidden_size, "input length mismatch");
        let mut buf: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
        self.forward(&mut buf);
        for (dst, &src) in data.iter_mut().zip(buf.iter()) {
            *dst = f16::from_f32(src);
        }
    }
}

// ── RmsNorm ──────────────────────────────────────────────────────────

/// Root Mean Square Layer Normalization.
///
/// Unlike [`LayerNorm`], `RMSNorm` does not subtract the mean; it only
/// divides by the root-mean-square of the elements.
///
/// ```
/// use bitnet_neon_layernorm::RmsNorm;
///
/// let rms = RmsNorm::new(4, 1e-5);
/// let mut v = [1.0f32, 2.0, 3.0, 4.0];
/// rms.forward(&mut v);
/// ```
#[derive(Debug, Clone)]
pub struct RmsNorm {
    hidden_size: usize,
    epsilon: f32,
    gamma: Option<Vec<f32>>,
}

impl RmsNorm {
    /// Create a new `RmsNorm` without scale weights.
    #[must_use]
    pub const fn new(hidden_size: usize, epsilon: f32) -> Self {
        Self { hidden_size, epsilon, gamma: None }
    }

    /// Create a new `RmsNorm` with scale weights.
    ///
    /// # Panics
    ///
    /// Panics if `gamma.len()` differs from `hidden_size`.
    #[must_use]
    pub fn with_scale(hidden_size: usize, epsilon: f32, gamma: Vec<f32>) -> Self {
        assert_eq!(gamma.len(), hidden_size, "gamma length mismatch");
        Self { hidden_size, epsilon, gamma: Some(gamma) }
    }

    /// Return the expected hidden size.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Return the epsilon value.
    #[must_use]
    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Apply `RMSNorm` in-place on an `f32` slice.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != hidden_size`.
    pub fn forward(&self, data: &mut [f32]) {
        assert_eq!(data.len(), self.hidden_size, "input length mismatch");
        match &self.gamma {
            Some(g) => rms_norm_scale_f32_dispatch(data, g, self.epsilon),
            None => rms_norm_f32_dispatch(data, self.epsilon),
        }
    }

    /// Apply `RMSNorm` on a batch of vectors stored contiguously.
    pub fn forward_batch(&self, data: &mut [f32]) {
        assert!(
            data.len().is_multiple_of(self.hidden_size),
            "batch data length must be a multiple of hidden_size"
        );
        for chunk in data.chunks_mut(self.hidden_size) {
            self.forward(chunk);
        }
    }

    /// Apply `RMSNorm` on `f16` input.
    pub fn forward_f16(&self, data: &mut [f16]) {
        assert_eq!(data.len(), self.hidden_size, "input length mismatch");
        let mut buf: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
        self.forward(&mut buf);
        for (dst, &src) in data.iter_mut().zip(buf.iter()) {
            *dst = f16::from_f32(src);
        }
    }
}

// ── GroupNorm ─────────────────────────────────────────────────────────

/// Group Normalization.
///
/// Splits channels into `num_groups` groups and normalizes each group
/// independently, followed by a per-channel affine transform.
///
/// ```
/// use bitnet_neon_layernorm::GroupNorm;
///
/// let gn = GroupNorm::new(2, 4, 1e-5, vec![1.0; 4], vec![0.0; 4]);
/// let mut v = [1.0f32, 2.0, 3.0, 4.0];
/// gn.forward(&mut v);
/// ```
#[derive(Debug, Clone)]
pub struct GroupNorm {
    num_groups: usize,
    num_channels: usize,
    epsilon: f32,
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl GroupNorm {
    /// Create a new `GroupNorm`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `num_channels` is not divisible by `num_groups`
    /// - `gamma` or `beta` length differs from `num_channels`
    #[must_use]
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        epsilon: f32,
        gamma: Vec<f32>,
        beta: Vec<f32>,
    ) -> Self {
        assert!(
            num_groups > 0 && num_channels.is_multiple_of(num_groups),
            "num_channels must be divisible by num_groups"
        );
        assert_eq!(gamma.len(), num_channels, "gamma length mismatch");
        assert_eq!(beta.len(), num_channels, "beta length mismatch");
        Self { num_groups, num_channels, epsilon, gamma, beta }
    }

    /// Return the number of groups.
    #[must_use]
    pub const fn num_groups(&self) -> usize {
        self.num_groups
    }

    /// Return the number of channels.
    #[must_use]
    pub const fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Return the epsilon value.
    #[must_use]
    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Apply `GroupNorm` in-place.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != num_channels`.
    pub fn forward(&self, data: &mut [f32]) {
        assert_eq!(data.len(), self.num_channels, "input length mismatch");
        group_norm_f32_dispatch(data, self.num_groups, &self.gamma, &self.beta, self.epsilon);
    }

    /// Apply `GroupNorm` on a batch of vectors stored contiguously.
    pub fn forward_batch(&self, data: &mut [f32]) {
        assert!(
            data.len().is_multiple_of(self.num_channels),
            "batch data length must be a multiple of num_channels"
        );
        for chunk in data.chunks_mut(self.num_channels) {
            self.forward(chunk);
        }
    }

    /// Apply `GroupNorm` on `f16` input.
    pub fn forward_f16(&self, data: &mut [f16]) {
        assert_eq!(data.len(), self.num_channels, "input length mismatch");
        let mut buf: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
        self.forward(&mut buf);
        for (dst, &src) in data.iter_mut().zip(buf.iter()) {
            *dst = f16::from_f32(src);
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops
)]
mod tests {
    use super::*;
    use half::f16;
    use proptest::prelude::*;

    // ── scalar reference helpers ─────────────────────────────────────

    fn ref_mean(data: &[f32]) -> f32 {
        let s: f64 = data.iter().map(|&v| f64::from(v)).sum();
        (s / data.len() as f64) as f32
    }

    fn ref_var(data: &[f32], mean: f32) -> f32 {
        let m = f64::from(mean);
        let s: f64 = data.iter().map(|&v| (f64::from(v) - m).powi(2)).sum();
        (s / data.len() as f64) as f32
    }

    fn ref_layer_norm(data: &mut [f32], eps: f32) {
        let m = ref_mean(data);
        let v = ref_var(data, m);
        let inv = 1.0 / (v + eps).sqrt();
        for x in data.iter_mut() {
            *x = (*x - m) * inv;
        }
    }

    fn ref_rms_norm(data: &mut [f32], eps: f32) {
        let ms: f64 =
            data.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / data.len() as f64;
        let inv = 1.0 / (ms as f32 + eps).sqrt();
        for x in data.iter_mut() {
            *x *= inv;
        }
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y} (tol={tol})");
        }
    }

    // ── LayerNorm basic tests ────────────────────────────────────────

    #[test]
    fn layer_norm_basic_4() {
        let ln = LayerNorm::new(4, 1e-5);
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        let mut ref_data = data;
        ln.forward(&mut data);
        ref_layer_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn layer_norm_basic_8() {
        let ln = LayerNorm::new(8, 1e-5);
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut ref_data = data;
        ln.forward(&mut data);
        ref_layer_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn layer_norm_hidden_256() {
        let n = 256;
        let ln = LayerNorm::new(n, 1e-5);
        let mut data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut ref_data = data.clone();
        ln.forward(&mut data);
        ref_layer_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-4);
    }

    #[test]
    fn layer_norm_hidden_1024() {
        let n = 1024;
        let ln = LayerNorm::new(n, 1e-5);
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let mut ref_data = data.clone();
        ln.forward(&mut data);
        ref_layer_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-4);
    }

    #[test]
    fn layer_norm_hidden_2048() {
        let n = 2048;
        let ln = LayerNorm::new(n, 1e-6);
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).cos()).collect();
        let mut ref_data = data.clone();
        ln.forward(&mut data);
        ref_layer_norm(&mut ref_data, 1e-6);
        assert_close(&data, &ref_data, 1e-4);
    }

    #[test]
    fn layer_norm_non_aligned_5() {
        let ln = LayerNorm::new(5, 1e-5);
        let mut data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut ref_data = data;
        ln.forward(&mut data);
        ref_layer_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn layer_norm_non_aligned_7() {
        let ln = LayerNorm::new(7, 1e-5);
        let mut data = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        let mut ref_data = data;
        ln.forward(&mut data);
        ref_layer_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn layer_norm_single_element() {
        let ln = LayerNorm::new(1, 1e-5);
        let mut data = [42.0f32];
        ln.forward(&mut data);
        // Single element: (42 - 42) / sqrt(0 + eps) == 0
        assert!((data[0]).abs() < 1e-2);
    }

    #[test]
    fn layer_norm_constant_input() {
        let ln = LayerNorm::new(8, 1e-5);
        let mut data = [3.0f32; 8];
        ln.forward(&mut data);
        // All same → zero-mean, zero-var → output ≈ 0
        for &v in &data {
            assert!(v.abs() < 1e-2, "expected ~0, got {v}");
        }
    }

    #[test]
    fn layer_norm_zeros() {
        let ln = LayerNorm::new(4, 1e-5);
        let mut data = [0.0f32; 4];
        ln.forward(&mut data);
        for &v in &data {
            assert!(v.abs() < 1e-3);
        }
    }

    #[test]
    fn layer_norm_large_values() {
        let ln = LayerNorm::new(4, 1e-5);
        let mut data = [1e6, 2e6, 3e6, 4e6];
        let mut ref_data = data;
        ln.forward(&mut data);
        ref_layer_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-2);
    }

    #[test]
    fn layer_norm_negative_values() {
        let ln = LayerNorm::new(4, 1e-5);
        let mut data = [-3.0, -1.0, 1.0, 3.0];
        let mut ref_data = data;
        ln.forward(&mut data);
        ref_layer_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn layer_norm_epsilon_1e_8() {
        let ln = LayerNorm::new(4, 1e-8);
        let mut data = [1.0, 2.0, 3.0, 4.0];
        let mut ref_data = data;
        ln.forward(&mut data);
        ref_layer_norm(&mut ref_data, 1e-8);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn layer_norm_epsilon_1e_2() {
        let ln = LayerNorm::new(4, 1e-2);
        let mut data = [0.01, 0.02, 0.03, 0.04];
        let mut ref_data = data;
        ln.forward(&mut data);
        ref_layer_norm(&mut ref_data, 1e-2);
        assert_close(&data, &ref_data, 1e-5);
    }

    // ── LayerNorm affine tests ───────────────────────────────────────

    #[test]
    fn layer_norm_affine_basic() {
        let gamma = vec![2.0, 2.0, 2.0, 2.0];
        let beta = vec![1.0, 1.0, 1.0, 1.0];
        let ln = LayerNorm::with_affine(4, 1e-5, gamma.clone(), beta.clone());
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        let mut ref_data = data;
        ref_layer_norm(&mut ref_data, 1e-5);
        for i in 0..4 {
            ref_data[i] = gamma[i] * ref_data[i] + beta[i];
        }
        ln.forward(&mut data);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn layer_norm_affine_identity() {
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let ln_affine = LayerNorm::with_affine(8, 1e-5, gamma, beta);
        let ln_plain = LayerNorm::new(8, 1e-5);
        let base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut a = base;
        let mut b = base;
        ln_affine.forward(&mut a);
        ln_plain.forward(&mut b);
        assert_close(&a, &b, 1e-6);
    }

    #[test]
    fn layer_norm_affine_non_aligned() {
        let n = 5;
        let gamma = vec![0.5; n];
        let beta = vec![0.1; n];
        let ln = LayerNorm::with_affine(n, 1e-5, gamma.clone(), beta.clone());
        let mut data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut ref_data = data.clone();
        ref_layer_norm(&mut ref_data, 1e-5);
        for i in 0..n {
            ref_data[i] = gamma[i] * ref_data[i] + beta[i];
        }
        ln.forward(&mut data);
        assert_close(&data, &ref_data, 1e-5);
    }

    // ── LayerNorm batch tests ────────────────────────────────────────

    #[test]
    fn layer_norm_batch() {
        let ln = LayerNorm::new(4, 1e-5);
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut ref0 = [1.0f32, 2.0, 3.0, 4.0];
        let mut ref1 = [5.0f32, 6.0, 7.0, 8.0];
        ref_layer_norm(&mut ref0, 1e-5);
        ref_layer_norm(&mut ref1, 1e-5);
        ln.forward_batch(&mut data);
        assert_close(&data[..4], &ref0, 1e-5);
        assert_close(&data[4..], &ref1, 1e-5);
    }

    // ── LayerNorm f16 tests ──────────────────────────────────────────

    #[test]
    fn layer_norm_f16_basic() {
        let ln = LayerNorm::new(4, 1e-5);
        let mut data_f16: Vec<f16> =
            [1.0f32, 2.0, 3.0, 4.0].iter().map(|&v| f16::from_f32(v)).collect();
        let mut ref_data = [1.0f32, 2.0, 3.0, 4.0];
        ref_layer_norm(&mut ref_data, 1e-5);
        ln.forward_f16(&mut data_f16);
        let result: Vec<f32> = data_f16.iter().map(|v| v.to_f32()).collect();
        assert_close(&result, &ref_data, 5e-3); // f16 has lower precision
    }

    #[test]
    fn layer_norm_f16_hidden_8() {
        let ln = LayerNorm::new(8, 1e-5);
        let vals: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let mut data_f16: Vec<f16> = vals.iter().map(|&v| f16::from_f32(v)).collect();
        let mut ref_data = vals;
        ref_layer_norm(&mut ref_data, 1e-5);
        ln.forward_f16(&mut data_f16);
        let result: Vec<f32> = data_f16.iter().map(|v| v.to_f32()).collect();
        assert_close(&result, &ref_data, 5e-3);
    }

    // ── RmsNorm basic tests ──────────────────────────────────────────

    #[test]
    fn rms_norm_basic_4() {
        let rms = RmsNorm::new(4, 1e-5);
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        let mut ref_data = data;
        rms.forward(&mut data);
        ref_rms_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn rms_norm_basic_8() {
        let rms = RmsNorm::new(8, 1e-5);
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut ref_data = data;
        rms.forward(&mut data);
        ref_rms_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn rms_norm_hidden_256() {
        let n = 256;
        let rms = RmsNorm::new(n, 1e-5);
        let mut data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut ref_data = data.clone();
        rms.forward(&mut data);
        ref_rms_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-4);
    }

    #[test]
    fn rms_norm_hidden_1024() {
        let n = 1024;
        let rms = RmsNorm::new(n, 1e-5);
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let mut ref_data = data.clone();
        rms.forward(&mut data);
        ref_rms_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-4);
    }

    #[test]
    fn rms_norm_non_aligned_5() {
        let rms = RmsNorm::new(5, 1e-5);
        let mut data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut ref_data = data;
        rms.forward(&mut data);
        ref_rms_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn rms_norm_single_element() {
        let rms = RmsNorm::new(1, 1e-5);
        let mut data = [5.0f32];
        rms.forward(&mut data);
        // rms(5) = 5 / sqrt(25 + eps) ≈ 1.0
        assert!((data[0] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn rms_norm_zeros() {
        let rms = RmsNorm::new(4, 1e-5);
        let mut data = [0.0f32; 4];
        rms.forward(&mut data);
        for &v in &data {
            assert!(v.abs() < 1e-3);
        }
    }

    #[test]
    fn rms_norm_negative_values() {
        let rms = RmsNorm::new(4, 1e-5);
        let mut data = [-3.0, -1.0, 1.0, 3.0];
        let mut ref_data = data;
        rms.forward(&mut data);
        ref_rms_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn rms_norm_large_values() {
        let rms = RmsNorm::new(4, 1e-5);
        let mut data = [1e4, 2e4, 3e4, 4e4];
        let mut ref_data = data;
        rms.forward(&mut data);
        ref_rms_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-2);
    }

    #[test]
    fn rms_norm_epsilon_1e_8() {
        let rms = RmsNorm::new(4, 1e-8);
        let mut data = [1.0, 2.0, 3.0, 4.0];
        let mut ref_data = data;
        rms.forward(&mut data);
        ref_rms_norm(&mut ref_data, 1e-8);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn rms_norm_epsilon_large() {
        let rms = RmsNorm::new(4, 0.1);
        let mut data = [0.01, 0.02, 0.03, 0.04];
        let mut ref_data = data;
        rms.forward(&mut data);
        ref_rms_norm(&mut ref_data, 0.1);
        assert_close(&data, &ref_data, 1e-5);
    }

    // ── RmsNorm with scale tests ─────────────────────────────────────

    #[test]
    fn rms_norm_scale_basic() {
        let gamma = vec![2.0, 2.0, 2.0, 2.0];
        let rms = RmsNorm::with_scale(4, 1e-5, gamma.clone());
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        let mut ref_data = data;
        ref_rms_norm(&mut ref_data, 1e-5);
        for i in 0..4 {
            ref_data[i] *= gamma[i];
        }
        rms.forward(&mut data);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn rms_norm_scale_identity() {
        let gamma = vec![1.0; 8];
        let rms_scaled = RmsNorm::with_scale(8, 1e-5, gamma);
        let rms_plain = RmsNorm::new(8, 1e-5);
        let base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut a = base;
        let mut b = base;
        rms_scaled.forward(&mut a);
        rms_plain.forward(&mut b);
        assert_close(&a, &b, 1e-6);
    }

    // ── RmsNorm batch tests ──────────────────────────────────────────

    #[test]
    fn rms_norm_batch() {
        let rms = RmsNorm::new(4, 1e-5);
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut ref0 = [1.0f32, 2.0, 3.0, 4.0];
        let mut ref1 = [5.0f32, 6.0, 7.0, 8.0];
        ref_rms_norm(&mut ref0, 1e-5);
        ref_rms_norm(&mut ref1, 1e-5);
        rms.forward_batch(&mut data);
        assert_close(&data[..4], &ref0, 1e-5);
        assert_close(&data[4..], &ref1, 1e-5);
    }

    // ── RmsNorm f16 tests ────────────────────────────────────────────

    #[test]
    fn rms_norm_f16_basic() {
        let rms = RmsNorm::new(4, 1e-5);
        let mut data_f16: Vec<f16> =
            [1.0f32, 2.0, 3.0, 4.0].iter().map(|&v| f16::from_f32(v)).collect();
        let mut ref_data = [1.0f32, 2.0, 3.0, 4.0];
        ref_rms_norm(&mut ref_data, 1e-5);
        rms.forward_f16(&mut data_f16);
        let result: Vec<f32> = data_f16.iter().map(|v| v.to_f32()).collect();
        assert_close(&result, &ref_data, 5e-3);
    }

    // ── GroupNorm tests ──────────────────────────────────────────────

    #[test]
    fn group_norm_basic_2_groups() {
        let gn = GroupNorm::new(2, 4, 1e-5, vec![1.0; 4], vec![0.0; 4]);
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        let mut ref_data = data;
        // Group 0: [1, 2], Group 1: [3, 4]
        let m0 = 1.5f32;
        let v0 = 0.25f32;
        let inv0 = 1.0 / (v0 + 1e-5_f32).sqrt();
        ref_data[0] = (1.0 - m0) * inv0;
        ref_data[1] = (2.0 - m0) * inv0;
        let m1 = 3.5f32;
        let v1 = 0.25f32;
        let inv1 = 1.0 / (v1 + 1e-5_f32).sqrt();
        ref_data[2] = (3.0 - m1) * inv1;
        ref_data[3] = (4.0 - m1) * inv1;
        gn.forward(&mut data);
        assert_close(&data, &ref_data, 1e-4);
    }

    #[test]
    fn group_norm_4_groups() {
        let gn = GroupNorm::new(4, 8, 1e-5, vec![1.0; 8], vec![0.0; 8]);
        let mut data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        gn.forward(&mut data);
        // Each group of 2 should be normalized
        for chunk in data.chunks(2) {
            let mean: f32 = chunk.iter().sum::<f32>() / 2.0;
            assert!(mean.abs() < 0.1, "group mean should be ~0, got {mean}");
        }
    }

    #[test]
    fn group_norm_1_group_equals_layer_norm() {
        let n = 8;
        let gn = GroupNorm::new(1, n, 1e-5, vec![1.0; n], vec![0.0; n]);
        let ln = LayerNorm::new(n, 1e-5);
        let base: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let mut a = base.clone();
        let mut b = base;
        gn.forward(&mut a);
        ln.forward(&mut b);
        assert_close(&a, &b, 1e-5);
    }

    #[test]
    fn group_norm_with_affine() {
        let gamma = vec![2.0; 4];
        let beta = vec![0.5; 4];
        let gn = GroupNorm::new(2, 4, 1e-5, gamma, beta);
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        gn.forward(&mut data);
        // Just verify it runs and produces finite values
        for &v in &data {
            assert!(v.is_finite(), "expected finite value, got {v}");
        }
    }

    #[test]
    fn group_norm_hidden_256() {
        let n = 256;
        let groups = 8;
        let gn = GroupNorm::new(groups, n, 1e-5, vec![1.0; n], vec![0.0; n]);
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        gn.forward(&mut data);
        // Check each group has near-zero mean
        let cpg = n / groups;
        for g in 0..groups {
            let group = &data[g * cpg..(g + 1) * cpg];
            let mean: f32 = group.iter().sum::<f32>() / cpg as f32;
            assert!(mean.abs() < 0.1, "group {g} mean = {mean}");
        }
    }

    #[test]
    fn group_norm_batch() {
        let gn = GroupNorm::new(2, 4, 1e-5, vec![1.0; 4], vec![0.0; 4]);
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        gn.forward_batch(&mut data);
        for &v in &data {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn group_norm_f16_basic() {
        let gn = GroupNorm::new(2, 4, 1e-5, vec![1.0; 4], vec![0.0; 4]);
        let mut data_f16: Vec<f16> =
            [1.0f32, 2.0, 3.0, 4.0].iter().map(|&v| f16::from_f32(v)).collect();
        gn.forward_f16(&mut data_f16);
        let result: Vec<f32> = data_f16.iter().map(|v| v.to_f32()).collect();
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    // ── Accessor / constructor tests ─────────────────────────────────

    #[test]
    fn layer_norm_accessors() {
        let ln = LayerNorm::new(128, 1e-6);
        assert_eq!(ln.hidden_size(), 128);
        assert!((ln.epsilon() - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn rms_norm_accessors() {
        let rms = RmsNorm::new(64, 1e-8);
        assert_eq!(rms.hidden_size(), 64);
        assert!((rms.epsilon() - 1e-8).abs() < 1e-12);
    }

    #[test]
    fn group_norm_accessors() {
        let gn = GroupNorm::new(4, 16, 1e-5, vec![1.0; 16], vec![0.0; 16]);
        assert_eq!(gn.num_groups(), 4);
        assert_eq!(gn.num_channels(), 16);
        assert!((gn.epsilon() - 1e-5).abs() < 1e-10);
    }

    // ── Panic / edge case tests ──────────────────────────────────────

    #[test]
    #[should_panic(expected = "input length mismatch")]
    fn layer_norm_wrong_size_panics() {
        let ln = LayerNorm::new(4, 1e-5);
        let mut data = [1.0f32; 3];
        ln.forward(&mut data);
    }

    #[test]
    #[should_panic(expected = "input length mismatch")]
    fn rms_norm_wrong_size_panics() {
        let rms = RmsNorm::new(4, 1e-5);
        let mut data = [1.0f32; 5];
        rms.forward(&mut data);
    }

    #[test]
    #[should_panic(expected = "input length mismatch")]
    fn group_norm_wrong_size_panics() {
        let gn = GroupNorm::new(2, 4, 1e-5, vec![1.0; 4], vec![0.0; 4]);
        let mut data = [1.0f32; 6];
        gn.forward(&mut data);
    }

    #[test]
    #[should_panic(expected = "gamma length mismatch")]
    fn layer_norm_affine_gamma_mismatch_panics() {
        let _ = LayerNorm::with_affine(4, 1e-5, vec![1.0; 3], vec![0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "beta length mismatch")]
    fn layer_norm_affine_beta_mismatch_panics() {
        let _ = LayerNorm::with_affine(4, 1e-5, vec![1.0; 4], vec![0.0; 3]);
    }

    #[test]
    #[should_panic(expected = "gamma length mismatch")]
    fn rms_norm_scale_gamma_mismatch_panics() {
        let _ = RmsNorm::with_scale(4, 1e-5, vec![1.0; 5]);
    }

    #[test]
    #[should_panic(expected = "num_channels must be divisible by num_groups")]
    fn group_norm_indivisible_panics() {
        let _ = GroupNorm::new(3, 8, 1e-5, vec![1.0; 8], vec![0.0; 8]);
    }

    #[test]
    #[should_panic(expected = "batch data length must be a multiple of hidden_size")]
    fn layer_norm_batch_mismatch_panics() {
        let ln = LayerNorm::new(4, 1e-5);
        let mut data = [1.0f32; 5];
        ln.forward_batch(&mut data);
    }

    #[test]
    #[should_panic(expected = "batch data length must be a multiple of hidden_size")]
    fn rms_norm_batch_mismatch_panics() {
        let rms = RmsNorm::new(4, 1e-5);
        let mut data = [1.0f32; 7];
        rms.forward_batch(&mut data);
    }

    // ── Convenience function tests ───────────────────────────────────

    #[test]
    fn layer_norm_inplace_fn() {
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        let mut ref_data = data;
        crate::layer_norm_inplace(&mut data, 1e-5);
        ref_layer_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-5);
    }

    #[test]
    fn rms_norm_inplace_fn() {
        let mut data = [1.0f32, 2.0, 3.0, 4.0];
        let mut ref_data = data;
        crate::rms_norm_inplace(&mut data, 1e-5);
        ref_rms_norm(&mut ref_data, 1e-5);
        assert_close(&data, &ref_data, 1e-5);
    }

    // ── Output property tests ────────────────────────────────────────

    #[test]
    fn layer_norm_output_zero_mean() {
        let ln = LayerNorm::new(16, 1e-5);
        let mut data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 3.0).collect();
        ln.forward(&mut data);
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 1e-4, "expected ~0 mean, got {mean}");
    }

    #[test]
    fn layer_norm_output_unit_variance() {
        let ln = LayerNorm::new(64, 1e-5);
        let mut data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        ln.forward(&mut data);
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        assert!((var - 1.0).abs() < 0.05, "expected ~1 variance, got {var}");
    }

    #[test]
    fn rms_norm_preserves_sign() {
        let rms = RmsNorm::new(4, 1e-5);
        let mut data = [-2.0f32, -1.0, 1.0, 2.0];
        let signs: Vec<f32> = data.iter().map(|v| v.signum()).collect();
        rms.forward(&mut data);
        for (v, s) in data.iter().zip(signs.iter()) {
            assert_eq!(v.signum(), *s, "sign changed for {v}");
        }
    }

    #[test]
    fn rms_norm_unit_norm_property() {
        // After RMSNorm, mean(x^2) should be ≈ 1.0
        let rms = RmsNorm::new(32, 1e-5);
        let mut data: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.1).collect();
        rms.forward(&mut data);
        let ms: f32 = data.iter().map(|v| v * v).sum::<f32>() / data.len() as f32;
        assert!((ms - 1.0).abs() < 0.05, "expected mean(x^2)≈1, got {ms}");
    }

    // ── Clone / Debug tests ──────────────────────────────────────────

    #[test]
    fn layer_norm_clone_produces_same_result() {
        let ln = LayerNorm::with_affine(4, 1e-5, vec![2.0; 4], vec![1.0; 4]);
        let ln2 = ln.clone();
        let mut a = [1.0f32, 2.0, 3.0, 4.0];
        let mut b = a;
        ln.forward(&mut a);
        ln2.forward(&mut b);
        assert_close(&a, &b, 1e-7);
    }

    #[test]
    fn rms_norm_debug_impl() {
        let rms = RmsNorm::new(4, 1e-5);
        let dbg = format!("{rms:?}");
        assert!(dbg.contains("RmsNorm"));
    }

    #[test]
    fn group_norm_debug_impl() {
        let gn = GroupNorm::new(2, 4, 1e-5, vec![1.0; 4], vec![0.0; 4]);
        let dbg = format!("{gn:?}");
        assert!(dbg.contains("GroupNorm"));
    }

    // ── proptest ─────────────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_layer_norm_finite_output(
            data in proptest::collection::vec(-100.0f32..100.0, 4..=128)
        ) {
            let ln = LayerNorm::new(data.len(), 1e-5);
            let mut buf = data;
            ln.forward(&mut buf);
            for v in &buf {
                prop_assert!(v.is_finite(), "non-finite value: {}", v);
            }
        }

        #[test]
        fn prop_layer_norm_zero_mean(
            data in proptest::collection::vec(-50.0f32..50.0, 8..=128)
        ) {
            let ln = LayerNorm::new(data.len(), 1e-5);
            let mut buf = data;
            ln.forward(&mut buf);
            let mean: f32 = buf.iter().sum::<f32>() / buf.len() as f32;
            prop_assert!((mean).abs() < 0.01, "mean = {}", mean);
        }

        #[test]
        fn prop_rms_norm_finite_output(
            data in proptest::collection::vec(-100.0f32..100.0, 4..=128)
        ) {
            let rms = RmsNorm::new(data.len(), 1e-5);
            let mut buf = data;
            rms.forward(&mut buf);
            for v in &buf {
                prop_assert!(v.is_finite(), "non-finite value: {}", v);
            }
        }

        #[test]
        fn prop_rms_norm_preserves_sign(
            data in proptest::collection::vec(
                proptest::prelude::prop_oneof![
                    -100.0f32..=-0.01,
                    0.01f32..=100.0,
                ],
                4..=64
            )
        ) {
            let rms = RmsNorm::new(data.len(), 1e-5);
            let signs: Vec<f32> = data.iter().map(|v| v.signum()).collect();
            let mut buf = data;
            rms.forward(&mut buf);
            for (i, (v, s)) in buf.iter().zip(signs.iter()).enumerate() {
                prop_assert_eq!(
                    v.signum(), *s,
                    "sign mismatch at {}: {} vs expected sign {}", i, v, s
                );
            }
        }

        #[test]
        fn prop_layer_norm_matches_reference(
            data in proptest::collection::vec(-10.0f32..10.0, 4..=64)
        ) {
            let n = data.len();
            let ln = LayerNorm::new(n, 1e-5);
            let mut actual = data.clone();
            let mut expected = data;
            ln.forward(&mut actual);
            ref_layer_norm(&mut expected, 1e-5);
            for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                prop_assert!(
                    (a - e).abs() < 1e-3,
                    "mismatch at {i}: {a} vs {e}"
                );
            }
        }

        #[test]
        fn prop_rms_norm_matches_reference(
            data in proptest::collection::vec(-10.0f32..10.0, 4..=64)
        ) {
            let n = data.len();
            let rms = RmsNorm::new(n, 1e-5);
            let mut actual = data.clone();
            let mut expected = data;
            rms.forward(&mut actual);
            ref_rms_norm(&mut expected, 1e-5);
            for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                prop_assert!(
                    (a - e).abs() < 1e-3,
                    "mismatch at {i}: {a} vs {e}"
                );
            }
        }

        #[test]
        fn prop_group_norm_finite_output(
            size in (1usize..=8).prop_map(|g| g * 4),
        ) {
            let groups = 4.min(size);
            let groups = if size % groups == 0 { groups } else { 1 };
            let gn = GroupNorm::new(groups, size, 1e-5, vec![1.0; size], vec![0.0; size]);
            let mut data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            gn.forward(&mut data);
            for v in &data {
                prop_assert!(v.is_finite(), "non-finite: {}", v);
            }
        }

        #[test]
        fn prop_layer_norm_idempotent_mean(
            data in proptest::collection::vec(-10.0f32..10.0, 8..=32)
        ) {
            // After LayerNorm, applying it again should still give zero-mean
            let n = data.len();
            let ln = LayerNorm::new(n, 1e-5);
            let mut buf = data;
            ln.forward(&mut buf);
            ln.forward(&mut buf);
            let mean: f32 = buf.iter().sum::<f32>() / buf.len() as f32;
            prop_assert!((mean).abs() < 0.01, "mean after double apply = {}", mean);
        }
    }
}
