//! Numerical stability guards and validators for OpenCL GPU compute on Intel Arc A770.
//!
//! Provides NaN/Inf detection, gradient clipping, dynamic loss scaling,
//! numerical profiling, and GPU-vs-CPU comparison validation. All operations
//! include CPU reference implementations and embedded OpenCL kernel sources
//! for parallel execution on the GPU.

use std::fmt;

// ---------------------------------------------------------------------------
// StabilityConfig
// ---------------------------------------------------------------------------

/// Configuration for numerical stability enforcement.
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// Maximum allowed value; anything above is clamped.
    pub max_value: f32,
    /// Minimum allowed value; anything below is clamped.
    pub min_value: f32,
    /// Replacement value for NaN entries.
    pub nan_replacement: f32,
    /// Replacement value for Inf entries.
    pub inf_replacement: f32,
    /// Absolute tolerance for comparison validation.
    pub atol: f32,
    /// Relative tolerance for comparison validation.
    pub rtol: f32,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            max_value: 65504.0,  // f16 max
            min_value: -65504.0, // f16 min
            nan_replacement: 0.0,
            inf_replacement: 0.0,
            atol: 1e-5,
            rtol: 1e-3,
        }
    }
}

// ---------------------------------------------------------------------------
// NaN / Inf location tracking
// ---------------------------------------------------------------------------

/// A detected anomalous value with its position and original value.
#[derive(Debug, Clone, PartialEq)]
pub struct AnomalyLocation {
    /// Flat index into the tensor.
    pub index: usize,
    /// The anomalous value found at `index`.
    pub value: f32,
}

// ---------------------------------------------------------------------------
// NanDetector
// ---------------------------------------------------------------------------

/// Scans a tensor for NaN values and records their locations.
#[derive(Debug)]
pub struct NanDetector {
    locations: Vec<AnomalyLocation>,
}

impl NanDetector {
    /// Scan `data` for NaN values.
    pub fn scan(data: &[f32]) -> Self {
        let locations = data
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_nan())
            .map(|(i, &v)| AnomalyLocation { index: i, value: v })
            .collect();
        Self { locations }
    }

    /// Returns `true` when at least one NaN was found.
    pub fn has_nan(&self) -> bool {
        !self.locations.is_empty()
    }

    /// Number of NaN entries detected.
    pub fn count(&self) -> usize {
        self.locations.len()
    }

    /// Borrow the recorded locations.
    pub fn locations(&self) -> &[AnomalyLocation] {
        &self.locations
    }
}

// ---------------------------------------------------------------------------
// InfDetector
// ---------------------------------------------------------------------------

/// Scans a tensor for infinite values and records their locations.
#[derive(Debug)]
pub struct InfDetector {
    locations: Vec<AnomalyLocation>,
}

impl InfDetector {
    /// Scan `data` for +Inf / −Inf values.
    pub fn scan(data: &[f32]) -> Self {
        let locations = data
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_infinite())
            .map(|(i, &v)| AnomalyLocation { index: i, value: v })
            .collect();
        Self { locations }
    }

    /// Returns `true` when at least one Inf was found.
    pub fn has_inf(&self) -> bool {
        !self.locations.is_empty()
    }

    /// Number of Inf entries detected.
    pub fn count(&self) -> usize {
        self.locations.len()
    }

    /// Borrow the recorded locations.
    pub fn locations(&self) -> &[AnomalyLocation] {
        &self.locations
    }
}

// ---------------------------------------------------------------------------
// StabilityGuard
// ---------------------------------------------------------------------------

/// Wraps a tensor computation with NaN/Inf detection and value clamping.
///
/// After `sanitize()` the output is guaranteed free of NaN, Inf, and values
/// outside `[config.min_value, config.max_value]`.
pub struct StabilityGuard {
    config: StabilityConfig,
    nan_replaced: usize,
    inf_replaced: usize,
    clamped: usize,
}

impl StabilityGuard {
    pub fn new(config: StabilityConfig) -> Self {
        Self { config, nan_replaced: 0, inf_replaced: 0, clamped: 0 }
    }

    /// Create a guard with the default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StabilityConfig::default())
    }

    /// Sanitize `data` in-place: replace NaN/Inf, then clamp.
    pub fn sanitize(&mut self, data: &mut [f32]) {
        self.nan_replaced = 0;
        self.inf_replaced = 0;
        self.clamped = 0;

        for v in data.iter_mut() {
            if v.is_nan() {
                *v = self.config.nan_replacement;
                self.nan_replaced += 1;
            } else if v.is_infinite() {
                *v = self.config.inf_replacement;
                self.inf_replaced += 1;
            } else if *v > self.config.max_value {
                *v = self.config.max_value;
                self.clamped += 1;
            } else if *v < self.config.min_value {
                *v = self.config.min_value;
                self.clamped += 1;
            }
        }
    }

    /// Number of NaN values that were replaced in the last `sanitize` call.
    pub fn nan_replaced(&self) -> usize {
        self.nan_replaced
    }

    /// Number of Inf values that were replaced in the last `sanitize` call.
    pub fn inf_replaced(&self) -> usize {
        self.inf_replaced
    }

    /// Number of values clamped to the allowed range in the last `sanitize` call.
    pub fn clamped(&self) -> usize {
        self.clamped
    }
}

// ---------------------------------------------------------------------------
// GradientClipper
// ---------------------------------------------------------------------------

/// Clips gradient vectors by L2 norm to prevent gradient explosion.
pub struct GradientClipper {
    max_norm: f32,
}

impl GradientClipper {
    pub fn new(max_norm: f32) -> Self {
        Self { max_norm }
    }

    /// Compute the L2 norm of `grads`.
    pub fn l2_norm(grads: &[f32]) -> f32 {
        grads.iter().map(|g| g * g).sum::<f32>().sqrt()
    }

    /// Clip `grads` in-place if the L2 norm exceeds `max_norm`.
    /// Returns the original norm before clipping.
    pub fn clip(&self, grads: &mut [f32]) -> f32 {
        let norm = Self::l2_norm(grads);
        if norm > self.max_norm && norm > 0.0 {
            let scale = self.max_norm / norm;
            for g in grads.iter_mut() {
                *g *= scale;
            }
        }
        norm
    }
}

// ---------------------------------------------------------------------------
// LossScaler
// ---------------------------------------------------------------------------

/// Dynamic loss scaling for mixed-precision training.
///
/// Doubles the scale after `growth_interval` consecutive clean steps (no
/// overflow). Halves the scale immediately on overflow detection.
pub struct LossScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: u32,
    consecutive_clean: u32,
}

impl LossScaler {
    pub fn new(
        initial_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: u32,
    ) -> Self {
        Self {
            scale: initial_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            consecutive_clean: 0,
        }
    }

    /// Convenient constructor with common defaults (scale=1024, ×2/÷2, interval=2000).
    pub fn with_defaults() -> Self {
        Self::new(1024.0, 2.0, 0.5, 2000)
    }

    /// Current loss scale factor.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Scale a loss value up.
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// Un-scale gradients after the backward pass.
    pub fn unscale_gradients(&self, grads: &mut [f32]) {
        let inv = 1.0 / self.scale;
        for g in grads.iter_mut() {
            *g *= inv;
        }
    }

    /// Report whether the step contained overflow.
    /// Updates internal scale accordingly.
    pub fn update(&mut self, overflow_detected: bool) {
        if overflow_detected {
            self.scale *= self.backoff_factor;
            self.consecutive_clean = 0;
        } else {
            self.consecutive_clean += 1;
            if self.consecutive_clean >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.consecutive_clean = 0;
            }
        }
    }

    /// Check whether `data` contains NaN or Inf, indicating overflow.
    pub fn detect_overflow(data: &[f32]) -> bool {
        data.iter().any(|v| v.is_nan() || v.is_infinite())
    }
}

// ---------------------------------------------------------------------------
// NumericalProfile
// ---------------------------------------------------------------------------

/// Statistical profile of a tensor's numerical distribution.
#[derive(Debug, Clone)]
pub struct NumericalProfile {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub nan_count: usize,
    pub inf_count: usize,
    pub element_count: usize,
}

impl NumericalProfile {
    /// Profile the numerical distribution of `data`.
    pub fn compute(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                nan_count: 0,
                inf_count: 0,
                element_count: 0,
            };
        }

        let mut nan_count = 0usize;
        let mut inf_count = 0usize;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut finite_count = 0usize;

        for &v in data {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            } else {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
                sum += v as f64;
                finite_count += 1;
            }
        }

        let mean = if finite_count > 0 { (sum / finite_count as f64) as f32 } else { 0.0 };

        let std_dev = if finite_count > 0 {
            let var = data
                .iter()
                .filter(|v| v.is_finite())
                .map(|&v| {
                    let d = (v as f64) - (mean as f64);
                    d * d
                })
                .sum::<f64>()
                / finite_count as f64;
            var.sqrt() as f32
        } else {
            0.0
        };

        // If no finite values, set min/max to 0.
        if finite_count == 0 {
            min = 0.0;
            max = 0.0;
        }

        Self { min, max, mean, std_dev, nan_count, inf_count, element_count: data.len() }
    }

    /// True when no NaN or Inf values were detected.
    pub fn is_clean(&self) -> bool {
        self.nan_count == 0 && self.inf_count == 0
    }
}

impl fmt::Display for NumericalProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Profile(n={}, min={:.4}, max={:.4}, mean={:.4}, std={:.4}, \
             nan={}, inf={})",
            self.element_count,
            self.min,
            self.max,
            self.mean,
            self.std_dev,
            self.nan_count,
            self.inf_count,
        )
    }
}

// ---------------------------------------------------------------------------
// ComparisonValidator
// ---------------------------------------------------------------------------

/// Compares GPU (or candidate) results against a CPU reference using combined
/// absolute and relative tolerance:
///
/// ```text
///     |a − b| ≤ atol + rtol × |b|
/// ```
#[derive(Debug)]
pub struct ComparisonValidator {
    config: StabilityConfig,
}

/// Result of a comparison between two tensors.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Whether all element-wise checks passed.
    pub all_close: bool,
    /// Maximum absolute difference found.
    pub max_abs_diff: f32,
    /// Maximum relative difference found (where reference ≠ 0).
    pub max_rel_diff: f32,
    /// Number of elements that exceeded tolerance.
    pub mismatch_count: usize,
    /// Total number of elements compared.
    pub total_elements: usize,
}

impl fmt::Display for ComparisonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Comparison(close={}, mismatches={}/{}, max_abs={:.6e}, max_rel={:.6e})",
            self.all_close,
            self.mismatch_count,
            self.total_elements,
            self.max_abs_diff,
            self.max_rel_diff,
        )
    }
}

impl ComparisonValidator {
    pub fn new(config: StabilityConfig) -> Self {
        Self { config }
    }

    /// Create a validator with the default tolerances.
    pub fn with_defaults() -> Self {
        Self::new(StabilityConfig::default())
    }

    /// Create a validator with explicit tolerances.
    pub fn with_tolerances(atol: f32, rtol: f32) -> Self {
        Self::new(StabilityConfig { atol, rtol, ..StabilityConfig::default() })
    }

    /// Compare `candidate` against `reference` element-wise.
    pub fn validate(&self, candidate: &[f32], reference: &[f32]) -> ComparisonResult {
        assert_eq!(
            candidate.len(),
            reference.len(),
            "candidate and reference must have the same length"
        );

        let total = candidate.len();
        if total == 0 {
            return ComparisonResult {
                all_close: true,
                max_abs_diff: 0.0,
                max_rel_diff: 0.0,
                mismatch_count: 0,
                total_elements: 0,
            };
        }

        let mut max_abs: f32 = 0.0;
        let mut max_rel: f32 = 0.0;
        let mut mismatches = 0usize;

        for (&c, &r) in candidate.iter().zip(reference.iter()) {
            // Both NaN → considered matching
            if c.is_nan() && r.is_nan() {
                continue;
            }
            // One NaN → mismatch
            if c.is_nan() || r.is_nan() {
                mismatches += 1;
                continue;
            }

            let abs_diff = (c - r).abs();
            let rel_diff = if r.abs() > 0.0 { abs_diff / r.abs() } else { 0.0 };

            if abs_diff > max_abs {
                max_abs = abs_diff;
            }
            if rel_diff > max_rel {
                max_rel = rel_diff;
            }

            let tol = self.config.atol + self.config.rtol * r.abs();
            if abs_diff > tol {
                mismatches += 1;
            }
        }

        ComparisonResult {
            all_close: mismatches == 0,
            max_abs_diff: max_abs,
            max_rel_diff: max_rel,
            mismatch_count: mismatches,
            total_elements: total,
        }
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel sources for parallel NaN/Inf detection
// ---------------------------------------------------------------------------

/// Embedded OpenCL kernel source for parallel NaN/Inf scanning.
///
/// Each work-item inspects one element and writes a `1` into the corresponding
/// flag buffer when the value is NaN or Inf, enabling a subsequent reduction
/// pass on the host.
pub const OPENCL_NAN_INF_DETECT_KERNEL: &str = r#"
__kernel void nan_inf_detect(
    __global const float* data,
    __global int* nan_flags,
    __global int* inf_flags,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;

    float val = data[gid];
    nan_flags[gid] = isnan(val) ? 1 : 0;
    inf_flags[gid] = isinf(val) ? 1 : 0;
}
"#;

/// OpenCL kernel for parallel value clamping with NaN/Inf replacement.
pub const OPENCL_SANITIZE_KERNEL: &str = r#"
__kernel void sanitize(
    __global float* data,
    const float min_val,
    const float max_val,
    const float nan_replacement,
    const float inf_replacement,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;

    float val = data[gid];
    if (isnan(val)) {
        data[gid] = nan_replacement;
    } else if (isinf(val)) {
        data[gid] = inf_replacement;
    } else {
        data[gid] = clamp(val, min_val, max_val);
    }
}
"#;

/// OpenCL kernel for parallel L2 norm reduction (partial sums per work-group).
pub const OPENCL_L2_NORM_KERNEL: &str = r#"
__kernel void l2_norm_partial(
    __global const float* data,
    __global float* partial_sums,
    __local float* scratch,
    const int n
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    float val = (gid < n) ? data[gid] : 0.0f;
    scratch[lid] = val * val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial_sums[get_group_id(0)] = scratch[0];
    }
}
"#;

// ---------------------------------------------------------------------------
// CPU reference helpers (used by tests and as fallback)
// ---------------------------------------------------------------------------

/// CPU reference: count NaN values in `data`.
pub fn cpu_count_nan(data: &[f32]) -> usize {
    data.iter().filter(|v| v.is_nan()).count()
}

/// CPU reference: count Inf values in `data`.
pub fn cpu_count_inf(data: &[f32]) -> usize {
    data.iter().filter(|v| v.is_infinite()).count()
}

/// CPU reference: sanitize `data` in-place (replace NaN/Inf, clamp range).
pub fn cpu_sanitize(data: &mut [f32], config: &StabilityConfig) {
    for v in data.iter_mut() {
        if v.is_nan() {
            *v = config.nan_replacement;
        } else if v.is_infinite() {
            *v = config.inf_replacement;
        } else if *v > config.max_value {
            *v = config.max_value;
        } else if *v < config.min_value {
            *v = config.min_value;
        }
    }
}

/// CPU reference: compute L2 norm.
pub fn cpu_l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|v| v * v).sum::<f32>().sqrt()
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- NanDetector -------------------------------------------------------

    #[test]
    fn nan_detector_no_nans() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let det = NanDetector::scan(&data);
        assert!(!det.has_nan());
        assert_eq!(det.count(), 0);
    }

    #[test]
    fn nan_detector_single_nan_at_start() {
        let data = [f32::NAN, 1.0, 2.0];
        let det = NanDetector::scan(&data);
        assert!(det.has_nan());
        assert_eq!(det.count(), 1);
        assert_eq!(det.locations()[0].index, 0);
    }

    #[test]
    fn nan_detector_single_nan_at_end() {
        let data = [1.0, 2.0, f32::NAN];
        let det = NanDetector::scan(&data);
        assert!(det.has_nan());
        assert_eq!(det.count(), 1);
        assert_eq!(det.locations()[0].index, 2);
    }

    #[test]
    fn nan_detector_multiple_nans() {
        let data = [f32::NAN, 1.0, f32::NAN, 2.0, f32::NAN];
        let det = NanDetector::scan(&data);
        assert_eq!(det.count(), 3);
        let indices: Vec<usize> = det.locations().iter().map(|l| l.index).collect();
        assert_eq!(indices, vec![0, 2, 4]);
    }

    #[test]
    fn nan_detector_all_nan() {
        let data = [f32::NAN; 5];
        let det = NanDetector::scan(&data);
        assert_eq!(det.count(), 5);
    }

    #[test]
    fn nan_detector_empty() {
        let data: [f32; 0] = [];
        let det = NanDetector::scan(&data);
        assert!(!det.has_nan());
        assert_eq!(det.count(), 0);
    }

    #[test]
    fn nan_detector_inf_not_counted_as_nan() {
        let data = [f32::INFINITY, f32::NEG_INFINITY, 1.0];
        let det = NanDetector::scan(&data);
        assert!(!det.has_nan());
    }

    // -- InfDetector -------------------------------------------------------

    #[test]
    fn inf_detector_no_infs() {
        let data = [1.0, -2.0, 0.0];
        let det = InfDetector::scan(&data);
        assert!(!det.has_inf());
        assert_eq!(det.count(), 0);
    }

    #[test]
    fn inf_detector_positive_inf() {
        let data = [1.0, f32::INFINITY, 2.0];
        let det = InfDetector::scan(&data);
        assert!(det.has_inf());
        assert_eq!(det.count(), 1);
        assert_eq!(det.locations()[0].index, 1);
        assert!(det.locations()[0].value.is_sign_positive());
    }

    #[test]
    fn inf_detector_negative_inf() {
        let data = [f32::NEG_INFINITY, 0.0];
        let det = InfDetector::scan(&data);
        assert!(det.has_inf());
        assert_eq!(det.locations()[0].value, f32::NEG_INFINITY);
    }

    #[test]
    fn inf_detector_both_infs() {
        let data = [f32::INFINITY, 1.0, f32::NEG_INFINITY];
        let det = InfDetector::scan(&data);
        assert_eq!(det.count(), 2);
    }

    #[test]
    fn inf_detector_nan_not_counted_as_inf() {
        let data = [f32::NAN, 1.0];
        let det = InfDetector::scan(&data);
        assert!(!det.has_inf());
    }

    #[test]
    fn inf_detector_empty() {
        let data: [f32; 0] = [];
        let det = InfDetector::scan(&data);
        assert!(!det.has_inf());
        assert_eq!(det.count(), 0);
    }

    // -- StabilityGuard ----------------------------------------------------

    #[test]
    fn guard_replaces_nan() {
        let mut data = [1.0, f32::NAN, 3.0];
        let mut guard = StabilityGuard::with_defaults();
        guard.sanitize(&mut data);
        assert_eq!(data[1], 0.0);
        assert_eq!(guard.nan_replaced(), 1);
    }

    #[test]
    fn guard_replaces_inf() {
        let mut data = [f32::INFINITY, 2.0, f32::NEG_INFINITY];
        let mut guard = StabilityGuard::with_defaults();
        guard.sanitize(&mut data);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[2], 0.0);
        assert_eq!(guard.inf_replaced(), 2);
    }

    #[test]
    fn guard_clamps_high_values() {
        let config = StabilityConfig { max_value: 10.0, ..StabilityConfig::default() };
        let mut data = [5.0, 20.0, 100.0];
        let mut guard = StabilityGuard::new(config);
        guard.sanitize(&mut data);
        assert_eq!(data[0], 5.0);
        assert_eq!(data[1], 10.0);
        assert_eq!(data[2], 10.0);
        assert_eq!(guard.clamped(), 2);
    }

    #[test]
    fn guard_clamps_low_values() {
        let config = StabilityConfig { min_value: -5.0, ..StabilityConfig::default() };
        let mut data = [-3.0, -10.0, -100.0];
        let mut guard = StabilityGuard::new(config);
        guard.sanitize(&mut data);
        assert_eq!(data[0], -3.0);
        assert_eq!(data[1], -5.0);
        assert_eq!(data[2], -5.0);
        assert_eq!(guard.clamped(), 2);
    }

    #[test]
    fn guard_custom_nan_replacement() {
        let config = StabilityConfig { nan_replacement: -999.0, ..StabilityConfig::default() };
        let mut data = [f32::NAN];
        let mut guard = StabilityGuard::new(config);
        guard.sanitize(&mut data);
        assert_eq!(data[0], -999.0);
    }

    #[test]
    fn guard_custom_inf_replacement() {
        let config = StabilityConfig { inf_replacement: 42.0, ..StabilityConfig::default() };
        let mut data = [f32::INFINITY];
        let mut guard = StabilityGuard::new(config);
        guard.sanitize(&mut data);
        assert_eq!(data[0], 42.0);
    }

    #[test]
    fn guard_mixed_nan_inf() {
        let mut data = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 5.0];
        let mut guard = StabilityGuard::with_defaults();
        guard.sanitize(&mut data);
        assert_eq!(guard.nan_replaced(), 1);
        assert_eq!(guard.inf_replaced(), 2);
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn guard_empty_tensor() {
        let mut data: [f32; 0] = [];
        let mut guard = StabilityGuard::with_defaults();
        guard.sanitize(&mut data);
        assert_eq!(guard.nan_replaced(), 0);
        assert_eq!(guard.inf_replaced(), 0);
        assert_eq!(guard.clamped(), 0);
    }

    #[test]
    fn guard_output_never_contains_nan_or_inf() {
        // Property-style: various pathological inputs.
        let cases: Vec<Vec<f32>> = vec![
            vec![f32::NAN; 100],
            vec![f32::INFINITY; 50],
            vec![f32::NEG_INFINITY; 50],
            vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0],
            vec![1e38, -1e38, f32::NAN, f32::INFINITY],
            (0..256).map(|i| if i % 3 == 0 { f32::NAN } else { i as f32 }).collect(),
        ];
        for mut case in cases {
            let mut guard = StabilityGuard::with_defaults();
            guard.sanitize(&mut case);
            for (i, &v) in case.iter().enumerate() {
                assert!(v.is_finite(), "index {i} is not finite: {v}");
            }
        }
    }

    #[test]
    fn guard_resets_counters_on_second_call() {
        let mut guard = StabilityGuard::with_defaults();
        let mut a = [f32::NAN, f32::INFINITY];
        guard.sanitize(&mut a);
        assert_eq!(guard.nan_replaced(), 1);
        assert_eq!(guard.inf_replaced(), 1);

        let mut b = [1.0, 2.0];
        guard.sanitize(&mut b);
        assert_eq!(guard.nan_replaced(), 0);
        assert_eq!(guard.inf_replaced(), 0);
    }

    // -- GradientClipper ---------------------------------------------------

    #[test]
    fn clipper_no_clip_needed() {
        let mut grads = [1.0, 0.0, -1.0];
        let clipper = GradientClipper::new(10.0);
        let norm = clipper.clip(&mut grads);
        assert!((norm - 2.0f32.sqrt()).abs() < 1e-5);
        // Unchanged because norm < max_norm.
        assert_eq!(grads, [1.0, 0.0, -1.0]);
    }

    #[test]
    fn clipper_clips_to_max_norm() {
        let mut grads = [3.0, 4.0]; // norm = 5
        let clipper = GradientClipper::new(1.0);
        let norm = clipper.clip(&mut grads);
        assert!((norm - 5.0).abs() < 1e-5);
        let new_norm = GradientClipper::l2_norm(&grads);
        assert!((new_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn clipper_preserves_direction() {
        let mut grads = [6.0, 8.0]; // norm = 10
        let clipper = GradientClipper::new(5.0);
        clipper.clip(&mut grads);
        // Direction should be same: ratio grads[1]/grads[0] = 8/6 = 4/3
        assert!((grads[1] / grads[0] - 4.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn clipper_zero_norm() {
        let mut grads = [0.0, 0.0, 0.0];
        let clipper = GradientClipper::new(1.0);
        let norm = clipper.clip(&mut grads);
        assert_eq!(norm, 0.0);
        assert_eq!(grads, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn clipper_single_element() {
        let mut grads = [10.0];
        let clipper = GradientClipper::new(3.0);
        clipper.clip(&mut grads);
        assert!((grads[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn clipper_exact_max_norm() {
        let mut grads = [3.0, 4.0]; // norm = 5
        let clipper = GradientClipper::new(5.0);
        clipper.clip(&mut grads);
        // Should remain unchanged (norm == max_norm).
        assert!((grads[0] - 3.0).abs() < 1e-5);
        assert!((grads[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn clipper_large_gradients() {
        let mut grads = [1e6, -1e6];
        let clipper = GradientClipper::new(1.0);
        clipper.clip(&mut grads);
        let norm = GradientClipper::l2_norm(&grads);
        assert!((norm - 1.0).abs() < 1e-4);
    }

    // -- LossScaler --------------------------------------------------------

    #[test]
    fn scaler_initial_scale() {
        let scaler = LossScaler::with_defaults();
        assert_eq!(scaler.scale(), 1024.0);
    }

    #[test]
    fn scaler_scale_loss() {
        let scaler = LossScaler::with_defaults();
        assert_eq!(scaler.scale_loss(2.0), 2048.0);
    }

    #[test]
    fn scaler_unscale_gradients() {
        let scaler = LossScaler::new(4.0, 2.0, 0.5, 10);
        let mut grads = [8.0, 16.0, -4.0];
        scaler.unscale_gradients(&mut grads);
        assert_eq!(grads, [2.0, 4.0, -1.0]);
    }

    #[test]
    fn scaler_halves_on_overflow() {
        let mut scaler = LossScaler::new(1024.0, 2.0, 0.5, 10);
        scaler.update(true);
        assert_eq!(scaler.scale(), 512.0);
    }

    #[test]
    fn scaler_doubles_after_growth_interval() {
        let mut scaler = LossScaler::new(1024.0, 2.0, 0.5, 3);
        scaler.update(false);
        scaler.update(false);
        assert_eq!(scaler.scale(), 1024.0); // not yet
        scaler.update(false);
        assert_eq!(scaler.scale(), 2048.0); // 3 clean steps → double
    }

    #[test]
    fn scaler_overflow_resets_consecutive() {
        let mut scaler = LossScaler::new(1024.0, 2.0, 0.5, 3);
        scaler.update(false);
        scaler.update(false);
        scaler.update(true); // overflow resets counter
        assert_eq!(scaler.scale(), 512.0);
        // Three more clean steps needed from zero.
        scaler.update(false);
        scaler.update(false);
        scaler.update(false);
        assert_eq!(scaler.scale(), 1024.0);
    }

    #[test]
    fn scaler_detect_overflow_nan() {
        assert!(LossScaler::detect_overflow(&[1.0, f32::NAN, 3.0]));
    }

    #[test]
    fn scaler_detect_overflow_inf() {
        assert!(LossScaler::detect_overflow(&[1.0, f32::INFINITY]));
    }

    #[test]
    fn scaler_detect_overflow_clean() {
        assert!(!LossScaler::detect_overflow(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn scaler_multiple_overflows_keep_halving() {
        let mut scaler = LossScaler::new(1024.0, 2.0, 0.5, 10);
        scaler.update(true);
        scaler.update(true);
        scaler.update(true);
        assert_eq!(scaler.scale(), 128.0);
    }

    // -- NumericalProfile --------------------------------------------------

    #[test]
    fn profile_basic_stats() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let p = NumericalProfile::compute(&data);
        assert_eq!(p.min, 1.0);
        assert_eq!(p.max, 5.0);
        assert!((p.mean - 3.0).abs() < 1e-5);
        assert_eq!(p.nan_count, 0);
        assert_eq!(p.inf_count, 0);
        assert_eq!(p.element_count, 5);
        assert!(p.is_clean());
    }

    #[test]
    fn profile_with_nans() {
        let data = [1.0, f32::NAN, 3.0, f32::NAN];
        let p = NumericalProfile::compute(&data);
        assert_eq!(p.nan_count, 2);
        assert!(!p.is_clean());
        // Mean is computed over finite values only.
        assert!((p.mean - 2.0).abs() < 1e-5);
    }

    #[test]
    fn profile_with_infs() {
        let data = [1.0, f32::INFINITY, 3.0];
        let p = NumericalProfile::compute(&data);
        assert_eq!(p.inf_count, 1);
        assert!(!p.is_clean());
        assert!((p.mean - 2.0).abs() < 1e-5);
    }

    #[test]
    fn profile_empty() {
        let data: [f32; 0] = [];
        let p = NumericalProfile::compute(&data);
        assert_eq!(p.element_count, 0);
        assert_eq!(p.mean, 0.0);
        assert!(p.is_clean());
    }

    #[test]
    fn profile_all_nan() {
        let data = [f32::NAN; 4];
        let p = NumericalProfile::compute(&data);
        assert_eq!(p.nan_count, 4);
        assert_eq!(p.min, 0.0);
        assert_eq!(p.max, 0.0);
        assert_eq!(p.mean, 0.0);
    }

    #[test]
    fn profile_std_dev() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let p = NumericalProfile::compute(&data);
        // population std dev ≈ 2.0
        assert!((p.std_dev - 2.0).abs() < 0.01);
    }

    #[test]
    fn profile_single_element() {
        let data = [42.0];
        let p = NumericalProfile::compute(&data);
        assert_eq!(p.min, 42.0);
        assert_eq!(p.max, 42.0);
        assert_eq!(p.mean, 42.0);
        assert_eq!(p.std_dev, 0.0);
    }

    #[test]
    fn profile_display_format() {
        let data = [1.0, 2.0, 3.0];
        let p = NumericalProfile::compute(&data);
        let s = format!("{p}");
        assert!(s.contains("Profile("));
        assert!(s.contains("nan=0"));
        assert!(s.contains("inf=0"));
    }

    // -- ComparisonValidator -----------------------------------------------

    #[test]
    fn comparison_identical() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.0];
        let v = ComparisonValidator::with_defaults();
        let r = v.validate(&a, &b);
        assert!(r.all_close);
        assert_eq!(r.mismatch_count, 0);
    }

    #[test]
    fn comparison_within_tolerance() {
        let a = [1.0, 2.0001, 3.0];
        let b = [1.0, 2.0, 3.0];
        let v = ComparisonValidator::with_tolerances(1e-3, 1e-3);
        let r = v.validate(&a, &b);
        assert!(r.all_close);
    }

    #[test]
    fn comparison_outside_tolerance() {
        let a = [1.0, 3.0, 3.0];
        let b = [1.0, 2.0, 3.0];
        let v = ComparisonValidator::with_tolerances(1e-5, 1e-5);
        let r = v.validate(&a, &b);
        assert!(!r.all_close);
        assert_eq!(r.mismatch_count, 1);
    }

    #[test]
    fn comparison_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let v = ComparisonValidator::with_defaults();
        let r = v.validate(&a, &b);
        assert!(r.all_close);
        assert_eq!(r.total_elements, 0);
    }

    #[test]
    fn comparison_nan_both_sides_match() {
        let a = [f32::NAN];
        let b = [f32::NAN];
        let v = ComparisonValidator::with_defaults();
        let r = v.validate(&a, &b);
        assert!(r.all_close);
    }

    #[test]
    fn comparison_nan_one_side_mismatch() {
        let a = [f32::NAN];
        let b = [1.0];
        let v = ComparisonValidator::with_defaults();
        let r = v.validate(&a, &b);
        assert!(!r.all_close);
        assert_eq!(r.mismatch_count, 1);
    }

    #[test]
    fn comparison_max_abs_diff() {
        let a = [1.0, 5.0];
        let b = [1.0, 2.0];
        let v = ComparisonValidator::with_tolerances(10.0, 10.0);
        let r = v.validate(&a, &b);
        assert!((r.max_abs_diff - 3.0).abs() < 1e-5);
    }

    #[test]
    fn comparison_max_rel_diff() {
        let a = [2.0];
        let b = [1.0];
        let v = ComparisonValidator::with_tolerances(10.0, 10.0);
        let r = v.validate(&a, &b);
        assert!((r.max_rel_diff - 1.0).abs() < 1e-5);
    }

    #[test]
    fn comparison_display() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0];
        let v = ComparisonValidator::with_defaults();
        let r = v.validate(&a, &b);
        let s = format!("{r}");
        assert!(s.contains("Comparison("));
    }

    // -- CPU reference helpers ---------------------------------------------

    #[test]
    fn cpu_count_nan_works() {
        assert_eq!(cpu_count_nan(&[1.0, f32::NAN, 2.0, f32::NAN]), 2);
        assert_eq!(cpu_count_nan(&[1.0, 2.0]), 0);
        assert_eq!(cpu_count_nan(&[]), 0);
    }

    #[test]
    fn cpu_count_inf_works() {
        assert_eq!(cpu_count_inf(&[f32::INFINITY, 1.0, f32::NEG_INFINITY]), 2);
        assert_eq!(cpu_count_inf(&[1.0, 2.0]), 0);
    }

    #[test]
    fn cpu_sanitize_works() {
        let config = StabilityConfig {
            max_value: 10.0,
            min_value: -10.0,
            nan_replacement: -1.0,
            inf_replacement: 99.0,
            ..StabilityConfig::default()
        };
        let mut data = [f32::NAN, f32::INFINITY, 20.0, -20.0, 5.0];
        cpu_sanitize(&mut data, &config);
        assert_eq!(data, [-1.0, 99.0, 10.0, -10.0, 5.0]);
    }

    #[test]
    fn cpu_l2_norm_works() {
        let data = [3.0, 4.0];
        assert!((cpu_l2_norm(&data) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn cpu_l2_norm_empty() {
        assert_eq!(cpu_l2_norm(&[]), 0.0);
    }

    // -- OpenCL kernel source sanity checks --------------------------------

    #[test]
    fn opencl_nan_inf_kernel_contains_expected_functions() {
        assert!(OPENCL_NAN_INF_DETECT_KERNEL.contains("nan_inf_detect"));
        assert!(OPENCL_NAN_INF_DETECT_KERNEL.contains("isnan"));
        assert!(OPENCL_NAN_INF_DETECT_KERNEL.contains("isinf"));
    }

    #[test]
    fn opencl_sanitize_kernel_contains_clamp() {
        assert!(OPENCL_SANITIZE_KERNEL.contains("sanitize"));
        assert!(OPENCL_SANITIZE_KERNEL.contains("clamp"));
        assert!(OPENCL_SANITIZE_KERNEL.contains("isnan"));
        assert!(OPENCL_SANITIZE_KERNEL.contains("isinf"));
    }

    #[test]
    fn opencl_l2_norm_kernel_has_reduction() {
        assert!(OPENCL_L2_NORM_KERNEL.contains("l2_norm_partial"));
        assert!(OPENCL_L2_NORM_KERNEL.contains("barrier"));
    }

    // -- Config defaults ---------------------------------------------------

    #[test]
    fn config_defaults_sensible() {
        let c = StabilityConfig::default();
        assert!(c.max_value > 0.0);
        assert!(c.min_value < 0.0);
        assert!(c.atol > 0.0);
        assert!(c.rtol > 0.0);
        assert_eq!(c.nan_replacement, 0.0);
    }

    // -- Integration / cross-component ------------------------------------

    #[test]
    fn guard_then_profile_is_clean() {
        let mut data = [f32::NAN, f32::INFINITY, -1e38, 1e38, 0.0];
        let mut guard = StabilityGuard::with_defaults();
        guard.sanitize(&mut data);
        let profile = NumericalProfile::compute(&data);
        assert!(profile.is_clean());
    }

    #[test]
    fn comparison_after_sanitize_agrees() {
        let config = StabilityConfig::default();
        let mut gpu_result = [f32::NAN, 1.0, f32::INFINITY, 3.0];
        let mut cpu_result = gpu_result;

        let mut guard = StabilityGuard::new(config.clone());
        guard.sanitize(&mut gpu_result);
        cpu_sanitize(&mut cpu_result, &config);

        let v = ComparisonValidator::with_defaults();
        let r = v.validate(&gpu_result, &cpu_result);
        assert!(r.all_close);
    }

    #[test]
    fn end_to_end_guard_clip_profile() {
        let mut data = [f32::NAN, 100.0, -100.0, f32::INFINITY, 5.0];
        let config =
            StabilityConfig { max_value: 50.0, min_value: -50.0, ..StabilityConfig::default() };
        let mut guard = StabilityGuard::new(config);
        guard.sanitize(&mut data);
        assert!(data.iter().all(|v| v.is_finite()));

        let mut clipper_data = data.to_vec();
        let clipper = GradientClipper::new(10.0);
        clipper.clip(&mut clipper_data);
        let norm = GradientClipper::l2_norm(&clipper_data);
        assert!(norm <= 10.0 + 1e-5);

        let profile = NumericalProfile::compute(&clipper_data);
        assert!(profile.is_clean());
    }
}
