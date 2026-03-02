//! Cross-backend numerical verification framework.
//!
//! Compares CPU reference outputs against OpenCL GPU outputs to validate
//! correctness with configurable tolerances for Intel Arc A770.
//!
//! # Overview
//!
//! - [`ToleranceSpec`]: absolute, relative, and ULP tolerance thresholds
//! - [`NumericalVerifier`]: element-wise comparison of two `f32` slices
//! - [`OperationVerifier`]: per-operation tolerance profiles (matmul, softmax, …)
//! - [`DiffHistogram`]: error distribution bucketed by magnitude
//! - [`RegressionTracker`]: tracks numerical drift across successive runs

use std::collections::HashMap;
use std::fmt;

// ── Tolerance specification ──────────────────────────────────────

/// Configurable tolerance thresholds for numerical comparison.
#[derive(Debug, Clone, Copy)]
pub struct ToleranceSpec {
    /// Maximum allowable absolute difference.
    pub abs_tol: f32,
    /// Maximum allowable relative difference.
    pub rel_tol: f32,
    /// Maximum allowable ULP (units in the last place) difference.
    pub ulp_tol: u32,
    /// Maximum number of element failures before the comparison is rejected.
    pub max_failures: usize,
}

impl ToleranceSpec {
    /// Strict tolerance suitable for elementwise operations (add, relu).
    pub fn strict() -> Self {
        Self { abs_tol: 1e-6, rel_tol: 1e-5, ulp_tol: 4, max_failures: 0 }
    }

    /// Default tolerance for most operations.
    pub fn default_tol() -> Self {
        Self { abs_tol: 1e-5, rel_tol: 1e-4, ulp_tol: 16, max_failures: 0 }
    }

    /// Relaxed tolerance for matmul and attention (accumulation error).
    pub fn relaxed() -> Self {
        Self { abs_tol: 1e-3, rel_tol: 1e-2, ulp_tol: 256, max_failures: 0 }
    }

    /// Builder: set `max_failures`.
    pub fn with_max_failures(mut self, n: usize) -> Self {
        self.max_failures = n;
        self
    }
}

impl Default for ToleranceSpec {
    fn default() -> Self {
        Self::default_tol()
    }
}

// ── Element-level comparison result ──────────────────────────────

/// Result of comparing a single pair of `f32` values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComparisonResult {
    /// Values are bit-identical.
    Match,
    /// Within absolute tolerance; carries the absolute difference.
    AbsDiff(f32),
    /// Within relative tolerance; carries the relative difference.
    RelDiff(f32),
    /// Within ULP tolerance; carries the ULP distance.
    UlpDiff(u32),
    /// One value is NaN while the other is not (or both NaN when not expected).
    NanMismatch,
}

impl ComparisonResult {
    /// `true` when the comparison indicates a failure.
    pub fn is_failure(&self) -> bool {
        matches!(self, ComparisonResult::NanMismatch)
    }
}

// ── Diff histogram ───────────────────────────────────────────────

/// Bucket boundaries for error-magnitude histogram.
const BUCKET_BOUNDS: [f32; 7] = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];

/// Error-distribution histogram with fixed logarithmic buckets.
#[derive(Debug, Clone)]
pub struct DiffHistogram {
    /// Counts for buckets `[0, 1e-7)`, `[1e-7, 1e-6)`, … `[1e-1, +∞)`.
    pub buckets: [u64; 8],
}

impl DiffHistogram {
    pub fn new() -> Self {
        Self { buckets: [0; 8] }
    }

    /// Record an absolute difference value.
    pub fn record(&mut self, abs_diff: f32) {
        let idx = BUCKET_BOUNDS.iter().position(|&b| abs_diff < b).unwrap_or(BUCKET_BOUNDS.len());
        self.buckets[idx] += 1;
    }

    /// Total number of recorded values.
    pub fn total(&self) -> u64 {
        self.buckets.iter().sum()
    }

    /// Labels for each bucket (useful for display).
    pub fn labels() -> [&'static str; 8] {
        ["<1e-7", "<1e-6", "<1e-5", "<1e-4", "<1e-3", "<1e-2", "<1e-1", ">=1e-1"]
    }
}

impl Default for DiffHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DiffHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let labels = Self::labels();
        for (label, &count) in labels.iter().zip(self.buckets.iter()) {
            writeln!(f, "  {label:>8}: {count}")?;
        }
        Ok(())
    }
}

// ── Verification report ──────────────────────────────────────────

/// Aggregate report for a slice-level numerical comparison.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Total number of elements compared.
    pub total_elements: usize,
    /// Number of elements that matched within tolerance.
    pub matches: usize,
    /// Number of elements that failed tolerance checks.
    pub failures: usize,
    /// Largest absolute difference observed.
    pub max_abs_diff: f32,
    /// Largest relative difference observed.
    pub max_rel_diff: f32,
    /// Error-distribution histogram.
    pub histogram: DiffHistogram,
}

impl VerificationReport {
    /// `true` when the comparison passed the tolerance spec.
    pub fn passed(&self, spec: &ToleranceSpec) -> bool {
        self.failures <= spec.max_failures
    }

    /// Fraction of elements that matched.
    pub fn match_rate(&self) -> f64 {
        if self.total_elements == 0 {
            return 1.0;
        }
        self.matches as f64 / self.total_elements as f64
    }
}

impl fmt::Display for VerificationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Verification: {}/{} pass", self.matches, self.total_elements)?;
        writeln!(f, "  failures:     {}", self.failures)?;
        writeln!(f, "  max abs diff: {:.e}", self.max_abs_diff)?;
        writeln!(f, "  max rel diff: {:.e}", self.max_rel_diff)?;
        write!(f, "  histogram:\n{}", self.histogram)
    }
}

// ── ULP helpers ──────────────────────────────────────────────────

/// Compute ULP (units in the last place) distance between two `f32` values.
///
/// Both values must be finite and have the same sign; returns `u32::MAX` for
/// sign mismatches, infinities, or NaN.
fn ulp_distance(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return u32::MAX;
    }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    // Sign-magnitude → two's-complement so subtraction works across zero
    let ai = if ai < 0 { i32::MIN.wrapping_sub(ai) } else { ai };
    let bi = if bi < 0 { i32::MIN.wrapping_sub(bi) } else { bi };
    (ai.wrapping_sub(bi)).unsigned_abs()
}

// ── Numerical verifier ───────────────────────────────────────────

/// Compares two `f32` slices element-wise against a [`ToleranceSpec`].
pub struct NumericalVerifier {
    spec: ToleranceSpec,
}

impl NumericalVerifier {
    pub fn new(spec: ToleranceSpec) -> Self {
        Self { spec }
    }

    /// Compare a single pair of values.
    pub fn compare_element(&self, reference: f32, candidate: f32) -> ComparisonResult {
        // NaN handling
        if reference.is_nan() || candidate.is_nan() {
            return if reference.is_nan() && candidate.is_nan() {
                ComparisonResult::Match
            } else {
                ComparisonResult::NanMismatch
            };
        }

        // Exact bit-match (also covers ±0, ±inf when identical)
        if reference.to_bits() == candidate.to_bits() {
            return ComparisonResult::Match;
        }

        let abs_diff = (reference - candidate).abs();

        // Absolute tolerance check
        if abs_diff <= self.spec.abs_tol {
            return ComparisonResult::AbsDiff(abs_diff);
        }

        // Relative tolerance check (denominator guards against div-by-zero)
        let denom = reference.abs().max(candidate.abs()).max(f32::MIN_POSITIVE);
        let rel_diff = abs_diff / denom;
        if rel_diff <= self.spec.rel_tol {
            return ComparisonResult::RelDiff(rel_diff);
        }

        // ULP tolerance check
        let ulp = ulp_distance(reference, candidate);
        if ulp <= self.spec.ulp_tol {
            return ComparisonResult::UlpDiff(ulp);
        }

        // Infinity mismatch
        if reference.is_infinite() || candidate.is_infinite() {
            return ComparisonResult::NanMismatch;
        }

        // Failed all tolerance gates — report as AbsDiff failure but the
        // caller (verify_slices) will count it as a failure since it
        // exceeds thresholds. We still tag it so the histogram gets data.
        ComparisonResult::AbsDiff(abs_diff)
    }

    /// Returns `true` if a [`ComparisonResult`] passes the tolerance spec.
    fn element_passes(&self, result: &ComparisonResult) -> bool {
        match *result {
            ComparisonResult::Match => true,
            ComparisonResult::AbsDiff(d) => d <= self.spec.abs_tol,
            ComparisonResult::RelDiff(d) => d <= self.spec.rel_tol,
            ComparisonResult::UlpDiff(u) => u <= self.spec.ulp_tol,
            ComparisonResult::NanMismatch => false,
        }
    }

    /// Compare two slices and produce a [`VerificationReport`].
    ///
    /// # Panics
    ///
    /// Panics if the slices differ in length.
    pub fn verify_slices(&self, reference: &[f32], candidate: &[f32]) -> VerificationReport {
        assert_eq!(
            reference.len(),
            candidate.len(),
            "slice length mismatch: {} vs {}",
            reference.len(),
            candidate.len()
        );

        let total_elements = reference.len();
        let mut matches: usize = 0;
        let mut failures: usize = 0;
        let mut max_abs_diff: f32 = 0.0;
        let mut max_rel_diff: f32 = 0.0;
        let mut histogram = DiffHistogram::new();

        for (&r, &c) in reference.iter().zip(candidate.iter()) {
            let cmp = self.compare_element(r, c);
            let abs_diff = if r.is_nan() || c.is_nan() { 0.0 } else { (r - c).abs() };
            let denom = r.abs().max(c.abs()).max(f32::MIN_POSITIVE);
            let rel_diff = if r.is_nan() || c.is_nan() { 0.0 } else { abs_diff / denom };

            histogram.record(abs_diff);

            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }
            if rel_diff > max_rel_diff {
                max_rel_diff = rel_diff;
            }

            if self.element_passes(&cmp) {
                matches += 1;
            } else {
                failures += 1;
            }
        }

        VerificationReport {
            total_elements,
            matches,
            failures,
            max_abs_diff,
            max_rel_diff,
            histogram,
        }
    }
}

// ── Operation verifier ───────────────────────────────────────────

/// Operation type determines which tolerance profile to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    Matmul,
    Softmax,
    RmsNorm,
    Rope,
    Attention,
    Elementwise,
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            OpKind::Matmul => "matmul",
            OpKind::Softmax => "softmax",
            OpKind::RmsNorm => "rmsnorm",
            OpKind::Rope => "rope",
            OpKind::Attention => "attention",
            OpKind::Elementwise => "elementwise",
        };
        f.write_str(name)
    }
}

/// Verifies specific operations using per-operation tolerance profiles.
pub struct OperationVerifier {
    profiles: HashMap<OpKind, ToleranceSpec>,
}

impl OperationVerifier {
    /// Create with default per-operation profiles tuned for Intel Arc A770.
    pub fn new() -> Self {
        let mut profiles = HashMap::new();
        profiles.insert(OpKind::Elementwise, ToleranceSpec::strict());
        profiles.insert(OpKind::Softmax, ToleranceSpec::default_tol());
        profiles.insert(OpKind::RmsNorm, ToleranceSpec::default_tol());
        profiles.insert(OpKind::Rope, ToleranceSpec::default_tol());
        profiles.insert(OpKind::Matmul, ToleranceSpec::relaxed());
        profiles.insert(OpKind::Attention, ToleranceSpec::relaxed());
        Self { profiles }
    }

    /// Override the tolerance profile for a given operation.
    pub fn set_tolerance(&mut self, op: OpKind, spec: ToleranceSpec) {
        self.profiles.insert(op, spec);
    }

    /// Retrieve the tolerance profile for an operation.
    pub fn tolerance_for(&self, op: OpKind) -> ToleranceSpec {
        self.profiles.get(&op).copied().unwrap_or_default()
    }

    /// Verify an operation's output against a CPU reference.
    pub fn verify(&self, op: OpKind, reference: &[f32], candidate: &[f32]) -> VerificationReport {
        let spec = self.tolerance_for(op);
        let verifier = NumericalVerifier::new(spec);
        verifier.verify_slices(reference, candidate)
    }

    /// Verify and return whether the result passes.
    pub fn verify_passes(&self, op: OpKind, reference: &[f32], candidate: &[f32]) -> bool {
        let spec = self.tolerance_for(op);
        let report = self.verify(op, reference, candidate);
        report.passed(&spec)
    }
}

impl Default for OperationVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ── Regression tracker ───────────────────────────────────────────

/// A single snapshot of numerical accuracy for regression tracking.
#[derive(Debug, Clone)]
pub struct AccuracySnapshot {
    /// Identifier for this run (e.g., git commit, timestamp).
    pub run_id: String,
    /// Maximum absolute difference observed.
    pub max_abs_diff: f32,
    /// Maximum relative difference observed.
    pub max_rel_diff: f32,
    /// Total failures observed.
    pub failures: usize,
}

/// Tracks numerical drift across successive runs.
pub struct RegressionTracker {
    history: Vec<AccuracySnapshot>,
    /// Maximum allowable increase in `max_abs_diff` between consecutive runs.
    pub drift_threshold: f32,
}

impl RegressionTracker {
    pub fn new(drift_threshold: f32) -> Self {
        Self { history: Vec::new(), drift_threshold }
    }

    /// Record a new snapshot.
    pub fn record(&mut self, snapshot: AccuracySnapshot) {
        self.history.push(snapshot);
    }

    /// Record from a [`VerificationReport`].
    pub fn record_report(&mut self, run_id: impl Into<String>, report: &VerificationReport) {
        self.history.push(AccuracySnapshot {
            run_id: run_id.into(),
            max_abs_diff: report.max_abs_diff,
            max_rel_diff: report.max_rel_diff,
            failures: report.failures,
        });
    }

    /// Check whether the latest run regressed relative to the previous one.
    ///
    /// Returns `Some((prev, current))` absolute-diff pair if regression
    /// detected, or `None` if no regression (or fewer than 2 snapshots).
    pub fn check_regression(&self) -> Option<(f32, f32)> {
        if self.history.len() < 2 {
            return None;
        }
        let prev = &self.history[self.history.len() - 2];
        let curr = &self.history[self.history.len() - 1];
        let drift = curr.max_abs_diff - prev.max_abs_diff;
        if drift > self.drift_threshold {
            Some((prev.max_abs_diff, curr.max_abs_diff))
        } else {
            None
        }
    }

    /// Return the full snapshot history.
    pub fn history(&self) -> &[AccuracySnapshot] {
        &self.history
    }

    /// Number of recorded snapshots.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether the tracker has no snapshots.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

// ── CPU reference implementations ────────────────────────────────

/// CPU reference: softmax over a 1-D slice (in-place).
pub fn cpu_softmax(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in data.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in data.iter_mut() {
            *v /= sum;
        }
    }
}

/// CPU reference: RMS normalization (out-of-place).
pub fn cpu_rmsnorm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(input.len(), weight.len());
    let n = input.len() as f32;
    let ss: f32 = input.iter().map(|&x| x * x).sum();
    let rms = (ss / n + eps).sqrt();
    input.iter().zip(weight.iter()).map(|(&x, &w)| (x / rms) * w).collect()
}

/// CPU reference: RoPE (Rotary Position Embedding) for a single position.
///
/// Applies rotation to pairs `(x[2i], x[2i+1])` using the standard formula.
pub fn cpu_rope(data: &mut [f32], position: usize, head_dim: usize, theta: f32) {
    assert!(head_dim % 2 == 0, "head_dim must be even");
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf(i as f32 / half as f32);
        let angle = position as f32 * freq;
        let (sin_a, cos_a) = (angle.sin(), angle.cos());
        let idx = 2 * i;
        if idx + 1 < data.len() {
            let x0 = data[idx];
            let x1 = data[idx + 1];
            data[idx] = x0 * cos_a - x1 * sin_a;
            data[idx + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/// CPU reference: naive matrix multiply `C = A × B` (row-major).
///
/// `A` is `m×k`, `B` is `k×n`, `C` is `m×n`.
pub fn cpu_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // ToleranceSpec
    // -----------------------------------------------------------------

    #[test]
    fn tolerance_strict_is_tighter_than_relaxed() {
        let strict = ToleranceSpec::strict();
        let relaxed = ToleranceSpec::relaxed();
        assert!(strict.abs_tol < relaxed.abs_tol);
        assert!(strict.rel_tol < relaxed.rel_tol);
        assert!(strict.ulp_tol < relaxed.ulp_tol);
    }

    #[test]
    fn tolerance_with_max_failures_builder() {
        let spec = ToleranceSpec::strict().with_max_failures(5);
        assert_eq!(spec.max_failures, 5);
        // Tolerances unchanged
        assert_eq!(spec.abs_tol, ToleranceSpec::strict().abs_tol);
    }

    #[test]
    fn tolerance_default_equals_default_tol() {
        let d = ToleranceSpec::default();
        let dt = ToleranceSpec::default_tol();
        assert_eq!(d.abs_tol, dt.abs_tol);
        assert_eq!(d.rel_tol, dt.rel_tol);
        assert_eq!(d.ulp_tol, dt.ulp_tol);
    }

    // -----------------------------------------------------------------
    // ComparisonResult
    // -----------------------------------------------------------------

    #[test]
    fn comparison_result_is_failure() {
        assert!(!ComparisonResult::Match.is_failure());
        assert!(!ComparisonResult::AbsDiff(0.001).is_failure());
        assert!(!ComparisonResult::RelDiff(0.001).is_failure());
        assert!(!ComparisonResult::UlpDiff(1).is_failure());
        assert!(ComparisonResult::NanMismatch.is_failure());
    }

    // -----------------------------------------------------------------
    // ULP distance
    // -----------------------------------------------------------------

    #[test]
    fn ulp_distance_identical() {
        assert_eq!(ulp_distance(1.0, 1.0), 0);
        assert_eq!(ulp_distance(0.0, 0.0), 0);
        assert_eq!(ulp_distance(-1.0, -1.0), 0);
    }

    #[test]
    fn ulp_distance_adjacent() {
        let a: f32 = 1.0;
        let b = f32::from_bits(a.to_bits() + 1);
        assert_eq!(ulp_distance(a, b), 1);
    }

    #[test]
    fn ulp_distance_nan_returns_max() {
        assert_eq!(ulp_distance(f32::NAN, 1.0), u32::MAX);
        assert_eq!(ulp_distance(1.0, f32::NAN), u32::MAX);
    }

    #[test]
    fn ulp_distance_inf_returns_max() {
        assert_eq!(ulp_distance(f32::INFINITY, 1.0), u32::MAX);
        assert_eq!(ulp_distance(1.0, f32::NEG_INFINITY), u32::MAX);
    }

    #[test]
    fn ulp_distance_symmetry() {
        let a = 1.0f32;
        let b = 1.0f32 + 1e-6;
        assert_eq!(ulp_distance(a, b), ulp_distance(b, a));
    }

    #[test]
    fn ulp_distance_across_zero() {
        // Small positive vs small negative
        let a: f32 = 1e-38;
        let b: f32 = -1e-38;
        let d = ulp_distance(a, b);
        // Distance should be non-trivial but finite
        assert!(d > 0);
        assert!(d < u32::MAX);
    }

    // -----------------------------------------------------------------
    // NumericalVerifier — element comparison
    // -----------------------------------------------------------------

    #[test]
    fn verify_exact_match() {
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        assert_eq!(v.compare_element(1.0, 1.0), ComparisonResult::Match);
    }

    #[test]
    fn verify_both_nan_is_match() {
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        assert_eq!(v.compare_element(f32::NAN, f32::NAN), ComparisonResult::Match);
    }

    #[test]
    fn verify_nan_vs_number_is_mismatch() {
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        assert_eq!(v.compare_element(f32::NAN, 1.0), ComparisonResult::NanMismatch);
        assert_eq!(v.compare_element(1.0, f32::NAN), ComparisonResult::NanMismatch);
    }

    #[test]
    fn verify_within_abs_tol() {
        let spec = ToleranceSpec { abs_tol: 0.01, rel_tol: 0.0, ulp_tol: 0, max_failures: 0 };
        let v = NumericalVerifier::new(spec);
        match v.compare_element(1.0, 1.005) {
            ComparisonResult::AbsDiff(d) => assert!(d <= 0.01),
            other => panic!("expected AbsDiff, got {other:?}"),
        }
    }

    #[test]
    fn verify_within_rel_tol() {
        // abs_tol very tight so only rel_tol matches
        let spec = ToleranceSpec { abs_tol: 1e-10, rel_tol: 0.01, ulp_tol: 0, max_failures: 0 };
        let v = NumericalVerifier::new(spec);
        let r = 100.0;
        let c = 100.5; // rel diff = 0.5/100.5 ≈ 0.005
        match v.compare_element(r, c) {
            ComparisonResult::RelDiff(d) => assert!(d <= 0.01),
            other => panic!("expected RelDiff, got {other:?}"),
        }
    }

    #[test]
    fn verify_within_ulp_tol() {
        let a: f32 = 1.0;
        let b = f32::from_bits(a.to_bits() + 3);
        let spec = ToleranceSpec { abs_tol: 0.0, rel_tol: 0.0, ulp_tol: 4, max_failures: 0 };
        let v = NumericalVerifier::new(spec);
        match v.compare_element(a, b) {
            ComparisonResult::UlpDiff(u) => assert!(u <= 4),
            other => panic!("expected UlpDiff, got {other:?}"),
        }
    }

    #[test]
    fn verify_exceeds_all_tolerances() {
        let spec = ToleranceSpec { abs_tol: 1e-10, rel_tol: 1e-10, ulp_tol: 0, max_failures: 0 };
        let v = NumericalVerifier::new(spec);
        let cmp = v.compare_element(1.0, 2.0);
        assert!(!v.element_passes(&cmp));
    }

    #[test]
    fn verify_positive_negative_zero() {
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        // +0.0 and -0.0 differ in bits but abs_diff is 0
        let cmp = v.compare_element(0.0, -0.0);
        assert!(v.element_passes(&cmp));
    }

    #[test]
    fn verify_inf_match() {
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        assert_eq!(v.compare_element(f32::INFINITY, f32::INFINITY), ComparisonResult::Match);
        assert_eq!(
            v.compare_element(f32::NEG_INFINITY, f32::NEG_INFINITY),
            ComparisonResult::Match
        );
    }

    #[test]
    fn verify_inf_mismatch() {
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        let cmp = v.compare_element(f32::INFINITY, f32::NEG_INFINITY);
        assert!(!v.element_passes(&cmp));
    }

    #[test]
    fn verify_inf_vs_finite() {
        let v = NumericalVerifier::new(ToleranceSpec::relaxed());
        let cmp = v.compare_element(f32::INFINITY, 1.0);
        assert!(!v.element_passes(&cmp));
    }

    // -----------------------------------------------------------------
    // NumericalVerifier — slice comparison
    // -----------------------------------------------------------------

    #[test]
    fn verify_slices_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        let report = v.verify_slices(&data, &data);
        assert_eq!(report.total_elements, 4);
        assert_eq!(report.matches, 4);
        assert_eq!(report.failures, 0);
        assert_eq!(report.max_abs_diff, 0.0);
    }

    #[test]
    fn verify_slices_within_tolerance() {
        let reference = vec![1.0, 2.0, 3.0];
        let candidate = vec![1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 5e-8];
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        let report = v.verify_slices(&reference, &candidate);
        assert_eq!(report.failures, 0);
    }

    #[test]
    fn verify_slices_some_failures() {
        let reference = vec![1.0, 2.0, 3.0];
        let candidate = vec![1.0, 5.0, 3.0]; // middle element way off
        let spec = ToleranceSpec::strict();
        let v = NumericalVerifier::new(spec);
        let report = v.verify_slices(&reference, &candidate);
        assert_eq!(report.failures, 1);
        assert_eq!(report.matches, 2);
        assert!((report.max_abs_diff - 3.0).abs() < 1e-6);
    }

    #[test]
    fn verify_slices_max_failures_allows_some() {
        let reference = vec![1.0, 2.0, 3.0, 4.0];
        let candidate = vec![1.0, 200.0, 3.0, 400.0]; // 2 failures
        let spec = ToleranceSpec::strict().with_max_failures(2);
        let v = NumericalVerifier::new(spec);
        let report = v.verify_slices(&reference, &candidate);
        assert_eq!(report.failures, 2);
        assert!(report.passed(&spec));
    }

    #[test]
    fn verify_slices_empty() {
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        let report = v.verify_slices(&[], &[]);
        assert_eq!(report.total_elements, 0);
        assert_eq!(report.failures, 0);
        assert_eq!(report.match_rate(), 1.0);
    }

    #[test]
    fn verify_slices_single_element() {
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        let report = v.verify_slices(&[42.0], &[42.0]);
        assert_eq!(report.total_elements, 1);
        assert_eq!(report.matches, 1);
    }

    #[test]
    fn verify_slices_all_zeros() {
        let zeros = vec![0.0f32; 100];
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        let report = v.verify_slices(&zeros, &zeros);
        assert_eq!(report.failures, 0);
        assert_eq!(report.matches, 100);
    }

    #[test]
    fn verify_slices_all_same_value() {
        let vals = vec![3.14f32; 50];
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        let report = v.verify_slices(&vals, &vals);
        assert_eq!(report.failures, 0);
    }

    #[test]
    #[should_panic(expected = "slice length mismatch")]
    fn verify_slices_length_mismatch_panics() {
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        v.verify_slices(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn verify_slices_nan_both_sides() {
        let reference = vec![f32::NAN, 1.0, f32::NAN];
        let candidate = vec![f32::NAN, 1.0, f32::NAN];
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        let report = v.verify_slices(&reference, &candidate);
        assert_eq!(report.failures, 0);
    }

    #[test]
    fn verify_slices_nan_one_side() {
        let reference = vec![1.0, f32::NAN, 3.0];
        let candidate = vec![1.0, 2.0, 3.0];
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        let report = v.verify_slices(&reference, &candidate);
        assert_eq!(report.failures, 1);
    }

    #[test]
    fn verify_slices_mixed_inf() {
        let reference = vec![1.0, f32::INFINITY, 3.0];
        let candidate = vec![1.0, f32::INFINITY, 3.0];
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        let report = v.verify_slices(&reference, &candidate);
        assert_eq!(report.failures, 0);
    }

    // -----------------------------------------------------------------
    // DiffHistogram
    // -----------------------------------------------------------------

    #[test]
    fn histogram_empty() {
        let h = DiffHistogram::new();
        assert_eq!(h.total(), 0);
        assert!(h.buckets.iter().all(|&b| b == 0));
    }

    #[test]
    fn histogram_smallest_bucket() {
        let mut h = DiffHistogram::new();
        h.record(1e-8); // < 1e-7
        assert_eq!(h.buckets[0], 1);
        assert_eq!(h.total(), 1);
    }

    #[test]
    fn histogram_largest_bucket() {
        let mut h = DiffHistogram::new();
        h.record(0.5); // >= 1e-1
        assert_eq!(h.buckets[7], 1);
    }

    #[test]
    fn histogram_each_bucket() {
        let mut h = DiffHistogram::new();
        // Values placed into each bucket
        let values = [1e-8, 5e-7, 5e-6, 5e-5, 5e-4, 5e-3, 5e-2, 0.5];
        for v in &values {
            h.record(*v);
        }
        assert_eq!(h.total(), 8);
        for (i, &count) in h.buckets.iter().enumerate() {
            assert_eq!(count, 1, "bucket {i} should have count 1");
        }
    }

    #[test]
    fn histogram_zero_goes_to_first_bucket() {
        let mut h = DiffHistogram::new();
        h.record(0.0);
        assert_eq!(h.buckets[0], 1);
    }

    #[test]
    fn histogram_boundary_value() {
        let mut h = DiffHistogram::new();
        // Exactly 1e-7 goes to bucket 1 (≥1e-7, <1e-6)
        h.record(1e-7);
        assert_eq!(h.buckets[1], 1);
    }

    #[test]
    fn histogram_display_works() {
        let h = DiffHistogram::new();
        let s = format!("{h}");
        assert!(s.contains("<1e-7"));
        assert!(s.contains(">=1e-1"));
    }

    // -----------------------------------------------------------------
    // VerificationReport
    // -----------------------------------------------------------------

    #[test]
    fn report_passed_with_zero_failures() {
        let spec = ToleranceSpec::strict();
        let report = VerificationReport {
            total_elements: 100,
            matches: 100,
            failures: 0,
            max_abs_diff: 0.0,
            max_rel_diff: 0.0,
            histogram: DiffHistogram::new(),
        };
        assert!(report.passed(&spec));
    }

    #[test]
    fn report_failed_with_failures() {
        let spec = ToleranceSpec::strict(); // max_failures = 0
        let report = VerificationReport {
            total_elements: 100,
            matches: 99,
            failures: 1,
            max_abs_diff: 0.1,
            max_rel_diff: 0.01,
            histogram: DiffHistogram::new(),
        };
        assert!(!report.passed(&spec));
    }

    #[test]
    fn report_match_rate() {
        let report = VerificationReport {
            total_elements: 200,
            matches: 150,
            failures: 50,
            max_abs_diff: 0.0,
            max_rel_diff: 0.0,
            histogram: DiffHistogram::new(),
        };
        assert!((report.match_rate() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn report_match_rate_empty() {
        let report = VerificationReport {
            total_elements: 0,
            matches: 0,
            failures: 0,
            max_abs_diff: 0.0,
            max_rel_diff: 0.0,
            histogram: DiffHistogram::new(),
        };
        assert_eq!(report.match_rate(), 1.0);
    }

    #[test]
    fn report_display_works() {
        let report = VerificationReport {
            total_elements: 10,
            matches: 9,
            failures: 1,
            max_abs_diff: 0.001,
            max_rel_diff: 0.0001,
            histogram: DiffHistogram::new(),
        };
        let s = format!("{report}");
        assert!(s.contains("9/10"));
        assert!(s.contains("failures"));
    }

    // -----------------------------------------------------------------
    // OperationVerifier
    // -----------------------------------------------------------------

    #[test]
    fn op_verifier_default_profiles_exist() {
        let ov = OperationVerifier::new();
        for op in [
            OpKind::Matmul,
            OpKind::Softmax,
            OpKind::RmsNorm,
            OpKind::Rope,
            OpKind::Attention,
            OpKind::Elementwise,
        ] {
            let _spec = ov.tolerance_for(op);
        }
    }

    #[test]
    fn op_verifier_matmul_more_relaxed_than_elementwise() {
        let ov = OperationVerifier::new();
        let matmul = ov.tolerance_for(OpKind::Matmul);
        let elem = ov.tolerance_for(OpKind::Elementwise);
        assert!(matmul.abs_tol > elem.abs_tol);
    }

    #[test]
    fn op_verifier_identical_passes_all() {
        let ov = OperationVerifier::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        for op in [
            OpKind::Matmul,
            OpKind::Softmax,
            OpKind::RmsNorm,
            OpKind::Rope,
            OpKind::Attention,
            OpKind::Elementwise,
        ] {
            assert!(ov.verify_passes(op, &data, &data), "failed for {op}");
        }
    }

    #[test]
    fn op_verifier_custom_tolerance() {
        let mut ov = OperationVerifier::new();
        let custom = ToleranceSpec { abs_tol: 0.5, rel_tol: 0.5, ulp_tol: 1000, max_failures: 0 };
        ov.set_tolerance(OpKind::Softmax, custom);
        let spec = ov.tolerance_for(OpKind::Softmax);
        assert_eq!(spec.abs_tol, 0.5);
    }

    #[test]
    fn op_verifier_matmul_tolerates_accumulation_error() {
        let ov = OperationVerifier::new();
        // Simulates small matmul accumulation error
        let reference = vec![100.0; 16];
        let candidate: Vec<f32> = reference.iter().map(|&x| x + 5e-4).collect();
        assert!(ov.verify_passes(OpKind::Matmul, &reference, &candidate));
    }

    #[test]
    fn op_verifier_elementwise_strict() {
        let ov = OperationVerifier::new();
        let reference = vec![1.0; 16];
        // Diff of 1e-4 should fail strict elementwise
        let candidate: Vec<f32> = reference.iter().map(|&x| x + 1e-4).collect();
        assert!(!ov.verify_passes(OpKind::Elementwise, &reference, &candidate));
    }

    #[test]
    fn op_verifier_verify_returns_report() {
        let ov = OperationVerifier::new();
        let report = ov.verify(OpKind::Softmax, &[1.0, 2.0], &[1.0, 2.0]);
        assert_eq!(report.total_elements, 2);
        assert_eq!(report.failures, 0);
    }

    // -----------------------------------------------------------------
    // RegressionTracker
    // -----------------------------------------------------------------

    #[test]
    fn regression_tracker_empty() {
        let tracker = RegressionTracker::new(0.01);
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
        assert!(tracker.check_regression().is_none());
    }

    #[test]
    fn regression_tracker_single_snapshot() {
        let mut tracker = RegressionTracker::new(0.01);
        tracker.record(AccuracySnapshot {
            run_id: "run1".into(),
            max_abs_diff: 0.001,
            max_rel_diff: 0.0001,
            failures: 0,
        });
        assert_eq!(tracker.len(), 1);
        assert!(tracker.check_regression().is_none());
    }

    #[test]
    fn regression_tracker_no_regression() {
        let mut tracker = RegressionTracker::new(0.01);
        tracker.record(AccuracySnapshot {
            run_id: "run1".into(),
            max_abs_diff: 0.005,
            max_rel_diff: 0.001,
            failures: 0,
        });
        tracker.record(AccuracySnapshot {
            run_id: "run2".into(),
            max_abs_diff: 0.006, // drift = 0.001 < threshold 0.01
            max_rel_diff: 0.001,
            failures: 0,
        });
        assert!(tracker.check_regression().is_none());
    }

    #[test]
    fn regression_tracker_detects_drift() {
        let mut tracker = RegressionTracker::new(0.01);
        tracker.record(AccuracySnapshot {
            run_id: "run1".into(),
            max_abs_diff: 0.001,
            max_rel_diff: 0.0001,
            failures: 0,
        });
        tracker.record(AccuracySnapshot {
            run_id: "run2".into(),
            max_abs_diff: 0.05, // drift = 0.049 > threshold 0.01
            max_rel_diff: 0.005,
            failures: 0,
        });
        let regression = tracker.check_regression();
        assert!(regression.is_some());
        let (prev, curr) = regression.unwrap();
        assert!((prev - 0.001).abs() < 1e-6);
        assert!((curr - 0.05).abs() < 1e-6);
    }

    #[test]
    fn regression_tracker_record_report() {
        let mut tracker = RegressionTracker::new(0.1);
        let v = NumericalVerifier::new(ToleranceSpec::strict());
        let report = v.verify_slices(&[1.0, 2.0], &[1.0, 2.0]);
        tracker.record_report("test-run", &report);
        assert_eq!(tracker.len(), 1);
        assert_eq!(tracker.history()[0].run_id, "test-run");
    }

    #[test]
    fn regression_tracker_history_preserved() {
        let mut tracker = RegressionTracker::new(0.1);
        for i in 0..5 {
            tracker.record(AccuracySnapshot {
                run_id: format!("run{i}"),
                max_abs_diff: 0.001 * (i as f32 + 1.0),
                max_rel_diff: 0.0001,
                failures: 0,
            });
        }
        assert_eq!(tracker.len(), 5);
        assert_eq!(tracker.history()[0].run_id, "run0");
        assert_eq!(tracker.history()[4].run_id, "run4");
    }

    #[test]
    fn regression_tracker_improvement_not_flagged() {
        let mut tracker = RegressionTracker::new(0.01);
        tracker.record(AccuracySnapshot {
            run_id: "run1".into(),
            max_abs_diff: 0.05,
            max_rel_diff: 0.005,
            failures: 0,
        });
        // Improved accuracy (lower max_abs_diff)
        tracker.record(AccuracySnapshot {
            run_id: "run2".into(),
            max_abs_diff: 0.001,
            max_rel_diff: 0.0001,
            failures: 0,
        });
        assert!(tracker.check_regression().is_none());
    }

    // -----------------------------------------------------------------
    // CPU reference: softmax
    // -----------------------------------------------------------------

    #[test]
    fn cpu_softmax_empty() {
        let mut data = vec![];
        cpu_softmax(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn cpu_softmax_single() {
        let mut data = vec![5.0];
        cpu_softmax(&mut data);
        assert!((data[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cpu_softmax_sums_to_one() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        cpu_softmax(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    }

    #[test]
    fn cpu_softmax_monotonicity() {
        let mut data = vec![1.0, 2.0, 3.0];
        cpu_softmax(&mut data);
        assert!(data[0] < data[1]);
        assert!(data[1] < data[2]);
    }

    #[test]
    fn cpu_softmax_numerical_stability() {
        let mut data = vec![1000.0, 1001.0, 1002.0];
        cpu_softmax(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(data.iter().all(|&v| v.is_finite()));
    }

    // -----------------------------------------------------------------
    // CPU reference: rmsnorm
    // -----------------------------------------------------------------

    #[test]
    fn cpu_rmsnorm_unit_weight() {
        let input = vec![3.0, 4.0];
        let weight = vec![1.0, 1.0];
        let output = cpu_rmsnorm(&input, &weight, 1e-6);
        // RMS = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        let rms = (12.5f32 + 1e-6).sqrt();
        assert!((output[0] - 3.0 / rms).abs() < 1e-5);
        assert!((output[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn cpu_rmsnorm_with_weight() {
        let input = vec![1.0, 1.0];
        let weight = vec![2.0, 3.0];
        let output = cpu_rmsnorm(&input, &weight, 1e-6);
        let rms = (1.0f32 + 1e-6).sqrt();
        assert!((output[0] - 2.0 / rms).abs() < 1e-5);
        assert!((output[1] - 3.0 / rms).abs() < 1e-5);
    }

    // -----------------------------------------------------------------
    // CPU reference: RoPE
    // -----------------------------------------------------------------

    #[test]
    fn cpu_rope_position_zero_is_identity() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut data = original.clone();
        cpu_rope(&mut data, 0, 4, 10000.0);
        // At position 0, angle = 0 → cos=1, sin=0 → identity
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "RoPE at pos 0 should be identity");
        }
    }

    #[test]
    fn cpu_rope_preserves_norm() {
        let mut data = vec![1.0, 0.0, 0.0, 1.0];
        let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        cpu_rope(&mut data, 5, 4, 10000.0);
        let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-5, "RoPE should preserve vector norm");
    }

    // -----------------------------------------------------------------
    // CPU reference: matmul
    // -----------------------------------------------------------------

    #[test]
    fn cpu_matmul_identity() {
        // 2×2 identity × [1,2;3,4]
        let a = vec![1.0, 0.0, 0.0, 1.0]; // I₂
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        cpu_matmul(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cpu_matmul_known_result() {
        // [1,2;3,4] × [5,6;7,8] = [19,22;43,50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        cpu_matmul(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn cpu_matmul_non_square() {
        // 1×3 × 3×1 = 1×1
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut c = vec![0.0; 1];
        cpu_matmul(&a, &b, &mut c, 1, 1, 3);
        assert_eq!(c, vec![32.0]);
    }

    // -----------------------------------------------------------------
    // Large tensor performance
    // -----------------------------------------------------------------

    #[test]
    fn verify_large_tensor() {
        let n = 100_000;
        let reference: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let candidate: Vec<f32> = reference.iter().map(|&x| x + 1e-7).collect();
        let v = NumericalVerifier::new(ToleranceSpec::default_tol());
        let report = v.verify_slices(&reference, &candidate);
        assert_eq!(report.total_elements, n);
        assert_eq!(report.failures, 0);
    }

    // -----------------------------------------------------------------
    // Property tests: symmetry & transitivity
    // -----------------------------------------------------------------

    #[test]
    fn comparison_symmetry() {
        let spec = ToleranceSpec::default_tol();
        let v = NumericalVerifier::new(spec);
        let pairs = [(1.0f32, 1.0 + 1e-6), (0.0, 1e-8), (100.0, 100.001)];
        for (a, b) in pairs {
            let ab = v.element_passes(&v.compare_element(a, b));
            let ba = v.element_passes(&v.compare_element(b, a));
            assert_eq!(ab, ba, "symmetry failed for ({a}, {b})");
        }
    }

    #[test]
    fn tolerance_transitivity() {
        // If strict passes, then default and relaxed must also pass
        let strict_v = NumericalVerifier::new(ToleranceSpec::strict());
        let default_v = NumericalVerifier::new(ToleranceSpec::default_tol());
        let relaxed_v = NumericalVerifier::new(ToleranceSpec::relaxed());

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let small_diff: Vec<f32> = data.iter().map(|&x| x + 1e-7).collect();

        let strict_report = strict_v.verify_slices(&data, &small_diff);
        let default_report = default_v.verify_slices(&data, &small_diff);
        let relaxed_report = relaxed_v.verify_slices(&data, &small_diff);

        if strict_report.failures == 0 {
            assert_eq!(default_report.failures, 0, "default should pass if strict passes");
            assert_eq!(relaxed_report.failures, 0, "relaxed should pass if strict passes");
        }
    }

    #[test]
    fn comparison_nan_symmetry() {
        let v = NumericalVerifier::new(ToleranceSpec::default_tol());
        let r1 = v.compare_element(f32::NAN, 1.0);
        let r2 = v.compare_element(1.0, f32::NAN);
        assert_eq!(r1, r2);
    }
}
