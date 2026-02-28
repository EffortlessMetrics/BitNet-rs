//! GPU debug layer for development diagnostics.
//!
//! Wraps GPU operations with configurable logging, validation,
//! tensor snapshots, and breakpoints for debugging inference issues.

use std::time::{SystemTime, UNIX_EPOCH};

// ── Configuration ─────────────────────────────────────────────────────────

/// Controls which diagnostics the debug layer performs.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct DebugConfig {
    /// Master switch — when `false` the layer is a no-op.
    pub enabled: bool,
    /// Log kernel launch events.
    pub log_kernels: bool,
    /// Log host↔device transfer events.
    pub log_transfers: bool,
    /// Log memory allocation events.
    pub log_allocations: bool,
    /// Run output validation after each operation.
    pub validate_outputs: bool,
    /// Check for NaN values in outputs.
    pub nan_check: bool,
    /// Check for Inf values in outputs.
    pub inf_check: bool,
    /// Optional value-range check `(min, max)`.
    pub range_check: Option<(f64, f64)>,
    /// Take a tensor snapshot every N operations.
    pub snapshot_interval: Option<usize>,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_kernels: true,
            log_transfers: true,
            log_allocations: true,
            validate_outputs: true,
            nan_check: true,
            inf_check: true,
            range_check: None,
            snapshot_interval: None,
        }
    }
}

impl DebugConfig {
    /// A disabled configuration — all checks off.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            log_kernels: false,
            log_transfers: false,
            log_allocations: false,
            validate_outputs: false,
            nan_check: false,
            inf_check: false,
            range_check: None,
            snapshot_interval: None,
        }
    }
}

// ── Events ────────────────────────────────────────────────────────────────

/// Category tag for a debug event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugCategory {
    KernelLaunch,
    MemoryOp,
    Transfer,
    Validation,
    Breakpoint,
    Warning,
    Error,
}

/// Severity level for a debug event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DebugSeverity {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
}

/// A single debug event recorded by the layer.
#[derive(Debug, Clone)]
pub struct DebugEvent {
    pub timestamp_ms: u64,
    pub category: DebugCategory,
    pub message: String,
    pub severity: DebugSeverity,
    pub context: Vec<(String, String)>,
}

impl DebugEvent {
    /// Create a new event stamped with the current wall-clock time.
    #[must_use]
    pub fn new(
        category: DebugCategory,
        severity: DebugSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            timestamp_ms: now_ms(),
            category,
            message: message.into(),
            severity,
            context: Vec::new(),
        }
    }

    /// Attach a key-value context pair.
    #[must_use]
    pub fn with_context(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.context.push((key.into(), val.into()));
        self
    }
}

// ── Tensor snapshot ───────────────────────────────────────────────────────

/// Statistical snapshot of a tensor captured at a point in time.
#[derive(Debug, Clone)]
pub struct TensorSnapshot {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub nan_count: usize,
    pub inf_count: usize,
    pub zero_count: usize,
    pub sample_values: Vec<f64>,
}

impl TensorSnapshot {
    /// Build a snapshot from an `f32` buffer with the given logical shape.
    #[must_use]
    pub fn from_f32(name: impl Into<String>, data: &[f32], shape: Vec<usize>) -> Self {
        let name = name.into();
        if data.is_empty() {
            return Self {
                name,
                shape,
                dtype: "f32".into(),
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                nan_count: 0,
                inf_count: 0,
                zero_count: 0,
                sample_values: Vec::new(),
            };
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0_f64;
        let mut nan_count = 0_usize;
        let mut inf_count = 0_usize;
        let mut zero_count = 0_usize;
        let mut finite_count = 0_usize;

        for &v in data {
            let v64 = f64::from(v);
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            } else {
                if v64 < min {
                    min = v64;
                }
                if v64 > max {
                    max = v64;
                }
                sum += v64;
                finite_count += 1;
            }
            if v == 0.0 {
                zero_count += 1;
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let mean = if finite_count > 0 { sum / finite_count as f64 } else { 0.0 };

        #[allow(clippy::cast_precision_loss)]
        let variance = if finite_count > 0 {
            data.iter()
                .filter(|v| v.is_finite())
                .map(|&v| {
                    let d = f64::from(v) - mean;
                    d * d
                })
                .sum::<f64>()
                / finite_count as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        // Keep first up to 10 values as samples.
        let sample_values: Vec<f64> = data.iter().take(10).map(|&v| f64::from(v)).collect();

        if finite_count == 0 {
            min = 0.0;
            max = 0.0;
        }

        Self {
            name,
            shape,
            dtype: "f32".into(),
            min,
            max,
            mean,
            std_dev,
            nan_count,
            inf_count,
            zero_count,
            sample_values,
        }
    }
}

// ── Breakpoints ───────────────────────────────────────────────────────────

/// Condition that triggers a breakpoint.
#[derive(Debug, Clone)]
pub enum BreakCondition {
    NanDetected,
    InfDetected,
    ValueOutOfRange(f64, f64),
    KernelNamed(String),
    AfterNOps(usize),
    Custom(String),
}

/// A named breakpoint with a trigger condition and hit counter.
#[derive(Debug, Clone)]
pub struct Breakpoint {
    pub name: String,
    pub condition: BreakCondition,
    pub hit_count: u32,
    pub enabled: bool,
}

impl Breakpoint {
    #[must_use]
    pub fn new(name: impl Into<String>, condition: BreakCondition) -> Self {
        Self { name: name.into(), condition, hit_count: 0, enabled: true }
    }
}

// ── Report ────────────────────────────────────────────────────────────────

/// Aggregated summary produced by [`DebugLayer::generate_report`].
#[derive(Debug, Clone)]
pub struct DebugReport {
    pub total_events: usize,
    pub warnings: usize,
    pub errors: usize,
    pub nan_detections: usize,
    pub inf_detections: usize,
    pub breakpoints_hit: usize,
    pub snapshots: Vec<TensorSnapshot>,
}

// ── Debug layer ───────────────────────────────────────────────────────────

/// Core debug layer that wraps GPU operations with diagnostics.
pub struct DebugLayer {
    config: DebugConfig,
    log: Vec<DebugEvent>,
    tensor_snapshots: Vec<TensorSnapshot>,
    breakpoints: Vec<Breakpoint>,
    op_counter: usize,
}

impl DebugLayer {
    /// Create a new debug layer with the given configuration.
    #[must_use]
    pub const fn new(config: DebugConfig) -> Self {
        Self {
            config,
            log: Vec::new(),
            tensor_snapshots: Vec::new(),
            breakpoints: Vec::new(),
            op_counter: 0,
        }
    }

    // ── Event logging ─────────────────────────────────────────────────

    /// Record an event (no-op when the layer is disabled).
    pub fn log_event(&mut self, event: DebugEvent) {
        if !self.config.enabled {
            return;
        }
        self.log.push(event);
    }

    /// Total number of recorded events.
    #[must_use]
    pub const fn event_count(&self) -> usize {
        self.log.len()
    }

    /// Return references to events matching `category`.
    #[must_use]
    pub fn events_by_category(&self, category: DebugCategory) -> Vec<&DebugEvent> {
        self.log.iter().filter(|e| e.category == category).collect()
    }

    /// Return references to events at or above `min_severity`.
    #[must_use]
    pub fn events_by_severity(&self, min_severity: DebugSeverity) -> Vec<&DebugEvent> {
        self.log.iter().filter(|e| e.severity >= min_severity).collect()
    }

    // ── Snapshots ─────────────────────────────────────────────────────

    /// Capture a tensor snapshot (no-op when disabled).
    pub fn take_snapshot(&mut self, name: impl Into<String>, data: &[f32], shape: Vec<usize>) {
        if !self.config.enabled {
            return;
        }
        let snap = TensorSnapshot::from_f32(name, data, shape);
        self.tensor_snapshots.push(snap);
    }

    // ── Validation helpers ────────────────────────────────────────────

    /// Returns `true` when `data` contains at least one NaN.
    #[must_use]
    pub fn check_nan(data: &[f32]) -> bool {
        data.iter().any(|v| v.is_nan())
    }

    /// Returns `true` when `data` contains at least one infinity.
    #[must_use]
    pub fn check_inf(data: &[f32]) -> bool {
        data.iter().any(|v| v.is_infinite())
    }

    /// Return indices of elements outside `[min, max]`.
    #[must_use]
    pub fn check_range(data: &[f32], min: f64, max: f64) -> Vec<usize> {
        data.iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                let v64 = f64::from(v);
                if v.is_nan() || v64 < min || v64 > max { Some(i) } else { None }
            })
            .collect()
    }

    // ── Breakpoints ───────────────────────────────────────────────────

    /// Register a breakpoint.
    pub fn add_breakpoint(&mut self, bp: Breakpoint) {
        self.breakpoints.push(bp);
    }

    /// Evaluate all enabled breakpoints against the current context.
    ///
    /// `context` carries information about the current operation:
    /// - `"nan"` → NaN was detected
    /// - `"inf"` → Inf was detected
    /// - `"kernel:<name>"` → a kernel with this name was launched
    /// - `"value_oob:<val>"` → a value was out of bounds
    ///
    /// The first matching, enabled breakpoint is returned (and its
    /// hit-count incremented).
    pub fn check_breakpoints(&mut self, context: &str) -> Option<usize> {
        self.op_counter += 1;
        for (idx, bp) in self.breakpoints.iter_mut().enumerate() {
            if !bp.enabled {
                continue;
            }
            let hit = match &bp.condition {
                BreakCondition::NanDetected => context == "nan",
                BreakCondition::InfDetected => context == "inf",
                BreakCondition::KernelNamed(name) => {
                    context.strip_prefix("kernel:") == Some(name.as_str())
                }
                BreakCondition::ValueOutOfRange(_, _) => context.starts_with("value_oob:"),
                BreakCondition::AfterNOps(n) => self.op_counter >= *n,
                BreakCondition::Custom(pat) => context.contains(pat.as_str()),
            };
            if hit {
                bp.hit_count += 1;
                return Some(idx);
            }
        }
        None
    }

    // ── Reporting ─────────────────────────────────────────────────────

    /// Produce an aggregated debug report.
    #[must_use]
    pub fn generate_report(&self) -> DebugReport {
        let warnings = self.log.iter().filter(|e| e.severity == DebugSeverity::Warning).count();
        let errors = self.log.iter().filter(|e| e.severity == DebugSeverity::Error).count();
        let nan_detections =
            self.log.iter().filter(|e| e.message.to_lowercase().contains("nan")).count();
        let inf_detections =
            self.log.iter().filter(|e| e.message.to_lowercase().contains("inf")).count();
        let breakpoints_hit: u32 = self.breakpoints.iter().map(|bp| bp.hit_count).sum();

        DebugReport {
            total_events: self.log.len(),
            warnings,
            errors,
            nan_detections,
            inf_detections,
            breakpoints_hit: breakpoints_hit as usize,
            snapshots: self.tensor_snapshots.clone(),
        }
    }

    // ── Housekeeping ──────────────────────────────────────────────────

    /// Clear all recorded events, snapshots, and reset op counter.
    pub fn clear(&mut self) {
        self.log.clear();
        self.tensor_snapshots.clear();
        self.op_counter = 0;
        for bp in &mut self.breakpoints {
            bp.hit_count = 0;
        }
    }

    /// Read-only access to configuration.
    #[must_use]
    pub const fn config(&self) -> &DebugConfig {
        &self.config
    }

    /// Number of snapshots captured so far.
    #[must_use]
    pub const fn snapshot_count(&self) -> usize {
        self.tensor_snapshots.len()
    }

    /// Number of registered breakpoints.
    #[must_use]
    pub const fn breakpoint_count(&self) -> usize {
        self.breakpoints.len()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn now_ms() -> u64 {
    #[allow(clippy::cast_possible_truncation)]
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_layer() -> DebugLayer {
        DebugLayer::new(DebugConfig::default())
    }

    // ── NaN detection ─────────────────────────────────────────────────

    #[test]
    fn nan_detected_in_data() {
        assert!(DebugLayer::check_nan(&[1.0, f32::NAN, 3.0]));
    }

    #[test]
    fn no_nan_in_clean_data() {
        assert!(!DebugLayer::check_nan(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn nan_only_data() {
        assert!(DebugLayer::check_nan(&[f32::NAN, f32::NAN]));
    }

    #[test]
    fn nan_check_empty() {
        assert!(!DebugLayer::check_nan(&[]));
    }

    #[test]
    fn nan_at_start() {
        assert!(DebugLayer::check_nan(&[f32::NAN, 1.0, 2.0]));
    }

    #[test]
    fn nan_at_end() {
        assert!(DebugLayer::check_nan(&[1.0, 2.0, f32::NAN]));
    }

    // ── Inf detection ─────────────────────────────────────────────────

    #[test]
    fn inf_detected_positive() {
        assert!(DebugLayer::check_inf(&[1.0, f32::INFINITY, 3.0]));
    }

    #[test]
    fn inf_detected_negative() {
        assert!(DebugLayer::check_inf(&[f32::NEG_INFINITY, 1.0]));
    }

    #[test]
    fn no_inf_in_clean_data() {
        assert!(!DebugLayer::check_inf(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn inf_check_empty() {
        assert!(!DebugLayer::check_inf(&[]));
    }

    #[test]
    fn inf_both_signs() {
        assert!(DebugLayer::check_inf(&[f32::INFINITY, f32::NEG_INFINITY]));
    }

    // ── Range checks ──────────────────────────────────────────────────

    #[test]
    fn range_all_within() {
        assert!(DebugLayer::check_range(&[0.5, 0.8, 0.1], 0.0, 1.0).is_empty());
    }

    #[test]
    fn range_below_min() {
        let violations = DebugLayer::check_range(&[-0.5, 0.5], 0.0, 1.0);
        assert_eq!(violations, vec![0]);
    }

    #[test]
    fn range_above_max() {
        let violations = DebugLayer::check_range(&[0.5, 1.5], 0.0, 1.0);
        assert_eq!(violations, vec![1]);
    }

    #[test]
    fn range_multiple_violations() {
        let violations = DebugLayer::check_range(&[-1.0, 0.5, 2.0], 0.0, 1.0);
        assert_eq!(violations, vec![0, 2]);
    }

    #[test]
    fn range_nan_counted_as_violation() {
        let violations = DebugLayer::check_range(&[0.5, f32::NAN], 0.0, 1.0);
        assert_eq!(violations, vec![1]);
    }

    #[test]
    fn range_empty_data() {
        assert!(DebugLayer::check_range(&[], 0.0, 1.0).is_empty());
    }

    #[test]
    fn range_exact_boundaries() {
        assert!(DebugLayer::check_range(&[0.0, 1.0], 0.0, 1.0).is_empty());
    }

    // ── TensorSnapshot ────────────────────────────────────────────────

    #[test]
    fn snapshot_basic_stats() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let snap = TensorSnapshot::from_f32("test", &data, vec![5]);
        assert_eq!(snap.name, "test");
        assert_eq!(snap.shape, vec![5]);
        assert_eq!(snap.dtype, "f32");
        assert!((snap.min - 1.0).abs() < 1e-6);
        assert!((snap.max - 5.0).abs() < 1e-6);
    }

    #[test]
    fn snapshot_mean_calculation() {
        let data = [2.0_f32, 4.0, 6.0];
        let snap = TensorSnapshot::from_f32("mean_test", &data, vec![3]);
        assert!((snap.mean - 4.0).abs() < 1e-6);
    }

    #[test]
    fn snapshot_std_dev() {
        let data = [2.0_f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let snap = TensorSnapshot::from_f32("std", &data, vec![8]);
        // population std dev of that data = 2.0
        assert!((snap.std_dev - 2.0).abs() < 0.01);
    }

    #[test]
    fn snapshot_zero_count() {
        let data = [0.0_f32, 1.0, 0.0, 2.0, 0.0];
        let snap = TensorSnapshot::from_f32("zeros", &data, vec![5]);
        assert_eq!(snap.zero_count, 3);
    }

    #[test]
    fn snapshot_nan_count() {
        let data = [1.0_f32, f32::NAN, 3.0, f32::NAN];
        let snap = TensorSnapshot::from_f32("nans", &data, vec![4]);
        assert_eq!(snap.nan_count, 2);
    }

    #[test]
    fn snapshot_inf_count() {
        let data = [f32::INFINITY, 1.0, f32::NEG_INFINITY];
        let snap = TensorSnapshot::from_f32("infs", &data, vec![3]);
        assert_eq!(snap.inf_count, 2);
    }

    #[test]
    fn snapshot_empty_data() {
        let snap = TensorSnapshot::from_f32("empty", &[], vec![0]);
        assert_eq!(snap.nan_count, 0);
        assert_eq!(snap.inf_count, 0);
        assert!((snap.mean).abs() < 1e-10);
        assert!(snap.sample_values.is_empty());
    }

    #[test]
    fn snapshot_sample_values_capped_at_10() {
        #[allow(clippy::cast_precision_loss)]
        let data: Vec<f32> = (0..100).map(|i: i32| i as f32).collect();
        let snap = TensorSnapshot::from_f32("big", &data, vec![100]);
        assert_eq!(snap.sample_values.len(), 10);
        assert!((snap.sample_values[0]).abs() < 1e-6);
        assert!((snap.sample_values[9] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn snapshot_single_element() {
        let snap = TensorSnapshot::from_f32("one", &[42.0], vec![1]);
        assert!((snap.min - 42.0).abs() < 1e-6);
        assert!((snap.max - 42.0).abs() < 1e-6);
        assert!((snap.mean - 42.0).abs() < 1e-6);
        assert!((snap.std_dev).abs() < 1e-6);
    }

    #[test]
    fn snapshot_all_same_values() {
        let data = [5.0_f32; 8];
        let snap = TensorSnapshot::from_f32("const", &data, vec![8]);
        assert!((snap.std_dev).abs() < 1e-6);
        assert!((snap.mean - 5.0).abs() < 1e-6);
    }

    #[test]
    fn snapshot_negative_values() {
        let data = [-3.0_f32, -1.0, -2.0];
        let snap = TensorSnapshot::from_f32("neg", &data, vec![3]);
        assert!((snap.min - (-3.0)).abs() < 1e-6);
        assert!((snap.max - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn snapshot_mixed_nan_inf() {
        let data = [f32::NAN, f32::INFINITY, 1.0, f32::NEG_INFINITY];
        let snap = TensorSnapshot::from_f32("mix", &data, vec![4]);
        assert_eq!(snap.nan_count, 1);
        assert_eq!(snap.inf_count, 2);
        assert!((snap.mean - 1.0).abs() < 1e-6);
    }

    #[test]
    fn snapshot_preserves_shape() {
        let data = [1.0_f32; 12];
        let snap = TensorSnapshot::from_f32("shaped", &data, vec![3, 4]);
        assert_eq!(snap.shape, vec![3, 4]);
    }

    // ── Breakpoint: NaN triggers ──────────────────────────────────────

    #[test]
    fn breakpoint_nan_triggers() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new("bp_nan", BreakCondition::NanDetected));
        assert!(layer.check_breakpoints("nan").is_some());
    }

    #[test]
    fn breakpoint_nan_no_trigger_on_clean() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new("bp_nan", BreakCondition::NanDetected));
        assert!(layer.check_breakpoints("clean").is_none());
    }

    #[test]
    fn breakpoint_nan_increments_hit_count() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new("bp_nan", BreakCondition::NanDetected));
        layer.check_breakpoints("nan");
        layer.check_breakpoints("nan");
        let report = layer.generate_report();
        assert_eq!(report.breakpoints_hit, 2);
    }

    // ── Breakpoint: Inf triggers ──────────────────────────────────────

    #[test]
    fn breakpoint_inf_triggers() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new("bp_inf", BreakCondition::InfDetected));
        assert!(layer.check_breakpoints("inf").is_some());
    }

    #[test]
    fn breakpoint_inf_no_false_positive() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new("bp_inf", BreakCondition::InfDetected));
        assert!(layer.check_breakpoints("nan").is_none());
    }

    // ── Breakpoint: value out of range ────────────────────────────────

    #[test]
    fn breakpoint_value_out_of_range_triggers() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new(
            "bp_range",
            BreakCondition::ValueOutOfRange(-1.0, 1.0),
        ));
        assert!(layer.check_breakpoints("value_oob:5.0").is_some());
    }

    #[test]
    fn breakpoint_value_in_range_no_trigger() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new(
            "bp_range",
            BreakCondition::ValueOutOfRange(-1.0, 1.0),
        ));
        assert!(layer.check_breakpoints("clean").is_none());
    }

    // ── Breakpoint: kernel name match ─────────────────────────────────

    #[test]
    fn breakpoint_kernel_name_match() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new(
            "bp_kern",
            BreakCondition::KernelNamed("matmul".into()),
        ));
        assert!(layer.check_breakpoints("kernel:matmul").is_some());
    }

    #[test]
    fn breakpoint_kernel_name_mismatch() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new(
            "bp_kern",
            BreakCondition::KernelNamed("matmul".into()),
        ));
        assert!(layer.check_breakpoints("kernel:softmax").is_none());
    }

    // ── Breakpoint: after N ops ───────────────────────────────────────

    #[test]
    fn breakpoint_after_n_ops_triggers() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new("bp_ops", BreakCondition::AfterNOps(3)));
        // ops 1 & 2 don't trigger; op 3 does
        assert!(layer.check_breakpoints("x").is_none());
        assert!(layer.check_breakpoints("x").is_none());
        assert!(layer.check_breakpoints("x").is_some());
    }

    #[test]
    fn breakpoint_after_n_ops_stays_triggered() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new("bp_ops", BreakCondition::AfterNOps(1)));
        assert!(layer.check_breakpoints("x").is_some());
        assert!(layer.check_breakpoints("x").is_some());
    }

    // ── Breakpoint: custom ────────────────────────────────────────────

    #[test]
    fn breakpoint_custom_pattern() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new(
            "bp_custom",
            BreakCondition::Custom("divergence".into()),
        ));
        assert!(layer.check_breakpoints("loss divergence detected").is_some());
    }

    #[test]
    fn breakpoint_disabled_does_not_trigger() {
        let mut layer = default_layer();
        let mut bp = Breakpoint::new("bp_off", BreakCondition::NanDetected);
        bp.enabled = false;
        layer.add_breakpoint(bp);
        assert!(layer.check_breakpoints("nan").is_none());
    }

    // ── Multiple breakpoints ──────────────────────────────────────────

    #[test]
    fn multiple_breakpoints_first_match_wins() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new("first", BreakCondition::NanDetected));
        layer.add_breakpoint(Breakpoint::new("second", BreakCondition::Custom("nan".into())));
        let idx = layer.check_breakpoints("nan").unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn multiple_breakpoints_independent_hit_counts() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new("nan_bp", BreakCondition::NanDetected));
        layer.add_breakpoint(Breakpoint::new("inf_bp", BreakCondition::InfDetected));
        layer.check_breakpoints("nan");
        layer.check_breakpoints("inf");
        layer.check_breakpoints("nan");
        let report = layer.generate_report();
        assert_eq!(report.breakpoints_hit, 3);
    }

    // ── Debug report ──────────────────────────────────────────────────

    #[test]
    fn report_empty_layer() {
        let layer = default_layer();
        let report = layer.generate_report();
        assert_eq!(report.total_events, 0);
        assert_eq!(report.warnings, 0);
        assert_eq!(report.errors, 0);
    }

    #[test]
    fn report_counts_warnings() {
        let mut layer = default_layer();
        layer.log_event(DebugEvent::new(DebugCategory::Warning, DebugSeverity::Warning, "warn"));
        layer.log_event(DebugEvent::new(DebugCategory::Error, DebugSeverity::Error, "err"));
        let report = layer.generate_report();
        assert_eq!(report.warnings, 1);
        assert_eq!(report.errors, 1);
        assert_eq!(report.total_events, 2);
    }

    #[test]
    fn report_nan_detections() {
        let mut layer = default_layer();
        layer.log_event(DebugEvent::new(
            DebugCategory::Validation,
            DebugSeverity::Error,
            "NaN detected in output",
        ));
        let report = layer.generate_report();
        assert_eq!(report.nan_detections, 1);
    }

    #[test]
    fn report_inf_detections() {
        let mut layer = default_layer();
        layer.log_event(DebugEvent::new(
            DebugCategory::Validation,
            DebugSeverity::Error,
            "Inf value found",
        ));
        let report = layer.generate_report();
        assert_eq!(report.inf_detections, 1);
    }

    #[test]
    fn report_includes_snapshots() {
        let mut layer = default_layer();
        layer.take_snapshot("t1", &[1.0, 2.0], vec![2]);
        let report = layer.generate_report();
        assert_eq!(report.snapshots.len(), 1);
        assert_eq!(report.snapshots[0].name, "t1");
    }

    #[test]
    fn report_breakpoints_hit_count() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new("bp", BreakCondition::NanDetected));
        layer.check_breakpoints("nan");
        layer.check_breakpoints("nan");
        let report = layer.generate_report();
        assert_eq!(report.breakpoints_hit, 2);
    }

    // ── Disabled layer ────────────────────────────────────────────────

    #[test]
    fn disabled_layer_ignores_events() {
        let mut layer = DebugLayer::new(DebugConfig::disabled());
        layer.log_event(DebugEvent::new(
            DebugCategory::KernelLaunch,
            DebugSeverity::Info,
            "should be ignored",
        ));
        assert_eq!(layer.event_count(), 0);
    }

    #[test]
    fn disabled_layer_ignores_snapshots() {
        let mut layer = DebugLayer::new(DebugConfig::disabled());
        layer.take_snapshot("ignored", &[1.0], vec![1]);
        assert_eq!(layer.snapshot_count(), 0);
    }

    #[test]
    fn disabled_config_all_flags_off() {
        let cfg = DebugConfig::disabled();
        assert!(!cfg.enabled);
        assert!(!cfg.log_kernels);
        assert!(!cfg.nan_check);
    }

    // ── Event filtering ───────────────────────────────────────────────

    #[test]
    fn filter_by_category() {
        let mut layer = default_layer();
        layer.log_event(DebugEvent::new(
            DebugCategory::KernelLaunch,
            DebugSeverity::Info,
            "launch",
        ));
        layer.log_event(DebugEvent::new(DebugCategory::Transfer, DebugSeverity::Debug, "copy"));
        layer.log_event(DebugEvent::new(
            DebugCategory::KernelLaunch,
            DebugSeverity::Info,
            "launch2",
        ));
        let kernel_events = layer.events_by_category(DebugCategory::KernelLaunch);
        assert_eq!(kernel_events.len(), 2);
    }

    #[test]
    fn filter_by_category_empty() {
        let layer = default_layer();
        assert!(layer.events_by_category(DebugCategory::Error).is_empty());
    }

    #[test]
    fn filter_by_severity() {
        let mut layer = default_layer();
        layer.log_event(DebugEvent::new(DebugCategory::Warning, DebugSeverity::Trace, "trace"));
        layer.log_event(DebugEvent::new(DebugCategory::Warning, DebugSeverity::Warning, "warn"));
        layer.log_event(DebugEvent::new(DebugCategory::Error, DebugSeverity::Error, "err"));
        let warnings_up = layer.events_by_severity(DebugSeverity::Warning);
        assert_eq!(warnings_up.len(), 2);
    }

    // ── Clear / housekeeping ──────────────────────────────────────────

    #[test]
    fn clear_resets_events() {
        let mut layer = default_layer();
        layer.log_event(DebugEvent::new(DebugCategory::KernelLaunch, DebugSeverity::Info, "ev"));
        layer.take_snapshot("s", &[1.0], vec![1]);
        layer.add_breakpoint(Breakpoint::new("bp", BreakCondition::NanDetected));
        layer.check_breakpoints("nan");
        layer.clear();
        assert_eq!(layer.event_count(), 0);
        assert_eq!(layer.snapshot_count(), 0);
    }

    #[test]
    fn clear_resets_breakpoint_hit_counts() {
        let mut layer = default_layer();
        layer.add_breakpoint(Breakpoint::new("bp", BreakCondition::NanDetected));
        layer.check_breakpoints("nan");
        layer.clear();
        let report = layer.generate_report();
        assert_eq!(report.breakpoints_hit, 0);
    }

    // ── Event context ─────────────────────────────────────────────────

    #[test]
    fn event_with_context() {
        let event = DebugEvent::new(DebugCategory::KernelLaunch, DebugSeverity::Info, "launch")
            .with_context("kernel", "matmul")
            .with_context("grid", "128x128");
        assert_eq!(event.context.len(), 2);
        assert_eq!(event.context[0].0, "kernel");
    }

    #[test]
    fn event_timestamp_is_nonzero() {
        let event = DebugEvent::new(DebugCategory::KernelLaunch, DebugSeverity::Info, "ts");
        assert!(event.timestamp_ms > 0);
    }

    // ── Large tensor performance ──────────────────────────────────────

    #[test]
    fn large_tensor_snapshot() {
        #[allow(clippy::cast_precision_loss)]
        let data: Vec<f32> = (0..100_000).map(|i: i32| i as f32).collect();
        let snap = TensorSnapshot::from_f32("large", &data, vec![100_000]);
        assert_eq!(snap.nan_count, 0);
        assert_eq!(snap.inf_count, 0);
        assert!(snap.max > snap.min);
        assert_eq!(snap.sample_values.len(), 10);
    }

    // ── DebugConfig default ───────────────────────────────────────────

    #[test]
    fn default_config_enabled() {
        let cfg = DebugConfig::default();
        assert!(cfg.enabled);
        assert!(cfg.nan_check);
        assert!(cfg.inf_check);
        assert!(cfg.range_check.is_none());
        assert!(cfg.snapshot_interval.is_none());
    }

    // ── Edge-case: all-NaN tensor ─────────────────────────────────────

    #[test]
    fn snapshot_all_nan() {
        let data = [f32::NAN; 5];
        let snap = TensorSnapshot::from_f32("all_nan", &data, vec![5]);
        assert_eq!(snap.nan_count, 5);
        assert!((snap.mean).abs() < 1e-10);
        assert!((snap.std_dev).abs() < 1e-10);
    }

    // ── Breakpoint count ──────────────────────────────────────────────

    #[test]
    fn breakpoint_count_tracks_additions() {
        let mut layer = default_layer();
        assert_eq!(layer.breakpoint_count(), 0);
        layer.add_breakpoint(Breakpoint::new("a", BreakCondition::NanDetected));
        layer.add_breakpoint(Breakpoint::new("b", BreakCondition::InfDetected));
        assert_eq!(layer.breakpoint_count(), 2);
    }
}
