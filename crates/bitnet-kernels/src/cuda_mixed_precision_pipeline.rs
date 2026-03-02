//! CUDA mixed-precision inference pipeline with automatic precision selection.
//!
//! Provides per-layer precision assignment (FP32, FP16, BF16, INT8), dynamic
//! loss scaling for gradient stability, cast insertion at precision boundaries,
//! and memory-savings estimation. Sensitive layers (attention, normalization)
//! are kept at FP32 to preserve numerical stability while compute-heavy layers
//! (linear projections, feed-forward) run at reduced precision for throughput.
//!
//! All public items are feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use std::collections::HashMap;
use std::fmt;

// ── Precision types ─────────────────────────────────────────────────────────

/// Numeric precision for a single tensor or layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaPrecision {
    /// 32-bit IEEE 754 single precision.
    FP32,
    /// 16-bit IEEE 754 half precision.
    FP16,
    /// 16-bit Brain Float (truncated mantissa of FP32).
    BF16,
    /// 8-bit signed integer with per-tensor scale.
    INT8,
}

impl CudaPrecision {
    /// Bytes per element.
    pub fn size_bytes(self) -> usize {
        match self {
            Self::FP32 => 4,
            Self::FP16 | Self::BF16 => 2,
            Self::INT8 => 1,
        }
    }

    /// Representable range as `(min, max)`.
    pub fn range(self) -> (f64, f64) {
        match self {
            Self::FP32 => (f32::MIN as f64, f32::MAX as f64),
            Self::FP16 => (-65504.0, 65504.0),
            Self::BF16 => (-3.389e38, 3.389e38),
            Self::INT8 => (-128.0, 127.0),
        }
    }
}

impl fmt::Display for CudaPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FP32 => write!(f, "FP32"),
            Self::FP16 => write!(f, "FP16"),
            Self::BF16 => write!(f, "BF16"),
            Self::INT8 => write!(f, "INT8"),
        }
    }
}

// ── Precision policy ────────────────────────────────────────────────────────

/// High-level precision policy that the scheduler expands into per-layer assignments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrecisionPolicy {
    /// Full FP32 everywhere — maximum accuracy, maximum memory.
    Full,
    /// FP16 for compute, FP32 accumulation for sensitive layers.
    Half,
    /// BF16 for compute, FP32 accumulation for sensitive layers.
    BFloat,
    /// INT8 for weights with FP32 accumulation and sensitive layers kept at FP32.
    Int8,
    /// Automatic: attention/norm → FP32, linear/ffn → FP16, embedding → FP16.
    #[default]
    Auto,
}

impl fmt::Display for PrecisionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "Full(FP32)"),
            Self::Half => write!(f, "Half(FP16)"),
            Self::BFloat => write!(f, "BFloat(BF16)"),
            Self::Int8 => write!(f, "Int8"),
            Self::Auto => write!(f, "Auto"),
        }
    }
}

// ── Layer classification ────────────────────────────────────────────────────

/// Classification of a transformer layer for precision assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    Attention,
    Normalization,
    Linear,
    FeedForward,
    Embedding,
    Residual,
    Softmax,
    Output,
}

impl LayerKind {
    /// Whether this layer is numerically sensitive and should prefer FP32.
    pub fn is_sensitive(self) -> bool {
        matches!(self, Self::Attention | Self::Normalization | Self::Softmax | Self::Residual)
    }
}

impl fmt::Display for LayerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Attention => "attention",
            Self::Normalization => "normalization",
            Self::Linear => "linear",
            Self::FeedForward => "feed_forward",
            Self::Embedding => "embedding",
            Self::Residual => "residual",
            Self::Softmax => "softmax",
            Self::Output => "output",
        };
        f.write_str(s)
    }
}

// ── Layer descriptor ────────────────────────────────────────────────────────

/// Metadata for a single layer in the pipeline.
#[derive(Debug, Clone)]
pub struct LayerDescriptor {
    /// Unique name (e.g. `"layer_0.attn"`).
    pub name: String,
    /// Classification.
    pub kind: LayerKind,
    /// Number of parameters in the layer.
    pub param_count: usize,
    /// Index within the model.
    pub index: usize,
}

// ── Cast operation ──────────────────────────────────────────────────────────

/// Describes a precision cast inserted at a boundary between two layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CastOp {
    pub from_layer: String,
    pub to_layer: String,
    pub from_precision: CudaPrecision,
    pub to_precision: CudaPrecision,
}

impl CastOp {
    /// Estimated cost weight (higher = more expensive).
    pub fn cost(&self) -> f64 {
        if self.from_precision == self.to_precision {
            return 0.0;
        }
        // INT8 casts are more expensive due to quantization overhead
        if self.from_precision == CudaPrecision::INT8 || self.to_precision == CudaPrecision::INT8 {
            2.0
        } else {
            1.0
        }
    }
}

// ── Loss scaler ─────────────────────────────────────────────────────────────

/// Dynamic loss scaler that adjusts the scale factor based on overflow detection.
///
/// Keeps a running scale factor that is halved on overflow and slowly grown
/// when consecutive steps succeed, stabilising mixed-precision training-like
/// forward passes.
#[derive(Debug, Clone)]
pub struct LossScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: u32,
    consecutive_ok: u32,
    overflow_count: u64,
    step_count: u64,
}

impl LossScaler {
    /// Create a new loss scaler with the given initial scale.
    pub fn new(initial_scale: f64) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            consecutive_ok: 0,
            overflow_count: 0,
            step_count: 0,
        }
    }

    /// Current scale factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Total overflow events observed.
    pub fn overflow_count(&self) -> u64 {
        self.overflow_count
    }

    /// Total steps processed.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Scale a value up for reduced-precision computation.
    pub fn scale_up(&self, value: f64) -> f64 {
        value * self.scale
    }

    /// Unscale (divide) a value back after reduced-precision computation.
    pub fn unscale(&self, value: f64) -> f64 {
        value / self.scale
    }

    /// Check if a value overflows the target precision range.
    pub fn check_overflow(&self, value: f64, precision: CudaPrecision) -> bool {
        let (min, max) = precision.range();
        value.is_nan() || value.is_infinite() || value < min || value > max
    }

    /// Report a successful step (no overflow).
    pub fn report_ok(&mut self) {
        self.step_count += 1;
        self.consecutive_ok += 1;
        if self.consecutive_ok >= self.growth_interval {
            self.scale *= self.growth_factor;
            self.consecutive_ok = 0;
        }
    }

    /// Report an overflow event — halves the scale.
    pub fn report_overflow(&mut self) {
        self.step_count += 1;
        self.overflow_count += 1;
        self.consecutive_ok = 0;
        self.scale *= self.backoff_factor;
        if self.scale < 1.0 {
            self.scale = 1.0;
        }
    }
}

impl Default for LossScaler {
    fn default() -> Self {
        Self::new(65536.0)
    }
}

// ── Precision scheduler ─────────────────────────────────────────────────────

/// Assigns precision to each layer based on the active [`PrecisionPolicy`] and
/// detects cast boundaries.
#[derive(Debug)]
pub struct PrecisionScheduler {
    policy: PrecisionPolicy,
    overrides: HashMap<String, CudaPrecision>,
}

impl PrecisionScheduler {
    pub fn new(policy: PrecisionPolicy) -> Self {
        Self { policy, overrides: HashMap::new() }
    }

    /// Override precision for a specific layer by name.
    pub fn set_override(&mut self, layer_name: impl Into<String>, precision: CudaPrecision) {
        self.overrides.insert(layer_name.into(), precision);
    }

    /// Resolve precision for a layer, consulting overrides then policy.
    pub fn resolve(&self, layer: &LayerDescriptor) -> CudaPrecision {
        if let Some(&p) = self.overrides.get(&layer.name) {
            return p;
        }
        match self.policy {
            PrecisionPolicy::Full => CudaPrecision::FP32,
            PrecisionPolicy::Half => {
                if layer.kind.is_sensitive() {
                    CudaPrecision::FP32
                } else {
                    CudaPrecision::FP16
                }
            }
            PrecisionPolicy::BFloat => {
                if layer.kind.is_sensitive() {
                    CudaPrecision::FP32
                } else {
                    CudaPrecision::BF16
                }
            }
            PrecisionPolicy::Int8 => {
                if layer.kind.is_sensitive() {
                    CudaPrecision::FP32
                } else {
                    CudaPrecision::INT8
                }
            }
            PrecisionPolicy::Auto => Self::auto_precision(layer),
        }
    }

    /// Automatic precision selection per layer kind.
    fn auto_precision(layer: &LayerDescriptor) -> CudaPrecision {
        match layer.kind {
            LayerKind::Attention | LayerKind::Normalization | LayerKind::Softmax => {
                CudaPrecision::FP32
            }
            LayerKind::Residual => CudaPrecision::FP32,
            LayerKind::Linear | LayerKind::FeedForward => CudaPrecision::FP16,
            LayerKind::Embedding => CudaPrecision::FP16,
            LayerKind::Output => CudaPrecision::FP32,
        }
    }

    /// Assign precision to every layer and return the assignments.
    pub fn assign(&self, layers: &[LayerDescriptor]) -> Vec<(String, CudaPrecision)> {
        layers.iter().map(|l| (l.name.clone(), self.resolve(l))).collect()
    }

    /// Detect cast operations needed between consecutive layers.
    pub fn detect_casts(&self, layers: &[LayerDescriptor]) -> Vec<CastOp> {
        if layers.len() < 2 {
            return Vec::new();
        }
        let mut casts = Vec::new();
        let mut prev_prec = self.resolve(&layers[0]);
        let mut prev_name = layers[0].name.clone();
        for layer in &layers[1..] {
            let cur_prec = self.resolve(layer);
            if cur_prec != prev_prec {
                casts.push(CastOp {
                    from_layer: prev_name.clone(),
                    to_layer: layer.name.clone(),
                    from_precision: prev_prec,
                    to_precision: cur_prec,
                });
            }
            prev_prec = cur_prec;
            prev_name = layer.name.clone();
        }
        casts
    }

    /// Estimate total memory in bytes for the given layers under the active policy.
    pub fn estimate_memory(&self, layers: &[LayerDescriptor]) -> usize {
        layers.iter().map(|l| l.param_count * self.resolve(l).size_bytes()).sum()
    }

    /// Estimate memory savings vs all-FP32 baseline (returns ratio 0.0–1.0).
    pub fn memory_savings_ratio(&self, layers: &[LayerDescriptor]) -> f64 {
        let fp32_total: usize = layers.iter().map(|l| l.param_count * 4).sum();
        if fp32_total == 0 {
            return 0.0;
        }
        let mixed = self.estimate_memory(layers);
        1.0 - (mixed as f64 / fp32_total as f64)
    }
}

// ── Stability monitor ───────────────────────────────────────────────────────

/// Tracks per-layer numerical stability and suggests precision upgrades.
#[derive(Debug)]
pub struct StabilityMonitor {
    history: HashMap<String, Vec<f64>>,
    nan_threshold: u32,
    variance_threshold: f64,
}

impl StabilityMonitor {
    pub fn new() -> Self {
        Self { history: HashMap::new(), nan_threshold: 3, variance_threshold: 1e6 }
    }

    /// Record an activation magnitude for a layer.
    pub fn record(&mut self, layer_name: &str, magnitude: f64) {
        self.history.entry(layer_name.to_string()).or_default().push(magnitude);
    }

    /// Check if a layer is unstable (too many NaN/Inf or high variance).
    pub fn is_unstable(&self, layer_name: &str) -> bool {
        let Some(values) = self.history.get(layer_name) else {
            return false;
        };
        let nan_count = values.iter().filter(|v| v.is_nan() || v.is_infinite()).count();
        if nan_count as u32 >= self.nan_threshold {
            return true;
        }
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.len() < 2 {
            return false;
        }
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        let variance =
            finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (finite.len() - 1) as f64;
        variance > self.variance_threshold
    }

    /// Suggest precision upgrades for unstable layers.
    pub fn suggest_upgrades(&self) -> Vec<(String, CudaPrecision)> {
        self.history
            .keys()
            .filter(|name| self.is_unstable(name))
            .map(|name| (name.clone(), CudaPrecision::FP32))
            .collect()
    }
}

impl Default for StabilityMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ── Mixed-precision pipeline ────────────────────────────────────────────────

/// Top-level pipeline that wires together scheduler, loss scaler, and stability
/// monitor for end-to-end mixed-precision inference.
#[derive(Debug)]
pub struct MixedPrecisionPipeline {
    scheduler: PrecisionScheduler,
    scaler: LossScaler,
    monitor: StabilityMonitor,
    layers: Vec<LayerDescriptor>,
    dynamic_adjustment: bool,
}

impl MixedPrecisionPipeline {
    /// Create a pipeline from a policy and layer list.
    pub fn new(policy: PrecisionPolicy, layers: Vec<LayerDescriptor>) -> Self {
        Self {
            scheduler: PrecisionScheduler::new(policy),
            scaler: LossScaler::default(),
            monitor: StabilityMonitor::new(),
            layers,
            dynamic_adjustment: true,
        }
    }

    /// Enable or disable dynamic precision adjustment.
    pub fn set_dynamic_adjustment(&mut self, enabled: bool) {
        self.dynamic_adjustment = enabled;
    }

    /// Access the loss scaler.
    pub fn scaler(&self) -> &LossScaler {
        &self.scaler
    }

    /// Access the loss scaler mutably.
    pub fn scaler_mut(&mut self) -> &mut LossScaler {
        &mut self.scaler
    }

    /// Current precision assignment for every layer.
    pub fn assignments(&self) -> Vec<(String, CudaPrecision)> {
        self.scheduler.assign(&self.layers)
    }

    /// Cast operations required between layers.
    pub fn casts(&self) -> Vec<CastOp> {
        self.scheduler.detect_casts(&self.layers)
    }

    /// Estimated memory usage in bytes.
    pub fn estimated_memory(&self) -> usize {
        self.scheduler.estimate_memory(&self.layers)
    }

    /// Memory savings vs FP32 baseline (0.0–1.0).
    pub fn memory_savings(&self) -> f64 {
        self.scheduler.memory_savings_ratio(&self.layers)
    }

    /// Precision for a named layer.
    pub fn precision_for(&self, layer_name: &str) -> Option<CudaPrecision> {
        self.layers.iter().find(|l| l.name == layer_name).map(|l| self.scheduler.resolve(l))
    }

    /// Record an activation magnitude and apply dynamic adjustment if enabled.
    pub fn record_activation(&mut self, layer_name: &str, magnitude: f64) {
        self.monitor.record(layer_name, magnitude);
        if self.dynamic_adjustment {
            if magnitude.is_nan() || magnitude.is_infinite() {
                self.scaler.report_overflow();
            } else {
                self.scaler.report_ok();
            }
            // Apply stability-driven upgrades
            for (name, prec) in self.monitor.suggest_upgrades() {
                self.scheduler.set_override(&name, prec);
            }
        }
    }

    /// Override the precision of a specific layer.
    pub fn set_layer_override(&mut self, layer_name: &str, precision: CudaPrecision) {
        self.scheduler.set_override(layer_name.to_string(), precision);
    }

    /// Number of layers in the pipeline.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Number of cast boundaries in the current assignment.
    pub fn cast_count(&self) -> usize {
        self.casts().len()
    }

    /// Total estimated cast cost.
    pub fn total_cast_cost(&self) -> f64 {
        self.casts().iter().map(|c| c.cost()).sum()
    }

    /// Sensitive layer names.
    pub fn sensitive_layers(&self) -> Vec<String> {
        self.layers.iter().filter(|l| l.kind.is_sensitive()).map(|l| l.name.clone()).collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────────────────────

    fn sample_layers() -> Vec<LayerDescriptor> {
        vec![
            LayerDescriptor {
                name: "embed".into(),
                kind: LayerKind::Embedding,
                param_count: 1_000_000,
                index: 0,
            },
            LayerDescriptor {
                name: "layer_0.attn".into(),
                kind: LayerKind::Attention,
                param_count: 2_000_000,
                index: 1,
            },
            LayerDescriptor {
                name: "layer_0.norm".into(),
                kind: LayerKind::Normalization,
                param_count: 4_096,
                index: 2,
            },
            LayerDescriptor {
                name: "layer_0.ffn".into(),
                kind: LayerKind::FeedForward,
                param_count: 4_000_000,
                index: 3,
            },
            LayerDescriptor {
                name: "layer_0.residual".into(),
                kind: LayerKind::Residual,
                param_count: 4_096,
                index: 4,
            },
            LayerDescriptor {
                name: "layer_0.softmax".into(),
                kind: LayerKind::Softmax,
                param_count: 0,
                index: 5,
            },
            LayerDescriptor {
                name: "output".into(),
                kind: LayerKind::Output,
                param_count: 1_000_000,
                index: 6,
            },
        ]
    }

    // ── CudaPrecision ───────────────────────────────────────────────────────

    #[test]
    fn test_precision_size_bytes() {
        assert_eq!(CudaPrecision::FP32.size_bytes(), 4);
        assert_eq!(CudaPrecision::FP16.size_bytes(), 2);
        assert_eq!(CudaPrecision::BF16.size_bytes(), 2);
        assert_eq!(CudaPrecision::INT8.size_bytes(), 1);
    }

    #[test]
    fn test_precision_display() {
        assert_eq!(CudaPrecision::FP32.to_string(), "FP32");
        assert_eq!(CudaPrecision::FP16.to_string(), "FP16");
        assert_eq!(CudaPrecision::BF16.to_string(), "BF16");
        assert_eq!(CudaPrecision::INT8.to_string(), "INT8");
    }

    #[test]
    fn test_precision_range_fp16() {
        let (min, max) = CudaPrecision::FP16.range();
        assert!(min < 0.0);
        assert!(max > 0.0);
        assert!((max - 65504.0).abs() < f64::EPSILON);
    }

    // ── PrecisionPolicy ─────────────────────────────────────────────────────

    #[test]
    fn test_policy_default_is_auto() {
        assert_eq!(PrecisionPolicy::default(), PrecisionPolicy::Auto);
    }

    #[test]
    fn test_policy_display() {
        assert_eq!(PrecisionPolicy::Full.to_string(), "Full(FP32)");
        assert_eq!(PrecisionPolicy::Half.to_string(), "Half(FP16)");
        assert_eq!(PrecisionPolicy::Auto.to_string(), "Auto");
    }

    // ── LayerKind ───────────────────────────────────────────────────────────

    #[test]
    fn test_sensitive_layers() {
        assert!(LayerKind::Attention.is_sensitive());
        assert!(LayerKind::Normalization.is_sensitive());
        assert!(LayerKind::Softmax.is_sensitive());
        assert!(LayerKind::Residual.is_sensitive());
        assert!(!LayerKind::Linear.is_sensitive());
        assert!(!LayerKind::FeedForward.is_sensitive());
        assert!(!LayerKind::Embedding.is_sensitive());
        assert!(!LayerKind::Output.is_sensitive());
    }

    // ── PrecisionScheduler ──────────────────────────────────────────────────

    #[test]
    fn test_scheduler_full_policy_all_fp32() {
        let sched = PrecisionScheduler::new(PrecisionPolicy::Full);
        for layer in &sample_layers() {
            assert_eq!(sched.resolve(layer), CudaPrecision::FP32);
        }
    }

    #[test]
    fn test_scheduler_half_policy_sensitive_fp32() {
        let sched = PrecisionScheduler::new(PrecisionPolicy::Half);
        let layers = sample_layers();
        let attn = layers.iter().find(|l| l.kind == LayerKind::Attention).unwrap();
        let ffn = layers.iter().find(|l| l.kind == LayerKind::FeedForward).unwrap();
        assert_eq!(sched.resolve(attn), CudaPrecision::FP32);
        assert_eq!(sched.resolve(ffn), CudaPrecision::FP16);
    }

    #[test]
    fn test_scheduler_bfloat_policy() {
        let sched = PrecisionScheduler::new(PrecisionPolicy::BFloat);
        let layers = sample_layers();
        let norm = layers.iter().find(|l| l.kind == LayerKind::Normalization).unwrap();
        let ffn = layers.iter().find(|l| l.kind == LayerKind::FeedForward).unwrap();
        assert_eq!(sched.resolve(norm), CudaPrecision::FP32);
        assert_eq!(sched.resolve(ffn), CudaPrecision::BF16);
    }

    #[test]
    fn test_scheduler_int8_policy() {
        let sched = PrecisionScheduler::new(PrecisionPolicy::Int8);
        let layers = sample_layers();
        let ffn = layers.iter().find(|l| l.kind == LayerKind::FeedForward).unwrap();
        let attn = layers.iter().find(|l| l.kind == LayerKind::Attention).unwrap();
        assert_eq!(sched.resolve(ffn), CudaPrecision::INT8);
        assert_eq!(sched.resolve(attn), CudaPrecision::FP32);
    }

    #[test]
    fn test_scheduler_auto_policy_precision_map() {
        let sched = PrecisionScheduler::new(PrecisionPolicy::Auto);
        let layers = sample_layers();
        let embed = layers.iter().find(|l| l.kind == LayerKind::Embedding).unwrap();
        let attn = layers.iter().find(|l| l.kind == LayerKind::Attention).unwrap();
        let ffn = layers.iter().find(|l| l.kind == LayerKind::FeedForward).unwrap();
        let output = layers.iter().find(|l| l.kind == LayerKind::Output).unwrap();
        assert_eq!(sched.resolve(embed), CudaPrecision::FP16);
        assert_eq!(sched.resolve(attn), CudaPrecision::FP32);
        assert_eq!(sched.resolve(ffn), CudaPrecision::FP16);
        assert_eq!(sched.resolve(output), CudaPrecision::FP32);
    }

    #[test]
    fn test_scheduler_override_takes_precedence() {
        let mut sched = PrecisionScheduler::new(PrecisionPolicy::Full);
        sched.set_override("layer_0.ffn", CudaPrecision::INT8);
        let layers = sample_layers();
        let ffn = layers.iter().find(|l| l.name == "layer_0.ffn").unwrap();
        assert_eq!(sched.resolve(ffn), CudaPrecision::INT8);
    }

    // ── Cast detection ──────────────────────────────────────────────────────

    #[test]
    fn test_detect_casts_full_policy_no_casts() {
        let sched = PrecisionScheduler::new(PrecisionPolicy::Full);
        let casts = sched.detect_casts(&sample_layers());
        assert!(casts.is_empty(), "Full policy should produce no casts");
    }

    #[test]
    fn test_detect_casts_auto_has_boundaries() {
        let sched = PrecisionScheduler::new(PrecisionPolicy::Auto);
        let casts = sched.detect_casts(&sample_layers());
        assert!(!casts.is_empty(), "Auto policy should insert casts at boundaries");
    }

    #[test]
    fn test_detect_casts_single_layer() {
        let sched = PrecisionScheduler::new(PrecisionPolicy::Auto);
        let layers = vec![LayerDescriptor {
            name: "only".into(),
            kind: LayerKind::Linear,
            param_count: 100,
            index: 0,
        }];
        assert!(sched.detect_casts(&layers).is_empty());
    }

    #[test]
    fn test_cast_op_cost_same_precision() {
        let cast = CastOp {
            from_layer: "a".into(),
            to_layer: "b".into(),
            from_precision: CudaPrecision::FP32,
            to_precision: CudaPrecision::FP32,
        };
        assert!((cast.cost() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cast_op_cost_int8_higher() {
        let float_cast = CastOp {
            from_layer: "a".into(),
            to_layer: "b".into(),
            from_precision: CudaPrecision::FP32,
            to_precision: CudaPrecision::FP16,
        };
        let int8_cast = CastOp {
            from_layer: "a".into(),
            to_layer: "b".into(),
            from_precision: CudaPrecision::FP32,
            to_precision: CudaPrecision::INT8,
        };
        assert!(int8_cast.cost() > float_cast.cost());
    }

    // ── Memory savings ──────────────────────────────────────────────────────

    #[test]
    fn test_memory_savings_full_is_zero() {
        let sched = PrecisionScheduler::new(PrecisionPolicy::Full);
        let ratio = sched.memory_savings_ratio(&sample_layers());
        assert!(ratio.abs() < f64::EPSILON, "Full FP32 should have 0% savings");
    }

    #[test]
    fn test_memory_savings_auto_positive() {
        let sched = PrecisionScheduler::new(PrecisionPolicy::Auto);
        let ratio = sched.memory_savings_ratio(&sample_layers());
        assert!(ratio > 0.0, "Auto policy should save memory vs full FP32, got {ratio}");
    }

    #[test]
    fn test_memory_savings_empty_layers() {
        let sched = PrecisionScheduler::new(PrecisionPolicy::Auto);
        let ratio = sched.memory_savings_ratio(&[]);
        assert!((ratio - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_estimate_matches_manual() {
        let layers = vec![
            LayerDescriptor {
                name: "a".into(),
                kind: LayerKind::Linear,
                param_count: 1000,
                index: 0,
            },
            LayerDescriptor {
                name: "b".into(),
                kind: LayerKind::Attention,
                param_count: 1000,
                index: 1,
            },
        ];
        let sched = PrecisionScheduler::new(PrecisionPolicy::Half);
        // Linear → FP16 (2 bytes), Attention → FP32 (4 bytes)
        let expected = 1000 * 2 + 1000 * 4;
        assert_eq!(sched.estimate_memory(&layers), expected);
    }

    // ── LossScaler ──────────────────────────────────────────────────────────

    #[test]
    fn test_loss_scaler_default() {
        let s = LossScaler::default();
        assert!((s.scale() - 65536.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_loss_scaler_scale_up_down() {
        let s = LossScaler::new(100.0);
        assert!((s.scale_up(2.0) - 200.0).abs() < f64::EPSILON);
        assert!((s.unscale(200.0) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_loss_scaler_overflow_halves_scale() {
        let mut s = LossScaler::new(1024.0);
        s.report_overflow();
        assert!((s.scale() - 512.0).abs() < f64::EPSILON);
        assert_eq!(s.overflow_count(), 1);
    }

    #[test]
    fn test_loss_scaler_min_scale_clamped() {
        let mut s = LossScaler::new(1.0);
        s.report_overflow();
        assert!(s.scale() >= 1.0, "Scale must not drop below 1.0");
    }

    #[test]
    fn test_loss_scaler_growth_after_interval() {
        let mut s = LossScaler::new(100.0);
        for _ in 0..2000 {
            s.report_ok();
        }
        assert!(s.scale() > 100.0, "Scale should grow after growth interval");
    }

    #[test]
    fn test_loss_scaler_check_overflow_nan() {
        let s = LossScaler::new(1.0);
        assert!(s.check_overflow(f64::NAN, CudaPrecision::FP16));
        assert!(s.check_overflow(f64::INFINITY, CudaPrecision::FP16));
    }

    #[test]
    fn test_loss_scaler_check_overflow_in_range() {
        let s = LossScaler::new(1.0);
        assert!(!s.check_overflow(1.0, CudaPrecision::FP16));
        assert!(!s.check_overflow(0.0, CudaPrecision::FP32));
    }

    #[test]
    fn test_loss_scaler_check_overflow_fp16_out_of_range() {
        let s = LossScaler::new(1.0);
        assert!(s.check_overflow(70000.0, CudaPrecision::FP16));
    }

    // ── StabilityMonitor ────────────────────────────────────────────────────

    #[test]
    fn test_stability_monitor_stable_by_default() {
        let mon = StabilityMonitor::new();
        assert!(!mon.is_unstable("layer_0"));
    }

    #[test]
    fn test_stability_monitor_nan_triggers_unstable() {
        let mut mon = StabilityMonitor::new();
        for _ in 0..3 {
            mon.record("layer_0", f64::NAN);
        }
        assert!(mon.is_unstable("layer_0"));
    }

    #[test]
    fn test_stability_monitor_high_variance_unstable() {
        let mut mon = StabilityMonitor::new();
        mon.record("layer_x", 0.0);
        mon.record("layer_x", 1e8);
        assert!(mon.is_unstable("layer_x"));
    }

    #[test]
    fn test_stability_monitor_suggests_fp32() {
        let mut mon = StabilityMonitor::new();
        for _ in 0..5 {
            mon.record("bad_layer", f64::NAN);
        }
        let upgrades = mon.suggest_upgrades();
        assert!(upgrades.iter().any(|(n, p)| n == "bad_layer" && *p == CudaPrecision::FP32));
    }

    // ── MixedPrecisionPipeline ──────────────────────────────────────────────

    #[test]
    fn test_pipeline_layer_count() {
        let pipe = MixedPrecisionPipeline::new(PrecisionPolicy::Auto, sample_layers());
        assert_eq!(pipe.layer_count(), 7);
    }

    #[test]
    fn test_pipeline_sensitive_layers() {
        let pipe = MixedPrecisionPipeline::new(PrecisionPolicy::Auto, sample_layers());
        let sensitive = pipe.sensitive_layers();
        assert!(sensitive.contains(&"layer_0.attn".to_string()));
        assert!(sensitive.contains(&"layer_0.norm".to_string()));
        assert!(!sensitive.contains(&"embed".to_string()));
    }

    #[test]
    fn test_pipeline_memory_savings_auto() {
        let pipe = MixedPrecisionPipeline::new(PrecisionPolicy::Auto, sample_layers());
        let savings = pipe.memory_savings();
        assert!(savings > 0.0 && savings < 1.0);
    }

    #[test]
    fn test_pipeline_dynamic_adjustment_upgrades() {
        let mut pipe = MixedPrecisionPipeline::new(PrecisionPolicy::Auto, sample_layers());
        // Record NaN activations to trigger upgrade
        for _ in 0..5 {
            pipe.record_activation("layer_0.ffn", f64::NAN);
        }
        // Should be upgraded to FP32
        assert_eq!(pipe.precision_for("layer_0.ffn"), Some(CudaPrecision::FP32));
    }

    #[test]
    fn test_pipeline_dynamic_adjustment_disabled() {
        let mut pipe = MixedPrecisionPipeline::new(PrecisionPolicy::Auto, sample_layers());
        pipe.set_dynamic_adjustment(false);
        for _ in 0..5 {
            pipe.record_activation("layer_0.ffn", f64::NAN);
        }
        // Should still be FP16 since dynamic adjustment is off
        assert_eq!(pipe.precision_for("layer_0.ffn"), Some(CudaPrecision::FP16));
    }

    #[test]
    fn test_pipeline_override() {
        let mut pipe = MixedPrecisionPipeline::new(PrecisionPolicy::Full, sample_layers());
        pipe.set_layer_override("embed", CudaPrecision::BF16);
        assert_eq!(pipe.precision_for("embed"), Some(CudaPrecision::BF16));
    }

    #[test]
    fn test_pipeline_cast_count_full_zero() {
        let pipe = MixedPrecisionPipeline::new(PrecisionPolicy::Full, sample_layers());
        assert_eq!(pipe.cast_count(), 0);
    }

    #[test]
    fn test_pipeline_cast_count_auto_nonzero() {
        let pipe = MixedPrecisionPipeline::new(PrecisionPolicy::Auto, sample_layers());
        assert!(pipe.cast_count() > 0);
    }

    // ── Property tests ──────────────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_precision() -> impl Strategy<Value = CudaPrecision> {
            prop_oneof![
                Just(CudaPrecision::FP32),
                Just(CudaPrecision::FP16),
                Just(CudaPrecision::BF16),
                Just(CudaPrecision::INT8),
            ]
        }

        fn arb_policy() -> impl Strategy<Value = PrecisionPolicy> {
            prop_oneof![
                Just(PrecisionPolicy::Full),
                Just(PrecisionPolicy::Half),
                Just(PrecisionPolicy::BFloat),
                Just(PrecisionPolicy::Int8),
                Just(PrecisionPolicy::Auto),
            ]
        }

        fn arb_layer_kind() -> impl Strategy<Value = LayerKind> {
            prop_oneof![
                Just(LayerKind::Attention),
                Just(LayerKind::Normalization),
                Just(LayerKind::Linear),
                Just(LayerKind::FeedForward),
                Just(LayerKind::Embedding),
                Just(LayerKind::Residual),
                Just(LayerKind::Softmax),
                Just(LayerKind::Output),
            ]
        }

        proptest! {
            #[test]
            fn prop_size_bytes_positive(p in arb_precision()) {
                prop_assert!(p.size_bytes() > 0);
            }

            #[test]
            fn prop_full_policy_always_fp32(kind in arb_layer_kind(), pc in 1_usize..10_000) {
                let sched = PrecisionScheduler::new(PrecisionPolicy::Full);
                let layer = LayerDescriptor {
                    name: "test".into(),
                    kind,
                    param_count: pc,
                    index: 0,
                };
                prop_assert_eq!(sched.resolve(&layer), CudaPrecision::FP32);
            }

            #[test]
            fn prop_sensitive_layers_fp32_under_auto(kind in arb_layer_kind(), pc in 1_usize..10_000) {
                let sched = PrecisionScheduler::new(PrecisionPolicy::Auto);
                let layer = LayerDescriptor {
                    name: "test".into(),
                    kind,
                    param_count: pc,
                    index: 0,
                };
                if kind.is_sensitive() {
                    prop_assert_eq!(sched.resolve(&layer), CudaPrecision::FP32);
                }
            }

            #[test]
            fn prop_memory_savings_bounded(policy in arb_policy()) {
                let sched = PrecisionScheduler::new(policy);
                let layers = sample_layers();
                let ratio = sched.memory_savings_ratio(&layers);
                prop_assert!(ratio >= 0.0 && ratio <= 1.0,
                    "savings ratio out of bounds: {}", ratio);
            }

            #[test]
            fn prop_loss_scaler_scale_positive(initial in 1.0_f64..1e12) {
                let s = LossScaler::new(initial);
                prop_assert!(s.scale() > 0.0);
            }

            #[test]
            fn prop_loss_scaler_round_trip(value in -1e6_f64..1e6) {
                let s = LossScaler::new(256.0);
                let scaled = s.scale_up(value);
                let back = s.unscale(scaled);
                let diff = (back - value).abs();
                prop_assert!(diff < 1e-6, "round-trip error too large: {}", diff);
            }

            #[test]
            fn prop_cast_count_le_layers_minus_one(policy in arb_policy()) {
                let layers = sample_layers();
                let sched = PrecisionScheduler::new(policy);
                let casts = sched.detect_casts(&layers);
                prop_assert!(casts.len() < layers.len(),
                    "cast count {} >= layer count {}", casts.len(), layers.len());
            }
        }
    }
}
