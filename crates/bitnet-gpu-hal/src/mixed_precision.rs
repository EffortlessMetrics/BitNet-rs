//! Module stub - implementation pending merge from feature branch
//! Mixed precision compute with type casting and overflow detection.
//!
//! Provides [`PrecisionType`] formats (FP32 / FP16 / BF16 / INT8 / INT4 /
//! TF32 / `FP8_E4M3` / `FP8_E5M2`), a [`PrecisionPolicy`] for per-layer
//! assignment, [`TypeCaster`] for conversions with proper rounding/clamping,
//! [`LossScaler`] for dynamic loss scaling, [`OverflowDetector`] for NaN/Inf
//! detection, [`AccumulationBuffer`] for higher-precision partial-result
//! accumulation, [`MixedPrecisionMatmul`] for FP16→FP32→FP16 matmul,
//! [`PrecisionAnalyzer`] for per-layer precision recommendations, and
//! [`MixedPrecisionEngine`] that orchestrates the full pipeline.

use std::collections::HashMap;
use std::fmt;

// ── Precision formats ────────────────────────────────────────────────

/// Numeric precision format for mixed-precision compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionType {
    /// 32-bit IEEE 754 floating point.
    FP32,
    /// 16-bit IEEE 754 floating point.
    FP16,
    /// 16-bit Brain floating point (truncated FP32 mantissa).
    BF16,
    /// 8-bit signed integer quantisation.
    INT8,
    /// 4-bit signed integer quantisation.
    INT4,
    /// TensorFloat-32 (19-bit: 8-bit exponent, 10-bit mantissa).
    TF32,
    /// FP8 E4M3 (1 sign, 4 exponent, 3 mantissa).
    FP8E4M3,
    /// FP8 E5M2 (1 sign, 5 exponent, 2 mantissa).
    FP8E5M2,
}

impl PrecisionType {
    /// Size in bits of a single element.
    #[must_use]
    pub const fn bits(&self) -> u32 {
        match self {
            Self::FP32 => 32,
            Self::FP16 | Self::BF16 => 16,
            Self::INT8 | Self::FP8E4M3 | Self::FP8E5M2 => 8,
            Self::INT4 => 4,
            Self::TF32 => 19,
        }
    }

    /// Size in bytes of a single element (rounded up for sub-byte).
    #[must_use]
    pub const fn bytes(&self) -> u32 {
        self.bits().div_ceil(8)
    }

    /// Whether the format is a floating-point type.
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            Self::FP32 | Self::FP16 | Self::BF16 | Self::TF32 | Self::FP8E4M3 | Self::FP8E5M2
        )
    }

    /// Whether the format is a quantised integer type.
    #[must_use]
    pub const fn is_quantized(&self) -> bool {
        matches!(self, Self::INT8 | Self::INT4)
    }

    /// Maximum representable value (approximate for float types).
    #[must_use]
    pub fn max_value(&self) -> f64 {
        match self {
            Self::FP32 | Self::TF32 => f64::from(f32::MAX),
            Self::FP16 => 65504.0,
            Self::BF16 => 3.389_531_389_251_535e38,
            Self::INT8 => 127.0,
            Self::INT4 => 7.0,
            Self::FP8E4M3 => 448.0,
            Self::FP8E5M2 => 57344.0,
        }
    }

    /// Dynamic range in powers of two (log2).
    #[must_use]
    pub const fn dynamic_range_log2(&self) -> f64 {
        match self {
            Self::FP32 | Self::BF16 | Self::TF32 => 254.0,
            Self::FP16 | Self::FP8E5M2 => 30.0,
            Self::INT8 => 7.97,
            Self::INT4 => 3.91,
            Self::FP8E4M3 => 16.0,
        }
    }
}

impl fmt::Display for PrecisionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FP32 => write!(f, "FP32"),
            Self::FP16 => write!(f, "FP16"),
            Self::BF16 => write!(f, "BF16"),
            Self::INT8 => write!(f, "INT8"),
            Self::INT4 => write!(f, "INT4"),
            Self::TF32 => write!(f, "TF32"),
            Self::FP8E4M3 => write!(f, "FP8_E4M3"),
            Self::FP8E5M2 => write!(f, "FP8_E5M2"),
        }
    }
}

// ── Precision config ─────────────────────────────────────────────────

/// Configuration describing the dtypes used in a mixed-precision pipeline.
#[derive(Debug, Clone)]
pub struct PrecisionConfig {
    /// Precision used for compute (e.g. matrix multiply).
    pub compute_dtype: PrecisionType,
    /// Precision used for accumulation of partial results.
    pub accumulate_dtype: PrecisionType,
    /// Precision used for weight / activation storage.
    pub storage_dtype: PrecisionType,
    /// Static loss-scale factor (1.0 = disabled).
    pub loss_scale: f64,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            compute_dtype: PrecisionType::FP16,
            accumulate_dtype: PrecisionType::FP32,
            storage_dtype: PrecisionType::FP16,
            loss_scale: 1.0,
        }
    }
}

// ── Precision policy ─────────────────────────────────────────────────

/// Layer type classification for precision assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    /// Self-attention layers.
    Attention,
    /// Feed-forward / MLP layers.
    FeedForward,
    /// Embedding or output projection layers.
    Embedding,
    /// Layer normalisation.
    LayerNorm,
    /// Generic / unknown layer.
    Other,
}

/// Rules controlling which layers use which precision.
#[derive(Debug, Clone)]
pub struct PrecisionPolicy {
    /// Per-layer-kind precision overrides.
    pub layer_rules: HashMap<LayerKind, PrecisionConfig>,
    /// Default config applied when no rule matches.
    pub default_config: PrecisionConfig,
    /// Per-layer name overrides (highest priority).
    pub name_overrides: HashMap<String, PrecisionType>,
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        let mut layer_rules = HashMap::new();
        layer_rules.insert(
            LayerKind::Attention,
            PrecisionConfig {
                compute_dtype: PrecisionType::FP16,
                accumulate_dtype: PrecisionType::FP32,
                storage_dtype: PrecisionType::FP16,
                loss_scale: 1.0,
            },
        );
        layer_rules.insert(
            LayerKind::FeedForward,
            PrecisionConfig {
                compute_dtype: PrecisionType::FP16,
                accumulate_dtype: PrecisionType::FP32,
                storage_dtype: PrecisionType::FP16,
                loss_scale: 1.0,
            },
        );
        layer_rules.insert(
            LayerKind::LayerNorm,
            PrecisionConfig {
                compute_dtype: PrecisionType::FP32,
                accumulate_dtype: PrecisionType::FP32,
                storage_dtype: PrecisionType::FP32,
                loss_scale: 1.0,
            },
        );
        Self {
            layer_rules,
            default_config: PrecisionConfig::default(),
            name_overrides: HashMap::new(),
        }
    }
}

impl PrecisionPolicy {
    /// Resolve the full config for a given layer.
    #[must_use]
    pub fn resolve(&self, name: &str, kind: LayerKind) -> &PrecisionConfig {
        if self.name_overrides.contains_key(name) {
            return self.layer_rules.get(&kind).unwrap_or(&self.default_config);
        }
        self.layer_rules.get(&kind).unwrap_or(&self.default_config)
    }

    /// Resolve the compute dtype for a named layer, respecting overrides.
    #[must_use]
    pub fn compute_dtype_for(&self, name: &str, kind: LayerKind) -> PrecisionType {
        if let Some(&dt) = self.name_overrides.get(name) {
            return dt;
        }
        self.layer_rules.get(&kind).unwrap_or(&self.default_config).compute_dtype
    }
}

// ── Type caster ──────────────────────────────────────────────────────

/// Error returned by precision conversion operations.
#[derive(Debug, thiserror::Error)]
pub enum CastError {
    /// Value overflows the target precision range.
    #[error("overflow: value {value} exceeds {target} range")]
    Overflow { value: f64, target: PrecisionType },
    /// Unsupported conversion pair.
    #[error("unsupported conversion: {from} -> {to}")]
    Unsupported { from: PrecisionType, to: PrecisionType },
    /// Empty input slice.
    #[error("empty input")]
    EmptyInput,
}

/// Converts between precision types with proper rounding and clamping.
#[derive(Debug, Default)]
pub struct TypeCaster;

impl TypeCaster {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Cast a slice of f32 values to the target precision, returning
    /// f64 for uniform downstream handling.
    ///
    /// # Errors
    ///
    /// Returns [`CastError`] on empty input or unsupported source dtype.
    pub fn cast(
        &self,
        data: &[f32],
        from: PrecisionType,
        to: PrecisionType,
    ) -> Result<Vec<f64>, CastError> {
        if data.is_empty() {
            return Err(CastError::EmptyInput);
        }
        if from == to {
            return Ok(data.iter().map(|&v| f64::from(v)).collect());
        }
        if from != PrecisionType::FP32 {
            return Err(CastError::Unsupported { from, to });
        }
        match to {
            PrecisionType::FP16 => Ok(data.iter().map(|&v| Self::to_fp16(v)).collect()),
            PrecisionType::BF16 => Ok(data.iter().map(|&v| Self::to_bf16(v)).collect()),
            PrecisionType::INT8 => Ok(Self::to_int8(data)),
            PrecisionType::INT4 => Ok(Self::to_int4(data)),
            PrecisionType::TF32 => Ok(data.iter().map(|&v| Self::to_tf32(v)).collect()),
            PrecisionType::FP8E4M3 => Ok(data.iter().map(|&v| Self::to_fp8_e4m3(v)).collect()),
            PrecisionType::FP8E5M2 => Ok(data.iter().map(|&v| Self::to_fp8_e5m2(v)).collect()),
            PrecisionType::FP32 => Ok(data.iter().map(|&v| f64::from(v)).collect()),
        }
    }

    /// Round-trip error: cast `FP32 → via → compare`.
    ///
    /// # Errors
    ///
    /// Returns [`CastError`] if conversion is unsupported or empty.
    pub fn round_trip_error(&self, data: &[f32], via: PrecisionType) -> Result<f64, CastError> {
        let converted = self.cast(data, PrecisionType::FP32, via)?;
        let max_err = data
            .iter()
            .zip(converted.iter())
            .map(|(&o, &c)| (f64::from(o) - c).abs())
            .fold(0.0_f64, f64::max);
        Ok(max_err)
    }

    /// Whether this caster supports the given conversion pair.
    #[must_use]
    pub const fn supports(&self, from: PrecisionType, to: PrecisionType) -> bool {
        from as u8 == to as u8 || from as u8 == PrecisionType::FP32 as u8
    }

    // ── Private helpers ──────────────────────────────────────────────

    fn to_fp16(v: f32) -> f64 {
        let clamped = v.clamp(-65504.0, 65504.0);
        let scale = 2048.0_f64;
        (f64::from(clamped) * scale).round() / scale
    }

    fn to_bf16(v: f32) -> f64 {
        let bits = v.to_bits();
        let truncated = f32::from_bits(bits & 0xFFFF_0000);
        f64::from(truncated)
    }

    fn to_tf32(v: f32) -> f64 {
        let bits = v.to_bits();
        let truncated = f32::from_bits(bits & 0xFFFF_E000);
        f64::from(truncated)
    }

    fn to_fp8_e4m3(v: f32) -> f64 {
        let clamped = v.clamp(-448.0, 448.0);
        let scale = 8.0_f64;
        (f64::from(clamped) * scale).round() / scale
    }

    fn to_fp8_e5m2(v: f32) -> f64 {
        let clamped = v.clamp(-57344.0, 57344.0);
        let scale = 4.0_f64;
        (f64::from(clamped) * scale).round() / scale
    }

    fn to_int8(data: &[f32]) -> Vec<f64> {
        if data.is_empty() {
            return Vec::new();
        }
        let absmax = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        if absmax == 0.0 {
            return vec![0.0; data.len()];
        }
        let scale = 127.0_f64 / f64::from(absmax);
        data.iter()
            .map(|&v| {
                let q = (f64::from(v) * scale).round().clamp(-127.0, 127.0);
                q / scale
            })
            .collect()
    }

    fn to_int4(data: &[f32]) -> Vec<f64> {
        if data.is_empty() {
            return Vec::new();
        }
        let absmax = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        if absmax == 0.0 {
            return vec![0.0; data.len()];
        }
        let scale = 7.0_f64 / f64::from(absmax);
        data.iter()
            .map(|&v| {
                let q = (f64::from(v) * scale).round().clamp(-7.0, 7.0);
                q / scale
            })
            .collect()
    }
}

// ── Loss scaler ──────────────────────────────────────────────────────

/// Dynamic loss scaler for mixed-precision training.
///
/// Scales gradients up before the backward pass to prevent underflow in
/// reduced-precision formats, then scales them back down. On overflow the
/// scale is reduced and the optimiser step is skipped.
#[derive(Debug, Clone)]
pub struct LossScaler {
    current_scale: f64,
    growth_interval: u64,
    growth_factor: f64,
    backoff_factor: f64,
    min_scale: f64,
    steps_since_last_overflow: u64,
    overflow_count: u64,
    total_steps: u64,
}

impl LossScaler {
    /// Create a new loss scaler with the given initial scale.
    #[must_use]
    pub const fn new(initial_scale: f64) -> Self {
        Self {
            current_scale: initial_scale,
            growth_interval: 2000,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            min_scale: 1.0,
            steps_since_last_overflow: 0,
            overflow_count: 0,
            total_steps: 0,
        }
    }

    /// Builder: set the growth interval.
    #[must_use]
    pub const fn with_growth_interval(mut self, n: u64) -> Self {
        self.growth_interval = n;
        self
    }

    /// Builder: set the growth factor.
    #[must_use]
    pub const fn with_growth_factor(mut self, f: f64) -> Self {
        self.growth_factor = f;
        self
    }

    /// Builder: set the back-off factor.
    #[must_use]
    pub const fn with_backoff_factor(mut self, f: f64) -> Self {
        self.backoff_factor = f;
        self
    }

    /// Builder: set the minimum allowed scale.
    #[must_use]
    pub const fn with_min_scale(mut self, s: f64) -> Self {
        self.min_scale = s;
        self
    }

    /// Current loss scale value.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.current_scale
    }

    /// Scale a gradient value up.
    #[must_use]
    pub fn scale_gradient(&self, grad: f64) -> f64 {
        grad * self.current_scale
    }

    /// Un-scale a gradient value (inverse of [`Self::scale_gradient`]).
    #[must_use]
    pub fn unscale_gradient(&self, scaled_grad: f64) -> f64 {
        scaled_grad / self.current_scale
    }

    /// Record a step outcome. Returns `true` if the step is valid (no
    /// overflow) and the optimiser should apply the update.
    pub fn update(&mut self, overflow_detected: bool) -> bool {
        self.total_steps += 1;
        if overflow_detected {
            self.overflow_count += 1;
            self.steps_since_last_overflow = 0;
            self.current_scale *= self.backoff_factor;
            if self.current_scale < self.min_scale {
                self.current_scale = self.min_scale;
            }
            log::debug!("LossScaler overflow: scale reduced to {}", self.current_scale,);
            return false;
        }
        self.steps_since_last_overflow += 1;
        if self.steps_since_last_overflow >= self.growth_interval {
            self.current_scale *= self.growth_factor;
            self.steps_since_last_overflow = 0;
            log::debug!("LossScaler growth: scale increased to {}", self.current_scale,);
        }
        true
    }

    /// Number of overflow events recorded.
    #[must_use]
    pub const fn overflow_count(&self) -> u64 {
        self.overflow_count
    }

    /// Total steps processed.
    #[must_use]
    pub const fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Steps since the last overflow.
    #[must_use]
    pub const fn steps_since_overflow(&self) -> u64 {
        self.steps_since_last_overflow
    }
}

// ── Overflow detector ────────────────────────────────────────────────

/// Detects NaN / Inf arising from precision underflow or overflow.
#[derive(Debug, Default)]
pub struct OverflowDetector {
    nan_count: u64,
    inf_count: u64,
    checks: u64,
}

impl OverflowDetector {
    #[must_use]
    pub const fn new() -> Self {
        Self { nan_count: 0, inf_count: 0, checks: 0 }
    }

    /// Check a slice of f64 values for NaN or Inf, updating counters.
    /// Returns `true` if any anomaly is found.
    pub fn check(&mut self, values: &[f64]) -> bool {
        self.checks += 1;
        let mut found = false;
        for &v in values {
            if v.is_nan() {
                self.nan_count += 1;
                found = true;
            } else if v.is_infinite() {
                self.inf_count += 1;
                found = true;
            }
        }
        found
    }

    /// Check a slice of f32 values.
    pub fn check_f32(&mut self, values: &[f32]) -> bool {
        self.checks += 1;
        let mut found = false;
        for &v in values {
            if v.is_nan() {
                self.nan_count += 1;
                found = true;
            } else if v.is_infinite() {
                self.inf_count += 1;
                found = true;
            }
        }
        found
    }

    /// Total NaN values detected across all checks.
    #[must_use]
    pub const fn nan_count(&self) -> u64 {
        self.nan_count
    }

    /// Total Inf values detected across all checks.
    #[must_use]
    pub const fn inf_count(&self) -> u64 {
        self.inf_count
    }

    /// Total number of check calls.
    #[must_use]
    pub const fn total_checks(&self) -> u64 {
        self.checks
    }

    /// Whether any anomaly has ever been detected.
    #[must_use]
    pub const fn has_detected_anomaly(&self) -> bool {
        self.nan_count > 0 || self.inf_count > 0
    }

    /// Reset all counters.
    pub const fn reset(&mut self) {
        self.nan_count = 0;
        self.inf_count = 0;
        self.checks = 0;
    }
}

// ── Accumulation buffer ──────────────────────────────────────────────

/// Higher-precision buffer for accumulating partial results.
///
/// Computations are performed in a reduced-precision format but partial
/// sums are accumulated in f64 to avoid catastrophic cancellation.
#[derive(Debug, Clone)]
pub struct AccumulationBuffer {
    data: Vec<f64>,
    accumulate_dtype: PrecisionType,
}

impl AccumulationBuffer {
    /// Create a zero-initialised buffer of the given length.
    #[must_use]
    pub fn new(len: usize, accumulate_dtype: PrecisionType) -> Self {
        Self { data: vec![0.0; len], accumulate_dtype }
    }

    /// Accumulate (add) a slice of partial results into the buffer.
    ///
    /// # Panics
    ///
    /// Panics if `values.len() != self.len()`.
    pub fn accumulate(&mut self, values: &[f64]) {
        assert_eq!(values.len(), self.data.len(), "accumulate length mismatch");
        for (acc, &v) in self.data.iter_mut().zip(values.iter()) {
            *acc += v;
        }
    }

    /// Read the accumulated results.
    #[must_use]
    pub fn result(&self) -> &[f64] {
        &self.data
    }

    /// Length of the buffer.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reset all elements to zero.
    pub fn reset(&mut self) {
        self.data.fill(0.0);
    }

    /// The accumulation dtype.
    #[must_use]
    pub const fn dtype(&self) -> PrecisionType {
        self.accumulate_dtype
    }
}

// ── Mixed-precision matmul ───────────────────────────────────────────

/// FP16 input → FP32 accumulate → FP16 output matrix multiply.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
pub struct MixedPrecisionMatmul {
    input_dtype: PrecisionType,
    accumulate_dtype: PrecisionType,
    output_dtype: PrecisionType,
}

impl Default for MixedPrecisionMatmul {
    fn default() -> Self {
        Self {
            input_dtype: PrecisionType::FP16,
            accumulate_dtype: PrecisionType::FP32,
            output_dtype: PrecisionType::FP16,
        }
    }
}

impl MixedPrecisionMatmul {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with explicit dtype triple.
    #[must_use]
    pub const fn with_dtypes(
        input: PrecisionType,
        accumulate: PrecisionType,
        output: PrecisionType,
    ) -> Self {
        Self { input_dtype: input, accumulate_dtype: accumulate, output_dtype: output }
    }

    /// Perform C = A × B using mixed-precision accumulation.
    ///
    /// `lhs` is (rows × inner), `rhs` is (inner × cols), result is
    /// (rows × cols), all stored as row-major f32 slices.
    ///
    /// # Errors
    ///
    /// Returns [`CastError`] if dimensions do not match or if precision
    /// conversion fails.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul(
        &self,
        lhs: &[f32],
        rows: usize,
        inner: usize,
        rhs: &[f32],
        cols: usize,
    ) -> Result<Vec<f32>, CastError> {
        if lhs.len() != rows * inner || rhs.len() != inner * cols {
            return Err(CastError::EmptyInput);
        }
        let caster = TypeCaster::new();

        let lhs_cast = caster.cast(lhs, PrecisionType::FP32, self.input_dtype)?;
        let rhs_cast = caster.cast(rhs, PrecisionType::FP32, self.input_dtype)?;

        let mut out = vec![0.0_f64; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0.0_f64;
                for p in 0..inner {
                    sum += lhs_cast[i * inner + p] * rhs_cast[p * cols + j];
                }
                out[i * cols + j] = sum;
            }
        }

        #[allow(clippy::cast_possible_truncation)]
        let result: Vec<f32> = out.iter().map(|&v| v as f32).collect();
        let _output_cast = caster.cast(&result, PrecisionType::FP32, self.output_dtype)?;

        Ok(result)
    }

    #[must_use]
    pub const fn input_dtype(&self) -> PrecisionType {
        self.input_dtype
    }

    #[must_use]
    pub const fn accumulate_dtype(&self) -> PrecisionType {
        self.accumulate_dtype
    }

    #[must_use]
    pub const fn output_dtype(&self) -> PrecisionType {
        self.output_dtype
    }
}

// ── Precision analyser ───────────────────────────────────────────────

/// Report from analysing precision trade-offs for one layer.
#[derive(Debug, Clone)]
pub struct PrecisionReport {
    /// Layer name.
    pub layer_name: String,
    /// Recommended precision.
    pub recommended: PrecisionType,
    /// Estimated memory savings ratio (0.0–1.0).
    pub memory_savings: f64,
    /// Estimated throughput multiplier vs FP32.
    pub throughput_multiplier: f64,
    /// Maximum absolute conversion error observed.
    pub max_error: f64,
    /// Mean absolute conversion error observed.
    pub mean_error: f64,
}

/// Analyses a model's layers and recommends per-layer precision settings.
#[derive(Debug)]
pub struct PrecisionAnalyzer {
    error_threshold: f64,
}

impl Default for PrecisionAnalyzer {
    fn default() -> Self {
        Self { error_threshold: 0.01 }
    }
}

impl PrecisionAnalyzer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum acceptable conversion error.
    #[must_use]
    pub const fn with_error_threshold(mut self, t: f64) -> Self {
        self.error_threshold = t;
        self
    }

    /// Analyse a single layer's weights and recommend a precision.
    ///
    /// # Errors
    ///
    /// Returns [`CastError`] if the sample data is empty.
    pub fn analyse_layer(&self, name: &str, sample: &[f32]) -> Result<PrecisionReport, CastError> {
        if sample.is_empty() {
            return Err(CastError::EmptyInput);
        }
        let caster = TypeCaster::new();
        let candidates =
            [PrecisionType::FP16, PrecisionType::BF16, PrecisionType::INT8, PrecisionType::FP8E4M3];

        let mut best = PrecisionType::FP32;
        let mut best_savings = 0.0_f64;
        let mut best_max_err = 0.0_f64;
        let mut best_mean_err = 0.0_f64;

        for &candidate in &candidates {
            if let Ok(converted) = caster.cast(sample, PrecisionType::FP32, candidate) {
                let errors: Vec<f64> = sample
                    .iter()
                    .zip(converted.iter())
                    .map(|(&o, &c)| (f64::from(o) - c).abs())
                    .collect();
                let max_e = errors.iter().copied().fold(0.0_f64, f64::max);
                if max_e <= self.error_threshold {
                    let savings =
                        1.0 - (f64::from(candidate.bits()) / f64::from(PrecisionType::FP32.bits()));
                    if savings > best_savings {
                        best = candidate;
                        best_savings = savings;
                        best_max_err = max_e;
                        #[allow(clippy::cast_precision_loss)]
                        {
                            best_mean_err = errors.iter().sum::<f64>() / errors.len() as f64;
                        }
                    }
                }
            }
        }

        let throughput_multiplier = f64::from(PrecisionType::FP32.bits()) / f64::from(best.bits());

        Ok(PrecisionReport {
            layer_name: name.to_string(),
            recommended: best,
            memory_savings: best_savings,
            throughput_multiplier,
            max_error: best_max_err,
            mean_error: best_mean_err,
        })
    }

    /// Analyse multiple layers and return reports.
    ///
    /// # Errors
    ///
    /// Returns [`CastError`] if any sample is empty.
    pub fn analyse_model(
        &self,
        layers: &[(&str, &[f32])],
    ) -> Result<Vec<PrecisionReport>, CastError> {
        layers.iter().map(|&(name, sample)| self.analyse_layer(name, sample)).collect()
    }
}

// ── Mixed-precision engine ───────────────────────────────────────────

/// A single layer's precision assignment within the engine.
#[derive(Debug, Clone)]
pub struct LayerPrecision {
    /// Layer name.
    pub name: String,
    /// Layer type classification.
    pub kind: LayerKind,
    /// Assigned compute precision.
    pub precision: PrecisionType,
    /// Number of parameters in this layer.
    pub param_count: u64,
}

/// Orchestrator: analyse → set policy → cast → compute → cast back.
///
/// Ties together [`PrecisionPolicy`], [`TypeCaster`], [`LossScaler`],
/// and [`OverflowDetector`] into a single workflow.
#[derive(Debug)]
pub struct MixedPrecisionEngine {
    policy: PrecisionPolicy,
    layers: Vec<LayerPrecision>,
    loss_scaler: Option<LossScaler>,
    overflow_detector: OverflowDetector,
}

impl MixedPrecisionEngine {
    /// Create an engine from a policy.
    #[must_use]
    pub const fn new(policy: PrecisionPolicy) -> Self {
        Self {
            policy,
            layers: Vec::new(),
            loss_scaler: None,
            overflow_detector: OverflowDetector::new(),
        }
    }

    /// Enable dynamic loss scaling with the given initial scale.
    pub const fn enable_loss_scaling(&mut self, initial_scale: f64) {
        self.loss_scaler = Some(LossScaler::new(initial_scale));
    }

    /// Register a layer with automatic precision assignment.
    pub fn add_layer(&mut self, name: impl Into<String>, kind: LayerKind, param_count: u64) {
        let name = name.into();
        let precision = self.policy.compute_dtype_for(&name, kind);
        self.layers.push(LayerPrecision { name, kind, precision, param_count });
    }

    /// Return all registered layers.
    #[must_use]
    pub fn layers(&self) -> &[LayerPrecision] {
        &self.layers
    }

    /// Return the precision assigned to a layer by name.
    #[must_use]
    pub fn layer_precision(&self, name: &str) -> Option<PrecisionType> {
        self.layers.iter().find(|l| l.name == name).map(|l| l.precision)
    }

    /// Immutable reference to the loss scaler, if enabled.
    #[must_use]
    pub const fn loss_scaler(&self) -> Option<&LossScaler> {
        self.loss_scaler.as_ref()
    }

    /// Mutable reference to the loss scaler, if enabled.
    pub const fn loss_scaler_mut(&mut self) -> Option<&mut LossScaler> {
        self.loss_scaler.as_mut()
    }

    /// Immutable reference to the overflow detector.
    #[must_use]
    pub const fn overflow_detector(&self) -> &OverflowDetector {
        &self.overflow_detector
    }

    /// Mutable reference to the overflow detector.
    pub const fn overflow_detector_mut(&mut self) -> &mut OverflowDetector {
        &mut self.overflow_detector
    }

    /// Total parameter count across all registered layers.
    #[must_use]
    pub fn total_params(&self) -> u64 {
        self.layers.iter().map(|l| l.param_count).sum()
    }

    /// Estimated memory usage in bytes with current precision
    /// assignments.
    #[must_use]
    pub fn estimated_memory_bytes(&self) -> u64 {
        self.layers.iter().map(|l| l.param_count * u64::from(l.precision.bytes())).sum()
    }

    /// Estimated memory if all layers used the given precision.
    #[must_use]
    pub fn memory_at_precision(&self, precision: PrecisionType) -> u64 {
        self.total_params() * u64::from(precision.bytes())
    }

    /// Memory savings ratio compared to full FP32.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn memory_savings_ratio(&self) -> f64 {
        let fp32 = self.memory_at_precision(PrecisionType::FP32);
        if fp32 == 0 {
            return 0.0;
        }
        let current = self.estimated_memory_bytes();
        1.0 - (current as f64 / fp32 as f64)
    }

    /// Current active policy.
    #[must_use]
    pub const fn policy(&self) -> &PrecisionPolicy {
        &self.policy
    }

    /// Run a forward-pass step: cast inputs, check for overflow, and
    /// optionally apply loss scaling. Returns the cast result.
    ///
    /// # Errors
    ///
    /// Returns [`CastError`] on conversion failure.
    pub fn forward_step(&mut self, layer_name: &str, data: &[f32]) -> Result<Vec<f64>, CastError> {
        let precision = self.layer_precision(layer_name).unwrap_or(PrecisionType::FP32);
        let caster = TypeCaster::new();
        let cast = caster.cast(data, PrecisionType::FP32, precision)?;

        let has_overflow = self.overflow_detector.check(&cast);
        if let Some(ref mut scaler) = self.loss_scaler {
            scaler.update(has_overflow);
        }
        Ok(cast)
    }
}

// ──────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── PrecisionType ────────────────────────────────────────────────

    #[test]
    fn precision_bits_basic() {
        assert_eq!(PrecisionType::FP32.bits(), 32);
        assert_eq!(PrecisionType::FP16.bits(), 16);
        assert_eq!(PrecisionType::BF16.bits(), 16);
        assert_eq!(PrecisionType::INT8.bits(), 8);
        assert_eq!(PrecisionType::INT4.bits(), 4);
    }

    #[test]
    fn precision_bits_new_types() {
        assert_eq!(PrecisionType::TF32.bits(), 19);
        assert_eq!(PrecisionType::FP8E4M3.bits(), 8);
        assert_eq!(PrecisionType::FP8E5M2.bits(), 8);
    }

    #[test]
    fn precision_bytes() {
        assert_eq!(PrecisionType::FP32.bytes(), 4);
        assert_eq!(PrecisionType::FP16.bytes(), 2);
        assert_eq!(PrecisionType::INT8.bytes(), 1);
        assert_eq!(PrecisionType::INT4.bytes(), 1);
        assert_eq!(PrecisionType::TF32.bytes(), 3);
        assert_eq!(PrecisionType::FP8E4M3.bytes(), 1);
    }

    #[test]
    fn precision_is_float() {
        assert!(PrecisionType::FP32.is_float());
        assert!(PrecisionType::FP16.is_float());
        assert!(PrecisionType::BF16.is_float());
        assert!(PrecisionType::TF32.is_float());
        assert!(PrecisionType::FP8E4M3.is_float());
        assert!(PrecisionType::FP8E5M2.is_float());
        assert!(!PrecisionType::INT8.is_float());
        assert!(!PrecisionType::INT4.is_float());
    }

    #[test]
    fn precision_is_quantized() {
        assert!(!PrecisionType::FP32.is_quantized());
        assert!(PrecisionType::INT8.is_quantized());
        assert!(PrecisionType::INT4.is_quantized());
        assert!(!PrecisionType::FP8E4M3.is_quantized());
    }

    #[test]
    fn precision_display() {
        assert_eq!(format!("{}", PrecisionType::FP32), "FP32");
        assert_eq!(format!("{}", PrecisionType::TF32), "TF32");
        assert_eq!(format!("{}", PrecisionType::FP8E4M3), "FP8_E4M3");
        assert_eq!(format!("{}", PrecisionType::FP8E5M2), "FP8_E5M2");
    }

    #[test]
    fn precision_max_value_ordering() {
        assert!(PrecisionType::FP32.max_value() > PrecisionType::FP16.max_value());
        assert!(PrecisionType::FP16.max_value() > PrecisionType::INT8.max_value());
        assert!(PrecisionType::INT8.max_value() > PrecisionType::INT4.max_value());
    }

    #[test]
    fn precision_fp8_max_values() {
        assert_eq!(PrecisionType::FP8E4M3.max_value(), 448.0);
        assert_eq!(PrecisionType::FP8E5M2.max_value(), 57344.0);
    }

    #[test]
    fn precision_dynamic_range() {
        assert!(
            PrecisionType::FP32.dynamic_range_log2() > PrecisionType::FP16.dynamic_range_log2()
        );
        assert_eq!(
            PrecisionType::BF16.dynamic_range_log2(),
            PrecisionType::FP32.dynamic_range_log2(),
        );
        assert_eq!(
            PrecisionType::TF32.dynamic_range_log2(),
            PrecisionType::FP32.dynamic_range_log2(),
        );
    }

    #[test]
    fn precision_eq_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PrecisionType::FP32);
        set.insert(PrecisionType::FP16);
        set.insert(PrecisionType::FP32);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn precision_clone_copy() {
        let p = PrecisionType::INT8;
        let p2 = p;
        assert_eq!(p, p2);
    }

    // ── PrecisionConfig ──────────────────────────────────────────────

    #[test]
    fn config_defaults() {
        let cfg = PrecisionConfig::default();
        assert_eq!(cfg.compute_dtype, PrecisionType::FP16);
        assert_eq!(cfg.accumulate_dtype, PrecisionType::FP32);
        assert_eq!(cfg.storage_dtype, PrecisionType::FP16);
        assert_eq!(cfg.loss_scale, 1.0);
    }

    #[test]
    fn config_custom() {
        let cfg = PrecisionConfig {
            compute_dtype: PrecisionType::BF16,
            accumulate_dtype: PrecisionType::FP32,
            storage_dtype: PrecisionType::INT8,
            loss_scale: 512.0,
        };
        assert_eq!(cfg.compute_dtype, PrecisionType::BF16);
        assert_eq!(cfg.loss_scale, 512.0);
    }

    // ── PrecisionPolicy ──────────────────────────────────────────────

    #[test]
    fn policy_default_has_layer_rules() {
        let pol = PrecisionPolicy::default();
        assert!(pol.layer_rules.contains_key(&LayerKind::Attention));
        assert!(pol.layer_rules.contains_key(&LayerKind::LayerNorm));
    }

    #[test]
    fn policy_layernorm_uses_fp32() {
        let pol = PrecisionPolicy::default();
        let dt = pol.compute_dtype_for("ln0", LayerKind::LayerNorm);
        assert_eq!(dt, PrecisionType::FP32);
    }

    #[test]
    fn policy_attention_uses_fp16() {
        let pol = PrecisionPolicy::default();
        let dt = pol.compute_dtype_for("attn0", LayerKind::Attention);
        assert_eq!(dt, PrecisionType::FP16);
    }

    #[test]
    fn policy_name_override() {
        let mut pol = PrecisionPolicy::default();
        pol.name_overrides.insert("special".to_string(), PrecisionType::INT8);
        let dt = pol.compute_dtype_for("special", LayerKind::Attention);
        assert_eq!(dt, PrecisionType::INT8);
    }

    #[test]
    fn policy_fallback_to_default_config() {
        let pol = PrecisionPolicy::default();
        let dt = pol.compute_dtype_for("unknown", LayerKind::Other);
        assert_eq!(dt, pol.default_config.compute_dtype);
    }

    // ── TypeCaster ───────────────────────────────────────────────────

    #[test]
    fn caster_identity() {
        let c = TypeCaster::new();
        let data = vec![1.0_f32, -2.0, 3.5];
        let r = c.cast(&data, PrecisionType::FP32, PrecisionType::FP32).unwrap();
        assert_eq!(r, vec![1.0, -2.0, 3.5]);
    }

    #[test]
    fn caster_fp32_to_fp16_small() {
        let c = TypeCaster::new();
        let data = vec![1.0_f32, 0.5, -0.25];
        let r = c.cast(&data, PrecisionType::FP32, PrecisionType::FP16).unwrap();
        for (&o, &v) in data.iter().zip(r.iter()) {
            assert!((f64::from(o) - v).abs() < 0.01);
        }
    }

    #[test]
    fn caster_fp16_clamps_overflow() {
        let c = TypeCaster::new();
        let data = vec![100_000.0_f32];
        let r = c.cast(&data, PrecisionType::FP32, PrecisionType::FP16).unwrap();
        assert!(r[0] <= 65504.0);
    }

    #[test]
    fn caster_bf16_large_value() {
        let c = TypeCaster::new();
        let data = vec![1.0e30_f32];
        let r = c.cast(&data, PrecisionType::FP32, PrecisionType::BF16).unwrap();
        let rel = (f64::from(data[0]) - r[0]).abs() / f64::from(data[0]).abs();
        assert!(rel < 0.01);
    }

    #[test]
    fn caster_int8_symmetric() {
        let c = TypeCaster::new();
        let data = vec![1.0_f32, -1.0, 0.0, 0.5];
        let r = c.cast(&data, PrecisionType::FP32, PrecisionType::INT8).unwrap();
        assert!((r[0] + r[1]).abs() < 0.02);
        assert!(r[2].abs() < 0.01);
    }

    #[test]
    fn caster_int8_all_zeros() {
        let c = TypeCaster::new();
        let data = vec![0.0_f32; 4];
        let r = c.cast(&data, PrecisionType::FP32, PrecisionType::INT8).unwrap();
        assert!(r.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn caster_int4_range() {
        let c = TypeCaster::new();
        let data = vec![1.0_f32, -1.0, 0.5, -0.5, 0.0];
        let r = c.cast(&data, PrecisionType::FP32, PrecisionType::INT4).unwrap();
        for (&o, &v) in data.iter().zip(r.iter()) {
            assert!((f64::from(o) - v).abs() < 0.2);
        }
    }

    #[test]
    fn caster_tf32() {
        let c = TypeCaster::new();
        let data = vec![3.14159_f32, -2.71828];
        let r = c.cast(&data, PrecisionType::FP32, PrecisionType::TF32).unwrap();
        for (&o, &v) in data.iter().zip(r.iter()) {
            assert!((f64::from(o) - v).abs() < 0.01);
        }
    }

    #[test]
    fn caster_fp8_e4m3() {
        let c = TypeCaster::new();
        let data = vec![1.0_f32, -0.5, 100.0];
        let r = c.cast(&data, PrecisionType::FP32, PrecisionType::FP8E4M3).unwrap();
        for (&o, &v) in data.iter().zip(r.iter()) {
            assert!((f64::from(o) - v).abs() < 0.2);
        }
    }

    #[test]
    fn caster_fp8_e4m3_clamps() {
        let c = TypeCaster::new();
        let data = vec![1000.0_f32];
        let r = c.cast(&data, PrecisionType::FP32, PrecisionType::FP8E4M3).unwrap();
        assert!(r[0] <= 448.0);
    }

    #[test]
    fn caster_fp8_e5m2() {
        let c = TypeCaster::new();
        let data = vec![1.0_f32, -0.5, 100.0];
        let r = c.cast(&data, PrecisionType::FP32, PrecisionType::FP8E5M2).unwrap();
        for (&o, &v) in data.iter().zip(r.iter()) {
            assert!((f64::from(o) - v).abs() < 0.5);
        }
    }

    #[test]
    fn caster_fp8_e5m2_clamps() {
        let c = TypeCaster::new();
        let data = vec![100_000.0_f32];
        let r = c.cast(&data, PrecisionType::FP32, PrecisionType::FP8E5M2).unwrap();
        assert!(r[0] <= 57344.0);
    }

    #[test]
    fn caster_empty_input() {
        let c = TypeCaster::new();
        assert!(c.cast(&[], PrecisionType::FP32, PrecisionType::FP16).is_err());
    }

    #[test]
    fn caster_unsupported_source() {
        let c = TypeCaster::new();
        let data = vec![1.0_f32];
        assert!(c.cast(&data, PrecisionType::INT8, PrecisionType::FP16).is_err());
    }

    #[test]
    fn caster_supports() {
        let c = TypeCaster::new();
        assert!(c.supports(PrecisionType::FP32, PrecisionType::FP16));
        assert!(c.supports(PrecisionType::FP32, PrecisionType::TF32));
        assert!(c.supports(PrecisionType::FP16, PrecisionType::FP16));
        assert!(!c.supports(PrecisionType::INT8, PrecisionType::FP16));
    }

    #[test]
    fn caster_round_trip_fp16() {
        let c = TypeCaster::new();
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        let err = c.round_trip_error(&data, PrecisionType::FP16).unwrap();
        assert!(err < 0.01);
    }

    #[test]
    fn caster_round_trip_int8() {
        let c = TypeCaster::new();
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.01).collect();
        let err = c.round_trip_error(&data, PrecisionType::INT8).unwrap();
        assert!(err < 0.01);
    }

    #[test]
    fn caster_round_trip_bf16() {
        let c = TypeCaster::new();
        let data = vec![1.0_f32, 100.0, 0.001];
        let err = c.round_trip_error(&data, PrecisionType::BF16).unwrap();
        assert!(err < 1.0);
    }

    #[test]
    fn caster_round_trip_tf32() {
        let c = TypeCaster::new();
        let data = vec![3.14_f32, -2.71, 0.001];
        let err = c.round_trip_error(&data, PrecisionType::TF32).unwrap();
        assert!(err < 0.01);
    }

    // ── LossScaler ───────────────────────────────────────────────────

    #[test]
    fn scaler_initial_scale() {
        let s = LossScaler::new(1024.0);
        assert_eq!(s.scale(), 1024.0);
    }

    #[test]
    fn scaler_scale_gradient() {
        let s = LossScaler::new(256.0);
        assert_eq!(s.scale_gradient(1.0), 256.0);
        assert_eq!(s.scale_gradient(0.5), 128.0);
    }

    #[test]
    fn scaler_unscale_gradient() {
        let s = LossScaler::new(256.0);
        assert_eq!(s.unscale_gradient(256.0), 1.0);
    }

    #[test]
    fn scaler_round_trip() {
        let s = LossScaler::new(1024.0);
        let v = 3.14;
        let restored = s.unscale_gradient(s.scale_gradient(v));
        assert!((v - restored).abs() < 1e-10);
    }

    #[test]
    fn scaler_backoff_on_overflow() {
        let mut s = LossScaler::new(1024.0);
        assert!(!s.update(true));
        assert_eq!(s.scale(), 512.0);
    }

    #[test]
    fn scaler_growth_after_interval() {
        let mut s = LossScaler::new(100.0).with_growth_interval(3).with_growth_factor(2.0);
        for _ in 0..3 {
            assert!(s.update(false));
        }
        assert_eq!(s.scale(), 200.0);
    }

    #[test]
    fn scaler_no_growth_before_interval() {
        let mut s = LossScaler::new(100.0).with_growth_interval(5);
        for _ in 0..4 {
            s.update(false);
        }
        assert_eq!(s.scale(), 100.0);
    }

    #[test]
    fn scaler_min_scale_floor() {
        let mut s = LossScaler::new(2.0);
        s.update(true);
        s.update(true);
        assert_eq!(s.scale(), 1.0);
    }

    #[test]
    fn scaler_overflow_count() {
        let mut s = LossScaler::new(1024.0);
        s.update(true);
        s.update(false);
        s.update(true);
        assert_eq!(s.overflow_count(), 2);
        assert_eq!(s.total_steps(), 3);
    }

    #[test]
    fn scaler_steps_since_overflow() {
        let mut s = LossScaler::new(1024.0);
        s.update(false);
        s.update(false);
        assert_eq!(s.steps_since_overflow(), 2);
        s.update(true);
        assert_eq!(s.steps_since_overflow(), 0);
    }

    #[test]
    fn scaler_builder_chain() {
        let s = LossScaler::new(64.0)
            .with_growth_interval(10)
            .with_growth_factor(4.0)
            .with_backoff_factor(0.25)
            .with_min_scale(0.5);
        assert_eq!(s.scale(), 64.0);
    }

    #[test]
    fn scaler_multiple_backoffs() {
        let mut s = LossScaler::new(1024.0);
        s.update(true);
        s.update(true);
        s.update(true);
        assert_eq!(s.scale(), 128.0);
        assert_eq!(s.overflow_count(), 3);
    }

    // ── OverflowDetector ─────────────────────────────────────────────

    #[test]
    fn overflow_detects_nan() {
        let mut d = OverflowDetector::new();
        assert!(d.check(&[1.0, f64::NAN, 3.0]));
        assert_eq!(d.nan_count(), 1);
    }

    #[test]
    fn overflow_detects_inf() {
        let mut d = OverflowDetector::new();
        assert!(d.check(&[f64::INFINITY]));
        assert!(d.check(&[f64::NEG_INFINITY]));
        assert_eq!(d.inf_count(), 2);
    }

    #[test]
    fn overflow_clean_data() {
        let mut d = OverflowDetector::new();
        assert!(!d.check(&[1.0, -2.0, 0.0, 1e30]));
        assert!(!d.has_detected_anomaly());
    }

    #[test]
    fn overflow_check_f32() {
        let mut d = OverflowDetector::new();
        assert!(d.check_f32(&[f32::NAN]));
        assert!(d.check_f32(&[f32::INFINITY]));
        assert_eq!(d.nan_count(), 1);
        assert_eq!(d.inf_count(), 1);
    }

    #[test]
    fn overflow_total_checks() {
        let mut d = OverflowDetector::new();
        d.check(&[1.0]);
        d.check(&[2.0]);
        d.check_f32(&[3.0]);
        assert_eq!(d.total_checks(), 3);
    }

    #[test]
    fn overflow_reset() {
        let mut d = OverflowDetector::new();
        d.check(&[f64::NAN, f64::INFINITY]);
        d.reset();
        assert_eq!(d.nan_count(), 0);
        assert_eq!(d.inf_count(), 0);
        assert_eq!(d.total_checks(), 0);
        assert!(!d.has_detected_anomaly());
    }

    #[test]
    fn overflow_multiple_nans() {
        let mut d = OverflowDetector::new();
        d.check(&[f64::NAN, f64::NAN, f64::NAN]);
        assert_eq!(d.nan_count(), 3);
    }

    // ── AccumulationBuffer ───────────────────────────────────────────

    #[test]
    fn accum_new_zeroed() {
        let buf = AccumulationBuffer::new(4, PrecisionType::FP32);
        assert_eq!(buf.result(), &[0.0; 4]);
        assert_eq!(buf.len(), 4);
        assert!(!buf.is_empty());
    }

    #[test]
    fn accum_accumulate() {
        let mut buf = AccumulationBuffer::new(3, PrecisionType::FP32);
        buf.accumulate(&[1.0, 2.0, 3.0]);
        buf.accumulate(&[0.5, 0.5, 0.5]);
        assert_eq!(buf.result(), &[1.5, 2.5, 3.5]);
    }

    #[test]
    fn accum_reset() {
        let mut buf = AccumulationBuffer::new(2, PrecisionType::FP32);
        buf.accumulate(&[10.0, 20.0]);
        buf.reset();
        assert_eq!(buf.result(), &[0.0, 0.0]);
    }

    #[test]
    fn accum_dtype() {
        let buf = AccumulationBuffer::new(1, PrecisionType::FP32);
        assert_eq!(buf.dtype(), PrecisionType::FP32);
    }

    #[test]
    fn accum_empty() {
        let buf = AccumulationBuffer::new(0, PrecisionType::FP32);
        assert!(buf.is_empty());
    }

    #[test]
    #[should_panic(expected = "accumulate length mismatch")]
    fn accum_length_mismatch_panics() {
        let mut buf = AccumulationBuffer::new(2, PrecisionType::FP32);
        buf.accumulate(&[1.0, 2.0, 3.0]);
    }

    #[test]
    fn accum_multiple_rounds() {
        let mut buf = AccumulationBuffer::new(2, PrecisionType::FP32);
        for _ in 0..100 {
            buf.accumulate(&[0.01, 0.02]);
        }
        assert!((buf.result()[0] - 1.0).abs() < 1e-10);
        assert!((buf.result()[1] - 2.0).abs() < 1e-10);
    }

    // ── MixedPrecisionMatmul ─────────────────────────────────────────

    #[test]
    fn matmul_default_dtypes() {
        let mm = MixedPrecisionMatmul::new();
        assert_eq!(mm.input_dtype(), PrecisionType::FP16);
        assert_eq!(mm.accumulate_dtype(), PrecisionType::FP32);
        assert_eq!(mm.output_dtype(), PrecisionType::FP16);
    }

    #[test]
    fn matmul_identity_2x2() {
        let mm = MixedPrecisionMatmul::new();
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        let c = mm.matmul(&a, 2, 2, &b, 2).unwrap();
        assert!((c[0] - 5.0).abs() < 0.1);
        assert!((c[1] - 6.0).abs() < 0.1);
        assert!((c[2] - 7.0).abs() < 0.1);
        assert!((c[3] - 8.0).abs() < 0.1);
    }

    #[test]
    fn matmul_1x1() {
        let mm = MixedPrecisionMatmul::new();
        let c = mm.matmul(&[3.0_f32], 1, 1, &[4.0_f32], 1).unwrap();
        assert!((c[0] - 12.0).abs() < 0.1);
    }

    #[test]
    fn matmul_custom_dtypes() {
        let mm = MixedPrecisionMatmul::with_dtypes(
            PrecisionType::BF16,
            PrecisionType::FP32,
            PrecisionType::BF16,
        );
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![1.0_f32, 0.0, 0.0, 1.0];
        let c = mm.matmul(&a, 2, 2, &b, 2).unwrap();
        assert!((c[0] - 1.0).abs() < 0.5);
    }

    #[test]
    fn matmul_dimension_mismatch() {
        let mm = MixedPrecisionMatmul::new();
        assert!(mm.matmul(&[1.0_f32, 2.0], 1, 1, &[3.0], 1).is_err());
    }

    #[test]
    fn matmul_rectangular() {
        let mm = MixedPrecisionMatmul::new();
        // (1×3) × (3×1) = (1×1): 1*4 + 2*5 + 3*6 = 32
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        let c = mm.matmul(&a, 1, 3, &b, 1).unwrap();
        assert!((c[0] - 32.0).abs() < 0.5);
    }

    // ── PrecisionAnalyzer ────────────────────────────────────────────

    #[test]
    fn analyzer_small_weights_recommends_reduced() {
        let a = PrecisionAnalyzer::new().with_error_threshold(0.01);
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.001).collect();
        let report = a.analyse_layer("layer0", &data).unwrap();
        assert_ne!(report.recommended, PrecisionType::FP32);
        assert!(report.memory_savings > 0.0);
    }

    #[test]
    fn analyzer_large_weights_stays_fp32() {
        let a = PrecisionAnalyzer::new().with_error_threshold(0.0001);
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 1000.0).collect();
        let report = a.analyse_layer("layer0", &data).unwrap();
        assert!(report.max_error <= 0.0001 || report.recommended == PrecisionType::FP32);
    }

    #[test]
    fn analyzer_empty_input() {
        let a = PrecisionAnalyzer::new();
        assert!(a.analyse_layer("x", &[]).is_err());
    }

    #[test]
    fn analyzer_model() {
        let a = PrecisionAnalyzer::new().with_error_threshold(0.01);
        let d1: Vec<f32> = (0..32).map(|i| i as f32 * 0.001).collect();
        let d2: Vec<f32> = (0..32).map(|i| i as f32 * 0.002).collect();
        let reports = a.analyse_model(&[("l0", &d1), ("l1", &d2)]).unwrap();
        assert_eq!(reports.len(), 2);
    }

    #[test]
    fn analyzer_report_fields() {
        let a = PrecisionAnalyzer::new().with_error_threshold(1.0);
        let data = vec![0.1_f32, 0.2, 0.3];
        let r = a.analyse_layer("test", &data).unwrap();
        assert_eq!(r.layer_name, "test");
        assert!(r.throughput_multiplier >= 1.0);
        assert!(r.mean_error >= 0.0);
        assert!(r.max_error >= r.mean_error);
    }

    // ── MixedPrecisionEngine ─────────────────────────────────────────

    #[test]
    fn engine_default_empty() {
        let e = MixedPrecisionEngine::new(PrecisionPolicy::default());
        assert!(e.layers().is_empty());
        assert!(e.loss_scaler().is_none());
    }

    #[test]
    fn engine_add_layer_attention() {
        let mut e = MixedPrecisionEngine::new(PrecisionPolicy::default());
        e.add_layer("attn0", LayerKind::Attention, 1000);
        assert_eq!(e.layer_precision("attn0"), Some(PrecisionType::FP16));
    }

    #[test]
    fn engine_add_layer_layernorm() {
        let mut e = MixedPrecisionEngine::new(PrecisionPolicy::default());
        e.add_layer("ln0", LayerKind::LayerNorm, 100);
        assert_eq!(e.layer_precision("ln0"), Some(PrecisionType::FP32));
    }

    #[test]
    fn engine_name_override() {
        let mut pol = PrecisionPolicy::default();
        pol.name_overrides.insert("special".to_string(), PrecisionType::INT4);
        let mut e = MixedPrecisionEngine::new(pol);
        e.add_layer("normal", LayerKind::Other, 100);
        e.add_layer("special", LayerKind::Other, 100);
        assert_eq!(e.layer_precision("special"), Some(PrecisionType::INT4));
    }

    #[test]
    fn engine_total_params() {
        let mut e = MixedPrecisionEngine::new(PrecisionPolicy::default());
        e.add_layer("a", LayerKind::Attention, 1000);
        e.add_layer("b", LayerKind::FeedForward, 2000);
        assert_eq!(e.total_params(), 3000);
    }

    #[test]
    fn engine_memory_fp32_baseline() {
        let mut pol = PrecisionPolicy::default();
        pol.default_config.compute_dtype = PrecisionType::FP32;
        pol.layer_rules.clear();
        let mut e = MixedPrecisionEngine::new(pol);
        e.add_layer("a", LayerKind::Attention, 1000);
        assert_eq!(e.estimated_memory_bytes(), 4000);
    }

    #[test]
    fn engine_memory_savings() {
        let mut e = MixedPrecisionEngine::new(PrecisionPolicy::default());
        e.add_layer("attn", LayerKind::Attention, 1000);
        let ratio = e.memory_savings_ratio();
        assert!(ratio > 0.0);
    }

    #[test]
    fn engine_memory_at_precision() {
        let mut e = MixedPrecisionEngine::new(PrecisionPolicy::default());
        e.add_layer("a", LayerKind::Attention, 500);
        e.add_layer("b", LayerKind::FeedForward, 500);
        assert_eq!(e.memory_at_precision(PrecisionType::FP32), 4000);
        assert_eq!(e.memory_at_precision(PrecisionType::FP16), 2000);
    }

    #[test]
    fn engine_loss_scaling() {
        let mut e = MixedPrecisionEngine::new(PrecisionPolicy::default());
        e.enable_loss_scaling(1024.0);
        assert!(e.loss_scaler().is_some());
        assert_eq!(e.loss_scaler().unwrap().scale(), 1024.0);
    }

    #[test]
    fn engine_overflow_detector() {
        let e = MixedPrecisionEngine::new(PrecisionPolicy::default());
        assert!(!e.overflow_detector().has_detected_anomaly());
    }

    #[test]
    fn engine_forward_step() {
        let mut e = MixedPrecisionEngine::new(PrecisionPolicy::default());
        e.add_layer("attn", LayerKind::Attention, 100);
        let r = e.forward_step("attn", &[1.0_f32, 2.0, 3.0]).unwrap();
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn engine_forward_step_unknown_layer() {
        let mut e = MixedPrecisionEngine::new(PrecisionPolicy::default());
        let r = e.forward_step("missing", &[1.0_f32]).unwrap();
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn engine_forward_with_loss_scaler() {
        let mut e = MixedPrecisionEngine::new(PrecisionPolicy::default());
        e.enable_loss_scaling(512.0);
        e.add_layer("a", LayerKind::Attention, 10);
        let _ = e.forward_step("a", &[1.0_f32, 2.0]).unwrap();
        assert_eq!(e.loss_scaler().unwrap().total_steps(), 1);
    }

    #[test]
    fn engine_policy_ref() {
        let e = MixedPrecisionEngine::new(PrecisionPolicy::default());
        assert!(e.policy().layer_rules.contains_key(&LayerKind::Attention));
    }

    // ── proptest ─────────────────────────────────────────────────────

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn fp16_clamped(v in -1e6_f32..1e6) {
                let c = TypeCaster::new();
                let r = c.cast(
                    &[v],
                    PrecisionType::FP32,
                    PrecisionType::FP16,
                ).unwrap();
                prop_assert!(r[0].abs() <= 65504.0 + 0.01);
            }

            #[test]
            fn int8_in_range(
                values in proptest::collection::vec(
                    -1000.0_f32..1000.0, 1..100
                )
            ) {
                let c = TypeCaster::new();
                let r = c.cast(
                    &values,
                    PrecisionType::FP32,
                    PrecisionType::INT8,
                ).unwrap();
                let absmax = values.iter()
                    .map(|v| v.abs())
                    .fold(0.0_f32, f32::max);
                for &v in &r {
                    prop_assert!(
                        v.abs() <= f64::from(absmax) + 0.01,
                        "INT8 value {v} out of range"
                    );
                }
            }

            #[test]
            fn fp8_e4m3_clamped(v in -1e6_f32..1e6) {
                let c = TypeCaster::new();
                let r = c.cast(
                    &[v],
                    PrecisionType::FP32,
                    PrecisionType::FP8E4M3,
                ).unwrap();
                prop_assert!(r[0].abs() <= 448.0 + 0.2);
            }

            #[test]
            fn accumulation_preserves_sum(
                values in proptest::collection::vec(
                    -100.0_f64..100.0, 1..50
                )
            ) {
                let n = values.len();
                let mut buf = AccumulationBuffer::new(
                    n, PrecisionType::FP32,
                );
                buf.accumulate(&values);
                let expected: f64 = values.iter().sum();
                let actual: f64 =
                    buf.result().iter().sum();
                prop_assert!(
                    (expected - actual).abs() < 1e-10
                );
            }
        }
    }
}
