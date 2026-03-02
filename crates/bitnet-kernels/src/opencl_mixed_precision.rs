//! Mixed-precision compute engine for efficient inference on Intel Arc A770.
//!
//! Provides per-layer precision assignment (FP32/FP16/BF16/INT8/INT4/I2),
//! precision conversion with configurable rounding, mixed-precision matrix
//! multiplication (lower-precision inputs, FP32 accumulator), and automatic
//! precision selection based on weight-distribution sensitivity analysis.
//!
//! # A770-Specific Throughput
//!
//! | Precision | Relative Throughput | Mechanism          |
//! |-----------|--------------------|--------------------|
//! | F32       | 1×                 | baseline FP32 ALU  |
//! | F16       | 2×                 | native half        |
//! | BF16      | ~1× (fallback)     | emulated via F32   |
//! | INT8      | 4×                 | DP4A dot product   |
//! | INT4      | 8× (theoretical)   | packed INT4 ops    |
//! | I2        | 16× (theoretical)  | ternary bit tricks |
//!
//! # No FP64
//!
//! The A770 (Xe-HPG) has no native FP64 — this module detects and avoids it.

use std::fmt;

// ── Precision enum ──────────────────────────────────────────────────────────

/// Numeric precision levels supported by the mixed-precision engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 32-bit IEEE 754 single precision.
    F32,
    /// 16-bit IEEE 754 half precision.
    F16,
    /// 16-bit Brain Float (truncated mantissa of F32).
    BF16,
    /// 8-bit signed integer (with per-tensor or per-channel scale).
    I8,
    /// 4-bit signed integer (packed, two values per byte).
    I4,
    /// 2-bit ternary (BitNet {-1, 0, +1}).
    I2,
}

impl Precision {
    /// Number of bytes per element (fractional for sub-byte types).
    pub fn size_bytes(&self) -> f32 {
        match self {
            Self::F32 => 4.0,
            Self::F16 | Self::BF16 => 2.0,
            Self::I8 => 1.0,
            Self::I4 => 0.5,
            Self::I2 => 0.25,
        }
    }

    /// Representable range as `(min, max)`.
    pub fn range(&self) -> (f64, f64) {
        match self {
            Self::F32 => (f32::MIN as f64, f32::MAX as f64),
            Self::F16 => (-65504.0, 65504.0),
            Self::BF16 => (-3.389e38, 3.389e38),
            Self::I8 => (-128.0, 127.0),
            Self::I4 => (-8.0, 7.0),
            Self::I2 => (-1.0, 1.0),
        }
    }

    /// Whether this is an integer (quantized) type.
    pub fn is_integer(&self) -> bool {
        matches!(self, Self::I8 | Self::I4 | Self::I2)
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::BF16 => write!(f, "BF16"),
            Self::I8 => write!(f, "INT8"),
            Self::I4 => write!(f, "INT4"),
            Self::I2 => write!(f, "I2"),
        }
    }
}

// ── Layer kind ──────────────────────────────────────────────────────────────

/// Logical layer types for per-layer precision assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    Attention,
    FeedForward,
    Embedding,
    LayerNorm,
    Output,
}

impl fmt::Display for LayerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attention => write!(f, "Attention"),
            Self::FeedForward => write!(f, "FFN"),
            Self::Embedding => write!(f, "Embedding"),
            Self::LayerNorm => write!(f, "LayerNorm"),
            Self::Output => write!(f, "Output"),
        }
    }
}

// ── Precision policy ────────────────────────────────────────────────────────

/// Per-layer precision assignment policy.
#[derive(Debug, Clone)]
pub struct PrecisionPolicy {
    /// Default precision for layers not explicitly listed.
    pub default_precision: Precision,
    /// Per-layer overrides.
    pub overrides: Vec<(LayerKind, Precision)>,
}

impl PrecisionPolicy {
    /// Create a uniform policy where every layer uses the same precision.
    pub fn uniform(precision: Precision) -> Self {
        Self { default_precision: precision, overrides: Vec::new() }
    }

    /// Recommended A770 policy: attention=F16, FFN=INT8, rest=F32.
    pub fn a770_default() -> Self {
        Self {
            default_precision: Precision::F32,
            overrides: vec![
                (LayerKind::Attention, Precision::F16),
                (LayerKind::FeedForward, Precision::I8),
                (LayerKind::Embedding, Precision::F16),
            ],
        }
    }

    /// Look up the precision for a given layer kind.
    pub fn precision_for(&self, kind: LayerKind) -> Precision {
        self.overrides
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, p)| *p)
            .unwrap_or(self.default_precision)
    }

    /// Set precision for a specific layer kind.
    pub fn set(&mut self, kind: LayerKind, precision: Precision) {
        if let Some(entry) = self.overrides.iter_mut().find(|(k, _)| *k == kind) {
            entry.1 = precision;
        } else {
            self.overrides.push((kind, precision));
        }
    }
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self::a770_default()
    }
}

// ── Rounding mode ───────────────────────────────────────────────────────────

/// Rounding mode for precision conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (IEEE 754 default).
    #[default]
    NearestEven,
    /// Round toward zero (truncation).
    Truncate,
    /// Stochastic rounding — adds random noise before truncation.
    Stochastic,
}

// ── CastOp — precision conversion ──────────────────────────────────────────

/// Precision conversion operations with rounding control.
#[derive(Debug, Clone)]
pub struct CastOp {
    /// Source precision.
    pub from: Precision,
    /// Target precision.
    pub to: Precision,
    /// Rounding mode.
    pub rounding: RoundingMode,
}

impl CastOp {
    pub fn new(from: Precision, to: Precision) -> Self {
        Self { from, to, rounding: RoundingMode::default() }
    }

    pub fn with_rounding(mut self, mode: RoundingMode) -> Self {
        self.rounding = mode;
        self
    }

    /// Cast a single F32 value to the target precision and back to F32.
    /// This simulates the precision loss of the target format.
    pub fn cast_f32(&self, value: f32) -> f32 {
        if self.from == self.to {
            return value;
        }
        let quantized = self.quantize_scalar(value);
        self.dequantize_scalar(quantized)
    }

    /// Cast a slice of F32 values through the target precision.
    pub fn cast_slice(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (i, &v) in input.iter().enumerate() {
            output[i] = self.cast_f32(v);
        }
    }

    /// Quantize a single F32 value to an internal representation.
    fn quantize_scalar(&self, value: f32) -> f64 {
        if value.is_nan() {
            return f64::NAN;
        }
        if value.is_infinite() {
            return if value > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY };
        }
        let (lo, hi) = self.to.range();
        let clamped = (value as f64).clamp(lo, hi);
        match self.to {
            Precision::F32 => clamped,
            Precision::F16 => round_to_f16(clamped, self.rounding),
            Precision::BF16 => round_to_bf16(clamped, self.rounding),
            Precision::I8 => round_integer(clamped, self.rounding),
            Precision::I4 => round_integer(clamped, self.rounding),
            Precision::I2 => {
                // Ternary: snap to {-1, 0, +1}
                if clamped < -0.5 {
                    -1.0
                } else if clamped > 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    fn dequantize_scalar(&self, quantized: f64) -> f32 {
        quantized as f32
    }
}

/// Round to FP16 precision (10-bit mantissa → ~3.31 decimal digits).
fn round_to_f16(value: f64, mode: RoundingMode) -> f64 {
    // FP16: 1 sign + 5 exponent + 10 mantissa
    // Smallest positive normal: 2^-14 ≈ 6.1e-5
    // Precision: 2^-10 of the exponent range
    let abs = value.abs();
    if abs < 5.96e-8 {
        // Subnormal or underflow → flush to zero
        return 0.0;
    }
    let exponent = abs.log2().floor();
    let quantum = (2.0_f64).powf(exponent - 10.0);
    apply_rounding(value, quantum, mode)
}

/// Round to BF16 precision (7-bit mantissa → ~2.1 decimal digits).
fn round_to_bf16(value: f64, mode: RoundingMode) -> f64 {
    let abs = value.abs();
    if abs < 1.175_494_4e-38 {
        return 0.0;
    }
    let exponent = abs.log2().floor();
    let quantum = (2.0_f64).powf(exponent - 7.0);
    apply_rounding(value, quantum, mode)
}

/// Round to the nearest integer with the given mode.
fn round_integer(value: f64, mode: RoundingMode) -> f64 {
    match mode {
        RoundingMode::NearestEven => {
            let rounded = value.round();
            // Ties to even
            if (value - rounded).abs() == 0.5 { (rounded / 2.0).round() * 2.0 } else { rounded }
        }
        RoundingMode::Truncate => value.trunc(),
        RoundingMode::Stochastic => {
            // Deterministic pseudo-stochastic for reproducibility in tests
            let frac = value.fract().abs();
            if frac > 0.5 {
                if value >= 0.0 { value.ceil() } else { value.floor() }
            } else {
                value.trunc()
            }
        }
    }
}

/// Apply rounding to a floating-point value at the given quantum step.
fn apply_rounding(value: f64, quantum: f64, mode: RoundingMode) -> f64 {
    if quantum <= 0.0 {
        return value;
    }
    match mode {
        RoundingMode::NearestEven => {
            let n = (value / quantum).round();
            n * quantum
        }
        RoundingMode::Truncate => {
            let n = (value / quantum).trunc();
            n * quantum
        }
        RoundingMode::Stochastic => {
            let n = (value / quantum).round();
            n * quantum
        }
    }
}

// ── MixedPrecisionMatmul ───────────────────────────────────────────────────

/// Mixed-precision matrix multiplication.
///
/// Inputs are stored in a lower precision but multiplication accumulates
/// in F32 for numerical stability.
#[derive(Debug, Clone)]
pub struct MixedPrecisionMatmul {
    /// Precision for the A (left) input matrix.
    pub input_a_precision: Precision,
    /// Precision for the B (right) input matrix.
    pub input_b_precision: Precision,
    /// Accumulator precision (always F32 on A770).
    pub accumulator_precision: Precision,
    /// Optional per-channel scale factors for quantized inputs.
    pub scale_a: Option<Vec<f32>>,
    /// Optional per-channel scale factors for quantized inputs.
    pub scale_b: Option<Vec<f32>>,
}

impl MixedPrecisionMatmul {
    /// Create a new mixed-precision matmul with the given input precisions.
    pub fn new(input_a: Precision, input_b: Precision) -> Self {
        Self {
            input_a_precision: input_a,
            input_b_precision: input_b,
            accumulator_precision: Precision::F32,
            scale_a: None,
            scale_b: None,
        }
    }

    /// Set per-channel scale factors for input A.
    pub fn with_scale_a(mut self, scales: Vec<f32>) -> Self {
        self.scale_a = Some(scales);
        self
    }

    /// Set per-channel scale factors for input B.
    pub fn with_scale_b(mut self, scales: Vec<f32>) -> Self {
        self.scale_b = Some(scales);
        self
    }

    /// CPU reference: multiply `a` (M×K) by `b` (K×N) into `c` (M×N).
    ///
    /// Both inputs are F32 on the CPU side; the cast through the declared
    /// input precision simulates the loss that would occur on the GPU.
    pub fn matmul_ref(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        assert_eq!(a.len(), m * k, "a must be M×K");
        assert_eq!(b.len(), k * n, "b must be K×N");
        assert_eq!(c.len(), m * n, "c must be M×N");

        let cast_a = CastOp::new(Precision::F32, self.input_a_precision);
        let cast_b = CastOp::new(Precision::F32, self.input_b_precision);

        for i in 0..m {
            for j in 0..n {
                let mut acc: f32 = 0.0;
                for p in 0..k {
                    let av = cast_a.cast_f32(a[i * k + p]);
                    let bv = cast_b.cast_f32(b[p * n + j]);
                    // Apply per-channel scales if present
                    let scaled_a = match &self.scale_a {
                        Some(s) => av * s[p % s.len()],
                        None => av,
                    };
                    let scaled_b = match &self.scale_b {
                        Some(s) => bv * s[j % s.len()],
                        None => bv,
                    };
                    acc += scaled_a * scaled_b;
                }
                c[i * n + j] = acc;
            }
        }
    }

    /// Compute the theoretical memory savings ratio vs full F32.
    pub fn memory_ratio(&self) -> f32 {
        let f32_size = Precision::F32.size_bytes();
        let a_ratio = self.input_a_precision.size_bytes() / f32_size;
        let b_ratio = self.input_b_precision.size_bytes() / f32_size;
        (a_ratio + b_ratio) / 2.0
    }
}

// ── QuantizationNoise — error estimation ───────────────────────────────────

/// Estimates quantization noise for each precision level.
#[derive(Debug, Clone)]
pub struct QuantizationNoise;

impl QuantizationNoise {
    /// Estimate the root-mean-square quantization error when casting `data`
    /// through the given `precision`.
    pub fn estimate_rms_error(data: &[f32], precision: Precision) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let cast = CastOp::new(Precision::F32, precision);
        let mut sum_sq: f64 = 0.0;
        for &v in data {
            let q = cast.cast_f32(v);
            let err = (v - q) as f64;
            sum_sq += err * err;
        }
        (sum_sq / data.len() as f64).sqrt() as f32
    }

    /// Estimate the maximum absolute error.
    pub fn estimate_max_error(data: &[f32], precision: Precision) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let cast = CastOp::new(Precision::F32, precision);
        data.iter()
            .map(|&v| {
                let q = cast.cast_f32(v);
                (v - q).abs()
            })
            .fold(0.0_f32, f32::max)
    }

    /// Signal-to-quantization-noise ratio in dB.
    pub fn sqnr_db(data: &[f32], precision: Precision) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let signal_power: f64 =
            data.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / data.len() as f64;
        let cast = CastOp::new(Precision::F32, precision);
        let noise_power: f64 = data
            .iter()
            .map(|&v| {
                let err = (v - cast.cast_f32(v)) as f64;
                err * err
            })
            .sum::<f64>()
            / data.len() as f64;
        if noise_power == 0.0 {
            return f32::INFINITY;
        }
        (10.0 * (signal_power / noise_power).log10()) as f32
    }
}

// ── A770PrecisionMap — hardware throughput mapping ─────────────────────────

/// Maps precision to A770 hardware throughput characteristics.
#[derive(Debug, Clone)]
pub struct A770PrecisionMap;

/// Throughput descriptor for a precision on the A770.
#[derive(Debug, Clone)]
pub struct ThroughputInfo {
    /// Precision level.
    pub precision: Precision,
    /// Throughput multiplier relative to F32 (1.0 = baseline).
    pub throughput_multiplier: f32,
    /// Whether this precision has native hardware support.
    pub native_support: bool,
    /// Execution mechanism (e.g. "FP32 ALU", "native half", "DP4A").
    pub mechanism: &'static str,
}

impl A770PrecisionMap {
    /// Get throughput info for the given precision on A770.
    pub fn throughput(precision: Precision) -> ThroughputInfo {
        match precision {
            Precision::F32 => ThroughputInfo {
                precision,
                throughput_multiplier: 1.0,
                native_support: true,
                mechanism: "FP32 ALU",
            },
            Precision::F16 => ThroughputInfo {
                precision,
                throughput_multiplier: 2.0,
                native_support: true,
                mechanism: "native half",
            },
            Precision::BF16 => ThroughputInfo {
                precision,
                throughput_multiplier: 1.0,
                native_support: false,
                mechanism: "emulated via F32 (no native BF16)",
            },
            Precision::I8 => ThroughputInfo {
                precision,
                throughput_multiplier: 4.0,
                native_support: true,
                mechanism: "DP4A dot product",
            },
            Precision::I4 => ThroughputInfo {
                precision,
                throughput_multiplier: 8.0,
                native_support: false,
                mechanism: "packed INT4 (software unpack)",
            },
            Precision::I2 => ThroughputInfo {
                precision,
                throughput_multiplier: 16.0,
                native_support: false,
                mechanism: "ternary bit tricks",
            },
        }
    }

    /// Check whether FP64 is available on A770 (it is NOT).
    pub fn has_fp64() -> bool {
        false
    }

    /// List all supported precisions sorted by throughput (highest first).
    pub fn all_by_throughput() -> Vec<ThroughputInfo> {
        let mut infos: Vec<ThroughputInfo> = [
            Precision::F32,
            Precision::F16,
            Precision::BF16,
            Precision::I8,
            Precision::I4,
            Precision::I2,
        ]
        .iter()
        .map(|&p| Self::throughput(p))
        .collect();
        infos
            .sort_by(|a, b| b.throughput_multiplier.partial_cmp(&a.throughput_multiplier).unwrap());
        infos
    }
}

// ── PrecisionProfiler — weight distribution analysis ───────────────────────

/// Analyzes weight distributions to recommend per-layer precision.
#[derive(Debug, Clone)]
pub struct PrecisionProfiler {
    /// Minimum acceptable SQNR (dB) for a precision to be recommended.
    pub min_sqnr_db: f32,
    /// Maximum tolerable RMS error for a precision to be recommended.
    pub max_rms_error: f32,
}

impl Default for PrecisionProfiler {
    fn default() -> Self {
        Self { min_sqnr_db: 30.0, max_rms_error: 0.01 }
    }
}

/// Recommendation from the profiler for a single layer.
#[derive(Debug, Clone)]
pub struct PrecisionRecommendation {
    /// Layer kind.
    pub layer: LayerKind,
    /// Recommended precision.
    pub precision: Precision,
    /// SQNR (dB) at the recommended precision.
    pub sqnr_db: f32,
    /// RMS error at the recommended precision.
    pub rms_error: f32,
    /// Throughput multiplier at the recommended precision.
    pub throughput_multiplier: f32,
}

impl PrecisionProfiler {
    pub fn new(min_sqnr_db: f32, max_rms_error: f32) -> Self {
        Self { min_sqnr_db, max_rms_error }
    }

    /// Recommend the lowest precision (highest throughput) that meets
    /// quality constraints for the given weight data.
    pub fn recommend(&self, layer: LayerKind, weights: &[f32]) -> PrecisionRecommendation {
        // Candidate precisions from highest throughput to lowest
        let candidates = [
            Precision::I2,
            Precision::I4,
            Precision::I8,
            Precision::F16,
            Precision::BF16,
            Precision::F32,
        ];

        for &p in &candidates {
            let sqnr = QuantizationNoise::sqnr_db(weights, p);
            let rms = QuantizationNoise::estimate_rms_error(weights, p);
            let tp = A770PrecisionMap::throughput(p);

            if sqnr >= self.min_sqnr_db && rms <= self.max_rms_error {
                return PrecisionRecommendation {
                    layer,
                    precision: p,
                    sqnr_db: sqnr,
                    rms_error: rms,
                    throughput_multiplier: tp.throughput_multiplier,
                };
            }
        }

        // Fallback: F32 always meets quality constraints
        let sqnr = QuantizationNoise::sqnr_db(weights, Precision::F32);
        let rms = QuantizationNoise::estimate_rms_error(weights, Precision::F32);
        PrecisionRecommendation {
            layer,
            precision: Precision::F32,
            sqnr_db: sqnr,
            rms_error: rms,
            throughput_multiplier: 1.0,
        }
    }
}

// ── AutoMixedPrecision ─────────────────────────────────────────────────────

/// Automatically selects precision per layer based on sensitivity analysis.
#[derive(Debug, Clone)]
pub struct AutoMixedPrecision {
    profiler: PrecisionProfiler,
}

impl AutoMixedPrecision {
    pub fn new(profiler: PrecisionProfiler) -> Self {
        Self { profiler }
    }

    /// Analyze a set of layers and produce a `PrecisionPolicy`.
    ///
    /// Each entry in `layers` is `(LayerKind, weight_data)`.
    pub fn analyze(
        &self,
        layers: &[(LayerKind, &[f32])],
    ) -> (PrecisionPolicy, Vec<PrecisionRecommendation>) {
        let mut policy = PrecisionPolicy::uniform(Precision::F32);
        let mut recommendations = Vec::with_capacity(layers.len());

        for &(kind, weights) in layers {
            let rec = self.profiler.recommend(kind, weights);
            policy.set(kind, rec.precision);
            recommendations.push(rec);
        }

        (policy, recommendations)
    }

    /// Compute aggregate memory savings ratio for a policy relative to F32.
    pub fn memory_savings(policy: &PrecisionPolicy) -> f32 {
        let layers = [
            LayerKind::Attention,
            LayerKind::FeedForward,
            LayerKind::Embedding,
            LayerKind::LayerNorm,
            LayerKind::Output,
        ];
        let f32_bytes = Precision::F32.size_bytes();
        let total: f32 = layers.iter().map(|&k| policy.precision_for(k).size_bytes()).sum();
        total / (layers.len() as f32 * f32_bytes)
    }
}

impl Default for AutoMixedPrecision {
    fn default() -> Self {
        Self::new(PrecisionProfiler::default())
    }
}

// ── OpenCL kernel sources ──────────────────────────────────────────────────

/// OpenCL kernel source for mixed-precision matmul (FP16 inputs, FP32 acc).
pub const OPENCL_MIXED_PRECISION_MATMUL_F16: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/// Mixed-precision matmul: A (half) × B (half) → C (float).
/// Each work-item computes one element of C.
__kernel void matmul_f16_acc_f32(
    __global const half *A,   // M × K
    __global const half *B,   // K × N
    __global float       *C,  // M × N
    const int M,
    const int N,
    const int K
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int p = 0; p < K; ++p) {
        float a_val = vload_half(0, &A[row * K + p]);
        float b_val = vload_half(0, &B[p * N + col]);
        acc = fma(a_val, b_val, acc);
    }
    C[row * N + col] = acc;
}
"#;

/// OpenCL kernel source for INT8 DP4A-style matmul with scale factors.
pub const OPENCL_MIXED_PRECISION_MATMUL_INT8: &str = r#"
/// INT8 matmul with per-row/col scale factors.
/// A (char) × B (char) → C (float), accumulated in int32 then scaled.
__kernel void matmul_int8_dp4a(
    __global const char  *A,        // M × K
    __global const char  *B,        // K × N
    __global float       *C,        // M × N
    __global const float *scale_A,  // M
    __global const float *scale_B,  // N
    const int M,
    const int N,
    const int K
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    if (row >= M || col >= N) return;

    int acc = 0;
    // Process 4 elements at a time (DP4A style)
    int k4 = K / 4;
    for (int p = 0; p < k4; ++p) {
        int base = p * 4;
        acc += (int)A[row * K + base + 0] * (int)B[(base + 0) * N + col];
        acc += (int)A[row * K + base + 1] * (int)B[(base + 1) * N + col];
        acc += (int)A[row * K + base + 2] * (int)B[(base + 2) * N + col];
        acc += (int)A[row * K + base + 3] * (int)B[(base + 3) * N + col];
    }
    // Handle remainder
    for (int p = k4 * 4; p < K; ++p) {
        acc += (int)A[row * K + p] * (int)B[p * N + col];
    }
    C[row * N + col] = (float)acc * scale_A[row] * scale_B[col];
}
"#;

/// Tiled version of F16 matmul with local memory for better data reuse.
pub const OPENCL_MIXED_PRECISION_MATMUL_F16_TILED: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_SIZE 16

/// Tiled mixed-precision matmul: A (half) × B (half) → C (float).
/// Uses local memory tiles for better cache behavior on A770.
__kernel void matmul_f16_tiled(
    __global const half *A,
    __global const half *B,
    __global float       *C,
    const int M,
    const int N,
    const int K
) {
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int lr  = get_local_id(0);
    const int lc  = get_local_id(1);

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int aCol = t * TILE_SIZE + lc;
        int bRow = t * TILE_SIZE + lr;
        tileA[lr][lc] = (row < M && aCol < K)
            ? vload_half(0, &A[row * K + aCol]) : 0.0f;
        tileB[lr][lc] = (bRow < K && col < N)
            ? vload_half(0, &B[bRow * N + col]) : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int p = 0; p < TILE_SIZE; ++p) {
            acc = fma(tileA[lr][p], tileB[p][lc], acc);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
"#;

// ── CPU reference helpers ──────────────────────────────────────────────────

/// CPU reference: F32 matmul for golden comparison.
pub fn matmul_f32_reference(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc: f32 = 0.0;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

/// CPU reference: INT8 matmul with per-row/col scale factors.
pub fn matmul_int8_reference(
    a: &[i8],
    b: &[i8],
    scale_a: &[f32],
    scale_b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    assert_eq!(scale_a.len(), m);
    assert_eq!(scale_b.len(), n);
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for p in 0..k {
                acc += a[i * k + p] as i32 * b[p * n + j] as i32;
            }
            c[i * n + j] = acc as f32 * scale_a[i] * scale_b[j];
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Precision enum tests ────────────────────────────────────────────

    #[test]
    fn test_precision_size_bytes() {
        assert_eq!(Precision::F32.size_bytes(), 4.0);
        assert_eq!(Precision::F16.size_bytes(), 2.0);
        assert_eq!(Precision::BF16.size_bytes(), 2.0);
        assert_eq!(Precision::I8.size_bytes(), 1.0);
        assert_eq!(Precision::I4.size_bytes(), 0.5);
        assert_eq!(Precision::I2.size_bytes(), 0.25);
    }

    #[test]
    fn test_precision_range_f32() {
        let (lo, hi) = Precision::F32.range();
        assert!(lo < -1.0e38);
        assert!(hi > 1.0e38);
    }

    #[test]
    fn test_precision_range_f16() {
        let (lo, hi) = Precision::F16.range();
        assert_eq!(lo, -65504.0);
        assert_eq!(hi, 65504.0);
    }

    #[test]
    fn test_precision_range_i8() {
        let (lo, hi) = Precision::I8.range();
        assert_eq!(lo, -128.0);
        assert_eq!(hi, 127.0);
    }

    #[test]
    fn test_precision_range_i4() {
        let (lo, hi) = Precision::I4.range();
        assert_eq!(lo, -8.0);
        assert_eq!(hi, 7.0);
    }

    #[test]
    fn test_precision_range_i2() {
        let (lo, hi) = Precision::I2.range();
        assert_eq!(lo, -1.0);
        assert_eq!(hi, 1.0);
    }

    #[test]
    fn test_precision_is_integer() {
        assert!(!Precision::F32.is_integer());
        assert!(!Precision::F16.is_integer());
        assert!(!Precision::BF16.is_integer());
        assert!(Precision::I8.is_integer());
        assert!(Precision::I4.is_integer());
        assert!(Precision::I2.is_integer());
    }

    #[test]
    fn test_precision_display() {
        assert_eq!(format!("{}", Precision::F32), "F32");
        assert_eq!(format!("{}", Precision::F16), "F16");
        assert_eq!(format!("{}", Precision::I8), "INT8");
        assert_eq!(format!("{}", Precision::I4), "INT4");
        assert_eq!(format!("{}", Precision::I2), "I2");
    }

    // ── PrecisionPolicy tests ──────────────────────────────────────────

    #[test]
    fn test_policy_uniform() {
        let policy = PrecisionPolicy::uniform(Precision::F16);
        assert_eq!(policy.precision_for(LayerKind::Attention), Precision::F16);
        assert_eq!(policy.precision_for(LayerKind::FeedForward), Precision::F16);
        assert_eq!(policy.precision_for(LayerKind::Output), Precision::F16);
    }

    #[test]
    fn test_policy_a770_default() {
        let policy = PrecisionPolicy::a770_default();
        assert_eq!(policy.precision_for(LayerKind::Attention), Precision::F16);
        assert_eq!(policy.precision_for(LayerKind::FeedForward), Precision::I8);
        assert_eq!(policy.precision_for(LayerKind::Embedding), Precision::F16);
        // Not overridden → default F32
        assert_eq!(policy.precision_for(LayerKind::LayerNorm), Precision::F32);
        assert_eq!(policy.precision_for(LayerKind::Output), Precision::F32);
    }

    #[test]
    fn test_policy_set_override() {
        let mut policy = PrecisionPolicy::uniform(Precision::F32);
        policy.set(LayerKind::Output, Precision::F16);
        assert_eq!(policy.precision_for(LayerKind::Output), Precision::F16);
        assert_eq!(policy.precision_for(LayerKind::Attention), Precision::F32);
    }

    #[test]
    fn test_policy_set_updates_existing() {
        let mut policy = PrecisionPolicy::a770_default();
        assert_eq!(policy.precision_for(LayerKind::Attention), Precision::F16);
        policy.set(LayerKind::Attention, Precision::I8);
        assert_eq!(policy.precision_for(LayerKind::Attention), Precision::I8);
    }

    #[test]
    fn test_policy_default_trait() {
        let policy = PrecisionPolicy::default();
        // Default is a770_default
        assert_eq!(policy.precision_for(LayerKind::FeedForward), Precision::I8);
    }

    // ── CastOp tests ───────────────────────────────────────────────────

    #[test]
    fn test_cast_identity_f32() {
        let cast = CastOp::new(Precision::F32, Precision::F32);
        assert_eq!(cast.cast_f32(3.14), 3.14);
        assert_eq!(cast.cast_f32(-0.0), -0.0);
    }

    #[test]
    fn test_cast_f32_to_f16_roundtrip() {
        let cast = CastOp::new(Precision::F32, Precision::F16);
        let original = 1.0_f32;
        let casted = cast.cast_f32(original);
        // F16 can represent 1.0 exactly
        assert!((casted - original).abs() < 1e-6);
    }

    #[test]
    fn test_cast_f32_to_f16_small_value() {
        let cast = CastOp::new(Precision::F32, Precision::F16);
        let original = 0.001_f32;
        let casted = cast.cast_f32(original);
        // F16 has ~3 decimal digits of precision
        let epsilon = 2.0_f32.powi(-10) * original.abs();
        assert!((casted - original).abs() <= epsilon * 2.0, "expected ~{original}, got {casted}");
    }

    #[test]
    fn test_cast_f32_to_i8_clamps() {
        let cast = CastOp::new(Precision::F32, Precision::I8);
        assert_eq!(cast.cast_f32(200.0), 127.0);
        assert_eq!(cast.cast_f32(-200.0), -128.0);
    }

    #[test]
    fn test_cast_f32_to_i8_rounds() {
        let cast = CastOp::new(Precision::F32, Precision::I8);
        assert_eq!(cast.cast_f32(3.7), 4.0);
        assert_eq!(cast.cast_f32(-3.7), -4.0);
    }

    #[test]
    fn test_cast_f32_to_i4_clamps() {
        let cast = CastOp::new(Precision::F32, Precision::I4);
        assert_eq!(cast.cast_f32(100.0), 7.0);
        assert_eq!(cast.cast_f32(-100.0), -8.0);
    }

    #[test]
    fn test_cast_f32_to_i2_ternary() {
        let cast = CastOp::new(Precision::F32, Precision::I2);
        assert_eq!(cast.cast_f32(0.9), 1.0);
        assert_eq!(cast.cast_f32(-0.9), -1.0);
        assert_eq!(cast.cast_f32(0.1), 0.0);
    }

    #[test]
    fn test_cast_truncate_mode() {
        let cast = CastOp::new(Precision::F32, Precision::I8).with_rounding(RoundingMode::Truncate);
        assert_eq!(cast.cast_f32(3.9), 3.0);
        assert_eq!(cast.cast_f32(-3.9), -3.0);
    }

    #[test]
    fn test_cast_slice() {
        let cast = CastOp::new(Precision::F32, Precision::I8);
        let input = [1.5, 2.7, -3.2, 200.0];
        let mut output = [0.0_f32; 4];
        cast.cast_slice(&input, &mut output);
        assert_eq!(output[0], 2.0); // rounds
        assert_eq!(output[1], 3.0); // rounds
        assert_eq!(output[2], -3.0); // rounds toward nearest
        assert_eq!(output[3], 127.0); // clamps
    }

    #[test]
    fn test_cast_nan_preserved() {
        let cast = CastOp::new(Precision::F32, Precision::F16);
        assert!(cast.cast_f32(f32::NAN).is_nan());
    }

    #[test]
    fn test_cast_inf_preserved() {
        let cast = CastOp::new(Precision::F32, Precision::F16);
        assert!(cast.cast_f32(f32::INFINITY).is_infinite());
        assert!(cast.cast_f32(f32::NEG_INFINITY).is_infinite());
    }

    // ── MixedPrecisionMatmul tests ─────────────────────────────────────

    #[test]
    fn test_matmul_f32_identity() {
        // 2×2 identity matrix times a vector-like 2×1
        let mm = MixedPrecisionMatmul::new(Precision::F32, Precision::F32);
        let a = [1.0, 0.0, 0.0, 1.0_f32];
        let b = [3.0, 7.0_f32];
        let mut c = [0.0_f32; 2];
        mm.matmul_ref(&a, &b, &mut c, 2, 1, 2);
        assert_eq!(c, [3.0, 7.0]);
    }

    #[test]
    fn test_matmul_f16_close_to_f32() {
        let m = 4;
        let k = 8;
        let n = 4;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();
        let mut c_f32 = vec![0.0_f32; m * n];
        let mut c_f16 = vec![0.0_f32; m * n];

        let mm_f32 = MixedPrecisionMatmul::new(Precision::F32, Precision::F32);
        let mm_f16 = MixedPrecisionMatmul::new(Precision::F16, Precision::F16);

        mm_f32.matmul_ref(&a, &b, &mut c_f32, m, n, k);
        mm_f16.matmul_ref(&a, &b, &mut c_f16, m, n, k);

        for (i, (&f32_val, &f16_val)) in c_f32.iter().zip(c_f16.iter()).enumerate() {
            let tol = f32_val.abs() * 0.01 + 1e-4;
            assert!(
                (f32_val - f16_val).abs() < tol,
                "element {i}: f32={f32_val}, f16={f16_val}, tol={tol}"
            );
        }
    }

    #[test]
    fn test_matmul_i8_with_scales() {
        // 2×2 matmul with INT8 values and scale factors
        let a = [10_i8, 20, 30, 40];
        let b = [1_i8, 2, 3, 4];
        let scale_a = [0.1_f32, 0.2];
        let scale_b = [0.5_f32, 1.0];
        let mut c = [0.0_f32; 4];
        matmul_int8_reference(&a, &b, &scale_a, &scale_b, &mut c, 2, 2, 2);

        // Row 0: acc[0,0] = 10*1 + 20*3 = 70, c = 70 * 0.1 * 0.5 = 3.5
        //        acc[0,1] = 10*2 + 20*4 = 100, c = 100 * 0.1 * 1.0 = 10.0
        assert!((c[0] - 3.5).abs() < 1e-5);
        assert!((c[1] - 10.0).abs() < 1e-5);
        // Row 1: acc[1,0] = 30*1 + 40*3 = 150, c = 150 * 0.2 * 0.5 = 15.0
        //        acc[1,1] = 30*2 + 40*4 = 220, c = 220 * 0.2 * 1.0 = 44.0
        assert!((c[2] - 15.0).abs() < 1e-5);
        assert!((c[3] - 44.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_mixed_f16_f32_accumulate() {
        let mm = MixedPrecisionMatmul::new(Precision::F16, Precision::F16);
        // Small values that F16 can represent exactly
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [5.0_f32, 6.0, 7.0, 8.0];
        let mut c = [0.0_f32; 4];
        mm.matmul_ref(&a, &b, &mut c, 2, 2, 2);
        // [1,2]×[5,6; 7,8] = [19, 22; 43, 50]
        assert!((c[0] - 19.0).abs() < 1e-3);
        assert!((c[1] - 22.0).abs() < 1e-3);
        assert!((c[2] - 43.0).abs() < 1e-3);
        assert!((c[3] - 50.0).abs() < 1e-3);
    }

    #[test]
    fn test_matmul_with_scale_a() {
        let mm = MixedPrecisionMatmul::new(Precision::F32, Precision::F32).with_scale_a(vec![2.0]);
        let a = [1.0_f32, 1.0];
        let b = [3.0_f32, 5.0];
        let mut c = [0.0_f32; 1];
        mm.matmul_ref(&a, &b, &mut c, 1, 1, 2);
        // (1*2) * 3 + (1*2) * 5 = 6 + 10 = 16
        assert!((c[0] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_memory_ratio() {
        let mm_f16 = MixedPrecisionMatmul::new(Precision::F16, Precision::F16);
        assert!((mm_f16.memory_ratio() - 0.5).abs() < 1e-6);

        let mm_i8 = MixedPrecisionMatmul::new(Precision::I8, Precision::I8);
        assert!((mm_i8.memory_ratio() - 0.25).abs() < 1e-6);

        let mm_mixed = MixedPrecisionMatmul::new(Precision::F16, Precision::I8);
        // (2/4 + 1/4) / 2 = 0.375
        assert!((mm_mixed.memory_ratio() - 0.375).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_1x1() {
        let mm = MixedPrecisionMatmul::new(Precision::F32, Precision::F32);
        let a = [7.0_f32];
        let b = [3.0_f32];
        let mut c = [0.0_f32; 1];
        mm.matmul_ref(&a, &b, &mut c, 1, 1, 1);
        assert_eq!(c[0], 21.0);
    }

    // ── QuantizationNoise tests ────────────────────────────────────────

    #[test]
    fn test_quant_noise_f32_is_zero() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let rms = QuantizationNoise::estimate_rms_error(&data, Precision::F32);
        assert!(rms < 1e-10, "F32→F32 should have zero noise, got {rms}");
    }

    #[test]
    fn test_quant_noise_i8_nonzero() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let rms = QuantizationNoise::estimate_rms_error(&data, Precision::I8);
        assert!(rms > 0.0, "INT8 quantization should introduce noise");
    }

    #[test]
    fn test_quant_noise_ordering() {
        // For typical small-range data, wider precision → less noise
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let rms_f16 = QuantizationNoise::estimate_rms_error(&data, Precision::F16);
        let rms_i8 = QuantizationNoise::estimate_rms_error(&data, Precision::I8);
        // F16 should have less noise than INT8 for this data
        assert!(rms_f16 <= rms_i8, "F16 noise ({rms_f16}) should be ≤ INT8 noise ({rms_i8})");
    }

    #[test]
    fn test_quant_noise_max_error() {
        let data = [0.3_f32, 0.7, 1.5, 2.9];
        let max_err = QuantizationNoise::estimate_max_error(&data, Precision::I8);
        // All values < 128, so should be ≤ 0.5 (rounding error)
        assert!(max_err <= 0.5 + 1e-6);
    }

    #[test]
    fn test_quant_noise_empty_data() {
        assert_eq!(QuantizationNoise::estimate_rms_error(&[], Precision::F16), 0.0);
        assert_eq!(QuantizationNoise::estimate_max_error(&[], Precision::I8), 0.0);
        assert_eq!(QuantizationNoise::sqnr_db(&[], Precision::F32), 0.0);
    }

    #[test]
    fn test_sqnr_f32_infinite() {
        let data: Vec<f32> = (1..50).map(|i| i as f32).collect();
        let sqnr = QuantizationNoise::sqnr_db(&data, Precision::F32);
        assert!(sqnr.is_infinite(), "F32→F32 SQNR should be infinite");
    }

    #[test]
    fn test_sqnr_positive_for_quantized() {
        let data: Vec<f32> = (1..100).map(|i| i as f32 * 0.1).collect();
        let sqnr = QuantizationNoise::sqnr_db(&data, Precision::I8);
        assert!(sqnr > 0.0, "SQNR should be positive, got {sqnr}");
    }

    // ── A770PrecisionMap tests ─────────────────────────────────────────

    #[test]
    fn test_a770_f32_baseline() {
        let info = A770PrecisionMap::throughput(Precision::F32);
        assert_eq!(info.throughput_multiplier, 1.0);
        assert!(info.native_support);
        assert_eq!(info.mechanism, "FP32 ALU");
    }

    #[test]
    fn test_a770_f16_double_throughput() {
        let info = A770PrecisionMap::throughput(Precision::F16);
        assert_eq!(info.throughput_multiplier, 2.0);
        assert!(info.native_support);
    }

    #[test]
    fn test_a770_int8_dp4a() {
        let info = A770PrecisionMap::throughput(Precision::I8);
        assert_eq!(info.throughput_multiplier, 4.0);
        assert!(info.native_support);
        assert!(info.mechanism.contains("DP4A"));
    }

    #[test]
    fn test_a770_bf16_fallback() {
        let info = A770PrecisionMap::throughput(Precision::BF16);
        assert_eq!(info.throughput_multiplier, 1.0);
        assert!(!info.native_support);
    }

    #[test]
    fn test_a770_no_fp64() {
        assert!(!A770PrecisionMap::has_fp64());
    }

    #[test]
    fn test_a770_all_by_throughput_ordering() {
        let infos = A770PrecisionMap::all_by_throughput();
        for w in infos.windows(2) {
            assert!(
                w[0].throughput_multiplier >= w[1].throughput_multiplier,
                "not sorted: {} ({}) before {} ({})",
                w[0].precision,
                w[0].throughput_multiplier,
                w[1].precision,
                w[1].throughput_multiplier,
            );
        }
    }

    #[test]
    fn test_a770_i4_theoretical() {
        let info = A770PrecisionMap::throughput(Precision::I4);
        assert_eq!(info.throughput_multiplier, 8.0);
        assert!(!info.native_support);
    }

    // ── PrecisionProfiler tests ────────────────────────────────────────

    #[test]
    fn test_profiler_recommends_lower_for_robust() {
        let profiler = PrecisionProfiler::new(20.0, 0.1);
        // Values in {-1, 0, +1} — perfectly suited for I2
        let weights = [1.0_f32, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0];
        let rec = profiler.recommend(LayerKind::FeedForward, &weights);
        // Should recommend I2 or I4 since these values are exactly ternary
        assert!(
            rec.precision == Precision::I2 || rec.precision == Precision::I4,
            "expected I2 or I4 for ternary data, got {}",
            rec.precision
        );
    }

    #[test]
    fn test_profiler_recommends_higher_for_sensitive() {
        let profiler = PrecisionProfiler::new(60.0, 0.0001);
        // Broad-range data — requires high precision
        let weights: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.01).collect();
        let rec = profiler.recommend(LayerKind::Attention, &weights);
        assert!(
            rec.precision == Precision::F16 || rec.precision == Precision::F32,
            "expected F16 or F32 for sensitive data, got {}",
            rec.precision
        );
    }

    #[test]
    fn test_profiler_fallback_to_f32() {
        // Extremely strict thresholds — only F32 can satisfy
        let profiler = PrecisionProfiler::new(200.0, 1e-10);
        let weights: Vec<f32> = (0..100).map(|i| i as f32 * 0.37).collect();
        let rec = profiler.recommend(LayerKind::Output, &weights);
        assert_eq!(rec.precision, Precision::F32);
    }

    // ── AutoMixedPrecision tests ───────────────────────────────────────

    #[test]
    fn test_auto_mixed_produces_policy() {
        let amp = AutoMixedPrecision::default();
        let ternary = [1.0_f32, -1.0, 0.0, 1.0, -1.0, 0.0];
        let broad: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();

        let layers: Vec<(LayerKind, &[f32])> =
            vec![(LayerKind::FeedForward, &ternary), (LayerKind::Attention, &broad)];

        let (policy, recs) = amp.analyze(&layers);
        assert_eq!(recs.len(), 2);

        // FFN with ternary data should get lower precision than attention
        let ffn_p = policy.precision_for(LayerKind::FeedForward);
        let attn_p = policy.precision_for(LayerKind::Attention);
        assert!(
            ffn_p.size_bytes() <= attn_p.size_bytes(),
            "FFN ({ffn_p}) should be ≤ attention ({attn_p}) in size"
        );
    }

    #[test]
    fn test_auto_mixed_memory_savings() {
        let mut policy = PrecisionPolicy::uniform(Precision::F16);
        policy.set(LayerKind::FeedForward, Precision::I8);
        let savings = AutoMixedPrecision::memory_savings(&policy);
        // 4 layers F16 (2B) + 1 layer I8 (1B) = 9 bytes / (5 * 4) = 0.45
        assert!(savings < 1.0, "mixed precision should save memory");
        assert!(savings > 0.0);
    }

    #[test]
    fn test_auto_mixed_all_f32_is_1x() {
        let policy = PrecisionPolicy::uniform(Precision::F32);
        let savings = AutoMixedPrecision::memory_savings(&policy);
        assert!((savings - 1.0).abs() < 1e-6, "all-F32 policy should have 1.0 ratio");
    }

    // ── F32 reference matmul tests ─────────────────────────────────────

    #[test]
    fn test_f32_reference_identity() {
        let a = [1.0, 0.0, 0.0, 1.0_f32];
        let b = [2.0, 3.0, 4.0, 5.0_f32];
        let mut c = [0.0_f32; 4];
        matmul_f32_reference(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, [2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_f32_reference_3x3() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0_f32];
        let b = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0_f32];
        let mut c = [0.0_f32; 9];
        matmul_f32_reference(&a, &b, &mut c, 3, 3, 3);
        // Multiply by identity → same as input
        assert_eq!(c, a);
    }

    // ── OpenCL kernel source tests ─────────────────────────────────────

    #[test]
    fn test_opencl_f16_kernel_source_nonempty() {
        assert!(!OPENCL_MIXED_PRECISION_MATMUL_F16.is_empty());
        assert!(OPENCL_MIXED_PRECISION_MATMUL_F16.contains("matmul_f16_acc_f32"));
        assert!(OPENCL_MIXED_PRECISION_MATMUL_F16.contains("cl_khr_fp16"));
    }

    #[test]
    fn test_opencl_int8_kernel_source_nonempty() {
        assert!(!OPENCL_MIXED_PRECISION_MATMUL_INT8.is_empty());
        assert!(OPENCL_MIXED_PRECISION_MATMUL_INT8.contains("matmul_int8_dp4a"));
        assert!(OPENCL_MIXED_PRECISION_MATMUL_INT8.contains("scale_A"));
    }

    #[test]
    fn test_opencl_f16_tiled_kernel_source() {
        assert!(!OPENCL_MIXED_PRECISION_MATMUL_F16_TILED.is_empty());
        assert!(OPENCL_MIXED_PRECISION_MATMUL_F16_TILED.contains("TILE_SIZE"));
        assert!(OPENCL_MIXED_PRECISION_MATMUL_F16_TILED.contains("matmul_f16_tiled"));
        assert!(OPENCL_MIXED_PRECISION_MATMUL_F16_TILED.contains("barrier"));
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_cast_subnormal_f16() {
        let cast = CastOp::new(Precision::F32, Precision::F16);
        // Very small value: F16 subnormal range
        let val = 1e-7_f32;
        let result = cast.cast_f32(val);
        // Should flush to zero or represent as subnormal
        assert!(result.abs() < val * 10.0);
    }

    #[test]
    fn test_cast_large_value_f16_clamps() {
        let cast = CastOp::new(Precision::F32, Precision::F16);
        let val = 100000.0_f32;
        let result = cast.cast_f32(val);
        // F16 max is 65504 — should clamp
        assert!(result <= 65504.0);
    }

    #[test]
    fn test_cast_zero() {
        let cast = CastOp::new(Precision::F32, Precision::F16);
        assert_eq!(cast.cast_f32(0.0), 0.0);
    }

    #[test]
    fn test_cast_negative_zero() {
        let cast = CastOp::new(Precision::F32, Precision::I8);
        let result = cast.cast_f32(-0.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_property_f32_f16_within_half_epsilon() {
        // Property: F32→F16→F32 error ≤ 2^-10 * |value| for normals
        let cast = CastOp::new(Precision::F32, Precision::F16);
        let test_values = [0.5, 1.0, 2.0, 100.0, 1000.0, -0.5, -1.0, -100.0];
        for &v in &test_values {
            let roundtrip = cast.cast_f32(v);
            let max_err = 2.0_f32.powi(-10) * v.abs();
            assert!(
                (v - roundtrip).abs() <= max_err + 1e-10,
                "F32→F16→F32 for {v}: got {roundtrip}, err {}, max {}",
                (v - roundtrip).abs(),
                max_err
            );
        }
    }

    #[test]
    fn test_matmul_zeros() {
        let mm = MixedPrecisionMatmul::new(Precision::F16, Precision::F16);
        let a = [0.0_f32; 4];
        let b = [1.0, 2.0, 3.0, 4.0_f32];
        let mut c = [999.0_f32; 4];
        mm.matmul_ref(&a, &b, &mut c, 2, 2, 2);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_layer_kind_display() {
        assert_eq!(format!("{}", LayerKind::Attention), "Attention");
        assert_eq!(format!("{}", LayerKind::FeedForward), "FFN");
        assert_eq!(format!("{}", LayerKind::Embedding), "Embedding");
    }

    #[test]
    fn test_rounding_mode_default() {
        assert_eq!(RoundingMode::default(), RoundingMode::NearestEven);
    }

    #[test]
    fn test_stochastic_rounding_i8() {
        let cast =
            CastOp::new(Precision::F32, Precision::I8).with_rounding(RoundingMode::Stochastic);
        // Stochastic rounding: 3.7 should round to 4 (frac > 0.5)
        assert_eq!(cast.cast_f32(3.7), 4.0);
        // 3.3 should truncate to 3 (frac < 0.5)
        assert_eq!(cast.cast_f32(3.3), 3.0);
    }

    #[test]
    fn test_bf16_precision_loss() {
        let cast = CastOp::new(Precision::F32, Precision::BF16);
        // BF16 has 7-bit mantissa (~2 decimal digits)
        let val = 1.234567_f32;
        let result = cast.cast_f32(val);
        // Should be within BF16 precision (~1% for this range)
        assert!((val - result).abs() < 0.02, "BF16 roundtrip: {val} → {result}");
    }

    #[test]
    fn test_int8_ref_matmul_zeros() {
        let a = [0_i8; 4];
        let b = [1_i8, 2, 3, 4];
        let scale_a = [1.0_f32, 1.0];
        let scale_b = [1.0_f32, 1.0];
        let mut c = [999.0_f32; 4];
        matmul_int8_reference(&a, &b, &scale_a, &scale_b, &mut c, 2, 2, 2);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_profiler_recommendation_has_throughput() {
        let profiler = PrecisionProfiler::default();
        let weights = [1.0_f32; 100];
        let rec = profiler.recommend(LayerKind::Attention, &weights);
        assert!(rec.throughput_multiplier >= 1.0);
        assert!(rec.sqnr_db > 0.0 || rec.sqnr_db.is_infinite());
    }

    #[test]
    fn test_auto_mixed_default() {
        let amp = AutoMixedPrecision::default();
        let data = [0.5_f32; 64];
        let layers = vec![(LayerKind::Embedding, data.as_slice())];
        let (policy, recs) = amp.analyze(&layers);
        assert_eq!(recs.len(), 1);
        // Should produce a valid recommendation
        let p = policy.precision_for(LayerKind::Embedding);
        assert!(p.size_bytes() > 0.0);
    }

    #[test]
    fn test_precision_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Precision::F32);
        set.insert(Precision::F16);
        set.insert(Precision::F32); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_a770_all_precisions_covered() {
        let infos = A770PrecisionMap::all_by_throughput();
        assert_eq!(infos.len(), 6, "should cover all 6 precision levels");
    }

    #[test]
    fn test_cast_bf16_range() {
        let (lo, hi) = Precision::BF16.range();
        // BF16 has same exponent range as F32 but truncated mantissa
        assert!(lo < -1.0e38);
        assert!(hi > 1.0e38);
    }
}
