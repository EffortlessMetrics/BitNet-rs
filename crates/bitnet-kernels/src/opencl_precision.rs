//! Mixed-precision compute support for A770 OpenCL inference.
//!
//! Intel Arc A770 supports FP16 compute at double throughput versus FP32
//! but may need FP32 accumulation for numerical stability. This module
//! handles precision management, CPU-side FP16↔FP32 conversion, and
//! provides OpenCL kernel helper sources for mixed-precision patterns.

use std::fmt;

// ---------------------------------------------------------------------------
// ComputePrecision
// ---------------------------------------------------------------------------

/// Precision mode for an individual compute operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputePrecision {
    /// Full 32-bit floating-point (safe, slower).
    Float32,
    /// Half 16-bit floating-point (fast — A770 has 2× FP16 throughput).
    Float16,
    /// FP16 compute with FP32 accumulation (best balance).
    Mixed,
    /// INT8 with DP4A dot product (A770 supports this).
    Int8,
}

impl ComputePrecision {
    /// Number of bytes per scalar element in this precision.
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 | Self::Mixed => 2,
            Self::Int8 => 1,
        }
    }

    /// Theoretical throughput speed-up relative to FP32 on A770 hardware.
    pub fn theoretical_speedup_vs_f32(&self) -> f64 {
        match self {
            Self::Float32 => 1.0,
            Self::Float16 => 2.0,
            Self::Mixed => 1.8,
            Self::Int8 => 4.0,
        }
    }
}

impl fmt::Display for ComputePrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float32 => write!(f, "FP32"),
            Self::Float16 => write!(f, "FP16"),
            Self::Mixed => write!(f, "Mixed (FP16 compute / FP32 accum)"),
            Self::Int8 => write!(f, "INT8 (DP4A)"),
        }
    }
}

// ---------------------------------------------------------------------------
// PrecisionPolicy
// ---------------------------------------------------------------------------

/// Per-operation precision policy for an inference session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrecisionPolicy {
    /// Default precision for operations not explicitly listed.
    pub default_compute: ComputePrecision,
    /// Accumulation precision (always Float32 for stability).
    pub accumulation: ComputePrecision,
    /// Normalization layers (RMSNorm / LayerNorm — always Float32).
    pub normalization: ComputePrecision,
    /// Attention score computation.
    pub attention_scores: ComputePrecision,
    /// Matrix multiplication.
    pub matmul: ComputePrecision,
    /// Softmax (always Float32 for numerical stability).
    pub softmax: ComputePrecision,
}

impl PrecisionPolicy {
    /// All operations in FP32 — maximum safety.
    pub fn conservative() -> Self {
        Self {
            default_compute: ComputePrecision::Float32,
            accumulation: ComputePrecision::Float32,
            normalization: ComputePrecision::Float32,
            attention_scores: ComputePrecision::Float32,
            matmul: ComputePrecision::Float32,
            softmax: ComputePrecision::Float32,
        }
    }

    /// Mixed precision for matmul; FP32 for norms, softmax, accumulation.
    pub fn balanced() -> Self {
        Self {
            default_compute: ComputePrecision::Mixed,
            accumulation: ComputePrecision::Float32,
            normalization: ComputePrecision::Float32,
            attention_scores: ComputePrecision::Mixed,
            matmul: ComputePrecision::Mixed,
            softmax: ComputePrecision::Float32,
        }
    }

    /// FP16 everywhere except accumulation (which stays FP32).
    pub fn aggressive() -> Self {
        Self {
            default_compute: ComputePrecision::Float16,
            accumulation: ComputePrecision::Float32,
            normalization: ComputePrecision::Float16,
            attention_scores: ComputePrecision::Float16,
            matmul: ComputePrecision::Float16,
            softmax: ComputePrecision::Float16,
        }
    }

    /// Policy tuned for Intel Arc A770 capabilities.
    ///
    /// * FP16 matmul / attention (leverage 2× throughput)
    /// * FP32 accumulation, norms, softmax (numerical safety)
    pub fn for_a770() -> Self {
        Self {
            default_compute: ComputePrecision::Mixed,
            accumulation: ComputePrecision::Float32,
            normalization: ComputePrecision::Float32,
            attention_scores: ComputePrecision::Float16,
            matmul: ComputePrecision::Float16,
            softmax: ComputePrecision::Float32,
        }
    }

    /// Estimated memory reduction factor compared to all-Float32 (0.0–1.0).
    ///
    /// A factor of 0.5 means roughly half the memory.
    pub fn memory_reduction_factor(&self) -> f64 {
        let ops = [
            &self.default_compute,
            &self.accumulation,
            &self.normalization,
            &self.attention_scores,
            &self.matmul,
            &self.softmax,
        ];
        let total_bytes: usize = ops.iter().map(|p| p.bytes_per_element()).sum();
        let fp32_bytes = ops.len() * 4;
        total_bytes as f64 / fp32_bytes as f64
    }

    /// Aggregate theoretical speed-up vs all-Float32.
    pub fn expected_speedup(&self) -> f64 {
        let ops = [
            &self.default_compute,
            &self.accumulation,
            &self.normalization,
            &self.attention_scores,
            &self.matmul,
            &self.softmax,
        ];
        let sum: f64 = ops.iter().map(|p| p.theoretical_speedup_vs_f32()).sum();
        sum / ops.len() as f64
    }
}

impl fmt::Display for PrecisionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PrecisionPolicy {{ default={}, accum={}, norm={}, attn={}, matmul={}, softmax={} }}",
            self.default_compute,
            self.accumulation,
            self.normalization,
            self.attention_scores,
            self.matmul,
            self.softmax,
        )
    }
}

// ---------------------------------------------------------------------------
// PrecisionError
// ---------------------------------------------------------------------------

/// Errors arising from precision conversion.
#[derive(Debug, Clone)]
pub enum PrecisionError {
    /// Value exceeds FP16 representable range (>65504).
    Overflow { value: f32 },
    /// Value too small for FP16 (non-zero but rounds to zero).
    Underflow { value: f32 },
    /// NaN encountered during conversion.
    NaNDetected,
    /// Infinity encountered during conversion.
    InfDetected,
}

impl fmt::Display for PrecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow { value } => {
                write!(f, "FP16 overflow: value {value} exceeds max representable 65504.0")
            }
            Self::Underflow { value } => {
                write!(f, "FP16 underflow: value {value} is too small to represent")
            }
            Self::NaNDetected => write!(f, "NaN detected during FP16 conversion"),
            Self::InfDetected => write!(f, "infinity detected during FP16 conversion"),
        }
    }
}

impl std::error::Error for PrecisionError {}

// ---------------------------------------------------------------------------
// F16Converter  (CPU-side IEEE 754 half-precision helpers)
// ---------------------------------------------------------------------------

/// CPU-side FP16 ↔ FP32 conversion utilities for validation and data prep.
pub struct F16Converter;

impl F16Converter {
    /// Maximum finite value representable in FP16.
    pub fn max_representable_f16() -> f32 {
        65504.0
    }

    /// Smallest positive normal FP16 value (~6.10e-5).
    pub fn min_positive_f16() -> f32 {
        // 2^{-14} = 6.103515625e-5
        6.103_515_6e-5
    }

    /// Machine epsilon for FP16 (~9.77e-4).
    pub fn f16_epsilon() -> f32 {
        // 2^{-10}
        9.765_625e-4
    }

    /// Convert an `f32` value to IEEE 754 half-precision bits (`u16`).
    ///
    /// Uses the standard rounding-to-nearest-even approach.
    pub fn f32_to_f16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exponent = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x007F_FFFF;

        // NaN
        if exponent == 255 && mantissa != 0 {
            return (sign | 0x7E00) as u16; // canonical NaN
        }
        // Inf
        if exponent == 255 {
            return (sign | 0x7C00) as u16;
        }

        let new_exp = exponent - 127 + 15; // re-bias

        if new_exp >= 31 {
            // Overflow → Inf
            return (sign | 0x7C00) as u16;
        }

        if new_exp <= 0 {
            // Denorm or underflow
            if new_exp < -10 {
                return sign as u16; // too small → ±0
            }
            let m = mantissa | 0x0080_0000; // implicit 1
            let shift = (1 - new_exp) as u32 + 13;
            let half_mantissa = if shift < 32 { m >> shift } else { 0 };
            return (sign | half_mantissa) as u16;
        }

        // Normal
        let half_mantissa = mantissa >> 13;
        (sign | ((new_exp as u32) << 10) | half_mantissa) as u16
    }

    /// Convert IEEE 754 half-precision bits (`u16`) back to `f32`.
    pub fn f16_bits_to_f32(bits: u16) -> f32 {
        let sign = ((bits as u32) & 0x8000) << 16;
        let exponent = ((bits as u32) >> 10) & 0x1F;
        let mantissa = (bits as u32) & 0x03FF;

        if exponent == 0 {
            if mantissa == 0 {
                // ±0
                return f32::from_bits(sign);
            }
            // Denorm → normalise
            let mut m = mantissa;
            let mut e = 0i32;
            loop {
                e += 1;
                m <<= 1;
                if m & 0x0400 != 0 {
                    break;
                }
            }
            let f32_exp = ((127 - 15 - e) as u32) << 23;
            let f32_man = (m & 0x03FF) << 13;
            return f32::from_bits(sign | f32_exp | f32_man);
        }

        if exponent == 31 {
            if mantissa == 0 {
                // ±Inf
                return f32::from_bits(sign | 0x7F80_0000);
            }
            // NaN
            return f32::from_bits(sign | 0x7FC0_0000 | (mantissa << 13));
        }

        // Normal
        let f32_exp = ((exponent + 127 - 15) as u32) << 23;
        let f32_man = mantissa << 13;
        f32::from_bits(sign | f32_exp | f32_man)
    }

    /// Bulk-convert an `f32` buffer to FP16 bits.
    pub fn convert_buffer_f32_to_f16(input: &[f32]) -> Vec<u16> {
        input.iter().copied().map(Self::f32_to_f16_bits).collect()
    }

    /// Bulk-convert FP16 bits back to `f32`.
    pub fn convert_buffer_f16_to_f32(input: &[u16]) -> Vec<f32> {
        input.iter().copied().map(Self::f16_bits_to_f32).collect()
    }

    /// Validate that a value can be represented in FP16 without error.
    pub fn validate_f16(value: f32) -> Result<u16, PrecisionError> {
        if value.is_nan() {
            return Err(PrecisionError::NaNDetected);
        }
        if value.is_infinite() {
            return Err(PrecisionError::InfDetected);
        }
        if value.abs() > Self::max_representable_f16() {
            return Err(PrecisionError::Overflow { value });
        }
        if value != 0.0 && value.abs() < Self::min_positive_f16() * f32::EPSILON {
            return Err(PrecisionError::Underflow { value });
        }
        Ok(Self::f32_to_f16_bits(value))
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source helpers
// ---------------------------------------------------------------------------

/// OpenCL kernel helper source for mixed-precision FP16 load/store and FP32
/// accumulate patterns. Prepend this to any kernel that needs FP16 support.
pub const MIXED_PRECISION_HELPERS_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ---- FP16 load / store helpers -------------------------------------------

/// Load a half-precision value from global memory and widen to float.
inline float load_f16(__global const half* ptr, int idx) {
    return vload_half(idx, (__global const half*)ptr);
}

/// Store a float value as half-precision into global memory.
inline void store_f16(__global half* ptr, int idx, float val) {
    vstore_half(val, idx, (__global half*)ptr);
}

// ---- FP32 accumulate helpers ---------------------------------------------

/// Dot product of two FP16 vectors accumulated in FP32.
///   a, b: pointers to half-precision data
///   n:    number of elements
inline float dot_f16_accum_f32(__global const half* a,
                               __global const half* b,
                               int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        float va = vload_half(i, (__global const half*)a);
        float vb = vload_half(i, (__global const half*)b);
        acc = fma(va, vb, acc);
    }
    return acc;
}

/// Vectorised 4-wide FP16→FP32 dot accumulate (unrolled inner loop).
inline float dot4_f16_accum_f32(__global const half* a,
                                __global const half* b,
                                int n) {
    float4 acc4 = (float4)(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float4 va = (float4)(vload_half(i, (__global const half*)a),
                             vload_half(i+1, (__global const half*)a),
                             vload_half(i+2, (__global const half*)a),
                             vload_half(i+3, (__global const half*)a));
        float4 vb = (float4)(vload_half(i, (__global const half*)b),
                             vload_half(i+1, (__global const half*)b),
                             vload_half(i+2, (__global const half*)b),
                             vload_half(i+3, (__global const half*)b));
        acc4 = fma(va, vb, acc4);
    }
    float acc = acc4.x + acc4.y + acc4.z + acc4.w;
    for (; i < n; ++i) {
        float va = vload_half(i, (__global const half*)a);
        float vb = vload_half(i, (__global const half*)b);
        acc = fma(va, vb, acc);
    }
    return acc;
}

// ---- Mixed-precision matrix helpers --------------------------------------

/// Mixed-precision GEMV: y = A * x
///   A is row-major half[M][K], x is half[K], y is float[M]
__kernel void gemv_f16_accum_f32(__global const half* A,
                                 __global const half* x,
                                 __global float* y,
                                 int M, int K) {
    int row = get_global_id(0);
    if (row >= M) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_val = vload_half(row * K + k, (__global const half*)A);
        float x_val = vload_half(k, (__global const half*)x);
        acc = fma(a_val, x_val, acc);
    }
    y[row] = acc;
}
"#;

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ComputePrecision
    // -----------------------------------------------------------------------

    #[test]
    fn test_bytes_per_element_f32() {
        assert_eq!(ComputePrecision::Float32.bytes_per_element(), 4);
    }

    #[test]
    fn test_bytes_per_element_f16() {
        assert_eq!(ComputePrecision::Float16.bytes_per_element(), 2);
    }

    #[test]
    fn test_bytes_per_element_mixed() {
        assert_eq!(ComputePrecision::Mixed.bytes_per_element(), 2);
    }

    #[test]
    fn test_bytes_per_element_int8() {
        assert_eq!(ComputePrecision::Int8.bytes_per_element(), 1);
    }

    #[test]
    fn test_speedup_f32() {
        assert!((ComputePrecision::Float32.theoretical_speedup_vs_f32() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speedup_f16() {
        assert!((ComputePrecision::Float16.theoretical_speedup_vs_f32() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speedup_mixed() {
        assert!((ComputePrecision::Mixed.theoretical_speedup_vs_f32() - 1.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speedup_int8() {
        assert!((ComputePrecision::Int8.theoretical_speedup_vs_f32() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_precision_display() {
        assert_eq!(format!("{}", ComputePrecision::Float32), "FP32");
        assert_eq!(format!("{}", ComputePrecision::Float16), "FP16");
        assert!(format!("{}", ComputePrecision::Mixed).contains("Mixed"));
        assert!(format!("{}", ComputePrecision::Int8).contains("INT8"));
    }

    // -----------------------------------------------------------------------
    // PrecisionPolicy presets
    // -----------------------------------------------------------------------

    #[test]
    fn test_conservative_all_fp32() {
        let p = PrecisionPolicy::conservative();
        assert_eq!(p.default_compute, ComputePrecision::Float32);
        assert_eq!(p.accumulation, ComputePrecision::Float32);
        assert_eq!(p.normalization, ComputePrecision::Float32);
        assert_eq!(p.attention_scores, ComputePrecision::Float32);
        assert_eq!(p.matmul, ComputePrecision::Float32);
        assert_eq!(p.softmax, ComputePrecision::Float32);
    }

    #[test]
    fn test_balanced_matmul_mixed() {
        let p = PrecisionPolicy::balanced();
        assert_eq!(p.matmul, ComputePrecision::Mixed);
    }

    #[test]
    fn test_balanced_softmax_fp32() {
        let p = PrecisionPolicy::balanced();
        assert_eq!(p.softmax, ComputePrecision::Float32);
    }

    #[test]
    fn test_balanced_normalization_fp32() {
        let p = PrecisionPolicy::balanced();
        assert_eq!(p.normalization, ComputePrecision::Float32);
    }

    #[test]
    fn test_balanced_accumulation_fp32() {
        let p = PrecisionPolicy::balanced();
        assert_eq!(p.accumulation, ComputePrecision::Float32);
    }

    #[test]
    fn test_aggressive_accumulation_fp32() {
        let p = PrecisionPolicy::aggressive();
        assert_eq!(p.accumulation, ComputePrecision::Float32);
    }

    #[test]
    fn test_aggressive_matmul_fp16() {
        let p = PrecisionPolicy::aggressive();
        assert_eq!(p.matmul, ComputePrecision::Float16);
    }

    #[test]
    fn test_a770_matmul_fp16() {
        let p = PrecisionPolicy::for_a770();
        assert_eq!(p.matmul, ComputePrecision::Float16);
    }

    #[test]
    fn test_a770_attention_fp16() {
        let p = PrecisionPolicy::for_a770();
        assert_eq!(p.attention_scores, ComputePrecision::Float16);
    }

    #[test]
    fn test_a770_accumulation_fp32() {
        let p = PrecisionPolicy::for_a770();
        assert_eq!(p.accumulation, ComputePrecision::Float32);
    }

    #[test]
    fn test_a770_normalization_fp32() {
        let p = PrecisionPolicy::for_a770();
        assert_eq!(p.normalization, ComputePrecision::Float32);
    }

    #[test]
    fn test_a770_softmax_fp32() {
        let p = PrecisionPolicy::for_a770();
        assert_eq!(p.softmax, ComputePrecision::Float32);
    }

    #[test]
    fn test_conservative_memory_factor_1() {
        let factor = PrecisionPolicy::conservative().memory_reduction_factor();
        assert!((factor - 1.0).abs() < f64::EPSILON, "all-FP32 should be factor 1.0");
    }

    #[test]
    fn test_aggressive_memory_less_than_conservative() {
        let cons = PrecisionPolicy::conservative().memory_reduction_factor();
        let aggr = PrecisionPolicy::aggressive().memory_reduction_factor();
        assert!(aggr < cons, "aggressive should use less memory");
    }

    #[test]
    fn test_conservative_speedup_1() {
        let s = PrecisionPolicy::conservative().expected_speedup();
        assert!((s - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aggressive_speedup_greater_than_1() {
        let s = PrecisionPolicy::aggressive().expected_speedup();
        assert!(s > 1.0, "aggressive should be faster than FP32");
    }

    #[test]
    fn test_balanced_speedup_between_cons_aggr() {
        let c = PrecisionPolicy::conservative().expected_speedup();
        let b = PrecisionPolicy::balanced().expected_speedup();
        let a = PrecisionPolicy::aggressive().expected_speedup();
        assert!(c < b && b < a, "balanced speedup should be between conservative and aggressive");
    }

    #[test]
    fn test_policy_display() {
        let p = PrecisionPolicy::for_a770();
        let s = format!("{p}");
        assert!(s.contains("PrecisionPolicy"));
    }

    #[test]
    fn test_policy_equality() {
        assert_eq!(PrecisionPolicy::conservative(), PrecisionPolicy::conservative());
        assert_ne!(PrecisionPolicy::conservative(), PrecisionPolicy::aggressive());
    }

    // -----------------------------------------------------------------------
    // F16Converter — roundtrip & special values
    // -----------------------------------------------------------------------

    #[test]
    fn test_f16_roundtrip_one() {
        let bits = F16Converter::f32_to_f16_bits(1.0);
        let back = F16Converter::f16_bits_to_f32(bits);
        assert!((back - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_f16_roundtrip_negative() {
        let bits = F16Converter::f32_to_f16_bits(-3.5);
        let back = F16Converter::f16_bits_to_f32(bits);
        assert!((back - (-3.5)).abs() < 1e-2);
    }

    #[test]
    fn test_f16_roundtrip_small() {
        let val = 0.001;
        let bits = F16Converter::f32_to_f16_bits(val);
        let back = F16Converter::f16_bits_to_f32(bits);
        assert!((back - val).abs() < 1e-3);
    }

    #[test]
    fn test_f16_roundtrip_large() {
        let val = 65504.0_f32;
        let bits = F16Converter::f32_to_f16_bits(val);
        let back = F16Converter::f16_bits_to_f32(bits);
        assert!((back - val).abs() < 1.0);
    }

    #[test]
    fn test_f16_zero() {
        let bits = F16Converter::f32_to_f16_bits(0.0);
        assert_eq!(bits, 0x0000);
        let back = F16Converter::f16_bits_to_f32(bits);
        assert_eq!(back, 0.0);
    }

    #[test]
    fn test_f16_negative_zero() {
        let bits = F16Converter::f32_to_f16_bits(-0.0);
        assert_eq!(bits, 0x8000);
        let back = F16Converter::f16_bits_to_f32(bits);
        assert!(back == 0.0 && back.is_sign_negative());
    }

    #[test]
    fn test_f16_inf() {
        let bits = F16Converter::f32_to_f16_bits(f32::INFINITY);
        assert_eq!(bits, 0x7C00);
        let back = F16Converter::f16_bits_to_f32(bits);
        assert!(back.is_infinite() && back.is_sign_positive());
    }

    #[test]
    fn test_f16_neg_inf() {
        let bits = F16Converter::f32_to_f16_bits(f32::NEG_INFINITY);
        assert_eq!(bits, 0xFC00);
        let back = F16Converter::f16_bits_to_f32(bits);
        assert!(back.is_infinite() && back.is_sign_negative());
    }

    #[test]
    fn test_f16_nan() {
        let bits = F16Converter::f32_to_f16_bits(f32::NAN);
        let back = F16Converter::f16_bits_to_f32(bits);
        assert!(back.is_nan());
    }

    #[test]
    fn test_f16_overflow_to_inf() {
        let bits = F16Converter::f32_to_f16_bits(100_000.0);
        let back = F16Converter::f16_bits_to_f32(bits);
        assert!(back.is_infinite());
    }

    #[test]
    fn test_f16_denormal() {
        // Smallest FP16 denormal: 2^{-24} ≈ 5.96e-8
        let val = 5.96e-8_f32;
        let bits = F16Converter::f32_to_f16_bits(val);
        let back = F16Converter::f16_bits_to_f32(bits);
        // Denormals lose precision but should be close to zero
        assert!(back.abs() < 1e-4);
    }

    #[test]
    fn test_buffer_f32_to_f16_roundtrip() {
        let input = vec![0.0, 1.0, -1.0, 0.5, 100.0];
        let f16 = F16Converter::convert_buffer_f32_to_f16(&input);
        let back = F16Converter::convert_buffer_f16_to_f32(&f16);
        assert_eq!(back.len(), input.len());
        for (a, b) in input.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.1, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_large_buffer_conversion() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.1 - 50.0).collect();
        let f16 = F16Converter::convert_buffer_f32_to_f16(&input);
        assert_eq!(f16.len(), 1024);
        let back = F16Converter::convert_buffer_f16_to_f32(&f16);
        assert_eq!(back.len(), 1024);
        for (a, b) in input.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.2, "mismatch at value {a}: got {b}");
        }
    }

    #[test]
    fn test_f16_constants() {
        assert_eq!(F16Converter::max_representable_f16(), 65504.0);
        assert!(F16Converter::min_positive_f16() > 0.0);
        assert!(F16Converter::min_positive_f16() < 1e-4);
        assert!(F16Converter::f16_epsilon() > 0.0);
        assert!(F16Converter::f16_epsilon() < 1e-2);
    }

    // -----------------------------------------------------------------------
    // PrecisionError
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_overflow() {
        let res = F16Converter::validate_f16(100_000.0);
        assert!(matches!(res, Err(PrecisionError::Overflow { .. })));
    }

    #[test]
    fn test_validate_nan() {
        let res = F16Converter::validate_f16(f32::NAN);
        assert!(matches!(res, Err(PrecisionError::NaNDetected)));
    }

    #[test]
    fn test_validate_inf() {
        let res = F16Converter::validate_f16(f32::INFINITY);
        assert!(matches!(res, Err(PrecisionError::InfDetected)));
    }

    #[test]
    fn test_validate_ok() {
        let res = F16Converter::validate_f16(1.0);
        assert!(res.is_ok());
    }

    #[test]
    fn test_error_display_overflow() {
        let e = PrecisionError::Overflow { value: 99999.0 };
        let msg = format!("{e}");
        assert!(msg.contains("overflow"));
    }

    #[test]
    fn test_error_display_nan() {
        let e = PrecisionError::NaNDetected;
        let msg = format!("{e}");
        assert!(msg.contains("NaN"));
    }

    // -----------------------------------------------------------------------
    // OpenCL helpers source
    // -----------------------------------------------------------------------

    #[test]
    fn test_opencl_helpers_contain_pragma() {
        assert!(MIXED_PRECISION_HELPERS_SRC.contains("cl_khr_fp16"));
    }

    #[test]
    fn test_opencl_helpers_contain_dot() {
        assert!(MIXED_PRECISION_HELPERS_SRC.contains("dot_f16_accum_f32"));
    }

    #[test]
    fn test_opencl_helpers_contain_gemv() {
        assert!(MIXED_PRECISION_HELPERS_SRC.contains("gemv_f16_accum_f32"));
    }

    #[test]
    fn test_opencl_helpers_contain_load_store() {
        assert!(MIXED_PRECISION_HELPERS_SRC.contains("load_f16"));
        assert!(MIXED_PRECISION_HELPERS_SRC.contains("store_f16"));
    }
}
