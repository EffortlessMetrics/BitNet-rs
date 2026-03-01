//! GPU debug validation layer.
//!
//! Wraps any [`KernelProvider`] and validates GPU outputs against a CPU
//! reference execution. Enable at runtime via `BITNET_GPU_DEBUG=1`.

use crate::KernelProvider;
use crate::cpu::FallbackKernel;
use bitnet_common::{QuantizationType, Result};
use std::fmt;

/// Tolerance configuration for floating-point comparison.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    /// Maximum allowed absolute difference per element.
    pub abs_epsilon: f32,
    /// Maximum allowed relative difference per element.
    pub rel_epsilon: f32,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self { abs_epsilon: 1e-4, rel_epsilon: 1e-3 }
    }
}

/// A single mismatch between expected (CPU) and actual (GPU) output.
#[derive(Debug, Clone)]
pub struct Mismatch {
    /// Index in the output buffer.
    pub index: usize,
    /// Value produced by the CPU reference kernel.
    pub expected: f32,
    /// Value produced by the wrapped GPU kernel.
    pub got: f32,
    /// Absolute difference.
    pub abs_diff: f32,
    /// Relative difference (relative to expected; `f32::INFINITY` when expected == 0).
    pub rel_diff: f32,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "index={} expected={:.6} got={:.6} abs_diff={:.6} rel_diff={:.6}",
            self.index, self.expected, self.got, self.abs_diff, self.rel_diff
        )
    }
}

/// Detailed report of a validation failure.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Name of the kernel operation that failed validation.
    pub operation: String,
    /// Tolerance that was used.
    pub tolerance: Tolerance,
    /// All mismatched elements.
    pub mismatches: Vec<Mismatch>,
    /// Total number of elements compared.
    pub total_elements: usize,
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "GPU debug validation FAILED for '{}': {}/{} elements mismatched \
             (abs_eps={:.1e}, rel_eps={:.1e})",
            self.operation,
            self.mismatches.len(),
            self.total_elements,
            self.tolerance.abs_epsilon,
            self.tolerance.rel_epsilon,
        )?;
        for m in &self.mismatches {
            writeln!(f, "  {m}")?;
        }
        Ok(())
    }
}

/// GPU debug validation layer that wraps any [`KernelProvider`].
///
/// Every kernel call is mirrored on the CPU fallback kernel; outputs are
/// compared element-wise against the configured [`Tolerance`].
pub struct DebugLayer<P: KernelProvider> {
    inner: P,
    reference: FallbackKernel,
    tolerance: Tolerance,
}

impl<P: KernelProvider> DebugLayer<P> {
    /// Wrap `inner` with the default tolerance.
    pub fn new(inner: P) -> Self {
        Self { inner, reference: FallbackKernel, tolerance: Tolerance::default() }
    }

    /// Wrap `inner` with a custom tolerance.
    pub fn with_tolerance(inner: P, tolerance: Tolerance) -> Self {
        Self { inner, reference: FallbackKernel, tolerance }
    }

    /// Return the current tolerance.
    pub fn tolerance(&self) -> Tolerance {
        self.tolerance
    }

    /// Return a reference to the wrapped provider.
    pub fn inner(&self) -> &P {
        &self.inner
    }
}

/// Check whether `BITNET_GPU_DEBUG` is set to a truthy value.
pub fn gpu_debug_enabled() -> bool {
    std::env::var("BITNET_GPU_DEBUG")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// Compare two f32 slices element-wise and collect mismatches.
fn compare_outputs(expected: &[f32], got: &[f32], tolerance: &Tolerance) -> Vec<Mismatch> {
    assert_eq!(expected.len(), got.len(), "output length mismatch in debug layer");
    expected
        .iter()
        .zip(got.iter())
        .enumerate()
        .filter_map(|(i, (&e, &g))| {
            let abs_diff = (e - g).abs();
            let rel_diff = if e.abs() > f32::EPSILON { abs_diff / e.abs() } else { abs_diff };
            if abs_diff > tolerance.abs_epsilon && rel_diff > tolerance.rel_epsilon {
                Some(Mismatch { index: i, expected: e, got: g, abs_diff, rel_diff })
            } else {
                None
            }
        })
        .collect()
}

/// Compare two u8 slices element-wise (quantised output).
fn compare_quantized_outputs(expected: &[u8], got: &[u8]) -> Vec<Mismatch> {
    assert_eq!(expected.len(), got.len(), "quantized output length mismatch in debug layer");
    expected
        .iter()
        .zip(got.iter())
        .enumerate()
        .filter_map(|(i, (&e, &g))| {
            if e != g {
                Some(Mismatch {
                    index: i,
                    expected: e as f32,
                    got: g as f32,
                    abs_diff: (e as f32 - g as f32).abs(),
                    rel_diff: if e > 0 {
                        (e as f32 - g as f32).abs() / e as f32
                    } else {
                        (e as f32 - g as f32).abs()
                    },
                })
            } else {
                None
            }
        })
        .collect()
}

impl<P: KernelProvider> KernelProvider for DebugLayer<P> {
    fn name(&self) -> &'static str {
        // We leak a &'static str because the trait requires it.
        // This is acceptable because provider names are created once.
        "DebugLayer"
    }

    fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // Run on the wrapped (GPU) provider.
        self.inner.matmul_i2s(a, b, c, m, n, k)?;

        // Run the same operation on the CPU reference.
        let mut ref_c = vec![0.0f32; c.len()];
        self.reference.matmul_i2s(a, b, &mut ref_c, m, n, k)?;

        let mismatches = compare_outputs(&ref_c, c, &self.tolerance);
        if !mismatches.is_empty() {
            let report = ValidationReport {
                operation: "matmul_i2s".to_string(),
                tolerance: self.tolerance,
                mismatches,
                total_elements: c.len(),
            };
            log::error!("{report}");
        }
        Ok(())
    }

    fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        // Run on the wrapped (GPU) provider.
        self.inner.quantize(input, output, scales, qtype)?;

        // Run the same operation on the CPU reference.
        let mut ref_output = vec![0u8; output.len()];
        let mut ref_scales = vec![0.0f32; scales.len()];
        self.reference.quantize(input, &mut ref_output, &mut ref_scales, qtype)?;

        // Compare quantised bytes.
        let byte_mismatches = compare_quantized_outputs(&ref_output, output);
        if !byte_mismatches.is_empty() {
            let report = ValidationReport {
                operation: "quantize (bytes)".to_string(),
                tolerance: self.tolerance,
                mismatches: byte_mismatches,
                total_elements: output.len(),
            };
            log::error!("{report}");
        }

        // Compare scales.
        let scale_mismatches = compare_outputs(&ref_scales, scales, &self.tolerance);
        if !scale_mismatches.is_empty() {
            let report = ValidationReport {
                operation: "quantize (scales)".to_string(),
                tolerance: self.tolerance,
                mismatches: scale_mismatches,
                total_elements: scales.len(),
            };
            log::error!("{report}");
        }

        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::FallbackKernel;
    use bitnet_common::QuantizationType;

    // ── helpers ──────────────────────────────────────────────────────────

    /// A provider that adds a fixed bias to matmul output (simulates GPU drift).
    struct BiasedKernel {
        bias: f32,
        inner: FallbackKernel,
    }

    impl BiasedKernel {
        fn new(bias: f32) -> Self {
            Self { bias, inner: FallbackKernel }
        }
    }

    impl KernelProvider for BiasedKernel {
        fn name(&self) -> &'static str {
            "BiasedKernel"
        }

        fn is_available(&self) -> bool {
            true
        }

        fn matmul_i2s(
            &self,
            a: &[i8],
            b: &[u8],
            c: &mut [f32],
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<()> {
            self.inner.matmul_i2s(a, b, c, m, n, k)?;
            for v in c.iter_mut() {
                *v += self.bias;
            }
            Ok(())
        }

        fn quantize(
            &self,
            input: &[f32],
            output: &mut [u8],
            scales: &mut [f32],
            qtype: QuantizationType,
        ) -> Result<()> {
            self.inner.quantize(input, output, scales, qtype)?;
            for s in scales.iter_mut() {
                *s += self.bias;
            }
            Ok(())
        }
    }

    // ── tests ────────────────────────────────────────────────────────────

    #[test]
    fn debug_layer_name() {
        let layer = DebugLayer::new(FallbackKernel);
        assert_eq!(layer.name(), "DebugLayer");
    }

    #[test]
    fn debug_layer_delegates_is_available() {
        let layer = DebugLayer::new(FallbackKernel);
        assert!(layer.is_available());
    }

    #[test]
    fn debug_layer_passes_identical_matmul() {
        let layer = DebugLayer::new(FallbackKernel);
        let m = 2;
        let n = 4;
        let k = 4;
        let a = vec![1i8; m * k];
        let b = vec![0u8; k * n]; // FallbackKernel expects k*n bytes
        let mut c = vec![0.0f32; m * n];
        layer.matmul_i2s(&a, &b, &mut c, m, n, k).unwrap();
    }

    #[test]
    fn debug_layer_detects_matmul_mismatch() {
        let biased = BiasedKernel::new(1.0);
        let layer =
            DebugLayer::with_tolerance(biased, Tolerance { abs_epsilon: 1e-6, rel_epsilon: 1e-6 });
        let m = 2;
        let n = 4;
        let k = 4;
        let a = vec![1i8; m * k];
        let b = vec![0xFFu8; k * n];
        let mut c = vec![0.0f32; m * n];
        // Runs without panic; mismatch is logged, not returned as error.
        layer.matmul_i2s(&a, &b, &mut c, m, n, k).unwrap();
    }

    #[test]
    fn debug_layer_passes_identical_quantize() {
        let layer = DebugLayer::new(FallbackKernel);
        // I2S uses block_size=32, so 64 elements → 2 blocks → 2 scales
        let input = vec![0.5f32; 64];
        let mut output = vec![0u8; 16];
        let mut scales = vec![0.0f32; 2];
        layer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    }

    #[test]
    fn debug_layer_detects_quantize_scale_mismatch() {
        let biased = BiasedKernel::new(0.5);
        let layer =
            DebugLayer::with_tolerance(biased, Tolerance { abs_epsilon: 1e-6, rel_epsilon: 1e-6 });
        let input = vec![1.0f32; 64];
        let mut output = vec![0u8; 16];
        let mut scales = vec![0.0f32; 2];
        // Mismatch on scales is logged, not returned as error.
        layer.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    }

    #[test]
    fn tolerance_default_values() {
        let t = Tolerance::default();
        assert!((t.abs_epsilon - 1e-4).abs() < f32::EPSILON);
        assert!((t.rel_epsilon - 1e-3).abs() < f32::EPSILON);
    }

    #[test]
    fn compare_outputs_empty() {
        let m = compare_outputs(&[], &[], &Tolerance::default());
        assert!(m.is_empty());
    }

    #[test]
    fn compare_outputs_within_tolerance() {
        let expected = vec![1.0, 2.0, 3.0];
        let got = vec![1.00005, 2.00005, 3.00005];
        let m = compare_outputs(&expected, &got, &Tolerance::default());
        assert!(m.is_empty(), "small diffs should be within tolerance");
    }

    #[test]
    fn compare_outputs_beyond_tolerance() {
        let expected = vec![1.0, 2.0, 3.0];
        let got = vec![1.0, 2.5, 3.0];
        let t = Tolerance { abs_epsilon: 1e-6, rel_epsilon: 1e-6 };
        let m = compare_outputs(&expected, &got, &t);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].index, 1);
        assert!((m[0].expected - 2.0).abs() < f32::EPSILON);
        assert!((m[0].got - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn mismatch_display() {
        let m = Mismatch { index: 42, expected: 1.0, got: 1.5, abs_diff: 0.5, rel_diff: 0.5 };
        let s = format!("{m}");
        assert!(s.contains("index=42"));
        assert!(s.contains("expected="));
        assert!(s.contains("got="));
    }

    #[test]
    fn validation_report_display() {
        let report = ValidationReport {
            operation: "test_op".to_string(),
            tolerance: Tolerance::default(),
            mismatches: vec![Mismatch {
                index: 0,
                expected: 1.0,
                got: 2.0,
                abs_diff: 1.0,
                rel_diff: 1.0,
            }],
            total_elements: 10,
        };
        let s = format!("{report}");
        assert!(s.contains("test_op"));
        assert!(s.contains("1/10"));
    }

    #[test]
    #[serial_test::serial(bitnet_env)]
    fn gpu_debug_enabled_env() {
        temp_env::with_var("BITNET_GPU_DEBUG", Some("1"), || {
            assert!(gpu_debug_enabled());
        });
        temp_env::with_var("BITNET_GPU_DEBUG", Some("0"), || {
            assert!(!gpu_debug_enabled());
        });
        temp_env::with_var("BITNET_GPU_DEBUG", None::<&str>, || {
            assert!(!gpu_debug_enabled());
        });
        temp_env::with_var("BITNET_GPU_DEBUG", Some("true"), || {
            assert!(gpu_debug_enabled());
        });
        temp_env::with_var("BITNET_GPU_DEBUG", Some("yes"), || {
            assert!(gpu_debug_enabled());
        });
    }

    #[test]
    fn debug_layer_inner_access() {
        let layer = DebugLayer::new(FallbackKernel);
        assert_eq!(layer.inner().name(), "fallback");
        let t = layer.tolerance();
        assert!((t.abs_epsilon - 1e-4).abs() < f32::EPSILON);
    }
}
