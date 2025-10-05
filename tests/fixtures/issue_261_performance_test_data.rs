//! Issue #261 Performance Measurement Test Fixtures
//!
//! Realistic performance baselines and measurement data for CPU/GPU inference.
//! Supports AC7 (CPU performance baselines) and AC8 (GPU performance baselines).
//!
//! Baseline targets:
//! - CPU I2S AVX2: 15-20 tok/s
//! - CPU I2S AVX-512: 20-25 tok/s
//! - CPU TL1 NEON: 12-18 tok/s
//! - CPU TL2 AVX: 10-15 tok/s
//! - GPU I2S CUDA: 50-100 tok/s
//! - GPU Mixed Precision: 80-120 tok/s

#![allow(dead_code)]

use std::time::Duration;

/// Performance baseline fixture for specific architecture and quantization
#[derive(Debug, Clone)]
pub struct PerformanceBaselineFixture {
    pub baseline_id: &'static str,
    pub architecture: CpuArchitecture,
    pub quantization_type: QuantizationType,
    pub target_tokens_per_sec: PerformanceRange,
    pub latency_percentiles: LatencyPercentiles,
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub statistical_targets: StatisticalTargets,
    pub description: &'static str,
}

/// CPU architecture enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArchitecture {
    X86_64Scalar,
    X86_64AVX2,
    X86_64AVX512,
    Aarch64Scalar,
    Aarch64NEON,
}

/// Quantization type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    I2S,
    TL1,
    TL2,
}

/// Performance range (min, max)
#[derive(Debug, Clone, Copy)]
pub struct PerformanceRange {
    pub min_tokens_per_sec: f32,
    pub max_tokens_per_sec: f32,
    pub target_mean: f32,
    pub target_std_dev: f32,
}

impl PerformanceRange {
    pub fn contains(&self, value: f32) -> bool {
        value >= self.min_tokens_per_sec && value <= self.max_tokens_per_sec
    }
}

/// Latency percentiles for performance validation
#[derive(Debug, Clone, Copy)]
pub struct LatencyPercentiles {
    pub p50_ms: f32,
    pub p95_ms: f32,
    pub p99_ms: f32,
    pub max_ms: f32,
}

/// Statistical validation targets
#[derive(Debug, Clone, Copy)]
pub struct StatisticalTargets {
    pub max_coefficient_of_variation: f32, // < 5%
    pub min_sample_size: usize,
    pub outlier_threshold: f32,
}

/// Mock performance detection fixture
#[derive(Debug, Clone)]
pub struct MockPerformanceFixture {
    pub test_id: &'static str,
    pub reported_tokens_per_sec: f32,
    pub computation_type: ComputationType,
    pub expected_detection: MockDetectionResult,
    pub description: &'static str,
}

/// Computation type for mock detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputationType {
    RealQuantized,
    MockFallback,
    DequantizationFallback,
}

/// Mock detection result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockDetectionResult {
    Pass,           // Realistic performance
    FailSuspicious, // >150 tok/s indicates mock
    FailTooFast,    // >200 tok/s definitely mock
}

/// GPU performance baseline fixture
#[derive(Debug, Clone)]
pub struct GpuPerformanceFixture {
    pub baseline_id: &'static str,
    pub device_name: &'static str,
    pub compute_capability: ComputeCapability,
    pub quantization_type: QuantizationType,
    pub precision_mode: PrecisionMode,
    pub target_tokens_per_sec: PerformanceRange,
    pub memory_bandwidth_gbps: f32,
    pub tensor_core_utilization: f32,
    pub latency_percentiles: LatencyPercentiles,
    pub description: &'static str,
}

/// CUDA compute capability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
}

impl ComputeCapability {
    pub fn supports_tensor_cores(&self) -> bool {
        self.major >= 7 // Volta and newer
    }

    pub fn supports_bf16(&self) -> bool {
        self.major >= 8 // Ampere and newer
    }
}

/// GPU precision mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    FP32,
    FP16,
    BF16,
    MixedPrecision,
}

// ============================================================================
// CPU Performance Baseline Fixtures (AC7)
// ============================================================================

/// Load CPU I2S performance baselines
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
pub fn load_cpu_i2s_baselines() -> Vec<PerformanceBaselineFixture> {
    vec![
        // I2S AVX2 baseline
        PerformanceBaselineFixture {
            baseline_id: "cpu_i2s_avx2",
            architecture: CpuArchitecture::X86_64AVX2,
            quantization_type: QuantizationType::I2S,
            target_tokens_per_sec: PerformanceRange {
                min_tokens_per_sec: 15.0,
                max_tokens_per_sec: 20.0,
                target_mean: 17.5,
                target_std_dev: 1.2,
            },
            latency_percentiles: LatencyPercentiles {
                p50_ms: 55.0,
                p95_ms: 65.0,
                p99_ms: 70.0,
                max_ms: 80.0,
            },
            warmup_iterations: 5,
            measurement_iterations: 20,
            statistical_targets: StatisticalTargets {
                max_coefficient_of_variation: 0.05, // 5%
                min_sample_size: 20,
                outlier_threshold: 2.0, // 2 standard deviations
            },
            description: "CPU I2S performance baseline for x86_64 AVX2",
        },
        // I2S AVX-512 baseline
        PerformanceBaselineFixture {
            baseline_id: "cpu_i2s_avx512",
            architecture: CpuArchitecture::X86_64AVX512,
            quantization_type: QuantizationType::I2S,
            target_tokens_per_sec: PerformanceRange {
                min_tokens_per_sec: 20.0,
                max_tokens_per_sec: 25.0,
                target_mean: 22.5,
                target_std_dev: 1.0,
            },
            latency_percentiles: LatencyPercentiles {
                p50_ms: 42.0,
                p95_ms: 48.0,
                p99_ms: 52.0,
                max_ms: 60.0,
            },
            warmup_iterations: 5,
            measurement_iterations: 20,
            statistical_targets: StatisticalTargets {
                max_coefficient_of_variation: 0.05,
                min_sample_size: 20,
                outlier_threshold: 2.0,
            },
            description: "CPU I2S performance baseline for x86_64 AVX-512",
        },
    ]
}

/// Load CPU TL1 performance baselines (ARM NEON)
#[cfg(all(feature = "cpu", target_arch = "aarch64"))]
pub fn load_cpu_tl1_baselines() -> Vec<PerformanceBaselineFixture> {
    vec![PerformanceBaselineFixture {
        baseline_id: "cpu_tl1_neon",
        architecture: CpuArchitecture::Aarch64NEON,
        quantization_type: QuantizationType::TL1,
        target_tokens_per_sec: PerformanceRange {
            min_tokens_per_sec: 12.0,
            max_tokens_per_sec: 18.0,
            target_mean: 15.0,
            target_std_dev: 1.5,
        },
        latency_percentiles: LatencyPercentiles {
            p50_ms: 62.0,
            p95_ms: 72.0,
            p99_ms: 78.0,
            max_ms: 85.0,
        },
        warmup_iterations: 5,
        measurement_iterations: 20,
        statistical_targets: StatisticalTargets {
            max_coefficient_of_variation: 0.05,
            min_sample_size: 20,
            outlier_threshold: 2.0,
        },
        description: "CPU TL1 performance baseline for ARM NEON",
    }]
}

/// Load CPU TL2 performance baselines (x86 AVX)
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
pub fn load_cpu_tl2_baselines() -> Vec<PerformanceBaselineFixture> {
    vec![PerformanceBaselineFixture {
        baseline_id: "cpu_tl2_avx2",
        architecture: CpuArchitecture::X86_64AVX2,
        quantization_type: QuantizationType::TL2,
        target_tokens_per_sec: PerformanceRange {
            min_tokens_per_sec: 10.0,
            max_tokens_per_sec: 15.0,
            target_mean: 12.5,
            target_std_dev: 1.3,
        },
        latency_percentiles: LatencyPercentiles {
            p50_ms: 75.0,
            p95_ms: 88.0,
            p99_ms: 95.0,
            max_ms: 105.0,
        },
        warmup_iterations: 5,
        measurement_iterations: 20,
        statistical_targets: StatisticalTargets {
            max_coefficient_of_variation: 0.05,
            min_sample_size: 20,
            outlier_threshold: 2.0,
        },
        description: "CPU TL2 performance baseline for x86 AVX2",
    }]
}

// ============================================================================
// GPU Performance Baseline Fixtures (AC8)
// ============================================================================

/// Load GPU I2S performance baselines
#[cfg(feature = "gpu")]
pub fn load_gpu_i2s_baselines() -> Vec<GpuPerformanceFixture> {
    vec![
        // CUDA basic I2S
        GpuPerformanceFixture {
            baseline_id: "gpu_i2s_cuda_basic",
            device_name: "Generic CUDA GPU",
            compute_capability: ComputeCapability { major: 7, minor: 5 },
            quantization_type: QuantizationType::I2S,
            precision_mode: PrecisionMode::FP32,
            target_tokens_per_sec: PerformanceRange {
                min_tokens_per_sec: 50.0,
                max_tokens_per_sec: 100.0,
                target_mean: 75.0,
                target_std_dev: 8.0,
            },
            memory_bandwidth_gbps: 320.0,
            tensor_core_utilization: 0.6,
            latency_percentiles: LatencyPercentiles {
                p50_ms: 12.0,
                p95_ms: 18.0,
                p99_ms: 22.0,
                max_ms: 28.0,
            },
            description: "GPU I2S performance baseline for CUDA (FP32)",
        },
        // CUDA mixed precision
        GpuPerformanceFixture {
            baseline_id: "gpu_i2s_cuda_mixed_precision",
            device_name: "CUDA GPU with Tensor Cores",
            compute_capability: ComputeCapability { major: 8, minor: 0 },
            quantization_type: QuantizationType::I2S,
            precision_mode: PrecisionMode::MixedPrecision,
            target_tokens_per_sec: PerformanceRange {
                min_tokens_per_sec: 80.0,
                max_tokens_per_sec: 120.0,
                target_mean: 100.0,
                target_std_dev: 10.0,
            },
            memory_bandwidth_gbps: 560.0,
            tensor_core_utilization: 0.85,
            latency_percentiles: LatencyPercentiles {
                p50_ms: 9.0,
                p95_ms: 13.0,
                p99_ms: 16.0,
                max_ms: 20.0,
            },
            description: "GPU I2S performance baseline with mixed precision (FP16/BF16)",
        },
        // High-end GPU
        GpuPerformanceFixture {
            baseline_id: "gpu_i2s_cuda_high_end",
            device_name: "High-end CUDA GPU (A100/H100 class)",
            compute_capability: ComputeCapability { major: 9, minor: 0 },
            quantization_type: QuantizationType::I2S,
            precision_mode: PrecisionMode::BF16,
            target_tokens_per_sec: PerformanceRange {
                min_tokens_per_sec: 150.0,
                max_tokens_per_sec: 250.0,
                target_mean: 200.0,
                target_std_dev: 20.0,
            },
            memory_bandwidth_gbps: 1555.0,
            tensor_core_utilization: 0.92,
            latency_percentiles: LatencyPercentiles {
                p50_ms: 4.5,
                p95_ms: 6.0,
                p99_ms: 7.5,
                max_ms: 10.0,
            },
            description: "GPU I2S performance baseline for high-end hardware",
        },
    ]
}

// ============================================================================
// Mock Performance Detection Fixtures (AC6)
// ============================================================================

/// Load mock performance detection test fixtures
pub fn load_mock_detection_fixtures() -> Vec<MockPerformanceFixture> {
    vec![
        // Realistic CPU performance - should pass
        MockPerformanceFixture {
            test_id: "realistic_cpu_i2s",
            reported_tokens_per_sec: 17.5,
            computation_type: ComputationType::RealQuantized,
            expected_detection: MockDetectionResult::Pass,
            description: "Realistic CPU I2S performance (should pass validation)",
        },
        // Realistic GPU performance - should pass
        MockPerformanceFixture {
            test_id: "realistic_gpu_i2s",
            reported_tokens_per_sec: 85.0,
            computation_type: ComputationType::RealQuantized,
            expected_detection: MockDetectionResult::Pass,
            description: "Realistic GPU I2S performance (should pass validation)",
        },
        // Suspicious mock performance - should fail
        MockPerformanceFixture {
            test_id: "suspicious_mock_160",
            reported_tokens_per_sec: 160.0,
            computation_type: ComputationType::MockFallback,
            expected_detection: MockDetectionResult::FailSuspicious,
            description: "Suspicious performance (>150 tok/s, likely mock)",
        },
        // Definitely mock performance - should fail
        MockPerformanceFixture {
            test_id: "definitely_mock_250",
            reported_tokens_per_sec: 250.0,
            computation_type: ComputationType::MockFallback,
            expected_detection: MockDetectionResult::FailTooFast,
            description: "Definitely mock performance (>200 tok/s)",
        },
        // Dequantization fallback - should fail in strict mode
        MockPerformanceFixture {
            test_id: "dequant_fallback_45",
            reported_tokens_per_sec: 45.0,
            computation_type: ComputationType::DequantizationFallback,
            expected_detection: MockDetectionResult::FailSuspicious,
            description: "Dequantization fallback (should fail in strict mode)",
        },
        // Unrealistically fast CPU - should fail
        MockPerformanceFixture {
            test_id: "unrealistic_cpu_180",
            reported_tokens_per_sec: 180.0,
            computation_type: ComputationType::MockFallback,
            expected_detection: MockDetectionResult::FailSuspicious,
            description: "Unrealistically fast CPU performance (likely mock)",
        },
    ]
}

// ============================================================================
// Performance Measurement Utilities
// ============================================================================

/// Performance measurement result
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    pub tokens_per_sec: f32,
    pub latency_ms: f32,
    pub timestamp: Duration,
    pub computation_type: ComputationType,
}

/// Statistical summary of performance measurements
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub coefficient_of_variation: f32,
    pub p50: f32,
    pub p95: f32,
    pub p99: f32,
    pub sample_count: usize,
}

impl PerformanceStatistics {
    /// Create statistics from measurements
    pub fn from_measurements(measurements: &[f32]) -> Self {
        let sample_count = measurements.len();
        if sample_count == 0 {
            return Self::default();
        }

        let sum: f32 = measurements.iter().sum();
        let mean = sum / sample_count as f32;

        let variance: f32 =
            measurements.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / sample_count as f32;
        let std_dev = variance.sqrt();

        let mut sorted = measurements.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = sorted[sample_count / 2];
        let p95 = sorted[(sample_count as f32 * 0.95) as usize];
        let p99 = sorted[(sample_count as f32 * 0.99) as usize];

        Self {
            mean,
            std_dev,
            min: sorted[0],
            max: sorted[sample_count - 1],
            coefficient_of_variation: std_dev / mean,
            p50,
            p95,
            p99,
            sample_count,
        }
    }
}

impl Default for PerformanceStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            coefficient_of_variation: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
            sample_count: 0,
        }
    }
}

/// Validate performance against baseline
pub fn validate_performance_against_baseline(
    measured: f32,
    baseline: &PerformanceBaselineFixture,
) -> bool {
    baseline.target_tokens_per_sec.contains(measured)
}

/// Detect mock performance
pub fn detect_mock_performance(tokens_per_sec: f32, device_type: &str) -> MockDetectionResult {
    if tokens_per_sec > 200.0 {
        MockDetectionResult::FailTooFast
    } else if tokens_per_sec > 150.0 {
        MockDetectionResult::FailSuspicious
    } else if device_type == "cpu" && tokens_per_sec > 30.0 {
        MockDetectionResult::FailSuspicious
    } else {
        MockDetectionResult::Pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(feature = "cpu", target_arch = "x86_64"))]
    fn test_cpu_baseline_ranges() {
        let baselines = load_cpu_i2s_baselines();
        for baseline in baselines {
            assert!(baseline.target_tokens_per_sec.min_tokens_per_sec > 0.0);
            assert!(
                baseline.target_tokens_per_sec.max_tokens_per_sec
                    > baseline.target_tokens_per_sec.min_tokens_per_sec
            );
            assert!(
                baseline.target_tokens_per_sec.target_mean
                    >= baseline.target_tokens_per_sec.min_tokens_per_sec
            );
            assert!(
                baseline.target_tokens_per_sec.target_mean
                    <= baseline.target_tokens_per_sec.max_tokens_per_sec
            );
        }
    }

    #[test]
    fn test_mock_detection() {
        assert_eq!(detect_mock_performance(17.5, "cpu"), MockDetectionResult::Pass);
        assert_eq!(detect_mock_performance(160.0, "gpu"), MockDetectionResult::FailSuspicious);
        assert_eq!(detect_mock_performance(250.0, "gpu"), MockDetectionResult::FailTooFast);
    }

    #[test]
    fn test_performance_statistics() {
        let measurements = vec![15.0, 16.0, 17.0, 18.0, 19.0, 20.0];
        let stats = PerformanceStatistics::from_measurements(&measurements);
        assert_eq!(stats.mean, 17.5);
        assert!(stats.std_dev > 0.0);
        assert_eq!(stats.sample_count, 6);
    }

    #[test]
    fn test_compute_capability_features() {
        let volta = ComputeCapability { major: 7, minor: 0 };
        assert!(volta.supports_tensor_cores());
        assert!(!volta.supports_bf16());

        let ampere = ComputeCapability { major: 8, minor: 0 };
        assert!(ampere.supports_tensor_cores());
        assert!(ampere.supports_bf16());
    }
}
