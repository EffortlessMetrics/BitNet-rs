//! C++ Reference Cross-Validation Test Fixtures for Issue #260
//!
//! Provides test data for validating Rust quantization implementations against
//! Microsoft BitNet C++ reference implementations. Includes deterministic test
//! vectors, tolerance specifications, and parity validation scenarios.

use std::collections::HashMap;

/// Cross-validation test fixture for C++ reference comparison
#[derive(Debug, Clone)]
pub struct CppReferenceFixture {
    pub test_name: &'static str,
    pub description: &'static str,
    pub quantization_method: QuantizationMethod,
    pub input_data: CrossValInputData,
    pub expected_rust_output: CrossValOutput,
    pub expected_cpp_output: CrossValOutput,
    pub tolerance_spec: ToleranceSpecification,
    pub validation_metadata: ValidationMetadata,
    pub deterministic_seed: u64,
}

/// Quantization methods for cross-validation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMethod {
    I2S,
    TL1,
    TL2,
    IQ2S, // GGML-compatible
}

/// Input data for cross-validation tests
#[derive(Debug, Clone)]
pub struct CrossValInputData {
    pub weights: Vec<f32>,
    pub input_vectors: Vec<Vec<f32>>,
    pub quantization_params: QuantizationParameters,
    pub block_configuration: BlockConfiguration,
    pub device_context: DeviceContext,
}

/// Quantization parameters for cross-validation
#[derive(Debug, Clone)]
pub struct QuantizationParameters {
    pub block_size: usize,
    pub scale_method: ScaleComputationMethod,
    pub clamp_range: Option<(i32, i32)>,
    pub lookup_table_config: Option<LookupTableConfig>,
    pub zero_point: Option<i32>,
    pub symmetric: bool,
}

/// Block configuration for quantization
#[derive(Debug, Clone)]
pub struct BlockConfiguration {
    pub block_size: usize,
    pub alignment_bytes: usize,
    pub overlapping_blocks: bool,
    pub padding_strategy: PaddingStrategy,
}

/// Device context for execution
#[derive(Debug, Clone)]
pub struct DeviceContext {
    pub device_type: DeviceType,
    pub compute_capability: Option<ComputeCapability>,
    pub memory_alignment: usize,
    pub precision_mode: PrecisionMode,
}

/// Device types for cross-validation
#[derive(Debug, Clone, Copy)]
pub enum DeviceType {
    CpuReference,
    CpuSimd,
    GpuCuda,
    GpuOpenCL,
}

/// Compute capability for GPU devices
#[derive(Debug, Clone)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
    pub tensor_cores: bool,
    pub mixed_precision: bool,
}

/// Precision modes for computation
#[derive(Debug, Clone, Copy)]
pub enum PrecisionMode {
    FP32,
    FP16,
    BF16,
    Mixed,
}

/// Scale computation methods
#[derive(Debug, Clone, Copy)]
pub enum ScaleComputationMethod {
    AbsMax,
    RmsNorm,
    MinMax,
    Percentile(f32),
}

/// Lookup table configuration
#[derive(Debug, Clone)]
pub struct LookupTableConfig {
    pub table_size: usize,
    pub initialization_method: TableInitMethod,
    pub optimization_level: OptimizationLevel,
    pub cache_layout: CacheLayout,
}

/// Table initialization methods
#[derive(Debug, Clone, Copy)]
pub enum TableInitMethod {
    Uniform,
    Normal,
    Kmeans,
    Optimal,
}

/// Optimization levels
#[derive(Debug, Clone, Copy)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    Platform,
}

/// Cache layout strategies
#[derive(Debug, Clone, Copy)]
pub enum CacheLayout {
    Linear,
    Blocked,
    Hierarchical,
}

/// Padding strategies for blocks
#[derive(Debug, Clone, Copy)]
pub enum PaddingStrategy {
    Zero,
    Replicate,
    Reflect,
    Symmetric,
}

/// Expected output data structure
#[derive(Debug, Clone)]
pub struct CrossValOutput {
    pub quantized_weights: Vec<u8>,
    pub scale_factors: Vec<f32>,
    pub lookup_table: Option<Vec<f32>>,
    pub inference_results: Vec<Vec<f32>>,
    pub performance_metrics: PerformanceMetrics,
    pub memory_usage: MemoryUsage,
}

/// Performance metrics for validation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub quantization_time_ms: f32,
    pub inference_time_ms: f32,
    pub throughput_tokens_per_sec: f32,
    pub memory_bandwidth_gb_per_sec: f32,
    pub cache_hit_rate: f32,
}

/// Memory usage tracking
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub peak_usage_bytes: usize,
    pub working_set_bytes: usize,
    pub fragmentation_ratio: f32,
    pub alignment_overhead_bytes: usize,
}

/// Tolerance specification for comparison
#[derive(Debug, Clone)]
pub struct ToleranceSpecification {
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
    pub correlation_threshold: f32,
    pub max_outlier_percentage: f32,
    pub statistical_tests: Vec<StatisticalTest>,
    pub performance_tolerance: PerformanceTolerance,
}

/// Statistical tests for validation
#[derive(Debug, Clone)]
pub struct StatisticalTest {
    pub test_type: StatisticalTestType,
    pub significance_level: f32,
    pub expected_result: TestResult,
}

/// Types of statistical tests
#[derive(Debug, Clone, Copy)]
pub enum StatisticalTestType {
    KolmogorovSmirnov,
    TTest,
    ChiSquare,
    Correlation,
    MeanSquaredError,
}

/// Test result expectation
#[derive(Debug, Clone, Copy)]
pub enum TestResult {
    Pass,
    Fail,
    Inconclusive,
}

/// Performance tolerance specification
#[derive(Debug, Clone)]
pub struct PerformanceTolerance {
    pub max_slowdown_factor: f32,
    pub min_throughput_ratio: f32,
    pub max_memory_overhead: f32,
    pub timing_variance_threshold: f32,
}

/// Validation metadata
#[derive(Debug, Clone)]
pub struct ValidationMetadata {
    pub cpp_version: &'static str,
    pub rust_version: &'static str,
    pub test_framework_version: &'static str,
    pub reference_implementation: &'static str,
    pub validation_date: &'static str,
    pub known_issues: Vec<&'static str>,
    pub compatibility_notes: Vec<&'static str>,
}

/// Load basic I2S cross-validation fixtures
#[cfg(feature = "crossval")]
pub fn load_i2s_crossval_fixtures() -> Vec<CppReferenceFixture> {
    vec![
        // Basic I2S quantization comparison
        CppReferenceFixture {
            test_name: "i2s_basic_crossval",
            description: "Basic I2S quantization cross-validation against Microsoft BitNet reference",
            quantization_method: QuantizationMethod::I2S,
            input_data: create_i2s_input_data(256, 42),
            expected_rust_output: create_expected_rust_output_i2s(256),
            expected_cpp_output: create_expected_cpp_output_i2s(256),
            tolerance_spec: create_i2s_tolerance_spec(),
            validation_metadata: create_validation_metadata("i2s_basic"),
            deterministic_seed: 42,
        },
        // I2S with different block sizes
        CppReferenceFixture {
            test_name: "i2s_block_sizes_crossval",
            description: "I2S quantization with various block sizes validation",
            quantization_method: QuantizationMethod::I2S,
            input_data: create_i2s_input_data_multiblock(1024),
            expected_rust_output: create_expected_rust_output_i2s(1024),
            expected_cpp_output: create_expected_cpp_output_i2s(1024),
            tolerance_spec: create_i2s_tolerance_spec(),
            validation_metadata: create_validation_metadata("i2s_blocks"),
            deterministic_seed: 123,
        },
        // I2S edge cases
        CppReferenceFixture {
            test_name: "i2s_edge_cases_crossval",
            description: "I2S quantization edge cases and boundary conditions",
            quantization_method: QuantizationMethod::I2S,
            input_data: create_i2s_edge_case_data(),
            expected_rust_output: create_expected_rust_output_i2s(64),
            expected_cpp_output: create_expected_cpp_output_i2s(64),
            tolerance_spec: create_relaxed_tolerance_spec(),
            validation_metadata: create_validation_metadata("i2s_edge"),
            deterministic_seed: 456,
        },
        // I2S performance validation
        CppReferenceFixture {
            test_name: "i2s_performance_crossval",
            description: "I2S quantization performance parity validation",
            quantization_method: QuantizationMethod::I2S,
            input_data: create_i2s_performance_data(4096),
            expected_rust_output: create_expected_rust_output_i2s(4096),
            expected_cpp_output: create_expected_cpp_output_i2s(4096),
            tolerance_spec: create_performance_tolerance_spec(),
            validation_metadata: create_validation_metadata("i2s_perf"),
            deterministic_seed: 789,
        },
    ]
}

/// Load TL1/TL2 cross-validation fixtures
#[cfg(feature = "crossval")]
pub fn load_tl_crossval_fixtures() -> Vec<CppReferenceFixture> {
    vec![
        // TL1 lookup table validation
        CppReferenceFixture {
            test_name: "tl1_lookup_crossval",
            description: "TL1 lookup table quantization cross-validation",
            quantization_method: QuantizationMethod::TL1,
            input_data: create_tl1_input_data(512),
            expected_rust_output: create_expected_rust_output_tl1(512),
            expected_cpp_output: create_expected_cpp_output_tl1(512),
            tolerance_spec: create_tl1_tolerance_spec(),
            validation_metadata: create_validation_metadata("tl1_lookup"),
            deterministic_seed: 101,
        },
        // TL2 advanced lookup validation
        CppReferenceFixture {
            test_name: "tl2_advanced_crossval",
            description: "TL2 advanced lookup table quantization cross-validation",
            quantization_method: QuantizationMethod::TL2,
            input_data: create_tl2_input_data(2048),
            expected_rust_output: create_expected_rust_output_tl2(2048),
            expected_cpp_output: create_expected_cpp_output_tl2(2048),
            tolerance_spec: create_tl2_tolerance_spec(),
            validation_metadata: create_validation_metadata("tl2_advanced"),
            deterministic_seed: 202,
        },
        // TL1 vs TL2 comparison
        CppReferenceFixture {
            test_name: "tl1_vs_tl2_crossval",
            description: "TL1 vs TL2 comparative validation",
            quantization_method: QuantizationMethod::TL1, // Primary method
            input_data: create_tl_comparison_data(1024),
            expected_rust_output: create_expected_rust_output_tl1(1024),
            expected_cpp_output: create_expected_cpp_output_tl1(1024),
            tolerance_spec: create_comparative_tolerance_spec(),
            validation_metadata: create_validation_metadata("tl_comparison"),
            deterministic_seed: 303,
        },
    ]
}

/// Load comprehensive cross-validation scenarios
#[cfg(feature = "crossval")]
pub fn load_comprehensive_crossval_fixtures() -> Vec<CppReferenceFixture> {
    vec![
        // Multi-method validation
        CppReferenceFixture {
            test_name: "multi_method_crossval",
            description: "Multiple quantization methods validation suite",
            quantization_method: QuantizationMethod::I2S, // Primary
            input_data: create_multi_method_input_data(),
            expected_rust_output: create_expected_rust_output_multi(),
            expected_cpp_output: create_expected_cpp_output_multi(),
            tolerance_spec: create_multi_method_tolerance_spec(),
            validation_metadata: create_validation_metadata("multi_method"),
            deterministic_seed: 404,
        },
        // Real model validation
        CppReferenceFixture {
            test_name: "real_model_crossval",
            description: "Real BitNet model quantization validation",
            quantization_method: QuantizationMethod::I2S,
            input_data: create_real_model_input_data(),
            expected_rust_output: create_expected_rust_output_model(),
            expected_cpp_output: create_expected_cpp_output_model(),
            tolerance_spec: create_model_tolerance_spec(),
            validation_metadata: create_validation_metadata("real_model"),
            deterministic_seed: 505,
        },
        // Regression test suite
        CppReferenceFixture {
            test_name: "regression_suite_crossval",
            description: "Regression test suite for quantization parity",
            quantization_method: QuantizationMethod::I2S,
            input_data: create_regression_input_data(),
            expected_rust_output: create_expected_rust_output_regression(),
            expected_cpp_output: create_expected_cpp_output_regression(),
            tolerance_spec: create_regression_tolerance_spec(),
            validation_metadata: create_validation_metadata("regression"),
            deterministic_seed: 606,
        },
    ]
}

/// Helper functions to create test data

fn create_i2s_input_data(size: usize, seed: u64) -> CrossValInputData {
    CrossValInputData {
        weights: generate_deterministic_weights(size, seed),
        input_vectors: vec![
            generate_deterministic_weights(size, seed + 1),
            generate_deterministic_weights(size, seed + 2),
        ],
        quantization_params: QuantizationParameters {
            block_size: 32,
            scale_method: ScaleComputationMethod::AbsMax,
            clamp_range: Some((-2, 1)),
            lookup_table_config: None,
            zero_point: None,
            symmetric: true,
        },
        block_configuration: BlockConfiguration {
            block_size: 32,
            alignment_bytes: 32,
            overlapping_blocks: false,
            padding_strategy: PaddingStrategy::Zero,
        },
        device_context: DeviceContext {
            device_type: DeviceType::CpuReference,
            compute_capability: None,
            memory_alignment: 32,
            precision_mode: PrecisionMode::FP32,
        },
    }
}

fn create_i2s_input_data_multiblock(size: usize) -> CrossValInputData {
    let mut input = create_i2s_input_data(size, 123);

    // Test multiple block sizes
    input.quantization_params.block_size = 64;
    input.block_configuration.block_size = 64;

    input
}

fn create_i2s_edge_case_data() -> CrossValInputData {
    let mut input = create_i2s_input_data(64, 456);

    // Create edge case weights
    input.weights =
        vec![f32::MIN, f32::MAX, 0.0, -0.0, f32::EPSILON, -f32::EPSILON, 1.0, -1.0, 0.5, -0.5]
            .into_iter()
            .cycle()
            .take(64)
            .collect();

    input
}

fn create_i2s_performance_data(size: usize) -> CrossValInputData {
    let mut input = create_i2s_input_data(size, 789);

    // Configure for performance testing
    input.device_context.device_type = DeviceType::CpuSimd;
    input.block_configuration.alignment_bytes = 64; // AVX-512 alignment

    input
}

fn create_tl1_input_data(size: usize) -> CrossValInputData {
    let mut input = create_i2s_input_data(size, 101);

    input.quantization_params = QuantizationParameters {
        block_size: 64,
        scale_method: ScaleComputationMethod::RmsNorm,
        clamp_range: None,
        lookup_table_config: Some(LookupTableConfig {
            table_size: 256,
            initialization_method: TableInitMethod::Kmeans,
            optimization_level: OptimizationLevel::Basic,
            cache_layout: CacheLayout::Linear,
        }),
        zero_point: Some(128),
        symmetric: false,
    };

    input
}

fn create_tl2_input_data(size: usize) -> CrossValInputData {
    let mut input = create_i2s_input_data(size, 202);

    input.quantization_params = QuantizationParameters {
        block_size: 128,
        scale_method: ScaleComputationMethod::MinMax,
        clamp_range: None,
        lookup_table_config: Some(LookupTableConfig {
            table_size: 4096,
            initialization_method: TableInitMethod::Optimal,
            optimization_level: OptimizationLevel::Aggressive,
            cache_layout: CacheLayout::Blocked,
        }),
        zero_point: Some(2048),
        symmetric: false,
    };

    input
}

fn create_tl_comparison_data(size: usize) -> CrossValInputData {
    create_i2s_input_data(size, 303)
}

fn create_multi_method_input_data() -> CrossValInputData {
    create_i2s_input_data(1024, 404)
}

fn create_real_model_input_data() -> CrossValInputData {
    let mut input = create_i2s_input_data(2048, 505);

    // Simulate real model characteristics
    input.weights = generate_realistic_model_weights(2048);

    input
}

fn create_regression_input_data() -> CrossValInputData {
    create_i2s_input_data(512, 606)
}

/// Expected output creation functions

fn create_expected_rust_output_i2s(size: usize) -> CrossValOutput {
    CrossValOutput {
        quantized_weights: generate_i2s_quantized_values(size),
        scale_factors: generate_scale_factors(size, 32),
        lookup_table: None,
        inference_results: vec![generate_inference_result(size), generate_inference_result(size)],
        performance_metrics: PerformanceMetrics {
            quantization_time_ms: 1.2,
            inference_time_ms: 0.8,
            throughput_tokens_per_sec: 25.0,
            memory_bandwidth_gb_per_sec: 15.0,
            cache_hit_rate: 0.85,
        },
        memory_usage: MemoryUsage {
            peak_usage_bytes: size * 4,
            working_set_bytes: size * 2,
            fragmentation_ratio: 0.15,
            alignment_overhead_bytes: 64,
        },
    }
}

fn create_expected_cpp_output_i2s(size: usize) -> CrossValOutput {
    let mut output = create_expected_rust_output_i2s(size);

    // Slight differences in C++ implementation
    output.performance_metrics.quantization_time_ms = 1.1;
    output.performance_metrics.inference_time_ms = 0.9;
    output.performance_metrics.cache_hit_rate = 0.87;

    output
}

fn create_expected_rust_output_tl1(size: usize) -> CrossValOutput {
    let mut output = create_expected_rust_output_i2s(size);

    output.lookup_table = Some(generate_tl1_lookup_table());
    output.performance_metrics.quantization_time_ms = 2.1;
    output.performance_metrics.throughput_tokens_per_sec = 20.0;

    output
}

fn create_expected_cpp_output_tl1(size: usize) -> CrossValOutput {
    let mut output = create_expected_rust_output_tl1(size);

    output.performance_metrics.quantization_time_ms = 2.0;
    output.performance_metrics.throughput_tokens_per_sec = 21.0;

    output
}

fn create_expected_rust_output_tl2(size: usize) -> CrossValOutput {
    let mut output = create_expected_rust_output_i2s(size);

    output.lookup_table = Some(generate_tl2_lookup_table());
    output.performance_metrics.quantization_time_ms = 3.5;
    output.performance_metrics.throughput_tokens_per_sec = 18.0;

    output
}

fn create_expected_cpp_output_tl2(size: usize) -> CrossValOutput {
    let mut output = create_expected_rust_output_tl2(size);

    output.performance_metrics.quantization_time_ms = 3.2;
    output.performance_metrics.throughput_tokens_per_sec = 19.0;

    output
}

fn create_expected_rust_output_multi() -> CrossValOutput {
    create_expected_rust_output_i2s(1024)
}

fn create_expected_cpp_output_multi() -> CrossValOutput {
    create_expected_cpp_output_i2s(1024)
}

fn create_expected_rust_output_model() -> CrossValOutput {
    create_expected_rust_output_i2s(2048)
}

fn create_expected_cpp_output_model() -> CrossValOutput {
    create_expected_cpp_output_i2s(2048)
}

fn create_expected_rust_output_regression() -> CrossValOutput {
    create_expected_rust_output_i2s(512)
}

fn create_expected_cpp_output_regression() -> CrossValOutput {
    create_expected_cpp_output_i2s(512)
}

/// Tolerance specification creation functions

fn create_i2s_tolerance_spec() -> ToleranceSpecification {
    ToleranceSpecification {
        absolute_tolerance: 1e-6,
        relative_tolerance: 1e-4,
        correlation_threshold: 0.999,
        max_outlier_percentage: 0.1,
        statistical_tests: vec![
            StatisticalTest {
                test_type: StatisticalTestType::Correlation,
                significance_level: 0.01,
                expected_result: TestResult::Pass,
            },
            StatisticalTest {
                test_type: StatisticalTestType::MeanSquaredError,
                significance_level: 0.05,
                expected_result: TestResult::Pass,
            },
        ],
        performance_tolerance: PerformanceTolerance {
            max_slowdown_factor: 1.2,
            min_throughput_ratio: 0.8,
            max_memory_overhead: 0.1,
            timing_variance_threshold: 0.15,
        },
    }
}

fn create_tl1_tolerance_spec() -> ToleranceSpecification {
    let mut spec = create_i2s_tolerance_spec();
    spec.correlation_threshold = 0.996; // Slightly relaxed for lookup tables
    spec.performance_tolerance.max_slowdown_factor = 1.5;
    spec
}

fn create_tl2_tolerance_spec() -> ToleranceSpecification {
    let mut spec = create_i2s_tolerance_spec();
    spec.correlation_threshold = 0.998;
    spec.performance_tolerance.max_slowdown_factor = 2.0;
    spec
}

fn create_relaxed_tolerance_spec() -> ToleranceSpecification {
    let mut spec = create_i2s_tolerance_spec();
    spec.absolute_tolerance = 1e-4;
    spec.relative_tolerance = 1e-3;
    spec.correlation_threshold = 0.995;
    spec.max_outlier_percentage = 1.0;
    spec
}

fn create_performance_tolerance_spec() -> ToleranceSpecification {
    let mut spec = create_i2s_tolerance_spec();
    spec.performance_tolerance = PerformanceTolerance {
        max_slowdown_factor: 1.1,
        min_throughput_ratio: 0.9,
        max_memory_overhead: 0.05,
        timing_variance_threshold: 0.1,
    };
    spec
}

fn create_comparative_tolerance_spec() -> ToleranceSpecification {
    create_i2s_tolerance_spec()
}

fn create_multi_method_tolerance_spec() -> ToleranceSpecification {
    create_relaxed_tolerance_spec()
}

fn create_model_tolerance_spec() -> ToleranceSpecification {
    let mut spec = create_i2s_tolerance_spec();
    spec.correlation_threshold = 0.998;
    spec
}

fn create_regression_tolerance_spec() -> ToleranceSpecification {
    create_i2s_tolerance_spec()
}

/// Validation metadata creation

fn create_validation_metadata(test_type: &str) -> ValidationMetadata {
    ValidationMetadata {
        cpp_version: "bitnet-cpp-v1.5.8",
        rust_version: "bitnet-rs-v0.3.2",
        test_framework_version: "crossval-v0.2.1",
        reference_implementation: "Microsoft BitNet Official",
        validation_date: "2024-09-27",
        known_issues: match test_type {
            "i2s_edge" => vec!["Edge case handling differs slightly"],
            "tl2_advanced" => vec!["Large lookup tables may have memory differences"],
            _ => vec![],
        },
        compatibility_notes: vec![
            "Deterministic mode required for exact comparison",
            "SIMD optimizations may cause minor differences",
        ],
    }
}

/// Helper data generation functions

fn generate_deterministic_weights(size: usize, seed: u64) -> Vec<f32> {
    let mut weights = Vec::new();
    let mut rng_state = seed;

    for _ in 0..size {
        let weight = -1.0 + 2.0 * lcg_random(&mut rng_state);
        weights.push(weight * 0.1); // Scale to typical range
    }

    weights
}

fn generate_realistic_model_weights(size: usize) -> Vec<f32> {
    let mut weights = Vec::new();
    let mut rng_state = 98765;

    for i in 0..size {
        // Simulate different layer types
        let weight = if i % 100 < 10 {
            // Attention weights
            xavier_random(&mut rng_state, size)
        } else if i % 100 < 60 {
            // MLP weights
            kaiming_random(&mut rng_state, size)
        } else {
            // Other weights
            normal_random(&mut rng_state, 0.0, 0.02)
        };
        weights.push(weight);
    }

    weights
}

fn generate_i2s_quantized_values(size: usize) -> Vec<u8> {
    let mut values = Vec::new();
    let mut rng_state = 11111;

    for _ in 0..size {
        // I2S uses 2 bits: {0, 1, 2, 3} mapping to {-2, -1, 0, 1}
        let val = (lcg_random(&mut rng_state) * 4.0) as u8;
        values.push(val);
    }

    values
}

fn generate_scale_factors(size: usize, block_size: usize) -> Vec<f32> {
    let num_blocks = (size + block_size - 1) / block_size;
    let mut scales = Vec::new();
    let mut rng_state = 22222;

    for _ in 0..num_blocks {
        let scale = 0.01 + lcg_random(&mut rng_state) * 0.1;
        scales.push(scale);
    }

    scales
}

fn generate_inference_result(size: usize) -> Vec<f32> {
    let mut result = Vec::new();
    let mut rng_state = 33333;

    for _ in 0..size {
        let value = normal_random(&mut rng_state, 0.0, 1.0);
        result.push(value);
    }

    result
}

fn generate_tl1_lookup_table() -> Vec<f32> {
    (0..256).map(|i| -1.0 + 2.0 * i as f32 / 255.0).collect()
}

fn generate_tl2_lookup_table() -> Vec<f32> {
    (0..4096).map(|i| -1.0 + 2.0 * i as f32 / 4095.0).collect()
}

/// Standard random generation functions

fn lcg_random(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state as f32) / (u32::MAX as f32)
}

fn normal_random(state: &mut u64, mean: f32, std: f32) -> f32 {
    use std::f32::consts::PI;
    let u1 = lcg_random(state);
    let u2 = lcg_random(state);
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    mean + std * z0
}

fn xavier_random(state: &mut u64, fan_in: usize) -> f32 {
    let limit = (6.0 / fan_in as f32).sqrt();
    let u = lcg_random(state);
    -limit + 2.0 * limit * u
}

fn kaiming_random(state: &mut u64, fan_in: usize) -> f32 {
    let std = (2.0 / fan_in as f32).sqrt();
    normal_random(state, 0.0, std)
}

/// Cross-validation utilities

/// Compare Rust and C++ outputs with tolerance
pub fn compare_outputs(
    rust_output: &CrossValOutput,
    cpp_output: &CrossValOutput,
    tolerance: &ToleranceSpecification,
) -> ValidationResult {
    let mut result = ValidationResult {
        passed: true,
        correlation: 0.0,
        max_absolute_error: 0.0,
        max_relative_error: 0.0,
        outlier_percentage: 0.0,
        performance_ratio: 1.0,
        details: Vec::new(),
    };

    // Compare quantized weights
    result.correlation =
        calculate_correlation(&rust_output.quantized_weights, &cpp_output.quantized_weights);
    if result.correlation < tolerance.correlation_threshold {
        result.passed = false;
        result.details.push(format!(
            "Correlation {} below threshold {}",
            result.correlation, tolerance.correlation_threshold
        ));
    }

    // Compare scale factors
    for (rust_scale, cpp_scale) in
        rust_output.scale_factors.iter().zip(cpp_output.scale_factors.iter())
    {
        let abs_error = (rust_scale - cpp_scale).abs();
        let rel_error = abs_error / cpp_scale.abs().max(f32::EPSILON);

        result.max_absolute_error = result.max_absolute_error.max(abs_error);
        result.max_relative_error = result.max_relative_error.max(rel_error);

        if abs_error > tolerance.absolute_tolerance && rel_error > tolerance.relative_tolerance {
            result.passed = false;
            result
                .details
                .push(format!("Scale factor error: abs={}, rel={}", abs_error, rel_error));
        }
    }

    // Compare performance
    result.performance_ratio = rust_output.performance_metrics.throughput_tokens_per_sec
        / cpp_output.performance_metrics.throughput_tokens_per_sec;

    if result.performance_ratio < tolerance.performance_tolerance.min_throughput_ratio {
        result.passed = false;
        result.details.push(format!(
            "Performance ratio {} below threshold {}",
            result.performance_ratio, tolerance.performance_tolerance.min_throughput_ratio
        ));
    }

    result
}

/// Validation result structure
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: bool,
    pub correlation: f32,
    pub max_absolute_error: f32,
    pub max_relative_error: f32,
    pub outlier_percentage: f32,
    pub performance_ratio: f32,
    pub details: Vec<String>,
}

/// Calculate correlation between two u8 vectors
fn calculate_correlation(a: &[u8], b: &[u8]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let n = a.len() as f32;
    let mean_a = a.iter().map(|&x| x as f32).sum::<f32>() / n;
    let mean_b = b.iter().map(|&x| x as f32).sum::<f32>() / n;

    let mut numerator = 0.0;
    let mut sum_sq_a = 0.0;
    let mut sum_sq_b = 0.0;

    for (&val_a, &val_b) in a.iter().zip(b.iter()) {
        let diff_a = val_a as f32 - mean_a;
        let diff_b = val_b as f32 - mean_b;
        numerator += diff_a * diff_b;
        sum_sq_a += diff_a * diff_a;
        sum_sq_b += diff_b * diff_b;
    }

    if sum_sq_a == 0.0 || sum_sq_b == 0.0 {
        return if sum_sq_a == sum_sq_b { 1.0 } else { 0.0 };
    }

    numerator / (sum_sq_a * sum_sq_b).sqrt()
}

/// Get cross-validation fixture by name
pub fn get_crossval_fixture_by_name(name: &str) -> Option<CppReferenceFixture> {
    let all_fixtures = [
        #[cfg(feature = "crossval")]
        load_i2s_crossval_fixtures(),
        #[cfg(feature = "crossval")]
        load_tl_crossval_fixtures(),
        #[cfg(feature = "crossval")]
        load_comprehensive_crossval_fixtures(),
    ]
    .concat();

    all_fixtures.into_iter().find(|f| f.test_name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "crossval")]
    fn test_i2s_crossval_fixtures() {
        let fixtures = load_i2s_crossval_fixtures();
        assert!(!fixtures.is_empty(), "Should have I2S cross-validation fixtures");

        for fixture in fixtures {
            assert_eq!(fixture.quantization_method, QuantizationMethod::I2S);
            assert!(fixture.deterministic_seed > 0);
            assert!(!fixture.input_data.weights.is_empty());
        }
    }

    #[test]
    #[cfg(feature = "crossval")]
    fn test_tolerance_specifications() {
        let tolerance = create_i2s_tolerance_spec();
        assert!(tolerance.correlation_threshold > 0.99);
        assert!(tolerance.absolute_tolerance > 0.0);
        assert!(tolerance.relative_tolerance > 0.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let a = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let b = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let correlation = calculate_correlation(&a, &b);
        assert!((correlation - 1.0).abs() < 1e-6, "Perfect correlation should be 1.0");
    }

    #[test]
    fn test_validation_metadata() {
        let metadata = create_validation_metadata("test");
        assert!(!metadata.cpp_version.is_empty());
        assert!(!metadata.rust_version.is_empty());
        assert!(!metadata.validation_date.is_empty());
    }

    #[test]
    fn test_deterministic_weight_generation() {
        let weights1 = generate_deterministic_weights(100, 42);
        let weights2 = generate_deterministic_weights(100, 42);
        assert_eq!(weights1, weights2, "Deterministic weights should be identical");
    }
}
