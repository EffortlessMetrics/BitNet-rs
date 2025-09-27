//! Mock Detection and Strict Mode Test Fixtures for Issue #260
//!
//! Provides comprehensive test data for detecting mock computation patterns
//! and validating strict mode behavior. Includes statistical fingerprinting,
//! performance characteristics, and environment variable configurations.

#![allow(dead_code)]

use std::collections::HashMap;
use std::env;

/// Mock computation detection test fixture
#[derive(Debug, Clone)]
pub struct MockDetectionFixture {
    pub test_name: &'static str,
    pub computation_data: ComputationData,
    pub expected_mock_probability: f32,
    pub detection_methods: Vec<DetectionMethod>,
    pub confidence_threshold: f32,
    pub false_positive_tolerance: f32,
}

/// Computational data for analysis
#[derive(Debug, Clone)]
pub struct ComputationData {
    pub input_vectors: Vec<Vec<f32>>,
    pub output_vectors: Vec<Vec<f32>>,
    pub computation_times: Vec<f32>, // milliseconds
    pub memory_access_patterns: Vec<MemoryAccess>,
    pub intermediate_values: Option<Vec<Vec<f32>>>,
    pub computation_graph: ComputationGraph,
}

/// Memory access pattern for analysis
#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub address_sequence: Vec<usize>,
    pub access_times: Vec<f32>,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub bandwidth_utilization: f32,
}

/// Computation graph representation
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    pub operations: Vec<Operation>,
    pub data_dependencies: Vec<(usize, usize)>,
    pub parallel_regions: Vec<(usize, usize)>,
    pub optimization_shortcuts: Vec<Shortcut>,
}

/// Individual operation in computation
#[derive(Debug, Clone)]
pub struct Operation {
    pub op_type: OperationType,
    pub input_shapes: Vec<(usize, usize)>,
    pub output_shape: (usize, usize),
    pub flops: usize,
    pub is_optimized: bool,
    pub uses_simd: bool,
}

/// Types of operations
#[derive(Debug, Clone, Copy)]
pub enum OperationType {
    MatrixMultiply,
    Quantization,
    Dequantization,
    LookupTable,
    BiasAdd,
    Activation,
    Copy,
    Zero,
}

/// Optimization shortcuts that might indicate mocking
#[derive(Debug, Clone)]
pub struct Shortcut {
    pub shortcut_type: ShortcutType,
    pub conditions: Vec<&'static str>,
    pub performance_impact: f32,
    pub suspicious_score: f32,
}

/// Types of computation shortcuts
#[derive(Debug, Clone, Copy)]
pub enum ShortcutType {
    ZeroMultiplication,
    IdentityTransform,
    LookupBypass,
    ConstantFolding,
    PatternReuse,
    CacheReplay,
}

/// Mock detection methods
#[derive(Debug, Clone, Copy)]
pub enum DetectionMethod {
    StatisticalAnalysis,
    PerformanceFingerprinting,
    OutputPatternAnalysis,
    MemoryAccessAnalysis,
    ComputationGraphAnalysis,
    TimingAnalysis,
}

/// Strict mode configuration test fixture
#[derive(Debug, Clone)]
pub struct StrictModeFixture {
    pub scenario: &'static str,
    pub environment_variables: HashMap<String, String>,
    pub expected_behavior: StrictModeBehavior,
    pub test_conditions: Vec<TestCondition>,
    pub validation_criteria: ValidationCriteria,
}

/// Strict mode behavior types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrictModeBehavior {
    FailFast,
    WarnAndContinue,
    SilentFallback,
    DisallowMocks,
    RequireRealComputation,
}

/// Test conditions for strict mode
#[derive(Debug, Clone)]
pub struct TestCondition {
    pub condition_name: &'static str,
    pub setup_action: SetupAction,
    pub expected_result: ExpectedResult,
    pub error_pattern: Option<&'static str>,
}

/// Setup actions for test conditions
#[derive(Debug, Clone)]
pub enum SetupAction {
    EnableStrictMode,
    DisableStrictMode,
    SetEnvironmentVariable(String, String),
    RemoveEnvironmentVariable(String),
    SimulateMissingKernel,
    SimulateCorruptedData,
    ForceGpuFallback,
    ForceCpuFallback,
}

/// Expected test results
#[derive(Debug, Clone, Copy)]
pub enum ExpectedResult {
    Success,
    Error,
    Warning,
    Fallback,
    PerformanceDegradation,
}

/// Validation criteria for strict mode
#[derive(Debug, Clone)]
pub struct ValidationCriteria {
    pub allow_fallbacks: bool,
    pub require_quantization: bool,
    pub max_performance_degradation: f32,
    pub min_accuracy_threshold: f32,
    pub error_tolerance: ErrorTolerance,
}

/// Error tolerance configuration
#[derive(Debug, Clone)]
pub struct ErrorTolerance {
    pub allow_recoverable_errors: bool,
    pub max_retry_attempts: usize,
    pub timeout_seconds: f32,
    pub critical_error_types: Vec<&'static str>,
}

/// Statistical analysis fixture for mock detection
#[derive(Debug, Clone)]
pub struct StatisticalAnalysisFixture {
    pub analysis_type: &'static str,
    pub sample_data: Vec<f32>,
    pub expected_distribution: DistributionType,
    pub anomaly_threshold: f32,
    pub statistical_tests: Vec<StatisticalTest>,
    pub mock_indicators: Vec<MockIndicator>,
}

/// Distribution types for statistical analysis
#[derive(Debug, Clone, Copy)]
pub enum DistributionType {
    Normal,
    Uniform,
    Deterministic,
    Bimodal,
    Sparse,
    Suspicious,
}

/// Statistical tests for mock detection
#[derive(Debug, Clone)]
pub struct StatisticalTest {
    pub test_name: &'static str,
    pub test_statistic: f32,
    pub p_value: f32,
    pub critical_value: f32,
    pub interpretation: TestInterpretation,
}

/// Test interpretation results
#[derive(Debug, Clone, Copy)]
pub enum TestInterpretation {
    LikelyReal,
    LikelyMock,
    Inconclusive,
    RequiresMoreData,
}

/// Mock indicators for pattern detection
#[derive(Debug, Clone)]
pub struct MockIndicator {
    pub indicator_name: &'static str,
    pub confidence_score: f32,
    pub evidence: Vec<&'static str>,
    pub counter_evidence: Vec<&'static str>,
}

/// Load mock detection test fixtures
pub fn load_mock_detection_fixtures() -> Vec<MockDetectionFixture> {
    vec![
        // Clear mock computation pattern
        MockDetectionFixture {
            test_name: "obvious_mock_pattern",
            computation_data: create_mock_computation_data(),
            expected_mock_probability: 0.95,
            detection_methods: vec![
                DetectionMethod::StatisticalAnalysis,
                DetectionMethod::OutputPatternAnalysis,
                DetectionMethod::TimingAnalysis,
            ],
            confidence_threshold: 0.9,
            false_positive_tolerance: 0.01,
        },
        // Real computation pattern
        MockDetectionFixture {
            test_name: "real_computation_pattern",
            computation_data: create_real_computation_data(),
            expected_mock_probability: 0.05,
            detection_methods: vec![
                DetectionMethod::PerformanceFingerprinting,
                DetectionMethod::MemoryAccessAnalysis,
                DetectionMethod::ComputationGraphAnalysis,
            ],
            confidence_threshold: 0.9,
            false_positive_tolerance: 0.05,
        },
        // Borderline case - optimized real computation
        MockDetectionFixture {
            test_name: "optimized_real_computation",
            computation_data: create_optimized_computation_data(),
            expected_mock_probability: 0.25,
            detection_methods: vec![
                DetectionMethod::StatisticalAnalysis,
                DetectionMethod::PerformanceFingerprinting,
            ],
            confidence_threshold: 0.8,
            false_positive_tolerance: 0.1,
        },
        // Sophisticated mock - harder to detect
        MockDetectionFixture {
            test_name: "sophisticated_mock",
            computation_data: create_sophisticated_mock_data(),
            expected_mock_probability: 0.7,
            detection_methods: vec![
                DetectionMethod::MemoryAccessAnalysis,
                DetectionMethod::ComputationGraphAnalysis,
                DetectionMethod::TimingAnalysis,
            ],
            confidence_threshold: 0.85,
            false_positive_tolerance: 0.15,
        },
        // Corrupted/noisy data
        MockDetectionFixture {
            test_name: "corrupted_data_analysis",
            computation_data: create_corrupted_computation_data(),
            expected_mock_probability: 0.4, // Inconclusive
            detection_methods: vec![DetectionMethod::StatisticalAnalysis],
            confidence_threshold: 0.7,
            false_positive_tolerance: 0.2,
        },
    ]
}

/// Load strict mode configuration fixtures
pub fn load_strict_mode_fixtures() -> Vec<StrictModeFixture> {
    vec![
        // Basic strict mode enabled
        StrictModeFixture {
            scenario: "strict_mode_enabled",
            environment_variables: {
                let mut env = HashMap::new();
                env.insert("BITNET_STRICT_MODE".to_string(), "1".to_string());
                env
            },
            expected_behavior: StrictModeBehavior::FailFast,
            test_conditions: vec![
                TestCondition {
                    condition_name: "mock_computation_detected",
                    setup_action: SetupAction::SimulateMissingKernel,
                    expected_result: ExpectedResult::Error,
                    error_pattern: Some("Strict mode: Mock computation detected"),
                },
                TestCondition {
                    condition_name: "fallback_attempted",
                    setup_action: SetupAction::ForceGpuFallback,
                    expected_result: ExpectedResult::Error,
                    error_pattern: Some("Strict mode: Fallback not allowed"),
                },
            ],
            validation_criteria: ValidationCriteria {
                allow_fallbacks: false,
                require_quantization: true,
                max_performance_degradation: 0.0,
                min_accuracy_threshold: 0.999,
                error_tolerance: ErrorTolerance {
                    allow_recoverable_errors: false,
                    max_retry_attempts: 0,
                    timeout_seconds: 10.0,
                    critical_error_types: vec!["MockComputationError", "FallbackError"],
                },
            },
        },
        // Strict mode disabled - permissive
        StrictModeFixture {
            scenario: "strict_mode_disabled",
            environment_variables: {
                let mut env = HashMap::new();
                env.insert("BITNET_STRICT_MODE".to_string(), "0".to_string());
                env
            },
            expected_behavior: StrictModeBehavior::SilentFallback,
            test_conditions: vec![
                TestCondition {
                    condition_name: "fallback_allowed",
                    setup_action: SetupAction::ForceGpuFallback,
                    expected_result: ExpectedResult::Success,
                    error_pattern: None,
                },
                TestCondition {
                    condition_name: "mock_computation_tolerated",
                    setup_action: SetupAction::SimulateMissingKernel,
                    expected_result: ExpectedResult::Success,
                    error_pattern: None,
                },
            ],
            validation_criteria: ValidationCriteria {
                allow_fallbacks: true,
                require_quantization: false,
                max_performance_degradation: 0.5,
                min_accuracy_threshold: 0.95,
                error_tolerance: ErrorTolerance {
                    allow_recoverable_errors: true,
                    max_retry_attempts: 3,
                    timeout_seconds: 30.0,
                    critical_error_types: vec!["SystemError"],
                },
            },
        },
        // Warning mode - strict but not failing
        StrictModeFixture {
            scenario: "warning_mode",
            environment_variables: {
                let mut env = HashMap::new();
                env.insert("BITNET_STRICT_MODE".to_string(), "warn".to_string());
                env
            },
            expected_behavior: StrictModeBehavior::WarnAndContinue,
            test_conditions: vec![
                TestCondition {
                    condition_name: "mock_detected_warning",
                    setup_action: SetupAction::SimulateMissingKernel,
                    expected_result: ExpectedResult::Warning,
                    error_pattern: Some("Warning: Mock computation detected"),
                },
                TestCondition {
                    condition_name: "performance_degradation_warning",
                    setup_action: SetupAction::ForceCpuFallback,
                    expected_result: ExpectedResult::Warning,
                    error_pattern: Some("Warning: Performance degradation"),
                },
            ],
            validation_criteria: ValidationCriteria {
                allow_fallbacks: true,
                require_quantization: false,
                max_performance_degradation: 0.3,
                min_accuracy_threshold: 0.98,
                error_tolerance: ErrorTolerance {
                    allow_recoverable_errors: true,
                    max_retry_attempts: 2,
                    timeout_seconds: 20.0,
                    critical_error_types: vec!["FatalError"],
                },
            },
        },
        // Custom strict configuration
        StrictModeFixture {
            scenario: "custom_strict_config",
            environment_variables: {
                let mut env = HashMap::new();
                env.insert("BITNET_STRICT_MODE".to_string(), "1".to_string());
                env.insert("BITNET_STRICT_QUANTIZATION".to_string(), "1".to_string());
                env.insert("BITNET_STRICT_NO_FALLBACKS".to_string(), "1".to_string());
                env.insert("BITNET_STRICT_NO_FAKE_GPU".to_string(), "1".to_string());
                env
            },
            expected_behavior: StrictModeBehavior::RequireRealComputation,
            test_conditions: vec![
                TestCondition {
                    condition_name: "fake_gpu_detected",
                    setup_action: SetupAction::SetEnvironmentVariable(
                        "CUDA_VISIBLE_DEVICES".to_string(),
                        "-1".to_string(),
                    ),
                    expected_result: ExpectedResult::Error,
                    error_pattern: Some("Strict mode: Fake GPU environment detected"),
                },
                TestCondition {
                    condition_name: "quantization_required",
                    setup_action: SetupAction::SimulateCorruptedData,
                    expected_result: ExpectedResult::Error,
                    error_pattern: Some("Strict mode: Real quantization required"),
                },
            ],
            validation_criteria: ValidationCriteria {
                allow_fallbacks: false,
                require_quantization: true,
                max_performance_degradation: 0.0,
                min_accuracy_threshold: 0.9999,
                error_tolerance: ErrorTolerance {
                    allow_recoverable_errors: false,
                    max_retry_attempts: 0,
                    timeout_seconds: 5.0,
                    critical_error_types: vec!["MockError", "FallbackError", "FakeGpuError"],
                },
            },
        },
    ]
}

/// Load statistical analysis fixtures
pub fn load_statistical_analysis_fixtures() -> Vec<StatisticalAnalysisFixture> {
    vec![
        // Normal computation distribution
        StatisticalAnalysisFixture {
            analysis_type: "normal_computation_outputs",
            sample_data: generate_normal_distribution(1000, 0.0, 1.0),
            expected_distribution: DistributionType::Normal,
            anomaly_threshold: 0.05,
            statistical_tests: vec![
                StatisticalTest {
                    test_name: "kolmogorov_smirnov",
                    test_statistic: 0.03,
                    p_value: 0.8,
                    critical_value: 0.05,
                    interpretation: TestInterpretation::LikelyReal,
                },
                StatisticalTest {
                    test_name: "shapiro_wilk",
                    test_statistic: 0.995,
                    p_value: 0.6,
                    critical_value: 0.05,
                    interpretation: TestInterpretation::LikelyReal,
                },
            ],
            mock_indicators: vec![MockIndicator {
                indicator_name: "variance_check",
                confidence_score: 0.9,
                evidence: vec!["normal_variance", "expected_mean"],
                counter_evidence: vec![],
            }],
        },
        // Suspicious deterministic pattern
        StatisticalAnalysisFixture {
            analysis_type: "deterministic_mock_pattern",
            sample_data: generate_deterministic_pattern(1000),
            expected_distribution: DistributionType::Deterministic,
            anomaly_threshold: 0.01,
            statistical_tests: vec![
                StatisticalTest {
                    test_name: "runs_test",
                    test_statistic: 0.0,
                    p_value: 0.001,
                    critical_value: 0.05,
                    interpretation: TestInterpretation::LikelyMock,
                },
                StatisticalTest {
                    test_name: "autocorrelation",
                    test_statistic: 0.99,
                    p_value: 0.0,
                    critical_value: 0.1,
                    interpretation: TestInterpretation::LikelyMock,
                },
            ],
            mock_indicators: vec![
                MockIndicator {
                    indicator_name: "perfect_correlation",
                    confidence_score: 0.95,
                    evidence: vec!["zero_variance", "perfect_periodicity"],
                    counter_evidence: vec![],
                },
                MockIndicator {
                    indicator_name: "unrealistic_precision",
                    confidence_score: 0.88,
                    evidence: vec!["excessive_decimal_precision", "artificial_patterns"],
                    counter_evidence: vec![],
                },
            ],
        },
        // Quantization artifact analysis
        StatisticalAnalysisFixture {
            analysis_type: "quantization_artifacts",
            sample_data: generate_quantized_distribution(1000),
            expected_distribution: DistributionType::Sparse,
            anomaly_threshold: 0.1,
            statistical_tests: vec![StatisticalTest {
                test_name: "discretization_check",
                test_statistic: 0.8,
                p_value: 0.2,
                critical_value: 0.05,
                interpretation: TestInterpretation::LikelyReal,
            }],
            mock_indicators: vec![MockIndicator {
                indicator_name: "quantization_levels",
                confidence_score: 0.7,
                evidence: vec!["discrete_values", "expected_range"],
                counter_evidence: vec!["some_noise"],
            }],
        },
    ]
}

/// Helper functions to create test data
fn create_mock_computation_data() -> ComputationData {
    ComputationData {
        input_vectors: vec![vec![1.0; 256], vec![0.5; 256], vec![2.0; 256]],
        output_vectors: vec![
            vec![1.0; 256], // Suspicious: input == output
            vec![0.5; 256],
            vec![2.0; 256],
        ],
        computation_times: vec![0.001, 0.001, 0.001], // Suspiciously consistent
        memory_access_patterns: vec![MemoryAccess {
            address_sequence: vec![0, 4, 8, 12], // Too regular
            access_times: vec![1.0; 4],
            cache_hits: 4,
            cache_misses: 0,            // Suspiciously perfect
            bandwidth_utilization: 0.1, // Too low
        }],
        intermediate_values: None, // Missing intermediate computation
        computation_graph: ComputationGraph {
            operations: vec![Operation {
                op_type: OperationType::Copy, // Suspicious: just copying
                input_shapes: vec![(256, 1)],
                output_shape: (256, 1),
                flops: 0, // No computation
                is_optimized: false,
                uses_simd: false,
            }],
            data_dependencies: vec![],
            parallel_regions: vec![],
            optimization_shortcuts: vec![Shortcut {
                shortcut_type: ShortcutType::IdentityTransform,
                conditions: vec!["always"],
                performance_impact: 1.0,
                suspicious_score: 0.9,
            }],
        },
    }
}

fn create_real_computation_data() -> ComputationData {
    let mut rng_state = 12345;

    ComputationData {
        input_vectors: vec![
            generate_realistic_vector(256, &mut rng_state),
            generate_realistic_vector(256, &mut rng_state),
            generate_realistic_vector(256, &mut rng_state),
        ],
        output_vectors: vec![
            generate_realistic_vector(256, &mut rng_state),
            generate_realistic_vector(256, &mut rng_state),
            generate_realistic_vector(256, &mut rng_state),
        ],
        computation_times: vec![1.2, 1.1, 1.3], // Realistic variation
        memory_access_patterns: vec![MemoryAccess {
            address_sequence: generate_realistic_addresses(100),
            access_times: generate_realistic_times(100),
            cache_hits: 75,
            cache_misses: 25,
            bandwidth_utilization: 0.8,
        }],
        intermediate_values: Some(vec![
            generate_realistic_vector(128, &mut rng_state),
            generate_realistic_vector(64, &mut rng_state),
        ]),
        computation_graph: ComputationGraph {
            operations: vec![
                Operation {
                    op_type: OperationType::MatrixMultiply,
                    input_shapes: vec![(256, 256), (256, 256)],
                    output_shape: (256, 256),
                    flops: 256 * 256 * 256 * 2,
                    is_optimized: true,
                    uses_simd: true,
                },
                Operation {
                    op_type: OperationType::Quantization,
                    input_shapes: vec![(256, 256)],
                    output_shape: (256, 256),
                    flops: 256 * 256,
                    is_optimized: true,
                    uses_simd: true,
                },
            ],
            data_dependencies: vec![(0, 1)],
            parallel_regions: vec![(0, 2)],
            optimization_shortcuts: vec![],
        },
    }
}

fn create_optimized_computation_data() -> ComputationData {
    let mut rng_state = 23456;

    ComputationData {
        input_vectors: vec![generate_realistic_vector(256, &mut rng_state)],
        output_vectors: vec![generate_realistic_vector(256, &mut rng_state)],
        computation_times: vec![0.1], // Very fast due to optimization
        memory_access_patterns: vec![MemoryAccess {
            address_sequence: generate_blocked_addresses(64),
            access_times: vec![0.5; 64],
            cache_hits: 60,
            cache_misses: 4,
            bandwidth_utilization: 0.95,
        }],
        intermediate_values: Some(vec![]),
        computation_graph: ComputationGraph {
            operations: vec![Operation {
                op_type: OperationType::LookupTable,
                input_shapes: vec![(256, 1)],
                output_shape: (256, 1),
                flops: 256, // Lookup operations
                is_optimized: true,
                uses_simd: true,
            }],
            data_dependencies: vec![],
            parallel_regions: vec![],
            optimization_shortcuts: vec![Shortcut {
                shortcut_type: ShortcutType::CacheReplay,
                conditions: vec!["cache_hit_rate > 0.9"],
                performance_impact: 0.1,
                suspicious_score: 0.3,
            }],
        },
    }
}

fn create_sophisticated_mock_data() -> ComputationData {
    let mut rng_state = 34567;

    ComputationData {
        input_vectors: vec![generate_realistic_vector(256, &mut rng_state)],
        output_vectors: vec![add_subtle_patterns(generate_realistic_vector(256, &mut rng_state))],
        computation_times: vec![0.8], // Realistic timing
        memory_access_patterns: vec![MemoryAccess {
            address_sequence: generate_realistic_addresses(80),
            access_times: generate_realistic_times(80),
            cache_hits: 65,
            cache_misses: 15,
            bandwidth_utilization: 0.7,
        }],
        intermediate_values: Some(vec![generate_realistic_vector(128, &mut rng_state)]),
        computation_graph: ComputationGraph {
            operations: vec![Operation {
                op_type: OperationType::MatrixMultiply,
                input_shapes: vec![(256, 128)],
                output_shape: (256, 128),
                flops: 256 * 128 * 128,
                is_optimized: true,
                uses_simd: true,
            }],
            data_dependencies: vec![],
            parallel_regions: vec![],
            optimization_shortcuts: vec![Shortcut {
                shortcut_type: ShortcutType::PatternReuse,
                conditions: vec!["pattern_detected"],
                performance_impact: 0.2,
                suspicious_score: 0.6, // Moderately suspicious
            }],
        },
    }
}

fn create_corrupted_computation_data() -> ComputationData {
    let mut rng_state = 45678;

    ComputationData {
        input_vectors: vec![add_corruption(generate_realistic_vector(256, &mut rng_state))],
        output_vectors: vec![add_corruption(generate_realistic_vector(256, &mut rng_state))],
        computation_times: vec![f32::NAN], // Corrupted timing
        memory_access_patterns: vec![MemoryAccess {
            address_sequence: vec![], // Missing data
            access_times: vec![],
            cache_hits: 0,
            cache_misses: 0,
            bandwidth_utilization: f32::NAN,
        }],
        intermediate_values: None,
        computation_graph: ComputationGraph {
            operations: vec![],
            data_dependencies: vec![],
            parallel_regions: vec![],
            optimization_shortcuts: vec![],
        },
    }
}

/// Helper functions for data generation
fn generate_realistic_vector(size: usize, rng_state: &mut u64) -> Vec<f32> {
    (0..size).map(|_| normal_random(rng_state, 0.0, 1.0)).collect()
}

fn generate_realistic_addresses(count: usize) -> Vec<usize> {
    let mut addresses = Vec::new();
    let mut addr = 0x1000;
    let mut rng_state = 11111;

    for _ in 0..count {
        addresses.push(addr);
        addr += (4 + (lcg_random(&mut rng_state) * 64.0) as usize) & !3; // Aligned addresses with some randomness
    }

    addresses
}

fn generate_realistic_times(count: usize) -> Vec<f32> {
    let mut times = Vec::new();
    let mut rng_state = 22222;

    for _ in 0..count {
        let time = 0.5 + lcg_random(&mut rng_state) * 2.0; // 0.5-2.5ms range
        times.push(time);
    }

    times
}

fn generate_blocked_addresses(count: usize) -> Vec<usize> {
    let mut addresses = Vec::new();
    let block_size = 64;

    for i in 0..count {
        let block = i / block_size;
        let offset = i % block_size;
        addresses.push(0x1000 + block * 256 + offset * 4);
    }

    addresses
}

fn add_subtle_patterns(mut vec: Vec<f32>) -> Vec<f32> {
    // Add subtle but detectable patterns
    for (i, item) in vec.iter_mut().enumerate() {
        if i % 16 == 0 {
            *item *= 1.001; // Tiny but consistent bias
        }
    }
    vec
}

fn add_corruption(mut vec: Vec<f32>) -> Vec<f32> {
    // Add some NaN and infinite values
    if vec.len() > 10 {
        vec[5] = f32::NAN;
        let len = vec.len();
        vec[len - 1] = f32::INFINITY;
    }
    vec
}

fn generate_normal_distribution(size: usize, mean: f32, std: f32) -> Vec<f32> {
    let mut data = Vec::new();
    let mut rng_state = 33333;

    for _ in 0..size {
        data.push(normal_random(&mut rng_state, mean, std));
    }

    data
}

fn generate_deterministic_pattern(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32 * 0.1).sin()).collect()
}

fn generate_quantized_distribution(size: usize) -> Vec<f32> {
    let mut data = Vec::new();
    let mut rng_state = 44444;
    let levels = [-2.0, -1.0, 0.0, 1.0]; // I2S quantization levels

    for _ in 0..size {
        let level_idx = (lcg_random(&mut rng_state) * 4.0) as usize;
        let base_value = levels[level_idx.min(3)];
        let noise = normal_random(&mut rng_state, 0.0, 0.01); // Small quantization noise
        data.push(base_value + noise);
    }

    data
}

/// Standard helper functions
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

/// Validate strict mode configuration
pub fn validate_strict_mode_config() -> Result<StrictModeBehavior, String> {
    let strict_mode = env::var("BITNET_STRICT_MODE").unwrap_or_default();

    match strict_mode.as_str() {
        "1" | "true" => Ok(StrictModeBehavior::FailFast),
        "0" | "false" | "" => Ok(StrictModeBehavior::SilentFallback),
        "warn" => Ok(StrictModeBehavior::WarnAndContinue),
        _ => Err(format!("Invalid BITNET_STRICT_MODE value: {}", strict_mode)),
    }
}

/// Check if mock computation should be detected
pub fn should_detect_mocks() -> bool {
    env::var("BITNET_STRICT_MODE").unwrap_or_default() != "0"
}

/// Get mock detection threshold from environment
pub fn get_mock_detection_threshold() -> f32 {
    env::var("BITNET_MOCK_DETECTION_THRESHOLD").unwrap_or_default().parse().unwrap_or(0.8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_detection_fixtures() {
        let fixtures = load_mock_detection_fixtures();
        assert!(!fixtures.is_empty(), "Should have mock detection fixtures");

        for fixture in fixtures {
            assert!(
                fixture.expected_mock_probability >= 0.0
                    && fixture.expected_mock_probability <= 1.0
            );
            assert!(fixture.confidence_threshold > 0.0 && fixture.confidence_threshold <= 1.0);
        }
    }

    #[test]
    fn test_strict_mode_fixtures() {
        let fixtures = load_strict_mode_fixtures();
        assert!(!fixtures.is_empty(), "Should have strict mode fixtures");

        for fixture in fixtures {
            assert!(!fixture.test_conditions.is_empty(), "Should have test conditions");
        }
    }

    #[test]
    fn test_statistical_analysis_fixtures() {
        let fixtures = load_statistical_analysis_fixtures();
        assert!(!fixtures.is_empty(), "Should have statistical analysis fixtures");

        for fixture in fixtures {
            assert!(!fixture.sample_data.is_empty(), "Should have sample data");
            assert!(!fixture.statistical_tests.is_empty(), "Should have statistical tests");
        }
    }

    #[test]
    fn test_strict_mode_config_validation() {
        // Test valid configurations
        unsafe {
            env::set_var("BITNET_STRICT_MODE", "1");
            assert_eq!(validate_strict_mode_config().unwrap(), StrictModeBehavior::FailFast);

            env::set_var("BITNET_STRICT_MODE", "warn");
            assert_eq!(validate_strict_mode_config().unwrap(), StrictModeBehavior::WarnAndContinue);

            env::remove_var("BITNET_STRICT_MODE");
            assert_eq!(validate_strict_mode_config().unwrap(), StrictModeBehavior::SilentFallback);
        }
    }

    #[test]
    fn test_mock_detection_threshold() {
        unsafe {
            env::set_var("BITNET_MOCK_DETECTION_THRESHOLD", "0.9");
            assert_eq!(get_mock_detection_threshold(), 0.9);

            env::remove_var("BITNET_MOCK_DETECTION_THRESHOLD");
            assert_eq!(get_mock_detection_threshold(), 0.8); // Default
        }
    }
}
