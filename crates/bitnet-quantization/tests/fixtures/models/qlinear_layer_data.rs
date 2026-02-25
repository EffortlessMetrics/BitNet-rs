//! QLinear Layer Test Fixtures for Issue #260 Mock Elimination
//!
//! Provides realistic test data for QLinear layer replacement validation.
//! Includes GGUF model compatibility, quantized layer configurations,
//! and integration test scenarios for all quantization types.

#![allow(unused_imports)]
#![allow(dead_code)]

use std::collections::HashMap;

/// QLinear layer configuration for testing
#[derive(Debug, Clone)]
pub struct QLinearLayerFixture {
    pub layer_name: &'static str,
    pub input_shape: (usize, usize),
    pub output_shape: (usize, usize),
    pub quantization_type: QuantizationType,
    pub weight_data: Vec<f32>,
    pub bias_data: Option<Vec<f32>>,
    pub quantized_weights: QuantizedWeightData,
    pub expected_output_range: (f32, f32),
    pub layer_type: LayerType,
    pub gguf_compatible: bool,
}

/// Quantization type for layer testing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    I2S,
    TL1,
    TL2,
    IQ2S, // GGML-compatible via FFI
}

/// Layer type classification
#[derive(Debug, Clone, Copy)]
pub enum LayerType {
    Attention,
    Mlp,
    Embedding,
    Output,
    Norm,
}

/// Quantized weight data structure
#[derive(Debug, Clone)]
pub struct QuantizedWeightData {
    pub quantized_values: Vec<u8>,
    pub scales: Vec<f32>,
    pub block_size: usize,
    pub compression_ratio: f32,
    pub quantization_params: QuantizationParams,
}

/// Quantization parameters for different methods
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    pub method: QuantizationType,
    pub block_size: usize,
    pub lookup_table: Option<Vec<f32>>,
    pub zero_point: Option<i32>,
    pub clamp_range: Option<(i32, i32)>,
}

/// GGUF model test fixture
#[derive(Debug, Clone)]
pub struct GgufModelFixture {
    pub model_name: &'static str,
    pub file_size_bytes: usize,
    pub tensor_count: usize,
    pub vocab_size: u32,
    pub context_length: u32,
    pub model_type: &'static str,
    pub quantization_types: Vec<QuantizationType>,
    pub layers: Vec<QLinearLayerFixture>,
    pub weight_mapper_compatible: bool,
    pub tensor_alignment: usize,
}

/// Integration test scenario for layer replacement
#[derive(Debug, Clone)]
pub struct LayerReplacementScenario {
    pub scenario_name: &'static str,
    pub original_layer: MockLinearLayer,
    pub replacement_layer: QLinearLayerFixture,
    pub test_inputs: Vec<Vec<f32>>,
    pub expected_outputs: Vec<Vec<f32>>,
    pub tolerance: f32,
    pub performance_target: PerformanceTarget,
    pub mock_detection_required: bool,
}

/// Mock linear layer for comparison
#[derive(Debug, Clone)]
pub struct MockLinearLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub uses_mock_computation: bool,
    pub mock_fingerprint: MockFingerprint,
}

/// Mock computation fingerprint for detection
#[derive(Debug, Clone)]
pub struct MockFingerprint {
    pub deterministic_pattern: bool,
    pub zero_variance_regions: Vec<(usize, usize)>,
    pub suspicious_correlations: Vec<f32>,
    pub computation_shortcuts: Vec<&'static str>,
}

/// Performance target for layer replacement
#[derive(Debug, Clone)]
pub struct PerformanceTarget {
    pub min_throughput_cpu: f32,  // tok/s
    pub min_throughput_gpu: f32,  // tok/s
    pub max_memory_overhead: f32, // relative to FP32
    pub max_accuracy_loss: f32,   // correlation degradation
}

/// Fallback detection data
#[derive(Debug, Clone)]
pub struct FallbackDetectionData {
    pub scenario: &'static str,
    pub trigger_conditions: Vec<FallbackTrigger>,
    pub expected_fallback_path: FallbackPath,
    pub strict_mode_behavior: StrictModeBehavior,
    pub detection_method: DetectionMethod,
}

/// Fallback trigger conditions
#[derive(Debug, Clone)]
pub enum FallbackTrigger {
    MissingKernel(&'static str),
    UnsupportedDevice,
    InsufficientMemory,
    InvalidTensorShape,
    CorruptedWeights,
}

/// Fallback execution path
#[derive(Debug, Clone, Copy)]
pub enum FallbackPath {
    Dequantization,
    MockComputation,
    CPUFallback,
    ErrorPath,
}

/// Strict mode behavior specification
#[derive(Debug, Clone, Copy)]
pub enum StrictModeBehavior {
    FailFast,
    AllowWithWarning,
    SilentFallback,
}

/// Mock detection method
#[derive(Debug, Clone, Copy)]
pub enum DetectionMethod {
    StatisticalAnalysis,
    ComputationFingerprinting,
    OutputPatternAnalysis,
    PerformanceCharacteristics,
}

/// Load basic QLinear layer test fixtures
pub fn load_qlinear_layer_fixtures() -> Vec<QLinearLayerFixture> {
    vec![
        // Small attention layer
        QLinearLayerFixture {
            layer_name: "attention_query_small",
            input_shape: (1, 256),
            output_shape: (1, 256),
            quantization_type: QuantizationType::I2S,
            weight_data: generate_attention_weights(256, 256),
            bias_data: Some(generate_bias_vector(256)),
            quantized_weights: create_quantized_weights(256 * 256, QuantizationType::I2S, 32),
            expected_output_range: (-2.0, 2.0),
            layer_type: LayerType::Attention,
            gguf_compatible: true,
        },
        // Medium MLP layer with TL1
        QLinearLayerFixture {
            layer_name: "mlp_hidden_medium",
            input_shape: (1, 512),
            output_shape: (1, 2048),
            quantization_type: QuantizationType::TL1,
            weight_data: generate_mlp_weights(512, 2048),
            bias_data: Some(generate_bias_vector(2048)),
            quantized_weights: create_quantized_weights(512 * 2048, QuantizationType::TL1, 64),
            expected_output_range: (-3.0, 3.0),
            layer_type: LayerType::Mlp,
            gguf_compatible: true,
        },
        // Large layer with TL2 (reduced to keep fixture memory <2MB)
        QLinearLayerFixture {
            layer_name: "large_projection_tl2",
            input_shape: (1, 512),
            output_shape: (1, 1024),
            quantization_type: QuantizationType::TL2,
            weight_data: generate_projection_weights(512, 1024),
            bias_data: None, // No bias for some layers
            quantized_weights: create_quantized_weights(512 * 1024, QuantizationType::TL2, 128),
            expected_output_range: (-4.0, 4.0),
            layer_type: LayerType::Mlp,
            gguf_compatible: true,
        },
        // Embedding layer (reduced vocab to keep fixture memory <1MB)
        QLinearLayerFixture {
            layer_name: "embedding_lookup",
            input_shape: (1, 512),
            output_shape: (1, 768),
            quantization_type: QuantizationType::I2S,
            weight_data: generate_embedding_weights(512, 768),
            bias_data: None,
            quantized_weights: create_quantized_weights(512 * 768, QuantizationType::I2S, 64),
            expected_output_range: (-1.0, 1.0),
            layer_type: LayerType::Embedding,
            gguf_compatible: true,
        },
        // Output projection (reduced vocab to keep fixture memory <1MB)
        QLinearLayerFixture {
            layer_name: "output_projection",
            input_shape: (1, 768),
            output_shape: (1, 512),
            quantization_type: QuantizationType::TL1,
            weight_data: generate_output_weights(768, 512),
            bias_data: Some(generate_bias_vector(512)),
            quantized_weights: create_quantized_weights(768 * 512, QuantizationType::TL1, 64),
            expected_output_range: (-10.0, 10.0),
            layer_type: LayerType::Output,
            gguf_compatible: true,
        },
    ]
}

/// Load GGUF model test fixtures
pub fn load_gguf_model_fixtures() -> Vec<GgufModelFixture> {
    // Use tiny hidden_dim (32) to keep fixture allocation under 1MB per model.
    // The metadata (tensor_count, vocab_size, …) still reflects realistic model sizes.
    vec![
        GgufModelFixture {
            model_name: "bitnet_small_1b",
            file_size_bytes: 1024 * 1024 * 512, // 512MB
            tensor_count: 148,
            vocab_size: 50257,
            context_length: 2048,
            model_type: "bitnet",
            quantization_types: vec![QuantizationType::I2S],
            layers: generate_model_layers(2, 32, QuantizationType::I2S),
            weight_mapper_compatible: true,
            tensor_alignment: 32,
        },
        GgufModelFixture {
            model_name: "bitnet_medium_3b",
            file_size_bytes: 1024 * 1024 * 1536, // 1.5GB
            tensor_count: 228,
            vocab_size: 50257,
            context_length: 4096,
            model_type: "bitnet",
            quantization_types: vec![QuantizationType::I2S, QuantizationType::TL1],
            layers: generate_model_layers(2, 32, QuantizationType::TL1),
            weight_mapper_compatible: true,
            tensor_alignment: 32,
        },
        GgufModelFixture {
            model_name: "bitnet_large_7b",
            file_size_bytes: 1024 * 1024 * 3584, // 3.5GB
            tensor_count: 308,
            vocab_size: 50257,
            context_length: 8192,
            model_type: "bitnet",
            quantization_types: vec![
                QuantizationType::I2S,
                QuantizationType::TL1,
                QuantizationType::TL2,
            ],
            layers: generate_model_layers(2, 32, QuantizationType::TL2),
            weight_mapper_compatible: true,
            tensor_alignment: 32,
        },
    ]
}

/// Load layer replacement test scenarios
pub fn load_layer_replacement_scenarios() -> Vec<LayerReplacementScenario> {
    vec![
        LayerReplacementScenario {
            scenario_name: "mock_to_i2s_replacement",
            original_layer: create_mock_layer(256, 256, true),
            replacement_layer: load_qlinear_layer_fixtures()[0].clone(),
            test_inputs: generate_test_inputs(5, 256),
            expected_outputs: generate_expected_outputs(5, 256),
            tolerance: 0.1,
            performance_target: PerformanceTarget {
                min_throughput_cpu: 15.0,
                min_throughput_gpu: 50.0,
                max_memory_overhead: 0.25, // 25% of FP32
                max_accuracy_loss: 0.002,  // 0.2% correlation loss
            },
            mock_detection_required: true,
        },
        LayerReplacementScenario {
            scenario_name: "large_layer_tl2_replacement",
            original_layer: create_mock_layer(512, 1024, true),
            replacement_layer: load_qlinear_layer_fixtures()[2].clone(),
            test_inputs: generate_test_inputs(3, 512),
            expected_outputs: generate_expected_outputs(3, 1024),
            tolerance: 0.05,
            performance_target: PerformanceTarget {
                min_throughput_cpu: 8.0,
                min_throughput_gpu: 30.0,
                max_memory_overhead: 0.2,
                max_accuracy_loss: 0.001,
            },
            mock_detection_required: true,
        },
        LayerReplacementScenario {
            scenario_name: "batch_processing_scenario",
            original_layer: create_mock_layer(512, 2048, false),
            replacement_layer: load_qlinear_layer_fixtures()[1].clone(),
            test_inputs: generate_test_inputs(8, 512), // Batch size 8
            expected_outputs: generate_expected_outputs(8, 2048),
            tolerance: 0.08,
            performance_target: PerformanceTarget {
                min_throughput_cpu: 12.0,
                min_throughput_gpu: 40.0,
                max_memory_overhead: 0.3,
                max_accuracy_loss: 0.0015,
            },
            mock_detection_required: false,
        },
    ]
}

/// Load fallback detection test data
pub fn load_fallback_detection_fixtures() -> Vec<FallbackDetectionData> {
    vec![
        FallbackDetectionData {
            scenario: "missing_i2s_kernel",
            trigger_conditions: vec![FallbackTrigger::MissingKernel("i2s_quantized_matmul")],
            expected_fallback_path: FallbackPath::Dequantization,
            strict_mode_behavior: StrictModeBehavior::FailFast,
            detection_method: DetectionMethod::ComputationFingerprinting,
        },
        FallbackDetectionData {
            scenario: "gpu_memory_insufficient",
            trigger_conditions: vec![FallbackTrigger::InsufficientMemory],
            expected_fallback_path: FallbackPath::CPUFallback,
            strict_mode_behavior: StrictModeBehavior::AllowWithWarning,
            detection_method: DetectionMethod::PerformanceCharacteristics,
        },
        FallbackDetectionData {
            scenario: "corrupted_quantized_weights",
            trigger_conditions: vec![FallbackTrigger::CorruptedWeights],
            expected_fallback_path: FallbackPath::ErrorPath,
            strict_mode_behavior: StrictModeBehavior::FailFast,
            detection_method: DetectionMethod::StatisticalAnalysis,
        },
        FallbackDetectionData {
            scenario: "unsupported_device_fallback",
            trigger_conditions: vec![FallbackTrigger::UnsupportedDevice],
            expected_fallback_path: FallbackPath::CPUFallback,
            strict_mode_behavior: StrictModeBehavior::SilentFallback,
            detection_method: DetectionMethod::OutputPatternAnalysis,
        },
    ]
}

/// Generate realistic weight matrices for different layer types
fn generate_attention_weights(input_dim: usize, output_dim: usize) -> Vec<f32> {
    let mut weights = Vec::with_capacity(input_dim * output_dim);
    let mut rng_state = 12345;

    for _ in 0..(input_dim * output_dim) {
        // Xavier initialization for attention weights
        let limit = (6.0 / (input_dim + output_dim) as f32).sqrt();
        let weight = -limit + 2.0 * limit * lcg_random(&mut rng_state);
        weights.push(weight);
    }

    weights
}

fn generate_mlp_weights(input_dim: usize, output_dim: usize) -> Vec<f32> {
    let mut weights = Vec::with_capacity(input_dim * output_dim);
    let mut rng_state = 23456;

    for _ in 0..(input_dim * output_dim) {
        // Kaiming initialization for MLP weights
        let std = (2.0 / input_dim as f32).sqrt();
        let weight = normal_random(&mut rng_state, 0.0, std);
        weights.push(weight);
    }

    weights
}

fn generate_projection_weights(input_dim: usize, output_dim: usize) -> Vec<f32> {
    let mut weights = Vec::with_capacity(input_dim * output_dim);
    let mut rng_state = 34567;

    for _ in 0..(input_dim * output_dim) {
        // Normal initialization for projection layers
        let weight = normal_random(&mut rng_state, 0.0, 0.02);
        weights.push(weight);
    }

    weights
}

fn generate_embedding_weights(vocab_size: usize, embed_dim: usize) -> Vec<f32> {
    let mut weights = Vec::with_capacity(vocab_size * embed_dim);
    let mut rng_state = 45678;

    for _ in 0..(vocab_size * embed_dim) {
        // Uniform initialization for embeddings
        let weight = -0.1 + 0.2 * lcg_random(&mut rng_state);
        weights.push(weight);
    }

    weights
}

fn generate_output_weights(input_dim: usize, vocab_size: usize) -> Vec<f32> {
    let mut weights = Vec::with_capacity(input_dim * vocab_size);
    let mut rng_state = 56789;

    for _ in 0..(input_dim * vocab_size) {
        // Xavier initialization for output projection
        let limit = (6.0 / (input_dim + vocab_size) as f32).sqrt();
        let weight = -limit + 2.0 * limit * lcg_random(&mut rng_state);
        weights.push(weight);
    }

    weights
}

fn generate_bias_vector(size: usize) -> Vec<f32> {
    let mut bias = Vec::with_capacity(size);
    let mut rng_state = 67890;

    for _ in 0..size {
        // Small random bias
        let b = normal_random(&mut rng_state, 0.0, 0.01);
        bias.push(b);
    }

    bias
}

/// Create quantized weight data
fn create_quantized_weights(
    size: usize,
    qtype: QuantizationType,
    block_size: usize,
) -> QuantizedWeightData {
    let num_blocks = size.div_ceil(block_size);
    let mut rng_state = 78901;

    let quantized_values = match qtype {
        QuantizationType::I2S => {
            (0..size)
                .map(|_| {
                    (lcg_random(&mut rng_state) * 4.0) as u8 // {0, 1, 2, 3} mapped to {-2, -1, 0, 1}
                })
                .collect()
        }
        QuantizationType::TL1 => {
            (0..size).map(|_| (lcg_random(&mut rng_state) * 256.0) as u8).collect()
        }
        QuantizationType::TL2 => {
            (0..size).map(|_| (lcg_random(&mut rng_state) * 256.0) as u8).collect()
        }
        QuantizationType::IQ2S => {
            (0..size).map(|_| (lcg_random(&mut rng_state) * 16.0) as u8).collect()
        }
    };

    let scales = (0..num_blocks).map(|_| 0.01 + lcg_random(&mut rng_state) * 0.1).collect();

    let compression_ratio = match qtype {
        QuantizationType::I2S => 16.0, // 2 bits per weight
        QuantizationType::TL1 => 4.0,  // 8 bits per weight
        QuantizationType::TL2 => 4.0,  // 8 bits per weight
        QuantizationType::IQ2S => 8.0, // 4 bits per weight
    };

    let quantization_params = QuantizationParams {
        method: qtype,
        block_size,
        lookup_table: match qtype {
            QuantizationType::TL1 => {
                Some((0..256).map(|i| -1.0 + 2.0 * i as f32 / 255.0).collect())
            }
            QuantizationType::TL2 => {
                Some((0..4096).map(|i| -1.0 + 2.0 * i as f32 / 4095.0).collect())
            }
            _ => None,
        },
        zero_point: None,
        clamp_range: match qtype {
            QuantizationType::I2S => Some((-2, 1)),
            _ => None,
        },
    };

    QuantizedWeightData {
        quantized_values,
        scales,
        block_size,
        compression_ratio,
        quantization_params,
    }
}

/// Generate model layers for a specific architecture
fn generate_model_layers(
    num_layers: usize,
    hidden_dim: usize,
    qtype: QuantizationType,
) -> Vec<QLinearLayerFixture> {
    let mut layers = Vec::new();

    for _i in 0..num_layers {
        // Query projection
        layers.push(QLinearLayerFixture {
            layer_name: "attention_query",
            input_shape: (1, hidden_dim),
            output_shape: (1, hidden_dim),
            quantization_type: qtype,
            weight_data: generate_attention_weights(hidden_dim, hidden_dim),
            bias_data: Some(generate_bias_vector(hidden_dim)),
            quantized_weights: create_quantized_weights(hidden_dim * hidden_dim, qtype, 64),
            expected_output_range: (-2.0, 2.0),
            layer_type: LayerType::Attention,
            gguf_compatible: true,
        });

        // MLP layers
        let mlp_dim = hidden_dim * 4;
        layers.push(QLinearLayerFixture {
            layer_name: "mlp_up",
            input_shape: (1, hidden_dim),
            output_shape: (1, mlp_dim),
            quantization_type: qtype,
            weight_data: generate_mlp_weights(hidden_dim, mlp_dim),
            bias_data: Some(generate_bias_vector(mlp_dim)),
            quantized_weights: create_quantized_weights(hidden_dim * mlp_dim, qtype, 128),
            expected_output_range: (-3.0, 3.0),
            layer_type: LayerType::Mlp,
            gguf_compatible: true,
        });
    }

    layers
}

/// Create mock layer for comparison
fn create_mock_layer(input_dim: usize, output_dim: usize, uses_mock: bool) -> MockLinearLayer {
    MockLinearLayer {
        input_dim,
        output_dim,
        weights: generate_attention_weights(input_dim, output_dim),
        bias: Some(generate_bias_vector(output_dim)),
        uses_mock_computation: uses_mock,
        mock_fingerprint: MockFingerprint {
            deterministic_pattern: uses_mock,
            zero_variance_regions: if uses_mock { vec![(0, 10), (50, 60)] } else { vec![] },
            suspicious_correlations: if uses_mock { vec![1.0, 0.0, 1.0] } else { vec![] },
            computation_shortcuts: if uses_mock {
                vec!["zero_multiply", "identity_transform"]
            } else {
                vec![]
            },
        },
    }
}

/// Generate test inputs for validation
fn generate_test_inputs(batch_size: usize, input_dim: usize) -> Vec<Vec<f32>> {
    let mut inputs = Vec::new();
    let mut rng_state = 89012;

    for _ in 0..batch_size {
        let input = (0..input_dim).map(|_| normal_random(&mut rng_state, 0.0, 1.0)).collect();
        inputs.push(input);
    }

    inputs
}

/// Generate expected outputs for validation
fn generate_expected_outputs(batch_size: usize, output_dim: usize) -> Vec<Vec<f32>> {
    let mut outputs = Vec::new();
    let mut rng_state = 90123;

    for _ in 0..batch_size {
        let output = (0..output_dim).map(|_| normal_random(&mut rng_state, 0.0, 2.0)).collect();
        outputs.push(output);
    }

    outputs
}

/// Helper functions
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

/// Validate QLinear fixture integrity
pub fn validate_qlinear_fixture(fixture: &QLinearLayerFixture) -> Result<(), String> {
    let expected_weight_count = fixture.input_shape.0
        * fixture.input_shape.1
        * fixture.output_shape.0
        * fixture.output_shape.1;

    if fixture.weight_data.len() != expected_weight_count {
        return Err(format!(
            "Weight count mismatch: expected {}, got {}",
            expected_weight_count,
            fixture.weight_data.len()
        ));
    }

    if let Some(ref bias) = fixture.bias_data {
        let expected_bias_count = fixture.output_shape.0 * fixture.output_shape.1;
        if bias.len() != expected_bias_count {
            return Err(format!(
                "Bias count mismatch: expected {}, got {}",
                expected_bias_count,
                bias.len()
            ));
        }
    }

    if fixture.quantized_weights.quantized_values.len() != fixture.weight_data.len() {
        return Err("Quantized values count must match weight count".to_string());
    }

    Ok(())
}

/// Get fixture by layer name
pub fn get_qlinear_fixture_by_name(name: &str) -> Option<QLinearLayerFixture> {
    load_qlinear_layer_fixtures().into_iter().find(|f| f.layer_name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qlinear_fixtures_validation() {
        let fixtures = load_qlinear_layer_fixtures();
        for fixture in fixtures {
            validate_qlinear_fixture(&fixture).expect("QLinear fixture should be valid");
        }
    }

    #[test]
    fn test_gguf_model_fixtures() {
        let models = load_gguf_model_fixtures();
        assert!(!models.is_empty(), "Should have GGUF model fixtures");

        for model in models {
            assert!(model.tensor_count > 0, "Model should have tensors");
            assert!(model.vocab_size > 0, "Model should have vocabulary");
            assert!(!model.layers.is_empty(), "Model should have layers");
        }
    }

    #[test]
    fn test_layer_replacement_scenarios() {
        let scenarios = load_layer_replacement_scenarios();
        assert!(!scenarios.is_empty(), "Should have replacement scenarios");

        for scenario in scenarios {
            assert_eq!(
                scenario.test_inputs.len(),
                scenario.expected_outputs.len(),
                "Input and output count should match"
            );
        }
    }

    #[test]
    fn test_fallback_detection_fixtures() {
        let fallback_data = load_fallback_detection_fixtures();
        assert!(!fallback_data.is_empty(), "Should have fallback detection data");

        for data in fallback_data {
            assert!(!data.trigger_conditions.is_empty(), "Should have trigger conditions");
        }
    }
}
