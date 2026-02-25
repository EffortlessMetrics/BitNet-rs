# Quantization-Aware Testing Strategy for Real BitNet Model Integration

## Overview

This document defines the comprehensive testing strategy for real BitNet model integration with focus on quantization accuracy validation, device-aware execution testing, and TDD compliance across all quantization formats (I2S, TL1, TL2).

## Testing Hierarchy and Strategy

### 1. Unit Testing Layer: Quantization Primitives

**Objective**: Validate individual quantization operations with synthetic and real tensor data.

```rust
// crates/bitnet-quantization/tests/unit_tests.rs
#[cfg(test)]
mod quantization_unit_tests {
    use super::*;
    use proptest::prelude::*;

    // Property-based testing for quantization invariants
    proptest! {
        #[test]
        fn test_i2s_quantization_round_trip(
            values in prop::collection::vec(-2.0f32..2.0f32, 1..1000)
        ) {
            let tensor = Tensor::from_vec(values.clone(), (values.len(),), &Device::Cpu)?;
            let quantizer = I2SQuantizer::new();

            let quantized = quantizer.quantize(&tensor)?;
            let dequantized = quantizer.dequantize(&quantized)?;

            // Check that dequantization stays within expected bounds
            let original_data: Vec<f32> = tensor.to_vec1()?;
            let recovered_data: Vec<f32> = dequantized.to_vec1()?;

            for (orig, recovered) in original_data.iter().zip(recovered_data.iter()) {
                let error = (orig - recovered).abs();
                prop_assert!(error < 0.1, "Quantization error too large: {} vs {}", orig, recovered);
            }
        }
    }

    #[test]
    fn test_real_tensor_quantization_accuracy() {
        let real_tensor = load_real_model_tensor_sample("attention.weight.0").unwrap();

        for format in [QuantizationFormat::I2S, QuantizationFormat::TL1, QuantizationFormat::TL2] {
            let quantizer = create_quantizer(format);

            // Quantize real tensor
            let quantized = quantizer.quantize(&real_tensor).unwrap();

            // Validate quantization properties
            validate_quantization_format(&quantized, format);

            // Check dequantization accuracy
            let dequantized = quantizer.dequantize(&quantized).unwrap();
            let mse = calculate_mse(&real_tensor, &dequantized);

            // Format-specific accuracy thresholds
            let max_mse = match format {
                QuantizationFormat::I2S => 1e-3,
                QuantizationFormat::TL1 => 5e-4,
                QuantizationFormat::TL2 => 1e-4,
            };

            assert!(mse < max_mse, "MSE too high for {:?}: {} > {}", format, mse, max_mse);
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_cpu_quantization_parity() {
        let test_tensor = create_test_tensor(1024, 1024);

        for format in [QuantizationFormat::I2S, QuantizationFormat::TL1, QuantizationFormat::TL2] {
            let cpu_quantizer = CPUQuantizer::new(format);
            let gpu_quantizer = GPUQuantizer::new(format).unwrap();

            let cpu_result = cpu_quantizer.quantize(&test_tensor).unwrap();
            let gpu_result = gpu_quantizer.quantize(&test_tensor).unwrap();

            // GPU and CPU results should be numerically equivalent
            let cpu_dequant = cpu_quantizer.dequantize(&cpu_result).unwrap();
            let gpu_dequant = gpu_quantizer.dequantize(&gpu_result).unwrap();

            let parity_error = calculate_mse(&cpu_dequant, &gpu_dequant);
            assert!(parity_error < 1e-6, "GPU/CPU parity error too high: {}", parity_error);
        }
    }
}
```

### 2. Integration Testing Layer: Real Model Quantization

**Objective**: Test complete quantization pipeline with real BitNet model weights.

```rust
// crates/bitnet-quantization/tests/integration_tests.rs
#[cfg(feature = "integration-tests")]
mod integration_tests {
    use super::*;

    struct RealModelQuantizationTest {
        model_path: PathBuf,
        expected_quantization: QuantizationFormat,
        tolerance_config: ToleranceConfig,
    }

    impl RealModelQuantizationTest {
        fn new(model_path: PathBuf) -> Result<Self, TestError> {
            // Detect quantization format from GGUF metadata
            let model = GGUFModel::load(&model_path)?;
            let metadata = model.get_metadata();

            let expected_quantization = match metadata.get("general.quantization_version") {
                Some("i2_s") => QuantizationFormat::I2S,
                Some("tl1") => QuantizationFormat::TL1,
                Some("tl2") => QuantizationFormat::TL2,
                _ => return Err(TestError::UnsupportedQuantization),
            };

            Ok(Self {
                model_path,
                expected_quantization,
                tolerance_config: ToleranceConfig::for_quantization(expected_quantization),
            })
        }

        async fn run_quantization_validation(&self) -> Result<ValidationReport, TestError> {
            let mut report = ValidationReport::new();

            // Load real model
            let model = BitNetModel::load(&self.model_path).await?;

            // Test each layer quantization
            for (layer_name, layer_weights) in model.get_quantized_weights() {
                let validation_result = self.validate_layer_quantization(layer_name, &layer_weights).await?;
                report.layer_results.push(validation_result);
            }

            // Test end-to-end quantization accuracy
            let e2e_result = self.validate_end_to_end_accuracy(&model).await?;
            report.end_to_end_result = Some(e2e_result);

            Ok(report)
        }

        async fn validate_layer_quantization(&self, layer_name: &str, weights: &QuantizedTensor) -> Result<LayerValidationResult, TestError> {
            // 1. Format validation
            assert_eq!(weights.format(), self.expected_quantization);

            // 2. Dequantization accuracy
            let dequantized = weights.dequantize()?;
            let reference_weights = self.load_reference_weights(layer_name).await?;

            let mse = calculate_mse(&reference_weights, &dequantized);
            let max_error = calculate_max_error(&reference_weights, &dequantized);

            // 3. Device-aware validation
            let device_results = self.validate_device_quantization(weights).await?;

            Ok(LayerValidationResult {
                layer_name: layer_name.to_string(),
                format: weights.format(),
                mse,
                max_error,
                passes_tolerance: mse < self.tolerance_config.mse_threshold,
                device_results,
            })
        }

        async fn validate_device_quantization(&self, weights: &QuantizedTensor) -> Result<DeviceValidationResults, TestError> {
            let mut results = DeviceValidationResults::new();

            // CPU validation (always available)
            let cpu_result = self.validate_cpu_quantization(weights).await?;
            results.cpu = Some(cpu_result);

            // GPU validation (if available)
            #[cfg(feature = "gpu")]
            if let Ok(gpu_device) = GPUDevice::new() {
                let gpu_result = self.validate_gpu_quantization(weights, &gpu_device).await?;
                results.gpu = Some(gpu_result);

                // Cross-device parity check
                if let Some(ref cpu_result) = results.cpu {
                    let parity_error = calculate_mse(&cpu_result.dequantized, &gpu_result.dequantized);
                    results.parity_error = Some(parity_error);
                    results.parity_passes = parity_error < 1e-5;
                }
            }

            Ok(results)
        }
    }

    #[tokio::test]
    async fn test_real_bitnet_2b_quantization() {
        let model_path = get_test_model_path("bitnet-2b");
        let test = RealModelQuantizationTest::new(model_path).unwrap();

        let report = test.run_quantization_validation().await.unwrap();

        // All layers should pass quantization validation
        for layer_result in &report.layer_results {
            assert!(layer_result.passes_tolerance,
                   "Layer {} failed quantization tolerance: MSE = {}",
                   layer_result.layer_name, layer_result.mse);
        }

        // End-to-end accuracy should be preserved
        if let Some(e2e_result) = &report.end_to_end_result {
            assert!(e2e_result.perplexity_preserved, "Perplexity not preserved after quantization");
            assert!(e2e_result.inference_accuracy > 0.95, "Inference accuracy too low: {}", e2e_result.inference_accuracy);
        }
    }

    #[tokio::test]
    async fn test_quantization_cross_validation() {
        let rust_model = load_rust_bitnet_model().await.unwrap();
        let cpp_reference = load_cpp_reference_model().await.unwrap();

        let test_tensors = extract_test_tensors(&rust_model);

        for (tensor_name, rust_tensor) in test_tensors {
            let cpp_tensor = cpp_reference.get_tensor(&tensor_name).unwrap();

            // Compare quantized representations
            let quantization_diff = compare_quantized_tensors(&rust_tensor, &cpp_tensor);
            assert!(quantization_diff < 1e-5, "Quantization differs from C++ reference: {}", quantization_diff);

            // Compare dequantized outputs
            let rust_dequant = rust_tensor.dequantize().unwrap();
            let cpp_dequant = cpp_tensor.dequantize().unwrap();

            let dequant_diff = calculate_mse(&rust_dequant, &cpp_dequant);
            assert!(dequant_diff < 1e-4, "Dequantization differs from C++ reference: {}", dequant_diff);
        }
    }
}
```

### 3. Performance Testing Layer: Device-Aware Quantization

**Objective**: Validate quantization performance across CPU/GPU backends with real models.

```rust
// crates/bitnet-quantization/benches/real_model_quantization_benchmarks.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_real_model_quantization(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Load real model weights
    let model_path = std::env::var("BITNET_MODEL_PATH").unwrap_or_else(|_| {
        eprintln!("⚠️  BITNET_MODEL_PATH not set, skipping real model benchmarks");
        return;
    });

    let model = rt.block_on(async {
        BitNetModel::load(&model_path).await.expect("Failed to load model")
    });

    let mut group = c.benchmark_group("real_model_quantization");

    // Test different layer sizes from real model
    let test_layers = extract_representative_layers(&model);

    for (layer_name, layer_weights) in test_layers {
        let tensor_size = layer_weights.numel();

        // CPU quantization benchmarks
        for format in [QuantizationFormat::I2S, QuantizationFormat::TL1, QuantizationFormat::TL2] {
            group.bench_with_input(
                BenchmarkId::new("cpu_quantize", format!("{}_{:?}_{}", layer_name, format, tensor_size)),
                &(&layer_weights, format),
                |b, (weights, format)| {
                    let quantizer = CPUQuantizer::new(*format);
                    b.iter(|| quantizer.quantize(weights).unwrap());
                },
            );

            group.bench_with_input(
                BenchmarkId::new("cpu_dequantize", format!("{}_{:?}_{}", layer_name, format, tensor_size)),
                &(&layer_weights, format),
                |b, (weights, format)| {
                    let quantizer = CPUQuantizer::new(*format);
                    let quantized = quantizer.quantize(weights).unwrap();
                    b.iter(|| quantizer.dequantize(&quantized).unwrap());
                },
            );
        }

        // GPU quantization benchmarks (if available)
        #[cfg(feature = "gpu")]
        {
            if let Ok(gpu_device) = GPUDevice::new() {
                for format in [QuantizationFormat::I2S, QuantizationFormat::TL1, QuantizationFormat::TL2] {
                    group.bench_with_input(
                        BenchmarkId::new("gpu_quantize", format!("{}_{:?}_{}", layer_name, format, tensor_size)),
                        &(&layer_weights, format),
                        |b, (weights, format)| {
                            let quantizer = GPUQuantizer::new(*format).unwrap();
                            b.to_async(&rt).iter(|| async {
                                quantizer.quantize(weights).await.unwrap()
                            });
                        },
                    );
                }
            }
        }
    }

    group.finish();
}

fn benchmark_quantization_accuracy_vs_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_vs_performance");

    // Test accuracy-performance trade-offs
    let test_cases = vec![
        ("fast", ToleranceConfig::fast()),
        ("balanced", ToleranceConfig::balanced()),
        ("accurate", ToleranceConfig::accurate()),
    ];

    for (profile_name, tolerance_config) in test_cases {
        let real_weights = load_real_model_sample_weights();

        group.bench_with_input(
            BenchmarkId::new("quantization_profile", profile_name),
            &(&real_weights, &tolerance_config),
            |b, (weights, config)| {
                b.iter(|| {
                    let quantizer = AdaptiveQuantizer::new(config.clone());
                    let quantized = quantizer.quantize(weights).unwrap();
                    let dequantized = quantizer.dequantize(&quantized).unwrap();

                    // Measure accuracy as part of benchmark
                    let mse = calculate_mse(weights, &dequantized);
                    criterion::black_box(mse);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_real_model_quantization, benchmark_quantization_accuracy_vs_performance);
criterion_main!(benches);
```

### 4. End-to-End Testing: Complete Pipeline Validation

**Objective**: Validate quantization accuracy preservation through complete inference pipeline.

```rust
// crates/bitnet-inference/tests/e2e_quantization_tests.rs
#[cfg(feature = "integration-tests")]
mod e2e_tests {
    use super::*;

    #[tokio::test]
    async fn test_e2e_quantization_accuracy_preservation() {
        let model_path = get_test_model_path();

        // Load reference (unquantized) model for comparison
        let reference_model = load_reference_model(&model_path).await.unwrap();

        // Load quantized BitNet model
        let quantized_model = BitNetModel::load(&model_path).await.unwrap();

        let test_prompts = [
            "The capital of France is",
            "Explain quantum computing",
            "Write a simple function to",
        ];

        for prompt in test_prompts {
            // Reference inference (if available)
            let reference_result = reference_model.infer(prompt).await.unwrap();

            // Quantized inference
            let quantized_result = quantized_model.infer(prompt).await.unwrap();

            // Compare results
            let token_similarity = calculate_token_similarity(
                &reference_result.tokens,
                &quantized_result.tokens
            );

            assert!(token_similarity > 0.90,
                   "Token similarity too low for '{}': {}", prompt, token_similarity);

            // Compare logits (if available)
            if let (Some(ref_logits), Some(quant_logits)) = (&reference_result.logits, &quantized_result.logits) {
                let logit_correlation = calculate_pearson_correlation(ref_logits, quant_logits);
                assert!(logit_correlation > 0.95,
                       "Logit correlation too low for '{}': {}", prompt, logit_correlation);
            }
        }
    }

    #[tokio::test]
    async fn test_perplexity_preservation_across_quantization() {
        let model_path = get_test_model_path();
        let corpus_path = get_test_corpus_path();

        let model = BitNetModel::load(&model_path).await.unwrap();
        let perplexity_calculator = PerplexityCalculator::new(model);

        // Calculate perplexity with different quantization settings
        let results = vec![
            ("original", QuantizationConfig::none()),
            ("i2s", QuantizationConfig::i2s()),
            ("tl1", QuantizationConfig::tl1()),
            ("tl2", QuantizationConfig::tl2()),
        ];

        let mut perplexity_results = Vec::new();

        for (config_name, quant_config) in results {
            let ppl_result = perplexity_calculator
                .calculate_with_quantization(&corpus_path, quant_config)
                .await
                .unwrap();

            perplexity_results.push((config_name, ppl_result.perplexity));
        }

        // Reference perplexity (original/unquantized)
        let reference_ppl = perplexity_results[0].1;

        // Validate quantized perplexities are within tolerance
        for (config_name, ppl) in &perplexity_results[1..] {
            let relative_diff = (ppl - reference_ppl).abs() / reference_ppl;
            assert!(relative_diff < 0.02,
                   "Perplexity degradation too high for {}: {:.1}%",
                   config_name, relative_diff * 100.0);
        }
    }

    #[tokio::test]
    async fn test_device_quantization_consistency() {
        let model_path = get_test_model_path();
        let test_prompt = "Device consistency test prompt";

        // CPU inference
        let cpu_engine = InferenceEngine::new_cpu(&model_path).await.unwrap();
        let cpu_result = cpu_engine.infer(test_prompt).await.unwrap();

        // GPU inference (if available)
        #[cfg(feature = "gpu")]
        if let Ok(gpu_engine) = InferenceEngine::new_gpu(&model_path).await {
            let gpu_result = gpu_engine.infer(test_prompt).await.unwrap();

            // Results should be identical for deterministic inference
            assert_eq!(cpu_result.tokens, gpu_result.tokens,
                      "CPU/GPU quantization results differ");

            // Performance should be significantly better on GPU
            let speedup = cpu_result.inference_time.as_secs_f64() / gpu_result.inference_time.as_secs_f64();
            assert!(speedup > 2.0, "GPU speedup insufficient: {:.1}x", speedup);
        }
    }
}
```

### 5. TDD Test-First Development Strategy

**Implementation Process**:

1. **Red Phase**: Write failing tests for quantization functionality
2. **Green Phase**: Implement minimal code to pass tests
3. **Refactor Phase**: Optimize quantization algorithms while maintaining test coverage

**Example TDD Cycle for I2S Quantization**:

```rust
// Step 1: Red - Write failing test
#[test]
fn test_i2s_quantization_basic_functionality() {
    let input = Tensor::from_vec(vec![1.5, -0.8, 2.1, -1.2], (4,), &Device::Cpu).unwrap();
    let quantizer = I2SQuantizer::new();

    let quantized = quantizer.quantize(&input).unwrap();
    let dequantized = quantizer.dequantize(&quantized).unwrap();

    // This will fail initially
    assert!(validate_quantization_accuracy(&input, &dequantized, 0.1));
}

// Step 2: Green - Implement minimal I2S quantizer
impl I2SQuantizer {
    pub fn quantize(&self, input: &Tensor) -> Result<QuantizedTensor, QuantizationError> {
        // Minimal implementation that passes the test
        // (Will be refined in refactor phase)
        todo!("Implement I2S quantization")
    }
}

// Step 3: Refactor - Add real model validation
#[test]
fn test_i2s_quantization_with_real_weights() {
    let real_weights = load_real_bitnet_layer_weights("attention.0").unwrap();
    let quantizer = I2SQuantizer::new();

    let quantized = quantizer.quantize(&real_weights).unwrap();

    // Ensure real weights maintain accuracy
    validate_real_weight_quantization(&real_weights, &quantized);
}
```

## Test Data Management

### Real Model Test Fixtures

```rust
// tests/common/fixtures.rs
pub struct RealModelFixtures {
    models_dir: PathBuf,
    cache: ModelCache,
}

impl RealModelFixtures {
    pub fn new() -> Result<Self, FixtureError> {
        let models_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("bitnet-test-models");

        std::fs::create_dir_all(&models_dir)?;

        Ok(Self {
            models_dir,
            cache: ModelCache::new(),
        })
    }

    pub async fn get_bitnet_2b_model(&mut self) -> Result<PathBuf, FixtureError> {
        self.cache.get_or_download(
            "microsoft/bitnet-b1.58-2B-4T-gguf",
            "ggml-model-i2_s.gguf"
        ).await
    }

    pub async fn get_test_corpus(&self) -> Result<PathBuf, FixtureError> {
        // Download or use cached test corpus for perplexity testing
        let corpus_path = self.models_dir.join("test_corpus.txt");

        if !corpus_path.exists() {
            self.download_test_corpus(&corpus_path).await?;
        }

        Ok(corpus_path)
    }

    pub fn cleanup_old_fixtures(&self) -> Result<(), FixtureError> {
        // Clean up fixtures older than 7 days
        self.cache.cleanup_old_models(7)
    }
}
```

### Test Configuration Management

```rust
// tests/common/config.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationTestConfig {
    pub tolerance: ToleranceConfig,
    pub test_modes: Vec<TestMode>,
    pub device_config: DeviceTestConfig,
    pub model_requirements: ModelRequirements,
}

impl QuantizationTestConfig {
    pub fn for_ci() -> Self {
        Self {
            tolerance: ToleranceConfig::strict(),
            test_modes: vec![TestMode::Fast, TestMode::Integration],
            device_config: DeviceTestConfig::cpu_only(),
            model_requirements: ModelRequirements::essential_only(),
        }
    }

    pub fn for_local_development() -> Self {
        Self {
            tolerance: ToleranceConfig::balanced(),
            test_modes: vec![TestMode::Fast, TestMode::Integration, TestMode::Performance],
            device_config: DeviceTestConfig::auto_detect(),
            model_requirements: ModelRequirements::full_suite(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TestMode {
    Fast,           // Mock models, CPU only
    Integration,    // Real models, basic validation
    Performance,    // Benchmarking with real models
    CrossVal,       // C++ cross-validation
}
```

## Continuous Integration Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/quantization-testing.yml
name: Quantization Testing

on:
  pull_request:
    paths:
      - 'crates/bitnet-quantization/**'
      - 'crates/bitnet-kernels/**'
  push:
    branches: [main]

jobs:
  fast-quantization-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: 1.90.0
      - name: Run fast quantization tests
        run: |
          cargo test --no-default-features -p bitnet-quantization --no-default-features --features cpu
          cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu

  real-model-quantization-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    needs: fast-quantization-tests
    steps:
      - uses: actions/checkout@v4
      - name: Cache test models
        uses: actions/cache@v4
        with:
          path: ~/.cache/bitnet-test-models
          key: quantization-models-${{ hashFiles('tests/model-requirements.json') }}
      - name: Download test models
        run: |
          cargo run -p xtask -- download-test-models --cache-dir ~/.cache/bitnet-test-models
      - name: Run integration tests
        env:
          BITNET_TEST_MODELS_DIR: ~/.cache/bitnet-test-models
          BITNET_STRICT_TOKENIZERS: 1
        run: |
          cargo test --no-default-features --features "cpu,integration-tests" \
            -p bitnet-quantization \
            -p bitnet-inference \
            -- --test-threads=1

  gpu-quantization-tests:
    runs-on: gpu-runner  # Self-hosted runner with GPU
    timeout-minutes: 20
    needs: fast-quantization-tests
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Run GPU quantization tests
        env:
          BITNET_TEST_MODELS_DIR: /opt/bitnet-models
        run: |
          cargo test --no-default-features --features "gpu,integration-tests" \
            -p bitnet-quantization \
            -p bitnet-kernels \
            -- --test-threads=1
```

## Success Metrics and Validation Criteria

### Quantitative Metrics

1. **Accuracy Preservation**:
   - I2S quantization: MSE < 1e-3 vs reference
   - TL1/TL2 quantization: MSE < 5e-4 vs reference
   - End-to-end perplexity: <2% degradation

2. **Performance Targets**:
   - GPU quantization: >5x speedup vs CPU
   - CPU quantization: >70% throughput of unquantized
   - Memory usage: <50% of unquantized model

3. **Cross-Validation**:
   - Token match rate: >95% vs C++ reference
   - Logit correlation: >0.98 Pearson coefficient
   - Performance parity: Within 20% of C++ implementation

### Qualitative Validation

1. **Test Coverage**: >90% line coverage for quantization code
2. **Documentation**: All quantization functions documented with examples
3. **Error Handling**: Graceful failure modes for all error conditions
4. **Device Compatibility**: Consistent behavior across CPU/GPU backends

This comprehensive testing strategy ensures real BitNet model integration maintains quantization accuracy while enabling device-aware optimization and TDD compliance throughout the development process.
