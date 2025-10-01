# [Testing] Enhance I2S quantization stability test with comprehensive input distribution coverage

## Problem Description

The `i2s_quantization_stability_test` in `crates/bitnet-quantization/src/property_tests.rs` uses simplified test data generation with only sinusoidal patterns. This limited input distribution may not adequately test quantization stability across the range of real neural network weight and activation patterns.

## Environment

- **File**: `crates/bitnet-quantization/src/property_tests.rs`
- **Function**: `i2s_quantization_stability_test`
- **Component**: Property-based testing for I2S quantization
- **Type**: Test enhancement and simulation improvement
- **MSRV**: Rust 1.90.0

## Current Implementation Analysis

```rust
#[test]
fn i2s_quantization_stability_test(
    block_count in 1usize..100,
    scale_factor in 0.1f32..10.0f32
) {
    // Limited test data generation
    let mut test_data = vec![0.0f32; total_elems];
    for (i, elem) in test_data.iter_mut().enumerate().take(total_elems) {
        *elem = ((i as f32).sin() * scale_factor).clamp(-2.0, 2.0);
    }
    // ... rest of test
}
```

**Limitations:**
- Only sinusoidal distribution patterns
- No coverage of neural network weight distributions
- Missing adversarial cases (sparse, bimodal, heavy-tailed)
- No testing of edge cases (zeros, infinities, denormals)

## Proposed Solution

Implement comprehensive distribution testing with realistic neural network patterns:

```rust
use proptest::prelude::*;
use rand::distributions::{Distribution, Normal, Uniform};

#[derive(Debug, Clone)]
pub enum TestDistribution {
    Uniform { min: f32, max: f32 },
    Normal { mean: f32, std: f32 },
    Exponential { lambda: f32 },
    Bimodal { peak1: f32, peak2: f32, mixing: f32 },
    Sparse { sparsity: f32, magnitude: f32 },
    NeuralWeights { layer_type: LayerType },
    Adversarial { pattern: AdversarialPattern },
}

#[derive(Debug, Clone)]
pub enum LayerType {
    Embedding,
    Attention,
    FeedForward,
    LayerNorm,
}

#[derive(Debug, Clone)]
pub enum AdversarialPattern {
    Checkerboard,
    HighFrequency,
    StepFunction,
    OutlierSpikes,
}

proptest! {
    #[test]
    fn comprehensive_i2s_quantization_stability_test(
        block_count in 1usize..100,
        scale_factor in 0.1f32..10.0f32,
        distribution in distribution_strategy(),
    ) {
        let layout = I2SLayout::default();
        let total_elems = block_count * layout.block_size;

        // Generate test data based on distribution type
        let test_data = generate_test_data(&distribution, total_elems, scale_factor);

        // Perform quantization stability testing
        test_quantization_stability(&test_data, &layout)?;
    }
}

fn distribution_strategy() -> impl Strategy<Value = TestDistribution> {
    prop_oneof![
        // Standard distributions
        uniform_strategy(),
        normal_strategy(),
        exponential_strategy(),
        bimodal_strategy(),
        sparse_strategy(),

        // Neural network specific distributions
        neural_weights_strategy(),
        adversarial_strategy(),
    ]
}

fn generate_test_data(distribution: &TestDistribution, size: usize, scale: f32) -> Vec<f32> {
    match distribution {
        TestDistribution::Uniform { min, max } => {
            generate_uniform_data(size, *min * scale, *max * scale)
        }
        TestDistribution::Normal { mean, std } => {
            generate_normal_data(size, *mean * scale, *std * scale)
        }
        TestDistribution::NeuralWeights { layer_type } => {
            generate_neural_weight_distribution(size, layer_type, scale)
        }
        TestDistribution::Adversarial { pattern } => {
            generate_adversarial_pattern(size, pattern, scale)
        }
        // ... other cases
    }
}

fn generate_neural_weight_distribution(size: usize, layer_type: &LayerType, scale: f32) -> Vec<f32> {
    match layer_type {
        LayerType::Embedding => {
            // Embedding weights: typically normal with small variance
            let normal = Normal::new(0.0, 0.1 * scale as f64).unwrap();
            generate_from_distribution(size, normal)
        }
        LayerType::Attention => {
            // Attention weights: Xavier/Glorot initialization pattern
            let fan_in = (size as f32).sqrt();
            let std = (2.0 / fan_in).sqrt() * scale;
            let normal = Normal::new(0.0, std as f64).unwrap();
            generate_from_distribution(size, normal)
        }
        LayerType::FeedForward => {
            // Feed-forward weights: He initialization
            let std = (2.0 / size as f32).sqrt() * scale;
            let normal = Normal::new(0.0, std as f64).unwrap();
            generate_from_distribution(size, normal)
        }
        LayerType::LayerNorm => {
            // Layer norm: weights close to 1, biases close to 0
            let mut data = vec![1.0; size];
            let noise = Normal::new(0.0, 0.01 * scale as f64).unwrap();
            for elem in &mut data {
                *elem += noise.sample(&mut rand::thread_rng()) as f32;
            }
            data
        }
    }
}

fn generate_adversarial_pattern(size: usize, pattern: &AdversarialPattern, scale: f32) -> Vec<f32> {
    match pattern {
        AdversarialPattern::Checkerboard => {
            (0..size).map(|i| if i % 2 == 0 { scale } else { -scale }).collect()
        }
        AdversarialPattern::HighFrequency => {
            (0..size).map(|i| (i as f32 * 10.0).sin() * scale).collect()
        }
        AdversarialPattern::StepFunction => {
            (0..size).map(|i| if i < size / 2 { -scale } else { scale }).collect()
        }
        AdversarialPattern::OutlierSpikes => {
            let mut data = vec![0.0; size];
            let spike_indices = [size / 4, size / 2, 3 * size / 4];
            for &idx in &spike_indices {
                if idx < size {
                    data[idx] = scale * 10.0; // Large outliers
                }
            }
            data
        }
    }
}
```

## Enhanced Validation

```rust
fn test_quantization_stability(data: &[f32], layout: &I2SLayout) -> Result<()> {
    // 1. Basic quantization roundtrip
    let quantized = quantize_i2s(data, layout)?;
    let dequantized = dequantize_i2s(&quantized, layout)?;

    // 2. Stability metrics
    let stability_metrics = calculate_stability_metrics(data, &dequantized);

    // 3. Distribution-specific validations
    validate_quantization_quality(data, &dequantized, &stability_metrics)?;

    // 4. Edge case handling
    validate_edge_case_handling(data, &quantized)?;

    Ok(())
}

#[derive(Debug)]
struct StabilityMetrics {
    snr_db: f32,
    max_error: f32,
    mean_squared_error: f32,
    quantile_errors: [f32; 5], // 5%, 25%, 50%, 75%, 95%
    zero_preservation: f32,
    outlier_handling: f32,
}

fn validate_quantization_quality(
    original: &[f32],
    quantized: &[f32],
    metrics: &StabilityMetrics,
) -> Result<()> {
    // SNR should be reasonable for the quantization method
    prop_assert!(metrics.snr_db > 20.0, "SNR too low: {:.2} dB", metrics.snr_db);

    // Maximum error should be bounded
    prop_assert!(metrics.max_error < 2.0, "Maximum error too large: {:.4}", metrics.max_error);

    // Check distribution preservation
    let original_stats = calculate_distribution_stats(original);
    let quantized_stats = calculate_distribution_stats(quantized);

    let mean_diff = (original_stats.mean - quantized_stats.mean).abs();
    prop_assert!(mean_diff < 0.1, "Mean preservation failed: diff = {:.4}", mean_diff);

    // Validate quantile preservation
    for (i, &error) in metrics.quantile_errors.iter().enumerate() {
        let percentile = [5, 25, 50, 75, 95][i];
        prop_assert!(error < 0.5, "{}th percentile error too large: {:.4}", percentile, error);
    }

    Ok(())
}
```

## Implementation Plan

### Phase 1: Distribution Generators (1 day)
- [ ] Implement comprehensive test distribution generators
- [ ] Add neural network weight pattern simulation
- [ ] Create adversarial pattern generators
- [ ] Add distribution parameter strategies

### Phase 2: Enhanced Validation (1 day)
- [ ] Implement stability metrics calculation
- [ ] Add distribution preservation validation
- [ ] Create edge case testing
- [ ] Add quantization quality assessment

### Phase 3: Integration and Testing (0.5 days)
- [ ] Integrate with existing property tests
- [ ] Add performance benchmarks for different distributions
- [ ] Validate test coverage improvements
- [ ] Document test patterns and expected behaviors

## Acceptance Criteria

### Functional Requirements
- [ ] Comprehensive input distribution coverage
- [ ] Neural network weight pattern testing
- [ ] Adversarial case handling validation
- [ ] Edge case stability verification

### Quality Requirements
- [ ] Property tests pass with enhanced distributions
- [ ] Test execution time remains reasonable
- [ ] Clear failure diagnostics for different patterns
- [ ] Reproducible test results with seeds

## Labels

`testing`, `quantization`, `property-based-testing`, `i2s`, `simulation`, `medium-priority`