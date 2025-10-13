# Cross-Validation Specification with C++ Reference Implementation

**Component**: Comprehensive validation against C++ reference implementation (llama.cpp)
**Location**: `crossval/` crate and integrated validation throughout BitNet.rs
**Dependencies**: C++ reference bindings, statistical analysis, deterministic testing

## Overview

This specification defines a comprehensive cross-validation framework for BitNet.rs neural network inference, ensuring >99.9% accuracy correlation with C++ reference implementations (primarily llama.cpp). The framework validates quantization accuracy, numerical stability, generation consistency, and performance parity across CPU and GPU implementations while providing detailed diagnostic information for any deviations.

## Cross-Validation Architecture

### Integration with Reference Implementation

```rust
// Core cross-validation framework
pub struct CrossValidationFramework {
    // C++ reference integration
    cpp_reference: CppReferenceEngine,
    rust_implementation: BitNetInferenceEngine,

    // Validation configuration
    validation_config: CrossValidationConfig,
    tolerance_config: ToleranceConfig,

    // Statistical analysis
    statistical_analyzer: StatisticalAnalyzer,
    deviation_tracker: DeviationTracker,

    // Reporting
    report_generator: ValidationReportGenerator,
}

impl CrossValidationFramework {
    /// Create framework with C++ reference integration
    pub fn new(
        model_path: &str,
        tokenizer_path: &str,
        validation_config: CrossValidationConfig
    ) -> Result<Self> {
        // Initialize C++ reference (llama.cpp integration)
        let cpp_reference = CppReferenceEngine::load(model_path, tokenizer_path)?;

        // Initialize Rust implementation
        let rust_implementation = BitNetInferenceEngine::load(
            model_path,
            tokenizer_path,
            validation_config.quantization_type,
            validation_config.device
        )?;

        // Configure deterministic execution for both implementations
        cpp_reference.set_deterministic(true, validation_config.seed)?;
        rust_implementation.set_deterministic(true, validation_config.seed)?;

        Ok(Self {
            cpp_reference,
            rust_implementation,
            validation_config,
            tolerance_config: ToleranceConfig::default(),
            statistical_analyzer: StatisticalAnalyzer::new(),
            deviation_tracker: DeviationTracker::new(),
            report_generator: ValidationReportGenerator::new(),
        })
    }

    /// Run comprehensive cross-validation suite
    pub fn run_cross_validation(&mut self) -> Result<CrossValidationReport> {
        let mut report = CrossValidationReport::new();

        // Phase 1: Forward pass validation
        let forward_validation = self.validate_forward_passes()?;
        report.add_phase_result("forward_pass", forward_validation);

        // Phase 2: Generation consistency validation
        let generation_validation = self.validate_generation_consistency()?;
        report.add_phase_result("generation", generation_validation);

        // Phase 3: Quantization accuracy validation
        let quantization_validation = self.validate_quantization_accuracy()?;
        report.add_phase_result("quantization", quantization_validation);

        // Phase 4: Performance parity validation
        let performance_validation = self.validate_performance_parity()?;
        report.add_phase_result("performance", performance_validation);

        // Phase 5: Numerical stability validation
        let stability_validation = self.validate_numerical_stability()?;
        report.add_phase_result("stability", stability_validation);

        // Generate final assessment
        report.overall_assessment = self.generate_overall_assessment(&report)?;

        Ok(report)
    }
}
```

### Validation Configuration

```rust
/// Comprehensive cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    // Model configuration
    pub model_path: String,
    pub tokenizer_path: String,
    pub quantization_type: QuantizationType,
    pub device: Device,

    // Deterministic execution
    pub seed: u64,
    pub enable_deterministic_mode: bool,

    // Validation scope
    pub validate_forward_pass: bool,
    pub validate_generation: bool,
    pub validate_quantization: bool,
    pub validate_performance: bool,
    pub validate_stability: bool,

    // Test parameters
    pub num_test_cases: usize,
    pub max_sequence_length: usize,
    pub generation_length: usize,
    pub batch_sizes: Vec<usize>,

    // Statistical requirements
    pub min_correlation_threshold: f64,    // 0.999
    pub max_mse_threshold: f64,            // 1e-6
    pub max_deviation_percentage: f64,     // 0.1%
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: String::new(),
            quantization_type: QuantizationType::I2S,
            device: Device::Cpu,

            seed: 42,
            enable_deterministic_mode: true,

            validate_forward_pass: true,
            validate_generation: true,
            validate_quantization: true,
            validate_performance: true,
            validate_stability: true,

            num_test_cases: 100,
            max_sequence_length: 512,
            generation_length: 50,
            batch_sizes: vec![1, 2, 4, 8],

            min_correlation_threshold: 0.999,
            max_mse_threshold: 1e-6,
            max_deviation_percentage: 0.1,
        }
    }
}

/// Tolerance configuration for numerical comparisons
#[derive(Debug, Clone)]
pub struct ToleranceConfig {
    // Absolute tolerances
    pub logits_absolute_tolerance: f64,        // 1e-4
    pub attention_weights_tolerance: f64,      // 1e-5
    pub hidden_states_tolerance: f64,          // 1e-4

    // Relative tolerances
    pub logits_relative_tolerance: f64,        // 1e-3
    pub generation_token_tolerance: f64,       // 0.0 (exact match)

    // Statistical tolerances
    pub correlation_tolerance: f64,            // 0.001 (99.9% minimum)
    pub mse_tolerance: f64,                    // 1e-6
    pub distribution_tolerance: f64,           // 0.01 (KL divergence)

    // Performance tolerances
    pub performance_deviation_tolerance: f64,  // 2.0 (within 2x)
    pub memory_deviation_tolerance: f64,       // 1.5 (within 50%)
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            logits_absolute_tolerance: 1e-4,
            attention_weights_tolerance: 1e-5,
            hidden_states_tolerance: 1e-4,

            logits_relative_tolerance: 1e-3,
            generation_token_tolerance: 0.0,

            correlation_tolerance: 0.001,
            mse_tolerance: 1e-6,
            distribution_tolerance: 0.01,

            performance_deviation_tolerance: 2.0,
            memory_deviation_tolerance: 1.5,
        }
    }
}
```

## Forward Pass Validation

### Tensor-Level Comparison

```rust
impl CrossValidationFramework {
    /// Validate forward pass outputs against C++ reference
    fn validate_forward_passes(&mut self) -> Result<ForwardPassValidationResult> {
        let mut validation_result = ForwardPassValidationResult::new();

        // Generate test cases with varying complexity
        let test_cases = self.generate_forward_pass_test_cases()?;

        for (case_idx, test_case) in test_cases.iter().enumerate() {
            let case_result = self.validate_single_forward_pass(test_case, case_idx)?;
            validation_result.add_case_result(case_result);
        }

        // Statistical analysis of results
        validation_result.statistical_summary = self.analyze_forward_pass_statistics(&validation_result)?;

        Ok(validation_result)
    }

    fn validate_single_forward_pass(&mut self, test_case: &ForwardPassTestCase, case_idx: usize) -> Result<ForwardPassCaseResult> {
        // Execute C++ reference forward pass
        let cpp_start = Instant::now();
        let cpp_result = self.cpp_reference.forward(&test_case.input_tokens)?;
        let cpp_time = cpp_start.elapsed();

        // Execute Rust implementation forward pass
        let rust_start = Instant::now();
        let rust_result = self.rust_implementation.forward(&test_case.input_tokens)?;
        let rust_time = rust_start.elapsed();

        // Compare logits (most critical comparison)
        let logits_comparison = self.compare_logits(&cpp_result.logits, &rust_result.logits)?;

        // Compare hidden states (if available)
        let hidden_states_comparison = if let (Some(cpp_hidden), Some(rust_hidden)) =
            (&cpp_result.hidden_states, &rust_result.hidden_states)
        {
            Some(self.compare_hidden_states(cpp_hidden, rust_hidden)?)
        } else {
            None
        };

        // Compare attention weights (if available)
        let attention_comparison = if let (Some(cpp_attn), Some(rust_attn)) =
            (&cpp_result.attention_weights, &rust_result.attention_weights)
        {
            Some(self.compare_attention_weights(cpp_attn, rust_attn)?)
        } else {
            None
        };

        Ok(ForwardPassCaseResult {
            case_index: case_idx,
            test_case: test_case.clone(),

            logits_comparison,
            hidden_states_comparison,
            attention_comparison,

            performance_comparison: PerformanceComparison {
                cpp_time,
                rust_time,
                speedup_ratio: cpp_time.as_secs_f64() / rust_time.as_secs_f64(),
            },

            passed: logits_comparison.meets_tolerance(&self.tolerance_config),
        })
    }

    /// Compare logits with comprehensive statistical analysis
    fn compare_logits(&self, cpp_logits: &Tensor, rust_logits: &Tensor) -> Result<TensorComparison> {
        // Shape validation
        if cpp_logits.dims() != rust_logits.dims() {
            return Err(ValidationError::ShapeMismatch {
                expected: cpp_logits.dims().to_vec(),
                actual: rust_logits.dims().to_vec(),
            });
        }

        // Convert to vectors for analysis
        let cpp_data: Vec<f32> = cpp_logits.flatten_all()?.to_vec1()?;
        let rust_data: Vec<f32> = rust_logits.flatten_all()?.to_vec1()?;

        // Comprehensive statistical comparison
        let comparison = TensorComparison {
            // Correlation analysis
            pearson_correlation: self.statistical_analyzer.pearson_correlation(&cpp_data, &rust_data)?,
            spearman_correlation: self.statistical_analyzer.spearman_correlation(&cpp_data, &rust_data)?,

            // Error metrics
            mean_squared_error: self.statistical_analyzer.mean_squared_error(&cpp_data, &rust_data)?,
            mean_absolute_error: self.statistical_analyzer.mean_absolute_error(&cpp_data, &rust_data)?,
            max_absolute_error: self.statistical_analyzer.max_absolute_error(&cpp_data, &rust_data)?,

            // Distribution comparison
            kl_divergence: self.statistical_analyzer.kl_divergence(&cpp_data, &rust_data)?,
            js_divergence: self.statistical_analyzer.js_divergence(&cpp_data, &rust_data)?,

            // Element-wise tolerances
            elements_within_absolute_tolerance: self.count_within_absolute_tolerance(
                &cpp_data, &rust_data, self.tolerance_config.logits_absolute_tolerance
            ),
            elements_within_relative_tolerance: self.count_within_relative_tolerance(
                &cpp_data, &rust_data, self.tolerance_config.logits_relative_tolerance
            ),

            // Value distribution analysis
            value_distribution_analysis: self.analyze_value_distributions(&cpp_data, &rust_data)?,
        };

        Ok(comparison)
    }

    fn generate_forward_pass_test_cases(&self) -> Result<Vec<ForwardPassTestCase>> {
        let mut test_cases = Vec::new();

        // Single token cases
        test_cases.extend(self.generate_single_token_cases()?);

        // Short sequence cases
        test_cases.extend(self.generate_short_sequence_cases()?);

        // Medium sequence cases
        test_cases.extend(self.generate_medium_sequence_cases()?);

        // Long sequence cases (if supported)
        if self.validation_config.max_sequence_length > 256 {
            test_cases.extend(self.generate_long_sequence_cases()?);
        }

        // Edge cases
        test_cases.extend(self.generate_edge_cases()?);

        // Batch cases
        test_cases.extend(self.generate_batch_cases()?);

        Ok(test_cases)
    }

    fn generate_single_token_cases(&self) -> Result<Vec<ForwardPassTestCase>> {
        let mut cases = Vec::new();

        // Test common tokens
        let common_tokens = vec![1, 2, 13, 29, 50, 100, 1000, 10000]; // Representative token IDs

        for token in common_tokens {
            cases.push(ForwardPassTestCase {
                name: format!("single_token_{}", token),
                input_tokens: vec![token],
                expected_complexity: TestComplexity::Low,
                batch_size: 1,
            });
        }

        // Test boundary tokens
        let vocab_size = self.rust_implementation.config().vocab_size;
        let boundary_tokens = vec![0, 1, vocab_size - 2, vocab_size - 1];

        for token in boundary_tokens {
            if token < vocab_size as u32 {
                cases.push(ForwardPassTestCase {
                    name: format!("boundary_token_{}", token),
                    input_tokens: vec![token],
                    expected_complexity: TestComplexity::Low,
                    batch_size: 1,
                });
            }
        }

        Ok(cases)
    }

    fn generate_batch_cases(&self) -> Result<Vec<ForwardPassTestCase>> {
        let mut cases = Vec::new();

        for &batch_size in &self.validation_config.batch_sizes {
            if batch_size > 1 {
                // Generate identical sequences for batch
                let sequence = self.generate_test_sequence(32)?;
                let batched_sequences = vec![sequence.clone(); batch_size];

                cases.push(ForwardPassTestCase {
                    name: format!("batch_identical_sequences_size_{}", batch_size),
                    input_tokens: batched_sequences.into_iter().flatten().collect(),
                    expected_complexity: TestComplexity::Medium,
                    batch_size,
                });

                // Generate different sequences for batch
                let mut different_sequences = Vec::new();
                for i in 0..batch_size {
                    let mut seq = self.generate_test_sequence(32)?;
                    seq.push(i as u32 + 1000); // Make sequences different
                    different_sequences.push(seq);
                }

                cases.push(ForwardPassTestCase {
                    name: format!("batch_different_sequences_size_{}", batch_size),
                    input_tokens: different_sequences.into_iter().flatten().collect(),
                    expected_complexity: TestComplexity::High,
                    batch_size,
                });
            }
        }

        Ok(cases)
    }
}
```

## Generation Consistency Validation

### Token-Level Generation Comparison

```rust
impl CrossValidationFramework {
    /// Validate generation consistency between implementations
    fn validate_generation_consistency(&mut self) -> Result<GenerationValidationResult> {
        let mut validation_result = GenerationValidationResult::new();

        // Test deterministic generation (must be identical)
        let deterministic_validation = self.validate_deterministic_generation()?;
        validation_result.deterministic_validation = Some(deterministic_validation);

        // Test sampling consistency (statistical similarity)
        let sampling_validation = self.validate_sampling_consistency()?;
        validation_result.sampling_validation = Some(sampling_validation);

        // Test generation quality metrics
        let quality_validation = self.validate_generation_quality()?;
        validation_result.quality_validation = Some(quality_validation);

        Ok(validation_result)
    }

    fn validate_deterministic_generation(&mut self) -> Result<DeterministicGenerationValidation> {
        let mut validation = DeterministicGenerationValidation::new();

        // Configure deterministic generation
        let generation_config = GenerationConfig {
            max_new_tokens: self.validation_config.generation_length,
            temperature: 0.0, // Deterministic (greedy decoding)
            do_sample: false,
            seed: Some(self.validation_config.seed),
            ..Default::default()
        };

        // Generate test prompts
        let test_prompts = self.generate_test_prompts_for_generation()?;

        for (prompt_idx, prompt) in test_prompts.iter().enumerate() {
            let case_result = self.validate_single_deterministic_generation(
                prompt,
                &generation_config,
                prompt_idx
            )?;
            validation.add_case_result(case_result);
        }

        // Calculate overall deterministic accuracy
        validation.calculate_overall_accuracy();

        Ok(validation)
    }

    fn validate_single_deterministic_generation(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        case_idx: usize
    ) -> Result<DeterministicCaseResult> {
        // Tokenize prompt consistently
        let prompt_tokens = self.rust_implementation.tokenize(prompt)?;

        // Generate with C++ reference (multiple runs to verify determinism)
        let mut cpp_generations = Vec::new();
        for run in 0..3 {
            self.cpp_reference.reset_state()?;
            self.cpp_reference.set_seed(config.seed.unwrap())?;

            let cpp_result = self.cpp_reference.generate(&prompt_tokens, config)?;
            cpp_generations.push(cpp_result);
        }

        // Generate with Rust implementation (multiple runs)
        let mut rust_generations = Vec::new();
        for run in 0..3 {
            self.rust_implementation.reset_state()?;
            self.rust_implementation.set_seed(config.seed.unwrap())?;

            let rust_result = self.rust_implementation.generate(&prompt_tokens, config)?;
            rust_generations.push(rust_result);
        }

        // Verify internal determinism (same implementation, multiple runs)
        let cpp_internal_consistent = self.verify_internal_consistency(&cpp_generations);
        let rust_internal_consistent = self.verify_internal_consistency(&rust_generations);

        // Compare between implementations (first generation from each)
        let cross_implementation_comparison = self.compare_generation_outputs(
            &cpp_generations[0],
            &rust_generations[0]
        )?;

        Ok(DeterministicCaseResult {
            case_index: case_idx,
            prompt: prompt.to_string(),

            cpp_internal_consistent,
            rust_internal_consistent,
            cross_implementation_comparison,

            cpp_generations: cpp_generations.into_iter().map(|g| g.generated_tokens).collect(),
            rust_generations: rust_generations.into_iter().map(|g| g.generated_tokens).collect(),

            exact_match: cross_implementation_comparison.exact_token_match,
        })
    }

    fn compare_generation_outputs(
        &self,
        cpp_result: &GenerationResult,
        rust_result: &GenerationResult
    ) -> Result<GenerationComparison> {
        let cpp_tokens = &cpp_result.generated_tokens;
        let rust_tokens = &rust_result.generated_tokens;

        // Exact token comparison
        let exact_match = cpp_tokens == rust_tokens;

        // Token-level accuracy
        let min_length = cpp_tokens.len().min(rust_tokens.len());
        let matching_tokens = cpp_tokens.iter()
            .zip(rust_tokens.iter())
            .take(min_length)
            .filter(|(a, b)| a == b)
            .count();

        let token_accuracy = if min_length > 0 {
            matching_tokens as f64 / min_length as f64
        } else {
            0.0
        };

        // Prefix accuracy (accuracy at each position)
        let mut prefix_accuracies = Vec::new();
        for i in 1..=min_length {
            let prefix_match = cpp_tokens[..i] == rust_tokens[..i];
            prefix_accuracies.push(prefix_match);
        }

        // Text similarity (decode and compare)
        let cpp_text = self.rust_implementation.decode(&cpp_tokens)?;
        let rust_text = self.rust_implementation.decode(&rust_tokens)?;
        let text_similarity = self.calculate_text_similarity(&cpp_text, &rust_text)?;

        // Performance comparison
        let performance_comparison = PerformanceComparison {
            cpp_time: cpp_result.timing.total_time,
            rust_time: rust_result.timing.total_time,
            speedup_ratio: cpp_result.timing.total_time.as_secs_f64() /
                          rust_result.timing.total_time.as_secs_f64(),
        };

        Ok(GenerationComparison {
            exact_token_match: exact_match,
            token_accuracy,
            prefix_accuracies,
            text_similarity,
            performance_comparison,

            length_difference: rust_tokens.len() as i32 - cpp_tokens.len() as i32,
            first_divergence_position: self.find_first_divergence(cpp_tokens, rust_tokens),
        })
    }

    fn validate_sampling_consistency(&mut self) -> Result<SamplingValidationResult> {
        let mut validation = SamplingValidationResult::new();

        // Test with various sampling configurations
        let sampling_configs = vec![
            GenerationConfig {
                temperature: 0.8,
                top_k: Some(50),
                top_p: Some(0.95),
                ..Default::default()
            },
            GenerationConfig {
                temperature: 1.0,
                top_k: Some(10),
                top_p: None,
                ..Default::default()
            },
            GenerationConfig {
                temperature: 0.5,
                top_k: None,
                top_p: Some(0.9),
                ..Default::default()
            },
        ];

        for (config_idx, config) in sampling_configs.iter().enumerate() {
            let config_validation = self.validate_sampling_for_config(config, config_idx)?;
            validation.add_config_result(config_validation);
        }

        Ok(validation)
    }

    fn validate_sampling_for_config(
        &mut self,
        config: &GenerationConfig,
        config_idx: usize
    ) -> Result<SamplingConfigValidation> {
        let mut config_validation = SamplingConfigValidation::new(config.clone());

        let test_prompts = vec![
            "The quick brown fox",
            "Once upon a time",
            "In the year 2024",
            "Artificial intelligence is",
        ];

        for prompt in test_prompts {
            // Generate multiple samples from each implementation
            let cpp_samples = self.generate_multiple_samples(&self.cpp_reference, prompt, config, 10)?;
            let rust_samples = self.generate_multiple_samples(&self.rust_implementation, prompt, config, 10)?;

            // Analyze statistical properties of generations
            let cpp_stats = self.analyze_generation_statistics(&cpp_samples)?;
            let rust_stats = self.analyze_generation_statistics(&rust_samples)?;

            // Compare statistical distributions
            let distribution_comparison = self.compare_sampling_distributions(&cpp_stats, &rust_stats)?;

            config_validation.add_prompt_result(SamplingPromptResult {
                prompt: prompt.to_string(),
                cpp_statistics: cpp_stats,
                rust_statistics: rust_stats,
                distribution_comparison,
            });
        }

        Ok(config_validation)
    }
}
```

## Quantization Accuracy Validation

### Precision Loss Analysis

```rust
impl CrossValidationFramework {
    /// Validate quantization accuracy against FP32 reference
    fn validate_quantization_accuracy(&mut self) -> Result<QuantizationValidationResult> {
        let mut validation = QuantizationValidationResult::new();

        // Test different quantization types
        let quantization_types = vec![
            QuantizationType::I2S,
            QuantizationType::TL1,
            QuantizationType::TL2,
        ];

        for qtype in quantization_types {
            let qtype_validation = self.validate_quantization_type(qtype)?;
            validation.add_quantization_type_result(qtype, qtype_validation);
        }

        // Test quantization round-trip accuracy
        let roundtrip_validation = self.validate_quantization_roundtrip()?;
        validation.roundtrip_validation = Some(roundtrip_validation);

        // Test accumulation error over multiple operations
        let accumulation_validation = self.validate_quantization_error_accumulation()?;
        validation.accumulation_validation = Some(accumulation_validation);

        Ok(validation)
    }

    fn validate_quantization_type(&mut self, qtype: QuantizationType) -> Result<QuantizationTypeValidation> {
        let mut validation = QuantizationTypeValidation::new(qtype);

        // Create FP32 reference model
        let fp32_reference = self.create_fp32_reference_model()?;

        // Create quantized model
        let quantized_model = self.create_quantized_model(qtype)?;

        // Test on various input patterns
        let test_patterns = self.generate_quantization_test_patterns()?;

        for (pattern_idx, pattern) in test_patterns.iter().enumerate() {
            let case_result = self.validate_quantization_case(
                &fp32_reference,
                &quantized_model,
                pattern,
                pattern_idx
            )?;
            validation.add_case_result(case_result);
        }

        // Layer-wise accuracy analysis
        validation.layer_wise_analysis = self.analyze_layer_wise_quantization_accuracy(
            &fp32_reference,
            &quantized_model
        )?;

        Ok(validation)
    }

    fn validate_quantization_case(
        &self,
        fp32_model: &dyn BitNetInference,
        quantized_model: &dyn BitNetInference,
        test_pattern: &QuantizationTestPattern,
        case_idx: usize
    ) -> Result<QuantizationCaseResult> {
        // Forward pass with FP32 model
        let fp32_result = fp32_model.forward(&test_pattern.input_tensor, None)?;

        // Forward pass with quantized model
        let quantized_result = quantized_model.forward(&test_pattern.input_tensor, None)?;

        // Compare outputs
        let output_comparison = self.compare_logits(&fp32_result.logits, &quantized_result.logits)?;

        // Analyze quantization error patterns
        let error_analysis = self.analyze_quantization_errors(&fp32_result.logits, &quantized_result.logits)?;

        // Test extreme value handling
        let extreme_value_analysis = self.analyze_extreme_value_quantization(
            &fp32_result.logits,
            &quantized_result.logits
        )?;

        Ok(QuantizationCaseResult {
            case_index: case_idx,
            test_pattern: test_pattern.clone(),
            output_comparison,
            error_analysis,
            extreme_value_analysis,
            accuracy_score: output_comparison.pearson_correlation,
        })
    }

    fn validate_quantization_roundtrip(&self) -> Result<RoundtripValidationResult> {
        let mut validation = RoundtripValidationResult::new();

        // Test weight quantization/dequantization accuracy
        let weight_matrices = self.extract_test_weight_matrices()?;

        for (matrix_idx, weight_matrix) in weight_matrices.iter().enumerate() {
            let roundtrip_result = self.test_weight_roundtrip(weight_matrix, matrix_idx)?;
            validation.add_weight_result(roundtrip_result);
        }

        // Test activation quantization (if applicable)
        let activation_tensors = self.generate_test_activation_tensors()?;

        for (tensor_idx, activation) in activation_tensors.iter().enumerate() {
            let roundtrip_result = self.test_activation_roundtrip(activation, tensor_idx)?;
            validation.add_activation_result(roundtrip_result);
        }

        Ok(validation)
    }

    fn test_weight_roundtrip(&self, original_weight: &Tensor, matrix_idx: usize) -> Result<RoundtripTestResult> {
        let quantizer = self.get_quantizer_for_type(self.validation_config.quantization_type)?;

        // Quantize the weight
        let quantized = quantizer.quantize_tensor(original_weight)?;

        // Dequantize back to FP32
        let reconstructed = quantizer.dequantize_tensor(&quantized)?;

        // Compare original and reconstructed
        let comparison = self.compare_tensor_reconstruction(original_weight, &reconstructed)?;

        // Analyze quantization artifacts
        let artifact_analysis = self.analyze_quantization_artifacts(original_weight, &reconstructed)?;

        Ok(RoundtripTestResult {
            matrix_index: matrix_idx,
            original_shape: original_weight.dims().to_vec(),
            quantization_type: self.validation_config.quantization_type,
            comparison,
            artifact_analysis,
            compression_ratio: quantized.compression_ratio(),
        })
    }

    fn analyze_quantization_errors(&self, fp32_logits: &Tensor, quantized_logits: &Tensor) -> Result<QuantizationErrorAnalysis> {
        let fp32_data: Vec<f32> = fp32_logits.flatten_all()?.to_vec1()?;
        let quantized_data: Vec<f32> = quantized_logits.flatten_all()?.to_vec1()?;

        // Element-wise error analysis
        let errors: Vec<f32> = fp32_data.iter()
            .zip(quantized_data.iter())
            .map(|(fp32, quant)| fp32 - quant)
            .collect();

        // Statistical error metrics
        let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;
        let error_std = self.calculate_standard_deviation(&errors)?;
        let max_error = errors.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let min_error = errors.iter().fold(f32::INFINITY, |acc, &x| acc.min(x.abs()));

        // Error distribution analysis
        let error_histogram = self.create_error_histogram(&errors, 50)?;

        // Outlier analysis
        let outlier_threshold = error_std * 3.0; // 3-sigma rule
        let outlier_indices: Vec<usize> = errors.iter()
            .enumerate()
            .filter(|(_, &error)| error.abs() > outlier_threshold)
            .map(|(idx, _)| idx)
            .collect();

        // Systematic bias detection
        let systematic_bias = self.detect_systematic_bias(&fp32_data, &quantized_data)?;

        Ok(QuantizationErrorAnalysis {
            mean_error,
            error_standard_deviation: error_std,
            max_absolute_error: max_error,
            min_absolute_error: min_error,
            error_histogram,
            outlier_count: outlier_indices.len(),
            outlier_percentage: outlier_indices.len() as f64 / errors.len() as f64 * 100.0,
            systematic_bias,
        })
    }
}
```

## Performance Parity Validation

### Benchmarking Against Reference

```rust
impl CrossValidationFramework {
    /// Validate performance parity with C++ reference implementation
    fn validate_performance_parity(&mut self) -> Result<PerformanceValidationResult> {
        let mut validation = PerformanceValidationResult::new();

        // Throughput validation
        let throughput_validation = self.validate_throughput_parity()?;
        validation.throughput_validation = Some(throughput_validation);

        // Latency validation
        let latency_validation = self.validate_latency_parity()?;
        validation.latency_validation = Some(latency_validation);

        // Memory usage validation
        let memory_validation = self.validate_memory_usage_parity()?;
        validation.memory_validation = Some(memory_validation);

        // Scalability validation
        let scalability_validation = self.validate_scalability_parity()?;
        validation.scalability_validation = Some(scalability_validation);

        Ok(validation)
    }

    fn validate_throughput_parity(&mut self) -> Result<ThroughputValidationResult> {
        let mut validation = ThroughputValidationResult::new();

        // Test different sequence lengths
        let sequence_lengths = vec![1, 10, 50, 100, 256, 512];

        for seq_len in sequence_lengths {
            let throughput_comparison = self.benchmark_throughput(seq_len)?;
            validation.add_sequence_result(seq_len, throughput_comparison);
        }

        // Test batch processing
        let batch_sizes = vec![1, 2, 4, 8, 16];

        for batch_size in batch_sizes {
            if batch_size <= self.validation_config.batch_sizes.iter().max().copied().unwrap_or(1) {
                let batch_comparison = self.benchmark_batch_throughput(batch_size)?;
                validation.add_batch_result(batch_size, batch_comparison);
            }
        }

        Ok(validation)
    }

    fn benchmark_throughput(&mut self, sequence_length: usize) -> Result<ThroughputComparison> {
        let num_benchmark_runs = 20;
        let num_warmup_runs = 5;

        // Generate test sequence
        let test_tokens = self.generate_test_sequence(sequence_length)?;

        // Benchmark C++ reference
        let cpp_times = self.benchmark_implementation(
            &mut self.cpp_reference,
            &test_tokens,
            num_warmup_runs,
            num_benchmark_runs
        )?;

        // Benchmark Rust implementation
        let rust_times = self.benchmark_implementation(
            &mut self.rust_implementation,
            &test_tokens,
            num_warmup_runs,
            num_benchmark_runs
        )?;

        // Calculate statistics
        let cpp_stats = self.calculate_timing_statistics(&cpp_times)?;
        let rust_stats = self.calculate_timing_statistics(&rust_times)?;

        // Calculate throughput (tokens per second)
        let cpp_throughput = sequence_length as f64 / cpp_stats.median.as_secs_f64();
        let rust_throughput = sequence_length as f64 / rust_stats.median.as_secs_f64();

        Ok(ThroughputComparison {
            sequence_length,
            cpp_throughput,
            rust_throughput,
            speedup_ratio: rust_throughput / cpp_throughput,
            cpp_timing_stats: cpp_stats,
            rust_timing_stats: rust_stats,
            meets_performance_target: rust_throughput >= cpp_throughput * 0.5, // Within 50% is acceptable
        })
    }

    fn validate_memory_usage_parity(&mut self) -> Result<MemoryValidationResult> {
        let mut validation = MemoryValidationResult::new();

        // Test model loading memory
        let model_memory_comparison = self.compare_model_loading_memory()?;
        validation.model_loading_memory = Some(model_memory_comparison);

        // Test inference memory
        let inference_memory_comparison = self.compare_inference_memory()?;
        validation.inference_memory = Some(inference_memory_comparison);

        // Test KV cache memory
        let cache_memory_comparison = self.compare_cache_memory()?;
        validation.cache_memory = Some(cache_memory_comparison);

        // Memory growth analysis
        let memory_growth_analysis = self.analyze_memory_growth()?;
        validation.memory_growth = Some(memory_growth_analysis);

        Ok(validation)
    }

    fn compare_inference_memory(&mut self) -> Result<MemoryComparison> {
        // Measure memory before inference
        let cpp_memory_before = self.measure_memory_usage(&self.cpp_reference)?;
        let rust_memory_before = self.measure_memory_usage(&self.rust_implementation)?;

        // Run inference with memory tracking
        let test_tokens = self.generate_test_sequence(256)?;

        let cpp_memory_peak = self.measure_memory_during_inference(&mut self.cpp_reference, &test_tokens)?;
        let rust_memory_peak = self.measure_memory_during_inference(&mut self.rust_implementation, &test_tokens)?;

        // Calculate memory usage differences
        let cpp_inference_memory = cpp_memory_peak - cpp_memory_before.current_usage;
        let rust_inference_memory = rust_memory_peak - rust_memory_before.current_usage;

        Ok(MemoryComparison {
            cpp_memory_usage: cpp_inference_memory,
            rust_memory_usage: rust_inference_memory,
            memory_ratio: rust_inference_memory as f64 / cpp_inference_memory as f64,
            acceptable: rust_inference_memory <= cpp_inference_memory * 2, // Within 2x is acceptable
        })
    }
}
```

## Statistical Analysis Framework

### Comprehensive Statistical Validation

```rust
/// Advanced statistical analyzer for cross-validation
pub struct StatisticalAnalyzer {
    // Configuration
    significance_level: f64, // 0.05 for 95% confidence

    // Statistical test implementations
    correlation_analyzer: CorrelationAnalyzer,
    distribution_tester: DistributionTester,
    hypothesis_tester: HypothesisTester,
}

impl StatisticalAnalyzer {
    pub fn new() -> Self {
        Self {
            significance_level: 0.05,
            correlation_analyzer: CorrelationAnalyzer::new(),
            distribution_tester: DistributionTester::new(),
            hypothesis_tester: HypothesisTester::new(),
        }
    }

    /// Comprehensive correlation analysis
    pub fn comprehensive_correlation_analysis(
        &self,
        reference_data: &[f32],
        test_data: &[f32]
    ) -> Result<CorrelationAnalysisResult> {
        // Pearson correlation (linear relationship)
        let pearson_r = self.pearson_correlation(reference_data, test_data)?;
        let pearson_p_value = self.pearson_p_value(reference_data, test_data, pearson_r)?;
        let pearson_confidence_interval = self.pearson_confidence_interval(pearson_r, reference_data.len())?;

        // Spearman correlation (monotonic relationship)
        let spearman_rho = self.spearman_correlation(reference_data, test_data)?;
        let spearman_p_value = self.spearman_p_value(reference_data, test_data, spearman_rho)?;

        // Kendall's tau (rank correlation)
        let kendall_tau = self.kendall_tau(reference_data, test_data)?;

        // Coefficient of determination
        let r_squared = pearson_r * pearson_r;

        Ok(CorrelationAnalysisResult {
            pearson_correlation: pearson_r,
            pearson_p_value,
            pearson_confidence_interval,
            pearson_significant: pearson_p_value < self.significance_level,

            spearman_correlation: spearman_rho,
            spearman_p_value,
            spearman_significant: spearman_p_value < self.significance_level,

            kendall_tau,
            r_squared,

            overall_assessment: self.assess_correlation_strength(pearson_r, spearman_rho, kendall_tau),
        })
    }

    /// Advanced distribution comparison
    pub fn compare_distributions(
        &self,
        reference_data: &[f32],
        test_data: &[f32]
    ) -> Result<DistributionComparisonResult> {
        // Kolmogorov-Smirnov test for distribution equality
        let ks_test_result = self.kolmogorov_smirnov_test(reference_data, test_data)?;

        // Anderson-Darling test for distribution goodness of fit
        let ad_test_result = self.anderson_darling_test(reference_data, test_data)?;

        // Mann-Whitney U test for distribution differences
        let mann_whitney_result = self.mann_whitney_u_test(reference_data, test_data)?;

        // Distribution moments comparison
        let moments_comparison = self.compare_distribution_moments(reference_data, test_data)?;

        // Quantile-Quantile analysis
        let qq_analysis = self.quantile_quantile_analysis(reference_data, test_data)?;

        // Jensen-Shannon divergence
        let js_divergence = self.jensen_shannon_divergence(reference_data, test_data)?;

        Ok(DistributionComparisonResult {
            ks_test: ks_test_result,
            ad_test: ad_test_result,
            mann_whitney_test: mann_whitney_result,
            moments_comparison,
            qq_analysis,
            js_divergence,
            distributions_similar: self.assess_distribution_similarity(
                &ks_test_result, &ad_test_result, &mann_whitney_result
            ),
        })
    }

    /// Outlier detection and analysis
    pub fn detect_and_analyze_outliers(
        &self,
        reference_data: &[f32],
        test_data: &[f32]
    ) -> Result<OutlierAnalysisResult> {
        // Z-score based outlier detection
        let reference_z_outliers = self.detect_z_score_outliers(reference_data, 3.0)?;
        let test_z_outliers = self.detect_z_score_outliers(test_data, 3.0)?;

        // Interquartile range (IQR) based outlier detection
        let reference_iqr_outliers = self.detect_iqr_outliers(reference_data)?;
        let test_iqr_outliers = self.detect_iqr_outliers(test_data)?;

        // Modified Z-score using median absolute deviation
        let reference_mad_outliers = self.detect_mad_outliers(reference_data)?;
        let test_mad_outliers = self.detect_mad_outliers(test_data)?;

        // Isolation forest for multivariate outlier detection
        let isolation_outliers = self.detect_isolation_outliers(reference_data, test_data)?;

        Ok(OutlierAnalysisResult {
            reference_z_outliers,
            test_z_outliers,
            reference_iqr_outliers,
            test_iqr_outliers,
            reference_mad_outliers,
            test_mad_outliers,
            isolation_outliers,
            outlier_impact_analysis: self.analyze_outlier_impact(reference_data, test_data)?,
        })
    }

    fn pearson_correlation(&self, x: &[f32], y: &[f32]) -> Result<f64> {
        if x.len() != y.len() || x.is_empty() {
            return Err(StatisticalError::InvalidInput);
        }

        let n = x.len() as f64;
        let mean_x = x.iter().map(|&v| v as f64).sum::<f64>() / n;
        let mean_y = y.iter().map(|&v| v as f64).sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let xi = xi as f64;
            let yi = yi as f64;
            let dx = xi - mean_x;
            let dy = yi - mean_y;

            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator == 0.0 {
            Ok(0.0) // Perfect correlation when one variable is constant
        } else {
            Ok(numerator / denominator)
        }
    }
}
```

## Integration with xtask

### Command-Line Cross-Validation

```rust
// xtask integration for cross-validation
// Location: xtask/src/crossval.rs

use clap::Args;
use crossval::CrossValidationFramework;

#[derive(Args)]
pub struct CrossValArgs {
    /// Path to the model file
    #[arg(long, short)]
    pub model: String,

    /// Path to the tokenizer file
    #[arg(long, short)]
    pub tokenizer: Option<String>,

    /// Quantization type to test
    #[arg(long, default_value = "i2s")]
    pub quantization: String,

    /// Device to use (cpu/gpu)
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Random seed for deterministic validation
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Number of test cases to run
    #[arg(long, default_value = "100")]
    pub num_tests: usize,

    /// Enable comprehensive validation (slower)
    #[arg(long)]
    pub comprehensive: bool,

    /// Output detailed report
    #[arg(long)]
    pub detailed: bool,

    /// Save report to file
    #[arg(long)]
    pub output: Option<String>,
}

pub fn run_crossval(args: CrossValArgs) -> Result<()> {
    println!("🔍 Starting cross-validation against C++ reference implementation");

    // Configure cross-validation
    let config = CrossValidationConfig {
        model_path: args.model.clone(),
        tokenizer_path: args.tokenizer.unwrap_or_else(|| {
            // Auto-detect tokenizer path
            let model_dir = std::path::Path::new(&args.model).parent().unwrap();
            model_dir.join("tokenizer.json").to_string_lossy().to_string()
        }),
        quantization_type: parse_quantization_type(&args.quantization)?,
        device: parse_device(&args.device)?,
        seed: args.seed,
        num_test_cases: args.num_tests,
        ..Default::default()
    };

    // Initialize cross-validation framework
    let mut framework = CrossValidationFramework::new(
        &config.model_path,
        &config.tokenizer_path,
        config
    )?;

    println!("✅ Initialized C++ reference and Rust implementation");

    // Run cross-validation
    let start_time = std::time::Instant::now();
    let report = framework.run_cross_validation()?;
    let duration = start_time.elapsed();

    // Display results
    display_validation_results(&report, args.detailed);

    // Save detailed report if requested
    if let Some(output_path) = args.output {
        save_detailed_report(&report, &output_path)?;
        println!("📊 Detailed report saved to: {}", output_path);
    }

    println!("⏱️  Cross-validation completed in {:.2}s", duration.as_secs_f64());

    // Exit with appropriate code
    if report.overall_assessment.passed {
        println!("✅ Cross-validation PASSED - Rust implementation matches C++ reference");
        Ok(())
    } else {
        println!("❌ Cross-validation FAILED - Found significant deviations");
        Err(anyhow::anyhow!("Cross-validation failed: {}", report.overall_assessment.failure_summary))
    }
}

fn display_validation_results(report: &CrossValidationReport, detailed: bool) {
    println!("\n📊 Cross-Validation Results");
    println!("=" .repeat(60));

    // Forward pass validation
    if let Some(forward_result) = &report.phases.get("forward_pass") {
        println!("🔄 Forward Pass Validation:");
        println!("   Correlation: {:.6}", forward_result.statistical_summary.mean_correlation);
        println!("   MSE: {:.2e}", forward_result.statistical_summary.mean_mse);
        println!("   Passed: {}/{} test cases",
               forward_result.passed_cases, forward_result.total_cases);
    }

    // Generation validation
    if let Some(gen_result) = &report.phases.get("generation") {
        if let Some(det_val) = &gen_result.deterministic_validation {
            println!("🎯 Deterministic Generation:");
            println!("   Exact matches: {:.1}%", det_val.exact_match_percentage);
            println!("   Token accuracy: {:.3}", det_val.average_token_accuracy);
        }
    }

    // Quantization validation
    if let Some(quant_result) = &report.phases.get("quantization") {
        println!("⚡ Quantization Accuracy:");
        println!("   Quantization correlation: {:.6}", quant_result.overall_correlation);
        println!("   Compression ratio: {:.1}x", quant_result.compression_ratio);
    }

    // Performance validation
    if let Some(perf_result) = &report.phases.get("performance") {
        println!("🚀 Performance Comparison:");
        if let Some(throughput) = &perf_result.throughput_validation {
            println!("   Rust vs C++ throughput: {:.2}x", throughput.average_speedup_ratio);
        }
        if let Some(memory) = &perf_result.memory_validation {
            println!("   Memory usage ratio: {:.2}x", memory.average_memory_ratio);
        }
    }

    // Overall assessment
    println!("\n🎯 Overall Assessment:");
    println!("   Status: {}", if report.overall_assessment.passed { "PASSED ✅" } else { "FAILED ❌" });
    println!("   Confidence: {:.1}%", report.overall_assessment.confidence_percentage);

    if !report.overall_assessment.passed {
        println!("   Issues:");
        for issue in &report.overall_assessment.critical_issues {
            println!("     - {}", issue);
        }
    }

    if detailed {
        display_detailed_statistics(report);
    }
}

// Command integration in main xtask
// Location: xtask/src/main.rs
#[derive(Subcommand)]
pub enum Commands {
    // ... existing commands

    /// Cross-validate against C++ reference implementation
    Crossval(crossval::CrossValArgs),
}

pub fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        // ... existing command handlers

        Commands::Crossval(args) => crossval::run_crossval(args),
    }
}
```

## Testing Integration

### Comprehensive Test Coverage

```rust
#[cfg(test)]
mod cross_validation_tests {
    use super::*;

    #[test]
    fn test_cross_validation_framework_initialization() { // AC:4
        let config = CrossValidationConfig {
            model_path: "tests/data/test_model.gguf".to_string(),
            tokenizer_path: "tests/data/tokenizer.json".to_string(),
            ..Default::default()
        };

        let framework = CrossValidationFramework::new(
            &config.model_path,
            &config.tokenizer_path,
            config
        );

        assert!(framework.is_ok(), "Framework initialization failed");
    }

    #[test]
    fn test_deterministic_generation_validation() { // AC:7
        let mut framework = create_test_framework().unwrap();

        let validation_result = framework.validate_deterministic_generation().unwrap();

        // All deterministic generations should be identical
        assert!(validation_result.overall_accuracy > 0.99,
               "Deterministic accuracy too low: {:.3}", validation_result.overall_accuracy);

        // Internal consistency check
        for case in &validation_result.case_results {
            assert!(case.cpp_internal_consistent, "C++ reference not internally consistent");
            assert!(case.rust_internal_consistent, "Rust implementation not internally consistent");
        }
    }

    #[test]
    fn test_quantization_accuracy_validation() { // AC:4
        let mut framework = create_test_framework().unwrap();

        let validation_result = framework.validate_quantization_accuracy().unwrap();

        // Check quantization correlation threshold
        for (qtype, type_validation) in &validation_result.quantization_types {
            assert!(type_validation.overall_correlation > 0.99,
                   "Quantization {:?} correlation too low: {:.6}", qtype, type_validation.overall_correlation);

            // Check layer-wise accuracy
            if let Some(layer_analysis) = &type_validation.layer_wise_analysis {
                for layer_result in &layer_analysis.layer_results {
                    assert!(layer_result.accuracy > 0.95,
                           "Layer {} accuracy too low: {:.3}", layer_result.layer_index, layer_result.accuracy);
                }
            }
        }
    }

    #[test]
    fn test_statistical_analysis_accuracy() { // AC:4
        let analyzer = StatisticalAnalyzer::new();

        // Test with known correlated data
        let reference_data: Vec<f32> = (0..1000).map(|x| x as f32 * 0.5 + 10.0).collect();
        let test_data: Vec<f32> = reference_data.iter().map(|&x| x + 0.01).collect(); // Small noise

        let correlation_result = analyzer.comprehensive_correlation_analysis(&reference_data, &test_data).unwrap();

        assert!(correlation_result.pearson_correlation > 0.999,
               "Correlation analysis inaccurate: {:.6}", correlation_result.pearson_correlation);
        assert!(correlation_result.pearson_significant, "Correlation should be significant");
    }

    #[test]
    fn test_performance_parity_validation() { // AC:5
        let mut framework = create_test_framework().unwrap();

        let performance_validation = framework.validate_performance_parity().unwrap();

        // Check throughput parity
        if let Some(throughput_val) = &performance_validation.throughput_validation {
            // Rust implementation should be within 2x of C++ reference
            for result in &throughput_val.sequence_results {
                assert!(result.speedup_ratio > 0.5 && result.speedup_ratio < 2.0,
                       "Performance ratio out of acceptable range: {:.2}x", result.speedup_ratio);
            }
        }

        // Check memory usage parity
        if let Some(memory_val) = &performance_validation.memory_validation {
            if let Some(inference_memory) = &memory_val.inference_memory {
                assert!(inference_memory.memory_ratio < 3.0,
                       "Memory usage ratio too high: {:.2}x", inference_memory.memory_ratio);
            }
        }
    }

    #[test]
    fn test_error_accumulation_validation() { // AC:4
        let mut framework = create_test_framework().unwrap();

        // Test with long sequences to check error accumulation
        framework.validation_config.max_sequence_length = 1024;
        framework.validation_config.generation_length = 200;

        let validation_result = framework.run_cross_validation().unwrap();

        // Error should not accumulate significantly over long sequences
        if let Some(forward_result) = validation_result.phases.get("forward_pass") {
            // Check that correlation remains high even for long sequences
            let long_sequence_cases: Vec<_> = forward_result.case_results.iter()
                .filter(|case| case.test_case.input_tokens.len() > 512)
                .collect();

            if !long_sequence_cases.is_empty() {
                let avg_correlation = long_sequence_cases.iter()
                    .map(|case| case.logits_comparison.pearson_correlation)
                    .sum::<f64>() / long_sequence_cases.len() as f64;

                assert!(avg_correlation > 0.995,
                       "Long sequence correlation degraded: {:.6}", avg_correlation);
            }
        }
    }

    #[test]
    fn test_batch_processing_validation() { // AC:3, AC:4
        let mut framework = create_test_framework().unwrap();
        framework.validation_config.batch_sizes = vec![1, 2, 4, 8];

        let validation_result = framework.validate_forward_passes().unwrap();

        // Check that batch processing maintains accuracy
        let batch_cases: Vec<_> = validation_result.case_results.iter()
            .filter(|case| case.test_case.batch_size > 1)
            .collect();

        for case in batch_cases {
            assert!(case.logits_comparison.pearson_correlation > 0.999,
                   "Batch processing accuracy degraded: {:.6}", case.logits_comparison.pearson_correlation);
        }
    }

    // Helper function for test setup
    fn create_test_framework() -> Result<CrossValidationFramework> {
        let config = CrossValidationConfig {
            model_path: "tests/data/test_model.gguf".to_string(),
            tokenizer_path: "tests/data/tokenizer.json".to_string(),
            num_test_cases: 50, // Smaller for faster testing
            max_sequence_length: 256,
            generation_length: 20,
            ..Default::default()
        };

        CrossValidationFramework::new(
            &config.model_path,
            &config.tokenizer_path,
            config
        )
    }
}
```

This comprehensive cross-validation specification ensures >99.9% accuracy correlation with C++ reference implementations while providing detailed diagnostic capabilities, statistical analysis, and seamless integration with BitNet.rs development workflows through xtask commands and automated testing.
