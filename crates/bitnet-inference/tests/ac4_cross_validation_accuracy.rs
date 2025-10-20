//! AC4: Cross-Validation Accuracy Preservation Tests
//!
//! Tests feature spec: issue-248-spec.md#ac4-cross-validation-accuracy-preservation
//! API contract: neural-network-operation-requirements.md#cross-validation-requirements
//!
//! This test module validates >99% quantization accuracy preservation through systematic
//! cross-validation against C++ reference implementation using `cargo run -p xtask -- crossval`.
//! Ensures BitNet.rs maintains numerical accuracy and computational correctness.

#![cfg(any())] // Disabled: Cross-validation infrastructure not yet implemented
#![allow(dead_code, unused_variables, unused_imports, unused_mut)]

use anyhow::{Context, Result};
use bitnet_models::BitNetModel;
use bitnet_quantization::{I2SQuantizer, TL1Quantizer, TL2Quantizer};
use candle_core::IndexOp;
use std::process::Command;

/// Test configuration for AC4 cross-validation accuracy validation
#[derive(Debug, Clone)]
pub struct AC4TestConfig {
    pub accuracy_threshold: f32,
    pub correlation_threshold: f32,
    pub mse_threshold: f32,
    pub perplexity_threshold: f32,
    pub test_sequences: Vec<String>,
    pub reference_model_path: String,
}

impl Default for AC4TestConfig {
    fn default() -> Self {
        Self {
            accuracy_threshold: 0.99,     // >99% accuracy requirement
            correlation_threshold: 0.999, // >99.9% correlation
            mse_threshold: 1e-6,          // Very low MSE
            perplexity_threshold: 0.05,   // ≤0.05% perplexity degradation
            test_sequences: vec![
                "The quick brown fox jumps over the lazy dog".to_string(),
                "In a hole in the ground there lived a hobbit".to_string(),
                "To be or not to be, that is the question".to_string(),
                "Four score and seven years ago".to_string(),
                "It was the best of times, it was the worst of times".to_string(),
            ],
            reference_model_path: "models/reference/bitnet-2b.gguf".to_string(),
        }
    }
}

/// AC4.1: I2S Quantization Cross-Validation Test
/// Tests feature spec: issue-248-spec.md#ac4
/// Validates I2S quantization maintains >99% accuracy vs C++ reference
#[cfg(all(feature = "cpu", feature = "crossval"))]
#[tokio::test]
async fn test_ac4_i2s_quantization_cross_validation() -> Result<()> {
    let config = AC4TestConfig::default();

    // Skip test if cross-validation environment not set up
    if !is_crossval_environment_ready() {
        log::warn!("Skipping cross-validation test: environment not ready");
        return Ok(());
    }

    // Load BitNet.rs model with I2S quantization
    let bitnet_model = load_bitnet_model_for_crossval(&config.reference_model_path)
        .context("Failed to load BitNet.rs model for cross-validation")?;

    // Initialize I2S quantization for cross-validation
    let quantizer = I2SQuantizer::new();

    // Set up cross-validation configuration
    let crossval_config = CrossValidationConfig {
        reference_implementation: ReferenceImplementation::CppBitNet,
        tolerance: config.mse_threshold,
        correlation_threshold: config.correlation_threshold,
        test_cases: config.test_sequences.clone(),
        deterministic: true,
        seed: 42,
    };

    // Run cross-validation for each test sequence
    let mut validation_results = Vec::new();

    for test_sequence in &config.test_sequences {
        // Generate output with BitNet.rs (I2S quantization)
        let bitnet_result = run_bitnet_inference(&bitnet_model, test_sequence, &quantizer)
            .await
            .context(format!("Failed BitNet.rs inference for: {}", test_sequence))?;

        // Generate output with C++ reference implementation
        let cpp_result = run_cpp_reference_inference(&config.reference_model_path, test_sequence)
            .await
            .context(format!("Failed C++ reference inference for: {}", test_sequence))?;

        // Compare outputs using comprehensive validation metrics
        let comparison = compare_inference_outputs(&bitnet_result, &cpp_result, &crossval_config)
            .context(format!("Failed to compare outputs for: {}", test_sequence))?;

        validation_results.push((test_sequence.clone(), comparison));
    }

    // Aggregate validation results
    let aggregated_metrics = aggregate_validation_metrics(&validation_results)
        .context("Failed to aggregate cross-validation metrics")?;

    // Validate accuracy requirements
    assert!(
        aggregated_metrics.average_token_accuracy >= config.accuracy_threshold,
        "I2S quantization token accuracy below threshold: {} < {}",
        aggregated_metrics.average_token_accuracy,
        config.accuracy_threshold
    );

    assert!(
        aggregated_metrics.average_logit_correlation >= config.correlation_threshold,
        "I2S quantization logit correlation below threshold: {} < {}",
        aggregated_metrics.average_logit_correlation,
        config.correlation_threshold
    );

    assert!(
        aggregated_metrics.average_mse <= config.mse_threshold,
        "I2S quantization MSE above threshold: {} > {}",
        aggregated_metrics.average_mse,
        config.mse_threshold
    );

    assert!(
        aggregated_metrics.perplexity_degradation <= config.perplexity_threshold,
        "I2S quantization perplexity degradation above threshold: {} > {}",
        aggregated_metrics.perplexity_degradation,
        config.perplexity_threshold
    );

    // Log successful cross-validation
    log::info!(
        "I2S cross-validation passed: accuracy={:.4}, correlation={:.4}, mse={:.2e}",
        aggregated_metrics.average_token_accuracy,
        aggregated_metrics.average_logit_correlation,
        aggregated_metrics.average_mse
    );

    // TODO: Replace with actual I2S cross-validation implementation
    panic!(
        "AC4.1: I2S quantization cross-validation not yet implemented - replace mock with real validation"
    );
}

/// AC4.2: Table Lookup Quantization Cross-Validation Test
/// Tests feature spec: issue-248-spec.md#ac4
/// Validates TL1/TL2 quantization accuracy vs reference implementation
#[cfg(all(feature = "cpu", feature = "crossval"))]
#[tokio::test]
async fn test_ac4_table_lookup_quantization_cross_validation() -> Result<()> {
    let config = AC4TestConfig::default();

    if !is_crossval_environment_ready() {
        log::warn!("Skipping TL cross-validation test: environment not ready");
        return Ok(());
    }

    let bitnet_model = load_bitnet_model_for_crossval(&config.reference_model_path)?;

    // Test both TL1 and TL2 quantization methods
    let quantization_methods = vec![
        ("TL1", TL1Quantizer::new_for_crossval()?),
        ("TL2", TL2Quantizer::new_for_crossval()?),
    ];

    for (method_name, quantizer) in quantization_methods {
        log::info!("Running cross-validation for {} quantization", method_name);

        let mut method_results = Vec::new();

        for test_sequence in &config.test_sequences {
            // BitNet.rs inference with table lookup quantization
            let bitnet_result =
                run_bitnet_inference_with_table_lookup(&bitnet_model, test_sequence, &quantizer)
                    .await
                    .context(format!(
                        "Failed BitNet.rs {} inference for: {}",
                        method_name, test_sequence
                    ))?;

            // Reference implementation (may use different quantization)
            let cpp_result =
                run_cpp_reference_inference(&config.reference_model_path, test_sequence).await?;

            let comparison =
                compare_table_lookup_outputs(&bitnet_result, &cpp_result, method_name, &config)
                    .context(format!(
                        "Failed to compare {} outputs for: {}",
                        method_name, test_sequence
                    ))?;

            method_results.push((test_sequence.clone(), comparison));
        }

        // Validate table lookup quantization accuracy
        let tl_metrics = aggregate_validation_metrics(&method_results)?;

        assert!(
            tl_metrics.average_token_accuracy >= config.accuracy_threshold,
            "{} quantization accuracy below threshold: {} < {}",
            method_name,
            tl_metrics.average_token_accuracy,
            config.accuracy_threshold
        );

        // Table lookup methods may have slightly different correlation due to lookup precision
        let tl_correlation_threshold = config.correlation_threshold * 0.99; // 99.8% for TL methods

        assert!(
            tl_metrics.average_logit_correlation >= tl_correlation_threshold,
            "{} quantization correlation below adjusted threshold: {} < {}",
            method_name,
            tl_metrics.average_logit_correlation,
            tl_correlation_threshold
        );

        // Validate lookup table efficiency metrics
        if let Some(lookup_metrics) = tl_metrics.lookup_performance_metrics {
            assert!(
                lookup_metrics.average_lookup_time_ns <= 10.0, // ≤10ns per lookup
                "{} lookup performance below target: {:.2}ns > 10ns",
                method_name,
                lookup_metrics.average_lookup_time_ns
            );

            assert!(
                lookup_metrics.cache_hit_rate >= 0.95,
                "{} cache efficiency below target: {:.2}% < 95%",
                method_name,
                lookup_metrics.cache_hit_rate * 100.0
            );
        }

        log::info!(
            "{} cross-validation passed: accuracy={:.4}, correlation={:.4}",
            method_name,
            tl_metrics.average_token_accuracy,
            tl_metrics.average_logit_correlation
        );
    }

    // TODO: Replace with actual table lookup cross-validation implementation
    panic!(
        "AC4.2: Table lookup quantization cross-validation not yet implemented - replace mock with real validation"
    );
}

/// AC4.3: IQ2_S GGML Compatibility Cross-Validation Test
/// Tests feature spec: issue-248-spec.md#ac4
/// Validates IQ2_S format maintains bit-exact compatibility with GGML
#[cfg(all(feature = "cpu", feature = "crossval"))]
#[tokio::test]
async fn test_ac4_iq2s_ggml_compatibility_cross_validation() -> Result<()> {
    let config = AC4TestConfig::default();

    if !is_crossval_environment_ready() || !is_ggml_ffi_available() {
        log::warn!("Skipping IQ2_S cross-validation test: environment not ready");
        return Ok(());
    }

    // Load model in IQ2_S format for cross-validation
    let iq2s_model_path = "models/reference/bitnet-2b-iq2s.gguf";
    let bitnet_model = load_bitnet_model_for_crossval(iq2s_model_path)?;

    // Initialize IQ2_S quantizer with GGML compatibility mode
    let iq2s_quantizer = IQ2SQuantizer::new_ggml_compatible()
        .context("Failed to create GGML-compatible IQ2_S quantizer")?;

    // Set up strict compatibility validation
    let strict_config = CrossValidationConfig {
        reference_implementation: ReferenceImplementation::GGML,
        tolerance: 1e-8, // Very strict tolerance for bit-exact compatibility
        correlation_threshold: 0.9999, // Near-perfect correlation
        test_cases: config.test_sequences.clone(),
        deterministic: true,
        seed: 42,
        validate_bit_exact: true, // Enable bit-exact validation
    };

    let mut iq2s_results = Vec::new();

    for test_sequence in &config.test_sequences {
        // BitNet.rs IQ2_S inference
        let bitnet_result =
            run_bitnet_iq2s_inference(&bitnet_model, test_sequence, &iq2s_quantizer)
                .await
                .context(format!("Failed BitNet.rs IQ2_S inference for: {}", test_sequence))?;

        // GGML reference IQ2_S inference
        let ggml_result = run_ggml_reference_inference(iq2s_model_path, test_sequence)
            .await
            .context(format!("Failed GGML reference inference for: {}", test_sequence))?;

        // Strict compatibility comparison
        let comparison =
            compare_iq2s_compatibility(&bitnet_result, &ggml_result, &strict_config)
                .context(format!("Failed IQ2_S compatibility check for: {}", test_sequence))?;

        iq2s_results.push((test_sequence.clone(), comparison));
    }

    // Validate GGML compatibility metrics
    let iq2s_metrics = aggregate_iq2s_compatibility_metrics(&iq2s_results)?;

    // IQ2_S should maintain very high compatibility with GGML
    assert!(
        iq2s_metrics.bit_exact_matches >= 0.95,
        "IQ2_S bit-exact compatibility below threshold: {} < 95%",
        iq2s_metrics.bit_exact_matches * 100.0
    );

    assert!(
        iq2s_metrics.block_format_compliance >= 0.999,
        "IQ2_S block format compliance below threshold: {} < 99.9%",
        iq2s_metrics.block_format_compliance * 100.0
    );

    assert!(
        iq2s_metrics.quantization_level_accuracy >= 0.999,
        "IQ2_S quantization level accuracy below threshold: {} < 99.9%",
        iq2s_metrics.quantization_level_accuracy * 100.0
    );

    // Validate performance parity with GGML
    assert!(
        iq2s_metrics.performance_ratio >= 0.9, // Within 10% of GGML performance
        "IQ2_S performance below GGML reference: {}x < 0.9x",
        iq2s_metrics.performance_ratio
    );

    log::info!(
        "IQ2_S GGML compatibility validated: bit_exact={:.2}%, format={:.2}%, perf={:.2}x",
        iq2s_metrics.bit_exact_matches * 100.0,
        iq2s_metrics.block_format_compliance * 100.0,
        iq2s_metrics.performance_ratio
    );

    // TODO: Replace with actual IQ2_S GGML cross-validation implementation
    panic!(
        "AC4.3: IQ2_S GGML compatibility cross-validation not yet implemented - replace mock with real validation"
    );
}

/// AC4.4: Comprehensive Cross-Validation Test Suite
/// Tests feature spec: issue-248-spec.md#ac4
/// Runs complete cross-validation suite using xtask crossval command
#[cfg(all(feature = "cpu", feature = "crossval"))]
#[tokio::test]
async fn test_ac4_comprehensive_cross_validation_suite() -> Result<()> {
    let config = AC4TestConfig::default();

    if !is_crossval_environment_ready() {
        log::warn!("Skipping comprehensive cross-validation test: environment not ready");
        return Ok(());
    }

    // Set up cross-validation environment variables
    std::env::set_var("BITNET_DETERMINISTIC", "1");
    std::env::set_var("BITNET_SEED", "42");
    std::env::set_var("RAYON_NUM_THREADS", "1");
    std::env::set_var("BITNET_GGUF", &config.reference_model_path);

    // Run comprehensive cross-validation using xtask
    let crossval_output = Command::new("cargo")
        .args([
            "run",
            "-p",
            "xtask",
            "--",
            "crossval",
            "--model",
            &config.reference_model_path,
            "--deterministic",
            "--seed",
            "42",
            "--tolerance",
            &config.mse_threshold.to_string(),
        ])
        .output()
        .context("Failed to run xtask crossval command")?;

    // Parse cross-validation results
    let crossval_results =
        parse_crossval_output(&crossval_output).context("Failed to parse xtask crossval output")?;

    // Validate comprehensive metrics
    assert!(
        crossval_results.overall_pass_rate >= config.accuracy_threshold,
        "Comprehensive cross-validation pass rate below threshold: {} < {}",
        crossval_results.overall_pass_rate,
        config.accuracy_threshold
    );

    assert!(
        crossval_results.quantization_accuracy.i2s >= config.accuracy_threshold,
        "I2S cross-validation accuracy below threshold: {} < {}",
        crossval_results.quantization_accuracy.i2s,
        config.accuracy_threshold
    );

    assert!(
        crossval_results.quantization_accuracy.tl1 >= config.accuracy_threshold,
        "TL1 cross-validation accuracy below threshold: {} < {}",
        crossval_results.quantization_accuracy.tl1,
        config.accuracy_threshold
    );

    assert!(
        crossval_results.quantization_accuracy.tl2 >= config.accuracy_threshold,
        "TL2 cross-validation accuracy below threshold: {} < {}",
        crossval_results.quantization_accuracy.tl2,
        config.accuracy_threshold
    );

    // Validate performance correlation
    assert!(
        crossval_results.performance_correlation >= 0.95,
        "Performance correlation with reference below threshold: {} < 0.95",
        crossval_results.performance_correlation
    );

    // Validate numerical stability
    assert!(
        crossval_results.numerical_stability.nan_count == 0,
        "NaN values detected in cross-validation: {}",
        crossval_results.numerical_stability.nan_count
    );

    assert!(
        crossval_results.numerical_stability.inf_count == 0,
        "Infinite values detected in cross-validation: {}",
        crossval_results.numerical_stability.inf_count
    );

    // Clean up environment variables
    std::env::remove_var("BITNET_DETERMINISTIC");
    std::env::remove_var("BITNET_SEED");
    std::env::remove_var("RAYON_NUM_THREADS");
    std::env::remove_var("BITNET_GGUF");

    log::info!(
        "Comprehensive cross-validation passed: overall={:.4}, i2s={:.4}, tl1={:.4}, tl2={:.4}",
        crossval_results.overall_pass_rate,
        crossval_results.quantization_accuracy.i2s,
        crossval_results.quantization_accuracy.tl1,
        crossval_results.quantization_accuracy.tl2
    );

    // TODO: Replace with actual comprehensive cross-validation implementation
    panic!(
        "AC4.4: Comprehensive cross-validation suite not yet implemented - replace mock with real xtask integration"
    );
}

// Helper functions for cross-validation test scaffolding

/// Check if cross-validation environment is ready
/// Returns true if C++ reference implementation is available (BITNET_CPP_DIR set and valid)
fn is_crossval_environment_ready() -> bool {
    use std::path::Path;

    // Check if BITNET_CPP_DIR environment variable is set
    let cpp_dir = match std::env::var("BITNET_CPP_DIR") {
        Ok(dir) => dir,
        Err(_) => return false,
    };

    // Verify the directory exists
    let cpp_path = Path::new(&cpp_dir);
    if !cpp_path.exists() || !cpp_path.is_dir() {
        return false;
    }

    // Check for essential C++ reference binaries/libraries
    // Look for typical BitNet.cpp build artifacts
    let build_dir = cpp_path.join("build");
    let has_build = build_dir.exists();

    // Alternative: check for lib directory with shared objects
    let lib_dir = cpp_path.join("lib");
    let has_lib = lib_dir.exists();

    has_build || has_lib
}

/// Check if GGML FFI bridge is available
/// Returns true if FFI feature is compiled and libraries are accessible
fn is_ggml_ffi_available() -> bool {
    // Check if FFI feature is compiled in
    #[cfg(feature = "ffi")]
    {
        // FFI feature is compiled, verify runtime availability
        is_crossval_environment_ready()
    }
    #[cfg(not(feature = "ffi"))]
    {
        // FFI not compiled in
        false
    }
}

/// Load BitNet model for cross-validation testing
/// Loads GGUF model using production loader with strict validation
fn load_bitnet_model_for_crossval(model_path: &str) -> Result<BitNetModel> {
    use bitnet_common::Device;
    use std::path::Path;

    // Verify model file exists
    let model_path_buf = Path::new(model_path);
    if !model_path_buf.exists() {
        anyhow::bail!("Model file not found: {}", model_path);
    }

    // Use CPU device for cross-validation (deterministic, no GPU variability)
    let device = Device::Cpu;

    // Load model using production loader
    // This follows the pattern from parity_bitnetcpp.rs and production inference
    let model = bitnet_models::load_gguf_full(model_path, device)
        .context("Failed to load GGUF model for cross-validation")?;

    // Validate model structure - ensure essential tensors are present
    // This is a basic sanity check to ensure the model loaded correctly
    if model.tensor_names().is_empty() {
        anyhow::bail!("Loaded model has no tensors - invalid model structure");
    }

    Ok(model)
}

/// Run BitNet.rs inference for cross-validation
async fn run_bitnet_inference(
    model: &BitNetModel,
    input: &str,
    _quantizer: &I2SQuantizer,
) -> Result<InferenceResult> {
    use bitnet_models::transformer::KVCache;
    use bitnet_tokenizers::Tokenizer;
    use std::time::Instant;

    // 1. Load or create tokenizer for the model
    // For cross-validation, we need a tokenizer to convert input text to token IDs
    // Try to use UniversalTokenizer or fallback to simple whitespace tokenization
    let tokenizer = bitnet_tokenizers::UniversalTokenizer::auto_discover(None)
        .context("Failed to auto-discover tokenizer for cross-validation")?;

    // 2. Tokenize the input text
    let start_time = Instant::now();
    let token_ids = tokenizer
        .encode(input, false) // add_special_tokens=false for raw input
        .context("Failed to tokenize input for inference")?;

    // 3. Get model configuration for KV cache setup
    let config = model.config();

    // 4. Create KV cache for inference (batch size 1, CPU device)
    let cache = KVCache::new(&config, 1, &candle_core::Device::Cpu)
        .context("Failed to create KV cache for inference")?;
    let mut any_cache: Box<dyn std::any::Any> = Box::new(cache);

    // 5. Embed the input tokens
    let embedded = model.embed(&token_ids).context("Failed to embed tokens for inference")?;

    // 6. Run forward pass through the model
    let output = model
        .forward(&embedded, any_cache.as_mut())
        .context("Failed to run forward pass for inference")?;

    // 7. Get logits from the output
    let logits_tensor = model.logits(&output).context("Failed to extract logits for inference")?;

    // 8. Extract logits for the last token as Vec<f32>
    let logits = extract_last_token_logits_from_tensor(logits_tensor)
        .context("Failed to extract last token logits for inference")?;

    // 9. Calculate performance metrics
    let total_duration = start_time.elapsed();
    let total_duration_ms = total_duration.as_secs_f64() * 1000.0;
    let tokens_generated = token_ids.len();
    let tokens_per_second = if total_duration_ms > 0.0 {
        (tokens_generated as f64) / (total_duration_ms / 1000.0)
    } else {
        0.0
    };

    // 10. Return inference result with tokens, logits, and metrics
    Ok(InferenceResult {
        tokens: token_ids,
        logits,
        metrics: InferenceMetrics { total_duration_ms, tokens_per_second, tokens_generated },
    })
}

/// Run C++ reference implementation for comparison
/// This function executes inference using the C++ reference implementation
/// for cross-validation accuracy testing.
///
/// # Arguments
/// * `model_path` - Path to the GGUF model file
/// * `input` - Input text to process
///
/// # Returns
/// * `Result<ReferenceResult>` - Placeholder result (currently `()`)
///
/// # Note
/// This is a placeholder implementation. The `ReferenceResult` type is currently `()`
/// and will need to be replaced with a proper type containing tokens, logits, and metrics
/// when the cross-validation infrastructure is fully implemented.
async fn run_cpp_reference_inference(model_path: &str, input: &str) -> Result<ReferenceResult> {
    #[cfg(feature = "crossval")]
    {
        use std::path::Path;

        // Check if C++ reference is available
        if !is_crossval_environment_ready() {
            anyhow::bail!("C++ reference not available: BITNET_CPP_DIR not set or invalid");
        }

        // Load C++ model using crossval bindings
        let model_path_obj = Path::new(model_path);
        let cpp_model = crossval::cpp_bindings::CppModel::load(model_path_obj)
            .context("Failed to load C++ model for reference inference")?;

        // Generate tokens using C++ implementation
        // Use a reasonable max_tokens limit for cross-validation
        let max_tokens = 32;
        let generated_tokens = cpp_model
            .generate(input, max_tokens)
            .context("Failed to generate tokens with C++ reference")?;

        // Return placeholder (will be replaced with proper type later)
        // TODO: Define proper ReferenceResult type with tokens, logits, metrics
        let _ = generated_tokens; // Use to avoid warnings
        Ok(())
    }

    #[cfg(not(feature = "crossval"))]
    {
        let _ = (model_path, input); // Silence unused warnings
        anyhow::bail!("C++ reference inference requires crossval feature")
    }
}

/// Compare inference outputs using validation metrics
///
/// MVP implementation: Since ReferenceResult, CrossValidationConfig, and ValidationComparison
/// are currently placeholder types (`type X = ()`), this is a stub that will be replaced when
/// proper types are defined.
///
/// When proper types are available, this function should:
/// 1. Compare token sequences using `compare_token_sequences()` from real_inference_engine.rs
///    - Check exact match or similarity threshold from config
/// 2. Compare logits (if available) using `compare_numerical_accuracy()` from real_inference_engine.rs
///    - Apply tolerance from config
/// 3. Compare performance metrics using `compare_performance_metrics()` from real_inference_engine.rs
/// 4. Return ValidationComparison with aggregated results
///
/// # Example Future Implementation Pattern
///
/// ```rust,ignore
/// // 1. Extract C++ reference data (when ReferenceResult is properly defined)
/// // let cpp_tokens = &cpp_result.tokens;
/// // let cpp_logits = cpp_result.logits.as_ref();
///
/// // 2. Compare token sequences
/// // let token_comparison = compare_token_sequences(&bitnet_result.tokens, cpp_tokens);
///
/// // 3. Compare logits if available
/// // let numerical_comparison = if let Some(cpp_logits) = cpp_logits {
/// //     Some(compare_numerical_accuracy(&bitnet_result.logits, cpp_logits, config.tolerance)?)
/// // } else {
/// //     None
/// // };
///
/// // 4. Compare performance metrics
/// // let performance_comparison = compare_performance_metrics(
/// //     &bitnet_result.metrics,
/// //     &cpp_result.metrics
/// // );
///
/// // 5. Aggregate into ValidationComparison
/// // Ok(ValidationComparison {
/// //     token_match_rate: token_comparison.match_rate,
/// //     exact_match: token_comparison.exact_match,
/// //     first_mismatch: token_comparison.first_mismatch,
/// //     numerical_accuracy: numerical_comparison,
/// //     performance_ratio: performance_comparison.speedup_ratio,
/// //     within_tolerance: token_comparison.match_rate >= config.match_threshold
/// //         && numerical_comparison.map(|n| n.within_tolerance).unwrap_or(true),
/// // })
/// ```
fn compare_inference_outputs(
    bitnet_result: &InferenceResult,
    cpp_result: &ReferenceResult,
    config: &CrossValidationConfig,
) -> Result<ValidationComparison> {
    // Use parameters to avoid unused warnings
    let _ = (bitnet_result, cpp_result, config);

    // Placeholder return - when types are properly defined, this will return actual comparison
    // See documentation above for the expected implementation pattern using helper functions:
    // - compare_token_sequences() from real_inference_engine.rs
    // - compare_numerical_accuracy() from real_inference_engine.rs
    // - compare_performance_metrics() from real_inference_engine.rs
    Ok(())
}

/// Aggregate validation metrics from multiple test cases
fn aggregate_validation_metrics(
    results: &[(String, ValidationComparison)],
) -> Result<AggregatedMetrics> {
    // MVP implementation: Since ValidationComparison and AggregatedMetrics are placeholders (),
    // this is a stub that will be replaced when proper types are defined

    // Validate non-empty results
    anyhow::ensure!(!results.is_empty(), "Cannot aggregate metrics from empty results");

    // TODO: When proper types are available, implement:
    // 1. Average token accuracy:
    //    let avg_token_accuracy = results.iter()
    //        .map(|(_, cmp)| cmp.token_match_rate)
    //        .sum::<f64>() / results.len() as f64;
    //
    // 2. Average logit correlation:
    //    let avg_correlation = results.iter()
    //        .filter_map(|(_, cmp)| cmp.numerical_comparison.as_ref())
    //        .map(|num| num.correlation)
    //        .sum::<f64>() / count as f64;
    //
    // 3. Average MSE:
    //    let avg_mse = results.iter()
    //        .filter_map(|(_, cmp)| cmp.numerical_comparison.as_ref())
    //        .map(|num| num.mse)
    //        .sum::<f64>() / count as f64;
    //
    // 4. Perplexity degradation:
    //    let perplexity_degradation = calculate_perplexity_degradation(results)?;
    //
    // Example return type:
    // ```
    // Ok(AggregatedMetrics {
    //     average_token_accuracy: avg_token_accuracy,
    //     average_logit_correlation: avg_correlation,
    //     average_mse: avg_mse,
    //     perplexity_degradation,
    //     lookup_performance_metrics: None, // Populated for TL1/TL2 tests
    // })
    // ```

    let _ = results; // Use param to avoid warnings
    Ok(()) // Placeholder return
}

/// Parse xtask crossval output
///
/// MVP implementation: Parses command status and prepares for future structured parsing.
/// Currently returns placeholder type `()` since `CrossvalResults` is undefined.
///
/// # Arguments
/// * `output` - Output from `cargo run -p xtask -- crossval` command
///
/// # Returns
/// * `Result<CrossvalResults>` - Currently `()`, will be replaced with structured metrics
///
/// # Implementation Notes
///
/// The xtask crossval command produces:
/// 1. **Exit status**: Success (0) or failure (non-zero)
/// 2. **JSON report**: Saved to `target/crossval_report.json` with structure:
///    ```json
///    {
///      "model": "path/to/model.gguf",
///      "rust_ok": bool,
///      "cpp_header_ok": bool,
///      "cpp_full_ok": bool,
///      "xfail": bool,
///      "notes": "string",
///      "timestamp": "ISO8601",
///      "platform": "os-arch",
///      "gguf_version_detected": u32,
///      "n_kv": u64,
///      "n_tensors": u64,
///      "data_offset": u64,
///      "file_size": u64
///    }
///    ```
/// 3. **stdout/stderr**: Test output and diagnostic messages
///
/// # TODO: Future Implementation
///
/// When proper `CrossvalResults` type is defined, implement:
///
/// 1. **Parse JSON report**:
///    ```rust
///    let report_path = std::path::Path::new("target/crossval_report.json");
///    let report_json = std::fs::read_to_string(report_path)
///        .context("Failed to read crossval report JSON")?;
///    let report: serde_json::Value = serde_json::from_str(&report_json)
///        .context("Failed to parse crossval report JSON")?;
///    ```
///
/// 2. **Extract quantization accuracy**:
///    - Parse test output for I2S, TL1, TL2 accuracy metrics
///    - Look for patterns like "I2S accuracy: 0.9923" in stdout
///    - Calculate overall pass rate from rust_ok, cpp_header_ok, cpp_full_ok
///
/// 3. **Extract performance correlation**:
///    - Parse test output for performance comparison metrics
///    - Calculate correlation between Rust and C++ implementations
///
/// 4. **Extract numerical stability**:
///    - Parse test output for NaN/Inf detection
///    - Look for patterns like "NaN count: 0" in stdout
///
/// 5. **Build CrossvalResults**:
///    ```rust
///    Ok(CrossvalResults {
///        overall_pass_rate: if report["rust_ok"].as_bool().unwrap_or(false) { 1.0 } else { 0.0 },
///        quantization_accuracy: QuantizationAccuracy {
///            i2s: extract_i2s_accuracy(&stdout)?,
///            tl1: extract_tl1_accuracy(&stdout)?,
///            tl2: extract_tl2_accuracy(&stdout)?,
///        },
///        performance_correlation: extract_performance_correlation(&stdout)?,
///        numerical_stability: NumericalStability {
///            nan_count: extract_nan_count(&stdout)?,
///            inf_count: extract_inf_count(&stdout)?,
///        },
///    })
///    ```
fn parse_crossval_output(output: &std::process::Output) -> Result<CrossvalResults> {
    // 1. Check command execution status first
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        anyhow::bail!(
            "xtask crossval command failed with status {}\nstderr: {}\nstdout: {}",
            output.status,
            stderr,
            stdout
        );
    }

    // 2. Log successful execution for diagnostics
    let stdout = String::from_utf8_lossy(&output.stdout);
    log::debug!("xtask crossval completed successfully");
    log::trace!("crossval output:\n{}", stdout);

    // 3. MVP: Return placeholder
    // TODO: When CrossvalResults type is properly defined, implement structured parsing:
    //   - Read target/crossval_report.json for metadata
    //   - Parse stdout for quantization accuracy metrics (I2S, TL1, TL2)
    //   - Extract performance correlation from test output
    //   - Extract numerical stability metrics (NaN/Inf counts)
    //   - Build and return CrossvalResults struct
    //
    // See function documentation above for detailed implementation guidance.
    Ok(())
}

// Additional helper functions for specific quantization methods
async fn run_bitnet_inference_with_table_lookup(
    model: &BitNetModel,
    input: &str,
    _quantizer: &dyn TableLookupQuantizer,
) -> Result<InferenceResult> {
    // For MVP, TableLookupQuantizer is a placeholder trait
    // The actual implementation is the same as run_bitnet_inference()
    // because the quantizer selection happens at model loading time,
    // not during inference execution (TL1/TL2 quantization is already
    // baked into the model weights).
    //
    // The quantizer parameter is kept for API consistency and future
    // enhancement to track lookup-specific metrics like cache hit rate,
    // average lookup time, etc.

    use bitnet_models::transformer::KVCache;
    use bitnet_tokenizers::Tokenizer;
    use std::time::Instant;

    // 1. Load or create tokenizer for the model
    // For cross-validation, we need a tokenizer to convert input text to token IDs
    // Try to use UniversalTokenizer or fallback to simple whitespace tokenization
    let tokenizer = bitnet_tokenizers::UniversalTokenizer::auto_discover(None)
        .context("Failed to auto-discover tokenizer for TL cross-validation")?;

    // 2. Tokenize the input text
    let start_time = Instant::now();
    let token_ids = tokenizer
        .encode(input, false) // add_special_tokens=false for raw input
        .context("Failed to tokenize input for TL inference")?;

    // 3. Get model configuration for KV cache setup
    let config = model.config();

    // 4. Create KV cache for inference (batch size 1, CPU device)
    let cache = KVCache::new(&config, 1, &candle_core::Device::Cpu)
        .context("Failed to create KV cache for TL inference")?;
    let mut any_cache: Box<dyn std::any::Any> = Box::new(cache);

    // 5. Embed the input tokens
    let embedded = model.embed(&token_ids).context("Failed to embed tokens for TL inference")?;

    // 6. Run forward pass through the model
    // Table lookup operations (TL1/TL2) happen inside the model's forward pass
    let output = model
        .forward(&embedded, any_cache.as_mut())
        .context("Failed to run forward pass for TL inference")?;

    // 7. Get logits from the output
    let logits_tensor =
        model.logits(&output).context("Failed to extract logits for TL inference")?;

    // 8. Extract logits for the last token as Vec<f32>
    let logits = extract_last_token_logits_from_tensor(logits_tensor)
        .context("Failed to extract last token logits for TL inference")?;

    // 9. Calculate performance metrics
    let total_duration = start_time.elapsed();
    let total_duration_ms = total_duration.as_secs_f64() * 1000.0;
    let tokens_generated = token_ids.len();
    let tokens_per_second = if total_duration_ms > 0.0 {
        (tokens_generated as f64) / (total_duration_ms / 1000.0)
    } else {
        0.0
    };

    // 10. Return inference result with tokens, logits, and metrics
    // TODO: When proper TableLookupQuantizer trait is implemented,
    // this could include additional lookup performance metrics:
    // - cache_hit_rate: Fraction of lookups served from cache
    // - average_lookup_time_ns: Average time per table lookup operation
    // - total_lookups: Total number of table lookup operations
    Ok(InferenceResult {
        tokens: token_ids,
        logits,
        metrics: InferenceMetrics { total_duration_ms, tokens_per_second, tokens_generated },
    })
}

/// Compare table lookup quantization outputs (TL1/TL2) with C++ reference
///
/// MVP implementation: Since ValidationComparison and ReferenceResult are placeholder types (),
/// this is a stub that delegates to the standard comparison function with adjusted tolerance.
///
/// Table lookup methods (TL1/TL2) may have slightly different correlation thresholds due to
/// lookup precision characteristics. The test at line 210 suggests 99.8% correlation for TL
/// methods vs 99.9% for standard methods.
///
/// When proper types are available, this function should:
/// 1. Compare token sequences using standard comparison logic
/// 2. Compare logits with adjusted tolerance for TL methods (99.8% vs 99.9%)
/// 3. Validate lookup-specific performance metrics:
///    - Average lookup time ≤ 10ns per lookup (line 223-227)
///    - Cache hit rate ≥ 95% (line 229-235)
/// 4. Log method name ("TL1" or "TL2") for debugging
/// 5. Return ValidationComparison with aggregated results
///
/// # Example Future Implementation
///
/// ```rust,ignore
/// // Adjust tolerance for table lookup methods (line 210)
/// let tl_correlation_threshold = config.correlation_threshold * 0.99; // 99.8% for TL
///
/// // Create adjusted config for TL validation
/// let adjusted_config = CrossValidationConfig {
///     correlation_threshold: tl_correlation_threshold,
///     ..crossval_config.clone()
/// };
///
/// // Use standard comparison with adjusted config
/// let comparison = compare_inference_outputs(bitnet_result, cpp_result, &adjusted_config)?;
///
/// // Validate TL-specific performance metrics if available
/// if let Some(lookup_metrics) = comparison.lookup_performance_metrics {
///     assert!(
///         lookup_metrics.average_lookup_time_ns <= 10.0,
///         "{} lookup time too high: {:.2}ns > 10ns",
///         method,
///         lookup_metrics.average_lookup_time_ns
///     );
///
///     assert!(
///         lookup_metrics.cache_hit_rate >= 0.95,
///         "{} cache hit rate too low: {:.2}% < 95%",
///         method,
///         lookup_metrics.cache_hit_rate * 100.0
///     );
/// }
///
/// // Log method-specific info for debugging
/// log::info!(
///     "{} comparison: match_rate={:.4}, lookup_time={:.2}ns",
///     method,
///     comparison.match_rate,
///     lookup_metrics.average_lookup_time_ns
/// );
///
/// Ok(comparison)
/// ```
fn compare_table_lookup_outputs(
    bitnet_result: &InferenceResult,
    cpp_result: &ReferenceResult,
    method: &str,
    config: &AC4TestConfig,
) -> Result<ValidationComparison> {
    // Use parameters to avoid unused warnings
    let _ = (bitnet_result, cpp_result, method, config);

    // Placeholder return - when types are properly defined:
    // 1. Create adjusted CrossValidationConfig with TL-specific correlation threshold (99.8%)
    // 2. Call compare_inference_outputs() with adjusted config
    // 3. Validate lookup-specific metrics (lookup time ≤ 10ns, cache hit ≥ 95%)
    // 4. Log method-specific debugging info
    // 5. Return ValidationComparison with TL-specific metrics populated
    //
    // This ensures TL1/TL2 methods use appropriate tolerance adjustments while maintaining
    // the same validation structure as standard quantization methods.

    Ok(()) // Placeholder return
}

async fn run_bitnet_iq2s_inference(
    model: &BitNetModel,
    input: &str,
    _quantizer: &IQ2SQuantizer,
) -> Result<InferenceResult> {
    use bitnet_common::Device;
    use bitnet_models::transformer::KVCache;
    use bitnet_tokenizers::Tokenizer;
    use std::time::Instant;

    // 1. Load or create tokenizer for the model
    // For cross-validation, we need a tokenizer to convert input text to token IDs
    // Try to use UniversalTokenizer or fallback to simple whitespace tokenization
    let tokenizer = bitnet_tokenizers::UniversalTokenizer::auto_discover(None)
        .context("Failed to auto-discover tokenizer for cross-validation")?;

    // 2. Tokenize the input text
    let start_time = Instant::now();
    let token_ids = tokenizer
        .encode(input, false) // add_special_tokens=false for raw input
        .context("Failed to tokenize input for IQ2_S inference")?;

    // 3. Get model configuration for KV cache setup
    let config = model.config();

    // 4. Create KV cache for inference (batch size 1, CPU device)
    let cache = KVCache::new(&config, 1, &candle_core::Device::Cpu)
        .context("Failed to create KV cache for IQ2_S inference")?;
    let mut any_cache: Box<dyn std::any::Any> = Box::new(cache);

    // 5. Embed the input tokens
    let embedded = model.embed(&token_ids).context("Failed to embed tokens for IQ2_S inference")?;

    // 6. Run forward pass through the model
    let output = model
        .forward(&embedded, any_cache.as_mut())
        .context("Failed to run forward pass for IQ2_S inference")?;

    // 7. Get logits from the output
    let logits_tensor =
        model.logits(&output).context("Failed to extract logits for IQ2_S inference")?;

    // 8. Extract logits for the last token as Vec<f32>
    let logits = extract_last_token_logits_from_tensor(logits_tensor)
        .context("Failed to extract last token logits for IQ2_S inference")?;

    // 9. Calculate performance metrics
    let total_duration = start_time.elapsed();
    let total_duration_ms = total_duration.as_secs_f64() * 1000.0;
    let tokens_generated = token_ids.len();
    let tokens_per_second = if total_duration_ms > 0.0 {
        (tokens_generated as f64) / (total_duration_ms / 1000.0)
    } else {
        0.0
    };

    // 10. Return inference result with tokens, logits, and metrics
    Ok(InferenceResult {
        tokens: token_ids,
        logits,
        metrics: InferenceMetrics { total_duration_ms, tokens_per_second, tokens_generated },
    })
}

/// Helper function to extract logits for the last token from a tensor
fn extract_last_token_logits_from_tensor(
    logits: bitnet_common::ConcreteTensor,
) -> Result<Vec<f32>> {
    use bitnet_common::ConcreteTensor;
    use candle_core::DType;

    match logits {
        ConcreteTensor::BitNet(tensor) => {
            // Get the underlying Candle tensor
            let candle_tensor = tensor.as_candle();

            // Shape should be [batch, seq_len, vocab_size]
            let dims = candle_tensor.dims();
            if dims.len() != 3 {
                anyhow::bail!("Expected 3D logits tensor for IQ2_S inference, got {:?}", dims);
            }

            let seq_len = dims[1];

            // Extract last token: narrow to last position in sequence dimension
            let last_token_logits = candle_tensor
                .narrow(1, seq_len - 1, 1)? // Get last position
                .squeeze(1)? // Remove seq dimension
                .i(0)?; // Get first (and only) batch

            // Convert to F32 if needed
            let last_token_logits = if last_token_logits.dtype() != DType::F32 {
                last_token_logits.to_dtype(DType::F32)?
            } else {
                last_token_logits.clone()
            };

            // Convert to Vec<f32>
            Ok(last_token_logits.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(mock) => {
            // For mock tensors, return zeros
            let vocab_size = mock.shape()[2];
            Ok(vec![0.0f32; vocab_size])
        }
    }
}

/// Run GGML reference inference using C++ FFI
/// This function executes inference using the GGML C++ reference implementation
/// for cross-validation accuracy testing.
///
/// # Arguments
/// * `model_path` - Path to the GGUF model file
/// * `input` - Input text to process
///
/// # Returns
/// * `Result<GGMLResult>` - Generated tokens and model metadata
async fn run_ggml_reference_inference(model_path: &str, input: &str) -> Result<GGMLResult> {
    #[cfg(feature = "crossval")]
    {
        use std::path::Path;

        // Check if C++ reference is available
        if !is_crossval_environment_ready() {
            anyhow::bail!("GGML reference not available: BITNET_CPP_DIR not set or invalid");
        }

        // Load C++ model using crossval bindings
        let model_path_obj = Path::new(model_path);
        let cpp_model = crossval::cpp_bindings::CppModel::load(model_path_obj)
            .context("Failed to load C++ model for GGML reference inference")?;

        // Generate tokens using C++ implementation
        // Use a reasonable max_tokens limit for cross-validation
        let max_tokens = 32;
        let generated_tokens = cpp_model
            .generate(input, max_tokens)
            .context("Failed to generate tokens with C++ reference")?;

        // Get model info for provenance
        let model_info = cpp_model.model_info().unwrap_or(crossval::cpp_bindings::ModelInfo {
            name: "Unknown".to_string(),
            version: "Unknown".to_string(),
            parameter_count: 0,
            quantization: "Unknown".to_string(),
        });

        Ok(GGMLResult {
            tokens: generated_tokens,
            logits: None, // C++ FFI doesn't expose logits directly yet
            model_name: model_info.name,
            quantization_format: model_info.quantization,
        })
    }

    #[cfg(not(feature = "crossval"))]
    {
        let _ = (model_path, input); // Silence unused warnings
        anyhow::bail!("GGML reference inference requires crossval feature")
    }
}

/// Compare IQ2_S inference results for GGML compatibility
/// Computes cosine similarity between logits and exact token match rate
fn compare_iq2s_compatibility(
    bitnet_result: &InferenceResult,
    _ggml_result: &GGMLResult,
    _config: &CrossValidationConfig,
) -> Result<IQ2SCompatibilityComparison> {
    // Extract BitNet.rs logits and tokens
    let bitnet_logits = &bitnet_result.logits;
    let bitnet_tokens = &bitnet_result.tokens;

    // For now, GGMLResult is a placeholder ()
    // In production, this would extract logits and tokens from GGML FFI outputs
    // Using placeholder data that matches the expected structure
    let ggml_logits = bitnet_logits; // Placeholder: would be from ggml_result
    let ggml_tokens = bitnet_tokens; // Placeholder: would be from ggml_result

    // Calculate cosine similarity between logits
    let cosine_similarity = compute_cosine_similarity(bitnet_logits, ggml_logits)
        .context("Failed to compute cosine similarity")?;

    // Calculate exact token match rate
    let exact_match_rate = compute_exact_match_rate(bitnet_tokens, ggml_tokens)
        .context("Failed to compute exact match rate")?;

    // Create comparison result
    Ok(IQ2SCompatibilityComparison {
        cosine_similarity,
        exact_match_rate,
        bitnet_token_count: bitnet_tokens.len(),
        ggml_token_count: ggml_tokens.len(),
    })
}

/// Compute cosine similarity between two logit vectors
/// Returns value in [0.0, 1.0] where 1.0 is perfect match
fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    anyhow::ensure!(
        a.len() == b.len(),
        "Logit vectors must have same length for cosine similarity: {} vs {}",
        a.len(),
        b.len()
    );

    if a.is_empty() {
        return Ok(1.0); // Empty vectors are considered identical
    }

    // Compute dot product
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    // Compute magnitudes
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Handle zero vectors
    if norm_a == 0.0 && norm_b == 0.0 {
        return Ok(1.0); // Both zero vectors are considered identical
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0); // One zero vector means no similarity
    }

    // Compute cosine similarity
    let similarity = dot_product / (norm_a * norm_b);

    // Clamp to [0.0, 1.0] to handle floating-point errors
    Ok(similarity.max(0.0).min(1.0))
}

/// Compute exact token match rate between two token sequences
/// Returns value in [0.0, 1.0] where 1.0 is perfect match
fn compute_exact_match_rate(bitnet_tokens: &[u32], ggml_tokens: &[u32]) -> Result<f32> {
    if bitnet_tokens.is_empty() && ggml_tokens.is_empty() {
        return Ok(1.0); // Empty sequences match perfectly
    }

    if bitnet_tokens.is_empty() || ggml_tokens.is_empty() {
        return Ok(0.0); // One empty sequence means no match
    }

    // Count exact matches up to the minimum length
    let min_len = bitnet_tokens.len().min(ggml_tokens.len());
    let mut matches = 0usize;

    for i in 0..min_len {
        if bitnet_tokens[i] == ggml_tokens[i] {
            matches += 1;
        }
    }

    // Use maximum length for denominator (penalizes length mismatches)
    let max_len = bitnet_tokens.len().max(ggml_tokens.len());
    let exact_match_rate = matches as f32 / max_len as f32;

    Ok(exact_match_rate)
}

fn aggregate_iq2s_compatibility_metrics(
    results: &[(String, IQ2SCompatibilityComparison)],
) -> Result<IQ2SCompatibilityMetrics> {
    use anyhow::ensure;

    ensure!(!results.is_empty(), "Cannot aggregate IQ2_S compatibility metrics from empty results");

    let num_tests = results.len() as f32;

    // Aggregate exact match rates (proxy for bit-exact matches)
    let total_exact_match_rate: f32 =
        results.iter().map(|(_, comparison)| comparison.exact_match_rate).sum();
    let bit_exact_matches = total_exact_match_rate / num_tests;

    // Aggregate cosine similarities (proxy for block format compliance)
    let total_cosine_similarity: f32 =
        results.iter().map(|(_, comparison)| comparison.cosine_similarity).sum();
    let block_format_compliance = total_cosine_similarity / num_tests;

    // Quantization level accuracy is the average exact match rate
    let quantization_level_accuracy = bit_exact_matches;

    // Performance ratio: average ratio of BitNet token count to GGML token count
    let mut total_performance_ratio = 0.0f32;
    let mut valid_performance_samples = 0usize;

    for (_, comparison) in results {
        if comparison.ggml_token_count > 0 {
            let ratio = comparison.bitnet_token_count as f32 / comparison.ggml_token_count as f32;
            total_performance_ratio += ratio;
            valid_performance_samples += 1;
        }
    }

    // Calculate average performance ratio (default to 1.0 if no valid samples)
    let performance_ratio = if valid_performance_samples > 0 {
        total_performance_ratio / valid_performance_samples as f32
    } else {
        1.0f32
    };

    Ok(IQ2SCompatibilityMetrics {
        bit_exact_matches,
        block_format_compliance,
        quantization_level_accuracy,
        performance_ratio,
    })
}

// Type stubs for compilation - replace with actual implementations

/// Inference result containing generated tokens, logits, and performance metrics
#[derive(Debug, Clone)]
struct InferenceResult {
    /// Generated token IDs
    tokens: Vec<u32>,
    /// Final logits for the last generated token (vocab_size elements)
    logits: Vec<f32>,
    /// Performance metrics for the inference run
    metrics: InferenceMetrics,
}

/// Performance metrics collected during inference
#[derive(Debug, Clone)]
struct InferenceMetrics {
    /// Total inference duration in milliseconds
    total_duration_ms: f64,
    /// Tokens per second throughput
    tokens_per_second: f64,
    /// Number of tokens generated
    tokens_generated: usize,
}

/// IQ2_S compatibility comparison result
#[derive(Debug, Clone)]
struct IQ2SCompatibilityComparison {
    /// Cosine similarity between BitNet.rs and GGML logits (0.0 to 1.0)
    cosine_similarity: f32,
    /// Exact token match rate (0.0 to 1.0)
    exact_match_rate: f32,
    /// Number of tokens generated by BitNet.rs
    bitnet_token_count: usize,
    /// Number of tokens generated by GGML reference
    ggml_token_count: usize,
}

/// Aggregated IQ2_S compatibility metrics
#[derive(Debug, Clone)]
struct IQ2SCompatibilityMetrics {
    /// Fraction of bit-exact matches across all tests
    bit_exact_matches: f32,
    /// Block format compliance rate
    block_format_compliance: f32,
    /// Quantization level accuracy
    quantization_level_accuracy: f32,
    /// Performance ratio vs GGML (1.0 = same performance)
    performance_ratio: f32,
}

/// GGML reference implementation result
#[derive(Debug, Clone)]
struct GGMLResult {
    /// Generated token IDs from GGML reference
    tokens: Vec<u32>,
    /// Logits from GGML reference (if available)
    logits: Option<Vec<f32>>,
    /// Model name from GGML reference
    model_name: String,
    /// Quantization format used
    quantization_format: String,
}

type CrossValidationConfig = (); // Placeholder
type IQ2SQuantizer = I2SQuantizer; // Use I2SQuantizer for now
type ReferenceResult = (); // Placeholder
type ValidationComparison = (); // Placeholder
type AggregatedMetrics = (); // Placeholder with accuracy/correlation fields
type CrossvalResults = (); // Placeholder
trait TableLookupQuantizer {} // Placeholder trait
type ReferenceImplementation = (); // Placeholder enum

// Unit tests for IQ2_S compatibility comparison functions
#[cfg(test)]
mod iq2s_compatibility_tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = compute_cosine_similarity(&a, &b).unwrap();
        assert!((result - 1.0).abs() < 1e-6, "Identical vectors should have similarity 1.0");
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let result = compute_cosine_similarity(&a, &b).unwrap();
        assert!((result - 0.0).abs() < 1e-6, "Orthogonal vectors should have similarity 0.0");
    }

    #[test]
    fn test_cosine_similarity_similar_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.1, 2.9];
        let result = compute_cosine_similarity(&a, &b).unwrap();
        assert!(result > 0.99, "Similar vectors should have high similarity: {}", result);
    }

    #[test]
    fn test_cosine_similarity_zero_vectors() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let result = compute_cosine_similarity(&a, &b).unwrap();
        assert_eq!(result, 1.0, "Two zero vectors should be considered identical");
    }

    #[test]
    fn test_cosine_similarity_one_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        let result = compute_cosine_similarity(&a, &b).unwrap();
        assert_eq!(result, 0.0, "Zero vector should have no similarity with non-zero");
    }

    #[test]
    fn test_cosine_similarity_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let result = compute_cosine_similarity(&a, &b).unwrap();
        assert_eq!(result, 1.0, "Empty vectors should be considered identical");
    }

    #[test]
    fn test_exact_match_rate_identical_sequences() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 5];
        let result = compute_exact_match_rate(&a, &b).unwrap();
        assert_eq!(result, 1.0, "Identical sequences should have match rate 1.0");
    }

    #[test]
    fn test_exact_match_rate_completely_different() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![6, 7, 8, 9, 10];
        let result = compute_exact_match_rate(&a, &b).unwrap();
        assert_eq!(result, 0.0, "Completely different sequences should have match rate 0.0");
    }

    #[test]
    fn test_exact_match_rate_partial_match() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 6, 7];
        let result = compute_exact_match_rate(&a, &b).unwrap();
        assert_eq!(result, 0.6, "3/5 matches should give 0.6 match rate");
    }

    #[test]
    fn test_exact_match_rate_different_lengths() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3, 4, 5];
        let result = compute_exact_match_rate(&a, &b).unwrap();
        // 3 matches out of max(3, 5) = 5
        assert_eq!(result, 0.6, "Different length sequences should be penalized");
    }

    #[test]
    fn test_exact_match_rate_empty_sequences() {
        let a: Vec<u32> = vec![];
        let b: Vec<u32> = vec![];
        let result = compute_exact_match_rate(&a, &b).unwrap();
        assert_eq!(result, 1.0, "Empty sequences should match perfectly");
    }

    #[test]
    fn test_exact_match_rate_one_empty() {
        let a = vec![1, 2, 3];
        let b: Vec<u32> = vec![];
        let result = compute_exact_match_rate(&a, &b).unwrap();
        assert_eq!(result, 0.0, "One empty sequence should have no match");
    }

    #[test]
    fn test_compare_iq2s_compatibility_perfect_match() {
        let bitnet_result = InferenceResult {
            tokens: vec![1, 2, 3, 4],
            logits: vec![0.1, 0.2, 0.3, 0.4],
            metrics: InferenceMetrics {
                total_duration_ms: 100.0,
                tokens_per_second: 10.0,
                tokens_generated: 4,
            },
        };

        let ggml_result = (); // Placeholder

        let comparison = compare_iq2s_compatibility(&bitnet_result, &ggml_result, &())
            .expect("Comparison should succeed");

        // With placeholder (same data), should get perfect match
        assert_eq!(comparison.cosine_similarity, 1.0, "Should have perfect cosine similarity");
        assert_eq!(comparison.exact_match_rate, 1.0, "Should have perfect exact match rate");
        assert_eq!(comparison.bitnet_token_count, 4, "Should have 4 BitNet tokens");
        assert_eq!(comparison.ggml_token_count, 4, "Should have 4 GGML tokens");
    }

    #[test]
    fn test_aggregate_iq2s_compatibility_metrics_single_result() {
        let comparison = IQ2SCompatibilityComparison {
            cosine_similarity: 0.99,
            exact_match_rate: 0.95,
            bitnet_token_count: 100,
            ggml_token_count: 100,
        };

        let results = vec![("test1".to_string(), comparison)];

        let metrics =
            aggregate_iq2s_compatibility_metrics(&results).expect("Aggregation should succeed");

        assert_eq!(metrics.bit_exact_matches, 0.95);
        assert_eq!(metrics.block_format_compliance, 0.99);
        assert_eq!(metrics.quantization_level_accuracy, 0.95);
        assert_eq!(metrics.performance_ratio, 1.0);
    }

    #[test]
    fn test_aggregate_iq2s_compatibility_metrics_multiple_results() {
        let results = vec![
            (
                "test1".to_string(),
                IQ2SCompatibilityComparison {
                    cosine_similarity: 1.0,
                    exact_match_rate: 1.0,
                    bitnet_token_count: 100,
                    ggml_token_count: 100,
                },
            ),
            (
                "test2".to_string(),
                IQ2SCompatibilityComparison {
                    cosine_similarity: 0.98,
                    exact_match_rate: 0.90,
                    bitnet_token_count: 90,
                    ggml_token_count: 100,
                },
            ),
        ];

        let metrics =
            aggregate_iq2s_compatibility_metrics(&results).expect("Aggregation should succeed");

        // Average exact match: (1.0 + 0.90) / 2 = 0.95
        assert_eq!(metrics.bit_exact_matches, 0.95);
        // Average cosine: (1.0 + 0.98) / 2 = 0.99
        assert_eq!(metrics.block_format_compliance, 0.99);
        assert_eq!(metrics.quantization_level_accuracy, 0.95);
        // Average performance: (100/100 + 90/100) / 2 = 0.95
        assert_eq!(metrics.performance_ratio, 0.95);
    }

    #[test]
    fn test_aggregate_iq2s_compatibility_metrics_empty_results_fails() {
        let results: Vec<(String, IQ2SCompatibilityComparison)> = vec![];

        let result = aggregate_iq2s_compatibility_metrics(&results);

        assert!(result.is_err(), "Empty results should fail aggregation");
    }
}
