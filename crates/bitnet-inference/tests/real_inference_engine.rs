//! Real Inference Engine Tests for bitnet-inference
//!
//! Tests feature spec: real-bitnet-model-integration-architecture.md#inference-stage
//! Tests API contract: real-model-api-contracts.md#production-inference-engine-contract
//!
//! This module contains comprehensive test scaffolding for real BitNet model inference,
//! performance metrics collection, and cross-validation framework integration.

#![cfg(any())] // Disabled: ProductionInferenceEngine not yet implemented

#[cfg(all(feature = "inference", feature = "crossval"))]
use std::env;
#[allow(unused_imports)]
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
#[allow(unused_imports)]
use std::time::Instant;

// Note: All tests in this file are disabled until production API types are available
// NOTE: Requires ProductionInferenceEngine, InferenceMetrics, and related types
#[cfg(all(feature = "inference", any()))] // Disabled: ProductionInferenceEngine not yet implemented
#[allow(unused_imports)]
use bitnet_inference::GenerationConfig;

#[cfg(all(feature = "inference", any()))]
// Disabled: ProductionInferenceEngine not yet implemented
use bitnet_models::BitNetModel;

#[cfg(all(feature = "inference", any()))]
// Disabled: ProductionInferenceEngine not yet implemented
use bitnet_tokenizers::UniversalTokenizer;

/// Test configuration for inference engine tests
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct InferenceTestConfig {
    model_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    device_preference: String,
    max_tokens: u32,
    timeout: Duration,
    enable_metrics: bool,
}

// Disable all tests until types are available - tests use non-existent API
#[cfg(all(feature = "inference", any()))] // any() = false, disables tests

impl InferenceTestConfig {
    #[allow(dead_code)]
    fn from_env() -> Self {
        Self {
            model_path: env::var("BITNET_GGUF").ok().map(PathBuf::from),
            tokenizer_path: env::var("BITNET_TOKENIZER").ok().map(PathBuf::from),
            device_preference: env::var("BITNET_DEVICE").unwrap_or_else(|_| "auto".to_string()),
            max_tokens: env::var("BITNET_MAX_TOKENS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(32),
            timeout: Duration::from_secs(120),
            enable_metrics: !env::var("BITNET_FAST_TESTS").unwrap_or_default().eq("1"),
        }
    }

    #[allow(dead_code)]
    fn skip_if_no_model(&self) {
        if self.model_path.is_none() || !self.model_path.as_ref().unwrap().exists() {
            eprintln!("Skipping real inference test - set BITNET_GGUF environment variable");
            std::process::exit(0);
        }
    }
}

// ==============================================================================
// AC3: End-to-End Inference Pipeline Tests
// Tests feature spec: real-bitnet-model-integration-architecture.md#ac3
// ==============================================================================

/// Test inference engine real model integration
/// Validates complete inference pipeline with real models and performance metrics
#[test]
#[cfg(all(feature = "inference", any()))] // Disabled: ProductionInferenceEngine not yet implemented
fn test_inference_engine_real_model_integration() {
    // AC:3
    let config = InferenceTestConfig::from_env();
    config.skip_if_no_model();

    let model_path = config.model_path.unwrap();
    let test_prompt = "The capital of France is";

    // TODO: This test will initially fail - drives InferenceEngine implementation
    let engine_config = EngineConfig {
        device_preference: config.device_preference.clone(),
        enable_performance_monitoring: true,
        prefill_optimization: true,
        batch_processing: false,
        memory_optimization: true,
    };

    // Load model and create engine
    let model = load_real_model(&model_path).expect("Model should load");
    let tokenizer = create_or_load_tokenizer(&model, config.tokenizer_path.as_ref())
        .expect("Tokenizer should be available");

    let mut engine = ProductionInferenceEngine::new(model, tokenizer, engine_config)
        .expect("Engine should initialize");

    // Test end-to-end inference
    let start_time = Instant::now();
    let inference_result = futures::executor::block_on(engine.infer_with_metrics(test_prompt))
        .expect("Inference should succeed");

    let total_duration = start_time.elapsed();

    // Validate inference result
    assert!(!inference_result.text.is_empty(), "Should generate non-empty text");
    assert!(!inference_result.tokens.is_empty(), "Should generate tokens");
    assert!(total_duration < config.timeout, "Should complete within timeout");

    // Validate performance metrics
    let metrics = &inference_result.metrics;
    assert!(metrics.total_duration > Duration::ZERO, "Should record total duration");
    assert!(metrics.prefill_duration > Duration::ZERO, "Should record prefill duration");
    assert!(metrics.decode_duration > Duration::ZERO, "Should record decode duration");
    assert!(metrics.tokens_per_second > 0.0, "Should calculate throughput");

    // Validate device information
    assert!(!inference_result.device_info.device_type.is_empty(), "Should report device type");

    println!("Generated text: {}", inference_result.text);
    println!("Tokens per second: {:.2}", metrics.tokens_per_second);
    println!("✅ Inference engine real model integration test scaffolding created");
}

/// Test performance metrics collection framework
/// Validates comprehensive performance monitoring and metrics collection
#[test]
#[cfg(all(feature = "inference", any()))] // Disabled: ProductionInferenceEngine not yet implemented
fn test_performance_metrics_collection_framework() {
    // AC:3
    let config = InferenceTestConfig::from_env();

    if !config.enable_metrics {
        println!("Skipping performance metrics test - BITNET_FAST_TESTS=1");
        return;
    }

    config.skip_if_no_model();
    let model_path = config.model_path.unwrap();

    // TODO: This test will initially fail - drives PerformanceMonitor implementation
    let model = load_real_model(&model_path).expect("Model should load");
    let tokenizer = create_or_load_tokenizer(&model, config.tokenizer_path.as_ref())
        .expect("Tokenizer should be available");

    let engine_config = EngineConfig {
        enable_performance_monitoring: true,
        detailed_metrics: true,
        memory_tracking: true,
        device_profiling: true,
        ..Default::default()
    };

    let mut engine = ProductionInferenceEngine::new(model, tokenizer, engine_config)
        .expect("Engine should initialize");

    // Test multiple inference runs for statistical metrics
    let test_prompts = vec![
        "Hello world",
        "The quick brown fox",
        "In the beginning",
        "Machine learning is",
        "Neural networks can",
    ];

    let mut all_metrics = Vec::new();

    for prompt in test_prompts {
        let result = futures::executor::block_on(engine.infer_with_metrics(prompt))
            .expect("Inference should succeed");

        all_metrics.push(result.metrics);
    }

    // Validate comprehensive metrics collection
    assert_eq!(all_metrics.len(), 5, "Should collect metrics for all runs");

    // Test timing metrics validation
    for metrics in &all_metrics {
        validate_timing_metrics(&metrics).expect("Timing metrics should be valid");
        validate_throughput_metrics(&metrics).expect("Throughput metrics should be valid");
        validate_memory_metrics(&metrics).expect("Memory metrics should be valid");
    }

    // Test performance statistics
    let performance_stats = calculate_performance_statistics(&all_metrics);
    assert!(performance_stats.mean_throughput > 0.0, "Should calculate mean throughput");
    assert!(performance_stats.throughput_std_dev >= 0.0, "Should calculate throughput std dev");
    assert!(performance_stats.mean_latency > Duration::ZERO, "Should calculate mean latency");

    // Test device performance metrics
    #[cfg(feature = "gpu")]
    {
        let gpu_metrics = engine.get_device_performance_metrics();
        if let Some(gpu_stats) = gpu_metrics.gpu_metrics {
            assert!(gpu_stats.utilization_percent >= 0.0, "GPU utilization should be valid");
            assert!(gpu_stats.memory_used_mb >= 0, "GPU memory usage should be valid");
        }
    }

    // Test performance regression detection
    let baseline_throughput = 10.0; // tokens/sec baseline
    let current_throughput = performance_stats.mean_throughput;
    let performance_ratio = current_throughput / baseline_throughput;

    if performance_ratio < 0.8 {
        println!("Warning: Performance regression detected - {:.2}x baseline", performance_ratio);
    }

    println!("Performance statistics: {:#?}", performance_stats);
    println!("✅ Performance metrics collection framework test scaffolding created");
}

/// Test explicit prefill operation with cache warming and timing
/// Validates dedicated prefill functionality for cache optimization
#[test]
#[cfg(all(feature = "inference", any()))] // Disabled: ProductionInferenceEngine not yet implemented
fn test_explicit_prefill_operation_with_timing() {
    // AC:3
    let config = InferenceTestConfig::from_env();
    config.skip_if_no_model();

    let model_path = config.model_path.unwrap();
    let test_prompt =
        "This is a longer prompt that should benefit from explicit prefill optimization";

    // TODO: This test will initially fail - drives explicit prefill implementation
    let model = load_real_model(&model_path).expect("Model should load");
    let tokenizer = create_or_load_tokenizer(&model, config.tokenizer_path.as_ref())
        .expect("Tokenizer should be available");

    let mut engine =
        ProductionInferenceEngine::new(model, tokenizer, EngineConfig::with_prefill_optimization())
            .expect("Engine should initialize");

    // Tokenize the prompt
    let input_tokens = engine.tokenize(test_prompt).expect("Tokenization should succeed");
    assert!(!input_tokens.is_empty(), "Should produce tokens");

    // Test explicit prefill operation
    let prefill_start = Instant::now();
    let prefill_result =
        futures::executor::block_on(engine.prefill(&input_tokens)).expect("Prefill should succeed");
    let prefill_duration = prefill_start.elapsed();

    // Validate prefill result
    assert!(prefill_result.cache_entries > 0, "Should populate cache entries");
    assert!(prefill_result.processed_tokens == input_tokens.len(), "Should process all tokens");
    assert!(prefill_duration < Duration::from_secs(10), "Prefill should be reasonably fast");

    // Test generation after prefill
    let generation_config = GenerationConfig {
        max_new_tokens: 16,
        temperature: 0.7,
        use_cache: true,
        deterministic: false,
    };

    let generation_start = Instant::now();
    let generation_result =
        futures::executor::block_on(engine.generate_tokens(&input_tokens, generation_config))
            .expect("Generation should succeed");
    let generation_duration = generation_start.elapsed();

    // Validate generation benefited from prefill
    assert!(!generation_result.tokens.is_empty(), "Should generate tokens");
    assert!(generation_result.cache_hit_rate > 0.5, "Should have good cache hit rate");

    // Compare with cold generation (no prefill)
    let cold_engine = create_cold_engine(&model_path).expect("Cold engine should initialize");
    let cold_start = Instant::now();
    let _cold_result = futures::executor::block_on(cold_engine.infer_with_metrics(test_prompt))
        .expect("Cold inference should succeed");
    let cold_duration = cold_start.elapsed();

    // Prefilled generation should be faster
    let total_prefill_time = prefill_duration + generation_duration;
    if total_prefill_time < cold_duration {
        println!(
            "Prefill optimization effective: {:.2}x speedup",
            cold_duration.as_secs_f64() / total_prefill_time.as_secs_f64()
        );
    }

    println!("✅ Explicit prefill operation test scaffolding created");
}

/// Test batch inference with performance optimization
/// Validates efficient batch processing with real models
#[test]
#[cfg(all(feature = "inference", any()))] // Disabled: ProductionInferenceEngine not yet implemented
fn test_batch_inference_performance_optimization() {
    // AC:3
    let config = InferenceTestConfig::from_env();
    config.skip_if_no_model();

    let model_path = config.model_path.unwrap();
    let batch_prompts = vec![
        "The first test prompt",
        "The second test prompt",
        "The third test prompt",
        "The fourth test prompt",
    ];

    // TODO: This test will initially fail - drives batch inference implementation
    let model = load_real_model(&model_path).expect("Model should load");
    let tokenizer = create_or_load_tokenizer(&model, config.tokenizer_path.as_ref())
        .expect("Tokenizer should be available");

    let engine_config = EngineConfig {
        batch_processing: true,
        batch_size: batch_prompts.len(),
        memory_optimization: true,
        ..Default::default()
    };

    let mut engine = ProductionInferenceEngine::new(model, tokenizer, engine_config)
        .expect("Engine should initialize");

    // Test batch inference
    let batch_start = Instant::now();
    let batch_results = futures::executor::block_on(engine.infer_batch(&batch_prompts))
        .expect("Batch inference should succeed");
    let batch_duration = batch_start.elapsed();

    // Validate batch results
    assert_eq!(batch_results.len(), batch_prompts.len(), "Should process all prompts");

    for (i, result) in batch_results.iter().enumerate() {
        assert!(!result.text.is_empty(), "Result {} should generate text", i);
        assert!(!result.tokens.is_empty(), "Result {} should generate tokens", i);
        assert!(result.metrics.total_duration > Duration::ZERO, "Result {} should have timing", i);
    }

    // Test batch efficiency vs individual inference
    let individual_start = Instant::now();
    let mut individual_results = Vec::new();

    for prompt in &batch_prompts {
        let result = futures::executor::block_on(engine.infer_with_metrics(prompt))
            .expect("Individual inference should succeed");
        individual_results.push(result);
    }

    let individual_duration = individual_start.elapsed();

    // Calculate efficiency metrics
    let batch_throughput = batch_prompts.len() as f64 / batch_duration.as_secs_f64();
    let individual_throughput = batch_prompts.len() as f64 / individual_duration.as_secs_f64();
    let efficiency_ratio = batch_throughput / individual_throughput;

    println!("Batch throughput: {:.2} prompts/sec", batch_throughput);
    println!("Individual throughput: {:.2} prompts/sec", individual_throughput);
    println!("Batch efficiency: {:.2}x", efficiency_ratio);

    // Batch processing should be at least as efficient
    assert!(efficiency_ratio >= 0.8, "Batch processing should be reasonably efficient");

    println!("✅ Batch inference performance optimization test scaffolding created");
}

// ==============================================================================
// AC7: Cross-Validation Framework Integration Tests
// Tests feature spec: real-bitnet-model-integration-architecture.md#ac7
// ==============================================================================

/// Test C++ reference cross-validation integration
/// Validates inference output parity with C++ reference implementation
#[test]
#[cfg(all(feature = "inference", feature = "crossval"))]
fn test_cpp_inference_cross_validation() {
    // AC:7
    let config = InferenceTestConfig::from_env();
    config.skip_if_no_model();

    let cpp_dir =
        env::var("BITNET_CPP_DIR").expect("BITNET_CPP_DIR must be set for cross-validation");
    let model_path = config.model_path.unwrap();
    let test_prompt = "The capital of France is";

    // TODO: This test will initially fail - drives cross-validation implementation
    let model = load_real_model(&model_path).expect("Model should load");
    let tokenizer = create_or_load_tokenizer(&model, config.tokenizer_path.as_ref())
        .expect("Tokenizer should be available");

    let mut engine =
        ProductionInferenceEngine::new(model, tokenizer, EngineConfig::for_cross_validation())
            .expect("Engine should initialize");

    // Generate Rust implementation result
    let rust_result = futures::executor::block_on(engine.infer_with_metrics(test_prompt))
        .expect("Rust inference should succeed");

    // Generate C++ reference result
    let cpp_result = run_cpp_reference_inference(&cpp_dir, &model_path, test_prompt)
        .expect("C++ reference should succeed");

    // Cross-validate token sequences
    let token_comparison = compare_token_sequences(&rust_result.tokens, &cpp_result.tokens);

    if token_comparison.exact_match {
        println!("✅ Exact token match with C++ reference");
    } else {
        println!("Token differences detected:");
        println!("  Match rate: {:.2}%", token_comparison.match_rate * 100.0);
        println!("  First difference at position: {:?}", token_comparison.first_mismatch);

        // Allow some tolerance for non-deterministic generation
        assert!(token_comparison.match_rate >= 0.95, "Token match rate should be ≥95%");
    }

    // Cross-validate numerical accuracy if logits available
    if let (Some(rust_logits), Some(cpp_logits)) = (&rust_result.logits, &cpp_result.logits) {
        let numerical_comparison = compare_numerical_accuracy(rust_logits, cpp_logits, 1e-4);
        assert!(
            numerical_comparison.within_tolerance,
            "Numerical accuracy should be within tolerance"
        );

        println!(
            "Numerical accuracy: max_diff={:.6}, rmse={:.6}",
            numerical_comparison.max_difference, numerical_comparison.rmse
        );
    }

    // Cross-validate performance characteristics
    let performance_comparison =
        compare_performance_metrics(&rust_result.metrics, &cpp_result.metrics);

    println!("Performance comparison:");
    println!("  Rust throughput: {:.2} tokens/sec", rust_result.metrics.tokens_per_second);
    println!("  C++ throughput: {:.2} tokens/sec", cpp_result.metrics.tokens_per_second);
    println!("  Speedup: {:.2}x", performance_comparison.speedup_ratio);

    println!("✅ C++ inference cross-validation test scaffolding created");
}

/// Test perplexity calculation validation against reference
/// Validates perplexity calculations match C++ reference implementation
#[test]
#[cfg(all(feature = "inference", feature = "crossval", any()))] // Disabled: ProductionInferenceEngine not yet implemented
fn test_perplexity_calculation_cross_validation() {
    // AC:8
    let config = InferenceTestConfig::from_env();
    config.skip_if_no_model();

    let model_path = config.model_path.unwrap();
    let test_corpus = "The quick brown fox jumps over the lazy dog. This is a test sentence for perplexity calculation.";

    // TODO: This test will initially fail - drives perplexity calculation implementation
    let model = load_real_model(&model_path).expect("Model should load");
    let tokenizer = create_or_load_tokenizer(&model, config.tokenizer_path.as_ref())
        .expect("Tokenizer should be available");

    let mut engine =
        ProductionInferenceEngine::new(model, tokenizer, EngineConfig::for_evaluation())
            .expect("Engine should initialize");

    // Calculate perplexity with Rust implementation
    let rust_perplexity = futures::executor::block_on(engine.calculate_perplexity(test_corpus))
        .expect("Rust perplexity calculation should succeed");

    println!("Rust perplexity: {:.4}", rust_perplexity.value);

    // Calculate perplexity with C++ reference if available
    if let Ok(cpp_dir) = env::var("BITNET_CPP_DIR") {
        let cpp_perplexity = calculate_cpp_reference_perplexity(&cpp_dir, &model_path, test_corpus)
            .expect("C++ perplexity calculation should succeed");

        println!("C++ perplexity: {:.4}", cpp_perplexity.value);

        // Validate perplexity parity within tolerance
        let perplexity_diff = (rust_perplexity.value - cpp_perplexity.value).abs();
        let relative_error = perplexity_diff / cpp_perplexity.value;

        println!("Perplexity difference: {:.6}", perplexity_diff);
        println!("Relative error: {:.6} ({:.2}%)", relative_error, relative_error * 100.0);

        assert!(relative_error <= 0.001, "Perplexity should match within 0.1% tolerance");
    } else {
        println!("Skipping C++ perplexity comparison - BITNET_CPP_DIR not set");
    }

    // Validate perplexity calculation properties
    assert!(rust_perplexity.value > 1.0, "Perplexity should be > 1.0");
    assert!(rust_perplexity.value < 10000.0, "Perplexity should be reasonable");
    assert!(rust_perplexity.token_count > 0, "Should process tokens");
    assert!(rust_perplexity.log_likelihood < 0.0, "Log likelihood should be negative");

    println!("✅ Perplexity calculation cross-validation test scaffolding created");
}

// ==============================================================================
// Device-Aware Execution Tests
// ==============================================================================

/// Test device-aware inference execution with automatic fallback
/// Validates GPU acceleration with transparent CPU fallback
#[test]
#[cfg(all(feature = "inference", feature = "gpu", any()))] // Disabled: ProductionInferenceEngine not yet implemented
fn test_device_aware_inference_with_fallback() {
    // AC:3
    let config = InferenceTestConfig::from_env();
    config.skip_if_no_model();

    let model_path = config.model_path.unwrap();
    let test_prompt = "Device-aware execution test";

    // TODO: This test will initially fail - drives device-aware execution
    let model = load_real_model(&model_path).expect("Model should load");
    let tokenizer = create_or_load_tokenizer(&model, config.tokenizer_path.as_ref())
        .expect("Tokenizer should be available");

    // Test GPU-preferred configuration
    let gpu_config = EngineConfig {
        device_preference: "gpu".to_string(),
        cpu_fallback: true,
        device_validation: true,
        ..Default::default()
    };

    let mut gpu_engine =
        ProductionInferenceEngine::new(model.clone(), tokenizer.clone(), gpu_config)
            .expect("GPU engine should initialize");

    let gpu_result = futures::executor::block_on(gpu_engine.infer_with_metrics(test_prompt))
        .expect("GPU inference should succeed (with fallback)");

    // Test CPU-only configuration
    let cpu_config = EngineConfig { device_preference: "cpu".to_string(), ..Default::default() };

    let mut cpu_engine = ProductionInferenceEngine::new(model, tokenizer, cpu_config)
        .expect("CPU engine should initialize");

    let cpu_result = futures::executor::block_on(cpu_engine.infer_with_metrics(test_prompt))
        .expect("CPU inference should succeed");

    // Validate device execution results
    assert!(!gpu_result.text.is_empty(), "GPU result should generate text");
    assert!(!cpu_result.text.is_empty(), "CPU result should generate text");

    // Validate device information
    println!("GPU device used: {}", gpu_result.device_info.device_type);
    println!("CPU device used: {}", cpu_result.device_info.device_type);

    // Compare performance characteristics
    let gpu_throughput = gpu_result.metrics.tokens_per_second;
    let cpu_throughput = cpu_result.metrics.tokens_per_second;

    println!("GPU throughput: {:.2} tokens/sec", gpu_throughput);
    println!("CPU throughput: {:.2} tokens/sec", cpu_throughput);

    if gpu_throughput > cpu_throughput {
        println!("GPU acceleration effective: {:.2}x speedup", gpu_throughput / cpu_throughput);
    } else {
        println!("CPU fallback active or GPU not available");
    }

    // Test deterministic consistency between devices
    if env::var("BITNET_DETERMINISTIC").map(|v| v == "1").unwrap_or(false) {
        assert_eq!(
            gpu_result.tokens, cpu_result.tokens,
            "Deterministic mode should produce identical tokens"
        );
    }

    println!("✅ Device-aware inference with fallback test scaffolding created");
}

// ==============================================================================
// Helper Functions (Initially will not compile - drive implementation)
// ==============================================================================

#[cfg(all(feature = "inference", any()))] // Disabled: ProductionInferenceEngine not yet implemented
fn load_real_model(model_path: &Path) -> Result<BitNetModel, Box<dyn std::error::Error>> {
    use bitnet_common::Device;

    // Verify model file exists
    if !model_path.exists() {
        return Err(format!("Model file not found: {}", model_path.display()).into());
    }

    // Use CPU device for deterministic testing
    let device = Device::Cpu;

    // Load model using production loader
    // This follows the pattern from ac4_cross_validation_accuracy.rs:507-532
    let model = bitnet_models::load_gguf_full(model_path.to_str().unwrap(), device)
        .map_err(|e| format!("Failed to load GGUF model: {}", e))?;

    // Validate model structure - ensure essential tensors are present
    // This is a basic sanity check to ensure the model loaded correctly
    if model.tensor_names().is_empty() {
        return Err("Loaded model has no tensors - invalid model structure".into());
    }

    Ok(model)
}

#[cfg(all(feature = "inference", any()))] // Disabled: ProductionInferenceEngine not yet implemented
fn create_or_load_tokenizer(
    model: &BitNetModel,
    tokenizer_path: Option<&PathBuf>,
) -> Result<UniversalTokenizer, Box<dyn std::error::Error>> {
    use bitnet_tokenizers::TokenizerBackend;

    // Priority 1: Try explicit tokenizer path if provided
    if let Some(path) = tokenizer_path {
        if !path.exists() {
            return Err(format!("Tokenizer path does not exist: {}", path.display()).into());
        }

        // Load via universal loader and wrap with UniversalTokenizer
        // Note: The loader returns Arc<dyn Tokenizer> but we need UniversalTokenizer
        // For test scaffolding, we create from model config as a workaround
        let _external_tokenizer = bitnet_tokenizers::loader::load_tokenizer(path)
            .map_err(|e| format!("Failed to load tokenizer from {}: {}", path.display(), e))?;

        // Create UniversalTokenizer from model with Mock backend for test scaffolding
        // In production, this would use the actual loaded tokenizer
        return UniversalTokenizer::from_gguf_model_with_preference(model, TokenizerBackend::Mock)
            .map_err(|e| format!("Failed to create tokenizer from model: {}", e).into());
    }

    // Priority 2: Check BITNET_TOKENIZER environment variable
    if let Ok(env_path_str) = std::env::var("BITNET_TOKENIZER") {
        let env_path = PathBuf::from(env_path_str);
        if env_path.exists() {
            let _external_tokenizer = bitnet_tokenizers::loader::load_tokenizer(&env_path)
                .map_err(|e| {
                    format!(
                        "Failed to load tokenizer from BITNET_TOKENIZER={}: {}",
                        env_path.display(),
                        e
                    )
                })?;

            return UniversalTokenizer::from_gguf_model_with_preference(
                model,
                TokenizerBackend::Mock,
            )
            .map_err(|e| format!("Failed to create tokenizer from model: {}", e).into());
        }
    }

    // Priority 3: Fall back to creating from model config with Mock backend
    // This provides basic tokenization for tests even without external tokenizer
    UniversalTokenizer::from_gguf_model_with_preference(model, TokenizerBackend::Mock)
        .map_err(|e| {
            format!(
                "Failed to auto-discover tokenizer: {}. \
                 Provide explicit path via tokenizer_path parameter or set BITNET_TOKENIZER environment variable.",
                e
            )
            .into()
        })
}

#[cfg(all(feature = "inference", feature = "crossval"))]
fn validate_timing_metrics(metrics: &InferenceMetrics) -> Result<(), Box<dyn std::error::Error>> {
    // Validate all durations are positive
    if metrics.total_duration <= Duration::ZERO {
        return Err(
            format!("Total duration must be positive, got {:?}", metrics.total_duration).into()
        );
    }

    if metrics.prefill_duration <= Duration::ZERO {
        return Err(format!(
            "Prefill duration must be positive, got {:?}",
            metrics.prefill_duration
        )
        .into());
    }

    if metrics.decode_duration <= Duration::ZERO {
        return Err(
            format!("Decode duration must be positive, got {:?}", metrics.decode_duration).into()
        );
    }

    // Validate timing consistency: total ≈ prefill + decode (allow 10ms tolerance for overhead)
    let expected_total = metrics.prefill_duration + metrics.decode_duration;
    let difference = if metrics.total_duration > expected_total {
        metrics.total_duration - expected_total
    } else {
        expected_total - metrics.total_duration
    };

    if difference > Duration::from_millis(10) {
        return Err(format!(
            "Timing inconsistency: total={:?}, prefill={:?}, decode={:?}, difference={:?}",
            metrics.total_duration, metrics.prefill_duration, metrics.decode_duration, difference
        )
        .into());
    }

    Ok(())
}

#[cfg(all(feature = "inference", feature = "crossval"))]
fn validate_throughput_metrics(
    metrics: &InferenceMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    // Validate tokens_per_second is positive
    if metrics.tokens_per_second <= 0.0 {
        return Err(format!(
            "Tokens per second must be positive, got {}",
            metrics.tokens_per_second
        )
        .into());
    }

    // Validate tokens_per_second is not NaN or infinite
    if !metrics.tokens_per_second.is_finite() {
        return Err(
            format!("Tokens per second must be finite, got {}", metrics.tokens_per_second).into()
        );
    }

    // Validate throughput is reasonable (not impossibly high)
    // QK256 MVP: ~0.1 tok/s, Production I2S: ~10-100 tok/s, GPU: ~100-1000 tok/s
    const MAX_REASONABLE_THROUGHPUT: f64 = 10000.0; // 10k tok/s ceiling
    if metrics.tokens_per_second > MAX_REASONABLE_THROUGHPUT {
        return Err(format!(
            "Throughput suspiciously high: {} tok/s exceeds reasonable maximum of {} tok/s",
            metrics.tokens_per_second, MAX_REASONABLE_THROUGHPUT
        )
        .into());
    }

    // Cross-check throughput calculation consistency with total_duration
    // tokens_per_second should be calculated as: tokens_generated / total_duration.as_secs_f64()
    // We validate that the reported throughput is physically achievable given the total_duration
    // Minimum time per token at max throughput: 1 / MAX_REASONABLE_THROUGHPUT
    let min_time_per_token_secs = 1.0 / MAX_REASONABLE_THROUGHPUT;
    let min_required_duration = Duration::from_secs_f64(min_time_per_token_secs);

    if metrics.total_duration < min_required_duration && metrics.tokens_per_second > 1.0 {
        // If total_duration is very small but throughput is claimed to be > 1 tok/s,
        // ensure it's physically consistent
        // Allow some tolerance for measurement overhead (1ms minimum)
        if metrics.total_duration < Duration::from_millis(1) {
            return Err(format!(
                "Duration too small for reliable throughput measurement: {:?}",
                metrics.total_duration
            )
            .into());
        }
    }

    Ok(())
}

#[cfg(all(feature = "inference", feature = "crossval"))]
fn validate_memory_metrics(metrics: &InferenceMetrics) -> Result<(), Box<dyn std::error::Error>> {
    // MVP: Memory metrics not yet tracked in InferenceMetrics
    // This is a placeholder for future memory validation
    //
    // When memory tracking is added to InferenceMetrics, this will validate:
    // - Memory usage is positive (memory_usage_mb > 0.0)
    // - Memory usage is within reasonable bounds for BitNet model size
    //   - BitNet 2B model: ~500MB-2GB depending on quantization
    //   - MAX_REASONABLE_MEMORY_MB: 10GB ceiling for safety
    // - Peak memory usage doesn't exceed system constraints
    // - No memory leaks detected between inference runs
    //
    // Expected InferenceMetrics extension (future):
    //   memory_usage_mb: Option<f64>,        // Current memory usage in MB
    //   peak_memory_mb: Option<f64>,         // Peak memory during inference
    //   memory_allocations: Option<usize>,   // Number of allocations
    //
    // Future implementation example:
    //   if let Some(memory_usage) = metrics.memory_usage_mb {
    //       if memory_usage <= 0.0 {
    //           return Err(format!("Memory usage must be positive, got {} MB", memory_usage).into());
    //       }
    //       const MAX_REASONABLE_MEMORY_MB: f64 = 10000.0; // 10GB ceiling
    //       if memory_usage > MAX_REASONABLE_MEMORY_MB {
    //           return Err(format!(
    //               "Memory usage suspiciously high: {} MB exceeds {} MB",
    //               memory_usage,
    //               MAX_REASONABLE_MEMORY_MB
    //           ).into());
    //       }
    //   }

    let _ = metrics; // Use parameter to avoid unused warnings
    Ok(())
}

#[cfg(all(feature = "inference", feature = "crossval"))]
fn calculate_performance_statistics(metrics: &[InferenceMetrics]) -> PerformanceStatistics {
    assert!(!metrics.is_empty(), "Cannot calculate statistics from empty metrics");

    let n = metrics.len() as f64;

    // Calculate mean throughput
    let mean_throughput: f64 = metrics.iter().map(|m| m.tokens_per_second).sum::<f64>() / n;

    // Calculate throughput standard deviation
    let variance: f64 = metrics
        .iter()
        .map(|m| {
            let diff = m.tokens_per_second - mean_throughput;
            diff * diff
        })
        .sum::<f64>()
        / n;
    let throughput_std_dev = variance.sqrt();

    // Calculate mean latency
    let total_duration_sum: Duration = metrics.iter().map(|m| m.total_duration).sum();
    let mean_latency = total_duration_sum / metrics.len() as u32;

    PerformanceStatistics { mean_throughput, throughput_std_dev, mean_latency }
}

#[cfg(all(feature = "inference", any()))] // Disabled: ProductionInferenceEngine not yet implemented
fn create_cold_engine(model_path: &Path) -> Result<ProductionInferenceEngine, InferenceError> {
    use bitnet_common::Device;
    use bitnet_inference::ProductionInferenceEngine;
    use bitnet_models::{LoadConfig, ModelLoader};
    use std::sync::Arc;

    // Determine device from environment or default to CPU
    let device_str = std::env::var("BITNET_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    let device = match device_str.to_lowercase().as_str() {
        "gpu" | "cuda" => Device::Cuda(0),
        "metal" => Device::Metal,
        _ => Device::Cpu,
    };

    // Load the model using ModelLoader
    let loader = ModelLoader::new(device);
    let load_config = LoadConfig {
        use_mmap: true,
        validate_checksums: false, // Disable for cold start speed
        progress_callback: None,
    };

    let model = loader.load_with_config(model_path, &load_config).map_err(|e| {
        InferenceError::GenerationFailed { reason: format!("Failed to load model: {}", e) }
    })?;

    // Load or create tokenizer
    // Try to load tokenizer from same directory as model or embedded in GGUF
    let tokenizer_path = model_path.with_extension("json");
    let tokenizer: Arc<dyn bitnet_tokenizers::Tokenizer> = if tokenizer_path.exists() {
        bitnet_tokenizers::load_tokenizer(&tokenizer_path).map_err(|e| {
            InferenceError::TokenizationFailed {
                reason: format!("Failed to load tokenizer: {}", e),
            }
        })?
    } else {
        // Try loading tokenizer from GGUF file itself
        bitnet_tokenizers::load_tokenizer(model_path).map_err(|e| {
            InferenceError::TokenizationFailed {
                reason: format!("Failed to load embedded tokenizer: {}", e),
            }
        })?
    };

    // Create a cold engine (no prefill, no cache warming)
    let engine = ProductionInferenceEngine::new(Arc::from(model), tokenizer, device).map_err(
        |e| match e {
            bitnet_common::BitNetError::Inference(err) => err,
            other => InferenceError::GenerationFailed {
                reason: format!("Engine creation failed: {}", other),
            },
        },
    )?;

    Ok(engine)
}

#[cfg(all(feature = "inference", feature = "crossval"))]
fn run_cpp_reference_inference(
    _cpp_dir: &str,
    model_path: &Path,
    prompt: &str,
) -> Result<CppInferenceResult, Box<dyn std::error::Error>> {
    // Check if C++ reference is available via FFI
    #[cfg(feature = "ffi")]
    {
        use bitnet_sys::wrapper::{self, Session as CppSession};

        // Initialize C++ backend
        wrapper::init_backend();

        // Load model with deterministic settings for cross-validation
        let model_path_str = model_path.to_str().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid model path encoding")
        })?;

        let mut session = CppSession::load_deterministic(model_path_str)?;

        // Measure inference performance
        let start_time = Instant::now();

        // Generate tokens using C++ reference implementation
        // Use default max_tokens=32 for cross-validation tests
        let generated_tokens = session.generate_greedy(prompt, 32)?;

        let inference_duration = start_time.elapsed();

        // Free the C++ backend resources
        wrapper::free_backend();

        // Calculate tokens per second
        let tokens_per_second = if inference_duration.as_secs_f64() > 0.0 {
            generated_tokens.len() as f64 / inference_duration.as_secs_f64()
        } else {
            0.0
        };

        // Convert i32 tokens to u32 for compatibility with Rust implementation
        let tokens: Vec<u32> = generated_tokens.iter().map(|&t| t as u32).collect();

        // Note: Logits are not extracted in this basic implementation
        // They would require additional C++ API calls to retrieve per-token logits
        let logits = None;

        Ok(CppInferenceResult { tokens, logits, metrics: CppMetrics { tokens_per_second } })
    }

    #[cfg(not(feature = "ffi"))]
    {
        Err(format!(
            "C++ reference inference requires 'ffi' feature and BITNET_CPP_DIR environment variable. \
             Set BITNET_CPP_DIR to the path of bitnet.cpp repository."
        )
        .into())
    }
}

#[cfg(all(feature = "inference", feature = "crossval"))]
fn compare_token_sequences(rust_tokens: &[u32], cpp_tokens: &[u32]) -> TokenComparison {
    // Check for exact match first
    let exact_match = rust_tokens == cpp_tokens;

    if exact_match {
        return TokenComparison { exact_match: true, match_rate: 1.0, first_mismatch: None };
    }

    // Calculate match rate and find first mismatch
    let min_len = rust_tokens.len().min(cpp_tokens.len());
    let mut matching_tokens = 0;
    let mut first_mismatch = None;

    for (i, (rust_token, cpp_token)) in rust_tokens.iter().zip(cpp_tokens.iter()).enumerate() {
        if rust_token == cpp_token {
            matching_tokens += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = Some(i);
        }
    }

    // If one sequence is longer, the first mismatch is at the end of the shorter sequence
    if first_mismatch.is_none() && rust_tokens.len() != cpp_tokens.len() {
        first_mismatch = Some(min_len);
    }

    // Calculate match rate based on the longer sequence length
    let max_len = rust_tokens.len().max(cpp_tokens.len());
    let match_rate = if max_len > 0 { matching_tokens as f64 / max_len as f64 } else { 1.0 };

    TokenComparison { exact_match: false, match_rate, first_mismatch }
}

#[cfg(all(feature = "inference", feature = "crossval"))]
fn compare_numerical_accuracy(
    rust_logits: &[f32],
    cpp_logits: &[f32],
    tolerance: f32,
) -> NumericalComparison {
    // Ensure both arrays have the same length
    assert_eq!(
        rust_logits.len(),
        cpp_logits.len(),
        "Logit arrays must have the same length for comparison"
    );

    if rust_logits.is_empty() {
        return NumericalComparison { within_tolerance: true, max_difference: 0.0, rmse: 0.0 };
    }

    // Calculate max absolute error
    let mut max_diff = 0.0_f32;
    let mut sum_squared_error = 0.0_f32;

    for (rust_val, cpp_val) in rust_logits.iter().zip(cpp_logits.iter()) {
        let abs_diff = (rust_val - cpp_val).abs();
        max_diff = max_diff.max(abs_diff);
        sum_squared_error += abs_diff * abs_diff;
    }

    // Calculate RMSE (Root Mean Square Error)
    let rmse = (sum_squared_error / rust_logits.len() as f32).sqrt();

    // Check if all differences are within tolerance
    let within_tolerance = max_diff <= tolerance;

    NumericalComparison { within_tolerance, max_difference: max_diff, rmse }
}

#[cfg(all(feature = "inference", feature = "crossval"))]
fn compare_performance_metrics(
    rust_metrics: &InferenceMetrics,
    cpp_metrics: &CppMetrics,
) -> PerformanceComparison {
    // Calculate speedup ratio (Rust throughput / C++ throughput)
    // A ratio > 1.0 means Rust is faster, < 1.0 means C++ is faster
    let speedup_ratio = if cpp_metrics.tokens_per_second > 0.0 {
        rust_metrics.tokens_per_second / cpp_metrics.tokens_per_second
    } else {
        // If C++ throughput is 0, we can't calculate meaningful speedup
        // Return 1.0 to indicate equivalent performance (avoid division by zero)
        1.0
    };

    PerformanceComparison { speedup_ratio }
}

#[cfg(all(feature = "inference", feature = "crossval"))]
fn calculate_cpp_reference_perplexity(
    _cpp_dir: &str,
    model_path: &Path,
    corpus: &str,
) -> Result<PerplexityResult, Box<dyn std::error::Error>> {
    #[cfg(feature = "ffi")]
    {
        use bitnet_sys::{
            BitnetContext, BitnetModel, bitnet_eval_tokens, bitnet_tokenize_text, cpp_vocab_size,
        };

        // Load C++ model
        let model_str = model_path.to_string_lossy().to_string();
        let cpp_model = BitnetModel::from_file(&model_str)
            .map_err(|e| format!("Failed to load C++ model: {:?}", e))?;

        // Create context with reasonable defaults
        let cpp_ctx = BitnetContext::new(&cpp_model, 4096, 1, 0)
            .map_err(|e| format!("Failed to create C++ context: {:?}", e))?;

        // Get vocabulary size
        let vocab_size =
            cpp_vocab_size(&cpp_ctx).map_err(|e| format!("Failed to get vocab size: {:?}", e))?;

        // Tokenize the corpus with add_bos=true, parse_special=true (standard settings)
        let token_ids = bitnet_tokenize_text(&cpp_model, corpus, true, true)
            .map_err(|e| format!("Failed to tokenize corpus: {:?}", e))?;

        if token_ids.is_empty() {
            return Err("Tokenization produced no tokens".into());
        }

        let token_count = token_ids.len();

        // Calculate perplexity using standard formula:
        // For each token position i (except the first), get logits for position i-1
        // and compute cross-entropy against ground truth token i
        let mut total_log_likelihood = 0.0_f64;
        let mut valid_tokens = 0usize;

        // Process each token position (skip first token as we need previous context)
        for i in 1..token_ids.len() {
            // Get context tokens up to position i (excluding current token)
            let context_tokens = &token_ids[0..i];
            let target_token = token_ids[i] as u32;

            // Evaluate logits for the context
            let logits = bitnet_eval_tokens(&cpp_ctx, context_tokens, vocab_size)
                .map_err(|e| format!("Failed to evaluate tokens at position {}: {:?}", i, e))?;

            if logits.len() != vocab_size {
                return Err(format!("Expected {} logits, got {}", vocab_size, logits.len()).into());
            }

            // Ensure target token is within vocabulary
            if (target_token as usize) >= vocab_size {
                return Err(format!(
                    "Target token {} exceeds vocab size {}",
                    target_token, vocab_size
                )
                .into());
            }

            // Compute softmax (numerically stable version)
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f64 = logits.iter().map(|&x| ((x - max_logit) as f64).exp()).sum();

            // Get probability of target token
            let target_logit = logits[target_token as usize];
            let target_prob = ((target_logit - max_logit) as f64).exp() / sum_exp;

            // Add to log likelihood (clamp probability to avoid log(0))
            let clamped_prob = target_prob.max(1e-10);
            total_log_likelihood += clamped_prob.ln();
            valid_tokens += 1;
        }

        if valid_tokens == 0 {
            return Err("No valid tokens for perplexity calculation".into());
        }

        // Calculate perplexity: exp(-average log likelihood)
        let avg_log_likelihood = total_log_likelihood / valid_tokens as f64;
        let perplexity = (-avg_log_likelihood).exp();

        Ok(PerplexityResult {
            value: perplexity,
            token_count: valid_tokens,
            log_likelihood: total_log_likelihood,
        })
    }

    #[cfg(not(feature = "ffi"))]
    {
        Err("FFI feature not enabled - cannot compute C++ reference perplexity".into())
    }
}

// Type definitions that will be implemented
#[cfg(all(feature = "inference", feature = "crossval"))]
struct PerformanceStatistics {
    mean_throughput: f64,
    throughput_std_dev: f64,
    mean_latency: Duration,
}

#[cfg(all(feature = "inference", any()))] // Disabled: ProductionInferenceEngine not yet implemented
struct DeviceInfo {
    device_type: String,
}

#[cfg(all(feature = "inference", feature = "crossval"))]
struct InferenceMetrics {
    /// Total inference duration
    total_duration: Duration,
    /// Prefill stage duration
    prefill_duration: Duration,
    /// Decode stage duration
    decode_duration: Duration,
    /// Tokens per second throughput
    tokens_per_second: f64,
}

#[cfg(all(feature = "inference", any()))] // Disabled: ProductionInferenceEngine not yet implemented
struct PerplexityResult {
    value: f64,
    token_count: usize,
    log_likelihood: f64,
}

#[cfg(all(feature = "inference", feature = "crossval"))]
struct CppInferenceResult {
    tokens: Vec<u32>,
    logits: Option<Vec<f32>>,
    metrics: CppMetrics,
}

#[cfg(all(feature = "inference", feature = "crossval"))]
struct CppMetrics {
    tokens_per_second: f64,
}

#[cfg(all(feature = "inference", feature = "crossval"))]
struct TokenComparison {
    exact_match: bool,
    match_rate: f64,
    first_mismatch: Option<usize>,
}

#[cfg(all(feature = "inference", feature = "crossval"))]
struct NumericalComparison {
    within_tolerance: bool,
    max_difference: f32,
    rmse: f32,
}

#[cfg(all(feature = "inference", feature = "crossval"))]
struct PerformanceComparison {
    speedup_ratio: f64,
}

#[cfg(all(feature = "inference", any()))] // Disabled: ProductionInferenceEngine not yet implemented
struct EngineConfig {
    device_preference: String,
    enable_performance_monitoring: bool,
    prefill_optimization: bool,
    batch_processing: bool,
    memory_optimization: bool,
    detailed_metrics: bool,
    memory_tracking: bool,
    device_profiling: bool,
    batch_size: usize,
    cpu_fallback: bool,
    device_validation: bool,
}

#[cfg(all(feature = "inference", any()))] // Disabled: ProductionInferenceEngine not yet implemented
impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            device_preference: "auto".to_string(),
            enable_performance_monitoring: false,
            prefill_optimization: true,
            batch_processing: false,
            memory_optimization: true,
            detailed_metrics: false,
            memory_tracking: false,
            device_profiling: false,
            batch_size: 1,
            cpu_fallback: true,
            device_validation: false,
        }
    }
}

#[cfg(all(feature = "inference", any()))] // Disabled: ProductionInferenceEngine not yet implemented
impl EngineConfig {
    fn with_prefill_optimization() -> Self {
        Self {
            prefill_optimization: true,
            enable_performance_monitoring: true,
            memory_optimization: true,
            batch_processing: false,
            ..Default::default()
        }
    }

    fn for_cross_validation() -> Self {
        // Cross-validation mode: deterministic, CPU-only execution for parity with C++ reference
        // Requirements:
        // - Deterministic execution (no GPU non-determinism)
        // - CPU-only (force CPU device preference)
        // - Disable optimizations that may affect output
        // - Enable performance monitoring to compare with C++ throughput
        // - Single-threaded (set via RAYON_NUM_THREADS=1 environment variable)
        // - Sequential processing (batch_size: 1)
        //
        // Expected environment variables:
        // - BITNET_DETERMINISTIC=1
        // - BITNET_SEED=42
        // - RAYON_NUM_THREADS=1
        Self {
            device_preference: "cpu".to_string(), // Force CPU for determinism
            enable_performance_monitoring: true,  // Compare throughput with C++
            prefill_optimization: false,          // Disable for determinism
            batch_processing: false,              // Sequential processing
            memory_optimization: false,           // Avoid optimization artifacts
            detailed_metrics: true,               // Detailed comparison data
            memory_tracking: true,                // Track memory usage
            device_profiling: false,              // CPU doesn't need profiling
            batch_size: 1,                        // Single sequence
            cpu_fallback: false,                  // Already CPU-only
            device_validation: true,              // Validate device selection
        }
    }

    fn for_evaluation() -> Self {
        // Evaluation mode: optimized for perplexity calculation and model quality testing
        // Requirements:
        // - Optimize for accuracy over speed
        // - Enable detailed metrics for perplexity calculation
        // - Deterministic execution for reproducible evaluation
        // - Support batch processing for efficient evaluation on large corpora
        // - Track memory usage for large evaluation sets
        //
        // Design rationale:
        // - batch_processing: true - Efficient corpus evaluation
        // - batch_size: 8 - Balance memory and throughput
        // - detailed_metrics: true - Capture all perplexity components
        // - prefill_optimization: true - Fast context processing
        // - memory_tracking: true - Monitor large corpus processing
        Self {
            device_preference: "auto".to_string(), // Auto-select for performance
            enable_performance_monitoring: true,   // Track evaluation time
            prefill_optimization: true,            // Efficient context processing
            batch_processing: true,                // Evaluate multiple sequences
            memory_optimization: true,             // Handle large corpora
            detailed_metrics: true,                // Capture perplexity metrics
            memory_tracking: true,                 // Monitor memory usage
            device_profiling: false,               // Not needed for evaluation
            batch_size: 8,                         // Reasonable batch size
            cpu_fallback: true,                    // Graceful degradation
            device_validation: true,               // Ensure device availability
        }
    }
}

// ==============================================================================
// Unit Tests for Helper Functions
// ==============================================================================

#[cfg(all(test, feature = "crossval"))]
mod numerical_accuracy_tests {
    use super::*;

    #[test]
    fn test_compare_numerical_accuracy_exact_match() {
        let rust_logits = vec![1.0, 2.0, 3.0, 4.0];
        let cpp_logits = vec![1.0, 2.0, 3.0, 4.0];

        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 1e-6);

        assert!(result.within_tolerance, "Exact match should be within tolerance");
        assert_eq!(result.max_difference, 0.0, "Max difference should be 0.0");
        assert_eq!(result.rmse, 0.0, "RMSE should be 0.0");
    }

    #[test]
    fn test_compare_numerical_accuracy_small_difference_within_tolerance() {
        let rust_logits = vec![1.0, 2.0, 3.0, 4.0];
        let cpp_logits = vec![1.00001, 2.00001, 3.00001, 4.00001];

        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 1e-4);

        assert!(result.within_tolerance, "Small differences should be within tolerance");
        assert!(result.max_difference < 1e-4, "Max difference should be < 1e-4");
        assert!(result.rmse < 1e-4, "RMSE should be < 1e-4");
    }

    #[test]
    fn test_compare_numerical_accuracy_large_difference_exceeds_tolerance() {
        let rust_logits = vec![1.0, 2.0, 3.0, 4.0];
        let cpp_logits = vec![1.1, 2.1, 3.1, 4.1];

        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 1e-4);

        assert!(!result.within_tolerance, "Large differences should exceed tolerance");
        assert!((result.max_difference - 0.1).abs() < 1e-6, "Max difference should be ~0.1");
        assert!(result.rmse > 0.0, "RMSE should be > 0.0");
    }

    #[test]
    fn test_compare_numerical_accuracy_empty_arrays() {
        let rust_logits: Vec<f32> = vec![];
        let cpp_logits: Vec<f32> = vec![];

        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 1e-4);

        assert!(result.within_tolerance, "Empty arrays should be within tolerance");
        assert_eq!(result.max_difference, 0.0, "Max difference should be 0.0");
        assert_eq!(result.rmse, 0.0, "RMSE should be 0.0");
    }

    #[test]
    fn test_compare_numerical_accuracy_rmse_calculation() {
        // Test RMSE calculation with known values
        let rust_logits = vec![0.0, 0.0, 0.0, 0.0];
        let cpp_logits = vec![1.0, 1.0, 1.0, 1.0];

        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 2.0);

        // RMSE should be sqrt((1^2 + 1^2 + 1^2 + 1^2) / 4) = sqrt(1) = 1.0
        assert!((result.rmse - 1.0).abs() < 1e-6, "RMSE should be 1.0");
        assert!(result.within_tolerance, "Should be within tolerance of 2.0");
        assert_eq!(result.max_difference, 1.0, "Max difference should be 1.0");
    }

    #[test]
    #[should_panic(expected = "Logit arrays must have the same length for comparison")]
    fn test_compare_numerical_accuracy_mismatched_lengths() {
        let rust_logits = vec![1.0, 2.0, 3.0];
        let cpp_logits = vec![1.0, 2.0, 3.0, 4.0];

        compare_numerical_accuracy(&rust_logits, &cpp_logits, 1e-4);
    }

    #[test]
    fn test_compare_numerical_accuracy_mixed_positive_negative() {
        let rust_logits = vec![-1.0, 2.0, -3.0, 4.0];
        let cpp_logits = vec![-1.1, 2.1, -3.1, 4.1];

        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 0.15);

        assert!(result.within_tolerance, "Should be within tolerance of 0.15");
        assert!((result.max_difference - 0.1).abs() < 1e-6, "Max difference should be ~0.1");
    }
}

// ==============================================================================
// Unit tests for token sequence comparison
// ==============================================================================

#[cfg(all(test, feature = "crossval"))]
mod token_comparison_tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let rust_tokens = vec![1, 2, 3, 4, 5];
        let cpp_tokens = vec![1, 2, 3, 4, 5];

        let result = compare_token_sequences(&rust_tokens, &cpp_tokens);

        assert!(result.exact_match, "Should be exact match");
        assert_eq!(result.match_rate, 1.0, "Match rate should be 100%");
        assert_eq!(result.first_mismatch, None, "Should have no mismatch");
    }

    #[test]
    fn test_partial_match() {
        let rust_tokens = vec![1, 2, 3, 4, 5];
        let cpp_tokens = vec![1, 2, 9, 4, 5];

        let result = compare_token_sequences(&rust_tokens, &cpp_tokens);

        assert!(!result.exact_match, "Should not be exact match");
        assert_eq!(result.match_rate, 0.8, "Match rate should be 80% (4/5)");
        assert_eq!(result.first_mismatch, Some(2), "First mismatch at index 2");
    }

    #[test]
    fn test_different_lengths_shorter_rust() {
        let rust_tokens = vec![1, 2, 3];
        let cpp_tokens = vec![1, 2, 3, 4, 5];

        let result = compare_token_sequences(&rust_tokens, &cpp_tokens);

        assert!(!result.exact_match, "Should not be exact match");
        assert_eq!(result.match_rate, 0.6, "Match rate should be 60% (3/5)");
        assert_eq!(result.first_mismatch, Some(3), "First mismatch at end of shorter sequence");
    }

    #[test]
    fn test_different_lengths_longer_rust() {
        let rust_tokens = vec![1, 2, 3, 4, 5];
        let cpp_tokens = vec![1, 2, 3];

        let result = compare_token_sequences(&rust_tokens, &cpp_tokens);

        assert!(!result.exact_match, "Should not be exact match");
        assert_eq!(result.match_rate, 0.6, "Match rate should be 60% (3/5)");
        assert_eq!(result.first_mismatch, Some(3), "First mismatch at end of shorter sequence");
    }

    #[test]
    fn test_completely_different() {
        let rust_tokens = vec![1, 2, 3, 4, 5];
        let cpp_tokens = vec![6, 7, 8, 9, 10];

        let result = compare_token_sequences(&rust_tokens, &cpp_tokens);

        assert!(!result.exact_match, "Should not be exact match");
        assert_eq!(result.match_rate, 0.0, "Match rate should be 0%");
        assert_eq!(result.first_mismatch, Some(0), "First mismatch at index 0");
    }

    #[test]
    fn test_empty_sequences() {
        let rust_tokens: Vec<u32> = vec![];
        let cpp_tokens: Vec<u32> = vec![];

        let result = compare_token_sequences(&rust_tokens, &cpp_tokens);

        assert!(result.exact_match, "Empty sequences should be exact match");
        assert_eq!(result.match_rate, 1.0, "Match rate should be 100%");
        assert_eq!(result.first_mismatch, None, "Should have no mismatch");
    }

    #[test]
    fn test_one_empty_sequence() {
        let rust_tokens = vec![1, 2, 3];
        let cpp_tokens: Vec<u32> = vec![];

        let result = compare_token_sequences(&rust_tokens, &cpp_tokens);

        assert!(!result.exact_match, "Should not be exact match");
        assert_eq!(result.match_rate, 0.0, "Match rate should be 0%");
        assert_eq!(result.first_mismatch, Some(0), "First mismatch at index 0");
    }

    #[test]
    fn test_single_token_match() {
        let rust_tokens = vec![42];
        let cpp_tokens = vec![42];

        let result = compare_token_sequences(&rust_tokens, &cpp_tokens);

        assert!(result.exact_match, "Should be exact match");
        assert_eq!(result.match_rate, 1.0, "Match rate should be 100%");
        assert_eq!(result.first_mismatch, None, "Should have no mismatch");
    }

    #[test]
    fn test_single_token_mismatch() {
        let rust_tokens = vec![42];
        let cpp_tokens = vec![99];

        let result = compare_token_sequences(&rust_tokens, &cpp_tokens);

        assert!(!result.exact_match, "Should not be exact match");
        assert_eq!(result.match_rate, 0.0, "Match rate should be 0%");
        assert_eq!(result.first_mismatch, Some(0), "First mismatch at index 0");
    }

    #[test]
    fn test_high_match_rate_threshold() {
        // Test the 95% threshold mentioned in the test
        let rust_tokens = vec![1; 100];
        let mut cpp_tokens = vec![1; 100];

        // Introduce 5 mismatches (95% match rate)
        for i in (0..100).step_by(20) {
            cpp_tokens[i] = 999;
        }

        let result = compare_token_sequences(&rust_tokens, &cpp_tokens);

        assert!(!result.exact_match, "Should not be exact match");
        assert_eq!(result.match_rate, 0.95, "Match rate should be 95%");
        assert!(result.match_rate >= 0.95, "Should meet 95% threshold");
        assert_eq!(result.first_mismatch, Some(0), "First mismatch at index 0");
    }

    #[test]
    fn test_mismatch_in_middle() {
        let rust_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let cpp_tokens = vec![1, 2, 3, 4, 99, 6, 7, 8, 9, 10];

        let result = compare_token_sequences(&rust_tokens, &cpp_tokens);

        assert!(!result.exact_match, "Should not be exact match");
        assert_eq!(result.match_rate, 0.9, "Match rate should be 90% (9/10)");
        assert_eq!(result.first_mismatch, Some(4), "First mismatch at index 4");
    }

    #[test]
    fn test_multiple_mismatches() {
        let rust_tokens = vec![1, 2, 3, 4, 5];
        let cpp_tokens = vec![1, 99, 3, 88, 5];

        let result = compare_token_sequences(&rust_tokens, &cpp_tokens);

        assert!(!result.exact_match, "Should not be exact match");
        assert_eq!(result.match_rate, 0.6, "Match rate should be 60% (3/5)");
        assert_eq!(result.first_mismatch, Some(1), "First mismatch at index 1");
    }

    #[test]
    fn test_longer_sequences() {
        let rust_tokens: Vec<u32> = (0..1000).collect();
        let cpp_tokens: Vec<u32> = (0..1000).collect();

        let result = compare_token_sequences(&rust_tokens, &cpp_tokens);

        assert!(result.exact_match, "Should be exact match");
        assert_eq!(result.match_rate, 1.0, "Match rate should be 100%");
        assert_eq!(result.first_mismatch, None, "Should have no mismatch");
    }
}

// ==============================================================================
// Unit tests for timing metrics validation
// ==============================================================================

#[cfg(all(test, feature = "crossval"))]
mod timing_metrics_validation_tests {
    use super::*;

    #[test]
    fn test_validate_timing_metrics_valid_consistent() {
        let metrics = InferenceMetrics {
            total_duration: Duration::from_millis(100),
            prefill_duration: Duration::from_millis(40),
            decode_duration: Duration::from_millis(60),
            tokens_per_second: 10.0,
        };

        let result = validate_timing_metrics(&metrics);
        assert!(result.is_ok(), "Valid consistent metrics should pass: {:?}", result.err());
    }

    #[test]
    fn test_validate_timing_metrics_with_small_overhead() {
        // Total duration slightly exceeds sum due to overhead (within 10ms tolerance)
        let metrics = InferenceMetrics {
            total_duration: Duration::from_millis(105),
            prefill_duration: Duration::from_millis(40),
            decode_duration: Duration::from_millis(60),
            tokens_per_second: 10.0,
        };

        let result = validate_timing_metrics(&metrics);
        assert!(result.is_ok(), "Metrics with small overhead should pass: {:?}", result.err());
    }

    #[test]
    fn test_validate_timing_metrics_zero_total_duration() {
        let metrics = InferenceMetrics {
            total_duration: Duration::ZERO,
            prefill_duration: Duration::from_millis(40),
            decode_duration: Duration::from_millis(60),
            tokens_per_second: 10.0,
        };

        let result = validate_timing_metrics(&metrics);
        assert!(result.is_err(), "Zero total duration should fail");
        assert!(result.unwrap_err().to_string().contains("Total duration must be positive"));
    }

    #[test]
    fn test_validate_timing_metrics_zero_prefill_duration() {
        let metrics = InferenceMetrics {
            total_duration: Duration::from_millis(100),
            prefill_duration: Duration::ZERO,
            decode_duration: Duration::from_millis(60),
            tokens_per_second: 10.0,
        };

        let result = validate_timing_metrics(&metrics);
        assert!(result.is_err(), "Zero prefill duration should fail");
        assert!(result.unwrap_err().to_string().contains("Prefill duration must be positive"));
    }

    #[test]
    fn test_validate_timing_metrics_zero_decode_duration() {
        let metrics = InferenceMetrics {
            total_duration: Duration::from_millis(100),
            prefill_duration: Duration::from_millis(40),
            decode_duration: Duration::ZERO,
            tokens_per_second: 10.0,
        };

        let result = validate_timing_metrics(&metrics);
        assert!(result.is_err(), "Zero decode duration should fail");
        assert!(result.unwrap_err().to_string().contains("Decode duration must be positive"));
    }

    #[test]
    fn test_validate_timing_metrics_timing_inconsistency_exceeds_tolerance() {
        // Total duration is 50ms less than sum (exceeds 10ms tolerance)
        let metrics = InferenceMetrics {
            total_duration: Duration::from_millis(50),
            prefill_duration: Duration::from_millis(40),
            decode_duration: Duration::from_millis(60),
            tokens_per_second: 10.0,
        };

        let result = validate_timing_metrics(&metrics);
        assert!(result.is_err(), "Large timing inconsistency should fail");
        assert!(result.unwrap_err().to_string().contains("Timing inconsistency"));
    }

    #[test]
    fn test_validate_timing_metrics_total_exceeds_sum_beyond_tolerance() {
        // Total duration is 50ms more than sum (exceeds 10ms tolerance)
        let metrics = InferenceMetrics {
            total_duration: Duration::from_millis(150),
            prefill_duration: Duration::from_millis(40),
            decode_duration: Duration::from_millis(60),
            tokens_per_second: 10.0,
        };

        let result = validate_timing_metrics(&metrics);
        assert!(result.is_err(), "Total exceeding sum beyond tolerance should fail");
        assert!(result.unwrap_err().to_string().contains("Timing inconsistency"));
    }

    #[test]
    fn test_validate_timing_metrics_exact_tolerance_boundary() {
        // Exactly 10ms difference - should still pass (boundary case)
        let metrics = InferenceMetrics {
            total_duration: Duration::from_millis(110),
            prefill_duration: Duration::from_millis(40),
            decode_duration: Duration::from_millis(60),
            tokens_per_second: 10.0,
        };

        let result = validate_timing_metrics(&metrics);
        assert!(result.is_ok(), "Exactly at tolerance boundary should pass: {:?}", result.err());
    }

    #[test]
    fn test_validate_timing_metrics_microsecond_precision() {
        // Test with microsecond precision
        let metrics = InferenceMetrics {
            total_duration: Duration::from_micros(100_500),
            prefill_duration: Duration::from_micros(40_200),
            decode_duration: Duration::from_micros(60_300),
            tokens_per_second: 10.0,
        };

        let result = validate_timing_metrics(&metrics);
        assert!(result.is_ok(), "Microsecond precision metrics should pass: {:?}", result.err());
    }

    #[test]
    fn test_validate_timing_metrics_realistic_inference_times() {
        // Realistic inference times: 2 seconds total, 500ms prefill, 1.5s decode
        let metrics = InferenceMetrics {
            total_duration: Duration::from_millis(2000),
            prefill_duration: Duration::from_millis(500),
            decode_duration: Duration::from_millis(1500),
            tokens_per_second: 16.0,
        };

        let result = validate_timing_metrics(&metrics);
        assert!(result.is_ok(), "Realistic inference times should pass: {:?}", result.err());
    }
}
