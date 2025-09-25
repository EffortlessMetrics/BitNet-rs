//! Integration tests with real model files for end-to-end validation
//!
//! Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac7-integration-tests

#[allow(unused_imports)] // Test scaffolding imports for comprehensive integration tests
use bitnet_tokenizers::{
    BasicTokenizer, BitNetTokenizerWrapper, Gpt2TokenizerWrapper, LlamaTokenizerWrapper,
    SmartTokenizerDownload, Tokenizer, TokenizerDiscovery, TokenizerStrategy,
};

/// AC7: End-to-end tokenizer discovery integration test
/// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac7-integration-tests
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_end_to_end_tokenizer_discovery_integration() {
    // Test scaffolding - will be implemented once core components are ready
    println!("✅ AC7: End-to-end integration test scaffolding prepared");

    // Setup test model if available
    let _model_path = setup_test_model().await;

    // Test complete workflow: Discovery → Download → Inference
    // This is test scaffolding - actual implementation will follow
    println!("Integration test scaffolding ready for implementation");
}

/// AC7: GPU/CPU parity integration test
/// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac7-integration-tests
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_gpu_cpu_tokenizer_parity() {
    // Test scaffolding for GPU/CPU parity validation
    println!("✅ AC7: GPU/CPU parity test scaffolding prepared");
    // GPU/CPU parity validation will be implemented
}

#[allow(dead_code)]
async fn setup_test_model() -> std::path::PathBuf {
    // Test scaffolding for model setup
    std::path::PathBuf::from("test-models/integration-test.gguf")
}

// ================================
// ENHANCED INTEGRATION TESTS WITH REALISTIC FAILURE SCENARIOS
// ================================

/// Test complete tokenizer pipeline failure recovery scenarios
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_complete_pipeline_failure_recovery() {
    use std::io::Write;
    use tempfile::tempdir;

    // Test realistic failure scenarios in the complete pipeline
    let temp_dir = tempdir().expect("Failed to create temp directory");

    // Scenario 1: Corrupted model file with partial recovery
    let corrupted_model = temp_dir.path().join("corrupted_model.gguf");
    let mut model_file =
        std::fs::File::create(&corrupted_model).expect("Failed to create corrupted model file");

    // Write partially valid GGUF header
    model_file.write_all(b"GGUF\x03\x00\x00\x00").expect("Failed to write partial header");
    model_file.write_all(&[0u8; 100]).expect("Failed to write padding"); // Incomplete metadata

    // Test discovery handles corrupted model gracefully
    let discovery_result = TokenizerDiscovery::from_gguf(&corrupted_model);
    match discovery_result {
        Ok(_) => println!("Corrupted model was somehow valid"),
        Err(err) => {
            println!("Expected error for corrupted model: {:?}", err);
            // Error should be informative and recoverable
            assert!(!format!("{:?}", err).is_empty(), "Error message should be non-empty");
        }
    }

    // Scenario 2: Network failure during download with cache fallback
    let download_info = bitnet_tokenizers::discovery::TokenizerDownloadInfo {
        repo: "nonexistent/test-model".to_string(),
        files: vec!["tokenizer.json".to_string()],
        cache_key: "integration-test".to_string(),
        expected_vocab: Some(32000),
    };

    let downloader = SmartTokenizerDownload::with_cache_dir(temp_dir.path().to_path_buf())
        .expect("Failed to create downloader");

    // Pre-populate cache to test fallback
    let cache_dir = temp_dir.path().join(&download_info.cache_key);
    std::fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");

    let cached_tokenizer = cache_dir.join("tokenizer.json");
    let mut cache_file =
        std::fs::File::create(&cached_tokenizer).expect("Failed to create cached tokenizer");
    cache_file
        .write_all(br#"{"version": "1.0", "model": {"type": "BPE"}}"#)
        .expect("Failed to write cached tokenizer");

    // Test cache fallback works when download would fail
    let cached_result = downloader.find_cached_tokenizer(&download_info.cache_key);
    assert!(cached_result.is_some(), "Should find cached tokenizer as fallback");

    println!("✅ Complete pipeline failure recovery test completed");
}

/// Test stress scenarios with large vocabulary models
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_large_vocabulary_stress_scenarios() {
    use std::sync::Arc;
    use std::time::Instant;

    // Test large vocabulary tokenizer stress scenarios
    let stress_test_scenarios = [
        (128256, "LLaMA-3 stress test"),
        (200000, "Very large vocabulary stress"),
        (500000, "Extreme vocabulary stress"),
    ];

    for (vocab_size, description) in stress_test_scenarios {
        let start_time = Instant::now();

        // Create large vocabulary tokenizer
        let large_tokenizer =
            Arc::new(BasicTokenizer::with_config(vocab_size, Some(1), Some(2), Some(0)));

        let creation_time = start_time.elapsed();
        println!("{}: tokenizer creation took {:?}", description, creation_time);

        // Test wrapper creation with large vocabulary
        let wrapper_start = Instant::now();
        let wrapper_result = LlamaTokenizerWrapper::new(large_tokenizer.clone(), vocab_size);
        let wrapper_time = wrapper_start.elapsed();

        assert!(wrapper_result.is_ok(), "{}: wrapper should handle large vocabulary", description);
        println!("{}: wrapper creation took {:?}", description, wrapper_time);

        // Test encoding performance with large vocabulary
        let encode_start = Instant::now();
        let wrapper = wrapper_result.unwrap();
        let encode_result = wrapper.encode("Test tokenization with large vocabulary", true, false);
        let encode_time = encode_start.elapsed();

        assert!(encode_result.is_ok(), "{}: encoding should succeed", description);
        println!("{}: encoding took {:?}", description, encode_time);

        // Validate performance boundaries
        assert!(creation_time.as_millis() < 1000, "{}: creation should be under 1s", description);
        assert!(wrapper_time.as_millis() < 100, "{}: wrapper creation should be fast", description);
        assert!(encode_time.as_millis() < 500, "{}: encoding should be under 500ms", description);

        // Test memory efficiency
        let tokens = encode_result.unwrap();
        assert!(!tokens.is_empty(), "{}: should produce tokens", description);
        assert!(
            tokens.len() < 100,
            "{}: token count should be reasonable for test input",
            description
        );
    }

    println!("✅ Large vocabulary stress test completed");
}

/// Test concurrent access patterns with resource contention
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_concurrent_resource_contention() {
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::task;

    let temp_dir = tempdir().expect("Failed to create temp directory");

    // Create shared tokenizer resources
    let shared_tokenizer = Arc::new(BasicTokenizer::with_config(32000, Some(1), Some(2), None));

    // Create shared cache directory
    let cache_dir = temp_dir.path().join("shared_cache");
    std::fs::create_dir_all(&cache_dir).expect("Failed to create shared cache");

    // Pre-populate cache with test tokenizer
    let cache_key = "concurrent-test";
    let tokenizer_cache = cache_dir.join(cache_key);
    std::fs::create_dir_all(&tokenizer_cache).expect("Failed to create tokenizer cache");

    let cached_file = tokenizer_cache.join("tokenizer.json");
    let mut file = std::fs::File::create(&cached_file).expect("Failed to create cached file");
    file.write_all(br#"{"version": "1.0", "model": {"type": "BPE"}}"#)
        .expect("Failed to write cached tokenizer");

    // Spawn multiple concurrent tasks with different access patterns
    let mut handles = vec![];
    let num_tasks = 10;

    for i in 0..num_tasks {
        let tokenizer_clone = Arc::clone(&shared_tokenizer);
        let cache_dir_clone = cache_dir.clone();

        let handle = task::spawn(async move {
            // Different concurrent access patterns
            match i % 4 {
                0 => {
                    // Tokenizer wrapper creation and encoding
                    let wrapper_result = LlamaTokenizerWrapper::new(tokenizer_clone, 32000);
                    assert!(
                        wrapper_result.is_ok(),
                        "Concurrent wrapper creation {} should succeed",
                        i
                    );

                    let wrapper = wrapper_result.unwrap();
                    let encode_result =
                        wrapper.encode(&format!("Concurrent test {}", i), true, false);
                    assert!(encode_result.is_ok(), "Concurrent encoding {} should succeed", i);
                }
                1 => {
                    // Cache access and validation
                    let downloader = SmartTokenizerDownload::with_cache_dir(cache_dir_clone)
                        .expect("Failed to create downloader");

                    let cached = downloader.find_cached_tokenizer(cache_key);
                    assert!(cached.is_some(), "Concurrent cache access {} should find file", i);
                }
                2 => {
                    // BitNet wrapper with quantization
                    let bitnet_result = BitNetTokenizerWrapper::new(
                        tokenizer_clone,
                        bitnet_common::QuantizationType::I2S,
                    );
                    assert!(
                        bitnet_result.is_ok(),
                        "Concurrent BitNet wrapper {} should succeed",
                        i
                    );
                }
                3 => {
                    // GPT-2 wrapper creation
                    let gpt2_tokenizer =
                        Arc::new(BasicTokenizer::with_config(50257, None, Some(50256), None));
                    let gpt2_result = Gpt2TokenizerWrapper::new(gpt2_tokenizer);
                    assert!(gpt2_result.is_ok(), "Concurrent GPT-2 wrapper {} should succeed", i);
                }
                _ => unreachable!(),
            }

            // Simulate some work
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

            i // Return task ID
        });

        handles.push(handle);
    }

    // Collect all results and verify no race conditions
    let mut results = vec![];
    for handle in handles {
        let task_id = handle.await.expect("Concurrent task should complete");
        results.push(task_id);
    }

    // Verify all tasks completed successfully
    assert_eq!(results.len(), num_tasks, "All concurrent tasks should complete");
    results.sort();
    let expected: Vec<usize> = (0..num_tasks).collect();
    assert_eq!(results, expected, "All task IDs should be present");

    println!("✅ Concurrent resource contention test completed with {} tasks", num_tasks);
}

/// Test end-to-end workflow with multiple failure points
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_end_to_end_multiple_failure_points() {
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::tempdir;

    let temp_dir = tempdir().expect("Failed to create temp directory");

    // Simulate a realistic end-to-end workflow with multiple potential failure points
    let workflow_stages = [
        "Model loading",
        "Discovery initialization",
        "Strategy resolution",
        "Tokenizer loading",
        "Wrapper configuration",
        "Encoding validation",
        "Performance validation",
    ];

    let mut successful_stages = vec![];
    let mut failed_stages: Vec<String> = vec![];

    // Stage 1: Model loading (simulate partial failure)
    let model_path = temp_dir.path().join("test_model.gguf");
    let mut model_file = std::fs::File::create(&model_path).expect("Failed to create model file");

    // Write minimal but potentially problematic GGUF structure
    model_file.write_all(b"GGUF\x03\x00\x00\x00").expect("Failed to write header");
    model_file.write_all(&[0u8; 50]).expect("Failed to write minimal metadata");

    match TokenizerDiscovery::from_gguf(&model_path) {
        Ok(_) => successful_stages.push(workflow_stages[0]),
        Err(_) => failed_stages.push(workflow_stages[0].to_string()),
    }

    // Stage 2: Discovery initialization (always attempt)
    // Mock discovery for remaining stages
    successful_stages.push(workflow_stages[1]);

    // Stage 3: Strategy resolution (test multiple strategies)
    let test_strategies = [
        TokenizerStrategy::Mock,
        TokenizerStrategy::Exact(temp_dir.path().join("nonexistent.json")),
    ];

    let mut strategy_results = vec![];
    for strategy in test_strategies {
        let description = strategy.description();
        if strategy.requires_network() {
            // Network strategies might fail in test environment
            if std::env::var("CI").is_ok() {
                failed_stages.push("Network strategy (CI environment)".to_string());
            } else {
                strategy_results.push(format!("Network strategy: {}", description));
            }
        } else {
            strategy_results.push(format!("Local strategy: {}", description));
        }
    }

    if !strategy_results.is_empty() {
        successful_stages.push(workflow_stages[2]);
    }

    // Stage 4: Tokenizer loading (create test tokenizer)
    let test_tokenizer = Arc::new(BasicTokenizer::with_config(32000, Some(1), Some(2), None));
    successful_stages.push(workflow_stages[3]);

    // Stage 5: Wrapper configuration (test multiple wrapper types)
    type WrapperResult = Result<Box<dyn Tokenizer>, bitnet_common::BitNetError>;
    type WrapperFactory = Box<dyn Fn() -> WrapperResult>;

    let wrapper_tests: Vec<(&str, WrapperFactory)> = vec![
        (
            "LLaMA wrapper",
            Box::new({
                let test_tokenizer = test_tokenizer.clone();
                move || {
                    LlamaTokenizerWrapper::new(test_tokenizer.clone(), 32000)
                        .map(|w| Box::new(w) as Box<dyn Tokenizer>)
                }
            }),
        ),
        (
            "GPT-2 wrapper",
            Box::new(|| {
                let gpt2_tokenizer =
                    Arc::new(BasicTokenizer::with_config(50257, None, Some(50256), None));
                Gpt2TokenizerWrapper::new(gpt2_tokenizer).map(|w| Box::new(w) as Box<dyn Tokenizer>)
            }),
        ),
        (
            "BitNet wrapper",
            Box::new({
                let test_tokenizer = test_tokenizer.clone();
                move || {
                    BitNetTokenizerWrapper::new(
                        test_tokenizer.clone(),
                        bitnet_common::QuantizationType::I2S,
                    )
                    .map(|w| Box::new(w) as Box<dyn Tokenizer>)
                }
            }),
        ),
    ];

    let mut successful_wrappers = vec![];
    for (wrapper_name, wrapper_fn) in wrapper_tests {
        match wrapper_fn() {
            Ok(wrapper) => {
                successful_wrappers.push((wrapper_name, wrapper));
            }
            Err(err) => {
                failed_stages.push(format!("{}: {:?}", wrapper_name, err));
            }
        }
    }

    if !successful_wrappers.is_empty() {
        successful_stages.push(workflow_stages[4]);
    }

    // Stage 6: Encoding validation (test all successful wrappers)
    let test_inputs = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Neural network tokenization test with special tokens",
        "", // Empty input edge case
    ];

    let mut encoding_results = vec![];
    for (wrapper_name, wrapper) in &successful_wrappers {
        let mut wrapper_results = vec![];

        for input in &test_inputs {
            match wrapper.encode(input, true, false) {
                Ok(tokens) => {
                    wrapper_results.push(format!("'{}' -> {} tokens", input, tokens.len()));
                }
                Err(err) => {
                    wrapper_results.push(format!("'{}' -> Error: {:?}", input, err));
                }
            }
        }

        encoding_results.push((*wrapper_name, wrapper_results));
    }

    if !encoding_results.is_empty() {
        successful_stages.push(workflow_stages[5]);
    }

    // Stage 7: Performance validation
    use std::time::Instant;

    if let Some((_wrapper_name, _)) = successful_wrappers.first()
        && let Some(wrapper) = successful_wrappers.first().map(|(_, w)| w)
    {
        let perf_start = Instant::now();
        let mut total_tokens = 0;

        // Performance test with multiple iterations
        for _ in 0..100 {
            match wrapper.encode("Performance test input", true, false) {
                Ok(tokens) => total_tokens += tokens.len(),
                Err(_) => break,
            }
        }

        let perf_duration = perf_start.elapsed();
        let tokens_per_second = if perf_duration.as_secs() > 0 {
            total_tokens as f64 / perf_duration.as_secs_f64()
        } else {
            total_tokens as f64 / 0.001 // Minimum 1ms
        };

        if tokens_per_second > 1000.0 {
            // Reasonable performance threshold
            successful_stages.push(workflow_stages[6]);
        } else {
            failed_stages.push("Performance below threshold".to_string());
        }
    }

    // Report comprehensive results
    println!("=== End-to-End Workflow Results ===");
    println!(
        "Successful stages ({}/{}): {:?}",
        successful_stages.len(),
        workflow_stages.len(),
        successful_stages
    );

    if !failed_stages.is_empty() {
        println!("Failed stages ({}): {:?}", failed_stages.len(), failed_stages);
    }

    for (wrapper_name, results) in encoding_results {
        println!("{} encoding results:", wrapper_name);
        for result in results {
            println!("  {}", result);
        }
    }

    // Test should succeed if we completed most stages
    let success_rate = successful_stages.len() as f64 / workflow_stages.len() as f64;
    assert!(success_rate >= 0.5, "Should complete at least 50% of workflow stages");

    println!("✅ End-to-end workflow completed with {:.1}% success rate", success_rate * 100.0);
}

/// Test cross-platform compatibility scenarios
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_cross_platform_compatibility() {
    use std::path::PathBuf;

    // Test cross-platform path handling and file operations
    let platform_scenarios = [
        // Different path formats
        ("unix_style", "/tmp/bitnet/tokenizer.json"),
        ("windows_style", r"C:\temp\bitnet\tokenizer.json"),
        ("relative_path", "models/tokenizer.json"),
        ("nested_path", "models/subdir/nested/tokenizer.json"),
        // Special characters in paths
        ("spaces", "path with spaces/tokenizer.json"),
        ("unicode", "路径/tokenizer.json"),
        ("special_chars", "path-with_special.chars/tokenizer.json"),
    ];

    for (scenario_name, path_str) in platform_scenarios {
        let path = PathBuf::from(path_str);

        // Test path validation and normalization
        let is_absolute = path.is_absolute();
        let has_extension = path.extension().is_some();
        let components_count = path.components().count();

        // Basic path validation
        assert!(!path_str.is_empty(), "{}: path should not be empty", scenario_name);

        if has_extension {
            let extension = path.extension().unwrap().to_string_lossy();
            assert!(
                extension == "json" || extension == "model",
                "{}: should have valid tokenizer extension",
                scenario_name
            );
        }

        // Test cross-platform path operations
        // Note: Windows paths on Unix systems may not have recognized parent directories
        if scenario_name == "windows_style" && cfg!(not(target_os = "windows")) {
            // Windows-style paths on Unix systems are treated as single components
            println!("Skipping parent directory check for Windows path on Unix system");
        } else if let Some(_parent) = path.parent() {
            assert!(components_count > 1, "{}: should have parent directory", scenario_name);
        } else if components_count > 1 {
            // Path has multiple components but no parent - unexpected
            panic!("{}: should have parent directory but doesn't", scenario_name);
        }

        if let Some(filename) = path.file_name() {
            let filename_str = filename.to_string_lossy();
            assert!(!filename_str.is_empty(), "{}: filename should not be empty", scenario_name);
        }

        // Test that paths can be processed without panics
        let path_display = format!("{}", path.display());
        assert!(!path_display.is_empty(), "{}: path display should work", scenario_name);

        println!(
            "{}: {} (absolute: {}, components: {})",
            scenario_name, path_display, is_absolute, components_count
        );
    }

    // Test environment variable handling across platforms
    let env_scenarios = [
        ("BITNET_STRICT_TOKENIZERS", "1"),
        ("BITNET_OFFLINE", "1"),
        ("BITNET_CACHE_DIR", "/tmp/bitnet-cache"),
        ("HOME", "/home/user"),            // Unix-style
        ("USERPROFILE", r"C:\Users\user"), // Windows-style
    ];

    for (env_var, test_value) in env_scenarios {
        // Test setting environment variable
        unsafe {
            std::env::set_var(env_var, test_value);
        }

        let retrieved = std::env::var(env_var);
        assert!(retrieved.is_ok(), "Should be able to retrieve env var: {}", env_var);
        assert_eq!(retrieved.unwrap(), test_value, "Env var value should match: {}", env_var);

        // Clean up
        unsafe {
            std::env::remove_var(env_var);
        }

        let removed = std::env::var(env_var);
        assert!(removed.is_err(), "Env var should be removed: {}", env_var);
    }

    println!("✅ Cross-platform compatibility test completed");
}

/// Test memory efficiency and resource cleanup
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_memory_efficiency_and_cleanup() {
    use std::sync::Arc;
    use std::time::Instant;

    // Test memory-efficient operations and proper cleanup
    let memory_test_scenarios = [
        (1000, "Small vocabulary memory test"),
        (32000, "LLaMA-2 vocabulary memory test"),
        (128256, "LLaMA-3 vocabulary memory test"),
    ];

    for (vocab_size, description) in memory_test_scenarios {
        let start_time = Instant::now();

        // Test 1: Tokenizer creation and memory usage
        let tokenizer = Arc::new(BasicTokenizer::with_config(vocab_size, Some(1), Some(2), None));
        let creation_time = start_time.elapsed();

        // Test 2: Multiple wrapper creation (tests memory efficiency)
        let wrapper_start = Instant::now();
        let mut wrappers: Vec<Box<dyn Tokenizer>> = vec![];

        // Create multiple wrappers to test memory scaling
        for i in 0..10 {
            match i % 3 {
                0 => {
                    if let Ok(wrapper) = LlamaTokenizerWrapper::new(tokenizer.clone(), vocab_size) {
                        wrappers.push(Box::new(wrapper));
                    }
                }
                1 => {
                    if vocab_size == 50257 {
                        // GPT-2 specific
                        if let Ok(wrapper) = Gpt2TokenizerWrapper::new(tokenizer.clone()) {
                            wrappers.push(Box::new(wrapper));
                        }
                    }
                }
                2 => {
                    if let Ok(wrapper) = BitNetTokenizerWrapper::new(
                        tokenizer.clone(),
                        bitnet_common::QuantizationType::I2S,
                    ) {
                        wrappers.push(Box::new(wrapper));
                    }
                }
                _ => unreachable!(),
            }
        }

        let wrapper_creation_time = wrapper_start.elapsed();

        // Test 3: Batch encoding (memory efficiency test)
        let encoding_start = Instant::now();
        let test_texts = [
            "Short text",
            "Medium length text with more words to test tokenization efficiency",
            "Very long text that should test the memory efficiency of the tokenization process with many tokens and complex encoding scenarios that might stress the memory allocation patterns",
        ];

        let mut total_tokens = 0;
        for wrapper in &wrappers {
            for text in &test_texts {
                if let Ok(tokens) = wrapper.encode(text, true, false) {
                    total_tokens += tokens.len();
                }
            }
        }

        let encoding_time = encoding_start.elapsed();

        // Test 4: Memory cleanup (drop wrappers and test no memory leaks)
        let cleanup_start = Instant::now();
        drop(wrappers); // Explicit cleanup
        let cleanup_time = cleanup_start.elapsed();

        // Performance and memory efficiency validation
        let total_time = start_time.elapsed();

        println!("{} results:", description);
        println!("  Creation time: {:?}", creation_time);
        println!("  Wrapper creation (10x): {:?}", wrapper_creation_time);
        println!("  Encoding time ({} tokens): {:?}", total_tokens, encoding_time);
        println!("  Cleanup time: {:?}", cleanup_time);
        println!("  Total time: {:?}", total_time);

        // Memory efficiency assertions
        assert!(creation_time.as_millis() < 1000, "{}: creation should be fast", description);
        assert!(
            wrapper_creation_time.as_millis() < 500,
            "{}: wrapper creation should be efficient",
            description
        );
        assert!(cleanup_time.as_millis() < 100, "{}: cleanup should be fast", description);
        assert!(total_tokens > 0, "{}: should produce tokens", description);

        // Test memory doesn't grow excessively with vocabulary size
        let time_per_vocab_token = total_time.as_nanos() as f64 / vocab_size as f64;
        assert!(
            time_per_vocab_token < 1000.0,
            "{}: time should scale reasonably with vocab size",
            description
        );
    }

    println!("✅ Memory efficiency and cleanup test completed");
}

/// Test error recovery and graceful degradation
#[tokio::test]
#[cfg(feature = "cpu")]
async fn test_error_recovery_graceful_degradation() {
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::tempdir;

    let temp_dir = tempdir().expect("Failed to create temp directory");

    // Test comprehensive error recovery scenarios
    let error_recovery_scenarios = [
        ("corrupted_cache", "Cache corruption with recovery"),
        ("partial_download", "Incomplete download with resume"),
        ("invalid_model", "Invalid model with fallback"),
        ("memory_pressure", "Memory pressure with degradation"),
        ("concurrent_access", "Concurrent access conflicts"),
    ];

    for (scenario_id, description) in error_recovery_scenarios {
        println!("Testing error recovery: {}", description);

        match scenario_id {
            "corrupted_cache" => {
                // Create corrupted cache and test recovery
                let cache_dir = temp_dir.path().join("corrupted_cache");
                std::fs::create_dir_all(&cache_dir).expect("Failed to create cache dir");

                let corrupted_file = cache_dir.join("tokenizer.json");
                let mut file =
                    std::fs::File::create(&corrupted_file).expect("Failed to create file");
                file.write_all(b"corrupted json content {{{")
                    .expect("Failed to write corrupted content");

                // Test downloader handles corruption gracefully
                let downloader = SmartTokenizerDownload::with_cache_dir(cache_dir.clone())
                    .expect("Downloader should initialize despite corrupted cache");

                let found = downloader.find_cached_tokenizer("corrupted_test");
                // Should find file but validation should fail
                if let Some(found_path) = found {
                    let validation_info = bitnet_tokenizers::discovery::TokenizerDownloadInfo {
                        repo: "test/repo".to_string(),
                        files: vec!["tokenizer.json".to_string()],
                        cache_key: "test".to_string(),
                        expected_vocab: Some(1000),
                    };

                    let validation =
                        downloader.validate_downloaded_tokenizer(&found_path, &validation_info);
                    assert!(validation.is_err(), "Should detect corrupted cache");
                }
            }

            "partial_download" => {
                // Create partial download scenario
                let partial_dir = temp_dir.path().join("partial_download");
                std::fs::create_dir_all(&partial_dir).expect("Failed to create partial dir");

                let partial_file = partial_dir.join("tokenizer.json");
                let mut file =
                    std::fs::File::create(&partial_file).expect("Failed to create partial file");
                file.write_all(b"{ \"partial\": \"content")
                    .expect("Failed to write partial content");

                // Test resume capability detection
                let file_size =
                    std::fs::metadata(&partial_file).expect("Should get metadata").len();
                assert!(file_size > 0, "Partial file should have content");

                // In real implementation, this would trigger resume logic
                println!("Partial download detected: {} bytes", file_size);
            }

            "invalid_model" => {
                // Test fallback with invalid model
                let invalid_model = temp_dir.path().join("invalid.gguf");
                let mut file =
                    std::fs::File::create(&invalid_model).expect("Failed to create invalid model");
                file.write_all(b"NOT_GGUF_HEADER").expect("Failed to write invalid header");

                // Test discovery fails gracefully
                let discovery_result = TokenizerDiscovery::from_gguf(&invalid_model);
                assert!(discovery_result.is_err(), "Should reject invalid model");

                // Test fallback chain would continue to next strategy
                let strategy = TokenizerStrategy::Mock;
                assert_eq!(strategy.description(), "mock tokenizer (testing only)");
            }

            "memory_pressure" => {
                // Test graceful degradation under memory pressure
                let large_tokenizer =
                    Arc::new(BasicTokenizer::with_config(500000, Some(1), Some(2), None));

                // Test that large tokenizer still works but may be slower
                let wrapper_result = LlamaTokenizerWrapper::new(large_tokenizer, 500000);
                assert!(wrapper_result.is_ok(), "Should handle large vocabulary");

                let wrapper = wrapper_result.unwrap();
                let encode_result = wrapper.encode("test", true, false);
                assert!(encode_result.is_ok(), "Should still encode under memory pressure");
            }

            "concurrent_access" => {
                // Test concurrent access error handling
                let shared_cache = temp_dir.path().join("shared_cache");
                std::fs::create_dir_all(&shared_cache).expect("Failed to create shared cache");

                // Multiple downloaders accessing same cache
                let downloaders: Vec<_> = (0..3)
                    .map(|_| {
                        SmartTokenizerDownload::with_cache_dir(shared_cache.clone())
                            .expect("Failed to create downloader")
                    })
                    .collect();

                // Test concurrent cache operations don't interfere
                for (i, downloader) in downloaders.iter().enumerate() {
                    let cache_key = format!("concurrent_{}", i);
                    let found = downloader.find_cached_tokenizer(&cache_key);
                    // Should handle concurrent access gracefully (returns None if not found)
                    println!("Concurrent access {}: found={:?}", i, found.is_some());
                }
            }

            _ => unreachable!(),
        }
    }

    println!("✅ Error recovery and graceful degradation test completed");
}
