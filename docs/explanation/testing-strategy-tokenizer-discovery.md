# Testing Strategy: Tokenizer Discovery and Cross-Validation Framework

## Executive Summary

This document defines a comprehensive testing strategy for BitNet.rs tokenizer discovery system, ensuring production-grade reliability through systematic unit testing, integration testing, cross-validation with C++ reference implementations, and performance regression testing. The strategy emphasizes neural network model compatibility, quantization accuracy, and device-aware optimization validation.

## Testing Architecture Overview

### Test Organization Structure
```
tests/
├── unit/                          # Unit tests for individual components
│   ├── discovery/                 # TokenizerDiscovery tests
│   ├── download/                  # SmartTokenizerDownload tests
│   ├── strategy/                  # TokenizerStrategy resolution tests
│   └── wrappers/                  # Neural network model wrapper tests
├── integration/                   # End-to-end integration tests
│   ├── real_models/               # Tests with actual model files
│   ├── network_simulation/        # Network condition simulation
│   └── device_selection/          # GPU/CPU selection tests
├── crossval/                      # Cross-validation with C++ reference
│   ├── tokenization_parity/       # Tokenization accuracy tests
│   ├── performance_parity/        # Performance comparison tests
│   └── quantization_compat/       # Quantization compatibility tests
├── performance/                   # Performance regression tests
│   ├── benchmarks/                # Standardized benchmarks
│   ├── memory_profiling/          # Memory usage validation
│   └── scaling/                   # Concurrent usage tests
└── e2e/                          # Complete workflow validation
    ├── xtask_integration/         # CLI integration tests
    ├── production_scenarios/      # Production use cases
    └── error_recovery/            # Error handling validation
```

## Unit Testing Strategy

### 1. TokenizerDiscovery Unit Tests

```rust
// tests/unit/discovery/test_gguf_parsing.rs
#[cfg(test)]
mod gguf_parsing_tests {
    use super::*;
    use bitnet_tokenizers::discovery::TokenizerDiscovery;
    use tempfile::NamedTempFile;

    #[test]
    fn test_llama3_vocab_size_extraction() -> Result<()> {
        // AC1: TokenizerDiscovery for GGUF metadata parsing
        let test_gguf = create_test_gguf_with_metadata(
            128256,      // LLaMA-3 vocab size
            "llama",     // Model architecture
            Some("I2S"), // Quantization type
        )?;

        let discovery = TokenizerDiscovery::from_gguf(&test_gguf)?;

        assert_eq!(discovery.vocab_size(), 128256);
        assert_eq!(discovery.model_type(), "llama");
        assert!(discovery.requires_large_vocab_optimization());

        Ok(())
    }

    #[test]
    fn test_gpt2_metadata_extraction() -> Result<()> {
        let test_gguf = create_test_gguf_with_metadata(50257, "gpt2", None)?;
        let discovery = TokenizerDiscovery::from_gguf(&test_gguf)?;

        assert_eq!(discovery.vocab_size(), 50257);
        assert_eq!(discovery.model_type(), "gpt2");
        assert!(!discovery.requires_large_vocab_optimization());

        Ok(())
    }

    #[test]
    fn test_strategy_discovery_colocated_files() -> Result<()> {
        let (model_path, temp_dir) = create_test_model_with_colocated_tokenizer()?;
        let discovery = TokenizerDiscovery::from_gguf(&model_path)?;

        let strategy = discovery.discover_tokenizer_strategy()?;

        match strategy {
            TokenizerStrategy::Discovered(path) => {
                assert!(path.file_name().unwrap() == "tokenizer.json");
                assert!(path.exists());
            }
            _ => panic!("Expected Discovered strategy, got: {:?}", strategy),
        }

        Ok(())
    }

    #[test]
    fn test_download_strategy_inference() -> Result<()> {
        // Test neural network model compatibility matrix
        let test_cases = vec![
            (128256, "llama", "meta-llama/Meta-Llama-3-8B"),
            (32000, "llama", "meta-llama/Llama-2-7b-hf"),
            (50257, "gpt2", "openai-community/gpt2"),
        ];

        for (vocab_size, model_type, expected_repo) in test_cases {
            let test_gguf = create_test_gguf_with_metadata(vocab_size, model_type, None)?;
            let discovery = TokenizerDiscovery::from_gguf(&test_gguf)?;

            let strategy = discovery.discover_tokenizer_strategy()?;

            if let TokenizerStrategy::NeedsDownload(info) = strategy {
                assert_eq!(info.repo, expected_repo);
                assert!(info.files.contains(&"tokenizer.json".to_string()));
            }
        }

        Ok(())
    }

    #[test]
    fn test_strict_mode_enforcement() -> Result<()> {
        // Test AC10: Proper error handling with anyhow::Result
        temp_env::with_var("BITNET_STRICT_TOKENIZERS", Some("1"), || {
            let test_gguf = create_test_gguf_minimal()?; // No tokenizer info
            let discovery = TokenizerDiscovery::from_gguf(&test_gguf)?;

            let result = discovery.discover_tokenizer_strategy();

            assert!(result.is_err());
            let error_msg = result.unwrap_err().to_string();
            assert!(error_msg.contains("strict mode"));

            Ok::<(), anyhow::Error>(())
        })?;

        Ok(())
    }
}
```

### 2. SmartTokenizerDownload Unit Tests

```rust
// tests/unit/download/test_smart_download.rs
#[cfg(test)]
mod smart_download_tests {
    use super::*;
    use bitnet_tokenizers::downloader::SmartTokenizerDownload;
    use mockito::{mock, Mock, Server};
    use tokio;

    #[tokio::test]
    async fn test_successful_tokenizer_download() -> Result<()> {
        // AC2: SmartTokenizerDownload for missing tokenizer files
        let mut server = Server::new_async().await;
        let mock_tokenizer_content = r#"{"version": "1.0", "truncation": null}"#;

        let _mock = server.mock("GET", "/meta-llama/Llama-2-7b-hf/resolve/main/tokenizer.json")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(mock_tokenizer_content)
            .create_async().await;

        let downloader = SmartTokenizerDownload::new_with_base_url(&server.url())?;
        let download_info = TokenizerDownloadInfo {
            repo: "meta-llama/Llama-2-7b-hf".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "test-llama2".to_string(),
            expected_vocab: Some(32000),
        };

        let result_path = downloader.download_tokenizer(&download_info).await?;

        assert!(result_path.exists());
        assert_eq!(result_path.file_name().unwrap(), "tokenizer.json");

        // Verify cached for subsequent requests
        let cached_path = downloader.find_cached_tokenizer("test-llama2");
        assert!(cached_path.is_some());
        assert_eq!(cached_path.unwrap(), result_path);

        Ok(())
    }

    #[tokio::test]
    async fn test_download_with_resume() -> Result<()> {
        let mut server = Server::new_async().await;
        let full_content = "x".repeat(1024 * 100); // 100KB test content

        // Simulate partial download
        let _partial_mock = server.mock("GET", "/test-repo/resolve/main/tokenizer.json")
            .with_status(206) // Partial content
            .with_header("content-range", "bytes 0-49999/100000")
            .with_body(&full_content[0..50000])
            .create_async().await;

        let _complete_mock = server.mock("GET", "/test-repo/resolve/main/tokenizer.json")
            .with_header("range", "bytes=50000-")
            .with_status(206)
            .with_header("content-range", "bytes 50000-99999/100000")
            .with_body(&full_content[50000..])
            .create_async().await;

        let downloader = SmartTokenizerDownload::new_with_base_url(&server.url())?;
        let download_info = TokenizerDownloadInfo {
            repo: "test-repo".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "test-resume".to_string(),
            expected_vocab: None,
        };

        let result_path = downloader.download_tokenizer(&download_info).await?;

        let downloaded_content = std::fs::read_to_string(&result_path)?;
        assert_eq!(downloaded_content.len(), full_content.len());

        Ok(())
    }

    #[tokio::test]
    async fn test_network_error_handling() -> Result<()> {
        let mut server = Server::new_async().await;
        let _mock = server.mock("GET", "/nonexistent/resolve/main/tokenizer.json")
            .with_status(404)
            .create_async().await;

        let downloader = SmartTokenizerDownload::new_with_base_url(&server.url())?;
        let download_info = TokenizerDownloadInfo {
            repo: "nonexistent".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "test-404".to_string(),
            expected_vocab: None,
        };

        let result = downloader.download_tokenizer(&download_info).await;

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("HTTP 404"));

        Ok(())
    }
}
```

### 3. Neural Network Model Wrapper Tests

```rust
// tests/unit/wrappers/test_neural_network_wrappers.rs
#[cfg(test)]
mod neural_network_wrapper_tests {
    use super::*;
    use bitnet_tokenizers::strategy::{LlamaTokenizerWrapper, Gpt2TokenizerWrapper};

    #[test]
    fn test_llama_tokenizer_wrapper_special_tokens() -> Result<()> {
        // AC3: Production-ready TokenizerStrategy implementations
        let base_tokenizer = create_mock_hf_tokenizer(32000)?;
        let wrapper = LlamaTokenizerWrapper::new(base_tokenizer, 32000, LlamaVariant::Llama2)?;

        // Test BOS token insertion
        let tokens = wrapper.encode("Hello world", true, false)?;
        assert_eq!(tokens[0], 1); // LLaMA BOS token
        assert!(tokens.len() > 1);

        // Test vocab size validation
        let invalid_tokens = vec![32000, 32001]; // Beyond vocab size
        let decode_result = wrapper.decode(&invalid_tokens);
        assert!(decode_result.is_err());

        // Test special token IDs
        assert_eq!(wrapper.bos_token_id(), Some(1));
        assert_eq!(wrapper.eos_token_id(), Some(2));

        Ok(())
    }

    #[test]
    fn test_gpt2_tokenizer_wrapper_behavior() -> Result<()> {
        let base_tokenizer = create_mock_hf_tokenizer(50257)?;
        let wrapper = Gpt2TokenizerWrapper::new(base_tokenizer)?;

        // GPT-2 should not add BOS token
        let tokens = wrapper.encode("Hello world", true, false)?;
        assert_ne!(tokens[0], 1); // No BOS token for GPT-2

        // Test EOS token handling
        assert_eq!(wrapper.eos_token_id(), Some(50256));
        assert_eq!(wrapper.bos_token_id(), None);

        Ok(())
    }

    #[test]
    fn test_large_vocabulary_optimization() -> Result<()> {
        // Test LLaMA-3 with 128K vocabulary
        let base_tokenizer = create_mock_hf_tokenizer(128256)?;
        let wrapper = LlamaTokenizerWrapper::new(base_tokenizer, 128256, LlamaVariant::Llama3)?;

        // Test that large vocabulary doesn't impact encoding performance significantly
        let large_text = "This is a test of large vocabulary tokenization ".repeat(100);

        let start = std::time::Instant::now();
        let tokens = wrapper.encode(&large_text, true, false)?;
        let encode_duration = start.elapsed();

        assert!(encode_duration.as_millis() < 50); // Should be under 50ms
        assert!(tokens.len() > 100);

        // All tokens should be within vocab range
        for &token in &tokens {
            assert!((token as usize) < 128256, "Token {} exceeds vocab size", token);
        }

        Ok(())
    }
}
```

## Integration Testing Strategy

### 1. Real Model Integration Tests

```rust
// tests/integration/real_models/test_model_integration.rs
#[cfg(test)]
mod real_model_integration {
    use super::*;
    use bitnet_tokenizers::discovery::TokenizerDiscovery;
    use bitnet_tokenizers::strategy::TokenizerStrategyResolver;

    #[tokio::test]
    async fn test_llama2_real_model_integration() -> Result<()> {
        // AC7: Integration tests with real model files
        let model_path = download_test_model("microsoft/bitnet-b1.58-2B-4T-gguf", "ggml-model-i2_s.gguf").await?;

        let discovery = TokenizerDiscovery::from_gguf(&model_path)?;
        assert!(discovery.vocab_size() > 0);
        assert!(!discovery.model_type().is_empty());

        let strategy = discovery.discover_tokenizer_strategy()?;
        let resolver = TokenizerStrategyResolver::new(discovery).await?;
        let tokenizer = resolver.resolve_tokenizer(strategy).await?;

        // Test with real text
        let test_texts = vec![
            "The capital of France is",
            "Neural networks are",
            "BitNet quantization enables",
        ];

        for text in test_texts {
            let tokens = tokenizer.encode(text, true, true)?;
            assert!(!tokens.is_empty());

            let decoded = tokenizer.decode(&tokens)?;
            // For real tokenizers, decoded text should be meaningful
            assert!(!decoded.contains("Generated text from"));
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_end_to_end_xtask_integration() -> Result<()> {
        // AC4: Integration with cargo xtask infer command
        let model_path = get_or_download_test_model().await?;

        let output = tokio::process::Command::new("cargo")
            .args(&[
                "run", "-p", "xtask", "--",
                "infer",
                "--model", model_path.to_str().unwrap(),
                "--prompt", "Test inference with automatic tokenizer discovery",
                "--max-new-tokens", "10",
                "--auto-download",
                "--deterministic"
            ])
            .env("BITNET_DETERMINISTIC", "1")
            .env("BITNET_SEED", "42")
            .output()
            .await?;

        assert!(output.status.success(), "xtask infer failed: {}",
            String::from_utf8_lossy(&output.stderr));

        let stdout = String::from_utf8(output.stdout)?;
        let stderr = String::from_utf8(output.stderr)?;

        // Verify automatic tokenizer discovery worked
        assert!(
            stderr.contains("Auto-discovering tokenizer") ||
            stderr.contains("Loading discovered tokenizer") ||
            stderr.contains("Using cached tokenizer")
        );

        // Verify real inference output (not mock)
        assert!(!stdout.contains("Generated text from"));
        assert!(!stdout.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_cpu_tokenizer_consistency() -> Result<()> {
        // Test device-aware tokenization consistency
        let model_path = get_or_download_test_model().await?;

        // Run with CPU features
        let cpu_output = tokio::process::Command::new("cargo")
            .args(&[
                "run", "-p", "xtask", "--no-default-features", "--features", "cpu", "--",
                "infer",
                "--model", model_path.to_str().unwrap(),
                "--prompt", "Consistent tokenization test",
                "--deterministic"
            ])
            .env("BITNET_DETERMINISTIC", "1")
            .env("BITNET_SEED", "42")
            .output()
            .await?;

        // Run with GPU features (if available)
        let gpu_output = tokio::process::Command::new("cargo")
            .args(&[
                "run", "-p", "xtask", "--no-default-features", "--features", "gpu", "--",
                "infer",
                "--model", model_path.to_str().unwrap(),
                "--prompt", "Consistent tokenization test",
                "--deterministic"
            ])
            .env("BITNET_DETERMINISTIC", "1")
            .env("BITNET_SEED", "42")
            .output()
            .await;

        if cpu_output.status.success() {
            let cpu_result = String::from_utf8(cpu_output.stdout)?;

            if let Ok(gpu_output) = gpu_output {
                if gpu_output.status.success() {
                    let gpu_result = String::from_utf8(gpu_output.stdout)?;

                    // With deterministic settings, tokenization should be identical
                    assert_eq!(cpu_result.trim(), gpu_result.trim(),
                        "CPU and GPU tokenization results differ");
                }
            }
        }

        Ok(())
    }
}
```

### 2. Network Simulation Tests

```rust
// tests/integration/network_simulation/test_network_conditions.rs
#[cfg(test)]
mod network_simulation_tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_slow_network_conditions() -> Result<()> {
        // Simulate slow network for tokenizer downloads
        let mut server = mockito::Server::new_async().await;
        let tokenizer_content = create_mock_tokenizer_json();

        let _mock = server.mock("GET", "/slow-repo/resolve/main/tokenizer.json")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body_from_fn(move |_| {
                // Simulate slow response
                std::thread::sleep(Duration::from_millis(100));
                tokenizer_content.clone()
            })
            .create_async().await;

        let downloader = SmartTokenizerDownload::new_with_base_url(&server.url())?;
        let download_info = TokenizerDownloadInfo {
            repo: "slow-repo".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "test-slow".to_string(),
            expected_vocab: None,
        };

        let start = std::time::Instant::now();
        let result = downloader.download_tokenizer(&download_info).await?;
        let duration = start.elapsed();

        assert!(result.exists());
        assert!(duration.as_millis() >= 100); // Should respect slow response
        assert!(duration.as_millis() < 10000); // But complete within reasonable time

        Ok(())
    }

    #[tokio::test]
    async fn test_offline_mode_behavior() -> Result<()> {
        // AC5: Fallback strategy system
        temp_env::with_var("BITNET_OFFLINE", Some("1"), || async {
            let discovery = create_test_discovery_without_local_tokenizer()?;
            let strategy = discovery.discover_tokenizer_strategy()?;

            // In offline mode, should not attempt downloads
            match strategy {
                TokenizerStrategy::NeedsDownload(_) => {
                    panic!("Should not attempt download in offline mode");
                }
                TokenizerStrategy::Mock => {
                    // Acceptable fallback in offline mode
                }
                _ => {} // Other strategies are fine
            }

            Ok::<(), anyhow::Error>(())
        }).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_network_interruption_recovery() -> Result<()> {
        let mut server = mockito::Server::new_async().await;
        let tokenizer_content = create_mock_tokenizer_json();

        // First request fails
        let _fail_mock = server.mock("GET", "/unstable-repo/resolve/main/tokenizer.json")
            .with_status(500)
            .expect(1)
            .create_async().await;

        // Second request succeeds
        let _success_mock = server.mock("GET", "/unstable-repo/resolve/main/tokenizer.json")
            .with_status(200)
            .with_body(&tokenizer_content)
            .expect(1)
            .create_async().await;

        let downloader = SmartTokenizerDownload::new_with_base_url(&server.url())?
            .with_retry_config(3, Duration::from_millis(100))?;

        let download_info = TokenizerDownloadInfo {
            repo: "unstable-repo".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "test-retry".to_string(),
            expected_vocab: None,
        };

        // Should succeed after retry
        let result = downloader.download_tokenizer(&download_info).await?;
        assert!(result.exists());

        Ok(())
    }
}
```

## Cross-Validation Framework

### 1. C++ Reference Implementation Parity

```rust
// tests/crossval/tokenization_parity/test_cpp_parity.rs
#[cfg(test)]
mod cpp_parity_tests {
    use super::*;
    use bitnet_tokenizers::crossval::CppReferenceValidator;

    #[test]
    fn test_llama_tokenization_parity_with_cpp() -> Result<()> {
        // AC6: Cross-validation tests with universal tokenizer
        let model_path = std::env::var("BITNET_GGUF")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("models/test/llama-2-7b.gguf"));

        if !model_path.exists() {
            return Ok(()); // Skip if no reference model available
        }

        // Load tokenizer via discovery system
        let discovery = TokenizerDiscovery::from_gguf(&model_path)?;
        let strategy = discovery.discover_tokenizer_strategy()?;
        let resolver = TokenizerStrategyResolver::new(discovery).await?;
        let rust_tokenizer = resolver.resolve_tokenizer(strategy).await?;

        // Initialize C++ reference validator
        let cpp_validator = CppReferenceValidator::new(&model_path)?;

        let test_cases = vec![
            "The quick brown fox jumps over the lazy dog",
            "Neural network quantization with BitNet enables efficient inference",
            "LLaMA-2 uses a vocabulary of 32,000 tokens for encoding text",
            "Special tokens like <s> and </s> are used for sequence boundaries",
            "",  // Empty string edge case
            "单词", // Unicode characters
        ];

        for text in test_cases {
            // Tokenize with Rust implementation
            let rust_tokens = rust_tokenizer.encode(text, true, true)?;

            // Tokenize with C++ reference
            let cpp_tokens = cpp_validator.tokenize(text, true)?;

            // Compare results
            assert_token_arrays_equivalent(&rust_tokens, &cpp_tokens, text)?;

            // Test decoding parity
            let rust_decoded = rust_tokenizer.decode(&rust_tokens)?;
            let cpp_decoded = cpp_validator.decode(&cpp_tokens)?;

            assert_text_similarity(&rust_decoded, &cpp_decoded, 0.95)?;
        }

        Ok(())
    }

    #[test]
    fn test_quantization_compatibility_cross_validation() -> Result<()> {
        // Test tokenizer compatibility with different quantization formats
        let model_path = get_test_model_path()?;
        let discovery = TokenizerDiscovery::from_gguf(&model_path)?;
        let strategy = discovery.discover_tokenizer_strategy()?;
        let resolver = TokenizerStrategyResolver::new(discovery).await?;
        let tokenizer = resolver.resolve_tokenizer(strategy).await?;

        let quantization_types = vec![
            QuantizationType::I2S,
            QuantizationType::TL1,
            QuantizationType::TL2,
            QuantizationType::IQ2S,
        ];

        for quant_type in quantization_types {
            let test_text = "Test quantization compatibility with tokenizer";
            let tokens = tokenizer.encode(test_text, true, true)?;

            // Validate all tokens are within quantization-safe ranges
            validate_tokens_for_quantization(&tokens, quant_type, tokenizer.vocab_size())?;

            // Cross-validate with C++ implementation for same quantization
            let cpp_validator = CppReferenceValidator::new(&model_path)?
                .with_quantization(quant_type);

            let cpp_tokens = cpp_validator.tokenize(test_text, true)?;
            assert_token_compatibility(&tokens, &cpp_tokens, quant_type)?;
        }

        Ok(())
    }

    fn validate_tokens_for_quantization(
        tokens: &[u32],
        quant_type: QuantizationType,
        vocab_size: usize
    ) -> Result<()> {
        match quant_type {
            QuantizationType::I2S => {
                // I2S supports full vocabulary range
                for &token in tokens {
                    assert!((token as usize) < vocab_size,
                        "I2S: Token {} exceeds vocab size {}", token, vocab_size);
                }
            }
            QuantizationType::TL1 | QuantizationType::TL2 => {
                // Table lookup methods may have constraints
                for &token in tokens {
                    assert!((token as usize) < 65536,
                        "TL: Token {} exceeds lookup table limit", token);
                }
            }
            QuantizationType::IQ2S => {
                // GGML-compatible quantization
                for &token in tokens {
                    assert!((token as usize) < vocab_size,
                        "IQ2S: Token {} exceeds vocab size {}", token, vocab_size);
                }
            }
        }
        Ok(())
    }
}
```

### 2. Performance Parity Tests

```rust
// tests/crossval/performance_parity/test_performance_crossval.rs
#[cfg(test)]
mod performance_parity_tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn test_tokenization_performance_vs_cpp() -> Result<()> {
        let model_path = get_test_model_path()?;

        // Setup Rust tokenizer via discovery
        let discovery = TokenizerDiscovery::from_gguf(&model_path)?;
        let strategy = discovery.discover_tokenizer_strategy()?;
        let resolver = TokenizerStrategyResolver::new(discovery).await?;
        let rust_tokenizer = resolver.resolve_tokenizer(strategy).await?;

        // Setup C++ reference
        let cpp_validator = CppReferenceValidator::new(&model_path)?;

        // Performance test corpus
        let test_corpus = load_performance_test_corpus()?;
        let warmup_iterations = 10;
        let benchmark_iterations = 100;

        // Warmup both implementations
        for _ in 0..warmup_iterations {
            rust_tokenizer.encode(&test_corpus, true, true)?;
            cpp_validator.tokenize(&test_corpus, true)?;
        }

        // Benchmark Rust implementation
        let rust_start = Instant::now();
        for _ in 0..benchmark_iterations {
            rust_tokenizer.encode(&test_corpus, true, true)?;
        }
        let rust_duration = rust_start.elapsed();

        // Benchmark C++ implementation
        let cpp_start = Instant::now();
        for _ in 0..benchmark_iterations {
            cpp_validator.tokenize(&test_corpus, true)?;
        }
        let cpp_duration = cpp_start.elapsed();

        // Performance analysis
        let rust_tokens_per_sec = calculate_tokens_per_second(&test_corpus, rust_duration, benchmark_iterations);
        let cpp_tokens_per_sec = calculate_tokens_per_second(&test_corpus, cpp_duration, benchmark_iterations);

        let performance_ratio = rust_tokens_per_sec / cpp_tokens_per_sec;

        // Rust should be within 10% of C++ performance (target from performance requirements)
        assert!(performance_ratio >= 0.90,
            "Rust performance ({:.0} tok/s) is more than 10% slower than C++ ({:.0} tok/s), ratio: {:.2}",
            rust_tokens_per_sec, cpp_tokens_per_sec, performance_ratio);

        println!("✅ Performance parity validated:");
        println!("   Rust: {:.0} tokens/sec", rust_tokens_per_sec);
        println!("   C++:  {:.0} tokens/sec", cpp_tokens_per_sec);
        println!("   Ratio: {:.2}", performance_ratio);

        Ok(())
    }

    #[test]
    fn test_memory_usage_parity() -> Result<()> {
        let model_path = get_test_model_path()?;

        // Measure Rust memory usage
        let rust_memory_before = get_process_memory_usage()?;
        let discovery = TokenizerDiscovery::from_gguf(&model_path)?;
        let strategy = discovery.discover_tokenizer_strategy()?;
        let resolver = TokenizerStrategyResolver::new(discovery).await?;
        let _rust_tokenizer = resolver.resolve_tokenizer(strategy).await?;
        let rust_memory_after = get_process_memory_usage()?;
        let rust_memory_overhead = rust_memory_after - rust_memory_before;

        // Compare with C++ reference memory usage
        let cpp_memory_before = get_process_memory_usage()?;
        let _cpp_validator = CppReferenceValidator::new(&model_path)?;
        let cpp_memory_after = get_process_memory_usage()?;
        let cpp_memory_overhead = cpp_memory_after - cpp_memory_before;

        let memory_ratio = rust_memory_overhead as f64 / cpp_memory_overhead as f64;

        // Rust should use at most 10% more memory than C++ (from performance requirements)
        assert!(memory_ratio <= 1.10,
            "Rust memory overhead ({} MB) exceeds C++ by more than 10% ({} MB), ratio: {:.2}",
            rust_memory_overhead / (1024 * 1024),
            cpp_memory_overhead / (1024 * 1024),
            memory_ratio);

        println!("✅ Memory usage parity validated:");
        println!("   Rust overhead: {} MB", rust_memory_overhead / (1024 * 1024));
        println!("   C++ overhead:  {} MB", cpp_memory_overhead / (1024 * 1024));
        println!("   Ratio: {:.2}", memory_ratio);

        Ok(())
    }
}
```

## Performance Regression Testing

### 1. Automated Benchmarking

```rust
// tests/performance/benchmarks/test_tokenizer_benchmarks.rs
#[cfg(test)]
mod tokenizer_benchmarks {
    use super::*;
    use criterion::{Criterion, BenchmarkId};
    use bitnet_tokenizers::performance::TokenizerBenchmark;

    #[test]
    fn benchmark_tokenizer_discovery_overhead() -> Result<()> {
        let test_models = get_test_model_suite()?;

        for (name, model_path) in test_models {
            let overhead = TokenizerBenchmark::measure_discovery_overhead(&model_path)?;

            // From performance requirements: discovery should be <100ms cached, <5s download
            match name.as_str() {
                model if model.contains("cached") => {
                    assert!(overhead < Duration::from_millis(100),
                        "Discovery overhead for {} ({:?}) exceeds 100ms limit", name, overhead);
                }
                model if model.contains("download") => {
                    assert!(overhead < Duration::from_secs(5),
                        "Discovery overhead for {} ({:?}) exceeds 5s limit", name, overhead);
                }
                _ => {}
            }

            println!("✅ Discovery overhead for {}: {:?}", name, overhead);
        }

        Ok(())
    }

    #[test]
    fn benchmark_large_vocabulary_performance() -> Result<()> {
        let large_vocab_models = vec![
            ("llama3-128k", create_test_tokenizer(128256)?),
            ("llama2-32k", create_test_tokenizer(32000)?),
            ("gpt2-50k", create_test_tokenizer(50257)?),
        ];

        let test_corpus = load_large_test_corpus()?; // 10KB+ text

        for (name, tokenizer) in large_vocab_models {
            let metrics = TokenizerBenchmark::measure_large_vocab_performance(&tokenizer, &test_corpus)?;

            // Performance targets from requirements
            let expected_min_tokens_per_sec = match name {
                "llama3-128k" => 10_000.0, // Large vocab GPU target
                "llama2-32k" => 15_000.0,  // Medium vocab GPU target
                "gpt2-50k" => 20_000.0,    // Standard vocab GPU target
                _ => 5_000.0,
            };

            assert!(metrics.tokens_per_second >= expected_min_tokens_per_sec,
                "Tokenization performance for {} ({:.0} tok/s) below minimum {:.0} tok/s",
                name, metrics.tokens_per_second, expected_min_tokens_per_sec);

            println!("✅ Large vocab performance for {}: {:.0} tok/s, {:.1} MB memory, {:.2}% cache hit",
                name, metrics.tokens_per_second, metrics.memory_usage_mb, metrics.cache_hit_rate * 100.0);
        }

        Ok(())
    }

    #[tokio::test]
    async fn benchmark_download_performance() -> Result<()> {
        let download_scenarios = vec![
            ("small-tokenizer", create_small_download_info()?),
            ("large-tokenizer", create_large_download_info()?),
            ("multi-file", create_multi_file_download_info()?),
        ];

        for (name, download_info) in download_scenarios {
            let metrics = TokenizerBenchmark::measure_download_performance(&download_info).await?;

            // Network performance targets
            assert!(metrics.download_time < Duration::from_secs(300),
                "Download time for {} ({:?}) exceeds 5 minute timeout", name, metrics.download_time);

            assert!(metrics.network_efficiency >= 0.70,
                "Network efficiency for {} ({:.2}) below 70% target", name, metrics.network_efficiency);

            println!("✅ Download performance for {}: {:?} download, {:?} cache, {:.2}% efficiency",
                name, metrics.download_time, metrics.cache_time, metrics.network_efficiency * 100.0);
        }

        Ok(())
    }
}
```

### 2. Memory Profiling Tests

```rust
// tests/performance/memory_profiling/test_memory_usage.rs
#[cfg(test)]
mod memory_profiling_tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Custom allocator for memory tracking
    struct TrackingAllocator;

    static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = System.alloc(layout);
            if !ptr.is_null() {
                ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
            System.dealloc(ptr, layout);
        }
    }

    #[global_allocator]
    static ALLOCATOR: TrackingAllocator = TrackingAllocator;

    #[tokio::test]
    async fn test_tokenizer_discovery_memory_usage() -> Result<()> {
        // Reset memory counter
        let baseline_memory = ALLOCATED.load(Ordering::SeqCst);

        // Perform tokenizer discovery
        let model_path = get_test_model_path()?;
        let discovery = TokenizerDiscovery::from_gguf(&model_path)?;
        let strategy = discovery.discover_tokenizer_strategy()?;
        let resolver = TokenizerStrategyResolver::new(discovery).await?;
        let _tokenizer = resolver.resolve_tokenizer(strategy).await?;

        let peak_memory = ALLOCATED.load(Ordering::SeqCst);
        let memory_overhead = peak_memory - baseline_memory;

        // From performance requirements: <100MB overhead for medium vocab
        let max_overhead_bytes = 100 * 1024 * 1024; // 100MB

        assert!(memory_overhead < max_overhead_bytes,
            "Tokenizer discovery memory overhead ({} MB) exceeds 100MB limit",
            memory_overhead / (1024 * 1024));

        println!("✅ Memory overhead: {} MB", memory_overhead / (1024 * 1024));

        Ok(())
    }

    #[test]
    fn test_memory_leak_detection() -> Result<()> {
        let baseline_memory = ALLOCATED.load(Ordering::SeqCst);

        // Perform multiple tokenizer operations
        for i in 0..10 {
            let model_path = get_test_model_path()?;
            let discovery = TokenizerDiscovery::from_gguf(&model_path)?;
            let strategy = discovery.discover_tokenizer_strategy()?;

            // Strategy should be dropped here
            drop(strategy);
            drop(discovery);

            // Force garbage collection
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let final_memory = ALLOCATED.load(Ordering::SeqCst);
        let memory_growth = final_memory.saturating_sub(baseline_memory);

        // From performance requirements: <1MB memory leak threshold
        let max_leak_bytes = 1024 * 1024; // 1MB

        assert!(memory_growth < max_leak_bytes,
            "Memory leak detected: {} KB growth after multiple operations",
            memory_growth / 1024);

        println!("✅ Memory leak test passed: {} KB growth", memory_growth / 1024);

        Ok(())
    }
}
```

## Error Recovery and Edge Case Testing

### 1. Error Handling Validation

```rust
// tests/e2e/error_recovery/test_error_scenarios.rs
#[cfg(test)]
mod error_recovery_tests {
    use super::*;

    #[tokio::test]
    async fn test_comprehensive_error_recovery() -> Result<()> {
        // AC5: Fallback strategy system with proper error reporting
        let error_scenarios = vec![
            ErrorScenario::MissingModel,
            ErrorScenario::CorruptedGguf,
            ErrorScenario::NetworkFailure,
            ErrorScenario::InvalidTokenizer,
            ErrorScenario::DiskSpaceExhausted,
            ErrorScenario::PermissionDenied,
        ];

        for scenario in error_scenarios {
            let result = simulate_error_scenario(scenario).await;

            match result {
                Ok(_) => panic!("Expected error for scenario {:?}", scenario),
                Err(error) => {
                    // Verify error provides actionable suggestions
                    let error_msg = error.to_string();
                    assert!(!error_msg.is_empty());

                    if let Some(discovery_error) = error.downcast_ref::<TokenizerDiscoveryError>() {
                        let suggestions = discovery_error.suggestions();
                        assert!(!suggestions.is_empty(), "Error should provide actionable suggestions");

                        // Verify suggestions are actionable
                        for suggestion in &suggestions {
                            assert!(suggestion.contains("Use ") || suggestion.contains("Check ") || suggestion.contains("Remove "),
                                "Suggestion should be actionable: {}", suggestion);
                        }
                    }
                }
            }

            println!("✅ Error recovery validated for: {:?}", scenario);
        }

        Ok(())
    }

    #[test]
    fn test_strict_mode_error_boundaries() -> Result<()> {
        // Test error boundaries with strict mode enabled
        temp_env::with_var("BITNET_STRICT_TOKENIZERS", Some("1"), || {
            let test_scenarios = vec![
                (create_model_without_tokenizer()?, "No compatible tokenizer found"),
                (create_model_with_invalid_metadata()?, "GGUF metadata parsing failed"),
                (create_model_with_wrong_vocab_size()?, "Tokenizer validation failed"),
            ];

            for (model_path, expected_error_fragment) in test_scenarios {
                let discovery = TokenizerDiscovery::from_gguf(&model_path);

                match discovery {
                    Ok(discovery) => {
                        let strategy_result = discovery.discover_tokenizer_strategy();
                        assert!(strategy_result.is_err(), "Should fail in strict mode");

                        let error_msg = strategy_result.unwrap_err().to_string();
                        assert!(error_msg.contains(expected_error_fragment),
                            "Error message should contain '{}', got: {}", expected_error_fragment, error_msg);
                    }
                    Err(_) => {
                        // Early failure is also acceptable in strict mode
                    }
                }
            }

            Ok::<(), anyhow::Error>(())
        })?;

        Ok(())
    }

    async fn simulate_error_scenario(scenario: ErrorScenario) -> Result<Arc<dyn Tokenizer>> {
        match scenario {
            ErrorScenario::MissingModel => {
                let nonexistent_path = PathBuf::from("/nonexistent/model.gguf");
                let discovery = TokenizerDiscovery::from_gguf(&nonexistent_path)?;
                let strategy = discovery.discover_tokenizer_strategy()?;
                let resolver = TokenizerStrategyResolver::new(discovery).await?;
                resolver.resolve_tokenizer(strategy).await
            }
            ErrorScenario::NetworkFailure => {
                // Simulate network failure during download
                temp_env::with_var("BITNET_FORCE_NETWORK_FAILURE", Some("1"), || async {
                    let model_path = create_model_requiring_download()?;
                    let discovery = TokenizerDiscovery::from_gguf(&model_path)?;
                    let strategy = discovery.discover_tokenizer_strategy()?;
                    let resolver = TokenizerStrategyResolver::new(discovery).await?;
                    resolver.resolve_tokenizer(strategy).await
                }).await
            }
            // ... implement other scenarios
            _ => todo!("Implement remaining error scenarios"),
        }
    }
}
```

## Test Execution Framework

### 1. Test Suite Organization

```bash
#!/bin/bash
# scripts/run-tokenizer-tests.sh

set -e

echo "🧪 Running BitNet.rs Tokenizer Discovery Test Suite"

# Environment setup
export RUST_LOG=debug
export BITNET_TEST_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42

# Unit tests
echo "📋 Running unit tests..."
cargo test --no-default-features --features cpu -p bitnet-tokenizers unit::

# Integration tests with real models
echo "🔗 Running integration tests..."
if [ -n "$BITNET_TEST_MODELS_PATH" ]; then
    cargo test --no-default-features --features cpu integration::real_models::
else
    echo "⚠️  Skipping real model tests (BITNET_TEST_MODELS_PATH not set)"
fi

# Cross-validation tests
echo "⚖️  Running cross-validation tests..."
if [ -n "$BITNET_GGUF" ]; then
    cargo test --no-default-features --features cpu,ffi crossval::
else
    echo "⚠️  Skipping cross-validation (BITNET_GGUF not set)"
fi

# Performance regression tests
echo "📈 Running performance tests..."
cargo test --release --no-default-features --features cpu performance::

# Error recovery tests
echo "🛡️  Running error recovery tests..."
cargo test --no-default-features --features cpu e2e::error_recovery::

# Feature flag compatibility tests
echo "🏁 Running feature flag tests..."
cargo test --no-default-features --features cpu
cargo test --no-default-features --features gpu
cargo test --no-default-features --features spm
cargo test --no-default-features --features cpu,smp,ffi

echo "✅ All tokenizer discovery tests passed!"
```

### 2. Continuous Integration Configuration

```yaml
# .github/workflows/tokenizer-discovery-tests.yml
name: Tokenizer Discovery Tests

on:
  push:
    paths:
      - 'crates/bitnet-tokenizers/**'
      - 'docs/explanation/*tokenizer*'
      - 'tests/**/*tokenizer*'
  pull_request:
    paths:
      - 'crates/bitnet-tokenizers/**'
      - 'docs/explanation/*tokenizer*'
      - 'tests/**/*tokenizer*'

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        features: [cpu, gpu, spm, "cpu,spm", "gpu,spm,ffi"]

    steps:
    - uses: actions/checkout@v4
    - uses: actions-rs/toolchain@v1
      with:
        toolchain: 1.90.0
        override: true

    - name: Run unit tests
      run: |
        cargo test --no-default-features --features ${{ matrix.features }} \
          -p bitnet-tokenizers unit::

    - name: Run integration tests
      run: |
        cargo test --no-default-features --features ${{ matrix.features }} \
          integration::network_simulation::

  cross-validation:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'

    steps:
    - uses: actions/checkout@v4
    - name: Download test models
      run: |
        mkdir -p models/test
        wget -O models/test/llama-2-7b.gguf \
          "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf"

    - name: Run cross-validation tests
      env:
        BITNET_GGUF: models/test/llama-2-7b.gguf
        BITNET_DETERMINISTIC: 1
        BITNET_SEED: 42
      run: |
        cargo test --no-default-features --features cpu,ffi crossval::

  performance-regression:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run performance benchmarks
      run: |
        cargo test --release --no-default-features --features cpu \
          performance:: -- --nocapture

    - name: Compare with baseline
      run: |
        ./scripts/performance-regression-check.sh
```

## Conclusion

This comprehensive testing strategy ensures BitNet.rs tokenizer discovery system meets production-grade reliability standards through:

- **Systematic Unit Testing**: Individual component validation with neural network model-specific scenarios
- **Integration Testing**: Real-world model compatibility and end-to-end workflow validation
- **Cross-Validation Framework**: Parity testing with C++ reference implementation for accuracy and performance
- **Performance Regression Testing**: Continuous monitoring of performance metrics and memory usage
- **Error Recovery Validation**: Comprehensive error handling and fallback strategy testing
- **Continuous Integration**: Automated testing across feature flag combinations and platforms

The testing framework validates all 10 acceptance criteria from Issue #249 while ensuring compatibility with BitNet.rs neural network inference pipeline, quantization formats (I2S/TL1/TL2), and production deployment requirements.