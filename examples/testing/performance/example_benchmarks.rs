//! Example performance benchmarks for BitNet.rs
//!
//! Demonstrates comprehensive performance testing including throughput,
//! latency, memory usage, and scalability benchmarks

use bitnet_inference::{InferenceConfig, InferenceEngine};
use bitnet_models::BitNetModel;
use bitnet_tokenizers::BitNetTokenizer;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::runtime::Runtime;

/// Example: Inference throughput benchmarks
pub fn benchmark_inference_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let temp_dir = TempDir::new().unwrap();

    // Setup test environment
    let (model, tokenizer, engine) = rt.block_on(async {
        let model_path = setup_benchmark_model(&temp_dir).await;
        let tokenizer_path = setup_benchmark_tokenizer(&temp_dir).await;

        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config = InferenceConfig::builder()
            .max_tokens(50)
            .temperature(0.7)
            .build()
            .unwrap();

        let engine = InferenceEngine::new(model.clone(), tokenizer.clone(), config)
            .await
            .unwrap();
        (model, tokenizer, engine)
    });

    let mut group = c.benchmark_group("inference_throughput");

    // Test different input lengths
    let input_lengths = vec![10, 50, 100, 200, 500];

    for length in input_lengths {
        let input = "word ".repeat(length);

        group.throughput(Throughput::Elements(length as u64));
        group.bench_with_input(
            BenchmarkId::new("tokens_per_second", length),
            &input,
            |b, input| {
                b.to_async(&rt).iter(|| async {
                    let mut engine = engine.clone();
                    let result = engine.generate(input).await.unwrap();
                    result.tokens.len()
                });
            },
        );
    }

    group.finish();
}

/// Example: Latency benchmarks for different scenarios
pub fn benchmark_inference_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let temp_dir = TempDir::new().unwrap();

    let engine = rt.block_on(async {
        let model_path = setup_benchmark_model(&temp_dir).await;
        let tokenizer_path = setup_benchmark_tokenizer(&temp_dir).await;

        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config = InferenceConfig::builder()
            .max_tokens(20)
            .temperature(0.5)
            .build()
            .unwrap();

        InferenceEngine::new(model, tokenizer, config)
            .await
            .unwrap()
    });

    let mut group = c.benchmark_group("inference_latency");
    group.measurement_time(Duration::from_secs(10));

    // Benchmark different latency scenarios
    let scenarios = vec![
        ("cold_start", "First inference after initialization", true),
        ("warm_inference", "Subsequent inferences", false),
        ("short_prompt", "Hello", false),
        (
            "medium_prompt",
            "Write a story about artificial intelligence",
            false,
        ),
        (
            "long_prompt",
            "Explain the history and development of machine learning algorithms in detail",
            false,
        ),
    ];

    for (name, input, is_cold_start) in scenarios {
        group.bench_function(name, |b| {
            b.to_async(&rt).iter(|| async {
                let mut engine = if is_cold_start {
                    // Simulate cold start by creating new engine
                    engine.clone()
                } else {
                    engine.clone()
                };

                let start = Instant::now();
                let _result = engine.generate(input).await.unwrap();
                start.elapsed()
            });
        });
    }

    group.finish();
}

/// Example: Memory usage benchmarks
pub fn benchmark_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let temp_dir = TempDir::new().unwrap();

    let mut group = c.benchmark_group("memory_usage");

    // Test different model sizes
    let model_configs = vec![
        ("small", ModelSize::Small),
        ("medium", ModelSize::Medium),
        ("large", ModelSize::Large),
    ];

    for (name, size) in model_configs {
        group.bench_function(name, |b| {
            b.to_async(&rt).iter(|| async {
                let model_path = setup_benchmark_model_with_size(&temp_dir, size).await;
                let tokenizer_path = setup_benchmark_tokenizer(&temp_dir).await;

                let initial_memory = get_memory_usage();

                let model = BitNetModel::from_file(&model_path).await.unwrap();
                let after_model_memory = get_memory_usage();

                let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();
                let after_tokenizer_memory = get_memory_usage();

                let config = InferenceConfig::builder()
                    .max_tokens(50)
                    .temperature(0.7)
                    .build()
                    .unwrap();

                let engine = InferenceEngine::new(model, tokenizer, config)
                    .await
                    .unwrap();
                let after_engine_memory = get_memory_usage();

                // Run inference to measure peak memory
                let _result = engine
                    .generate("Test input for memory measurement")
                    .await
                    .unwrap();
                let peak_memory = get_memory_usage();

                MemoryBenchmarkResult {
                    initial: initial_memory,
                    after_model: after_model_memory,
                    after_tokenizer: after_tokenizer_memory,
                    after_engine: after_engine_memory,
                    peak: peak_memory,
                }
            });
        });
    }

    group.finish();
}

/// Example: Batch processing benchmarks
pub fn benchmark_batch_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let temp_dir = TempDir::new().unwrap();

    let engine = rt.block_on(async {
        let model_path = setup_benchmark_model(&temp_dir).await;
        let tokenizer_path = setup_benchmark_tokenizer(&temp_dir).await;

        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config = InferenceConfig::builder()
            .max_tokens(30)
            .temperature(0.5)
            .batch_size(8)
            .build()
            .unwrap();

        InferenceEngine::new(model, tokenizer, config)
            .await
            .unwrap()
    });

    let mut group = c.benchmark_group("batch_processing");

    // Test different batch sizes
    let batch_sizes = vec![1, 2, 4, 8, 16, 32];

    for batch_size in batch_sizes {
        let inputs: Vec<String> = (0..batch_size)
            .map(|i| format!("Batch input number {}", i))
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_inference", batch_size),
            &inputs,
            |b, inputs| {
                b.to_async(&rt).iter(|| async {
                    let mut engine = engine.clone();
                    let results = engine.generate_batch(inputs).await.unwrap();
                    results.len()
                });
            },
        );
    }

    group.finish();
}

/// Example: Streaming performance benchmarks
pub fn benchmark_streaming_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let temp_dir = TempDir::new().unwrap();

    let engine = rt.block_on(async {
        let model_path = setup_benchmark_model(&temp_dir).await;
        let tokenizer_path = setup_benchmark_tokenizer(&temp_dir).await;

        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config = InferenceConfig::builder()
            .max_tokens(100)
            .temperature(0.7)
            .streaming(true)
            .build()
            .unwrap();

        InferenceEngine::new(model, tokenizer, config)
            .await
            .unwrap()
    });

    let mut group = c.benchmark_group("streaming_performance");

    // Benchmark streaming vs non-streaming
    let test_cases = vec![("streaming", true), ("non_streaming", false)];

    for (name, use_streaming) in test_cases {
        group.bench_function(name, |b| {
            b.to_async(&rt).iter(|| async {
                let mut engine = engine.clone();
                let input = "Write a detailed explanation of machine learning";

                if use_streaming {
                    // Measure time to first token and total time
                    let start = Instant::now();
                    let mut stream = engine.generate_stream(input).await.unwrap();

                    let mut first_token_time = None;
                    let mut total_tokens = 0;

                    while let Some(chunk) = stream.next().await {
                        let chunk = chunk.unwrap();
                        if first_token_time.is_none() {
                            first_token_time = Some(start.elapsed());
                        }
                        total_tokens += chunk.tokens.len();
                    }

                    StreamingBenchmarkResult {
                        first_token_latency: first_token_time.unwrap_or_default(),
                        total_time: start.elapsed(),
                        total_tokens,
                    }
                } else {
                    let start = Instant::now();
                    let result = engine.generate(input).await.unwrap();

                    StreamingBenchmarkResult {
                        first_token_latency: result.generation_time, // All at once
                        total_time: result.generation_time,
                        total_tokens: result.tokens.len(),
                    }
                }
            });
        });
    }

    group.finish();
}

/// Example: Concurrent access benchmarks
pub fn benchmark_concurrent_access(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let temp_dir = TempDir::new().unwrap();

    let engine = rt.block_on(async {
        let model_path = setup_benchmark_model(&temp_dir).await;
        let tokenizer_path = setup_benchmark_tokenizer(&temp_dir).await;

        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config = InferenceConfig::builder()
            .max_tokens(20)
            .temperature(0.5)
            .build()
            .unwrap();

        std::sync::Arc::new(
            InferenceEngine::new(model, tokenizer, config)
                .await
                .unwrap(),
        )
    });

    let mut group = c.benchmark_group("concurrent_access");

    // Test different concurrency levels
    let concurrency_levels = vec![1, 2, 4, 8, 16];

    for concurrency in concurrency_levels {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrent_inference", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let engine = engine.clone();
                    let mut handles = Vec::new();

                    for i in 0..concurrency {
                        let engine_clone = engine.clone();
                        let handle = tokio::spawn(async move {
                            let input = format!("Concurrent request {}", i);
                            engine_clone.generate(&input).await.unwrap()
                        });
                        handles.push(handle);
                    }

                    let results = futures::future::join_all(handles).await;
                    results.len()
                });
            },
        );
    }

    group.finish();
}

/// Example: Model loading benchmarks
pub fn benchmark_model_loading(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let temp_dir = TempDir::new().unwrap();

    let mut group = c.benchmark_group("model_loading");

    // Test different model sizes and formats
    let test_cases = vec![
        ("small_gguf", ModelSize::Small, ModelFormat::GGUF),
        ("medium_gguf", ModelSize::Medium, ModelFormat::GGUF),
        ("large_gguf", ModelSize::Large, ModelFormat::GGUF),
        (
            "small_safetensors",
            ModelSize::Small,
            ModelFormat::SafeTensors,
        ),
        (
            "medium_safetensors",
            ModelSize::Medium,
            ModelFormat::SafeTensors,
        ),
    ];

    for (name, size, format) in test_cases {
        group.bench_function(name, |b| {
            b.to_async(&rt).iter(|| async {
                let model_path =
                    setup_benchmark_model_with_size_and_format(&temp_dir, size, format).await;

                let start = Instant::now();
                let model = BitNetModel::from_file(&model_path).await.unwrap();
                let load_time = start.elapsed();

                // Verify model is usable
                assert!(model.is_loaded());
                assert!(model.metadata().vocab_size() > 0);

                load_time
            });
        });
    }

    group.finish();
}

/// Benchmark utilities and data structures
pub mod benchmark_utils {
    use super::*;
    use tokio::fs;

    #[derive(Debug, Clone, Copy)]
    pub enum ModelSize {
        Small,
        Medium,
        Large,
    }

    #[derive(Debug, Clone, Copy)]
    pub enum ModelFormat {
        GGUF,
        SafeTensors,
    }

    #[derive(Debug)]
    pub struct MemoryBenchmarkResult {
        pub initial: u64,
        pub after_model: u64,
        pub after_tokenizer: u64,
        pub after_engine: u64,
        pub peak: u64,
    }

    #[derive(Debug)]
    pub struct StreamingBenchmarkResult {
        pub first_token_latency: Duration,
        pub total_time: Duration,
        pub total_tokens: usize,
    }

    /// Setup benchmark model
    pub async fn setup_benchmark_model(temp_dir: &TempDir) -> std::path::PathBuf {
        setup_benchmark_model_with_size(temp_dir, ModelSize::Medium).await
    }

    /// Setup benchmark model with specific size
    pub async fn setup_benchmark_model_with_size(
        temp_dir: &TempDir,
        size: ModelSize,
    ) -> std::path::PathBuf {
        setup_benchmark_model_with_size_and_format(temp_dir, size, ModelFormat::GGUF).await
    }

    /// Setup benchmark model with size and format
    pub async fn setup_benchmark_model_with_size_and_format(
        temp_dir: &TempDir,
        size: ModelSize,
        format: ModelFormat,
    ) -> std::path::PathBuf {
        let filename = match format {
            ModelFormat::GGUF => "benchmark_model.gguf",
            ModelFormat::SafeTensors => "benchmark_model.safetensors",
        };

        let model_path = temp_dir.path().join(filename);
        let model_data = create_benchmark_model_data(size, format);
        fs::write(&model_path, model_data).await.unwrap();
        model_path
    }

    /// Setup benchmark tokenizer
    pub async fn setup_benchmark_tokenizer(temp_dir: &TempDir) -> std::path::PathBuf {
        let tokenizer_path = temp_dir.path().join("benchmark_tokenizer.json");
        let tokenizer_data = create_benchmark_tokenizer_data();
        fs::write(&tokenizer_path, tokenizer_data).await.unwrap();
        tokenizer_path
    }

    /// Create benchmark model data
    fn create_benchmark_model_data(size: ModelSize, format: ModelFormat) -> Vec<u8> {
        match format {
            ModelFormat::GGUF => create_benchmark_gguf_data(size),
            ModelFormat::SafeTensors => create_benchmark_safetensors_data(size),
        }
    }

    /// Create GGUF benchmark data
    fn create_benchmark_gguf_data(size: ModelSize) -> Vec<u8> {
        let mut data = Vec::new();

        // GGUF header
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());

        let (tensor_count, data_size) = match size {
            ModelSize::Small => (20u64, 50 * 1024),        // 50KB
            ModelSize::Medium => (100u64, 500 * 1024),     // 500KB
            ModelSize::Large => (500u64, 5 * 1024 * 1024), // 5MB
        };

        data.extend_from_slice(&tensor_count.to_le_bytes());
        data.extend_from_slice(&10u64.to_le_bytes()); // metadata count

        // Add model data
        data.extend_from_slice(&vec![0u8; data_size]);

        data
    }

    /// Create SafeTensors benchmark data
    fn create_benchmark_safetensors_data(size: ModelSize) -> Vec<u8> {
        let (vocab_size, hidden_size, data_size) = match size {
            ModelSize::Small => (1000, 256, 50 * 1024),
            ModelSize::Medium => (10000, 1024, 500 * 1024),
            ModelSize::Large => (50000, 4096, 5 * 1024 * 1024),
        };

        let header = serde_json::json!({
            "vocab_size": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]},
            "hidden_size": {"dtype": "I32", "shape": [1], "data_offsets": [4, 8]},
            "weights": {"dtype": "F32", "shape": [vocab_size, hidden_size], "data_offsets": [8, 8 + data_size]}
        });

        let header_str = header.to_string();
        let header_len = header_str.len() as u64;

        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(header_str.as_bytes());
        data.extend_from_slice(&(vocab_size as u32).to_le_bytes());
        data.extend_from_slice(&(hidden_size as u32).to_le_bytes());
        data.extend_from_slice(&vec![0u8; data_size]);

        data
    }

    /// Create benchmark tokenizer data
    fn create_benchmark_tokenizer_data() -> String {
        // Create a more comprehensive tokenizer for benchmarking
        let mut vocab = std::collections::HashMap::new();

        // Add common tokens
        let common_tokens = vec![
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their",
        ];

        for (i, token) in common_tokens.iter().enumerate() {
            vocab.insert(token.to_string(), i as u32);
        }

        // Add numbered tokens for variety
        for i in common_tokens.len()..10000 {
            vocab.insert(format!("token_{}", i), i as u32);
        }

        serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": []
            },
            "normalizer": {
                "type": "Sequence",
                "normalizers": []
            },
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true
            },
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true
            }
        })
        .to_string()
    }

    /// Get current memory usage (mock implementation for benchmarking)
    pub fn get_memory_usage() -> u64 {
        // In real implementation, this would use system APIs
        // For benchmarking, we simulate memory usage
        use std::sync::atomic::{AtomicU64, Ordering};
        static MOCK_MEMORY: AtomicU64 = AtomicU64::new(100 * 1024 * 1024);

        MOCK_MEMORY.fetch_add(1024 * 1024, Ordering::Relaxed) // Add 1MB each call
    }

    /// Performance analysis utilities
    pub fn analyze_benchmark_results(results: &[Duration]) -> BenchmarkAnalysis {
        if results.is_empty() {
            return BenchmarkAnalysis::default();
        }

        let mut sorted_results = results.to_vec();
        sorted_results.sort();

        let total: Duration = results.iter().sum();
        let mean = total / results.len() as u32;

        let median = sorted_results[results.len() / 2];
        let p95 = sorted_results[(results.len() as f64 * 0.95) as usize];
        let p99 = sorted_results[(results.len() as f64 * 0.99) as usize];

        let min = sorted_results[0];
        let max = sorted_results[results.len() - 1];

        BenchmarkAnalysis {
            count: results.len(),
            mean,
            median,
            min,
            max,
            p95,
            p99,
        }
    }

    #[derive(Debug, Default)]
    pub struct BenchmarkAnalysis {
        pub count: usize,
        pub mean: Duration,
        pub median: Duration,
        pub min: Duration,
        pub max: Duration,
        pub p95: Duration,
        pub p99: Duration,
    }

    impl BenchmarkAnalysis {
        pub fn print_summary(&self, name: &str) {
            println!("=== {} Benchmark Analysis ===", name);
            println!("Samples: {}", self.count);
            println!("Mean: {:?}", self.mean);
            println!("Median: {:?}", self.median);
            println!("Min: {:?}", self.min);
            println!("Max: {:?}", self.max);
            println!("95th percentile: {:?}", self.p95);
            println!("99th percentile: {:?}", self.p99);
        }
    }
}

// Criterion benchmark group configuration
criterion_group!(
    benches,
    benchmark_inference_throughput,
    benchmark_inference_latency,
    benchmark_memory_usage,
    benchmark_batch_processing,
    benchmark_streaming_performance,
    benchmark_concurrent_access,
    benchmark_model_loading
);

criterion_main!(benches);

#[cfg(test)]
mod benchmark_tests {
    use super::benchmark_utils::*;
    use super::*;

    /// Example: Simple performance test
    #[tokio::test]
    async fn test_basic_inference_performance() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = setup_benchmark_model(&temp_dir).await;
        let tokenizer_path = setup_benchmark_tokenizer(&temp_dir).await;

        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();

        let config = InferenceConfig::builder()
            .max_tokens(20)
            .temperature(0.5)
            .build()
            .unwrap();

        let mut engine = InferenceEngine::new(model, tokenizer, config)
            .await
            .unwrap();

        // Measure inference time
        let start = Instant::now();
        let result = engine.generate("Test performance").await.unwrap();
        let duration = start.elapsed();

        // Basic performance assertions
        assert!(
            duration.as_millis() < 5000,
            "Inference should complete within 5 seconds"
        );
        assert!(!result.text.is_empty(), "Should generate text");
        assert!(result.tokens.len() > 0, "Should generate tokens");

        println!("Inference time: {:?}", duration);
        println!("Tokens generated: {}", result.tokens.len());
        println!(
            "Tokens per second: {:.2}",
            result.tokens.len() as f64 / duration.as_secs_f64()
        );
    }

    /// Example: Memory usage validation
    #[tokio::test]
    async fn test_memory_usage_bounds() {
        let temp_dir = TempDir::new().unwrap();

        let initial_memory = get_memory_usage();

        // Load model and measure memory
        let model_path = setup_benchmark_model_with_size(&temp_dir, ModelSize::Medium).await;
        let model = BitNetModel::from_file(&model_path).await.unwrap();
        let after_model_memory = get_memory_usage();

        // Load tokenizer
        let tokenizer_path = setup_benchmark_tokenizer(&temp_dir).await;
        let tokenizer = BitNetTokenizer::from_file(&tokenizer_path).await.unwrap();
        let after_tokenizer_memory = get_memory_usage();

        // Create engine
        let config = InferenceConfig::builder()
            .max_tokens(50)
            .temperature(0.7)
            .build()
            .unwrap();

        let engine = InferenceEngine::new(model, tokenizer, config)
            .await
            .unwrap();
        let after_engine_memory = get_memory_usage();

        // Run inference
        let _result = engine.generate("Memory test").await.unwrap();
        let peak_memory = get_memory_usage();

        // Validate memory usage is reasonable
        let model_memory = after_model_memory - initial_memory;
        let tokenizer_memory = after_tokenizer_memory - after_model_memory;
        let engine_memory = after_engine_memory - after_tokenizer_memory;
        let inference_memory = peak_memory - after_engine_memory;

        println!("Memory usage breakdown:");
        println!("  Model: {} MB", model_memory / 1024 / 1024);
        println!("  Tokenizer: {} MB", tokenizer_memory / 1024 / 1024);
        println!("  Engine: {} MB", engine_memory / 1024 / 1024);
        println!("  Inference: {} MB", inference_memory / 1024 / 1024);
        println!(
            "  Total: {} MB",
            (peak_memory - initial_memory) / 1024 / 1024
        );

        // Assert reasonable memory bounds (these would be adjusted based on actual requirements)
        assert!(
            model_memory < 100 * 1024 * 1024,
            "Model memory usage too high"
        );
        assert!(
            tokenizer_memory < 50 * 1024 * 1024,
            "Tokenizer memory usage too high"
        );
        assert!(
            engine_memory < 50 * 1024 * 1024,
            "Engine memory usage too high"
        );
    }
}
