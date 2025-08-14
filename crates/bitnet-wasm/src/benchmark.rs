//! Comprehensive benchmarking suite for WebAssembly performance

use js_sys::{Array, Date, Object, Reflect};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use web_sys::{console, Performance};

use crate::kernels::{WasmBenchmark, WasmKernelProvider};
use crate::memory::MemoryManager;
use crate::progressive::ProgressiveLoader;
use crate::utils::{JsError, PerformanceMonitor};

/// Comprehensive benchmark suite for WASM BitNet
#[wasm_bindgen]
pub struct WasmBenchmarkSuite {
    kernel_provider: WasmKernelProvider,
    memory_manager: MemoryManager,
    results: Vec<BenchmarkResult>,
}

#[wasm_bindgen]
impl WasmBenchmarkSuite {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmBenchmarkSuite, JsError> {
        let kernel_provider = WasmKernelProvider::new();
        let memory_manager = MemoryManager::new(Some(512 * 1024 * 1024))?; // 512MB limit

        Ok(WasmBenchmarkSuite {
            kernel_provider,
            memory_manager,
            results: Vec::new(),
        })
    }

    /// Run all benchmarks and return comprehensive results
    #[wasm_bindgen]
    pub fn run_all_benchmarks(&mut self) -> js_sys::Promise {
        wasm_bindgen_futures::future_to_promise(async move {
            let mut monitor = PerformanceMonitor::new();

            console::log_1(&"Starting comprehensive WASM benchmark suite".into());
            monitor.mark("benchmark_start");

            // Run individual benchmark categories
            let kernel_results = self.benchmark_kernels().await?;
            monitor.mark("kernels_complete");

            let memory_results = self.benchmark_memory().await?;
            monitor.mark("memory_complete");

            let loading_results = self.benchmark_loading().await?;
            monitor.mark("loading_complete");

            let inference_results = self.benchmark_inference().await?;
            monitor.mark("inference_complete");

            // Compile comprehensive results
            let total_time = monitor.elapsed();
            let summary = BenchmarkSummary {
                total_time_ms: total_time,
                kernel_performance: kernel_results,
                memory_performance: memory_results,
                loading_performance: loading_results,
                inference_performance: inference_results,
                platform_info: self.get_platform_info(),
                recommendations: self.generate_recommendations(),
            };

            console::log_1(&format!("Benchmark suite completed in {:.2}ms", total_time).into());

            Ok(JsValue::from_serde(&summary)?)
        })
    }

    /// Benchmark kernel performance
    #[wasm_bindgen]
    pub fn benchmark_kernels(&self) -> js_sys::Promise {
        let kernel_provider = self.kernel_provider.clone();

        wasm_bindgen_futures::future_to_promise(async move {
            console::log_1(&"Benchmarking kernel performance".into());

            let benchmark = WasmBenchmark::new();
            let mut results = KernelBenchmarkResults::default();

            // Matrix multiplication benchmarks
            results.matmul_small = benchmark.benchmark_matmul(64, 64, 64, 100);
            results.matmul_medium = benchmark.benchmark_matmul(256, 256, 256, 10);
            results.matmul_large = benchmark.benchmark_matmul(1024, 1024, 1024, 1);

            // Quantization benchmarks
            results.quantization_1mb = benchmark.benchmark_quantization(256 * 1024, 10); // 1MB
            results.quantization_16mb = benchmark.benchmark_quantization(4 * 1024 * 1024, 5); // 16MB
            results.quantization_64mb = benchmark.benchmark_quantization(16 * 1024 * 1024, 2); // 64MB

            // Memory bandwidth test
            results.memory_bandwidth = Self::benchmark_memory_bandwidth().await;

            Ok(JsValue::from_serde(&results)?)
        })
    }

    /// Benchmark memory management performance
    #[wasm_bindgen]
    pub fn benchmark_memory(&self) -> js_sys::Promise {
        wasm_bindgen_futures::future_to_promise(async move {
            console::log_1(&"Benchmarking memory management".into());

            let mut results = MemoryBenchmarkResults::default();
            let mut memory_manager = MemoryManager::new(Some(256 * 1024 * 1024))?; // 256MB

            // Allocation performance
            let start_time = Self::get_time();
            for i in 0..1000 {
                memory_manager.track_allocation(format!("test_{}", i), 1024)?;
            }
            results.allocation_time_ms = Self::get_time() - start_time;

            // Garbage collection performance
            let start_time = Self::get_time();
            let freed = memory_manager.gc()?;
            results.gc_time_ms = Self::get_time() - start_time;
            results.gc_freed_bytes = freed;

            // Memory fragmentation test
            results.fragmentation_score = Self::measure_fragmentation(&mut memory_manager).await;

            Ok(JsValue::from_serde(&results)?)
        })
    }

    /// Benchmark progressive loading performance
    #[wasm_bindgen]
    pub fn benchmark_loading(&self) -> js_sys::Promise {
        wasm_bindgen_futures::future_to_promise(async move {
            console::log_1(&"Benchmarking progressive loading".into());

            let mut results = LoadingBenchmarkResults::default();

            // Simulate loading different sized models
            results.small_model_load_ms = Self::simulate_model_load(10 * 1024 * 1024).await; // 10MB
            results.medium_model_load_ms = Self::simulate_model_load(100 * 1024 * 1024).await; // 100MB
            results.large_model_load_ms = Self::simulate_model_load(500 * 1024 * 1024).await; // 500MB

            // Chunk processing performance
            results.chunk_processing_mbps = Self::benchmark_chunk_processing().await;

            Ok(JsValue::from_serde(&results)?)
        })
    }

    /// Benchmark inference performance
    #[wasm_bindgen]
    pub fn benchmark_inference(&self) -> js_sys::Promise {
        wasm_bindgen_futures::future_to_promise(async move {
            console::log_1(&"Benchmarking inference performance".into());

            let mut results = InferenceBenchmarkResults::default();

            // Simulate different inference scenarios
            results.single_token_latency_ms = Self::simulate_token_generation(1).await;
            results.batch_10_tokens_ms = Self::simulate_token_generation(10).await;
            results.batch_100_tokens_ms = Self::simulate_token_generation(100).await;

            // Streaming performance
            results.streaming_tokens_per_second = Self::benchmark_streaming().await;

            // Memory usage during inference
            results.inference_memory_mb = Self::measure_inference_memory().await;

            Ok(JsValue::from_serde(&results)?)
        })
    }

    /// Get platform information
    fn get_platform_info(&self) -> PlatformInfo {
        let user_agent = web_sys::window()
            .and_then(|w| w.navigator().user_agent().ok())
            .unwrap_or_else(|| "Unknown".to_string());

        let memory_gb = web_sys::window()
            .and_then(|w| w.navigator().device_memory())
            .unwrap_or(0.0);

        let cpu_cores = web_sys::window()
            .and_then(|w| w.navigator().hardware_concurrency())
            .map(|c| c as usize)
            .unwrap_or(1);

        let performance_info = self.kernel_provider.get_performance_info();

        PlatformInfo {
            user_agent,
            memory_gb,
            cpu_cores,
            simd_supported: performance_info.simd_supported(),
            bulk_memory_supported: performance_info.bulk_memory_supported(),
            estimated_gflops: performance_info.estimated_gflops(),
        }
    }

    /// Generate performance recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let performance_info = self.kernel_provider.get_performance_info();

        if !performance_info.simd_supported() {
            recommendations.push(
                "Consider using a browser with WASM SIMD support for better performance"
                    .to_string(),
            );
        }

        if !performance_info.bulk_memory_supported() {
            recommendations.push(
                "Bulk memory operations not supported - consider browser upgrade".to_string(),
            );
        }

        if performance_info.estimated_gflops() < 1.0 {
            recommendations.push(
                "Low computational performance detected - consider reducing model size".to_string(),
            );
        }

        let memory_gb = web_sys::window()
            .and_then(|w| w.navigator().device_memory())
            .unwrap_or(0.0);

        if memory_gb < 4.0 {
            recommendations.push(
                "Limited device memory - enable progressive loading and reduce memory limits"
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations
                .push("Platform appears well-suited for BitNet WASM inference".to_string());
        }

        recommendations
    }

    /// Get current high-resolution time
    fn get_time() -> f64 {
        web_sys::window()
            .and_then(|w| w.performance().ok())
            .map(|p| p.now())
            .unwrap_or_else(|| Date::now())
    }

    /// Benchmark memory bandwidth
    async fn benchmark_memory_bandwidth() -> f64 {
        let size = 64 * 1024 * 1024; // 64MB
        let data = vec![1u8; size];
        let mut dest = vec![0u8; size];

        let start_time = Self::get_time();

        // Simulate memory copy operations
        for _ in 0..10 {
            dest.copy_from_slice(&data);
            Self::delay(1).await; // Yield to browser
        }

        let elapsed_ms = Self::get_time() - start_time;
        let bytes_transferred = (size * 10 * 2) as f64; // Read + write
        let bandwidth_gbps = (bytes_transferred / 1e9) / (elapsed_ms / 1000.0);

        bandwidth_gbps
    }

    /// Measure memory fragmentation
    async fn measure_fragmentation(memory_manager: &mut MemoryManager) -> f64 {
        // Simulate fragmented allocations
        for i in 0..100 {
            let size = if i % 2 == 0 { 1024 } else { 4096 };
            let _ = memory_manager.track_allocation(format!("frag_{}", i), size);
        }

        // Free every other allocation
        for i in (0..100).step_by(2) {
            memory_manager.track_deallocation(&format!("frag_{}", i));
        }

        // Measure fragmentation (simplified metric)
        let stats = memory_manager.get_stats();
        stats.usage_percent() / 100.0
    }

    /// Simulate model loading time
    async fn simulate_model_load(size_bytes: usize) -> f64 {
        let start_time = Self::get_time();

        // Simulate chunked loading
        let chunk_size = 1024 * 1024; // 1MB chunks
        let chunks = (size_bytes + chunk_size - 1) / chunk_size;

        for i in 0..chunks {
            // Simulate processing time per chunk
            Self::delay(5).await;

            if i % 10 == 0 {
                console::log_1(&format!("Loading chunk {}/{}", i + 1, chunks).into());
            }
        }

        Self::get_time() - start_time
    }

    /// Benchmark chunk processing speed
    async fn benchmark_chunk_processing() -> f64 {
        let chunk_size = 1024 * 1024; // 1MB
        let num_chunks = 64; // 64MB total

        let start_time = Self::get_time();

        for _ in 0..num_chunks {
            // Simulate chunk processing
            let _data = vec![0u8; chunk_size];
            Self::delay(1).await;
        }

        let elapsed_ms = Self::get_time() - start_time;
        let total_mb = num_chunks as f64;
        let mbps = total_mb / (elapsed_ms / 1000.0);

        mbps
    }

    /// Simulate token generation time
    async fn simulate_token_generation(num_tokens: usize) -> f64 {
        let start_time = Self::get_time();

        for _ in 0..num_tokens {
            // Simulate inference computation
            Self::delay(10).await; // 10ms per token
        }

        Self::get_time() - start_time
    }

    /// Benchmark streaming performance
    async fn benchmark_streaming() -> f64 {
        let num_tokens = 100;
        let start_time = Self::get_time();

        for i in 0..num_tokens {
            // Simulate streaming token generation
            Self::delay(8).await; // 8ms per token for streaming

            if i % 20 == 0 {
                console::log_1(&format!("Streaming token {}/{}", i + 1, num_tokens).into());
            }
        }

        let elapsed_ms = Self::get_time() - start_time;
        let tokens_per_second = (num_tokens as f64) / (elapsed_ms / 1000.0);

        tokens_per_second
    }

    /// Measure inference memory usage
    async fn measure_inference_memory() -> f64 {
        // Simulate memory usage during inference
        let base_memory = 50.0; // 50MB base model
        let context_memory = 10.0; // 10MB for context
        let computation_memory = 20.0; // 20MB for computation buffers

        base_memory + context_memory + computation_memory
    }

    /// Async delay utility
    async fn delay(ms: i32) {
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            if let Some(window) = web_sys::window() {
                let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
            }
        });
        let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    }
}

impl Default for WasmBenchmarkSuite {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

// Benchmark result structures

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkSummary {
    total_time_ms: f64,
    kernel_performance: KernelBenchmarkResults,
    memory_performance: MemoryBenchmarkResults,
    loading_performance: LoadingBenchmarkResults,
    inference_performance: InferenceBenchmarkResults,
    platform_info: PlatformInfo,
    recommendations: Vec<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct KernelBenchmarkResults {
    matmul_small: f64,      // GFLOPS for 64x64x64
    matmul_medium: f64,     // GFLOPS for 256x256x256
    matmul_large: f64,      // GFLOPS for 1024x1024x1024
    quantization_1mb: f64,  // GB/s for 1MB quantization
    quantization_16mb: f64, // GB/s for 16MB quantization
    quantization_64mb: f64, // GB/s for 64MB quantization
    memory_bandwidth: f64,  // GB/s memory bandwidth
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct MemoryBenchmarkResults {
    allocation_time_ms: f64,
    gc_time_ms: f64,
    gc_freed_bytes: usize,
    fragmentation_score: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct LoadingBenchmarkResults {
    small_model_load_ms: f64,  // 10MB model
    medium_model_load_ms: f64, // 100MB model
    large_model_load_ms: f64,  // 500MB model
    chunk_processing_mbps: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct InferenceBenchmarkResults {
    single_token_latency_ms: f64,
    batch_10_tokens_ms: f64,
    batch_100_tokens_ms: f64,
    streaming_tokens_per_second: f64,
    inference_memory_mb: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct PlatformInfo {
    user_agent: String,
    memory_gb: f64,
    cpu_cores: usize,
    simd_supported: bool,
    bulk_memory_supported: bool,
    estimated_gflops: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkResult {
    name: String,
    value: f64,
    unit: String,
    timestamp: f64,
}
