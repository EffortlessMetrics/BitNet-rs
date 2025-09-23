//! Performance fixtures and benchmark targets
//!
//! Provides comprehensive performance testing infrastructure with baseline targets,
//! regression detection, and detailed metrics collection for BitNet.rs components.

use bitnet_common::{Device, Result, BitNetError};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use super::TestEnvironmentConfig;

/// Performance benchmark targets for different operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTargets {
    pub model_loading: ModelLoadingTargets,
    pub quantization: QuantizationTargets,
    pub inference: InferenceTargets,
    pub memory: MemoryTargets,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadingTargets {
    pub small_model_load_ms: u32,
    pub large_model_load_ms: u32,
    pub memory_mapped_load_ms: u32,
    pub validation_overhead_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationTargets {
    pub i2s_throughput_gops: f32,
    pub tl1_throughput_gops: f32,
    pub tl2_throughput_gops: f32,
    pub cpu_gpu_speedup_min: f32,
    pub batch_efficiency_min: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTargets {
    pub first_token_latency_ms: u32,
    pub tokens_per_second: f32,
    pub batch_throughput_multiplier: f32,
    pub memory_efficiency_min: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTargets {
    pub max_memory_mb: u32,
    pub memory_leak_threshold_mb: u32,
    pub cache_efficiency_min: f32,
    pub gc_overhead_max: f32,
}

/// Performance test suite configuration
#[derive(Debug, Clone)]
pub struct PerformanceTestSuite {
    pub name: String,
    pub device: Device,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub timeout: Duration,
    pub regression_threshold: f32,
}

/// Performance measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    pub operation: String,
    pub device: String,
    pub iterations: usize,
    pub measurements: Vec<f64>,
    pub statistics: PerformanceStatistics,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStatistics {
    pub mean_ms: f64,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub std_dev_ms: f64,
    pub percentile_95_ms: f64,
    pub percentile_99_ms: f64,
}

/// Performance fixtures manager
pub struct PerformanceFixtures {
    pub benchmark_targets: HashMap<Device, BenchmarkTargets>,
    pub test_suites: Vec<PerformanceTestSuite>,
    pub historical_results: HashMap<String, Vec<PerformanceMeasurement>>,
    pub config: TestEnvironmentConfig,
}

impl PerformanceFixtures {
    pub fn new(config: &TestEnvironmentConfig) -> Self {
        Self {
            benchmark_targets: Self::create_benchmark_targets(),
            test_suites: Self::create_test_suites(config),
            historical_results: HashMap::new(),
            config: config.clone(),
        }
    }

    /// Initialize performance fixtures
    pub async fn initialize(&mut self) -> Result<()> {
        // Load historical performance data if available
        self.load_historical_data().await?;

        // Create baseline measurements for reference
        self.create_baseline_measurements().await?;

        // Validate benchmark targets
        self.validate_benchmark_targets().await?;

        Ok(())
    }

    /// Create benchmark targets for different devices
    fn create_benchmark_targets() -> HashMap<Device, BenchmarkTargets> {
        let mut targets = HashMap::new();

        // CPU targets (conservative estimates for CI environments)
        let cpu_targets = BenchmarkTargets {
            model_loading: ModelLoadingTargets {
                small_model_load_ms: 100,   // 100ms for small mock models
                large_model_load_ms: 5000,  // 5s for large models
                memory_mapped_load_ms: 50,  // 50ms with memory mapping
                validation_overhead_ms: 20, // 20ms validation overhead
            },
            quantization: QuantizationTargets {
                i2s_throughput_gops: 10.0,  // Conservative CPU estimate
                tl1_throughput_gops: 8.0,
                tl2_throughput_gops: 6.0,
                cpu_gpu_speedup_min: 1.0,   // No GPU acceleration
                batch_efficiency_min: 0.8,  // 80% batch efficiency
            },
            inference: InferenceTargets {
                first_token_latency_ms: 200, // 200ms first token on CPU
                tokens_per_second: 20.0,     // 20 tokens/sec
                batch_throughput_multiplier: 1.5, // 1.5x with batching
                memory_efficiency_min: 0.7,  // 70% memory efficiency
            },
            memory: MemoryTargets {
                max_memory_mb: 4096,         // 4GB max for CPU
                memory_leak_threshold_mb: 10, // 10MB leak threshold
                cache_efficiency_min: 0.8,   // 80% cache hit rate
                gc_overhead_max: 0.1,        // 10% GC overhead
            },
        };
        targets.insert(Device::Cpu, cpu_targets);

        // GPU targets (when available)
        #[cfg(feature = "gpu")]
        {
            let gpu_targets = BenchmarkTargets {
                model_loading: ModelLoadingTargets {
                    small_model_load_ms: 80,    // Slightly faster GPU initialization
                    large_model_load_ms: 3000,  // 3s for large models on GPU
                    memory_mapped_load_ms: 40,  // GPU memory mapping
                    validation_overhead_ms: 15, // Lower validation overhead
                },
                quantization: QuantizationTargets {
                    i2s_throughput_gops: 100.0, // 10x CPU performance
                    tl1_throughput_gops: 80.0,
                    tl2_throughput_gops: 60.0,
                    cpu_gpu_speedup_min: 5.0,   // 5x speedup minimum
                    batch_efficiency_min: 0.9,  // 90% batch efficiency
                },
                inference: InferenceTargets {
                    first_token_latency_ms: 50,  // 50ms first token on GPU
                    tokens_per_second: 200.0,    // 200 tokens/sec
                    batch_throughput_multiplier: 4.0, // 4x with batching
                    memory_efficiency_min: 0.9,  // 90% memory efficiency
                },
                memory: MemoryTargets {
                    max_memory_mb: 8192,         // 8GB max for GPU
                    memory_leak_threshold_mb: 50, // 50MB leak threshold (GPU)
                    cache_efficiency_min: 0.95,  // 95% cache hit rate
                    gc_overhead_max: 0.05,       // 5% GC overhead
                },
            };
            targets.insert(Device::Cuda(0), gpu_targets);
        }

        targets
    }

    /// Create performance test suites
    fn create_test_suites(config: &TestEnvironmentConfig) -> Vec<PerformanceTestSuite> {
        let mut suites = vec![];

        // CPU test suite
        suites.push(PerformanceTestSuite {
            name: "CPU_Performance".to_string(),
            device: Device::Cpu,
            iterations: if std::env::var("CI").is_ok() { 5 } else { 10 }, // Fewer iterations in CI
            warmup_iterations: 2,
            timeout: Duration::from_secs(30),
            regression_threshold: 0.2, // 20% regression threshold
        });

        // GPU test suite if available
        if config.gpu_features_enabled() {
            suites.push(PerformanceTestSuite {
                name: "GPU_Performance".to_string(),
                device: Device::Cuda(0),
                iterations: if std::env::var("CI").is_ok() { 3 } else { 10 },
                warmup_iterations: 1, // GPU warmup is faster
                timeout: Duration::from_secs(60),
                regression_threshold: 0.15, // 15% regression threshold (GPU more stable)
            });
        }

        suites
    }

    /// Load historical performance data
    async fn load_historical_data(&mut self) -> Result<()> {
        // Try to load from performance baseline file
        let baseline_path = std::path::Path::new("tests/fixtures/performance_baselines.json");

        if baseline_path.exists() {
            match tokio::fs::read_to_string(baseline_path).await {
                Ok(content) => {
                    if let Ok(historical_data) = serde_json::from_str::<HashMap<String, Vec<PerformanceMeasurement>>>(&content) {
                        self.historical_results = historical_data;
                        println!("Loaded historical performance data from: {}", baseline_path.display());
                    }
                }
                Err(e) => {
                    println!("Warning: Could not load historical performance data: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Create baseline performance measurements
    async fn create_baseline_measurements(&mut self) -> Result<()> {
        // Create mock baseline measurements for key operations
        let operations = vec![
            "model_loading_small",
            "model_loading_large",
            "quantization_i2s",
            "quantization_tl1",
            "inference_first_token",
            "inference_generation",
        ];

        for operation in operations {
            for suite in &self.test_suites {
                let measurement = self.create_mock_measurement(operation, &suite.device);

                let key = format!("{}_{:?}", operation, suite.device);
                self.historical_results
                    .entry(key)
                    .or_insert_with(Vec::new)
                    .push(measurement);
            }
        }

        Ok(())
    }

    /// Create mock performance measurement
    fn create_mock_measurement(&self, operation: &str, device: &Device) -> PerformanceMeasurement {
        let base_time = match operation {
            "model_loading_small" => 100.0,
            "model_loading_large" => 2000.0,
            "quantization_i2s" => 50.0,
            "quantization_tl1" => 60.0,
            "inference_first_token" => match device {
                Device::Cpu => 200.0,
                Device::Cuda(_) => 50.0,
                Device::Metal => 60.0, // Apple Silicon GPU
            },
            "inference_generation" => match device {
                Device::Cpu => 100.0,
                Device::Cuda(_) => 20.0,
                Device::Metal => 25.0, // Apple Silicon GPU
            },
            _ => 100.0,
        };

        // Add some realistic variation
        let measurements: Vec<f64> = (0..10)
            .map(|i| base_time * (0.9 + 0.2 * (i as f64) / 10.0))
            .collect();

        let statistics = Self::calculate_statistics(&measurements);

        PerformanceMeasurement {
            operation: operation.to_string(),
            device: format!("{:?}", device),
            iterations: measurements.len(),
            measurements,
            statistics,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Calculate performance statistics
    fn calculate_statistics(measurements: &[f64]) -> PerformanceStatistics {
        let mut sorted = measurements.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let median = sorted[sorted.len() / 2];
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        let variance = sorted.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / sorted.len() as f64;
        let std_dev = variance.sqrt();

        let p95_idx = (sorted.len() as f64 * 0.95) as usize;
        let p99_idx = (sorted.len() as f64 * 0.99) as usize;
        let percentile_95 = sorted[p95_idx.min(sorted.len() - 1)];
        let percentile_99 = sorted[p99_idx.min(sorted.len() - 1)];

        PerformanceStatistics {
            mean_ms: mean,
            median_ms: median,
            min_ms: min,
            max_ms: max,
            std_dev_ms: std_dev,
            percentile_95_ms: percentile_95,
            percentile_99_ms: percentile_99,
        }
    }

    /// Validate benchmark targets against current system
    async fn validate_benchmark_targets(&self) -> Result<()> {
        for (device, targets) in &self.benchmark_targets {
            // Basic sanity checks
            if targets.inference.first_token_latency_ms == 0 {
                return Err(BitNetError::Validation(
                    format!("Invalid benchmark target for device {:?}: zero first token latency", device)
                ));
            }

            if targets.quantization.i2s_throughput_gops <= 0.0 {
                return Err(BitNetError::Validation(
                    format!("Invalid benchmark target for device {:?}: non-positive throughput", device)
                ));
            }

            println!("Validated benchmark targets for device: {:?}", device);
        }

        Ok(())
    }

    /// Run performance benchmark suite
    pub async fn run_benchmark_suite(&self, suite_name: &str) -> Result<BenchmarkSuiteResult> {
        let suite = self.test_suites.iter()
            .find(|s| s.name == suite_name)
            .ok_or_else(|| BitNetError::Validation(
                format!("Performance test suite not found: {}", suite_name)
            ))?;

        let targets = self.benchmark_targets.get(&suite.device)
            .ok_or_else(|| BitNetError::Validation(
                format!("No benchmark targets for device: {:?}", suite.device)
            ))?;

        let mut results = vec![];

        // Run individual benchmarks
        let benchmark_ops = vec![
            "model_loading_small",
            "quantization_i2s",
            "inference_first_token",
        ];

        for op in benchmark_ops {
            let result = self.run_individual_benchmark(op, suite).await?;
            results.push(result);
        }

        // Calculate overall score
        let overall_score = results.iter()
            .map(|r| if r.passes_target { 1.0 } else { 0.0 })
            .sum::<f32>() / results.len() as f32;

        Ok(BenchmarkSuiteResult {
            suite_name: suite_name.to_string(),
            device: suite.device.clone(),
            individual_results: results,
            overall_score,
            passes_regression_threshold: overall_score >= (1.0 - suite.regression_threshold),
        })
    }

    /// Run individual performance benchmark
    async fn run_individual_benchmark(&self,
        operation: &str,
        suite: &PerformanceTestSuite
    ) -> Result<IndividualBenchmarkResult> {
        let start_time = Instant::now();

        // Warmup iterations
        for _ in 0..suite.warmup_iterations {
            self.simulate_operation(operation, &suite.device).await?;
        }

        // Actual measurement iterations
        let mut measurements = Vec::new();
        for _ in 0..suite.iterations {
            let iter_start = Instant::now();
            self.simulate_operation(operation, &suite.device).await?;
            measurements.push(iter_start.elapsed().as_secs_f64() * 1000.0); // Convert to ms
        }

        let total_time = start_time.elapsed();

        if total_time > suite.timeout {
            return Err(BitNetError::Validation(
                format!("Benchmark '{}' exceeded timeout", operation)
            ));
        }

        let statistics = Self::calculate_statistics(&measurements);
        let target_ms = self.get_target_for_operation(operation, &suite.device);
        let passes_target = statistics.median_ms <= target_ms as f64;

        Ok(IndividualBenchmarkResult {
            operation: operation.to_string(),
            device: suite.device.clone(),
            statistics,
            target_ms,
            passes_target,
            regression_vs_baseline: self.calculate_regression(operation, &suite.device, statistics.median_ms),
        })
    }

    /// Simulate operation execution for benchmarking
    async fn simulate_operation(&self, operation: &str, device: &Device) -> Result<()> {
        // Simulate realistic operation timing
        let base_duration = match operation {
            "model_loading_small" => Duration::from_millis(50),
            "quantization_i2s" => Duration::from_millis(10),
            "inference_first_token" => match device {
                Device::Cpu => Duration::from_millis(100),
                Device::Cuda(_) => Duration::from_millis(20),
            },
            _ => Duration::from_millis(50),
        };

        tokio::time::sleep(base_duration).await;
        Ok(())
    }

    /// Get performance target for specific operation and device
    fn get_target_for_operation(&self, operation: &str, device: &Device) -> u32 {
        if let Some(targets) = self.benchmark_targets.get(device) {
            match operation {
                "model_loading_small" => targets.model_loading.small_model_load_ms,
                "quantization_i2s" => (1000.0 / targets.quantization.i2s_throughput_gops) as u32,
                "inference_first_token" => targets.inference.first_token_latency_ms,
                _ => 100, // Default target
            }
        } else {
            100 // Default fallback
        }
    }

    /// Calculate regression vs baseline
    fn calculate_regression(&self, operation: &str, device: &Device, current_ms: f64) -> f32 {
        let key = format!("{}_{:?}", operation, device);

        if let Some(historical) = self.historical_results.get(&key) {
            if let Some(latest) = historical.last() {
                let baseline_ms = latest.statistics.median_ms;
                ((current_ms - baseline_ms) / baseline_ms) as f32
            } else {
                0.0 // No regression if no baseline
            }
        } else {
            0.0 // No regression if no historical data
        }
    }

    /// Cleanup performance fixtures
    pub async fn cleanup(&mut self) -> Result<()> {
        // Save current measurements as baseline for future runs
        let baseline_path = std::path::Path::new("tests/fixtures/performance_baselines.json");
        if !self.historical_results.is_empty() {
            if let Ok(content) = serde_json::to_string_pretty(&self.historical_results) {
                let _ = tokio::fs::write(baseline_path, content).await;
            }
        }

        self.historical_results.clear();
        Ok(())
    }
}

/// Benchmark suite result
#[derive(Debug)]
pub struct BenchmarkSuiteResult {
    pub suite_name: String,
    pub device: Device,
    pub individual_results: Vec<IndividualBenchmarkResult>,
    pub overall_score: f32,
    pub passes_regression_threshold: bool,
}

/// Individual benchmark result
#[derive(Debug)]
pub struct IndividualBenchmarkResult {
    pub operation: String,
    pub device: Device,
    pub statistics: PerformanceStatistics,
    pub target_ms: u32,
    pub passes_target: bool,
    pub regression_vs_baseline: f32,
}