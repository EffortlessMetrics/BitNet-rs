# Performance Testing and Benchmarking Guide

## Introduction

This guide covers performance testing and benchmarking in the BitNet.rs testing framework. It includes techniques for measuring performance, establishing baselines, detecting regressions, and optimizing critical paths.

## Overview

Performance testing in BitNet.rs focuses on:
- **Throughput**: Tokens processed per second
- **Latency**: Time to first token and total inference time
- **Memory Usage**: Peak and average memory consumption
- **Resource Efficiency**: CPU utilization and cache performance
- **Scalability**: Performance under different loads and configurations

## Performance Testing Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Performance Testing Framework                   │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│ │   Benchmarks    │ │   Profiling     │ │   Monitoring    │   │
│ │                 │ │                 │ │                 │   │
│ │ • Micro         │ │ • CPU profiling │ │ • Real-time     │   │
│ │ • Integration   │ │ • Memory        │ │ • Historical    │   │
│ │ • Regression    │ │ • I/O analysis  │ │ • Alerting      │   │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│ │   Baselines     │ │   Comparison    │ │   Reporting     │   │
│ │                 │ │                 │ │                 │   │
│ │ • Historical    │ │ • Rust vs C++   │ │ • Dashboards    │   │
│ │ • Platform      │ │ • Version diff  │ │ • Trends        │   │
│ │ • Configuration │ │ • Regression    │ │ • Alerts        │   │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Setting Up Performance Testing

### Prerequisites

1. **Install benchmarking tools:**
```bash
# Criterion for Rust benchmarks
cargo install cargo-criterion

# Flamegraph for profiling
cargo install flamegraph

# Hyperfine for command-line benchmarking
cargo install hyperfine

# System monitoring tools
sudo apt-get install htop iotop sysstat  # Linux
brew install htop  # macOS
```

2. **Configure system for benchmarking:**
```bash
# Disable CPU frequency scaling (Linux)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set CPU affinity for consistent results
taskset -c 0-3 cargo bench

# Disable swap to avoid memory performance variations
sudo swapoff -a
```

3. **Set up performance test configuration:**
```toml
# tests/performance.toml
[performance]
enabled = true
baseline_dir = "benchmarks/baselines"
report_dir = "target/performance-reports"
cpu_affinity = [0, 1, 2, 3]
warmup_iterations = 5
measurement_iterations = 100

[models]
small = { path = "fixtures/small_model.gguf", max_tokens = 100 }
medium = { path = "fixtures/medium_model.gguf", max_tokens = 200 }
large = { path = "fixtures/large_model.gguf", max_tokens = 500 }

[scenarios]
quick_response = { max_tokens = 10, temperature = 0.0 }
creative_generation = { max_tokens = 100, temperature = 0.8 }
long_context = { max_tokens = 500, temperature = 0.5 }
```

## Micro-Benchmarks

### Basic Benchmark Structure

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use bitnet_models::BitNetModel;
use std::time::Duration;

fn benchmark_model_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_loading");
    
    // Configure benchmark parameters
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    
    let model_sizes = vec!["small", "medium", "large"];
    
    for size in model_sizes {
        let model_path = format!("fixtures/{}_model.gguf", size);
        
        group.bench_with_input(
            BenchmarkId::new("load_model", size),
            &model_path,
            |b, path| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        let model = BitNetModel::from_file(black_box(path)).await.unwrap();
                        black_box(model);
                    });
            },
        );
    }
    
    group.finish();
}

fn benchmark_tokenization(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let model = rt.block_on(async {
        BitNetModel::from_file("fixtures/small_model.gguf").await.unwrap()
    });
    
    let mut group = c.benchmark_group("tokenization");
    
    let test_inputs = vec![
        ("short", "Hello world"),
        ("medium", "The quick brown fox jumps over the lazy dog. ".repeat(10)),
        ("long", "Lorem ipsum dolor sit amet. ".repeat(100)),
    ];
    
    for (name, input) in test_inputs {
        group.bench_with_input(
            BenchmarkId::new("tokenize", name),
            &input,
            |b, text| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        let tokens = model.tokenize(black_box(text)).await.unwrap();
                        black_box(tokens);
                    });
            },
        );
    }
    
    group.finish();
}

fn benchmark_inference(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let model = rt.block_on(async {
        BitNetModel::from_file("fixtures/small_model.gguf").await.unwrap()
    });
    
    let mut group = c.benchmark_group("inference");
    group.throughput(criterion::Throughput::Elements(1)); // 1 inference per iteration
    
    let test_cases = vec![
        ("greedy", InferenceConfig { temperature: 0.0, max_tokens: 50, ..Default::default() }),
        ("sampling", InferenceConfig { temperature: 0.7, max_tokens: 50, ..Default::default() }),
        ("creative", InferenceConfig { temperature: 1.0, top_p: 0.9, max_tokens: 50, ..Default::default() }),
    ];
    
    for (name, config) in test_cases {
        let tokens = vec![1, 2, 3, 4, 5]; // Simple input tokens
        
        group.bench_with_input(
            BenchmarkId::new("generate", name),
            &(tokens.clone(), config.clone()),
            |b, (tokens, config)| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        let result = model.generate_from_tokens(
                            black_box(tokens),
                            black_box(config)
                        ).await.unwrap();
                        black_box(result);
                    });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_model_loading, benchmark_tokenization, benchmark_inference);
criterion_main!(benches);
```

### Memory Benchmarks

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

// Custom allocator to track memory usage
struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let size = layout.size();
            let current = ALLOCATED.fetch_add(size, Ordering::Relaxed) + size;
            
            // Update peak if necessary
            let mut peak = PEAK_ALLOCATED.load(Ordering::Relaxed);
            while current > peak {
                match PEAK_ALLOCATED.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(new_peak) => peak = new_peak,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    group.bench_function("model_loading_memory", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            
            for _ in 0..iters {
                // Reset memory tracking
                ALLOCATED.store(0, Ordering::Relaxed);
                PEAK_ALLOCATED.store(0, Ordering::Relaxed);
                
                let rt = tokio::runtime::Runtime::new().unwrap();
                let _model = rt.block_on(async {
                    BitNetModel::from_file("fixtures/small_model.gguf").await.unwrap()
                });
                
                let peak_memory = PEAK_ALLOCATED.load(Ordering::Relaxed);
                println!("Peak memory usage: {} MB", peak_memory / 1024 / 1024);
            }
            
            start.elapsed()
        });
    });
    
    group.finish();
}
```

## Integration Benchmarks

### End-to-End Performance Tests

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::time::Instant;

fn benchmark_complete_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_workflow");
    group.sample_size(20); // Fewer samples for longer tests
    
    let scenarios = vec![
        ("chat_response", "Hello, how can I help you today?", 50),
        ("code_generation", "Write a function to calculate fibonacci numbers", 200),
        ("story_generation", "Once upon a time in a distant galaxy", 300),
    ];
    
    for (name, prompt, max_tokens) in scenarios {
        group.bench_with_input(
            BenchmarkId::new("workflow", name),
            &(prompt, max_tokens),
            |b, (prompt, max_tokens)| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter_custom(|iters| async move {
                        let start = Instant::now();
                        
                        for _ in 0..iters {
                            // Complete workflow: load model, tokenize, generate, decode
                            let model = BitNetModel::from_file("fixtures/medium_model.gguf")
                                .await.unwrap();
                            
                            let tokens = model.tokenize(prompt).await.unwrap();
                            
                            let config = InferenceConfig {
                                max_tokens: *max_tokens,
                                temperature: 0.7,
                                ..Default::default()
                            };
                            
                            let result = model.generate_from_tokens(&tokens, &config)
                                .await.unwrap();
                            
                            let _output = model.decode(&result.tokens).await.unwrap();
                        }
                        
                        start.elapsed()
                    });
            },
        );
    }
    
    group.finish();
}

fn benchmark_concurrent_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_inference");
    
    let concurrency_levels = vec![1, 2, 4, 8];
    
    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::new("concurrent", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter_custom(|iters| async move {
                        let start = Instant::now();
                        
                        for _ in 0..iters {
                            let model = std::sync::Arc::new(
                                BitNetModel::from_file("fixtures/small_model.gguf")
                                    .await.unwrap()
                            );
                            
                            let tasks: Vec<_> = (0..concurrency)
                                .map(|i| {
                                    let model = model.clone();
                                    tokio::spawn(async move {
                                        let prompt = format!("Task {} prompt", i);
                                        let tokens = model.tokenize(&prompt).await.unwrap();
                                        let config = InferenceConfig {
                                            max_tokens: 20,
                                            temperature: 0.0,
                                            ..Default::default()
                                        };
                                        model.generate_from_tokens(&tokens, &config).await.unwrap()
                                    })
                                })
                                .collect();
                            
                            futures::future::join_all(tasks).await;
                        }
                        
                        start.elapsed()
                    });
            },
        );
    }
    
    group.finish();
}

criterion_group!(integration_benches, benchmark_complete_workflow, benchmark_concurrent_inference);
criterion_main!(integration_benches);
```

## Performance Profiling

### CPU Profiling

```bash
# Generate flamegraph for specific benchmark
cargo flamegraph --bench inference_bench

# Profile specific test
cargo flamegraph --test performance_tests -- test_inference_performance

# Profile with specific features
cargo flamegraph --features "gpu,optimized" --bench full_benchmark
```

### Memory Profiling

```bash
# Use Valgrind for memory analysis (Linux)
valgrind --tool=massif --massif-out-file=massif.out cargo bench
ms_print massif.out > memory_profile.txt

# Use Heaptrack (Linux)
heaptrack cargo bench
heaptrack_gui heaptrack.cargo.*.gz

# Use Instruments (macOS)
instruments -t "Allocations" cargo bench
```

### Custom Profiling Integration

```rust
use std::time::{Duration, Instant};
use std::collections::HashMap;

pub struct PerformanceProfiler {
    measurements: HashMap<String, Vec<Duration>>,
    memory_measurements: HashMap<String, Vec<usize>>,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            measurements: HashMap::new(),
            memory_measurements: HashMap::new(),
        }
    }
    
    pub fn time_operation<F, R>(&mut self, name: &str, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();
        
        self.measurements
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
        
        result
    }
    
    pub async fn time_async_operation<F, Fut, R>(&mut self, name: &str, operation: F) -> R
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let start = Instant::now();
        let result = operation().await;
        let duration = start.elapsed();
        
        self.measurements
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
        
        result
    }
    
    pub fn record_memory_usage(&mut self, name: &str, bytes: usize) {
        self.memory_measurements
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(bytes);
    }
    
    pub fn generate_report(&self) -> PerformanceReport {
        let mut operations = Vec::new();
        
        for (name, durations) in &self.measurements {
            let total: Duration = durations.iter().sum();
            let avg = total / durations.len() as u32;
            let min = *durations.iter().min().unwrap();
            let max = *durations.iter().max().unwrap();
            
            operations.push(OperationStats {
                name: name.clone(),
                count: durations.len(),
                total_time: total,
                average_time: avg,
                min_time: min,
                max_time: max,
            });
        }
        
        PerformanceReport { operations }
    }
}

#[derive(Debug)]
pub struct PerformanceReport {
    pub operations: Vec<OperationStats>,
}

#[derive(Debug)]
pub struct OperationStats {
    pub name: String,
    pub count: usize,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

// Usage in tests
#[tokio::test]
async fn test_with_profiling() {
    let mut profiler = PerformanceProfiler::new();
    
    let model = profiler.time_async_operation("model_loading", || async {
        BitNetModel::from_file("fixtures/test_model.gguf").await.unwrap()
    }).await;
    
    let tokens = profiler.time_async_operation("tokenization", || async {
        model.tokenize("Hello world").await.unwrap()
    }).await;
    
    let _result = profiler.time_async_operation("inference", || async {
        model.generate_from_tokens(&tokens, &InferenceConfig::default()).await.unwrap()
    }).await;
    
    let report = profiler.generate_report();
    println!("{:#?}", report);
    
    // Assert performance requirements
    for op in &report.operations {
        match op.name.as_str() {
            "model_loading" => assert!(op.average_time < Duration::from_secs(5)),
            "tokenization" => assert!(op.average_time < Duration::from_millis(100)),
            "inference" => assert!(op.average_time < Duration::from_secs(2)),
            _ => {}
        }
    }
}
```

## Baseline Management

### Establishing Baselines

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub platform: PlatformInfo,
    pub benchmarks: HashMap<String, BenchmarkBaseline>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PlatformInfo {
    pub os: String,
    pub arch: String,
    pub cpu_model: String,
    pub memory_gb: u64,
    pub rust_version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkBaseline {
    pub name: String,
    pub average_time_ns: u64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_bytes: u64,
    pub samples: usize,
}

impl PerformanceBaseline {
    pub fn new(version: String) -> Self {
        Self {
            version,
            timestamp: chrono::Utc::now(),
            platform: PlatformInfo::current(),
            benchmarks: HashMap::new(),
        }
    }
    
    pub fn add_benchmark(&mut self, benchmark: BenchmarkBaseline) {
        self.benchmarks.insert(benchmark.name.clone(), benchmark);
    }
    
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
    
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let baseline = serde_json::from_str(&json)?;
        Ok(baseline)
    }
    
    pub fn compare_with(&self, other: &PerformanceBaseline) -> ComparisonReport {
        let mut comparisons = Vec::new();
        
        for (name, current) in &self.benchmarks {
            if let Some(baseline) = other.benchmarks.get(name) {
                let time_ratio = current.average_time_ns as f64 / baseline.average_time_ns as f64;
                let throughput_ratio = current.throughput_ops_per_sec / baseline.throughput_ops_per_sec;
                let memory_ratio = current.memory_usage_bytes as f64 / baseline.memory_usage_bytes as f64;
                
                comparisons.push(BenchmarkComparison {
                    name: name.clone(),
                    time_ratio,
                    throughput_ratio,
                    memory_ratio,
                    is_regression: time_ratio > 1.1 || memory_ratio > 1.2, // 10% time or 20% memory regression
                });
            }
        }
        
        ComparisonReport {
            current_version: self.version.clone(),
            baseline_version: other.version.clone(),
            comparisons,
        }
    }
}

#[derive(Debug)]
pub struct ComparisonReport {
    pub current_version: String,
    pub baseline_version: String,
    pub comparisons: Vec<BenchmarkComparison>,
}

#[derive(Debug)]
pub struct BenchmarkComparison {
    pub name: String,
    pub time_ratio: f64,
    pub throughput_ratio: f64,
    pub memory_ratio: f64,
    pub is_regression: bool,
}

impl PlatformInfo {
    pub fn current() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_model: get_cpu_model(),
            memory_gb: get_total_memory_gb(),
            rust_version: get_rust_version(),
        }
    }
}

// Helper functions (platform-specific implementations)
fn get_cpu_model() -> String {
    // Implementation depends on platform
    "Unknown CPU".to_string()
}

fn get_total_memory_gb() -> u64 {
    // Implementation depends on platform
    8 // Default fallback
}

fn get_rust_version() -> String {
    std::env::var("RUSTC_VERSION")
        .unwrap_or_else(|_| "Unknown".to_string())
}
```

### Automated Baseline Updates

```bash
#!/bin/bash
# scripts/update_baselines.sh

set -e

echo "Updating performance baselines..."

# Get current version
VERSION=$(git describe --tags --always)
BASELINE_DIR="benchmarks/baselines"
BASELINE_FILE="$BASELINE_DIR/baseline_$VERSION.json"

# Create baseline directory
mkdir -p "$BASELINE_DIR"

# Run benchmarks and generate baseline
cargo bench --no-default-features --features cpu --bench performance_suite -- --save-baseline "$BASELINE_FILE"

# Compare with previous baseline
PREVIOUS_BASELINE=$(ls -t "$BASELINE_DIR"/baseline_*.json | head -2 | tail -1)
if [ -n "$PREVIOUS_BASELINE" ] && [ "$PREVIOUS_BASELINE" != "$BASELINE_FILE" ]; then
    echo "Comparing with previous baseline: $PREVIOUS_BASELINE"
    cargo run --bin compare_baselines -- "$BASELINE_FILE" "$PREVIOUS_BASELINE"
fi

echo "Baseline updated: $BASELINE_FILE"
```

## Regression Detection

### Automated Regression Testing

```rust
#[tokio::test]
async fn test_performance_regression() {
    let current_results = run_performance_suite().await;
    
    // Load baseline
    let baseline_path = "benchmarks/baselines/current_baseline.json";
    let baseline = PerformanceBaseline::load_from_file(baseline_path)
        .expect("Failed to load performance baseline");
    
    // Compare results
    let comparison = current_results.compare_with(&baseline);
    
    // Check for regressions
    let mut regressions = Vec::new();
    for comp in &comparison.comparisons {
        if comp.is_regression {
            regressions.push(comp);
        }
    }
    
    if !regressions.is_empty() {
        let mut error_msg = String::from("Performance regressions detected:\n");
        for regression in &regressions {
            error_msg.push_str(&format!(
                "  {}: {:.2}x slower, {:.2}x more memory\n",
                regression.name,
                regression.time_ratio,
                regression.memory_ratio
            ));
        }
        panic!("{}", error_msg);
    }
    
    println!("All performance benchmarks within acceptable ranges");
}

async fn run_performance_suite() -> PerformanceBaseline {
    let mut baseline = PerformanceBaseline::new("current".to_string());
    
    // Model loading benchmark
    let model_loading_time = benchmark_model_loading().await;
    baseline.add_benchmark(BenchmarkBaseline {
        name: "model_loading".to_string(),
        average_time_ns: model_loading_time.as_nanos() as u64,
        throughput_ops_per_sec: 1.0 / model_loading_time.as_secs_f64(),
        memory_usage_bytes: get_peak_memory_usage(),
        samples: 10,
    });
    
    // Inference benchmark
    let inference_time = benchmark_inference().await;
    baseline.add_benchmark(BenchmarkBaseline {
        name: "inference".to_string(),
        average_time_ns: inference_time.as_nanos() as u64,
        throughput_ops_per_sec: 50.0 / inference_time.as_secs_f64(), // 50 tokens generated
        memory_usage_bytes: get_peak_memory_usage(),
        samples: 100,
    });
    
    baseline
}
```

### CI Integration for Performance Monitoring

```yaml
# .github/workflows/performance.yml
name: Performance Monitoring

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  performance:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
    
    - name: Install benchmarking tools
      run: |
        cargo install cargo-criterion
        cargo install flamegraph
    
    - name: Cache benchmarks
      uses: actions/cache@v3
      with:
        path: |
          target/criterion
          benchmarks/baselines
        key: performance-${{ runner.os }}-${{ hashFiles('Cargo.lock') }}
    
    - name: Run performance benchmarks
      run: |
        cargo bench --no-default-features --features cpu --bench performance_suite
        cargo test --no-default-features --features cpu --test performance_regression_tests
    
    - name: Generate performance report
      run: |
        cargo run --bin generate_performance_report
    
    - name: Upload performance artifacts
      uses: actions/upload-artifact@v3
      with:
        name: performance-reports
        path: |
          target/criterion/
          target/performance-reports/
    
    - name: Comment PR with performance results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('target/performance-reports/summary.md', 'utf8');
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: `## Performance Report\n\n${report}`
          });
```

## Performance Optimization

### Identifying Bottlenecks

```rust
use std::time::Instant;

#[tokio::test]
async fn identify_bottlenecks() {
    let mut profiler = PerformanceProfiler::new();
    
    // Profile each component separately
    let model = profiler.time_async_operation("model_loading", || async {
        BitNetModel::from_file("fixtures/large_model.gguf").await.unwrap()
    }).await;
    
    let input_text = "Generate a comprehensive analysis of machine learning trends";
    
    let tokens = profiler.time_async_operation("tokenization", || async {
        model.tokenize(input_text).await.unwrap()
    }).await;
    
    // Profile different inference configurations
    let configs = vec![
        ("greedy", InferenceConfig { temperature: 0.0, max_tokens: 100, ..Default::default() }),
        ("sampling", InferenceConfig { temperature: 0.7, max_tokens: 100, ..Default::default() }),
        ("nucleus", InferenceConfig { temperature: 0.8, top_p: 0.9, max_tokens: 100, ..Default::default() }),
    ];
    
    for (name, config) in configs {
        profiler.time_async_operation(&format!("inference_{}", name), || async {
            model.generate_from_tokens(&tokens, &config).await.unwrap()
        }).await;
    }
    
    // Analyze results
    let report = profiler.generate_report();
    
    // Find the slowest operations
    let mut operations = report.operations;
    operations.sort_by(|a, b| b.average_time.cmp(&a.average_time));
    
    println!("Performance bottlenecks (slowest first):");
    for (i, op) in operations.iter().take(5).enumerate() {
        println!("{}. {}: {:?} (avg)", i + 1, op.name, op.average_time);
    }
    
    // Set performance targets
    for op in &operations {
        match op.name.as_str() {
            "model_loading" => {
                assert!(op.average_time < Duration::from_secs(10), 
                       "Model loading too slow: {:?}", op.average_time);
            }
            "tokenization" => {
                assert!(op.average_time < Duration::from_millis(50),
                       "Tokenization too slow: {:?}", op.average_time);
            }
            name if name.starts_with("inference_") => {
                assert!(op.average_time < Duration::from_secs(5),
                       "Inference too slow: {:?}", op.average_time);
            }
            _ => {}
        }
    }
}
```

### Memory Optimization Testing

```rust
#[tokio::test]
async fn test_memory_optimization() {
    let memory_tracker = MemoryTracker::new();
    
    // Test memory usage patterns
    let baseline_memory = memory_tracker.current_usage();
    
    {
        let model = BitNetModel::from_file("fixtures/medium_model.gguf").await.unwrap();
        let model_memory = memory_tracker.current_usage() - baseline_memory;
        
        println!("Model memory usage: {} MB", model_memory / 1024 / 1024);
        assert!(model_memory < 2 * 1024 * 1024 * 1024, "Model uses too much memory"); // 2GB limit
        
        // Test inference memory usage
        let inference_baseline = memory_tracker.current_usage();
        
        let tokens = model.tokenize("Test input for memory analysis").await.unwrap();
        let _result = model.generate_from_tokens(&tokens, &InferenceConfig {
            max_tokens: 100,
            ..Default::default()
        }).await.unwrap();
        
        let inference_memory = memory_tracker.current_usage() - inference_baseline;
        println!("Inference memory overhead: {} MB", inference_memory / 1024 / 1024);
        assert!(inference_memory < 500 * 1024 * 1024, "Inference uses too much additional memory"); // 500MB limit
    }
    
    // Test memory cleanup
    tokio::time::sleep(Duration::from_millis(100)).await; // Allow cleanup
    let final_memory = memory_tracker.current_usage();
    let memory_leak = final_memory - baseline_memory;
    
    println!("Potential memory leak: {} MB", memory_leak / 1024 / 1024);
    assert!(memory_leak < 10 * 1024 * 1024, "Significant memory leak detected"); // 10MB tolerance
}
```

## Reporting and Visualization

### Performance Dashboard

```rust
use serde_json::json;
use std::fs;

pub fn generate_performance_dashboard(results: &[PerformanceBaseline]) -> Result<(), Box<dyn std::error::Error>> {
    let mut chart_data = Vec::new();
    
    for result in results {
        for (name, benchmark) in &result.benchmarks {
            chart_data.push(json!({
                "timestamp": result.timestamp,
                "version": result.version,
                "benchmark": name,
                "average_time_ms": benchmark.average_time_ns as f64 / 1_000_000.0,
                "throughput": benchmark.throughput_ops_per_sec,
                "memory_mb": benchmark.memory_usage_bytes as f64 / 1024.0 / 1024.0
            }));
        }
    }
    
    let html_template = include_str!("templates/performance_dashboard.html");
    let html_content = html_template.replace("{{CHART_DATA}}", &serde_json::to_string(&chart_data)?);
    
    fs::write("target/performance-reports/dashboard.html", html_content)?;
    
    println!("Performance dashboard generated: target/performance-reports/dashboard.html");
    Ok(())
}
```

This comprehensive performance testing guide provides the foundation for maintaining high performance standards in BitNet.rs through systematic benchmarking, profiling, and regression detection.