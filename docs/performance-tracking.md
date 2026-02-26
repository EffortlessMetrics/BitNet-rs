# Performance Tracking Infrastructure

This document describes the comprehensive performance tracking capabilities implemented in BitNet-rs, providing detailed insights into inference performance, resource usage, and optimization opportunities.

## Overview

The performance tracking infrastructure includes:

- **Real-time metrics collection**: Tracks latency, throughput, memory usage
- **Detailed timing breakdown**: Separate tracking for tokenization, forward pass, sampling
- **Cache performance monitoring**: Hit rates and memory efficiency tracking
- **Device memory tracking**: Comprehensive host and GPU memory monitoring
- **Environment-based configuration**: Support for deterministic testing and optimization
- **Validation and error handling**: Ensures metrics integrity and proper error reporting
- **Thread-safe statistics**: Concurrent access to performance data with Arc<Mutex> safety

## Core Components

### PerformanceMetrics

The `PerformanceMetrics` struct provides comprehensive performance insights:

```rust
use bitnet_inference::engine::PerformanceMetrics;

// Default metrics with validation
let metrics = PerformanceMetrics::default();
assert!(metrics.validate().is_ok());

// Check efficiency ratio (tokens per millisecond)
let efficiency = metrics.efficiency_ratio();
```

#### Key Metrics

- **`total_latency_ms`**: End-to-end inference time
- **`tokens_per_second`**: Throughput measurement
- **`first_token_latency_ms`**: Time to generate first token (critical for streaming)
- **`average_token_latency_ms`**: Per-token generation time
- **`cache_hit_rate`**: KV-cache efficiency (0.0 to 1.0)
- **`memory_usage_bytes`**: Current memory consumption
- **Component timing breakdown**:
  - `tokenizer_encode_time_ms`
  - `tokenizer_decode_time_ms`
  - `forward_pass_time_ms`
  - `sampling_time_ms`

#### Validation

All performance metrics include built-in validation:

```rust
let mut metrics = PerformanceMetrics::default();
metrics.tokens_per_second = -1.0; // Invalid
assert!(metrics.validate().is_err());

metrics.cache_hit_rate = Some(1.5); // Invalid (must be 0.0-1.0)
assert!(metrics.validate().is_err());
```

### PerformanceTracker

The `PerformanceTracker` manages cumulative performance statistics:

```rust
use bitnet_inference::engine::PerformanceTracker;

let mut tracker = PerformanceTracker::new();

// Record inference
tracker.record_inference(50, 1000); // 50 tokens, 1000ms

// Record cache operations
tracker.record_cache_hit();
tracker.record_cache_miss();

// Get computed metrics
let hit_rate = tracker.get_cache_hit_rate(); // Some(0.5)
let throughput = tracker.get_average_tokens_per_second(); // 50.0
```

### Enhanced InferenceEngine

The `InferenceEngine` now includes integrated performance tracking:

```rust
use bitnet_inference::engine::InferenceEngine;
use bitnet_common::Device;

let engine = InferenceEngine::new(model, tokenizer, Device::Cpu)?;

// Generate with automatic performance tracking
let result = engine.generate("Hello, world!").await?;

// Get detailed performance metrics
let metrics = engine.get_performance_metrics().await?;
println!("Throughput: {:.2} tokens/sec", metrics.tokens_per_second);
println!("Cache hit rate: {:.2}", metrics.cache_hit_rate.unwrap_or(0.0));

// Reset tracking for benchmarking
engine.reset_performance_tracking()?;
```

### Device Memory Tracking

BitNet-rs provides comprehensive device-aware memory tracking through the `DeviceAwareQuantizer`:

```rust
use bitnet_kernels::device_aware::{DeviceAwareQuantizer, DeviceStats};
use bitnet_common::Device;

// Create quantizer with memory tracking
let quantizer = DeviceAwareQuantizer::new(Device::Cuda(0))?;

// Perform operations with automatic memory tracking
let input = vec![1.0, 2.0, 3.0, 4.0];
let result = quantizer.quantize(&input, QuantizationType::I2S)?;

// Get comprehensive device statistics
if let Some(stats) = quantizer.get_stats() {
    // Memory usage information
    println!("Host Memory: {} MB used / {} MB total",
        stats.memory_used_bytes / (1024 * 1024),
        stats.memory_total_bytes / (1024 * 1024)
    );

    // Operation counts and performance
    println!("Operations: {} GPU, {} CPU, {} fallbacks",
        stats.gpu_operations, stats.cpu_operations, stats.fallback_count);

    println!("Performance: {:.2} ms total quantization, {:.2} ms total matmul",
        stats.performance.total_quantization_time_ms,
        stats.performance.total_matmul_time_ms);

    // Memory efficiency metrics
    if let Some(efficiency) = stats.memory_efficiency() {
        println!("Memory Efficiency: {:.1}%", efficiency * 100.0);
    }
}
```

#### Memory Tracking Features

- **Process-Specific Memory**: Uses `memory-stats` crate for accurate current process memory usage
- **System Memory Monitoring**: Uses `sysinfo` crate with optimized memory refresh for total system memory
- **GPU Memory Integration**: CUDA cuMemGetInfo_v2 for device memory statistics (compiled if gpu feature enabled)
- **Thread-Safe Access**: All memory statistics protected by Arc<Mutex<DeviceStatsInternal>>
- **Real-Time Updates**: Memory stats automatically updated during quantization and matrix operations
- **Memory Efficiency Calculation**: Derived metrics like usage percentages and efficiency ratios

#### Memory Tracking Dependencies

The memory tracking functionality relies on two key dependencies:

- **`memory-stats`**: Cross-platform process memory statistics
  - Provides accurate current process physical memory usage
  - Minimal overhead suitable for real-time tracking
  - Works on Windows, macOS, and Linux

- **`sysinfo`**: System information and monitoring
  - Provides total system memory information
  - Optimized memory refresh to avoid unnecessary system calls
  - Cross-platform compatibility with consistent API
  - **NEW**: Enhanced system metrics collection for CPU usage, disk usage, and network statistics
  - Real-time system monitoring for comprehensive observability

### Practical Memory Tracking Example

Here's a complete example demonstrating memory tracking in practice:

```bash
# Run the comprehensive device stats demo
cargo run --example device_stats_demo --no-default-features --features cpu

# For GPU memory tracking (requires CUDA)
cargo run --example device_stats_demo --no-default-features --features gpu
```

The example output includes:

```
=== BitNet-rs Device-Aware Quantization with Memory Tracking ===

📊 Initial statistics:
   Memory used: 45.23 MB
   Memory total: 16384.00 MB
   Memory utilization: 0.3%

🔧 Performing quantization operations...
   Operation 1: 1024 elements
   Operation 2: 2048 elements
   ...

📈 Final statistics:
   Total operations: 8
   - Quantization ops: 5
   - MatMul ops: 3
   Memory tracking:
   - Memory used: 47.85 MB (50,175,942 bytes)
   - Memory total: 16384.00 MB (17,179,869,184 bytes)
   - Memory utilization: 0.3%
```

This demonstrates how memory tracking integrates seamlessly with quantization operations, providing real-time insights into memory consumption patterns.

### System Metrics Integration

BitNet-rs now includes comprehensive system-level metrics collection through the server monitoring infrastructure. This integration provides full observability into both application-level and system-level performance.

#### Available System Metrics

The system metrics collection, powered by the `sysinfo` crate, provides:

```rust
use bitnet_server::monitoring::SystemMetrics;

// System metrics are automatically collected
let system_metrics = SystemMetrics::new();

// Metrics available through Prometheus endpoints:
// - system_cpu_usage_percent
// - system_memory_usage_percent
// - system_disk_usage_percent
// - system_network_bytes_received_total
// - system_network_bytes_sent_total
// - system_uptime_seconds
```

#### Integration with Performance Tracking

The system metrics work alongside existing performance tracking:

```rust
use bitnet_server::monitoring::MetricsCollector;
use bitnet_inference::engine::InferenceEngine;

// Create metrics collector with system monitoring
let collector = MetricsCollector::new(&monitoring_config)?;

// Perform inference operations
let engine = InferenceEngine::new(model, tokenizer, Device::Cpu)?;
let tracker = collector.track_request("request-123".to_string());

// Generate text with both application and system monitoring
let result = engine.generate("Hello world").await?;
tracker.record_tokens(result.tokens.len() as u64);

// Collect comprehensive system metrics
collector.collect_system_metrics().await?;

// System metrics are automatically exposed via Prometheus
```

#### Real-Time System Monitoring

System metrics are collected at configurable intervals and provide insights into:

- **CPU Usage**: Real-time CPU utilization percentage across all cores
- **Memory Usage**: System memory utilization (used vs total)
- **Disk Usage**: Storage utilization for inference workloads
- **Network I/O**: Bytes received/sent for distributed inference scenarios
- **Process Uptime**: Service availability and restart tracking

#### Performance Correlation Analysis

The system metrics enable correlation between application performance and system resources:

```rust
// Example: Correlate token throughput with system resources
let app_metrics = engine.get_performance_metrics().await?;
let system_stats = collector.get_system_stats().await?;

if app_metrics.tokens_per_second < 50.0 && system_stats.cpu_usage > 90.0 {
    warn!("Low throughput may be due to CPU saturation: {:.1}% CPU usage",
          system_stats.cpu_usage);
}

if system_stats.memory_usage > 95.0 {
    warn!("High memory pressure detected: {:.1}% memory usage",
          system_stats.memory_usage);
}
```

## Environment Variable Configuration

Performance behavior can be controlled via environment variables:

### Deterministic Execution

```bash
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
```

This enables reproducible results for testing and validation.

### Performance Tuning

```bash
export BITNET_BATCH_SIZE=4
export BITNET_MEMORY_LIMIT=1GB
export BITNET_NUM_THREADS=8
```

### Validation

Environment variables are validated at runtime:

```rust
let mut engine = create_engine().await;
match engine.apply_env_performance_config() {
    Ok(()) => println!("Environment configuration applied"),
    Err(e) => eprintln!("Invalid environment config: {}", e),
}
```

## Usage Examples

### Basic Performance Monitoring

```rust
use bitnet_inference::engine::InferenceEngine;
use bitnet_common::Device;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let engine = InferenceEngine::new(model, tokenizer, Device::Cpu)?;

    // Generate text with tracking
    let result = engine.generate("Explain quantum computing").await?;

    // Get performance insights
    let metrics = engine.get_performance_metrics().await?;

    println!("Generated {} tokens", metrics.tokens_generated);
    println!("Throughput: {:.2} tokens/sec", metrics.tokens_per_second);
    println!("Total latency: {}ms", metrics.total_latency_ms);

    if let Some(first_token_ms) = metrics.first_token_latency_ms {
        println!("Time to first token: {}ms", first_token_ms);
    }

    if let Some(hit_rate) = metrics.cache_hit_rate {
        println!("Cache hit rate: {:.1}%", hit_rate * 100.0);
    }

    Ok(())
}
```

### Performance Benchmarking

```rust
use bitnet_inference::config::GenerationConfig;
use std::time::Instant;

async fn benchmark_performance(engine: &InferenceEngine) -> anyhow::Result<()> {
    let prompts = vec![
        "Write a short story about AI",
        "Explain machine learning",
        "Describe quantum physics",
    ];

    // Reset tracking for clean benchmark
    engine.reset_performance_tracking()?;

    let benchmark_start = Instant::now();

    for (i, prompt) in prompts.iter().enumerate() {
        let config = GenerationConfig::default()
            .with_max_tokens(100)
            .with_temperature(0.7);

        let _result = engine.generate_with_config(prompt, &config).await?;

        let current_metrics = engine.get_performance_metrics().await?;
        println!("Prompt {}: {:.2} tokens/sec", i + 1, current_metrics.tokens_per_second);
    }

    let total_duration = benchmark_start.elapsed();
    let final_metrics = engine.get_performance_metrics().await?;

    println!("\nBenchmark Results:");
    println!("Total time: {:.2}s", total_duration.as_secs_f64());
    println!("Total tokens: {}", final_metrics.tokens_generated);
    println!("Average throughput: {:.2} tokens/sec", final_metrics.tokens_per_second);
    println!("Efficiency ratio: {:.4} tokens/ms", final_metrics.efficiency_ratio());

    Ok(())
}
```

### Deterministic Testing

```bash
#!/bin/bash
# Set up deterministic environment
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# Run tests
cargo test --no-default-features --package bitnet-inference --features integration-tests
```

```rust
#[tokio::test]
async fn test_deterministic_performance() {
    use std::env;

    // Ensure deterministic environment
    unsafe {
        env::set_var("BITNET_DETERMINISTIC", "1");
        env::set_var("BITNET_SEED", "42");
    }

    let mut engine = create_test_engine().await;
    engine.apply_env_performance_config()?;

    // Multiple runs should produce identical metrics
    let config = GenerationConfig::default().with_max_tokens(10);

    let result1 = engine.generate_with_config("test prompt", &config).await?;
    engine.reset_performance_tracking()?;

    let result2 = engine.generate_with_config("test prompt", &config).await?;

    // Results should be identical in deterministic mode
    assert_eq!(result1, result2);

    // Clean up
    unsafe {
        env::remove_var("BITNET_DETERMINISTIC");
        env::remove_var("BITNET_SEED");
    }
}
```

### Custom Performance Analysis

```rust
use bitnet_inference::engine::PerformanceMetrics;

fn analyze_performance(metrics: &PerformanceMetrics) {
    println!("Performance Analysis:");
    println!("===================");

    // Throughput analysis
    if metrics.tokens_per_second > 100.0 {
        println!("✅ Excellent throughput: {:.1} tokens/sec", metrics.tokens_per_second);
    } else if metrics.tokens_per_second > 50.0 {
        println!("⚠️  Good throughput: {:.1} tokens/sec", metrics.tokens_per_second);
    } else {
        println!("❌ Low throughput: {:.1} tokens/sec", metrics.tokens_per_second);
    }

    // Latency analysis
    if let Some(first_token_ms) = metrics.first_token_latency_ms {
        if first_token_ms < 100 {
            println!("✅ Fast first token: {}ms", first_token_ms);
        } else {
            println!("⚠️  Slow first token: {}ms", first_token_ms);
        }
    }

    // Cache efficiency
    if let Some(hit_rate) = metrics.cache_hit_rate {
        if hit_rate > 0.8 {
            println!("✅ Excellent cache efficiency: {:.1}%", hit_rate * 100.0);
        } else if hit_rate > 0.5 {
            println!("⚠️  Good cache efficiency: {:.1}%", hit_rate * 100.0);
        } else {
            println!("❌ Poor cache efficiency: {:.1}%", hit_rate * 100.0);
        }
    }

    // Memory usage
    if let Some(memory_bytes) = metrics.memory_usage_bytes {
        let memory_mb = memory_bytes as f64 / 1024.0 / 1024.0;
        println!("Memory usage: {:.1} MB", memory_mb);
    }

    // Component breakdown
    println!("\nTiming Breakdown:");
    if let Some(encode_ms) = metrics.tokenizer_encode_time_ms {
        println!("  Tokenizer encode: {}ms", encode_ms);
    }
    if let Some(forward_ms) = metrics.forward_pass_time_ms {
        println!("  Forward pass: {}ms", forward_ms);
    }
    if let Some(decode_ms) = metrics.tokenizer_decode_time_ms {
        println!("  Tokenizer decode: {}ms", decode_ms);
    }
}
```

## Error Handling

The performance tracking infrastructure includes comprehensive error handling:

```rust
// Validation errors
let mut metrics = PerformanceMetrics::default();
metrics.tokens_per_second = -1.0;

match metrics.validate() {
    Ok(()) => println!("Metrics valid"),
    Err(e) => eprintln!("Validation error: {}", e),
}

// Environment configuration errors
match engine.apply_env_performance_config() {
    Ok(()) => println!("Environment applied"),
    Err(e) => eprintln!("Environment error: {}", e),
}

// Tracker access errors (thread safety)
match engine.get_performance_metrics().await {
    Ok(metrics) => println!("Current metrics: {:?}", metrics),
    Err(e) => eprintln!("Failed to get metrics: {}", e),
}
```

## Testing

The performance tracking infrastructure includes comprehensive test coverage:

```bash
# Run performance tracking tests
cargo test --no-default-features --package bitnet-inference \
    --features integration-tests \
    --test performance_tracking_tests

# Run specific test categories
cargo test --no-default-features --features cpu --test performance_tracking_tests performance_metrics_tests
cargo test --no-default-features --features cpu --test performance_tracking_tests performance_tracker_tests
cargo test --no-default-features --features cpu --test performance_tracking_tests environment_variable_tests

# Test InferenceEngine performance integration
cargo test --no-default-features -p bitnet-inference --features integration-tests test_engine_performance_tracking_integration

# Test platform-specific performance tracking
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_memory_tracking
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_performance_tracking

# Test comprehensive memory tracking with device awareness
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_memory_tracking_comprehensive
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_device_memory_tracking

# Test memory efficiency metrics
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_memory_efficiency_tracking

# GPU performance validation with comprehensive metrics
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_cuda_validation_comprehensive
```

## Best Practices

### 1. Regular Monitoring

```rust
// Set up periodic performance monitoring
let mut interval = tokio::time::interval(Duration::from_secs(30));

loop {
    interval.tick().await;

    let metrics = engine.get_performance_metrics().await?;
    if metrics.tokens_per_second < 50.0 {
        warn!("Low throughput detected: {:.1} tokens/sec", metrics.tokens_per_second);
    }

    if let Some(hit_rate) = metrics.cache_hit_rate {
        if hit_rate < 0.7 {
            warn!("Low cache hit rate: {:.1}%", hit_rate * 100.0);
        }
    }
}
```

### 2. Performance Regression Testing

```rust
#[tokio::test]
async fn test_performance_regression() {
    let engine = create_test_engine().await;

    // Expected baseline performance
    const MIN_TOKENS_PER_SECOND: f64 = 50.0;
    const MIN_CACHE_HIT_RATE: f64 = 0.7;

    let config = GenerationConfig::default().with_max_tokens(100);
    let _result = engine.generate_with_config("test prompt", &config).await?;

    let metrics = engine.get_performance_metrics().await?;

    assert!(metrics.tokens_per_second >= MIN_TOKENS_PER_SECOND,
        "Performance regression: {:.1} < {:.1} tokens/sec",
        metrics.tokens_per_second, MIN_TOKENS_PER_SECOND);

    if let Some(hit_rate) = metrics.cache_hit_rate {
        assert!(hit_rate >= MIN_CACHE_HIT_RATE,
            "Cache efficiency regression: {:.1}% < {:.1}%",
            hit_rate * 100.0, MIN_CACHE_HIT_RATE * 100.0);
    }
}
```

### 3. Resource Monitoring

```rust
async fn monitor_resources(engine: &InferenceEngine) -> anyhow::Result<()> {
    let metrics = engine.get_performance_metrics().await?;

    // Memory usage alerts
    if let Some(memory_bytes) = metrics.memory_usage_bytes {
        const MAX_MEMORY_MB: f64 = 1024.0; // 1GB
        let memory_mb = memory_bytes as f64 / 1024.0 / 1024.0;

        if memory_mb > MAX_MEMORY_MB {
            warn!("High memory usage: {:.1} MB > {:.1} MB", memory_mb, MAX_MEMORY_MB);
        }
    }

    // Efficiency monitoring
    let efficiency = metrics.efficiency_ratio();
    if efficiency < 0.05 { // Less than 0.05 tokens/ms
        warn!("Low efficiency: {:.4} tokens/ms", efficiency);
    }

    Ok(())
}
```

## Integration with Existing Systems

The performance tracking infrastructure integrates seamlessly with existing BitNet-rs components:

- **Inference Engine**: Automatic tracking during generation
- **Streaming API**: Performance metrics for streaming operations
- **CLI Tools**: Performance reporting in command-line interfaces
- **Server**: HTTP API endpoints for performance monitoring
- **Testing**: Deterministic performance validation

This provides comprehensive visibility into system performance across all usage patterns and deployment scenarios.
