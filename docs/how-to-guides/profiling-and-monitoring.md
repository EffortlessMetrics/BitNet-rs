# How to Profile and Monitor Performance

This guide explains how to profile and monitor the performance of BitNet.rs.

## Performance Profiling

### 1. CPU Profiling

```bash
# Install profiling tools
cargo install flamegraph

# Generate CPU flamegraph
cargo flamegraph --bin bitnet-server

# Use perf (Linux)
perf record --call-graph=dwarf ./target/release/bitnet-server
perf report
```

### 2. Memory Profiling

```bash
# Use heaptrack (Linux)
heaptrack ./target/release/bitnet-server
heaptrack_gui heaptrack.bitnet-server.*.gz

# Use Instruments (macOS)
instruments -t "Allocations" ./target/release/bitnet-server
```

### 3. GPU Profiling

```bash
# NVIDIA Nsight Systems
nsys profile ./target/release/bitnet-server

# NVIDIA Nsight Compute
ncu --set full ./target/release/bitnet-server
```

## Runtime Monitoring

### 1. Metrics Collection

```rust
use prometheus::{Counter, Histogram, Gauge, register_counter, register_histogram, register_gauge};

struct PerformanceMetrics {
    requests_total: Counter,
    request_duration: Histogram,
    active_requests: Gauge,
    tokens_per_second: Gauge,
    memory_usage: Gauge,
}

impl PerformanceMetrics {
    fn new() -> Result<Self> {
        Ok(Self {
            requests_total: register_counter!("requests_total", "Total requests")?,
            request_duration: register_histogram!("request_duration_seconds", "Request duration")?,
            active_requests: register_gauge!("active_requests", "Active requests")?,
            tokens_per_second: register_gauge!("tokens_per_second", "Tokens per second")?,
            memory_usage: register_gauge!("memory_usage_bytes", "Memory usage")?,
        })
    }

    fn record_request(&self, duration: Duration, tokens: usize) {
        self.requests_total.inc();
        self.request_duration.observe(duration.as_secs_f64());

        let tokens_per_sec = tokens as f64 / duration.as_secs_f64();
        self.tokens_per_second.set(tokens_per_sec);
    }
}
```

### 2. Health Checks

```rust
#[derive(Debug, Serialize)]
struct HealthStatus {
    status: String,
    latency_p50: f64,
    latency_p95: f64,
    memory_usage_mb: f64,
    gpu_utilization: f64,
    error_rate: f64,
}

async fn health_check(engine: &InferenceEngine) -> HealthStatus {
    let start = Instant::now();
    let test_result = engine.generate("test").await;
    let latency = start.elapsed();

    HealthStatus {
        status: if test_result.is_ok() { "healthy" } else { "unhealthy" }.to_string(),
        latency_p50: get_latency_percentile(0.5),
        latency_p95: get_latency_percentile(0.95),
        memory_usage_mb: get_memory_usage_mb(),
        gpu_utilization: get_gpu_utilization(),
        error_rate: get_error_rate(),
    }
}
```
