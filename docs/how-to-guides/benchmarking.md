# How to Benchmark Performance

This guide explains how to benchmark the performance of BitNet.rs.

## Comprehensive Benchmarks

### 1. Latency Benchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_inference_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let engine = rt.block_on(create_test_engine()).unwrap();

    c.bench_function("inference_latency", |b| {
        b.to_async(&rt).iter(|| async {
            let result = engine.generate(black_box("Hello, world!")).await;
            black_box(result)
        })
    });
}

criterion_group!(benches, bench_inference_latency);
criterion_main!(benches);
```

### 2. Throughput Benchmarks

```rust
fn bench_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let engine = rt.block_on(create_test_engine()).unwrap();

    let prompts: Vec<String> = (0..100)
        .map(|i| format!("Test prompt {}", i))
        .collect();

    c.bench_function("throughput_100_requests", |b| {
        b.to_async(&rt).iter(|| async {
            let tasks = prompts.iter().map(|prompt| {
                engine.generate(black_box(prompt))
            });

            let results = futures_util::future::join_all(tasks).await;
            black_box(results)
        })
    });
}
```

### 3. Memory Benchmarks

```rust
fn bench_memory_usage(c: &mut Criterion) {
    c.bench_function("memory_usage", |b| {
        b.iter_custom(|iters| {
            let start_memory = get_memory_usage();
            let start_time = Instant::now();

            for _ in 0..iters {
                let engine = create_test_engine_sync();
                black_box(engine);
            }

            let end_memory = get_memory_usage();
            let duration = start_time.elapsed();

            println!("Memory delta: {} MB", (end_memory - start_memory) / 1024 / 1024);
            duration
        })
    });
}
```

## Performance Regression Testing

```rust
// Automated performance regression detection
#[tokio::test]
async fn test_performance_regression() {
    let engine = create_test_engine().await.unwrap();

    // Baseline performance (update these values after verified improvements)
    const EXPECTED_LATENCY_MS: u64 = 100;
    const EXPECTED_THROUGHPUT_TPS: f64 = 10.0;

    // Measure current performance
    let start = Instant::now();
    let response = engine.generate("Performance test prompt").await.unwrap();
    let latency = start.elapsed();

    let tokens = response.split_whitespace().count();
    let throughput = tokens as f64 / latency.as_secs_f64();

    // Assert performance hasn't regressed
    assert!(
        latency.as_millis() <= EXPECTED_LATENCY_MS as u128,
        "Latency regression: {}ms > {}ms",
        latency.as_millis(),
        EXPECTED_LATENCY_MS
    );

    assert!(
        throughput >= EXPECTED_THROUGHPUT_TPS,
        "Throughput regression: {:.2} < {:.2} tokens/sec",
        throughput,
        EXPECTED_THROUGHPUT_TPS
    );
}
```
