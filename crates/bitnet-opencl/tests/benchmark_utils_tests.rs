use bitnet_opencl::benchmark_utils::{
    BenchmarkComparison, BenchmarkResult, BenchmarkRunner, format_benchmark_report, format_duration,
};
use std::time::Duration;

// ── helpers ─────────────────────────────────────────────────────────────────

fn sample_result(
    name: &str,
    backend: &str,
    durations_us: &[u64],
    elements: u64,
) -> BenchmarkResult {
    let samples: Vec<Duration> = durations_us.iter().map(|&us| Duration::from_micros(us)).collect();
    BenchmarkResult::new(name, backend, samples, elements)
}

// ── BenchmarkResult statistics ──────────────────────────────────────────────

#[test]
fn test_mean_single_sample() {
    let r = sample_result("op", "cpu", &[100], 1);
    assert_eq!(r.mean(), Duration::from_micros(100));
}

#[test]
fn test_mean_multiple_samples() {
    let r = sample_result("op", "cpu", &[100, 200, 300], 1);
    assert_eq!(r.mean(), Duration::from_micros(200));
}

#[test]
fn test_mean_empty() {
    let r = BenchmarkResult::new("op", "cpu", vec![], 1);
    assert_eq!(r.mean(), Duration::ZERO);
}

#[test]
fn test_std_dev_identical_samples() {
    let r = sample_result("op", "cpu", &[100, 100, 100], 1);
    assert_eq!(r.std_dev(), Duration::ZERO);
}

#[test]
fn test_std_dev_varied_samples() {
    // Samples: 100, 200, 300 µs → mean 200, variance = 6666.67 µs²
    // std_dev ≈ 81.65 µs
    let r = sample_result("op", "cpu", &[100, 200, 300], 1);
    let sd_us = r.std_dev().as_micros();
    assert!(sd_us >= 80 && sd_us <= 83, "std_dev was {sd_us} µs");
}

#[test]
fn test_std_dev_single_sample() {
    let r = sample_result("op", "cpu", &[42], 1);
    assert_eq!(r.std_dev(), Duration::ZERO);
}

#[test]
fn test_min_max() {
    let r = sample_result("op", "cpu", &[50, 100, 200, 300, 500], 1);
    assert_eq!(r.min(), Duration::from_micros(50));
    assert_eq!(r.max(), Duration::from_micros(500));
}

#[test]
fn test_min_max_empty() {
    let r = BenchmarkResult::new("op", "cpu", vec![], 1);
    assert_eq!(r.min(), Duration::ZERO);
    assert_eq!(r.max(), Duration::ZERO);
}

#[test]
fn test_percentile_p50() {
    // Sorted: 10, 20, 30, 40, 50 → P50 = index 2 = 30
    let r = sample_result("op", "cpu", &[30, 10, 50, 20, 40], 1);
    assert_eq!(r.percentile(50.0), Duration::from_micros(30));
}

#[test]
fn test_percentile_p0_and_p100() {
    let r = sample_result("op", "cpu", &[10, 20, 30, 40, 50], 1);
    assert_eq!(r.percentile(0.0), Duration::from_micros(10));
    assert_eq!(r.percentile(100.0), Duration::from_micros(50));
}

#[test]
fn test_percentile_p95() {
    let samples: Vec<u64> = (1..=100).collect();
    let r = sample_result("op", "cpu", &samples, 1);
    let p95 = r.percentile(95.0);
    // Index = round(0.95 * 99) = round(94.05) = 94 → value 95 µs
    assert_eq!(p95, Duration::from_micros(95));
}

#[test]
fn test_percentile_empty() {
    let r = BenchmarkResult::new("op", "cpu", vec![], 1);
    assert_eq!(r.percentile(50.0), Duration::ZERO);
}

#[test]
fn test_throughput_eps() {
    // 1000 elements, mean = 1ms → 1_000_000 elements/s
    let r = sample_result("op", "cpu", &[1000, 1000, 1000], 1000);
    let eps = r.throughput_eps();
    assert!((eps - 1_000_000.0).abs() < 1.0, "throughput was {eps}");
}

#[test]
fn test_throughput_zero_mean() {
    let r = BenchmarkResult::new("op", "cpu", vec![], 1000);
    assert_eq!(r.throughput_eps(), 0.0);
}

#[test]
fn test_count() {
    let r = sample_result("op", "cpu", &[1, 2, 3, 4, 5], 1);
    assert_eq!(r.count(), 5);
}

#[test]
fn test_with_memory() {
    let r = sample_result("op", "cpu", &[100], 1).with_memory(1024);
    assert_eq!(r.peak_memory_bytes, Some(1024));
}

// ── BenchmarkRunner ─────────────────────────────────────────────────────────

#[test]
fn test_runner_collects_correct_count() {
    let runner = BenchmarkRunner::new(50, 5);
    let mut counter = 0u64;
    let result = runner.run("count_test", "cpu", 1, || {
        counter += 1;
    });
    // warmup(5) + iterations(50) = 55 total calls
    assert_eq!(counter, 55);
    assert_eq!(result.count(), 50);
}

#[test]
fn test_runner_default() {
    let runner = BenchmarkRunner::default();
    assert_eq!(runner.iterations, 100);
    assert_eq!(runner.warmup, 10);
}

#[test]
fn test_runner_result_name_and_backend() {
    let runner = BenchmarkRunner::new(10, 0);
    let result = runner.run("my_bench", "opencl", 42, || {});
    assert_eq!(result.name, "my_bench");
    assert_eq!(result.backend, "opencl");
    assert_eq!(result.elements_per_iter, 42);
}

// ── format_benchmark_report ─────────────────────────────────────────────────

#[test]
fn test_report_contains_key_fields() {
    let r = sample_result("matmul", "cpu", &[1000, 2000, 3000], 256);
    let report = format_benchmark_report(&r);
    assert!(report.contains("matmul"), "missing name");
    assert!(report.contains("cpu"), "missing backend");
    assert!(report.contains("Samples:"), "missing Samples");
    assert!(report.contains("Mean:"), "missing Mean");
    assert!(report.contains("Std Dev:"), "missing Std Dev");
    assert!(report.contains("P50:"), "missing P50");
    assert!(report.contains("Throughput:"), "missing Throughput");
}

#[test]
fn test_report_with_memory() {
    let r = sample_result("matmul", "cpu", &[1000], 1).with_memory(65536);
    let report = format_benchmark_report(&r);
    assert!(report.contains("Peak Mem:"), "missing Peak Mem");
    assert!(report.contains("65536"), "missing memory value");
}

#[test]
fn test_report_without_memory() {
    let r = sample_result("matmul", "cpu", &[1000], 1);
    let report = format_benchmark_report(&r);
    assert!(!report.contains("Peak Mem:"), "should not have Peak Mem");
}

// ── format_duration ─────────────────────────────────────────────────────────

#[test]
fn test_format_duration_nanos() {
    assert_eq!(format_duration(Duration::from_nanos(42)), "42 ns");
}

#[test]
fn test_format_duration_micros() {
    let s = format_duration(Duration::from_micros(500));
    assert!(s.contains("µs"), "expected µs, got {s}");
}

#[test]
fn test_format_duration_millis() {
    let s = format_duration(Duration::from_millis(42));
    assert!(s.contains("ms"), "expected ms, got {s}");
}

#[test]
fn test_format_duration_seconds() {
    let s = format_duration(Duration::from_secs(2));
    assert!(s.contains("s"), "expected s, got {s}");
}

// ── BenchmarkComparison ─────────────────────────────────────────────────────

#[test]
fn test_comparison_fastest_backend() {
    let mut cmp = BenchmarkComparison::new("matmul 256");
    cmp.add_result(sample_result("matmul", "cpu", &[200, 200], 256));
    cmp.add_result(sample_result("matmul", "opencl", &[100, 100], 256));
    assert_eq!(cmp.fastest_backend(), Some("opencl"));
}

#[test]
fn test_comparison_fastest_empty() {
    let cmp = BenchmarkComparison::new("empty");
    assert_eq!(cmp.fastest_backend(), None);
}

#[test]
fn test_comparison_speedups() {
    let mut cmp = BenchmarkComparison::new("matmul 256");
    cmp.add_result(sample_result("matmul", "cpu", &[200, 200], 256));
    cmp.add_result(sample_result("matmul", "opencl", &[100, 100], 256));
    let speedups = cmp.speedups();
    assert_eq!(speedups.len(), 2);
    // opencl is 1.0× (fastest), cpu is 2.0×
    for (backend, factor) in &speedups {
        match *backend {
            "opencl" => assert!((*factor - 1.0).abs() < 0.01, "opencl factor: {factor}"),
            "cpu" => assert!((*factor - 2.0).abs() < 0.01, "cpu factor: {factor}"),
            other => panic!("unexpected backend: {other}"),
        }
    }
}

#[test]
fn test_comparison_display() {
    let mut cmp = BenchmarkComparison::new("softmax 32k");
    cmp.add_result(sample_result("softmax", "cpu", &[500, 600], 32000));
    cmp.add_result(sample_result("softmax", "opencl", &[200, 250], 32000));
    let table = cmp.to_string();
    assert!(table.contains("softmax 32k"), "missing label");
    assert!(table.contains("cpu"), "missing cpu");
    assert!(table.contains("opencl"), "missing opencl");
    assert!(table.contains("Fastest:"), "missing Fastest line");
}

#[test]
fn test_comparison_speedups_empty() {
    let cmp = BenchmarkComparison::new("empty");
    assert!(cmp.speedups().is_empty());
}

// ── Serialization round-trip ────────────────────────────────────────────────

#[test]
fn test_result_serde_roundtrip() {
    let r = sample_result("matmul", "cpu", &[100, 200, 300], 1024).with_memory(4096);
    let json = serde_json::to_string(&r).unwrap();
    let restored: BenchmarkResult = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.name, "matmul");
    assert_eq!(restored.backend, "cpu");
    assert_eq!(restored.count(), 3);
    assert_eq!(restored.peak_memory_bytes, Some(4096));
}

#[test]
fn test_comparison_serde_roundtrip() {
    let mut cmp = BenchmarkComparison::new("test");
    cmp.add_result(sample_result("op", "cpu", &[100], 1));
    let json = serde_json::to_string(&cmp).unwrap();
    let restored: BenchmarkComparison = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.label, "test");
    assert_eq!(restored.results.len(), 1);
}
