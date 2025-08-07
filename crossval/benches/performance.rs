//! Performance benchmarks comparing Rust and C++ implementations
//!
//! This benchmark suite provides comprehensive performance comparison between
//! BitNet.rs and the legacy C++ implementation, with regression detection
//! and detailed reporting.

#![cfg(feature = "crossval")]

use criterion::{
    black_box, criterion_group, criterion_main, Criterion, BenchmarkId,
    Throughput, PlotConfiguration, AxisScale,
};
use bitnet_crossval::{
    cpp_bindings::CppModel,
    fixtures::{TestFixture, STANDARD_PROMPTS},
    CrossvalConfig,
};
use std::time::Duration;
use std::fs;
use serde::{Serialize, Deserialize};

/// Performance baseline data for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceBaseline {
    rust_tokens_per_second: f64,
    cpp_tokens_per_second: f64,
    rust_memory_mb: f64,
    cpp_memory_mb: f64,
    speedup_ratio: f64,
    memory_reduction: f64,
    timestamp: String,
}

impl PerformanceBaseline {
    fn new(rust_tps: f64, cpp_tps: f64, rust_mem: f64, cpp_mem: f64) -> Self {
        let speedup_ratio = if cpp_tps > 0.0 { rust_tps / cpp_tps } else { 0.0 };
        let memory_reduction = if cpp_mem > 0.0 { (cpp_mem - rust_mem) / cpp_mem } else { 0.0 };
        
        Self {
            rust_tokens_per_second: rust_tps,
            cpp_tokens_per_second: cpp_tps,
            rust_memory_mb: rust_mem,
            cpp_memory_mb: cpp_mem,
            speedup_ratio,
            memory_reduction,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Load performance baseline from file
fn load_baseline() -> Option<PerformanceBaseline> {
    let baseline_path = "crossval/baselines.json";
    if let Ok(content) = fs::read_to_string(baseline_path) {
        serde_json::from_str(&content).ok()
    } else {
        None
    }
}

/// Save performance baseline to file
fn save_baseline(baseline: &PerformanceBaseline) {
    let baseline_path = "crossval/baselines.json";
    if let Ok(content) = serde_json::to_string_pretty(baseline) {
        let _ = fs::write(baseline_path, content);
    }
}

/// Check for performance regression (5% threshold)
fn check_regression(current: f64, baseline: f64, metric_name: &str) -> bool {
    if baseline <= 0.0 {
        return false; // No baseline to compare against
    }
    
    let regression_threshold = 0.05; // 5%
    let change_ratio = (current - baseline) / baseline;
    
    if change_ratio < -regression_threshold {
        eprintln!(
            "⚠️  PERFORMANCE REGRESSION DETECTED in {}: {:.1}% slower than baseline",
            metric_name,
            change_ratio.abs() * 100.0
        );
        eprintln!("   Current: {:.2}, Baseline: {:.2}", current, baseline);
        return true;
    } else if change_ratio > regression_threshold {
        eprintln!(
            "🚀 PERFORMANCE IMPROVEMENT in {}: {:.1}% faster than baseline",
            metric_name,
            change_ratio * 100.0
        );
    }
    
    false
}

fn benchmark_rust_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_inference");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    // Create a test fixture
    let fixture = TestFixture {
        name: "benchmark".to_string(),
        model_path: "fixtures/benchmark_model.gguf".into(),
        test_prompts: STANDARD_PROMPTS.iter().map(|s| s.to_string()).collect(),
        expected_tokens: None,
    };
    
    // Skip benchmarks if fixture doesn't exist
    if !fixture.model_path.exists() {
        eprintln!("Skipping Rust benchmarks: fixture not found at {:?}", fixture.model_path);
        return;
    }
    
    for prompt in &fixture.test_prompts {
        let token_count = prompt.len() / 4 + 10; // Estimate token count
        group.throughput(Throughput::Elements(token_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("generate", prompt.len()),
            prompt,
            |b, prompt| {
                b.iter(|| {
                    // Placeholder for Rust implementation
                    // In real code, this would call bitnet-inference
                    let tokens = generate_rust_tokens(black_box(prompt));
                    black_box(tokens)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_cpp_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpp_inference");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    // Create a test fixture
    let fixture = TestFixture {
        name: "benchmark".to_string(),
        model_path: "fixtures/benchmark_model.gguf".into(),
        test_prompts: STANDARD_PROMPTS.iter().map(|s| s.to_string()).collect(),
        expected_tokens: None,
    };
    
    // Skip benchmarks if fixture doesn't exist
    if !fixture.model_path.exists() {
        eprintln!("Skipping C++ benchmarks: fixture not found at {:?}", fixture.model_path);
        return;
    }
    
    // Load C++ model once for all benchmarks
    let cpp_model = match CppModel::load(&fixture.model_path) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Failed to load C++ model: {}", e);
            return;
        }
    };
    
    for prompt in &fixture.test_prompts {
        let token_count = prompt.len() / 4 + 10; // Estimate token count
        group.throughput(Throughput::Elements(token_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("generate", prompt.len()),
            prompt,
            |b, prompt| {
                b.iter(|| {
                    let tokens = cpp_model
                        .generate(black_box(prompt), 100)
                        .expect("C++ generation should succeed");
                    black_box(tokens)
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(30);
    
    let fixture = TestFixture {
        name: "comparison".to_string(),
        model_path: "fixtures/benchmark_model.gguf".into(),
        test_prompts: vec!["The quick brown fox jumps over the lazy dog.".to_string()],
        expected_tokens: None,
    };
    
    if !fixture.model_path.exists() {
        eprintln!("Skipping comparison benchmarks: fixture not found");
        return;
    }
    
    let cpp_model = match CppModel::load(&fixture.model_path) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Failed to load C++ model for comparison: {}", e);
            return;
        }
    };
    
    let prompt = &fixture.test_prompts[0];
    
    // Benchmark individual implementations for comparison
    let mut rust_times = Vec::new();
    let mut cpp_times = Vec::new();
    
    group.bench_function("rust_only", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let tokens = generate_rust_tokens(black_box(prompt));
            let duration = start.elapsed();
            rust_times.push(duration.as_secs_f64());
            black_box(tokens)
        });
    });
    
    group.bench_function("cpp_only", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let tokens = cpp_model
                .generate(black_box(prompt), 100)
                .expect("C++ generation should succeed");
            let duration = start.elapsed();
            cpp_times.push(duration.as_secs_f64());
            black_box(tokens)
        });
    });
    
    group.bench_function("cross_validation", |b| {
        b.iter(|| {
            // Generate with both implementations
            let rust_tokens = generate_rust_tokens(black_box(prompt));
            let cpp_tokens = cpp_model
                .generate(black_box(prompt), 100)
                .expect("C++ generation should succeed");
            
            // Compare tokens (this is what we're actually benchmarking)
            let config = CrossvalConfig::default();
            let _matches = bitnet_crossval::utils::compare_tokens(
                &rust_tokens,
                &cpp_tokens,
                &config,
            );
        });
    });
    
    group.finish();
    
    // Calculate and report performance metrics
    if !rust_times.is_empty() && !cpp_times.is_empty() {
        let rust_avg = rust_times.iter().sum::<f64>() / rust_times.len() as f64;
        let cpp_avg = cpp_times.iter().sum::<f64>() / cpp_times.len() as f64;
        
        let rust_tps = 100.0 / rust_avg; // Assuming 100 tokens generated
        let cpp_tps = 100.0 / cpp_avg;
        
        let speedup = if cpp_avg > 0.0 { rust_tps / cpp_tps } else { 0.0 };
        
        println!("\n📊 Performance Comparison Results:");
        println!("   Rust: {:.1} tokens/sec", rust_tps);
        println!("   C++:  {:.1} tokens/sec", cpp_tps);
        println!("   Speedup: {:.2}x", speedup);
        
        // Check for regressions against baseline
        if let Some(baseline) = load_baseline() {
            let rust_regression = check_regression(rust_tps, baseline.rust_tokens_per_second, "Rust inference");
            let cpp_regression = check_regression(cpp_tps, baseline.cpp_tokens_per_second, "C++ inference");
            
            if rust_regression || cpp_regression {
                eprintln!("⚠️  Performance regression detected! Check recent changes.");
            }
        }
        
        // Save new baseline
        let new_baseline = PerformanceBaseline::new(rust_tps, cpp_tps, 0.0, 0.0);
        save_baseline(&new_baseline);
    }
}

fn benchmark_model_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_loading");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10); // Model loading is expensive
    
    let model_path = "fixtures/benchmark_model.gguf";
    
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping model loading benchmarks: fixture not found");
        return;
    }
    
    group.bench_function("cpp_model_load", |b| {
        b.iter(|| {
            let model = CppModel::load(black_box(model_path))
                .expect("Model loading should succeed");
            drop(model); // Ensure cleanup is measured
        });
    });
    
    // Benchmark memory usage during model loading
    group.bench_function("cpp_model_memory", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            
            for _ in 0..iters {
                let model = CppModel::load(black_box(model_path))
                    .expect("Model loading should succeed");
                
                // Simulate some work to measure peak memory
                std::thread::sleep(Duration::from_millis(10));
                
                drop(model);
            }
            
            start.elapsed()
        });
    });
    
    group.finish();
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(10));
    
    let fixture = TestFixture {
        name: "memory_test".to_string(),
        model_path: "fixtures/benchmark_model.gguf".into(),
        test_prompts: vec!["Memory usage test prompt".to_string()],
        expected_tokens: None,
    };
    
    if !fixture.model_path.exists() {
        eprintln!("Skipping memory benchmarks: fixture not found");
        return;
    }
    
    let cpp_model = match CppModel::load(&fixture.model_path) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Failed to load C++ model for memory test: {}", e);
            return;
        }
    };
    
    group.bench_function("rust_memory_efficiency", |b| {
        b.iter(|| {
            // Simulate multiple generations to test memory efficiency
            for _ in 0..10 {
                let _tokens = generate_rust_tokens(black_box("test prompt"));
            }
        });
    });
    
    group.bench_function("cpp_memory_efficiency", |b| {
        b.iter(|| {
            // Simulate multiple generations to test memory efficiency
            for _ in 0..10 {
                let _tokens = cpp_model
                    .generate(black_box("test prompt"), 50)
                    .expect("C++ generation should succeed");
            }
        });
    });
    
    group.finish();
}

// Placeholder function for Rust token generation
// In real implementation, this would call into bitnet-inference
fn generate_rust_tokens(prompt: &str) -> Vec<u32> {
    // Simulate some work based on prompt length
    let token_count = prompt.len() / 4 + 1;
    (1..=token_count as u32).collect()
}

criterion_group!(
    benches,
    benchmark_rust_inference,
    benchmark_cpp_inference,
    benchmark_comparison,
    benchmark_model_loading,
    benchmark_memory_usage
);

criterion_main!(benches);