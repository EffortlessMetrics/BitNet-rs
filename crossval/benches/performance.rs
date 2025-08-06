//! Performance benchmarks comparing Rust and C++ implementations

#![cfg(feature = "crossval")]

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use bitnet_crossval::{
    cpp_bindings::CppModel,
    fixtures::{TestFixture, STANDARD_PROMPTS},
    CrossvalConfig,
};
use std::time::Duration;

fn benchmark_rust_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_inference");
    group.measurement_time(Duration::from_secs(10));
    
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
        group.bench_with_input(
            BenchmarkId::new("generate", prompt.len()),
            prompt,
            |b, prompt| {
                b.iter(|| {
                    // Placeholder for Rust implementation
                    // In real code, this would call bitnet-inference
                    let _tokens = generate_rust_tokens(black_box(prompt));
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_cpp_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpp_inference");
    group.measurement_time(Duration::from_secs(10));
    
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
        group.bench_with_input(
            BenchmarkId::new("generate", prompt.len()),
            prompt,
            |b, prompt| {
                b.iter(|| {
                    let _tokens = cpp_model
                        .generate(black_box(prompt), 100)
                        .expect("C++ generation should succeed");
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison");
    group.measurement_time(Duration::from_secs(15));
    
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
    
    group.bench_function("rust_vs_cpp", |b| {
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
}

fn benchmark_model_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_loading");
    
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
    benchmark_model_loading
);

criterion_main!(benches);