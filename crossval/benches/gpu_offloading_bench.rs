// GPU Offloading Performance Benchmark
//
// Benchmark scaffolding for GPU layer offloading performance validation (v0.2.0)
//
// **Specification**: docs/explanation/cpp-wrapper-gpu-layer-config.md
//
// **Acceptance Criteria**: AC10 - Verify GPU inference is ≥5× faster than CPU
//
// **Performance Targets**:
// - GPU 24 layers: ≥5× speedup over CPU baseline (2B models)
// - GPU all layers: ≥10× speedup over CPU baseline
// - Latency reduction: P50 < 20s for 10-token sequence (vs ~100s CPU)
//
// **Benchmark Methodology**:
// 1. CPU baseline: n_gpu_layers=0 (pure CPU inference)
// 2. GPU incremental: 8, 16, 24 layers (measure scaling)
// 3. GPU full: n_gpu_layers=-1 (auto-detect all layers)
// 4. Metrics: Tokens/sec, latency (P50/P95/P99), VRAM usage
//
// **TDD Status**: Benchmark compiles but fails due to missing GPU layer configuration

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::path::Path;

// NOTE: Benchmark requires FFI feature for BitnetSession
#[cfg(feature = "ffi")]
use crossval::cpp_bindings::BitnetSession;

// Test configuration constants
const TEST_MODEL_PATH: &str = "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";
const TEST_CONTEXT_SIZE: i32 = 512;
const TEST_SEQUENCE_LENGTH: usize = 10; // 10-token sequence for benchmarking

// GPU layer configurations to benchmark
const GPU_LAYER_CONFIGS: &[i32] = &[
    0,  // CPU baseline
    8,  // GPU 8 layers
    16, // GPU 16 layers
    24, // GPU 24 layers (AC10 target)
    -1, // GPU all layers (auto-detect)
];

/// AC:AC10 - Benchmark GPU speedup over CPU baseline
///
/// **Test Objective**: Validate GPU inference is ≥5× faster than CPU for 2B models
///
/// **Specification Reference**: docs/explanation/cpp-wrapper-gpu-layer-config.md#ac10
///
/// **Benchmark Strategy**:
/// 1. Run CPU baseline (n_gpu_layers=0)
/// 2. Run GPU configurations (8, 16, 24, -1 layers)
/// 3. Measure tokens/sec and latency
/// 4. Verify GPU 24 layers ≥5× speedup
///
/// **Expected Outcome**: GPU 24 layers achieves ≥5× speedup over CPU baseline
#[cfg(feature = "ffi")]
fn bench_gpu_speedup(c: &mut Criterion) {
    let model_path = Path::new(TEST_MODEL_PATH);

    // Verify test model exists before benchmarking
    if !model_path.exists() {
        eprintln!("WARNING: Test model not found at {}", TEST_MODEL_PATH);
        eprintln!("Run: cargo run -p xtask -- download-model");
        return;
    }

    let mut group = c.benchmark_group("gpu_offloading");

    // Set throughput for tokens/sec calculation
    group.throughput(Throughput::Elements(TEST_SEQUENCE_LENGTH as u64));

    // Generate test token sequence (simple sequence for reproducibility)
    let test_tokens: Vec<i32> = (1..=TEST_SEQUENCE_LENGTH as i32).collect();

    // Benchmark each GPU layer configuration
    for &n_gpu_layers in GPU_LAYER_CONFIGS {
        let config_name = match n_gpu_layers {
            0 => "cpu_only",
            -1 => "gpu_all_layers",
            n => {
                let mut buf = String::new();
                buf.push_str("gpu_");
                buf.push_str(&n.to_string());
                buf.push_str("_layers");
                buf
            }
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(config_name),
            &n_gpu_layers,
            |b, &layers| {
                // Create session once per benchmark iteration
                let session = BitnetSession::create(model_path, TEST_CONTEXT_SIZE, layers)
                    .expect("Failed to create BitnetSession for benchmark");

                b.iter(|| {
                    // NOTE: This requires implementing eval_and_get_logits() method
                    // For now, this is a placeholder showing the expected API
                    unimplemented!(
                        "eval_and_get_logits() not yet implemented - blocked by Socket 1 inference API"
                    );

                    // Expected implementation:
                    // let logits = session.eval_and_get_logits(&test_tokens, 0)
                    //     .expect("Failed to get logits during benchmark");
                    //
                    // black_box(logits); // Prevent compiler optimization
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: GPU memory efficiency (VRAM usage per layer)
///
/// **Test Objective**: Measure VRAM consumption scaling with layer count
///
/// **Specification Reference**: docs/explanation/cpp-wrapper-gpu-layer-config.md#62
///
/// **Expected Outcome**: ~100-500MB per billion parameters offloaded
#[cfg(feature = "ffi")]
fn bench_gpu_memory_scaling(c: &mut Criterion) {
    let model_path = Path::new(TEST_MODEL_PATH);

    if !model_path.exists() {
        eprintln!("WARNING: Test model not found for memory benchmark");
        return;
    }

    let mut group = c.benchmark_group("gpu_memory_scaling");

    // Benchmark memory allocation overhead for different layer counts
    for &n_gpu_layers in &[0, 8, 16, 24, -1] {
        let config_name = format!("memory_layers_{}", n_gpu_layers);

        group.bench_function(&config_name, |b| {
            b.iter(|| {
                // Measure session creation time (includes GPU memory allocation)
                let session = BitnetSession::create(model_path, TEST_CONTEXT_SIZE, n_gpu_layers)
                    .expect("Failed to create session for memory benchmark");

                black_box(session);
            });
        });
    }

    group.finish();
}

/// Benchmark: GPU layer configuration overhead
///
/// **Test Objective**: Measure overhead of GPU layer offloading configuration
///
/// **Expected Outcome**: Negligible overhead (< 10ms) for layer configuration
#[cfg(feature = "ffi")]
fn bench_gpu_config_overhead(c: &mut Criterion) {
    let model_path = Path::new(TEST_MODEL_PATH);

    if !model_path.exists() {
        eprintln!("WARNING: Test model not found for config overhead benchmark");
        return;
    }

    c.bench_function("gpu_config_overhead", |b| {
        b.iter(|| {
            // Measure time from BitnetSession::create call to first inference
            let session = BitnetSession::create(model_path, TEST_CONTEXT_SIZE, 24)
                .expect("Failed to create session for overhead benchmark");

            // NOTE: First inference includes configuration overhead
            unimplemented!("First inference timing not yet implemented");

            // Expected implementation:
            // let tokens = vec![1, 2];
            // let logits = session.eval_and_get_logits(&tokens, 0)
            //     .expect("Failed first inference");
            //
            // black_box(logits);
        });
    });
}

// Benchmark group configuration
#[cfg(feature = "ffi")]
criterion_group! {
    name = gpu_benches;
    config = Criterion::default()
        .sample_size(10)       // Reduced sample size for slow GPU initialization
        .measurement_time(std::time::Duration::from_secs(30)); // 30s per benchmark
    targets = bench_gpu_speedup, bench_gpu_memory_scaling, bench_gpu_config_overhead
}

// Fallback benchmark group when FFI not available
#[cfg(not(feature = "ffi"))]
criterion_group!(gpu_benches, bench_placeholder);

#[cfg(not(feature = "ffi"))]
fn bench_placeholder(_c: &mut Criterion) {
    eprintln!("GPU benchmarks require --features ffi");
}

criterion_main!(gpu_benches);

// =============================================================================
// Benchmark Execution Instructions
// =============================================================================
//
// **Run GPU benchmarks**:
// ```bash
// cargo bench --bench gpu_offloading_bench --features ffi
// ```
//
// **Run with GPU feature enabled** (requires CUDA):
// ```bash
// cargo bench --bench gpu_offloading_bench --features ffi,gpu
// ```
//
// **Run specific benchmark**:
// ```bash
// cargo bench --bench gpu_offloading_bench --features ffi -- gpu_offloading
// ```
//
// **Generate criterion reports**:
// Criterion automatically generates HTML reports in `target/criterion/`
//
// **Interpret results**:
// - Look for "change" column showing speedup relative to CPU baseline
// - GPU 24 layers should show ≥5× improvement (AC10 requirement)
// - Memory scaling should show linear VRAM usage with layer count
//
// **Troubleshooting**:
// - "Test model not found": Run `cargo run -p xtask -- download-model`
// - "GPU unavailable": Verify CUDA_VISIBLE_DEVICES and CUDA runtime
// - "Benchmark timeout": Increase `measurement_time` for slow hardware
//
// **Next Steps**:
// 1. Implement eval_and_get_logits() method for Socket 1
// 2. Add VRAM usage tracking via CUDA runtime APIs
// 3. Enable benchmarks incrementally as implementation progresses
// 4. Collect baseline performance data on reference hardware
