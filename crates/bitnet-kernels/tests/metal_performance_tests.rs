#![cfg(all(target_os = "macos", feature = "cpu"))]

//! Metal GPU performance counter and timing infrastructure tests for Apple Silicon.
//!
//! All tests validate calculations and models, not actual GPU measurements.

/// Apple M-series GPU specifications for validation.
struct AppleGpuSpec {
    name: &'static str,
    gpu_cores: u32,
    max_clock_mhz: u32,
    /// Memory bandwidth in GB/s
    memory_bandwidth_gbs: f64,
    /// Threads per SIMD group (always 32 on Apple GPUs)
    threads_per_simdgroup: u32,
    /// Maximum threads per threadgroup
    max_threads_per_threadgroup: u32,
    /// TDP in watts
    tdp_watts: f64,
}

const M1: AppleGpuSpec = AppleGpuSpec {
    name: "M1",
    gpu_cores: 8,
    max_clock_mhz: 1278,
    memory_bandwidth_gbs: 68.25,
    threads_per_simdgroup: 32,
    max_threads_per_threadgroup: 1024,
    tdp_watts: 20.0,
};

const M1_PRO: AppleGpuSpec = AppleGpuSpec {
    name: "M1 Pro",
    gpu_cores: 16,
    max_clock_mhz: 1296,
    memory_bandwidth_gbs: 200.0,
    threads_per_simdgroup: 32,
    max_threads_per_threadgroup: 1024,
    tdp_watts: 30.0,
};

const M2: AppleGpuSpec = AppleGpuSpec {
    name: "M2",
    gpu_cores: 10,
    max_clock_mhz: 1398,
    memory_bandwidth_gbs: 100.0,
    threads_per_simdgroup: 32,
    max_threads_per_threadgroup: 1024,
    tdp_watts: 22.0,
};

const M3: AppleGpuSpec = AppleGpuSpec {
    name: "M3",
    gpu_cores: 10,
    max_clock_mhz: 1380,
    memory_bandwidth_gbs: 100.0,
    threads_per_simdgroup: 32,
    max_threads_per_threadgroup: 1024,
    tdp_watts: 22.0,
};

const M4: AppleGpuSpec = AppleGpuSpec {
    name: "M4",
    gpu_cores: 10,
    max_clock_mhz: 1398,
    memory_bandwidth_gbs: 120.0,
    threads_per_simdgroup: 32,
    max_threads_per_threadgroup: 1024,
    tdp_watts: 22.0,
};

/// Calculate GPU timestamp delta in nanoseconds.
fn timestamp_delta_ns(start_ticks: u64, end_ticks: u64, resolution_ns: f64) -> f64 {
    (end_ticks - start_ticks) as f64 * resolution_ns
}

/// Calculate expected dispatch time in microseconds.
fn expected_dispatch_time_us(
    workgroup_count: u64,
    estimated_cycles_per_group: u64,
    clock_mhz: u32,
) -> f64 {
    let total_cycles = workgroup_count * estimated_cycles_per_group;
    // cycles / (cycles_per_second) = seconds; convert to microseconds
    total_cycles as f64 / (clock_mhz as f64) // MHz cancels to μs
}

/// Estimate memory bandwidth achieved in GB/s.
fn estimate_bandwidth_gbs(transfer_bytes: u64, time_ns: f64) -> f64 {
    let time_s = time_ns / 1e9;
    (transfer_bytes as f64) / (time_s * 1e9) // bytes/s → GB/s
}

/// Calculate theoretical GPU occupancy.
fn calculate_occupancy(active_threads: u32, max_threads_per_gpu: u32) -> f64 {
    active_threads as f64 / max_threads_per_gpu as f64
}

/// Estimate register pressure from kernel complexity.
fn estimate_register_usage(temp_variables: u32, loop_depth: u32, uses_matrix_ops: bool) -> u32 {
    let base = temp_variables;
    let loop_overhead = loop_depth * 2; // index + accumulator per loop
    let matrix_overhead = if uses_matrix_ops { 16 } else { 0 }; // matrix tile registers
    base + loop_overhead + matrix_overhead
}

/// Model cache hit rate based on working set vs cache size.
fn model_cache_hit_rate(working_set_bytes: u64, cache_size_bytes: u64) -> f64 {
    if working_set_bytes <= cache_size_bytes {
        // Entire working set fits: near-perfect hit rate
        1.0 - (working_set_bytes as f64 / cache_size_bytes as f64) * 0.02
    } else {
        // Exceeds cache: hit rate degrades proportionally
        let ratio = cache_size_bytes as f64 / working_set_bytes as f64;
        ratio.min(1.0).max(0.0)
    }
}

/// Calculate SIMD utilization efficiency.
fn simd_utilization(active_lanes: u32, total_lanes: u32) -> f64 {
    active_lanes as f64 / total_lanes as f64
}

/// Estimate performance per watt.
fn perf_per_watt(operations_per_second: f64, tdp_watts: f64) -> f64 {
    operations_per_second / tdp_watts
}

/// Calculate latency hiding ratio (computation / memory latency).
fn latency_hiding_ratio(computation_ns: f64, memory_latency_ns: f64) -> f64 {
    computation_ns / memory_latency_ns
}

#[test]
fn test_gpu_timestamp_resolution() {
    // Metal GPU timestamp counters on Apple Silicon typically have 1ns resolution
    let resolution_ns = 1.0_f64;

    let start_ticks: u64 = 1_000_000;
    let end_ticks: u64 = 1_001_000;
    let delta = timestamp_delta_ns(start_ticks, end_ticks, resolution_ns);

    assert!((delta - 1000.0).abs() < f64::EPSILON, "Expected 1000ns delta, got {delta}ns");

    // Verify sub-microsecond precision
    let fine_start: u64 = 500;
    let fine_end: u64 = 750;
    let fine_delta = timestamp_delta_ns(fine_start, fine_end, resolution_ns);
    assert!(
        (fine_delta - 250.0).abs() < f64::EPSILON,
        "Expected 250ns for fine-grained timing, got {fine_delta}ns"
    );

    // Verify with non-1ns resolution (some older GPUs)
    let coarse_resolution = 41.667; // ~24 MHz counter
    let coarse_delta = timestamp_delta_ns(100, 200, coarse_resolution);
    assert!(
        (coarse_delta - 4166.7).abs() < 0.1,
        "Expected ~4166.7ns with coarse counter, got {coarse_delta}ns"
    );
}

#[test]
fn test_dispatch_timing_calculation() {
    // M1 GPU: 1278 MHz, 256 workgroups, 100 cycles each
    let time_us = expected_dispatch_time_us(256, 100, M1.max_clock_mhz);
    // 256 * 100 = 25600 cycles; 25600 / 1278 ≈ 20.03 μs
    assert!((time_us - 20.03).abs() < 0.1, "Expected ~20.03μs dispatch time, got {time_us}μs");

    // Larger workload: 1024 groups, 500 cycles
    let large_time = expected_dispatch_time_us(1024, 500, M2.max_clock_mhz);
    // 512000 / 1398 ≈ 366.24 μs
    assert!(
        (large_time - 366.24).abs() < 0.5,
        "Expected ~366.24μs for large dispatch, got {large_time}μs"
    );

    // Single workgroup: minimal dispatch
    let minimal = expected_dispatch_time_us(1, 10, M1.max_clock_mhz);
    assert!(minimal < 0.01, "Single workgroup should be < 0.01μs compute, got {minimal}μs");
}

#[test]
fn test_memory_bandwidth_estimation() {
    // Transfer 1 GB in 14.65ms → ~68.25 GB/s (M1 theoretical max)
    let time_ns = 14.65e6; // 14.65ms in ns
    let transfer_bytes = 1_000_000_000_u64; // 1 GB
    let bandwidth = estimate_bandwidth_gbs(transfer_bytes, time_ns);

    assert!(
        (bandwidth - M1.memory_bandwidth_gbs).abs() < 1.0,
        "Expected ~{} GB/s, got {bandwidth} GB/s",
        M1.memory_bandwidth_gbs
    );

    // Achievable bandwidth is typically 70-85% of theoretical
    let achievable_ratio = 0.80;
    let achievable = M1.memory_bandwidth_gbs * achievable_ratio;
    assert!(
        achievable > 50.0,
        "Achievable bandwidth should exceed 50 GB/s on M1, got {achievable}"
    );

    // M1 Pro has much higher bandwidth
    assert!(
        M1_PRO.memory_bandwidth_gbs > M1.memory_bandwidth_gbs * 2.0,
        "M1 Pro bandwidth should be >2× M1"
    );
}

#[test]
fn test_occupancy_calculation() {
    // Full occupancy: all threads active
    let max_threads = M1.gpu_cores * M1.max_threads_per_threadgroup;
    let full = calculate_occupancy(max_threads, max_threads);
    assert!((full - 1.0).abs() < f64::EPSILON, "Full occupancy should be 1.0, got {full}");

    // Half occupancy
    let half = calculate_occupancy(max_threads / 2, max_threads);
    assert!((half - 0.5).abs() < f64::EPSILON, "Half occupancy should be 0.5, got {half}");

    // Low occupancy scenario: single threadgroup on full GPU
    let low = calculate_occupancy(M1.max_threads_per_threadgroup, max_threads);
    let expected_low = 1.0 / M1.gpu_cores as f64;
    assert!(
        (low - expected_low).abs() < 0.001,
        "Single-threadgroup occupancy should be {expected_low}, got {low}"
    );

    // Occupancy always in [0, 1]
    assert!((0.0..=1.0).contains(&full));
    assert!((0.0..=1.0).contains(&half));
    assert!((0.0..=1.0).contains(&low));
}

#[test]
fn test_register_pressure_estimation() {
    // Simple kernel: few temps, no loops, no matrix ops
    let simple = estimate_register_usage(4, 0, false);
    assert_eq!(simple, 4, "Simple kernel should use 4 registers");

    // Loop kernel: adds index + accumulator per loop level
    let looped = estimate_register_usage(4, 3, false);
    assert_eq!(looped, 10, "Looped kernel: 4 base + 3*2 loop overhead = 10");

    // Matrix kernel: adds 16 for tile registers
    let matrix = estimate_register_usage(8, 1, true);
    assert_eq!(matrix, 26, "Matrix kernel: 8 base + 1*2 loop + 16 matrix = 26");

    // Complex kernel with everything
    let complex = estimate_register_usage(12, 4, true);
    assert_eq!(complex, 36, "Complex: 12 + 8 + 16 = 36 registers");

    // Apple GPUs have 32K registers per SIMD group (approximate)
    let max_registers_per_thread: u32 = 256;
    assert!(
        complex < max_registers_per_thread,
        "Register usage {complex} should be within limit {max_registers_per_thread}"
    );
}

#[test]
fn test_cache_hit_rate_modeling() {
    // Working set fits entirely in L1 cache (32 KB typical)
    let l1_size = 32 * 1024; // 32 KB
    let small_set = 16 * 1024; // 16 KB
    let hit_rate = model_cache_hit_rate(small_set, l1_size);
    assert!(hit_rate > 0.98, "Small working set should have >98% hit rate, got {hit_rate}");

    // Working set equals cache size
    let equal_hit = model_cache_hit_rate(l1_size, l1_size);
    assert!(equal_hit > 0.97, "Equal working set should still have high hit rate, got {equal_hit}");

    // Working set 2× cache: ~50% hit rate
    let double_set = l1_size * 2;
    let double_hit = model_cache_hit_rate(double_set, l1_size);
    assert!(
        (double_hit - 0.5).abs() < 0.01,
        "2× working set should yield ~50% hit rate, got {double_hit}"
    );

    // Working set 10× cache: ~10% hit rate
    let large_set = l1_size * 10;
    let large_hit = model_cache_hit_rate(large_set, l1_size);
    assert!(
        (large_hit - 0.1).abs() < 0.01,
        "10× working set should yield ~10% hit rate, got {large_hit}"
    );

    // L2 cache (Apple M1 ~8 MB shared)
    let l2_size: u64 = 8 * 1024 * 1024;
    let l2_fit = model_cache_hit_rate(4 * 1024 * 1024, l2_size);
    assert!(l2_fit > 0.98, "4MB in 8MB L2 should have high hit rate");
}

#[test]
fn test_simd_utilization_calculation() {
    // Full SIMD utilization: all 32 lanes active (Apple GPU SIMD width)
    let full = simd_utilization(32, 32);
    assert!((full - 1.0).abs() < f64::EPSILON, "Full SIMD should be 1.0, got {full}");

    // Divergent branch: only half lanes active
    let half = simd_utilization(16, 32);
    assert!((half - 0.5).abs() < f64::EPSILON, "Half lanes should be 0.5 utilization, got {half}");

    // Worst case: single lane active (highly divergent)
    let worst = simd_utilization(1, 32);
    assert!(
        (worst - 1.0 / 32.0).abs() < f64::EPSILON,
        "Single lane should be 1/32 utilization, got {worst}"
    );

    // Typical good workload: 28/32 lanes (some edge masking)
    let typical = simd_utilization(28, 32);
    assert!(typical > 0.85, "Typical workload should have >85% SIMD utilization");

    // SIMD width is always 32 on Apple GPUs
    assert_eq!(M1.threads_per_simdgroup, 32);
    assert_eq!(M2.threads_per_simdgroup, 32);
    assert_eq!(M3.threads_per_simdgroup, 32);
}

#[test]
fn test_power_efficiency_estimation() {
    // M1 GPU: estimate GFLOPS/W
    // ~2.6 TFLOPS FP32 theoretical for 8-core M1
    let m1_gflops = 2600.0_f64; // 2.6 TFLOPS
    let m1_ops_per_sec = m1_gflops * 1e9;
    let m1_ppw = perf_per_watt(m1_ops_per_sec, M1.tdp_watts);
    let m1_gflops_per_watt = m1_ppw / 1e9;

    assert!(
        m1_gflops_per_watt > 100.0,
        "M1 should achieve >100 GFLOPS/W, got {m1_gflops_per_watt}"
    );

    // M4 should be more efficient than M1 (better perf/watt)
    let m4_gflops = 3700.0_f64; // estimated ~3.7 TFLOPS
    let m4_ppw = perf_per_watt(m4_gflops * 1e9, M4.tdp_watts);
    let m4_gflops_per_watt = m4_ppw / 1e9;

    assert!(
        m4_gflops_per_watt > m1_gflops_per_watt,
        "M4 ({m4_gflops_per_watt} GFLOPS/W) should be more efficient than M1 ({m1_gflops_per_watt} GFLOPS/W)"
    );
}

#[test]
fn test_latency_hiding_ratio() {
    // Good latency hiding: computation >> memory latency
    let good = latency_hiding_ratio(1000.0, 100.0);
    assert!(good > 5.0, "Good latency hiding should have ratio >5, got {good}");

    // Poor latency hiding: memory bound
    let poor = latency_hiding_ratio(50.0, 200.0);
    assert!(poor < 1.0, "Memory-bound kernel should have ratio <1, got {poor}");

    // Balanced pipeline
    let balanced = latency_hiding_ratio(150.0, 150.0);
    assert!(
        (balanced - 1.0).abs() < f64::EPSILON,
        "Balanced pipeline should have ratio ~1.0, got {balanced}"
    );

    // Apple GPU memory latency is typically 100-200ns for L2 miss
    let typical_mem_latency_ns = 150.0;
    // FMA operations: ~4 cycles at ~1.3 GHz ≈ 3ns
    let fma_time_ns = 4.0 / 1.3; // cycles / GHz = ns
    let fma_ratio = latency_hiding_ratio(fma_time_ns, typical_mem_latency_ns);
    // Need many concurrent operations to hide memory latency
    assert!(fma_ratio < 0.1, "Single FMA can't hide memory latency; need concurrent ops");
}

#[test]
fn test_kernel_launch_overhead() {
    // Metal kernel launch overhead is typically ~2-10μs on Apple Silicon
    let min_launch_overhead_us = 2.0_f64;
    let max_launch_overhead_us = 10.0_f64;
    let typical_launch_overhead_us = 5.0_f64;

    assert!(
        typical_launch_overhead_us >= min_launch_overhead_us,
        "Typical launch overhead should be >= minimum"
    );
    assert!(
        typical_launch_overhead_us <= max_launch_overhead_us,
        "Typical launch overhead should be <= maximum"
    );

    // For small kernels, launch overhead dominates
    let small_kernel_compute_us = 1.0;
    let total_with_overhead = small_kernel_compute_us + typical_launch_overhead_us;
    let overhead_fraction = typical_launch_overhead_us / total_with_overhead;
    assert!(
        overhead_fraction > 0.8,
        "Launch overhead should dominate for tiny kernels, fraction: {overhead_fraction}"
    );

    // For large kernels, launch overhead is negligible
    let large_kernel_compute_us = 10_000.0;
    let large_total = large_kernel_compute_us + typical_launch_overhead_us;
    let large_overhead_fraction = typical_launch_overhead_us / large_total;
    assert!(
        large_overhead_fraction < 0.001,
        "Launch overhead should be negligible for large kernels, fraction: {large_overhead_fraction}"
    );

    // Batch amortization: N kernels share command buffer encoding
    let batch_size = 100_u32;
    let amortized_per_kernel = typical_launch_overhead_us / batch_size as f64;
    assert!(
        amortized_per_kernel < 0.1,
        "Batched launch should amortize to <0.1μs/kernel, got {amortized_per_kernel}"
    );
}

#[test]
fn test_apple_gpu_specifications() {
    let specs = [&M1, &M1_PRO, &M2, &M3, &M4];

    for spec in &specs {
        // All Apple GPUs use 32-wide SIMD groups
        assert_eq!(spec.threads_per_simdgroup, 32, "{}: SIMD width must be 32", spec.name);

        // Max threads per threadgroup is 1024
        assert_eq!(
            spec.max_threads_per_threadgroup, 1024,
            "{}: max threads per threadgroup must be 1024",
            spec.name
        );

        // Clock speeds in reasonable range (1000-2000 MHz)
        assert!(
            (1000..=2000).contains(&spec.max_clock_mhz),
            "{}: clock {} MHz out of range",
            spec.name,
            spec.max_clock_mhz
        );

        // GPU cores: at least 7, at most 80 (M2 Ultra)
        assert!(
            (7..=80).contains(&spec.gpu_cores),
            "{}: {} GPU cores out of range",
            spec.name,
            spec.gpu_cores
        );

        // Memory bandwidth positive and reasonable
        assert!(
            spec.memory_bandwidth_gbs > 50.0 && spec.memory_bandwidth_gbs < 1000.0,
            "{}: bandwidth {} GB/s out of range",
            spec.name,
            spec.memory_bandwidth_gbs
        );

        // TDP reasonable for Apple Silicon (10-120W range)
        assert!(
            spec.tdp_watts >= 10.0 && spec.tdp_watts <= 120.0,
            "{}: TDP {} W out of range",
            spec.name,
            spec.tdp_watts
        );
    }

    // M-series progression: later chips should have >= bandwidth
    assert!(M2.memory_bandwidth_gbs >= M1.memory_bandwidth_gbs);
    assert!(M4.memory_bandwidth_gbs >= M2.memory_bandwidth_gbs);

    // Pro has more cores than base
    assert!(M1_PRO.gpu_cores > M1.gpu_cores);
}

#[test]
fn test_performance_regression_thresholds() {
    // Define acceptable performance variance for stable workloads
    let stable_variance_pct = 5.0_f64;

    // Simulate baseline and measured performance
    let baseline_tflops = 2.6_f64;

    // Within threshold: ±5%
    let good_measurement = baseline_tflops * 0.97; // 3% regression
    let regression_pct = ((baseline_tflops - good_measurement) / baseline_tflops) * 100.0;
    assert!(
        regression_pct <= stable_variance_pct,
        "3% regression should be within {stable_variance_pct}% threshold"
    );

    // Outside threshold: >5% regression
    let bad_measurement = baseline_tflops * 0.90; // 10% regression
    let bad_regression_pct = ((baseline_tflops - bad_measurement) / baseline_tflops) * 100.0;
    assert!(
        bad_regression_pct > stable_variance_pct,
        "10% regression should exceed {stable_variance_pct}% threshold"
    );

    // Improvement should not trigger regression alert
    let improved = baseline_tflops * 1.05; // 5% improvement
    let improvement_pct = ((improved - baseline_tflops) / baseline_tflops) * 100.0;
    assert!(improvement_pct > 0.0, "Improvements should be positive, got {improvement_pct}%");

    // Bandwidth regression thresholds (tighter for memory-bound kernels)
    let bw_baseline = 68.0_f64; // GB/s
    let bw_threshold_pct = 3.0_f64; // tighter threshold
    let bw_measured = 66.0_f64;
    let bw_regression = ((bw_baseline - bw_measured) / bw_baseline) * 100.0;
    assert!(
        bw_regression < bw_threshold_pct,
        "Bandwidth regression {bw_regression:.1}% should be within {bw_threshold_pct}%"
    );

    // Latency regression thresholds (even tighter for latency-sensitive ops)
    let lat_baseline_us = 5.0_f64;
    let lat_threshold_pct = 2.0_f64;
    let lat_measured_us = 5.05_f64;
    let lat_regression = ((lat_measured_us - lat_baseline_us) / lat_baseline_us) * 100.0;
    assert!(
        lat_regression <= lat_threshold_pct,
        "Latency regression {lat_regression:.1}% should be within {lat_threshold_pct}%"
    );
}
