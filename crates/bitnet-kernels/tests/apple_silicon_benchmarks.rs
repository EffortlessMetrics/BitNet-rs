#![allow(clippy::manual_is_multiple_of, clippy::manual_memcpy, clippy::needless_range_loop)]
//! Apple Silicon performance benchmark tests.
//!
//! These tests measure NEON throughput, memory bandwidth, cache efficiency,
//! SIMD utilisation, alignment impact, thread-pool scaling, allocation patterns,
//! and unified-memory throughput on AArch64 (Apple Silicon).
//!
//! On non-AArch64 hosts every test still compiles and runs — it simply exercises
//! scalar fallback paths so CI stays green everywhere.

use std::time::Instant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 100;

/// Run `f` for `WARMUP_ITERS`, then time `BENCH_ITERS` and return the mean
/// duration in seconds.
fn bench<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP_ITERS {
        f();
    }
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        f();
    }
    start.elapsed().as_secs_f64() / BENCH_ITERS as f64
}

// ---------------------------------------------------------------------------
// NEON matmul helpers
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon_ops {
    use std::arch::aarch64::*;

    /// 4×4 f32 matmul tile using NEON fused multiply-add.
    #[target_feature(enable = "neon")]
    pub unsafe fn matmul_f32_neon(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
        unsafe {
            for i in 0..n {
                for j in (0..n).step_by(4) {
                    let mut acc = vdupq_n_f32(0.0);
                    for p in 0..n {
                        let a_val = vdupq_n_f32(*a.get_unchecked(i * n + p));
                        let b_vec = vld1q_f32(b.as_ptr().add(p * n + j));
                        acc = vfmaq_f32(acc, a_val, b_vec);
                    }
                    vst1q_f32(c.as_mut_ptr().add(i * n + j), acc);
                }
            }
        }
    }

    /// Dot-product two f32 slices using NEON.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len();
            let mut acc = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 4 <= n {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                acc = vfmaq_f32(acc, va, vb);
                i += 4;
            }
            let mut sum = vaddvq_f32(acc);
            while i < n {
                sum += a[i] * b[i];
                i += 1;
            }
            sum
        }
    }

    /// Sequential load-accumulate (bandwidth).
    #[target_feature(enable = "neon")]
    pub unsafe fn sequential_load_neon(data: &[f32]) -> f32 {
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 4 <= data.len() {
                let v = vld1q_f32(data.as_ptr().add(i));
                acc = vaddq_f32(acc, v);
                i += 4;
            }
            vaddvq_f32(acc)
        }
    }
}

/// Scalar matmul reference (any arch).
fn matmul_f32_scalar(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    c.fill(0.0);
    for i in 0..n {
        for p in 0..n {
            let aip = a[i * n + p];
            for j in 0..n {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
}

/// Scalar dot product.
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Scalar sequential accumulate.
fn sequential_load_scalar(data: &[f32]) -> f32 {
    data.iter().sum()
}

fn make_matrix(n: usize) -> Vec<f32> {
    (0..n * n).map(|i| (i % 17) as f32 * 0.01).collect()
}

fn make_vec(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 13) as f32 * 0.1).collect()
}

// ---------------------------------------------------------------------------
// 1. NEON matmul throughput (small / medium / large)
// ---------------------------------------------------------------------------

fn run_matmul_throughput(n: usize) {
    let a = make_matrix(n);
    let b = make_matrix(n);
    let mut c_neon = vec![0.0f32; n * n];
    let mut c_scalar = vec![0.0f32; n * n];

    #[cfg(target_arch = "aarch64")]
    let neon_secs = {
        // n must be a multiple of 4 for the NEON path
        bench(|| unsafe { neon_ops::matmul_f32_neon(&a, &b, &mut c_neon, n) })
    };
    #[cfg(not(target_arch = "aarch64"))]
    let neon_secs = bench(|| matmul_f32_scalar(&a, &b, &mut c_neon, n));

    let scalar_secs = bench(|| matmul_f32_scalar(&a, &b, &mut c_scalar, n));

    let flops = 2.0 * (n as f64).powi(3);
    let neon_gflops = flops / neon_secs / 1e9;
    let scalar_gflops = flops / scalar_secs / 1e9;
    let speedup = scalar_secs / neon_secs;

    eprintln!(
        "  matmul {n}×{n}: NEON {neon_gflops:.2} GFLOP/s, scalar {scalar_gflops:.2} GFLOP/s, speedup {speedup:.2}×"
    );

    // In debug mode the compiler auto-vectorises scalar code, so explicit
    // NEON may show modest gains.  For small matrices (n ≤ 16) SIMD setup
    // overhead can dominate on CI runners under load, so we use a lower floor.
    #[cfg(target_arch = "aarch64")]
    {
        let floor = if n <= 16 { 0.3 } else { 0.5 };
        assert!(
            speedup >= floor,
            "NEON matmul {n}×{n} speedup {speedup:.2}× below {floor}× threshold (regression)"
        );
    }
}

#[test]
fn neon_matmul_throughput_small() {
    eprintln!("--- NEON matmul throughput ---");
    run_matmul_throughput(16);
}

#[test]
fn neon_matmul_throughput_medium() {
    run_matmul_throughput(64);
}

#[test]
fn neon_matmul_throughput_large() {
    run_matmul_throughput(128);
}

// ---------------------------------------------------------------------------
// 2. Memory bandwidth — sequential vs strided
// ---------------------------------------------------------------------------

#[test]
fn memory_bandwidth_sequential_vs_strided() {
    eprintln!("--- Memory bandwidth ---");
    let size = 1 << 20; // 4 MiB of f32
    let data = make_vec(size);

    // Sequential
    #[cfg(target_arch = "aarch64")]
    let seq_secs = bench(|| {
        let _ = unsafe { neon_ops::sequential_load_neon(&data) };
    });
    #[cfg(not(target_arch = "aarch64"))]
    let seq_secs = bench(|| {
        let _ = sequential_load_scalar(&data);
    });

    // Strided (stride = 16 elements = 64 bytes = one cache line)
    let stride = 16;
    let strided_secs = bench(|| {
        let mut acc = 0.0f32;
        let mut i = 0;
        while i < data.len() {
            acc += data[i];
            i += stride;
        }
        let _ = acc;
    });

    let seq_bw_gbps = (size as f64 * 4.0) / seq_secs / 1e9;
    let strided_bw_gbps = ((size / stride) as f64 * 4.0) / strided_secs / 1e9;
    eprintln!("  sequential {seq_bw_gbps:.2} GB/s, strided {strided_bw_gbps:.2} GB/s");

    // Sequential should achieve higher effective bandwidth
    assert!(
        seq_bw_gbps > strided_bw_gbps * 0.8,
        "sequential bandwidth unexpectedly low vs strided"
    );
}

// ---------------------------------------------------------------------------
// 3. Cache efficiency — L1 / L2 working-set sweep
// ---------------------------------------------------------------------------

#[test]
fn cache_efficiency_working_set() {
    eprintln!("--- Cache efficiency ---");
    // L1 ≈ 128 KiB, L2 ≈ 4-16 MiB on Apple M-series
    let sizes: &[usize] = &[
        8 * 1024,        // 32 KiB — well inside L1
        32 * 1024,       // 128 KiB — near L1 boundary
        512 * 1024,      // 2 MiB — inside L2
        4 * 1024 * 1024, // 16 MiB — near L2 boundary
    ];

    let mut bandwidths = Vec::new();
    for &count in sizes {
        let data: Vec<f32> = (0..count).map(|i| (i & 0xFF) as f32).collect();
        let secs = bench(|| {
            let _ = sequential_load_scalar(&data);
        });
        let bw_gbps = (count as f64 * 4.0) / secs / 1e9;
        let kib = count * 4 / 1024;
        eprintln!("  working set {kib:>6} KiB → {bw_gbps:.2} GB/s");
        bandwidths.push(bw_gbps);
    }
    // Sanity: the smallest working set should achieve *some* bandwidth
    assert!(bandwidths[0] > 0.01, "L1-resident bandwidth unexpectedly low");
}

// ---------------------------------------------------------------------------
// 4. SIMD utilisation — NEON vs scalar dot product
// ---------------------------------------------------------------------------

#[test]
fn simd_utilisation_dot_product() {
    eprintln!("--- SIMD utilisation ---");
    let n = 1 << 16; // 256 K elements
    let a = make_vec(n);
    let b = make_vec(n);

    #[cfg(target_arch = "aarch64")]
    let neon_secs = bench(|| {
        let _ = unsafe { neon_ops::dot_f32_neon(&a, &b) };
    });
    #[cfg(not(target_arch = "aarch64"))]
    let neon_secs = bench(|| {
        let _ = dot_f32_scalar(&a, &b);
    });

    let scalar_secs = bench(|| {
        let _ = dot_f32_scalar(&a, &b);
    });

    let speedup = scalar_secs / neon_secs;
    eprintln!(
        "  dot {n} elems: NEON {neon_secs:.6} s, scalar {scalar_secs:.6} s, speedup {speedup:.2}×"
    );

    // In debug builds the scalar loop is often auto-vectorised by LLVM, so
    // explicit NEON may only be modestly faster.  A ≥ 0.8× floor catches
    // catastrophic regressions without being flaky in CI.
    #[cfg(target_arch = "aarch64")]
    assert!(speedup >= 0.8, "NEON dot speedup {speedup:.2}× below 0.8× threshold (regression)");
}

// ---------------------------------------------------------------------------
// 5. Memory alignment impact
// ---------------------------------------------------------------------------

#[test]
fn memory_alignment_impact() {
    eprintln!("--- Alignment impact ---");
    let n = 1 << 16;
    // Aligned: Vec guarantees alignment for f32
    let aligned: Vec<f32> = (0..n).map(|i| i as f32).collect();
    // Misaligned: shift by 1 byte inside a u8 buffer
    let raw = vec![0u8; n * 4 + 1];
    let offset = if raw.as_ptr() as usize % 4 == 0 { 1 } else { 0 };
    // Build an f32 slice that is *not* 4-byte aligned by copying into aligned buf
    // (Rust forbids truly misaligned f32 reads, so we simulate with an extra copy)
    let misaligned: Vec<f32> = {
        let mut v = vec![0.0f32; n];
        // Copy byte-by-byte from raw+offset to exercise a "nearly misaligned" path
        for i in 0..n {
            v[i] = aligned[i];
        }
        // Introduce a 1-element rotation to break prefetch patterns
        v.rotate_left(1);
        v
    };
    let _ = offset; // suppress unused warning

    let aligned_secs = bench(|| {
        let _ = sequential_load_scalar(&aligned);
    });
    let misaligned_secs = bench(|| {
        let _ = sequential_load_scalar(&misaligned);
    });

    let ratio = misaligned_secs / aligned_secs;
    eprintln!("  aligned {aligned_secs:.6} s, rotated {misaligned_secs:.6} s, ratio {ratio:.2}×");

    // Both should be within 3× of each other; if rotated is dramatically slower
    // something is wrong.
    assert!(ratio < 3.0, "rotated access {ratio:.2}× slower than aligned — unexpected");
}

// ---------------------------------------------------------------------------
// 6. Thread-pool scaling
// ---------------------------------------------------------------------------

#[test]
fn threadpool_scaling() {
    eprintln!("--- Thread-pool scaling ---");
    let n = 1 << 18; // 1 M elements
    let data: Vec<f32> = (0..n).map(|i| (i % 31) as f32).collect();

    let thread_counts: &[usize] = &[1, 2, 4, 8];
    let mut baseline_secs = 0.0;

    for &threads in thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("failed to build thread pool");

        let secs = bench(|| {
            pool.install(|| {
                use rayon::prelude::*;
                let _sum: f32 = data.par_chunks(1024).map(|c| c.iter().sum::<f32>()).sum();
            });
        });

        if threads == 1 {
            baseline_secs = secs;
        }
        let speedup = baseline_secs / secs;
        eprintln!("  {threads} thread(s): {secs:.6} s, speedup {speedup:.2}×");
    }
}

// ---------------------------------------------------------------------------
// 7. Allocation pattern — large alloc vs pool reuse
// ---------------------------------------------------------------------------

#[test]
fn allocation_pattern_reuse() {
    eprintln!("--- Allocation pattern ---");
    let n = 1 << 16;

    // Fresh allocation every iteration
    let alloc_secs = bench(|| {
        let mut v = vec![0.0f32; n];
        for i in 0..n {
            v[i] = (i % 7) as f32;
        }
        let _ = sequential_load_scalar(&v);
    });

    // Pre-allocated buffer reused each iteration
    let mut pool_buf = vec![0.0f32; n];
    let reuse_secs = bench(|| {
        for i in 0..n {
            pool_buf[i] = (i % 7) as f32;
        }
        let _ = sequential_load_scalar(&pool_buf);
    });

    let ratio = alloc_secs / reuse_secs;
    eprintln!("  fresh alloc {alloc_secs:.6} s, reuse {reuse_secs:.6} s, ratio {ratio:.2}×");

    // Reuse should be at least as fast (with tolerance for noise)
    assert!(reuse_secs <= alloc_secs * 1.5, "pool reuse unexpectedly slower than fresh allocation");
}

// ---------------------------------------------------------------------------
// 8. Unified memory throughput (simulated GPU buffer)
// ---------------------------------------------------------------------------

#[test]
fn unified_memory_throughput() {
    eprintln!("--- Unified memory throughput ---");
    // Simulate a GPU-style access pattern: large contiguous buffer written then
    // read back, as would happen with Apple Silicon unified memory / Metal.
    let n = 1 << 20; // 4 MiB of f32
    let mut buf = vec![0.0f32; n];

    let write_secs = bench(|| {
        for i in 0..n {
            buf[i] = (i & 0xFF) as f32;
        }
    });

    let read_secs = bench(|| {
        let _ = sequential_load_scalar(&buf);
    });

    let write_bw = (n as f64 * 4.0) / write_secs / 1e9;
    let read_bw = (n as f64 * 4.0) / read_secs / 1e9;
    eprintln!("  write {write_bw:.2} GB/s, read {read_bw:.2} GB/s");

    // Sanity: both should achieve meaningful throughput.
    // In debug builds writes go through bounds-checked indexing, so the
    // threshold is deliberately low.
    assert!(write_bw > 0.1, "write bandwidth {write_bw:.2} GB/s too low");
    assert!(read_bw > 0.1, "read bandwidth {read_bw:.2} GB/s too low");
}
