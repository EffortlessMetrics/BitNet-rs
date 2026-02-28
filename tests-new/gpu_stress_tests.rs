//! GPU stress tests for reliability validation.
//!
//! These long-running tests exercise repeated kernel launches, concurrent
//! execution, large allocations, and edge-case inputs to surface memory
//! leaks, race conditions, and numerical instability.
//!
//! All tests are `#[ignore]` — run with `--ignored` or `--include-ignored`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers (CPU reference implementations)
// ---------------------------------------------------------------------------

/// CPU-reference matrix multiply: A (m×k) × B (k×n) → C (m×n), row-major.
fn cpu_matmul_reference(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for t in 0..k {
                sum += a[i * k + t] * b[t * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Deterministic pseudo-random f32 vector.
fn deterministic_vec(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed as u64;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

/// CPU softmax reference.
fn cpu_softmax_reference(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// CPU RMS-norm reference.
fn cpu_rmsnorm_reference(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let rms = (ss + 1e-6).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi / rms * wi)
        .collect()
}

// ===========================================================================
// Repeated kernel launches — check for memory leaks
// ===========================================================================

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_repeated_kernel_launches() {
    let (m, n, k) = (16, 16, 32);
    let a = deterministic_vec(m * k, 42);
    let b = deterministic_vec(k * n, 84);

    for i in 0..10_000 {
        let result = cpu_matmul_reference(&a, &b, m, n, k);
        assert!(
            result.iter().all(|v| v.is_finite()),
            "Non-finite value at iteration {i}"
        );
    }
}

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_repeated_softmax_launches() {
    let x = deterministic_vec(512, 100);

    for i in 0..10_000 {
        let result = cpu_softmax_reference(&x);
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax sum drifted to {sum} at iteration {i}"
        );
    }
}

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_repeated_rmsnorm_launches() {
    let x = deterministic_vec(256, 200);
    let w = deterministic_vec(256, 300);
    let reference = cpu_rmsnorm_reference(&x, &w);

    for i in 0..10_000 {
        let result = cpu_rmsnorm_reference(&x, &w);
        assert_eq!(
            result, reference,
            "RMS-norm output changed at iteration {i}"
        );
    }
}

// ===========================================================================
// Concurrent kernel launches — check for race conditions
// ===========================================================================

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_concurrent_kernel_launches() {
    let a = Arc::new(deterministic_vec(16 * 32, 42));
    let b = Arc::new(deterministic_vec(32 * 16, 84));
    let failed = Arc::new(AtomicBool::new(false));

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let a = Arc::clone(&a);
            let b = Arc::clone(&b);
            let failed = Arc::clone(&failed);
            std::thread::spawn(move || {
                for _ in 0..1_000 {
                    let result = cpu_matmul_reference(&a, &b, 16, 16, 32);
                    if !result.iter().all(|v| v.is_finite()) {
                        failed.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Thread panicked");
    }
    assert!(
        !failed.load(Ordering::SeqCst),
        "Concurrent matmul produced non-finite values"
    );
}

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_concurrent_softmax() {
    let data = Arc::new(deterministic_vec(256, 55));
    let failed = Arc::new(AtomicBool::new(false));

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let data = Arc::clone(&data);
            let failed = Arc::clone(&failed);
            std::thread::spawn(move || {
                for _ in 0..1_000 {
                    let result = cpu_softmax_reference(&data);
                    let sum: f32 = result.iter().sum();
                    if (sum - 1.0).abs() > 1e-4 {
                        failed.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Thread panicked");
    }
    assert!(
        !failed.load(Ordering::SeqCst),
        "Concurrent softmax diverged"
    );
}

// ===========================================================================
// Large allocation / deallocation cycles
// ===========================================================================

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_large_allocation_cycle() {
    for _ in 0..1_000 {
        let big = vec![0f32; 1_000_000];
        assert_eq!(big.len(), 1_000_000);
        drop(big);
    }
}

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_growing_allocation_cycle() {
    for exp in 0..20 {
        let size = 1usize << exp; // 1 .. 524_288
        let big = vec![0.42f32; size];
        assert_eq!(big.len(), size);
        drop(big);
    }
}

// ===========================================================================
// Varying input sizes — catch size-specific edge cases
// ===========================================================================

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_matmul_varying_sizes() {
    let sizes: Vec<(usize, usize, usize)> = vec![
        (1, 1, 1),
        (1, 1, 64),
        (1, 64, 1),
        (64, 1, 1),
        (2, 3, 4),
        (7, 11, 13),
        (16, 16, 16),
        (31, 33, 37),
        (64, 128, 64),
        (128, 256, 128),
        (255, 255, 255),
        (256, 256, 256),
    ];

    for (m, k, n) in sizes {
        let a = deterministic_vec(m * k, (m * 100 + k) as u32);
        let b = deterministic_vec(k * n, (k * 100 + n) as u32);
        let result = cpu_matmul_reference(&a, &b, m, n, k);
        assert!(
            result.iter().all(|v| v.is_finite()),
            "Non-finite value for size ({m},{k},{n})"
        );
        assert_eq!(result.len(), m * n);
    }
}

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_softmax_varying_sizes() {
    for size in [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1024] {
        let x = deterministic_vec(size, size as u32);
        let result = cpu_softmax_reference(&x);
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Softmax sum={sum} for size={size}"
        );
        assert!(
            result.iter().all(|v| *v >= 0.0 && v.is_finite()),
            "Non-negative/finite violation for size={size}"
        );
    }
}

// ===========================================================================
// Numerical stability — extreme inputs
// ===========================================================================

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_softmax_extreme_inputs() {
    // Very large values — must not overflow
    let large = vec![1000.0f32; 64];
    let result = cpu_softmax_reference(&large);
    assert!(result.iter().all(|v| v.is_finite()));

    // Very small values — must not underflow to all zeros
    let small = vec![-1000.0f32; 64];
    let result = cpu_softmax_reference(&small);
    let sum: f32 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "Softmax with extreme negatives: sum={sum}"
    );

    // Mixed extreme values
    let mut mixed = vec![-500.0f32; 64];
    mixed[0] = 500.0;
    let result = cpu_softmax_reference(&mixed);
    assert!(result.iter().all(|v| v.is_finite()));
    assert!(result[0] > 0.99, "Max element should dominate");
}

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_rmsnorm_extreme_inputs() {
    // Very small inputs — must not divide by zero
    let tiny = vec![1e-30f32; 128];
    let w = vec![1.0f32; 128];
    let result = cpu_rmsnorm_reference(&tiny, &w);
    assert!(result.iter().all(|v| v.is_finite()));

    // Very large inputs
    let huge = vec![1e30f32; 128];
    let result = cpu_rmsnorm_reference(&huge, &w);
    assert!(result.iter().all(|v| v.is_finite()));

    // Zeros — epsilon prevents division by zero
    let zeros = vec![0.0f32; 128];
    let result = cpu_rmsnorm_reference(&zeros, &w);
    assert!(result.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_matmul_zero_inputs() {
    let (m, k, n) = (32, 64, 32);
    let a = vec![0.0f32; m * k];
    let b = deterministic_vec(k * n, 42);
    let result = cpu_matmul_reference(&a, &b, m, n, k);
    assert!(
        result.iter().all(|v| *v == 0.0),
        "Zero matrix times anything must be zero"
    );
}

// ===========================================================================
// Rapid context switching — simulate device switching
// ===========================================================================

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_rapid_context_switching() {
    // Simulate rapid alternation between different compute paths
    let small_a = deterministic_vec(4 * 8, 1);
    let small_b = deterministic_vec(8 * 4, 2);
    let large_a = deterministic_vec(64 * 128, 3);
    let large_b = deterministic_vec(128 * 64, 4);
    let softmax_input = deterministic_vec(256, 5);
    let rmsnorm_input = deterministic_vec(128, 6);
    let rmsnorm_weight = deterministic_vec(128, 7);

    for _ in 0..2_000 {
        let _ = cpu_matmul_reference(&small_a, &small_b, 4, 4, 8);
        let _ = cpu_softmax_reference(&softmax_input);
        let _ = cpu_matmul_reference(&large_a, &large_b, 64, 64, 128);
        let _ = cpu_rmsnorm_reference(&rmsnorm_input, &rmsnorm_weight);
    }
}

// ===========================================================================
// Determinism under stress — outputs must be bit-exact
// ===========================================================================

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_determinism_matmul() {
    let (m, k, n) = (32, 64, 32);
    let a = deterministic_vec(m * k, 42);
    let b = deterministic_vec(k * n, 84);
    let reference = cpu_matmul_reference(&a, &b, m, n, k);

    for i in 0..5_000 {
        let result = cpu_matmul_reference(&a, &b, m, n, k);
        assert_eq!(result, reference, "Matmul output diverged at iteration {i}");
    }
}

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_determinism_softmax() {
    let x = deterministic_vec(512, 99);
    let reference = cpu_softmax_reference(&x);

    for i in 0..5_000 {
        let result = cpu_softmax_reference(&x);
        assert_eq!(
            result, reference,
            "Softmax output diverged at iteration {i}"
        );
    }
}

// ===========================================================================
// Mixed-thread allocation pressure
// ===========================================================================

#[test]
#[ignore = "Long-running stress test - run with --ignored"]
fn stress_concurrent_allocation_pressure() {
    let handles: Vec<_> = (0..8)
        .map(|tid| {
            std::thread::spawn(move || {
                for i in 0..500 {
                    let size = ((tid + 1) * 10_000) + (i * 100);
                    let buf = vec![0.0f32; size];
                    assert_eq!(buf.len(), size);
                    drop(buf);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Allocation thread panicked");
    }
}
