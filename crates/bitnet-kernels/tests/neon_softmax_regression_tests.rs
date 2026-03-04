#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
//! NEON softmax regression tests for Apple Silicon (aarch64).
//!
//! Validates NEON-accelerated softmax against the scalar CPU reference for
//! numerical parity, edge cases, and special float handling.
//!
//! Tests marked `#[ignore]` are TDD scaffolds gated on the NEON softmax wiring
//! PRs; the non-ignored tests validate the CPU reference independently.

#![cfg(all(feature = "cpu", target_arch = "aarch64"))]
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    clippy::manual_div_ceil,
    clippy::useless_vec,
    clippy::approx_constant,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::assertions_on_constants
)]

use bitnet_kernels::cpu::neon_softmax::{softmax_neon, softmax_neon_inplace, softmax_scalar};
use bitnet_kernels::softmax_utils::{is_valid_distribution, softmax_f32};

// ── Helpers ────────────────────────────────────────────────────────────

/// Tolerance for NEON-vs-scalar parity (fast exp polynomial has ~2e-4 max error).
const NEON_TOL: f32 = 5e-4;

/// Tolerance for exact-math reference checks.
const REF_TOL: f32 = 1e-6;

/// Safe wrapper around the NEON softmax.
fn run_neon_softmax(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    // SAFETY: test runs only on aarch64 (cfg gate above).
    unsafe { softmax_neon(input, &mut output) };
    output
}

/// Run the scalar reference softmax (out-of-place, matching NEON signature).
fn run_scalar_softmax(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    softmax_scalar(input, &mut output);
    output
}

/// Assert two slices are element-wise close within `tol`.
fn assert_slices_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch {} vs {}", a.len(), b.len());
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!((av - bv).abs() < tol, "{ctx}[{i}]: {av} vs {bv} (diff {})", (av - bv).abs());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 1. CPU scalar reference (non-ignored — validates independently)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn cpu_scalar_softmax_sums_to_one() {
    for &len in &[1, 2, 4, 7, 8, 13, 16, 33, 64, 128, 255, 256, 512] {
        let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let out = run_scalar_softmax(&input);
        let sum: f64 = out.iter().map(|&v| v as f64).sum();
        assert!((sum - 1.0).abs() < 1e-5, "len={len}: sum={sum}, expected 1.0");
    }
}

#[test]
fn cpu_scalar_softmax_monotonic_increasing() {
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let out = run_scalar_softmax(&input);
    for w in out.windows(2) {
        assert!(w[0] < w[1], "expected strictly increasing, got {} >= {}", w[0], w[1]);
    }
}

#[test]
fn cpu_scalar_softmax_uniform_input() {
    let input = vec![3.0f32; 8];
    let out = run_scalar_softmax(&input);
    let expected = 1.0 / 8.0;
    for (i, &v) in out.iter().enumerate() {
        assert!((v - expected).abs() < REF_TOL, "uniform[{i}]: expected {expected}, got {v}");
    }
}

#[test]
fn cpu_scalar_softmax_large_values_stable() {
    let input = vec![1000.0, 1001.0, 1002.0, 1003.0];
    let out = run_scalar_softmax(&input);
    assert!(is_valid_distribution(&out, 1e-5));
    for &v in &out {
        assert!(v.is_finite(), "expected finite, got {v}");
    }
}

#[test]
fn cpu_scalar_softmax_negative_values() {
    let input = vec![-10.0, -5.0, -1.0, 0.0, 1.0];
    let out = run_scalar_softmax(&input);
    assert!(is_valid_distribution(&out, 1e-5));
    for w in out.windows(2) {
        assert!(w[0] < w[1], "expected monotonic increase");
    }
}

#[test]
fn cpu_scalar_softmax_single_element() {
    let out = run_scalar_softmax(&[42.0]);
    assert!((out[0] - 1.0).abs() < REF_TOL, "single element should be 1.0, got {}", out[0]);
}

#[test]
fn cpu_scalar_softmax_empty() {
    let out = run_scalar_softmax(&[]);
    assert!(out.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
// 2. NEON-vs-scalar parity (TDD scaffolds — ignored until wiring lands)
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_parity_small_inputs() {
    for &len in &[1, 2, 3, 4, 5, 6, 7, 8] {
        let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let neon_out = run_neon_softmax(&input);
        let scalar_out = run_scalar_softmax(&input);
        assert_slices_close(&neon_out, &scalar_out, NEON_TOL, &format!("small len={len}"));
    }
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_parity_powers_of_two() {
    for &len in &[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let input: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.01).sin()).collect();
        let neon_out = run_neon_softmax(&input);
        let scalar_out = run_scalar_softmax(&input);
        assert_slices_close(&neon_out, &scalar_out, NEON_TOL, &format!("pow2 len={len}"));
    }
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_parity_non_power_of_two() {
    for &len in &[3, 5, 7, 9, 13, 17, 31, 33, 63, 65, 127, 129, 255, 257] {
        let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.3 - 5.0).collect();
        let neon_out = run_neon_softmax(&input);
        let scalar_out = run_scalar_softmax(&input);
        assert_slices_close(&neon_out, &scalar_out, NEON_TOL, &format!("non-pow2 len={len}"));
    }
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_parity_large_input() {
    let len = 4096;
    let input: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.001).cos() * 10.0).collect();
    let neon_out = run_neon_softmax(&input);
    let scalar_out = run_scalar_softmax(&input);
    assert_slices_close(&neon_out, &scalar_out, NEON_TOL, "large_4096");
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Edge cases — NEON path
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_empty_input() {
    let mut output: Vec<f32> = vec![];
    unsafe { softmax_neon(&[], &mut output) };
    assert!(output.is_empty());
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_single_element() {
    let out = run_neon_softmax(&[99.0]);
    assert!((out[0] - 1.0).abs() < 1e-3, "single element should be ~1.0, got {}", out[0]);
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_all_zeros() {
    let input = vec![0.0f32; 16];
    let out = run_neon_softmax(&input);
    let expected = 1.0 / 16.0;
    for (i, &v) in out.iter().enumerate() {
        assert!((v - expected).abs() < 1e-3, "all_zeros[{i}]: expected {expected}, got {v}");
    }
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_all_equal_negative() {
    let input = vec![-7.0f32; 32];
    let out = run_neon_softmax(&input);
    let expected = 1.0 / 32.0;
    for (i, &v) in out.iter().enumerate() {
        assert!((v - expected).abs() < 1e-3, "equal_neg[{i}]: expected {expected}, got {v}");
    }
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_large_magnitude_stable() {
    let input = vec![500.0, 501.0, 502.0, 503.0, 504.0, 505.0, 506.0, 507.0];
    let out = run_neon_softmax(&input);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-3, "large magnitude sum = {sum}");
    for &v in &out {
        assert!(v.is_finite(), "expected finite, got {v}");
        assert!(v >= 0.0, "expected non-negative, got {v}");
    }
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_very_negative_stable() {
    let input = vec![-500.0, -501.0, -502.0, -503.0];
    let out = run_neon_softmax(&input);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-3, "very negative sum = {sum}");
    for &v in &out {
        assert!(v.is_finite(), "expected finite, got {v}");
        assert!(v >= 0.0, "expected non-negative, got {v}");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 4. NaN / Inf handling — NEON path
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_nan_in_input() {
    let input = vec![1.0, f32::NAN, 3.0, 4.0];
    let out = run_neon_softmax(&input);
    // NaN propagation is implementation-defined; just ensure no panic and
    // the non-NaN elements are finite or NaN (not random garbage).
    assert_eq!(out.len(), 4);
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_positive_inf_in_input() {
    let input = vec![1.0, f32::INFINITY, 3.0, 4.0];
    let out = run_neon_softmax(&input);
    assert_eq!(out.len(), 4);
    // The +Inf element should dominate: its softmax value should be ~1.0.
    // Others should be ~0.0 (or NaN — clamping may prevent this).
    // Mainly assert no panic.
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_negative_inf_in_input() {
    let input = vec![f32::NEG_INFINITY, 1.0, 2.0, 3.0];
    let out = run_neon_softmax(&input);
    assert_eq!(out.len(), 4);
    // -Inf element should have probability ~0.0 after max subtraction.
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_all_inf() {
    let input = vec![f32::INFINITY; 4];
    let out = run_neon_softmax(&input);
    assert_eq!(out.len(), 4);
    // All-Inf triggers 0/0 after max subtraction; just assert no panic.
}

// ═══════════════════════════════════════════════════════════════════════
// 5. In-place NEON softmax
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_inplace_matches_out_of_place() {
    let input: Vec<f32> = (0..17).map(|i| (i as f32) * 0.3 - 2.5).collect();
    let out_of_place = run_neon_softmax(&input);

    let mut inplace = input.clone();
    unsafe { softmax_neon_inplace(&mut inplace) };

    assert_slices_close(&inplace, &out_of_place, 1e-6, "inplace_vs_oop");
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_inplace_empty() {
    let mut data: Vec<f32> = vec![];
    unsafe { softmax_neon_inplace(&mut data) };
    assert!(data.is_empty());
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_inplace_single() {
    let mut data = vec![42.0f32];
    unsafe { softmax_neon_inplace(&mut data) };
    assert!((data[0] - 1.0).abs() < 1e-3, "inplace single = {}", data[0]);
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Distribution invariants — NEON path
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_output_is_valid_distribution() {
    for &len in &[4, 8, 15, 16, 17, 32, 64, 100, 128, 256] {
        let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.2 - 3.0).collect();
        let out = run_neon_softmax(&input);
        assert!(is_valid_distribution(&out, 1e-3), "len={len}: not a valid distribution");
    }
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_output_monotonic_for_sorted_input() {
    let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let out = run_neon_softmax(&input);
    for w in out.windows(2) {
        assert!(w[0] < w[1], "expected monotonic increase, got {} >= {}", w[0], w[1]);
    }
}

#[test]
#[ignore = "TDD scaffold: NEON softmax regression test"]
fn neon_softmax_is_deterministic() {
    let input: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.7).sin()).collect();
    let out1 = run_neon_softmax(&input);
    let out2 = run_neon_softmax(&input);
    assert_eq!(out1, out2, "NEON softmax must be deterministic");
}
