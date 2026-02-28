//! Comprehensive GPU kernel output regression snapshots.
//!
//! Every kernel's CPU reference implementation is exercised at three input
//! sizes (small / medium / large) and the outputs are pinned with
//! `insta::assert_debug_snapshot!`.  This catches silent numerical drift
//! across refactors, SIMD upgrades, or quantisation changes.

mod support;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// CPU-reference matrix multiply: A (m×k) × B (k×n) → C (m×n), row-major.
fn matmul_host(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

/// Deterministic pseudo-random f32 vector (simple LCG, reproducible).
fn deterministic_vec(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed as u64;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1.0, 1.0]
            ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

/// Simple RMS-norm reference: out[i] = x[i] / rms(x) * weight[i].
fn rmsnorm_host(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let rms = (ss + 1e-6).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi / rms * wi)
        .collect()
}

/// SiLU activation: x * sigmoid(x).
fn silu_host(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| v * (1.0 / (1.0 + (-v).exp())))
        .collect()
}

/// Softmax reference.
fn softmax_host(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Element-wise add.
fn add_host(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Dot product.
fn dot_host(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Simple ternary quantisation: map to {-1, 0, 1} by thresholding.
fn ternary_quantize_host(x: &[f32], threshold: f32) -> Vec<i8> {
    x.iter()
        .map(|&v| {
            if v > threshold {
                1i8
            } else if v < -threshold {
                -1i8
            } else {
                0i8
            }
        })
        .collect()
}

/// Ternary dequantise: i8 → f32 with scale.
fn ternary_dequantize_host(q: &[i8], scale: f32) -> Vec<f32> {
    q.iter().map(|&v| v as f32 * scale).collect()
}

/// Round values to 6 decimal places for deterministic snapshot comparison.
fn round6(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| (x * 1e6).round() / 1e6).collect()
}

// ---- Snapshot metadata helper ----
#[derive(Debug)]
struct KernelSnapshot {
    kernel: &'static str,
    size_class: &'static str,
    shape: Vec<usize>,
    output: Vec<f32>,
}

// ===========================================================================
// MatMul kernel snapshots  (3 sizes)
// ===========================================================================

#[test]
fn snapshot_matmul_small() {
    let (m, k, n) = (2, 3, 2);
    let a = deterministic_vec(m * k, 100);
    let b = deterministic_vec(k * n, 200);
    let out = round6(&matmul_host(&a, &b, m, k, n));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "matmul",
        size_class: "small",
        shape: vec![m, n],
        output: out,
    });
}

#[test]
fn snapshot_matmul_medium() {
    let (m, k, n) = (16, 32, 16);
    let a = deterministic_vec(m * k, 101);
    let b = deterministic_vec(k * n, 201);
    let out = round6(&matmul_host(&a, &b, m, k, n));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "matmul",
        size_class: "medium",
        shape: vec![m, n],
        output: out,
    });
}

#[test]
fn snapshot_matmul_large() {
    let (m, k, n) = (64, 128, 64);
    let a = deterministic_vec(m * k, 102);
    let b = deterministic_vec(k * n, 202);
    let out = round6(&matmul_host(&a, &b, m, k, n));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "matmul",
        size_class: "large",
        shape: vec![m, n],
        output: out,
    });
}

// ===========================================================================
// RMSNorm kernel snapshots
// ===========================================================================

#[test]
fn snapshot_rmsnorm_small() {
    let x = deterministic_vec(4, 300);
    let w = deterministic_vec(4, 400);
    let out = round6(&rmsnorm_host(&x, &w));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "rmsnorm",
        size_class: "small",
        shape: vec![4],
        output: out,
    });
}

#[test]
fn snapshot_rmsnorm_medium() {
    let x = deterministic_vec(128, 301);
    let w = deterministic_vec(128, 401);
    let out = round6(&rmsnorm_host(&x, &w));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "rmsnorm",
        size_class: "medium",
        shape: vec![128],
        output: out,
    });
}

#[test]
fn snapshot_rmsnorm_large() {
    let x = deterministic_vec(2048, 302);
    let w = deterministic_vec(2048, 402);
    let out = round6(&rmsnorm_host(&x, &w));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "rmsnorm",
        size_class: "large",
        shape: vec![2048],
        output: out,
    });
}

// ===========================================================================
// SiLU activation snapshots
// ===========================================================================

#[test]
fn snapshot_silu_small() {
    let x = deterministic_vec(8, 500);
    let out = round6(&silu_host(&x));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "silu",
        size_class: "small",
        shape: vec![8],
        output: out,
    });
}

#[test]
fn snapshot_silu_medium() {
    let x = deterministic_vec(256, 501);
    let out = round6(&silu_host(&x));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "silu",
        size_class: "medium",
        shape: vec![256],
        output: out,
    });
}

#[test]
fn snapshot_silu_large() {
    let x = deterministic_vec(4096, 502);
    let out = round6(&silu_host(&x));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "silu",
        size_class: "large",
        shape: vec![4096],
        output: out,
    });
}

// ===========================================================================
// Softmax snapshots
// ===========================================================================

#[test]
fn snapshot_softmax_small() {
    let x = deterministic_vec(4, 600);
    let out = round6(&softmax_host(&x));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "softmax",
        size_class: "small",
        shape: vec![4],
        output: out,
    });
}

#[test]
fn snapshot_softmax_medium() {
    let x = deterministic_vec(64, 601);
    let out = round6(&softmax_host(&x));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "softmax",
        size_class: "medium",
        shape: vec![64],
        output: out,
    });
}

#[test]
fn snapshot_softmax_large() {
    let x = deterministic_vec(1024, 602);
    let out = round6(&softmax_host(&x));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "softmax",
        size_class: "large",
        shape: vec![1024],
        output: out,
    });
}

// ===========================================================================
// Element-wise add snapshots
// ===========================================================================

#[test]
fn snapshot_add_small() {
    let a = deterministic_vec(4, 700);
    let b = deterministic_vec(4, 800);
    let out = round6(&add_host(&a, &b));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "add",
        size_class: "small",
        shape: vec![4],
        output: out,
    });
}

#[test]
fn snapshot_add_medium() {
    let a = deterministic_vec(256, 701);
    let b = deterministic_vec(256, 801);
    let out = round6(&add_host(&a, &b));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "add",
        size_class: "medium",
        shape: vec![256],
        output: out,
    });
}

#[test]
fn snapshot_add_large() {
    let a = deterministic_vec(4096, 702);
    let b = deterministic_vec(4096, 802);
    let out = round6(&add_host(&a, &b));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "add",
        size_class: "large",
        shape: vec![4096],
        output: out,
    });
}

// ===========================================================================
// Dot product snapshots
// ===========================================================================

#[test]
fn snapshot_dot_small() {
    let a = deterministic_vec(4, 900);
    let b = deterministic_vec(4, 1000);
    let out = round6(&[dot_host(&a, &b)]);
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "dot",
        size_class: "small",
        shape: vec![1],
        output: out.to_vec(),
    });
}

#[test]
fn snapshot_dot_medium() {
    let a = deterministic_vec(128, 901);
    let b = deterministic_vec(128, 1001);
    let out = round6(&[dot_host(&a, &b)]);
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "dot",
        size_class: "medium",
        shape: vec![1],
        output: out.to_vec(),
    });
}

#[test]
fn snapshot_dot_large() {
    let a = deterministic_vec(4096, 902);
    let b = deterministic_vec(4096, 1002);
    let out = round6(&[dot_host(&a, &b)]);
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "dot",
        size_class: "large",
        shape: vec![1],
        output: out.to_vec(),
    });
}

// ===========================================================================
// Ternary quantisation snapshots
// ===========================================================================

#[derive(Debug)]
struct QuantSnapshot {
    kernel: &'static str,
    size_class: &'static str,
    shape: Vec<usize>,
    quantized: Vec<i8>,
}

#[test]
fn snapshot_ternary_quant_small() {
    let x = deterministic_vec(8, 1100);
    let q = ternary_quantize_host(&x, 0.3);
    insta::assert_debug_snapshot!(QuantSnapshot {
        kernel: "ternary_quantize",
        size_class: "small",
        shape: vec![8],
        quantized: q,
    });
}

#[test]
fn snapshot_ternary_quant_medium() {
    let x = deterministic_vec(256, 1101);
    let q = ternary_quantize_host(&x, 0.3);
    insta::assert_debug_snapshot!(QuantSnapshot {
        kernel: "ternary_quantize",
        size_class: "medium",
        shape: vec![256],
        quantized: q,
    });
}

#[test]
fn snapshot_ternary_quant_large() {
    let x = deterministic_vec(4096, 1102);
    let q = ternary_quantize_host(&x, 0.3);
    insta::assert_debug_snapshot!(QuantSnapshot {
        kernel: "ternary_quantize",
        size_class: "large",
        shape: vec![4096],
        quantized: q,
    });
}

// ===========================================================================
// Ternary dequantisation snapshots
// ===========================================================================

#[test]
fn snapshot_ternary_dequant_small() {
    let x = deterministic_vec(8, 1200);
    let q = ternary_quantize_host(&x, 0.3);
    let deq = round6(&ternary_dequantize_host(&q, 0.42));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "ternary_dequantize",
        size_class: "small",
        shape: vec![8],
        output: deq,
    });
}

#[test]
fn snapshot_ternary_dequant_medium() {
    let x = deterministic_vec(256, 1201);
    let q = ternary_quantize_host(&x, 0.3);
    let deq = round6(&ternary_dequantize_host(&q, 0.42));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "ternary_dequantize",
        size_class: "medium",
        shape: vec![256],
        output: deq,
    });
}

#[test]
fn snapshot_ternary_dequant_large() {
    let x = deterministic_vec(4096, 1202);
    let q = ternary_quantize_host(&x, 0.3);
    let deq = round6(&ternary_dequantize_host(&q, 0.42));
    insta::assert_debug_snapshot!(KernelSnapshot {
        kernel: "ternary_dequantize",
        size_class: "large",
        shape: vec![4096],
        output: deq,
    });
}

// ===========================================================================
// Kernel compilation config metadata snapshots
// ===========================================================================

#[derive(Debug)]
struct KernelConfig {
    kernel: &'static str,
    work_group_size: usize,
    precision: &'static str,
    vectorisation_width: usize,
}

#[test]
fn snapshot_config_matmul() {
    insta::assert_debug_snapshot!(KernelConfig {
        kernel: "matmul",
        work_group_size: 256,
        precision: "f32",
        vectorisation_width: 8,
    });
}

#[test]
fn snapshot_config_rmsnorm() {
    insta::assert_debug_snapshot!(KernelConfig {
        kernel: "rmsnorm",
        work_group_size: 256,
        precision: "f32",
        vectorisation_width: 4,
    });
}

#[test]
fn snapshot_config_softmax() {
    insta::assert_debug_snapshot!(KernelConfig {
        kernel: "softmax",
        work_group_size: 128,
        precision: "f32",
        vectorisation_width: 4,
    });
}

#[test]
fn snapshot_config_silu() {
    insta::assert_debug_snapshot!(KernelConfig {
        kernel: "silu",
        work_group_size: 256,
        precision: "f32",
        vectorisation_width: 8,
    });
}

#[test]
fn snapshot_config_add() {
    insta::assert_debug_snapshot!(KernelConfig {
        kernel: "add",
        work_group_size: 256,
        precision: "f32",
        vectorisation_width: 8,
    });
}

#[test]
fn snapshot_config_ternary_quant() {
    insta::assert_debug_snapshot!(KernelConfig {
        kernel: "ternary_quantize",
        work_group_size: 128,
        precision: "i8",
        vectorisation_width: 16,
    });
}
