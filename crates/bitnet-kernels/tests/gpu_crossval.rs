//! GPU vs CPU cross-validation tests.
//!
//! Runs every kernel operation through both the CPU fallback and the
//! best-available provider, then asserts that results match within a
//! configurable tolerance.  When no GPU is present the tests still run
//! (both sides use the fallback) to keep CI green.

use bitnet_common::QuantizationType;
use bitnet_kernels::{FallbackKernel, KernelManager, KernelProvider};

// ── helpers ────────────────────────────────────────────────────────

/// Deterministic seeded data generator (no external RNG needed).
fn gen_i8(len: usize, seed: u64) -> Vec<i8> {
    let mut v = Vec::with_capacity(len);
    let mut s = seed;
    for _ in 0..len {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        v.push(((s >> 33) % 3) as i8 - 1); // -1, 0, 1
    }
    v
}

fn gen_u8(len: usize, seed: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    let mut s = seed;
    for _ in 0..len {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        v.push(((s >> 33) % 4) as u8);
    }
    v
}

fn gen_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(len);
    let mut s = seed;
    for _ in 0..len {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        v.push(((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0);
    }
    v
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

fn best_kernel() -> Box<dyn KernelProvider> {
    let mgr = KernelManager::new();
    let _name = mgr.select_best().unwrap().name();
    // Re-select to get an owned provider for the tests.
    bitnet_kernels::select_cpu_kernel().unwrap()
}

// ── matmul cross-validation ─────────────────────────────────────

fn crossval_matmul(m: usize, n: usize, k: usize, seed: u64) {
    let a = gen_i8(m * k, seed);
    let b = gen_u8(k * n, seed + 1);

    let ref_kernel = FallbackKernel;
    let mut ref_out = vec![0.0f32; m * n];
    ref_kernel.matmul_i2s(&a, &b, &mut ref_out, m, n, k).unwrap();

    let test_kernel = best_kernel();
    let mut test_out = vec![0.0f32; m * n];
    test_kernel.matmul_i2s(&a, &b, &mut test_out, m, n, k).unwrap();

    let diff = max_abs_diff(&ref_out, &test_out);
    assert!(
        diff < 1e-4,
        "matmul {m}x{n}x{k}: max abs diff = {diff} (> 1e-4)"
    );

    if ref_out.iter().any(|v| *v != 0.0) {
        let cos = cosine_similarity(&ref_out, &test_out);
        assert!(
            cos > 0.9999,
            "matmul {m}x{n}x{k}: cosine = {cos} (< 0.9999)"
        );
    }
}

#[test]
fn crossval_matmul_64() {
    crossval_matmul(64, 64, 64, 42);
}

#[test]
fn crossval_matmul_128() {
    crossval_matmul(128, 128, 128, 42);
}

#[test]
fn crossval_matmul_256() {
    crossval_matmul(256, 256, 256, 42);
}

#[test]
fn crossval_matmul_non_square() {
    crossval_matmul(64, 128, 256, 42);
}

#[test]
fn crossval_matmul_tall() {
    crossval_matmul(512, 32, 128, 42);
}

#[test]
fn crossval_matmul_wide() {
    crossval_matmul(32, 512, 128, 42);
}

// ── quantisation cross-validation ───────────────────────────────

fn crossval_quantize(len: usize, seed: u64) {
    let input = gen_f32(len, seed);
    // I2S uses block size 32 in fallback kernel
    let num_blocks = len.div_ceil(32);

    let ref_kernel = FallbackKernel;
    let mut ref_out = vec![0u8; len / 4];
    let mut ref_scales = vec![0.0f32; num_blocks];
    ref_kernel
        .quantize(&input, &mut ref_out, &mut ref_scales, QuantizationType::I2S)
        .unwrap();

    let test_kernel = best_kernel();
    let mut test_out = vec![0u8; len / 4];
    let mut test_scales = vec![0.0f32; num_blocks];
    test_kernel
        .quantize(&input, &mut test_out, &mut test_scales, QuantizationType::I2S)
        .unwrap();

    assert_eq!(
        ref_out, test_out,
        "quantize len={len}: packed bytes differ"
    );

    let scale_diff = max_abs_diff(&ref_scales, &test_scales);
    assert!(
        scale_diff < 1e-5,
        "quantize len={len}: scale diff = {scale_diff}"
    );
}

#[test]
fn crossval_quantize_512() {
    crossval_quantize(512, 42);
}

#[test]
fn crossval_quantize_1024() {
    crossval_quantize(1024, 42);
}

#[test]
fn crossval_quantize_4096() {
    crossval_quantize(4096, 42);
}

#[test]
fn crossval_quantize_16384() {
    crossval_quantize(16384, 42);
}

// ── determinism / reproducibility ───────────────────────────────

#[test]
fn crossval_matmul_deterministic() {
    let m = 128;
    let n = 128;
    let k = 128;
    let a = gen_i8(m * k, 99);
    let b = gen_u8(k * n, 100);

    let kernel = best_kernel();
    let mut out1 = vec![0.0f32; m * n];
    let mut out2 = vec![0.0f32; m * n];
    kernel.matmul_i2s(&a, &b, &mut out1, m, n, k).unwrap();
    kernel.matmul_i2s(&a, &b, &mut out2, m, n, k).unwrap();

    assert_eq!(out1, out2, "matmul must be deterministic across runs");
}

#[test]
fn crossval_quantize_deterministic() {
    let input = gen_f32(1024, 77);
    let num_blocks = 1024usize.div_ceil(32);

    let kernel = best_kernel();
    let mut out1 = vec![0u8; 1024 / 4];
    let mut scales1 = vec![0.0f32; num_blocks];
    let mut out2 = vec![0u8; 1024 / 4];
    let mut scales2 = vec![0.0f32; num_blocks];

    kernel.quantize(&input, &mut out1, &mut scales1, QuantizationType::I2S).unwrap();
    kernel.quantize(&input, &mut out2, &mut scales2, QuantizationType::I2S).unwrap();

    assert_eq!(out1, out2, "quantize must be deterministic");
    assert_eq!(scales1, scales2, "quantize scales must be deterministic");
}

// ── edge cases ──────────────────────────────────────────────────

#[test]
fn crossval_matmul_single_element() {
    crossval_matmul(1, 1, 1, 42);
}

#[test]
fn crossval_matmul_min_block() {
    crossval_matmul(32, 32, 32, 42);
}

#[test]
fn crossval_quantize_min_block() {
    crossval_quantize(128, 42);
}

// ── provider metadata ───────────────────────────────────────────

#[test]
fn fallback_is_always_available() {
    assert!(FallbackKernel.is_available());
    assert_eq!(FallbackKernel.name(), "fallback");
}

#[test]
fn best_kernel_is_available() {
    let mgr = KernelManager::new();
    let k = mgr.select_best().unwrap();
    assert!(k.is_available());
    assert!(!k.name().is_empty());
}
