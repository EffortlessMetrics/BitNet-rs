//! Snapshot wave 13 — stabilize bitnet-rocm public API surface.
//!
//! Pins Debug representations of HIP kernel source types and verifies
//! that embedded kernel sources contain expected HIP entry points.

use bitnet_rocm::kernels::{HipKernelSource, kernel_source};

// ── HipKernelSource variants ─────────────────────────────────────────────────

#[test]
fn hip_kernel_source_all_variants_debug() {
    let variants = [
        HipKernelSource::Matmul,
        HipKernelSource::Softmax,
        HipKernelSource::RmsNorm,
        HipKernelSource::Rope,
        HipKernelSource::Attention,
        HipKernelSource::Elementwise,
    ];
    insta::assert_debug_snapshot!("hip_kernel_source_variants", variants);
}

// ── Kernel source content signatures ─────────────────────────────────────────

#[test]
fn matmul_kernel_contains_global_entry() {
    let src = kernel_source(HipKernelSource::Matmul);
    assert!(src.contains("__global__"), "matmul kernel must have a __global__ entry point");
    let first_line = src.lines().next().unwrap_or("");
    insta::assert_snapshot!("matmul_kernel_header", first_line);
}

#[test]
fn softmax_kernel_contains_global_entry() {
    let src = kernel_source(HipKernelSource::Softmax);
    assert!(src.contains("__global__"), "softmax kernel must have a __global__ entry point");
    let first_line = src.lines().next().unwrap_or("");
    insta::assert_snapshot!("softmax_kernel_header", first_line);
}

#[test]
fn rmsnorm_kernel_contains_global_entry() {
    let src = kernel_source(HipKernelSource::RmsNorm);
    assert!(src.contains("__global__"), "rmsnorm kernel must have a __global__ entry point");
    let first_line = src.lines().next().unwrap_or("");
    insta::assert_snapshot!("rmsnorm_kernel_header", first_line);
}

#[test]
fn rope_kernel_contains_global_entry() {
    let src = kernel_source(HipKernelSource::Rope);
    assert!(src.contains("__global__"), "rope kernel must have a __global__ entry point");
    let first_line = src.lines().next().unwrap_or("");
    insta::assert_snapshot!("rope_kernel_header", first_line);
}

#[test]
fn attention_kernel_contains_global_entry() {
    let src = kernel_source(HipKernelSource::Attention);
    assert!(src.contains("__global__"), "attention kernel must have a __global__ entry point");
    let first_line = src.lines().next().unwrap_or("");
    insta::assert_snapshot!("attention_kernel_header", first_line);
}

#[test]
fn elementwise_kernel_contains_global_entry() {
    let src = kernel_source(HipKernelSource::Elementwise);
    assert!(src.contains("__global__"), "elementwise kernel must have a __global__ entry point");
    let first_line = src.lines().next().unwrap_or("");
    insta::assert_snapshot!("elementwise_kernel_header", first_line);
}
