//! Insta snapshot tests for the OpenCL kernel infrastructure.
//!
//! Snapshot tests guard against unintended changes in kernel source strings,
//! selector decisions, and error messages.

use bitnet_opencl::buffers;
use bitnet_opencl::kernels;
use bitnet_opencl::selector::KernelSelector;
use bitnet_opencl::workgroup::WorkgroupConfig;
use bitnet_opencl::{OpenClDeviceInfo, OpenClKernel};

// ── Kernel source snapshots ──────────────────────────────────────────

#[test]
fn snapshot_qk256_gemv_source() {
    insta::assert_snapshot!("qk256_gemv_source", kernels::QK256_GEMV_SOURCE);
}

#[test]
fn snapshot_rmsnorm_source() {
    insta::assert_snapshot!("rmsnorm_source", kernels::RMSNORM_SOURCE);
}

#[test]
fn snapshot_attention_source() {
    insta::assert_snapshot!("attention_source", kernels::ATTENTION_SOURCE);
}

#[test]
fn snapshot_all_kernel_names() {
    let names: Vec<&str> = kernels::all_kernel_sources().into_iter().map(|(n, _)| n).collect();
    insta::assert_yaml_snapshot!("kernel_names", names);
}

// ── KernelSelector decision snapshots ────────────────────────────────

fn default_device() -> OpenClDeviceInfo {
    OpenClDeviceInfo::default()
}

fn small_device() -> OpenClDeviceInfo {
    OpenClDeviceInfo {
        max_workgroup_size: 64,
        max_compute_units: 4,
        global_mem_bytes: 512 * 1024 * 1024,
        local_mem_bytes: 16 * 1024,
        preferred_workgroup_multiple: 16,
        ..Default::default()
    }
}

#[test]
fn snapshot_gemv_selector_decisions() {
    let device = default_device();
    let cases: Vec<(usize, usize, String)> = vec![
        (0, 0, "zero_zero"),
        (1, 256, "tiny"),
        (64, 256, "small"),
        (256, 256, "medium_square"),
        (2048, 2048, "large"),
        (16384, 16384, "very_large"),
    ]
    .into_iter()
    .map(|(n, k, label)| {
        let variant = KernelSelector::select_gemv(n, k, &device);
        (n, k, format!("{label}: {variant}"))
    })
    .collect();

    insta::assert_yaml_snapshot!("gemv_selector_decisions", cases);
}

#[test]
fn snapshot_gemv_selector_small_device() {
    let device = small_device();
    let cases: Vec<(usize, usize, String)> =
        vec![(64, 256, "small"), (2048, 2048, "large"), (16384, 16384, "very_large")]
            .into_iter()
            .map(|(n, k, label)| {
                let variant = KernelSelector::select_gemv(n, k, &device);
                (n, k, format!("{label}: {variant}"))
            })
            .collect();

    insta::assert_yaml_snapshot!("gemv_selector_small_device", cases);
}

#[test]
fn snapshot_rmsnorm_selector_decisions() {
    let device = default_device();
    let cases: Vec<(usize, String)> = vec![0, 32, 128, 256, 512, 2048, 4096]
        .into_iter()
        .map(|hidden| {
            let variant = KernelSelector::select_rmsnorm(hidden, &device);
            (hidden, format!("{variant}"))
        })
        .collect();

    insta::assert_yaml_snapshot!("rmsnorm_selector_decisions", cases);
}

#[test]
fn snapshot_attention_selector_decisions() {
    let device = default_device();
    let cases: Vec<(usize, usize, usize, String)> = vec![
        (0, 0, 0, "zero"),
        (1, 1, 64, "minimal"),
        (32, 32, 64, "small"),
        (512, 512, 128, "medium"),
        (2048, 2048, 128, "large"),
    ]
    .into_iter()
    .map(|(sq, skv, hd, label)| {
        let variant = KernelSelector::select_attention(sq, skv, hd, &device);
        (sq, skv, hd, format!("{label}: {variant}"))
    })
    .collect();

    insta::assert_yaml_snapshot!("attention_selector_decisions", cases);
}

// ── Workgroup configuration snapshots ────────────────────────────────

#[test]
fn snapshot_workgroup_configs() {
    let device = default_device();
    let cases: Vec<(usize, String)> = vec![1, 7, 32, 255, 256, 1000, 4096, 100_000]
        .into_iter()
        .map(|size| {
            let cfg = WorkgroupConfig::for_1d(size, &device).unwrap();
            let desc = format!(
                "local={} global={} groups={}",
                cfg.local_size, cfg.global_size, cfg.num_groups
            );
            (size, desc)
        })
        .collect();

    insta::assert_yaml_snapshot!("workgroup_configs_default_device", cases);
}

#[test]
fn snapshot_workgroup_configs_small_device() {
    let device = small_device();
    let cases: Vec<(usize, String)> = vec![1, 16, 64, 100, 1024]
        .into_iter()
        .map(|size| {
            let cfg = WorkgroupConfig::for_1d(size, &device).unwrap();
            let desc = format!(
                "local={} global={} groups={}",
                cfg.local_size, cfg.global_size, cfg.num_groups
            );
            (size, desc)
        })
        .collect();

    insta::assert_yaml_snapshot!("workgroup_configs_small_device", cases);
}

// ── Error message snapshots ──────────────────────────────────────────

#[test]
fn snapshot_error_matmul_unavailable() {
    let kernel = OpenClKernel::new();
    let err = kernel.matmul_i2s(&[1i8], &[1u8], &mut [0.0f32], 1, 1, 1).unwrap_err();
    insta::assert_snapshot!("error_matmul_unavailable", format!("{err}"));
}

#[test]
fn snapshot_error_quantize_unavailable() {
    let kernel = OpenClKernel::new();
    let err = kernel
        .quantize(&[1.0f32], &mut [0u8], &mut [0.0f32], bitnet_common::QuantizationType::I2S)
        .unwrap_err();
    insta::assert_snapshot!("error_quantize_unavailable", format!("{err}"));
}

#[test]
fn snapshot_error_zero_workgroup() {
    let device = default_device();
    let err = WorkgroupConfig::for_1d(0, &device).unwrap_err();
    insta::assert_snapshot!("error_zero_workgroup", format!("{err}"));
}

#[test]
fn snapshot_error_qk256_bad_k() {
    let err = buffers::qk256_weight_bytes(1, 100).unwrap_err();
    insta::assert_snapshot!("error_qk256_bad_k", format!("{err}"));
}

#[test]
fn snapshot_error_qk256_zero_k() {
    let err = buffers::qk256_weight_bytes(1, 0).unwrap_err();
    insta::assert_snapshot!("error_qk256_zero_k", format!("{err}"));
}

// ── Buffer size snapshots ────────────────────────────────────────────

#[test]
fn snapshot_qk256_buffer_sizes() {
    let cases: Vec<(usize, usize, usize, usize)> =
        vec![(1, 256), (1, 512), (2048, 2048), (4096, 4096)]
            .into_iter()
            .map(|(n, k)| {
                let weight = buffers::qk256_weight_bytes(n, k).unwrap();
                let scale = buffers::qk256_scale_bytes(n, k).unwrap();
                (n, k, weight, scale)
            })
            .collect();

    insta::assert_yaml_snapshot!("qk256_buffer_sizes", cases);
}
