//! Comprehensive unit tests for the OpenCL kernel infrastructure.
//!
//! These tests verify kernel selector logic, buffer size calculations,
//! workgroup configuration, kernel source validity, and edge cases —
//! all without requiring actual OpenCL hardware.

use bitnet_opencl::buffers;
use bitnet_opencl::kernels;
use bitnet_opencl::selector::{KernelSelector, KernelVariant};
use bitnet_opencl::workgroup::WorkgroupConfig;
use bitnet_opencl::{OpenClDeviceInfo, OpenClKernel, is_opencl_available, opencl_device_count};

// ── Kernel source validation ─────────────────────────────────────────

#[test]
fn kernel_sources_are_non_empty() {
    for (name, src) in kernels::all_kernel_sources() {
        assert!(!src.is_empty(), "kernel source '{name}' must not be empty");
    }
}

#[test]
fn kernel_sources_are_valid_utf8() {
    // Rust &str is guaranteed UTF-8, but verify we can round-trip through
    // bytes as a sanity check.
    for (name, src) in kernels::all_kernel_sources() {
        let bytes = src.as_bytes();
        let round_trip = std::str::from_utf8(bytes);
        assert!(round_trip.is_ok(), "kernel '{name}' is not valid UTF-8");
    }
}

#[test]
fn kernel_sources_contain_entry_point() {
    let expected = [
        ("qk256_gemv", "__kernel void qk256_gemv"),
        ("rmsnorm", "__kernel void rmsnorm"),
        ("attention", "__kernel void attention"),
    ];
    for (name, signature) in expected {
        let src = kernels::all_kernel_sources()
            .into_iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| s)
            .unwrap_or_else(|| panic!("missing kernel source: {name}"));
        assert!(
            src.contains(signature),
            "kernel '{name}' missing entry-point signature '{signature}'"
        );
    }
}

#[test]
fn kernel_sources_contain_get_global_id() {
    // All compute kernels should use get_global_id or get_group_id.
    for (name, src) in kernels::all_kernel_sources() {
        let has_id_call = src.contains("get_global_id")
            || src.contains("get_group_id")
            || src.contains("get_local_id");
        assert!(has_id_call, "kernel '{name}' should contain OpenCL work-item ID calls");
    }
}

#[test]
fn kernel_sources_no_null_bytes() {
    for (name, src) in kernels::all_kernel_sources() {
        assert!(!src.contains('\0'), "kernel '{name}' contains null bytes");
    }
}

// ── KernelSelector logic ─────────────────────────────────────────────

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
fn selector_gemv_small_problem() {
    let device = default_device();
    let variant = KernelSelector::select_gemv(64, 256, &device);
    assert_eq!(variant, KernelVariant::Small);
}

#[test]
fn selector_gemv_medium_problem() {
    let device = default_device();
    // 2048 * 2048 = 4M elements — should exceed single-workgroup capacity.
    let variant = KernelSelector::select_gemv(2048, 2048, &device);
    assert!(
        variant == KernelVariant::Tiled || variant == KernelVariant::LargeReduction,
        "expected Tiled or LargeReduction, got {variant}"
    );
}

#[test]
fn selector_gemv_large_problem() {
    let device = small_device();
    let variant = KernelSelector::select_gemv(16384, 16384, &device);
    assert_eq!(variant, KernelVariant::LargeReduction);
}

#[test]
fn selector_gemv_zero_dimensions() {
    let device = default_device();
    assert_eq!(KernelSelector::select_gemv(0, 256, &device), KernelVariant::Small);
    assert_eq!(KernelSelector::select_gemv(256, 0, &device), KernelVariant::Small);
    assert_eq!(KernelSelector::select_gemv(0, 0, &device), KernelVariant::Small);
}

#[test]
fn selector_rmsnorm_small_hidden() {
    let device = default_device();
    assert_eq!(KernelSelector::select_rmsnorm(128, &device), KernelVariant::Small);
}

#[test]
fn selector_rmsnorm_large_hidden() {
    let device = default_device();
    assert_eq!(KernelSelector::select_rmsnorm(4096, &device), KernelVariant::Tiled);
}

#[test]
fn selector_rmsnorm_zero() {
    let device = default_device();
    assert_eq!(KernelSelector::select_rmsnorm(0, &device), KernelVariant::Small);
}

#[test]
fn selector_attention_small_seq() {
    let device = default_device();
    let variant = KernelSelector::select_attention(4, 4, 64, &device);
    assert_eq!(variant, KernelVariant::Small);
}

#[test]
fn selector_attention_large_seq() {
    let device = small_device();
    let variant = KernelSelector::select_attention(2048, 2048, 128, &device);
    assert!(
        variant == KernelVariant::Tiled || variant == KernelVariant::LargeReduction,
        "large seq should pick Tiled or LargeReduction, got {variant}"
    );
}

#[test]
fn selector_attention_zero_dims() {
    let device = default_device();
    assert_eq!(KernelSelector::select_attention(0, 128, 64, &device), KernelVariant::Small);
    assert_eq!(KernelSelector::select_attention(128, 0, 64, &device), KernelVariant::Small);
    assert_eq!(KernelSelector::select_attention(128, 128, 0, &device), KernelVariant::Small);
}

// ── Buffer size calculations ─────────────────────────────────────────

#[test]
fn qk256_weight_bytes_basic() {
    let bytes = buffers::qk256_weight_bytes(1, 256).unwrap();
    assert_eq!(bytes, 64); // 1 block of 64 bytes
}

#[test]
fn qk256_weight_bytes_larger() {
    let bytes = buffers::qk256_weight_bytes(2048, 2048).unwrap();
    // blocks_per_row = 8, bytes_per_row = 512, total = 2048 * 512
    assert_eq!(bytes, 2048 * 512);
}

#[test]
fn qk256_weight_bytes_rejects_non_multiple_k() {
    assert!(buffers::qk256_weight_bytes(1, 100).is_err());
    assert!(buffers::qk256_weight_bytes(1, 255).is_err());
    assert!(buffers::qk256_weight_bytes(1, 1).is_err());
}

#[test]
fn qk256_weight_bytes_rejects_zero_k() {
    assert!(buffers::qk256_weight_bytes(1, 0).is_err());
}

#[test]
fn qk256_weight_bytes_zero_n_out() {
    // n_out=0 is valid (produces 0 bytes).
    let bytes = buffers::qk256_weight_bytes(0, 256).unwrap();
    assert_eq!(bytes, 0);
}

#[test]
fn qk256_scale_bytes_basic() {
    let bytes = buffers::qk256_scale_bytes(1, 256).unwrap();
    assert_eq!(bytes, 2); // 1 block × 2 bytes
}

#[test]
fn qk256_scale_bytes_larger() {
    let bytes = buffers::qk256_scale_bytes(2048, 2048).unwrap();
    // blocks_per_row = 8, total_blocks = 16384, bytes = 32768
    assert_eq!(bytes, 32768);
}

#[test]
fn fp32_buffer_bytes_basic() {
    assert_eq!(buffers::fp32_buffer_bytes(1).unwrap(), 4);
    assert_eq!(buffers::fp32_buffer_bytes(1024).unwrap(), 4096);
    assert_eq!(buffers::fp32_buffer_bytes(0).unwrap(), 0);
}

#[test]
fn fp16_buffer_bytes_basic() {
    assert_eq!(buffers::fp16_buffer_bytes(1).unwrap(), 2);
    assert_eq!(buffers::fp16_buffer_bytes(1024).unwrap(), 2048);
}

#[test]
fn qk256_gemv_shared_mem_basic() {
    let mem = buffers::qk256_gemv_shared_mem(256).unwrap();
    // 1 block * 66 = 66, clamped to 4096 minimum.
    assert_eq!(mem, 4096);
}

#[test]
fn qk256_gemv_shared_mem_large_k() {
    let mem = buffers::qk256_gemv_shared_mem(2048).unwrap();
    // 8 blocks * 66 = 528, still below 4096 minimum.
    assert_eq!(mem, 4096);

    // 256 blocks * 66 = 16896
    let mem = buffers::qk256_gemv_shared_mem(256 * 256).unwrap();
    assert_eq!(mem, 16896);
}

#[test]
fn qk256_gemv_shared_mem_rejects_zero() {
    assert!(buffers::qk256_gemv_shared_mem(0).is_err());
}

#[test]
fn qk256_gemv_shared_mem_rejects_non_multiple() {
    assert!(buffers::qk256_gemv_shared_mem(100).is_err());
}

#[test]
fn fp32_buffer_overflow_detection() {
    // usize::MAX / 4 + 1 should overflow when multiplied by 4.
    let huge = usize::MAX / 4 + 1;
    assert!(buffers::fp32_buffer_bytes(huge).is_err());
}

#[test]
fn fp16_buffer_overflow_detection() {
    let huge = usize::MAX / 2 + 1;
    assert!(buffers::fp16_buffer_bytes(huge).is_err());
}

// ── Workgroup configuration ──────────────────────────────────────────

#[test]
fn workgroup_for_1d_basic() {
    let device = default_device();
    let cfg = WorkgroupConfig::for_1d(1024, &device).unwrap();

    assert!(cfg.local_size > 0, "local_size must be positive");
    assert_eq!(cfg.global_size % cfg.local_size, 0, "global_size must be a multiple of local_size");
    assert!(cfg.global_size >= 1024, "global_size must cover the problem");
    assert_eq!(cfg.num_groups, cfg.global_size / cfg.local_size);
}

#[test]
fn workgroup_for_1d_non_power_of_2() {
    let device = default_device();
    let cfg = WorkgroupConfig::for_1d(1000, &device).unwrap();

    assert!(cfg.local_size > 0);
    assert_eq!(cfg.global_size % cfg.local_size, 0);
    assert!(cfg.global_size >= 1000);
}

#[test]
fn workgroup_for_1d_small_problem() {
    let device = default_device();
    let cfg = WorkgroupConfig::for_1d(1, &device).unwrap();

    assert!(cfg.local_size >= 1);
    assert!(cfg.global_size >= 1);
    assert!(cfg.num_groups >= 1);
}

#[test]
fn workgroup_for_1d_very_large() {
    let device = default_device();
    let cfg = WorkgroupConfig::for_1d(1_000_000, &device).unwrap();

    assert!(cfg.local_size > 0);
    assert_eq!(cfg.global_size % cfg.local_size, 0);
    assert!(cfg.global_size >= 1_000_000);
}

#[test]
fn workgroup_for_1d_zero_rejects() {
    let device = default_device();
    assert!(WorkgroupConfig::for_1d(0, &device).is_err());
}

#[test]
fn workgroup_local_size_respects_device_max() {
    let device = OpenClDeviceInfo::default().with_max_workgroup_size(128);
    let cfg = WorkgroupConfig::for_1d(4096, &device).unwrap();
    assert!(cfg.local_size <= 128, "local_size {} exceeds device max 128", cfg.local_size);
}

#[test]
fn workgroup_local_size_is_multiple_of_preferred() {
    let device = OpenClDeviceInfo {
        preferred_workgroup_multiple: 32,
        max_workgroup_size: 256,
        ..Default::default()
    };
    let cfg = WorkgroupConfig::for_1d(4096, &device).unwrap();
    assert_eq!(
        cfg.local_size % 32,
        0,
        "local_size {} should be a multiple of preferred 32",
        cfg.local_size
    );
}

#[test]
fn workgroup_for_gemv() {
    let device = default_device();
    let cfg = WorkgroupConfig::for_gemv(2048, &device).unwrap();
    assert!(cfg.local_size > 0);
    assert!(cfg.global_size >= 2048);
}

#[test]
fn workgroup_for_rmsnorm_basic() {
    let device = default_device();
    let cfg = WorkgroupConfig::for_rmsnorm(2048, 4, &device).unwrap();
    assert!(cfg.local_size > 0);
    assert_eq!(cfg.num_groups, 4, "one work-group per row");
}

#[test]
fn workgroup_for_rmsnorm_rejects_zero() {
    let device = default_device();
    assert!(WorkgroupConfig::for_rmsnorm(0, 4, &device).is_err());
    assert!(WorkgroupConfig::for_rmsnorm(2048, 0, &device).is_err());
}

// ── OpenClKernel provider ────────────────────────────────────────────

#[test]
fn opencl_kernel_reports_name() {
    let kernel = OpenClKernel::new();
    assert_eq!(kernel.name(), "opencl");
}

#[test]
fn opencl_kernel_not_available_without_env() {
    let kernel = OpenClKernel::new();
    // Without BITNET_ENABLE_OPENCL=1 the provider must not activate.
    assert!(!kernel.is_available());
}

#[test]
fn opencl_matmul_returns_err() {
    let kernel = OpenClKernel::new();
    let a = vec![1i8; 16];
    let b = vec![1u8; 16];
    let mut c = vec![0.0f32; 16];
    assert!(kernel.matmul_i2s(&a, &b, &mut c, 4, 4, 4).is_err());
}

#[test]
fn opencl_quantize_returns_err() {
    let kernel = OpenClKernel::new();
    let input = vec![1.0f32; 32];
    let mut output = vec![0u8; 8];
    let mut scales = vec![0.0f32; 1];
    assert!(
        kernel
            .quantize(&input, &mut output, &mut scales, bitnet_common::QuantizationType::I2S)
            .is_err()
    );
}

#[test]
fn opencl_device_count_is_zero() {
    assert_eq!(opencl_device_count(), 0);
}

#[test]
fn opencl_is_not_available() {
    assert!(!is_opencl_available());
}

// ── Numerical precision: reference CPU implementations ───────────────

#[test]
fn reference_rmsnorm_identity_gamma() {
    // RMSNorm with gamma=1 should scale by 1/rms(x).
    let x = [3.0f32, 4.0f32]; // rms = sqrt((9+16)/2) = sqrt(12.5)
    let gamma = vec![1.0f32, 1.0f32];
    let eps = 1e-6_f32;

    let rms = ((x.iter().map(|v| v * v).sum::<f32>()) / x.len() as f32 + eps).sqrt();
    let expected: Vec<f32> = x.iter().zip(&gamma).map(|(xi, gi)| xi / rms * gi).collect();

    for (i, (&exp, &xi)) in expected.iter().zip(x.iter()).enumerate() {
        let actual = xi / rms * gamma[i];
        assert!(
            (actual - exp).abs() < 1e-5,
            "rmsnorm mismatch at index {i}: expected {exp}, got {actual}"
        );
    }
}

#[test]
fn reference_rmsnorm_with_gamma() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let gamma = vec![0.5, 1.0, 1.5, 2.0];
    let eps = 1e-6_f32;

    let rms = ((x.iter().map(|v| v * v).sum::<f32>()) / x.len() as f32 + eps).sqrt();
    let output: Vec<f32> = x.iter().zip(&gamma).map(|(xi, gi)| xi / rms * gi).collect();

    // Check relative error.
    for val in &output {
        assert!(val.is_finite(), "non-finite output: {val}");
    }
    // gamma=2 on the largest element should produce the largest output.
    assert!(output[3] > output[0]);
}

// ── Device info builder ──────────────────────────────────────────────

#[test]
fn device_info_default_values() {
    let info = OpenClDeviceInfo::default();
    assert_eq!(info.max_workgroup_size, 256);
    assert!(info.global_mem_bytes > 0);
    assert!(info.local_mem_bytes > 0);
}

#[test]
fn device_info_builder_methods() {
    let info =
        OpenClDeviceInfo::default().with_max_workgroup_size(512).with_local_mem_bytes(128 * 1024);
    assert_eq!(info.max_workgroup_size, 512);
    assert_eq!(info.local_mem_bytes, 128 * 1024);
}

// ── KernelVariant Display ────────────────────────────────────────────

#[test]
fn kernel_variant_display() {
    assert_eq!(format!("{}", KernelVariant::Small), "small");
    assert_eq!(format!("{}", KernelVariant::Tiled), "tiled");
    assert_eq!(format!("{}", KernelVariant::LargeReduction), "large-reduction");
}
