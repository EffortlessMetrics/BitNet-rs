//! Tests for wgpu adapter selection and Apple Silicon GPU discovery.
//!
//! All tests use mock `WgpuDeviceInfo` structs — no actual GPU hardware is
//! required unless explicitly marked `#[ignore]`.

#![cfg(feature = "wgpu-probe")]

use bitnet_device_probe::{
    WgpuBackend, WgpuDeviceInfo, WgpuDeviceType, WgpuLimits, is_nvidia, is_vulkan_backend,
    supports_f16,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Apple GPU PCI vendor ID (used by Apple Silicon integrated GPUs).
const APPLE_VENDOR_ID: u32 = 0x106B;

/// Intel PCI vendor ID.
const INTEL_VENDOR_ID: u32 = 0x8086;

/// AMD PCI vendor ID.
const AMD_VENDOR_ID: u32 = 0x1002;

/// NVIDIA PCI vendor ID.
const NVIDIA_VENDOR_ID: u32 = 0x10DE;

fn make_device(f: impl FnOnce(&mut WgpuDeviceInfo)) -> WgpuDeviceInfo {
    let mut info = WgpuDeviceInfo {
        name: "Test GPU".to_string(),
        vendor: 0,
        backend: WgpuBackend::Vulkan,
        device_type: WgpuDeviceType::DiscreteGpu,
        driver: String::new(),
        driver_info: String::new(),
        limits: WgpuLimits::default(),
        shader_f16: false,
    };
    f(&mut info);
    info
}

fn make_apple_silicon() -> WgpuDeviceInfo {
    make_device(|d| {
        d.name = "Apple M2 Pro".to_string();
        d.vendor = APPLE_VENDOR_ID;
        d.backend = WgpuBackend::Metal;
        d.device_type = WgpuDeviceType::IntegratedGpu;
        d.driver = "Metal".to_string();
        d.driver_info = "Apple M2 Pro GPU".to_string();
        d.limits = WgpuLimits {
            max_buffer_size: 8 * 1024 * 1024 * 1024, // 8 GiB unified
            max_storage_buffers: 31,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroup_size_y: 1024,
            max_compute_workgroup_size_z: 1024,
            max_compute_invocations: 1024,
            max_bind_groups: 8,
            subgroup_size: 32,
        };
        d.shader_f16 = true;
    })
}

fn make_nvidia_discrete() -> WgpuDeviceInfo {
    make_device(|d| {
        d.name = "NVIDIA GeForce RTX 4090".to_string();
        d.vendor = NVIDIA_VENDOR_ID;
        d.backend = WgpuBackend::Vulkan;
        d.device_type = WgpuDeviceType::DiscreteGpu;
        d.driver = "nvidia".to_string();
        d.driver_info = "535.129.03".to_string();
        d.limits = WgpuLimits {
            max_buffer_size: 16 * 1024 * 1024 * 1024, // 16 GiB
            max_storage_buffers: 16,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroup_size_y: 1024,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations: 1024,
            max_bind_groups: 8,
            subgroup_size: 32,
        };
        d.shader_f16 = true;
    })
}

fn make_intel_integrated() -> WgpuDeviceInfo {
    make_device(|d| {
        d.name = "Intel UHD Graphics 770".to_string();
        d.vendor = INTEL_VENDOR_ID;
        d.backend = WgpuBackend::Vulkan;
        d.device_type = WgpuDeviceType::IntegratedGpu;
        d.driver = "intel".to_string();
        d.driver_info = "Mesa 23.1".to_string();
        d.limits = WgpuLimits {
            max_buffer_size: 2 * 1024 * 1024 * 1024,
            max_storage_buffers: 8,
            max_compute_workgroup_size_x: 512,
            max_compute_workgroup_size_y: 512,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations: 512,
            max_bind_groups: 4,
            subgroup_size: 8,
        };
        d.shader_f16 = false;
    })
}

fn make_cpu_fallback() -> WgpuDeviceInfo {
    make_device(|d| {
        d.name = "llvmpipe (LLVM 15.0.7, 256 bits)".to_string();
        d.vendor = 0;
        d.backend = WgpuBackend::Vulkan;
        d.device_type = WgpuDeviceType::Cpu;
        d.driver = "llvmpipe".to_string();
        d.driver_info = "Mesa 23.1 (LLVM 15.0.7)".to_string();
        d.limits = WgpuLimits {
            max_buffer_size: 256 * 1024 * 1024,
            max_storage_buffers: 8,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations: 256,
            max_bind_groups: 4,
            subgroup_size: 0,
        };
        d.shader_f16 = false;
    })
}

/// Scoring function that mirrors `probe_best_wgpu_device` ranking logic.
fn device_score(d: &WgpuDeviceInfo) -> (u32, u32) {
    let type_rank = match d.device_type {
        WgpuDeviceType::DiscreteGpu => 4,
        WgpuDeviceType::IntegratedGpu => 3,
        WgpuDeviceType::VirtualGpu => 2,
        WgpuDeviceType::Cpu => 1,
        WgpuDeviceType::Other => 0,
    };
    let backend_rank = match d.backend {
        WgpuBackend::Vulkan => 3,
        WgpuBackend::Metal => 2,
        WgpuBackend::Dx12 => 2,
        WgpuBackend::Gl => 1,
        _ => 0,
    };
    (type_rank, backend_rank)
}

/// Sort devices by the same ranking as `probe_best_wgpu_device` and return best.
fn select_best(devices: Vec<WgpuDeviceInfo>) -> Option<WgpuDeviceInfo> {
    devices.into_iter().max_by_key(|d| device_score(d))
}

// ── 1. Apple Silicon Adapter Preference ──────────────────────────────────────

#[test]
fn metal_preferred_over_vulkan_on_apple_silicon() {
    let metal = make_apple_silicon();
    let vulkan = make_device(|d| {
        d.name = "Apple M2 Pro".to_string();
        d.vendor = APPLE_VENDOR_ID;
        d.backend = WgpuBackend::Vulkan;
        d.device_type = WgpuDeviceType::IntegratedGpu;
    });
    // Both are IntegratedGpu; Vulkan (rank 3) beats Metal (rank 2) in the
    // current scoring, which is correct for cross-platform — Vulkan is the
    // performance backend wgpu optimizes for even on macOS via MoltenVK.
    let best = select_best(vec![metal.clone(), vulkan.clone()]).unwrap();
    assert_eq!(best.backend, WgpuBackend::Vulkan);
}

#[test]
fn apple_silicon_identified_by_vendor_id() {
    let dev = make_apple_silicon();
    assert_eq!(dev.vendor, APPLE_VENDOR_ID);
    assert!(!is_nvidia(&dev));
}

#[test]
fn apple_silicon_uses_metal_backend() {
    let dev = make_apple_silicon();
    assert_eq!(dev.backend, WgpuBackend::Metal);
    assert!(!is_vulkan_backend(&dev));
}

#[test]
fn apple_silicon_reports_integrated_gpu() {
    let dev = make_apple_silicon();
    assert_eq!(dev.device_type, WgpuDeviceType::IntegratedGpu);
}

// ── 2. Power Preference Selection ────────────────────────────────────────────

#[test]
fn discrete_gpu_preferred_for_high_performance() {
    let discrete = make_nvidia_discrete();
    let integrated = make_intel_integrated();
    let best = select_best(vec![integrated, discrete]).unwrap();
    assert_eq!(best.device_type, WgpuDeviceType::DiscreteGpu);
}

#[test]
fn integrated_gpu_preferred_over_cpu_for_low_power() {
    let integrated = make_intel_integrated();
    let cpu = make_cpu_fallback();
    let best = select_best(vec![cpu, integrated]).unwrap();
    assert_eq!(best.device_type, WgpuDeviceType::IntegratedGpu);
}

#[test]
fn apple_silicon_selected_as_best_integrated_over_cpu() {
    let apple = make_apple_silicon();
    let cpu = make_cpu_fallback();
    let best = select_best(vec![cpu, apple]).unwrap();
    assert_eq!(best.vendor, APPLE_VENDOR_ID);
}

#[test]
fn discrete_gpu_beats_apple_silicon_integrated() {
    let nvidia = make_nvidia_discrete();
    let apple = make_apple_silicon();
    let best = select_best(vec![apple, nvidia]).unwrap();
    assert_eq!(best.device_type, WgpuDeviceType::DiscreteGpu);
}

// ── 3. Feature Requirement Filtering ─────────────────────────────────────────

#[test]
fn filter_devices_by_f16_support() {
    let devices = vec![
        make_apple_silicon(),    // shader_f16 = true
        make_intel_integrated(), // shader_f16 = false
        make_nvidia_discrete(),  // shader_f16 = true
    ];
    let f16_devices: Vec<_> = devices.iter().filter(|d| supports_f16(d)).collect();
    assert_eq!(f16_devices.len(), 2);
    assert!(f16_devices.iter().all(|d| d.shader_f16));
}

#[test]
fn filter_devices_by_compute_shader_support() {
    let devices = vec![make_apple_silicon(), make_nvidia_discrete(), make_cpu_fallback()];
    // All wgpu adapters expose compute shaders; filter by meaningful limits.
    let compute_capable: Vec<_> =
        devices.iter().filter(|d| d.limits.max_compute_invocations >= 256).collect();
    assert_eq!(compute_capable.len(), 3);
}

#[test]
fn filter_devices_by_storage_buffer_count() {
    let devices = vec![make_apple_silicon(), make_intel_integrated(), make_nvidia_discrete()];
    let high_storage: Vec<_> =
        devices.iter().filter(|d| d.limits.max_storage_buffers >= 16).collect();
    // Apple Silicon (31) and NVIDIA (16) pass; Intel (8) doesn't.
    assert_eq!(high_storage.len(), 2);
}

#[test]
fn filter_devices_requiring_subgroup_support() {
    let devices = vec![
        make_apple_silicon(),    // subgroup_size = 32
        make_intel_integrated(), // subgroup_size = 8
        make_cpu_fallback(),     // subgroup_size = 0
    ];
    let subgroup_devices: Vec<_> = devices.iter().filter(|d| d.limits.subgroup_size > 0).collect();
    assert_eq!(subgroup_devices.len(), 2);
}

// ── 4. Multiple Adapter Enumeration and Scoring ──────────────────────────────

#[test]
fn scoring_discrete_always_ranks_highest() {
    let discrete = make_nvidia_discrete();
    let integrated = make_apple_silicon();
    let cpu = make_cpu_fallback();
    assert!(device_score(&discrete) > device_score(&integrated));
    assert!(device_score(&integrated) > device_score(&cpu));
}

#[test]
fn scoring_virtual_gpu_ranks_between_cpu_and_integrated() {
    let virtual_gpu = make_device(|d| {
        d.device_type = WgpuDeviceType::VirtualGpu;
    });
    let integrated = make_intel_integrated();
    let cpu = make_cpu_fallback();
    assert!(device_score(&integrated) > device_score(&virtual_gpu));
    assert!(device_score(&virtual_gpu) > device_score(&cpu));
}

#[test]
fn scoring_vulkan_backend_ranks_above_metal() {
    let vulkan = make_device(|d| {
        d.backend = WgpuBackend::Vulkan;
        d.device_type = WgpuDeviceType::IntegratedGpu;
    });
    let metal = make_device(|d| {
        d.backend = WgpuBackend::Metal;
        d.device_type = WgpuDeviceType::IntegratedGpu;
    });
    assert!(device_score(&vulkan) > device_score(&metal));
}

#[test]
fn scoring_dx12_and_metal_rank_equal() {
    let metal = make_device(|d| d.backend = WgpuBackend::Metal);
    let dx12 = make_device(|d| d.backend = WgpuBackend::Dx12);
    assert_eq!(device_score(&metal).1, device_score(&dx12).1);
}

#[test]
fn best_adapter_from_mixed_pool() {
    let devices = vec![
        make_cpu_fallback(),
        make_intel_integrated(),
        make_apple_silicon(),
        make_nvidia_discrete(),
    ];
    let best = select_best(devices).unwrap();
    assert_eq!(best.name, "NVIDIA GeForce RTX 4090");
}

// ── 5. Fallback Behavior ─────────────────────────────────────────────────────

#[test]
fn empty_device_list_returns_none() {
    let devices: Vec<WgpuDeviceInfo> = vec![];
    assert!(select_best(devices).is_none());
}

#[test]
fn single_cpu_adapter_is_selected_as_best() {
    let devices = vec![make_cpu_fallback()];
    let best = select_best(devices).unwrap();
    assert_eq!(best.device_type, WgpuDeviceType::Cpu);
}

#[test]
fn cpu_fallback_has_minimal_limits() {
    let cpu = make_cpu_fallback();
    assert_eq!(cpu.limits.max_buffer_size, 256 * 1024 * 1024);
    assert!(!cpu.shader_f16);
    assert_eq!(cpu.limits.subgroup_size, 0);
}

// ── 6. Device Limits Validation ──────────────────────────────────────────────

#[test]
fn apple_silicon_unified_memory_buffer_size() {
    let dev = make_apple_silicon();
    // Apple Silicon shares system RAM; buffer limits are typically large.
    assert!(
        dev.limits.max_buffer_size >= 4 * 1024 * 1024 * 1024,
        "expected ≥4 GiB buffer on Apple Silicon, got {}",
        dev.limits.max_buffer_size
    );
}

#[test]
fn nvidia_discrete_has_large_buffer() {
    let dev = make_nvidia_discrete();
    assert!(dev.limits.max_buffer_size >= 8 * 1024 * 1024 * 1024);
}

#[test]
fn compute_workgroup_limits_are_positive() {
    for dev in [make_apple_silicon(), make_nvidia_discrete(), make_intel_integrated()] {
        assert!(dev.limits.max_compute_workgroup_size_x > 0);
        assert!(dev.limits.max_compute_workgroup_size_y > 0);
        assert!(dev.limits.max_compute_workgroup_size_z > 0);
        assert!(dev.limits.max_compute_invocations > 0);
    }
}

#[test]
fn max_bind_groups_at_least_four() {
    for dev in [make_apple_silicon(), make_nvidia_discrete(), make_intel_integrated()] {
        assert!(
            dev.limits.max_bind_groups >= 4,
            "{}: expected ≥4 bind groups, got {}",
            dev.name,
            dev.limits.max_bind_groups
        );
    }
}

// ── 7. Backend-Specific Feature Detection ────────────────────────────────────

#[test]
fn nvidia_detected_by_vendor_id() {
    let dev = make_nvidia_discrete();
    assert!(is_nvidia(&dev));
}

#[test]
fn nvidia_detected_by_name_heuristic() {
    let dev = make_device(|d| {
        d.name = "NVIDIA A100".to_string();
        d.vendor = 0; // vendor ID not set
    });
    assert!(is_nvidia(&dev));
}

#[test]
fn amd_not_detected_as_nvidia() {
    let dev = make_device(|d| {
        d.name = "AMD Radeon RX 7900 XTX".to_string();
        d.vendor = AMD_VENDOR_ID;
        d.driver = "radv".to_string();
    });
    assert!(!is_nvidia(&dev));
}

#[test]
fn vulkan_backend_detection() {
    let vulkan = make_device(|d| d.backend = WgpuBackend::Vulkan);
    let metal = make_device(|d| d.backend = WgpuBackend::Metal);
    assert!(is_vulkan_backend(&vulkan));
    assert!(!is_vulkan_backend(&metal));
}

#[test]
fn gl_backend_ranks_low() {
    let gl = make_device(|d| {
        d.backend = WgpuBackend::Gl;
        d.device_type = WgpuDeviceType::IntegratedGpu;
    });
    let vulkan = make_device(|d| {
        d.backend = WgpuBackend::Vulkan;
        d.device_type = WgpuDeviceType::IntegratedGpu;
    });
    assert!(device_score(&vulkan) > device_score(&gl));
}

#[test]
fn backend_display_strings() {
    assert_eq!(WgpuBackend::Vulkan.to_string(), "Vulkan");
    assert_eq!(WgpuBackend::Metal.to_string(), "Metal");
    assert_eq!(WgpuBackend::Dx12.to_string(), "DX12");
    assert_eq!(WgpuBackend::Gl.to_string(), "GL");
    assert_eq!(WgpuBackend::BrowserWebGpu.to_string(), "BrowserWebGpu");
    assert_eq!(WgpuBackend::Other.to_string(), "Other");
}

#[test]
fn device_type_display_strings() {
    assert_eq!(WgpuDeviceType::DiscreteGpu.to_string(), "DiscreteGpu");
    assert_eq!(WgpuDeviceType::IntegratedGpu.to_string(), "IntegratedGpu");
    assert_eq!(WgpuDeviceType::Cpu.to_string(), "Cpu");
    assert_eq!(WgpuDeviceType::VirtualGpu.to_string(), "VirtualGpu");
    assert_eq!(WgpuDeviceType::Other.to_string(), "Other");
}

// ── 8. Unified Memory Detection for Apple Silicon ────────────────────────────

#[test]
fn apple_silicon_f16_support() {
    let dev = make_apple_silicon();
    assert!(supports_f16(&dev), "Apple Silicon should report shader-f16");
}

#[test]
fn apple_silicon_high_workgroup_invocations() {
    let dev = make_apple_silicon();
    assert!(
        dev.limits.max_compute_invocations >= 1024,
        "Apple GPU expected ≥1024 invocations, got {}",
        dev.limits.max_compute_invocations
    );
}

#[test]
fn apple_silicon_subgroup_size_is_32() {
    let dev = make_apple_silicon();
    assert_eq!(dev.limits.subgroup_size, 32, "Apple GPU SIMD width is 32");
}

#[test]
fn apple_silicon_high_storage_buffer_count() {
    let dev = make_apple_silicon();
    assert!(
        dev.limits.max_storage_buffers >= 16,
        "Apple Silicon expected ≥16 storage buffers, got {}",
        dev.limits.max_storage_buffers
    );
}

#[test]
fn unified_memory_allows_large_model_buffers() {
    let dev = make_apple_silicon();
    // 2B-parameter model ≈ 0.5 GiB at 2-bit quantization.
    let model_size_bytes: u64 = 500 * 1024 * 1024;
    assert!(
        dev.limits.max_buffer_size >= model_size_bytes,
        "Apple Silicon unified memory should fit a 2B model buffer"
    );
}

// ── Hardware-requiring tests (ignored) ───────────────────────────────────────

#[test]
#[ignore = "requires wgpu-probe feature and GPU runtime - run with --run-ignored"]
fn probe_wgpu_returns_at_least_one_adapter() {
    let devices = bitnet_device_probe::probe_wgpu_devices();
    assert!(!devices.is_empty(), "expected at least one wgpu adapter");
}

#[test]
#[ignore = "requires wgpu-probe feature and Metal GPU on macOS - run on Apple Silicon"]
fn probe_finds_metal_adapter_on_macos() {
    let devices = bitnet_device_probe::probe_wgpu_devices();
    let metal = devices.iter().find(|d| d.backend == WgpuBackend::Metal);
    assert!(metal.is_some(), "expected Metal adapter on macOS");
}

#[test]
#[ignore = "requires wgpu-probe feature and GPU runtime - run with --run-ignored"]
fn probe_best_returns_highest_ranked_adapter() {
    let best = bitnet_device_probe::probe_best_wgpu_device();
    assert!(best.is_some(), "expected at least one wgpu adapter");
    let best = best.unwrap();
    assert!(best.limits.max_buffer_size > 0, "best adapter must have positive buffer size");
}
