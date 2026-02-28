//! Metal backend tests for Apple Silicon GPU support.
//!
//! Validates Metal device detection, capability probing, buffer alignment,
//! workgroup limits, memory estimation, shader availability, and CPU parity.
//! All tests are gated on `aarch64` + `macos` where appropriate.

use bitnet_common::Device;
use bitnet_kernels::capability_matrix::{
    DeviceClass, DeviceProfile, OperationCategory, PrecisionSupport, SupportLevel,
};
use bitnet_kernels::gpu_utils::{GpuInfo, get_gpu_info};

// ── Constants for Metal hardware constraints ────────────────────────────────

/// Metal requires 256-byte buffer alignment on Apple GPUs.
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Maximum threads per threadgroup on Apple Silicon (M1/M2/M3).
const METAL_MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// Minimum unified memory on any Apple Silicon Mac (8 GB).
const METAL_MIN_UNIFIED_MEMORY_GB: u64 = 8;

// ═══════════════════════════════════════════════════════════════════════════
// 1. Metal device detection and capability probing
// ═══════════════════════════════════════════════════════════════════════════

mod metal_device_detection {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn device_metal_variant_exists() {
        let device = Device::Metal;
        assert_eq!(device, Device::Metal);
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn device_metal_is_not_cpu() {
        let device = Device::Metal;
        assert_ne!(device, Device::Cpu);
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn device_metal_is_not_cuda() {
        let device = Device::Metal;
        assert_ne!(device, Device::Cuda(0));
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn device_metal_debug_format_contains_metal() {
        let device = Device::Metal;
        let debug = format!("{device:?}");
        assert!(debug.contains("Metal"), "Debug output should contain 'Metal', got: {debug}");
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn device_metal_serialization_roundtrip() {
        let device = Device::Metal;
        let json = serde_json::to_string(&device).unwrap();
        let deserialized: Device = serde_json::from_str(&json).unwrap();
        assert_eq!(device, deserialized);
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn device_metal_ordering_is_deterministic() {
        let d1 = Device::Metal;
        let d2 = Device::Metal;
        assert_eq!(d1.cmp(&d2), std::cmp::Ordering::Equal);
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[ignore = "requires Metal GPU runtime — run on macOS Apple Silicon"]
    fn gpu_info_detects_metal_on_apple_silicon() {
        let info = get_gpu_info();
        assert!(info.metal, "Metal should be detected on Apple Silicon macOS");
        assert!(info.any_available(), "At least one GPU backend should be available");
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[ignore = "requires Metal GPU runtime — run on macOS Apple Silicon"]
    fn gpu_info_summary_includes_metal() {
        let info = get_gpu_info();
        if info.metal {
            let summary = info.summary();
            assert!(
                summary.contains("Metal"),
                "Summary should mention Metal when detected: {summary}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Metal feature gate compilation
// ═══════════════════════════════════════════════════════════════════════════

mod metal_feature_gate {
    use super::*;

    #[test]
    fn gpu_info_struct_has_metal_field() {
        // If this compiles, the `metal` field exists on GpuInfo.
        let info = GpuInfo {
            cuda: false,
            cuda_version: None,
            metal: false,
            rocm: false,
            rocm_version: None,
            opengl: false,
            wgpu: false,
        };
        let _ = info.metal;
    }

    #[test]
    fn gpu_info_metal_false_does_not_imply_any_available() {
        let info = GpuInfo {
            cuda: false,
            cuda_version: None,
            metal: false,
            rocm: false,
            rocm_version: None,
            opengl: false,
            wgpu: false,
        };
        assert!(!info.any_available());
    }

    #[test]
    fn gpu_info_metal_true_implies_any_available() {
        let info = GpuInfo {
            cuda: false,
            cuda_version: None,
            metal: true,
            rocm: false,
            rocm_version: None,
            opengl: false,
            wgpu: false,
        };
        assert!(info.any_available());
    }

    #[test]
    fn device_class_apple_metal_exists() {
        let dc = DeviceClass::AppleMetal;
        assert_eq!(dc.to_string(), "Apple Metal");
    }

    #[test]
    fn device_class_all_contains_apple_metal() {
        assert!(
            DeviceClass::ALL.contains(&DeviceClass::AppleMetal),
            "DeviceClass::ALL must include AppleMetal"
        );
    }

    #[test]
    fn device_metal_variant_compiles_on_all_targets() {
        // Verifies that Device::Metal is available regardless of platform.
        let _d = Device::Metal;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Metal buffer alignment requirements
// ═══════════════════════════════════════════════════════════════════════════

mod metal_buffer_alignment {
    use super::*;

    #[test]
    fn alignment_constant_is_256() {
        assert_eq!(METAL_BUFFER_ALIGNMENT, 256);
    }

    #[test]
    fn alignment_is_power_of_two() {
        assert!(METAL_BUFFER_ALIGNMENT.is_power_of_two(), "Metal alignment must be a power of two");
    }

    /// Verify that a buffer size rounded up to Metal alignment is correct.
    #[test]
    fn align_up_small_sizes() {
        let cases: &[(usize, usize)] =
            &[(0, 0), (1, 256), (255, 256), (256, 256), (257, 512), (512, 512), (1000, 1024)];
        for &(input, expected) in cases {
            let aligned = metal_align_up(input);
            assert_eq!(aligned, expected, "align_up({input}) should be {expected}, got {aligned}");
        }
    }

    #[test]
    fn aligned_buffers_are_always_multiples_of_256() {
        for size in [1_usize, 64, 128, 255, 256, 300, 1024, 4096, 65535] {
            let aligned = metal_align_up(size);
            assert!(
                aligned % METAL_BUFFER_ALIGNMENT == 0 || size == 0,
                "Aligned size {aligned} (from {size}) not a multiple of 256"
            );
            assert!(aligned >= size, "Aligned size {aligned} must be >= input {size}");
        }
    }

    /// Helper: round `size` up to the next multiple of `METAL_BUFFER_ALIGNMENT`.
    fn metal_align_up(size: usize) -> usize {
        if size == 0 {
            return 0;
        }
        let mask = METAL_BUFFER_ALIGNMENT - 1;
        (size + mask) & !mask
    }
}

#[cfg(test)]
mod metal_buffer_alignment_proptest {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn aligned_size_gte_original(size in 0_usize..10_000_000) {
            let aligned = metal_align_up(size);
            prop_assert!(aligned >= size);
        }

        #[test]
        fn aligned_size_is_multiple_of_256(size in 1_usize..10_000_000) {
            let aligned = metal_align_up(size);
            prop_assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
        }

        #[test]
        fn aligned_size_overhead_less_than_alignment(
            size in 1_usize..10_000_000
        ) {
            let aligned = metal_align_up(size);
            let overhead = aligned - size;
            prop_assert!(overhead < METAL_BUFFER_ALIGNMENT);
        }
    }

    fn metal_align_up(size: usize) -> usize {
        if size == 0 {
            return 0;
        }
        let mask = METAL_BUFFER_ALIGNMENT - 1;
        (size + mask) & !mask
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Metal workgroup size limits
// ═══════════════════════════════════════════════════════════════════════════

mod metal_workgroup_limits {
    use super::*;

    #[test]
    fn max_threads_per_threadgroup_is_1024() {
        assert_eq!(METAL_MAX_THREADS_PER_THREADGROUP, 1024);
    }

    #[test]
    fn max_threads_is_power_of_two() {
        assert!(METAL_MAX_THREADS_PER_THREADGROUP.is_power_of_two());
    }

    /// Common 1-D dispatch sizes must not exceed the threadgroup limit.
    #[test]
    fn common_dispatch_sizes_within_limit() {
        let common_sizes: &[u32] = &[32, 64, 128, 256, 512, 1024];
        for &size in common_sizes {
            assert!(
                size <= METAL_MAX_THREADS_PER_THREADGROUP,
                "Dispatch size {size} exceeds Metal threadgroup limit"
            );
        }
    }

    /// 2-D threadgroup dimensions (width × height) must stay within limits.
    #[test]
    fn two_d_threadgroup_within_limit() {
        let grids: &[(u32, u32)] = &[
            (32, 32),  // 1024
            (16, 16),  // 256
            (8, 128),  // 1024
            (1, 1024), // 1024
            (1024, 1), // 1024
            (64, 16),  // 1024
        ];
        for &(w, h) in grids {
            let total = w * h;
            assert!(
                total <= METAL_MAX_THREADS_PER_THREADGROUP,
                "Threadgroup {w}×{h} = {total} exceeds limit"
            );
        }
    }

    /// 3-D threadgroup dimensions must respect the 1024-thread cap.
    #[test]
    fn three_d_threadgroup_within_limit() {
        let grids: &[(u32, u32, u32)] = &[
            (8, 8, 16), // 1024
            (4, 4, 64), // 1024
            (16, 8, 8), // 1024
            (1, 1, 1024),
            (32, 32, 1), // 1024
        ];
        for &(x, y, z) in grids {
            let total = x * y * z;
            assert!(
                total <= METAL_MAX_THREADS_PER_THREADGROUP,
                "Threadgroup {x}×{y}×{z} = {total} exceeds limit"
            );
        }
    }

    /// Verify that exceeding the limit is correctly detected.
    #[test]
    fn oversized_threadgroup_detected() {
        let total = 33_u32 * 33;
        assert!(
            total > METAL_MAX_THREADS_PER_THREADGROUP,
            "33×33 = {total} should exceed the limit"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Metal device memory estimation
// ═══════════════════════════════════════════════════════════════════════════

mod metal_memory_estimation {
    use super::*;

    #[test]
    fn minimum_unified_memory_is_8gb() {
        assert_eq!(METAL_MIN_UNIFIED_MEMORY_GB, 8);
    }

    /// On macOS aarch64 the system should report at least 8 GB total RAM
    /// (since Apple Silicon Macs have unified memory shared with the GPU).
    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn system_reports_at_least_min_memory() {
        use sysinfo::System;
        let sys = System::new_all();
        let total_gb = sys.total_memory() / (1024 * 1024 * 1024);
        assert!(
            total_gb >= METAL_MIN_UNIFIED_MEMORY_GB,
            "Expected >= {METAL_MIN_UNIFIED_MEMORY_GB} GB, got {total_gb} GB"
        );
    }

    /// Estimate whether a model fits in Metal unified memory.
    #[test]
    fn model_memory_estimation_2b_params() {
        // 2 billion params × 2 bits / 8 = 500 MB (rough I2_S estimate)
        let param_count: u64 = 2_000_000_000;
        let bytes = param_count * 2 / 8; // 2-bit quantisation
        let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!(
            gb < METAL_MIN_UNIFIED_MEMORY_GB as f64,
            "2B-param I2_S model ({gb:.2} GB) should fit in min Metal memory"
        );
    }

    #[test]
    fn apple_m1_profile_memory_is_sane() {
        let profile = DeviceProfile::apple_m1();
        assert_eq!(profile.device_class, DeviceClass::AppleMetal);
        assert!(
            profile.memory_gb as u64 >= METAL_MIN_UNIFIED_MEMORY_GB,
            "Apple M1 profile memory {} GB should be >= {METAL_MIN_UNIFIED_MEMORY_GB}",
            profile.memory_gb
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Metal shader compilation availability
// ═══════════════════════════════════════════════════════════════════════════

mod metal_shader_availability {
    use super::*;

    /// Verify the Apple M1 capability profile includes matrix ops.
    #[test]
    fn apple_m1_supports_matmul_fp32() {
        let profile = DeviceProfile::apple_m1();
        let entry = profile.capabilities.iter().find(|e| {
            e.operation == OperationCategory::MatrixOps && e.precision == PrecisionSupport::FP32
        });
        assert!(entry.is_some(), "Apple M1 should have MatrixOps/FP32 capability");
        if let Some(e) = entry {
            assert!(
                matches!(e.support, SupportLevel::Full(_)),
                "MatrixOps/FP32 should be Full support"
            );
        }
    }

    #[test]
    fn apple_m1_supports_matmul_fp16() {
        let profile = DeviceProfile::apple_m1();
        let entry = profile.capabilities.iter().find(|e| {
            e.operation == OperationCategory::MatrixOps && e.precision == PrecisionSupport::FP16
        });
        assert!(entry.is_some(), "Apple M1 should have native FP16 MatrixOps");
    }

    #[test]
    fn apple_m1_binary_quantized_is_partial() {
        let profile = DeviceProfile::apple_m1();
        let entry = profile.capabilities.iter().find(|e| {
            e.operation == OperationCategory::QuantizedOps
                && e.precision == PrecisionSupport::Binary
        });
        assert!(entry.is_some(), "Apple M1 should list Binary/QuantizedOps");
        if let Some(e) = entry {
            assert!(
                matches!(e.support, SupportLevel::Partial(_)),
                "Binary QuantizedOps should be Partial (requires Metal compute shader)"
            );
        }
    }

    /// Metal shader compiler (`xcrun -sdk macosx metal`) should be present
    /// on a macOS developer system.
    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[ignore = "requires Xcode or Command Line Tools installed"]
    fn metal_shader_compiler_available() {
        let status = std::process::Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "--version"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        assert!(
            status.is_ok_and(|s| s.success()),
            "Metal shader compiler (xcrun metal) should be available"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Metal vs CPU parity checks (stubs)
// ═══════════════════════════════════════════════════════════════════════════

mod metal_cpu_parity {
    use super::*;

    #[test]
    #[ignore = "TDD scaffold: Metal matmul kernel not yet implemented"]
    fn metal_matmul_matches_cpu_small() {
        // Stub: when Metal matmul is implemented, this should verify that
        // a small 4×4 matmul on Metal produces the same output as the CPU
        // FallbackKernel within f32 epsilon tolerance.
        unimplemented!("Metal matmul kernel parity test");
    }

    #[test]
    #[ignore = "TDD scaffold: Metal quantize kernel not yet implemented"]
    fn metal_quantize_matches_cpu() {
        // Stub: verify Metal I2_S quantisation produces identical packed
        // output and scales as the CPU reference.
        unimplemented!("Metal quantize kernel parity test");
    }

    #[test]
    #[ignore = "TDD scaffold: Metal norm kernel not yet implemented"]
    fn metal_layernorm_matches_cpu() {
        // Stub: verify Metal LayerNorm matches CPU within tolerance.
        unimplemented!("Metal LayerNorm kernel parity test");
    }

    #[test]
    #[ignore = "TDD scaffold: Metal activation kernel not yet implemented"]
    fn metal_silu_activation_matches_cpu() {
        // Stub: verify Metal SiLU activation matches CPU within tolerance.
        unimplemented!("Metal SiLU activation kernel parity test");
    }

    /// CPU fallback kernel is always available — baseline for future parity.
    #[test]
    fn cpu_fallback_available_as_parity_baseline() {
        use bitnet_kernels::{FallbackKernel, KernelProvider};
        let cpu = FallbackKernel;
        assert!(cpu.is_available(), "CPU fallback must always be available");
        assert_eq!(cpu.name(), "cpu-fallback");
    }

    /// Cosine similarity helper used for future parity checks.
    #[test]
    fn cosine_similarity_identical_vectors() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical vectors should have cosine similarity ~1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "Orthogonal vectors should have cosine similarity ~0.0, got {sim}"
        );
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return if norm_a == norm_b { 1.0 } else { 0.0 };
        }
        dot / (norm_a * norm_b)
    }
}
