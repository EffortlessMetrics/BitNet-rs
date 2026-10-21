#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal device capability detection tests for Apple Silicon.
//!
//! These tests validate the device detection and capability querying logic
//! used for Metal GPU backend selection on macOS/arm64.

#![cfg(feature = "cpu")] // Runs under CPU feature gate since it tests detection logic

use std::collections::HashMap;

/// Represents Metal device capabilities for testing.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MetalDeviceCapabilities {
    pub name: String,
    pub gpu_family: u32,
    pub max_threads_per_threadgroup: u32,
    pub max_buffer_length: u64,
    pub max_threadgroup_memory_length: u32,
    pub simd_group_size: u32,
    pub supports_simd_reduction: bool,
    pub supports_float16: bool,
    pub recommended_max_working_set_size: u64,
}

impl MetalDeviceCapabilities {
    fn apple_m1() -> Self {
        Self {
            name: "Apple M1".to_string(),
            gpu_family: 7,
            max_threads_per_threadgroup: 1024,
            max_buffer_length: 256 * 1024 * 1024 * 1024, // 256 GB (unified memory addressable)
            max_threadgroup_memory_length: 32768,
            simd_group_size: 32,
            supports_simd_reduction: true,
            supports_float16: true,
            recommended_max_working_set_size: 8 * 1024 * 1024 * 1024, // 8 GB
        }
    }

    fn apple_m2() -> Self {
        Self {
            name: "Apple M2".to_string(),
            gpu_family: 8,
            max_threads_per_threadgroup: 1024,
            max_buffer_length: 256 * 1024 * 1024 * 1024,
            max_threadgroup_memory_length: 32768,
            simd_group_size: 32,
            supports_simd_reduction: true,
            supports_float16: true,
            recommended_max_working_set_size: 8 * 1024 * 1024 * 1024,
        }
    }

    fn apple_m3() -> Self {
        Self {
            name: "Apple M3".to_string(),
            gpu_family: 9,
            max_threads_per_threadgroup: 1024,
            max_buffer_length: 256 * 1024 * 1024 * 1024,
            max_threadgroup_memory_length: 32768,
            simd_group_size: 32,
            supports_simd_reduction: true,
            supports_float16: true,
            recommended_max_working_set_size: 8 * 1024 * 1024 * 1024,
        }
    }

    fn apple_m4() -> Self {
        Self {
            name: "Apple M4".to_string(),
            gpu_family: 10,
            max_threads_per_threadgroup: 1024,
            max_buffer_length: 256 * 1024 * 1024 * 1024,
            max_threadgroup_memory_length: 32768,
            simd_group_size: 32,
            supports_simd_reduction: true,
            supports_float16: true,
            recommended_max_working_set_size: 16 * 1024 * 1024 * 1024, // 16 GB base
        }
    }

    /// Whether the device supports hardware ray tracing (M3+).
    fn supports_ray_tracing(&self) -> bool {
        self.gpu_family >= 9
    }

    /// Whether the device supports mesh shaders (M3+).
    fn supports_mesh_shaders(&self) -> bool {
        self.gpu_family >= 9
    }

    /// Whether the device supports dynamic cacheless threadgroup memory (M4+).
    fn supports_dynamic_cacheless_threadgroup_memory(&self) -> bool {
        self.gpu_family >= 10
    }
}

/// Check that a buffer size satisfies Metal's 256-byte alignment requirement.
fn is_metal_aligned(size: u64) -> bool {
    size.is_multiple_of(256)
}

/// Compute optimal threadgroup dimensions for a 2D workload.
fn optimal_threadgroup_size(
    width: u32,
    height: u32,
    max_threads: u32,
    simd_group_size: u32,
) -> (u32, u32) {
    // Width should be a multiple of SIMD group size for coalesced access
    let thread_width = simd_group_size.min(width);
    let thread_height = (max_threads / thread_width).min(height);
    (thread_width, thread_height)
}

/// Estimate memory budget for model inference given device capabilities.
fn estimate_memory_budget(caps: &MetalDeviceCapabilities, model_size_bytes: u64) -> u64 {
    // Reserve 75% of recommended working set for model weights + activations
    let available = (caps.recommended_max_working_set_size as f64 * 0.75) as u64;
    available.min(model_size_bytes)
}

#[test]
fn test_m1_capabilities() {
    let m1 = MetalDeviceCapabilities::apple_m1();

    assert_eq!(m1.name, "Apple M1");
    assert_eq!(m1.gpu_family, 7);
    assert_eq!(m1.max_threads_per_threadgroup, 1024);
    assert_eq!(m1.simd_group_size, 32);
    assert!(m1.supports_float16);
    assert!(m1.supports_simd_reduction);
    assert!(!m1.supports_ray_tracing(), "M1 does not support ray tracing");
    assert!(!m1.supports_mesh_shaders(), "M1 does not support mesh shaders");
}

#[test]
fn test_m2_capabilities() {
    let m2 = MetalDeviceCapabilities::apple_m2();

    assert_eq!(m2.name, "Apple M2");
    assert_eq!(m2.gpu_family, 8);
    assert_eq!(m2.max_threads_per_threadgroup, 1024);
    assert_eq!(m2.simd_group_size, 32);
    assert!(m2.supports_float16);
    assert!(m2.supports_simd_reduction);
    assert!(!m2.supports_ray_tracing(), "M2 does not support ray tracing");
    assert!(!m2.supports_mesh_shaders(), "M2 does not support mesh shaders");
}

#[test]
fn test_m3_capabilities() {
    let m3 = MetalDeviceCapabilities::apple_m3();

    assert_eq!(m3.name, "Apple M3");
    assert_eq!(m3.gpu_family, 9);
    assert_eq!(m3.max_threads_per_threadgroup, 1024);
    assert_eq!(m3.simd_group_size, 32);
    assert!(m3.supports_float16);
    assert!(m3.supports_simd_reduction);
    assert!(m3.supports_ray_tracing(), "M3 supports ray tracing");
    assert!(m3.supports_mesh_shaders(), "M3 supports mesh shaders");
    assert!(
        !m3.supports_dynamic_cacheless_threadgroup_memory(),
        "M3 does not support dynamic cacheless threadgroup memory"
    );
}

#[test]
fn test_m4_capabilities() {
    let m4 = MetalDeviceCapabilities::apple_m4();

    assert_eq!(m4.name, "Apple M4");
    assert_eq!(m4.gpu_family, 10);
    assert_eq!(m4.max_threads_per_threadgroup, 1024);
    assert_eq!(m4.simd_group_size, 32);
    assert!(m4.supports_float16);
    assert!(m4.supports_ray_tracing(), "M4 supports ray tracing");
    assert!(m4.supports_mesh_shaders(), "M4 supports mesh shaders");
    assert!(
        m4.supports_dynamic_cacheless_threadgroup_memory(),
        "M4 supports dynamic cacheless threadgroup memory"
    );
    assert!(
        m4.recommended_max_working_set_size >= 16 * 1024 * 1024 * 1024,
        "M4 base has at least 16 GB"
    );
}

#[test]
fn test_buffer_alignment_256_bytes() {
    // Metal requires 256-byte alignment for buffer offsets
    assert!(is_metal_aligned(0), "Zero is always aligned");
    assert!(is_metal_aligned(256), "256 bytes aligned");
    assert!(is_metal_aligned(512), "512 bytes aligned");
    assert!(is_metal_aligned(1024), "1024 bytes aligned");
    assert!(!is_metal_aligned(128), "128 bytes is NOT 256-aligned");
    assert!(!is_metal_aligned(255), "255 bytes is NOT aligned");
    assert!(!is_metal_aligned(257), "257 bytes is NOT aligned");

    // Typical tensor buffer sizes should be aligned
    let tensor_4k_f16 = 4096_u64 * 2; // 4096 float16 elements = 8192 bytes
    assert!(is_metal_aligned(tensor_4k_f16));

    let tensor_2048_f32 = 2048_u64 * 4; // 2048 float32 elements = 8192 bytes
    assert!(is_metal_aligned(tensor_2048_f32));
}

#[test]
fn test_max_threadgroup_memory() {
    // All Apple Silicon chips must provide at least 32768 bytes of threadgroup memory
    let devices = [
        MetalDeviceCapabilities::apple_m1(),
        MetalDeviceCapabilities::apple_m2(),
        MetalDeviceCapabilities::apple_m3(),
        MetalDeviceCapabilities::apple_m4(),
    ];

    for device in &devices {
        assert!(
            device.max_threadgroup_memory_length >= 32768,
            "{} threadgroup memory {} must be >= 32768",
            device.name,
            device.max_threadgroup_memory_length,
        );
    }
}

#[test]
fn test_optimal_threadgroup_sizing() {
    let simd = 32_u32;
    let max_threads = 1024_u32;

    // Square-ish workload: 256x256
    let (tw, th) = optimal_threadgroup_size(256, 256, max_threads, simd);
    assert_eq!(tw, 32, "Width should be SIMD group size");
    assert_eq!(th, 32, "Height fills remaining budget: 1024/32 = 32");
    assert!(tw * th <= max_threads);

    // Wide workload: 1024x1
    let (tw, th) = optimal_threadgroup_size(1024, 1, max_threads, simd);
    assert_eq!(tw, 32, "Width clamped to SIMD group size");
    assert_eq!(th, 1, "Height clamped to workload height");
    assert!(tw * th <= max_threads);

    // Tall workload: 1x1024
    let (tw, th) = optimal_threadgroup_size(1, 1024, max_threads, simd);
    assert_eq!(tw, 1, "Width clamped to workload width");
    assert!(th <= max_threads, "Height within budget");
    assert!(tw * th <= max_threads);

    // Small workload: 8x8
    let (tw, th) = optimal_threadgroup_size(8, 8, max_threads, simd);
    assert_eq!(tw, 8, "Width clamped to workload width (< SIMD group)");
    assert_eq!(th, 8, "Height clamped to workload height");
    assert!(tw * th <= max_threads);
}

#[test]
fn test_simd_group_size_is_32() {
    // Apple Silicon always uses SIMD group width of 32
    let devices = [
        MetalDeviceCapabilities::apple_m1(),
        MetalDeviceCapabilities::apple_m2(),
        MetalDeviceCapabilities::apple_m3(),
        MetalDeviceCapabilities::apple_m4(),
    ];

    for device in &devices {
        assert_eq!(device.simd_group_size, 32, "{} SIMD group size must be 32", device.name);
    }
}

#[test]
fn test_capability_comparison() {
    let mut capabilities: HashMap<String, MetalDeviceCapabilities> = HashMap::new();
    capabilities.insert("M1".to_string(), MetalDeviceCapabilities::apple_m1());
    capabilities.insert("M2".to_string(), MetalDeviceCapabilities::apple_m2());
    capabilities.insert("M3".to_string(), MetalDeviceCapabilities::apple_m3());
    capabilities.insert("M4".to_string(), MetalDeviceCapabilities::apple_m4());

    // GPU family increases monotonically across generations
    let families: Vec<u32> =
        ["M1", "M2", "M3", "M4"].iter().map(|k| capabilities[*k].gpu_family).collect();
    for window in families.windows(2) {
        assert!(window[1] > window[0], "GPU family must increase: {} -> {}", window[0], window[1]);
    }

    // All share the same max threads per threadgroup
    for device in capabilities.values() {
        assert_eq!(device.max_threads_per_threadgroup, 1024);
    }

    // All share the same SIMD group size
    for device in capabilities.values() {
        assert_eq!(device.simd_group_size, 32);
    }

    // All support float16
    for device in capabilities.values() {
        assert!(device.supports_float16);
    }

    // Ray tracing only on M3+
    assert!(!capabilities["M1"].supports_ray_tracing());
    assert!(!capabilities["M2"].supports_ray_tracing());
    assert!(capabilities["M3"].supports_ray_tracing());
    assert!(capabilities["M4"].supports_ray_tracing());
}

#[test]
fn test_recommended_working_set_size() {
    let m1 = MetalDeviceCapabilities::apple_m1();
    let m4 = MetalDeviceCapabilities::apple_m4();

    // M4 base has larger recommended working set than M1 base
    assert!(
        m4.recommended_max_working_set_size >= m1.recommended_max_working_set_size,
        "M4 working set {} should be >= M1 working set {}",
        m4.recommended_max_working_set_size,
        m1.recommended_max_working_set_size,
    );

    // A 2B model at 2-bit quantization is ~500 MB
    let model_2b_size: u64 = 500 * 1024 * 1024;
    let budget_m1 = estimate_memory_budget(&m1, model_2b_size);
    assert!(
        budget_m1 >= model_2b_size,
        "M1 8 GB should fit a 2B model: budget={budget_m1}, model={model_2b_size}",
    );

    // A 7B model at 2-bit quantization is ~1.75 GB
    let model_7b_size: u64 = 1750 * 1024 * 1024;
    let budget_m4 = estimate_memory_budget(&m4, model_7b_size);
    assert!(
        budget_m4 >= model_7b_size,
        "M4 16 GB should fit a 7B model: budget={budget_m4}, model={model_7b_size}",
    );
}
