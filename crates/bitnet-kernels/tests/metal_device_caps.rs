#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types)]
//! Metal device capability detection tests for Apple Silicon.
//!
//! Tests the LOGIC of Metal device property detection, feature support,
//! memory limits, compute capabilities, and Apple GPU family classification
//! using mock/simulation structs that mirror real Metal API properties.
//! Runs on any platform (no actual GPU required).

// ── Core types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GpuFamily {
    Apple1,
    Apple2,
    Apple3,
    Apple4,
    Apple5,
    Apple6,
    Apple7,
    Apple8,
    Apple9,
    Common1,
    Common2,
    Common3,
    Metal3,
}

impl GpuFamily {
    fn generation_number(&self) -> Option<u32> {
        match self {
            GpuFamily::Apple1 => Some(1),
            GpuFamily::Apple2 => Some(2),
            GpuFamily::Apple3 => Some(3),
            GpuFamily::Apple4 => Some(4),
            GpuFamily::Apple5 => Some(5),
            GpuFamily::Apple6 => Some(6),
            GpuFamily::Apple7 => Some(7),
            GpuFamily::Apple8 => Some(8),
            GpuFamily::Apple9 => Some(9),
            _ => None,
        }
    }

    fn is_apple_family(&self) -> bool {
        self.generation_number().is_some()
    }

    fn supports_family(&self, required: &GpuFamily) -> bool {
        match (self.generation_number(), required.generation_number()) {
            (Some(have), Some(need)) => have >= need,
            _ => self == required,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Precision {
    Float32,
    Float16,
    BFloat16,
    Int8,
    Int4,
    Int2,
}

#[derive(Debug, Clone)]
struct MetalDeviceInfo {
    name: String,
    gpu_family: GpuFamily,
    max_threads_per_threadgroup: u32,
    max_buffer_length: u64,
    max_threadgroup_memory_length: u32,
    supports_float16: bool,
    supports_bfloat16: bool,
    supports_simdgroup: bool,
    supports_raytracing: bool,
    supports_mesh_shaders: bool,
    supports_dynamic_caching: bool,
    supports_matrix_ops: bool,
    recommended_max_working_set_size: u64,
    simd_width: u32,
    max_total_threadgroup_threads: u32,
    max_texture_2d_width: u32,
    max_texture_2d_height: u32,
    max_texture_3d_size: u32,
    gpu_core_count: u32,
}

// ── Device presets ──────────────────────────────────────────────────────────

impl MetalDeviceInfo {
    fn m1() -> Self {
        Self {
            name: "Apple M1".into(),
            gpu_family: GpuFamily::Apple7,
            max_threads_per_threadgroup: 1024,
            max_buffer_length: 256 * 1024 * 1024 * 1024, // 256 GB unified addr
            max_threadgroup_memory_length: 32 * 1024,
            supports_float16: true,
            supports_bfloat16: false,
            supports_simdgroup: true,
            supports_raytracing: false,
            supports_mesh_shaders: false,
            supports_dynamic_caching: false,
            supports_matrix_ops: false,
            recommended_max_working_set_size: 8 * GB,
            simd_width: 32,
            max_total_threadgroup_threads: 1024,
            max_texture_2d_width: 16384,
            max_texture_2d_height: 16384,
            max_texture_3d_size: 2048,
            gpu_core_count: 8,
        }
    }

    fn m1_pro() -> Self {
        Self {
            name: "Apple M1 Pro".into(),
            recommended_max_working_set_size: 16 * GB,
            gpu_core_count: 16,
            ..Self::m1()
        }
    }

    fn m1_max() -> Self {
        Self {
            name: "Apple M1 Max".into(),
            recommended_max_working_set_size: 32 * GB,
            gpu_core_count: 32,
            ..Self::m1()
        }
    }

    fn m1_ultra() -> Self {
        Self {
            name: "Apple M1 Ultra".into(),
            recommended_max_working_set_size: 64 * GB,
            gpu_core_count: 64,
            ..Self::m1()
        }
    }

    fn m2() -> Self {
        Self {
            name: "Apple M2".into(),
            gpu_family: GpuFamily::Apple8,
            max_threads_per_threadgroup: 1024,
            max_buffer_length: 256 * 1024 * 1024 * 1024,
            max_threadgroup_memory_length: 32 * 1024,
            supports_float16: true,
            supports_bfloat16: false,
            supports_simdgroup: true,
            supports_raytracing: false,
            supports_mesh_shaders: false,
            supports_dynamic_caching: false,
            supports_matrix_ops: false,
            recommended_max_working_set_size: 8 * GB,
            simd_width: 32,
            max_total_threadgroup_threads: 1024,
            max_texture_2d_width: 16384,
            max_texture_2d_height: 16384,
            max_texture_3d_size: 2048,
            gpu_core_count: 10,
        }
    }

    fn m2_pro() -> Self {
        Self {
            name: "Apple M2 Pro".into(),
            recommended_max_working_set_size: 16 * GB,
            gpu_core_count: 19,
            ..Self::m2()
        }
    }

    fn m2_max() -> Self {
        Self {
            name: "Apple M2 Max".into(),
            recommended_max_working_set_size: 32 * GB,
            gpu_core_count: 38,
            ..Self::m2()
        }
    }

    fn m2_ultra() -> Self {
        Self {
            name: "Apple M2 Ultra".into(),
            recommended_max_working_set_size: 64 * GB,
            gpu_core_count: 76,
            ..Self::m2()
        }
    }

    fn m3() -> Self {
        Self {
            name: "Apple M3".into(),
            gpu_family: GpuFamily::Apple9,
            max_threads_per_threadgroup: 1024,
            max_buffer_length: 256 * 1024 * 1024 * 1024,
            max_threadgroup_memory_length: 32 * 1024,
            supports_float16: true,
            supports_bfloat16: true,
            supports_simdgroup: true,
            supports_raytracing: true,
            supports_mesh_shaders: true,
            supports_dynamic_caching: true,
            supports_matrix_ops: true,
            recommended_max_working_set_size: 8 * GB,
            simd_width: 32,
            max_total_threadgroup_threads: 1024,
            max_texture_2d_width: 16384,
            max_texture_2d_height: 16384,
            max_texture_3d_size: 2048,
            gpu_core_count: 10,
        }
    }

    fn m3_pro() -> Self {
        Self {
            name: "Apple M3 Pro".into(),
            recommended_max_working_set_size: 18 * GB,
            gpu_core_count: 18,
            ..Self::m3()
        }
    }

    fn m3_max() -> Self {
        Self {
            name: "Apple M3 Max".into(),
            recommended_max_working_set_size: 36 * GB,
            gpu_core_count: 40,
            ..Self::m3()
        }
    }

    fn m3_ultra() -> Self {
        Self {
            name: "Apple M3 Ultra".into(),
            recommended_max_working_set_size: 64 * GB,
            gpu_core_count: 80,
            ..Self::m3()
        }
    }

    fn m4() -> Self {
        Self {
            name: "Apple M4".into(),
            gpu_family: GpuFamily::Apple9, // Apple9+ (same family ID, enhanced)
            max_threads_per_threadgroup: 1024,
            max_buffer_length: 256 * 1024 * 1024 * 1024,
            max_threadgroup_memory_length: 32 * 1024,
            supports_float16: true,
            supports_bfloat16: true,
            supports_simdgroup: true,
            supports_raytracing: true,
            supports_mesh_shaders: true,
            supports_dynamic_caching: true,
            supports_matrix_ops: true,
            recommended_max_working_set_size: 16 * GB,
            simd_width: 32,
            max_total_threadgroup_threads: 1024,
            max_texture_2d_width: 16384,
            max_texture_2d_height: 16384,
            max_texture_3d_size: 2048,
            gpu_core_count: 10,
        }
    }

    fn m4_pro() -> Self {
        Self {
            name: "Apple M4 Pro".into(),
            recommended_max_working_set_size: 24 * GB,
            gpu_core_count: 20,
            ..Self::m4()
        }
    }

    fn m4_max() -> Self {
        Self {
            name: "Apple M4 Max".into(),
            recommended_max_working_set_size: 64 * GB,
            gpu_core_count: 40,
            ..Self::m4()
        }
    }

    fn a14() -> Self {
        Self {
            name: "Apple A14 Bionic".into(),
            gpu_family: GpuFamily::Apple7,
            max_threads_per_threadgroup: 1024,
            max_buffer_length: 256 * 1024 * 1024 * 1024,
            max_threadgroup_memory_length: 32 * 1024,
            supports_float16: true,
            supports_bfloat16: false,
            supports_simdgroup: true,
            supports_raytracing: false,
            supports_mesh_shaders: false,
            supports_dynamic_caching: false,
            supports_matrix_ops: false,
            recommended_max_working_set_size: 4 * GB,
            simd_width: 32,
            max_total_threadgroup_threads: 1024,
            max_texture_2d_width: 16384,
            max_texture_2d_height: 16384,
            max_texture_3d_size: 2048,
            gpu_core_count: 4,
        }
    }

    fn a15() -> Self {
        Self {
            name: "Apple A15 Bionic".into(),
            gpu_family: GpuFamily::Apple8,
            recommended_max_working_set_size: 6 * GB,
            gpu_core_count: 5,
            ..Self::a14()
        }
    }

    fn a17_pro() -> Self {
        Self {
            name: "Apple A17 Pro".into(),
            gpu_family: GpuFamily::Apple9,
            supports_bfloat16: true,
            supports_raytracing: true,
            supports_mesh_shaders: true,
            supports_dynamic_caching: true,
            supports_matrix_ops: true,
            recommended_max_working_set_size: 6 * GB,
            gpu_core_count: 6,
            ..Self::a14()
        }
    }
}

const GB: u64 = 1024 * 1024 * 1024;
const KB: u32 = 1024;

// ── Validation helpers ──────────────────────────────────────────────────────

fn validate_workgroup_size(device: &MetalDeviceInfo, requested: [u32; 3]) -> bool {
    let total = requested[0] as u64 * requested[1] as u64 * requested[2] as u64;
    total > 0
        && total <= device.max_threads_per_threadgroup as u64
        && requested[0] <= device.max_threads_per_threadgroup
        && requested[1] <= device.max_threads_per_threadgroup
        && requested[2] <= device.max_threads_per_threadgroup
}

fn optimal_threadgroup_size(device: &MetalDeviceInfo, total_elements: u64) -> [u32; 3] {
    let max = device.max_threads_per_threadgroup;
    let simd = device.simd_width;

    if total_elements == 0 {
        return [1, 1, 1];
    }

    // 1-D dispatch: prefer SIMD-aligned width up to max threads
    let width = if total_elements <= simd as u64 {
        total_elements as u32
    } else if total_elements <= max as u64 {
        // Round down to nearest SIMD multiple
        ((total_elements as u32) / simd) * simd
    } else {
        max
    };

    [width.max(1), 1, 1]
}

fn supports_inference_precision(device: &MetalDeviceInfo, precision: Precision) -> bool {
    match precision {
        Precision::Float32 => true, // always supported
        Precision::Float16 => device.supports_float16,
        Precision::BFloat16 => device.supports_bfloat16,
        Precision::Int8 => device.gpu_family.supports_family(&GpuFamily::Apple7),
        Precision::Int4 => device.gpu_family.supports_family(&GpuFamily::Apple8),
        Precision::Int2 => device.gpu_family.supports_family(&GpuFamily::Apple8),
    }
}

/// Returns safe fallback precision when the requested one is unavailable.
fn fallback_precision(device: &MetalDeviceInfo, requested: Precision) -> Precision {
    if supports_inference_precision(device, requested) {
        return requested;
    }
    match requested {
        Precision::BFloat16 => {
            if device.supports_float16 {
                Precision::Float16
            } else {
                Precision::Float32
            }
        }
        Precision::Float16 => Precision::Float32,
        Precision::Int2 | Precision::Int4 => {
            if device.supports_float16 {
                Precision::Float16
            } else {
                Precision::Float32
            }
        }
        _ => Precision::Float32,
    }
}

/// Estimate max model size (parameters) that fits in recommended working set.
fn max_model_params(device: &MetalDeviceInfo, precision: Precision) -> u64 {
    let bytes_per_param: f64 = match precision {
        Precision::Float32 => 4.0,
        Precision::Float16 | Precision::BFloat16 => 2.0,
        Precision::Int8 => 1.0,
        Precision::Int4 => 0.5,
        Precision::Int2 => 0.25,
    };
    // Reserve 20% headroom for activations/KV cache
    let usable = (device.recommended_max_working_set_size as f64) * 0.8;
    (usable / bytes_per_param) as u64
}

/// Check if a buffer allocation is within device limits.
fn can_allocate_buffer(device: &MetalDeviceInfo, size_bytes: u64) -> bool {
    size_bytes <= device.max_buffer_length && size_bytes <= device.recommended_max_working_set_size
}

/// Compute dispatch grid dimensions for a 1-D workload.
fn compute_dispatch_1d(device: &MetalDeviceInfo, total: u64) -> ([u64; 3], [u32; 3]) {
    let tg = optimal_threadgroup_size(device, total);
    let tg_width = tg[0] as u64;
    let grid_x = (total + tg_width - 1) / tg_width;
    ([grid_x, 1, 1], tg)
}

/// Returns true if the device can support the given Metal feature set level.
fn supports_metal_version(device: &MetalDeviceInfo, major: u32, minor: u32) -> bool {
    match (major, minor) {
        (1, _) => true, // Metal 1.x always available
        (2, 0) => device.gpu_family.supports_family(&GpuFamily::Apple3),
        (2, 1) | (2, 2) | (2, 3) | (2, 4) => device.gpu_family.supports_family(&GpuFamily::Apple5),
        (3, 0) | (3, 1) | (3, 2) => device.gpu_family.supports_family(&GpuFamily::Apple9),
        _ => false,
    }
}

/// Returns the chip "tier" for kernel dispatch heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChipTier {
    /// M3/M4 class — use all advanced features
    Flagship,
    /// M1/M2 class — use float16 + simd, no bfloat
    Performance,
    /// A-series mobile — smaller memory, fewer cores
    Mobile,
    /// Unknown / very old
    Legacy,
}

fn classify_chip(device: &MetalDeviceInfo) -> ChipTier {
    if device.supports_dynamic_caching && device.supports_raytracing {
        ChipTier::Flagship
    } else if device.supports_simdgroup && device.supports_float16 {
        if device.gpu_core_count >= 8 { ChipTier::Performance } else { ChipTier::Mobile }
    } else {
        ChipTier::Legacy
    }
}

/// Suggested shared memory tile size for matmul kernels.
fn matmul_tile_size(device: &MetalDeviceInfo) -> u32 {
    let shared = device.max_threadgroup_memory_length;
    // Each tile: 2 matrices × tile² × 4 bytes (f32) or × 2 (f16)
    let bytes_per_element: u32 = if device.supports_float16 { 2 } else { 4 };
    // tile² × 2 × bpe ≤ shared_mem
    let max_tile_sq = shared / (2 * bytes_per_element);
    let tile = (max_tile_sq as f64).sqrt() as u32;
    // Round down to power-of-2 for alignment
    let mut pot = 1u32;
    while pot * 2 <= tile {
        pot *= 2;
    }
    pot.min(64) // cap at 64 for occupancy
}

// ── All device presets for parametric tests ─────────────────────────────────

fn all_devices() -> Vec<MetalDeviceInfo> {
    vec![
        MetalDeviceInfo::m1(),
        MetalDeviceInfo::m1_pro(),
        MetalDeviceInfo::m1_max(),
        MetalDeviceInfo::m1_ultra(),
        MetalDeviceInfo::m2(),
        MetalDeviceInfo::m2_pro(),
        MetalDeviceInfo::m2_max(),
        MetalDeviceInfo::m2_ultra(),
        MetalDeviceInfo::m3(),
        MetalDeviceInfo::m3_pro(),
        MetalDeviceInfo::m3_max(),
        MetalDeviceInfo::m3_ultra(),
        MetalDeviceInfo::m4(),
        MetalDeviceInfo::m4_pro(),
        MetalDeviceInfo::m4_max(),
        MetalDeviceInfo::a14(),
        MetalDeviceInfo::a15(),
        MetalDeviceInfo::a17_pro(),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

mod device_detection {
    use super::*;

    #[test]
    fn m1_has_correct_gpu_family() {
        assert_eq!(MetalDeviceInfo::m1().gpu_family, GpuFamily::Apple7);
    }

    #[test]
    fn m2_has_correct_gpu_family() {
        assert_eq!(MetalDeviceInfo::m2().gpu_family, GpuFamily::Apple8);
    }

    #[test]
    fn m3_has_correct_gpu_family() {
        assert_eq!(MetalDeviceInfo::m3().gpu_family, GpuFamily::Apple9);
    }

    #[test]
    fn m4_has_correct_gpu_family() {
        assert_eq!(MetalDeviceInfo::m4().gpu_family, GpuFamily::Apple9);
    }

    #[test]
    fn a14_shares_family_with_m1() {
        assert_eq!(MetalDeviceInfo::a14().gpu_family, MetalDeviceInfo::m1().gpu_family);
    }

    #[test]
    fn a15_shares_family_with_m2() {
        assert_eq!(MetalDeviceInfo::a15().gpu_family, MetalDeviceInfo::m2().gpu_family);
    }

    #[test]
    fn a17_pro_shares_family_with_m3() {
        assert_eq!(MetalDeviceInfo::a17_pro().gpu_family, MetalDeviceInfo::m3().gpu_family);
    }

    #[test]
    fn pro_max_ultra_share_gpu_family_with_base() {
        let base = MetalDeviceInfo::m1();
        assert_eq!(MetalDeviceInfo::m1_pro().gpu_family, base.gpu_family);
        assert_eq!(MetalDeviceInfo::m1_max().gpu_family, base.gpu_family);
        assert_eq!(MetalDeviceInfo::m1_ultra().gpu_family, base.gpu_family);
    }

    #[test]
    fn m2_variants_share_gpu_family() {
        let base = MetalDeviceInfo::m2();
        for variant in
            [MetalDeviceInfo::m2_pro(), MetalDeviceInfo::m2_max(), MetalDeviceInfo::m2_ultra()]
        {
            assert_eq!(variant.gpu_family, base.gpu_family, "{}", variant.name);
        }
    }

    #[test]
    fn m3_variants_share_gpu_family() {
        let base = MetalDeviceInfo::m3();
        for variant in
            [MetalDeviceInfo::m3_pro(), MetalDeviceInfo::m3_max(), MetalDeviceInfo::m3_ultra()]
        {
            assert_eq!(variant.gpu_family, base.gpu_family, "{}", variant.name);
        }
    }

    #[test]
    fn gpu_family_generation_numbers_are_sequential() {
        let families = [
            GpuFamily::Apple1,
            GpuFamily::Apple2,
            GpuFamily::Apple3,
            GpuFamily::Apple4,
            GpuFamily::Apple5,
            GpuFamily::Apple6,
            GpuFamily::Apple7,
            GpuFamily::Apple8,
            GpuFamily::Apple9,
        ];
        for (i, fam) in families.iter().enumerate() {
            assert_eq!(fam.generation_number(), Some(i as u32 + 1));
        }
    }

    #[test]
    fn common_families_have_no_generation_number() {
        assert_eq!(GpuFamily::Common1.generation_number(), None);
        assert_eq!(GpuFamily::Common2.generation_number(), None);
        assert_eq!(GpuFamily::Common3.generation_number(), None);
    }

    #[test]
    fn metal3_has_no_generation_number() {
        assert_eq!(GpuFamily::Metal3.generation_number(), None);
    }

    #[test]
    fn all_apple_families_are_apple() {
        for fam in [
            GpuFamily::Apple1,
            GpuFamily::Apple2,
            GpuFamily::Apple3,
            GpuFamily::Apple4,
            GpuFamily::Apple5,
            GpuFamily::Apple6,
            GpuFamily::Apple7,
            GpuFamily::Apple8,
            GpuFamily::Apple9,
        ] {
            assert!(fam.is_apple_family(), "{fam:?}");
        }
    }

    #[test]
    fn common_families_not_apple() {
        assert!(!GpuFamily::Common1.is_apple_family());
        assert!(!GpuFamily::Common2.is_apple_family());
        assert!(!GpuFamily::Common3.is_apple_family());
        assert!(!GpuFamily::Metal3.is_apple_family());
    }

    #[test]
    fn chip_tier_m3_is_flagship() {
        assert_eq!(classify_chip(&MetalDeviceInfo::m3()), ChipTier::Flagship);
    }

    #[test]
    fn chip_tier_m4_is_flagship() {
        assert_eq!(classify_chip(&MetalDeviceInfo::m4()), ChipTier::Flagship);
    }

    #[test]
    fn chip_tier_m1_is_performance() {
        assert_eq!(classify_chip(&MetalDeviceInfo::m1()), ChipTier::Performance);
    }

    #[test]
    fn chip_tier_m2_is_performance() {
        assert_eq!(classify_chip(&MetalDeviceInfo::m2()), ChipTier::Performance);
    }

    #[test]
    fn chip_tier_a14_is_mobile() {
        assert_eq!(classify_chip(&MetalDeviceInfo::a14()), ChipTier::Mobile);
    }

    #[test]
    fn chip_tier_a15_is_mobile() {
        assert_eq!(classify_chip(&MetalDeviceInfo::a15()), ChipTier::Mobile);
    }

    #[test]
    fn chip_tier_a17_pro_is_flagship() {
        assert_eq!(classify_chip(&MetalDeviceInfo::a17_pro()), ChipTier::Flagship);
    }

    #[test]
    fn all_devices_have_nonempty_name() {
        for dev in all_devices() {
            assert!(!dev.name.is_empty(), "device name must not be empty");
        }
    }

    #[test]
    fn all_device_names_contain_apple() {
        for dev in all_devices() {
            assert!(dev.name.starts_with("Apple"), "{} should start with Apple", dev.name);
        }
    }

    #[test]
    fn ultra_variants_have_double_max_core_count() {
        assert!(
            MetalDeviceInfo::m1_ultra().gpu_core_count >= 2 * MetalDeviceInfo::m1().gpu_core_count
        );
        assert!(
            MetalDeviceInfo::m2_ultra().gpu_core_count >= 2 * MetalDeviceInfo::m2().gpu_core_count
        );
    }
}

mod feature_support {
    use super::*;

    #[test]
    fn m1_supports_float16() {
        assert!(MetalDeviceInfo::m1().supports_float16);
    }

    #[test]
    fn m1_does_not_support_bfloat16() {
        assert!(!MetalDeviceInfo::m1().supports_bfloat16);
    }

    #[test]
    fn m2_does_not_support_bfloat16() {
        assert!(!MetalDeviceInfo::m2().supports_bfloat16);
    }

    #[test]
    fn m3_supports_bfloat16() {
        assert!(MetalDeviceInfo::m3().supports_bfloat16);
    }

    #[test]
    fn m4_supports_bfloat16() {
        assert!(MetalDeviceInfo::m4().supports_bfloat16);
    }

    #[test]
    fn m1_supports_simdgroup() {
        assert!(MetalDeviceInfo::m1().supports_simdgroup);
    }

    #[test]
    fn all_devices_support_simdgroup() {
        for dev in all_devices() {
            assert!(dev.supports_simdgroup, "{} should support simdgroup", dev.name);
        }
    }

    #[test]
    fn all_devices_support_float16() {
        for dev in all_devices() {
            assert!(dev.supports_float16, "{} should support float16", dev.name);
        }
    }

    #[test]
    fn m1_no_raytracing() {
        assert!(!MetalDeviceInfo::m1().supports_raytracing);
    }

    #[test]
    fn m2_no_raytracing() {
        assert!(!MetalDeviceInfo::m2().supports_raytracing);
    }

    #[test]
    fn m3_supports_raytracing() {
        assert!(MetalDeviceInfo::m3().supports_raytracing);
    }

    #[test]
    fn m4_supports_raytracing() {
        assert!(MetalDeviceInfo::m4().supports_raytracing);
    }

    #[test]
    fn m3_supports_mesh_shaders() {
        assert!(MetalDeviceInfo::m3().supports_mesh_shaders);
    }

    #[test]
    fn m1_no_mesh_shaders() {
        assert!(!MetalDeviceInfo::m1().supports_mesh_shaders);
    }

    #[test]
    fn m3_supports_dynamic_caching() {
        assert!(MetalDeviceInfo::m3().supports_dynamic_caching);
    }

    #[test]
    fn m1_no_dynamic_caching() {
        assert!(!MetalDeviceInfo::m1().supports_dynamic_caching);
    }

    #[test]
    fn m3_supports_matrix_ops() {
        assert!(MetalDeviceInfo::m3().supports_matrix_ops);
    }

    #[test]
    fn m1_no_matrix_ops() {
        assert!(!MetalDeviceInfo::m1().supports_matrix_ops);
    }

    #[test]
    fn bfloat_only_on_apple9_plus() {
        for dev in all_devices() {
            if dev.supports_bfloat16 {
                assert!(
                    dev.gpu_family.supports_family(&GpuFamily::Apple9),
                    "{} has bfloat but is not Apple9+",
                    dev.name
                );
            }
        }
    }

    #[test]
    fn raytracing_only_on_apple9_plus() {
        for dev in all_devices() {
            if dev.supports_raytracing {
                assert!(
                    dev.gpu_family.supports_family(&GpuFamily::Apple9),
                    "{} has raytracing but is not Apple9+",
                    dev.name
                );
            }
        }
    }

    #[test]
    fn all_m_series_base_support_f32_precision() {
        for dev in [
            MetalDeviceInfo::m1(),
            MetalDeviceInfo::m2(),
            MetalDeviceInfo::m3(),
            MetalDeviceInfo::m4(),
        ] {
            assert!(supports_inference_precision(&dev, Precision::Float32));
        }
    }

    #[test]
    fn all_m_series_support_f16_precision() {
        for dev in [
            MetalDeviceInfo::m1(),
            MetalDeviceInfo::m2(),
            MetalDeviceInfo::m3(),
            MetalDeviceInfo::m4(),
        ] {
            assert!(supports_inference_precision(&dev, Precision::Float16));
        }
    }

    #[test]
    fn only_m3_plus_support_bf16_precision() {
        assert!(!supports_inference_precision(&MetalDeviceInfo::m1(), Precision::BFloat16));
        assert!(!supports_inference_precision(&MetalDeviceInfo::m2(), Precision::BFloat16));
        assert!(supports_inference_precision(&MetalDeviceInfo::m3(), Precision::BFloat16));
        assert!(supports_inference_precision(&MetalDeviceInfo::m4(), Precision::BFloat16));
    }

    #[test]
    fn int8_requires_apple7_plus() {
        assert!(supports_inference_precision(&MetalDeviceInfo::m1(), Precision::Int8));
        assert!(supports_inference_precision(&MetalDeviceInfo::m3(), Precision::Int8));
    }

    #[test]
    fn int4_requires_apple8_plus() {
        assert!(!supports_inference_precision(&MetalDeviceInfo::a14(), Precision::Int4));
        assert!(supports_inference_precision(&MetalDeviceInfo::m2(), Precision::Int4));
        assert!(supports_inference_precision(&MetalDeviceInfo::m3(), Precision::Int4));
    }

    #[test]
    fn int2_requires_apple8_plus() {
        assert!(!supports_inference_precision(&MetalDeviceInfo::m1(), Precision::Int2));
        assert!(supports_inference_precision(&MetalDeviceInfo::m2(), Precision::Int2));
        assert!(supports_inference_precision(&MetalDeviceInfo::m4(), Precision::Int2));
    }
}

mod memory_limits {
    use super::*;

    #[test]
    fn m1_base_has_8gb_recommended() {
        assert_eq!(MetalDeviceInfo::m1().recommended_max_working_set_size, 8 * GB);
    }

    #[test]
    fn m1_pro_has_16gb() {
        assert_eq!(MetalDeviceInfo::m1_pro().recommended_max_working_set_size, 16 * GB);
    }

    #[test]
    fn m1_max_has_32gb() {
        assert_eq!(MetalDeviceInfo::m1_max().recommended_max_working_set_size, 32 * GB);
    }

    #[test]
    fn m1_ultra_has_64gb() {
        assert_eq!(MetalDeviceInfo::m1_ultra().recommended_max_working_set_size, 64 * GB);
    }

    #[test]
    fn m2_memory_tiers_increase() {
        let base = MetalDeviceInfo::m2().recommended_max_working_set_size;
        let pro = MetalDeviceInfo::m2_pro().recommended_max_working_set_size;
        let max = MetalDeviceInfo::m2_max().recommended_max_working_set_size;
        let ultra = MetalDeviceInfo::m2_ultra().recommended_max_working_set_size;
        assert!(base < pro, "pro should have more memory than base");
        assert!(pro < max, "max should have more memory than pro");
        assert!(max < ultra, "ultra should have more memory than max");
    }

    #[test]
    fn m3_memory_tiers_increase() {
        let base = MetalDeviceInfo::m3().recommended_max_working_set_size;
        let pro = MetalDeviceInfo::m3_pro().recommended_max_working_set_size;
        let max = MetalDeviceInfo::m3_max().recommended_max_working_set_size;
        let ultra = MetalDeviceInfo::m3_ultra().recommended_max_working_set_size;
        assert!(base < pro);
        assert!(pro < max);
        assert!(max < ultra);
    }

    #[test]
    fn all_devices_have_256gb_buffer_limit() {
        let expected = 256 * GB;
        for dev in all_devices() {
            assert_eq!(dev.max_buffer_length, expected, "{} has wrong max_buffer_length", dev.name);
        }
    }

    #[test]
    fn all_devices_have_32kb_shared_memory() {
        for dev in all_devices() {
            assert_eq!(
                dev.max_threadgroup_memory_length,
                32 * KB,
                "{} has wrong shared memory",
                dev.name
            );
        }
    }

    #[test]
    fn shared_memory_sufficient_for_8x8_f32_tile() {
        // 2 matrices × 8² × 4 bytes = 512 bytes
        for dev in all_devices() {
            assert!(dev.max_threadgroup_memory_length >= 512, "{} can't fit 8x8 tile", dev.name);
        }
    }

    #[test]
    fn shared_memory_sufficient_for_32x32_f16_tile() {
        // 2 matrices × 32² × 2 bytes = 4096 bytes
        for dev in all_devices() {
            assert!(
                dev.max_threadgroup_memory_length >= 4096,
                "{} can't fit 32x32 f16 tile",
                dev.name
            );
        }
    }

    #[test]
    fn can_allocate_1gb_buffer_on_all_devices() {
        for dev in all_devices() {
            assert!(can_allocate_buffer(&dev, GB), "{}", dev.name);
        }
    }

    #[test]
    fn cannot_allocate_more_than_working_set() {
        let dev = MetalDeviceInfo::m1();
        let too_large = dev.recommended_max_working_set_size + 1;
        assert!(!can_allocate_buffer(&dev, too_large));
    }

    #[test]
    fn mobile_devices_have_smaller_working_set() {
        let a14 = MetalDeviceInfo::a14();
        let m1 = MetalDeviceInfo::m1();
        assert!(
            a14.recommended_max_working_set_size < m1.recommended_max_working_set_size,
            "A14 should have less memory than M1"
        );
    }

    #[test]
    fn max_model_params_m1_f16_at_least_3b() {
        let dev = MetalDeviceInfo::m1();
        let params = max_model_params(&dev, Precision::Float16);
        assert!(params >= 3_000_000_000, "M1 8GB should fit 3B+ f16 params, got {params}");
    }

    #[test]
    fn max_model_params_m1_f32_less_than_f16() {
        let dev = MetalDeviceInfo::m1();
        let f32_params = max_model_params(&dev, Precision::Float32);
        let f16_params = max_model_params(&dev, Precision::Float16);
        assert!(f16_params > f32_params, "f16 should fit more params than f32");
    }

    #[test]
    fn max_model_params_int2_more_than_int8() {
        let dev = MetalDeviceInfo::m3();
        let int2 = max_model_params(&dev, Precision::Int2);
        let int8 = max_model_params(&dev, Precision::Int8);
        assert!(int2 > int8);
    }

    #[test]
    fn max_model_params_scales_with_memory() {
        let m1 = max_model_params(&MetalDeviceInfo::m1(), Precision::Float16);
        let m1_max = max_model_params(&MetalDeviceInfo::m1_max(), Precision::Float16);
        assert!(m1_max > m1 * 3, "M1 Max should fit >3× the params of M1");
    }

    #[test]
    fn texture_limits_are_16k_2d() {
        for dev in all_devices() {
            assert_eq!(dev.max_texture_2d_width, 16384, "{}", dev.name);
            assert_eq!(dev.max_texture_2d_height, 16384, "{}", dev.name);
        }
    }

    #[test]
    fn texture_3d_limit_is_2048() {
        for dev in all_devices() {
            assert_eq!(dev.max_texture_3d_size, 2048, "{}", dev.name);
        }
    }
}

mod compute_capabilities {
    use super::*;

    #[test]
    fn all_devices_have_1024_max_threads() {
        for dev in all_devices() {
            assert_eq!(dev.max_threads_per_threadgroup, 1024, "{} expected 1024 threads", dev.name);
        }
    }

    #[test]
    fn all_devices_simd_width_32() {
        for dev in all_devices() {
            assert_eq!(dev.simd_width, 32, "{} expected SIMD width 32", dev.name);
        }
    }

    #[test]
    fn validate_1d_256_on_all_devices() {
        for dev in all_devices() {
            assert!(
                validate_workgroup_size(&dev, [256, 1, 1]),
                "{} should accept [256,1,1]",
                dev.name
            );
        }
    }

    #[test]
    fn validate_2d_32x32_on_all_devices() {
        for dev in all_devices() {
            assert!(
                validate_workgroup_size(&dev, [32, 32, 1]),
                "{} should accept [32,32,1]",
                dev.name
            );
        }
    }

    #[test]
    fn validate_3d_8x8x8() {
        let dev = MetalDeviceInfo::m1();
        assert!(validate_workgroup_size(&dev, [8, 8, 8])); // 512 ≤ 1024
    }

    #[test]
    fn reject_oversized_1d_workgroup() {
        let dev = MetalDeviceInfo::m1();
        assert!(!validate_workgroup_size(&dev, [2048, 1, 1]));
    }

    #[test]
    fn reject_oversized_2d_workgroup() {
        let dev = MetalDeviceInfo::m1();
        assert!(!validate_workgroup_size(&dev, [64, 64, 1])); // 4096 > 1024
    }

    #[test]
    fn reject_oversized_3d_workgroup() {
        let dev = MetalDeviceInfo::m1();
        assert!(!validate_workgroup_size(&dev, [16, 16, 16])); // 4096 > 1024
    }

    #[test]
    fn reject_zero_workgroup_dimension() {
        let dev = MetalDeviceInfo::m1();
        assert!(!validate_workgroup_size(&dev, [0, 1, 1]));
        assert!(!validate_workgroup_size(&dev, [1, 0, 1]));
        assert!(!validate_workgroup_size(&dev, [1, 1, 0]));
    }

    #[test]
    fn optimal_threadgroup_small_workload() {
        let dev = MetalDeviceInfo::m1();
        let tg = optimal_threadgroup_size(&dev, 16);
        assert_eq!(tg, [16, 1, 1]);
    }

    #[test]
    fn optimal_threadgroup_simd_aligned() {
        let dev = MetalDeviceInfo::m1();
        let tg = optimal_threadgroup_size(&dev, 100);
        assert_eq!(tg[0] % dev.simd_width, 0, "should be SIMD-aligned");
        assert!(tg[0] <= 100);
    }

    #[test]
    fn optimal_threadgroup_caps_at_max() {
        let dev = MetalDeviceInfo::m1();
        let tg = optimal_threadgroup_size(&dev, 1_000_000);
        assert_eq!(tg[0], dev.max_threads_per_threadgroup);
    }

    #[test]
    fn optimal_threadgroup_zero_elements() {
        let dev = MetalDeviceInfo::m1();
        let tg = optimal_threadgroup_size(&dev, 0);
        assert_eq!(tg, [1, 1, 1]);
    }

    #[test]
    fn optimal_threadgroup_exact_simd_width() {
        let dev = MetalDeviceInfo::m1();
        let tg = optimal_threadgroup_size(&dev, 32);
        assert_eq!(tg[0], 32);
    }

    #[test]
    fn optimal_threadgroup_exact_max() {
        let dev = MetalDeviceInfo::m1();
        let tg = optimal_threadgroup_size(&dev, 1024);
        assert_eq!(tg[0], 1024);
    }

    #[test]
    fn dispatch_1d_small() {
        let dev = MetalDeviceInfo::m1();
        let (grid, tg) = compute_dispatch_1d(&dev, 256);
        assert_eq!(tg[0], 256);
        assert_eq!(grid[0], 1);
    }

    #[test]
    fn dispatch_1d_large() {
        let dev = MetalDeviceInfo::m1();
        let (grid, tg) = compute_dispatch_1d(&dev, 4096);
        // tg should be 1024, grid should be ceil(4096/1024) = 4
        assert_eq!(tg[0], 1024);
        assert_eq!(grid[0], 4);
    }

    #[test]
    fn dispatch_1d_non_divisible() {
        let dev = MetalDeviceInfo::m1();
        let (grid, tg) = compute_dispatch_1d(&dev, 5000);
        let covered = grid[0] * tg[0] as u64;
        assert!(covered >= 5000, "dispatch must cover all elements");
    }

    #[test]
    fn max_total_threadgroup_threads_matches_max() {
        for dev in all_devices() {
            assert_eq!(
                dev.max_total_threadgroup_threads, dev.max_threads_per_threadgroup,
                "{}: total should equal per-threadgroup max",
                dev.name
            );
        }
    }

    #[test]
    fn m1_gpu_cores_8() {
        assert_eq!(MetalDeviceInfo::m1().gpu_core_count, 8);
    }

    #[test]
    fn m1_pro_gpu_cores_16() {
        assert_eq!(MetalDeviceInfo::m1_pro().gpu_core_count, 16);
    }

    #[test]
    fn m3_max_gpu_cores_40() {
        assert_eq!(MetalDeviceInfo::m3_max().gpu_core_count, 40);
    }

    #[test]
    fn gpu_cores_increase_base_to_ultra() {
        let devices = [
            MetalDeviceInfo::m2(),
            MetalDeviceInfo::m2_pro(),
            MetalDeviceInfo::m2_max(),
            MetalDeviceInfo::m2_ultra(),
        ];
        for w in devices.windows(2) {
            assert!(
                w[1].gpu_core_count > w[0].gpu_core_count,
                "{} should have more cores than {}",
                w[1].name,
                w[0].name
            );
        }
    }

    #[test]
    fn matmul_tile_size_is_power_of_two() {
        for dev in all_devices() {
            let tile = matmul_tile_size(&dev);
            assert!(tile.is_power_of_two(), "{}: tile={tile} not pow2", dev.name);
        }
    }

    #[test]
    fn matmul_tile_size_at_most_64() {
        for dev in all_devices() {
            let tile = matmul_tile_size(&dev);
            assert!(tile <= 64, "{}: tile={tile} exceeds cap", dev.name);
        }
    }

    #[test]
    fn matmul_tile_fits_in_shared_memory() {
        for dev in all_devices() {
            let tile = matmul_tile_size(&dev);
            let bpe: u32 = if dev.supports_float16 { 2 } else { 4 };
            let usage = 2 * tile * tile * bpe;
            assert!(
                usage <= dev.max_threadgroup_memory_length,
                "{}: tile {tile} uses {usage} bytes > {}",
                dev.name,
                dev.max_threadgroup_memory_length
            );
        }
    }
}

mod apple_gpu_family {
    use super::*;

    #[test]
    fn apple9_supports_apple7() {
        assert!(GpuFamily::Apple9.supports_family(&GpuFamily::Apple7));
    }

    #[test]
    fn apple9_supports_apple8() {
        assert!(GpuFamily::Apple9.supports_family(&GpuFamily::Apple8));
    }

    #[test]
    fn apple9_supports_apple9() {
        assert!(GpuFamily::Apple9.supports_family(&GpuFamily::Apple9));
    }

    #[test]
    fn apple7_does_not_support_apple8() {
        assert!(!GpuFamily::Apple7.supports_family(&GpuFamily::Apple8));
    }

    #[test]
    fn apple7_does_not_support_apple9() {
        assert!(!GpuFamily::Apple7.supports_family(&GpuFamily::Apple9));
    }

    #[test]
    fn apple8_supports_apple7() {
        assert!(GpuFamily::Apple8.supports_family(&GpuFamily::Apple7));
    }

    #[test]
    fn apple8_does_not_support_apple9() {
        assert!(!GpuFamily::Apple8.supports_family(&GpuFamily::Apple9));
    }

    #[test]
    fn apple1_supports_self() {
        assert!(GpuFamily::Apple1.supports_family(&GpuFamily::Apple1));
    }

    #[test]
    fn apple1_does_not_support_apple2() {
        assert!(!GpuFamily::Apple1.supports_family(&GpuFamily::Apple2));
    }

    #[test]
    fn common_families_support_self() {
        assert!(GpuFamily::Common1.supports_family(&GpuFamily::Common1));
        assert!(GpuFamily::Common2.supports_family(&GpuFamily::Common2));
        assert!(GpuFamily::Common3.supports_family(&GpuFamily::Common3));
    }

    #[test]
    fn common_does_not_support_different_common() {
        assert!(!GpuFamily::Common1.supports_family(&GpuFamily::Common2));
        assert!(!GpuFamily::Common2.supports_family(&GpuFamily::Common3));
    }

    #[test]
    fn metal3_supports_self() {
        assert!(GpuFamily::Metal3.supports_family(&GpuFamily::Metal3));
    }

    #[test]
    fn metal3_does_not_support_apple_families() {
        assert!(!GpuFamily::Metal3.supports_family(&GpuFamily::Apple9));
    }

    #[test]
    fn apple_does_not_support_common() {
        assert!(!GpuFamily::Apple9.supports_family(&GpuFamily::Common1));
    }

    #[test]
    fn metal_version_1x_always_supported() {
        for dev in all_devices() {
            assert!(supports_metal_version(&dev, 1, 0), "{}", dev.name);
            assert!(supports_metal_version(&dev, 1, 2), "{}", dev.name);
        }
    }

    #[test]
    fn metal_version_3_requires_apple9() {
        assert!(!supports_metal_version(&MetalDeviceInfo::m1(), 3, 0));
        assert!(!supports_metal_version(&MetalDeviceInfo::m2(), 3, 0));
        assert!(supports_metal_version(&MetalDeviceInfo::m3(), 3, 0));
        assert!(supports_metal_version(&MetalDeviceInfo::m4(), 3, 0));
    }

    #[test]
    fn metal_version_2_requires_apple3_plus() {
        // Apple7+ trivially supports Apple3+, so all our presets pass
        for dev in all_devices() {
            assert!(supports_metal_version(&dev, 2, 0), "{}", dev.name);
        }
    }

    #[test]
    fn unknown_metal_version_not_supported() {
        let dev = MetalDeviceInfo::m4();
        assert!(!supports_metal_version(&dev, 99, 0));
    }
}

mod precision_fallback {
    use super::*;

    #[test]
    fn bf16_falls_back_to_f16_on_m1() {
        let dev = MetalDeviceInfo::m1();
        assert_eq!(fallback_precision(&dev, Precision::BFloat16), Precision::Float16);
    }

    #[test]
    fn bf16_stays_bf16_on_m3() {
        let dev = MetalDeviceInfo::m3();
        assert_eq!(fallback_precision(&dev, Precision::BFloat16), Precision::BFloat16);
    }

    #[test]
    fn f32_never_falls_back() {
        for dev in all_devices() {
            assert_eq!(
                fallback_precision(&dev, Precision::Float32),
                Precision::Float32,
                "{}",
                dev.name
            );
        }
    }

    #[test]
    fn f16_stays_f16_on_all_devices() {
        for dev in all_devices() {
            assert_eq!(
                fallback_precision(&dev, Precision::Float16),
                Precision::Float16,
                "{}",
                dev.name
            );
        }
    }

    #[test]
    fn int2_falls_back_to_f16_on_m1() {
        let dev = MetalDeviceInfo::m1();
        assert_eq!(fallback_precision(&dev, Precision::Int2), Precision::Float16);
    }

    #[test]
    fn int4_falls_back_to_f16_on_m1() {
        let dev = MetalDeviceInfo::m1();
        assert_eq!(fallback_precision(&dev, Precision::Int4), Precision::Float16);
    }

    #[test]
    fn int2_stays_on_m2() {
        let dev = MetalDeviceInfo::m2();
        assert_eq!(fallback_precision(&dev, Precision::Int2), Precision::Int2);
    }

    #[test]
    fn int4_stays_on_m3() {
        let dev = MetalDeviceInfo::m3();
        assert_eq!(fallback_precision(&dev, Precision::Int4), Precision::Int4);
    }

    #[test]
    fn int8_stays_on_all_m_series() {
        for dev in [
            MetalDeviceInfo::m1(),
            MetalDeviceInfo::m2(),
            MetalDeviceInfo::m3(),
            MetalDeviceInfo::m4(),
        ] {
            assert_eq!(fallback_precision(&dev, Precision::Int8), Precision::Int8, "{}", dev.name);
        }
    }
}

mod workgroup_validation_edge_cases {
    use super::*;

    #[test]
    fn max_1d_workgroup() {
        let dev = MetalDeviceInfo::m1();
        assert!(validate_workgroup_size(&dev, [1024, 1, 1]));
    }

    #[test]
    fn max_plus_one_1d_rejected() {
        let dev = MetalDeviceInfo::m1();
        assert!(!validate_workgroup_size(&dev, [1025, 1, 1]));
    }

    #[test]
    fn single_thread_workgroup() {
        let dev = MetalDeviceInfo::m1();
        assert!(validate_workgroup_size(&dev, [1, 1, 1]));
    }

    #[test]
    fn asymmetric_2d_valid() {
        let dev = MetalDeviceInfo::m1();
        assert!(validate_workgroup_size(&dev, [512, 2, 1])); // 1024
    }

    #[test]
    fn asymmetric_2d_just_over() {
        let dev = MetalDeviceInfo::m1();
        assert!(!validate_workgroup_size(&dev, [512, 3, 1])); // 1536 > 1024
    }

    #[test]
    fn power_of_two_sizes() {
        let dev = MetalDeviceInfo::m1();
        for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            assert!(
                validate_workgroup_size(&dev, [size, 1, 1]),
                "pow2 size {size} should be valid"
            );
        }
    }

    #[test]
    fn non_power_of_two_sizes() {
        let dev = MetalDeviceInfo::m1();
        for size in [3, 5, 7, 13, 17, 33, 100, 255, 999, 1023] {
            assert!(
                validate_workgroup_size(&dev, [size, 1, 1]),
                "size {size} should be valid (≤1024)"
            );
        }
    }

    #[test]
    fn all_zeros_rejected() {
        let dev = MetalDeviceInfo::m1();
        assert!(!validate_workgroup_size(&dev, [0, 0, 0]));
    }

    #[test]
    fn mixed_zero_rejected() {
        let dev = MetalDeviceInfo::m1();
        assert!(!validate_workgroup_size(&dev, [32, 0, 1]));
    }

    #[test]
    fn exact_cube_root_1024() {
        let dev = MetalDeviceInfo::m1();
        // 10³ = 1000 ≤ 1024
        assert!(validate_workgroup_size(&dev, [10, 10, 10]));
    }

    #[test]
    fn cube_over_limit() {
        let dev = MetalDeviceInfo::m1();
        // 11³ = 1331 > 1024
        assert!(!validate_workgroup_size(&dev, [11, 11, 11]));
    }
}

mod dispatch_grid {
    use super::*;

    #[test]
    fn dispatch_single_element() {
        let dev = MetalDeviceInfo::m1();
        let (grid, tg) = compute_dispatch_1d(&dev, 1);
        assert_eq!(grid[0], 1);
        assert_eq!(tg[0], 1);
    }

    #[test]
    fn dispatch_exact_threadgroup() {
        let dev = MetalDeviceInfo::m1();
        let (grid, tg) = compute_dispatch_1d(&dev, 1024);
        assert_eq!(grid[0], 1);
        assert_eq!(tg[0], 1024);
    }

    #[test]
    fn dispatch_covers_all_elements() {
        let dev = MetalDeviceInfo::m1();
        for n in [1, 31, 32, 33, 100, 1023, 1024, 1025, 4096, 65536, 1_000_000] {
            let (grid, tg) = compute_dispatch_1d(&dev, n);
            let covered = grid[0] * tg[0] as u64;
            assert!(
                covered >= n,
                "dispatch for {n}: grid={} tg={} covers {covered}",
                grid[0],
                tg[0]
            );
        }
    }

    #[test]
    fn dispatch_grid_y_z_always_one() {
        let dev = MetalDeviceInfo::m1();
        for n in [1, 100, 10000, 1_000_000] {
            let (grid, tg) = compute_dispatch_1d(&dev, n);
            assert_eq!(grid[1], 1);
            assert_eq!(grid[2], 1);
            assert_eq!(tg[1], 1);
            assert_eq!(tg[2], 1);
        }
    }

    #[test]
    fn dispatch_overflow_safe_for_large_n() {
        let dev = MetalDeviceInfo::m1();
        let (grid, tg) = compute_dispatch_1d(&dev, u64::MAX / 2);
        let covered = grid[0].checked_mul(tg[0] as u64);
        assert!(covered.is_some(), "should not overflow");
    }
}
