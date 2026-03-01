//! Metal kernel validation tests for Apple Silicon.
//!
//! Tests Metal kernel configuration, workgroup sizing, buffer alignment,
//! dispatch dimension validation, and Apple Silicon memory architecture
//! assumptions. All tests run without real Metal hardware unless explicitly
//! marked with `#[ignore]`.

use bitnet_common::Device;
use bitnet_kernels::capability_matrix::{
    CapabilityQuery, CompatibilityReport, DeviceCapabilityMatrix, DeviceClass, DeviceProfile,
    OperationCategory, PrecisionSupport, SupportLevel, apple_m1,
};
use bitnet_kernels::gpu_utils::GpuInfo;

// ── Metal hardware constants ────────────────────────────────────────────────

/// Metal requires 256-byte buffer alignment on Apple GPUs.
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Maximum threads per threadgroup on Apple Silicon.
const METAL_MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// Maximum threadgroup memory (32 KB on Apple Silicon).
const METAL_MAX_THREADGROUP_MEMORY: usize = 32 * 1024;

/// Metal maximum buffer size (256 MB guaranteed minimum across all Apple GPUs).
const METAL_MIN_MAX_BUFFER_SIZE: usize = 256 * 1024 * 1024;

/// Typical page size on Apple Silicon (16 KB).
const APPLE_SILICON_PAGE_SIZE: usize = 16 * 1024;

/// Minimum unified memory on any Apple Silicon Mac.
const METAL_MIN_UNIFIED_MEMORY_GB: u64 = 8;

/// SIMD width on Apple Silicon GPUs (32 threads per SIMD group).
const METAL_SIMD_WIDTH: u32 = 32;

// ── Helper: align a size up to Metal buffer alignment ───────────────────────

fn metal_align_up(size: usize) -> usize {
    (size + METAL_BUFFER_ALIGNMENT - 1) & !(METAL_BUFFER_ALIGNMENT - 1)
}

/// Compute optimal threadgroup size for a 1D dispatch.
fn optimal_threadgroup_1d(total_threads: u32) -> u32 {
    if total_threads == 0 {
        return 0;
    }
    let max = METAL_MAX_THREADS_PER_THREADGROUP;
    if total_threads <= max {
        // Round up to next SIMD-width multiple
        let aligned = (total_threads + METAL_SIMD_WIDTH - 1) / METAL_SIMD_WIDTH * METAL_SIMD_WIDTH;
        aligned.min(max)
    } else {
        max
    }
}

/// Compute grid dimensions for a 1D dispatch.
fn dispatch_grid_1d(total_threads: u32, threadgroup_size: u32) -> u32 {
    if threadgroup_size == 0 {
        return 0;
    }
    (total_threads + threadgroup_size - 1) / threadgroup_size
}

/// Validate a Metal dispatch configuration.
fn validate_dispatch(grid: [u32; 3], threadgroup: [u32; 3]) -> Result<(), &'static str> {
    let tg_total = threadgroup[0] as u64 * threadgroup[1] as u64 * threadgroup[2] as u64;
    if tg_total == 0 {
        return Err("threadgroup size must be non-zero in all dimensions");
    }
    if tg_total > METAL_MAX_THREADS_PER_THREADGROUP as u64 {
        return Err("threadgroup exceeds maximum threads per threadgroup");
    }
    if grid[0] == 0 || grid[1] == 0 || grid[2] == 0 {
        return Err("grid size must be non-zero in all dimensions");
    }
    // Each threadgroup dimension must be ≤ 1024
    for dim in &threadgroup {
        if *dim > METAL_MAX_THREADS_PER_THREADGROUP {
            return Err("single threadgroup dimension exceeds maximum");
        }
    }
    Ok(())
}

/// Validate a buffer fits Metal constraints.
fn validate_buffer_size(size: usize) -> Result<usize, &'static str> {
    if size == 0 {
        return Err("buffer size must be non-zero");
    }
    let aligned = metal_align_up(size);
    if aligned > METAL_MIN_MAX_BUFFER_SIZE {
        return Err("buffer exceeds minimum guaranteed Metal buffer size");
    }
    Ok(aligned)
}

/// Validate threadgroup memory allocation.
fn validate_threadgroup_memory(bytes: usize) -> Result<(), &'static str> {
    if bytes > METAL_MAX_THREADGROUP_MEMORY {
        return Err("threadgroup memory exceeds 32 KB limit");
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Workgroup / threadgroup size validation
// ═══════════════════════════════════════════════════════════════════════════

mod workgroup_size_validation {
    use super::*;

    #[test]
    fn threadgroup_size_cannot_exceed_1024() {
        let result = validate_dispatch([1, 1, 1], [1025, 1, 1]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds maximum"));
    }

    #[test]
    fn threadgroup_product_cannot_exceed_1024() {
        // 32 × 32 × 2 = 2048 > 1024
        let result = validate_dispatch([1, 1, 1], [32, 32, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn threadgroup_at_max_1024_is_valid() {
        let result = validate_dispatch([1, 1, 1], [1024, 1, 1]);
        assert!(result.is_ok());
    }

    #[test]
    fn threadgroup_2d_at_max_is_valid() {
        // 32 × 32 = 1024
        let result = validate_dispatch([1, 1, 1], [32, 32, 1]);
        assert!(result.is_ok());
    }

    #[test]
    fn threadgroup_3d_at_max_is_valid() {
        // 8 × 8 × 16 = 1024
        let result = validate_dispatch([1, 1, 1], [8, 8, 16]);
        assert!(result.is_ok());
    }

    #[test]
    fn threadgroup_zero_dimension_is_invalid() {
        let result = validate_dispatch([1, 1, 1], [0, 1, 1]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("non-zero"));
    }

    #[test]
    fn optimal_threadgroup_rounds_to_simd_width() {
        // 33 threads → rounds up to 64 (next multiple of 32)
        let tg = optimal_threadgroup_1d(33);
        assert_eq!(tg % METAL_SIMD_WIDTH, 0);
        assert_eq!(tg, 64);
    }

    #[test]
    fn optimal_threadgroup_exact_simd_multiple() {
        let tg = optimal_threadgroup_1d(128);
        assert_eq!(tg, 128);
    }

    #[test]
    fn optimal_threadgroup_caps_at_max() {
        let tg = optimal_threadgroup_1d(2048);
        assert_eq!(tg, METAL_MAX_THREADS_PER_THREADGROUP);
    }

    #[test]
    fn optimal_threadgroup_single_thread() {
        let tg = optimal_threadgroup_1d(1);
        assert_eq!(tg, METAL_SIMD_WIDTH); // rounds up to 32
    }

    #[test]
    fn optimal_threadgroup_zero_threads() {
        assert_eq!(optimal_threadgroup_1d(0), 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Buffer alignment and sizing
// ═══════════════════════════════════════════════════════════════════════════

mod buffer_alignment {
    use super::*;

    #[test]
    fn alignment_of_one_byte() {
        assert_eq!(metal_align_up(1), METAL_BUFFER_ALIGNMENT);
    }

    #[test]
    fn alignment_already_aligned() {
        assert_eq!(metal_align_up(256), 256);
    }

    #[test]
    fn alignment_just_over_boundary() {
        assert_eq!(metal_align_up(257), 512);
    }

    #[test]
    fn alignment_large_buffer() {
        let size = 1_000_000;
        let aligned = metal_align_up(size);
        assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
        assert!(aligned >= size);
        assert!(aligned - size < METAL_BUFFER_ALIGNMENT);
    }

    #[test]
    fn alignment_zero_stays_zero() {
        assert_eq!(metal_align_up(0), 0);
    }

    #[test]
    fn validate_buffer_rejects_zero() {
        assert!(validate_buffer_size(0).is_err());
    }

    #[test]
    fn validate_buffer_returns_aligned_size() {
        let result = validate_buffer_size(100).unwrap();
        assert_eq!(result, METAL_BUFFER_ALIGNMENT);
    }

    #[test]
    fn validate_buffer_rejects_oversized() {
        let too_big = METAL_MIN_MAX_BUFFER_SIZE + 1;
        assert!(validate_buffer_size(too_big).is_err());
    }

    #[test]
    fn validate_buffer_at_max_size() {
        // Exactly at the limit (already aligned since it's a power-of-two multiple)
        let result = validate_buffer_size(METAL_MIN_MAX_BUFFER_SIZE);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), METAL_MIN_MAX_BUFFER_SIZE);
    }

    #[test]
    fn page_alignment_implies_buffer_alignment() {
        // Apple Silicon page size (16 KB) is a multiple of buffer alignment (256)
        assert_eq!(APPLE_SILICON_PAGE_SIZE % METAL_BUFFER_ALIGNMENT, 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Dispatch dimension validation
// ═══════════════════════════════════════════════════════════════════════════

mod dispatch_dimensions {
    use super::*;

    #[test]
    fn grid_1d_exact_division() {
        let grids = dispatch_grid_1d(1024, 256);
        assert_eq!(grids, 4);
    }

    #[test]
    fn grid_1d_with_remainder() {
        let grids = dispatch_grid_1d(1000, 256);
        assert_eq!(grids, 4); // ceil(1000/256) = 4
    }

    #[test]
    fn grid_1d_single_threadgroup() {
        let grids = dispatch_grid_1d(128, 256);
        assert_eq!(grids, 1);
    }

    #[test]
    fn grid_zero_threadgroup_returns_zero() {
        assert_eq!(dispatch_grid_1d(100, 0), 0);
    }

    #[test]
    fn validate_dispatch_valid_1d() {
        let result = validate_dispatch([4, 1, 1], [256, 1, 1]);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_dispatch_valid_2d() {
        let result = validate_dispatch([8, 8, 1], [16, 16, 1]);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_dispatch_grid_zero_x_rejected() {
        let result = validate_dispatch([0, 1, 1], [256, 1, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn validate_dispatch_grid_zero_y_rejected() {
        let result = validate_dispatch([1, 0, 1], [256, 1, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn validate_dispatch_grid_zero_z_rejected() {
        let result = validate_dispatch([1, 1, 0], [256, 1, 1]);
        assert!(result.is_err());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Threadgroup memory validation
// ═══════════════════════════════════════════════════════════════════════════

mod threadgroup_memory {
    use super::*;

    #[test]
    fn threadgroup_memory_within_limit() {
        assert!(validate_threadgroup_memory(16 * 1024).is_ok());
    }

    #[test]
    fn threadgroup_memory_at_limit() {
        assert!(validate_threadgroup_memory(METAL_MAX_THREADGROUP_MEMORY).is_ok());
    }

    #[test]
    fn threadgroup_memory_exceeds_limit() {
        assert!(validate_threadgroup_memory(METAL_MAX_THREADGROUP_MEMORY + 1).is_err());
    }

    #[test]
    fn threadgroup_memory_zero_is_valid() {
        assert!(validate_threadgroup_memory(0).is_ok());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Kernel argument validation
// ═══════════════════════════════════════════════════════════════════════════

mod kernel_argument_validation {
    use super::*;

    /// Metal kernel argument descriptor (index + expected size).
    struct MetalKernelArg {
        index: u32,
        size: usize,
        is_buffer: bool,
    }

    fn validate_kernel_args(args: &[MetalKernelArg]) -> Result<(), String> {
        let mut seen_indices = std::collections::HashSet::new();
        for arg in args {
            if !seen_indices.insert(arg.index) {
                return Err(format!("duplicate argument index {}", arg.index));
            }
            if arg.is_buffer && arg.size % METAL_BUFFER_ALIGNMENT != 0 {
                return Err(format!(
                    "buffer arg {} size {} not aligned to {}",
                    arg.index, arg.size, METAL_BUFFER_ALIGNMENT
                ));
            }
        }
        // Indices should be contiguous from 0
        let max_idx = args.iter().map(|a| a.index).max().unwrap_or(0);
        if !args.is_empty() && seen_indices.len() != (max_idx + 1) as usize {
            return Err("argument indices are not contiguous from 0".to_string());
        }
        Ok(())
    }

    #[test]
    fn valid_contiguous_args() {
        let args = vec![
            MetalKernelArg { index: 0, size: 256, is_buffer: true },
            MetalKernelArg { index: 1, size: 512, is_buffer: true },
            MetalKernelArg { index: 2, size: 4, is_buffer: false },
        ];
        assert!(validate_kernel_args(&args).is_ok());
    }

    #[test]
    fn duplicate_arg_index_rejected() {
        let args = vec![
            MetalKernelArg { index: 0, size: 256, is_buffer: true },
            MetalKernelArg { index: 0, size: 512, is_buffer: true },
        ];
        let result = validate_kernel_args(&args);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("duplicate"));
    }

    #[test]
    fn non_contiguous_args_rejected() {
        let args = vec![
            MetalKernelArg { index: 0, size: 256, is_buffer: true },
            MetalKernelArg { index: 2, size: 256, is_buffer: true },
        ];
        assert!(validate_kernel_args(&args).is_err());
    }

    #[test]
    fn unaligned_buffer_arg_rejected() {
        let args = vec![MetalKernelArg { index: 0, size: 100, is_buffer: true }];
        let result = validate_kernel_args(&args);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not aligned"));
    }

    #[test]
    fn scalar_arg_ignores_alignment() {
        let args = vec![MetalKernelArg { index: 0, size: 4, is_buffer: false }];
        assert!(validate_kernel_args(&args).is_ok());
    }

    #[test]
    fn empty_args_is_valid() {
        let args: Vec<MetalKernelArg> = vec![];
        assert!(validate_kernel_args(&args).is_ok());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Apple Silicon capability matrix integration
// ═══════════════════════════════════════════════════════════════════════════

mod apple_silicon_capabilities {
    use super::*;

    #[test]
    fn apple_m1_device_class_is_metal() {
        let profile = apple_m1();
        assert_eq!(profile.device_class, DeviceClass::AppleMetal);
    }

    #[test]
    fn apple_m1_has_at_least_8_compute_units() {
        let profile = apple_m1();
        assert!(profile.compute_units >= 8);
    }

    #[test]
    fn apple_m1_unified_memory_at_least_minimum() {
        let profile = apple_m1();
        assert!(
            profile.memory_gb as u64 >= METAL_MIN_UNIFIED_MEMORY_GB,
            "M1 profile reports {}GB, expected >= {}GB",
            profile.memory_gb,
            METAL_MIN_UNIFIED_MEMORY_GB,
        );
    }

    #[test]
    fn apple_m1_supports_fp16_matmul() {
        let q = CapabilityQuery::new(&apple_m1());
        assert!(q.supports(OperationCategory::MatrixOps, PrecisionSupport::FP16));
    }

    #[test]
    fn apple_m1_best_matmul_precision_is_fp16() {
        let q = CapabilityQuery::new(&apple_m1());
        let best = q.best_precision_for(OperationCategory::MatrixOps);
        assert_eq!(best, Some(PrecisionSupport::FP16));
    }

    #[test]
    fn apple_m1_int8_quantized_is_emulated() {
        let profile = apple_m1();
        let level = profile.lookup(OperationCategory::QuantizedOps, PrecisionSupport::INT8);
        assert!(matches!(level, SupportLevel::Emulated));
    }

    #[test]
    fn apple_m1_binary_quantized_is_partial() {
        let profile = apple_m1();
        let level = profile.lookup(OperationCategory::QuantizedOps, PrecisionSupport::Binary);
        assert!(matches!(level, SupportLevel::Partial(_)));
    }

    #[test]
    fn apple_m1_full_support_count_is_positive() {
        let profile = apple_m1();
        assert!(profile.full_support_count() > 0);
    }

    #[test]
    fn apple_m1_compatibility_for_basic_inference() {
        let profile = apple_m1();
        let required = [
            (OperationCategory::MatrixOps, PrecisionSupport::FP32),
            (OperationCategory::NormOps, PrecisionSupport::FP32),
            (OperationCategory::ActivationOps, PrecisionSupport::FP32),
        ];
        let report = CompatibilityReport::generate(&profile, &required);
        assert!(
            report.overall_ready,
            "M1 should support basic FP32 inference: {}",
            report.summary()
        );
    }

    #[test]
    fn apple_m1_compatibility_report_format() {
        let profile = apple_m1();
        let required = [(OperationCategory::MatrixOps, PrecisionSupport::FP32)];
        let report = CompatibilityReport::generate(&profile, &required);
        let summary = report.summary();
        assert!(summary.contains("Apple M1"));
        assert!(summary.contains("READY"));
    }

    #[test]
    fn builtin_matrix_contains_apple_metal() {
        let matrix = DeviceCapabilityMatrix::with_builtin_profiles();
        let metal_profile = matrix.profile_for_class(DeviceClass::AppleMetal);
        assert!(metal_profile.is_some(), "builtin matrix should include AppleMetal");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Unified memory architecture assumptions
// ═══════════════════════════════════════════════════════════════════════════

mod unified_memory {
    use super::*;

    #[test]
    fn device_metal_variant_exists() {
        let _device = Device::Metal;
    }

    #[test]
    fn device_metal_is_distinct_from_cuda() {
        assert_ne!(Device::Metal, Device::Cuda(0));
    }

    #[test]
    fn gpu_info_metal_field_defaults_false() {
        let info = GpuInfo {
            cuda: false,
            cuda_version: None,
            metal: false,
            rocm: false,
            rocm_version: None,
            opengl: false,
            wgpu: false,
        };
        assert!(!info.metal);
        assert!(!info.any_available());
    }

    #[test]
    fn gpu_info_metal_true_implies_available() {
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
    fn gpu_info_summary_mentions_metal_when_present() {
        let info = GpuInfo {
            cuda: false,
            cuda_version: None,
            metal: true,
            rocm: false,
            rocm_version: None,
            opengl: false,
            wgpu: false,
        };
        let summary = info.summary();
        assert!(
            summary.to_lowercase().contains("metal"),
            "summary should mention Metal: {summary}"
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    #[ignore = "requires Metal GPU runtime — run on macOS Apple Silicon hardware"]
    fn runtime_metal_detection_on_macos() {
        let info = bitnet_kernels::gpu_utils::get_gpu_info();
        // On a real macOS Apple Silicon machine, Metal should be detected
        assert!(info.metal, "Metal should be available on macOS Apple Silicon");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Combined / integration validation scenarios
// ═══════════════════════════════════════════════════════════════════════════

mod integration_scenarios {
    use super::*;

    #[test]
    fn matmul_dispatch_configuration() {
        let m: u32 = 512;
        let n: u32 = 512;
        let tg_x: u32 = 16;
        let tg_y: u32 = 16;

        let grid_x = dispatch_grid_1d(n, tg_x);
        let grid_y = dispatch_grid_1d(m, tg_y);

        let result = validate_dispatch([grid_x, grid_y, 1], [tg_x, tg_y, 1]);
        assert!(result.is_ok());
    }

    #[test]
    fn elementwise_dispatch_large_tensor() {
        let num_elements: u32 = 2_000_000;
        let tg_size = optimal_threadgroup_1d(num_elements);
        let grid = dispatch_grid_1d(num_elements, tg_size);

        assert_eq!(tg_size, METAL_MAX_THREADS_PER_THREADGROUP);
        let result = validate_dispatch([grid, 1, 1], [tg_size, 1, 1]);
        assert!(result.is_ok());
    }

    #[test]
    fn buffer_allocation_for_f32_tensor() {
        let elements = 1000;
        let bytes = elements * std::mem::size_of::<f32>();
        let aligned = validate_buffer_size(bytes).unwrap();
        assert!(aligned >= bytes);
        assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
    }

    #[test]
    fn reduction_shared_memory_fits_threadgroup() {
        // Reduction kernel: each thread contributes one f32 to shared memory
        let threads_per_group = 256_usize;
        let shared_mem = threads_per_group * std::mem::size_of::<f32>();
        assert!(validate_threadgroup_memory(shared_mem).is_ok());
    }

    #[test]
    fn attention_shared_memory_for_head_dim_128() {
        // Q, K tiles for one head (head_dim=128, f16 = 2 bytes)
        let head_dim = 128_usize;
        let tile_rows = 8_usize;
        // Q tile + K tile in shared memory
        let shared_mem = 2 * tile_rows * head_dim * 2; // 2 tiles × 8×128 × 2 bytes
        assert!(
            validate_threadgroup_memory(shared_mem).is_ok(),
            "attention shared memory {}B should fit in {}B threadgroup limit",
            shared_mem,
            METAL_MAX_THREADGROUP_MEMORY,
        );
    }
}
