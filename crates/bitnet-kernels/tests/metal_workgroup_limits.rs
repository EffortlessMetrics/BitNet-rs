#![allow(clippy::approx_constant)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_abs_diff)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_contains)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::no_effect)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::manual_saturating_arithmetic)]

//! Metal workgroup and dispatch limit validation for Apple Silicon GPUs.
//! Tests ensure correct handling of Metal compute constraints without GPU runtime.
#![cfg(target_os = "macos")]

// ── Apple Silicon Metal constants ───────────────────────────────────

/// Maximum threads per threadgroup on all Apple Silicon (M1–M4).
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// SIMD width (thread execution width) on Apple Silicon GPUs.
const SIMD_WIDTH: u32 = 32;

/// Maximum threadgroup memory (shared memory) per threadgroup in bytes.
const MAX_THREADGROUP_MEMORY: u32 = 32_768;

/// Maximum Metal buffers per render/compute stage.
const MAX_BUFFERS_PER_STAGE: u32 = 31;

/// Metal buffer alignment requirement in bytes.
const BUFFER_ALIGNMENT: usize = 16;

/// Maximum dispatch dimension per axis (Metal spec).
const MAX_DISPATCH_DIM: u32 = u32::MAX;

/// Texture dimension limit for 2D textures on Apple Silicon.
const MAX_TEXTURE_DIM_2D: u32 = 16_384;

// ── Helper structs ──────────────────────────────────────────────────

/// Workgroup (threadgroup) configuration for a Metal compute dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkgroupConfig {
    width: u32,
    height: u32,
    depth: u32,
}

impl WorkgroupConfig {
    fn new_1d(width: u32) -> Self {
        Self { width, height: 1, depth: 1 }
    }

    fn new_2d(width: u32, height: u32) -> Self {
        Self { width, height, depth: 1 }
    }

    fn new_3d(width: u32, height: u32, depth: u32) -> Self {
        Self { width, height, depth }
    }

    fn total_threads(&self) -> u64 {
        self.width as u64 * self.height as u64 * self.depth as u64
    }
}

/// Dispatch configuration: grid dimensions and threadgroup size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DispatchConfig {
    grid: [u32; 3],
    threadgroup: [u32; 3],
}

/// Apple Silicon GPU limits for a specific chip generation.
#[derive(Debug, Clone)]
struct AppleSiliconLimits {
    max_threads_per_threadgroup: u32,
    simd_width: u32,
    max_threadgroup_memory: u32,
    max_buffers_per_stage: u32,
    max_texture_dim_2d: u32,
    chip_name: &'static str,
}

impl AppleSiliconLimits {
    fn m1() -> Self {
        Self {
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            max_threadgroup_memory: 32_768,
            max_buffers_per_stage: 31,
            max_texture_dim_2d: 16_384,
            chip_name: "M1",
        }
    }

    fn m2() -> Self {
        Self { chip_name: "M2", ..Self::m1() }
    }

    fn m3() -> Self {
        Self { chip_name: "M3", ..Self::m1() }
    }

    fn m4() -> Self {
        Self { chip_name: "M4", ..Self::m1() }
    }
}

/// Errors from workgroup / dispatch validation.
#[derive(Debug, PartialEq, Eq)]
enum MetalLimitError {
    ZeroDimension,
    ExceedsMaxThreadsPerThreadgroup { total: u64, max: u32 },
    NotSimdAligned { dim: u32, simd_width: u32 },
    ExceedsThreadgroupMemory { requested: u32, max: u32 },
    ExceedsBufferLimit { count: u32, max: u32 },
    ZeroAlignment,
    ZeroElements,
    DispatchOverflow,
}

// ── Validation helpers ──────────────────────────────────────────────

fn validate_workgroup(
    config: &WorkgroupConfig,
    limits: &AppleSiliconLimits,
) -> Result<(), MetalLimitError> {
    if config.width == 0 || config.height == 0 || config.depth == 0 {
        return Err(MetalLimitError::ZeroDimension);
    }
    let total = config.total_threads();
    if total > limits.max_threads_per_threadgroup as u64 {
        return Err(MetalLimitError::ExceedsMaxThreadsPerThreadgroup {
            total,
            max: limits.max_threads_per_threadgroup,
        });
    }
    Ok(())
}

fn validate_workgroup_simd_aligned(
    config: &WorkgroupConfig,
    limits: &AppleSiliconLimits,
) -> Result<(), MetalLimitError> {
    validate_workgroup(config, limits)?;
    if config.width % limits.simd_width != 0 && !config.width.is_power_of_two() {
        return Err(MetalLimitError::NotSimdAligned {
            dim: config.width,
            simd_width: limits.simd_width,
        });
    }
    Ok(())
}

fn validate_threadgroup_memory(
    bytes: u32,
    limits: &AppleSiliconLimits,
) -> Result<(), MetalLimitError> {
    if bytes > limits.max_threadgroup_memory {
        return Err(MetalLimitError::ExceedsThreadgroupMemory {
            requested: bytes,
            max: limits.max_threadgroup_memory,
        });
    }
    Ok(())
}

fn validate_buffer_count(count: u32, limits: &AppleSiliconLimits) -> Result<(), MetalLimitError> {
    if count > limits.max_buffers_per_stage {
        return Err(MetalLimitError::ExceedsBufferLimit {
            count,
            max: limits.max_buffers_per_stage,
        });
    }
    Ok(())
}

/// Ceiling division that guards against zero divisor.
fn ceil_div(total: u64, group_size: u32) -> Result<u64, MetalLimitError> {
    if group_size == 0 {
        return Err(MetalLimitError::ZeroDimension);
    }
    let g = group_size as u64;
    Ok((total + g - 1) / g)
}

/// Calculate a 1-D dispatch for `total_elements` using `threadgroup_size`.
fn calculate_dispatch_1d(
    total_elements: u64,
    threadgroup_size: u32,
) -> Result<DispatchConfig, MetalLimitError> {
    if total_elements == 0 {
        return Err(MetalLimitError::ZeroElements);
    }
    let groups = ceil_div(total_elements, threadgroup_size)?;
    if groups > MAX_DISPATCH_DIM as u64 {
        return Err(MetalLimitError::DispatchOverflow);
    }
    Ok(DispatchConfig { grid: [groups as u32, 1, 1], threadgroup: [threadgroup_size, 1, 1] })
}

/// Calculate a 2-D dispatch for a matrix of `(rows, cols)`.
fn calculate_dispatch_2d(
    rows: u32,
    cols: u32,
    tg_width: u32,
    tg_height: u32,
) -> Result<DispatchConfig, MetalLimitError> {
    if rows == 0 || cols == 0 {
        return Err(MetalLimitError::ZeroElements);
    }
    let gx = ceil_div(cols as u64, tg_width)?;
    let gy = ceil_div(rows as u64, tg_height)?;
    if gx > MAX_DISPATCH_DIM as u64 || gy > MAX_DISPATCH_DIM as u64 {
        return Err(MetalLimitError::DispatchOverflow);
    }
    Ok(DispatchConfig { grid: [gx as u32, gy as u32, 1], threadgroup: [tg_width, tg_height, 1] })
}

/// Calculate a 3-D dispatch for batched operations `(batch, rows, cols)`.
fn calculate_dispatch_3d(
    batch: u32,
    rows: u32,
    cols: u32,
    tg_x: u32,
    tg_y: u32,
    tg_z: u32,
) -> Result<DispatchConfig, MetalLimitError> {
    if batch == 0 || rows == 0 || cols == 0 {
        return Err(MetalLimitError::ZeroElements);
    }
    let gx = ceil_div(cols as u64, tg_x)?;
    let gy = ceil_div(rows as u64, tg_y)?;
    let gz = ceil_div(batch as u64, tg_z)?;
    if gx > MAX_DISPATCH_DIM as u64 || gy > MAX_DISPATCH_DIM as u64 || gz > MAX_DISPATCH_DIM as u64
    {
        return Err(MetalLimitError::DispatchOverflow);
    }
    Ok(DispatchConfig { grid: [gx as u32, gy as u32, gz as u32], threadgroup: [tg_x, tg_y, tg_z] })
}

/// Align `size` up to the next multiple of `alignment`.
fn align_buffer_size(size: usize, alignment: usize) -> Result<usize, MetalLimitError> {
    if alignment == 0 {
        return Err(MetalLimitError::ZeroAlignment);
    }
    debug_assert!(alignment.is_power_of_two());
    Ok((size + alignment - 1) & !(alignment - 1))
}

/// Compute aligned buffer size for `count` elements of `elem_bytes` each.
fn aligned_element_buffer(
    count: usize,
    elem_bytes: usize,
    alignment: usize,
) -> Result<usize, MetalLimitError> {
    let raw = count.checked_mul(elem_bytes).ok_or(MetalLimitError::DispatchOverflow)?;
    align_buffer_size(raw, alignment)
}

/// Choose an optimal 1-D threadgroup size (multiple of SIMD width, ≤ max).
fn optimal_threadgroup_1d(total: u64) -> u32 {
    if total == 0 {
        return 0;
    }
    let clamped = total.min(MAX_THREADS_PER_THREADGROUP as u64) as u32;
    let rounded = ((clamped + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
    rounded.min(MAX_THREADS_PER_THREADGROUP)
}

// ═════════════════════════════════════════════════════════════════════
//  1. Workgroup Size Validation  (25+ tests)
// ═════════════════════════════════════════════════════════════════════

mod workgroup_size {
    use super::*;

    // ── Valid 1-D sizes ─────────────────────────────────────────────

    #[test]
    fn valid_1d_size_1() {
        let cfg = WorkgroupConfig::new_1d(1);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn valid_1d_size_32() {
        let cfg = WorkgroupConfig::new_1d(32);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn valid_1d_size_64() {
        let cfg = WorkgroupConfig::new_1d(64);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn valid_1d_size_128() {
        let cfg = WorkgroupConfig::new_1d(128);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn valid_1d_size_256() {
        let cfg = WorkgroupConfig::new_1d(256);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn valid_1d_size_512() {
        let cfg = WorkgroupConfig::new_1d(512);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn valid_1d_size_1024() {
        let cfg = WorkgroupConfig::new_1d(1024);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
        assert_eq!(cfg.total_threads(), MAX_THREADS_PER_THREADGROUP as u64);
    }

    #[test]
    fn valid_1d_simd_multiple_96() {
        let cfg = WorkgroupConfig::new_1d(96); // 32 * 3
        assert!(validate_workgroup_simd_aligned(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn valid_1d_simd_multiple_160() {
        let cfg = WorkgroupConfig::new_1d(160); // 32 * 5
        assert!(validate_workgroup_simd_aligned(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    // ── Invalid 1-D sizes ───────────────────────────────────────────

    #[test]
    fn invalid_1d_size_zero() {
        let cfg = WorkgroupConfig::new_1d(0);
        assert_eq!(
            validate_workgroup(&cfg, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::ZeroDimension),
        );
    }

    #[test]
    fn invalid_1d_size_1025() {
        let cfg = WorkgroupConfig::new_1d(1025);
        assert!(matches!(
            validate_workgroup(&cfg, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::ExceedsMaxThreadsPerThreadgroup { total: 1025, .. }),
        ));
    }

    #[test]
    fn invalid_1d_size_2048() {
        let cfg = WorkgroupConfig::new_1d(2048);
        assert!(matches!(
            validate_workgroup(&cfg, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::ExceedsMaxThreadsPerThreadgroup { total: 2048, .. }),
        ));
    }

    #[test]
    fn invalid_1d_not_simd_aligned_33() {
        let cfg = WorkgroupConfig::new_1d(33);
        assert!(matches!(
            validate_workgroup_simd_aligned(&cfg, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::NotSimdAligned { dim: 33, .. }),
        ));
    }

    #[test]
    fn invalid_1d_not_simd_aligned_100() {
        let cfg = WorkgroupConfig::new_1d(100);
        assert!(matches!(
            validate_workgroup_simd_aligned(&cfg, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::NotSimdAligned { dim: 100, .. }),
        ));
    }

    // ── 2-D workgroup validation ────────────────────────────────────

    #[test]
    fn valid_2d_16x16() {
        let cfg = WorkgroupConfig::new_2d(16, 16);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
        assert_eq!(cfg.total_threads(), 256);
    }

    #[test]
    fn valid_2d_32x8() {
        let cfg = WorkgroupConfig::new_2d(32, 8);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
        assert_eq!(cfg.total_threads(), 256);
    }

    #[test]
    fn valid_2d_8x32() {
        let cfg = WorkgroupConfig::new_2d(8, 32);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
        assert_eq!(cfg.total_threads(), 256);
    }

    #[test]
    fn valid_2d_32x32() {
        let cfg = WorkgroupConfig::new_2d(32, 32);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
        assert_eq!(cfg.total_threads(), 1024);
    }

    #[test]
    fn invalid_2d_exceeds_max_33x32() {
        let cfg = WorkgroupConfig::new_2d(33, 32);
        assert!(matches!(
            validate_workgroup(&cfg, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::ExceedsMaxThreadsPerThreadgroup { total: 1056, .. }),
        ));
    }

    #[test]
    fn invalid_2d_zero_height() {
        let cfg = WorkgroupConfig::new_2d(32, 0);
        assert_eq!(
            validate_workgroup(&cfg, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::ZeroDimension),
        );
    }

    #[test]
    fn invalid_2d_zero_width() {
        let cfg = WorkgroupConfig::new_2d(0, 16);
        assert_eq!(
            validate_workgroup(&cfg, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::ZeroDimension),
        );
    }

    // ── 3-D workgroup validation ────────────────────────────────────

    #[test]
    fn valid_3d_4x4x64() {
        let cfg = WorkgroupConfig::new_3d(4, 4, 64);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
        assert_eq!(cfg.total_threads(), 1024);
    }

    #[test]
    fn valid_3d_8x8x16() {
        let cfg = WorkgroupConfig::new_3d(8, 8, 16);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
        assert_eq!(cfg.total_threads(), 1024);
    }

    #[test]
    fn valid_3d_1x1x1() {
        let cfg = WorkgroupConfig::new_3d(1, 1, 1);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn valid_3d_32x4x8() {
        let cfg = WorkgroupConfig::new_3d(32, 4, 8);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
        assert_eq!(cfg.total_threads(), 1024);
    }

    #[test]
    fn invalid_3d_exceeds_max() {
        let cfg = WorkgroupConfig::new_3d(16, 16, 8);
        assert!(matches!(
            validate_workgroup(&cfg, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::ExceedsMaxThreadsPerThreadgroup { total: 2048, .. }),
        ));
    }

    #[test]
    fn invalid_3d_zero_depth() {
        let cfg = WorkgroupConfig::new_3d(8, 8, 0);
        assert_eq!(
            validate_workgroup(&cfg, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::ZeroDimension),
        );
    }

    #[test]
    fn total_threads_product_correct() {
        let cfg = WorkgroupConfig::new_3d(10, 10, 10);
        assert_eq!(cfg.total_threads(), 1000);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }
}

// ═════════════════════════════════════════════════════════════════════
//  2. Dispatch Size Calculation  (25+ tests)
// ═════════════════════════════════════════════════════════════════════

mod dispatch_size {
    use super::*;

    // ── 1-D dispatch ────────────────────────────────────────────────

    #[test]
    fn dispatch_1d_exact_multiple() {
        let d = calculate_dispatch_1d(1024, 256).unwrap();
        assert_eq!(d.grid, [4, 1, 1]);
        assert_eq!(d.threadgroup, [256, 1, 1]);
    }

    #[test]
    fn dispatch_1d_not_exact_rounds_up() {
        let d = calculate_dispatch_1d(1000, 256).unwrap();
        assert_eq!(d.grid, [4, 1, 1]); // ceil(1000/256) = 4
    }

    #[test]
    fn dispatch_1d_single_element() {
        let d = calculate_dispatch_1d(1, 256).unwrap();
        assert_eq!(d.grid, [1, 1, 1]);
    }

    #[test]
    fn dispatch_1d_one_group_needed() {
        let d = calculate_dispatch_1d(128, 256).unwrap();
        assert_eq!(d.grid, [1, 1, 1]);
    }

    #[test]
    fn dispatch_1d_exact_boundary() {
        let d = calculate_dispatch_1d(256, 256).unwrap();
        assert_eq!(d.grid, [1, 1, 1]);
    }

    #[test]
    fn dispatch_1d_boundary_plus_one() {
        let d = calculate_dispatch_1d(257, 256).unwrap();
        assert_eq!(d.grid, [2, 1, 1]);
    }

    #[test]
    fn dispatch_1d_large_vector() {
        let d = calculate_dispatch_1d(1_000_000, 1024).unwrap();
        assert_eq!(d.grid[0], 977); // ceil(1_000_000 / 1024)
    }

    #[test]
    fn dispatch_1d_zero_elements_error() {
        assert_eq!(calculate_dispatch_1d(0, 256), Err(MetalLimitError::ZeroElements),);
    }

    #[test]
    fn dispatch_1d_zero_threadgroup_error() {
        assert_eq!(calculate_dispatch_1d(1024, 0), Err(MetalLimitError::ZeroDimension),);
    }

    #[test]
    fn dispatch_1d_prime_count() {
        let d = calculate_dispatch_1d(997, 32).unwrap();
        assert_eq!(d.grid[0], 32); // ceil(997/32) = 32
    }

    #[test]
    fn dispatch_1d_threadgroup_larger_than_elements() {
        let d = calculate_dispatch_1d(10, 256).unwrap();
        assert_eq!(d.grid, [1, 1, 1]);
        assert_eq!(d.threadgroup, [256, 1, 1]);
    }

    // ── 2-D dispatch ────────────────────────────────────────────────

    #[test]
    fn dispatch_2d_square_matrix() {
        let d = calculate_dispatch_2d(256, 256, 16, 16).unwrap();
        assert_eq!(d.grid, [16, 16, 1]);
    }

    #[test]
    fn dispatch_2d_non_square_matrix() {
        let d = calculate_dispatch_2d(100, 200, 16, 16).unwrap();
        assert_eq!(d.grid[0], 13); // ceil(200/16)
        assert_eq!(d.grid[1], 7); // ceil(100/16)
    }

    #[test]
    fn dispatch_2d_single_row() {
        let d = calculate_dispatch_2d(1, 1024, 32, 1).unwrap();
        assert_eq!(d.grid, [32, 1, 1]);
    }

    #[test]
    fn dispatch_2d_single_column() {
        let d = calculate_dispatch_2d(1024, 1, 1, 32).unwrap();
        assert_eq!(d.grid, [1, 32, 1]);
    }

    #[test]
    fn dispatch_2d_zero_rows_error() {
        assert_eq!(calculate_dispatch_2d(0, 256, 16, 16), Err(MetalLimitError::ZeroElements),);
    }

    #[test]
    fn dispatch_2d_zero_cols_error() {
        assert_eq!(calculate_dispatch_2d(256, 0, 16, 16), Err(MetalLimitError::ZeroElements),);
    }

    #[test]
    fn dispatch_2d_large_matrix() {
        let d = calculate_dispatch_2d(4096, 4096, 16, 16).unwrap();
        assert_eq!(d.grid, [256, 256, 1]);
    }

    // ── 3-D dispatch ────────────────────────────────────────────────

    #[test]
    fn dispatch_3d_batched_matmul() {
        let d = calculate_dispatch_3d(8, 64, 64, 16, 16, 1).unwrap();
        assert_eq!(d.grid, [4, 4, 8]);
    }

    #[test]
    fn dispatch_3d_single_batch() {
        let d = calculate_dispatch_3d(1, 256, 256, 16, 16, 1).unwrap();
        assert_eq!(d.grid, [16, 16, 1]);
    }

    #[test]
    fn dispatch_3d_zero_batch_error() {
        assert_eq!(calculate_dispatch_3d(0, 64, 64, 8, 8, 1), Err(MetalLimitError::ZeroElements),);
    }

    #[test]
    fn dispatch_3d_large_batch() {
        let d = calculate_dispatch_3d(128, 32, 32, 32, 8, 1).unwrap();
        assert_eq!(d.grid, [1, 4, 128]);
    }

    #[test]
    fn dispatch_3d_all_ones() {
        let d = calculate_dispatch_3d(1, 1, 1, 1, 1, 1).unwrap();
        assert_eq!(d.grid, [1, 1, 1]);
    }

    // ── Ceiling division ────────────────────────────────────────────

    #[test]
    fn ceil_div_exact() {
        assert_eq!(ceil_div(256, 32).unwrap(), 8);
    }

    #[test]
    fn ceil_div_remainder() {
        assert_eq!(ceil_div(257, 32).unwrap(), 9);
    }

    #[test]
    fn ceil_div_one() {
        assert_eq!(ceil_div(1, 32).unwrap(), 1);
    }

    #[test]
    fn ceil_div_zero_divisor_error() {
        assert_eq!(ceil_div(100, 0), Err(MetalLimitError::ZeroDimension));
    }
}

// ═════════════════════════════════════════════════════════════════════
//  3. Buffer Alignment  (15+ tests)
// ═════════════════════════════════════════════════════════════════════

mod buffer_alignment {
    use super::*;

    #[test]
    fn align_zero_size() {
        assert_eq!(align_buffer_size(0, 16).unwrap(), 0);
    }

    #[test]
    fn align_already_aligned() {
        assert_eq!(align_buffer_size(16, 16).unwrap(), 16);
    }

    #[test]
    fn align_round_up() {
        assert_eq!(align_buffer_size(1, 16).unwrap(), 16);
    }

    #[test]
    fn align_round_up_17() {
        assert_eq!(align_buffer_size(17, 16).unwrap(), 32);
    }

    #[test]
    fn align_large_size() {
        assert_eq!(align_buffer_size(1000, 16).unwrap(), 1008);
    }

    #[test]
    fn align_zero_alignment_error() {
        assert_eq!(align_buffer_size(100, 0), Err(MetalLimitError::ZeroAlignment),);
    }

    #[test]
    fn align_256_byte_requirement() {
        // Some Metal operations require 256-byte alignment.
        assert_eq!(align_buffer_size(100, 256).unwrap(), 256);
        assert_eq!(align_buffer_size(257, 256).unwrap(), 512);
    }

    // ── Element buffer calculations ─────────────────────────────────

    #[test]
    fn aligned_f32_buffer() {
        // 100 f32 elements = 400 bytes → align to 16 = 400
        assert_eq!(aligned_element_buffer(100, 4, 16).unwrap(), 400);
    }

    #[test]
    fn aligned_f16_buffer() {
        // 100 f16 elements = 200 bytes → align to 16 = 208
        assert_eq!(aligned_element_buffer(100, 2, 16).unwrap(), 208);
    }

    #[test]
    fn aligned_i8_buffer() {
        // 100 i8 elements = 100 bytes → align to 16 = 112
        assert_eq!(aligned_element_buffer(100, 1, 16).unwrap(), 112);
    }

    #[test]
    fn aligned_f32_exact() {
        // 4 f32 elements = 16 bytes → exactly aligned
        assert_eq!(aligned_element_buffer(4, 4, 16).unwrap(), 16);
    }

    #[test]
    fn aligned_f16_odd_count() {
        // 3 f16 elements = 6 bytes → align to 16
        assert_eq!(aligned_element_buffer(3, 2, 16).unwrap(), 16);
    }

    #[test]
    fn aligned_buffer_single_byte() {
        assert_eq!(aligned_element_buffer(1, 1, 16).unwrap(), 16);
    }

    #[test]
    fn aligned_buffer_large_element_count() {
        // 1M f32 elements = 4MB → already aligned to 16
        assert_eq!(aligned_element_buffer(1_000_000, 4, 16).unwrap(), 4_000_000);
    }

    #[test]
    fn aligned_buffer_overflow_protection() {
        // usize::MAX elements × 4 would overflow
        assert_eq!(
            aligned_element_buffer(usize::MAX, 4, 16),
            Err(MetalLimitError::DispatchOverflow),
        );
    }

    // ── Shared memory alignment ─────────────────────────────────────

    #[test]
    fn shared_memory_16_byte_aligned() {
        // Apple Silicon shared memory requires 16-byte alignment.
        for size in [1_usize, 5, 15, 16, 17, 31, 32, 33, 100] {
            let aligned = align_buffer_size(size, 16).unwrap();
            assert_eq!(aligned % 16, 0, "size {size} aligned to {aligned}");
            assert!(aligned >= size);
        }
    }

    #[test]
    fn buffer_offset_alignment() {
        // Buffer offsets must be multiples of 16 bytes.
        let offsets = [0, 16, 32, 48, 64, 256, 4096];
        for &off in &offsets {
            assert_eq!(off % BUFFER_ALIGNMENT, 0, "offset {off} not aligned");
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
//  4. Apple Silicon Constraints  (20+ tests)
// ═════════════════════════════════════════════════════════════════════

mod apple_silicon_constraints {
    use super::*;

    // ── Per-chip limit consistency ──────────────────────────────────

    #[test]
    fn m1_max_threads_1024() {
        let l = AppleSiliconLimits::m1();
        assert_eq!(l.max_threads_per_threadgroup, 1024);
    }

    #[test]
    fn m2_max_threads_1024() {
        let l = AppleSiliconLimits::m2();
        assert_eq!(l.max_threads_per_threadgroup, 1024);
    }

    #[test]
    fn m3_max_threads_1024() {
        let l = AppleSiliconLimits::m3();
        assert_eq!(l.max_threads_per_threadgroup, 1024);
    }

    #[test]
    fn m4_max_threads_1024() {
        let l = AppleSiliconLimits::m4();
        assert_eq!(l.max_threads_per_threadgroup, 1024);
    }

    #[test]
    fn all_chips_simd_width_32() {
        for l in [
            AppleSiliconLimits::m1(),
            AppleSiliconLimits::m2(),
            AppleSiliconLimits::m3(),
            AppleSiliconLimits::m4(),
        ] {
            assert_eq!(l.simd_width, SIMD_WIDTH, "chip {}", l.chip_name);
        }
    }

    #[test]
    fn all_chips_threadgroup_memory_32k() {
        for l in [
            AppleSiliconLimits::m1(),
            AppleSiliconLimits::m2(),
            AppleSiliconLimits::m3(),
            AppleSiliconLimits::m4(),
        ] {
            assert_eq!(l.max_threadgroup_memory, 32_768, "chip {}", l.chip_name);
        }
    }

    #[test]
    fn all_chips_max_buffers_31() {
        for l in [
            AppleSiliconLimits::m1(),
            AppleSiliconLimits::m2(),
            AppleSiliconLimits::m3(),
            AppleSiliconLimits::m4(),
        ] {
            assert_eq!(l.max_buffers_per_stage, 31, "chip {}", l.chip_name);
        }
    }

    #[test]
    fn all_chips_texture_dim_16384() {
        for l in [
            AppleSiliconLimits::m1(),
            AppleSiliconLimits::m2(),
            AppleSiliconLimits::m3(),
            AppleSiliconLimits::m4(),
        ] {
            assert_eq!(l.max_texture_dim_2d, MAX_TEXTURE_DIM_2D, "chip {}", l.chip_name);
        }
    }

    // ── Threadgroup memory validation ───────────────────────────────

    #[test]
    fn threadgroup_memory_exactly_max() {
        assert!(validate_threadgroup_memory(32_768, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn threadgroup_memory_exceeds_max() {
        assert_eq!(
            validate_threadgroup_memory(32_769, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::ExceedsThreadgroupMemory { requested: 32_769, max: 32_768 }),
        );
    }

    #[test]
    fn threadgroup_memory_zero_ok() {
        assert!(validate_threadgroup_memory(0, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn threadgroup_memory_typical_tile() {
        // 16×16 tile of f32 = 1024 bytes — well within limits.
        let bytes = 16 * 16 * 4;
        assert!(validate_threadgroup_memory(bytes, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn threadgroup_memory_large_tile() {
        // 64×64 tile of f32 = 16384 bytes — half of max.
        let bytes = 64 * 64 * 4;
        assert!(validate_threadgroup_memory(bytes, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn threadgroup_memory_max_tile() {
        // 128×64 tile of f32 = 32768 bytes — exactly max.
        let bytes = 128 * 64 * 4;
        assert_eq!(bytes, 32_768);
        assert!(validate_threadgroup_memory(bytes, &AppleSiliconLimits::m1()).is_ok());
    }

    // ── Buffer count validation ─────────────────────────────────────

    #[test]
    fn buffer_count_exactly_max() {
        assert!(validate_buffer_count(31, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn buffer_count_exceeds_max() {
        assert_eq!(
            validate_buffer_count(32, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::ExceedsBufferLimit { count: 32, max: 31 }),
        );
    }

    #[test]
    fn buffer_count_zero_ok() {
        assert!(validate_buffer_count(0, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn buffer_count_typical_kernel() {
        // Typical matmul: A, B, C, params = 4 buffers
        assert!(validate_buffer_count(4, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn buffer_count_complex_kernel() {
        // Complex kernel: 10 buffers
        assert!(validate_buffer_count(10, &AppleSiliconLimits::m1()).is_ok());
    }

    // ── SIMD-aligned optimal threadgroup ────────────────────────────

    #[test]
    fn optimal_1d_small_input() {
        let tg = optimal_threadgroup_1d(10);
        assert_eq!(tg, SIMD_WIDTH); // rounded up to 32
    }

    #[test]
    fn optimal_1d_exact_simd() {
        let tg = optimal_threadgroup_1d(32);
        assert_eq!(tg, SIMD_WIDTH);
    }

    #[test]
    fn optimal_1d_large_input() {
        let tg = optimal_threadgroup_1d(100_000);
        assert_eq!(tg, MAX_THREADS_PER_THREADGROUP); // clamped
    }

    #[test]
    fn optimal_1d_zero() {
        assert_eq!(optimal_threadgroup_1d(0), 0);
    }
}

// ═════════════════════════════════════════════════════════════════════
//  5. Edge Cases and Error Handling  (15+ tests)
// ═════════════════════════════════════════════════════════════════════

mod edge_cases {
    use super::*;

    #[test]
    fn zero_size_tensor_dispatch() {
        assert_eq!(calculate_dispatch_1d(0, 256), Err(MetalLimitError::ZeroElements),);
    }

    #[test]
    fn single_element_tensor() {
        let d = calculate_dispatch_1d(1, 256).unwrap();
        assert_eq!(d.grid, [1, 1, 1]);
    }

    #[test]
    fn very_large_tensor_1d() {
        // 4 billion elements — still fits in u64 dispatch.
        let d = calculate_dispatch_1d(4_000_000_000, 1024).unwrap();
        assert_eq!(d.grid[0], 3_906_250);
    }

    #[test]
    fn workgroup_larger_than_total() {
        let d = calculate_dispatch_1d(5, 1024).unwrap();
        assert_eq!(d.grid, [1, 1, 1]);
        assert_eq!(d.threadgroup[0], 1024);
    }

    #[test]
    fn max_dispatch_dim_boundary() {
        // Ensure we can fill up to MAX_DISPATCH_DIM groups.
        let total = MAX_DISPATCH_DIM as u64; // u32::MAX
        let d = calculate_dispatch_1d(total, 1).unwrap();
        assert_eq!(d.grid[0], MAX_DISPATCH_DIM);
    }

    #[test]
    fn workgroup_all_dims_one() {
        let cfg = WorkgroupConfig::new_3d(1, 1, 1);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
        assert_eq!(cfg.total_threads(), 1);
    }

    #[test]
    fn workgroup_max_in_single_dim() {
        let cfg = WorkgroupConfig::new_1d(1024);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn workgroup_max_split_2d() {
        // 32 × 32 = 1024 (max)
        let cfg = WorkgroupConfig::new_2d(32, 32);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn workgroup_max_split_3d() {
        // 8 × 8 × 16 = 1024
        let cfg = WorkgroupConfig::new_3d(8, 8, 16);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn dispatch_2d_zero_threadgroup_width() {
        assert_eq!(calculate_dispatch_2d(64, 64, 0, 16), Err(MetalLimitError::ZeroDimension),);
    }

    #[test]
    fn dispatch_2d_zero_threadgroup_height() {
        assert_eq!(calculate_dispatch_2d(64, 64, 16, 0), Err(MetalLimitError::ZeroDimension),);
    }

    #[test]
    fn dispatch_3d_zero_threadgroup_z() {
        assert_eq!(calculate_dispatch_3d(4, 64, 64, 8, 8, 0), Err(MetalLimitError::ZeroDimension),);
    }

    #[test]
    fn overflow_protection_element_buffer() {
        assert_eq!(
            aligned_element_buffer(usize::MAX, 2, 16),
            Err(MetalLimitError::DispatchOverflow),
        );
    }

    #[test]
    fn overflow_protection_huge_elements() {
        assert_eq!(
            aligned_element_buffer(usize::MAX / 2, 4, 16),
            Err(MetalLimitError::DispatchOverflow),
        );
    }

    #[test]
    fn workgroup_near_boundary_1023() {
        let cfg = WorkgroupConfig::new_1d(1023);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn workgroup_exactly_boundary_1024() {
        let cfg = WorkgroupConfig::new_1d(1024);
        assert!(validate_workgroup(&cfg, &AppleSiliconLimits::m1()).is_ok());
    }

    #[test]
    fn workgroup_just_over_boundary_1025() {
        let cfg = WorkgroupConfig::new_1d(1025);
        assert!(matches!(
            validate_workgroup(&cfg, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::ExceedsMaxThreadsPerThreadgroup { .. }),
        ));
    }

    #[test]
    fn dispatch_1d_threadgroup_one() {
        // Degenerate case: 1 thread per group.
        let d = calculate_dispatch_1d(100, 1).unwrap();
        assert_eq!(d.grid[0], 100);
    }

    #[test]
    fn workgroup_u32_max_single_dim() {
        let cfg = WorkgroupConfig::new_1d(u32::MAX);
        assert!(matches!(
            validate_workgroup(&cfg, &AppleSiliconLimits::m1()),
            Err(MetalLimitError::ExceedsMaxThreadsPerThreadgroup { .. }),
        ));
    }

    #[test]
    fn dispatch_2d_non_divisible_both_axes() {
        let d = calculate_dispatch_2d(100, 100, 32, 32).unwrap();
        assert_eq!(d.grid[0], 4); // ceil(100/32) = 4
        assert_eq!(d.grid[1], 4); // ceil(100/32) = 4
    }
}
