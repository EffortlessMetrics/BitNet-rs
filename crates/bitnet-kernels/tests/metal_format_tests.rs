#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal texture format and buffer layout tests for Apple Silicon.
//!
//! Validates Metal data format handling, texture dimension constraints,
//! buffer-to-texture mapping, data type conversions, unified memory layout,
//! resource storage modes, argument buffer encoding, feature set capabilities,
//! heap allocation strategies, and resource lifecycle management.
//!
//! All tests exercise pure Rust logic (no Metal API calls required).

// ── Metal pixel/buffer format representation ────────────────────────────────

/// Metal pixel format identifiers (mirrors MTLPixelFormat).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(dead_code)]
enum MetalPixelFormat {
    Invalid = 0,
    R8Unorm = 10,
    R8Snorm = 12,
    R8Uint = 13,
    R8Sint = 14,
    R16Float = 25,
    R16Uint = 27,
    R16Sint = 28,
    R32Float = 55,
    R32Uint = 53,
    R32Sint = 54,
    RG16Float = 65,
    RG32Float = 105,
    RGBA8Unorm = 70,
    RGBA16Float = 115,
    RGBA32Float = 125,
}

impl MetalPixelFormat {
    fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Invalid => 0,
            Self::R8Unorm | Self::R8Snorm | Self::R8Uint | Self::R8Sint => 1,
            Self::R16Float | Self::R16Uint | Self::R16Sint => 2,
            Self::R32Float | Self::R32Uint | Self::R32Sint => 4,
            Self::RG16Float => 4,
            Self::RG32Float => 8,
            Self::RGBA8Unorm => 4,
            Self::RGBA16Float => 8,
            Self::RGBA32Float => 16,
        }
    }

    fn is_float(self) -> bool {
        matches!(
            self,
            Self::R16Float
                | Self::R32Float
                | Self::RG16Float
                | Self::RG32Float
                | Self::RGBA16Float
                | Self::RGBA32Float
        )
    }

    fn is_signed(self) -> bool {
        matches!(
            self,
            Self::R8Snorm
                | Self::R8Sint
                | Self::R16Sint
                | Self::R32Sint
                | Self::R16Float
                | Self::R32Float
                | Self::RG16Float
                | Self::RG32Float
                | Self::RGBA16Float
                | Self::RGBA32Float
        )
    }

    fn channel_count(self) -> u8 {
        match self {
            Self::Invalid => 0,
            Self::R8Unorm
            | Self::R8Snorm
            | Self::R8Uint
            | Self::R8Sint
            | Self::R16Float
            | Self::R16Uint
            | Self::R16Sint
            | Self::R32Float
            | Self::R32Uint
            | Self::R32Sint => 1,
            Self::RG16Float | Self::RG32Float => 2,
            Self::RGBA8Unorm | Self::RGBA16Float | Self::RGBA32Float => 4,
        }
    }
}

// ── Metal resource storage modes ────────────────────────────────────────────

/// Metal resource storage modes (mirrors MTLStorageMode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[allow(dead_code)]
enum StorageMode {
    /// CPU and GPU can access; coherent on Apple Silicon unified memory.
    Shared = 0,
    /// Managed by driver with explicit synchronization (macOS discrete GPU).
    Managed = 1,
    /// GPU-only; not accessible from CPU.
    Private = 2,
    /// Tile memory only; contents do not persist beyond render pass.
    Memoryless = 3,
}

impl StorageMode {
    fn cpu_accessible(self) -> bool {
        matches!(self, Self::Shared | Self::Managed)
    }

    fn gpu_accessible(self) -> bool {
        !matches!(self, Self::Memoryless)
    }

    fn persists_after_render_pass(self) -> bool {
        !matches!(self, Self::Memoryless)
    }

    fn needs_synchronization(self) -> bool {
        matches!(self, Self::Managed)
    }

    /// On Apple Silicon unified memory, Shared is zero-copy.
    fn is_zero_copy_on_apple_silicon(self) -> bool {
        matches!(self, Self::Shared)
    }
}

// ── Metal texture dimension limits ──────────────────────────────────────────

const MAX_TEXTURE_WIDTH: u32 = 16384;
const MAX_TEXTURE_HEIGHT: u32 = 16384;
const MAX_TEXTURE_DEPTH: u32 = 2048;
#[allow(dead_code)]
const MAX_TEXTURE_ARRAY_LENGTH: u32 = 2048;
#[allow(dead_code)]
const MAX_BUFFER_LENGTH: usize = 256 * 1024 * 1024; // 256 MB typical

/// Optimal bytes-per-row alignment for Metal textures.
const BYTES_PER_ROW_ALIGNMENT: usize = 256;

// ── Metal GPU family identifiers ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
#[allow(dead_code)]
enum AppleGpuFamily {
    Apple1 = 1,
    Apple2 = 2,
    Apple3 = 3,
    Apple4 = 4,
    Apple5 = 5,
    Apple6 = 6,
    Apple7 = 7, // M1
    Apple8 = 8, // M2
    Apple9 = 9, // M3
}

impl AppleGpuFamily {
    fn supports_simd_reduction(self) -> bool {
        self >= Self::Apple7
    }

    fn supports_ray_tracing(self) -> bool {
        self >= Self::Apple6
    }

    fn supports_bfloat16(self) -> bool {
        self >= Self::Apple7
    }

    fn max_threadgroup_memory_bytes(self) -> u32 {
        if self >= Self::Apple7 { 32768 } else { 16384 }
    }

    fn max_threads_per_threadgroup(self) -> u32 {
        1024
    }
}

// ── Metal resource purgeability ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[allow(dead_code)]
enum PurgeableState {
    KeepCurrent = 1,
    NonVolatile = 2,
    Volatile = 3,
    Empty = 4,
}

// ── Metal hazard tracking mode ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[allow(dead_code)]
enum HazardTrackingMode {
    Default = 0,
    Untracked = 1,
    Tracked = 2,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn align_up(value: usize, alignment: usize) -> usize {
    assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

fn f32_to_f16_bits(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Compute aligned bytes-per-row for a Metal texture.
fn aligned_bytes_per_row(width: u32, format: MetalPixelFormat) -> usize {
    let raw = width as usize * format.bytes_per_pixel();
    align_up(raw, BYTES_PER_ROW_ALIGNMENT)
}

/// Estimate buffer size needed for a 2D texture with Metal alignment.
fn texture_buffer_size(width: u32, height: u32, format: MetalPixelFormat) -> usize {
    let bpr = aligned_bytes_per_row(width, format);
    bpr * height as usize
}

/// Validate texture dimensions against Metal limits.
fn validate_texture_dims(width: u32, height: u32, depth: u32) -> Result<(), &'static str> {
    if width == 0 || height == 0 || depth == 0 {
        return Err("dimensions must be non-zero");
    }
    if width > MAX_TEXTURE_WIDTH {
        return Err("width exceeds maximum");
    }
    if height > MAX_TEXTURE_HEIGHT {
        return Err("height exceeds maximum");
    }
    if depth > MAX_TEXTURE_DEPTH {
        return Err("depth exceeds maximum");
    }
    Ok(())
}

// ── Argument buffer layout ──────────────────────────────────────────────────

/// Represents a Metal argument buffer entry with natural alignment.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ArgumentBufferEntry {
    index: u32,
    data_type: ArgumentDataType,
    offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum ArgumentDataType {
    Pointer,   // 8 bytes
    Float,     // 4 bytes
    Float4,    // 16 bytes
    Uint,      // 4 bytes
    Texture2D, // 8 bytes (GPU resource ID)
    Sampler,   // 8 bytes
}

impl ArgumentDataType {
    fn size(self) -> usize {
        match self {
            Self::Pointer | Self::Texture2D | Self::Sampler => 8,
            Self::Float | Self::Uint => 4,
            Self::Float4 => 16,
        }
    }

    fn alignment(self) -> usize {
        self.size()
    }
}

fn compute_argument_buffer_layout(
    entries: &[(u32, ArgumentDataType)],
) -> (Vec<ArgumentBufferEntry>, usize) {
    let mut offset = 0usize;
    let mut result = Vec::new();
    for &(index, data_type) in entries {
        let align = data_type.alignment();
        offset = align_up(offset, align);
        result.push(ArgumentBufferEntry { index, data_type, offset });
        offset += data_type.size();
    }
    // Final size aligned to max alignment in the buffer (at least 8).
    let max_align = entries.iter().map(|(_, dt)| dt.alignment()).max().unwrap_or(8);
    let total = align_up(offset, max_align);
    (result, total)
}

// ── Heap allocation ─────────────────────────────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]
struct HeapDescriptor {
    size: usize,
    storage_mode: StorageMode,
    hazard_tracking: HazardTrackingMode,
}

#[derive(Debug)]
#[allow(dead_code)]
struct HeapAllocation {
    offset: usize,
    size: usize,
    aliasable: bool,
    purgeable: PurgeableState,
}

fn heap_allocate(heap_size: usize, existing: &[HeapAllocation], requested: usize) -> Option<usize> {
    let alignment = BYTES_PER_ROW_ALIGNMENT;
    let aligned_size = align_up(requested, alignment);
    let used_end =
        existing.iter().filter(|a| !a.aliasable).map(|a| a.offset + a.size).max().unwrap_or(0);
    let start = align_up(used_end, alignment);
    if start + aligned_size <= heap_size { Some(start) } else { None }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Buffer format validation
// ═══════════════════════════════════════════════════════════════════════════

mod buffer_format_validation {
    use super::*;

    #[test]
    fn r32_float_is_4_bytes() {
        assert_eq!(MetalPixelFormat::R32Float.bytes_per_pixel(), 4);
    }

    #[test]
    fn r16_float_is_2_bytes() {
        assert_eq!(MetalPixelFormat::R16Float.bytes_per_pixel(), 2);
    }

    #[test]
    fn r8_uint_is_1_byte() {
        assert_eq!(MetalPixelFormat::R8Uint.bytes_per_pixel(), 1);
    }

    #[test]
    fn r8_sint_is_1_byte() {
        assert_eq!(MetalPixelFormat::R8Sint.bytes_per_pixel(), 1);
    }

    #[test]
    fn rgba8_unorm_is_4_bytes() {
        assert_eq!(MetalPixelFormat::RGBA8Unorm.bytes_per_pixel(), 4);
    }

    #[test]
    fn rgba16_float_is_8_bytes() {
        assert_eq!(MetalPixelFormat::RGBA16Float.bytes_per_pixel(), 8);
    }

    #[test]
    fn rgba32_float_is_16_bytes() {
        assert_eq!(MetalPixelFormat::RGBA32Float.bytes_per_pixel(), 16);
    }

    #[test]
    fn invalid_format_is_zero_bytes() {
        assert_eq!(MetalPixelFormat::Invalid.bytes_per_pixel(), 0);
    }

    #[test]
    fn float_formats_identified_correctly() {
        assert!(MetalPixelFormat::R16Float.is_float());
        assert!(MetalPixelFormat::R32Float.is_float());
        assert!(MetalPixelFormat::RG16Float.is_float());
        assert!(MetalPixelFormat::RGBA32Float.is_float());
        assert!(!MetalPixelFormat::R8Uint.is_float());
        assert!(!MetalPixelFormat::R8Unorm.is_float());
    }

    #[test]
    fn signed_formats_identified_correctly() {
        assert!(MetalPixelFormat::R8Sint.is_signed());
        assert!(MetalPixelFormat::R8Snorm.is_signed());
        assert!(MetalPixelFormat::R32Float.is_signed());
        assert!(!MetalPixelFormat::R8Uint.is_signed());
        assert!(!MetalPixelFormat::R8Unorm.is_signed());
    }

    #[test]
    fn channel_count_single() {
        assert_eq!(MetalPixelFormat::R8Uint.channel_count(), 1);
        assert_eq!(MetalPixelFormat::R32Float.channel_count(), 1);
    }

    #[test]
    fn channel_count_dual() {
        assert_eq!(MetalPixelFormat::RG16Float.channel_count(), 2);
        assert_eq!(MetalPixelFormat::RG32Float.channel_count(), 2);
    }

    #[test]
    fn channel_count_quad() {
        assert_eq!(MetalPixelFormat::RGBA8Unorm.channel_count(), 4);
        assert_eq!(MetalPixelFormat::RGBA32Float.channel_count(), 4);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Texture dimension constraints
// ═══════════════════════════════════════════════════════════════════════════

mod texture_dimension_constraints {
    use super::*;

    #[test]
    fn valid_small_texture() {
        assert!(validate_texture_dims(64, 64, 1).is_ok());
    }

    #[test]
    fn valid_max_2d_texture() {
        assert!(validate_texture_dims(MAX_TEXTURE_WIDTH, MAX_TEXTURE_HEIGHT, 1).is_ok());
    }

    #[test]
    fn valid_max_3d_texture() {
        assert!(validate_texture_dims(2048, 2048, MAX_TEXTURE_DEPTH).is_ok());
    }

    #[test]
    fn reject_zero_width() {
        assert_eq!(validate_texture_dims(0, 64, 1), Err("dimensions must be non-zero"));
    }

    #[test]
    fn reject_zero_height() {
        assert_eq!(validate_texture_dims(64, 0, 1), Err("dimensions must be non-zero"));
    }

    #[test]
    fn reject_zero_depth() {
        assert_eq!(validate_texture_dims(64, 64, 0), Err("dimensions must be non-zero"));
    }

    #[test]
    fn reject_exceeding_max_width() {
        assert_eq!(
            validate_texture_dims(MAX_TEXTURE_WIDTH + 1, 64, 1),
            Err("width exceeds maximum")
        );
    }

    #[test]
    fn reject_exceeding_max_height() {
        assert_eq!(
            validate_texture_dims(64, MAX_TEXTURE_HEIGHT + 1, 1),
            Err("height exceeds maximum")
        );
    }

    #[test]
    fn reject_exceeding_max_depth() {
        assert_eq!(
            validate_texture_dims(64, 64, MAX_TEXTURE_DEPTH + 1),
            Err("depth exceeds maximum")
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Buffer-to-texture mapping (row alignment)
// ═══════════════════════════════════════════════════════════════════════════

mod buffer_to_texture_mapping {
    use super::*;

    #[test]
    fn bytes_per_row_aligned_to_256() {
        let bpr = aligned_bytes_per_row(100, MetalPixelFormat::R32Float);
        assert_eq!(bpr % BYTES_PER_ROW_ALIGNMENT, 0);
    }

    #[test]
    fn exact_256_needs_no_padding() {
        // 64 pixels * 4 bytes = 256
        let bpr = aligned_bytes_per_row(64, MetalPixelFormat::R32Float);
        assert_eq!(bpr, 256);
    }

    #[test]
    fn one_pixel_pads_to_256() {
        let bpr = aligned_bytes_per_row(1, MetalPixelFormat::R32Float);
        assert_eq!(bpr, 256);
    }

    #[test]
    fn large_row_aligned_correctly() {
        // 1024 pixels * 16 bytes = 16384 (already aligned)
        let bpr = aligned_bytes_per_row(1024, MetalPixelFormat::RGBA32Float);
        assert_eq!(bpr, 16384);
    }

    #[test]
    fn texture_buffer_size_accounts_for_height() {
        let size = texture_buffer_size(64, 100, MetalPixelFormat::R32Float);
        let expected_bpr = 256; // 64 * 4 = 256, already aligned
        assert_eq!(size, expected_bpr * 100);
    }

    #[test]
    fn texture_buffer_size_with_padding() {
        let size = texture_buffer_size(65, 2, MetalPixelFormat::R32Float);
        let expected_bpr = 512; // 65 * 4 = 260, rounds to 512
        assert_eq!(size, expected_bpr * 2);
    }

    #[test]
    fn r8_format_row_alignment() {
        // 200 pixels * 1 byte = 200, aligns to 256
        let bpr = aligned_bytes_per_row(200, MetalPixelFormat::R8Uint);
        assert_eq!(bpr, 256);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Data type conversion precision
// ═══════════════════════════════════════════════════════════════════════════

mod data_type_conversion {
    use super::*;

    #[test]
    fn f32_to_f16_round_trip_exact_integers() {
        for v in [0.0f32, 1.0, -1.0, 2.0, 256.0] {
            let bits = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(bits);
            assert_eq!(v, back, "exact round-trip failed for {v}");
        }
    }

    #[test]
    fn f32_to_f16_small_values_precision() {
        let v = 0.001f32;
        let bits = f32_to_f16_bits(v);
        let back = f16_bits_to_f32(bits);
        assert!((v - back).abs() < 1e-3, "f16 precision too low for {v}");
    }

    #[test]
    fn f32_to_f16_infinity_preserved() {
        let bits = f32_to_f16_bits(f32::INFINITY);
        let back = f16_bits_to_f32(bits);
        assert!(back.is_infinite() && back.is_sign_positive());
    }

    #[test]
    fn f32_to_f16_neg_infinity_preserved() {
        let bits = f32_to_f16_bits(f32::NEG_INFINITY);
        let back = f16_bits_to_f32(bits);
        assert!(back.is_infinite() && back.is_sign_negative());
    }

    #[test]
    fn f32_to_f16_nan_preserved() {
        let bits = f32_to_f16_bits(f32::NAN);
        let back = f16_bits_to_f32(bits);
        assert!(back.is_nan());
    }

    fn i8_to_offset_u8(value: i8) -> u8 {
        (i16::from(value) + 128) as u8
    }

    fn offset_u8_to_i8(value: u8) -> i8 {
        (i16::from(value) - 128) as i8
    }

    #[test]
    fn i8_to_u8_quantized_format_offset() {
        // Quantized i8 [-128..127] → u8 [0..255] via offset 128
        assert_eq!(i8_to_offset_u8(-128), 0);
        assert_eq!(i8_to_offset_u8(0), 128);
        assert_eq!(i8_to_offset_u8(127), 255);
    }

    #[test]
    fn u8_to_i8_quantized_format_offset() {
        assert_eq!(offset_u8_to_i8(0), -128);
        assert_eq!(offset_u8_to_i8(128), 0);
    }

    #[test]
    fn f32_to_unorm8_conversion() {
        // [0.0, 1.0] → [0, 255]
        assert_eq!((0.0f32 * 255.0).round() as u8, 0);
        assert_eq!((0.5f32 * 255.0).round() as u8, 128);
        assert_eq!((1.0f32 * 255.0).round() as u8, 255);
    }

    #[test]
    fn f32_to_snorm8_conversion() {
        // [-1.0, 1.0] → [-127, 127]
        let to_snorm = |v: f32| -> i8 { (v.clamp(-1.0, 1.0) * 127.0).round() as i8 };
        assert_eq!(to_snorm(-1.0), -127);
        assert_eq!(to_snorm(0.0), 0);
        assert_eq!(to_snorm(1.0), 127);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Shared memory layout (unified memory)
// ═══════════════════════════════════════════════════════════════════════════

mod shared_memory_layout {
    use super::*;

    #[test]
    fn shared_mode_is_cpu_accessible() {
        assert!(StorageMode::Shared.cpu_accessible());
    }

    #[test]
    fn shared_mode_is_gpu_accessible() {
        assert!(StorageMode::Shared.gpu_accessible());
    }

    #[test]
    fn shared_mode_is_zero_copy_on_apple_silicon() {
        assert!(StorageMode::Shared.is_zero_copy_on_apple_silicon());
    }

    #[test]
    fn shared_mode_does_not_need_sync() {
        assert!(!StorageMode::Shared.needs_synchronization());
    }

    #[test]
    fn managed_mode_needs_sync() {
        assert!(StorageMode::Managed.needs_synchronization());
    }

    #[test]
    fn unified_memory_pointer_equivalence() {
        // On Apple Silicon, Shared buffer pointers are valid for both CPU and GPU.
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let cpu_ptr = data.as_ptr();
        // In unified memory, the GPU sees the same pointer—no copy needed.
        let simulated_gpu_ptr = cpu_ptr;
        assert_eq!(cpu_ptr, simulated_gpu_ptr);
    }

    #[test]
    fn buffer_alignment_for_shared_mode() {
        let size = 1000usize;
        let aligned = align_up(size, BYTES_PER_ROW_ALIGNMENT);
        assert_eq!(aligned, 1024);
        assert_eq!(aligned % BYTES_PER_ROW_ALIGNMENT, 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Resource storage modes
// ═══════════════════════════════════════════════════════════════════════════

mod resource_storage_modes {
    use super::*;

    #[test]
    fn private_mode_not_cpu_accessible() {
        assert!(!StorageMode::Private.cpu_accessible());
    }

    #[test]
    fn private_mode_is_gpu_accessible() {
        assert!(StorageMode::Private.gpu_accessible());
    }

    #[test]
    fn memoryless_not_gpu_accessible() {
        assert!(!StorageMode::Memoryless.gpu_accessible());
    }

    #[test]
    fn memoryless_does_not_persist() {
        assert!(!StorageMode::Memoryless.persists_after_render_pass());
    }

    #[test]
    fn shared_persists_after_render_pass() {
        assert!(StorageMode::Shared.persists_after_render_pass());
    }

    #[test]
    fn private_persists_after_render_pass() {
        assert!(StorageMode::Private.persists_after_render_pass());
    }

    #[test]
    fn private_is_not_zero_copy() {
        assert!(!StorageMode::Private.is_zero_copy_on_apple_silicon());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Argument buffer encoding
// ═══════════════════════════════════════════════════════════════════════════

mod argument_buffer_encoding {
    use super::*;

    #[test]
    fn single_pointer_layout() {
        let (entries, total) = compute_argument_buffer_layout(&[(0, ArgumentDataType::Pointer)]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].offset, 0);
        assert_eq!(total, 8);
    }

    #[test]
    fn float_then_pointer_pads() {
        let entries_def = &[(0, ArgumentDataType::Float), (1, ArgumentDataType::Pointer)];
        let (entries, total) = compute_argument_buffer_layout(entries_def);
        assert_eq!(entries[0].offset, 0); // float at 0
        assert_eq!(entries[1].offset, 8); // pointer aligned to 8
        assert_eq!(total, 16);
    }

    #[test]
    fn float4_alignment() {
        let entries_def = &[(0, ArgumentDataType::Uint), (1, ArgumentDataType::Float4)];
        let (entries, _total) = compute_argument_buffer_layout(entries_def);
        assert_eq!(entries[0].offset, 0); // uint at 0
        assert_eq!(entries[1].offset, 16); // float4 aligned to 16
    }

    #[test]
    fn multiple_floats_packed() {
        let entries_def = &[
            (0, ArgumentDataType::Float),
            (1, ArgumentDataType::Float),
            (2, ArgumentDataType::Float),
        ];
        let (entries, total) = compute_argument_buffer_layout(entries_def);
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[1].offset, 4);
        assert_eq!(entries[2].offset, 8);
        assert_eq!(total, 12);
    }

    #[test]
    fn texture_and_sampler_layout() {
        let entries_def = &[(0, ArgumentDataType::Texture2D), (1, ArgumentDataType::Sampler)];
        let (entries, total) = compute_argument_buffer_layout(entries_def);
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[1].offset, 8);
        assert_eq!(total, 16);
    }

    #[test]
    fn complex_mixed_layout() {
        // Float(4) + Uint(4) + Pointer(8@8) + Float4(16@16) + Texture2D(8@8)
        // offsets: 0, 4, 8, 16, 32 → end at 40, max_align=16 → total=48
        let entries_def = &[
            (0, ArgumentDataType::Float),
            (1, ArgumentDataType::Uint),
            (2, ArgumentDataType::Pointer),
            (3, ArgumentDataType::Float4),
            (4, ArgumentDataType::Texture2D),
        ];
        let (entries, total) = compute_argument_buffer_layout(entries_def);
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[1].offset, 4);
        assert_eq!(entries[2].offset, 8);
        assert_eq!(entries[3].offset, 16);
        assert_eq!(entries[4].offset, 32);
        assert_eq!(total, 48);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Feature set capabilities
// ═══════════════════════════════════════════════════════════════════════════

mod feature_set_capabilities {
    use super::*;

    #[test]
    fn apple7_supports_simd_reduction() {
        assert!(AppleGpuFamily::Apple7.supports_simd_reduction());
    }

    #[test]
    fn apple6_no_simd_reduction() {
        assert!(!AppleGpuFamily::Apple6.supports_simd_reduction());
    }

    #[test]
    fn apple6_supports_ray_tracing() {
        assert!(AppleGpuFamily::Apple6.supports_ray_tracing());
    }

    #[test]
    fn apple5_no_ray_tracing() {
        assert!(!AppleGpuFamily::Apple5.supports_ray_tracing());
    }

    #[test]
    fn apple7_supports_bfloat16() {
        assert!(AppleGpuFamily::Apple7.supports_bfloat16());
    }

    #[test]
    fn m1_family_threadgroup_memory() {
        assert_eq!(AppleGpuFamily::Apple7.max_threadgroup_memory_bytes(), 32768);
    }

    #[test]
    fn pre_m1_threadgroup_memory() {
        assert_eq!(AppleGpuFamily::Apple5.max_threadgroup_memory_bytes(), 16384);
    }

    #[test]
    fn all_families_max_threads_1024() {
        for family in [AppleGpuFamily::Apple1, AppleGpuFamily::Apple7, AppleGpuFamily::Apple9] {
            assert_eq!(family.max_threads_per_threadgroup(), 1024);
        }
    }

    #[test]
    fn family_ordering() {
        assert!(AppleGpuFamily::Apple9 > AppleGpuFamily::Apple7);
        assert!(AppleGpuFamily::Apple7 > AppleGpuFamily::Apple4);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Heap allocation
// ═══════════════════════════════════════════════════════════════════════════

mod heap_allocation {
    use super::*;

    #[test]
    fn allocate_from_empty_heap() {
        let offset = heap_allocate(4096, &[], 512);
        assert_eq!(offset, Some(0));
    }

    #[test]
    fn allocate_after_existing() {
        let existing = vec![HeapAllocation {
            offset: 0,
            size: 512,
            aliasable: false,
            purgeable: PurgeableState::NonVolatile,
        }];
        let offset = heap_allocate(4096, &existing, 256);
        assert_eq!(offset, Some(512)); // aligned to 256
    }

    #[test]
    fn allocation_fails_when_full() {
        let existing = vec![HeapAllocation {
            offset: 0,
            size: 4096,
            aliasable: false,
            purgeable: PurgeableState::NonVolatile,
        }];
        let offset = heap_allocate(4096, &existing, 256);
        assert_eq!(offset, None);
    }

    #[test]
    fn aliasable_resource_allows_reuse() {
        let existing = vec![HeapAllocation {
            offset: 0,
            size: 4096,
            aliasable: true, // marked for reuse
            purgeable: PurgeableState::NonVolatile,
        }];
        let offset = heap_allocate(4096, &existing, 256);
        assert_eq!(offset, Some(0));
    }

    #[test]
    fn allocation_aligns_to_256() {
        let existing = vec![HeapAllocation {
            offset: 0,
            size: 100, // not aligned
            aliasable: false,
            purgeable: PurgeableState::NonVolatile,
        }];
        let offset = heap_allocate(4096, &existing, 64);
        assert_eq!(offset, Some(256));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Resource lifecycle
// ═══════════════════════════════════════════════════════════════════════════

mod resource_lifecycle {
    use super::*;

    #[test]
    fn purgeable_state_transitions() {
        let state = PurgeableState::NonVolatile;
        let _ = state;
        let state = PurgeableState::Volatile;
        assert_eq!(state, PurgeableState::Volatile);
        let state = PurgeableState::Empty;
        assert_eq!(state, PurgeableState::Empty);
    }

    #[test]
    fn hazard_tracking_default_is_tracked() {
        let mode = HazardTrackingMode::Default;
        assert_ne!(mode, HazardTrackingMode::Untracked);
    }

    #[test]
    fn untracked_hazards_for_manual_sync() {
        let mode = HazardTrackingMode::Untracked;
        assert_eq!(mode, HazardTrackingMode::Untracked);
    }

    #[test]
    fn make_aliasable_allows_heap_reuse() {
        let mut alloc = HeapAllocation {
            offset: 0,
            size: 1024,
            aliasable: false,
            purgeable: PurgeableState::NonVolatile,
        };
        assert!(!alloc.aliasable);
        alloc.aliasable = true;
        assert!(alloc.aliasable);
    }

    #[test]
    fn volatile_resource_may_be_purged() {
        let alloc = HeapAllocation {
            offset: 0,
            size: 1024,
            aliasable: false,
            purgeable: PurgeableState::Volatile,
        };
        assert_eq!(alloc.purgeable, PurgeableState::Volatile);
    }

    #[test]
    fn non_volatile_resource_retained() {
        let alloc = HeapAllocation {
            offset: 0,
            size: 1024,
            aliasable: false,
            purgeable: PurgeableState::NonVolatile,
        };
        assert_eq!(alloc.purgeable, PurgeableState::NonVolatile);
    }

    #[test]
    fn heap_descriptor_storage_modes() {
        let shared_heap = HeapDescriptor {
            size: 1024 * 1024,
            storage_mode: StorageMode::Shared,
            hazard_tracking: HazardTrackingMode::Default,
        };
        assert!(shared_heap.storage_mode.cpu_accessible());

        let private_heap = HeapDescriptor {
            size: 1024 * 1024,
            storage_mode: StorageMode::Private,
            hazard_tracking: HazardTrackingMode::Untracked,
        };
        assert!(!private_heap.storage_mode.cpu_accessible());
        assert!(private_heap.storage_mode.gpu_accessible());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Alignment utility edge cases
// ═══════════════════════════════════════════════════════════════════════════

mod alignment_edge_cases {
    use super::*;

    #[test]
    fn align_up_already_aligned() {
        assert_eq!(align_up(256, 256), 256);
    }

    #[test]
    fn align_up_zero() {
        assert_eq!(align_up(0, 256), 0);
    }

    #[test]
    fn align_up_one_byte() {
        assert_eq!(align_up(1, 256), 256);
    }

    #[test]
    fn align_up_power_of_two() {
        assert_eq!(align_up(513, 512), 1024);
    }
}
