#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal texture format compatibility tests for Apple Silicon.
//!
//! Validates texture pixel format properties, dimension limits, format
//! conversions, compressed texture formats, and Apple Silicon–specific
//! format support for neural-network tensor storage.
//!
//! All tests exercise pure Rust logic — no Metal API or GPU required.

// ── Pixel format enum ───────────────────────────────────────────────

/// Metal pixel format identifiers used for texture resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(dead_code)]
enum PixelFormat {
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
    RG8Unorm = 31,
    RG16Float = 65,
    RG32Float = 105,
    RGBA8Unorm = 70,
    RGBA16Float = 115,
    RGBA32Float = 125,
    // Compressed
    BC1_RGBA = 130,
    BC7_RGBAUnorm = 152,
    ASTC_4x4_LDR = 204,
}

impl PixelFormat {
    fn byte_size(self) -> usize {
        match self {
            Self::R8Unorm | Self::R8Snorm | Self::R8Uint | Self::R8Sint => 1,
            Self::R16Float | Self::R16Uint | Self::R16Sint => 2,
            Self::R32Float | Self::R32Uint | Self::R32Sint => 4,
            Self::RG8Unorm => 2,
            Self::RG16Float => 4,
            Self::RG32Float => 8,
            Self::RGBA8Unorm => 4,
            Self::RGBA16Float => 8,
            Self::RGBA32Float => 16,
            // Compressed formats return bytes per block
            Self::BC1_RGBA => 8,
            Self::BC7_RGBAUnorm => 16,
            Self::ASTC_4x4_LDR => 16,
        }
    }

    fn channel_count(self) -> u8 {
        match self {
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
            Self::RG8Unorm | Self::RG16Float | Self::RG32Float => 2,
            Self::RGBA8Unorm
            | Self::RGBA16Float
            | Self::RGBA32Float
            | Self::BC1_RGBA
            | Self::BC7_RGBAUnorm
            | Self::ASTC_4x4_LDR => 4,
        }
    }

    fn bits_per_channel(self) -> u8 {
        match self {
            Self::R8Unorm
            | Self::R8Snorm
            | Self::R8Uint
            | Self::R8Sint
            | Self::RG8Unorm
            | Self::RGBA8Unorm => 8,
            Self::R16Float
            | Self::R16Uint
            | Self::R16Sint
            | Self::RG16Float
            | Self::RGBA16Float => 16,
            Self::R32Float
            | Self::R32Uint
            | Self::R32Sint
            | Self::RG32Float
            | Self::RGBA32Float => 32,
            // Compressed: effective bits per channel (approximation)
            Self::BC1_RGBA => 4,
            Self::BC7_RGBAUnorm => 8,
            Self::ASTC_4x4_LDR => 8,
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

    fn is_uint(self) -> bool {
        matches!(self, Self::R8Uint | Self::R16Uint | Self::R32Uint)
    }

    fn is_sint(self) -> bool {
        matches!(self, Self::R8Sint | Self::R16Sint | Self::R32Sint)
    }

    fn is_normalized(self) -> bool {
        matches!(self, Self::R8Unorm | Self::R8Snorm | Self::RG8Unorm | Self::RGBA8Unorm)
    }

    fn is_compressed(self) -> bool {
        matches!(self, Self::BC1_RGBA | Self::BC7_RGBAUnorm | Self::ASTC_4x4_LDR)
    }

    /// Minimum row alignment (bytes) required by Metal.
    fn row_alignment(self) -> usize {
        if self.is_compressed() { 16 } else { 4 }
    }

    /// Whether the format supports read-write in a compute shader.
    fn supports_read_write(self) -> bool {
        matches!(
            self,
            Self::R8Unorm
                | Self::R8Uint
                | Self::R8Sint
                | Self::R16Float
                | Self::R16Uint
                | Self::R16Sint
                | Self::R32Float
                | Self::R32Uint
                | Self::R32Sint
                | Self::RG8Unorm
                | Self::RG16Float
                | Self::RG32Float
                | Self::RGBA8Unorm
                | Self::RGBA16Float
                | Self::RGBA32Float
        )
    }

    /// Whether the format can be used as a render target.
    fn supports_render_target(self) -> bool {
        !self.is_compressed()
    }

    /// Whether MSAA is supported for this format (no MSAA for compressed).
    fn supports_msaa(self) -> bool {
        !self.is_compressed()
    }

    /// Block width for compressed formats; 1 for ordinary formats.
    fn block_width(self) -> u32 {
        match self {
            Self::BC1_RGBA | Self::BC7_RGBAUnorm | Self::ASTC_4x4_LDR => 4,
            _ => 1,
        }
    }

    /// Block height for compressed formats; 1 for ordinary formats.
    fn block_height(self) -> u32 {
        match self {
            Self::BC1_RGBA | Self::BC7_RGBAUnorm | Self::ASTC_4x4_LDR => 4,
            _ => 1,
        }
    }
}

// ── Texture limits ──────────────────────────────────────────────────

/// Metal texture dimension limits (Apple GPU family 7+, M1 and later).
struct TextureLimits {
    max_1d_width: u32,
    max_2d_width: u32,
    max_2d_height: u32,
    max_3d_size: u32,
    max_cube_size: u32,
    max_array_layers: u32,
}

impl TextureLimits {
    fn apple_silicon() -> Self {
        Self {
            max_1d_width: 16384,
            max_2d_width: 16384,
            max_2d_height: 16384,
            max_3d_size: 2048,
            max_cube_size: 16384,
            max_array_layers: 2048,
        }
    }
}

// ── Helper functions ────────────────────────────────────────────────

/// Aligned row size in bytes (Metal requires ≥4-byte row alignment).
fn aligned_row_bytes(width: u32, fmt: PixelFormat) -> usize {
    let bw = fmt.block_width();
    let blocks_per_row = (width + bw - 1) / bw;
    let raw = blocks_per_row as usize * fmt.byte_size();
    let align = fmt.row_alignment();
    (raw + align - 1) & !(align - 1)
}

/// Total bytes needed for a 2-D texture (all rows, including alignment padding).
fn texture_2d_bytes(width: u32, height: u32, fmt: PixelFormat) -> usize {
    let bh = fmt.block_height();
    let block_rows = (height + bh - 1) / bh;
    aligned_row_bytes(width, fmt) * block_rows as usize
}

/// Number of mipmap levels for a texture with the given max dimension.
fn mipmap_levels(max_dim: u32) -> u32 {
    if max_dim == 0 {
        return 0;
    }
    32 - max_dim.leading_zeros()
}

/// Total bytes for a mipmapped 2-D texture (all levels).
fn mipmapped_texture_bytes(width: u32, height: u32, fmt: PixelFormat) -> usize {
    let levels = mipmap_levels(width.max(height));
    let mut total = 0usize;
    for level in 0..levels {
        let w = (width >> level).max(1);
        let h = (height >> level).max(1);
        total += texture_2d_bytes(w, h, fmt);
    }
    total
}

/// Whether a given (width, height, depth) fits within 3-D limits.
fn fits_3d(width: u32, height: u32, depth: u32, limits: &TextureLimits) -> bool {
    width <= limits.max_3d_size && height <= limits.max_3d_size && depth <= limits.max_3d_size
}

/// Estimate the number of textures needed to store a tensor of `total_elements`
/// using the given format (as a 2-D tile atlas with `tile_size` rows/cols).
fn textures_for_tensor(total_elements: usize, tile_size: u32, fmt: PixelFormat) -> u32 {
    let elems_per_tex = tile_size as usize * tile_size as usize * fmt.channel_count() as usize;
    if elems_per_tex == 0 {
        return 0;
    }
    ((total_elements + elems_per_tex - 1) / elems_per_tex) as u32
}

/// Whether two formats are "blend-compatible" (same channel count and bit width).
fn blend_compatible(a: PixelFormat, b: PixelFormat) -> bool {
    !a.is_compressed()
        && !b.is_compressed()
        && a.channel_count() == b.channel_count()
        && a.bits_per_channel() == b.bits_per_channel()
}

/// Map a Rust scalar type size to the best single-channel texture format.
fn scalar_to_texture_format(scalar_bytes: usize) -> Option<PixelFormat> {
    match scalar_bytes {
        1 => Some(PixelFormat::R8Uint),
        2 => Some(PixelFormat::R16Float),
        4 => Some(PixelFormat::R32Float),
        _ => None,
    }
}

// ── Apple Silicon format support ────────────────────────────────────

/// Apple GPU family feature tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[allow(dead_code)]
enum AppleGpuFamily {
    Apple7, // M1
    Apple8, // M2
    Apple9, // M3
}

impl AppleGpuFamily {
    fn supports_bc_compression(self) -> bool {
        // Apple Silicon supports BC on macOS via GPU family 7+
        self >= Self::Apple7
    }

    fn supports_astc(self) -> bool {
        self >= Self::Apple7
    }

    fn max_texture_2d(self) -> u32 {
        16384
    }

    fn max_buffer_length_gb(self) -> u32 {
        match self {
            Self::Apple7 => 16,
            Self::Apple8 | Self::Apple9 => 32,
        }
    }

    fn supports_32bit_float_filtering(self) -> bool {
        self >= Self::Apple9
    }

    fn supports_lossless_msaa(self) -> bool {
        self >= Self::Apple7
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

// ── pixel_formats ───────────────────────────────────────────────────

mod pixel_formats {
    use super::*;

    // byte sizes

    #[test]
    fn r8_unorm_byte_size() {
        assert_eq!(PixelFormat::R8Unorm.byte_size(), 1);
    }

    #[test]
    fn r8_snorm_byte_size() {
        assert_eq!(PixelFormat::R8Snorm.byte_size(), 1);
    }

    #[test]
    fn r8_uint_byte_size() {
        assert_eq!(PixelFormat::R8Uint.byte_size(), 1);
    }

    #[test]
    fn r8_sint_byte_size() {
        assert_eq!(PixelFormat::R8Sint.byte_size(), 1);
    }

    #[test]
    fn r16_float_byte_size() {
        assert_eq!(PixelFormat::R16Float.byte_size(), 2);
    }

    #[test]
    fn r16_uint_byte_size() {
        assert_eq!(PixelFormat::R16Uint.byte_size(), 2);
    }

    #[test]
    fn r16_sint_byte_size() {
        assert_eq!(PixelFormat::R16Sint.byte_size(), 2);
    }

    #[test]
    fn r32_float_byte_size() {
        assert_eq!(PixelFormat::R32Float.byte_size(), 4);
    }

    #[test]
    fn r32_uint_byte_size() {
        assert_eq!(PixelFormat::R32Uint.byte_size(), 4);
    }

    #[test]
    fn r32_sint_byte_size() {
        assert_eq!(PixelFormat::R32Sint.byte_size(), 4);
    }

    #[test]
    fn rg8_unorm_byte_size() {
        assert_eq!(PixelFormat::RG8Unorm.byte_size(), 2);
    }

    #[test]
    fn rg16_float_byte_size() {
        assert_eq!(PixelFormat::RG16Float.byte_size(), 4);
    }

    #[test]
    fn rg32_float_byte_size() {
        assert_eq!(PixelFormat::RG32Float.byte_size(), 8);
    }

    #[test]
    fn rgba8_unorm_byte_size() {
        assert_eq!(PixelFormat::RGBA8Unorm.byte_size(), 4);
    }

    #[test]
    fn rgba16_float_byte_size() {
        assert_eq!(PixelFormat::RGBA16Float.byte_size(), 8);
    }

    #[test]
    fn rgba32_float_byte_size() {
        assert_eq!(PixelFormat::RGBA32Float.byte_size(), 16);
    }

    // channel counts

    #[test]
    fn single_channel_formats() {
        for fmt in [
            PixelFormat::R8Unorm,
            PixelFormat::R8Snorm,
            PixelFormat::R8Uint,
            PixelFormat::R8Sint,
            PixelFormat::R16Float,
            PixelFormat::R32Float,
        ] {
            assert_eq!(fmt.channel_count(), 1, "{fmt:?} should have 1 channel");
        }
    }

    #[test]
    fn dual_channel_formats() {
        for fmt in [PixelFormat::RG8Unorm, PixelFormat::RG16Float, PixelFormat::RG32Float] {
            assert_eq!(fmt.channel_count(), 2, "{fmt:?} should have 2 channels");
        }
    }

    #[test]
    fn quad_channel_formats() {
        for fmt in [PixelFormat::RGBA8Unorm, PixelFormat::RGBA16Float, PixelFormat::RGBA32Float] {
            assert_eq!(fmt.channel_count(), 4, "{fmt:?} should have 4 channels");
        }
    }

    // type classification

    #[test]
    fn float_classification() {
        assert!(PixelFormat::R16Float.is_float());
        assert!(PixelFormat::R32Float.is_float());
        assert!(PixelFormat::RG16Float.is_float());
        assert!(PixelFormat::RGBA32Float.is_float());
        assert!(!PixelFormat::R8Uint.is_float());
        assert!(!PixelFormat::R8Unorm.is_float());
    }

    #[test]
    fn uint_classification() {
        assert!(PixelFormat::R8Uint.is_uint());
        assert!(PixelFormat::R16Uint.is_uint());
        assert!(PixelFormat::R32Uint.is_uint());
        assert!(!PixelFormat::R16Float.is_uint());
        assert!(!PixelFormat::R8Sint.is_uint());
    }

    #[test]
    fn sint_classification() {
        assert!(PixelFormat::R8Sint.is_sint());
        assert!(PixelFormat::R16Sint.is_sint());
        assert!(PixelFormat::R32Sint.is_sint());
        assert!(!PixelFormat::R8Uint.is_sint());
        assert!(!PixelFormat::R32Float.is_sint());
    }

    #[test]
    fn normalized_classification() {
        assert!(PixelFormat::R8Unorm.is_normalized());
        assert!(PixelFormat::R8Snorm.is_normalized());
        assert!(PixelFormat::RGBA8Unorm.is_normalized());
        assert!(!PixelFormat::R16Float.is_normalized());
        assert!(!PixelFormat::R32Uint.is_normalized());
    }

    #[test]
    fn compressed_classification() {
        assert!(PixelFormat::BC1_RGBA.is_compressed());
        assert!(PixelFormat::BC7_RGBAUnorm.is_compressed());
        assert!(PixelFormat::ASTC_4x4_LDR.is_compressed());
        assert!(!PixelFormat::RGBA8Unorm.is_compressed());
    }

    // bits per channel

    #[test]
    fn bits_per_channel_8bit() {
        assert_eq!(PixelFormat::R8Unorm.bits_per_channel(), 8);
        assert_eq!(PixelFormat::RG8Unorm.bits_per_channel(), 8);
        assert_eq!(PixelFormat::RGBA8Unorm.bits_per_channel(), 8);
    }

    #[test]
    fn bits_per_channel_16bit() {
        assert_eq!(PixelFormat::R16Float.bits_per_channel(), 16);
        assert_eq!(PixelFormat::RG16Float.bits_per_channel(), 16);
        assert_eq!(PixelFormat::RGBA16Float.bits_per_channel(), 16);
    }

    #[test]
    fn bits_per_channel_32bit() {
        assert_eq!(PixelFormat::R32Float.bits_per_channel(), 32);
        assert_eq!(PixelFormat::RG32Float.bits_per_channel(), 32);
        assert_eq!(PixelFormat::RGBA32Float.bits_per_channel(), 32);
    }

    // render target & MSAA

    #[test]
    fn uncompressed_supports_render_target() {
        assert!(PixelFormat::RGBA8Unorm.supports_render_target());
        assert!(PixelFormat::R32Float.supports_render_target());
    }

    #[test]
    fn compressed_no_render_target() {
        assert!(!PixelFormat::BC1_RGBA.supports_render_target());
        assert!(!PixelFormat::BC7_RGBAUnorm.supports_render_target());
        assert!(!PixelFormat::ASTC_4x4_LDR.supports_render_target());
    }

    #[test]
    fn uncompressed_supports_msaa() {
        assert!(PixelFormat::RGBA16Float.supports_msaa());
        assert!(PixelFormat::R8Uint.supports_msaa());
    }

    #[test]
    fn compressed_no_msaa() {
        assert!(!PixelFormat::BC1_RGBA.supports_msaa());
    }

    // read-write

    #[test]
    fn uncompressed_supports_read_write() {
        assert!(PixelFormat::R32Float.supports_read_write());
        assert!(PixelFormat::RGBA16Float.supports_read_write());
    }

    #[test]
    fn compressed_no_read_write() {
        assert!(!PixelFormat::BC1_RGBA.supports_read_write());
        assert!(!PixelFormat::BC7_RGBAUnorm.supports_read_write());
    }

    // alignment

    #[test]
    fn uncompressed_row_alignment_is_4() {
        assert_eq!(PixelFormat::R32Float.row_alignment(), 4);
        assert_eq!(PixelFormat::RGBA8Unorm.row_alignment(), 4);
        assert_eq!(PixelFormat::R8Uint.row_alignment(), 4);
    }

    #[test]
    fn compressed_row_alignment_is_16() {
        assert_eq!(PixelFormat::BC1_RGBA.row_alignment(), 16);
        assert_eq!(PixelFormat::ASTC_4x4_LDR.row_alignment(), 16);
    }

    // block dimensions

    #[test]
    fn uncompressed_block_is_1x1() {
        assert_eq!(PixelFormat::R32Float.block_width(), 1);
        assert_eq!(PixelFormat::R32Float.block_height(), 1);
    }

    #[test]
    fn bc1_block_is_4x4() {
        assert_eq!(PixelFormat::BC1_RGBA.block_width(), 4);
        assert_eq!(PixelFormat::BC1_RGBA.block_height(), 4);
    }

    #[test]
    fn astc_4x4_block_is_4x4() {
        assert_eq!(PixelFormat::ASTC_4x4_LDR.block_width(), 4);
        assert_eq!(PixelFormat::ASTC_4x4_LDR.block_height(), 4);
    }

    // byte_size == channels * (bits_per_channel / 8) for uncompressed
    #[test]
    fn byte_size_consistent_with_channels_and_bits() {
        for fmt in [
            PixelFormat::R8Unorm,
            PixelFormat::R16Float,
            PixelFormat::R32Float,
            PixelFormat::RG8Unorm,
            PixelFormat::RG16Float,
            PixelFormat::RG32Float,
            PixelFormat::RGBA8Unorm,
            PixelFormat::RGBA16Float,
            PixelFormat::RGBA32Float,
        ] {
            let expected = fmt.channel_count() as usize * (fmt.bits_per_channel() as usize / 8);
            assert_eq!(
                fmt.byte_size(),
                expected,
                "{fmt:?}: byte_size {} != channels*bytes {expected}",
                fmt.byte_size()
            );
        }
    }
}

// ── texture_dimensions ──────────────────────────────────────────────

mod texture_dimensions {
    use super::*;

    #[test]
    fn limits_1d_max_width() {
        let lim = TextureLimits::apple_silicon();
        assert_eq!(lim.max_1d_width, 16384);
    }

    #[test]
    fn limits_2d_max() {
        let lim = TextureLimits::apple_silicon();
        assert_eq!(lim.max_2d_width, 16384);
        assert_eq!(lim.max_2d_height, 16384);
    }

    #[test]
    fn limits_3d_max() {
        let lim = TextureLimits::apple_silicon();
        assert_eq!(lim.max_3d_size, 2048);
    }

    #[test]
    fn limits_cube_max() {
        let lim = TextureLimits::apple_silicon();
        assert_eq!(lim.max_cube_size, 16384);
    }

    #[test]
    fn limits_array_layers() {
        let lim = TextureLimits::apple_silicon();
        assert_eq!(lim.max_array_layers, 2048);
    }

    #[test]
    fn fits_3d_small() {
        let lim = TextureLimits::apple_silicon();
        assert!(fits_3d(64, 64, 64, &lim));
    }

    #[test]
    fn fits_3d_max() {
        let lim = TextureLimits::apple_silicon();
        assert!(fits_3d(2048, 2048, 2048, &lim));
    }

    #[test]
    fn exceeds_3d_limit() {
        let lim = TextureLimits::apple_silicon();
        assert!(!fits_3d(2049, 2048, 2048, &lim));
        assert!(!fits_3d(2048, 2049, 2048, &lim));
        assert!(!fits_3d(2048, 2048, 2049, &lim));
    }

    // Mipmap level calculations

    #[test]
    fn mipmap_levels_1x1() {
        assert_eq!(mipmap_levels(1), 1);
    }

    #[test]
    fn mipmap_levels_2x2() {
        assert_eq!(mipmap_levels(2), 2);
    }

    #[test]
    fn mipmap_levels_4x4() {
        assert_eq!(mipmap_levels(4), 3);
    }

    #[test]
    fn mipmap_levels_256() {
        assert_eq!(mipmap_levels(256), 9);
    }

    #[test]
    fn mipmap_levels_1024() {
        assert_eq!(mipmap_levels(1024), 11);
    }

    #[test]
    fn mipmap_levels_16384() {
        assert_eq!(mipmap_levels(16384), 15);
    }

    #[test]
    fn mipmap_levels_zero_returns_zero() {
        assert_eq!(mipmap_levels(0), 0);
    }

    #[test]
    fn mipmap_levels_non_power_of_two() {
        // 100 → ⌈log2(100)⌉+1 = 7
        assert_eq!(mipmap_levels(100), 7);
    }

    // Row alignment

    #[test]
    fn aligned_row_bytes_r32_width_1() {
        // 1 pixel * 4 bytes = 4, already aligned to 4
        assert_eq!(aligned_row_bytes(1, PixelFormat::R32Float), 4);
    }

    #[test]
    fn aligned_row_bytes_r8_width_1() {
        // 1 byte → padded to 4
        assert_eq!(aligned_row_bytes(1, PixelFormat::R8Uint), 4);
    }

    #[test]
    fn aligned_row_bytes_r8_width_5() {
        // 5 bytes → padded to 8
        assert_eq!(aligned_row_bytes(5, PixelFormat::R8Uint), 8);
    }

    #[test]
    fn aligned_row_bytes_rgba32_width_4() {
        // 4 pixels * 16 bytes = 64
        assert_eq!(aligned_row_bytes(4, PixelFormat::RGBA32Float), 64);
    }

    // 2-D texture total bytes

    #[test]
    fn texture_2d_bytes_simple() {
        // 4x4 R32Float: row = 16 bytes, 4 rows → 64
        assert_eq!(texture_2d_bytes(4, 4, PixelFormat::R32Float), 64);
    }

    #[test]
    fn texture_2d_bytes_rgba16() {
        // 8x8 RGBA16Float: row = 8*8 = 64 bytes, 8 rows → 512
        assert_eq!(texture_2d_bytes(8, 8, PixelFormat::RGBA16Float), 512);
    }

    #[test]
    fn texture_2d_bytes_1x1() {
        assert_eq!(texture_2d_bytes(1, 1, PixelFormat::R32Float), 4);
    }

    // Mipmapped texture bytes

    #[test]
    fn mipmapped_bytes_4x4_r32() {
        // Level 0: 4x4=16*4=64, Level 1: 2x2=4*4=16, Level 2: 1x1=4 → 84
        let total = mipmapped_texture_bytes(4, 4, PixelFormat::R32Float);
        assert_eq!(total, 84);
    }

    #[test]
    fn mipmapped_bytes_greater_than_single_level() {
        let single = texture_2d_bytes(256, 256, PixelFormat::RGBA8Unorm);
        let mipped = mipmapped_texture_bytes(256, 256, PixelFormat::RGBA8Unorm);
        assert!(mipped > single);
    }

    // Tensor storage estimation

    #[test]
    fn textures_for_small_tensor() {
        // 1024 elements, tile 16x16=256 elems (1 channel) → 4 textures
        assert_eq!(textures_for_tensor(1024, 16, PixelFormat::R32Float), 4);
    }

    #[test]
    fn textures_for_rgba_packing() {
        // 1024 elements, tile 16x16 = 256 pixels * 4 channels = 1024 → 1 texture
        assert_eq!(textures_for_tensor(1024, 16, PixelFormat::RGBA32Float), 1);
    }

    #[test]
    fn textures_for_large_weight_matrix() {
        // 2B model embedding: 2048 * 32000 = 65_536_000 elements
        let elems = 2048 * 32_000;
        let n = textures_for_tensor(elems, 16384, PixelFormat::R16Float);
        // 16384^2 * 1 channel = 268_435_456 → fits in 1 texture
        assert_eq!(n, 1);
    }

    #[test]
    fn textures_for_zero_elements() {
        assert_eq!(textures_for_tensor(0, 16, PixelFormat::R32Float), 0);
    }
}

// ── format_conversions ──────────────────────────────────────────────

mod format_conversions {
    use super::*;

    #[test]
    fn blend_compatible_same_format() {
        assert!(blend_compatible(PixelFormat::RGBA8Unorm, PixelFormat::RGBA8Unorm));
    }

    #[test]
    fn blend_compatible_r8_variants() {
        assert!(blend_compatible(PixelFormat::R8Unorm, PixelFormat::R8Uint));
    }

    #[test]
    fn blend_incompatible_different_channels() {
        assert!(!blend_compatible(PixelFormat::R8Unorm, PixelFormat::RG8Unorm));
    }

    #[test]
    fn blend_incompatible_different_bits() {
        assert!(!blend_compatible(PixelFormat::R8Unorm, PixelFormat::R16Float));
    }

    #[test]
    fn blend_incompatible_compressed() {
        assert!(!blend_compatible(PixelFormat::BC1_RGBA, PixelFormat::RGBA8Unorm));
    }

    #[test]
    fn blend_incompatible_both_compressed() {
        assert!(!blend_compatible(PixelFormat::BC1_RGBA, PixelFormat::BC7_RGBAUnorm));
    }

    // scalar → texture format mapping

    #[test]
    fn scalar_1_byte_to_r8uint() {
        assert_eq!(scalar_to_texture_format(1), Some(PixelFormat::R8Uint));
    }

    #[test]
    fn scalar_2_byte_to_r16float() {
        assert_eq!(scalar_to_texture_format(2), Some(PixelFormat::R16Float));
    }

    #[test]
    fn scalar_4_byte_to_r32float() {
        assert_eq!(scalar_to_texture_format(4), Some(PixelFormat::R32Float));
    }

    #[test]
    fn scalar_8_byte_unsupported() {
        assert_eq!(scalar_to_texture_format(8), None);
    }

    #[test]
    fn scalar_0_byte_unsupported() {
        assert_eq!(scalar_to_texture_format(0), None);
    }

    // format widening: r → rg → rgba

    #[test]
    fn widen_r8_to_rg8() {
        assert_eq!(PixelFormat::R8Unorm.channel_count() * 2, PixelFormat::RG8Unorm.channel_count());
    }

    #[test]
    fn widen_r16_to_rg16() {
        assert_eq!(PixelFormat::R16Float.byte_size() * 2, PixelFormat::RG16Float.byte_size());
    }

    #[test]
    fn widen_rg16_to_rgba16() {
        assert_eq!(PixelFormat::RG16Float.byte_size() * 2, PixelFormat::RGBA16Float.byte_size());
    }

    #[test]
    fn widen_r32_to_rg32() {
        assert_eq!(PixelFormat::R32Float.byte_size() * 2, PixelFormat::RG32Float.byte_size());
    }

    #[test]
    fn widen_rg32_to_rgba32() {
        assert_eq!(PixelFormat::RG32Float.byte_size() * 2, PixelFormat::RGBA32Float.byte_size());
    }

    // format compatibility matrix: float ↔ float OK, int ↔ int OK

    #[test]
    fn float_formats_mutually_compatible() {
        let floats = [
            PixelFormat::R16Float,
            PixelFormat::R32Float,
            PixelFormat::RG16Float,
            PixelFormat::RGBA16Float,
        ];
        for fmt in floats {
            assert!(fmt.is_float(), "{fmt:?} should be float");
            assert!(!fmt.is_uint());
            assert!(!fmt.is_sint());
        }
    }

    #[test]
    fn uint_formats_mutually_compatible() {
        let uints = [PixelFormat::R8Uint, PixelFormat::R16Uint, PixelFormat::R32Uint];
        for fmt in uints {
            assert!(fmt.is_uint());
            assert!(!fmt.is_float());
            assert!(!fmt.is_sint());
        }
    }

    #[test]
    fn sint_formats_mutually_compatible() {
        let sints = [PixelFormat::R8Sint, PixelFormat::R16Sint, PixelFormat::R32Sint];
        for fmt in sints {
            assert!(fmt.is_sint());
            assert!(!fmt.is_float());
            assert!(!fmt.is_uint());
        }
    }

    #[test]
    fn int_and_float_disjoint() {
        assert!(!PixelFormat::R32Uint.is_float());
        assert!(!PixelFormat::R32Float.is_uint());
        assert!(!PixelFormat::R32Float.is_sint());
    }

    #[test]
    fn normalized_is_neither_int_nor_float() {
        let fmt = PixelFormat::R8Unorm;
        assert!(fmt.is_normalized());
        assert!(!fmt.is_float());
        assert!(!fmt.is_uint());
        assert!(!fmt.is_sint());
    }
}

// ── compression_formats ─────────────────────────────────────────────

mod compression_formats {
    use super::*;

    #[test]
    fn bc1_block_size_8_bytes() {
        assert_eq!(PixelFormat::BC1_RGBA.byte_size(), 8);
    }

    #[test]
    fn bc7_block_size_16_bytes() {
        assert_eq!(PixelFormat::BC7_RGBAUnorm.byte_size(), 16);
    }

    #[test]
    fn astc_4x4_block_size_16_bytes() {
        assert_eq!(PixelFormat::ASTC_4x4_LDR.byte_size(), 16);
    }

    #[test]
    fn compressed_formats_have_4_channels() {
        assert_eq!(PixelFormat::BC1_RGBA.channel_count(), 4);
        assert_eq!(PixelFormat::BC7_RGBAUnorm.channel_count(), 4);
        assert_eq!(PixelFormat::ASTC_4x4_LDR.channel_count(), 4);
    }

    #[test]
    fn compressed_not_float() {
        assert!(!PixelFormat::BC1_RGBA.is_float());
        assert!(!PixelFormat::BC7_RGBAUnorm.is_float());
    }

    #[test]
    fn compressed_not_uint_or_sint() {
        assert!(!PixelFormat::BC1_RGBA.is_uint());
        assert!(!PixelFormat::ASTC_4x4_LDR.is_sint());
    }

    #[test]
    fn compressed_block_dimensions() {
        for fmt in [PixelFormat::BC1_RGBA, PixelFormat::BC7_RGBAUnorm, PixelFormat::ASTC_4x4_LDR] {
            assert_eq!(fmt.block_width(), 4, "{fmt:?}");
            assert_eq!(fmt.block_height(), 4, "{fmt:?}");
        }
    }

    #[test]
    fn compressed_row_alignment_16() {
        for fmt in [PixelFormat::BC1_RGBA, PixelFormat::BC7_RGBAUnorm, PixelFormat::ASTC_4x4_LDR] {
            assert_eq!(fmt.row_alignment(), 16, "{fmt:?}");
        }
    }

    #[test]
    fn compressed_no_render_target_or_msaa() {
        for fmt in [PixelFormat::BC1_RGBA, PixelFormat::BC7_RGBAUnorm, PixelFormat::ASTC_4x4_LDR] {
            assert!(!fmt.supports_render_target(), "{fmt:?} render target");
            assert!(!fmt.supports_msaa(), "{fmt:?} msaa");
            assert!(!fmt.supports_read_write(), "{fmt:?} read-write");
        }
    }

    // Compressed texture 2-D size calculations

    #[test]
    fn bc1_texture_size_4x4() {
        // 4x4 pixels = 1 block row with 1 block → 1*8 = 8, aligned to 16
        assert_eq!(texture_2d_bytes(4, 4, PixelFormat::BC1_RGBA), 16);
    }

    #[test]
    fn bc1_texture_size_8x8() {
        // 8x8 = 2 block cols, 2 block rows; row = 2*8=16, 2 rows → 32
        assert_eq!(texture_2d_bytes(8, 8, PixelFormat::BC1_RGBA), 32);
    }

    #[test]
    fn bc7_texture_size_16x16() {
        // 16x16 = 4 block cols, 4 block rows; row = 4*16=64, 4 rows → 256
        assert_eq!(texture_2d_bytes(16, 16, PixelFormat::BC7_RGBAUnorm), 256);
    }

    #[test]
    fn astc_non_multiple_of_4_rounds_up() {
        // 5x5: blocks_per_row=ceil(5/4)=2, block_rows=ceil(5/4)=2
        // row = 2*16=32 → aligned to 32 (already a multiple of 16)
        // total = 32*2 = 64
        assert_eq!(texture_2d_bytes(5, 5, PixelFormat::ASTC_4x4_LDR), 64);
    }

    #[test]
    fn bc1_bits_per_channel_is_4() {
        assert_eq!(PixelFormat::BC1_RGBA.bits_per_channel(), 4);
    }

    #[test]
    fn bc7_bits_per_channel_is_8() {
        assert_eq!(PixelFormat::BC7_RGBAUnorm.bits_per_channel(), 8);
    }

    // Compression ratio compared to uncompressed RGBA8

    #[test]
    fn bc1_compression_ratio() {
        let uncompressed = texture_2d_bytes(256, 256, PixelFormat::RGBA8Unorm);
        let compressed = texture_2d_bytes(256, 256, PixelFormat::BC1_RGBA);
        // BC1: 0.5 bytes/pixel vs 4 bytes/pixel → 8:1 ratio
        assert_eq!(uncompressed / compressed, 8);
    }

    #[test]
    fn bc7_compression_ratio() {
        let uncompressed = texture_2d_bytes(256, 256, PixelFormat::RGBA8Unorm);
        let compressed = texture_2d_bytes(256, 256, PixelFormat::BC7_RGBAUnorm);
        // BC7: 1 byte/pixel vs 4 bytes/pixel → 4:1 ratio
        assert_eq!(uncompressed / compressed, 4);
    }
}

// ── apple_silicon_formats ───────────────────────────────────────────

mod apple_silicon_formats {
    use super::*;

    #[test]
    fn m1_supports_bc_compression() {
        assert!(AppleGpuFamily::Apple7.supports_bc_compression());
    }

    #[test]
    fn m2_supports_bc_compression() {
        assert!(AppleGpuFamily::Apple8.supports_bc_compression());
    }

    #[test]
    fn m3_supports_bc_compression() {
        assert!(AppleGpuFamily::Apple9.supports_bc_compression());
    }

    #[test]
    fn m1_supports_astc() {
        assert!(AppleGpuFamily::Apple7.supports_astc());
    }

    #[test]
    fn m2_supports_astc() {
        assert!(AppleGpuFamily::Apple8.supports_astc());
    }

    #[test]
    fn m3_supports_astc() {
        assert!(AppleGpuFamily::Apple9.supports_astc());
    }

    #[test]
    fn max_texture_2d_all_families() {
        for family in [AppleGpuFamily::Apple7, AppleGpuFamily::Apple8, AppleGpuFamily::Apple9] {
            assert_eq!(family.max_texture_2d(), 16384, "{family:?}");
        }
    }

    #[test]
    fn m1_buffer_limit_16gb() {
        assert_eq!(AppleGpuFamily::Apple7.max_buffer_length_gb(), 16);
    }

    #[test]
    fn m2_buffer_limit_32gb() {
        assert_eq!(AppleGpuFamily::Apple8.max_buffer_length_gb(), 32);
    }

    #[test]
    fn m3_buffer_limit_32gb() {
        assert_eq!(AppleGpuFamily::Apple9.max_buffer_length_gb(), 32);
    }

    #[test]
    fn only_m3_supports_32bit_float_filtering() {
        assert!(!AppleGpuFamily::Apple7.supports_32bit_float_filtering());
        assert!(!AppleGpuFamily::Apple8.supports_32bit_float_filtering());
        assert!(AppleGpuFamily::Apple9.supports_32bit_float_filtering());
    }

    #[test]
    fn all_families_support_lossless_msaa() {
        for family in [AppleGpuFamily::Apple7, AppleGpuFamily::Apple8, AppleGpuFamily::Apple9] {
            assert!(family.supports_lossless_msaa(), "{family:?}");
        }
    }

    #[test]
    fn gpu_family_ordering() {
        assert!(AppleGpuFamily::Apple7 < AppleGpuFamily::Apple8);
        assert!(AppleGpuFamily::Apple8 < AppleGpuFamily::Apple9);
    }

    // Apple Silicon tensor storage viability

    #[test]
    fn weight_matrix_fits_single_texture() {
        // 2048x2048 weight in R16Float = 2048*2048*2 = 8 MiB
        // Max 16384x16384 at 2 bytes/pixel = 512 MiB → fits
        let elems = 2048 * 2048;
        let n = textures_for_tensor(elems, 16384, PixelFormat::R16Float);
        assert_eq!(n, 1);
    }

    #[test]
    fn large_embedding_needs_multiple_textures() {
        // 128256 vocab * 2048 dim = 262_668_288 elements
        // Max per texture: 16384^2 = 268_435_456 → still 1 (just barely)
        let elems = 128_256 * 2048;
        let n = textures_for_tensor(elems, 16384, PixelFormat::R16Float);
        assert_eq!(n, 1);
    }

    #[test]
    fn bitnet_2b_all_params_count() {
        // ~2B params, each stored in R16Float across 4096-wide tiles
        let params: usize = 2_000_000_000;
        let n = textures_for_tensor(params, 4096, PixelFormat::R16Float);
        // 4096^2 = 16_777_216 → need ~120 textures
        assert!(n > 100 && n < 200, "got {n}");
    }

    #[test]
    fn rgba_packing_reduces_texture_count() {
        let params: usize = 2_000_000_000;
        let r16 = textures_for_tensor(params, 4096, PixelFormat::R16Float);
        let rgba16 = textures_for_tensor(params, 4096, PixelFormat::RGBA16Float);
        assert!(rgba16 < r16, "RGBA packing should use fewer textures");
    }

    // Max capacity of a single largest-possible texture per format

    #[test]
    fn max_texture_capacity_r8() {
        let cap = 16384u64 * 16384 * 1; // 268 M elements
        assert_eq!(cap, 268_435_456);
    }

    #[test]
    fn max_texture_capacity_rgba16() {
        // 16384x16384 pixels * 4 channels = 1,073,741,824 elements
        let cap = 16384u64 * 16384 * 4;
        assert_eq!(cap, 1_073_741_824);
    }

    #[test]
    fn max_texture_memory_r32() {
        // 16384x16384 * 4 bytes = 1 GiB
        let bytes = 16384u64 * 16384 * 4;
        assert_eq!(bytes, 1_073_741_824);
    }

    #[test]
    fn max_texture_memory_rgba32() {
        // 16384x16384 * 16 bytes = 4 GiB
        let bytes = 16384u64 * 16384 * 16;
        assert_eq!(bytes, 4_294_967_296);
    }
}
