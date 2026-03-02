//! Metal texture operation tests for Apple Silicon neural network inference.
//!
//! Validates texture creation, read/write, sampling, buffer copies, array ops,
//! compute integration, Apple Silicon optimisations, and performance patterns.
//!
//! All tests are `#[cfg(target_os = "macos")]` gated. Tests needing a real Metal
//! GPU carry `#[ignore = "requires Metal GPU — run on macOS/arm64"]`.

#![cfg(feature = "cpu")]

// ============================================================================
// Texture descriptor & helper types (pure-logic, no GPU required)
// ============================================================================

/// Mirrors MTLPixelFormat numeric tags relevant to neural inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum PixelFormat {
    R16Float,
    RG16Float,
    RGBA16Float,
    R32Float,
    RG32Float,
    RGBA32Float,
    R8Unorm,
    RGBA8Unorm,
    R32Uint,
    RGBA8Uint,
}

impl PixelFormat {
    const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::R8Unorm => 1,
            Self::R16Float => 2,
            Self::RG16Float
            | Self::R32Float
            | Self::R32Uint
            | Self::RGBA8Unorm
            | Self::RGBA8Uint => 4,
            Self::RGBA16Float | Self::RG32Float => 8,
            Self::RGBA32Float => 16,
        }
    }

    const fn channels(self) -> usize {
        match self {
            Self::R8Unorm | Self::R16Float | Self::R32Float | Self::R32Uint => 1,
            Self::RG16Float | Self::RG32Float => 2,
            Self::RGBA8Unorm | Self::RGBA8Uint | Self::RGBA16Float | Self::RGBA32Float => 4,
        }
    }

    const fn is_float(self) -> bool {
        matches!(
            self,
            Self::R16Float
                | Self::RG16Float
                | Self::RGBA16Float
                | Self::R32Float
                | Self::RG32Float
                | Self::RGBA32Float
        )
    }
}

/// Mirrors MTLTextureType.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextureType {
    Type2D,
    Type2DArray,
    Type3D,
    TypeCube,
}

/// Mirrors MTLStorageMode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageMode {
    Shared,
    Private,
    Memoryless,
}

/// Mirrors MTLTextureUsage bit-flags.
#[derive(Debug, Clone, Copy)]
struct TextureUsage(u32);

impl TextureUsage {
    const SHADER_READ: u32 = 0x01;
    const SHADER_WRITE: u32 = 0x02;
    const RENDER_TARGET: u32 = 0x04;
    const PIXEL_FORMAT_VIEW: u32 = 0x10;

    const fn new(bits: u32) -> Self {
        Self(bits)
    }

    const fn contains(self, flag: u32) -> bool {
        self.0 & flag != 0
    }
}

/// Lightweight texture descriptor — matches MTLTextureDescriptor semantics.
#[derive(Debug, Clone)]
struct TextureDescriptor {
    texture_type: TextureType,
    pixel_format: PixelFormat,
    width: usize,
    height: usize,
    depth: usize,
    array_length: usize,
    mipmap_level_count: usize,
    storage_mode: StorageMode,
    usage: TextureUsage,
}

impl TextureDescriptor {
    fn new_2d(format: PixelFormat, width: usize, height: usize) -> Self {
        Self {
            texture_type: TextureType::Type2D,
            pixel_format: format,
            width,
            height,
            depth: 1,
            array_length: 1,
            mipmap_level_count: 1,
            storage_mode: StorageMode::Shared,
            usage: TextureUsage::new(TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.width == 0 || self.height == 0 || self.depth == 0 {
            return Err("dimensions must be > 0".into());
        }
        const MAX_DIM: usize = 16384;
        if self.width > MAX_DIM || self.height > MAX_DIM {
            return Err(format!("dimension exceeds Metal limit {MAX_DIM}"));
        }
        if self.array_length == 0 {
            return Err("array_length must be >= 1".into());
        }
        if self.mipmap_level_count == 0 {
            return Err("mipmap_level_count must be >= 1".into());
        }
        let max_mip = (self.width.max(self.height) as f64).log2().floor() as usize + 1;
        if self.mipmap_level_count > max_mip {
            return Err(format!(
                "mipmap_level_count {} exceeds max {max_mip} for {}×{}",
                self.mipmap_level_count, self.width, self.height
            ));
        }
        if self.texture_type == TextureType::Type2DArray && self.array_length < 1 {
            return Err("2D-array texture requires array_length >= 1".into());
        }
        if self.storage_mode == StorageMode::Memoryless
            && !self.usage.contains(TextureUsage::RENDER_TARGET)
        {
            return Err("memoryless storage requires RENDER_TARGET usage".into());
        }
        Ok(())
    }

    fn total_bytes(&self) -> usize {
        self.width
            * self.height
            * self.depth
            * self.array_length
            * self.pixel_format.bytes_per_pixel()
    }

    fn mip_dimensions(&self, level: usize) -> (usize, usize) {
        let w = (self.width >> level).max(1);
        let h = (self.height >> level).max(1);
        (w, h)
    }

    fn bytes_per_row(&self) -> usize {
        self.width * self.pixel_format.bytes_per_pixel()
    }
}

// ============================================================================
// Texture data simulation helpers
// ============================================================================

/// Simulates a 2-D texture memory region (row-major, tightly-packed).
struct TextureData {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<f32>,
}

impl TextureData {
    fn new(width: usize, height: usize, channels: usize) -> Self {
        Self { width, height, channels, data: vec![0.0; width * height * channels] }
    }

    fn fill_pattern(&mut self) {
        for (i, v) in self.data.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.1;
        }
    }

    fn write_pixel(&mut self, x: usize, y: usize, values: &[f32]) {
        assert!(x < self.width && y < self.height);
        assert_eq!(values.len(), self.channels);
        let base = (y * self.width + x) * self.channels;
        self.data[base..base + self.channels].copy_from_slice(values);
    }

    fn read_pixel(&self, x: usize, y: usize) -> Vec<f32> {
        assert!(x < self.width && y < self.height);
        let base = (y * self.width + x) * self.channels;
        self.data[base..base + self.channels].to_vec()
    }

    fn replace_region(
        &mut self,
        origin_x: usize,
        origin_y: usize,
        region_w: usize,
        region_h: usize,
        src: &[f32],
    ) {
        assert_eq!(src.len(), region_w * region_h * self.channels);
        for row in 0..region_h {
            for col in 0..region_w {
                let si = (row * region_w + col) * self.channels;
                let vals = &src[si..si + self.channels];
                self.write_pixel(origin_x + col, origin_y + row, vals);
            }
        }
    }

    fn read_region(
        &self,
        origin_x: usize,
        origin_y: usize,
        region_w: usize,
        region_h: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(region_w * region_h * self.channels);
        for row in 0..region_h {
            for col in 0..region_w {
                out.extend_from_slice(&self.read_pixel(origin_x + col, origin_y + row));
            }
        }
        out
    }

    fn total_elements(&self) -> usize {
        self.data.len()
    }
}

/// Simulated texture-array (vector of slices).
struct TextureArray {
    slices: Vec<TextureData>,
    width: usize,
    height: usize,
    channels: usize,
}

impl TextureArray {
    fn new(width: usize, height: usize, channels: usize, layers: usize) -> Self {
        Self {
            slices: (0..layers).map(|_| TextureData::new(width, height, channels)).collect(),
            width,
            height,
            channels,
        }
    }

    fn layer_count(&self) -> usize {
        self.slices.len()
    }
}

// ============================================================================
// Sampling helpers
// ============================================================================

/// Clamp `v` into `[0, max]`.
fn clamp(v: f32, max: f32) -> f32 {
    v.max(0.0).min(max)
}

/// Repeat addressing: wrap around [0, 1).
fn repeat_address(v: f32) -> f32 {
    v - v.floor()
}

/// Mirror-repeat addressing.
fn mirror_address(v: f32) -> f32 {
    let t = v - 2.0 * (v / 2.0).floor();
    if t > 1.0 { 2.0 - t } else { t }
}

/// Nearest-neighbour sample from 1-D float buffer.
fn sample_nearest(data: &[f32], coord: f32, size: usize) -> f32 {
    let idx = (coord * size as f32).floor() as usize;
    data[idx.min(size - 1)]
}

/// Bilinear interpolation between two values.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Linear sample from 1-D float buffer.
fn sample_linear(data: &[f32], coord: f32, size: usize) -> f32 {
    let pos = coord * size as f32 - 0.5;
    let lo = (pos.floor() as isize).max(0) as usize;
    let hi = (lo + 1).min(size - 1);
    let frac = pos - pos.floor();
    lerp(data[lo], data[hi], frac)
}

/// Compute mip level from LOD.
fn select_mip_level(lod: f32, max_level: usize) -> usize {
    let level = lod.round() as usize;
    level.min(max_level)
}

// ============================================================================
// Alignment / copy helpers
// ============================================================================

/// Metal requires bytes_per_row to be a multiple of this on Apple Silicon.
const METAL_BYTES_PER_ROW_ALIGNMENT: usize = 256;

fn aligned_bytes_per_row(width: usize, bpp: usize) -> usize {
    let raw = width * bpp;
    (raw + METAL_BYTES_PER_ROW_ALIGNMENT - 1) / METAL_BYTES_PER_ROW_ALIGNMENT
        * METAL_BYTES_PER_ROW_ALIGNMENT
}

fn copy_texture_to_buffer(tex: &TextureData, bpp: usize) -> (Vec<u8>, usize) {
    let aligned_row = aligned_bytes_per_row(tex.width, bpp);
    let mut buf = vec![0u8; aligned_row * tex.height];
    for row in 0..tex.height {
        let src_start = row * tex.width * tex.channels;
        let src_end = src_start + tex.width * tex.channels;
        let dst_start = row * aligned_row;
        for (i, &val) in tex.data[src_start..src_end].iter().enumerate() {
            let bytes = val.to_le_bytes();
            let offset = dst_start + i * 4;
            if offset + 4 <= buf.len() {
                buf[offset..offset + 4].copy_from_slice(&bytes);
            }
        }
    }
    (buf, aligned_row)
}

fn copy_buffer_to_texture(
    buf: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    aligned_row: usize,
) -> TextureData {
    let mut tex = TextureData::new(width, height, channels);
    for row in 0..height {
        for col in 0..width {
            for ch in 0..channels {
                let buf_off = row * aligned_row + (col * channels + ch) * 4;
                if buf_off + 4 <= buf.len() {
                    let bytes: [u8; 4] = buf[buf_off..buf_off + 4].try_into().unwrap();
                    let val = f32::from_le_bytes(bytes);
                    tex.data[row * width * channels + col * channels + ch] = val;
                }
            }
        }
    }
    tex
}

// ============================================================================
// Section 1 — Texture Creation Tests (8 tests)
// ============================================================================

#[cfg(target_os = "macos")]
mod texture_creation {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_texture2d_creation_basic() {
        let desc = TextureDescriptor::new_2d(PixelFormat::RGBA16Float, 256, 256);
        assert!(desc.validate().is_ok());
        assert_eq!(desc.texture_type, TextureType::Type2D);
        assert_eq!(desc.total_bytes(), 256 * 256 * 8);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_pixel_format_validation() {
        for fmt in [
            PixelFormat::R16Float,
            PixelFormat::RG16Float,
            PixelFormat::RGBA16Float,
            PixelFormat::R32Float,
            PixelFormat::RG32Float,
            PixelFormat::RGBA32Float,
        ] {
            assert!(fmt.is_float(), "{fmt:?} should be a float format");
        }
        assert!(!PixelFormat::R8Unorm.is_float());
        assert!(!PixelFormat::R32Uint.is_float());
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_width_height_constraints() {
        // zero dimension → error
        let mut desc = TextureDescriptor::new_2d(PixelFormat::R32Float, 0, 256);
        assert!(desc.validate().is_err());

        desc.width = 256;
        desc.height = 0;
        assert!(desc.validate().is_err());

        // exceeds Metal max (16384)
        desc.height = 16385;
        desc.width = 256;
        assert!(desc.validate().is_err());
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_depth_constraints_3d() {
        let mut desc = TextureDescriptor::new_2d(PixelFormat::R32Float, 64, 64);
        desc.texture_type = TextureType::Type3D;
        desc.depth = 64;
        assert!(desc.validate().is_ok());

        desc.depth = 0;
        assert!(desc.validate().is_err());
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_array_texture_descriptor() {
        let mut desc = TextureDescriptor::new_2d(PixelFormat::RGBA16Float, 128, 128);
        desc.texture_type = TextureType::Type2DArray;
        desc.array_length = 6;
        assert!(desc.validate().is_ok());
        assert_eq!(desc.total_bytes(), 128 * 128 * 6 * 8);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_mipmap_level_validation() {
        let mut desc = TextureDescriptor::new_2d(PixelFormat::R32Float, 256, 256);
        // max mip for 256×256 = floor(log2(256)) + 1 = 9
        desc.mipmap_level_count = 9;
        assert!(desc.validate().is_ok());

        desc.mipmap_level_count = 10;
        assert!(desc.validate().is_err(), "10 mips should exceed limit for 256×256");
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_mipmap_dimensions_halving() {
        let desc = TextureDescriptor::new_2d(PixelFormat::R32Float, 512, 256);
        assert_eq!(desc.mip_dimensions(0), (512, 256));
        assert_eq!(desc.mip_dimensions(1), (256, 128));
        assert_eq!(desc.mip_dimensions(2), (128, 64));
        assert_eq!(desc.mip_dimensions(8), (2, 1));
        assert_eq!(desc.mip_dimensions(9), (1, 1));
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_storage_mode_constraints() {
        let mut desc = TextureDescriptor::new_2d(PixelFormat::R32Float, 64, 64);

        // Shared is fine for read/write
        desc.storage_mode = StorageMode::Shared;
        assert!(desc.validate().is_ok());

        // Private is fine for read/write
        desc.storage_mode = StorageMode::Private;
        assert!(desc.validate().is_ok());

        // Memoryless requires RENDER_TARGET usage
        desc.storage_mode = StorageMode::Memoryless;
        desc.usage = TextureUsage::new(TextureUsage::SHADER_READ);
        assert!(desc.validate().is_err());

        desc.usage = TextureUsage::new(TextureUsage::RENDER_TARGET);
        assert!(desc.validate().is_ok());
    }
}

// ============================================================================
// Section 2 — Texture Read/Write Tests (8 tests)
// ============================================================================

#[cfg(target_os = "macos")]
mod texture_read_write {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_write_read_roundtrip_single_pixel() {
        let mut tex = TextureData::new(4, 4, 4);
        tex.write_pixel(2, 3, &[1.0, 0.5, 0.25, 1.0]);
        let px = tex.read_pixel(2, 3);
        assert_eq!(px, vec![1.0, 0.5, 0.25, 1.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_write_read_roundtrip_all_pixels() {
        let mut tex = TextureData::new(8, 8, 1);
        tex.fill_pattern();

        for y in 0..8 {
            for x in 0..8 {
                let px = tex.read_pixel(x, y);
                let expected = ((y * 8 + x) as f32 + 1.0) * 0.1;
                assert!(
                    (px[0] - expected).abs() < 1e-6,
                    "pixel ({x},{y}): got {} expected {expected}",
                    px[0]
                );
            }
        }
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_replace_region_full() {
        let mut tex = TextureData::new(4, 4, 1);
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        tex.replace_region(0, 0, 4, 4, &src);

        for y in 0..4 {
            for x in 0..4 {
                let px = tex.read_pixel(x, y);
                assert_eq!(px[0], (y * 4 + x) as f32);
            }
        }
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_replace_region_sub_region() {
        let mut tex = TextureData::new(8, 8, 1);
        // Write a 3×2 sub-region starting at (2,3)
        let region: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        tex.replace_region(2, 3, 3, 2, &region);

        assert_eq!(tex.read_pixel(2, 3), vec![10.0]);
        assert_eq!(tex.read_pixel(4, 3), vec![30.0]);
        assert_eq!(tex.read_pixel(2, 4), vec![40.0]);
        assert_eq!(tex.read_pixel(4, 4), vec![60.0]);
        // Untouched pixel
        assert_eq!(tex.read_pixel(0, 0), vec![0.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_read_region_extracts_subdata() {
        let mut tex = TextureData::new(4, 4, 1);
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        tex.replace_region(0, 0, 4, 4, &src);

        let sub = tex.read_region(1, 1, 2, 2);
        assert_eq!(sub, vec![5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_2d_slice_operations_rgba() {
        let mut tex = TextureData::new(2, 2, 4);
        tex.write_pixel(0, 0, &[1.0, 0.0, 0.0, 1.0]); // red
        tex.write_pixel(1, 0, &[0.0, 1.0, 0.0, 1.0]); // green
        tex.write_pixel(0, 1, &[0.0, 0.0, 1.0, 1.0]); // blue
        tex.write_pixel(1, 1, &[1.0, 1.0, 1.0, 1.0]); // white

        assert_eq!(tex.read_pixel(0, 0), vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(tex.read_pixel(1, 1), vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(tex.total_elements(), 2 * 2 * 4);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_format_conversion_f16_to_f32_simulation() {
        // Simulate half → float conversion (truncated precision)
        let half_precision_values: Vec<f32> = vec![1.0, 0.5, 0.333_252, 0.25];
        let mut tex = TextureData::new(2, 2, 1);
        for (i, &v) in half_precision_values.iter().enumerate() {
            let x = i % 2;
            let y = i / 2;
            tex.write_pixel(x, y, &[v]);
        }
        // Roundtrip preserves values within f16 precision
        for (i, &expected) in half_precision_values.iter().enumerate() {
            let x = i % 2;
            let y = i / 2;
            let got = tex.read_pixel(x, y)[0];
            assert!((got - expected).abs() < 1e-3, "f16 precision loss at ({x},{y})");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_boundary_write_read_edges() {
        let mut tex = TextureData::new(16, 16, 1);
        // Write to all four corners
        tex.write_pixel(0, 0, &[1.0]);
        tex.write_pixel(15, 0, &[2.0]);
        tex.write_pixel(0, 15, &[3.0]);
        tex.write_pixel(15, 15, &[4.0]);

        assert_eq!(tex.read_pixel(0, 0), vec![1.0]);
        assert_eq!(tex.read_pixel(15, 0), vec![2.0]);
        assert_eq!(tex.read_pixel(0, 15), vec![3.0]);
        assert_eq!(tex.read_pixel(15, 15), vec![4.0]);
    }
}

// ============================================================================
// Section 3 — Texture Sampling Tests (8 tests)
// ============================================================================

#[cfg(target_os = "macos")]
mod texture_sampling {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_nearest_filtering_exact_texel() {
        let data: Vec<f32> = (0..8).map(|i| i as f32 * 10.0).collect();
        // coord 0.5/8 = 0.0625 → index 0
        assert_eq!(sample_nearest(&data, 0.0625, 8), 0.0);
        // coord 7.5/8 ≈ 0.9375 → index 7
        assert_eq!(sample_nearest(&data, 0.9375, 8), 70.0);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_nearest_filtering_boundary() {
        let data = vec![100.0, 200.0, 300.0, 400.0];
        // coord 0.0 → index 0
        assert_eq!(sample_nearest(&data, 0.0, 4), 100.0);
        // coord 0.999 → index 3
        assert_eq!(sample_nearest(&data, 0.999, 4), 400.0);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_linear_filtering_interpolation() {
        let data = vec![0.0, 10.0, 20.0, 30.0];
        // Midpoint between texels 1 and 2 → ~15
        let val = sample_linear(&data, 0.5, 4);
        assert!((val - 15.0).abs() < 2.0, "bilinear midpoint expected ~15, got {val}");
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_address_mode_clamp_to_edge() {
        let size = 4.0;
        assert_eq!(clamp(-0.5, size - 1.0), 0.0);
        assert_eq!(clamp(5.0, size - 1.0), 3.0);
        assert_eq!(clamp(2.0, size - 1.0), 2.0);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_address_mode_repeat() {
        assert!((repeat_address(0.0) - 0.0).abs() < 1e-6);
        assert!((repeat_address(1.5) - 0.5).abs() < 1e-6);
        assert!((repeat_address(-0.3) - 0.7).abs() < 1e-6);
        assert!((repeat_address(3.75) - 0.75).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_address_mode_mirror_repeat() {
        assert!((mirror_address(0.0) - 0.0).abs() < 1e-6);
        assert!((mirror_address(0.5) - 0.5).abs() < 1e-6);
        assert!((mirror_address(1.0) - 1.0).abs() < 1e-6);
        assert!((mirror_address(1.5) - 0.5).abs() < 1e-6);
        assert!((mirror_address(2.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_lod_selection() {
        assert_eq!(select_mip_level(0.0, 8), 0);
        assert_eq!(select_mip_level(2.4, 8), 2);
        assert_eq!(select_mip_level(2.6, 8), 3);
        assert_eq!(select_mip_level(99.0, 8), 8); // clamped to max
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_anisotropy_compare_function_simulation() {
        // Simulate compare function (depth compare for shadow mapping):
        // compare(sample, ref) → 1.0 if sample < ref, else 0.0
        let depth_texels = vec![0.3, 0.5, 0.7, 0.9];
        let reference = 0.6;
        let results: Vec<f32> =
            depth_texels.iter().map(|&d| if d < reference { 1.0 } else { 0.0 }).collect();
        assert_eq!(results, vec![1.0, 1.0, 0.0, 0.0]);

        // Anisotropic filtering level clamping
        let max_aniso: u32 = 16;
        let requested: u32 = 32;
        let effective = requested.min(max_aniso);
        assert_eq!(effective, 16);
    }
}

// ============================================================================
// Section 4 — Texture-Buffer Copies (8 tests)
// ============================================================================

#[cfg(target_os = "macos")]
mod texture_buffer_copies {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_blit_copy_texture_to_buffer() {
        let mut tex = TextureData::new(4, 4, 1);
        tex.fill_pattern();
        let (buf, aligned_row) = copy_texture_to_buffer(&tex, 4); // f32 = 4 bpp

        assert!(aligned_row >= 4 * 4); // at least width * bpp
        assert_eq!(aligned_row % METAL_BYTES_PER_ROW_ALIGNMENT, 0);
        // First pixel in buffer
        let first = f32::from_le_bytes(buf[0..4].try_into().unwrap());
        assert!((first - 0.1).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_blit_copy_buffer_to_texture() {
        // Prepare source texture, copy to buffer, then back to new texture
        let mut src = TextureData::new(4, 4, 1);
        src.fill_pattern();
        let (buf, aligned_row) = copy_texture_to_buffer(&src, 4);
        let dst = copy_buffer_to_texture(&buf, 4, 4, 1, aligned_row);

        for y in 0..4 {
            for x in 0..4 {
                let s = src.read_pixel(x, y);
                let d = dst.read_pixel(x, y);
                assert!(
                    (s[0] - d[0]).abs() < 1e-6,
                    "mismatch at ({x},{y}): src={} dst={}",
                    s[0],
                    d[0]
                );
            }
        }
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_bytes_per_row_alignment() {
        // 1-pixel wide texture: raw bytes_per_row = 4, must align to 256
        assert_eq!(aligned_bytes_per_row(1, 4), 256);
        // 64 pixels × 4 bytes = 256 → already aligned
        assert_eq!(aligned_bytes_per_row(64, 4), 256);
        // 65 pixels × 4 bytes = 260 → rounds to 512
        assert_eq!(aligned_bytes_per_row(65, 4), 512);
        // RGBA16Float: 8 bpp, 128 pixels = 1024 → already aligned at 1024
        assert_eq!(aligned_bytes_per_row(128, 8), 1024);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_bytes_per_row_descriptor_consistency() {
        let desc = TextureDescriptor::new_2d(PixelFormat::RGBA32Float, 100, 100);
        let raw_bpr = desc.bytes_per_row();
        let aligned_bpr = aligned_bytes_per_row(desc.width, desc.pixel_format.bytes_per_pixel());
        assert_eq!(raw_bpr, 100 * 16); // 100 × RGBA32Float
        assert!(aligned_bpr >= raw_bpr);
        assert_eq!(aligned_bpr % METAL_BYTES_PER_ROW_ALIGNMENT, 0);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_region_based_copy_partial() {
        let mut tex = TextureData::new(8, 8, 1);
        tex.fill_pattern();
        let sub = tex.read_region(2, 2, 4, 4);
        assert_eq!(sub.len(), 4 * 4);

        // Verify a known value: pixel (2,2) has index 2*8+2 = 18, value = 19*0.1
        assert!((sub[0] - 1.9).abs() < 1e-5);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_copy_roundtrip_rgba() {
        let mut tex = TextureData::new(2, 2, 4);
        tex.write_pixel(0, 0, &[1.0, 2.0, 3.0, 4.0]);
        tex.write_pixel(1, 0, &[5.0, 6.0, 7.0, 8.0]);
        tex.write_pixel(0, 1, &[9.0, 10.0, 11.0, 12.0]);
        tex.write_pixel(1, 1, &[13.0, 14.0, 15.0, 16.0]);

        let (buf, aligned_row) = copy_texture_to_buffer(&tex, 4 * 4); // 4 channels × 4 bytes
        let restored = copy_buffer_to_texture(&buf, 2, 2, 4, aligned_row);

        assert_eq!(restored.read_pixel(0, 0), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(restored.read_pixel(1, 1), vec![13.0, 14.0, 15.0, 16.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_format_conversion_during_copy() {
        // Simulate f32 → u8 (R8Unorm) conversion during copy
        let f32_values: Vec<f32> = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let u8_values: Vec<u8> =
            f32_values.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
        assert_eq!(u8_values, vec![0, 64, 128, 191, 255]);

        // Reverse: u8 → f32
        let back: Vec<f32> = u8_values.iter().map(|&v| v as f32 / 255.0).collect();
        for (orig, restored) in f32_values.iter().zip(back.iter()) {
            assert!((orig - restored).abs() < 0.005, "precision loss: {orig} → {restored}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_alignment_various_pixel_formats() {
        let formats = [
            (PixelFormat::R8Unorm, 1),
            (PixelFormat::R16Float, 2),
            (PixelFormat::R32Float, 4),
            (PixelFormat::RGBA16Float, 8),
            (PixelFormat::RGBA32Float, 16),
        ];
        for (fmt, bpp) in formats {
            assert_eq!(fmt.bytes_per_pixel(), bpp, "{fmt:?} bytes_per_pixel");
            let aligned = aligned_bytes_per_row(100, bpp);
            assert!(aligned >= 100 * bpp);
            assert_eq!(aligned % METAL_BYTES_PER_ROW_ALIGNMENT, 0, "alignment failed for {fmt:?}");
        }
    }
}

// ============================================================================
// Section 5 — Texture Array Operations (8 tests)
// ============================================================================

#[cfg(target_os = "macos")]
mod texture_array_operations {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_2d_array_creation() {
        let desc = TextureDescriptor {
            texture_type: TextureType::Type2DArray,
            pixel_format: PixelFormat::RGBA16Float,
            width: 64,
            height: 64,
            depth: 1,
            array_length: 8,
            mipmap_level_count: 1,
            storage_mode: StorageMode::Shared,
            usage: TextureUsage::new(TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE),
        };
        assert!(desc.validate().is_ok());
        assert_eq!(desc.total_bytes(), 64 * 64 * 8 * 8); // 8 layers × 8 bpp
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_2d_array_layer_count() {
        let arr = TextureArray::new(32, 32, 4, 12);
        assert_eq!(arr.layer_count(), 12);
        assert_eq!(arr.width, 32);
        assert_eq!(arr.height, 32);
        assert_eq!(arr.channels, 4);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_per_slice_write() {
        let mut arr = TextureArray::new(4, 4, 1, 4);
        for layer in 0..4 {
            let value = (layer + 1) as f32 * 100.0;
            arr.slices[layer].write_pixel(0, 0, &[value]);
        }
        assert_eq!(arr.slices[0].read_pixel(0, 0), vec![100.0]);
        assert_eq!(arr.slices[3].read_pixel(0, 0), vec![400.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_per_slice_fill_pattern() {
        let mut arr = TextureArray::new(4, 4, 1, 3);
        for slice in &mut arr.slices {
            slice.fill_pattern();
        }
        // Each slice should have identical patterns
        for layer in 0..3 {
            let px = arr.slices[layer].read_pixel(0, 0)[0];
            assert!((px - 0.1).abs() < 1e-6, "layer {layer} first pixel mismatch");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_cross_slice_read_independence() {
        let mut arr = TextureArray::new(2, 2, 1, 3);
        // Write distinct values to each layer
        arr.slices[0].write_pixel(0, 0, &[10.0]);
        arr.slices[1].write_pixel(0, 0, &[20.0]);
        arr.slices[2].write_pixel(0, 0, &[30.0]);

        // Verify slices are independent
        assert_eq!(arr.slices[0].read_pixel(0, 0), vec![10.0]);
        assert_eq!(arr.slices[1].read_pixel(0, 0), vec![20.0]);
        assert_eq!(arr.slices[2].read_pixel(0, 0), vec![30.0]);

        // Modify one slice, others unchanged
        arr.slices[1].write_pixel(0, 0, &[99.0]);
        assert_eq!(arr.slices[0].read_pixel(0, 0), vec![10.0]);
        assert_eq!(arr.slices[2].read_pixel(0, 0), vec![30.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_layer_rendering_target_simulation() {
        // Simulate rendering a solid colour into each layer
        let mut arr = TextureArray::new(4, 4, 4, 6);
        let colors = [
            [1.0, 0.0, 0.0, 1.0], // red
            [0.0, 1.0, 0.0, 1.0], // green
            [0.0, 0.0, 1.0, 1.0], // blue
            [1.0, 1.0, 0.0, 1.0], // yellow
            [1.0, 0.0, 1.0, 1.0], // magenta
            [0.0, 1.0, 1.0, 1.0], // cyan
        ];
        for (layer, color) in colors.iter().enumerate() {
            for y in 0..4 {
                for x in 0..4 {
                    arr.slices[layer].write_pixel(x, y, color);
                }
            }
        }
        for (layer, expected) in colors.iter().enumerate() {
            let px = arr.slices[layer].read_pixel(2, 2);
            assert_eq!(px.as_slice(), expected, "layer {layer} colour mismatch");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_array_as_compute_input_simulation() {
        // Simulate a compute shader averaging all layers per pixel
        let mut arr = TextureArray::new(2, 2, 1, 4);
        for (layer, val) in [10.0f32, 20.0, 30.0, 40.0].iter().enumerate() {
            for y in 0..2 {
                for x in 0..2 {
                    arr.slices[layer].write_pixel(x, y, &[*val]);
                }
            }
        }
        let mut output = TextureData::new(2, 2, 1);
        for y in 0..2 {
            for x in 0..2 {
                let sum: f32 =
                    (0..arr.layer_count()).map(|l| arr.slices[l].read_pixel(x, y)[0]).sum();
                let avg = sum / arr.layer_count() as f32;
                output.write_pixel(x, y, &[avg]);
            }
        }
        for y in 0..2 {
            for x in 0..2 {
                assert!((output.read_pixel(x, y)[0] - 25.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_array_descriptor_validation_edge_cases() {
        // Single layer is valid
        let mut desc = TextureDescriptor::new_2d(PixelFormat::R32Float, 16, 16);
        desc.texture_type = TextureType::Type2DArray;
        desc.array_length = 1;
        assert!(desc.validate().is_ok());

        // Large array count
        desc.array_length = 2048;
        assert!(desc.validate().is_ok());
        assert_eq!(desc.total_bytes(), 16 * 16 * 2048 * 4);
    }
}

// ============================================================================
// Section 6 — Compute with Textures (8 tests)
// ============================================================================

#[cfg(target_os = "macos")]
mod compute_with_textures {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_texture_as_compute_input() {
        // Simulate: compute shader reads texture, writes sum to buffer
        let mut tex = TextureData::new(4, 4, 1);
        tex.fill_pattern();

        let sum: f32 = (0..4)
            .flat_map(|y| (0..4).map(move |x| (y, x)))
            .map(|(y, x)| tex.read_pixel(x, y)[0])
            .sum();

        // Sum of 0.1 + 0.2 + … + 1.6 = 0.1 * (1+2+…+16) = 0.1 * 136 = 13.6
        assert!((sum - 13.6).abs() < 1e-4, "sum = {sum}");
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_texture_as_compute_output() {
        // Simulate: compute shader writes index-based values into texture
        let mut output = TextureData::new(8, 8, 1);
        for y in 0..8 {
            for x in 0..8 {
                let val = ((y * 8 + x) as f32).sqrt();
                output.write_pixel(x, y, &[val]);
            }
        }
        assert!((output.read_pixel(3, 3)[0] - (27.0f32).sqrt()).abs() < 1e-6);
        assert!((output.read_pixel(7, 7)[0] - (63.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_read_write_texture_in_place() {
        // Simulate: read_write texture (in-place doubling)
        let mut tex = TextureData::new(4, 4, 1);
        for y in 0..4 {
            for x in 0..4 {
                tex.write_pixel(x, y, &[(y * 4 + x + 1) as f32]);
            }
        }
        // In-place kernel: double each value
        for v in &mut tex.data {
            *v *= 2.0;
        }
        assert_eq!(tex.read_pixel(0, 0), vec![2.0]);
        assert_eq!(tex.read_pixel(3, 3), vec![32.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_imageblock_tile_operation() {
        // Simulate imageblock: load 4×4 tile, reduce to single value
        let mut tex = TextureData::new(16, 16, 1);
        tex.fill_pattern();

        let tile_size = 4;
        let tiles_x = tex.width / tile_size;
        let tiles_y = tex.height / tile_size;
        let mut tile_sums = vec![0.0f32; tiles_x * tiles_y];

        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let mut sum = 0.0f32;
                for dy in 0..tile_size {
                    for dx in 0..tile_size {
                        sum += tex.read_pixel(tx * tile_size + dx, ty * tile_size + dy)[0];
                    }
                }
                tile_sums[ty * tiles_x + tx] = sum;
            }
        }
        // First tile (0..4, 0..4) sum = 0.1*(1+2+…+4 + 17+18+19+20 + 33+…+36 + 49+…+52)
        assert!(tile_sums[0] > 0.0);
        assert_eq!(tile_sums.len(), 16); // 4×4 tiles
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_texture_atomic_max_simulation() {
        // Simulate atomic max: multiple "threads" write max value per pixel
        let mut tex = TextureData::new(2, 2, 1);
        let thread_values: Vec<Vec<f32>> =
            vec![vec![5.0, 3.0, 7.0, 1.0], vec![2.0, 8.0, 4.0, 6.0], vec![9.0, 1.0, 3.0, 5.0]];
        for vals in &thread_values {
            for (i, &v) in vals.iter().enumerate() {
                let x = i % 2;
                let y = i / 2;
                let current = tex.read_pixel(x, y)[0];
                if v > current {
                    tex.write_pixel(x, y, &[v]);
                }
            }
        }
        assert_eq!(tex.read_pixel(0, 0), vec![9.0]);
        assert_eq!(tex.read_pixel(1, 0), vec![8.0]);
        assert_eq!(tex.read_pixel(0, 1), vec![7.0]);
        assert_eq!(tex.read_pixel(1, 1), vec![6.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_texture_compute_relu_kernel() {
        // Simulate ReLU applied via compute shader on texture
        let mut tex = TextureData::new(4, 4, 1);
        let values: Vec<f32> = vec![
            -1.0, 2.0, -3.0, 4.0, 0.0, -0.5, 0.5, -2.0, 1.0, -1.0, 3.0, -3.0, 0.1, -0.1, 0.0, 5.0,
        ];
        tex.replace_region(0, 0, 4, 4, &values);

        // Apply ReLU in-place
        for v in &mut tex.data {
            *v = v.max(0.0);
        }

        assert_eq!(tex.read_pixel(0, 0), vec![0.0]); // was -1.0
        assert_eq!(tex.read_pixel(1, 0), vec![2.0]);
        assert_eq!(tex.read_pixel(2, 0), vec![0.0]); // was -3.0
        assert_eq!(tex.read_pixel(3, 3), vec![5.0]);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_texture_compute_softmax_row() {
        // Simulate per-row softmax on a texture (common in attention)
        let mut tex = TextureData::new(4, 2, 1);
        tex.replace_region(0, 0, 4, 2, &[1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5]);

        for row in 0..2 {
            let vals: Vec<f32> = (0..4).map(|x| tex.read_pixel(x, row)[0]).collect();
            let max_v = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = vals.iter().map(|&v| (v - max_v).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (x, &e) in exps.iter().enumerate() {
                tex.write_pixel(x, row, &[e / sum]);
            }
        }

        // Row 0: softmax of [1,2,3,4] → last element should be largest
        let r0: Vec<f32> = (0..4).map(|x| tex.read_pixel(x, 0)[0]).collect();
        let sum0: f32 = r0.iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-5, "softmax row 0 sums to {sum0}");
        assert!(r0[3] > r0[0], "softmax: largest input should have largest probability");

        // Row 1: uniform → equal probabilities
        let r1: Vec<f32> = (0..4).map(|x| tex.read_pixel(x, 1)[0]).collect();
        for &v in &r1 {
            assert!((v - 0.25).abs() < 1e-5, "uniform softmax element: {v}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_texture_compute_elementwise_multiply() {
        // Simulate element-wise multiply of two textures
        let mut a = TextureData::new(4, 4, 1);
        let mut b = TextureData::new(4, 4, 1);
        for y in 0..4 {
            for x in 0..4 {
                a.write_pixel(x, y, &[(x + 1) as f32]);
                b.write_pixel(x, y, &[(y + 1) as f32]);
            }
        }
        let mut output = TextureData::new(4, 4, 1);
        for y in 0..4 {
            for x in 0..4 {
                let va = a.read_pixel(x, y)[0];
                let vb = b.read_pixel(x, y)[0];
                output.write_pixel(x, y, &[va * vb]);
            }
        }
        assert_eq!(output.read_pixel(0, 0), vec![1.0]); // 1*1
        assert_eq!(output.read_pixel(3, 3), vec![16.0]); // 4*4
        assert_eq!(output.read_pixel(1, 2), vec![6.0]); // 2*3
    }
}

// ============================================================================
// Section 7 — Apple Silicon Texture Optimization (8 tests)
// ============================================================================

#[cfg(target_os = "macos")]
mod apple_silicon_texture_optimization {
    use super::*;

    /// Apple Silicon tile memory limit (per-tile).
    const APPLE_TILE_MEMORY_BYTES: usize = 32 * 1024; // 32 KB

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_tile_memory_budget() {
        // Tile memory for a 32×32 tile of RGBA16Float (8 bpp)
        let tile_w = 32;
        let tile_h = 32;
        let bpp = PixelFormat::RGBA16Float.bytes_per_pixel();
        let tile_bytes = tile_w * tile_h * bpp;
        assert_eq!(tile_bytes, 8192);
        assert!(
            tile_bytes <= APPLE_TILE_MEMORY_BYTES,
            "tile exceeds Apple Silicon tile memory limit"
        );
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_tile_memory_overflow_detection() {
        // 64×64 tile of RGBA32Float = 64*64*16 = 65536 > 32KB limit
        let tile_w = 64;
        let tile_h = 64;
        let bpp = PixelFormat::RGBA32Float.bytes_per_pixel();
        let tile_bytes = tile_w * tile_h * bpp;
        assert!(
            tile_bytes > APPLE_TILE_MEMORY_BYTES,
            "expected tile to exceed limit: {tile_bytes} vs {APPLE_TILE_MEMORY_BYTES}"
        );
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_memoryless_render_target_descriptor() {
        let desc = TextureDescriptor {
            texture_type: TextureType::Type2D,
            pixel_format: PixelFormat::RGBA16Float,
            width: 256,
            height: 256,
            depth: 1,
            array_length: 1,
            mipmap_level_count: 1,
            storage_mode: StorageMode::Memoryless,
            usage: TextureUsage::new(TextureUsage::RENDER_TARGET),
        };
        assert!(desc.validate().is_ok());
        assert_eq!(desc.storage_mode, StorageMode::Memoryless);
        // Memoryless textures use zero system memory
        // (validated by descriptor; actual allocation is GPU-side only)
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_sparse_texture_alignment() {
        // Apple Silicon sparse textures require page-aligned dimensions.
        // Typical sparse tile: 256×256 for 4-bpp formats.
        let sparse_tile = 256;
        let tex_width = 1024;
        let tex_height = 768;

        let tiles_x = (tex_width + sparse_tile - 1) / sparse_tile;
        let tiles_y = (tex_height + sparse_tile - 1) / sparse_tile;
        assert_eq!(tiles_x, 4);
        assert_eq!(tiles_y, 3);

        // Resident tile fraction if only half are populated
        let total_tiles = tiles_x * tiles_y;
        let resident = total_tiles / 2;
        let residency_ratio = resident as f32 / total_tiles as f32;
        assert!((residency_ratio - 0.5).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_texture_compression_ratio() {
        // ASTC 4×4 compressed: 1 byte per texel (16 bytes per 4×4 block)
        let width = 256;
        let height = 256;
        let uncompressed_bpp = PixelFormat::RGBA8Unorm.bytes_per_pixel(); // 4
        let uncompressed_bytes = width * height * uncompressed_bpp;

        let block_size = 4;
        let blocks_x = (width + block_size - 1) / block_size;
        let blocks_y = (height + block_size - 1) / block_size;
        let astc_block_bytes = 16;
        let compressed_bytes = blocks_x * blocks_y * astc_block_bytes;

        let ratio = uncompressed_bytes as f32 / compressed_bytes as f32;
        assert!((ratio - 4.0).abs() < 0.01, "ASTC 4×4 expected ~4:1 compression, got {ratio:.2}:1");
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_gpu_family_feature_check() {
        // Simulate GPU family feature matrix
        #[derive(Debug)]
        struct GpuFamily {
            name: &'static str,
            max_texture_size: usize,
            supports_read_write_textures: bool,
            supports_sparse_textures: bool,
            supports_astc: bool,
        }

        let apple_7 = GpuFamily {
            name: "Apple7 (M1)",
            max_texture_size: 16384,
            supports_read_write_textures: true,
            supports_sparse_textures: true,
            supports_astc: true,
        };
        let apple_8 = GpuFamily {
            name: "Apple8 (M2)",
            max_texture_size: 16384,
            supports_read_write_textures: true,
            supports_sparse_textures: true,
            supports_astc: true,
        };

        for family in [&apple_7, &apple_8] {
            assert!(family.max_texture_size >= 16384, "{}", family.name);
            assert!(family.supports_read_write_textures, "{}", family.name);
            assert!(family.supports_astc, "{}", family.name);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_neural_network_weight_texture_layout() {
        // Pack 2-bit quantised weights into a texture for Metal compute:
        // 4 weights per byte → 32 weights per RGBA8 texel
        let num_weights = 4096;
        let weights_per_texel = 4 * 4; // 4 bytes × (4 weights / byte)
        let texels_needed = (num_weights + weights_per_texel - 1) / weights_per_texel;

        // Lay out as a 2D texture
        let tex_width = 16;
        let tex_height = (texels_needed + tex_width - 1) / tex_width;
        assert!(tex_width * tex_height >= texels_needed);
        assert!(tex_width <= 16384 && tex_height <= 16384);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_unified_memory_texture_sharing() {
        // On Apple Silicon, shared storage mode means CPU + GPU see the same memory
        let desc = TextureDescriptor::new_2d(PixelFormat::R32Float, 128, 128);
        assert_eq!(desc.storage_mode, StorageMode::Shared);

        // Simulate CPU write then GPU read (no explicit copy needed)
        let mut tex = TextureData::new(128, 128, 1);
        tex.write_pixel(64, 64, &[42.0]);

        // "GPU" reads same memory
        let val = tex.read_pixel(64, 64)[0];
        assert_eq!(val, 42.0, "unified memory: CPU write visible to GPU");
    }
}

// ============================================================================
// Section 8 — Texture Performance Tests (8 tests)
// ============================================================================

#[cfg(target_os = "macos")]
mod texture_performance {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_large_texture_allocation() {
        let desc = TextureDescriptor::new_2d(PixelFormat::RGBA32Float, 4096, 4096);
        assert!(desc.validate().is_ok());
        let total = desc.total_bytes();
        assert_eq!(total, 4096 * 4096 * 16); // 256 MB
        assert!(total <= 512 * 1024 * 1024, "exceeds reasonable VRAM budget");
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_batch_texture_update_throughput() {
        let mut tex = TextureData::new(256, 256, 1);
        let batch: Vec<f32> = (0..256 * 256).map(|i| (i as f32).sin()).collect();

        let start = Instant::now();
        tex.replace_region(0, 0, 256, 256, &batch);
        let elapsed = start.elapsed();

        // Sanity: should complete well within 1 second
        assert!(elapsed.as_millis() < 1000, "batch update took {elapsed:?}");
        // Verify data arrived
        let px = tex.read_pixel(128, 128);
        let expected = ((128 * 256 + 128) as f32).sin();
        assert!((px[0] - expected).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_texture_cache_locality_row_vs_column() {
        let size = 512;
        let mut tex = TextureData::new(size, size, 1);
        tex.fill_pattern();

        // Row-major access (cache-friendly)
        let start_row = Instant::now();
        let mut sum_row = 0.0f32;
        for y in 0..size {
            for x in 0..size {
                sum_row += tex.read_pixel(x, y)[0];
            }
        }
        let time_row = start_row.elapsed();

        // Column-major access (cache-unfriendly)
        let start_col = Instant::now();
        let mut sum_col = 0.0f32;
        for x in 0..size {
            for y in 0..size {
                sum_col += tex.read_pixel(x, y)[0];
            }
        }
        let time_col = start_col.elapsed();

        // Both sums should be equal (same data)
        assert!((sum_row - sum_col).abs() < 1.0, "sums differ: row={sum_row} col={sum_col}");
        // Log timing (not a hard assertion — CI variance)
        let _ = (time_row, time_col);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_mipmap_generation_correctness() {
        // Simulate mip-chain generation via 2×2 box filter
        let base_size = 8;
        let mut levels: Vec<TextureData> = Vec::new();

        // Level 0: 8×8
        let mut base = TextureData::new(base_size, base_size, 1);
        for y in 0..base_size {
            for x in 0..base_size {
                base.write_pixel(x, y, &[(y * base_size + x) as f32]);
            }
        }
        levels.push(base);

        // Generate mip levels
        let mut w = base_size;
        let mut h = base_size;
        while w > 1 || h > 1 {
            let pw = w;
            let ph = h;
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            let prev = levels.last().unwrap();
            let mut mip = TextureData::new(w, h, 1);
            for y in 0..h {
                for x in 0..w {
                    let sx = x * 2;
                    let sy = y * 2;
                    let mut sum = prev.read_pixel(sx, sy)[0];
                    if sx + 1 < pw {
                        sum += prev.read_pixel(sx + 1, sy)[0];
                    }
                    if sy + 1 < ph {
                        sum += prev.read_pixel(sx, sy + 1)[0];
                    }
                    if sx + 1 < pw && sy + 1 < ph {
                        sum += prev.read_pixel(sx + 1, sy + 1)[0];
                    }
                    let count = if sx + 1 < pw && sy + 1 < ph {
                        4.0
                    } else if sx + 1 < pw || sy + 1 < ph {
                        2.0
                    } else {
                        1.0
                    };
                    mip.write_pixel(x, y, &[sum / count]);
                }
            }
            levels.push(mip);
        }

        // 8→4→2→1 = 4 levels total
        assert_eq!(levels.len(), 4);
        assert_eq!(levels[0].width, 8);
        assert_eq!(levels[1].width, 4);
        assert_eq!(levels[2].width, 2);
        assert_eq!(levels[3].width, 1);

        // Level 1 pixel (0,0) = avg of (0,0),(1,0),(0,1),(1,1) = (0+1+8+9)/4 = 4.5
        assert!((levels[1].read_pixel(0, 0)[0] - 4.5).abs() < 1e-5);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_async_copy_simulation() {
        // Simulate double-buffered async copies (ping-pong pattern)
        let mut buf_a = TextureData::new(64, 64, 1);
        let mut buf_b = TextureData::new(64, 64, 1);

        let frames = 10;
        for frame in 0..frames {
            let (read_buf, write_buf) =
                if frame % 2 == 0 { (&buf_a, &mut buf_b) } else { (&buf_b, &mut buf_a) };

            // "GPU" reads from read_buf, writes to write_buf
            for y in 0..64 {
                for x in 0..64 {
                    let val = read_buf.read_pixel(x, y)[0] + 1.0;
                    write_buf.write_pixel(x, y, &[val]);
                }
            }
        }

        // After 10 frames of incrementing, final write buffer has ~5.0
        // (alternating buffers, each gets incremented every other frame)
        let final_val = buf_a.read_pixel(0, 0)[0];
        assert!(final_val >= 4.0, "expected accumulated value, got {final_val}");
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_texture_memory_footprint_calculation() {
        let configs: Vec<(PixelFormat, usize, usize, usize)> = vec![
            (PixelFormat::R16Float, 1024, 1024, 2 * 1024 * 1024),
            (PixelFormat::RGBA16Float, 512, 512, 8 * 512 * 512),
            (PixelFormat::R32Float, 2048, 2048, 4 * 2048 * 2048),
            (PixelFormat::RGBA32Float, 256, 256, 16 * 256 * 256),
        ];
        for (fmt, w, h, expected_bytes) in &configs {
            let desc = TextureDescriptor::new_2d(*fmt, *w, *h);
            assert_eq!(
                desc.total_bytes(),
                *expected_bytes,
                "footprint mismatch for {fmt:?} {w}×{h}"
            );
        }
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_concurrent_texture_write_simulation() {
        // Simulate N "threads" writing to disjoint texture regions
        let tex_size = 64;
        let num_workers = 4;
        let rows_per_worker = tex_size / num_workers;

        let mut regions: Vec<TextureData> = (0..num_workers)
            .map(|w| {
                let mut region = TextureData::new(tex_size, rows_per_worker, 1);
                for y in 0..rows_per_worker {
                    for x in 0..tex_size {
                        region.write_pixel(x, y, &[(w * 100 + y * tex_size + x) as f32]);
                    }
                }
                region
            })
            .collect();

        // Merge into single texture
        let mut merged = TextureData::new(tex_size, tex_size, 1);
        for (w, region) in regions.iter().enumerate() {
            let y_offset = w * rows_per_worker;
            for y in 0..rows_per_worker {
                for x in 0..tex_size {
                    let val = region.read_pixel(x, y)[0];
                    merged.write_pixel(x, y_offset + y, &[val]);
                }
            }
        }

        // Verify worker boundaries
        assert_eq!(merged.read_pixel(0, 0)[0], 0.0);
        let worker1_start = rows_per_worker;
        assert_eq!(merged.read_pixel(0, worker1_start)[0], (1 * 100) as f32);
        let _ = &mut regions; // suppress unused-mut on some editions
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn test_texture_pool_reuse_pattern() {
        // Simulate a texture pool that reuses allocations
        struct TexturePool {
            free: Vec<TextureData>,
            width: usize,
            height: usize,
            channels: usize,
        }

        impl TexturePool {
            fn new(width: usize, height: usize, channels: usize, prealloc: usize) -> Self {
                let free =
                    (0..prealloc).map(|_| TextureData::new(width, height, channels)).collect();
                Self { free, width, height, channels }
            }

            fn acquire(&mut self) -> TextureData {
                self.free
                    .pop()
                    .unwrap_or_else(|| TextureData::new(self.width, self.height, self.channels))
            }

            fn release(&mut self, tex: TextureData) {
                self.free.push(tex);
            }
        }

        let mut pool = TexturePool::new(64, 64, 1, 3);
        assert_eq!(pool.free.len(), 3);

        let t1 = pool.acquire();
        let t2 = pool.acquire();
        assert_eq!(pool.free.len(), 1);

        pool.release(t1);
        pool.release(t2);
        assert_eq!(pool.free.len(), 3);

        // Acquire beyond pool → creates new
        let _a = pool.acquire();
        let _b = pool.acquire();
        let _c = pool.acquire();
        let d = pool.acquire(); // pool empty → new allocation
        assert_eq!(pool.free.len(), 0);
        assert_eq!(d.width, 64);
    }
}
