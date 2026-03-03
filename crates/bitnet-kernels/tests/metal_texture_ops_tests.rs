//! Metal texture operation tests using local mock types.
//! No GPU hardware or external bitnet crate dependencies required.

// ---------------------------------------------------------------------------
// Local mock types
// ---------------------------------------------------------------------------

// Minimal bitflags macro (no external dep).
macro_rules! bitflags_local {
    ($Name:ident : $T:ty { $($FLAG:ident = $val:expr,)* }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        struct $Name($T);
        #[allow(dead_code)]
        impl $Name {
            $(const $FLAG: Self = Self($val);)*
            fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }
        }
        impl std::ops::BitOr for $Name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PixelFormat {
    RGBA8Unorm,
    BGRA8Unorm,
    R8Unorm,
    RG8Unorm,
    R16Float,
    RG16Float,
    RGBA16Float,
    R32Float,
    RG32Float,
    RGBA32Float,
    R32Uint,
    R32Sint,
    Depth32Float,
    Stencil8,
    Depth24UnormStencil8,
    Depth32FloatStencil8,
    BC1RGBA,
    BC7RGBA,
    Invalid,
}

impl PixelFormat {
    fn bytes_per_pixel(self) -> Option<usize> {
        match self {
            Self::R8Unorm | Self::Stencil8 => Some(1),
            Self::RG8Unorm | Self::R16Float => Some(2),
            Self::RGBA8Unorm
            | Self::BGRA8Unorm
            | Self::RG16Float
            | Self::R32Float
            | Self::R32Uint
            | Self::R32Sint
            | Self::Depth32Float
            | Self::Depth24UnormStencil8 => Some(4),
            Self::RGBA16Float | Self::RG32Float | Self::Depth32FloatStencil8 => Some(8),
            Self::RGBA32Float => Some(16),
            // Compressed and invalid formats have no simple per-pixel size.
            Self::BC1RGBA | Self::BC7RGBA | Self::Invalid => None,
        }
    }

    fn is_depth(self) -> bool {
        matches!(self, Self::Depth32Float | Self::Depth24UnormStencil8 | Self::Depth32FloatStencil8)
    }

    fn is_stencil(self) -> bool {
        matches!(self, Self::Stencil8 | Self::Depth24UnormStencil8 | Self::Depth32FloatStencil8)
    }

    fn is_compressed(self) -> bool {
        matches!(self, Self::BC1RGBA | Self::BC7RGBA)
    }

    fn is_color(self) -> bool {
        !self.is_depth() && !self.is_stencil() && self != Self::Invalid
    }

    fn supports_render_target(self) -> bool {
        self.is_color() && !self.is_compressed()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextureType {
    Texture2D,
    Texture3D,
    TextureCube,
    Texture2DArray,
    TextureCubeArray,
    Texture2DMultisample,
    Texture1D,
}

bitflags_local! {
    TextureUsage: u32 {
        READ          = 0b0001,
        WRITE         = 0b0010,
        RENDER_TARGET = 0b0100,
        SHADER_READ   = 0b1000,
    }
}

impl TextureUsage {
    fn is_valid(self) -> bool {
        self.0 != 0 && self.0 & 0b1111 == self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MipFilterMode {
    NotMipmapped,
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
    ClampToZero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SwizzleChannel {
    Red,
    Green,
    Blue,
    Alpha,
    Zero,
    One,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TextureSwizzle {
    r: SwizzleChannel,
    g: SwizzleChannel,
    b: SwizzleChannel,
    a: SwizzleChannel,
}

impl TextureSwizzle {
    fn identity() -> Self {
        Self {
            r: SwizzleChannel::Red,
            g: SwizzleChannel::Green,
            b: SwizzleChannel::Blue,
            a: SwizzleChannel::Alpha,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageMode {
    Shared,
    Private,
    Managed,
}

#[derive(Debug, Clone)]
struct TextureDescriptor {
    texture_type: TextureType,
    pixel_format: PixelFormat,
    width: u32,
    height: u32,
    depth: u32,
    mip_levels: u32,
    array_length: u32,
    sample_count: u32,
    usage: TextureUsage,
    storage_mode: StorageMode,
}

impl TextureDescriptor {
    fn new_2d(format: PixelFormat, width: u32, height: u32) -> Self {
        Self {
            texture_type: TextureType::Texture2D,
            pixel_format: format,
            width,
            height,
            depth: 1,
            mip_levels: 1,
            array_length: 1,
            sample_count: 1,
            usage: TextureUsage(TextureUsage::READ.0 | TextureUsage::WRITE.0),
            storage_mode: StorageMode::Shared,
        }
    }

    fn new_3d(format: PixelFormat, width: u32, height: u32, depth: u32) -> Self {
        Self {
            texture_type: TextureType::Texture3D,
            pixel_format: format,
            width,
            height,
            depth,
            mip_levels: 1,
            array_length: 1,
            sample_count: 1,
            usage: TextureUsage(TextureUsage::READ.0 | TextureUsage::WRITE.0),
            storage_mode: StorageMode::Shared,
        }
    }

    fn new_cube(format: PixelFormat, size: u32) -> Self {
        Self {
            texture_type: TextureType::TextureCube,
            pixel_format: format,
            width: size,
            height: size,
            depth: 1,
            mip_levels: 1,
            array_length: 6,
            sample_count: 1,
            usage: TextureUsage(TextureUsage::READ.0 | TextureUsage::SHADER_READ.0),
            storage_mode: StorageMode::Private,
        }
    }

    fn new_2d_array(format: PixelFormat, width: u32, height: u32, array_length: u32) -> Self {
        Self {
            texture_type: TextureType::Texture2DArray,
            pixel_format: format,
            width,
            height,
            depth: 1,
            mip_levels: 1,
            array_length,
            sample_count: 1,
            usage: TextureUsage(TextureUsage::READ.0 | TextureUsage::WRITE.0),
            storage_mode: StorageMode::Shared,
        }
    }

    fn max_mip_levels(&self) -> u32 {
        let max_dim = self.width.max(self.height).max(self.depth);
        if max_dim == 0 {
            return 0;
        }
        (max_dim as f64).log2().floor() as u32 + 1
    }

    fn validate(&self) -> Result<(), TextureError> {
        if self.width == 0 || self.height == 0 {
            return Err(TextureError::InvalidDimensions {
                width: self.width,
                height: self.height,
                depth: self.depth,
            });
        }
        if self.pixel_format == PixelFormat::Invalid {
            return Err(TextureError::UnsupportedFormat(self.pixel_format));
        }
        if self.texture_type == TextureType::Texture3D && self.depth == 0 {
            return Err(TextureError::InvalidDimensions {
                width: self.width,
                height: self.height,
                depth: self.depth,
            });
        }
        if self.texture_type == TextureType::TextureCube && self.width != self.height {
            return Err(TextureError::InvalidDimensions {
                width: self.width,
                height: self.height,
                depth: self.depth,
            });
        }
        if self.mip_levels > self.max_mip_levels() && self.mip_levels != 0 {
            return Err(TextureError::InvalidMipLevels {
                requested: self.mip_levels,
                max: self.max_mip_levels(),
            });
        }
        if !self.usage.is_valid() {
            return Err(TextureError::InvalidUsage);
        }
        if self.width > 16384 || self.height > 16384 {
            return Err(TextureError::DimensionsTooLarge {
                width: self.width,
                height: self.height,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct MockTexture {
    descriptor: TextureDescriptor,
    id: u64,
    data: Vec<u8>,
}

impl MockTexture {
    fn create(desc: TextureDescriptor) -> Result<Self, TextureError> {
        desc.validate()?;
        let bpp = desc.pixel_format.bytes_per_pixel().unwrap_or(1) as u64;
        let total: u64 = desc.width as u64 * desc.height as u64 * desc.depth as u64 * bpp;
        if total > 512 * 1024 * 1024 {
            return Err(TextureError::OutOfMemory { requested: total as usize });
        }
        let data = vec![0u8; total as usize];
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Self { descriptor: desc, id, data })
    }

    fn width(&self) -> u32 {
        self.descriptor.width
    }
    fn height(&self) -> u32 {
        self.descriptor.height
    }
    fn depth(&self) -> u32 {
        self.descriptor.depth
    }
    fn pixel_format(&self) -> PixelFormat {
        self.descriptor.pixel_format
    }
    fn texture_type(&self) -> TextureType {
        self.descriptor.texture_type
    }
    fn mip_levels(&self) -> u32 {
        self.descriptor.mip_levels
    }
    fn usage(&self) -> TextureUsage {
        self.descriptor.usage
    }
    fn storage_mode(&self) -> StorageMode {
        self.descriptor.storage_mode
    }
    fn array_length(&self) -> u32 {
        self.descriptor.array_length
    }
    fn sample_count(&self) -> u32 {
        self.descriptor.sample_count
    }

    fn row_pitch(&self) -> usize {
        let bpp = self.descriptor.pixel_format.bytes_per_pixel().unwrap_or(1);
        let raw = self.descriptor.width as usize * bpp;
        align_up(raw, 256)
    }

    fn slice_pitch(&self) -> usize {
        self.row_pitch() * self.descriptor.height as usize
    }

    fn mip_width(&self, level: u32) -> u32 {
        (self.descriptor.width >> level).max(1)
    }
    fn mip_height(&self, level: u32) -> u32 {
        (self.descriptor.height >> level).max(1)
    }
    fn mip_depth(&self, level: u32) -> u32 {
        (self.descriptor.depth >> level).max(1)
    }

    fn write_pixel(&mut self, x: u32, y: u32, pixel: &[u8]) -> Result<(), TextureError> {
        if !self.descriptor.usage.contains(TextureUsage::WRITE) {
            return Err(TextureError::UsageNotPermitted);
        }
        let bpp = self
            .descriptor
            .pixel_format
            .bytes_per_pixel()
            .ok_or(TextureError::UnsupportedFormat(self.descriptor.pixel_format))?;
        if pixel.len() != bpp {
            return Err(TextureError::InvalidData);
        }
        if x >= self.descriptor.width || y >= self.descriptor.height {
            return Err(TextureError::OutOfBounds);
        }
        let offset = (y as usize * self.descriptor.width as usize + x as usize) * bpp;
        self.data[offset..offset + bpp].copy_from_slice(pixel);
        Ok(())
    }

    fn read_pixel(&self, x: u32, y: u32) -> Result<Vec<u8>, TextureError> {
        if !self.descriptor.usage.contains(TextureUsage::READ) {
            return Err(TextureError::UsageNotPermitted);
        }
        let bpp = self
            .descriptor
            .pixel_format
            .bytes_per_pixel()
            .ok_or(TextureError::UnsupportedFormat(self.descriptor.pixel_format))?;
        if x >= self.descriptor.width || y >= self.descriptor.height {
            return Err(TextureError::OutOfBounds);
        }
        let offset = (y as usize * self.descriptor.width as usize + x as usize) * bpp;
        Ok(self.data[offset..offset + bpp].to_vec())
    }

    fn new_view(
        &self,
        format: PixelFormat,
        swizzle: TextureSwizzle,
    ) -> Result<TextureView, TextureError> {
        // Format reinterpretation requires same bytes-per-pixel.
        let src_bpp = self
            .descriptor
            .pixel_format
            .bytes_per_pixel()
            .ok_or(TextureError::UnsupportedFormat(self.descriptor.pixel_format))?;
        let dst_bpp = format.bytes_per_pixel().ok_or(TextureError::UnsupportedFormat(format))?;
        if src_bpp != dst_bpp {
            return Err(TextureError::IncompatibleViewFormat);
        }
        Ok(TextureView {
            parent_id: self.id,
            format,
            swizzle,
            base_mip: 0,
            mip_count: self.descriptor.mip_levels,
            base_layer: 0,
            layer_count: self.descriptor.array_length,
        })
    }

    fn new_subresource_view(
        &self,
        base_mip: u32,
        mip_count: u32,
        base_layer: u32,
        layer_count: u32,
    ) -> Result<TextureView, TextureError> {
        if base_mip + mip_count > self.descriptor.mip_levels {
            return Err(TextureError::InvalidMipLevels {
                requested: base_mip + mip_count,
                max: self.descriptor.mip_levels,
            });
        }
        if base_layer + layer_count > self.descriptor.array_length {
            return Err(TextureError::OutOfBounds);
        }
        Ok(TextureView {
            parent_id: self.id,
            format: self.descriptor.pixel_format,
            swizzle: TextureSwizzle::identity(),
            base_mip,
            mip_count,
            base_layer,
            layer_count,
        })
    }
}

#[derive(Debug, Clone)]
struct TextureView {
    parent_id: u64,
    format: PixelFormat,
    swizzle: TextureSwizzle,
    base_mip: u32,
    mip_count: u32,
    base_layer: u32,
    layer_count: u32,
}

#[derive(Debug, Clone)]
struct SamplerDescriptor {
    min_filter: FilterMode,
    mag_filter: FilterMode,
    mip_filter: MipFilterMode,
    address_mode_s: AddressMode,
    address_mode_t: AddressMode,
    address_mode_r: AddressMode,
    max_anisotropy: u32,
    lod_min_clamp: f32,
    lod_max_clamp: f32,
}

impl SamplerDescriptor {
    fn nearest() -> Self {
        Self {
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            mip_filter: MipFilterMode::NotMipmapped,
            address_mode_s: AddressMode::ClampToEdge,
            address_mode_t: AddressMode::ClampToEdge,
            address_mode_r: AddressMode::ClampToEdge,
            max_anisotropy: 1,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1000.0,
        }
    }

    fn linear() -> Self {
        Self {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mip_filter: MipFilterMode::Linear,
            address_mode_s: AddressMode::ClampToEdge,
            address_mode_t: AddressMode::ClampToEdge,
            address_mode_r: AddressMode::ClampToEdge,
            max_anisotropy: 1,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1000.0,
        }
    }

    fn validate(&self) -> Result<(), TextureError> {
        if self.max_anisotropy == 0 || self.max_anisotropy > 16 {
            return Err(TextureError::InvalidSamplerConfig);
        }
        if self.lod_min_clamp < 0.0 {
            return Err(TextureError::InvalidSamplerConfig);
        }
        if self.lod_max_clamp < self.lod_min_clamp {
            return Err(TextureError::InvalidSamplerConfig);
        }
        if self.max_anisotropy > 1 && self.min_filter == FilterMode::Nearest {
            return Err(TextureError::InvalidSamplerConfig);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct MockSampler {
    descriptor: SamplerDescriptor,
    id: u64,
}

impl MockSampler {
    fn create(desc: SamplerDescriptor) -> Result<Self, TextureError> {
        desc.validate()?;
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Self { descriptor: desc, id })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CopyRegion {
    src_x: u32,
    src_y: u32,
    src_z: u32,
    dst_x: u32,
    dst_y: u32,
    dst_z: u32,
    width: u32,
    height: u32,
    depth: u32,
}

#[derive(Debug, Clone)]
struct BufferTextureCopy {
    buffer_offset: usize,
    bytes_per_row: usize,
    bytes_per_image: usize,
    texture_x: u32,
    texture_y: u32,
    texture_z: u32,
    width: u32,
    height: u32,
    depth: u32,
}

#[derive(Debug, Clone, PartialEq)]
struct ShaderBinding {
    texture_id: u64,
    sampler_id: Option<u64>,
    slot: u32,
    access: BindingAccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindingAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(Debug, Clone, PartialEq)]
enum TextureError {
    InvalidDimensions { width: u32, height: u32, depth: u32 },
    UnsupportedFormat(PixelFormat),
    InvalidMipLevels { requested: u32, max: u32 },
    InvalidUsage,
    OutOfMemory { requested: usize },
    OutOfBounds,
    UsageNotPermitted,
    InvalidData,
    IncompatibleViewFormat,
    InvalidSamplerConfig,
    FormatMismatch,
    DimensionsTooLarge { width: u32, height: u32 },
    CopyRegionOutOfBounds,
    InvalidBinding,
}

fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

fn compute_mip_size(base: u32, level: u32) -> u32 {
    (base >> level).max(1)
}

fn validate_copy_region(
    src: &MockTexture,
    dst: &MockTexture,
    region: &CopyRegion,
) -> Result<(), TextureError> {
    if src.pixel_format() != dst.pixel_format() {
        return Err(TextureError::FormatMismatch);
    }
    if region.src_x + region.width > src.width() || region.src_y + region.height > src.height() {
        return Err(TextureError::CopyRegionOutOfBounds);
    }
    if region.dst_x + region.width > dst.width() || region.dst_y + region.height > dst.height() {
        return Err(TextureError::CopyRegionOutOfBounds);
    }
    Ok(())
}

fn validate_buffer_texture_copy(
    buffer_len: usize,
    copy: &BufferTextureCopy,
    tex: &MockTexture,
) -> Result<(), TextureError> {
    let bpp = tex
        .pixel_format()
        .bytes_per_pixel()
        .ok_or(TextureError::UnsupportedFormat(tex.pixel_format()))?;
    let required = copy.buffer_offset
        + (copy.height as usize - 1) * copy.bytes_per_row
        + copy.width as usize * bpp;
    if required > buffer_len {
        return Err(TextureError::OutOfBounds);
    }
    if copy.texture_x + copy.width > tex.width() || copy.texture_y + copy.height > tex.height() {
        return Err(TextureError::CopyRegionOutOfBounds);
    }
    Ok(())
}

fn validate_binding(binding: &ShaderBinding, max_slot: u32) -> Result<(), TextureError> {
    if binding.slot > max_slot {
        return Err(TextureError::InvalidBinding);
    }
    if binding.texture_id == 0 {
        return Err(TextureError::InvalidBinding);
    }
    Ok(())
}

fn compute_total_mip_storage(width: u32, height: u32, bpp: usize, levels: u32) -> usize {
    let mut total = 0usize;
    for l in 0..levels {
        let w = compute_mip_size(width, l) as usize;
        let h = compute_mip_size(height, l) as usize;
        total += w * h * bpp;
    }
    total
}

// =========================================================================
// Tests
// =========================================================================

// -------------------------------------------------------------------------
// 1. Texture Creation (12 tests)
// -------------------------------------------------------------------------

#[test]
fn test_texture_2d_rgba8_creation() {
    let desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 256, 256);
    let tex = MockTexture::create(desc).unwrap();
    assert_eq!(tex.width(), 256);
    assert_eq!(tex.height(), 256);
    assert_eq!(tex.depth(), 1);
    assert_eq!(tex.texture_type(), TextureType::Texture2D);
}

#[test]
fn test_texture_2d_non_square_creation() {
    let desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 512, 128);
    let tex = MockTexture::create(desc).unwrap();
    assert_eq!(tex.width(), 512);
    assert_eq!(tex.height(), 128);
}

#[test]
fn test_texture_3d_creation() {
    let desc = TextureDescriptor::new_3d(PixelFormat::RGBA8Unorm, 64, 64, 32);
    let tex = MockTexture::create(desc).unwrap();
    assert_eq!(tex.depth(), 32);
    assert_eq!(tex.texture_type(), TextureType::Texture3D);
}

#[test]
fn test_texture_cube_creation() {
    let desc = TextureDescriptor::new_cube(PixelFormat::RGBA8Unorm, 128);
    let tex = MockTexture::create(desc).unwrap();
    assert_eq!(tex.width(), 128);
    assert_eq!(tex.height(), 128);
    assert_eq!(tex.array_length(), 6);
    assert_eq!(tex.texture_type(), TextureType::TextureCube);
}

#[test]
fn test_texture_2d_array_creation() {
    let desc = TextureDescriptor::new_2d_array(PixelFormat::RGBA8Unorm, 64, 64, 8);
    let tex = MockTexture::create(desc).unwrap();
    assert_eq!(tex.array_length(), 8);
    assert_eq!(tex.texture_type(), TextureType::Texture2DArray);
}

#[test]
fn test_texture_1x1_creation() {
    let desc = TextureDescriptor::new_2d(PixelFormat::R8Unorm, 1, 1);
    let tex = MockTexture::create(desc).unwrap();
    assert_eq!(tex.width(), 1);
    assert_eq!(tex.height(), 1);
    assert_eq!(tex.data.len(), 1);
}

#[test]
fn test_texture_unique_ids() {
    let a = MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)).unwrap();
    let b = MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)).unwrap();
    assert_ne!(a.id, b.id);
}

#[test]
fn test_texture_data_initialized_to_zero() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 8, 8)).unwrap();
    assert!(tex.data.iter().all(|&b| b == 0));
}

#[test]
fn test_texture_data_length() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 16, 16)).unwrap();
    assert_eq!(tex.data.len(), 16 * 16 * 4);
}

#[test]
fn test_texture_cube_requires_square() {
    let mut desc = TextureDescriptor::new_cube(PixelFormat::RGBA8Unorm, 64);
    desc.height = 32; // break square requirement
    let err = MockTexture::create(desc).unwrap_err();
    assert!(matches!(err, TextureError::InvalidDimensions { .. }));
}

#[test]
fn test_texture_private_storage_mode() {
    let desc = TextureDescriptor::new_cube(PixelFormat::RGBA8Unorm, 64);
    let tex = MockTexture::create(desc).unwrap();
    assert_eq!(tex.storage_mode(), StorageMode::Private);
}

#[test]
fn test_texture_shared_storage_mode() {
    let desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64);
    let tex = MockTexture::create(desc).unwrap();
    assert_eq!(tex.storage_mode(), StorageMode::Shared);
}

// -------------------------------------------------------------------------
// 2. Texture Format (12 tests)
// -------------------------------------------------------------------------

#[test]
fn test_format_rgba8_bytes() {
    assert_eq!(PixelFormat::RGBA8Unorm.bytes_per_pixel(), Some(4));
}

#[test]
fn test_format_bgra8_bytes() {
    assert_eq!(PixelFormat::BGRA8Unorm.bytes_per_pixel(), Some(4));
}

#[test]
fn test_format_r16float_bytes() {
    assert_eq!(PixelFormat::R16Float.bytes_per_pixel(), Some(2));
}

#[test]
fn test_format_rg32float_bytes() {
    assert_eq!(PixelFormat::RG32Float.bytes_per_pixel(), Some(8));
}

#[test]
fn test_format_rgba32float_bytes() {
    assert_eq!(PixelFormat::RGBA32Float.bytes_per_pixel(), Some(16));
}

#[test]
fn test_format_r8_bytes() {
    assert_eq!(PixelFormat::R8Unorm.bytes_per_pixel(), Some(1));
}

#[test]
fn test_format_invalid_no_size() {
    assert_eq!(PixelFormat::Invalid.bytes_per_pixel(), None);
}

#[test]
fn test_format_compressed_no_per_pixel_size() {
    assert!(PixelFormat::BC1RGBA.bytes_per_pixel().is_none());
    assert!(PixelFormat::BC7RGBA.bytes_per_pixel().is_none());
}

#[test]
fn test_format_depth32_is_depth() {
    assert!(PixelFormat::Depth32Float.is_depth());
    assert!(!PixelFormat::RGBA8Unorm.is_depth());
}

#[test]
fn test_format_stencil8_is_stencil() {
    assert!(PixelFormat::Stencil8.is_stencil());
    assert!(!PixelFormat::R32Float.is_stencil());
}

#[test]
fn test_format_depth_stencil_combined() {
    let fmt = PixelFormat::Depth32FloatStencil8;
    assert!(fmt.is_depth());
    assert!(fmt.is_stencil());
}

#[test]
fn test_format_is_color() {
    assert!(PixelFormat::RGBA8Unorm.is_color());
    assert!(PixelFormat::R32Float.is_color());
    assert!(!PixelFormat::Depth32Float.is_color());
    assert!(!PixelFormat::Invalid.is_color());
}

// -------------------------------------------------------------------------
// 3. Texture Usage (11 tests)
// -------------------------------------------------------------------------

#[test]
fn test_usage_read_flag() {
    let u = TextureUsage::READ;
    assert!(u.contains(TextureUsage::READ));
    assert!(!u.contains(TextureUsage::WRITE));
}

#[test]
fn test_usage_write_flag() {
    let u = TextureUsage::WRITE;
    assert!(u.contains(TextureUsage::WRITE));
}

#[test]
fn test_usage_combined_read_write() {
    let u = TextureUsage::READ | TextureUsage::WRITE;
    assert!(u.contains(TextureUsage::READ));
    assert!(u.contains(TextureUsage::WRITE));
}

#[test]
fn test_usage_render_target_flag() {
    let u = TextureUsage::RENDER_TARGET;
    assert!(u.contains(TextureUsage::RENDER_TARGET));
    assert!(!u.contains(TextureUsage::READ));
}

#[test]
fn test_usage_shader_read_flag() {
    let u = TextureUsage::SHADER_READ;
    assert!(u.contains(TextureUsage::SHADER_READ));
}

#[test]
fn test_usage_all_flags_combined() {
    let u = TextureUsage::READ
        | TextureUsage::WRITE
        | TextureUsage::RENDER_TARGET
        | TextureUsage::SHADER_READ;
    assert!(u.contains(TextureUsage::READ));
    assert!(u.contains(TextureUsage::WRITE));
    assert!(u.contains(TextureUsage::RENDER_TARGET));
    assert!(u.contains(TextureUsage::SHADER_READ));
}

#[test]
fn test_usage_empty_is_invalid() {
    let u = TextureUsage(0);
    assert!(!u.is_valid());
}

#[test]
fn test_usage_valid_single() {
    assert!(TextureUsage::READ.is_valid());
    assert!(TextureUsage::WRITE.is_valid());
}

#[test]
fn test_usage_render_target_requires_color_format() {
    assert!(PixelFormat::RGBA8Unorm.supports_render_target());
    assert!(!PixelFormat::Depth32Float.supports_render_target());
}

#[test]
fn test_usage_compressed_not_render_target() {
    assert!(!PixelFormat::BC1RGBA.supports_render_target());
    assert!(!PixelFormat::BC7RGBA.supports_render_target());
}

#[test]
fn test_usage_write_pixel_requires_write_usage() {
    let desc = TextureDescriptor {
        usage: TextureUsage::READ,
        ..TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)
    };
    let mut tex = MockTexture::create(desc).unwrap();
    let err = tex.write_pixel(0, 0, &[1, 2, 3, 4]).unwrap_err();
    assert_eq!(err, TextureError::UsageNotPermitted);
}

// -------------------------------------------------------------------------
// 4. Mipmap (11 tests)
// -------------------------------------------------------------------------

#[test]
fn test_mipmap_max_levels_256() {
    let desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 256, 256);
    assert_eq!(desc.max_mip_levels(), 9); // log2(256)+1
}

#[test]
fn test_mipmap_max_levels_1024x512() {
    let desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 1024, 512);
    assert_eq!(desc.max_mip_levels(), 11); // log2(1024)+1
}

#[test]
fn test_mipmap_max_levels_1x1() {
    let desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 1, 1);
    assert_eq!(desc.max_mip_levels(), 1);
}

#[test]
fn test_mipmap_level_dimensions() {
    let desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 128, 128);
    let mut d = desc.clone();
    d.mip_levels = 8;
    let tex = MockTexture::create(d).unwrap();
    assert_eq!(tex.mip_width(0), 128);
    assert_eq!(tex.mip_width(1), 64);
    assert_eq!(tex.mip_width(2), 32);
    assert_eq!(tex.mip_width(7), 1);
}

#[test]
fn test_mipmap_height_halves() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 256)).unwrap();
    assert_eq!(tex.mip_height(0), 256);
    assert_eq!(tex.mip_height(1), 128);
    assert_eq!(tex.mip_height(3), 32);
}

#[test]
fn test_mipmap_depth_3d() {
    let tex =
        MockTexture::create(TextureDescriptor::new_3d(PixelFormat::R32Float, 32, 32, 16)).unwrap();
    assert_eq!(tex.mip_depth(0), 16);
    assert_eq!(tex.mip_depth(1), 8);
    assert_eq!(tex.mip_depth(4), 1);
}

#[test]
fn test_mipmap_never_zero() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)).unwrap();
    for l in 0..20 {
        assert!(tex.mip_width(l) >= 1);
        assert!(tex.mip_height(l) >= 1);
    }
}

#[test]
fn test_mipmap_too_many_levels_rejected() {
    let mut desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64);
    desc.mip_levels = 20; // max is 7
    let err = MockTexture::create(desc).unwrap_err();
    assert!(matches!(err, TextureError::InvalidMipLevels { .. }));
}

#[test]
fn test_mipmap_total_storage() {
    // 256×256 RGBA8, 9 mip levels
    let total = compute_total_mip_storage(256, 256, 4, 9);
    // level sizes: 256²·4 + 128²·4 + 64²·4 + … + 1²·4
    let expected: usize = (0..9)
        .map(|l| compute_mip_size(256, l) as usize * compute_mip_size(256, l) as usize * 4)
        .sum();
    assert_eq!(total, expected);
}

#[test]
fn test_mipmap_compute_mip_size_utility() {
    assert_eq!(compute_mip_size(512, 0), 512);
    assert_eq!(compute_mip_size(512, 1), 256);
    assert_eq!(compute_mip_size(512, 9), 1);
    assert_eq!(compute_mip_size(512, 30), 1);
}

#[test]
fn test_mipmap_non_power_of_two() {
    assert_eq!(compute_mip_size(300, 1), 150);
    assert_eq!(compute_mip_size(300, 2), 75);
    assert_eq!(compute_mip_size(300, 3), 37);
}

// -------------------------------------------------------------------------
// 5. Texture Sampling (11 tests)
// -------------------------------------------------------------------------

#[test]
fn test_sampler_nearest_creation() {
    let s = MockSampler::create(SamplerDescriptor::nearest()).unwrap();
    assert_eq!(s.descriptor.min_filter, FilterMode::Nearest);
    assert_eq!(s.descriptor.mag_filter, FilterMode::Nearest);
}

#[test]
fn test_sampler_linear_creation() {
    let s = MockSampler::create(SamplerDescriptor::linear()).unwrap();
    assert_eq!(s.descriptor.min_filter, FilterMode::Linear);
    assert_eq!(s.descriptor.mag_filter, FilterMode::Linear);
}

#[test]
fn test_sampler_bilinear_mip_linear() {
    let desc =
        SamplerDescriptor { mip_filter: MipFilterMode::Linear, ..SamplerDescriptor::linear() };
    let s = MockSampler::create(desc).unwrap();
    assert_eq!(s.descriptor.mip_filter, MipFilterMode::Linear);
}

#[test]
fn test_sampler_anisotropic_valid() {
    let desc = SamplerDescriptor { max_anisotropy: 8, ..SamplerDescriptor::linear() };
    let s = MockSampler::create(desc).unwrap();
    assert_eq!(s.descriptor.max_anisotropy, 8);
}

#[test]
fn test_sampler_anisotropy_max_16() {
    let desc = SamplerDescriptor { max_anisotropy: 16, ..SamplerDescriptor::linear() };
    assert!(MockSampler::create(desc).is_ok());
}

#[test]
fn test_sampler_anisotropy_over_16_rejected() {
    let desc = SamplerDescriptor { max_anisotropy: 17, ..SamplerDescriptor::linear() };
    assert!(MockSampler::create(desc).is_err());
}

#[test]
fn test_sampler_anisotropy_zero_rejected() {
    let desc = SamplerDescriptor { max_anisotropy: 0, ..SamplerDescriptor::linear() };
    let err = MockSampler::create(desc).unwrap_err();
    assert_eq!(err, TextureError::InvalidSamplerConfig);
}

#[test]
fn test_sampler_anisotropy_requires_linear() {
    let desc = SamplerDescriptor { max_anisotropy: 4, ..SamplerDescriptor::nearest() };
    let err = MockSampler::create(desc).unwrap_err();
    assert_eq!(err, TextureError::InvalidSamplerConfig);
}

#[test]
fn test_sampler_address_modes() {
    let desc = SamplerDescriptor {
        address_mode_s: AddressMode::Repeat,
        address_mode_t: AddressMode::MirrorRepeat,
        address_mode_r: AddressMode::ClampToZero,
        ..SamplerDescriptor::linear()
    };
    let s = MockSampler::create(desc).unwrap();
    assert_eq!(s.descriptor.address_mode_s, AddressMode::Repeat);
    assert_eq!(s.descriptor.address_mode_t, AddressMode::MirrorRepeat);
    assert_eq!(s.descriptor.address_mode_r, AddressMode::ClampToZero);
}

#[test]
fn test_sampler_lod_clamp_valid() {
    let desc = SamplerDescriptor {
        lod_min_clamp: 0.0,
        lod_max_clamp: 10.0,
        ..SamplerDescriptor::linear()
    };
    assert!(MockSampler::create(desc).is_ok());
}

#[test]
fn test_sampler_lod_min_negative_rejected() {
    let desc = SamplerDescriptor { lod_min_clamp: -1.0, ..SamplerDescriptor::linear() };
    let err = MockSampler::create(desc).unwrap_err();
    assert_eq!(err, TextureError::InvalidSamplerConfig);
}

// -------------------------------------------------------------------------
// 6. Texture Copy (11 tests)
// -------------------------------------------------------------------------

#[test]
fn test_copy_texture_to_texture_same_format() {
    let src =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64)).unwrap();
    let dst =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64)).unwrap();
    let region = CopyRegion {
        src_x: 0,
        src_y: 0,
        src_z: 0,
        dst_x: 0,
        dst_y: 0,
        dst_z: 0,
        width: 64,
        height: 64,
        depth: 1,
    };
    assert!(validate_copy_region(&src, &dst, &region).is_ok());
}

#[test]
fn test_copy_partial_region() {
    let src =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 128, 128)).unwrap();
    let dst =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64)).unwrap();
    let region = CopyRegion {
        src_x: 32,
        src_y: 32,
        src_z: 0,
        dst_x: 0,
        dst_y: 0,
        dst_z: 0,
        width: 32,
        height: 32,
        depth: 1,
    };
    assert!(validate_copy_region(&src, &dst, &region).is_ok());
}

#[test]
fn test_copy_format_mismatch_rejected() {
    let src =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64)).unwrap();
    let dst =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::R32Float, 64, 64)).unwrap();
    let region = CopyRegion {
        src_x: 0,
        src_y: 0,
        src_z: 0,
        dst_x: 0,
        dst_y: 0,
        dst_z: 0,
        width: 64,
        height: 64,
        depth: 1,
    };
    let err = validate_copy_region(&src, &dst, &region).unwrap_err();
    assert_eq!(err, TextureError::FormatMismatch);
}

#[test]
fn test_copy_src_out_of_bounds() {
    let src =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 32, 32)).unwrap();
    let dst =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64)).unwrap();
    let region = CopyRegion {
        src_x: 16,
        src_y: 16,
        src_z: 0,
        dst_x: 0,
        dst_y: 0,
        dst_z: 0,
        width: 32,
        height: 32,
        depth: 1,
    };
    let err = validate_copy_region(&src, &dst, &region).unwrap_err();
    assert_eq!(err, TextureError::CopyRegionOutOfBounds);
}

#[test]
fn test_copy_dst_out_of_bounds() {
    let src =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 128, 128)).unwrap();
    let dst =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 32, 32)).unwrap();
    let region = CopyRegion {
        src_x: 0,
        src_y: 0,
        src_z: 0,
        dst_x: 0,
        dst_y: 0,
        dst_z: 0,
        width: 64,
        height: 64,
        depth: 1,
    };
    let err = validate_copy_region(&src, &dst, &region).unwrap_err();
    assert_eq!(err, TextureError::CopyRegionOutOfBounds);
}

#[test]
fn test_copy_buffer_to_texture_valid() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 16, 16)).unwrap();
    let buffer = vec![0u8; 16 * 16 * 4];
    let copy = BufferTextureCopy {
        buffer_offset: 0,
        bytes_per_row: 16 * 4,
        bytes_per_image: 16 * 16 * 4,
        texture_x: 0,
        texture_y: 0,
        texture_z: 0,
        width: 16,
        height: 16,
        depth: 1,
    };
    assert!(validate_buffer_texture_copy(buffer.len(), &copy, &tex).is_ok());
}

#[test]
fn test_copy_buffer_to_texture_offset() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 8, 8)).unwrap();
    let buffer = vec![0u8; 512 + 8 * 8 * 4];
    let copy = BufferTextureCopy {
        buffer_offset: 512,
        bytes_per_row: 8 * 4,
        bytes_per_image: 8 * 8 * 4,
        texture_x: 0,
        texture_y: 0,
        texture_z: 0,
        width: 8,
        height: 8,
        depth: 1,
    };
    assert!(validate_buffer_texture_copy(buffer.len(), &copy, &tex).is_ok());
}

#[test]
fn test_copy_buffer_too_small() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 16, 16)).unwrap();
    let buffer = [0u8; 100]; // too small
    let copy = BufferTextureCopy {
        buffer_offset: 0,
        bytes_per_row: 16 * 4,
        bytes_per_image: 16 * 16 * 4,
        texture_x: 0,
        texture_y: 0,
        texture_z: 0,
        width: 16,
        height: 16,
        depth: 1,
    };
    let err = validate_buffer_texture_copy(buffer.len(), &copy, &tex).unwrap_err();
    assert_eq!(err, TextureError::OutOfBounds);
}

#[test]
fn test_copy_buffer_texture_region_exceeds() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 16, 16)).unwrap();
    let buffer = vec![0u8; 16 * 16 * 4];
    let copy = BufferTextureCopy {
        buffer_offset: 0,
        bytes_per_row: 16 * 4,
        bytes_per_image: 16 * 16 * 4,
        texture_x: 8,
        texture_y: 8,
        texture_z: 0,
        width: 16, // exceeds texture
        height: 16,
        depth: 1,
    };
    let err = validate_buffer_texture_copy(buffer.len(), &copy, &tex).unwrap_err();
    assert_eq!(err, TextureError::CopyRegionOutOfBounds);
}

#[test]
fn test_copy_zero_size_region_valid() {
    let src =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64)).unwrap();
    let dst =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64)).unwrap();
    let region = CopyRegion {
        src_x: 0,
        src_y: 0,
        src_z: 0,
        dst_x: 0,
        dst_y: 0,
        dst_z: 0,
        width: 0,
        height: 0,
        depth: 1,
    };
    assert!(validate_copy_region(&src, &dst, &region).is_ok());
}

#[test]
fn test_copy_texture_to_buffer_valid() {
    let tex = MockTexture::create(TextureDescriptor::new_2d(PixelFormat::R32Float, 8, 8)).unwrap();
    let buffer = vec![0u8; 8 * 8 * 4];
    let copy = BufferTextureCopy {
        buffer_offset: 0,
        bytes_per_row: 8 * 4,
        bytes_per_image: 8 * 8 * 4,
        texture_x: 0,
        texture_y: 0,
        texture_z: 0,
        width: 8,
        height: 8,
        depth: 1,
    };
    assert!(validate_buffer_texture_copy(buffer.len(), &copy, &tex).is_ok());
}

// -------------------------------------------------------------------------
// 7. Texture Views (11 tests)
// -------------------------------------------------------------------------

#[test]
fn test_view_same_format() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64)).unwrap();
    let view = tex.new_view(PixelFormat::RGBA8Unorm, TextureSwizzle::identity()).unwrap();
    assert_eq!(view.format, PixelFormat::RGBA8Unorm);
    assert_eq!(view.parent_id, tex.id);
}

#[test]
fn test_view_reinterpret_rgba8_to_bgra8() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 32, 32)).unwrap();
    let view = tex.new_view(PixelFormat::BGRA8Unorm, TextureSwizzle::identity()).unwrap();
    assert_eq!(view.format, PixelFormat::BGRA8Unorm);
}

#[test]
fn test_view_incompatible_format_rejected() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 32, 32)).unwrap();
    // RGBA8 is 4 bytes, RGBA32Float is 16 bytes — incompatible.
    let err = tex.new_view(PixelFormat::RGBA32Float, TextureSwizzle::identity()).unwrap_err();
    assert_eq!(err, TextureError::IncompatibleViewFormat);
}

#[test]
fn test_view_swizzle_identity() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 8, 8)).unwrap();
    let view = tex.new_view(PixelFormat::RGBA8Unorm, TextureSwizzle::identity()).unwrap();
    assert_eq!(view.swizzle.r, SwizzleChannel::Red);
    assert_eq!(view.swizzle.g, SwizzleChannel::Green);
    assert_eq!(view.swizzle.b, SwizzleChannel::Blue);
    assert_eq!(view.swizzle.a, SwizzleChannel::Alpha);
}

#[test]
fn test_view_swizzle_custom() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 8, 8)).unwrap();
    let swizzle = TextureSwizzle {
        r: SwizzleChannel::Blue,
        g: SwizzleChannel::Green,
        b: SwizzleChannel::Red,
        a: SwizzleChannel::One,
    };
    let view = tex.new_view(PixelFormat::RGBA8Unorm, swizzle).unwrap();
    assert_eq!(view.swizzle.r, SwizzleChannel::Blue);
    assert_eq!(view.swizzle.a, SwizzleChannel::One);
}

#[test]
fn test_view_subresource_mip_slice() {
    let mut desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 256, 256);
    desc.mip_levels = 5;
    let tex = MockTexture::create(desc).unwrap();
    let view = tex.new_subresource_view(2, 3, 0, 1).unwrap();
    assert_eq!(view.base_mip, 2);
    assert_eq!(view.mip_count, 3);
}

#[test]
fn test_view_subresource_mip_out_of_range() {
    let mut desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64);
    desc.mip_levels = 4;
    let tex = MockTexture::create(desc).unwrap();
    let err = tex.new_subresource_view(3, 3, 0, 1).unwrap_err();
    assert!(matches!(err, TextureError::InvalidMipLevels { .. }));
}

#[test]
fn test_view_subresource_layer_slice() {
    let desc = TextureDescriptor::new_2d_array(PixelFormat::RGBA8Unorm, 32, 32, 10);
    let tex = MockTexture::create(desc).unwrap();
    let view = tex.new_subresource_view(0, 1, 3, 4).unwrap();
    assert_eq!(view.base_layer, 3);
    assert_eq!(view.layer_count, 4);
}

#[test]
fn test_view_subresource_layer_out_of_range() {
    let desc = TextureDescriptor::new_2d_array(PixelFormat::RGBA8Unorm, 32, 32, 4);
    let tex = MockTexture::create(desc).unwrap();
    let err = tex.new_subresource_view(0, 1, 2, 5).unwrap_err();
    assert_eq!(err, TextureError::OutOfBounds);
}

#[test]
fn test_view_preserves_parent_id() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 16, 16)).unwrap();
    let v1 = tex.new_view(PixelFormat::RGBA8Unorm, TextureSwizzle::identity()).unwrap();
    let v2 = tex.new_subresource_view(0, 1, 0, 1).unwrap();
    assert_eq!(v1.parent_id, tex.id);
    assert_eq!(v2.parent_id, tex.id);
}

#[test]
fn test_view_swizzle_zero_channel() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)).unwrap();
    let swizzle = TextureSwizzle {
        r: SwizzleChannel::Zero,
        g: SwizzleChannel::Zero,
        b: SwizzleChannel::Zero,
        a: SwizzleChannel::One,
    };
    let view = tex.new_view(PixelFormat::RGBA8Unorm, swizzle).unwrap();
    assert_eq!(view.swizzle.r, SwizzleChannel::Zero);
}

// -------------------------------------------------------------------------
// 8. Texture Binding (11 tests)
// -------------------------------------------------------------------------

#[test]
fn test_binding_read_only_valid() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 16, 16)).unwrap();
    let binding = ShaderBinding {
        texture_id: tex.id,
        sampler_id: None,
        slot: 0,
        access: BindingAccess::ReadOnly,
    };
    assert!(validate_binding(&binding, 31).is_ok());
}

#[test]
fn test_binding_with_sampler() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 16, 16)).unwrap();
    let sampler = MockSampler::create(SamplerDescriptor::linear()).unwrap();
    let binding = ShaderBinding {
        texture_id: tex.id,
        sampler_id: Some(sampler.id),
        slot: 1,
        access: BindingAccess::ReadOnly,
    };
    assert!(validate_binding(&binding, 31).is_ok());
    assert_eq!(binding.sampler_id, Some(sampler.id));
}

#[test]
fn test_binding_write_only() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::R32Float, 16, 16)).unwrap();
    let binding = ShaderBinding {
        texture_id: tex.id,
        sampler_id: None,
        slot: 5,
        access: BindingAccess::WriteOnly,
    };
    assert!(validate_binding(&binding, 31).is_ok());
    assert_eq!(binding.access, BindingAccess::WriteOnly);
}

#[test]
fn test_binding_read_write() {
    let tex = MockTexture::create(TextureDescriptor::new_2d(PixelFormat::R32Uint, 8, 8)).unwrap();
    let binding = ShaderBinding {
        texture_id: tex.id,
        sampler_id: None,
        slot: 0,
        access: BindingAccess::ReadWrite,
    };
    assert!(validate_binding(&binding, 31).is_ok());
}

#[test]
fn test_binding_slot_exceeds_max() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 8, 8)).unwrap();
    let binding = ShaderBinding {
        texture_id: tex.id,
        sampler_id: None,
        slot: 32,
        access: BindingAccess::ReadOnly,
    };
    let err = validate_binding(&binding, 31).unwrap_err();
    assert_eq!(err, TextureError::InvalidBinding);
}

#[test]
fn test_binding_zero_texture_id_rejected() {
    let binding =
        ShaderBinding { texture_id: 0, sampler_id: None, slot: 0, access: BindingAccess::ReadOnly };
    let err = validate_binding(&binding, 31).unwrap_err();
    assert_eq!(err, TextureError::InvalidBinding);
}

#[test]
fn test_binding_multiple_slots() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 8, 8)).unwrap();
    let bindings: Vec<ShaderBinding> = (0..8)
        .map(|slot| ShaderBinding {
            texture_id: tex.id,
            sampler_id: None,
            slot,
            access: BindingAccess::ReadOnly,
        })
        .collect();
    for b in &bindings {
        assert!(validate_binding(b, 31).is_ok());
    }
    assert_eq!(bindings.len(), 8);
}

#[test]
fn test_binding_max_slot_boundary() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)).unwrap();
    let ok_binding = ShaderBinding {
        texture_id: tex.id,
        sampler_id: None,
        slot: 31,
        access: BindingAccess::ReadOnly,
    };
    assert!(validate_binding(&ok_binding, 31).is_ok());
}

#[test]
fn test_binding_different_textures_same_slot() {
    let t1 = MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)).unwrap();
    let t2 = MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)).unwrap();
    let b1 = ShaderBinding {
        texture_id: t1.id,
        sampler_id: None,
        slot: 0,
        access: BindingAccess::ReadOnly,
    };
    let b2 = ShaderBinding {
        texture_id: t2.id,
        sampler_id: None,
        slot: 0,
        access: BindingAccess::ReadOnly,
    };
    assert_ne!(b1, b2);
}

#[test]
fn test_binding_sampler_state_none() {
    let tex = MockTexture::create(TextureDescriptor::new_2d(PixelFormat::R32Float, 4, 4)).unwrap();
    let binding = ShaderBinding {
        texture_id: tex.id,
        sampler_id: None,
        slot: 0,
        access: BindingAccess::ReadWrite,
    };
    assert!(binding.sampler_id.is_none());
}

#[test]
fn test_binding_equality_check() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)).unwrap();
    let b1 = ShaderBinding {
        texture_id: tex.id,
        sampler_id: None,
        slot: 3,
        access: BindingAccess::ReadOnly,
    };
    let b2 = b1.clone();
    assert_eq!(b1, b2);
}

// -------------------------------------------------------------------------
// 9. Texture Memory (11 tests)
// -------------------------------------------------------------------------

#[test]
fn test_memory_row_pitch_aligned_256() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64)).unwrap();
    let rp = tex.row_pitch();
    assert_eq!(rp % 256, 0);
}

#[test]
fn test_memory_row_pitch_minimum() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 1, 1)).unwrap();
    let rp = tex.row_pitch();
    assert!(rp >= 4); // at least 1 pixel × 4 bpp
    assert_eq!(rp % 256, 0);
}

#[test]
fn test_memory_slice_pitch() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64)).unwrap();
    assert_eq!(tex.slice_pitch(), tex.row_pitch() * 64);
}

#[test]
fn test_memory_align_up_utility() {
    assert_eq!(align_up(1, 256), 256);
    assert_eq!(align_up(256, 256), 256);
    assert_eq!(align_up(257, 256), 512);
    assert_eq!(align_up(0, 256), 0);
}

#[test]
fn test_memory_r8_row_pitch() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::R8Unorm, 100, 100)).unwrap();
    let rp = tex.row_pitch();
    assert!(rp >= 100);
    assert_eq!(rp % 256, 0);
}

#[test]
fn test_memory_rgba32float_row_pitch() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA32Float, 32, 32)).unwrap();
    let raw = 32 * 16; // 32 pixels × 16 bpp
    let expected = align_up(raw, 256);
    assert_eq!(tex.row_pitch(), expected);
}

#[test]
fn test_memory_3d_total_data_size() {
    let tex =
        MockTexture::create(TextureDescriptor::new_3d(PixelFormat::R32Float, 16, 16, 8)).unwrap();
    assert_eq!(tex.data.len(), 16 * 16 * 8 * 4);
}

#[test]
fn test_memory_storage_mode_shared() {
    let desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 32, 32);
    assert_eq!(desc.storage_mode, StorageMode::Shared);
}

#[test]
fn test_memory_storage_mode_private() {
    let desc = TextureDescriptor::new_cube(PixelFormat::RGBA8Unorm, 32);
    assert_eq!(desc.storage_mode, StorageMode::Private);
}

#[test]
fn test_memory_managed_storage_mode() {
    let desc = TextureDescriptor {
        storage_mode: StorageMode::Managed,
        ..TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 16, 16)
    };
    let tex = MockTexture::create(desc).unwrap();
    assert_eq!(tex.storage_mode(), StorageMode::Managed);
}

#[test]
fn test_memory_large_texture_allocation() {
    // 4096×4096 RGBA8 = 64 MiB — should succeed.
    let tex = MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4096, 4096))
        .unwrap();
    assert_eq!(tex.data.len(), 4096 * 4096 * 4);
}

// -------------------------------------------------------------------------
// 10. Error Handling (12 tests)
// -------------------------------------------------------------------------

#[test]
fn test_error_zero_width() {
    let desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 0, 64);
    let err = MockTexture::create(desc).unwrap_err();
    assert!(matches!(err, TextureError::InvalidDimensions { .. }));
}

#[test]
fn test_error_zero_height() {
    let desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 0);
    let err = MockTexture::create(desc).unwrap_err();
    assert!(matches!(err, TextureError::InvalidDimensions { .. }));
}

#[test]
fn test_error_zero_depth_3d() {
    let desc = TextureDescriptor::new_3d(PixelFormat::RGBA8Unorm, 32, 32, 0);
    let err = MockTexture::create(desc).unwrap_err();
    assert!(matches!(err, TextureError::InvalidDimensions { .. }));
}

#[test]
fn test_error_invalid_pixel_format() {
    let desc = TextureDescriptor::new_2d(PixelFormat::Invalid, 32, 32);
    let err = MockTexture::create(desc).unwrap_err();
    assert!(matches!(err, TextureError::UnsupportedFormat(_)));
}

#[test]
fn test_error_dimensions_too_large() {
    let desc = TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 20000, 20000);
    let err = MockTexture::create(desc).unwrap_err();
    assert!(matches!(err, TextureError::DimensionsTooLarge { .. }));
}

#[test]
fn test_error_oom_simulation() {
    // 16384×16384 RGBA32Float = 16384² × 16 = 4 GiB → exceeds 512 MiB
    // limit.
    let desc = TextureDescriptor::new_2d(PixelFormat::RGBA32Float, 16384, 16384);
    let err = MockTexture::create(desc).unwrap_err();
    assert!(matches!(err, TextureError::OutOfMemory { .. }));
}

#[test]
fn test_error_write_out_of_bounds() {
    let mut tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)).unwrap();
    let err = tex.write_pixel(4, 0, &[1, 2, 3, 4]).unwrap_err();
    assert_eq!(err, TextureError::OutOfBounds);
}

#[test]
fn test_error_read_out_of_bounds() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)).unwrap();
    let err = tex.read_pixel(0, 4).unwrap_err();
    assert_eq!(err, TextureError::OutOfBounds);
}

#[test]
fn test_error_write_wrong_pixel_size() {
    let mut tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)).unwrap();
    // Provide 3 bytes instead of 4.
    let err = tex.write_pixel(0, 0, &[1, 2, 3]).unwrap_err();
    assert_eq!(err, TextureError::InvalidData);
}

#[test]
fn test_error_read_without_read_usage() {
    let desc = TextureDescriptor {
        usage: TextureUsage::WRITE,
        ..TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)
    };
    let tex = MockTexture::create(desc).unwrap();
    let err = tex.read_pixel(0, 0).unwrap_err();
    assert_eq!(err, TextureError::UsageNotPermitted);
}

#[test]
fn test_error_empty_usage_rejected() {
    let desc = TextureDescriptor {
        usage: TextureUsage(0),
        ..TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)
    };
    let err = MockTexture::create(desc).unwrap_err();
    assert_eq!(err, TextureError::InvalidUsage);
}

#[test]
fn test_error_sampler_lod_max_less_than_min() {
    let desc =
        SamplerDescriptor { lod_min_clamp: 5.0, lod_max_clamp: 2.0, ..SamplerDescriptor::linear() };
    let err = MockSampler::create(desc).unwrap_err();
    assert_eq!(err, TextureError::InvalidSamplerConfig);
}

// -------------------------------------------------------------------------
// Additional cross-cutting tests to exceed 100 total
// -------------------------------------------------------------------------

#[test]
fn test_pixel_write_then_read_roundtrip() {
    let mut tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 8, 8)).unwrap();
    tex.write_pixel(3, 5, &[10, 20, 30, 255]).unwrap();
    let px = tex.read_pixel(3, 5).unwrap();
    assert_eq!(px, vec![10, 20, 30, 255]);
}

#[test]
fn test_pixel_write_boundary() {
    let mut tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 4, 4)).unwrap();
    tex.write_pixel(3, 3, &[1, 2, 3, 4]).unwrap();
    let px = tex.read_pixel(3, 3).unwrap();
    assert_eq!(px, vec![1, 2, 3, 4]);
}

#[test]
fn test_r32float_data_length() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::R32Float, 10, 10)).unwrap();
    assert_eq!(tex.data.len(), 10 * 10 * 4);
}

#[test]
fn test_rg16float_data_length() {
    let tex = MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RG16Float, 8, 8)).unwrap();
    assert_eq!(tex.data.len(), 8 * 8 * 4);
}

#[test]
fn test_multisample_descriptor() {
    let desc = TextureDescriptor {
        texture_type: TextureType::Texture2DMultisample,
        sample_count: 4,
        ..TextureDescriptor::new_2d(PixelFormat::RGBA8Unorm, 64, 64)
    };
    let tex = MockTexture::create(desc).unwrap();
    assert_eq!(tex.sample_count(), 4);
    assert_eq!(tex.texture_type(), TextureType::Texture2DMultisample);
}

#[test]
fn test_cube_array_texture_type() {
    let desc = TextureDescriptor {
        texture_type: TextureType::TextureCubeArray,
        array_length: 12,
        ..TextureDescriptor::new_cube(PixelFormat::RGBA8Unorm, 32)
    };
    let tex = MockTexture::create(desc).unwrap();
    assert_eq!(tex.texture_type(), TextureType::TextureCubeArray);
    assert_eq!(tex.array_length(), 12);
}

#[test]
fn test_format_r32uint_is_color() {
    assert!(PixelFormat::R32Uint.is_color());
    assert!(!PixelFormat::R32Uint.is_depth());
}

#[test]
fn test_format_r32sint_bytes() {
    assert_eq!(PixelFormat::R32Sint.bytes_per_pixel(), Some(4));
}

#[test]
fn test_format_rg8_creation() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RG8Unorm, 16, 16)).unwrap();
    assert_eq!(tex.data.len(), 16 * 16 * 2);
}

#[test]
fn test_format_rgba16float_data_length() {
    let tex =
        MockTexture::create(TextureDescriptor::new_2d(PixelFormat::RGBA16Float, 4, 4)).unwrap();
    assert_eq!(tex.data.len(), 4 * 4 * 8);
}
