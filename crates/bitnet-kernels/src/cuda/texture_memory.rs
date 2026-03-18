//! CUDA texture memory operations for cache-optimized reads.
//!
//! # Overview
//!
//! CUDA texture memory provides a hardware-managed cache optimized for spatial
//! locality access patterns.  Texture fetches go through a dedicated read-only
//! cache separate from L1/L2, making them ideal for:
//!
//! - **Weight lookups** with irregular access patterns
//! - **2-D convolution** where neighbouring elements are re-read by multiple
//!   threads
//! - **Interpolation** between adjacent values (hardware bilinear filtering)
//! - **Batch embedding / table lookups** with high cache reuse
//!
//! # Kernel strategy
//!
//! Each operation is backed by a CUDA kernel that binds a `cudaTextureObject_t`
//! and issues `tex1Dfetch` / `tex2D` intrinsics.  The texture object is
//! created from a device buffer with configurable filter and address modes so
//! the hardware cache line fetcher can optimise the access pattern.
//!
//! # CPU fallback
//!
//! Every GPU function has a corresponding pure-Rust CPU fallback that
//! replicates the texture semantics (clamping, wrapping, linear filtering)
//! using standard arithmetic, enabling correctness testing on any host.

use bitnet_common::{KernelError, Result};

// ───────────────────────────────────────────────────────────────────
// Configuration types
// ───────────────────────────────────────────────────────────────────

/// Texture filtering mode — controls how out-of-texel reads are interpolated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterMode {
    /// Return the nearest texel value (no interpolation).
    Point,
    /// Linearly interpolate between adjacent texels.
    Linear,
}

impl FilterMode {
    /// CUDA `cudaTextureFilterMode` integer representation.
    #[inline]
    pub fn as_cuda_int(self) -> i32 {
        match self {
            Self::Point => 0,  // cudaFilterModePoint
            Self::Linear => 1, // cudaFilterModeLinear
        }
    }
}

/// Address mode — controls how out-of-bounds coordinates are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressMode {
    /// Clamp coordinates to `[0, size-1]`.
    Clamp,
    /// Wrap coordinates modulo the texture size.
    Wrap,
    /// Mirror coordinates at the boundary.
    Mirror,
    /// Return zero for out-of-bounds coordinates.
    Border,
}

impl AddressMode {
    /// CUDA `cudaTextureAddressMode` integer representation.
    #[inline]
    pub fn as_cuda_int(self) -> i32 {
        match self {
            Self::Clamp => 0,  // cudaAddressModeClamp
            Self::Wrap => 1,   // cudaAddressModeWrap
            Self::Mirror => 2, // cudaAddressModeMirror
            Self::Border => 3, // cudaAddressModeBorder
        }
    }
}

/// Configuration for a CUDA texture object.
#[derive(Debug, Clone)]
pub struct TextureConfig {
    /// Width in elements (1-D length or 2-D width).
    pub width: usize,
    /// Height in elements (1 for 1-D textures).
    pub height: usize,
    /// Filtering mode for reads between texel centres.
    pub filter_mode: FilterMode,
    /// Address mode for each dimension.
    pub address_mode: AddressMode,
    /// Whether coordinates are normalised to `[0, 1)`.
    pub normalized_coords: bool,
    /// Number of channels per texel (1 for scalar, 4 for RGBA-style).
    pub channels: usize,
    /// Read mode — `false` = element type, `true` = normalised float.
    pub read_as_normalized_float: bool,
}

impl TextureConfig {
    /// Create a 1-D texture configuration.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError`] if `width` is zero.
    pub fn new_1d(width: usize) -> Result<Self> {
        if width == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "texture width must be > 0".into(),
            }
            .into());
        }
        Ok(Self {
            width,
            height: 1,
            filter_mode: FilterMode::Point,
            address_mode: AddressMode::Clamp,
            normalized_coords: false,
            channels: 1,
            read_as_normalized_float: false,
        })
    }

    /// Create a 2-D texture configuration.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError`] if either dimension is zero.
    pub fn new_2d(width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "texture width and height must be > 0".into(),
            }
            .into());
        }
        Ok(Self {
            width,
            height,
            filter_mode: FilterMode::Point,
            address_mode: AddressMode::Clamp,
            normalized_coords: false,
            channels: 1,
            read_as_normalized_float: false,
        })
    }

    /// Builder: set the filter mode.
    #[inline]
    pub fn with_filter_mode(mut self, mode: FilterMode) -> Self {
        self.filter_mode = mode;
        self
    }

    /// Builder: set the address mode.
    #[inline]
    pub fn with_address_mode(mut self, mode: AddressMode) -> Self {
        self.address_mode = mode;
        self
    }

    /// Builder: enable or disable normalised coordinates.
    #[inline]
    pub fn with_normalized_coords(mut self, enabled: bool) -> Self {
        self.normalized_coords = enabled;
        self
    }

    /// Builder: set the number of channels.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError`] if `channels` is not 1, 2, or 4.
    pub fn with_channels(mut self, channels: usize) -> Result<Self> {
        if channels != 1 && channels != 2 && channels != 4 {
            return Err(KernelError::InvalidArguments {
                reason: "texture channels must be 1, 2, or 4".into(),
            }
            .into());
        }
        self.channels = channels;
        Ok(self)
    }

    /// Builder: enable normalised-float read mode.
    #[inline]
    pub fn with_normalized_float_read(mut self, enabled: bool) -> Self {
        self.read_as_normalized_float = enabled;
        self
    }

    /// Total number of elements in the texture (width × height × channels).
    #[inline]
    pub fn total_elements(&self) -> usize {
        self.width * self.height * self.channels
    }

    /// Whether this is a 1-D texture.
    #[inline]
    pub fn is_1d(&self) -> bool {
        self.height == 1
    }

    /// CUDA grid dimensions for a kernel covering every texel.
    pub fn grid_dim(&self, threads_per_block: usize) -> (u32, u32, u32) {
        let blocks_x = self.width.div_ceil(threads_per_block);
        (blocks_x as u32, self.height as u32, 1)
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self, threads_per_block: usize) -> (u32, u32, u32) {
        (threads_per_block.min(self.width) as u32, 1, 1)
    }
}

// ───────────────────────────────────────────────────────────────────
// Texture object wrapper
// ───────────────────────────────────────────────────────────────────

/// Opaque handle representing a CUDA texture object.
///
/// On the GPU path this wraps a `cudaTextureObject_t`.  On CPU this stores the
/// raw data buffer together with the [`TextureConfig`] so that CPU fallback
/// fetch functions can replicate texture semantics.
#[derive(Debug, Clone)]
pub struct TextureObject {
    /// Identifier (monotonically increasing per-process).
    pub id: u64,
    /// Configuration that was used to create this texture.
    pub config: TextureConfig,
    /// CPU-side copy of the bound data (used by CPU fallback path).
    pub data: Vec<f32>,
}

impl TextureObject {
    /// Returns the width of the bound texture.
    #[inline]
    pub fn width(&self) -> usize {
        self.config.width
    }

    /// Returns the height of the bound texture.
    #[inline]
    pub fn height(&self) -> usize {
        self.config.height
    }

    /// Returns the number of channels.
    #[inline]
    pub fn channels(&self) -> usize {
        self.config.channels
    }
}

// ───────────────────────────────────────────────────────────────────
// CUDA kernel sources
// ───────────────────────────────────────────────────────────────────

/// CUDA C source for a 1-D texture fetch kernel.
///
/// Grid: `(ceil(n / blockDim.x), 1, 1)`.  Block: `(256, 1, 1)`.
///
/// Each thread fetches one element via `tex1Dfetch` and writes it to the
/// output buffer.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TEXTURE_FETCH_1D_KERNEL_SRC: &str = r#"
extern "C" __global__ void texture_fetch_1d(
    cudaTextureObject_t tex,
    float* __restrict__ output,
    const int* __restrict__ indices,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int idx = indices[tid];
        output[tid] = tex1Dfetch<float>(tex, idx);
    }
}
"#;

/// CUDA C source for a 2-D texture fetch kernel.
///
/// Grid: `(ceil(width / 16), ceil(height / 16), 1)`.  Block: `(16, 16, 1)`.
///
/// Threads fetch via `tex2D<float>` using `(x + 0.5f, y + 0.5f)` for
/// texel-centre addressing.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TEXTURE_FETCH_2D_KERNEL_SRC: &str = r#"
extern "C" __global__ void texture_fetch_2d(
    cudaTextureObject_t tex,
    float* __restrict__ output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        output[y * width + x] = tex2D<float>(tex, (float)x + 0.5f, (float)y + 0.5f);
    }
}
"#;

/// CUDA C source for a texture-based 2-D gather (nearest-neighbour).
///
/// Grid: `(ceil(n / 256), 1, 1)`.  Block: `(256, 1, 1)`.
///
/// Each thread reads one `(x, y)` coordinate pair and gathers the
/// corresponding texel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TEXTURE_GATHER_KERNEL_SRC: &str = r#"
extern "C" __global__ void texture_gather(
    cudaTextureObject_t tex,
    const float* __restrict__ coords_x,
    const float* __restrict__ coords_y,
    float* __restrict__ output,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float x = coords_x[tid];
        float y = coords_y[tid];
        output[tid] = tex2D<float>(tex, x + 0.5f, y + 0.5f);
    }
}
"#;

/// CUDA C source for a texture-based 2-D convolution kernel.
///
/// Grid: `(ceil(width / 16), ceil(height / 16), 1)`.  Block: `(16, 16, 1)`.
///
/// Each thread computes one output pixel by convolving a `kH × kW` filter
/// over the input texture.  Texture clamping automatically handles border
/// pixels.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TEXTURE_CONV2D_KERNEL_SRC: &str = r#"
extern "C" __global__ void texture_conv2d(
    cudaTextureObject_t tex,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int width,
    int height,
    int kW,
    int kH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float sum = 0.0f;
        int half_kW = kW / 2;
        int half_kH = kH / 2;
        for (int ky = 0; ky < kH; ky++) {
            for (int kx = 0; kx < kW; kx++) {
                float val = tex2D<float>(tex, (float)(x - half_kW + kx) + 0.5f,
                                              (float)(y - half_kH + ky) + 0.5f);
                sum += val * kernel[ky * kW + kx];
            }
        }
        output[y * width + x] = sum;
    }
}
"#;

/// CUDA C source for bilinear texture interpolation.
///
/// Grid: `(ceil(n / 256), 1, 1)`.  Block: `(256, 1, 1)`.
///
/// Each thread samples the texture at a fractional coordinate using
/// hardware bilinear filtering (requires `cudaFilterModeLinear`).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TEXTURE_INTERPOLATE_KERNEL_SRC: &str = r#"
extern "C" __global__ void texture_interpolate(
    cudaTextureObject_t tex,
    const float* __restrict__ coords_x,
    const float* __restrict__ coords_y,
    float* __restrict__ output,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = tex2D<float>(tex, coords_x[tid], coords_y[tid]);
    }
}
"#;

/// CUDA C source for batched 1-D texture lookups.
///
/// Grid: `(ceil(total / 256), 1, 1)`.  Block: `(256, 1, 1)`.
///
/// `indices` contains `batch_size × lookup_len` integer indices.  Each
/// thread fetches one value.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const TEXTURE_BATCH_LOOKUP_KERNEL_SRC: &str = r#"
extern "C" __global__ void texture_batch_lookup(
    cudaTextureObject_t tex,
    const int* __restrict__ indices,
    float* __restrict__ output,
    int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        output[tid] = tex1Dfetch<float>(tex, indices[tid]);
    }
}
"#;

// ───────────────────────────────────────────────────────────────────
// GPU launch stubs
// ───────────────────────────────────────────────────────────────────

/// Launch config for 1-D texture fetch on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_texture_fetch_1d(n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    let threads = 256u32;
    let blocks = (n as u32).div_ceil(threads);
    ((blocks, 1, 1), (threads, 1, 1))
}

/// Launch config for 2-D texture fetch on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_texture_fetch_2d(width: usize, height: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    let tx = 16u32;
    let ty = 16u32;
    let bx = (width as u32).div_ceil(tx);
    let by = (height as u32).div_ceil(ty);
    ((bx, by, 1), (tx, ty, 1))
}

/// Launch config for texture gather on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_texture_gather(n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    let threads = 256u32;
    let blocks = (n as u32).div_ceil(threads);
    ((blocks, 1, 1), (threads, 1, 1))
}

/// Launch config for texture-based 2-D convolution on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_texture_conv2d(width: usize, height: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    let tx = 16u32;
    let ty = 16u32;
    let bx = (width as u32).div_ceil(tx);
    let by = (height as u32).div_ceil(ty);
    ((bx, by, 1), (tx, ty, 1))
}

/// Launch config for texture interpolation on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_texture_interpolate(n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    let threads = 256u32;
    let blocks = (n as u32).div_ceil(threads);
    ((blocks, 1, 1), (threads, 1, 1))
}

/// Launch config for batched texture lookups on GPU.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_texture_batch_lookup(total: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    let threads = 256u32;
    let blocks = (total as u32).div_ceil(threads);
    ((blocks, 1, 1), (threads, 1, 1))
}

// ───────────────────────────────────────────────────────────────────
// Address-mode helpers (used by CPU fallback)
// ───────────────────────────────────────────────────────────────────

/// Apply the chosen address mode to a 1-D integer index.
///
/// Returns `None` when the `Border` mode is active and the index is
/// out-of-bounds (caller should return 0.0).
fn apply_address_mode(idx: i64, size: usize, mode: AddressMode) -> Option<usize> {
    let n = size as i64;
    if n == 0 {
        return None;
    }
    match mode {
        AddressMode::Clamp => Some(idx.clamp(0, n - 1) as usize),
        AddressMode::Wrap => {
            let wrapped = ((idx % n) + n) % n;
            Some(wrapped as usize)
        }
        AddressMode::Mirror => {
            let period = if n <= 1 { 1 } else { 2 * (n - 1) };
            let mut t = ((idx % period) + period) % period;
            if t >= n {
                t = period - t;
            }
            Some(t.clamp(0, n - 1) as usize)
        }
        AddressMode::Border => {
            if idx < 0 || idx >= n {
                None
            } else {
                Some(idx as usize)
            }
        }
    }
}

/// Linear interpolation factor: split a float coordinate into a base index
/// and a fractional part `t ∈ [0, 1)`.
fn lerp_index(coord: f32, _size: usize) -> (i64, f32) {
    // Shift by -0.5 so that integer coords map to texel centres.
    let c = coord - 0.5;
    let base = c.floor() as i64;
    let frac = c - (base as f32);
    (base, frac)
}

// ───────────────────────────────────────────────────────────────────
// Texture creation / destruction (CPU path)
// ───────────────────────────────────────────────────────────────────

/// Monotonically increasing ID counter for texture objects.
static NEXT_TEXTURE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Create a 1-D texture object from a flat `f32` slice.
///
/// On the CPU path this copies `data` into the returned [`TextureObject`].
/// On GPU the data would be bound to a `cudaTextureObject_t`.
///
/// # Errors
///
/// Returns [`KernelError`] if `data.len()` does not match
/// `config.total_elements()`.
pub fn create_texture_1d(data: &[f32], config: TextureConfig) -> Result<TextureObject> {
    if !config.is_1d() {
        return Err(KernelError::InvalidArguments {
            reason: "create_texture_1d requires height == 1".into(),
        }
        .into());
    }
    if data.len() != config.total_elements() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "data length {} does not match texture total elements {}",
                data.len(),
                config.total_elements(),
            ),
        }
        .into());
    }
    let id = NEXT_TEXTURE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(TextureObject { id, config, data: data.to_vec() })
}

/// Create a 2-D texture object from a flat `f32` slice (row-major).
///
/// # Errors
///
/// Returns [`KernelError`] if the data length does not match
/// `width × height × channels`.
pub fn create_texture_2d(data: &[f32], config: TextureConfig) -> Result<TextureObject> {
    if config.is_1d() {
        return Err(KernelError::InvalidArguments {
            reason: "create_texture_2d requires height > 1".into(),
        }
        .into());
    }
    if data.len() != config.total_elements() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "data length {} does not match texture total elements {}",
                data.len(),
                config.total_elements(),
            ),
        }
        .into());
    }
    let id = NEXT_TEXTURE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(TextureObject { id, config, data: data.to_vec() })
}

/// Destroy a texture object (no-op on the CPU path).
///
/// On the GPU path this would call `cudaDestroyTextureObject`.
pub fn destroy_texture(_tex: TextureObject) {
    // CPU path: the `Vec<f32>` is dropped when the `TextureObject` goes
    // out of scope.  Nothing else to clean up.
}

// ───────────────────────────────────────────────────────────────────
// CPU fallback implementations
// ───────────────────────────────────────────────────────────────────

/// 1-D texture fetch (CPU fallback).
///
/// Fetches `indices.len()` values from the 1-D texture using the configured
/// address mode.  When `FilterMode::Linear` is active, indices that fall
/// between texels are linearly interpolated.
///
/// # Errors
///
/// Returns [`KernelError`] if the texture is not 1-D.
pub fn texture_fetch_1d(tex: &TextureObject, indices: &[i32]) -> Result<Vec<f32>> {
    if !tex.config.is_1d() {
        return Err(KernelError::InvalidArguments {
            reason: "texture_fetch_1d requires a 1-D texture".into(),
        }
        .into());
    }
    let w = tex.config.width;
    let mode = tex.config.address_mode;
    let filter = tex.config.filter_mode;
    let data = &tex.data;

    let out: Vec<f32> = indices
        .iter()
        .map(|&idx| match filter {
            FilterMode::Point => {
                apply_address_mode(idx as i64, w, mode).map(|i| data[i]).unwrap_or(0.0)
            }
            FilterMode::Linear => {
                // Integer index i maps to CUDA texel-centre coordinate i + 0.5.
                let coord = idx as f32 + 0.5;
                let (base, t) = lerp_index(coord, w);
                let v0 = apply_address_mode(base, w, mode).map(|i| data[i]).unwrap_or(0.0);
                let v1 = apply_address_mode(base + 1, w, mode).map(|i| data[i]).unwrap_or(0.0);
                v0 * (1.0 - t) + v1 * t
            }
        })
        .collect();

    Ok(out)
}

/// 2-D texture fetch (CPU fallback).
///
/// Reads every texel in the `width × height` grid and returns a flat
/// row-major `Vec<f32>`.
///
/// # Errors
///
/// Returns [`KernelError`] if the texture height is 1 (use
/// [`texture_fetch_1d`] instead).
pub fn texture_fetch_2d(tex: &TextureObject) -> Result<Vec<f32>> {
    if tex.config.is_1d() {
        return Err(KernelError::InvalidArguments {
            reason: "texture_fetch_2d requires a 2-D texture".into(),
        }
        .into());
    }
    let w = tex.config.width;
    let h = tex.config.height;
    let mode = tex.config.address_mode;
    let data = &tex.data;

    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let xi = apply_address_mode(x as i64, w, mode);
            let yi = apply_address_mode(y as i64, h, mode);
            let val = match (xi, yi) {
                (Some(ix), Some(iy)) => data[iy * w + ix],
                _ => 0.0,
            };
            out[y * w + x] = val;
        }
    }
    Ok(out)
}

/// 2-D texture gather at arbitrary `(x, y)` coordinates (CPU fallback).
///
/// Each coordinate pair `(coords_x[i], coords_y[i])` is resolved using
/// the texture's address mode.  Point filtering uses nearest-neighbour;
/// linear filtering interpolates the four surrounding texels.
///
/// # Errors
///
/// Returns [`KernelError`] if coordinate arrays differ in length.
pub fn texture_gather(tex: &TextureObject, coords_x: &[f32], coords_y: &[f32]) -> Result<Vec<f32>> {
    if coords_x.len() != coords_y.len() {
        return Err(KernelError::InvalidArguments {
            reason: "coords_x and coords_y must have the same length".into(),
        }
        .into());
    }
    let w = tex.config.width;
    let h = tex.config.height;
    let mode = tex.config.address_mode;
    let filter = tex.config.filter_mode;
    let data = &tex.data;

    let fetch = |ix: i64, iy: i64| -> f32 {
        let xi = apply_address_mode(ix, w, mode);
        let yi = apply_address_mode(iy, h, mode);
        match (xi, yi) {
            (Some(x), Some(y)) => data[y * w + x],
            _ => 0.0,
        }
    };

    let out: Vec<f32> = coords_x
        .iter()
        .zip(coords_y.iter())
        .map(|(&cx, &cy)| match filter {
            FilterMode::Point => {
                let ix = cx.round() as i64;
                let iy = cy.round() as i64;
                fetch(ix, iy)
            }
            FilterMode::Linear => {
                let (bx, tx) = lerp_index(cx, w);
                let (by, ty) = lerp_index(cy, h);
                let v00 = fetch(bx, by);
                let v10 = fetch(bx + 1, by);
                let v01 = fetch(bx, by + 1);
                let v11 = fetch(bx + 1, by + 1);
                let top = v00 * (1.0 - tx) + v10 * tx;
                let bot = v01 * (1.0 - tx) + v11 * tx;
                top * (1.0 - ty) + bot * ty
            }
        })
        .collect();

    Ok(out)
}

/// 2-D convolution using texture-fetch semantics (CPU fallback).
///
/// Applies a `kernel_h × kernel_w` convolution filter to the texture.
/// Border handling is determined by the texture's address mode — out-of-
/// bounds reads are clamped/wrapped/zeroed automatically.
///
/// # Errors
///
/// Returns [`KernelError`] if the kernel dimensions are zero or the
/// kernel slice length does not match `kernel_w × kernel_h`.
pub fn texture_conv2d(
    tex: &TextureObject,
    kernel: &[f32],
    kernel_w: usize,
    kernel_h: usize,
) -> Result<Vec<f32>> {
    if kernel_w == 0 || kernel_h == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "convolution kernel dimensions must be > 0".into(),
        }
        .into());
    }
    if kernel.len() != kernel_w * kernel_h {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "kernel length {} does not match {}x{}",
                kernel.len(),
                kernel_w,
                kernel_h,
            ),
        }
        .into());
    }

    let w = tex.config.width;
    let h = tex.config.height;
    let mode = tex.config.address_mode;
    let data = &tex.data;
    let half_kw = (kernel_w / 2) as i64;
    let half_kh = (kernel_h / 2) as i64;

    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let sx = (x as i64) - half_kw + (kx as i64);
                    let sy = (y as i64) - half_kh + (ky as i64);
                    let xi = apply_address_mode(sx, w, mode);
                    let yi = apply_address_mode(sy, h, mode);
                    let val = match (xi, yi) {
                        (Some(ix), Some(iy)) => data[iy * w + ix],
                        _ => 0.0,
                    };
                    sum += val * kernel[ky * kernel_w + kx];
                }
            }
            out[y * w + x] = sum;
        }
    }
    Ok(out)
}

/// Bilinear texture interpolation at fractional coordinates (CPU fallback).
///
/// Each coordinate pair `(coords_x[i], coords_y[i])` is interpolated using
/// the four surrounding texels with bilinear weights, regardless of the
/// texture's [`FilterMode`] setting.
///
/// # Errors
///
/// Returns [`KernelError`] if coordinate arrays differ in length.
pub fn texture_interpolate(
    tex: &TextureObject,
    coords_x: &[f32],
    coords_y: &[f32],
) -> Result<Vec<f32>> {
    if coords_x.len() != coords_y.len() {
        return Err(KernelError::InvalidArguments {
            reason: "coords_x and coords_y must have the same length".into(),
        }
        .into());
    }
    let w = tex.config.width;
    let h = tex.config.height;
    let mode = tex.config.address_mode;
    let data = &tex.data;

    let fetch = |ix: i64, iy: i64| -> f32 {
        let xi = apply_address_mode(ix, w, mode);
        let yi = apply_address_mode(iy, h, mode);
        match (xi, yi) {
            (Some(x), Some(y)) => data[y * w + x],
            _ => 0.0,
        }
    };

    let out: Vec<f32> = coords_x
        .iter()
        .zip(coords_y.iter())
        .map(|(&cx, &cy)| {
            let (bx, tx) = lerp_index(cx, w);
            let (by, ty) = lerp_index(cy, h);
            let v00 = fetch(bx, by);
            let v10 = fetch(bx + 1, by);
            let v01 = fetch(bx, by + 1);
            let v11 = fetch(bx + 1, by + 1);
            let top = v00 * (1.0 - tx) + v10 * tx;
            let bot = v01 * (1.0 - tx) + v11 * tx;
            top * (1.0 - ty) + bot * ty
        })
        .collect();

    Ok(out)
}

/// Batched 1-D texture lookup (CPU fallback).
///
/// Equivalent to `texture_fetch_1d` but accepts a flat array of indices
/// spanning `batch_size` independent lookups of `lookup_len` each.
///
/// # Errors
///
/// Returns [`KernelError`] if the texture is not 1-D or the indices
/// length is not `batch_size × lookup_len`.
pub fn texture_batch_lookup(
    tex: &TextureObject,
    indices: &[i32],
    batch_size: usize,
    lookup_len: usize,
) -> Result<Vec<f32>> {
    if !tex.config.is_1d() {
        return Err(KernelError::InvalidArguments {
            reason: "texture_batch_lookup requires a 1-D texture".into(),
        }
        .into());
    }
    if indices.len() != batch_size * lookup_len {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "indices length {} does not match batch_size({}) x lookup_len({})",
                indices.len(),
                batch_size,
                lookup_len,
            ),
        }
        .into());
    }
    // Re-use the single-batch fetch implementation.
    texture_fetch_1d(tex, indices)
}

// ───────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────

    fn data_1d() -> Vec<f32> {
        vec![10.0, 20.0, 30.0, 40.0, 50.0]
    }

    fn data_2d() -> Vec<f32> {
        #[rustfmt::skip]
        let d = vec![
             1.0,  2.0,  3.0,  4.0,
             5.0,  6.0,  7.0,  8.0,
             9.0, 10.0, 11.0, 12.0,
        ];
        d
    }

    fn make_1d(data: &[f32]) -> TextureObject {
        let cfg = TextureConfig::new_1d(data.len()).unwrap();
        create_texture_1d(data, cfg).unwrap()
    }

    fn make_2d(data: &[f32], w: usize, h: usize) -> TextureObject {
        let cfg = TextureConfig::new_2d(w, h).unwrap();
        create_texture_2d(data, cfg).unwrap()
    }

    fn make_1d_with_mode(data: &[f32], addr: AddressMode) -> TextureObject {
        let cfg = TextureConfig::new_1d(data.len()).unwrap().with_address_mode(addr);
        create_texture_1d(data, cfg).unwrap()
    }

    fn make_1d_linear(data: &[f32]) -> TextureObject {
        let cfg = TextureConfig::new_1d(data.len()).unwrap().with_filter_mode(FilterMode::Linear);
        create_texture_1d(data, cfg).unwrap()
    }

    fn make_2d_with_mode(data: &[f32], w: usize, h: usize, addr: AddressMode) -> TextureObject {
        let cfg = TextureConfig::new_2d(w, h).unwrap().with_address_mode(addr);
        create_texture_2d(data, cfg).unwrap()
    }

    fn make_2d_linear(data: &[f32], w: usize, h: usize) -> TextureObject {
        let cfg = TextureConfig::new_2d(w, h).unwrap().with_filter_mode(FilterMode::Linear);
        create_texture_2d(data, cfg).unwrap()
    }

    // ── FilterMode / AddressMode enum tests ─────────────────────

    #[test]
    fn test_filter_mode_point_cuda_int() {
        assert_eq!(FilterMode::Point.as_cuda_int(), 0);
    }

    #[test]
    fn test_filter_mode_linear_cuda_int() {
        assert_eq!(FilterMode::Linear.as_cuda_int(), 1);
    }

    #[test]
    fn test_address_mode_clamp_cuda_int() {
        assert_eq!(AddressMode::Clamp.as_cuda_int(), 0);
    }

    #[test]
    fn test_address_mode_wrap_cuda_int() {
        assert_eq!(AddressMode::Wrap.as_cuda_int(), 1);
    }

    #[test]
    fn test_address_mode_mirror_cuda_int() {
        assert_eq!(AddressMode::Mirror.as_cuda_int(), 2);
    }

    #[test]
    fn test_address_mode_border_cuda_int() {
        assert_eq!(AddressMode::Border.as_cuda_int(), 3);
    }

    // ── TextureConfig tests ─────────────────────────────────────

    #[test]
    fn test_config_1d_defaults() {
        let cfg = TextureConfig::new_1d(128).unwrap();
        assert_eq!(cfg.width, 128);
        assert_eq!(cfg.height, 1);
        assert!(cfg.is_1d());
        assert_eq!(cfg.filter_mode, FilterMode::Point);
        assert_eq!(cfg.address_mode, AddressMode::Clamp);
        assert!(!cfg.normalized_coords);
        assert_eq!(cfg.channels, 1);
        assert!(!cfg.read_as_normalized_float);
    }

    #[test]
    fn test_config_2d_defaults() {
        let cfg = TextureConfig::new_2d(64, 32).unwrap();
        assert_eq!(cfg.width, 64);
        assert_eq!(cfg.height, 32);
        assert!(!cfg.is_1d());
        assert_eq!(cfg.total_elements(), 64 * 32);
    }

    #[test]
    fn test_config_rejects_zero_width() {
        assert!(TextureConfig::new_1d(0).is_err());
    }

    #[test]
    fn test_config_rejects_zero_2d_width() {
        assert!(TextureConfig::new_2d(0, 10).is_err());
    }

    #[test]
    fn test_config_rejects_zero_2d_height() {
        assert!(TextureConfig::new_2d(10, 0).is_err());
    }

    #[test]
    fn test_config_builder_filter_mode() {
        let cfg = TextureConfig::new_1d(16).unwrap().with_filter_mode(FilterMode::Linear);
        assert_eq!(cfg.filter_mode, FilterMode::Linear);
    }

    #[test]
    fn test_config_builder_address_mode() {
        let cfg = TextureConfig::new_1d(16).unwrap().with_address_mode(AddressMode::Wrap);
        assert_eq!(cfg.address_mode, AddressMode::Wrap);
    }

    #[test]
    fn test_config_builder_normalized_coords() {
        let cfg = TextureConfig::new_1d(16).unwrap().with_normalized_coords(true);
        assert!(cfg.normalized_coords);
    }

    #[test]
    fn test_config_builder_channels_1() {
        let cfg = TextureConfig::new_1d(16).unwrap().with_channels(1).unwrap();
        assert_eq!(cfg.channels, 1);
    }

    #[test]
    fn test_config_builder_channels_2() {
        let cfg = TextureConfig::new_1d(16).unwrap().with_channels(2).unwrap();
        assert_eq!(cfg.channels, 2);
    }

    #[test]
    fn test_config_builder_channels_4() {
        let cfg = TextureConfig::new_1d(16).unwrap().with_channels(4).unwrap();
        assert_eq!(cfg.channels, 4);
    }

    #[test]
    fn test_config_builder_channels_rejects_3() {
        assert!(TextureConfig::new_1d(16).unwrap().with_channels(3).is_err());
    }

    #[test]
    fn test_config_builder_channels_rejects_0() {
        assert!(TextureConfig::new_1d(16).unwrap().with_channels(0).is_err());
    }

    #[test]
    fn test_config_builder_normalized_float_read() {
        let cfg = TextureConfig::new_1d(16).unwrap().with_normalized_float_read(true);
        assert!(cfg.read_as_normalized_float);
    }

    #[test]
    fn test_config_total_elements_with_channels() {
        let cfg = TextureConfig::new_2d(8, 4).unwrap().with_channels(4).unwrap();
        assert_eq!(cfg.total_elements(), 8 * 4 * 4);
    }

    #[test]
    fn test_config_grid_dim() {
        let cfg = TextureConfig::new_2d(100, 50).unwrap();
        let (gx, gy, gz) = cfg.grid_dim(32);
        assert_eq!(gx, 4); // ceil(100/32)
        assert_eq!(gy, 50);
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_config_block_dim() {
        let cfg = TextureConfig::new_2d(100, 50).unwrap();
        let (bx, by, bz) = cfg.block_dim(256);
        assert_eq!(bx, 100); // min(256, 100)
        assert_eq!(by, 1);
        assert_eq!(bz, 1);
    }

    #[test]
    fn test_config_block_dim_capped() {
        let cfg = TextureConfig::new_1d(512).unwrap();
        let (bx, _, _) = cfg.block_dim(256);
        assert_eq!(bx, 256);
    }

    // ── TextureObject tests ─────────────────────────────────────

    #[test]
    fn test_texture_object_accessors() {
        let tex = make_2d(&data_2d(), 4, 3);
        assert_eq!(tex.width(), 4);
        assert_eq!(tex.height(), 3);
        assert_eq!(tex.channels(), 1);
    }

    #[test]
    fn test_texture_object_unique_ids() {
        let t1 = make_1d(&data_1d());
        let t2 = make_1d(&data_1d());
        assert_ne!(t1.id, t2.id);
    }

    // ── create_texture_1d tests ─────────────────────────────────

    #[test]
    fn test_create_1d_basic() {
        let d = data_1d();
        let tex = make_1d(&d);
        assert_eq!(tex.data, d);
        assert!(tex.config.is_1d());
    }

    #[test]
    fn test_create_1d_data_length_mismatch() {
        let cfg = TextureConfig::new_1d(3).unwrap();
        assert!(create_texture_1d(&[1.0, 2.0], cfg).is_err());
    }

    #[test]
    fn test_create_1d_rejects_2d_config() {
        let cfg = TextureConfig::new_2d(3, 2).unwrap();
        assert!(create_texture_1d(&[0.0; 6], cfg).is_err());
    }

    // ── create_texture_2d tests ─────────────────────────────────

    #[test]
    fn test_create_2d_basic() {
        let d = data_2d();
        let tex = make_2d(&d, 4, 3);
        assert_eq!(tex.data, d);
        assert!(!tex.config.is_1d());
    }

    #[test]
    fn test_create_2d_data_length_mismatch() {
        let cfg = TextureConfig::new_2d(4, 3).unwrap();
        assert!(create_texture_2d(&[0.0; 10], cfg).is_err());
    }

    #[test]
    fn test_create_2d_rejects_1d_config() {
        let cfg = TextureConfig::new_1d(5).unwrap();
        assert!(create_texture_2d(&[0.0; 5], cfg).is_err());
    }

    // ── destroy_texture tests ───────────────────────────────────

    #[test]
    fn test_destroy_texture_does_not_panic() {
        let tex = make_1d(&data_1d());
        destroy_texture(tex);
    }

    // ── texture_fetch_1d tests ──────────────────────────────────

    #[test]
    fn test_fetch_1d_in_bounds() {
        let tex = make_1d(&data_1d());
        let out = texture_fetch_1d(&tex, &[0, 2, 4]).unwrap();
        assert_eq!(out, vec![10.0, 30.0, 50.0]);
    }

    #[test]
    fn test_fetch_1d_clamp_negative() {
        let tex = make_1d(&data_1d());
        let out = texture_fetch_1d(&tex, &[-1]).unwrap();
        assert_eq!(out, vec![10.0]);
    }

    #[test]
    fn test_fetch_1d_clamp_above() {
        let tex = make_1d(&data_1d());
        let out = texture_fetch_1d(&tex, &[100]).unwrap();
        assert_eq!(out, vec![50.0]);
    }

    #[test]
    fn test_fetch_1d_wrap_negative() {
        let tex = make_1d_with_mode(&data_1d(), AddressMode::Wrap);
        let out = texture_fetch_1d(&tex, &[-1]).unwrap();
        assert_eq!(out, vec![50.0]);
    }

    #[test]
    fn test_fetch_1d_wrap_above() {
        let tex = make_1d_with_mode(&data_1d(), AddressMode::Wrap);
        let out = texture_fetch_1d(&tex, &[7]).unwrap();
        assert_eq!(out, vec![30.0]);
    }

    #[test]
    fn test_fetch_1d_border_out_of_bounds() {
        let tex = make_1d_with_mode(&data_1d(), AddressMode::Border);
        let out = texture_fetch_1d(&tex, &[-1, 5]).unwrap();
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn test_fetch_1d_border_in_bounds() {
        let tex = make_1d_with_mode(&data_1d(), AddressMode::Border);
        let out = texture_fetch_1d(&tex, &[0, 4]).unwrap();
        assert_eq!(out, vec![10.0, 50.0]);
    }

    #[test]
    fn test_fetch_1d_mirror() {
        let d = vec![10.0, 20.0, 30.0];
        let tex = make_1d_with_mode(&d, AddressMode::Mirror);
        let out = texture_fetch_1d(&tex, &[3, 4]).unwrap();
        assert_eq!(out, vec![20.0, 10.0]);
    }

    #[test]
    fn test_fetch_1d_empty_indices() {
        let tex = make_1d(&data_1d());
        let out = texture_fetch_1d(&tex, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_fetch_1d_rejects_2d_texture() {
        let tex = make_2d(&data_2d(), 4, 3);
        assert!(texture_fetch_1d(&tex, &[0]).is_err());
    }

    #[test]
    fn test_fetch_1d_linear_at_texel_centre() {
        let d = vec![0.0, 10.0, 20.0, 30.0];
        let tex = make_1d_linear(&d);
        let out = texture_fetch_1d(&tex, &[0, 1, 2, 3]).unwrap();
        for (i, v) in out.iter().enumerate() {
            assert!((v - d[i]).abs() < 1e-5, "idx {i}: got {v}, want {}", d[i]);
        }
    }

    // ── texture_fetch_2d tests ──────────────────────────────────

    #[test]
    fn test_fetch_2d_identity() {
        let d = data_2d();
        let tex = make_2d(&d, 4, 3);
        let out = texture_fetch_2d(&tex).unwrap();
        assert_eq!(out, d);
    }

    #[test]
    fn test_fetch_2d_rejects_1d_texture() {
        let tex = make_1d(&data_1d());
        assert!(texture_fetch_2d(&tex).is_err());
    }

    #[test]
    fn test_fetch_2d_small_grid() {
        let d = vec![42.0, 99.0];
        let tex = make_2d(&d, 1, 2);
        let out = texture_fetch_2d(&tex).unwrap();
        assert_eq!(out, d);
    }

    // ── texture_gather tests ────────────────────────────────────

    #[test]
    fn test_gather_at_integer_coords() {
        let tex = make_2d(&data_2d(), 4, 3);
        let cx = vec![0.0, 3.0, 1.0];
        let cy = vec![0.0, 2.0, 1.0];
        let out = texture_gather(&tex, &cx, &cy).unwrap();
        assert_eq!(out, vec![1.0, 12.0, 6.0]);
    }

    #[test]
    fn test_gather_mismatched_coords() {
        let tex = make_2d(&data_2d(), 4, 3);
        assert!(texture_gather(&tex, &[0.0, 1.0], &[0.0]).is_err());
    }

    #[test]
    fn test_gather_empty_coords() {
        let tex = make_2d(&data_2d(), 4, 3);
        let out = texture_gather(&tex, &[], &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_gather_clamp_oob() {
        let tex = make_2d(&data_2d(), 4, 3);
        let cx = vec![-10.0, 100.0];
        let cy = vec![-10.0, 100.0];
        let out = texture_gather(&tex, &cx, &cy).unwrap();
        assert_eq!(out, vec![1.0, 12.0]);
    }

    #[test]
    fn test_gather_border_oob_returns_zero() {
        let tex = make_2d_with_mode(&data_2d(), 4, 3, AddressMode::Border);
        let cx = vec![-1.0, 10.0];
        let cy = vec![0.0, 0.0];
        let out = texture_gather(&tex, &cx, &cy).unwrap();
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn test_gather_linear_centre() {
        let tex = make_2d_linear(&data_2d(), 4, 3);
        let cx = vec![0.5, 1.5];
        let cy = vec![0.5, 0.5];
        let out = texture_gather(&tex, &cx, &cy).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_gather_linear_midpoint() {
        let d = vec![10.0, 20.0, 30.0, 40.0];
        let tex = make_2d_linear(&d, 2, 2);
        let cx = [1.0];
        let cy = [0.5];
        let out = texture_gather(&tex, &cx, &cy).unwrap();
        assert!((out[0] - 15.0).abs() < 1e-4, "got {}", out[0]);
    }

    // ── texture_conv2d tests ────────────────────────────────────

    #[test]
    fn test_conv2d_identity_kernel() {
        let d = data_2d();
        let tex = make_2d(&d, 4, 3);
        let out = texture_conv2d(&tex, &[1.0], 1, 1).unwrap();
        assert_eq!(out, d);
    }

    #[test]
    fn test_conv2d_zero_kernel() {
        let d = data_2d();
        let tex = make_2d(&d, 4, 3);
        let out = texture_conv2d(&tex, &[0.0; 9], 3, 3).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_conv2d_constant_kernel() {
        let d = [1.0; 9];
        let tex = make_2d(&d, 3, 3);
        let k = [1.0; 9];
        let out = texture_conv2d(&tex, &k, 3, 3).unwrap();
        assert_eq!(out[4], 9.0);
    }

    #[test]
    fn test_conv2d_kernel_dim_mismatch() {
        let tex = make_2d(&data_2d(), 4, 3);
        assert!(texture_conv2d(&tex, &[1.0; 8], 3, 3).is_err());
    }

    #[test]
    fn test_conv2d_zero_kernel_dim() {
        let tex = make_2d(&data_2d(), 4, 3);
        assert!(texture_conv2d(&tex, &[], 0, 0).is_err());
    }

    #[test]
    fn test_conv2d_border_mode_zeros_oob() {
        let d = [1.0; 4];
        let tex = make_2d_with_mode(&d, 2, 2, AddressMode::Border);
        let k = [1.0; 9];
        let out = texture_conv2d(&tex, &k, 3, 3).unwrap();
        assert_eq!(out[0], 4.0);
    }

    #[test]
    fn test_conv2d_clamp_edge_replication() {
        let d = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let tex = make_2d(&d, 3, 2);
        let k = vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let out = texture_conv2d(&tex, &k, 3, 3).unwrap();
        assert!((out[1] - 90.0).abs() < 1e-5, "got {}", out[1]);
    }

    // ── texture_interpolate tests ───────────────────────────────

    #[test]
    fn test_interpolate_at_centres() {
        let d = data_2d();
        let tex = make_2d(&d, 4, 3);
        let cx = vec![0.5, 1.5, 3.5];
        let cy = vec![0.5, 0.5, 2.5];
        let out = texture_interpolate(&tex, &cx, &cy).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 2.0).abs() < 1e-5);
        assert!((out[2] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_interpolate_midpoint() {
        let d = vec![0.0, 10.0, 0.0, 10.0];
        let tex = make_2d(&d, 2, 2);
        let cx = [1.0];
        let cy = [1.0];
        let out = texture_interpolate(&tex, &cx, &cy).unwrap();
        assert!((out[0] - 5.0).abs() < 1e-4, "got {}", out[0]);
    }

    #[test]
    fn test_interpolate_mismatched_coords() {
        let tex = make_2d(&data_2d(), 4, 3);
        assert!(texture_interpolate(&tex, &[0.0], &[]).is_err());
    }

    #[test]
    fn test_interpolate_empty() {
        let tex = make_2d(&data_2d(), 4, 3);
        let out = texture_interpolate(&tex, &[], &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_interpolate_clamp_oob() {
        let d = vec![5.0, 15.0, 25.0, 35.0];
        let tex = make_2d(&d, 2, 2);
        let cx = vec![-100.0];
        let cy = vec![-100.0];
        let out = texture_interpolate(&tex, &cx, &cy).unwrap();
        assert!((out[0] - 5.0).abs() < 1e-4, "got {}", out[0]);
    }

    // ── texture_batch_lookup tests ──────────────────────────────

    #[test]
    fn test_batch_lookup_basic() {
        let tex = make_1d(&data_1d());
        let indices = vec![4, 0, 2, 1, 3, 0];
        let out = texture_batch_lookup(&tex, &indices, 2, 3).unwrap();
        assert_eq!(out, vec![50.0, 10.0, 30.0, 20.0, 40.0, 10.0]);
    }

    #[test]
    fn test_batch_lookup_length_mismatch() {
        let tex = make_1d(&data_1d());
        assert!(texture_batch_lookup(&tex, &[0, 1], 2, 2).is_err());
    }

    #[test]
    fn test_batch_lookup_rejects_2d_texture() {
        let tex = make_2d(&data_2d(), 4, 3);
        assert!(texture_batch_lookup(&tex, &[0, 1, 2, 3], 2, 2).is_err());
    }

    #[test]
    fn test_batch_lookup_single_batch() {
        let tex = make_1d(&data_1d());
        let out = texture_batch_lookup(&tex, &[3, 1], 1, 2).unwrap();
        assert_eq!(out, vec![40.0, 20.0]);
    }

    #[test]
    fn test_batch_lookup_clamp() {
        let tex = make_1d(&data_1d());
        let out = texture_batch_lookup(&tex, &[-5, 100], 1, 2).unwrap();
        assert_eq!(out, vec![10.0, 50.0]);
    }

    // ── address_mode helper tests ───────────────────────────────

    #[test]
    fn test_apply_clamp_in_bounds() {
        assert_eq!(apply_address_mode(2, 5, AddressMode::Clamp), Some(2));
    }

    #[test]
    fn test_apply_clamp_below() {
        assert_eq!(apply_address_mode(-3, 5, AddressMode::Clamp), Some(0));
    }

    #[test]
    fn test_apply_clamp_above() {
        assert_eq!(apply_address_mode(10, 5, AddressMode::Clamp), Some(4));
    }

    #[test]
    fn test_apply_wrap_positive() {
        assert_eq!(apply_address_mode(7, 5, AddressMode::Wrap), Some(2));
    }

    #[test]
    fn test_apply_wrap_negative() {
        assert_eq!(apply_address_mode(-1, 5, AddressMode::Wrap), Some(4));
    }

    #[test]
    fn test_apply_wrap_exact_multiple() {
        assert_eq!(apply_address_mode(10, 5, AddressMode::Wrap), Some(0));
    }

    #[test]
    fn test_apply_mirror_in_bounds() {
        assert_eq!(apply_address_mode(1, 5, AddressMode::Mirror), Some(1));
    }

    #[test]
    fn test_apply_mirror_at_edge() {
        assert_eq!(apply_address_mode(4, 5, AddressMode::Mirror), Some(4));
    }

    #[test]
    fn test_apply_mirror_one_past() {
        assert_eq!(apply_address_mode(5, 5, AddressMode::Mirror), Some(3));
    }

    #[test]
    fn test_apply_mirror_negative() {
        assert_eq!(apply_address_mode(-1, 5, AddressMode::Mirror), Some(1));
    }

    #[test]
    fn test_apply_border_in_bounds() {
        assert_eq!(apply_address_mode(3, 5, AddressMode::Border), Some(3));
    }

    #[test]
    fn test_apply_border_below() {
        assert_eq!(apply_address_mode(-1, 5, AddressMode::Border), None);
    }

    #[test]
    fn test_apply_border_above() {
        assert_eq!(apply_address_mode(5, 5, AddressMode::Border), None);
    }

    #[test]
    fn test_apply_address_mode_zero_size() {
        assert_eq!(apply_address_mode(0, 0, AddressMode::Clamp), None);
    }

    // ── lerp_index tests ────────────────────────────────────────

    #[test]
    fn test_lerp_index_integer() {
        let (base, frac) = lerp_index(0.5, 10);
        assert_eq!(base, 0);
        assert!(frac.abs() < 1e-6);
    }

    #[test]
    fn test_lerp_index_midpoint() {
        let (base, frac) = lerp_index(1.0, 10);
        assert_eq!(base, 0);
        assert!((frac - 0.5).abs() < 1e-6);
    }

    // ── GPU launch config tests ─────────────────────────────────

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    mod gpu_launch_tests {
        use super::super::*;

        #[test]
        fn test_launch_1d_single_block() {
            let ((bx, by, bz), (tx, ty, tz)) = launch_texture_fetch_1d(100);
            assert_eq!(bx, 1);
            assert_eq!(by, 1);
            assert_eq!(bz, 1);
            assert_eq!(tx, 256);
            assert_eq!(ty, 1);
            assert_eq!(tz, 1);
        }

        #[test]
        fn test_launch_1d_multi_block() {
            let ((bx, _, _), _) = launch_texture_fetch_1d(1000);
            assert_eq!(bx, 4);
        }

        #[test]
        fn test_launch_2d_small() {
            let ((bx, by, _), (tx, ty, _)) = launch_texture_fetch_2d(8, 8);
            assert_eq!(bx, 1);
            assert_eq!(by, 1);
            assert_eq!(tx, 16);
            assert_eq!(ty, 16);
        }

        #[test]
        fn test_launch_2d_large() {
            let ((bx, by, _), _) = launch_texture_fetch_2d(100, 100);
            assert_eq!(bx, 7);
            assert_eq!(by, 7);
        }

        #[test]
        fn test_launch_gather() {
            let ((bx, _, _), (tx, _, _)) = launch_texture_gather(512);
            assert_eq!(bx, 2);
            assert_eq!(tx, 256);
        }

        #[test]
        fn test_launch_conv2d() {
            let ((bx, by, _), (tx, ty, _)) = launch_texture_conv2d(32, 32);
            assert_eq!(bx, 2);
            assert_eq!(by, 2);
            assert_eq!(tx, 16);
            assert_eq!(ty, 16);
        }

        #[test]
        fn test_launch_interpolate() {
            let ((bx, _, _), _) = launch_texture_interpolate(256);
            assert_eq!(bx, 1);
        }

        #[test]
        fn test_launch_batch_lookup() {
            let ((bx, _, _), _) = launch_texture_batch_lookup(600);
            assert_eq!(bx, 3);
        }
    }

    // ── CUDA kernel source existence tests ──────────────────────

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    mod kernel_source_tests {
        use super::super::*;

        #[test]
        fn test_fetch_1d_kernel_contains_function() {
            assert!(TEXTURE_FETCH_1D_KERNEL_SRC.contains("texture_fetch_1d"));
        }

        #[test]
        fn test_fetch_2d_kernel_contains_function() {
            assert!(TEXTURE_FETCH_2D_KERNEL_SRC.contains("texture_fetch_2d"));
        }

        #[test]
        fn test_gather_kernel_contains_function() {
            assert!(TEXTURE_GATHER_KERNEL_SRC.contains("texture_gather"));
        }

        #[test]
        fn test_conv2d_kernel_contains_function() {
            assert!(TEXTURE_CONV2D_KERNEL_SRC.contains("texture_conv2d"));
        }

        #[test]
        fn test_interpolate_kernel_contains_function() {
            assert!(TEXTURE_INTERPOLATE_KERNEL_SRC.contains("texture_interpolate"));
        }

        #[test]
        fn test_batch_lookup_kernel_contains_function() {
            assert!(TEXTURE_BATCH_LOOKUP_KERNEL_SRC.contains("texture_batch_lookup"));
        }

        #[test]
        fn test_fetch_1d_kernel_has_extern_c() {
            assert!(TEXTURE_FETCH_1D_KERNEL_SRC.contains("extern \"C\""));
        }

        #[test]
        fn test_fetch_2d_kernel_has_tex2d() {
            assert!(TEXTURE_FETCH_2D_KERNEL_SRC.contains("tex2D"));
        }

        #[test]
        fn test_conv2d_kernel_has_loop() {
            assert!(TEXTURE_CONV2D_KERNEL_SRC.contains("for"));
        }
    }
}
