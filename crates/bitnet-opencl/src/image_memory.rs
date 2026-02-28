//! OpenCL image/texture memory for weight matrix storage.
//!
//! Intel GPUs have hardware texture units that can accelerate read-only
//! data access through the L2 cache. This module provides an abstraction
//! for storing weight matrices as `cl_image2d` objects, with automatic
//! fallback to standard buffer memory when image support is unavailable.
//!
//! # Image format selection
//!
//! The module probes device capabilities and selects the best image
//! format from a preference list:
//!
//! | Format           | Channels | Type   | Bytes/pixel |
//! |------------------|----------|--------|-------------|
//! | `CL_R + FLOAT`   | 1        | f32    | 4           |
//! | `CL_R + HALF`    | 1        | f16    | 2           |
//! | `CL_RGBA + FLOAT`| 4        | f32    | 16          |
//! | `CL_RGBA + HALF` | 4        | f16    | 8           |

use std::fmt;

// ---------------------------------------------------------------------------
// Image channel / type enums (pure-Rust, no OpenCL runtime dependency)
// ---------------------------------------------------------------------------

/// OpenCL image channel order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageChannelOrder {
    /// Single channel (`CL_R`).
    R,
    /// Four channels (`CL_RGBA`).
    Rgba,
}

impl fmt::Display for ImageChannelOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::R => write!(f, "CL_R"),
            Self::Rgba => write!(f, "CL_RGBA"),
        }
    }
}

/// OpenCL image channel data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageChannelType {
    /// 32-bit IEEE 754 float (`CL_FLOAT`).
    Float,
    /// 16-bit IEEE 754 half-precision float (`CL_HALF_FLOAT`).
    Half,
}

impl fmt::Display for ImageChannelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float => write!(f, "CL_FLOAT"),
            Self::Half => write!(f, "CL_HALF_FLOAT"),
        }
    }
}

/// A concrete image format combining channel order and data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageFormat {
    pub channel_order: ImageChannelOrder,
    pub channel_type: ImageChannelType,
}

impl ImageFormat {
    /// Create a new image format.
    pub const fn new(channel_order: ImageChannelOrder, channel_type: ImageChannelType) -> Self {
        Self {
            channel_order,
            channel_type,
        }
    }

    /// Bytes per pixel for this format.
    pub const fn bytes_per_pixel(&self) -> usize {
        let channels = match self.channel_order {
            ImageChannelOrder::R => 1,
            ImageChannelOrder::Rgba => 4,
        };
        let type_size = match self.channel_type {
            ImageChannelType::Float => 4,
            ImageChannelType::Half => 2,
        };
        channels * type_size
    }

    /// Number of channels.
    pub const fn num_channels(&self) -> usize {
        match self.channel_order {
            ImageChannelOrder::R => 1,
            ImageChannelOrder::Rgba => 4,
        }
    }
}

impl fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} × {}", self.channel_order, self.channel_type)
    }
}

// ---------------------------------------------------------------------------
// Preferred format list (highest preference first)
// ---------------------------------------------------------------------------

/// Default preference order for image formats on Intel GPUs.
///
/// `CL_RGBA × HALF` is preferred because it packs 4 half-precision values
/// per texel, maximising L2 cache line utilisation on Intel Arc.
pub const PREFERRED_FORMATS: &[ImageFormat] = &[
    ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelType::Half),
    ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelType::Float),
    ImageFormat::new(ImageChannelOrder::R, ImageChannelType::Half),
    ImageFormat::new(ImageChannelOrder::R, ImageChannelType::Float),
];

// ---------------------------------------------------------------------------
// Device image capabilities
// ---------------------------------------------------------------------------

/// Capabilities of a device regarding 2-D image objects.
#[derive(Debug, Clone)]
pub struct ImageCapabilities {
    /// Whether the device supports image objects at all.
    pub images_supported: bool,
    /// Maximum width of a 2-D image (pixels), 0 if unsupported.
    pub max_width: usize,
    /// Maximum height of a 2-D image (pixels), 0 if unsupported.
    pub max_height: usize,
    /// Set of formats the device can read from.
    pub supported_formats: Vec<ImageFormat>,
}

impl ImageCapabilities {
    /// A capabilities struct indicating no image support.
    pub fn unsupported() -> Self {
        Self {
            images_supported: false,
            max_width: 0,
            max_height: 0,
            supported_formats: Vec::new(),
        }
    }

    /// Pick the best format from `PREFERRED_FORMATS` that the device supports.
    pub fn best_format(&self) -> Option<ImageFormat> {
        if !self.images_supported {
            return None;
        }
        PREFERRED_FORMATS
            .iter()
            .find(|pf| self.supported_formats.contains(pf))
            .copied()
    }

    /// Check whether a given matrix (rows × cols) fits within image limits
    /// for the given format.
    ///
    /// The matrix is mapped to a 2-D image as:
    ///   width  = cols / format.num_channels()
    ///   height = rows
    pub fn matrix_fits(&self, rows: usize, cols: usize, format: &ImageFormat) -> bool {
        if !self.images_supported {
            return false;
        }
        let channels = format.num_channels();
        if cols % channels != 0 {
            return false;
        }
        let width = cols / channels;
        width <= self.max_width && rows <= self.max_height
    }
}

// ---------------------------------------------------------------------------
// Backing store
// ---------------------------------------------------------------------------

/// How a weight matrix is actually stored on the device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryBacking {
    /// Stored as a `cl_image2d` object with the given format.
    Image(ImageFormat),
    /// Stored as a plain `cl_mem` buffer (fallback).
    Buffer,
}

impl fmt::Display for MemoryBacking {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image(fmt_) => write!(f, "Image({})", fmt_),
            Self::Buffer => write!(f, "Buffer"),
        }
    }
}

// ---------------------------------------------------------------------------
// Weight image descriptor
// ---------------------------------------------------------------------------

/// Descriptor for a weight matrix stored in OpenCL image or buffer memory.
///
/// This is a *logical* descriptor — it does not hold an actual OpenCL handle
/// (which would require `opencl3` at runtime). Kernel launch code uses this
/// to decide how to bind the matrix.
#[derive(Debug, Clone)]
pub struct WeightImageDescriptor {
    /// Human-readable name (e.g. `"layers.0.attn.q_proj"`).
    pub name: String,
    /// Number of rows in the weight matrix.
    pub rows: usize,
    /// Number of columns in the weight matrix.
    pub cols: usize,
    /// How the data is actually stored on device.
    pub backing: MemoryBacking,
    /// Image width in pixels (0 when backing == Buffer).
    pub image_width: usize,
    /// Image height in pixels (0 when backing == Buffer).
    pub image_height: usize,
    /// Total bytes allocated on device.
    pub size_bytes: usize,
}

impl WeightImageDescriptor {
    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.rows * self.cols
    }

    /// Whether this weight is stored as an image.
    pub fn is_image(&self) -> bool {
        matches!(self.backing, MemoryBacking::Image(_))
    }
}

impl fmt::Display for WeightImageDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}×{} ({}, {:.1} KB)",
            self.name,
            self.rows,
            self.cols,
            self.backing,
            self.size_bytes as f64 / 1024.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Image memory manager
// ---------------------------------------------------------------------------

/// Statistics about image memory allocations.
#[derive(Debug, Clone, Default)]
pub struct ImageMemoryStats {
    /// Total weight matrices managed.
    pub total_weights: usize,
    /// How many are stored as images.
    pub image_backed: usize,
    /// How many fell back to plain buffers.
    pub buffer_backed: usize,
    /// Total bytes used by image-backed weights.
    pub image_bytes: u64,
    /// Total bytes used by buffer-backed weights.
    pub buffer_bytes: u64,
}

impl ImageMemoryStats {
    /// Fraction of weights that are image-backed.
    pub fn image_ratio(&self) -> f64 {
        if self.total_weights == 0 {
            return 0.0;
        }
        self.image_backed as f64 / self.total_weights as f64
    }
}

/// Manager for allocating weight matrices into OpenCL image or buffer memory.
///
/// Call [`ImageMemoryManager::new`] with the device's [`ImageCapabilities`],
/// then use [`allocate`](ImageMemoryManager::allocate) for each weight matrix.
/// The manager automatically falls back to buffer memory when image memory
/// is unsupported or the matrix exceeds image dimension limits.
#[derive(Debug)]
pub struct ImageMemoryManager {
    capabilities: ImageCapabilities,
    /// Format selected for this manager (if any).
    selected_format: Option<ImageFormat>,
    /// All descriptors managed.
    descriptors: Vec<WeightImageDescriptor>,
}

impl ImageMemoryManager {
    /// Create a new manager from device capabilities.
    ///
    /// The best available image format is selected automatically.
    pub fn new(capabilities: ImageCapabilities) -> Self {
        let selected_format = capabilities.best_format();
        if let Some(fmt) = &selected_format {
            log::info!("ImageMemoryManager: selected format {fmt}");
        } else {
            log::info!("ImageMemoryManager: no image support, all weights will use buffer memory");
        }
        Self {
            capabilities,
            selected_format,
            descriptors: Vec::new(),
        }
    }

    /// Create a manager that always falls back to buffer memory.
    pub fn buffer_only() -> Self {
        Self {
            capabilities: ImageCapabilities::unsupported(),
            selected_format: None,
            descriptors: Vec::new(),
        }
    }

    /// The format this manager is using, if any.
    pub fn selected_format(&self) -> Option<ImageFormat> {
        self.selected_format
    }

    /// Allocate a weight matrix, choosing image or buffer backing.
    ///
    /// Returns a [`WeightImageDescriptor`] that describes *how* the weight
    /// should be stored. The caller is responsible for the actual OpenCL
    /// allocation using the returned descriptor.
    pub fn allocate(&mut self, name: impl Into<String>, rows: usize, cols: usize) -> WeightImageDescriptor {
        let name = name.into();

        let (backing, image_width, image_height, size_bytes) =
            if let Some(fmt) = self.selected_format {
                if self.capabilities.matrix_fits(rows, cols, &fmt) {
                    let channels = fmt.num_channels();
                    let w = cols / channels;
                    let h = rows;
                    let bytes = w * h * fmt.bytes_per_pixel();
                    (MemoryBacking::Image(fmt), w, h, bytes)
                } else {
                    log::debug!(
                        "Weight '{name}' ({rows}×{cols}) exceeds image limits, using buffer"
                    );
                    let bytes = rows * cols * 4; // assume f32 fallback
                    (MemoryBacking::Buffer, 0, 0, bytes)
                }
            } else {
                let bytes = rows * cols * 4;
                (MemoryBacking::Buffer, 0, 0, bytes)
            };

        let desc = WeightImageDescriptor {
            name,
            rows,
            cols,
            backing,
            image_width,
            image_height,
            size_bytes,
        };
        self.descriptors.push(desc.clone());
        desc
    }

    /// Return statistics about managed weights.
    pub fn stats(&self) -> ImageMemoryStats {
        let mut s = ImageMemoryStats::default();
        for d in &self.descriptors {
            s.total_weights += 1;
            if d.is_image() {
                s.image_backed += 1;
                s.image_bytes += d.size_bytes as u64;
            } else {
                s.buffer_backed += 1;
                s.buffer_bytes += d.size_bytes as u64;
            }
        }
        s
    }

    /// Immutable view of all managed descriptors.
    pub fn descriptors(&self) -> &[WeightImageDescriptor] {
        &self.descriptors
    }

    /// Reset the manager, clearing all tracked descriptors.
    pub fn reset(&mut self) {
        self.descriptors.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn intel_arc_caps() -> ImageCapabilities {
        ImageCapabilities {
            images_supported: true,
            max_width: 16384,
            max_height: 16384,
            supported_formats: vec![
                ImageFormat::new(ImageChannelOrder::R, ImageChannelType::Float),
                ImageFormat::new(ImageChannelOrder::R, ImageChannelType::Half),
                ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelType::Float),
                ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelType::Half),
            ],
        }
    }

    #[test]
    fn test_image_format_bytes_per_pixel() {
        assert_eq!(
            ImageFormat::new(ImageChannelOrder::R, ImageChannelType::Float).bytes_per_pixel(),
            4
        );
        assert_eq!(
            ImageFormat::new(ImageChannelOrder::R, ImageChannelType::Half).bytes_per_pixel(),
            2
        );
        assert_eq!(
            ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelType::Float).bytes_per_pixel(),
            16
        );
        assert_eq!(
            ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelType::Half).bytes_per_pixel(),
            8
        );
    }

    #[test]
    fn test_best_format_prefers_rgba_half() {
        let caps = intel_arc_caps();
        let best = caps.best_format().expect("should find a format");
        assert_eq!(best.channel_order, ImageChannelOrder::Rgba);
        assert_eq!(best.channel_type, ImageChannelType::Half);
    }

    #[test]
    fn test_best_format_none_when_unsupported() {
        let caps = ImageCapabilities::unsupported();
        assert!(caps.best_format().is_none());
    }

    #[test]
    fn test_matrix_fits_within_limits() {
        let caps = intel_arc_caps();
        let fmt = ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelType::Half);
        // 4096×4096 -> image width = 4096/4 = 1024, height = 4096
        assert!(caps.matrix_fits(4096, 4096, &fmt));
    }

    #[test]
    fn test_matrix_exceeds_limits() {
        let caps = ImageCapabilities {
            images_supported: true,
            max_width: 1024,
            max_height: 1024,
            supported_formats: PREFERRED_FORMATS.to_vec(),
        };
        let fmt = ImageFormat::new(ImageChannelOrder::R, ImageChannelType::Float);
        // 2048 rows > max_height 1024
        assert!(!caps.matrix_fits(2048, 512, &fmt));
    }

    #[test]
    fn test_matrix_cols_not_divisible_by_channels() {
        let caps = intel_arc_caps();
        let fmt = ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelType::Half);
        // 1000 cols not divisible by 4
        assert!(!caps.matrix_fits(512, 1000, &fmt));
    }

    #[test]
    fn test_allocate_uses_image_when_possible() {
        let caps = intel_arc_caps();
        let mut mgr = ImageMemoryManager::new(caps);
        let desc = mgr.allocate("q_proj", 4096, 4096);
        assert!(desc.is_image());
        assert_eq!(desc.image_width, 4096 / 4); // RGBA = 4 channels
        assert_eq!(desc.image_height, 4096);
        assert_eq!(desc.backing, MemoryBacking::Image(
            ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelType::Half)
        ));
    }

    #[test]
    fn test_allocate_falls_back_to_buffer() {
        let caps = ImageCapabilities {
            images_supported: true,
            max_width: 256,
            max_height: 256,
            supported_formats: PREFERRED_FORMATS.to_vec(),
        };
        let mut mgr = ImageMemoryManager::new(caps);
        // 4096×4096 won't fit in 256×256 image
        let desc = mgr.allocate("huge_weight", 4096, 4096);
        assert!(!desc.is_image());
        assert_eq!(desc.backing, MemoryBacking::Buffer);
        assert_eq!(desc.image_width, 0);
    }

    #[test]
    fn test_buffer_only_manager() {
        let mut mgr = ImageMemoryManager::buffer_only();
        assert!(mgr.selected_format().is_none());
        let desc = mgr.allocate("w", 128, 128);
        assert_eq!(desc.backing, MemoryBacking::Buffer);
        assert_eq!(desc.size_bytes, 128 * 128 * 4);
    }

    #[test]
    fn test_stats_tracking() {
        let caps = ImageCapabilities {
            images_supported: true,
            max_width: 4096,
            max_height: 4096,
            supported_formats: PREFERRED_FORMATS.to_vec(),
        };
        let mut mgr = ImageMemoryManager::new(caps);
        // This fits: 256×256 -> image width 64, height 256 (within 4096)
        mgr.allocate("small", 256, 256);
        // This won't fit: 8192 rows > max 4096
        mgr.allocate("big", 8192, 8192);

        let stats = mgr.stats();
        assert_eq!(stats.total_weights, 2);
        assert_eq!(stats.image_backed, 1);
        assert_eq!(stats.buffer_backed, 1);
        assert!(stats.image_bytes > 0);
        assert!(stats.buffer_bytes > 0);
        assert!((stats.image_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_descriptor_display() {
        let desc = WeightImageDescriptor {
            name: "layers.0.q_proj".into(),
            rows: 4096,
            cols: 4096,
            backing: MemoryBacking::Image(
                ImageFormat::new(ImageChannelOrder::Rgba, ImageChannelType::Half),
            ),
            image_width: 1024,
            image_height: 4096,
            size_bytes: 1024 * 4096 * 8,
        };
        let s = desc.to_string();
        assert!(s.contains("layers.0.q_proj"));
        assert!(s.contains("4096×4096"));
        assert!(s.contains("Image"));
    }

    #[test]
    fn test_reset_clears_descriptors() {
        let caps = intel_arc_caps();
        let mut mgr = ImageMemoryManager::new(caps);
        mgr.allocate("w1", 128, 128);
        mgr.allocate("w2", 256, 256);
        assert_eq!(mgr.descriptors().len(), 2);
        mgr.reset();
        assert_eq!(mgr.descriptors().len(), 0);
        assert_eq!(mgr.stats().total_weights, 0);
    }
}
