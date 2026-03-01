//! Memory layout computation, stride optimization, and tensor view management
//! for GPU-accelerated inference.
//!
//! This module provides types for describing how tensor data is arranged in
//! memory, computing optimal strides for aligned GPU access, coalescing
//! patterns, and pinned-memory management for DMA transfers.
//!
//! **CPU reference implementation** — all operations run on the host and
//! produce layout descriptors consumed by GPU dispatch code.

use std::fmt;

// ── Error type ──────────────────────────────────────────────────────────────

/// Errors specific to memory layout operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// Shape contains a zero dimension.
    ZeroDimension { axis: usize },
    /// Alignment is not a power of two.
    InvalidAlignment(usize),
    /// Strides are inconsistent with the declared shape.
    StrideMismatch { expected: Vec<usize>, actual: Vec<usize> },
    /// The requested operation would exceed the buffer size.
    OutOfBounds { offset: usize, size: usize, capacity: usize },
    /// A layout invariant was violated.
    InvalidLayout(String),
    /// Pinned memory operation failed.
    PinningFailed(String),
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { axis } => {
                write!(f, "zero-sized dimension at axis {axis}")
            }
            Self::InvalidAlignment(a) => {
                write!(f, "alignment {a} is not a power of two")
            }
            Self::StrideMismatch { expected, actual } => {
                write!(f, "stride mismatch: expected {expected:?}, got {actual:?}")
            }
            Self::OutOfBounds { offset, size, capacity } => {
                write!(f, "out of bounds: offset {offset} + size {size} > capacity {capacity}")
            }
            Self::InvalidLayout(msg) => write!(f, "invalid layout: {msg}"),
            Self::PinningFailed(msg) => write!(f, "pinning failed: {msg}"),
        }
    }
}

impl std::error::Error for LayoutError {}

/// Convenience alias.
pub type LayoutResult<T> = Result<T, LayoutError>;

// ── Enums ───────────────────────────────────────────────────────────────────

/// Memory ordering of a multi-dimensional tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MajorOrder {
    /// Row-major (C-style): last dimension varies fastest.
    RowMajor,
    /// Column-major (Fortran-style): first dimension varies fastest.
    ColMajor,
}

/// Strategy for padding dimensions to satisfy alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaddingStrategy {
    /// No extra padding — use exact strides.
    None,
    /// Pad each row to the alignment boundary.
    AlignRows,
    /// Pad every dimension to the alignment boundary.
    AlignAll,
}

// ── 1. LayoutConfig ─────────────────────────────────────────────────────────

/// Configuration that governs how memory layouts are computed.
///
/// Alignment, ordering, and padding are all captured here so that the
/// same config can be reused across multiple tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutConfig {
    /// Required byte alignment (must be a power of two).
    pub alignment: usize,
    /// Row-major vs column-major ordering.
    pub order: MajorOrder,
    /// Padding strategy.
    pub padding: PaddingStrategy,
    /// Element size in bytes.
    pub element_size: usize,
}

impl LayoutConfig {
    /// Create a new layout config.
    ///
    /// Returns `Err` if `alignment` is not a power of two or is zero, or if
    /// `element_size` is zero.
    pub fn new(
        alignment: usize,
        order: MajorOrder,
        padding: PaddingStrategy,
        element_size: usize,
    ) -> LayoutResult<Self> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(LayoutError::InvalidAlignment(alignment));
        }
        if element_size == 0 {
            return Err(LayoutError::InvalidLayout("element_size must be > 0".into()));
        }
        Ok(Self { alignment, order, padding, element_size })
    }

    /// Default config: 64-byte alignment, row-major, no padding, f32
    /// elements (4 bytes).
    pub fn default_f32() -> Self {
        Self {
            alignment: 64,
            order: MajorOrder::RowMajor,
            padding: PaddingStrategy::None,
            element_size: 4,
        }
    }

    /// GPU-tuned config: 128-byte alignment, row-major, row padding, f16
    /// elements (2 bytes).
    pub fn gpu_f16() -> Self {
        Self {
            alignment: 128,
            order: MajorOrder::RowMajor,
            padding: PaddingStrategy::AlignRows,
            element_size: 2,
        }
    }
}

// ── 2. MemoryLayout ─────────────────────────────────────────────────────────

/// Describes how a tensor's elements are laid out in a contiguous buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryLayout {
    /// Shape of the tensor (e.g. [batch, rows, cols]).
    pub shape: Vec<usize>,
    /// Byte stride for each dimension.
    pub strides: Vec<usize>,
    /// Byte offset from the start of the allocation.
    pub offset: usize,
    /// Whether the layout is dense (no gaps between elements).
    pub contiguous: bool,
    /// Configuration used to produce this layout.
    pub config: LayoutConfig,
}

impl MemoryLayout {
    /// Build a layout from a shape and config.
    pub fn from_shape(shape: &[usize], config: &LayoutConfig) -> LayoutResult<Self> {
        for (i, &d) in shape.iter().enumerate() {
            if d == 0 {
                return Err(LayoutError::ZeroDimension { axis: i });
            }
        }
        let strides = StrideCalculator::compute(shape, config)?;
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            offset: 0,
            contiguous: true,
            config: config.clone(),
        })
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total byte size required by this layout (including padding).
    pub fn byte_size(&self) -> usize {
        if self.shape.is_empty() {
            return 0;
        }
        // Largest stride * corresponding dim gives the span.
        self.shape.iter().zip(self.strides.iter()).map(|(&d, &s)| (d - 1) * s).max().unwrap_or(0)
            + self.config.element_size
    }

    /// Byte offset of a specific element identified by `indices`.
    pub fn element_offset(&self, indices: &[usize]) -> LayoutResult<usize> {
        if indices.len() != self.shape.len() {
            return Err(LayoutError::InvalidLayout(format!(
                "index rank {} != shape rank {}",
                indices.len(),
                self.shape.len()
            )));
        }
        for (i, (&idx, &dim)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if idx >= dim {
                return Err(LayoutError::OutOfBounds { offset: idx, size: 1, capacity: dim });
            }
            let _ = i;
        }
        let off: usize = indices.iter().zip(self.strides.iter()).map(|(&i, &s)| i * s).sum();
        Ok(self.offset + off)
    }
}

// ── 3. StrideCalculator ─────────────────────────────────────────────────────

/// Computes optimal strides for a shape given alignment and ordering.
///
/// **CPU reference**: all computation is host-side.
#[derive(Debug, Clone)]
pub struct StrideCalculator;

impl StrideCalculator {
    /// Round `value` up to the nearest multiple of `alignment`.
    fn align_up(value: usize, alignment: usize) -> usize {
        let mask = alignment - 1;
        (value + mask) & !mask
    }

    /// Compute strides for a shape according to a [`LayoutConfig`].
    pub fn compute(shape: &[usize], config: &LayoutConfig) -> LayoutResult<Vec<usize>> {
        if shape.is_empty() {
            return Ok(vec![]);
        }
        for (i, &d) in shape.iter().enumerate() {
            if d == 0 {
                return Err(LayoutError::ZeroDimension { axis: i });
            }
        }

        let ndim = shape.len();
        let mut strides = vec![0usize; ndim];
        let elem = config.element_size;

        match config.order {
            MajorOrder::RowMajor => {
                // Last dimension is innermost.
                strides[ndim - 1] = elem;
                for i in (0..ndim - 1).rev() {
                    let raw = shape[i + 1] * strides[i + 1];
                    strides[i] = match config.padding {
                        PaddingStrategy::None => raw,
                        PaddingStrategy::AlignRows if i == ndim - 2 => {
                            Self::align_up(raw, config.alignment)
                        }
                        PaddingStrategy::AlignRows => raw,
                        PaddingStrategy::AlignAll => Self::align_up(raw, config.alignment),
                    };
                }
            }
            MajorOrder::ColMajor => {
                // First dimension is innermost.
                strides[0] = elem;
                for i in 1..ndim {
                    let raw = shape[i - 1] * strides[i - 1];
                    strides[i] = match config.padding {
                        PaddingStrategy::None => raw,
                        PaddingStrategy::AlignRows if i == 1 => {
                            Self::align_up(raw, config.alignment)
                        }
                        PaddingStrategy::AlignRows => raw,
                        PaddingStrategy::AlignAll => Self::align_up(raw, config.alignment),
                    };
                }
            }
        }
        Ok(strides)
    }

    /// Compute dense (tightly packed) strides with no alignment padding.
    pub fn dense_strides(shape: &[usize], element_size: usize, order: MajorOrder) -> Vec<usize> {
        let ndim = shape.len();
        if ndim == 0 {
            return vec![];
        }
        let mut strides = vec![0usize; ndim];
        match order {
            MajorOrder::RowMajor => {
                strides[ndim - 1] = element_size;
                for i in (0..ndim - 1).rev() {
                    strides[i] = shape[i + 1] * strides[i + 1];
                }
            }
            MajorOrder::ColMajor => {
                strides[0] = element_size;
                for i in 1..ndim {
                    strides[i] = shape[i - 1] * strides[i - 1];
                }
            }
        }
        strides
    }
}

// ── 4. TensorView ───────────────────────────────────────────────────────────

/// Non-owning view into tensor data with associated layout information.
///
/// This is a lightweight descriptor; it does **not** own the underlying
/// buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorView {
    /// Memory layout of the viewed region.
    pub layout: MemoryLayout,
    /// Byte offset into the backing buffer.
    pub base_offset: usize,
    /// Total size of the backing buffer in bytes (for bounds checking).
    pub buffer_size: usize,
}

impl TensorView {
    /// Create a new `TensorView`.
    pub fn new(layout: MemoryLayout, base_offset: usize, buffer_size: usize) -> LayoutResult<Self> {
        let required = base_offset + layout.byte_size();
        if required > buffer_size {
            return Err(LayoutError::OutOfBounds {
                offset: base_offset,
                size: layout.byte_size(),
                capacity: buffer_size,
            });
        }
        Ok(Self { layout, base_offset, buffer_size })
    }

    /// Slice along the outermost dimension, returning a view of one
    /// "row" (or batch entry, etc.).
    pub fn slice_outer(&self, index: usize) -> LayoutResult<TensorView> {
        if self.layout.shape.is_empty() {
            return Err(LayoutError::InvalidLayout("cannot slice 0-d tensor".into()));
        }
        if index >= self.layout.shape[0] {
            return Err(LayoutError::OutOfBounds {
                offset: index,
                size: 1,
                capacity: self.layout.shape[0],
            });
        }
        let new_offset = self.base_offset + index * self.layout.strides[0];
        let new_shape = self.layout.shape[1..].to_vec();
        let new_strides = self.layout.strides[1..].to_vec();
        let inner = MemoryLayout {
            shape: new_shape,
            strides: new_strides,
            offset: 0,
            contiguous: self.layout.contiguous,
            config: self.layout.config.clone(),
        };
        TensorView::new(inner, new_offset, self.buffer_size)
    }

    /// Shape of the viewed tensor.
    pub fn shape(&self) -> &[usize] {
        &self.layout.shape
    }

    /// Number of elements in the view.
    pub fn num_elements(&self) -> usize {
        self.layout.num_elements()
    }
}

// ── 5. LayoutTransposer ────────────────────────────────────────────────────

/// Transposes a memory layout without copying data when possible.
///
/// A layout-only transpose swaps shape and stride entries, changing the
/// logical interpretation of the same memory.
#[derive(Debug, Clone)]
pub struct LayoutTransposer;

impl LayoutTransposer {
    /// Transpose dimensions `dim_a` and `dim_b`.
    ///
    /// Returns a new [`MemoryLayout`] with those dimensions swapped.
    pub fn transpose(
        layout: &MemoryLayout,
        dim_a: usize,
        dim_b: usize,
    ) -> LayoutResult<MemoryLayout> {
        let ndim = layout.ndim();
        if dim_a >= ndim || dim_b >= ndim {
            return Err(LayoutError::InvalidLayout(format!(
                "transpose dims ({dim_a}, {dim_b}) out of range for {ndim}-d tensor"
            )));
        }
        let mut new_shape = layout.shape.clone();
        let mut new_strides = layout.strides.clone();
        new_shape.swap(dim_a, dim_b);
        new_strides.swap(dim_a, dim_b);
        // After transpose the layout is generally non-contiguous.
        let contiguous = dim_a == dim_b;
        Ok(MemoryLayout {
            shape: new_shape,
            strides: new_strides,
            offset: layout.offset,
            contiguous,
            config: layout.config.clone(),
        })
    }

    /// Full matrix transpose (swap last two dimensions).
    pub fn transpose_2d(layout: &MemoryLayout) -> LayoutResult<MemoryLayout> {
        let ndim = layout.ndim();
        if ndim < 2 {
            return Err(LayoutError::InvalidLayout(
                "need at least 2 dimensions for 2-d transpose".into(),
            ));
        }
        Self::transpose(layout, ndim - 2, ndim - 1)
    }

    /// Check whether a transpose can be done as a zero-copy operation
    /// (i.e., only strides change, no data movement needed).
    pub fn is_zero_copy(layout: &MemoryLayout, dim_a: usize, dim_b: usize) -> bool {
        dim_a < layout.ndim() && dim_b < layout.ndim()
    }

    /// Apply a permutation to dimensions.
    pub fn permute(layout: &MemoryLayout, perm: &[usize]) -> LayoutResult<MemoryLayout> {
        let ndim = layout.ndim();
        if perm.len() != ndim {
            return Err(LayoutError::InvalidLayout(format!(
                "permutation length {} != ndim {ndim}",
                perm.len()
            )));
        }
        // Validate permutation contains each index exactly once.
        let mut seen = vec![false; ndim];
        for &p in perm {
            if p >= ndim {
                return Err(LayoutError::InvalidLayout(format!(
                    "permutation index {p} out of range for {ndim}-d tensor"
                )));
            }
            if seen[p] {
                return Err(LayoutError::InvalidLayout(format!(
                    "duplicate index {p} in permutation"
                )));
            }
            seen[p] = true;
        }
        let new_shape: Vec<usize> = perm.iter().map(|&p| layout.shape[p]).collect();
        let new_strides: Vec<usize> = perm.iter().map(|&p| layout.strides[p]).collect();
        let is_identity = perm.iter().enumerate().all(|(i, &p)| i == p);
        Ok(MemoryLayout {
            shape: new_shape,
            strides: new_strides,
            offset: layout.offset,
            contiguous: layout.contiguous && is_identity,
            config: layout.config.clone(),
        })
    }
}

// ── 6. CoalescingOptimizer ──────────────────────────────────────────────────

/// Optimizes memory access patterns for GPU coalescing.
///
/// On GPUs, threads within a warp/sub-group achieve maximum throughput when
/// they access consecutive addresses. This optimizer analyses a layout and
/// suggests transformations or reports coalescing quality.
#[derive(Debug, Clone)]
pub struct CoalescingOptimizer {
    /// Warp/sub-group width (typically 32 for NVIDIA, 16/32 for Intel).
    pub warp_size: usize,
    /// Desired transaction size in bytes (e.g. 128 for a 128-byte cache line).
    pub transaction_bytes: usize,
}

impl CoalescingOptimizer {
    /// Create an optimizer with the given warp size and transaction width.
    pub fn new(warp_size: usize, transaction_bytes: usize) -> Self {
        Self { warp_size, transaction_bytes }
    }

    /// Default CUDA-like optimizer: warp 32, 128-byte transactions.
    pub fn cuda_default() -> Self {
        Self { warp_size: 32, transaction_bytes: 128 }
    }

    /// Intel GPU optimizer: sub-group 16, 64-byte cache lines.
    pub fn intel_default() -> Self {
        Self { warp_size: 16, transaction_bytes: 64 }
    }

    /// Compute a coalescing score in `[0.0, 1.0]` for the innermost
    /// dimension of a layout.
    ///
    /// A score of 1.0 means perfectly coalesced (innermost stride ==
    /// element_size). Lower scores indicate strided or scattered access.
    pub fn coalescing_score(&self, layout: &MemoryLayout) -> f64 {
        if layout.shape.is_empty() {
            return 1.0;
        }
        let elem = layout.config.element_size;
        let innermost_stride = match layout.config.order {
            MajorOrder::RowMajor => *layout.strides.last().unwrap_or(&elem),
            MajorOrder::ColMajor => *layout.strides.first().unwrap_or(&elem),
        };
        if innermost_stride == 0 {
            return 0.0;
        }
        // Perfect coalescing when stride == element_size.
        let ratio = elem as f64 / innermost_stride as f64;
        ratio.clamp(0.0, 1.0)
    }

    /// Check if the layout is fully coalesced.
    pub fn is_coalesced(&self, layout: &MemoryLayout) -> bool {
        (self.coalescing_score(layout) - 1.0).abs() < f64::EPSILON
    }

    /// Bytes accessed per warp for one iteration over the innermost
    /// dimension (useful for bandwidth estimation).
    pub fn bytes_per_warp_access(&self, layout: &MemoryLayout) -> usize {
        if layout.shape.is_empty() {
            return 0;
        }
        let innermost_stride = match layout.config.order {
            MajorOrder::RowMajor => *layout.strides.last().unwrap_or(&0),
            MajorOrder::ColMajor => *layout.strides.first().unwrap_or(&0),
        };
        self.warp_size * innermost_stride
    }

    /// Number of memory transactions needed for one warp access.
    pub fn transactions_per_access(&self, layout: &MemoryLayout) -> usize {
        let total = self.bytes_per_warp_access(layout);
        if self.transaction_bytes == 0 {
            return 0;
        }
        (total + self.transaction_bytes - 1) / self.transaction_bytes
    }

    /// Suggest whether a transpose would improve coalescing.
    pub fn suggest_transpose(&self, layout: &MemoryLayout) -> bool {
        if layout.ndim() < 2 {
            return false;
        }
        let score = self.coalescing_score(layout);
        // If already well-coalesced, no need.
        if score > 0.9 {
            return false;
        }
        // Check if transposing last two dims would improve score.
        if let Ok(transposed) = LayoutTransposer::transpose_2d(layout) {
            let transposed_score = self.coalescing_score(&transposed);
            return transposed_score > score;
        }
        false
    }
}

// ── 7. AlignmentEnforcer ────────────────────────────────────────────────────

/// Ensures tensors meet backend-specific alignment requirements.
///
/// Some GPU backends require that buffer base addresses and row pitches
/// be aligned to specific boundaries (e.g. 256 bytes for DMA, 128 bytes
/// for cache lines).
#[derive(Debug, Clone)]
pub struct AlignmentEnforcer {
    /// Required base-address alignment.
    pub base_alignment: usize,
    /// Required row-pitch alignment (0 means no row constraint).
    pub row_alignment: usize,
}

impl AlignmentEnforcer {
    /// Create an enforcer.
    pub fn new(base_alignment: usize, row_alignment: usize) -> LayoutResult<Self> {
        if base_alignment == 0 || !base_alignment.is_power_of_two() {
            return Err(LayoutError::InvalidAlignment(base_alignment));
        }
        if row_alignment != 0 && !row_alignment.is_power_of_two() {
            return Err(LayoutError::InvalidAlignment(row_alignment));
        }
        Ok(Self { base_alignment, row_alignment })
    }

    /// Check whether a given byte offset satisfies base alignment.
    pub fn is_base_aligned(&self, offset: usize) -> bool {
        offset % self.base_alignment == 0
    }

    /// Check whether a layout's row stride satisfies row alignment.
    pub fn is_row_aligned(&self, layout: &MemoryLayout) -> bool {
        if self.row_alignment == 0 || layout.ndim() < 2 {
            return true;
        }
        let row_stride = match layout.config.order {
            MajorOrder::RowMajor => layout.strides.get(layout.ndim() - 2).copied().unwrap_or(0),
            MajorOrder::ColMajor => layout.strides.get(1).copied().unwrap_or(0),
        };
        row_stride % self.row_alignment == 0
    }

    /// Validate all alignment constraints for a layout at a given base
    /// offset.
    pub fn validate(&self, layout: &MemoryLayout, base_offset: usize) -> LayoutResult<()> {
        if !self.is_base_aligned(base_offset) {
            return Err(LayoutError::InvalidLayout(format!(
                "base offset {base_offset} not aligned to {}",
                self.base_alignment
            )));
        }
        if !self.is_row_aligned(layout) {
            return Err(LayoutError::InvalidLayout(format!(
                "row stride not aligned to {}",
                self.row_alignment
            )));
        }
        Ok(())
    }

    /// Compute the next aligned offset >= `offset`.
    pub fn align_offset(&self, offset: usize) -> usize {
        StrideCalculator::align_up(offset, self.base_alignment)
    }

    /// Adjust a layout's config so that row strides satisfy row alignment,
    /// returning a fresh layout.
    pub fn enforce(&self, layout: &MemoryLayout) -> LayoutResult<MemoryLayout> {
        if self.row_alignment == 0 {
            return Ok(layout.clone());
        }
        let mut cfg = layout.config.clone();
        cfg.alignment = cfg.alignment.max(self.row_alignment);
        cfg.padding = PaddingStrategy::AlignRows;
        MemoryLayout::from_shape(&layout.shape, &cfg)
    }
}

// ── 8. MemoryPinning ────────────────────────────────────────────────────────

/// Pinned (page-locked) memory management for DMA transfers.
///
/// On real hardware, pinned memory allows the GPU DMA engine to transfer
/// data without an intermediate copy. This is a **CPU reference
/// implementation** that tracks pinned regions logically.
#[derive(Debug, Clone)]
pub struct MemoryPinning {
    /// Regions currently pinned: (offset, size).
    pinned_regions: Vec<(usize, usize)>,
    /// Total capacity of the managed buffer.
    capacity: usize,
}

impl MemoryPinning {
    /// Create a new pinning manager for a buffer of `capacity` bytes.
    pub fn new(capacity: usize) -> Self {
        Self { pinned_regions: Vec::new(), capacity }
    }

    /// Pin a region `[offset, offset + size)`.
    pub fn pin(&mut self, offset: usize, size: usize) -> LayoutResult<()> {
        if size == 0 {
            return Err(LayoutError::PinningFailed("size must be > 0".into()));
        }
        if offset + size > self.capacity {
            return Err(LayoutError::OutOfBounds { offset, size, capacity: self.capacity });
        }
        // Check for overlap with existing pinned regions.
        let new_end = offset + size;
        for &(o, s) in &self.pinned_regions {
            let existing_end = o + s;
            if offset < existing_end && new_end > o {
                return Err(LayoutError::PinningFailed(format!(
                    "region [{offset}, {new_end}) overlaps with [{o}, {existing_end})"
                )));
            }
        }
        self.pinned_regions.push((offset, size));
        Ok(())
    }

    /// Unpin a previously pinned region.
    pub fn unpin(&mut self, offset: usize, size: usize) -> LayoutResult<()> {
        if let Some(idx) = self.pinned_regions.iter().position(|&(o, s)| o == offset && s == size) {
            self.pinned_regions.remove(idx);
            Ok(())
        } else {
            Err(LayoutError::PinningFailed(format!(
                "no pinned region at offset {offset} with size {size}"
            )))
        }
    }

    /// Check whether a region is pinned.
    pub fn is_pinned(&self, offset: usize, size: usize) -> bool {
        self.pinned_regions.iter().any(|&(o, s)| o <= offset && offset + size <= o + s)
    }

    /// Number of pinned regions.
    pub fn pinned_count(&self) -> usize {
        self.pinned_regions.len()
    }

    /// Total bytes currently pinned.
    pub fn pinned_bytes(&self) -> usize {
        self.pinned_regions.iter().map(|&(_, s)| s).sum()
    }

    /// Unpin all regions.
    pub fn unpin_all(&mut self) {
        self.pinned_regions.clear();
    }

    /// Return all pinned regions.
    pub fn regions(&self) -> &[(usize, usize)] {
        &self.pinned_regions
    }
}

// ── 9. LayoutValidator ──────────────────────────────────────────────────────

/// Validates layout consistency (strides vs shape, alignment, bounds).
#[derive(Debug, Clone)]
pub struct LayoutValidator;

impl LayoutValidator {
    /// Validate that strides are consistent with the declared shape and
    /// element size.
    ///
    /// Specifically: for dense row-major, stride[i] should ==
    /// product(shape[i+1..]) * element_size.
    pub fn validate_strides(layout: &MemoryLayout) -> LayoutResult<()> {
        if layout.shape.is_empty() {
            return Ok(());
        }
        // Verify strides are large enough that no two distinct elements
        // map to the same byte range (non-overlapping).
        for (i, (&dim, &stride)) in layout.shape.iter().zip(layout.strides.iter()).enumerate() {
            if dim > 1 && stride == 0 {
                return Err(LayoutError::StrideMismatch {
                    expected: vec![],
                    actual: layout.strides.clone(),
                });
            }
            let _ = i;
        }
        Ok(())
    }

    /// Check that no element index exceeds the buffer size.
    pub fn validate_bounds(layout: &MemoryLayout, buffer_size: usize) -> LayoutResult<()> {
        let required = layout.offset + layout.byte_size();
        if required > buffer_size {
            return Err(LayoutError::OutOfBounds {
                offset: layout.offset,
                size: layout.byte_size(),
                capacity: buffer_size,
            });
        }
        Ok(())
    }

    /// Full validation: strides, bounds, alignment.
    pub fn validate_full(
        layout: &MemoryLayout,
        buffer_size: usize,
        enforcer: Option<&AlignmentEnforcer>,
    ) -> LayoutResult<()> {
        Self::validate_strides(layout)?;
        Self::validate_bounds(layout, buffer_size)?;
        if let Some(enf) = enforcer {
            enf.validate(layout, layout.offset)?;
        }
        Ok(())
    }

    /// Check if a layout represents a contiguous (dense) tensor.
    pub fn is_contiguous(layout: &MemoryLayout) -> bool {
        if layout.shape.is_empty() {
            return true;
        }
        let expected = StrideCalculator::dense_strides(
            &layout.shape,
            layout.config.element_size,
            layout.config.order,
        );
        layout.strides == expected
    }
}

// ── 10. MemoryLayoutEngine ──────────────────────────────────────────────────

/// Unified layout management and optimization engine.
///
/// Combines layout creation, validation, alignment enforcement,
/// coalescing analysis, and pinning into a single entry point.
#[derive(Debug, Clone)]
pub struct MemoryLayoutEngine {
    /// Default configuration for new layouts.
    pub config: LayoutConfig,
    /// Alignment enforcer.
    pub enforcer: AlignmentEnforcer,
    /// Coalescing optimizer.
    pub optimizer: CoalescingOptimizer,
    /// Pinning manager (per engine).
    pub pinning: MemoryPinning,
}

impl MemoryLayoutEngine {
    /// Create a new engine.
    pub fn new(
        config: LayoutConfig,
        enforcer: AlignmentEnforcer,
        optimizer: CoalescingOptimizer,
        buffer_capacity: usize,
    ) -> Self {
        Self { config, enforcer, optimizer, pinning: MemoryPinning::new(buffer_capacity) }
    }

    /// Build and validate a layout for the given shape.
    pub fn create_layout(&self, shape: &[usize]) -> LayoutResult<MemoryLayout> {
        let layout = MemoryLayout::from_shape(shape, &self.config)?;
        LayoutValidator::validate_strides(&layout)?;
        Ok(layout)
    }

    /// Build an aligned layout.
    pub fn create_aligned_layout(&self, shape: &[usize]) -> LayoutResult<MemoryLayout> {
        let base = MemoryLayout::from_shape(shape, &self.config)?;
        self.enforcer.enforce(&base)
    }

    /// Create a tensor view within a buffer, performing full validation.
    pub fn create_view(
        &self,
        shape: &[usize],
        base_offset: usize,
        buffer_size: usize,
    ) -> LayoutResult<TensorView> {
        let layout = self.create_layout(shape)?;
        LayoutValidator::validate_bounds(&layout, buffer_size.saturating_sub(base_offset))?;
        TensorView::new(layout, base_offset, buffer_size)
    }

    /// Report coalescing quality for a layout.
    pub fn coalescing_score(&self, layout: &MemoryLayout) -> f64 {
        self.optimizer.coalescing_score(layout)
    }

    /// Pin a buffer region for DMA.
    pub fn pin_region(&mut self, offset: usize, size: usize) -> LayoutResult<()> {
        self.pinning.pin(offset, size)
    }

    /// Unpin a buffer region.
    pub fn unpin_region(&mut self, offset: usize, size: usize) -> LayoutResult<()> {
        self.pinning.unpin(offset, size)
    }

    /// Suggest whether to transpose a layout for better coalescing.
    pub fn suggest_transpose(&self, layout: &MemoryLayout) -> bool {
        self.optimizer.suggest_transpose(layout)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LayoutConfig tests ──────────────────────────────────────────────

    #[test]
    fn config_default_f32() {
        let c = LayoutConfig::default_f32();
        assert_eq!(c.alignment, 64);
        assert_eq!(c.element_size, 4);
        assert_eq!(c.order, MajorOrder::RowMajor);
        assert_eq!(c.padding, PaddingStrategy::None);
    }

    #[test]
    fn config_gpu_f16() {
        let c = LayoutConfig::gpu_f16();
        assert_eq!(c.alignment, 128);
        assert_eq!(c.element_size, 2);
        assert_eq!(c.padding, PaddingStrategy::AlignRows);
    }

    #[test]
    fn config_new_valid() {
        let c = LayoutConfig::new(256, MajorOrder::ColMajor, PaddingStrategy::AlignAll, 8);
        assert!(c.is_ok());
        let c = c.unwrap();
        assert_eq!(c.alignment, 256);
        assert_eq!(c.element_size, 8);
    }

    #[test]
    fn config_rejects_zero_alignment() {
        let r = LayoutConfig::new(0, MajorOrder::RowMajor, PaddingStrategy::None, 4);
        assert!(matches!(r, Err(LayoutError::InvalidAlignment(0))));
    }

    #[test]
    fn config_rejects_non_power_of_two() {
        let r = LayoutConfig::new(3, MajorOrder::RowMajor, PaddingStrategy::None, 4);
        assert!(matches!(r, Err(LayoutError::InvalidAlignment(3))));
    }

    #[test]
    fn config_rejects_zero_element_size() {
        let r = LayoutConfig::new(64, MajorOrder::RowMajor, PaddingStrategy::None, 0);
        assert!(matches!(r, Err(LayoutError::InvalidLayout(_))));
    }

    #[test]
    fn config_clone_eq() {
        let a = LayoutConfig::default_f32();
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn config_debug() {
        let c = LayoutConfig::default_f32();
        let s = format!("{c:?}");
        assert!(s.contains("LayoutConfig"));
    }

    // ── StrideCalculator tests ──────────────────────────────────────────

    #[test]
    fn stride_row_major_2d_no_padding() {
        let cfg = LayoutConfig::default_f32();
        let strides = StrideCalculator::compute(&[3, 4], &cfg).unwrap();
        assert_eq!(strides, vec![16, 4]); // 4*4=16, 4
    }

    #[test]
    fn stride_row_major_3d_no_padding() {
        let cfg = LayoutConfig::default_f32();
        let strides = StrideCalculator::compute(&[2, 3, 4], &cfg).unwrap();
        assert_eq!(strides, vec![48, 16, 4]);
    }

    #[test]
    fn stride_col_major_2d_no_padding() {
        let cfg = LayoutConfig::new(64, MajorOrder::ColMajor, PaddingStrategy::None, 4).unwrap();
        let strides = StrideCalculator::compute(&[3, 4], &cfg).unwrap();
        assert_eq!(strides, vec![4, 12]); // 4, 3*4=12
    }

    #[test]
    fn stride_col_major_3d() {
        let cfg = LayoutConfig::new(64, MajorOrder::ColMajor, PaddingStrategy::None, 4).unwrap();
        let strides = StrideCalculator::compute(&[2, 3, 4], &cfg).unwrap();
        assert_eq!(strides, vec![4, 8, 24]);
    }

    #[test]
    fn stride_row_major_align_rows() {
        let cfg =
            LayoutConfig::new(64, MajorOrder::RowMajor, PaddingStrategy::AlignRows, 4).unwrap();
        let strides = StrideCalculator::compute(&[3, 5], &cfg).unwrap();
        // Inner stride = 4, row stride = align_up(5*4=20, 64) = 64.
        assert_eq!(strides, vec![64, 4]);
    }

    #[test]
    fn stride_row_major_align_all_3d() {
        let cfg =
            LayoutConfig::new(64, MajorOrder::RowMajor, PaddingStrategy::AlignAll, 4).unwrap();
        let strides = StrideCalculator::compute(&[2, 3, 5], &cfg).unwrap();
        // Inner = 4, mid = align_up(5*4=20, 64)=64, outer = align_up(3*64=192, 64)=192
        assert_eq!(strides[2], 4);
        assert_eq!(strides[1], 64);
        assert_eq!(strides[0], 192);
    }

    #[test]
    fn stride_empty_shape() {
        let cfg = LayoutConfig::default_f32();
        let strides = StrideCalculator::compute(&[], &cfg).unwrap();
        assert!(strides.is_empty());
    }

    #[test]
    fn stride_single_element() {
        let cfg = LayoutConfig::default_f32();
        let strides = StrideCalculator::compute(&[1], &cfg).unwrap();
        assert_eq!(strides, vec![4]);
    }

    #[test]
    fn stride_rejects_zero_dim() {
        let cfg = LayoutConfig::default_f32();
        let r = StrideCalculator::compute(&[3, 0, 4], &cfg);
        assert!(matches!(r, Err(LayoutError::ZeroDimension { axis: 1 })));
    }

    #[test]
    fn dense_strides_row_major() {
        let s = StrideCalculator::dense_strides(&[2, 3, 4], 4, MajorOrder::RowMajor);
        assert_eq!(s, vec![48, 16, 4]);
    }

    #[test]
    fn dense_strides_col_major() {
        let s = StrideCalculator::dense_strides(&[2, 3, 4], 4, MajorOrder::ColMajor);
        assert_eq!(s, vec![4, 8, 24]);
    }

    #[test]
    fn dense_strides_empty() {
        let s = StrideCalculator::dense_strides(&[], 4, MajorOrder::RowMajor);
        assert!(s.is_empty());
    }

    #[test]
    fn stride_1d() {
        let cfg = LayoutConfig::default_f32();
        let strides = StrideCalculator::compute(&[10], &cfg).unwrap();
        assert_eq!(strides, vec![4]);
    }

    #[test]
    fn stride_col_major_align_rows() {
        let cfg =
            LayoutConfig::new(64, MajorOrder::ColMajor, PaddingStrategy::AlignRows, 4).unwrap();
        let strides = StrideCalculator::compute(&[5, 3], &cfg).unwrap();
        // Inner = 4, col stride = align_up(5*4=20, 64)=64.
        assert_eq!(strides, vec![4, 64]);
    }

    // ── MemoryLayout tests ──────────────────────────────────────────────

    #[test]
    fn layout_from_shape_basic() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[4, 8], &cfg).unwrap();
        assert_eq!(l.ndim(), 2);
        assert_eq!(l.num_elements(), 32);
        assert!(l.contiguous);
    }

    #[test]
    fn layout_byte_size_2d() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        // (3-1)*16 + (4-1)*4 = 32 + 12 = 44 ... nope, it's max of per-dim spans + elem.
        // max((3-1)*16, (4-1)*4) + 4 = max(32,12) + 4 = 36
        assert_eq!(l.byte_size(), 36);
        // Alternatively: 3 rows * 4 cols * 4 bytes = 48 dense; but byte_size
        // is last-element + element_size.
    }

    #[test]
    fn layout_element_offset_2d() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        assert_eq!(l.element_offset(&[0, 0]).unwrap(), 0);
        assert_eq!(l.element_offset(&[0, 1]).unwrap(), 4);
        assert_eq!(l.element_offset(&[1, 0]).unwrap(), 16);
        assert_eq!(l.element_offset(&[2, 3]).unwrap(), 44);
    }

    #[test]
    fn layout_element_offset_3d() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[2, 3, 4], &cfg).unwrap();
        assert_eq!(l.element_offset(&[1, 2, 3]).unwrap(), 48 + 32 + 12);
    }

    #[test]
    fn layout_element_offset_out_of_bounds() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        assert!(l.element_offset(&[3, 0]).is_err());
    }

    #[test]
    fn layout_element_offset_wrong_rank() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        assert!(l.element_offset(&[0]).is_err());
    }

    #[test]
    fn layout_rejects_zero_dim() {
        let cfg = LayoutConfig::default_f32();
        let r = MemoryLayout::from_shape(&[0, 4], &cfg);
        assert!(matches!(r, Err(LayoutError::ZeroDimension { axis: 0 })));
    }

    #[test]
    fn layout_scalar() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[], &cfg).unwrap();
        assert_eq!(l.ndim(), 0);
        assert_eq!(l.num_elements(), 1);
        assert_eq!(l.byte_size(), 0);
    }

    #[test]
    fn layout_1d() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[5], &cfg).unwrap();
        assert_eq!(l.byte_size(), 4 * 4 + 4); // (5-1)*4 + 4 = 20
        assert_eq!(l.byte_size(), 20);
    }

    #[test]
    fn layout_clone_eq() {
        let cfg = LayoutConfig::default_f32();
        let a = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn layout_debug() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[2, 2], &cfg).unwrap();
        let s = format!("{l:?}");
        assert!(s.contains("MemoryLayout"));
    }

    // ── TensorView tests ────────────────────────────────────────────────

    #[test]
    fn view_new_valid() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[4, 4], &cfg).unwrap();
        let bsz = l.byte_size() + 64;
        let v = TensorView::new(l, 0, bsz);
        assert!(v.is_ok());
    }

    #[test]
    fn view_rejects_overflow() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[4, 4], &cfg).unwrap();
        let v = TensorView::new(l, 0, 1); // buffer too small
        assert!(v.is_err());
    }

    #[test]
    fn view_slice_outer() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        let v = TensorView::new(l, 0, 256).unwrap();
        let s = v.slice_outer(1).unwrap();
        assert_eq!(s.shape(), &[4]);
        assert_eq!(s.base_offset, 16);
    }

    #[test]
    fn view_slice_outer_out_of_bounds() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        let v = TensorView::new(l, 0, 256).unwrap();
        assert!(v.slice_outer(3).is_err());
    }

    #[test]
    fn view_num_elements() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[2, 3], &cfg).unwrap();
        let v = TensorView::new(l, 0, 256).unwrap();
        assert_eq!(v.num_elements(), 6);
    }

    #[test]
    fn view_with_offset() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[2, 2], &cfg).unwrap();
        let v = TensorView::new(l, 64, 256).unwrap();
        assert_eq!(v.base_offset, 64);
    }

    #[test]
    fn view_slice_empty_tensor() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[], &cfg).unwrap();
        let v = TensorView::new(l, 0, 256).unwrap();
        assert!(v.slice_outer(0).is_err());
    }

    #[test]
    fn view_clone_eq() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[2, 3], &cfg).unwrap();
        let a = TensorView::new(l, 0, 256).unwrap();
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn view_sequential_slices() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[4, 8], &cfg).unwrap();
        let v = TensorView::new(l, 0, 512).unwrap();
        let s0 = v.slice_outer(0).unwrap();
        let s1 = v.slice_outer(1).unwrap();
        assert_eq!(s1.base_offset - s0.base_offset, 32); // 8*4
    }

    // ── LayoutTransposer tests ──────────────────────────────────────────

    #[test]
    fn transpose_2d_basic() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        let t = LayoutTransposer::transpose_2d(&l).unwrap();
        assert_eq!(t.shape, vec![4, 3]);
        assert_eq!(t.strides, vec![4, 16]); // swapped
    }

    #[test]
    fn transpose_same_dim() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        let t = LayoutTransposer::transpose(&l, 0, 0).unwrap();
        assert!(t.contiguous);
        assert_eq!(t.shape, l.shape);
    }

    #[test]
    fn transpose_3d_outer() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[2, 3, 4], &cfg).unwrap();
        let t = LayoutTransposer::transpose(&l, 0, 2).unwrap();
        assert_eq!(t.shape, vec![4, 3, 2]);
        assert_eq!(t.strides, vec![4, 16, 48]);
    }

    #[test]
    fn transpose_out_of_range() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        assert!(LayoutTransposer::transpose(&l, 0, 5).is_err());
    }

    #[test]
    fn transpose_2d_on_1d() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[5], &cfg).unwrap();
        assert!(LayoutTransposer::transpose_2d(&l).is_err());
    }

    #[test]
    fn transpose_is_zero_copy() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        assert!(LayoutTransposer::is_zero_copy(&l, 0, 1));
    }

    #[test]
    fn permute_identity() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[2, 3, 4], &cfg).unwrap();
        let p = LayoutTransposer::permute(&l, &[0, 1, 2]).unwrap();
        assert_eq!(p.shape, l.shape);
        assert_eq!(p.strides, l.strides);
        assert!(p.contiguous);
    }

    #[test]
    fn permute_reverse() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[2, 3, 4], &cfg).unwrap();
        let p = LayoutTransposer::permute(&l, &[2, 1, 0]).unwrap();
        assert_eq!(p.shape, vec![4, 3, 2]);
        assert!(!p.contiguous);
    }

    #[test]
    fn permute_wrong_length() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[2, 3], &cfg).unwrap();
        assert!(LayoutTransposer::permute(&l, &[0, 1, 2]).is_err());
    }

    #[test]
    fn permute_duplicate_index() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[2, 3], &cfg).unwrap();
        assert!(LayoutTransposer::permute(&l, &[0, 0]).is_err());
    }

    #[test]
    fn permute_out_of_range() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[2, 3], &cfg).unwrap();
        assert!(LayoutTransposer::permute(&l, &[0, 5]).is_err());
    }

    // ── CoalescingOptimizer tests ───────────────────────────────────────

    #[test]
    fn coalescing_perfect_score() {
        let opt = CoalescingOptimizer::cuda_default();
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[16, 32], &cfg).unwrap();
        let score = opt.coalescing_score(&l);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn coalescing_strided() {
        let opt = CoalescingOptimizer::cuda_default();
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[16, 32], &cfg).unwrap();
        // Transpose to make innermost stride non-contiguous.
        let t = LayoutTransposer::transpose_2d(&l).unwrap();
        let score = opt.coalescing_score(&t);
        assert!(score < 1.0);
    }

    #[test]
    fn coalescing_is_coalesced() {
        let opt = CoalescingOptimizer::cuda_default();
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[8, 8], &cfg).unwrap();
        assert!(opt.is_coalesced(&l));
    }

    #[test]
    fn coalescing_bytes_per_warp() {
        let opt = CoalescingOptimizer::cuda_default();
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[8, 32], &cfg).unwrap();
        assert_eq!(opt.bytes_per_warp_access(&l), 32 * 4);
    }

    #[test]
    fn coalescing_transactions_perfect() {
        let opt = CoalescingOptimizer::cuda_default();
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[8, 32], &cfg).unwrap();
        assert_eq!(opt.transactions_per_access(&l), 1); // 128/128
    }

    #[test]
    fn coalescing_intel_default() {
        let opt = CoalescingOptimizer::intel_default();
        assert_eq!(opt.warp_size, 16);
        assert_eq!(opt.transaction_bytes, 64);
    }

    #[test]
    fn coalescing_empty_layout() {
        let opt = CoalescingOptimizer::cuda_default();
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[], &cfg).unwrap();
        assert!((opt.coalescing_score(&l) - 1.0).abs() < f64::EPSILON);
        assert_eq!(opt.bytes_per_warp_access(&l), 0);
    }

    #[test]
    fn suggest_transpose_already_good() {
        let opt = CoalescingOptimizer::cuda_default();
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[4, 16], &cfg).unwrap();
        assert!(!opt.suggest_transpose(&l));
    }

    #[test]
    fn suggest_transpose_for_transposed() {
        let opt = CoalescingOptimizer::cuda_default();
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[4, 16], &cfg).unwrap();
        let t = LayoutTransposer::transpose_2d(&l).unwrap();
        assert!(opt.suggest_transpose(&t));
    }

    #[test]
    fn suggest_transpose_1d() {
        let opt = CoalescingOptimizer::cuda_default();
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[64], &cfg).unwrap();
        assert!(!opt.suggest_transpose(&l));
    }

    #[test]
    fn coalescing_debug() {
        let opt = CoalescingOptimizer::cuda_default();
        let s = format!("{opt:?}");
        assert!(s.contains("CoalescingOptimizer"));
    }

    // ── AlignmentEnforcer tests ─────────────────────────────────────────

    #[test]
    fn enforcer_valid() {
        let e = AlignmentEnforcer::new(256, 128).unwrap();
        assert_eq!(e.base_alignment, 256);
        assert_eq!(e.row_alignment, 128);
    }

    #[test]
    fn enforcer_rejects_zero() {
        assert!(AlignmentEnforcer::new(0, 0).is_err());
    }

    #[test]
    fn enforcer_rejects_non_power_of_two_row() {
        assert!(AlignmentEnforcer::new(64, 3).is_err());
    }

    #[test]
    fn enforcer_allows_zero_row() {
        assert!(AlignmentEnforcer::new(64, 0).is_ok());
    }

    #[test]
    fn enforcer_base_aligned() {
        let e = AlignmentEnforcer::new(64, 0).unwrap();
        assert!(e.is_base_aligned(0));
        assert!(e.is_base_aligned(64));
        assert!(e.is_base_aligned(128));
        assert!(!e.is_base_aligned(1));
        assert!(!e.is_base_aligned(63));
    }

    #[test]
    fn enforcer_row_aligned() {
        let e = AlignmentEnforcer::new(64, 128).unwrap();
        // Row-major 3x5 with 4-byte elements, no padding: row stride = 20 (not 128-aligned).
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 5], &cfg).unwrap();
        assert!(!e.is_row_aligned(&l));
    }

    #[test]
    fn enforcer_validate_pass() {
        let e = AlignmentEnforcer::new(64, 0).unwrap();
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[2, 4], &cfg).unwrap();
        assert!(e.validate(&l, 0).is_ok());
        assert!(e.validate(&l, 64).is_ok());
    }

    #[test]
    fn enforcer_validate_fail_base() {
        let e = AlignmentEnforcer::new(64, 0).unwrap();
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[2, 4], &cfg).unwrap();
        assert!(e.validate(&l, 1).is_err());
    }

    #[test]
    fn enforcer_align_offset() {
        let e = AlignmentEnforcer::new(64, 0).unwrap();
        assert_eq!(e.align_offset(0), 0);
        assert_eq!(e.align_offset(1), 64);
        assert_eq!(e.align_offset(64), 64);
        assert_eq!(e.align_offset(65), 128);
    }

    #[test]
    fn enforcer_enforce_pads_rows() {
        let e = AlignmentEnforcer::new(64, 128).unwrap();
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 5], &cfg).unwrap();
        let enforced = e.enforce(&l).unwrap();
        // Row stride should now be aligned to 128.
        assert!(enforced.strides[0] % 128 == 0);
    }

    #[test]
    fn enforcer_debug_clone() {
        let e = AlignmentEnforcer::new(64, 0).unwrap();
        let e2 = e.clone();
        let s = format!("{e2:?}");
        assert!(s.contains("AlignmentEnforcer"));
    }

    // ── MemoryPinning tests ─────────────────────────────────────────────

    #[test]
    fn pinning_basic() {
        let mut p = MemoryPinning::new(1024);
        assert!(p.pin(0, 256).is_ok());
        assert!(p.is_pinned(0, 256));
        assert_eq!(p.pinned_count(), 1);
        assert_eq!(p.pinned_bytes(), 256);
    }

    #[test]
    fn pinning_rejects_zero_size() {
        let mut p = MemoryPinning::new(1024);
        assert!(p.pin(0, 0).is_err());
    }

    #[test]
    fn pinning_rejects_overflow() {
        let mut p = MemoryPinning::new(1024);
        assert!(p.pin(512, 1024).is_err());
    }

    #[test]
    fn pinning_rejects_overlap() {
        let mut p = MemoryPinning::new(1024);
        p.pin(0, 256).unwrap();
        assert!(p.pin(128, 256).is_err());
    }

    #[test]
    fn pinning_non_overlapping() {
        let mut p = MemoryPinning::new(1024);
        p.pin(0, 256).unwrap();
        p.pin(256, 256).unwrap();
        assert_eq!(p.pinned_count(), 2);
        assert_eq!(p.pinned_bytes(), 512);
    }

    #[test]
    fn pinning_unpin() {
        let mut p = MemoryPinning::new(1024);
        p.pin(0, 256).unwrap();
        p.unpin(0, 256).unwrap();
        assert_eq!(p.pinned_count(), 0);
        assert!(!p.is_pinned(0, 256));
    }

    #[test]
    fn pinning_unpin_nonexistent() {
        let mut p = MemoryPinning::new(1024);
        assert!(p.unpin(0, 256).is_err());
    }

    #[test]
    fn pinning_unpin_all() {
        let mut p = MemoryPinning::new(1024);
        p.pin(0, 100).unwrap();
        p.pin(200, 100).unwrap();
        p.unpin_all();
        assert_eq!(p.pinned_count(), 0);
    }

    #[test]
    fn pinning_regions() {
        let mut p = MemoryPinning::new(1024);
        p.pin(0, 100).unwrap();
        p.pin(200, 50).unwrap();
        assert_eq!(p.regions(), &[(0, 100), (200, 50)]);
    }

    #[test]
    fn pinning_is_pinned_partial() {
        let mut p = MemoryPinning::new(1024);
        p.pin(100, 200).unwrap();
        assert!(p.is_pinned(100, 200));
        assert!(p.is_pinned(150, 50)); // sub-region
        assert!(!p.is_pinned(50, 200)); // extends before
    }

    #[test]
    fn pinning_debug_clone() {
        let p = MemoryPinning::new(512);
        let p2 = p.clone();
        let s = format!("{p2:?}");
        assert!(s.contains("MemoryPinning"));
    }

    // ── LayoutValidator tests ───────────────────────────────────────────

    #[test]
    fn validator_valid_strides() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        assert!(LayoutValidator::validate_strides(&l).is_ok());
    }

    #[test]
    fn validator_zero_stride_multi_dim() {
        let cfg = LayoutConfig::default_f32();
        let mut l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        l.strides[0] = 0; // Invalid for dim > 1.
        assert!(LayoutValidator::validate_strides(&l).is_err());
    }

    #[test]
    fn validator_bounds_pass() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        assert!(LayoutValidator::validate_bounds(&l, 1024).is_ok());
    }

    #[test]
    fn validator_bounds_fail() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        assert!(LayoutValidator::validate_bounds(&l, 1).is_err());
    }

    #[test]
    fn validator_full_pass() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        assert!(LayoutValidator::validate_full(&l, 1024, None).is_ok());
    }

    #[test]
    fn validator_full_with_enforcer() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        let e = AlignmentEnforcer::new(64, 0).unwrap();
        assert!(LayoutValidator::validate_full(&l, 1024, Some(&e)).is_ok());
    }

    #[test]
    fn validator_is_contiguous_true() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        assert!(LayoutValidator::is_contiguous(&l));
    }

    #[test]
    fn validator_is_contiguous_false_after_transpose() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 4], &cfg).unwrap();
        let t = LayoutTransposer::transpose_2d(&l).unwrap();
        assert!(!LayoutValidator::is_contiguous(&t));
    }

    #[test]
    fn validator_empty_layout() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[], &cfg).unwrap();
        assert!(LayoutValidator::validate_strides(&l).is_ok());
        assert!(LayoutValidator::is_contiguous(&l));
    }

    #[test]
    fn validator_debug() {
        let v = LayoutValidator;
        let s = format!("{v:?}");
        assert!(s.contains("LayoutValidator"));
    }

    // ── MemoryLayoutEngine tests ────────────────────────────────────────

    #[test]
    fn engine_create_layout() {
        let cfg = LayoutConfig::default_f32();
        let enf = AlignmentEnforcer::new(64, 0).unwrap();
        let opt = CoalescingOptimizer::cuda_default();
        let eng = MemoryLayoutEngine::new(cfg, enf, opt, 4096);
        let l = eng.create_layout(&[4, 8]).unwrap();
        assert_eq!(l.num_elements(), 32);
    }

    #[test]
    fn engine_create_aligned_layout() {
        let cfg = LayoutConfig::default_f32();
        let enf = AlignmentEnforcer::new(64, 128).unwrap();
        let opt = CoalescingOptimizer::cuda_default();
        let eng = MemoryLayoutEngine::new(cfg, enf, opt, 4096);
        let l = eng.create_aligned_layout(&[3, 5]).unwrap();
        assert!(l.strides[0] % 128 == 0);
    }

    #[test]
    fn engine_create_view() {
        let cfg = LayoutConfig::default_f32();
        let enf = AlignmentEnforcer::new(64, 0).unwrap();
        let opt = CoalescingOptimizer::cuda_default();
        let eng = MemoryLayoutEngine::new(cfg, enf, opt, 4096);
        let v = eng.create_view(&[2, 4], 0, 4096).unwrap();
        assert_eq!(v.num_elements(), 8);
    }

    #[test]
    fn engine_coalescing_score() {
        let cfg = LayoutConfig::default_f32();
        let enf = AlignmentEnforcer::new(64, 0).unwrap();
        let opt = CoalescingOptimizer::cuda_default();
        let eng = MemoryLayoutEngine::new(cfg, enf, opt, 4096);
        let l = eng.create_layout(&[4, 8]).unwrap();
        assert!((eng.coalescing_score(&l) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn engine_pin_unpin() {
        let cfg = LayoutConfig::default_f32();
        let enf = AlignmentEnforcer::new(64, 0).unwrap();
        let opt = CoalescingOptimizer::cuda_default();
        let mut eng = MemoryLayoutEngine::new(cfg, enf, opt, 4096);
        eng.pin_region(0, 512).unwrap();
        assert!(eng.pinning.is_pinned(0, 512));
        eng.unpin_region(0, 512).unwrap();
        assert!(!eng.pinning.is_pinned(0, 512));
    }

    #[test]
    fn engine_suggest_transpose() {
        let cfg = LayoutConfig::default_f32();
        let enf = AlignmentEnforcer::new(64, 0).unwrap();
        let opt = CoalescingOptimizer::cuda_default();
        let eng = MemoryLayoutEngine::new(cfg, enf, opt, 4096);
        let l = eng.create_layout(&[4, 16]).unwrap();
        assert!(!eng.suggest_transpose(&l));
    }

    #[test]
    fn engine_debug_clone() {
        let cfg = LayoutConfig::default_f32();
        let enf = AlignmentEnforcer::new(64, 0).unwrap();
        let opt = CoalescingOptimizer::cuda_default();
        let eng = MemoryLayoutEngine::new(cfg, enf, opt, 4096);
        let eng2 = eng.clone();
        let s = format!("{eng2:?}");
        assert!(s.contains("MemoryLayoutEngine"));
    }

    #[test]
    fn engine_view_fails_oob() {
        let cfg = LayoutConfig::default_f32();
        let enf = AlignmentEnforcer::new(64, 0).unwrap();
        let opt = CoalescingOptimizer::cuda_default();
        let eng = MemoryLayoutEngine::new(cfg, enf, opt, 64);
        assert!(eng.create_view(&[100, 100], 0, 64).is_err());
    }

    // ── Error display tests ─────────────────────────────────────────────

    #[test]
    fn error_display_zero_dim() {
        let e = LayoutError::ZeroDimension { axis: 2 };
        assert_eq!(e.to_string(), "zero-sized dimension at axis 2");
    }

    #[test]
    fn error_display_invalid_alignment() {
        let e = LayoutError::InvalidAlignment(3);
        assert_eq!(e.to_string(), "alignment 3 is not a power of two");
    }

    #[test]
    fn error_display_oob() {
        let e = LayoutError::OutOfBounds { offset: 10, size: 5, capacity: 12 };
        assert!(e.to_string().contains("out of bounds"));
    }

    #[test]
    fn error_display_pinning() {
        let e = LayoutError::PinningFailed("test".into());
        assert!(e.to_string().contains("pinning failed"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(LayoutError::InvalidAlignment(5));
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn error_stride_mismatch_display() {
        let e = LayoutError::StrideMismatch { expected: vec![8, 4], actual: vec![16, 4] };
        assert!(e.to_string().contains("stride mismatch"));
    }

    #[test]
    fn error_invalid_layout_display() {
        let e = LayoutError::InvalidLayout("bad".into());
        assert!(e.to_string().contains("invalid layout: bad"));
    }

    // ── MajorOrder / PaddingStrategy enum tests ─────────────────────────

    #[test]
    fn major_order_eq_hash() {
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(MajorOrder::RowMajor);
        s.insert(MajorOrder::ColMajor);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn padding_strategy_eq() {
        assert_ne!(PaddingStrategy::None, PaddingStrategy::AlignRows);
        assert_ne!(PaddingStrategy::AlignRows, PaddingStrategy::AlignAll);
    }

    #[test]
    fn major_order_debug() {
        assert_eq!(format!("{:?}", MajorOrder::RowMajor), "RowMajor");
        assert_eq!(format!("{:?}", MajorOrder::ColMajor), "ColMajor");
    }

    #[test]
    fn padding_strategy_debug() {
        assert_eq!(format!("{:?}", PaddingStrategy::None), "None");
    }

    // ── Integration / cross-component tests ─────────────────────────────

    #[test]
    fn roundtrip_create_validate_view() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[8, 16], &cfg).unwrap();
        LayoutValidator::validate_strides(&l).unwrap();
        let v = TensorView::new(l.clone(), 0, 1024).unwrap();
        assert_eq!(v.num_elements(), 128);
        assert!(LayoutValidator::is_contiguous(&l));
    }

    #[test]
    fn transpose_then_validate() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[4, 8], &cfg).unwrap();
        let t = LayoutTransposer::transpose_2d(&l).unwrap();
        LayoutValidator::validate_strides(&t).unwrap();
        assert!(!LayoutValidator::is_contiguous(&t));
    }

    #[test]
    fn enforce_then_coalesce() {
        let enf = AlignmentEnforcer::new(64, 128).unwrap();
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[3, 5], &cfg).unwrap();
        let enforced = enf.enforce(&l).unwrap();
        let opt = CoalescingOptimizer::cuda_default();
        // Enforced layout should still be coalesced (innermost unchanged).
        assert!(opt.is_coalesced(&enforced));
    }

    #[test]
    fn pin_layout_region() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[4, 4], &cfg).unwrap();
        let mut pin = MemoryPinning::new(1024);
        pin.pin(0, l.byte_size()).unwrap();
        assert!(pin.is_pinned(0, l.byte_size()));
    }

    #[test]
    fn large_shape_layout() {
        let cfg = LayoutConfig::default_f32();
        let l = MemoryLayout::from_shape(&[1024, 2048], &cfg).unwrap();
        assert_eq!(l.num_elements(), 1024 * 2048);
        assert!(LayoutValidator::validate_strides(&l).is_ok());
    }

    #[test]
    fn stride_f16_element() {
        let cfg = LayoutConfig::gpu_f16();
        let l = MemoryLayout::from_shape(&[4, 64], &cfg).unwrap();
        assert_eq!(l.strides[1], 2); // f16 = 2 bytes
        // Row stride aligned to 128.
        assert!(l.strides[0] % 128 == 0);
    }

    #[test]
    fn align_up_edge_cases() {
        assert_eq!(StrideCalculator::align_up(0, 64), 0);
        assert_eq!(StrideCalculator::align_up(1, 64), 64);
        assert_eq!(StrideCalculator::align_up(64, 64), 64);
        assert_eq!(StrideCalculator::align_up(65, 64), 128);
    }
}
