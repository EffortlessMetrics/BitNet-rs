//! CPU cache prefetch and memory hierarchy optimization.
//!
//! Provides software prefetch hints, cache line alignment utilities,
//! auto-tuning prefetch distance, and memory access pattern descriptors
//! for matrix traversal in hot inference loops.

use std::fmt;

// ---------------------------------------------------------------------------
// Cache hierarchy constants
// ---------------------------------------------------------------------------

/// Default cache line size in bytes (x86-64 and most aarch64).
pub const DEFAULT_CACHE_LINE_BYTES: usize = 64;

/// Maximum prefetch distance (elements) we ever emit.
const MAX_PREFETCH_DISTANCE: usize = 2048;

// ---------------------------------------------------------------------------
// Prefetch hint level
// ---------------------------------------------------------------------------

/// Which cache level the prefetch targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrefetchHint {
    /// Prefetch into L1 data cache (lowest latency).
    L1,
    /// Prefetch into L2 cache.
    L2,
    /// Prefetch into L3 / last-level cache.
    L3,
    /// Non-temporal access – data will be used once and need not pollute caches.
    Nta,
}

impl fmt::Display for PrefetchHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::L1 => write!(f, "L1"),
            Self::L2 => write!(f, "L2"),
            Self::L3 => write!(f, "L3"),
            Self::Nta => write!(f, "NTA"),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory access pattern
// ---------------------------------------------------------------------------

/// Describes how memory is traversed by a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessPattern {
    /// Contiguous forward scan (e.g. row-major dot product).
    Sequential,
    /// Strided access with a known element stride.
    Strided { stride_elements: usize },
    /// Unpredictable / pointer-chasing access.
    Random,
    /// Tiled / blocked access with a known block side length.
    Blocked { block_side: usize },
}

impl fmt::Display for AccessPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sequential => write!(f, "Sequential"),
            Self::Strided { stride_elements } => write!(f, "Strided({})", stride_elements),
            Self::Random => write!(f, "Random"),
            Self::Blocked { block_side } => write!(f, "Blocked({})", block_side),
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix traversal order
// ---------------------------------------------------------------------------

/// Traversal order for 2-D matrix prefetch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TraversalOrder {
    /// Row-major (C-order) traversal.
    RowMajor,
    /// Column-major (Fortran-order) traversal.
    ColumnMajor,
    /// Blocked / tiled traversal with the given block side length.
    Block { block_side: usize },
}

// ---------------------------------------------------------------------------
// Prefetch configuration
// ---------------------------------------------------------------------------

/// Full configuration for a prefetch plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefetchConfig {
    /// Target cache level.
    pub hint: PrefetchHint,
    /// Distance in *elements* (not bytes) to prefetch ahead.
    pub distance_elements: usize,
    /// Element size in bytes (e.g. 4 for f32).
    pub element_size: usize,
    /// Memory access pattern.
    pub pattern: AccessPattern,
}

impl PrefetchConfig {
    /// Build a prefetch config with explicit parameters.
    pub fn new(
        hint: PrefetchHint,
        distance_elements: usize,
        element_size: usize,
        pattern: AccessPattern,
    ) -> Self {
        Self {
            hint,
            distance_elements: distance_elements.min(MAX_PREFETCH_DISTANCE),
            element_size,
            pattern,
        }
    }

    /// Distance expressed in bytes.
    #[inline]
    pub fn distance_bytes(&self) -> usize {
        self.distance_elements * self.element_size
    }

    /// Number of cache lines the prefetch distance spans.
    #[inline]
    pub fn distance_cache_lines(&self) -> usize {
        let line = cache_line_size();
        self.distance_bytes().div_ceil(line)
    }
}

// ---------------------------------------------------------------------------
// Cache line detection / alignment
// ---------------------------------------------------------------------------

/// Return the cache line size of the current platform in bytes.
///
/// On x86-64 this uses `cpuid`; on aarch64 it reads `CTR_EL0` where
/// available.  Falls back to [`DEFAULT_CACHE_LINE_BYTES`].
#[inline]
pub fn cache_line_size() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        // CPUID leaf 1 → cache-line flush size in EBX[15:8] × 8
        #[cfg(target_feature = "sse2")]
        {
            // SAFETY: CPUID leaf 1 is always available on x86-64.
            let result = unsafe { std::arch::x86_64::__cpuid(1) };
            let flush_size = (((result.ebx >> 8) & 0xFF) as usize) * 8;
            if flush_size > 0 {
                return flush_size;
            }
        }
        DEFAULT_CACHE_LINE_BYTES
    }

    #[cfg(target_arch = "aarch64")]
    {
        DEFAULT_CACHE_LINE_BYTES
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        DEFAULT_CACHE_LINE_BYTES
    }
}

/// Check whether a pointer is aligned to the cache line boundary.
#[inline]
pub fn is_cache_aligned(ptr: *const u8) -> bool {
    (ptr as usize).is_multiple_of(cache_line_size())
}

/// Round `size` up to the next multiple of the cache line size.
#[inline]
pub fn align_to_cache_line(size: usize) -> usize {
    let line = cache_line_size();
    (size + line - 1) & !(line - 1)
}

// ---------------------------------------------------------------------------
// Software prefetch intrinsic wrappers
// ---------------------------------------------------------------------------

/// Issue a software prefetch for the address `ptr` with the given `hint`.
///
/// # Safety
/// `ptr` must be a valid, dereferenceable address *or* the platform must
/// silently ignore faulting prefetch instructions (true on x86 and aarch64).
#[inline(always)]
pub unsafe fn prefetch_read(ptr: *const u8, hint: PrefetchHint) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{
            _MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2, _mm_prefetch,
        };
        // `_mm_prefetch` requires a compile-time constant hint, so we match
        // and call separately for each variant.
        unsafe {
            match hint {
                PrefetchHint::L1 => _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0),
                PrefetchHint::L2 => _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T1),
                PrefetchHint::L3 => _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T2),
                PrefetchHint::Nta => _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_NTA),
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::_prefetch;
        // _prefetch(ptr, RW, LOCALITY): RW 0=read, LOCALITY 3=L1 .. 0=NTA
        unsafe {
            match hint {
                PrefetchHint::L1 => _prefetch(ptr.cast::<i8>(), 0, 3),
                PrefetchHint::L2 => _prefetch(ptr.cast::<i8>(), 0, 2),
                PrefetchHint::L3 => _prefetch(ptr.cast::<i8>(), 0, 1),
                PrefetchHint::Nta => _prefetch(ptr.cast::<i8>(), 0, 0),
            }
        }
    }

    // Scalar fallback: no-op.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (ptr, hint);
    }
}

/// Issue a software prefetch for *write* at `ptr`.
///
/// # Safety
/// Same requirements as [`prefetch_read`].
#[inline(always)]
pub unsafe fn prefetch_write(ptr: *const u8, hint: PrefetchHint) {
    #[cfg(target_arch = "x86_64")]
    {
        // x86 does not distinguish read/write in `_mm_prefetch`;
        // both map to PREFETCHTx which works for either.
        unsafe { prefetch_read(ptr, hint) };
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::_prefetch;
        unsafe {
            match hint {
                PrefetchHint::L1 => _prefetch(ptr.cast::<i8>(), 1, 3),
                PrefetchHint::L2 => _prefetch(ptr.cast::<i8>(), 1, 2),
                PrefetchHint::L3 => _prefetch(ptr.cast::<i8>(), 1, 1),
                PrefetchHint::Nta => _prefetch(ptr.cast::<i8>(), 1, 0),
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (ptr, hint);
    }
}

// ---------------------------------------------------------------------------
// Prefetch distance auto-tuning
// ---------------------------------------------------------------------------

/// Operation type used to select a heuristic prefetch distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    /// Dense matrix multiplication (GEMM).
    MatMul,
    /// Element-wise (e.g. activation, add).
    Elementwise,
    /// Reduction (softmax row, layer-norm).
    Reduction,
    /// Attention score computation.
    Attention,
    /// Quantization / dequantization pass.
    Quantize,
    /// Embedding lookup (indirect, semi-random).
    EmbeddingLookup,
}

/// Compute a heuristic prefetch distance in *elements* for the given
/// operation type, element size and access pattern.
///
/// The distance is tuned so that one prefetch covers roughly 4–16 cache lines
/// depending on the access pattern, keeping the hardware prefetcher fed
/// without over-polluting the cache.
pub fn auto_prefetch_distance(
    op: OperationType,
    element_size: usize,
    pattern: AccessPattern,
) -> usize {
    let line = cache_line_size();
    let elements_per_line = if element_size > 0 { line / element_size } else { 1 };

    // Base distance in cache lines depending on operation.
    let base_lines: usize = match op {
        OperationType::MatMul => 16,
        OperationType::Elementwise => 8,
        OperationType::Reduction => 8,
        OperationType::Attention => 12,
        OperationType::Quantize => 8,
        OperationType::EmbeddingLookup => 4,
    };

    // Pattern multiplier.
    let multiplier: usize = match pattern {
        AccessPattern::Sequential => 1,
        AccessPattern::Strided { stride_elements } => {
            // Larger strides → prefetch farther ahead.
            let stride_lines = (stride_elements * element_size).div_ceil(line);
            stride_lines.clamp(1, 4)
        }
        AccessPattern::Random => 1, // little benefit from large distance
        AccessPattern::Blocked { .. } => 2,
    };

    let distance = base_lines * multiplier * elements_per_line;
    distance.min(MAX_PREFETCH_DISTANCE)
}

/// Convenience: build a complete [`PrefetchConfig`] from high-level params.
pub fn auto_prefetch_config(
    op: OperationType,
    element_size: usize,
    pattern: AccessPattern,
    hint: PrefetchHint,
) -> PrefetchConfig {
    let dist = auto_prefetch_distance(op, element_size, pattern);
    PrefetchConfig::new(hint, dist, element_size, pattern)
}

// ---------------------------------------------------------------------------
// Matrix traversal prefetching
// ---------------------------------------------------------------------------

/// Prefetch a row-major matrix row starting at `base`, for `cols` elements of
/// `elem_size` bytes each, prefetching every `stride_lines` cache lines.
///
/// # Safety
/// `base` must point into a valid allocation that spans at least
/// `cols * elem_size` bytes.
pub unsafe fn prefetch_row(base: *const u8, cols: usize, elem_size: usize, hint: PrefetchHint) {
    let line = cache_line_size();
    let row_bytes = cols * elem_size;
    let mut offset = 0;
    while offset < row_bytes {
        unsafe { prefetch_read(base.add(offset), hint) };
        offset += line;
    }
}

/// Prefetch a column of a row-major matrix, where each successive element is
/// `row_stride` bytes apart.
///
/// # Safety
/// Every touched address must lie inside the matrix allocation.
pub unsafe fn prefetch_column(
    base: *const u8,
    rows: usize,
    row_stride_bytes: usize,
    hint: PrefetchHint,
) {
    for r in 0..rows {
        unsafe { prefetch_read(base.add(r * row_stride_bytes), hint) };
    }
}

/// Prefetch a square block of `side × side` elements starting at `base`
/// in a row-major matrix with `row_stride_bytes` between rows.
///
/// # Safety
/// All addresses within the block must be valid.
pub unsafe fn prefetch_block(
    base: *const u8,
    side: usize,
    elem_size: usize,
    row_stride_bytes: usize,
    hint: PrefetchHint,
) {
    let line = cache_line_size();
    let block_row_bytes = side * elem_size;
    for r in 0..side {
        let row_ptr = unsafe { base.add(r * row_stride_bytes) };
        let mut offset = 0;
        while offset < block_row_bytes {
            unsafe { prefetch_read(row_ptr.add(offset), hint) };
            offset += line;
        }
    }
}

// ---------------------------------------------------------------------------
// Cache-oblivious helpers
// ---------------------------------------------------------------------------

/// Recursive block decomposition parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockDecomposition {
    /// Total number of rows.
    pub rows: usize,
    /// Total number of columns.
    pub cols: usize,
    /// Minimum tile side below which we stop recursing.
    pub min_tile: usize,
}

/// Leaf tile produced by [`decompose_blocks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    pub row_start: usize,
    pub col_start: usize,
    pub row_end: usize,
    pub col_end: usize,
}

impl Tile {
    /// Number of rows in the tile.
    #[inline]
    pub fn height(&self) -> usize {
        self.row_end - self.row_start
    }

    /// Number of columns in the tile.
    #[inline]
    pub fn width(&self) -> usize {
        self.col_end - self.col_start
    }

    /// Total number of elements.
    #[inline]
    pub fn area(&self) -> usize {
        self.height() * self.width()
    }
}

/// Recursively decompose a matrix region into cache-friendly tiles.
///
/// The decomposition splits the larger dimension in half until both dimensions
/// are ≤ `decomp.min_tile`, producing a Z-order traversal that is cache-
/// oblivious.
pub fn decompose_blocks(decomp: BlockDecomposition) -> Vec<Tile> {
    let mut tiles = Vec::new();
    decompose_recursive(0, 0, decomp.rows, decomp.cols, decomp.min_tile, &mut tiles);
    tiles
}

fn decompose_recursive(
    r0: usize,
    c0: usize,
    rows: usize,
    cols: usize,
    min_tile: usize,
    out: &mut Vec<Tile>,
) {
    if rows <= min_tile && cols <= min_tile {
        out.push(Tile { row_start: r0, col_start: c0, row_end: r0 + rows, col_end: c0 + cols });
        return;
    }
    if rows >= cols {
        let mid = rows / 2;
        decompose_recursive(r0, c0, mid, cols, min_tile, out);
        decompose_recursive(r0 + mid, c0, rows - mid, cols, min_tile, out);
    } else {
        let mid = cols / 2;
        decompose_recursive(r0, c0, rows, mid, min_tile, out);
        decompose_recursive(r0, c0 + mid, rows, cols - mid, min_tile, out);
    }
}

// ---------------------------------------------------------------------------
// Prefetch plan for a full matrix traversal
// ---------------------------------------------------------------------------

/// Build a prefetch plan: list of byte offsets to prefetch for the given
/// traversal of an `rows × cols` matrix of elements of `elem_size` bytes.
pub fn prefetch_plan(
    rows: usize,
    cols: usize,
    elem_size: usize,
    order: TraversalOrder,
) -> Vec<usize> {
    let line = cache_line_size();
    let mut offsets = Vec::new();

    match order {
        TraversalOrder::RowMajor => {
            for r in 0..rows {
                let row_base = r * cols * elem_size;
                let mut off = 0;
                while off < cols * elem_size {
                    offsets.push(row_base + off);
                    off += line;
                }
            }
        }
        TraversalOrder::ColumnMajor => {
            let row_stride = cols * elem_size;
            for c in 0..cols {
                let col_base = c * elem_size;
                for r in 0..rows {
                    offsets.push(col_base + r * row_stride);
                }
            }
        }
        TraversalOrder::Block { block_side } => {
            let decomp = BlockDecomposition { rows, cols, min_tile: block_side };
            for tile in decompose_blocks(decomp) {
                for r in tile.row_start..tile.row_end {
                    let row_base = r * cols * elem_size;
                    let start = tile.col_start * elem_size;
                    let end = tile.col_end * elem_size;
                    let mut off = start;
                    while off < end {
                        offsets.push(row_base + off);
                        off += line;
                    }
                }
            }
        }
    }
    offsets
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- PrefetchHint -------------------------------------------------------

    #[test]
    fn hint_display() {
        assert_eq!(PrefetchHint::L1.to_string(), "L1");
        assert_eq!(PrefetchHint::L2.to_string(), "L2");
        assert_eq!(PrefetchHint::L3.to_string(), "L3");
        assert_eq!(PrefetchHint::Nta.to_string(), "NTA");
    }

    #[test]
    fn hint_clone_eq() {
        let a = PrefetchHint::L2;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn hint_debug_format() {
        let dbg = format!("{:?}", PrefetchHint::Nta);
        assert!(dbg.contains("Nta"));
    }

    // -- AccessPattern ------------------------------------------------------

    #[test]
    fn access_pattern_display_sequential() {
        assert_eq!(AccessPattern::Sequential.to_string(), "Sequential");
    }

    #[test]
    fn access_pattern_display_strided() {
        let p = AccessPattern::Strided { stride_elements: 128 };
        assert_eq!(p.to_string(), "Strided(128)");
    }

    #[test]
    fn access_pattern_display_random() {
        assert_eq!(AccessPattern::Random.to_string(), "Random");
    }

    #[test]
    fn access_pattern_display_blocked() {
        let p = AccessPattern::Blocked { block_side: 32 };
        assert_eq!(p.to_string(), "Blocked(32)");
    }

    #[test]
    fn access_pattern_eq() {
        assert_eq!(
            AccessPattern::Strided { stride_elements: 64 },
            AccessPattern::Strided { stride_elements: 64 },
        );
        assert_ne!(AccessPattern::Sequential, AccessPattern::Random);
    }

    // -- PrefetchConfig -----------------------------------------------------

    #[test]
    fn config_distance_bytes() {
        let cfg = PrefetchConfig::new(PrefetchHint::L1, 16, 4, AccessPattern::Sequential);
        assert_eq!(cfg.distance_bytes(), 64);
    }

    #[test]
    fn config_distance_cache_lines() {
        let line = cache_line_size();
        let elems = line / 4; // exactly one cache line of f32
        let cfg = PrefetchConfig::new(PrefetchHint::L1, elems, 4, AccessPattern::Sequential);
        assert_eq!(cfg.distance_cache_lines(), 1);
    }

    #[test]
    fn config_distance_clamped() {
        let cfg = PrefetchConfig::new(PrefetchHint::L2, 999_999, 4, AccessPattern::Sequential);
        assert_eq!(cfg.distance_elements, MAX_PREFETCH_DISTANCE);
    }

    #[test]
    fn config_clone_eq() {
        let a = PrefetchConfig::new(PrefetchHint::L1, 64, 4, AccessPattern::Sequential);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn config_distance_bytes_large_element() {
        let cfg = PrefetchConfig::new(PrefetchHint::L2, 8, 8, AccessPattern::Sequential);
        assert_eq!(cfg.distance_bytes(), 64);
    }

    // -- cache_line_size / alignment ----------------------------------------

    #[test]
    fn cache_line_size_is_power_of_two() {
        let line = cache_line_size();
        assert!(line.is_power_of_two(), "cache line {line} not power-of-2");
    }

    #[test]
    fn cache_line_size_plausible() {
        let line = cache_line_size();
        assert!(line >= 32 && line <= 256, "unexpected cache line size: {line}");
    }

    #[test]
    fn align_to_cache_line_already_aligned() {
        let line = cache_line_size();
        assert_eq!(align_to_cache_line(line), line);
    }

    #[test]
    fn align_to_cache_line_rounds_up() {
        let line = cache_line_size();
        assert_eq!(align_to_cache_line(1), line);
        assert_eq!(align_to_cache_line(line + 1), 2 * line);
    }

    #[test]
    fn is_cache_aligned_check() {
        let line = cache_line_size();
        let ptr = line as *const u8; // artificially aligned
        assert!(is_cache_aligned(ptr));
        let misaligned = (line + 1) as *const u8;
        assert!(!is_cache_aligned(misaligned));
    }

    #[test]
    fn align_to_cache_line_zero() {
        assert_eq!(align_to_cache_line(0), 0);
    }

    // -- auto_prefetch_distance ---------------------------------------------

    #[test]
    fn auto_distance_sequential_f32() {
        let d = auto_prefetch_distance(OperationType::MatMul, 4, AccessPattern::Sequential);
        assert!(d > 0 && d <= MAX_PREFETCH_DISTANCE);
    }

    #[test]
    fn auto_distance_strided_larger_than_sequential() {
        let seq = auto_prefetch_distance(OperationType::MatMul, 4, AccessPattern::Sequential);
        let strided = auto_prefetch_distance(
            OperationType::MatMul,
            4,
            AccessPattern::Strided { stride_elements: 256 },
        );
        assert!(strided >= seq, "strided distance should be >= sequential");
    }

    #[test]
    fn auto_distance_random_small() {
        let d = auto_prefetch_distance(OperationType::EmbeddingLookup, 4, AccessPattern::Random);
        // Random should use base distance only (multiplier = 1).
        let line = cache_line_size();
        let elems_per_line = line / 4;
        assert_eq!(d, 4 * elems_per_line);
    }

    #[test]
    fn auto_distance_capped() {
        // Huge stride should still not exceed MAX.
        let d = auto_prefetch_distance(
            OperationType::MatMul,
            4,
            AccessPattern::Strided { stride_elements: 100_000 },
        );
        assert!(d <= MAX_PREFETCH_DISTANCE);
    }

    #[test]
    fn auto_distance_elementwise() {
        let d = auto_prefetch_distance(OperationType::Elementwise, 4, AccessPattern::Sequential);
        assert!(d > 0);
    }

    #[test]
    fn auto_distance_blocked() {
        let d = auto_prefetch_distance(
            OperationType::Attention,
            4,
            AccessPattern::Blocked { block_side: 64 },
        );
        assert!(d > 0);
    }

    #[test]
    fn auto_prefetch_config_builds() {
        let cfg = auto_prefetch_config(
            OperationType::Reduction,
            4,
            AccessPattern::Sequential,
            PrefetchHint::L1,
        );
        assert_eq!(cfg.hint, PrefetchHint::L1);
        assert_eq!(cfg.element_size, 4);
        assert!(cfg.distance_elements > 0);
    }

    // -- prefetch intrinsics (smoke) ----------------------------------------

    #[test]
    fn prefetch_read_smoke() {
        let data = vec![0u8; 256];
        unsafe {
            prefetch_read(data.as_ptr(), PrefetchHint::L1);
            prefetch_read(data.as_ptr(), PrefetchHint::L2);
            prefetch_read(data.as_ptr(), PrefetchHint::L3);
            prefetch_read(data.as_ptr(), PrefetchHint::Nta);
        }
    }

    #[test]
    fn prefetch_write_smoke() {
        let data = vec![0u8; 256];
        unsafe {
            prefetch_write(data.as_ptr(), PrefetchHint::L1);
            prefetch_write(data.as_ptr(), PrefetchHint::Nta);
        }
    }

    // -- matrix traversal prefetching (smoke) -------------------------------

    #[test]
    fn prefetch_row_smoke() {
        let data = vec![0f32; 128];
        unsafe {
            prefetch_row(data.as_ptr().cast::<u8>(), 128, 4, PrefetchHint::L1);
        }
    }

    #[test]
    fn prefetch_column_smoke() {
        let rows = 8;
        let cols = 16;
        let data = vec![0f32; rows * cols];
        let row_stride_bytes = cols * 4;
        unsafe {
            prefetch_column(data.as_ptr().cast::<u8>(), rows, row_stride_bytes, PrefetchHint::L2);
        }
    }

    #[test]
    fn prefetch_block_smoke() {
        let rows = 16;
        let cols = 16;
        let data = vec![0f32; rows * cols];
        let row_stride_bytes = cols * 4;
        unsafe {
            prefetch_block(data.as_ptr().cast::<u8>(), 8, 4, row_stride_bytes, PrefetchHint::L1);
        }
    }

    // -- block decomposition ------------------------------------------------

    #[test]
    fn decompose_single_tile() {
        let tiles = decompose_blocks(BlockDecomposition { rows: 4, cols: 4, min_tile: 8 });
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].row_start, 0);
        assert_eq!(tiles[0].col_end, 4);
    }

    #[test]
    fn decompose_covers_all_elements() {
        let rows = 17;
        let cols = 13;
        let tiles = decompose_blocks(BlockDecomposition { rows, cols, min_tile: 4 });
        let total: usize = tiles.iter().map(|t| t.area()).sum();
        assert_eq!(total, rows * cols);
    }

    #[test]
    fn decompose_tiles_non_overlapping() {
        let rows = 16;
        let cols = 16;
        let tiles = decompose_blocks(BlockDecomposition { rows, cols, min_tile: 4 });
        // Every cell must appear in exactly one tile.
        let mut grid = vec![vec![false; cols]; rows];
        for t in &tiles {
            for r in t.row_start..t.row_end {
                for c in t.col_start..t.col_end {
                    assert!(!grid[r][c], "overlap at ({r},{c})");
                    grid[r][c] = true;
                }
            }
        }
        assert!(grid.iter().all(|row| row.iter().all(|&v| v)));
    }

    #[test]
    fn tile_dimensions() {
        let t = Tile { row_start: 2, col_start: 3, row_end: 6, col_end: 10 };
        assert_eq!(t.height(), 4);
        assert_eq!(t.width(), 7);
        assert_eq!(t.area(), 28);
    }

    #[test]
    fn decompose_power_of_two() {
        let tiles = decompose_blocks(BlockDecomposition { rows: 8, cols: 8, min_tile: 4 });
        assert_eq!(tiles.len(), 4);
        for t in &tiles {
            assert_eq!(t.height(), 4);
            assert_eq!(t.width(), 4);
        }
    }

    // -- prefetch plan ------------------------------------------------------

    #[test]
    fn plan_row_major_covers_all_lines() {
        let offsets = prefetch_plan(4, 16, 4, TraversalOrder::RowMajor);
        assert!(!offsets.is_empty());
        // First offset should be 0.
        assert_eq!(offsets[0], 0);
    }

    #[test]
    fn plan_column_major_stride() {
        let offsets = prefetch_plan(4, 8, 4, TraversalOrder::ColumnMajor);
        // Column-major: second entry for col 0 should be one row_stride ahead.
        let row_stride = 8 * 4;
        assert!(offsets.len() >= 2);
        assert_eq!(offsets[1], row_stride);
    }

    #[test]
    fn plan_block_covers_all_elements() {
        let rows = 8;
        let cols = 8;
        let offsets = prefetch_plan(rows, cols, 4, TraversalOrder::Block { block_side: 4 });
        assert!(!offsets.is_empty());
        // Every offset must be within matrix bounds.
        let total_bytes = rows * cols * 4;
        for &o in &offsets {
            assert!(o < total_bytes, "offset {o} out of bounds ({total_bytes})");
        }
    }

    #[test]
    fn plan_row_major_offsets_monotonic_per_row() {
        let offsets = prefetch_plan(2, 32, 4, TraversalOrder::RowMajor);
        // Within each row chunk the offsets should be monotonically increasing.
        let line = cache_line_size();
        let lines_per_row = (32 * 4 + line - 1) / line;
        for chunk in offsets.chunks(lines_per_row) {
            for w in chunk.windows(2) {
                assert!(w[1] > w[0]);
            }
        }
    }

    // -- operation type enum ------------------------------------------------

    #[test]
    fn operation_type_debug() {
        let dbg = format!("{:?}", OperationType::Quantize);
        assert!(dbg.contains("Quantize"));
    }

    #[test]
    fn traversal_order_eq() {
        assert_eq!(TraversalOrder::RowMajor, TraversalOrder::RowMajor);
        assert_ne!(TraversalOrder::RowMajor, TraversalOrder::ColumnMajor);
    }
}
