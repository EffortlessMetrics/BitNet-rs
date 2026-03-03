//! CUDA shared memory management for kernel launches.
//!
//! Provides layout computation, bank-conflict avoidance, double-buffering
//! offsets, and tile-size selection for GEMM kernels.

use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of shared-memory banks on modern NVIDIA GPUs (Kepler+).
pub const NUM_BANKS: usize = 32;

/// Default alignment in bytes for shared memory allocations.
pub const DEFAULT_ALIGNMENT: usize = 128;

/// Maximum shared memory per block on most architectures (48 KiB).
pub const DEFAULT_SHARED_MEM_LIMIT: usize = 48 * 1024;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during shared memory layout computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SharedMemError {
    /// The requested allocation could not be satisfied.
    AllocationFailed(String),
    /// The computed layout exceeds the device limit.
    ExceedsLimit { required: usize, limit: usize },
    /// An alignment constraint was violated.
    AlignmentError { requested: usize, actual: usize },
    /// A bank-conflict–free layout could not be produced.
    BankConflict { index: usize, bank: usize },
}

impl fmt::Display for SharedMemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllocationFailed(msg) => write!(f, "allocation failed: {msg}"),
            Self::ExceedsLimit { required, limit } => {
                write!(f, "shared memory {required} B exceeds device limit {limit} B")
            }
            Self::AlignmentError { requested, actual } => {
                write!(f, "alignment error: requested {requested}, got {actual}")
            }
            Self::BankConflict { index, bank } => {
                write!(f, "bank conflict at index {index}, bank {bank}")
            }
        }
    }
}

impl std::error::Error for SharedMemError {}

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Describes a single shared-memory region request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMemConfig {
    /// Size in bytes of the region.
    pub size: usize,
    /// Whether to add padding to avoid bank conflicts.
    pub bank_conflict_free: bool,
    /// Extra padding bytes to append after the region.
    pub padding: usize,
}

/// A computed shared-memory layout for one or more regions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMemLayout {
    /// Byte offset of each region inside shared memory.
    pub offsets: Vec<usize>,
    /// Total bytes required (including all padding/alignment).
    pub total_size: usize,
    /// Alignment guarantee of the layout.
    pub alignment: usize,
}

/// Tile dimensions for a GEMM kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    /// Whether double-buffering is enabled.
    pub double_buffer: bool,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Round `value` up to the next multiple of `align`.
///
/// `align` **must** be a power of two; the function does not validate this
/// because it is only called from contexts that guarantee it.
#[inline]
fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two(), "alignment must be power of two");
    (value + align - 1) & !(align - 1)
}

/// Compute a packed shared-memory layout for the given region configs.
///
/// Each region is aligned to [`DEFAULT_ALIGNMENT`]. When
/// `bank_conflict_free` is set on a config, an extra padding row of
/// [`NUM_BANKS`] bytes is added per 128-byte row so that successive
/// column accesses map to different banks.
pub fn compute_shared_mem_layout(
    configs: &[SharedMemConfig],
) -> Result<SharedMemLayout, SharedMemError> {
    if configs.is_empty() {
        return Ok(SharedMemLayout {
            offsets: vec![],
            total_size: 0,
            alignment: DEFAULT_ALIGNMENT,
        });
    }

    let mut offsets = Vec::with_capacity(configs.len());
    let mut cursor: usize = 0;

    for (i, cfg) in configs.iter().enumerate() {
        // Align cursor to DEFAULT_ALIGNMENT boundary.
        cursor = align_up(cursor, DEFAULT_ALIGNMENT);
        offsets.push(cursor);

        let padded_size = if cfg.bank_conflict_free {
            // For every 128-byte row, add one bank-width of padding so that
            // column-major accesses hit different banks.
            let row_bytes = NUM_BANKS * std::mem::size_of::<f32>(); // 128 B
            if row_bytes == 0 {
                return Err(SharedMemError::AllocationFailed("zero row width".to_string()));
            }
            let num_rows = cfg.size.div_ceil(row_bytes);
            let padding_per_row = std::mem::size_of::<f32>(); // 4 B
            cfg.size + num_rows * padding_per_row
        } else {
            cfg.size
        };

        let total_region = padded_size
            .checked_add(cfg.padding)
            .ok_or_else(|| SharedMemError::AllocationFailed(format!("overflow in region {i}")))?;

        cursor = cursor
            .checked_add(total_region)
            .ok_or_else(|| SharedMemError::AllocationFailed(format!("overflow in region {i}")))?;
    }

    // Final alignment.
    cursor = align_up(cursor, DEFAULT_ALIGNMENT);

    Ok(SharedMemLayout { offsets, total_size: cursor, alignment: DEFAULT_ALIGNMENT })
}

/// Choose an optimal tile size for a GEMM of dimensions (m × k) × (k × n).
///
/// The tile is chosen so that the shared-memory footprint of the A and B
/// tiles (with optional double-buffering) fits within `shared_mem_limit`.
pub fn optimal_tile_size(m: usize, n: usize, k: usize, shared_mem_limit: usize) -> TileConfig {
    // Candidate tile dimensions (prefer larger tiles).
    const CANDIDATES: &[usize] = &[128, 64, 32, 16, 8];
    let elem = std::mem::size_of::<f32>();

    for &tm in CANDIDATES {
        for &tn in CANDIDATES {
            for &tk in CANDIDATES {
                // Shared mem: tile_A (tm×tk) + tile_B (tk×tn).
                let single = (tm * tk + tk * tn) * elem;

                // Try double-buffered first.
                if single * 2 <= shared_mem_limit {
                    return TileConfig {
                        tile_m: tm.min(m).max(1),
                        tile_n: tn.min(n).max(1),
                        tile_k: tk.min(k).max(1),
                        double_buffer: true,
                    };
                }
                // Fall back to single buffer.
                if single <= shared_mem_limit {
                    return TileConfig {
                        tile_m: tm.min(m).max(1),
                        tile_n: tn.min(n).max(1),
                        tile_k: tk.min(k).max(1),
                        double_buffer: false,
                    };
                }
            }
        }
    }

    // Absolute minimum tile.
    TileConfig { tile_m: 1, tile_n: 1, tile_k: 1, double_buffer: false }
}

/// Remap a linear index so that successive threads access different banks.
///
/// The classic trick: `new_idx = idx + idx / num_banks`.
#[inline]
pub fn bank_conflict_free_index(idx: usize, num_banks: usize) -> usize {
    if num_banks == 0 {
        return idx;
    }
    idx + idx / num_banks
}

/// Compute the byte offset into a double-buffer given the buffer slot
/// (0 or 1), the element size, and the element count per buffer.
#[inline]
pub fn double_buffer_offset(buffer_idx: usize, element_size: usize, count: usize) -> usize {
    buffer_idx * element_size * count
}

/// Validate that a layout fits within the device's shared-memory limit.
pub fn validate_shared_mem_usage(
    layout: &SharedMemLayout,
    device_limit: usize,
) -> Result<(), SharedMemError> {
    if layout.total_size > device_limit {
        return Err(SharedMemError::ExceedsLimit {
            required: layout.total_size,
            limit: device_limit,
        });
    }
    // Verify alignment invariant.
    if layout.alignment == 0 || !layout.alignment.is_power_of_two() {
        return Err(SharedMemError::AlignmentError {
            requested: DEFAULT_ALIGNMENT,
            actual: layout.alignment,
        });
    }
    for &off in &layout.offsets {
        if off % layout.alignment != 0 {
            return Err(SharedMemError::AlignmentError {
                requested: layout.alignment,
                actual: off,
            });
        }
    }
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // compute_shared_mem_layout
    // -----------------------------------------------------------------------

    #[test]
    fn layout_empty_configs() {
        let layout = compute_shared_mem_layout(&[]).unwrap();
        assert_eq!(layout.offsets, Vec::<usize>::new());
        assert_eq!(layout.total_size, 0);
    }

    #[test]
    fn layout_single_region_no_padding() {
        let cfg = SharedMemConfig { size: 256, bank_conflict_free: false, padding: 0 };
        let layout = compute_shared_mem_layout(&[cfg]).unwrap();
        assert_eq!(layout.offsets, vec![0]);
        assert!(layout.total_size >= 256);
        assert_eq!(layout.total_size % DEFAULT_ALIGNMENT, 0);
    }

    #[test]
    fn layout_single_region_with_padding() {
        let cfg = SharedMemConfig { size: 100, bank_conflict_free: false, padding: 28 };
        let layout = compute_shared_mem_layout(&[cfg]).unwrap();
        assert!(layout.total_size >= 128);
    }

    #[test]
    fn layout_bank_conflict_free_increases_size() {
        let plain = SharedMemConfig { size: 512, bank_conflict_free: false, padding: 0 };
        let bcf = SharedMemConfig { size: 512, bank_conflict_free: true, padding: 0 };
        let l1 = compute_shared_mem_layout(&[plain]).unwrap();
        let l2 = compute_shared_mem_layout(&[bcf]).unwrap();
        assert!(l2.total_size >= l1.total_size);
    }

    #[test]
    fn layout_multiple_regions_are_aligned() {
        let cfgs = vec![
            SharedMemConfig { size: 300, bank_conflict_free: false, padding: 0 },
            SharedMemConfig { size: 500, bank_conflict_free: false, padding: 0 },
            SharedMemConfig { size: 100, bank_conflict_free: false, padding: 0 },
        ];
        let layout = compute_shared_mem_layout(&cfgs).unwrap();
        assert_eq!(layout.offsets.len(), 3);
        for &off in &layout.offsets {
            assert_eq!(off % DEFAULT_ALIGNMENT, 0);
        }
    }

    #[test]
    fn layout_offsets_are_non_decreasing() {
        let cfgs = vec![
            SharedMemConfig { size: 64, bank_conflict_free: false, padding: 0 },
            SharedMemConfig { size: 64, bank_conflict_free: false, padding: 0 },
        ];
        let layout = compute_shared_mem_layout(&cfgs).unwrap();
        assert!(layout.offsets[1] > layout.offsets[0]);
    }

    #[test]
    fn layout_total_size_aligned() {
        let cfg = SharedMemConfig { size: 1, bank_conflict_free: false, padding: 0 };
        let layout = compute_shared_mem_layout(&[cfg]).unwrap();
        assert_eq!(layout.total_size % DEFAULT_ALIGNMENT, 0);
    }

    #[test]
    fn layout_zero_size_region() {
        let cfg = SharedMemConfig { size: 0, bank_conflict_free: false, padding: 0 };
        let layout = compute_shared_mem_layout(&[cfg]).unwrap();
        assert_eq!(layout.total_size, 0);
    }

    #[test]
    fn layout_zero_size_bank_conflict_free() {
        let cfg = SharedMemConfig { size: 0, bank_conflict_free: true, padding: 0 };
        let layout = compute_shared_mem_layout(&[cfg]).unwrap();
        assert_eq!(layout.offsets.len(), 1);
    }

    #[test]
    fn layout_large_region() {
        let cfg = SharedMemConfig { size: 48 * 1024, bank_conflict_free: false, padding: 0 };
        let layout = compute_shared_mem_layout(&[cfg]).unwrap();
        assert!(layout.total_size >= 48 * 1024);
    }

    #[test]
    fn layout_alignment_value() {
        let cfg = SharedMemConfig { size: 256, bank_conflict_free: false, padding: 0 };
        let layout = compute_shared_mem_layout(&[cfg]).unwrap();
        assert_eq!(layout.alignment, DEFAULT_ALIGNMENT);
    }

    #[test]
    fn layout_padding_only() {
        let cfg = SharedMemConfig { size: 0, bank_conflict_free: false, padding: 64 };
        let layout = compute_shared_mem_layout(&[cfg]).unwrap();
        assert!(layout.total_size >= 64);
    }

    #[test]
    fn layout_bank_conflict_free_small_size() {
        let cfg = SharedMemConfig { size: 4, bank_conflict_free: true, padding: 0 };
        let layout = compute_shared_mem_layout(&[cfg]).unwrap();
        assert!(layout.total_size > 0);
    }

    // -----------------------------------------------------------------------
    // optimal_tile_size
    // -----------------------------------------------------------------------

    #[test]
    fn tile_basic() {
        let tc = optimal_tile_size(256, 256, 256, DEFAULT_SHARED_MEM_LIMIT);
        assert!(tc.tile_m > 0 && tc.tile_n > 0 && tc.tile_k > 0);
    }

    #[test]
    fn tile_fits_in_shared_mem() {
        let tc = optimal_tile_size(1024, 1024, 1024, DEFAULT_SHARED_MEM_LIMIT);
        let elem = std::mem::size_of::<f32>();
        let footprint = (tc.tile_m * tc.tile_k + tc.tile_k * tc.tile_n) * elem;
        let factor = if tc.double_buffer { 2 } else { 1 };
        assert!(footprint * factor <= DEFAULT_SHARED_MEM_LIMIT);
    }

    #[test]
    fn tile_small_matrix() {
        let tc = optimal_tile_size(4, 4, 4, DEFAULT_SHARED_MEM_LIMIT);
        assert!(tc.tile_m <= 4);
        assert!(tc.tile_n <= 4);
        assert!(tc.tile_k <= 4);
    }

    #[test]
    fn tile_very_small_limit() {
        let tc = optimal_tile_size(256, 256, 256, 32);
        // Must still produce valid tiles.
        assert!(tc.tile_m >= 1);
        assert!(tc.tile_n >= 1);
        assert!(tc.tile_k >= 1);
    }

    #[test]
    fn tile_zero_limit() {
        let tc = optimal_tile_size(128, 128, 128, 0);
        assert_eq!(tc.tile_m, 1);
        assert_eq!(tc.tile_n, 1);
        assert_eq!(tc.tile_k, 1);
        assert!(!tc.double_buffer);
    }

    #[test]
    fn tile_prefers_double_buffer_when_room() {
        let tc = optimal_tile_size(128, 128, 128, 1024 * 1024);
        assert!(tc.double_buffer);
    }

    #[test]
    fn tile_m_clamped_to_matrix() {
        let tc = optimal_tile_size(10, 256, 256, DEFAULT_SHARED_MEM_LIMIT);
        assert!(tc.tile_m <= 10);
    }

    #[test]
    fn tile_n_clamped_to_matrix() {
        let tc = optimal_tile_size(256, 5, 256, DEFAULT_SHARED_MEM_LIMIT);
        assert!(tc.tile_n <= 5);
    }

    #[test]
    fn tile_k_clamped_to_matrix() {
        let tc = optimal_tile_size(256, 256, 3, DEFAULT_SHARED_MEM_LIMIT);
        assert!(tc.tile_k <= 3);
    }

    #[test]
    fn tile_single_element_matrix() {
        let tc = optimal_tile_size(1, 1, 1, DEFAULT_SHARED_MEM_LIMIT);
        assert_eq!(tc.tile_m, 1);
        assert_eq!(tc.tile_n, 1);
        assert_eq!(tc.tile_k, 1);
    }

    #[test]
    fn tile_large_shared_mem() {
        let tc = optimal_tile_size(512, 512, 512, 256 * 1024);
        assert!(tc.tile_m >= 64);
    }

    #[test]
    fn tile_asymmetric_matrix() {
        let tc = optimal_tile_size(1024, 8, 1024, DEFAULT_SHARED_MEM_LIMIT);
        assert!(tc.tile_n <= 8);
    }

    // -----------------------------------------------------------------------
    // bank_conflict_free_index
    // -----------------------------------------------------------------------

    #[test]
    fn bcf_index_zero() {
        assert_eq!(bank_conflict_free_index(0, NUM_BANKS), 0);
    }

    #[test]
    fn bcf_index_less_than_banks() {
        assert_eq!(bank_conflict_free_index(15, NUM_BANKS), 15);
    }

    #[test]
    fn bcf_index_equal_to_banks() {
        assert_eq!(bank_conflict_free_index(32, NUM_BANKS), 33);
    }

    #[test]
    fn bcf_index_multiple_of_banks() {
        assert_eq!(bank_conflict_free_index(64, NUM_BANKS), 66);
    }

    #[test]
    fn bcf_index_various() {
        // 33 → 33 + 33/32 = 33 + 1 = 34
        assert_eq!(bank_conflict_free_index(33, NUM_BANKS), 34);
    }

    #[test]
    fn bcf_index_zero_banks_passthrough() {
        assert_eq!(bank_conflict_free_index(42, 0), 42);
    }

    #[test]
    fn bcf_index_one_bank() {
        // Every index maps: idx + idx/1 = 2*idx
        assert_eq!(bank_conflict_free_index(5, 1), 10);
    }

    #[test]
    fn bcf_index_large() {
        let idx: usize = 1_000_000;
        let result = bank_conflict_free_index(idx, NUM_BANKS);
        assert_eq!(result, idx + idx / NUM_BANKS);
    }

    #[test]
    fn bcf_consecutive_threads_different_banks() {
        // Threads 0..32 should all hit different banks after remapping.
        let mut banks = std::collections::HashSet::new();
        for t in 0..NUM_BANKS {
            let remapped = bank_conflict_free_index(t, NUM_BANKS);
            banks.insert(remapped % NUM_BANKS);
        }
        // All 32 threads map to distinct banks.
        assert_eq!(banks.len(), NUM_BANKS);
    }

    // -----------------------------------------------------------------------
    // double_buffer_offset
    // -----------------------------------------------------------------------

    #[test]
    fn db_offset_buffer_zero() {
        assert_eq!(double_buffer_offset(0, 4, 256), 0);
    }

    #[test]
    fn db_offset_buffer_one() {
        assert_eq!(double_buffer_offset(1, 4, 256), 1024);
    }

    #[test]
    fn db_offset_zero_count() {
        assert_eq!(double_buffer_offset(1, 4, 0), 0);
    }

    #[test]
    fn db_offset_zero_element_size() {
        assert_eq!(double_buffer_offset(1, 0, 256), 0);
    }

    #[test]
    fn db_offset_large_buffer_idx() {
        assert_eq!(double_buffer_offset(3, 4, 100), 1200);
    }

    #[test]
    fn db_offset_element_size_8() {
        assert_eq!(double_buffer_offset(1, 8, 128), 1024);
    }

    // -----------------------------------------------------------------------
    // validate_shared_mem_usage
    // -----------------------------------------------------------------------

    #[test]
    fn validate_ok() {
        let layout =
            SharedMemLayout { offsets: vec![0], total_size: 1024, alignment: DEFAULT_ALIGNMENT };
        assert!(validate_shared_mem_usage(&layout, DEFAULT_SHARED_MEM_LIMIT).is_ok());
    }

    #[test]
    fn validate_exceeds_limit() {
        let layout = SharedMemLayout {
            offsets: vec![0],
            total_size: DEFAULT_SHARED_MEM_LIMIT + 1,
            alignment: DEFAULT_ALIGNMENT,
        };
        let err = validate_shared_mem_usage(&layout, DEFAULT_SHARED_MEM_LIMIT).unwrap_err();
        assert!(matches!(err, SharedMemError::ExceedsLimit { .. }));
    }

    #[test]
    fn validate_exact_limit() {
        let layout = SharedMemLayout {
            offsets: vec![0],
            total_size: DEFAULT_SHARED_MEM_LIMIT,
            alignment: DEFAULT_ALIGNMENT,
        };
        assert!(validate_shared_mem_usage(&layout, DEFAULT_SHARED_MEM_LIMIT).is_ok());
    }

    #[test]
    fn validate_bad_alignment_zero() {
        let layout = SharedMemLayout { offsets: vec![0], total_size: 256, alignment: 0 };
        let err = validate_shared_mem_usage(&layout, DEFAULT_SHARED_MEM_LIMIT).unwrap_err();
        assert!(matches!(err, SharedMemError::AlignmentError { .. }));
    }

    #[test]
    fn validate_bad_alignment_non_power_of_two() {
        let layout = SharedMemLayout { offsets: vec![0], total_size: 256, alignment: 3 };
        let err = validate_shared_mem_usage(&layout, DEFAULT_SHARED_MEM_LIMIT).unwrap_err();
        assert!(matches!(err, SharedMemError::AlignmentError { .. }));
    }

    #[test]
    fn validate_misaligned_offset() {
        let layout =
            SharedMemLayout { offsets: vec![0, 13], total_size: 256, alignment: DEFAULT_ALIGNMENT };
        let err = validate_shared_mem_usage(&layout, DEFAULT_SHARED_MEM_LIMIT).unwrap_err();
        assert!(matches!(err, SharedMemError::AlignmentError { .. }));
    }

    #[test]
    fn validate_empty_layout() {
        let layout =
            SharedMemLayout { offsets: vec![], total_size: 0, alignment: DEFAULT_ALIGNMENT };
        assert!(validate_shared_mem_usage(&layout, DEFAULT_SHARED_MEM_LIMIT).is_ok());
    }

    #[test]
    fn validate_zero_limit() {
        let layout =
            SharedMemLayout { offsets: vec![0], total_size: 1, alignment: DEFAULT_ALIGNMENT };
        let err = validate_shared_mem_usage(&layout, 0).unwrap_err();
        assert!(matches!(err, SharedMemError::ExceedsLimit { .. }));
    }

    // -----------------------------------------------------------------------
    // Error Display
    // -----------------------------------------------------------------------

    #[test]
    fn error_display_allocation_failed() {
        let e = SharedMemError::AllocationFailed("oom".into());
        assert!(e.to_string().contains("oom"));
    }

    #[test]
    fn error_display_exceeds_limit() {
        let e = SharedMemError::ExceedsLimit { required: 100, limit: 50 };
        let s = e.to_string();
        assert!(s.contains("100") && s.contains("50"));
    }

    #[test]
    fn error_display_alignment() {
        let e = SharedMemError::AlignmentError { requested: 128, actual: 13 };
        assert!(e.to_string().contains("128"));
    }

    #[test]
    fn error_display_bank_conflict() {
        let e = SharedMemError::BankConflict { index: 5, bank: 3 };
        assert!(e.to_string().contains("5"));
    }

    // -----------------------------------------------------------------------
    // align_up helper
    // -----------------------------------------------------------------------

    #[test]
    fn align_up_already_aligned() {
        assert_eq!(align_up(128, 128), 128);
    }

    #[test]
    fn align_up_rounds_up() {
        assert_eq!(align_up(1, 128), 128);
    }

    #[test]
    fn align_up_zero() {
        assert_eq!(align_up(0, 128), 0);
    }

    // -----------------------------------------------------------------------
    // Integration / round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn layout_then_validate() {
        let cfgs = vec![
            SharedMemConfig { size: 4096, bank_conflict_free: true, padding: 0 },
            SharedMemConfig { size: 2048, bank_conflict_free: false, padding: 64 },
        ];
        let layout = compute_shared_mem_layout(&cfgs).unwrap();
        validate_shared_mem_usage(&layout, DEFAULT_SHARED_MEM_LIMIT).unwrap();
    }

    #[test]
    fn tile_shared_mem_validates() {
        let tc = optimal_tile_size(256, 256, 256, DEFAULT_SHARED_MEM_LIMIT);
        let elem = std::mem::size_of::<f32>();
        let a_size = tc.tile_m * tc.tile_k * elem;
        let b_size = tc.tile_k * tc.tile_n * elem;
        let factor = if tc.double_buffer { 2 } else { 1 };

        let cfgs: Vec<SharedMemConfig> = (0..factor)
            .flat_map(|_| {
                vec![
                    SharedMemConfig { size: a_size, bank_conflict_free: true, padding: 0 },
                    SharedMemConfig { size: b_size, bank_conflict_free: true, padding: 0 },
                ]
            })
            .collect();

        let layout = compute_shared_mem_layout(&cfgs).unwrap();
        validate_shared_mem_usage(&layout, DEFAULT_SHARED_MEM_LIMIT).unwrap();
    }

    #[test]
    fn layout_multiple_bank_conflict_free() {
        let cfgs = vec![
            SharedMemConfig { size: 1024, bank_conflict_free: true, padding: 0 },
            SharedMemConfig { size: 1024, bank_conflict_free: true, padding: 0 },
            SharedMemConfig { size: 1024, bank_conflict_free: true, padding: 0 },
        ];
        let layout = compute_shared_mem_layout(&cfgs).unwrap();
        assert_eq!(layout.offsets.len(), 3);
        // All offsets aligned.
        for &off in &layout.offsets {
            assert_eq!(off % DEFAULT_ALIGNMENT, 0);
        }
    }

    #[test]
    fn constants_sanity() {
        assert_eq!(NUM_BANKS, 32);
        assert!(DEFAULT_ALIGNMENT.is_power_of_two());
        assert!(DEFAULT_SHARED_MEM_LIMIT > 0);
    }

    // -----------------------------------------------------------------------
    // proptest
    // -----------------------------------------------------------------------

    mod proptests {
        use super::super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn layout_total_size_is_aligned(
                size in 0_usize..8192,
                padding in 0_usize..256,
                bcf in proptest::bool::ANY,
            ) {
                let cfg = SharedMemConfig { size, bank_conflict_free: bcf, padding };
                let layout = compute_shared_mem_layout(&[cfg]).unwrap();
                prop_assert_eq!(layout.total_size % DEFAULT_ALIGNMENT, 0);
            }

            #[test]
            fn layout_offsets_are_aligned(
                sizes in proptest::collection::vec(1_usize..4096, 1..6),
            ) {
                let cfgs: Vec<SharedMemConfig> = sizes
                    .into_iter()
                    .map(|s| SharedMemConfig { size: s, bank_conflict_free: false, padding: 0 })
                    .collect();
                let layout = compute_shared_mem_layout(&cfgs).unwrap();
                for &off in &layout.offsets {
                    prop_assert_eq!(off % DEFAULT_ALIGNMENT, 0);
                }
            }

            #[test]
            fn bank_conflict_free_index_monotonic(
                idx in 0_usize..100_000,
                banks in 1_usize..128,
            ) {
                let a = bank_conflict_free_index(idx, banks);
                let b = bank_conflict_free_index(idx + 1, banks);
                prop_assert!(b > a, "bcf index must be strictly increasing");
            }

            #[test]
            fn tile_fits_limit(
                m in 1_usize..2048,
                n in 1_usize..2048,
                k in 1_usize..2048,
                limit in 64_usize..262_144,
            ) {
                let tc = optimal_tile_size(m, n, k, limit);
                let elem = std::mem::size_of::<f32>();
                let footprint = (tc.tile_m * tc.tile_k + tc.tile_k * tc.tile_n) * elem;
                let factor = if tc.double_buffer { 2 } else { 1 };
                prop_assert!(footprint * factor <= limit);
            }

            #[test]
            fn double_buffer_offset_is_linear(
                buf in 0_usize..8,
                elem in 1_usize..64,
                count in 0_usize..1024,
            ) {
                let off = double_buffer_offset(buf, elem, count);
                prop_assert_eq!(off, buf * elem * count);
            }
        }
    }
}
