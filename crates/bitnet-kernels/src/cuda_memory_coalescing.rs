//! CUDA memory coalescing analysis and optimization.
//!
//! Provides tools for analyzing GPU memory access patterns, simulating
//! memory transactions, and suggesting layout optimizations to improve
//! coalescing efficiency.
//!
//! All analysis runs on CPU as a reference implementation — no GPU required.

use std::collections::HashSet;
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Warp size (threads per warp) on NVIDIA GPUs.
pub const WARP_SIZE: u32 = 32;

/// Size of a single memory transaction in bytes (L1 cache-line).
pub const TRANSACTION_SIZE_32B: u32 = 32;

/// Size of a coalesced memory transaction in bytes (L2 cache-line).
pub const TRANSACTION_SIZE_128B: u32 = 128;

/// Shared-memory bank count on modern NVIDIA GPUs.
pub const SHARED_MEMORY_BANKS: u32 = 32;

/// Default shared-memory bank width in bytes.
pub const BANK_WIDTH_BYTES: u32 = 4;

// ---------------------------------------------------------------------------
// MemoryAccessPattern
// ---------------------------------------------------------------------------

/// Classification of a warp's memory access pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryAccessPattern {
    /// All threads in a warp access consecutive addresses (ideal).
    Coalesced,
    /// Threads access addresses with a constant stride > 1 element.
    Strided {
        /// Distance in *elements* between consecutive thread accesses.
        stride: u32,
    },
    /// Threads access arbitrary non-uniform addresses.
    Scattered,
    /// All threads read the same address.
    Broadcast,
}

impl fmt::Display for MemoryAccessPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Coalesced => write!(f, "Coalesced"),
            Self::Strided { stride } => write!(f, "Strided(stride={stride})"),
            Self::Scattered => write!(f, "Scattered"),
            Self::Broadcast => write!(f, "Broadcast"),
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryLayout
// ---------------------------------------------------------------------------

/// Describes how a tensor is laid out in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLayout {
    /// Row-major (C order) — last dimension is contiguous.
    RowMajor,
    /// Column-major (Fortran order) — first dimension is contiguous.
    ColumnMajor,
    /// Tiled layout with a given tile width and height.
    Tiled { tile_w: u32, tile_h: u32 },
}

impl fmt::Display for MemoryLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowMajor => write!(f, "RowMajor"),
            Self::ColumnMajor => write!(f, "ColumnMajor"),
            Self::Tiled { tile_w, tile_h } => {
                write!(f, "Tiled({tile_w}x{tile_h})")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AccessReport
// ---------------------------------------------------------------------------

/// Statistics produced by analysing a set of memory accesses.
#[derive(Debug, Clone, PartialEq)]
pub struct AccessReport {
    /// Detected access pattern.
    pub pattern: MemoryAccessPattern,
    /// Number of 32-byte transactions required.
    pub transactions_32b: u32,
    /// Number of 128-byte transactions required.
    pub transactions_128b: u32,
    /// Ratio of useful bytes to total fetched bytes (0.0–1.0).
    pub coalescing_ratio: f64,
    /// Effective bandwidth as a fraction of peak (0.0–1.0).
    pub effective_bandwidth_ratio: f64,
    /// Bytes fetched but not used by any thread.
    pub wasted_bytes: u64,
    /// Total bytes actually requested by threads.
    pub requested_bytes: u64,
}

impl AccessReport {
    /// A perfect coalescing ratio is 1.0.
    #[must_use]
    pub fn is_perfectly_coalesced(&self) -> bool {
        (self.coalescing_ratio - 1.0).abs() < f64::EPSILON
    }
}

// ---------------------------------------------------------------------------
// CoalescingAnalyzer
// ---------------------------------------------------------------------------

/// Analyzes memory access patterns for a warp of threads.
#[derive(Debug, Clone)]
pub struct CoalescingAnalyzer {
    /// Number of threads in the warp (typically 32).
    warp_size: u32,
    /// Element size in bytes.
    element_bytes: u32,
}

impl CoalescingAnalyzer {
    /// Create an analyzer for the given element size.
    #[must_use]
    pub fn new(element_bytes: u32) -> Self {
        Self { warp_size: WARP_SIZE, element_bytes: element_bytes.max(1) }
    }

    /// Create an analyzer with a custom warp size (useful for testing).
    #[must_use]
    pub fn with_warp_size(element_bytes: u32, warp_size: u32) -> Self {
        Self { warp_size: warp_size.max(1), element_bytes: element_bytes.max(1) }
    }

    /// Return the configured warp size.
    #[must_use]
    pub fn warp_size(&self) -> u32 {
        self.warp_size
    }

    /// Classify a warp access given a base address and per-element stride.
    #[must_use]
    pub fn classify(&self, _base_addr: u64, stride_elements: u32) -> MemoryAccessPattern {
        if stride_elements == 0 {
            return MemoryAccessPattern::Broadcast;
        }
        let stride_bytes = stride_elements as u64 * self.element_bytes as u64;
        if stride_bytes == self.element_bytes as u64 {
            MemoryAccessPattern::Coalesced
        } else {
            MemoryAccessPattern::Strided { stride: stride_elements }
        }
    }

    /// Classify from an explicit list of byte-addresses (one per thread).
    #[must_use]
    pub fn classify_addresses(&self, addresses: &[u64]) -> MemoryAccessPattern {
        if addresses.is_empty() {
            return MemoryAccessPattern::Coalesced;
        }
        // All same → Broadcast
        if addresses.iter().all(|&a| a == addresses[0]) {
            return MemoryAccessPattern::Broadcast;
        }
        // Check for constant stride
        if addresses.len() >= 2 {
            let stride = addresses[1] as i64 - addresses[0] as i64;
            let is_strided = addresses.windows(2).all(|w| (w[1] as i64 - w[0] as i64) == stride);
            if is_strided {
                let stride_elements = stride.unsigned_abs() / self.element_bytes as u64;
                if stride_elements == 1 {
                    return MemoryAccessPattern::Coalesced;
                }
                return MemoryAccessPattern::Strided { stride: stride_elements as u32 };
            }
        }
        MemoryAccessPattern::Scattered
    }

    /// Produce a full [`AccessReport`] for the given per-thread byte addresses.
    #[must_use]
    pub fn analyze(&self, addresses: &[u64]) -> AccessReport {
        let pattern = self.classify_addresses(addresses);
        let requested_bytes = addresses.len() as u64 * self.element_bytes as u64;

        let (tx_32, tx_128) =
            TransactionSimulator::count_transactions(addresses, self.element_bytes);

        let fetched_32 = tx_32 as u64 * TRANSACTION_SIZE_32B as u64;
        let _fetched_128 = tx_128 as u64 * TRANSACTION_SIZE_128B as u64;
        let wasted_32 = fetched_32.saturating_sub(requested_bytes);

        let coalescing_ratio =
            if fetched_32 == 0 { 1.0 } else { requested_bytes as f64 / fetched_32 as f64 };

        // Effective bandwidth: what fraction of ideal (one 128B tx per 32 threads
        // of 4-byte elements) we achieve.
        let ideal_128b_txns = requested_bytes.div_ceil(TRANSACTION_SIZE_128B as u64);
        let effective_bandwidth_ratio =
            if tx_128 == 0 { 1.0 } else { ideal_128b_txns as f64 / tx_128 as f64 };

        AccessReport {
            pattern,
            transactions_32b: tx_32,
            transactions_128b: tx_128,
            coalescing_ratio: coalescing_ratio.min(1.0),
            effective_bandwidth_ratio: effective_bandwidth_ratio.min(1.0),
            wasted_bytes: wasted_32,
            requested_bytes,
        }
    }
}

impl Default for CoalescingAnalyzer {
    fn default() -> Self {
        Self::new(4) // f32
    }
}

// ---------------------------------------------------------------------------
// TransactionSimulator
// ---------------------------------------------------------------------------

/// Simulates memory transactions for a set of thread addresses.
pub struct TransactionSimulator;

impl TransactionSimulator {
    /// Count (32B transactions, 128B transactions) required to service all
    /// accesses.  Each address is assumed to access `element_bytes` bytes.
    #[must_use]
    pub fn count_transactions(addresses: &[u64], element_bytes: u32) -> (u32, u32) {
        if addresses.is_empty() {
            return (0, 0);
        }
        let elem = element_bytes as u64;
        let mut lines_32: HashSet<u64> = HashSet::new();
        let mut lines_128: HashSet<u64> = HashSet::new();
        for &addr in addresses {
            // The element may span two cache lines if misaligned.
            let first = addr;
            let last = addr + elem.saturating_sub(1);
            lines_32.insert(first / TRANSACTION_SIZE_32B as u64);
            lines_32.insert(last / TRANSACTION_SIZE_32B as u64);
            lines_128.insert(first / TRANSACTION_SIZE_128B as u64);
            lines_128.insert(last / TRANSACTION_SIZE_128B as u64);
        }
        (lines_32.len() as u32, lines_128.len() as u32)
    }

    /// Build the byte-address list for a strided access pattern.
    #[must_use]
    pub fn generate_addresses(base: u64, stride_bytes: u64, count: u32) -> Vec<u64> {
        (0..count).map(|i| base + i as u64 * stride_bytes).collect()
    }
}

// ---------------------------------------------------------------------------
// CoalescingOptimizer
// ---------------------------------------------------------------------------

/// Suggests layout transformations to improve coalescing.
#[derive(Debug, Clone)]
pub struct CoalescingOptimizer;

/// A recommended layout change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutRecommendation {
    /// Current layout.
    pub current: MemoryLayout,
    /// Suggested layout.
    pub suggested: MemoryLayout,
    /// Human-readable reason for the suggestion.
    pub reason: String,
}

impl CoalescingOptimizer {
    /// Suggest a better layout given the current one and an observed pattern.
    #[must_use]
    pub fn suggest(
        current_layout: MemoryLayout,
        pattern: MemoryAccessPattern,
    ) -> Option<LayoutRecommendation> {
        match (current_layout, pattern) {
            // Column-major with strided access → switch to row-major.
            (MemoryLayout::ColumnMajor, MemoryAccessPattern::Strided { .. }) => {
                Some(LayoutRecommendation {
                    current: current_layout,
                    suggested: MemoryLayout::RowMajor,
                    reason: "Column-major layout causes strided access; \
                             switch to row-major for coalesced reads"
                        .into(),
                })
            }
            // Row-major with strided access → try tiling.
            (MemoryLayout::RowMajor, MemoryAccessPattern::Strided { stride }) => {
                let tile = stride.next_power_of_two().clamp(16, 64);
                Some(LayoutRecommendation {
                    current: current_layout,
                    suggested: MemoryLayout::Tiled { tile_w: tile, tile_h: tile },
                    reason: format!(
                        "Row-major with stride {stride}; tiled layout ({tile}x{tile}) \
                         can reduce transaction count"
                    ),
                })
            }
            // Scattered → always recommend tiling.
            (layout, MemoryAccessPattern::Scattered)
                if layout != MemoryLayout::Tiled { tile_w: 32, tile_h: 32 } =>
            {
                Some(LayoutRecommendation {
                    current: current_layout,
                    suggested: MemoryLayout::Tiled { tile_w: 32, tile_h: 32 },
                    reason: "Scattered access detected; tiled layout may improve locality".into(),
                })
            }
            // Already optimal.
            _ => None,
        }
    }

    /// Suggest padding for a row width (in bytes) to meet alignment.
    #[must_use]
    pub fn suggest_padding(row_bytes: u32, alignment: u32) -> u32 {
        if alignment == 0 {
            return 0;
        }
        let remainder = row_bytes % alignment;
        if remainder == 0 { 0 } else { alignment - remainder }
    }
}

// ---------------------------------------------------------------------------
// Bank-conflict helpers
// ---------------------------------------------------------------------------

/// Compute the number of shared-memory bank conflicts for the given
/// per-thread word offsets (measured in `BANK_WIDTH_BYTES`-wide words).
///
/// Returns the *conflict degree* — the maximum number of threads that hit the
/// same bank.  A value of 1 means no conflicts.
#[must_use]
pub fn bank_conflict_degree(offsets: &[u32]) -> u32 {
    if offsets.is_empty() {
        return 0;
    }
    let mut counts = [0u32; 32]; // SHARED_MEMORY_BANKS = 32
    for &off in offsets {
        let bank = off % SHARED_MEMORY_BANKS;
        counts[bank as usize] += 1;
    }
    *counts.iter().max().unwrap_or(&0)
}

/// Check whether the access is conflict-free.
#[must_use]
pub fn is_bank_conflict_free(offsets: &[u32]) -> bool {
    bank_conflict_degree(offsets) <= 1
}

// ---------------------------------------------------------------------------
// Alignment / padding utilities
// ---------------------------------------------------------------------------

/// Round `value` up to the next multiple of `alignment`.
/// Returns `value` unchanged when `alignment` is 0.
#[must_use]
pub fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let mask = alignment - 1;
    // Works correctly only when alignment is a power of two, but still gives a
    // correct result for non-power-of-two via the general formula.
    if alignment.is_power_of_two() {
        (value + mask) & !mask
    } else {
        value.div_ceil(alignment) * alignment
    }
}

/// Compute the padding (in bytes) needed to align `size` to `alignment`.
#[must_use]
pub fn padding_for_alignment(size: u64, alignment: u64) -> u64 {
    align_up(size, alignment) - size
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // MemoryAccessPattern display / equality
    // ---------------------------------------------------------------

    #[test]
    fn pattern_display_coalesced() {
        assert_eq!(MemoryAccessPattern::Coalesced.to_string(), "Coalesced");
    }

    #[test]
    fn pattern_display_strided() {
        let p = MemoryAccessPattern::Strided { stride: 4 };
        assert_eq!(p.to_string(), "Strided(stride=4)");
    }

    #[test]
    fn pattern_display_scattered() {
        assert_eq!(MemoryAccessPattern::Scattered.to_string(), "Scattered");
    }

    #[test]
    fn pattern_display_broadcast() {
        assert_eq!(MemoryAccessPattern::Broadcast.to_string(), "Broadcast");
    }

    // ---------------------------------------------------------------
    // CoalescingAnalyzer::classify (stride-based)
    // ---------------------------------------------------------------

    #[test]
    fn classify_stride_zero_is_broadcast() {
        let a = CoalescingAnalyzer::new(4);
        assert_eq!(a.classify(0, 0), MemoryAccessPattern::Broadcast);
    }

    #[test]
    fn classify_stride_one_is_coalesced() {
        let a = CoalescingAnalyzer::new(4);
        assert_eq!(a.classify(0, 1), MemoryAccessPattern::Coalesced);
    }

    #[test]
    fn classify_stride_gt_one_is_strided() {
        let a = CoalescingAnalyzer::new(4);
        assert_eq!(a.classify(0, 8), MemoryAccessPattern::Strided { stride: 8 });
    }

    // ---------------------------------------------------------------
    // CoalescingAnalyzer::classify_addresses
    // ---------------------------------------------------------------

    #[test]
    fn classify_addresses_empty() {
        let a = CoalescingAnalyzer::new(4);
        assert_eq!(a.classify_addresses(&[]), MemoryAccessPattern::Coalesced);
    }

    #[test]
    fn classify_addresses_single_thread() {
        let a = CoalescingAnalyzer::new(4);
        assert_eq!(a.classify_addresses(&[100]), MemoryAccessPattern::Broadcast);
    }

    #[test]
    fn classify_addresses_broadcast() {
        let a = CoalescingAnalyzer::new(4);
        let addrs = vec![256; 32];
        assert_eq!(a.classify_addresses(&addrs), MemoryAccessPattern::Broadcast);
    }

    #[test]
    fn classify_addresses_coalesced() {
        let a = CoalescingAnalyzer::new(4);
        let addrs: Vec<u64> = (0..32).map(|i| 1024 + i * 4).collect();
        assert_eq!(a.classify_addresses(&addrs), MemoryAccessPattern::Coalesced);
    }

    #[test]
    fn classify_addresses_strided() {
        let a = CoalescingAnalyzer::new(4);
        // stride = 3 elements → 12 bytes between consecutive
        let addrs: Vec<u64> = (0..32).map(|i| 0 + i * 12).collect();
        assert_eq!(a.classify_addresses(&addrs), MemoryAccessPattern::Strided { stride: 3 });
    }

    #[test]
    fn classify_addresses_scattered() {
        let a = CoalescingAnalyzer::new(4);
        let addrs = vec![0, 100, 37, 4096, 12, 999, 48, 7000];
        assert_eq!(a.classify_addresses(&addrs), MemoryAccessPattern::Scattered);
    }

    // ---------------------------------------------------------------
    // Full analyze()
    // ---------------------------------------------------------------

    #[test]
    fn analyze_perfect_coalescing_f32() {
        let a = CoalescingAnalyzer::new(4);
        let addrs: Vec<u64> = (0..32).map(|i| i * 4).collect(); // 128 B contiguous
        let report = a.analyze(&addrs);
        assert_eq!(report.pattern, MemoryAccessPattern::Coalesced);
        assert_eq!(report.requested_bytes, 128);
        assert!(report.coalescing_ratio > 0.99);
    }

    #[test]
    fn analyze_broadcast_single_address() {
        let a = CoalescingAnalyzer::new(4);
        let addrs = vec![0u64; 32];
        let report = a.analyze(&addrs);
        assert_eq!(report.pattern, MemoryAccessPattern::Broadcast);
        // Only one 32B line touched
        assert_eq!(report.transactions_32b, 1);
    }

    #[test]
    fn analyze_strided_access() {
        let a = CoalescingAnalyzer::new(4);
        // stride = 2 elements → 8 bytes apart
        let addrs: Vec<u64> = (0..32).map(|i| i * 8).collect();
        let report = a.analyze(&addrs);
        assert_eq!(report.pattern, MemoryAccessPattern::Strided { stride: 2 });
        // 32 threads × 8 byte stride = 248 byte span, needs >1 128B tx
        assert!(report.transactions_128b >= 2);
    }

    #[test]
    fn analyze_empty_addresses() {
        let a = CoalescingAnalyzer::new(4);
        let report = a.analyze(&[]);
        assert_eq!(report.transactions_32b, 0);
        assert_eq!(report.transactions_128b, 0);
        assert_eq!(report.requested_bytes, 0);
    }

    #[test]
    fn report_is_perfectly_coalesced() {
        let a = CoalescingAnalyzer::new(4);
        let addrs: Vec<u64> = (0..32).map(|i| i * 4).collect();
        let report = a.analyze(&addrs);
        assert!(report.is_perfectly_coalesced());
    }

    // ---------------------------------------------------------------
    // TransactionSimulator
    // ---------------------------------------------------------------

    #[test]
    fn transactions_empty() {
        let (t32, t128) = TransactionSimulator::count_transactions(&[], 4);
        assert_eq!(t32, 0);
        assert_eq!(t128, 0);
    }

    #[test]
    fn transactions_single_element() {
        let (t32, t128) = TransactionSimulator::count_transactions(&[0], 4);
        assert_eq!(t32, 1);
        assert_eq!(t128, 1);
    }

    #[test]
    fn transactions_one_cache_line_32b() {
        // 8 × 4-byte elements all within one 32B line
        let addrs: Vec<u64> = (0..8).map(|i| i * 4).collect();
        let (t32, _) = TransactionSimulator::count_transactions(&addrs, 4);
        assert_eq!(t32, 1);
    }

    #[test]
    fn transactions_two_128b_lines() {
        // 32 × 4-byte at stride 8 → spans 256 bytes → 2 × 128B
        let addrs: Vec<u64> = (0..32).map(|i| i * 8).collect();
        let (_, t128) = TransactionSimulator::count_transactions(&addrs, 4);
        assert_eq!(t128, 2);
    }

    #[test]
    fn transactions_misaligned_base() {
        // Start at byte 30, element size 4 → straddles 32B boundary
        let (t32, _) = TransactionSimulator::count_transactions(&[30], 4);
        assert_eq!(t32, 2); // bytes 30..33 crosses 32B boundary at 32
    }

    #[test]
    fn generate_addresses_basic() {
        let addrs = TransactionSimulator::generate_addresses(0, 4, 4);
        assert_eq!(addrs, vec![0, 4, 8, 12]);
    }

    #[test]
    fn generate_addresses_with_base_offset() {
        let addrs = TransactionSimulator::generate_addresses(100, 8, 3);
        assert_eq!(addrs, vec![100, 108, 116]);
    }

    // ---------------------------------------------------------------
    // CoalescingOptimizer
    // ---------------------------------------------------------------

    #[test]
    fn suggest_colmajor_strided_to_rowmajor() {
        let rec = CoalescingOptimizer::suggest(
            MemoryLayout::ColumnMajor,
            MemoryAccessPattern::Strided { stride: 128 },
        );
        assert!(rec.is_some());
        let rec = rec.unwrap();
        assert_eq!(rec.suggested, MemoryLayout::RowMajor);
    }

    #[test]
    fn suggest_rowmajor_strided_to_tiled() {
        let rec = CoalescingOptimizer::suggest(
            MemoryLayout::RowMajor,
            MemoryAccessPattern::Strided { stride: 16 },
        );
        assert!(rec.is_some());
        let rec = rec.unwrap();
        assert!(matches!(rec.suggested, MemoryLayout::Tiled { .. }));
    }

    #[test]
    fn suggest_scattered_to_tiled() {
        let rec =
            CoalescingOptimizer::suggest(MemoryLayout::RowMajor, MemoryAccessPattern::Scattered);
        assert!(rec.is_some());
        assert_eq!(rec.unwrap().suggested, MemoryLayout::Tiled { tile_w: 32, tile_h: 32 });
    }

    #[test]
    fn suggest_none_for_coalesced() {
        let rec =
            CoalescingOptimizer::suggest(MemoryLayout::RowMajor, MemoryAccessPattern::Coalesced);
        assert!(rec.is_none());
    }

    #[test]
    fn suggest_none_for_broadcast() {
        let rec =
            CoalescingOptimizer::suggest(MemoryLayout::RowMajor, MemoryAccessPattern::Broadcast);
        assert!(rec.is_none());
    }

    #[test]
    fn suggest_padding_aligned() {
        assert_eq!(CoalescingOptimizer::suggest_padding(128, 128), 0);
    }

    #[test]
    fn suggest_padding_unaligned() {
        assert_eq!(CoalescingOptimizer::suggest_padding(100, 128), 28);
    }

    #[test]
    fn suggest_padding_zero_alignment() {
        assert_eq!(CoalescingOptimizer::suggest_padding(100, 0), 0);
    }

    // ---------------------------------------------------------------
    // Bank-conflict helpers
    // ---------------------------------------------------------------

    #[test]
    fn bank_conflict_free_sequential() {
        let offsets: Vec<u32> = (0..32).collect();
        assert_eq!(bank_conflict_degree(&offsets), 1);
        assert!(is_bank_conflict_free(&offsets));
    }

    #[test]
    fn bank_conflict_all_same_bank() {
        // All threads hit bank 0 (stride = 32 words)
        let offsets: Vec<u32> = (0..32).map(|i| i * 32).collect();
        assert_eq!(bank_conflict_degree(&offsets), 32);
        assert!(!is_bank_conflict_free(&offsets));
    }

    #[test]
    fn bank_conflict_two_way() {
        // stride 16 with 32 threads: banks cycle every 32 words,
        // so offsets 0,16,32,48,... → banks 0,16,0,16,...  → 2 distinct banks
        // with 16 threads each.
        let offsets: Vec<u32> = (0..32).map(|i| i * 16).collect();
        let degree = bank_conflict_degree(&offsets);
        assert_eq!(degree, 16);
    }

    #[test]
    fn bank_conflict_empty() {
        assert_eq!(bank_conflict_degree(&[]), 0);
    }

    // ---------------------------------------------------------------
    // Alignment / padding utilities
    // ---------------------------------------------------------------

    #[test]
    fn align_up_already_aligned() {
        assert_eq!(align_up(128, 128), 128);
    }

    #[test]
    fn align_up_needs_padding() {
        assert_eq!(align_up(100, 128), 128);
    }

    #[test]
    fn align_up_zero_alignment() {
        assert_eq!(align_up(42, 0), 42);
    }

    #[test]
    fn align_up_non_power_of_two() {
        assert_eq!(align_up(10, 6), 12);
    }

    #[test]
    fn padding_for_alignment_exact() {
        assert_eq!(padding_for_alignment(128, 128), 0);
    }

    #[test]
    fn padding_for_alignment_needs_pad() {
        assert_eq!(padding_for_alignment(100, 128), 28);
    }

    // ---------------------------------------------------------------
    // MemoryLayout display
    // ---------------------------------------------------------------

    #[test]
    fn layout_display() {
        assert_eq!(MemoryLayout::RowMajor.to_string(), "RowMajor");
        assert_eq!(MemoryLayout::ColumnMajor.to_string(), "ColumnMajor");
        assert_eq!(MemoryLayout::Tiled { tile_w: 16, tile_h: 16 }.to_string(), "Tiled(16x16)");
    }

    // ---------------------------------------------------------------
    // Edge cases
    // ---------------------------------------------------------------

    #[test]
    fn analyzer_element_size_one() {
        let a = CoalescingAnalyzer::new(1);
        let addrs: Vec<u64> = (0..32).collect();
        let report = a.analyze(&addrs);
        assert_eq!(report.pattern, MemoryAccessPattern::Coalesced);
        assert_eq!(report.requested_bytes, 32);
    }

    #[test]
    fn analyzer_large_element() {
        let a = CoalescingAnalyzer::new(16); // e.g. float4
        let addrs: Vec<u64> = (0..32).map(|i| i * 16).collect();
        let report = a.analyze(&addrs);
        assert_eq!(report.pattern, MemoryAccessPattern::Coalesced);
        assert_eq!(report.requested_bytes, 512);
    }

    #[test]
    fn analyzer_custom_warp_size() {
        let a = CoalescingAnalyzer::with_warp_size(4, 8);
        assert_eq!(a.warp_size, 8);
        assert_eq!(a.classify(0, 1), MemoryAccessPattern::Coalesced);
    }

    #[test]
    fn wasted_bytes_with_stride() {
        let a = CoalescingAnalyzer::new(4);
        // Stride 8 → every other 4B slot is wasted
        let addrs: Vec<u64> = (0..4).map(|i| i * 8).collect();
        let report = a.analyze(&addrs);
        assert!(report.wasted_bytes > 0);
    }

    #[test]
    fn zero_wasted_for_dense_access() {
        let a = CoalescingAnalyzer::new(4);
        // 8 × 4B = 32B = exactly one 32B line, zero waste
        let addrs: Vec<u64> = (0..8).map(|i| i * 4).collect();
        let report = a.analyze(&addrs);
        assert_eq!(report.wasted_bytes, 0);
    }

    // ---------------------------------------------------------------
    // Property-style invariant tests
    // ---------------------------------------------------------------

    #[test]
    fn invariant_coalescing_ratio_bounded() {
        let a = CoalescingAnalyzer::new(4);
        for stride in [1, 2, 4, 8, 16, 32, 64] {
            let addrs: Vec<u64> = (0..32).map(|i| i * stride * 4).collect();
            let report = a.analyze(&addrs);
            assert!(
                report.coalescing_ratio >= 0.0 && report.coalescing_ratio <= 1.0,
                "coalescing_ratio out of range for stride {stride}: {}",
                report.coalescing_ratio
            );
        }
    }

    #[test]
    fn invariant_bandwidth_ratio_bounded() {
        let a = CoalescingAnalyzer::new(4);
        for stride in [1, 2, 4, 8, 16] {
            let addrs: Vec<u64> = (0..32).map(|i| i * stride * 4).collect();
            let report = a.analyze(&addrs);
            assert!(
                report.effective_bandwidth_ratio >= 0.0 && report.effective_bandwidth_ratio <= 1.0,
                "effective_bandwidth_ratio out of range for stride {stride}: {}",
                report.effective_bandwidth_ratio
            );
        }
    }

    #[test]
    fn invariant_transactions_non_decreasing_with_stride() {
        let a = CoalescingAnalyzer::new(4);
        let mut prev_128 = 0u32;
        for stride in [1u64, 2, 4, 8] {
            let addrs: Vec<u64> = (0..32).map(|i| i * stride * 4).collect();
            let report = a.analyze(&addrs);
            assert!(
                report.transactions_128b >= prev_128,
                "128B txns should not decrease: stride={stride}, got {}, prev {prev_128}",
                report.transactions_128b
            );
            prev_128 = report.transactions_128b;
        }
    }

    #[test]
    fn invariant_requested_equals_threads_times_elem() {
        let a = CoalescingAnalyzer::new(4);
        for n in [1, 8, 16, 32] {
            let addrs: Vec<u64> = (0..n).map(|i| i * 4).collect();
            let report = a.analyze(&addrs);
            assert_eq!(report.requested_bytes, n * 4);
        }
    }

    #[test]
    fn invariant_wasted_never_exceeds_fetched() {
        let a = CoalescingAnalyzer::new(4);
        for stride in [1u64, 3, 7, 16, 64] {
            let addrs: Vec<u64> = (0..32).map(|i| i * stride * 4).collect();
            let report = a.analyze(&addrs);
            let fetched = report.transactions_32b as u64 * TRANSACTION_SIZE_32B as u64;
            assert!(
                report.wasted_bytes <= fetched,
                "wasted_bytes ({}) > fetched ({}) for stride {stride}",
                report.wasted_bytes,
                fetched
            );
        }
    }

    #[test]
    fn invariant_bank_conflict_degree_bounded_by_thread_count() {
        for n in [1u32, 8, 16, 32] {
            let offsets: Vec<u32> = (0..n).collect();
            let degree = bank_conflict_degree(&offsets);
            assert!(degree <= n, "degree {degree} > thread count {n}");
        }
    }

    #[test]
    fn suggest_tiled_stride_clamped() {
        // Very large stride → tile should be clamped at 64.
        let rec = CoalescingOptimizer::suggest(
            MemoryLayout::RowMajor,
            MemoryAccessPattern::Strided { stride: 1024 },
        );
        if let Some(r) = rec {
            if let MemoryLayout::Tiled { tile_w, tile_h } = r.suggested {
                assert!(tile_w <= 64 && tile_h <= 64);
            }
        }
    }
}
