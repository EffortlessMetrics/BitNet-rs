//! CUDA memory coalescing analysis for optimized GPU memory access.
//!
//! # Overview
//!
//! Provides tools for analysing and optimising GPU global-memory access patterns.
//! Coalesced memory accesses are critical for CUDA performance — when threads in a
//! warp access contiguous, aligned addresses the hardware can merge requests into
//! fewer memory transactions.
//!
//! Key components:
//!
//! - [`AccessPattern`] / [`MemoryAccessDescriptor`] — describe how threads access memory.
//! - [`CoalescingAnalyzer`] — estimate coalescing efficiency for a given pattern.
//! - [`LayoutTransformer`] — convert AoS ↔ SoA and apply padding for alignment.
//! - [`WarpTransactionAnalyzer`] — model warp-level 32-/128-byte transactions.
//! - [`BankConflictDetector`] — detect shared-memory bank conflicts.
//! - [`AccessReorderAdvisor`] — recommend reorderings for improved coalescing.
//!
//! All CUDA kernel source constants are feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`. The analysis types and CPU
//! helpers compile unconditionally for testing on non-GPU hosts.

use bitnet_common::{KernelError, Result};
use std::collections::{BTreeMap, HashMap};

// ══════════════════════════════════════════════════════════════════════
// Constants
// ══════════════════════════════════════════════════════════════════════

/// Number of threads per warp on NVIDIA GPUs.
pub const WARP_SIZE: usize = 32;

/// Width of a global-memory cache line (bytes).  Memory transactions are
/// issued in 32-byte or 128-byte segments depending on the cache mode.
pub const CACHE_LINE_BYTES: usize = 128;

/// Shared-memory bank width in bytes (4 bytes / 32-bit words).
pub const SHARED_MEM_BANK_WIDTH: usize = 4;

/// Number of shared-memory banks on modern NVIDIA GPUs.
pub const NUM_SHARED_MEM_BANKS: usize = 32;

/// Minimum alignment (bytes) for efficient global memory access.
pub const MIN_GLOBAL_ALIGNMENT: usize = 128;

// ══════════════════════════════════════════════════════════════════════
// Access pattern types
// ══════════════════════════════════════════════════════════════════════

/// Classification of a warp's global memory access pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessPattern {
    /// Threads access consecutive addresses (stride = element size).
    Sequential,
    /// Threads access addresses with a constant stride > element size.
    Strided {
        /// Stride in bytes between successive threads.
        stride_bytes: usize,
    },
    /// All threads read the same address.
    Broadcast,
    /// Threads access addresses with no regular pattern.
    Scattered,
    /// Threads in the warp access a blocked sub-range.
    Blocked {
        /// Number of consecutive elements per thread.
        block_size: usize,
    },
}

impl AccessPattern {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Strided { .. } => "strided",
            Self::Broadcast => "broadcast",
            Self::Scattered => "scattered",
            Self::Blocked { .. } => "blocked",
        }
    }
}

impl std::fmt::Display for AccessPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sequential => write!(f, "sequential"),
            Self::Strided { stride_bytes } => write!(f, "strided({}B)", stride_bytes),
            Self::Broadcast => write!(f, "broadcast"),
            Self::Scattered => write!(f, "scattered"),
            Self::Blocked { block_size } => write!(f, "blocked({})", block_size),
        }
    }
}

/// Descriptor for a memory access issued by a warp.
#[derive(Debug, Clone)]
pub struct MemoryAccessDescriptor {
    /// Base address (byte offset from allocation start).
    pub base_address: usize,
    /// Size in bytes of each element accessed per thread.
    pub element_size: usize,
    /// Access pattern classification.
    pub pattern: AccessPattern,
    /// Number of active threads (≤ WARP_SIZE).
    pub active_threads: usize,
    /// Whether the access is a read or write.
    pub is_write: bool,
}

impl MemoryAccessDescriptor {
    /// Create a new descriptor.
    pub fn new(
        base_address: usize,
        element_size: usize,
        pattern: AccessPattern,
        active_threads: usize,
        is_write: bool,
    ) -> Result<Self> {
        if element_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "element_size must be non-zero".into(),
            }
            .into());
        }
        if active_threads == 0 || active_threads > WARP_SIZE {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "active_threads must be in [1, {}], got {}",
                    WARP_SIZE, active_threads
                ),
            }
            .into());
        }
        Ok(Self { base_address, element_size, pattern, active_threads, is_write })
    }

    /// Compute byte addresses for every active thread in the warp.
    pub fn thread_addresses(&self) -> Vec<usize> {
        let mut addrs = Vec::with_capacity(self.active_threads);
        for tid in 0..self.active_threads {
            let addr = match self.pattern {
                AccessPattern::Sequential => self.base_address + tid * self.element_size,
                AccessPattern::Strided { stride_bytes } => self.base_address + tid * stride_bytes,
                AccessPattern::Broadcast => self.base_address,
                AccessPattern::Scattered => {
                    // Simulate a scattered pattern using a simple hash.
                    self.base_address
                        .wrapping_add(tid.wrapping_mul(997))
                        .wrapping_mul(self.element_size)
                }
                AccessPattern::Blocked { block_size } => {
                    self.base_address + tid * block_size * self.element_size
                }
            };
            addrs.push(addr);
        }
        addrs
    }
}

// ══════════════════════════════════════════════════════════════════════
// Coalescing efficiency
// ══════════════════════════════════════════════════════════════════════

/// Result of a coalescing analysis.
#[derive(Debug, Clone)]
pub struct CoalescingReport {
    /// Efficiency in [0.0, 1.0].  1.0 = perfectly coalesced.
    pub efficiency: f64,
    /// Number of 128-byte transactions required.
    pub transactions_128b: usize,
    /// Number of 32-byte transactions required.
    pub transactions_32b: usize,
    /// Total bytes actually accessed by the warp.
    pub useful_bytes: usize,
    /// Total bytes transferred (including wasted bandwidth).
    pub transferred_bytes: usize,
    /// Human-readable recommendation.
    pub recommendation: String,
}

/// Analyzes memory access coalescing efficiency.
#[derive(Debug, Default)]
pub struct CoalescingAnalyzer {
    /// Cache-line size used for transaction modelling (default 128 B).
    pub cache_line_size: usize,
}

impl CoalescingAnalyzer {
    /// Create a new analyzer with the default 128-byte cache line.
    pub fn new() -> Self {
        Self { cache_line_size: CACHE_LINE_BYTES }
    }

    /// Create an analyzer with a custom cache-line size.
    pub fn with_cache_line(cache_line_size: usize) -> Result<Self> {
        if cache_line_size == 0 || !cache_line_size.is_power_of_two() {
            return Err(KernelError::InvalidArguments {
                reason: "cache_line_size must be a non-zero power of two".into(),
            }
            .into());
        }
        Ok(Self { cache_line_size })
    }

    /// Analyze a single memory access descriptor.
    pub fn analyze(&self, desc: &MemoryAccessDescriptor) -> CoalescingReport {
        let addrs = desc.thread_addresses();
        let useful_bytes = desc.active_threads * desc.element_size;

        // Count distinct cache lines touched.
        let mut cache_lines: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for &addr in &addrs {
            let start = addr;
            let end = addr + desc.element_size;
            let first_line = start / self.cache_line_size;
            let last_line = (end.saturating_sub(1)) / self.cache_line_size;
            for line in first_line..=last_line {
                cache_lines.insert(line);
            }
        }
        let transactions_128b = cache_lines.len();
        let transferred_bytes = transactions_128b * self.cache_line_size;

        // 32-byte sub-sector transactions.
        let mut sectors: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for &addr in &addrs {
            let start = addr;
            let end = addr + desc.element_size;
            let first_sector = start / 32;
            let last_sector = (end.saturating_sub(1)) / 32;
            for s in first_sector..=last_sector {
                sectors.insert(s);
            }
        }
        let transactions_32b = sectors.len();

        let efficiency = if transferred_bytes > 0 {
            (useful_bytes as f64) / (transferred_bytes as f64)
        } else {
            0.0
        };

        let recommendation = self.recommend(&desc.pattern, efficiency);

        CoalescingReport {
            efficiency,
            transactions_128b,
            transactions_32b,
            useful_bytes,
            transferred_bytes,
            recommendation,
        }
    }

    /// Analyze a batch of descriptors and return per-descriptor reports.
    pub fn analyze_batch(&self, descriptors: &[MemoryAccessDescriptor]) -> Vec<CoalescingReport> {
        descriptors.iter().map(|d| self.analyze(d)).collect()
    }

    /// Compute aggregate efficiency over a batch.
    pub fn aggregate_efficiency(reports: &[CoalescingReport]) -> f64 {
        if reports.is_empty() {
            return 0.0;
        }
        let total_useful: usize = reports.iter().map(|r| r.useful_bytes).sum();
        let total_transferred: usize = reports.iter().map(|r| r.transferred_bytes).sum();
        if total_transferred == 0 {
            return 0.0;
        }
        total_useful as f64 / total_transferred as f64
    }

    fn recommend(&self, pattern: &AccessPattern, efficiency: f64) -> String {
        match pattern {
            AccessPattern::Sequential if efficiency >= 0.9 => {
                "Access pattern is well-coalesced; no changes needed.".into()
            }
            AccessPattern::Sequential => {
                "Sequential but not fully coalesced — check base-address alignment.".into()
            }
            AccessPattern::Strided { stride_bytes } => {
                format!(
                    "Strided access ({stride_bytes}B) causes {:.0}% wasted bandwidth. \
                     Consider AoS→SoA transformation or shared-memory staging.",
                    (1.0 - efficiency) * 100.0,
                )
            }
            AccessPattern::Broadcast => {
                "Broadcast — all threads read one address; consider using `__ldg` or \
                 constant memory."
                    .into()
            }
            AccessPattern::Scattered => {
                "Scattered access — worst-case coalescing. Use shared-memory gather \
                 or restructure data layout."
                    .into()
            }
            AccessPattern::Blocked { .. } => {
                format!(
                    "Blocked access at {:.1}% efficiency. Transpose to sequential \
                     per-warp access for improvement.",
                    efficiency * 100.0,
                )
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Warp-level transaction analysis
// ══════════════════════════════════════════════════════════════════════

/// A memory transaction issued by the hardware for a warp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryTransaction {
    /// Cache-line-aligned start address.
    pub address: usize,
    /// Size of this transaction in bytes (32 or 128).
    pub size: usize,
    /// Indices of threads served by this transaction.
    pub thread_ids: Vec<usize>,
}

/// Models how a warp's access is broken into hardware transactions.
#[derive(Debug)]
pub struct WarpTransactionAnalyzer {
    /// Transaction granularity in bytes (128 for L1, 32 for L2 sector).
    pub transaction_size: usize,
}

impl WarpTransactionAnalyzer {
    /// Create an analyzer for 128-byte L1 transactions.
    pub fn l1() -> Self {
        Self { transaction_size: 128 }
    }

    /// Create an analyzer for 32-byte L2 sector transactions.
    pub fn l2_sector() -> Self {
        Self { transaction_size: 32 }
    }

    /// Create with a custom transaction size (must be a power of two).
    pub fn new(transaction_size: usize) -> Result<Self> {
        if transaction_size == 0 || !transaction_size.is_power_of_two() {
            return Err(KernelError::InvalidArguments {
                reason: "transaction_size must be a non-zero power of two".into(),
            }
            .into());
        }
        Ok(Self { transaction_size })
    }

    /// Decompose a descriptor into hardware memory transactions.
    pub fn decompose(&self, desc: &MemoryAccessDescriptor) -> Vec<MemoryTransaction> {
        let addrs = desc.thread_addresses();
        // Map: segment_start → thread indices.
        let mut segments: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

        for (tid, &addr) in addrs.iter().enumerate() {
            let start = addr;
            let end = addr + desc.element_size;
            let first_seg = start / self.transaction_size;
            let last_seg = (end.saturating_sub(1)) / self.transaction_size;
            for seg in first_seg..=last_seg {
                segments.entry(seg * self.transaction_size).or_default().push(tid);
            }
        }

        segments
            .into_iter()
            .map(|(address, mut thread_ids)| {
                thread_ids.sort_unstable();
                thread_ids.dedup();
                MemoryTransaction { address, size: self.transaction_size, thread_ids }
            })
            .collect()
    }

    /// Count the total number of transactions for a descriptor.
    pub fn transaction_count(&self, desc: &MemoryAccessDescriptor) -> usize {
        self.decompose(desc).len()
    }

    /// Ideal (minimum) number of transactions for perfectly coalesced access.
    pub fn ideal_transactions(&self, desc: &MemoryAccessDescriptor) -> usize {
        let total_bytes = desc.active_threads * desc.element_size;
        total_bytes.div_ceil(self.transaction_size)
    }

    /// Ratio of ideal to actual transactions (1.0 = perfect).
    pub fn transaction_efficiency(&self, desc: &MemoryAccessDescriptor) -> f64 {
        let actual = self.transaction_count(desc);
        if actual == 0 {
            return 0.0;
        }
        let ideal = self.ideal_transactions(desc);
        ideal as f64 / actual as f64
    }
}

// ══════════════════════════════════════════════════════════════════════
// Bank conflict detection (shared memory)
// ══════════════════════════════════════════════════════════════════════

/// Result of a shared-memory bank-conflict analysis.
#[derive(Debug, Clone)]
pub struct BankConflictReport {
    /// Number of bank conflicts detected.
    pub conflict_count: usize,
    /// Number of N-way conflicts per bank (bank_id → degree).
    pub per_bank_conflicts: HashMap<usize, usize>,
    /// Maximum conflict degree across all banks.
    pub max_conflict_degree: usize,
    /// Whether the access is conflict-free.
    pub is_conflict_free: bool,
    /// Recommended padding to eliminate conflicts.
    pub suggested_padding: usize,
}

/// Detects bank conflicts in shared-memory access patterns.
#[derive(Debug)]
pub struct BankConflictDetector {
    /// Number of banks.
    pub num_banks: usize,
    /// Bank width in bytes.
    pub bank_width: usize,
}

impl Default for BankConflictDetector {
    fn default() -> Self {
        Self { num_banks: NUM_SHARED_MEM_BANKS, bank_width: SHARED_MEM_BANK_WIDTH }
    }
}

impl BankConflictDetector {
    /// Create with default GPU parameters (32 banks, 4-byte width).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom bank configuration.
    pub fn with_config(num_banks: usize, bank_width: usize) -> Result<Self> {
        if num_banks == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_banks must be non-zero".into(),
            }
            .into());
        }
        if bank_width == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "bank_width must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { num_banks, bank_width })
    }

    /// Compute the bank index for a byte address.
    pub fn bank_for_address(&self, byte_address: usize) -> usize {
        (byte_address / self.bank_width) % self.num_banks
    }

    /// Analyze bank conflicts for a set of byte addresses (one per thread).
    pub fn analyze(&self, addresses: &[usize]) -> BankConflictReport {
        // Count accesses per bank.
        let mut bank_accesses: HashMap<usize, usize> = HashMap::new();
        for &addr in addresses {
            let bank = self.bank_for_address(addr);
            *bank_accesses.entry(bank).or_insert(0) += 1;
        }

        let mut conflict_count: usize = 0;
        let mut per_bank_conflicts: HashMap<usize, usize> = HashMap::new();
        let mut max_degree: usize = 1;

        for (&bank, &count) in &bank_accesses {
            if count > 1 {
                per_bank_conflicts.insert(bank, count);
                conflict_count += count - 1;
                if count > max_degree {
                    max_degree = count;
                }
            }
        }

        let is_conflict_free = conflict_count == 0;
        let suggested_padding = if is_conflict_free { 0 } else { self.bank_width };

        BankConflictReport {
            conflict_count,
            per_bank_conflicts,
            max_conflict_degree: max_degree,
            is_conflict_free,
            suggested_padding,
        }
    }

    /// Analyze bank conflicts for a 2D shared-memory tile access.
    /// `row_stride` is the byte stride between rows (including any padding).
    pub fn analyze_2d(
        &self,
        rows: usize,
        cols: usize,
        element_size: usize,
        row_stride: usize,
    ) -> BankConflictReport {
        // Simulate column-wise access (each thread reads one column across
        // rows) which is the typical conflict-prone pattern.
        let n = rows.min(WARP_SIZE);
        let addrs: Vec<usize> = (0..n).map(|r| r * row_stride).collect();
        let mut report = self.analyze(&addrs);

        // Suggest padding that shifts successive rows to different banks.
        if !report.is_conflict_free {
            // Try padding amounts 1..num_banks until conflict-free.
            for pad in 1..=self.num_banks {
                let padded_stride = cols * element_size + pad * self.bank_width;
                let padded_addrs: Vec<usize> = (0..n).map(|r| r * padded_stride).collect();
                let check = self.analyze(&padded_addrs);
                if check.is_conflict_free {
                    report.suggested_padding = pad * self.bank_width;
                    break;
                }
            }
        }
        report
    }
}

// ══════════════════════════════════════════════════════════════════════
// Data layout transformation (AoS ↔ SoA)
// ══════════════════════════════════════════════════════════════════════

/// Data layout enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataLayout {
    /// Array of Structures: fields interleaved per element.
    AoS,
    /// Structure of Arrays: each field stored contiguously.
    SoA,
    /// Hybrid: small groups of elements stored as AoS within SoA tiles.
    AoSoA {
        /// Tile width in number of elements.
        tile_width: usize,
    },
}

impl std::fmt::Display for DataLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AoS => write!(f, "AoS"),
            Self::SoA => write!(f, "SoA"),
            Self::AoSoA { tile_width } => write!(f, "AoSoA({})", tile_width),
        }
    }
}

/// Transforms data between AoS and SoA layouts.
#[derive(Debug)]
pub struct LayoutTransformer;

impl LayoutTransformer {
    /// Convert an AoS buffer to SoA layout.
    ///
    /// `data` — flat array in AoS order: `[e0_f0, e0_f1, …, e0_fN, e1_f0, …]`.
    /// `num_elements` — number of structures.
    /// `num_fields` — number of fields per structure.
    ///
    /// Returns SoA: `[e0_f0, e1_f0, …, eM_f0, e0_f1, e1_f1, …]`.
    pub fn aos_to_soa(data: &[f32], num_elements: usize, num_fields: usize) -> Result<Vec<f32>> {
        let expected = num_elements * num_fields;
        if data.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "data length ({}) != num_elements ({}) * num_fields ({})",
                    data.len(),
                    num_elements,
                    num_fields
                ),
            }
            .into());
        }

        let mut out = vec![0.0f32; expected];
        for elem in 0..num_elements {
            for field in 0..num_fields {
                out[field * num_elements + elem] = data[elem * num_fields + field];
            }
        }
        Ok(out)
    }

    /// Convert an SoA buffer to AoS layout.
    pub fn soa_to_aos(data: &[f32], num_elements: usize, num_fields: usize) -> Result<Vec<f32>> {
        let expected = num_elements * num_fields;
        if data.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "data length ({}) != num_elements ({}) * num_fields ({})",
                    data.len(),
                    num_elements,
                    num_fields
                ),
            }
            .into());
        }

        let mut out = vec![0.0f32; expected];
        for elem in 0..num_elements {
            for field in 0..num_fields {
                out[elem * num_fields + field] = data[field * num_elements + elem];
            }
        }
        Ok(out)
    }

    /// Convert AoS to hybrid AoSoA layout.
    ///
    /// The output is grouped into tiles of `tile_width` elements. Within each
    /// tile the data is in AoS order, but tiles are laid out contiguously per
    /// field for improved coalescing.
    pub fn aos_to_aosoa(
        data: &[f32],
        num_elements: usize,
        num_fields: usize,
        tile_width: usize,
    ) -> Result<Vec<f32>> {
        let expected = num_elements * num_fields;
        if data.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "data length ({}) != num_elements ({}) * num_fields ({})",
                    data.len(),
                    num_elements,
                    num_fields
                ),
            }
            .into());
        }
        if tile_width == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "tile_width must be non-zero".into(),
            }
            .into());
        }

        let num_tiles = num_elements.div_ceil(tile_width);
        let padded_len = num_tiles * tile_width * num_fields;
        let mut out = vec![0.0f32; padded_len];

        for elem in 0..num_elements {
            let tile = elem / tile_width;
            let lane = elem % tile_width;
            for field in 0..num_fields {
                let dst = tile * (tile_width * num_fields) + field * tile_width + lane;
                out[dst] = data[elem * num_fields + field];
            }
        }
        Ok(out)
    }

    /// Estimate the coalescing improvement from AoS → SoA conversion.
    ///
    /// Returns (original_efficiency, converted_efficiency).
    pub fn estimate_improvement(
        num_elements: usize,
        num_fields: usize,
        element_size: usize,
    ) -> (f64, f64) {
        let stride_aos = num_fields * element_size;

        // AoS: strided access when reading a single field across elements.
        let aos_useful = WARP_SIZE.min(num_elements) * element_size;
        let aos_lines = (WARP_SIZE.min(num_elements) * stride_aos).div_ceil(CACHE_LINE_BYTES);
        let aos_transferred = aos_lines * CACHE_LINE_BYTES;
        let eff_aos =
            if aos_transferred > 0 { aos_useful as f64 / aos_transferred as f64 } else { 0.0 };

        // SoA: sequential access.
        let soa_useful = WARP_SIZE.min(num_elements) * element_size;
        let soa_transferred_bytes = WARP_SIZE.min(num_elements) * element_size;
        let soa_lines = soa_transferred_bytes.div_ceil(CACHE_LINE_BYTES);
        let soa_transferred = soa_lines * CACHE_LINE_BYTES;
        let eff_soa =
            if soa_transferred > 0 { soa_useful as f64 / soa_transferred as f64 } else { 0.0 };

        (eff_aos, eff_soa)
    }
}

// ══════════════════════════════════════════════════════════════════════
// Padding strategies
// ══════════════════════════════════════════════════════════════════════

/// Padding strategy for aligned memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// No padding.
    None,
    /// Pad rows to a multiple of `alignment` bytes.
    RowAlignment { alignment: usize },
    /// Add `extra_bytes` padding at the end of each row to avoid bank conflicts.
    BankConflictAvoidance { extra_bytes: usize },
    /// Pad to a power-of-two row width.
    PowerOfTwo,
}

/// Compute the padded row stride given a row width, element size, and strategy.
pub fn compute_padded_stride(
    row_width: usize,
    element_size: usize,
    strategy: PaddingStrategy,
) -> usize {
    let raw_bytes = row_width * element_size;
    match strategy {
        PaddingStrategy::None => raw_bytes,
        PaddingStrategy::RowAlignment { alignment } => {
            if alignment == 0 || !alignment.is_power_of_two() {
                return raw_bytes;
            }
            (raw_bytes + alignment - 1) & !(alignment - 1)
        }
        PaddingStrategy::BankConflictAvoidance { extra_bytes } => raw_bytes + extra_bytes,
        PaddingStrategy::PowerOfTwo => raw_bytes.next_power_of_two(),
    }
}

/// Apply padding to a 2D buffer (row-major).  Returns padded buffer and new
/// row stride in bytes.
pub fn pad_2d_buffer(
    data: &[f32],
    rows: usize,
    cols: usize,
    strategy: PaddingStrategy,
) -> Result<(Vec<f32>, usize)> {
    let expected = rows * cols;
    if data.len() != expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("data length ({}) != rows ({}) * cols ({})", data.len(), rows, cols),
        }
        .into());
    }

    let elem_size = std::mem::size_of::<f32>();
    let padded_stride = compute_padded_stride(cols, elem_size, strategy);
    let padded_cols = padded_stride / elem_size;

    let mut out = vec![0.0f32; rows * padded_cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * padded_cols + c] = data[r * cols + c];
        }
    }
    Ok((out, padded_stride))
}

// ══════════════════════════════════════════════════════════════════════
// Access reordering advisor
// ══════════════════════════════════════════════════════════════════════

/// Severity of a recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational only.
    Info,
    /// Minor improvement possible.
    Low,
    /// Moderate improvement possible.
    Medium,
    /// Significant improvement possible.
    High,
    /// Critical — severe performance loss.
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Info => "info",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        };
        write!(f, "{}", s)
    }
}

/// A single reordering recommendation.
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// Severity level.
    pub severity: Severity,
    /// Short title.
    pub title: String,
    /// Detailed description.
    pub description: String,
    /// Estimated efficiency improvement (0.0–1.0 scale).
    pub estimated_improvement: f64,
}

/// Advisor that inspects access patterns and emits reordering recommendations.
#[derive(Debug, Default)]
pub struct AccessReorderAdvisor;

impl AccessReorderAdvisor {
    /// Create a new advisor.
    pub fn new() -> Self {
        Self
    }

    /// Analyze a descriptor and return recommendations.
    pub fn advise(&self, desc: &MemoryAccessDescriptor) -> Vec<Recommendation> {
        let analyzer = CoalescingAnalyzer::new();
        let report = analyzer.analyze(desc);
        let mut recs = Vec::new();

        // Alignment recommendation.
        if !desc.base_address.is_multiple_of(MIN_GLOBAL_ALIGNMENT) {
            recs.push(Recommendation {
                severity: Severity::Medium,
                title: "Misaligned base address".into(),
                description: format!(
                    "Base address 0x{:x} is not aligned to {} bytes. \
                     Align allocations to {} B for optimal transactions.",
                    desc.base_address, MIN_GLOBAL_ALIGNMENT, MIN_GLOBAL_ALIGNMENT,
                ),
                estimated_improvement: 0.05,
            });
        }

        // Pattern-specific advice.
        match desc.pattern {
            AccessPattern::Strided { stride_bytes } if stride_bytes > desc.element_size => {
                let wasted = 1.0 - report.efficiency;
                let severity = if wasted > 0.75 {
                    Severity::Critical
                } else if wasted > 0.5 {
                    Severity::High
                } else if wasted > 0.25 {
                    Severity::Medium
                } else {
                    Severity::Low
                };
                recs.push(Recommendation {
                    severity,
                    title: "Strided access pattern".into(),
                    description: format!(
                        "Stride of {} bytes wastes {:.0}% bandwidth. \
                         Convert AoS→SoA or use shared-memory staging.",
                        stride_bytes,
                        wasted * 100.0,
                    ),
                    estimated_improvement: wasted * 0.8,
                });
            }
            AccessPattern::Scattered => {
                recs.push(Recommendation {
                    severity: Severity::Critical,
                    title: "Scattered memory access".into(),
                    description: "Completely irregular access pattern. Consider \
                                  binning or sorting data for spatial locality."
                        .into(),
                    estimated_improvement: 0.5,
                });
            }
            AccessPattern::Blocked { block_size } if block_size > 1 => {
                recs.push(Recommendation {
                    severity: Severity::Medium,
                    title: "Blocked access pattern".into(),
                    description: format!(
                        "Block size {} causes gaps between thread accesses. \
                         Consider tiling or thread-mapping changes.",
                        block_size,
                    ),
                    estimated_improvement: 0.2,
                });
            }
            _ => {}
        }

        // Under-utilised warp.
        if desc.active_threads < WARP_SIZE / 2 {
            recs.push(Recommendation {
                severity: Severity::Low,
                title: "Low warp utilization".into(),
                description: format!(
                    "Only {}/{} threads active — consider increasing occupancy.",
                    desc.active_threads, WARP_SIZE,
                ),
                estimated_improvement: 0.1,
            });
        }

        recs
    }

    /// Analyze multiple descriptors and deduplicate recommendations.
    pub fn advise_batch(&self, descriptors: &[MemoryAccessDescriptor]) -> Vec<Recommendation> {
        let mut all: Vec<Recommendation> = Vec::new();
        let mut seen_titles: std::collections::HashSet<String> = std::collections::HashSet::new();

        for desc in descriptors {
            for rec in self.advise(desc) {
                if seen_titles.insert(rec.title.clone()) {
                    all.push(rec);
                }
            }
        }
        all.sort_by(|a, b| b.severity.cmp(&a.severity));
        all
    }
}

// ══════════════════════════════════════════════════════════════════════
// Detect access pattern from raw addresses
// ══════════════════════════════════════════════════════════════════════

/// Detect the access pattern from a list of per-thread addresses.
pub fn detect_pattern(addresses: &[usize], element_size: usize) -> AccessPattern {
    if addresses.is_empty() {
        return AccessPattern::Sequential;
    }
    if addresses.len() == 1 {
        return AccessPattern::Sequential;
    }

    // Check broadcast.
    if addresses.iter().all(|&a| a == addresses[0]) {
        return AccessPattern::Broadcast;
    }

    // Check sequential.
    let is_sequential = addresses.windows(2).all(|w| w[1].wrapping_sub(w[0]) == element_size);
    if is_sequential {
        return AccessPattern::Sequential;
    }

    // Check constant stride.
    let stride = addresses[1].wrapping_sub(addresses[0]);
    if stride > 0 {
        let is_strided = addresses.windows(2).all(|w| w[1].wrapping_sub(w[0]) == stride);
        if is_strided {
            return AccessPattern::Strided { stride_bytes: stride };
        }
    }

    AccessPattern::Scattered
}

// ══════════════════════════════════════════════════════════════════════
// CUDA kernel source — coalesced copy / transpose helpers
// ══════════════════════════════════════════════════════════════════════

/// CUDA C kernel source for coalesced memory operations.
///
/// Contains a coalesced AoS→SoA transpose kernel and a vectorized 128-bit
/// copy kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const MEMORY_COALESCING_KERNEL_SRC: &str = r#"
// Coalesced AoS→SoA transpose kernel.
// Each block handles one tile of `tile_w` elements.
// `num_elements` — total structures, `num_fields` — fields per structure.
extern "C" __global__ void aos_to_soa_f32(
    const float* __restrict__ aos_in,
    float*       __restrict__ soa_out,
    int num_elements,
    int num_fields)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements) return;

    for (int f = 0; f < num_fields; ++f) {
        soa_out[f * num_elements + gid] = aos_in[gid * num_fields + f];
    }
}

// SoA→AoS transpose kernel.
extern "C" __global__ void soa_to_aos_f32(
    const float* __restrict__ soa_in,
    float*       __restrict__ aos_out,
    int num_elements,
    int num_fields)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements) return;

    for (int f = 0; f < num_fields; ++f) {
        aos_out[gid * num_fields + f] = soa_in[f * num_elements + gid];
    }
}

// Vectorised 128-bit (float4) copy for aligned, coalesced transfers.
extern "C" __global__ void coalesced_copy_f32x4(
    const float4* __restrict__ src,
    float4*       __restrict__ dst,
    int n_float4)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_float4) return;
    dst[gid] = src[gid];
}

// Shared-memory bank-conflict-free transpose (TILE x TILE with +1 padding).
// Transposes a TILE x TILE tile of a larger matrix.
#define TILE 32
extern "C" __global__ void bank_conflict_free_transpose(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int width,
    int height)
{
    __shared__ float tile[TILE][TILE + 1]; // +1 padding avoids bank conflicts

    int xIdx = blockIdx.x * TILE + threadIdx.x;
    int yIdx = blockIdx.y * TILE + threadIdx.y;

    if (xIdx < width && yIdx < height)
        tile[threadIdx.y][threadIdx.x] = in[yIdx * width + xIdx];
    __syncthreads();

    xIdx = blockIdx.y * TILE + threadIdx.x;
    yIdx = blockIdx.x * TILE + threadIdx.y;

    if (xIdx < height && yIdx < width)
        out[yIdx * height + xIdx] = tile[threadIdx.x][threadIdx.y];
}
#undef TILE
"#;

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── AccessPattern ────────────────────────────────────────────────

    #[test]
    fn access_pattern_label_sequential() {
        assert_eq!(AccessPattern::Sequential.label(), "sequential");
    }

    #[test]
    fn access_pattern_label_strided() {
        assert_eq!(AccessPattern::Strided { stride_bytes: 16 }.label(), "strided");
    }

    #[test]
    fn access_pattern_label_broadcast() {
        assert_eq!(AccessPattern::Broadcast.label(), "broadcast");
    }

    #[test]
    fn access_pattern_label_scattered() {
        assert_eq!(AccessPattern::Scattered.label(), "scattered");
    }

    #[test]
    fn access_pattern_label_blocked() {
        assert_eq!(AccessPattern::Blocked { block_size: 4 }.label(), "blocked");
    }

    #[test]
    fn access_pattern_display() {
        assert_eq!(format!("{}", AccessPattern::Sequential), "sequential");
        assert_eq!(format!("{}", AccessPattern::Strided { stride_bytes: 8 }), "strided(8B)");
        assert_eq!(format!("{}", AccessPattern::Broadcast), "broadcast");
        assert_eq!(format!("{}", AccessPattern::Blocked { block_size: 4 }), "blocked(4)");
    }

    // ── MemoryAccessDescriptor ───────────────────────────────────────

    #[test]
    fn descriptor_rejects_zero_element_size() {
        let r = MemoryAccessDescriptor::new(0, 0, AccessPattern::Sequential, 32, false);
        assert!(r.is_err());
    }

    #[test]
    fn descriptor_rejects_zero_active_threads() {
        let r = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 0, false);
        assert!(r.is_err());
    }

    #[test]
    fn descriptor_rejects_excess_active_threads() {
        let r = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 33, false);
        assert!(r.is_err());
    }

    #[test]
    fn descriptor_valid_creation() {
        let d = MemoryAccessDescriptor::new(256, 4, AccessPattern::Sequential, 32, false).unwrap();
        assert_eq!(d.base_address, 256);
        assert_eq!(d.element_size, 4);
        assert_eq!(d.active_threads, 32);
        assert!(!d.is_write);
    }

    #[test]
    fn thread_addresses_sequential() {
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 4, false).unwrap();
        assert_eq!(d.thread_addresses(), vec![0, 4, 8, 12]);
    }

    #[test]
    fn thread_addresses_strided() {
        let d = MemoryAccessDescriptor::new(
            0,
            4,
            AccessPattern::Strided { stride_bytes: 16 },
            4,
            false,
        )
        .unwrap();
        assert_eq!(d.thread_addresses(), vec![0, 16, 32, 48]);
    }

    #[test]
    fn thread_addresses_broadcast() {
        let d = MemoryAccessDescriptor::new(100, 4, AccessPattern::Broadcast, 4, false).unwrap();
        assert_eq!(d.thread_addresses(), vec![100, 100, 100, 100]);
    }

    #[test]
    fn thread_addresses_blocked() {
        let d =
            MemoryAccessDescriptor::new(0, 4, AccessPattern::Blocked { block_size: 2 }, 3, false)
                .unwrap();
        assert_eq!(d.thread_addresses(), vec![0, 8, 16]);
    }

    #[test]
    fn thread_addresses_scattered_distinct() {
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Scattered, 4, false).unwrap();
        let addrs = d.thread_addresses();
        assert_eq!(addrs.len(), 4);
    }

    // ── CoalescingAnalyzer ───────────────────────────────────────────

    #[test]
    fn analyzer_default_cache_line() {
        let a = CoalescingAnalyzer::new();
        assert_eq!(a.cache_line_size, 128);
    }

    #[test]
    fn analyzer_custom_cache_line() {
        let a = CoalescingAnalyzer::with_cache_line(64).unwrap();
        assert_eq!(a.cache_line_size, 64);
    }

    #[test]
    fn analyzer_rejects_non_power_of_two() {
        assert!(CoalescingAnalyzer::with_cache_line(0).is_err());
        assert!(CoalescingAnalyzer::with_cache_line(48).is_err());
    }

    #[test]
    fn sequential_access_high_efficiency() {
        let a = CoalescingAnalyzer::new();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 32, false).unwrap();
        let r = a.analyze(&d);
        assert!(r.efficiency >= 0.9, "eff={}", r.efficiency);
        assert!(r.transactions_128b >= 1);
    }

    #[test]
    fn broadcast_access_single_transaction() {
        let a = CoalescingAnalyzer::new();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Broadcast, 32, false).unwrap();
        let r = a.analyze(&d);
        // Broadcast: all 32 threads read same address → 1 cache line.
        assert_eq!(r.transactions_128b, 1);
        assert!(r.recommendation.contains("Broadcast"));
    }

    #[test]
    fn strided_access_lower_efficiency() {
        let a = CoalescingAnalyzer::new();
        let d = MemoryAccessDescriptor::new(
            0,
            4,
            AccessPattern::Strided { stride_bytes: 64 },
            32,
            false,
        )
        .unwrap();
        let r = a.analyze(&d);
        assert!(r.efficiency < 0.5, "eff={}", r.efficiency);
    }

    #[test]
    fn analyzer_useful_bytes_correct() {
        let a = CoalescingAnalyzer::new();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 16, false).unwrap();
        let r = a.analyze(&d);
        assert_eq!(r.useful_bytes, 64);
    }

    #[test]
    fn analyze_batch_returns_correct_count() {
        let a = CoalescingAnalyzer::new();
        let d1 = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 32, false).unwrap();
        let d2 = MemoryAccessDescriptor::new(0, 4, AccessPattern::Broadcast, 32, false).unwrap();
        let reports = a.analyze_batch(&[d1, d2]);
        assert_eq!(reports.len(), 2);
    }

    #[test]
    fn aggregate_efficiency_empty() {
        assert_eq!(CoalescingAnalyzer::aggregate_efficiency(&[]), 0.0);
    }

    #[test]
    fn aggregate_efficiency_single() {
        let a = CoalescingAnalyzer::new();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 32, false).unwrap();
        let reports = vec![a.analyze(&d)];
        let eff = CoalescingAnalyzer::aggregate_efficiency(&reports);
        assert!(eff > 0.0);
    }

    #[test]
    fn recommendation_sequential_well_coalesced() {
        let a = CoalescingAnalyzer::new();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 32, false).unwrap();
        let r = a.analyze(&d);
        assert!(r.recommendation.contains("well-coalesced"));
    }

    #[test]
    fn recommendation_broadcast() {
        let a = CoalescingAnalyzer::new();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Broadcast, 32, false).unwrap();
        let r = a.analyze(&d);
        assert!(r.recommendation.contains("Broadcast"));
    }

    #[test]
    fn recommendation_scattered() {
        let a = CoalescingAnalyzer::new();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Scattered, 32, false).unwrap();
        let r = a.analyze(&d);
        assert!(r.recommendation.contains("Scattered"));
    }

    #[test]
    fn transactions_32b_gte_128b() {
        let a = CoalescingAnalyzer::new();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 32, false).unwrap();
        let r = a.analyze(&d);
        assert!(r.transactions_32b >= r.transactions_128b);
    }

    // ── WarpTransactionAnalyzer ──────────────────────────────────────

    #[test]
    fn warp_txn_l1_size() {
        let w = WarpTransactionAnalyzer::l1();
        assert_eq!(w.transaction_size, 128);
    }

    #[test]
    fn warp_txn_l2_size() {
        let w = WarpTransactionAnalyzer::l2_sector();
        assert_eq!(w.transaction_size, 32);
    }

    #[test]
    fn warp_txn_custom_rejects_non_power_of_two() {
        assert!(WarpTransactionAnalyzer::new(0).is_err());
        assert!(WarpTransactionAnalyzer::new(48).is_err());
    }

    #[test]
    fn warp_txn_sequential_one_line() {
        let w = WarpTransactionAnalyzer::l1();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 32, false).unwrap();
        let txns = w.decompose(&d);
        // 32 threads × 4 bytes = 128 bytes → 1 transaction.
        assert_eq!(txns.len(), 1);
        assert_eq!(txns[0].size, 128);
    }

    #[test]
    fn warp_txn_strided_more_transactions() {
        let w = WarpTransactionAnalyzer::l1();
        let d = MemoryAccessDescriptor::new(
            0,
            4,
            AccessPattern::Strided { stride_bytes: 128 },
            32,
            false,
        )
        .unwrap();
        let txns = w.decompose(&d);
        assert!(txns.len() > 1, "txns={}", txns.len());
    }

    #[test]
    fn warp_txn_broadcast_single_transaction() {
        let w = WarpTransactionAnalyzer::l1();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Broadcast, 32, false).unwrap();
        let txns = w.decompose(&d);
        assert_eq!(txns.len(), 1);
    }

    #[test]
    fn warp_txn_transaction_count() {
        let w = WarpTransactionAnalyzer::l1();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 32, false).unwrap();
        assert_eq!(w.transaction_count(&d), 1);
    }

    #[test]
    fn warp_txn_ideal_transactions() {
        let w = WarpTransactionAnalyzer::l1();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 32, false).unwrap();
        assert_eq!(w.ideal_transactions(&d), 1);
    }

    #[test]
    fn warp_txn_efficiency_perfect() {
        let w = WarpTransactionAnalyzer::l1();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 32, false).unwrap();
        let eff = w.transaction_efficiency(&d);
        assert!((eff - 1.0).abs() < 1e-9, "eff={}", eff);
    }

    #[test]
    fn warp_txn_efficiency_strided_less_than_one() {
        let w = WarpTransactionAnalyzer::l1();
        let d = MemoryAccessDescriptor::new(
            0,
            4,
            AccessPattern::Strided { stride_bytes: 128 },
            32,
            false,
        )
        .unwrap();
        let eff = w.transaction_efficiency(&d);
        assert!(eff < 1.0, "eff={}", eff);
    }

    #[test]
    fn warp_txn_thread_ids_populated() {
        let w = WarpTransactionAnalyzer::l1();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 32, false).unwrap();
        let txns = w.decompose(&d);
        let total_threads: usize = txns.iter().map(|t| t.thread_ids.len()).sum();
        assert_eq!(total_threads, 32);
    }

    #[test]
    fn warp_txn_l2_more_sectors_than_l1_lines() {
        let l1 = WarpTransactionAnalyzer::l1();
        let l2 = WarpTransactionAnalyzer::l2_sector();
        let d = MemoryAccessDescriptor::new(
            0,
            4,
            AccessPattern::Strided { stride_bytes: 32 },
            32,
            false,
        )
        .unwrap();
        assert!(l2.transaction_count(&d) >= l1.transaction_count(&d));
    }

    // ── BankConflictDetector ─────────────────────────────────────────

    #[test]
    fn bank_detector_defaults() {
        let d = BankConflictDetector::new();
        assert_eq!(d.num_banks, 32);
        assert_eq!(d.bank_width, 4);
    }

    #[test]
    fn bank_detector_custom() {
        let d = BankConflictDetector::with_config(16, 8).unwrap();
        assert_eq!(d.num_banks, 16);
        assert_eq!(d.bank_width, 8);
    }

    #[test]
    fn bank_detector_rejects_zero_banks() {
        assert!(BankConflictDetector::with_config(0, 4).is_err());
    }

    #[test]
    fn bank_detector_rejects_zero_width() {
        assert!(BankConflictDetector::with_config(32, 0).is_err());
    }

    #[test]
    fn bank_for_address_sequential() {
        let d = BankConflictDetector::new();
        assert_eq!(d.bank_for_address(0), 0);
        assert_eq!(d.bank_for_address(4), 1);
        assert_eq!(d.bank_for_address(128), 0); // 128 / 4 = 32, 32 % 32 = 0
    }

    #[test]
    fn no_bank_conflicts_sequential() {
        let d = BankConflictDetector::new();
        let addrs: Vec<usize> = (0..32).map(|i| i * 4).collect();
        let r = d.analyze(&addrs);
        assert!(r.is_conflict_free);
        assert_eq!(r.conflict_count, 0);
        assert_eq!(r.max_conflict_degree, 1);
    }

    #[test]
    fn bank_conflicts_same_bank() {
        let d = BankConflictDetector::new();
        // All access bank 0 (stride = 128 bytes = 32 banks × 4 bytes).
        let addrs: Vec<usize> = (0..4).map(|i| i * 128).collect();
        let r = d.analyze(&addrs);
        assert!(!r.is_conflict_free);
        assert_eq!(r.conflict_count, 3);
        assert_eq!(r.max_conflict_degree, 4);
    }

    #[test]
    fn bank_conflict_suggested_padding_nonzero() {
        let d = BankConflictDetector::new();
        let addrs: Vec<usize> = (0..4).map(|i| i * 128).collect();
        let r = d.analyze(&addrs);
        assert!(r.suggested_padding > 0);
    }

    #[test]
    fn bank_conflict_2d_no_conflict() {
        let d = BankConflictDetector::new();
        // 32 cols × 4 bytes = 128 B stride → bank 0 for every row.
        // With different element sizes we can avoid it. 33 cols avoids it.
        let r = d.analyze_2d(32, 33, 4, 33 * 4);
        assert!(r.is_conflict_free);
    }

    #[test]
    fn bank_conflict_2d_with_conflict() {
        let d = BankConflictDetector::new();
        // 32 cols, stride = 32*4=128 → every row maps to bank 0.
        let r = d.analyze_2d(32, 32, 4, 128);
        assert!(!r.is_conflict_free);
        assert!(r.suggested_padding > 0);
    }

    #[test]
    fn bank_conflict_per_bank_map() {
        let d = BankConflictDetector::new();
        let addrs = vec![0, 128, 4, 132]; // bank 0: 0,128; bank 1: 4,132
        let r = d.analyze(&addrs);
        assert_eq!(r.per_bank_conflicts.len(), 2);
    }

    // ── LayoutTransformer (AoS ↔ SoA) ───────────────────────────────

    #[test]
    fn aos_to_soa_basic() {
        // 2 elements, 3 fields: [a0,b0,c0, a1,b1,c1]
        let aos = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let soa = LayoutTransformer::aos_to_soa(&aos, 2, 3).unwrap();
        assert_eq!(soa, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn soa_to_aos_basic() {
        let soa = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let aos = LayoutTransformer::soa_to_aos(&soa, 2, 3).unwrap();
        assert_eq!(aos, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn aos_soa_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let soa = LayoutTransformer::aos_to_soa(&original, 4, 2).unwrap();
        let back = LayoutTransformer::soa_to_aos(&soa, 4, 2).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn aos_to_soa_single_field() {
        let data = vec![10.0, 20.0, 30.0];
        let soa = LayoutTransformer::aos_to_soa(&data, 3, 1).unwrap();
        assert_eq!(soa, data);
    }

    #[test]
    fn aos_to_soa_single_element() {
        let data = vec![1.0, 2.0, 3.0];
        let soa = LayoutTransformer::aos_to_soa(&data, 1, 3).unwrap();
        assert_eq!(soa, data);
    }

    #[test]
    fn aos_to_soa_wrong_length() {
        let data = vec![1.0, 2.0];
        assert!(LayoutTransformer::aos_to_soa(&data, 3, 1).is_err());
    }

    #[test]
    fn soa_to_aos_wrong_length() {
        let data = [1.0];
        assert!(LayoutTransformer::soa_to_aos(&data, 3, 2).is_err());
    }

    #[test]
    fn aos_to_aosoa_basic() {
        // 4 elems, 2 fields, tile=2 → tiles: [e0,e1] [e2,e3]
        let aos = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let aosoa = LayoutTransformer::aos_to_aosoa(&aos, 4, 2, 2).unwrap();
        // tile0: field0=[1,3], field1=[2,4] → [1,3,2,4]
        // tile1: field0=[5,7], field1=[6,8] → [5,7,6,8]
        assert_eq!(aosoa, vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);
    }

    #[test]
    fn aos_to_aosoa_zero_tile_width_err() {
        let data = vec![1.0, 2.0];
        assert!(LayoutTransformer::aos_to_aosoa(&data, 1, 2, 0).is_err());
    }

    #[test]
    fn aos_to_aosoa_wrong_length_err() {
        let data = [1.0];
        assert!(LayoutTransformer::aos_to_aosoa(&data, 2, 2, 2).is_err());
    }

    #[test]
    fn estimate_improvement_soa_better() {
        let (eff_aos, eff_soa) = LayoutTransformer::estimate_improvement(1024, 8, 4);
        assert!(eff_soa >= eff_aos, "SoA ({}) should be >= AoS ({})", eff_soa, eff_aos);
    }

    #[test]
    fn estimate_improvement_single_field() {
        let (eff_aos, eff_soa) = LayoutTransformer::estimate_improvement(1024, 1, 4);
        // With a single field AoS == SoA.
        assert!((eff_aos - eff_soa).abs() < 1e-6);
    }

    // ── Padding ──────────────────────────────────────────────────────

    #[test]
    fn padding_none() {
        assert_eq!(compute_padded_stride(32, 4, PaddingStrategy::None), 128);
    }

    #[test]
    fn padding_row_alignment_128() {
        let s = compute_padded_stride(30, 4, PaddingStrategy::RowAlignment { alignment: 128 });
        assert_eq!(s, 128);
    }

    #[test]
    fn padding_row_alignment_256() {
        let s = compute_padded_stride(33, 4, PaddingStrategy::RowAlignment { alignment: 256 });
        assert_eq!(s, 256);
    }

    #[test]
    fn padding_bank_conflict_avoidance() {
        let s =
            compute_padded_stride(32, 4, PaddingStrategy::BankConflictAvoidance { extra_bytes: 4 });
        assert_eq!(s, 132);
    }

    #[test]
    fn padding_power_of_two() {
        let s = compute_padded_stride(33, 4, PaddingStrategy::PowerOfTwo);
        // 33 * 4 = 132 → next power of two = 256
        assert_eq!(s, 256);
    }

    #[test]
    fn padding_power_of_two_exact() {
        let s = compute_padded_stride(32, 4, PaddingStrategy::PowerOfTwo);
        assert_eq!(s, 128);
    }

    #[test]
    fn pad_2d_buffer_none() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let (padded, stride) = pad_2d_buffer(&data, 2, 2, PaddingStrategy::None).unwrap();
        assert_eq!(stride, 8);
        assert_eq!(padded, data);
    }

    #[test]
    fn pad_2d_buffer_alignment() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (padded, stride) =
            pad_2d_buffer(&data, 2, 3, PaddingStrategy::RowAlignment { alignment: 16 }).unwrap();
        assert_eq!(stride, 16);
        let cols = stride / 4;
        assert_eq!(padded[0], 1.0);
        assert_eq!(padded[cols], 4.0);
    }

    #[test]
    fn pad_2d_buffer_wrong_length() {
        assert!(pad_2d_buffer(&[1.0], 2, 2, PaddingStrategy::None).is_err());
    }

    // ── AccessReorderAdvisor ─────────────────────────────────────────

    #[test]
    fn advisor_sequential_aligned_no_critical() {
        let advisor = AccessReorderAdvisor::new();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 32, false).unwrap();
        let recs = advisor.advise(&d);
        assert!(recs.iter().all(|r| r.severity != Severity::Critical));
    }

    #[test]
    fn advisor_scattered_has_critical() {
        let advisor = AccessReorderAdvisor::new();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Scattered, 32, false).unwrap();
        let recs = advisor.advise(&d);
        assert!(recs.iter().any(|r| r.severity == Severity::Critical));
    }

    #[test]
    fn advisor_misaligned_base() {
        let advisor = AccessReorderAdvisor::new();
        let d = MemoryAccessDescriptor::new(3, 4, AccessPattern::Sequential, 32, false).unwrap();
        let recs = advisor.advise(&d);
        assert!(recs.iter().any(|r| r.title.contains("Misaligned")));
    }

    #[test]
    fn advisor_strided_large_stride() {
        let advisor = AccessReorderAdvisor::new();
        let d = MemoryAccessDescriptor::new(
            0,
            4,
            AccessPattern::Strided { stride_bytes: 512 },
            32,
            false,
        )
        .unwrap();
        let recs = advisor.advise(&d);
        assert!(recs.iter().any(|r| r.title.contains("Strided")));
    }

    #[test]
    fn advisor_low_utilization() {
        let advisor = AccessReorderAdvisor::new();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 4, false).unwrap();
        let recs = advisor.advise(&d);
        assert!(recs.iter().any(|r| r.title.contains("Low warp")));
    }

    #[test]
    fn advisor_batch_deduplicates() {
        let advisor = AccessReorderAdvisor::new();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Scattered, 32, false).unwrap();
        let recs = advisor.advise_batch(&[d.clone(), d]);
        let titles: Vec<_> = recs.iter().map(|r| &r.title).collect();
        let unique: std::collections::HashSet<_> = titles.iter().collect();
        assert_eq!(titles.len(), unique.len(), "duplicates found");
    }

    #[test]
    fn advisor_batch_sorted_by_severity() {
        let advisor = AccessReorderAdvisor::new();
        let d1 = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 4, false).unwrap();
        let d2 = MemoryAccessDescriptor::new(0, 4, AccessPattern::Scattered, 32, false).unwrap();
        let recs = advisor.advise_batch(&[d1, d2]);
        for w in recs.windows(2) {
            assert!(w[0].severity >= w[1].severity);
        }
    }

    #[test]
    fn advisor_blocked_pattern() {
        let advisor = AccessReorderAdvisor::new();
        let d =
            MemoryAccessDescriptor::new(0, 4, AccessPattern::Blocked { block_size: 4 }, 32, false)
                .unwrap();
        let recs = advisor.advise(&d);
        assert!(recs.iter().any(|r| r.title.contains("Blocked")));
    }

    // ── detect_pattern ───────────────────────────────────────────────

    #[test]
    fn detect_empty() {
        assert_eq!(detect_pattern(&[], 4), AccessPattern::Sequential);
    }

    #[test]
    fn detect_single_address() {
        assert_eq!(detect_pattern(&[100], 4), AccessPattern::Sequential);
    }

    #[test]
    fn detect_broadcast_pattern() {
        assert_eq!(detect_pattern(&[8, 8, 8, 8], 4), AccessPattern::Broadcast);
    }

    #[test]
    fn detect_sequential_pattern() {
        assert_eq!(detect_pattern(&[0, 4, 8, 12], 4), AccessPattern::Sequential);
    }

    #[test]
    fn detect_strided_pattern() {
        assert_eq!(
            detect_pattern(&[0, 16, 32, 48], 4),
            AccessPattern::Strided { stride_bytes: 16 }
        );
    }

    #[test]
    fn detect_scattered_pattern() {
        assert_eq!(detect_pattern(&[0, 100, 7, 999], 4), AccessPattern::Scattered);
    }

    // ── DataLayout Display ───────────────────────────────────────────

    #[test]
    fn data_layout_display_aos() {
        assert_eq!(format!("{}", DataLayout::AoS), "AoS");
    }

    #[test]
    fn data_layout_display_soa() {
        assert_eq!(format!("{}", DataLayout::SoA), "SoA");
    }

    #[test]
    fn data_layout_display_aosoa() {
        assert_eq!(format!("{}", DataLayout::AoSoA { tile_width: 8 }), "AoSoA(8)");
    }

    // ── Severity Display ─────────────────────────────────────────────

    #[test]
    fn severity_display() {
        assert_eq!(format!("{}", Severity::Info), "info");
        assert_eq!(format!("{}", Severity::Critical), "critical");
    }

    #[test]
    fn severity_ordering() {
        assert!(Severity::Critical > Severity::High);
        assert!(Severity::High > Severity::Medium);
        assert!(Severity::Medium > Severity::Low);
        assert!(Severity::Low > Severity::Info);
    }

    // ── Constants ────────────────────────────────────────────────────

    #[test]
    fn warp_size_is_32() {
        assert_eq!(WARP_SIZE, 32);
    }

    #[test]
    fn cache_line_is_128() {
        assert_eq!(CACHE_LINE_BYTES, 128);
    }

    #[test]
    fn shared_mem_banks_is_32() {
        assert_eq!(NUM_SHARED_MEM_BANKS, 32);
    }

    #[test]
    fn shared_mem_bank_width_is_4() {
        assert_eq!(SHARED_MEM_BANK_WIDTH, 4);
    }

    // ── Edge cases / integration ─────────────────────────────────────

    #[test]
    fn single_thread_sequential() {
        let a = CoalescingAnalyzer::new();
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 1, false).unwrap();
        let r = a.analyze(&d);
        assert!(r.efficiency > 0.0);
        assert_eq!(r.useful_bytes, 4);
    }

    #[test]
    fn large_element_size() {
        let a = CoalescingAnalyzer::new();
        let d = MemoryAccessDescriptor::new(0, 64, AccessPattern::Sequential, 32, false).unwrap();
        let r = a.analyze(&d);
        assert_eq!(r.useful_bytes, 32 * 64);
    }

    #[test]
    fn misaligned_base_still_works() {
        let a = CoalescingAnalyzer::new();
        let d = MemoryAccessDescriptor::new(7, 4, AccessPattern::Sequential, 32, false).unwrap();
        let r = a.analyze(&d);
        assert!(r.efficiency > 0.0);
        assert!(r.transactions_128b >= 1);
    }

    #[test]
    fn write_descriptor() {
        let d = MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 32, true).unwrap();
        assert!(d.is_write);
    }

    #[test]
    fn strided_stride_equals_element_size_is_sequential_efficiency() {
        let a = CoalescingAnalyzer::new();
        let d_seq =
            MemoryAccessDescriptor::new(0, 4, AccessPattern::Sequential, 32, false).unwrap();
        let d_str = MemoryAccessDescriptor::new(
            0,
            4,
            AccessPattern::Strided { stride_bytes: 4 },
            32,
            false,
        )
        .unwrap();
        let r_seq = a.analyze(&d_seq);
        let r_str = a.analyze(&d_str);
        assert!((r_seq.efficiency - r_str.efficiency).abs() < 1e-9);
    }

    #[test]
    fn padded_stride_alignment_zero_fallback() {
        // alignment 0 → no-op
        let s = compute_padded_stride(32, 4, PaddingStrategy::RowAlignment { alignment: 0 });
        assert_eq!(s, 128);
    }

    #[test]
    fn padded_stride_alignment_non_power_of_two_fallback() {
        let s = compute_padded_stride(32, 4, PaddingStrategy::RowAlignment { alignment: 3 });
        assert_eq!(s, 128);
    }

    // ── CUDA kernel source available under gpu feature ───────────────

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn kernel_src_not_empty() {
        assert!(!MEMORY_COALESCING_KERNEL_SRC.is_empty());
    }
}
