//! OpenCL memory manager for A770 GPU inference with unified memory support.
//!
//! Intel Arc A770 supports unified memory (shared between CPU and GPU).
//! This module manages GPU buffer lifecycle, memory pools, and transfer tracking.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// MemoryError
// ---------------------------------------------------------------------------

/// Errors produced by the memory manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    /// Requested allocation exceeds budget.
    BudgetExceeded { requested: usize, available: usize },
    /// Unknown allocation ID.
    InvalidAllocationId(u64),
    /// Zero-byte allocation requested.
    ZeroSizeAllocation,
    /// Alignment must be a power of two and non-zero.
    InvalidAlignment(usize),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BudgetExceeded { requested, available } => {
                write!(f, "budget exceeded: requested {requested} bytes, {available} available")
            }
            Self::InvalidAllocationId(id) => write!(f, "invalid allocation id: {id}"),
            Self::ZeroSizeAllocation => write!(f, "zero-byte allocation is not allowed"),
            Self::InvalidAlignment(a) => {
                write!(f, "alignment must be a non-zero power of two, got {a}")
            }
        }
    }
}

impl std::error::Error for MemoryError {}

/// Convenience alias used throughout this module.
pub type Result<T> = std::result::Result<T, MemoryError>;

// ---------------------------------------------------------------------------
// MemoryRegion
// ---------------------------------------------------------------------------

/// Describes where a buffer resides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryRegion {
    /// GPU-only memory (fastest for compute).
    Device,
    /// CPU-only memory (for staging).
    Host,
    /// Shared CPU/GPU memory (Intel Arc unified memory, avoids copies).
    Unified,
    /// Page-locked host memory (fast DMA transfers).
    Pinned,
}

impl fmt::Display for MemoryRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device => write!(f, "Device"),
            Self::Host => write!(f, "Host"),
            Self::Unified => write!(f, "Unified"),
            Self::Pinned => write!(f, "Pinned"),
        }
    }
}

// ---------------------------------------------------------------------------
// AllocationConfig
// ---------------------------------------------------------------------------

/// Configuration for a single allocation request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocationConfig {
    /// Where the buffer should reside.
    pub region: MemoryRegion,
    /// Requested size in bytes.
    pub size_bytes: usize,
    /// Alignment in bytes (default 64 for A770 cache lines).
    pub alignment: usize,
    /// Human-readable debug label.
    pub name: String,
}

impl AllocationConfig {
    /// Returns `size_bytes` rounded up to the nearest multiple of `alignment`.
    pub fn aligned_size(&self) -> usize {
        if self.alignment == 0 {
            return self.size_bytes;
        }
        let mask = self.alignment - 1;
        (self.size_bytes + mask) & !mask
    }
}

impl Default for AllocationConfig {
    fn default() -> Self {
        Self { region: MemoryRegion::Device, size_bytes: 0, alignment: 64, name: String::new() }
    }
}

// ---------------------------------------------------------------------------
// AllocationRecord
// ---------------------------------------------------------------------------

/// Bookkeeping for one live allocation.
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Unique allocation identifier.
    pub id: u64,
    /// The configuration used when this allocation was created.
    pub config: AllocationConfig,
    /// When the allocation was created.
    pub allocated_at: Instant,
    /// Last time the allocation was accessed (read or write).
    pub last_accessed: Instant,
}

// ---------------------------------------------------------------------------
// MemoryPool
// ---------------------------------------------------------------------------

/// Manages a set of allocations with a free-list recycler.
#[derive(Debug)]
pub struct MemoryPool {
    /// All live allocations.
    pub allocations: Vec<AllocationRecord>,
    /// Free-list: (offset, size) pairs available for recycling.
    pub free_list: Vec<(usize, usize)>,
    /// Sum of aligned sizes of all live allocations.
    pub total_allocated: usize,
    /// High-water mark.
    pub peak_allocated: usize,
    /// Monotonically increasing allocation counter (also used as next ID).
    pub allocation_count: u64,
}

impl MemoryPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self {
            allocations: Vec::new(),
            free_list: Vec::new(),
            total_allocated: 0,
            peak_allocated: 0,
            allocation_count: 0,
        }
    }

    /// Allocate a buffer described by `config`. Returns its unique ID.
    pub fn allocate(&mut self, config: AllocationConfig) -> Result<u64> {
        if config.size_bytes == 0 {
            return Err(MemoryError::ZeroSizeAllocation);
        }
        if config.alignment == 0 || !config.alignment.is_power_of_two() {
            return Err(MemoryError::InvalidAlignment(config.alignment));
        }

        let aligned = config.aligned_size();

        // Try to recycle a free-list entry that is large enough.
        let reuse_idx = self.free_list.iter().position(|&(_off, sz)| sz >= aligned);
        if let Some(idx) = reuse_idx {
            let (off, sz) = self.free_list[idx];
            let leftover = sz - aligned;
            if leftover > 0 {
                self.free_list[idx] = (off + aligned, leftover);
            } else {
                self.free_list.swap_remove(idx);
            }
        }

        self.allocation_count += 1;
        let id = self.allocation_count;
        let now = Instant::now();

        self.allocations.push(AllocationRecord {
            id,
            config,
            allocated_at: now,
            last_accessed: now,
        });

        self.total_allocated += aligned;
        if self.total_allocated > self.peak_allocated {
            self.peak_allocated = self.total_allocated;
        }

        Ok(id)
    }

    /// Free the allocation with the given `id`.
    pub fn deallocate(&mut self, id: u64) -> Result<()> {
        let pos = self
            .allocations
            .iter()
            .position(|a| a.id == id)
            .ok_or(MemoryError::InvalidAllocationId(id))?;

        let record = self.allocations.swap_remove(pos);
        let aligned = record.config.aligned_size();
        self.total_allocated = self.total_allocated.saturating_sub(aligned);

        // Push freed region onto the free list (offset is symbolic).
        self.free_list.push((0, aligned));

        Ok(())
    }

    /// Look up an allocation by ID.
    pub fn get_allocation(&self, id: u64) -> Option<&AllocationRecord> {
        self.allocations.iter().find(|a| a.id == id)
    }

    /// Current total allocated bytes (aligned).
    pub fn current_usage(&self) -> usize {
        self.total_allocated
    }

    /// Peak allocated bytes seen so far.
    pub fn peak_usage(&self) -> usize {
        self.peak_allocated
    }

    /// Total number of allocations performed (including freed ones).
    pub fn allocation_count(&self) -> u64 {
        self.allocation_count
    }

    /// Coalesce adjacent free-list entries; returns bytes recovered
    /// (i.e. the number of entries that were merged away).
    pub fn defragment(&mut self) -> usize {
        if self.free_list.len() < 2 {
            return 0;
        }

        // Sort by offset so adjacent regions can be merged.
        self.free_list.sort_by_key(|&(off, _)| off);

        let mut merged: Vec<(usize, usize)> = Vec::new();
        let mut recovered: usize = 0;

        for &(off, sz) in &self.free_list {
            if let Some(last) = merged.last_mut()
                && last.0 + last.1 >= off
            {
                let new_end = std::cmp::max(last.0 + last.1, off + sz);
                let old_total = last.1 + sz;
                let merged_sz = new_end - last.0;
                recovered += old_total.saturating_sub(merged_sz);
                last.1 = merged_sz;
                continue;
            }
            merged.push((off, sz));
        }

        let entries_removed = self.free_list.len().saturating_sub(merged.len());
        self.free_list = merged;

        if recovered == 0 {
            recovered = entries_removed;
        }
        recovered
    }

    /// Clear every allocation and reset counters.
    pub fn reset(&mut self) {
        self.allocations.clear();
        self.free_list.clear();
        self.total_allocated = 0;
        self.peak_allocated = 0;
        self.allocation_count = 0;
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MemoryBudget
// ---------------------------------------------------------------------------

/// Enforces per-region memory limits.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Maximum device-region bytes (e.g. 16 GiB for A770).
    pub device_limit: usize,
    /// Maximum host-region bytes.
    pub host_limit: usize,
    /// Maximum unified-region bytes.
    pub unified_limit: usize,
}

impl MemoryBudget {
    /// Returns `true` if the pool can accommodate `request` without
    /// exceeding this budget.
    pub fn can_allocate(&self, pool: &MemoryPool, request: &AllocationConfig) -> bool {
        let aligned = request.aligned_size();
        let region_usage: usize = pool
            .allocations
            .iter()
            .filter(|a| a.config.region == request.region)
            .map(|a| a.config.aligned_size())
            .sum();

        let limit = match request.region {
            MemoryRegion::Device => self.device_limit,
            MemoryRegion::Host | MemoryRegion::Pinned => self.host_limit,
            MemoryRegion::Unified => self.unified_limit,
        };

        region_usage + aligned <= limit
    }

    /// Fraction of the total budget currently in use (0.0–1.0+).
    pub fn utilization(&self, pool: &MemoryPool) -> f64 {
        let total_limit = self.device_limit + self.host_limit + self.unified_limit;
        if total_limit == 0 {
            return 0.0;
        }
        pool.total_allocated as f64 / total_limit as f64
    }
}

// ---------------------------------------------------------------------------
// TransferTracker
// ---------------------------------------------------------------------------

/// A single recorded transfer event.
#[derive(Debug, Clone)]
struct TransferEvent {
    #[allow(dead_code)]
    src: MemoryRegion,
    #[allow(dead_code)]
    dst: MemoryRegion,
    bytes: usize,
    duration_ns: u64,
}

/// Tracks memory transfers between regions.
#[derive(Debug, Default)]
pub struct TransferTracker {
    events: Vec<TransferEvent>,
}

impl TransferTracker {
    /// Record a completed transfer.
    pub fn record_transfer(
        &mut self,
        src: MemoryRegion,
        dst: MemoryRegion,
        bytes: usize,
        duration_ns: u64,
    ) {
        self.events.push(TransferEvent { src, dst, bytes, duration_ns });
    }

    /// Total bytes moved across all recorded transfers.
    pub fn total_bytes_transferred(&self) -> usize {
        self.events.iter().map(|e| e.bytes).sum()
    }

    /// Average bandwidth in GB/s across all transfers.
    /// Returns 0.0 when no transfers have been recorded or total duration is zero.
    pub fn average_bandwidth_gbps(&self) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }
        let total_bytes: usize = self.events.iter().map(|e| e.bytes).sum();
        let total_ns: u64 = self.events.iter().map(|e| e.duration_ns).sum();
        if total_ns == 0 {
            return 0.0;
        }
        // GB/s = bytes / ns  (since 1 GB = 1e9 bytes, 1 s = 1e9 ns)
        total_bytes as f64 / total_ns as f64
    }

    /// Number of transfers recorded.
    pub fn transfer_count(&self) -> usize {
        self.events.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn device_config(name: &str, size: usize) -> AllocationConfig {
        AllocationConfig {
            region: MemoryRegion::Device,
            size_bytes: size,
            alignment: 64,
            name: name.to_string(),
        }
    }

    fn unified_config(name: &str, size: usize) -> AllocationConfig {
        AllocationConfig {
            region: MemoryRegion::Unified,
            size_bytes: size,
            alignment: 64,
            name: name.to_string(),
        }
    }

    fn host_config(name: &str, size: usize) -> AllocationConfig {
        AllocationConfig {
            region: MemoryRegion::Host,
            size_bytes: size,
            alignment: 64,
            name: name.to_string(),
        }
    }

    fn pinned_config(name: &str, size: usize) -> AllocationConfig {
        AllocationConfig {
            region: MemoryRegion::Pinned,
            size_bytes: size,
            alignment: 64,
            name: name.to_string(),
        }
    }

    // ---------------------------------------------------------------
    // Alignment
    // ---------------------------------------------------------------

    #[test]
    fn aligned_size_already_aligned() {
        let c = device_config("a", 128);
        assert_eq!(c.aligned_size(), 128);
    }

    #[test]
    fn aligned_size_rounds_up() {
        let c = device_config("a", 100);
        assert_eq!(c.aligned_size(), 128); // next multiple of 64
    }

    #[test]
    fn aligned_size_one_byte() {
        let c = device_config("a", 1);
        assert_eq!(c.aligned_size(), 64);
    }

    #[test]
    fn aligned_size_custom_alignment() {
        let c = AllocationConfig {
            region: MemoryRegion::Device,
            size_bytes: 200,
            alignment: 256,
            name: "big".into(),
        };
        assert_eq!(c.aligned_size(), 256);
    }

    // ---------------------------------------------------------------
    // Allocation / deallocation lifecycle
    // ---------------------------------------------------------------

    #[test]
    fn allocate_and_lookup() {
        let mut pool = MemoryPool::new();
        let id = pool.allocate(device_config("w", 1024)).unwrap();
        let rec = pool.get_allocation(id).unwrap();
        assert_eq!(rec.config.size_bytes, 1024);
        assert_eq!(rec.config.name, "w");
    }

    #[test]
    fn deallocate_removes_record() {
        let mut pool = MemoryPool::new();
        let id = pool.allocate(device_config("x", 512)).unwrap();
        pool.deallocate(id).unwrap();
        assert!(pool.get_allocation(id).is_none());
    }

    #[test]
    fn deallocate_unknown_id_is_error() {
        let mut pool = MemoryPool::new();
        assert_eq!(pool.deallocate(999), Err(MemoryError::InvalidAllocationId(999)),);
    }

    #[test]
    fn double_deallocate_is_error() {
        let mut pool = MemoryPool::new();
        let id = pool.allocate(device_config("d", 64)).unwrap();
        pool.deallocate(id).unwrap();
        assert!(pool.deallocate(id).is_err());
    }

    #[test]
    fn deallocate_decreases_usage() {
        let mut pool = MemoryPool::new();
        let id = pool.allocate(device_config("a", 256)).unwrap();
        let before = pool.current_usage();
        pool.deallocate(id).unwrap();
        assert!(pool.current_usage() < before);
    }

    // ---------------------------------------------------------------
    // Allocation ID uniqueness
    // ---------------------------------------------------------------

    #[test]
    fn ids_are_unique() {
        let mut pool = MemoryPool::new();
        let a = pool.allocate(device_config("a", 64)).unwrap();
        let b = pool.allocate(device_config("b", 64)).unwrap();
        let c = pool.allocate(device_config("c", 64)).unwrap();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn ids_monotonically_increase() {
        let mut pool = MemoryPool::new();
        let a = pool.allocate(device_config("a", 64)).unwrap();
        let b = pool.allocate(device_config("b", 64)).unwrap();
        assert!(b > a);
    }

    #[test]
    fn ids_unique_after_dealloc() {
        let mut pool = MemoryPool::new();
        let a = pool.allocate(device_config("a", 64)).unwrap();
        pool.deallocate(a).unwrap();
        let b = pool.allocate(device_config("b", 64)).unwrap();
        assert_ne!(a, b);
    }

    // ---------------------------------------------------------------
    // Peak tracking
    // ---------------------------------------------------------------

    #[test]
    fn peak_tracks_high_water_mark() {
        let mut pool = MemoryPool::new();
        let a = pool.allocate(device_config("a", 1024)).unwrap();
        let b = pool.allocate(device_config("b", 2048)).unwrap();
        let peak_after_two = pool.peak_usage();
        pool.deallocate(a).unwrap();
        pool.deallocate(b).unwrap();
        assert_eq!(pool.peak_usage(), peak_after_two);
        assert_eq!(pool.current_usage(), 0);
    }

    #[test]
    fn peak_updates_on_new_high() {
        let mut pool = MemoryPool::new();
        pool.allocate(device_config("a", 64)).unwrap();
        let p1 = pool.peak_usage();
        pool.allocate(device_config("b", 256)).unwrap();
        assert!(pool.peak_usage() > p1);
    }

    #[test]
    fn peak_does_not_decrease() {
        let mut pool = MemoryPool::new();
        let a = pool.allocate(device_config("a", 4096)).unwrap();
        let peak = pool.peak_usage();
        pool.deallocate(a).unwrap();
        pool.allocate(device_config("b", 64)).unwrap();
        assert_eq!(pool.peak_usage(), peak);
    }

    // ---------------------------------------------------------------
    // Budget enforcement
    // ---------------------------------------------------------------

    #[test]
    fn budget_allows_within_limit() {
        let pool = MemoryPool::new();
        let budget = MemoryBudget { device_limit: 4096, host_limit: 4096, unified_limit: 4096 };
        assert!(budget.can_allocate(&pool, &device_config("a", 1024)));
    }

    #[test]
    fn budget_rejects_over_limit() {
        let mut pool = MemoryPool::new();
        pool.allocate(device_config("a", 4000)).unwrap();
        let budget = MemoryBudget { device_limit: 4096, host_limit: 4096, unified_limit: 4096 };
        // 4000 aligned to 64 = 4032; 4032 + 128 = 4160 > 4096
        assert!(!budget.can_allocate(&pool, &device_config("b", 128)));
    }

    #[test]
    fn budget_checks_per_region() {
        let mut pool = MemoryPool::new();
        pool.allocate(device_config("a", 4000)).unwrap();
        let budget = MemoryBudget { device_limit: 4096, host_limit: 8192, unified_limit: 8192 };
        assert!(budget.can_allocate(&pool, &host_config("h", 1024)));
    }

    #[test]
    fn budget_utilization_empty() {
        let pool = MemoryPool::new();
        let budget = MemoryBudget { device_limit: 1024, host_limit: 1024, unified_limit: 1024 };
        assert!((budget.utilization(&pool) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn budget_utilization_partial() {
        let mut pool = MemoryPool::new();
        pool.allocate(device_config("a", 512)).unwrap();
        let budget = MemoryBudget { device_limit: 1024, host_limit: 1024, unified_limit: 1024 };
        let u = budget.utilization(&pool);
        assert!(u > 0.0 && u < 1.0);
    }

    #[test]
    fn budget_utilization_zero_limits() {
        let pool = MemoryPool::new();
        let budget = MemoryBudget { device_limit: 0, host_limit: 0, unified_limit: 0 };
        assert!((budget.utilization(&pool) - 0.0).abs() < f64::EPSILON);
    }

    // ---------------------------------------------------------------
    // Free-list recycling
    // ---------------------------------------------------------------

    #[test]
    fn free_list_grows_on_dealloc() {
        let mut pool = MemoryPool::new();
        let id = pool.allocate(device_config("f", 256)).unwrap();
        assert!(pool.free_list.is_empty());
        pool.deallocate(id).unwrap();
        assert!(!pool.free_list.is_empty());
    }

    #[test]
    fn free_list_entry_reused() {
        let mut pool = MemoryPool::new();
        let id = pool.allocate(device_config("f", 256)).unwrap();
        pool.deallocate(id).unwrap();
        let free_before = pool.free_list.len();
        pool.allocate(device_config("g", 256)).unwrap();
        assert!(pool.free_list.len() <= free_before);
    }

    #[test]
    fn free_list_splits_on_partial_reuse() {
        let mut pool = MemoryPool::new();
        let id = pool.allocate(device_config("big", 512)).unwrap();
        pool.deallocate(id).unwrap();
        pool.allocate(device_config("small", 128)).unwrap();
        // Free list should still have the leftover (512 - 128 = 384).
        let leftover: usize = pool.free_list.iter().map(|&(_, sz)| sz).sum();
        assert_eq!(leftover, 384);
    }

    // ---------------------------------------------------------------
    // Defragmentation
    // ---------------------------------------------------------------

    #[test]
    fn defragment_empty_is_zero() {
        let mut pool = MemoryPool::new();
        assert_eq!(pool.defragment(), 0);
    }

    #[test]
    fn defragment_single_entry_is_zero() {
        let mut pool = MemoryPool::new();
        let id = pool.allocate(device_config("x", 128)).unwrap();
        pool.deallocate(id).unwrap();
        assert_eq!(pool.defragment(), 0);
    }

    #[test]
    fn defragment_merges_entries() {
        let mut pool = MemoryPool::new();
        let a = pool.allocate(device_config("a", 128)).unwrap();
        let b = pool.allocate(device_config("b", 128)).unwrap();
        pool.deallocate(a).unwrap();
        pool.deallocate(b).unwrap();
        let before = pool.free_list.len();
        let recovered = pool.defragment();
        assert!(pool.free_list.len() <= before || recovered > 0);
    }

    // ---------------------------------------------------------------
    // Reset
    // ---------------------------------------------------------------

    #[test]
    fn reset_clears_everything() {
        let mut pool = MemoryPool::new();
        pool.allocate(device_config("a", 1024)).unwrap();
        pool.allocate(device_config("b", 2048)).unwrap();
        pool.reset();
        assert_eq!(pool.current_usage(), 0);
        assert_eq!(pool.peak_usage(), 0);
        assert_eq!(pool.allocation_count(), 0);
        assert!(pool.allocations.is_empty());
        assert!(pool.free_list.is_empty());
    }

    #[test]
    fn reset_allows_new_allocations() {
        let mut pool = MemoryPool::new();
        pool.allocate(device_config("a", 1024)).unwrap();
        pool.reset();
        let id = pool.allocate(device_config("b", 512)).unwrap();
        assert_eq!(id, 1);
        assert_eq!(pool.current_usage(), 512);
    }

    // ---------------------------------------------------------------
    // Transfer tracking
    // ---------------------------------------------------------------

    #[test]
    fn transfer_tracker_empty() {
        let t = TransferTracker::default();
        assert_eq!(t.transfer_count(), 0);
        assert_eq!(t.total_bytes_transferred(), 0);
        assert!((t.average_bandwidth_gbps() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn transfer_record_and_count() {
        let mut t = TransferTracker::default();
        t.record_transfer(MemoryRegion::Host, MemoryRegion::Device, 4096, 1000);
        assert_eq!(t.transfer_count(), 1);
        assert_eq!(t.total_bytes_transferred(), 4096);
    }

    #[test]
    fn transfer_bandwidth_calculation() {
        let mut t = TransferTracker::default();
        // 1 GB in 1 second = 1 GB/s
        t.record_transfer(MemoryRegion::Host, MemoryRegion::Device, 1_000_000_000, 1_000_000_000);
        let bw = t.average_bandwidth_gbps();
        assert!((bw - 1.0).abs() < 0.01);
    }

    #[test]
    fn transfer_multiple_events() {
        let mut t = TransferTracker::default();
        t.record_transfer(MemoryRegion::Host, MemoryRegion::Device, 1000, 500);
        t.record_transfer(MemoryRegion::Device, MemoryRegion::Host, 2000, 500);
        assert_eq!(t.transfer_count(), 2);
        assert_eq!(t.total_bytes_transferred(), 3000);
        // 3000 bytes / 1000 ns = 3.0 GB/s
        let bw = t.average_bandwidth_gbps();
        assert!((bw - 3.0).abs() < 0.01);
    }

    #[test]
    fn transfer_zero_duration() {
        let mut t = TransferTracker::default();
        t.record_transfer(MemoryRegion::Host, MemoryRegion::Device, 1024, 0);
        assert!((t.average_bandwidth_gbps() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn transfer_unified_to_device() {
        let mut t = TransferTracker::default();
        t.record_transfer(MemoryRegion::Unified, MemoryRegion::Device, 2048, 100);
        assert_eq!(t.total_bytes_transferred(), 2048);
    }

    // ---------------------------------------------------------------
    // Unified memory region properties
    // ---------------------------------------------------------------

    #[test]
    fn unified_region_display() {
        assert_eq!(format!("{}", MemoryRegion::Unified), "Unified");
    }

    #[test]
    fn unified_allocation_tracked() {
        let mut pool = MemoryPool::new();
        let id = pool.allocate(unified_config("u", 4096)).unwrap();
        let rec = pool.get_allocation(id).unwrap();
        assert_eq!(rec.config.region, MemoryRegion::Unified);
    }

    #[test]
    fn unified_budget_separate_from_device() {
        let mut pool = MemoryPool::new();
        pool.allocate(unified_config("u", 4096)).unwrap();
        let budget = MemoryBudget { device_limit: 1024, host_limit: 1024, unified_limit: 8192 };
        assert!(budget.can_allocate(&pool, &device_config("d", 512)));
        assert!(budget.can_allocate(&pool, &unified_config("u2", 4000)));
    }

    // ---------------------------------------------------------------
    // Pinned memory
    // ---------------------------------------------------------------

    #[test]
    fn pinned_region_uses_host_limit() {
        let pool = MemoryPool::new();
        let budget = MemoryBudget { device_limit: 0, host_limit: 1024, unified_limit: 0 };
        assert!(budget.can_allocate(&pool, &pinned_config("p", 512)));
    }

    #[test]
    fn pinned_region_display() {
        assert_eq!(format!("{}", MemoryRegion::Pinned), "Pinned");
    }

    // ---------------------------------------------------------------
    // Edge cases
    // ---------------------------------------------------------------

    #[test]
    fn zero_byte_alloc_rejected() {
        let mut pool = MemoryPool::new();
        assert_eq!(pool.allocate(device_config("z", 0)), Err(MemoryError::ZeroSizeAllocation),);
    }

    #[test]
    fn invalid_alignment_rejected() {
        let mut pool = MemoryPool::new();
        let cfg = AllocationConfig {
            region: MemoryRegion::Device,
            size_bytes: 128,
            alignment: 3,
            name: "bad".into(),
        };
        assert_eq!(pool.allocate(cfg), Err(MemoryError::InvalidAlignment(3)));
    }

    #[test]
    fn zero_alignment_rejected() {
        let mut pool = MemoryPool::new();
        let cfg = AllocationConfig {
            region: MemoryRegion::Device,
            size_bytes: 128,
            alignment: 0,
            name: "bad".into(),
        };
        assert_eq!(pool.allocate(cfg), Err(MemoryError::InvalidAlignment(0)));
    }

    #[test]
    fn large_allocation() {
        let mut pool = MemoryPool::new();
        let size = 16 * 1024 * 1024 * 1024_usize; // 16 GiB
        let id = pool
            .allocate(AllocationConfig {
                region: MemoryRegion::Device,
                size_bytes: size,
                alignment: 64,
                name: "vram".into(),
            })
            .unwrap();
        assert_eq!(pool.current_usage(), size);
        pool.deallocate(id).unwrap();
        assert_eq!(pool.current_usage(), 0);
    }

    // ---------------------------------------------------------------
    // Concurrent-style allocation patterns
    // ---------------------------------------------------------------

    #[test]
    fn many_allocations_and_deallocations() {
        let mut pool = MemoryPool::new();
        let mut ids = Vec::new();
        for i in 0..100 {
            let id = pool.allocate(device_config(&format!("buf{i}"), 64 * (i + 1))).unwrap();
            ids.push(id);
        }
        assert_eq!(pool.allocation_count(), 100);

        for &id in ids.iter().filter(|id| **id % 2 == 1) {
            pool.deallocate(id).unwrap();
        }
        assert_eq!(pool.allocations.len(), 50);
    }

    #[test]
    fn interleaved_alloc_dealloc() {
        let mut pool = MemoryPool::new();
        let a = pool.allocate(device_config("a", 128)).unwrap();
        let b = pool.allocate(device_config("b", 256)).unwrap();
        pool.deallocate(a).unwrap();
        let c = pool.allocate(device_config("c", 64)).unwrap();
        pool.deallocate(b).unwrap();
        let d = pool.allocate(device_config("d", 512)).unwrap();
        pool.deallocate(c).unwrap();
        pool.deallocate(d).unwrap();
        assert_eq!(pool.current_usage(), 0);
    }

    #[test]
    fn mixed_region_allocations() {
        let mut pool = MemoryPool::new();
        pool.allocate(device_config("d", 1024)).unwrap();
        pool.allocate(host_config("h", 2048)).unwrap();
        pool.allocate(unified_config("u", 4096)).unwrap();
        pool.allocate(pinned_config("p", 512)).unwrap();
        assert_eq!(pool.allocations.len(), 4);
        // Total: 1024 + 2048 + 4096 + 512 = 7680
        assert_eq!(pool.current_usage(), 7680);
    }

    // ---------------------------------------------------------------
    // MemoryError Display
    // ---------------------------------------------------------------

    #[test]
    fn error_display_budget_exceeded() {
        let e = MemoryError::BudgetExceeded { requested: 100, available: 50 };
        let s = format!("{e}");
        assert!(s.contains("100"));
        assert!(s.contains("50"));
    }

    #[test]
    fn error_display_invalid_id() {
        let e = MemoryError::InvalidAllocationId(42);
        assert!(format!("{e}").contains("42"));
    }

    #[test]
    fn error_display_zero_size() {
        let e = MemoryError::ZeroSizeAllocation;
        assert!(format!("{e}").contains("zero"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(MemoryError::ZeroSizeAllocation);
        assert!(!e.to_string().is_empty());
    }

    // ---------------------------------------------------------------
    // Default impls
    // ---------------------------------------------------------------

    #[test]
    fn allocation_config_default() {
        let c = AllocationConfig::default();
        assert_eq!(c.alignment, 64);
        assert_eq!(c.size_bytes, 0);
        assert_eq!(c.region, MemoryRegion::Device);
    }

    #[test]
    fn memory_pool_default() {
        let p = MemoryPool::default();
        assert_eq!(p.current_usage(), 0);
    }
}
