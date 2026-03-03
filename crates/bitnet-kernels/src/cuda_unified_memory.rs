//! CUDA Unified Memory management with page-migration hints and prefetch.
//!
//! Provides [`UnifiedMemoryAllocator`] for managing CUDA managed memory with
//! advisory hints (read-mostly, preferred location, accessed-by) and async
//! prefetch operations.  All code is feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//!
//! On non-GPU builds the module is not compiled at all; tests use mock
//! device IDs so no real GPU hardware is required.

use bitnet_common::{KernelError, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ── Device identifiers ───────────────────────────────────────────────

/// Logical device identifier for unified memory operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceId {
    /// The host (CPU) side.
    Host,
    /// A CUDA device identified by ordinal.
    Gpu(u32),
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceId::Host => write!(f, "Host"),
            DeviceId::Gpu(id) => write!(f, "GPU:{id}"),
        }
    }
}

// ── Memory advice ────────────────────────────────────────────────────

/// Advice hints that map to `cudaMemAdvise` flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryAdvice {
    /// Data will mostly be read (enables read-duplication across devices).
    SetReadMostly,
    /// Unset a previous `SetReadMostly` hint.
    UnsetReadMostly,
    /// Hint the preferred physical location for the allocation.
    SetPreferredLocation(DeviceId),
    /// Hint that the given device will access the allocation.
    SetAccessedBy(DeviceId),
    /// Unset a previous `SetAccessedBy` hint.
    UnsetAccessedBy(DeviceId),
}

impl std::fmt::Display for MemoryAdvice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryAdvice::SetReadMostly => write!(f, "SetReadMostly"),
            MemoryAdvice::UnsetReadMostly => write!(f, "UnsetReadMostly"),
            MemoryAdvice::SetPreferredLocation(d) => {
                write!(f, "SetPreferredLocation({d})")
            }
            MemoryAdvice::SetAccessedBy(d) => write!(f, "SetAccessedBy({d})"),
            MemoryAdvice::UnsetAccessedBy(d) => write!(f, "UnsetAccessedBy({d})"),
        }
    }
}

// ── Prefetch target ──────────────────────────────────────────────────

/// Target for an asynchronous prefetch operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchTarget {
    /// Prefetch pages to the specified CUDA device.
    ToDevice(u32),
    /// Prefetch pages back to host memory.
    ToHost,
}

impl std::fmt::Display for PrefetchTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrefetchTarget::ToDevice(id) => write!(f, "GPU:{id}"),
            PrefetchTarget::ToHost => write!(f, "Host"),
        }
    }
}

// ── Allocation record ────────────────────────────────────────────────

/// Metadata for a single unified-memory allocation.
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Unique identifier for this allocation.
    pub id: u64,
    /// Requested size in bytes.
    pub size: usize,
    /// Optional human-readable label.
    pub label: Option<String>,
    /// Advice hints that have been applied.
    pub advice: Vec<MemoryAdvice>,
    /// Whether a prefetch has been issued.
    pub prefetched: bool,
}

// ── Usage statistics ─────────────────────────────────────────────────

/// Aggregate memory usage statistics.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total bytes currently allocated.
    pub total_allocated: usize,
    /// Number of live allocations.
    pub allocation_count: usize,
    /// Peak allocation in bytes since last reset.
    pub peak_allocated: usize,
    /// Number of allocations that were made.
    pub total_allocations: u64,
    /// Number of deallocations that were made.
    pub total_deallocations: u64,
    /// Number of prefetch operations issued.
    pub prefetch_count: u64,
    /// Number of advice hints applied.
    pub advice_count: u64,
}

// ── Oversubscription policy ──────────────────────────────────────────

/// Policy to apply when unified memory is over-subscribed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OversubscriptionPolicy {
    /// Return an error when the budget is exceeded.
    Reject,
    /// Allow oversubscription but log a warning.
    #[default]
    WarnAndContinue,
    /// Evict oldest allocations until within budget.
    EvictOldest,
}

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for [`UnifiedMemoryAllocator`].
#[derive(Debug, Clone)]
pub struct UnifiedMemoryConfig {
    /// Hard memory budget in bytes (`0` = unlimited).
    pub memory_budget: usize,
    /// Policy when budget is exceeded.
    pub oversubscription_policy: OversubscriptionPolicy,
    /// Default device for prefetch operations.
    pub default_device: DeviceId,
    /// Alignment for allocations in bytes (must be power of two, ≥ 1).
    pub alignment: usize,
}

impl Default for UnifiedMemoryConfig {
    fn default() -> Self {
        Self {
            memory_budget: 0,
            oversubscription_policy: OversubscriptionPolicy::default(),
            default_device: DeviceId::Gpu(0),
            alignment: 256,
        }
    }
}

impl UnifiedMemoryConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<()> {
        if self.alignment == 0 || !self.alignment.is_power_of_two() {
            return Err(KernelError::InvalidArguments {
                reason: "alignment must be a non-zero power of two".into(),
            }
            .into());
        }
        Ok(())
    }
}

// ── Allocator ────────────────────────────────────────────────────────

/// Thread-safe CUDA unified memory allocator with advisory hints.
///
/// Tracks allocations, advice, and prefetch state.  On non-GPU hosts the
/// allocator works entirely in software (no real CUDA calls) which makes it
/// suitable for mock-based testing.
#[derive(Debug)]
pub struct UnifiedMemoryAllocator {
    inner: Arc<Mutex<AllocatorInner>>,
}

#[derive(Debug)]
struct AllocatorInner {
    config: UnifiedMemoryConfig,
    next_id: u64,
    allocations: HashMap<u64, AllocationRecord>,
    stats: MemoryStats,
}

impl UnifiedMemoryAllocator {
    /// Create a new allocator with the given configuration.
    pub fn new(config: UnifiedMemoryConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            inner: Arc::new(Mutex::new(AllocatorInner {
                config,
                next_id: 1,
                allocations: HashMap::new(),
                stats: MemoryStats::default(),
            })),
        })
    }

    /// Create an allocator with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(UnifiedMemoryConfig::default())
    }

    // ── Allocation ───────────────────────────────────────────────────

    /// Allocate `size` bytes of unified memory.
    ///
    /// Returns the allocation id on success.  If `size` is zero an error is
    /// returned.
    pub fn allocate(&self, size: usize, label: Option<&str>) -> Result<u64> {
        if size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "allocation size must be non-zero".into(),
            }
            .into());
        }

        let mut inner = self.inner.lock().unwrap();
        let aligned_size = align_up(size, inner.config.alignment);

        // Budget check.
        if inner.config.memory_budget > 0 {
            let new_total = inner.stats.total_allocated + aligned_size;
            if new_total > inner.config.memory_budget {
                match inner.config.oversubscription_policy {
                    OversubscriptionPolicy::Reject => {
                        return Err(KernelError::GpuError {
                            reason: format!(
                                "unified memory budget exceeded: requested {} bytes \
                                 (aligned {aligned_size}), budget {} with {} in use",
                                size, inner.config.memory_budget, inner.stats.total_allocated,
                            ),
                        }
                        .into());
                    }
                    OversubscriptionPolicy::WarnAndContinue => {
                        log::warn!(
                            "unified memory oversubscribed: {new_total} > {}",
                            inner.config.memory_budget
                        );
                    }
                    OversubscriptionPolicy::EvictOldest => {
                        evict_oldest(&mut inner, aligned_size)?;
                    }
                }
            }
        }

        let id = inner.next_id;
        inner.next_id += 1;

        let record = AllocationRecord {
            id,
            size: aligned_size,
            label: label.map(String::from),
            advice: Vec::new(),
            prefetched: false,
        };

        inner.allocations.insert(id, record);
        inner.stats.total_allocated += aligned_size;
        inner.stats.allocation_count += 1;
        inner.stats.total_allocations += 1;
        if inner.stats.total_allocated > inner.stats.peak_allocated {
            inner.stats.peak_allocated = inner.stats.total_allocated;
        }

        Ok(id)
    }

    // ── Deallocation ─────────────────────────────────────────────────

    /// Free a previously allocated region.
    pub fn deallocate(&self, id: u64) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        let record = inner.allocations.remove(&id).ok_or_else(|| {
            KernelError::InvalidArguments { reason: format!("no allocation with id {id}") }
        })?;
        inner.stats.total_allocated = inner.stats.total_allocated.saturating_sub(record.size);
        inner.stats.allocation_count = inner.stats.allocation_count.saturating_sub(1);
        inner.stats.total_deallocations += 1;
        Ok(())
    }

    // ── Advice ───────────────────────────────────────────────────────

    /// Apply a memory advice hint to an existing allocation.
    pub fn advise(&self, id: u64, advice: MemoryAdvice) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        let record = inner.allocations.get_mut(&id).ok_or_else(|| {
            KernelError::InvalidArguments { reason: format!("no allocation with id {id}") }
        })?;
        record.advice.push(advice);
        inner.stats.advice_count += 1;
        Ok(())
    }

    /// Apply multiple advice hints to an allocation atomically.
    pub fn advise_many(&self, id: u64, advice_list: &[MemoryAdvice]) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.allocations.contains_key(&id) {
            return Err(KernelError::InvalidArguments {
                reason: format!("no allocation with id {id}"),
            }
            .into());
        }
        let count = advice_list.len() as u64;
        let record = inner.allocations.get_mut(&id).unwrap();
        for &a in advice_list {
            record.advice.push(a);
        }
        inner.stats.advice_count += count;
        Ok(())
    }

    // ── Prefetch ─────────────────────────────────────────────────────

    /// Issue a prefetch for the allocation to the given target.
    pub fn prefetch(&self, id: u64, target: PrefetchTarget) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.allocations.contains_key(&id) {
            return Err(KernelError::InvalidArguments {
                reason: format!("no allocation with id {id}"),
            }
            .into());
        }
        let record = inner.allocations.get_mut(&id).unwrap();
        record.prefetched = true;
        let alloc_id = record.id;
        let alloc_size = record.size;
        inner.stats.prefetch_count += 1;
        log::debug!("prefetch alloc {alloc_id} ({alloc_size} bytes) → {target}");
        Ok(())
    }

    /// Prefetch an allocation to the allocator's default device.
    pub fn prefetch_to_default(&self, id: u64) -> Result<()> {
        let default_target = {
            let inner = self.inner.lock().unwrap();
            match inner.config.default_device {
                DeviceId::Gpu(ordinal) => PrefetchTarget::ToDevice(ordinal),
                DeviceId::Host => PrefetchTarget::ToHost,
            }
        };
        self.prefetch(id, default_target)
    }

    /// Batch-prefetch multiple allocations to the same target.
    pub fn prefetch_many(&self, ids: &[u64], target: PrefetchTarget) -> Result<()> {
        for &id in ids {
            self.prefetch(id, target)?;
        }
        Ok(())
    }

    // ── Queries ──────────────────────────────────────────────────────

    /// Look up a single allocation record.
    pub fn get_allocation(&self, id: u64) -> Result<AllocationRecord> {
        let inner = self.inner.lock().unwrap();
        inner.allocations.get(&id).cloned().ok_or_else(|| {
            KernelError::InvalidArguments { reason: format!("no allocation with id {id}") }.into()
        })
    }

    /// Return a snapshot of aggregate memory statistics.
    pub fn stats(&self) -> MemoryStats {
        self.inner.lock().unwrap().stats.clone()
    }

    /// Reset peak allocation tracking.
    pub fn reset_peak(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.stats.peak_allocated = inner.stats.total_allocated;
    }

    /// Return the number of live allocations.
    pub fn live_allocation_count(&self) -> usize {
        self.inner.lock().unwrap().allocations.len()
    }

    /// Return total bytes currently allocated.
    pub fn total_allocated(&self) -> usize {
        self.inner.lock().unwrap().stats.total_allocated
    }

    /// Return whether the allocator is currently over budget.
    pub fn is_oversubscribed(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.config.memory_budget > 0 && inner.stats.total_allocated > inner.config.memory_budget
    }

    /// Return the configured budget (0 = unlimited).
    pub fn budget(&self) -> usize {
        self.inner.lock().unwrap().config.memory_budget
    }

    /// Return a list of all live allocation ids.
    pub fn live_allocation_ids(&self) -> Vec<u64> {
        let inner = self.inner.lock().unwrap();
        let mut ids: Vec<u64> = inner.allocations.keys().copied().collect();
        ids.sort_unstable();
        ids
    }
}

impl Clone for UnifiedMemoryAllocator {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Round `size` up to the next multiple of `alignment`.
fn align_up(size: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (size + alignment - 1) & !(alignment - 1)
}

/// Evict the oldest allocations until `needed` bytes can fit within budget.
fn evict_oldest(inner: &mut AllocatorInner, needed: usize) -> Result<()> {
    let budget = inner.config.memory_budget;
    // Sort by id (ascending = oldest first).
    let mut ids: Vec<u64> = inner.allocations.keys().copied().collect();
    ids.sort_unstable();

    for id in ids {
        if inner.stats.total_allocated + needed <= budget {
            break;
        }
        if let Some(record) = inner.allocations.remove(&id) {
            inner.stats.total_allocated = inner.stats.total_allocated.saturating_sub(record.size);
            inner.stats.allocation_count = inner.stats.allocation_count.saturating_sub(1);
            inner.stats.total_deallocations += 1;
            log::debug!("evicted allocation {id} ({} bytes)", record.size);
        }
    }
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_allocator() -> UnifiedMemoryAllocator {
        UnifiedMemoryAllocator::with_defaults().unwrap()
    }

    fn budget_allocator(budget: usize) -> UnifiedMemoryAllocator {
        UnifiedMemoryAllocator::new(UnifiedMemoryConfig {
            memory_budget: budget,
            ..Default::default()
        })
        .unwrap()
    }

    // ── Basic allocation / deallocation ──────────────────────────────

    #[test]
    fn test_allocate_returns_unique_ids() {
        let alloc = default_allocator();
        let id1 = alloc.allocate(1024, None).unwrap();
        let id2 = alloc.allocate(2048, None).unwrap();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_allocate_zero_size_errors() {
        let alloc = default_allocator();
        assert!(alloc.allocate(0, None).is_err());
    }

    #[test]
    fn test_deallocate_success() {
        let alloc = default_allocator();
        let id = alloc.allocate(512, None).unwrap();
        assert!(alloc.deallocate(id).is_ok());
        assert_eq!(alloc.live_allocation_count(), 0);
    }

    #[test]
    fn test_deallocate_unknown_id_errors() {
        let alloc = default_allocator();
        assert!(alloc.deallocate(999).is_err());
    }

    #[test]
    fn test_double_deallocate_errors() {
        let alloc = default_allocator();
        let id = alloc.allocate(256, None).unwrap();
        alloc.deallocate(id).unwrap();
        assert!(alloc.deallocate(id).is_err());
    }

    // ── Alignment ────────────────────────────────────────────────────

    #[test]
    fn test_allocation_is_aligned() {
        let alloc = default_allocator();
        let id = alloc.allocate(100, None).unwrap();
        let record = alloc.get_allocation(id).unwrap();
        assert_eq!(record.size % 256, 0);
        assert!(record.size >= 100);
    }

    #[test]
    fn test_custom_alignment() {
        let alloc = UnifiedMemoryAllocator::new(UnifiedMemoryConfig {
            alignment: 64,
            ..Default::default()
        })
        .unwrap();
        let id = alloc.allocate(50, None).unwrap();
        let record = alloc.get_allocation(id).unwrap();
        assert_eq!(record.size % 64, 0);
    }

    #[test]
    fn test_invalid_alignment_rejected() {
        let res =
            UnifiedMemoryAllocator::new(UnifiedMemoryConfig { alignment: 0, ..Default::default() });
        assert!(res.is_err());
        let res =
            UnifiedMemoryAllocator::new(UnifiedMemoryConfig { alignment: 3, ..Default::default() });
        assert!(res.is_err());
    }

    // ── Labels ───────────────────────────────────────────────────────

    #[test]
    fn test_label_stored() {
        let alloc = default_allocator();
        let id = alloc.allocate(128, Some("weights")).unwrap();
        let record = alloc.get_allocation(id).unwrap();
        assert_eq!(record.label.as_deref(), Some("weights"));
    }

    #[test]
    fn test_no_label() {
        let alloc = default_allocator();
        let id = alloc.allocate(128, None).unwrap();
        let record = alloc.get_allocation(id).unwrap();
        assert!(record.label.is_none());
    }

    // ── Advice ───────────────────────────────────────────────────────

    #[test]
    fn test_advise_read_mostly() {
        let alloc = default_allocator();
        let id = alloc.allocate(256, None).unwrap();
        alloc.advise(id, MemoryAdvice::SetReadMostly).unwrap();
        let record = alloc.get_allocation(id).unwrap();
        assert_eq!(record.advice.len(), 1);
        assert_eq!(record.advice[0], MemoryAdvice::SetReadMostly);
    }

    #[test]
    fn test_advise_preferred_location() {
        let alloc = default_allocator();
        let id = alloc.allocate(256, None).unwrap();
        alloc.advise(id, MemoryAdvice::SetPreferredLocation(DeviceId::Gpu(0))).unwrap();
        let record = alloc.get_allocation(id).unwrap();
        assert_eq!(record.advice[0], MemoryAdvice::SetPreferredLocation(DeviceId::Gpu(0)));
    }

    #[test]
    fn test_advise_accessed_by() {
        let alloc = default_allocator();
        let id = alloc.allocate(256, None).unwrap();
        alloc.advise(id, MemoryAdvice::SetAccessedBy(DeviceId::Gpu(1))).unwrap();
        let record = alloc.get_allocation(id).unwrap();
        assert_eq!(record.advice[0], MemoryAdvice::SetAccessedBy(DeviceId::Gpu(1)));
    }

    #[test]
    fn test_advise_invalid_id_errors() {
        let alloc = default_allocator();
        assert!(alloc.advise(42, MemoryAdvice::SetReadMostly).is_err());
    }

    #[test]
    fn test_advise_many() {
        let alloc = default_allocator();
        let id = alloc.allocate(512, None).unwrap();
        let hints = [
            MemoryAdvice::SetReadMostly,
            MemoryAdvice::SetPreferredLocation(DeviceId::Host),
            MemoryAdvice::SetAccessedBy(DeviceId::Gpu(0)),
        ];
        alloc.advise_many(id, &hints).unwrap();
        let record = alloc.get_allocation(id).unwrap();
        assert_eq!(record.advice.len(), 3);
    }

    #[test]
    fn test_unset_read_mostly() {
        let alloc = default_allocator();
        let id = alloc.allocate(256, None).unwrap();
        alloc.advise(id, MemoryAdvice::SetReadMostly).unwrap();
        alloc.advise(id, MemoryAdvice::UnsetReadMostly).unwrap();
        let record = alloc.get_allocation(id).unwrap();
        assert_eq!(record.advice.len(), 2);
    }

    #[test]
    fn test_unset_accessed_by() {
        let alloc = default_allocator();
        let id = alloc.allocate(256, None).unwrap();
        alloc.advise(id, MemoryAdvice::SetAccessedBy(DeviceId::Gpu(0))).unwrap();
        alloc.advise(id, MemoryAdvice::UnsetAccessedBy(DeviceId::Gpu(0))).unwrap();
        let record = alloc.get_allocation(id).unwrap();
        assert_eq!(record.advice.len(), 2);
    }

    // ── Prefetch ─────────────────────────────────────────────────────

    #[test]
    fn test_prefetch_to_device() {
        let alloc = default_allocator();
        let id = alloc.allocate(1024, None).unwrap();
        alloc.prefetch(id, PrefetchTarget::ToDevice(0)).unwrap();
        let record = alloc.get_allocation(id).unwrap();
        assert!(record.prefetched);
    }

    #[test]
    fn test_prefetch_to_host() {
        let alloc = default_allocator();
        let id = alloc.allocate(1024, None).unwrap();
        alloc.prefetch(id, PrefetchTarget::ToHost).unwrap();
        let record = alloc.get_allocation(id).unwrap();
        assert!(record.prefetched);
    }

    #[test]
    fn test_prefetch_to_default() {
        let alloc = default_allocator();
        let id = alloc.allocate(1024, None).unwrap();
        alloc.prefetch_to_default(id).unwrap();
        let record = alloc.get_allocation(id).unwrap();
        assert!(record.prefetched);
    }

    #[test]
    fn test_prefetch_invalid_id_errors() {
        let alloc = default_allocator();
        assert!(alloc.prefetch(999, PrefetchTarget::ToHost).is_err());
    }

    #[test]
    fn test_prefetch_many() {
        let alloc = default_allocator();
        let id1 = alloc.allocate(512, None).unwrap();
        let id2 = alloc.allocate(512, None).unwrap();
        alloc.prefetch_many(&[id1, id2], PrefetchTarget::ToDevice(0)).unwrap();
        assert!(alloc.get_allocation(id1).unwrap().prefetched);
        assert!(alloc.get_allocation(id2).unwrap().prefetched);
    }

    // ── Statistics ───────────────────────────────────────────────────

    #[test]
    fn test_stats_after_allocations() {
        let alloc = default_allocator();
        alloc.allocate(256, None).unwrap();
        alloc.allocate(512, None).unwrap();
        let stats = alloc.stats();
        assert_eq!(stats.allocation_count, 2);
        assert_eq!(stats.total_allocations, 2);
        assert!(stats.total_allocated >= 768);
    }

    #[test]
    fn test_stats_after_deallocation() {
        let alloc = default_allocator();
        let id = alloc.allocate(256, None).unwrap();
        alloc.deallocate(id).unwrap();
        let stats = alloc.stats();
        assert_eq!(stats.allocation_count, 0);
        assert_eq!(stats.total_deallocations, 1);
        assert_eq!(stats.total_allocated, 0);
    }

    #[test]
    fn test_peak_tracking() {
        let alloc = default_allocator();
        let id1 = alloc.allocate(1024, None).unwrap();
        let _id2 = alloc.allocate(2048, None).unwrap();
        let peak_after_two = alloc.stats().peak_allocated;
        alloc.deallocate(id1).unwrap();
        let stats = alloc.stats();
        assert_eq!(stats.peak_allocated, peak_after_two);
        assert!(stats.total_allocated < peak_after_two);
    }

    #[test]
    fn test_reset_peak() {
        let alloc = default_allocator();
        alloc.allocate(4096, None).unwrap();
        let peak1 = alloc.stats().peak_allocated;
        alloc.reset_peak();
        let peak2 = alloc.stats().peak_allocated;
        assert_eq!(peak2, alloc.total_allocated());
        assert!(peak2 <= peak1);
    }

    #[test]
    fn test_advice_and_prefetch_counters() {
        let alloc = default_allocator();
        let id = alloc.allocate(256, None).unwrap();
        alloc.advise(id, MemoryAdvice::SetReadMostly).unwrap();
        alloc.prefetch(id, PrefetchTarget::ToDevice(0)).unwrap();
        let stats = alloc.stats();
        assert_eq!(stats.advice_count, 1);
        assert_eq!(stats.prefetch_count, 1);
    }

    // ── Budget / oversubscription ────────────────────────────────────

    #[test]
    fn test_reject_policy_prevents_oversubscription() {
        let alloc = UnifiedMemoryAllocator::new(UnifiedMemoryConfig {
            memory_budget: 512,
            oversubscription_policy: OversubscriptionPolicy::Reject,
            ..Default::default()
        })
        .unwrap();
        // With 256 alignment, even 256 bytes → 256 aligned.
        let _id = alloc.allocate(256, None).unwrap();
        // Second 256 → total 512, still fine.
        let _id2 = alloc.allocate(256, None).unwrap();
        // Third should be rejected.
        assert!(alloc.allocate(256, None).is_err());
    }

    #[test]
    fn test_warn_policy_allows_oversubscription() {
        let alloc = UnifiedMemoryAllocator::new(UnifiedMemoryConfig {
            memory_budget: 256,
            oversubscription_policy: OversubscriptionPolicy::WarnAndContinue,
            ..Default::default()
        })
        .unwrap();
        alloc.allocate(256, None).unwrap();
        // Over budget but allowed.
        assert!(alloc.allocate(256, None).is_ok());
        assert!(alloc.is_oversubscribed());
    }

    #[test]
    fn test_evict_oldest_policy() {
        let alloc = UnifiedMemoryAllocator::new(UnifiedMemoryConfig {
            memory_budget: 512,
            oversubscription_policy: OversubscriptionPolicy::EvictOldest,
            alignment: 256,
            ..Default::default()
        })
        .unwrap();
        let _id1 = alloc.allocate(256, Some("old")).unwrap();
        let _id2 = alloc.allocate(256, Some("newer")).unwrap();
        // Budget full (512), this should evict oldest to make room.
        let id3 = alloc.allocate(256, Some("newest")).unwrap();
        assert!(alloc.get_allocation(id3).is_ok());
        // The oldest allocation should have been evicted.
        assert_eq!(alloc.live_allocation_count(), 2);
    }

    #[test]
    fn test_is_oversubscribed_false_when_within_budget() {
        let alloc = budget_allocator(1024);
        alloc.allocate(256, None).unwrap();
        assert!(!alloc.is_oversubscribed());
    }

    #[test]
    fn test_unlimited_budget_never_oversubscribed() {
        let alloc = default_allocator();
        alloc.allocate(1_000_000, None).unwrap();
        assert!(!alloc.is_oversubscribed());
    }

    // ── Query helpers ────────────────────────────────────────────────

    #[test]
    fn test_get_allocation_missing() {
        let alloc = default_allocator();
        assert!(alloc.get_allocation(1).is_err());
    }

    #[test]
    fn test_live_allocation_ids() {
        let alloc = default_allocator();
        let id1 = alloc.allocate(128, None).unwrap();
        let id2 = alloc.allocate(128, None).unwrap();
        let ids = alloc.live_allocation_ids();
        assert_eq!(ids, vec![id1, id2]);
    }

    #[test]
    fn test_budget_returns_configured_value() {
        let alloc = budget_allocator(4096);
        assert_eq!(alloc.budget(), 4096);
    }

    // ── Clone shares state ───────────────────────────────────────────

    #[test]
    fn test_clone_shares_state() {
        let alloc = default_allocator();
        let id = alloc.allocate(128, None).unwrap();
        let cloned = alloc.clone();
        assert!(cloned.get_allocation(id).is_ok());
        let id2 = cloned.allocate(256, None).unwrap();
        assert!(alloc.get_allocation(id2).is_ok());
    }

    // ── Display impls ────────────────────────────────────────────────

    #[test]
    fn test_display_device_id() {
        assert_eq!(DeviceId::Host.to_string(), "Host");
        assert_eq!(DeviceId::Gpu(3).to_string(), "GPU:3");
    }

    #[test]
    fn test_display_memory_advice() {
        assert_eq!(MemoryAdvice::SetReadMostly.to_string(), "SetReadMostly");
        assert_eq!(
            MemoryAdvice::SetPreferredLocation(DeviceId::Host).to_string(),
            "SetPreferredLocation(Host)"
        );
    }

    #[test]
    fn test_display_prefetch_target() {
        assert_eq!(PrefetchTarget::ToDevice(0).to_string(), "GPU:0");
        assert_eq!(PrefetchTarget::ToHost.to_string(), "Host");
    }

    // ── align_up helper ──────────────────────────────────────────────

    #[test]
    fn test_align_up_exact() {
        assert_eq!(align_up(256, 256), 256);
    }

    #[test]
    fn test_align_up_rounds() {
        assert_eq!(align_up(100, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }
}
