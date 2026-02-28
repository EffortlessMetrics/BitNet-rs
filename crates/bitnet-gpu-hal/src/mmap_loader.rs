//! Memory-mapped model loader for zero-copy tensor access.
//!
//! Provides efficient model loading via simulated memory-mapping,
//! with lazy tensor loading, prefetch scheduling, and memory budget
//! enforcement for large model files.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Data types ───────────────────────────────────────────────────────────────

/// Tensor element data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I2,
    U8,
}

impl DType {
    /// Size of one element in bytes.
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::I2 | Self::U8 => 1,
        }
    }
}

// ── MmapConfig ───────────────────────────────────────────────────────────────

/// Configuration for memory-mapped file access.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct MmapConfig {
    /// Whether to use memory mapping (vs. regular read).
    pub use_mmap: bool,
    /// Pre-fault all pages on mapping.
    pub prefault: bool,
    /// Attempt to use huge pages (2 MB / 1 GB).
    pub huge_pages: bool,
    /// Lock mapped pages into physical memory.
    pub lock_memory: bool,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self { use_mmap: true, prefault: false, huge_pages: false, lock_memory: false }
    }
}

// ── MmapRegion ───────────────────────────────────────────────────────────────

/// A contiguous mapped region within a file.
#[derive(Debug)]
pub struct MmapRegion {
    /// Byte offset from start of file.
    pub offset: usize,
    /// Length of the region in bytes.
    pub length: usize,
    /// Simulated pointer (index into backing buffer).
    ptr: usize,
    /// Whether the region is locked into physical memory.
    pub is_locked: bool,
}

impl MmapRegion {
    /// Create a new mapped region.
    const fn new(offset: usize, length: usize, ptr: usize) -> Self {
        Self { offset, length, ptr, is_locked: false }
    }

    /// Return the simulated pointer value.
    #[must_use]
    pub const fn ptr(&self) -> usize {
        self.ptr
    }

    /// Lock this region into physical memory.
    pub const fn lock(&mut self) {
        self.is_locked = true;
    }

    /// Unlock this region.
    pub const fn unlock(&mut self) {
        self.is_locked = false;
    }
}

// ── TensorView ───────────────────────────────────────────────────────────────

/// Zero-copy view into memory-mapped tensor data.
#[derive(Debug)]
pub struct TensorView<'a> {
    /// Tensor name.
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: DType,
    /// Raw byte slice of the tensor data.
    pub data: &'a [u8],
}

impl TensorView<'_> {
    /// Number of elements in the tensor.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.byte_size()
    }
}

// ── TensorDescriptor ─────────────────────────────────────────────────────────

/// Metadata for a tensor stored in a model file.
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    /// Tensor name.
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: DType,
    /// Byte offset within the file.
    pub offset: usize,
    /// Total byte length.
    pub byte_length: usize,
}

// ── MmapModelFile ────────────────────────────────────────────────────────────

/// A memory-mapped model file providing zero-copy tensor access.
///
/// In this CPU simulation the file is backed by an in-memory `Vec<u8>`.
#[derive(Debug)]
pub struct MmapModelFile {
    /// Simulated file content.
    data: Vec<u8>,
    /// Configuration used for mapping.
    config: MmapConfig,
    /// Registered tensor descriptors.
    tensors: HashMap<String, TensorDescriptor>,
    /// Active mapped regions.
    regions: Vec<MmapRegion>,
}

impl MmapModelFile {
    /// Open a model file from raw bytes with the given configuration.
    #[must_use]
    pub fn open(data: Vec<u8>, config: MmapConfig) -> Self {
        Self { data, config, tensors: HashMap::new(), regions: Vec::new() }
    }

    /// Register a tensor descriptor for later access.
    pub fn register_tensor(&mut self, desc: TensorDescriptor) {
        self.tensors.insert(desc.name.clone(), desc);
    }

    /// Map a region of the file.
    pub fn map_region(&mut self, offset: usize, length: usize) -> Result<usize, MmapError> {
        if offset + length > self.data.len() {
            return Err(MmapError::OutOfBounds { offset, length, file_size: self.data.len() });
        }
        let idx = self.regions.len();
        let mut region = MmapRegion::new(offset, length, offset);
        if self.config.lock_memory {
            region.lock();
        }
        self.regions.push(region);
        Ok(idx)
    }

    /// Get a tensor view by name.
    pub fn tensor_view(&self, name: &str) -> Result<TensorView<'_>, MmapError> {
        let desc =
            self.tensors.get(name).ok_or_else(|| MmapError::TensorNotFound(name.to_string()))?;
        let end = desc.offset + desc.byte_length;
        if end > self.data.len() {
            return Err(MmapError::OutOfBounds {
                offset: desc.offset,
                length: desc.byte_length,
                file_size: self.data.len(),
            });
        }
        Ok(TensorView {
            name: desc.name.clone(),
            shape: desc.shape.clone(),
            dtype: desc.dtype,
            data: &self.data[desc.offset..end],
        })
    }

    /// Get raw data slice.
    #[must_use]
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }

    /// Total file size in bytes.
    #[must_use]
    pub const fn file_size(&self) -> usize {
        self.data.len()
    }

    /// Number of registered tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// The configuration used.
    #[must_use]
    pub const fn config(&self) -> &MmapConfig {
        &self.config
    }

    /// Number of active mapped regions.
    #[must_use]
    pub const fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Get a reference to a mapped region by index.
    #[must_use]
    pub fn region(&self, idx: usize) -> Option<&MmapRegion> {
        self.regions.get(idx)
    }
}

// ── LazyTensorLoader ─────────────────────────────────────────────────────────

/// Demand-paged tensor loader that loads tensors on first access.
///
/// Tracks which tensors have been paged in and records access order.
#[derive(Debug)]
pub struct LazyTensorLoader {
    /// Tensor descriptors indexed by name.
    descriptors: HashMap<String, TensorDescriptor>,
    /// Set of tensors that have been loaded (paged in).
    loaded: HashMap<String, Vec<u8>>,
    /// Access order (most recent last).
    access_order: Vec<String>,
    /// Statistics.
    stats: MmapStats,
}

impl LazyTensorLoader {
    /// Create a new lazy loader.
    #[must_use]
    pub fn new() -> Self {
        Self {
            descriptors: HashMap::new(),
            loaded: HashMap::new(),
            access_order: Vec::new(),
            stats: MmapStats::new(),
        }
    }

    /// Register a tensor descriptor.
    pub fn register(&mut self, desc: TensorDescriptor, file_data: &[u8]) -> Result<(), MmapError> {
        let end = desc.offset + desc.byte_length;
        if end > file_data.len() {
            return Err(MmapError::OutOfBounds {
                offset: desc.offset,
                length: desc.byte_length,
                file_size: file_data.len(),
            });
        }
        self.loaded.remove(&desc.name);
        self.descriptors.insert(desc.name.clone(), desc);
        Ok(())
    }

    /// Get tensor data, loading on first access (demand paging).
    pub fn get(&mut self, name: &str, file_data: &[u8]) -> Result<&[u8], MmapError> {
        let desc = self
            .descriptors
            .get(name)
            .ok_or_else(|| MmapError::TensorNotFound(name.to_string()))?
            .clone();

        if !self.loaded.contains_key(name) {
            // Simulate page fault: load from backing data.
            let end = desc.offset + desc.byte_length;
            if end > file_data.len() {
                return Err(MmapError::OutOfBounds {
                    offset: desc.offset,
                    length: desc.byte_length,
                    file_size: file_data.len(),
                });
            }
            let segment = file_data[desc.offset..end].to_vec();
            self.loaded.insert(name.to_string(), segment);
            self.stats.page_faults.fetch_add(1, Ordering::Relaxed);
            self.stats.tensors_loaded.fetch_add(1, Ordering::Relaxed);
        }

        // Update access order (move to back).
        self.access_order.retain(|n| n != name);
        self.access_order.push(name.to_string());

        Ok(self.loaded.get(name).unwrap())
    }

    /// Check whether a tensor has been loaded.
    #[must_use]
    pub fn is_loaded(&self, name: &str) -> bool {
        self.loaded.contains_key(name)
    }

    /// Number of currently loaded tensors.
    #[must_use]
    pub fn loaded_count(&self) -> usize {
        self.loaded.len()
    }

    /// Evict a tensor from memory.
    pub fn evict(&mut self, name: &str) -> bool {
        if self.loaded.remove(name).is_some() {
            self.access_order.retain(|n| n != name);
            self.stats.tensors_evicted.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Evict the least-recently-used tensor.
    pub fn evict_lru(&mut self) -> Option<String> {
        if self.access_order.is_empty() {
            return None;
        }
        let lru_name = self.access_order[0].clone();
        self.evict(&lru_name);
        Some(lru_name)
    }

    /// Get current statistics.
    #[must_use]
    pub const fn stats(&self) -> &MmapStats {
        &self.stats
    }

    /// Return the access order (oldest first).
    #[must_use]
    pub fn access_order(&self) -> &[String] {
        &self.access_order
    }

    /// Number of registered tensor descriptors.
    #[must_use]
    pub fn descriptor_count(&self) -> usize {
        self.descriptors.len()
    }
}

impl Default for LazyTensorLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ── PrefetchScheduler ────────────────────────────────────────────────────────

/// Predicts which tensors will be needed next and triggers readahead.
///
/// Tracks access patterns to build a simple sequential prediction model.
#[derive(Debug)]
pub struct PrefetchScheduler {
    /// Ordered list of tensor names forming the predicted access pattern.
    pattern: Vec<String>,
    /// Index of the last accessed tensor in the pattern.
    current_index: Option<usize>,
    /// Number of tensors to prefetch ahead.
    lookahead: usize,
    /// Number of successful prefetch predictions.
    prefetch_hits: u64,
    /// Total prefetch attempts.
    prefetch_attempts: u64,
}

impl PrefetchScheduler {
    /// Create a scheduler with a given lookahead window.
    #[must_use]
    pub const fn new(lookahead: usize) -> Self {
        Self {
            pattern: Vec::new(),
            current_index: None,
            lookahead,
            prefetch_hits: 0,
            prefetch_attempts: 0,
        }
    }

    /// Set the expected access pattern (e.g., layer order).
    pub fn set_pattern(&mut self, pattern: Vec<String>) {
        self.pattern = pattern;
        self.current_index = None;
    }

    /// Record an access and return tensors to prefetch.
    pub fn on_access(&mut self, name: &str) -> Vec<String> {
        // Find where this tensor is in the pattern.
        let pos = self.pattern.iter().position(|n| n == name);

        if let Some(idx) = pos {
            // Check if this was a predicted prefetch.
            if let Some(prev) = self.current_index
                && idx == prev + 1
            {
                self.prefetch_hits += 1;
            }
            self.current_index = Some(idx);

            // Return the next `lookahead` tensors.
            let start = idx + 1;
            let end = (start + self.lookahead).min(self.pattern.len());
            let to_prefetch: Vec<String> = self.pattern[start..end].to_vec();
            self.prefetch_attempts += to_prefetch.len() as u64;
            to_prefetch
        } else {
            // Unknown tensor — can't predict.
            self.current_index = None;
            Vec::new()
        }
    }

    /// Prefetch hit rate (0.0–1.0). Returns 0.0 if no attempts.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        if self.prefetch_attempts == 0 {
            return 0.0;
        }
        self.prefetch_hits as f64 / self.prefetch_attempts as f64
    }

    /// Number of prefetch hits.
    #[must_use]
    pub const fn prefetch_hits(&self) -> u64 {
        self.prefetch_hits
    }

    /// Number of prefetch attempts.
    #[must_use]
    pub const fn prefetch_attempts(&self) -> u64 {
        self.prefetch_attempts
    }

    /// Current lookahead window size.
    #[must_use]
    pub const fn lookahead(&self) -> usize {
        self.lookahead
    }

    /// The expected access pattern.
    #[must_use]
    pub fn pattern(&self) -> &[String] {
        &self.pattern
    }
}

// ── MemoryBudgetEnforcer ─────────────────────────────────────────────────────

/// Limits total memory used by mapped tensors and evicts LRU entries
/// when the budget is exceeded.
#[derive(Debug)]
pub struct MemoryBudgetEnforcer {
    /// Maximum allowed bytes.
    budget_bytes: usize,
    /// Current usage in bytes.
    current_bytes: usize,
    /// Tracked allocations: name → size.
    allocations: HashMap<String, usize>,
    /// Access order for LRU eviction (oldest first).
    access_order: Vec<String>,
    /// Total number of evictions performed.
    evictions: u64,
}

impl MemoryBudgetEnforcer {
    /// Create an enforcer with the given byte budget.
    #[must_use]
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            current_bytes: 0,
            allocations: HashMap::new(),
            access_order: Vec::new(),
            evictions: 0,
        }
    }

    /// Try to allocate `size` bytes for `name`.
    ///
    /// Returns names of tensors that were evicted to make room.
    /// Returns an error if the single tensor exceeds the budget.
    pub fn allocate(&mut self, name: &str, size: usize) -> Result<Vec<String>, MmapError> {
        if size > self.budget_bytes {
            return Err(MmapError::BudgetExceeded { requested: size, budget: self.budget_bytes });
        }

        // If already allocated, treat as a touch.
        if self.allocations.contains_key(name) {
            self.touch(name);
            return Ok(Vec::new());
        }

        let mut evicted = Vec::new();

        // Evict until there is room.
        while self.current_bytes + size > self.budget_bytes {
            if let Some(lru) = self.evict_lru_internal() {
                evicted.push(lru);
            } else {
                break;
            }
        }

        self.allocations.insert(name.to_string(), size);
        self.current_bytes += size;
        self.access_order.push(name.to_string());
        Ok(evicted)
    }

    /// Touch a tensor (update LRU position).
    pub fn touch(&mut self, name: &str) {
        self.access_order.retain(|n| n != name);
        if self.allocations.contains_key(name) {
            self.access_order.push(name.to_string());
        }
    }

    /// Explicitly free a tensor's allocation.
    pub fn free(&mut self, name: &str) -> bool {
        if let Some(size) = self.allocations.remove(name) {
            self.current_bytes = self.current_bytes.saturating_sub(size);
            self.access_order.retain(|n| n != name);
            true
        } else {
            false
        }
    }

    fn evict_lru_internal(&mut self) -> Option<String> {
        if self.access_order.is_empty() {
            return None;
        }
        let lru = self.access_order.remove(0);
        if let Some(size) = self.allocations.remove(&lru) {
            self.current_bytes = self.current_bytes.saturating_sub(size);
            self.evictions += 1;
        }
        Some(lru)
    }

    /// Current memory usage in bytes.
    #[must_use]
    pub const fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Budget limit in bytes.
    #[must_use]
    pub const fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// Number of active allocations.
    #[must_use]
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Total evictions performed.
    #[must_use]
    pub const fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Remaining budget in bytes.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.budget_bytes.saturating_sub(self.current_bytes)
    }

    /// Whether the budget is currently exceeded.
    #[must_use]
    pub const fn is_over_budget(&self) -> bool {
        self.current_bytes > self.budget_bytes
    }
}

// ── HugePageAllocator ────────────────────────────────────────────────────────

/// Huge page sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HugePageSize {
    /// 2 MB huge page (common on `x86_64`).
    TwoMb,
    /// 1 GB huge page (for very large models).
    OneGb,
}

impl HugePageSize {
    /// Size in bytes.
    #[must_use]
    pub const fn bytes(self) -> usize {
        match self {
            Self::TwoMb => 2 * 1024 * 1024,
            Self::OneGb => 1024 * 1024 * 1024,
        }
    }
}

/// Attempts to allocate memory using huge pages for large model files.
///
/// In this CPU simulation, allocation uses `Vec<u8>` aligned to
/// the requested huge-page boundary.
#[derive(Debug)]
pub struct HugePageAllocator {
    /// Preferred huge page size.
    page_size: HugePageSize,
    /// Total bytes allocated.
    allocated_bytes: usize,
    /// Number of allocations performed.
    allocation_count: usize,
    /// Whether huge pages are available (simulated).
    available: bool,
}

impl HugePageAllocator {
    /// Create an allocator with the given page size preference.
    #[must_use]
    pub const fn new(page_size: HugePageSize) -> Self {
        Self { page_size, allocated_bytes: 0, allocation_count: 0, available: true }
    }

    /// Simulate allocating `size` bytes with huge pages.
    ///
    /// Returns a zero-filled buffer rounded up to the page boundary.
    pub fn allocate(&mut self, size: usize) -> Result<Vec<u8>, MmapError> {
        if !self.available {
            return Err(MmapError::HugePagesUnavailable);
        }
        if size == 0 {
            return Ok(Vec::new());
        }
        let page = self.page_size.bytes();
        let aligned_size = size.div_ceil(page) * page;
        let buf = vec![0u8; aligned_size];
        self.allocated_bytes += aligned_size;
        self.allocation_count += 1;
        Ok(buf)
    }

    /// Preferred page size.
    #[must_use]
    pub const fn page_size(&self) -> HugePageSize {
        self.page_size
    }

    /// Total bytes allocated.
    #[must_use]
    pub const fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Number of allocations.
    #[must_use]
    pub const fn allocation_count(&self) -> usize {
        self.allocation_count
    }

    /// Whether huge pages are reported as available.
    #[must_use]
    pub const fn is_available(&self) -> bool {
        self.available
    }

    /// Simulate huge pages becoming unavailable.
    pub const fn set_available(&mut self, available: bool) {
        self.available = available;
    }
}

// ── MmapStats ────────────────────────────────────────────────────────────────

/// Statistics for memory-mapped model loading.
#[derive(Debug)]
pub struct MmapStats {
    /// Number of simulated page faults (first-access loads).
    pub page_faults: AtomicU64,
    /// Number of tensors loaded into memory.
    pub tensors_loaded: AtomicU64,
    /// Number of tensors evicted from memory.
    pub tensors_evicted: AtomicU64,
    /// Bytes of memory currently locked.
    pub memory_locked_bytes: AtomicU64,
    /// Number of successful prefetch predictions.
    pub prefetch_hits: AtomicU64,
}

impl MmapStats {
    /// Create zeroed statistics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            page_faults: AtomicU64::new(0),
            tensors_loaded: AtomicU64::new(0),
            tensors_evicted: AtomicU64::new(0),
            memory_locked_bytes: AtomicU64::new(0),
            prefetch_hits: AtomicU64::new(0),
        }
    }

    /// Snapshot all counters.
    #[must_use]
    pub fn snapshot(&self) -> MmapStatsSnapshot {
        MmapStatsSnapshot {
            page_faults: self.page_faults.load(Ordering::Relaxed),
            tensors_loaded: self.tensors_loaded.load(Ordering::Relaxed),
            tensors_evicted: self.tensors_evicted.load(Ordering::Relaxed),
            memory_locked_bytes: self.memory_locked_bytes.load(Ordering::Relaxed),
            prefetch_hits: self.prefetch_hits.load(Ordering::Relaxed),
        }
    }
}

impl Default for MmapStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Point-in-time snapshot of [`MmapStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MmapStatsSnapshot {
    pub page_faults: u64,
    pub tensors_loaded: u64,
    pub tensors_evicted: u64,
    pub memory_locked_bytes: u64,
    pub prefetch_hits: u64,
}

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors produced by the mmap loader subsystem.
#[derive(Debug)]
pub enum MmapError {
    /// Requested region is outside file bounds.
    OutOfBounds { offset: usize, length: usize, file_size: usize },
    /// Tensor not found by name.
    TensorNotFound(String),
    /// Allocation exceeds memory budget.
    BudgetExceeded { requested: usize, budget: usize },
    /// Huge pages are not available.
    HugePagesUnavailable,
}

impl std::fmt::Display for MmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfBounds { offset, length, file_size } => {
                write!(f, "region [{offset}..{}] exceeds file size {file_size}", offset + length)
            }
            Self::TensorNotFound(name) => {
                write!(f, "tensor not found: {name}")
            }
            Self::BudgetExceeded { requested, budget } => write!(
                f,
                "allocation of {requested} bytes exceeds budget \
                 of {budget} bytes"
            ),
            Self::HugePagesUnavailable => {
                write!(f, "huge pages are not available")
            }
        }
    }
}

impl std::error::Error for MmapError {}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ───────────────────────────────────────────────────────────

    /// Build a fake model file of `size` bytes filled with sequential
    /// values (mod 256).
    fn fake_file(size: usize) -> Vec<u8> {
        #[allow(clippy::cast_possible_truncation)]
        let v: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        v
    }

    fn sample_descriptor(
        name: &str,
        offset: usize,
        shape: Vec<usize>,
        dtype: DType,
    ) -> TensorDescriptor {
        let numel: usize = shape.iter().product();
        TensorDescriptor {
            name: name.to_string(),
            shape,
            dtype,
            offset,
            byte_length: numel * dtype.byte_size(),
        }
    }

    // ── DType tests ──────────────────────────────────────────────────────

    #[test]
    fn dtype_byte_sizes() {
        assert_eq!(DType::F32.byte_size(), 4);
        assert_eq!(DType::F16.byte_size(), 2);
        assert_eq!(DType::BF16.byte_size(), 2);
        assert_eq!(DType::I8.byte_size(), 1);
        assert_eq!(DType::I2.byte_size(), 1);
        assert_eq!(DType::U8.byte_size(), 1);
    }

    // ── MmapConfig tests ─────────────────────────────────────────────────

    #[test]
    fn config_default() {
        let cfg = MmapConfig::default();
        assert!(cfg.use_mmap);
        assert!(!cfg.prefault);
        assert!(!cfg.huge_pages);
        assert!(!cfg.lock_memory);
    }

    #[test]
    fn config_custom() {
        let cfg =
            MmapConfig { use_mmap: false, prefault: true, huge_pages: true, lock_memory: true };
        assert!(!cfg.use_mmap);
        assert!(cfg.prefault);
        assert!(cfg.huge_pages);
        assert!(cfg.lock_memory);
    }

    // ── MmapRegion tests ─────────────────────────────────────────────────

    #[test]
    fn region_creation() {
        let r = MmapRegion::new(100, 200, 100);
        assert_eq!(r.offset, 100);
        assert_eq!(r.length, 200);
        assert_eq!(r.ptr(), 100);
        assert!(!r.is_locked);
    }

    #[test]
    fn region_lock_unlock() {
        let mut r = MmapRegion::new(0, 64, 0);
        assert!(!r.is_locked);
        r.lock();
        assert!(r.is_locked);
        r.unlock();
        assert!(!r.is_locked);
    }

    // ── MmapModelFile tests ──────────────────────────────────────────────

    #[test]
    fn model_file_open_empty() {
        let mf = MmapModelFile::open(vec![], MmapConfig::default());
        assert_eq!(mf.file_size(), 0);
        assert_eq!(mf.tensor_count(), 0);
        assert_eq!(mf.region_count(), 0);
    }

    #[test]
    fn model_file_open_with_data() {
        let data = fake_file(1024);
        let mf = MmapModelFile::open(data.clone(), MmapConfig::default());
        assert_eq!(mf.file_size(), 1024);
        assert_eq!(mf.raw_data(), &data[..]);
    }

    #[test]
    fn model_file_map_region() {
        let data = fake_file(512);
        let mut mf = MmapModelFile::open(data, MmapConfig::default());
        let idx = mf.map_region(0, 256).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(mf.region_count(), 1);
        let r = mf.region(idx).unwrap();
        assert_eq!(r.offset, 0);
        assert_eq!(r.length, 256);
    }

    #[test]
    fn model_file_map_region_out_of_bounds() {
        let data = fake_file(100);
        let mut mf = MmapModelFile::open(data, MmapConfig::default());
        let err = mf.map_region(50, 200).unwrap_err();
        assert!(matches!(err, MmapError::OutOfBounds { .. }));
    }

    #[test]
    fn model_file_map_multiple_regions() {
        let data = fake_file(1024);
        let mut mf = MmapModelFile::open(data, MmapConfig::default());
        let r0 = mf.map_region(0, 256).unwrap();
        let r1 = mf.map_region(256, 256).unwrap();
        let r2 = mf.map_region(512, 512).unwrap();
        assert_eq!(r0, 0);
        assert_eq!(r1, 1);
        assert_eq!(r2, 2);
        assert_eq!(mf.region_count(), 3);
    }

    #[test]
    fn model_file_lock_memory_config() {
        let data = fake_file(256);
        let cfg = MmapConfig { lock_memory: true, ..MmapConfig::default() };
        let mut mf = MmapModelFile::open(data, cfg);
        let idx = mf.map_region(0, 128).unwrap();
        assert!(mf.region(idx).unwrap().is_locked);
    }

    #[test]
    fn model_file_register_tensor() {
        let data = fake_file(256);
        let mut mf = MmapModelFile::open(data, MmapConfig::default());
        let desc = sample_descriptor("w0", 0, vec![8, 8], DType::F32);
        mf.register_tensor(desc);
        assert_eq!(mf.tensor_count(), 1);
    }

    #[test]
    fn model_file_tensor_view() {
        let data = fake_file(1024);
        let mut mf = MmapModelFile::open(data.clone(), MmapConfig::default());
        let desc = sample_descriptor("w0", 0, vec![4, 4], DType::F32);
        mf.register_tensor(desc);
        let view = mf.tensor_view("w0").unwrap();
        assert_eq!(view.name, "w0");
        assert_eq!(view.shape, vec![4, 4]);
        assert_eq!(view.dtype, DType::F32);
        assert_eq!(view.numel(), 16);
        assert_eq!(view.byte_size(), 64);
        assert_eq!(view.data.len(), 64);
        assert_eq!(view.data, &data[0..64]);
    }

    #[test]
    fn model_file_tensor_view_not_found() {
        let mf = MmapModelFile::open(fake_file(64), MmapConfig::default());
        let err = mf.tensor_view("nonexistent").unwrap_err();
        assert!(matches!(err, MmapError::TensorNotFound(_)));
    }

    #[test]
    fn model_file_tensor_view_out_of_bounds() {
        let data = fake_file(32);
        let mut mf = MmapModelFile::open(data, MmapConfig::default());
        // 8*8*4 = 256 bytes but file is only 32
        let desc = sample_descriptor("big", 0, vec![8, 8], DType::F32);
        mf.register_tensor(desc);
        let err = mf.tensor_view("big").unwrap_err();
        assert!(matches!(err, MmapError::OutOfBounds { .. }));
    }

    #[test]
    fn model_file_multiple_tensors() {
        let data = fake_file(1024);
        let mut mf = MmapModelFile::open(data.clone(), MmapConfig::default());
        let d0 = sample_descriptor("a", 0, vec![16], DType::F32); // 64 bytes
        let d1 = sample_descriptor("b", 64, vec![32], DType::F16); // 64 bytes
        let d2 = sample_descriptor("c", 128, vec![64], DType::I8); // 64 bytes
        mf.register_tensor(d0);
        mf.register_tensor(d1);
        mf.register_tensor(d2);
        assert_eq!(mf.tensor_count(), 3);

        let va = mf.tensor_view("a").unwrap();
        assert_eq!(va.data, &data[0..64]);
        let vb = mf.tensor_view("b").unwrap();
        assert_eq!(vb.data, &data[64..128]);
        let vc = mf.tensor_view("c").unwrap();
        assert_eq!(vc.data, &data[128..192]);
    }

    #[test]
    fn model_file_config_accessors() {
        let cfg =
            MmapConfig { use_mmap: false, prefault: true, huge_pages: false, lock_memory: true };
        let mf = MmapModelFile::open(vec![], cfg);
        assert!(!mf.config().use_mmap);
        assert!(mf.config().prefault);
        assert!(mf.config().lock_memory);
    }

    // ── TensorView tests ─────────────────────────────────────────────────

    #[test]
    fn tensor_view_numel_scalar() {
        let data = [0u8; 4];
        let view = TensorView { name: "s".into(), shape: vec![1], dtype: DType::F32, data: &data };
        assert_eq!(view.numel(), 1);
        assert_eq!(view.byte_size(), 4);
    }

    #[test]
    fn tensor_view_numel_matrix() {
        let data = vec![0u8; 48];
        let view =
            TensorView { name: "m".into(), shape: vec![3, 4], dtype: DType::F32, data: &data };
        assert_eq!(view.numel(), 12);
        assert_eq!(view.byte_size(), 48);
    }

    #[test]
    fn tensor_view_numel_3d() {
        let data = vec![0u8; 24];
        let view =
            TensorView { name: "t".into(), shape: vec![2, 3, 4], dtype: DType::I8, data: &data };
        assert_eq!(view.numel(), 24);
        assert_eq!(view.byte_size(), 24);
    }

    #[test]
    fn tensor_view_f16_byte_size() {
        let data = vec![0u8; 16];
        let view = TensorView { name: "h".into(), shape: vec![8], dtype: DType::F16, data: &data };
        assert_eq!(view.byte_size(), 16);
    }

    // ── LazyTensorLoader tests ───────────────────────────────────────────

    #[test]
    fn lazy_loader_new_is_empty() {
        let loader = LazyTensorLoader::new();
        assert_eq!(loader.loaded_count(), 0);
        assert_eq!(loader.descriptor_count(), 0);
    }

    #[test]
    fn lazy_loader_default() {
        let loader = LazyTensorLoader::default();
        assert_eq!(loader.loaded_count(), 0);
    }

    #[test]
    fn lazy_loader_register_and_get() {
        let file = fake_file(256);
        let mut loader = LazyTensorLoader::new();
        let desc = sample_descriptor("w", 0, vec![16], DType::F32);
        loader.register(desc, &file).unwrap();
        assert!(!loader.is_loaded("w"));

        let data = loader.get("w", &file).unwrap();
        assert_eq!(data.len(), 64);
        assert!(loader.is_loaded("w"));
        assert_eq!(loader.loaded_count(), 1);
    }

    #[test]
    fn lazy_loader_demand_paging_page_fault() {
        let file = fake_file(128);
        let mut loader = LazyTensorLoader::new();
        let desc = sample_descriptor("t", 0, vec![32], DType::I8);
        loader.register(desc, &file).unwrap();

        let snap_before = loader.stats().snapshot();
        assert_eq!(snap_before.page_faults, 0);

        let _ = loader.get("t", &file).unwrap();
        let snap_after = loader.stats().snapshot();
        assert_eq!(snap_after.page_faults, 1);
        assert_eq!(snap_after.tensors_loaded, 1);
    }

    #[test]
    fn lazy_loader_second_access_no_fault() {
        let file = fake_file(256);
        let mut loader = LazyTensorLoader::new();
        let desc = sample_descriptor("t", 0, vec![16], DType::F32);
        loader.register(desc, &file).unwrap();

        let _ = loader.get("t", &file).unwrap();
        let _ = loader.get("t", &file).unwrap();
        assert_eq!(loader.stats().snapshot().page_faults, 1);
    }

    #[test]
    fn lazy_loader_not_found() {
        let file = fake_file(64);
        let mut loader = LazyTensorLoader::new();
        let err = loader.get("missing", &file).unwrap_err();
        assert!(matches!(err, MmapError::TensorNotFound(_)));
    }

    #[test]
    fn lazy_loader_evict() {
        let file = fake_file(256);
        let mut loader = LazyTensorLoader::new();
        let desc = sample_descriptor("t", 0, vec![16], DType::F32);
        loader.register(desc, &file).unwrap();
        let _ = loader.get("t", &file).unwrap();
        assert!(loader.is_loaded("t"));

        assert!(loader.evict("t"));
        assert!(!loader.is_loaded("t"));
        assert_eq!(loader.stats().snapshot().tensors_evicted, 1);
    }

    #[test]
    fn lazy_loader_evict_not_loaded() {
        let mut loader = LazyTensorLoader::new();
        assert!(!loader.evict("nonexistent"));
    }

    #[test]
    fn lazy_loader_evict_lru() {
        let file = fake_file(512);
        let mut loader = LazyTensorLoader::new();
        let d0 = sample_descriptor("a", 0, vec![16], DType::F32);
        let d1 = sample_descriptor("b", 64, vec![16], DType::F32);
        let d2 = sample_descriptor("c", 128, vec![16], DType::F32);
        loader.register(d0, &file).unwrap();
        loader.register(d1, &file).unwrap();
        loader.register(d2, &file).unwrap();

        let _ = loader.get("a", &file).unwrap();
        let _ = loader.get("b", &file).unwrap();
        let _ = loader.get("c", &file).unwrap();

        // "a" is LRU
        let evicted = loader.evict_lru().unwrap();
        assert_eq!(evicted, "a");
        assert!(!loader.is_loaded("a"));
        assert!(loader.is_loaded("b"));
        assert!(loader.is_loaded("c"));
    }

    #[test]
    fn lazy_loader_evict_lru_empty() {
        let mut loader = LazyTensorLoader::new();
        assert!(loader.evict_lru().is_none());
    }

    #[test]
    fn lazy_loader_access_order_updates() {
        let file = fake_file(512);
        let mut loader = LazyTensorLoader::new();
        let d0 = sample_descriptor("a", 0, vec![16], DType::F32);
        let d1 = sample_descriptor("b", 64, vec![16], DType::F32);
        loader.register(d0, &file).unwrap();
        loader.register(d1, &file).unwrap();

        let _ = loader.get("a", &file).unwrap();
        let _ = loader.get("b", &file).unwrap();
        assert_eq!(loader.access_order(), &["a", "b"]);

        // Re-access "a" → moves to back
        let _ = loader.get("a", &file).unwrap();
        assert_eq!(loader.access_order(), &["b", "a"]);
    }

    #[test]
    fn lazy_loader_reload_after_eviction() {
        let file = fake_file(256);
        let mut loader = LazyTensorLoader::new();
        let desc = sample_descriptor("t", 0, vec![16], DType::F32);
        loader.register(desc, &file).unwrap();

        let _ = loader.get("t", &file).unwrap();
        loader.evict("t");
        assert!(!loader.is_loaded("t"));

        let data = loader.get("t", &file).unwrap();
        assert_eq!(data.len(), 64);
        assert!(loader.is_loaded("t"));
        assert_eq!(loader.stats().snapshot().page_faults, 2);
    }

    // ── PrefetchScheduler tests ──────────────────────────────────────────

    #[test]
    fn prefetch_empty_pattern() {
        let mut sched = PrefetchScheduler::new(2);
        let pf = sched.on_access("x");
        assert!(pf.is_empty());
    }

    #[test]
    fn prefetch_sequential_access() {
        let mut sched = PrefetchScheduler::new(2);
        sched.set_pattern(vec!["l0".into(), "l1".into(), "l2".into(), "l3".into()]);

        let pf = sched.on_access("l0");
        assert_eq!(pf, vec!["l1", "l2"]);

        let pf = sched.on_access("l1");
        assert_eq!(pf, vec!["l2", "l3"]);
    }

    #[test]
    fn prefetch_at_end_of_pattern() {
        let mut sched = PrefetchScheduler::new(3);
        sched.set_pattern(vec!["a".into(), "b".into(), "c".into()]);

        let pf = sched.on_access("b");
        assert_eq!(pf, vec!["c"]);

        let pf = sched.on_access("c");
        assert!(pf.is_empty());
    }

    #[test]
    fn prefetch_unknown_tensor() {
        let mut sched = PrefetchScheduler::new(2);
        sched.set_pattern(vec!["a".into(), "b".into()]);
        let pf = sched.on_access("unknown");
        assert!(pf.is_empty());
    }

    #[test]
    fn prefetch_hit_tracking() {
        let mut sched = PrefetchScheduler::new(1);
        sched.set_pattern(vec!["a".into(), "b".into(), "c".into(), "d".into()]);

        sched.on_access("a"); // prefetches "b"
        sched.on_access("b"); // hit! prefetches "c"
        sched.on_access("c"); // hit! prefetches "d"

        assert_eq!(sched.prefetch_hits(), 2);
    }

    #[test]
    fn prefetch_hit_rate() {
        let mut sched = PrefetchScheduler::new(1);
        sched.set_pattern(vec!["a".into(), "b".into(), "c".into()]);

        sched.on_access("a"); // prefetches "b" (1 attempt)
        sched.on_access("b"); // hit! prefetches "c" (1 attempt)
        // 2 attempts, 1 hit → 0.5
        let rate = sched.hit_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn prefetch_hit_rate_zero_attempts() {
        let sched = PrefetchScheduler::new(1);
        assert!(sched.hit_rate().abs() < f64::EPSILON);
    }

    #[test]
    fn prefetch_lookahead_window() {
        let mut sched = PrefetchScheduler::new(5);
        assert_eq!(sched.lookahead(), 5);
        sched.set_pattern(vec!["a".into(), "b".into(), "c".into()]);
        let pf = sched.on_access("a");
        // Only 2 tensors after "a", even though lookahead is 5
        assert_eq!(pf.len(), 2);
    }

    #[test]
    fn prefetch_pattern_accessor() {
        let mut sched = PrefetchScheduler::new(1);
        let pattern = vec!["x".to_string(), "y".to_string()];
        sched.set_pattern(pattern.clone());
        assert_eq!(sched.pattern(), &pattern[..]);
    }

    // ── MemoryBudgetEnforcer tests ───────────────────────────────────────

    #[test]
    fn budget_new() {
        let enf = MemoryBudgetEnforcer::new(1024);
        assert_eq!(enf.budget_bytes(), 1024);
        assert_eq!(enf.current_bytes(), 0);
        assert_eq!(enf.remaining(), 1024);
        assert!(!enf.is_over_budget());
    }

    #[test]
    fn budget_allocate_within() {
        let mut enf = MemoryBudgetEnforcer::new(1024);
        let evicted = enf.allocate("a", 256).unwrap();
        assert!(evicted.is_empty());
        assert_eq!(enf.current_bytes(), 256);
        assert_eq!(enf.allocation_count(), 1);
        assert_eq!(enf.remaining(), 768);
    }

    #[test]
    fn budget_allocate_exceeds_single() {
        let mut enf = MemoryBudgetEnforcer::new(100);
        let err = enf.allocate("big", 200).unwrap_err();
        assert!(matches!(err, MmapError::BudgetExceeded { .. }));
    }

    #[test]
    fn budget_eviction_on_pressure() {
        let mut enf = MemoryBudgetEnforcer::new(256);
        enf.allocate("a", 128).unwrap();
        enf.allocate("b", 128).unwrap();
        // Budget full; allocating "c" should evict "a" (LRU)
        let evicted = enf.allocate("c", 128).unwrap();
        assert_eq!(evicted, vec!["a"]);
        assert_eq!(enf.current_bytes(), 256);
        assert_eq!(enf.allocation_count(), 2);
    }

    #[test]
    fn budget_multiple_evictions() {
        let mut enf = MemoryBudgetEnforcer::new(256);
        enf.allocate("a", 64).unwrap();
        enf.allocate("b", 64).unwrap();
        enf.allocate("c", 64).unwrap();
        enf.allocate("d", 64).unwrap();
        // Need 256 bytes, must evict all four
        let evicted = enf.allocate("e", 256).unwrap();
        assert_eq!(evicted.len(), 4);
        assert_eq!(enf.allocation_count(), 1);
    }

    #[test]
    fn budget_touch_updates_lru() {
        let mut enf = MemoryBudgetEnforcer::new(256);
        enf.allocate("a", 128).unwrap();
        enf.allocate("b", 128).unwrap();
        // Touch "a" so "b" becomes LRU
        enf.touch("a");
        let evicted = enf.allocate("c", 128).unwrap();
        assert_eq!(evicted, vec!["b"]);
    }

    #[test]
    fn budget_free() {
        let mut enf = MemoryBudgetEnforcer::new(512);
        enf.allocate("a", 256).unwrap();
        assert!(enf.free("a"));
        assert_eq!(enf.current_bytes(), 0);
        assert_eq!(enf.allocation_count(), 0);
    }

    #[test]
    fn budget_free_nonexistent() {
        let mut enf = MemoryBudgetEnforcer::new(512);
        assert!(!enf.free("ghost"));
    }

    #[test]
    fn budget_double_allocate_is_touch() {
        let mut enf = MemoryBudgetEnforcer::new(512);
        enf.allocate("a", 128).unwrap();
        let evicted = enf.allocate("a", 128).unwrap();
        assert!(evicted.is_empty());
        assert_eq!(enf.current_bytes(), 128);
        assert_eq!(enf.allocation_count(), 1);
    }

    #[test]
    fn budget_evictions_counter() {
        let mut enf = MemoryBudgetEnforcer::new(128);
        enf.allocate("a", 128).unwrap();
        enf.allocate("b", 128).unwrap(); // evicts "a"
        assert_eq!(enf.evictions(), 1);
    }

    #[test]
    fn budget_zero_budget() {
        let mut enf = MemoryBudgetEnforcer::new(0);
        // Budget 0 means any non-zero alloc exceeds budget
        let err = enf.allocate("a", 1).unwrap_err();
        assert!(matches!(err, MmapError::BudgetExceeded { .. }));
    }

    #[test]
    fn budget_zero_size_allocation() {
        let mut enf = MemoryBudgetEnforcer::new(0);
        // Zero-size allocation should succeed even with zero budget
        let evicted = enf.allocate("empty", 0).unwrap();
        assert!(evicted.is_empty());
        assert_eq!(enf.current_bytes(), 0);
    }

    // ── HugePageAllocator tests ──────────────────────────────────────────

    #[test]
    fn huge_page_sizes() {
        assert_eq!(HugePageSize::TwoMb.bytes(), 2 * 1024 * 1024);
        assert_eq!(HugePageSize::OneGb.bytes(), 1024 * 1024 * 1024);
    }

    #[test]
    fn huge_page_allocate() {
        let mut alloc = HugePageAllocator::new(HugePageSize::TwoMb);
        let buf = alloc.allocate(1000).unwrap();
        // Rounded up to 2 MB
        assert_eq!(buf.len(), 2 * 1024 * 1024);
        assert_eq!(alloc.allocation_count(), 1);
        assert_eq!(alloc.allocated_bytes(), 2 * 1024 * 1024);
    }

    #[test]
    fn huge_page_allocate_exact() {
        let mut alloc = HugePageAllocator::new(HugePageSize::TwoMb);
        let page = HugePageSize::TwoMb.bytes();
        let buf = alloc.allocate(page).unwrap();
        assert_eq!(buf.len(), page);
    }

    #[test]
    fn huge_page_allocate_zero() {
        let mut alloc = HugePageAllocator::new(HugePageSize::TwoMb);
        let buf = alloc.allocate(0).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn huge_page_unavailable() {
        let mut alloc = HugePageAllocator::new(HugePageSize::TwoMb);
        alloc.set_available(false);
        assert!(!alloc.is_available());
        let err = alloc.allocate(1024).unwrap_err();
        assert!(matches!(err, MmapError::HugePagesUnavailable));
    }

    #[test]
    fn huge_page_multiple_allocations() {
        let mut alloc = HugePageAllocator::new(HugePageSize::TwoMb);
        alloc.allocate(100).unwrap();
        alloc.allocate(200).unwrap();
        assert_eq!(alloc.allocation_count(), 2);
        assert_eq!(alloc.allocated_bytes(), 2 * 2 * 1024 * 1024);
    }

    #[test]
    fn huge_page_1gb_size() {
        let mut alloc = HugePageAllocator::new(HugePageSize::OneGb);
        assert_eq!(alloc.page_size(), HugePageSize::OneGb);
        // Don't actually allocate 1 GB in tests — just check page_size
        assert_eq!(alloc.page_size().bytes(), 1024 * 1024 * 1024);
        // Allocate 0 to exercise the path without huge memory use
        let buf = alloc.allocate(0).unwrap();
        assert!(buf.is_empty());
    }

    // ── MmapStats tests ──────────────────────────────────────────────────

    #[test]
    fn stats_new_zeroed() {
        let stats = MmapStats::new();
        let snap = stats.snapshot();
        assert_eq!(snap.page_faults, 0);
        assert_eq!(snap.tensors_loaded, 0);
        assert_eq!(snap.tensors_evicted, 0);
        assert_eq!(snap.memory_locked_bytes, 0);
        assert_eq!(snap.prefetch_hits, 0);
    }

    #[test]
    fn stats_default() {
        let stats = MmapStats::default();
        assert_eq!(stats.snapshot().page_faults, 0);
    }

    #[test]
    fn stats_increment_and_snapshot() {
        let stats = MmapStats::new();
        stats.page_faults.fetch_add(5, Ordering::Relaxed);
        stats.tensors_loaded.fetch_add(3, Ordering::Relaxed);
        stats.tensors_evicted.fetch_add(1, Ordering::Relaxed);
        stats.memory_locked_bytes.fetch_add(4096, Ordering::Relaxed);
        stats.prefetch_hits.fetch_add(2, Ordering::Relaxed);

        let snap = stats.snapshot();
        assert_eq!(snap.page_faults, 5);
        assert_eq!(snap.tensors_loaded, 3);
        assert_eq!(snap.tensors_evicted, 1);
        assert_eq!(snap.memory_locked_bytes, 4096);
        assert_eq!(snap.prefetch_hits, 2);
    }

    #[test]
    fn stats_snapshot_equality() {
        let a = MmapStatsSnapshot {
            page_faults: 1,
            tensors_loaded: 2,
            tensors_evicted: 3,
            memory_locked_bytes: 4,
            prefetch_hits: 5,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── Error display tests ──────────────────────────────────────────────

    #[test]
    fn error_display_out_of_bounds() {
        let err = MmapError::OutOfBounds { offset: 100, length: 200, file_size: 150 };
        let msg = err.to_string();
        assert!(msg.contains("300"));
        assert!(msg.contains("150"));
    }

    #[test]
    fn error_display_tensor_not_found() {
        let err = MmapError::TensorNotFound("foo".into());
        assert!(err.to_string().contains("foo"));
    }

    #[test]
    fn error_display_budget_exceeded() {
        let err = MmapError::BudgetExceeded { requested: 1024, budget: 512 };
        let msg = err.to_string();
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn error_display_huge_pages() {
        let err = MmapError::HugePagesUnavailable;
        assert!(err.to_string().contains("huge pages"));
    }

    // ── Integration-style tests ──────────────────────────────────────────

    #[test]
    fn integration_lazy_load_with_prefetch() {
        let file = fake_file(1024);
        let mut loader = LazyTensorLoader::new();
        let mut sched = PrefetchScheduler::new(1);

        let names: Vec<String> = (0..4).map(|i| format!("layer{i}")).collect();
        for (i, name) in names.iter().enumerate() {
            let desc = sample_descriptor(name, i * 64, vec![16], DType::F32);
            loader.register(desc, &file).unwrap();
        }
        sched.set_pattern(names.clone());

        // Simulate sequential access
        for name in &names {
            let _prefetch = sched.on_access(name);
            let _ = loader.get(name, &file).unwrap();
        }

        assert_eq!(loader.loaded_count(), 4);
        assert_eq!(sched.prefetch_hits(), 3); // l1, l2, l3 hit
    }

    #[test]
    fn integration_budget_enforcer_with_loader() {
        let file = fake_file(1024);
        let mut loader = LazyTensorLoader::new();
        let mut budget = MemoryBudgetEnforcer::new(128);

        let d0 = sample_descriptor("a", 0, vec![16], DType::F32);
        let d1 = sample_descriptor("b", 64, vec![16], DType::F32);
        let d2 = sample_descriptor("c", 128, vec![16], DType::F32);
        loader.register(d0, &file).unwrap();
        loader.register(d1, &file).unwrap();
        loader.register(d2, &file).unwrap();

        // Load "a" and "b" (fills budget)
        let data_a = loader.get("a", &file).unwrap();
        budget.allocate("a", data_a.len()).unwrap();
        let data_b = loader.get("b", &file).unwrap();
        budget.allocate("b", data_b.len()).unwrap();
        assert_eq!(budget.current_bytes(), 128);

        // Load "c" → evicts "a"
        let data_c = loader.get("c", &file).unwrap();
        let evicted = budget.allocate("c", data_c.len()).unwrap();
        assert_eq!(evicted, vec!["a"]);

        // Evict from loader too
        for name in &evicted {
            loader.evict(name);
        }
        assert!(!loader.is_loaded("a"));
        assert!(loader.is_loaded("b"));
        assert!(loader.is_loaded("c"));
    }

    #[test]
    fn integration_full_pipeline() {
        let file = fake_file(2048);
        let config =
            MmapConfig { use_mmap: true, prefault: true, huge_pages: false, lock_memory: false };
        let mut model = MmapModelFile::open(file, config);

        // Register tensors
        let layer_names: Vec<String> = (0..8).map(|i| format!("layer.{i}.weight")).collect();
        for (i, name) in layer_names.iter().enumerate() {
            let desc = sample_descriptor(name, i * 64, vec![4, 4], DType::F32);
            model.register_tensor(desc.clone());
        }

        // Verify all tensor views work
        for name in &layer_names {
            let view = model.tensor_view(name).unwrap();
            assert_eq!(view.shape, vec![4, 4]);
            assert_eq!(view.dtype, DType::F32);
            assert_eq!(view.byte_size(), 64);
        }

        assert_eq!(model.tensor_count(), 8);
    }

    #[test]
    fn integration_prefetch_accuracy_nonsequential() {
        let mut sched = PrefetchScheduler::new(1);
        sched.set_pattern(vec!["a".into(), "b".into(), "c".into(), "d".into()]);

        // Access out of order: a, c (skip b)
        sched.on_access("a"); // prefetches "b"
        sched.on_access("c"); // not a hit (expected "b", got "c")

        // hits should be 0 since we skipped "b"
        assert_eq!(sched.prefetch_hits(), 0);
    }

    #[test]
    fn single_tensor_file() {
        let data = vec![42u8; 16];
        let mut mf = MmapModelFile::open(data, MmapConfig::default());
        let desc = TensorDescriptor {
            name: "only".into(),
            shape: vec![4],
            dtype: DType::F32,
            offset: 0,
            byte_length: 16,
        };
        mf.register_tensor(desc);
        let view = mf.tensor_view("only").unwrap();
        assert_eq!(view.data, &[42u8; 16]);
    }

    #[test]
    fn region_exact_file_size() {
        let data = fake_file(64);
        let mut mf = MmapModelFile::open(data, MmapConfig::default());
        // Map entire file
        let idx = mf.map_region(0, 64).unwrap();
        let r = mf.region(idx).unwrap();
        assert_eq!(r.length, 64);
    }

    #[test]
    fn region_zero_length() {
        let data = fake_file(64);
        let mut mf = MmapModelFile::open(data, MmapConfig::default());
        let idx = mf.map_region(32, 0).unwrap();
        let r = mf.region(idx).unwrap();
        assert_eq!(r.length, 0);
    }
}
