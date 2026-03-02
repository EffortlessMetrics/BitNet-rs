//! CUDA constant memory pool management with LUT loading and cache optimization.
//!
//! # Overview
//!
//! CUDA constant memory is a small (typically 64 KB) read-only region that is
//! cached in a dedicated on-chip cache, providing broadcast capability — a
//! single read can satisfy all threads in a warp simultaneously.  This module
//! provides:
//!
//! - [`ConstantMemoryPool`] — pool-based allocation/deallocation with usage tracking
//! - [`LutEntry`] / [`LutKind`] — lookup table loading (quantization scales, RoPE
//!   sin/cos tables)
//! - [`InvalidationStrategy`] — cache invalidation policies (LRU, generation-based,
//!   explicit)
//! - [`BroadcastBinding`] — parameter broadcast via constant memory
//! - [`ConstantMemoryConfig`] — size limits and overflow detection
//! - [`ConfigParameterStore`] — hyperparameter and shape-info storage
//! - [`ReadOnlyCacheHint`] — optimization hints for the read-only data cache
//! - [`ConstantMemoryProfiler`] — hit-rate estimation and access tracking
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations are provided for testing on non-GPU hosts.

use bitnet_common::{KernelError, Result};
use std::collections::HashMap;

// ── Constants ────────────────────────────────────────────────────────

/// Default CUDA constant memory size (64 KB).
pub const DEFAULT_CONSTANT_MEMORY_SIZE: usize = 64 * 1024;

/// Minimum alignment for constant memory allocations (16 bytes for float4).
pub const CONSTANT_MEMORY_ALIGNMENT: usize = 16;

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for the constant memory pool.
#[derive(Debug, Clone)]
pub struct ConstantMemoryConfig {
    /// Total constant memory size in bytes (GPU hardware limit).
    pub total_size: usize,
    /// Alignment requirement in bytes (must be a power of two).
    pub alignment: usize,
    /// Reserved bytes for system/driver use.
    pub reserved_bytes: usize,
    /// Whether to enable profiling of access patterns.
    pub enable_profiling: bool,
}

impl Default for ConstantMemoryConfig {
    fn default() -> Self {
        Self {
            total_size: DEFAULT_CONSTANT_MEMORY_SIZE,
            alignment: CONSTANT_MEMORY_ALIGNMENT,
            reserved_bytes: 0,
            enable_profiling: false,
        }
    }
}

impl ConstantMemoryConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<()> {
        if self.total_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "total_size must be non-zero".into(),
            }
            .into());
        }
        if !self.alignment.is_power_of_two() {
            return Err(KernelError::InvalidArguments {
                reason: "alignment must be a power of two".into(),
            }
            .into());
        }
        if self.reserved_bytes >= self.total_size {
            return Err(KernelError::InvalidArguments {
                reason: "reserved_bytes must be less than total_size".into(),
            }
            .into());
        }
        Ok(())
    }

    /// Usable capacity after subtracting reserved bytes.
    pub fn usable_capacity(&self) -> usize {
        self.total_size - self.reserved_bytes
    }
}

// ── Allocation handle ────────────────────────────────────────────────

/// Unique handle for a constant memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstantSlotId(u64);

/// A region within constant memory.
#[derive(Debug, Clone)]
pub struct ConstantSlot {
    /// Unique identifier.
    pub id: ConstantSlotId,
    /// Byte offset from constant memory base.
    pub offset: usize,
    /// Size in bytes.
    pub size: usize,
    /// Human-readable label.
    pub label: String,
    /// Generation counter for invalidation.
    pub generation: u64,
}

// ── Lookup table kinds ───────────────────────────────────────────────

/// Kind of lookup table stored in constant memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LutKind {
    /// Quantization scale factors (per-block scales for I2_S / TL1 / TL2).
    QuantizationScales,
    /// RoPE cosine table.
    RopeCosTable,
    /// RoPE sine table.
    RopeSinTable,
    /// Dequantization mapping (2-bit → signed value).
    DequantMap,
    /// Custom user-defined table.
    Custom,
}

impl std::fmt::Display for LutKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QuantizationScales => write!(f, "quant_scales"),
            Self::RopeCosTable => write!(f, "rope_cos"),
            Self::RopeSinTable => write!(f, "rope_sin"),
            Self::DequantMap => write!(f, "dequant_map"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

/// A lookup table entry loaded into constant memory.
#[derive(Debug, Clone)]
pub struct LutEntry {
    /// Slot backing this LUT.
    pub slot_id: ConstantSlotId,
    /// Kind of lookup table.
    pub kind: LutKind,
    /// Number of elements in the table.
    pub num_elements: usize,
    /// Size per element in bytes.
    pub element_size: usize,
}

impl LutEntry {
    /// Total size in bytes.
    pub fn total_bytes(&self) -> usize {
        self.num_elements * self.element_size
    }
}

// ── Cache invalidation ───────────────────────────────────────────────

/// Strategy for invalidating constant memory cache entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidationStrategy {
    /// Least-recently-used eviction.
    Lru,
    /// Evict entries whose generation is older than the current generation.
    GenerationBased,
    /// Manual / explicit invalidation only.
    Explicit,
}

// ── Broadcast binding ────────────────────────────────────────────────

/// A parameter broadcast binding — shares a value across all GPU threads
/// via constant memory.
#[derive(Debug, Clone)]
pub struct BroadcastBinding {
    /// Slot backing this broadcast.
    pub slot_id: ConstantSlotId,
    /// Human-readable parameter name.
    pub name: String,
    /// Size in bytes.
    pub size: usize,
}

// ── Read-only cache hints ────────────────────────────────────────────

/// Optimization hints for the CUDA read-only data cache (`__ldg`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReadOnlyCacheHint {
    /// Use the texture / L1 read-only path (`__ldg`).
    #[default]
    UseReadOnlyCache,
    /// Use the default L1/L2 cache path.
    DefaultCache,
    /// Bypass caching entirely (streaming access pattern).
    Streaming,
}

// ── Config parameter storage ─────────────────────────────────────────

/// Type of a configuration parameter stored in constant memory.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    /// 32-bit unsigned integer.
    U32(u32),
    /// 32-bit float.
    F32(f32),
    /// Pair of u32 (e.g. shape dimensions).
    Pair(u32, u32),
    /// Arbitrary byte blob.
    Bytes(Vec<u8>),
}

impl ConfigValue {
    /// Size of this value in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::U32(_) => 4,
            Self::F32(_) => 4,
            Self::Pair(_, _) => 8,
            Self::Bytes(b) => b.len(),
        }
    }
}

/// Storage for hyperparameters and shape information in constant memory.
#[derive(Debug)]
pub struct ConfigParameterStore {
    params: HashMap<String, (ConstantSlotId, ConfigValue)>,
}

impl ConfigParameterStore {
    /// Create a new parameter store (internal — callers use
    /// [`ConstantMemoryPool::create_config_store`]).
    fn new() -> Self {
        Self { params: HashMap::new() }
    }

    /// Number of stored parameters.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Look up a parameter by name.
    pub fn get(&self, name: &str) -> Option<&ConfigValue> {
        self.params.get(name).map(|(_, v)| v)
    }

    /// List all parameter names.
    pub fn names(&self) -> Vec<&str> {
        self.params.keys().map(String::as_str).collect()
    }

    /// Total bytes consumed by all stored parameters.
    pub fn total_bytes(&self) -> usize {
        self.params.values().map(|(_, v)| v.size_bytes()).sum()
    }
}

// ── Profiler ─────────────────────────────────────────────────────────

/// Access-pattern profiling for constant memory.
#[derive(Debug, Clone, Default)]
pub struct ConstantMemoryProfiler {
    /// Per-slot access counts.
    access_counts: HashMap<ConstantSlotId, u64>,
    /// Total number of accesses recorded.
    total_accesses: u64,
    /// Total number of cache-miss estimates.
    estimated_misses: u64,
}

impl ConstantMemoryProfiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        Self { access_counts: HashMap::new(), total_accesses: 0, estimated_misses: 0 }
    }

    /// Record an access to `slot`.
    pub fn record_access(&mut self, slot: ConstantSlotId) {
        *self.access_counts.entry(slot).or_insert(0) += 1;
        self.total_accesses += 1;
    }

    /// Record an estimated cache miss.
    pub fn record_miss(&mut self) {
        self.estimated_misses += 1;
    }

    /// Total accesses recorded.
    pub fn total_accesses(&self) -> u64 {
        self.total_accesses
    }

    /// Total estimated misses.
    pub fn total_misses(&self) -> u64 {
        self.estimated_misses
    }

    /// Estimated hit rate in [0.0, 1.0].  Returns 1.0 if no accesses.
    pub fn hit_rate(&self) -> f64 {
        if self.total_accesses == 0 {
            return 1.0;
        }
        let hits = self.total_accesses.saturating_sub(self.estimated_misses);
        hits as f64 / self.total_accesses as f64
    }

    /// Access count for a specific slot.
    pub fn slot_accesses(&self, slot: ConstantSlotId) -> u64 {
        self.access_counts.get(&slot).copied().unwrap_or(0)
    }

    /// Return the hottest slot (most accesses).
    pub fn hottest_slot(&self) -> Option<(ConstantSlotId, u64)> {
        self.access_counts.iter().max_by_key(|&(_, &c)| c).map(|(&id, &c)| (id, c))
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        self.access_counts.clear();
        self.total_accesses = 0;
        self.estimated_misses = 0;
    }
}

// ── Pool statistics ──────────────────────────────────────────────────

/// Snapshot of constant memory pool usage.
#[derive(Debug, Clone)]
pub struct ConstantMemoryStats {
    /// Total capacity in bytes (hardware limit minus reserved).
    pub capacity: usize,
    /// Bytes currently allocated.
    pub used: usize,
    /// Bytes available.
    pub free: usize,
    /// Number of live allocations.
    pub num_slots: usize,
    /// Number of loaded LUTs.
    pub num_luts: usize,
    /// Number of broadcast bindings.
    pub num_broadcasts: usize,
    /// Utilisation ratio in [0.0, 1.0].
    pub utilisation: f64,
    /// Whether the pool is at or near capacity (>95%).
    pub near_overflow: bool,
}

// ── Constant memory pool ─────────────────────────────────────────────

/// CUDA constant memory pool with allocation tracking, LUT loading,
/// invalidation, broadcast support, and profiling.
///
/// On CPU builds this operates on simulated offsets — no actual device
/// memory is touched.
#[derive(Debug)]
pub struct ConstantMemoryPool {
    config: ConstantMemoryConfig,
    /// Monotonically increasing slot-ID counter.
    next_id: u64,
    /// Current generation for invalidation.
    generation: u64,
    /// Active slots, keyed by ID.
    slots: HashMap<ConstantSlotId, ConstantSlot>,
    /// LRU ordering — front = least-recently-used.
    lru_order: Vec<ConstantSlotId>,
    /// Active invalidation strategy.
    invalidation: InvalidationStrategy,
    /// Loaded LUTs.
    lut_entries: HashMap<ConstantSlotId, LutEntry>,
    /// Broadcast bindings.
    broadcasts: HashMap<ConstantSlotId, BroadcastBinding>,
    /// Cache hint per slot.
    cache_hints: HashMap<ConstantSlotId, ReadOnlyCacheHint>,
    /// Profiler (always present; recording controlled by config flag).
    profiler: ConstantMemoryProfiler,
    /// Bytes currently in use.
    bytes_used: usize,
}

impl ConstantMemoryPool {
    /// Create a new pool with the given configuration.
    pub fn new(config: ConstantMemoryConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            next_id: 0,
            generation: 0,
            slots: HashMap::new(),
            lru_order: Vec::new(),
            invalidation: InvalidationStrategy::Lru,
            lut_entries: HashMap::new(),
            broadcasts: HashMap::new(),
            cache_hints: HashMap::new(),
            profiler: ConstantMemoryProfiler::new(),
            bytes_used: 0,
        })
    }

    /// Create a pool with default 64 KB configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(ConstantMemoryConfig::default())
    }

    // ── Allocation ───────────────────────────────────────────────────

    /// Align `size` up to the configured alignment boundary.
    fn align_up(&self, size: usize) -> usize {
        let mask = self.config.alignment - 1;
        (size + mask) & !mask
    }

    /// Allocate a slot of `size` bytes with the given label.
    ///
    /// Returns an error if the allocation would overflow constant memory.
    pub fn allocate(&mut self, size: usize, label: impl Into<String>) -> Result<ConstantSlot> {
        let label = label.into();
        if size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "allocation size must be non-zero".into(),
            }
            .into());
        }

        let aligned = self.align_up(size);
        let capacity = self.config.usable_capacity();

        if self.bytes_used + aligned > capacity {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "constant memory overflow: requested {} bytes (aligned {}), \
                     used {}/{} bytes",
                    size, aligned, self.bytes_used, capacity,
                ),
            }
            .into());
        }

        let id = ConstantSlotId(self.next_id);
        self.next_id += 1;

        let offset = self.bytes_used;
        let slot = ConstantSlot { id, offset, size: aligned, label, generation: self.generation };

        self.slots.insert(id, slot.clone());
        self.lru_order.push(id);
        self.bytes_used += aligned;

        Ok(slot)
    }

    /// Deallocate a previously allocated slot.
    pub fn deallocate(&mut self, id: ConstantSlotId) -> Result<()> {
        let slot = self.slots.remove(&id).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("unknown slot id {:?}", id),
        })?;

        self.bytes_used = self.bytes_used.saturating_sub(slot.size);
        self.lru_order.retain(|&s| s != id);
        self.lut_entries.remove(&id);
        self.broadcasts.remove(&id);
        self.cache_hints.remove(&id);

        Ok(())
    }

    /// Number of live allocations.
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Bytes currently in use.
    pub fn bytes_used(&self) -> usize {
        self.bytes_used
    }

    /// Bytes still available.
    pub fn bytes_free(&self) -> usize {
        self.config.usable_capacity().saturating_sub(self.bytes_used)
    }

    /// Whether the pool is completely full.
    pub fn is_full(&self) -> bool {
        self.bytes_used >= self.config.usable_capacity()
    }

    // ── LUT loading ──────────────────────────────────────────────────

    /// Load a lookup table into constant memory.
    ///
    /// `data` is the raw table content.  Returns the [`LutEntry`] handle.
    pub fn load_lut(
        &mut self,
        kind: LutKind,
        data: &[f32],
        label: impl Into<String>,
    ) -> Result<LutEntry> {
        let total = std::mem::size_of_val(data);
        let slot = self.allocate(total, label)?;
        let entry = LutEntry {
            slot_id: slot.id,
            kind,
            num_elements: data.len(),
            element_size: std::mem::size_of::<f32>(),
        };
        self.lut_entries.insert(slot.id, entry.clone());
        Ok(entry)
    }

    /// Load a byte-level lookup table (e.g. dequant map).
    pub fn load_lut_bytes(
        &mut self,
        kind: LutKind,
        data: &[u8],
        label: impl Into<String>,
    ) -> Result<LutEntry> {
        let slot = self.allocate(data.len(), label)?;
        let entry = LutEntry { slot_id: slot.id, kind, num_elements: data.len(), element_size: 1 };
        self.lut_entries.insert(slot.id, entry.clone());
        Ok(entry)
    }

    /// Number of loaded LUTs.
    pub fn num_luts(&self) -> usize {
        self.lut_entries.len()
    }

    /// Look up a LUT entry by slot ID.
    pub fn get_lut(&self, id: ConstantSlotId) -> Option<&LutEntry> {
        self.lut_entries.get(&id)
    }

    /// Find LUT entries by kind.
    pub fn find_luts_by_kind(&self, kind: LutKind) -> Vec<&LutEntry> {
        self.lut_entries.values().filter(|e| e.kind == kind).collect()
    }

    // ── Invalidation ─────────────────────────────────────────────────

    /// Set the invalidation strategy.
    pub fn set_invalidation_strategy(&mut self, strategy: InvalidationStrategy) {
        self.invalidation = strategy;
    }

    /// Current invalidation strategy.
    pub fn invalidation_strategy(&self) -> InvalidationStrategy {
        self.invalidation
    }

    /// Advance the generation counter (for generation-based invalidation).
    pub fn advance_generation(&mut self) {
        self.generation += 1;
    }

    /// Current generation.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Evict one entry according to the active invalidation strategy.
    ///
    /// Returns the ID of the evicted slot, or `None` if the pool is empty.
    pub fn evict_one(&mut self) -> Result<Option<ConstantSlotId>> {
        match self.invalidation {
            InvalidationStrategy::Lru => self.evict_lru(),
            InvalidationStrategy::GenerationBased => self.evict_by_generation(),
            InvalidationStrategy::Explicit => Ok(None),
        }
    }

    /// Evict the least-recently-used slot.
    fn evict_lru(&mut self) -> Result<Option<ConstantSlotId>> {
        if let Some(&id) = self.lru_order.first() {
            self.deallocate(id)?;
            return Ok(Some(id));
        }
        Ok(None)
    }

    /// Evict the oldest-generation slot.
    fn evict_by_generation(&mut self) -> Result<Option<ConstantSlotId>> {
        let oldest = self.slots.values().min_by_key(|s| s.generation).map(|s| s.id);
        if let Some(id) = oldest {
            self.deallocate(id)?;
            return Ok(Some(id));
        }
        Ok(None)
    }

    /// Evict all entries with generation older than `min_generation`.
    pub fn evict_older_than(&mut self, min_generation: u64) -> Result<usize> {
        let to_evict: Vec<ConstantSlotId> =
            self.slots.values().filter(|s| s.generation < min_generation).map(|s| s.id).collect();
        let count = to_evict.len();
        for id in to_evict {
            self.deallocate(id)?;
        }
        Ok(count)
    }

    /// Invalidate (deallocate) all entries.
    pub fn invalidate_all(&mut self) -> Result<usize> {
        let ids: Vec<ConstantSlotId> = self.slots.keys().copied().collect();
        let count = ids.len();
        for id in ids {
            self.deallocate(id)?;
        }
        Ok(count)
    }

    /// Touch a slot to move it to the back of the LRU list.
    pub fn touch(&mut self, id: ConstantSlotId) {
        if self.slots.contains_key(&id) {
            self.lru_order.retain(|&s| s != id);
            self.lru_order.push(id);
            if self.config.enable_profiling {
                self.profiler.record_access(id);
            }
        }
    }

    // ── Broadcast ────────────────────────────────────────────────────

    /// Create a broadcast binding — shares a parameter value across all
    /// GPU threads via constant memory.
    pub fn create_broadcast(
        &mut self,
        name: impl Into<String>,
        size: usize,
    ) -> Result<BroadcastBinding> {
        let name = name.into();
        let slot = self.allocate(size, format!("broadcast:{}", &name))?;
        let binding = BroadcastBinding { slot_id: slot.id, name, size };
        self.broadcasts.insert(slot.id, binding.clone());
        Ok(binding)
    }

    /// Number of active broadcast bindings.
    pub fn num_broadcasts(&self) -> usize {
        self.broadcasts.len()
    }

    /// Look up a broadcast by slot ID.
    pub fn get_broadcast(&self, id: ConstantSlotId) -> Option<&BroadcastBinding> {
        self.broadcasts.get(&id)
    }

    /// Find a broadcast binding by name.
    pub fn find_broadcast_by_name(&self, name: &str) -> Option<&BroadcastBinding> {
        self.broadcasts.values().find(|b| b.name == name)
    }

    // ── Cache hints ──────────────────────────────────────────────────

    /// Set a read-only cache hint for a slot.
    pub fn set_cache_hint(&mut self, id: ConstantSlotId, hint: ReadOnlyCacheHint) -> Result<()> {
        if !self.slots.contains_key(&id) {
            return Err(KernelError::InvalidArguments {
                reason: format!("unknown slot id {:?}", id),
            }
            .into());
        }
        self.cache_hints.insert(id, hint);
        Ok(())
    }

    /// Get the cache hint for a slot.
    pub fn cache_hint(&self, id: ConstantSlotId) -> ReadOnlyCacheHint {
        self.cache_hints.get(&id).copied().unwrap_or_default()
    }

    // ── Config parameter store ───────────────────────────────────────

    /// Create a [`ConfigParameterStore`] backed by this pool.
    pub fn create_config_store(&self) -> ConfigParameterStore {
        ConfigParameterStore::new()
    }

    /// Store a configuration parameter in constant memory.
    pub fn store_config(
        &mut self,
        store: &mut ConfigParameterStore,
        name: impl Into<String>,
        value: ConfigValue,
    ) -> Result<ConstantSlotId> {
        let name = name.into();
        let size = value.size_bytes();
        let slot = self.allocate(size, format!("config:{}", &name))?;
        let id = slot.id;
        store.params.insert(name, (id, value));
        Ok(id)
    }

    // ── Overflow detection ───────────────────────────────────────────

    /// Check whether allocating `size` bytes would overflow.
    pub fn would_overflow(&self, size: usize) -> bool {
        let aligned = self.align_up(size);
        self.bytes_used + aligned > self.config.usable_capacity()
    }

    /// Utilisation ratio in [0.0, 1.0].
    pub fn utilisation(&self) -> f64 {
        let cap = self.config.usable_capacity();
        if cap == 0 {
            return 0.0;
        }
        self.bytes_used as f64 / cap as f64
    }

    /// Whether the pool is near capacity (>95% used).
    pub fn near_overflow(&self) -> bool {
        self.utilisation() > 0.95
    }

    // ── Profiling ────────────────────────────────────────────────────

    /// Get a reference to the profiler.
    pub fn profiler(&self) -> &ConstantMemoryProfiler {
        &self.profiler
    }

    /// Get a mutable reference to the profiler.
    pub fn profiler_mut(&mut self) -> &mut ConstantMemoryProfiler {
        &mut self.profiler
    }

    /// Record an access to `slot` (delegates to profiler).
    pub fn record_access(&mut self, id: ConstantSlotId) {
        self.touch(id);
    }

    /// Record an estimated cache miss.
    pub fn record_miss(&mut self) {
        self.profiler.record_miss();
    }

    // ── Statistics ───────────────────────────────────────────────────

    /// Snapshot of current pool statistics.
    pub fn stats(&self) -> ConstantMemoryStats {
        let capacity = self.config.usable_capacity();
        ConstantMemoryStats {
            capacity,
            used: self.bytes_used,
            free: capacity.saturating_sub(self.bytes_used),
            num_slots: self.slots.len(),
            num_luts: self.lut_entries.len(),
            num_broadcasts: self.broadcasts.len(),
            utilisation: self.utilisation(),
            near_overflow: self.near_overflow(),
        }
    }

    /// Get a reference to the current configuration.
    pub fn config(&self) -> &ConstantMemoryConfig {
        &self.config
    }

    /// Get slot info by ID.
    pub fn get_slot(&self, id: ConstantSlotId) -> Option<&ConstantSlot> {
        self.slots.get(&id)
    }

    /// Iterate over all live slots.
    pub fn slots(&self) -> impl Iterator<Item = &ConstantSlot> {
        self.slots.values()
    }
}

// ── CUDA kernel source ───────────────────────────────────────────────

/// CUDA kernel source for constant memory declarations and accessors.
///
/// This template declares a 64 KB constant memory region and provides
/// helper functions for loading LUTs and broadcasting parameters.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const CONSTANT_MEMORY_KERNEL_SRC: &str = r#"
// Constant memory region (64 KB max)
__constant__ unsigned char c_const_mem[65536];

// Load a float LUT element from constant memory at the given byte offset.
__device__ __forceinline__ float const_lut_f32(int offset, int idx) {
    return ((const float*)(c_const_mem + offset))[idx];
}

// Load a broadcast float parameter from constant memory.
__device__ __forceinline__ float const_broadcast_f32(int offset) {
    return *((const float*)(c_const_mem + offset));
}

// Load a broadcast uint32 parameter from constant memory.
__device__ __forceinline__ unsigned int const_broadcast_u32(int offset) {
    return *((const unsigned int*)(c_const_mem + offset));
}

// Prefetch constant memory (hint to the compiler for read-only data cache).
__device__ __forceinline__ void const_prefetch(const void* ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}
"#;

// ── Precompute helpers (CPU) ─────────────────────────────────────────

/// Precompute a RoPE cosine/sine table for the given head dimension and
/// max sequence length.  Returns `(cos_table, sin_table)` each of length
/// `max_seq_len * head_dim / 2`.
pub fn precompute_rope_tables(
    head_dim: usize,
    max_seq_len: usize,
    theta_base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let n = max_seq_len * half_dim;
    let mut cos_table = Vec::with_capacity(n);
    let mut sin_table = Vec::with_capacity(n);

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let exponent = -2.0 * (i as f32) / (head_dim as f32);
            let inv_freq = theta_base.powf(exponent);
            let angle = (pos as f32) * inv_freq;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
    }

    (cos_table, sin_table)
}

/// Build a 2-bit dequantization map: index ∈ {0,1,2,3} → {0, 1, -1, 0}.
pub fn build_dequant_i2s_map() -> [f32; 4] {
    [0.0, 1.0, -1.0, 0.0]
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> ConstantMemoryConfig {
        ConstantMemoryConfig::default()
    }

    fn small_config() -> ConstantMemoryConfig {
        ConstantMemoryConfig {
            total_size: 1024,
            alignment: 16,
            reserved_bytes: 0,
            enable_profiling: false,
        }
    }

    fn profiling_config() -> ConstantMemoryConfig {
        ConstantMemoryConfig {
            total_size: 4096,
            alignment: 16,
            reserved_bytes: 0,
            enable_profiling: true,
        }
    }

    // ── Config validation ────────────────────────────────────────────

    #[test]
    fn config_default_is_valid() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn config_zero_total_size_rejected() {
        let cfg = ConstantMemoryConfig { total_size: 0, ..default_config() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_non_power_of_two_alignment_rejected() {
        let cfg = ConstantMemoryConfig { alignment: 7, ..default_config() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_reserved_exceeds_total_rejected() {
        let cfg =
            ConstantMemoryConfig { total_size: 1024, reserved_bytes: 1024, ..default_config() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_usable_capacity() {
        let cfg =
            ConstantMemoryConfig { total_size: 4096, reserved_bytes: 256, ..default_config() };
        assert_eq!(cfg.usable_capacity(), 3840);
    }

    // ── Pool construction ────────────────────────────────────────────

    #[test]
    fn pool_with_defaults_succeeds() {
        assert!(ConstantMemoryPool::with_defaults().is_ok());
    }

    #[test]
    fn pool_new_with_valid_config() {
        let pool = ConstantMemoryPool::new(small_config()).unwrap();
        assert_eq!(pool.num_slots(), 0);
        assert_eq!(pool.bytes_used(), 0);
    }

    #[test]
    fn pool_new_with_invalid_config_fails() {
        let cfg = ConstantMemoryConfig { total_size: 0, ..default_config() };
        assert!(ConstantMemoryPool::new(cfg).is_err());
    }

    // ── Allocation basics ────────────────────────────────────────────

    #[test]
    fn allocate_single_slot() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        let slot = pool.allocate(64, "test").unwrap();
        assert_eq!(slot.offset, 0);
        assert!(slot.size >= 64);
        assert_eq!(pool.num_slots(), 1);
    }

    #[test]
    fn allocate_zero_bytes_rejected() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        assert!(pool.allocate(0, "zero").is_err());
    }

    #[test]
    fn allocate_respects_alignment() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        let s1 = pool.allocate(1, "tiny").unwrap();
        assert_eq!(s1.size, 16); // aligned up to 16
        let s2 = pool.allocate(17, "next").unwrap();
        assert_eq!(s2.size, 32); // 17 aligned up to 32
        assert_eq!(s2.offset, 16);
    }

    #[test]
    fn allocate_overflow_detected() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        pool.allocate(512, "half").unwrap();
        assert!(pool.allocate(1024, "too-big").is_err());
    }

    #[test]
    fn allocate_fills_exactly() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        pool.allocate(1024, "all").unwrap();
        assert!(pool.is_full());
    }

    #[test]
    fn slot_ids_are_unique() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let mut ids = std::collections::HashSet::new();
        for i in 0..20 {
            let slot = pool.allocate(16, format!("s{}", i)).unwrap();
            assert!(ids.insert(slot.id));
        }
    }

    // ── Deallocation ─────────────────────────────────────────────────

    #[test]
    fn deallocate_frees_space() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        let s = pool.allocate(256, "a").unwrap();
        assert_eq!(pool.bytes_used(), 256);
        pool.deallocate(s.id).unwrap();
        assert_eq!(pool.bytes_used(), 0);
        assert_eq!(pool.num_slots(), 0);
    }

    #[test]
    fn deallocate_unknown_slot_fails() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        assert!(pool.deallocate(ConstantSlotId(999)).is_err());
    }

    #[test]
    fn deallocate_twice_fails() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        let s = pool.allocate(64, "once").unwrap();
        pool.deallocate(s.id).unwrap();
        assert!(pool.deallocate(s.id).is_err());
    }

    #[test]
    fn allocate_after_deallocate() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        let s = pool.allocate(512, "first").unwrap();
        pool.deallocate(s.id).unwrap();
        let s2 = pool.allocate(512, "second").unwrap();
        assert!(s2.size >= 512);
    }

    // ── LUT loading ──────────────────────────────────────────────────

    #[test]
    fn load_lut_f32() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let entry = pool.load_lut(LutKind::QuantizationScales, &data, "scales").unwrap();
        assert_eq!(entry.kind, LutKind::QuantizationScales);
        assert_eq!(entry.num_elements, 64);
        assert_eq!(entry.element_size, 4);
        assert_eq!(entry.total_bytes(), 256);
        assert_eq!(pool.num_luts(), 1);
    }

    #[test]
    fn load_lut_bytes() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let data: Vec<u8> = vec![0, 1, 2, 3];
        let entry = pool.load_lut_bytes(LutKind::DequantMap, &data, "dequant_map").unwrap();
        assert_eq!(entry.kind, LutKind::DequantMap);
        assert_eq!(entry.num_elements, 4);
        assert_eq!(entry.element_size, 1);
    }

    #[test]
    fn load_rope_cos_sin_tables() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let (cos_table, sin_table) = precompute_rope_tables(64, 16, 10000.0);
        let cos_entry = pool.load_lut(LutKind::RopeCosTable, &cos_table, "rope_cos").unwrap();
        let sin_entry = pool.load_lut(LutKind::RopeSinTable, &sin_table, "rope_sin").unwrap();
        assert_eq!(cos_entry.kind, LutKind::RopeCosTable);
        assert_eq!(sin_entry.kind, LutKind::RopeSinTable);
        assert_eq!(pool.num_luts(), 2);
    }

    #[test]
    fn load_lut_overflow() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        let data: Vec<f32> = vec![0.0; 512]; // 2048 bytes > 1024
        assert!(pool.load_lut(LutKind::Custom, &data, "huge").is_err());
    }

    #[test]
    fn get_lut_by_id() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let entry = pool.load_lut(LutKind::QuantizationScales, &data, "s").unwrap();
        let retrieved = pool.get_lut(entry.slot_id).unwrap();
        assert_eq!(retrieved.kind, LutKind::QuantizationScales);
    }

    #[test]
    fn find_luts_by_kind() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let data = vec![1.0; 4];
        pool.load_lut(LutKind::QuantizationScales, &data, "s1").unwrap();
        pool.load_lut(LutKind::QuantizationScales, &data, "s2").unwrap();
        pool.load_lut(LutKind::RopeCosTable, &data, "cos").unwrap();
        let scales = pool.find_luts_by_kind(LutKind::QuantizationScales);
        assert_eq!(scales.len(), 2);
        let cos = pool.find_luts_by_kind(LutKind::RopeCosTable);
        assert_eq!(cos.len(), 1);
    }

    #[test]
    fn lut_deallocated_with_slot() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let data = vec![1.0; 4];
        let entry = pool.load_lut(LutKind::Custom, &data, "tmp").unwrap();
        let id = entry.slot_id;
        pool.deallocate(id).unwrap();
        assert!(pool.get_lut(id).is_none());
        assert_eq!(pool.num_luts(), 0);
    }

    // ── Invalidation strategies ──────────────────────────────────────

    #[test]
    fn invalidation_default_is_lru() {
        let pool = ConstantMemoryPool::new(small_config()).unwrap();
        assert_eq!(pool.invalidation_strategy(), InvalidationStrategy::Lru);
    }

    #[test]
    fn set_invalidation_strategy() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        pool.set_invalidation_strategy(InvalidationStrategy::GenerationBased);
        assert_eq!(pool.invalidation_strategy(), InvalidationStrategy::GenerationBased);
    }

    #[test]
    fn evict_lru_order() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        let s1 = pool.allocate(64, "first").unwrap();
        let _s2 = pool.allocate(64, "second").unwrap();
        // s1 is LRU — should be evicted first.
        let evicted = pool.evict_one().unwrap();
        assert_eq!(evicted, Some(s1.id));
    }

    #[test]
    fn evict_lru_after_touch() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        let s1 = pool.allocate(64, "first").unwrap();
        let s2 = pool.allocate(64, "second").unwrap();
        pool.touch(s1.id); // s1 now most-recently-used
        let evicted = pool.evict_one().unwrap();
        assert_eq!(evicted, Some(s2.id));
    }

    #[test]
    fn evict_generation_based() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        let s1 = pool.allocate(64, "old").unwrap();
        pool.advance_generation();
        let _s2 = pool.allocate(64, "new").unwrap();
        pool.set_invalidation_strategy(InvalidationStrategy::GenerationBased);
        let evicted = pool.evict_one().unwrap();
        assert_eq!(evicted, Some(s1.id));
    }

    #[test]
    fn evict_explicit_does_nothing() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        pool.allocate(64, "keep").unwrap();
        pool.set_invalidation_strategy(InvalidationStrategy::Explicit);
        let evicted = pool.evict_one().unwrap();
        assert_eq!(evicted, None);
    }

    #[test]
    fn evict_older_than_generation() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        pool.allocate(64, "gen0_a").unwrap();
        pool.allocate(64, "gen0_b").unwrap();
        pool.advance_generation();
        pool.allocate(64, "gen1").unwrap();
        let evicted = pool.evict_older_than(1).unwrap();
        assert_eq!(evicted, 2);
        assert_eq!(pool.num_slots(), 1);
    }

    #[test]
    fn invalidate_all_clears_pool() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        for i in 0..5 {
            pool.allocate(64, format!("s{}", i)).unwrap();
        }
        let count = pool.invalidate_all().unwrap();
        assert_eq!(count, 5);
        assert_eq!(pool.num_slots(), 0);
        assert_eq!(pool.bytes_used(), 0);
    }

    #[test]
    fn advance_generation_increments() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        assert_eq!(pool.generation(), 0);
        pool.advance_generation();
        assert_eq!(pool.generation(), 1);
        pool.advance_generation();
        assert_eq!(pool.generation(), 2);
    }

    #[test]
    fn evict_from_empty_pool() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        let evicted = pool.evict_one().unwrap();
        assert_eq!(evicted, None);
    }

    // ── Broadcast ────────────────────────────────────────────────────

    #[test]
    fn create_broadcast_binding() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let bc = pool.create_broadcast("learning_rate", 4).unwrap();
        assert_eq!(bc.name, "learning_rate");
        assert_eq!(bc.size, 4);
        assert_eq!(pool.num_broadcasts(), 1);
    }

    #[test]
    fn get_broadcast_by_id() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let bc = pool.create_broadcast("lr", 4).unwrap();
        let retrieved = pool.get_broadcast(bc.slot_id).unwrap();
        assert_eq!(retrieved.name, "lr");
    }

    #[test]
    fn find_broadcast_by_name() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        pool.create_broadcast("alpha", 4).unwrap();
        pool.create_broadcast("beta", 4).unwrap();
        let found = pool.find_broadcast_by_name("alpha").unwrap();
        assert_eq!(found.name, "alpha");
        assert!(pool.find_broadcast_by_name("gamma").is_none());
    }

    #[test]
    fn broadcast_deallocated_with_slot() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let bc = pool.create_broadcast("tmp", 4).unwrap();
        pool.deallocate(bc.slot_id).unwrap();
        assert_eq!(pool.num_broadcasts(), 0);
    }

    #[test]
    fn multiple_broadcasts() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        for i in 0..10 {
            pool.create_broadcast(format!("p{}", i), 4).unwrap();
        }
        assert_eq!(pool.num_broadcasts(), 10);
    }

    // ── Overflow detection ───────────────────────────────────────────

    #[test]
    fn would_overflow_false_when_space() {
        let pool = ConstantMemoryPool::new(small_config()).unwrap();
        assert!(!pool.would_overflow(512));
    }

    #[test]
    fn would_overflow_true_when_full() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        pool.allocate(1024, "all").unwrap();
        assert!(pool.would_overflow(1));
    }

    #[test]
    fn would_overflow_accounts_for_alignment() {
        let cfg = ConstantMemoryConfig {
            total_size: 32,
            alignment: 16,
            reserved_bytes: 0,
            enable_profiling: false,
        };
        let mut pool = ConstantMemoryPool::new(cfg).unwrap();
        pool.allocate(16, "a").unwrap();
        // 1 byte aligns up to 16, total would be 32 = capacity → fits
        assert!(!pool.would_overflow(1));
        // 17 bytes aligns up to 32, total would be 48 > 32 → overflow
        assert!(pool.would_overflow(17));
    }

    #[test]
    fn is_full_after_capacity_exhausted() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        pool.allocate(1024, "fill").unwrap();
        assert!(pool.is_full());
    }

    #[test]
    fn near_overflow_threshold() {
        let cfg = ConstantMemoryConfig {
            total_size: 1024,
            alignment: 16,
            reserved_bytes: 0,
            enable_profiling: false,
        };
        let mut pool = ConstantMemoryPool::new(cfg).unwrap();
        pool.allocate(976, "almost").unwrap(); // 976 / 1024 ≈ 0.953 > 0.95
        assert!(pool.near_overflow());
    }

    #[test]
    fn utilisation_ratio() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        pool.allocate(512, "half").unwrap();
        let u = pool.utilisation();
        assert!((u - 0.5).abs() < 1e-6);
    }

    // ── Cache hints ──────────────────────────────────────────────────

    #[test]
    fn default_cache_hint_is_read_only() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let s = pool.allocate(16, "x").unwrap();
        assert_eq!(pool.cache_hint(s.id), ReadOnlyCacheHint::UseReadOnlyCache);
    }

    #[test]
    fn set_and_get_cache_hint() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let s = pool.allocate(16, "x").unwrap();
        pool.set_cache_hint(s.id, ReadOnlyCacheHint::Streaming).unwrap();
        assert_eq!(pool.cache_hint(s.id), ReadOnlyCacheHint::Streaming);
    }

    #[test]
    fn set_cache_hint_unknown_slot_fails() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        assert!(pool.set_cache_hint(ConstantSlotId(999), ReadOnlyCacheHint::DefaultCache).is_err());
    }

    #[test]
    fn cache_hint_removed_on_deallocate() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let s = pool.allocate(16, "x").unwrap();
        pool.set_cache_hint(s.id, ReadOnlyCacheHint::Streaming).unwrap();
        pool.deallocate(s.id).unwrap();
        // Default returned for unknown slot.
        assert_eq!(pool.cache_hint(s.id), ReadOnlyCacheHint::UseReadOnlyCache);
    }

    // ── Config parameter store ───────────────────────────────────────

    #[test]
    fn config_store_empty() {
        let pool = ConstantMemoryPool::new(default_config()).unwrap();
        let store = pool.create_config_store();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn store_and_get_u32_param() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let mut store = pool.create_config_store();
        pool.store_config(&mut store, "batch_size", ConfigValue::U32(32)).unwrap();
        let val = store.get("batch_size").unwrap();
        assert_eq!(*val, ConfigValue::U32(32));
    }

    #[test]
    fn store_and_get_f32_param() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let mut store = pool.create_config_store();
        pool.store_config(&mut store, "epsilon", ConfigValue::F32(1e-5)).unwrap();
        match store.get("epsilon") {
            Some(ConfigValue::F32(v)) => assert!((*v - 1e-5).abs() < 1e-10),
            other => panic!("expected F32, got {:?}", other),
        }
    }

    #[test]
    fn store_pair_param() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let mut store = pool.create_config_store();
        pool.store_config(&mut store, "shape", ConfigValue::Pair(128, 256)).unwrap();
        assert_eq!(*store.get("shape").unwrap(), ConfigValue::Pair(128, 256));
    }

    #[test]
    fn store_bytes_param() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let mut store = pool.create_config_store();
        pool.store_config(&mut store, "mask", ConfigValue::Bytes(vec![0xFF, 0x00, 0xFF])).unwrap();
        match store.get("mask") {
            Some(ConfigValue::Bytes(b)) => assert_eq!(b, &[0xFF, 0x00, 0xFF]),
            other => panic!("expected Bytes, got {:?}", other),
        }
    }

    #[test]
    fn config_store_names() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let mut store = pool.create_config_store();
        pool.store_config(&mut store, "a", ConfigValue::U32(1)).unwrap();
        pool.store_config(&mut store, "b", ConfigValue::U32(2)).unwrap();
        let mut names = store.names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn config_store_total_bytes() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let mut store = pool.create_config_store();
        pool.store_config(&mut store, "x", ConfigValue::U32(1)).unwrap();
        pool.store_config(&mut store, "y", ConfigValue::Pair(0, 0)).unwrap();
        assert_eq!(store.total_bytes(), 12); // 4 + 8
    }

    #[test]
    fn config_value_size_bytes() {
        assert_eq!(ConfigValue::U32(0).size_bytes(), 4);
        assert_eq!(ConfigValue::F32(0.0).size_bytes(), 4);
        assert_eq!(ConfigValue::Pair(0, 0).size_bytes(), 8);
        assert_eq!(ConfigValue::Bytes(vec![0; 10]).size_bytes(), 10);
    }

    // ── Profiling ────────────────────────────────────────────────────

    #[test]
    fn profiler_initial_state() {
        let prof = ConstantMemoryProfiler::new();
        assert_eq!(prof.total_accesses(), 0);
        assert_eq!(prof.total_misses(), 0);
        assert!((prof.hit_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn profiler_record_access() {
        let mut prof = ConstantMemoryProfiler::new();
        let slot = ConstantSlotId(0);
        prof.record_access(slot);
        prof.record_access(slot);
        assert_eq!(prof.total_accesses(), 2);
        assert_eq!(prof.slot_accesses(slot), 2);
    }

    #[test]
    fn profiler_record_miss() {
        let mut prof = ConstantMemoryProfiler::new();
        prof.record_access(ConstantSlotId(0));
        prof.record_access(ConstantSlotId(0));
        prof.record_miss();
        assert_eq!(prof.total_misses(), 1);
        assert!((prof.hit_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn profiler_hit_rate_no_misses() {
        let mut prof = ConstantMemoryProfiler::new();
        for _ in 0..100 {
            prof.record_access(ConstantSlotId(0));
        }
        assert!((prof.hit_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn profiler_hit_rate_all_misses() {
        let mut prof = ConstantMemoryProfiler::new();
        for _ in 0..10 {
            prof.record_access(ConstantSlotId(0));
            prof.record_miss();
        }
        assert!((prof.hit_rate() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn profiler_hottest_slot() {
        let mut prof = ConstantMemoryProfiler::new();
        let s0 = ConstantSlotId(0);
        let s1 = ConstantSlotId(1);
        prof.record_access(s0);
        prof.record_access(s1);
        prof.record_access(s1);
        prof.record_access(s1);
        let (id, count) = prof.hottest_slot().unwrap();
        assert_eq!(id, s1);
        assert_eq!(count, 3);
    }

    #[test]
    fn profiler_hottest_slot_empty() {
        let prof = ConstantMemoryProfiler::new();
        assert!(prof.hottest_slot().is_none());
    }

    #[test]
    fn profiler_reset() {
        let mut prof = ConstantMemoryProfiler::new();
        prof.record_access(ConstantSlotId(0));
        prof.record_miss();
        prof.reset();
        assert_eq!(prof.total_accesses(), 0);
        assert_eq!(prof.total_misses(), 0);
    }

    #[test]
    fn pool_profiling_via_touch() {
        let mut pool = ConstantMemoryPool::new(profiling_config()).unwrap();
        let s = pool.allocate(16, "x").unwrap();
        pool.touch(s.id);
        pool.touch(s.id);
        assert_eq!(pool.profiler().slot_accesses(s.id), 2);
    }

    #[test]
    fn pool_profiling_disabled_no_recording() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        let s = pool.allocate(16, "x").unwrap();
        pool.touch(s.id);
        // Profiling disabled — should not record.
        assert_eq!(pool.profiler().slot_accesses(s.id), 0);
    }

    #[test]
    fn pool_record_miss() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        pool.record_miss();
        assert_eq!(pool.profiler().total_misses(), 1);
    }

    // ── Statistics ───────────────────────────────────────────────────

    #[test]
    fn stats_empty_pool() {
        let pool = ConstantMemoryPool::new(small_config()).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.capacity, 1024);
        assert_eq!(stats.used, 0);
        assert_eq!(stats.free, 1024);
        assert_eq!(stats.num_slots, 0);
        assert_eq!(stats.num_luts, 0);
        assert_eq!(stats.num_broadcasts, 0);
        assert!(!stats.near_overflow);
    }

    #[test]
    fn stats_after_allocations() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        pool.allocate(256, "a").unwrap();
        pool.allocate(256, "b").unwrap();
        let stats = pool.stats();
        assert_eq!(stats.used, 512);
        assert_eq!(stats.free, 512);
        assert_eq!(stats.num_slots, 2);
    }

    #[test]
    fn stats_with_luts_and_broadcasts() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        pool.load_lut(LutKind::QuantizationScales, &[1.0; 4], "s").unwrap();
        pool.create_broadcast("lr", 4).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.num_luts, 1);
        assert_eq!(stats.num_broadcasts, 1);
    }

    // ── Precompute helpers ───────────────────────────────────────────

    #[test]
    fn precompute_rope_tables_dimensions() {
        let (cos, sin) = precompute_rope_tables(64, 128, 10000.0);
        assert_eq!(cos.len(), 128 * 32); // max_seq_len * half_dim
        assert_eq!(sin.len(), 128 * 32);
    }

    #[test]
    fn precompute_rope_tables_position_zero() {
        let (cos, sin) = precompute_rope_tables(4, 1, 10000.0);
        // At position 0, angle = 0 for all dims → cos=1, sin=0.
        for &c in &cos {
            assert!((c - 1.0).abs() < 1e-6);
        }
        for &s in &sin {
            assert!(s.abs() < 1e-6);
        }
    }

    #[test]
    fn precompute_rope_tables_values_bounded() {
        let (cos, sin) = precompute_rope_tables(64, 512, 10000.0);
        for &c in &cos {
            assert!(c >= -1.0 && c <= 1.0, "cos out of [-1,1]: {}", c);
        }
        for &s in &sin {
            assert!(s >= -1.0 && s <= 1.0, "sin out of [-1,1]: {}", s);
        }
    }

    #[test]
    fn build_dequant_map() {
        let map = build_dequant_i2s_map();
        assert_eq!(map, [0.0, 1.0, -1.0, 0.0]);
    }

    // ── LutKind display ──────────────────────────────────────────────

    #[test]
    fn lut_kind_display() {
        assert_eq!(LutKind::QuantizationScales.to_string(), "quant_scales");
        assert_eq!(LutKind::RopeCosTable.to_string(), "rope_cos");
        assert_eq!(LutKind::RopeSinTable.to_string(), "rope_sin");
        assert_eq!(LutKind::DequantMap.to_string(), "dequant_map");
        assert_eq!(LutKind::Custom.to_string(), "custom");
    }

    // ── Slot access ──────────────────────────────────────────────────

    #[test]
    fn get_slot_by_id() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let s = pool.allocate(32, "test_slot").unwrap();
        let slot = pool.get_slot(s.id).unwrap();
        assert_eq!(slot.label, "test_slot");
        assert_eq!(slot.size, 32);
    }

    #[test]
    fn get_slot_unknown_returns_none() {
        let pool = ConstantMemoryPool::new(default_config()).unwrap();
        assert!(pool.get_slot(ConstantSlotId(42)).is_none());
    }

    #[test]
    fn iterate_slots() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        pool.allocate(16, "a").unwrap();
        pool.allocate(16, "b").unwrap();
        pool.allocate(16, "c").unwrap();
        let labels: Vec<&str> = pool.slots().map(|s| s.label.as_str()).collect();
        assert_eq!(labels.len(), 3);
        assert!(labels.contains(&"a"));
        assert!(labels.contains(&"b"));
        assert!(labels.contains(&"c"));
    }

    // ── Stress / pattern tests ───────────────────────────────────────

    #[test]
    fn alloc_dealloc_cycle_no_leak() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        for i in 0..50 {
            let s = pool.allocate(16, format!("cycle_{}", i)).unwrap();
            pool.deallocate(s.id).unwrap();
        }
        assert_eq!(pool.bytes_used(), 0);
        assert_eq!(pool.num_slots(), 0);
    }

    #[test]
    fn fill_then_evict_all_lru() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        // Fill 1024 bytes with 16-byte slots → 64 slots.
        for i in 0..64 {
            pool.allocate(16, format!("s{}", i)).unwrap();
        }
        assert!(pool.is_full());
        while pool.num_slots() > 0 {
            pool.evict_one().unwrap();
        }
        assert_eq!(pool.bytes_used(), 0);
    }

    #[test]
    fn mixed_lut_broadcast_config() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        let mut store = pool.create_config_store();
        pool.load_lut(LutKind::QuantizationScales, &[1.0; 8], "scales").unwrap();
        pool.create_broadcast("epsilon", 4).unwrap();
        pool.store_config(&mut store, "heads", ConfigValue::U32(8)).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.num_luts, 1);
        assert_eq!(stats.num_broadcasts, 1);
        assert!(stats.used > 0);
    }

    #[test]
    fn evict_then_reallocate() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        pool.allocate(512, "big1").unwrap();
        pool.allocate(512, "big2").unwrap();
        assert!(pool.is_full());
        pool.evict_one().unwrap(); // Evict big1
        let s = pool.allocate(256, "smaller").unwrap();
        assert!(s.size >= 256);
    }

    #[test]
    fn bytes_free_consistent() {
        let mut pool = ConstantMemoryPool::new(small_config()).unwrap();
        assert_eq!(pool.bytes_free(), 1024);
        pool.allocate(256, "a").unwrap();
        assert_eq!(pool.bytes_free(), 768);
        pool.allocate(256, "b").unwrap();
        assert_eq!(pool.bytes_free(), 512);
    }

    #[test]
    fn config_accessor() {
        let pool = ConstantMemoryPool::new(small_config()).unwrap();
        assert_eq!(pool.config().total_size, 1024);
        assert_eq!(pool.config().alignment, 16);
    }

    #[test]
    fn reserved_bytes_reduces_capacity() {
        let cfg = ConstantMemoryConfig {
            total_size: 1024,
            reserved_bytes: 256,
            alignment: 16,
            enable_profiling: false,
        };
        let mut pool = ConstantMemoryPool::new(cfg).unwrap();
        assert_eq!(pool.bytes_free(), 768);
        // Cannot allocate more than usable capacity.
        assert!(pool.allocate(800, "too-big").is_err());
        pool.allocate(768, "fits").unwrap();
        assert!(pool.is_full());
    }

    #[test]
    fn constant_memory_size_is_64kb() {
        assert_eq!(DEFAULT_CONSTANT_MEMORY_SIZE, 65536);
    }

    #[test]
    fn constant_memory_alignment_is_16() {
        assert_eq!(CONSTANT_MEMORY_ALIGNMENT, 16);
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn kernel_source_not_empty() {
        assert!(!CONSTANT_MEMORY_KERNEL_SRC.is_empty());
        assert!(CONSTANT_MEMORY_KERNEL_SRC.contains("c_const_mem"));
    }

    #[test]
    fn profiler_default_trait() {
        let prof = ConstantMemoryProfiler::default();
        assert_eq!(prof.total_accesses(), 0);
    }

    #[test]
    fn slot_generation_matches_pool() {
        let mut pool = ConstantMemoryPool::new(default_config()).unwrap();
        pool.advance_generation();
        pool.advance_generation();
        let s = pool.allocate(16, "gen2").unwrap();
        assert_eq!(s.generation, 2);
    }
}
