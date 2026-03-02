//! Multi-model serving support for OpenCL (Intel Arc A770).
//!
//! Manages multiple loaded models on a single GPU with memory partitioning,
//! context switching, fair scheduling, and configurable eviction policies.
//! All scheduling and eviction algorithms have CPU reference implementations
//! so the module compiles and tests unconditionally (no feature gates).

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

// ---------------------------------------------------------------------------
// ModelState
// ---------------------------------------------------------------------------

/// Lifecycle state of a model slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelState {
    /// Model weights are being transferred to the device.
    Loading,
    /// Model is resident and ready for inference.
    Ready,
    /// Model is being evicted from device memory.
    Evicting,
    /// Model has been fully removed from device memory.
    Evicted,
}

impl fmt::Display for ModelState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loading => write!(f, "loading"),
            Self::Ready => write!(f, "ready"),
            Self::Evicting => write!(f, "evicting"),
            Self::Evicted => write!(f, "evicted"),
        }
    }
}

// ---------------------------------------------------------------------------
// EvictionPolicy
// ---------------------------------------------------------------------------

/// Policy used to select which model to evict under memory pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictionPolicy {
    /// Least-recently-used: evict the model that has not been accessed the longest.
    LRU,
    /// Least-frequently-used: evict the model with the fewest accesses.
    LFU,
    /// Priority-based: evict the lowest-priority model first.
    Priority,
    /// Size-based: evict the model consuming the most memory first.
    SizeBased,
}

impl fmt::Display for EvictionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LRU => write!(f, "lru"),
            Self::LFU => write!(f, "lfu"),
            Self::Priority => write!(f, "priority"),
            Self::SizeBased => write!(f, "size_based"),
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryPartition
// ---------------------------------------------------------------------------

/// Tracks a per-model GPU memory allocation within the total budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPartition {
    /// Unique identifier matching the parent `ModelSlot`.
    pub model_id: String,
    /// Bytes allocated on the device for this model.
    pub allocated_bytes: usize,
    /// Maximum bytes this partition is allowed to use.
    pub budget_bytes: usize,
}

impl MemoryPartition {
    pub fn new(model_id: impl Into<String>, budget_bytes: usize) -> Self {
        Self { model_id: model_id.into(), allocated_bytes: 0, budget_bytes }
    }

    /// Try to allocate `bytes` within this partition. Returns `false` if
    /// the request would exceed the budget.
    pub fn allocate(&mut self, bytes: usize) -> bool {
        if self.allocated_bytes.saturating_add(bytes) > self.budget_bytes {
            return false;
        }
        self.allocated_bytes += bytes;
        true
    }

    /// Free `bytes` from this partition (clamped to zero).
    pub fn deallocate(&mut self, bytes: usize) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(bytes);
    }

    /// Remaining bytes available in this partition.
    pub fn available(&self) -> usize {
        self.budget_bytes.saturating_sub(self.allocated_bytes)
    }

    /// Utilisation ratio in `[0.0, 1.0]`.
    pub fn utilization(&self) -> f64 {
        if self.budget_bytes == 0 {
            return 0.0;
        }
        self.allocated_bytes as f64 / self.budget_bytes as f64
    }
}

// ---------------------------------------------------------------------------
// ModelSlot
// ---------------------------------------------------------------------------

/// Represents a single loaded (or loading) model on the GPU.
#[derive(Debug, Clone)]
pub struct ModelSlot {
    /// Unique model identifier (e.g. a GGUF path hash).
    pub id: String,
    /// Memory budget assigned to this model (bytes).
    pub memory_budget: usize,
    /// Scheduling priority (higher = more important, 0 is lowest).
    pub priority: u32,
    /// Monotonic counter bumped on every access (for LRU / LFU).
    pub access_count: u64,
    /// Logical timestamp of last access (monotonic tick, not wall-clock).
    pub last_used_tick: u64,
    /// Current lifecycle state.
    pub state: ModelState,
    /// Associated memory partition.
    pub partition: MemoryPartition,
}

impl ModelSlot {
    pub fn new(id: impl Into<String>, memory_budget: usize, priority: u32) -> Self {
        let id = id.into();
        let partition = MemoryPartition::new(&id, memory_budget);
        Self {
            id,
            memory_budget,
            priority,
            access_count: 0,
            last_used_tick: 0,
            state: ModelState::Loading,
            partition,
        }
    }

    /// Mark the model as ready for inference.
    pub fn mark_ready(&mut self) {
        self.state = ModelState::Ready;
    }

    /// Mark the model as being evicted.
    pub fn mark_evicting(&mut self) {
        self.state = ModelState::Evicting;
    }

    /// Mark the model as fully evicted and release its partition.
    pub fn mark_evicted(&mut self) {
        self.state = ModelState::Evicted;
        let used = self.partition.allocated_bytes;
        self.partition.deallocate(used);
    }

    /// Record an access (bumps count and tick).
    pub fn touch(&mut self, tick: u64) {
        self.access_count += 1;
        self.last_used_tick = tick;
    }
}

// ---------------------------------------------------------------------------
// ModelSwitchCost
// ---------------------------------------------------------------------------

/// Estimates the overhead of switching between models on the device.
#[derive(Debug, Clone)]
pub struct ModelSwitchCost {
    /// Base latency for any context switch (nanoseconds).
    pub base_latency_ns: u64,
    /// Additional latency per megabyte of weights to transfer.
    pub per_mb_latency_ns: u64,
}

impl Default for ModelSwitchCost {
    fn default() -> Self {
        Self {
            // 500 µs base + 100 µs/MB  — ballpark for PCIe 4.0 ×16
            base_latency_ns: 500_000,
            per_mb_latency_ns: 100_000,
        }
    }
}

impl ModelSwitchCost {
    pub fn new(base_latency_ns: u64, per_mb_latency_ns: u64) -> Self {
        Self { base_latency_ns, per_mb_latency_ns }
    }

    /// Estimated switch time for a model of `size_bytes`.
    pub fn estimate_ns(&self, size_bytes: usize) -> u64 {
        let mb = size_bytes as u64 / (1024 * 1024);
        self.base_latency_ns.saturating_add(mb.saturating_mul(self.per_mb_latency_ns))
    }

    /// Same as [`estimate_ns`] but returns a `Duration`.
    pub fn estimate_duration(&self, size_bytes: usize) -> Duration {
        Duration::from_nanos(self.estimate_ns(size_bytes))
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source for device-side memory copy
// ---------------------------------------------------------------------------

/// OpenCL kernel source for bulk memory copy between partitions.
pub const MULTI_MODEL_MEMORY_COPY_CL: &str = r#"
__kernel void multi_model_memcpy(
    __global const float* src,
    __global float* dst,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid < count) {
        dst[gid] = src[gid];
    }
}

__kernel void multi_model_zero_fill(
    __global float* buf,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid < count) {
        buf[gid] = 0.0f;
    }
}
"#;

// ---------------------------------------------------------------------------
// MultiModelManager
// ---------------------------------------------------------------------------

/// Manages multiple model slots on a single GPU device.
///
/// All methods are CPU-reference implementations; the actual GPU transfers
/// would be driven by an OpenCL command queue in production.
pub struct MultiModelManager {
    /// Total device memory budget (bytes).
    total_budget: usize,
    /// Bytes currently committed across all partitions.
    used_bytes: usize,
    /// Eviction policy in effect.
    policy: EvictionPolicy,
    /// Model slots keyed by model id.
    slots: HashMap<String, ModelSlot>,
    /// Monotonic tick counter for LRU ordering.
    tick: u64,
    /// Context switch cost estimator.
    switch_cost: ModelSwitchCost,
}

impl fmt::Debug for MultiModelManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiModelManager")
            .field("total_budget", &self.total_budget)
            .field("used_bytes", &self.used_bytes)
            .field("policy", &self.policy)
            .field("num_slots", &self.slots.len())
            .field("tick", &self.tick)
            .finish()
    }
}

impl MultiModelManager {
    /// Create a manager with the given total device memory budget.
    pub fn new(total_budget: usize, policy: EvictionPolicy) -> Self {
        Self {
            total_budget,
            used_bytes: 0,
            policy,
            slots: HashMap::new(),
            tick: 0,
            switch_cost: ModelSwitchCost::default(),
        }
    }

    /// Override the default switch-cost estimator.
    pub fn with_switch_cost(mut self, cost: ModelSwitchCost) -> Self {
        self.switch_cost = cost;
        self
    }

    // -- accessors --

    pub fn total_budget(&self) -> usize {
        self.total_budget
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub fn available_bytes(&self) -> usize {
        self.total_budget.saturating_sub(self.used_bytes)
    }

    pub fn policy(&self) -> EvictionPolicy {
        self.policy
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn get_slot(&self, model_id: &str) -> Option<&ModelSlot> {
        self.slots.get(model_id)
    }

    pub fn switch_cost(&self) -> &ModelSwitchCost {
        &self.switch_cost
    }

    /// All model ids currently tracked (any state).
    pub fn model_ids(&self) -> Vec<String> {
        self.slots.keys().cloned().collect()
    }

    // -- lifecycle --

    /// Load a model into a slot. Returns `Err` if the budget is insufficient
    /// or a model with the same id already exists.
    pub fn load_model(
        &mut self,
        id: impl Into<String>,
        size_bytes: usize,
        priority: u32,
    ) -> Result<(), String> {
        let id = id.into();
        if self.slots.contains_key(&id) {
            return Err(format!("model '{id}' already loaded"));
        }
        if size_bytes > self.available_bytes() {
            return Err(format!(
                "insufficient memory: need {size_bytes}, available {}",
                self.available_bytes()
            ));
        }
        let mut slot = ModelSlot::new(&id, size_bytes, priority);
        // Simulate loading: allocate the full budget in the partition.
        if !slot.partition.allocate(size_bytes) {
            return Err("partition allocation failed".into());
        }
        slot.mark_ready();
        self.tick += 1;
        slot.touch(self.tick);
        self.used_bytes += size_bytes;
        self.slots.insert(id, slot);
        Ok(())
    }

    /// Explicitly unload a model, freeing its memory.
    pub fn unload_model(&mut self, model_id: &str) -> Result<(), String> {
        let slot =
            self.slots.get_mut(model_id).ok_or_else(|| format!("model '{model_id}' not found"))?;
        slot.mark_evicting();
        let freed = slot.partition.allocated_bytes;
        slot.mark_evicted();
        self.used_bytes = self.used_bytes.saturating_sub(freed);
        self.slots.remove(model_id);
        Ok(())
    }

    /// Record an access to a model (updates LRU / LFU counters).
    pub fn access_model(&mut self, model_id: &str) -> Result<(), String> {
        let slot =
            self.slots.get_mut(model_id).ok_or_else(|| format!("model '{model_id}' not found"))?;
        if slot.state != ModelState::Ready {
            return Err(format!("model '{model_id}' is not ready (state={})", slot.state));
        }
        self.tick += 1;
        slot.touch(self.tick);
        Ok(())
    }

    /// Estimate the context-switch cost to activate `model_id`.
    pub fn estimate_switch_cost(&self, model_id: &str) -> Option<u64> {
        self.slots.get(model_id).map(|s| self.switch_cost.estimate_ns(s.memory_budget))
    }

    // -- eviction --

    /// Select the best eviction candidate according to the current policy.
    /// Only `Ready` models are candidates. Returns `None` if no candidate.
    pub fn eviction_candidate(&self) -> Option<&str> {
        let candidates: Vec<&ModelSlot> =
            self.slots.values().filter(|s| s.state == ModelState::Ready).collect();
        if candidates.is_empty() {
            return None;
        }
        let best = match self.policy {
            EvictionPolicy::LRU => candidates.iter().min_by_key(|s| s.last_used_tick),
            EvictionPolicy::LFU => candidates.iter().min_by_key(|s| s.access_count),
            EvictionPolicy::Priority => candidates.iter().min_by_key(|s| s.priority),
            EvictionPolicy::SizeBased => candidates.iter().max_by_key(|s| s.memory_budget),
        };
        best.map(|s| s.id.as_str())
    }

    /// Evict one model according to the current policy. Returns the id
    /// of the evicted model or `None` if no candidate was found.
    pub fn evict_one(&mut self) -> Option<String> {
        let id = self.eviction_candidate()?.to_owned();
        self.unload_model(&id).ok()?;
        Some(id)
    }

    /// Evict models until at least `needed` bytes are free.
    /// Returns the list of evicted model ids.
    pub fn evict_until_free(&mut self, needed: usize) -> Vec<String> {
        let mut evicted = Vec::new();
        while self.available_bytes() < needed {
            match self.evict_one() {
                Some(id) => evicted.push(id),
                None => break,
            }
        }
        evicted
    }

    /// Load a model, evicting others if necessary.
    pub fn load_or_evict(
        &mut self,
        id: impl Into<String>,
        size_bytes: usize,
        priority: u32,
    ) -> Result<Vec<String>, String> {
        let id = id.into();
        let evicted = self.evict_until_free(size_bytes);
        self.load_model(id, size_bytes, priority)?;
        Ok(evicted)
    }

    // -- scheduling helpers --

    /// Return model ids ordered by scheduling priority (highest first),
    /// breaking ties with most-recent access.
    pub fn schedule_order(&self) -> Vec<String> {
        let mut ready: Vec<&ModelSlot> =
            self.slots.values().filter(|s| s.state == ModelState::Ready).collect();
        ready.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then_with(|| b.last_used_tick.cmp(&a.last_used_tick))
        });
        ready.iter().map(|s| s.id.clone()).collect()
    }

    /// Fair round-robin schedule: returns model ids in round-robin order
    /// among `Ready` models sorted by least-recently-used first.
    pub fn fair_round_robin(&self) -> Vec<String> {
        let mut ready: Vec<&ModelSlot> =
            self.slots.values().filter(|s| s.state == ModelState::Ready).collect();
        ready.sort_by_key(|s| s.last_used_tick);
        ready.iter().map(|s| s.id.clone()).collect()
    }

    /// Weighted fair schedule: each model gets a share proportional to its
    /// priority. Returns `(model_id, weight)` pairs summing to 1.0.
    pub fn weighted_fair_shares(&self) -> Vec<(String, f64)> {
        let ready: Vec<&ModelSlot> =
            self.slots.values().filter(|s| s.state == ModelState::Ready).collect();
        let total_priority: u64 = ready.iter().map(|s| s.priority as u64).sum();
        if total_priority == 0 {
            // Equal share when all priorities are zero.
            let n = ready.len();
            if n == 0 {
                return Vec::new();
            }
            let share = 1.0 / n as f64;
            return ready.iter().map(|s| (s.id.clone(), share)).collect();
        }
        ready
            .iter()
            .map(|s| {
                let share = s.priority as f64 / total_priority as f64;
                (s.id.clone(), share)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source string accessor
// ---------------------------------------------------------------------------

/// Returns the OpenCL kernel source for multi-model memory operations.
pub fn kernel_source() -> &'static str {
    MULTI_MODEL_MEMORY_COPY_CL
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===== ModelState =====

    #[test]
    fn model_state_display() {
        assert_eq!(ModelState::Loading.to_string(), "loading");
        assert_eq!(ModelState::Ready.to_string(), "ready");
        assert_eq!(ModelState::Evicting.to_string(), "evicting");
        assert_eq!(ModelState::Evicted.to_string(), "evicted");
    }

    #[test]
    fn model_state_equality() {
        assert_eq!(ModelState::Ready, ModelState::Ready);
        assert_ne!(ModelState::Loading, ModelState::Evicted);
    }

    // ===== EvictionPolicy =====

    #[test]
    fn eviction_policy_display() {
        assert_eq!(EvictionPolicy::LRU.to_string(), "lru");
        assert_eq!(EvictionPolicy::LFU.to_string(), "lfu");
        assert_eq!(EvictionPolicy::Priority.to_string(), "priority");
        assert_eq!(EvictionPolicy::SizeBased.to_string(), "size_based");
    }

    // ===== MemoryPartition =====

    #[test]
    fn partition_new_defaults() {
        let p = MemoryPartition::new("m1", 1024);
        assert_eq!(p.model_id, "m1");
        assert_eq!(p.allocated_bytes, 0);
        assert_eq!(p.budget_bytes, 1024);
        assert_eq!(p.available(), 1024);
    }

    #[test]
    fn partition_allocate_within_budget() {
        let mut p = MemoryPartition::new("m1", 1000);
        assert!(p.allocate(400));
        assert_eq!(p.allocated_bytes, 400);
        assert!(p.allocate(600));
        assert_eq!(p.allocated_bytes, 1000);
        assert_eq!(p.available(), 0);
    }

    #[test]
    fn partition_allocate_exceeds_budget() {
        let mut p = MemoryPartition::new("m1", 500);
        assert!(p.allocate(400));
        assert!(!p.allocate(200));
        assert_eq!(p.allocated_bytes, 400);
    }

    #[test]
    fn partition_deallocate() {
        let mut p = MemoryPartition::new("m1", 1000);
        p.allocate(800);
        p.deallocate(300);
        assert_eq!(p.allocated_bytes, 500);
    }

    #[test]
    fn partition_deallocate_clamps_to_zero() {
        let mut p = MemoryPartition::new("m1", 1000);
        p.allocate(100);
        p.deallocate(500);
        assert_eq!(p.allocated_bytes, 0);
    }

    #[test]
    fn partition_utilization_zero_budget() {
        let p = MemoryPartition::new("m1", 0);
        assert_eq!(p.utilization(), 0.0);
    }

    #[test]
    fn partition_utilization_half() {
        let mut p = MemoryPartition::new("m1", 1000);
        p.allocate(500);
        assert!((p.utilization() - 0.5).abs() < 1e-9);
    }

    // ===== ModelSlot =====

    #[test]
    fn slot_new_starts_loading() {
        let s = ModelSlot::new("test", 2048, 5);
        assert_eq!(s.id, "test");
        assert_eq!(s.memory_budget, 2048);
        assert_eq!(s.priority, 5);
        assert_eq!(s.state, ModelState::Loading);
        assert_eq!(s.access_count, 0);
        assert_eq!(s.last_used_tick, 0);
    }

    #[test]
    fn slot_lifecycle_transitions() {
        let mut s = ModelSlot::new("m", 1024, 1);
        assert_eq!(s.state, ModelState::Loading);
        s.mark_ready();
        assert_eq!(s.state, ModelState::Ready);
        s.mark_evicting();
        assert_eq!(s.state, ModelState::Evicting);
        s.mark_evicted();
        assert_eq!(s.state, ModelState::Evicted);
        assert_eq!(s.partition.allocated_bytes, 0);
    }

    #[test]
    fn slot_touch_increments_counters() {
        let mut s = ModelSlot::new("m", 1024, 1);
        s.touch(10);
        assert_eq!(s.access_count, 1);
        assert_eq!(s.last_used_tick, 10);
        s.touch(20);
        assert_eq!(s.access_count, 2);
        assert_eq!(s.last_used_tick, 20);
    }

    // ===== ModelSwitchCost =====

    #[test]
    fn switch_cost_default() {
        let c = ModelSwitchCost::default();
        assert_eq!(c.base_latency_ns, 500_000);
        assert_eq!(c.per_mb_latency_ns, 100_000);
    }

    #[test]
    fn switch_cost_estimate_small() {
        let c = ModelSwitchCost::new(1000, 10);
        // 512 KB < 1 MB → 0 MB contribution
        assert_eq!(c.estimate_ns(512 * 1024), 1000);
    }

    #[test]
    fn switch_cost_estimate_large() {
        let c = ModelSwitchCost::new(1000, 100);
        // 10 MB
        let expected = 1000 + 10 * 100;
        assert_eq!(c.estimate_ns(10 * 1024 * 1024), expected);
    }

    #[test]
    fn switch_cost_duration() {
        let c = ModelSwitchCost::new(5000, 200);
        let d = c.estimate_duration(2 * 1024 * 1024);
        assert_eq!(d, Duration::from_nanos(5000 + 2 * 200));
    }

    // ===== MultiModelManager basics =====

    #[test]
    fn manager_new() {
        let m = MultiModelManager::new(1_000_000, EvictionPolicy::LRU);
        assert_eq!(m.total_budget(), 1_000_000);
        assert_eq!(m.used_bytes(), 0);
        assert_eq!(m.available_bytes(), 1_000_000);
        assert_eq!(m.slot_count(), 0);
        assert_eq!(m.policy(), EvictionPolicy::LRU);
    }

    #[test]
    fn manager_load_single_model() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        assert!(m.load_model("a", 5000, 1).is_ok());
        assert_eq!(m.used_bytes(), 5000);
        assert_eq!(m.slot_count(), 1);
        let slot = m.get_slot("a").unwrap();
        assert_eq!(slot.state, ModelState::Ready);
    }

    #[test]
    fn manager_load_duplicate_rejected() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        m.load_model("a", 1000, 1).unwrap();
        let err = m.load_model("a", 1000, 1).unwrap_err();
        assert!(err.contains("already loaded"));
    }

    #[test]
    fn manager_load_exceeds_budget() {
        let mut m = MultiModelManager::new(1000, EvictionPolicy::LRU);
        let err = m.load_model("big", 2000, 1).unwrap_err();
        assert!(err.contains("insufficient memory"));
    }

    #[test]
    fn manager_unload_model() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        m.load_model("a", 3000, 1).unwrap();
        m.unload_model("a").unwrap();
        assert_eq!(m.used_bytes(), 0);
        assert_eq!(m.slot_count(), 0);
    }

    #[test]
    fn manager_unload_nonexistent() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        assert!(m.unload_model("ghost").is_err());
    }

    #[test]
    fn manager_access_model() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        m.load_model("a", 1000, 1).unwrap();
        m.access_model("a").unwrap();
        let slot = m.get_slot("a").unwrap();
        assert_eq!(slot.access_count, 2); // load touch + access
    }

    #[test]
    fn manager_access_nonexistent() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        assert!(m.access_model("ghost").is_err());
    }

    // ===== Eviction policy: LRU =====

    #[test]
    fn eviction_lru_selects_oldest() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        m.load_model("old", 1000, 1).unwrap();
        m.load_model("new", 1000, 1).unwrap();
        // "old" was loaded first ⇒ lower tick ⇒ eviction candidate
        assert_eq!(m.eviction_candidate(), Some("old"));
    }

    #[test]
    fn eviction_lru_updates_after_access() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        m.load_model("a", 1000, 1).unwrap();
        m.load_model("b", 1000, 1).unwrap();
        // Access "a" to refresh it
        m.access_model("a").unwrap();
        assert_eq!(m.eviction_candidate(), Some("b"));
    }

    // ===== Eviction policy: LFU =====

    #[test]
    fn eviction_lfu_selects_least_accessed() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LFU);
        m.load_model("hot", 1000, 1).unwrap();
        m.load_model("cold", 1000, 1).unwrap();
        // Access "hot" multiple times
        m.access_model("hot").unwrap();
        m.access_model("hot").unwrap();
        assert_eq!(m.eviction_candidate(), Some("cold"));
    }

    #[test]
    fn eviction_lfu_tie_broken_deterministically() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LFU);
        m.load_model("a", 1000, 1).unwrap();
        m.load_model("b", 1000, 1).unwrap();
        // Both have access_count=1 (from load touch). Either is acceptable.
        let c = m.eviction_candidate().unwrap();
        assert!(c == "a" || c == "b");
    }

    // ===== Eviction policy: Priority =====

    #[test]
    fn eviction_priority_selects_lowest() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::Priority);
        m.load_model("high", 1000, 10).unwrap();
        m.load_model("low", 1000, 1).unwrap();
        assert_eq!(m.eviction_candidate(), Some("low"));
    }

    #[test]
    fn eviction_priority_equal_priorities() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::Priority);
        m.load_model("a", 1000, 5).unwrap();
        m.load_model("b", 1000, 5).unwrap();
        let c = m.eviction_candidate().unwrap();
        assert!(c == "a" || c == "b");
    }

    // ===== Eviction policy: SizeBased =====

    #[test]
    fn eviction_size_selects_largest() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::SizeBased);
        m.load_model("small", 1000, 1).unwrap();
        m.load_model("big", 5000, 1).unwrap();
        assert_eq!(m.eviction_candidate(), Some("big"));
    }

    #[test]
    fn eviction_size_equal_sizes() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::SizeBased);
        m.load_model("a", 2000, 1).unwrap();
        m.load_model("b", 2000, 1).unwrap();
        let c = m.eviction_candidate().unwrap();
        assert!(c == "a" || c == "b");
    }

    // ===== Evict-one / evict-until-free =====

    #[test]
    fn evict_one_returns_id() {
        let mut m = MultiModelManager::new(5000, EvictionPolicy::LRU);
        m.load_model("a", 2000, 1).unwrap();
        m.load_model("b", 2000, 1).unwrap();
        let evicted = m.evict_one().unwrap();
        assert_eq!(evicted, "a"); // "a" is LRU
        assert_eq!(m.slot_count(), 1);
    }

    #[test]
    fn evict_one_empty_returns_none() {
        let mut m = MultiModelManager::new(5000, EvictionPolicy::LRU);
        assert!(m.evict_one().is_none());
    }

    #[test]
    fn evict_until_free_multiple() {
        let mut m = MultiModelManager::new(6000, EvictionPolicy::SizeBased);
        m.load_model("a", 2000, 1).unwrap();
        m.load_model("b", 2000, 1).unwrap();
        m.load_model("c", 2000, 1).unwrap();
        let evicted = m.evict_until_free(5000);
        // Need to free 5000; each slot is 2000 ⇒ evict 3
        assert!(evicted.len() >= 2);
        assert!(m.available_bytes() >= 5000);
    }

    // ===== load_or_evict =====

    #[test]
    fn load_or_evict_no_eviction_needed() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        let evicted = m.load_or_evict("a", 5000, 1).unwrap();
        assert!(evicted.is_empty());
        assert_eq!(m.slot_count(), 1);
    }

    #[test]
    fn load_or_evict_triggers_eviction() {
        let mut m = MultiModelManager::new(5000, EvictionPolicy::LRU);
        m.load_model("old", 3000, 1).unwrap();
        let evicted = m.load_or_evict("new", 4000, 2).unwrap();
        assert_eq!(evicted, vec!["old"]);
        assert_eq!(m.slot_count(), 1);
        assert!(m.get_slot("new").is_some());
    }

    // ===== Scheduling =====

    #[test]
    fn schedule_order_by_priority() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        m.load_model("low", 1000, 1).unwrap();
        m.load_model("high", 1000, 10).unwrap();
        m.load_model("mid", 1000, 5).unwrap();
        let order = m.schedule_order();
        assert_eq!(order[0], "high");
        assert_eq!(order[1], "mid");
        assert_eq!(order[2], "low");
    }

    #[test]
    fn schedule_order_tiebreak_by_recency() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        m.load_model("a", 1000, 5).unwrap();
        m.load_model("b", 1000, 5).unwrap();
        // Access "a" to make it more recent
        m.access_model("a").unwrap();
        let order = m.schedule_order();
        assert_eq!(order[0], "a");
        assert_eq!(order[1], "b");
    }

    #[test]
    fn fair_round_robin_lru_order() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        m.load_model("a", 1000, 1).unwrap();
        m.load_model("b", 1000, 1).unwrap();
        m.load_model("c", 1000, 1).unwrap();
        // "a" has lowest tick → served first in round-robin
        let rr = m.fair_round_robin();
        assert_eq!(rr[0], "a");
    }

    #[test]
    fn weighted_fair_shares_proportional() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        m.load_model("a", 1000, 3).unwrap();
        m.load_model("b", 1000, 1).unwrap();
        let shares = m.weighted_fair_shares();
        let a_share = shares.iter().find(|(id, _)| id == "a").unwrap().1;
        let b_share = shares.iter().find(|(id, _)| id == "b").unwrap().1;
        assert!((a_share - 0.75).abs() < 1e-9);
        assert!((b_share - 0.25).abs() < 1e-9);
    }

    #[test]
    fn weighted_fair_shares_all_zero_priority() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        m.load_model("a", 1000, 0).unwrap();
        m.load_model("b", 1000, 0).unwrap();
        let shares = m.weighted_fair_shares();
        for (_, share) in &shares {
            assert!((*share - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn weighted_fair_shares_empty() {
        let m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        assert!(m.weighted_fair_shares().is_empty());
    }

    // ===== Edge cases =====

    #[test]
    fn zero_budget_rejects_all() {
        let mut m = MultiModelManager::new(0, EvictionPolicy::LRU);
        assert!(m.load_model("a", 1, 1).is_err());
    }

    #[test]
    fn single_slot_evict_and_reload() {
        let mut m = MultiModelManager::new(1000, EvictionPolicy::LRU);
        m.load_model("a", 1000, 1).unwrap();
        m.evict_one();
        assert_eq!(m.slot_count(), 0);
        assert!(m.load_model("b", 1000, 1).is_ok());
    }

    #[test]
    fn all_models_evicted_candidate_is_none() {
        let mut m = MultiModelManager::new(3000, EvictionPolicy::LRU);
        m.load_model("a", 1000, 1).unwrap();
        m.load_model("b", 1000, 1).unwrap();
        m.evict_one();
        m.evict_one();
        assert!(m.eviction_candidate().is_none());
    }

    #[test]
    fn model_ids_returns_all() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        m.load_model("x", 1000, 1).unwrap();
        m.load_model("y", 1000, 1).unwrap();
        let mut ids = m.model_ids();
        ids.sort();
        assert_eq!(ids, vec!["x", "y"]);
    }

    #[test]
    fn estimate_switch_cost_for_known_model() {
        let budget = 4 * 1024 * 1024; // 4 MB
        let mut m = MultiModelManager::new(budget, EvictionPolicy::LRU);
        m.load_model("a", 2 * 1024 * 1024, 1).unwrap();
        let ns = m.estimate_switch_cost("a").unwrap();
        assert!(ns > 0);
    }

    #[test]
    fn estimate_switch_cost_unknown_model() {
        let m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        assert!(m.estimate_switch_cost("ghost").is_none());
    }

    #[test]
    fn kernel_source_not_empty() {
        let src = kernel_source();
        assert!(src.contains("multi_model_memcpy"));
        assert!(src.contains("multi_model_zero_fill"));
    }

    #[test]
    fn manager_debug_format() {
        let m = MultiModelManager::new(4096, EvictionPolicy::Priority);
        let dbg = format!("{m:?}");
        assert!(dbg.contains("MultiModelManager"));
        assert!(dbg.contains("4096"));
    }

    #[test]
    fn with_switch_cost_builder() {
        let cost = ModelSwitchCost::new(999, 111);
        let m = MultiModelManager::new(1000, EvictionPolicy::LRU).with_switch_cost(cost);
        assert_eq!(m.switch_cost().base_latency_ns, 999);
        assert_eq!(m.switch_cost().per_mb_latency_ns, 111);
    }

    #[test]
    fn memory_pressure_eviction_frees_enough() {
        let mut m = MultiModelManager::new(10_000, EvictionPolicy::LRU);
        m.load_model("a", 4000, 1).unwrap();
        m.load_model("b", 4000, 2).unwrap();
        // Need 8000 free — must evict both
        let evicted = m.evict_until_free(8000);
        assert_eq!(evicted.len(), 2);
        assert!(m.available_bytes() >= 8000);
    }

    // ===== Property tests =====

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn partition_alloc_never_exceeds_budget(
                budget in 1usize..10_000,
                alloc in 0usize..20_000,
            ) {
                let mut p = MemoryPartition::new("t", budget);
                let _ = p.allocate(alloc);
                prop_assert!(p.allocated_bytes <= p.budget_bytes);
            }

            #[test]
            fn partition_utilization_in_range(
                budget in 1usize..10_000,
                alloc in 0usize..20_000,
            ) {
                let mut p = MemoryPartition::new("t", budget);
                let _ = p.allocate(alloc);
                let u = p.utilization();
                prop_assert!(u >= 0.0);
                prop_assert!(u <= 1.0);
            }

            #[test]
            fn manager_used_never_exceeds_budget(
                budget in 1000usize..100_000,
                sizes in proptest::collection::vec(100usize..5000, 1..10),
            ) {
                let mut m = MultiModelManager::new(budget, EvictionPolicy::LRU);
                for (i, &size) in sizes.iter().enumerate() {
                    let _ = m.load_model(format!("m{i}"), size, 1);
                }
                prop_assert!(m.used_bytes() <= m.total_budget());
            }

            #[test]
            fn evict_until_free_yields_enough(
                budget in 5000usize..50_000,
                needed in 0usize..50_000,
            ) {
                let mut m = MultiModelManager::new(budget, EvictionPolicy::LRU);
                // Fill with small models
                for i in 0..5 {
                    let _ = m.load_model(format!("m{i}"), budget / 5, 1);
                }
                m.evict_until_free(needed);
                if needed <= budget {
                    prop_assert!(m.available_bytes() >= needed);
                }
            }

            #[test]
            fn weighted_shares_sum_to_one(
                priorities in proptest::collection::vec(0u32..100, 1..8),
            ) {
                let mut m = MultiModelManager::new(1_000_000, EvictionPolicy::LRU);
                for (i, &p) in priorities.iter().enumerate() {
                    let _ = m.load_model(format!("m{i}"), 1000, p);
                }
                let shares = m.weighted_fair_shares();
                if !shares.is_empty() {
                    let total: f64 = shares.iter().map(|(_, s)| s).sum();
                    prop_assert!((total - 1.0).abs() < 1e-6,
                        "shares sum to {} instead of 1.0", total);
                }
            }

            #[test]
            fn schedule_order_length_matches_ready_count(
                count in 1usize..10,
            ) {
                let mut m = MultiModelManager::new(count * 1000, EvictionPolicy::LRU);
                for i in 0..count {
                    let _ = m.load_model(format!("m{i}"), 1000, i as u32);
                }
                let order = m.schedule_order();
                let ready_count = (0..count)
                    .filter(|i| {
                        m.get_slot(&format!("m{i}"))
                            .is_some_and(|s| s.state == ModelState::Ready)
                    })
                    .count();
                prop_assert_eq!(order.len(), ready_count);
            }
        }
    }
}
