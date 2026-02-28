//! Multi-model serving with slot management, routing, and swapping.
//!
//! Enables a single `BitNet` server process to host several models
//! concurrently, routing inference requests to the right model while
//! managing GPU/CPU memory through LRU eviction and hot-model pinning.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the multi-model subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiModelError {
    /// No slot could be allocated (all are occupied and pinned).
    NoSlotsAvailable,
    /// The requested model is not loaded and cannot be found.
    ModelNotFound(String),
    /// Memory budget would be exceeded by loading the model.
    MemoryBudgetExceeded { required: u64, available: u64 },
    /// A slot index is out of range.
    InvalidSlotIndex(usize),
    /// Duplicate model name in preload list.
    DuplicateModel(String),
    /// Configuration validation failed.
    InvalidConfig(String),
}

impl std::fmt::Display for MultiModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSlotsAvailable => write!(f, "no model slots available"),
            Self::ModelNotFound(name) => {
                write!(f, "model not found: {name}")
            }
            Self::MemoryBudgetExceeded { required, available } => {
                write!(
                    f,
                    "memory budget exceeded: need {required} bytes, \
                     {available} available"
                )
            }
            Self::InvalidSlotIndex(idx) => {
                write!(f, "invalid slot index: {idx}")
            }
            Self::DuplicateModel(name) => {
                write!(f, "duplicate model name: {name}")
            }
            Self::InvalidConfig(msg) => {
                write!(f, "invalid config: {msg}")
            }
        }
    }
}

impl std::error::Error for MultiModelError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the [`SlotManager`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSlotConfig {
    /// Maximum number of model slots available.
    pub max_slots: usize,
    /// Total memory budget in bytes (across all slots).
    pub memory_budget: u64,
    /// Models to preload at startup (name → path).
    pub preload_models: Vec<PreloadEntry>,
}

/// A model to preload at startup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreloadEntry {
    /// Logical name used by the router.
    pub name: String,
    /// Filesystem path to the model file.
    pub path: String,
    /// Estimated memory footprint in bytes.
    pub estimated_memory: u64,
}

impl Default for ModelSlotConfig {
    fn default() -> Self {
        Self { max_slots: 4, memory_budget: 8 * 1024 * 1024 * 1024, preload_models: Vec::new() }
    }
}

impl ModelSlotConfig {
    /// Validate configuration values.
    ///
    /// # Errors
    ///
    /// Returns [`MultiModelError::InvalidConfig`] when constraints are
    /// violated.
    pub fn validate(&self) -> Result<(), MultiModelError> {
        if self.max_slots == 0 {
            return Err(MultiModelError::InvalidConfig("max_slots must be > 0".into()));
        }
        if self.memory_budget == 0 {
            return Err(MultiModelError::InvalidConfig("memory_budget must be > 0".into()));
        }
        if self.preload_models.len() > self.max_slots {
            return Err(MultiModelError::InvalidConfig("preload list exceeds max_slots".into()));
        }
        let mut seen = std::collections::HashSet::new();
        for entry in &self.preload_models {
            if !seen.insert(&entry.name) {
                return Err(MultiModelError::DuplicateModel(entry.name.clone()));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Model slot
// ---------------------------------------------------------------------------

/// Residence of a model's weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelResidence {
    /// Weights are in GPU memory — ready for inference.
    Gpu,
    /// Weights have been offloaded to CPU RAM.
    Cpu,
    /// Weights have been offloaded to disk.
    Disk,
}

/// A loaded model occupying a slot.
#[derive(Debug, Clone)]
pub struct ModelSlot {
    /// Logical model name.
    pub name: String,
    /// Filesystem path to the model file.
    pub path: String,
    /// Memory consumed by this slot (bytes).
    pub memory_usage: u64,
    /// Where the model weights currently reside.
    pub residence: ModelResidence,
    /// Whether this slot is pinned (exempt from eviction).
    pub pinned: bool,
    /// Monotonic timestamp of last access for LRU ordering.
    last_access: Instant,
    /// Total number of inference requests served.
    request_count: u64,
}

impl ModelSlot {
    /// Create a new slot in GPU residence.
    pub fn new(name: impl Into<String>, path: impl Into<String>, memory_usage: u64) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            memory_usage,
            residence: ModelResidence::Gpu,
            pinned: false,
            last_access: Instant::now(),
            request_count: 0,
        }
    }

    /// Record an access (updates LRU timestamp and request count).
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
        self.request_count += 1;
    }

    /// Total number of inference requests served by this slot.
    pub fn request_count(&self) -> u64 {
        self.request_count
    }

    /// Last access timestamp.
    pub fn last_access(&self) -> Instant {
        self.last_access
    }
}

// ---------------------------------------------------------------------------
// Slot manager (LRU eviction)
// ---------------------------------------------------------------------------

/// Manages model slots with LRU eviction.
pub struct SlotManager {
    config: ModelSlotConfig,
    slots: HashMap<String, ModelSlot>,
    /// LRU order: front = least-recently-used.
    lru_order: VecDeque<String>,
    memory_used: u64,
}

impl SlotManager {
    /// Create a new [`SlotManager`] from validated configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration validation fails.
    pub fn new(config: ModelSlotConfig) -> Result<Self, MultiModelError> {
        config.validate()?;
        let mut mgr =
            Self { config, slots: HashMap::new(), lru_order: VecDeque::new(), memory_used: 0 };
        let preloads: Vec<_> = mgr.config.preload_models.clone();
        for entry in preloads {
            mgr.allocate(entry.name, entry.path, entry.estimated_memory)?;
        }
        Ok(mgr)
    }

    /// Number of occupied slots.
    pub fn active_slots(&self) -> usize {
        self.slots.len()
    }

    /// Memory currently consumed across all slots.
    pub fn memory_used(&self) -> u64 {
        self.memory_used
    }

    /// Memory remaining within the budget.
    pub fn memory_available(&self) -> u64 {
        self.config.memory_budget.saturating_sub(self.memory_used)
    }

    /// Return a reference to a slot by model name.
    pub fn get(&self, name: &str) -> Option<&ModelSlot> {
        self.slots.get(name)
    }

    /// Return a mutable reference to a slot and refresh its LRU position.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ModelSlot> {
        if self.slots.contains_key(name) {
            self.promote_lru(name);
        }
        self.slots.get_mut(name)
    }

    /// Allocate a new slot, evicting LRU models if necessary.
    ///
    /// # Errors
    ///
    /// Returns [`MultiModelError`] when the model cannot be loaded.
    pub fn allocate(
        &mut self,
        name: impl Into<String>,
        path: impl Into<String>,
        memory: u64,
    ) -> Result<(), MultiModelError> {
        let name = name.into();
        if self.slots.contains_key(&name) {
            return Ok(());
        }
        // Evict until we have capacity.
        while self.slots.len() >= self.config.max_slots
            || self.memory_used + memory > self.config.memory_budget
        {
            if !self.evict_one()? {
                // Nothing left to evict.
                if self.memory_used + memory > self.config.memory_budget {
                    return Err(MultiModelError::MemoryBudgetExceeded {
                        required: memory,
                        available: self.memory_available(),
                    });
                }
                return Err(MultiModelError::NoSlotsAvailable);
            }
        }
        let path = path.into();
        let slot = ModelSlot::new(name.clone(), path, memory);
        self.slots.insert(name.clone(), slot);
        self.lru_order.push_back(name);
        self.memory_used += memory;
        Ok(())
    }

    /// Free a slot by model name, returning the released memory.
    pub fn free(&mut self, name: &str) -> Option<u64> {
        let slot = self.slots.remove(name)?;
        self.lru_order.retain(|n| n != name);
        self.memory_used = self.memory_used.saturating_sub(slot.memory_usage);
        Some(slot.memory_usage)
    }

    /// Pin a model so it won't be evicted.
    pub fn pin(&mut self, name: &str) -> Result<(), MultiModelError> {
        self.slots
            .get_mut(name)
            .ok_or_else(|| MultiModelError::ModelNotFound(name.into()))?
            .pinned = true;
        Ok(())
    }

    /// Unpin a previously pinned model.
    pub fn unpin(&mut self, name: &str) -> Result<(), MultiModelError> {
        self.slots
            .get_mut(name)
            .ok_or_else(|| MultiModelError::ModelNotFound(name.into()))?
            .pinned = false;
        Ok(())
    }

    /// Names of all loaded models.
    pub fn loaded_models(&self) -> Vec<String> {
        self.slots.keys().cloned().collect()
    }

    // -- internal helpers ---------------------------------------------------

    /// Evict the least-recently-used **unpinned** model. Returns `true` if a
    /// model was evicted, `false` if nothing eligible could be found.
    fn evict_one(&mut self) -> Result<bool, MultiModelError> {
        let victim = self
            .lru_order
            .iter()
            .find(|n| self.slots.get(n.as_str()).is_some_and(|s| !s.pinned))
            .cloned();
        if let Some(name) = victim {
            self.free(&name);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn promote_lru(&mut self, name: &str) {
        self.lru_order.retain(|n| n != name);
        self.lru_order.push_back(name.to_string());
    }
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

/// Strategy for routing inference requests to models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Route by explicit model name in the request.
    ByModelName,
    /// Route based on the request type (e.g. chat vs completion).
    ByRequestType(HashMap<String, String>),
    /// Distribute across replicas round-robin.
    RoundRobin,
    /// Send to the replica with the fewest in-flight requests.
    LeastLoaded,
}

impl Default for RoutingStrategy {
    fn default() -> Self {
        Self::ByModelName
    }
}

/// An inference request to be routed.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Explicit model name (used by `ByModelName`).
    pub model_name: Option<String>,
    /// Request type tag (used by `ByRequestType`).
    pub request_type: Option<String>,
    /// The prompt text.
    pub prompt: String,
}

/// Routes inference requests to the correct model slot.
pub struct ModelRouter {
    strategy: RoutingStrategy,
    default_model: Option<String>,
    /// Round-robin counter.
    rr_counter: usize,
}

impl ModelRouter {
    /// Create a new router with the given strategy.
    pub fn new(strategy: RoutingStrategy) -> Self {
        Self { strategy, default_model: None, rr_counter: 0 }
    }

    /// Set a default model for when routing cannot determine one.
    pub fn set_default_model(&mut self, name: impl Into<String>) {
        self.default_model = Some(name.into());
    }

    /// Current routing strategy.
    pub fn strategy(&self) -> &RoutingStrategy {
        &self.strategy
    }

    /// Resolve which model should handle the request.
    ///
    /// # Errors
    ///
    /// Returns [`MultiModelError::ModelNotFound`] when no model can be
    /// determined.
    pub fn route(
        &mut self,
        request: &InferenceRequest,
        available: &[String],
    ) -> Result<String, MultiModelError> {
        if available.is_empty() {
            return Err(MultiModelError::NoSlotsAvailable);
        }
        match &self.strategy {
            RoutingStrategy::ByModelName => {
                let name = request.model_name.as_deref().or(self.default_model.as_deref());
                match name {
                    Some(n) if available.contains(&n.to_string()) => Ok(n.to_string()),
                    Some(n) => Err(MultiModelError::ModelNotFound(n.to_string())),
                    None => Err(MultiModelError::ModelNotFound("<no model specified>".into())),
                }
            }
            RoutingStrategy::ByRequestType(mapping) => {
                let target = request
                    .request_type
                    .as_deref()
                    .and_then(|rt| mapping.get(rt))
                    .or(self.default_model.as_ref());
                match target {
                    Some(n) if available.contains(n) => Ok(n.clone()),
                    Some(n) => Err(MultiModelError::ModelNotFound(n.clone())),
                    None => Err(MultiModelError::ModelNotFound("<unmapped request type>".into())),
                }
            }
            RoutingStrategy::RoundRobin => {
                let idx = self.rr_counter % available.len();
                self.rr_counter = self.rr_counter.wrapping_add(1);
                Ok(available[idx].clone())
            }
            RoutingStrategy::LeastLoaded => {
                // Without runtime load info we fall back to first.
                Ok(available[0].clone())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Swap policy
// ---------------------------------------------------------------------------

/// Policy used by [`ModelSwapper`] to choose eviction candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwapPolicy {
    /// Evict the least-recently-used model.
    LRU,
    /// Evict the least-frequently-used model.
    LFU,
    /// Evict models with the lowest priority score.
    Priority,
    /// Do not auto-evict; the caller manages swaps explicitly.
    Manual,
}

impl Default for SwapPolicy {
    fn default() -> Self {
        Self::LRU
    }
}

/// Swaps models between GPU, CPU, and disk tiers.
pub struct ModelSwapper {
    policy: SwapPolicy,
    swap_history: Vec<SwapEvent>,
}

/// A record of a single swap operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapEvent {
    /// Model that was swapped.
    pub model_name: String,
    /// Where it came from.
    pub from: ModelResidence,
    /// Where it went.
    pub to: ModelResidence,
}

impl ModelSwapper {
    /// Create a swapper with the given policy.
    pub fn new(policy: SwapPolicy) -> Self {
        Self { policy, swap_history: Vec::new() }
    }

    /// Current swap policy.
    pub fn policy(&self) -> SwapPolicy {
        self.policy
    }

    /// Request that `model_name` be moved to `target` residence.
    ///
    /// # Errors
    ///
    /// Returns [`MultiModelError::ModelNotFound`] when the model isn't
    /// loaded.
    pub fn swap(
        &mut self,
        slots: &mut SlotManager,
        model_name: &str,
        target: ModelResidence,
    ) -> Result<(), MultiModelError> {
        let slot = slots
            .get_mut(model_name)
            .ok_or_else(|| MultiModelError::ModelNotFound(model_name.into()))?;
        let from = slot.residence;
        slot.residence = target;
        self.swap_history.push(SwapEvent { model_name: model_name.to_string(), from, to: target });
        Ok(())
    }

    /// Choose an eviction candidate from the slot manager using the
    /// configured policy, ignoring pinned models.
    pub fn pick_eviction_candidate(&self, slots: &SlotManager) -> Option<String> {
        let candidates: Vec<_> = slots
            .slots
            .values()
            .filter(|s| !s.pinned && s.residence == ModelResidence::Gpu)
            .collect();
        if candidates.is_empty() {
            return None;
        }
        match self.policy {
            SwapPolicy::LRU => {
                candidates.iter().min_by_key(|s| s.last_access).map(|s| s.name.clone())
            }
            SwapPolicy::LFU => {
                candidates.iter().min_by_key(|s| s.request_count).map(|s| s.name.clone())
            }
            SwapPolicy::Priority => {
                // Lowest request_count used as proxy for priority.
                candidates.iter().min_by_key(|s| s.request_count).map(|s| s.name.clone())
            }
            SwapPolicy::Manual => None,
        }
    }

    /// Number of swaps that have occurred.
    pub fn swap_count(&self) -> usize {
        self.swap_history.len()
    }

    /// Full swap history.
    pub fn history(&self) -> &[SwapEvent] {
        &self.swap_history
    }
}

// ---------------------------------------------------------------------------
// Load balancer
// ---------------------------------------------------------------------------

/// Distributes requests across model replicas.
pub struct ModelLoadBalancer {
    /// Model name → list of replica identifiers.
    replicas: HashMap<String, Vec<String>>,
    /// Per-replica in-flight request count.
    in_flight: HashMap<String, usize>,
    /// Round-robin counters per model.
    rr_counters: HashMap<String, usize>,
}

impl ModelLoadBalancer {
    /// Create a new, empty load balancer.
    pub fn new() -> Self {
        Self { replicas: HashMap::new(), in_flight: HashMap::new(), rr_counters: HashMap::new() }
    }

    /// Register a replica for `model_name`.
    pub fn add_replica(&mut self, model_name: impl Into<String>, replica_id: impl Into<String>) {
        let model = model_name.into();
        let replica = replica_id.into();
        self.replicas.entry(model).or_default().push(replica.clone());
        self.in_flight.entry(replica).or_insert(0);
    }

    /// Pick the replica with the fewest in-flight requests.
    pub fn pick_least_loaded(&self, model_name: &str) -> Option<String> {
        self.replicas.get(model_name).and_then(|reps| {
            reps.iter().min_by_key(|r| self.in_flight.get(*r).copied().unwrap_or(0)).cloned()
        })
    }

    /// Pick a replica using round-robin.
    pub fn pick_round_robin(&mut self, model_name: &str) -> Option<String> {
        let reps = self.replicas.get(model_name)?;
        if reps.is_empty() {
            return None;
        }
        let counter = self.rr_counters.entry(model_name.into()).or_insert(0);
        let idx = *counter % reps.len();
        *counter = counter.wrapping_add(1);
        Some(reps[idx].clone())
    }

    /// Increment in-flight counter for a replica.
    pub fn begin_request(&mut self, replica_id: &str) {
        *self.in_flight.entry(replica_id.into()).or_insert(0) += 1;
    }

    /// Decrement in-flight counter for a replica.
    pub fn end_request(&mut self, replica_id: &str) {
        if let Some(count) = self.in_flight.get_mut(replica_id) {
            *count = count.saturating_sub(1);
        }
    }

    /// Number of replicas for a model.
    pub fn replica_count(&self, model_name: &str) -> usize {
        self.replicas.get(model_name).map_or(0, |r| r.len())
    }
}

impl Default for ModelLoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Per-model performance metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetricsEntry {
    /// Total inference requests served.
    pub total_requests: u64,
    /// Cumulative tokens generated.
    pub total_tokens: u64,
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Memory consumption in bytes.
    pub memory_bytes: u64,
    /// Number of swap operations involving this model.
    pub swap_count: u64,
}

/// Aggregated metrics across all loaded models.
pub struct MultiModelMetrics {
    per_model: HashMap<String, ModelMetricsEntry>,
}

impl MultiModelMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self { per_model: HashMap::new() }
    }

    /// Record a completed request for `model_name`.
    pub fn record_request(&mut self, model_name: &str, tokens: u64, latency_ms: f64) {
        let entry = self.per_model.entry(model_name.into()).or_default();
        entry.total_requests += 1;
        entry.total_tokens += tokens;
        // Running average.
        let n = entry.total_requests as f64;
        entry.avg_latency_ms = entry.avg_latency_ms * ((n - 1.0) / n) + latency_ms / n;
    }

    /// Record a swap event for `model_name`.
    pub fn record_swap(&mut self, model_name: &str) {
        self.per_model.entry(model_name.into()).or_default().swap_count += 1;
    }

    /// Update memory usage for `model_name`.
    pub fn set_memory(&mut self, model_name: &str, bytes: u64) {
        self.per_model.entry(model_name.into()).or_default().memory_bytes = bytes;
    }

    /// Get metrics for a specific model.
    pub fn get(&self, model_name: &str) -> Option<&ModelMetricsEntry> {
        self.per_model.get(model_name)
    }

    /// Snapshot of all per-model metrics.
    pub fn snapshot(&self) -> &HashMap<String, ModelMetricsEntry> {
        &self.per_model
    }

    /// Total requests across all models.
    pub fn total_requests(&self) -> u64 {
        self.per_model.values().map(|e| e.total_requests).sum()
    }
}

impl Default for MultiModelMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Multi-model server (top-level façade)
// ---------------------------------------------------------------------------

/// Top-level server managing multiple loaded models.
pub struct MultiModelServer {
    slots: SlotManager,
    router: ModelRouter,
    swapper: ModelSwapper,
    balancer: ModelLoadBalancer,
    metrics: MultiModelMetrics,
}

impl MultiModelServer {
    /// Create a new server with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration validation or preloading fails.
    pub fn new(
        config: ModelSlotConfig,
        routing: RoutingStrategy,
        swap_policy: SwapPolicy,
    ) -> Result<Self, MultiModelError> {
        let slots = SlotManager::new(config)?;
        Ok(Self {
            slots,
            router: ModelRouter::new(routing),
            swapper: ModelSwapper::new(swap_policy),
            balancer: ModelLoadBalancer::new(),
            metrics: MultiModelMetrics::new(),
        })
    }

    /// Load a model into a slot.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot cannot be allocated.
    pub fn load_model(
        &mut self,
        name: impl Into<String>,
        path: impl Into<String>,
        memory: u64,
    ) -> Result<(), MultiModelError> {
        let name = name.into();
        let path = path.into();
        self.slots.allocate(name.clone(), path, memory)?;
        self.metrics.set_memory(&name, memory);
        Ok(())
    }

    /// Unload a model, freeing its slot.
    pub fn unload_model(&mut self, name: &str) -> Option<u64> {
        self.slots.free(name)
    }

    /// Pin a model to prevent eviction.
    pub fn pin_model(&mut self, name: &str) -> Result<(), MultiModelError> {
        self.slots.pin(name)
    }

    /// Unpin a previously pinned model.
    pub fn unpin_model(&mut self, name: &str) -> Result<(), MultiModelError> {
        self.slots.unpin(name)
    }

    /// Route an inference request and touch the target slot.
    ///
    /// # Errors
    ///
    /// Returns an error if routing fails.
    pub fn route_request(&mut self, request: &InferenceRequest) -> Result<String, MultiModelError> {
        let available = self.slots.loaded_models();
        let target = self.router.route(request, &available)?;
        if let Some(slot) = self.slots.get_mut(&target) {
            slot.touch();
        }
        Ok(target)
    }

    /// Swap a model to the given residence tier.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not loaded.
    pub fn swap_model(
        &mut self,
        name: &str,
        target: ModelResidence,
    ) -> Result<(), MultiModelError> {
        self.swapper.swap(&mut self.slots, name, target)?;
        self.metrics.record_swap(name);
        Ok(())
    }

    /// Record completion of an inference request in metrics.
    pub fn record_completion(&mut self, model_name: &str, tokens: u64, latency_ms: f64) {
        self.metrics.record_request(model_name, tokens, latency_ms);
    }

    /// Number of active model slots.
    pub fn active_slots(&self) -> usize {
        self.slots.active_slots()
    }

    /// Access the slot manager.
    pub fn slot_manager(&self) -> &SlotManager {
        &self.slots
    }

    /// Access metrics.
    pub fn metrics(&self) -> &MultiModelMetrics {
        &self.metrics
    }

    /// Access the load balancer.
    pub fn load_balancer(&self) -> &ModelLoadBalancer {
        &self.balancer
    }

    /// Mutable access to the load balancer.
    pub fn load_balancer_mut(&mut self) -> &mut ModelLoadBalancer {
        &mut self.balancer
    }

    /// Set the default model for routing.
    pub fn set_default_model(&mut self, name: impl Into<String>) {
        self.router.set_default_model(name);
    }

    /// List names of all loaded models.
    pub fn loaded_models(&self) -> Vec<String> {
        self.slots.loaded_models()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- ModelSlotConfig tests ----------------------------------------------

    #[test]
    fn default_config_is_valid() {
        let cfg = ModelSlotConfig::default();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.max_slots, 4);
    }

    #[test]
    fn config_zero_slots_invalid() {
        let cfg = ModelSlotConfig { max_slots: 0, ..Default::default() };
        assert!(matches!(cfg.validate(), Err(MultiModelError::InvalidConfig(_))));
    }

    #[test]
    fn config_zero_budget_invalid() {
        let cfg = ModelSlotConfig { memory_budget: 0, ..Default::default() };
        assert!(matches!(cfg.validate(), Err(MultiModelError::InvalidConfig(_))));
    }

    #[test]
    fn config_preload_exceeds_slots() {
        let cfg = ModelSlotConfig {
            max_slots: 1,
            preload_models: vec![
                PreloadEntry { name: "a".into(), path: "a.gguf".into(), estimated_memory: 100 },
                PreloadEntry { name: "b".into(), path: "b.gguf".into(), estimated_memory: 100 },
            ],
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(MultiModelError::InvalidConfig(_))));
    }

    #[test]
    fn config_duplicate_preload_names() {
        let cfg = ModelSlotConfig {
            max_slots: 4,
            preload_models: vec![
                PreloadEntry { name: "m".into(), path: "m.gguf".into(), estimated_memory: 100 },
                PreloadEntry { name: "m".into(), path: "m2.gguf".into(), estimated_memory: 100 },
            ],
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(MultiModelError::DuplicateModel(_))));
    }

    #[test]
    fn config_serialization_roundtrip() {
        let cfg = ModelSlotConfig {
            max_slots: 2,
            memory_budget: 4096,
            preload_models: vec![PreloadEntry {
                name: "test".into(),
                path: "test.gguf".into(),
                estimated_memory: 1024,
            }],
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: ModelSlotConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_slots, 2);
        assert_eq!(back.memory_budget, 4096);
        assert_eq!(back.preload_models.len(), 1);
    }

    // -- ModelSlot tests ----------------------------------------------------

    #[test]
    fn model_slot_new_defaults() {
        let slot = ModelSlot::new("m", "m.gguf", 1024);
        assert_eq!(slot.name, "m");
        assert_eq!(slot.residence, ModelResidence::Gpu);
        assert!(!slot.pinned);
        assert_eq!(slot.request_count(), 0);
    }

    #[test]
    fn model_slot_touch_increments() {
        let mut slot = ModelSlot::new("m", "m.gguf", 1024);
        let before = slot.last_access();
        slot.touch();
        assert_eq!(slot.request_count(), 1);
        assert!(slot.last_access() >= before);
    }

    // -- SlotManager tests --------------------------------------------------

    #[test]
    fn slot_manager_allocate_and_get() {
        let cfg = ModelSlotConfig { max_slots: 2, memory_budget: 2048, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("a", "a.gguf", 512).unwrap();
        assert_eq!(mgr.active_slots(), 1);
        assert_eq!(mgr.memory_used(), 512);
        assert!(mgr.get("a").is_some());
    }

    #[test]
    fn slot_manager_duplicate_allocate_is_noop() {
        let cfg = ModelSlotConfig { max_slots: 2, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("a", "a.gguf", 512).unwrap();
        mgr.allocate("a", "a.gguf", 512).unwrap();
        assert_eq!(mgr.active_slots(), 1);
    }

    #[test]
    fn slot_manager_free_releases_memory() {
        let cfg = ModelSlotConfig { max_slots: 2, memory_budget: 2048, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("a", "a.gguf", 512).unwrap();
        let freed = mgr.free("a");
        assert_eq!(freed, Some(512));
        assert_eq!(mgr.active_slots(), 0);
        assert_eq!(mgr.memory_used(), 0);
    }

    #[test]
    fn slot_manager_free_nonexistent_returns_none() {
        let cfg = ModelSlotConfig { max_slots: 2, memory_budget: 2048, preload_models: vec![] };
        let mgr = SlotManager::new(cfg).unwrap();
        assert!(mgr.slots.get("nope").is_none());
    }

    #[test]
    fn slot_manager_lru_eviction() {
        let cfg = ModelSlotConfig { max_slots: 2, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("a", "a.gguf", 100).unwrap();
        mgr.allocate("b", "b.gguf", 100).unwrap();
        // Accessing "a" promotes it; "b" stays LRU.
        mgr.get_mut("a");
        mgr.allocate("c", "c.gguf", 100).unwrap();
        assert!(mgr.get("b").is_none(), "b should have been evicted");
        assert!(mgr.get("a").is_some());
        assert!(mgr.get("c").is_some());
    }

    #[test]
    fn slot_manager_pinned_model_not_evicted() {
        let cfg = ModelSlotConfig { max_slots: 2, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("a", "a.gguf", 100).unwrap();
        mgr.pin("a").unwrap();
        mgr.allocate("b", "b.gguf", 100).unwrap();
        // "a" is pinned, "b" is the only eviction candidate.
        mgr.allocate("c", "c.gguf", 100).unwrap();
        assert!(mgr.get("a").is_some(), "pinned model must survive");
        assert!(mgr.get("b").is_none(), "b should be evicted");
        assert!(mgr.get("c").is_some());
    }

    #[test]
    fn slot_manager_all_pinned_returns_error() {
        let cfg = ModelSlotConfig { max_slots: 1, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("a", "a.gguf", 100).unwrap();
        mgr.pin("a").unwrap();
        let err = mgr.allocate("b", "b.gguf", 100).unwrap_err();
        assert!(matches!(err, MultiModelError::NoSlotsAvailable));
    }

    #[test]
    fn slot_manager_memory_budget_exceeded() {
        let cfg = ModelSlotConfig { max_slots: 4, memory_budget: 500, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("a", "a.gguf", 400).unwrap();
        mgr.pin("a").unwrap();
        let err = mgr.allocate("b", "b.gguf", 400).unwrap_err();
        assert!(matches!(err, MultiModelError::MemoryBudgetExceeded { .. }));
    }

    #[test]
    fn slot_manager_preload() {
        let cfg = ModelSlotConfig {
            max_slots: 2,
            memory_budget: 4096,
            preload_models: vec![PreloadEntry {
                name: "pre".into(),
                path: "pre.gguf".into(),
                estimated_memory: 256,
            }],
        };
        let mgr = SlotManager::new(cfg).unwrap();
        assert_eq!(mgr.active_slots(), 1);
        assert!(mgr.get("pre").is_some());
    }

    #[test]
    fn slot_manager_loaded_models_list() {
        let cfg = ModelSlotConfig { max_slots: 4, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("x", "x.gguf", 100).unwrap();
        mgr.allocate("y", "y.gguf", 100).unwrap();
        let mut names = mgr.loaded_models();
        names.sort();
        assert_eq!(names, vec!["x", "y"]);
    }

    #[test]
    fn slot_manager_unpin() {
        let cfg = ModelSlotConfig { max_slots: 2, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("a", "a.gguf", 100).unwrap();
        mgr.pin("a").unwrap();
        assert!(mgr.get("a").unwrap().pinned);
        mgr.unpin("a").unwrap();
        assert!(!mgr.get("a").unwrap().pinned);
    }

    #[test]
    fn slot_manager_pin_nonexistent_errors() {
        let cfg = ModelSlotConfig { max_slots: 2, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        assert!(matches!(mgr.pin("nope"), Err(MultiModelError::ModelNotFound(_))));
    }

    #[test]
    fn slot_manager_memory_available() {
        let cfg = ModelSlotConfig { max_slots: 4, memory_budget: 1000, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        assert_eq!(mgr.memory_available(), 1000);
        mgr.allocate("a", "a.gguf", 400).unwrap();
        assert_eq!(mgr.memory_available(), 600);
    }

    // -- Routing tests ------------------------------------------------------

    #[test]
    fn route_by_model_name() {
        let mut router = ModelRouter::new(RoutingStrategy::ByModelName);
        let req = InferenceRequest {
            model_name: Some("alpha".into()),
            request_type: None,
            prompt: "hi".into(),
        };
        let avail = vec!["alpha".into(), "beta".into()];
        assert_eq!(router.route(&req, &avail).unwrap(), "alpha");
    }

    #[test]
    fn route_by_model_name_missing() {
        let mut router = ModelRouter::new(RoutingStrategy::ByModelName);
        let req = InferenceRequest {
            model_name: Some("gamma".into()),
            request_type: None,
            prompt: "hi".into(),
        };
        let avail = vec!["alpha".into()];
        assert!(matches!(router.route(&req, &avail), Err(MultiModelError::ModelNotFound(_))));
    }

    #[test]
    fn route_by_model_name_default() {
        let mut router = ModelRouter::new(RoutingStrategy::ByModelName);
        router.set_default_model("beta");
        let req = InferenceRequest { model_name: None, request_type: None, prompt: "hi".into() };
        let avail = vec!["alpha".into(), "beta".into()];
        assert_eq!(router.route(&req, &avail).unwrap(), "beta");
    }

    #[test]
    fn route_by_model_name_no_model_no_default() {
        let mut router = ModelRouter::new(RoutingStrategy::ByModelName);
        let req = InferenceRequest { model_name: None, request_type: None, prompt: "hi".into() };
        let avail = vec!["a".into()];
        assert!(router.route(&req, &avail).is_err());
    }

    #[test]
    fn route_round_robin() {
        let mut router = ModelRouter::new(RoutingStrategy::RoundRobin);
        let avail = vec!["a".into(), "b".into(), "c".into()];
        let req = InferenceRequest { model_name: None, request_type: None, prompt: "hi".into() };
        let r0 = router.route(&req, &avail).unwrap();
        let r1 = router.route(&req, &avail).unwrap();
        let r2 = router.route(&req, &avail).unwrap();
        let r3 = router.route(&req, &avail).unwrap();
        assert_eq!(r0, "a");
        assert_eq!(r1, "b");
        assert_eq!(r2, "c");
        assert_eq!(r3, "a"); // wraps
    }

    #[test]
    fn route_by_request_type() {
        let mut map = HashMap::new();
        map.insert("chat".into(), "chat-model".into());
        map.insert("completion".into(), "comp-model".into());
        let mut router = ModelRouter::new(RoutingStrategy::ByRequestType(map));
        let avail = vec!["chat-model".into(), "comp-model".into()];
        let req = InferenceRequest {
            model_name: None,
            request_type: Some("chat".into()),
            prompt: "hi".into(),
        };
        assert_eq!(router.route(&req, &avail).unwrap(), "chat-model");
    }

    #[test]
    fn route_by_request_type_unmapped() {
        let map = HashMap::new();
        let mut router = ModelRouter::new(RoutingStrategy::ByRequestType(map));
        let req = InferenceRequest {
            model_name: None,
            request_type: Some("unknown".into()),
            prompt: "hi".into(),
        };
        let avail = vec!["a".into()];
        assert!(router.route(&req, &avail).is_err());
    }

    #[test]
    fn route_least_loaded_returns_first() {
        let mut router = ModelRouter::new(RoutingStrategy::LeastLoaded);
        let avail = vec!["a".into(), "b".into()];
        let req = InferenceRequest { model_name: None, request_type: None, prompt: "hi".into() };
        assert_eq!(router.route(&req, &avail).unwrap(), "a");
    }

    #[test]
    fn route_empty_available_errors() {
        let mut router = ModelRouter::new(RoutingStrategy::RoundRobin);
        let req = InferenceRequest { model_name: None, request_type: None, prompt: "hi".into() };
        assert!(matches!(router.route(&req, &[]), Err(MultiModelError::NoSlotsAvailable)));
    }

    #[test]
    fn router_strategy_accessor() {
        let router = ModelRouter::new(RoutingStrategy::RoundRobin);
        assert_eq!(*router.strategy(), RoutingStrategy::RoundRobin);
    }

    // -- Swap tests ---------------------------------------------------------

    #[test]
    fn swap_model_changes_residence() {
        let cfg = ModelSlotConfig { max_slots: 2, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("a", "a.gguf", 100).unwrap();
        let mut swapper = ModelSwapper::new(SwapPolicy::LRU);
        swapper.swap(&mut mgr, "a", ModelResidence::Cpu).unwrap();
        assert_eq!(mgr.get("a").unwrap().residence, ModelResidence::Cpu);
        assert_eq!(swapper.swap_count(), 1);
    }

    #[test]
    fn swap_nonexistent_model_errors() {
        let cfg = ModelSlotConfig { max_slots: 2, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        let mut swapper = ModelSwapper::new(SwapPolicy::LRU);
        assert!(matches!(
            swapper.swap(&mut mgr, "nope", ModelResidence::Cpu),
            Err(MultiModelError::ModelNotFound(_))
        ));
    }

    #[test]
    fn swap_history_records_events() {
        let cfg = ModelSlotConfig { max_slots: 2, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("a", "a.gguf", 100).unwrap();
        let mut swapper = ModelSwapper::new(SwapPolicy::LRU);
        swapper.swap(&mut mgr, "a", ModelResidence::Cpu).unwrap();
        swapper.swap(&mut mgr, "a", ModelResidence::Disk).unwrap();
        let hist = swapper.history();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0].from, ModelResidence::Gpu);
        assert_eq!(hist[0].to, ModelResidence::Cpu);
        assert_eq!(hist[1].from, ModelResidence::Cpu);
        assert_eq!(hist[1].to, ModelResidence::Disk);
    }

    #[test]
    fn swap_policy_default_is_lru() {
        assert_eq!(SwapPolicy::default(), SwapPolicy::LRU);
    }

    #[test]
    fn swap_pick_eviction_lru() {
        let cfg = ModelSlotConfig { max_slots: 4, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("old", "old.gguf", 100).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        mgr.allocate("new", "new.gguf", 100).unwrap();
        let swapper = ModelSwapper::new(SwapPolicy::LRU);
        let victim = swapper.pick_eviction_candidate(&mgr);
        assert_eq!(victim, Some("old".into()));
    }

    #[test]
    fn swap_pick_eviction_lfu() {
        let cfg = ModelSlotConfig { max_slots: 4, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("hot", "h.gguf", 100).unwrap();
        mgr.allocate("cold", "c.gguf", 100).unwrap();
        // Touch "hot" several times.
        for _ in 0..5 {
            mgr.get_mut("hot").unwrap().touch();
        }
        let swapper = ModelSwapper::new(SwapPolicy::LFU);
        let victim = swapper.pick_eviction_candidate(&mgr);
        assert_eq!(victim, Some("cold".into()));
    }

    #[test]
    fn swap_pick_eviction_manual_returns_none() {
        let cfg = ModelSlotConfig { max_slots: 4, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("a", "a.gguf", 100).unwrap();
        let swapper = ModelSwapper::new(SwapPolicy::Manual);
        assert!(swapper.pick_eviction_candidate(&mgr).is_none());
    }

    #[test]
    fn swap_pick_eviction_skips_pinned() {
        let cfg = ModelSlotConfig { max_slots: 4, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("pinned", "p.gguf", 100).unwrap();
        mgr.pin("pinned").unwrap();
        mgr.allocate("free", "f.gguf", 100).unwrap();
        let swapper = ModelSwapper::new(SwapPolicy::LRU);
        assert_eq!(swapper.pick_eviction_candidate(&mgr), Some("free".into()));
    }

    #[test]
    fn swap_pick_eviction_skips_non_gpu() {
        let cfg = ModelSlotConfig { max_slots: 4, memory_budget: 4096, preload_models: vec![] };
        let mut mgr = SlotManager::new(cfg).unwrap();
        mgr.allocate("cpu_model", "c.gguf", 100).unwrap();
        mgr.get_mut("cpu_model").unwrap().residence = ModelResidence::Cpu;
        let swapper = ModelSwapper::new(SwapPolicy::LRU);
        assert!(swapper.pick_eviction_candidate(&mgr).is_none());
    }

    // -- Load balancer tests ------------------------------------------------

    #[test]
    fn load_balancer_add_and_count() {
        let mut lb = ModelLoadBalancer::new();
        lb.add_replica("m", "r1");
        lb.add_replica("m", "r2");
        assert_eq!(lb.replica_count("m"), 2);
        assert_eq!(lb.replica_count("other"), 0);
    }

    #[test]
    fn load_balancer_least_loaded() {
        let mut lb = ModelLoadBalancer::new();
        lb.add_replica("m", "r1");
        lb.add_replica("m", "r2");
        lb.begin_request("r1");
        lb.begin_request("r1");
        lb.begin_request("r2");
        assert_eq!(lb.pick_least_loaded("m"), Some("r2".into()));
    }

    #[test]
    fn load_balancer_round_robin() {
        let mut lb = ModelLoadBalancer::new();
        lb.add_replica("m", "r1");
        lb.add_replica("m", "r2");
        assert_eq!(lb.pick_round_robin("m"), Some("r1".into()));
        assert_eq!(lb.pick_round_robin("m"), Some("r2".into()));
        assert_eq!(lb.pick_round_robin("m"), Some("r1".into()));
    }

    #[test]
    fn load_balancer_end_request() {
        let mut lb = ModelLoadBalancer::new();
        lb.add_replica("m", "r1");
        lb.begin_request("r1");
        lb.begin_request("r1");
        lb.end_request("r1");
        // r1 has 1 in-flight now.
        lb.add_replica("m", "r2");
        assert_eq!(lb.pick_least_loaded("m"), Some("r2".into()));
    }

    #[test]
    fn load_balancer_no_replicas() {
        let lb = ModelLoadBalancer::new();
        assert!(lb.pick_least_loaded("none").is_none());
    }

    #[test]
    fn load_balancer_default() {
        let lb = ModelLoadBalancer::default();
        assert_eq!(lb.replica_count("any"), 0);
    }

    // -- Metrics tests ------------------------------------------------------

    #[test]
    fn metrics_record_request() {
        let mut m = MultiModelMetrics::new();
        m.record_request("a", 10, 50.0);
        m.record_request("a", 20, 100.0);
        let entry = m.get("a").unwrap();
        assert_eq!(entry.total_requests, 2);
        assert_eq!(entry.total_tokens, 30);
    }

    #[test]
    fn metrics_record_swap() {
        let mut m = MultiModelMetrics::new();
        m.record_swap("a");
        m.record_swap("a");
        assert_eq!(m.get("a").unwrap().swap_count, 2);
    }

    #[test]
    fn metrics_set_memory() {
        let mut m = MultiModelMetrics::new();
        m.set_memory("a", 9999);
        assert_eq!(m.get("a").unwrap().memory_bytes, 9999);
    }

    #[test]
    fn metrics_total_requests() {
        let mut m = MultiModelMetrics::new();
        m.record_request("a", 1, 1.0);
        m.record_request("b", 1, 1.0);
        m.record_request("b", 1, 1.0);
        assert_eq!(m.total_requests(), 3);
    }

    #[test]
    fn metrics_snapshot_returns_all() {
        let mut m = MultiModelMetrics::new();
        m.record_request("x", 1, 1.0);
        m.record_request("y", 1, 1.0);
        assert_eq!(m.snapshot().len(), 2);
    }

    #[test]
    fn metrics_default() {
        let m = MultiModelMetrics::default();
        assert_eq!(m.total_requests(), 0);
    }

    #[test]
    fn metrics_entry_serialization_roundtrip() {
        let entry = ModelMetricsEntry {
            total_requests: 42,
            total_tokens: 1000,
            avg_latency_ms: 12.5,
            memory_bytes: 2048,
            swap_count: 3,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: ModelMetricsEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.total_requests, 42);
        assert_eq!(back.total_tokens, 1000);
    }

    // -- MultiModelServer integration tests --------------------------------

    #[test]
    fn server_load_and_unload() {
        let cfg = ModelSlotConfig { max_slots: 4, memory_budget: 4096, preload_models: vec![] };
        let mut srv =
            MultiModelServer::new(cfg, RoutingStrategy::ByModelName, SwapPolicy::LRU).unwrap();
        srv.load_model("a", "a.gguf", 512).unwrap();
        assert_eq!(srv.active_slots(), 1);
        assert!(srv.loaded_models().contains(&"a".into()));
        srv.unload_model("a");
        assert_eq!(srv.active_slots(), 0);
    }

    #[test]
    fn server_route_and_record() {
        let cfg = ModelSlotConfig { max_slots: 4, memory_budget: 4096, preload_models: vec![] };
        let mut srv =
            MultiModelServer::new(cfg, RoutingStrategy::ByModelName, SwapPolicy::LRU).unwrap();
        srv.load_model("alpha", "a.gguf", 256).unwrap();
        srv.set_default_model("alpha");
        let req = InferenceRequest {
            model_name: Some("alpha".into()),
            request_type: None,
            prompt: "hello".into(),
        };
        let target = srv.route_request(&req).unwrap();
        assert_eq!(target, "alpha");
        srv.record_completion("alpha", 10, 50.0);
        assert_eq!(srv.metrics().total_requests(), 1);
    }

    #[test]
    fn server_swap_model() {
        let cfg = ModelSlotConfig { max_slots: 4, memory_budget: 4096, preload_models: vec![] };
        let mut srv =
            MultiModelServer::new(cfg, RoutingStrategy::ByModelName, SwapPolicy::LRU).unwrap();
        srv.load_model("a", "a.gguf", 256).unwrap();
        srv.swap_model("a", ModelResidence::Cpu).unwrap();
        assert_eq!(srv.slot_manager().get("a").unwrap().residence, ModelResidence::Cpu);
    }

    #[test]
    fn server_pin_and_unpin() {
        let cfg = ModelSlotConfig { max_slots: 2, memory_budget: 4096, preload_models: vec![] };
        let mut srv =
            MultiModelServer::new(cfg, RoutingStrategy::ByModelName, SwapPolicy::LRU).unwrap();
        srv.load_model("a", "a.gguf", 100).unwrap();
        srv.pin_model("a").unwrap();
        assert!(srv.slot_manager().get("a").unwrap().pinned);
        srv.unpin_model("a").unwrap();
        assert!(!srv.slot_manager().get("a").unwrap().pinned);
    }

    #[test]
    fn server_cold_start_preload() {
        let cfg = ModelSlotConfig {
            max_slots: 4,
            memory_budget: 4096,
            preload_models: vec![PreloadEntry {
                name: "hot".into(),
                path: "hot.gguf".into(),
                estimated_memory: 256,
            }],
        };
        let srv =
            MultiModelServer::new(cfg, RoutingStrategy::ByModelName, SwapPolicy::LRU).unwrap();
        assert_eq!(srv.active_slots(), 1);
        assert!(srv.slot_manager().get("hot").is_some());
    }

    #[test]
    fn server_load_balancer_access() {
        let cfg = ModelSlotConfig::default();
        let mut srv =
            MultiModelServer::new(cfg, RoutingStrategy::ByModelName, SwapPolicy::LRU).unwrap();
        srv.load_balancer_mut().add_replica("m", "r1");
        assert_eq!(srv.load_balancer().replica_count("m"), 1);
    }

    // -- Error display tests -----------------------------------------------

    #[test]
    fn error_display_no_slots() {
        let e = MultiModelError::NoSlotsAvailable;
        assert!(e.to_string().contains("no model slots"));
    }

    #[test]
    fn error_display_model_not_found() {
        let e = MultiModelError::ModelNotFound("x".into());
        assert!(e.to_string().contains("x"));
    }

    #[test]
    fn error_display_memory_budget() {
        let e = MultiModelError::MemoryBudgetExceeded { required: 100, available: 50 };
        let s = e.to_string();
        assert!(s.contains("100"));
        assert!(s.contains("50"));
    }

    #[test]
    fn error_display_invalid_slot_index() {
        let e = MultiModelError::InvalidSlotIndex(99);
        assert!(e.to_string().contains("99"));
    }

    #[test]
    fn error_display_duplicate() {
        let e = MultiModelError::DuplicateModel("d".into());
        assert!(e.to_string().contains("d"));
    }

    #[test]
    fn error_display_invalid_config() {
        let e = MultiModelError::InvalidConfig("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    // -- Routing strategy default ------------------------------------------

    #[test]
    fn routing_strategy_default_is_by_model_name() {
        assert_eq!(RoutingStrategy::default(), RoutingStrategy::ByModelName);
    }

    #[test]
    fn routing_strategy_serialization_roundtrip() {
        let strat = RoutingStrategy::RoundRobin;
        let json = serde_json::to_string(&strat).unwrap();
        let back: RoutingStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, RoutingStrategy::RoundRobin);
    }

    // -- Residence enum tests -----------------------------------------------

    #[test]
    fn residence_serialization_roundtrip() {
        for r in [ModelResidence::Gpu, ModelResidence::Cpu, ModelResidence::Disk] {
            let json = serde_json::to_string(&r).unwrap();
            let back: ModelResidence = serde_json::from_str(&json).unwrap();
            assert_eq!(back, r);
        }
    }
}
