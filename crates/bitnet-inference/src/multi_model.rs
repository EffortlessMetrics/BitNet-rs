//! Multi-model GPU serving: load, manage, and hot-swap multiple models
//! on a single GPU with memory partitioning and LRU eviction.

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from the multi-model serving subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiModelError {
    /// Model with this ID is already loaded.
    AlreadyLoaded { model_id: String },
    /// Model with this ID is not loaded.
    NotLoaded { model_id: String },
    /// Not enough GPU memory to load the model.
    OutOfMemory { requested_bytes: usize, available_bytes: usize },
    /// Memory partition exceeds the total pool.
    InvalidPartition { model_id: String, requested_bytes: usize, pool_bytes: usize },
    /// Eviction failed to free enough memory.
    EvictionFailed { needed_bytes: usize },
}

impl fmt::Display for MultiModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyLoaded { model_id } => {
                write!(f, "model {model_id:?} is already loaded")
            }
            Self::NotLoaded { model_id } => write!(f, "model {model_id:?} is not loaded"),
            Self::OutOfMemory { requested_bytes, available_bytes } => {
                write!(f, "out of GPU memory: need {requested_bytes} B, have {available_bytes} B")
            }
            Self::InvalidPartition { model_id, requested_bytes, pool_bytes } => write!(
                f,
                "partition for {model_id:?} ({requested_bytes} B) exceeds pool ({pool_bytes} B)"
            ),
            Self::EvictionFailed { needed_bytes } => {
                write!(f, "eviction failed to free {needed_bytes} B")
            }
        }
    }
}

impl std::error::Error for MultiModelError {}

// ---------------------------------------------------------------------------
// Model slot
// ---------------------------------------------------------------------------

/// Metadata for a loaded model.
#[derive(Debug, Clone)]
pub struct ModelSlot {
    /// Unique identifier for this model.
    pub model_id: String,
    /// GPU memory consumed by this model (bytes).
    pub memory_bytes: usize,
    /// When this model was loaded.
    pub loaded_at: Instant,
    /// When this model was last used for inference.
    pub last_used: Instant,
}

// ---------------------------------------------------------------------------
// GPU memory pool
// ---------------------------------------------------------------------------

/// Tracks GPU memory allocation across models.
#[derive(Debug)]
pub struct GpuMemoryPool {
    total_bytes: usize,
    used_bytes: usize,
}

impl GpuMemoryPool {
    pub fn new(total_bytes: usize) -> Self {
        Self { total_bytes, used_bytes: 0 }
    }

    /// Total pool capacity in bytes.
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Bytes currently in use.
    #[inline]
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Bytes currently free.
    #[inline]
    pub fn available_bytes(&self) -> usize {
        self.total_bytes.saturating_sub(self.used_bytes)
    }

    /// Try to reserve `bytes`.  Returns `Ok(())` or an OOM error.
    pub fn reserve(&mut self, bytes: usize) -> Result<(), MultiModelError> {
        if bytes > self.available_bytes() {
            return Err(MultiModelError::OutOfMemory {
                requested_bytes: bytes,
                available_bytes: self.available_bytes(),
            });
        }
        self.used_bytes += bytes;
        Ok(())
    }

    /// Release `bytes` back to the pool.
    pub fn release(&mut self, bytes: usize) {
        self.used_bytes = self.used_bytes.saturating_sub(bytes);
    }
}

// ---------------------------------------------------------------------------
// Multi-model manager
// ---------------------------------------------------------------------------

/// Manages multiple models on a single GPU with memory partitioning and LRU
/// eviction.
#[derive(Debug)]
pub struct MultiModelManager {
    pool: GpuMemoryPool,
    slots: HashMap<String, ModelSlot>,
}

impl MultiModelManager {
    /// Create a new manager with a GPU memory pool of `total_gpu_bytes`.
    pub fn new(total_gpu_bytes: usize) -> Self {
        Self { pool: GpuMemoryPool::new(total_gpu_bytes), slots: HashMap::new() }
    }

    /// Number of models currently loaded.
    #[inline]
    pub fn model_count(&self) -> usize {
        self.slots.len()
    }

    /// Immutable access to the memory pool.
    #[inline]
    pub fn pool(&self) -> &GpuMemoryPool {
        &self.pool
    }

    /// List IDs of all loaded models.
    pub fn loaded_model_ids(&self) -> Vec<String> {
        self.slots.keys().cloned().collect()
    }

    /// Check whether `model_id` is currently loaded.
    pub fn is_loaded(&self, model_id: &str) -> bool {
        self.slots.contains_key(model_id)
    }

    // -- load / unload ----------------------------------------------------

    /// Load a model into GPU memory.
    ///
    /// If `memory_bytes` exceeds the pool this returns
    /// [`MultiModelError::InvalidPartition`].  If the pool cannot satisfy the
    /// request (even after eviction attempts) it returns
    /// [`MultiModelError::OutOfMemory`].
    pub fn load_model(
        &mut self,
        model_id: impl Into<String>,
        memory_bytes: usize,
    ) -> Result<(), MultiModelError> {
        let model_id = model_id.into();
        if self.slots.contains_key(&model_id) {
            return Err(MultiModelError::AlreadyLoaded { model_id: model_id.clone() });
        }
        if memory_bytes > self.pool.total_bytes() {
            return Err(MultiModelError::InvalidPartition {
                model_id: model_id.clone(),
                requested_bytes: memory_bytes,
                pool_bytes: self.pool.total_bytes(),
            });
        }
        self.pool.reserve(memory_bytes)?;
        let now = Instant::now();
        self.slots.insert(
            model_id.clone(),
            ModelSlot { model_id, memory_bytes, loaded_at: now, last_used: now },
        );
        Ok(())
    }

    /// Unload a model, freeing its GPU memory partition.
    pub fn unload_model(&mut self, model_id: &str) -> Result<ModelSlot, MultiModelError> {
        let slot = self
            .slots
            .remove(model_id)
            .ok_or_else(|| MultiModelError::NotLoaded { model_id: model_id.to_string() })?;
        self.pool.release(slot.memory_bytes);
        Ok(slot)
    }

    /// Hot-swap: atomically unload `old_id` and load `new_id`.
    ///
    /// If the new model requires more memory than the old one freed, the
    /// difference must be available in the pool.
    pub fn hot_swap(
        &mut self,
        old_id: &str,
        new_id: impl Into<String>,
        new_memory_bytes: usize,
    ) -> Result<(), MultiModelError> {
        let freed = self.unload_model(old_id)?;
        match self.load_model(new_id, new_memory_bytes) {
            Ok(()) => Ok(()),
            Err(e) => {
                // Roll back: re-load the old model.
                let now = Instant::now();
                self.pool.reserve(freed.memory_bytes).expect("rollback reserve should never fail");
                self.slots.insert(
                    freed.model_id.clone(),
                    ModelSlot {
                        model_id: freed.model_id,
                        memory_bytes: freed.memory_bytes,
                        loaded_at: freed.loaded_at,
                        last_used: now,
                    },
                );
                Err(e)
            }
        }
    }

    // -- LRU eviction -----------------------------------------------------

    /// Record a use of `model_id` (updates `last_used` timestamp).
    pub fn touch(&mut self, model_id: &str) -> Result<(), MultiModelError> {
        let slot = self
            .slots
            .get_mut(model_id)
            .ok_or_else(|| MultiModelError::NotLoaded { model_id: model_id.to_string() })?;
        slot.last_used = Instant::now();
        Ok(())
    }

    /// Return the model ID of the least-recently-used model, if any.
    pub fn lru_model_id(&self) -> Option<String> {
        self.slots.values().min_by_key(|s| s.last_used).map(|s| s.model_id.clone())
    }

    /// Evict the LRU model to free memory.  Returns the evicted slot.
    pub fn evict_lru(&mut self) -> Result<ModelSlot, MultiModelError> {
        let lru_id =
            self.lru_model_id().ok_or(MultiModelError::EvictionFailed { needed_bytes: 0 })?;
        self.unload_model(&lru_id)
    }

    /// Evict LRU models until at least `needed_bytes` are free.
    ///
    /// Returns the list of evicted model IDs.
    pub fn evict_until_free(
        &mut self,
        needed_bytes: usize,
    ) -> Result<Vec<String>, MultiModelError> {
        let mut evicted = Vec::new();
        while self.pool.available_bytes() < needed_bytes {
            if self.slots.is_empty() {
                return Err(MultiModelError::EvictionFailed { needed_bytes });
            }
            let slot = self.evict_lru()?;
            evicted.push(slot.model_id);
        }
        Ok(evicted)
    }

    /// Load a model, evicting LRU models if necessary to make room.
    pub fn load_or_evict(
        &mut self,
        model_id: impl Into<String>,
        memory_bytes: usize,
    ) -> Result<Vec<String>, MultiModelError> {
        let model_id = model_id.into();
        if self.slots.contains_key(&model_id) {
            return Err(MultiModelError::AlreadyLoaded { model_id: model_id.clone() });
        }
        if memory_bytes > self.pool.total_bytes() {
            return Err(MultiModelError::InvalidPartition {
                model_id: model_id.clone(),
                requested_bytes: memory_bytes,
                pool_bytes: self.pool.total_bytes(),
            });
        }
        let evicted = self.evict_until_free(memory_bytes)?;
        self.load_model(model_id, memory_bytes)?;
        Ok(evicted)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const GB: usize = 1024 * 1024 * 1024;

    #[test]
    fn test_load_and_unload_model() {
        let mut mgr = MultiModelManager::new(8 * GB);
        mgr.load_model("model-a", 2 * GB).unwrap();
        assert!(mgr.is_loaded("model-a"));
        assert_eq!(mgr.model_count(), 1);
        assert_eq!(mgr.pool().used_bytes(), 2 * GB);

        mgr.unload_model("model-a").unwrap();
        assert!(!mgr.is_loaded("model-a"));
        assert_eq!(mgr.pool().used_bytes(), 0);
    }

    #[test]
    fn test_multiple_models_memory_partitioning() {
        let mut mgr = MultiModelManager::new(8 * GB);
        mgr.load_model("model-a", 2 * GB).unwrap();
        mgr.load_model("model-b", 3 * GB).unwrap();
        assert_eq!(mgr.model_count(), 2);
        assert_eq!(mgr.pool().used_bytes(), 5 * GB);
        assert_eq!(mgr.pool().available_bytes(), 3 * GB);
    }

    #[test]
    fn test_load_rejects_duplicate() {
        let mut mgr = MultiModelManager::new(8 * GB);
        mgr.load_model("model-a", 1 * GB).unwrap();
        assert!(matches!(
            mgr.load_model("model-a", 1 * GB),
            Err(MultiModelError::AlreadyLoaded { .. })
        ));
    }

    #[test]
    fn test_load_rejects_oversized_model() {
        let mut mgr = MultiModelManager::new(4 * GB);
        assert!(matches!(
            mgr.load_model("huge", 8 * GB),
            Err(MultiModelError::InvalidPartition { .. })
        ));
    }

    #[test]
    fn test_load_rejects_when_pool_full() {
        let mut mgr = MultiModelManager::new(4 * GB);
        mgr.load_model("model-a", 3 * GB).unwrap();
        assert!(matches!(
            mgr.load_model("model-b", 2 * GB),
            Err(MultiModelError::OutOfMemory { .. })
        ));
    }

    #[test]
    fn test_hot_swap_same_size() {
        let mut mgr = MultiModelManager::new(4 * GB);
        mgr.load_model("old", 2 * GB).unwrap();
        mgr.hot_swap("old", "new", 2 * GB).unwrap();
        assert!(!mgr.is_loaded("old"));
        assert!(mgr.is_loaded("new"));
        assert_eq!(mgr.pool().used_bytes(), 2 * GB);
    }

    #[test]
    fn test_hot_swap_rollback_on_failure() {
        let mut mgr = MultiModelManager::new(4 * GB);
        mgr.load_model("old", 2 * GB).unwrap();
        mgr.load_model("other", 1 * GB).unwrap();
        // Try to swap "old" (2 GB) for "big" (4 GB) — not enough room.
        let err = mgr.hot_swap("old", "big", 4 * GB);
        assert!(err.is_err());
        // "old" should be restored after rollback.
        assert!(mgr.is_loaded("old"));
        assert_eq!(mgr.model_count(), 2);
    }

    #[test]
    fn test_lru_eviction_frees_oldest() {
        let mut mgr = MultiModelManager::new(8 * GB);
        mgr.load_model("model-a", 2 * GB).unwrap();
        // Ensure model-b has a later timestamp
        std::thread::sleep(std::time::Duration::from_millis(10));
        mgr.load_model("model-b", 2 * GB).unwrap();

        let evicted = mgr.evict_lru().unwrap();
        assert_eq!(evicted.model_id, "model-a");
        assert!(!mgr.is_loaded("model-a"));
        assert!(mgr.is_loaded("model-b"));
    }

    #[test]
    fn test_evict_until_free() {
        let mut mgr = MultiModelManager::new(8 * GB);
        mgr.load_model("a", 2 * GB).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        mgr.load_model("b", 2 * GB).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        mgr.load_model("c", 2 * GB).unwrap();

        // 8 GB pool, 6 GB used, 2 GB free. Need 5 GB → evict a + b (4 GB freed → 6 GB free).
        let evicted = mgr.evict_until_free(5 * GB).unwrap();
        assert_eq!(evicted.len(), 2);
        assert!(evicted.contains(&"a".to_string()));
        assert!(evicted.contains(&"b".to_string()));
        assert!(mgr.is_loaded("c"));
    }

    #[test]
    fn test_load_or_evict_triggers_eviction() {
        let mut mgr = MultiModelManager::new(4 * GB);
        mgr.load_model("old", 3 * GB).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        let evicted = mgr.load_or_evict("new", 3 * GB).unwrap();
        assert_eq!(evicted, vec!["old".to_string()]);
        assert!(!mgr.is_loaded("old"));
        assert!(mgr.is_loaded("new"));
    }

    #[test]
    fn test_touch_updates_last_used() {
        let mut mgr = MultiModelManager::new(8 * GB);
        mgr.load_model("model-a", 1 * GB).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        mgr.load_model("model-b", 1 * GB).unwrap();

        // model-a is older, but touch it to make it "recent"
        std::thread::sleep(std::time::Duration::from_millis(10));
        mgr.touch("model-a").unwrap();

        // LRU should now be model-b
        assert_eq!(mgr.lru_model_id().unwrap(), "model-b");
    }

    #[test]
    fn test_unload_nonexistent_model() {
        let mut mgr = MultiModelManager::new(4 * GB);
        assert!(matches!(mgr.unload_model("ghost"), Err(MultiModelError::NotLoaded { .. })));
    }

    #[test]
    fn test_gpu_memory_pool_reserve_and_release() {
        let mut pool = GpuMemoryPool::new(100);
        pool.reserve(60).unwrap();
        assert_eq!(pool.available_bytes(), 40);
        pool.release(30);
        assert_eq!(pool.available_bytes(), 70);
        pool.release(200); // saturating
        assert_eq!(pool.used_bytes(), 0);
    }
}
