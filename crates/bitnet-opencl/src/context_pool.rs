//! GPU context pool for rapid OpenCL context switching.
//!
//! Manages multiple OpenCL contexts to enable fast model switching
//! without recompilation overhead. Supports lazy creation, per-model
//! caching, and memory-pressure eviction.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use log::{debug, info, warn};

/// Unique identifier for a cached context, typically derived from the model path.
pub type ContextId = String;

/// Memory usage in bytes reported by the context.
pub type MemoryBytes = u64;

/// Error types specific to context pool operations.
#[derive(Debug, thiserror::Error)]
pub enum ContextPoolError {
    #[error("context pool capacity exhausted (max {max})")]
    CapacityExhausted { max: usize },

    #[error("context creation failed for '{id}': {reason}")]
    CreationFailed { id: String, reason: String },

    #[error("context '{id}' not found in pool")]
    NotFound { id: String },

    #[error("memory pressure exceeded threshold ({used} / {limit} bytes)")]
    MemoryPressure { used: MemoryBytes, limit: MemoryBytes },

    #[error("lock poisoned: {0}")]
    LockPoisoned(String),
}

/// Configuration for the context pool.
#[derive(Debug, Clone)]
pub struct ContextPoolConfig {
    /// Maximum number of contexts that can be cached simultaneously.
    pub max_contexts: usize,
    /// Memory limit in bytes; eviction triggers when total exceeds this.
    pub memory_limit: MemoryBytes,
    /// Time after which an unused context becomes eligible for eviction.
    pub idle_timeout: Duration,
    /// Whether to create contexts lazily (on first use) or eagerly.
    pub lazy_creation: bool,
}

impl Default for ContextPoolConfig {
    fn default() -> Self {
        Self {
            max_contexts: 4,
            memory_limit: 2 * 1024 * 1024 * 1024, // 2 GiB
            idle_timeout: Duration::from_secs(300),
            lazy_creation: true,
        }
    }
}

/// Metadata about a cached OpenCL context.
#[derive(Debug, Clone)]
pub struct ContextEntry {
    /// Unique identifier (usually model path or hash).
    pub id: ContextId,
    /// Estimated GPU memory consumed by compiled programs in this context.
    pub memory_usage: MemoryBytes,
    /// When this context was created.
    pub created_at: Instant,
    /// Last time this context was acquired for use.
    pub last_used: Instant,
    /// Number of times this context has been acquired.
    pub use_count: u64,
    /// Whether the context is currently in active use.
    pub in_use: bool,
    /// Whether programs have been compiled (false if lazy and not yet used).
    pub compiled: bool,
}

impl ContextEntry {
    fn new(id: ContextId, memory_usage: MemoryBytes) -> Self {
        let now = Instant::now();
        Self {
            id,
            memory_usage,
            created_at: now,
            last_used: now,
            use_count: 0,
            in_use: false,
            compiled: false,
        }
    }

    fn idle_duration(&self) -> Duration {
        self.last_used.elapsed()
    }
}

/// Trait for context factories that create real OpenCL contexts.
///
/// Abstracted to allow testing without actual GPU hardware.
pub trait ContextFactory: Send + Sync {
    /// Create a new context for the given model id.
    /// Returns the estimated memory usage of the created context.
    fn create_context(&self, id: &str) -> Result<MemoryBytes, ContextPoolError>;

    /// Compile programs for a lazily-created context.
    fn compile_programs(&self, id: &str) -> Result<(), ContextPoolError>;

    /// Release GPU resources for a context.
    fn release_context(&self, id: &str) -> Result<(), ContextPoolError>;

    /// Query current total GPU memory usage.
    fn total_gpu_memory_used(&self) -> MemoryBytes;
}

/// A pool of OpenCL contexts that supports rapid switching between models.
///
/// # Design
///
/// - **Per-model caching**: Each model gets its own context with pre-compiled
///   kernels, avoiding recompilation on switch.
/// - **Lazy creation**: Contexts are only fully initialized (programs compiled)
///   when first used, reducing startup time.
/// - **LRU eviction**: When memory pressure is high or the pool is full,
///   the least-recently-used idle context is evicted.
pub struct ContextPool {
    config: ContextPoolConfig,
    entries: Arc<Mutex<HashMap<ContextId, ContextEntry>>>,
    factory: Arc<dyn ContextFactory>,
}

impl ContextPool {
    /// Create a new context pool with the given configuration and factory.
    pub fn new(config: ContextPoolConfig, factory: Arc<dyn ContextFactory>) -> Self {
        info!(
            "ContextPool created: max_contexts={}, memory_limit={}MiB, lazy={}",
            config.max_contexts,
            config.memory_limit / (1024 * 1024),
            config.lazy_creation
        );
        Self { config, entries: Arc::new(Mutex::new(HashMap::new())), factory }
    }

    /// Acquire a context for the given model, creating it if needed.
    ///
    /// If the context already exists it is returned immediately.
    /// If `lazy_creation` is enabled, programs are compiled on first acquire.
    pub fn acquire(&self, id: &str) -> Result<ContextEntry, ContextPoolError> {
        let mut entries =
            self.entries.lock().map_err(|e| ContextPoolError::LockPoisoned(e.to_string()))?;

        if let Some(entry) = entries.get_mut(id) {
            // Existing context — ensure programs are compiled.
            if !entry.compiled {
                self.factory.compile_programs(id)?;
                entry.compiled = true;
            }
            entry.last_used = Instant::now();
            entry.use_count += 1;
            entry.in_use = true;
            debug!("Acquired existing context '{}' (use_count={})", id, entry.use_count);
            return Ok(entry.clone());
        }

        // Need to create a new context — check capacity.
        if entries.len() >= self.config.max_contexts {
            // Try to evict an idle context first.
            if !Self::evict_lru(&mut entries, &self.factory, &self.config)? {
                return Err(ContextPoolError::CapacityExhausted { max: self.config.max_contexts });
            }
        }

        // Check memory pressure before creating.
        self.check_memory_pressure(&entries)?;

        // Create the context.
        let memory_usage = self.factory.create_context(id)?;
        let mut entry = ContextEntry::new(id.to_string(), memory_usage);

        if !self.config.lazy_creation {
            self.factory.compile_programs(id)?;
            entry.compiled = true;
        }

        entry.use_count = 1;
        entry.in_use = true;
        entry.last_used = Instant::now();

        // If lazy, we compile on first acquire.
        if self.config.lazy_creation && !entry.compiled {
            self.factory.compile_programs(id)?;
            entry.compiled = true;
        }

        let result = entry.clone();
        entries.insert(id.to_string(), entry);
        info!("Created new context '{}' (memory={}KiB)", id, memory_usage / 1024);
        Ok(result)
    }

    /// Release a context back to the pool (mark as not in-use).
    pub fn release(&self, id: &str) -> Result<(), ContextPoolError> {
        let mut entries =
            self.entries.lock().map_err(|e| ContextPoolError::LockPoisoned(e.to_string()))?;

        let entry =
            entries.get_mut(id).ok_or_else(|| ContextPoolError::NotFound { id: id.to_string() })?;

        entry.in_use = false;
        debug!("Released context '{}'", id);
        Ok(())
    }

    /// Explicitly evict a specific context, freeing its GPU resources.
    pub fn evict(&self, id: &str) -> Result<(), ContextPoolError> {
        let mut entries =
            self.entries.lock().map_err(|e| ContextPoolError::LockPoisoned(e.to_string()))?;

        if entries.remove(id).is_some() {
            self.factory.release_context(id)?;
            info!("Evicted context '{}'", id);
            Ok(())
        } else {
            Err(ContextPoolError::NotFound { id: id.to_string() })
        }
    }

    /// Return the number of contexts currently in the pool.
    pub fn len(&self) -> usize {
        self.entries.lock().map(|e| e.len()).unwrap_or(0)
    }

    /// Return whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return total memory usage across all cached contexts.
    pub fn total_memory_usage(&self) -> MemoryBytes {
        self.entries.lock().map(|e| e.values().map(|v| v.memory_usage).sum()).unwrap_or(0)
    }

    /// Return a snapshot of all context entries.
    pub fn entries(&self) -> Vec<ContextEntry> {
        self.entries.lock().map(|e| e.values().cloned().collect()).unwrap_or_default()
    }

    /// Evict all idle contexts that have exceeded the idle timeout.
    pub fn evict_expired(&self) -> Result<usize, ContextPoolError> {
        let mut entries =
            self.entries.lock().map_err(|e| ContextPoolError::LockPoisoned(e.to_string()))?;

        let timeout = self.config.idle_timeout;
        let expired: Vec<ContextId> = entries
            .iter()
            .filter(|(_, e)| !e.in_use && e.idle_duration() > timeout)
            .map(|(k, _)| k.clone())
            .collect();

        let count = expired.len();
        for id in &expired {
            entries.remove(id);
            let _ = self.factory.release_context(id);
            debug!("Evicted expired context '{}'", id);
        }

        if count > 0 {
            info!("Evicted {} expired context(s)", count);
        }
        Ok(count)
    }

    /// Check whether total memory exceeds the configured limit.
    fn check_memory_pressure(
        &self,
        entries: &HashMap<ContextId, ContextEntry>,
    ) -> Result<(), ContextPoolError> {
        let total: MemoryBytes = entries.values().map(|e| e.memory_usage).sum();
        if total >= self.config.memory_limit {
            warn!("Memory pressure: {} / {} bytes used", total, self.config.memory_limit);
            return Err(ContextPoolError::MemoryPressure {
                used: total,
                limit: self.config.memory_limit,
            });
        }
        Ok(())
    }

    /// Evict the least-recently-used idle context. Returns true if one was evicted.
    fn evict_lru(
        entries: &mut HashMap<ContextId, ContextEntry>,
        factory: &Arc<dyn ContextFactory>,
        _config: &ContextPoolConfig,
    ) -> Result<bool, ContextPoolError> {
        // Find the LRU idle context.
        let lru_id = entries
            .iter()
            .filter(|(_, e)| !e.in_use)
            .min_by_key(|(_, e)| e.last_used)
            .map(|(k, _)| k.clone());

        if let Some(id) = lru_id {
            entries.remove(&id);
            factory.release_context(&id)?;
            info!("LRU-evicted context '{}'", id);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

// SAFETY: ContextPool uses Arc<Mutex<_>> internally for all mutable state.
unsafe impl Send for ContextPool {}
unsafe impl Sync for ContextPool {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Mock factory that tracks creation/release without real OpenCL.
    struct MockFactory {
        create_count: AtomicU64,
        release_count: AtomicU64,
        compile_count: AtomicU64,
        memory_per_context: MemoryBytes,
        fail_on_create: Mutex<Option<String>>,
    }

    impl MockFactory {
        fn new(memory_per_context: MemoryBytes) -> Self {
            Self {
                create_count: AtomicU64::new(0),
                release_count: AtomicU64::new(0),
                compile_count: AtomicU64::new(0),
                memory_per_context,
                fail_on_create: Mutex::new(None),
            }
        }

        fn set_fail_on_create(&self, id: &str) {
            *self.fail_on_create.lock().unwrap() = Some(id.to_string());
        }

        fn creates(&self) -> u64 {
            self.create_count.load(Ordering::SeqCst)
        }

        fn releases(&self) -> u64 {
            self.release_count.load(Ordering::SeqCst)
        }

        fn compiles(&self) -> u64 {
            self.compile_count.load(Ordering::SeqCst)
        }
    }

    impl ContextFactory for MockFactory {
        fn create_context(&self, id: &str) -> Result<MemoryBytes, ContextPoolError> {
            let fail = self.fail_on_create.lock().unwrap();
            if let Some(ref fail_id) = *fail {
                if fail_id == id {
                    return Err(ContextPoolError::CreationFailed {
                        id: id.to_string(),
                        reason: "mock failure".into(),
                    });
                }
            }
            self.create_count.fetch_add(1, Ordering::SeqCst);
            Ok(self.memory_per_context)
        }

        fn compile_programs(&self, _id: &str) -> Result<(), ContextPoolError> {
            self.compile_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn release_context(&self, _id: &str) -> Result<(), ContextPoolError> {
            self.release_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn total_gpu_memory_used(&self) -> MemoryBytes {
            0
        }
    }

    fn default_pool(factory: Arc<MockFactory>) -> ContextPool {
        ContextPool::new(ContextPoolConfig::default(), factory)
    }

    #[test]
    fn test_acquire_creates_new_context() {
        let factory = Arc::new(MockFactory::new(1024));
        let pool = default_pool(factory.clone());

        let entry = pool.acquire("model-a").unwrap();
        assert_eq!(entry.id, "model-a");
        assert_eq!(entry.use_count, 1);
        assert!(entry.compiled);
        assert_eq!(factory.creates(), 1);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_acquire_reuses_cached_context() {
        let factory = Arc::new(MockFactory::new(1024));
        let pool = default_pool(factory.clone());

        pool.acquire("model-a").unwrap();
        pool.release("model-a").unwrap();

        let entry = pool.acquire("model-a").unwrap();
        assert_eq!(entry.use_count, 2);
        // Only one create call, context was reused.
        assert_eq!(factory.creates(), 1);
    }

    #[test]
    fn test_lazy_context_compiles_on_acquire() {
        let factory = Arc::new(MockFactory::new(1024));
        let config = ContextPoolConfig { lazy_creation: true, ..Default::default() };
        let pool = ContextPool::new(config, factory.clone());

        let entry = pool.acquire("model-lazy").unwrap();
        assert!(entry.compiled);
        assert_eq!(factory.compiles(), 1);
    }

    #[test]
    fn test_capacity_exhausted_without_idle() {
        let factory = Arc::new(MockFactory::new(1024));
        let config = ContextPoolConfig { max_contexts: 2, ..Default::default() };
        let pool = ContextPool::new(config, factory.clone());

        pool.acquire("m1").unwrap();
        pool.acquire("m2").unwrap();
        // Both contexts are in_use, so LRU eviction cannot free any.
        let err = pool.acquire("m3").unwrap_err();
        assert!(matches!(err, ContextPoolError::CapacityExhausted { max: 2 }));
    }

    #[test]
    fn test_lru_eviction_when_pool_full() {
        let factory = Arc::new(MockFactory::new(1024));
        let config = ContextPoolConfig { max_contexts: 2, ..Default::default() };
        let pool = ContextPool::new(config, factory.clone());

        pool.acquire("m1").unwrap();
        pool.release("m1").unwrap();
        pool.acquire("m2").unwrap();
        pool.release("m2").unwrap();

        // Pool is full (2/2), but both are idle. Acquiring m3 should evict m1 (LRU).
        let entry = pool.acquire("m3").unwrap();
        assert_eq!(entry.id, "m3");
        assert_eq!(pool.len(), 2);
        assert_eq!(factory.releases(), 1); // m1 was evicted
    }

    #[test]
    fn test_explicit_eviction() {
        let factory = Arc::new(MockFactory::new(1024));
        let pool = default_pool(factory.clone());

        pool.acquire("model-evict").unwrap();
        pool.release("model-evict").unwrap();
        assert_eq!(pool.len(), 1);

        pool.evict("model-evict").unwrap();
        assert_eq!(pool.len(), 0);
        assert_eq!(factory.releases(), 1);
    }

    #[test]
    fn test_evict_nonexistent_returns_error() {
        let factory = Arc::new(MockFactory::new(1024));
        let pool = default_pool(factory);

        let err = pool.evict("no-such-context").unwrap_err();
        assert!(matches!(err, ContextPoolError::NotFound { .. }));
    }

    #[test]
    fn test_memory_pressure_prevents_creation() {
        let factory = Arc::new(MockFactory::new(512 * 1024 * 1024)); // 512 MiB each
        let config = ContextPoolConfig {
            max_contexts: 8,
            memory_limit: 1024 * 1024 * 1024, // 1 GiB limit
            ..Default::default()
        };
        let pool = ContextPool::new(config, factory.clone());

        pool.acquire("m1").unwrap();
        pool.release("m1").unwrap();
        pool.acquire("m2").unwrap();
        pool.release("m2").unwrap();

        // Total = 1 GiB (2 × 512 MiB), at limit. Next create should fail.
        let err = pool.acquire("m3").unwrap_err();
        assert!(matches!(err, ContextPoolError::MemoryPressure { .. }));
    }

    #[test]
    fn test_evict_expired_idle_contexts() {
        let factory = Arc::new(MockFactory::new(1024));
        let config = ContextPoolConfig {
            idle_timeout: Duration::from_millis(0), // expire immediately
            ..Default::default()
        };
        let pool = ContextPool::new(config, factory.clone());

        pool.acquire("m1").unwrap();
        pool.release("m1").unwrap();
        pool.acquire("m2").unwrap();
        pool.release("m2").unwrap();

        // Both are idle and the timeout is 0ms, so both should be expired.
        std::thread::sleep(Duration::from_millis(5));
        let evicted = pool.evict_expired().unwrap();
        assert_eq!(evicted, 2);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_total_memory_usage_tracking() {
        let factory = Arc::new(MockFactory::new(256 * 1024)); // 256 KiB each
        let pool = default_pool(factory);

        pool.acquire("a").unwrap();
        pool.acquire("b").unwrap();
        assert_eq!(pool.total_memory_usage(), 512 * 1024);
    }

    #[test]
    fn test_creation_failure_propagates() {
        let factory = Arc::new(MockFactory::new(1024));
        factory.set_fail_on_create("bad-model");
        let pool = default_pool(factory);

        let err = pool.acquire("bad-model").unwrap_err();
        assert!(matches!(err, ContextPoolError::CreationFailed { .. }));
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_release_nonexistent_returns_error() {
        let factory = Arc::new(MockFactory::new(1024));
        let pool = default_pool(factory);

        let err = pool.release("ghost").unwrap_err();
        assert!(matches!(err, ContextPoolError::NotFound { .. }));
    }
}
