//! Pooled OpenCL context manager for efficient resource sharing.
//!
//! Provides [`ContextPool`] to share OpenCL contexts and command queues
//! across multiple inference requests, reducing setup overhead on
//! devices like the Intel Arc A770.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the context pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of live contexts.
    pub max_contexts: usize,
    /// Maximum command queues per context.
    pub max_queues_per_context: usize,
    /// Seconds a context may sit idle before eviction.
    pub idle_timeout_secs: u64,
    /// Number of contexts to pre-allocate on warm-up.
    pub warm_pool_size: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_contexts: 4,
            max_queues_per_context: 8,
            idle_timeout_secs: 300,
            warm_pool_size: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

/// A handle to a pooled OpenCL context (CPU reference).
#[derive(Debug, Clone)]
pub struct ContextHandle {
    /// Unique identifier.
    pub id: u64,
    /// When the context was created.
    pub created_at: Instant,
    /// Last time the context was acquired.
    pub last_used: Instant,
    /// Number of outstanding borrows.
    pub ref_count: usize,
}

/// Priority level for a command queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueuePriority {
    /// Highest priority — latency-sensitive work.
    High,
    /// Default priority.
    Normal,
    /// Lower priority batch work.
    Low,
    /// Background / idle work.
    Background,
}

/// A handle to a pooled command queue (CPU reference).
#[derive(Debug, Clone)]
pub struct QueueHandle {
    /// Owning context id.
    pub context_id: u64,
    /// Unique queue identifier.
    pub queue_id: u64,
    /// Whether this queue is currently checked out.
    pub in_use: bool,
    /// Queue priority.
    pub priority: QueuePriority,
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Cumulative pool statistics.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total_contexts_created: u64,
    pub total_queues_created: u64,
    pub context_hits: u64,
    pub context_misses: u64,
    pub peak_active_contexts: usize,
    pub peak_active_queues: usize,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    /// All context slots are in use and none are reusable.
    PoolExhausted,
    /// Failed to create a new context.
    ContextCreationFailed,
    /// Failed to create a new command queue.
    QueueCreationFailed,
    /// The supplied handle id does not exist.
    InvalidHandle,
    /// An operation timed out.
    Timeout,
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PoolExhausted => write!(f, "context pool exhausted"),
            Self::ContextCreationFailed => {
                write!(f, "context creation failed")
            }
            Self::QueueCreationFailed => {
                write!(f, "queue creation failed")
            }
            Self::InvalidHandle => write!(f, "invalid handle"),
            Self::Timeout => write!(f, "operation timed out"),
        }
    }
}

impl std::error::Error for PoolError {}

// ---------------------------------------------------------------------------
// Pool
// ---------------------------------------------------------------------------

/// Pooled OpenCL context and queue manager (CPU reference implementation).
#[derive(Debug)]
pub struct ContextPool {
    pub contexts: Vec<ContextHandle>,
    pub queues: Vec<QueueHandle>,
    pub config: PoolConfig,
    pub stats: PoolStats,
    next_context_id: u64,
    next_queue_id: u64,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Create a new, empty context pool with the given configuration.
pub fn create_context_pool(config: PoolConfig) -> ContextPool {
    ContextPool {
        contexts: Vec::new(),
        queues: Vec::new(),
        config,
        stats: PoolStats::default(),
        next_context_id: 1,
        next_queue_id: 1,
    }
}

/// Acquire a context from the pool, reusing an idle one if possible.
///
/// Returns the context `id` on success.
pub fn cpu_acquire_context(
    pool: &mut ContextPool,
) -> Result<u64, PoolError> {
    // Try to reuse an idle (ref_count == 0) context.
    if let Some(idx) = pool
        .contexts
        .iter()
        .position(|c| c.ref_count == 0)
    {
        pool.contexts[idx].ref_count += 1;
        pool.contexts[idx].last_used = Instant::now();
        pool.stats.context_hits += 1;
        let id = pool.contexts[idx].id;
        update_peak_contexts(pool);
        return Ok(id);
    }

    // No idle context — create a new one if capacity allows.
    if pool.contexts.len() >= pool.config.max_contexts {
        return Err(PoolError::PoolExhausted);
    }

    let id = pool.next_context_id;
    pool.next_context_id += 1;
    let now = Instant::now();
    pool.contexts.push(ContextHandle {
        id,
        created_at: now,
        last_used: now,
        ref_count: 1,
    });
    pool.stats.total_contexts_created += 1;
    pool.stats.context_misses += 1;
    update_peak_contexts(pool);
    Ok(id)
}

/// Release a previously acquired context back to the pool.
pub fn cpu_release_context(
    pool: &mut ContextPool,
    context_id: u64,
) {
    if let Some(ctx) =
        pool.contexts.iter_mut().find(|c| c.id == context_id)
    {
        ctx.ref_count = ctx.ref_count.saturating_sub(1);
        ctx.last_used = Instant::now();
    }
}

/// Acquire a command queue bound to `context_id` with the requested
/// priority.
pub fn cpu_acquire_queue(
    pool: &mut ContextPool,
    context_id: u64,
    priority: QueuePriority,
) -> Result<u64, PoolError> {
    // Validate context exists.
    if !pool.contexts.iter().any(|c| c.id == context_id) {
        return Err(PoolError::InvalidHandle);
    }

    // Try to reuse an idle queue on the same context, preferring one
    // that already has the requested priority.
    if let Some(idx) = pool
        .queues
        .iter()
        .position(|q| {
            q.context_id == context_id
                && !q.in_use
                && q.priority == priority
        })
    {
        pool.queues[idx].in_use = true;
        pool.stats.context_hits += 1;
        let qid = pool.queues[idx].queue_id;
        update_peak_queues(pool);
        return Ok(qid);
    }

    // Reuse any idle queue on the same context (change priority).
    if let Some(idx) = pool
        .queues
        .iter()
        .position(|q| q.context_id == context_id && !q.in_use)
    {
        pool.queues[idx].in_use = true;
        pool.queues[idx].priority = priority;
        let qid = pool.queues[idx].queue_id;
        update_peak_queues(pool);
        return Ok(qid);
    }

    // Check per-context queue limit.
    let ctx_queue_count = pool
        .queues
        .iter()
        .filter(|q| q.context_id == context_id)
        .count();
    if ctx_queue_count >= pool.config.max_queues_per_context {
        return Err(PoolError::QueueCreationFailed);
    }

    let qid = pool.next_queue_id;
    pool.next_queue_id += 1;
    pool.queues.push(QueueHandle {
        context_id,
        queue_id: qid,
        in_use: true,
        priority,
    });
    pool.stats.total_queues_created += 1;
    update_peak_queues(pool);
    Ok(qid)
}

/// Release a command queue back to the pool.
pub fn cpu_release_queue(pool: &mut ContextPool, queue_id: u64) {
    if let Some(q) =
        pool.queues.iter_mut().find(|q| q.queue_id == queue_id)
    {
        q.in_use = false;
    }
}

/// Evict contexts that have been idle longer than `max_idle_secs` and
/// have no outstanding references. Returns the number evicted.
pub fn cpu_evict_idle(
    pool: &mut ContextPool,
    max_idle_secs: u64,
) -> usize {
    let now = Instant::now();
    let mut evicted = 0usize;

    pool.contexts.retain(|ctx| {
        if ctx.ref_count == 0
            && now
                .duration_since(ctx.last_used)
                .as_secs()
                >= max_idle_secs
        {
            evicted += 1;
            false
        } else {
            true
        }
    });

    // Also remove orphaned queues.
    let live_ids: Vec<u64> =
        pool.contexts.iter().map(|c| c.id).collect();
    pool.queues.retain(|q| live_ids.contains(&q.context_id));

    evicted
}

/// Return `(context_utilization, queue_utilization)` in `[0.0, 1.0]`.
pub fn cpu_pool_utilization(pool: &ContextPool) -> (f32, f32) {
    let ctx_util = if pool.config.max_contexts == 0 {
        0.0
    } else {
        let active =
            pool.contexts.iter().filter(|c| c.ref_count > 0).count();
        active as f32 / pool.config.max_contexts as f32
    };

    let total_queue_cap = pool.config.max_contexts
        * pool.config.max_queues_per_context;
    let queue_util = if total_queue_cap == 0 {
        0.0
    } else {
        let active =
            pool.queues.iter().filter(|q| q.in_use).count();
        active as f32 / total_queue_cap as f32
    };

    (ctx_util, queue_util)
}

/// Pre-allocate `count` idle contexts (capped at `max_contexts`).
pub fn cpu_warm_pool(pool: &mut ContextPool, count: usize) {
    for _ in 0..count {
        if pool.contexts.len() >= pool.config.max_contexts {
            break;
        }
        let id = pool.next_context_id;
        pool.next_context_id += 1;
        let now = Instant::now();
        pool.contexts.push(ContextHandle {
            id,
            created_at: now,
            last_used: now,
            ref_count: 0,
        });
        pool.stats.total_contexts_created += 1;
    }
}

/// Return a snapshot of the pool statistics.
pub fn cpu_get_stats(pool: &ContextPool) -> PoolStats {
    pool.stats.clone()
}

/// Resize the pool's maximum context capacity.
///
/// If `new_max` is smaller than the current number of contexts, only
/// idle (ref_count == 0) contexts are removed to try to reach the new
/// limit. In-use contexts are never dropped.
pub fn cpu_resize_pool(pool: &mut ContextPool, new_max: usize) {
    pool.config.max_contexts = new_max;

    // Shrink if necessary — evict idle contexts from the back.
    while pool.contexts.len() > new_max {
        if let Some(pos) = pool
            .contexts
            .iter()
            .rposition(|c| c.ref_count == 0)
        {
            let removed_id = pool.contexts.remove(pos).id;
            pool.queues.retain(|q| q.context_id != removed_id);
        } else {
            break; // all remaining contexts are in use
        }
    }
}

/// Human-readable pool status string.
pub fn format_pool_status(pool: &ContextPool) -> String {
    let active_ctx =
        pool.contexts.iter().filter(|c| c.ref_count > 0).count();
    let idle_ctx =
        pool.contexts.iter().filter(|c| c.ref_count == 0).count();
    let active_q =
        pool.queues.iter().filter(|q| q.in_use).count();
    let idle_q =
        pool.queues.iter().filter(|q| !q.in_use).count();
    let (ctx_util, q_util) = cpu_pool_utilization(pool);

    format!(
        "ContextPool {{ contexts: {}/{} (active={}, idle={}), \
         queues: {} (active={}, idle={}), \
         utilization: ctx={:.1}% q={:.1}%, \
         hits={}, misses={} }}",
        pool.contexts.len(),
        pool.config.max_contexts,
        active_ctx,
        idle_ctx,
        pool.queues.len(),
        active_q,
        idle_q,
        ctx_util * 100.0,
        q_util * 100.0,
        pool.stats.context_hits,
        pool.stats.context_misses,
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn update_peak_contexts(pool: &mut ContextPool) {
    let active =
        pool.contexts.iter().filter(|c| c.ref_count > 0).count();
    if active > pool.stats.peak_active_contexts {
        pool.stats.peak_active_contexts = active;
    }
}

fn update_peak_queues(pool: &mut ContextPool) {
    let active =
        pool.queues.iter().filter(|q| q.in_use).count();
    if active > pool.stats.peak_active_queues {
        pool.stats.peak_active_queues = active;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pool() -> ContextPool {
        create_context_pool(PoolConfig::default())
    }

    fn small_pool() -> ContextPool {
        create_context_pool(PoolConfig {
            max_contexts: 2,
            max_queues_per_context: 2,
            idle_timeout_secs: 1,
            warm_pool_size: 0,
        })
    }

    // -- create pool ------------------------------------------------------

    #[test]
    fn test_create_pool_empty() {
        let pool = default_pool();
        assert!(pool.contexts.is_empty());
        assert!(pool.queues.is_empty());
    }

    #[test]
    fn test_create_pool_config_preserved() {
        let cfg = PoolConfig {
            max_contexts: 16,
            max_queues_per_context: 4,
            idle_timeout_secs: 600,
            warm_pool_size: 2,
        };
        let pool = create_context_pool(cfg.clone());
        assert_eq!(pool.config.max_contexts, 16);
        assert_eq!(pool.config.max_queues_per_context, 4);
        assert_eq!(pool.config.idle_timeout_secs, 600);
        assert_eq!(pool.config.warm_pool_size, 2);
    }

    // -- acquire / release context ----------------------------------------

    #[test]
    fn test_acquire_context_returns_id() {
        let mut pool = default_pool();
        let id = cpu_acquire_context(&mut pool).unwrap();
        assert!(id > 0);
    }

    #[test]
    fn test_release_context_decrements_ref() {
        let mut pool = default_pool();
        let id = cpu_acquire_context(&mut pool).unwrap();
        assert_eq!(pool.contexts[0].ref_count, 1);
        cpu_release_context(&mut pool, id);
        assert_eq!(pool.contexts[0].ref_count, 0);
    }

    #[test]
    fn test_context_round_trip() {
        let mut pool = default_pool();
        let id = cpu_acquire_context(&mut pool).unwrap();
        cpu_release_context(&mut pool, id);
        // Context stays in pool after release.
        assert_eq!(pool.contexts.len(), 1);
    }

    // -- acquire / release queue ------------------------------------------

    #[test]
    fn test_acquire_queue_returns_id() {
        let mut pool = default_pool();
        let ctx = cpu_acquire_context(&mut pool).unwrap();
        let qid =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::Normal)
                .unwrap();
        assert!(qid > 0);
    }

    #[test]
    fn test_release_queue_marks_not_in_use() {
        let mut pool = default_pool();
        let ctx = cpu_acquire_context(&mut pool).unwrap();
        let qid =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::Normal)
                .unwrap();
        assert!(pool.queues[0].in_use);
        cpu_release_queue(&mut pool, qid);
        assert!(!pool.queues[0].in_use);
    }

    #[test]
    fn test_queue_round_trip() {
        let mut pool = default_pool();
        let ctx = cpu_acquire_context(&mut pool).unwrap();
        let qid =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::High)
                .unwrap();
        cpu_release_queue(&mut pool, qid);
        assert_eq!(pool.queues.len(), 1);
        assert!(!pool.queues[0].in_use);
    }

    #[test]
    fn test_acquire_queue_invalid_context() {
        let mut pool = default_pool();
        let res =
            cpu_acquire_queue(&mut pool, 999, QueuePriority::Normal);
        assert_eq!(res, Err(PoolError::InvalidHandle));
    }

    // -- pool hit / miss --------------------------------------------------

    #[test]
    fn test_pool_hit_reuses_context() {
        let mut pool = default_pool();
        let id1 = cpu_acquire_context(&mut pool).unwrap();
        cpu_release_context(&mut pool, id1);
        let id2 = cpu_acquire_context(&mut pool).unwrap();
        assert_eq!(id1, id2, "should reuse the same context");
    }

    #[test]
    fn test_pool_miss_creates_new() {
        let mut pool = default_pool();
        let id1 = cpu_acquire_context(&mut pool).unwrap();
        // id1 still in use — next acquire must create a new one.
        let id2 = cpu_acquire_context(&mut pool).unwrap();
        assert_ne!(id1, id2);
    }

    // -- exhaustion -------------------------------------------------------

    #[test]
    fn test_pool_exhausted() {
        let mut pool = small_pool();
        let _a = cpu_acquire_context(&mut pool).unwrap();
        let _b = cpu_acquire_context(&mut pool).unwrap();
        assert_eq!(
            cpu_acquire_context(&mut pool),
            Err(PoolError::PoolExhausted)
        );
    }

    #[test]
    fn test_queue_creation_failed_at_limit() {
        let mut pool = small_pool();
        let ctx = cpu_acquire_context(&mut pool).unwrap();
        let _q1 =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::Normal)
                .unwrap();
        let _q2 =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::Normal)
                .unwrap();
        assert_eq!(
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::Normal),
            Err(PoolError::QueueCreationFailed)
        );
    }

    // -- eviction ---------------------------------------------------------

    #[test]
    fn test_evict_idle_removes_old() {
        let mut pool = default_pool();
        let id = cpu_acquire_context(&mut pool).unwrap();
        cpu_release_context(&mut pool, id);
        // Evict with 0-second threshold — everything idle is old.
        let evicted = cpu_evict_idle(&mut pool, 0);
        assert_eq!(evicted, 1);
        assert!(pool.contexts.is_empty());
    }

    #[test]
    fn test_evict_idle_keeps_in_use() {
        let mut pool = default_pool();
        let _id = cpu_acquire_context(&mut pool).unwrap();
        let evicted = cpu_evict_idle(&mut pool, 0);
        assert_eq!(evicted, 0);
        assert_eq!(pool.contexts.len(), 1);
    }

    #[test]
    fn test_evict_removes_orphaned_queues() {
        let mut pool = default_pool();
        let ctx = cpu_acquire_context(&mut pool).unwrap();
        let _q =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::Normal)
                .unwrap();
        cpu_release_queue(&mut pool, _q);
        cpu_release_context(&mut pool, ctx);
        cpu_evict_idle(&mut pool, 0);
        assert!(pool.queues.is_empty());
    }

    // -- utilization ------------------------------------------------------

    #[test]
    fn test_utilization_empty() {
        let pool = default_pool();
        let (cu, qu) = cpu_pool_utilization(&pool);
        assert!((cu - 0.0).abs() < f32::EPSILON);
        assert!((qu - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_utilization_full_contexts() {
        let mut pool = small_pool(); // max 2
        let _a = cpu_acquire_context(&mut pool).unwrap();
        let _b = cpu_acquire_context(&mut pool).unwrap();
        let (cu, _) = cpu_pool_utilization(&pool);
        assert!((cu - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_utilization_half() {
        let mut pool = small_pool(); // max 2
        let _a = cpu_acquire_context(&mut pool).unwrap();
        let (cu, _) = cpu_pool_utilization(&pool);
        assert!((cu - 0.5).abs() < f32::EPSILON);
    }

    // -- warm pool --------------------------------------------------------

    #[test]
    fn test_warm_pool_preallocates() {
        let mut pool = default_pool(); // max 4
        cpu_warm_pool(&mut pool, 3);
        assert_eq!(pool.contexts.len(), 3);
        // All idle.
        assert!(pool.contexts.iter().all(|c| c.ref_count == 0));
    }

    #[test]
    fn test_warm_pool_capped_at_max() {
        let mut pool = small_pool(); // max 2
        cpu_warm_pool(&mut pool, 10);
        assert_eq!(pool.contexts.len(), 2);
    }

    // -- resize -----------------------------------------------------------

    #[test]
    fn test_resize_grow() {
        let mut pool = small_pool();
        cpu_resize_pool(&mut pool, 8);
        assert_eq!(pool.config.max_contexts, 8);
    }

    #[test]
    fn test_resize_shrink_evicts_idle() {
        let mut pool = default_pool();
        cpu_warm_pool(&mut pool, 4);
        cpu_resize_pool(&mut pool, 2);
        assert!(pool.contexts.len() <= 2);
    }

    #[test]
    fn test_resize_shrink_keeps_in_use() {
        let mut pool = default_pool();
        let _a = cpu_acquire_context(&mut pool).unwrap();
        let _b = cpu_acquire_context(&mut pool).unwrap();
        let _c = cpu_acquire_context(&mut pool).unwrap();
        cpu_resize_pool(&mut pool, 1);
        // All 3 are in use — cannot evict.
        assert_eq!(pool.contexts.len(), 3);
    }

    // -- priority queues --------------------------------------------------

    #[test]
    fn test_priority_queue_acquired() {
        let mut pool = default_pool();
        let ctx = cpu_acquire_context(&mut pool).unwrap();
        let qh =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::High)
                .unwrap();
        let ql =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::Low)
                .unwrap();
        assert_ne!(qh, ql);
        let high_q =
            pool.queues.iter().find(|q| q.queue_id == qh).unwrap();
        assert_eq!(high_q.priority, QueuePriority::High);
    }

    #[test]
    fn test_queue_reuse_same_priority() {
        let mut pool = default_pool();
        let ctx = cpu_acquire_context(&mut pool).unwrap();
        let q1 =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::Normal)
                .unwrap();
        cpu_release_queue(&mut pool, q1);
        let q2 =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::Normal)
                .unwrap();
        assert_eq!(q1, q2, "should reuse released queue");
    }

    // -- stats ------------------------------------------------------------

    #[test]
    fn test_stats_hit_miss_counts() {
        let mut pool = default_pool();
        let id = cpu_acquire_context(&mut pool).unwrap();
        cpu_release_context(&mut pool, id);
        let _ = cpu_acquire_context(&mut pool).unwrap();
        let stats = cpu_get_stats(&pool);
        assert_eq!(stats.context_misses, 1);
        assert_eq!(stats.context_hits, 1);
    }

    #[test]
    fn test_stats_total_created() {
        let mut pool = default_pool();
        let _a = cpu_acquire_context(&mut pool).unwrap();
        let _b = cpu_acquire_context(&mut pool).unwrap();
        assert_eq!(cpu_get_stats(&pool).total_contexts_created, 2);
    }

    #[test]
    fn test_stats_total_queues_created() {
        let mut pool = default_pool();
        let ctx = cpu_acquire_context(&mut pool).unwrap();
        let _ =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::Normal)
                .unwrap();
        let _ =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::High)
                .unwrap();
        assert_eq!(cpu_get_stats(&pool).total_queues_created, 2);
    }

    // -- multiple acquire -------------------------------------------------

    #[test]
    fn test_multiple_acquire_different_ids() {
        let mut pool = default_pool();
        let a = cpu_acquire_context(&mut pool).unwrap();
        let b = cpu_acquire_context(&mut pool).unwrap();
        let c = cpu_acquire_context(&mut pool).unwrap();
        let ids = [a, b, c];
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j]);
            }
        }
    }

    // -- ref counting -----------------------------------------------------

    #[test]
    fn test_ref_count_prevents_eviction() {
        let mut pool = default_pool();
        let id = cpu_acquire_context(&mut pool).unwrap();
        // Context still in use — evict should skip it.
        let evicted = cpu_evict_idle(&mut pool, 0);
        assert_eq!(evicted, 0);
        assert_eq!(pool.contexts.len(), 1);
        cpu_release_context(&mut pool, id);
    }

    // -- peak tracking ----------------------------------------------------

    #[test]
    fn test_peak_active_contexts() {
        let mut pool = default_pool();
        let a = cpu_acquire_context(&mut pool).unwrap();
        let b = cpu_acquire_context(&mut pool).unwrap();
        cpu_release_context(&mut pool, a);
        // Peak should still be 2.
        assert_eq!(cpu_get_stats(&pool).peak_active_contexts, 2);
        cpu_release_context(&mut pool, b);
    }

    #[test]
    fn test_peak_active_queues() {
        let mut pool = default_pool();
        let ctx = cpu_acquire_context(&mut pool).unwrap();
        let q1 =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::Normal)
                .unwrap();
        let _q2 =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::High)
                .unwrap();
        cpu_release_queue(&mut pool, q1);
        assert_eq!(cpu_get_stats(&pool).peak_active_queues, 2);
    }

    // -- format status ----------------------------------------------------

    #[test]
    fn test_format_status_contains_info() {
        let mut pool = default_pool();
        let _ = cpu_acquire_context(&mut pool).unwrap();
        let status = format_pool_status(&pool);
        assert!(status.contains("ContextPool"));
        assert!(status.contains("active=1"));
        assert!(status.contains("utilization"));
    }

    #[test]
    fn test_format_status_empty_pool() {
        let pool = default_pool();
        let status = format_pool_status(&pool);
        assert!(status.contains("active=0"));
    }

    // -- property: released contexts reusable -----------------------------

    #[test]
    fn test_released_contexts_reusable() {
        let mut pool = small_pool();
        let a = cpu_acquire_context(&mut pool).unwrap();
        let b = cpu_acquire_context(&mut pool).unwrap();
        cpu_release_context(&mut pool, a);
        cpu_release_context(&mut pool, b);
        // Both should be reusable.
        let c = cpu_acquire_context(&mut pool).unwrap();
        let d = cpu_acquire_context(&mut pool).unwrap();
        assert!(c == a || c == b);
        assert!(d == a || d == b);
    }

    // -- property: utilization in [0, 1] ----------------------------------

    #[test]
    fn test_utilization_bounds() {
        let mut pool = default_pool();
        for _ in 0..pool.config.max_contexts {
            let _ = cpu_acquire_context(&mut pool).unwrap();
        }
        let (cu, qu) = cpu_pool_utilization(&pool);
        assert!((0.0..=1.0).contains(&cu));
        assert!((0.0..=1.0).contains(&qu));
    }

    // -- edge: max_contexts=1, max_queues=1 -------------------------------

    #[test]
    fn test_edge_single_context() {
        let mut pool = create_context_pool(PoolConfig {
            max_contexts: 1,
            max_queues_per_context: 1,
            idle_timeout_secs: 60,
            warm_pool_size: 0,
        });
        let ctx = cpu_acquire_context(&mut pool).unwrap();
        assert_eq!(
            cpu_acquire_context(&mut pool),
            Err(PoolError::PoolExhausted)
        );
        cpu_release_context(&mut pool, ctx);
        let ctx2 = cpu_acquire_context(&mut pool).unwrap();
        assert_eq!(ctx, ctx2);
    }

    #[test]
    fn test_edge_single_queue() {
        let mut pool = create_context_pool(PoolConfig {
            max_contexts: 1,
            max_queues_per_context: 1,
            idle_timeout_secs: 60,
            warm_pool_size: 0,
        });
        let ctx = cpu_acquire_context(&mut pool).unwrap();
        let q =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::Normal)
                .unwrap();
        assert_eq!(
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::High),
            Err(PoolError::QueueCreationFailed)
        );
        cpu_release_queue(&mut pool, q);
        let q2 =
            cpu_acquire_queue(&mut pool, ctx, QueuePriority::High)
                .unwrap();
        assert_eq!(q, q2);
    }

    // -- error display ----------------------------------------------------

    #[test]
    fn test_pool_error_display() {
        assert_eq!(
            PoolError::PoolExhausted.to_string(),
            "context pool exhausted"
        );
        assert_eq!(
            PoolError::InvalidHandle.to_string(),
            "invalid handle"
        );
        assert_eq!(
            PoolError::Timeout.to_string(),
            "operation timed out"
        );
    }

    // -- default config ---------------------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = PoolConfig::default();
        assert_eq!(cfg.max_contexts, 4);
        assert_eq!(cfg.max_queues_per_context, 8);
        assert_eq!(cfg.idle_timeout_secs, 300);
        assert_eq!(cfg.warm_pool_size, 1);
    }

    // -- queue priority ordering ------------------------------------------

    #[test]
    fn test_queue_priority_ord() {
        assert!(QueuePriority::High < QueuePriority::Normal);
        assert!(QueuePriority::Normal < QueuePriority::Low);
        assert!(QueuePriority::Low < QueuePriority::Background);
    }

    // -- warm then acquire ------------------------------------------------

    #[test]
    fn test_warm_then_acquire_reuses() {
        let mut pool = default_pool();
        cpu_warm_pool(&mut pool, 2);
        let id = cpu_acquire_context(&mut pool).unwrap();
        // Should reuse a warmed context, not create new.
        assert_eq!(pool.stats.context_hits, 1);
        assert_eq!(pool.stats.context_misses, 0);
        cpu_release_context(&mut pool, id);
    }

    // -- utilization with zero config -------------------------------------

    #[test]
    fn test_utilization_zero_max() {
        let pool = create_context_pool(PoolConfig {
            max_contexts: 0,
            max_queues_per_context: 0,
            idle_timeout_secs: 60,
            warm_pool_size: 0,
        });
        let (cu, qu) = cpu_pool_utilization(&pool);
        assert!((cu - 0.0).abs() < f32::EPSILON);
        assert!((qu - 0.0).abs() < f32::EPSILON);
    }

    // -- release non-existent is no-op ------------------------------------

    #[test]
    fn test_release_nonexistent_context_noop() {
        let mut pool = default_pool();
        cpu_release_context(&mut pool, 42); // should not panic
        assert!(pool.contexts.is_empty());
    }

    #[test]
    fn test_release_nonexistent_queue_noop() {
        let mut pool = default_pool();
        cpu_release_queue(&mut pool, 42); // should not panic
    }
}
