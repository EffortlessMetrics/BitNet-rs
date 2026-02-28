//! CPU thread pool for inference workloads with work-stealing scheduling.
//!
//! Wraps [`rayon::ThreadPool`] with inference-specific configuration,
//! metrics collection, and optional NUMA-aware thread pinning.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Configuration for [`InferenceThreadPool`].
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads. Defaults to the number of available CPUs.
    pub num_threads: usize,
    /// Whether to attempt NUMA-aware thread affinity (best-effort).
    pub affinity: bool,
    /// Thread scheduling priority hint (0 = normal, higher = elevated).
    /// Only advisory; actual effect is platform-dependent.
    pub priority: u8,
    /// Prefix for worker thread names (e.g. `"bitnet-inf"`).
    pub name_prefix: String,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get().max(1),
            affinity: false,
            priority: 0,
            name_prefix: "bitnet-inf".to_string(),
        }
    }
}

/// Live metrics snapshot from the thread pool.
#[derive(Debug, Clone, Copy)]
pub struct ThreadPoolMetrics {
    /// Number of threads currently executing work.
    pub active_threads: usize,
    /// Approximate number of pending tasks in the work-stealing queues.
    pub queue_depth: usize,
    /// Cumulative tasks completed since pool creation.
    pub tasks_completed: u64,
    /// Pool utilization ratio `[0.0, 1.0]` averaged since last reset.
    pub utilization: f64,
}

/// Shared counters backing [`ThreadPoolMetrics`].
#[derive(Debug)]
struct MetricsInner {
    active: AtomicUsize,
    queued: AtomicUsize,
    completed: AtomicU64,
    busy_ns: AtomicU64,
    wall_start: Instant,
    num_threads: usize,
}

impl MetricsInner {
    fn new(num_threads: usize) -> Self {
        Self {
            active: AtomicUsize::new(0),
            queued: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            busy_ns: AtomicU64::new(0),
            wall_start: Instant::now(),
            num_threads,
        }
    }

    fn snapshot(&self) -> ThreadPoolMetrics {
        let wall_ns = self.wall_start.elapsed().as_nanos().max(1) as f64;
        let busy = self.busy_ns.load(Ordering::Relaxed) as f64;
        let capacity = wall_ns * self.num_threads as f64;
        ThreadPoolMetrics {
            active_threads: self.active.load(Ordering::Relaxed),
            queue_depth: self.queued.load(Ordering::Relaxed),
            tasks_completed: self.completed.load(Ordering::Relaxed),
            utilization: (busy / capacity).min(1.0),
        }
    }
}

/// RAII guard that tracks a task's active duration in [`MetricsInner`].
struct TaskGuard {
    metrics: Arc<MetricsInner>,
    start: Instant,
}

impl TaskGuard {
    fn new(metrics: &Arc<MetricsInner>) -> Self {
        metrics.active.fetch_add(1, Ordering::Relaxed);
        Self { metrics: Arc::clone(metrics), start: Instant::now() }
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_nanos() as u64;
        self.metrics.busy_ns.fetch_add(elapsed, Ordering::Relaxed);
        self.metrics.active.fetch_sub(1, Ordering::Relaxed);
        self.metrics.completed.fetch_add(1, Ordering::Relaxed);
    }
}

/// A CPU thread pool optimized for neural-network inference workloads.
///
/// Built on top of [`rayon::ThreadPool`] (work-stealing), this wrapper adds
/// configurable thread naming, optional NUMA affinity, and live metrics.
pub struct InferenceThreadPool {
    pool: rayon::ThreadPool,
    metrics: Arc<MetricsInner>,
    config: ThreadPoolConfig,
    numa_available: bool,
}

impl InferenceThreadPool {
    /// Create a new thread pool from the given configuration.
    pub fn new(config: ThreadPoolConfig) -> Result<Self, rayon::ThreadPoolBuildError> {
        let numa_available = config.affinity && Self::detect_numa();
        let prefix = config.name_prefix.clone();
        let affinity = config.affinity;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .thread_name(move |idx| format!("{prefix}-{idx}"))
            .start_handler(move |idx| {
                if affinity {
                    Self::pin_thread_best_effort(idx);
                }
            })
            .build()?;

        let metrics = Arc::new(MetricsInner::new(config.num_threads));
        Ok(Self { pool, metrics, config, numa_available })
    }

    /// Create a pool with default configuration.
    pub fn with_defaults() -> Result<Self, rayon::ThreadPoolBuildError> {
        Self::new(ThreadPoolConfig::default())
    }

    /// The configuration this pool was created with.
    pub fn config(&self) -> &ThreadPoolConfig {
        &self.config
    }

    /// Whether NUMA topology was detected at construction time.
    pub fn numa_available(&self) -> bool {
        self.numa_available
    }

    /// Return a snapshot of current pool metrics.
    pub fn metrics(&self) -> ThreadPoolMetrics {
        self.metrics.snapshot()
    }

    /// Number of worker threads in the pool.
    pub fn num_threads(&self) -> usize {
        self.config.num_threads
    }

    // ----- parallel primitives -----

    /// Apply `f` to every index in `range`, splitting work into chunks of
    /// `chunk_size` distributed across the thread pool.
    ///
    /// This is the primary data-parallel primitive for inference operators
    /// (e.g. row-partitioned matmul, per-head attention).
    pub fn parallel_for<F>(&self, range: Range<usize>, chunk_size: usize, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        let chunk = chunk_size.max(1);
        let metrics = &self.metrics;
        metrics.queued.fetch_add(1, Ordering::Relaxed);

        self.pool.install(|| {
            metrics.queued.fetch_sub(1, Ordering::Relaxed);
            let _guard = TaskGuard::new(metrics);

            rayon::scope(|s| {
                let mut start = range.start;
                while start < range.end {
                    let end = (start + chunk).min(range.end);
                    let f_ref = &f;
                    let lo = start;
                    s.spawn(move |_| {
                        for i in lo..end {
                            f_ref(i);
                        }
                    });
                    start = end;
                }
            });
        });
    }

    /// Map-reduce over `range`: apply `f` to each index, then fold
    /// results together with `combine`, starting from `identity`.
    pub fn parallel_reduce<T, F, C>(&self, range: Range<usize>, identity: T, f: F, combine: C) -> T
    where
        T: Send + Sync + Clone,
        F: Fn(usize) -> T + Send + Sync,
        C: Fn(T, T) -> T + Send + Sync,
    {
        let metrics = &self.metrics;
        metrics.queued.fetch_add(1, Ordering::Relaxed);

        self.pool.install(|| {
            metrics.queued.fetch_sub(1, Ordering::Relaxed);
            let _guard = TaskGuard::new(metrics);

            use rayon::prelude::*;
            (range.start..range.end).into_par_iter().map(&f).reduce(|| identity.clone(), &combine)
        })
    }

    /// Execute an arbitrary closure inside the pool's work-stealing
    /// [`rayon::Scope`], allowing fine-grained task spawning.
    pub fn scoped_execute<F>(&self, f: F)
    where
        F: FnOnce(&rayon::Scope<'_>) + Send,
    {
        let metrics = &self.metrics;
        metrics.queued.fetch_add(1, Ordering::Relaxed);

        self.pool.install(|| {
            metrics.queued.fetch_sub(1, Ordering::Relaxed);
            let _guard = TaskGuard::new(metrics);
            rayon::scope(f);
        });
    }

    // ----- NUMA helpers (best-effort) -----

    /// Probe whether the OS exposes a multi-node NUMA topology.
    fn detect_numa() -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/sys/devices/system/node/node1").exists()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Best-effort thread-to-core pinning via `sched_setaffinity` on Linux.
    fn pin_thread_best_effort(thread_index: usize) {
        #[cfg(target_os = "linux")]
        {
            let cpus = num_cpus::get().max(1);
            let core = thread_index % cpus;

            // SAFETY: we pass a correctly-sized cpu_set_t to sched_setaffinity
            // for the calling thread (tid 0).
            unsafe {
                let mut set: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_SET(core, &mut set);
                libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = thread_index;
        }
    }
}

impl std::fmt::Debug for InferenceThreadPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceThreadPool")
            .field("num_threads", &self.config.num_threads)
            .field("name_prefix", &self.config.name_prefix)
            .field("numa_available", &self.numa_available)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn test_pool(threads: usize) -> InferenceThreadPool {
        InferenceThreadPool::new(ThreadPoolConfig {
            num_threads: threads,
            name_prefix: "test".to_string(),
            ..Default::default()
        })
        .expect("failed to build test pool")
    }

    // ---- construction & config ----

    #[test]
    fn test_default_config_uses_available_cpus() {
        let cfg = ThreadPoolConfig::default();
        assert!(cfg.num_threads >= 1);
        assert_eq!(cfg.name_prefix, "bitnet-inf");
        assert!(!cfg.affinity);
        assert_eq!(cfg.priority, 0);
    }

    #[test]
    fn test_pool_creation_with_defaults() {
        let pool = InferenceThreadPool::with_defaults().unwrap();
        assert!(pool.num_threads() >= 1);
    }

    #[test]
    fn test_pool_respects_thread_count() {
        let pool = test_pool(4);
        assert_eq!(pool.num_threads(), 4);
        assert_eq!(pool.config().num_threads, 4);
    }

    #[test]
    fn test_pool_custom_name_prefix() {
        let pool = InferenceThreadPool::new(ThreadPoolConfig {
            num_threads: 2,
            name_prefix: "custom-prefix".to_string(),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(pool.config().name_prefix, "custom-prefix");
    }

    #[test]
    fn test_pool_debug_impl() {
        let pool = test_pool(2);
        let dbg = format!("{:?}", pool);
        assert!(dbg.contains("InferenceThreadPool"));
        assert!(dbg.contains("num_threads"));
    }

    // ---- parallel_for ----

    #[test]
    fn test_parallel_for_basic() {
        let pool = test_pool(4);
        let results = Arc::new(Mutex::new(vec![0usize; 100]));

        let r = Arc::clone(&results);
        pool.parallel_for(0..100, 10, move |i| {
            r.lock().unwrap()[i] = i * 2;
        });

        let data = results.lock().unwrap();
        for i in 0..100 {
            assert_eq!(data[i], i * 2, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_parallel_for_empty_range() {
        let pool = test_pool(2);
        let counter = Arc::new(AtomicUsize::new(0));

        let c = Arc::clone(&counter);
        pool.parallel_for(0..0, 1, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_parallel_for_single_element() {
        let pool = test_pool(2);
        let counter = Arc::new(AtomicUsize::new(0));

        let c = Arc::clone(&counter);
        pool.parallel_for(0..1, 1, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_parallel_for_chunk_larger_than_range() {
        let pool = test_pool(2);
        let counter = Arc::new(AtomicUsize::new(0));

        let c = Arc::clone(&counter);
        pool.parallel_for(0..5, 100, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_parallel_for_chunk_size_one() {
        let pool = test_pool(4);
        let sum = Arc::new(AtomicUsize::new(0));

        let s = Arc::clone(&sum);
        pool.parallel_for(0..8, 1, move |i| {
            s.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 28); // 0+1+..+7
    }

    // ---- parallel_reduce ----

    #[test]
    fn test_parallel_reduce_sum() {
        let pool = test_pool(4);
        let total = pool.parallel_reduce(0..100, 0u64, |i| i as u64, |a, b| a + b);
        assert_eq!(total, 4950); // 0+1+..+99
    }

    #[test]
    fn test_parallel_reduce_max() {
        let pool = test_pool(2);
        let max = pool.parallel_reduce(1..50, 0usize, |i| i * i, |a, b| a.max(b));
        assert_eq!(max, 49 * 49);
    }

    #[test]
    fn test_parallel_reduce_empty_range() {
        let pool = test_pool(2);
        let result = pool.parallel_reduce(0..0, 42, |i| i, |a, b| a + b);
        assert_eq!(result, 42); // identity returned
    }

    #[test]
    fn test_parallel_reduce_single_element() {
        let pool = test_pool(2);
        let result = pool.parallel_reduce(5..6, 0, |i| i, |a, b| a + b);
        assert_eq!(result, 5);
    }

    // ---- scoped_execute ----

    #[test]
    fn test_scoped_execute_basic() {
        let pool = test_pool(4);
        let val = Arc::new(AtomicUsize::new(0));

        let v = Arc::clone(&val);
        pool.scoped_execute(move |s| {
            for _ in 0..10 {
                let v2 = Arc::clone(&v);
                s.spawn(move |_| {
                    v2.fetch_add(1, Ordering::Relaxed);
                });
            }
        });

        assert_eq!(val.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_scoped_execute_nested() {
        let pool = test_pool(4);
        let counter = Arc::new(AtomicUsize::new(0));

        let c = Arc::clone(&counter);
        pool.scoped_execute(move |s| {
            let c2 = Arc::clone(&c);
            s.spawn(move |s2| {
                c2.fetch_add(1, Ordering::Relaxed);
                let c3 = Arc::clone(&c2);
                s2.spawn(move |_| {
                    c3.fetch_add(1, Ordering::Relaxed);
                });
            });
        });

        assert_eq!(counter.load(Ordering::Relaxed), 2);
    }

    // ---- deterministic execution ----

    #[test]
    fn test_deterministic_with_single_thread() {
        let pool = test_pool(1);

        // Run twice and verify identical ordering (deterministic).
        let collect = || {
            let results = Arc::new(Mutex::new(Vec::new()));
            let r = Arc::clone(&results);
            pool.parallel_for(0..20, 5, move |i| {
                r.lock().unwrap().push(i);
            });
            Arc::try_unwrap(results).unwrap().into_inner().unwrap()
        };

        let run1 = collect();
        let run2 = collect();
        assert_eq!(run1, run2, "single-thread execution must be deterministic");
        // All indices must be present.
        let mut sorted = run1.clone();
        sorted.sort();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn test_reduce_deterministic_single_thread() {
        let pool = test_pool(1);
        let a = pool.parallel_reduce(0..50, 0u64, |i| i as u64, |a, b| a + b);
        let b = pool.parallel_reduce(0..50, 0u64, |i| i as u64, |a, b| a + b);
        assert_eq!(a, b);
        assert_eq!(a, 1225);
    }

    // ---- metrics ----

    #[test]
    fn test_metrics_initial_state() {
        let pool = test_pool(2);
        let m = pool.metrics();
        assert_eq!(m.active_threads, 0);
        assert_eq!(m.tasks_completed, 0);
    }

    #[test]
    fn test_metrics_after_work() {
        let pool = test_pool(2);

        pool.parallel_for(0..100, 25, |_| {});
        pool.parallel_for(0..100, 25, |_| {});

        let m = pool.metrics();
        assert_eq!(m.tasks_completed, 2);
        assert!(m.utilization >= 0.0 && m.utilization <= 1.0);
    }

    #[test]
    fn test_metrics_utilization_bounded() {
        let pool = test_pool(4);
        // Do some actual work to drive utilization
        for _ in 0..10 {
            pool.parallel_for(0..1000, 50, |i| {
                std::hint::black_box(i * i);
            });
        }
        let m = pool.metrics();
        assert!(m.utilization >= 0.0);
        assert!(m.utilization <= 1.0);
    }

    // ---- concurrent safety ----

    #[test]
    fn test_concurrent_parallel_for_from_multiple_threads() {
        let pool = Arc::new(test_pool(4));
        let total = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|s| {
            for _ in 0..4 {
                let pool = Arc::clone(&pool);
                let total = Arc::clone(&total);
                s.spawn(move || {
                    pool.parallel_for(0..100, 10, |_| {
                        total.fetch_add(1, Ordering::Relaxed);
                    });
                });
            }
        });

        assert_eq!(total.load(Ordering::Relaxed), 400);
    }

    // ---- NUMA detection ----

    #[test]
    fn test_numa_detection_consistent() {
        let a = InferenceThreadPool::detect_numa();
        let b = InferenceThreadPool::detect_numa();
        assert_eq!(a, b);
    }

    #[test]
    fn test_pool_with_affinity_enabled() {
        // Should not panic even if NUMA is unavailable.
        let pool = InferenceThreadPool::new(ThreadPoolConfig {
            num_threads: 2,
            affinity: true,
            ..Default::default()
        })
        .unwrap();
        pool.parallel_for(0..10, 5, |_| {});
    }
}
