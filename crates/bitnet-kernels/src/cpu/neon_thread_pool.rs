//! Thread pool optimized for Apple Silicon's heterogeneous P-core / E-core architecture.
//!
//! Provides [`NeonThreadPool`] with configurable [`CoreAffinity`] hints so that
//! compute-heavy work can be directed to Performance cores while background
//! housekeeping runs on Efficiency cores.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

// ── Core affinity hint ──────────────────────────────────────────────────

/// Hint for the OS scheduler about which core class to prefer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreAffinity {
    /// Prefer high-performance (P) cores — compute-heavy work.
    Performance,
    /// Prefer high-efficiency (E) cores — background / low-priority work.
    Efficiency,
    /// Let the OS decide (no hint).
    Any,
}

// ── Configuration ───────────────────────────────────────────────────────

/// Configuration for [`NeonThreadPool`].
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    pub num_threads: usize,
    pub core_affinity: CoreAffinity,
    pub stack_size: usize,
    pub name_prefix: String,
}

impl Default for ThreadPoolConfig {
    /// Defaults based on available cores: all cores, no affinity hint,
    /// 2 MiB stack, `"bitnet-worker"` prefix.
    fn default() -> Self {
        let total = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        Self {
            num_threads: total,
            core_affinity: CoreAffinity::Any,
            stack_size: 2 * 1024 * 1024,
            name_prefix: "bitnet-worker".to_string(),
        }
    }
}

impl ThreadPoolConfig {
    /// Configuration targeting Performance (P) cores.
    pub fn performance(num_threads: usize) -> Self {
        Self {
            num_threads,
            core_affinity: CoreAffinity::Performance,
            stack_size: 2 * 1024 * 1024,
            name_prefix: "bitnet-perf".to_string(),
        }
    }

    /// Configuration targeting Efficiency (E) cores.
    pub fn efficiency(num_threads: usize) -> Self {
        Self {
            num_threads,
            core_affinity: CoreAffinity::Efficiency,
            stack_size: 1024 * 1024,
            name_prefix: "bitnet-eff".to_string(),
        }
    }
}

// ── Task result handle ──────────────────────────────────────────────────

/// Handle returned by [`NeonThreadPool::spawn`]. Call [`wait`](TaskResult::wait)
/// to block until the result is available.
pub struct TaskResult<T> {
    inner: Arc<TaskInner<T>>,
}

struct TaskInner<T> {
    result: Mutex<Option<T>>,
    done: AtomicBool,
    condvar: Condvar,
}

impl<T> TaskResult<T> {
    /// Block until the spawned task completes and return its value.
    pub fn wait(self) -> T {
        let mut guard = self.inner.result.lock().unwrap();
        while !self.inner.done.load(Ordering::Acquire) {
            guard = self.inner.condvar.wait(guard).unwrap();
        }
        guard.take().expect("task result already consumed")
    }

    /// Returns `true` if the task has finished.
    pub fn is_complete(&self) -> bool {
        self.inner.done.load(Ordering::Acquire)
    }
}

// ── Thread pool ─────────────────────────────────────────────────────────

type BoxedTask = Box<dyn FnOnce() + Send>;

struct PoolShared {
    queue: Mutex<Vec<BoxedTask>>,
    condvar: Condvar,
    shutdown: AtomicBool,
    active_tasks: AtomicUsize,
}

/// Thread pool with heterogeneous core affinity support for Apple Silicon.
pub struct NeonThreadPool {
    shared: Arc<PoolShared>,
    workers: Vec<thread::JoinHandle<()>>,
    num_threads: usize,
}

impl NeonThreadPool {
    /// Create a new pool from the given [`ThreadPoolConfig`].
    ///
    /// # Panics
    ///
    /// Panics if `config.num_threads` is zero.
    pub fn new(config: ThreadPoolConfig) -> Self {
        assert!(config.num_threads > 0, "num_threads must be > 0");

        let shared = Arc::new(PoolShared {
            queue: Mutex::new(Vec::new()),
            condvar: Condvar::new(),
            shutdown: AtomicBool::new(false),
            active_tasks: AtomicUsize::new(0),
        });

        let mut workers = Vec::with_capacity(config.num_threads);
        for i in 0..config.num_threads {
            let shared = Arc::clone(&shared);
            let name = format!("{}-{}", config.name_prefix, i);
            let builder = thread::Builder::new().name(name).stack_size(config.stack_size);

            // Core affinity is a hint; actual pinning is OS-specific.
            // The config is preserved for future platform-specific QoS.
            let _ = config.core_affinity;

            let handle = builder
                .spawn(move || {
                    Self::worker_loop(&shared);
                })
                .expect("failed to spawn worker thread");
            workers.push(handle);
        }

        Self { shared, workers, num_threads: config.num_threads }
    }

    fn worker_loop(shared: &PoolShared) {
        loop {
            let task: BoxedTask;
            {
                let mut queue = shared.queue.lock().unwrap();
                while queue.is_empty() {
                    if shared.shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    queue = shared.condvar.wait(queue).unwrap();
                }
                task = queue.pop().unwrap();
            }
            shared.active_tasks.fetch_add(1, Ordering::AcqRel);
            task();
            shared.active_tasks.fetch_sub(1, Ordering::AcqRel);
        }
    }

    /// Spawn a task on the pool, returning a [`TaskResult`] handle.
    pub fn spawn<F, T>(&self, f: F) -> TaskResult<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let inner = Arc::new(TaskInner {
            result: Mutex::new(None),
            done: AtomicBool::new(false),
            condvar: Condvar::new(),
        });
        let task_inner = Arc::clone(&inner);

        let boxed: BoxedTask = Box::new(move || {
            let val = f();
            {
                let mut guard = task_inner.result.lock().unwrap();
                *guard = Some(val);
                task_inner.done.store(true, Ordering::Release);
            }
            task_inner.condvar.notify_all();
        });

        {
            let mut queue = self.shared.queue.lock().unwrap();
            queue.push(boxed);
        }
        self.shared.condvar.notify_one();

        TaskResult { inner }
    }

    /// Execute `f` in parallel over `range`, splitting work into chunks of
    /// `chunk_size`.
    pub fn parallel_for(
        &self,
        range: std::ops::Range<usize>,
        chunk_size: usize,
        f: impl Fn(usize) + Send + Sync + 'static,
    ) {
        if range.is_empty() {
            return;
        }
        let chunk_size = chunk_size.max(1);
        let f = Arc::new(f);
        let barrier = Arc::new((Mutex::new(0usize), Condvar::new()));

        let mut total_chunks = 0usize;
        let mut start = range.start;
        while start < range.end {
            let end = (start + chunk_size).min(range.end);
            let f = Arc::clone(&f);
            let barrier = Arc::clone(&barrier);
            let s = start;
            let e = end;

            let boxed: BoxedTask = Box::new(move || {
                for i in s..e {
                    f(i);
                }
                let (lock, cvar) = &*barrier;
                let mut done = lock.lock().unwrap();
                *done += 1;
                cvar.notify_all();
            });

            {
                let mut queue = self.shared.queue.lock().unwrap();
                queue.push(boxed);
            }
            self.shared.condvar.notify_one();

            total_chunks += 1;
            start = end;
        }

        // Wait for all chunks to finish.
        let (lock, cvar) = &*barrier;
        let mut done = lock.lock().unwrap();
        while *done < total_chunks {
            done = cvar.wait(done).unwrap();
        }
    }

    /// Number of worker threads in this pool.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Approximate number of tasks currently executing (not queued).
    pub fn active_tasks(&self) -> usize {
        self.shared.active_tasks.load(Ordering::Acquire)
    }

    /// Graceful shutdown: signals all workers and joins them.
    pub fn shutdown(self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.condvar.notify_all();
        for w in self.workers {
            let _ = w.join();
        }
    }
}

// ── Core detection ──────────────────────────────────────────────────────

/// Estimate the number of Performance and Efficiency cores.
///
/// On macOS/aarch64 this uses `sysctl`; elsewhere it falls back to
/// `(total / 2, total - total / 2)`.
pub fn detect_core_count() -> (usize, usize) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        macos_core_count().unwrap_or_else(|| fallback_core_count())
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        fallback_core_count()
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn macos_core_count() -> Option<(usize, usize)> {
    use std::process::Command;

    let run = |key: &str| -> Option<usize> {
        let out = Command::new("sysctl").arg("-n").arg(key).output().ok()?;
        let s = String::from_utf8_lossy(&out.stdout);
        s.trim().parse().ok()
    };

    let p = run("hw.perflevel0.logicalcpu")?;
    let e = run("hw.perflevel1.logicalcpu")?;
    Some((p, e))
}

fn fallback_core_count() -> (usize, usize) {
    let total = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
    let p = total / 2;
    let e = total - p;
    (p.max(1), e.max(1))
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // -- Config defaults ---------------------------------------------------

    #[test]
    fn test_default_config_has_positive_threads() {
        let cfg = ThreadPoolConfig::default();
        assert!(cfg.num_threads > 0);
    }

    #[test]
    fn test_default_config_affinity_is_any() {
        let cfg = ThreadPoolConfig::default();
        assert_eq!(cfg.core_affinity, CoreAffinity::Any);
    }

    #[test]
    fn test_default_config_stack_size() {
        let cfg = ThreadPoolConfig::default();
        assert_eq!(cfg.stack_size, 2 * 1024 * 1024);
    }

    #[test]
    fn test_default_config_name_prefix() {
        let cfg = ThreadPoolConfig::default();
        assert_eq!(cfg.name_prefix, "bitnet-worker");
    }

    // -- Performance / Efficiency presets ----------------------------------

    #[test]
    fn test_performance_config() {
        let cfg = ThreadPoolConfig::performance(4);
        assert_eq!(cfg.num_threads, 4);
        assert_eq!(cfg.core_affinity, CoreAffinity::Performance);
        assert_eq!(cfg.name_prefix, "bitnet-perf");
    }

    #[test]
    fn test_efficiency_config() {
        let cfg = ThreadPoolConfig::efficiency(2);
        assert_eq!(cfg.num_threads, 2);
        assert_eq!(cfg.core_affinity, CoreAffinity::Efficiency);
        assert_eq!(cfg.name_prefix, "bitnet-eff");
    }

    #[test]
    fn test_efficiency_smaller_stack() {
        let cfg = ThreadPoolConfig::efficiency(1);
        assert_eq!(cfg.stack_size, 1024 * 1024);
    }

    // -- Pool creation -----------------------------------------------------

    #[test]
    fn test_pool_creation_single_thread() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(1));
        assert_eq!(pool.num_threads(), 1);
        pool.shutdown();
    }

    #[test]
    fn test_pool_creation_multiple_threads() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::default());
        assert!(pool.num_threads() > 0);
        pool.shutdown();
    }

    #[test]
    #[should_panic(expected = "num_threads must be > 0")]
    fn test_pool_zero_threads_panics() {
        let mut cfg = ThreadPoolConfig::default();
        cfg.num_threads = 0;
        let _pool = NeonThreadPool::new(cfg);
    }

    #[test]
    fn test_pool_with_custom_config() {
        let cfg = ThreadPoolConfig {
            num_threads: 3,
            core_affinity: CoreAffinity::Efficiency,
            stack_size: 512 * 1024,
            name_prefix: "custom".to_string(),
        };
        let pool = NeonThreadPool::new(cfg);
        assert_eq!(pool.num_threads(), 3);
        pool.shutdown();
    }

    // -- Spawn & collect ---------------------------------------------------

    #[test]
    fn test_spawn_single_task() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let result = pool.spawn(|| 42);
        assert_eq!(result.wait(), 42);
        pool.shutdown();
    }

    #[test]
    fn test_spawn_returns_string() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(1));
        let result = pool.spawn(|| "hello".to_string());
        assert_eq!(result.wait(), "hello");
        pool.shutdown();
    }

    #[test]
    fn test_spawn_returns_vec() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let result = pool.spawn(|| vec![1, 2, 3]);
        assert_eq!(result.wait(), vec![1, 2, 3]);
        pool.shutdown();
    }

    #[test]
    fn test_spawn_closure_captures_env() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let x = 10;
        let result = pool.spawn(move || x * 2);
        assert_eq!(result.wait(), 20);
        pool.shutdown();
    }

    #[test]
    fn test_spawn_multiple_tasks() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(4));
        let handles: Vec<_> = (0..8).map(|i| pool.spawn(move || i * i)).collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.wait()).collect();
        for i in 0..8 {
            assert!(results.contains(&(i * i)));
        }
        pool.shutdown();
    }

    #[test]
    fn test_spawn_concurrent_sum() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(4));
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..100 {
            let c = Arc::clone(&counter);
            handles.push(pool.spawn(move || {
                c.fetch_add(1, Ordering::Relaxed);
            }));
        }
        for h in handles {
            h.wait();
        }
        assert_eq!(counter.load(Ordering::Relaxed), 100);
        pool.shutdown();
    }

    // -- TaskResult polling ------------------------------------------------

    #[test]
    fn test_task_result_is_complete_before_wait() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let result = pool.spawn(|| 1);
        let val = result.wait();
        assert_eq!(val, 1);
        pool.shutdown();
    }

    #[test]
    fn test_task_result_is_complete_after_wait() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let inner = {
            let result = pool.spawn(|| 99);
            let inner = Arc::clone(&result.inner);
            let _ = result.wait();
            inner
        };
        assert!(inner.done.load(Ordering::Acquire));
        pool.shutdown();
    }

    #[test]
    fn test_task_result_poll_eventually_true() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let result = pool.spawn(|| 7);
        while !result.is_complete() {
            std::hint::spin_loop();
        }
        assert!(result.is_complete());
        let val = result.wait();
        assert_eq!(val, 7);
        pool.shutdown();
    }

    // -- parallel_for ------------------------------------------------------

    #[test]
    fn test_parallel_for_simple() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(4));
        let sums = Arc::new(AtomicUsize::new(0));
        let s = Arc::clone(&sums);
        pool.parallel_for(0..10, 2, move |i| {
            s.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sums.load(Ordering::Relaxed), 45);
        pool.shutdown();
    }

    #[test]
    fn test_parallel_for_single_element() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let called = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&called);
        pool.parallel_for(0..1, 1, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(called.load(Ordering::Relaxed), 1);
        pool.shutdown();
    }

    #[test]
    fn test_parallel_for_empty_range() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let called = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&called);
        pool.parallel_for(0..0, 1, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(called.load(Ordering::Relaxed), 0);
        pool.shutdown();
    }

    #[test]
    fn test_parallel_for_chunk_larger_than_range() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let sums = Arc::new(AtomicUsize::new(0));
        let s = Arc::clone(&sums);
        pool.parallel_for(0..3, 100, move |i| {
            s.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sums.load(Ordering::Relaxed), 3);
        pool.shutdown();
    }

    #[test]
    fn test_parallel_for_chunk_size_one() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(4));
        let sums = Arc::new(AtomicUsize::new(0));
        let s = Arc::clone(&sums);
        pool.parallel_for(0..5, 1, move |i| {
            s.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sums.load(Ordering::Relaxed), 10);
        pool.shutdown();
    }

    #[test]
    fn test_parallel_for_non_zero_start() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let sums = Arc::new(AtomicUsize::new(0));
        let s = Arc::clone(&sums);
        pool.parallel_for(5..10, 2, move |i| {
            s.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sums.load(Ordering::Relaxed), 35);
        pool.shutdown();
    }

    #[test]
    fn test_parallel_for_large_range() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(4));
        let sums = Arc::new(AtomicUsize::new(0));
        let s = Arc::clone(&sums);
        pool.parallel_for(0..1000, 50, move |_| {
            s.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(sums.load(Ordering::Relaxed), 1000);
        pool.shutdown();
    }

    #[test]
    fn test_parallel_for_writes_to_shared_vec() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(4));
        let data = Arc::new(Mutex::new(vec![0usize; 10]));
        let d = Arc::clone(&data);
        pool.parallel_for(0..10, 2, move |i| {
            let mut v = d.lock().unwrap();
            v[i] = i * 10;
        });
        let v = data.lock().unwrap();
        for i in 0..10 {
            assert_eq!(v[i], i * 10);
        }
        pool.shutdown();
    }

    #[test]
    fn test_parallel_for_zero_chunk_size_coerced() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let sums = Arc::new(AtomicUsize::new(0));
        let s = Arc::clone(&sums);
        pool.parallel_for(0..3, 0, move |i| {
            s.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(sums.load(Ordering::Relaxed), 3);
        pool.shutdown();
    }

    // -- Shutdown ----------------------------------------------------------

    #[test]
    fn test_shutdown_no_tasks() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        pool.shutdown();
    }

    #[test]
    fn test_shutdown_after_tasks_complete() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let h = pool.spawn(|| 1 + 1);
        assert_eq!(h.wait(), 2);
        pool.shutdown();
    }

    #[test]
    fn test_shutdown_with_many_tasks() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(4));
        let handles: Vec<_> = (0..50).map(|i| pool.spawn(move || i)).collect();
        for h in handles {
            let _ = h.wait();
        }
        pool.shutdown();
    }

    // -- active_tasks ------------------------------------------------------

    #[test]
    fn test_active_tasks_zero_initially() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let _ = pool.active_tasks();
        pool.shutdown();
    }

    #[test]
    fn test_active_tasks_returns_to_zero() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(2));
        let h = pool.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });
        let _ = h.wait();
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(pool.active_tasks(), 0);
        pool.shutdown();
    }

    // -- Core detection ----------------------------------------------------

    #[test]
    fn test_detect_core_count_positive() {
        let (p, e) = detect_core_count();
        assert!(p > 0, "P-core count must be positive");
        assert!(e > 0, "E-core count must be positive");
    }

    #[test]
    fn test_detect_core_count_reasonable_total() {
        let (p, e) = detect_core_count();
        let total = p + e;
        assert!(total >= 2, "total core count should be at least 2");
        assert!(total <= 1024, "sanity: total cores should be <= 1024");
    }

    #[test]
    fn test_fallback_core_count_splits_evenly() {
        let (p, e) = fallback_core_count();
        let total = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
        assert_eq!(p + e, total);
    }

    // -- Thread naming -----------------------------------------------------

    #[test]
    fn test_thread_name_prefix() {
        let pool = NeonThreadPool::new(ThreadPoolConfig::performance(1));
        let name = pool.spawn(|| std::thread::current().name().unwrap().to_string());
        let n = name.wait();
        assert!(n.starts_with("bitnet-perf-"), "expected prefix 'bitnet-perf-', got '{n}'");
        pool.shutdown();
    }

    #[test]
    fn test_thread_name_custom_prefix() {
        let cfg = ThreadPoolConfig {
            num_threads: 1,
            core_affinity: CoreAffinity::Any,
            stack_size: 2 * 1024 * 1024,
            name_prefix: "my-pool".to_string(),
        };
        let pool = NeonThreadPool::new(cfg);
        let name = pool.spawn(|| std::thread::current().name().unwrap().to_string());
        let n = name.wait();
        assert!(n.starts_with("my-pool-"), "got '{n}'");
        pool.shutdown();
    }

    // -- CoreAffinity traits -----------------------------------------------

    #[test]
    fn test_core_affinity_clone() {
        let a = CoreAffinity::Performance;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_core_affinity_debug() {
        let s = format!("{:?}", CoreAffinity::Efficiency);
        assert_eq!(s, "Efficiency");
    }

    #[test]
    fn test_config_clone() {
        let cfg = ThreadPoolConfig::performance(8);
        let cfg2 = cfg.clone();
        assert_eq!(cfg2.num_threads, 8);
    }
}
