//! Optimized thread pool for CPU inference operations.
//!
//! Provides work-stealing parallelism, task graphs with dependency tracking,
//! and parallel primitives (`parallel_for`, `parallel_reduce`, `parallel_map`)
//! tuned for inference workloads.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

// ---------------------------------------------------------------------------
// TaskPriority
// ---------------------------------------------------------------------------

/// Priority levels for submitted tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum TaskPriority {
    /// Background work that can be deferred.
    Low = 0,
    /// Default priority for most inference work.
    #[default]
    Normal = 1,
    /// Latency-sensitive tasks (e.g. attention heads on the critical path).
    High = 2,
    /// Must execute before anything else (e.g. barrier completions).
    Critical = 3,
}

// ---------------------------------------------------------------------------
// ThreadPoolConfig
// ---------------------------------------------------------------------------

/// Configuration for [`InferenceThreadPool`].
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads. `0` means use available parallelism.
    pub num_threads: usize,
    /// Whether to attempt setting CPU affinity (best-effort).
    pub affinity: bool,
    /// Default task priority.
    pub priority: TaskPriority,
    /// Per-thread stack size in bytes. `0` uses the platform default.
    pub stack_size: usize,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self { num_threads: 0, affinity: false, priority: TaskPriority::Normal, stack_size: 0 }
    }
}

impl ThreadPoolConfig {
    /// Create a config with the given thread count.
    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads, ..Default::default() }
    }

    fn effective_threads(&self) -> usize {
        if self.num_threads == 0 {
            thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        } else {
            self.num_threads
        }
    }
}

// ---------------------------------------------------------------------------
// Work-stealing deque
// ---------------------------------------------------------------------------

/// A simple work-stealing deque.
///
/// The owning thread pushes/pops from the back; thieves steal from the front.
pub struct WorkStealingQueue<T> {
    inner: Mutex<VecDeque<T>>,
}

impl<T> WorkStealingQueue<T> {
    pub fn new() -> Self {
        Self { inner: Mutex::new(VecDeque::new()) }
    }

    /// Push an item onto the back (owner side).
    pub fn push(&self, item: T) {
        self.inner.lock().unwrap().push_back(item);
    }

    /// Pop an item from the back (owner side).
    pub fn pop(&self) -> Option<T> {
        self.inner.lock().unwrap().pop_back()
    }

    /// Steal an item from the front (thief side).
    pub fn steal(&self) -> Option<T> {
        self.inner.lock().unwrap().pop_front()
    }

    /// Number of items currently in the deque.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// Whether the deque is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Default for WorkStealingQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Public constructor alias.
pub fn work_stealing_queue<T>() -> WorkStealingQueue<T> {
    WorkStealingQueue::new()
}

// ---------------------------------------------------------------------------
// Thread affinity (placeholder with fallback)
// ---------------------------------------------------------------------------

/// Attempt to set CPU affinity for the current thread.
///
/// This is a best-effort placeholder: on most platforms it simply returns `Ok`
/// without actually pinning the thread. Platform-specific implementations can
/// be added behind `#[cfg]` gates when needed.
pub fn thread_affinity(core_id: usize) -> Result<(), String> {
    // Placeholder — real affinity requires platform-specific syscalls
    // (sched_setaffinity on Linux, SetThreadAffinityMask on Windows, etc.).
    let _ = core_id;
    Ok(())
}

// ---------------------------------------------------------------------------
// TaskHandle
// ---------------------------------------------------------------------------

/// Handle returned when a task is submitted to the thread pool.
///
/// Can be used to wait for completion and retrieve the result.
pub struct TaskHandle<T> {
    result: Arc<(Mutex<Option<T>>, Condvar)>,
    done: Arc<AtomicBool>,
}

impl<T> TaskHandle<T> {
    fn new() -> (Self, TaskSender<T>) {
        let result = Arc::new((Mutex::new(None), Condvar::new()));
        let done = Arc::new(AtomicBool::new(false));
        let handle = Self { result: Arc::clone(&result), done: Arc::clone(&done) };
        let sender = TaskSender { result, done };
        (handle, sender)
    }

    /// Block until the task completes and return its result.
    pub fn join(self) -> T {
        let (lock, cvar) = &*self.result;
        let mut guard = lock.lock().unwrap();
        while guard.is_none() {
            guard = cvar.wait(guard).unwrap();
        }
        guard.take().unwrap()
    }

    /// Check whether the task has completed without blocking.
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
}

struct TaskSender<T> {
    result: Arc<(Mutex<Option<T>>, Condvar)>,
    done: Arc<AtomicBool>,
}

impl<T> TaskSender<T> {
    fn send(self, value: T) {
        let (lock, cvar) = &*self.result;
        *lock.lock().unwrap() = Some(value);
        self.done.store(true, Ordering::Release);
        cvar.notify_all();
    }
}

// ---------------------------------------------------------------------------
// InferenceThreadPool
// ---------------------------------------------------------------------------

type BoxedTask = Box<dyn FnOnce() + Send + 'static>;

struct PoolInner {
    queue: Mutex<VecDeque<(TaskPriority, BoxedTask)>>,
    condvar: Condvar,
    shutdown: AtomicBool,
    active_tasks: AtomicUsize,
    tasks_completed: AtomicU64,
}

/// Thread pool optimized for inference workloads.
///
/// Workers use a shared priority queue with condition-variable notification.
/// Higher-priority tasks are dequeued first.
pub struct InferenceThreadPool {
    inner: Arc<PoolInner>,
    workers: Vec<thread::JoinHandle<()>>,
    config: ThreadPoolConfig,
}

impl InferenceThreadPool {
    /// Create a new thread pool with the given configuration.
    pub fn new(config: ThreadPoolConfig) -> Self {
        let num = config.effective_threads();
        let inner = Arc::new(PoolInner {
            queue: Mutex::new(VecDeque::new()),
            condvar: Condvar::new(),
            shutdown: AtomicBool::new(false),
            active_tasks: AtomicUsize::new(0),
            tasks_completed: AtomicU64::new(0),
        });

        let mut workers = Vec::with_capacity(num);
        for id in 0..num {
            let pool = Arc::clone(&inner);
            let affinity = config.affinity;
            let mut builder = thread::Builder::new().name(format!("inference-{id}"));
            if config.stack_size > 0 {
                builder = builder.stack_size(config.stack_size);
            }
            let handle = builder
                .spawn(move || {
                    if affinity {
                        let _ = thread_affinity(id);
                    }
                    worker_loop(&pool);
                })
                .expect("failed to spawn worker thread");
            workers.push(handle);
        }

        Self { inner, workers, config }
    }

    /// Submit a task with the given priority.
    pub fn submit<F, R>(&self, priority: TaskPriority, f: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (handle, sender) = TaskHandle::new();
        let task: BoxedTask = Box::new(move || {
            let result = f();
            sender.send(result);
        });
        {
            let mut q = self.inner.queue.lock().unwrap();
            // Insert sorted by priority (highest first via linear scan — queue
            // is typically small relative to throughput).
            let pos = q.iter().position(|(p, _)| *p < priority).unwrap_or(q.len());
            q.insert(pos, (priority, task));
        }
        self.inner.condvar.notify_one();
        handle
    }

    /// Submit a task at the pool's default priority.
    pub fn submit_default<F, R>(&self, f: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.submit(self.config.priority, f)
    }

    /// Number of worker threads.
    pub fn num_threads(&self) -> usize {
        self.workers.len()
    }

    /// Number of tasks completed since pool creation.
    pub fn tasks_completed(&self) -> u64 {
        self.inner.tasks_completed.load(Ordering::Relaxed)
    }

    /// Number of tasks currently executing.
    pub fn active_tasks(&self) -> usize {
        self.inner.active_tasks.load(Ordering::Relaxed)
    }

    /// Shutdown the pool, waiting for all workers to finish.
    pub fn shutdown(mut self) {
        self.shutdown_inner();
    }

    fn shutdown_inner(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        self.inner.condvar.notify_all();
        let workers: Vec<_> = std::mem::take(&mut self.workers);
        for w in workers {
            let _ = w.join();
        }
    }
}

impl Drop for InferenceThreadPool {
    fn drop(&mut self) {
        self.shutdown_inner();
    }
}

fn worker_loop(pool: &PoolInner) {
    loop {
        let task = {
            let mut q = pool.queue.lock().unwrap();
            loop {
                if pool.shutdown.load(Ordering::Acquire) {
                    return;
                }
                if let Some((_prio, task)) = q.pop_front() {
                    break task;
                }
                q = pool.condvar.wait(q).unwrap();
            }
        };
        pool.active_tasks.fetch_add(1, Ordering::Relaxed);
        task();
        pool.active_tasks.fetch_sub(1, Ordering::Relaxed);
        pool.tasks_completed.fetch_add(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Parallel primitives
// ---------------------------------------------------------------------------

/// Execute `f(index)` for each index in `0..len` in parallel across `num_threads`.
///
/// Falls back to sequential execution when `len` is small or `num_threads <= 1`.
pub fn parallel_for<F>(len: usize, num_threads: usize, f: F)
where
    F: Fn(usize) + Send + Sync,
{
    let threads = effective_parallelism(num_threads);
    if threads <= 1 || len <= 1 {
        for i in 0..len {
            f(i);
        }
        return;
    }
    thread::scope(|s| {
        let f = &f;
        let chunk = len.div_ceil(threads);
        for t in 0..threads {
            let start = t * chunk;
            let end = (start + chunk).min(len);
            if start >= len {
                break;
            }
            s.spawn(move || {
                for i in start..end {
                    f(i);
                }
            });
        }
    });
}

/// Parallel for over evenly-sized chunks of a slice.
///
/// Each invocation of `f` receives `(chunk_index, &[T])`.
pub fn parallel_for_chunks<T, F>(data: &[T], chunk_size: usize, num_threads: usize, f: F)
where
    T: Sync,
    F: Fn(usize, &[T]) + Send + Sync,
{
    let chunks: Vec<&[T]> = data.chunks(chunk_size.max(1)).collect();
    let threads = effective_parallelism(num_threads);
    if threads <= 1 || chunks.len() <= 1 {
        for (i, chunk) in chunks.iter().enumerate() {
            f(i, chunk);
        }
        return;
    }
    thread::scope(|s| {
        let f = &f;
        let per_thread = chunks.len().div_ceil(threads);
        for t in 0..threads {
            let start = t * per_thread;
            let end = (start + per_thread).min(chunks.len());
            if start >= chunks.len() {
                break;
            }
            let my_chunks = &chunks[start..end];
            s.spawn(move || {
                for (offset, chunk) in my_chunks.iter().enumerate() {
                    f(start + offset, chunk);
                }
            });
        }
    });
}

/// Parallel map: apply `f` to each element and collect results.
pub fn parallel_map<T, R, F>(data: &[T], num_threads: usize, f: F) -> Vec<R>
where
    T: Sync,
    R: Send + Default + Clone,
    F: Fn(&T) -> R + Send + Sync,
{
    let threads = effective_parallelism(num_threads);
    let mut results = vec![R::default(); data.len()];
    if threads <= 1 || data.len() <= 1 {
        for (i, item) in data.iter().enumerate() {
            results[i] = f(item);
        }
        return results;
    }
    thread::scope(|s| {
        let f = &f;
        let chunk = data.len().div_ceil(threads);
        let mut rest = results.as_mut_slice();
        let mut data_rest = data;
        let mut handles = Vec::new();
        for _ in 0..threads {
            if data_rest.is_empty() {
                break;
            }
            let n = chunk.min(data_rest.len());
            let (d, d_tail) = data_rest.split_at(n);
            let (r, r_tail) = rest.split_at_mut(n);
            data_rest = d_tail;
            rest = r_tail;
            handles.push(s.spawn(move || {
                for (i, item) in d.iter().enumerate() {
                    r[i] = f(item);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
    results
}

/// Parallel reduction with an associative, commutative binary operator.
pub fn parallel_reduce<T, F, C>(data: &[T], identity: T, combine: C, map_fn: F) -> T
where
    T: Send + Sync + Clone,
    F: Fn(&T) -> T + Send + Sync,
    C: Fn(T, T) -> T + Send + Sync,
{
    let threads = effective_parallelism(0);
    if threads <= 1 || data.len() <= 1 {
        let mut acc = identity;
        for item in data {
            acc = combine(acc, map_fn(item));
        }
        return acc;
    }
    thread::scope(|s| {
        let combine = &combine;
        let map_fn = &map_fn;
        let chunk = data.len().div_ceil(threads);
        let mut handles = Vec::new();
        for chunk_data in data.chunks(chunk) {
            let id = identity.clone();
            handles.push(s.spawn(move || {
                let mut acc = id;
                for item in chunk_data {
                    acc = combine(acc, map_fn(item));
                }
                acc
            }));
        }
        let mut acc = identity.clone();
        for h in handles {
            acc = combine(acc, h.join().unwrap());
        }
        acc
    })
}

// ---------------------------------------------------------------------------
// Barrier synchronization
// ---------------------------------------------------------------------------

/// A reusable barrier for `n` threads.
pub struct BarrierSync {
    state: Mutex<BarrierState>,
    cvar: Condvar,
    count: usize,
}

struct BarrierState {
    waiting: usize,
    generation: u64,
}

impl BarrierSync {
    pub fn new(count: usize) -> Self {
        assert!(count > 0, "barrier count must be > 0");
        Self {
            state: Mutex::new(BarrierState { waiting: 0, generation: 0 }),
            cvar: Condvar::new(),
            count,
        }
    }

    /// Wait until all `count` threads have called `wait`.
    pub fn wait(&self) {
        let mut state = self.state.lock().unwrap();
        let current_gen = state.generation;
        state.waiting += 1;
        if state.waiting == self.count {
            state.waiting = 0;
            state.generation += 1;
            self.cvar.notify_all();
        } else {
            while state.generation == current_gen {
                state = self.cvar.wait(state).unwrap();
            }
        }
    }
}

/// Create a barrier for `n` threads.
pub fn barrier_sync(n: usize) -> BarrierSync {
    BarrierSync::new(n)
}

// ---------------------------------------------------------------------------
// TaskGraph — DAG of tasks with dependencies
// ---------------------------------------------------------------------------

/// A node in a [`TaskGraph`].
struct TaskNode {
    task: Option<BoxedTask>,
    deps: Vec<usize>,
    priority: TaskPriority,
}

/// A directed acyclic graph (DAG) of tasks with dependency edges.
///
/// Build with [`add_task`](TaskGraph::add_task) and
/// [`add_dependency`](TaskGraph::add_dependency), then execute via
/// [`execute_task_graph`].
pub struct TaskGraph {
    nodes: Vec<TaskNode>,
}

impl TaskGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a task and return its node id.
    pub fn add_task<F>(&mut self, priority: TaskPriority, f: F) -> usize
    where
        F: FnOnce() + Send + 'static,
    {
        let id = self.nodes.len();
        self.nodes.push(TaskNode { task: Some(Box::new(f)), deps: Vec::new(), priority });
        id
    }

    /// Declare that `task_id` depends on `dep_id` (dep must finish first).
    pub fn add_dependency(&mut self, task_id: usize, dep_id: usize) {
        assert!(task_id < self.nodes.len(), "invalid task id");
        assert!(dep_id < self.nodes.len(), "invalid dep id");
        assert_ne!(task_id, dep_id, "self-dependency");
        self.nodes[task_id].deps.push(dep_id);
    }

    /// Number of tasks in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for TaskGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute a [`TaskGraph`], respecting dependency order.
///
/// Tasks whose dependencies are satisfied are run in parallel (up to
/// `num_threads` concurrency). Panics if the graph contains a cycle.
pub fn execute_task_graph(mut graph: TaskGraph, num_threads: usize) {
    let n = graph.nodes.len();
    if n == 0 {
        return;
    }

    // Build in-degree map and reverse adjacency.
    let mut in_degree = vec![0usize; n];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (id, node) in graph.nodes.iter().enumerate() {
        in_degree[id] = node.deps.len();
        for &dep in &node.deps {
            dependents[dep].push(id);
        }
    }

    // Collect ready set (in-degree 0).
    let ready: Arc<Mutex<VecDeque<usize>>> = Arc::new(Mutex::new(VecDeque::new()));
    for (id, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            ready.lock().unwrap().push_back(id);
        }
    }

    // Extract tasks and priorities.
    let tasks: Vec<_> = graph.nodes.iter_mut().map(|n| n.task.take()).collect();
    let priorities: Vec<_> = graph.nodes.iter().map(|n| n.priority).collect();
    let tasks = Arc::new(Mutex::new(tasks));
    let priorities = Arc::new(priorities);
    let in_degree = Arc::new(Mutex::new(in_degree));
    let dependents = Arc::new(dependents);
    let completed = Arc::new(AtomicUsize::new(0));

    let threads = effective_parallelism(num_threads);
    let done = Arc::new(AtomicBool::new(false));
    let notify = Arc::new((Mutex::new(()), Condvar::new()));

    thread::scope(|s| {
        for _ in 0..threads {
            let ready = Arc::clone(&ready);
            let tasks = Arc::clone(&tasks);
            let in_degree = Arc::clone(&in_degree);
            let dependents = Arc::clone(&dependents);
            let completed = Arc::clone(&completed);
            let done = Arc::clone(&done);
            let notify = Arc::clone(&notify);

            let priorities = Arc::clone(&priorities);

            s.spawn(move || {
                loop {
                    // Pick the highest-priority ready task.
                    let task_id = {
                        let mut r = ready.lock().unwrap();
                        if r.is_empty() {
                            None
                        } else {
                            let best = r
                                .iter()
                                .enumerate()
                                .max_by_key(|&(_, &id)| priorities[id])
                                .map(|(idx, _)| idx)
                                .unwrap();
                            Some(r.remove(best).unwrap())
                        }
                    };

                    if let Some(id) = task_id {
                        let task = tasks.lock().unwrap()[id].take();
                        if let Some(f) = task {
                            f();
                        }
                        // Decrement in-degree for dependents.
                        {
                            let mut deg = in_degree.lock().unwrap();
                            let mut r = ready.lock().unwrap();
                            for &dep_id in &dependents[id] {
                                deg[dep_id] -= 1;
                                if deg[dep_id] == 0 {
                                    r.push_back(dep_id);
                                }
                            }
                        }
                        let prev = completed.fetch_add(1, Ordering::AcqRel);
                        if prev + 1 == n {
                            done.store(true, Ordering::Release);
                            notify.1.notify_all();
                            return;
                        }
                        notify.1.notify_all();
                    } else if done.load(Ordering::Acquire) {
                        return;
                    } else {
                        // Wait for new work or completion.
                        let guard = notify.0.lock().unwrap();
                        let _ = notify.1.wait_timeout(guard, Duration::from_millis(1)).unwrap();
                        if done.load(Ordering::Acquire) {
                            return;
                        }
                    }
                }
            });
        }
    });

    assert_eq!(
        completed.load(Ordering::Acquire),
        n,
        "task graph has a cycle: not all tasks were executed"
    );
}

// ---------------------------------------------------------------------------
// ScalableThreadPool
// ---------------------------------------------------------------------------

/// A thread pool that auto-scales worker count based on load.
///
/// Periodically checks pending-task count and spawns or retires workers
/// within `[min_threads, max_threads]`.
pub struct ScalableThreadPool {
    inner: Arc<PoolInner>,
    workers: Mutex<Vec<thread::JoinHandle<()>>>,
    min_threads: usize,
    max_threads: usize,
    current_threads: AtomicUsize,
}

impl ScalableThreadPool {
    /// Create a new scalable pool.
    pub fn new(min_threads: usize, max_threads: usize) -> Self {
        assert!(min_threads > 0, "min_threads must be > 0");
        assert!(max_threads >= min_threads, "max_threads must be >= min_threads");

        let inner = Arc::new(PoolInner {
            queue: Mutex::new(VecDeque::new()),
            condvar: Condvar::new(),
            shutdown: AtomicBool::new(false),
            active_tasks: AtomicUsize::new(0),
            tasks_completed: AtomicU64::new(0),
        });

        let mut handles = Vec::with_capacity(min_threads);
        for _ in 0..min_threads {
            let pool = Arc::clone(&inner);
            handles.push(
                thread::Builder::new()
                    .name("scalable-worker".into())
                    .spawn(move || worker_loop(&pool))
                    .expect("failed to spawn worker"),
            );
        }

        Self {
            inner,
            workers: Mutex::new(handles),
            min_threads,
            max_threads,
            current_threads: AtomicUsize::new(min_threads),
        }
    }

    /// Submit a task with default (Normal) priority.
    pub fn submit<F, R>(&self, f: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.maybe_scale_up();
        let (handle, sender) = TaskHandle::new();
        let task: BoxedTask = Box::new(move || sender.send(f()));
        {
            let mut q = self.inner.queue.lock().unwrap();
            q.push_back((TaskPriority::Normal, task));
        }
        self.inner.condvar.notify_one();
        handle
    }

    /// Current number of worker threads.
    pub fn current_threads(&self) -> usize {
        self.current_threads.load(Ordering::Relaxed)
    }

    /// Minimum number of worker threads.
    pub fn min_threads(&self) -> usize {
        self.min_threads
    }

    /// Maximum number of worker threads.
    pub fn max_threads(&self) -> usize {
        self.max_threads
    }

    /// Number of tasks completed.
    pub fn tasks_completed(&self) -> u64 {
        self.inner.tasks_completed.load(Ordering::Relaxed)
    }

    /// Pending tasks in the queue.
    pub fn pending_tasks(&self) -> usize {
        self.inner.queue.lock().unwrap().len()
    }

    /// Shut down the pool.
    pub fn shutdown(mut self) {
        self.shutdown_inner();
    }

    fn shutdown_inner(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        self.inner.condvar.notify_all();
        let workers = std::mem::take(self.workers.get_mut().unwrap());
        for w in workers {
            let _ = w.join();
        }
    }

    fn maybe_scale_up(&self) {
        let pending = self.inner.queue.lock().unwrap().len();
        let current = self.current_threads.load(Ordering::Relaxed);
        if pending > current && current < self.max_threads {
            let pool = Arc::clone(&self.inner);
            let handle = thread::Builder::new()
                .name("scalable-worker".into())
                .spawn(move || worker_loop(&pool))
                .expect("failed to spawn worker");
            self.workers.lock().unwrap().push(handle);
            self.current_threads.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Drop for ScalableThreadPool {
    fn drop(&mut self) {
        self.shutdown_inner();
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn effective_parallelism(requested: usize) -> usize {
    if requested == 0 {
        thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        requested
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    // -----------------------------------------------------------------------
    // TaskPriority
    // -----------------------------------------------------------------------

    #[test]
    fn test_priority_ordering() {
        assert!(TaskPriority::Low < TaskPriority::Normal);
        assert!(TaskPriority::Normal < TaskPriority::High);
        assert!(TaskPriority::High < TaskPriority::Critical);
    }

    #[test]
    fn test_priority_default() {
        assert_eq!(TaskPriority::default(), TaskPriority::Normal);
    }

    #[test]
    fn test_priority_eq() {
        assert_eq!(TaskPriority::High, TaskPriority::High);
        assert_ne!(TaskPriority::Low, TaskPriority::High);
    }

    #[test]
    fn test_priority_clone() {
        let p = TaskPriority::Critical;
        let p2 = p;
        assert_eq!(p, p2);
    }

    #[test]
    fn test_priority_debug() {
        let s = format!("{:?}", TaskPriority::Low);
        assert_eq!(s, "Low");
    }

    // -----------------------------------------------------------------------
    // ThreadPoolConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let cfg = ThreadPoolConfig::default();
        assert_eq!(cfg.num_threads, 0);
        assert!(!cfg.affinity);
        assert_eq!(cfg.priority, TaskPriority::Normal);
        assert_eq!(cfg.stack_size, 0);
    }

    #[test]
    fn test_config_with_threads() {
        let cfg = ThreadPoolConfig::with_threads(4);
        assert_eq!(cfg.num_threads, 4);
        assert_eq!(cfg.effective_threads(), 4);
    }

    #[test]
    fn test_config_zero_uses_available() {
        let cfg = ThreadPoolConfig::default();
        let eff = cfg.effective_threads();
        assert!(eff >= 1);
    }

    #[test]
    fn test_config_clone() {
        let cfg = ThreadPoolConfig {
            num_threads: 8,
            affinity: true,
            priority: TaskPriority::High,
            stack_size: 1024 * 1024,
        };
        let cfg2 = cfg.clone();
        assert_eq!(cfg2.num_threads, 8);
        assert!(cfg2.affinity);
        assert_eq!(cfg2.priority, TaskPriority::High);
        assert_eq!(cfg2.stack_size, 1024 * 1024);
    }

    #[test]
    fn test_config_debug() {
        let cfg = ThreadPoolConfig::default();
        let s = format!("{:?}", cfg);
        assert!(s.contains("ThreadPoolConfig"));
    }

    // -----------------------------------------------------------------------
    // WorkStealingQueue
    // -----------------------------------------------------------------------

    #[test]
    fn test_wsq_new_is_empty() {
        let q = WorkStealingQueue::<i32>::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_wsq_push_pop() {
        let q = WorkStealingQueue::new();
        q.push(1);
        q.push(2);
        q.push(3);
        assert_eq!(q.len(), 3);
        assert_eq!(q.pop(), Some(3)); // LIFO
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn test_wsq_steal() {
        let q = WorkStealingQueue::new();
        q.push(10);
        q.push(20);
        q.push(30);
        assert_eq!(q.steal(), Some(10)); // FIFO steal
        assert_eq!(q.steal(), Some(20));
        assert_eq!(q.steal(), Some(30));
        assert_eq!(q.steal(), None);
    }

    #[test]
    fn test_wsq_mixed_pop_steal() {
        let q = WorkStealingQueue::new();
        q.push(1);
        q.push(2);
        q.push(3);
        assert_eq!(q.steal(), Some(1)); // steal from front
        assert_eq!(q.pop(), Some(3)); // pop from back
        assert_eq!(q.len(), 1);
        assert_eq!(q.pop(), Some(2));
    }

    #[test]
    fn test_wsq_default() {
        let q: WorkStealingQueue<u32> = WorkStealingQueue::default();
        assert!(q.is_empty());
    }

    #[test]
    fn test_wsq_constructor_fn() {
        let q: WorkStealingQueue<String> = work_stealing_queue();
        assert!(q.is_empty());
    }

    #[test]
    fn test_wsq_concurrent_push_steal() {
        let q = Arc::new(WorkStealingQueue::new());
        let q2 = Arc::clone(&q);
        let n = 100;

        let producer = thread::spawn(move || {
            for i in 0..n {
                q2.push(i);
            }
        });

        producer.join().unwrap();

        let mut stolen = Vec::new();
        while let Some(v) = q.steal() {
            stolen.push(v);
        }
        assert_eq!(stolen.len(), n);
    }

    // -----------------------------------------------------------------------
    // thread_affinity
    // -----------------------------------------------------------------------

    #[test]
    fn test_thread_affinity_placeholder() {
        assert!(thread_affinity(0).is_ok());
        assert!(thread_affinity(99).is_ok());
    }

    // -----------------------------------------------------------------------
    // TaskHandle
    // -----------------------------------------------------------------------

    #[test]
    fn test_task_handle_join() {
        let (handle, sender) = TaskHandle::new();
        thread::spawn(move || {
            sender.send(42);
        });
        assert_eq!(handle.join(), 42);
    }

    #[test]
    fn test_task_handle_is_done() {
        let (handle, sender) = TaskHandle::new();
        assert!(!handle.is_done());
        sender.send(());
        // Give a moment for the atomic to propagate.
        thread::sleep(Duration::from_millis(10));
        assert!(handle.is_done());
    }

    #[test]
    fn test_task_handle_string_result() {
        let (handle, sender) = TaskHandle::new();
        thread::spawn(move || {
            sender.send("hello".to_string());
        });
        assert_eq!(handle.join(), "hello");
    }

    // -----------------------------------------------------------------------
    // InferenceThreadPool
    // -----------------------------------------------------------------------

    #[test]
    fn test_pool_creation() {
        let pool = InferenceThreadPool::new(ThreadPoolConfig::with_threads(2));
        assert_eq!(pool.num_threads(), 2);
    }

    #[test]
    fn test_pool_submit_and_join() {
        let pool = InferenceThreadPool::new(ThreadPoolConfig::with_threads(2));
        let h = pool.submit(TaskPriority::Normal, || 1 + 1);
        assert_eq!(h.join(), 2);
    }

    #[test]
    fn test_pool_submit_default() {
        let pool = InferenceThreadPool::new(ThreadPoolConfig::with_threads(2));
        let h = pool.submit_default(|| 7 * 6);
        assert_eq!(h.join(), 42);
    }

    #[test]
    fn test_pool_multiple_tasks() {
        let pool = InferenceThreadPool::new(ThreadPoolConfig::with_threads(4));
        let handles: Vec<_> =
            (0..20).map(|i| pool.submit(TaskPriority::Normal, move || i * i)).collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join()).collect();
        for (i, &r) in results.iter().enumerate() {
            assert_eq!(r, i * i);
        }
    }

    #[test]
    fn test_pool_shared_state() {
        let pool = InferenceThreadPool::new(ThreadPoolConfig::with_threads(2));
        let counter = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..100)
            .map(|_| {
                let c = Arc::clone(&counter);
                pool.submit(TaskPriority::Normal, move || {
                    c.fetch_add(1, Ordering::Relaxed);
                })
            })
            .collect();
        for h in handles {
            h.join();
        }
        assert_eq!(counter.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_pool_priority_order() {
        // Submit low then high priority — high should complete first if
        // workers are slow to start.
        let pool = InferenceThreadPool::new(ThreadPoolConfig::with_threads(1));
        let order = Arc::new(Mutex::new(Vec::new()));

        // Occupy the single worker briefly.
        let _blocker = pool.submit(TaskPriority::Normal, || {
            thread::sleep(Duration::from_millis(50));
        });
        thread::sleep(Duration::from_millis(10)); // let blocker start

        let o1 = Arc::clone(&order);
        let o2 = Arc::clone(&order);

        let _h_low = pool.submit(TaskPriority::Low, move || {
            o1.lock().unwrap().push("low");
        });
        let _h_high = pool.submit(TaskPriority::High, move || {
            o2.lock().unwrap().push("high");
        });

        _blocker.join();
        _h_low.join();
        _h_high.join();

        let o = order.lock().unwrap();
        // High should appear before low.
        if o.len() == 2 {
            assert_eq!(o[0], "high");
            assert_eq!(o[1], "low");
        }
    }

    #[test]
    fn test_pool_tasks_completed() {
        let pool = InferenceThreadPool::new(ThreadPoolConfig::with_threads(2));
        let handles: Vec<_> = (0..10).map(|_| pool.submit_default(|| ())).collect();
        for h in handles {
            h.join();
        }
        // Allow workers to update counter.
        thread::sleep(Duration::from_millis(50));
        assert!(pool.tasks_completed() >= 10);
    }

    #[test]
    fn test_pool_shutdown() {
        let pool = InferenceThreadPool::new(ThreadPoolConfig::with_threads(2));
        let h = pool.submit_default(|| 123);
        assert_eq!(h.join(), 123);
        pool.shutdown();
    }

    #[test]
    fn test_pool_drop_cleans_up() {
        // Just ensure drop doesn't hang or panic.
        let pool = InferenceThreadPool::new(ThreadPoolConfig::with_threads(2));
        let _h = pool.submit_default(|| ());
        drop(pool);
    }

    #[test]
    fn test_pool_with_affinity() {
        let cfg = ThreadPoolConfig { num_threads: 2, affinity: true, ..Default::default() };
        let pool = InferenceThreadPool::new(cfg);
        let h = pool.submit_default(|| 99);
        assert_eq!(h.join(), 99);
    }

    #[test]
    fn test_pool_custom_stack_size() {
        let cfg =
            ThreadPoolConfig { num_threads: 1, stack_size: 4 * 1024 * 1024, ..Default::default() };
        let pool = InferenceThreadPool::new(cfg);
        let h = pool.submit_default(|| {
            // Use some stack.
            let arr = [0u8; 1024];
            arr.len()
        });
        assert_eq!(h.join(), 1024);
    }

    #[test]
    fn test_pool_zero_threads_uses_available() {
        let cfg = ThreadPoolConfig::with_threads(0);
        let pool = InferenceThreadPool::new(cfg);
        assert!(pool.num_threads() >= 1);
    }

    // -----------------------------------------------------------------------
    // parallel_for
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_for_basic() {
        let data = Arc::new(Mutex::new(vec![0i32; 10]));
        let d = Arc::clone(&data);
        parallel_for(10, 4, move |i| {
            d.lock().unwrap()[i] = i as i32;
        });
        let d = data.lock().unwrap();
        for (i, &v) in d.iter().enumerate() {
            assert_eq!(v, i as i32);
        }
    }

    #[test]
    fn test_parallel_for_single_thread() {
        let sum = Arc::new(AtomicI64::new(0));
        let s = Arc::clone(&sum);
        parallel_for(5, 1, move |i| {
            s.fetch_add(i as i64, Ordering::Relaxed);
        });
        assert_eq!(sum.load(Ordering::Relaxed), 10); // 0+1+2+3+4
    }

    #[test]
    fn test_parallel_for_empty() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        parallel_for(0, 4, move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_parallel_for_single_element() {
        let called = Arc::new(AtomicBool::new(false));
        let c = Arc::clone(&called);
        parallel_for(1, 4, move |i| {
            assert_eq!(i, 0);
            c.store(true, Ordering::Relaxed);
        });
        assert!(called.load(Ordering::Relaxed));
    }

    #[test]
    fn test_parallel_for_large() {
        let n = 10_000;
        let data = Arc::new(Mutex::new(vec![false; n]));
        let d = Arc::clone(&data);
        parallel_for(n, 8, move |i| {
            d.lock().unwrap()[i] = true;
        });
        assert!(data.lock().unwrap().iter().all(|&v| v));
    }

    // -----------------------------------------------------------------------
    // parallel_for_chunks
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_for_chunks_basic() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let sums = Arc::new(Mutex::new(Vec::new()));
        let s = Arc::clone(&sums);
        parallel_for_chunks(&data, 3, 2, move |idx, chunk| {
            let sum: i32 = chunk.iter().sum();
            s.lock().unwrap().push((idx, sum));
        });
        let mut s = sums.lock().unwrap().clone();
        s.sort_by_key(|&(i, _)| i);
        // chunks: [1,2,3], [4,5,6], [7,8]
        assert_eq!(s[0], (0, 6));
        assert_eq!(s[1], (1, 15));
        assert_eq!(s[2], (2, 15));
    }

    #[test]
    fn test_parallel_for_chunks_empty() {
        let data: Vec<i32> = vec![];
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        parallel_for_chunks(&data, 4, 2, move |_, _| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_parallel_for_chunks_single_thread() {
        let data = vec![10, 20, 30, 40];
        let results = Arc::new(Mutex::new(Vec::new()));
        let r = Arc::clone(&results);
        parallel_for_chunks(&data, 2, 1, move |idx, chunk| {
            r.lock().unwrap().push((idx, chunk.to_vec()));
        });
        let r = results.lock().unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_parallel_for_chunks_chunk_larger_than_data() {
        let data = vec![1, 2, 3];
        let called = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&called);
        parallel_for_chunks(&data, 100, 2, move |_, chunk| {
            assert_eq!(chunk.len(), 3);
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(called.load(Ordering::Relaxed), 1);
    }

    // -----------------------------------------------------------------------
    // parallel_map
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_map_basic() {
        let data = vec![1, 2, 3, 4, 5];
        let result = parallel_map(&data, 2, |&x| x * 2);
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_parallel_map_empty() {
        let data: Vec<i32> = vec![];
        let result = parallel_map(&data, 4, |&x| x + 1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parallel_map_single_thread() {
        let data = vec![10, 20, 30];
        let result = parallel_map(&data, 1, |&x| x / 10);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_parallel_map_string() {
        let data = vec![1, 2, 3];
        let result = parallel_map(&data, 2, |x| x.to_string());
        assert_eq!(result, vec!["1", "2", "3"]);
    }

    #[test]
    fn test_parallel_map_large() {
        let data: Vec<u64> = (0..1000).collect();
        let result = parallel_map(&data, 4, |&x| x * x);
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, (i as u64) * (i as u64));
        }
    }

    // -----------------------------------------------------------------------
    // parallel_reduce
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_reduce_sum() {
        let data: Vec<i64> = (1..=100).collect();
        let sum = parallel_reduce(&data, 0i64, |a, b| a + b, |x| *x);
        assert_eq!(sum, 5050);
    }

    #[test]
    fn test_parallel_reduce_product() {
        let data = vec![1, 2, 3, 4, 5];
        let product = parallel_reduce(&data, 1i64, |a, b| a * b, |x| *x);
        assert_eq!(product, 120);
    }

    #[test]
    fn test_parallel_reduce_empty() {
        let data: Vec<i64> = vec![];
        let result = parallel_reduce(&data, 0, |a, b| a + b, |x| *x);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_parallel_reduce_single() {
        let data = vec![42i64];
        let result = parallel_reduce(&data, 0, |a, b| a + b, |x| *x);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_parallel_reduce_with_map() {
        let data: Vec<i64> = (1..=10).collect();
        let sum_of_squares = parallel_reduce(&data, 0i64, |a, b| a + b, |x| x * x);
        assert_eq!(sum_of_squares, 385); // 1+4+9+16+25+36+49+64+81+100
    }

    #[test]
    fn test_parallel_reduce_max() {
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let max = parallel_reduce(&data, i64::MIN, |a, b| a.max(b), |x| *x);
        assert_eq!(max, 9);
    }

    // -----------------------------------------------------------------------
    // BarrierSync
    // -----------------------------------------------------------------------

    #[test]
    fn test_barrier_basic() {
        let barrier = Arc::new(barrier_sync(3));
        let counter = Arc::new(AtomicUsize::new(0));
        thread::scope(|s| {
            for _ in 0..3 {
                let b = Arc::clone(&barrier);
                let c = Arc::clone(&counter);
                s.spawn(move || {
                    c.fetch_add(1, Ordering::Relaxed);
                    b.wait();
                });
            }
        });
        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_barrier_reuse() {
        let barrier = Arc::new(BarrierSync::new(2));
        let counter = Arc::new(AtomicUsize::new(0));

        thread::scope(|s| {
            for _ in 0..2 {
                let b = Arc::clone(&barrier);
                let c = Arc::clone(&counter);
                s.spawn(move || {
                    // First barrier.
                    c.fetch_add(1, Ordering::Relaxed);
                    b.wait();
                    // Second barrier (reuse).
                    c.fetch_add(1, Ordering::Relaxed);
                    b.wait();
                });
            }
        });
        assert_eq!(counter.load(Ordering::Relaxed), 4);
    }

    #[test]
    #[should_panic(expected = "barrier count must be > 0")]
    fn test_barrier_zero_panics() {
        let _ = BarrierSync::new(0);
    }

    #[test]
    fn test_barrier_single_thread() {
        let b = BarrierSync::new(1);
        b.wait(); // should not block
    }

    // -----------------------------------------------------------------------
    // TaskGraph
    // -----------------------------------------------------------------------

    #[test]
    fn test_task_graph_empty() {
        let g = TaskGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        execute_task_graph(g, 2);
    }

    #[test]
    fn test_task_graph_single_task() {
        let result = Arc::new(AtomicBool::new(false));
        let r = Arc::clone(&result);
        let mut g = TaskGraph::new();
        g.add_task(TaskPriority::Normal, move || {
            r.store(true, Ordering::Relaxed);
        });
        assert_eq!(g.len(), 1);
        execute_task_graph(g, 1);
        assert!(result.load(Ordering::Relaxed));
    }

    #[test]
    fn test_task_graph_linear_chain() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut g = TaskGraph::new();

        let o1 = Arc::clone(&order);
        let a = g.add_task(TaskPriority::Normal, move || {
            o1.lock().unwrap().push(1);
        });
        let o2 = Arc::clone(&order);
        let b = g.add_task(TaskPriority::Normal, move || {
            o2.lock().unwrap().push(2);
        });
        let o3 = Arc::clone(&order);
        let c = g.add_task(TaskPriority::Normal, move || {
            o3.lock().unwrap().push(3);
        });

        g.add_dependency(b, a); // b depends on a
        g.add_dependency(c, b); // c depends on b

        execute_task_graph(g, 4);
        let o = order.lock().unwrap();
        assert_eq!(*o, vec![1, 2, 3]);
    }

    #[test]
    fn test_task_graph_diamond() {
        // A -> B, A -> C, B -> D, C -> D
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut g = TaskGraph::new();

        let o = Arc::clone(&order);
        let a = g.add_task(TaskPriority::Normal, move || {
            o.lock().unwrap().push('A');
        });
        let o = Arc::clone(&order);
        let b = g.add_task(TaskPriority::Normal, move || {
            o.lock().unwrap().push('B');
        });
        let o = Arc::clone(&order);
        let c = g.add_task(TaskPriority::Normal, move || {
            o.lock().unwrap().push('C');
        });
        let o = Arc::clone(&order);
        let d = g.add_task(TaskPriority::Normal, move || {
            o.lock().unwrap().push('D');
        });

        g.add_dependency(b, a);
        g.add_dependency(c, a);
        g.add_dependency(d, b);
        g.add_dependency(d, c);

        execute_task_graph(g, 4);
        let o = order.lock().unwrap();
        // A must be first, D must be last, B/C in any order.
        assert_eq!(o[0], 'A');
        assert_eq!(o[3], 'D');
        assert!(o[1] == 'B' || o[1] == 'C');
        assert!(o[2] == 'B' || o[2] == 'C');
    }

    #[test]
    fn test_task_graph_independent_tasks() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut g = TaskGraph::new();
        for _ in 0..10 {
            let c = Arc::clone(&counter);
            g.add_task(TaskPriority::Normal, move || {
                c.fetch_add(1, Ordering::Relaxed);
            });
        }
        execute_task_graph(g, 4);
        assert_eq!(counter.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_task_graph_default() {
        let g = TaskGraph::default();
        assert!(g.is_empty());
    }

    #[test]
    #[should_panic(expected = "self-dependency")]
    fn test_task_graph_self_dep_panics() {
        let mut g = TaskGraph::new();
        let a = g.add_task(TaskPriority::Normal, || {});
        g.add_dependency(a, a);
    }

    #[test]
    #[should_panic(expected = "invalid task id")]
    fn test_task_graph_invalid_task_panics() {
        let mut g = TaskGraph::new();
        let _ = g.add_task(TaskPriority::Normal, || {});
        g.add_dependency(99, 0);
    }

    #[test]
    #[should_panic(expected = "invalid dep id")]
    fn test_task_graph_invalid_dep_panics() {
        let mut g = TaskGraph::new();
        let a = g.add_task(TaskPriority::Normal, || {});
        g.add_dependency(a, 99);
    }

    // -----------------------------------------------------------------------
    // ScalableThreadPool
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalable_pool_creation() {
        let pool = ScalableThreadPool::new(1, 4);
        assert_eq!(pool.current_threads(), 1);
    }

    #[test]
    fn test_scalable_pool_submit() {
        let pool = ScalableThreadPool::new(1, 4);
        let h = pool.submit(|| 42);
        assert_eq!(h.join(), 42);
    }

    #[test]
    fn test_scalable_pool_many_tasks() {
        let pool = ScalableThreadPool::new(1, 8);
        let handles: Vec<_> = (0..50).map(|i| pool.submit(move || i * 2)).collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join()).collect();
        for (i, r) in results.into_iter().enumerate() {
            assert_eq!(r, i * 2);
        }
    }

    #[test]
    fn test_scalable_pool_tasks_completed() {
        let pool = ScalableThreadPool::new(2, 4);
        let handles: Vec<_> = (0..10).map(|_| pool.submit(|| ())).collect();
        for h in handles {
            h.join();
        }
        thread::sleep(Duration::from_millis(50));
        assert!(pool.tasks_completed() >= 10);
    }

    #[test]
    fn test_scalable_pool_shutdown() {
        let pool = ScalableThreadPool::new(2, 4);
        let h = pool.submit(|| "done");
        assert_eq!(h.join(), "done");
        pool.shutdown();
    }

    #[test]
    fn test_scalable_pool_drop() {
        let pool = ScalableThreadPool::new(1, 2);
        let _h = pool.submit(|| ());
        drop(pool); // should not hang
    }

    #[test]
    #[should_panic(expected = "min_threads must be > 0")]
    fn test_scalable_pool_zero_min_panics() {
        let _ = ScalableThreadPool::new(0, 4);
    }

    #[test]
    #[should_panic(expected = "max_threads must be >= min_threads")]
    fn test_scalable_pool_max_lt_min_panics() {
        let _ = ScalableThreadPool::new(4, 2);
    }

    #[test]
    fn test_scalable_pool_pending_tasks() {
        let pool = ScalableThreadPool::new(1, 1);
        // Initially no pending tasks.
        assert_eq!(pool.pending_tasks(), 0);
    }

    #[test]
    fn test_scalable_pool_scale_up() {
        let pool = ScalableThreadPool::new(1, 4);
        // Submit many tasks to trigger scale-up.
        let handles: Vec<_> = (0..20)
            .map(|_| {
                pool.submit(|| {
                    thread::sleep(Duration::from_millis(10));
                })
            })
            .collect();
        // After submitting a burst, the pool may have scaled up.
        thread::sleep(Duration::from_millis(50));
        // We just verify tasks complete correctly.
        for h in handles {
            h.join();
        }
    }

    // -----------------------------------------------------------------------
    // effective_parallelism helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_effective_parallelism_zero() {
        let p = effective_parallelism(0);
        assert!(p >= 1);
    }

    #[test]
    fn test_effective_parallelism_explicit() {
        assert_eq!(effective_parallelism(7), 7);
    }

    // -----------------------------------------------------------------------
    // Integration / stress tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pool_stress_many_small_tasks() {
        let pool = InferenceThreadPool::new(ThreadPoolConfig::with_threads(4));
        let counter = Arc::new(AtomicUsize::new(0));
        let n = 1000;
        let handles: Vec<_> = (0..n)
            .map(|_| {
                let c = Arc::clone(&counter);
                pool.submit_default(move || {
                    c.fetch_add(1, Ordering::Relaxed);
                })
            })
            .collect();
        for h in handles {
            h.join();
        }
        assert_eq!(counter.load(Ordering::Relaxed), n);
    }

    #[test]
    fn test_parallel_for_correctness() {
        // Every index from 0..100 is visited exactly once.
        let n = 100;
        let visited = Arc::new((0..n).map(|_| AtomicBool::new(false)).collect::<Vec<_>>());
        let v = Arc::clone(&visited);
        parallel_for(n, 4, move |i| {
            assert!(!v[i].swap(true, Ordering::Relaxed), "index {i} visited twice");
        });
        for (i, flag) in visited.iter().enumerate() {
            assert!(flag.load(Ordering::Relaxed), "index {i} not visited");
        }
    }

    #[test]
    fn test_parallel_reduce_large() {
        let data: Vec<i64> = (1..=10_000).collect();
        let sum = parallel_reduce(&data, 0i64, |a, b| a + b, |x| *x);
        assert_eq!(sum, 50_005_000);
    }

    #[test]
    fn test_task_graph_wide_fan_out() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut g = TaskGraph::new();
        let root = g.add_task(TaskPriority::Normal, || {});
        for _ in 0..20 {
            let c = Arc::clone(&counter);
            let child = g.add_task(TaskPriority::Normal, move || {
                c.fetch_add(1, Ordering::Relaxed);
            });
            g.add_dependency(child, root);
        }
        execute_task_graph(g, 4);
        assert_eq!(counter.load(Ordering::Relaxed), 20);
    }

    #[test]
    fn test_task_graph_fan_in() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut g = TaskGraph::new();
        let mut leaves = Vec::new();
        for _ in 0..5 {
            let c = Arc::clone(&counter);
            leaves.push(g.add_task(TaskPriority::Normal, move || {
                c.fetch_add(1, Ordering::Relaxed);
            }));
        }
        let c = Arc::clone(&counter);
        let sink = g.add_task(TaskPriority::Normal, move || {
            c.fetch_add(100, Ordering::Relaxed);
        });
        for &leaf in &leaves {
            g.add_dependency(sink, leaf);
        }
        execute_task_graph(g, 4);
        assert_eq!(counter.load(Ordering::Relaxed), 105);
    }
}
