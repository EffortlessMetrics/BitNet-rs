//! Threading utilities for the C API
//!
//! This module provides thread safety guarantees, concurrent access support,
//! and thread pool management for the C API.

use crate::BitNetCError;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads
    pub num_threads: usize,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Thread stack size in bytes
    pub stack_size: Option<usize>,
    /// Thread name prefix
    pub thread_name_prefix: String,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            max_queue_size: 1000,
            stack_size: None,
            thread_name_prefix: "bitnet-worker".to_string(),
        }
    }
}

/// Thread pool for managing concurrent operations
pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: std::sync::mpsc::Sender<Job>,
    config: ThreadPoolConfig,
    active_jobs: Arc<AtomicUsize>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl ThreadPool {
    /// Create a new thread pool with default configuration
    pub fn new() -> Result<Self, BitNetCError> {
        Self::with_config(ThreadPoolConfig::default())
    }

    /// Create a new thread pool with custom configuration
    pub fn with_config(config: ThreadPoolConfig) -> Result<Self, BitNetCError> {
        let (sender, receiver) = std::sync::mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        let active_jobs = Arc::new(AtomicUsize::new(0));

        let mut workers = Vec::with_capacity(config.num_threads);

        for id in 0..config.num_threads {
            let worker = Worker::new(id, Arc::clone(&receiver), Arc::clone(&active_jobs), &config)?;
            workers.push(worker);
        }

        Ok(ThreadPool { workers, sender, config, active_jobs })
    }

    /// Execute a job on the thread pool
    pub fn execute<F>(&self, job: F) -> Result<(), BitNetCError>
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(job);

        self.sender.send(job).map_err(|_| {
            BitNetCError::ThreadSafety("Failed to send job to thread pool".to_string())
        })?;

        self.active_jobs.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    /// Get the number of active jobs
    pub fn active_job_count(&self) -> usize {
        self.active_jobs.load(Ordering::SeqCst)
    }

    /// Get the number of worker threads
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Wait for all active jobs to complete
    pub fn wait_for_completion(&self) -> Result<(), BitNetCError> {
        while self.active_job_count() > 0 {
            thread::sleep(std::time::Duration::from_millis(10));
        }
        Ok(())
    }

    /// Get thread pool statistics
    pub fn get_stats(&self) -> ThreadPoolStats {
        ThreadPoolStats {
            num_threads: self.workers.len(),
            active_jobs: self.active_job_count(),
            max_queue_size: self.config.max_queue_size,
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Wait for all jobs to complete
        let _ = self.wait_for_completion();

        // Workers will automatically stop when the sender is dropped
    }
}

/// Worker thread in the thread pool
struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new(
        id: usize,
        receiver: Arc<Mutex<std::sync::mpsc::Receiver<Job>>>,
        active_jobs: Arc<AtomicUsize>,
        config: &ThreadPoolConfig,
    ) -> Result<Self, BitNetCError> {
        let mut builder =
            thread::Builder::new().name(format!("{}-{}", config.thread_name_prefix, id));

        if let Some(stack_size) = config.stack_size {
            builder = builder.stack_size(stack_size);
        }

        let thread = builder
            .spawn(move || {
                loop {
                    let job = match receiver.lock() {
                        Ok(receiver) => {
                            match receiver.recv() {
                                Ok(job) => job,
                                Err(_) => break, // Channel closed
                            }
                        }
                        Err(_) => break, // Mutex poisoned
                    };

                    job();
                    active_jobs.fetch_sub(1, Ordering::SeqCst);
                }
            })
            .map_err(|e| {
                BitNetCError::ThreadSafety(format!("Failed to spawn worker thread: {}", e))
            })?;

        Ok(Worker { id, thread: Some(thread) })
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Thread pool statistics
#[derive(Debug, Clone)]
pub struct ThreadPoolStats {
    pub num_threads: usize,
    pub active_jobs: usize,
    pub max_queue_size: usize,
}

/// Thread-safe reference counter for tracking object lifetimes
pub struct ThreadSafeRefCounter<T> {
    data: Arc<RwLock<T>>,
    ref_count: Arc<AtomicUsize>,
}

impl<T> ThreadSafeRefCounter<T> {
    pub fn new(data: T) -> Self {
        Self { data: Arc::new(RwLock::new(data)), ref_count: Arc::new(AtomicUsize::new(1)) }
    }

    pub fn clone_ref(&self) -> Self {
        self.ref_count.fetch_add(1, Ordering::SeqCst);
        Self { data: Arc::clone(&self.data), ref_count: Arc::clone(&self.ref_count) }
    }

    pub fn read(&self) -> Result<std::sync::RwLockReadGuard<T>, BitNetCError> {
        self.data
            .read()
            .map_err(|_| BitNetCError::ThreadSafety("Failed to acquire read lock".to_string()))
    }

    pub fn write(&self) -> Result<std::sync::RwLockWriteGuard<T>, BitNetCError> {
        self.data
            .write()
            .map_err(|_| BitNetCError::ThreadSafety("Failed to acquire write lock".to_string()))
    }

    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::SeqCst)
    }
}

impl<T> Drop for ThreadSafeRefCounter<T> {
    fn drop(&mut self) {
        self.ref_count.fetch_sub(1, Ordering::SeqCst);
    }
}

// Thread-local storage for C API state
thread_local! {
    static THREAD_LOCAL_STATE: std::cell::RefCell<HashMap<String, Box<dyn std::any::Any>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Set thread-local value
pub fn set_thread_local<T: 'static>(key: &str, value: T) {
    THREAD_LOCAL_STATE.with(|state| {
        state.borrow_mut().insert(key.to_string(), Box::new(value));
    });
}

/// Get thread-local value
pub fn get_thread_local<T: 'static + Clone>(key: &str) -> Option<T> {
    THREAD_LOCAL_STATE
        .with(|state| state.borrow().get(key).and_then(|any| any.downcast_ref::<T>()).cloned())
}

/// Clear thread-local storage
pub fn clear_thread_local() {
    THREAD_LOCAL_STATE.with(|state| {
        state.borrow_mut().clear();
    });
}

/// Global thread pool manager
pub struct ThreadManager {
    thread_pool: RwLock<Option<ThreadPool>>,
    num_threads: AtomicUsize,
}

impl ThreadManager {
    pub fn new() -> Self {
        Self { thread_pool: RwLock::new(None), num_threads: AtomicUsize::new(num_cpus::get()) }
    }

    /// Initialize the thread pool
    pub fn initialize(&self) -> Result<(), BitNetCError> {
        let mut pool = self.thread_pool.write().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire thread pool write lock".to_string())
        })?;

        if pool.is_none() {
            let config = ThreadPoolConfig {
                num_threads: self.num_threads.load(Ordering::SeqCst),
                ..ThreadPoolConfig::default()
            };
            *pool = Some(ThreadPool::with_config(config)?);
        }

        Ok(())
    }

    /// Set the number of threads
    pub fn set_num_threads(&self, num_threads: usize) -> Result<(), BitNetCError> {
        if num_threads == 0 {
            return Err(BitNetCError::InvalidArgument(
                "num_threads must be greater than 0".to_string(),
            ));
        }

        self.num_threads.store(num_threads, Ordering::SeqCst);

        // Recreate thread pool with new thread count
        let mut pool = self.thread_pool.write().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire thread pool write lock".to_string())
        })?;

        let config = ThreadPoolConfig { num_threads, ..ThreadPoolConfig::default() };
        *pool = Some(ThreadPool::with_config(config)?);

        Ok(())
    }

    /// Get the current number of threads
    pub fn get_num_threads(&self) -> usize {
        self.num_threads.load(Ordering::SeqCst)
    }

    /// Execute a job on the thread pool
    pub fn execute<F>(&self, job: F) -> Result<(), BitNetCError>
    where
        F: FnOnce() + Send + 'static,
    {
        let pool = self.thread_pool.read().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire thread pool read lock".to_string())
        })?;

        match pool.as_ref() {
            Some(pool) => pool.execute(job),
            None => Err(BitNetCError::ThreadSafety("Thread pool not initialized".to_string())),
        }
    }

    /// Get thread pool statistics
    pub fn get_stats(&self) -> Result<ThreadPoolStats, BitNetCError> {
        let pool = self.thread_pool.read().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire thread pool read lock".to_string())
        })?;

        match pool.as_ref() {
            Some(pool) => Ok(pool.get_stats()),
            None => Err(BitNetCError::ThreadSafety("Thread pool not initialized".to_string())),
        }
    }

    /// Wait for all active jobs to complete
    pub fn wait_for_completion(&self) -> Result<(), BitNetCError> {
        let pool = self.thread_pool.read().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire thread pool read lock".to_string())
        })?;

        match pool.as_ref() {
            Some(pool) => pool.wait_for_completion(),
            None => Ok(()), // No pool means no jobs
        }
    }

    /// Cleanup the thread pool
    pub fn cleanup(&self) -> Result<(), BitNetCError> {
        let mut pool = self.thread_pool.write().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire thread pool write lock".to_string())
        })?;

        if let Some(pool_instance) = pool.take() {
            pool_instance.wait_for_completion()?;
            // Pool will be dropped here, cleaning up all threads
        }

        Ok(())
    }
}

// Global thread manager instance
static THREAD_MANAGER: std::sync::OnceLock<ThreadManager> = std::sync::OnceLock::new();

/// Initialize the thread pool
pub fn initialize_thread_pool() -> Result<(), BitNetCError> {
    let manager = THREAD_MANAGER.get_or_init(|| ThreadManager::new());
    manager.initialize()
}

/// Get the global thread manager instance
pub fn get_thread_manager() -> &'static ThreadManager {
    THREAD_MANAGER.get_or_init(|| ThreadManager::new())
}

/// Set the number of threads
pub fn set_num_threads(num_threads: usize) -> Result<(), BitNetCError> {
    get_thread_manager().set_num_threads(num_threads)
}

/// Get the current number of threads
pub fn get_num_threads() -> usize {
    get_thread_manager().get_num_threads()
}

/// Execute a job on the thread pool
pub fn execute<F>(job: F) -> Result<(), BitNetCError>
where
    F: FnOnce() + Send + 'static,
{
    get_thread_manager().execute(job)
}

/// Cleanup the thread pool
pub fn cleanup_thread_pool() -> Result<(), BitNetCError> {
    if let Some(manager) = THREAD_MANAGER.get() {
        manager.cleanup()
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPool::new().unwrap();
        assert!(pool.worker_count() > 0);
        assert_eq!(pool.active_job_count(), 0);
    }

    #[test]
    fn test_thread_pool_execution() {
        let pool = ThreadPool::new().unwrap();
        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = Arc::clone(&executed);

        pool.execute(move || {
            executed_clone.store(true, Ordering::SeqCst);
        })
        .unwrap();

        // Wait a bit for the job to execute
        thread::sleep(Duration::from_millis(100));
        assert!(executed.load(Ordering::SeqCst));
    }

    #[test]
    fn test_thread_safe_ref_counter() {
        let counter = ThreadSafeRefCounter::new(42);
        assert_eq!(counter.ref_count(), 1);

        let cloned = counter.clone_ref();
        assert_eq!(counter.ref_count(), 2);
        assert_eq!(cloned.ref_count(), 2);

        {
            let data = counter.read().unwrap();
            assert_eq!(*data, 42);
        }

        drop(cloned);
        assert_eq!(counter.ref_count(), 1);
    }

    #[test]
    fn test_thread_local_storage() {
        set_thread_local("test_key", 123);
        assert_eq!(get_thread_local::<i32>("test_key"), Some(123));
        assert_eq!(get_thread_local::<i32>("nonexistent"), None);

        clear_thread_local();
        assert_eq!(get_thread_local::<i32>("test_key"), None);
    }

    #[test]
    fn test_thread_manager() {
        let manager = ThreadManager::new();
        assert!(manager.initialize().is_ok());

        let _original_threads = manager.get_num_threads();
        assert!(manager.set_num_threads(4).is_ok());
        assert_eq!(manager.get_num_threads(), 4);

        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = Arc::clone(&executed);

        assert!(manager
            .execute(move || {
                executed_clone.store(true, Ordering::SeqCst);
            })
            .is_ok());

        thread::sleep(Duration::from_millis(100));
        assert!(executed.load(Ordering::SeqCst));

        assert!(manager.cleanup().is_ok());
    }

    #[test]
    fn test_thread_pool_config() {
        let config = ThreadPoolConfig {
            num_threads: 2,
            max_queue_size: 100,
            stack_size: Some(1024 * 1024),
            thread_name_prefix: "test-worker".to_string(),
        };

        let pool = ThreadPool::with_config(config).unwrap();
        assert_eq!(pool.worker_count(), 2);
    }
}
