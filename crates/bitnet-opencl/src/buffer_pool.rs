//! Size-bucketed buffer pool for OpenCL device memory reuse.
//!
//! Allocating and deallocating GPU buffers on every inference call is
//! expensive.  `BufferPool` keeps returned buffers in power-of-two size
//! buckets so subsequent requests can be served from the pool without
//! hitting the OpenCL allocator.

use crate::buffer::{AccessMode, OpenClBuffer};
use crate::context::OpenClContext;
use crate::error::Result;
use std::collections::HashMap;
use std::sync::Mutex;
use tracing::debug;

/// Configuration for the buffer pool.
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// Maximum number of buffers to keep per size bucket.
    pub max_per_bucket: usize,
    /// Maximum total bytes retained across all buckets.
    pub max_total_bytes: usize,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            max_per_bucket: 8,
            max_total_bytes: 256 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BucketKey {
    elem_size: usize,
    capacity: usize,
}

struct PoolEntry<T: Copy + Send + 'static> {
    buffer: OpenClBuffer<T>,
    byte_size: usize,
}

/// Thread-safe buffer pool that reuses device allocations.
pub struct BufferPool {
    config: BufferPoolConfig,
    inner: Mutex<PoolInner>,
}

struct PoolInner {
    buckets: HashMap<BucketKey, Vec<Box<dyn std::any::Any + Send>>>,
    total_bytes: usize,
    stats: PoolStats,
}

/// Counters exposed for diagnostics.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub hits: u64,
    pub misses: u64,
    pub returns: u64,
    pub evictions: u64,
}

impl BufferPool {
    pub fn new(config: BufferPoolConfig) -> Self {
        Self {
            config,
            inner: Mutex::new(PoolInner {
                buckets: HashMap::new(),
                total_bytes: 0,
                stats: PoolStats::default(),
            }),
        }
    }

    pub fn acquire<T: Copy + Send + 'static>(
        &self,
        ctx: &OpenClContext,
        min_len: usize,
        mode: AccessMode,
    ) -> Result<OpenClBuffer<T>> {
        let rounded = next_power_of_two(min_len).max(1);
        let key = BucketKey {
            elem_size: std::mem::size_of::<T>(),
            capacity: rounded,
        };

        let mut inner =
            self.inner.lock().expect("buffer pool lock poisoned");

        if let Some(bucket) = inner.buckets.get_mut(&key) {
            if let Some(entry_any) = bucket.pop() {
                if let Ok(entry) =
                    entry_any.downcast::<PoolEntry<T>>()
                {
                    inner.total_bytes -= entry.byte_size;
                    inner.stats.hits += 1;
                    debug!(
                        "buffer pool hit: {}x{} ({} bytes)",
                        key.capacity, key.elem_size, entry.byte_size
                    );
                    return Ok(entry.buffer);
                }
            }
        }

        inner.stats.misses += 1;
        drop(inner);

        debug!(
            "buffer pool miss: allocating {}x{} ({} bytes)",
            rounded,
            std::mem::size_of::<T>(),
            rounded * std::mem::size_of::<T>()
        );
        OpenClBuffer::new(ctx, rounded, mode)
    }

    pub fn release<T: Copy + Send + 'static>(
        &self,
        buffer: OpenClBuffer<T>,
    ) {
        let byte_size = buffer.len * std::mem::size_of::<T>();
        let key = BucketKey {
            elem_size: std::mem::size_of::<T>(),
            capacity: buffer.len,
        };

        let mut inner =
            self.inner.lock().expect("buffer pool lock poisoned");
        inner.stats.returns += 1;

        if inner.total_bytes + byte_size > self.config.max_total_bytes {
            inner.stats.evictions += 1;
            debug!("buffer pool eviction: total would exceed budget");
            return;
        }

        let bucket_len = inner
            .buckets
            .entry(key)
            .or_default()
            .len();
        if bucket_len >= self.config.max_per_bucket {
            inner.stats.evictions += 1;
            debug!("buffer pool eviction: bucket full");
            return;
        }

        inner.total_bytes += byte_size;
        inner
            .buckets
            .get_mut(&key)
            .unwrap()
            .push(Box::new(PoolEntry { buffer, byte_size }));
    }

    pub fn stats(&self) -> PoolStats {
        self.inner
            .lock()
            .expect("buffer pool lock poisoned")
            .stats
            .clone()
    }

    pub fn clear(&self) {
        let mut inner =
            self.inner.lock().expect("buffer pool lock poisoned");
        inner.buckets.clear();
        inner.total_bytes = 0;
        inner.stats = PoolStats::default();
    }

    pub fn retained_bytes(&self) -> usize {
        self.inner
            .lock()
            .expect("buffer pool lock poisoned")
            .total_bytes
    }
}

fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    n.next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_of_two_rounding() {
        assert_eq!(next_power_of_two(0), 0);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(4), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(1023), 1024);
        assert_eq!(next_power_of_two(1024), 1024);
        assert_eq!(next_power_of_two(1025), 2048);
    }

    #[test]
    fn default_config_values() {
        let cfg = BufferPoolConfig::default();
        assert_eq!(cfg.max_per_bucket, 8);
        assert_eq!(cfg.max_total_bytes, 256 * 1024 * 1024);
    }

    #[test]
    fn pool_starts_empty() {
        let pool = BufferPool::new(BufferPoolConfig::default());
        assert_eq!(pool.retained_bytes(), 0);
        let stats = pool.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn clear_resets_state() {
        let pool = BufferPool::new(BufferPoolConfig::default());
        pool.clear();
        assert_eq!(pool.retained_bytes(), 0);
    }

    #[test]
    fn bucket_key_equality() {
        let a = BucketKey {
            elem_size: 4,
            capacity: 1024,
        };
        let b = BucketKey {
            elem_size: 4,
            capacity: 1024,
        };
        let c = BucketKey {
            elem_size: 1,
            capacity: 1024,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn pool_stats_clone() {
        let s = PoolStats {
            hits: 5,
            misses: 3,
            returns: 2,
            evictions: 1,
        };
        let s2 = s.clone();
        assert_eq!(s2.hits, 5);
        assert_eq!(s2.misses, 3);
    }

    #[test]
    fn acquire_release_cycle_with_hardware() {
        let ctx = crate::context::OpenClContext::new_intel();
        if let Ok(ctx) = ctx {
            let pool = BufferPool::new(BufferPoolConfig::default());

            let buf: OpenClBuffer<f32> = pool
                .acquire(&ctx, 100, AccessMode::ReadWrite)
                .expect("alloc");
            assert!(buf.len >= 100);
            assert_eq!(pool.stats().misses, 1);

            pool.release(buf);
            assert!(pool.retained_bytes() > 0);

            let _buf2: OpenClBuffer<f32> = pool
                .acquire(&ctx, 100, AccessMode::ReadWrite)
                .expect("alloc");
            assert_eq!(pool.stats().hits, 1);
        }
    }

    #[test]
    fn eviction_on_bucket_full() {
        let ctx = crate::context::OpenClContext::new_intel();
        if let Ok(ctx) = ctx {
            let cfg = BufferPoolConfig {
                max_per_bucket: 2,
                max_total_bytes: usize::MAX,
            };
            let pool = BufferPool::new(cfg);

            let b1: OpenClBuffer<f32> = pool
                .acquire(&ctx, 8, AccessMode::ReadWrite)
                .unwrap();
            let b2: OpenClBuffer<f32> = pool
                .acquire(&ctx, 8, AccessMode::ReadWrite)
                .unwrap();
            let b3: OpenClBuffer<f32> = pool
                .acquire(&ctx, 8, AccessMode::ReadWrite)
                .unwrap();
            pool.release(b1);
            pool.release(b2);
            pool.release(b3);
            assert_eq!(pool.stats().evictions, 1);
        }
    }

    #[test]
    fn eviction_on_byte_budget() {
        let ctx = crate::context::OpenClContext::new_intel();
        if let Ok(ctx) = ctx {
            let cfg = BufferPoolConfig {
                max_per_bucket: 100,
                max_total_bytes: 64,
            };
            let pool = BufferPool::new(cfg);

            let buf: OpenClBuffer<f32> = pool
                .acquire(&ctx, 128, AccessMode::ReadWrite)
                .unwrap();
            pool.release(buf);
            assert_eq!(pool.stats().evictions, 1);
            assert_eq!(pool.retained_bytes(), 0);
        }
    }
}