//! Memory pool allocator for efficient tensor allocation.
//!
//! Provides a thread-safe, size-bucketed pool that recycles byte buffers.
//! Allocations are rounded up to the nearest power-of-two *size class*,
//! which limits internal fragmentation while maximising reuse.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

// ── Statistics ───────────────────────────────────────────────────────

/// Cumulative statistics for a [`TensorPool`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Number of times `allocate` returned an existing buffer.
    pub hits: u64,
    /// Number of times `allocate` had to create a new buffer.
    pub misses: u64,
    /// Bytes currently held *inside the pool* (not lent out).
    pub pooled_bytes: usize,
    /// Bytes currently lent out via live `PooledBuffer` handles.
    pub active_bytes: usize,
}

impl PoolStats {
    /// Total `allocate` calls (`hits + misses`).
    pub fn total_allocations(&self) -> u64 {
        self.hits + self.misses
    }
}

// ── Pool internals ──────────────────────────────────────────────────

/// Per-bucket free list.
struct Bucket {
    buffers: Vec<Vec<u8>>,
}

struct PoolInner {
    max_size_bytes: usize,
    buckets: HashMap<usize, Bucket>,
    stats: PoolStats,
}

// ── TensorPool ──────────────────────────────────────────────────────

/// Thread-safe, size-bucketed memory pool for tensor scratch buffers.
///
/// Buffers are rounded up to the nearest power-of-two size class. When a
/// [`PooledBuffer`] is dropped, its backing memory is returned to the
/// appropriate bucket — unless the pool has already reached `max_size_bytes`
/// of pooled memory, in which case the buffer is freed immediately.
#[derive(Clone)]
pub struct TensorPool {
    inner: Arc<Mutex<PoolInner>>,
}

impl TensorPool {
    /// Create a new pool that will cache at most `max_size_bytes` of idle
    /// memory.
    pub fn new(max_size_bytes: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PoolInner {
                max_size_bytes,
                buckets: HashMap::new(),
                stats: PoolStats::default(),
            })),
        }
    }

    /// Allocate a zeroed buffer of *at least* `size` bytes.
    ///
    /// The returned buffer's length equals the rounded-up size class (a
    /// power of two, minimum 64).
    pub fn allocate(&self, size: usize) -> PooledBuffer {
        let bucket_size = bucket_for(size);
        let mut inner = self.inner.lock().expect("pool lock poisoned");

        let recycled = inner.buckets.get_mut(&bucket_size).and_then(|b| b.buffers.pop());

        let buf = match recycled {
            Some(mut v) => {
                inner.stats.hits += 1;
                inner.stats.pooled_bytes -= bucket_size;
                v.iter_mut().for_each(|b| *b = 0);
                v
            }
            None => {
                inner.stats.misses += 1;
                vec![0u8; bucket_size]
            }
        };

        inner.stats.active_bytes += bucket_size;

        PooledBuffer { buf: Some(buf), pool: Arc::clone(&self.inner) }
    }

    /// Snapshot of current pool statistics.
    pub fn stats(&self) -> PoolStats {
        self.inner.lock().expect("pool lock poisoned").stats.clone()
    }

    /// Drop all cached buffers, freeing idle memory immediately.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("pool lock poisoned");
        inner.buckets.clear();
        inner.stats.pooled_bytes = 0;
    }
}

/// Round `size` up to the next power of two, with a minimum of 64 bytes.
fn bucket_for(size: usize) -> usize {
    let min = 64;
    if size <= min {
        return min;
    }
    size.next_power_of_two()
}

// ── PooledBuffer ────────────────────────────────────────────────────

/// RAII byte buffer that returns its memory to a [`TensorPool`] on drop.
pub struct PooledBuffer {
    buf: Option<Vec<u8>>,
    pool: Arc<Mutex<PoolInner>>,
}

impl PooledBuffer {
    /// Reinterpret the buffer as a slice of `f32` values.
    ///
    /// # Panics
    /// Panics if the buffer length is not a multiple of 4.
    pub fn as_f32_slice(&self) -> &[f32] {
        let bytes = self.deref();
        assert!(
            bytes.len().is_multiple_of(std::mem::size_of::<f32>()),
            "buffer length {} is not a multiple of 4",
            bytes.len()
        );
        bytemuck::cast_slice(bytes)
    }

    /// Reinterpret the buffer as a mutable slice of `f32` values.
    ///
    /// # Panics
    /// Panics if the buffer length is not a multiple of 4.
    pub fn as_f32_mut_slice(&mut self) -> &mut [f32] {
        let bytes = self.deref_mut();
        assert!(
            bytes.len().is_multiple_of(std::mem::size_of::<f32>()),
            "buffer length {} is not a multiple of 4",
            bytes.len()
        );
        bytemuck::cast_slice_mut(bytes)
    }
}

impl Deref for PooledBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.buf.as_ref().expect("buffer already returned")
    }
}

impl DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.buf.as_mut().expect("buffer already returned")
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            let bucket_size = buf.len();
            let mut inner = self.pool.lock().expect("pool lock poisoned");
            inner.stats.active_bytes = inner.stats.active_bytes.saturating_sub(bucket_size);

            if inner.stats.pooled_bytes + bucket_size <= inner.max_size_bytes {
                inner.stats.pooled_bytes += bucket_size;
                inner
                    .buckets
                    .entry(bucket_size)
                    .or_insert_with(|| Bucket { buffers: Vec::new() })
                    .buffers
                    .push(buf);
            }
            // else: drop `buf`, freeing the memory
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn basic_alloc_dealloc() {
        let pool = TensorPool::new(4096);
        let buf = pool.allocate(100);
        assert!(buf.len() >= 100);
        drop(buf);
        let stats = pool.stats();
        assert_eq!(stats.total_allocations(), 1);
    }

    #[test]
    fn buffer_is_zeroed() {
        let pool = TensorPool::new(4096);
        let mut buf = pool.allocate(64);
        // Write non-zero data.
        buf[0] = 0xFF;
        drop(buf);

        // Re-allocate: recycled buffer must be zeroed.
        let buf2 = pool.allocate(64);
        assert!(buf2.iter().all(|&b| b == 0));
    }

    #[test]
    fn size_bucketing_rounds_up() {
        let pool = TensorPool::new(4096);
        // Request 100 bytes → bucket 128.
        let buf = pool.allocate(100);
        assert_eq!(buf.len(), 128);
    }

    #[test]
    fn minimum_bucket_is_64() {
        let pool = TensorPool::new(4096);
        let buf = pool.allocate(1);
        assert_eq!(buf.len(), 64);
    }

    #[test]
    fn exact_power_of_two_no_rounding() {
        let pool = TensorPool::new(4096);
        let buf = pool.allocate(256);
        assert_eq!(buf.len(), 256);
    }

    #[test]
    fn pool_reuse_hit() {
        let pool = TensorPool::new(4096);
        let buf = pool.allocate(128);
        drop(buf);

        let _buf2 = pool.allocate(128);
        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn different_sizes_are_separate_buckets() {
        let pool = TensorPool::new(8192);
        let a = pool.allocate(64);
        let b = pool.allocate(256);
        drop(a);
        drop(b);

        // Requesting 256 should reuse the 256 bucket, not the 64 one.
        let _c = pool.allocate(256);
        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn stats_accuracy() {
        let pool = TensorPool::new(4096);
        {
            let _a = pool.allocate(64);
            let _b = pool.allocate(128);
            let stats = pool.stats();
            assert_eq!(stats.misses, 2);
            assert_eq!(stats.active_bytes, 64 + 128);
            assert_eq!(stats.pooled_bytes, 0);
        }
        // Both dropped — they should be pooled now.
        let stats = pool.stats();
        assert_eq!(stats.active_bytes, 0);
        assert_eq!(stats.pooled_bytes, 64 + 128);
    }

    #[test]
    fn max_capacity_enforcement() {
        // Pool can hold at most 128 bytes.
        let pool = TensorPool::new(128);
        let a = pool.allocate(128);
        let b = pool.allocate(128);
        drop(a);
        drop(b);

        // Only one of the two 128-byte buffers fits.
        let stats = pool.stats();
        assert_eq!(stats.pooled_bytes, 128);

        // Allocating again: one hit (pooled), one miss.
        let _c = pool.allocate(128);
        let _d = pool.allocate(128);
        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 3);
    }

    #[test]
    fn clear_frees_pooled_memory() {
        let pool = TensorPool::new(4096);
        let buf = pool.allocate(128);
        drop(buf);
        assert_eq!(pool.stats().pooled_bytes, 128);

        pool.clear();
        assert_eq!(pool.stats().pooled_bytes, 0);

        // Next allocation is a miss (nothing cached).
        let _buf2 = pool.allocate(128);
        let stats = pool.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn drop_returns_to_pool() {
        let pool = TensorPool::new(4096);
        let buf = pool.allocate(64);
        assert_eq!(pool.stats().pooled_bytes, 0);
        drop(buf);
        assert_eq!(pool.stats().pooled_bytes, 64);
    }

    #[test]
    fn thread_safety() {
        let pool = TensorPool::new(1 << 20);
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let p = pool.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        let mut buf = p.allocate(256);
                        buf[0] = 42;
                        // drop returns to pool
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        let stats = pool.stats();
        assert_eq!(stats.total_allocations(), 800);
        assert_eq!(stats.active_bytes, 0);
    }

    #[test]
    fn as_f32_slice_roundtrip() {
        let pool = TensorPool::new(4096);
        let mut buf = pool.allocate(16); // bucket 64 → 16 f32s
        let floats = buf.as_f32_mut_slice();
        floats[0] = 1.0;
        floats[1] = -2.5;

        let read = buf.as_f32_slice();
        assert_eq!(read[0], 1.0);
        assert_eq!(read[1], -2.5);
    }

    #[test]
    #[should_panic(expected = "not a multiple of 4")]
    fn as_f32_slice_bad_alignment() {
        let pool = TensorPool::new(4096);
        // Bucket 64, but we fabricate a bad-length buffer via the public API:
        // Actually bucket 64 is fine (64 % 4 == 0), so we test indirectly.
        // This test verifies the assertion message by using an impossible
        // scenario — we construct a PooledBuffer manually for testing.
        let inner = Arc::new(Mutex::new(PoolInner {
            max_size_bytes: 4096,
            buckets: HashMap::new(),
            stats: PoolStats::default(),
        }));
        let pb = PooledBuffer { buf: Some(vec![0u8; 3]), pool: inner };
        let _ = pb.as_f32_slice();
    }

    #[test]
    fn zero_size_allocation() {
        let pool = TensorPool::new(4096);
        let buf = pool.allocate(0);
        // Minimum bucket is 64.
        assert_eq!(buf.len(), 64);
    }

    #[test]
    fn multiple_reuse_cycles() {
        let pool = TensorPool::new(4096);
        for _ in 0..10 {
            let buf = pool.allocate(128);
            drop(buf);
        }
        let stats = pool.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 9);
    }

    #[test]
    fn clone_shares_pool() {
        let pool = TensorPool::new(4096);
        let pool2 = pool.clone();
        let buf = pool.allocate(128);
        drop(buf);

        // Reuse from cloned handle.
        let _buf2 = pool2.allocate(128);
        assert_eq!(pool.stats().hits, 1);
    }
}
