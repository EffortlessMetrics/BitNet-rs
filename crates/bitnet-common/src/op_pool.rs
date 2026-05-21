//! Operation pool for reusing computation buffers.
//!
//! Reduces allocation overhead during inference by pooling intermediate buffers.

use std::collections::HashMap;

/// Buffer key (shape + dtype).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferKey {
    pub shape: Vec<usize>,
    pub dtype: String,
}

impl BufferKey {
    pub fn new(shape: &[usize], dtype: &str) -> Self {
        Self { shape: shape.to_vec(), dtype: dtype.to_string() }
    }

    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A pooled buffer.
#[derive(Debug)]
pub struct PooledBuffer {
    pub key: BufferKey,
    pub data: Vec<u8>,
    pub in_use: bool,
}

/// Operation buffer pool.
#[derive(Debug)]
pub struct OpPool {
    buffers: HashMap<BufferKey, Vec<PooledBuffer>>,
    total_allocated: usize,
    max_bytes: usize,
    hits: u64,
    misses: u64,
}

impl OpPool {
    pub fn new(max_bytes: usize) -> Self {
        Self { buffers: HashMap::new(), total_allocated: 0, max_bytes, hits: 0, misses: 0 }
    }

    /// Acquire a buffer with the given shape and dtype.
    pub fn acquire(&mut self, shape: &[usize], dtype: &str, bytes_per_element: usize) -> Vec<u8> {
        let key = BufferKey::new(shape, dtype);
        let needed = self.needed_bytes(&key, bytes_per_element);

        if let Some(buffer) = self.acquire_existing_buffer(&key) {
            self.hits += 1;
            return buffer;
        }

        self.misses += 1;
        self.ensure_capacity(needed);
        self.allocate_new_buffer(key, shape, dtype, needed)
    }

    /// Release a buffer back to the pool.
    pub fn release(&mut self, shape: &[usize], dtype: &str) {
        let key = BufferKey::new(shape, dtype);
        if let Some(buffers) = self.buffers.get_mut(&key)
            && let Some(buf) = buffers.iter_mut().find(|b| b.in_use)
        {
            buf.in_use = false;
        }
    }

    /// Evict unused buffers.
    fn evict_lru(&mut self) {
        for buffers in self.buffers.values_mut() {
            buffers.retain(|b| {
                if !b.in_use {
                    self.total_allocated = self.total_allocated.saturating_sub(b.data.len());
                    false
                } else {
                    true
                }
            });
        }
    }

    fn needed_bytes(&self, key: &BufferKey, bytes_per_element: usize) -> usize {
        key.elements() * bytes_per_element
    }

    fn acquire_existing_buffer(&mut self, key: &BufferKey) -> Option<Vec<u8>> {
        let buffers = self.buffers.get_mut(key)?;
        let buf = buffers.iter_mut().find(|b| !b.in_use)?;
        buf.in_use = true;
        Some(buf.data.clone())
    }

    fn ensure_capacity(&mut self, needed: usize) {
        if self.total_allocated + needed > self.max_bytes {
            self.evict_lru();
        }
    }

    fn allocate_new_buffer(
        &mut self,
        key: BufferKey,
        shape: &[usize],
        dtype: &str,
        needed: usize,
    ) -> Vec<u8> {
        let data = vec![0u8; needed];
        self.total_allocated += needed;
        let entry = self.buffers.entry(key).or_default();
        entry.push(PooledBuffer {
            key: BufferKey::new(shape, dtype),
            data: data.clone(),
            in_use: true,
        });
        data
    }

    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    pub fn stats(&self) -> PoolStats {
        let mut total_buffers = 0;
        let mut in_use = 0;
        for buffers in self.buffers.values() {
            total_buffers += buffers.len();
            in_use += buffers.iter().filter(|b| b.in_use).count();
        }
        PoolStats {
            total_buffers,
            in_use,
            available: total_buffers - in_use,
            total_bytes: self.total_allocated,
            hit_rate: self.hit_rate(),
        }
    }

    /// Clear all buffers.
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.total_allocated = 0;
    }
}

/// Pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_buffers: usize,
    pub in_use: usize,
    pub available: usize,
    pub total_bytes: usize,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acquire_new() {
        let mut pool = OpPool::new(1024 * 1024);
        let buf = pool.acquire(&[4, 4], "f32", 4);
        assert_eq!(buf.len(), 64); // 4*4*4
    }

    #[test]
    fn test_release_reuse() {
        let mut pool = OpPool::new(1024 * 1024);
        pool.acquire(&[4, 4], "f32", 4);
        pool.release(&[4, 4], "f32");
        let _ = pool.acquire(&[4, 4], "f32", 4);
        assert!(pool.hit_rate() > 0.0);
    }

    #[test]
    fn test_hit_rate() {
        let mut pool = OpPool::new(1024 * 1024);
        pool.acquire(&[2, 2], "f32", 4);
        pool.release(&[2, 2], "f32");
        pool.acquire(&[2, 2], "f32", 4); // hit
        assert!((pool.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_stats() {
        let mut pool = OpPool::new(1024 * 1024);
        pool.acquire(&[2, 2], "f32", 4);
        let stats = pool.stats();
        assert_eq!(stats.total_buffers, 1);
        assert_eq!(stats.in_use, 1);
    }

    #[test]
    fn test_stats_after_release() {
        let mut pool = OpPool::new(1024 * 1024);
        pool.acquire(&[2, 2], "f32", 4);
        pool.release(&[2, 2], "f32");
        let stats = pool.stats();
        assert_eq!(stats.available, 1);
    }

    #[test]
    fn test_clear() {
        let mut pool = OpPool::new(1024 * 1024);
        pool.acquire(&[4, 4], "f32", 4);
        pool.clear();
        assert_eq!(pool.total_allocated(), 0);
    }

    #[test]
    fn test_buffer_key() {
        let k1 = BufferKey::new(&[3, 4], "f32");
        let k2 = BufferKey::new(&[3, 4], "f32");
        assert_eq!(k1, k2);
        assert_eq!(k1.elements(), 12);
    }

    #[test]
    fn test_different_shapes() {
        let mut pool = OpPool::new(1024 * 1024);
        pool.acquire(&[2, 2], "f32", 4);
        pool.acquire(&[3, 3], "f32", 4);
        let stats = pool.stats();
        assert_eq!(stats.total_buffers, 2);
    }

    #[test]
    fn test_eviction_on_limit() {
        let mut pool = OpPool::new(100);
        pool.acquire(&[5], "f32", 4); // 20 bytes
        pool.release(&[5], "f32");
        pool.acquire(&[20], "f32", 4); // 80 bytes, triggers evict
        assert!(pool.total_allocated() <= 100);
    }

    #[test]
    fn test_empty_hit_rate() {
        let pool = OpPool::new(1024);
        assert_eq!(pool.hit_rate(), 0.0);
    }

    #[test]
    fn test_total_allocated() {
        let mut pool = OpPool::new(1024 * 1024);
        pool.acquire(&[10], "f32", 4); // 40 bytes
        assert_eq!(pool.total_allocated(), 40);
    }

    #[test]
    fn test_different_dtypes() {
        let mut pool = OpPool::new(1024 * 1024);
        pool.acquire(&[4], "f32", 4);
        pool.acquire(&[4], "f16", 2);
        assert_eq!(pool.stats().total_buffers, 2);
    }
}
