//! Tensor buffer pooling to reduce allocation churn.
//!
//! Reuse pre-allocated f32 buffers during inference to avoid
//! per-token allocation overhead on CPU.

/// A pooled buffer that can be checked out and returned.
#[derive(Debug)]
pub struct PooledBuffer {
    data: Vec<f32>,
    capacity: usize,
}

impl PooledBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { data: vec![0.0; capacity], capacity }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Zero out the buffer for reuse.
    pub fn clear(&mut self) {
        self.data.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Resize if needed (may reallocate).
    pub fn ensure_capacity(&mut self, min_cap: usize) {
        if self.capacity < min_cap {
            self.data.resize(min_cap, 0.0);
            self.capacity = min_cap;
        }
    }
}

/// Pool of reusable buffers, organized by size class.
#[derive(Debug)]
pub struct BufferPool {
    free_buffers: Vec<PooledBuffer>,
    active_count: usize,
    total_allocated: usize,
    max_pool_size: usize,
}

impl BufferPool {
    pub fn new(max_pool_size: usize) -> Self {
        Self { free_buffers: Vec::new(), active_count: 0, total_allocated: 0, max_pool_size }
    }

    /// Get a buffer of at least `min_size` elements.
    pub fn acquire(&mut self, min_size: usize) -> PooledBuffer {
        // Try to find a suitable buffer in the pool
        if let Some(idx) = self.free_buffers.iter().position(|b| b.capacity >= min_size) {
            let mut buf = self.free_buffers.swap_remove(idx);
            buf.clear();
            self.active_count += 1;
            return buf;
        }

        // Allocate new
        self.total_allocated += 1;
        self.active_count += 1;
        PooledBuffer::new(min_size)
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&mut self, buf: PooledBuffer) {
        self.active_count = self.active_count.saturating_sub(1);
        if self.free_buffers.len() < self.max_pool_size {
            self.free_buffers.push(buf);
        }
        // Otherwise drop the buffer
    }

    /// Number of buffers currently checked out.
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Number of buffers available in the pool.
    pub fn free_count(&self) -> usize {
        self.free_buffers.len()
    }

    /// Total buffers allocated over lifetime.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Total memory held in pool (bytes, f32 = 4 bytes each).
    pub fn pool_memory_bytes(&self) -> usize {
        self.free_buffers.iter().map(|b| b.capacity * 4).sum()
    }

    /// Clear all pooled buffers.
    pub fn drain(&mut self) {
        self.free_buffers.clear();
    }

    /// Pool utilization: active / (active + free).
    pub fn utilization(&self) -> f64 {
        let total = self.active_count + self.free_buffers.len();
        if total == 0 {
            return 0.0;
        }
        self.active_count as f64 / total as f64
    }
}

/// Pre-configured pool sizes for common inference scenarios.
pub fn inference_pool() -> BufferPool {
    BufferPool::new(32) // Keep up to 32 reusable buffers
}

pub fn small_pool() -> BufferPool {
    BufferPool::new(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pooled_buffer_new() {
        let buf = PooledBuffer::new(1024);
        assert_eq!(buf.len(), 1024);
        assert_eq!(buf.capacity(), 1024);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_pooled_buffer_clear() {
        let mut buf = PooledBuffer::new(4);
        buf.as_mut_slice()[0] = 42.0;
        buf.clear();
        assert_eq!(buf.as_slice()[0], 0.0);
    }

    #[test]
    fn test_ensure_capacity() {
        let mut buf = PooledBuffer::new(4);
        buf.ensure_capacity(8);
        assert!(buf.capacity() >= 8);
    }

    #[test]
    fn test_pool_acquire_release() {
        let mut pool = BufferPool::new(4);
        let buf = pool.acquire(1024);
        assert_eq!(pool.active_count(), 1);
        assert_eq!(pool.free_count(), 0);
        pool.release(buf);
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn test_pool_reuse() {
        let mut pool = BufferPool::new(4);
        let buf = pool.acquire(1024);
        pool.release(buf);
        let buf2 = pool.acquire(512); // Should reuse the 1024 buffer
        assert!(buf2.capacity() >= 512);
        assert_eq!(pool.total_allocated(), 1); // Only 1 allocation
        pool.release(buf2);
    }

    #[test]
    fn test_pool_max_size() {
        let mut pool = BufferPool::new(2);
        let b1 = pool.acquire(100);
        let b2 = pool.acquire(100);
        let b3 = pool.acquire(100);
        pool.release(b1);
        pool.release(b2);
        pool.release(b3); // This one should be dropped (max=2)
        assert_eq!(pool.free_count(), 2);
    }

    #[test]
    fn test_pool_memory() {
        let mut pool = BufferPool::new(4);
        let buf = pool.acquire(1024);
        pool.release(buf);
        assert_eq!(pool.pool_memory_bytes(), 1024 * 4);
    }

    #[test]
    fn test_pool_drain() {
        let mut pool = BufferPool::new(4);
        let buf = pool.acquire(100);
        pool.release(buf);
        pool.drain();
        assert_eq!(pool.free_count(), 0);
    }

    #[test]
    fn test_utilization() {
        let mut pool = BufferPool::new(4);
        assert_eq!(pool.utilization(), 0.0);
        let buf = pool.acquire(100);
        assert_eq!(pool.utilization(), 1.0);
        pool.release(buf);
        assert_eq!(pool.utilization(), 0.0);
    }

    #[test]
    fn test_inference_pool() {
        let pool = inference_pool();
        assert_eq!(pool.free_count(), 0);
    }

    #[test]
    fn test_small_pool() {
        let mut pool = small_pool();
        let buf = pool.acquire(64);
        pool.release(buf);
        assert_eq!(pool.free_count(), 1);
    }

    #[test]
    fn test_multiple_acquires() {
        let mut pool = BufferPool::new(10);
        let b1 = pool.acquire(100);
        let b2 = pool.acquire(200);
        assert_eq!(pool.active_count(), 2);
        assert_eq!(pool.total_allocated(), 2);
        pool.release(b1);
        pool.release(b2);
    }
}
