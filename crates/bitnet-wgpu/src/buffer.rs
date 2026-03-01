//! GPU buffer management and pool-based allocation.

use wgpu::BufferUsages;

use crate::device::WgpuDevice;
use crate::error::WgpuError;

/// Metadata-rich wrapper around a `wgpu::Buffer`.
pub struct GpuBuffer {
    inner: wgpu::Buffer,
    size: u64,
    usage: BufferUsages,
    label: String,
}

impl GpuBuffer {
    /// Create a storage buffer (read/write from shaders, copy-src for readback).
    pub fn new_storage(device: &WgpuDevice, size: u64, label: &str) -> Self {
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        let inner = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Self { inner, size, usage, label: label.to_string() }
    }

    /// Create a uniform buffer (read-only from shaders).
    pub fn new_uniform(device: &WgpuDevice, size: u64, label: &str) -> Self {
        let usage = BufferUsages::UNIFORM | BufferUsages::COPY_DST;
        let inner = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Self { inner, size, usage, label: label.to_string() }
    }

    /// Create a buffer with explicit usage flags.
    pub fn new_with_usage(
        device: &WgpuDevice,
        size: u64,
        usage: BufferUsages,
        label: &str,
    ) -> Self {
        let inner = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Self { inner, size, usage, label: label.to_string() }
    }

    /// Write data from the host into this buffer.
    pub fn write<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, data: &[T]) {
        queue.write_buffer(&self.inner, 0, bytemuck::cast_slice(data));
    }

    /// Asynchronously read buffer contents back to the host.
    pub async fn read_async<T: bytemuck::Pod>(
        &self,
        device: &WgpuDevice,
    ) -> Result<Vec<T>, WgpuError> {
        let staging = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback-staging"),
            size: self.size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback-encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.inner, 0, &staging, 0, self.size);
        device.queue().submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.device().poll(wgpu::Maintain::Wait);

        rx.recv().map_err(|e| WgpuError::mapping(e))?.map_err(|e| WgpuError::mapping(e))?;

        let view = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        Ok(result)
    }

    /// Blocking read-back wrapper.
    pub fn read_blocking<T: bytemuck::Pod>(
        &self,
        device: &WgpuDevice,
    ) -> Result<Vec<T>, WgpuError> {
        pollster::block_on(self.read_async(device))
    }

    /// Buffer size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Buffer usage flags.
    pub fn usage(&self) -> BufferUsages {
        self.usage
    }

    /// Human-readable label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Access the underlying `wgpu::Buffer`.
    pub fn inner(&self) -> &wgpu::Buffer {
        &self.inner
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("label", &self.label)
            .field("size", &self.size)
            .field("usage", &self.usage)
            .finish()
    }
}

/// Statistics for the buffer pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total buffers allocated (including reused).
    pub allocations: u64,
    /// Buffers served from the free list.
    pub reuses: u64,
    /// Buffers currently in the free list.
    pub free_count: usize,
    /// Buffers currently checked out.
    pub active_count: usize,
}

impl PoolStats {
    /// Fraction of allocations served from the free list (0.0–1.0).
    pub fn reuse_rate(&self) -> f64 {
        if self.allocations == 0 { 0.0 } else { self.reuses as f64 / self.allocations as f64 }
    }
}

/// Pool of reusable GPU buffers keyed by (size, usage).
#[derive(Default)]
pub struct BufferPool {
    /// Free buffers available for reuse, keyed by (size, usage-bits).
    free: Vec<(u64, BufferUsages, GpuBuffer)>,
    stats: PoolStats,
}

impl BufferPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate (or reuse) a buffer with the given size and usage.
    pub fn allocate(&mut self, device: &WgpuDevice, size: u64, usage: BufferUsages) -> GpuBuffer {
        self.stats.allocations += 1;

        // Try to reuse a matching buffer.
        if let Some(idx) = self.free.iter().position(|(s, u, _)| *s >= size && *u == usage) {
            self.stats.reuses += 1;
            self.stats.free_count -= 1;
            self.stats.active_count += 1;
            let (_, _, buf) = self.free.swap_remove(idx);
            return buf;
        }

        self.stats.active_count += 1;
        GpuBuffer::new_with_usage(device, size, usage, "pool-buffer")
    }

    /// Return a buffer to the pool for future reuse.
    pub fn release(&mut self, buffer: GpuBuffer) {
        let size = buffer.size();
        let usage = buffer.usage();
        self.stats.active_count = self.stats.active_count.saturating_sub(1);
        self.stats.free_count += 1;
        self.free.push((size, usage, buffer));
    }

    /// Current pool statistics.
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool")
            .field("free", &self.free.len())
            .field("stats", &self.stats)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── GPU-gated tests ──────────────────────────────────────────────

    #[test]
    #[ignore = "requires GPU runtime"]
    fn buffer_new_storage() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let buf = GpuBuffer::new_storage(&dev, 1024, "test-storage");
        assert_eq!(buf.size(), 1024);
        assert!(buf.usage().contains(BufferUsages::STORAGE));
        assert_eq!(buf.label(), "test-storage");
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn buffer_new_uniform() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let buf = GpuBuffer::new_uniform(&dev, 256, "test-uniform");
        assert_eq!(buf.size(), 256);
        assert!(buf.usage().contains(BufferUsages::UNIFORM));
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn buffer_write_read_roundtrip_f32() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let buf = GpuBuffer::new_storage(&dev, (data.len() * 4) as u64, "roundtrip");
        buf.write(dev.queue(), &data);
        let read_back: Vec<f32> = buf.read_blocking(&dev).unwrap();
        assert_eq!(read_back, data);
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn buffer_write_read_roundtrip_u32() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let data: Vec<u32> = vec![0xDEAD, 0xBEEF, 0xCAFE, 0xBABE];
        let buf = GpuBuffer::new_storage(&dev, (data.len() * 4) as u64, "roundtrip-u32");
        buf.write(dev.queue(), &data);
        let read_back: Vec<u32> = buf.read_blocking(&dev).unwrap();
        assert_eq!(read_back, data);
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn buffer_debug_output() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let buf = GpuBuffer::new_storage(&dev, 64, "dbg-test");
        let dbg = format!("{buf:?}");
        assert!(dbg.contains("GpuBuffer"));
        assert!(dbg.contains("dbg-test"));
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn pool_allocate_returns_buffer() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let mut pool = BufferPool::new();
        let buf = pool.allocate(&dev, 512, BufferUsages::STORAGE);
        assert!(buf.size() >= 512);
        assert_eq!(pool.stats().allocations, 1);
        assert_eq!(pool.stats().active_count, 1);
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn pool_reuse_after_release() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let mut pool = BufferPool::new();
        let buf = pool.allocate(&dev, 512, BufferUsages::STORAGE);
        pool.release(buf);
        assert_eq!(pool.stats().free_count, 1);

        let _buf2 = pool.allocate(&dev, 512, BufferUsages::STORAGE);
        assert_eq!(pool.stats().reuses, 1);
        assert_eq!(pool.stats().free_count, 0);
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn pool_no_reuse_for_different_usage() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let mut pool = BufferPool::new();
        let buf = pool.allocate(&dev, 512, BufferUsages::STORAGE);
        pool.release(buf);

        // Request UNIFORM — should not reuse the STORAGE buffer.
        let _buf2 = pool.allocate(&dev, 512, BufferUsages::UNIFORM);
        assert_eq!(pool.stats().reuses, 0);
    }

    #[test]
    #[ignore = "requires GPU runtime"]
    fn pool_reuse_larger_buffer() {
        let dev = WgpuDevice::new_blocking(&Default::default()).unwrap();
        let mut pool = BufferPool::new();
        let buf = pool.allocate(&dev, 1024, BufferUsages::STORAGE);
        pool.release(buf);

        // Requesting 512 — should reuse the 1024 buffer.
        let reused = pool.allocate(&dev, 512, BufferUsages::STORAGE);
        assert!(reused.size() >= 512);
        assert_eq!(pool.stats().reuses, 1);
    }

    // ── Non-GPU tests ────────────────────────────────────────────────

    #[test]
    fn pool_stats_initial() {
        let pool = BufferPool::new();
        let stats = pool.stats();
        assert_eq!(stats.allocations, 0);
        assert_eq!(stats.reuses, 0);
        assert_eq!(stats.free_count, 0);
        assert_eq!(stats.active_count, 0);
    }

    #[test]
    fn pool_stats_reuse_rate_empty() {
        let stats = PoolStats::default();
        assert_eq!(stats.reuse_rate(), 0.0);
    }

    #[test]
    fn pool_stats_reuse_rate_computed() {
        let stats = PoolStats { allocations: 10, reuses: 3, free_count: 0, active_count: 0 };
        let rate = stats.reuse_rate();
        assert!((rate - 0.3).abs() < 1e-9);
    }

    #[test]
    fn pool_debug_output() {
        let pool = BufferPool::new();
        let dbg = format!("{pool:?}");
        assert!(dbg.contains("BufferPool"));
    }
}
