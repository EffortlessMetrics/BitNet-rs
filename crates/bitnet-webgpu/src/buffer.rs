//! GPU buffer management with staging readback.

use crate::error::{Result, WebGpuError};
use bytemuck::Pod;

/// A GPU storage buffer paired with an optional staging buffer for readback.
pub struct GpuBuffer {
    pub storage: wgpu::Buffer,
    pub size: u64,
}

impl GpuBuffer {
    /// Create a GPU storage buffer initialised from `data`.
    pub fn from_slice<T: Pod>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[T],
        label: &str,
    ) -> Self {
        let bytes = bytemuck::cast_slice(data);
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&storage, 0, bytes);
        Self {
            storage,
            size: bytes.len() as u64,
        }
    }

    /// Create an uninitialised storage buffer of `len` elements.
    pub fn new_uninit<T: Pod>(device: &wgpu::Device, len: usize, label: &str) -> Self {
        let size = (len * std::mem::size_of::<T>()) as u64;
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { storage, size }
    }

    /// Read the buffer contents back to the CPU.
    pub async fn read_back<T: Pod>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<T>> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-readback"),
            size: self.size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback-encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.storage, 0, &staging, 0, self.size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| WebGpuError::BufferMap(e.to_string()))?
            .map_err(|e: wgpu::BufferAsyncError| WebGpuError::BufferMap(e.to_string()))?;

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(result)
    }

    /// Create a uniform buffer from a `Pod` value.
    pub fn new_uniform<T: Pod>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        value: &T,
        label: &str,
    ) -> wgpu::Buffer {
        let bytes = bytemuck::bytes_of(value);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytes);
        buf
    }
}
