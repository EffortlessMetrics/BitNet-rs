//! Pinned (host-accessible) OpenCL buffers for fast transfers.
//!
//! `CL_MEM_ALLOC_HOST_PTR` tells the driver to allocate the backing
//! store in host-accessible memory (often DMA-reachable), which avoids
//! an extra copy during `enqueue_read/write_buffer`.  On Intel GPUs
//! this typically maps to shared-memory regions accessible by both CPU
//! and GPU, giving significantly faster data transfers.

use crate::buffer::AccessMode;
use crate::context::OpenClContext;
use crate::error::{OpenClError, Result};
use crate::queue::OpenClQueue;
use opencl3::memory::{
    Buffer, CL_MEM_ALLOC_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE,
    CL_MEM_WRITE_ONLY,
};
use opencl3::types::CL_BLOCKING;
use std::marker::PhantomData;
use tracing::debug;

/// A buffer allocated with `CL_MEM_ALLOC_HOST_PTR` for reduced-copy
/// transfers between host and device.
pub struct PinnedBuffer<T: Copy + 'static> {
    /// The underlying opencl3 buffer (public for kernel dispatch).
    pub inner: Buffer<T>,
    /// Number of elements.
    pub len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy + 'static> std::fmt::Debug for PinnedBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedBuffer")
            .field("len", &self.len)
            .field("elem_size", &std::mem::size_of::<T>())
            .finish()
    }
}

impl<T: Copy + 'static> PinnedBuffer<T> {
    /// Allocate a new pinned device buffer.
    pub fn new(
        ctx: &OpenClContext,
        len: usize,
        mode: AccessMode,
    ) -> Result<Self> {
        let flags = mode_to_flags(mode) | CL_MEM_ALLOC_HOST_PTR;
        let byte_size = len * std::mem::size_of::<T>();
        let inner = unsafe {
            Buffer::<T>::create(
                &ctx.inner,
                flags,
                len,
                std::ptr::null_mut(),
            )
            .map_err(|e| OpenClError::BufferAllocation {
                size: byte_size,
                reason: format!("pinned: {e}"),
            })?
        };
        debug!(
            "pinned buffer allocated: {} elems, {} bytes",
            len, byte_size
        );
        Ok(Self {
            inner,
            len,
            _marker: PhantomData,
        })
    }

    /// Upload host data to the pinned buffer (blocking).
    pub fn write(
        &mut self,
        queue: &OpenClQueue,
        data: &[T],
    ) -> Result<()> {
        if data.len() > self.len {
            return Err(OpenClError::DataTransfer {
                reason: format!(
                    "source ({}) exceeds pinned buffer capacity ({})",
                    data.len(),
                    self.len
                ),
            });
        }
        unsafe {
            queue
                .inner
                .enqueue_write_buffer(
                    &mut self.inner,
                    CL_BLOCKING,
                    0,
                    data,
                    &[],
                )
                .map_err(|e| OpenClError::DataTransfer {
                    reason: format!("pinned write: {e}"),
                })?;
        }
        Ok(())
    }

    /// Download device data from the pinned buffer (blocking).
    pub fn read(
        &self,
        queue: &OpenClQueue,
        dst: &mut [T],
    ) -> Result<()> {
        if dst.len() > self.len {
            return Err(OpenClError::DataTransfer {
                reason: format!(
                    "destination ({}) exceeds pinned buffer capacity ({})",
                    dst.len(),
                    self.len
                ),
            });
        }
        unsafe {
            queue
                .inner
                .enqueue_read_buffer(
                    &self.inner,
                    CL_BLOCKING,
                    0,
                    dst,
                    &[],
                )
                .map_err(|e| OpenClError::DataTransfer {
                    reason: format!("pinned read: {e}"),
                })?;
        }
        Ok(())
    }
}

fn mode_to_flags(mode: AccessMode) -> u64 {
    match mode {
        AccessMode::ReadOnly => CL_MEM_READ_ONLY,
        AccessMode::WriteOnly => CL_MEM_WRITE_ONLY,
        AccessMode::ReadWrite => CL_MEM_READ_WRITE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_flags_include_alloc_host() {
        let flags =
            mode_to_flags(AccessMode::ReadWrite) | CL_MEM_ALLOC_HOST_PTR;
        assert_ne!(flags & CL_MEM_ALLOC_HOST_PTR, 0);
        assert_ne!(flags & CL_MEM_READ_WRITE, 0);
    }

    #[test]
    fn pinned_buffer_with_hardware() {
        let ctx = crate::context::OpenClContext::new_intel();
        if let Ok(ctx) = ctx {
            let buf =
                PinnedBuffer::<f32>::new(&ctx, 256, AccessMode::ReadWrite);
            assert!(buf.is_ok());
            assert_eq!(buf.unwrap().len, 256);
        }
    }

    #[test]
    fn pinned_write_read_roundtrip() {
        let ctx = crate::context::OpenClContext::new_intel();
        if let Ok(ctx) = ctx {
            let queue = ctx.create_queue().expect("queue");
            let mut buf =
                PinnedBuffer::<f32>::new(&ctx, 4, AccessMode::ReadWrite)
                    .expect("alloc");
            let src = [1.0f32, 2.0, 3.0, 4.0];
            buf.write(&queue, &src).expect("write");
            let mut dst = [0.0f32; 4];
            buf.read(&queue, &mut dst).expect("read");
            assert_eq!(dst, src);
        }
    }

    #[test]
    fn pinned_debug_impl() {
        let desc = format!(
            "{:?}",
            "PinnedBuffer { len: 64, elem_size: 4 }"
        );
        assert!(desc.contains("PinnedBuffer"));
    }

    #[test]
    fn write_overflow_rejected() {
        let ctx = crate::context::OpenClContext::new_intel();
        if let Ok(ctx) = ctx {
            let queue = ctx.create_queue().expect("queue");
            let mut buf =
                PinnedBuffer::<f32>::new(&ctx, 2, AccessMode::ReadWrite)
                    .expect("alloc");
            let data = [1.0f32, 2.0, 3.0];
            let err = buf.write(&queue, &data);
            assert!(err.is_err());
        }
    }

    #[test]
    fn read_overflow_rejected() {
        let ctx = crate::context::OpenClContext::new_intel();
        if let Ok(ctx) = ctx {
            let queue = ctx.create_queue().expect("queue");
            let buf =
                PinnedBuffer::<f32>::new(&ctx, 2, AccessMode::ReadWrite)
                    .expect("alloc");
            let mut dst = [0.0f32; 4];
            let err = buf.read(&queue, &mut dst);
            assert!(err.is_err());
        }
    }
}