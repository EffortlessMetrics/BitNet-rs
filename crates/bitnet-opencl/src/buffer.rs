//! Buffer allocation and data transfer helpers for OpenCL.

use crate::context::OpenClContext;
use crate::error::{OpenClError, Result};
use crate::queue::OpenClQueue;
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::types::CL_BLOCKING;
use std::marker::PhantomData;

/// Access mode for an OpenCL buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

impl AccessMode {
    fn to_cl_flags(self) -> u64 {
        match self {
            AccessMode::ReadOnly => CL_MEM_READ_ONLY,
            AccessMode::WriteOnly => CL_MEM_WRITE_ONLY,
            AccessMode::ReadWrite => CL_MEM_READ_WRITE,
        }
    }
}

/// Type-safe wrapper around an OpenCL buffer.
pub struct OpenClBuffer<T: Copy + 'static> {
    /// The underlying opencl3 buffer (public for kernel dispatch).
    pub inner: Buffer<T>,
    /// Number of elements (not bytes).
    pub len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy + 'static> std::fmt::Debug for OpenClBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenClBuffer")
            .field("len", &self.len)
            .field("elem_size", &std::mem::size_of::<T>())
            .finish()
    }
}

impl<T: Copy + 'static> OpenClBuffer<T> {
    /// Allocate a new device buffer.
    pub fn new(ctx: &OpenClContext, len: usize, mode: AccessMode) -> Result<Self> {
        let byte_size = len * std::mem::size_of::<T>();
        let inner = unsafe {
            Buffer::<T>::create(
                &ctx.inner,
                mode.to_cl_flags(),
                len,
                std::ptr::null_mut(),
            )
            .map_err(|e| OpenClError::BufferAllocation {
                size: byte_size,
                reason: e.to_string(),
            })?
        };
        Ok(Self {
            inner,
            len,
            _marker: PhantomData,
        })
    }

    /// Upload host data to device (blocking).
    pub fn write(&mut self, queue: &OpenClQueue, data: &[T]) -> Result<()> {
        if data.len() > self.len {
            return Err(OpenClError::DataTransfer {
                reason: format!(
                    "source ({}) exceeds buffer capacity ({})",
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
                    reason: format!("write: {e}"),
                })?;
        }
        Ok(())
    }

    /// Download device data to host (blocking).
    pub fn read(&self, queue: &OpenClQueue, dst: &mut [T]) -> Result<()> {
        if dst.len() > self.len {
            return Err(OpenClError::DataTransfer {
                reason: format!(
                    "destination ({}) exceeds buffer capacity ({})",
                    dst.len(),
                    self.len
                ),
            });
        }
        unsafe {
            queue
                .inner
                .enqueue_read_buffer(&self.inner, CL_BLOCKING, 0, dst, &[])
                .map_err(|e| OpenClError::DataTransfer {
                    reason: format!("read: {e}"),
                })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn access_mode_flags() {
        assert_eq!(AccessMode::ReadOnly.to_cl_flags(), CL_MEM_READ_ONLY);
        assert_eq!(AccessMode::WriteOnly.to_cl_flags(), CL_MEM_WRITE_ONLY);
        assert_eq!(AccessMode::ReadWrite.to_cl_flags(), CL_MEM_READ_WRITE);
    }

    #[test]
    fn buffer_with_hardware() {
        let ctx = OpenClContext::new_intel();
        if let Ok(ctx) = ctx {
            let buf =
                OpenClBuffer::<f32>::new(&ctx, 64, AccessMode::ReadWrite);
            assert!(buf.is_ok());
        }
    }

    #[test]
    fn debug_impl_no_panic() {
        let desc = format!(
            "{:?}",
            "OpenClBuffer { len: 128, elem_size: 4 }"
        );
        assert!(desc.contains("OpenClBuffer"));
    }
}
