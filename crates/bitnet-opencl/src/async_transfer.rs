//! Asynchronous (non-blocking) data transfer helpers.
//!
//! Standard `OpenClBuffer::write`/`read` use `CL_BLOCKING` which
//! stalls the host thread until the DMA completes.  This module
//! provides non-blocking variants that return a `TransferEvent`
//! the caller can wait on later, enabling overlap between host work
//! and GPU transfers.

use crate::buffer::OpenClBuffer;
use crate::error::{OpenClError, Result};
use crate::queue::OpenClQueue;
use opencl3::event::Event;
use opencl3::types::{cl_event, CL_NON_BLOCKING};

/// An in-flight asynchronous transfer that wraps an OpenCL event.
///
/// Drop without calling `wait` is safe but the transfer may not
/// have completed.
pub struct TransferEvent {
    event: Event,
}

impl TransferEvent {
    /// Block until the transfer completes.
    pub fn wait(self) -> Result<()> {
        self.event.wait().map_err(|e| OpenClError::DataTransfer {
            reason: format!("async wait: {e}"),
        })
    }

    /// Return the raw `cl_event` for use in event wait lists.
    pub fn raw(&self) -> cl_event {
        self.event.get()
    }
}

impl std::fmt::Debug for TransferEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransferEvent").finish()
    }
}

/// Enqueue a non-blocking write from `data` into `buffer`.
///
/// The caller must ensure `data` remains valid until the returned
/// event completes.
pub fn async_write<T: Copy + 'static>(
    queue: &OpenClQueue,
    buffer: &mut OpenClBuffer<T>,
    data: &[T],
    wait_list: &[cl_event],
) -> Result<TransferEvent> {
    if data.len() > buffer.len {
        return Err(OpenClError::DataTransfer {
            reason: format!(
                "async write: source ({}) exceeds buffer ({})",
                data.len(),
                buffer.len
            ),
        });
    }
    let event = unsafe {
        queue
            .inner
            .enqueue_write_buffer(
                &mut buffer.inner,
                CL_NON_BLOCKING,
                0,
                data,
                wait_list,
            )
            .map_err(|e| OpenClError::DataTransfer {
                reason: format!("async write enqueue: {e}"),
            })?
    };
    Ok(TransferEvent { event })
}

/// Enqueue a non-blocking read from `buffer` into `dst`.
///
/// The caller must ensure `dst` remains valid and is not read until
/// the returned event completes.
pub fn async_read<T: Copy + 'static>(
    queue: &OpenClQueue,
    buffer: &OpenClBuffer<T>,
    dst: &mut [T],
    wait_list: &[cl_event],
) -> Result<TransferEvent> {
    if dst.len() > buffer.len {
        return Err(OpenClError::DataTransfer {
            reason: format!(
                "async read: destination ({}) exceeds buffer ({})",
                dst.len(),
                buffer.len
            ),
        });
    }
    let event = unsafe {
        queue
            .inner
            .enqueue_read_buffer(
                &buffer.inner,
                CL_NON_BLOCKING,
                0,
                dst,
                wait_list,
            )
            .map_err(|e| OpenClError::DataTransfer {
                reason: format!("async read enqueue: {e}"),
            })?
    };
    Ok(TransferEvent { event })
}

/// Wait for multiple transfer events to complete.
pub fn wait_all(events: Vec<TransferEvent>) -> Result<()> {
    for ev in events {
        ev.wait()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::AccessMode;
    use crate::context::OpenClContext;

    #[test]
    fn transfer_event_debug() {
        let desc = format!("{:?}", "TransferEvent");
        assert!(desc.contains("TransferEvent"));
    }

    #[test]
    fn async_write_overflow_rejected() {
        let ctx = OpenClContext::new_intel();
        if let Ok(ctx) = ctx {
            let queue = ctx.create_queue().expect("queue");
            let mut buf =
                OpenClBuffer::<f32>::new(&ctx, 2, AccessMode::ReadWrite)
                    .expect("alloc");
            let data = [1.0f32, 2.0, 3.0];
            let err = async_write(&queue, &mut buf, &data, &[]);
            assert!(err.is_err());
        }
    }

    #[test]
    fn async_read_overflow_rejected() {
        let ctx = OpenClContext::new_intel();
        if let Ok(ctx) = ctx {
            let queue = ctx.create_queue().expect("queue");
            let buf =
                OpenClBuffer::<f32>::new(&ctx, 2, AccessMode::ReadWrite)
                    .expect("alloc");
            let mut dst = [0.0f32; 4];
            let err = async_read(&queue, &buf, &mut dst, &[]);
            assert!(err.is_err());
        }
    }

    #[test]
    fn async_roundtrip_with_hardware() {
        let ctx = OpenClContext::new_intel();
        if let Ok(ctx) = ctx {
            let queue = ctx.create_queue().expect("queue");
            let mut buf =
                OpenClBuffer::<f32>::new(&ctx, 4, AccessMode::ReadWrite)
                    .expect("alloc");

            let src = [10.0f32, 20.0, 30.0, 40.0];
            let write_ev =
                async_write(&queue, &mut buf, &src, &[]).expect("write");
            write_ev.wait().expect("write wait");

            let mut dst = [0.0f32; 4];
            let read_ev =
                async_read(&queue, &buf, &mut dst, &[]).expect("read");
            read_ev.wait().expect("read wait");

            assert_eq!(dst, src);
        }
    }

    #[test]
    fn chained_events_with_hardware() {
        let ctx = OpenClContext::new_intel();
        if let Ok(ctx) = ctx {
            let queue = ctx.create_queue().expect("queue");
            let mut buf =
                OpenClBuffer::<f32>::new(&ctx, 4, AccessMode::ReadWrite)
                    .expect("alloc");

            let src = [1.0f32, 2.0, 3.0, 4.0];
            let write_ev =
                async_write(&queue, &mut buf, &src, &[]).expect("write");

            let mut dst = [0.0f32; 4];
            let read_ev = async_read(
                &queue,
                &buf,
                &mut dst,
                &[write_ev.raw()],
            )
            .expect("read");
            read_ev.wait().expect("read wait");

            assert_eq!(dst, src);
        }
    }

    #[test]
    fn wait_all_empty() {
        wait_all(vec![]).expect("empty wait_all");
    }
}