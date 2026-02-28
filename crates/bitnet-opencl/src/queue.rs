//! Command queue management for OpenCL.

use crate::context::OpenClContext;
use crate::error::{OpenClError, Result};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use tracing::debug;

/// Wrapper around an OpenCL command queue with profiling support.
pub struct OpenClQueue {
    /// The underlying opencl3 command queue.
    pub inner: CommandQueue,
}

impl std::fmt::Debug for OpenClQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenClQueue").finish()
    }
}

// SAFETY: CommandQueue serialises operations internally.
unsafe impl Send for OpenClQueue {}
unsafe impl Sync for OpenClQueue {}

impl OpenClQueue {
    /// Create a new command queue with profiling enabled.
    pub fn new(ctx: &OpenClContext) -> Result<Self> {
        let inner = CommandQueue::create_default_with_properties(
            &ctx.inner,
            CL_QUEUE_PROFILING_ENABLE,
            0,
        )
        .map_err(|e| OpenClError::QueueCreation {
            reason: e.to_string(),
        })?;

        debug!("OpenCL command queue created");
        Ok(Self { inner })
    }

    /// Flush pending commands to the device (non-blocking).
    pub fn flush(&self) -> Result<()> {
        self.inner.flush().map_err(|e| OpenClError::QueueCreation {
            reason: format!("flush failed: {e}"),
        })
    }

    /// Block until all enqueued commands have completed.
    pub fn finish(&self) -> Result<()> {
        self.inner
            .finish()
            .map_err(|e| OpenClError::QueueCreation {
                reason: format!("finish failed: {e}"),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_debug_impl() {
        let ctx = OpenClContext::new_intel();
        if let Ok(ctx) = ctx {
            if let Ok(q) = OpenClQueue::new(&ctx) {
                let dbg = format!("{:?}", q);
                assert!(dbg.contains("OpenClQueue"));
            }
        }
    }
}
