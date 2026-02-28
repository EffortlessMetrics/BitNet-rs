//! HIP stream management.

use crate::error::{Result, RocmError};
use crate::ffi::HipStream;
use tracing::debug;

/// A managed HIP stream.
pub struct Stream {
    handle: HipStream,
    is_default: bool,
}

impl Stream {
    /// Create a new non-blocking stream (stub).
    pub fn new() -> Result<Self> {
        debug!("hipStreamCreateWithFlags (stub)");
        // Stub: would call hipStreamCreateWithFlags via dlsym.
        Err(RocmError::RuntimeNotFound(
            "HIP runtime not linked — stub only".into(),
        ))
    }

    /// Wrap the default (null) stream.
    pub fn default_stream() -> Self {
        Self {
            handle: crate::ffi::HIP_STREAM_DEFAULT,
            is_default: true,
        }
    }

    /// Synchronise the stream (wait for all queued work).
    pub fn synchronize(&self) -> Result<()> {
        debug!("hipStreamSynchronize (stub)");
        if self.is_default {
            return Ok(());
        }
        Err(RocmError::RuntimeNotFound(
            "HIP runtime not linked — stub only".into(),
        ))
    }

    /// Raw handle for kernel launch configs.
    pub fn handle(&self) -> HipStream {
        self.handle
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.is_default && !self.handle.is_null() {
            debug!("hipStreamDestroy (stub)");
            // Stub: would call hipStreamDestroy.
        }
    }
}
