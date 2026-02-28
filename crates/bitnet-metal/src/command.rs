//! Metal command buffer lifecycle management.
//!
//! Abstracts the MTLCommandBuffer lifecycle: create → encode → commit → wait.
//! On non-macOS platforms, provides a stub that validates the dispatch sequence.

use crate::error::{MetalError, Result};
use crate::pipeline::PipelineDescriptor;
use tracing::debug;

/// State machine for command buffer lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandBufferState {
    /// Buffer created, ready for encoding.
    Created,
    /// Compute commands have been encoded.
    Encoded,
    /// Buffer committed to the GPU command queue.
    Committed,
    /// GPU execution completed.
    Completed,
}

/// Abstraction over a Metal command buffer.
///
/// Enforces the correct lifecycle: Created → Encoded → Committed → Completed.
#[derive(Debug)]
pub struct CommandBuffer {
    state: CommandBufferState,
    encoded_dispatches: u32,
}

impl CommandBuffer {
    /// Create a new command buffer.
    pub fn new() -> Self {
        Self {
            state: CommandBufferState::Created,
            encoded_dispatches: 0,
        }
    }

    /// Current state of the command buffer.
    pub fn state(&self) -> CommandBufferState {
        self.state
    }

    /// Number of compute dispatches encoded.
    pub fn dispatch_count(&self) -> u32 {
        self.encoded_dispatches
    }

    /// Encode a compute dispatch for the given pipeline and grid.
    pub fn encode_dispatch(
        &mut self,
        pipeline: &PipelineDescriptor,
        grid_size: (u32, u32, u32),
    ) -> Result<()> {
        if self.state != CommandBufferState::Created
            && self.state != CommandBufferState::Encoded
        {
            return Err(MetalError::CommandBuffer(format!(
                "cannot encode in state {:?}",
                self.state
            )));
        }
        debug!(
            kernel = pipeline.function_name,
            grid = ?grid_size,
            threadgroup = ?pipeline.threadgroup_size,
            "encoding compute dispatch"
        );
        self.encoded_dispatches += 1;
        self.state = CommandBufferState::Encoded;
        Ok(())
    }

    /// Commit the command buffer for execution.
    pub fn commit(&mut self) -> Result<()> {
        if self.state != CommandBufferState::Encoded {
            return Err(MetalError::CommandBuffer(format!(
                "cannot commit in state {:?} (must encode first)",
                self.state
            )));
        }
        debug!(dispatches = self.encoded_dispatches, "committing command buffer");
        self.state = CommandBufferState::Committed;
        Ok(())
    }

    /// Wait for GPU execution to complete (stub: immediately transitions).
    pub fn wait_until_completed(&mut self) -> Result<()> {
        if self.state != CommandBufferState::Committed {
            return Err(MetalError::CommandBuffer(format!(
                "cannot wait in state {:?} (must commit first)",
                self.state
            )));
        }
        self.state = CommandBufferState::Completed;
        Ok(())
    }
}

impl Default for CommandBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader;

    fn test_pipeline() -> PipelineDescriptor {
        PipelineDescriptor {
            function_name: "matmul",
            source: shader::MATMUL_MSL,
            threadgroup_size: (16, 16, 1),
        }
    }

    #[test]
    fn new_buffer_is_created() {
        let buf = CommandBuffer::new();
        assert_eq!(buf.state(), CommandBufferState::Created);
        assert_eq!(buf.dispatch_count(), 0);
    }

    #[test]
    fn encode_transitions_to_encoded() {
        let mut buf = CommandBuffer::new();
        buf.encode_dispatch(&test_pipeline(), (4, 4, 1)).unwrap();
        assert_eq!(buf.state(), CommandBufferState::Encoded);
        assert_eq!(buf.dispatch_count(), 1);
    }

    #[test]
    fn multiple_encodes_accumulate() {
        let mut buf = CommandBuffer::new();
        buf.encode_dispatch(&test_pipeline(), (4, 4, 1)).unwrap();
        buf.encode_dispatch(&test_pipeline(), (8, 8, 1)).unwrap();
        assert_eq!(buf.dispatch_count(), 2);
    }

    #[test]
    fn commit_after_encode_succeeds() {
        let mut buf = CommandBuffer::new();
        buf.encode_dispatch(&test_pipeline(), (4, 4, 1)).unwrap();
        assert!(buf.commit().is_ok());
        assert_eq!(buf.state(), CommandBufferState::Committed);
    }

    #[test]
    fn commit_without_encode_fails() {
        let mut buf = CommandBuffer::new();
        assert!(buf.commit().is_err());
    }

    #[test]
    fn wait_after_commit_succeeds() {
        let mut buf = CommandBuffer::new();
        buf.encode_dispatch(&test_pipeline(), (4, 4, 1)).unwrap();
        buf.commit().unwrap();
        assert!(buf.wait_until_completed().is_ok());
        assert_eq!(buf.state(), CommandBufferState::Completed);
    }

    #[test]
    fn wait_without_commit_fails() {
        let mut buf = CommandBuffer::new();
        buf.encode_dispatch(&test_pipeline(), (4, 4, 1)).unwrap();
        assert!(buf.wait_until_completed().is_err());
    }

    #[test]
    fn full_lifecycle() {
        let mut buf = CommandBuffer::new();
        let pipe = test_pipeline();

        assert_eq!(buf.state(), CommandBufferState::Created);
        buf.encode_dispatch(&pipe, (4, 4, 1)).unwrap();
        assert_eq!(buf.state(), CommandBufferState::Encoded);
        buf.commit().unwrap();
        assert_eq!(buf.state(), CommandBufferState::Committed);
        buf.wait_until_completed().unwrap();
        assert_eq!(buf.state(), CommandBufferState::Completed);
    }
}
