//! Level-Zero context management.
//!
//! A `ZeContext` wraps `ze_context_handle_t` and is the scope for
//! memory allocations, modules, and command queues.

use crate::error::Result;
use crate::ffi::ZeContextHandle;

/// Configuration for creating a Level-Zero context.
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Optional flags (reserved for future use).
    pub flags: u32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self { flags: 0 }
    }
}

/// Builder for `LevelZeroContext`.
#[derive(Debug)]
pub struct ContextBuilder {
    config: ContextConfig,
    driver_index: usize,
}

impl ContextBuilder {
    /// Create a new context builder targeting the given driver index.
    pub fn new(driver_index: usize) -> Self {
        Self {
            config: ContextConfig::default(),
            driver_index,
        }
    }

    /// Set context flags.
    pub fn flags(mut self, flags: u32) -> Self {
        self.config.flags = flags;
        self
    }

    /// Build the context.
    ///
    /// Placeholder: real implementation calls `zeContextCreate`.
    pub fn build(self) -> Result<LevelZeroContext> {
        tracing::debug!(
            driver_index = self.driver_index,
            "Creating Level-Zero context (placeholder)"
        );
        Ok(LevelZeroContext {
            _config: self.config,
            driver_index: self.driver_index,
            _handle: None,
        })
    }
}

/// An owned Level-Zero context.
///
/// Manages the lifetime of a `ze_context_handle_t`.
#[derive(Debug)]
pub struct LevelZeroContext {
    _config: ContextConfig,
    driver_index: usize,
    _handle: Option<ZeContextHandle>,
}

impl LevelZeroContext {
    /// Driver index this context is associated with.
    pub fn driver_index(&self) -> usize {
        self.driver_index
    }

    /// Whether this context has a live L0 handle.
    pub fn is_initialized(&self) -> bool {
        self._handle.is_some()
    }
}
