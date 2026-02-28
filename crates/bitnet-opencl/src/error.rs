//! Error types for the async GPU execution engine.

/// Result alias for GPU operations.
pub type GpuResult<T> = Result<T, GpuError>;

/// Errors that can occur during GPU command management.
#[derive(Debug, Clone, thiserror::Error)]
pub enum GpuError {
    /// Requested queue index is out of range.
    #[error("invalid queue index {requested} (only {available} queues available)")]
    InvalidQueue { requested: usize, available: usize },

    /// A dependency index in an execution plan is invalid
    /// (forward reference or out of bounds).
    #[error(
        "invalid dependency: op {op_index} depends on {dep_index} \
         which has not been submitted yet"
    )]
    InvalidDependency { op_index: usize, dep_index: usize },

    /// A submitted GPU operation failed.
    #[error("GPU operation failed: {op}")]
    OperationFailed { op: String },

    /// Attempted to read a result that is not yet available.
    #[error("operation has not completed yet")]
    OperationIncomplete,

    /// Pipeline depth is out of the allowed range.
    #[error("pipeline depth {depth} out of allowed range [{min}..={max}]")]
    InvalidPipelineDepth { depth: usize, min: usize, max: usize },
}
