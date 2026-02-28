//! OpenCL backend for BitNet GPU inference.
//!
//! Provides asynchronous kernel execution, event-based synchronization,
//! and double-buffered compute/transfer overlap for Intel and other
//! OpenCL-capable GPUs.

pub mod async_exec;
