//! OpenCL backend for BitNet GPU inference.
//!
//! Provides asynchronous kernel execution, event-based synchronization,
//! and double-buffered compute/transfer overlap for Intel and other
//! OpenCL-capable GPUs.

pub mod wasm_shim;

pub use wasm_shim::{
    ArgQualifier, KernelArg, KernelSignature, MockOpenClContext, parse_kernel_signatures,
};

pub mod usm;
pub mod p2p;
pub mod async_exec;
