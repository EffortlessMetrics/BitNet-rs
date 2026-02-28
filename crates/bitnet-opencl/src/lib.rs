//! WASM-compatible OpenCL kernel validation shim.
//!
//! Provides compile-time kernel source validation and a mock OpenCL context
//! that works on all targets, including wasm32. Real OpenCL FFI calls are
//! gated behind `#[cfg(not(target_arch = "wasm32"))]`.

pub mod wasm_shim;

pub use wasm_shim::{
    ArgQualifier, KernelArg, KernelSignature, MockOpenClContext, parse_kernel_signatures,
};
//! OpenCL GPU backend and memory management for BitNet inference.
pub mod memory_defrag;
