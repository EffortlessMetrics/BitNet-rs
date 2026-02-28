//! OpenCL backend for BitNet inference.
//!
//! This crate provides OpenCL 3.0 compute backend support for BitNet-rs,
//! including:
//!
//! - **WASM-compatible kernel validation** ([`wasm_shim`]) — parse and validate
//!   OpenCL kernel source on any target (including `wasm32`) without FFI
//! - **Unified Shared Memory (USM)** ([`usm`]) — zero-copy host↔device data
//!   access via `clSVMAlloc`, with automatic fallback to explicit buffer copies
//! - **Peer-to-peer transfers** ([`p2p`]) — GPU-to-GPU memory transfers with
//!   bandwidth measurement and automatic fallback to host-staged copies
//!
//! # Feature Gates
//!
//! Real OpenCL FFI calls are gated behind `#[cfg(not(target_arch = "wasm32"))]`.
//! The [`wasm_shim`] module provides pure-Rust validation that works everywhere.

pub mod wasm_shim;
pub mod usm;
pub mod p2p;

pub use wasm_shim::{
    ArgQualifier, KernelArg, KernelSignature, MockOpenClContext, parse_kernel_signatures,
};
