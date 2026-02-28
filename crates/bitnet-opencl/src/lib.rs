//! OpenCL 3.0 backend for BitNet inference.
//!
//! Provides Unified Shared Memory (USM) support for zero-copy host-device
//! data access, with automatic fallback to explicit buffer copies when USM
//! is unavailable.

pub mod usm;
