//! Backward-compatible shim for token streaming primitives.
//!
//! The implementation has been extracted to the `bitnet-token-stream-core`
//! SRP microcrate so other crates can share it without depending on
//! `bitnet-inference` internals.

pub use bitnet_token_stream_core::*;
