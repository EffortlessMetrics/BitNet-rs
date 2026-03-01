//! Compatibility shim for token streaming primitives.
//!
//! The implementation lives in `bitnet-token-stream-core` so other crates can
//! reuse the same streaming behavior without depending on `bitnet-inference`.

pub use bitnet_token_stream_core::*;
