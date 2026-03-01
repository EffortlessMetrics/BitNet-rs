//! Compatibility shim for token streaming primitives.
//!
//! The implementation has been extracted into the `bitnet-token-stream`
//! microcrate to keep streaming concerns decoupled from inference engines.

pub use bitnet_token_stream::{StreamConfig, StreamEvent, StreamStats, TokenBuffer, TokenStream};
