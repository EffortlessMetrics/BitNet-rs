//! Backward-compatible shim around the SRP-extracted `bitnet-token-stream-core` crate.

pub use bitnet_token_stream_core::{
    StreamConfig, StreamEvent, StreamStats, TokenBuffer, TokenStream,
};
