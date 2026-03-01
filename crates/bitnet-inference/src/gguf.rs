//! Re-export of `bitnet-gguf::kv` for backward compatibility.
//!
//! GGUF header/KV inspection utilities have been extracted into
//! the `bitnet-gguf` SRP microcrate. This module is kept so existing
//! `bitnet_inference::gguf::*` imports continue to work.

pub use bitnet_gguf::kv::*;
