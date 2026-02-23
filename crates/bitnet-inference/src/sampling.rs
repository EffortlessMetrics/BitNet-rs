//! Re-export of `bitnet-sampling` for backward compatibility.
//!
//! The main `SamplingConfig` and `SamplingStrategy` types have moved to
//! the `bitnet-sampling` crate. This module re-exports them.
//!
//! Note: `generation::sampling` provides a Candle-tensor-based sampling
//! implementation that remains in `bitnet-inference`.

pub use bitnet_sampling::*;
