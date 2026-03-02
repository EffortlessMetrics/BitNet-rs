//! Decode-loop generation contracts for `BitNet` inference.
//!
//! This crate provides orchestration-facing generation types:
//! - generation config (via `bitnet-generation-config-core`)
//! - stream events (via `bitnet-generation-events-core`)
//!
//! Decode-loop stop criteria and stop-check logic live in
//! `bitnet-generation-stop-core` and are re-exported here.

pub use bitnet_generation_config_core::GenerationConfig;
pub use bitnet_generation_events_core::{GenerationStats, StreamEvent, TokenEvent};
pub use bitnet_generation_stop_core::{StopCriteria, StopReason, check_stop};
