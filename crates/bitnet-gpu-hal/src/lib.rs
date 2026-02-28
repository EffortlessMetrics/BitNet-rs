//! GPU hardware abstraction layer for `BitNet` inference.

pub mod model_warmup;
//!
//! Provides checkpoint management for saving and resuming inference state,
//! with incremental diffs, compression, and automatic scheduling.

pub mod checkpoint_manager;
