// GPU hardware abstraction layer for `BitNet` inference.

pub mod deployment_manager;
pub mod model_warmup;
//
// Provides checkpoint management for saving and resuming inference state,
// with incremental diffs, compression, and automatic scheduling.

pub mod checkpoint_manager;
