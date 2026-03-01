// GPU hardware abstraction layer for `BitNet` inference.

pub mod backend_selector;
pub mod bench_harness;
pub mod deployment_manager;
pub mod hal_traits;
pub mod model_warmup;

// Provides checkpoint management for saving and resuming inference state,
// with incremental diffs, compression, and automatic scheduling.
pub mod checkpoint_manager;

// Provides batched tokenization, parallel encoding/decoding,
// and hardware abstraction for GPU-accelerated inference pipelines.
pub mod batched_tokenization;
pub mod streaming_aggregator;
