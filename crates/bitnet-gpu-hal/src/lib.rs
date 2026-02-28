// GPU hardware abstraction layer for `BitNet` inference.

pub mod async_runtime;
pub mod backend_selector;
pub mod bench_harness;
pub mod deployment_manager;
pub mod generation;
pub mod hal_traits;
pub mod model_warmup;
pub mod test_harness;
pub mod weight_compression;
use std::fmt;

// Provides checkpoint management for saving and resuming inference state,
// with incremental diffs, compression, and automatic scheduling.
pub mod checkpoint_manager;

// Provides batched tokenization, parallel encoding/decoding,
// and hardware abstraction for GPU-accelerated inference pipelines.
pub mod batched_tokenization;
pub mod model_architecture;
pub mod streaming_aggregator;
// Parallel communication primitives for distributed GPU inference:
// all-reduce, all-gather, reduce-scatter, broadcast, ring/tree
// topologies, double-buffered comm, and profiling.
pub mod parallel_communication;
