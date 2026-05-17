// GPU hardware abstraction layer for `BitNet` inference.

// Scaffold crate — suppress noisy nursery/pedantic lints until the API stabilises.
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::unused_self)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::cloned_instead_of_copied)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::while_float)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::manual_div_ceil)]
#![cfg_attr(
    test,
    allow(
        clippy::approx_constant,
        clippy::default_constructed_unit_structs,
        clippy::erasing_op,
        clippy::field_reassign_with_default,
        clippy::identity_op,
        clippy::io_other_error,
        clippy::manual_range_contains,
        clippy::needless_range_loop,
        clippy::redundant_closure,
        clippy::result_large_err,
        clippy::useless_vec
    )
)]

// === GPU Backend Implementations ===
pub mod cuda_backend;
pub mod level_zero_backend;
pub mod metal_backend;
pub mod opencl_backend;
pub mod rocm_backend;
pub mod vulkan_compute;
pub mod webgpu_backend;

// === HAL Core ===
pub mod async_runtime;
pub mod backend_selector;
pub mod bench_harness;
pub mod config_management;
pub mod deployment_manager;
pub mod device_abstraction;
pub mod embedding_layer;
pub mod error_taxonomy;
pub mod hal_traits;
pub use hal_traits::{HalError, HalResult};

// === Compute Kernels ===
pub mod activation_functions;
pub mod attention_compute;
pub mod convolution_kernels;
pub mod embedding_operations;
pub mod layer_norm;
pub mod matmul_kernels;
pub mod normalization_variants;
pub mod softmax_kernel;

// === Memory Management ===
pub mod gpu_buffer;
pub mod mmap_io;
pub mod tensor_memory_pool;

// === Tensor Operations ===
pub mod dynamic_shapes;
pub mod shape_tracker;
pub mod sparse_operations;
pub mod tensor_ops_v2;
pub mod tensor_serde;

// === Model Architecture ===
pub mod attention_mechanism;
pub mod attention_patterns;
pub mod cross_attention;
pub mod ffn_block;
pub mod function_calling;
pub mod mqa_gqa;
pub mod rope_kernels;
pub mod transformer_block;

pub mod beam_search;
// === Inference Pipeline ===
pub mod autoregressive_generator;
pub mod context_window;
pub mod dynamic_batching;
pub mod inference_pipeline;
pub mod inference_session;
pub mod kv_cache_manager;
pub mod sampling_strategies;

// === Quantization ===
pub mod mixed_precision;
pub mod model_quantizer;
pub mod quant_calibration;
pub mod quantization_toolkit;
pub mod weight_compression;

// === Optimization ===
pub mod compute_graph;
pub mod execution_planner;
pub mod kernel_autotuner;
pub mod kernel_fusion;
pub mod operator_registry;
pub mod optimization_passes;
pub mod simd_dispatch;

// === I/O & Serialization ===
pub mod gguf_loader;
pub mod gguf_writer;
pub mod mmap_io_v2;
pub mod model_export;
pub mod model_serialization;
pub mod tokenizer_pipeline;
pub mod tokenizer_wrapper;

// === Profiling & Debugging ===
pub mod benchmark_harness;
pub mod continuous_profiling;
pub mod gpu_memory_profiler;
pub mod gpu_topology;
pub mod model_debugger;

// === Testing & Validation ===
pub mod activation_kernels;
pub mod compatibility_checker;
pub mod cross_compile;
pub mod e2e_integration;
pub mod kernel_jit;
pub mod model_validator;
pub mod test_harness;
pub mod token_streaming;

// === Infrastructure ===
pub mod arch_registry;
pub mod logging;
pub mod migration_tool;
pub mod observability;
pub mod thread_pool;

// === Distributed ===
pub mod distributed;
pub mod distributed_inference;
pub mod multi_device;
pub mod parallel_communication;

// === Server & Serving ===
pub mod cache_system;
pub mod inference_scheduler;
pub mod inference_tracing;
pub mod rate_limiter;
pub mod server_protocol;
pub mod serving_runtime;

// === ML Operations ===
pub mod gradient_checkpoint;
pub mod instruction_tuning;
pub mod model_architecture;
pub mod model_hub;
pub mod model_pruning;
pub mod multimodal_fusion;
pub mod semantic_search;

// === SPIR-V ===
pub mod perf_comparison;
pub mod spirv_compiler;

// === Docker/CI ===
pub mod docker_ci;

// === Existing Modules (prior waves) ===
pub mod generation;
pub mod guided_generation;
pub mod model_warmup;
pub mod prompt_cache;

//
// Provides checkpoint management for saving and resuming inference state,
// with incremental diffs, compression, and automatic scheduling.

pub mod checkpoint_manager;
// Provides batched tokenization, parallel encoding/decoding,
// and hardware abstraction for GPU-accelerated inference pipelines.
pub mod batched_tokenization;
pub mod prompt_processing;
pub mod streaming_aggregator;
pub mod structured_output;
// Parallel communication primitives for distributed GPU inference:
// all-reduce, all-gather, reduce-scatter, broadcast, ring/tree
// topologies, double-buffered comm, and profiling.
// Structured error taxonomy for GPU HAL with rich context,
// recovery strategies, and structured reporting.
pub mod error_recovery;
// GPU hardware abstraction layer for `BitNet` inference.
// GPU hardware abstraction layer for `BitNet` inference.
// GPU hardware abstraction layer for `BitNet` inference.
// DAG-based execution planner with memory planning, stream scheduling,
// kernel launch configuration, pipeline parallelism, and cost modeling.

// Provides memory layout computation, stride optimization,
// tensor views, coalescing, alignment, and pinned memory management.
pub mod memory_defrag;
pub mod memory_layout;
// OpenAI-compatible API server with SSE streaming, auth, health checks,
// and Prometheus metrics for GPU-accelerated inference.
pub mod api_server;
pub mod ffi_safety;
pub mod weight_loader;

pub mod api_gateway;
