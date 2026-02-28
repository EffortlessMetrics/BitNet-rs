// GPU hardware abstraction layer for `BitNet` inference.

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
pub mod deployment_manager;
pub mod device_abstraction;
pub mod embedding_layer;
pub mod error_taxonomy;
pub mod hal_traits;

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
pub mod rope_kernels;
pub mod transformer_block;

// === Inference Pipeline ===
pub mod autoregressive_generator;
pub mod context_window;
pub mod dynamic_batching;
pub mod inference_pipeline;
pub mod kv_cache_manager;
pub mod sampling_strategies;

// === Quantization ===
pub mod mixed_precision;
pub mod model_quantizer;
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
pub mod model_debugger;

// === Testing & Validation ===
pub mod compatibility_checker;
pub mod e2e_integration;
pub mod model_validator;
pub mod test_harness;

// === Infrastructure ===
pub mod arch_registry;
pub mod config_management;
pub mod logging;
pub mod migration_tool;
pub mod thread_pool;

// === Distributed ===
pub mod distributed_inference;
pub mod multi_device;
pub mod parallel_communication;

// === Server & Serving ===
pub mod cache_system;
pub mod inference_scheduler;
pub mod server_protocol;
pub mod serving_runtime;

// === ML Operations ===
pub mod gradient_checkpoint;
pub mod instruction_tuning;
pub mod model_hub;
pub mod model_pruning;

// === SPIR-V ===
pub mod perf_comparison;
pub mod spirv_compiler;

// === Docker/CI ===
pub mod docker_ci;

// === Existing Modules (prior waves) ===

// Provides checkpoint management for saving and resuming inference state,
// with incremental diffs, compression, and automatic scheduling.
pub mod checkpoint_manager;
// Provides batched tokenization, parallel encoding/decoding,
// and hardware abstraction for GPU-accelerated inference pipelines.
pub mod batched_tokenization;
pub mod prompt_processing;
pub mod streaming_aggregator;
// Parallel communication primitives for distributed GPU inference:
// all-reduce, all-gather, reduce-scatter, broadcast, ring/tree
// topologies, double-buffered comm, and profiling.
// Structured error taxonomy for GPU HAL with rich context,
// recovery strategies, and structured reporting.
pub mod generation;
pub mod model_warmup;
