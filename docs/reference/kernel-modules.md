# Kernel Module Reference

Complete reference for all kernel modules in `bitnet-kernels` (58 source files in `src/`, plus backend subdirectories).

## CPU Modules (`cpu/` — 66 files)

### Generic (architecture-independent)

| Module | Purpose |
|--------|---------|
| `activations.rs` | Activation functions (GELU, SiLU, ReLU) |
| `attention.rs` | Multi-head / grouped-query attention |
| `attention_mask.rs` | Causal and padding mask generation |
| `batch.rs` | Batch-level utilities |
| `batch_norm.rs` / `batch_normalization.rs` | Batch normalization |
| `cache_matmul.rs` | KV-cache-aware matrix multiply |
| `concat.rs` | Tensor concatenation |
| `conv2d.rs` / `convolution.rs` | 2-D convolution kernels |
| `dequant.rs` | Dequantization (I2_S → float) |
| `elementwise_ops.rs` | Element-wise add, mul, etc. |
| `embedding.rs` | Token embedding lookup |
| `fallback.rs` | Scalar fallback implementations |
| `ffn.rs` | Feed-forward network (SwiGLU) |
| `fusion.rs` / `layer_fusion.rs` | Fused attention + FFN passes |
| `gather.rs` / `scatter_gather.rs` | Memory-layout-aware data movement |
| `gating.rs` | Gating mechanisms |
| `kv_cache.rs` | KV-cache append and eviction |
| `layer_norm.rs` | LayerNorm / RMSNorm |
| `linear.rs` | Dense linear layers |
| `loss.rs` | Loss computation |
| `matrix_ops.rs` | Generic matrix operations |
| `pipeline_parallel.rs` | Pipeline parallelism |
| `pooling.rs` | Pooling operations |
| `quantize.rs` | Float → quantized encoding |
| `quantized_matmul.rs` | Quantized GEMV/GEMM |
| `quantized_pipeline.rs` | End-to-end quantized inference pipeline |
| `reduction.rs` | Sum, max, argmax reductions |
| `residual.rs` | Residual connections |
| `rope.rs` / `rope_simd.rs` | RoPE positional embeddings (scalar + SIMD) |
| `simd_math.rs` / `simd_matmul.rs` | SIMD math helpers and matmul |
| `softmax.rs` | Numerically-stable softmax |
| `tensor_parallel.rs` | Tensor parallelism |
| `transpose.rs` | Matrix transpose |

### x86 SIMD (AVX2 / AVX-512)

| Module | Feature gate | Purpose |
|--------|-------------|---------|
| `x86.rs` | `feature = "avx2"` or `feature = "avx512"` | AVX2/AVX-512 GEMV and dequantization |
| `x86_qk256_property_tests.rs` | `feature = "avx2"` | Property-based correctness tests for QK256 AVX2 path |

### ARM NEON (24 modules)

All NEON modules are gated on `target_arch = "aarch64"` and `feature = "neon"`.

| Module | Purpose |
|--------|---------|
| `arm.rs` | ARM architecture detection |
| `neon_activations.rs` / `neon_activation_suite.rs` | NEON-accelerated activations |
| `neon_batch_norm.rs` / `neon_batch_norm_v2.rs` | NEON batch normalization |
| `neon_batch_scheduler.rs` | NEON batch scheduling |
| `neon_convolution.rs` | NEON convolution |
| `neon_data_layout.rs` | NEON data layout transforms |
| `neon_elementwise.rs` | NEON element-wise ops |
| `neon_inference_bridge.rs` | NEON ↔ inference engine bridge |
| `neon_kv_cache.rs` | NEON KV-cache operations |
| `neon_layernorm.rs` | NEON LayerNorm |
| `neon_multi_head_linear.rs` | NEON multi-head linear projection |
| `neon_online_softmax.rs` | NEON online (streaming) softmax |
| `neon_padding_clipping.rs` | NEON padding and clipping |
| `neon_pooling.rs` | NEON pooling |
| `neon_quantized_gemm.rs` / `neon_quantized_matmul.rs` | NEON quantized GEMM/GEMV |
| `neon_reductions.rs` | NEON reductions |
| `neon_rope.rs` | NEON RoPE |
| `neon_scatter_gather.rs` | NEON scatter/gather |
| `neon_sliding_window_attention.rs` | NEON sliding-window attention |
| `neon_softmax.rs` | NEON softmax |
| `neon_transpose.rs` | NEON transpose |

## CUDA Modules (`cuda/` — 39 files)

Feature gate: `#[cfg(any(feature = "gpu", feature = "cuda"))]`

### Core Compute

| Module | Purpose |
|--------|---------|
| `matmul.rs` | Dense matrix multiply |
| `softmax.rs` | GPU softmax |
| `layernorm.rs` / `rmsnorm.rs` | LayerNorm and RMSNorm |
| `attention.rs` / `fused_attention.rs` / `multi_head_attention.rs` | Attention variants |
| `ffn.rs` | Feed-forward network |
| `embedding.rs` / `embedding_ops.rs` | Embedding lookup |
| `rope.rs` | RoPE positional embeddings |
| `gating.rs` | Gating mechanisms |
| `linear.rs` | Dense linear layers |
| `dequant.rs` / `quantize.rs` | Quantization/dequantization |
| `elementwise.rs` | Element-wise ops |
| `batch_norm.rs` | Batch normalization |
| `conv1d.rs` | 1-D convolution |
| `pooling.rs` | Pooling |
| `residual.rs` | Residual connections |
| `loss.rs` | Loss computation |
| `transpose.rs` | Transpose |
| `fusion.rs` | Fused operations |
| `sparse.rs` | Sparse tensor operations |

### Quantized Compute

| Module | Purpose |
|--------|---------|
| `quantized_gemm.rs` | Quantized general matrix multiply |
| `quantized_matmul.rs` | Quantized matmul variants |
| `qk256_gemv.rs` | QK256 2-bit GEMV kernel |

### GPU Infrastructure

| Module | Purpose |
|--------|---------|
| `memory_pool.rs` | Device memory pooling and allocation |
| `stream_mgmt.rs` | CUDA stream creation and synchronization |
| `warp_ops.rs` | Warp-level primitives (shuffle, reduce) |
| `cooperative_groups.rs` | Cooperative group operations |
| `graph_exec.rs` | CUDA graph capture and replay |
| `shader_cache.rs` | Compiled kernel caching |
| `profiling.rs` | Kernel timing and profiling |
| `kv_cache.rs` / `kv_cache_gpu.rs` | GPU KV-cache management |

## OpenCL Modules (42 top-level files)

Experimental Intel Arc backend. All modules are at `bitnet-kernels/src/opencl_*.rs`.

### Compute

| Module | Purpose |
|--------|---------|
| `opencl_attention.rs` / `opencl_flash_attention.rs` / `opencl_gqa.rs` | Attention kernels |
| `opencl_ffn.rs` | Feed-forward network |
| `opencl_layer_norm.rs` | LayerNorm |
| `opencl_reductions.rs` | Reduction operations |
| `opencl_softmax_variants.rs` | Softmax variants |
| `opencl_elementwise.rs` | Element-wise operations |
| `opencl_embedding.rs` / `opencl_token_embed.rs` | Embedding lookup |
| `opencl_quantized.rs` / `opencl_quantized_matmul.rs` / `opencl_matmul_variants.rs` | Quantized compute |
| `opencl_mixed_precision.rs` | Mixed-precision operations |

### Infrastructure

| Module | Purpose |
|--------|---------|
| `opencl_context.rs` | OpenCL context management |
| `opencl_cmd_queue.rs` | Command queue management |
| `opencl_buffer.rs` / `opencl_memory.rs` | Buffer and memory management |
| `opencl_device_caps.rs` | Device capability detection |
| `opencl_work_size.rs` | Work-size optimization (Intel Arc tuned) |
| `opencl_kernel_sources.rs` | Built-in kernel source registry |
| `opencl_program_cache.rs` | Compiled program caching |
| `opencl_registry.rs` | Kernel registry |

### Pipeline

| Module | Purpose |
|--------|---------|
| `opencl_pipeline.rs` | Inference pipeline |
| `opencl_continuous_batch.rs` | Continuous batching |
| `opencl_graph_compiler.rs` | Graph compilation |
| `opencl_layer_compose.rs` | Layer composition |
| `opencl_transformer.rs` | Full transformer forward pass |
| `opencl_engine_bridge.rs` | Engine integration |
| `opencl_model_converter.rs` | Model format conversion |
| `opencl_token_gen.rs` | Token generation |

### Caching and Utilities

| Module | Purpose |
|--------|---------|
| `opencl_cache.rs` / `opencl_prefix_cache.rs` | General and prefix caching |
| `opencl_kv_cache.rs` / `opencl_rope_cache.rs` | KV-cache and RoPE caching |
| `opencl_autotuner.rs` | Auto-tuning kernel parameters |
| `opencl_profiling.rs` / `opencl_telemetry.rs` | Profiling and telemetry |
| `opencl_async_executor.rs` | Async kernel execution |
| `opencl_numerical_stability.rs` | Numerical stability guards |
| `opencl_weight_manager.rs` | Weight loading and management |

## Other Backend Modules

| Backend | Location | Feature gate | Modules |
|---------|----------|-------------|---------|
| Metal | `metal_compute.rs` | `feature = "metal"` | Apple Metal compute |
| ROCm | `rocm/` (4 files) | `feature = "rocm"` | `attention`, `qk256_gemv`, `rmsnorm` |
| NPU | `npu/` (2 files) | `feature = "npu-backend"` | C++ bridge (`bridge.rs`, `cpp_bridge.cpp`) |
| Mixed GPU | `gpu/` (14 files) | `feature = "gpu"` | Shared GPU utils, OpenCL dispatch, validation, benchmarks, SPIR-V cache |

## Shared Infrastructure

| Module | Purpose |
|--------|---------|
| `lib.rs` | Crate root, module declarations, `KernelProvider` trait, `KernelManager` |
| `kernels.rs` | Runtime kernel provider selection (CUDA > CPU fallback) |
| `capability_matrix.rs` | Per-backend capability reporting |
| `device_aware.rs` | Device-aware kernel dispatch |
| `device_features.rs` | `gpu_compiled()`, `gpu_available_runtime()` helpers |
| `convolution.rs` | Generic convolution |
| `reduction.rs` / `shaped_reduction.rs` | Reduction primitives |
| `scatter_gather.rs` | Top-level scatter/gather |
| `tl_lut.rs` | Table-lookup LUT generation (TL1/TL2) |
| `simd_diagnostics.rs` | SIMD feature detection diagnostics |
| `perf_tracker.rs` | Kernel performance tracking |
| `gpu_utils.rs` | GPU utility helpers |
| `stubs.rs` | Stub implementations for disabled backends |
| `ffi.rs` | C++ FFI bridge (feature `ffi`) |
| `benchmarks/` | Kernel micro-benchmarks |

## Feature Gate Summary

| Feature | Enables |
|---------|---------|
| `cpu` | CPU kernels, SIMD paths |
| `gpu` | CUDA + shared GPU modules |
| `cuda` | Alias for `gpu` (backward compat) |
| `avx2` | x86 AVX2 SIMD kernels |
| `avx512` | x86 AVX-512 SIMD kernels |
| `neon` | ARM NEON SIMD kernels |
| `metal` | Apple Metal backend |
| `vulkan` | Vulkan compute backend |
| `rocm` | AMD ROCm backend |
| `opencl` | Intel Arc OpenCL backend |
| `oneapi` | Intel oneAPI backend |
| `npu-backend` | NPU C++ bridge |
| `ffi` | C++ FFI bridge |

Always use the unified GPU predicate in code:

```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
```
