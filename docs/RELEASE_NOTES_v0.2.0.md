# BitNet-rs v0.2.0-alpha.1 Release Notes

## Multi-GPU Backend Support

This release introduces comprehensive multi-GPU backend support for BitNet-rs,
enabling inference on Intel Arc, AMD ROCm, Apple Metal, and WebGPU devices in
addition to the existing NVIDIA CUDA and CPU backends.

### New Features

#### GPU Backends
- **Intel Arc (OpenCL)**: Full inference support via `--features oneapi`
  - 20+ OpenCL compute kernels (matmul, attention, softmax, RoPE, etc.)
  - Flash Attention with O(N) memory
  - Ternary matmul optimization for 1-bit weights
  - Fused LayerNorm+Linear kernel
- **Vulkan**: Cross-vendor GPU compute via `--features vulkan` (experimental)
- **AMD ROCm**: HIP-based kernels via `--features rocm` (experimental)
- **Apple Metal**: MSL compute kernels via `--features metal` (experimental)
- **WebGPU**: Browser-compatible WGSL kernels via `--features webgpu` (experimental)

#### GPU Infrastructure
- Unified Hardware Abstraction Layer (bitnet-gpu-hal)
- Runtime backend discovery and selection
- GPU memory estimation and planning
- Kernel fusion and optimization passes
- Flash Attention, PagedAttention KV cache
- Multi-device scheduling and load balancing

#### Server Features
- Smart request routing to GPU backends
- Model registry with GPU affinity
- WebSocket streaming for real-time token output
- Canary deployment for GPU backends
- GPU-aware rate limiting

#### Testing & CI
- Mock GPU backend for CI testing without hardware
- Property tests for all kernel operations
- Cross-backend numerical validation
- Benchmark regression detection
- Feature flag compile matrix

### Breaking Changes
- `Device` enum now includes `OpenCL(usize)`, `Vulkan(usize)`, `Rocm(usize)` variants
- `KernelBackend` enum expanded with new variants

### Known Limitations
- GPU backends are experimental; CPU and CUDA remain the recommended production paths
- QK256 GPU kernels are scalar-only (no SIMD vectorization on GPU)
- Real hardware testing is limited to NVIDIA CUDA; other backends tested via mock only
- WebGPU and ROCm backends have minimal kernel coverage

### Migration Guide
See `docs/GPU_MIGRATION_GUIDE.md` for details on upgrading from v0.1.x.
