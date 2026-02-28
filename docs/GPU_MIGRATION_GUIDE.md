# Migration Guide: v0.1.x → v0.2.0

## Device Enum Changes

The `Device` enum in `bitnet-common` has new variants. If you match on `Device`,
add wildcard arms:

```rust
match device {
    Device::Cpu => { /* ... */ }
    Device::Cuda(id) => { /* ... */ }
    Device::Metal => { /* ... */ }
    // New in v0.2.0:
    Device::OpenCL(id) => { /* ... */ }
    Device::Vulkan(id) => { /* ... */ }
    Device::Rocm(id) => { /* ... */ }
    _ => { /* future-proof */ }
}
```

## Feature Flags

New feature flags available:
- `oneapi` — Intel GPU (OpenCL) support
- `vulkan` — Vulkan compute backend (experimental)
- `rocm` — AMD ROCm/HIP backend (experimental)
- `webgpu` — WebGPU/WGSL backend (experimental)

Existing flags (`cpu`, `gpu`, `full-cli`) are unchanged.

## CLI Changes

New `--device` options: `opencl`, `vulkan`, `rocm`, `auto`

```bash
# Auto-detect best device (default)
bitnet-cli run --device auto --model model.gguf

# Force OpenCL
bitnet-cli run --device opencl --model model.gguf
```
