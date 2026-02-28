# Migration Guide: v0.1.x → v0.2.0

This guide covers breaking changes, new features, and the steps required to
upgrade from v0.1.x to v0.2.0.

## Breaking Changes

### Device Enum Expansion

The `Device` enum now includes variants for additional GPU backends:

```rust
// v0.1.x
enum Device {
    Cpu,
    Cuda(usize),
}

// v0.2.0
enum Device {
    Cpu,
    Cuda(usize),
    OpenCL(usize),
    Vulkan(usize),
    Rocm(usize),
    Metal(usize),
    WebGpu(usize),
}
```

**Action required:** Add a wildcard arm or handle the new variants in every
`match` on `Device`.

### New Feature Flags

v0.2.0 introduces per-backend feature flags. The existing `gpu` umbrella feature
still works, but you can now select individual backends:

| Feature   | Backend              |
|-----------|----------------------|
| `oneapi`  | Intel oneAPI / Level Zero |
| `opencl`  | OpenCL (Intel Arc, etc.) |
| `vulkan`  | Vulkan compute shaders   |
| `webgpu`  | WebGPU / wgpu            |
| `rocm`    | AMD ROCm / HIP           |
| `metal`   | Apple Metal              |

**Action required:** Update your `Cargo.toml` dependency to enable the
backend(s) you need. The `gpu` feature continues to activate the CUDA path.

### KernelBackend Expansion

`KernelBackend` has new variants matching the GPU backends above. If you
implement or match on `KernelBackend`, update accordingly.

### BackendStartupSummary

`BackendStartupSummary` now includes fields for the selected GPU backend and
detected capabilities. Code that destructures or serializes the summary must
account for the new fields.

## New Features

### Intel Arc GPU Support

v0.2.0 adds first-class Intel Arc GPU support via the OpenCL backend. Enable
with `--features opencl` and pass `--device opencl:0` at runtime.

### Multi-Backend Dispatch

The kernel manager can now dispatch to any compiled backend at runtime based on
device availability and user preference. The selection order is:

1. Explicitly requested backend (`--device`)
2. Best available (CUDA > OpenCL > Vulkan > CPU fallback)

### GPU Telemetry & Metrics

New telemetry hooks expose per-kernel timing, memory utilization, and queue
depth. Enable with `BITNET_GPU_TELEMETRY=1`.

### Paged KV Cache

The KV cache now supports paged allocation, reducing memory fragmentation for
long-context inference.

### GPU-Accelerated Sampling

Top-k, top-p, and temperature sampling can run on-device when a GPU backend is
active, avoiding a round-trip to host memory.

## Upgrade Steps

### 1. Update `Cargo.toml`

```toml
[dependencies]
bitnet = { version = "0.2", features = ["cpu"] }
# or, for Intel GPU support:
bitnet = { version = "0.2", features = ["cpu", "opencl"] }
```

### 2. Handle New `Device` Variants

Search your code for `match` expressions on `Device` and add arms for the new
variants (or use a wildcard):

```rust
match device {
    Device::Cpu => { /* ... */ }
    Device::Cuda(id) => { /* ... */ }
    _ => return Err(Error::UnsupportedDevice(device)),
}
```

### 3. Update Backend Configuration

If you configure backends programmatically, review `BackendStartupSummary` and
`KernelBackend` for new fields and variants. Serialization formats (JSON
receipts, telemetry logs) have additional keys.

### 4. Review Feature Gates in CI

Ensure your CI matrix covers the new feature flags if you test GPU paths:

```yaml
strategy:
  matrix:
    features: ["cpu", "gpu", "opencl", "vulkan"]
```
