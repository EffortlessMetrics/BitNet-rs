# CUDA → OpenCL Kernel Migration Guide

A systematic guide for porting BitNet-rs GPU kernels from CUDA to OpenCL,
targeting Intel Arc GPUs via the `opencl3` crate.

## Table of Contents

1. [Concept Mapping](#concept-mapping)
2. [Type Mapping](#type-mapping)
3. [API Mapping](#api-mapping)
4. [Memory Model Differences](#memory-model-differences)
5. [Kernel Language Mapping](#kernel-language-mapping)
6. [Common Pitfalls](#common-pitfalls)
7. [Full Migration Walkthrough](#full-migration-walkthrough)
8. [Performance Comparison Methodology](#performance-comparison-methodology)
9. [BitNet-rs Specific Patterns](#bitnet-rs-specific-patterns)

---

## Concept Mapping

| CUDA Concept | OpenCL Equivalent | BitNet-rs Crate |
|---|---|---|
| CUDA Context (`CudaContext`) | OpenCL Context (`Context`) | `bitnet-kernels::gpu::cuda` → `opencl` |
| CUDA Stream (`CudaStream`) | Command Queue (`CommandQueue`) | Same module |
| CUDA Module (`CudaModule`) | Program (`Program`) | Same module |
| CUDA Function (`CudaFunction`) | Kernel (`Kernel`) | Same module |
| Device memory (`CudaSlice<T>`) | Buffer (`Buffer<T>`) | Same module |
| PTX (compiled IR) | SPIR-V or CL source | Build-time vs runtime compilation |
| `.cu` kernel source | `.cl` kernel source | `kernels/` subdirectory |
| Grid / Block dimensions | Global / Local work size | Launch configuration |
| `__shared__` memory | `__local` memory | Kernel source |
| `threadIdx.x` | `get_local_id(0)` | Kernel source |
| `blockIdx.x` | `get_group_id(0)` | Kernel source |
| `blockDim.x` | `get_local_size(0)` | Kernel source |
| `__syncthreads()` | `barrier(CLK_LOCAL_MEM_FENCE)` | Kernel source |
| Warp (32 threads) | Sub-group (varies, typically 8–32) | Hardware-dependent |
| `nvrtc` (runtime compilation) | `Program::create_from_source` | Runtime |
| `nvidia-smi` | `clinfo` | Detection tools |

## Type Mapping

### Host-Side Rust Types

| CUDA (cudarc) | OpenCL (opencl3) | Notes |
|---|---|---|
| `Arc<CudaContext>` | `Context` | OpenCL `Context` is already ref-counted |
| `Arc<CudaStream>` | `CommandQueue` | Must enable `CL_QUEUE_PROFILING_ENABLE` for timing |
| `Arc<CudaModule>` | `Program` | Built from source string, not PTX |
| `CudaFunction` | `Kernel` (from `Program`) | Extracted per entry point |
| `CudaSlice<f32>` | `Buffer<cl_float>` | Use `CL_MEM_READ_ONLY` / `CL_MEM_WRITE_ONLY` |
| `LaunchConfig` | `(global_work_size, local_work_size)` tuple | See launch config section |
| `PushKernelArg` | `kernel.set_arg(index, &value)` | Explicit index-based |

### Kernel-Side Types

| CUDA | OpenCL | Notes |
|---|---|---|
| `float` | `float` | Same |
| `int` | `int` | Same |
| `char` / `int8_t` | `char` | Signed 8-bit |
| `unsigned char` | `uchar` | Unsigned 8-bit |
| `half` (FP16) | `half` | Requires `cl_khr_fp16` extension |
| `__half2` | `half2` | Requires `cl_khr_fp16` extension |

## API Mapping

### Device Initialization

**CUDA:**
```rust
use cudarc::driver::CudaContext;

let ctx = CudaContext::new(device_id)?;
let stream = ctx.default_stream();
```

**OpenCL:**
```rust
use opencl3::platform::get_platforms;
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::context::Context;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};

let platforms = get_platforms()?;
let device_ids = platforms[0].get_devices(CL_DEVICE_TYPE_GPU)?;
let device = Device::new(device_ids[0]);
let context = Context::from_device(&device)?;
let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)?;
```

### Kernel Compilation

**CUDA:**
```rust
use cudarc::nvrtc::compile_ptx;

let ptx = compile_ptx(include_str!("kernels/bitnet_kernels.cu"))?;
let module = ctx.load_module(ptx)?;
let function = module.get_func("matmul_i2s")?;
```

**OpenCL:**
```rust
use opencl3::program::Program;
use opencl3::kernel::Kernel;

let source = include_str!("kernels/bitnet_kernels.cl");
let program = Program::create_and_build_from_source(&context, source, "")?;
let kernel = Kernel::create(&program, "matmul_i2s")?;
```

### Memory Allocation and Transfer

**CUDA:**
```rust
// Host → Device
let d_input = ctx.alloc_zeros::<f32>(n)?;
ctx.memcpy_htod(&d_input, &host_data)?;

// Device → Host
let mut output = vec![0f32; n];
ctx.memcpy_dtoh(&mut output, &d_output)?;
```

**OpenCL:**
```rust
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::types::CL_BLOCKING;

// Host → Device
let d_input = unsafe {
    Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, n, std::ptr::null_mut())?
};
unsafe {
    queue.enqueue_write_buffer(&d_input, CL_BLOCKING, 0, &host_data, &[])?;
};

// Device → Host
let mut output = vec![0f32; n];
unsafe {
    queue.enqueue_read_buffer(&d_output, CL_BLOCKING, 0, &mut output, &[])?;
};
```

### Kernel Launch

**CUDA:**
```rust
let config = LaunchConfig {
    grid_dim: (grid_x, grid_y, 1),
    block_dim: (block_x, block_y, 1),
    shared_mem_bytes: 0,
};
unsafe { function.launch(config, (&d_a, &d_b, &d_c, m, n, k)) }?;
```

**OpenCL:**
```rust
let kernel = Kernel::create(&program, "matmul_i2s")?;
kernel.set_arg(0, &d_a)?;
kernel.set_arg(1, &d_b)?;
kernel.set_arg(2, &d_c)?;
kernel.set_arg(3, &(m as cl_int))?;
kernel.set_arg(4, &(n as cl_int))?;
kernel.set_arg(5, &(k as cl_int))?;

let global_work_size = [grid_x * block_x, grid_y * block_y];
let local_work_size = [block_x, block_y];
unsafe {
    queue.enqueue_nd_range_kernel(&kernel, 2, None, &global_work_size, Some(&local_work_size), &[])?;
};
queue.finish()?;
```

## Memory Model Differences

### Key Differences

| Aspect | CUDA | OpenCL |
|---|---|---|
| Unified memory | `cudaMallocManaged` | SVM (`clSVMAlloc`) — limited support |
| Shared memory size | Fixed at launch | `kernel.set_arg_local_buffer(idx, size)` |
| Memory ordering | Relaxed within warp | Explicit fences required |
| Pinned memory | `cudaMallocHost` | `CL_MEM_ALLOC_HOST_PTR` |
| Texture memory | Texture objects | Image objects (less common for compute) |

### Synchronization

**CUDA:** Implicit within a warp (32 threads execute in lockstep).

**OpenCL:** Sub-group size varies by vendor (Intel Arc: typically 8, 16, or 32).
Always use explicit barriers:

```opencl
// CUDA: __syncthreads()
// OpenCL equivalent:
barrier(CLK_LOCAL_MEM_FENCE);
```

## Kernel Language Mapping

### Function Qualifiers

| CUDA | OpenCL | Purpose |
|---|---|---|
| `__global__` | `__kernel` | Entry point |
| `__device__` | (none needed) | Helper functions are implicitly device |
| `__host__` | N/A | CPU-only (not in kernel source) |
| `__constant__` | `__constant` | Constant memory |
| `__shared__` | `__local` | Workgroup-shared memory |

### Built-in Variables

| CUDA | OpenCL |
|---|---|
| `threadIdx.x` | `get_local_id(0)` |
| `threadIdx.y` | `get_local_id(1)` |
| `blockIdx.x` | `get_group_id(0)` |
| `blockIdx.y` | `get_group_id(1)` |
| `blockDim.x` | `get_local_size(0)` |
| `gridDim.x` | `get_num_groups(0)` |

### Example: Matrix Multiplication Kernel

**CUDA (`bitnet_kernels.cu`):**
```cuda
__global__ void matmul_i2s(
    const int8_t* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += (float)weights[row * K + k] * input[k * N + col];
        }
        output[row * N + col] = sum;
    }
}
```

**OpenCL (`bitnet_kernels.cl`):**
```opencl
__kernel void matmul_i2s(
    __global const char* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    int M, int N, int K
) {
    int row = get_group_id(1) * get_local_size(1) + get_local_id(1);
    int col = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += (float)weights[row * K + k] * input[k * N + col];
        }
        output[row * N + col] = sum;
    }
}
```

**Changes required:**
1. `__global__` → `__kernel`
2. `const int8_t*` → `__global const char*`
3. `const float*` → `__global const float*`
4. `__restrict__` → `restrict`
5. `blockIdx/threadIdx/blockDim` → `get_group_id/get_local_id/get_local_size`

## Common Pitfalls

### 1. Sub-group Size Assumptions

**Problem:** CUDA code assumes warp size = 32 for warp-level reductions.
Intel Arc sub-group sizes are typically 8, 16, or 32.

**Solution:** Query sub-group size at runtime:
```opencl
// In kernel:
uint sg_size = get_sub_group_size();

// In host (OpenCL 2.1+):
// Check CL_DEVICE_SUB_GROUP_SIZES_INTEL extension
```

### 2. Local Memory Size Limits

**Problem:** CUDA shared memory is typically 48 KB per block (configurable to 96 KB).
Intel Arc local memory is typically 64 KB per workgroup.

**Solution:** Check `CL_DEVICE_LOCAL_MEM_SIZE` and adapt tile sizes accordingly.

### 3. Work-Group Size Constraints

**Problem:** CUDA maximum block size is 1024 threads per block.
OpenCL maximum work-group size varies by device.

**Solution:** Query `CL_DEVICE_MAX_WORK_GROUP_SIZE` and `CL_KERNEL_WORK_GROUP_SIZE`.

### 4. Missing `__syncthreads()` Equivalent Scope

**Problem:** `barrier()` in OpenCL only synchronizes within a work-group,
same as `__syncthreads()`. But cross-workgroup synchronization requires
splitting into separate kernel dispatches.

**Solution:** Same as CUDA — no cross-block synchronization within a kernel.
Design algorithms to avoid it.

### 5. Half-Precision (FP16) Support

**Problem:** CUDA natively supports `__half` and `__half2` types.
OpenCL FP16 requires the `cl_khr_fp16` extension.

**Solution:**
```opencl
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// Then use half, half2, etc.
```

Intel Arc GPUs support this extension.

### 6. Atomic Operations

**Problem:** CUDA has native `atomicAdd(float*)`.
OpenCL `atomic_add` only works on integers by default.

**Solution:** Use `cl_ext_float_atomics` extension on Intel GPUs, or
implement a CAS-based float atomic:
```opencl
void atomicAdd_f(__global float* addr, float val) {
    union { unsigned int u; float f; } prev, curr;
    do {
        prev.f = *addr;
        curr.f = prev.f + val;
    } while (atomic_cmpxchg((__global unsigned int*)addr, prev.u, curr.u) != prev.u);
}
```

### 7. Error Handling Differences

**Problem:** CUDA errors are typically caught via `Result` from `cudarc`.
OpenCL uses integer error codes.

**Solution:** The `opencl3` crate wraps error codes into Rust `Result` types,
similar to `cudarc`. Use `?` propagation as normal:
```rust
let program = Program::create_and_build_from_source(&context, source, "")
    .map_err(|e| KernelError::GpuError {
        reason: format!("OpenCL program build failed: {:?}", e),
    })?;
```

## Full Migration Walkthrough

This section walks through migrating the BitNet-rs `CudaKernel` to `OpenClKernel`.

### Step 1: Create the OpenCL Kernel Source

Convert `kernels/bitnet_kernels.cu` → `kernels/bitnet_kernels.cl`:

- Replace all CUDA qualifiers with OpenCL equivalents (see table above)
- Replace thread indexing functions
- Add `__global` qualifiers to all pointer parameters
- Replace `__restrict__` with `restrict`
- Add extension pragmas for FP16 if needed

### Step 2: Create the Rust Host Module

The structure mirrors `cuda.rs`:

```rust
// opencl.rs — already exists in bitnet-kernels/src/gpu/opencl.rs
pub struct OpenClKernel {
    platform_name: String,
    device_name: String,
    ready: bool,
    context: Option<OpenClContext>,
}

struct OpenClContext {
    context: Context,
    queue: CommandQueue,
    matmul_program: Option<Program>,
}
```

### Step 3: Implement `KernelProvider` Trait

Map each CUDA operation to its OpenCL equivalent:

| `KernelProvider` method | CUDA impl | OpenCL impl |
|---|---|---|
| `matmul` | Launch `matmul_i2s` kernel | Launch `matmul_i2s` kernel |
| `quantize` | Launch quantize kernels | Launch quantize kernels |
| `name()` | `"cuda"` | `"opencl"` |
| `is_available()` | `nvidia-smi` check | `clinfo` check |

### Step 4: Wire into `KernelManager`

Add an OpenCL provider to the kernel selection priority chain:

```rust
// In bitnet-kernels/src/kernels.rs
// CUDA > OpenCL > CPU fallback
if cfg!(any(feature = "gpu", feature = "cuda")) {
    if let Ok(cuda) = CudaKernel::new() {
        providers.push(Box::new(cuda));
    }
}
if cfg!(feature = "oneapi") {
    if let Ok(opencl) = OpenClKernel::new() {
        providers.push(Box::new(opencl));
    }
}
providers.push(Box::new(FallbackKernel::new()));
```

### Step 5: Test and Validate

1. **Unit tests:** Run with `BITNET_GPU_FAKE=oneapi` for CI without hardware
2. **Integration tests:** Verify numerical parity against CPU reference
3. **Performance tests:** Compare throughput using the methodology below

## Performance Comparison Methodology

When comparing CUDA and OpenCL kernel performance:

### Metrics to Capture

| Metric | Tool | Unit |
|---|---|---|
| Kernel execution time | OpenCL profiling / `nsys` | microseconds |
| Memory bandwidth | Transfer size / time | GB/s |
| Throughput | Tokens / second | tok/s |
| Memory utilization | `nvidia-smi` / `clinfo` | MB / % |

### Benchmarking Protocol

1. **Warm-up:** Run 10 iterations before measuring (JIT compilation, cache warm-up)
2. **Measurement:** Run 100 iterations, report median and p95
3. **Comparison:** Run identical workloads on both backends
4. **Workloads:** Use the standard benchmark dimensions from `benches/srp_ops.rs`

### OpenCL Profiling Setup

```rust
// Enable profiling on queue creation
let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)?;

// After kernel dispatch, read timing events
let event = /* event from enqueue_nd_range_kernel */;
let start = event.profiling_command_start()?;
let end = event.profiling_command_end()?;
let elapsed_ns = end - start;
```

### Expected Performance Characteristics

| Operation | CUDA (A100) | OpenCL (Arc A770) | Notes |
|---|---|---|---|
| matmul_i2s (2B model) | Baseline | ~0.6–0.8× | Memory bandwidth limited |
| quantize_tl1 | Baseline | ~0.5–0.7× | Compute bound |
| Memory transfer H→D | ~30 GB/s | ~24 GB/s | PCIe 4.0 limited |

> These are rough estimates. Actual numbers depend on model size, batch size,
> and driver maturity. Always measure with your specific hardware.

## BitNet-rs Specific Patterns

### Feature Gate Convention

All GPU code must use the unified predicate — never `#[cfg(feature = "cuda")]` alone:

```rust
// ✅ Correct — covers both `gpu` and `cuda` features
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn cuda_function() { ... }

// ✅ Correct — OpenCL-specific
#[cfg(feature = "oneapi")]
pub fn opencl_function() { ... }

// ❌ Wrong — misses the `gpu` umbrella feature
#[cfg(feature = "cuda")]
pub fn bad_cuda_function() { ... }
```

### Error Wrapping Convention

All GPU errors must be wrapped in `KernelError::GpuError`:

```rust
use bitnet_common::KernelError;

some_opencl_call()
    .map_err(|e| KernelError::GpuError {
        reason: format!("OpenCL operation failed: {:?}", e),
    })?;
```

### Device Detection

Use `bitnet-device-probe` for runtime detection:

```rust
use bitnet_device_probe::{probe_device, gpu_compiled, oneapi_compiled};

let probe = probe_device();
if probe.oneapi_available {
    // Intel GPU available, initialize OpenCL kernel
}
```

### Testing with GPU Fakes

For CI testing without hardware:

```rust
#[test]
#[serial_test::serial(bitnet_env)]
fn test_opencl_kernel_path() {
    temp_env::with_var("BITNET_GPU_FAKE", Some("oneapi"), || {
        assert!(bitnet_device_probe::oneapi_available_runtime());
    });
}
```
