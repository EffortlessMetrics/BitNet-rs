# GPU API Reference

> Complete public API reference for all bitnet-rs GPU backend crates.

Each section documents one crate, organised by module. For every public item
the signature, description, parameters, return value, and a usage example are
provided.

---

## Table of Contents

- [bitnet-opencl](#bitnet-opencl)
- [bitnet-vulkan](#bitnet-vulkan)
- [bitnet-rocm](#bitnet-rocm)
- [bitnet-metal](#bitnet-metal)
- [bitnet-webgpu](#bitnet-webgpu)
- [bitnet-level-zero](#bitnet-level-zero)

---

## bitnet-opencl

OpenCL 3.0 backend with WASM-compatible kernel validation, Unified Shared
Memory (USM) support, and GPU-to-GPU peer-to-peer transfers.

### Quick Start

```rust
use bitnet_opencl::usm::{detect_usm_capabilities, DataPath};

let caps = detect_usm_capabilities(raw_bitfield);
let path = DataPath::new(caps);
let device_bytes = path.upload(host_data)?;
```

---

### Module `wasm_shim`

WASM-compatible kernel validation — no FFI required.

#### `parse_kernel_signatures`

```rust
pub fn parse_kernel_signatures(source: &str) -> Vec<KernelSignature>
```

Parse OpenCL kernel source and extract all `__kernel` function signatures.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `&str` | OpenCL C kernel source code |

**Returns:** `Vec<KernelSignature>` — one entry per `__kernel` function found.

```rust
let sigs = parse_kernel_signatures("__kernel void matmul(__global float* A) {}");
assert_eq!(sigs[0].name, "matmul");
assert_eq!(sigs[0].args.len(), 1);
```

---

#### `validate_kernel_source`

```rust
pub fn validate_kernel_source(source: &str) -> Result<Vec<KernelSignature>, KernelValidationError>
```

Validate kernel source: checks for no-kernel, empty-arg, duplicate-name issues.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `&str` | OpenCL C kernel source |

**Returns:** `Result<Vec<KernelSignature>, KernelValidationError>`

---

#### `source_contains_kernel`

```rust
pub fn source_contains_kernel(source: &str) -> bool
```

Quick check whether source contains at least one `__kernel` function.

---

#### `struct KernelSignature`

A parsed OpenCL kernel function signature.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `String` | Kernel function name |
| `args` | `Vec<KernelArg>` | Parsed argument list |

---

#### `struct KernelArg`

A single kernel function argument.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `String` | Argument name |
| `qualifier` | `ArgQualifier` | Address-space qualifier |
| `type_name` | `String` | C type name (e.g. `"float"`) |
| `is_pointer` | `bool` | Whether the argument is a pointer |

---

#### `enum ArgQualifier`

OpenCL address-space qualifier.

| Variant | Description |
|---------|-------------|
| `Global` | `__global` memory |
| `Local` | `__local` (work-group shared) memory |
| `Constant` | `__constant` memory |
| `Private` | `__private` (per-item) memory |

---

#### `enum KernelValidationError`

| Variant | Description |
|---------|-------------|
| `NoKernelsFound` | No `__kernel` functions in source |
| `EmptyArgList` | A kernel has zero arguments |
| `ArgCountMismatch` | Expected vs parsed arg count differs |
| `DuplicateKernelName` | Two kernels share a name |

---

#### `struct MockOpenClContext`

Mock OpenCL context for testing kernel argument setup without FFI.

```rust
pub fn new() -> Self
pub fn compile_program(&mut self, name: &str, source: &str)
    -> Result<(), MockError>
pub fn kernel_names(&self, program_name: &str)
    -> Result<Vec<String>, MockError>
pub fn kernel_signature(&self, kernel_name: &str)
    -> Result<&KernelSignature, MockError>
pub fn set_kernel_arg(&mut self, kernel_name: &str, index: usize, value: MockArgValue)
    -> Result<(), MockError>
pub fn all_args_set(&self, kernel_name: &str)
    -> Result<bool, MockError>
pub fn program_count(&self) -> usize
```

```rust
let mut ctx = MockOpenClContext::new();
ctx.compile_program("prog", KERNEL_SRC)?;
let names = ctx.kernel_names("prog")?;
ctx.set_kernel_arg(&names[0], 0, MockArgValue::Buffer(42))?;
assert!(ctx.all_args_set(&names[0])?);
```

---

#### `enum MockArgValue`

Mock kernel argument values for testing.

#### `enum MockError`

Errors from mock context operations (program not found, kernel not found, etc.).

---

### Module `usm`

OpenCL 3.0 Unified Shared Memory support with fallback to explicit copies.

#### `struct SvmCapabilities`

Bitflags representing OpenCL SVM capabilities.

| Constant | Value | Description |
|----------|-------|-------------|
| `COARSE_GRAIN_BUFFER` | `1 << 0` | Coarse-grain buffer SVM |
| `FINE_GRAIN_BUFFER` | `1 << 1` | Fine-grain buffer SVM |
| `FINE_GRAIN_SYSTEM` | `1 << 2` | Fine-grain system SVM |
| `ATOMICS` | `1 << 3` | SVM atomics |
| `NONE` | `0` | No SVM support |

```rust
pub const fn from_raw(bits: u64) -> Self
pub const fn bits(self) -> u64
pub const fn contains(self, flag: Self) -> bool
pub const fn supports_usm(self) -> bool
pub const fn supports_zero_copy(self) -> bool
```

```rust
let caps = SvmCapabilities::from_raw(0b0110); // fine-grain buffer + system
assert!(caps.supports_usm());
assert!(caps.supports_zero_copy());
```

---

#### `detect_usm_capabilities`

```rust
pub fn detect_usm_capabilities(raw_device_caps: u64) -> SvmCapabilities
```

Detect SVM capabilities from raw device bitfield.

---

#### `enum TransferMode`

| Variant | Description |
|---------|-------------|
| `Unified` | Zero-copy shared address space |
| `ExplicitCopy` | Requires `clEnqueueRead/WriteBuffer` |

```rust
pub fn select(caps: SvmCapabilities) -> Self
```

---

#### `struct UsmAllocator`

Zero-copy USM allocator wrapping `clSVMAlloc` / `clSVMFree`.

```rust
pub fn new(capabilities: SvmCapabilities) -> Self
pub fn capabilities(&self) -> SvmCapabilities
pub fn transfer_mode(&self) -> TransferMode
pub fn allocated_bytes(&self) -> usize
pub fn alloc(&self, layout: Layout) -> Result<UsmAllocation, UsmError>
pub unsafe fn free(&self, allocation: UsmAllocation) -> Result<(), UsmError>
```

---

#### `struct UsmAllocation`

Handle to a USM (SVM) allocation.

```rust
pub fn as_ptr(&self) -> *mut u8
pub fn layout(&self) -> Layout
pub fn size(&self) -> usize
```

---

#### `enum UsmError`

| Variant | Description |
|---------|-------------|
| `UsmNotSupported` | Device lacks SVM/USM |
| `AllocationFailed` | `clSVMAlloc` returned null |
| `NullPointer` | Attempted to free a null pointer |
| `ZeroSizeLayout` | Layout has zero size |

---

#### `struct ExplicitBuffer`

Fallback host-device copy when USM is unavailable.

```rust
pub fn new(size: usize) -> Self
pub fn size(&self) -> usize
pub fn write_from_host(&mut self, host_data: &[u8]) -> usize
pub fn read_to_host(&self, dest: &mut [u8]) -> usize
```

---

#### `enum DataPath`

Unified data path (USM zero-copy or explicit copy).

```rust
pub fn new(caps: SvmCapabilities) -> Self
pub fn is_zero_copy(&self) -> bool
pub fn upload(&self, host_data: &[u8]) -> Result<Vec<u8>, UsmError>
```

```rust
let path = DataPath::new(detect_usm_capabilities(0b0010));
assert!(path.is_zero_copy());
let gpu_data = path.upload(&[1, 2, 3, 4])?;
```

---

### Module `p2p`

GPU-to-GPU peer-to-peer memory transfer.

#### `struct DeviceId`

| Field | Type | Description |
|-------|------|-------------|
| `index` | `usize` | Platform-specific device index |
| `backend` | `BackendKind` | Backend type |

#### `enum BackendKind`

| Variant | Description |
|---------|-------------|
| `Cuda` | NVIDIA CUDA |
| `OpenCl` | OpenCL |

#### `struct P2PCapability`

| Field | Type | Description |
|-------|------|-------------|
| `src` | `DeviceId` | Source device |
| `dst` | `DeviceId` | Destination device |
| `direct_supported` | `bool` | Whether direct GPU-GPU is possible |
| `reason` | `Option<String>` | Reason if not supported |

#### `trait P2PProbe`

```rust
pub trait P2PProbe: Send + Sync {
    fn probe(&self, src: &DeviceId, dst: &DeviceId) -> P2PCapability;
}
```

#### `struct FallbackProbe`

Default probe — always reports P2P unavailable.

---

## bitnet-vulkan

Vulkan 1.x compute backend with buffer management, pipeline creation, and
command buffer recording.

### Quick Start

```rust
use bitnet_vulkan::{
    instance::{VulkanInstance, InstanceConfig},
    device::{select_physical_device, create_logical_device, DeviceSelector},
    pipeline::ComputePipelineBuilder,
    buffer::{allocate_buffer, BufferDescriptor, BufferUsage},
    command::{CommandPool, record_and_submit_compute},
};

let instance = VulkanInstance::new(&InstanceConfig {
    app_name: "bitnet".into(),
    app_version: 1,
    enable_validation: cfg!(debug_assertions),
})?;

let selected = select_physical_device(
    instance.raw(),
    &DeviceSelector { prefer_discrete: true, require_compute_queue: true },
)?;

let device = create_logical_device(instance.raw(), &selected)?;
```

---

### Module `instance`

#### `struct InstanceConfig`

| Field | Type | Description |
|-------|------|-------------|
| `app_name` | `String` | Application name for Vulkan driver |
| `app_version` | `u32` | Application version (Vulkan packed) |
| `enable_validation` | `bool` | Enable validation layers |

#### `struct VulkanInstance`

```rust
pub fn new(config: &InstanceConfig) -> Result<Self>
pub fn validation_enabled(&self) -> bool
pub fn raw(&self) -> &ash::Instance
pub fn entry(&self) -> &ash::Entry
```

---

### Module `device`

#### `struct DeviceSelector`

| Field | Type | Description |
|-------|------|-------------|
| `prefer_discrete` | `bool` | Prefer discrete over integrated |
| `require_compute_queue` | `bool` | Require dedicated compute queue family |

#### `struct SelectedDevice`

| Field | Type | Description |
|-------|------|-------------|
| `physical_device` | `vk::PhysicalDevice` | Vulkan physical device handle |
| `compute_queue_family` | `u32` | Compute queue family index |
| `device_name` | `String` | Device name from driver |
| `device_type` | `vk::PhysicalDeviceType` | Device type |

#### `select_physical_device`

```rust
pub fn select_physical_device(
    instance: &ash::Instance,
    selector: &DeviceSelector,
) -> Result<SelectedDevice>
```

Select the best physical device for compute workloads.

#### `create_logical_device`

```rust
pub fn create_logical_device(
    instance: &ash::Instance,
    selected: &SelectedDevice,
) -> Result<ash::Device>
```

Create a logical device with a single compute queue.

---

### Module `buffer`

#### `enum BufferUsage`

| Variant | Description |
|---------|-------------|
| `Staging` | Host-visible staging buffer |
| `DeviceLocal` | Device-local compute buffer |
| `Storage` | Device-local storage buffer (SSBO) |

#### `struct BufferDescriptor`

| Field | Type | Description |
|-------|------|-------------|
| `size` | `vk::DeviceSize` | Size in bytes |
| `usage` | `BufferUsage` | Usage pattern |
| `label` | `String` | Debug label |

#### `struct GpuBuffer`

| Field | Type | Description |
|-------|------|-------------|
| `buffer` | `vk::Buffer` | Raw Vulkan buffer handle |
| `memory` | `vk::DeviceMemory` | Backing device memory |
| `size` | `vk::DeviceSize` | Allocated size |
| `usage` | `BufferUsage` | Requested usage |

```rust
pub unsafe fn destroy(&self, device: &ash::Device)
```

#### `find_memory_type`

```rust
pub fn find_memory_type(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32>
```

#### `allocate_buffer`

```rust
pub fn allocate_buffer(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    descriptor: &BufferDescriptor,
) -> Result<GpuBuffer>
```

---

### Module `command`

#### `struct CommandPool`

```rust
pub fn new(device: &ash::Device, queue_family: u32) -> Result<Self>
pub fn allocate_command_buffer(&self, device: &ash::Device)
    -> Result<vk::CommandBuffer>
pub unsafe fn destroy(&self, device: &ash::Device)
```

| Field | Type | Description |
|-------|------|-------------|
| `pool` | `vk::CommandPool` | Raw command pool |
| `queue_family` | `u32` | Target queue family |

#### `record_and_submit_compute`

```rust
pub fn record_and_submit_compute(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    bind_group: vk::DescriptorSet,
    dispatch: (u32, u32, u32),
    queue: vk::Queue,
    fence: vk::Fence,
) -> Result<()>
```

Record begin, bind pipeline, dispatch, end — then submit with fence wait.

---

### Module `pipeline`

#### `struct ComputePipelineBuilder`

```rust
pub fn new(spirv: &[u32]) -> Self
pub fn entry_point(mut self, name: &str) -> Self
pub fn push_constant_size(mut self, size: u32) -> Self
pub fn descriptor_set_count(mut self, count: u32) -> Self
pub fn build(&self, device: &ash::Device) -> Result<ComputePipeline>
```

#### `struct ComputePipeline`

| Field | Type | Description |
|-------|------|-------------|
| `pipeline` | `vk::Pipeline` | Pipeline handle |
| `layout` | `vk::PipelineLayout` | Pipeline layout |
| `shader_module` | `vk::ShaderModule` | Shader module |

```rust
pub unsafe fn destroy(&self, device: &ash::Device)
```

---

### Module `kernels`

#### Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `MATMUL_GLSL` | `&str` | (embedded) | GLSL source for tiled matmul |
| `MATMUL_SPIRV` | `&[u8]` | (embedded) | Pre-compiled SPIR-V binary |
| `TILE_SIZE` | `u32` | `16` | Workgroup tile dimension |

---

### Module `error`

#### `enum VulkanError`

| Variant | Description |
|---------|-------------|
| `InstanceCreation` | Instance creation failed |
| `NoDevice` | No suitable physical device |
| `DeviceCreation` | Logical device creation failed |
| `PipelineCreation` | Compute pipeline creation failed |
| `ShaderCompilation` | Shader module loading failed |
| `BufferAllocation` | Buffer allocation failed |
| `MemoryTypeNotFound` | Required memory type unavailable |
| `CommandBufferError` | Command buffer recording/submission error |
| `QueueError` | Queue submission/sync error |
| `VkError(i32)` | Raw Vulkan error code |

```rust
pub type Result<T> = std::result::Result<T, VulkanError>;
```

---

## bitnet-rocm

AMD ROCm/HIP backend — dynamically loaded, compiles on any platform.

### Quick Start

```rust
use bitnet_rocm::{RocmBackend, enumerate_devices, DeviceBuffer, Stream, LaunchConfig};

let backend = RocmBackend::new(0)?;
if backend.is_available() {
    let devices = enumerate_devices()?;
    println!("Found {} AMD GPUs", devices.len());

    let stream = Stream::new()?;
    let buf = DeviceBuffer::alloc(4096)?;
    buf.copy_from_host(&data)?;
    stream.synchronize()?;
}
```

---

### Module root (`lib.rs`)

#### `struct RocmBackend`

```rust
pub fn new(device_index: usize) -> error::Result<Self>
pub fn name(&self) -> &'static str           // "rocm"
pub fn is_available(&self) -> bool
pub fn device_index(&self) -> usize
pub fn device_info(&self) -> Option<&RocmDeviceInfo>
```

---

### Module `device`

#### `struct RocmDeviceInfo`

| Field | Type | Description |
|-------|------|-------------|
| `index` | `usize` | Device ordinal |
| `name` | `String` | Device name |
| `total_memory_mib` | `usize` | VRAM in MiB |
| `compute_units` | `i32` | Compute unit count |
| `warp_size` | `i32` | Wavefront width |

#### `enumerate_devices`

```rust
pub fn enumerate_devices() -> Result<Vec<RocmDeviceInfo>>
```

Enumerate AMD GPUs via dynamic HIP loading. Returns empty vec if HIP absent.

#### `hip_runtime_available`

```rust
pub fn hip_runtime_available() -> bool
```

---

### Module `error`

#### `enum HipErrorCode`

| Variant | Raw | Description |
|---------|-----|-------------|
| `Success` | 0 | No error |
| `InvalidValue` | 1 | Invalid argument |
| `OutOfMemory` | 2 | Allocation failed |
| `NotInitialized` | 3 | HIP not initialized |
| `Deinitialized` | 4 | HIP deinitialized |
| `Unknown` | 9999 | Unrecognized code |

```rust
pub fn from_raw(code: u32) -> Self
```

#### `enum RocmError`

Backend-level error wrapping `HipErrorCode` with context strings.

#### `check_hip`

```rust
pub fn check_hip(status: u32, context: &str) -> Result<()>
```

Convert a HIP status code to `Result`, attaching `context` on failure.

---

### Module `kernel`

#### `struct LaunchConfig`

| Field | Type | Description |
|-------|------|-------------|
| `grid` | `(u32, u32, u32)` | Grid dimensions |
| `block` | `(u32, u32, u32)` | Block dimensions |
| `shared_mem_bytes` | `u32` | Dynamic shared memory |
| `stream` | `HipStream` | Target stream handle |

```rust
pub fn linear(n: u32, block_size: u32) -> Self
pub fn grid_2d(rows: u32, cols: u32, block_x: u32, block_y: u32) -> Self
```

```rust
let config = LaunchConfig::linear(1024, 256); // 4 blocks x 256 threads
let config = LaunchConfig::grid_2d(512, 512, 16, 16); // 32x32 blocks
```

#### `launch_kernel`

```rust
pub unsafe fn launch_kernel(
    function: HipFunction,
    config: &LaunchConfig,
    args: &[*mut c_void],
) -> Result<()>
```

**Safety:** Caller must ensure `function` and `args` are valid.

---

### Module `memory`

#### `struct DeviceBuffer`

```rust
pub fn alloc(size: usize) -> Result<Self>
pub fn copy_from_host(&self, src: &[u8]) -> Result<()>
pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()>
pub fn as_ptr(&self) -> HipDevicePtr
pub fn size(&self) -> usize
```

#### `memcpy_kind_for`

```rust
pub fn memcpy_kind_for(src_is_device: bool, dst_is_device: bool) -> HipMemcpyKind
```

---

### Module `stream`

#### `struct Stream`

```rust
pub fn new() -> Result<Self>
pub fn default_stream() -> Self
pub fn synchronize(&self) -> Result<()>
pub fn handle(&self) -> HipStream
```

---

### Module `ffi`

Low-level HIP FFI types (dynamically loaded — no build-time ROCm dependency).

| Type Alias | Maps To | Description |
|-----------|---------|-------------|
| `HipStream` | `*mut c_void` | Opaque stream handle |
| `HipModule` | `*mut c_void` | Opaque module handle |
| `HipFunction` | `*mut c_void` | Opaque kernel handle |
| `HipDevicePtr` | `*mut c_void` | Opaque device pointer |

| Constant | Description |
|----------|-------------|
| `HIP_STREAM_DEFAULT` | Null stream (synchronous) |

#### `struct HipDeviceProperties`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `[u8; 256]` | Device name bytes |
| `total_global_mem` | `usize` | Total VRAM |
| `shared_mem_per_block` | `usize` | Per-block shared memory |
| `warp_size` | `i32` | Wavefront width |
| `max_threads_per_block` | `i32` | Max threads per block |
| `multi_processor_count` | `i32` | CU count |
| `compute_units` | `i32` | Compute units |

```rust
pub fn device_name(&self) -> String
pub fn total_memory_mib(&self) -> usize
```

#### `enum HipMemcpyKind`

Host-to-host, host-to-device, device-to-host, device-to-device.

---

## bitnet-metal

Apple Metal compute backend for Apple Silicon GPUs.

### Quick Start

```rust
use bitnet_metal::capabilities::query_device;
use bitnet_metal::shader;

if let Some(info) = query_device() {
    println!("Metal device: {} (unified={})", info.name, info.has_unified_memory);
    let _kernel_src = shader::MATMUL_MSL;
}
```

---

### Module `capabilities`

#### `struct MetalDeviceInfo`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `String` | Device name |
| `registry_id` | `u64` | IORegistry ID |
| `max_threads_per_threadgroup` | `u64` | Max threadgroup size |
| `max_buffer_length` | `u64` | Max buffer allocation |
| `has_unified_memory` | `bool` | Unified memory architecture |
| `recommended_max_working_set_size` | `u64` | Recommended VRAM budget |

#### `query_device`

```rust
pub fn query_device() -> Option<MetalDeviceInfo>
```

Query Metal device capabilities. Returns `None` on non-macOS platforms.

---

### Module `shader`

Inline MSL kernel sources (embedded at compile time).

| Constant | Type | Description |
|----------|------|-------------|
| `MATMUL_MSL` | `&str` | Matrix multiplication MSL kernel |
| `SOFTMAX_MSL` | `&str` | Row-wise softmax MSL kernel |
| `RMSNORM_MSL` | `&str` | RMS normalization MSL kernel |

---

### Module `error`

#### `enum MetalError`

Metal backend error type.

```rust
pub type Result<T> = std::result::Result<T, MetalError>;
```

---

## bitnet-webgpu

WebGPU compute backend via wgpu — cross-platform (Vulkan, Metal, DX12, GL,
browser WebGPU).

### Quick Start

```rust
use bitnet_webgpu::{WebGpuBackend, MatmulParams, ElementwiseOp};

let backend = WebGpuBackend::new().await?;
println!("Using {} via {:?}", backend.gpu.adapter_name(), backend.gpu.backend());

let result = backend.matmul(
    &a, &b,
    MatmulParams { m: 64, n: 64, k: 64, _pad: 0 },
).await?;
```

---

### Module root (`lib.rs`)

#### `struct WebGpuBackend`

| Field | Type | Description |
|-------|------|-------------|
| `gpu` | `WebGpuDevice` | Device + queue handle |

```rust
pub async fn new() -> Result<Self>
pub async fn matmul(&self, a: &[f32], b: &[f32], params: MatmulParams)
    -> Result<Vec<f32>>
pub async fn softmax(&self, input: &[f32], n: u32)
    -> Result<Vec<f32>>
pub async fn elementwise(&self, a: &[f32], b: &[f32], op: ElementwiseOp)
    -> Result<Vec<f32>>
```

#### `struct MatmulParams`

`#[repr(C)]` uniform parameters — `Pod + Zeroable`.

| Field | Type | Description |
|-------|------|-------------|
| `m` | `u32` | Rows of A / output |
| `n` | `u32` | Columns of B / output |
| `k` | `u32` | Inner dimension |
| `_pad` | `u32` | Padding for alignment |

#### `struct SoftmaxParams`

| Field | Type | Description |
|-------|------|-------------|
| `n` | `u32` | Row length |
| `_pad` | `u32` | Padding |

#### `struct ElementwiseParams`

| Field | Type | Description |
|-------|------|-------------|
| `len` | `u32` | Number of elements |
| `op` | `u32` | Operation code |

#### `enum ElementwiseOp`

| Variant | Value | Description |
|---------|-------|-------------|
| `Add` | 0 | Element-wise addition |
| `Mul` | 1 | Element-wise multiplication |
| `Relu` | 2 | ReLU activation |
| `Silu` | 3 | SiLU activation |

---

### Module `device`

#### `struct WebGpuDevice`

| Field | Type | Description |
|-------|------|-------------|
| `instance` | `wgpu::Instance` | wgpu instance |
| `adapter` | `wgpu::Adapter` | Selected adapter |
| `device` | `wgpu::Device` | Logical device |
| `queue` | `wgpu::Queue` | Command queue |

```rust
pub async fn new() -> Result<Self>
pub fn adapter_name(&self) -> String
pub fn backend(&self) -> wgpu::Backend
pub fn max_buffer_size(&self) -> u64
```

---

### Module `buffer`

#### `struct GpuBuffer`

| Field | Type | Description |
|-------|------|-------------|
| `storage` | `wgpu::Buffer` | GPU storage buffer |
| `size` | `u64` | Buffer size in bytes |

```rust
pub fn from_slice<T: Pod>(device: &wgpu::Device, data: &[T], label: &str) -> Self
pub fn new_uninit<T: Pod>(device: &wgpu::Device, len: usize, label: &str) -> Self
pub async fn read_back<T: Pod>(
    &self, device: &wgpu::Device, queue: &wgpu::Queue,
) -> Result<Vec<T>>
pub fn new_uniform<T: Pod>(device: &wgpu::Device, value: &T, label: &str) -> Self
```

```rust
let buf = GpuBuffer::from_slice(&device, &weights, "layer0_weights");
let output = GpuBuffer::new_uninit::<f32>(&device, 1024, "output");
let result: Vec<f32> = output.read_back(&device, &queue).await?;
```

---

### Module `pipeline`

#### `struct ComputePipeline`

| Field | Type | Description |
|-------|------|-------------|
| `pipeline` | `wgpu::ComputePipeline` | Compiled pipeline |
| `bind_group_layout` | `wgpu::BindGroupLayout` | Layout for binding |

```rust
pub fn new(
    device: &wgpu::Device,
    wgsl_source: &str,
    label: &str,
    entry_point: &str,
) -> Result<Self>

pub fn bind_group(
    &self,
    device: &wgpu::Device,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup
```

---

### Module `shader`

| Constant | Type | Description |
|----------|------|-------------|
| `MATMUL_WGSL` | `&str` | Matrix multiply WGSL shader |
| `SOFTMAX_WGSL` | `&str` | Softmax WGSL shader |
| `ELEMENTWISE_WGSL` | `&str` | Element-wise ops WGSL shader |

---

### Module `error`

#### `enum WebGpuError`

WebGPU backend error type.

```rust
pub type Result<T> = std::result::Result<T, WebGpuError>;
```

---

## bitnet-level-zero

Intel Level-Zero backend — native API for Intel Arc and Data Center GPUs.
All symbols resolved at runtime via `libloading` (no build-time SDK dependency).

### Quick Start

```rust
use bitnet_level_zero::{
    driver::{enumerate_drivers, select_best_gpu},
    context::ContextBuilder,
    module::ModuleBuilder,
    kernel::{KernelBuilder, GroupSize, DispatchDimensions},
    memory::MemoryAllocBuilder,
};

// Enumerate
let drivers = enumerate_drivers()?;
let gpu = select_best_gpu()?;

// Create context + module
let ctx = ContextBuilder::new(gpu.driver_index).build()?;
let module = ModuleBuilder::from_spirv(&spirv_bytes).build(&ctx)?;

// Create kernel
let kernel = KernelBuilder::new("matmul")
    .group_size(GroupSize::new_1d(256))
    .build(&module)?;

// Allocate device memory
let buf = MemoryAllocBuilder::device(4096)
    .alignment(64)
    .allocate(&ctx)?;
```

---

### Module `driver`

#### `struct DriverInfo`

| Field | Type | Description |
|-------|------|-------------|
| `index` | `usize` | Driver index (0-based) |
| `device_count` | `usize` | Devices on this driver |
| `api_version` | `(u32, u32)` | API (major, minor) |

#### `enumerate_drivers`

```rust
pub fn enumerate_drivers() -> Result<Vec<DriverInfo>>
```

Returns empty vec if Level-Zero is not installed.

#### `is_runtime_available`

```rust
pub fn is_runtime_available() -> bool
```

#### `loader_library_name`

```rust
pub fn loader_library_name() -> &'static str
```

Returns `"ze_loader.dll"` (Windows), `"libze_loader.so"` (Linux), or
`"libze_loader.dylib"` (macOS).

#### `struct DeviceEntry`

| Field | Type | Description |
|-------|------|-------------|
| `driver_index` | `usize` | Parent driver index |
| `device_index` | `usize` | Device index within driver |
| `properties` | `ZeDeviceProperties` | Device properties |

#### `enumerate_gpu_devices`

```rust
pub fn enumerate_gpu_devices() -> Result<Vec<DeviceEntry>>
```

Enumerate all GPU devices across all drivers.

#### `select_best_gpu`

```rust
pub fn select_best_gpu() -> Result<DeviceEntry>
```

Select GPU with highest EU count.

---

### Module `context`

#### `struct ContextBuilder`

```rust
pub fn new(driver_index: usize) -> Self
pub fn flags(mut self, flags: u32) -> Self
pub fn build(self) -> Result<LevelZeroContext>
```

#### `struct LevelZeroContext`

```rust
pub fn driver_index(&self) -> usize
pub fn is_initialized(&self) -> bool
```

#### `struct ContextConfig`

| Field | Type | Description |
|-------|------|-------------|
| `flags` | `u32` | Reserved flags |

---

### Module `device`

#### `struct DeviceCapabilities`

Comprehensive device capability snapshot.

| Field | Type | Description |
|-------|------|-------------|
| `properties` | `ZeDeviceProperties` | Core properties |
| `compute` | `ZeComputeProperties` | Compute limits |
| `memory` | `Vec<ZeMemoryProperties>` | Memory domains |

```rust
pub fn total_eus(&self) -> u32
pub fn total_threads(&self) -> u32
pub fn is_gpu(&self) -> bool
pub fn name(&self) -> &str
```

#### `struct DeviceQuery`

Builder for filtering devices by capabilities.

```rust
pub fn new() -> Self
pub fn device_type(mut self, dt: ZeDeviceType) -> Self
pub fn min_memory(mut self, bytes: u64) -> Self
pub fn min_eus(mut self, count: u32) -> Self
pub fn matches(&self, caps: &DeviceCapabilities) -> bool
```

```rust
let query = DeviceQuery::new()
    .device_type(ZeDeviceType::Gpu)
    .min_eus(128)
    .min_memory(4 * 1024 * 1024 * 1024);
```

---

### Module `module`

#### `struct ModuleBuilder`

```rust
pub fn from_spirv(spirv: &[u8]) -> Self
pub fn format(mut self, fmt: ZeModuleFormat) -> Self
pub fn build_flags(mut self, flags: impl Into<String>) -> Self
pub fn spirv_size(&self) -> usize
pub fn build(self, ctx: &LevelZeroContext) -> Result<LevelZeroModule>
```

#### `struct LevelZeroModule`

```rust
pub fn spirv_size(&self) -> usize
pub fn is_initialized(&self) -> bool
pub fn build_flags(&self) -> Option<&str>
```

#### `struct ModuleConfig`

| Field | Type | Description |
|-------|------|-------------|
| `format` | `ZeModuleFormat` | SPIR-V or native |
| `build_flags` | `Option<String>` | Compiler flags |

---

### Module `kernel`

#### `struct KernelBuilder`

```rust
pub fn new(name: impl Into<String>) -> Self
pub fn group_size(mut self, gs: GroupSize) -> Self
pub fn name(&self) -> &str
pub fn build(self, module: &LevelZeroModule) -> Result<LevelZeroKernel>
```

#### `struct LevelZeroKernel`

```rust
pub fn name(&self) -> &str
pub fn group_size(&self) -> &GroupSize
pub fn is_initialized(&self) -> bool
```

#### `struct DispatchDimensions`

| Field | Type | Description |
|-------|------|-------------|
| `group_count_x` | `u32` | Groups in X |
| `group_count_y` | `u32` | Groups in Y |
| `group_count_z` | `u32` | Groups in Z |

```rust
pub fn new_1d(groups: u32) -> Self
pub fn new_2d(x: u32, y: u32) -> Self
pub fn new_3d(x: u32, y: u32, z: u32) -> Self
pub fn total_groups(&self) -> u64
```

#### `struct GroupSize`

| Field | Type | Description |
|-------|------|-------------|
| `x` | `u32` | Threads in X |
| `y` | `u32` | Threads in Y |
| `z` | `u32` | Threads in Z |

```rust
pub fn new_1d(size: u32) -> Self
pub fn total_threads(&self) -> u32
```

---

### Module `memory`

#### `struct MemoryAllocBuilder`

```rust
pub fn device(size: usize) -> Self
pub fn host(size: usize) -> Self
pub fn shared(size: usize) -> Self
pub fn alignment(mut self, align: usize) -> Self
pub fn size(&self) -> usize
pub fn memory_type(&self) -> ZeMemoryType
pub fn allocate(self, ctx: &LevelZeroContext) -> Result<DeviceBuffer>
```

```rust
let weights = MemoryAllocBuilder::device(model_size)
    .alignment(64)
    .allocate(&ctx)?;

let staging = MemoryAllocBuilder::host(batch_size)
    .allocate(&ctx)?;

let shared = MemoryAllocBuilder::shared(small_tensor_size)
    .allocate(&ctx)?;
```

#### `struct DeviceBuffer`

```rust
pub fn size(&self) -> usize
pub fn memory_type(&self) -> ZeMemoryType
pub fn is_allocated(&self) -> bool
```

#### `estimate_total_memory`

```rust
pub fn estimate_total_memory(tensor_sizes: &[usize]) -> usize
```

#### `estimate_aligned_memory`

```rust
pub fn estimate_aligned_memory(tensor_sizes: &[usize], alignment: usize) -> usize
```

---

### Module `error`

#### `enum LevelZeroError`

| Variant | Description |
|---------|-------------|
| `RuntimeNotFound` | Level-Zero loader library not found |
| `ApiError` | L0 API call returned error |
| `NoDevice` | No compatible GPU found |
| `ModuleCompilationFailed` | SPIR-V compilation failed |
| `KernelNotFound` | Kernel name not in module |
| `AllocationFailed` | Memory allocation failed |
| `InvalidArgument` | Invalid builder argument |
| `UnsupportedVersion` | Driver version unsupported |

```rust
pub type Result<T> = std::result::Result<T, LevelZeroError>;
```

#### `check`

```rust
pub fn check(result: ZeResult) -> Result<()>
```

---

### Module `ffi`

Low-level Level-Zero FFI types. All symbols resolved at runtime via
`libloading`.

#### `enum ZeResult`

Level-Zero result codes.

```rust
pub fn from_raw(val: u32) -> Self
pub fn is_success(self) -> bool
pub fn as_raw(self) -> u32
```

#### Opaque Handle Types

| Type | Wraps |
|------|-------|
| `ZeDriverHandle` | `ze_driver_handle_t` |
| `ZeDeviceHandle` | `ze_device_handle_t` |
| `ZeContextHandle` | `ze_context_handle_t` |
| `ZeModuleHandle` | `ze_module_handle_t` |
| `ZeKernelHandle` | `ze_kernel_handle_t` |
| `ZeCommandQueueHandle` | `ze_command_queue_handle_t` |
| `ZeCommandListHandle` | `ze_command_list_handle_t` |
| `ZeEventPoolHandle` | `ze_event_pool_handle_t` |
| `ZeEventHandle` | `ze_event_handle_t` |

#### `enum ZeDeviceType`

| Variant | Description |
|---------|-------------|
| `Gpu` | GPU device |
| `Cpu` | CPU device |
| `Fpga` | FPGA device |
| `Mca` | Memory Copy Accelerator |

#### `enum ZeMemoryType`

| Variant | Description |
|---------|-------------|
| `Host` | Host-visible pinned memory |
| `Device` | Device-local memory |
| `Shared` | Unified shared memory |

#### `enum ZeModuleFormat`

| Variant | Description |
|---------|-------------|
| `IlSpirv` | SPIR-V IL format |
| `Native` | Native binary format |

#### `struct ZeDeviceProperties`

| Field | Type | Description |
|-------|------|-------------|
| `device_type` | `ZeDeviceType` | GPU / CPU / FPGA / MCA |
| `vendor_id` | `u32` | PCI vendor ID |
| `device_id` | `u32` | PCI device ID |
| `core_clock_rate` | `u32` | Max clock (MHz) |
| `max_mem_alloc_size` | `u64` | Max single allocation |
| `max_hardware_contexts` | `u32` | Hardware contexts |
| `num_threads_per_eu` | `u32` | Threads per EU |
| `num_eu_per_subslice` | `u32` | EUs per subslice |
| `num_subslices_per_slice` | `u32` | Subslices per slice |
| `num_slices` | `u32` | Slices |
| `name` | `String` | Device name |

#### `struct ZeComputeProperties`

| Field | Type | Description |
|-------|------|-------------|
| `max_total_group_size` | `u32` | Max work-group size |
| `max_group_size_x/y/z` | `u32` | Per-dimension limits |
| `max_group_count_x/y/z` | `u32` | Max dispatch dimensions |
| `max_shared_local_memory` | `u32` | Shared local memory (bytes) |
| `sub_group_sizes` | `Vec<u32>` | Supported subgroup sizes |

#### `struct ZeMemoryProperties`

| Field | Type | Description |
|-------|------|-------------|
| `total_size` | `u64` | Total memory (bytes) |
| `max_clock_rate` | `u32` | Memory clock (MHz) |
| `max_bus_width` | `u32` | Bus width (bits) |
