# ADR-003: GPU Hardware Abstraction Layer (HAL) traits

- **Status:** Accepted
- **Date:** 2025-07-14
- **Context:** The existing `KernelProvider` trait in `bitnet-kernels` provides a
  per-operation abstraction (matmul, quantize) with automatic best-provider selection
  via `KernelManager`.  As we add non-CUDA GPU backends
  ([ADR-001](./ADR-001-opencl-initial-backend.md),
  [ADR-002](./ADR-002-microcrate-per-backend.md)), we need a lower-level hardware
  abstraction that models GPU resources (devices, buffers, kernels, command queues)
  uniformly across OpenCL, Vulkan, Level Zero, Metal, and WebGPU.

  Without a shared abstraction, inference code would branch on every backend, and
  adding a new backend would require changes throughout the stack.  The existing
  `Device` enum (`Cpu`, `Cuda(usize)`, `Hip(usize)`, `Npu`, `Metal`) and
  `DeviceAwareQuantizer` already demonstrate the pattern of device-polymorphic
  dispatch.

- **Decision:** Introduce a `bitnet-gpu-hal` crate defining three core traits:

  ```rust
  /// A discovered GPU device.
  pub trait GpuDevice: Send + Sync {
      fn name(&self) -> &str;
      fn vendor(&self) -> GpuVendor;
      fn memory_bytes(&self) -> u64;
      fn create_buffer(&self, size: usize, usage: BufferUsage) -> Result<Box<dyn GpuBuffer>>;
      fn create_kernel(&self, source: &KernelSource) -> Result<Box<dyn GpuKernel>>;
      fn create_queue(&self) -> Result<Box<dyn GpuQueue>>;
  }

  /// A device-resident memory allocation.
  pub trait GpuBuffer: Send + Sync {
      fn size(&self) -> usize;
      fn write(&self, queue: &dyn GpuQueue, data: &[u8]) -> Result<()>;
      fn read(&self, queue: &dyn GpuQueue, data: &mut [u8]) -> Result<()>;
  }

  /// A compiled compute kernel ready for dispatch.
  pub trait GpuKernel: Send + Sync {
      fn dispatch(
          &self,
          queue: &dyn GpuQueue,
          global_work_size: [usize; 3],
          local_work_size: Option<[usize; 3]>,
          args: &[KernelArg],
      ) -> Result<()>;
  }
  ```

  Backend crates (`bitnet-opencl`, `bitnet-vulkan`, etc.) implement these traits.
  Higher-level code in `bitnet-kernels` and `bitnet-inference` programs against the
  trait objects, never against backend-specific types.

  A `MockGpuDevice` implementation is provided in `bitnet-gpu-hal` (behind a `mock`
  feature) for deterministic unit testing without real hardware.

- **Consequences:**
  - *Positive:* Inference code is backend-agnostic — same `matmul_i2s` kernel launch
    works on CUDA, OpenCL, and Vulkan through the HAL.
  - *Positive:* `MockGpuDevice` enables full kernel-dispatch test coverage in CI
    without GPU runners.
  - *Positive:* New backends only need to implement the three traits; no changes to
    inference, quantization, or model loading.
  - *Positive:* Extends naturally from the existing `KernelProvider` / `KernelManager`
    pattern — the HAL sits below `KernelProvider` as its implementation substrate.
  - *Negative:* Trait-object dispatch adds a virtual call per kernel launch (~5 ns);
    negligible compared to kernel execution time (microseconds to milliseconds).
  - *Negative:* Lowest-common-denominator API may not expose vendor-specific
    optimizations (e.g., CUDA cooperative groups, Metal argument buffers); backends
    can provide escape hatches via `as_any()` downcasting when needed.

## Alternatives considered

- **Enum dispatch (no trait objects):** Avoids vtable overhead but requires a central
  enum listing all backends; adding a backend changes the enum, violating
  open/closed principle and complicating feature gating.
- **Use `wgpu` / `wgpu-hal` directly:** Mature Rust GPU abstraction but tailored to
  graphics pipelines; compute-only workloads require workarounds; transitive
  dependency weight is significant (~50 crates).
- **Thin per-backend wrappers only (no shared traits):** Simpler initially but forces
  `#[cfg]` branches in every call site; tested by experience in the CUDA-only era.

## How to revert

Remove `bitnet-gpu-hal` and wire backend crates directly into `KernelManager` via
per-backend `KernelProvider` implementations (the current CUDA pattern).  This is the
pre-HAL status quo and remains a valid architecture for a small number of backends.
