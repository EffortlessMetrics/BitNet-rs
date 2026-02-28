//! Intel Level-Zero backend for BitNet GPU inference.
//!
//! This crate provides Rust wrappers around the Intel Level-Zero API for
//! running BitNet inference on Intel GPUs (Arc, Data Center Max, etc.).
//!
//! **Dynamic loading**: The crate never links against Level-Zero at compile
//! time. FFI types are defined locally and the runtime library (`ze_loader`)
//! is loaded via `libloading` at runtime. This means the crate compiles
//! everywhere but only functions where Level-Zero is installed.

pub mod context;
pub mod device;
pub mod driver;
pub mod error;
pub mod ffi;
pub mod kernel;
pub mod memory;
pub mod module;

pub use context::{ContextBuilder, LevelZeroContext};
pub use device::{DeviceCapabilities, DeviceQuery};
pub use driver::{enumerate_drivers, enumerate_gpu_devices, is_runtime_available, select_best_gpu};
pub use error::{LevelZeroError, Result};
pub use kernel::{DispatchDimensions, GroupSize, KernelBuilder, LevelZeroKernel};
pub use memory::{DeviceBuffer, MemoryAllocBuilder};
pub use module::{LevelZeroModule, ModuleBuilder};

/// Top-level backend handle that ties together driver, context, and device.
#[derive(Debug)]
pub struct LevelZeroBackend {
    context: LevelZeroContext,
    driver_index: usize,
    device_index: usize,
}

impl LevelZeroBackend {
    /// Create a new backend using the first available GPU.
    ///
    /// Placeholder: real implementation enumerates drivers, selects a device,
    /// and creates a context.
    pub fn new() -> Result<Self> {
        if !is_runtime_available() {
            return Err(LevelZeroError::RuntimeNotFound(
                "Level-Zero loader not found on this system".into(),
            ));
        }

        let ctx = ContextBuilder::new(0).build()?;
        Ok(Self { context: ctx, driver_index: 0, device_index: 0 })
    }

    /// Create a backend targeting a specific driver and device.
    pub fn with_device(driver_index: usize, device_index: usize) -> Result<Self> {
        let ctx = ContextBuilder::new(driver_index).build()?;
        Ok(Self { context: ctx, driver_index, device_index })
    }

    /// The driver index.
    pub fn driver_index(&self) -> usize {
        self.driver_index
    }

    /// The device index within the driver.
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Reference to the underlying context.
    pub fn context(&self) -> &LevelZeroContext {
        &self.context
    }

    /// Load a SPIR-V module into this backend's context.
    pub fn load_module(&self, spirv: &[u8]) -> Result<LevelZeroModule> {
        ModuleBuilder::from_spirv(spirv).build(&self.context)
    }

    /// Allocate device memory.
    pub fn alloc_device(&self, size: usize) -> Result<DeviceBuffer> {
        MemoryAllocBuilder::device(size).allocate(&self.context)
    }

    /// Allocate shared (host+device) memory.
    pub fn alloc_shared(&self, size: usize) -> Result<DeviceBuffer> {
        MemoryAllocBuilder::shared(size).allocate(&self.context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::*;

    #[test]
    fn ze_result_success_is_zero() {
        assert_eq!(ZeResult::Success.as_raw(), 0);
    }

    #[test]
    fn ze_result_roundtrip() {
        let codes = [
            ZeResult::Success,
            ZeResult::NotReady,
            ZeResult::ErrorDeviceLost,
            ZeResult::ErrorOutOfHostMemory,
            ZeResult::ErrorOutOfDeviceMemory,
            ZeResult::ErrorModuleBuildFailure,
            ZeResult::ErrorUninitialized,
            ZeResult::ErrorInvalidArgument,
        ];
        for code in &codes {
            assert_eq!(ZeResult::from_raw(code.as_raw()), *code);
        }
    }

    #[test]
    fn ze_result_unknown_code_maps_to_error_unknown() {
        assert_eq!(ZeResult::from_raw(0xDEADBEEF), ZeResult::ErrorUnknown);
    }

    #[test]
    fn ze_result_display_includes_hex() {
        let s = format!("{}", ZeResult::ErrorDeviceLost);
        assert!(s.contains("0x70000001"), "display should include hex: {s}");
    }

    #[test]
    fn ze_device_type_from_raw_roundtrip() {
        assert_eq!(ZeDeviceType::from_raw(1), Some(ZeDeviceType::Gpu));
        assert_eq!(ZeDeviceType::from_raw(2), Some(ZeDeviceType::Cpu));
        assert_eq!(ZeDeviceType::from_raw(99), None);
    }

    #[test]
    fn handle_types_are_pointer_sized() {
        assert_eq!(std::mem::size_of::<ZeDriverHandle>(), std::mem::size_of::<*mut ()>());
        assert_eq!(std::mem::size_of::<ZeDeviceHandle>(), std::mem::size_of::<*mut ()>());
        assert_eq!(std::mem::size_of::<ZeContextHandle>(), std::mem::size_of::<*mut ()>());
        assert_eq!(std::mem::size_of::<ZeModuleHandle>(), std::mem::size_of::<*mut ()>());
        assert_eq!(std::mem::size_of::<ZeKernelHandle>(), std::mem::size_of::<*mut ()>());
    }

    #[test]
    fn error_from_ze_result() {
        let err: LevelZeroError = ZeResult::ErrorDeviceLost.into();
        match err {
            LevelZeroError::ApiError { result } => {
                assert_eq!(result, ZeResult::ErrorDeviceLost);
            }
            _ => panic!("expected ApiError"),
        }
    }

    #[test]
    fn error_check_success() {
        assert!(error::check(ZeResult::Success).is_ok());
    }

    #[test]
    fn error_check_failure() {
        assert!(error::check(ZeResult::ErrorOutOfDeviceMemory).is_err());
    }

    #[test]
    fn context_builder_creates_placeholder() {
        let ctx = ContextBuilder::new(0).flags(0).build().unwrap();
        assert_eq!(ctx.driver_index(), 0);
        assert!(!ctx.is_initialized());
    }

    #[test]
    fn module_builder_rejects_empty_spirv() {
        let ctx = ContextBuilder::new(0).build().unwrap();
        let result = ModuleBuilder::from_spirv(&[]).build(&ctx);
        assert!(result.is_err());
    }

    #[test]
    fn module_builder_accepts_spirv() {
        let ctx = ContextBuilder::new(0).build().unwrap();
        let spirv = vec![0x03, 0x02, 0x23, 0x07]; // SPIR-V magic
        let module = ModuleBuilder::from_spirv(&spirv).build(&ctx).unwrap();
        assert_eq!(module.spirv_size(), 4);
    }

    #[test]
    fn kernel_builder_rejects_empty_name() {
        let ctx = ContextBuilder::new(0).build().unwrap();
        let spirv = vec![0x03, 0x02, 0x23, 0x07];
        let module = ModuleBuilder::from_spirv(&spirv).build(&ctx).unwrap();
        let result = KernelBuilder::new("").build(&module);
        assert!(result.is_err());
    }

    #[test]
    fn kernel_builder_creates_kernel() {
        let ctx = ContextBuilder::new(0).build().unwrap();
        let spirv = vec![0x03, 0x02, 0x23, 0x07];
        let module = ModuleBuilder::from_spirv(&spirv).build(&ctx).unwrap();
        let kernel = KernelBuilder::new("matmul_i2s")
            .group_size(GroupSize::new_1d(128))
            .build(&module)
            .unwrap();
        assert_eq!(kernel.name(), "matmul_i2s");
        assert_eq!(kernel.group_size().total_threads(), 128);
    }

    #[test]
    fn memory_alloc_rejects_zero_size() {
        let ctx = ContextBuilder::new(0).build().unwrap();
        let result = MemoryAllocBuilder::device(0).allocate(&ctx);
        assert!(result.is_err());
    }

    #[test]
    fn memory_alloc_creates_buffer() {
        let ctx = ContextBuilder::new(0).build().unwrap();
        let buf = MemoryAllocBuilder::device(4096).alignment(256).allocate(&ctx).unwrap();
        assert_eq!(buf.size(), 4096);
        assert_eq!(buf.memory_type(), ZeMemoryType::Device);
    }

    #[test]
    fn memory_estimation() {
        let sizes = vec![1024, 2048, 4096];
        assert_eq!(memory::estimate_total_memory(&sizes), 7168);
        assert_eq!(memory::estimate_aligned_memory(&sizes, 256), 7168);
        assert_eq!(memory::estimate_aligned_memory(&[100, 200], 256), 512);
    }

    #[test]
    fn device_query_matches_gpu() {
        let caps = DeviceCapabilities {
            properties: ZeDeviceProperties {
                device_type: ZeDeviceType::Gpu,
                num_slices: 2,
                num_subslices_per_slice: 8,
                num_eu_per_subslice: 16,
                num_threads_per_eu: 7,
                ..Default::default()
            },
            compute: ZeComputeProperties::default(),
            memory: vec![ZeMemoryProperties {
                total_size: 8 * 1024 * 1024 * 1024,
                max_clock_rate: 2100,
                max_bus_width: 256,
            }],
        };

        let query = DeviceQuery::new().device_type(ZeDeviceType::Gpu).min_eus(100);
        assert!(query.matches(&caps));

        let query_cpu = DeviceQuery::new().device_type(ZeDeviceType::Cpu);
        assert!(!query_cpu.matches(&caps));
    }

    #[test]
    fn dispatch_dimensions_total() {
        let d = DispatchDimensions::new_2d(4, 8);
        assert_eq!(d.total_groups(), 32);
    }

    #[test]
    fn driver_enumerate_returns_empty_without_runtime() {
        let drivers = enumerate_drivers().unwrap();
        let _ = drivers;
    }

    #[test]
    fn backend_with_device_creates_placeholder() {
        let backend = LevelZeroBackend::with_device(0, 0).unwrap();
        assert_eq!(backend.driver_index(), 0);
        assert_eq!(backend.device_index(), 0);
    }
}
