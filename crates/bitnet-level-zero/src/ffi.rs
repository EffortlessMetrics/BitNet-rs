//! Basic FFI type definitions for Intel Level-Zero.
//!
//! These are Rust-side definitions of Level-Zero C types. We never link
//! statically -- all symbols are resolved at runtime via `libloading`.

use std::fmt;

/// Level-Zero result codes (`ze_result_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ZeResult {
    Success = 0x0,
    NotReady = 0x1,
    ErrorDeviceLost = 0x70000001,
    ErrorOutOfHostMemory = 0x70000002,
    ErrorOutOfDeviceMemory = 0x70000003,
    ErrorModuleBuildFailure = 0x70000004,
    ErrorModuleLinkFailure = 0x70000005,
    ErrorInsufficientPermissions = 0x70010000,
    ErrorNotAvailable = 0x70010001,
    ErrorUninitialized = 0x78000001,
    ErrorUnsupportedVersion = 0x78000002,
    ErrorUnsupportedFeature = 0x78000003,
    ErrorInvalidArgument = 0x78000004,
    ErrorInvalidNullHandle = 0x78000005,
    ErrorHandleObjectInUse = 0x78000006,
    ErrorInvalidNullPointer = 0x78000007,
    ErrorInvalidSize = 0x78000008,
    ErrorUnsupportedSize = 0x78000009,
    ErrorUnsupportedAlignment = 0x7800000a,
    ErrorInvalidEnumeration = 0x7800000d,
    ErrorInvalidNativeKernel = 0x7800000e,
    ErrorInvalidGlobalName = 0x7800000f,
    ErrorInvalidKernelName = 0x78000010,
    ErrorInvalidFunctionName = 0x78000011,
    ErrorInvalidGroupSizeDimension = 0x78000012,
    ErrorInvalidGlobalWidthDimension = 0x78000013,
    ErrorUnknown = 0x7ffffffe,
}

impl ZeResult {
    /// Convert from a raw u32 value, returning `ErrorUnknown` for unrecognized codes.
    pub fn from_raw(val: u32) -> Self {
        match val {
            0x0 => Self::Success,
            0x1 => Self::NotReady,
            0x70000001 => Self::ErrorDeviceLost,
            0x70000002 => Self::ErrorOutOfHostMemory,
            0x70000003 => Self::ErrorOutOfDeviceMemory,
            0x70000004 => Self::ErrorModuleBuildFailure,
            0x70000005 => Self::ErrorModuleLinkFailure,
            0x70010000 => Self::ErrorInsufficientPermissions,
            0x70010001 => Self::ErrorNotAvailable,
            0x78000001 => Self::ErrorUninitialized,
            0x78000002 => Self::ErrorUnsupportedVersion,
            0x78000003 => Self::ErrorUnsupportedFeature,
            0x78000004 => Self::ErrorInvalidArgument,
            0x78000005 => Self::ErrorInvalidNullHandle,
            0x78000006 => Self::ErrorHandleObjectInUse,
            0x78000007 => Self::ErrorInvalidNullPointer,
            0x78000008 => Self::ErrorInvalidSize,
            0x78000009 => Self::ErrorUnsupportedSize,
            0x7800000a => Self::ErrorUnsupportedAlignment,
            0x7800000d => Self::ErrorInvalidEnumeration,
            0x7800000e => Self::ErrorInvalidNativeKernel,
            0x7800000f => Self::ErrorInvalidGlobalName,
            0x78000010 => Self::ErrorInvalidKernelName,
            0x78000011 => Self::ErrorInvalidFunctionName,
            0x78000012 => Self::ErrorInvalidGroupSizeDimension,
            0x78000013 => Self::ErrorInvalidGlobalWidthDimension,
            _ => Self::ErrorUnknown,
        }
    }

    /// Whether this result indicates success.
    pub fn is_success(self) -> bool {
        self == Self::Success
    }

    /// Return the raw u32 value.
    pub fn as_raw(self) -> u32 {
        self as u32
    }
}

impl fmt::Display for ZeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?} (0x{:08x})", *self as u32)
    }
}

// --- Opaque handle types ---

/// Opaque driver handle (`ze_driver_handle_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ZeDriverHandle(pub *mut std::ffi::c_void);

/// Opaque device handle (`ze_device_handle_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ZeDeviceHandle(pub *mut std::ffi::c_void);

/// Opaque context handle (`ze_context_handle_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ZeContextHandle(pub *mut std::ffi::c_void);

/// Opaque module handle (`ze_module_handle_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ZeModuleHandle(pub *mut std::ffi::c_void);

/// Opaque kernel handle (`ze_kernel_handle_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ZeKernelHandle(pub *mut std::ffi::c_void);

/// Opaque command queue handle (`ze_command_queue_handle_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ZeCommandQueueHandle(pub *mut std::ffi::c_void);

/// Opaque command list handle (`ze_command_list_handle_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ZeCommandListHandle(pub *mut std::ffi::c_void);

/// Opaque event pool handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ZeEventPoolHandle(pub *mut std::ffi::c_void);

/// Opaque event handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ZeEventHandle(pub *mut std::ffi::c_void);

// Safety: L0 handles are thread-safe per specification.
unsafe impl Send for ZeDriverHandle {}
unsafe impl Sync for ZeDriverHandle {}
unsafe impl Send for ZeDeviceHandle {}
unsafe impl Sync for ZeDeviceHandle {}
unsafe impl Send for ZeContextHandle {}
unsafe impl Sync for ZeContextHandle {}
unsafe impl Send for ZeModuleHandle {}
unsafe impl Sync for ZeModuleHandle {}
unsafe impl Send for ZeKernelHandle {}
unsafe impl Sync for ZeKernelHandle {}
unsafe impl Send for ZeCommandQueueHandle {}
unsafe impl Sync for ZeCommandQueueHandle {}
unsafe impl Send for ZeCommandListHandle {}
unsafe impl Sync for ZeCommandListHandle {}
unsafe impl Send for ZeEventPoolHandle {}
unsafe impl Sync for ZeEventPoolHandle {}
unsafe impl Send for ZeEventHandle {}
unsafe impl Sync for ZeEventHandle {}

// --- Descriptor / property structs ---

/// Device type enumeration (`ze_device_type_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ZeDeviceType {
    Gpu = 1,
    Cpu = 2,
    Fpga = 3,
    Mca = 4,
    Vpu = 5,
}

impl ZeDeviceType {
    pub fn from_raw(val: u32) -> Option<Self> {
        match val {
            1 => Some(Self::Gpu),
            2 => Some(Self::Cpu),
            3 => Some(Self::Fpga),
            4 => Some(Self::Mca),
            5 => Some(Self::Vpu),
            _ => None,
        }
    }
}

/// Memory allocation type (`ze_memory_type_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ZeMemoryType {
    Host = 0x1,
    Device = 0x2,
    Shared = 0x3,
}

/// Module format (`ze_module_format_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ZeModuleFormat {
    IlSpirv = 0x1,
    NativeCode = 0x2,
}

/// Device properties (simplified view of `ze_device_properties_t`).
#[derive(Debug, Clone)]
pub struct ZeDeviceProperties {
    pub device_type: ZeDeviceType,
    pub vendor_id: u32,
    pub device_id: u32,
    pub core_clock_rate: u32,
    pub max_mem_alloc_size: u64,
    pub max_hardware_contexts: u32,
    pub max_command_queue_priority: u32,
    pub num_threads_per_eu: u32,
    pub num_eu_per_subslice: u32,
    pub num_subslices_per_slice: u32,
    pub num_slices: u32,
    pub name: String,
}

impl Default for ZeDeviceProperties {
    fn default() -> Self {
        Self {
            device_type: ZeDeviceType::Gpu,
            vendor_id: 0,
            device_id: 0,
            core_clock_rate: 0,
            max_mem_alloc_size: 0,
            max_hardware_contexts: 0,
            max_command_queue_priority: 0,
            num_threads_per_eu: 0,
            num_eu_per_subslice: 0,
            num_subslices_per_slice: 0,
            num_slices: 0,
            name: String::new(),
        }
    }
}

/// Compute properties (simplified view of `ze_device_compute_properties_t`).
#[derive(Debug, Clone)]
pub struct ZeComputeProperties {
    pub max_total_group_size: u32,
    pub max_group_size_x: u32,
    pub max_group_size_y: u32,
    pub max_group_size_z: u32,
    pub max_group_count_x: u32,
    pub max_group_count_y: u32,
    pub max_group_count_z: u32,
    pub max_shared_local_memory: u32,
    pub sub_group_sizes: Vec<u32>,
}

impl Default for ZeComputeProperties {
    fn default() -> Self {
        Self {
            max_total_group_size: 256,
            max_group_size_x: 256,
            max_group_size_y: 256,
            max_group_size_z: 256,
            max_group_count_x: u32::MAX,
            max_group_count_y: u32::MAX,
            max_group_count_z: u32::MAX,
            max_shared_local_memory: 65536,
            sub_group_sizes: vec![8, 16, 32],
        }
    }
}

/// Memory properties (simplified view).
#[derive(Debug, Clone)]
pub struct ZeMemoryProperties {
    pub total_size: u64,
    pub max_clock_rate: u32,
    pub max_bus_width: u32,
}
