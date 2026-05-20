//! OpenCL device detection via dynamic library loading.
//!
//! Loads `OpenCL.dll` (Windows) or `libOpenCL.so` (Linux) at runtime so the
//! code compiles and runs even when no OpenCL SDK is installed.  When the
//! library is absent every query returns an empty result instead of panicking.
#![allow(
    clippy::borrow_as_ptr,
    clippy::doc_markdown,
    clippy::if_not_else,
    clippy::missing_const_for_fn,
    clippy::redundant_closure_for_method_calls,
    clippy::ref_as_ptr,
    clippy::too_many_lines
)]

use std::{ffi::CString, fmt, os::raw::c_void, ptr, time::Instant};

use bitnet_intel_gpu_id::{
    is_intel_arc_device as shared_is_intel_arc_device, is_intel_vendor as shared_is_intel_vendor,
};

// ── OpenCL C constants ──────────────────────────────────────────────────────

const CL_SUCCESS: i32 = 0;
const CL_TRUE: u32 = 1;

// Platform info keys
const CL_PLATFORM_PROFILE: u32 = 0x0900;
const CL_PLATFORM_VERSION: u32 = 0x0901;
const CL_PLATFORM_NAME: u32 = 0x0902;
const CL_PLATFORM_VENDOR: u32 = 0x0903;
const CL_PLATFORM_EXTENSIONS: u32 = 0x0904;

// Device info keys
const CL_DEVICE_TYPE: u32 = 0x1000;
const CL_DEVICE_NAME: u32 = 0x102B;
const CL_DEVICE_VENDOR: u32 = 0x102C;
const CL_DEVICE_VERSION: u32 = 0x102F;
const CL_DEVICE_MAX_COMPUTE_UNITS: u32 = 0x1002;
const CL_DEVICE_GLOBAL_MEM_SIZE: u32 = 0x101F;
const CL_DEVICE_MAX_WORK_GROUP_SIZE: u32 = 0x1004;
const CL_DEVICE_MAX_CLOCK_FREQUENCY: u32 = 0x100C;
const CL_DEVICE_EXTENSIONS: u32 = 0x1030;
const CL_DRIVER_VERSION: u32 = 0x102D;

// Device type bitmask values
const CL_DEVICE_TYPE_ALL: u64 = 0xFFFF_FFFF;
const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_DEVICE_TYPE_CPU: u64 = 1 << 1;
const CL_DEVICE_TYPE_ACCELERATOR: u64 = 1 << 3;

// Memory flags
const CL_MEM_READ_ONLY: u64 = 1 << 2;
const CL_MEM_WRITE_ONLY: u64 = 1 << 1;

// Program build info keys
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;

// ── Public types ────────────────────────────────────────────────────────────

/// The type of an OpenCL device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenClDeviceType {
    Cpu,
    Gpu,
    Accelerator,
    Other(u64),
}

impl fmt::Display for OpenClDeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu => write!(f, "GPU"),
            Self::Accelerator => write!(f, "Accelerator"),
            Self::Other(v) => write!(f, "Other({v:#x})"),
        }
    }
}

impl OpenClDeviceType {
    fn from_bits(bits: u64) -> Self {
        if bits & CL_DEVICE_TYPE_GPU != 0 {
            Self::Gpu
        } else if bits & CL_DEVICE_TYPE_CPU != 0 {
            Self::Cpu
        } else if bits & CL_DEVICE_TYPE_ACCELERATOR != 0 {
            Self::Accelerator
        } else {
            Self::Other(bits)
        }
    }
}

/// Information about an OpenCL platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenClPlatformInfo {
    pub name: String,
    pub vendor: String,
    pub version: String,
    pub profile: String,
    pub extensions: Vec<String>,
}

/// Information about an OpenCL device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenClDeviceInfo {
    pub name: String,
    pub vendor: String,
    pub device_type: OpenClDeviceType,
    pub device_version: String,
    pub driver_version: String,
    pub compute_units: u32,
    pub global_mem_bytes: u64,
    pub max_workgroup_size: usize,
    pub max_clock_mhz: u32,
    pub extensions: Vec<String>,
    /// Index of the parent platform in the probe result.
    pub platform_index: usize,
}

impl OpenClDeviceInfo {
    /// Returns `true` if this device is a GPU.
    pub fn is_gpu(&self) -> bool {
        self.device_type == OpenClDeviceType::Gpu
    }

    /// Returns `true` if vendor string contains "Intel" (case-insensitive).
    pub fn is_intel(&self) -> bool {
        shared_is_intel_vendor(&self.vendor)
    }

    /// Returns `true` if this looks like an Intel Arc GPU.
    pub fn is_intel_arc(&self) -> bool {
        self.is_gpu() && shared_is_intel_arc_device(&self.vendor, &self.name)
    }
}

/// Aggregated result from an OpenCL probe.
#[derive(Debug, Clone, Default)]
pub struct OpenClProbeResult {
    pub platforms: Vec<OpenClPlatformInfo>,
    pub devices: Vec<OpenClDeviceInfo>,
    /// `true` if the OpenCL runtime library was loaded successfully.
    pub runtime_available: bool,
    /// Human-readable error if the probe failed.
    pub error: Option<String>,
}

impl OpenClProbeResult {
    /// Return only GPU devices.
    pub fn gpu_devices(&self) -> Vec<&OpenClDeviceInfo> {
        self.devices.iter().filter(|d| d.is_gpu()).collect()
    }

    /// Return only Intel devices.
    pub fn intel_devices(&self) -> Vec<&OpenClDeviceInfo> {
        self.devices.iter().filter(|d| d.is_intel()).collect()
    }

    /// Return only Intel Arc GPUs.
    pub fn intel_arc_devices(&self) -> Vec<&OpenClDeviceInfo> {
        self.devices.iter().filter(|d| d.is_intel_arc()).collect()
    }
}

/// Full probe result combining OpenCL with other probe sources.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub opencl: OpenClProbeResult,
    pub cpu: super::CpuCapabilities,
}

/// Raw native OpenCL tiny vector-add smoke result.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenClTinyVectorAddExecution {
    pub passed: bool,
    pub proof_stage: String,
    pub platform_index: Option<usize>,
    pub device_index: Option<usize>,
    pub platform_name: Option<String>,
    pub device_name: Option<String>,
    pub vendor: Option<String>,
    pub driver_version: Option<String>,
    pub kernel_name: String,
    pub input_len: usize,
    pub tolerance: f32,
    pub max_abs_error: Option<f32>,
    pub mean_abs_error: Option<f32>,
    pub enqueue_ms: Option<f64>,
    pub readback_ms: Option<f64>,
    pub kernel_execution: bool,
    pub fallback_used: bool,
    pub error: Option<String>,
}

impl ProbeResult {
    /// Build a full probe result from all sources.
    pub fn detect() -> Self {
        Self { opencl: probe_opencl(), cpu: super::probe_cpu() }
    }

    /// `true` if any Intel Arc GPU was found.
    pub fn has_intel_arc(&self) -> bool {
        !self.opencl.intel_arc_devices().is_empty()
    }
}

// ── Intel Arc detector ──────────────────────────────────────────────────────

/// Heuristic detector for Intel Arc GPUs.
pub struct IntelArcDetector;

impl IntelArcDetector {
    /// Returns `true` if the vendor/device pair looks like an Intel Arc.
    pub fn is_arc(vendor: &str, device_name: &str) -> bool {
        shared_is_intel_arc_device(vendor, device_name)
    }

    /// Returns `true` if the vendor looks like Intel (regardless of device).
    pub fn is_intel_vendor(vendor: &str) -> bool {
        shared_is_intel_vendor(vendor)
    }
}

// ── Dynamic library loading ─────────────────────────────────────────────────

/// Name of the OpenCL shared library per platform.
#[cfg(target_os = "windows")]
const OPENCL_LIB_NAME: &str = "OpenCL.dll";

#[cfg(target_os = "linux")]
const OPENCL_LIB_NAME: &str = "libOpenCL.so";

#[cfg(target_os = "macos")]
const OPENCL_LIB_NAME: &str = "libOpenCL.dylib";

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
const OPENCL_LIB_NAME: &str = "libOpenCL.so";

// OpenCL C function type aliases.
type ClGetPlatformIDs = unsafe extern "C" fn(u32, *mut usize, *mut u32) -> i32;
type ClGetPlatformInfo = unsafe extern "C" fn(usize, u32, usize, *mut u8, *mut usize) -> i32;
type ClGetDeviceIDs = unsafe extern "C" fn(usize, u64, u32, *mut usize, *mut u32) -> i32;
type ClGetDeviceInfo = unsafe extern "C" fn(usize, u32, usize, *mut u8, *mut usize) -> i32;
type ClCreateContext = unsafe extern "C" fn(
    *const isize,
    u32,
    *const usize,
    Option<unsafe extern "C" fn(*const i8, *const c_void, usize, *mut c_void)>,
    *mut c_void,
    *mut i32,
) -> usize;
type ClCreateCommandQueue = unsafe extern "C" fn(usize, usize, u64, *mut i32) -> usize;
type ClCreateProgramWithSource =
    unsafe extern "C" fn(usize, u32, *const *const i8, *const usize, *mut i32) -> usize;
type ClBuildProgram = unsafe extern "C" fn(
    usize,
    u32,
    *const usize,
    *const i8,
    Option<unsafe extern "C" fn(usize, *mut c_void)>,
    *mut c_void,
) -> i32;
type ClGetProgramBuildInfo =
    unsafe extern "C" fn(usize, usize, u32, usize, *mut u8, *mut usize) -> i32;
type ClCreateKernel = unsafe extern "C" fn(usize, *const i8, *mut i32) -> usize;
type ClCreateBuffer = unsafe extern "C" fn(usize, u64, usize, *mut c_void, *mut i32) -> usize;
type ClSetKernelArg = unsafe extern "C" fn(usize, u32, usize, *const c_void) -> i32;
type ClEnqueueWriteBuffer = unsafe extern "C" fn(
    usize,
    usize,
    u32,
    usize,
    usize,
    *const c_void,
    u32,
    *const usize,
    *mut usize,
) -> i32;
type ClEnqueueNDRangeKernel = unsafe extern "C" fn(
    usize,
    usize,
    u32,
    *const usize,
    *const usize,
    *const usize,
    u32,
    *const usize,
    *mut usize,
) -> i32;
type ClEnqueueReadBuffer = unsafe extern "C" fn(
    usize,
    usize,
    u32,
    usize,
    usize,
    *mut c_void,
    u32,
    *const usize,
    *mut usize,
) -> i32;
type ClFinish = unsafe extern "C" fn(usize) -> i32;
type ClReleaseMemObject = unsafe extern "C" fn(usize) -> i32;
type ClReleaseKernel = unsafe extern "C" fn(usize) -> i32;
type ClReleaseProgram = unsafe extern "C" fn(usize) -> i32;
type ClReleaseCommandQueue = unsafe extern "C" fn(usize) -> i32;
type ClReleaseContext = unsafe extern "C" fn(usize) -> i32;

/// Holds dynamically-loaded OpenCL function pointers.
struct OpenClFunctions {
    get_platform_ids: ClGetPlatformIDs,
    get_platform_info: ClGetPlatformInfo,
    get_device_ids: ClGetDeviceIDs,
    get_device_info: ClGetDeviceInfo,
    // Keep library alive while functions are in use.
    _lib: libloading::Library,
}

/// Holds dynamically-loaded OpenCL function pointers needed for kernel smoke.
struct OpenClExecutionFunctions {
    get_platform_ids: ClGetPlatformIDs,
    get_platform_info: ClGetPlatformInfo,
    get_device_ids: ClGetDeviceIDs,
    get_device_info: ClGetDeviceInfo,
    create_context: ClCreateContext,
    create_command_queue: ClCreateCommandQueue,
    create_program_with_source: ClCreateProgramWithSource,
    build_program: ClBuildProgram,
    get_program_build_info: ClGetProgramBuildInfo,
    create_kernel: ClCreateKernel,
    create_buffer: ClCreateBuffer,
    set_kernel_arg: ClSetKernelArg,
    enqueue_write_buffer: ClEnqueueWriteBuffer,
    enqueue_nd_range_kernel: ClEnqueueNDRangeKernel,
    enqueue_read_buffer: ClEnqueueReadBuffer,
    finish: ClFinish,
    release_mem_object: ClReleaseMemObject,
    release_kernel: ClReleaseKernel,
    release_program: ClReleaseProgram,
    release_command_queue: ClReleaseCommandQueue,
    release_context: ClReleaseContext,
    _lib: libloading::Library,
}

impl OpenClFunctions {
    /// Try to load the OpenCL shared library and resolve symbols.
    fn load() -> Result<Self, String> {
        // SAFETY: We load a well-known system library and resolve standard
        // OpenCL ICD entry points.  The loaded functions are called with
        // correct argument types matching the OpenCL C API.
        let lib = unsafe { libloading::Library::new(OPENCL_LIB_NAME) }
            .map_err(|e| format!("failed to load {OPENCL_LIB_NAME}: {e}"))?;

        unsafe {
            let get_platform_ids: ClGetPlatformIDs =
                *lib.get(b"clGetPlatformIDs\0").map_err(|e| format!("clGetPlatformIDs: {e}"))?;
            let get_platform_info: ClGetPlatformInfo =
                *lib.get(b"clGetPlatformInfo\0").map_err(|e| format!("clGetPlatformInfo: {e}"))?;
            let get_device_ids: ClGetDeviceIDs =
                *lib.get(b"clGetDeviceIDs\0").map_err(|e| format!("clGetDeviceIDs: {e}"))?;
            let get_device_info: ClGetDeviceInfo =
                *lib.get(b"clGetDeviceInfo\0").map_err(|e| format!("clGetDeviceInfo: {e}"))?;

            Ok(Self {
                get_platform_ids,
                get_platform_info,
                get_device_ids,
                get_device_info,
                _lib: lib,
            })
        }
    }
}

impl OpenClExecutionFunctions {
    fn load() -> Result<Self, String> {
        // SAFETY: We load a well-known system OpenCL ICD library and resolve
        // standard OpenCL entry points. Calls below use OpenCL C ABI-compatible
        // argument types.
        let lib = unsafe { libloading::Library::new(OPENCL_LIB_NAME) }
            .map_err(|e| format!("failed to load {OPENCL_LIB_NAME}: {e}"))?;

        unsafe {
            Ok(Self {
                get_platform_ids: *lib
                    .get(b"clGetPlatformIDs\0")
                    .map_err(|e| format!("clGetPlatformIDs: {e}"))?,
                get_platform_info: *lib
                    .get(b"clGetPlatformInfo\0")
                    .map_err(|e| format!("clGetPlatformInfo: {e}"))?,
                get_device_ids: *lib
                    .get(b"clGetDeviceIDs\0")
                    .map_err(|e| format!("clGetDeviceIDs: {e}"))?,
                get_device_info: *lib
                    .get(b"clGetDeviceInfo\0")
                    .map_err(|e| format!("clGetDeviceInfo: {e}"))?,
                create_context: *lib
                    .get(b"clCreateContext\0")
                    .map_err(|e| format!("clCreateContext: {e}"))?,
                create_command_queue: *lib
                    .get(b"clCreateCommandQueue\0")
                    .map_err(|e| format!("clCreateCommandQueue: {e}"))?,
                create_program_with_source: *lib
                    .get(b"clCreateProgramWithSource\0")
                    .map_err(|e| format!("clCreateProgramWithSource: {e}"))?,
                build_program: *lib
                    .get(b"clBuildProgram\0")
                    .map_err(|e| format!("clBuildProgram: {e}"))?,
                get_program_build_info: *lib
                    .get(b"clGetProgramBuildInfo\0")
                    .map_err(|e| format!("clGetProgramBuildInfo: {e}"))?,
                create_kernel: *lib
                    .get(b"clCreateKernel\0")
                    .map_err(|e| format!("clCreateKernel: {e}"))?,
                create_buffer: *lib
                    .get(b"clCreateBuffer\0")
                    .map_err(|e| format!("clCreateBuffer: {e}"))?,
                set_kernel_arg: *lib
                    .get(b"clSetKernelArg\0")
                    .map_err(|e| format!("clSetKernelArg: {e}"))?,
                enqueue_write_buffer: *lib
                    .get(b"clEnqueueWriteBuffer\0")
                    .map_err(|e| format!("clEnqueueWriteBuffer: {e}"))?,
                enqueue_nd_range_kernel: *lib
                    .get(b"clEnqueueNDRangeKernel\0")
                    .map_err(|e| format!("clEnqueueNDRangeKernel: {e}"))?,
                enqueue_read_buffer: *lib
                    .get(b"clEnqueueReadBuffer\0")
                    .map_err(|e| format!("clEnqueueReadBuffer: {e}"))?,
                finish: *lib.get(b"clFinish\0").map_err(|e| format!("clFinish: {e}"))?,
                release_mem_object: *lib
                    .get(b"clReleaseMemObject\0")
                    .map_err(|e| format!("clReleaseMemObject: {e}"))?,
                release_kernel: *lib
                    .get(b"clReleaseKernel\0")
                    .map_err(|e| format!("clReleaseKernel: {e}"))?,
                release_program: *lib
                    .get(b"clReleaseProgram\0")
                    .map_err(|e| format!("clReleaseProgram: {e}"))?,
                release_command_queue: *lib
                    .get(b"clReleaseCommandQueue\0")
                    .map_err(|e| format!("clReleaseCommandQueue: {e}"))?,
                release_context: *lib
                    .get(b"clReleaseContext\0")
                    .map_err(|e| format!("clReleaseContext: {e}"))?,
                _lib: lib,
            })
        }
    }
}

// ── Helper: query CL string info ────────────────────────────────────────────

fn query_string(func: ClGetPlatformInfo, handle: usize, param: u32) -> String {
    let mut size: usize = 0;
    // SAFETY: Querying buffer size with null output pointer is allowed by the
    // OpenCL spec.
    let rc = unsafe { func(handle, param, 0, std::ptr::null_mut(), &mut size) };
    if rc != CL_SUCCESS || size == 0 {
        return String::new();
    }
    let mut buf = vec![0u8; size];
    let rc = unsafe { func(handle, param, size, buf.as_mut_ptr(), std::ptr::null_mut()) };
    if rc != CL_SUCCESS {
        return String::new();
    }
    // Trim trailing NUL
    if buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8_lossy(&buf).trim().to_owned()
}

fn query_device_string(func: ClGetDeviceInfo, handle: usize, param: u32) -> String {
    let mut size: usize = 0;
    let rc = unsafe { func(handle, param, 0, std::ptr::null_mut(), &mut size) };
    if rc != CL_SUCCESS || size == 0 {
        return String::new();
    }
    let mut buf = vec![0u8; size];
    let rc = unsafe { func(handle, param, size, buf.as_mut_ptr(), std::ptr::null_mut()) };
    if rc != CL_SUCCESS {
        return String::new();
    }
    if buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8_lossy(&buf).trim().to_owned()
}

fn query_device_u32(func: ClGetDeviceInfo, handle: usize, param: u32) -> u32 {
    let mut val: u32 = 0;
    let rc = unsafe {
        func(
            handle,
            param,
            std::mem::size_of::<u32>(),
            (&mut val as *mut u32).cast(),
            std::ptr::null_mut(),
        )
    };
    if rc != CL_SUCCESS { 0 } else { val }
}

fn query_device_u64(func: ClGetDeviceInfo, handle: usize, param: u32) -> u64 {
    let mut val: u64 = 0;
    let rc = unsafe {
        func(
            handle,
            param,
            std::mem::size_of::<u64>(),
            (&mut val as *mut u64).cast(),
            std::ptr::null_mut(),
        )
    };
    if rc != CL_SUCCESS { 0 } else { val }
}

fn query_device_usize(func: ClGetDeviceInfo, handle: usize, param: u32) -> usize {
    let mut val: usize = 0;
    let rc = unsafe {
        func(
            handle,
            param,
            std::mem::size_of::<usize>(),
            (&mut val as *mut usize).cast(),
            std::ptr::null_mut(),
        )
    };
    if rc != CL_SUCCESS { 0 } else { val }
}

fn query_device_type(func: ClGetDeviceInfo, handle: usize) -> OpenClDeviceType {
    let mut val: u64 = 0;
    let rc = unsafe {
        func(
            handle,
            CL_DEVICE_TYPE,
            std::mem::size_of::<u64>(),
            (&mut val as *mut u64).cast(),
            std::ptr::null_mut(),
        )
    };
    if rc != CL_SUCCESS { OpenClDeviceType::Other(0) } else { OpenClDeviceType::from_bits(val) }
}

fn parse_extensions(ext_string: &str) -> Vec<String> {
    ext_string.split_whitespace().filter(|s| !s.is_empty()).map(ToOwned::to_owned).collect()
}

// ── Main probe function ─────────────────────────────────────────────────────

/// Probe the system for OpenCL platforms and devices.
///
/// Returns an [`OpenClProbeResult`] that is always safe to inspect — if the
/// OpenCL library is not installed the result will have `runtime_available =
/// false` and empty platform/device lists.
pub fn probe_opencl() -> OpenClProbeResult {
    let funcs = match OpenClFunctions::load() {
        Ok(f) => f,
        Err(e) => {
            log::debug!("OpenCL not available: {e}");
            return OpenClProbeResult {
                runtime_available: false,
                error: Some(e),
                ..Default::default()
            };
        }
    };

    let mut num_platforms: u32 = 0;
    // SAFETY: Standard OpenCL ICD call with valid pointer.
    let rc = unsafe { (funcs.get_platform_ids)(0, std::ptr::null_mut(), &mut num_platforms) };
    if rc != CL_SUCCESS || num_platforms == 0 {
        return OpenClProbeResult {
            runtime_available: true,
            error: Some(format!("clGetPlatformIDs returned {rc}, platforms={num_platforms}")),
            ..Default::default()
        };
    }

    let mut platform_ids = vec![0usize; num_platforms as usize];
    let rc = unsafe {
        (funcs.get_platform_ids)(num_platforms, platform_ids.as_mut_ptr(), std::ptr::null_mut())
    };
    if rc != CL_SUCCESS {
        return OpenClProbeResult {
            runtime_available: true,
            error: Some(format!("clGetPlatformIDs (fetch) returned {rc}")),
            ..Default::default()
        };
    }

    let mut platforms = Vec::with_capacity(num_platforms as usize);
    let mut all_devices = Vec::new();

    for (pidx, &plat_id) in platform_ids.iter().enumerate() {
        let name = query_string(funcs.get_platform_info, plat_id, CL_PLATFORM_NAME);
        let vendor = query_string(funcs.get_platform_info, plat_id, CL_PLATFORM_VENDOR);
        let version = query_string(funcs.get_platform_info, plat_id, CL_PLATFORM_VERSION);
        let profile = query_string(funcs.get_platform_info, plat_id, CL_PLATFORM_PROFILE);
        let ext_str = query_string(funcs.get_platform_info, plat_id, CL_PLATFORM_EXTENSIONS);

        platforms.push(OpenClPlatformInfo {
            name,
            vendor,
            version,
            profile,
            extensions: parse_extensions(&ext_str),
        });

        // Enumerate devices on this platform
        let mut num_devices: u32 = 0;
        let rc = unsafe {
            (funcs.get_device_ids)(
                plat_id,
                CL_DEVICE_TYPE_ALL,
                0,
                std::ptr::null_mut(),
                &mut num_devices,
            )
        };
        if rc != CL_SUCCESS || num_devices == 0 {
            continue;
        }

        let mut device_ids = vec![0usize; num_devices as usize];
        let rc = unsafe {
            (funcs.get_device_ids)(
                plat_id,
                CL_DEVICE_TYPE_ALL,
                num_devices,
                device_ids.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != CL_SUCCESS {
            continue;
        }

        for &dev_id in &device_ids {
            let dev_name = query_device_string(funcs.get_device_info, dev_id, CL_DEVICE_NAME);
            let dev_vendor = query_device_string(funcs.get_device_info, dev_id, CL_DEVICE_VENDOR);
            let dev_version = query_device_string(funcs.get_device_info, dev_id, CL_DEVICE_VERSION);
            let driver_version =
                query_device_string(funcs.get_device_info, dev_id, CL_DRIVER_VERSION);
            let ext_str = query_device_string(funcs.get_device_info, dev_id, CL_DEVICE_EXTENSIONS);
            let compute_units =
                query_device_u32(funcs.get_device_info, dev_id, CL_DEVICE_MAX_COMPUTE_UNITS);
            let global_mem_bytes =
                query_device_u64(funcs.get_device_info, dev_id, CL_DEVICE_GLOBAL_MEM_SIZE);
            let max_workgroup_size =
                query_device_usize(funcs.get_device_info, dev_id, CL_DEVICE_MAX_WORK_GROUP_SIZE);
            let max_clock_mhz =
                query_device_u32(funcs.get_device_info, dev_id, CL_DEVICE_MAX_CLOCK_FREQUENCY);
            let device_type = query_device_type(funcs.get_device_info, dev_id);

            all_devices.push(OpenClDeviceInfo {
                name: dev_name,
                vendor: dev_vendor,
                device_type,
                device_version: dev_version,
                driver_version,
                compute_units,
                global_mem_bytes,
                max_workgroup_size,
                max_clock_mhz,
                extensions: parse_extensions(&ext_str),
                platform_index: pidx,
            });
        }
    }

    OpenClProbeResult { platforms, devices: all_devices, runtime_available: true, error: None }
}

/// Convenience: list all OpenCL devices found on this system.
pub fn list_opencl_devices() -> Vec<OpenClDeviceInfo> {
    probe_opencl().devices
}

/// Convenience: returns `true` if at least one Intel Arc GPU is found via
/// OpenCL.
pub fn is_intel_arc_available() -> bool {
    probe_opencl().devices.iter().any(|d| d.is_intel_arc())
}

/// Compile and run a tiny vector-add kernel on the Arc 140V OpenCL device.
pub fn run_intel_arc_140v_tiny_vector_add_smoke() -> OpenClTinyVectorAddExecution {
    run_intel_arc_tiny_vector_add_smoke(
        name_matches_arc_140v,
        "Arc 140V OpenCL device was not visible",
    )
}

/// Compile and run a tiny vector-add kernel on the Arc A770 OpenCL device.
pub fn run_intel_arc_a770_tiny_vector_add_smoke() -> OpenClTinyVectorAddExecution {
    run_intel_arc_tiny_vector_add_smoke(
        name_matches_arc_a770,
        "Intel Arc A770 OpenCL device was not visible",
    )
}

fn run_intel_arc_tiny_vector_add_smoke(
    device_name_matches: fn(&str) -> bool,
    missing_device_error: &str,
) -> OpenClTinyVectorAddExecution {
    const KERNEL_NAME: &str = "tiny_vector_add";
    const INPUT_LEN: usize = 16;
    const TOLERANCE: f32 = 1.0e-6;

    let mut result = OpenClTinyVectorAddExecution {
        passed: false,
        proof_stage: "runtime_detected".to_owned(),
        platform_index: None,
        device_index: None,
        platform_name: None,
        device_name: None,
        vendor: None,
        driver_version: None,
        kernel_name: KERNEL_NAME.to_owned(),
        input_len: INPUT_LEN,
        tolerance: TOLERANCE,
        max_abs_error: None,
        mean_abs_error: None,
        enqueue_ms: None,
        readback_ms: None,
        kernel_execution: false,
        fallback_used: false,
        error: None,
    };

    let funcs = match OpenClExecutionFunctions::load() {
        Ok(funcs) => funcs,
        Err(err) => {
            result.error = Some(err);
            return result;
        }
    };

    let Some(selected_device) = find_matching_intel_arc_gpu_device(&funcs, device_name_matches)
    else {
        result.error = Some(missing_device_error.to_owned());
        return result;
    };

    result.platform_index = Some(selected_device.platform_index);
    result.device_index = Some(selected_device.device_index);
    result.platform_name = Some(selected_device.platform_name.clone());
    result.device_name = Some(selected_device.device_name.clone());
    result.vendor = Some(selected_device.vendor.clone());
    result.driver_version = Some(selected_device.driver_version.clone());

    match run_vector_add_on_device(
        &funcs,
        selected_device.platform_id,
        selected_device.device_id,
        INPUT_LEN,
        TOLERANCE,
    ) {
        Ok(metrics) => {
            result.proof_stage = "kernel_smoke_tested".to_owned();
            result.passed = metrics.passed;
            result.kernel_execution = true;
            result.max_abs_error = Some(metrics.max_abs_error);
            result.mean_abs_error = Some(metrics.mean_abs_error);
            result.enqueue_ms = Some(metrics.enqueue_ms);
            result.readback_ms = Some(metrics.readback_ms);
            if !metrics.passed {
                result.error = Some(format!(
                    "tiny OpenCL vector add exceeded tolerance: max_abs_error={} tolerance={}",
                    metrics.max_abs_error, TOLERANCE
                ));
            }
        }
        Err(err) => {
            result.error = Some(err);
        }
    }

    result
}

struct VectorAddMetrics {
    passed: bool,
    max_abs_error: f32,
    mean_abs_error: f32,
    enqueue_ms: f64,
    readback_ms: f64,
}

struct MatchedIntelArcGpuDevice {
    platform_id: usize,
    device_id: usize,
    platform_index: usize,
    device_index: usize,
    platform_name: String,
    device_name: String,
    vendor: String,
    driver_version: String,
}

fn find_matching_intel_arc_gpu_device(
    funcs: &OpenClExecutionFunctions,
    device_name_matches: fn(&str) -> bool,
) -> Option<MatchedIntelArcGpuDevice> {
    let mut num_platforms: u32 = 0;
    let rc = unsafe { (funcs.get_platform_ids)(0, ptr::null_mut(), &mut num_platforms) };
    if rc != CL_SUCCESS || num_platforms == 0 {
        return None;
    }

    let mut platform_ids = vec![0usize; num_platforms as usize];
    let rc = unsafe {
        (funcs.get_platform_ids)(num_platforms, platform_ids.as_mut_ptr(), ptr::null_mut())
    };
    if rc != CL_SUCCESS {
        return None;
    }

    for (platform_index, &platform_id) in platform_ids.iter().enumerate() {
        let platform_name = query_string(funcs.get_platform_info, platform_id, CL_PLATFORM_NAME);
        let mut num_devices: u32 = 0;
        let rc = unsafe {
            (funcs.get_device_ids)(
                platform_id,
                CL_DEVICE_TYPE_GPU,
                0,
                ptr::null_mut(),
                &mut num_devices,
            )
        };
        if rc != CL_SUCCESS || num_devices == 0 {
            continue;
        }

        let mut device_ids = vec![0usize; num_devices as usize];
        let rc = unsafe {
            (funcs.get_device_ids)(
                platform_id,
                CL_DEVICE_TYPE_GPU,
                num_devices,
                device_ids.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        if rc != CL_SUCCESS {
            continue;
        }

        for (device_index, &device_id) in device_ids.iter().enumerate() {
            let device_name = query_device_string(funcs.get_device_info, device_id, CL_DEVICE_NAME);
            let vendor = query_device_string(funcs.get_device_info, device_id, CL_DEVICE_VENDOR);
            if !device_name_matches(&device_name) || !shared_is_intel_vendor(&vendor) {
                continue;
            }
            let driver_version =
                query_device_string(funcs.get_device_info, device_id, CL_DRIVER_VERSION);
            return Some(MatchedIntelArcGpuDevice {
                platform_id,
                device_id,
                platform_index,
                device_index,
                platform_name,
                device_name,
                vendor,
                driver_version,
            });
        }
    }

    None
}

fn name_matches_arc_140v(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    (lower.contains("arc") && lower.contains("140v")) || lower.contains("64a0")
}

fn name_matches_arc_a770(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    (lower.contains("arc") && lower.contains("a770")) || lower.contains("56a0")
}

fn run_vector_add_on_device(
    funcs: &OpenClExecutionFunctions,
    _platform_id: usize,
    device_id: usize,
    input_len: usize,
    tolerance: f32,
) -> Result<VectorAddMetrics, String> {
    let kernel_source = CString::new(
        r#"
        __kernel void tiny_vector_add(
            __global const float* a,
            __global const float* b,
            __global float* out
        ) {
            const uint gid = get_global_id(0);
            out[gid] = a[gid] + b[gid];
        }
        "#,
    )
    .map_err(|err| format!("build OpenCL source CString: {err}"))?;
    let kernel_name =
        CString::new("tiny_vector_add").map_err(|err| format!("build kernel name: {err}"))?;

    let a: Vec<f32> = (0..input_len).map(|idx| idx as f32).collect();
    let b: Vec<f32> = (0..input_len).map(|idx| 0.5 + idx as f32 * 0.25).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(lhs, rhs)| lhs + rhs).collect();
    let mut actual = vec![0.0f32; input_len];
    let bytes = input_len * std::mem::size_of::<f32>();

    let mut err = CL_SUCCESS;
    let context = unsafe {
        (funcs.create_context)(ptr::null(), 1, &device_id, None, ptr::null_mut(), &mut err)
    };
    if err != CL_SUCCESS || context == 0 {
        return Err(format!("clCreateContext failed with {err}"));
    }

    let mut queue = 0usize;
    let mut program = 0usize;
    let mut kernel = 0usize;
    let mut a_buf = 0usize;
    let mut b_buf = 0usize;
    let mut out_buf = 0usize;

    let run_result = (|| {
        queue = unsafe { (funcs.create_command_queue)(context, device_id, 0, &mut err) };
        check_handle(queue, err, "clCreateCommandQueue")?;

        let source_ptr = kernel_source.as_ptr();
        let source_len = kernel_source.as_bytes().len();
        program = unsafe {
            (funcs.create_program_with_source)(context, 1, &source_ptr, &source_len, &mut err)
        };
        check_handle(program, err, "clCreateProgramWithSource")?;

        let build_rc = unsafe {
            (funcs.build_program)(program, 1, &device_id, ptr::null(), None, ptr::null_mut())
        };
        if build_rc != CL_SUCCESS {
            return Err(format!(
                "clBuildProgram failed with {build_rc}: {}",
                program_build_log(funcs, program, device_id)
            ));
        }

        kernel = unsafe { (funcs.create_kernel)(program, kernel_name.as_ptr(), &mut err) };
        check_handle(kernel, err, "clCreateKernel")?;

        a_buf = unsafe {
            (funcs.create_buffer)(context, CL_MEM_READ_ONLY, bytes, ptr::null_mut(), &mut err)
        };
        check_handle(a_buf, err, "clCreateBuffer(a)")?;
        b_buf = unsafe {
            (funcs.create_buffer)(context, CL_MEM_READ_ONLY, bytes, ptr::null_mut(), &mut err)
        };
        check_handle(b_buf, err, "clCreateBuffer(b)")?;
        out_buf = unsafe {
            (funcs.create_buffer)(context, CL_MEM_WRITE_ONLY, bytes, ptr::null_mut(), &mut err)
        };
        check_handle(out_buf, err, "clCreateBuffer(out)")?;

        let start_enqueue = Instant::now();
        check_rc(
            unsafe {
                (funcs.enqueue_write_buffer)(
                    queue,
                    a_buf,
                    CL_TRUE,
                    0,
                    bytes,
                    a.as_ptr().cast(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                )
            },
            "clEnqueueWriteBuffer(a)",
        )?;
        check_rc(
            unsafe {
                (funcs.enqueue_write_buffer)(
                    queue,
                    b_buf,
                    CL_TRUE,
                    0,
                    bytes,
                    b.as_ptr().cast(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                )
            },
            "clEnqueueWriteBuffer(b)",
        )?;

        set_kernel_mem_arg(funcs, kernel, 0, a_buf)?;
        set_kernel_mem_arg(funcs, kernel, 1, b_buf)?;
        set_kernel_mem_arg(funcs, kernel, 2, out_buf)?;

        let global = [input_len];
        check_rc(
            unsafe {
                (funcs.enqueue_nd_range_kernel)(
                    queue,
                    kernel,
                    1,
                    ptr::null(),
                    global.as_ptr(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                )
            },
            "clEnqueueNDRangeKernel",
        )?;
        check_rc(unsafe { (funcs.finish)(queue) }, "clFinish(kernel)")?;
        let enqueue_ms = start_enqueue.elapsed().as_secs_f64() * 1000.0;

        let start_readback = Instant::now();
        check_rc(
            unsafe {
                (funcs.enqueue_read_buffer)(
                    queue,
                    out_buf,
                    CL_TRUE,
                    0,
                    bytes,
                    actual.as_mut_ptr().cast(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                )
            },
            "clEnqueueReadBuffer(out)",
        )?;
        check_rc(unsafe { (funcs.finish)(queue) }, "clFinish(readback)")?;
        let readback_ms = start_readback.elapsed().as_secs_f64() * 1000.0;

        let mut max_abs_error = 0.0f32;
        let mut sum_abs_error = 0.0f32;
        for (actual, expected) in actual.iter().zip(&expected) {
            let error = (actual - expected).abs();
            max_abs_error = max_abs_error.max(error);
            sum_abs_error += error;
        }
        let mean_abs_error = sum_abs_error / input_len as f32;

        Ok(VectorAddMetrics {
            passed: max_abs_error <= tolerance,
            max_abs_error,
            mean_abs_error,
            enqueue_ms,
            readback_ms,
        })
    })();

    unsafe {
        if out_buf != 0 {
            let _ = (funcs.release_mem_object)(out_buf);
        }
        if b_buf != 0 {
            let _ = (funcs.release_mem_object)(b_buf);
        }
        if a_buf != 0 {
            let _ = (funcs.release_mem_object)(a_buf);
        }
        if kernel != 0 {
            let _ = (funcs.release_kernel)(kernel);
        }
        if program != 0 {
            let _ = (funcs.release_program)(program);
        }
        if queue != 0 {
            let _ = (funcs.release_command_queue)(queue);
        }
        let _ = (funcs.release_context)(context);
    }

    run_result
}

fn set_kernel_mem_arg(
    funcs: &OpenClExecutionFunctions,
    kernel: usize,
    index: u32,
    buffer: usize,
) -> Result<(), String> {
    check_rc(
        unsafe {
            (funcs.set_kernel_arg)(
                kernel,
                index,
                std::mem::size_of::<usize>(),
                (&buffer as *const usize).cast(),
            )
        },
        "clSetKernelArg",
    )
}

fn program_build_log(funcs: &OpenClExecutionFunctions, program: usize, device_id: usize) -> String {
    let mut size = 0usize;
    let rc = unsafe {
        (funcs.get_program_build_info)(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            0,
            ptr::null_mut(),
            &mut size,
        )
    };
    if rc != CL_SUCCESS || size == 0 {
        return "<build log unavailable>".to_owned();
    }

    let mut buf = vec![0u8; size];
    let rc = unsafe {
        (funcs.get_program_build_info)(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            size,
            buf.as_mut_ptr(),
            ptr::null_mut(),
        )
    };
    if rc != CL_SUCCESS {
        return format!("<build log query failed with {rc}>");
    }
    if buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8_lossy(&buf).trim().to_owned()
}

fn check_handle(handle: usize, rc: i32, label: &str) -> Result<(), String> {
    if rc == CL_SUCCESS && handle != 0 { Ok(()) } else { Err(format!("{label} failed with {rc}")) }
}

fn check_rc(rc: i32, label: &str) -> Result<(), String> {
    if rc == CL_SUCCESS { Ok(()) } else { Err(format!("{label} failed with {rc}")) }
}

// ── Mock / fallback helpers (always available, useful for testing) ───────────

/// Build an [`OpenClProbeResult`] representing a system with no OpenCL.
pub fn mock_no_opencl() -> OpenClProbeResult {
    OpenClProbeResult {
        runtime_available: false,
        error: Some("mock: OpenCL not available".into()),
        ..Default::default()
    }
}

/// Build a mock [`OpenClDeviceInfo`] for testing.
pub fn mock_device(
    name: &str,
    vendor: &str,
    device_type: OpenClDeviceType,
    compute_units: u32,
    global_mem_bytes: u64,
) -> OpenClDeviceInfo {
    OpenClDeviceInfo {
        name: name.to_owned(),
        vendor: vendor.to_owned(),
        device_type,
        device_version: "OpenCL 3.0".to_owned(),
        driver_version: "1.0.0".to_owned(),
        compute_units,
        global_mem_bytes,
        max_workgroup_size: 256,
        max_clock_mhz: 2100,
        extensions: vec![],
        platform_index: 0,
    }
}

/// Build a mock [`OpenClPlatformInfo`] for testing.
pub fn mock_platform(name: &str, vendor: &str) -> OpenClPlatformInfo {
    OpenClPlatformInfo {
        name: name.to_owned(),
        vendor: vendor.to_owned(),
        version: "OpenCL 3.0 (mock)".to_owned(),
        profile: "FULL_PROFILE".to_owned(),
        extensions: vec![],
    }
}

/// Build a mock probe result with the given platforms and devices.
pub fn mock_probe_result(
    platforms: Vec<OpenClPlatformInfo>,
    devices: Vec<OpenClDeviceInfo>,
) -> OpenClProbeResult {
    OpenClProbeResult { platforms, devices, runtime_available: true, error: None }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── OpenClDeviceType tests ──────────────────────────────────────────

    #[test]
    fn device_type_from_gpu_bit() {
        assert_eq!(OpenClDeviceType::from_bits(CL_DEVICE_TYPE_GPU), OpenClDeviceType::Gpu);
    }

    #[test]
    fn device_type_from_cpu_bit() {
        assert_eq!(OpenClDeviceType::from_bits(CL_DEVICE_TYPE_CPU), OpenClDeviceType::Cpu);
    }

    #[test]
    fn device_type_from_accel_bit() {
        assert_eq!(
            OpenClDeviceType::from_bits(CL_DEVICE_TYPE_ACCELERATOR),
            OpenClDeviceType::Accelerator,
        );
    }

    #[test]
    fn device_type_from_zero() {
        assert_eq!(OpenClDeviceType::from_bits(0), OpenClDeviceType::Other(0));
    }

    #[test]
    fn device_type_gpu_takes_priority() {
        // If both GPU and CPU bits set, GPU wins
        let bits = CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU;
        assert_eq!(OpenClDeviceType::from_bits(bits), OpenClDeviceType::Gpu);
    }

    #[test]
    fn device_type_display_gpu() {
        assert_eq!(format!("{}", OpenClDeviceType::Gpu), "GPU");
    }

    #[test]
    fn device_type_display_cpu() {
        assert_eq!(format!("{}", OpenClDeviceType::Cpu), "CPU");
    }

    #[test]
    fn device_type_display_accelerator() {
        assert_eq!(format!("{}", OpenClDeviceType::Accelerator), "Accelerator");
    }

    #[test]
    fn device_type_display_other() {
        let s = format!("{}", OpenClDeviceType::Other(0x42));
        assert!(s.contains("0x42"));
    }

    // ── OpenClDeviceInfo helpers ────────────────────────────────────────

    #[test]
    fn device_is_gpu_true() {
        let d = mock_device("Test GPU", "Vendor", OpenClDeviceType::Gpu, 16, 1024);
        assert!(d.is_gpu());
    }

    #[test]
    fn device_is_gpu_false_for_cpu() {
        let d = mock_device("Test CPU", "Vendor", OpenClDeviceType::Cpu, 8, 512);
        assert!(!d.is_gpu());
    }

    #[test]
    fn device_is_intel_true() {
        let d = mock_device("Test", "Intel(R) Corporation", OpenClDeviceType::Gpu, 16, 1024);
        assert!(d.is_intel());
    }

    #[test]
    fn device_is_intel_case_insensitive() {
        let d = mock_device("Test", "INTEL Corporation", OpenClDeviceType::Gpu, 16, 1024);
        assert!(d.is_intel());
    }

    #[test]
    fn device_is_intel_false_for_nvidia() {
        let d = mock_device("Test", "NVIDIA Corporation", OpenClDeviceType::Gpu, 16, 1024);
        assert!(!d.is_intel());
    }

    #[test]
    fn device_is_intel_arc_positive() {
        let d = mock_device(
            "Intel(R) Arc(TM) A770 Graphics",
            "Intel(R) Corporation",
            OpenClDeviceType::Gpu,
            512,
            16 * 1024 * 1024 * 1024,
        );
        assert!(d.is_intel_arc());
    }

    #[test]
    fn device_is_intel_arc_false_for_uhd() {
        let d = mock_device(
            "Intel(R) UHD Graphics 770",
            "Intel(R) Corporation",
            OpenClDeviceType::Gpu,
            32,
            4 * 1024 * 1024 * 1024,
        );
        assert!(!d.is_intel_arc());
    }

    #[test]
    fn device_is_intel_arc_false_for_cpu_type() {
        let d =
            mock_device("Intel Arc Something", "Intel Corporation", OpenClDeviceType::Cpu, 8, 1024);
        assert!(!d.is_intel_arc());
    }

    #[test]
    fn device_is_intel_arc_false_for_amd() {
        let d = mock_device(
            "Some Arc GPU",
            "Advanced Micro Devices",
            OpenClDeviceType::Gpu,
            64,
            8 * 1024 * 1024 * 1024,
        );
        assert!(!d.is_intel_arc());
    }

    // ── IntelArcDetector tests ─────────────────────────────────────────

    #[test]
    fn arc_detector_a770() {
        assert!(IntelArcDetector::is_arc("Intel(R) Corporation", "Intel(R) Arc(TM) A770 Graphics"));
    }

    #[test]
    fn selected_a770_smoke_matcher_accepts_only_a770_identity() {
        assert!(name_matches_arc_a770("Intel(R) Arc(TM) A770 Graphics"));
        assert!(name_matches_arc_a770("Intel Arc 0x56A0"));
        assert!(!name_matches_arc_a770("Intel(R) Arc(TM) 140V Graphics"));
        assert!(!name_matches_arc_a770("Intel(R) Arc(TM) A750 Graphics"));
        assert!(!name_matches_arc_a770("Intel(R) UHD Graphics 770"));
    }

    #[test]
    fn selected_arc140v_smoke_matcher_rejects_a770_identity() {
        assert!(name_matches_arc_140v("Intel(R) Arc(TM) 140V Graphics"));
        assert!(name_matches_arc_140v("Intel Arc 0x64A0"));
        assert!(!name_matches_arc_140v("Intel(R) Arc(TM) A770 Graphics"));
    }

    #[test]
    fn arc_detector_a750() {
        assert!(IntelArcDetector::is_arc("Intel Corporation", "Intel Arc A750"));
    }

    #[test]
    fn arc_detector_a580() {
        assert!(IntelArcDetector::is_arc("Intel", "Arc A580 Graphics"));
    }

    #[test]
    fn arc_detector_a380() {
        assert!(IntelArcDetector::is_arc("Intel(R) Corporation", "Intel(R) Arc(TM) A380 Graphics"));
    }

    #[test]
    fn arc_detector_a310() {
        assert!(IntelArcDetector::is_arc("Intel", "Arc A310"));
    }

    #[test]
    fn arc_detector_b580() {
        assert!(IntelArcDetector::is_arc("Intel Corporation", "Intel Arc B580"));
    }

    #[test]
    fn arc_detector_b570() {
        assert!(IntelArcDetector::is_arc("Intel", "Arc B570"));
    }

    #[test]
    fn arc_detector_pro() {
        assert!(IntelArcDetector::is_arc("Intel Corporation", "Intel Arc Pro A60M"));
    }

    #[test]
    fn arc_detector_generic_arc() {
        assert!(IntelArcDetector::is_arc("Intel", "Some Future Arc Device"));
    }

    #[test]
    fn arc_detector_reject_non_intel() {
        assert!(!IntelArcDetector::is_arc("AMD", "Arc GPU"));
    }

    #[test]
    fn arc_detector_reject_uhd() {
        assert!(!IntelArcDetector::is_arc("Intel Corporation", "Intel UHD Graphics 770"));
    }

    #[test]
    fn arc_detector_reject_iris() {
        assert!(!IntelArcDetector::is_arc("Intel Corporation", "Intel Iris Xe Graphics"));
    }

    #[test]
    fn arc_detector_case_insensitive_vendor() {
        assert!(IntelArcDetector::is_arc("INTEL", "Arc A770"));
    }

    #[test]
    fn arc_detector_case_insensitive_device() {
        assert!(IntelArcDetector::is_arc("Intel", "ARC A770"));
    }

    #[test]
    fn arc_detector_is_intel_vendor() {
        assert!(IntelArcDetector::is_intel_vendor("Intel(R) Corporation"));
        assert!(IntelArcDetector::is_intel_vendor("INTEL"));
        assert!(IntelArcDetector::is_intel_vendor("intel"));
        assert!(!IntelArcDetector::is_intel_vendor("NVIDIA"));
        assert!(!IntelArcDetector::is_intel_vendor("AMD"));
    }

    // ── OpenClProbeResult methods ──────────────────────────────────────

    #[test]
    fn probe_result_gpu_devices_filters_correctly() {
        let result = mock_probe_result(
            vec![mock_platform("Test", "Intel")],
            vec![
                mock_device("GPU1", "Intel", OpenClDeviceType::Gpu, 16, 1024),
                mock_device("CPU1", "Intel", OpenClDeviceType::Cpu, 8, 512),
                mock_device("GPU2", "NVIDIA", OpenClDeviceType::Gpu, 32, 2048),
            ],
        );
        let gpus = result.gpu_devices();
        assert_eq!(gpus.len(), 2);
        assert!(gpus.iter().all(|d| d.is_gpu()));
    }

    #[test]
    fn probe_result_intel_devices_filters_correctly() {
        let result = mock_probe_result(
            vec![mock_platform("Test", "Mixed")],
            vec![
                mock_device("GPU1", "Intel Corporation", OpenClDeviceType::Gpu, 16, 1024),
                mock_device("GPU2", "NVIDIA Corporation", OpenClDeviceType::Gpu, 32, 2048),
                mock_device("CPU1", "Intel(R)", OpenClDeviceType::Cpu, 8, 512),
            ],
        );
        let intel = result.intel_devices();
        assert_eq!(intel.len(), 2);
    }

    #[test]
    fn probe_result_intel_arc_devices() {
        let result = mock_probe_result(
            vec![mock_platform("Intel OpenCL", "Intel Corporation")],
            vec![
                mock_device(
                    "Intel Arc A770",
                    "Intel Corporation",
                    OpenClDeviceType::Gpu,
                    512,
                    16 << 30,
                ),
                mock_device(
                    "Intel UHD 770",
                    "Intel Corporation",
                    OpenClDeviceType::Gpu,
                    32,
                    4 << 30,
                ),
                mock_device("RTX 4090", "NVIDIA", OpenClDeviceType::Gpu, 128, 24 << 30),
            ],
        );
        let arcs = result.intel_arc_devices();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].name, "Intel Arc A770");
    }

    #[test]
    fn probe_result_empty_when_no_devices() {
        let result = mock_probe_result(vec![], vec![]);
        assert!(result.gpu_devices().is_empty());
        assert!(result.intel_devices().is_empty());
        assert!(result.intel_arc_devices().is_empty());
    }

    // ── Mock helpers ───────────────────────────────────────────────────

    #[test]
    fn mock_no_opencl_has_correct_state() {
        let result = mock_no_opencl();
        assert!(!result.runtime_available);
        assert!(result.error.is_some());
        assert!(result.platforms.is_empty());
        assert!(result.devices.is_empty());
    }

    #[test]
    fn mock_device_populates_fields() {
        let d = mock_device("My GPU", "My Vendor", OpenClDeviceType::Gpu, 64, 8192);
        assert_eq!(d.name, "My GPU");
        assert_eq!(d.vendor, "My Vendor");
        assert_eq!(d.device_type, OpenClDeviceType::Gpu);
        assert_eq!(d.compute_units, 64);
        assert_eq!(d.global_mem_bytes, 8192);
        assert_eq!(d.max_workgroup_size, 256);
    }

    #[test]
    fn mock_platform_populates_fields() {
        let p = mock_platform("My Platform", "My Vendor");
        assert_eq!(p.name, "My Platform");
        assert_eq!(p.vendor, "My Vendor");
        assert_eq!(p.profile, "FULL_PROFILE");
    }

    // ── ProbeResult ────────────────────────────────────────────────────

    #[test]
    fn full_probe_result_detect_runs() {
        let result = ProbeResult::detect();
        // CPU probe always works
        assert!(result.cpu.core_count >= 1);
    }

    #[test]
    fn full_probe_result_has_intel_arc_with_mock() {
        let mut pr = ProbeResult::detect();
        // Override opencl with mock containing Arc
        pr.opencl = mock_probe_result(
            vec![mock_platform("Intel", "Intel")],
            vec![mock_device(
                "Intel Arc A770",
                "Intel Corporation",
                OpenClDeviceType::Gpu,
                512,
                16 << 30,
            )],
        );
        assert!(pr.has_intel_arc());
    }

    #[test]
    fn full_probe_result_no_intel_arc_without_devices() {
        let mut pr = ProbeResult::detect();
        pr.opencl = mock_no_opencl();
        assert!(!pr.has_intel_arc());
    }

    // ── parse_extensions ───────────────────────────────────────────────

    #[test]
    fn parse_extensions_empty() {
        assert!(parse_extensions("").is_empty());
    }

    #[test]
    fn parse_extensions_single() {
        let exts = parse_extensions("cl_khr_fp64");
        assert_eq!(exts, vec!["cl_khr_fp64"]);
    }

    #[test]
    fn parse_extensions_multiple() {
        let exts = parse_extensions("cl_khr_fp64 cl_khr_fp16 cl_intel_subgroups");
        assert_eq!(exts.len(), 3);
        assert_eq!(exts[0], "cl_khr_fp64");
        assert_eq!(exts[2], "cl_intel_subgroups");
    }

    #[test]
    fn parse_extensions_with_extra_whitespace() {
        let exts = parse_extensions("  cl_khr_fp64   cl_khr_fp16  ");
        assert_eq!(exts.len(), 2);
    }

    // ── Probe fallback (no OpenCL installed) ───────────────────────────

    #[test]
    fn probe_opencl_graceful_when_missing() {
        // This test runs even when OpenCL IS installed — it just validates
        // the probe doesn't panic.
        let result = probe_opencl();
        // runtime_available depends on the machine; just check no panic.
        let _ = result.runtime_available;
        let _ = result.platforms.len();
        let _ = result.devices.len();
    }

    #[test]
    fn list_opencl_devices_does_not_panic() {
        let devices = list_opencl_devices();
        // May be empty on machines without OpenCL
        let _ = devices.len();
    }

    #[test]
    fn is_intel_arc_available_does_not_panic() {
        let _ = is_intel_arc_available();
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn device_type_equality() {
        assert_eq!(OpenClDeviceType::Gpu, OpenClDeviceType::Gpu);
        assert_ne!(OpenClDeviceType::Gpu, OpenClDeviceType::Cpu);
    }

    #[test]
    fn device_type_clone() {
        let dt = OpenClDeviceType::Gpu;
        let dt2 = dt;
        assert_eq!(dt, dt2);
    }

    #[test]
    fn device_info_clone() {
        let d = mock_device("Test", "Vendor", OpenClDeviceType::Gpu, 16, 1024);
        let d2 = d.clone();
        assert_eq!(d, d2);
    }

    #[test]
    fn platform_info_clone() {
        let p = mock_platform("Test", "Vendor");
        let p2 = p.clone();
        assert_eq!(p, p2);
    }

    #[test]
    fn probe_result_default_is_empty() {
        let r = OpenClProbeResult::default();
        assert!(!r.runtime_available);
        assert!(r.platforms.is_empty());
        assert!(r.devices.is_empty());
        assert!(r.error.is_none());
    }

    #[test]
    fn multiple_arc_devices_detected() {
        let result = mock_probe_result(
            vec![mock_platform("Intel", "Intel")],
            vec![
                mock_device(
                    "Intel Arc A770",
                    "Intel Corporation",
                    OpenClDeviceType::Gpu,
                    512,
                    16 << 30,
                ),
                mock_device(
                    "Intel Arc A380",
                    "Intel Corporation",
                    OpenClDeviceType::Gpu,
                    128,
                    6 << 30,
                ),
            ],
        );
        assert_eq!(result.intel_arc_devices().len(), 2);
    }

    #[test]
    fn mixed_vendor_probe_result() {
        let result = mock_probe_result(
            vec![mock_platform("Intel OpenCL", "Intel"), mock_platform("NVIDIA CUDA", "NVIDIA")],
            vec![
                mock_device(
                    "Intel Arc B580",
                    "Intel Corporation",
                    OpenClDeviceType::Gpu,
                    256,
                    12 << 30,
                ),
                mock_device("RTX 4080", "NVIDIA Corporation", OpenClDeviceType::Gpu, 76, 16 << 30),
                mock_device(
                    "Intel Core i9",
                    "Intel Corporation",
                    OpenClDeviceType::Cpu,
                    24,
                    64 << 30,
                ),
            ],
        );
        assert_eq!(result.gpu_devices().len(), 2);
        assert_eq!(result.intel_devices().len(), 2);
        assert_eq!(result.intel_arc_devices().len(), 1);
    }
}
