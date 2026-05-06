//! NVIDIA CUDA runtime probe for proof-bench receipts.

use serde::{Deserialize, Serialize};

/// Normalized CUDA/NVML probe facts for an NVIDIA CUDA lane.
///
/// This is runtime visibility only. A successful probe means a CUDA context was
/// created and device identity was recorded; it does not prove kernel execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NvidiaCudaProbe {
    pub available: bool,
    pub device_count: usize,
    pub selected_device_index: Option<usize>,
    pub selected_device_name: Option<String>,
    pub compute_capability: Option<String>,
    pub driver_version: Option<String>,
    pub cuda_runtime_version: Option<String>,
    pub cuda_toolkit_version: Option<String>,
    pub nvrtc_version: Option<String>,
    pub nvml_available: bool,
    pub vram_bytes: Option<u64>,
    pub power_limit_watts: Option<f64>,
    pub power_draw_watts: Option<f64>,
    pub temperature_c: Option<f64>,
    pub failure_reason: Option<String>,
}

impl NvidiaCudaProbe {
    fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            available: false,
            device_count: 0,
            selected_device_index: None,
            selected_device_name: None,
            compute_capability: None,
            driver_version: None,
            cuda_runtime_version: None,
            cuda_toolkit_version: query_nvcc_version(),
            nvrtc_version: None,
            nvml_available: false,
            vram_bytes: None,
            power_limit_watts: None,
            power_draw_watts: None,
            temperature_c: None,
            failure_reason: Some(reason.into()),
        }
    }

    #[cfg(feature = "cuda")]
    fn with_smi(mut self, smi: Option<NvidiaSmiGpuQuery>) -> Self {
        let Some(smi) = smi else {
            return self;
        };

        self.nvml_available = true;
        self.selected_device_name = self.selected_device_name.or(smi.name);
        self.compute_capability = self.compute_capability.or(smi.compute_capability);
        self.driver_version = self.driver_version.or(smi.driver_version);
        self.vram_bytes = self.vram_bytes.or(smi.vram_bytes);
        self.power_limit_watts = self.power_limit_watts.or(smi.power_limit_watts);
        self.power_draw_watts = self.power_draw_watts.or(smi.power_draw_watts);
        self.temperature_c = self.temperature_c.or(smi.temperature_c);
        self
    }
}

/// Probe the selected NVIDIA CUDA device.
///
/// `selected_device_index` defaults to device 0. The result never represents a
/// CPU fallback as success.
pub fn probe_nvidia_cuda(selected_device_index: Option<usize>) -> NvidiaCudaProbe {
    imp::probe_nvidia_cuda(selected_device_index)
}

#[cfg(feature = "cuda")]
mod imp {
    use super::{
        NvidiaCudaProbe, query_cuda_runtime_version, query_nvcc_version, query_nvidia_smi,
        query_nvrtc_version,
    };
    use cudarc::driver::{CudaContext, result::device as cu_device, sys::CUdevice_attribute};

    pub(super) fn probe_nvidia_cuda(selected_device_index: Option<usize>) -> NvidiaCudaProbe {
        let index = selected_device_index.unwrap_or(0);
        let smi = query_nvidia_smi(index);
        let cuda_toolkit_version = query_nvcc_version();
        let nvrtc_version = query_nvrtc_version();
        let driver_version = smi.as_ref().and_then(|query| query.driver_version.clone());
        let cuda_runtime_version = query_cuda_runtime_version();

        let device_count = match CudaContext::device_count() {
            Ok(count) if count > 0 => count as usize,
            Ok(_) => {
                return NvidiaCudaProbe::unavailable("no CUDA devices found").with_smi(smi);
            }
            Err(err) => {
                return NvidiaCudaProbe::unavailable(format!(
                    "CUDA device count query failed: {err:?}"
                ))
                .with_smi(smi);
            }
        };

        if index >= device_count {
            let mut probe = NvidiaCudaProbe::unavailable(format!(
                "requested CUDA device index {index} is out of range for {device_count} devices"
            ))
            .with_smi(smi);
            probe.device_count = device_count;
            return probe;
        }

        let ctx = match CudaContext::new(index) {
            Ok(ctx) => ctx,
            Err(err) => {
                let mut probe = NvidiaCudaProbe::unavailable(format!(
                    "failed to create CUDA context for device {index}: {err:?}"
                ))
                .with_smi(smi);
                probe.device_count = device_count;
                probe.selected_device_index = Some(index);
                return probe;
            }
        };

        let name = ctx.name().ok();
        let major =
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).ok();
        let minor =
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).ok();
        let compute_capability = major.zip(minor).map(|(major, minor)| format!("{major}.{minor}"));
        let vram_bytes = unsafe { cu_device::total_mem(ctx.cu_device()) }
            .ok()
            .map(|bytes| u64::try_from(bytes).unwrap_or(u64::MAX));

        NvidiaCudaProbe {
            available: true,
            device_count,
            selected_device_index: Some(index),
            selected_device_name: name,
            compute_capability,
            driver_version,
            cuda_runtime_version,
            cuda_toolkit_version,
            nvrtc_version,
            nvml_available: false,
            vram_bytes,
            power_limit_watts: None,
            power_draw_watts: None,
            temperature_c: None,
            failure_reason: None,
        }
        .with_smi(smi)
    }
}

#[cfg(not(feature = "cuda"))]
mod imp {
    use super::NvidiaCudaProbe;

    pub(super) fn probe_nvidia_cuda(_selected_device_index: Option<usize>) -> NvidiaCudaProbe {
        NvidiaCudaProbe::unavailable("compiled without the cuda feature")
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq)]
struct NvidiaSmiGpuQuery {
    name: Option<String>,
    driver_version: Option<String>,
    compute_capability: Option<String>,
    vram_bytes: Option<u64>,
    power_limit_watts: Option<f64>,
    power_draw_watts: Option<f64>,
    temperature_c: Option<f64>,
}

#[cfg(feature = "cuda")]
fn query_nvidia_smi(device_index: usize) -> Option<NvidiaSmiGpuQuery> {
    let device_index_arg = device_index.to_string();
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "-i",
            device_index_arg.as_str(),
            "--query-gpu=name,driver_version,memory.total,power.limit,power.draw,temperature.gpu,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().find(|line| !line.trim().is_empty())?;
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    if fields.len() < 7 {
        return None;
    }

    Some(NvidiaSmiGpuQuery {
        name: non_empty_string(fields[0]),
        driver_version: non_empty_string(fields[1]),
        vram_bytes: parse_mib_to_bytes(fields[2]),
        power_limit_watts: parse_f64_field(fields[3]),
        power_draw_watts: parse_f64_field(fields[4]),
        temperature_c: parse_f64_field(fields[5]),
        compute_capability: non_empty_string(fields[6]),
    })
}

#[cfg(feature = "cuda")]
fn query_nvrtc_version() -> Option<String> {
    type NvrtcVersionFn =
        unsafe extern "C" fn(*mut std::ffi::c_int, *mut std::ffi::c_int) -> std::ffi::c_int;

    for candidate in nvrtc_library_candidates() {
        // SAFETY: We only load the library long enough to look up and call the
        // stable `nvrtcVersion` function with valid out-pointers.
        let Ok(library) = (unsafe { libloading::Library::new(candidate) }) else {
            continue;
        };
        // SAFETY: Symbol type matches the documented NVRTC C API.
        let version = unsafe { library.get::<NvrtcVersionFn>(b"nvrtcVersion\0") };
        let Ok(version) = version else {
            continue;
        };
        let mut major = 0;
        let mut minor = 0;
        // SAFETY: `major` and `minor` are valid mutable out-pointers.
        if unsafe { version(&mut major, &mut minor) } == 0 {
            return Some(format!("{major}.{minor}"));
        }
    }
    None
}

#[cfg(feature = "cuda")]
fn query_cuda_runtime_version() -> Option<String> {
    type CudaRuntimeGetVersionFn = unsafe extern "C" fn(*mut std::ffi::c_int) -> std::ffi::c_int;

    for candidate in cudart_library_candidates() {
        // SAFETY: We only load the library long enough to look up and call the
        // stable `cudaRuntimeGetVersion` function with a valid out-pointer.
        let Ok(library) = (unsafe { libloading::Library::new(candidate) }) else {
            continue;
        };
        // SAFETY: Symbol type matches the documented CUDA Runtime C API.
        let version = unsafe { library.get::<CudaRuntimeGetVersionFn>(b"cudaRuntimeGetVersion\0") };
        let Ok(version) = version else {
            continue;
        };
        let mut raw_version = 0;
        // SAFETY: `raw_version` is a valid mutable out-pointer.
        if unsafe { version(&mut raw_version) } == 0 {
            return format_cuda_version(raw_version);
        }
    }
    None
}

#[cfg(feature = "cuda")]
fn nvrtc_library_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        &["nvrtc64_120_0.dll", "nvrtc64_120.dll", "nvrtc64_12.dll", "nvrtc64.dll", "nvrtc.dll"]
    }

    #[cfg(target_os = "linux")]
    {
        &["libnvrtc.so.12", "libnvrtc.so"]
    }

    #[cfg(target_os = "macos")]
    {
        &["libnvrtc.dylib"]
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        &["nvrtc"]
    }
}

#[cfg(feature = "cuda")]
fn cudart_library_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        &["cudart64_120.dll", "cudart64_12.dll", "cudart64.dll", "cudart.dll"]
    }

    #[cfg(target_os = "linux")]
    {
        &["libcudart.so.12", "libcudart.so"]
    }

    #[cfg(target_os = "macos")]
    {
        &["libcudart.dylib"]
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        &["cudart"]
    }
}

fn query_nvcc_version() -> Option<String> {
    let output = std::process::Command::new("nvcc").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().find_map(parse_nvcc_version_line)
}

fn parse_nvcc_version_line(line: &str) -> Option<String> {
    let release = line.split("release ").nth(1)?;
    let version = release.split([',', ' ']).find(|part| !part.is_empty())?;
    non_empty_string(version)
}

#[cfg(any(feature = "cuda", test))]
fn format_cuda_version(version: i32) -> Option<String> {
    if version <= 0 {
        return None;
    }

    let major = version / 1000;
    let minor = (version % 1000) / 10;
    let patch = version % 10;
    if patch == 0 {
        Some(format!("{major}.{minor}"))
    } else {
        Some(format!("{major}.{minor}.{patch}"))
    }
}

fn non_empty_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("[N/A]") {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

#[cfg(feature = "cuda")]
fn parse_f64_field(value: &str) -> Option<f64> {
    non_empty_string(value)?.parse::<f64>().ok()
}

#[cfg(feature = "cuda")]
fn parse_mib_to_bytes(value: &str) -> Option<u64> {
    let mib = non_empty_string(value)?.parse::<u64>().ok()?;
    mib.checked_mul(1024)?.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_version_int_formats_major_minor() {
        assert_eq!(format_cuda_version(12000).as_deref(), Some("12.0"));
        assert_eq!(format_cuda_version(12080).as_deref(), Some("12.8"));
        assert_eq!(format_cuda_version(0), None);
    }

    #[test]
    fn nvcc_release_line_parses_version() {
        assert_eq!(
            parse_nvcc_version_line("Cuda compilation tools, release 12.8, V12.8.93").as_deref(),
            Some("12.8")
        );
    }

    #[test]
    fn nvidia_cuda_probe_serializes_failure_shape() {
        let probe = NvidiaCudaProbe::unavailable("not available");
        let value = serde_json::to_value(probe).expect("probe should serialize");
        assert_eq!(value["available"], false);
        assert_eq!(value["failure_reason"], "not available");
    }

    #[test]
    fn nvidia_cuda_probe_never_panics() {
        let probe = probe_nvidia_cuda(Some(0));
        if probe.available {
            assert!(probe.selected_device_index.is_some());
            assert!(probe.failure_reason.is_none());
        } else {
            assert!(probe.failure_reason.is_some());
        }
    }
}
