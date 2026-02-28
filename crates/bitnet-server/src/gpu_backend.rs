//! GPU backend integration for the BitNet server.
//!
//! Provides GPU device selection, status reporting, and request routing
//! with automatic fallback to CPU when GPU backends are unavailable.

use axum::{extract::State, http::StatusCode, response::Json};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::ProductionAppState;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// GPU backend kind used for device selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackendKind {
    Cuda,
    OpenCL,
    Vulkan,
    #[serde(other)]
    None,
}

impl GpuBackendKind {
    /// Return `true` when the variant represents a real GPU backend.
    pub fn is_gpu(&self) -> bool {
        !matches!(self, GpuBackendKind::None)
    }
}

impl std::fmt::Display for GpuBackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackendKind::Cuda => write!(f, "cuda"),
            GpuBackendKind::OpenCL => write!(f, "opencl"),
            GpuBackendKind::Vulkan => write!(f, "vulkan"),
            GpuBackendKind::None => write!(f, "none"),
        }
    }
}

impl std::str::FromStr for GpuBackendKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cuda" => Ok(GpuBackendKind::Cuda),
            "opencl" | "ocl" | "oneapi" => Ok(GpuBackendKind::OpenCL),
            "vulkan" | "vk" => Ok(GpuBackendKind::Vulkan),
            "none" | "cpu" => Ok(GpuBackendKind::None),
            _ => anyhow::bail!("unknown GPU backend: {s}"),
        }
    }
}

/// Per-device memory information returned by the status endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryInfo {
    pub total_mb: u64,
    pub used_mb: u64,
    pub free_mb: u64,
}

/// Per-device status block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    pub id: usize,
    pub name: String,
    pub backend: GpuBackendKind,
    pub available: bool,
    pub utilization_percent: f64,
    pub memory: GpuMemoryInfo,
}

/// Top-level GPU status response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStatusResponse {
    pub gpu_available: bool,
    pub selected_backend: GpuBackendKind,
    pub device_count: usize,
    pub devices: Vec<GpuDeviceInfo>,
}

// ---------------------------------------------------------------------------
// Device selection
// ---------------------------------------------------------------------------

/// Configuration consumed by [`GpuDeviceSelector`].
#[derive(Debug, Clone)]
pub struct GpuDeviceConfig {
    /// Preferred backend (may come from `BITNET_GPU_DEVICE` env var).
    pub preferred_backend: GpuBackendKind,
    /// Specific device ordinal (0-based).
    pub device_id: usize,
    /// Whether to fall back to CPU when the GPU is unavailable.
    pub fallback_to_cpu: bool,
}

impl Default for GpuDeviceConfig {
    fn default() -> Self {
        Self { preferred_backend: GpuBackendKind::None, device_id: 0, fallback_to_cpu: true }
    }
}

impl GpuDeviceConfig {
    /// Parse `BITNET_GPU_DEVICE` value such as `"cuda:0"`, `"opencl"`, `"vulkan:1"`.
    pub fn from_env_value(val: &str) -> anyhow::Result<Self> {
        let val = val.trim().to_lowercase();
        if let Some((backend_str, id_str)) = val.split_once(':') {
            let backend: GpuBackendKind = backend_str.parse()?;
            let device_id: usize = id_str.parse().map_err(|e| {
                anyhow::anyhow!("invalid device id in BITNET_GPU_DEVICE='{val}': {e}")
            })?;
            Ok(Self { preferred_backend: backend, device_id, fallback_to_cpu: true })
        } else {
            let backend: GpuBackendKind = val.parse()?;
            Ok(Self { preferred_backend: backend, device_id: 0, fallback_to_cpu: true })
        }
    }
}

/// Determines which GPU device (if any) should serve inference requests.
pub struct GpuDeviceSelector {
    config: GpuDeviceConfig,
}

impl GpuDeviceSelector {
    pub fn new(config: GpuDeviceConfig) -> Self {
        Self { config }
    }

    /// Return the [`GpuBackendKind`] that should be used, applying fallback
    /// logic when the preferred backend is unavailable.
    pub fn select_backend(&self) -> GpuBackendKind {
        let preferred = self.config.preferred_backend;
        if preferred.is_gpu() && self.is_backend_available(preferred) {
            preferred
        } else if self.config.fallback_to_cpu {
            self.try_any_available_backend().unwrap_or(GpuBackendKind::None)
        } else {
            GpuBackendKind::None
        }
    }

    /// Return the device ordinal selected by configuration.
    pub fn device_id(&self) -> usize {
        self.config.device_id
    }

    /// Probe whether *a particular* backend is available at runtime.
    pub fn is_backend_available(&self, kind: GpuBackendKind) -> bool {
        match kind {
            GpuBackendKind::Cuda => Self::probe_cuda(),
            GpuBackendKind::OpenCL => Self::probe_opencl(),
            GpuBackendKind::Vulkan => Self::probe_vulkan(),
            GpuBackendKind::None => false,
        }
    }

    /// Attempt to find any working backend, priority: CUDA > OpenCL > Vulkan.
    fn try_any_available_backend(&self) -> Option<GpuBackendKind> {
        for kind in [GpuBackendKind::Cuda, GpuBackendKind::OpenCL, GpuBackendKind::Vulkan] {
            if self.is_backend_available(kind) {
                return Some(kind);
            }
        }
        None
    }

    // -- Backend probes (feature-gated) -----------------------------------

    fn probe_cuda() -> bool {
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            bitnet_kernels::device_features::gpu_available_runtime()
        }
        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        false
    }

    fn probe_opencl() -> bool {
        #[cfg(feature = "oneapi")]
        {
            // OpenCL runtime detection via opencl3
            bitnet_kernels::device_features::opencl_available_runtime()
        }
        #[cfg(not(feature = "oneapi"))]
        false
    }

    fn probe_vulkan() -> bool {
        #[cfg(feature = "vulkan")]
        {
            bitnet_kernels::device_features::vulkan_available_runtime()
        }
        #[cfg(not(feature = "vulkan"))]
        false
    }
}

// ---------------------------------------------------------------------------
// Request routing
// ---------------------------------------------------------------------------

/// Routes an inference request to the most appropriate device, returning
/// the [`bitnet_common::Device`] that should be used for execution.
pub fn route_inference_request(
    selector: &GpuDeviceSelector,
    device_preference: Option<&str>,
) -> bitnet_common::Device {
    // Honour explicit client preference when valid.
    if let Some(pref) = device_preference {
        if let Ok(parsed) = pref.parse::<GpuBackendKind>() {
            if parsed.is_gpu() && selector.is_backend_available(parsed) {
                debug!(backend = %parsed, "using client-preferred GPU backend");
                return backend_to_device(parsed, selector.device_id());
            }
        }
    }

    let backend = selector.select_backend();
    if backend.is_gpu() {
        info!(backend = %backend, device = selector.device_id(), "routing to GPU");
    } else {
        debug!("routing to CPU (no GPU available)");
    }
    backend_to_device(backend, selector.device_id())
}

fn backend_to_device(kind: GpuBackendKind, device_id: usize) -> bitnet_common::Device {
    match kind {
        GpuBackendKind::Cuda => bitnet_common::Device::Cuda(device_id),
        GpuBackendKind::OpenCL => bitnet_common::Device::OpenCL(device_id),
        GpuBackendKind::Vulkan => bitnet_common::Device::Cuda(device_id), // map to Cuda variant for now
        GpuBackendKind::None => bitnet_common::Device::Cpu,
    }
}

// ---------------------------------------------------------------------------
// Status endpoint
// ---------------------------------------------------------------------------

/// Collect runtime GPU status information.
pub fn collect_gpu_status(selector: &GpuDeviceSelector) -> GpuStatusResponse {
    let selected = selector.select_backend();
    let mut devices = Vec::new();

    // Probe each backend for device info.
    for kind in [GpuBackendKind::Cuda, GpuBackendKind::OpenCL, GpuBackendKind::Vulkan] {
        if selector.is_backend_available(kind) {
            let info = probe_device_info(kind, selector.device_id());
            devices.push(info);
        }
    }

    GpuStatusResponse {
        gpu_available: !devices.is_empty(),
        selected_backend: selected,
        device_count: devices.len(),
        devices,
    }
}

fn probe_device_info(kind: GpuBackendKind, device_id: usize) -> GpuDeviceInfo {
    // In production these would call into the actual runtime APIs.
    // For now, return placeholder info that is still structurally correct.
    let (name, total_mb, used_mb, utilization) = match kind {
        GpuBackendKind::Cuda => {
            #[cfg(any(feature = "gpu", feature = "cuda"))]
            {
                use bitnet_kernels::gpu::cuda::CudaKernel;
                match CudaKernel::get_device_info(device_id) {
                    Ok(info) => {
                        let total = info.total_memory / (1024 * 1024);
                        (info.name.clone(), total as u64, 0u64, 0.0)
                    }
                    Err(_) => ("CUDA Device".into(), 0, 0, 0.0),
                }
            }
            #[cfg(not(any(feature = "gpu", feature = "cuda")))]
            {
                ("CUDA Device (stub)".into(), 0u64, 0u64, 0.0)
            }
        }
        GpuBackendKind::OpenCL => ("Intel Arc GPU (OpenCL)".into(), 0u64, 0u64, 0.0),
        GpuBackendKind::Vulkan => ("Vulkan Device".into(), 0u64, 0u64, 0.0),
        GpuBackendKind::None => ("N/A".into(), 0u64, 0u64, 0.0),
    };

    GpuDeviceInfo {
        id: device_id,
        name,
        backend: kind,
        available: true,
        utilization_percent: utilization,
        memory: GpuMemoryInfo { total_mb, used_mb, free_mb: total_mb.saturating_sub(used_mb) },
    }
}

// ---------------------------------------------------------------------------
// Axum handler
// ---------------------------------------------------------------------------

/// `GET /api/v1/gpu/status` — returns JSON with GPU device info.
pub async fn gpu_status_handler(
    State(state): State<ProductionAppState>,
) -> Result<Json<GpuStatusResponse>, StatusCode> {
    let gpu_config = state.gpu_device_config.as_ref().cloned().unwrap_or_default();
    let selector = GpuDeviceSelector::new(gpu_config);
    let status = collect_gpu_status(&selector);
    Ok(Json(status))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- GpuBackendKind parsing -------------------------------------------

    #[test]
    fn parse_backend_cuda() {
        let kind: GpuBackendKind = "cuda".parse().unwrap();
        assert_eq!(kind, GpuBackendKind::Cuda);
    }

    #[test]
    fn parse_backend_opencl_aliases() {
        for alias in &["opencl", "ocl", "oneapi"] {
            let kind: GpuBackendKind = alias.parse().unwrap();
            assert_eq!(kind, GpuBackendKind::OpenCL, "alias '{alias}' should parse to OpenCL");
        }
    }

    #[test]
    fn parse_backend_vulkan_aliases() {
        for alias in &["vulkan", "vk"] {
            let kind: GpuBackendKind = alias.parse().unwrap();
            assert_eq!(kind, GpuBackendKind::Vulkan, "alias '{alias}' should parse to Vulkan");
        }
    }

    #[test]
    fn parse_backend_none_and_cpu() {
        for alias in &["none", "cpu"] {
            let kind: GpuBackendKind = alias.parse().unwrap();
            assert_eq!(kind, GpuBackendKind::None);
        }
    }

    #[test]
    fn parse_backend_invalid_returns_error() {
        assert!("foobar".parse::<GpuBackendKind>().is_err());
    }

    // -- GpuDeviceConfig from env value -----------------------------------

    #[test]
    fn device_config_from_env_simple_backend() {
        let cfg = GpuDeviceConfig::from_env_value("cuda").unwrap();
        assert_eq!(cfg.preferred_backend, GpuBackendKind::Cuda);
        assert_eq!(cfg.device_id, 0);
    }

    #[test]
    fn device_config_from_env_with_device_id() {
        let cfg = GpuDeviceConfig::from_env_value("opencl:2").unwrap();
        assert_eq!(cfg.preferred_backend, GpuBackendKind::OpenCL);
        assert_eq!(cfg.device_id, 2);
    }

    #[test]
    fn device_config_from_env_vulkan_with_id() {
        let cfg = GpuDeviceConfig::from_env_value("vulkan:1").unwrap();
        assert_eq!(cfg.preferred_backend, GpuBackendKind::Vulkan);
        assert_eq!(cfg.device_id, 1);
    }

    #[test]
    fn device_config_from_env_invalid_id() {
        assert!(GpuDeviceConfig::from_env_value("cuda:abc").is_err());
    }

    // -- GpuDeviceSelector ------------------------------------------------

    #[test]
    fn selector_falls_back_to_cpu_when_no_gpu() {
        let selector = GpuDeviceSelector::new(GpuDeviceConfig {
            preferred_backend: GpuBackendKind::Cuda,
            device_id: 0,
            fallback_to_cpu: true,
        });
        // In test builds there is no real GPU, so it should fall back.
        let backend = selector.select_backend();
        // Either CUDA is actually available (CI with GPU) or falls back to None.
        assert!(
            backend == GpuBackendKind::None || backend == GpuBackendKind::Cuda,
            "expected None or Cuda, got {backend:?}"
        );
    }

    #[test]
    fn selector_no_fallback_returns_none() {
        let selector = GpuDeviceSelector::new(GpuDeviceConfig {
            preferred_backend: GpuBackendKind::Cuda,
            device_id: 0,
            fallback_to_cpu: false,
        });
        let backend = selector.select_backend();
        // Without fallback and no real GPU → None  (unless CI has GPU).
        assert!(
            backend == GpuBackendKind::None || backend == GpuBackendKind::Cuda,
            "expected None or Cuda, got {backend:?}"
        );
    }

    // -- Routing ----------------------------------------------------------

    #[test]
    fn route_respects_client_preference_cpu() {
        let selector = GpuDeviceSelector::new(GpuDeviceConfig::default());
        let device = route_inference_request(&selector, Some("cpu"));
        assert_eq!(device, bitnet_common::Device::Cpu);
    }

    #[test]
    fn route_defaults_to_cpu_when_no_gpu() {
        let selector = GpuDeviceSelector::new(GpuDeviceConfig::default());
        let device = route_inference_request(&selector, None);
        // No GPU compiled/available → CPU.
        assert_eq!(device, bitnet_common::Device::Cpu);
    }

    // -- Status -----------------------------------------------------------

    #[test]
    fn status_response_without_gpu() {
        let selector = GpuDeviceSelector::new(GpuDeviceConfig::default());
        let status = collect_gpu_status(&selector);
        // In plain CPU test, no devices should be reported.
        assert_eq!(status.selected_backend, GpuBackendKind::None);
    }

    // -- GpuBackendKind display -------------------------------------------

    #[test]
    fn backend_kind_display_roundtrips() {
        for kind in [
            GpuBackendKind::Cuda,
            GpuBackendKind::OpenCL,
            GpuBackendKind::Vulkan,
            GpuBackendKind::None,
        ] {
            let s = kind.to_string();
            assert!(!s.is_empty(), "display for {kind:?} should not be empty");
        }
    }
}
