//! Multi-device discovery and selection for Intel GPU (OpenCL) support.
//!
//! Provides enumeration, scoring, and selection among multiple OpenCL devices
//! with heuristics tuned for Intel Arc discrete GPUs.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors arising from multi-device discovery and selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiDeviceError {
    /// No OpenCL platforms were found on the system.
    NoPlatformsFound,
    /// Platforms exist but contain no devices.
    NoDevicesFound,
    /// No device matches the requested selector criteria.
    NoMatchingDevice,
    /// The requested device is not available.
    DeviceNotAvailable,
    /// The device is incompatible for the given reason.
    IncompatibleDevice(String),
}

impl fmt::Display for MultiDeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPlatformsFound => write!(f, "no OpenCL platforms found"),
            Self::NoDevicesFound => write!(f, "no OpenCL devices found"),
            Self::NoMatchingDevice => {
                write!(f, "no device matches the selector criteria")
            }
            Self::DeviceNotAvailable => {
                write!(f, "requested device is not available")
            }
            Self::IncompatibleDevice(reason) => {
                write!(f, "incompatible device: {reason}")
            }
        }
    }
}

impl std::error::Error for MultiDeviceError {}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Broad classification of an OpenCL device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    DiscreteGpu,
    IntegratedGpu,
    Cpu,
    Accelerator,
    Unknown,
}

/// Information about a single OpenCL device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor: String,
    pub device_type: DeviceType,
    pub global_mem_bytes: u64,
    pub local_mem_bytes: u32,
    pub max_compute_units: u32,
    pub max_workgroup_size: u32,
    pub max_clock_mhz: u32,
    pub driver_version: String,
    pub opencl_version: String,
    pub extensions: Vec<String>,
    pub pci_bus_id: Option<String>,
}

/// Composite score used to rank devices.
#[derive(Debug, Clone)]
pub struct DeviceScore {
    pub compute_score: f64,
    pub memory_score: f64,
    pub overall_score: f64,
    pub flags: Vec<String>,
}

/// User-specified device selection preference.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionPreference {
    HighestCompute,
    LargestMemory,
    LowestLatency,
    /// Select by name substring match.
    Specific(String),
    Auto,
}

/// Criteria used to filter and rank devices.
#[derive(Debug, Clone)]
pub struct DeviceSelector {
    pub preference: SelectionPreference,
    pub min_memory_gb: Option<f32>,
    pub required_extensions: Vec<String>,
}

impl Default for DeviceSelector {
    fn default() -> Self {
        Self {
            preference: SelectionPreference::Auto,
            min_memory_gb: None,
            required_extensions: Vec::new(),
        }
    }
}

/// Information about a single OpenCL platform and its devices.
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    pub name: String,
    pub vendor: String,
    pub version: String,
    pub devices: Vec<DeviceInfo>,
}

/// Aggregated topology across all platforms.
#[derive(Debug, Clone)]
pub struct DeviceTopology {
    pub platforms: Vec<PlatformInfo>,
    pub discrete_gpu_count: usize,
    pub integrated_gpu_count: usize,
    pub total_device_count: usize,
}

// ---------------------------------------------------------------------------
// OpenCL kernel source for device micro-benchmarks
// ---------------------------------------------------------------------------

/// OpenCL C source containing simple micro-benchmark kernels.
pub const DEVICE_QUERY_SRC: &str = r#"
/// Simple memory copy kernel for bandwidth estimation.
__kernel void device_bandwidth_test(
    __global const float* src,
    __global float* dst,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid < count) {
        dst[gid] = src[gid];
    }
}

/// FMA loop kernel for compute throughput estimation.
__kernel void device_compute_test(
    __global float* out,
    const uint iterations)
{
    uint gid = get_global_id(0);
    float a = (float)gid * 0.001f;
    float b = 1.0f;
    for (uint i = 0; i < iterations; ++i) {
        b = fma(a, b, a);
    }
    out[gid] = b;
}
"#;

// ---------------------------------------------------------------------------
// Mock enumeration (CPU reference implementation)
// ---------------------------------------------------------------------------

/// Returns realistic mock platform/device data for testing.
///
/// Platforms returned:
/// 1. **Intel(R) OpenCL Graphics** – Arc A770 (discrete) + UHD 770 (integrated)
/// 2. **Intel(R) OpenCL** – CPU device
pub fn mock_enumerate_platforms() -> Vec<PlatformInfo> {
    let a770 = DeviceInfo {
        name: "Intel(R) Arc(TM) A770 Graphics".into(),
        vendor: "Intel(R) Corporation".into(),
        device_type: DeviceType::DiscreteGpu,
        global_mem_bytes: 16 * 1024 * 1024 * 1024, // 16 GB
        local_mem_bytes: 64 * 1024,                // 64 KB
        max_compute_units: 512,
        max_workgroup_size: 1024,
        max_clock_mhz: 2100,
        driver_version: "23.35.27191.42".into(),
        opencl_version: "OpenCL 3.0 NEO".into(),
        extensions: vec![
            "cl_khr_fp16".into(),
            "cl_khr_fp64".into(),
            "cl_khr_subgroups".into(),
            "cl_intel_subgroups".into(),
            "cl_intel_required_subgroup_size".into(),
            "cl_intel_dot_accumulate".into(),
        ],
        pci_bus_id: Some("0000:03:00.0".into()),
    };

    let uhd770 = DeviceInfo {
        name: "Intel(R) UHD Graphics 770".into(),
        vendor: "Intel(R) Corporation".into(),
        device_type: DeviceType::IntegratedGpu,
        global_mem_bytes: 4 * 1024 * 1024 * 1024, // 4 GB (shared)
        local_mem_bytes: 64 * 1024,
        max_compute_units: 32,
        max_workgroup_size: 512,
        max_clock_mhz: 1500,
        driver_version: "23.35.27191.42".into(),
        opencl_version: "OpenCL 3.0 NEO".into(),
        extensions: vec![
            "cl_khr_fp16".into(),
            "cl_khr_subgroups".into(),
            "cl_intel_subgroups".into(),
        ],
        pci_bus_id: Some("0000:00:02.0".into()),
    };

    let cpu_device = DeviceInfo {
        name: "Intel(R) Core(TM) i7-13700K".into(),
        vendor: "Intel(R) Corporation".into(),
        device_type: DeviceType::Cpu,
        global_mem_bytes: 32 * 1024 * 1024 * 1024, // 32 GB
        local_mem_bytes: 32 * 1024,
        max_compute_units: 24,
        max_workgroup_size: 8192,
        max_clock_mhz: 5400,
        driver_version: "2024.18.7.0.11".into(),
        opencl_version: "OpenCL 3.0".into(),
        extensions: vec!["cl_khr_fp64".into(), "cl_khr_global_int32_base_atomics".into()],
        pci_bus_id: None,
    };

    vec![
        PlatformInfo {
            name: "Intel(R) OpenCL Graphics".into(),
            vendor: "Intel(R) Corporation".into(),
            version: "OpenCL 3.0".into(),
            devices: vec![a770, uhd770],
        },
        PlatformInfo {
            name: "Intel(R) OpenCL".into(),
            vendor: "Intel(R) Corporation".into(),
            version: "OpenCL 3.0".into(),
            devices: vec![cpu_device],
        },
    ]
}

/// Build a [`DeviceTopology`] from a list of platforms.
pub fn build_topology(platforms: Vec<PlatformInfo>) -> DeviceTopology {
    let mut discrete = 0usize;
    let mut integrated = 0usize;
    let mut total = 0usize;
    for p in &platforms {
        for d in &p.devices {
            total += 1;
            match d.device_type {
                DeviceType::DiscreteGpu => discrete += 1,
                DeviceType::IntegratedGpu => integrated += 1,
                _ => {}
            }
        }
    }
    DeviceTopology {
        platforms,
        discrete_gpu_count: discrete,
        integrated_gpu_count: integrated,
        total_device_count: total,
    }
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Score a device using a weighted heuristic.
///
/// * `compute_score` = `compute_units * clock_mhz * (2.0 if discrete GPU)`
/// * `memory_score`  = `global_mem / 1 GB * (1.5 if > 8 GB)`
/// * `overall_score` = `0.6 * compute_score + 0.4 * memory_score`
pub fn score_device(info: &DeviceInfo) -> DeviceScore {
    let discrete_mult = if info.device_type == DeviceType::DiscreteGpu { 2.0 } else { 1.0 };
    let compute_score = info.max_compute_units as f64 * info.max_clock_mhz as f64 * discrete_mult;

    let mem_gb = info.global_mem_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let mem_mult = if mem_gb > 8.0 { 1.5 } else { 1.0 };
    let memory_score = mem_gb * mem_mult;

    let overall_score = 0.6 * compute_score + 0.4 * memory_score;

    let mut flags = Vec::new();
    if info.device_type == DeviceType::DiscreteGpu {
        flags.push("discrete".into());
    }
    if mem_gb > 8.0 {
        flags.push("high_memory".into());
    }
    if is_intel_arc(info) {
        flags.push("intel_arc".into());
    }

    DeviceScore { compute_score, memory_score, overall_score, flags }
}

// ---------------------------------------------------------------------------
// Filtering & selection helpers
// ---------------------------------------------------------------------------

/// Returns `true` if the device looks like an Intel Arc GPU.
///
/// Heuristic: name contains "Arc" **or** (vendor is Intel + discrete GPU +
/// more than 4 GB VRAM).
pub fn is_intel_arc(info: &DeviceInfo) -> bool {
    let name_match = info.name.contains("Arc");
    let vendor_intel = info.vendor.contains("Intel");
    let is_discrete = info.device_type == DeviceType::DiscreteGpu;
    let large_mem = info.global_mem_bytes > 4 * 1024 * 1024 * 1024;

    name_match || (vendor_intel && is_discrete && large_mem)
}

/// Infer device capabilities from its [`DeviceInfo`].
pub fn detect_device_capabilities(info: &DeviceInfo) -> Vec<String> {
    let mut caps = Vec::new();
    let has_ext = |name: &str| info.extensions.iter().any(|e| e == name);

    if has_ext("cl_khr_fp16") {
        caps.push("fp16".into());
    }
    if has_ext("cl_khr_fp64") {
        caps.push("fp64".into());
    }
    if has_ext("cl_intel_dot_accumulate") {
        caps.push("int8_dp4a".into());
    }
    if has_ext("cl_khr_subgroups") || has_ext("cl_intel_subgroups") {
        caps.push("subgroup_ops".into());
    }

    caps
}

/// Recommend a workgroup size based on device type.
pub fn recommend_workgroup_size(info: &DeviceInfo) -> u32 {
    match info.device_type {
        DeviceType::DiscreteGpu => 256,
        DeviceType::IntegratedGpu => 64,
        DeviceType::Cpu => 1,
        DeviceType::Accelerator => 128,
        DeviceType::Unknown => 64,
    }
}

/// Check whether a device passes the selector filters (memory + extensions).
fn device_passes_filter(info: &DeviceInfo, selector: &DeviceSelector) -> bool {
    if let Some(min_gb) = selector.min_memory_gb {
        let mem_gb = info.global_mem_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        if mem_gb < min_gb as f64 {
            return false;
        }
    }
    for ext in &selector.required_extensions {
        if !info.extensions.iter().any(|e| e == ext) {
            return false;
        }
    }
    true
}

/// Return all devices that pass the selector filters, together with their
/// scores, as `(platform_idx, device_idx, DeviceScore)` triples.
pub fn filter_devices(
    topology: &DeviceTopology,
    selector: &DeviceSelector,
) -> Vec<(usize, usize, DeviceScore)> {
    let mut results = Vec::new();
    for (pi, platform) in topology.platforms.iter().enumerate() {
        for (di, device) in platform.devices.iter().enumerate() {
            if device_passes_filter(device, selector) {
                let score = score_device(device);
                results.push((pi, di, score));
            }
        }
    }
    results
}

/// Select the single best device according to the given [`DeviceSelector`].
///
/// Returns `(platform_idx, device_idx)` into `topology.platforms`.
pub fn select_best_device(
    topology: &DeviceTopology,
    selector: &DeviceSelector,
) -> Result<(usize, usize), MultiDeviceError> {
    if topology.platforms.is_empty() {
        return Err(MultiDeviceError::NoPlatformsFound);
    }
    if topology.total_device_count == 0 {
        return Err(MultiDeviceError::NoDevicesFound);
    }

    // Handle Specific preference separately.
    if let SelectionPreference::Specific(ref name) = selector.preference {
        for (pi, platform) in topology.platforms.iter().enumerate() {
            for (di, device) in platform.devices.iter().enumerate() {
                if device.name.contains(name.as_str()) && device_passes_filter(device, selector) {
                    return Ok((pi, di));
                }
            }
        }
        return Err(MultiDeviceError::NoMatchingDevice);
    }

    let candidates = filter_devices(topology, selector);
    if candidates.is_empty() {
        return Err(MultiDeviceError::NoMatchingDevice);
    }

    let best = match selector.preference {
        SelectionPreference::HighestCompute => candidates.iter().max_by(|a, b| {
            a.2.compute_score.partial_cmp(&b.2.compute_score).unwrap_or(std::cmp::Ordering::Equal)
        }),
        SelectionPreference::LargestMemory => candidates.iter().max_by(|a, b| {
            a.2.memory_score.partial_cmp(&b.2.memory_score).unwrap_or(std::cmp::Ordering::Equal)
        }),
        SelectionPreference::LowestLatency => {
            // Prefer discrete GPUs for lowest latency; fall back to
            // overall score.
            candidates.iter().max_by(|a, b| {
                a.2.overall_score
                    .partial_cmp(&b.2.overall_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        }
        // Auto + Specific (handled above)
        _ => candidates.iter().max_by(|a, b| {
            a.2.overall_score.partial_cmp(&b.2.overall_score).unwrap_or(std::cmp::Ordering::Equal)
        }),
    };

    best.map(|(pi, di, _)| (*pi, *di)).ok_or(MultiDeviceError::NoMatchingDevice)
}

/// Produce a human-readable report of all platforms and devices.
pub fn format_device_report(topology: &DeviceTopology) -> String {
    let mut out = String::new();
    out.push_str("=== OpenCL Device Topology ===\n");
    out.push_str(&format!(
        "Platforms: {}  |  Devices: {} (discrete GPU: {}, \
         integrated GPU: {})\n\n",
        topology.platforms.len(),
        topology.total_device_count,
        topology.discrete_gpu_count,
        topology.integrated_gpu_count,
    ));

    for (pi, platform) in topology.platforms.iter().enumerate() {
        out.push_str(&format!("Platform {}: {} ({})\n", pi, platform.name, platform.version,));
        for (di, dev) in platform.devices.iter().enumerate() {
            let mem_gb = dev.global_mem_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let score = score_device(dev);
            out.push_str(&format!(
                "  [{},{}] {:40} {:?}  {:.1} GB  CU={:<4} \
                 Clock={:<5} MHz  Score={:.0}\n",
                pi,
                di,
                dev.name,
                dev.device_type,
                mem_gb,
                dev.max_compute_units,
                dev.max_clock_mhz,
                score.overall_score,
            ));
        }
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ----------------------------------------------------------

    fn mock_topology() -> DeviceTopology {
        build_topology(mock_enumerate_platforms())
    }

    fn default_selector() -> DeviceSelector {
        DeviceSelector::default()
    }

    fn a770_device() -> DeviceInfo {
        mock_enumerate_platforms()[0].devices[0].clone()
    }

    fn uhd_device() -> DeviceInfo {
        mock_enumerate_platforms()[0].devices[1].clone()
    }

    fn cpu_device() -> DeviceInfo {
        mock_enumerate_platforms()[1].devices[0].clone()
    }

    // -- Mock enumeration -------------------------------------------------

    #[test]
    fn mock_returns_two_platforms() {
        let platforms = mock_enumerate_platforms();
        assert_eq!(platforms.len(), 2);
    }

    #[test]
    fn mock_gpu_platform_has_two_devices() {
        let platforms = mock_enumerate_platforms();
        assert_eq!(platforms[0].devices.len(), 2);
    }

    #[test]
    fn mock_cpu_platform_has_one_device() {
        let platforms = mock_enumerate_platforms();
        assert_eq!(platforms[1].devices.len(), 1);
    }

    #[test]
    fn mock_a770_is_discrete() {
        assert_eq!(a770_device().device_type, DeviceType::DiscreteGpu);
    }

    #[test]
    fn mock_uhd_is_integrated() {
        assert_eq!(uhd_device().device_type, DeviceType::IntegratedGpu);
    }

    #[test]
    fn mock_cpu_device_type() {
        assert_eq!(cpu_device().device_type, DeviceType::Cpu);
    }

    // -- Topology builder -------------------------------------------------

    #[test]
    fn topology_counts_correct() {
        let topo = mock_topology();
        assert_eq!(topo.discrete_gpu_count, 1);
        assert_eq!(topo.integrated_gpu_count, 1);
        assert_eq!(topo.total_device_count, 3);
    }

    // -- Scoring ----------------------------------------------------------

    #[test]
    fn discrete_gpu_scores_higher_than_integrated() {
        let s_a770 = score_device(&a770_device());
        let s_uhd = score_device(&uhd_device());
        assert!(
            s_a770.overall_score > s_uhd.overall_score,
            "A770 ({}) should outscore UHD ({})",
            s_a770.overall_score,
            s_uhd.overall_score
        );
    }

    #[test]
    fn cpu_scores_lower_than_discrete_gpu() {
        let s_a770 = score_device(&a770_device());
        let s_cpu = score_device(&cpu_device());
        assert!(
            s_a770.overall_score > s_cpu.overall_score,
            "A770 ({}) should outscore CPU ({})",
            s_a770.overall_score,
            s_cpu.overall_score
        );
    }

    #[test]
    fn overall_score_is_non_negative() {
        for dev in [a770_device(), uhd_device(), cpu_device()] {
            let s = score_device(&dev);
            assert!(s.overall_score >= 0.0);
            assert!(s.compute_score >= 0.0);
            assert!(s.memory_score >= 0.0);
        }
    }

    #[test]
    fn scoring_is_deterministic() {
        let a = score_device(&a770_device());
        let b = score_device(&a770_device());
        assert_eq!(a.overall_score, b.overall_score);
        assert_eq!(a.compute_score, b.compute_score);
        assert_eq!(a.memory_score, b.memory_score);
    }

    #[test]
    fn a770_has_discrete_flag() {
        let s = score_device(&a770_device());
        assert!(s.flags.contains(&"discrete".to_string()));
    }

    #[test]
    fn a770_has_high_memory_flag() {
        let s = score_device(&a770_device());
        assert!(s.flags.contains(&"high_memory".to_string()));
    }

    #[test]
    fn a770_has_intel_arc_flag() {
        let s = score_device(&a770_device());
        assert!(s.flags.contains(&"intel_arc".to_string()));
    }

    // -- Auto selection ---------------------------------------------------

    #[test]
    fn auto_picks_a770() {
        let topo = mock_topology();
        let sel = default_selector();
        let (pi, di) = select_best_device(&topo, &sel).unwrap();
        assert_eq!(topo.platforms[pi].devices[di].name, "Intel(R) Arc(TM) A770 Graphics");
    }

    // -- HighestCompute preference ----------------------------------------

    #[test]
    fn highest_compute_picks_a770() {
        let topo = mock_topology();
        let sel = DeviceSelector {
            preference: SelectionPreference::HighestCompute,
            ..default_selector()
        };
        let (pi, di) = select_best_device(&topo, &sel).unwrap();
        assert_eq!(topo.platforms[pi].devices[di].name, "Intel(R) Arc(TM) A770 Graphics");
    }

    // -- LargestMemory preference -----------------------------------------

    #[test]
    fn largest_memory_picks_highest_mem_device() {
        let topo = mock_topology();
        let sel =
            DeviceSelector { preference: SelectionPreference::LargestMemory, ..default_selector() };
        let (pi, di) = select_best_device(&topo, &sel).unwrap();
        // CPU has 32 GB, largest memory_score (32 * 1.5 = 48).
        let chosen = &topo.platforms[pi].devices[di];
        assert_eq!(chosen.name, "Intel(R) Core(TM) i7-13700K");
    }

    // -- Specific preference ----------------------------------------------

    #[test]
    fn specific_finds_by_name_substring() {
        let topo = mock_topology();
        let sel = DeviceSelector {
            preference: SelectionPreference::Specific("UHD".into()),
            ..default_selector()
        };
        let (pi, di) = select_best_device(&topo, &sel).unwrap();
        assert!(topo.platforms[pi].devices[di].name.contains("UHD"));
    }

    #[test]
    fn specific_not_found_returns_error() {
        let topo = mock_topology();
        let sel = DeviceSelector {
            preference: SelectionPreference::Specific("NonexistentCard".into()),
            ..default_selector()
        };
        assert_eq!(select_best_device(&topo, &sel), Err(MultiDeviceError::NoMatchingDevice));
    }

    // -- Required extensions filter ---------------------------------------

    #[test]
    fn required_extensions_filter_works() {
        let topo = mock_topology();
        let sel = DeviceSelector {
            preference: SelectionPreference::Auto,
            required_extensions: vec!["cl_khr_fp16".into()],
            ..default_selector()
        };
        let candidates = filter_devices(&topo, &sel);
        // Only A770 and UHD have fp16.
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn required_extension_excludes_non_matching() {
        let topo = mock_topology();
        let sel = DeviceSelector {
            preference: SelectionPreference::Auto,
            required_extensions: vec!["cl_intel_dot_accumulate".into()],
            ..default_selector()
        };
        let candidates = filter_devices(&topo, &sel);
        // Only A770 has cl_intel_dot_accumulate.
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, 0); // platform 0
        assert_eq!(candidates[0].1, 0); // device 0
    }

    // -- Minimum memory filter --------------------------------------------

    #[test]
    fn min_memory_excludes_low_mem_devices() {
        let topo = mock_topology();
        let sel = DeviceSelector {
            preference: SelectionPreference::Auto,
            min_memory_gb: Some(8.0),
            ..default_selector()
        };
        let candidates = filter_devices(&topo, &sel);
        // A770 (16 GB) and CPU (32 GB) pass; UHD (4 GB) excluded.
        assert_eq!(candidates.len(), 2);
    }

    // -- Error paths ------------------------------------------------------

    #[test]
    fn no_platforms_returns_error() {
        let topo = build_topology(vec![]);
        let sel = default_selector();
        assert_eq!(select_best_device(&topo, &sel), Err(MultiDeviceError::NoPlatformsFound));
    }

    #[test]
    fn no_devices_returns_error() {
        let topo = build_topology(vec![PlatformInfo {
            name: "Empty".into(),
            vendor: "Test".into(),
            version: "1.0".into(),
            devices: vec![],
        }]);
        assert_eq!(
            select_best_device(&topo, &default_selector()),
            Err(MultiDeviceError::NoDevicesFound)
        );
    }

    #[test]
    fn no_matching_device_with_impossible_filter() {
        let topo = mock_topology();
        let sel = DeviceSelector {
            preference: SelectionPreference::Auto,
            min_memory_gb: Some(1024.0),
            ..default_selector()
        };
        assert_eq!(select_best_device(&topo, &sel), Err(MultiDeviceError::NoMatchingDevice));
    }

    // -- Intel Arc detection ----------------------------------------------

    #[test]
    fn a770_detected_as_arc() {
        assert!(is_intel_arc(&a770_device()));
    }

    #[test]
    fn uhd_not_detected_as_arc() {
        assert!(!is_intel_arc(&uhd_device()));
    }

    #[test]
    fn cpu_not_detected_as_arc() {
        assert!(!is_intel_arc(&cpu_device()));
    }

    // -- Capability detection ---------------------------------------------

    #[test]
    fn a770_has_fp16_and_dp4a() {
        let caps = detect_device_capabilities(&a770_device());
        assert!(caps.contains(&"fp16".to_string()));
        assert!(caps.contains(&"int8_dp4a".to_string()));
    }

    #[test]
    fn a770_has_subgroup_ops() {
        let caps = detect_device_capabilities(&a770_device());
        assert!(caps.contains(&"subgroup_ops".to_string()));
    }

    #[test]
    fn cpu_has_fp64_no_fp16() {
        let caps = detect_device_capabilities(&cpu_device());
        assert!(caps.contains(&"fp64".to_string()));
        assert!(!caps.contains(&"fp16".to_string()));
    }

    // -- Workgroup size recommendation ------------------------------------

    #[test]
    fn workgroup_256_for_discrete() {
        assert_eq!(recommend_workgroup_size(&a770_device()), 256);
    }

    #[test]
    fn workgroup_64_for_integrated() {
        assert_eq!(recommend_workgroup_size(&uhd_device()), 64);
    }

    #[test]
    fn workgroup_1_for_cpu() {
        assert_eq!(recommend_workgroup_size(&cpu_device()), 1);
    }

    // -- Report formatting ------------------------------------------------

    #[test]
    fn report_contains_header() {
        let report = format_device_report(&mock_topology());
        assert!(report.contains("OpenCL Device Topology"));
    }

    #[test]
    fn report_contains_device_names() {
        let report = format_device_report(&mock_topology());
        assert!(report.contains("A770"));
        assert!(report.contains("UHD"));
        assert!(report.contains("i7-13700K"));
    }

    #[test]
    fn report_contains_platform_count() {
        let report = format_device_report(&mock_topology());
        assert!(report.contains("Platforms: 2"));
    }

    // -- Edge cases -------------------------------------------------------

    #[test]
    fn empty_extensions_device_scores_without_panic() {
        let mut dev = a770_device();
        dev.extensions.clear();
        let s = score_device(&dev);
        assert!(s.overall_score >= 0.0);
    }

    #[test]
    fn zero_compute_units_scores_zero_compute() {
        let mut dev = a770_device();
        dev.max_compute_units = 0;
        let s = score_device(&dev);
        assert_eq!(s.compute_score, 0.0);
    }

    #[test]
    fn filter_devices_returns_all_when_no_constraints() {
        let topo = mock_topology();
        let sel = default_selector();
        let results = filter_devices(&topo, &sel);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn error_display_messages() {
        assert_eq!(MultiDeviceError::NoPlatformsFound.to_string(), "no OpenCL platforms found");
        assert_eq!(
            MultiDeviceError::IncompatibleDevice("bad".into()).to_string(),
            "incompatible device: bad"
        );
    }

    #[test]
    fn device_query_src_contains_kernels() {
        assert!(DEVICE_QUERY_SRC.contains("device_bandwidth_test"));
        assert!(DEVICE_QUERY_SRC.contains("device_compute_test"));
    }

    #[test]
    fn detect_capabilities_empty_extensions() {
        let mut dev = a770_device();
        dev.extensions.clear();
        let caps = detect_device_capabilities(&dev);
        assert!(caps.is_empty());
    }
}
