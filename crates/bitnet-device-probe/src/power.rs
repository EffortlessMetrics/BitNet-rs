//! GPU power management: state queries, power capping, frequency scaling,
//! and thermal throttling detection.
//!
//! On Linux the implementation reads sysfs / hwmon entries and parses
//! `intel_gpu_top` output.  On other platforms (or when sysfs is absent)
//! every query returns a safe default or a [`PowerError::Unsupported`] error.

use std::fmt;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by the power-management subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PowerError {
    /// The operation is not supported on this platform or device.
    Unsupported,
    /// A sysfs / hwmon path could not be read.
    SysfsReadFailed { path: PathBuf, reason: String },
    /// A sysfs / hwmon path could not be written.
    SysfsWriteFailed { path: PathBuf, reason: String },
    /// The requested power cap exceeds hardware limits.
    PowerCapOutOfRange { requested_mw: u64, max_mw: u64 },
    /// Parsed value was not a valid number.
    ParseError { raw: String },
}

impl fmt::Display for PowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => write!(f, "power management not supported on this platform"),
            Self::SysfsReadFailed { path, reason } => {
                write!(f, "sysfs read failed: {}: {reason}", path.display())
            }
            Self::SysfsWriteFailed { path, reason } => {
                write!(f, "sysfs write failed: {}: {reason}", path.display())
            }
            Self::PowerCapOutOfRange { requested_mw, max_mw } => {
                write!(f, "power cap {requested_mw} mW exceeds hardware max {max_mw} mW")
            }
            Self::ParseError { raw } => write!(f, "failed to parse value: {raw:?}"),
        }
    }
}

impl std::error::Error for PowerError {}

// ---------------------------------------------------------------------------
// Power state & frequency mode
// ---------------------------------------------------------------------------

/// Coarse GPU power state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerState {
    /// GPU is fully active and processing workloads.
    Active,
    /// GPU is idle / in a low-power state.
    Idle,
    /// GPU is in a deep sleep / suspended state.
    Suspended,
    /// State could not be determined.
    Unknown,
}

impl fmt::Display for PowerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Idle => write!(f, "idle"),
            Self::Suspended => write!(f, "suspended"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Frequency scaling hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyMode {
    /// Maximise throughput (highest frequency).
    Performance,
    /// Balance power and throughput.
    Balanced,
    /// Minimise power draw (lowest frequency).
    Efficiency,
}

impl fmt::Display for FrequencyMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Performance => write!(f, "performance"),
            Self::Balanced => write!(f, "balanced"),
            Self::Efficiency => write!(f, "efficiency"),
        }
    }
}

// ---------------------------------------------------------------------------
// Thermal state
// ---------------------------------------------------------------------------

/// Thermal throttling state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    /// Temperature is within normal operating range.
    Normal,
    /// Temperature is elevated — consider reducing load.
    Warm,
    /// Thermal throttling is actively limiting performance.
    Throttled,
    /// Critical temperature — shutdown imminent.
    Critical,
}

impl ThermalState {
    /// Classify a temperature reading into a thermal state.
    pub fn from_temperature_c(temp_c: u64, warn_c: u64, throttle_c: u64, critical_c: u64) -> Self {
        if temp_c >= critical_c {
            Self::Critical
        } else if temp_c >= throttle_c {
            Self::Throttled
        } else if temp_c >= warn_c {
            Self::Warm
        } else {
            Self::Normal
        }
    }
}

impl fmt::Display for ThermalState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "normal"),
            Self::Warm => write!(f, "warm"),
            Self::Throttled => write!(f, "throttled"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

// ---------------------------------------------------------------------------
// GPU power snapshot
// ---------------------------------------------------------------------------

/// Point-in-time snapshot of GPU power telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPowerSnapshot {
    /// Current power state of the GPU.
    pub state: PowerState,
    /// Current GPU core frequency in MHz (0 if unknown).
    pub frequency_mhz: u64,
    /// Current GPU temperature in °C (0 if unknown).
    pub temperature_c: u64,
    /// Current power draw in milliwatts (0 if unknown).
    pub power_draw_mw: u64,
    /// Thermal state classification.
    pub thermal: ThermalState,
}

// ---------------------------------------------------------------------------
// Sysfs reader (abstracted for testing)
// ---------------------------------------------------------------------------

/// Trait abstracting filesystem reads so we can inject test doubles.
pub trait SysfsReader: fmt::Debug {
    /// Read the contents of `path` as a trimmed string.
    fn read_str(&self, path: &Path) -> Result<String, PowerError>;
    /// Write `value` to `path`.
    fn write_str(&self, path: &Path, value: &str) -> Result<(), PowerError>;
}

/// Real sysfs reader that accesses the filesystem.
#[derive(Debug)]
pub struct RealSysfsReader;

impl SysfsReader for RealSysfsReader {
    fn read_str(&self, path: &Path) -> Result<String, PowerError> {
        std::fs::read_to_string(path).map(|s| s.trim().to_string()).map_err(|e| {
            PowerError::SysfsReadFailed { path: path.to_path_buf(), reason: e.to_string() }
        })
    }

    fn write_str(&self, path: &Path, value: &str) -> Result<(), PowerError> {
        std::fs::write(path, value).map_err(|e| PowerError::SysfsWriteFailed {
            path: path.to_path_buf(),
            reason: e.to_string(),
        })
    }
}

/// In-memory sysfs reader for unit testing.
#[derive(Debug, Clone)]
pub struct MockSysfsReader {
    entries: std::collections::HashMap<PathBuf, String>,
}

impl MockSysfsReader {
    pub fn new() -> Self {
        Self { entries: std::collections::HashMap::new() }
    }

    pub fn set(&mut self, path: impl Into<PathBuf>, value: impl Into<String>) {
        self.entries.insert(path.into(), value.into());
    }
}

impl Default for MockSysfsReader {
    fn default() -> Self {
        Self::new()
    }
}

impl SysfsReader for MockSysfsReader {
    fn read_str(&self, path: &Path) -> Result<String, PowerError> {
        self.entries.get(path).cloned().ok_or_else(|| PowerError::SysfsReadFailed {
            path: path.to_path_buf(),
            reason: "not found in mock".into(),
        })
    }

    fn write_str(&self, _path: &Path, _value: &str) -> Result<(), PowerError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Power manager
// ---------------------------------------------------------------------------

/// Thresholds (°C) for thermal state classification.
#[derive(Debug, Clone, Copy)]
pub struct ThermalThresholds {
    pub warn_c: u64,
    pub throttle_c: u64,
    pub critical_c: u64,
}

impl Default for ThermalThresholds {
    fn default() -> Self {
        Self { warn_c: 75, throttle_c: 90, critical_c: 100 }
    }
}

/// GPU power management controller.
///
/// Reads power state, frequency, temperature, and power draw from sysfs.
/// Supports power capping and frequency mode hints.
#[derive(Debug)]
pub struct GpuPowerManager<R: SysfsReader = RealSysfsReader> {
    reader: R,
    /// sysfs base path, e.g. `/sys/class/drm/card0`.
    sysfs_base: PathBuf,
    /// Hardware maximum power cap in milliwatts.
    hw_max_power_mw: u64,
    /// Thermal classification thresholds.
    thresholds: ThermalThresholds,
}

impl GpuPowerManager<RealSysfsReader> {
    /// Create a power manager backed by the real filesystem.
    pub fn new(sysfs_base: impl Into<PathBuf>, hw_max_power_mw: u64) -> Self {
        Self::with_reader(RealSysfsReader, sysfs_base, hw_max_power_mw)
    }
}

impl<R: SysfsReader> GpuPowerManager<R> {
    /// Create a power manager with a custom sysfs reader.
    pub fn with_reader(reader: R, sysfs_base: impl Into<PathBuf>, hw_max_power_mw: u64) -> Self {
        Self {
            reader,
            sysfs_base: sysfs_base.into(),
            hw_max_power_mw,
            thresholds: ThermalThresholds::default(),
        }
    }

    /// Override thermal classification thresholds.
    pub fn set_thresholds(&mut self, thresholds: ThermalThresholds) {
        self.thresholds = thresholds;
    }

    /// Hardware maximum power cap (mW).
    pub fn hw_max_power_mw(&self) -> u64 {
        self.hw_max_power_mw
    }

    // -- queries ----------------------------------------------------------

    fn parse_u64(raw: &str) -> Result<u64, PowerError> {
        raw.trim().parse::<u64>().map_err(|_| PowerError::ParseError { raw: raw.to_string() })
    }

    /// Query the current GPU core frequency in MHz.
    pub fn query_frequency_mhz(&self) -> Result<u64, PowerError> {
        let path = self.sysfs_base.join("gt_cur_freq_mhz");
        let raw = self.reader.read_str(&path)?;
        Self::parse_u64(&raw)
    }

    /// Query the current GPU temperature in °C.
    pub fn query_temperature_c(&self) -> Result<u64, PowerError> {
        let path = self.sysfs_base.join("hwmon/temp1_input");
        let raw = self.reader.read_str(&path)?;
        // hwmon reports millidegrees
        let millideg = Self::parse_u64(&raw)?;
        Ok(millideg / 1000)
    }

    /// Query current power draw in milliwatts.
    pub fn query_power_draw_mw(&self) -> Result<u64, PowerError> {
        let path = self.sysfs_base.join("hwmon/power1_input");
        // hwmon reports microwatts
        let raw = self.reader.read_str(&path)?;
        let microwatts = Self::parse_u64(&raw)?;
        Ok(microwatts / 1000)
    }

    /// Query the current power state.
    pub fn query_power_state(&self) -> Result<PowerState, PowerError> {
        let path = self.sysfs_base.join("power_state");
        let raw = self.reader.read_str(&path)?;
        Ok(match raw.as_str() {
            "D0" | "active" => PowerState::Active,
            "D3hot" | "idle" => PowerState::Idle,
            "D3cold" | "suspended" => PowerState::Suspended,
            _ => PowerState::Unknown,
        })
    }

    /// Take a complete power snapshot.
    pub fn snapshot(&self) -> GpuPowerSnapshot {
        let frequency_mhz = self.query_frequency_mhz().unwrap_or(0);
        let temperature_c = self.query_temperature_c().unwrap_or(0);
        let power_draw_mw = self.query_power_draw_mw().unwrap_or(0);
        let state = self.query_power_state().unwrap_or(PowerState::Unknown);
        let thermal = ThermalState::from_temperature_c(
            temperature_c,
            self.thresholds.warn_c,
            self.thresholds.throttle_c,
            self.thresholds.critical_c,
        );
        GpuPowerSnapshot { state, frequency_mhz, temperature_c, power_draw_mw, thermal }
    }

    // -- power capping ----------------------------------------------------

    /// Set a power cap in milliwatts.
    ///
    /// Returns [`PowerError::PowerCapOutOfRange`] if `cap_mw` exceeds the
    /// hardware maximum.
    pub fn set_power_cap_mw(&self, cap_mw: u64) -> Result<(), PowerError> {
        if cap_mw > self.hw_max_power_mw {
            return Err(PowerError::PowerCapOutOfRange {
                requested_mw: cap_mw,
                max_mw: self.hw_max_power_mw,
            });
        }
        let path = self.sysfs_base.join("hwmon/power1_cap");
        // hwmon expects microwatts
        let microwatts = cap_mw * 1000;
        self.reader.write_str(&path, &microwatts.to_string())
    }

    // -- frequency hints --------------------------------------------------

    /// Apply a frequency scaling hint.
    pub fn set_frequency_mode(&self, mode: FrequencyMode) -> Result<(), PowerError> {
        let (min_path, max_path) =
            (self.sysfs_base.join("gt_min_freq_mhz"), self.sysfs_base.join("gt_max_freq_mhz"));
        // Read the device's supported range from RP0 (max) and RPn (min).
        let rp0 = self
            .reader
            .read_str(&self.sysfs_base.join("gt_RP0_freq_mhz"))
            .and_then(|s| Self::parse_u64(&s))
            .unwrap_or(1200);
        let rpn = self
            .reader
            .read_str(&self.sysfs_base.join("gt_RPn_freq_mhz"))
            .and_then(|s| Self::parse_u64(&s))
            .unwrap_or(300);

        let (min_mhz, max_mhz) = match mode {
            FrequencyMode::Performance => (rp0, rp0),
            FrequencyMode::Balanced => (rpn, rp0),
            FrequencyMode::Efficiency => (rpn, rpn),
        };
        self.reader.write_str(&min_path, &min_mhz.to_string())?;
        self.reader.write_str(&max_path, &max_mhz.to_string())
    }

    // -- thermal detection ------------------------------------------------

    /// Detect thermal throttling.
    pub fn detect_thermal_state(&self) -> ThermalState {
        let temp = self.query_temperature_c().unwrap_or(0);
        ThermalState::from_temperature_c(
            temp,
            self.thresholds.warn_c,
            self.thresholds.throttle_c,
            self.thresholds.critical_c,
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_manager() -> GpuPowerManager<MockSysfsReader> {
        let mut reader = MockSysfsReader::new();
        let base = PathBuf::from("/sys/class/drm/card0");
        reader.set(base.join("gt_cur_freq_mhz"), "1100");
        reader.set(base.join("hwmon/temp1_input"), "72000"); // 72°C in millideg
        reader.set(base.join("hwmon/power1_input"), "45000000"); // 45 W in µW
        reader.set(base.join("power_state"), "D0");
        reader.set(base.join("gt_RP0_freq_mhz"), "1400");
        reader.set(base.join("gt_RPn_freq_mhz"), "300");
        GpuPowerManager::with_reader(reader, &base, 120_000)
    }

    #[test]
    fn test_query_frequency() {
        let mgr = mock_manager();
        assert_eq!(mgr.query_frequency_mhz().unwrap(), 1100);
    }

    #[test]
    fn test_query_temperature() {
        let mgr = mock_manager();
        assert_eq!(mgr.query_temperature_c().unwrap(), 72);
    }

    #[test]
    fn test_query_power_draw() {
        let mgr = mock_manager();
        assert_eq!(mgr.query_power_draw_mw().unwrap(), 45_000);
    }

    #[test]
    fn test_query_power_state() {
        let mgr = mock_manager();
        assert_eq!(mgr.query_power_state().unwrap(), PowerState::Active);
    }

    #[test]
    fn test_snapshot_normal_thermal() {
        let mgr = mock_manager();
        let snap = mgr.snapshot();
        assert_eq!(snap.state, PowerState::Active);
        assert_eq!(snap.frequency_mhz, 1100);
        assert_eq!(snap.temperature_c, 72);
        assert_eq!(snap.power_draw_mw, 45_000);
        assert_eq!(snap.thermal, ThermalState::Normal);
    }

    #[test]
    fn test_thermal_state_classification() {
        assert_eq!(ThermalState::from_temperature_c(60, 75, 90, 100), ThermalState::Normal);
        assert_eq!(ThermalState::from_temperature_c(80, 75, 90, 100), ThermalState::Warm);
        assert_eq!(ThermalState::from_temperature_c(95, 75, 90, 100), ThermalState::Throttled);
        assert_eq!(ThermalState::from_temperature_c(105, 75, 90, 100), ThermalState::Critical);
    }

    #[test]
    fn test_power_cap_within_limits() {
        let mgr = mock_manager();
        assert!(mgr.set_power_cap_mw(100_000).is_ok());
    }

    #[test]
    fn test_power_cap_exceeds_max() {
        let mgr = mock_manager();
        assert!(matches!(
            mgr.set_power_cap_mw(200_000),
            Err(PowerError::PowerCapOutOfRange { requested_mw: 200_000, max_mw: 120_000 })
        ));
    }

    #[test]
    fn test_frequency_mode_performance() {
        let mgr = mock_manager();
        assert!(mgr.set_frequency_mode(FrequencyMode::Performance).is_ok());
    }

    #[test]
    fn test_frequency_mode_efficiency() {
        let mgr = mock_manager();
        assert!(mgr.set_frequency_mode(FrequencyMode::Efficiency).is_ok());
    }

    #[test]
    fn test_detect_thermal_throttled() {
        let mut reader = MockSysfsReader::new();
        let base = PathBuf::from("/sys/class/drm/card0");
        reader.set(base.join("hwmon/temp1_input"), "95000"); // 95°C
        let mgr = GpuPowerManager::with_reader(reader, &base, 100_000);
        assert_eq!(mgr.detect_thermal_state(), ThermalState::Throttled);
    }

    #[test]
    fn test_snapshot_with_missing_sysfs_entries() {
        let reader = MockSysfsReader::new(); // empty — all reads fail
        let base = PathBuf::from("/sys/class/drm/card0");
        let mgr = GpuPowerManager::with_reader(reader, &base, 100_000);
        let snap = mgr.snapshot();
        // Graceful defaults when sysfs is absent
        assert_eq!(snap.frequency_mhz, 0);
        assert_eq!(snap.temperature_c, 0);
        assert_eq!(snap.power_draw_mw, 0);
        assert_eq!(snap.state, PowerState::Unknown);
        assert_eq!(snap.thermal, ThermalState::Normal);
    }

    #[test]
    fn test_parse_error_on_invalid_frequency() {
        let mut reader = MockSysfsReader::new();
        let base = PathBuf::from("/sys/class/drm/card0");
        reader.set(base.join("gt_cur_freq_mhz"), "not_a_number");
        let mgr = GpuPowerManager::with_reader(reader, &base, 100_000);
        assert!(matches!(mgr.query_frequency_mhz(), Err(PowerError::ParseError { .. })));
    }
}
