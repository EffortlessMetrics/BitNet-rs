//! GPU power management and thermal monitoring.
//!
//! Provides [`PowerManager`] for tracking device power states, enforcing thermal
//! policies, and generating energy-efficiency reports.

// ── Power mode ──────────────────────────────────────────────────────────────

/// Operating power mode for a GPU device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PowerMode {
    /// Maximum clocks, full power.
    Performance,
    /// Moderate clocks, moderate power.
    Balanced,
    /// Low clocks, minimal power.
    Efficient,
    /// User-specified clock frequencies.
    Custom { gpu_clock_mhz: u32, mem_clock_mhz: u32 },
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for [`PowerManager`].
#[derive(Debug, Clone)]
pub struct PowerConfig {
    /// How often to poll device state, in milliseconds.
    pub polling_interval_ms: u64,
    /// Temperature (°C) at which thermal throttling activates.
    pub thermal_throttle_celsius: f64,
    /// Optional hard power ceiling in watts.
    pub power_limit_watts: Option<f64>,
    /// Default power mode applied to new devices.
    pub power_mode: PowerMode,
}

impl Default for PowerConfig {
    fn default() -> Self {
        Self {
            polling_interval_ms: 1000,
            thermal_throttle_celsius: 85.0,
            power_limit_watts: None,
            power_mode: PowerMode::Balanced,
        }
    }
}

// ── Per-device state ────────────────────────────────────────────────────────

/// Snapshot of a single device's power/thermal state.
#[derive(Debug, Clone)]
pub struct DevicePowerState {
    pub device_name: String,
    pub temperature_celsius: f64,
    pub power_draw_watts: f64,
    pub gpu_clock_mhz: u32,
    pub mem_clock_mhz: u32,
    pub fan_speed_percent: u32,
    pub throttling: bool,
    pub power_mode: PowerMode,
}

// ── Telemetry sample ────────────────────────────────────────────────────────

/// A single timestamped telemetry sample.
#[derive(Debug, Clone)]
pub struct PowerSample {
    pub timestamp_ms: u64,
    pub temperature: f64,
    pub power_watts: f64,
    pub gpu_clock: u32,
}

// ── Thermal policy ──────────────────────────────────────────────────────────

/// Temperature thresholds governing automatic responses.
#[derive(Debug, Clone)]
pub struct ThermalPolicy {
    /// Temperature at which a warning is emitted.
    pub warning_celsius: f64,
    /// Temperature at which the device should be throttled.
    pub throttle_celsius: f64,
    /// Temperature at which the device should be shut down.
    pub shutdown_celsius: f64,
}

impl Default for ThermalPolicy {
    fn default() -> Self {
        Self { warning_celsius: 75.0, throttle_celsius: 85.0, shutdown_celsius: 95.0 }
    }
}

/// Action recommended after evaluating a device's temperature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThermalAction {
    /// Temperature is within safe limits.
    Ok,
    /// Device should be throttled to reduce temperature.
    Throttle,
    /// Device should be shut down immediately.
    Shutdown,
}

// ── Report ──────────────────────────────────────────────────────────────────

/// Aggregated power/thermal report over a monitoring window.
#[derive(Debug, Clone)]
pub struct PowerReport {
    pub avg_power_watts: f64,
    pub peak_power_watts: f64,
    pub avg_temperature: f64,
    pub peak_temperature: f64,
    pub throttle_events: u32,
    pub energy_joules: f64,
    pub tokens_per_joule: f64,
    pub duration_seconds: f64,
}

// ── Manager ─────────────────────────────────────────────────────────────────

/// Central coordinator for GPU power management and thermal monitoring.
pub struct PowerManager {
    devices: Vec<DevicePowerState>,
    config: PowerConfig,
    history: Vec<PowerSample>,
    thermal_policy: ThermalPolicy,
    throttle_events: u32,
}

impl PowerManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: PowerConfig) -> Self {
        Self {
            devices: Vec::new(),
            config,
            history: Vec::new(),
            thermal_policy: ThermalPolicy::default(),
            throttle_events: 0,
        }
    }

    /// Create a manager with both power config and thermal policy.
    pub const fn with_thermal_policy(config: PowerConfig, thermal_policy: ThermalPolicy) -> Self {
        Self {
            devices: Vec::new(),
            config,
            history: Vec::new(),
            thermal_policy,
            throttle_events: 0,
        }
    }

    /// Insert or update the state for the named device.
    pub fn update(&mut self, device_name: &str, state: DevicePowerState) {
        if let Some(existing) = self.devices.iter_mut().find(|d| d.device_name == device_name) {
            if state.throttling && !existing.throttling {
                self.throttle_events += 1;
            }
            *existing = state;
        } else {
            if state.throttling {
                self.throttle_events += 1;
            }
            self.devices.push(state);
        }
    }

    /// Evaluate the thermal policy for a device by name.
    pub fn check_thermal(&self, device_name: &str) -> ThermalAction {
        let Some(dev) = self.current_state(device_name) else {
            return ThermalAction::Ok;
        };
        if dev.temperature_celsius >= self.thermal_policy.shutdown_celsius {
            ThermalAction::Shutdown
        } else if dev.temperature_celsius >= self.thermal_policy.throttle_celsius {
            ThermalAction::Throttle
        } else {
            ThermalAction::Ok
        }
    }

    /// Apply a power mode to the named device, adjusting clock fields for
    /// the [`PowerMode::Custom`] variant.
    pub fn apply_power_mode(&mut self, device_name: &str, mode: PowerMode) -> bool {
        let Some(dev) = self.devices.iter_mut().find(|d| d.device_name == device_name) else {
            return false;
        };
        match &mode {
            PowerMode::Performance => {
                dev.gpu_clock_mhz = 2100;
                dev.mem_clock_mhz = 1200;
            }
            PowerMode::Balanced => {
                dev.gpu_clock_mhz = 1500;
                dev.mem_clock_mhz = 900;
            }
            PowerMode::Efficient => {
                dev.gpu_clock_mhz = 800;
                dev.mem_clock_mhz = 600;
            }
            PowerMode::Custom { gpu_clock_mhz, mem_clock_mhz } => {
                dev.gpu_clock_mhz = *gpu_clock_mhz;
                dev.mem_clock_mhz = *mem_clock_mhz;
            }
        }
        dev.power_mode = mode;
        true
    }

    /// Return a reference to the current state of the named device.
    pub fn current_state(&self, device_name: &str) -> Option<&DevicePowerState> {
        self.devices.iter().find(|d| d.device_name == device_name)
    }

    /// Record a telemetry sample into the history buffer.
    pub fn record_sample(&mut self, sample: PowerSample) {
        self.history.push(sample);
    }

    /// Generate an aggregated report from recorded history.
    ///
    /// If no samples exist the report contains all zeros.
    pub fn generate_report(&self) -> PowerReport {
        if self.history.is_empty() {
            return PowerReport {
                avg_power_watts: 0.0,
                peak_power_watts: 0.0,
                avg_temperature: 0.0,
                peak_temperature: 0.0,
                throttle_events: self.throttle_events,
                energy_joules: 0.0,
                tokens_per_joule: 0.0,
                duration_seconds: 0.0,
            };
        }

        #[allow(clippy::cast_precision_loss)]
        let n = self.history.len() as f64;
        let sum_power: f64 = self.history.iter().map(|s| s.power_watts).sum();
        let sum_temp: f64 = self.history.iter().map(|s| s.temperature).sum();
        let peak_power =
            self.history.iter().map(|s| s.power_watts).fold(f64::NEG_INFINITY, f64::max);
        let peak_temp =
            self.history.iter().map(|s| s.temperature).fold(f64::NEG_INFINITY, f64::max);

        let first_ts = self.history.first().unwrap().timestamp_ms;
        let last_ts = self.history.last().unwrap().timestamp_ms;
        #[allow(clippy::cast_precision_loss)]
        let duration_seconds = (last_ts - first_ts) as f64 / 1000.0;

        let avg_power = sum_power / n;
        let energy_joules = avg_power * duration_seconds;

        PowerReport {
            avg_power_watts: avg_power,
            peak_power_watts: peak_power,
            avg_temperature: sum_temp / n,
            peak_temperature: peak_temp,
            throttle_events: self.throttle_events,
            energy_joules,
            tokens_per_joule: 0.0,
            duration_seconds,
        }
    }

    /// Calculate energy efficiency as tokens per joule.
    ///
    /// Returns 0.0 when energy is zero.
    pub fn energy_efficiency(&self, tokens: u64, duration_secs: f64) -> f64 {
        if self.history.is_empty() || duration_secs <= 0.0 {
            return 0.0;
        }
        let avg_power: f64 = self.history.iter().map(|s| s.power_watts).sum::<f64>() / {
            #[allow(clippy::cast_precision_loss)]
            let len = self.history.len() as f64;
            len
        };
        let energy = avg_power * duration_secs;
        if energy <= 0.0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let tok = tokens as f64;
        tok / energy
    }

    /// Returns `true` if the device temperature meets or exceeds the
    /// configured throttle threshold.
    pub fn should_throttle(&self, device_name: &str) -> bool {
        self.current_state(device_name)
            .is_some_and(|d| d.temperature_celsius >= self.thermal_policy.throttle_celsius)
    }

    /// Clear all recorded history samples and reset throttle counter.
    pub fn reset_history(&mut self) {
        self.history.clear();
        self.throttle_events = 0;
    }

    /// Return a reference to the thermal policy.
    pub const fn thermal_policy(&self) -> &ThermalPolicy {
        &self.thermal_policy
    }

    /// Return a reference to the power config.
    pub const fn config(&self) -> &PowerConfig {
        &self.config
    }

    /// Number of throttle events recorded so far.
    pub const fn throttle_events(&self) -> u32 {
        self.throttle_events
    }

    /// Return a slice of all tracked devices.
    pub fn devices(&self) -> &[DevicePowerState] {
        &self.devices
    }

    /// Return a slice of all recorded history samples.
    pub fn history(&self) -> &[PowerSample] {
        &self.history
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    fn default_manager() -> PowerManager {
        PowerManager::new(PowerConfig::default())
    }

    fn device_state(name: &str, temp: f64, power: f64) -> DevicePowerState {
        DevicePowerState {
            device_name: name.to_string(),
            temperature_celsius: temp,
            power_draw_watts: power,
            gpu_clock_mhz: 1500,
            mem_clock_mhz: 900,
            fan_speed_percent: 50,
            throttling: false,
            power_mode: PowerMode::Balanced,
        }
    }

    fn sample(ts: u64, temp: f64, watts: f64, clock: u32) -> PowerSample {
        PowerSample { timestamp_ms: ts, temperature: temp, power_watts: watts, gpu_clock: clock }
    }

    // ── thermal policy tests ────────────────────────────────────────────

    #[test]
    fn temp_below_all_thresholds_returns_ok() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 60.0, 100.0));
        assert_eq!(mgr.check_thermal("gpu0"), ThermalAction::Ok);
    }

    #[test]
    fn temp_at_throttle_returns_throttle() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 85.0, 100.0));
        assert_eq!(mgr.check_thermal("gpu0"), ThermalAction::Throttle);
    }

    #[test]
    fn temp_above_throttle_returns_throttle() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 90.0, 100.0));
        assert_eq!(mgr.check_thermal("gpu0"), ThermalAction::Throttle);
    }

    #[test]
    fn temp_at_shutdown_returns_shutdown() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 95.0, 100.0));
        assert_eq!(mgr.check_thermal("gpu0"), ThermalAction::Shutdown);
    }

    #[test]
    fn temp_above_shutdown_returns_shutdown() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 110.0, 100.0));
        assert_eq!(mgr.check_thermal("gpu0"), ThermalAction::Shutdown);
    }

    #[test]
    fn unknown_device_thermal_returns_ok() {
        let mgr = default_manager();
        assert_eq!(mgr.check_thermal("nonexistent"), ThermalAction::Ok);
    }

    #[test]
    fn custom_thermal_policy_thresholds() {
        let policy =
            ThermalPolicy { warning_celsius: 50.0, throttle_celsius: 60.0, shutdown_celsius: 70.0 };
        let mut mgr = PowerManager::with_thermal_policy(PowerConfig::default(), policy);
        mgr.update("gpu0", device_state("gpu0", 65.0, 100.0));
        assert_eq!(mgr.check_thermal("gpu0"), ThermalAction::Throttle);
    }

    #[test]
    fn custom_policy_shutdown_threshold() {
        let policy =
            ThermalPolicy { warning_celsius: 50.0, throttle_celsius: 60.0, shutdown_celsius: 70.0 };
        let mut mgr = PowerManager::with_thermal_policy(PowerConfig::default(), policy);
        mgr.update("gpu0", device_state("gpu0", 72.0, 100.0));
        assert_eq!(mgr.check_thermal("gpu0"), ThermalAction::Shutdown);
    }

    // ── should_throttle ─────────────────────────────────────────────────

    #[test]
    fn should_throttle_false_when_cool() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 50.0, 100.0));
        assert!(!mgr.should_throttle("gpu0"));
    }

    #[test]
    fn should_throttle_true_when_hot() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 90.0, 100.0));
        assert!(mgr.should_throttle("gpu0"));
    }

    #[test]
    fn should_throttle_false_for_unknown_device() {
        let mgr = default_manager();
        assert!(!mgr.should_throttle("ghost"));
    }

    // ── power mode ──────────────────────────────────────────────────────

    #[test]
    fn apply_performance_mode() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 60.0, 100.0));
        assert!(mgr.apply_power_mode("gpu0", PowerMode::Performance));
        let s = mgr.current_state("gpu0").unwrap();
        assert_eq!(s.gpu_clock_mhz, 2100);
        assert_eq!(s.mem_clock_mhz, 1200);
        assert_eq!(s.power_mode, PowerMode::Performance);
    }

    #[test]
    fn apply_balanced_mode() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 60.0, 100.0));
        assert!(mgr.apply_power_mode("gpu0", PowerMode::Balanced));
        let s = mgr.current_state("gpu0").unwrap();
        assert_eq!(s.gpu_clock_mhz, 1500);
        assert_eq!(s.mem_clock_mhz, 900);
    }

    #[test]
    fn apply_efficient_mode() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 60.0, 100.0));
        assert!(mgr.apply_power_mode("gpu0", PowerMode::Efficient));
        let s = mgr.current_state("gpu0").unwrap();
        assert_eq!(s.gpu_clock_mhz, 800);
        assert_eq!(s.mem_clock_mhz, 600);
    }

    #[test]
    fn apply_custom_mode() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 60.0, 100.0));
        let mode = PowerMode::Custom { gpu_clock_mhz: 1800, mem_clock_mhz: 1000 };
        assert!(mgr.apply_power_mode("gpu0", mode));
        let s = mgr.current_state("gpu0").unwrap();
        assert_eq!(s.gpu_clock_mhz, 1800);
        assert_eq!(s.mem_clock_mhz, 1000);
    }

    #[test]
    fn apply_power_mode_to_missing_device_returns_false() {
        let mut mgr = default_manager();
        assert!(!mgr.apply_power_mode("nope", PowerMode::Performance));
    }

    #[test]
    fn power_mode_switch_performance_to_efficient() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 60.0, 100.0));
        mgr.apply_power_mode("gpu0", PowerMode::Performance);
        mgr.apply_power_mode("gpu0", PowerMode::Efficient);
        let s = mgr.current_state("gpu0").unwrap();
        assert_eq!(s.power_mode, PowerMode::Efficient);
        assert_eq!(s.gpu_clock_mhz, 800);
    }

    // ── device tracking ─────────────────────────────────────────────────

    #[test]
    fn update_adds_new_device() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 60.0, 100.0));
        assert!(mgr.current_state("gpu0").is_some());
    }

    #[test]
    fn update_replaces_existing_device() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 60.0, 100.0));
        mgr.update("gpu0", device_state("gpu0", 70.0, 120.0));
        let s = mgr.current_state("gpu0").unwrap();
        assert!((s.temperature_celsius - 70.0).abs() < f64::EPSILON);
    }

    #[test]
    fn multiple_devices_tracked_independently() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 60.0, 100.0));
        mgr.update("gpu1", device_state("gpu1", 80.0, 200.0));
        let s0 = mgr.current_state("gpu0").unwrap();
        let s1 = mgr.current_state("gpu1").unwrap();
        assert!((s0.temperature_celsius - 60.0).abs() < f64::EPSILON);
        assert!((s1.temperature_celsius - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn current_state_returns_none_for_unknown() {
        let mgr = default_manager();
        assert!(mgr.current_state("unknown").is_none());
    }

    #[test]
    fn devices_returns_all_tracked() {
        let mut mgr = default_manager();
        mgr.update("a", device_state("a", 50.0, 80.0));
        mgr.update("b", device_state("b", 55.0, 90.0));
        assert_eq!(mgr.devices().len(), 2);
    }

    // ── fan speed ───────────────────────────────────────────────────────

    #[test]
    fn fan_speed_stored_correctly() {
        let mut mgr = default_manager();
        let mut state = device_state("gpu0", 60.0, 100.0);
        state.fan_speed_percent = 75;
        mgr.update("gpu0", state);
        assert_eq!(mgr.current_state("gpu0").unwrap().fan_speed_percent, 75);
    }

    #[test]
    fn fan_speed_zero_allowed() {
        let mut mgr = default_manager();
        let mut state = device_state("gpu0", 30.0, 50.0);
        state.fan_speed_percent = 0;
        mgr.update("gpu0", state);
        assert_eq!(mgr.current_state("gpu0").unwrap().fan_speed_percent, 0);
    }

    #[test]
    fn fan_speed_max_allowed() {
        let mut mgr = default_manager();
        let mut state = device_state("gpu0", 90.0, 250.0);
        state.fan_speed_percent = 100;
        mgr.update("gpu0", state);
        assert_eq!(mgr.current_state("gpu0").unwrap().fan_speed_percent, 100);
    }

    // ── history & samples ───────────────────────────────────────────────

    #[test]
    fn record_sample_adds_to_history() {
        let mut mgr = default_manager();
        mgr.record_sample(sample(0, 60.0, 100.0, 1500));
        assert_eq!(mgr.history().len(), 1);
    }

    #[test]
    fn multiple_samples_preserve_order() {
        let mut mgr = default_manager();
        mgr.record_sample(sample(0, 60.0, 100.0, 1500));
        mgr.record_sample(sample(1000, 65.0, 110.0, 1500));
        mgr.record_sample(sample(2000, 70.0, 120.0, 1500));
        assert_eq!(mgr.history().len(), 3);
        assert_eq!(mgr.history()[0].timestamp_ms, 0);
        assert_eq!(mgr.history()[2].timestamp_ms, 2000);
    }

    #[test]
    fn sample_timestamps_recorded() {
        let mut mgr = default_manager();
        mgr.record_sample(sample(42_000, 55.0, 90.0, 1400));
        assert_eq!(mgr.history()[0].timestamp_ms, 42_000);
    }

    #[test]
    fn reset_history_clears_samples() {
        let mut mgr = default_manager();
        mgr.record_sample(sample(0, 60.0, 100.0, 1500));
        mgr.record_sample(sample(1000, 65.0, 110.0, 1500));
        mgr.reset_history();
        assert!(mgr.history().is_empty());
    }

    #[test]
    fn reset_history_clears_throttle_events() {
        let mut mgr = default_manager();
        let mut s = device_state("gpu0", 90.0, 200.0);
        s.throttling = true;
        mgr.update("gpu0", s);
        assert_eq!(mgr.throttle_events(), 1);
        mgr.reset_history();
        assert_eq!(mgr.throttle_events(), 0);
    }

    // ── throttle event counting ─────────────────────────────────────────

    #[test]
    fn throttle_event_counted_on_new_device() {
        let mut mgr = default_manager();
        let mut s = device_state("gpu0", 90.0, 200.0);
        s.throttling = true;
        mgr.update("gpu0", s);
        assert_eq!(mgr.throttle_events(), 1);
    }

    #[test]
    fn throttle_event_counted_on_transition() {
        let mut mgr = default_manager();
        mgr.update("gpu0", device_state("gpu0", 60.0, 100.0));
        let mut hot = device_state("gpu0", 95.0, 250.0);
        hot.throttling = true;
        mgr.update("gpu0", hot);
        assert_eq!(mgr.throttle_events(), 1);
    }

    #[test]
    fn no_double_count_when_already_throttling() {
        let mut mgr = default_manager();
        let mut s1 = device_state("gpu0", 90.0, 200.0);
        s1.throttling = true;
        mgr.update("gpu0", s1);
        let mut s2 = device_state("gpu0", 91.0, 210.0);
        s2.throttling = true;
        mgr.update("gpu0", s2);
        assert_eq!(mgr.throttle_events(), 1);
    }

    #[test]
    fn multiple_throttle_transitions_counted() {
        let mut mgr = default_manager();
        // first throttle
        let mut s = device_state("gpu0", 90.0, 200.0);
        s.throttling = true;
        mgr.update("gpu0", s);
        // cool down
        mgr.update("gpu0", device_state("gpu0", 60.0, 100.0));
        // second throttle
        let mut s2 = device_state("gpu0", 92.0, 220.0);
        s2.throttling = true;
        mgr.update("gpu0", s2);
        assert_eq!(mgr.throttle_events(), 2);
    }

    // ── report generation ───────────────────────────────────────────────

    #[test]
    fn empty_history_report_all_zeros() {
        let mgr = default_manager();
        let r = mgr.generate_report();
        assert!((r.avg_power_watts).abs() < f64::EPSILON);
        assert!((r.peak_power_watts).abs() < f64::EPSILON);
        assert!((r.avg_temperature).abs() < f64::EPSILON);
        assert!((r.peak_temperature).abs() < f64::EPSILON);
        assert!((r.energy_joules).abs() < f64::EPSILON);
        assert!((r.duration_seconds).abs() < f64::EPSILON);
    }

    #[test]
    fn report_avg_power() {
        let mut mgr = default_manager();
        mgr.record_sample(sample(0, 60.0, 100.0, 1500));
        mgr.record_sample(sample(1000, 60.0, 200.0, 1500));
        let r = mgr.generate_report();
        assert!((r.avg_power_watts - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn report_peak_power() {
        let mut mgr = default_manager();
        mgr.record_sample(sample(0, 60.0, 100.0, 1500));
        mgr.record_sample(sample(1000, 60.0, 300.0, 1500));
        mgr.record_sample(sample(2000, 60.0, 200.0, 1500));
        let r = mgr.generate_report();
        assert!((r.peak_power_watts - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn report_avg_temperature() {
        let mut mgr = default_manager();
        mgr.record_sample(sample(0, 50.0, 100.0, 1500));
        mgr.record_sample(sample(1000, 70.0, 100.0, 1500));
        let r = mgr.generate_report();
        assert!((r.avg_temperature - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn report_peak_temperature() {
        let mut mgr = default_manager();
        mgr.record_sample(sample(0, 50.0, 100.0, 1500));
        mgr.record_sample(sample(1000, 80.0, 100.0, 1500));
        mgr.record_sample(sample(2000, 65.0, 100.0, 1500));
        let r = mgr.generate_report();
        assert!((r.peak_temperature - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn report_duration_seconds() {
        let mut mgr = default_manager();
        mgr.record_sample(sample(0, 60.0, 100.0, 1500));
        mgr.record_sample(sample(5000, 60.0, 100.0, 1500));
        let r = mgr.generate_report();
        assert!((r.duration_seconds - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn report_energy_joules() {
        let mut mgr = default_manager();
        // 100W for 10 seconds → 1000 J
        mgr.record_sample(sample(0, 60.0, 100.0, 1500));
        mgr.record_sample(sample(10_000, 60.0, 100.0, 1500));
        let r = mgr.generate_report();
        assert!((r.energy_joules - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn report_throttle_events_in_report() {
        let mut mgr = default_manager();
        let mut s = device_state("gpu0", 90.0, 200.0);
        s.throttling = true;
        mgr.update("gpu0", s);
        mgr.record_sample(sample(0, 90.0, 200.0, 1500));
        let r = mgr.generate_report();
        assert_eq!(r.throttle_events, 1);
    }

    #[test]
    fn report_single_sample_zero_duration() {
        let mut mgr = default_manager();
        mgr.record_sample(sample(1000, 60.0, 100.0, 1500));
        let r = mgr.generate_report();
        assert!((r.duration_seconds).abs() < f64::EPSILON);
        assert!((r.energy_joules).abs() < f64::EPSILON);
    }

    // ── energy efficiency ───────────────────────────────────────────────

    #[test]
    fn energy_efficiency_basic() {
        let mut mgr = default_manager();
        // 100W avg, 10s → 1000J, 500 tokens → 0.5 tok/J
        mgr.record_sample(sample(0, 60.0, 100.0, 1500));
        mgr.record_sample(sample(10_000, 60.0, 100.0, 1500));
        let eff = mgr.energy_efficiency(500, 10.0);
        assert!((eff - 0.5).abs() < 1e-9);
    }

    #[test]
    fn energy_efficiency_zero_duration() {
        let mut mgr = default_manager();
        mgr.record_sample(sample(0, 60.0, 100.0, 1500));
        assert!((mgr.energy_efficiency(100, 0.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn energy_efficiency_no_history() {
        let mgr = default_manager();
        assert!((mgr.energy_efficiency(100, 10.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn energy_efficiency_zero_tokens() {
        let mut mgr = default_manager();
        mgr.record_sample(sample(0, 60.0, 100.0, 1500));
        mgr.record_sample(sample(10_000, 60.0, 100.0, 1500));
        assert!((mgr.energy_efficiency(0, 10.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn energy_efficiency_high_throughput() {
        let mut mgr = default_manager();
        // 200W avg, 5s → 1000J, 10000 tokens → 10 tok/J
        mgr.record_sample(sample(0, 60.0, 200.0, 1500));
        mgr.record_sample(sample(5000, 60.0, 200.0, 1500));
        let eff = mgr.energy_efficiency(10_000, 5.0);
        assert!((eff - 10.0).abs() < 1e-9);
    }

    // ── config & defaults ───────────────────────────────────────────────

    #[test]
    fn default_config_values() {
        let cfg = PowerConfig::default();
        assert_eq!(cfg.polling_interval_ms, 1000);
        assert!((cfg.thermal_throttle_celsius - 85.0).abs() < f64::EPSILON);
        assert!(cfg.power_limit_watts.is_none());
        assert_eq!(cfg.power_mode, PowerMode::Balanced);
    }

    #[test]
    fn default_thermal_policy_values() {
        let tp = ThermalPolicy::default();
        assert!((tp.warning_celsius - 75.0).abs() < f64::EPSILON);
        assert!((tp.throttle_celsius - 85.0).abs() < f64::EPSILON);
        assert!((tp.shutdown_celsius - 95.0).abs() < f64::EPSILON);
    }

    #[test]
    fn config_power_limit_watts() {
        let cfg = PowerConfig { power_limit_watts: Some(300.0), ..PowerConfig::default() };
        assert_eq!(cfg.power_limit_watts, Some(300.0));
    }

    #[test]
    fn manager_config_accessor() {
        let cfg = PowerConfig { polling_interval_ms: 500, ..PowerConfig::default() };
        let mgr = PowerManager::new(cfg);
        assert_eq!(mgr.config().polling_interval_ms, 500);
    }

    #[test]
    fn manager_thermal_policy_accessor() {
        let policy =
            ThermalPolicy { warning_celsius: 60.0, throttle_celsius: 70.0, shutdown_celsius: 80.0 };
        let mgr = PowerManager::with_thermal_policy(PowerConfig::default(), policy);
        assert!((mgr.thermal_policy().throttle_celsius - 70.0).abs() < f64::EPSILON);
    }
}
