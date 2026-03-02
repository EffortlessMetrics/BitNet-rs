//! OpenCL GPU power management for Intel Arc A770.
//!
//! Monitors and controls power states, thermal throttling, frequency scaling,
//! and energy-efficient inference scheduling. All logic uses CPU-side reference
//! implementations — actual GPU sysfs / driver queries are gated behind the
//! `oneapi` feature.

use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// PowerState — GPU power state machine
// ---------------------------------------------------------------------------

/// Discrete GPU power states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PowerState {
    /// GPU is idle — clocks may be gated.
    Idle,
    /// Low-power mode — reduced frequency for light workloads.
    LowPower,
    /// Normal operating state.
    Normal,
    /// Boost — elevated clocks within TDP headroom.
    Boost,
    /// Throttled — clocks reduced due to thermal or power limit.
    Throttled,
}

impl PowerState {
    /// Returns `true` when the GPU is actively executing work.
    pub fn is_active(self) -> bool {
        matches!(self, Self::Normal | Self::Boost | Self::LowPower)
    }

    /// Valid successor states from the current state.
    pub fn valid_transitions(self) -> &'static [PowerState] {
        match self {
            Self::Idle => &[Self::LowPower, Self::Normal, Self::Boost],
            Self::LowPower => &[Self::Idle, Self::Normal, Self::Throttled],
            Self::Normal => &[Self::Idle, Self::LowPower, Self::Boost, Self::Throttled],
            Self::Boost => &[Self::Normal, Self::Throttled],
            Self::Throttled => &[Self::LowPower, Self::Normal, Self::Idle],
        }
    }

    /// Whether a transition to `target` is valid.
    pub fn can_transition_to(self, target: Self) -> bool {
        self.valid_transitions().contains(&target)
    }
}

impl fmt::Display for PowerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::LowPower => write!(f, "LowPower"),
            Self::Normal => write!(f, "Normal"),
            Self::Boost => write!(f, "Boost"),
            Self::Throttled => write!(f, "Throttled"),
        }
    }
}

// ---------------------------------------------------------------------------
// ThermalZone
// ---------------------------------------------------------------------------

/// Temperature reading with alert thresholds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalZone {
    /// Current temperature in °C.
    pub temperature_c: f64,
    /// Threshold at which throttling begins (°C).
    pub threshold_c: f64,
    /// Critical shutdown temperature (°C).
    pub critical_c: f64,
}

impl ThermalZone {
    /// Create a new thermal zone with the given thresholds.
    pub fn new(temperature_c: f64, threshold_c: f64, critical_c: f64) -> Self {
        Self { temperature_c, threshold_c, critical_c }
    }

    /// Whether the temperature has reached the throttle threshold.
    pub fn is_throttling(&self) -> bool {
        self.temperature_c >= self.threshold_c
    }

    /// Whether the temperature has reached the critical limit.
    pub fn is_critical(&self) -> bool {
        self.temperature_c >= self.critical_c
    }

    /// Headroom before throttling begins (°C). Returns 0.0 if already throttling.
    pub fn headroom_c(&self) -> f64 {
        (self.threshold_c - self.temperature_c).max(0.0)
    }

    /// Normalised thermal load in `[0.0, 1.0]` relative to threshold.
    pub fn thermal_load(&self) -> f64 {
        if self.threshold_c <= 0.0 {
            return 1.0;
        }
        (self.temperature_c / self.threshold_c).clamp(0.0, 1.0)
    }
}

impl fmt::Display for ThermalZone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.1}°C (throttle: {:.1}°C, critical: {:.1}°C)",
            self.temperature_c, self.threshold_c, self.critical_c,
        )
    }
}

// ---------------------------------------------------------------------------
// FrequencyInfo
// ---------------------------------------------------------------------------

/// GPU clock frequency information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyInfo {
    /// Current operating frequency (MHz).
    pub current_mhz: u32,
    /// Minimum supported frequency (MHz).
    pub min_mhz: u32,
    /// Maximum sustained frequency (MHz).
    pub max_mhz: u32,
    /// Boost frequency (MHz) — may exceed `max_mhz` transiently.
    pub boost_mhz: u32,
}

impl FrequencyInfo {
    pub fn new(current_mhz: u32, min_mhz: u32, max_mhz: u32, boost_mhz: u32) -> Self {
        Self { current_mhz, min_mhz, max_mhz, boost_mhz }
    }

    /// Fraction of max sustained frequency currently in use `[0.0, ∞)`.
    pub fn utilisation(&self) -> f64 {
        if self.max_mhz == 0 {
            return 0.0;
        }
        self.current_mhz as f64 / self.max_mhz as f64
    }

    /// Whether the GPU is running above its sustained max (i.e. in boost).
    pub fn is_boosting(&self) -> bool {
        self.current_mhz > self.max_mhz
    }

    /// Whether the current frequency is at or below the minimum.
    pub fn is_at_minimum(&self) -> bool {
        self.current_mhz <= self.min_mhz
    }

    /// Clamp `current_mhz` to `[min_mhz, cap]`.
    pub fn clamped(&self, cap_mhz: u32) -> Self {
        Self { current_mhz: self.current_mhz.clamp(self.min_mhz, cap_mhz), ..*self }
    }
}

impl fmt::Display for FrequencyInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}MHz (min: {}, max: {}, boost: {})",
            self.current_mhz, self.min_mhz, self.max_mhz, self.boost_mhz,
        )
    }
}

// ---------------------------------------------------------------------------
// PowerProfile / PowerConfig
// ---------------------------------------------------------------------------

/// Custom power configuration knobs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerConfig {
    /// Maximum board power draw (W).
    pub max_power_w: f64,
    /// Target GPU temperature (°C).
    pub target_temp_c: f64,
    /// Frequency cap (MHz). 0 = uncapped.
    pub freq_cap_mhz: u32,
}

impl PowerConfig {
    pub fn new(max_power_w: f64, target_temp_c: f64, freq_cap_mhz: u32) -> Self {
        Self { max_power_w, target_temp_c, freq_cap_mhz }
    }
}

/// Pre-defined power profiles.
#[derive(Debug, Clone, PartialEq)]
pub enum PowerProfile {
    /// Maximum clocks, no power cap.
    Performance,
    /// Balanced thermal/performance trade-off.
    Balanced,
    /// Aggressive power saving — reduced clocks and power cap.
    PowerSaver,
    /// User-supplied configuration.
    Custom(PowerConfig),
}

impl PowerProfile {
    /// Materialise the profile into a concrete [`PowerConfig`].
    ///
    /// Default values model the Intel Arc A770 (225 W TDP, 100 °C Tj).
    pub fn to_config(&self) -> PowerConfig {
        match self {
            Self::Performance => PowerConfig::new(225.0, 100.0, 0),
            Self::Balanced => PowerConfig::new(190.0, 88.0, 2100),
            Self::PowerSaver => PowerConfig::new(120.0, 75.0, 1800),
            Self::Custom(cfg) => *cfg,
        }
    }

    /// Suggested [`PowerState`] given a thermal zone reading.
    pub fn recommend_state(&self, thermal: &ThermalZone) -> PowerState {
        let cfg = self.to_config();
        if thermal.is_critical() {
            return PowerState::Throttled;
        }
        if thermal.temperature_c >= cfg.target_temp_c {
            return PowerState::Throttled;
        }
        match self {
            Self::Performance => {
                if thermal.headroom_c() > 15.0 {
                    PowerState::Boost
                } else {
                    PowerState::Normal
                }
            }
            Self::Balanced => PowerState::Normal,
            Self::PowerSaver => PowerState::LowPower,
            Self::Custom(_) => PowerState::Normal,
        }
    }
}

impl fmt::Display for PowerProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Performance => write!(f, "Performance"),
            Self::Balanced => write!(f, "Balanced"),
            Self::PowerSaver => write!(f, "PowerSaver"),
            Self::Custom(cfg) => {
                write!(
                    f,
                    "Custom({}W, {}°C, {}MHz)",
                    cfg.max_power_w, cfg.target_temp_c, cfg.freq_cap_mhz
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ThrottleEvent / ThrottleDetector
// ---------------------------------------------------------------------------

/// The reason a throttle event was detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThrottleReason {
    Thermal,
    PowerBudget,
}

/// A recorded throttle event.
#[derive(Debug, Clone)]
pub struct ThrottleEvent {
    pub reason: ThrottleReason,
    pub timestamp: Instant,
    pub temperature_c: f64,
    pub power_w: f64,
}

/// Detects thermal and power-budget throttling events.
#[derive(Debug)]
pub struct ThrottleDetector {
    thermal_threshold_c: f64,
    power_limit_w: f64,
    events: Vec<ThrottleEvent>,
}

impl ThrottleDetector {
    pub fn new(thermal_threshold_c: f64, power_limit_w: f64) -> Self {
        Self { thermal_threshold_c, power_limit_w, events: Vec::new() }
    }

    /// Check a sample and record a throttle event if limits are exceeded.
    /// Returns `Some(reason)` when throttling is detected.
    pub fn check(&mut self, temperature_c: f64, power_w: f64) -> Option<ThrottleReason> {
        let reason = if temperature_c >= self.thermal_threshold_c {
            Some(ThrottleReason::Thermal)
        } else if power_w >= self.power_limit_w {
            Some(ThrottleReason::PowerBudget)
        } else {
            None
        };

        if let Some(r) = reason {
            self.events.push(ThrottleEvent {
                reason: r,
                timestamp: Instant::now(),
                temperature_c,
                power_w,
            });
        }
        reason
    }

    /// Total number of throttle events recorded.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Events filtered by reason.
    pub fn events_by_reason(&self, reason: ThrottleReason) -> Vec<&ThrottleEvent> {
        self.events.iter().filter(|e| e.reason == reason).collect()
    }

    /// Whether *any* throttle event has ever been recorded.
    pub fn has_throttled(&self) -> bool {
        !self.events.is_empty()
    }

    /// Clear the event history.
    pub fn reset(&mut self) {
        self.events.clear();
    }
}

// ---------------------------------------------------------------------------
// EnergyEstimator
// ---------------------------------------------------------------------------

/// Estimates energy consumption per token during inference.
#[derive(Debug)]
pub struct EnergyEstimator {
    samples: Vec<EnergySample>,
}

/// A single power-draw sample associated with token generation.
#[derive(Debug, Clone, Copy)]
pub struct EnergySample {
    /// Instantaneous power draw (W).
    pub power_w: f64,
    /// Duration of the sample window.
    pub duration: Duration,
    /// Number of tokens generated during this window.
    pub tokens: u32,
}

impl EnergySample {
    pub fn new(power_w: f64, duration: Duration, tokens: u32) -> Self {
        Self { power_w, duration, tokens }
    }

    /// Energy consumed during this sample (joules).
    pub fn energy_j(&self) -> f64 {
        self.power_w * self.duration.as_secs_f64()
    }

    /// Energy per token (joules). Returns 0.0 if no tokens were generated.
    pub fn energy_per_token_j(&self) -> f64 {
        if self.tokens == 0 {
            return 0.0;
        }
        self.energy_j() / self.tokens as f64
    }
}

impl EnergyEstimator {
    pub fn new() -> Self {
        Self { samples: Vec::new() }
    }

    /// Record a power/token sample.
    pub fn record(&mut self, sample: EnergySample) {
        self.samples.push(sample);
    }

    /// Total energy consumed across all samples (joules).
    pub fn total_energy_j(&self) -> f64 {
        self.samples.iter().map(|s| s.energy_j()).sum()
    }

    /// Total tokens generated across all samples.
    pub fn total_tokens(&self) -> u64 {
        self.samples.iter().map(|s| s.tokens as u64).sum()
    }

    /// Average energy per token (joules). Returns 0.0 if no tokens recorded.
    pub fn avg_energy_per_token_j(&self) -> f64 {
        let total_tokens = self.total_tokens();
        if total_tokens == 0 {
            return 0.0;
        }
        self.total_energy_j() / total_tokens as f64
    }

    /// Average power draw across all samples (W). Returns 0.0 if empty.
    pub fn avg_power_w(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let total_w: f64 = self.samples.iter().map(|s| s.power_w).sum();
        total_w / self.samples.len() as f64
    }

    /// Number of recorded samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Clear all samples.
    pub fn reset(&mut self) {
        self.samples.clear();
    }
}

impl Default for EnergyEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PowerSnapshot / PowerMonitor
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of GPU power telemetry.
#[derive(Debug, Clone)]
pub struct PowerSnapshot {
    pub state: PowerState,
    pub thermal: ThermalZone,
    pub frequency: FrequencyInfo,
    pub power_w: f64,
    pub timestamp: Instant,
}

/// Tracks GPU power telemetry over time.
#[derive(Debug)]
pub struct PowerMonitor {
    history: Vec<PowerSnapshot>,
    current_state: PowerState,
    profile: PowerProfile,
}

impl PowerMonitor {
    pub fn new(profile: PowerProfile) -> Self {
        Self { history: Vec::new(), current_state: PowerState::Idle, profile }
    }

    /// Record a telemetry snapshot and update state based on profile.
    pub fn record(&mut self, thermal: ThermalZone, frequency: FrequencyInfo, power_w: f64) {
        let recommended = self.profile.recommend_state(&thermal);
        // Only transition if valid, otherwise stay in current state.
        if self.current_state.can_transition_to(recommended) {
            self.current_state = recommended;
        }

        self.history.push(PowerSnapshot {
            state: self.current_state,
            thermal,
            frequency,
            power_w,
            timestamp: Instant::now(),
        });
    }

    /// Current power state.
    pub fn current_state(&self) -> PowerState {
        self.current_state
    }

    /// Number of snapshots in history.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Peek at the latest snapshot, if any.
    pub fn latest(&self) -> Option<&PowerSnapshot> {
        self.history.last()
    }

    /// Average power draw across all snapshots (W).
    pub fn avg_power_w(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let total: f64 = self.history.iter().map(|s| s.power_w).sum();
        total / self.history.len() as f64
    }

    /// Peak power draw recorded (W).
    pub fn peak_power_w(&self) -> f64 {
        self.history.iter().map(|s| s.power_w).fold(0.0_f64, f64::max)
    }

    /// Peak temperature recorded (°C).
    pub fn peak_temperature_c(&self) -> f64 {
        self.history.iter().map(|s| s.thermal.temperature_c).fold(0.0_f64, f64::max)
    }

    /// Snapshots where the GPU was in the [`Throttled`](PowerState::Throttled) state.
    pub fn throttled_snapshots(&self) -> Vec<&PowerSnapshot> {
        self.history.iter().filter(|s| s.state == PowerState::Throttled).collect()
    }

    /// Replace the active power profile.
    pub fn set_profile(&mut self, profile: PowerProfile) {
        self.profile = profile;
    }

    /// Force a state transition (e.g. for testing or manual override).
    pub fn force_state(&mut self, state: PowerState) {
        self.current_state = state;
    }

    /// Clear all history and reset to Idle.
    pub fn reset(&mut self) {
        self.history.clear();
        self.current_state = PowerState::Idle;
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    fn default_thermal() -> ThermalZone {
        ThermalZone::new(65.0, 90.0, 105.0)
    }

    fn hot_thermal() -> ThermalZone {
        ThermalZone::new(92.0, 90.0, 105.0)
    }

    fn critical_thermal() -> ThermalZone {
        ThermalZone::new(106.0, 90.0, 105.0)
    }

    fn default_freq() -> FrequencyInfo {
        FrequencyInfo::new(2100, 300, 2100, 2400)
    }

    // ===================================================================
    // PowerState
    // ===================================================================

    #[test]
    fn test_power_state_display() {
        assert_eq!(PowerState::Idle.to_string(), "Idle");
        assert_eq!(PowerState::Throttled.to_string(), "Throttled");
    }

    #[test]
    fn test_power_state_is_active() {
        assert!(!PowerState::Idle.is_active());
        assert!(PowerState::LowPower.is_active());
        assert!(PowerState::Normal.is_active());
        assert!(PowerState::Boost.is_active());
        assert!(!PowerState::Throttled.is_active());
    }

    #[test]
    fn test_idle_valid_transitions() {
        let valid = PowerState::Idle.valid_transitions();
        assert!(valid.contains(&PowerState::Normal));
        assert!(valid.contains(&PowerState::LowPower));
        assert!(valid.contains(&PowerState::Boost));
        assert!(!valid.contains(&PowerState::Throttled));
    }

    #[test]
    fn test_boost_can_transition_to_throttled() {
        assert!(PowerState::Boost.can_transition_to(PowerState::Throttled));
        assert!(PowerState::Boost.can_transition_to(PowerState::Normal));
        assert!(!PowerState::Boost.can_transition_to(PowerState::Idle));
    }

    #[test]
    fn test_throttled_cannot_transition_to_boost() {
        assert!(!PowerState::Throttled.can_transition_to(PowerState::Boost));
    }

    #[test]
    fn test_self_transition_is_invalid() {
        // No state lists itself as a valid successor.
        for state in [
            PowerState::Idle,
            PowerState::LowPower,
            PowerState::Normal,
            PowerState::Boost,
            PowerState::Throttled,
        ] {
            assert!(!state.can_transition_to(state), "{state} should not self-transition");
        }
    }

    #[test]
    fn test_normal_has_most_transitions() {
        assert_eq!(PowerState::Normal.valid_transitions().len(), 4);
    }

    // ===================================================================
    // ThermalZone
    // ===================================================================

    #[test]
    fn test_thermal_not_throttling() {
        let tz = default_thermal();
        assert!(!tz.is_throttling());
        assert!(!tz.is_critical());
    }

    #[test]
    fn test_thermal_throttling() {
        let tz = hot_thermal();
        assert!(tz.is_throttling());
        assert!(!tz.is_critical());
    }

    #[test]
    fn test_thermal_critical() {
        let tz = critical_thermal();
        assert!(tz.is_throttling());
        assert!(tz.is_critical());
    }

    #[test]
    fn test_thermal_headroom() {
        let tz = default_thermal();
        assert!((tz.headroom_c() - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thermal_headroom_when_throttling() {
        let tz = hot_thermal();
        assert_eq!(tz.headroom_c(), 0.0);
    }

    #[test]
    fn test_thermal_load_normal() {
        let tz = ThermalZone::new(45.0, 90.0, 105.0);
        assert!((tz.thermal_load() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thermal_load_at_threshold() {
        let tz = ThermalZone::new(90.0, 90.0, 105.0);
        assert!((tz.thermal_load() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thermal_load_zero_threshold() {
        let tz = ThermalZone::new(50.0, 0.0, 105.0);
        assert_eq!(tz.thermal_load(), 1.0);
    }

    #[test]
    fn test_thermal_display() {
        let tz = default_thermal();
        let s = tz.to_string();
        assert!(s.contains("65.0°C"));
        assert!(s.contains("throttle"));
    }

    #[test]
    fn test_thermal_exact_threshold() {
        let tz = ThermalZone::new(90.0, 90.0, 105.0);
        assert!(tz.is_throttling());
        assert!(!tz.is_critical());
    }

    // ===================================================================
    // FrequencyInfo
    // ===================================================================

    #[test]
    fn test_frequency_utilisation_full() {
        let fi = FrequencyInfo::new(2100, 300, 2100, 2400);
        assert!((fi.utilisation() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frequency_utilisation_half() {
        let fi = FrequencyInfo::new(1050, 300, 2100, 2400);
        assert!((fi.utilisation() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frequency_utilisation_zero_max() {
        let fi = FrequencyInfo::new(0, 0, 0, 0);
        assert_eq!(fi.utilisation(), 0.0);
    }

    #[test]
    fn test_frequency_is_boosting() {
        let fi = FrequencyInfo::new(2300, 300, 2100, 2400);
        assert!(fi.is_boosting());
    }

    #[test]
    fn test_frequency_not_boosting() {
        let fi = default_freq();
        assert!(!fi.is_boosting());
    }

    #[test]
    fn test_frequency_at_minimum() {
        let fi = FrequencyInfo::new(300, 300, 2100, 2400);
        assert!(fi.is_at_minimum());
    }

    #[test]
    fn test_frequency_clamped() {
        let fi = FrequencyInfo::new(2400, 300, 2100, 2400);
        let clamped = fi.clamped(2000);
        assert_eq!(clamped.current_mhz, 2000);
    }

    #[test]
    fn test_frequency_clamped_below_min() {
        let fi = FrequencyInfo::new(100, 300, 2100, 2400);
        let clamped = fi.clamped(2000);
        assert_eq!(clamped.current_mhz, 300);
    }

    #[test]
    fn test_frequency_display() {
        let fi = default_freq();
        let s = fi.to_string();
        assert!(s.contains("2100MHz"));
        assert!(s.contains("boost: 2400"));
    }

    // ===================================================================
    // PowerProfile / PowerConfig
    // ===================================================================

    #[test]
    fn test_performance_profile_config() {
        let cfg = PowerProfile::Performance.to_config();
        assert_eq!(cfg.max_power_w, 225.0);
        assert_eq!(cfg.freq_cap_mhz, 0);
    }

    #[test]
    fn test_balanced_profile_config() {
        let cfg = PowerProfile::Balanced.to_config();
        assert!(cfg.max_power_w < 225.0);
        assert!(cfg.freq_cap_mhz > 0);
    }

    #[test]
    fn test_power_saver_profile_config() {
        let cfg = PowerProfile::PowerSaver.to_config();
        assert!(cfg.max_power_w < 190.0);
        assert!(cfg.target_temp_c < 88.0);
    }

    #[test]
    fn test_custom_profile_roundtrip() {
        let custom = PowerConfig::new(150.0, 80.0, 1900);
        let profile = PowerProfile::Custom(custom);
        assert_eq!(profile.to_config(), custom);
    }

    #[test]
    fn test_profile_recommend_throttled_on_critical() {
        assert_eq!(
            PowerProfile::Performance.recommend_state(&critical_thermal()),
            PowerState::Throttled,
        );
    }

    #[test]
    fn test_performance_recommends_boost_when_cool() {
        let cool = ThermalZone::new(50.0, 90.0, 105.0);
        assert_eq!(PowerProfile::Performance.recommend_state(&cool), PowerState::Boost);
    }

    #[test]
    fn test_power_saver_recommends_low_power() {
        let cool = ThermalZone::new(50.0, 90.0, 105.0);
        assert_eq!(PowerProfile::PowerSaver.recommend_state(&cool), PowerState::LowPower);
    }

    #[test]
    fn test_profile_display() {
        assert_eq!(PowerProfile::Balanced.to_string(), "Balanced");
        let custom = PowerProfile::Custom(PowerConfig::new(100.0, 70.0, 1500));
        assert!(custom.to_string().contains("Custom"));
    }

    // ===================================================================
    // ThrottleDetector
    // ===================================================================

    #[test]
    fn test_detector_no_throttle() {
        let mut det = ThrottleDetector::new(90.0, 225.0);
        assert_eq!(det.check(80.0, 180.0), None);
        assert!(!det.has_throttled());
        assert_eq!(det.event_count(), 0);
    }

    #[test]
    fn test_detector_thermal_throttle() {
        let mut det = ThrottleDetector::new(90.0, 225.0);
        let r = det.check(95.0, 180.0);
        assert_eq!(r, Some(ThrottleReason::Thermal));
        assert_eq!(det.event_count(), 1);
    }

    #[test]
    fn test_detector_power_throttle() {
        let mut det = ThrottleDetector::new(90.0, 225.0);
        let r = det.check(80.0, 230.0);
        assert_eq!(r, Some(ThrottleReason::PowerBudget));
    }

    #[test]
    fn test_detector_thermal_takes_priority() {
        let mut det = ThrottleDetector::new(90.0, 225.0);
        let r = det.check(95.0, 230.0);
        assert_eq!(r, Some(ThrottleReason::Thermal));
    }

    #[test]
    fn test_detector_events_by_reason() {
        let mut det = ThrottleDetector::new(90.0, 225.0);
        det.check(95.0, 180.0);
        det.check(80.0, 230.0);
        det.check(92.0, 200.0);
        assert_eq!(det.events_by_reason(ThrottleReason::Thermal).len(), 2);
        assert_eq!(det.events_by_reason(ThrottleReason::PowerBudget).len(), 1);
    }

    #[test]
    fn test_detector_reset() {
        let mut det = ThrottleDetector::new(90.0, 225.0);
        det.check(95.0, 180.0);
        assert!(det.has_throttled());
        det.reset();
        assert!(!det.has_throttled());
        assert_eq!(det.event_count(), 0);
    }

    #[test]
    fn test_detector_exact_threshold() {
        let mut det = ThrottleDetector::new(90.0, 225.0);
        assert_eq!(det.check(90.0, 200.0), Some(ThrottleReason::Thermal));
    }

    // ===================================================================
    // EnergyEstimator
    // ===================================================================

    #[test]
    fn test_energy_sample_basic() {
        let s = EnergySample::new(150.0, Duration::from_secs(2), 10);
        assert!((s.energy_j() - 300.0).abs() < f64::EPSILON);
        assert!((s.energy_per_token_j() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_energy_sample_zero_tokens() {
        let s = EnergySample::new(150.0, Duration::from_secs(1), 0);
        assert_eq!(s.energy_per_token_j(), 0.0);
    }

    #[test]
    fn test_estimator_empty() {
        let est = EnergyEstimator::new();
        assert_eq!(est.total_energy_j(), 0.0);
        assert_eq!(est.total_tokens(), 0);
        assert_eq!(est.avg_energy_per_token_j(), 0.0);
        assert_eq!(est.avg_power_w(), 0.0);
    }

    #[test]
    fn test_estimator_single_sample() {
        let mut est = EnergyEstimator::new();
        est.record(EnergySample::new(200.0, Duration::from_secs(1), 5));
        assert!((est.total_energy_j() - 200.0).abs() < f64::EPSILON);
        assert_eq!(est.total_tokens(), 5);
        assert!((est.avg_energy_per_token_j() - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimator_multiple_samples() {
        let mut est = EnergyEstimator::new();
        est.record(EnergySample::new(100.0, Duration::from_secs(1), 10));
        est.record(EnergySample::new(200.0, Duration::from_secs(1), 10));
        assert!((est.total_energy_j() - 300.0).abs() < f64::EPSILON);
        assert_eq!(est.total_tokens(), 20);
        assert!((est.avg_energy_per_token_j() - 15.0).abs() < f64::EPSILON);
        assert!((est.avg_power_w() - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimator_reset() {
        let mut est = EnergyEstimator::new();
        est.record(EnergySample::new(100.0, Duration::from_secs(1), 5));
        est.reset();
        assert_eq!(est.sample_count(), 0);
        assert_eq!(est.total_energy_j(), 0.0);
    }

    // ===================================================================
    // PowerMonitor
    // ===================================================================

    #[test]
    fn test_monitor_initial_state() {
        let mon = PowerMonitor::new(PowerProfile::Balanced);
        assert_eq!(mon.current_state(), PowerState::Idle);
        assert_eq!(mon.history_len(), 0);
    }

    #[test]
    fn test_monitor_record_transitions_to_normal() {
        let mut mon = PowerMonitor::new(PowerProfile::Balanced);
        mon.record(default_thermal(), default_freq(), 180.0);
        assert_eq!(mon.current_state(), PowerState::Normal);
        assert_eq!(mon.history_len(), 1);
    }

    #[test]
    fn test_monitor_record_throttled() {
        let mut mon = PowerMonitor::new(PowerProfile::Balanced);
        // First transition Idle -> Normal
        mon.record(default_thermal(), default_freq(), 180.0);
        // Then Normal -> Throttled
        mon.record(critical_thermal(), default_freq(), 220.0);
        assert_eq!(mon.current_state(), PowerState::Throttled);
    }

    #[test]
    fn test_monitor_invalid_transition_stays_current() {
        let mut mon = PowerMonitor::new(PowerProfile::Performance);
        // Move to Boost first (Idle -> Boost is valid)
        let cool = ThermalZone::new(50.0, 90.0, 105.0);
        mon.record(cool, default_freq(), 200.0);
        assert_eq!(mon.current_state(), PowerState::Boost);
        // Boost -> LowPower is NOT valid (PowerSaver would recommend LowPower)
        mon.set_profile(PowerProfile::PowerSaver);
        mon.record(ThermalZone::new(50.0, 90.0, 105.0), default_freq(), 100.0);
        // Should stay Boost since Boost->LowPower is invalid
        assert_eq!(mon.current_state(), PowerState::Boost);
    }

    #[test]
    fn test_monitor_avg_power() {
        let mut mon = PowerMonitor::new(PowerProfile::Balanced);
        mon.record(default_thermal(), default_freq(), 100.0);
        mon.record(default_thermal(), default_freq(), 200.0);
        assert!((mon.avg_power_w() - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_monitor_peak_power() {
        let mut mon = PowerMonitor::new(PowerProfile::Balanced);
        mon.record(default_thermal(), default_freq(), 150.0);
        mon.record(default_thermal(), default_freq(), 220.0);
        mon.record(default_thermal(), default_freq(), 180.0);
        assert!((mon.peak_power_w() - 220.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_monitor_peak_temperature() {
        let mut mon = PowerMonitor::new(PowerProfile::Balanced);
        mon.record(ThermalZone::new(60.0, 90.0, 105.0), default_freq(), 180.0);
        mon.record(ThermalZone::new(85.0, 90.0, 105.0), default_freq(), 180.0);
        assert!((mon.peak_temperature_c() - 85.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_monitor_throttled_snapshots() {
        let mut mon = PowerMonitor::new(PowerProfile::Balanced);
        mon.record(default_thermal(), default_freq(), 180.0);
        mon.record(critical_thermal(), default_freq(), 220.0);
        mon.record(default_thermal(), default_freq(), 180.0);
        assert_eq!(mon.throttled_snapshots().len(), 1);
    }

    #[test]
    fn test_monitor_latest() {
        let mut mon = PowerMonitor::new(PowerProfile::Balanced);
        assert!(mon.latest().is_none());
        mon.record(default_thermal(), default_freq(), 180.0);
        assert!(mon.latest().is_some());
        assert!((mon.latest().unwrap().power_w - 180.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_monitor_force_state() {
        let mut mon = PowerMonitor::new(PowerProfile::Balanced);
        mon.force_state(PowerState::Boost);
        assert_eq!(mon.current_state(), PowerState::Boost);
    }

    #[test]
    fn test_monitor_reset() {
        let mut mon = PowerMonitor::new(PowerProfile::Balanced);
        mon.record(default_thermal(), default_freq(), 180.0);
        mon.reset();
        assert_eq!(mon.current_state(), PowerState::Idle);
        assert_eq!(mon.history_len(), 0);
    }

    #[test]
    fn test_monitor_empty_peaks() {
        let mon = PowerMonitor::new(PowerProfile::Balanced);
        assert_eq!(mon.peak_power_w(), 0.0);
        assert_eq!(mon.peak_temperature_c(), 0.0);
        assert_eq!(mon.avg_power_w(), 0.0);
    }

    // ===================================================================
    // Property / edge-case tests
    // ===================================================================

    #[test]
    fn test_all_states_have_at_least_one_transition() {
        for state in [
            PowerState::Idle,
            PowerState::LowPower,
            PowerState::Normal,
            PowerState::Boost,
            PowerState::Throttled,
        ] {
            assert!(
                !state.valid_transitions().is_empty(),
                "{state} must have at least one valid transition",
            );
        }
    }

    #[test]
    fn test_state_machine_reachability() {
        // Every state should be reachable from Normal (directly or via one hop).
        let all = [
            PowerState::Idle,
            PowerState::LowPower,
            PowerState::Normal,
            PowerState::Boost,
            PowerState::Throttled,
        ];
        for target in &all {
            let direct = PowerState::Normal.can_transition_to(*target);
            let via_hop = PowerState::Normal
                .valid_transitions()
                .iter()
                .any(|mid| mid.can_transition_to(*target));
            assert!(
                direct || via_hop || *target == PowerState::Normal,
                "{target} not reachable from Normal within 2 hops",
            );
        }
    }

    #[test]
    fn test_max_temperature_thermal_zone() {
        let tz = ThermalZone::new(f64::MAX, 90.0, 105.0);
        assert!(tz.is_throttling());
        assert!(tz.is_critical());
        assert_eq!(tz.headroom_c(), 0.0);
    }

    #[test]
    fn test_zero_frequency_info() {
        let fi = FrequencyInfo::new(0, 0, 0, 0);
        assert_eq!(fi.utilisation(), 0.0);
        assert!(!fi.is_boosting());
        assert!(fi.is_at_minimum());
    }

    #[test]
    fn test_instant_energy_sample() {
        let s = EnergySample::new(200.0, Duration::ZERO, 5);
        assert_eq!(s.energy_j(), 0.0);
        assert_eq!(s.energy_per_token_j(), 0.0);
    }

    #[test]
    fn test_power_config_equality() {
        let a = PowerConfig::new(225.0, 100.0, 0);
        let b = PowerConfig::new(225.0, 100.0, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_throttle_detector_many_events() {
        let mut det = ThrottleDetector::new(90.0, 225.0);
        for i in 0..100 {
            det.check(91.0 + i as f64 * 0.01, 180.0);
        }
        assert_eq!(det.event_count(), 100);
        assert_eq!(det.events_by_reason(ThrottleReason::Thermal).len(), 100);
        assert_eq!(det.events_by_reason(ThrottleReason::PowerBudget).len(), 0);
    }
}
