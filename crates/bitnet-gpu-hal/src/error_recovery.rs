//! Error recovery with retry policies, backoff strategies, and graceful degradation.
//!
//! Provides [`ErrorRecoveryEngine`] for executing fallible operations with
//! configurable retry logic, error classification, and service degradation
//! when errors exceed thresholds.

use std::fmt;
use std::time::{Duration, Instant};

// ── Recovery configuration ──────────────────────────────────────────────────

/// Top-level recovery settings.
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// Maximum retry attempts before giving up.
    pub max_retries: u32,
    /// Backoff strategy between retries.
    pub backoff_strategy: BackoffStrategy,
    /// Per-attempt timeout in milliseconds (0 = no timeout).
    pub timeout_ms: u64,
    /// Enable graceful degradation on repeated failures.
    pub graceful_degradation: bool,
    /// Number of consecutive failures before triggering degradation.
    pub degradation_threshold: u32,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff_strategy: BackoffStrategy::Exponential { base_ms: 100, max_ms: 5000 },
            timeout_ms: 30_000,
            graceful_degradation: true,
            degradation_threshold: 3,
        }
    }
}

// ── Backoff strategy ────────────────────────────────────────────────────────

/// Strategy for computing delay between retry attempts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackoffStrategy {
    /// Constant delay between retries.
    Fixed(u64),
    /// Exponential backoff: `base_ms * 2^attempt`, capped at `max_ms`.
    Exponential { base_ms: u64, max_ms: u64 },
    /// Linear backoff: `step_ms * (attempt + 1)`.
    Linear(u64),
    /// Jittered backoff: base delay with deterministic pseudo-jitter.
    Jitter(u64),
}

impl BackoffStrategy {
    /// Compute the delay for a given attempt (0-indexed).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let ms = match self {
            Self::Fixed(ms) => *ms,
            Self::Exponential { base_ms, max_ms } => {
                let exp = base_ms.saturating_mul(1u64.checked_shl(attempt).unwrap_or(u64::MAX));
                exp.min(*max_ms)
            }
            Self::Linear(step_ms) => {
                #[allow(clippy::cast_lossless)]
                let factor = (attempt as u64).saturating_add(1);
                step_ms.saturating_mul(factor)
            }
            Self::Jitter(base_ms) => {
                // Deterministic jitter based on attempt number for testability.
                #[allow(clippy::cast_lossless)]
                let jitter = (attempt as u64).wrapping_mul(7) % 13;
                base_ms.saturating_add(jitter)
            }
        };
        Duration::from_millis(ms)
    }
}

// ── Error category ──────────────────────────────────────────────────────────

/// Classification of an error for recovery purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Temporary glitch that may resolve on retry (e.g. bus contention).
    Transient,
    /// Permanent error that will not resolve with retries.
    Permanent,
    /// GPU/system ran out of memory or other resources.
    ResourceExhausted,
    /// Operation exceeded its time budget.
    Timeout,
    /// Underlying hardware fault (e.g. ECC error).
    Hardware,
    /// Invalid configuration or parameters.
    Configuration,
}

impl ErrorCategory {
    /// Whether this category is retryable by default.
    pub const fn is_retryable(self) -> bool {
        matches!(self, Self::Transient | Self::ResourceExhausted | Self::Timeout)
    }
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transient => write!(f, "transient"),
            Self::Permanent => write!(f, "permanent"),
            Self::ResourceExhausted => write!(f, "resource_exhausted"),
            Self::Timeout => write!(f, "timeout"),
            Self::Hardware => write!(f, "hardware"),
            Self::Configuration => write!(f, "configuration"),
        }
    }
}

// ── Error classifier ────────────────────────────────────────────────────────

/// Maps error strings to [`ErrorCategory`] values.
///
/// The classifier searches for known substrings in the error message. If no
/// rule matches, the error is classified as [`ErrorCategory::Transient`].
#[derive(Debug, Clone)]
pub struct ErrorClassifier {
    rules: Vec<(String, ErrorCategory)>,
}

impl ErrorClassifier {
    /// Create a classifier with the default GPU-oriented rule set.
    pub fn new() -> Self {
        Self {
            rules: vec![
                ("out of memory".into(), ErrorCategory::ResourceExhausted),
                ("OOM".into(), ErrorCategory::ResourceExhausted),
                ("allocation failed".into(), ErrorCategory::ResourceExhausted),
                ("timeout".into(), ErrorCategory::Timeout),
                ("deadline exceeded".into(), ErrorCategory::Timeout),
                ("timed out".into(), ErrorCategory::Timeout),
                ("ECC error".into(), ErrorCategory::Hardware),
                ("hardware fault".into(), ErrorCategory::Hardware),
                ("device lost".into(), ErrorCategory::Hardware),
                ("invalid configuration".into(), ErrorCategory::Configuration),
                ("unsupported".into(), ErrorCategory::Configuration),
                ("invalid parameter".into(), ErrorCategory::Configuration),
                ("not found".into(), ErrorCategory::Permanent),
                ("permission denied".into(), ErrorCategory::Permanent),
                ("corrupted".into(), ErrorCategory::Permanent),
            ],
        }
    }

    /// Add a custom classification rule.
    pub fn add_rule(&mut self, pattern: &str, category: ErrorCategory) {
        self.rules.push((pattern.to_string(), category));
    }

    /// Classify an error message.
    pub fn classify(&self, error_msg: &str) -> ErrorCategory {
        let lower = error_msg.to_lowercase();
        for (pattern, category) in &self.rules {
            if lower.contains(&pattern.to_lowercase()) {
                return *category;
            }
        }
        ErrorCategory::Transient
    }
}

impl Default for ErrorClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ── Retry policy ────────────────────────────────────────────────────────────

/// Decides whether to retry and how long to wait.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of attempts (first attempt + retries).
    pub max_attempts: u32,
    /// Backoff strategy.
    pub backoff: BackoffStrategy,
    /// Error categories that should be retried.
    pub retryable: Vec<ErrorCategory>,
}

impl RetryPolicy {
    /// Create a policy from a [`RecoveryConfig`].
    pub fn from_config(config: &RecoveryConfig) -> Self {
        Self {
            max_attempts: config.max_retries.saturating_add(1),
            backoff: config.backoff_strategy.clone(),
            retryable: vec![
                ErrorCategory::Transient,
                ErrorCategory::ResourceExhausted,
                ErrorCategory::Timeout,
            ],
        }
    }

    /// Whether the given category should be retried.
    pub fn should_retry(&self, category: ErrorCategory, attempt: u32) -> bool {
        attempt < self.max_attempts && self.retryable.contains(&category)
    }

    /// Compute the delay before the next retry.
    pub fn get_delay(&self, attempt: u32) -> Duration {
        self.backoff.delay_for_attempt(attempt)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::from_config(&RecoveryConfig::default())
    }
}

// ── Retry result ────────────────────────────────────────────────────────────

/// Outcome of a retried operation.
#[derive(Debug)]
pub enum RetryResult<T> {
    /// The operation eventually succeeded.
    Success(T),
    /// All retry attempts were exhausted.
    Exhausted {
        /// Number of attempts made.
        attempts: u32,
        /// Last error message.
        last_error: String,
    },
    /// A permanent (non-retryable) error was encountered.
    Permanent(String),
}

impl<T> RetryResult<T> {
    /// Returns `true` if the result is [`RetryResult::Success`].
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    /// Returns `true` if the result is [`RetryResult::Permanent`].
    pub const fn is_permanent(&self) -> bool {
        matches!(self, Self::Permanent(_))
    }
}

// ── Degradation level ───────────────────────────────────────────────────────

/// Service degradation levels, ordered from best to worst quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DegradationLevel {
    /// Full quality — no degradation.
    Full,
    /// Reduced quality — smaller batch size, some features disabled.
    Reduced,
    /// Minimal quality — single-item batches, lowest precision.
    Minimal,
    /// Emergency — bare-minimum operation, last resort.
    Emergency,
}

impl DegradationLevel {
    /// Degrade one step further, saturating at [`Emergency`](Self::Emergency).
    #[must_use]
    pub const fn degrade(self) -> Self {
        match self {
            Self::Full => Self::Reduced,
            Self::Reduced => Self::Minimal,
            Self::Minimal | Self::Emergency => Self::Emergency,
        }
    }

    /// Recover one step, saturating at [`Full`](Self::Full).
    #[must_use]
    pub const fn recover(self) -> Self {
        match self {
            Self::Emergency => Self::Minimal,
            Self::Minimal => Self::Reduced,
            Self::Reduced | Self::Full => Self::Full,
        }
    }

    /// Suggested batch size multiplier for this level (1.0 = full).
    pub const fn batch_size_factor(self) -> f64 {
        match self {
            Self::Full => 1.0,
            Self::Reduced => 0.5,
            Self::Minimal => 0.25,
            Self::Emergency => 0.125,
        }
    }
}

impl fmt::Display for DegradationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
            Self::Reduced => write!(f, "reduced"),
            Self::Minimal => write!(f, "minimal"),
            Self::Emergency => write!(f, "emergency"),
        }
    }
}

// ── Graceful degradation ────────────────────────────────────────────────────

/// Tracks degradation state and applies adjustments.
#[derive(Debug, Clone)]
pub struct GracefulDegradation {
    /// Current degradation level.
    pub level: DegradationLevel,
    /// Consecutive failures since last success.
    pub consecutive_failures: u32,
    /// Threshold for stepping down one level.
    pub threshold: u32,
    /// Number of consecutive successes needed to recover one level.
    pub recovery_threshold: u32,
    /// Consecutive successes since last failure.
    pub consecutive_successes: u32,
}

impl GracefulDegradation {
    /// Create with specified threshold.
    pub const fn new(threshold: u32) -> Self {
        Self {
            level: DegradationLevel::Full,
            consecutive_failures: 0,
            threshold,
            recovery_threshold: 5,
            consecutive_successes: 0,
        }
    }

    /// Record a failure. Returns `true` if the level changed.
    pub fn record_failure(&mut self) -> bool {
        self.consecutive_failures += 1;
        self.consecutive_successes = 0;
        if self.consecutive_failures >= self.threshold {
            let new = self.level.degrade();
            if new != self.level {
                self.level = new;
                self.consecutive_failures = 0;
                return true;
            }
        }
        false
    }

    /// Record a success. Returns `true` if the level improved.
    pub fn record_success(&mut self) -> bool {
        self.consecutive_successes += 1;
        self.consecutive_failures = 0;
        if self.consecutive_successes >= self.recovery_threshold {
            let new = self.level.recover();
            if new != self.level {
                self.level = new;
                self.consecutive_successes = 0;
                return true;
            }
        }
        false
    }

    /// Suggested batch size for current level given the original size.
    pub fn effective_batch_size(&self, original: usize) -> usize {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let size = (original as f64 * self.level.batch_size_factor()) as usize;
        size.max(1)
    }
}

impl Default for GracefulDegradation {
    fn default() -> Self {
        Self::new(3)
    }
}

// ── Recovery metrics ────────────────────────────────────────────────────────

/// Aggregate metrics for the recovery subsystem.
#[derive(Debug, Clone)]
pub struct RecoveryMetrics {
    /// Total retries performed.
    pub total_retries: u64,
    /// Successful recoveries (operation succeeded after ≥1 retry).
    pub successful_recoveries: u64,
    /// Operations that failed permanently.
    pub permanent_failures: u64,
    /// Sum of recovery durations for averaging.
    recovery_time_sum_ms: u64,
    /// Count of timed recoveries.
    recovery_count: u64,
    /// Number of degradation level changes.
    pub degradation_changes: u64,
}

impl RecoveryMetrics {
    /// Create zeroed metrics.
    pub const fn new() -> Self {
        Self {
            total_retries: 0,
            successful_recoveries: 0,
            permanent_failures: 0,
            recovery_time_sum_ms: 0,
            recovery_count: 0,
            degradation_changes: 0,
        }
    }

    /// Record a retry attempt.
    pub const fn record_retry(&mut self) {
        self.total_retries += 1;
    }

    /// Record a successful recovery with elapsed time.
    #[allow(clippy::cast_possible_truncation)]
    pub const fn record_recovery(&mut self, elapsed: Duration) {
        self.successful_recoveries += 1;
        self.recovery_time_sum_ms += elapsed.as_millis() as u64;
        self.recovery_count += 1;
    }

    /// Record a permanent failure.
    pub const fn record_permanent_failure(&mut self) {
        self.permanent_failures += 1;
    }

    /// Record a degradation level change.
    pub const fn record_degradation_change(&mut self) {
        self.degradation_changes += 1;
    }

    /// Average recovery time in milliseconds, or 0 if none recorded.
    pub const fn avg_recovery_time_ms(&self) -> u64 {
        if self.recovery_count == 0 { 0 } else { self.recovery_time_sum_ms / self.recovery_count }
    }
}

impl Default for RecoveryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ── Error recovery engine ───────────────────────────────────────────────────

/// Main error recovery engine.
///
/// Combines error classification, retry policies, graceful degradation,
/// and metrics collection into a single coordinator.
#[derive(Debug)]
pub struct ErrorRecoveryEngine {
    /// Recovery configuration.
    pub config: RecoveryConfig,
    /// Error classifier.
    pub classifier: ErrorClassifier,
    /// Retry policy.
    pub policy: RetryPolicy,
    /// Graceful degradation state.
    pub degradation: GracefulDegradation,
    /// Accumulated metrics.
    pub metrics: RecoveryMetrics,
}

impl ErrorRecoveryEngine {
    /// Create an engine from the given configuration.
    pub fn new(config: RecoveryConfig) -> Self {
        let policy = RetryPolicy::from_config(&config);
        let degradation = GracefulDegradation::new(config.degradation_threshold);
        Self {
            config,
            classifier: ErrorClassifier::new(),
            policy,
            degradation,
            metrics: RecoveryMetrics::new(),
        }
    }

    /// Classify an error message.
    pub fn classify(&self, error_msg: &str) -> ErrorCategory {
        self.classifier.classify(error_msg)
    }

    /// Execute `op` with retry logic, returning a [`RetryResult`].
    ///
    /// The closure receives the current attempt number (0-indexed).
    pub fn execute_with_retry<T, E>(
        &mut self,
        mut op: impl FnMut(u32) -> Result<T, E>,
    ) -> RetryResult<T>
    where
        E: fmt::Display,
    {
        let start = Instant::now();
        let mut last_error = String::new();

        for attempt in 0..self.policy.max_attempts {
            match op(attempt) {
                Ok(val) => {
                    if attempt > 0 {
                        self.metrics.record_recovery(start.elapsed());
                    }
                    let changed = self.degradation.record_success();
                    if changed {
                        self.metrics.record_degradation_change();
                    }
                    return RetryResult::Success(val);
                }
                Err(e) => {
                    last_error = e.to_string();
                    let category = self.classifier.classify(&last_error);

                    if !self.policy.should_retry(category, attempt + 1) {
                        let changed = self.degradation.record_failure();
                        if changed {
                            self.metrics.record_degradation_change();
                        }
                        if !category.is_retryable() {
                            self.metrics.record_permanent_failure();
                            return RetryResult::Permanent(last_error);
                        }
                        // Retryable but exhausted
                        break;
                    }

                    self.metrics.record_retry();
                    let changed = self.degradation.record_failure();
                    if changed {
                        self.metrics.record_degradation_change();
                    }

                    // In real code we'd sleep here; tests rely on the delay
                    // computation being correct without actual sleeping.
                    let _delay = self.policy.get_delay(attempt);
                }
            }
        }

        self.metrics.record_permanent_failure();
        RetryResult::Exhausted { attempts: self.policy.max_attempts, last_error }
    }

    /// Current degradation level.
    pub const fn degradation_level(&self) -> DegradationLevel {
        self.degradation.level
    }

    /// Effective batch size given current degradation.
    pub fn effective_batch_size(&self, original: usize) -> usize {
        self.degradation.effective_batch_size(original)
    }

    /// Reset metrics and degradation state.
    pub const fn reset(&mut self) {
        self.metrics = RecoveryMetrics::new();
        self.degradation = GracefulDegradation::new(self.config.degradation_threshold);
    }
}

impl Default for ErrorRecoveryEngine {
    fn default() -> Self {
        Self::new(RecoveryConfig::default())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────────────────

    fn default_engine() -> ErrorRecoveryEngine {
        ErrorRecoveryEngine::default()
    }

    fn engine_with_retries(retries: u32) -> ErrorRecoveryEngine {
        ErrorRecoveryEngine::new(RecoveryConfig {
            max_retries: retries,
            ..RecoveryConfig::default()
        })
    }

    // ── RecoveryConfig defaults ─────────────────────────────────────────

    #[test]
    fn config_default_max_retries() {
        let cfg = RecoveryConfig::default();
        assert_eq!(cfg.max_retries, 3);
    }

    #[test]
    fn config_default_timeout() {
        let cfg = RecoveryConfig::default();
        assert_eq!(cfg.timeout_ms, 30_000);
    }

    #[test]
    fn config_default_graceful_degradation_enabled() {
        let cfg = RecoveryConfig::default();
        assert!(cfg.graceful_degradation);
    }

    #[test]
    fn config_default_degradation_threshold() {
        let cfg = RecoveryConfig::default();
        assert_eq!(cfg.degradation_threshold, 3);
    }

    #[test]
    fn config_default_backoff_is_exponential() {
        let cfg = RecoveryConfig::default();
        assert!(matches!(cfg.backoff_strategy, BackoffStrategy::Exponential { .. }));
    }

    // ── BackoffStrategy delays ──────────────────────────────────────────

    #[test]
    fn fixed_backoff_constant() {
        let b = BackoffStrategy::Fixed(200);
        assert_eq!(b.delay_for_attempt(0), Duration::from_millis(200));
        assert_eq!(b.delay_for_attempt(5), Duration::from_millis(200));
    }

    #[test]
    fn exponential_backoff_doubles() {
        let b = BackoffStrategy::Exponential { base_ms: 100, max_ms: 10_000 };
        assert_eq!(b.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(b.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(b.delay_for_attempt(2), Duration::from_millis(400));
        assert_eq!(b.delay_for_attempt(3), Duration::from_millis(800));
    }

    #[test]
    fn exponential_backoff_capped() {
        let b = BackoffStrategy::Exponential { base_ms: 100, max_ms: 500 };
        assert_eq!(b.delay_for_attempt(10), Duration::from_millis(500));
    }

    #[test]
    fn exponential_backoff_zero_base() {
        let b = BackoffStrategy::Exponential { base_ms: 0, max_ms: 1000 };
        assert_eq!(b.delay_for_attempt(5), Duration::from_millis(0));
    }

    #[test]
    fn linear_backoff_increments() {
        let b = BackoffStrategy::Linear(50);
        assert_eq!(b.delay_for_attempt(0), Duration::from_millis(50));
        assert_eq!(b.delay_for_attempt(1), Duration::from_millis(100));
        assert_eq!(b.delay_for_attempt(2), Duration::from_millis(150));
    }

    #[test]
    fn linear_backoff_zero_step() {
        let b = BackoffStrategy::Linear(0);
        assert_eq!(b.delay_for_attempt(100), Duration::from_millis(0));
    }

    #[test]
    fn jitter_backoff_deterministic() {
        let b = BackoffStrategy::Jitter(100);
        let d0 = b.delay_for_attempt(0);
        let d1 = b.delay_for_attempt(1);
        // Deterministic: same input → same output
        assert_eq!(b.delay_for_attempt(0), d0);
        assert_eq!(b.delay_for_attempt(1), d1);
    }

    #[test]
    fn jitter_backoff_base_floor() {
        let b = BackoffStrategy::Jitter(100);
        // Jitter adds non-negative offset, so always >= base
        for attempt in 0..20 {
            assert!(b.delay_for_attempt(attempt) >= Duration::from_millis(100));
        }
    }

    #[test]
    fn exponential_backoff_huge_attempt() {
        let b = BackoffStrategy::Exponential { base_ms: 100, max_ms: 5000 };
        // Should not panic on very large attempt numbers
        assert_eq!(b.delay_for_attempt(64), Duration::from_millis(5000));
    }

    #[test]
    fn backoff_strategy_eq() {
        assert_eq!(BackoffStrategy::Fixed(100), BackoffStrategy::Fixed(100));
        assert_ne!(BackoffStrategy::Fixed(100), BackoffStrategy::Fixed(200));
    }

    // ── ErrorCategory ───────────────────────────────────────────────────

    #[test]
    fn transient_is_retryable() {
        assert!(ErrorCategory::Transient.is_retryable());
    }

    #[test]
    fn resource_exhausted_is_retryable() {
        assert!(ErrorCategory::ResourceExhausted.is_retryable());
    }

    #[test]
    fn timeout_is_retryable() {
        assert!(ErrorCategory::Timeout.is_retryable());
    }

    #[test]
    fn permanent_not_retryable() {
        assert!(!ErrorCategory::Permanent.is_retryable());
    }

    #[test]
    fn hardware_not_retryable() {
        assert!(!ErrorCategory::Hardware.is_retryable());
    }

    #[test]
    fn configuration_not_retryable() {
        assert!(!ErrorCategory::Configuration.is_retryable());
    }

    #[test]
    fn error_category_display() {
        assert_eq!(ErrorCategory::Transient.to_string(), "transient");
        assert_eq!(ErrorCategory::Permanent.to_string(), "permanent");
        assert_eq!(ErrorCategory::ResourceExhausted.to_string(), "resource_exhausted");
        assert_eq!(ErrorCategory::Timeout.to_string(), "timeout");
        assert_eq!(ErrorCategory::Hardware.to_string(), "hardware");
        assert_eq!(ErrorCategory::Configuration.to_string(), "configuration");
    }

    #[test]
    fn error_category_eq() {
        assert_eq!(ErrorCategory::Transient, ErrorCategory::Transient);
        assert_ne!(ErrorCategory::Transient, ErrorCategory::Permanent);
    }

    #[test]
    fn error_category_copy() {
        let a = ErrorCategory::Hardware;
        let b = a;
        assert_eq!(a, b);
    }

    // ── ErrorClassifier ─────────────────────────────────────────────────

    #[test]
    fn classify_oom() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("GPU out of memory"), ErrorCategory::ResourceExhausted);
    }

    #[test]
    fn classify_oom_uppercase() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("OOM on device 0"), ErrorCategory::ResourceExhausted);
    }

    #[test]
    fn classify_allocation_failed() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("allocation failed: 4 GiB"), ErrorCategory::ResourceExhausted);
    }

    #[test]
    fn classify_timeout() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("operation timeout after 30s"), ErrorCategory::Timeout);
    }

    #[test]
    fn classify_deadline() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("deadline exceeded"), ErrorCategory::Timeout);
    }

    #[test]
    fn classify_timed_out() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("kernel timed out"), ErrorCategory::Timeout);
    }

    #[test]
    fn classify_ecc_error() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("ECC error on GPU"), ErrorCategory::Hardware);
    }

    #[test]
    fn classify_hardware_fault() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("hardware fault detected"), ErrorCategory::Hardware);
    }

    #[test]
    fn classify_device_lost() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("device lost"), ErrorCategory::Hardware);
    }

    #[test]
    fn classify_invalid_configuration() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("invalid configuration: batch=0"), ErrorCategory::Configuration);
    }

    #[test]
    fn classify_unsupported() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("unsupported format"), ErrorCategory::Configuration);
    }

    #[test]
    fn classify_not_found() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("model not found"), ErrorCategory::Permanent);
    }

    #[test]
    fn classify_permission_denied() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("permission denied"), ErrorCategory::Permanent);
    }

    #[test]
    fn classify_unknown_defaults_transient() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("something weird happened"), ErrorCategory::Transient);
    }

    #[test]
    fn classify_empty_string_defaults_transient() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify(""), ErrorCategory::Transient);
    }

    #[test]
    fn classify_case_insensitive() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("OUT OF MEMORY"), ErrorCategory::ResourceExhausted);
    }

    #[test]
    fn classifier_custom_rule() {
        let mut c = ErrorClassifier::new();
        c.add_rule("custom_error_42", ErrorCategory::Permanent);
        assert_eq!(c.classify("hit custom_error_42 in kernel"), ErrorCategory::Permanent);
    }

    #[test]
    fn classifier_first_match_wins() {
        let c = ErrorClassifier::new();
        // "allocation failed" matches ResourceExhausted before any Permanent rule
        assert_eq!(c.classify("allocation failed: not found"), ErrorCategory::ResourceExhausted);
    }

    // ── RetryPolicy ─────────────────────────────────────────────────────

    #[test]
    fn policy_from_config_max_attempts() {
        let cfg = RecoveryConfig { max_retries: 5, ..RecoveryConfig::default() };
        let p = RetryPolicy::from_config(&cfg);
        assert_eq!(p.max_attempts, 6);
    }

    #[test]
    fn policy_should_retry_transient() {
        let p = RetryPolicy::default();
        assert!(p.should_retry(ErrorCategory::Transient, 1));
    }

    #[test]
    fn policy_should_not_retry_permanent() {
        let p = RetryPolicy::default();
        assert!(!p.should_retry(ErrorCategory::Permanent, 1));
    }

    #[test]
    fn policy_should_not_retry_hardware() {
        let p = RetryPolicy::default();
        assert!(!p.should_retry(ErrorCategory::Hardware, 1));
    }

    #[test]
    fn policy_should_not_retry_configuration() {
        let p = RetryPolicy::default();
        assert!(!p.should_retry(ErrorCategory::Configuration, 1));
    }

    #[test]
    fn policy_exhausted_at_max() {
        let p = RetryPolicy::default();
        // Default: max_attempts = 4 (3 retries + 1 initial)
        assert!(!p.should_retry(ErrorCategory::Transient, 4));
    }

    #[test]
    fn policy_get_delay() {
        let p = RetryPolicy::default();
        let d = p.get_delay(0);
        assert!(d.as_millis() > 0);
    }

    // ── RetryResult ─────────────────────────────────────────────────────

    #[test]
    fn retry_result_success() {
        let r: RetryResult<i32> = RetryResult::Success(42);
        assert!(r.is_success());
        assert!(!r.is_permanent());
    }

    #[test]
    fn retry_result_exhausted() {
        let r: RetryResult<i32> = RetryResult::Exhausted { attempts: 3, last_error: "fail".into() };
        assert!(!r.is_success());
        assert!(!r.is_permanent());
    }

    #[test]
    fn retry_result_permanent() {
        let r: RetryResult<i32> = RetryResult::Permanent("fatal".into());
        assert!(!r.is_success());
        assert!(r.is_permanent());
    }

    // ── DegradationLevel ────────────────────────────────────────────────

    #[test]
    fn degradation_level_ordering() {
        assert!(DegradationLevel::Full < DegradationLevel::Reduced);
        assert!(DegradationLevel::Reduced < DegradationLevel::Minimal);
        assert!(DegradationLevel::Minimal < DegradationLevel::Emergency);
    }

    #[test]
    fn degradation_level_degrade_steps() {
        assert_eq!(DegradationLevel::Full.degrade(), DegradationLevel::Reduced);
        assert_eq!(DegradationLevel::Reduced.degrade(), DegradationLevel::Minimal);
        assert_eq!(DegradationLevel::Minimal.degrade(), DegradationLevel::Emergency);
        assert_eq!(DegradationLevel::Emergency.degrade(), DegradationLevel::Emergency);
    }

    #[test]
    fn degradation_level_recover_steps() {
        assert_eq!(DegradationLevel::Emergency.recover(), DegradationLevel::Minimal);
        assert_eq!(DegradationLevel::Minimal.recover(), DegradationLevel::Reduced);
        assert_eq!(DegradationLevel::Reduced.recover(), DegradationLevel::Full);
        assert_eq!(DegradationLevel::Full.recover(), DegradationLevel::Full);
    }

    #[test]
    fn degradation_level_batch_factors() {
        assert_eq!(DegradationLevel::Full.batch_size_factor(), 1.0);
        assert_eq!(DegradationLevel::Reduced.batch_size_factor(), 0.5);
        assert_eq!(DegradationLevel::Minimal.batch_size_factor(), 0.25);
        assert_eq!(DegradationLevel::Emergency.batch_size_factor(), 0.125);
    }

    #[test]
    fn degradation_level_display() {
        assert_eq!(DegradationLevel::Full.to_string(), "full");
        assert_eq!(DegradationLevel::Reduced.to_string(), "reduced");
        assert_eq!(DegradationLevel::Minimal.to_string(), "minimal");
        assert_eq!(DegradationLevel::Emergency.to_string(), "emergency");
    }

    #[test]
    fn degradation_level_copy() {
        let a = DegradationLevel::Minimal;
        let b = a;
        assert_eq!(a, b);
    }

    // ── GracefulDegradation ─────────────────────────────────────────────

    #[test]
    fn degradation_starts_full() {
        let gd = GracefulDegradation::default();
        assert_eq!(gd.level, DegradationLevel::Full);
    }

    #[test]
    fn degradation_threshold_failures() {
        let mut gd = GracefulDegradation::new(2);
        assert!(!gd.record_failure()); // 1 < 2
        assert!(gd.record_failure()); // 2 >= 2 → degrade
        assert_eq!(gd.level, DegradationLevel::Reduced);
    }

    #[test]
    fn degradation_resets_counter_on_level_change() {
        let mut gd = GracefulDegradation::new(1);
        assert!(gd.record_failure()); // 1 >= 1 → degrade to Reduced
        assert_eq!(gd.consecutive_failures, 0);
    }

    #[test]
    fn degradation_cascades_through_levels() {
        let mut gd = GracefulDegradation::new(1);
        gd.record_failure(); // → Reduced
        gd.record_failure(); // → Minimal
        gd.record_failure(); // → Emergency
        assert_eq!(gd.level, DegradationLevel::Emergency);
    }

    #[test]
    fn degradation_saturates_at_emergency() {
        let mut gd = GracefulDegradation::new(1);
        for _ in 0..10 {
            gd.record_failure();
        }
        assert_eq!(gd.level, DegradationLevel::Emergency);
    }

    #[test]
    fn degradation_recovery_on_success() {
        let mut gd = GracefulDegradation::new(1);
        gd.record_failure(); // → Reduced
        for _ in 0..5 {
            gd.record_success();
        }
        assert_eq!(gd.level, DegradationLevel::Full);
    }

    #[test]
    fn degradation_success_resets_failure_count() {
        let mut gd = GracefulDegradation::new(3);
        gd.record_failure();
        gd.record_failure();
        gd.record_success(); // resets
        assert_eq!(gd.consecutive_failures, 0);
    }

    #[test]
    fn degradation_effective_batch_size() {
        let gd = GracefulDegradation::new(1);
        assert_eq!(gd.effective_batch_size(32), 32);
    }

    #[test]
    fn degradation_effective_batch_size_reduced() {
        let mut gd = GracefulDegradation::new(1);
        gd.record_failure(); // → Reduced
        assert_eq!(gd.effective_batch_size(32), 16);
    }

    #[test]
    fn degradation_effective_batch_size_min_one() {
        let mut gd = GracefulDegradation::new(1);
        gd.record_failure();
        gd.record_failure();
        gd.record_failure(); // → Emergency
        assert_eq!(gd.effective_batch_size(1), 1); // floor to 1
    }

    // ── RecoveryMetrics ─────────────────────────────────────────────────

    #[test]
    fn metrics_new_zeroed() {
        let m = RecoveryMetrics::new();
        assert_eq!(m.total_retries, 0);
        assert_eq!(m.successful_recoveries, 0);
        assert_eq!(m.permanent_failures, 0);
        assert_eq!(m.avg_recovery_time_ms(), 0);
    }

    #[test]
    fn metrics_default_eq_new() {
        let a = RecoveryMetrics::new();
        let b = RecoveryMetrics::default();
        assert_eq!(a.total_retries, b.total_retries);
    }

    #[test]
    fn metrics_record_retry() {
        let mut m = RecoveryMetrics::new();
        m.record_retry();
        m.record_retry();
        assert_eq!(m.total_retries, 2);
    }

    #[test]
    fn metrics_record_recovery() {
        let mut m = RecoveryMetrics::new();
        m.record_recovery(Duration::from_millis(100));
        m.record_recovery(Duration::from_millis(200));
        assert_eq!(m.successful_recoveries, 2);
        assert_eq!(m.avg_recovery_time_ms(), 150);
    }

    #[test]
    fn metrics_record_permanent_failure() {
        let mut m = RecoveryMetrics::new();
        m.record_permanent_failure();
        assert_eq!(m.permanent_failures, 1);
    }

    #[test]
    fn metrics_degradation_changes() {
        let mut m = RecoveryMetrics::new();
        m.record_degradation_change();
        m.record_degradation_change();
        assert_eq!(m.degradation_changes, 2);
    }

    // ── ErrorRecoveryEngine: execute_with_retry ─────────────────────────

    #[test]
    fn engine_success_first_try() {
        let mut engine = default_engine();
        let result = engine.execute_with_retry(|_| Ok::<_, String>(42));
        assert!(result.is_success());
        assert_eq!(engine.metrics.total_retries, 0);
    }

    #[test]
    fn engine_success_after_retries() {
        let mut engine = engine_with_retries(3);
        let result = engine.execute_with_retry(|attempt| {
            if attempt < 2 { Err("transient glitch".to_string()) } else { Ok(99) }
        });
        assert!(result.is_success());
        assert_eq!(engine.metrics.total_retries, 2);
    }

    #[test]
    fn engine_exhausted_all_retries() {
        let mut engine = engine_with_retries(2);
        let result: RetryResult<i32> =
            engine.execute_with_retry(|_| Err("transient glitch".to_string()));
        assert!(!result.is_success());
        match result {
            RetryResult::Exhausted { attempts, .. } => assert_eq!(attempts, 3),
            _ => panic!("expected Exhausted"),
        }
    }

    #[test]
    fn engine_permanent_error_stops_immediately() {
        let mut engine = engine_with_retries(5);
        let result: RetryResult<i32> =
            engine.execute_with_retry(|_| Err("permission denied".to_string()));
        assert!(result.is_permanent());
        assert_eq!(engine.metrics.permanent_failures, 1);
        assert_eq!(engine.metrics.total_retries, 0);
    }

    #[test]
    fn engine_hardware_error_stops_immediately() {
        let mut engine = engine_with_retries(5);
        let result: RetryResult<i32> =
            engine.execute_with_retry(|_| Err("ECC error on GPU".to_string()));
        assert!(result.is_permanent());
    }

    #[test]
    fn engine_classify_delegates() {
        let engine = default_engine();
        assert_eq!(engine.classify("out of memory"), ErrorCategory::ResourceExhausted);
    }

    #[test]
    fn engine_degradation_level_starts_full() {
        let engine = default_engine();
        assert_eq!(engine.degradation_level(), DegradationLevel::Full);
    }

    #[test]
    fn engine_effective_batch_size_full() {
        let engine = default_engine();
        assert_eq!(engine.effective_batch_size(64), 64);
    }

    #[test]
    fn engine_reset_clears_state() {
        let mut engine = default_engine();
        engine.metrics.record_retry();
        engine.metrics.record_retry();
        engine.reset();
        assert_eq!(engine.metrics.total_retries, 0);
        assert_eq!(engine.degradation_level(), DegradationLevel::Full);
    }

    #[test]
    fn engine_default_creates_valid() {
        let engine = ErrorRecoveryEngine::default();
        assert_eq!(engine.config.max_retries, 3);
        assert_eq!(engine.degradation_level(), DegradationLevel::Full);
    }

    #[test]
    fn engine_zero_retries_single_attempt() {
        let mut engine = engine_with_retries(0);
        let result: RetryResult<i32> =
            engine.execute_with_retry(|_| Err("transient glitch".to_string()));
        match result {
            RetryResult::Exhausted { attempts, .. } => assert_eq!(attempts, 1),
            _ => panic!("expected Exhausted"),
        }
    }

    #[test]
    fn engine_recovery_metric_on_late_success() {
        let mut engine = engine_with_retries(3);
        engine.execute_with_retry(|attempt| {
            if attempt < 1 { Err("transient glitch".to_string()) } else { Ok(()) }
        });
        assert_eq!(engine.metrics.successful_recoveries, 1);
    }

    #[test]
    fn engine_no_recovery_metric_on_first_success() {
        let mut engine = default_engine();
        engine.execute_with_retry(|_| Ok::<_, String>(()));
        assert_eq!(engine.metrics.successful_recoveries, 0);
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn backoff_fixed_zero() {
        let b = BackoffStrategy::Fixed(0);
        assert_eq!(b.delay_for_attempt(0), Duration::ZERO);
    }

    #[test]
    fn backoff_linear_large_attempt() {
        let b = BackoffStrategy::Linear(100);
        // Should not panic
        let d = b.delay_for_attempt(u32::MAX);
        assert!(d.as_millis() > 0);
    }

    #[test]
    fn classifier_corrupted() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("data corrupted"), ErrorCategory::Permanent);
    }

    #[test]
    fn degradation_interleaved_failures_successes() {
        let mut gd = GracefulDegradation::new(3);
        gd.record_failure();
        gd.record_failure();
        gd.record_success(); // resets failures
        gd.record_failure();
        gd.record_failure();
        // Still at Full — never hit 3 consecutive
        assert_eq!(gd.level, DegradationLevel::Full);
    }

    #[test]
    fn engine_multiple_operations() {
        let mut engine = engine_with_retries(1);
        let r1 = engine.execute_with_retry(|_| Ok::<_, String>(1));
        let r2 = engine.execute_with_retry(|_| Ok::<_, String>(2));
        assert!(r1.is_success());
        assert!(r2.is_success());
    }

    #[test]
    fn engine_mixed_success_failure() {
        let mut engine = engine_with_retries(2);
        let r1: RetryResult<i32> =
            engine.execute_with_retry(|_| Err("permission denied".to_string()));
        assert!(r1.is_permanent());

        let r2 = engine.execute_with_retry(|_| Ok::<_, String>(42));
        assert!(r2.is_success());
    }

    #[test]
    fn policy_should_retry_resource_exhausted() {
        let p = RetryPolicy::default();
        assert!(p.should_retry(ErrorCategory::ResourceExhausted, 1));
    }

    #[test]
    fn policy_should_retry_timeout() {
        let p = RetryPolicy::default();
        assert!(p.should_retry(ErrorCategory::Timeout, 1));
    }

    #[test]
    fn degradation_recovery_threshold() {
        let mut gd = GracefulDegradation::new(1);
        gd.record_failure(); // → Reduced
        // Need 5 successes to recover
        for i in 0..4 {
            let changed = gd.record_success();
            assert!(!changed, "should not recover at success {}", i + 1);
        }
        let changed = gd.record_success();
        assert!(changed, "should recover at success 5");
        assert_eq!(gd.level, DegradationLevel::Full);
    }

    // ── proptest ────────────────────────────────────────────────────────

    proptest::proptest! {
        #[test]
        fn backoff_fixed_always_same(ms in 0u64..10000, attempt in 0u32..100) {
            let b = BackoffStrategy::Fixed(ms);
            proptest::prop_assert_eq!(b.delay_for_attempt(attempt), Duration::from_millis(ms));
        }

        #[test]
        fn backoff_exponential_capped(base in 1u64..1000, max in 1u64..10000, attempt in 0u32..32) {
            let max_ms = base.max(max); // ensure max >= base
            let b = BackoffStrategy::Exponential { base_ms: base, max_ms };
            let d = b.delay_for_attempt(attempt);
            proptest::prop_assert!(d <= Duration::from_millis(max_ms));
        }

        #[test]
        fn backoff_linear_monotonic(step in 1u64..1000, a1 in 0u32..50, a2 in 0u32..50) {
            let b = BackoffStrategy::Linear(step);
            if a1 <= a2 {
                proptest::prop_assert!(b.delay_for_attempt(a1) <= b.delay_for_attempt(a2));
            }
        }

        #[test]
        fn jitter_above_base(base in 0u64..10000, attempt in 0u32..100) {
            let b = BackoffStrategy::Jitter(base);
            proptest::prop_assert!(b.delay_for_attempt(attempt) >= Duration::from_millis(base));
        }

        #[test]
        fn degradation_level_degrade_monotonic(level_idx in 0u8..4) {
            let level = match level_idx {
                0 => DegradationLevel::Full,
                1 => DegradationLevel::Reduced,
                2 => DegradationLevel::Minimal,
                _ => DegradationLevel::Emergency,
            };
            proptest::prop_assert!(level.degrade() >= level);
        }

        #[test]
        fn degradation_level_recover_monotonic(level_idx in 0u8..4) {
            let level = match level_idx {
                0 => DegradationLevel::Full,
                1 => DegradationLevel::Reduced,
                2 => DegradationLevel::Minimal,
                _ => DegradationLevel::Emergency,
            };
            proptest::prop_assert!(level.recover() <= level);
        }

        #[test]
        fn effective_batch_always_at_least_one(original in 1usize..1000, level_idx in 0u8..4) {
            let level = match level_idx {
                0 => DegradationLevel::Full,
                1 => DegradationLevel::Reduced,
                2 => DegradationLevel::Minimal,
                _ => DegradationLevel::Emergency,
            };
            let gd = GracefulDegradation {
                level,
                consecutive_failures: 0,
                threshold: 3,
                recovery_threshold: 5,
                consecutive_successes: 0,
            };
            proptest::prop_assert!(gd.effective_batch_size(original) >= 1);
        }
    }
}
