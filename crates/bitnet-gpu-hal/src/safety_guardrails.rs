//! Safety guardrails for GPU-accelerated inference pipelines.
//!
//! Provides input validation, output filtering, token budget enforcement,
//! rate limiting, content classification, circuit breaker, audit logging,
//! and a unified safety checking engine.
//!
//! All components are CPU-reference implementations suitable for use on both
//! CPU and GPU inference paths.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// SafetyLevel
// ---------------------------------------------------------------------------

/// Severity / classification level for content safety.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SafetyLevel {
    /// Content is safe for all audiences.
    Safe,
    /// Content may require review.
    Warning,
    /// Content is blocked by policy.
    Blocked,
}

impl std::fmt::Display for SafetyLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Safe => write!(f, "safe"),
            Self::Warning => write!(f, "warning"),
            Self::Blocked => write!(f, "blocked"),
        }
    }
}

// ---------------------------------------------------------------------------
// GuardrailError
// ---------------------------------------------------------------------------

/// Errors produced by the safety guardrails subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailError {
    /// Input failed validation.
    InputValidation(String),
    /// Output failed filtering.
    OutputFiltered(String),
    /// Token budget exhausted.
    TokenBudgetExceeded { limit: u64, requested: u64 },
    /// Rate limit exceeded.
    RateLimited { retry_after_ms: u64 },
    /// Circuit breaker is open.
    CircuitOpen { reason: String },
    /// Configuration error.
    Config(String),
}

impl std::fmt::Display for GuardrailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputValidation(msg) => write!(f, "input validation failed: {msg}"),
            Self::OutputFiltered(msg) => write!(f, "output filtered: {msg}"),
            Self::TokenBudgetExceeded { limit, requested } => {
                write!(f, "token budget exceeded: limit={limit}, requested={requested}")
            }
            Self::RateLimited { retry_after_ms } => {
                write!(f, "rate limited: retry after {retry_after_ms}ms")
            }
            Self::CircuitOpen { reason } => write!(f, "circuit open: {reason}"),
            Self::Config(msg) => write!(f, "config error: {msg}"),
        }
    }
}

// ---------------------------------------------------------------------------
// GuardrailConfig
// ---------------------------------------------------------------------------

/// Configuration for the safety guardrails pipeline.
///
/// Controls which checks are enabled, strictness levels, and custom rules.
///
/// # CPU Reference
///
/// All configuration is evaluated on the CPU regardless of inference device.
#[derive(Debug, Clone)]
pub struct GuardrailConfig {
    /// Enable input validation.
    pub input_validation_enabled: bool,
    /// Enable output filtering.
    pub output_filtering_enabled: bool,
    /// Enable token budget enforcement.
    pub token_budget_enabled: bool,
    /// Enable rate limiting.
    pub rate_limiting_enabled: bool,
    /// Enable content classification.
    pub content_classification_enabled: bool,
    /// Enable circuit breaker.
    pub circuit_breaker_enabled: bool,
    /// Enable audit logging.
    pub audit_logging_enabled: bool,
    /// Maximum input length in characters.
    pub max_input_length: usize,
    /// Maximum output length in characters.
    pub max_output_length: usize,
    /// Custom blocked patterns (substring matching).
    pub custom_blocked_patterns: Vec<String>,
    /// Strictness: if true, warnings are promoted to blocks.
    pub strict_mode: bool,
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        Self {
            input_validation_enabled: true,
            output_filtering_enabled: true,
            token_budget_enabled: true,
            rate_limiting_enabled: true,
            content_classification_enabled: true,
            circuit_breaker_enabled: true,
            audit_logging_enabled: true,
            max_input_length: 100_000,
            max_output_length: 100_000,
            custom_blocked_patterns: Vec::new(),
            strict_mode: false,
        }
    }
}

impl GuardrailConfig {
    /// Create a new config with all checks enabled and default limits.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a minimal config with all checks disabled.
    pub fn disabled() -> Self {
        Self {
            input_validation_enabled: false,
            output_filtering_enabled: false,
            token_budget_enabled: false,
            rate_limiting_enabled: false,
            content_classification_enabled: false,
            circuit_breaker_enabled: false,
            audit_logging_enabled: false,
            ..Self::default()
        }
    }

    /// Validate the configuration itself.
    pub fn validate(&self) -> Result<(), GuardrailError> {
        if self.max_input_length == 0 {
            return Err(GuardrailError::Config(
                "max_input_length must be > 0".into(),
            ));
        }
        if self.max_output_length == 0 {
            return Err(GuardrailError::Config(
                "max_output_length must be > 0".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// InputValidator
// ---------------------------------------------------------------------------

/// Validates input prompts before inference.
///
/// Checks for length limits, valid UTF-8 encoding, null bytes,
/// and potential prompt-injection patterns.
///
/// # CPU Reference
///
/// All validation runs on the CPU as a pre-processing step.
#[derive(Debug, Clone)]
pub struct InputValidator {
    max_length: usize,
    blocked_patterns: Vec<String>,
    allow_empty: bool,
}

impl InputValidator {
    /// Create a new validator with the given maximum input length.
    pub fn new(max_length: usize) -> Self {
        Self {
            max_length,
            blocked_patterns: Vec::new(),
            allow_empty: false,
        }
    }

    /// Add a blocked substring pattern.
    pub fn add_blocked_pattern(&mut self, pattern: &str) {
        self.blocked_patterns.push(pattern.to_lowercase());
    }

    /// Allow or disallow empty inputs.
    pub fn set_allow_empty(&mut self, allow: bool) {
        self.allow_empty = allow;
    }

    /// Validate the given input string.
    pub fn validate(&self, input: &str) -> Result<(), GuardrailError> {
        // Empty check
        if !self.allow_empty && input.trim().is_empty() {
            return Err(GuardrailError::InputValidation(
                "empty input not allowed".into(),
            ));
        }

        // Length check
        if input.len() > self.max_length {
            return Err(GuardrailError::InputValidation(format!(
                "input length {} exceeds maximum {}",
                input.len(),
                self.max_length
            )));
        }

        // Null byte check
        if input.contains('\0') {
            return Err(GuardrailError::InputValidation(
                "input contains null bytes".into(),
            ));
        }

        // Control character check (allow newline, tab, carriage return)
        for ch in input.chars() {
            if ch.is_control() && ch != '\n' && ch != '\r' && ch != '\t' {
                return Err(GuardrailError::InputValidation(format!(
                    "input contains disallowed control character U+{:04X}",
                    ch as u32
                )));
            }
        }

        // Blocked pattern check
        let lower = input.to_lowercase();
        for pattern in &self.blocked_patterns {
            if lower.contains(pattern.as_str()) {
                return Err(GuardrailError::InputValidation(format!(
                    "input matches blocked pattern: {pattern}"
                )));
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OutputFilter
// ---------------------------------------------------------------------------

/// Filters generated output for content policy compliance.
///
/// Detects PII patterns (emails, phone-like sequences) and applies
/// custom content-policy rules via blocked patterns.
///
/// # CPU Reference
///
/// Runs on the CPU as a post-processing step after token decoding.
#[derive(Debug, Clone)]
pub struct OutputFilter {
    max_length: usize,
    blocked_patterns: Vec<String>,
    redact_emails: bool,
    redact_phone_numbers: bool,
}

impl OutputFilter {
    /// Create a new output filter with the given maximum length.
    pub fn new(max_length: usize) -> Self {
        Self {
            max_length,
            blocked_patterns: Vec::new(),
            redact_emails: false,
            redact_phone_numbers: false,
        }
    }

    /// Add a blocked substring pattern for output.
    pub fn add_blocked_pattern(&mut self, pattern: &str) {
        self.blocked_patterns.push(pattern.to_lowercase());
    }

    /// Enable or disable email redaction.
    pub fn set_redact_emails(&mut self, redact: bool) {
        self.redact_emails = redact;
    }

    /// Enable or disable phone number redaction.
    pub fn set_redact_phone_numbers(&mut self, redact: bool) {
        self.redact_phone_numbers = redact;
    }

    /// Filter the output, returning the (possibly redacted) text or an error.
    pub fn filter(&self, output: &str) -> Result<String, GuardrailError> {
        // Length check
        if output.len() > self.max_length {
            return Err(GuardrailError::OutputFiltered(format!(
                "output length {} exceeds maximum {}",
                output.len(),
                self.max_length
            )));
        }

        // Blocked patterns
        let lower = output.to_lowercase();
        for pattern in &self.blocked_patterns {
            if lower.contains(pattern.as_str()) {
                return Err(GuardrailError::OutputFiltered(format!(
                    "output matches blocked pattern: {pattern}"
                )));
            }
        }

        let mut result = output.to_string();

        // Simple email redaction: find word@word.word patterns
        if self.redact_emails {
            result = Self::redact_email_patterns(&result);
        }

        // Simple phone redaction: sequences of digits with dashes/spaces
        if self.redact_phone_numbers {
            result = Self::redact_phone_patterns(&result);
        }

        Ok(result)
    }

    /// Redact email-like patterns (simple heuristic, no regex crate).
    fn redact_email_patterns(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if chars[i] == '@' && i > 0 && i + 1 < chars.len() {
                // Walk back to find start of local part
                let mut start = i;
                while start > 0
                    && (chars[start - 1].is_alphanumeric()
                        || chars[start - 1] == '.'
                        || chars[start - 1] == '_'
                        || chars[start - 1] == '-')
                {
                    start -= 1;
                }
                // Walk forward to find end of domain
                let mut end = i + 1;
                let mut has_dot = false;
                while end < chars.len()
                    && (chars[end].is_alphanumeric()
                        || chars[end] == '.'
                        || chars[end] == '-')
                {
                    if chars[end] == '.' {
                        has_dot = true;
                    }
                    end += 1;
                }
                if start < i && has_dot {
                    // Remove previously appended local-part chars
                    let local_len: usize =
                        chars[start..i].iter().map(|c| c.len_utf8()).sum();
                    result.truncate(result.len() - local_len);
                    result.push_str("[EMAIL REDACTED]");
                    i = end;
                    continue;
                }
            }
            result.push(chars[i]);
            i += 1;
        }
        result
    }

    /// Redact phone-like patterns (sequences of 7+ digits possibly separated
    /// by dashes, spaces, dots, or parens).
    fn redact_phone_patterns(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if chars[i].is_ascii_digit() || (chars[i] == '+' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) {
                let start = i;
                let mut digit_count = 0u32;
                let mut j = i;
                while j < chars.len()
                    && (chars[j].is_ascii_digit()
                        || chars[j] == '-'
                        || chars[j] == ' '
                        || chars[j] == '.'
                        || chars[j] == '('
                        || chars[j] == ')'
                        || chars[j] == '+')
                {
                    if chars[j].is_ascii_digit() {
                        digit_count += 1;
                    }
                    j += 1;
                }
                if digit_count >= 7 {
                    // Trim trailing separators
                    let mut end = j;
                    while end > start && !chars[end - 1].is_ascii_digit() {
                        end -= 1;
                    }
                    if end > start {
                        result.push_str("[PHONE REDACTED]");
                        i = end;
                        continue;
                    }
                }
            }
            result.push(chars[i]);
            i += 1;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// TokenBudgetEnforcer
// ---------------------------------------------------------------------------

/// Enforces token budget limits per request and per session.
///
/// Tracks cumulative token usage and rejects requests that would exceed
/// the configured budget.
///
/// # CPU Reference
///
/// Token counting is performed on the CPU before kernel dispatch.
#[derive(Debug, Clone)]
pub struct TokenBudgetEnforcer {
    /// Per-request token limit.
    per_request_limit: u64,
    /// Per-session token limit (0 = unlimited).
    per_session_limit: u64,
    /// Tokens consumed in the current session, keyed by session id.
    session_usage: HashMap<String, u64>,
}

impl TokenBudgetEnforcer {
    /// Create a new enforcer with per-request and per-session limits.
    pub fn new(per_request_limit: u64, per_session_limit: u64) -> Self {
        Self {
            per_request_limit,
            per_session_limit,
            session_usage: HashMap::new(),
        }
    }

    /// Check whether a request for `token_count` tokens is allowed.
    pub fn check_request(
        &self,
        session_id: &str,
        token_count: u64,
    ) -> Result<(), GuardrailError> {
        if token_count > self.per_request_limit {
            return Err(GuardrailError::TokenBudgetExceeded {
                limit: self.per_request_limit,
                requested: token_count,
            });
        }
        if self.per_session_limit > 0 {
            let used = self.session_usage.get(session_id).copied().unwrap_or(0);
            if used + token_count > self.per_session_limit {
                return Err(GuardrailError::TokenBudgetExceeded {
                    limit: self.per_session_limit,
                    requested: used + token_count,
                });
            }
        }
        Ok(())
    }

    /// Record that `token_count` tokens were consumed for a session.
    pub fn record_usage(&mut self, session_id: &str, token_count: u64) {
        *self
            .session_usage
            .entry(session_id.to_string())
            .or_insert(0) += token_count;
    }

    /// Return cumulative usage for a session.
    pub fn session_usage(&self, session_id: &str) -> u64 {
        self.session_usage.get(session_id).copied().unwrap_or(0)
    }

    /// Reset usage for a session.
    pub fn reset_session(&mut self, session_id: &str) {
        self.session_usage.remove(session_id);
    }

    /// Reset all session usage.
    pub fn reset_all(&mut self) {
        self.session_usage.clear();
    }
}

// ---------------------------------------------------------------------------
// RateLimiter
// ---------------------------------------------------------------------------

/// Token-bucket rate limiter keyed by an arbitrary identifier (user, session,
/// API key).
///
/// Each bucket refills at a fixed rate and allows bursts up to `capacity`.
///
/// # CPU Reference
///
/// Rate limiting runs entirely on the CPU.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    /// Maximum tokens per bucket (burst capacity).
    capacity: u64,
    /// Tokens added per second.
    refill_rate: f64,
    /// Per-key bucket state.
    buckets: HashMap<String, TokenBucket>,
}

#[derive(Debug, Clone)]
struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
}

impl RateLimiter {
    /// Create a rate limiter with `capacity` burst and `refill_rate` tokens/s.
    pub fn new(capacity: u64, refill_rate: f64) -> Self {
        Self {
            capacity,
            refill_rate,
            buckets: HashMap::new(),
        }
    }

    /// Try to consume one request for `key`. Returns `Ok(())` if allowed.
    pub fn try_acquire(&mut self, key: &str) -> Result<(), GuardrailError> {
        self.try_acquire_n(key, 1)
    }

    /// Try to consume `n` tokens for `key`.
    pub fn try_acquire_n(
        &mut self,
        key: &str,
        n: u64,
    ) -> Result<(), GuardrailError> {
        let now = Instant::now();
        let cap = self.capacity as f64;
        let bucket = self.buckets.entry(key.to_string()).or_insert(TokenBucket {
            tokens: cap,
            last_refill: now,
        });

        // Refill
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * self.refill_rate).min(cap);
        bucket.last_refill = now;

        let needed = n as f64;
        if bucket.tokens >= needed {
            bucket.tokens -= needed;
            Ok(())
        } else {
            let deficit = needed - bucket.tokens;
            let wait_ms = (deficit / self.refill_rate * 1000.0).ceil() as u64;
            Err(GuardrailError::RateLimited {
                retry_after_ms: wait_ms,
            })
        }
    }

    /// Return remaining tokens for `key` (creates bucket if absent).
    pub fn remaining(&mut self, key: &str) -> u64 {
        let now = Instant::now();
        let cap = self.capacity as f64;
        let bucket = self.buckets.entry(key.to_string()).or_insert(TokenBucket {
            tokens: cap,
            last_refill: now,
        });
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        let current = (bucket.tokens + elapsed * self.refill_rate).min(cap);
        current.floor() as u64
    }

    /// Reset the bucket for `key`.
    pub fn reset(&mut self, key: &str) {
        self.buckets.remove(key);
    }

    /// Reset all buckets.
    pub fn reset_all(&mut self) {
        self.buckets.clear();
    }
}

// ---------------------------------------------------------------------------
// ContentClassifier
// ---------------------------------------------------------------------------

/// Classifies content into safety levels based on keyword heuristics.
///
/// Maintains configurable word-lists for warning and blocked categories.
///
/// # CPU Reference
///
/// Classification is a CPU-only text scan.
#[derive(Debug, Clone)]
pub struct ContentClassifier {
    warning_patterns: Vec<String>,
    blocked_patterns: Vec<String>,
    strict: bool,
}

impl ContentClassifier {
    /// Create a classifier. If `strict`, warnings are promoted to blocked.
    pub fn new(strict: bool) -> Self {
        Self {
            warning_patterns: Vec::new(),
            blocked_patterns: Vec::new(),
            strict,
        }
    }

    /// Add a pattern that triggers a warning-level classification.
    pub fn add_warning_pattern(&mut self, pattern: &str) {
        self.warning_patterns.push(pattern.to_lowercase());
    }

    /// Add a pattern that triggers a blocked-level classification.
    pub fn add_blocked_pattern(&mut self, pattern: &str) {
        self.blocked_patterns.push(pattern.to_lowercase());
    }

    /// Classify the given text.
    pub fn classify(&self, text: &str) -> SafetyLevel {
        let lower = text.to_lowercase();

        for pattern in &self.blocked_patterns {
            if lower.contains(pattern.as_str()) {
                return SafetyLevel::Blocked;
            }
        }

        for pattern in &self.warning_patterns {
            if lower.contains(pattern.as_str()) {
                return if self.strict {
                    SafetyLevel::Blocked
                } else {
                    SafetyLevel::Warning
                };
            }
        }

        SafetyLevel::Safe
    }
}

// ---------------------------------------------------------------------------
// AuditLogger
// ---------------------------------------------------------------------------

/// In-memory audit logger that records safety-related events.
///
/// Each entry captures a timestamp, event type, and descriptive message.
///
/// # CPU Reference
///
/// Logging is a CPU-side bookkeeping operation.
#[derive(Debug, Clone)]
pub struct AuditLogger {
    entries: Vec<AuditEntry>,
    max_entries: usize,
}

/// A single audit log entry.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditEntry {
    /// Monotonic timestamp (duration since logger creation).
    pub timestamp: Duration,
    /// Event category.
    pub event_type: AuditEventType,
    /// Human-readable message.
    pub message: String,
    /// Optional key (user / session / API key).
    pub key: Option<String>,
}

/// Categories of auditable events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditEventType {
    /// Input validation succeeded.
    InputAccepted,
    /// Input validation failed.
    InputRejected,
    /// Output was filtered / redacted.
    OutputFiltered,
    /// Output passed filtering.
    OutputAccepted,
    /// Token budget check.
    TokenBudgetCheck,
    /// Rate limit triggered.
    RateLimitTriggered,
    /// Circuit breaker state change.
    CircuitBreakerStateChange,
    /// Content classified.
    ContentClassified,
    /// Safety check passed.
    SafetyCheckPassed,
    /// Safety check failed.
    SafetyCheckFailed,
}

impl AuditLogger {
    /// Create a new logger that keeps at most `max_entries` records.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    /// Log an event.
    pub fn log(
        &mut self,
        event_type: AuditEventType,
        message: &str,
        key: Option<&str>,
    ) {
        if self.entries.len() >= self.max_entries && self.max_entries > 0 {
            self.entries.remove(0);
        }
        self.entries.push(AuditEntry {
            // Use entries count as a monotonic proxy (no std::time::Instant
            // arithmetic needed in tests).
            timestamp: Duration::from_millis(self.entries.len() as u64),
            event_type,
            message: message.to_string(),
            key: key.map(String::from),
        });
    }

    /// Return all entries.
    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    /// Return entries of a specific type.
    pub fn entries_by_type(&self, event_type: AuditEventType) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Return entries for a specific key.
    pub fn entries_by_key(&self, key: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.key.as_deref() == Some(key))
            .collect()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of recorded entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// CircuitBreaker
// ---------------------------------------------------------------------------

/// Circuit breaker for safety-related failures.
///
/// Transitions: `Closed → Open` after `failure_threshold` consecutive
/// failures, `Open → HalfOpen` after `recovery_timeout`, `HalfOpen → Closed`
/// on success or back to `Open` on failure.
///
/// # CPU Reference
///
/// State management is CPU-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — requests are allowed.
    Closed,
    /// Breaker tripped — requests are rejected.
    Open,
    /// Tentatively allowing a single probe request.
    HalfOpen,
}

/// Circuit breaker that opens after repeated safety violations.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    failure_threshold: u32,
    success_threshold: u32,
    recovery_timeout: Duration,
    last_failure_time: Option<Instant>,
    total_trips: u64,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    ///
    /// * `failure_threshold` — consecutive failures before opening.
    /// * `success_threshold` — successes in half-open before closing.
    /// * `recovery_timeout` — time to wait before transitioning to half-open.
    pub fn new(
        failure_threshold: u32,
        success_threshold: u32,
        recovery_timeout: Duration,
    ) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            failure_threshold,
            success_threshold,
            recovery_timeout,
            last_failure_time: None,
            total_trips: 0,
        }
    }

    /// Current state.
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Check whether a request should be allowed.
    pub fn allow_request(&mut self) -> Result<(), GuardrailError> {
        match self.state {
            CircuitState::Closed => Ok(()),
            CircuitState::HalfOpen => Ok(()),
            CircuitState::Open => {
                // Check if recovery timeout has elapsed
                if let Some(last) = self.last_failure_time {
                    if last.elapsed() >= self.recovery_timeout {
                        self.state = CircuitState::HalfOpen;
                        self.success_count = 0;
                        return Ok(());
                    }
                }
                Err(GuardrailError::CircuitOpen {
                    reason: format!(
                        "circuit open after {} failures",
                        self.failure_count
                    ),
                })
            }
        }
    }

    /// Record a successful operation.
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            }
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failed operation.
    pub fn record_failure(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open;
                    self.last_failure_time = Some(Instant::now());
                    self.total_trips += 1;
                }
            }
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
                self.last_failure_time = Some(Instant::now());
                self.total_trips += 1;
            }
            CircuitState::Open => {}
        }
    }

    /// Force the breaker to the closed state.
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.last_failure_time = None;
    }

    /// Total number of times the breaker has tripped to open.
    pub fn total_trips(&self) -> u64 {
        self.total_trips
    }

    /// Current consecutive failure count.
    pub fn failure_count(&self) -> u32 {
        self.failure_count
    }
}

// ---------------------------------------------------------------------------
// GuardrailReport
// ---------------------------------------------------------------------------

/// A safety compliance report summarising guardrail activity.
///
/// # CPU Reference
///
/// Report generation is a CPU-only aggregation step.
#[derive(Debug, Clone)]
pub struct GuardrailReport {
    /// Total requests processed.
    pub total_requests: u64,
    /// Requests that passed all checks.
    pub passed: u64,
    /// Requests blocked by input validation.
    pub input_rejections: u64,
    /// Requests blocked by output filtering.
    pub output_rejections: u64,
    /// Requests blocked by token budget.
    pub budget_rejections: u64,
    /// Requests blocked by rate limiting.
    pub rate_limit_rejections: u64,
    /// Requests blocked by circuit breaker.
    pub circuit_breaker_rejections: u64,
    /// Content classification counts by level.
    pub classification_counts: HashMap<SafetyLevel, u64>,
}

impl GuardrailReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            passed: 0,
            input_rejections: 0,
            output_rejections: 0,
            budget_rejections: 0,
            rate_limit_rejections: 0,
            circuit_breaker_rejections: 0,
            classification_counts: HashMap::new(),
        }
    }

    /// Overall pass rate as a fraction in [0, 1].
    pub fn pass_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 1.0;
        }
        self.passed as f64 / self.total_requests as f64
    }

    /// Total rejections (sum of all rejection categories).
    pub fn total_rejections(&self) -> u64 {
        self.input_rejections
            + self.output_rejections
            + self.budget_rejections
            + self.rate_limit_rejections
            + self.circuit_breaker_rejections
    }

    /// Record a classification.
    pub fn record_classification(&mut self, level: SafetyLevel) {
        *self.classification_counts.entry(level).or_insert(0) += 1;
    }

    /// Format a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "total={} passed={} rejected={} pass_rate={:.1}%",
            self.total_requests,
            self.passed,
            self.total_rejections(),
            self.pass_rate() * 100.0,
        )
    }
}

impl Default for GuardrailReport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SafetyGuardrailEngine
// ---------------------------------------------------------------------------

/// Unified safety-checking pipeline that orchestrates all guardrail
/// components.
///
/// Call [`check_input`] before inference and [`check_output`] after
/// decoding. The engine maintains internal state (rate limits, circuit
/// breaker, audit log, report).
///
/// # CPU Reference
///
/// The engine runs entirely on the CPU.
#[derive(Debug)]
pub struct SafetyGuardrailEngine {
    config: GuardrailConfig,
    input_validator: InputValidator,
    output_filter: OutputFilter,
    token_budget: TokenBudgetEnforcer,
    rate_limiter: RateLimiter,
    classifier: ContentClassifier,
    circuit_breaker: CircuitBreaker,
    audit_logger: AuditLogger,
    report: GuardrailReport,
}

impl SafetyGuardrailEngine {
    /// Build an engine from the given configuration.
    pub fn new(config: GuardrailConfig) -> Self {
        let mut input_validator = InputValidator::new(config.max_input_length);
        for p in &config.custom_blocked_patterns {
            input_validator.add_blocked_pattern(p);
        }

        let mut output_filter = OutputFilter::new(config.max_output_length);
        for p in &config.custom_blocked_patterns {
            output_filter.add_blocked_pattern(p);
        }

        let classifier = ContentClassifier::new(config.strict_mode);

        Self {
            config,
            input_validator,
            output_filter,
            token_budget: TokenBudgetEnforcer::new(4096, 0),
            rate_limiter: RateLimiter::new(100, 10.0),
            classifier,
            circuit_breaker: CircuitBreaker::new(
                5,
                2,
                Duration::from_secs(30),
            ),
            audit_logger: AuditLogger::new(10_000),
            report: GuardrailReport::new(),
        }
    }

    /// Build with custom token budget limits.
    pub fn with_token_budget(
        mut self,
        per_request: u64,
        per_session: u64,
    ) -> Self {
        self.token_budget = TokenBudgetEnforcer::new(per_request, per_session);
        self
    }

    /// Build with custom rate limiter settings.
    pub fn with_rate_limiter(
        mut self,
        capacity: u64,
        refill_rate: f64,
    ) -> Self {
        self.rate_limiter = RateLimiter::new(capacity, refill_rate);
        self
    }

    /// Build with custom circuit breaker settings.
    pub fn with_circuit_breaker(
        mut self,
        failure_threshold: u32,
        success_threshold: u32,
        recovery_timeout: Duration,
    ) -> Self {
        self.circuit_breaker = CircuitBreaker::new(
            failure_threshold,
            success_threshold,
            recovery_timeout,
        );
        self
    }

    /// Validate an input prompt.
    ///
    /// Runs circuit breaker, rate limiter, input validation, content
    /// classification, and token budget checks.
    pub fn check_input(
        &mut self,
        input: &str,
        session_id: &str,
        token_count: u64,
    ) -> Result<SafetyLevel, GuardrailError> {
        self.report.total_requests += 1;

        // Circuit breaker
        if self.config.circuit_breaker_enabled {
            if let Err(e) = self.circuit_breaker.allow_request() {
                self.report.circuit_breaker_rejections += 1;
                self.audit_logger.log(
                    AuditEventType::CircuitBreakerStateChange,
                    &format!("circuit open: {e}"),
                    Some(session_id),
                );
                return Err(e);
            }
        }

        // Rate limiter
        if self.config.rate_limiting_enabled {
            if let Err(e) = self.rate_limiter.try_acquire(session_id) {
                self.report.rate_limit_rejections += 1;
                self.audit_logger.log(
                    AuditEventType::RateLimitTriggered,
                    "rate limit exceeded",
                    Some(session_id),
                );
                self.record_failure();
                return Err(e);
            }
        }

        // Input validation
        if self.config.input_validation_enabled {
            if let Err(e) = self.input_validator.validate(input) {
                self.report.input_rejections += 1;
                self.audit_logger.log(
                    AuditEventType::InputRejected,
                    &format!("{e}"),
                    Some(session_id),
                );
                self.record_failure();
                return Err(e);
            }
            self.audit_logger.log(
                AuditEventType::InputAccepted,
                "input validated",
                Some(session_id),
            );
        }

        // Token budget
        if self.config.token_budget_enabled {
            if let Err(e) =
                self.token_budget.check_request(session_id, token_count)
            {
                self.report.budget_rejections += 1;
                self.audit_logger.log(
                    AuditEventType::TokenBudgetCheck,
                    &format!("{e}"),
                    Some(session_id),
                );
                self.record_failure();
                return Err(e);
            }
        }

        // Content classification
        let level = if self.config.content_classification_enabled {
            let lvl = self.classifier.classify(input);
            self.report.record_classification(lvl);
            self.audit_logger.log(
                AuditEventType::ContentClassified,
                &format!("classified as {lvl}"),
                Some(session_id),
            );
            if lvl == SafetyLevel::Blocked {
                self.report.input_rejections += 1;
                self.record_failure();
                return Err(GuardrailError::InputValidation(
                    "content blocked by classifier".into(),
                ));
            }
            lvl
        } else {
            SafetyLevel::Safe
        };

        self.report.passed += 1;
        self.record_success();
        self.audit_logger.log(
            AuditEventType::SafetyCheckPassed,
            "input check passed",
            Some(session_id),
        );

        Ok(level)
    }

    /// Filter an output after inference.
    pub fn check_output(
        &mut self,
        output: &str,
        session_id: &str,
        tokens_used: u64,
    ) -> Result<String, GuardrailError> {
        // Record token usage
        if self.config.token_budget_enabled {
            self.token_budget.record_usage(session_id, tokens_used);
        }

        // Output filtering
        if self.config.output_filtering_enabled {
            match self.output_filter.filter(output) {
                Ok(filtered) => {
                    self.audit_logger.log(
                        AuditEventType::OutputAccepted,
                        "output accepted",
                        Some(session_id),
                    );
                    Ok(filtered)
                }
                Err(e) => {
                    self.report.output_rejections += 1;
                    self.audit_logger.log(
                        AuditEventType::OutputFiltered,
                        &format!("{e}"),
                        Some(session_id),
                    );
                    Err(e)
                }
            }
        } else {
            Ok(output.to_string())
        }
    }

    /// Access the current report.
    pub fn report(&self) -> &GuardrailReport {
        &self.report
    }

    /// Access the audit logger.
    pub fn audit_logger(&self) -> &AuditLogger {
        &self.audit_logger
    }

    /// Access the circuit breaker.
    pub fn circuit_breaker(&self) -> &CircuitBreaker {
        &self.circuit_breaker
    }

    /// Access the configuration.
    pub fn config(&self) -> &GuardrailConfig {
        &self.config
    }

    /// Mutable access to the content classifier for adding patterns.
    pub fn classifier_mut(&mut self) -> &mut ContentClassifier {
        &mut self.classifier
    }

    /// Mutable access to the output filter.
    pub fn output_filter_mut(&mut self) -> &mut OutputFilter {
        &mut self.output_filter
    }

    fn record_failure(&mut self) {
        if self.config.circuit_breaker_enabled {
            self.circuit_breaker.record_failure();
        }
    }

    fn record_success(&mut self) {
        if self.config.circuit_breaker_enabled {
            self.circuit_breaker.record_success();
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // GuardrailConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn config_default_has_all_enabled() {
        let c = GuardrailConfig::default();
        assert!(c.input_validation_enabled);
        assert!(c.output_filtering_enabled);
        assert!(c.token_budget_enabled);
        assert!(c.rate_limiting_enabled);
        assert!(c.content_classification_enabled);
        assert!(c.circuit_breaker_enabled);
        assert!(c.audit_logging_enabled);
    }

    #[test]
    fn config_disabled_has_none_enabled() {
        let c = GuardrailConfig::disabled();
        assert!(!c.input_validation_enabled);
        assert!(!c.output_filtering_enabled);
        assert!(!c.token_budget_enabled);
        assert!(!c.rate_limiting_enabled);
        assert!(!c.content_classification_enabled);
        assert!(!c.circuit_breaker_enabled);
        assert!(!c.audit_logging_enabled);
    }

    #[test]
    fn config_validate_ok() {
        assert!(GuardrailConfig::default().validate().is_ok());
    }

    #[test]
    fn config_validate_zero_input_length() {
        let mut c = GuardrailConfig::default();
        c.max_input_length = 0;
        assert!(matches!(c.validate(), Err(GuardrailError::Config(_))));
    }

    #[test]
    fn config_validate_zero_output_length() {
        let mut c = GuardrailConfig::default();
        c.max_output_length = 0;
        assert!(matches!(c.validate(), Err(GuardrailError::Config(_))));
    }

    #[test]
    fn config_new_equals_default() {
        let a = GuardrailConfig::new();
        let b = GuardrailConfig::default();
        assert_eq!(a.max_input_length, b.max_input_length);
        assert_eq!(a.strict_mode, b.strict_mode);
    }

    #[test]
    fn config_custom_blocked_patterns() {
        let mut c = GuardrailConfig::default();
        c.custom_blocked_patterns.push("bad".into());
        assert_eq!(c.custom_blocked_patterns.len(), 1);
    }

    #[test]
    fn config_strict_mode_default_false() {
        assert!(!GuardrailConfig::default().strict_mode);
    }

    // -----------------------------------------------------------------------
    // InputValidator tests
    // -----------------------------------------------------------------------

    #[test]
    fn input_valid_simple() {
        let v = InputValidator::new(1000);
        assert!(v.validate("Hello, world!").is_ok());
    }

    #[test]
    fn input_reject_empty() {
        let v = InputValidator::new(1000);
        assert!(v.validate("").is_err());
        assert!(v.validate("   ").is_err());
    }

    #[test]
    fn input_allow_empty_when_configured() {
        let mut v = InputValidator::new(1000);
        v.set_allow_empty(true);
        assert!(v.validate("").is_ok());
    }

    #[test]
    fn input_reject_too_long() {
        let v = InputValidator::new(10);
        assert!(v.validate("a]").is_ok());
        assert!(v.validate("a".repeat(11).as_str()).is_err());
    }

    #[test]
    fn input_reject_null_bytes() {
        let v = InputValidator::new(1000);
        assert!(v.validate("hello\0world").is_err());
    }

    #[test]
    fn input_reject_control_chars() {
        let v = InputValidator::new(1000);
        // BEL character
        let s = format!("hello{}world", '\x07');
        assert!(v.validate(&s).is_err());
    }

    #[test]
    fn input_allow_newline_tab_cr() {
        let v = InputValidator::new(1000);
        assert!(v.validate("hello\nworld").is_ok());
        assert!(v.validate("hello\tworld").is_ok());
        assert!(v.validate("hello\r\nworld").is_ok());
    }

    #[test]
    fn input_blocked_pattern() {
        let mut v = InputValidator::new(1000);
        v.add_blocked_pattern("forbidden");
        assert!(v.validate("this is forbidden text").is_err());
    }

    #[test]
    fn input_blocked_pattern_case_insensitive() {
        let mut v = InputValidator::new(1000);
        v.add_blocked_pattern("secret");
        assert!(v.validate("This is SECRET data").is_err());
    }

    #[test]
    fn input_multiple_blocked_patterns() {
        let mut v = InputValidator::new(1000);
        v.add_blocked_pattern("alpha");
        v.add_blocked_pattern("beta");
        assert!(v.validate("contains alpha").is_err());
        assert!(v.validate("contains beta").is_err());
        assert!(v.validate("contains gamma").is_ok());
    }

    #[test]
    fn input_exact_max_length() {
        let v = InputValidator::new(5);
        assert!(v.validate("abcde").is_ok());
        assert!(v.validate("abcdef").is_err());
    }

    #[test]
    fn input_unicode_valid() {
        let v = InputValidator::new(10000);
        assert!(v.validate("こんにちは世界").is_ok());
        assert!(v.validate("émojis: 🎉🚀").is_ok());
    }

    #[test]
    fn input_error_display() {
        let e = GuardrailError::InputValidation("test".into());
        assert!(format!("{e}").contains("test"));
    }

    // -----------------------------------------------------------------------
    // OutputFilter tests
    // -----------------------------------------------------------------------

    #[test]
    fn output_filter_passthrough() {
        let f = OutputFilter::new(1000);
        assert_eq!(f.filter("hello").unwrap(), "hello");
    }

    #[test]
    fn output_filter_too_long() {
        let f = OutputFilter::new(5);
        assert!(f.filter("abcdef").is_err());
    }

    #[test]
    fn output_filter_blocked_pattern() {
        let mut f = OutputFilter::new(1000);
        f.add_blocked_pattern("banned");
        assert!(f.filter("this is banned content").is_err());
    }

    #[test]
    fn output_filter_blocked_case_insensitive() {
        let mut f = OutputFilter::new(1000);
        f.add_blocked_pattern("restricted");
        assert!(f.filter("RESTRICTED zone").is_err());
    }

    #[test]
    fn output_redact_email() {
        let mut f = OutputFilter::new(10000);
        f.set_redact_emails(true);
        let result = f.filter("Contact user@example.com for details").unwrap();
        assert!(result.contains("[EMAIL REDACTED]"));
        assert!(!result.contains("user@example.com"));
    }

    #[test]
    fn output_redact_email_preserves_non_email_at() {
        let mut f = OutputFilter::new(10000);
        f.set_redact_emails(true);
        // Isolated @ should not cause issues
        let result = f.filter("value @ index").unwrap();
        assert!(result.contains("@"));
    }

    #[test]
    fn output_redact_phone() {
        let mut f = OutputFilter::new(10000);
        f.set_redact_phone_numbers(true);
        let result = f.filter("Call 555-123-4567 now").unwrap();
        assert!(result.contains("[PHONE REDACTED]"));
        assert!(!result.contains("555-123-4567"));
    }

    #[test]
    fn output_redact_phone_short_number_unchanged() {
        let mut f = OutputFilter::new(10000);
        f.set_redact_phone_numbers(true);
        let result = f.filter("Room 42 is ready").unwrap();
        assert_eq!(result, "Room 42 is ready");
    }

    #[test]
    fn output_redact_both() {
        let mut f = OutputFilter::new(10000);
        f.set_redact_emails(true);
        f.set_redact_phone_numbers(true);
        let result = f
            .filter("Email a@b.com or call 1234567890")
            .unwrap();
        assert!(result.contains("[EMAIL REDACTED]"));
        assert!(result.contains("[PHONE REDACTED]"));
    }

    #[test]
    fn output_filter_no_redaction_by_default() {
        let f = OutputFilter::new(10000);
        let result = f.filter("user@example.com 555-123-4567").unwrap();
        assert!(result.contains("user@example.com"));
        assert!(result.contains("555-123-4567"));
    }

    #[test]
    fn output_filter_exact_max_length() {
        let f = OutputFilter::new(5);
        assert!(f.filter("abcde").is_ok());
        assert!(f.filter("abcdef").is_err());
    }

    #[test]
    fn output_filter_empty_ok() {
        let f = OutputFilter::new(1000);
        assert_eq!(f.filter("").unwrap(), "");
    }

    // -----------------------------------------------------------------------
    // TokenBudgetEnforcer tests
    // -----------------------------------------------------------------------

    #[test]
    fn budget_within_request_limit() {
        let e = TokenBudgetEnforcer::new(100, 0);
        assert!(e.check_request("s1", 50).is_ok());
    }

    #[test]
    fn budget_exceed_request_limit() {
        let e = TokenBudgetEnforcer::new(100, 0);
        assert!(e.check_request("s1", 101).is_err());
    }

    #[test]
    fn budget_session_limit() {
        let mut e = TokenBudgetEnforcer::new(100, 200);
        assert!(e.check_request("s1", 100).is_ok());
        e.record_usage("s1", 100);
        assert!(e.check_request("s1", 100).is_ok());
        e.record_usage("s1", 100);
        assert!(e.check_request("s1", 1).is_err());
    }

    #[test]
    fn budget_session_usage_tracking() {
        let mut e = TokenBudgetEnforcer::new(1000, 0);
        assert_eq!(e.session_usage("s1"), 0);
        e.record_usage("s1", 50);
        assert_eq!(e.session_usage("s1"), 50);
        e.record_usage("s1", 30);
        assert_eq!(e.session_usage("s1"), 80);
    }

    #[test]
    fn budget_reset_session() {
        let mut e = TokenBudgetEnforcer::new(1000, 500);
        e.record_usage("s1", 400);
        e.reset_session("s1");
        assert_eq!(e.session_usage("s1"), 0);
        assert!(e.check_request("s1", 400).is_ok());
    }

    #[test]
    fn budget_reset_all() {
        let mut e = TokenBudgetEnforcer::new(1000, 500);
        e.record_usage("s1", 400);
        e.record_usage("s2", 300);
        e.reset_all();
        assert_eq!(e.session_usage("s1"), 0);
        assert_eq!(e.session_usage("s2"), 0);
    }

    #[test]
    fn budget_zero_session_limit_is_unlimited() {
        let mut e = TokenBudgetEnforcer::new(100, 0);
        for _ in 0..100 {
            e.record_usage("s1", 100);
        }
        // Per-request still enforced, but session is unlimited
        assert!(e.check_request("s1", 100).is_ok());
    }

    #[test]
    fn budget_independent_sessions() {
        let mut e = TokenBudgetEnforcer::new(100, 200);
        e.record_usage("s1", 150);
        // s2 is independent
        assert!(e.check_request("s2", 100).is_ok());
    }

    #[test]
    fn budget_error_display() {
        let e = GuardrailError::TokenBudgetExceeded {
            limit: 100,
            requested: 200,
        };
        let msg = format!("{e}");
        assert!(msg.contains("100"));
        assert!(msg.contains("200"));
    }

    // -----------------------------------------------------------------------
    // RateLimiter tests
    // -----------------------------------------------------------------------

    #[test]
    fn rate_limiter_allows_within_capacity() {
        let mut rl = RateLimiter::new(10, 1.0);
        for _ in 0..10 {
            assert!(rl.try_acquire("user1").is_ok());
        }
    }

    #[test]
    fn rate_limiter_rejects_over_capacity() {
        let mut rl = RateLimiter::new(3, 0.001);
        assert!(rl.try_acquire("u").is_ok());
        assert!(rl.try_acquire("u").is_ok());
        assert!(rl.try_acquire("u").is_ok());
        assert!(rl.try_acquire("u").is_err());
    }

    #[test]
    fn rate_limiter_independent_keys() {
        let mut rl = RateLimiter::new(2, 0.001);
        assert!(rl.try_acquire("a").is_ok());
        assert!(rl.try_acquire("a").is_ok());
        assert!(rl.try_acquire("a").is_err());
        // Different key should still have capacity
        assert!(rl.try_acquire("b").is_ok());
    }

    #[test]
    fn rate_limiter_remaining() {
        let mut rl = RateLimiter::new(10, 0.001);
        assert_eq!(rl.remaining("x"), 10);
        rl.try_acquire("x").unwrap();
        // After consuming 1, should have ~9
        assert!(rl.remaining("x") <= 10);
    }

    #[test]
    fn rate_limiter_reset_key() {
        let mut rl = RateLimiter::new(2, 0.001);
        rl.try_acquire("k").unwrap();
        rl.try_acquire("k").unwrap();
        assert!(rl.try_acquire("k").is_err());
        rl.reset("k");
        assert!(rl.try_acquire("k").is_ok());
    }

    #[test]
    fn rate_limiter_reset_all() {
        let mut rl = RateLimiter::new(1, 0.001);
        rl.try_acquire("a").unwrap();
        rl.try_acquire("b").unwrap();
        assert!(rl.try_acquire("a").is_err());
        rl.reset_all();
        assert!(rl.try_acquire("a").is_ok());
        assert!(rl.try_acquire("b").is_ok());
    }

    #[test]
    fn rate_limiter_try_acquire_n() {
        let mut rl = RateLimiter::new(10, 0.001);
        assert!(rl.try_acquire_n("u", 5).is_ok());
        assert!(rl.try_acquire_n("u", 5).is_ok());
        assert!(rl.try_acquire_n("u", 1).is_err());
    }

    #[test]
    fn rate_limiter_error_has_retry_after() {
        let mut rl = RateLimiter::new(1, 1.0);
        rl.try_acquire("u").unwrap();
        match rl.try_acquire("u") {
            Err(GuardrailError::RateLimited { retry_after_ms }) => {
                assert!(retry_after_ms > 0);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // ContentClassifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn classifier_safe_by_default() {
        let c = ContentClassifier::new(false);
        assert_eq!(c.classify("hello world"), SafetyLevel::Safe);
    }

    #[test]
    fn classifier_warning_pattern() {
        let mut c = ContentClassifier::new(false);
        c.add_warning_pattern("caution");
        assert_eq!(c.classify("use caution here"), SafetyLevel::Warning);
    }

    #[test]
    fn classifier_blocked_pattern() {
        let mut c = ContentClassifier::new(false);
        c.add_blocked_pattern("danger");
        assert_eq!(c.classify("danger zone"), SafetyLevel::Blocked);
    }

    #[test]
    fn classifier_blocked_takes_precedence() {
        let mut c = ContentClassifier::new(false);
        c.add_warning_pattern("alert");
        c.add_blocked_pattern("alert");
        assert_eq!(c.classify("alert!"), SafetyLevel::Blocked);
    }

    #[test]
    fn classifier_strict_promotes_warning() {
        let mut c = ContentClassifier::new(true);
        c.add_warning_pattern("maybe");
        assert_eq!(c.classify("maybe ok"), SafetyLevel::Blocked);
    }

    #[test]
    fn classifier_case_insensitive() {
        let mut c = ContentClassifier::new(false);
        c.add_blocked_pattern("stop");
        assert_eq!(c.classify("STOP right there"), SafetyLevel::Blocked);
    }

    #[test]
    fn classifier_no_patterns_always_safe() {
        let c = ContentClassifier::new(true);
        assert_eq!(c.classify("anything goes"), SafetyLevel::Safe);
    }

    #[test]
    fn classifier_multiple_patterns() {
        let mut c = ContentClassifier::new(false);
        c.add_warning_pattern("warn1");
        c.add_warning_pattern("warn2");
        c.add_blocked_pattern("block1");
        assert_eq!(c.classify("contains warn1"), SafetyLevel::Warning);
        assert_eq!(c.classify("contains warn2"), SafetyLevel::Warning);
        assert_eq!(c.classify("contains block1"), SafetyLevel::Blocked);
        assert_eq!(c.classify("safe text"), SafetyLevel::Safe);
    }

    // -----------------------------------------------------------------------
    // AuditLogger tests
    // -----------------------------------------------------------------------

    #[test]
    fn audit_logger_empty() {
        let l = AuditLogger::new(100);
        assert!(l.is_empty());
        assert_eq!(l.len(), 0);
    }

    #[test]
    fn audit_logger_basic_log() {
        let mut l = AuditLogger::new(100);
        l.log(AuditEventType::InputAccepted, "ok", Some("user1"));
        assert_eq!(l.len(), 1);
        assert_eq!(l.entries()[0].event_type, AuditEventType::InputAccepted);
        assert_eq!(l.entries()[0].message, "ok");
        assert_eq!(l.entries()[0].key.as_deref(), Some("user1"));
    }

    #[test]
    fn audit_logger_max_entries_eviction() {
        let mut l = AuditLogger::new(3);
        l.log(AuditEventType::InputAccepted, "1", None);
        l.log(AuditEventType::InputAccepted, "2", None);
        l.log(AuditEventType::InputAccepted, "3", None);
        l.log(AuditEventType::InputAccepted, "4", None);
        assert_eq!(l.len(), 3);
        assert_eq!(l.entries()[0].message, "2");
    }

    #[test]
    fn audit_logger_entries_by_type() {
        let mut l = AuditLogger::new(100);
        l.log(AuditEventType::InputAccepted, "a", None);
        l.log(AuditEventType::InputRejected, "b", None);
        l.log(AuditEventType::InputAccepted, "c", None);
        let accepted = l.entries_by_type(AuditEventType::InputAccepted);
        assert_eq!(accepted.len(), 2);
    }

    #[test]
    fn audit_logger_entries_by_key() {
        let mut l = AuditLogger::new(100);
        l.log(AuditEventType::InputAccepted, "a", Some("k1"));
        l.log(AuditEventType::InputAccepted, "b", Some("k2"));
        l.log(AuditEventType::InputAccepted, "c", Some("k1"));
        assert_eq!(l.entries_by_key("k1").len(), 2);
        assert_eq!(l.entries_by_key("k2").len(), 1);
        assert_eq!(l.entries_by_key("k3").len(), 0);
    }

    #[test]
    fn audit_logger_clear() {
        let mut l = AuditLogger::new(100);
        l.log(AuditEventType::InputAccepted, "a", None);
        l.clear();
        assert!(l.is_empty());
    }

    #[test]
    fn audit_logger_no_key() {
        let mut l = AuditLogger::new(100);
        l.log(AuditEventType::OutputAccepted, "msg", None);
        assert!(l.entries()[0].key.is_none());
    }

    #[test]
    fn audit_logger_timestamps_increase() {
        let mut l = AuditLogger::new(100);
        l.log(AuditEventType::InputAccepted, "1", None);
        l.log(AuditEventType::InputAccepted, "2", None);
        assert!(l.entries()[1].timestamp >= l.entries()[0].timestamp);
    }

    // -----------------------------------------------------------------------
    // CircuitBreaker tests
    // -----------------------------------------------------------------------

    #[test]
    fn circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new(3, 1, Duration::from_secs(5));
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn circuit_breaker_allows_when_closed() {
        let mut cb = CircuitBreaker::new(3, 1, Duration::from_secs(5));
        assert!(cb.allow_request().is_ok());
    }

    #[test]
    fn circuit_breaker_opens_after_threshold() {
        let mut cb = CircuitBreaker::new(3, 1, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn circuit_breaker_rejects_when_open() {
        let mut cb = CircuitBreaker::new(1, 1, Duration::from_secs(60));
        cb.record_failure();
        assert!(cb.allow_request().is_err());
    }

    #[test]
    fn circuit_breaker_half_open_after_timeout() {
        let mut cb = CircuitBreaker::new(1, 1, Duration::from_millis(1));
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(5));
        assert!(cb.allow_request().is_ok());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn circuit_breaker_closes_on_success_in_half_open() {
        let mut cb = CircuitBreaker::new(1, 1, Duration::from_millis(1));
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));
        cb.allow_request().unwrap();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn circuit_breaker_reopens_on_failure_in_half_open() {
        let mut cb = CircuitBreaker::new(1, 1, Duration::from_millis(1));
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));
        cb.allow_request().unwrap();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn circuit_breaker_reset() {
        let mut cb = CircuitBreaker::new(1, 1, Duration::from_secs(60));
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allow_request().is_ok());
    }

    #[test]
    fn circuit_breaker_success_resets_failure_count() {
        let mut cb = CircuitBreaker::new(3, 1, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert_eq!(cb.failure_count(), 0);
        // Now need 3 more failures to trip
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn circuit_breaker_total_trips() {
        let mut cb = CircuitBreaker::new(1, 1, Duration::from_millis(1));
        cb.record_failure();
        assert_eq!(cb.total_trips(), 1);
        std::thread::sleep(Duration::from_millis(5));
        cb.allow_request().unwrap();
        cb.record_failure();
        assert_eq!(cb.total_trips(), 2);
    }

    #[test]
    fn circuit_breaker_multi_success_threshold() {
        let mut cb = CircuitBreaker::new(1, 3, Duration::from_millis(1));
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));
        cb.allow_request().unwrap();
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    // -----------------------------------------------------------------------
    // GuardrailReport tests
    // -----------------------------------------------------------------------

    #[test]
    fn report_new_empty() {
        let r = GuardrailReport::new();
        assert_eq!(r.total_requests, 0);
        assert_eq!(r.passed, 0);
        assert_eq!(r.total_rejections(), 0);
    }

    #[test]
    fn report_pass_rate_empty() {
        assert!((GuardrailReport::new().pass_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn report_pass_rate_half() {
        let mut r = GuardrailReport::new();
        r.total_requests = 10;
        r.passed = 5;
        assert!((r.pass_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn report_total_rejections() {
        let mut r = GuardrailReport::new();
        r.input_rejections = 1;
        r.output_rejections = 2;
        r.budget_rejections = 3;
        r.rate_limit_rejections = 4;
        r.circuit_breaker_rejections = 5;
        assert_eq!(r.total_rejections(), 15);
    }

    #[test]
    fn report_classification_counts() {
        let mut r = GuardrailReport::new();
        r.record_classification(SafetyLevel::Safe);
        r.record_classification(SafetyLevel::Safe);
        r.record_classification(SafetyLevel::Warning);
        assert_eq!(r.classification_counts[&SafetyLevel::Safe], 2);
        assert_eq!(r.classification_counts[&SafetyLevel::Warning], 1);
    }

    #[test]
    fn report_summary_format() {
        let r = GuardrailReport::new();
        let s = r.summary();
        assert!(s.contains("total="));
        assert!(s.contains("passed="));
        assert!(s.contains("pass_rate="));
    }

    #[test]
    fn report_default() {
        let r = GuardrailReport::default();
        assert_eq!(r.total_requests, 0);
    }

    // -----------------------------------------------------------------------
    // SafetyLevel tests
    // -----------------------------------------------------------------------

    #[test]
    fn safety_level_display() {
        assert_eq!(format!("{}", SafetyLevel::Safe), "safe");
        assert_eq!(format!("{}", SafetyLevel::Warning), "warning");
        assert_eq!(format!("{}", SafetyLevel::Blocked), "blocked");
    }

    #[test]
    fn safety_level_eq() {
        assert_eq!(SafetyLevel::Safe, SafetyLevel::Safe);
        assert_ne!(SafetyLevel::Safe, SafetyLevel::Blocked);
    }

    // -----------------------------------------------------------------------
    // GuardrailError tests
    // -----------------------------------------------------------------------

    #[test]
    fn error_display_all_variants() {
        let variants = [
            GuardrailError::InputValidation("bad".into()),
            GuardrailError::OutputFiltered("nope".into()),
            GuardrailError::TokenBudgetExceeded {
                limit: 1,
                requested: 2,
            },
            GuardrailError::RateLimited { retry_after_ms: 5 },
            GuardrailError::CircuitOpen {
                reason: "tripped".into(),
            },
            GuardrailError::Config("invalid".into()),
        ];
        for v in &variants {
            assert!(!format!("{v}").is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // SafetyGuardrailEngine tests
    // -----------------------------------------------------------------------

    #[test]
    fn engine_check_input_basic() {
        let mut engine = SafetyGuardrailEngine::new(GuardrailConfig::default());
        let result = engine.check_input("Hello world", "s1", 10);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), SafetyLevel::Safe);
    }

    #[test]
    fn engine_check_input_empty_rejected() {
        let mut engine = SafetyGuardrailEngine::new(GuardrailConfig::default());
        assert!(engine.check_input("", "s1", 10).is_err());
    }

    #[test]
    fn engine_check_output_basic() {
        let mut engine = SafetyGuardrailEngine::new(GuardrailConfig::default());
        let result = engine.check_output("Generated text", "s1", 5);
        assert_eq!(result.unwrap(), "Generated text");
    }

    #[test]
    fn engine_check_output_blocked() {
        let mut config = GuardrailConfig::default();
        config.custom_blocked_patterns.push("forbidden".into());
        let mut engine = SafetyGuardrailEngine::new(config);
        assert!(engine.check_output("this is forbidden", "s1", 5).is_err());
    }

    #[test]
    fn engine_report_tracks_requests() {
        let mut engine = SafetyGuardrailEngine::new(GuardrailConfig::default());
        engine.check_input("a", "s1", 1).unwrap();
        engine.check_input("b", "s1", 1).unwrap();
        assert_eq!(engine.report().total_requests, 2);
        assert_eq!(engine.report().passed, 2);
    }

    #[test]
    fn engine_report_tracks_rejections() {
        let mut engine = SafetyGuardrailEngine::new(GuardrailConfig::default());
        let _ = engine.check_input("", "s1", 1);
        assert_eq!(engine.report().input_rejections, 1);
    }

    #[test]
    fn engine_audit_log_populated() {
        let mut engine = SafetyGuardrailEngine::new(GuardrailConfig::default());
        engine.check_input("hello", "s1", 1).unwrap();
        assert!(!engine.audit_logger().is_empty());
    }

    #[test]
    fn engine_disabled_config_passthrough() {
        let mut engine =
            SafetyGuardrailEngine::new(GuardrailConfig::disabled());
        assert!(engine.check_input("", "s1", 999_999).is_ok());
    }

    #[test]
    fn engine_with_token_budget() {
        let mut engine = SafetyGuardrailEngine::new(GuardrailConfig::default())
            .with_token_budget(10, 20);
        assert!(engine.check_input("ok", "s1", 10).is_ok());
        assert!(engine.check_input("ok", "s1", 11).is_err());
    }

    #[test]
    fn engine_with_rate_limiter() {
        let mut engine = SafetyGuardrailEngine::new(GuardrailConfig::default())
            .with_rate_limiter(2, 0.001);
        assert!(engine.check_input("a", "s1", 1).is_ok());
        assert!(engine.check_input("b", "s1", 1).is_ok());
        assert!(engine.check_input("c", "s1", 1).is_err());
    }

    #[test]
    fn engine_with_circuit_breaker() {
        let mut engine = SafetyGuardrailEngine::new(GuardrailConfig::default())
            .with_circuit_breaker(2, 1, Duration::from_secs(60));
        // Two failures should trip the breaker
        let _ = engine.check_input("", "s1", 1); // fail (empty)
        let _ = engine.check_input("", "s1", 1); // fail (empty)
        // Third should be circuit-broken
        let err = engine.check_input("valid", "s1", 1);
        assert!(matches!(err, Err(GuardrailError::CircuitOpen { .. })));
    }

    #[test]
    fn engine_classifier_blocks() {
        let mut engine = SafetyGuardrailEngine::new(GuardrailConfig::default());
        engine.classifier_mut().add_blocked_pattern("toxic");
        assert!(engine.check_input("this is toxic", "s1", 1).is_err());
    }

    #[test]
    fn engine_classifier_warning_not_blocked() {
        let mut engine = SafetyGuardrailEngine::new(GuardrailConfig::default());
        engine.classifier_mut().add_warning_pattern("edgy");
        let result = engine.check_input("edgy content", "s1", 1);
        assert_eq!(result.unwrap(), SafetyLevel::Warning);
    }

    #[test]
    fn engine_strict_mode_blocks_warnings() {
        let mut config = GuardrailConfig::default();
        config.strict_mode = true;
        let mut engine = SafetyGuardrailEngine::new(config);
        engine.classifier_mut().add_warning_pattern("risky");
        assert!(engine.check_input("risky text", "s1", 1).is_err());
    }

    #[test]
    fn engine_output_records_token_usage() {
        let mut engine = SafetyGuardrailEngine::new(GuardrailConfig::default())
            .with_token_budget(100, 50);
        engine.check_output("result", "s1", 30).unwrap();
        // Now session has 30 used, only 20 remaining
        assert!(engine.check_input("ok", "s1", 21).is_err());
    }

    #[test]
    fn engine_config_accessor() {
        let engine = SafetyGuardrailEngine::new(GuardrailConfig::default());
        assert!(engine.config().input_validation_enabled);
    }

    #[test]
    fn engine_circuit_breaker_accessor() {
        let engine = SafetyGuardrailEngine::new(GuardrailConfig::default());
        assert_eq!(engine.circuit_breaker().state(), CircuitState::Closed);
    }

    #[test]
    fn engine_output_filter_mut() {
        let mut engine = SafetyGuardrailEngine::new(GuardrailConfig::default());
        engine.output_filter_mut().set_redact_emails(true);
        let r = engine
            .check_output("contact a@b.com", "s1", 1)
            .unwrap();
        assert!(r.contains("[EMAIL REDACTED]"));
    }
}
