//! Configuration validation framework for `BitNet` inference settings.
//!
//! Provides composable validators for inference, model, device, and server
//! configurations with severity levels, human-readable suggestions, and
//! automatic configuration tuning based on hardware capabilities.

use std::fmt;
use std::path::Path;

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity level for a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    Hint,
    Info,
    Warning,
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hint => write!(f, "hint"),
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationResult
// ---------------------------------------------------------------------------

/// Outcome of evaluating a single validation rule.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub rule_name: String,
    pub passed: bool,
    pub severity: Severity,
    pub message: String,
    pub suggestion: Option<String>,
}

impl ValidationResult {
    #[must_use]
    pub fn pass(rule_name: impl Into<String>) -> Self {
        Self {
            rule_name: rule_name.into(),
            passed: true,
            severity: Severity::Info,
            message: String::new(),
            suggestion: None,
        }
    }

    #[must_use]
    pub fn fail(
        rule_name: impl Into<String>,
        severity: Severity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            rule_name: rule_name.into(),
            passed: false,
            severity,
            message: message.into(),
            suggestion: None,
        }
    }

    #[must_use]
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

// ---------------------------------------------------------------------------
// ValidationReport
// ---------------------------------------------------------------------------

/// Aggregated report produced by one or more validators.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub results: Vec<ValidationResult>,
}

impl ValidationReport {
    #[must_use]
    pub const fn new() -> Self {
        Self { results: Vec::new() }
    }

    pub fn push(&mut self, result: ValidationResult) {
        self.results.push(result);
    }

    pub fn merge(&mut self, other: &Self) {
        self.results.extend(other.results.iter().cloned());
    }

    #[must_use]
    pub fn error_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed && r.severity == Severity::Error).count()
    }

    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed && r.severity == Severity::Warning).count()
    }

    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.error_count() == 0
    }

    /// Return only results at or above the given severity.
    #[must_use]
    pub fn filter_by_severity(&self, min_severity: Severity) -> Vec<&ValidationResult> {
        self.results.iter().filter(|r| !r.passed && r.severity >= min_severity).collect()
    }

    #[must_use]
    pub fn failed_results(&self) -> Vec<&ValidationResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ValidationRule (type-erased, boxed closure)
// ---------------------------------------------------------------------------

/// A named validation rule that can be evaluated against an arbitrary config
/// represented as a set of key-value string pairs.
pub struct ValidationRule {
    pub name: String,
    pub description: String,
    pub severity: Severity,
    validate_fn: Box<dyn Fn(&ConfigMap) -> ValidationResult + Send + Sync>,
}

impl ValidationRule {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        severity: Severity,
        validate_fn: impl Fn(&ConfigMap) -> ValidationResult + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            severity,
            validate_fn: Box::new(validate_fn),
        }
    }

    #[must_use]
    pub fn evaluate(&self, config: &ConfigMap) -> ValidationResult {
        (self.validate_fn)(config)
    }
}

impl fmt::Debug for ValidationRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidationRule")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("severity", &self.severity)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// ConfigMap — lightweight config representation
// ---------------------------------------------------------------------------

/// Simple key→value configuration map used as the validation input.
pub type ConfigMap = std::collections::HashMap<String, String>;

// ---------------------------------------------------------------------------
// Validator trait
// ---------------------------------------------------------------------------

/// Trait implemented by each domain-specific validator.
pub trait Validator {
    /// Run all rules and return an aggregated report.
    fn validate(&self, config: &ConfigMap) -> ValidationReport;

    /// Human-readable name of this validator.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// InferenceConfigValidator
// ---------------------------------------------------------------------------

/// Validates inference parameters: temperature, `top_k`, `top_p`, `max_tokens`, etc.
pub struct InferenceConfigValidator {
    rules: Vec<ValidationRule>,
}

impl InferenceConfigValidator {
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new() -> Self {
        let mut rules = Vec::new();

        // -- temperature --
        rules.push(ValidationRule::new(
            "temperature_range",
            "Temperature must be in [0.0, 2.0]",
            Severity::Error,
            |cfg| {
                let Some(raw) = cfg.get("temperature") else {
                    return ValidationResult::pass("temperature_range");
                };
                let Ok(temp) = raw.parse::<f64>() else {
                    return ValidationResult::fail(
                        "temperature_range",
                        Severity::Error,
                        format!("temperature is not a valid number: {raw}"),
                    );
                };
                if temp < 0.0 {
                    return ValidationResult::fail(
                        "temperature_range",
                        Severity::Error,
                        format!("temperature must be >= 0.0, got {temp}"),
                    )
                    .with_suggestion("Set temperature to 0.0 for greedy decoding");
                }
                if temp > 2.0 {
                    return ValidationResult::fail(
                        "temperature_range",
                        Severity::Error,
                        format!("temperature must be <= 2.0, got {temp}"),
                    )
                    .with_suggestion("Reduce temperature to 2.0 or below");
                }
                ValidationResult::pass("temperature_range")
            },
        ));

        // -- temperature high warning --
        rules.push(ValidationRule::new(
            "temperature_high_warning",
            "Temperature above 1.5 may produce low-quality output",
            Severity::Warning,
            |cfg| {
                let Some(raw) = cfg.get("temperature") else {
                    return ValidationResult::pass("temperature_high_warning");
                };
                let Ok(temp) = raw.parse::<f64>() else {
                    return ValidationResult::pass("temperature_high_warning");
                };
                if temp > 1.5 && temp <= 2.0 {
                    return ValidationResult::fail(
                        "temperature_high_warning",
                        Severity::Warning,
                        format!("temperature {temp} is high; output quality may degrade"),
                    )
                    .with_suggestion("Consider a temperature between 0.7 and 1.0");
                }
                ValidationResult::pass("temperature_high_warning")
            },
        ));

        // -- top_k --
        rules.push(ValidationRule::new(
            "top_k_range",
            "top_k must be >= 0 (0 disables)",
            Severity::Error,
            |cfg| {
                let Some(raw) = cfg.get("top_k") else {
                    return ValidationResult::pass("top_k_range");
                };
                let Ok(k) = raw.parse::<i64>() else {
                    return ValidationResult::fail(
                        "top_k_range",
                        Severity::Error,
                        format!("top_k is not a valid integer: {raw}"),
                    );
                };
                if k < 0 {
                    return ValidationResult::fail(
                        "top_k_range",
                        Severity::Error,
                        format!("top_k must be >= 0, got {k}"),
                    )
                    .with_suggestion("Set top_k to 0 to disable, or a positive value");
                }
                ValidationResult::pass("top_k_range")
            },
        ));

        // -- top_k large warning --
        rules.push(ValidationRule::new(
            "top_k_large_warning",
            "Very large top_k values may slow sampling",
            Severity::Warning,
            |cfg| {
                let Some(raw) = cfg.get("top_k") else {
                    return ValidationResult::pass("top_k_large_warning");
                };
                let Ok(k) = raw.parse::<i64>() else {
                    return ValidationResult::pass("top_k_large_warning");
                };
                if k > 1000 {
                    return ValidationResult::fail(
                        "top_k_large_warning",
                        Severity::Warning,
                        format!("top_k={k} is unusually large"),
                    )
                    .with_suggestion("Typical top_k values are 10-100");
                }
                ValidationResult::pass("top_k_large_warning")
            },
        ));

        // -- top_p --
        rules.push(ValidationRule::new(
            "top_p_range",
            "top_p must be in (0.0, 1.0]",
            Severity::Error,
            |cfg| {
                let Some(raw) = cfg.get("top_p") else {
                    return ValidationResult::pass("top_p_range");
                };
                let Ok(p) = raw.parse::<f64>() else {
                    return ValidationResult::fail(
                        "top_p_range",
                        Severity::Error,
                        format!("top_p is not a valid number: {raw}"),
                    );
                };
                if p <= 0.0 || p > 1.0 {
                    return ValidationResult::fail(
                        "top_p_range",
                        Severity::Error,
                        format!("top_p must be in (0.0, 1.0], got {p}"),
                    )
                    .with_suggestion("Set top_p to 1.0 to disable nucleus sampling");
                }
                ValidationResult::pass("top_p_range")
            },
        ));

        // -- max_tokens --
        rules.push(ValidationRule::new(
            "max_tokens_range",
            "max_tokens must be >= 1",
            Severity::Error,
            |cfg| {
                let Some(raw) = cfg.get("max_tokens") else {
                    return ValidationResult::pass("max_tokens_range");
                };
                let Ok(n) = raw.parse::<i64>() else {
                    return ValidationResult::fail(
                        "max_tokens_range",
                        Severity::Error,
                        format!("max_tokens is not a valid integer: {raw}"),
                    );
                };
                if n < 1 {
                    return ValidationResult::fail(
                        "max_tokens_range",
                        Severity::Error,
                        format!("max_tokens must be >= 1, got {n}"),
                    )
                    .with_suggestion("Set max_tokens to at least 1");
                }
                ValidationResult::pass("max_tokens_range")
            },
        ));

        // -- max_tokens large warning --
        rules.push(ValidationRule::new(
            "max_tokens_large_warning",
            "Very large max_tokens may exhaust memory",
            Severity::Warning,
            |cfg| {
                let Some(raw) = cfg.get("max_tokens") else {
                    return ValidationResult::pass("max_tokens_large_warning");
                };
                let Ok(n) = raw.parse::<i64>() else {
                    return ValidationResult::pass("max_tokens_large_warning");
                };
                if n > 32768 {
                    return ValidationResult::fail(
                        "max_tokens_large_warning",
                        Severity::Warning,
                        format!("max_tokens={n} is very large; may cause OOM"),
                    )
                    .with_suggestion("Consider limiting max_tokens to the model's context length");
                }
                ValidationResult::pass("max_tokens_large_warning")
            },
        ));

        // -- repetition_penalty --
        rules.push(ValidationRule::new(
            "repetition_penalty_range",
            "repetition_penalty must be > 0.0",
            Severity::Error,
            |cfg| {
                let Some(raw) = cfg.get("repetition_penalty") else {
                    return ValidationResult::pass("repetition_penalty_range");
                };
                let Ok(rp) = raw.parse::<f64>() else {
                    return ValidationResult::fail(
                        "repetition_penalty_range",
                        Severity::Error,
                        format!("repetition_penalty is not a valid number: {raw}"),
                    );
                };
                if rp <= 0.0 {
                    return ValidationResult::fail(
                        "repetition_penalty_range",
                        Severity::Error,
                        format!("repetition_penalty must be > 0.0, got {rp}"),
                    )
                    .with_suggestion("Set repetition_penalty to 1.0 to disable");
                }
                ValidationResult::pass("repetition_penalty_range")
            },
        ));

        // -- seed non-negative --
        rules.push(ValidationRule::new(
            "seed_non_negative",
            "seed must be a non-negative integer when specified",
            Severity::Error,
            |cfg| {
                let Some(raw) = cfg.get("seed") else {
                    return ValidationResult::pass("seed_non_negative");
                };
                let Ok(s) = raw.parse::<i64>() else {
                    return ValidationResult::fail(
                        "seed_non_negative",
                        Severity::Error,
                        format!("seed is not a valid integer: {raw}"),
                    );
                };
                if s < 0 {
                    return ValidationResult::fail(
                        "seed_non_negative",
                        Severity::Error,
                        format!("seed must be >= 0, got {s}"),
                    )
                    .with_suggestion("Use a non-negative seed or omit for random");
                }
                ValidationResult::pass("seed_non_negative")
            },
        ));

        Self { rules }
    }
}

impl Default for InferenceConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl Validator for InferenceConfigValidator {
    fn validate(&self, config: &ConfigMap) -> ValidationReport {
        let mut report = ValidationReport::new();
        for rule in &self.rules {
            report.push(rule.evaluate(config));
        }
        report
    }

    fn name(&self) -> &'static str {
        "InferenceConfigValidator"
    }
}

// ---------------------------------------------------------------------------
// ModelConfigValidator
// ---------------------------------------------------------------------------

/// Validates model configuration: path existence, format, tokenizer.
pub struct ModelConfigValidator {
    rules: Vec<ValidationRule>,
}

impl ModelConfigValidator {
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new() -> Self {
        let mut rules = Vec::new();

        // -- model_path required --
        rules.push(ValidationRule::new(
            "model_path_required",
            "model_path must be specified",
            Severity::Error,
            |cfg| {
                if cfg.get("model_path").is_none_or(String::is_empty) {
                    return ValidationResult::fail(
                        "model_path_required",
                        Severity::Error,
                        "model_path is required",
                    )
                    .with_suggestion("Provide a path to a GGUF model file");
                }
                ValidationResult::pass("model_path_required")
            },
        ));

        // -- model_path exists --
        rules.push(ValidationRule::new(
            "model_path_exists",
            "model_path must point to an existing file",
            Severity::Error,
            |cfg| {
                let Some(path) = cfg.get("model_path") else {
                    return ValidationResult::pass("model_path_exists");
                };
                if path.is_empty() {
                    return ValidationResult::pass("model_path_exists");
                }
                if !Path::new(path).exists() {
                    return ValidationResult::fail(
                        "model_path_exists",
                        Severity::Error,
                        format!("model file not found: {path}"),
                    )
                    .with_suggestion("Check the path or download the model first");
                }
                ValidationResult::pass("model_path_exists")
            },
        ));

        // -- model format --
        rules.push(ValidationRule::new(
            "model_format_supported",
            "model format must be a supported type",
            Severity::Error,
            |cfg| {
                let Some(fmt) = cfg.get("model_format") else {
                    return ValidationResult::pass("model_format_supported");
                };
                let supported = ["gguf", "safetensors"];
                if !supported.contains(&fmt.to_lowercase().as_str()) {
                    return ValidationResult::fail(
                        "model_format_supported",
                        Severity::Error,
                        format!("unsupported model format: {fmt}"),
                    )
                    .with_suggestion("Supported formats: gguf, safetensors");
                }
                ValidationResult::pass("model_format_supported")
            },
        ));

        // -- tokenizer_path --
        rules.push(ValidationRule::new(
            "tokenizer_path_exists",
            "tokenizer_path must point to an existing file when specified",
            Severity::Error,
            |cfg| {
                let Some(path) = cfg.get("tokenizer_path") else {
                    return ValidationResult::pass("tokenizer_path_exists");
                };
                if path.is_empty() {
                    return ValidationResult::pass("tokenizer_path_exists");
                }
                if !Path::new(path).exists() {
                    return ValidationResult::fail(
                        "tokenizer_path_exists",
                        Severity::Error,
                        format!("tokenizer file not found: {path}"),
                    )
                    .with_suggestion("Ensure tokenizer.json exists alongside the model");
                }
                ValidationResult::pass("tokenizer_path_exists")
            },
        ));

        // -- context_length --
        rules.push(ValidationRule::new(
            "context_length_range",
            "context_length must be a positive power-of-two-friendly value",
            Severity::Warning,
            |cfg| {
                let Some(raw) = cfg.get("context_length") else {
                    return ValidationResult::pass("context_length_range");
                };
                let Ok(n) = raw.parse::<u64>() else {
                    return ValidationResult::fail(
                        "context_length_range",
                        Severity::Error,
                        format!("context_length is not a valid integer: {raw}"),
                    );
                };
                if n == 0 {
                    return ValidationResult::fail(
                        "context_length_range",
                        Severity::Error,
                        "context_length must be > 0".to_string(),
                    );
                }
                if n > 131_072 {
                    return ValidationResult::fail(
                        "context_length_range",
                        Severity::Warning,
                        format!("context_length={n} exceeds typical maximum (131072)"),
                    )
                    .with_suggestion("Most models support up to 131072 context length");
                }
                ValidationResult::pass("context_length_range")
            },
        ));

        Self { rules }
    }
}

impl Default for ModelConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl Validator for ModelConfigValidator {
    fn validate(&self, config: &ConfigMap) -> ValidationReport {
        let mut report = ValidationReport::new();
        for rule in &self.rules {
            report.push(rule.evaluate(config));
        }
        report
    }

    fn name(&self) -> &'static str {
        "ModelConfigValidator"
    }
}

// ---------------------------------------------------------------------------
// DeviceConfigValidator
// ---------------------------------------------------------------------------

/// Validates device configuration: availability, memory, feature support.
pub struct DeviceConfigValidator {
    rules: Vec<ValidationRule>,
}

impl DeviceConfigValidator {
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new() -> Self {
        let mut rules = Vec::new();

        // -- device_type --
        rules.push(ValidationRule::new(
            "device_type_supported",
            "device must be cpu, cuda, rocm, or vulkan",
            Severity::Error,
            |cfg| {
                let Some(dev) = cfg.get("device") else {
                    return ValidationResult::pass("device_type_supported");
                };
                let supported = ["cpu", "cuda", "rocm", "vulkan"];
                if !supported.contains(&dev.to_lowercase().as_str()) {
                    return ValidationResult::fail(
                        "device_type_supported",
                        Severity::Error,
                        format!("unsupported device type: {dev}"),
                    )
                    .with_suggestion("Supported devices: cpu, cuda, rocm, vulkan");
                }
                ValidationResult::pass("device_type_supported")
            },
        ));

        // -- gpu_memory_mb --
        rules.push(ValidationRule::new(
            "gpu_memory_sufficient",
            "GPU memory should be >= 512 MB for inference",
            Severity::Warning,
            |cfg| {
                let Some(raw) = cfg.get("gpu_memory_mb") else {
                    return ValidationResult::pass("gpu_memory_sufficient");
                };
                let Ok(mem) = raw.parse::<u64>() else {
                    return ValidationResult::fail(
                        "gpu_memory_sufficient",
                        Severity::Error,
                        format!("gpu_memory_mb is not a valid integer: {raw}"),
                    );
                };
                if mem < 512 {
                    return ValidationResult::fail(
                        "gpu_memory_sufficient",
                        Severity::Warning,
                        format!("gpu_memory_mb={mem} may be insufficient for inference"),
                    )
                    .with_suggestion("At least 512 MB GPU memory is recommended");
                }
                ValidationResult::pass("gpu_memory_sufficient")
            },
        ));

        // -- device_index --
        rules.push(ValidationRule::new(
            "device_index_valid",
            "device_index must be non-negative",
            Severity::Error,
            |cfg| {
                let Some(raw) = cfg.get("device_index") else {
                    return ValidationResult::pass("device_index_valid");
                };
                let Ok(idx) = raw.parse::<i64>() else {
                    return ValidationResult::fail(
                        "device_index_valid",
                        Severity::Error,
                        format!("device_index is not a valid integer: {raw}"),
                    );
                };
                if idx < 0 {
                    return ValidationResult::fail(
                        "device_index_valid",
                        Severity::Error,
                        format!("device_index must be >= 0, got {idx}"),
                    )
                    .with_suggestion("Use 0 for the first (default) device");
                }
                ValidationResult::pass("device_index_valid")
            },
        ));

        // -- compute_capability (CUDA) --
        rules.push(ValidationRule::new(
            "compute_capability_min",
            "CUDA compute capability should be >= 6.0",
            Severity::Warning,
            |cfg| {
                let Some(raw) = cfg.get("compute_capability") else {
                    return ValidationResult::pass("compute_capability_min");
                };
                let Ok(cc) = raw.parse::<f64>() else {
                    return ValidationResult::fail(
                        "compute_capability_min",
                        Severity::Error,
                        format!("compute_capability is not a valid number: {raw}"),
                    );
                };
                if cc < 6.0 {
                    return ValidationResult::fail(
                        "compute_capability_min",
                        Severity::Warning,
                        format!("compute_capability {cc} < 6.0; some kernels may be unavailable"),
                    )
                    .with_suggestion("CUDA compute capability >= 6.0 (Pascal+) is recommended");
                }
                ValidationResult::pass("compute_capability_min")
            },
        ));

        // -- threads (CPU) --
        rules.push(ValidationRule::new(
            "thread_count_valid",
            "threads must be >= 1",
            Severity::Error,
            |cfg| {
                let Some(raw) = cfg.get("threads") else {
                    return ValidationResult::pass("thread_count_valid");
                };
                let Ok(t) = raw.parse::<i64>() else {
                    return ValidationResult::fail(
                        "thread_count_valid",
                        Severity::Error,
                        format!("threads is not a valid integer: {raw}"),
                    );
                };
                if t < 1 {
                    return ValidationResult::fail(
                        "thread_count_valid",
                        Severity::Error,
                        format!("threads must be >= 1, got {t}"),
                    )
                    .with_suggestion("Set threads to at least 1");
                }
                ValidationResult::pass("thread_count_valid")
            },
        ));

        Self { rules }
    }
}

impl Default for DeviceConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl Validator for DeviceConfigValidator {
    fn validate(&self, config: &ConfigMap) -> ValidationReport {
        let mut report = ValidationReport::new();
        for rule in &self.rules {
            report.push(rule.evaluate(config));
        }
        report
    }

    fn name(&self) -> &'static str {
        "DeviceConfigValidator"
    }
}

// ---------------------------------------------------------------------------
// ServerConfigValidator
// ---------------------------------------------------------------------------

/// Validates server configuration: port, bind address, concurrency limits.
pub struct ServerConfigValidator {
    rules: Vec<ValidationRule>,
}

impl ServerConfigValidator {
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new() -> Self {
        let mut rules = Vec::new();

        // -- port --
        rules.push(ValidationRule::new(
            "port_range",
            "port must be in 1..=65535",
            Severity::Error,
            |cfg| {
                let Some(raw) = cfg.get("port") else {
                    return ValidationResult::pass("port_range");
                };
                let Ok(port) = raw.parse::<i64>() else {
                    return ValidationResult::fail(
                        "port_range",
                        Severity::Error,
                        format!("port is not a valid integer: {raw}"),
                    );
                };
                if !(1..=65535).contains(&port) {
                    return ValidationResult::fail(
                        "port_range",
                        Severity::Error,
                        format!("port must be in 1..=65535, got {port}"),
                    )
                    .with_suggestion("Use a port between 1024 and 65535");
                }
                ValidationResult::pass("port_range")
            },
        ));

        // -- privileged port warning --
        rules.push(ValidationRule::new(
            "port_privileged_warning",
            "Ports below 1024 require elevated privileges",
            Severity::Warning,
            |cfg| {
                let Some(raw) = cfg.get("port") else {
                    return ValidationResult::pass("port_privileged_warning");
                };
                let Ok(port) = raw.parse::<i64>() else {
                    return ValidationResult::pass("port_privileged_warning");
                };
                if (1..1024).contains(&port) {
                    return ValidationResult::fail(
                        "port_privileged_warning",
                        Severity::Warning,
                        format!("port {port} is privileged; may require root/admin"),
                    )
                    .with_suggestion("Use a port >= 1024 to avoid privilege issues");
                }
                ValidationResult::pass("port_privileged_warning")
            },
        ));

        // -- bind_address --
        rules.push(ValidationRule::new(
            "bind_address_valid",
            "bind_address must be a valid IP or hostname",
            Severity::Error,
            |cfg| {
                let Some(addr) = cfg.get("bind_address") else {
                    return ValidationResult::pass("bind_address_valid");
                };
                if addr.is_empty() {
                    return ValidationResult::fail(
                        "bind_address_valid",
                        Severity::Error,
                        "bind_address must not be empty",
                    )
                    .with_suggestion("Use 0.0.0.0 to bind to all interfaces");
                }
                // Basic validation: parseable as IP or is a plausible hostname
                let is_ip = addr.parse::<std::net::IpAddr>().is_ok();
                let is_hostname = addr.chars().all(|c| c.is_alphanumeric() || c == '.' || c == '-');
                if !is_ip && !is_hostname {
                    return ValidationResult::fail(
                        "bind_address_valid",
                        Severity::Error,
                        format!("bind_address is not a valid IP or hostname: {addr}"),
                    )
                    .with_suggestion("Use an IP like 0.0.0.0 or 127.0.0.1");
                }
                ValidationResult::pass("bind_address_valid")
            },
        ));

        // -- max_concurrent --
        rules.push(ValidationRule::new(
            "max_concurrent_range",
            "max_concurrent must be >= 1",
            Severity::Error,
            |cfg| {
                let Some(raw) = cfg.get("max_concurrent") else {
                    return ValidationResult::pass("max_concurrent_range");
                };
                let Ok(n) = raw.parse::<i64>() else {
                    return ValidationResult::fail(
                        "max_concurrent_range",
                        Severity::Error,
                        format!("max_concurrent is not a valid integer: {raw}"),
                    );
                };
                if n < 1 {
                    return ValidationResult::fail(
                        "max_concurrent_range",
                        Severity::Error,
                        format!("max_concurrent must be >= 1, got {n}"),
                    )
                    .with_suggestion("Set max_concurrent to at least 1");
                }
                ValidationResult::pass("max_concurrent_range")
            },
        ));

        // -- max_concurrent high warning --
        rules.push(ValidationRule::new(
            "max_concurrent_high_warning",
            "Very high max_concurrent may exhaust resources",
            Severity::Warning,
            |cfg| {
                let Some(raw) = cfg.get("max_concurrent") else {
                    return ValidationResult::pass("max_concurrent_high_warning");
                };
                let Ok(n) = raw.parse::<i64>() else {
                    return ValidationResult::pass("max_concurrent_high_warning");
                };
                if n > 256 {
                    return ValidationResult::fail(
                        "max_concurrent_high_warning",
                        Severity::Warning,
                        format!("max_concurrent={n} is very high"),
                    )
                    .with_suggestion("Consider limiting to available CPU/GPU resources");
                }
                ValidationResult::pass("max_concurrent_high_warning")
            },
        ));

        // -- timeout_secs --
        rules.push(ValidationRule::new(
            "timeout_secs_range",
            "timeout_secs must be > 0",
            Severity::Error,
            |cfg| {
                let Some(raw) = cfg.get("timeout_secs") else {
                    return ValidationResult::pass("timeout_secs_range");
                };
                let Ok(t) = raw.parse::<f64>() else {
                    return ValidationResult::fail(
                        "timeout_secs_range",
                        Severity::Error,
                        format!("timeout_secs is not a valid number: {raw}"),
                    );
                };
                if t <= 0.0 {
                    return ValidationResult::fail(
                        "timeout_secs_range",
                        Severity::Error,
                        format!("timeout_secs must be > 0, got {t}"),
                    )
                    .with_suggestion("Set timeout_secs to at least 1");
                }
                ValidationResult::pass("timeout_secs_range")
            },
        ));

        Self { rules }
    }
}

impl Default for ServerConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl Validator for ServerConfigValidator {
    fn validate(&self, config: &ConfigMap) -> ValidationReport {
        let mut report = ValidationReport::new();
        for rule in &self.rules {
            report.push(rule.evaluate(config));
        }
        report
    }

    fn name(&self) -> &'static str {
        "ServerConfigValidator"
    }
}

// ---------------------------------------------------------------------------
// CompositeValidator
// ---------------------------------------------------------------------------

/// Chains multiple validators and collects all results into a single report.
pub struct CompositeValidator {
    validators: Vec<Box<dyn Validator>>,
}

impl CompositeValidator {
    #[must_use]
    pub fn new() -> Self {
        Self { validators: Vec::new() }
    }

    pub fn add(&mut self, validator: impl Validator + 'static) {
        self.validators.push(Box::new(validator));
    }

    #[must_use]
    pub fn validator_count(&self) -> usize {
        self.validators.len()
    }
}

impl Default for CompositeValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl Validator for CompositeValidator {
    fn validate(&self, config: &ConfigMap) -> ValidationReport {
        let mut report = ValidationReport::new();
        for v in &self.validators {
            report.merge(&v.validate(config));
        }
        report
    }

    fn name(&self) -> &'static str {
        "CompositeValidator"
    }
}

// ---------------------------------------------------------------------------
// ConfigSuggester
// ---------------------------------------------------------------------------

/// Suggests optimal configuration values based on hardware capabilities.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub total_memory_mb: u64,
    pub gpu_memory_mb: Option<u64>,
    pub cpu_cores: u32,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_cuda: bool,
}

/// Suggested configuration based on hardware probing.
#[derive(Debug, Clone)]
pub struct ConfigSuggestion {
    pub key: String,
    pub value: String,
    pub reason: String,
}

/// Generates tuned configuration suggestions from a hardware profile.
pub struct ConfigSuggester;

impl ConfigSuggester {
    /// Produce a set of suggestions given a hardware profile.
    #[must_use]
    pub fn suggest(profile: &HardwareProfile) -> Vec<ConfigSuggestion> {
        let mut suggestions = Vec::new();

        // -- device --
        if profile.has_cuda {
            suggestions.push(ConfigSuggestion {
                key: "device".into(),
                value: "cuda".into(),
                reason: "CUDA GPU detected; GPU inference is fastest".into(),
            });
        } else {
            suggestions.push(ConfigSuggestion {
                key: "device".into(),
                value: "cpu".into(),
                reason: "No GPU detected; using CPU backend".into(),
            });
        }

        // -- threads --
        let threads = profile.cpu_cores.max(1);
        suggestions.push(ConfigSuggestion {
            key: "threads".into(),
            value: threads.to_string(),
            reason: format!("Matched to {threads} available CPU cores"),
        });

        // -- batch_size --
        let batch_size = if profile.gpu_memory_mb.unwrap_or(0) >= 8192 {
            32
        } else if profile.gpu_memory_mb.unwrap_or(0) >= 4096 {
            16
        } else if profile.gpu_memory_mb.unwrap_or(0) >= 2048 {
            8
        } else {
            1
        };
        suggestions.push(ConfigSuggestion {
            key: "batch_size".into(),
            value: batch_size.to_string(),
            reason: format!("Tuned for {} MB GPU memory", profile.gpu_memory_mb.unwrap_or(0)),
        });

        // -- context_length --
        let context_length = if profile.total_memory_mb >= 32768 {
            8192
        } else if profile.total_memory_mb >= 16384 {
            4096
        } else {
            2048
        };
        suggestions.push(ConfigSuggestion {
            key: "context_length".into(),
            value: context_length.to_string(),
            reason: format!("Tuned for {} MB total system memory", profile.total_memory_mb),
        });

        // -- SIMD hint --
        if profile.has_avx512 {
            suggestions.push(ConfigSuggestion {
                key: "simd_level".into(),
                value: "avx512".into(),
                reason: "AVX-512 detected; widest SIMD available".into(),
            });
        } else if profile.has_avx2 {
            suggestions.push(ConfigSuggestion {
                key: "simd_level".into(),
                value: "avx2".into(),
                reason: "AVX2 detected".into(),
            });
        }

        suggestions
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn cfg(pairs: &[(&str, &str)]) -> ConfigMap {
        pairs.iter().map(|(k, v)| ((*k).to_string(), (*v).to_string())).collect()
    }

    // -----------------------------------------------------------------------
    // Severity
    // -----------------------------------------------------------------------

    #[test]
    fn severity_ordering() {
        assert!(Severity::Hint < Severity::Info);
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
    }

    #[test]
    fn severity_display() {
        assert_eq!(Severity::Hint.to_string(), "hint");
        assert_eq!(Severity::Info.to_string(), "info");
        assert_eq!(Severity::Warning.to_string(), "warning");
        assert_eq!(Severity::Error.to_string(), "error");
    }

    #[test]
    fn severity_equality_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Severity::Error);
        set.insert(Severity::Error);
        assert_eq!(set.len(), 1);
    }

    // -----------------------------------------------------------------------
    // ValidationResult
    // -----------------------------------------------------------------------

    #[test]
    fn result_pass_defaults() {
        let r = ValidationResult::pass("test");
        assert!(r.passed);
        assert_eq!(r.severity, Severity::Info);
        assert!(r.message.is_empty());
        assert!(r.suggestion.is_none());
    }

    #[test]
    fn result_fail_with_suggestion() {
        let r = ValidationResult::fail("rule", Severity::Error, "bad").with_suggestion("fix it");
        assert!(!r.passed);
        assert_eq!(r.severity, Severity::Error);
        assert_eq!(r.message, "bad");
        assert_eq!(r.suggestion.as_deref(), Some("fix it"));
    }

    // -----------------------------------------------------------------------
    // ValidationReport
    // -----------------------------------------------------------------------

    #[test]
    fn report_empty_is_valid() {
        let r = ValidationReport::new();
        assert!(r.is_valid());
        assert_eq!(r.error_count(), 0);
        assert_eq!(r.warning_count(), 0);
    }

    #[test]
    fn report_counts_errors_and_warnings() {
        let mut r = ValidationReport::new();
        r.push(ValidationResult::fail("a", Severity::Error, "err"));
        r.push(ValidationResult::fail("b", Severity::Warning, "warn"));
        r.push(ValidationResult::pass("c"));
        assert_eq!(r.error_count(), 1);
        assert_eq!(r.warning_count(), 1);
        assert!(!r.is_valid());
    }

    #[test]
    fn report_warnings_only_is_valid() {
        let mut r = ValidationReport::new();
        r.push(ValidationResult::fail("a", Severity::Warning, "warn"));
        assert!(r.is_valid());
        assert_eq!(r.warning_count(), 1);
    }

    #[test]
    fn report_merge_combines_results() {
        let mut r1 = ValidationReport::new();
        r1.push(ValidationResult::fail("a", Severity::Error, "e1"));
        let mut r2 = ValidationReport::new();
        r2.push(ValidationResult::fail("b", Severity::Error, "e2"));
        r1.merge(&r2);
        assert_eq!(r1.error_count(), 2);
        assert_eq!(r1.results.len(), 2);
    }

    #[test]
    fn report_filter_by_severity() {
        let mut r = ValidationReport::new();
        r.push(ValidationResult::fail("a", Severity::Hint, "h"));
        r.push(ValidationResult::fail("b", Severity::Warning, "w"));
        r.push(ValidationResult::fail("c", Severity::Error, "e"));
        let errors = r.filter_by_severity(Severity::Error);
        assert_eq!(errors.len(), 1);
        let warnings_up = r.filter_by_severity(Severity::Warning);
        assert_eq!(warnings_up.len(), 2);
    }

    #[test]
    fn report_failed_results() {
        let mut r = ValidationReport::new();
        r.push(ValidationResult::pass("ok"));
        r.push(ValidationResult::fail("bad", Severity::Error, "e"));
        assert_eq!(r.failed_results().len(), 1);
    }

    #[test]
    fn report_default_trait() {
        let r = ValidationReport::default();
        assert!(r.is_valid());
    }

    // -----------------------------------------------------------------------
    // ValidationRule
    // -----------------------------------------------------------------------

    #[test]
    fn rule_evaluate_pass() {
        let rule = ValidationRule::new("always_ok", "passes", Severity::Error, |_| {
            ValidationResult::pass("always_ok")
        });
        let result = rule.evaluate(&HashMap::new());
        assert!(result.passed);
    }

    #[test]
    fn rule_evaluate_fail() {
        let rule = ValidationRule::new("always_fail", "fails", Severity::Error, |_| {
            ValidationResult::fail("always_fail", Severity::Error, "nope")
        });
        let result = rule.evaluate(&HashMap::new());
        assert!(!result.passed);
    }

    #[test]
    fn rule_debug_format() {
        let rule = ValidationRule::new("r", "d", Severity::Info, |_| ValidationResult::pass("r"));
        let dbg = format!("{rule:?}");
        assert!(dbg.contains("ValidationRule"));
        assert!(dbg.contains("\"r\""));
    }

    // -----------------------------------------------------------------------
    // InferenceConfigValidator — temperature
    // -----------------------------------------------------------------------

    #[test]
    fn temperature_valid_zero() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("temperature", "0.0")]));
        assert!(report.is_valid());
    }

    #[test]
    fn temperature_valid_one() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("temperature", "1.0")]));
        assert!(report.is_valid());
    }

    #[test]
    fn temperature_valid_two() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("temperature", "2.0")]));
        assert!(report.is_valid());
    }

    #[test]
    fn temperature_negative_is_error() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("temperature", "-0.5")]));
        assert!(!report.is_valid());
        let errs = report.filter_by_severity(Severity::Error);
        assert!(errs.iter().any(|r| r.rule_name == "temperature_range"));
    }

    #[test]
    fn temperature_above_two_is_error() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("temperature", "2.5")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn temperature_high_warning_at_1_8() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("temperature", "1.8")]));
        let warnings = report.filter_by_severity(Severity::Warning);
        assert!(warnings.iter().any(|r| r.rule_name == "temperature_high_warning"));
    }

    #[test]
    fn temperature_no_warning_at_1_0() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("temperature", "1.0")]));
        let warnings = report.filter_by_severity(Severity::Warning);
        assert!(!warnings.iter().any(|r| r.rule_name == "temperature_high_warning"));
    }

    #[test]
    fn temperature_invalid_string() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("temperature", "hot")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn temperature_missing_is_ok() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&HashMap::new());
        assert!(report.is_valid());
    }

    // -----------------------------------------------------------------------
    // InferenceConfigValidator — top_k
    // -----------------------------------------------------------------------

    #[test]
    fn top_k_zero_disabled_valid() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("top_k", "0")]));
        assert!(report.is_valid());
    }

    #[test]
    fn top_k_positive_valid() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("top_k", "50")]));
        assert!(report.is_valid());
    }

    #[test]
    fn top_k_negative_is_error() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("top_k", "-1")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn top_k_large_warning() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("top_k", "5000")]));
        let warnings = report.filter_by_severity(Severity::Warning);
        assert!(warnings.iter().any(|r| r.rule_name == "top_k_large_warning"));
    }

    // -----------------------------------------------------------------------
    // InferenceConfigValidator — top_p
    // -----------------------------------------------------------------------

    #[test]
    fn top_p_valid_one() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("top_p", "1.0")]));
        assert!(report.is_valid());
    }

    #[test]
    fn top_p_valid_mid() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("top_p", "0.9")]));
        assert!(report.is_valid());
    }

    #[test]
    fn top_p_zero_is_error() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("top_p", "0.0")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn top_p_negative_is_error() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("top_p", "-0.1")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn top_p_above_one_is_error() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("top_p", "1.1")]));
        assert!(!report.is_valid());
    }

    // -----------------------------------------------------------------------
    // InferenceConfigValidator — max_tokens
    // -----------------------------------------------------------------------

    #[test]
    fn max_tokens_valid() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("max_tokens", "128")]));
        assert!(report.is_valid());
    }

    #[test]
    fn max_tokens_one_valid() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("max_tokens", "1")]));
        assert!(report.is_valid());
    }

    #[test]
    fn max_tokens_zero_is_error() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("max_tokens", "0")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn max_tokens_negative_is_error() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("max_tokens", "-10")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn max_tokens_large_warning() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("max_tokens", "100000")]));
        let warnings = report.filter_by_severity(Severity::Warning);
        assert!(warnings.iter().any(|r| r.rule_name == "max_tokens_large_warning"));
    }

    // -----------------------------------------------------------------------
    // InferenceConfigValidator — repetition_penalty
    // -----------------------------------------------------------------------

    #[test]
    fn repetition_penalty_valid() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("repetition_penalty", "1.1")]));
        assert!(report.is_valid());
    }

    #[test]
    fn repetition_penalty_zero_is_error() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("repetition_penalty", "0.0")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn repetition_penalty_negative_is_error() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("repetition_penalty", "-1.0")]));
        assert!(!report.is_valid());
    }

    // -----------------------------------------------------------------------
    // InferenceConfigValidator — seed
    // -----------------------------------------------------------------------

    #[test]
    fn seed_valid() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("seed", "42")]));
        assert!(report.is_valid());
    }

    #[test]
    fn seed_zero_valid() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("seed", "0")]));
        assert!(report.is_valid());
    }

    #[test]
    fn seed_negative_is_error() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&cfg(&[("seed", "-1")]));
        assert!(!report.is_valid());
    }

    // -----------------------------------------------------------------------
    // InferenceConfigValidator — empty / all defaults
    // -----------------------------------------------------------------------

    #[test]
    fn inference_empty_config_is_valid() {
        let v = InferenceConfigValidator::new();
        let report = v.validate(&HashMap::new());
        assert!(report.is_valid());
    }

    #[test]
    fn inference_all_valid_params() {
        let v = InferenceConfigValidator::new();
        let c = cfg(&[
            ("temperature", "0.7"),
            ("top_k", "50"),
            ("top_p", "0.9"),
            ("max_tokens", "256"),
            ("repetition_penalty", "1.1"),
            ("seed", "42"),
        ]);
        let report = v.validate(&c);
        assert!(report.is_valid());
        assert_eq!(report.warning_count(), 0);
    }

    #[test]
    fn inference_multiple_errors() {
        let v = InferenceConfigValidator::new();
        let c =
            cfg(&[("temperature", "-1.0"), ("top_k", "-5"), ("top_p", "0.0"), ("max_tokens", "0")]);
        let report = v.validate(&c);
        assert!(!report.is_valid());
        assert!(report.error_count() >= 4);
    }

    #[test]
    fn inference_default_trait() {
        let v = InferenceConfigValidator::default();
        assert_eq!(v.name(), "InferenceConfigValidator");
    }

    // -----------------------------------------------------------------------
    // ModelConfigValidator
    // -----------------------------------------------------------------------

    #[test]
    fn model_path_missing_is_error() {
        let v = ModelConfigValidator::new();
        let report = v.validate(&HashMap::new());
        assert!(!report.is_valid());
    }

    #[test]
    fn model_path_empty_is_error() {
        let v = ModelConfigValidator::new();
        let report = v.validate(&cfg(&[("model_path", "")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn model_path_nonexistent_is_error() {
        let v = ModelConfigValidator::new();
        let report = v.validate(&cfg(&[("model_path", "/nonexistent/model.gguf")]));
        let errs = report.filter_by_severity(Severity::Error);
        assert!(errs.iter().any(|r| r.rule_name == "model_path_exists"));
    }

    #[test]
    fn model_format_gguf_valid() {
        let v = ModelConfigValidator::new();
        let c = cfg(&[("model_path", "."), ("model_format", "gguf")]);
        let report = v.validate(&c);
        assert!(
            !report
                .filter_by_severity(Severity::Error)
                .into_iter()
                .any(|r| r.rule_name == "model_format_supported")
        );
    }

    #[test]
    fn model_format_safetensors_valid() {
        let v = ModelConfigValidator::new();
        let c = cfg(&[("model_path", "."), ("model_format", "safetensors")]);
        let report = v.validate(&c);
        assert!(
            !report
                .filter_by_severity(Severity::Error)
                .into_iter()
                .any(|r| r.rule_name == "model_format_supported")
        );
    }

    #[test]
    fn model_format_unknown_is_error() {
        let v = ModelConfigValidator::new();
        let c = cfg(&[("model_path", "."), ("model_format", "pickle")]);
        let report = v.validate(&c);
        assert_eq!(
            report
                .filter_by_severity(Severity::Error)
                .into_iter()
                .filter(|r| r.rule_name == "model_format_supported")
                .count(),
            1
        );
    }

    #[test]
    fn model_context_length_zero_is_error() {
        let v = ModelConfigValidator::new();
        let c = cfg(&[("model_path", "."), ("context_length", "0")]);
        let report = v.validate(&c);
        assert_eq!(
            report
                .failed_results()
                .into_iter()
                .filter(|r| r.rule_name == "context_length_range")
                .count(),
            1
        );
    }

    #[test]
    fn model_context_length_large_warning() {
        let v = ModelConfigValidator::new();
        let c = cfg(&[("model_path", "."), ("context_length", "262144")]);
        let report = v.validate(&c);
        let warnings = report.filter_by_severity(Severity::Warning);
        assert!(warnings.iter().any(|r| r.rule_name == "context_length_range"));
    }

    #[test]
    fn model_default_trait() {
        let v = ModelConfigValidator::default();
        assert_eq!(v.name(), "ModelConfigValidator");
    }

    // -----------------------------------------------------------------------
    // DeviceConfigValidator
    // -----------------------------------------------------------------------

    #[test]
    fn device_cpu_valid() {
        let v = DeviceConfigValidator::new();
        let report = v.validate(&cfg(&[("device", "cpu")]));
        assert!(report.is_valid());
    }

    #[test]
    fn device_cuda_valid() {
        let v = DeviceConfigValidator::new();
        let report = v.validate(&cfg(&[("device", "cuda")]));
        assert!(report.is_valid());
    }

    #[test]
    fn device_unknown_is_error() {
        let v = DeviceConfigValidator::new();
        let report = v.validate(&cfg(&[("device", "tpu")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn device_case_insensitive() {
        let v = DeviceConfigValidator::new();
        let report = v.validate(&cfg(&[("device", "CUDA")]));
        assert!(report.is_valid());
    }

    #[test]
    fn device_missing_is_ok() {
        let v = DeviceConfigValidator::new();
        let report = v.validate(&HashMap::new());
        assert!(report.is_valid());
    }

    #[test]
    fn gpu_memory_sufficient() {
        let v = DeviceConfigValidator::new();
        let report = v.validate(&cfg(&[("gpu_memory_mb", "1024")]));
        assert!(report.is_valid());
    }

    #[test]
    fn gpu_memory_low_warning() {
        let v = DeviceConfigValidator::new();
        let report = v.validate(&cfg(&[("gpu_memory_mb", "256")]));
        let warnings = report.filter_by_severity(Severity::Warning);
        assert!(warnings.iter().any(|r| r.rule_name == "gpu_memory_sufficient"));
    }

    #[test]
    fn device_index_valid() {
        let v = DeviceConfigValidator::new();
        let report = v.validate(&cfg(&[("device_index", "0")]));
        assert!(report.is_valid());
    }

    #[test]
    fn device_index_negative_is_error() {
        let v = DeviceConfigValidator::new();
        let report = v.validate(&cfg(&[("device_index", "-1")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn compute_capability_valid() {
        let v = DeviceConfigValidator::new();
        let report = v.validate(&cfg(&[("compute_capability", "8.6")]));
        assert!(report.is_valid());
    }

    #[test]
    fn compute_capability_low_warning() {
        let v = DeviceConfigValidator::new();
        let report = v.validate(&cfg(&[("compute_capability", "5.0")]));
        let warnings = report.filter_by_severity(Severity::Warning);
        assert!(warnings.iter().any(|r| r.rule_name == "compute_capability_min"));
    }

    #[test]
    fn threads_valid() {
        let v = DeviceConfigValidator::new();
        let report = v.validate(&cfg(&[("threads", "4")]));
        assert!(report.is_valid());
    }

    #[test]
    fn threads_zero_is_error() {
        let v = DeviceConfigValidator::new();
        let report = v.validate(&cfg(&[("threads", "0")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn device_default_trait() {
        let v = DeviceConfigValidator::default();
        assert_eq!(v.name(), "DeviceConfigValidator");
    }

    // -----------------------------------------------------------------------
    // ServerConfigValidator — port
    // -----------------------------------------------------------------------

    #[test]
    fn port_valid() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("port", "8080")]));
        assert!(report.is_valid());
    }

    #[test]
    fn port_one_valid() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("port", "1")]));
        // Valid port but will get privileged warning
        let errs = report.filter_by_severity(Severity::Error);
        assert!(errs.is_empty());
    }

    #[test]
    fn port_65535_valid() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("port", "65535")]));
        assert!(report.is_valid());
    }

    #[test]
    fn port_zero_is_error() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("port", "0")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn port_negative_is_error() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("port", "-1")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn port_too_large_is_error() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("port", "70000")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn port_privileged_warning() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("port", "80")]));
        let warnings = report.filter_by_severity(Severity::Warning);
        assert!(warnings.iter().any(|r| r.rule_name == "port_privileged_warning"));
    }

    #[test]
    fn port_1024_no_privileged_warning() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("port", "1024")]));
        let warnings = report.filter_by_severity(Severity::Warning);
        assert!(!warnings.iter().any(|r| r.rule_name == "port_privileged_warning"));
    }

    // -----------------------------------------------------------------------
    // ServerConfigValidator — bind_address
    // -----------------------------------------------------------------------

    #[test]
    fn bind_address_ip_valid() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("bind_address", "0.0.0.0")]));
        assert!(report.is_valid());
    }

    #[test]
    fn bind_address_localhost_valid() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("bind_address", "127.0.0.1")]));
        assert!(report.is_valid());
    }

    #[test]
    fn bind_address_hostname_valid() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("bind_address", "my-host.local")]));
        assert!(report.is_valid());
    }

    #[test]
    fn bind_address_empty_is_error() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("bind_address", "")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn bind_address_invalid_chars_is_error() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("bind_address", "host name!@#")]));
        assert!(!report.is_valid());
    }

    // -----------------------------------------------------------------------
    // ServerConfigValidator — max_concurrent
    // -----------------------------------------------------------------------

    #[test]
    fn max_concurrent_valid() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("max_concurrent", "8")]));
        assert!(report.is_valid());
    }

    #[test]
    fn max_concurrent_zero_is_error() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("max_concurrent", "0")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn max_concurrent_high_warning() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("max_concurrent", "512")]));
        let warnings = report.filter_by_severity(Severity::Warning);
        assert!(warnings.iter().any(|r| r.rule_name == "max_concurrent_high_warning"));
    }

    // -----------------------------------------------------------------------
    // ServerConfigValidator — timeout_secs
    // -----------------------------------------------------------------------

    #[test]
    fn timeout_secs_valid() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("timeout_secs", "30")]));
        assert!(report.is_valid());
    }

    #[test]
    fn timeout_secs_zero_is_error() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("timeout_secs", "0")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn timeout_secs_negative_is_error() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&cfg(&[("timeout_secs", "-5")]));
        assert!(!report.is_valid());
    }

    #[test]
    fn server_empty_config_valid() {
        let v = ServerConfigValidator::new();
        let report = v.validate(&HashMap::new());
        assert!(report.is_valid());
    }

    #[test]
    fn server_default_trait() {
        let v = ServerConfigValidator::default();
        assert_eq!(v.name(), "ServerConfigValidator");
    }

    // -----------------------------------------------------------------------
    // CompositeValidator
    // -----------------------------------------------------------------------

    #[test]
    fn composite_empty_is_valid() {
        let cv = CompositeValidator::new();
        let report = cv.validate(&HashMap::new());
        assert!(report.is_valid());
    }

    #[test]
    fn composite_aggregates_results() {
        let mut cv = CompositeValidator::new();
        cv.add(InferenceConfigValidator::new());
        cv.add(ServerConfigValidator::new());
        assert_eq!(cv.validator_count(), 2);

        let c = cfg(&[("temperature", "-1.0"), ("port", "0")]);
        let report = cv.validate(&c);
        assert!(!report.is_valid());
        assert!(report.error_count() >= 2);
    }

    #[test]
    fn composite_all_passing() {
        let mut cv = CompositeValidator::new();
        cv.add(InferenceConfigValidator::new());
        cv.add(ServerConfigValidator::new());
        cv.add(DeviceConfigValidator::new());

        let c = cfg(&[
            ("temperature", "0.7"),
            ("top_k", "50"),
            ("top_p", "0.9"),
            ("max_tokens", "128"),
            ("port", "8080"),
            ("device", "cpu"),
        ]);
        let report = cv.validate(&c);
        assert!(report.is_valid());
        assert_eq!(report.warning_count(), 0);
    }

    #[test]
    fn composite_default_trait() {
        let cv = CompositeValidator::default();
        assert_eq!(cv.name(), "CompositeValidator");
    }

    #[test]
    fn composite_name() {
        let cv = CompositeValidator::new();
        assert_eq!(cv.name(), "CompositeValidator");
    }

    // -----------------------------------------------------------------------
    // ConfigSuggester
    // -----------------------------------------------------------------------

    #[test]
    fn suggest_cpu_no_gpu() {
        let profile = HardwareProfile {
            total_memory_mb: 16384,
            gpu_memory_mb: None,
            cpu_cores: 8,
            has_avx2: true,
            has_avx512: false,
            has_cuda: false,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        let device = suggestions.iter().find(|s| s.key == "device").unwrap();
        assert_eq!(device.value, "cpu");
    }

    #[test]
    fn suggest_cuda_when_available() {
        let profile = HardwareProfile {
            total_memory_mb: 32768,
            gpu_memory_mb: Some(8192),
            cpu_cores: 16,
            has_avx2: true,
            has_avx512: true,
            has_cuda: true,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        let device = suggestions.iter().find(|s| s.key == "device").unwrap();
        assert_eq!(device.value, "cuda");
    }

    #[test]
    fn suggest_threads_match_cores() {
        let profile = HardwareProfile {
            total_memory_mb: 8192,
            gpu_memory_mb: None,
            cpu_cores: 4,
            has_avx2: false,
            has_avx512: false,
            has_cuda: false,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        let threads = suggestions.iter().find(|s| s.key == "threads").unwrap();
        assert_eq!(threads.value, "4");
    }

    #[test]
    fn suggest_batch_size_large_gpu() {
        let profile = HardwareProfile {
            total_memory_mb: 65536,
            gpu_memory_mb: Some(16384),
            cpu_cores: 32,
            has_avx2: true,
            has_avx512: true,
            has_cuda: true,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        let batch = suggestions.iter().find(|s| s.key == "batch_size").unwrap();
        assert_eq!(batch.value, "32");
    }

    #[test]
    fn suggest_batch_size_small_gpu() {
        let profile = HardwareProfile {
            total_memory_mb: 16384,
            gpu_memory_mb: Some(2048),
            cpu_cores: 8,
            has_avx2: true,
            has_avx512: false,
            has_cuda: true,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        let batch = suggestions.iter().find(|s| s.key == "batch_size").unwrap();
        assert_eq!(batch.value, "8");
    }

    #[test]
    fn suggest_batch_size_no_gpu() {
        let profile = HardwareProfile {
            total_memory_mb: 8192,
            gpu_memory_mb: None,
            cpu_cores: 4,
            has_avx2: false,
            has_avx512: false,
            has_cuda: false,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        let batch = suggestions.iter().find(|s| s.key == "batch_size").unwrap();
        assert_eq!(batch.value, "1");
    }

    #[test]
    fn suggest_context_length_high_memory() {
        let profile = HardwareProfile {
            total_memory_mb: 65536,
            gpu_memory_mb: None,
            cpu_cores: 16,
            has_avx2: true,
            has_avx512: false,
            has_cuda: false,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        let ctx = suggestions.iter().find(|s| s.key == "context_length").unwrap();
        assert_eq!(ctx.value, "8192");
    }

    #[test]
    fn suggest_context_length_low_memory() {
        let profile = HardwareProfile {
            total_memory_mb: 4096,
            gpu_memory_mb: None,
            cpu_cores: 2,
            has_avx2: false,
            has_avx512: false,
            has_cuda: false,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        let ctx = suggestions.iter().find(|s| s.key == "context_length").unwrap();
        assert_eq!(ctx.value, "2048");
    }

    #[test]
    fn suggest_avx512_simd() {
        let profile = HardwareProfile {
            total_memory_mb: 32768,
            gpu_memory_mb: None,
            cpu_cores: 16,
            has_avx2: true,
            has_avx512: true,
            has_cuda: false,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        let simd = suggestions.iter().find(|s| s.key == "simd_level").unwrap();
        assert_eq!(simd.value, "avx512");
    }

    #[test]
    fn suggest_avx2_simd() {
        let profile = HardwareProfile {
            total_memory_mb: 16384,
            gpu_memory_mb: None,
            cpu_cores: 8,
            has_avx2: true,
            has_avx512: false,
            has_cuda: false,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        let simd = suggestions.iter().find(|s| s.key == "simd_level").unwrap();
        assert_eq!(simd.value, "avx2");
    }

    #[test]
    fn suggest_no_simd_hint_without_avx() {
        let profile = HardwareProfile {
            total_memory_mb: 8192,
            gpu_memory_mb: None,
            cpu_cores: 4,
            has_avx2: false,
            has_avx512: false,
            has_cuda: false,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        assert!(suggestions.iter().all(|s| s.key != "simd_level"));
    }

    // -----------------------------------------------------------------------
    // Validation idempotency
    // -----------------------------------------------------------------------

    #[test]
    fn validation_is_idempotent() {
        let v = InferenceConfigValidator::new();
        let c = cfg(&[
            ("temperature", "0.7"),
            ("top_k", "50"),
            ("top_p", "0.9"),
            ("max_tokens", "128"),
        ]);
        let r1 = v.validate(&c);
        let r2 = v.validate(&c);
        assert_eq!(r1.is_valid(), r2.is_valid());
        assert_eq!(r1.error_count(), r2.error_count());
        assert_eq!(r1.warning_count(), r2.warning_count());
        assert_eq!(r1.results.len(), r2.results.len());
        for (a, b) in r1.results.iter().zip(r2.results.iter()) {
            assert_eq!(a.rule_name, b.rule_name);
            assert_eq!(a.passed, b.passed);
            assert_eq!(a.severity, b.severity);
            assert_eq!(a.message, b.message);
        }
    }

    #[test]
    fn validation_idempotent_with_errors() {
        let v = InferenceConfigValidator::new();
        let c = cfg(&[("temperature", "-5.0"), ("top_k", "-1")]);
        let r1 = v.validate(&c);
        let r2 = v.validate(&c);
        assert_eq!(r1.error_count(), r2.error_count());
        assert_eq!(r1.results.len(), r2.results.len());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn suggestion_has_reason() {
        let profile = HardwareProfile {
            total_memory_mb: 16384,
            gpu_memory_mb: Some(4096),
            cpu_cores: 8,
            has_avx2: true,
            has_avx512: false,
            has_cuda: true,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        for s in &suggestions {
            assert!(!s.reason.is_empty(), "suggestion for {} has empty reason", s.key);
        }
    }

    #[test]
    fn all_failed_results_have_messages() {
        let v = InferenceConfigValidator::new();
        let c = cfg(&[("temperature", "-1.0"), ("top_k", "-5"), ("top_p", "0.0")]);
        let report = v.validate(&c);
        for r in report.failed_results() {
            assert!(!r.message.is_empty(), "rule {} failed without message", r.rule_name);
        }
    }

    #[test]
    fn composite_with_model_and_inference() {
        let mut cv = CompositeValidator::new();
        cv.add(InferenceConfigValidator::new());
        cv.add(ModelConfigValidator::new());
        let c = cfg(&[("temperature", "0.7"), ("model_path", ".")]);
        let report = cv.validate(&c);
        // model_path "." exists (current dir), temperature is valid
        assert!(report.is_valid());
    }

    #[test]
    fn batch_size_mid_gpu() {
        let profile = HardwareProfile {
            total_memory_mb: 32768,
            gpu_memory_mb: Some(4096),
            cpu_cores: 8,
            has_avx2: true,
            has_avx512: false,
            has_cuda: true,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        let batch = suggestions.iter().find(|s| s.key == "batch_size").unwrap();
        assert_eq!(batch.value, "16");
    }

    #[test]
    fn context_length_mid_memory() {
        let profile = HardwareProfile {
            total_memory_mb: 16384,
            gpu_memory_mb: None,
            cpu_cores: 8,
            has_avx2: false,
            has_avx512: false,
            has_cuda: false,
        };
        let suggestions = ConfigSuggester::suggest(&profile);
        let ctx = suggestions.iter().find(|s| s.key == "context_length").unwrap();
        assert_eq!(ctx.value, "4096");
    }
}
