//! Inference configuration builder with comprehensive validation.

use serde::Deserialize;
use std::env;
use std::fmt;

// ── RopeScaling ──────────────────────────────────────────────────────────

/// `RoPE` scaling strategy for extended context lengths.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    /// Linear interpolation with the given factor.
    Linear(f32),
    /// `NTK`-aware scaling with the given factor.
    Ntk(f32),
    /// `YaRN` scaling with factor and beta.
    Yarn { factor: f32, beta: f32 },
}

// ── BackendChoice ────────────────────────────────────────────────────────

/// Which compute backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendChoice {
    /// Automatically select the best available backend.
    #[default]
    Auto,
    /// CPU-only inference.
    Cpu,
    /// NVIDIA CUDA backend.
    Cuda,
    /// `OpenCL` backend.
    #[serde(rename = "opencl")]
    OpenCL,
    /// Vulkan backend.
    Vulkan,
    /// Apple Metal backend.
    Metal,
    /// AMD `ROCm` backend.
    Rocm,
    /// WebGPU backend.
    #[serde(rename = "webgpu")]
    WebGpu,
}

impl fmt::Display for BackendChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::OpenCL => write!(f, "opencl"),
            Self::Vulkan => write!(f, "vulkan"),
            Self::Metal => write!(f, "metal"),
            Self::Rocm => write!(f, "rocm"),
            Self::WebGpu => write!(f, "webgpu"),
        }
    }
}

impl BackendChoice {
    fn from_str_case_insensitive(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            "cuda" => Some(Self::Cuda),
            "opencl" => Some(Self::OpenCL),
            "vulkan" => Some(Self::Vulkan),
            "metal" => Some(Self::Metal),
            "rocm" => Some(Self::Rocm),
            "webgpu" => Some(Self::WebGpu),
            _ => None,
        }
    }
}

// ── Config sections ──────────────────────────────────────────────────────

/// Model-related configuration.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Path to the model file (required).
    pub model_path: String,
    /// Path to the tokenizer file.
    pub tokenizer_path: Option<String>,
    /// Model architecture name (e.g. "bitnet-b1.58").
    pub architecture: Option<String>,
    /// Maximum context length in tokens.
    pub context_length: usize,
    /// `RoPE` scaling strategy.
    pub rope_scaling: Option<RopeScaling>,
}

/// Generation/sampling configuration.
#[derive(Debug, Clone)]
pub struct GenConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature (0 = greedy, higher = more random).
    pub temperature: f32,
    /// Nucleus sampling threshold.
    pub top_p: f32,
    /// Top-K sampling.
    pub top_k: usize,
    /// Repetition penalty factor.
    pub repetition_penalty: f32,
    /// Frequency penalty factor.
    pub frequency_penalty: f32,
    /// Presence penalty factor.
    pub presence_penalty: f32,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Stop sequences.
    pub stop_sequences: Vec<String>,
}

/// Hardware/backend configuration.
#[derive(Debug, Clone)]
pub struct HardwareConfig {
    /// Compute backend to use.
    pub backend: BackendChoice,
    /// GPU device index.
    pub device_index: usize,
    /// Number of CPU threads.
    pub num_threads: usize,
    /// Number of layers to offload to GPU.
    pub gpu_layers: Option<usize>,
    /// Memory budget in bytes.
    pub memory_limit: Option<u64>,
    /// Whether to use flash attention.
    pub use_flash_attention: bool,
}

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Listen address.
    pub host: String,
    /// Listen port.
    pub port: u16,
    /// Maximum concurrent requests.
    pub max_concurrent_requests: usize,
    /// Per-request timeout in milliseconds.
    pub request_timeout_ms: u64,
    /// Whether to enable CORS.
    pub enable_cors: bool,
}

/// Advanced/experimental configuration.
#[derive(Debug, Clone)]
pub struct AdvancedConfig {
    /// Quantize the KV cache.
    pub kv_cache_quantization: bool,
    /// Enable speculative decoding.
    pub speculative_decoding: bool,
    /// Path to the draft model for speculative decoding.
    pub draft_model_path: Option<String>,
    /// Batch size for inference.
    pub batch_size: usize,
    /// Log level string (e.g. "info", "debug").
    pub log_level: String,
}

// ── InferenceConfig ──────────────────────────────────────────────────────

/// Top-level inference configuration.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub model: ModelConfig,
    pub generation: GenConfig,
    pub hardware: HardwareConfig,
    pub server: ServerConfig,
    pub advanced: AdvancedConfig,
}

impl InferenceConfig {
    /// Validate the entire configuration, returning errors and warnings.
    pub fn validate(&self) -> ConfigValidation {
        let mut v = ConfigValidation::new();

        // Model validation
        if self.model.model_path.is_empty() {
            v.error("model_path must not be empty");
        }
        if self.model.context_length == 0 {
            v.error("context_length must be > 0");
        }

        // Generation validation
        if self.generation.temperature < 0.0 || self.generation.temperature > 2.0
        {
            v.error("temperature must be in [0.0, 2.0]");
        }
        if self.generation.top_p <= 0.0 || self.generation.top_p > 1.0 {
            v.error("top_p must be in (0.0, 1.0]");
        }
        if self.generation.top_k == 0 {
            v.error("top_k must be > 0");
        }
        if self.generation.max_tokens == 0 {
            v.error("max_tokens must be > 0");
        }
        if self.generation.repetition_penalty < 0.0 {
            v.error("repetition_penalty must be >= 0.0");
        }
        if self.generation.frequency_penalty < 0.0 {
            v.error("frequency_penalty must be >= 0.0");
        }
        if self.generation.presence_penalty < 0.0 {
            v.error("presence_penalty must be >= 0.0");
        }

        // Warnings
        if self.generation.temperature > 1.5 {
            v.warning("temperature > 1.5 may produce incoherent output");
        }

        // Hardware validation
        if self.hardware.num_threads == 0 {
            v.error("num_threads must be > 0");
        }

        // Server validation
        if self.server.port == 0 {
            v.error("port must be in [1, 65535]");
        }
        if self.server.max_concurrent_requests == 0 {
            v.error("max_concurrent_requests must be > 0");
        }
        if self.server.request_timeout_ms == 0 {
            v.error("request_timeout_ms must be > 0");
        }

        // Advanced validation
        if self.advanced.batch_size == 0 {
            v.error("batch_size must be > 0");
        }
        if self.advanced.speculative_decoding
            && self.advanced.draft_model_path.is_none()
        {
            v.error(
                "draft_model_path is required when \
                 speculative_decoding is enabled",
            );
        }

        v
    }
}

// ── ConfigValidation ─────────────────────────────────────────────────────

/// Result of configuration validation.
#[derive(Debug, Clone, Default)]
pub struct ConfigValidation {
    /// Hard errors that prevent the config from being used.
    pub errors: Vec<String>,
    /// Soft warnings about potentially problematic values.
    pub warnings: Vec<String>,
}

impl ConfigValidation {
    fn new() -> Self {
        Self::default()
    }

    fn error(&mut self, msg: &str) {
        self.errors.push(msg.to_string());
    }

    fn warning(&mut self, msg: &str) {
        self.warnings.push(msg.to_string());
    }

    /// Returns `true` when no hard errors are present.
    #[allow(clippy::missing_const_for_fn)]
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns `true` when there are warnings.
    #[allow(clippy::missing_const_for_fn)]
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

impl fmt::Display for ConfigValidation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for e in &self.errors {
            writeln!(f, "ERROR: {e}")?;
        }
        for w in &self.warnings {
            writeln!(f, "WARN:  {w}")?;
        }
        Ok(())
    }
}

// ── ConfigBuilder ────────────────────────────────────────────────────────

/// Fluent builder for [`InferenceConfig`].
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ConfigBuilder {
    model_path: String,
    tokenizer_path: Option<String>,
    architecture: Option<String>,
    context_length: usize,
    rope_scaling: Option<RopeScaling>,

    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    seed: Option<u64>,
    stop_sequences: Vec<String>,

    backend: BackendChoice,
    device_index: usize,
    num_threads: usize,
    gpu_layers: Option<usize>,
    memory_limit: Option<u64>,
    use_flash_attention: bool,

    host: String,
    port: u16,
    max_concurrent_requests: usize,
    request_timeout_ms: u64,
    enable_cors: bool,

    kv_cache_quantization: bool,
    speculative_decoding: bool,
    draft_model_path: Option<String>,
    batch_size: usize,
    log_level: String,
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: None,
            architecture: None,
            context_length: 2048,
            rope_scaling: None,

            max_tokens: 256,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            stop_sequences: Vec::new(),

            backend: BackendChoice::Auto,
            device_index: 0,
            num_threads: 4,
            gpu_layers: None,
            memory_limit: None,
            use_flash_attention: false,

            host: "127.0.0.1".to_string(),
            port: 8080,
            max_concurrent_requests: 16,
            request_timeout_ms: 30_000,
            enable_cors: false,

            kv_cache_quantization: false,
            speculative_decoding: false,
            draft_model_path: None,
            batch_size: 1,
            log_level: "info".to_string(),
        }
    }
}

#[allow(
    clippy::return_self_not_must_use,
    clippy::missing_const_for_fn,
    clippy::collapsible_if
)]
impl ConfigBuilder {
    /// Create a new builder with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ── Model setters ────────────────────────────────────────────────

    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = path.into();
        self
    }

    pub fn tokenizer_path(mut self, path: impl Into<String>) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }

    pub fn architecture(mut self, arch: impl Into<String>) -> Self {
        self.architecture = Some(arch.into());
        self
    }

    pub fn context_length(mut self, len: usize) -> Self {
        self.context_length = len;
        self
    }

    pub fn rope_scaling(mut self, scaling: RopeScaling) -> Self {
        self.rope_scaling = Some(scaling);
        self
    }

    // ── Generation setters ───────────────────────────────────────────

    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn repetition_penalty(mut self, p: f32) -> Self {
        self.repetition_penalty = p;
        self
    }

    pub fn frequency_penalty(mut self, p: f32) -> Self {
        self.frequency_penalty = p;
        self
    }

    pub fn presence_penalty(mut self, p: f32) -> Self {
        self.presence_penalty = p;
        self
    }

    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    pub fn stop_sequences(mut self, seqs: Vec<String>) -> Self {
        self.stop_sequences = seqs;
        self
    }

    // ── Hardware setters ─────────────────────────────────────────────

    pub fn backend(mut self, b: BackendChoice) -> Self {
        self.backend = b;
        self
    }

    pub fn device_index(mut self, idx: usize) -> Self {
        self.device_index = idx;
        self
    }

    pub fn num_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    pub fn gpu_layers(mut self, n: usize) -> Self {
        self.gpu_layers = Some(n);
        self
    }

    pub fn memory_limit(mut self, bytes: u64) -> Self {
        self.memory_limit = Some(bytes);
        self
    }

    pub fn use_flash_attention(mut self, enable: bool) -> Self {
        self.use_flash_attention = enable;
        self
    }

    // ── Server setters ───────────────────────────────────────────────

    pub fn host(mut self, h: impl Into<String>) -> Self {
        self.host = h.into();
        self
    }

    pub fn port(mut self, p: u16) -> Self {
        self.port = p;
        self
    }

    pub fn max_concurrent_requests(mut self, n: usize) -> Self {
        self.max_concurrent_requests = n;
        self
    }

    pub fn request_timeout_ms(mut self, ms: u64) -> Self {
        self.request_timeout_ms = ms;
        self
    }

    pub fn enable_cors(mut self, enable: bool) -> Self {
        self.enable_cors = enable;
        self
    }

    // ── Advanced setters ─────────────────────────────────────────────

    pub fn kv_cache_quantization(mut self, enable: bool) -> Self {
        self.kv_cache_quantization = enable;
        self
    }

    pub fn speculative_decoding(mut self, enable: bool) -> Self {
        self.speculative_decoding = enable;
        self
    }

    pub fn draft_model_path(mut self, path: impl Into<String>) -> Self {
        self.draft_model_path = Some(path.into());
        self
    }

    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    pub fn log_level(mut self, level: impl Into<String>) -> Self {
        self.log_level = level.into();
        self
    }

    // ── Environment loading ──────────────────────────────────────────

    /// Populate builder fields from `BITNET_*` environment variables.
    ///
    /// Unset or unparseable variables are silently ignored, keeping
    /// existing builder defaults.
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn from_env() -> Self {
        let mut b = Self::new();

        if let Ok(v) = env::var("BITNET_MODEL_PATH") {
            b.model_path = v;
        }
        if let Ok(v) = env::var("BITNET_TOKENIZER_PATH") {
            b.tokenizer_path = Some(v);
        }
        if let Ok(v) = env::var("BITNET_CONTEXT_LENGTH") {
            if let Ok(n) = v.parse() {
                b.context_length = n;
            }
        }
        if let Ok(v) = env::var("BITNET_MAX_TOKENS") {
            if let Ok(n) = v.parse() {
                b.max_tokens = n;
            }
        }
        if let Ok(v) = env::var("BITNET_TEMPERATURE") {
            if let Ok(t) = v.parse() {
                b.temperature = t;
            }
        }
        if let Ok(v) = env::var("BITNET_TOP_P") {
            if let Ok(p) = v.parse() {
                b.top_p = p;
            }
        }
        if let Ok(v) = env::var("BITNET_TOP_K") {
            if let Ok(k) = v.parse() {
                b.top_k = k;
            }
        }
        if let Ok(v) = env::var("BITNET_SEED") {
            if let Ok(s) = v.parse() {
                b.seed = Some(s);
            }
        }
        if let Ok(v) = env::var("BITNET_BACKEND") {
            if let Some(backend) = BackendChoice::from_str_case_insensitive(&v)
            {
                b.backend = backend;
            }
        }
        if let Ok(v) = env::var("BITNET_NUM_THREADS") {
            if let Ok(n) = v.parse() {
                b.num_threads = n;
            }
        }
        if let Ok(v) = env::var("BITNET_PORT") {
            if let Ok(p) = v.parse() {
                b.port = p;
            }
        }
        if let Ok(v) = env::var("BITNET_HOST") {
            b.host = v;
        }
        if let Ok(v) = env::var("BITNET_LOG_LEVEL") {
            b.log_level = v;
        }
        if let Ok(v) = env::var("BITNET_BATCH_SIZE") {
            if let Ok(n) = v.parse() {
                b.batch_size = n;
            }
        }
        if let Ok(v) = env::var("BITNET_REPETITION_PENALTY") {
            if let Ok(p) = v.parse() {
                b.repetition_penalty = p;
            }
        }

        b
    }

    // ── TOML loading ─────────────────────────────────────────────────

    /// Parse a TOML string into a builder, layered on top of defaults.
    ///
    /// # Errors
    ///
    /// Returns an error string if the TOML is syntactically invalid.
    #[allow(clippy::too_many_lines)]
    pub fn from_toml(text: &str) -> Result<Self, String> {
        let doc: TomlConfig =
            toml::from_str(text).map_err(|e| format!("TOML parse error: {e}"))?;

        let mut b = Self::new();

        if let Some(m) = doc.model {
            if let Some(v) = m.model_path {
                b.model_path = v;
            }
            if let Some(v) = m.tokenizer_path {
                b.tokenizer_path = Some(v);
            }
            if let Some(v) = m.architecture {
                b.architecture = Some(v);
            }
            if let Some(v) = m.context_length {
                b.context_length = v;
            }
        }

        if let Some(g) = doc.generation {
            if let Some(v) = g.max_tokens {
                b.max_tokens = v;
            }
            if let Some(v) = g.temperature {
                b.temperature = v;
            }
            if let Some(v) = g.top_p {
                b.top_p = v;
            }
            if let Some(v) = g.top_k {
                b.top_k = v;
            }
            if let Some(v) = g.repetition_penalty {
                b.repetition_penalty = v;
            }
            if let Some(v) = g.frequency_penalty {
                b.frequency_penalty = v;
            }
            if let Some(v) = g.presence_penalty {
                b.presence_penalty = v;
            }
            if let Some(v) = g.seed {
                b.seed = Some(v);
            }
            if let Some(v) = g.stop_sequences {
                b.stop_sequences = v;
            }
        }

        if let Some(h) = doc.hardware {
            if let Some(v) = h.backend {
                b.backend = v;
            }
            if let Some(v) = h.device_index {
                b.device_index = v;
            }
            if let Some(v) = h.num_threads {
                b.num_threads = v;
            }
            if let Some(v) = h.gpu_layers {
                b.gpu_layers = Some(v);
            }
            if let Some(v) = h.memory_limit {
                b.memory_limit = Some(v);
            }
            if let Some(v) = h.use_flash_attention {
                b.use_flash_attention = v;
            }
        }

        if let Some(s) = doc.server {
            if let Some(v) = s.host {
                b.host = v;
            }
            if let Some(v) = s.port {
                b.port = v;
            }
            if let Some(v) = s.max_concurrent_requests {
                b.max_concurrent_requests = v;
            }
            if let Some(v) = s.request_timeout_ms {
                b.request_timeout_ms = v;
            }
            if let Some(v) = s.enable_cors {
                b.enable_cors = v;
            }
        }

        if let Some(a) = doc.advanced {
            if let Some(v) = a.kv_cache_quantization {
                b.kv_cache_quantization = v;
            }
            if let Some(v) = a.speculative_decoding {
                b.speculative_decoding = v;
            }
            if let Some(v) = a.draft_model_path {
                b.draft_model_path = Some(v);
            }
            if let Some(v) = a.batch_size {
                b.batch_size = v;
            }
            if let Some(v) = a.log_level {
                b.log_level = v;
            }
        }

        Ok(b)
    }

    // ── Build ────────────────────────────────────────────────────────

    /// Build the [`InferenceConfig`], running validation.
    ///
    /// Returns `Err(ConfigValidation)` when there are hard errors.
    pub fn build(self) -> Result<InferenceConfig, ConfigValidation> {
        let config = InferenceConfig {
            model: ModelConfig {
                model_path: self.model_path,
                tokenizer_path: self.tokenizer_path,
                architecture: self.architecture,
                context_length: self.context_length,
                rope_scaling: self.rope_scaling,
            },
            generation: GenConfig {
                max_tokens: self.max_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
                repetition_penalty: self.repetition_penalty,
                frequency_penalty: self.frequency_penalty,
                presence_penalty: self.presence_penalty,
                seed: self.seed,
                stop_sequences: self.stop_sequences,
            },
            hardware: HardwareConfig {
                backend: self.backend,
                device_index: self.device_index,
                num_threads: self.num_threads,
                gpu_layers: self.gpu_layers,
                memory_limit: self.memory_limit,
                use_flash_attention: self.use_flash_attention,
            },
            server: ServerConfig {
                host: self.host,
                port: self.port,
                max_concurrent_requests: self.max_concurrent_requests,
                request_timeout_ms: self.request_timeout_ms,
                enable_cors: self.enable_cors,
            },
            advanced: AdvancedConfig {
                kv_cache_quantization: self.kv_cache_quantization,
                speculative_decoding: self.speculative_decoding,
                draft_model_path: self.draft_model_path,
                batch_size: self.batch_size,
                log_level: self.log_level,
            },
        };

        let validation = config.validate();
        if validation.is_valid() {
            Ok(config)
        } else {
            Err(validation)
        }
    }
}

// ── TOML serde structs (private) ─────────────────────────────────────────

#[derive(Deserialize)]
struct TomlConfig {
    model: Option<TomlModel>,
    generation: Option<TomlGeneration>,
    hardware: Option<TomlHardware>,
    server: Option<TomlServer>,
    advanced: Option<TomlAdvanced>,
}

#[derive(Deserialize)]
struct TomlModel {
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    architecture: Option<String>,
    context_length: Option<usize>,
}

#[derive(Deserialize)]
struct TomlGeneration {
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<u64>,
    stop_sequences: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct TomlHardware {
    backend: Option<BackendChoice>,
    device_index: Option<usize>,
    num_threads: Option<usize>,
    gpu_layers: Option<usize>,
    memory_limit: Option<u64>,
    use_flash_attention: Option<bool>,
}

#[derive(Deserialize)]
struct TomlServer {
    host: Option<String>,
    port: Option<u16>,
    max_concurrent_requests: Option<usize>,
    request_timeout_ms: Option<u64>,
    enable_cors: Option<bool>,
}

#[derive(Deserialize)]
struct TomlAdvanced {
    kv_cache_quantization: Option<bool>,
    speculative_decoding: Option<bool>,
    draft_model_path: Option<String>,
    batch_size: Option<usize>,
    log_level: Option<String>,
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn valid_builder() -> ConfigBuilder {
        ConfigBuilder::new().model_path("model.gguf")
    }

    // ── Default / basics ─────────────────────────────────────────────

    #[test]
    fn default_config_with_model_is_valid() {
        let cfg = valid_builder().build().expect("should be valid");
        assert!(cfg.validate().is_valid());
    }

    #[test]
    fn default_builder_has_sensible_defaults() {
        let b = ConfigBuilder::new();
        assert_eq!(b.temperature, 1.0);
        assert_eq!(b.top_p, 0.9);
        assert_eq!(b.top_k, 50);
        assert_eq!(b.max_tokens, 256);
        assert_eq!(b.context_length, 2048);
        assert_eq!(b.backend, BackendChoice::Auto);
        assert_eq!(b.port, 8080);
        assert_eq!(b.batch_size, 1);
    }

    // ── Temperature validation ───────────────────────────────────────

    #[test]
    fn temperature_negative_rejected() {
        let res = valid_builder().temperature(-0.1).build();
        assert!(res.is_err());
        let v = res.unwrap_err();
        assert!(v.errors.iter().any(|e| e.contains("temperature")));
    }

    #[test]
    fn temperature_above_two_rejected() {
        let res = valid_builder().temperature(2.1).build();
        assert!(res.is_err());
        let v = res.unwrap_err();
        assert!(v.errors.iter().any(|e| e.contains("temperature")));
    }

    #[test]
    fn temperature_zero_accepted() {
        let cfg = valid_builder().temperature(0.0).build();
        assert!(cfg.is_ok());
    }

    #[test]
    fn temperature_two_accepted() {
        let cfg = valid_builder().temperature(2.0).build();
        assert!(cfg.is_ok());
    }

    #[test]
    fn temperature_high_warning() {
        let cfg = valid_builder().temperature(1.8).build().unwrap();
        let v = cfg.validate();
        assert!(v.has_warnings());
        assert!(v.warnings.iter().any(|w| w.contains("temperature")));
    }

    #[test]
    fn temperature_at_1_5_no_warning() {
        let cfg = valid_builder().temperature(1.5).build().unwrap();
        let v = cfg.validate();
        assert!(!v.has_warnings());
    }

    // ── top_p validation ─────────────────────────────────────────────

    #[test]
    fn top_p_zero_rejected() {
        let res = valid_builder().top_p(0.0).build();
        assert!(res.is_err());
        assert!(
            res.unwrap_err().errors.iter().any(|e| e.contains("top_p"))
        );
    }

    #[test]
    fn top_p_negative_rejected() {
        let res = valid_builder().top_p(-0.5).build();
        assert!(res.is_err());
    }

    #[test]
    fn top_p_above_one_rejected() {
        let res = valid_builder().top_p(1.01).build();
        assert!(res.is_err());
    }

    #[test]
    fn top_p_one_accepted() {
        assert!(valid_builder().top_p(1.0).build().is_ok());
    }

    #[test]
    fn top_p_small_positive_accepted() {
        assert!(valid_builder().top_p(0.01).build().is_ok());
    }

    // ── top_k validation ─────────────────────────────────────────────

    #[test]
    fn top_k_zero_rejected() {
        let res = valid_builder().top_k(0).build();
        assert!(res.is_err());
        assert!(
            res.unwrap_err().errors.iter().any(|e| e.contains("top_k"))
        );
    }

    #[test]
    fn top_k_one_accepted() {
        assert!(valid_builder().top_k(1).build().is_ok());
    }

    // ── Port validation ──────────────────────────────────────────────

    #[test]
    fn port_zero_rejected() {
        let res = valid_builder().port(0).build();
        assert!(res.is_err());
        assert!(
            res.unwrap_err().errors.iter().any(|e| e.contains("port"))
        );
    }

    #[test]
    fn port_one_accepted() {
        assert!(valid_builder().port(1).build().is_ok());
    }

    #[test]
    fn port_max_accepted() {
        assert!(valid_builder().port(65535).build().is_ok());
    }

    // ── Model path validation ────────────────────────────────────────

    #[test]
    fn empty_model_path_rejected() {
        let res = ConfigBuilder::new().build();
        assert!(res.is_err());
        assert!(
            res.unwrap_err()
                .errors
                .iter()
                .any(|e| e.contains("model_path"))
        );
    }

    #[test]
    fn model_path_set_accepted() {
        assert!(valid_builder().build().is_ok());
    }

    // ── Backend selection ────────────────────────────────────────────

    #[test]
    fn backend_defaults_to_auto() {
        let cfg = valid_builder().build().unwrap();
        assert_eq!(cfg.hardware.backend, BackendChoice::Auto);
    }

    #[test]
    fn backend_can_be_set_to_cuda() {
        let cfg =
            valid_builder().backend(BackendChoice::Cuda).build().unwrap();
        assert_eq!(cfg.hardware.backend, BackendChoice::Cuda);
    }

    #[test]
    fn backend_can_be_set_to_cpu() {
        let cfg =
            valid_builder().backend(BackendChoice::Cpu).build().unwrap();
        assert_eq!(cfg.hardware.backend, BackendChoice::Cpu);
    }

    #[test]
    fn backend_all_variants() {
        for b in [
            BackendChoice::Auto,
            BackendChoice::Cpu,
            BackendChoice::Cuda,
            BackendChoice::OpenCL,
            BackendChoice::Vulkan,
            BackendChoice::Metal,
            BackendChoice::Rocm,
            BackendChoice::WebGpu,
        ] {
            let cfg = valid_builder().backend(b).build().unwrap();
            assert_eq!(cfg.hardware.backend, b);
        }
    }

    #[test]
    fn backend_display() {
        assert_eq!(BackendChoice::Cuda.to_string(), "cuda");
        assert_eq!(BackendChoice::OpenCL.to_string(), "opencl");
        assert_eq!(BackendChoice::WebGpu.to_string(), "webgpu");
    }

    // ── Builder fluent chaining ──────────────────────────────────────

    #[test]
    fn fluent_chaining_all_fields() {
        let cfg = ConfigBuilder::new()
            .model_path("m.gguf")
            .tokenizer_path("t.json")
            .architecture("bitnet-b1.58")
            .context_length(4096)
            .max_tokens(512)
            .temperature(0.7)
            .top_p(0.95)
            .top_k(40)
            .repetition_penalty(1.1)
            .frequency_penalty(0.5)
            .presence_penalty(0.3)
            .seed(42)
            .stop_sequences(vec!["<|end|>".to_string()])
            .backend(BackendChoice::Cuda)
            .device_index(1)
            .num_threads(8)
            .gpu_layers(32)
            .memory_limit(4_000_000_000)
            .use_flash_attention(true)
            .host("0.0.0.0")
            .port(3000)
            .max_concurrent_requests(64)
            .request_timeout_ms(60_000)
            .enable_cors(true)
            .kv_cache_quantization(true)
            .speculative_decoding(true)
            .draft_model_path("draft.gguf")
            .batch_size(4)
            .log_level("debug")
            .build()
            .expect("all fields valid");

        assert_eq!(cfg.model.model_path, "m.gguf");
        assert_eq!(cfg.model.tokenizer_path.as_deref(), Some("t.json"));
        assert_eq!(cfg.model.architecture.as_deref(), Some("bitnet-b1.58"));
        assert_eq!(cfg.model.context_length, 4096);
        assert_eq!(cfg.generation.max_tokens, 512);
        assert!((cfg.generation.temperature - 0.7).abs() < f32::EPSILON);
        assert!((cfg.generation.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(cfg.generation.top_k, 40);
        assert_eq!(cfg.generation.seed, Some(42));
        assert_eq!(cfg.generation.stop_sequences, vec!["<|end|>"]);
        assert_eq!(cfg.hardware.backend, BackendChoice::Cuda);
        assert_eq!(cfg.hardware.device_index, 1);
        assert_eq!(cfg.hardware.num_threads, 8);
        assert_eq!(cfg.hardware.gpu_layers, Some(32));
        assert_eq!(cfg.hardware.memory_limit, Some(4_000_000_000));
        assert!(cfg.hardware.use_flash_attention);
        assert_eq!(cfg.server.host, "0.0.0.0");
        assert_eq!(cfg.server.port, 3000);
        assert_eq!(cfg.server.max_concurrent_requests, 64);
        assert_eq!(cfg.server.request_timeout_ms, 60_000);
        assert!(cfg.server.enable_cors);
        assert!(cfg.advanced.kv_cache_quantization);
        assert!(cfg.advanced.speculative_decoding);
        assert_eq!(
            cfg.advanced.draft_model_path.as_deref(),
            Some("draft.gguf")
        );
        assert_eq!(cfg.advanced.batch_size, 4);
        assert_eq!(cfg.advanced.log_level, "debug");
    }

    // ── Environment loading ──────────────────────────────────────────

    #[test]
    #[serial(bitnet_env)]
    fn from_env_reads_model_path() {
        temp_env::with_var("BITNET_MODEL_PATH", Some("env.gguf"), || {
            let b = ConfigBuilder::from_env();
            assert_eq!(b.model_path, "env.gguf");
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_reads_temperature() {
        temp_env::with_var("BITNET_TEMPERATURE", Some("0.5"), || {
            let b = ConfigBuilder::from_env();
            assert!((b.temperature - 0.5).abs() < f32::EPSILON);
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_reads_backend() {
        temp_env::with_var("BITNET_BACKEND", Some("cuda"), || {
            let b = ConfigBuilder::from_env();
            assert_eq!(b.backend, BackendChoice::Cuda);
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_reads_seed() {
        temp_env::with_var("BITNET_SEED", Some("42"), || {
            let b = ConfigBuilder::from_env();
            assert_eq!(b.seed, Some(42));
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_reads_port() {
        temp_env::with_var("BITNET_PORT", Some("3000"), || {
            let b = ConfigBuilder::from_env();
            assert_eq!(b.port, 3000);
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_reads_num_threads() {
        temp_env::with_var("BITNET_NUM_THREADS", Some("16"), || {
            let b = ConfigBuilder::from_env();
            assert_eq!(b.num_threads, 16);
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_reads_host() {
        temp_env::with_var("BITNET_HOST", Some("0.0.0.0"), || {
            let b = ConfigBuilder::from_env();
            assert_eq!(b.host, "0.0.0.0");
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_reads_log_level() {
        temp_env::with_var("BITNET_LOG_LEVEL", Some("debug"), || {
            let b = ConfigBuilder::from_env();
            assert_eq!(b.log_level, "debug");
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_reads_top_p() {
        temp_env::with_var("BITNET_TOP_P", Some("0.95"), || {
            let b = ConfigBuilder::from_env();
            assert!((b.top_p - 0.95).abs() < f32::EPSILON);
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_reads_top_k() {
        temp_env::with_var("BITNET_TOP_K", Some("100"), || {
            let b = ConfigBuilder::from_env();
            assert_eq!(b.top_k, 100);
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_invalid_temperature_keeps_default() {
        temp_env::with_var("BITNET_TEMPERATURE", Some("notanumber"), || {
            let b = ConfigBuilder::from_env();
            assert!((b.temperature - 1.0).abs() < f32::EPSILON);
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_reads_batch_size() {
        temp_env::with_var("BITNET_BATCH_SIZE", Some("8"), || {
            let b = ConfigBuilder::from_env();
            assert_eq!(b.batch_size, 8);
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_reads_max_tokens() {
        temp_env::with_var("BITNET_MAX_TOKENS", Some("1024"), || {
            let b = ConfigBuilder::from_env();
            assert_eq!(b.max_tokens, 1024);
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn from_env_reads_repetition_penalty() {
        temp_env::with_var(
            "BITNET_REPETITION_PENALTY",
            Some("1.2"),
            || {
                let b = ConfigBuilder::from_env();
                assert!((b.repetition_penalty - 1.2).abs() < f32::EPSILON);
            },
        );
    }

    // ── TOML parsing ─────────────────────────────────────────────────

    #[test]
    fn toml_basic_parsing() {
        let toml = r#"
[model]
model_path = "model.gguf"
context_length = 4096

[generation]
temperature = 0.7
max_tokens = 512
"#;
        let b = ConfigBuilder::from_toml(toml).unwrap();
        assert_eq!(b.model_path, "model.gguf");
        assert_eq!(b.context_length, 4096);
        assert!((b.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(b.max_tokens, 512);
    }

    #[test]
    fn toml_hardware_section() {
        let toml = r#"
[model]
model_path = "m.gguf"

[hardware]
backend = "cuda"
num_threads = 8
device_index = 1
use_flash_attention = true
gpu_layers = 32
memory_limit = 8000000000
"#;
        let b = ConfigBuilder::from_toml(toml).unwrap();
        assert_eq!(b.backend, BackendChoice::Cuda);
        assert_eq!(b.num_threads, 8);
        assert_eq!(b.device_index, 1);
        assert!(b.use_flash_attention);
        assert_eq!(b.gpu_layers, Some(32));
        assert_eq!(b.memory_limit, Some(8_000_000_000));
    }

    #[test]
    fn toml_server_section() {
        let toml = r#"
[model]
model_path = "m.gguf"

[server]
host = "0.0.0.0"
port = 3000
max_concurrent_requests = 32
request_timeout_ms = 60000
enable_cors = true
"#;
        let b = ConfigBuilder::from_toml(toml).unwrap();
        assert_eq!(b.host, "0.0.0.0");
        assert_eq!(b.port, 3000);
        assert_eq!(b.max_concurrent_requests, 32);
        assert_eq!(b.request_timeout_ms, 60_000);
        assert!(b.enable_cors);
    }

    #[test]
    fn toml_advanced_section() {
        let toml = r#"
[model]
model_path = "m.gguf"

[advanced]
kv_cache_quantization = true
speculative_decoding = true
draft_model_path = "draft.gguf"
batch_size = 4
log_level = "debug"
"#;
        let b = ConfigBuilder::from_toml(toml).unwrap();
        assert!(b.kv_cache_quantization);
        assert!(b.speculative_decoding);
        assert_eq!(b.draft_model_path.as_deref(), Some("draft.gguf"));
        assert_eq!(b.batch_size, 4);
        assert_eq!(b.log_level, "debug");
    }

    #[test]
    fn toml_generation_full() {
        let toml = r#"
[model]
model_path = "m.gguf"

[generation]
max_tokens = 1024
temperature = 0.8
top_p = 0.95
top_k = 40
repetition_penalty = 1.1
frequency_penalty = 0.5
presence_penalty = 0.3
seed = 42
stop_sequences = ["<|end|>", "STOP"]
"#;
        let b = ConfigBuilder::from_toml(toml).unwrap();
        assert_eq!(b.max_tokens, 1024);
        assert!((b.temperature - 0.8).abs() < f32::EPSILON);
        assert!((b.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(b.top_k, 40);
        assert!((b.repetition_penalty - 1.1).abs() < f32::EPSILON);
        assert!((b.frequency_penalty - 0.5).abs() < f32::EPSILON);
        assert!((b.presence_penalty - 0.3).abs() < f32::EPSILON);
        assert_eq!(b.seed, Some(42));
        assert_eq!(
            b.stop_sequences,
            vec!["<|end|>".to_string(), "STOP".to_string()]
        );
    }

    #[test]
    fn toml_invalid_syntax_returns_error() {
        let bad = "this is not [valid toml";
        assert!(ConfigBuilder::from_toml(bad).is_err());
    }

    #[test]
    fn toml_empty_uses_defaults() {
        let b = ConfigBuilder::from_toml("").unwrap();
        assert_eq!(b.temperature, 1.0);
        assert_eq!(b.top_k, 50);
    }

    #[test]
    fn toml_partial_model_section() {
        let toml = r#"
[model]
model_path = "partial.gguf"
"#;
        let b = ConfigBuilder::from_toml(toml).unwrap();
        assert_eq!(b.model_path, "partial.gguf");
        assert_eq!(b.context_length, 2048); // default kept
    }

    // ── Full config round-trip ───────────────────────────────────────

    #[test]
    fn full_config_round_trip_via_toml() {
        let toml = r#"
[model]
model_path = "round-trip.gguf"
tokenizer_path = "tok.json"
architecture = "bitnet"
context_length = 8192

[generation]
max_tokens = 100
temperature = 0.6
top_p = 0.8
top_k = 20
seed = 123

[hardware]
backend = "cpu"
num_threads = 2

[server]
port = 9090
host = "localhost"

[advanced]
batch_size = 2
log_level = "warn"
"#;
        let cfg = ConfigBuilder::from_toml(toml).unwrap().build().unwrap();
        assert_eq!(cfg.model.model_path, "round-trip.gguf");
        assert_eq!(
            cfg.model.tokenizer_path.as_deref(),
            Some("tok.json")
        );
        assert_eq!(cfg.model.context_length, 8192);
        assert_eq!(cfg.generation.max_tokens, 100);
        assert_eq!(cfg.generation.seed, Some(123));
        assert_eq!(cfg.hardware.backend, BackendChoice::Cpu);
        assert_eq!(cfg.server.port, 9090);
        assert_eq!(cfg.advanced.log_level, "warn");
    }

    // ── Server config defaults ───────────────────────────────────────

    #[test]
    fn server_defaults() {
        let cfg = valid_builder().build().unwrap();
        assert_eq!(cfg.server.host, "127.0.0.1");
        assert_eq!(cfg.server.port, 8080);
        assert_eq!(cfg.server.max_concurrent_requests, 16);
        assert_eq!(cfg.server.request_timeout_ms, 30_000);
        assert!(!cfg.server.enable_cors);
    }

    // ── Advanced config defaults ─────────────────────────────────────

    #[test]
    fn advanced_defaults() {
        let cfg = valid_builder().build().unwrap();
        assert!(!cfg.advanced.kv_cache_quantization);
        assert!(!cfg.advanced.speculative_decoding);
        assert!(cfg.advanced.draft_model_path.is_none());
        assert_eq!(cfg.advanced.batch_size, 1);
        assert_eq!(cfg.advanced.log_level, "info");
    }

    // ── RopeScaling variants ─────────────────────────────────────────

    #[test]
    fn rope_scaling_linear() {
        let cfg = valid_builder()
            .rope_scaling(RopeScaling::Linear(2.0))
            .build()
            .unwrap();
        assert_eq!(cfg.model.rope_scaling, Some(RopeScaling::Linear(2.0)));
    }

    #[test]
    fn rope_scaling_ntk() {
        let cfg = valid_builder()
            .rope_scaling(RopeScaling::Ntk(4.0))
            .build()
            .unwrap();
        assert_eq!(cfg.model.rope_scaling, Some(RopeScaling::Ntk(4.0)));
    }

    #[test]
    fn rope_scaling_yarn() {
        let cfg = valid_builder()
            .rope_scaling(RopeScaling::Yarn {
                factor: 2.0,
                beta: 0.5,
            })
            .build()
            .unwrap();
        assert_eq!(
            cfg.model.rope_scaling,
            Some(RopeScaling::Yarn { factor: 2.0, beta: 0.5 })
        );
    }

    #[test]
    fn rope_scaling_none_by_default() {
        let cfg = valid_builder().build().unwrap();
        assert!(cfg.model.rope_scaling.is_none());
    }

    // ── Multiple validation errors collected ─────────────────────────

    #[test]
    fn multiple_errors_collected() {
        let res = ConfigBuilder::new()
            .temperature(-1.0)
            .top_p(0.0)
            .top_k(0)
            .port(0)
            .build();
        assert!(res.is_err());
        let v = res.unwrap_err();
        assert!(v.errors.len() >= 4);
    }

    #[test]
    fn errors_and_warnings_separate() {
        // temperature=1.9 is valid but warns; top_k=0 is an error
        let res = ConfigBuilder::new()
            .model_path("m.gguf")
            .temperature(1.9)
            .top_k(0)
            .build();
        // top_k=0 causes error
        assert!(res.is_err());
        let v = res.unwrap_err();
        assert!(!v.errors.is_empty());
        assert!(v.has_warnings());
    }

    // ── Seed reproducibility config ──────────────────────────────────

    #[test]
    fn seed_is_none_by_default() {
        let cfg = valid_builder().build().unwrap();
        assert!(cfg.generation.seed.is_none());
    }

    #[test]
    fn seed_can_be_set() {
        let cfg = valid_builder().seed(42).build().unwrap();
        assert_eq!(cfg.generation.seed, Some(42));
    }

    #[test]
    fn seed_zero_accepted() {
        let cfg = valid_builder().seed(0).build().unwrap();
        assert_eq!(cfg.generation.seed, Some(0));
    }

    // ── Stop sequences ───────────────────────────────────────────────

    #[test]
    fn stop_sequences_empty_by_default() {
        let cfg = valid_builder().build().unwrap();
        assert!(cfg.generation.stop_sequences.is_empty());
    }

    #[test]
    fn stop_sequences_can_be_set() {
        let cfg = valid_builder()
            .stop_sequences(vec![
                "<|end|>".to_string(),
                "STOP".to_string(),
            ])
            .build()
            .unwrap();
        assert_eq!(cfg.generation.stop_sequences.len(), 2);
        assert_eq!(cfg.generation.stop_sequences[0], "<|end|>");
    }

    // ── Misc validation ──────────────────────────────────────────────

    #[test]
    fn max_tokens_zero_rejected() {
        let res = valid_builder().max_tokens(0).build();
        assert!(res.is_err());
    }

    #[test]
    fn num_threads_zero_rejected() {
        let res = valid_builder().num_threads(0).build();
        assert!(res.is_err());
    }

    #[test]
    fn batch_size_zero_rejected() {
        let res = valid_builder().batch_size(0).build();
        assert!(res.is_err());
    }

    #[test]
    fn context_length_zero_rejected() {
        let res = valid_builder().context_length(0).build();
        assert!(res.is_err());
    }

    #[test]
    fn speculative_without_draft_rejected() {
        let res = valid_builder().speculative_decoding(true).build();
        assert!(res.is_err());
        assert!(res
            .unwrap_err()
            .errors
            .iter()
            .any(|e| e.contains("draft_model_path")));
    }

    #[test]
    fn speculative_with_draft_accepted() {
        let cfg = valid_builder()
            .speculative_decoding(true)
            .draft_model_path("draft.gguf")
            .build();
        assert!(cfg.is_ok());
    }

    #[test]
    fn max_concurrent_requests_zero_rejected() {
        let res = valid_builder().max_concurrent_requests(0).build();
        assert!(res.is_err());
    }

    #[test]
    fn request_timeout_zero_rejected() {
        let res = valid_builder().request_timeout_ms(0).build();
        assert!(res.is_err());
    }

    #[test]
    fn negative_repetition_penalty_rejected() {
        let res = valid_builder().repetition_penalty(-0.1).build();
        assert!(res.is_err());
    }

    #[test]
    fn negative_frequency_penalty_rejected() {
        let res = valid_builder().frequency_penalty(-0.1).build();
        assert!(res.is_err());
    }

    #[test]
    fn negative_presence_penalty_rejected() {
        let res = valid_builder().presence_penalty(-0.1).build();
        assert!(res.is_err());
    }

    // ── ConfigValidation Display ─────────────────────────────────────

    #[test]
    fn validation_display_shows_errors_and_warnings() {
        let mut v = ConfigValidation::new();
        v.error("something wrong");
        v.warning("watch out");
        let s = v.to_string();
        assert!(s.contains("ERROR: something wrong"));
        assert!(s.contains("WARN:  watch out"));
    }

    #[test]
    fn validation_display_empty_is_empty() {
        let v = ConfigValidation::new();
        assert!(v.to_string().is_empty());
    }
}
