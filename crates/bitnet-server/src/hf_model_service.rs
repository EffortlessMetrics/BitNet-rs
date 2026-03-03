//! HuggingFace model service for SLM inference integration.
//!
//! Provides [`HfModelService`] for managing HuggingFace model lifecycle
//! (loading, state tracking, inference) within the HTTP server.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Model format
// ---------------------------------------------------------------------------

/// Supported model serialization formats.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Safetensors,
    Gguf,
    #[default]
    Auto,
}

impl ModelFormat {
    /// Detect format from a file path extension.
    pub fn from_path(path: &str) -> Self {
        match Path::new(path).extension().and_then(|e| e.to_str()) {
            Some("safetensors") => Self::Safetensors,
            Some("gguf") => Self::Gguf,
            _ => Self::Auto,
        }
    }
}

// ---------------------------------------------------------------------------
// Model info
// ---------------------------------------------------------------------------

/// Metadata describing a loaded model's architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub architecture: String,
    pub num_parameters: u64,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub max_context: usize,
    pub dtype: String,
}

impl ModelInfo {
    /// Estimate parameter count from architecture dimensions.
    ///
    /// Uses a simplified heuristic: `12 * num_layers * hidden_size²` which
    /// approximates the dominant weight matrices (QKV projections, FFN).
    pub fn estimate_params(hidden_size: usize, num_layers: usize) -> u64 {
        12 * (num_layers as u64) * (hidden_size as u64) * (hidden_size as u64)
    }
}

// ---------------------------------------------------------------------------
// Load state
// ---------------------------------------------------------------------------

/// Model lifecycle state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadState {
    NotLoaded,
    Loading,
    Ready,
    Error(String),
}

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

/// Request to load a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelLoadRequest {
    pub model_path: String,
    #[serde(default)]
    pub model_format: Option<ModelFormat>,
    #[serde(default)]
    pub architecture: Option<String>,
}

/// Response after attempting to load a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelLoadResponse {
    pub success: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_info: Option<ModelInfo>,
    pub load_time_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Request for dense model inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfInferenceRequest {
    pub prompt: String,
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: Option<usize>,
}

fn default_temperature() -> f32 {
    1.0
}

/// Validation error for inference requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceValidationError {
    EmptyPrompt,
    ZeroMaxTokens,
    NegativeTemperature,
}

impl std::fmt::Display for InferenceValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPrompt => write!(f, "prompt must not be empty"),
            Self::ZeroMaxTokens => write!(f, "max_tokens must be > 0"),
            Self::NegativeTemperature => write!(f, "temperature must be >= 0"),
        }
    }
}

impl HfInferenceRequest {
    /// Validate the request, returning the first error found.
    pub fn validate(&self) -> Result<(), InferenceValidationError> {
        if self.prompt.is_empty() {
            return Err(InferenceValidationError::EmptyPrompt);
        }
        if self.max_tokens == 0 {
            return Err(InferenceValidationError::ZeroMaxTokens);
        }
        if self.temperature < 0.0 {
            return Err(InferenceValidationError::NegativeTemperature);
        }
        Ok(())
    }
}

/// Response from dense model inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfInferenceResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub time_ms: u64,
}

// ---------------------------------------------------------------------------
// HfModelService
// ---------------------------------------------------------------------------

/// Service that manages HuggingFace model loading and inference.
pub struct HfModelService {
    state: ModelLoadState,
    model_info: Option<ModelInfo>,
}

impl HfModelService {
    /// Create a new service in the `NotLoaded` state.
    pub fn new() -> Self {
        Self { state: ModelLoadState::NotLoaded, model_info: None }
    }

    /// Current load state.
    pub fn state(&self) -> &ModelLoadState {
        &self.state
    }

    /// Model info, available only in the `Ready` state.
    pub fn model_info(&self) -> Option<&ModelInfo> {
        self.model_info.as_ref()
    }

    /// Transition to `Loading`.
    pub fn begin_loading(&mut self) {
        self.state = ModelLoadState::Loading;
    }

    /// Transition to `Ready` with the given model info.
    pub fn finish_loading(&mut self, info: ModelInfo) {
        self.model_info = Some(info);
        self.state = ModelLoadState::Ready;
    }

    /// Transition to `Error`.
    pub fn fail_loading(&mut self, reason: String) {
        self.model_info = None;
        self.state = ModelLoadState::Error(reason);
    }

    /// Simulate a model load (no real I/O) and return a response.
    ///
    /// In production this would delegate to `HuggingFaceLoader` from
    /// `bitnet-models`; here we perform the state-machine transition and
    /// build a `HfModelLoadResponse`.
    pub fn load_model(&mut self, req: &HfModelLoadRequest) -> HfModelLoadResponse {
        let start = Instant::now();
        self.begin_loading();

        let format =
            req.model_format.clone().unwrap_or_else(|| ModelFormat::from_path(&req.model_path));
        let arch = req.architecture.clone().unwrap_or_else(|| "unknown".to_string());

        // Build placeholder model info
        let info = ModelInfo {
            architecture: arch,
            num_parameters: 0,
            hidden_size: 0,
            num_layers: 0,
            num_heads: 0,
            vocab_size: 0,
            max_context: 0,
            dtype: match format {
                ModelFormat::Safetensors => "float16".to_string(),
                ModelFormat::Gguf => "i2_s".to_string(),
                ModelFormat::Auto => "auto".to_string(),
            },
        };

        self.finish_loading(info.clone());

        HfModelLoadResponse {
            success: true,
            model_info: Some(info),
            load_time_ms: start.elapsed().as_millis() as u64,
            error: None,
        }
    }
}

impl Default for HfModelService {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ModelLoadRequest serialization / deserialization --------------------

    #[test]
    fn load_request_serialization_roundtrip() {
        let req = HfModelLoadRequest {
            model_path: "models/phi4.safetensors".into(),
            model_format: Some(ModelFormat::Safetensors),
            architecture: Some("phi4".into()),
        };
        let json = serde_json::to_string(&req).unwrap();
        let deser: HfModelLoadRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.model_path, req.model_path);
        assert_eq!(deser.model_format, req.model_format);
        assert_eq!(deser.architecture, req.architecture);
    }

    #[test]
    fn load_request_deserialize_minimal() {
        let json = r#"{"model_path":"m.gguf"}"#;
        let req: HfModelLoadRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model_path, "m.gguf");
        assert!(req.model_format.is_none());
        assert!(req.architecture.is_none());
    }

    #[test]
    fn load_response_serialization_roundtrip() {
        let resp = HfModelLoadResponse {
            success: true,
            model_info: Some(ModelInfo {
                architecture: "llama3".into(),
                num_parameters: 7_000_000_000,
                hidden_size: 4096,
                num_layers: 32,
                num_heads: 32,
                vocab_size: 32000,
                max_context: 4096,
                dtype: "float16".into(),
            }),
            load_time_ms: 42,
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deser: HfModelLoadResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.success, true);
        assert_eq!(deser.load_time_ms, 42);
        assert!(deser.model_info.is_some());
    }

    // -- HfModelService state transitions -----------------------------------

    #[test]
    fn service_starts_not_loaded() {
        let svc = HfModelService::new();
        assert_eq!(*svc.state(), ModelLoadState::NotLoaded);
        assert!(svc.model_info().is_none());
    }

    #[test]
    fn service_transition_not_loaded_to_loading() {
        let mut svc = HfModelService::new();
        svc.begin_loading();
        assert_eq!(*svc.state(), ModelLoadState::Loading);
    }

    #[test]
    fn service_transition_loading_to_ready() {
        let mut svc = HfModelService::new();
        svc.begin_loading();
        svc.finish_loading(ModelInfo {
            architecture: "phi4".into(),
            num_parameters: 3_800_000_000,
            hidden_size: 3072,
            num_layers: 32,
            num_heads: 32,
            vocab_size: 100352,
            max_context: 16384,
            dtype: "float16".into(),
        });
        assert_eq!(*svc.state(), ModelLoadState::Ready);
        assert!(svc.model_info().is_some());
    }

    #[test]
    fn service_transition_loading_to_error() {
        let mut svc = HfModelService::new();
        svc.begin_loading();
        svc.fail_loading("file not found".into());
        assert_eq!(*svc.state(), ModelLoadState::Error("file not found".into()));
        assert!(svc.model_info().is_none());
    }

    #[test]
    fn service_load_model_ends_in_ready() {
        let mut svc = HfModelService::new();
        let resp = svc.load_model(&HfModelLoadRequest {
            model_path: "model.safetensors".into(),
            model_format: None,
            architecture: Some("llama3".into()),
        });
        assert!(resp.success);
        assert_eq!(*svc.state(), ModelLoadState::Ready);
    }

    // -- ModelInfo construction for architectures ---------------------------

    #[test]
    fn model_info_phi4() {
        let info = ModelInfo {
            architecture: "phi4".into(),
            num_parameters: 3_800_000_000,
            hidden_size: 3072,
            num_layers: 32,
            num_heads: 32,
            vocab_size: 100352,
            max_context: 16384,
            dtype: "bfloat16".into(),
        };
        assert_eq!(info.architecture, "phi4");
        assert_eq!(info.num_layers, 32);
    }

    #[test]
    fn model_info_llama3() {
        let info = ModelInfo {
            architecture: "llama3".into(),
            num_parameters: 8_000_000_000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            vocab_size: 128256,
            max_context: 8192,
            dtype: "float16".into(),
        };
        assert_eq!(info.architecture, "llama3");
        assert_eq!(info.vocab_size, 128256);
    }

    #[test]
    fn model_info_qwen2() {
        let info = ModelInfo {
            architecture: "qwen2".into(),
            num_parameters: 1_500_000_000,
            hidden_size: 1536,
            num_layers: 28,
            num_heads: 12,
            vocab_size: 151936,
            max_context: 32768,
            dtype: "bfloat16".into(),
        };
        assert_eq!(info.architecture, "qwen2");
        assert_eq!(info.max_context, 32768);
    }

    #[test]
    fn model_info_gemma() {
        let info = ModelInfo {
            architecture: "gemma".into(),
            num_parameters: 2_500_000_000,
            hidden_size: 2048,
            num_layers: 18,
            num_heads: 8,
            vocab_size: 256000,
            max_context: 8192,
            dtype: "float16".into(),
        };
        assert_eq!(info.architecture, "gemma");
        assert_eq!(info.num_heads, 8);
    }

    #[test]
    fn model_info_mistral() {
        let info = ModelInfo {
            architecture: "mistral".into(),
            num_parameters: 7_000_000_000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            vocab_size: 32000,
            max_context: 8192,
            dtype: "float16".into(),
        };
        assert_eq!(info.architecture, "mistral");
        assert_eq!(info.hidden_size, 4096);
    }

    // -- InferenceRequest validation ----------------------------------------

    #[test]
    fn inference_request_valid() {
        let req = HfInferenceRequest {
            prompt: "Hello".into(),
            max_tokens: 32,
            temperature: 0.7,
            top_k: Some(50),
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn inference_request_empty_prompt() {
        let req =
            HfInferenceRequest { prompt: "".into(), max_tokens: 32, temperature: 0.7, top_k: None };
        assert_eq!(req.validate().unwrap_err(), InferenceValidationError::EmptyPrompt);
    }

    #[test]
    fn inference_request_zero_max_tokens() {
        let req = HfInferenceRequest {
            prompt: "hi".into(),
            max_tokens: 0,
            temperature: 0.7,
            top_k: None,
        };
        assert_eq!(req.validate().unwrap_err(), InferenceValidationError::ZeroMaxTokens);
    }

    #[test]
    fn inference_request_negative_temperature() {
        let req = HfInferenceRequest {
            prompt: "hi".into(),
            max_tokens: 10,
            temperature: -0.5,
            top_k: None,
        };
        assert_eq!(req.validate().unwrap_err(), InferenceValidationError::NegativeTemperature);
    }

    #[test]
    fn inference_request_serialization_roundtrip() {
        let req = HfInferenceRequest {
            prompt: "What is 2+2?".into(),
            max_tokens: 64,
            temperature: 0.9,
            top_k: Some(40),
        };
        let json = serde_json::to_string(&req).unwrap();
        let deser: HfInferenceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.prompt, "What is 2+2?");
        assert_eq!(deser.max_tokens, 64);
    }

    // -- InferenceResponse construction -------------------------------------

    #[test]
    fn inference_response_construction() {
        let resp = HfInferenceResponse { text: "4".into(), tokens_generated: 1, time_ms: 10 };
        assert_eq!(resp.text, "4");
        assert_eq!(resp.tokens_generated, 1);
    }

    #[test]
    fn inference_response_serialization_roundtrip() {
        let resp =
            HfInferenceResponse { text: "hello world".into(), tokens_generated: 2, time_ms: 50 };
        let json = serde_json::to_string(&resp).unwrap();
        let deser: HfInferenceResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.tokens_generated, 2);
        assert_eq!(deser.time_ms, 50);
    }

    // -- Model format detection from path -----------------------------------

    #[test]
    fn format_from_safetensors_path() {
        assert_eq!(ModelFormat::from_path("model.safetensors"), ModelFormat::Safetensors);
    }

    #[test]
    fn format_from_gguf_path() {
        assert_eq!(ModelFormat::from_path("model.gguf"), ModelFormat::Gguf);
    }

    #[test]
    fn format_from_unknown_path() {
        assert_eq!(ModelFormat::from_path("model.bin"), ModelFormat::Auto);
    }

    #[test]
    fn format_from_no_extension() {
        assert_eq!(ModelFormat::from_path("model"), ModelFormat::Auto);
    }

    // -- Default values for optional fields ---------------------------------

    #[test]
    fn model_format_default_is_auto() {
        assert_eq!(ModelFormat::default(), ModelFormat::Auto);
    }

    #[test]
    fn hf_model_service_default_trait() {
        let svc = HfModelService::default();
        assert_eq!(*svc.state(), ModelLoadState::NotLoaded);
    }

    #[test]
    fn inference_request_defaults_via_deserialize() {
        let json = r#"{"prompt":"hi","max_tokens":10}"#;
        let req: HfInferenceRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.temperature, 1.0);
        assert!(req.top_k.is_none());
    }

    // -- Estimate params helper ---------------------------------------------

    #[test]
    fn estimate_params_basic() {
        // 12 * 32 * 4096^2 = 6_442_450_944
        let est = ModelInfo::estimate_params(4096, 32);
        assert_eq!(est, 6_442_450_944);
    }
}
