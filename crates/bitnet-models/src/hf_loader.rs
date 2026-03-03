//! HuggingFace model loader integrating SafeTensors reader, config auto-detection,
//! weight name mapping, and architecture registry.
//!
//! Loads any HuggingFace model directory containing `config.json` plus one or
//! more `.safetensors` files. Tensor names are automatically mapped from the
//! HF convention (`model.layers.N.self_attn.q_proj.weight`) to the internal
//! canonical convention (`layers.N.attention.q_proj.weight`).
//!
//! # Examples
//!
//! ```no_run
//! use bitnet_models::hf_loader::HfModelLoader;
//!
//! let loader = HfModelLoader::from_directory("models/phi-4").unwrap();
//! println!("Architecture: {}", loader.architecture());
//! println!("Tensors: {:?}", loader.tensor_names().len());
//!
//! let embeddings = loader.load_embeddings().unwrap();
//! let layer0 = loader.load_layer(0).unwrap();
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;
use tracing::debug;

use crate::architecture::{
    ArchitectureConfig, ModelArchitecture, detect_architecture, get_defaults,
};
use crate::safetensors_reader::SafeTensorsReader;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors specific to HuggingFace model loading.
#[derive(Debug, Error)]
pub enum HfLoaderError {
    /// `config.json` was not found in the model directory.
    #[error("config.json not found in {0}")]
    ConfigNotFound(String),

    /// No `.safetensors` files were found in the model directory.
    #[error("no safetensors files found in {0}")]
    NoSafetensors(String),

    /// `config.json` could not be parsed.
    #[error("failed to parse config.json: {0}")]
    ConfigParse(String),

    /// A requested tensor was not found after name mapping.
    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    /// Required tensors are missing from the model.
    #[error("missing required tensors: {0:?}")]
    IncompleteTensors(Vec<String>),

    /// An I/O error occurred.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// An error from the underlying SafeTensors reader.
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] crate::safetensors_reader::SafeTensorsReaderError),
}

type Result<T> = std::result::Result<T, HfLoaderError>;

// ---------------------------------------------------------------------------
// HfModelConfig — parsed from config.json
// ---------------------------------------------------------------------------

fn default_hidden_size() -> usize {
    4096
}
fn default_num_layers() -> usize {
    32
}
fn default_num_heads() -> usize {
    32
}
fn default_intermediate_size() -> usize {
    11008
}
fn default_vocab_size() -> usize {
    32000
}

/// Configuration parsed from a HuggingFace `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct HfModelConfig {
    /// Model type string (e.g. `"phi3"`, `"llama"`, `"qwen2"`).
    pub model_type: String,

    /// Hidden dimension / embedding size.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    /// Number of transformer layers.
    #[serde(alias = "n_layer", default = "default_num_layers")]
    pub num_hidden_layers: usize,

    /// Number of attention heads (queries).
    #[serde(alias = "n_head", default = "default_num_heads")]
    pub num_attention_heads: usize,

    /// Number of key/value heads (for GQA). Defaults to `num_attention_heads`.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Feed-forward intermediate dimension.
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

    /// Vocabulary size.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Maximum position embeddings (context length).
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    /// RoPE base frequency.
    #[serde(default)]
    pub rope_theta: Option<f64>,
}

impl HfModelConfig {
    /// Effective number of KV heads (falls back to `num_attention_heads` for MHA).
    pub fn effective_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Per-head dimension.
    pub fn head_dim(&self) -> usize {
        if self.num_attention_heads == 0 {
            return 0;
        }
        self.hidden_size / self.num_attention_heads
    }
}

// ---------------------------------------------------------------------------
// HF tensor name mapping
// ---------------------------------------------------------------------------

/// Map a HuggingFace SafeTensors tensor name to the internal canonical name.
///
/// The canonical naming convention matches the transformer module structure
/// used throughout the rest of the bitnet-rs codebase.
pub fn map_hf_name(name: &str) -> String {
    let stripped = name.strip_prefix("model.").unwrap_or(name);

    // Top-level tensors
    match stripped {
        "embed_tokens.weight" => return "embed_tokens.weight".to_string(),
        "norm.weight" | "norm.bias" => return format!("final_{stripped}"),
        _ => {}
    }
    if stripped == "lm_head.weight" || name == "lm_head.weight" {
        return "lm_head.weight".to_string();
    }

    // Layer tensors: layers.N.<component>
    if let Some(rest) = stripped.strip_prefix("layers.")
        && let Some(dot_pos) = rest.find('.')
    {
        let layer_num = &rest[..dot_pos];
        let component = &rest[dot_pos + 1..];

        let mapped = match component {
            // Attention projections
            "self_attn.q_proj.weight" => "attention.q_proj.weight",
            "self_attn.k_proj.weight" => "attention.k_proj.weight",
            "self_attn.v_proj.weight" => "attention.v_proj.weight",
            "self_attn.o_proj.weight" => "attention.o_proj.weight",

            // Norm layers
            "input_layernorm.weight" => "attention_norm.weight",
            "post_attention_layernorm.weight" => "post_attention_layernorm.weight",

            // MLP / feed-forward
            "mlp.gate_proj.weight" => "feed_forward.gate_proj.weight",
            "mlp.up_proj.weight" => "feed_forward.up_proj.weight",
            "mlp.down_proj.weight" => "feed_forward.down_proj.weight",

            // Pass through anything else (biases, unusual layers, etc.)
            other => other,
        };
        return format!("layers.{layer_num}.{mapped}");
    }

    // Fallback: return stripped name
    stripped.to_string()
}

// ---------------------------------------------------------------------------
// HfModelLoader
// ---------------------------------------------------------------------------

/// Load a HuggingFace model from a directory containing `config.json` and
/// one or more `.safetensors` files.
///
/// Integrates:
/// - [`SafeTensorsReader`] for tensor data access
/// - [`HfModelConfig`] for config.json parsing
/// - [`detect_architecture`] for architecture auto-detection
/// - [`map_hf_name`] for HF → internal name mapping
pub struct HfModelLoader {
    model_dir: PathBuf,
    config: HfModelConfig,
    architecture: ModelArchitecture,
    reader: SafeTensorsReader,
    /// Maps internal canonical name → original HF tensor name.
    name_map: HashMap<String, String>,
}

impl std::fmt::Debug for HfModelLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfModelLoader")
            .field("model_dir", &self.model_dir)
            .field("architecture", &self.architecture)
            .field("tensor_count", &self.name_map.len())
            .finish()
    }
}

impl HfModelLoader {
    /// Create a loader from a model directory, auto-detecting config and tensors.
    pub fn from_directory(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config_path = dir.join("config.json");
        if !config_path.exists() {
            return Err(HfLoaderError::ConfigNotFound(dir.display().to_string()));
        }

        let config_text = std::fs::read_to_string(&config_path)?;
        let config: HfModelConfig = serde_json::from_str(&config_text)
            .map_err(|e| HfLoaderError::ConfigParse(e.to_string()))?;

        Self::with_config(dir, config)
    }

    /// Create a loader from a directory with an explicit (pre-parsed) config.
    pub fn with_config(dir: impl AsRef<Path>, config: HfModelConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let architecture = detect_architecture(&config.model_type);
        let reader = Self::open_reader(&dir)?;

        // Build internal→HF name mapping
        let hf_names = reader.tensor_names();
        let mut name_map = HashMap::with_capacity(hf_names.len());
        for hf_name in &hf_names {
            let internal = map_hf_name(hf_name);
            name_map.insert(internal, hf_name.clone());
        }

        debug!(
            architecture = %architecture,
            tensors = name_map.len(),
            layers = config.num_hidden_layers,
            "HfModelLoader ready"
        );

        Ok(Self { model_dir: dir, config, architecture, reader, name_map })
    }

    /// Get the detected model configuration.
    pub fn config(&self) -> &HfModelConfig {
        &self.config
    }

    /// Get the detected model architecture.
    pub fn architecture(&self) -> &ModelArchitecture {
        &self.architecture
    }

    /// Get the architecture defaults for this model.
    pub fn architecture_defaults(&self) -> ArchitectureConfig {
        get_defaults(&self.architecture)
    }

    /// List all available tensors by their internal canonical names (sorted).
    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.name_map.keys().cloned().collect();
        names.sort();
        names
    }

    /// Load a specific tensor by its internal canonical name.
    pub fn load_tensor(&self, internal_name: &str) -> Result<Vec<f32>> {
        let hf_name = self
            .name_map
            .get(internal_name)
            .ok_or_else(|| HfLoaderError::TensorNotFound(internal_name.to_string()))?;
        Ok(self.reader.load_tensor(hf_name)?)
    }

    /// Load all tensors belonging to a specific layer, keyed by internal name.
    pub fn load_layer(&self, layer_idx: usize) -> Result<HashMap<String, Vec<f32>>> {
        let prefix = format!("layers.{layer_idx}.");
        let mut result = HashMap::new();

        for internal_name in self.name_map.keys() {
            if internal_name.starts_with(&prefix) {
                let data = self.load_tensor(internal_name)?;
                result.insert(internal_name.clone(), data);
            }
        }

        if result.is_empty() {
            return Err(HfLoaderError::TensorNotFound(format!("layer {layer_idx}")));
        }
        Ok(result)
    }

    /// Load the token embedding weights.
    pub fn load_embeddings(&self) -> Result<Vec<f32>> {
        self.load_tensor("embed_tokens.weight")
    }

    /// Validate that all tensors required for the configured architecture are present.
    ///
    /// Checks for: embeddings, final norm, and per-layer attention + FFN tensors.
    pub fn validate_completeness(&self) -> Result<()> {
        let mut missing = Vec::new();

        // Global tensors
        for name in &["embed_tokens.weight", "final_norm.weight"] {
            if !self.name_map.contains_key(*name) {
                missing.push((*name).to_string());
            }
        }

        // Per-layer tensors
        let layer_components = [
            "attention.q_proj.weight",
            "attention.k_proj.weight",
            "attention.v_proj.weight",
            "attention.o_proj.weight",
            "attention_norm.weight",
            "post_attention_layernorm.weight",
            "feed_forward.gate_proj.weight",
            "feed_forward.up_proj.weight",
            "feed_forward.down_proj.weight",
        ];

        for layer_idx in 0..self.config.num_hidden_layers {
            for component in &layer_components {
                let name = format!("layers.{layer_idx}.{component}");
                if !self.name_map.contains_key(&name) {
                    missing.push(name);
                }
            }
        }

        if missing.is_empty() { Ok(()) } else { Err(HfLoaderError::IncompleteTensors(missing)) }
    }

    /// Open a [`SafeTensorsReader`] from a model directory.
    ///
    /// Prefers sharded loading via `model.safetensors.index.json` if present,
    /// otherwise falls back to a single `model.safetensors` file, or the first
    /// `.safetensors` file found.
    fn open_reader(dir: &Path) -> Result<SafeTensorsReader> {
        // Sharded model?
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            debug!("Loading sharded safetensors from index");
            return Ok(SafeTensorsReader::from_sharded(dir, &index_path)?);
        }

        // Single model.safetensors?
        let single_path = dir.join("model.safetensors");
        if single_path.exists() {
            debug!("Loading single safetensors file");
            return Ok(SafeTensorsReader::from_file(&single_path)?);
        }

        // Any .safetensors file?
        let entries: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
            .collect();

        if let Some(entry) = entries.first() {
            debug!(path = %entry.path().display(), "Loading safetensors file");
            return Ok(SafeTensorsReader::from_file(entry.path())?);
        }

        Err(HfLoaderError::NoSafetensors(dir.display().to_string()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::Dtype as SafeDtype;
    use safetensors::tensor::TensorView;
    use tempfile::TempDir;

    /// Serialize tensors into safetensors bytes.
    fn make_safetensors(tensors: Vec<(&str, SafeDtype, Vec<usize>, &[u8])>) -> Vec<u8> {
        let views: Vec<(String, TensorView)> = tensors
            .into_iter()
            .map(|(name, dtype, shape, data)| {
                let view = TensorView::new(dtype, shape, data).unwrap();
                (name.to_string(), view)
            })
            .collect();
        safetensors::serialize(views, None).unwrap()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Create a mock HF model directory with a minimal 1-layer model.
    fn mock_hf_model_dir(dir: &Path) {
        // config.json
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 8,
            "vocab_size": 16,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0
        });
        std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config).unwrap())
            .unwrap();

        // Tensor data: all tiny shapes to keep tests fast
        let embed_data = f32_bytes(&vec![0.1; 16 * 4]); // [vocab=16, hidden=4]
        let norm_data = f32_bytes(&vec![1.0; 4]); // [hidden=4]
        let attn_proj = f32_bytes(&vec![0.5; 4 * 4]); // [4, 4]
        let ffn_gate = f32_bytes(&vec![0.3; 8 * 4]); // [inter=8, hidden=4]
        let ffn_down = f32_bytes(&vec![0.2; 4 * 8]); // [hidden=4, inter=8]

        let data = make_safetensors(vec![
            ("model.embed_tokens.weight", SafeDtype::F32, vec![16, 4], &embed_data),
            ("model.norm.weight", SafeDtype::F32, vec![4], &norm_data),
            ("lm_head.weight", SafeDtype::F32, vec![16, 4], &embed_data),
            ("model.layers.0.self_attn.q_proj.weight", SafeDtype::F32, vec![4, 4], &attn_proj),
            ("model.layers.0.self_attn.k_proj.weight", SafeDtype::F32, vec![4, 4], &attn_proj),
            ("model.layers.0.self_attn.v_proj.weight", SafeDtype::F32, vec![4, 4], &attn_proj),
            ("model.layers.0.self_attn.o_proj.weight", SafeDtype::F32, vec![4, 4], &attn_proj),
            ("model.layers.0.input_layernorm.weight", SafeDtype::F32, vec![4], &norm_data),
            ("model.layers.0.post_attention_layernorm.weight", SafeDtype::F32, vec![4], &norm_data),
            ("model.layers.0.mlp.gate_proj.weight", SafeDtype::F32, vec![8, 4], &ffn_gate),
            ("model.layers.0.mlp.up_proj.weight", SafeDtype::F32, vec![8, 4], &ffn_gate),
            ("model.layers.0.mlp.down_proj.weight", SafeDtype::F32, vec![4, 8], &ffn_down),
        ]);
        std::fs::write(dir.join("model.safetensors"), data).unwrap();
    }

    // -- Name mapping tests --

    #[test]
    fn map_hf_embed_tokens() {
        assert_eq!(map_hf_name("model.embed_tokens.weight"), "embed_tokens.weight");
    }

    #[test]
    fn map_hf_lm_head() {
        assert_eq!(map_hf_name("lm_head.weight"), "lm_head.weight");
    }

    #[test]
    fn map_hf_final_norm() {
        assert_eq!(map_hf_name("model.norm.weight"), "final_norm.weight");
    }

    #[test]
    fn map_hf_attention_projections() {
        assert_eq!(
            map_hf_name("model.layers.0.self_attn.q_proj.weight"),
            "layers.0.attention.q_proj.weight"
        );
        assert_eq!(
            map_hf_name("model.layers.5.self_attn.k_proj.weight"),
            "layers.5.attention.k_proj.weight"
        );
        assert_eq!(
            map_hf_name("model.layers.31.self_attn.v_proj.weight"),
            "layers.31.attention.v_proj.weight"
        );
        assert_eq!(
            map_hf_name("model.layers.0.self_attn.o_proj.weight"),
            "layers.0.attention.o_proj.weight"
        );
    }

    #[test]
    fn map_hf_norm_layers() {
        assert_eq!(
            map_hf_name("model.layers.0.input_layernorm.weight"),
            "layers.0.attention_norm.weight"
        );
        assert_eq!(
            map_hf_name("model.layers.0.post_attention_layernorm.weight"),
            "layers.0.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn map_hf_mlp_layers() {
        assert_eq!(
            map_hf_name("model.layers.0.mlp.gate_proj.weight"),
            "layers.0.feed_forward.gate_proj.weight"
        );
        assert_eq!(
            map_hf_name("model.layers.0.mlp.up_proj.weight"),
            "layers.0.feed_forward.up_proj.weight"
        );
        assert_eq!(
            map_hf_name("model.layers.0.mlp.down_proj.weight"),
            "layers.0.feed_forward.down_proj.weight"
        );
    }

    #[test]
    fn map_hf_unknown_passthrough() {
        assert_eq!(
            map_hf_name("model.layers.0.custom_component.weight"),
            "layers.0.custom_component.weight"
        );
        assert_eq!(map_hf_name("some_random_tensor"), "some_random_tensor");
    }

    // -- Config parsing tests --

    #[test]
    fn parse_config_json() {
        let json = r#"{
            "model_type": "phi3",
            "hidden_size": 3072,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "vocab_size": 32064,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0
        }"#;
        let config: HfModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, "phi3");
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.effective_kv_heads(), 8);
        assert_eq!(config.head_dim(), 96);
        assert_eq!(config.intermediate_size, 8192);
        assert_eq!(config.vocab_size, 32064);
    }

    #[test]
    fn parse_config_minimal() {
        let json = r#"{ "model_type": "llama" }"#;
        let config: HfModelConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, "llama");
        // Defaults
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.effective_kv_heads(), 32); // falls back to num_heads
    }

    // -- Loader integration tests --

    #[test]
    fn load_from_directory() {
        let dir = TempDir::new().unwrap();
        mock_hf_model_dir(dir.path());

        let loader = HfModelLoader::from_directory(dir.path()).unwrap();
        assert_eq!(*loader.architecture(), ModelArchitecture::Llama);
        assert_eq!(loader.config().hidden_size, 4);
        assert_eq!(loader.config().num_hidden_layers, 1);
    }

    #[test]
    fn tensor_names_are_canonical() {
        let dir = TempDir::new().unwrap();
        mock_hf_model_dir(dir.path());

        let loader = HfModelLoader::from_directory(dir.path()).unwrap();
        let names = loader.tensor_names();

        assert!(names.contains(&"embed_tokens.weight".to_string()));
        assert!(names.contains(&"final_norm.weight".to_string()));
        assert!(names.contains(&"lm_head.weight".to_string()));
        assert!(names.contains(&"layers.0.attention.q_proj.weight".to_string()));
        assert!(names.contains(&"layers.0.feed_forward.gate_proj.weight".to_string()));
    }

    #[test]
    fn load_tensor_by_internal_name() {
        let dir = TempDir::new().unwrap();
        mock_hf_model_dir(dir.path());

        let loader = HfModelLoader::from_directory(dir.path()).unwrap();
        let data = loader.load_tensor("layers.0.attention.q_proj.weight").unwrap();
        assert_eq!(data.len(), 16); // [4, 4] = 16 elements
    }

    #[test]
    fn load_embeddings() {
        let dir = TempDir::new().unwrap();
        mock_hf_model_dir(dir.path());

        let loader = HfModelLoader::from_directory(dir.path()).unwrap();
        let embed = loader.load_embeddings().unwrap();
        assert_eq!(embed.len(), 64); // [16, 4] = 64 elements
    }

    #[test]
    fn load_layer() {
        let dir = TempDir::new().unwrap();
        mock_hf_model_dir(dir.path());

        let loader = HfModelLoader::from_directory(dir.path()).unwrap();
        let layer = loader.load_layer(0).unwrap();

        // Should have attention (q,k,v,o) + norms (2) + ffn (gate, up, down) = 9
        assert_eq!(layer.len(), 9);
        assert!(layer.contains_key("layers.0.attention.q_proj.weight"));
        assert!(layer.contains_key("layers.0.feed_forward.down_proj.weight"));
    }

    #[test]
    fn load_layer_nonexistent() {
        let dir = TempDir::new().unwrap();
        mock_hf_model_dir(dir.path());

        let loader = HfModelLoader::from_directory(dir.path()).unwrap();
        let err = loader.load_layer(99).unwrap_err();
        assert!(matches!(err, HfLoaderError::TensorNotFound(_)));
    }

    #[test]
    fn validate_completeness_ok() {
        let dir = TempDir::new().unwrap();
        mock_hf_model_dir(dir.path());

        let loader = HfModelLoader::from_directory(dir.path()).unwrap();
        loader.validate_completeness().unwrap();
    }

    #[test]
    fn validate_completeness_missing_tensors() {
        let dir = TempDir::new().unwrap();

        // Config says 1 layer, but we only provide embeddings
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 8,
            "vocab_size": 16
        });
        std::fs::write(dir.path().join("config.json"), serde_json::to_string(&config).unwrap())
            .unwrap();

        let embed_data = f32_bytes(&vec![0.1; 64]);
        let data = make_safetensors(vec![(
            "model.embed_tokens.weight",
            SafeDtype::F32,
            vec![16, 4],
            &embed_data,
        )]);
        std::fs::write(dir.path().join("model.safetensors"), data).unwrap();

        let loader = HfModelLoader::from_directory(dir.path()).unwrap();
        let err = loader.validate_completeness().unwrap_err();
        match err {
            HfLoaderError::IncompleteTensors(missing) => {
                assert!(missing.contains(&"final_norm.weight".to_string()));
                assert!(missing.contains(&"layers.0.attention.q_proj.weight".to_string()));
            }
            other => panic!("expected IncompleteTensors, got: {other}"),
        }
    }

    #[test]
    fn error_missing_config_json() {
        let dir = TempDir::new().unwrap();

        // Safetensors but no config.json
        let data =
            make_safetensors(vec![("weight", SafeDtype::F32, vec![1], &1.0f32.to_le_bytes())]);
        std::fs::write(dir.path().join("model.safetensors"), data).unwrap();

        let err = HfModelLoader::from_directory(dir.path()).unwrap_err();
        assert!(matches!(err, HfLoaderError::ConfigNotFound(_)));
    }

    #[test]
    fn error_missing_safetensors() {
        let dir = TempDir::new().unwrap();

        // Config but no safetensors
        let config = serde_json::json!({ "model_type": "llama" });
        std::fs::write(dir.path().join("config.json"), serde_json::to_string(&config).unwrap())
            .unwrap();

        let err = HfModelLoader::from_directory(dir.path()).unwrap_err();
        assert!(matches!(err, HfLoaderError::NoSafetensors(_)));
    }

    #[test]
    fn error_tensor_not_found() {
        let dir = TempDir::new().unwrap();
        mock_hf_model_dir(dir.path());

        let loader = HfModelLoader::from_directory(dir.path()).unwrap();
        let err = loader.load_tensor("nonexistent.weight").unwrap_err();
        assert!(matches!(err, HfLoaderError::TensorNotFound(_)));
    }

    #[test]
    fn with_explicit_config() {
        let dir = TempDir::new().unwrap();
        mock_hf_model_dir(dir.path());

        let config = HfModelConfig {
            model_type: "phi3".to_string(),
            hidden_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: Some(2),
            intermediate_size: 8,
            vocab_size: 16,
            max_position_embeddings: Some(128),
            rope_theta: Some(10000.0),
        };

        let loader = HfModelLoader::with_config(dir.path(), config).unwrap();
        assert_eq!(*loader.architecture(), ModelArchitecture::Phi);
        assert_eq!(loader.config().hidden_size, 4);
    }

    #[test]
    fn architecture_defaults_accessible() {
        let dir = TempDir::new().unwrap();
        mock_hf_model_dir(dir.path());

        let loader = HfModelLoader::from_directory(dir.path()).unwrap();
        let defaults = loader.architecture_defaults();
        assert_eq!(defaults.architecture, ModelArchitecture::Llama);
        assert!(defaults.vocab_size > 0);
    }

    #[test]
    fn sharded_model_directory() {
        let dir = TempDir::new().unwrap();

        // config.json
        let config = serde_json::json!({
            "model_type": "phi3",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "vocab_size": 16
        });
        std::fs::write(dir.path().join("config.json"), serde_json::to_string(&config).unwrap())
            .unwrap();

        // Two shards
        let embed_data = f32_bytes(&vec![0.1; 64]);

        let shard1 = make_safetensors(vec![(
            "model.embed_tokens.weight",
            SafeDtype::F32,
            vec![16, 4],
            &embed_data,
        )]);
        let shard2 =
            make_safetensors(vec![("lm_head.weight", SafeDtype::F32, vec![16, 4], &embed_data)]);

        std::fs::write(dir.path().join("model-00001-of-00002.safetensors"), &shard1).unwrap();
        std::fs::write(dir.path().join("model-00002-of-00002.safetensors"), &shard2).unwrap();

        let index = serde_json::json!({
            "metadata": { "total_size": 512 },
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors"
            }
        });
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&index).unwrap(),
        )
        .unwrap();

        let loader = HfModelLoader::from_directory(dir.path()).unwrap();
        assert_eq!(*loader.architecture(), ModelArchitecture::Phi);
        assert!(loader.tensor_names().contains(&"embed_tokens.weight".to_string()));
        assert!(loader.tensor_names().contains(&"lm_head.weight".to_string()));

        let embed = loader.load_embeddings().unwrap();
        assert_eq!(embed.len(), 64);
    }
}
