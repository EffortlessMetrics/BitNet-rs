//! Model loader with comprehensive tracing and failure mode logging

use anyhow::{Context, Result, anyhow};
use bitnet_models::{Model, formats::{ModelFormat, gguf, safetensors}};
use bitnet_tokenizers::Tokenizer;
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, debug, warn};
use crate::engine::inspect_model;

/// Model loading metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderMetadata {
    /// Detected or specified format
    pub format: ModelFormat,
    /// How format was determined
    pub format_source: String,
    /// Tokenizer source
    pub tokenizer_source: String,
    /// Scoring policy
    pub scoring_policy: ScoringPolicy,
    /// Number of tensors loaded
    pub tensors_loaded: usize,
    /// Ignored/unmapped tensors
    pub ignored_tensors: Vec<String>,
    /// Model configuration
    pub model_config: ModelConfigInfo,
}

/// Scoring policy for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringPolicy {
    pub add_bos: bool,
    pub append_eos: bool,
    pub mask_pad: bool,
}

impl Default for ScoringPolicy {
    fn default() -> Self {
        Self {
            add_bos: true,
            append_eos: false,
            mask_pad: true,
        }
    }
}

/// Model configuration info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigInfo {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub context_length: usize,
}

/// Unified model loader with tracing
pub struct ModelLoader {
    /// Path to model file
    model_path: PathBuf,
    /// Optional tokenizer path
    tokenizer_path: Option<PathBuf>,
    /// Format override
    format_override: Option<ModelFormat>,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            tokenizer_path: None,
            format_override: None,
        }
    }
    
    /// Set tokenizer path
    pub fn with_tokenizer(mut self, path: impl AsRef<Path>) -> Self {
        self.tokenizer_path = Some(path.as_ref().to_path_buf());
        self
    }
    
    /// Override format detection
    pub fn with_format(mut self, format: ModelFormat) -> Self {
        self.format_override = Some(format);
        self
    }
    
    /// Load model with comprehensive tracing
    pub fn load(&self) -> Result<(Arc<dyn Model>, Arc<dyn Tokenizer>, LoaderMetadata)> {
        info!("Loading model from: {}", self.model_path.display());
        
        // Detect or use override format
        let (format, format_source) = if let Some(fmt) = self.format_override {
            info!("Using format override: {}", fmt.name());
            (fmt, "manual_override".to_string())
        } else {
            let detected = ModelFormat::detect_from_path(&self.model_path)
                .or_else(|_| ModelFormat::detect_from_header(&self.model_path))
                .context("Failed to detect model format")?;
            info!("Auto-detected format: {} from path/header", detected.name());
            (detected, "auto_detection".to_string())
        };
        
        // Load model based on format
        let (model, tokenizer, tokenizer_source, ignored_tensors) = match format {
            ModelFormat::Gguf => self.load_gguf()?,
            ModelFormat::SafeTensors => self.load_safetensors()?,
        };
        
        // Extract model configuration
        let config = model.config();
        let model_config = ModelConfigInfo {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            num_heads: config.num_attention_heads,
            context_length: config.max_position_embeddings,
        };
        
        // Determine scoring policy
        let scoring_policy = self.determine_scoring_policy(&model, &tokenizer);
        
        // Count loaded tensors
        let tensors_loaded = model.tensors().len();
        
        // Create metadata
        let metadata = LoaderMetadata {
            format,
            format_source,
            tokenizer_source,
            scoring_policy,
            tensors_loaded,
            ignored_tensors,
            model_config,
        };
        
        // Log decision trace
        info!(
            format = %metadata.format.name(),
            tokenizer = %metadata.tokenizer_source,
            policy = ?metadata.scoring_policy,
            tensors = metadata.tensors_loaded,
            ignored = metadata.ignored_tensors.len(),
            "Model loaded successfully"
        );
        
        if !metadata.ignored_tensors.is_empty() {
            warn!("Ignored {} tensors during loading:", metadata.ignored_tensors.len());
            for tensor in &metadata.ignored_tensors[..5.min(metadata.ignored_tensors.len())] {
                warn!("  - {}", tensor);
            }
            if metadata.ignored_tensors.len() > 5 {
                warn!("  ... and {} more", metadata.ignored_tensors.len() - 5);
            }
        }
        
        Ok((model, tokenizer, metadata))
    }
    
    fn load_gguf(&self) -> Result<(Arc<dyn Model>, Arc<dyn Tokenizer>, String, Vec<String>)> {
        debug!("Loading GGUF model");
        
        // Early header validation before heavy allocations
        let model_info = inspect_model(&self.model_path)
            .map_err(|e| anyhow!("GGUF header validation failed: {}", e))?;
        info!(
            "GGUF header validated: version={}, tensors={}, kvs={}",
            model_info.version(),
            model_info.n_tensors(),
            model_info.n_kv()
        );
        
        let loader = gguf::GgufLoader::new(&self.model_path)?;
        let model = loader.load_model()?;
        let ignored = loader.get_ignored_tensors();
        
        // Check for embedded tokenizer
        let (tokenizer, source) = if loader.has_embedded_tokenizer() {
            debug!("Using embedded GGUF tokenizer");
            let tok = loader.load_tokenizer()?;
            (tok, "embedded-gguf".to_string())
        } else if let Some(path) = &self.tokenizer_path {
            debug!("Loading external tokenizer from: {}", path.display());
            let tok = self.load_external_tokenizer(path)?;
            (tok, "external-json".to_string())
        } else {
            return Err(anyhow!("No tokenizer found (not embedded, no external path)"));
        };
        
        Ok((Arc::new(model), Arc::new(tokenizer), source, ignored))
    }
    
    fn load_safetensors(&self) -> Result<(Arc<dyn Model>, Arc<dyn Tokenizer>, String, Vec<String>)> {
        debug!("Loading SafeTensors model");
        
        let loader = safetensors::SafeTensorsLoader::new(&self.model_path)?;
        let model = loader.load_model()?;
        let ignored = loader.get_ignored_tensors();
        
        // SafeTensors always requires external tokenizer
        let path = self.tokenizer_path.as_ref()
            .ok_or_else(|| anyhow!("SafeTensors requires external tokenizer path"))?;
        
        debug!("Loading external tokenizer from: {}", path.display());
        let tokenizer = self.load_external_tokenizer(path)?;
        
        Ok((Arc::new(model), Arc::new(tokenizer), "external-json".to_string(), ignored))
    }
    
    fn load_external_tokenizer(&self, path: &Path) -> Result<Box<dyn Tokenizer>> {
        // Load tokenizer.json or HF tokenizer format
        bitnet_tokenizers::load_tokenizer(path)
            .context("Failed to load external tokenizer")
    }
    
    fn determine_scoring_policy(&self, model: &Arc<dyn Model>, tokenizer: &Arc<dyn Tokenizer>) -> ScoringPolicy {
        // Determine based on model type and tokenizer config
        let config = model.config();
        
        // Default policy with some model-specific overrides
        let mut policy = ScoringPolicy::default();
        
        // Check for special tokens in tokenizer
        if let Some(bos) = tokenizer.bos_token_id() {
            debug!("Tokenizer has BOS token: {}", bos);
            policy.add_bos = true;
        }
        
        if let Some(eos) = tokenizer.eos_token_id() {
            debug!("Tokenizer has EOS token: {}", eos);
            // Generally don't append EOS for perplexity evaluation
            policy.append_eos = false;
        }
        
        // Model-specific overrides
        if config.model_type.contains("gpt2") {
            policy.add_bos = false;  // GPT-2 doesn't use BOS
        }
        
        policy
    }
}

/// Helper to parse model format from string
pub fn parse_model_format(s: &str) -> Result<Option<ModelFormat>> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(None),
        "gguf" => Ok(Some(ModelFormat::Gguf)),
        "safetensors" => Ok(Some(ModelFormat::SafeTensors)),
        _ => Err(anyhow!("Unknown format: {}. Use: auto, gguf, or safetensors", s)),
    }
}