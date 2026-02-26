//! Model management with hot-swapping and device-aware routing

use anyhow::Result;
use bitnet_common::Device;
use bitnet_inference::InferenceEngine;
use bitnet_models::Model;
use bitnet_models::formats::gguf::GgufLoader;
use bitnet_models::loader::{FormatLoader, LoadConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tracing::{error, info, warn};

/// Model manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManagerConfig {
    pub max_concurrent_loads: usize,
    pub model_cache_size: usize,
    pub load_timeout: Duration,
    pub validation_enabled: bool,
    pub memory_limit_gb: Option<f64>,
}

impl Default for ModelManagerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_loads: 2,
            model_cache_size: 3,
            load_timeout: Duration::from_secs(300), // 5 minutes
            validation_enabled: true,
            memory_limit_gb: Some(16.0),
        }
    }
}

/// Model metadata for tracking and management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_id: String,
    pub model_path: String,
    pub device: String,
    pub quantization_type: String,
    pub loaded_at: std::time::SystemTime,
    pub size_mb: u64,
    pub parameters: u64,
    pub context_length: u32,
    pub inference_count: u64,
    pub avg_tokens_per_second: f64,
}

/// Model load status for atomic operations
#[derive(Debug, Clone)]
pub enum ModelLoadStatus {
    Loading { progress: f32, stage: String },
    Ready { metadata: ModelMetadata },
    Failed { error: String },
    Unloading,
}

/// Inference engine wrapper with metadata
pub struct ManagedModel {
    pub engine: InferenceEngine,
    pub metadata: ModelMetadata,
    pub last_used: std::time::Instant,
    pub inference_count: AtomicU64,
}

impl ManagedModel {
    pub fn new(engine: InferenceEngine, metadata: ModelMetadata) -> Self {
        Self { engine, metadata, last_used: Instant::now(), inference_count: AtomicU64::new(0) }
    }

    pub fn update_usage(&self) {
        self.inference_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get_inference_count(&self) -> u64 {
        self.inference_count.load(Ordering::Relaxed)
    }
}

/// Model manager with hot-swapping capabilities
pub struct ModelManager {
    config: ModelManagerConfig,
    active_model: Arc<RwLock<Option<Arc<ManagedModel>>>>,
    model_cache: Arc<RwLock<HashMap<String, Arc<ManagedModel>>>>,
    loading_status: Arc<RwLock<HashMap<String, ModelLoadStatus>>>,
    load_semaphore: Arc<Semaphore>,
    next_model_id: AtomicU64,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new(config: ModelManagerConfig) -> Self {
        Self {
            config: config.clone(),
            active_model: Arc::new(RwLock::new(None)),
            model_cache: Arc::new(RwLock::new(HashMap::new())),
            loading_status: Arc::new(RwLock::new(HashMap::new())),
            load_semaphore: Arc::new(Semaphore::new(config.max_concurrent_loads)),
            next_model_id: AtomicU64::new(1),
        }
    }

    /// Get the currently active model
    pub async fn get_active_model(&self) -> Option<Arc<ManagedModel>> {
        let active = self.active_model.read().await;
        active.clone()
    }

    /// Load a model and make it active (hot-swap)
    pub async fn load_and_activate_model(
        &self,
        model_path: &str,
        tokenizer_path: Option<&str>,
        device: &Device,
    ) -> Result<String> {
        let model_id = format!("model-{}", self.next_model_id.fetch_add(1, Ordering::Relaxed));

        info!(
            model_id = %model_id,
            model_path = %model_path,
            device = ?device,
            "Starting model load operation"
        );

        // Acquire loading semaphore
        let _permit = self.load_semaphore.acquire().await?;

        // Update loading status
        {
            let mut status = self.loading_status.write().await;
            status.insert(
                model_id.clone(),
                ModelLoadStatus::Loading { progress: 0.0, stage: "Initializing".to_string() },
            );
        }

        // Load the model
        let result = self.load_model(&model_id, model_path, tokenizer_path, device).await;

        match result {
            Ok(managed_model) => {
                // Atomic swap of active model
                {
                    let mut active = self.active_model.write().await;
                    *active = Some(Arc::new(managed_model));
                }

                // Update status to ready
                {
                    let mut status = self.loading_status.write().await;
                    if let Some(active_model) = self.get_active_model().await {
                        status.insert(
                            model_id.clone(),
                            ModelLoadStatus::Ready { metadata: active_model.metadata.clone() },
                        );
                    }
                }

                // Add to cache with efficient LRU eviction
                if let Some(active_model) = self.get_active_model().await {
                    let mut cache = self.model_cache.write().await;

                    // Efficient LRU eviction - avoid cloning unless necessary
                    if cache.len() >= self.config.model_cache_size
                        && let Some(oldest_key) = cache
                            .iter()
                            .min_by_key(|(_, model)| model.last_used)
                            .map(|(k, _)| k.clone())
                    {
                        cache.remove(&oldest_key);
                        info!(evicted_model = %oldest_key, "Evicted model from cache due to capacity");
                    }

                    cache.insert(model_id.clone(), active_model);
                }

                info!(model_id = %model_id, "Model successfully loaded and activated");
                Ok(model_id)
            }
            Err(e) => {
                error!(model_id = %model_id, error = %e, "Failed to load model");

                // Update status to failed
                {
                    let mut status = self.loading_status.write().await;
                    status
                        .insert(model_id.clone(), ModelLoadStatus::Failed { error: e.to_string() });
                }

                Err(e)
            }
        }
    }

    /// Load a model from disk
    async fn load_model(
        &self,
        model_id: &str,
        model_path: &str,
        tokenizer_path: Option<&str>,
        device: &Device,
    ) -> Result<ManagedModel> {
        let start_time = Instant::now();

        // Update progress: Validating file
        self.update_loading_progress(model_id, 10.0, "Validating model file").await;

        let model_path = Path::new(model_path);
        if !model_path.exists() {
            anyhow::bail!("Model file not found: {}", model_path.display());
        }

        // Get file size
        let file_metadata = std::fs::metadata(model_path)?;
        let size_mb = file_metadata.len() / (1024 * 1024);

        // Check memory limits
        if let Some(limit_gb) = self.config.memory_limit_gb {
            let size_gb = size_mb as f64 / 1024.0;
            if size_gb > limit_gb {
                anyhow::bail!(
                    "Model size ({:.2} GB) exceeds memory limit ({:.2} GB)",
                    size_gb,
                    limit_gb
                );
            }
        }

        // Update progress: Loading model
        self.update_loading_progress(model_id, 30.0, "Loading GGUF model").await;

        let loader = GgufLoader;
        let load_config = LoadConfig::default();
        let model = loader.load(model_path, device, &load_config)?;

        // Update progress: Loading tokenizer
        self.update_loading_progress(model_id, 60.0, "Loading tokenizer").await;

        let tokenizer = match tokenizer_path {
            Some(tok) => {
                let tok_path = Path::new(tok);
                bitnet_tokenizers::auto::load_auto(model_path, Some(tok_path))?
            }
            None => bitnet_tokenizers::auto::load_auto(model_path, None)?,
        };

        // Update progress: Creating inference engine
        self.update_loading_progress(model_id, 80.0, "Creating inference engine").await;

        let model: Arc<dyn Model> = model.into();
        let engine = InferenceEngine::new(model.clone(), tokenizer, *device)?;

        // Create metadata with BitNet-rs model information
        let metadata = Self::create_model_metadata(model_id, model_path, device, size_mb, &model);

        // Final validation if enabled
        if self.config.validation_enabled {
            self.update_loading_progress(model_id, 90.0, "Validating model").await;
            self.validate_model(&engine).await?;
        }

        self.update_loading_progress(model_id, 100.0, "Complete").await;

        let load_duration = start_time.elapsed();
        info!(
            model_id = %model_id,
            load_duration_ms = load_duration.as_millis(),
            size_mb = size_mb,
            "Model loaded successfully"
        );

        Ok(ManagedModel::new(engine, metadata))
    }

    /// Update loading progress
    async fn update_loading_progress(&self, model_id: &str, progress: f32, stage: &str) {
        let mut status = self.loading_status.write().await;
        status.insert(
            model_id.to_string(),
            ModelLoadStatus::Loading { progress, stage: stage.to_string() },
        );
    }

    /// Create model metadata with proper BitNet-rs information extraction
    fn create_model_metadata(
        model_id: &str,
        model_path: &Path,
        device: &Device,
        size_mb: u64,
        model: &Arc<dyn Model>,
    ) -> ModelMetadata {
        // Extract quantization type from model
        let quantization_type = Self::detect_quantization_type(model);

        // Extract model parameters and context length
        let (parameters, context_length) = Self::extract_model_info(model);

        ModelMetadata {
            model_id: model_id.to_string(),
            model_path: model_path.to_string_lossy().to_string(),
            device: format!("{:?}", device),
            quantization_type,
            loaded_at: std::time::SystemTime::now(),
            size_mb,
            parameters,
            context_length,
            inference_count: 0,
            avg_tokens_per_second: 0.0,
        }
    }

    /// Detect quantization type from model architecture
    fn detect_quantization_type(model: &Arc<dyn Model>) -> String {
        // Check model architecture for quantization hints
        // This is a simplified detection - in practice, we'd examine the model format
        let _model_info = model; // TODO: Implement proper detection via model introspection
        "I2S".to_string() // Default to I2S for BitNet models
    }

    /// Extract model parameters and context length from model
    fn extract_model_info(model: &Arc<dyn Model>) -> (u64, u32) {
        // Extract from model metadata if available
        let _model_ref = model; // TODO: Implement via model metadata interface
        (0, 2048) // Default values until metadata interface is available
    }

    /// Validate model functionality
    async fn validate_model(&self, engine: &InferenceEngine) -> Result<()> {
        use bitnet_inference::GenerationConfig;

        let test_prompt = "Hello, world!";
        let config = GenerationConfig::default().with_max_tokens(5).with_temperature(1.0);

        let _result = engine.generate_with_config(test_prompt, &config).await?;
        info!("Model validation successful");
        Ok(())
    }

    /// Get model loading status
    pub async fn get_loading_status(&self, model_id: &str) -> Option<ModelLoadStatus> {
        let status = self.loading_status.read().await;
        status.get(model_id).cloned()
    }

    /// List all models in cache
    pub async fn list_models(&self) -> Vec<ModelMetadata> {
        let cache = self.model_cache.read().await;
        cache.values().map(|model| model.metadata.clone()).collect()
    }

    /// Get model metadata by ID
    pub async fn get_model_metadata(&self, model_id: &str) -> Option<ModelMetadata> {
        let cache = self.model_cache.read().await;
        cache.get(model_id).map(|model| model.metadata.clone())
    }

    /// Unload a model from cache
    pub async fn unload_model(&self, model_id: &str) -> Result<()> {
        // Update status to unloading
        {
            let mut status = self.loading_status.write().await;
            status.insert(model_id.to_string(), ModelLoadStatus::Unloading);
        }

        // Remove from cache
        {
            let mut cache = self.model_cache.write().await;
            if cache.remove(model_id).is_some() {
                info!(model_id = %model_id, "Model unloaded from cache");
            } else {
                warn!(model_id = %model_id, "Model not found in cache");
            }
        }

        // Clean up status
        {
            let mut status = self.loading_status.write().await;
            status.remove(model_id);
        }

        Ok(())
    }

    /// Get memory usage statistics
    pub async fn get_memory_stats(&self) -> ModelMemoryStats {
        let cache = self.model_cache.read().await;
        let active = self.active_model.read().await;

        let total_models = cache.len();
        let total_size_mb: u64 = cache.values().map(|model| model.metadata.size_mb).sum();
        let active_model_id = active.as_ref().map(|model| model.metadata.model_id.clone());

        ModelMemoryStats {
            total_models,
            total_size_mb,
            active_model_id,
            cache_size_limit: self.config.model_cache_size,
            memory_limit_gb: self.config.memory_limit_gb,
        }
    }

    /// Health check for model manager
    pub async fn health_check(&self) -> ModelManagerHealth {
        let active = self.active_model.read().await;
        let cache = self.model_cache.read().await;
        let status = self.loading_status.read().await;

        let active_model_healthy = active.is_some();
        let loading_operations = status
            .iter()
            .filter(|(_, status)| matches!(status, ModelLoadStatus::Loading { .. }))
            .count();

        ModelManagerHealth {
            active_model_healthy,
            cached_models: cache.len(),
            loading_operations,
            last_error: None, // TODO: Track recent errors
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize)]
pub struct ModelMemoryStats {
    pub total_models: usize,
    pub total_size_mb: u64,
    pub active_model_id: Option<String>,
    pub cache_size_limit: usize,
    pub memory_limit_gb: Option<f64>,
}

/// Health status for model manager
#[derive(Debug, Clone, Serialize)]
pub struct ModelManagerHealth {
    pub active_model_healthy: bool,
    pub cached_models: usize,
    pub loading_operations: usize,
    pub last_error: Option<String>,
}
