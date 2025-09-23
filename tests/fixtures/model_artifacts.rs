//! Model artifacts and GGUF test fixtures
//!
//! Provides realistic test models, GGUF files, and tokenizer artifacts for comprehensive
//! BitNet model integration testing.

use bitnet_common::{Device, Result, BitNetError, ModelError};
use bitnet_models::{ModelLoader, FormatLoader};
use bitnet_tokenizers::UniversalTokenizer;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use super::{TestEnvironmentConfig, TestTier};

/// Test model configuration for fixture management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestModelConfig {
    pub model_id: String,
    pub model_path: PathBuf,
    pub tokenizer_path: Option<PathBuf>,
    pub quantization_type: String,
    pub parameter_count: u64,
    pub vocab_size: u32,
    pub context_length: u32,
    pub expected_architecture: String,
}

/// Mock model implementation for fast testing
#[derive(Debug, Clone)]
pub struct MockBitNetModel {
    pub config: TestModelConfig,
    pub mock_tensors: HashMap<String, MockTensor>,
    pub mock_metadata: MockMetadata,
}

#[derive(Debug, Clone)]
pub struct MockTensor {
    pub name: String,
    pub shape: Vec<u32>,
    pub dtype: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct MockMetadata {
    pub architecture: String,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub attention_heads: u32,
    pub head_dim: u32,
    pub parameter_count: u64,
}

impl MockBitNetModel {
    /// Create a small mock BitNet model for testing
    pub fn create_small_bitnet() -> Self {
        let config = TestModelConfig {
            model_id: "mock/bitnet-small".to_string(),
            model_path: PathBuf::from("tests/fixtures/mock_bitnet_small.gguf"),
            tokenizer_path: Some(PathBuf::from("tests/fixtures/mock_tokenizer.json")),
            quantization_type: "I2_S".to_string(),
            parameter_count: 1_000_000, // 1M parameters
            vocab_size: 32000,
            context_length: 2048,
            expected_architecture: "BitNet-b1.58".to_string(),
        };

        let mock_metadata = MockMetadata {
            architecture: "BitNet-b1.58".to_string(),
            vocab_size: 32000,
            hidden_size: 512,
            num_layers: 8,
            attention_heads: 8,
            head_dim: 64,
            parameter_count: 1_000_000,
        };

        let mut mock_tensors = HashMap::new();

        // Add essential model tensors
        mock_tensors.insert("embed_tokens.weight".to_string(), MockTensor {
            name: "embed_tokens.weight".to_string(),
            shape: vec![32000, 512],
            dtype: "I2_S".to_string(),
            size_bytes: 32000 * 512 * 2 / 8, // 2-bit quantized
        });

        for layer in 0..8 {
            // Attention weights
            mock_tensors.insert(format!("layers.{}.self_attn.q_proj.weight", layer), MockTensor {
                name: format!("layers.{}.self_attn.q_proj.weight", layer),
                shape: vec![512, 512],
                dtype: "I2_S".to_string(),
                size_bytes: 512 * 512 * 2 / 8,
            });

            mock_tensors.insert(format!("layers.{}.self_attn.k_proj.weight", layer), MockTensor {
                name: format!("layers.{}.self_attn.k_proj.weight", layer),
                shape: vec![512, 512],
                dtype: "I2_S".to_string(),
                size_bytes: 512 * 512 * 2 / 8,
            });

            mock_tensors.insert(format!("layers.{}.self_attn.v_proj.weight", layer), MockTensor {
                name: format!("layers.{}.self_attn.v_proj.weight", layer),
                shape: vec![512, 512],
                dtype: "I2_S".to_string(),
                size_bytes: 512 * 512 * 2 / 8,
            });

            // Feed-forward weights
            mock_tensors.insert(format!("layers.{}.mlp.gate_proj.weight", layer), MockTensor {
                name: format!("layers.{}.mlp.gate_proj.weight", layer),
                shape: vec![1024, 512],
                dtype: "I2_S".to_string(),
                size_bytes: 1024 * 512 * 2 / 8,
            });

            mock_tensors.insert(format!("layers.{}.mlp.up_proj.weight", layer), MockTensor {
                name: format!("layers.{}.mlp.up_proj.weight", layer),
                shape: vec![1024, 512],
                dtype: "I2_S".to_string(),
                size_bytes: 1024 * 512 * 2 / 8,
            });

            mock_tensors.insert(format!("layers.{}.mlp.down_proj.weight", layer), MockTensor {
                name: format!("layers.{}.mlp.down_proj.weight", layer),
                shape: vec![512, 1024],
                dtype: "I2_S".to_string(),
                size_bytes: 512 * 1024 * 2 / 8,
            });
        }

        // Output layer
        mock_tensors.insert("lm_head.weight".to_string(), MockTensor {
            name: "lm_head.weight".to_string(),
            shape: vec![32000, 512],
            dtype: "I2_S".to_string(),
            size_bytes: 32000 * 512 * 2 / 8,
        });

        Self {
            config,
            mock_tensors,
            mock_metadata,
        }
    }

    /// Create a large mock model for memory testing
    pub fn create_large_bitnet() -> Self {
        let mut model = Self::create_small_bitnet();
        model.config.model_id = "mock/bitnet-large".to_string();
        model.config.parameter_count = 3_000_000_000; // 3B parameters
        model.config.model_path = PathBuf::from("tests/fixtures/mock_bitnet_large.gguf");

        model.mock_metadata.parameter_count = 3_000_000_000;
        model.mock_metadata.hidden_size = 2048;
        model.mock_metadata.num_layers = 24;
        model.mock_metadata.attention_heads = 16;
        model.mock_metadata.head_dim = 128;

        model
    }

    /// Generate mock GGUF file content
    pub fn generate_mock_gguf(&self) -> Vec<u8> {
        let mut buffer = Vec::new();

        // GGUF magic number
        buffer.extend_from_slice(b"GGUF");

        // Version (3)
        buffer.extend_from_slice(&3u32.to_le_bytes());

        // Tensor count
        buffer.extend_from_slice(&(self.mock_tensors.len() as u64).to_le_bytes());

        // KV metadata count (simplified)
        buffer.extend_from_slice(&8u64.to_le_bytes());

        // Mock metadata
        self.add_kv_pair(&mut buffer, "general.architecture", &self.mock_metadata.architecture);
        self.add_kv_pair(&mut buffer, "general.name", &self.config.model_id);
        self.add_kv_pair(&mut buffer, "llama.vocab_size", &self.mock_metadata.vocab_size);
        self.add_kv_pair(&mut buffer, "llama.embedding_length", &self.mock_metadata.hidden_size);
        self.add_kv_pair(&mut buffer, "llama.block_count", &self.mock_metadata.num_layers);
        self.add_kv_pair(&mut buffer, "llama.attention.head_count", &self.mock_metadata.attention_heads);
        self.add_kv_pair(&mut buffer, "llama.attention.head_count_kv", &self.mock_metadata.attention_heads);
        self.add_kv_pair(&mut buffer, "llama.context_length", &self.config.context_length);

        // Tensor info section
        for tensor in self.mock_tensors.values() {
            // Tensor name
            let name_bytes = tensor.name.as_bytes();
            buffer.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            buffer.extend_from_slice(name_bytes);

            // Dimensions
            buffer.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for dim in &tensor.shape {
                buffer.extend_from_slice(&(*dim as u64).to_le_bytes());
            }

            // Type (I2_S = 24)
            buffer.extend_from_slice(&24u32.to_le_bytes());

            // Offset (mock)
            buffer.extend_from_slice(&(buffer.len() as u64 + 1000).to_le_bytes());
        }

        // Alignment to 32 bytes
        while buffer.len() % 32 != 0 {
            buffer.push(0);
        }

        // Mock tensor data (zeros)
        let total_tensor_size: u64 = self.mock_tensors.values().map(|t| t.size_bytes).sum();
        buffer.resize(buffer.len() + total_tensor_size as usize, 0);

        buffer
    }

    fn add_kv_pair<T: Serialize>(&self, buffer: &mut Vec<u8>, key: &str, value: &T) {
        // Key
        let key_bytes = key.as_bytes();
        buffer.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        buffer.extend_from_slice(key_bytes);

        // Value type and data (simplified - assumes u32 for now)
        buffer.extend_from_slice(&4u32.to_le_bytes()); // Type: UINT32

        if let Ok(serialized) = bincode::serialize(value) {
            if serialized.len() >= 4 {
                buffer.extend_from_slice(&serialized[..4]);
            } else {
                buffer.extend_from_slice(&[0u8; 4]);
            }
        } else {
            buffer.extend_from_slice(&[0u8; 4]);
        }
    }
}

/// Real model fixtures for production testing
pub struct RealModelFixtures {
    pub model_configs: HashMap<String, TestModelConfig>,
    pub cached_models: HashMap<String, PathBuf>,
    pub test_env_config: TestEnvironmentConfig,
}

impl RealModelFixtures {
    /// Create fixtures for real BitNet models
    pub fn new(config: &TestEnvironmentConfig) -> Self {
        let mut model_configs = HashMap::new();

        // Microsoft BitNet 2B model configuration
        model_configs.insert("bitnet-2b".to_string(), TestModelConfig {
            model_id: "microsoft/bitnet-b1.58-2B-4T-gguf".to_string(),
            model_path: PathBuf::from("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"),
            tokenizer_path: Some(PathBuf::from("models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json")),
            quantization_type: "I2_S".to_string(),
            parameter_count: 2_000_000_000,
            vocab_size: 128256, // LLaMA-3 tokenizer
            context_length: 2048,
            expected_architecture: "BitNet-b1.58".to_string(),
        });

        Self {
            model_configs,
            cached_models: HashMap::new(),
            test_env_config: config.clone(),
        }
    }

    /// Get model configuration by ID
    pub fn get_model_config(&self, model_id: &str) -> Option<&TestModelConfig> {
        self.model_configs.get(model_id)
    }

    /// Download and cache model if needed
    pub async fn ensure_model_available(&mut self, model_id: &str) -> Result<PathBuf> {
        if let Some(cached_path) = self.cached_models.get(model_id) {
            if cached_path.exists() {
                return Ok(cached_path.clone());
            }
        }

        let config = self.get_model_config(model_id)
            .ok_or_else(|| BitNetError::Model(ModelError::NotFound {
                path: model_id.to_string()
            }))?;

        // Check if model exists at expected path
        if config.model_path.exists() {
            self.cached_models.insert(model_id.to_string(), config.model_path.clone());
            return Ok(config.model_path.clone());
        }

        // For CI or when model not found, skip gracefully
        if std::env::var("CI").is_ok() && !config.model_path.exists() {
            return Err(BitNetError::Model(ModelError::NotFound {
                path: format!("Model {} not available in CI environment. Set BITNET_GGUF or use download-model xtask.", model_id)
            }));
        }

        Err(BitNetError::Model(ModelError::NotFound {
            path: config.model_path.display().to_string()
        }))
    }
}

/// Main model fixtures manager
pub struct ModelFixtures {
    pub mock_models: HashMap<String, MockBitNetModel>,
    pub real_models: RealModelFixtures,
    pub config: TestEnvironmentConfig,
}

impl ModelFixtures {
    pub fn new(config: &TestEnvironmentConfig) -> Self {
        let mut mock_models = HashMap::new();

        // Create different sized mock models
        mock_models.insert("small".to_string(), MockBitNetModel::create_small_bitnet());
        mock_models.insert("large".to_string(), MockBitNetModel::create_large_bitnet());

        Self {
            mock_models,
            real_models: RealModelFixtures::new(config),
            config: config.clone(),
        }
    }

    /// Initialize all model fixtures
    pub async fn initialize(&mut self) -> Result<()> {
        // Create mock GGUF files for testing
        self.create_mock_files().await?;

        // Validate real models if available
        if self.config.real_models_available() {
            self.validate_real_models().await?;
        }

        Ok(())
    }

    /// Create mock GGUF and tokenizer files
    async fn create_mock_files(&self) -> Result<()> {
        // Ensure fixtures directory exists
        fs::create_dir_all("tests/fixtures").await
            .map_err(BitNetError::Io)?;

        for (name, mock_model) in &self.mock_models {
            // Generate mock GGUF file
            let gguf_content = mock_model.generate_mock_gguf();
            fs::write(&mock_model.config.model_path, gguf_content).await
                .map_err(BitNetError::Io)?;

            // Generate mock tokenizer file
            if let Some(tokenizer_path) = &mock_model.config.tokenizer_path {
                let tokenizer_content = self.generate_mock_tokenizer_json(&mock_model.config);
                fs::write(tokenizer_path, tokenizer_content).await
                    .map_err(BitNetError::Io)?;
            }

            println!("Created mock model fixtures for: {}", name);
        }

        Ok(())
    }

    /// Generate mock tokenizer JSON content
    fn generate_mock_tokenizer_json(&self, config: &TestModelConfig) -> String {
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 128000,
                    "content": "<|begin_of_text|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                },
                {
                    "id": 128001,
                    "content": "<|end_of_text|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ],
            "normalizer": {
                "type": "NFC"
            },
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            },
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": false
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": true,
                "vocab": {},
                "merges": []
            }
        }).to_string()
    }

    /// Validate real models are properly formatted
    async fn validate_real_models(&mut self) -> Result<()> {
        if let Some(model_path) = &self.config.model_path {
            if model_path.exists() {
                // Basic validation - check it's a valid GGUF file
                let model_loader = ModelLoader::new(self.config.device_preference.clone());
                let _metadata = model_loader.extract_metadata(model_path)?;
                println!("Validated real model at: {}", model_path.display());
            }
        }

        Ok(())
    }

    /// Get mock model by name
    pub fn get_mock_model(&self, name: &str) -> Option<&MockBitNetModel> {
        self.mock_models.get(name)
    }

    /// Get appropriate model for test tier
    pub async fn get_model_for_tier(&mut self, tier: TestTier) -> Result<PathBuf> {
        match tier {
            TestTier::Fast => {
                // Return mock model path
                Ok(self.get_mock_model("small")
                    .ok_or_else(|| BitNetError::Model(ModelError::NotFound {
                        path: "mock/small".to_string()
                    }))?
                    .config.model_path.clone())
            },
            TestTier::Standard | TestTier::Full => {
                // Try to get real model, fallback to mock
                if let Ok(real_model_path) = self.real_models.ensure_model_available("bitnet-2b").await {
                    Ok(real_model_path)
                } else {
                    // Fallback to mock for testing
                    Ok(self.get_mock_model("small")
                        .ok_or_else(|| BitNetError::Model(ModelError::NotFound {
                            path: "mock/small".to_string()
                        }))?
                        .config.model_path.clone())
                }
            }
        }
    }

    /// Cleanup model fixtures
    pub async fn cleanup(&mut self) -> Result<()> {
        // Remove mock files
        for mock_model in self.mock_models.values() {
            if mock_model.config.model_path.exists() {
                fs::remove_file(&mock_model.config.model_path).await
                    .map_err(BitNetError::Io)?;
            }

            if let Some(tokenizer_path) = &mock_model.config.tokenizer_path {
                if tokenizer_path.exists() {
                    fs::remove_file(tokenizer_path).await
                        .map_err(BitNetError::Io)?;
                }
            }
        }

        Ok(())
    }
}