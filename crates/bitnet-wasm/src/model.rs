//! WebAssembly model wrapper for BitNet inference

use js_sys::{Object, Promise, Reflect, Uint8Array};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use web_sys::console;

use bitnet_common::{BitNetError, ModelError};
use bitnet_inference::InferenceEngine;

use crate::memory::MemoryManager;
use crate::utils::{JsError, to_js_error};

/// Configuration for WASM model loading
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmModelConfig {
    /// Maximum memory usage in bytes (default: 512MB)
    max_memory_bytes: Option<usize>,
    /// Enable progressive loading for large models
    progressive_loading: bool,
    /// Chunk size for progressive loading (default: 64MB)
    chunk_size_bytes: usize,
    /// Model format hint ("gguf", "safetensors", "auto")
    format_hint: String,
    /// Tokenizer type ("gpt2", "sentencepiece", "auto")
    tokenizer_type: String,
}

#[wasm_bindgen]
impl WasmModelConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmModelConfig {
        WasmModelConfig {
            max_memory_bytes: Some(512 * 1024 * 1024), // 512MB default
            progressive_loading: true,
            chunk_size_bytes: 64 * 1024 * 1024, // 64MB chunks
            format_hint: "auto".to_string(),
            tokenizer_type: "auto".to_string(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn max_memory_bytes(&self) -> Option<usize> {
        self.max_memory_bytes
    }

    #[wasm_bindgen(setter)]
    pub fn set_max_memory_bytes(&mut self, bytes: Option<usize>) {
        self.max_memory_bytes = bytes;
    }

    #[wasm_bindgen(getter)]
    pub fn progressive_loading(&self) -> bool {
        self.progressive_loading
    }

    #[wasm_bindgen(setter)]
    pub fn set_progressive_loading(&mut self, enabled: bool) {
        self.progressive_loading = enabled;
    }

    #[wasm_bindgen(getter)]
    pub fn chunk_size_bytes(&self) -> usize {
        self.chunk_size_bytes
    }

    #[wasm_bindgen(setter)]
    pub fn set_chunk_size_bytes(&mut self, bytes: usize) {
        self.chunk_size_bytes = bytes;
    }

    #[wasm_bindgen(getter)]
    pub fn format_hint(&self) -> String {
        self.format_hint.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_format_hint(&mut self, hint: String) {
        self.format_hint = hint;
    }

    #[wasm_bindgen(getter)]
    pub fn tokenizer_type(&self) -> String {
        self.tokenizer_type.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_tokenizer_type(&mut self, tokenizer_type: String) {
        self.tokenizer_type = tokenizer_type;
    }
}

impl Default for WasmModelConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// WebAssembly-compatible BitNet model wrapper
#[wasm_bindgen]
pub struct WasmBitNetModel {
    engine: Option<InferenceEngine>,
    memory_manager: MemoryManager,
    config: WasmModelConfig,
    model_info: HashMap<String, String>,
}

#[wasm_bindgen]
impl WasmBitNetModel {
    /// Create a new WASM BitNet model instance
    #[wasm_bindgen(constructor)]
    pub fn new(config: Option<WasmModelConfig>) -> Result<WasmBitNetModel, JsError> {
        let config = config.unwrap_or_default();
        let memory_manager = MemoryManager::new(config.max_memory_bytes)?;

        Ok(WasmBitNetModel { engine: None, memory_manager, config, model_info: HashMap::new() })
    }

    /// Load model from byte array (async)
    #[wasm_bindgen]
    pub fn load_from_bytes(
        &mut self,
        model_bytes: &Uint8Array,
        tokenizer_bytes: Option<Uint8Array>,
    ) -> Promise {
        let model_data = model_bytes.to_vec();
        let tokenizer_data = tokenizer_bytes.map(|bytes| bytes.to_vec());
        let config = self.config.clone();
        let max_memory = self.memory_manager.max_memory_bytes();

        wasm_bindgen_futures::future_to_promise(async move {
            Self::load_model_async(model_data, tokenizer_data, config, max_memory)
                .await
                .map(|result| JsValue::from_serde(&result).unwrap_or(JsValue::NULL))
                .map_err(to_js_error)
        })
    }

    /// Get model information
    #[wasm_bindgen]
    pub fn get_model_info(&self) -> Result<JsValue, JsError> {
        Ok(JsValue::from_serde(&self.model_info)?)
    }

    /// Check if model is loaded
    #[wasm_bindgen]
    pub fn is_loaded(&self) -> bool {
        self.engine.is_some()
    }

    /// Get current memory usage in bytes
    #[wasm_bindgen]
    pub fn get_memory_usage(&self) -> usize {
        self.memory_manager.current_usage()
    }

    /// Get maximum allowed memory in bytes
    #[wasm_bindgen]
    pub fn get_max_memory(&self) -> Option<usize> {
        self.memory_manager.max_memory_bytes()
    }

    /// Set memory limit (will trigger garbage collection if exceeded)
    #[wasm_bindgen]
    pub fn set_memory_limit(&mut self, bytes: Option<usize>) -> Result<(), JsError> {
        self.memory_manager.set_max_memory(bytes)?;
        self.config.max_memory_bytes = bytes;
        Ok(())
    }

    /// Force garbage collection
    #[wasm_bindgen]
    pub fn gc(&mut self) -> Result<usize, JsError> {
        self.memory_manager.gc()
    }

    /// Unload the model to free memory
    #[wasm_bindgen]
    pub fn unload(&mut self) -> Result<(), JsError> {
        self.engine = None;
        self.model_info.clear();
        self.memory_manager.gc()?;
        console::log_1(&"Model unloaded".into());
        Ok(())
    }
}

impl WasmBitNetModel {
    /// Internal async model loading implementation
    async fn load_model_async(
        model_data: Vec<u8>,
        _tokenizer_data: Option<Vec<u8>>,
        config: WasmModelConfig,
        max_memory: Option<usize>,
    ) -> Result<HashMap<String, String>, BitNetError> {
        console::log_1(&format!("Loading model with {} bytes", model_data.len()).into());

        if let Some(max_mem) = max_memory
            && model_data.len() > max_mem
        {
            return Err(BitNetError::Model(ModelError::LoadingFailed {
                reason: format!(
                    "Model size ({} bytes) exceeds memory limit ({} bytes)",
                    model_data.len(),
                    max_mem
                ),
            }));
        }

        Err(BitNetError::Model(ModelError::LoadingFailed {
            reason: format!(
                "WASM inference model loading is not implemented yet for format '{}' and tokenizer '{}'. This path intentionally returns an explicit not-implemented error instead of placeholder success.",
                config.format_hint, config.tokenizer_type
            ),
        }))
    }
}

/// JavaScript-friendly error type
#[wasm_bindgen]
pub struct ModelLoadResult {
    success: bool,
    error_message: Option<String>,
    model_info: Option<String>,
}

#[wasm_bindgen]
impl ModelLoadResult {
    #[wasm_bindgen(getter)]
    pub fn success(&self) -> bool {
        self.success
    }

    #[wasm_bindgen(getter)]
    pub fn error_message(&self) -> Option<String> {
        self.error_message.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn model_info(&self) -> Option<String> {
        self.model_info.clone()
    }
}
