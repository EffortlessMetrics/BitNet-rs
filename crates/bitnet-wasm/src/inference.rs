//! WebAssembly inference wrapper with streaming support

use js_sys::{Array, Object, Promise};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use web_sys::console;

use crate::model::WasmBitNetModel;
use crate::streaming::WasmGenerationStream;
use crate::utils::{JsError, to_js_error};

/// Configuration for inference generation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmGenerationConfig {
    /// Maximum number of new tokens to generate
    max_new_tokens: usize,
    /// Temperature for sampling (0.0 = greedy, higher = more random)
    temperature: f32,
    /// Top-k sampling parameter
    top_k: Option<usize>,
    /// Top-p (nucleus) sampling parameter
    top_p: Option<f32>,
    /// Repetition penalty
    repetition_penalty: f32,
    /// Random seed for deterministic generation
    seed: Option<u64>,
    /// Stop tokens (will halt generation if encountered)
    stop_tokens: Vec<String>,
    /// Enable streaming output
    streaming: bool,
}

#[wasm_bindgen]
impl WasmGenerationConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmGenerationConfig {
        WasmGenerationConfig {
            max_new_tokens: 100,
            temperature: 0.7,
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: 1.1,
            seed: None,
            stop_tokens: vec!["</s>".to_string(), "<|endoftext|>".to_string()],
            streaming: false,
        }
    }

    // Getters and setters for all fields
    #[wasm_bindgen(getter)]
    pub fn max_new_tokens(&self) -> usize {
        self.max_new_tokens
    }

    #[wasm_bindgen(setter)]
    pub fn set_max_new_tokens(&mut self, tokens: usize) {
        self.max_new_tokens = tokens;
    }

    #[wasm_bindgen(getter)]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    #[wasm_bindgen(setter)]
    pub fn set_temperature(&mut self, temp: f32) {
        self.temperature = temp.max(0.0);
    }

    #[wasm_bindgen(getter)]
    pub fn top_k(&self) -> Option<usize> {
        self.top_k
    }

    #[wasm_bindgen(setter)]
    pub fn set_top_k(&mut self, k: Option<usize>) {
        self.top_k = k;
    }

    #[wasm_bindgen(getter)]
    pub fn top_p(&self) -> Option<f32> {
        self.top_p
    }

    #[wasm_bindgen(setter)]
    pub fn set_top_p(&mut self, p: Option<f32>) {
        self.top_p = p.map(|v| v.clamp(0.0, 1.0));
    }

    #[wasm_bindgen(getter)]
    pub fn repetition_penalty(&self) -> f32 {
        self.repetition_penalty
    }

    #[wasm_bindgen(setter)]
    pub fn set_repetition_penalty(&mut self, penalty: f32) {
        self.repetition_penalty = penalty.max(0.0);
    }

    #[wasm_bindgen(getter)]
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    #[wasm_bindgen(setter)]
    pub fn set_seed(&mut self, seed: Option<u64>) {
        self.seed = seed;
    }

    #[wasm_bindgen(getter)]
    pub fn streaming(&self) -> bool {
        self.streaming
    }

    #[wasm_bindgen(setter)]
    pub fn set_streaming(&mut self, enabled: bool) {
        self.streaming = enabled;
    }

    /// Add a stop token
    #[wasm_bindgen]
    pub fn add_stop_token(&mut self, token: String) {
        if !self.stop_tokens.contains(&token) {
            self.stop_tokens.push(token);
        }
    }

    /// Remove a stop token
    #[wasm_bindgen]
    pub fn remove_stop_token(&mut self, token: String) {
        self.stop_tokens.retain(|t| t != &token);
    }

    /// Get stop tokens as JavaScript array
    #[wasm_bindgen]
    pub fn get_stop_tokens(&self) -> Array {
        let array = Array::new();
        for token in &self.stop_tokens {
            array.push(&JsValue::from_str(token));
        }
        array
    }

    /// Set stop tokens from JavaScript array
    #[wasm_bindgen]
    pub fn set_stop_tokens(&mut self, tokens: Array) {
        self.stop_tokens.clear();
        for i in 0..tokens.length() {
            if let Ok(token) = tokens.get(i).as_string() {
                self.stop_tokens.push(token);
            }
        }
    }
}

impl Default for WasmGenerationConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// WebAssembly inference wrapper
#[wasm_bindgen]
pub struct WasmInference {
    model: WasmBitNetModel,
    generation_stats: HashMap<String, f64>,
}

#[wasm_bindgen]
impl WasmInference {
    /// Create a new inference wrapper
    #[wasm_bindgen(constructor)]
    pub fn new(model: WasmBitNetModel) -> Result<WasmInference, JsError> {
        if !model.is_loaded() {
            return Err(JsError::new("Model must be loaded before creating inference wrapper"));
        }

        Ok(WasmInference { model, generation_stats: HashMap::new() })
    }

    /// Generate text synchronously (blocking)
    #[wasm_bindgen]
    pub fn generate(
        &mut self,
        prompt: &str,
        config: Option<WasmGenerationConfig>,
    ) -> Result<String, JsError> {
        let config = config.unwrap_or_default();

        if !self.model.is_loaded() {
            return Err(JsError::new("Model is not loaded"));
        }

        console::log_1(&format!("Generating text for prompt: {}", prompt).into());

        // For now, return a placeholder response
        // In a full implementation, this would call the actual inference engine
        let response = format!(
            "Generated response for: {} (max_tokens: {}, temp: {})",
            prompt, config.max_new_tokens, config.temperature
        );

        // Update generation stats
        self.generation_stats.insert("last_prompt_length".to_string(), prompt.len() as f64);
        self.generation_stats.insert("last_response_length".to_string(), response.len() as f64);
        self.generation_stats.insert("last_temperature".to_string(), config.temperature as f64);

        Ok(response)
    }

    /// Generate text asynchronously
    #[wasm_bindgen]
    pub fn generate_async(
        &mut self,
        prompt: &str,
        config: Option<WasmGenerationConfig>,
    ) -> Promise {
        let prompt = prompt.to_string();
        let config = config.unwrap_or_default();
        let model_loaded = self.model.is_loaded();

        wasm_bindgen_futures::future_to_promise(async move {
            if !model_loaded {
                return Err(to_js_error(JsError::new("Model is not loaded")));
            }

            console::log_1(&format!("Async generating text for prompt: {}", prompt).into());

            // Simulate async processing
            let response = Self::generate_async_impl(prompt, config).await?;

            Ok(JsValue::from_str(&response))
        })
    }

    /// Create a streaming generation iterator
    #[wasm_bindgen]
    pub fn generate_stream(
        &mut self,
        prompt: &str,
        config: Option<WasmGenerationConfig>,
    ) -> Result<WasmGenerationStream, JsError> {
        let config = config.unwrap_or_default();

        if !self.model.is_loaded() {
            return Err(JsError::new("Model is not loaded"));
        }

        if !config.streaming {
            return Err(JsError::new("Streaming must be enabled in generation config"));
        }

        WasmGenerationStream::new(prompt.to_string(), config)
    }

    /// Get generation statistics
    #[wasm_bindgen]
    pub fn get_stats(&self) -> Result<JsValue, JsError> {
        Ok(JsValue::from_serde(&self.generation_stats)?)
    }

    /// Reset generation statistics
    #[wasm_bindgen]
    pub fn reset_stats(&mut self) {
        self.generation_stats.clear();
    }

    /// Check if the model is ready for inference
    #[wasm_bindgen]
    pub fn is_ready(&self) -> bool {
        self.model.is_loaded()
    }

    /// Get current memory usage
    #[wasm_bindgen]
    pub fn get_memory_usage(&self) -> usize {
        self.model.get_memory_usage()
    }

    /// Force garbage collection
    #[wasm_bindgen]
    pub fn gc(&mut self) -> Result<usize, JsError> {
        self.model.gc()
    }
}

impl WasmInference {
    /// Internal async generation implementation
    async fn generate_async_impl(
        prompt: String,
        config: WasmGenerationConfig,
    ) -> Result<String, JsError> {
        // Simulate processing time based on max_new_tokens
        let delay_ms = (config.max_new_tokens as f64 * 10.0) as i32;

        // Use setTimeout to simulate async processing
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            let resolve = resolve.clone();
            let prompt = prompt.clone();
            let config = config.clone();

            web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(
                    &js_sys::Function::new_no_args(&format!(
                        "arguments[0]('{} - Generated with temp: {}')",
                        prompt, config.temperature
                    )),
                    delay_ms,
                )
                .unwrap();
        });

        let result = wasm_bindgen_futures::JsFuture::from(promise).await?;
        Ok(result.as_string().unwrap_or_default())
    }
}

/// Result type for generation operations
#[wasm_bindgen]
pub struct GenerationResult {
    text: String,
    tokens_generated: usize,
    time_ms: f64,
    tokens_per_second: f64,
}

#[wasm_bindgen]
impl GenerationResult {
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn tokens_generated(&self) -> usize {
        self.tokens_generated
    }

    #[wasm_bindgen(getter)]
    pub fn time_ms(&self) -> f64 {
        self.time_ms
    }

    #[wasm_bindgen(getter)]
    pub fn tokens_per_second(&self) -> f64 {
        self.tokens_per_second
    }
}
