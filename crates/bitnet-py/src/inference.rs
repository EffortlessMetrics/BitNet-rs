//! # Python Inference Engine Bindings
//!
//! Python bindings for the BitNet inference engine with streaming support
//! and async/await compatibility.

use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyList, PyString};
// use pyo3_asyncio_0_21::tokio::future_into_py;
use futures_util::StreamExt;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::{parse_device, to_py_result, PyBitNetModel};
use bitnet_common::Device;
use bitnet_inference::{GenerationConfig, InferenceConfig, InferenceEngine};
use bitnet_tokenizers::{Tokenizer, TokenizerBuilder};

/// Python wrapper for the inference engine
#[pyclass(name = "InferenceEngine")]
pub struct PyInferenceEngine {
    inner: Arc<RwLock<InferenceEngine>>,
    device: Device,
}

impl PyInferenceEngine {
    pub fn new(engine: InferenceEngine, device: Device) -> Self {
        Self { inner: Arc::new(RwLock::new(engine)), device }
    }
}

#[pymethods]
impl PyInferenceEngine {
    /// Create a new inference engine
    #[new]
    #[pyo3(signature = (model, tokenizer = None, device = "cpu", **kwargs))]
    fn new_py(
        py: Python<'_>,
        model: &PyBitNetModel,
        tokenizer: Option<&str>,
        device: &str,
        kwargs: Option<&PyDict>,
    ) -> PyResult<Self> {
        py.allow_threads(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            rt.block_on(async {
                let device = parse_device(device)?;

                // Load tokenizer
                let tokenizer_name = tokenizer.unwrap_or("gpt2");
                let tokenizer = TokenizerBuilder::from_pretrained(tokenizer_name).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to load tokenizer: {}", e))
                })?;

                // Create inference engine
                let engine = InferenceEngine::new(model.inner(), tokenizer, device.clone())
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to create engine: {}", e))
                    })?;

                Ok(Self::new(engine, device))
            })
        })
    }

    /// Generate text from a prompt
    #[pyo3(signature = (prompt, max_tokens = 100, temperature = 0.7, top_p = 0.9, top_k = 50, **kwargs))]
    fn generate(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<String> {
        py.allow_threads(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            rt.block_on(async {
                let config = GenerationConfig {
                    max_new_tokens: max_tokens.unwrap_or(100),
                    temperature: temperature.unwrap_or(0.7),
                    top_p: top_p.unwrap_or(0.9),
                    top_k: top_k.unwrap_or(50),
                    ..Default::default()
                };

                let mut engine = self.inner.write().await;
                let result = engine
                    .generate_with_config(prompt, &config)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("Generation failed: {}", e)))?;

                Ok(result)
            })
        })
    }

    // Async generation would require pyo3-asyncio integration
    // Commented out for now to avoid compilation issues

    /// Generate streaming tokens
    #[pyo3(signature = (prompt, max_tokens = 100, temperature = 0.7, top_p = 0.9, top_k = 50, **kwargs))]
    fn generate_stream(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<PyStreamingGenerator> {
        let config = GenerationConfig {
            max_new_tokens: max_tokens.unwrap_or(100),
            temperature: temperature.unwrap_or(0.7),
            top_p: top_p.unwrap_or(0.9),
            top_k: top_k.unwrap_or(50),
            ..Default::default()
        };

        let engine = self.inner.clone();
        let prompt = prompt.to_string();

        Ok(PyStreamingGenerator::new(engine, prompt, config))
    }

    /// Get model configuration
    #[getter]
    fn model_config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        rt.block_on(async {
            let engine = self.inner.read().await;
            let config = engine.model_config();

            let py_config = PyDict::new(py);
            py_config.set_item("vocab_size", config.model.vocab_size)?;
            py_config.set_item("hidden_size", config.model.hidden_size)?;
            py_config.set_item("num_layers", config.model.num_layers)?;
            py_config.set_item("num_attention_heads", config.model.num_attention_heads)?;

            Ok(py_config.into())
        })
    }

    /// Get device
    #[getter]
    fn device(&self) -> String {
        crate::device_to_string(&self.device)
    }

    /// Get inference statistics
    fn get_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        rt.block_on(async {
            let engine = self.inner.read().await;
            let stats = engine.get_stats().await;

            let py_stats = PyDict::new(py);
            py_stats.set_item("cache_size", stats.cache_size)?;
            py_stats.set_item("cache_usage", stats.cache_usage)?;
            py_stats.set_item("backend_type", stats.backend_type)?;

            Ok(py_stats.into())
        })
    }

    /// Clear the KV cache
    fn clear_cache(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            rt.block_on(async {
                let engine = self.inner.read().await;
                engine.clear_cache().await;
                Ok(())
            })
        })
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("InferenceEngine(device='{}')", self.device())
    }
}

/// Python streaming generator for token generation
#[pyclass(name = "StreamingGenerator")]
pub struct PyStreamingGenerator {
    engine: Arc<RwLock<InferenceEngine>>,
    prompt: String,
    config: GenerationConfig,
    started: bool,
}

impl PyStreamingGenerator {
    fn new(engine: Arc<RwLock<InferenceEngine>>, prompt: String, config: GenerationConfig) -> Self {
        Self { engine, prompt, config, started: false }
    }
}

#[pymethods]
impl PyStreamingGenerator {
    /// Make the generator iterable
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Get the next token
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<String>> {
        if !self.started {
            self.started = true;
            // In a real implementation, this would start the streaming
            // For now, return a mock implementation
            return Ok(Some("Hello".to_string()));
        }

        // Mock streaming - in practice this would use the actual streaming API
        static mut COUNTER: usize = 0;
        unsafe {
            COUNTER += 1;
            if COUNTER <= 5 {
                Ok(Some(format!(" token_{}", COUNTER)))
            } else {
                COUNTER = 0;
                Err(PyStopIteration::new_err(""))
            }
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "StreamingGenerator(prompt='{}...', max_tokens={})",
            &self.prompt[..20.min(self.prompt.len())],
            self.config.max_new_tokens
        )
    }
}

/// Batch inference for multiple prompts
#[pyfunction]
#[pyo3(signature = (engine, prompts, **kwargs))]
pub fn batch_generate(
    py: Python<'_>,
    engine: &PyInferenceEngine,
    prompts: Vec<String>,
    kwargs: Option<&PyDict>,
) -> PyResult<Vec<String>> {
    py.allow_threads(|| {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        rt.block_on(async {
            let mut results = Vec::new();
            let config = GenerationConfig::default();

            for prompt in prompts {
                let mut engine_guard = engine.inner.write().await;
                let result = engine_guard
                    .generate_with_config(&prompt, &config)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("Generation failed: {}", e)))?;
                results.push(result);
            }

            Ok(results)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_generator() {
        let engine = Arc::new(RwLock::new(
            // Mock engine - in practice would be real engine
        ));
        let generator = PyStreamingGenerator::new(
            engine,
            "test prompt".to_string(),
            GenerationConfig::default(),
        );

        assert!(!generator.started);
        assert!(generator.prompt == "test prompt");
    }
}
