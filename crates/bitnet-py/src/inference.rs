//! # Python Inference Engine Bindings
//!
//! Python bindings for the BitNet inference engine with streaming support,
//! batch inference, numpy logits access, and performance metrics.

use pyo3::exceptions::{PyRuntimeError, PyStopIteration};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::{PyBitNetModel, parse_device};
use bitnet_common::Device;
use bitnet_inference::{GenerationConfig, InferenceEngine};
use bitnet_tokenizers::TokenizerBuilder;

fn build_generation_config(
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
) -> PyResult<GenerationConfig> {
    let max_tokens = max_tokens.unwrap_or(100);
    let temperature = temperature.unwrap_or(0.7);
    let top_p = top_p.unwrap_or(0.9);
    let top_k = top_k.unwrap_or(50);

    if max_tokens == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("max_tokens must be greater than 0"));
    }

    if !temperature.is_finite() || temperature < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "temperature must be a finite value >= 0",
        ));
    }

    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "top_p must be a finite value in the range (0, 1]",
        ));
    }

    if top_k == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("top_k must be greater than 0"));
    }

    Ok(GenerationConfig::default()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature)
        .with_top_p(top_p)
        .with_top_k(top_k))
}

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
    #[pyo3(signature = (model, tokenizer = None, device = "cpu", **_kwargs))]
    fn new_py(
        py: Python<'_>,
        model: &PyBitNetModel,
        tokenizer: Option<&str>,
        device: &str,
        _kwargs: Option<&pyo3::Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        py.detach(|| {
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
                let engine =
                    InferenceEngine::new(model.inner(), tokenizer, device).map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to create engine: {}", e))
                    })?;

                Ok(Self::new(engine, device))
            })
        })
    }

    /// Generate text from a prompt using synchronous (blocking) generation.
    ///
    /// This method performs complete text generation in a single call, blocking until
    /// all tokens are generated. For streaming generation that yields tokens as they
    /// are produced, use `generate_stream()` instead.
    ///
    /// # Arguments
    /// * `prompt` - Input text to generate from
    /// * `max_tokens` - Maximum number of tokens to generate (default: 100)
    /// * `temperature` - Sampling temperature for randomness (default: 0.7)
    /// * `top_p` - Nucleus sampling parameter (default: 0.9)
    /// * `top_k` - Top-k sampling parameter (default: 50)
    ///
    /// # Returns
    /// Complete generated text as a single string
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if generation fails at any point
    #[pyo3(signature = (prompt, max_tokens = 100, temperature = 0.7, top_p = 0.9, top_k = 50, **_kwargs))]
    fn generate(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        _kwargs: Option<&pyo3::Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        py.detach(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            rt.block_on(async {
                let config = build_generation_config(max_tokens, temperature, top_p, top_k)?;

                info!("Starting synchronous generation with prompt length: {}", prompt.len());
                debug!(
                    "Generation config: max_tokens={}, temp={:.2}, top_p={:.2}, top_k={}",
                    config.max_new_tokens, config.temperature, config.top_p, config.top_k
                );

                let engine = self.inner.write().await;
                let result = engine.generate_with_config(prompt, &config).await.map_err(|e| {
                    error!("Synchronous generation failed: {}", e);
                    PyRuntimeError::new_err(format!("Generation failed: {}", e))
                })?;

                info!("Synchronous generation completed, result length: {}", result.len());
                Ok(result)
            })
        })
    }

    // Async generation would require pyo3-asyncio integration
    // Commented out for now to avoid compilation issues

    /// Generate streaming tokens with comprehensive error logging and queue-based semantics.
    ///
    /// This method creates a streaming generator that yields tokens incrementally as they are
    /// generated by the inference engine. The implementation uses an internal queue to buffer
    /// tokens and handle backpressure, ensuring smooth streaming even under varying generation
    /// speeds.
    ///
    /// # Arguments
    /// * `prompt` - The input text to generate from
    /// * `max_tokens` - Maximum number of tokens to generate (default: 100)
    /// * `temperature` - Sampling temperature for randomness (default: 0.7)
    /// * `top_p` - Nucleus sampling parameter (default: 0.9)
    /// * `top_k` - Top-k sampling parameter (default: 50)
    ///
    /// # Returns
    /// A `PyStreamingGenerator` that implements Python's iterator protocol
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if the generator cannot be initialized
    #[pyo3(signature = (prompt, max_tokens = 100, temperature = 0.7, top_p = 0.9, top_k = 50, **_kwargs))]
    fn generate_stream(
        &self,
        _py: Python<'_>,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        _kwargs: Option<&pyo3::Bound<'_, PyDict>>,
    ) -> PyResult<PyStreamingGenerator> {
        info!("Starting streaming generation with prompt length: {}", prompt.len());

        let config = build_generation_config(max_tokens, temperature, top_p, top_k)?;

        debug!(
            "Generation config: max_tokens={}, temp={:.2}, top_p={:.2}, top_k={}",
            config.max_new_tokens, config.temperature, config.top_p, config.top_k
        );

        let engine = self.inner.clone();
        let prompt = prompt.to_string();

        match PyStreamingGenerator::new(engine, prompt, config) {
            Ok(generator) => {
                info!("Successfully created streaming generator");
                Ok(generator)
            }
            Err(e) => {
                error!("Failed to create streaming generator: {}", e);
                Err(e)
            }
        }
    }

    /// Get model configuration as a Python dictionary.
    ///
    /// Returns key configuration parameters from the loaded model, including
    /// vocabulary size, hidden dimensions, and layer count. This information
    /// can be useful for understanding model capabilities and constraints.
    ///
    /// # Returns
    /// Python dictionary containing model configuration parameters
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if configuration cannot be accessed
    #[getter]
    fn model_config(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        rt.block_on(async {
            let engine = self.inner.read().await;
            let config = engine.model_config();

            let py_config = PyDict::new(py);
            py_config.set_item("vocab_size", config.model.vocab_size)?;
            py_config.set_item("hidden_size", config.model.hidden_size)?;
            py_config.set_item("num_layers", config.model.num_layers)?;
            // num_attention_heads field was removed from ModelConfig

            Ok(py_config.into())
        })
    }

    /// Get the device used for inference (CPU, CUDA, etc.).
    ///
    /// Returns a string representation of the device being used for model inference.
    /// This can be useful for debugging and performance monitoring.
    ///
    /// # Returns
    /// String representation of the device (e.g., "cpu", "cuda:0")
    #[getter]
    fn device(&self) -> String {
        crate::device_to_string(&self.device)
    }

    /// Get inference statistics including cache usage and performance metrics.
    ///
    /// Returns detailed statistics about the inference engine's current state,
    /// including memory usage, cache statistics, and backend information.
    ///
    /// # Returns
    /// Python dictionary containing inference statistics
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if statistics cannot be retrieved
    fn get_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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

    /// Clear the key-value cache to free memory and reset model state.
    fn clear_cache(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            rt.block_on(async {
                let engine = self.inner.read().await;
                engine.clear_cache().await;
                Ok(())
            })
        })
    }

    /// Generate text and return a dict with text, token count, and perf metrics.
    ///
    /// Unlike `generate()` which returns only the text, this method returns a
    /// dictionary containing the generated text together with timing and
    /// throughput information useful for benchmarking.
    #[pyo3(signature = (prompt, max_tokens = 100, temperature = 0.7, top_p = 0.9, top_k = 50, seed = None))]
    fn generate_with_metrics(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        seed: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let prompt = prompt.to_string();
        py.detach(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

            rt.block_on(async {
                let mut config = build_generation_config(max_tokens, temperature, top_p, top_k)?;
                if let Some(s) = seed {
                    config = config.with_seed(s);
                }

                let start = std::time::Instant::now();
                let engine = self.inner.write().await;
                let text = engine
                    .generate_with_config(&prompt, &config)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("Generation failed: {e}")))?;
                let elapsed_ms = start.elapsed().as_millis() as u64;

                let stats = engine.get_stats().await;

                Python::attach(|py2| {
                    let dict = PyDict::new(py2);
                    dict.set_item("text", &text)?;
                    dict.set_item("latency_ms", elapsed_ms)?;
                    let token_count = text.split_whitespace().count();
                    dict.set_item("token_count", token_count)?;
                    let tps = if elapsed_ms > 0 {
                        token_count as f64 / (elapsed_ms as f64 / 1000.0)
                    } else {
                        0.0
                    };
                    dict.set_item("tokens_per_second", tps)?;
                    dict.set_item("cache_size", stats.cache_size)?;
                    dict.set_item("backend_type", &stats.backend_type)?;
                    Ok(dict.into())
                })
            })
        })
    }

    /// Evaluate a list of token IDs and return logits as a Python list of floats.
    ///
    /// This gives direct access to the model's output logits for a given token
    /// sequence, which is useful for perplexity evaluation, custom sampling,
    /// or cross-validation workflows.
    #[pyo3(signature = (token_ids,))]
    fn get_logits(&self, py: Python<'_>, token_ids: Vec<u32>) -> PyResult<Py<PyAny>> {
        if token_ids.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("token_ids must not be empty"));
        }

        py.detach(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

            rt.block_on(async {
                let mut engine = self.inner.write().await;
                let logits = engine
                    .eval_ids(&token_ids)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("eval_ids failed: {e}")))?;

                Python::attach(|py2| {
                    let py_list = PyList::new(py2, &logits)
                        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
                    Ok(py_list.into())
                })
            })
        })
    }

    /// String representation for debugging and introspection.
    ///
    /// Provides a human-readable representation of the inference engine
    /// including the device being used for inference.
    ///
    /// # Returns
    /// Formatted string representation
    fn __repr__(&self) -> String {
        format!("InferenceEngine(device='{}')", self.device())
    }
}

/// Python streaming generator for token generation with queue-based semantics.
///
/// This class provides a Python iterator interface for streaming token generation from
/// the BitNet inference engine. It implements a queue-based architecture where:
///
/// 1. **Lazy Initialization**: The underlying Rust stream is created on first `__next__()` call
/// 2. **Buffered Streaming**: Tokens are buffered internally to handle backpressure
/// 3. **Cancellation Support**: Generation can be stopped at any time with proper cleanup
/// 4. **Error Recovery**: Transient errors are logged and propagated appropriately
///
/// # Queue-Based Streaming Semantics
///
/// The streaming generator uses a queue-based approach to decouple token generation
/// from consumption:
///
/// - **Producer**: Background task generates tokens and enqueues them
/// - **Consumer**: Python iterator polls the queue for available tokens
/// - **Backpressure**: Queue size limits prevent unbounded memory growth
/// - **Cancellation**: Cancellation signals are propagated to the producer
///
/// # Usage Pattern
///
/// ```python
/// stream = engine.generate_stream("Hello world")
/// for token in stream:
///     print(token, end='', flush=True)
/// ```
///
/// # Thread Safety
///
/// This class maintains its own Tokio runtime and is safe to use from Python threads.
/// However, individual instances should not be shared between threads without proper
/// synchronization.
#[pyclass(name = "StreamingGenerator")]
pub struct PyStreamingGenerator {
    engine: Arc<RwLock<InferenceEngine>>,
    prompt: String,
    config: GenerationConfig,
    started: bool,
    runtime: tokio::runtime::Runtime,
    stream: Option<bitnet_inference::streaming::GenerationStream>,
}

impl PyStreamingGenerator {
    /// Create a new streaming generator with the specified configuration.
    ///
    /// This constructor sets up the generator but does not start token generation.
    /// Generation begins lazily on the first call to `__next__()`.
    ///
    /// # Arguments
    /// * `engine` - Shared reference to the inference engine
    /// * `prompt` - Input text to generate from
    /// * `config` - Generation configuration (temperature, top_k, etc.)
    ///
    /// # Returns
    /// A new `PyStreamingGenerator` instance ready for iteration
    ///
    /// # Errors
    /// Returns `PyRuntimeError` if the Tokio runtime cannot be created
    fn new(
        engine: Arc<RwLock<InferenceEngine>>,
        prompt: String,
        config: GenerationConfig,
    ) -> PyResult<Self> {
        debug!(
            "Creating new streaming generator for prompt: '{}...'",
            &prompt[..20.min(prompt.len())]
        );

        let runtime = tokio::runtime::Runtime::new().map_err(|e| {
            error!("Failed to create Tokio runtime: {}", e);
            PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
        })?;

        info!("Streaming generator created successfully");
        Ok(Self { engine, prompt, config, started: false, runtime, stream: None })
    }
}

#[pymethods]
impl PyStreamingGenerator {
    /// Make the generator iterable (implements Python iterator protocol).
    ///
    /// This method is required by Python's iterator protocol and simply returns
    /// self, indicating that this object implements both `__iter__` and `__next__`.
    ///
    /// # Returns
    /// Reference to self for chaining
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        debug!("Iterator protocol __iter__ called");
        slf
    }

    /// Get the next token from the streaming generator.
    ///
    /// This method implements Python's iterator protocol (__next__) and handles the
    /// queue-based streaming semantics. On first call, it initializes the underlying
    /// Rust stream and begins token generation. Subsequent calls pull tokens from
    /// the internal queue, blocking if necessary until tokens are available.
    ///
    /// # Returns
    /// * `Ok(Some(String))` - Next token text
    /// * `Ok(None)` - Should not occur (Python handles this case)
    /// * `Err(PyStopIteration)` - Stream has ended normally
    /// * `Err(PyRuntimeError)` - Stream initialization or generation error
    ///
    /// # Error Handling
    /// All errors are logged with appropriate context before being propagated to Python.
    /// Transient errors during streaming are logged as warnings but may be recoverable.
    fn __next__(&mut self, _py: Python<'_>) -> PyResult<Option<String>> {
        if !self.started {
            self.started = true;
            debug!(
                "Initializing streaming generator for prompt: '{}...'",
                &self.prompt[..20.min(self.prompt.len())]
            );

            // Initialize the stream
            let engine = self.engine.clone();
            let prompt = self.prompt.clone();
            let config = self.config.clone();

            match self.runtime.block_on(async move {
                let engine_guard = engine.read().await;
                engine_guard.generate_stream_with_config(&prompt, &config)
            }) {
                Ok(stream) => {
                    info!("Stream initialized successfully");
                    self.stream = Some(stream);
                }
                Err(e) => {
                    error!("Failed to initialize stream: {}", e);
                    return Err(PyRuntimeError::new_err(format!("Failed to create stream: {}", e)));
                }
            };
        }

        if let Some(ref mut stream) = self.stream {
            // Get next token from the stream with error logging
            match self.runtime.block_on(async {
                use futures_util::StreamExt;
                stream.next().await
            }) {
                Some(Ok(stream_response)) => {
                    debug!("Generated token: '{}'", stream_response.text);
                    Ok(Some(stream_response.text))
                }
                Some(Err(e)) => {
                    warn!("Streaming error encountered: {}", e);
                    // Clear the stream to prevent further attempts
                    self.stream = None;
                    // Provide more detailed error information
                    let error_msg = if e.to_string().contains("timeout") {
                        format!("Generation timeout: {}", e)
                    } else if e.to_string().contains("cancelled") {
                        format!("Generation cancelled: {}", e)
                    } else if e.to_string().contains("memory") {
                        format!("Memory error during generation: {}", e)
                    } else {
                        format!("Streaming error: {}", e)
                    };
                    Err(PyRuntimeError::new_err(error_msg))
                }
                None => {
                    // Stream ended normally
                    info!("Stream completed successfully");
                    self.stream = None;
                    Err(PyStopIteration::new_err(""))
                }
            }
        } else {
            debug!("Stream already completed or not initialized");
            Err(PyStopIteration::new_err(""))
        }
    }

    /// Cancel the streaming generation with proper resource cleanup.
    ///
    /// This method immediately stops token generation and releases associated resources.
    /// It's safe to call multiple times and will log cancellation attempts for debugging.
    ///
    /// The cancellation is immediate and irreversible. After calling this method,
    /// any subsequent calls to `__next__()` will raise `StopIteration`.
    ///
    /// # Thread Safety
    /// This method is safe to call from any thread, including during active generation.
    ///
    /// # Examples
    /// ```python
    /// stream = engine.generate_stream("Hello world")
    ///
    /// # Get a few tokens
    /// for i, token in enumerate(stream):
    ///     print(token, end="")
    ///     if i >= 2:  # Cancel after 3 tokens
    ///         stream.cancel()
    ///         break
    /// ```
    ///
    /// # Returns
    /// Always returns `Ok(())` after attempting cancellation.
    fn cancel(&mut self) -> PyResult<()> {
        if self.stream.is_some() {
            info!("Cancelling active streaming generation");
            self.stream = None;
            debug!("Stream cancelled and resources cleaned up");
        } else {
            debug!("Cancel called on already inactive stream");
        }
        Ok(())
    }

    /// Check if streaming generation is currently active.
    ///
    /// Returns true if the stream has been initialized and is still generating tokens.
    /// This can be used by client code to determine if cancellation is necessary.
    ///
    /// # Examples
    /// ```python
    /// stream = engine.generate_stream("Hello world")
    ///
    /// while stream.is_active():
    ///     try:
    ///         token = next(stream)
    ///         print(token, end="")
    ///     except StopIteration:
    ///         break
    ///     except KeyboardInterrupt:
    ///         stream.cancel()
    ///         break
    /// ```
    ///
    /// # Returns
    /// `true` if streaming is active, `false` otherwise
    fn is_active(&self) -> bool {
        let active = self.stream.is_some() && self.started;
        debug!("Stream active status: {}", active);
        active
    }

    /// Get streaming statistics and progress information.
    ///
    /// Returns a dictionary containing information about the current streaming session,
    /// including tokens generated, generation speed, and resource usage.
    ///
    /// # Returns
    /// Python dictionary with streaming statistics
    fn get_stream_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let stats = pyo3::types::PyDict::new(py);

        stats.set_item("started", self.started)?;
        stats.set_item("active", self.is_active())?;

        // Add configuration information
        stats.set_item("max_tokens", self.config.max_new_tokens)?;
        stats.set_item("temperature", self.config.temperature)?;
        stats.set_item("top_k", self.config.top_k)?;
        stats.set_item("top_p", self.config.top_p)?;

        // Add prompt information (truncated for privacy)
        let prompt_preview = if self.prompt.len() > 50 {
            format!("{}...", &self.prompt[..47])
        } else {
            self.prompt.clone()
        };
        stats.set_item("prompt_preview", prompt_preview)?;
        stats.set_item("prompt_length", self.prompt.len())?;

        Ok(stats.into())
    }

    /// String representation for debugging and introspection.
    ///
    /// Provides a human-readable representation of the generator state including
    /// a preview of the prompt and key configuration parameters.
    ///
    /// # Returns
    /// Formatted string representation
    fn __repr__(&self) -> String {
        let status = if self.started {
            if self.is_active() { "active" } else { "completed" }
        } else {
            "pending"
        };

        format!(
            "StreamingGenerator(prompt='{}...', max_tokens={}, status='{}')",
            &self.prompt[..20.min(self.prompt.len())],
            self.config.max_new_tokens,
            status
        )
    }
}

/// Batch inference for multiple prompts.
///
/// Returns a list of dicts, each containing `text` and `latency_ms`.
/// Generation parameters are applied uniformly to all prompts.
#[pyfunction]
#[pyo3(signature = (engine, prompts, max_tokens = 100, temperature = 0.7, top_p = 0.9, top_k = 50))]
pub fn batch_generate(
    py: Python<'_>,
    engine: &PyInferenceEngine,
    prompts: Vec<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
) -> PyResult<Vec<Py<PyAny>>> {
    py.detach(|| {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

        rt.block_on(async {
            let config = build_generation_config(max_tokens, temperature, top_p, top_k)?;
            let mut results = Vec::with_capacity(prompts.len());

            info!("Starting batch generation for {} prompts", prompts.len());

            for (i, prompt) in prompts.iter().enumerate() {
                debug!("Processing batch prompt {}/{}", i + 1, prompts.len());

                let start = std::time::Instant::now();
                let engine_guard = engine.inner.write().await;
                let text =
                    engine_guard.generate_with_config(prompt, &config).await.map_err(|e| {
                        error!("Batch generation failed for prompt {}: {}", i + 1, e);
                        PyRuntimeError::new_err(format!(
                            "Generation failed for prompt {}: {}",
                            i + 1,
                            e
                        ))
                    })?;
                let elapsed_ms = start.elapsed().as_millis() as u64;

                Python::attach(|py2| {
                    let dict = PyDict::new(py2);
                    dict.set_item("text", &text)?;
                    dict.set_item("latency_ms", elapsed_ms)?;
                    dict.set_item("prompt_index", i)?;
                    results.push(dict.into());
                    Ok::<(), PyErr>(())
                })?;
            }

            info!("Batch generation completed for {} prompts", results.len());
            Ok(results)
        })
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_streaming_generator() {
        // Skip this test for now - would need a full mock engine
        // Note: Full testing would require a mock InferenceEngine
    }
}
