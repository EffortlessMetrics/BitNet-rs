//! Python Configuration Bindings

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use bitnet_common::{BitNetConfig, QuantizationType};
use bitnet_inference::GenerationConfig;

/// Python wrapper for BitNet configuration
#[pyclass(name = "BitNetConfig")]
#[derive(Clone)]
pub struct PyBitNetConfig {
    inner: BitNetConfig,
}

#[pymethods]
impl PyBitNetConfig {
    #[new]
    fn new() -> Self {
        Self { inner: BitNetConfig::default() }
    }

    fn __repr__(&self) -> String {
        format!("BitNetConfig(vocab_size={})", self.inner.model.vocab_size)
    }
}

/// Python wrapper for generation configuration
#[pyclass(name = "GenerationConfig")]
#[derive(Clone)]
pub struct PyGenerationConfig {
    inner: GenerationConfig,
}

#[pymethods]
impl PyGenerationConfig {
    #[new]
    #[pyo3(signature = (max_tokens = 100, temperature = 0.7, top_p = 0.9, top_k = 50))]
    fn new(max_tokens: u32, temperature: f32, top_p: f32, top_k: u32) -> Self {
        let config = GenerationConfig {
            max_new_tokens: max_tokens,
            temperature,
            top_p,
            top_k,
            ..Default::default()
        };
        Self { inner: config }
    }

    #[getter]
    fn max_tokens(&self) -> u32 {
        self.inner.max_new_tokens
    }

    #[getter]
    fn temperature(&self) -> f32 {
        self.inner.temperature
    }

    fn __repr__(&self) -> String {
        format!(
            "GenerationConfig(max_tokens={}, temperature={})",
            self.inner.max_new_tokens, self.inner.temperature
        )
    }
}

impl PyGenerationConfig {
    pub fn inner(&self) -> &GenerationConfig {
        &self.inner
    }
}
