//! Python Tokenizer Bindings

use bitnet_tokenizers::{Tokenizer, TokenizerBuilder};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;

/// Python wrapper for tokenizers
#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: Arc<dyn Tokenizer>,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let tokenizer = TokenizerBuilder::from_pretrained(name)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self { inner: tokenizer })
    }

    fn encode(&self, text: &str, add_special_tokens: Option<bool>) -> PyResult<Vec<u32>> {
        let add_special = add_special_tokens.unwrap_or(true);
        // For Python API compatibility, use add_special for both add_bos and add_special
        self.inner
            .encode(text, add_special, add_special)
            .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))
    }

    fn decode(&self, tokens: Vec<u32>, _skip_special_tokens: Option<bool>) -> PyResult<String> {
        // New API doesn't take skip_special_tokens parameter
        self.inner
            .decode(&tokens)
            .map_err(|e| PyRuntimeError::new_err(format!("Decoding failed: {}", e)))
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.vocab_size())
    }
}

impl PyTokenizer {
    pub fn inner(&self) -> Arc<dyn Tokenizer> {
        self.inner.clone()
    }
}
