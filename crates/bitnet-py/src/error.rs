use pyo3::prelude::*;
use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError, PyIOError};
use bitnet_common::BitNetError;

/// Python exception type for BitNet errors
#[pyclass(extends=PyException)]
pub struct PyBitNetError {
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub error_type: String,
}

#[pymethods]
impl PyBitNetError {
    #[new]
    fn new(message: String, error_type: Option<String>) -> Self {
        Self {
            message,
            error_type: error_type.unwrap_or_else(|| "BitNetError".to_string()),
        }
    }
    
    fn __str__(&self) -> String {
        format!("{}: {}", self.error_type, self.message)
    }
    
    fn __repr__(&self) -> String {
        format!("{}('{}')", self.error_type, self.message)
    }
}

impl From<BitNetError> for PyErr {
    fn from(err: BitNetError) -> Self {
        match err {
            BitNetError::Model(e) => PyValueError::new_err(format!("Model error: {}", e)),
            BitNetError::Quantization(e) => PyValueError::new_err(format!("Quantization error: {}", e)),
            BitNetError::Kernel(e) => PyRuntimeError::new_err(format!("Kernel error: {}", e)),
            BitNetError::Inference(e) => PyRuntimeError::new_err(format!("Inference error: {}", e)),
            BitNetError::Io(e) => PyIOError::new_err(format!("IO error: {}", e)),
            BitNetError::Config(e) => PyValueError::new_err(format!("Configuration error: {}", e)),
            _ => PyRuntimeError::new_err(format!("BitNet error: {}", err)),
        }
    }
}

impl From<anyhow::Error> for PyErr {
    fn from(err: anyhow::Error) -> Self {
        PyRuntimeError::new_err(format!("Error: {}", err))
    }
}