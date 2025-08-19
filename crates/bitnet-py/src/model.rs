//! # Python Model Bindings
//!
//! Python bindings for BitNet models with automatic memory management
//! and thread-safe access patterns.

use crate::{device_to_string, parse_device};
use bitnet_common::Device;
use bitnet_models::{loader::ModelLoader, Model};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

/// Python wrapper for BitNet models
#[pyclass(name = "BitNetModel")]
pub struct PyBitNetModel {
    inner: Arc<dyn Model>,
    device: Device,
}

impl PyBitNetModel {
    pub fn new(model: Box<dyn Model>) -> Self {
        Self {
            inner: Arc::from(model),
            device: Device::Cpu, // Default device
        }
    }

    pub fn inner(&self) -> Arc<dyn Model> {
        self.inner.clone()
    }
}

#[pymethods]
impl PyBitNetModel {
    /// Create a new BitNet model from file
    #[new]
    #[pyo3(signature = (path, device = "cpu", **kwargs))]
    fn new_py(py: Python<'_>, path: &str, device: &str, kwargs: Option<&pyo3::Bound<'_, PyDict>>) -> PyResult<Self> {
        py.allow_threads(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            rt.block_on(async {
                let device = parse_device(device)?;
                let loader = ModelLoader::new(device.clone());
                let model = loader
                    .load(path)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

                Ok(Self { inner: Arc::from(model), device })
            })
        })
    }

    /// Get model configuration as a dictionary
    #[getter]
    fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let config = self.inner.config();
        let py_config = PyDict::new_bound(py);

        // Model configuration
        let model_config = PyDict::new_bound(py);
        model_config.set_item("vocab_size", config.model.vocab_size)?;
        model_config.set_item("hidden_size", config.model.hidden_size)?;
        model_config.set_item("num_layers", config.model.num_layers)?;
        // num_attention_heads field was removed from ModelConfig
        model_config.set_item("max_position_embeddings", config.model.max_position_embeddings)?;
        py_config.set_item("model", model_config)?;

        // Quantization configuration
        let quant_config = PyDict::new_bound(py);
        quant_config.set_item("quantization_type", format!("{:?}", config.quantization.quantization_type))?;
        quant_config.set_item("block_size", config.quantization.block_size)?;
        py_config.set_item("quantization", quant_config)?;

        Ok(py_config.into())
    }

    /// Get the device the model is loaded on
    #[getter]
    fn device(&self) -> String {
        device_to_string(&self.device)
    }

    /// Get model parameter count
    #[getter]
    fn parameter_count(&self) -> usize {
        // Calculate parameter count from config
        let config = self.inner.config();
        let vocab_size = config.model.vocab_size;
        let hidden_size = config.model.hidden_size;
        let num_layers = config.model.num_layers;

        // Rough estimation: embedding + layers + output
        vocab_size * hidden_size
            + num_layers * hidden_size * hidden_size * 4
            + hidden_size * vocab_size
    }

    /// Get model memory usage in bytes
    #[getter]
    fn memory_usage(&self) -> usize {
        // Rough estimation based on parameter count and quantization
        let params = self.parameter_count();
        let config = self.inner.config();

        match config.quantization.quantization_type {
            bitnet_common::QuantizationType::I2S => params / 4, // 2 bits per parameter
            bitnet_common::QuantizationType::TL1 | bitnet_common::QuantizationType::TL2 => {
                params / 2
            } // 4 bits per parameter
        }
    }

    /// Get model architecture name
    #[getter]
    fn architecture(&self) -> &str {
        "BitNet"
    }

    /// Get model quantization type
    #[getter]
    fn quantization(&self) -> String {
        format!("{:?}", self.inner.config().quantization.quantization_type)
    }

    /// Check if model supports streaming
    #[getter]
    fn supports_streaming(&self) -> bool {
        true
    }

    /// Get model information as a dictionary
    fn info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let info = PyDict::new_bound(py);
        let config = self.inner.config();

        info.set_item("architecture", self.architecture())?;
        info.set_item("parameter_count", self.parameter_count())?;
        info.set_item("memory_usage", self.memory_usage())?;
        info.set_item("device", self.device())?;
        info.set_item("quantization", self.quantization())?;
        info.set_item("vocab_size", config.model.vocab_size)?;
        info.set_item("hidden_size", config.model.hidden_size)?;
        info.set_item("num_layers", config.model.num_layers)?;
        info.set_item("supports_streaming", self.supports_streaming())?;

        Ok(info.into())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "BitNetModel(architecture='{}', parameters={}, device='{}', quantization='{}')",
            self.architecture(),
            self.parameter_count(),
            self.device(),
            self.quantization()
        )
    }

    /// String representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python wrapper for model loader
#[pyclass(name = "ModelLoader")]
pub struct PyModelLoader {
    device: Device,
}

#[pymethods]
impl PyModelLoader {
    /// Create a new model loader
    #[new]
    #[pyo3(signature = (device = "cpu"))]
    fn new(device: &str) -> PyResult<Self> {
        let device = parse_device(device)?;
        Ok(Self { device })
    }

    /// Load a model from file
    #[pyo3(signature = (path, **kwargs))]
    fn load(&self, py: Python<'_>, path: &str, kwargs: Option<&pyo3::Bound<'_, PyDict>>) -> PyResult<PyBitNetModel> {
        py.allow_threads(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            rt.block_on(async {
                let loader = ModelLoader::new(self.device.clone());
                let model = loader
                    .load(path)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

                Ok(PyBitNetModel { inner: Arc::from(model), device: self.device.clone() })
            })
        })
    }

    /// Extract metadata from a model file without loading it
    fn extract_metadata(&self, py: Python<'_>, path: &str) -> PyResult<PyObject> {
        let path = path.to_string();
        let device = self.device.clone();
        
        let metadata = py.allow_threads(|| {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

            rt.block_on(async {
                let loader = ModelLoader::new(device);
                loader.extract_metadata(&path).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to extract metadata: {}", e))
                })
            })
        })?;

        let py_metadata = PyDict::new_bound(py);
        py_metadata.set_item("architecture", metadata.architecture)?;
        py_metadata.set_item("vocab_size", metadata.vocab_size)?;
        py_metadata.set_item("context_length", metadata.context_length)?;
        // parameter_count field was removed from ModelMetadata
        py_metadata.set_item("name", &metadata.name)?;
        py_metadata.set_item("version", &metadata.version)?;
        py_metadata.set_item("quantization", format!("{:?}", metadata.quantization))?;

        Ok(py_metadata.into())
    }

    /// List available formats
    fn available_formats(&self) -> Vec<String> {
        vec!["GGUF".to_string(), "SafeTensors".to_string(), "HuggingFace".to_string()]
    }

    /// Get device
    #[getter]
    fn device(&self) -> String {
        device_to_string(&self.device)
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("ModelLoader(device='{}')", self.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_loader_creation() {
        let loader = PyModelLoader::new("cpu").unwrap();
        assert_eq!(loader.device(), "cpu");

        let formats = loader.available_formats();
        assert!(formats.contains(&"GGUF".to_string()));
    }
}
