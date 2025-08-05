use pyo3::prelude::*;

mod model;
mod inference;
mod tokenizer;
mod config;
mod error;
mod utils;

use model::PyBitNetModel;
use inference::PyInferenceEngine;
use tokenizer::PyTokenizer;
use config::{PyModelArgs, PyGenArgs, PyInferenceConfig};
use error::PyBitNetError;

/// BitNet.cpp Python bindings - High-performance 1-bit LLM inference
#[pymodule]
fn _bitnet_py(py: Python, m: &PyModule) -> PyResult<()> {
    // Core classes
    m.add_class::<PyBitNetModel>()?;
    m.add_class::<PyInferenceEngine>()?;
    m.add_class::<PyTokenizer>()?;
    
    // Configuration classes
    m.add_class::<PyModelArgs>()?;
    m.add_class::<PyGenArgs>()?;
    m.add_class::<PyInferenceConfig>()?;
    
    // Exception types
    m.add("BitNetError", py.get_type::<PyBitNetError>())?;
    
    // Utility functions
    m.add_function(wrap_pyfunction!(utils::load_model, m)?)?;
    m.add_function(wrap_pyfunction!(utils::create_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(utils::benchmark_inference, m)?)?;
    
    // Version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}