//! # BitNet Python Bindings
//!
//! Python bindings for BitNet-rs providing a seamless migration path from
//! existing Python implementations with identical API compatibility.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod async_runtime;
mod config;
mod inference;
mod model;
mod tokenizer;
mod utils;

pub use async_runtime::*;
pub use config::*;
pub use inference::*;
pub use model::*;
pub use tokenizer::*;
pub use utils::*;

/// BitNet Python module
#[pymodule]
fn bitnet_py(py: Python<'_>, m: &pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "BitNet-rs Team")?;
    m.add("__description__", "High-performance BitNet inference in Rust with Python bindings")?;

    // Add main classes
    m.add_class::<PyBitNetModel>()?;
    m.add_class::<PyInferenceEngine>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyBitNetConfig>()?;
    m.add_class::<PyGenerationConfig>()?;
    m.add_class::<PyModelLoader>()?;

    // Add utility functions
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(list_available_models, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_info, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(batch_generate, m)?)?;

    // Add constants
    m.add("CPU", "cpu")?;
    m.add("CUDA", "cuda")?;
    m.add("METAL", "metal")?;

    // Add device detection functions
    m.add_function(wrap_pyfunction!(is_cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(is_metal_available, m)?)?;
    m.add_function(wrap_pyfunction!(get_cuda_device_count, m)?)?;

    // Add quantization types
    let quantization_types = PyDict::new(py);
    quantization_types.set_item("I2S", "i2s")?;
    quantization_types.set_item("TL1", "tl1")?;
    quantization_types.set_item("TL2", "tl2")?;
    m.add("QuantizationType", quantization_types)?;

    // Add example usage in docstring
    m.add(
        "__doc__",
        r#"
BitNet-rs Python Bindings

High-performance BitNet inference library with Python bindings.

Example usage:
    import bitnet_py as bitnet

    # Load a model
    model = bitnet.load_model("path/to/model.gguf")

    # Create inference engine
    engine = bitnet.InferenceEngine(model, device="cpu")

    # Generate text
    result = engine.generate("Hello, world!", max_tokens=100)
    print(result)

    # Streaming generation
    for token in engine.generate_stream("Once upon a time"):
        print(token, end="", flush=True)

For more examples, see the documentation at https://github.com/microsoft/BitNet
"#,
    )?;

    Ok(())
}

/// Load a model from file with automatic format detection
#[pyfunction]
#[pyo3(signature = (path, device = "cpu", **_kwargs))]
fn load_model(
    py: Python<'_>,
    path: &str,
    device: &str,
    _kwargs: Option<&pyo3::Bound<'_, PyDict>>,
) -> PyResult<PyBitNetModel> {
    py.detach(|| {
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create async runtime: {}", e))
        })?;

        rt.block_on(async {
            let device = parse_device(device)?;
            let loader = bitnet_models::loader::ModelLoader::new(device);

            let model = loader
                .load(path)
                .map_err(|e| PyIOError::new_err(format!("Failed to load model: {}", e)))?;

            Ok(PyBitNetModel::new(model))
        })
    })
}

/// List available models in a directory
#[pyfunction]
fn list_available_models(path: &str) -> PyResult<Vec<String>> {
    let path = std::path::Path::new(path);
    let mut models = Vec::new();

    if path.is_dir() {
        let entries = std::fs::read_dir(path)
            .map_err(|e| PyIOError::new_err(format!("Failed to read directory: {}", e)))?;

        for entry in entries {
            let entry = entry.map_err(|e| PyIOError::new_err(e.to_string()))?;
            let path = entry.path();

            if let Some(ext) = path.extension()
                && matches!(ext.to_str(), Some("gguf") | Some("safetensors") | Some("bin"))
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
            {
                models.push(name.to_string());
            }
        }
    }

    models.sort();
    Ok(models)
}

/// Get device information
#[pyfunction]
fn get_device_info(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let info = PyDict::new(py);

    // CPU information
    let cpu_info = PyDict::new(py);
    cpu_info.set_item("cores", num_cpus::get())?;
    cpu_info.set_item("available", true)?;
    info.set_item("cpu", cpu_info)?;

    // GPU information
    let gpu_info = PyDict::new(py);
    gpu_info.set_item("cuda_available", is_cuda_available())?;
    gpu_info.set_item("metal_available", is_metal_available())?;
    gpu_info.set_item("cuda_devices", get_cuda_device_count())?;
    info.set_item("gpu", gpu_info)?;

    Ok(info.into())
}

/// Set the number of CPU threads to use
#[pyfunction]
fn set_num_threads(num_threads: usize) -> PyResult<()> {
    // SAFETY: Setting environment variable is safe in this context as it's
    // used to configure thread pool size and doesn't affect memory safety
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", num_threads.to_string());
    }
    Ok(())
}

/// Check if CUDA is available
#[pyfunction]
fn is_cuda_available() -> bool {
    cfg!(feature = "gpu")
}

/// Check if Metal is available
#[pyfunction]
fn is_metal_available() -> bool {
    cfg!(all(feature = "gpu", target_os = "macos"))
}

/// Get the number of CUDA devices
#[pyfunction]
fn get_cuda_device_count() -> usize {
    if is_cuda_available() {
        // In a real implementation, this would query CUDA
        1
    } else {
        0
    }
}

/// Parse device string to Device enum
fn parse_device(device_str: &str) -> PyResult<bitnet_common::Device> {
    match device_str.to_lowercase().as_str() {
        "cpu" => Ok(bitnet_common::Device::Cpu),
        "cuda" | "cuda:0" => Ok(bitnet_common::Device::Cuda(0)),
        "metal" => Ok(bitnet_common::Device::Metal),
        device if device.starts_with("cuda:") => {
            let device_id = device[5..].parse::<usize>().map_err(|_| {
                PyValueError::new_err(format!("Invalid CUDA device ID: {}", device))
            })?;
            Ok(bitnet_common::Device::Cuda(device_id))
        }
        _ => Err(PyValueError::new_err(format!("Unsupported device: {}", device_str))),
    }
}

/// Convert Rust Result to Python Result
pub fn to_py_result<T, E: std::fmt::Display>(result: Result<T, E>) -> PyResult<T> {
    result.map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Convert device enum to string
pub fn device_to_string(device: &bitnet_common::Device) -> String {
    match device {
        bitnet_common::Device::Cpu => "cpu".to_string(),
        bitnet_common::Device::Cuda(id) => format!("cuda:{}", id),
        bitnet_common::Device::Metal => "metal".to_string(),
        bitnet_common::Device::OpenCL(id) => format!("opencl:{}", id),
        bitnet_common::Device::Hip(id) => format!("hip:{}", id),
        bitnet_common::Device::Npu => "npu".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_device() {
        assert!(matches!(parse_device("cpu").unwrap(), bitnet_common::Device::Cpu));
        assert!(matches!(parse_device("cuda").unwrap(), bitnet_common::Device::Cuda(0)));
        assert!(matches!(parse_device("cuda:1").unwrap(), bitnet_common::Device::Cuda(1)));
        assert!(matches!(parse_device("metal").unwrap(), bitnet_common::Device::Metal));
        assert!(parse_device("invalid").is_err());
    }

    #[test]
    fn test_device_to_string() {
        assert_eq!(device_to_string(&bitnet_common::Device::Cpu), "cpu");
        assert_eq!(device_to_string(&bitnet_common::Device::Cuda(0)), "cuda:0");
        assert_eq!(device_to_string(&bitnet_common::Device::Metal), "metal");
    }
}
