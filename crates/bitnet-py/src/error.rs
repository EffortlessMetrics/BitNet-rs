use bitnet_common::BitNetError;
use pyo3::exceptions::{PyException, PyIOError, PyRuntimeError};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Exception hierarchy — each maps to a BitNetError variant so Python callers
// can catch at the right granularity.
// ---------------------------------------------------------------------------

pyo3::create_exception!(
    bitnet_py,
    BitNetBaseError,
    PyException,
    "Base class for all BitNet errors."
);
pyo3::create_exception!(bitnet_py, ModelError, BitNetBaseError, "Model loading / format errors.");
pyo3::create_exception!(
    bitnet_py,
    QuantizationError,
    BitNetBaseError,
    "Quantization-related errors."
);
pyo3::create_exception!(bitnet_py, InferenceError, BitNetBaseError, "Inference runtime errors.");
pyo3::create_exception!(bitnet_py, KernelError, BitNetBaseError, "Compute kernel errors.");
pyo3::create_exception!(bitnet_py, ConfigError, BitNetBaseError, "Configuration errors.");
pyo3::create_exception!(bitnet_py, ValidationError, BitNetBaseError, "Model validation errors.");

/// Register all exception types on the Python module.
pub fn register_exceptions(py: Python<'_>, m: &pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    m.add("BitNetBaseError", py.get_type::<BitNetBaseError>())?;
    m.add("ModelError", py.get_type::<ModelError>())?;
    m.add("QuantizationError", py.get_type::<QuantizationError>())?;
    m.add("InferenceError", py.get_type::<InferenceError>())?;
    m.add("KernelError", py.get_type::<KernelError>())?;
    m.add("ConfigError", py.get_type::<ConfigError>())?;
    m.add("ValidationError", py.get_type::<ValidationError>())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Conversion helper — orphan rule prevents `impl From<BitNetError> for PyErr`
// so we provide a free function instead.
// ---------------------------------------------------------------------------

/// Convert a `BitNetError` into the matching Python exception.
#[allow(dead_code)]
pub fn bitnet_err_to_pyerr(err: BitNetError) -> PyErr {
    match err {
        BitNetError::Model(e) => ModelError::new_err(format!("{e}")),
        BitNetError::Quantization(e) => QuantizationError::new_err(format!("{e}")),
        BitNetError::Kernel(e) => KernelError::new_err(format!("{e}")),
        BitNetError::Inference(e) => InferenceError::new_err(format!("{e}")),
        BitNetError::Io(e) => PyIOError::new_err(format!("IO error: {e}")),
        BitNetError::Config(e) => ConfigError::new_err(e),
        BitNetError::Validation(e) => ValidationError::new_err(e),
        _ => PyRuntimeError::new_err(format!("BitNet error: {err}")),
    }
}
