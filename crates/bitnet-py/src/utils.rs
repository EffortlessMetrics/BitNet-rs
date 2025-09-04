//! Python Utility Functions

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Utility functions for BitNet Python bindings
///
/// Convert Python kwargs to Rust HashMap
pub fn kwargs_to_hashmap(
    kwargs: Option<&pyo3::Bound<'_, PyDict>>,
) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();

    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs {
            if let (Ok(key_str), Ok(value_str)) = (key.extract::<String>(), value.str())
                && let Ok(value_string) = value_str.extract::<String>()
            {
                map.insert(key_str, value_string);
            }
        }
    }

    map
}

/// Validate device string
pub fn validate_device(device: &str) -> PyResult<()> {
    match device.to_lowercase().as_str() {
        "cpu" | "cuda" | "metal" => Ok(()),
        device if device.starts_with("cuda:") => {
            let device_id = device[5..].parse::<usize>();
            if device_id.is_ok() {
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid CUDA device ID: {}",
                    device
                )))
            }
        }
        _ => {
            Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported device: {}", device)))
        }
    }
}
