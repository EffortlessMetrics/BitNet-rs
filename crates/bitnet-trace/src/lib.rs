//! Tensor activation tracing for BitNet.rs cross-validation debugging.
//!
//! This crate provides utilities for capturing and hashing tensor activations
//! during inference, enabling systematic cross-validation against reference
//! implementations.
//!
//! # Usage
//!
//! Tracing is controlled via the `BITNET_TRACE_DIR` environment variable:
//!
//! ```bash
//! export BITNET_TRACE_DIR=/tmp/bitnet-traces
//! cargo run -p bitnet-cli -- run --model model.gguf --prompt "test"
//! ```
//!
//! Trace files are written as JSON with the following format:
//!
//! ```json
//! {
//!   "name": "blk0/attn_norm",
//!   "shape": [1, 2560],
//!   "dtype": "F32",
//!   "blake3": "abc123...",
//!   "rms": 0.9982,
//!   "num_elements": 2560
//! }
//! ```
//!
//! # Example
//!
//! ```no_run
//! use candle_core::{Tensor, Device};
//! use bitnet_trace::dump_trace;
//!
//! # fn example() -> candle_core::Result<()> {
//! let tensor = Tensor::randn(0f32, 1f32, (32, 128), &Device::Cpu)?;
//! dump_trace("layer_output", &tensor)?;
//! # Ok(())
//! # }
//! ```

use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::PathBuf;

/// Trace record containing tensor metadata and hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    /// Tensor name/identifier
    pub name: String,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String,
    /// Blake3 hash of raw F32 bytes
    pub blake3: String,
    /// Root mean square of tensor values
    pub rms: f64,
    /// Total number of elements
    pub num_elements: usize,
}

/// Main tracing API - captures tensor activation and writes trace file.
///
/// Only performs tracing if `BITNET_TRACE_DIR` environment variable is set.
/// Returns `Ok(())` silently if tracing is disabled.
///
/// # Arguments
///
/// * `name` - Identifier for this tensor (e.g., "blk0/attn_norm")
/// * `tensor` - Tensor to trace
///
/// # Errors
///
/// Returns error if:
/// - Tensor conversion to F32 fails
/// - Trace directory cannot be created
/// - Trace file cannot be written
///
/// # Example
///
/// ```no_run
/// use candle_core::{Tensor, Device};
/// use bitnet_trace::dump_trace;
///
/// # fn example() -> candle_core::Result<()> {
/// let tensor = Tensor::randn(0f32, 1f32, (32, 128), &Device::Cpu)?;
/// dump_trace("my_layer/output", &tensor)?;
/// # Ok(())
/// # }
/// ```
pub fn dump_trace(name: &str, tensor: &Tensor) -> candle_core::Result<()> {
    // Check if tracing is enabled
    let trace_dir = match env::var("BITNET_TRACE_DIR") {
        Ok(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => return Ok(()), // Tracing disabled - return silently
    };

    // Create trace directory if it doesn't exist
    fs::create_dir_all(&trace_dir).map_err(|e| {
        candle_core::Error::Io(std::io::Error::other(format!(
            "Failed to create trace directory: {e}"
        )))
    })?;

    // Capture tensor metadata
    let shape = tensor.shape().dims().to_vec();
    let dtype = format!("{:?}", tensor.dtype());
    let num_elements = shape.iter().product();

    // Convert to F32 and flatten for hashing and RMS computation
    let tensor_f32 = tensor.to_dtype(DType::F32)?.flatten_all()?;
    let data = tensor_f32.to_vec1::<f32>()?;

    // Compute Blake3 hash of raw bytes
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let hash = blake3::hash(&bytes);
    let blake3_hex = hash.to_hex().to_string();

    // Compute RMS (root mean square)
    let sum_squares: f64 = data.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let rms = (sum_squares / data.len() as f64).sqrt();

    // Create trace record
    let record =
        TraceRecord { name: name.to_string(), shape, dtype, blake3: blake3_hex, rms, num_elements };

    // Write to trace file
    let trace_file = trace_dir.join(format!("{}.trace", sanitize_filename(name)));
    let json = serde_json::to_string_pretty(&record)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to serialize trace record: {}", e)))?;

    fs::write(&trace_file, json).map_err(|e| {
        candle_core::Error::Io(std::io::Error::other(format!("Failed to write trace file: {e}")))
    })?;

    Ok(())
}

/// Sanitize filename by replacing path separators with underscores.
fn sanitize_filename(name: &str) -> String {
    name.replace(['/', '\\'], "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::env;

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("blk0/attn_norm"), "blk0_attn_norm");
        assert_eq!(sanitize_filename("layer\\weights"), "layer_weights");
        assert_eq!(sanitize_filename("simple_name"), "simple_name");
    }

    #[test]
    fn test_dump_trace_disabled() {
        // Ensure BITNET_TRACE_DIR is not set
        env::remove_var("BITNET_TRACE_DIR");

        let tensor = Tensor::randn(0f32, 1f32, (4, 8), &Device::Cpu).unwrap();
        let result = dump_trace("test_disabled", &tensor);

        // Should succeed silently when tracing is disabled
        assert!(result.is_ok());
    }

    #[test]
    fn test_trace_record_serialization() {
        let record = TraceRecord {
            name: "test_layer".to_string(),
            shape: vec![1, 2560],
            dtype: "F32".to_string(),
            blake3: "abc123".to_string(),
            rms: 0.9982,
            num_elements: 2560,
        };

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: TraceRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(record.name, deserialized.name);
        assert_eq!(record.shape, deserialized.shape);
        assert_eq!(record.dtype, deserialized.dtype);
        assert_eq!(record.blake3, deserialized.blake3);
        assert_eq!(record.rms, deserialized.rms);
        assert_eq!(record.num_elements, deserialized.num_elements);
    }
}
