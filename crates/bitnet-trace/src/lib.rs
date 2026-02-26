//! Tensor activation tracing for BitNet-rs cross-validation debugging.
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
//! dump_trace("layer_output", &tensor, None, None, None)?;
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
    /// Token position (0 = prefill, 1+ = decode)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seq: Option<usize>,
    /// Layer index (-1 = embeddings/logits, 0+ = transformer layers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<isize>,
    /// Stage name (e.g., "embeddings", "q_proj", "attn_out", "ffn_out", "logits")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
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
/// * `seq` - Optional token position (0 = prefill, 1+ = decode)
/// * `layer` - Optional layer index (-1 = embeddings/logits, 0+ = transformer layers)
/// * `stage` - Optional stage name (e.g., "embeddings", "q_proj", "attn_out")
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
/// dump_trace("my_layer/output", &tensor, None, None, None)?;
/// # Ok(())
/// # }
/// ```
pub fn dump_trace(
    name: &str,
    tensor: &Tensor,
    seq: Option<usize>,
    layer: Option<isize>,
    stage: Option<&str>,
) -> candle_core::Result<()> {
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
    let record = TraceRecord {
        name: name.to_string(),
        shape,
        dtype,
        blake3: blake3_hex,
        rms,
        num_elements,
        seq,
        layer,
        stage: stage.map(|s| s.to_string()),
    };

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
    use serial_test::serial;

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("blk0/attn_norm"), "blk0_attn_norm");
        assert_eq!(sanitize_filename("layer\\weights"), "layer_weights");
        assert_eq!(sanitize_filename("simple_name"), "simple_name");
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_dump_trace_disabled() {
        // Ensure BITNET_TRACE_DIR is not set
        temp_env::with_var_unset("BITNET_TRACE_DIR", || {
            let tensor = Tensor::randn(0f32, 1f32, (4, 8), &Device::Cpu).unwrap();
            let result = dump_trace("test_disabled", &tensor, None, None, None);

            // Should succeed silently when tracing is disabled
            assert!(result.is_ok());
        });
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
            seq: None,
            layer: None,
            stage: None,
        };

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: TraceRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(record.name, deserialized.name);
        assert_eq!(record.shape, deserialized.shape);
        assert_eq!(record.dtype, deserialized.dtype);
        assert_eq!(record.blake3, deserialized.blake3);
        assert_eq!(record.rms, deserialized.rms);
        assert_eq!(record.num_elements, deserialized.num_elements);
        assert_eq!(record.seq, deserialized.seq);
        assert_eq!(record.layer, deserialized.layer);
        assert_eq!(record.stage, deserialized.stage);
    }

    #[test]
    fn test_trace_record_optional_fields_omitted() {
        // Test that None fields are omitted from JSON (backward compatibility)
        let record = TraceRecord {
            name: "test".to_string(),
            shape: vec![2, 2],
            dtype: "F32".to_string(),
            blake3: "abc123".to_string(),
            rms: 1.5,
            num_elements: 4,
            seq: None,
            layer: None,
            stage: None,
        };

        let json = serde_json::to_string(&record).unwrap();

        // Verify optional fields are not present in JSON
        assert!(!json.contains("\"seq\""), "seq should be omitted when None");
        assert!(!json.contains("\"layer\""), "layer should be omitted when None");
        assert!(!json.contains("\"stage\""), "stage should be omitted when None");

        // Verify required fields are present
        assert!(json.contains("\"name\""));
        assert!(json.contains("\"shape\""));
        assert!(json.contains("\"dtype\""));
        assert!(json.contains("\"blake3\""));
        assert!(json.contains("\"rms\""));
        assert!(json.contains("\"num_elements\""));
    }

    #[test]
    fn test_trace_record_optional_fields_included() {
        // Test that Some fields are included in JSON
        let record = TraceRecord {
            name: "test".to_string(),
            shape: vec![2, 2],
            dtype: "F32".to_string(),
            blake3: "abc123".to_string(),
            rms: 1.5,
            num_elements: 4,
            seq: Some(0),
            layer: Some(-1),
            stage: Some("embeddings".to_string()),
        };

        let json = serde_json::to_string(&record).unwrap();

        // Verify optional fields are present in JSON
        assert!(json.contains("\"seq\":0"), "seq should be included when Some");
        assert!(json.contains("\"layer\":-1"), "layer should be included when Some");
        assert!(json.contains("\"stage\":\"embeddings\""), "stage should be included when Some");
    }
}
