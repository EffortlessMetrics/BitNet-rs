//! Integration tests for bitnet-trace crate.

use bitnet_trace::{dump_trace, TraceRecord};
use candle_core::{Device, Tensor};
use std::env;
use std::fs;
use std::sync::Mutex;
use tempfile::TempDir;

// Mutex to serialize tests that modify environment variables
static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_dump_trace_integration() {
    let _guard = ENV_LOCK.lock().unwrap();

    // Create temporary directory for traces
    let temp_dir = TempDir::new().unwrap();
    let trace_dir = temp_dir.path().to_str().unwrap();

    // Set environment variable
    env::set_var("BITNET_TRACE_DIR", trace_dir);

    // Create a test tensor with known values
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_vec(data.clone(), (2, 2), &Device::Cpu).unwrap();

    // Dump trace
    let result = dump_trace("test_layer/output", &tensor, None, None, None);
    assert!(result.is_ok(), "dump_trace should succeed");

    // Verify trace file exists
    let trace_file = temp_dir.path().join("test_layer_output.trace");
    assert!(trace_file.exists(), "Trace file should exist");

    // Read and parse trace file
    let contents = fs::read_to_string(&trace_file).unwrap();
    let record: TraceRecord = serde_json::from_str(&contents).unwrap();

    // Verify trace record fields
    assert_eq!(record.name, "test_layer/output");
    assert_eq!(record.shape, vec![2, 2]);
    assert_eq!(record.dtype, "F32");
    assert_eq!(record.num_elements, 4);

    // Verify RMS calculation
    // RMS = sqrt((1^2 + 2^2 + 3^2 + 4^2) / 4) = sqrt(30/4) = sqrt(7.5) ≈ 2.7386
    let expected_rms = (30.0f64 / 4.0).sqrt();
    assert!(
        (record.rms - expected_rms).abs() < 1e-6,
        "RMS should be approximately {}, got {}",
        expected_rms,
        record.rms
    );

    // Verify Blake3 hash is non-empty and valid hex
    assert!(!record.blake3.is_empty(), "Blake3 hash should not be empty");
    assert_eq!(record.blake3.len(), 64, "Blake3 hash should be 64 hex characters");
    assert!(
        record.blake3.chars().all(|c| c.is_ascii_hexdigit()),
        "Blake3 hash should be valid hex"
    );

    // Compute expected Blake3 hash
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let expected_hash = blake3::hash(&bytes).to_hex().to_string();
    assert_eq!(record.blake3, expected_hash, "Blake3 hash should match expected value");

    // Cleanup
    env::remove_var("BITNET_TRACE_DIR");
}

#[test]
fn test_dump_trace_creates_directory() {
    let _guard = ENV_LOCK.lock().unwrap();

    // Create temporary directory that we'll delete
    let temp_dir = TempDir::new().unwrap();
    let trace_dir = temp_dir.path().join("subdir").join("traces");
    let trace_path = trace_dir.to_str().unwrap().to_string();

    // Set environment variable to non-existent directory
    env::set_var("BITNET_TRACE_DIR", &trace_path);

    // Create tensor and dump trace
    let tensor = Tensor::randn(0f32, 1f32, (4, 8), &Device::Cpu).unwrap();
    let result = dump_trace("auto_create_dir", &tensor, None, None, None);

    assert!(result.is_ok(), "Should succeed even if directory doesn't exist");
    assert!(trace_dir.exists(), "Trace directory should be created automatically");

    // Verify trace file was written
    let trace_file = trace_dir.join("auto_create_dir.trace");
    assert!(trace_file.exists(), "Trace file should exist");

    // Cleanup
    env::remove_var("BITNET_TRACE_DIR");
}

#[test]
fn test_dump_trace_different_dtypes() {
    let _guard = ENV_LOCK.lock().unwrap();

    let temp_dir = TempDir::new().unwrap();
    let trace_dir = temp_dir.path().to_str().unwrap();
    env::set_var("BITNET_TRACE_DIR", trace_dir);

    // Test with different dtypes
    let tensor_f32 = Tensor::randn(0f32, 1f32, (4, 8), &Device::Cpu).unwrap();
    let tensor_f64 = Tensor::randn(0f64, 1f64, (4, 8), &Device::Cpu).unwrap();

    // Both should succeed (f64 gets converted to f32)
    assert!(dump_trace("dtype_f32", &tensor_f32, None, None, None).is_ok());
    assert!(dump_trace("dtype_f64", &tensor_f64, None, None, None).is_ok());

    // Verify both trace files exist
    assert!(temp_dir.path().join("dtype_f32.trace").exists());
    assert!(temp_dir.path().join("dtype_f64.trace").exists());

    // Read f64 trace to verify dtype was recorded before conversion
    let contents = fs::read_to_string(temp_dir.path().join("dtype_f64.trace")).unwrap();
    let record: TraceRecord = serde_json::from_str(&contents).unwrap();
    assert_eq!(record.dtype, "F64");

    env::remove_var("BITNET_TRACE_DIR");
}

#[test]
fn test_dump_trace_various_shapes() {
    let _guard = ENV_LOCK.lock().unwrap();

    let temp_dir = TempDir::new().unwrap();
    let trace_dir = temp_dir.path().to_str().unwrap();
    env::set_var("BITNET_TRACE_DIR", trace_dir);

    // Test various tensor shapes
    let shapes = [
        vec![1],          // 1D
        vec![4, 8],       // 2D
        vec![2, 3, 4],    // 3D
        vec![2, 2, 2, 2], // 4D
    ];

    for (i, shape) in shapes.iter().enumerate() {
        let tensor = Tensor::randn(0f32, 1f32, shape.as_slice(), &Device::Cpu).unwrap();
        let name = format!("shape_test_{}", i);
        assert!(dump_trace(&name, &tensor, None, None, None).is_ok());

        // Read and verify shape
        let trace_file = temp_dir.path().join(format!("{}.trace", name));
        let contents = fs::read_to_string(&trace_file).unwrap();
        let record: TraceRecord = serde_json::from_str(&contents).unwrap();
        assert_eq!(record.shape, *shape);

        let expected_elements: usize = shape.iter().product();
        assert_eq!(record.num_elements, expected_elements);
    }

    env::remove_var("BITNET_TRACE_DIR");
}

#[test]
fn test_dump_trace_empty_trace_dir() {
    let _guard = ENV_LOCK.lock().unwrap();

    // Test with empty string for BITNET_TRACE_DIR
    env::set_var("BITNET_TRACE_DIR", "");

    let tensor = Tensor::randn(0f32, 1f32, (4, 8), &Device::Cpu).unwrap();
    let result = dump_trace("empty_dir", &tensor, None, None, None);

    // Should succeed silently (tracing disabled)
    assert!(result.is_ok());

    env::remove_var("BITNET_TRACE_DIR");
}
