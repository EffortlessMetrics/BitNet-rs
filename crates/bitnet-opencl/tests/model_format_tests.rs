//! Integration tests for model format detection and unified loading.

use std::path::Path;

use bitnet_opencl::error::ModelFormatError;
use bitnet_opencl::{
    DeviceId, GgufLoader, ModelFormat, ModelFormatDetector, ModelLoaderRegistry, ModelMetadata,
    QuantizationHint, SafeTensorsLoader, UnifiedModelLoader, detect_format,
};

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Build a minimal GGUF v2 file with the given tensor count.
fn make_gguf_bytes(tensor_count: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF"); // magic
    buf.extend_from_slice(&2u32.to_le_bytes()); // version
    buf.extend_from_slice(&tensor_count.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_count
    // Pad to make it look like a real file.
    buf.extend_from_slice(&[0u8; 64]);
    buf
}

/// Build a minimal SafeTensors file.
fn make_safetensors_bytes() -> Vec<u8> {
    let header = br#"{"weight":{"dtype":"F32","shape":[4,4],"data_offsets":[0,64]}}"#;
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header);
    buf.extend_from_slice(&[0u8; 64]); // dummy tensor data
    buf
}

/// Build bytes that look like an ONNX protobuf header.
fn make_onnx_bytes() -> Vec<u8> {
    // field tag 0x08, IR version 7, then filler.
    let mut buf = vec![0x08, 0x07, 0x12, 0x0A];
    buf.extend_from_slice(&[0u8; 60]);
    buf
}

// -----------------------------------------------------------------------
// Format detection – magic bytes
// -----------------------------------------------------------------------

#[test]
fn gguf_magic_bytes_detected() {
    let data = make_gguf_bytes(10);
    assert_eq!(ModelFormatDetector::from_magic_bytes(&data), Some(ModelFormat::Gguf));
}

#[test]
fn safetensors_header_detected() {
    let data = make_safetensors_bytes();
    assert_eq!(ModelFormatDetector::from_magic_bytes(&data), Some(ModelFormat::SafeTensors));
}

#[test]
fn onnx_magic_bytes_detected() {
    let data = make_onnx_bytes();
    assert_eq!(ModelFormatDetector::from_magic_bytes(&data), Some(ModelFormat::Onnx));
}

#[test]
fn unknown_magic_bytes_returns_none() {
    let data = [0xFF; 64];
    assert_eq!(ModelFormatDetector::from_magic_bytes(&data), None);
}

#[test]
fn empty_buffer_returns_none() {
    assert_eq!(ModelFormatDetector::from_magic_bytes(&[]), None);
}

// -----------------------------------------------------------------------
// Format detection – extensions
// -----------------------------------------------------------------------

#[test]
fn extension_gguf_detected() {
    assert_eq!(ModelFormatDetector::from_extension(Path::new("m.gguf")), Some(ModelFormat::Gguf));
}

#[test]
fn extension_safetensors_detected() {
    assert_eq!(
        ModelFormatDetector::from_extension(Path::new("m.safetensors")),
        Some(ModelFormat::SafeTensors)
    );
}

#[test]
fn extension_onnx_detected() {
    assert_eq!(ModelFormatDetector::from_extension(Path::new("m.onnx")), Some(ModelFormat::Onnx));
}

#[test]
fn extension_unknown_returns_none() {
    assert_eq!(ModelFormatDetector::from_extension(Path::new("m.pt")), None);
}

// -----------------------------------------------------------------------
// detect_format (file-based)
// -----------------------------------------------------------------------

#[test]
fn detect_format_file_not_found_error() {
    let err = detect_format(Path::new("definitely_does_not_exist.bin")).unwrap_err();
    assert!(matches!(err, ModelFormatError::FileNotFound(_)), "expected FileNotFound, got: {err}");
}

#[test]
fn detect_format_unknown_file_error_with_suggestion() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.xyz");
    std::fs::write(&path, &[0xFFu8; 64]).unwrap();
    let err = detect_format(&path).unwrap_err();
    match err {
        ModelFormatError::UnknownFormat { suggestion, .. } => {
            assert!(suggestion.contains("Supported formats"), "should mention supported formats");
        }
        other => {
            panic!("expected UnknownFormat, got: {other}")
        }
    }
}

#[test]
fn detect_format_gguf_by_extension() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, &make_gguf_bytes(5)).unwrap();
    assert_eq!(detect_format(&path).unwrap(), ModelFormat::Gguf);
}

#[test]
fn detect_format_safetensors_by_magic() {
    let dir = tempfile::tempdir().unwrap();
    // Use .bin extension so it falls through to magic-byte check.
    let path = dir.path().join("model.bin");
    std::fs::write(&path, &make_safetensors_bytes()).unwrap();
    assert_eq!(detect_format(&path).unwrap(), ModelFormat::SafeTensors);
}

// -----------------------------------------------------------------------
// Metadata extraction
// -----------------------------------------------------------------------

#[test]
fn gguf_loader_extracts_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, &make_gguf_bytes(30)).unwrap();
    let meta = GgufLoader.load_metadata(&path).unwrap();
    assert_eq!(meta.format, ModelFormat::Gguf);
    assert!(meta.num_layers.is_some(), "should estimate layers from tensor count");
}

#[test]
fn safetensors_loader_extracts_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.safetensors");
    std::fs::write(&path, &make_safetensors_bytes()).unwrap();
    let meta = SafeTensorsLoader.load_metadata(&path).unwrap();
    assert_eq!(meta.format, ModelFormat::SafeTensors);
}

#[test]
fn gguf_loader_file_not_found() {
    let err = GgufLoader.load_metadata(Path::new("nope.gguf")).unwrap_err();
    assert!(matches!(err, ModelFormatError::FileNotFound(_)), "expected FileNotFound");
}

#[test]
fn gguf_loader_corrupt_header_too_small() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    std::fs::write(&path, b"GGUF\x02\x00").unwrap();
    let err = GgufLoader.load_metadata(&path).unwrap_err();
    assert!(
        matches!(
            err,
            ModelFormatError::CorruptHeader { position, .. }
                if position < 24
        ),
        "expected CorruptHeader with position info, got: {err}"
    );
}

#[test]
fn gguf_loader_corrupt_header_bad_magic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.gguf");
    let mut data = make_gguf_bytes(5);
    data[0] = b'X';
    std::fs::write(&path, &data).unwrap();
    let err = GgufLoader.load_metadata(&path).unwrap_err();
    assert!(
        matches!(err, ModelFormatError::CorruptHeader { .. }),
        "expected CorruptHeader, got: {err}"
    );
}

#[test]
fn safetensors_loader_corrupt_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.safetensors");
    // Write header length pointing to data, but no JSON.
    let mut data = Vec::new();
    data.extend_from_slice(&100u64.to_le_bytes());
    data.extend_from_slice(&[0xFFu8; 100]); // garbage, not JSON
    std::fs::write(&path, &data).unwrap();
    let err = SafeTensorsLoader.load_metadata(&path).unwrap_err();
    assert!(
        matches!(err, ModelFormatError::CorruptHeader { .. }),
        "expected CorruptHeader, got: {err}"
    );
}

// -----------------------------------------------------------------------
// Memory estimation
// -----------------------------------------------------------------------

#[test]
fn memory_estimate_known_config_7b() {
    let meta = ModelMetadata {
        format: ModelFormat::Gguf,
        num_layers: Some(32),
        hidden_size: Some(4096),
        vocab_size: Some(32000),
        quantization_type: QuantizationHint::TwoBit,
    };
    let mem = GgufLoader.estimate_memory(&meta);
    // 32 layers × 4096² × 4 × 0.25 bytes ≈ 512 MiB weight data.
    assert!(mem > 500_000_000, "7B-class 2-bit model should need >500MB, got {mem}");
    assert!(mem < 2_000_000_000, "7B-class 2-bit model should need <2GB, got {mem}");
}

#[test]
fn memory_estimate_known_config_2b() {
    let meta = ModelMetadata {
        format: ModelFormat::Gguf,
        num_layers: Some(24),
        hidden_size: Some(2048),
        vocab_size: Some(32000),
        quantization_type: QuantizationHint::TwoBit,
    };
    let mem = GgufLoader.estimate_memory(&meta);
    assert!(mem > 50_000_000, "2B-class model should need >50MB, got {mem}");
    assert!(mem < 500_000_000, "2B-class model should need <500MB, got {mem}");
}

#[test]
fn memory_estimate_fp32_larger_than_quantized() {
    let base = ModelMetadata {
        format: ModelFormat::Gguf,
        num_layers: Some(24),
        hidden_size: Some(2048),
        vocab_size: Some(32000),
        quantization_type: QuantizationHint::None,
    };
    let quant = ModelMetadata { quantization_type: QuantizationHint::FourBit, ..base.clone() };
    let fp32_mem = GgufLoader.estimate_memory(&base);
    let q4_mem = GgufLoader.estimate_memory(&quant);
    assert!(fp32_mem > q4_mem, "FP32 ({fp32_mem}) should be larger than Q4 ({q4_mem})");
}

// -----------------------------------------------------------------------
// Registry
// -----------------------------------------------------------------------

#[test]
fn registry_returns_correct_loader_for_gguf() {
    let reg = ModelLoaderRegistry::with_defaults();
    let loader = reg.get_loader(ModelFormat::Gguf).unwrap();
    assert_eq!(loader.format(), ModelFormat::Gguf);
}

#[test]
fn registry_returns_correct_loader_for_safetensors() {
    let reg = ModelLoaderRegistry::with_defaults();
    let loader = reg.get_loader(ModelFormat::SafeTensors).unwrap();
    assert_eq!(loader.format(), ModelFormat::SafeTensors);
}

#[test]
fn registry_error_for_unregistered_format() {
    let reg = ModelLoaderRegistry::new();
    assert!(reg.get_loader(ModelFormat::Onnx).is_err(), "expected error for unregistered format");
}

// -----------------------------------------------------------------------
// Load weights (end-to-end with temp files)
// -----------------------------------------------------------------------

#[test]
fn gguf_load_weights_returns_file_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    let data = make_gguf_bytes(10);
    std::fs::write(&path, &data).unwrap();
    let weights = GgufLoader.load_weights(&path, DeviceId(0)).unwrap();
    assert_eq!(weights.source_format, ModelFormat::Gguf);
    assert_eq!(weights.total_bytes, data.len() as u64);
}

#[test]
fn safetensors_load_weights_returns_file_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("w.safetensors");
    let data = make_safetensors_bytes();
    std::fs::write(&path, &data).unwrap();
    let weights = SafeTensorsLoader.load_weights(&path, DeviceId(0)).unwrap();
    assert_eq!(weights.source_format, ModelFormat::SafeTensors);
    assert_eq!(weights.total_bytes, data.len() as u64);
}

// -----------------------------------------------------------------------
// Edge cases and error messages
// -----------------------------------------------------------------------

#[test]
fn error_display_file_not_found() {
    let err = ModelFormatError::FileNotFound("/a/b.gguf".into());
    let msg = format!("{err}");
    assert!(msg.contains("/a/b.gguf"), "message: {msg}");
}

#[test]
fn error_display_corrupt_header() {
    let err = ModelFormatError::CorruptHeader {
        path: "model.gguf".into(),
        position: 4,
        detail: "bad version".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("byte 4"), "message: {msg}");
    assert!(msg.contains("bad version"), "message: {msg}");
}

#[test]
fn model_format_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ModelFormat>();
    assert_send_sync::<ModelMetadata>();
}

// -----------------------------------------------------------------------
// Ignored tests requiring real model files
// -----------------------------------------------------------------------

#[test]
#[ignore = "requires real GGUF model file - set BITNET_GGUF"]
fn load_real_gguf_model() {
    let path = std::env::var("BITNET_GGUF").expect("set BITNET_GGUF to a .gguf path");
    let meta = GgufLoader.load_metadata(Path::new(&path)).unwrap();
    assert_eq!(meta.format, ModelFormat::Gguf);
}

#[test]
#[ignore = "requires real SafeTensors file - set BITNET_SAFETENSORS"]
fn load_real_safetensors_model() {
    let path =
        std::env::var("BITNET_SAFETENSORS").expect("set BITNET_SAFETENSORS to a .safetensors path");
    let meta = SafeTensorsLoader.load_metadata(Path::new(&path)).unwrap();
    assert_eq!(meta.format, ModelFormat::SafeTensors);
}
