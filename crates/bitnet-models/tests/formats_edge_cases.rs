//! Edge-case tests for model format detection and loader infrastructure.
//!
//! Tests cover ModelFormat enum, path-based detection, header-based detection,
//! serde roundtrips, SafeTensors/HuggingFace loader traits.

use bitnet_models::FormatLoader;
use bitnet_models::formats::ModelFormat;
use std::io::Write;
use std::path::Path;
use tempfile::NamedTempFile;

// ── ModelFormat enum basics ──────────────────────────────────────

#[test]
fn model_format_debug() {
    let st = ModelFormat::SafeTensors;
    let gguf = ModelFormat::Gguf;
    assert!(format!("{:?}", st).contains("SafeTensors"));
    assert!(format!("{:?}", gguf).contains("Gguf"));
}

#[test]
fn model_format_clone() {
    let f = ModelFormat::SafeTensors;
    let f2 = f.clone();
    assert_eq!(f, f2);
}

#[test]
fn model_format_copy() {
    let f = ModelFormat::Gguf;
    let f2 = f; // Copy
    let _f3 = f; // Still usable — Copy
    assert_eq!(f2, ModelFormat::Gguf);
}

#[test]
fn model_format_eq() {
    assert_eq!(ModelFormat::SafeTensors, ModelFormat::SafeTensors);
    assert_eq!(ModelFormat::Gguf, ModelFormat::Gguf);
    assert_ne!(ModelFormat::SafeTensors, ModelFormat::Gguf);
}

#[test]
fn model_format_name_safetensors() {
    assert_eq!(ModelFormat::SafeTensors.name(), "SafeTensors");
}

#[test]
fn model_format_name_gguf() {
    assert_eq!(ModelFormat::Gguf.name(), "GGUF");
}

#[test]
fn model_format_extension_safetensors() {
    assert_eq!(ModelFormat::SafeTensors.extension(), "safetensors");
}

#[test]
fn model_format_extension_gguf() {
    assert_eq!(ModelFormat::Gguf.extension(), "gguf");
}

// ── Serde roundtrip ──────────────────────────────────────────────

#[test]
fn model_format_serde_safetensors() {
    let f = ModelFormat::SafeTensors;
    let json = serde_json::to_string(&f).unwrap();
    let f2: ModelFormat = serde_json::from_str(&json).unwrap();
    assert_eq!(f, f2);
}

#[test]
fn model_format_serde_gguf() {
    let f = ModelFormat::Gguf;
    let json = serde_json::to_string(&f).unwrap();
    let f2: ModelFormat = serde_json::from_str(&json).unwrap();
    assert_eq!(f, f2);
}

// ── Path-based detection ─────────────────────────────────────────

#[test]
fn detect_gguf_from_extension() {
    let p = Path::new("model.gguf");
    let fmt = ModelFormat::detect_from_path(p).unwrap();
    assert_eq!(fmt, ModelFormat::Gguf);
}

#[test]
fn detect_safetensors_from_extension() {
    let p = Path::new("model.safetensors");
    let fmt = ModelFormat::detect_from_path(p).unwrap();
    assert_eq!(fmt, ModelFormat::SafeTensors);
}

#[test]
fn detect_bin_as_safetensors() {
    let p = Path::new("pytorch_model.bin");
    let fmt = ModelFormat::detect_from_path(p).unwrap();
    assert_eq!(fmt, ModelFormat::SafeTensors);
}

#[test]
fn detect_pt_as_safetensors() {
    let p = Path::new("model.pt");
    let fmt = ModelFormat::detect_from_path(p).unwrap();
    assert_eq!(fmt, ModelFormat::SafeTensors);
}

#[test]
fn detect_gguf_case_insensitive() {
    let p = Path::new("MODEL.GGUF");
    let fmt = ModelFormat::detect_from_path(p).unwrap();
    assert_eq!(fmt, ModelFormat::Gguf);
}

#[test]
fn detect_safetensors_case_insensitive() {
    let p = Path::new("model.SafeTensors");
    let fmt = ModelFormat::detect_from_path(p).unwrap();
    assert_eq!(fmt, ModelFormat::SafeTensors);
}

#[test]
fn detect_no_extension_errors_for_nonexistent_path() {
    // A path with no extension and no file to read header from
    let p = Path::new("model_without_extension");
    let result = ModelFormat::detect_from_path(p);
    assert!(result.is_err());
}

#[test]
fn detect_path_no_extension_dir_style() {
    let p = Path::new("/tmp/models/mymodel");
    let result = ModelFormat::detect_from_path(p);
    assert!(result.is_err());
}

// ── Header-based detection ───────────────────────────────────────

#[test]
fn detect_gguf_from_header_magic() {
    let mut f = NamedTempFile::new().unwrap();
    // GGUF magic bytes: "GGUF"
    let mut data = b"GGUF".to_vec();
    data.extend_from_slice(&[0u8; 12]); // pad to 16 bytes
    f.write_all(&data).unwrap();
    f.flush().unwrap();

    let fmt = ModelFormat::detect_from_header(f.path()).unwrap();
    assert_eq!(fmt, ModelFormat::Gguf);
}

#[test]
fn detect_safetensors_from_header_json_at_0() {
    let mut f = NamedTempFile::new().unwrap();
    // SafeTensors can start with '{' at byte 0
    let mut data = b"{\"metadata\":{}".to_vec();
    data.resize(16, 0);
    f.write_all(&data).unwrap();
    f.flush().unwrap();

    let fmt = ModelFormat::detect_from_header(f.path()).unwrap();
    assert_eq!(fmt, ModelFormat::SafeTensors);
}

#[test]
fn detect_safetensors_from_header_json_at_8() {
    let mut f = NamedTempFile::new().unwrap();
    // SafeTensors: 8-byte size prefix then JSON
    let mut data = vec![0u8; 8];
    data.push(b'{'); // byte 8 is '{'
    data.resize(16, 0);
    f.write_all(&data).unwrap();
    f.flush().unwrap();

    let fmt = ModelFormat::detect_from_header(f.path()).unwrap();
    assert_eq!(fmt, ModelFormat::SafeTensors);
}

#[test]
fn detect_unknown_header_defaults_to_safetensors() {
    let mut f = NamedTempFile::new().unwrap();
    // Random bytes — not GGUF magic, no JSON markers
    let data = vec![
        0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08,
    ];
    f.write_all(&data).unwrap();
    f.flush().unwrap();

    let fmt = ModelFormat::detect_from_header(f.path()).unwrap();
    assert_eq!(fmt, ModelFormat::SafeTensors);
}

#[test]
fn detect_from_header_file_too_small() {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(b"tiny").unwrap();
    f.flush().unwrap();

    let result = ModelFormat::detect_from_header(f.path());
    assert!(result.is_err());
}

#[test]
fn detect_from_header_empty_file() {
    let f = NamedTempFile::new().unwrap();

    let result = ModelFormat::detect_from_header(f.path());
    assert!(result.is_err());
}

#[test]
fn detect_from_header_nonexistent_file() {
    let p = Path::new("/nonexistent/path/to/model.bin");
    let result = ModelFormat::detect_from_header(p);
    assert!(result.is_err());
}

// ── Path detection falling back to header ────────────────────────

#[test]
fn detect_unknown_extension_falls_back_to_header_gguf() {
    let mut f = NamedTempFile::with_suffix(".xyz").unwrap();
    let mut data = b"GGUF".to_vec();
    data.extend_from_slice(&[0u8; 12]);
    f.write_all(&data).unwrap();
    f.flush().unwrap();

    let fmt = ModelFormat::detect_from_path(f.path()).unwrap();
    assert_eq!(fmt, ModelFormat::Gguf);
}

#[test]
fn detect_unknown_extension_falls_back_to_header_safetensors() {
    let mut f = NamedTempFile::with_suffix(".model").unwrap();
    let mut data = vec![0u8; 8];
    data.push(b'{');
    data.resize(16, 0);
    f.write_all(&data).unwrap();
    f.flush().unwrap();

    let fmt = ModelFormat::detect_from_path(f.path()).unwrap();
    assert_eq!(fmt, ModelFormat::SafeTensors);
}

// ── SafeTensors loader ───────────────────────────────────────────

#[test]
fn safetensors_loader_name() {
    use bitnet_models::formats::safetensors::SafeTensorsLoader;
    let loader = SafeTensorsLoader;
    assert_eq!(loader.name(), "SafeTensors");
}

#[test]
fn safetensors_loader_can_load_safetensors_ext() {
    use bitnet_models::formats::safetensors::SafeTensorsLoader;
    let loader = SafeTensorsLoader;
    assert!(loader.can_load(Path::new("model.safetensors")));
}

#[test]
fn safetensors_loader_cannot_load_gguf_ext() {
    use bitnet_models::formats::safetensors::SafeTensorsLoader;
    let loader = SafeTensorsLoader;
    assert!(!loader.can_load(Path::new("model.gguf")));
}

#[test]
fn safetensors_loader_detect_format_nonexistent() {
    use bitnet_models::formats::safetensors::SafeTensorsLoader;
    let loader = SafeTensorsLoader;
    let result = loader.detect_format(Path::new("/nonexistent/model.safetensors"));
    // Should either return error or Ok(false) for nonexistent file
    match result {
        Ok(detected) => {
            let _ = detected;
        }
        Err(_) => {} // Expected for nonexistent file
    }
}

#[test]
fn safetensors_loader_extract_metadata_nonexistent() {
    use bitnet_models::formats::safetensors::SafeTensorsLoader;
    let loader = SafeTensorsLoader;
    let result = loader.extract_metadata(Path::new("/nonexistent/model.safetensors"));
    assert!(result.is_err());
}

// ── HuggingFace loader ───────────────────────────────────────────

#[test]
fn huggingface_loader_name() {
    use bitnet_models::formats::huggingface::HuggingFaceLoader;
    let loader = HuggingFaceLoader;
    assert_eq!(loader.name(), "HuggingFace");
}

#[test]
fn huggingface_loader_can_load_nonexistent_dir() {
    use bitnet_models::formats::huggingface::HuggingFaceLoader;
    let loader = HuggingFaceLoader;
    // HuggingFace loader works with directories containing config.json
    assert!(!loader.can_load(Path::new("some/model/dir")));
}

#[test]
fn huggingface_loader_cannot_load_single_file() {
    use bitnet_models::formats::huggingface::HuggingFaceLoader;
    let loader = HuggingFaceLoader;
    assert!(!loader.can_load(Path::new("model.safetensors")));
}

#[test]
fn huggingface_loader_extract_metadata_nonexistent() {
    use bitnet_models::formats::huggingface::HuggingFaceLoader;
    let loader = HuggingFaceLoader;
    let result = loader.extract_metadata(Path::new("/nonexistent/model_dir"));
    assert!(result.is_err());
}

// ── ModelFormat exhaustive matching ──────────────────────────────

#[test]
fn model_format_all_variants_have_name() {
    let variants = [ModelFormat::SafeTensors, ModelFormat::Gguf];
    for v in &variants {
        assert!(!v.name().is_empty());
    }
}

#[test]
fn model_format_all_variants_have_extension() {
    let variants = [ModelFormat::SafeTensors, ModelFormat::Gguf];
    for v in &variants {
        assert!(!v.extension().is_empty());
    }
}

#[test]
fn model_format_extensions_are_lowercase() {
    let variants = [ModelFormat::SafeTensors, ModelFormat::Gguf];
    for v in &variants {
        let ext = v.extension();
        assert_eq!(ext, ext.to_lowercase());
    }
}

#[test]
fn model_format_names_are_unique() {
    let variants = [ModelFormat::SafeTensors, ModelFormat::Gguf];
    let names: Vec<_> = variants.iter().map(|v| v.name()).collect();
    let unique: std::collections::HashSet<_> = names.iter().collect();
    assert_eq!(names.len(), unique.len());
}
