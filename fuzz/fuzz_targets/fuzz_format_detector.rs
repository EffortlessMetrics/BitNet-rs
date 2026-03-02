//! Fuzz format detection with random bytes and filenames.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct FormatInput {
    /// Raw bytes for magic-based detection
    magic_bytes: Vec<u8>,
    /// Filename for extension-based detection
    filename: String,
    /// Shard filename for parse_shard_info
    shard_filename: String,
    /// File size for DetectedModel
    size_bytes: u64,
}

fuzz_target!(|input: FormatInput| {
    use bitnet_models::format_detector::{
        DetectedModel, ModelFormat, available_conversions, find_conversion, parse_shard_info,
    };
    use std::path::{Path, PathBuf};

    // Magic-byte detection — must not panic on any byte sequence
    let bytes: &[u8] = if input.magic_bytes.len() > 1024 {
        &input.magic_bytes[..1024]
    } else {
        &input.magic_bytes
    };
    let fmt_magic = ModelFormat::from_magic(bytes);
    let _ = fmt_magic.display_name();
    let _ = fmt_magic.is_supported();
    let _ = fmt_magic.needs_conversion();
    let _ = format!("{fmt_magic}");
    let _ = format!("{fmt_magic:?}");

    // Extension-based detection — must not panic on any path
    let fname: String = input.filename.chars().take(256).collect();
    let fmt_ext = ModelFormat::from_extension(Path::new(&fname));
    let _ = fmt_ext.display_name();
    let _ = fmt_ext.is_supported();
    let _ = fmt_ext.needs_conversion();

    // Shard parsing — must not panic on any string
    let shard: String = input.shard_filename.chars().take(256).collect();
    let _ = parse_shard_info(&shard);

    // DetectedModel construction
    let model = DetectedModel::new(PathBuf::from(&fname), fmt_ext, input.size_bytes);
    let _ = model.is_sharded();
    let _ = model.size_mb();
    let _ = model.size_gb();
    let _ = format!("{model:?}");

    // Sharded model
    let sharded =
        DetectedModel::new(PathBuf::from(&fname), fmt_ext, input.size_bytes).with_shard_info(1, 4);
    let _ = sharded.is_sharded();

    // Conversion queries — must not panic
    let _ = available_conversions();
    let _ = find_conversion(fmt_ext, ModelFormat::Gguf);
    let _ = find_conversion(fmt_magic, ModelFormat::SafeTensors);

    // Equality
    let _ = fmt_magic == fmt_ext;
    let _ = fmt_magic == ModelFormat::Unknown;
});
