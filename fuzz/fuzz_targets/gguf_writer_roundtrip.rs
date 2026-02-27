#![no_main]

use arbitrary::Arbitrary;
use bitnet_gguf::parse_header;
use bitnet_st2gguf::writer::{GgufWriter, MetadataValue};
use libfuzzer_sys::fuzz_target;

/// Structured input driving the GgufWriter → GgufReader round-trip.
#[derive(Arbitrary, Debug)]
struct WriterInput {
    /// Used as the `general.name` metadata string.
    model_name: String,
    /// Written as a u32 context-length entry.
    context_length: u32,
    /// An optional extra string metadata entry.
    extra_value: String,
    include_extra: bool,
    /// An optional bool metadata entry.
    flag_value: bool,
    include_flag: bool,
}

fuzz_target!(|input: WriterInput| {
    // Cap string sizes to keep individual runs fast.
    if input.model_name.len() > 128 || input.extra_value.len() > 256 {
        return;
    }

    // Build a minimal GGUF using GgufWriter.
    let mut writer = GgufWriter::new();
    writer.add_metadata("general.name", MetadataValue::String(input.model_name.clone()));
    writer.add_metadata("llama.context_length", MetadataValue::U32(input.context_length));

    let mut expected_kv: u64 = 2;
    if input.include_extra {
        writer.add_metadata("extra.value", MetadataValue::String(input.extra_value.clone()));
        expected_kv += 1;
    }
    if input.include_flag {
        writer.add_metadata("general.quantized", MetadataValue::Bool(input.flag_value));
        expected_kv += 1;
    }

    // Write to a temporary file; skip if the OS refuses (e.g. disk full).
    let tmp = match tempfile::NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };
    let path = tmp.path().to_path_buf();
    if writer.write_to_file(&path).is_err() {
        return;
    }

    // Read back the raw bytes and parse the header.
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return,
    };

    let info = match parse_header(&bytes) {
        Ok(h) => h,
        Err(e) => panic!("parse_header failed on GgufWriter output: {e}"),
    };

    // Invariants: GgufWriter always produces v3 and the count we wrote.
    assert_eq!(info.version, 3, "GgufWriter must produce GGUF v3");
    assert_eq!(
        info.metadata_count,
        expected_kv,
        "metadata_count {actual} != expected {expected_kv}",
        actual = info.metadata_count,
    );
});
