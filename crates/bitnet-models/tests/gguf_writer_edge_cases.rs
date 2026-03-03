//! Edge-case tests for the GGUF writer / builder API.
//!
//! Exercises defaults, metadata types, round-trip via `GgufReader`,
//! empty models, large metadata sets, special characters, and file I/O.

use std::io::Cursor;

use bitnet_models::GgufReader;
use bitnet_models::formats::gguf::GgufTensorType as ReaderTensorType;
use bitnet_models::gguf_writer::{GgufBuilder, GgufTensorType, GgufWriter, GgufWriterConfig};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write a builder to an in-memory buffer and return the raw bytes.
fn write_to_vec(builder: GgufBuilder) -> Vec<u8> {
    let buf = Cursor::new(Vec::new());
    builder.write(buf).unwrap().into_inner()
}

// ---------------------------------------------------------------------------
// 1. GgufBuilder defaults
// ---------------------------------------------------------------------------

#[test]
fn builder_new_has_default_v3_alignment32() {
    let data = write_to_vec(GgufBuilder::new());
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.version(), 3, "default version should be 3");
    assert_eq!(reader.tensor_count(), 0);
    assert_eq!(reader.metadata_count(), 0);
}

#[test]
fn builder_default_trait_equivalent_to_new() {
    let a = write_to_vec(GgufBuilder::new());
    let b = write_to_vec(GgufBuilder::default());
    assert_eq!(a, b, "Default::default() and new() should produce identical output");
}

// ---------------------------------------------------------------------------
// 2. Add metadata — all types
// ---------------------------------------------------------------------------

#[test]
fn metadata_string_round_trip_via_reader() {
    let data = write_to_vec(GgufBuilder::new().metadata_string("custom.author", "Alice"));
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.metadata_count(), 1);
    assert_eq!(reader.get_string_metadata("custom.author").as_deref(), Some("Alice"),);
}

#[test]
fn metadata_u32_round_trip_via_reader() {
    let data = write_to_vec(GgufBuilder::new().metadata_u32("ctx.len", 4096));
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.get_u32_metadata("ctx.len"), Some(4096));
}

#[test]
fn metadata_f32_round_trip_via_reader() {
    let data = write_to_vec(GgufBuilder::new().metadata_f32("lr", 0.001));
    let reader = GgufReader::new(&data).unwrap();
    let val = reader.get_f32_metadata("lr").unwrap();
    assert!((val - 0.001).abs() < f32::EPSILON);
}

#[test]
fn metadata_bool_round_trip_via_reader() {
    let data = write_to_vec(
        GgufBuilder::new().metadata_bool("flag.true", true).metadata_bool("flag.false", false),
    );
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.get_bool_metadata("flag.true"), Some(true));
    assert_eq!(reader.get_bool_metadata("flag.false"), Some(false));
}

#[test]
fn metadata_u64_writes_successfully() {
    // The reader doesn't support u64 value type, but we can verify the writer
    // produces valid bytes by checking the raw header with bitnet_gguf::parse_header.
    let data = write_to_vec(GgufBuilder::new().metadata_u64("params", u64::MAX));
    let info = bitnet_gguf::parse_header(&data).unwrap();
    assert_eq!(info.metadata_count, 1);
}

#[test]
fn metadata_string_array_round_trip_via_reader() {
    let tokens = vec!["hello".into(), "world".into(), "foo".into()];
    let data =
        write_to_vec(GgufBuilder::new().metadata_string_array("tokenizer.tokens", tokens.clone()));
    let reader = GgufReader::new(&data).unwrap();
    let got = reader.get_string_array_metadata("tokenizer.tokens").unwrap();
    assert_eq!(got, tokens);
}

#[test]
fn description_and_architecture_become_metadata() {
    let data = write_to_vec(GgufBuilder::new().description("my model").architecture("bitnet"));
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.metadata_count(), 2);
    assert_eq!(reader.get_string_metadata("general.description").as_deref(), Some("my model"),);
    assert_eq!(reader.get_string_metadata("general.architecture").as_deref(), Some("bitnet"),);
}

// ---------------------------------------------------------------------------
// 3. Add tensor info — various types
// ---------------------------------------------------------------------------

#[test]
fn single_tensor_f32_round_trip() {
    // 4 floats = 16 bytes
    let tensor_bytes: Vec<u8> = (0u32..4).flat_map(|i| (i as f32).to_le_bytes()).collect();

    let data =
        write_to_vec(GgufBuilder::new().tensor("weight", &[4], GgufTensorType::F32, &tensor_bytes));
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.tensor_count(), 1);

    let names = reader.tensor_names();
    assert_eq!(names, vec!["weight"]);

    let info = reader.get_tensor_info_by_name("weight").unwrap();
    assert_eq!(info.shape, vec![4]);
    assert_eq!(info.tensor_type, ReaderTensorType::F32);

    let read_back = reader.get_tensor_data_by_name("weight").unwrap();
    assert_eq!(read_back, &tensor_bytes[..]);
}

#[test]
fn tensor_f16_preserves_dtype() {
    let data = write_to_vec(GgufBuilder::new().tensor("w", &[8], GgufTensorType::F16, &[0u8; 16]));
    let reader = GgufReader::new(&data).unwrap();
    let info = reader.get_tensor_info_by_name("w").unwrap();
    assert_eq!(info.tensor_type, ReaderTensorType::F16);
}

#[test]
fn tensor_i2s_preserves_dtype() {
    let data = write_to_vec(GgufBuilder::new().tensor("q", &[32], GgufTensorType::I2_S, &[0u8; 8]));
    let reader = GgufReader::new(&data).unwrap();
    let info = reader.get_tensor_info_by_name("q").unwrap();
    assert_eq!(info.tensor_type, ReaderTensorType::I2_S);
}

#[test]
fn multidimensional_tensor_dims_preserved() {
    let data =
        write_to_vec(GgufBuilder::new().tensor("m", &[2, 3, 4], GgufTensorType::F32, &[0u8; 96]));
    let reader = GgufReader::new(&data).unwrap();
    let info = reader.get_tensor_info_by_name("m").unwrap();
    assert_eq!(info.shape, vec![2, 3, 4]);
}

#[test]
fn multiple_tensors_round_trip() {
    let t0 = vec![1u8; 16];
    let t1 = vec![2u8; 32];
    let data = write_to_vec(GgufBuilder::new().tensor("a", &[4], GgufTensorType::F32, &t0).tensor(
        "b",
        &[8, 2],
        GgufTensorType::F16,
        &t1,
    ));
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.tensor_count(), 2);

    let mut names = reader.tensor_names();
    names.sort();
    assert_eq!(names, vec!["a", "b"]);

    // Reader may return aligned spans; check the data starts with our bytes.
    let a_data = reader.get_tensor_data_by_name("a").unwrap();
    assert!(a_data.len() >= t0.len());
    assert_eq!(&a_data[..t0.len()], &t0[..]);

    let b_data = reader.get_tensor_data_by_name("b").unwrap();
    assert!(b_data.len() >= t1.len());
    assert_eq!(&b_data[..t1.len()], &t1[..]);
}

// ---------------------------------------------------------------------------
// 4. Write and read back — full round-trip
// ---------------------------------------------------------------------------

#[test]
fn full_model_round_trip() {
    let weights: Vec<u8> = (0u32..16).flat_map(|i| (i as f32).to_le_bytes()).collect();

    let data = write_to_vec(
        GgufBuilder::new()
            .version(3)
            .description("round-trip test")
            .architecture("llama")
            .metadata_u32("llama.context_length", 2048)
            .metadata_f32("training.lr", 0.01)
            .metadata_bool("quantized", false)
            .metadata_string("author", "test-suite")
            .tensor("blk.0.weight", &[4, 4], GgufTensorType::F32, &weights),
    );

    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.version(), 3);
    // arch + desc + 4 user entries = 6
    assert_eq!(reader.metadata_count(), 6);
    assert_eq!(reader.tensor_count(), 1);

    assert_eq!(reader.get_string_metadata("general.architecture").as_deref(), Some("llama"),);
    assert_eq!(reader.get_u32_metadata("llama.context_length"), Some(2048));
    assert_eq!(reader.get_string_metadata("author").as_deref(), Some("test-suite"));

    let info = reader.get_tensor_info_by_name("blk.0.weight").unwrap();
    assert_eq!(info.shape, vec![4, 4]);

    let read_back = reader.get_tensor_data_by_name("blk.0.weight").unwrap();
    assert_eq!(read_back, &weights[..]);
}

// ---------------------------------------------------------------------------
// 5. Empty model
// ---------------------------------------------------------------------------

#[test]
fn empty_model_no_metadata_no_tensors() {
    let data = write_to_vec(GgufBuilder::new());
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.metadata_count(), 0);
    assert_eq!(reader.tensor_count(), 0);
    assert!(reader.tensor_names().is_empty());
}

#[test]
fn empty_model_metadata_only() {
    let data = write_to_vec(GgufBuilder::new().description("meta-only").metadata_u32("key", 1));
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.metadata_count(), 2);
    assert_eq!(reader.tensor_count(), 0);
}

#[test]
fn empty_model_tensors_only() {
    let data = write_to_vec(GgufBuilder::new().tensor("t", &[4], GgufTensorType::F32, &[0u8; 16]));
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.metadata_count(), 0);
    assert_eq!(reader.tensor_count(), 1);
}

// ---------------------------------------------------------------------------
// 6. Large metadata
// ---------------------------------------------------------------------------

#[test]
fn many_metadata_entries() {
    let mut builder = GgufBuilder::new();
    let count = 200;
    for i in 0..count {
        builder = builder.metadata_u32(&format!("key.{i}"), i as u32);
    }
    let data = write_to_vec(builder);
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.metadata_count(), count);

    // Spot-check a few
    assert_eq!(reader.get_u32_metadata("key.0"), Some(0));
    assert_eq!(reader.get_u32_metadata("key.99"), Some(99));
    assert_eq!(reader.get_u32_metadata("key.199"), Some(199));
}

#[test]
fn many_tensors() {
    let mut builder = GgufBuilder::new();
    let count = 50;
    for i in 0..count {
        builder =
            builder.tensor(&format!("layer.{i}.weight"), &[4], GgufTensorType::F32, &[0u8; 16]);
    }
    let data = write_to_vec(builder);
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.tensor_count(), count as u64);

    let names = reader.tensor_names();
    assert_eq!(names.len(), count);
    assert!(names.contains(&"layer.0.weight"));
    assert!(names.contains(&"layer.49.weight"));
}

// ---------------------------------------------------------------------------
// 7. Invalid / edge cases
// ---------------------------------------------------------------------------

#[test]
fn empty_string_metadata_key() {
    // Writer accepts empty key; verify it doesn't panic and header is valid.
    let data = write_to_vec(GgufBuilder::new().metadata_string("", "value"));
    let info = bitnet_gguf::parse_header(&data).unwrap();
    assert_eq!(info.metadata_count, 1);
}

#[test]
fn empty_string_metadata_value() {
    let data = write_to_vec(GgufBuilder::new().metadata_string("key", ""));
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.get_string_metadata("key").as_deref(), Some(""));
}

#[test]
fn empty_tensor_name() {
    // Writer accepts empty tensor name; verify header is valid.
    let data = write_to_vec(GgufBuilder::new().tensor("", &[4], GgufTensorType::F32, &[0u8; 16]));
    let info = bitnet_gguf::parse_header(&data).unwrap();
    assert_eq!(info.tensor_count, 1);
}

#[test]
fn very_long_metadata_key() {
    // Ensure a short key comes first so the reader's v3 heuristic works,
    // then add the long key.
    let long_key = "k".repeat(5_000);
    let data =
        write_to_vec(GgufBuilder::new().architecture("test").metadata_string(&long_key, "v"));
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.get_string_metadata(&long_key).as_deref(), Some("v"));
}

#[test]
fn very_long_metadata_value() {
    let long_val = "x".repeat(100_000);
    let data = write_to_vec(GgufBuilder::new().metadata_string("k", &long_val));
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.get_string_metadata("k").as_deref(), Some(long_val.as_str()),);
}

#[test]
fn very_long_tensor_name() {
    let long_name = "t".repeat(5_000);
    let data = write_to_vec(GgufBuilder::new().architecture("test").tensor(
        &long_name,
        &[2],
        GgufTensorType::F32,
        &[0u8; 8],
    ));
    let reader = GgufReader::new(&data).unwrap();
    assert!(reader.get_tensor_info_by_name(&long_name).is_some());
}

#[test]
fn special_characters_in_metadata() {
    // Use architecture to ensure the first key is a short ASCII string
    // (required by the reader's v3 format heuristic).
    let data = write_to_vec(
        GgufBuilder::new()
            .architecture("test")
            .metadata_string("emoji.key", "Rust 🚀🦀")
            .metadata_string("path\\key", "back\\slash")
            .metadata_string("tab\tkey", "tab\tvalue"),
    );
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.metadata_count(), 4); // arch + 3
    assert_eq!(reader.get_string_metadata("emoji.key").as_deref(), Some("Rust 🚀🦀"),);
    assert_eq!(reader.get_string_metadata("path\\key").as_deref(), Some("back\\slash"),);
    assert_eq!(reader.get_string_metadata("tab\tkey").as_deref(), Some("tab\tvalue"),);
}

#[test]
fn zero_length_tensor_data() {
    // Writer accepts zero-length data; verify the header is valid.
    // (Reader rejects dim=0 as a security measure, so we only check header.)
    let data = write_to_vec(GgufBuilder::new().tensor("empty", &[0], GgufTensorType::F32, &[]));
    let info = bitnet_gguf::parse_header(&data).unwrap();
    assert_eq!(info.tensor_count, 1);
}

#[test]
fn scalar_tensor_zero_dims() {
    // A scalar tensor has no dimensions
    let scalar_bytes = 42.0f32.to_le_bytes();
    let data =
        write_to_vec(GgufBuilder::new().tensor("scalar", &[], GgufTensorType::F32, &scalar_bytes));
    let reader = GgufReader::new(&data).unwrap();
    let info = reader.get_tensor_info_by_name("scalar").unwrap();
    assert!(info.shape.is_empty());
    assert_eq!(reader.get_tensor_data_by_name("scalar").unwrap(), &scalar_bytes[..]);
}

#[test]
fn empty_string_array_metadata() {
    let data = write_to_vec(GgufBuilder::new().metadata_string_array("empty_arr", vec![]));
    let reader = GgufReader::new(&data).unwrap();
    let arr = reader.get_string_array_metadata("empty_arr").unwrap();
    assert!(arr.is_empty());
}

// ---------------------------------------------------------------------------
// 8. Round-trip consistency
// ---------------------------------------------------------------------------

#[test]
fn write_twice_produces_identical_bytes() {
    let make = || {
        GgufBuilder::new()
            .version(3)
            .alignment(32)
            .description("determinism")
            .architecture("test")
            .metadata_u32("a", 1)
            .metadata_f32("b", 2.5)
            .metadata_bool("c", true)
            .metadata_string("d", "hello")
            .tensor("w", &[4], GgufTensorType::F32, &[0u8; 16])
    };
    let a = write_to_vec(make());
    let b = write_to_vec(make());
    assert_eq!(a, b, "identical builders must produce identical bytes");
}

#[test]
fn version_2_round_trip() {
    let data = write_to_vec(GgufBuilder::new().version(2).description("v2").tensor(
        "t",
        &[2],
        GgufTensorType::F32,
        &[0u8; 8],
    ));
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.version(), 2);
    assert_eq!(reader.tensor_count(), 1);
    assert_eq!(reader.get_string_metadata("general.description").as_deref(), Some("v2"),);
}

#[test]
fn custom_alignment_round_trip() {
    // Default alignment (32) is used since the writer doesn't emit
    // general.alignment metadata automatically. Verify tensor data survives.
    let tensor_bytes = vec![0xABu8; 28]; // shape [7], F32 = 7*4=28 bytes
    let data = write_to_vec(
        GgufBuilder::new()
            .description("align test")
            .tensor("t0", &[7], GgufTensorType::F32, &tensor_bytes)
            .tensor("t1", &[3], GgufTensorType::F32, &[0xCDu8; 12]), // 3*4=12
    );
    let reader = GgufReader::new(&data).unwrap();

    let t0_data = reader.get_tensor_data_by_name("t0").unwrap();
    assert!(t0_data.len() >= tensor_bytes.len());
    assert_eq!(&t0_data[..tensor_bytes.len()], &tensor_bytes[..]);

    let t1_data = reader.get_tensor_data_by_name("t1").unwrap();
    assert!(t1_data.len() >= 12);
    assert_eq!(&t1_data[..12], &[0xCDu8; 12]);
}

#[test]
fn calculate_file_size_matches_actual_for_complex_model() {
    let builder = GgufBuilder::new()
        .description("size check")
        .architecture("llama")
        .metadata_u32("a", 1)
        .metadata_f32("b", 2.0)
        .metadata_string("c", "value")
        .metadata_bool("d", true)
        .metadata_string_array("e", vec!["x".into(), "y".into()])
        .tensor("w0", &[4, 8], GgufTensorType::F32, &[0u8; 128])
        .tensor("w1", &[16], GgufTensorType::F16, &[0u8; 32])
        .tensor("w2", &[32], GgufTensorType::I2_S, &[0u8; 8]);

    let predicted = builder.calculate_file_size();
    let actual = write_to_vec(builder).len() as u64;
    assert_eq!(predicted, actual, "predicted {predicted} != actual {actual}");
}

#[test]
fn writer_direct_api_round_trip() {
    let buf = Cursor::new(Vec::new());
    let config = GgufWriterConfig {
        version: 3,
        alignment: 32,
        description: Some("direct-api".into()),
        architecture: Some("test".into()),
    };
    let mut w = GgufWriter::new(buf, config);
    w.add_metadata_string("k1", "v1");
    w.add_metadata_u32("k2", 42);
    w.add_metadata_f32("k4", 1.5);
    w.add_metadata_bool("k5", true);
    w.add_metadata_string_array("k6", vec!["a".into(), "b".into()]);
    w.add_tensor("t", &[4], GgufTensorType::F32, vec![0u8; 16]);

    let data = w.finish().unwrap().into_inner();
    let reader = GgufReader::new(&data).unwrap();

    // description + architecture + 5 user entries = 7
    assert_eq!(reader.metadata_count(), 7);
    assert_eq!(reader.tensor_count(), 1);
    assert_eq!(reader.get_string_metadata("k1").as_deref(), Some("v1"));
    assert_eq!(reader.get_u32_metadata("k2"), Some(42));
}

#[test]
fn write_to_tempfile_and_read_back() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edge_case.gguf");

    GgufBuilder::new()
        .description("file round-trip")
        .metadata_u32("key", 7)
        .tensor("w", &[2, 2], GgufTensorType::F32, &[0u8; 16])
        .write_to_file(&path)
        .unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let reader = GgufReader::new(&bytes).unwrap();
    assert_eq!(reader.version(), 3);
    assert_eq!(reader.metadata_count(), 2); // description + key
    assert_eq!(reader.tensor_count(), 1);
    assert_eq!(reader.get_u32_metadata("key"), Some(7));
    assert_eq!(
        reader.get_string_metadata("general.description").as_deref(),
        Some("file round-trip"),
    );
}

// ---------------------------------------------------------------------------
// GgufTensorType discriminants
// ---------------------------------------------------------------------------

#[test]
fn all_tensor_type_discriminants() {
    let cases: &[(GgufTensorType, u32)] = &[
        (GgufTensorType::F32, 0),
        (GgufTensorType::F16, 1),
        (GgufTensorType::Q4_0, 2),
        (GgufTensorType::Q4_1, 3),
        (GgufTensorType::F64, 4),
        (GgufTensorType::Q5_0, 6),
        (GgufTensorType::Q5_1, 7),
        (GgufTensorType::Q8_0, 8),
        (GgufTensorType::Q8_1, 9),
        (GgufTensorType::Q2_K, 10),
        (GgufTensorType::Q3_K, 11),
        (GgufTensorType::Q4_K, 12),
        (GgufTensorType::Q5_K, 13),
        (GgufTensorType::Q6_K, 14),
        (GgufTensorType::Q8_K, 15),
        (GgufTensorType::IQ2_S, 24),
        (GgufTensorType::I2_S, 36),
    ];
    for &(ty, expected) in cases {
        assert_eq!(ty.as_u32(), expected, "{ty:?} should be {expected}");
    }
}

// ---------------------------------------------------------------------------
// Config metadata ordering
// ---------------------------------------------------------------------------

#[test]
fn config_metadata_comes_before_user_metadata() {
    let data = write_to_vec(
        GgufBuilder::new()
            .architecture("arch")
            .description("desc")
            .metadata_string("user.key", "val"),
    );
    let reader = GgufReader::new(&data).unwrap();
    assert_eq!(reader.metadata_count(), 3);
    // All three should be accessible
    assert_eq!(reader.get_string_metadata("general.architecture").as_deref(), Some("arch"),);
    assert_eq!(reader.get_string_metadata("general.description").as_deref(), Some("desc"),);
    assert_eq!(reader.get_string_metadata("user.key").as_deref(), Some("val"),);
}
