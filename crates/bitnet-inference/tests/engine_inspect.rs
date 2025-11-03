#[test]
fn engine_inspect_reads_header() {
    use bitnet_inference::engine::inspect_model;
    use tempfile::tempdir;
    let dir = tempdir().unwrap();
    let p = dir.path().join("m.gguf");
    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(b"GGUF");
    buf[4..8].copy_from_slice(2u32.to_le_bytes().as_slice());
    std::fs::write(&p, buf).unwrap();
    let info = inspect_model(&p).unwrap();
    assert_eq!(info.version(), 2);
    assert_eq!(info.n_tensors(), 0);
    assert_eq!(info.n_kv(), 0);
    assert!(info.kv_specs().is_empty());
    assert!(info.quantization_hints().is_empty());
    assert!(info.tensor_summaries().is_empty());
}
#[test]
fn engine_inspect_rejects_invalid() {
    use bitnet_inference::engine::inspect_model;
    use tempfile::tempdir;
    let dir = tempdir().unwrap();
    let p = dir.path().join("bad.gguf");
    let mut buf = [0u8; 24];
    buf[0..4].copy_from_slice(b"NOPE");
    std::fs::write(&p, buf).unwrap();
    let result = inspect_model(&p);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, bitnet_inference::gguf::GgufError::BadMagic(_)));
}
#[test]
fn engine_inspect_handles_short_file() {
    use bitnet_inference::engine::inspect_model;
    use tempfile::tempdir;
    let dir = tempdir().unwrap();
    let p = dir.path().join("short.gguf");
    std::fs::write(&p, b"GGUF").unwrap();
    let result = inspect_model(&p);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, bitnet_inference::gguf::GgufError::ShortHeader(_)));
}
#[test]
fn engine_inspect_rejects_bad_magic() {
    use bitnet_inference::engine::inspect_model;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("bad.gguf");
    std::fs::write(&p, b"NOPENOPENOPENOPE").unwrap();
    let err = inspect_model(&p).unwrap_err();
    let _ = format!("{err}");
}
#[test]
fn engine_inspect_parses_metadata_and_tensors() {
    use bitnet_inference::engine::inspect_model;
    use tempfile::tempdir;
    let dir = tempdir().unwrap();
    let p = dir.path().join("m.gguf");
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(2u32.to_le_bytes().as_slice());
    data.extend_from_slice(1u64.to_le_bytes().as_slice());
    data.extend_from_slice(1u64.to_le_bytes().as_slice());
    let key = b"general.file_type";
    data.extend_from_slice((key.len() as u64).to_le_bytes().as_slice());
    data.extend_from_slice(key);
    data.extend_from_slice(4u32.to_le_bytes().as_slice());
    data.extend_from_slice(1u32.to_le_bytes().as_slice());
    let tname = b"tensor";
    data.extend_from_slice((tname.len() as u64).to_le_bytes().as_slice());
    data.extend_from_slice(tname);
    data.extend_from_slice(1u32.to_le_bytes().as_slice());
    data.extend_from_slice(5u64.to_le_bytes().as_slice());
    data.extend_from_slice(0u32.to_le_bytes().as_slice());
    data.extend_from_slice(0u64.to_le_bytes().as_slice());
    std::fs::write(&p, data).unwrap();
    let info = inspect_model(&p).unwrap();
    assert_eq!(info.kv_specs().len(), 1);
    assert_eq!(info.quantization_hints().len(), 1);
    assert_eq!(info.tensor_summaries().len(), 1);
    assert_eq!(info.tensor_summaries()[0].name, "tensor");
    assert_eq!(info.tensor_summaries()[0].shape, vec![5]);
    assert_eq!(info.tensor_summaries()[0].parameter_count, 5);
    assert_eq!(info.tensor_summaries()[0].dtype, 0);
    assert_eq!(info.tensor_summaries()[0].dtype_name.as_ref().unwrap(), "F32");
    assert_eq!(info.tensor_summaries()[0].category.as_ref().unwrap(), "other");
}
#[test]
fn engine_inspect_comprehensive_metadata_categorization() {
    use bitnet_inference::engine::inspect_model;
    use tempfile::tempdir;
    let dir = tempdir().unwrap();
    let p = dir.path().join("comprehensive.gguf");
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(2u32.to_le_bytes().as_slice());
    data.extend_from_slice(3u64.to_le_bytes().as_slice());
    data.extend_from_slice(6u64.to_le_bytes().as_slice());
    let kvs = [
        ("general.architecture", "llama"),
        ("general.vocab_size", "32000"),
        ("tokenizer.bos_token", "<s>"),
        ("training.dataset", "WikiText"),
        ("general.file_type", "1"),
        ("bitnet.quantization", "I2_S"),
    ];
    for (key, value) in &kvs {
        data.extend_from_slice((key.len() as u64).to_le_bytes().as_slice());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(8u32.to_le_bytes().as_slice());
        data.extend_from_slice((value.len() as u64).to_le_bytes().as_slice());
        data.extend_from_slice(value.as_bytes());
    }
    let tensors = [
        ("token_embd.weight", vec![32000, 4096], 17u32),
        ("layers.0.attention.wq.weight", vec![4096, 4096], 18u32),
        ("layers.0.feed_forward.w1.bias", vec![11008], 0u32),
    ];
    for (name, shape, dtype) in &tensors {
        data.extend_from_slice((name.len() as u64).to_le_bytes().as_slice());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice((shape.len() as u32).to_le_bytes().as_slice());
        for dim in shape {
            data.extend_from_slice((*dim as u64).to_le_bytes().as_slice());
        }
        data.extend_from_slice(dtype.to_le_bytes().as_slice());
        data.extend_from_slice(0u64.to_le_bytes().as_slice());
    }
    std::fs::write(&p, data).unwrap();
    let mut info = inspect_model(&p).unwrap();
    assert_eq!(info.kv_specs().len(), 6);
    assert_eq!(info.tensor_summaries().len(), 3);
    assert!(info.quantization_hints().len() >= 2);
    let embedding = info.tensor_summaries().iter().find(|t| t.name.contains("embd")).unwrap();
    assert_eq!(embedding.category.as_ref().unwrap(), "embedding");
    assert_eq!(embedding.dtype_name.as_ref().unwrap(), "I2_S");
    assert_eq!(embedding.parameter_count, 32000 * 4096);
    let weight = info.tensor_summaries().iter().find(|t| t.name.contains("wq")).unwrap();
    assert_eq!(weight.category.as_ref().unwrap(), "weight");
    assert_eq!(weight.dtype_name.as_ref().unwrap(), "IQ2_S");
    let bias = info.tensor_summaries().iter().find(|t| t.name.contains("bias")).unwrap();
    assert_eq!(bias.category.as_ref().unwrap(), "bias");
    assert_eq!(bias.dtype_name.as_ref().unwrap(), "F32");
    let categorized = info.get_categorized_metadata();
    assert!(categorized.architecture.contains_key("general.architecture"));
    assert!(categorized.model_params.contains_key("general.vocab_size"));
    assert!(categorized.tokenizer.contains_key("tokenizer.bos_token"));
    assert!(categorized.training.contains_key("training.dataset"));
    assert!(categorized.quantization.contains_key("general.file_type"));
    assert!(categorized.quantization.contains_key("bitnet.quantization"));
    let stats = info.get_tensor_statistics();
    assert_eq!(stats.total_parameters, 32000 * 4096 + 4096 * 4096 + 11008);
    assert!(stats.parameters_by_category.contains_key("embedding"));
    assert!(stats.parameters_by_category.contains_key("weight"));
    assert!(stats.parameters_by_category.contains_key("bias"));
    assert_eq!(stats.unique_dtypes.len(), 3);
    assert!(stats.largest_tensor.is_some());
    assert!(stats.estimated_memory_bytes > 0);
}
#[test]
fn engine_inspect_json_serialization() {
    use bitnet_inference::engine::inspect_model;
    use tempfile::tempdir;
    let dir = tempdir().unwrap();
    let p = dir.path().join("json_test.gguf");
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(2u32.to_le_bytes().as_slice());
    data.extend_from_slice(1u64.to_le_bytes().as_slice());
    data.extend_from_slice(1u64.to_le_bytes().as_slice());
    let key = b"general.name";
    data.extend_from_slice((key.len() as u64).to_le_bytes().as_slice());
    data.extend_from_slice(key);
    data.extend_from_slice(8u32.to_le_bytes().as_slice());
    let value = b"test_model";
    data.extend_from_slice((value.len() as u64).to_le_bytes().as_slice());
    data.extend_from_slice(value);
    let tname = b"test.weight";
    data.extend_from_slice((tname.len() as u64).to_le_bytes().as_slice());
    data.extend_from_slice(tname);
    data.extend_from_slice(2u32.to_le_bytes().as_slice());
    data.extend_from_slice(10u64.to_le_bytes().as_slice());
    data.extend_from_slice(20u64.to_le_bytes().as_slice());
    data.extend_from_slice(17u32.to_le_bytes().as_slice());
    data.extend_from_slice(0u64.to_le_bytes().as_slice());
    std::fs::write(&p, data).unwrap();
    let mut info = inspect_model(&p).unwrap();
    let _ = info.get_categorized_metadata();
    let _ = info.get_tensor_statistics();
    let json_pretty = info.to_json().unwrap();
    let json_compact = info.to_json_compact().unwrap();
    assert!(json_pretty.contains("\"version\": 2"));
    assert!(json_pretty.contains("\"n_tensors\": 1"));
    assert!(json_pretty.contains("\"n_kv\": 1"));
    assert!(json_pretty.contains("\"general.name\""));
    assert!(json_pretty.contains("\"test_model\""));
    assert!(json_pretty.contains("\"test.weight\""));
    assert!(json_pretty.contains("\"I2_S\""));
    assert!(json_pretty.contains("\"weight\""));
    assert!(json_pretty.contains("\"parameter_count\": 200"));
    assert!(json_pretty.contains("\"categorized_metadata\""));
    assert!(json_pretty.contains("\"tensor_statistics\""));
    assert!(json_compact.len() < json_pretty.len());
    let parsed: serde_json::Value = serde_json::from_str(&json_pretty).unwrap();
    assert_eq!(parsed["header"]["version"], 2);
    assert_eq!(parsed["header"]["n_tensors"], 1);
}
#[test]
fn engine_inspect_categorization_functions() {
    use bitnet_inference::engine::{
        categorize_kv_key, categorize_tensor_name, format_dtype, format_gguf_value,
    };
    use bitnet_inference::gguf::GgufValue;
    assert_eq!(categorize_kv_key("general.vocab_size"), "model");
    assert_eq!(categorize_kv_key("general.architecture"), "architecture");
    assert_eq!(categorize_kv_key("tokenizer.bos_token"), "tokenizer");
    assert_eq!(categorize_kv_key("training.dataset"), "training");
    assert_eq!(categorize_kv_key("general.file_type"), "quantization");
    assert_eq!(categorize_kv_key("bitnet.quantization"), "quantization");
    assert_eq!(categorize_kv_key("random.key"), "other");
    assert_eq!(categorize_tensor_name("token_embd.weight"), "embedding");
    assert_eq!(categorize_tensor_name("layers.0.attention.wq.weight"), "weight");
    assert_eq!(categorize_tensor_name("layers.0.mlp.w1.bias"), "bias");
    assert_eq!(categorize_tensor_name("layers.0.attention_norm.weight"), "normalization");
    assert_eq!(categorize_tensor_name("layers.0.attention.wq"), "attention");
    assert_eq!(categorize_tensor_name("layers.0.mlp.gate_proj"), "feed_forward");
    assert_eq!(categorize_tensor_name("lm_head.weight"), "output_head");
    assert_eq!(categorize_tensor_name("unknown_tensor"), "other");
    assert_eq!(format_dtype(0), "F32");
    assert_eq!(format_dtype(1), "F16");
    assert_eq!(format_dtype(17), "I2_S");
    assert_eq!(format_dtype(18), "IQ2_S");
    assert_eq!(format_dtype(19), "TL1");
    assert_eq!(format_dtype(20), "TL2");
    assert_eq!(format_dtype(999), "Unknown(999)");
    assert_eq!(format_gguf_value(&GgufValue::U32(42)), "42");
    assert_eq!(format_gguf_value(&GgufValue::F32(std::f32::consts::PI)), "3.141593");
    assert_eq!(format_gguf_value(&GgufValue::F32(3.0)), "3");
    assert_eq!(format_gguf_value(&GgufValue::Bool(true)), "true");
    assert_eq!(format_gguf_value(&GgufValue::String("test".to_string())), "test");
    let arr_short = GgufValue::Array(vec![GgufValue::U32(1), GgufValue::U32(2)]);
    assert_eq!(format_gguf_value(&arr_short), "[1, 2]");
    let arr_long = GgufValue::Array(vec![
        GgufValue::U32(1),
        GgufValue::U32(2),
        GgufValue::U32(3),
        GgufValue::U32(4),
    ]);
    assert_eq!(format_gguf_value(&arr_long), "[1, 2, ... +2 more]");
}
#[test]
fn engine_inspect_memory_estimation() {
    use bitnet_inference::engine::estimate_bytes_per_param;
    assert_eq!(estimate_bytes_per_param(0), 4);
    assert_eq!(estimate_bytes_per_param(1), 2);
    assert_eq!(estimate_bytes_per_param(17), 1);
    assert_eq!(estimate_bytes_per_param(18), 3);
    assert_eq!(estimate_bytes_per_param(19), 1);
    assert_eq!(estimate_bytes_per_param(20), 1);
    assert_eq!(estimate_bytes_per_param(999), 2);
}
#[test]
fn engine_inspect_quantization_hint_detection() {
    use bitnet_inference::engine::inspect_model;
    use tempfile::tempdir;
    let dir = tempdir().unwrap();
    let p = dir.path().join("quant_hints.gguf");
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(2u32.to_le_bytes().as_slice());
    data.extend_from_slice(0u64.to_le_bytes().as_slice());
    data.extend_from_slice(5u64.to_le_bytes().as_slice());
    let quant_keys = [
        "general.file_type",
        "bitnet.i2s_enabled",
        "model.precision",
        "quantization.method",
        "tensor.data_type_bits",
    ];
    for key in &quant_keys {
        data.extend_from_slice((key.len() as u64).to_le_bytes().as_slice());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(8u32.to_le_bytes().as_slice());
        let value = b"test_value";
        data.extend_from_slice((value.len() as u64).to_le_bytes().as_slice());
        data.extend_from_slice(value);
    }
    std::fs::write(&p, data).unwrap();
    let info = inspect_model(&p).unwrap();
    assert_eq!(info.quantization_hints().len(), 5);
    let hint_keys: Vec<&str> = info.quantization_hints().iter().map(|kv| kv.key.as_str()).collect();
    for expected_key in &quant_keys {
        assert!(
            hint_keys.contains(expected_key),
            "Key {} not found in quantization hints",
            expected_key
        );
    }
}
