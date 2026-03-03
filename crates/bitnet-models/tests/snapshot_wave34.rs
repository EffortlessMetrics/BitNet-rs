//! Wave 34 snapshot tests for model metadata, tensor info, capability checks,
//! download manifests, format detection, fingerprints, health checks, loading
//! progress, layer inspection, and configuration types.
//!
//! Covers: ModelMetadata, CapabilityReport, DownloadManifest, ModelFormat,
//! ModelFingerprint, HealthReport, LoadingProgress, LayerInfo, ModelStructure,
//! GgufModelConfig, ProductionLoadConfig, MemoryRequirements, DeviceConfig,
//! ValidationResult, DetectedModel, ConversionCapability, and error variants.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

use bitnet_models::capability_check::{ModelCapability, check_requirements, detect_capabilities};
use bitnet_models::config::{ConfigError, GgufModelConfig, GgufQuantizationConfig};
use bitnet_models::download_manager::{
    DownloadManifest, DownloadProgress, DownloadSpec, llama3_8b_manifest, phi4_manifest,
    validate_download,
};
use bitnet_models::format_detector::{
    DetectedModel, ModelFormat as DetectorFormat, available_conversions, parse_shard_info,
};
use bitnet_models::formats::ModelFormat;
use bitnet_models::formats::gguf::GgufValue;
use bitnet_models::health_check::{CheckResult, HealthReport, HealthStatus};
use bitnet_models::layer_inspector::{LayerInfo, ModelStructure, NamingPattern};
use bitnet_models::loader::LoadConfig;
use bitnet_models::loading_progress::LoadEvent;
use bitnet_models::metadata_extractor::{self, ModelMetadata};
use bitnet_models::model_fingerprint::ModelFingerprint;
use bitnet_models::production_loader::{
    DeviceStrategy, MemoryRequirements, ProductionLoadConfig, ValidationResult,
};

// ============================================================================
// ModelMetadata
// ============================================================================

#[test]
fn w34_metadata_empty_debug() {
    let m = ModelMetadata::new();
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w34_metadata_complete_llama() {
    let m = ModelMetadata {
        model_type: Some("llama".into()),
        architecture: Some("LlamaForCausalLM".into()),
        hidden_size: Some(4096),
        num_layers: Some(32),
        num_heads: Some(32),
        num_kv_heads: Some(32),
        vocab_size: Some(32000),
        max_position: Some(4096),
        intermediate_size: Some(11008),
        activation: Some("silu".into()),
        norm_type: Some("rms_norm".into()),
        rope_base: Some(10000.0),
        tie_word_embeddings: Some(false),
        extra: HashMap::new(),
    };
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w34_metadata_bitnet_gqa() {
    let m = ModelMetadata {
        model_type: Some("bitnet".into()),
        architecture: Some("BitNetForCausalLM".into()),
        hidden_size: Some(2048),
        num_layers: Some(24),
        num_heads: Some(16),
        num_kv_heads: Some(4),
        vocab_size: Some(100000),
        max_position: Some(2048),
        intermediate_size: Some(5504),
        activation: Some("silu".into()),
        norm_type: Some("rms_norm".into()),
        rope_base: Some(500000.0),
        tie_word_embeddings: Some(true),
        extra: HashMap::new(),
    };
    insta::assert_snapshot!(format!(
        "head_dim={:?} gqa_groups={:?} complete={}",
        m.head_dim(),
        m.gqa_groups(),
        m.is_complete(),
    ));
}

#[test]
fn w34_metadata_missing_fields() {
    let m = ModelMetadata { hidden_size: Some(4096), ..Default::default() };
    insta::assert_debug_snapshot!(m.missing_fields());
}

#[test]
fn w34_metadata_from_kv_pairs() {
    let mut pairs = HashMap::new();
    pairs.insert("hidden_size".into(), "5120".into());
    pairs.insert("num_hidden_layers".into(), "40".into());
    pairs.insert("num_attention_heads".into(), "40".into());
    pairs.insert("num_key_value_heads".into(), "10".into());
    pairs.insert("vocab_size".into(), "100352".into());
    pairs.insert("hidden_act".into(), "silu".into());
    let m = metadata_extractor::from_kv_pairs(&pairs);
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w34_metadata_merge() {
    let mut a =
        ModelMetadata { hidden_size: Some(4096), num_layers: Some(32), ..Default::default() };
    let b = ModelMetadata {
        num_heads: Some(32),
        vocab_size: Some(32000),
        activation: Some("gelu".into()),
        ..Default::default()
    };
    a.merge(&b);
    insta::assert_debug_snapshot!(a);
}

// ============================================================================
// Capability checks
// ============================================================================

#[test]
fn w34_capabilities_llama3_debug() {
    let r = detect_capabilities("llama3");
    let mut caps: Vec<&str> = r.capabilities.iter().map(|c| c.name()).collect();
    caps.sort();
    insta::assert_debug_snapshot!(caps);
}

#[test]
fn w34_capabilities_codellama() {
    let r = detect_capabilities("codellama");
    let mut caps: Vec<&str> = r.capabilities.iter().map(|c| c.name()).collect();
    caps.sort();
    insta::assert_debug_snapshot!(caps);
}

#[test]
fn w34_capabilities_starcoder() {
    let r = detect_capabilities("starcoder");
    let mut caps: Vec<&str> = r.capabilities.iter().map(|c| c.name()).collect();
    caps.sort();
    insta::assert_debug_snapshot!(caps);
}

#[test]
fn w34_capabilities_phi4() {
    let r = detect_capabilities("phi4");
    let mut caps: Vec<&str> = r.capabilities.iter().map(|c| c.name()).collect();
    caps.sort();
    insta::assert_debug_snapshot!(caps);
}

#[test]
fn w34_capabilities_unknown_model() {
    let r = detect_capabilities("totally_unknown");
    let mut caps: Vec<&str> = r.capabilities.iter().map(|c| c.name()).collect();
    caps.sort();
    insta::assert_debug_snapshot!(caps);
}

#[test]
fn w34_capabilities_llava_vision() {
    let r = detect_capabilities("llava");
    let mut caps: Vec<&str> = r.capabilities.iter().map(|c| c.name()).collect();
    caps.sort();
    insta::assert_debug_snapshot!(caps);
}

#[test]
fn w34_capability_names_all() {
    let all_caps = [
        ModelCapability::TextGeneration,
        ModelCapability::ChatCompletion,
        ModelCapability::CodeGeneration,
        ModelCapability::FillInMiddle,
        ModelCapability::Embedding,
        ModelCapability::Classification,
        ModelCapability::ToolUse,
        ModelCapability::VisionInput,
        ModelCapability::AudioInput,
    ];
    let names: Vec<&str> = all_caps.iter().map(|c| c.name()).collect();
    insta::assert_debug_snapshot!(names);
}

#[test]
fn w34_check_requirements_ok() {
    let issues = check_requirements(4096, 32, 32000);
    insta::assert_debug_snapshot!(issues);
}

#[test]
fn w34_check_requirements_issues() {
    let issues = check_requirements(33, 0, 50);
    insta::assert_debug_snapshot!(issues);
}

// ============================================================================
// Download manifests
// ============================================================================

#[test]
fn w34_phi4_manifest_debug() {
    let m = phi4_manifest();
    insta::assert_snapshot!(format!(
        "id={} files={} total_expected={} has_checksums={}",
        m.model_id,
        m.file_count(),
        m.total_expected_bytes(),
        m.has_checksums(),
    ));
}

#[test]
fn w34_llama3_manifest_debug() {
    let m = llama3_8b_manifest();
    insta::assert_snapshot!(format!(
        "id={} files={} total_expected={} has_checksums={}",
        m.model_id,
        m.file_count(),
        m.total_expected_bytes(),
        m.has_checksums(),
    ));
}

#[test]
fn w34_download_spec_debug() {
    let spec = DownloadSpec {
        url: "https://example.com/model.safetensors".into(),
        filename: "model.safetensors".into(),
        expected_bytes: Some(4_800_000_000),
        sha256: Some("abc123def456".into()),
    };
    insta::assert_debug_snapshot!(spec);
}

#[test]
fn w34_download_progress_percent() {
    let p = DownloadProgress {
        file_index: 2,
        total_files: 6,
        bytes_downloaded: 7_200_000_000,
        bytes_total: 29_000_000_000,
        current_file: "model-00003-of-00006.safetensors".into(),
    };
    insta::assert_snapshot!(format!(
        "percent={:.1} file_percent={:.1} file={}/{}",
        p.percent(),
        p.file_percent(),
        p.file_index,
        p.total_files,
    ));
}

#[test]
fn w34_validate_download_missing() {
    let m = DownloadManifest {
        model_id: "test/model".into(),
        files: vec![
            DownloadSpec {
                url: "http://x/a.bin".into(),
                filename: "a.bin".into(),
                expected_bytes: Some(1000),
                sha256: None,
            },
            DownloadSpec {
                url: "http://x/b.bin".into(),
                filename: "b.bin".into(),
                expected_bytes: Some(2000),
                sha256: None,
            },
        ],
        total_bytes: Some(3000),
    };
    let sizes = HashMap::from([("a.bin".to_string(), 1000u64)]);
    let issues = validate_download(&m, &sizes);
    insta::assert_debug_snapshot!(issues);
}

// ============================================================================
// ModelFormat (formats module)
// ============================================================================

#[test]
fn w34_model_format_names() {
    let formats = [ModelFormat::SafeTensors, ModelFormat::Gguf];
    let info: Vec<String> =
        formats.iter().map(|f| format!("name={} ext={}", f.name(), f.extension())).collect();
    insta::assert_debug_snapshot!(info);
}

// ============================================================================
// Format detector
// ============================================================================

#[test]
fn w34_detector_format_variants() {
    let formats = [
        DetectorFormat::Gguf,
        DetectorFormat::SafeTensors,
        DetectorFormat::SafeTensorsIndex,
        DetectorFormat::PyTorchBin,
        DetectorFormat::OnnxModel,
        DetectorFormat::Unknown,
    ];
    let info: Vec<String> = formats
        .iter()
        .map(|f| {
            format!(
                "{} supported={} needs_conversion={}",
                f.display_name(),
                f.is_supported(),
                f.needs_conversion(),
            )
        })
        .collect();
    insta::assert_debug_snapshot!(info);
}

#[test]
fn w34_detected_model_single() {
    let m = DetectedModel::new(PathBuf::from("model.gguf"), DetectorFormat::Gguf, 2_147_483_648);
    insta::assert_snapshot!(format!(
        "format={} size_mb={:.1} size_gb={:.2} sharded={}",
        m.format.display_name(),
        m.size_mb(),
        m.size_gb(),
        m.is_sharded(),
    ));
}

#[test]
fn w34_detected_model_sharded() {
    let m = DetectedModel::new(
        PathBuf::from("model-00002-of-00006.safetensors"),
        DetectorFormat::SafeTensors,
        5_000_000_000,
    )
    .with_shard_info(2, 6);
    insta::assert_debug_snapshot!(m);
}

#[test]
fn w34_parse_shard_info_patterns() {
    let cases = [
        ("model-00001-of-00006.safetensors", parse_shard_info("model-00001-of-00006.safetensors")),
        (
            "weights-00003-of-00012.safetensors",
            parse_shard_info("weights-00003-of-00012.safetensors"),
        ),
        ("model.safetensors", parse_shard_info("model.safetensors")),
    ];
    let results: Vec<String> =
        cases.iter().map(|(name, info)| format!("{name} => {info:?}")).collect();
    insta::assert_debug_snapshot!(results);
}

#[test]
fn w34_available_conversions_debug() {
    let convs = available_conversions();
    let info: Vec<String> = convs
        .iter()
        .map(|c| {
            format!(
                "{}->{}: available={} desc={}",
                c.from.display_name(),
                c.to.display_name(),
                c.available,
                c.description,
            )
        })
        .collect();
    insta::assert_debug_snapshot!(info);
}

// ============================================================================
// ModelFingerprint
// ============================================================================

#[test]
fn w34_fingerprint_llama7b() {
    let fp = ModelFingerprint::new("llama")
        .with_param_count(6_738_415_616)
        .with_layers(32)
        .with_hidden_size(4096)
        .with_heads(32)
        .with_vocab_size(32000)
        .with_quant_type("f16");
    insta::assert_snapshot!(format!(
        "display={}\ncompact_id={}\nsize_label={}\nis_quantized={}\nest_bytes={}",
        fp,
        fp.compact_id(),
        fp.size_label(),
        fp.is_quantized(),
        fp.estimated_weight_bytes(),
    ));
}

#[test]
fn w34_fingerprint_bitnet2b_i2s() {
    let fp = ModelFingerprint::new("bitnet")
        .with_param_count(2_000_000_000)
        .with_layers(24)
        .with_hidden_size(2048)
        .with_heads(16)
        .with_vocab_size(100000)
        .with_quant_type("i2s")
        .with_tag("family", "microsoft-bitnet");
    insta::assert_debug_snapshot!(fp);
}

#[test]
fn w34_fingerprint_same_arch_check() {
    let a = ModelFingerprint::new("llama")
        .with_layers(32)
        .with_hidden_size(4096)
        .with_heads(32)
        .with_vocab_size(32000)
        .with_quant_type("f16");
    let b = ModelFingerprint::new("llama")
        .with_layers(32)
        .with_hidden_size(4096)
        .with_heads(32)
        .with_vocab_size(32000)
        .with_quant_type("q4_0");
    insta::assert_snapshot!(format!(
        "same_arch={} same_model_diff_quant={}",
        a.same_architecture(&b),
        a.same_model_different_quant(&b),
    ));
}

// ============================================================================
// HealthCheck
// ============================================================================

#[test]
fn w34_health_status_variants() {
    let statuses = [
        HealthStatus::Healthy,
        HealthStatus::Warning("minor alignment issue".into()),
        HealthStatus::Error("tensor shape mismatch".into()),
    ];
    let info: Vec<String> = statuses
        .iter()
        .map(|s| format!("{s:?} is_healthy={} is_error={}", s.is_healthy(), s.is_error()))
        .collect();
    insta::assert_debug_snapshot!(info);
}

#[test]
fn w34_health_report_all_healthy() {
    let report = HealthReport {
        checks: vec![
            CheckResult {
                name: "tensor_shapes".into(),
                status: HealthStatus::Healthy,
                duration_us: 150,
            },
            CheckResult {
                name: "weight_values".into(),
                status: HealthStatus::Healthy,
                duration_us: 200,
            },
            CheckResult {
                name: "vocab_size".into(),
                status: HealthStatus::Healthy,
                duration_us: 10,
            },
        ],
        model_path: "models/test.gguf".into(),
        total_duration_us: 360,
    };
    insta::assert_snapshot!(format!(
        "healthy={} errors={} warnings={} checks={}",
        report.is_healthy(),
        report.error_count(),
        report.warning_count(),
        report.checks.len(),
    ));
}

#[test]
fn w34_health_report_with_issues() {
    let report = HealthReport {
        checks: vec![
            CheckResult {
                name: "tensor_shapes".into(),
                status: HealthStatus::Healthy,
                duration_us: 150,
            },
            CheckResult {
                name: "weight_values".into(),
                status: HealthStatus::Warning("unusual range".into()),
                duration_us: 200,
            },
            CheckResult {
                name: "file_size".into(),
                status: HealthStatus::Error("too small for claimed param count".into()),
                duration_us: 5,
            },
        ],
        model_path: "models/bad.gguf".into(),
        total_duration_us: 355,
    };
    insta::assert_debug_snapshot!(report);
}

// ============================================================================
// LoadingProgress / LoadEvent / LoadSummary
// ============================================================================

#[test]
fn w34_load_event_variants() {
    let events: Vec<LoadEvent> = vec![
        LoadEvent::ShardStart { index: 0, total: 4, name: "shard-00001.safetensors".into() },
        LoadEvent::ShardDone { index: 0, total: 4, bytes: 4_900_000_000 },
        LoadEvent::TensorStart { name: "model.embed_tokens.weight".into(), elements: 131_072_000 },
        LoadEvent::TensorDone { name: "model.embed_tokens.weight".into(), elements: 131_072_000 },
        LoadEvent::Conversion { from_dtype: "bf16".into(), to_dtype: "f32".into(), elements: 4096 },
        LoadEvent::Complete { total_tensors: 291, total_bytes: 13_600_000_000, elapsed_ms: 5200 },
        LoadEvent::Error { message: "checksum mismatch".into() },
    ];
    let debug: Vec<String> = events.iter().map(|e| format!("{e:?}")).collect();
    insta::assert_debug_snapshot!(debug);
}

// ============================================================================
// LayerInspector
// ============================================================================

#[test]
fn w34_layer_info_debug() {
    let li = LayerInfo {
        index: 0,
        components: BTreeSet::from([
            "attn_q".into(),
            "attn_k".into(),
            "attn_v".into(),
            "attn_output".into(),
            "ffn_gate".into(),
            "ffn_up".into(),
            "ffn_down".into(),
        ]),
        tensor_count: 7,
    };
    insta::assert_debug_snapshot!(li);
}

#[test]
fn w34_model_structure_gguf() {
    let structure = ModelStructure {
        num_layers: 2,
        layers: vec![
            LayerInfo {
                index: 0,
                components: BTreeSet::from(["attn_q".into(), "ffn_gate".into()]),
                tensor_count: 2,
            },
            LayerInfo {
                index: 1,
                components: BTreeSet::from(["attn_q".into(), "ffn_gate".into()]),
                tensor_count: 2,
            },
        ],
        non_layer_tensors: vec!["token_embd.weight".into(), "output_norm.weight".into()],
        tensor_name_pattern: NamingPattern::Gguf,
    };
    insta::assert_debug_snapshot!(structure);
}

#[test]
fn w34_naming_pattern_display() {
    let patterns = [NamingPattern::HuggingFace, NamingPattern::Gguf, NamingPattern::Unknown];
    let displays: Vec<String> = patterns.iter().map(|p| format!("{p}")).collect();
    insta::assert_debug_snapshot!(displays);
}

// ============================================================================
// GgufModelConfig from metadata
// ============================================================================

fn llama_metadata() -> HashMap<String, GgufValue> {
    HashMap::from([
        ("general.architecture".into(), GgufValue::String("llama".into())),
        ("general.name".into(), GgufValue::String("TestLlama-7B".into())),
        ("llama.vocab_size".into(), GgufValue::U32(32000)),
        ("llama.embedding_length".into(), GgufValue::U32(4096)),
        ("llama.block_count".into(), GgufValue::U32(32)),
        ("llama.attention.head_count".into(), GgufValue::U32(32)),
        ("llama.attention.head_count_kv".into(), GgufValue::U32(32)),
        ("llama.feed_forward_length".into(), GgufValue::U32(11008)),
        ("llama.context_length".into(), GgufValue::U32(4096)),
        ("llama.rope.freq_base".into(), GgufValue::F32(10000.0)),
    ])
}

#[test]
fn w34_gguf_config_llama_debug() {
    let cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn w34_gguf_config_memory_estimate() {
    let cfg = GgufModelConfig::from_gguf_metadata(&llama_metadata()).unwrap();
    let est = cfg.memory_estimate();
    insta::assert_snapshot!(format!(
        "weight_bytes={} kv_cache_bytes={} total_bytes={}",
        est.weight_bytes, est.kv_cache_bytes, est.total_bytes,
    ));
}

#[test]
fn w34_gguf_quant_config_default() {
    let qcfg = GgufQuantizationConfig::default();
    insta::assert_debug_snapshot!(qcfg);
}

// ============================================================================
// ProductionLoadConfig / MemoryRequirements / DeviceConfig / ValidationResult
// ============================================================================

#[test]
fn w34_production_load_config_default() {
    let cfg = ProductionLoadConfig::default();
    insta::assert_snapshot!(format!(
        "strict={} align={} profile={} device={:?} max_size_gb={}",
        cfg.strict_validation,
        cfg.validate_tensor_alignment,
        cfg.profile_memory,
        cfg.target_device,
        cfg.max_model_size_bytes.unwrap_or(0) / (1024 * 1024 * 1024),
    ));
}

#[test]
fn w34_memory_requirements_debug() {
    let req = MemoryRequirements {
        total_mb: 8192,
        gpu_memory_mb: Some(6144),
        cpu_memory_mb: 7168,
        kv_cache_mb: 512,
        activation_mb: 256,
        headroom_mb: 256,
    };
    insta::assert_debug_snapshot!(req);
}

#[test]
fn w34_device_strategy_variants() {
    let strategies = [
        DeviceStrategy::CpuOnly,
        DeviceStrategy::GpuOnly,
        DeviceStrategy::Hybrid { cpu_layers: 8, gpu_layers: 24 },
    ];
    let debug: Vec<String> = strategies.iter().map(|s| format!("{s:?}")).collect();
    insta::assert_debug_snapshot!(debug);
}

#[test]
fn w34_validation_result_pass() {
    let vr = ValidationResult {
        passed: true,
        warnings: vec![],
        errors: vec![],
        alignment_issues: vec![],
        recommendations: vec!["Consider using AVX2 for better performance".into()],
    };
    insta::assert_debug_snapshot!(vr);
}

#[test]
fn w34_validation_result_fail() {
    let vr = ValidationResult {
        passed: false,
        warnings: vec!["LayerNorm gamma RMS out of expected range".into()],
        errors: vec!["Tensor blk.0.attn_q.weight: shape mismatch".into()],
        alignment_issues: vec!["blk.5.ffn_gate.weight: offset not 32-byte aligned".into()],
        recommendations: vec!["Re-export model with F16 LayerNorm".into()],
    };
    insta::assert_debug_snapshot!(vr);
}

// ============================================================================
// LoadConfig
// ============================================================================

#[test]
fn w34_load_config_default() {
    let cfg = LoadConfig::default();
    insta::assert_snapshot!(format!(
        "use_mmap={} validate_checksums={}",
        cfg.use_mmap, cfg.validate_checksums,
    ));
}

// ============================================================================
// ConfigError variants
// ============================================================================

#[test]
fn w34_config_error_missing_key() {
    let e = ConfigError::MissingKey("llama.vocab_size".into());
    insta::assert_snapshot!(format!("{e}"));
}

#[test]
fn w34_config_error_invalid_value() {
    let e = ConfigError::InvalidValue {
        key: "llama.block_count".into(),
        reason: "expected u32, got string".into(),
    };
    insta::assert_snapshot!(format!("{e}"));
}

#[test]
fn w34_config_error_validation() {
    let e = ConfigError::Validation("hidden_size must be divisible by num_heads".into());
    insta::assert_snapshot!(format!("{e}"));
}
