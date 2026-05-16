use anyhow::{Result, anyhow};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

pub(crate) fn load_json_receipt(path: &Path) -> Result<Value> {
    let content = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

pub(crate) fn validate_cuda_receipt_common<'a>(
    receipt: &'a Value,
    artifact_kind: &str,
    claim: &str,
) -> Result<&'a Value> {
    require_u64_eq(receipt, "schema", 1)?;
    require_string_eq(receipt, "artifact_kind", artifact_kind)?;
    require_string_eq(receipt, "machine_id", "windows-9950x3d-rtx5070ti")?;
    require_string_eq(receipt, "hardware_lane", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "requested_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "selected_backend", "nvidia-rtx-5070-ti-cuda")?;
    require_string_eq(receipt, "runtime_api", "cuda")?;
    require_string_eq(receipt, "claim", claim)?;
    require_bool_eq(receipt, "fallback_used", false)?;
    require_null(receipt, "fallback_backend")?;
    require_null(receipt, "fallback_reason")?;
    require_null(receipt, "error")?;

    let cuda = object_field(receipt, "cuda")?;
    require_bool_eq(cuda, "available", true)?;
    require_positive_u64(cuda, "device_count")?;
    require_cuda_device_index(cuda)?;
    require_rtx_5070_ti_name(cuda, "device_name")?;
    require_string_eq(cuda, "compute_capability", "12.0")?;
    require_string_non_empty_not_tbd(cuda, "driver_version")?;
    require_string_non_empty_not_tbd(cuda, "cuda_runtime_version")?;
    require_string_non_empty_not_tbd(cuda, "cuda_toolkit_version")?;
    require_string_non_empty_not_tbd(cuda, "nvrtc_version")?;
    require_positive_u64(cuda, "vram_bytes")?;

    let stats = first_kernel_stats(receipt)?;
    require_string_non_empty(stats, "kernel_id")?;
    require_positive_u64(stats, "invocations")?;
    require_u64_eq(stats, "fallback_invocations", 0)?;
    require_positive_u64(stats, "host_to_device_bytes")?;
    require_positive_u64(stats, "device_to_host_bytes")?;
    require_positive_u64(stats, "kernel_launches")?;
    require_optional_non_negative_number(stats, "kernel_time_ms")?;

    Ok(stats)
}

pub(crate) fn first_kernel_stats(receipt: &Value) -> Result<&Value> {
    let stats = object_field(receipt, "kernel_stats")?;
    let stats = stats.as_array().ok_or_else(|| anyhow!("kernel_stats must be an array"))?;
    stats.first().ok_or_else(|| anyhow!("kernel_stats must contain at least one entry"))
}

pub(crate) fn object_field<'a>(object: &'a Value, field: &str) -> Result<&'a Value> {
    object.get(field).ok_or_else(|| anyhow!("missing required field `{field}`"))
}

pub(crate) fn array_field<'a>(object: &'a Value, field: &str) -> Result<&'a Vec<Value>> {
    object_field(object, field)?
        .as_array()
        .ok_or_else(|| anyhow!("field `{field}` must be an array"))
}

pub(crate) fn required_string<'a>(object: &'a Value, field: &str) -> Result<&'a str> {
    object_field(object, field)?.as_str().ok_or_else(|| anyhow!("field `{field}` must be a string"))
}

pub(crate) fn required_u64(object: &Value, field: &str) -> Result<u64> {
    object_field(object, field)?
        .as_u64()
        .ok_or_else(|| anyhow!("field `{field}` must be an unsigned integer"))
}

pub(crate) fn require_string_eq(object: &Value, field: &str, expected: &str) -> Result<()> {
    let actual = required_string(object, field)?;
    if actual != expected {
        return Err(anyhow!("field `{field}` must be `{expected}`, got `{actual}`"));
    }
    Ok(())
}

pub(crate) fn require_string_non_empty(object: &Value, field: &str) -> Result<()> {
    let value = required_string(object, field)?;
    if value.trim().is_empty() {
        return Err(anyhow!("field `{field}` must not be empty"));
    }
    Ok(())
}

pub(crate) fn require_string_non_empty_not_tbd(object: &Value, field: &str) -> Result<()> {
    let value = required_string(object, field)?;
    if value.trim().is_empty() || value == "TBD" {
        return Err(anyhow!("field `{field}` must record a concrete value"));
    }
    Ok(())
}

pub(crate) const DENSE_ALL_LAYER_OPERATION_SEQUENCE: [(&str, &str); 14] = [
    ("attention_norm", "rmsnorm"),
    ("attention_q", "matmul"),
    ("attention_k", "matmul"),
    ("attention_v", "matmul"),
    ("rope", "rope"),
    ("attention_scores", "attention"),
    ("attention_softmax", "softmax"),
    ("attention_v_mix", "attention"),
    ("attention_output", "matmul"),
    ("ffn_norm", "rmsnorm"),
    ("mlp_gate", "matmul"),
    ("mlp_up", "matmul"),
    ("mlp_activation", "activation"),
    ("mlp_down", "matmul"),
];

pub(crate) fn dense_all_layer_operation_signature_sha256(operations: &[Value]) -> Result<String> {
    let signature = operations
        .iter()
        .map(dense_all_layer_operation_signature_entry)
        .collect::<Result<Vec<_>>>()?;
    let bytes = serde_json::to_vec(&Value::Array(signature))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

pub(crate) fn dense_all_layer_operation_signature_entry(op: &Value) -> Result<Value> {
    let mut entry = Map::new();
    entry.insert("role".to_string(), Value::String(required_string(op, "role")?.to_string()));
    entry.insert("op_type".to_string(), Value::String(required_string(op, "op_type")?.to_string()));
    entry.insert("source".to_string(), Value::String(required_string(op, "source")?.to_string()));
    entry.insert(
        "source_tensor_type".to_string(),
        op.get("source_tensor_type").cloned().unwrap_or(Value::Null),
    );
    entry
        .insert("source_shape".to_string(), op.get("source_shape").cloned().unwrap_or(Value::Null));
    entry.insert(
        "is_quantized".to_string(),
        op.get("is_quantized").cloned().unwrap_or(Value::Bool(false)),
    );
    entry.insert("route".to_string(), Value::String(required_string(op, "route")?.to_string()));
    entry.insert("status".to_string(), Value::String(required_string(op, "status")?.to_string()));
    Ok(Value::Object(entry))
}

pub(crate) fn require_sha256(object: &Value, field: &str) -> Result<()> {
    let value = required_string(object, field)?;
    if value.len() != 64 || !value.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(anyhow!("field `{field}` must be a 64-character sha256 hex digest"));
    }
    Ok(())
}

pub(crate) fn require_extractable_dense_linear_role(role: &str) -> Result<()> {
    const EXTRACTABLE_ROLES: &[&str] = &[
        "output",
        "attention_q",
        "attention_k",
        "attention_v",
        "attention_output",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    ];
    if !EXTRACTABLE_ROLES.contains(&role) {
        return Err(anyhow!(
            "linear_fixture.role must be an extractable dense linear role, got `{role}`"
        ));
    }
    Ok(())
}

pub(crate) fn require_extractable_dense_norm_role(role: &str) -> Result<()> {
    const EXTRACTABLE_ROLES: &[&str] = &["attention_norm", "ffn_norm"];
    if !EXTRACTABLE_ROLES.contains(&role) {
        return Err(anyhow!(
            "norm_fixtures.role must be an extractable dense norm role, got `{role}`"
        ));
    }
    Ok(())
}

pub(crate) fn require_rtx_5070_ti_name(object: &Value, field: &str) -> Result<()> {
    let value = required_string(object, field)?;
    let compact = value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    if !(compact.contains("nvidia") && compact.contains("rtx5070ti")) {
        return Err(anyhow!("field `{field}` must identify NVIDIA GeForce RTX 5070 Ti"));
    }
    Ok(())
}

pub(crate) fn require_bool_eq(object: &Value, field: &str, expected: bool) -> Result<()> {
    let actual = object_field(object, field)?
        .as_bool()
        .ok_or_else(|| anyhow!("field `{field}` must be a bool"))?;
    if actual != expected {
        return Err(anyhow!("field `{field}` must be `{expected}`, got `{actual}`"));
    }
    Ok(())
}

pub(crate) fn require_null(object: &Value, field: &str) -> Result<()> {
    if !object_field(object, field)?.is_null() {
        return Err(anyhow!("field `{field}` must be null"));
    }
    Ok(())
}

pub(crate) fn require_u64_eq(object: &Value, field: &str, expected: u64) -> Result<()> {
    let actual = object_field(object, field)?
        .as_u64()
        .ok_or_else(|| anyhow!("field `{field}` must be an unsigned integer"))?;
    if actual != expected {
        return Err(anyhow!("field `{field}` must be `{expected}`, got `{actual}`"));
    }
    Ok(())
}

pub(crate) fn require_positive_u64(object: &Value, field: &str) -> Result<()> {
    let actual = object_field(object, field)?
        .as_u64()
        .ok_or_else(|| anyhow!("field `{field}` must be an unsigned integer"))?;
    if actual == 0 {
        return Err(anyhow!("field `{field}` must be greater than zero"));
    }
    Ok(())
}

pub(crate) fn require_optional_positive_u64(object: &Value, field: &str) -> Result<()> {
    let value = object_field(object, field)?;
    if value.is_null() {
        return Ok(());
    }
    let actual = value
        .as_u64()
        .ok_or_else(|| anyhow!("field `{field}` must be null or an unsigned integer"))?;
    if actual == 0 {
        return Err(anyhow!("field `{field}` must be greater than zero when measured"));
    }
    Ok(())
}

pub(crate) fn reject_bitnet_packed_marker(value: &str, field: &str) -> Result<()> {
    let normalized = value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    const BITNET_PACKED_MARKERS: &[&str] = &["bitnet", "i2s", "iq2s", "qk256", "w158a8"];
    if BITNET_PACKED_MARKERS.iter().any(|marker| normalized.contains(marker)) {
        return Err(anyhow!(
            "field `{field}` must not identify BitNet packed I2_S/QK256 proof, got `{value}`"
        ));
    }
    Ok(())
}

pub(crate) fn require_cuda_device_index(cuda: &Value) -> Result<()> {
    if object_field(cuda, "device_index")
        .and_then(|value| {
            value
                .as_u64()
                .ok_or_else(|| anyhow!("field `device_index` must be an unsigned integer"))
        })
        .is_ok()
        || object_field(cuda, "selected_device_index")
            .and_then(|value| {
                value.as_u64().ok_or_else(|| {
                    anyhow!("field `selected_device_index` must be an unsigned integer")
                })
            })
            .is_ok()
    {
        return Ok(());
    }

    Err(anyhow!("cuda receipt must record `device_index` or `selected_device_index`"))
}

pub(crate) fn require_non_negative_number(object: &Value, field: &str) -> Result<()> {
    let actual = object_field(object, field)?
        .as_f64()
        .ok_or_else(|| anyhow!("field `{field}` must be a number"))?;
    if actual < 0.0 {
        return Err(anyhow!("field `{field}` must be non-negative"));
    }
    Ok(())
}

pub(crate) fn require_positive_number(object: &Value, field: &str) -> Result<()> {
    let actual = object_field(object, field)?
        .as_f64()
        .ok_or_else(|| anyhow!("field `{field}` must be a number"))?;
    if actual <= 0.0 {
        return Err(anyhow!("field `{field}` must be positive"));
    }
    Ok(())
}

pub(crate) fn require_number(object: &Value, field: &str) -> Result<()> {
    object_field(object, field)?
        .as_f64()
        .ok_or_else(|| anyhow!("field `{field}` must be a number"))?;
    Ok(())
}

pub(crate) fn validate_dense_boundary_tensor_fixture(
    fixture: &Value,
    expected_role: &str,
) -> Result<()> {
    require_string_non_empty(fixture, "name")?;
    reject_bitnet_packed_marker(
        required_string(fixture, "name")?,
        "model_boundary_fixtures.fixture.name",
    )?;
    require_string_eq(fixture, "role", expected_role)?;
    require_string_non_empty(fixture, "tensor_name")?;
    reject_bitnet_packed_marker(
        required_string(fixture, "tensor_name")?,
        "model_boundary_fixtures.fixture.tensor_name",
    )?;
    require_string_non_empty(fixture, "tensor_type")?;
    reject_bitnet_packed_marker(
        required_string(fixture, "tensor_type")?,
        "model_boundary_fixtures.fixture.tensor_type",
    )?;
    if array_field(fixture, "source_shape")?.is_empty() {
        return Err(anyhow!("model_boundary_fixtures.fixture.source_shape must not be empty"));
    }
    required_u64(fixture, "source_offset")?;
    require_positive_u64(fixture, "source_size_bytes")?;
    require_positive_u64(fixture, "value_count")?;
    require_positive_u64(fixture, "output_len")?;
    require_sha256(fixture, "output_sha256")?;
    require_non_negative_number(fixture, "max_abs")?;
    require_bool_eq(fixture, "dense_gguf_inference_claimed", false)?;
    require_bool_eq(fixture, "bitnet_packed_i2s_qk256_proof", false)?;
    Ok(())
}

pub(crate) fn require_optional_non_negative_number(object: &Value, field: &str) -> Result<()> {
    let value = object_field(object, field)?;
    if value.is_null() {
        return Ok(());
    }
    let actual =
        value.as_f64().ok_or_else(|| anyhow!("field `{field}` must be null or a number"))?;
    if actual < 0.0 {
        return Err(anyhow!("field `{field}` must be non-negative"));
    }
    Ok(())
}
