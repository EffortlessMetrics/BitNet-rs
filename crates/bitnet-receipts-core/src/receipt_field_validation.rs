//! Field-level helpers for receipt JSON validation.
//!
//! This module owns small, reusable predicates for required fields, scalar
//! checks, CUDA device identity, and dense-receipt guardrails so the main
//! receipt validator can focus on schema-specific proof flow.

use anyhow::{Result, anyhow};
use serde_json::Value;

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
