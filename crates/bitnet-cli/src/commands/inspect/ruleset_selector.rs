//! Select the validation ruleset for an inspected model.
//!
//! Single responsibility: given a gate mode (`none` / `auto` / `policy`) and
//! the GGUF metadata exposed by the reader, produce the `Ruleset` that the
//! tensor scanner will evaluate against. Architecture and `file_type` are
//! read here because they drive the auto-detection path and are also logged
//! once for observability.

use anyhow::Result;
use bitnet_models::formats::gguf::GgufReader;
use std::path::Path;
use tracing::debug;

use crate::ln_rules::{Ruleset, detect_rules, load_policy};

/// Resolve the ruleset for the requested gate mode.
///
/// - `none` returns a generic ruleset.
/// - `auto` detects a ruleset from the model architecture and file type.
/// - `policy` loads a ruleset from a YAML file; the lookup key defaults to
///   the model architecture if `policy_key` is `None`.
pub(crate) fn select_ruleset(
    reader: &GgufReader,
    gate: &str,
    policy: Option<&Path>,
    policy_key: Option<&str>,
) -> Result<Ruleset> {
    let arch = reader.get_string_metadata("general.architecture").unwrap_or_else(|| {
        debug!("'general.architecture' metadata not found, using 'unknown'");
        "unknown".to_string()
    });
    debug!("Architecture: {}", arch);

    let file_type = reader.get_u32_metadata("general.file_type").unwrap_or(0);
    debug!("File type: {}", file_type);

    let rules: Ruleset = match gate {
        "none" => crate::ln_rules::rules_generic(),
        "auto" => detect_rules(&arch, file_type),
        "policy" => {
            let pol = policy.ok_or_else(|| anyhow::anyhow!("--policy required for gate=policy"))?;
            let key = policy_key.unwrap_or(&arch);
            load_policy(pol, key)?
        }
        other => {
            return Err(anyhow::anyhow!(
                "Invalid gate mode '{}'. Must be one of: none, auto, policy.",
                other
            ));
        }
    };

    tracing::info!(
        "LN gate ruleset: {} (architecture: {}, file_type: {})",
        rules.name,
        arch,
        file_type
    );

    Ok(rules)
}
