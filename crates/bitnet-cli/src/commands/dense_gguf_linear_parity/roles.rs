//! Dense GGUF tensor role parsing and labeling.
//!
//! CLI-facing role aliases and receipt labels are kept together so command
//! execution can work with `DenseGgufTensorRole` values without owning parsing
//! policy.

use anyhow::{Result, anyhow, bail};
use bitnet_models::dense_gguf_descriptors::DenseGgufTensorRole;
use std::collections::BTreeSet;

const DEFAULT_ROLE_SWEEP: &[DenseGgufTensorRole] = &[
    DenseGgufTensorRole::AttentionQ,
    DenseGgufTensorRole::AttentionK,
    DenseGgufTensorRole::AttentionV,
    DenseGgufTensorRole::AttentionOutput,
    DenseGgufTensorRole::MlpGate,
    DenseGgufTensorRole::MlpUp,
    DenseGgufTensorRole::MlpDown,
    DenseGgufTensorRole::Output,
];

pub(super) fn parse_dense_linear_role(value: &str) -> Result<DenseGgufTensorRole> {
    let normalized = normalized_role_key(value);
    match normalized.as_str() {
        "output" => Ok(DenseGgufTensorRole::Output),
        "attentionq" | "attnq" | "q" => Ok(DenseGgufTensorRole::AttentionQ),
        "attentionk" | "attnk" | "k" => Ok(DenseGgufTensorRole::AttentionK),
        "attentionv" | "attnv" | "v" => Ok(DenseGgufTensorRole::AttentionV),
        "attentionoutput" | "attnoutput" | "o" => Ok(DenseGgufTensorRole::AttentionOutput),
        "mlpgate" | "gate" => Ok(DenseGgufTensorRole::MlpGate),
        "mlpup" | "up" => Ok(DenseGgufTensorRole::MlpUp),
        "mlpdown" | "down" => Ok(DenseGgufTensorRole::MlpDown),
        _ => Err(anyhow!(
            "unsupported dense linear role `{value}`; expected output, attention_q, attention_k, attention_v, attention_output, mlp_gate, mlp_up, or mlp_down"
        )),
    }
}

pub(super) fn parse_role_sweep(values: &[String]) -> Result<Vec<DenseGgufTensorRole>> {
    let roles = if values.is_empty() {
        DEFAULT_ROLE_SWEEP.to_vec()
    } else {
        values.iter().map(|value| parse_dense_linear_role(value)).collect::<Result<Vec<_>>>()?
    };

    if roles.len() < 2 {
        bail!("dense GGUF linear role sweep requires at least two roles");
    }

    ensure_unique_roles(&roles, "dense GGUF linear role sweep role")?;
    Ok(roles)
}

pub(super) fn parse_norm_roles(values: &[String]) -> Result<Vec<DenseGgufTensorRole>> {
    let roles = if values.is_empty() {
        vec![DenseGgufTensorRole::AttentionNorm, DenseGgufTensorRole::FfnNorm]
    } else {
        values.iter().map(|value| parse_dense_norm_role(value)).collect::<Result<Vec<_>>>()?
    };

    if roles.len() < 2 {
        bail!("dense GGUF norm fixture extraction requires attention_norm and ffn_norm roles");
    }

    ensure_unique_roles(&roles, "dense GGUF norm fixture role")?;
    for required in [DenseGgufTensorRole::AttentionNorm, DenseGgufTensorRole::FfnNorm] {
        if !roles.contains(&required) {
            bail!(
                "dense GGUF norm fixture extraction requires role `{}`",
                dense_role_label(required)
            );
        }
    }

    Ok(roles)
}

fn parse_dense_norm_role(value: &str) -> Result<DenseGgufTensorRole> {
    let normalized = normalized_role_key(value);
    match normalized.as_str() {
        "attentionnorm" | "attnnorm" | "inputlayernorm" => Ok(DenseGgufTensorRole::AttentionNorm),
        "ffnnorm" | "postattentionlayernorm" | "postattnnorm" => Ok(DenseGgufTensorRole::FfnNorm),
        _ => Err(anyhow!(
            "unsupported dense norm role `{value}`; expected attention_norm or ffn_norm"
        )),
    }
}

pub(super) fn dense_role_label(role: DenseGgufTensorRole) -> &'static str {
    match role {
        DenseGgufTensorRole::Output => "output",
        DenseGgufTensorRole::AttentionQ => "attention_q",
        DenseGgufTensorRole::AttentionK => "attention_k",
        DenseGgufTensorRole::AttentionV => "attention_v",
        DenseGgufTensorRole::AttentionOutput => "attention_output",
        DenseGgufTensorRole::MlpGate => "mlp_gate",
        DenseGgufTensorRole::MlpUp => "mlp_up",
        DenseGgufTensorRole::MlpDown => "mlp_down",
        DenseGgufTensorRole::TokenEmbedding => "token_embedding",
        DenseGgufTensorRole::AttentionNorm => "attention_norm",
        DenseGgufTensorRole::FfnNorm => "ffn_norm",
        DenseGgufTensorRole::Other => "other",
    }
}

fn normalized_role_key(value: &str) -> String {
    value.chars().filter(|ch| ch.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase()
}

fn ensure_unique_roles(roles: &[DenseGgufTensorRole], context: &str) -> Result<()> {
    let mut seen = BTreeSet::new();
    for role in roles {
        let label = dense_role_label(*role);
        if !seen.insert(label) {
            bail!("{context} `{label}` was requested more than once");
        }
    }
    Ok(())
}
