//! CLI identity constants and small hashing helpers.
//!
//! These values are shared by command metadata and receipt generation. Keeping
//! them in one SRP module avoids scattering hardware-lane labels and receipt
//! identity helpers through the CLI entrypoint.

use anyhow::Result;
use bitnet_startup_contract_guard::feature_line;

/// CLI interface version (SemVer for CLI surface compatibility).
pub(crate) const INTERFACE_VERSION: &str = "1.0.0";
pub(crate) const RTX_5070_TI_CUDA: &str = "nvidia-rtx-5070-ti-cuda";
pub(crate) const BITNET_CPP_ANSWER_TEMPLATE: &str = "bitnetcpp-answer";

pub(crate) fn bitnet_version() -> &'static str {
    use std::sync::OnceLock;
    static VERSION_STRING: OnceLock<String> = OnceLock::new();

    VERSION_STRING.get_or_init(|| {
        let features_line = feature_line();

        #[cfg(feature = "iq2s-ffi")]
        let ggml_line = format!("ggml: {}", bitnet_ggml_ffi::GGML_COMMIT);
        #[cfg(not(feature = "iq2s-ffi"))]
        let ggml_line = String::new();

        if ggml_line.is_empty() {
            format!("{}\n{}", env!("CARGO_PKG_VERSION"), features_line)
        } else {
            format!("{}\n{}\n{}", env!("CARGO_PKG_VERSION"), features_line, ggml_line)
        }
    })
}

pub(crate) fn sha256_token_ids(tokens: &[u32]) -> Result<String> {
    Ok(sha256_hex_bytes(&serde_json::to_vec(tokens)?))
}

pub(crate) fn critical_not_claims() -> Vec<&'static str> {
    vec![
        "selected_attention_residency",
        "resident_kv_decode",
        "attention_scores_residency",
        "softmax_residency",
        "attention_value_mix_residency",
        "full_support_op_residency",
        "full_device_residency",
        "completion",
    ]
}

fn sha256_hex_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
