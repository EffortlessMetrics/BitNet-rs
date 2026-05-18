//! Model and free-form label normalization helpers.

pub(crate) fn is_official_microsoft_2b(model: &str) -> bool {
    model.contains("microsoft_bitnet_b1_58_2b_4t") || model.contains("microsoft_bitnet_b158_2b_4t")
}

pub(crate) fn is_1bitllm_3b(model: &str) -> bool {
    model.contains("1bitllm_bitnet_b1_58_3b") || model.contains("bitnet_b1_58_3b")
}

pub(crate) fn normalize_label(label: &str) -> String {
    label
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}
