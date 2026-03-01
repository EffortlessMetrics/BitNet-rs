//! Exporter weight-name drift normalization helpers.
//!
//! This crate has a single responsibility: normalize known vendor-specific
//! naming drift to canonical internal names before mapping.

use std::borrow::Cow;

const DEFAULT_DRIFT_RULES: [(&str, &str); 2] =
    [("attention_sub_norm", "attn_sub_norm"), ("mlp_sub_layernorm", "ffn_sub_norm")];

/// Normalize exporter-specific tensor-name drift to canonical names.
///
/// Returns [`Cow::Borrowed`] when no rule matches and [`Cow::Owned`] when a
/// replacement was applied.
pub fn normalize_exporter_name_drift(name: &str) -> Cow<'_, str> {
    normalize_with_rules(name, &DEFAULT_DRIFT_RULES)
}

/// Normalize using custom replacement rules, applied in order.
///
/// Rules are `(from, to)` replacement pairs and use `str::replace` semantics.
pub fn normalize_with_rules<'a>(name: &'a str, rules: &[(&str, &str)]) -> Cow<'a, str> {
    let mut normalized = Cow::Borrowed(name);

    for (from, to) in rules {
        if normalized.contains(from) {
            normalized = Cow::Owned(normalized.replace(from, to));
        }
    }

    normalized
}
