use bitnet_weight_name_normalization::{normalize_exporter_name_drift, normalize_with_rules};

#[test]
fn default_rules_normalize_microsoft_attention_alias() {
    let normalized = normalize_exporter_name_drift("blk.0.attention_sub_norm.weight");
    assert_eq!(normalized, "blk.0.attn_sub_norm.weight");
}

#[test]
fn default_rules_normalize_ffn_alias() {
    let normalized = normalize_exporter_name_drift("blk.0.mlp_sub_layernorm.weight");
    assert_eq!(normalized, "blk.0.ffn_sub_norm.weight");
}

#[test]
fn default_rules_no_match_keeps_borrowed_input() {
    let source = "blk.0.attn_norm.weight";
    let normalized = normalize_exporter_name_drift(source);
    assert_eq!(normalized, source);
    assert!(matches!(normalized, std::borrow::Cow::Borrowed(_)));
}

#[test]
fn custom_rules_apply_in_order() {
    let rules = [("foo", "bar"), ("bar", "baz")];
    let normalized = normalize_with_rules("foo.weight", &rules);
    assert_eq!(normalized, "baz.weight");
}
