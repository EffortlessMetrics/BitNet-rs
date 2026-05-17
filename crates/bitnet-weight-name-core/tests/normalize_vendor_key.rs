//! Comprehensive coverage of `normalize_vendor_key` regex patterns.
//!
//! These integration tests exercise every regex branch and helper in
//! `bitnet_weight_name_core::normalize_vendor_key`, including multi-digit
//! layer indices and a battery of rejection cases. The two existing unit
//! tests in `src/lib.rs` cover a small spot-check; the cases here are
//! intentionally non-overlapping with those.

use bitnet_weight_name_core::normalize_vendor_key;

/// Assert that each `(input, expected)` pair maps to the expected canonical key.
fn assert_table(rows: &[(&str, &str)]) {
    for (input, expected) in rows {
        let actual = normalize_vendor_key(input);
        assert_eq!(
            actual.as_deref(),
            Some(*expected),
            "input {input:?} expected {expected:?} got {actual:?}",
        );
    }
}

#[test]
fn attention_blk_aliases() {
    // `blk.N.attn_{q,k,v,o,output}.weight` -> canonical projections.
    assert_table(&[
        ("blk.0.attn_q.weight", "layers.0.attention.q_proj.weight"),
        ("blk.0.attn_k.weight", "layers.0.attention.k_proj.weight"),
        ("blk.0.attn_v.weight", "layers.0.attention.v_proj.weight"),
        ("blk.0.attn_o.weight", "layers.0.attention.o_proj.weight"),
        // Multi-digit layer indices must capture fully (regression guard
        // for `\d+` vs `\d` ambiguity).
        ("blk.12.attn_q.weight", "layers.12.attention.q_proj.weight"),
        ("blk.27.attn_k.weight", "layers.27.attention.k_proj.weight"),
        // `attn_output` is the GGUF-vendor spelling of `attn_o`.
        ("blk.7.attn_output.weight", "layers.7.attention.o_proj.weight"),
        ("blk.100.attn_output.weight", "layers.100.attention.o_proj.weight"),
    ]);
}

#[test]
fn attention_llama_w_aliases() {
    // llama "wq/wk/wv/wo" tensor names with the full matrix of optional
    // `model.` prefix and optional `self_` infix.
    assert_table(&[
        // bare `layers.N.attn.w*`
        ("layers.5.attn.wq.weight", "layers.5.attention.q_proj.weight"),
        ("layers.5.attn.wk.weight", "layers.5.attention.k_proj.weight"),
        ("layers.5.attn.wv.weight", "layers.5.attention.v_proj.weight"),
        ("layers.5.attn.wo.weight", "layers.5.attention.o_proj.weight"),
        // `layers.N.self_attn.w*`
        ("layers.6.self_attn.wq.weight", "layers.6.attention.q_proj.weight"),
        ("layers.6.self_attn.wo.weight", "layers.6.attention.o_proj.weight"),
        // `model.layers.N.attn.w*`
        ("model.layers.8.attn.wk.weight", "layers.8.attention.k_proj.weight"),
        // `model.layers.N.self_attn.w*` with a multi-digit index
        ("model.layers.14.self_attn.wv.weight", "layers.14.attention.v_proj.weight"),
        ("model.layers.27.self_attn.wo.weight", "layers.27.attention.o_proj.weight"),
    ]);
}

#[test]
fn attention_llama_proj_aliases() {
    // llama `{q,k,v,o}_proj` tensor names: same prefix matrix as the
    // `wq/wk/wv/wo` family.
    assert_table(&[
        ("layers.0.attn.q_proj.weight", "layers.0.attention.q_proj.weight"),
        ("layers.0.attn.k_proj.weight", "layers.0.attention.k_proj.weight"),
        ("layers.0.attn.v_proj.weight", "layers.0.attention.v_proj.weight"),
        ("layers.0.attn.o_proj.weight", "layers.0.attention.o_proj.weight"),
        ("layers.3.self_attn.q_proj.weight", "layers.3.attention.q_proj.weight"),
        ("layers.3.self_attn.o_proj.weight", "layers.3.attention.o_proj.weight"),
        ("model.layers.16.self_attn.q_proj.weight", "layers.16.attention.q_proj.weight"),
        ("model.layers.16.self_attn.v_proj.weight", "layers.16.attention.v_proj.weight"),
        ("model.layers.31.self_attn.o_proj.weight", "layers.31.attention.o_proj.weight"),
    ]);
}

#[test]
fn attention_qk_norm_aliases() {
    // `blk.N.attn_{q,k}_norm.weight` (handled by `normalize_blk_attn_norm`)
    // and llama `self_attn.{q,k}_norm.weight` (handled by
    // `normalize_llama_attn_norm`).
    assert_table(&[
        ("blk.0.attn_q_norm.weight", "layers.0.attention.q_norm.weight"),
        ("blk.0.attn_k_norm.weight", "layers.0.attention.k_norm.weight"),
        ("blk.27.attn_q_norm.weight", "layers.27.attention.q_norm.weight"),
        ("blk.42.attn_k_norm.weight", "layers.42.attention.k_norm.weight"),
        // `normalize_llama_attn_norm` accepts both `self_attn.*_norm` and
        // its alias with `attn.*_norm` (created via the
        // `.self_attn. -> .attn.` substitution in the helper).
        ("layers.5.self_attn.q_norm.weight", "layers.5.attention.q_norm.weight"),
        ("layers.5.self_attn.k_norm.weight", "layers.5.attention.k_norm.weight"),
        ("layers.7.attn.q_norm.weight", "layers.7.attention.q_norm.weight"),
        ("layers.7.attn.k_norm.weight", "layers.7.attention.k_norm.weight"),
        ("model.layers.18.self_attn.q_norm.weight", "layers.18.attention.q_norm.weight"),
        ("model.layers.31.self_attn.k_norm.weight", "layers.31.attention.k_norm.weight"),
    ]);
}

#[test]
fn ffn_blk_aliases() {
    // `blk.N.ffn_{gate,up,down}.weight` and the `_proj` / `_inp` aliases
    // recognised by the three `re_blk_ffn_*` regexes.
    assert_table(&[
        ("blk.0.ffn_gate.weight", "layers.0.feed_forward.gate_proj.weight"),
        ("blk.0.ffn_up.weight", "layers.0.feed_forward.up_proj.weight"),
        ("blk.0.ffn_down.weight", "layers.0.feed_forward.down_proj.weight"),
        // `*_proj` aliases supported by the `(?:up|up_proj)` /
        // `(?:down|down_proj)` regex groups.
        ("blk.3.ffn_up_proj.weight", "layers.3.feed_forward.up_proj.weight"),
        ("blk.3.ffn_down_proj.weight", "layers.3.feed_forward.down_proj.weight"),
        // `ffn_gate_inp` (mixture-of-experts router) is currently aliased
        // to `gate_proj` by the `(?:_inp)?` group; encoding the current
        // behavior so a regression would surface here.
        ("blk.9.ffn_gate_inp.weight", "layers.9.feed_forward.gate_proj.weight"),
        // Multi-digit indices.
        ("blk.15.ffn_gate.weight", "layers.15.feed_forward.gate_proj.weight"),
        ("blk.27.ffn_up.weight", "layers.27.feed_forward.up_proj.weight"),
        ("blk.27.ffn_down.weight", "layers.27.feed_forward.down_proj.weight"),
    ]);
}

#[test]
fn ffn_llama_aliases() {
    // llama `mlp.{gate,up,down}_proj.weight` and the `feed_forward.w1/w2/w3`
    // legacy style. Both `mlp` and `feed_forward` infixes are valid for
    // each of `w1/w2/w3` and `{gate,up,down}_proj`.
    assert_table(&[
        // `mlp.gate_proj` / `w1` -> gate_proj (canonical)
        ("layers.0.mlp.gate_proj.weight", "layers.0.feed_forward.gate_proj.weight"),
        ("layers.0.feed_forward.gate_proj.weight", "layers.0.feed_forward.gate_proj.weight"),
        ("layers.0.mlp.w1.weight", "layers.0.feed_forward.gate_proj.weight"),
        ("layers.0.feed_forward.w1.weight", "layers.0.feed_forward.gate_proj.weight"),
        // `mlp.up_proj` / `w3` -> up_proj
        ("model.layers.2.mlp.up_proj.weight", "layers.2.feed_forward.up_proj.weight"),
        ("layers.2.feed_forward.up_proj.weight", "layers.2.feed_forward.up_proj.weight"),
        ("model.layers.2.mlp.w3.weight", "layers.2.feed_forward.up_proj.weight"),
        ("layers.2.feed_forward.w3.weight", "layers.2.feed_forward.up_proj.weight"),
        // `mlp.down_proj` / `w2` -> down_proj
        ("model.layers.31.mlp.down_proj.weight", "layers.31.feed_forward.down_proj.weight"),
        ("layers.31.feed_forward.down_proj.weight", "layers.31.feed_forward.down_proj.weight"),
        ("model.layers.31.mlp.w2.weight", "layers.31.feed_forward.down_proj.weight"),
        ("layers.31.feed_forward.w2.weight", "layers.31.feed_forward.down_proj.weight"),
    ]);
}

#[test]
fn attention_norm_and_ffn_norm() {
    // Pre-attention norm: `attention_norm` and HF's `input_layernorm`
    // both map to `attention_norm`. Post-attention norm: `ffn_norm` and
    // HF's `post_attention_layernorm` both map to `post_attention_layernorm`.
    assert_table(&[
        // attention_norm spelling
        ("layers.0.attention_norm.weight", "layers.0.attention_norm.weight"),
        ("model.layers.0.attention_norm.weight", "layers.0.attention_norm.weight"),
        // input_layernorm spelling (HF)
        ("layers.4.input_layernorm.weight", "layers.4.attention_norm.weight"),
        ("model.layers.4.input_layernorm.weight", "layers.4.attention_norm.weight"),
        // ffn_norm spelling
        ("layers.0.ffn_norm.weight", "layers.0.post_attention_layernorm.weight"),
        ("model.layers.0.ffn_norm.weight", "layers.0.post_attention_layernorm.weight"),
        // post_attention_layernorm spelling (HF)
        ("layers.27.post_attention_layernorm.weight", "layers.27.post_attention_layernorm.weight"),
        (
            "model.layers.27.post_attention_layernorm.weight",
            "layers.27.post_attention_layernorm.weight",
        ),
    ]);
}

#[test]
fn layer_index_extraction() {
    // Spot-check that multi-digit layer indices flow through every helper
    // path (regex `\d+` captures, `normalize_blk_attn_norm` digit check,
    // `normalize_llama_attn_norm` digit check).
    assert_eq!(
        normalize_vendor_key("blk.123.attn_q.weight").as_deref(),
        Some("layers.123.attention.q_proj.weight"),
    );
    assert_eq!(
        normalize_vendor_key("model.layers.456.self_attn.k_proj.weight").as_deref(),
        Some("layers.456.attention.k_proj.weight"),
    );
    assert_eq!(
        normalize_vendor_key("blk.999.attn_q_norm.weight").as_deref(),
        Some("layers.999.attention.q_norm.weight"),
    );
    assert_eq!(
        normalize_vendor_key("layers.777.self_attn.k_norm.weight").as_deref(),
        Some("layers.777.attention.k_norm.weight"),
    );
}

#[test]
fn rejects_unknown_patterns() {
    // The empty key, unrelated tensor names, wrong suffixes, non-numeric
    // indices, and partial matches must all yield `None`.
    let rejects: &[&str] = &[
        // Empty / trivially unrelated.
        "",
        "tok_embeddings.weight",
        "output.weight",
        "norm.weight",
        "rope_freqs.weight",
        // Wrong suffix (`.bias` instead of `.weight`).
        "blk.0.attn_q.bias",
        "blk.0.attn_q_norm.bias",
        "blk.0.ffn_gate.bias",
        "layers.0.self_attn.q_proj.bias",
        "model.layers.0.mlp.gate_proj.bias",
        "layers.0.input_layernorm.bias",
        "layers.0.ffn_norm.bias",
        // Non-numeric layer index.
        "blk.x.attn_q.weight",
        "blk..attn_q.weight",
        "layers.abc.self_attn.q_proj.weight",
        "model.layers.-1.mlp.gate_proj.weight",
        // Extra path components or missing pieces.
        "blk.0.attn_q.weight.extra",
        "prefix.blk.0.attn_q.weight",
        "layers.0.something_else.q_proj.weight",
        // `mlp` / `feed_forward` is required for w1/w2/w3 — `attn.w1` is
        // not a recognised FFN alias.
        "layers.0.attn.w1.weight",
        // `attention_norm` / `input_layernorm` require the layer prefix.
        "attention_norm.weight",
        "input_layernorm.weight",
        // Embedded layer index with junk before it.
        "blk.0a.attn_q.weight",
        "blk.0 .attn_q.weight",
    ];
    for key in rejects {
        assert!(
            normalize_vendor_key(key).is_none(),
            "expected {key:?} to be rejected, got {:?}",
            normalize_vendor_key(key),
        );
    }
}
