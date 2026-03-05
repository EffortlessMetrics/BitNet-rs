//! Vendor weight-name canonicalization helpers.
//!
//! This crate owns the mapping from exporter-specific tensor names
//! to the canonical BitNet transformer schema.

use regex::Regex;
use std::sync::OnceLock;

static RE_BLK_ATTN_Q: OnceLock<Regex> = OnceLock::new();
fn re_blk_attn_q() -> &'static Regex {
    RE_BLK_ATTN_Q.get_or_init(|| Regex::new(r"^blk\.(\d+)\.attn_q\.weight$").expect("valid regex"))
}

static RE_BLK_ATTN_K: OnceLock<Regex> = OnceLock::new();
fn re_blk_attn_k() -> &'static Regex {
    RE_BLK_ATTN_K.get_or_init(|| Regex::new(r"^blk\.(\d+)\.attn_k\.weight$").expect("valid regex"))
}

static RE_BLK_ATTN_V: OnceLock<Regex> = OnceLock::new();
fn re_blk_attn_v() -> &'static Regex {
    RE_BLK_ATTN_V.get_or_init(|| Regex::new(r"^blk\.(\d+)\.attn_v\.weight$").expect("valid regex"))
}

static RE_BLK_ATTN_O: OnceLock<Regex> = OnceLock::new();
fn re_blk_attn_o() -> &'static Regex {
    RE_BLK_ATTN_O
        .get_or_init(|| Regex::new(r"^blk\.(\d+)\.attn_o(?:utput)?\.weight$").expect("valid regex"))
}

static RE_LLAMA_WQ: OnceLock<Regex> = OnceLock::new();
fn re_llama_wq() -> &'static Regex {
    RE_LLAMA_WQ.get_or_init(|| {
        Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:self_)?attn\.wq\.weight$").expect("valid regex")
    })
}

static RE_LLAMA_WK: OnceLock<Regex> = OnceLock::new();
fn re_llama_wk() -> &'static Regex {
    RE_LLAMA_WK.get_or_init(|| {
        Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:self_)?attn\.wk\.weight$").expect("valid regex")
    })
}

static RE_LLAMA_WV: OnceLock<Regex> = OnceLock::new();
fn re_llama_wv() -> &'static Regex {
    RE_LLAMA_WV.get_or_init(|| {
        Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:self_)?attn\.wv\.weight$").expect("valid regex")
    })
}

static RE_LLAMA_WO: OnceLock<Regex> = OnceLock::new();
fn re_llama_wo() -> &'static Regex {
    RE_LLAMA_WO.get_or_init(|| {
        Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:self_)?attn\.wo\.weight$").expect("valid regex")
    })
}

static RE_BLK_FFN_GATE: OnceLock<Regex> = OnceLock::new();
fn re_blk_ffn_gate() -> &'static Regex {
    RE_BLK_FFN_GATE.get_or_init(|| {
        Regex::new(r"^blk\.(\d+)\.ffn_gate(?:_inp)?\.weight$").expect("valid regex")
    })
}

static RE_BLK_FFN_UP: OnceLock<Regex> = OnceLock::new();
fn re_blk_ffn_up() -> &'static Regex {
    RE_BLK_FFN_UP.get_or_init(|| {
        Regex::new(r"^blk\.(\d+)\.ffn_(?:up|up_proj)\.weight$").expect("valid regex")
    })
}

static RE_BLK_FFN_DOWN: OnceLock<Regex> = OnceLock::new();
fn re_blk_ffn_down() -> &'static Regex {
    RE_BLK_FFN_DOWN.get_or_init(|| {
        Regex::new(r"^blk\.(\d+)\.ffn_(?:down|down_proj)\.weight$").expect("valid regex")
    })
}

static RE_FFN_W1: OnceLock<Regex> = OnceLock::new();
fn re_ffn_w1() -> &'static Regex {
    RE_FFN_W1.get_or_init(|| {
        Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:mlp|feed_forward)\.(?:w1|gate_proj)\.weight$")
            .expect("valid regex")
    })
}

static RE_FFN_W3: OnceLock<Regex> = OnceLock::new();
fn re_ffn_w3() -> &'static Regex {
    RE_FFN_W3.get_or_init(|| {
        Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:mlp|feed_forward)\.(?:w3|up_proj)\.weight$")
            .expect("valid regex")
    })
}

static RE_FFN_W2: OnceLock<Regex> = OnceLock::new();
fn re_ffn_w2() -> &'static Regex {
    RE_FFN_W2.get_or_init(|| {
        Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:mlp|feed_forward)\.(?:w2|down_proj)\.weight$")
            .expect("valid regex")
    })
}

static RE_ATTN_NORM: OnceLock<Regex> = OnceLock::new();
fn re_attn_norm() -> &'static Regex {
    RE_ATTN_NORM.get_or_init(|| {
        Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:attention_norm|input_layernorm)\.weight$")
            .expect("valid regex")
    })
}

static RE_FFN_NORM: OnceLock<Regex> = OnceLock::new();
fn re_ffn_norm() -> &'static Regex {
    RE_FFN_NORM.get_or_init(|| {
        Regex::new(r"^(?:model\.)?layers\.(\d+)\.(?:post_attention_layernorm|ffn_norm)\.weight$")
            .expect("valid regex")
    })
}

/// Returns canonical key if `k` matches a known vendor pattern.
pub fn normalize_vendor_key(k: &str) -> Option<String> {
    macro_rules! cap {
        ($re_fn:expr, $k:expr, $fmt:expr) => {{ if let Some(c) = $re_fn().captures($k) { Some(format!($fmt, &c[1])) } else { None } }};
    }

    cap!(re_blk_attn_q, k, "layers.{}.attention.q_proj.weight")
        .or_else(|| cap!(re_blk_attn_k, k, "layers.{}.attention.k_proj.weight"))
        .or_else(|| cap!(re_blk_attn_v, k, "layers.{}.attention.v_proj.weight"))
        .or_else(|| cap!(re_blk_attn_o, k, "layers.{}.attention.o_proj.weight"))
        .or_else(|| cap!(re_llama_wq, k, "layers.{}.attention.q_proj.weight"))
        .or_else(|| cap!(re_llama_wk, k, "layers.{}.attention.k_proj.weight"))
        .or_else(|| cap!(re_llama_wv, k, "layers.{}.attention.v_proj.weight"))
        .or_else(|| cap!(re_llama_wo, k, "layers.{}.attention.o_proj.weight"))
        .or_else(|| cap!(re_blk_ffn_gate, k, "layers.{}.feed_forward.gate_proj.weight"))
        .or_else(|| cap!(re_blk_ffn_up, k, "layers.{}.feed_forward.up_proj.weight"))
        .or_else(|| cap!(re_blk_ffn_down, k, "layers.{}.feed_forward.down_proj.weight"))
        .or_else(|| cap!(re_ffn_w1, k, "layers.{}.feed_forward.gate_proj.weight"))
        .or_else(|| cap!(re_ffn_w3, k, "layers.{}.feed_forward.up_proj.weight"))
        .or_else(|| cap!(re_ffn_w2, k, "layers.{}.feed_forward.down_proj.weight"))
        .or_else(|| cap!(re_attn_norm, k, "layers.{}.attention_norm.weight"))
        .or_else(|| cap!(re_ffn_norm, k, "layers.{}.post_attention_layernorm.weight"))
}

#[cfg(test)]
mod tests {
    use super::normalize_vendor_key;

    #[test]
    fn maps_attention_and_ffn_aliases() {
        assert_eq!(
            normalize_vendor_key("blk.3.attn_output.weight").as_deref(),
            Some("layers.3.attention.o_proj.weight")
        );
        assert_eq!(
            normalize_vendor_key("model.layers.4.mlp.gate_proj.weight").as_deref(),
            Some("layers.4.feed_forward.gate_proj.weight")
        );
    }

    #[test]
    fn maps_norm_aliases_and_rejects_unknown() {
        assert_eq!(
            normalize_vendor_key("layers.2.input_layernorm.weight").as_deref(),
            Some("layers.2.attention_norm.weight")
        );
        assert!(normalize_vendor_key("layers.2.input_layernorm.bias").is_none());
    }
}
