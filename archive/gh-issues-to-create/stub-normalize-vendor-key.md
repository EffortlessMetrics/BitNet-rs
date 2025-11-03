# Stub code: `normalize_vendor_key` in `weight_mapper.rs` uses hardcoded regex patterns

The `normalize_vendor_key` function in `crates/bitnet-models/src/weight_mapper.rs` uses hardcoded regex patterns to normalize vendor keys. This might not be flexible enough to handle new vendor patterns. This is a form of stubbing.

**File:** `crates/bitnet-models/src/weight_mapper.rs`

**Function:** `normalize_vendor_key`

**Code:**
```rust
pub fn normalize_vendor_key(k: &str) -> Option<String> {
    macro_rules! cap {
        ($re:expr, $k:expr, $fmt:expr) => {{ if let Some(c) = $re.captures($k) { Some(format!($fmt, &c[1])) } else { None } }};
    }

    // Attention (blk.*)
    cap!(RE_BLK_ATTN_Q, k, "layers.{}.attention.q_proj.weight")
        .or_else(|| cap!(RE_BLK_ATTN_K, k, "layers.{}.attention.k_proj.weight"))
        .or_else(|| cap!(RE_BLK_ATTN_V, k, "layers.{}.attention.v_proj.weight"))
        .or_else(|| cap!(RE_BLK_ATTN_O, k, "layers.{}.attention.o_proj.weight"))
        // LLaMA-style attention
        .or_else(|| cap!(RE_LLAMA_WQ, k, "layers.{}.attention.q_proj.weight"))
        .or_else(|| cap!(RE_LLAMA_WK, k, "layers.{}.attention.k_proj.weight"))
        .or_else(|| cap!(RE_LLAMA_WV, k, "layers.{}.attention.wv.weight"))
        .or_else(|| cap!(RE_LLAMA_WO, k, "layers.{}.attention.wo.weight"))
        // FFN / MLP
        .or_else(|| cap!(RE_BLK_FFN_GATE, k, "layers.{}.feed_forward.gate_proj.weight"))
        .or_else(|| cap!(RE_BLK_FFN_UP,   k, "layers.{}.feed_forward.up_proj.weight"))
        .or_else(|| cap!(RE_BLK_FFN_DOWN, k, "layers.{}.feed_forward.down_proj.weight"))
        .or_else(|| cap!(RE_FFN_W1, k, "layers.{}.feed_forward.gate_proj.weight"))
        .or_else(|| cap!(RE_FFN_W3, k, "layers.{}.feed_forward.up_proj.weight"))
        .or_else(|| cap!(RE_FFN_W2, k, "layers.{}.feed_forward.down_proj.weight"))
        // Norms
        .or_else(|| cap!(RE_ATTN_NORM, k, "layers.{}.attention_norm.weight"))
        .or_else(|| cap!(RE_FFN_NORM,  k, "layers.{}.post_attention_layernorm.weight"))
}
```

## Proposed Fix

The `normalize_vendor_key` function should be implemented to use a more flexible and extensible approach for normalizing vendor keys. This could involve using a configuration file or a plugin system to define the mapping rules.

### Example Implementation

```rust
pub fn normalize_vendor_key(k: &str, mapping_rules: &HashMap<String, String>) -> Option<String> {
    if let Some(canonical_name) = mapping_rules.get(k) {
        Some(canonical_name.clone())
    } else {
        None
    }
}
```
