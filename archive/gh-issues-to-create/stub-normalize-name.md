# Stub code: `normalize_name` in `weight_mapper.rs` uses hardcoded string replacements

The `normalize_name` function in `crates/bitnet-models/src/weight_mapper.rs` uses hardcoded string replacements to normalize exporter name drift. This might not be flexible enough to handle new name variations. This is a form of stubbing.

**File:** `crates/bitnet-models/src/weight_mapper.rs`

**Function:** `normalize_name`

**Code:**
```rust
fn normalize_name(name: &str) -> Cow<'_, str> {
    if name.contains("attention_sub_norm") {
        // Map Microsoft's variation to our canonical name
        let s = name.replace("attention_sub_norm", "attn_sub_norm");
        return Cow::Owned(s);
    }
    if name.contains("mlp_sub_layernorm") {
        // Map to our canonical FFN sub norm
        let s = name.replace("mlp_sub_layernorm", "ffn_sub_norm");
        return Cow::Owned(s);
    }
    Cow::Borrowed(name)
}
```

## Proposed Fix

The `normalize_name` function should be implemented to use a more flexible and extensible approach for normalizing exporter name drift. This could involve using a configuration file or a plugin system to define the mapping rules.

### Example Implementation

```rust
fn normalize_name(name: &str, mapping_rules: &HashMap<String, String>) -> Cow<'_, str> {
    if let Some(canonical_name) = mapping_rules.get(name) {
        Cow::Owned(canonical_name.clone())
    } else {
        Cow::Borrowed(name)
    }
}
```
