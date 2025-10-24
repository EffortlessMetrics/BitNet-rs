# Stub code: `read_gguf_metadata` in `gguf_parity.rs` has hardcoded default values

The `read_gguf_metadata` function in `crates/bitnet-models/src/gguf_parity.rs` uses hardcoded default values for `vocab_size`, `hidden_size`, `num_layers`, and `num_heads` if they are not found in the GGUF metadata. This is a form of stubbing.

**File:** `crates/bitnet-models/src/gguf_parity.rs`

**Function:** `read_gguf_metadata`

**Code:**
```rust
pub fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata> {
    // ...

    // Extract required fields
    let vocab_size = metadata
        .get("llama.vocab_size")
        .or_else(|| metadata.get("tokenizer.ggml.tokens"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(50257); // GPT-2 default

    let hidden_size = metadata
        .get("llama.embedding_length")
        .or_else(|| metadata.get("llama.hidden_size"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(768);

    let num_layers = metadata
        .get("llama.block_count")
        .or_else(|| metadata.get("llama.layer_count"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12);

    let num_heads = metadata
        .get("llama.attention.head_count")
        .or_else(|| metadata.get("llama.head_count"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12);

    // ...
}
```

## Proposed Fix

The `read_gguf_metadata` function should not use hardcoded default values for `vocab_size`, `hidden_size`, `num_layers`, and `num_heads`. Instead, it should return an error if these fields are not found in the GGUF metadata. This will ensure that the function behaves correctly for all GGUF models.

### Example Implementation

```rust
pub fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata> {
    // ...

    // Extract required fields
    let vocab_size = metadata
        .get("llama.vocab_size")
        .or_else(|| metadata.get("tokenizer.ggml.tokens"))
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or_else(|| anyhow::anyhow!("Missing vocab_size in GGUF metadata"))?;

    let hidden_size = metadata
        .get("llama.embedding_length")
        .or_else(|| metadata.get("llama.hidden_size"))
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or_else(|| anyhow::anyhow!("Missing hidden_size in GGUF metadata"))?;

    let num_layers = metadata
        .get("llama.block_count")
        .or_else(|| metadata.get("llama.layer_count"))
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or_else(|| anyhow::anyhow!("Missing num_layers in GGUF metadata"))?;

    let num_heads = metadata
        .get("llama.attention.head_count")
        .or_else(|| metadata.get("llama.head_count"))
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or_else(|| anyhow::anyhow!("Missing num_heads in GGUF metadata"))?;

    // ...
}
```
