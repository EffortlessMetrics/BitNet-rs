# Stub code: `TokenizerDiscovery::extract_model_type` in `discovery.rs` has a placeholder for tensor patterns

The `TokenizerDiscovery::extract_model_type` function in `crates/bitnet-tokenizers/src/discovery.rs` has a comment "Fallback based on tensor patterns" but only checks for LLaMA patterns. It doesn't implement a comprehensive fallback based on tensor patterns. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/discovery.rs`

**Function:** `TokenizerDiscovery::extract_model_type`

**Code:**
```rust
    fn extract_model_type(reader: &GgufReader) -> Result<String> {
        // Try to get architecture from metadata
        if let Some(arch) = reader.get_string_metadata("general.architecture") {
            return Ok(arch);
        }

        // Alternative metadata keys
        let alt_keys = [
            "model.architecture",
            "transformer.architecture",
            "llama.architecture",
            "gpt.architecture",
        ];

        for key in &alt_keys {
            if let Some(arch) = reader.get_string_metadata(key) {
                return Ok(arch);
            }
        }

        // Try to infer from model name
        if let Some(name) = reader.get_string_metadata("general.name") {
            let name_lower = name.to_lowercase();
            if name_lower.contains("llama") {
                return Ok("llama".to_string());
            } else if name_lower.contains("gpt") {
                return Ok("gpt2".to_string());
            } else if name_lower.contains("bitnet") {
                return Ok("bitnet".to_string());
            }
        }

        // Fallback based on tensor patterns
        let tensor_names = reader.tensor_names();
        let has_llama_patterns = tensor_names.iter().any(|name| {
            name.contains("attn_q") || name.contains("attn_k") || name.contains("attn_v")
        });

        if has_llama_patterns { Ok("llama".to_string()) } else { Ok("transformer".to_string()) }
    }
```

## Proposed Fix

The `TokenizerDiscovery::extract_model_type` function should implement a comprehensive fallback based on tensor patterns. This would involve checking for patterns specific to different model architectures (e.g., BERT, GPT-Neo, T5) in the tensor names.

### Example Implementation

```rust
    fn extract_model_type(reader: &GgufReader) -> Result<String> {
        // ... (existing code) ...

        // Fallback based on tensor patterns
        let tensor_names = reader.tensor_names();

        if tensor_names.iter().any(|name| name.contains("llama")) {
            return Ok("llama".to_string());
        }
        if tensor_names.iter().any(|name| name.contains("gpt2")) {
            return Ok("gpt2".to_string());
        }
        if tensor_names.iter().any(|name| name.contains("bert")) {
            return Ok("bert".to_string());
        }
        // ... (more patterns for other architectures) ...

        Ok("transformer".to_string())
    }
```
