# Hardcoded value: `model.model_type.contains("gpt2")` in `loader.rs`

The `model.model_type.contains("gpt2")` check in `crates/bitnet-inference/src/loader.rs` is a hardcoded value that may not be appropriate for all models.

**File:** `crates/bitnet-inference/src/loader.rs`

**Code:**
```rust
        // Model-specific overrides
        if config.model_type.contains("gpt2") {
            policy.add_bos = false;  // GPT-2 doesn't use BOS
        }
```

## Proposed Fix

Instead of hardcoding the `gpt2` check, the `determine_scoring_policy` function should use a more generic approach to determine whether to add the BOS token. This could involve adding a `add_bos_token` field to the `ModelConfig` struct and setting it based on the model type during model loading.

### Example Implementation

```rust
// In bitnet_models/src/config.rs

pub struct ModelConfig {
    // ...
    pub add_bos_token: bool,
}

// In bitnet_inference/src/loader.rs

    fn determine_scoring_policy(&self, model: &Arc<dyn Model>, tokenizer: &Arc<dyn Tokenizer>) -> ScoringPolicy {
        let config = model.config();
        let mut policy = ScoringPolicy::default();

        policy.add_bos = config.add_bos_token;

        // ... rest of the function ...
    }
```
