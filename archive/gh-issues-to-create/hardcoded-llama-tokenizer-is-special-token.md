# Hardcoded values: `LlamaTokenizerWrapper::is_special_token` in `strategy.rs` has hardcoded values

The `LlamaTokenizerWrapper::is_special_token` function in `crates/bitnet-tokenizers/src/strategy.rs` has hardcoded values for LLaMA-2 and LLaMA-3 special tokens. These values may not be appropriate for all LLaMA variants. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/strategy.rs`

**Function:** `LlamaTokenizerWrapper::is_special_token`

**Code:**
```rust
    fn is_special_token(&self, token: u32) -> bool {
        match self.model_variant {
            LlamaVariant::Llama2 => {
                matches!(token, 0..=2) // UNK, BOS, EOS
            }
            LlamaVariant::Llama3 => {
                matches!(token, 128000..=128002) // LLaMA-3 special tokens
            }
            LlamaVariant::CodeLlama => {
                matches!(token, 0..=2) // Similar to LLaMA-2
            }
        }
    }
```

## Proposed Fix

The `LlamaTokenizerWrapper::is_special_token` function should be implemented to dynamically get the special token IDs from the tokenizer. This would involve adding `unk_token_id`, `bos_token_id`, and `eos_token_id` fields to the `LlamaTokenizerWrapper` struct and initializing them from the inner tokenizer.

### Example Implementation

```rust
pub struct LlamaTokenizerWrapper {
    inner: Arc<dyn Tokenizer>,
    vocab_size: usize,
    model_variant: LlamaVariant,
    unk_token_id: Option<u32>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

impl LlamaTokenizerWrapper {
    // ...
    fn is_special_token(&self, token: u32) -> bool {
        self.unk_token_id.map_or(false, |id| token == id) ||
        self.bos_token_id.map_or(false, |id| token == id) ||
        self.eos_token_id.map_or(false, |id| token == id)
    }
}
```
