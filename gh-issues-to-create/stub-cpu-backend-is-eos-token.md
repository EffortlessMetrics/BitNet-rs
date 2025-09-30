# Stub code: `CpuBackend::is_eos_token` in `cpu.rs` is a placeholder

The `CpuBackend::is_eos_token` function in `crates/bitnet-inference/src/cpu.rs` uses a hardcoded EOS token ID. It doesn't get the EOS token ID from the tokenizer. This is a form of stubbing and should be replaced with a real implementation.

**File:** `crates/bitnet-inference/src/cpu.rs`

**Function:** `CpuBackend::is_eos_token`

**Code:**
```rust
    fn is_eos_token(&self, token: u32) -> bool {
        token == 2 // Placeholder EOS token ID
    }
```

## Proposed Fix

The `CpuBackend::is_eos_token` function should be implemented to get the EOS token ID from the tokenizer. This would involve using the `bitnet_tokenizers` crate to get the EOS token ID.

### Example Implementation

```rust
    fn is_eos_token(&self, token: u32) -> bool {
        let tokenizer = bitnet_tokenizers::get_tokenizer(); // Assuming a global tokenizer instance
        tokenizer.eos_token_id().map_or(false, |eos_id| token == eos_id)
    }
```
