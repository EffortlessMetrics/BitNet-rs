# Stub code: `BitNetAttention::apply_gqa` in `attention.rs` is a simplified implementation

The `BitNetAttention::apply_gqa` function in `crates/bitnet-inference/src/layers/attention.rs` has a comment "This is a simplified implementation". It just clones the key and value states. It doesn't implement the full GQA logic. This is a form of stubbing.

**File:** `crates/bitnet-inference/src/layers/attention.rs`

**Function:** `BitNetAttention::apply_gqa`

**Code:**
```rust
    fn apply_gqa(
        &self,
        key_states: &BitNetTensor,
        value_states: &BitNetTensor,
    ) -> Result<(BitNetTensor, BitNetTensor)> {
        // For GQA, we repeat the key and value states for each group
        // This is a simplified implementation
        Ok((key_states.clone(), value_states.clone()))
    }
```

## Proposed Fix

The `BitNetAttention::apply_gqa` function should be implemented to perform the full Grouped Query Attention (GQA) logic. This would involve repeating the key and value states for each query head group.

### Example Implementation

```rust
    fn apply_gqa(
        &self,
        key_states: &BitNetTensor,
        value_states: &BitNetTensor,
    ) -> Result<(BitNetTensor, BitNetTensor)> {
        let num_query_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let num_kv_groups = num_query_heads / num_kv_heads;

        let key_states_reshaped = key_states.to_candle()?.reshape(&[key_states.shape()[0], key_states.shape()[1], num_kv_heads, head_dim])?;
        let value_states_reshaped = value_states.to_candle()?.reshape(&[value_states.shape()[0], value_states.shape()[1], num_kv_heads, head_dim])?;

        let repeated_key_states = key_states_reshaped.repeat_interleave(num_kv_groups, 2)?;
        let repeated_value_states = value_states_reshaped.repeat_interleave(num_kv_groups, 2)?;

        Ok((BitNetTensor::new(repeated_key_states), BitNetTensor::new(repeated_value_states)))
    }
```
