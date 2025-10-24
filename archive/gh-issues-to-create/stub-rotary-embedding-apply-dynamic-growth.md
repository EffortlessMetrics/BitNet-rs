# Stub code: `RotaryEmbedding::apply` in `transformer.rs` has a placeholder for dynamic growth

The `RotaryEmbedding::apply` function in `crates/bitnet-models/src/transformer.rs` has a comment "Sequence length {} exceeds max_seq_len {} (consider dynamic growth)". This suggests that dynamic growth is not implemented. This is a form of stubbing.

**File:** `crates/bitnet-models/src/transformer.rs`

**Function:** `RotaryEmbedding::apply`

**Code:**
```rust
    pub fn apply(&self, x: &Tensor, position: usize) -> Result<Tensor> {
        // x shape: [B, H, T, D] for multi-head attention
        if x.dims().len() == 4 {
            let (batch, n_heads, seq_len, head_dim) = x.dims4()?;
            let half_dim = head_dim / 2;

            // Reshape to separate real and imaginary parts
            let x_reshaped = x.reshape(&[batch, n_heads, seq_len, half_dim, 2])?;
            let x0 = x_reshaped.narrow(4, 0, 1)?.squeeze(4)?;
            let x1 = x_reshaped.narrow(4, 1, 1)?.squeeze(4)?;

            // Get cos/sin for the position
            let cos = self.cos.narrow(0, position, seq_len)?
                .unsqueeze(0)?  // Add batch dim
                .unsqueeze(1)?  // Add heads dim
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;
            let sin = self
                .sin
                .narrow(0, position, seq_len)?
                .unsqueeze(0)?
                .unsqueeze(1)?
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;

            let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
            let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

            let rotated = Tensor::stack(&[x0_rot, x1_rot], 4)?
                .reshape(&[batch, n_heads, seq_len, head_dim])?;

            Ok(rotated)
        } else {
            // Original 3D implementation for other uses
            let (_batch, _seq, dim) = x.dims3()?;
            let half_dim = dim / 2;

            let x_reshaped = x.reshape(&[x.dims()[0], x.dims()[1], half_dim, 2])?;
            let x0 = x_reshaped.narrow(3, 0, 1)?.squeeze(3)?;
            let x1 = x_reshaped.narrow(3, 1, 1)?.squeeze(3)?;

            let cos = self.cos.narrow(0, position, 1)?;
            let sin = self.sin.narrow(0, position, 1)?;

            let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
            let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

            let rotated =
                Tensor::stack(&[x0_rot, x1_rot], 3)?.reshape(&[x.dims()[0], x.dims()[1], dim])?;

            Ok(rotated)
        }
    }
```

## Proposed Fix

The `RotaryEmbedding::apply` function should be implemented to support dynamic growth of the rotary embeddings. This would involve dynamically resizing the `sin` and `cos` caches when the `max_seq_len` is exceeded.

### Example Implementation

```rust
    pub fn apply(&self, x: &Tensor, position: usize) -> Result<Tensor> {
        // ...

        if seq_len > self.max_seq_len {
            // Dynamically grow the sin and cos caches
            // This would involve re-calculating the sin and cos values for the new max_seq_len
            return Err(BitNetError::Validation(format!(
                "Sequence length {} exceeds max_seq_len {} (dynamic growth not implemented)",
                seq_len,
                self.max_seq_len
            )));
        }

        // ...
    }
```
