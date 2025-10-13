# Simulation: `cpu_optimizations::parallel_attention` in `cpu.rs` is a simplified implementation

The `cpu_optimizations::parallel_attention` function in `crates/bitnet-inference/src/cpu.rs` has a simplified softmax and weighted sum. It doesn't implement a full attention mechanism. This is a form of simulation and should be replaced with a real implementation.

**File:** `crates/bitnet-inference/src/cpu.rs`

**Function:** `cpu_optimizations::parallel_attention`

**Code:**
```rust
    pub fn parallel_attention(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        output: &mut [f32],
        seq_len: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> Result<()> {
        // Parallel processing by attention heads
        output
            .par_chunks_mut(seq_len * head_dim)
            .enumerate()
            .try_for_each(|(head_idx, head_output)| -> Result<()> {
                if head_idx >= num_heads {
                    return Ok(());
                }

                let q_offset = head_idx * seq_len * head_dim;
                let k_offset = head_idx * seq_len * head_dim;
                let v_offset = head_idx * seq_len * head_dim;

                // Compute attention scores for this head
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += query[q_offset + i * head_dim + d]
                                   * key[k_offset + j * head_dim + d];
                        }

                        // Apply softmax and compute weighted sum (simplified)
                        let weight = score.exp(); // Simplified softmax
                        for d in 0..head_dim {
                            head_output[i * head_dim + d] +=
                                weight * value[v_offset + j * head_dim + d];
                        }
                    }
                }

                Ok(())
            })?;

        Ok(())
    }
```

## Proposed Fix

The `cpu_optimizations::parallel_attention` function should be implemented to perform a full attention mechanism. This would involve:

1.  **Calculating attention scores:** Compute the dot product of the query and key matrices.
2.  **Applying a proper softmax function:** Apply a numerically stable softmax function to the attention scores.
3.  **Computing the weighted sum:** Compute the weighted sum of the value matrix using the attention weights.

### Example Implementation

```rust
    pub fn parallel_attention(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        output: &mut [f32],
        seq_len: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> Result<()> {
        // ... (calculate attention scores) ...

        // Apply softmax
        let softmax_scores = softmax(&attention_scores);

        // Compute weighted sum
        // ...

        Ok(())
    }

    fn softmax(input: &[f32]) -> Vec<f32> {
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp_values: f32 = exp_values.iter().sum();
        exp_values.iter().map(|&x| x / sum_exp_values).collect()
    }
```
