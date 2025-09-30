# Simulation: `linear_with_optional_bias` in `transformer.rs` injects zeros if bias is missing

The `linear_with_optional_bias` function in `crates/bitnet-models/src/transformer.rs` injects zeros if the bias tensor is missing. This might not be the desired behavior in all cases. This is a form of simulation.

**File:** `crates/bitnet-models/src/transformer.rs`

**Function:** `linear_with_optional_bias`

**Code:**
```rust
fn linear_with_optional_bias(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
) -> candle_core::Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;

    // Try to get bias, create zeros if missing
    let bias = match vb.get(out_dim, "bias") {
        Ok(b) => Some(b),
        Err(_) => {
            tracing::debug!("Bias tensor missing for linear layer; injecting zeros [{}]", out_dim);
            Some(Tensor::zeros(out_dim, DType::F32, vb.device())?)
        }
    };

    Ok(Linear::new(weight, bias))
}
```

## Proposed Fix

The `linear_with_optional_bias` function should not inject zeros if the bias tensor is missing. Instead, it should return an error if the bias tensor is missing. This will ensure that the function behaves correctly and doesn't hide missing bias tensors.

### Example Implementation

```rust
fn linear_with_optional_bias(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
) -> candle_core::Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;

    // Try to get bias, return error if missing
    let bias = match vb.get(out_dim, "bias") {
        Ok(b) => Some(b),
        Err(e) => {
            return Err(candle_core::Error::Msg(format!("Bias tensor missing for linear layer: {}", e)));
        }
    };

    Ok(Linear::new(weight, bias))
}
```
