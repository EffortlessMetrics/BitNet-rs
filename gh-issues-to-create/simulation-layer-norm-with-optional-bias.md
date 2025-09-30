# Simulation: `layer_norm_with_optional_bias` in `transformer.rs` falls back to RMSNorm if bias is missing

The `layer_norm_with_optional_bias` function in `crates/bitnet-models/src/transformer.rs` falls back to RMSNorm if the bias tensor is missing. This might not be the desired behavior in all cases. This is a form of simulation.

**File:** `crates/bitnet-models/src/transformer.rs`

**Function:** `layer_norm_with_optional_bias`

**Code:**
```rust
fn layer_norm_with_optional_bias(
    normalized_shape: usize,
    eps: f64,
    vb: VarBuilder,
) -> candle_core::Result<LayerNorm> {
    let weight = vb.get((normalized_shape,), "weight")?;
    match vb.get((normalized_shape,), "bias") {
        Ok(bias) => {
            // Bias exists → standard LayerNorm
            tracing::debug!("Using LayerNorm with bias [{}]", normalized_shape);
            Ok(LayerNorm::new(weight, bias, eps))
        }
        Err(_) => {
            // No bias → RMSNorm
            tracing::debug!(
                "Bias tensor missing for norm layer; using RMSNorm (no bias) [{}]",
                normalized_shape
            );
            Ok(LayerNorm::rms_norm(weight, eps))
        }
    }
}
```

## Proposed Fix

The `layer_norm_with_optional_bias` function should not fall back to RMSNorm if the bias tensor is missing. Instead, it should return an error if the bias tensor is missing. This will ensure that the function behaves correctly and doesn't hide missing bias tensors.

### Example Implementation

```rust
fn layer_norm_with_optional_bias(
    normalized_shape: usize,
    eps: f64,
    vb: VarBuilder,
) -> candle_core::Result<LayerNorm> {
    let weight = vb.get((normalized_shape,), "weight")?;
    match vb.get((normalized_shape,), "bias") {
        Ok(bias) => {
            // Bias exists → standard LayerNorm
            tracing::debug!("Using LayerNorm with bias [{}]", normalized_shape);
            Ok(LayerNorm::new(weight, bias, eps))
        }
        Err(e) => {
            return Err(candle_core::Error::Msg(format!("Bias tensor missing for norm layer: {}", e)));
        }
    }
}
```
