# Stub code: `dbg_stats` and `dbg_finite` in `transformer.rs` are conditionally compiled

The `dbg_stats` and `dbg_finite` functions in `crates/bitnet-models/src/transformer.rs` are conditionally compiled with the `DEBUG_ATTN` environment variable. This is a form of stubbing, as the debugging functionality is not always available.

**File:** `crates/bitnet-models/src/transformer.rs`

**Functions:**
* `dbg_stats`
* `dbg_finite`

**Code:**
```rust
/// Debug helper for tensor statistics (only runs if DEBUG_ATTN env var is set)
fn dbg_stats(tag: &str, t: &Tensor) -> candle_core::Result<()> {
    if std::env::var("DEBUG_ATTN").is_ok() {
        let mean = t.mean_all()?.to_scalar::<f32>()?;
        // Compute std manually: sqrt(E[(x - mean)^2])
        let diff = t.broadcast_sub(&t.mean_all()?)?;
        let variance = diff.sqr()?.mean_all()?;
        let std = variance.sqrt()?.to_scalar::<f32>()?;
        eprintln!("[dbg] {tag}: mean={mean:.6} std={std:.6}");
    }
    Ok(())
}

/// Debug helper for checking finite values
fn dbg_finite(tag: &str, t: &Tensor) -> candle_core::Result<()> {
    if std::env::var("DEBUG_ATTN").is_ok() {
        let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
        let n = v.len().min(4096);
        let mut n_nan = 0;
        let mut n_inf = 0;
        for &x in &v[..n] {
            if !x.is_finite() {
                if x.is_nan() {
                    n_nan += 1;
                } else {
                    n_inf += 1;
                }
            }
        }
        if n_nan + n_inf > 0 {
            eprintln!(
                "⚠️  [dbg] {tag}: non-finite values: NaN={n_nan} Inf={n_inf} (in first {n} elems)"
            );
        }
    }
    Ok(())
}
```

## Proposed Fix

The `dbg_stats` and `dbg_finite` functions should not be conditionally compiled with the `DEBUG_ATTN` environment variable. Instead, the debugging functionality should be integrated directly into the functions and controlled by a feature flag or a configuration option.

### Example Implementation

```rust
/// Debug helper for tensor statistics
fn dbg_stats(tag: &str, t: &Tensor, debug_enabled: bool) -> candle_core::Result<()> {
    if debug_enabled {
        let mean = t.mean_all()?.to_scalar::<f32>()?;
        let diff = t.broadcast_sub(&t.mean_all()?)?;
        let variance = diff.sqr()?.mean_all()?;
        let std = variance.sqrt()?.to_scalar::<f32>()?;
        eprintln!("[dbg] {tag}: mean={mean:.6} std={std:.6}");
    }
    Ok(())
}
```
