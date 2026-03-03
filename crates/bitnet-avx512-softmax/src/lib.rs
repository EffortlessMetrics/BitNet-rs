//! AVX-512 accelerated softmax kernels with scalar fallback.
//!
//! This crate provides numerically stable softmax implementations optimised for
//! `BitNet` inference workloads.  When running on a CPU with AVX-512F support the
//! hot loops are dispatched to hand-written intrinsics; otherwise an equivalent
//! scalar path is used transparently.
//!
//! # Variants
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`softmax`] | Standard numerically-stable softmax |
//! | [`softmax_inplace`] | In-place softmax (avoids allocation) |
//! | [`online_softmax`] | Single-pass online (streaming) softmax |
//! | [`log_softmax`] | Log-softmax (log∘softmax) |
//! | [`temperature_softmax`] | Temperature-scaled softmax |
//! | [`masked_softmax`] | Masked softmax for attention layers |
//! | [`batch_softmax`] | Batched softmax over multiple rows |
//! | [`batch_softmax_inplace`] | In-place batched softmax |
//!
//! # Runtime dispatch
//!
//! All public entry points call [`dispatch::has_avx512f`] once (via `std::arch::is_x86_feature_detected`)
//! and route to the appropriate backend.  The check is performed per-call so that
//! the same binary works on heterogeneous fleets.

// Re-export error types from bitnet-common.
pub use bitnet_common::BitNetError;

// ---------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx512;

mod dispatch;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the numerically-stable softmax of `logits`, returning a new `Vec<f32>`.
///
/// Uses the *max-subtract* trick: `softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))`.
///
/// # Errors
///
/// Returns [`BitNetError::Validation`] when `logits` is empty.
pub fn softmax(logits: &[f32]) -> Result<Vec<f32>, BitNetError> {
    validate_non_empty(logits)?;
    let mut out = logits.to_vec();
    dispatch::softmax_inplace_dispatch(&mut out);
    Ok(out)
}

/// Compute softmax **in-place**, overwriting `logits` with probabilities.
///
/// # Errors
///
/// Returns [`BitNetError::Validation`] when `logits` is empty.
pub fn softmax_inplace(logits: &mut [f32]) -> Result<(), BitNetError> {
    validate_non_empty(logits)?;
    dispatch::softmax_inplace_dispatch(logits);
    Ok(())
}

/// Single-pass online softmax (numerically stable, streaming friendly).
///
/// Implements the *online normalisation* trick that computes the softmax in a
/// single forward pass without a separate max-finding sweep.
///
/// # Errors
///
/// Returns [`BitNetError::Validation`] when `logits` is empty.
pub fn online_softmax(logits: &[f32]) -> Result<Vec<f32>, BitNetError> {
    validate_non_empty(logits)?;
    Ok(dispatch::online_softmax_dispatch(logits))
}

/// Log-softmax: `log_softmax(x)_i = x_i - max(x) - log(Σ exp(x_j - max(x)))`.
///
/// More numerically stable than computing `softmax` then `log`.
///
/// # Errors
///
/// Returns [`BitNetError::Validation`] when `logits` is empty.
pub fn log_softmax(logits: &[f32]) -> Result<Vec<f32>, BitNetError> {
    validate_non_empty(logits)?;
    Ok(dispatch::log_softmax_dispatch(logits))
}

/// Temperature-scaled softmax: `softmax(x / temperature)`.
///
/// `temperature` controls the sharpness of the distribution:
/// - `temperature < 1.0` → sharper (more confident)
/// - `temperature > 1.0` → smoother (more uniform)
///
/// # Errors
///
/// Returns [`BitNetError::Validation`] when `logits` is empty or
/// `temperature` is not positive and finite.
pub fn temperature_softmax(logits: &[f32], temperature: f32) -> Result<Vec<f32>, BitNetError> {
    validate_non_empty(logits)?;
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(BitNetError::Validation(format!(
            "temperature must be positive and finite, got {temperature}"
        )));
    }
    Ok(dispatch::temperature_softmax_dispatch(logits, temperature))
}

/// Masked softmax for attention layers.
///
/// Positions where `mask[i]` is `false` are set to `f32::NEG_INFINITY` before
/// the softmax so they receive zero probability.
///
/// # Errors
///
/// Returns [`BitNetError::Validation`] when `logits` is empty or
/// `logits.len() != mask.len()`.
pub fn masked_softmax(logits: &[f32], mask: &[bool]) -> Result<Vec<f32>, BitNetError> {
    validate_non_empty(logits)?;
    if logits.len() != mask.len() {
        return Err(BitNetError::Validation(format!(
            "logits length ({}) must equal mask length ({})",
            logits.len(),
            mask.len()
        )));
    }
    Ok(dispatch::masked_softmax_dispatch(logits, mask))
}

/// Batched softmax: apply softmax independently to each row of a matrix
/// stored as a flat slice with the given `row_len`.
///
/// # Errors
///
/// Returns [`BitNetError::Validation`] when `row_len` is zero or
/// `logits.len()` is not a multiple of `row_len`.
pub fn batch_softmax(logits: &[f32], row_len: usize) -> Result<Vec<f32>, BitNetError> {
    validate_batch(logits, row_len)?;
    let mut out = logits.to_vec();
    dispatch::batch_softmax_inplace_dispatch(&mut out, row_len);
    Ok(out)
}

/// In-place batched softmax.
///
/// # Errors
///
/// Returns [`BitNetError::Validation`] when `row_len` is zero or
/// `logits.len()` is not a multiple of `row_len`.
pub fn batch_softmax_inplace(logits: &mut [f32], row_len: usize) -> Result<(), BitNetError> {
    validate_batch(logits, row_len)?;
    dispatch::batch_softmax_inplace_dispatch(logits, row_len);
    Ok(())
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_non_empty(logits: &[f32]) -> Result<(), BitNetError> {
    if logits.is_empty() {
        return Err(BitNetError::Validation("logits slice must not be empty".into()));
    }
    Ok(())
}

fn validate_batch(logits: &[f32], row_len: usize) -> Result<(), BitNetError> {
    if row_len == 0 {
        return Err(BitNetError::Validation("row_len must be greater than zero".into()));
    }
    if !logits.len().is_multiple_of(row_len) {
        return Err(BitNetError::Validation(format!(
            "logits length ({}) must be a multiple of row_len ({})",
            logits.len(),
            row_len
        )));
    }
    if logits.is_empty() {
        return Err(BitNetError::Validation("logits slice must not be empty".into()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
