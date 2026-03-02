//! ARM NEON optimized softmax operations for `BitNet` LLM inference.
//!
//! Provides numerically stable softmax variants with NEON acceleration on
//! `aarch64` targets and automatic scalar fallbacks on other architectures.
//!
//! # Variants
//!
//! - [`softmax`] — standard softmax: `exp(x_i - max) / sum(exp(x_j - max))`
//! - [`softmax_inplace`] — in-place standard softmax
//! - [`log_softmax`] — numerically stable log-softmax via log-sum-exp
//! - [`temperature_softmax`] — temperature-scaled softmax
//! - [`online_softmax`] — single-pass online softmax (streaming friendly)
//! - [`batch_softmax`] / [`batch_softmax_inplace`] — row-wise batched softmax
//! - [`batch_log_softmax`] — row-wise batched log-softmax
//! - [`batch_temperature_softmax`] — row-wise batched temperature-scaled softmax

#![deny(unsafe_op_in_unsafe_fn)]

// ---------------------------------------------------------------------------
// NEON intrinsics (aarch64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon;

mod scalar;

// Re-export the implementation that matches the compile target.
// On aarch64 the NEON accelerated paths are used; everywhere else we fall back
// to portable scalar code.

#[cfg(target_arch = "aarch64")]
use neon as backend;

#[cfg(not(target_arch = "aarch64"))]
use scalar as backend;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Computes the softmax of `input`, returning a new `Vec<f32>`.
///
/// Uses the max-subtraction trick for numerical stability.
/// Returns an empty vector when `input` is empty.
#[must_use]
pub fn softmax(input: &[f32]) -> Vec<f32> {
    let mut out = input.to_vec();
    softmax_inplace(&mut out);
    out
}

/// Computes softmax **in place**, overwriting `input` with the result.
pub fn softmax_inplace(input: &mut [f32]) {
    if input.is_empty() {
        return;
    }
    backend::softmax_inplace(input);
}

/// Computes the log-softmax of `input`, returning a new `Vec<f32>`.
///
/// Uses the log-sum-exp trick: `log_softmax(x_i) = x_i - max - log(sum(exp(x_j - max)))`.
#[must_use]
pub fn log_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    backend::log_softmax(input)
}

/// Computes temperature-scaled softmax: `softmax(x / temperature)`.
///
/// # Panics
///
/// Panics if `temperature` is not positive and finite.
#[must_use]
pub fn temperature_softmax(input: &[f32], temperature: f32) -> Vec<f32> {
    assert!(
        temperature.is_finite() && temperature > 0.0,
        "temperature must be positive and finite, got {temperature}"
    );
    if input.is_empty() {
        return Vec::new();
    }
    backend::temperature_softmax(input, temperature)
}

/// Single-pass online softmax (numerically stable, streaming-friendly).
///
/// This algorithm processes elements in one pass, maintaining a running
/// maximum and normalisation factor, then performs a second pass to
/// normalise. Useful for very long sequences where two-pass max-finding
/// may be undesirable.
#[must_use]
pub fn online_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    backend::online_softmax(input)
}

/// Row-wise batched softmax.
///
/// `data` is a flat buffer of `rows × cols` elements stored in row-major
/// order. Each row is independently normalised via [`softmax_inplace`].
///
/// # Panics
///
/// Panics if `data.len()` is not evenly divisible by `cols`, or if `cols == 0`.
pub fn batch_softmax_inplace(data: &mut [f32], cols: usize) {
    assert!(cols > 0, "cols must be positive");
    assert!(
        data.len().is_multiple_of(cols),
        "data length {} is not divisible by cols {cols}",
        data.len()
    );
    for row in data.chunks_exact_mut(cols) {
        backend::softmax_inplace(row);
    }
}

/// Row-wise batched softmax, returning a new `Vec<f32>`.
///
/// See [`batch_softmax_inplace`] for semantics.
///
/// # Panics
///
/// Panics if `data.len()` is not evenly divisible by `cols`, or if `cols == 0`.
#[must_use]
pub fn batch_softmax(data: &[f32], cols: usize) -> Vec<f32> {
    let mut out = data.to_vec();
    batch_softmax_inplace(&mut out, cols);
    out
}

/// Row-wise batched log-softmax, returning a new `Vec<f32>`.
///
/// # Panics
///
/// Panics if `data.len()` is not evenly divisible by `cols`, or if `cols == 0`.
#[must_use]
pub fn batch_log_softmax(data: &[f32], cols: usize) -> Vec<f32> {
    assert!(cols > 0, "cols must be positive");
    assert!(
        data.len().is_multiple_of(cols),
        "data length {} is not divisible by cols {cols}",
        data.len()
    );
    let mut out = Vec::with_capacity(data.len());
    for row in data.chunks_exact(cols) {
        out.extend_from_slice(&backend::log_softmax(row));
    }
    out
}

/// Row-wise batched temperature-scaled softmax, returning a new `Vec<f32>`.
///
/// # Panics
///
/// Panics if `temperature` is not positive and finite, if `data.len()` is not
/// evenly divisible by `cols`, or if `cols == 0`.
#[must_use]
pub fn batch_temperature_softmax(data: &[f32], cols: usize, temperature: f32) -> Vec<f32> {
    assert!(
        temperature.is_finite() && temperature > 0.0,
        "temperature must be positive and finite, got {temperature}"
    );
    assert!(cols > 0, "cols must be positive");
    assert!(
        data.len().is_multiple_of(cols),
        "data length {} is not divisible by cols {cols}",
        data.len()
    );
    let mut out = Vec::with_capacity(data.len());
    for row in data.chunks_exact(cols) {
        out.extend_from_slice(&backend::temperature_softmax(row, temperature));
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
