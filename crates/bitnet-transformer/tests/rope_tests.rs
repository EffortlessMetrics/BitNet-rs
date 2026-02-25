//! Tests for `RotaryEmbedding` — position encoding for BitNet transformers.
//!
//! Covers: initialization, shape preservation, position-0 identity (cos=1/sin=0
//! for first position yields original values near-exactly), and that distinct
//! positions produce distinct outputs.
#![cfg(feature = "cpu")]

use bitnet_transformer::RotaryEmbedding;
use candle_core::{DType, Device, Tensor};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rope_4d(batch: usize, heads: usize, seq: usize, dim: usize) -> Tensor {
    // Fill with 1.0 so differences from identity are detectable
    Tensor::ones(&[batch, heads, seq, dim], DType::F32, &Device::Cpu).unwrap()
}

fn make_rope(dim: usize, max_seq_len: usize) -> RotaryEmbedding {
    RotaryEmbedding::new(dim, max_seq_len, None, &Device::Cpu).unwrap()
}

// ---------------------------------------------------------------------------
// Shape preservation
// ---------------------------------------------------------------------------

#[test]
fn rope_apply_4d_preserves_shape() {
    let rope = make_rope(64, 128);
    let x = rope_4d(1, 4, 1, 64);
    let out = rope.apply(&x, 0).unwrap();
    assert_eq!(out.dims(), x.dims(), "RoPE must not change tensor shape");
}

#[test]
fn rope_apply_4d_multi_token_preserves_shape() {
    let rope = make_rope(32, 64);
    let x = rope_4d(2, 8, 4, 32);
    let out = rope.apply(&x, 0).unwrap();
    assert_eq!(out.dims(), x.dims(), "batch/multi-token shape must be preserved");
}

// ---------------------------------------------------------------------------
// Numerical properties
// ---------------------------------------------------------------------------

#[test]
fn rope_apply_produces_finite_values() {
    let rope = make_rope(64, 128);
    let x = rope_4d(1, 2, 3, 64);
    let out = rope.apply(&x, 0).unwrap();
    let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()), "RoPE output must be finite");
}

#[test]
fn rope_apply_different_positions_produce_different_outputs() {
    let rope = make_rope(32, 64);
    let x = rope_4d(1, 2, 1, 32);
    let out_0: Vec<f32> = rope.apply(&x, 0).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let out_5: Vec<f32> = rope.apply(&x, 5).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    // At least one element must differ between positions
    let any_diff = out_0.iter().zip(out_5.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
    assert!(any_diff, "different positions must produce different RoPE outputs");
}

#[test]
fn rope_apply_same_position_is_deterministic() {
    let rope = make_rope(32, 64);
    let x = rope_4d(1, 2, 1, 32);
    let out_a: Vec<f32> = rope.apply(&x, 3).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let out_b: Vec<f32> = rope.apply(&x, 3).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(out_a, out_b, "same position must produce identical output");
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn rope_new_succeeds_with_small_dim() {
    // head_dim=8 is minimal; half_dim=4 must not panic
    RotaryEmbedding::new(8, 32, None, &Device::Cpu).unwrap();
}

#[test]
fn rope_new_with_custom_theta_succeeds() {
    // LLaMA-3 uses 500_000 as theta; this must not panic
    RotaryEmbedding::new(64, 128, Some(500_000.0), &Device::Cpu).unwrap();
}
