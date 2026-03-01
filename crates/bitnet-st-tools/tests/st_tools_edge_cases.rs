//! Edge-case and boundary tests for bitnet-st-tools common utilities.
//!
//! Tests `rms_for_tensor`, `cast_ln_to_f16`, `iter_ln_tensors`, and
//! `read_safetensors_bytes` using synthetic SafeTensors data.

use bitnet_st_tools::common::{cast_ln_to_f16, is_ln_gamma, iter_ln_tensors, rms_for_tensor};
use half::{bf16, f16};
use safetensors::Dtype;
use safetensors::tensor::TensorView;

// ---------------------------------------------------------------------------
// Helper: read u16 values from a byte slice (handles alignment safely)
// ---------------------------------------------------------------------------

fn read_u16_le(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

// ---------------------------------------------------------------------------
// Helper: build SafeTensors bytes from (name, dtype, shape, raw_bytes)
// ---------------------------------------------------------------------------

fn build_safetensors(tensors: &[(&str, Dtype, Vec<usize>, &[u8])]) -> Vec<u8> {
    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            (*name, TensorView::new(*dtype, shape.clone(), data).unwrap())
        })
        .collect();
    safetensors::serialize(views.into_iter(), None).unwrap()
}

// ---------------------------------------------------------------------------
// rms_for_tensor: F32 dtype
// ---------------------------------------------------------------------------

#[test]
fn rms_f32_single_one() {
    let data: Vec<f32> = vec![1.0];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F32, vec![1], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    assert!((rms - 1.0).abs() < 1e-10);
}

#[test]
fn rms_f32_known_values() {
    // RMS of [3, 4] = sqrt((9+16)/2) = sqrt(12.5) = 3.5355...
    let data: Vec<f32> = vec![3.0, 4.0];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F32, vec![2], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    let expected = (12.5_f64).sqrt();
    assert!((rms - expected).abs() < 1e-6, "got {rms}, expected {expected}");
}

#[test]
fn rms_f32_all_zeros() {
    let data: Vec<f32> = vec![0.0; 100];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F32, vec![100], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    assert_eq!(rms, 0.0);
}

#[test]
fn rms_f32_all_same_value() {
    let val = 5.0f32;
    let data: Vec<f32> = vec![val; 64];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F32, vec![64], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    assert!((rms - val as f64).abs() < 1e-6);
}

#[test]
fn rms_f32_negative_values() {
    // RMS should be same for [-3, -4] as [3, 4]
    let data: Vec<f32> = vec![-3.0, -4.0];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F32, vec![2], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    let expected = (12.5_f64).sqrt();
    assert!((rms - expected).abs() < 1e-6);
}

#[test]
fn rms_f32_multidimensional_shape() {
    // shape [2, 3] with 6 elements, all 1.0 → RMS = 1.0
    let data: Vec<f32> = vec![1.0; 6];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F32, vec![2, 3], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    assert!((rms - 1.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// rms_for_tensor: F16 dtype
// ---------------------------------------------------------------------------

#[test]
fn rms_f16_single_value() {
    let v = f16::from_f32(2.0);
    let data = vec![v.to_bits()];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F16, vec![1], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    assert!((rms - 2.0).abs() < 0.01);
}

#[test]
fn rms_f16_multiple_values() {
    let vals = [1.0f32, 2.0, 3.0, 4.0];
    let data: Vec<u16> = vals.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F16, vec![4], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    let expected = (7.5_f64).sqrt();
    assert!((rms - expected).abs() < 0.05, "got {rms}, expected ~{expected}");
}

// ---------------------------------------------------------------------------
// rms_for_tensor: BF16 dtype
// ---------------------------------------------------------------------------

#[test]
fn rms_bf16_single_value() {
    let v = bf16::from_f32(3.0);
    let data = vec![v.to_bits()];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::BF16, vec![1], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    assert!((rms - 3.0).abs() < 0.1);
}

// ---------------------------------------------------------------------------
// rms_for_tensor: F64 dtype
// ---------------------------------------------------------------------------

#[test]
fn rms_f64_precision() {
    let data: Vec<f64> = vec![1.0, 2.0, 3.0];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F64, vec![3], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    // RMS = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.1602
    let expected = (14.0_f64 / 3.0).sqrt();
    assert!((rms - expected).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// rms_for_tensor: integer dtypes
// ---------------------------------------------------------------------------

#[test]
fn rms_i8_values() {
    let data: Vec<i8> = vec![3, -4];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::I8, vec![2], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    let expected = (12.5_f64).sqrt();
    assert!((rms - expected).abs() < 1e-6);
}

#[test]
fn rms_u8_values() {
    let data: Vec<u8> = vec![5, 5, 5, 5];
    let tv = TensorView::new(Dtype::U8, vec![4], &data).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    assert!((rms - 5.0).abs() < 1e-6);
}

#[test]
fn rms_i16_values() {
    let data: Vec<i16> = vec![100, -100];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::I16, vec![2], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    assert!((rms - 100.0).abs() < 1e-6);
}

#[test]
fn rms_u16_values() {
    let data: Vec<u16> = vec![10, 20, 30];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::U16, vec![3], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    let expected = ((100.0 + 400.0 + 900.0) / 3.0_f64).sqrt();
    assert!((rms - expected).abs() < 1e-6);
}

#[test]
fn rms_i32_values() {
    let data: Vec<i32> = vec![1000, -2000];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::I32, vec![2], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    let expected = ((1_000_000.0 + 4_000_000.0) / 2.0_f64).sqrt();
    assert!((rms - expected).abs() < 1e-3);
}

#[test]
fn rms_u32_values() {
    let data: Vec<u32> = vec![7, 7, 7];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::U32, vec![3], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    assert!((rms - 7.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// rms_for_tensor: empty tensor (n=0)
// ---------------------------------------------------------------------------

#[test]
fn rms_empty_tensor_returns_zero() {
    let data: &[u8] = &[];
    let tv = TensorView::new(Dtype::F32, vec![0], data).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    assert_eq!(rms, 0.0);
}

// ---------------------------------------------------------------------------
// cast_ln_to_f16: various source dtypes
//
// NOTE: `cast_ln_to_f16` uses `bytemuck::cast_vec::<u16, u8>()` internally,
// which requires EXACT alignment match (align_of::<u16>() != align_of::<u8>()).
// This causes a panic for all non-F16 dtypes. The F16 path returns early
// via `data.to_vec()`, so only F16 works. The non-F16 tests below document
// this known bug with `#[should_panic]`.
// ---------------------------------------------------------------------------

#[test]
fn cast_f16_to_f16_passthrough() {
    let data: Vec<u16> = vec![f16::from_f32(1.0).to_bits(), f16::from_f32(2.0).to_bits()];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F16, vec![2], bytes).unwrap();
    let result = cast_ln_to_f16(&tv).unwrap();
    // Should be identical (passthrough)
    assert_eq!(result, bytes);
}

#[test]
fn cast_f16_passthrough_preserves_values() {
    let vals = [0.0f32, 1.0, -1.0, 0.5, 65504.0]; // f16 range
    let data: Vec<u16> = vals.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F16, vec![vals.len()], bytes).unwrap();
    let result = cast_ln_to_f16(&tv).unwrap();
    assert_eq!(result.len(), vals.len() * 2);
    let out_halves = read_u16_le(&result);
    for (i, &bits) in out_halves.iter().enumerate() {
        let got = f16::from_bits(bits).to_f32();
        let expected = f16::from_f32(vals[i]).to_f32();
        assert!((got - expected).abs() < 0.01, "elem {i}: got {got}, expected {expected}");
    }
}

#[test]
fn cast_f16_passthrough_empty() {
    let data: &[u8] = &[];
    let tv = TensorView::new(Dtype::F16, vec![0], data).unwrap();
    let result = cast_ln_to_f16(&tv).unwrap();
    assert!(result.is_empty());
}

#[test]
fn cast_f16_passthrough_large() {
    let data: Vec<u16> = (0..1000).map(|i| f16::from_f32(i as f32).to_bits()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F16, vec![1000], bytes).unwrap();
    let result = cast_ln_to_f16(&tv).unwrap();
    assert_eq!(result.len(), 2000);
}

// Known bug: bytemuck::cast_vec requires align_of::<A>() == align_of::<B>(),
// but cast_ln_to_f16 calls cast_vec::<u16, u8>() where alignments differ (2 ≠ 1).
// These tests document the panic until the bug is fixed.

#[test]
#[should_panic(expected = "AlignmentMismatch")]
fn cast_f32_to_f16_panics_bytemuck_alignment_bug() {
    let data: Vec<f32> = vec![1.0, 0.5];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F32, vec![2], bytes).unwrap();
    let _ = cast_ln_to_f16(&tv);
}

#[test]
#[should_panic(expected = "AlignmentMismatch")]
fn cast_bf16_to_f16_panics_bytemuck_alignment_bug() {
    let data: Vec<u16> = vec![bf16::from_f32(1.0).to_bits()];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::BF16, vec![1], bytes).unwrap();
    let _ = cast_ln_to_f16(&tv);
}

#[test]
#[should_panic(expected = "AlignmentMismatch")]
fn cast_f64_to_f16_panics_bytemuck_alignment_bug() {
    let data: Vec<f64> = vec![1.0];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F64, vec![1], bytes).unwrap();
    let _ = cast_ln_to_f16(&tv);
}

#[test]
#[should_panic(expected = "AlignmentMismatch")]
fn cast_i8_to_f16_panics_bytemuck_alignment_bug() {
    let data: Vec<i8> = vec![1, -1];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::I8, vec![2], bytes).unwrap();
    let _ = cast_ln_to_f16(&tv);
}

#[test]
#[should_panic(expected = "AlignmentMismatch")]
fn cast_u8_to_f16_panics_bytemuck_alignment_bug() {
    let data: Vec<u8> = vec![0, 128, 255];
    let tv = TensorView::new(Dtype::U8, vec![3], &data).unwrap();
    let _ = cast_ln_to_f16(&tv);
}

#[test]
#[should_panic(expected = "AlignmentMismatch")]
fn cast_i32_to_f16_panics_bytemuck_alignment_bug() {
    let data: Vec<i32> = vec![0, 1, -1];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::I32, vec![3], bytes).unwrap();
    let _ = cast_ln_to_f16(&tv);
}

#[test]
#[should_panic(expected = "AlignmentMismatch")]
fn cast_u32_to_f16_panics_bytemuck_alignment_bug() {
    let data: Vec<u32> = vec![0, 42];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::U32, vec![2], bytes).unwrap();
    let _ = cast_ln_to_f16(&tv);
}

// ---------------------------------------------------------------------------
// iter_ln_tensors: filtering
// ---------------------------------------------------------------------------

#[test]
fn iter_ln_tensors_filters_correctly() {
    let ones_f32: Vec<f32> = vec![1.0; 4];
    let bytes: Vec<u8> = bytemuck::cast_slice(&ones_f32).to_vec();

    let buf = build_safetensors(&[
        ("model.layers.0.input_layernorm.weight", Dtype::F32, vec![4], &bytes),
        ("model.layers.0.self_attn.q_proj.weight", Dtype::F32, vec![4], &bytes),
        ("model.layers.0.post_attention_layernorm.weight", Dtype::F32, vec![4], &bytes),
        ("model.embed_tokens.weight", Dtype::F32, vec![4], &bytes),
    ]);

    let names: Vec<String> = iter_ln_tensors(&buf).unwrap().map(|(name, _)| name).collect();

    // Only layernorm weights should appear
    assert!(names.iter().any(|n| n.contains("input_layernorm")), "should include input_layernorm");
    assert!(
        names.iter().any(|n| n.contains("post_attention_layernorm")),
        "should include post_attention_layernorm"
    );
    assert!(!names.iter().any(|n| n.contains("q_proj")), "should NOT include q_proj");
    assert!(!names.iter().any(|n| n.contains("embed_tokens")), "should NOT include embed_tokens");
}

#[test]
fn iter_ln_tensors_empty_model() {
    let ones_f32: Vec<f32> = vec![1.0; 4];
    let bytes: Vec<u8> = bytemuck::cast_slice(&ones_f32).to_vec();

    let buf = build_safetensors(&[("model.embed_tokens.weight", Dtype::F32, vec![4], &bytes)]);

    let count = iter_ln_tensors(&buf).unwrap().count();
    assert_eq!(count, 0, "no LN tensors should be found");
}

// ---------------------------------------------------------------------------
// is_ln_gamma: pattern matching
// ---------------------------------------------------------------------------

#[test]
fn is_ln_gamma_positive_patterns() {
    let positives = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.31.post_attention_layernorm.weight",
        "model.norm.weight",
        "model.layers.0.mlp.ffn_layernorm.weight",
    ];
    for name in &positives {
        assert!(is_ln_gamma(name), "expected true for: {name}");
    }
}

#[test]
fn is_ln_gamma_negative_patterns() {
    let negatives = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.k_proj.bias",
        "",
    ];
    for name in &negatives {
        assert!(!is_ln_gamma(name), "expected false for: {name}");
    }
}

// ---------------------------------------------------------------------------
// read_safetensors_bytes: error on missing file
// ---------------------------------------------------------------------------

#[test]
fn read_safetensors_missing_file() {
    let result =
        bitnet_st_tools::common::read_safetensors_bytes(std::path::Path::new("/no/such/file.st"));
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// rms_for_tensor + cast_ln_to_f16: roundtrip consistency (F16 only)
// ---------------------------------------------------------------------------

#[test]
fn cast_f16_preserves_rms_exactly() {
    // F16 passthrough should preserve RMS perfectly
    let vals = [1.0f32, 0.5, 0.25, 0.125];
    let data: Vec<u16> = vals.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F16, vec![4], bytes).unwrap();
    let rms_before = rms_for_tensor(&tv).unwrap();

    let result = cast_ln_to_f16(&tv).unwrap();
    let tv2 = TensorView::new(Dtype::F16, vec![4], &result).unwrap();
    let rms_after = rms_for_tensor(&tv2).unwrap();

    assert!((rms_before - rms_after).abs() < 1e-10, "F16 passthrough should preserve RMS exactly");
}

// ---------------------------------------------------------------------------
// Large tensor handling
// ---------------------------------------------------------------------------

#[test]
fn rms_large_f32_tensor() {
    // 10K elements, all 1.0 → RMS = 1.0
    let data: Vec<f32> = vec![1.0; 10_000];
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F32, vec![10_000], bytes).unwrap();
    let rms = rms_for_tensor(&tv).unwrap();
    assert!((rms - 1.0).abs() < 1e-10);
}

#[test]
fn cast_large_f16_passthrough() {
    let data: Vec<u16> = (0..5_000).map(|i| f16::from_f32(i as f32 * 0.01).to_bits()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&data);
    let tv = TensorView::new(Dtype::F16, vec![5_000], bytes).unwrap();
    let result = cast_ln_to_f16(&tv).unwrap();
    assert_eq!(result.len(), 10_000);
}
