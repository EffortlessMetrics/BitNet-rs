//! Unit tests for LayerNorm SafeTensors helpers using synthetic in-memory tensors.

use bitnet_safetensors_ln::{
    cast_ln_to_f16, iter_ln_tensors, read_safetensors_bytes, rms_for_tensor,
};
use half::{bf16, f16};
use safetensors::Dtype;
use safetensors::tensor::TensorView;

fn read_u16_le(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]])).collect()
}

fn tensor_view<'a>(dtype: Dtype, shape: Vec<usize>, data: &'a [u8]) -> TensorView<'a> {
    TensorView::new(dtype, shape, data).expect("test tensor view should be well formed")
}

fn build_safetensors(tensors: &[(&str, Dtype, Vec<usize>, &[u8])]) -> Vec<u8> {
    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| (*name, tensor_view(*dtype, shape.clone(), data)))
        .collect();
    safetensors::serialize(views, None).expect("synthetic safetensors should serialize")
}

#[test]
fn rms_for_f32_tensor_matches_known_value() {
    let data = [3.0_f32, 4.0];
    let tensor = tensor_view(Dtype::F32, vec![2], bytemuck::cast_slice(&data));

    let rms = rms_for_tensor(&tensor).expect("f32 RMS should be supported");

    assert!((rms - 12.5_f64.sqrt()).abs() < 1e-6);
}

#[test]
fn rms_for_integer_tensor_squares_signed_values() {
    let data = [-3_i16, 4];
    let tensor = tensor_view(Dtype::I16, vec![2], bytemuck::cast_slice(&data));

    let rms = rms_for_tensor(&tensor).expect("i16 RMS should be supported");

    assert!((rms - 12.5_f64.sqrt()).abs() < 1e-10);
}

#[test]
fn rms_for_zero_sized_tensor_is_zero_without_reading_data() {
    let tensor = tensor_view(Dtype::F32, vec![0], &[]);

    assert_eq!(rms_for_tensor(&tensor).expect("empty tensor RMS should succeed"), 0.0);
}

#[test]
fn rms_rejects_unsupported_dtype_with_dtype_in_message() {
    let data = [true, false];
    let tensor = tensor_view(Dtype::BOOL, vec![2], bytemuck::cast_slice(&data));

    let err = rms_for_tensor(&tensor).expect_err("bool RMS should be unsupported");

    assert!(err.to_string().contains("BOOL"));
}

#[test]
fn cast_ln_to_f16_converts_f32_values_to_little_endian_half_bytes() {
    let data = [1.0_f32, -2.5, 0.5];
    let tensor = tensor_view(Dtype::F32, vec![3], bytemuck::cast_slice(&data));

    let bytes = cast_ln_to_f16(&tensor).expect("f32 cast should succeed");
    let halves = read_u16_le(&bytes);

    let expected: Vec<u16> = data.iter().map(|&value| f16::from_f32(value).to_bits()).collect();
    assert_eq!(halves, expected);
}

#[test]
fn cast_ln_to_f16_returns_f16_input_bytes_unchanged() {
    let halves = [f16::from_f32(1.25).to_bits(), f16::from_f32(-0.75).to_bits()];
    let bytes: &[u8] = bytemuck::cast_slice(&halves);
    let tensor = tensor_view(Dtype::F16, vec![2], bytes);

    assert_eq!(cast_ln_to_f16(&tensor).expect("f16 should pass through"), bytes);
}

#[test]
fn cast_ln_to_f16_converts_bf16_and_unsigned_integer_inputs() {
    let bf16_values = [bf16::from_f32(2.0).to_bits(), bf16::from_f32(-3.0).to_bits()];
    let bf16_tensor = tensor_view(Dtype::BF16, vec![2], bytemuck::cast_slice(&bf16_values));
    let bf16_halves = read_u16_le(&cast_ln_to_f16(&bf16_tensor).expect("bf16 cast"));
    assert_eq!(bf16_halves, vec![f16::from_f32(2.0).to_bits(), f16::from_f32(-3.0).to_bits()]);

    let u8_values = [2_u8, 7];
    let u8_tensor = tensor_view(Dtype::U8, vec![2], &u8_values);
    let u8_halves = read_u16_le(&cast_ln_to_f16(&u8_tensor).expect("u8 cast"));
    assert_eq!(u8_halves, vec![f16::from_f32(2.0).to_bits(), f16::from_f32(7.0).to_bits()]);
}

#[test]
fn cast_ln_to_f16_rejects_unsupported_dtype_with_dtype_in_message() {
    let data = [true];
    let tensor = tensor_view(Dtype::BOOL, vec![1], bytemuck::cast_slice(&data));

    let err = cast_ln_to_f16(&tensor).expect_err("bool cast should be unsupported");

    assert!(err.to_string().contains("BOOL"));
}

#[test]
fn iter_ln_tensors_filters_to_layernorm_gamma_names() {
    let ln = [1.0_f32, 2.0];
    let dense = [3.0_f32, 4.0];
    let bytes = build_safetensors(&[
        ("model.layers.0.input_layernorm.weight", Dtype::F32, vec![2], bytemuck::cast_slice(&ln)),
        ("model.layers.0.mlp.down_proj.weight", Dtype::F32, vec![2], bytemuck::cast_slice(&dense)),
        ("model.norm.weight", Dtype::F32, vec![2], bytemuck::cast_slice(&ln)),
    ]);

    let mut names: Vec<String> = iter_ln_tensors(&bytes)
        .expect("synthetic safetensors should deserialize")
        .map(|(name, _)| name)
        .collect();
    names.sort();

    assert_eq!(names, vec!["model.layers.0.input_layernorm.weight", "model.norm.weight"]);
}

#[test]
fn iter_ln_tensors_rejects_invalid_safetensors_bytes() {
    assert!(iter_ln_tensors(b"not a safetensors file").is_err());
}

#[test]
fn read_safetensors_bytes_reads_file_contents_exactly() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fixture.safetensors");
    let bytes = b"fixture bytes";
    std::fs::write(&path, bytes).expect("write fixture");

    assert_eq!(read_safetensors_bytes(&path).expect("read fixture"), bytes);
}
