//! Unit tests for LayerNorm SafeTensors helpers using synthetic in-memory tensors.

use anyhow::Result;
use bitnet_safetensors_ln::{
    cast_ln_to_f16, iter_ln_tensors, read_safetensors_bytes, rms_for_tensor,
};
use half::{bf16, f16};
use safetensors::Dtype;
use safetensors::tensor::TensorView;

fn read_u16_le(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]])).collect()
}

fn tensor_view<'a>(dtype: Dtype, shape: Vec<usize>, data: &'a [u8]) -> Result<TensorView<'a>> {
    Ok(TensorView::new(dtype, shape, data)?)
}

fn build_safetensors(tensors: &[(&str, Dtype, Vec<usize>, &[u8])]) -> Result<Vec<u8>> {
    let views: Result<Vec<(&str, TensorView<'_>)>> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| Ok((*name, tensor_view(*dtype, shape.clone(), data)?)))
        .collect();
    Ok(safetensors::serialize(views?, None)?)
}

#[test]
fn rms_for_f32_tensor_matches_known_value() -> Result<()> {
    let data = [3.0_f32, 4.0];
    let tensor = tensor_view(Dtype::F32, vec![2], bytemuck::cast_slice(&data))?;

    let rms = rms_for_tensor(&tensor)?;

    assert!((rms - 12.5_f64.sqrt()).abs() < 1e-6);
    Ok(())
}

#[test]
fn rms_for_integer_tensor_squares_signed_values() -> Result<()> {
    let data = [-3_i16, 4];
    let tensor = tensor_view(Dtype::I16, vec![2], bytemuck::cast_slice(&data))?;

    let rms = rms_for_tensor(&tensor)?;

    assert!((rms - 12.5_f64.sqrt()).abs() < 1e-10);
    Ok(())
}

#[test]
fn rms_for_zero_sized_tensor_is_zero_without_reading_data() -> Result<()> {
    let tensor = tensor_view(Dtype::F32, vec![0], &[])?;

    assert_eq!(rms_for_tensor(&tensor)?, 0.0);
    Ok(())
}

#[test]
fn rms_rejects_unsupported_dtype_with_dtype_in_message() -> Result<()> {
    let data = [true, false];
    let tensor = tensor_view(Dtype::BOOL, vec![2], bytemuck::cast_slice(&data))?;

    let err = match rms_for_tensor(&tensor) {
        Ok(_) => anyhow::bail!("bool RMS should be unsupported"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("BOOL"));
    Ok(())
}

#[test]
fn cast_ln_to_f16_converts_f32_values_to_little_endian_half_bytes() -> Result<()> {
    let data = [1.0_f32, -2.5, 0.5];
    let tensor = tensor_view(Dtype::F32, vec![3], bytemuck::cast_slice(&data))?;

    let bytes = cast_ln_to_f16(&tensor)?;
    let halves = read_u16_le(&bytes);

    let expected: Vec<u16> = data.iter().map(|&value| f16::from_f32(value).to_bits()).collect();
    assert_eq!(halves, expected);
    Ok(())
}

#[test]
fn cast_ln_to_f16_returns_f16_input_bytes_unchanged() -> Result<()> {
    let halves = [f16::from_f32(1.25).to_bits(), f16::from_f32(-0.75).to_bits()];
    let bytes: &[u8] = bytemuck::cast_slice(&halves);
    let tensor = tensor_view(Dtype::F16, vec![2], bytes)?;

    assert_eq!(cast_ln_to_f16(&tensor)?, bytes);
    Ok(())
}

#[test]
fn cast_ln_to_f16_converts_bf16_and_unsigned_integer_inputs() -> Result<()> {
    let bf16_values = [bf16::from_f32(2.0).to_bits(), bf16::from_f32(-3.0).to_bits()];
    let bf16_tensor = tensor_view(Dtype::BF16, vec![2], bytemuck::cast_slice(&bf16_values))?;
    let bf16_halves = read_u16_le(&cast_ln_to_f16(&bf16_tensor)?);
    assert_eq!(bf16_halves, vec![f16::from_f32(2.0).to_bits(), f16::from_f32(-3.0).to_bits()]);

    let u8_values = [2_u8, 7];
    let u8_tensor = tensor_view(Dtype::U8, vec![2], &u8_values)?;
    let u8_halves = read_u16_le(&cast_ln_to_f16(&u8_tensor)?);
    assert_eq!(u8_halves, vec![f16::from_f32(2.0).to_bits(), f16::from_f32(7.0).to_bits()]);
    Ok(())
}

#[test]
fn cast_ln_to_f16_rejects_unsupported_dtype_with_dtype_in_message() -> Result<()> {
    let data = [true];
    let tensor = tensor_view(Dtype::BOOL, vec![1], bytemuck::cast_slice(&data))?;

    let err = match cast_ln_to_f16(&tensor) {
        Ok(_) => anyhow::bail!("bool cast should be unsupported"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("BOOL"));
    Ok(())
}

#[test]
fn iter_ln_tensors_filters_to_layernorm_gamma_names() -> Result<()> {
    let ln = [1.0_f32, 2.0];
    let dense = [3.0_f32, 4.0];
    let bytes = build_safetensors(&[
        ("model.layers.0.input_layernorm.weight", Dtype::F32, vec![2], bytemuck::cast_slice(&ln)),
        ("model.layers.0.mlp.down_proj.weight", Dtype::F32, vec![2], bytemuck::cast_slice(&dense)),
        ("model.norm.weight", Dtype::F32, vec![2], bytemuck::cast_slice(&ln)),
    ])?;

    let mut names: Vec<String> = iter_ln_tensors(&bytes)?.map(|(name, _)| name).collect();
    names.sort();

    assert_eq!(names, vec!["model.layers.0.input_layernorm.weight", "model.norm.weight"]);
    Ok(())
}

#[test]
fn iter_ln_tensors_rejects_invalid_safetensors_bytes() {
    assert!(iter_ln_tensors(b"not a safetensors file").is_err());
}

#[test]
fn read_safetensors_bytes_reads_file_contents_exactly() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("fixture.safetensors");
    let bytes = b"fixture bytes";
    std::fs::write(&path, bytes)?;

    assert_eq!(read_safetensors_bytes(&path)?, bytes);
    Ok(())
}
