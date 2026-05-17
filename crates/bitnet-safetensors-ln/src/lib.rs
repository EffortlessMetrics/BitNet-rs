//! Shared `LayerNorm` helpers for SafeTensors-based tools.
//!
//! This microcrate centralizes tensor-name filtering (`is_ln_gamma`),
//! `LayerNorm` tensor iteration, RMS computation, and casting LN gamma tensors
//! to f16.

use anyhow::{Result, anyhow};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use std::path::Path;

pub use bitnet_validation::is_ln_gamma;

/// Read a `SafeTensors` file fully into memory.
pub fn read_safetensors_bytes(path: &Path) -> Result<Vec<u8>> {
    Ok(std::fs::read(path)?)
}

/// Iterate LN tensors (name, tensor) from a loaded `SafeTensors` buffer.
pub fn iter_ln_tensors(
    buf: &[u8],
) -> Result<impl Iterator<Item = (String, safetensors::tensor::TensorView<'_>)>> {
    let st = SafeTensors::deserialize(buf)?;
    Ok(st.tensors().into_iter().filter(|(name, _)| is_ln_gamma(name)))
}

/// Compute RMS for the given raw tensor view (sqrt(mean(x^2))).
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
pub fn rms_for_tensor(t: &safetensors::tensor::TensorView<'_>) -> Result<f64> {
    let dtype = t.dtype();
    let shape = t.shape();
    let data = t.data();

    let n: usize = shape.iter().product::<usize>();
    if n == 0 {
        return Ok(0.0);
    }

    let rms = match dtype {
        Dtype::F16 => {
            let halves: &[u16] =
                bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad f16 buffer size"))?;
            let mut acc = 0.0f64;
            for &bits in halves.iter().take(n) {
                let v = f64::from(f16::from_bits(bits).to_f32());
                acc += v * v;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::BF16 => {
            let halves: &[u16] =
                bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad bf16 buffer size"))?;
            let mut acc = 0.0f64;
            for &bits in halves.iter().take(n) {
                let v = f64::from(bf16::from_bits(bits).to_f32());
                acc += v * v;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::F32 => {
            let xs: &[f32] =
                bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad f32 buffer size"))?;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = f64::from(v);
                acc += d * d;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::F64 => {
            let xs: &[f64] =
                bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad f64 buffer size"))?;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                acc += v * v;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::I8 => {
            let xs: &[i8] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad i8 size"))?;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = f64::from(v);
                acc += d * d;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::U8 => {
            let xs: &[u8] = data;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = f64::from(v);
                acc += d * d;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::I16 => {
            let xs: &[i16] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad i16 size"))?;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = f64::from(v);
                acc += d * d;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::U16 => {
            let xs: &[u16] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad u16 size"))?;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = f64::from(v);
                acc += d * d;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::I32 => {
            let xs: &[i32] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad i32 size"))?;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = f64::from(v);
                acc += d * d;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::U32 => {
            let xs: &[u32] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad u32 size"))?;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = f64::from(v);
                acc += d * d;
            }
            (acc / (n as f64)).sqrt()
        }
        _ => return Err(anyhow!("unsupported dtype for RMS: {dtype:?}")),
    };

    Ok(rms)
}

/// Cast LN gamma bytes to f16.
///
/// Returns owned little-endian f16 bytes for supported numeric dtypes, or an
/// error for dtypes that do not have a scalar numeric conversion here.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn cast_ln_to_f16(t: &safetensors::tensor::TensorView<'_>) -> Result<Vec<u8>> {
    let dtype = t.dtype();
    let shape = t.shape();
    let n: usize = shape.iter().product();
    let data = t.data();

    let mut out: Vec<u16> = Vec::with_capacity(n);

    match dtype {
        Dtype::F16 => return Ok(data.to_vec()),
        Dtype::F32 => {
            let xs: &[f32] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad f32 size"))?;
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(v).to_bits()));
        }
        Dtype::F64 => {
            let xs: &[f64] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad f64 size"))?;
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(v as f32).to_bits()));
        }
        Dtype::BF16 => {
            let xs: &[u16] =
                bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad bf16 size"))?;
            out.extend(xs.iter().take(n).map(|&b| {
                let v = bf16::from_bits(b).to_f32();
                f16::from_f32(v).to_bits()
            }));
        }
        Dtype::I8 => {
            let xs: &[i8] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad i8"))?;
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(f32::from(v)).to_bits()));
        }
        Dtype::U8 => {
            let xs = data;
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(f32::from(v)).to_bits()));
        }
        Dtype::I16 => {
            let xs: &[i16] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad i16"))?;
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(f32::from(v)).to_bits()));
        }
        Dtype::U16 => {
            let xs: &[u16] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad u16"))?;
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(f32::from(v)).to_bits()));
        }
        Dtype::I32 => {
            let xs: &[i32] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad i32"))?;
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(v as f32).to_bits()));
        }
        Dtype::U32 => {
            let xs: &[u32] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad u32"))?;
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(v as f32).to_bits()));
        }
        _ => return Err(anyhow!("unsupported dtype for LN cast: {dtype:?}")),
    }

    let mut bytes = Vec::with_capacity(out.len() * std::mem::size_of::<u16>());
    for bits in out {
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use std::collections::HashMap;

    fn build_f32_tensor(name: &str, values: &[f32]) -> Result<Vec<u8>> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, vec![values.len()], &bytes)?;
        let mut map: HashMap<String, TensorView<'_>> = HashMap::new();
        map.insert(name.to_string(), view);
        Ok(safetensors::serialize(map, None)?)
    }

    fn extract_view<'a>(buf: &'a [u8], name: &str) -> Result<TensorView<'a>> {
        let st = SafeTensors::deserialize(buf)?;
        Ok(st.tensor(name)?)
    }

    fn f16_values(bytes: &[u8]) -> Result<Vec<f32>> {
        let mut values = Vec::with_capacity(bytes.len() / std::mem::size_of::<u16>());
        for chunk in bytes.chunks_exact(std::mem::size_of::<u16>()) {
            let pair: [u8; 2] = chunk.try_into()?;
            values.push(f16::from_bits(u16::from_le_bytes(pair)).to_f32());
        }
        Ok(values)
    }

    // -------- iter_ln_tensors --------

    #[test]
    fn iter_ln_tensors_returns_only_ln_gamma_names() -> Result<()> {
        // Build a buffer with mixed tensors: one LN, one non-LN.
        let ln_bytes: Vec<u8> =
            [1.0f32, 2.0, 3.0].iter().flat_map(|v: &f32| v.to_le_bytes()).collect();
        let other_bytes: Vec<u8> =
            [4.0f32, 5.0].iter().flat_map(|v: &f32| v.to_le_bytes()).collect();

        let ln_view = TensorView::new(Dtype::F32, vec![3], &ln_bytes)?;
        let other_view = TensorView::new(Dtype::F32, vec![2], &other_bytes)?;

        let mut map: HashMap<String, TensorView<'_>> = HashMap::new();
        map.insert("blk.0.attn_norm.weight".to_string(), ln_view);
        map.insert("blk.0.attn_q.weight".to_string(), other_view);
        let buf = safetensors::serialize(map, None)?;

        let mut names: Vec<String> = iter_ln_tensors(&buf)?.map(|(n, _)| n).collect();
        names.sort();
        assert_eq!(names, vec!["blk.0.attn_norm.weight".to_string()]);
        Ok(())
    }

    #[test]
    fn iter_ln_tensors_returns_empty_when_no_ln_names() -> Result<()> {
        let buf = build_f32_tensor("blk.0.attn_q.weight", &[1.0, 2.0])?;
        let count = iter_ln_tensors(&buf)?.count();
        assert_eq!(count, 0);
        Ok(())
    }

    #[test]
    fn iter_ln_tensors_fails_on_invalid_buffer() {
        assert!(iter_ln_tensors(b"not a safetensors file").is_err());
    }

    // -------- rms_for_tensor --------

    #[test]
    fn rms_f32_matches_expected_value() -> Result<()> {
        let buf = build_f32_tensor("blk.0.attn_norm.weight", &[1.0, 2.0, 3.0])?;
        let view = extract_view(&buf, "blk.0.attn_norm.weight")?;
        let rms = rms_for_tensor(&view)?;
        // sqrt((1 + 4 + 9) / 3) = sqrt(14/3) ~= 2.16024689947
        assert!((rms - (14.0_f64 / 3.0).sqrt()).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn rms_returns_zero_for_empty_shape() -> Result<()> {
        let bytes: Vec<u8> = Vec::new();
        let view = TensorView::new(Dtype::F32, vec![0], &bytes)?;
        assert_eq!(rms_for_tensor(&view)?, 0.0);
        Ok(())
    }

    #[test]
    fn rms_f16_round_trips_through_conversion() -> Result<()> {
        let halves: Vec<u16> = [f16::from_f32(2.0), f16::from_f32(2.0), f16::from_f32(2.0)]
            .iter()
            .map(|h| h.to_bits())
            .collect();
        let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F16, vec![3], &bytes)?;
        let rms = rms_for_tensor(&view)?;
        // sqrt((4+4+4)/3) = 2.0
        assert!((rms - 2.0).abs() < 1e-3);
        Ok(())
    }

    #[test]
    fn rms_bf16_round_trips_through_conversion() -> Result<()> {
        let halves: Vec<u16> = [bf16::from_f32(3.0), bf16::from_f32(3.0), bf16::from_f32(3.0)]
            .iter()
            .map(|h| h.to_bits())
            .collect();
        let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::BF16, vec![3], &bytes)?;
        let rms = rms_for_tensor(&view)?;
        assert!((rms - 3.0).abs() < 1e-2);
        Ok(())
    }

    #[test]
    fn rms_f64_identity_for_zeros() -> Result<()> {
        let bytes: Vec<u8> =
            [0.0f64, 0.0, 0.0].iter().flat_map(|v: &f64| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F64, vec![3], &bytes)?;
        assert_eq!(rms_for_tensor(&view)?, 0.0);
        Ok(())
    }

    #[test]
    fn rms_i8_handles_negative_values() -> Result<()> {
        let bytes: Vec<u8> = vec![3i8 as u8, (-4i8) as u8, 0u8];
        let view = TensorView::new(Dtype::I8, vec![3], &bytes)?;
        let rms = rms_for_tensor(&view)?;
        // sqrt((9+16+0)/3) = sqrt(25/3) ~= 2.886751
        assert!((rms - (25.0_f64 / 3.0).sqrt()).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn rms_u8_uses_byte_values_directly() -> Result<()> {
        let bytes: Vec<u8> = vec![1, 2, 2];
        let view = TensorView::new(Dtype::U8, vec![3], &bytes)?;
        let rms = rms_for_tensor(&view)?;
        // sqrt((1+4+4)/3) = sqrt(3) ~= 1.7320508
        assert!((rms - 3.0_f64.sqrt()).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn rms_i16_range() -> Result<()> {
        let xs: [i16; 3] = [10, -10, 0];
        let bytes: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::I16, vec![3], &bytes)?;
        let rms = rms_for_tensor(&view)?;
        // sqrt((100+100+0)/3) = sqrt(200/3) ~= 8.165
        assert!((rms - (200.0_f64 / 3.0).sqrt()).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn rms_u16_range() -> Result<()> {
        let xs: [u16; 2] = [1, 1];
        let bytes: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::U16, vec![2], &bytes)?;
        assert!((rms_for_tensor(&view)? - 1.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn rms_i32_range() -> Result<()> {
        let xs: [i32; 2] = [3, 4];
        let bytes: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::I32, vec![2], &bytes)?;
        let rms = rms_for_tensor(&view)?;
        // sqrt((9+16)/2) = sqrt(12.5)
        assert!((rms - 12.5_f64.sqrt()).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn rms_u32_range() -> Result<()> {
        let xs: [u32; 1] = [5];
        let bytes: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::U32, vec![1], &bytes)?;
        assert!((rms_for_tensor(&view)? - 5.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn rms_unsupported_dtype_returns_error() -> Result<()> {
        let bytes: Vec<u8> = vec![1];
        let view = TensorView::new(Dtype::BOOL, vec![1], &bytes)?;
        let result = rms_for_tensor(&view);
        assert!(result.is_err());
        Ok(())
    }

    // -------- cast_ln_to_f16 --------
    //
    #[test]
    fn cast_f16_returns_input_passthrough() -> Result<()> {
        let halves: Vec<u16> =
            [1.0f32, 2.0, 3.0].iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F16, vec![3], &bytes)?;
        let out = cast_ln_to_f16(&view)?;
        assert_eq!(out, bytes);
        Ok(())
    }

    #[test]
    fn cast_f16_empty_passthrough_is_empty() -> Result<()> {
        let bytes: &[u8] = &[];
        let view = TensorView::new(Dtype::F16, vec![0], bytes)?;
        let out = cast_ln_to_f16(&view)?;
        assert!(out.is_empty());
        Ok(())
    }

    #[test]
    fn cast_f32_to_f16_converts_values() -> Result<()> {
        let bytes: Vec<u8> = [1.5f32, 2.5].iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, vec![2], &bytes)?;
        assert_eq!(f16_values(&cast_ln_to_f16(&view)?)?, vec![1.5, 2.5]);
        Ok(())
    }

    #[test]
    fn cast_f64_to_f16_converts_values() -> Result<()> {
        let bytes: Vec<u8> = [1.0f64].iter().flat_map(|v: &f64| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F64, vec![1], &bytes)?;
        assert_eq!(f16_values(&cast_ln_to_f16(&view)?)?, vec![1.0]);
        Ok(())
    }

    #[test]
    fn cast_bf16_to_f16_converts_values() -> Result<()> {
        let halves: Vec<u16> = vec![bf16::from_f32(1.0).to_bits()];
        let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::BF16, vec![1], &bytes)?;
        assert_eq!(f16_values(&cast_ln_to_f16(&view)?)?, vec![1.0]);
        Ok(())
    }

    #[test]
    fn cast_i8_to_f16_converts_values() -> Result<()> {
        let bytes: Vec<u8> = vec![(-1i8) as u8, 127u8];
        let view = TensorView::new(Dtype::I8, vec![2], &bytes)?;
        assert_eq!(f16_values(&cast_ln_to_f16(&view)?)?, vec![-1.0, 127.0]);
        Ok(())
    }

    #[test]
    fn cast_u8_to_f16_converts_values() -> Result<()> {
        let bytes: Vec<u8> = vec![0u8, 128, 255];
        let view = TensorView::new(Dtype::U8, vec![3], &bytes)?;
        assert_eq!(f16_values(&cast_ln_to_f16(&view)?)?, vec![0.0, 128.0, 255.0]);
        Ok(())
    }

    #[test]
    fn cast_i16_to_f16_converts_values() -> Result<()> {
        let bytes: Vec<u8> = [1i16, -1].iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::I16, vec![2], &bytes)?;
        assert_eq!(f16_values(&cast_ln_to_f16(&view)?)?, vec![1.0, -1.0]);
        Ok(())
    }

    #[test]
    fn cast_u16_to_f16_converts_values() -> Result<()> {
        let bytes: Vec<u8> = [1u16, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::U16, vec![2], &bytes)?;
        assert_eq!(f16_values(&cast_ln_to_f16(&view)?)?, vec![1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn cast_i32_to_f16_converts_values() -> Result<()> {
        let bytes: Vec<u8> = [1i32, -1].iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::I32, vec![2], &bytes)?;
        assert_eq!(f16_values(&cast_ln_to_f16(&view)?)?, vec![1.0, -1.0]);
        Ok(())
    }

    #[test]
    fn cast_u32_to_f16_converts_values() -> Result<()> {
        let bytes: Vec<u8> = [1u32, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::U32, vec![2], &bytes)?;
        assert_eq!(f16_values(&cast_ln_to_f16(&view)?)?, vec![1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn cast_unsupported_dtype_returns_error() -> Result<()> {
        let bytes: Vec<u8> = vec![1];
        let view = TensorView::new(Dtype::BOOL, vec![1], &bytes)?;
        let result = cast_ln_to_f16(&view);
        assert!(result.is_err());
        Ok(())
    }

    // -------- read_safetensors_bytes --------

    #[test]
    fn read_safetensors_bytes_round_trips_via_filesystem() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("tensor.safetensors");
        let buf = build_f32_tensor("blk.0.attn_norm.weight", &[1.0, 2.0])?;
        std::fs::write(&path, &buf)?;
        let read = read_safetensors_bytes(&path)?;
        assert_eq!(read, buf);
        Ok(())
    }

    #[test]
    fn read_safetensors_bytes_errors_for_missing_file() {
        let result = read_safetensors_bytes(std::path::Path::new(
            "/no/such/file_that_should_not_exist.safetensors",
        ));
        assert!(result.is_err());
    }
}
