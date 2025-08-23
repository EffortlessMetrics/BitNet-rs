use anyhow::{bail, Result};

pub fn rows_cols(dims: &[usize]) -> Result<(usize, usize)> {
    match dims.len() {
        0 => bail!("IQ2_S: empty tensor dims"),
        1 => Ok((1, dims[0])),
        _ => {
            let cols = *dims.last().unwrap();
            let rows = dims[..dims.len()-1].iter().copied().product();
            Ok((rows, cols))
        }
    }
}

/// Dequantize a packed IQ2_S tensor (row-major) to f32.
/// - `src_bytes`: raw packed bytes for the full tensor.
/// - `dims`: tensor dims, last dim = columns (ncols).
pub fn dequantize_to_f32(src_bytes: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
    if !bitnet_ggml_ffi::has_iq2s() {
        bail!("IQ2_S tensor found but feature `iq2s-ffi` is not enabled. Rebuild with `--features iq2s-ffi`.");
    }
    let (nrows, ncols) = rows_cols(dims)?;
    if nrows == 0 || ncols == 0 {
        bail!("IQ2_S invalid shape: {dims:?}");
    }

    // For GGUF quantized tensors, row size is constant; divide total.
    let total = src_bytes.len();
    let row_bytes = total / nrows;
    if row_bytes * nrows != total {
        bail!("IQ2_S row bytes do not divide evenly: total={} rows={}", total, nrows);
    }

    let mut out = vec![0f32; nrows * ncols];

    unsafe {
        for r in 0..nrows {
            let src_row = src_bytes.as_ptr().add(r * row_bytes) as *const std::ffi::c_void;
            let dst_row = out.as_mut_ptr().add(r * ncols);
            bitnet_ggml_ffi::dequantize_row_iq2_s(src_row, dst_row, ncols);
        }
    }
    Ok(out)
}

/// Optional: f16 materialization if your compute path prefers it
pub fn dequantize_to_f16(src_bytes: &[u8], dims: &[usize]) -> Result<Vec<half::f16>> {
    let f32s = dequantize_to_f32(src_bytes, dims)?;
    Ok(f32s.into_iter().map(half::f16::from_f32).collect())
}