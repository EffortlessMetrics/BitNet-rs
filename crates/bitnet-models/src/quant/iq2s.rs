use crate::quant::backend::Iq2sBackend;
use anyhow::{Result, bail};
use std::sync::Once;

static BACKEND_BANNER: Once = Once::new();

fn banner_once_iq2s_backend() {
    BACKEND_BANNER.call_once(|| {
        if std::env::var_os("BITNET_QUIET_BACKEND").is_none() {
            let backend = Iq2sBackend::selected();
            #[cfg(feature = "iq2s-ffi")]
            if backend == Iq2sBackend::Ffi {
                let qk = backend.qk();
                let bs = backend.block_bytes();
                let commit = bitnet_ggml_ffi::GGML_COMMIT;
                let short = commit.get(..commit.len().min(12)).unwrap_or(commit);
                tracing::info!("IQ2_S backend: ffi (ggml {}, qk={}, block={}B)", short, qk, bs);
            } else {
                tracing::info!("IQ2_S backend: rust");
            }
            #[cfg(not(feature = "iq2s-ffi"))]
            {
                tracing::info!("IQ2_S backend: rust (compiled w/o FFI)");
            }
        }
    });
}

pub fn rows_cols(dims: &[usize]) -> Result<(usize, usize)> {
    match dims.len() {
        0 => bail!("IQ2_S: empty tensor dims"),
        1 => Ok((1, dims[0])),
        _ => {
            let cols = *dims.last().unwrap();
            let rows = dims[..dims.len() - 1].iter().copied().product();
            Ok((rows, cols))
        }
    }
}

/// Dequantize a packed IQ2_S tensor (row-major) to f32.
/// - `src_bytes`: raw packed bytes for the full tensor.
/// - `dims`: tensor dims, last dim = columns (ncols).
pub fn dequantize_to_f32(src_bytes: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
    let backend = Iq2sBackend::selected();

    // Only bail when the chosen backend is FFI but the feature isn't compiled.
    #[cfg(feature = "iq2s-ffi")]
    if backend == Iq2sBackend::Ffi && !bitnet_ggml_ffi::has_iq2s() {
        bail!(
            "IQ2_S tensor found but feature `iq2s-ffi` is not enabled. Rebuild with `--features iq2s-ffi`."
        );
    }
    #[cfg(not(feature = "iq2s-ffi"))]
    if backend == Iq2sBackend::Ffi {
        bail!(
            "IQ2_S tensor found but feature `iq2s-ffi` is not enabled. Rebuild with `--features iq2s-ffi`."
        );
    }

    banner_once_iq2s_backend();
    let (nrows, ncols) = rows_cols(dims)?;
    if nrows == 0 || ncols == 0 {
        bail!("IQ2_S invalid shape: {dims:?}");
    }

    // Get constants from backend
    let backend = Iq2sBackend::selected();
    let qk = backend.qk();
    let block_bytes = backend.block_bytes();
    #[cfg(feature = "iq2s-ffi")]
    let requires_qk_multiple = bitnet_ggml_ffi::iq2s_requires_qk_multiple();
    #[cfg(not(feature = "iq2s-ffi"))]
    let requires_qk_multiple = false; // Pure Rust won't require QK multiple

    // Calculate expected size
    let blocks_per_row = ncols.div_ceil(qk);
    let expected_row_bytes = blocks_per_row * block_bytes;
    let expected_total = nrows * expected_row_bytes;

    if src_bytes.len() != expected_total {
        bail!(
            "IQ2_S byte length mismatch: got {} expected {} (rows={} cols={} qk={} block_bytes={})",
            src_bytes.len(),
            expected_total,
            nrows,
            ncols,
            qk,
            block_bytes
        );
    }

    let mut out = vec![0f32; nrows * ncols];

    // Create scratch buffer for tail safety if GGML requires QK-multiple
    let mut scratch = if requires_qk_multiple { vec![0f32; qk] } else { vec![] };

    unsafe {
        for r in 0..nrows {
            let mut src_offset = r * expected_row_bytes;
            let dst_base = r * ncols;
            let mut col = 0;

            while col < ncols {
                let remaining = ncols - col;
                let src_ptr = src_bytes.as_ptr().add(src_offset) as *const std::ffi::c_void;

                if remaining >= qk || !requires_qk_multiple {
                    // Full block or backend can handle partial blocks
                    let n = remaining.min(qk);
                    let dst_ptr = out.as_mut_ptr().add(dst_base + col);
                    backend.dequantize_row(src_ptr, dst_ptr, n);
                    col += n;
                } else {
                    // Partial block and backend requires QK multiple - use scratch
                    backend.dequantize_row(src_ptr, scratch.as_mut_ptr(), qk);
                    // Copy only the needed elements
                    let dst_slice = &mut out[dst_base + col..dst_base + col + remaining];
                    dst_slice.copy_from_slice(&scratch[..remaining]);
                    col += remaining;
                }

                src_offset += block_bytes;
            }
        }
    }
    Ok(out)
}

/// Optional: f16 materialization if your compute path prefers it
pub fn dequantize_to_f16(src_bytes: &[u8], dims: &[usize]) -> Result<Vec<half::f16>> {
    let f32s = dequantize_to_f32(src_bytes, dims)?;
    Ok(f32s.into_iter().map(half::f16::from_f32).collect())
}
