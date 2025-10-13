//! Common utilities for SafeTensors LayerNorm tools.
//!
//! Functions are used across multiple binaries, hence the allow(dead_code).

#![allow(dead_code)]

use anyhow::{Result, anyhow};
use half::{bf16, f16};
use regex::Regex;
use safetensors::{Dtype, SafeTensors};
use std::path::Path;

/// Match *norm.weight variants we care about:
///  - attn_norm, ffn_norm, ffn_layernorm, rms_norm
///  - input_layernorm, post_attention_layernorm
///  - final_layernorm, final_norm
///
/// (strict suffix ".weight")
pub fn is_ln_gamma(name: &str) -> bool {
    // cheap fast-path first
    if !name.ends_with(".weight") {
        return false;
    }
    // precise patterns
    lazy_static::lazy_static! {
        static ref RE: Regex = Regex::new(
            r#"(?x)
            (?:^|[./])
            (?:attn_norm|ffn_norm|ffn_layernorm|rms_norm|
               input_layernorm|post_attention_layernorm|
               final_layernorm|final_norm|norm)
            \.weight$
            "#
        )
        .unwrap();
    }
    RE.is_match(name)
}

/// Read a SafeTensors file fully into memory.
pub fn read_safetensors_bytes(path: &Path) -> Result<Vec<u8>> {
    Ok(std::fs::read(path)?)
}

/// Iterate LN tensors (name, tensor) from a loaded SafeTensors buffer.
pub fn iter_ln_tensors(
    buf: &[u8],
) -> Result<impl Iterator<Item = (String, safetensors::tensor::TensorView<'_>)>> {
    let st = SafeTensors::deserialize(buf)?;
    Ok(st.tensors().into_iter().filter(|(name, _)| is_ln_gamma(name)))
}

/// Compute RMS for the given raw tensor view (sqrt(mean(x^2))).
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
                let v = f16::from_bits(bits).to_f32() as f64;
                acc += v * v;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::BF16 => {
            let halves: &[u16] =
                bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad bf16 buffer size"))?;
            let mut acc = 0.0f64;
            for &bits in halves.iter().take(n) {
                let v = bf16::from_bits(bits).to_f32() as f64;
                acc += v * v;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::F32 => {
            let xs: &[f32] =
                bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad f32 buffer size"))?;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = v as f64;
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
        // accept integer norms too (rare), convert to f64
        Dtype::I8 => {
            let xs: &[i8] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad i8 size"))?;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = v as f64;
                acc += d * d;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::U8 => {
            let xs: &[u8] = data;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = v as f64;
                acc += d * d;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::I16 => {
            let xs: &[i16] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad i16 size"))?;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = v as f64;
                acc += d * d;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::U16 => {
            let xs: &[u16] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad u16 size"))?;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = v as f64;
                acc += d * d;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::I32 => {
            let xs: &[i32] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad i32 size"))?;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = v as f64;
                acc += d * d;
            }
            (acc / (n as f64)).sqrt()
        }
        Dtype::U32 => {
            let xs: &[u32] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad u32 size"))?;
            let mut acc = 0.0f64;
            for &v in xs.iter().take(n) {
                let d = v as f64;
                acc += d * d;
            }
            (acc / (n as f64)).sqrt()
        }
        _ => return Err(anyhow!("unsupported dtype for RMS: {:?}", dtype)),
    };

    Ok(rms)
}

/// Cast LN gamma bytes to f16 (returns owned Vec<u8> with f16 encoding).
pub fn cast_ln_to_f16(t: &safetensors::tensor::TensorView<'_>) -> Result<Vec<u8>> {
    let dtype = t.dtype();
    let shape = t.shape();
    let n: usize = shape.iter().product();
    let data = t.data();

    let mut out: Vec<u16> = Vec::with_capacity(n);

    match dtype {
        Dtype::F16 => {
            // already f16; just copy bytes
            return Ok(data.to_vec());
        }
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
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(v as f32).to_bits()));
        }
        Dtype::U8 => {
            let xs = data;
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(v as f32).to_bits()));
        }
        Dtype::I16 => {
            let xs: &[i16] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad i16"))?;
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(v as f32).to_bits()));
        }
        Dtype::U16 => {
            let xs: &[u16] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad u16"))?;
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(v as f32).to_bits()));
        }
        Dtype::I32 => {
            let xs: &[i32] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad i32"))?;
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(v as f32).to_bits()));
        }
        Dtype::U32 => {
            let xs: &[u32] = bytemuck::try_cast_slice(data).map_err(|_| anyhow!("bad u32"))?;
            out.extend(xs.iter().take(n).map(|&v| f16::from_f32(v as f32).to_bits()));
        }
        _ => return Err(anyhow!("unsupported dtype for LN cast: {:?}", dtype)),
    }

    Ok(bytemuck::cast_vec(out))
}
