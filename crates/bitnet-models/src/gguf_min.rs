//! Minimal GGUF reader: just enough to fetch tok_embeddings + output/lm_head.
//! - Supports GGUF v2/v3
//! - Tensors: f32 / f16 / I2_S -> f32
//! - Uses memmap for zero-copy reads where possible
//! - Integrates with BitNet quantization for I2_S support
//!
//! This is deliberately small: robust error messages, no full schema.

use anyhow::{bail, ensure, Context, Result};
use half::f16;
use memmap2::Mmap;
use std::{borrow::Cow, fs::File, io::{self, Read, Seek}, path::Path};
use bitnet_quantization::{I2SQuantizer, I2SLayout, QuantizedTensor};
use bitnet_common::QuantizationType;
use crate::formats::gguf::types::GgufTensorType;

#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    n_dims: u32,
    dims: Vec<u64>,
    ty: u32,      // ggml_type
    offset: u64,  // from start of data section (not file start)
}

// Helper functions for type checking
#[inline]
fn is_supported_ty(ty: u32) -> bool {
    matches!(
        GgufTensorType::from_u32(ty).ok(),
        Some(GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::I2_S)
    )
}

#[inline]
fn is_2d(dims: &[u64]) -> bool {
    dims.len() == 2
}

// Helper for shape-driven tensor selection
fn looks_like_embeddings(info: &TensorInfo) -> bool {
    is_2d(&info.dims) && 
    is_supported_ty(info.ty) &&
    (info.name.contains("emb") || info.name.contains("wte") || info.name.contains("embed"))
}

fn looks_like_output_matrix(info: &TensorInfo) -> bool {
    is_2d(&info.dims) &&
    is_supported_ty(info.ty) &&
    // de-prefer obvious non-heads
    !(info.name.contains("attn_output") || info.name.contains("blk") || info.name.contains("norm"))
}

#[derive(Debug)]
struct Parsed {
    alignment: u64,
    tensors: Vec<TensorInfo>,
    data_offset: u64, // absolute file offset where tensor data section starts
}

pub struct TwoTensors {
    /// [vocab, dim] f32
    pub tok_embeddings: Vec<f32>,
    /// [dim, vocab] f32
    pub lm_head: Vec<f32>,
    pub vocab: usize,
    pub dim: usize,
}

/// Public entry: load the two tensors or fail with a helpful error.
/// Supports f32/f16 tensors directly and I2_S quantized tensors via dequantization.
pub fn load_two<P: AsRef<Path>>(path: P) -> Result<TwoTensors> {
    let file = File::open(&path).with_context(|| format!("open {}", path.as_ref().display()))?;
    let mmap = unsafe { Mmap::map(&file) }.with_context(|| "mmap gguf file")?;
    let mut cursor = std::io::Cursor::new(&mmap[..]);

    let parsed = parse_header(&mut cursor).with_context(|| "parse GGUF header")?;
    let (tok_info, head_info) = pick_tensors(&parsed)
        .with_context(|| "locate tok_embeddings/output tensors")?;

    // Materialize as f32 (copy if needed).
    let tok = tensor_as_f32(&mmap, parsed.data_offset, &tok_info)?;
    let head = tensor_as_f32(&mmap, parsed.data_offset, &head_info)?;

    // Shapes can vary; deduce (vocab, dim) consistently.
    if tok_info.dims.len() != 2 || head_info.dims.len() != 2 {
        bail!("expected 2D tensors; got tok {:?}, head {:?}", tok_info.dims, head_info.dims);
    }

    let (ta, tb) = (tok_info.dims[0] as usize, tok_info.dims[1] as usize);
    let (ha, hb) = (head_info.dims[0] as usize, head_info.dims[1] as usize);
    
    // Check if we're using tied weights (same tensor for both)
    let using_tied_weights = tok_info.name == head_info.name;
    
    let (vocab, dim, tok_embeddings, lm_head) = if using_tied_weights {
        // BitNet models have embeddings as [dim, vocab]
        // We need: tok_embeddings [vocab, dim] and lm_head [dim, vocab]
        let (dim, vocab) = (ta, tb);
        
        // Transpose tok to get [vocab, dim]
        let mut tok_transposed = vec![0f32; vocab * dim];
        for i in 0..dim {
            for j in 0..vocab {
                tok_transposed[j * dim + i] = tok[i * vocab + j];
            }
        }
        
        // lm_head stays as [dim, vocab]
        (vocab, dim, tok_transposed, head.into_owned())
    } else if ta == hb && tb == ha {
        // Standard case: tok [vocab, dim], head [dim, vocab]
        (ta, tb, tok.into_owned(), head.into_owned())
    } else if ta == ha && tb == hb {
        // Both same shape - need to determine which is which
        // Assume tok is [vocab, dim] if vocab > dim (common case)
        let (vocab, dim) = if ta > tb { (ta, tb) } else { (tb, ta) };
        
        // May need to transpose head
        if ta > tb {
            // tok is [vocab, dim], need to transpose head from [vocab, dim] to [dim, vocab]
            let mut head_transposed = vec![0f32; dim * vocab];
            for i in 0..vocab {
                for j in 0..dim {
                    head_transposed[j * vocab + i] = head[i * dim + j];
                }
            }
            (vocab, dim, tok.into_owned(), head_transposed)
        } else {
            // tok is [dim, vocab], need to transpose it
            let mut tok_transposed = vec![0f32; vocab * dim];
            for i in 0..dim {
                for j in 0..vocab {
                    tok_transposed[j * dim + i] = tok[i * vocab + j];
                }
            }
            (vocab, dim, tok_transposed, head.into_owned())
        }
    } else {
        bail!("incompatible tensor shapes: tok={:?}, head={:?}", tok_info.dims, head_info.dims);
    };

    Ok(TwoTensors {
        tok_embeddings,
        lm_head,
        vocab,
        dim,
    })
}

// ---------- parsing ----------

fn parse_header<R: Read + Seek>(r: &mut R) -> Result<Parsed> {
    // magic "GGUF"
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file (missing magic)");
    }

    let version = read_u32(r)?;
    if version != 2 && version != 3 {
        bail!("unsupported GGUF version {version} (only v2/v3)");
    }

    let n_tensors = read_u64(r)?;
    let n_kv = read_u64(r)?;

    // default alignment if kv not present (llama.cpp typically writes 32)
    let mut alignment: u64 = 32;

    // read kv and capture alignment if present
    for _ in 0..n_kv {
        let key = read_string(r)?;
        let ty = read_u32(r)?;
        // We only care about general.alignment (uint64). Skip others.
        if key == "general.alignment" {
            alignment = read_gguf_value_uint64(r, ty)
                .with_context(|| "parse general.alignment")?;
        } else {
            skip_gguf_value(r, ty)?;
        }
    }

    // tensor infos
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = read_string(r)?;
        let n_dims = read_u32(r)?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(r)?);
        }
        let ty = read_u32(r)?;
        let offset = read_u64(r)?;
        tensors.push(TensorInfo { name, n_dims, dims, ty, offset });
    }

    // data section begins aligned after the tensor header table
    let here = r.stream_position()?;
    let data_offset = align_up(here, alignment);

    Ok(Parsed { alignment, tensors, data_offset })
}

fn pick_tensors(parsed: &Parsed) -> Result<(TensorInfo, TensorInfo)> {
    // Common names across ecosystems:
    // embeddings
    const TOK_NAMES: &[&str] = &[
        "tok_embeddings.weight",
        "token_embd.weight",
        "model.embed_tokens.weight",
        "transformer.wte.weight",
    ];
    // output head
    const HEAD_NAMES: &[&str] = &[
        "output.weight",
        "lm_head.weight",
        "model.lm_head.weight",
        "transformer.lm_head.weight",
    ];

    let find = |names: &[&str]| {
        parsed.tensors.iter()
            .find(|t| names.iter().any(|n| t.name == *n) && is_supported_ty(t.ty))
            .cloned()
    };

    // Find embeddings tensor - prefer by name first, then shape
    let tok = find(TOK_NAMES)
        .or_else(|| parsed.tensors.iter().find(|t| looks_like_embeddings(t)).cloned())
        .ok_or_else(|| anyhow::anyhow!("could not find token embeddings tensor"))?;

    // Find output tensor - shape-driven with name hints
    let head = find(HEAD_NAMES)
        .or_else(|| parsed.tensors.iter()
            .find(|t| looks_like_output_matrix(t) && 
                  (t.name.contains("lm_head") || t.name.contains("output")))
            .cloned())
        // If no output head found, BitNet models often use tied embeddings
        .unwrap_or_else(|| tok.clone());

    Ok((tok, head))
}

// ---------- tensor materialization ----------

fn tensor_as_f32<'a>(mmap: &'a [u8], data_base: u64, info: &TensorInfo) -> Result<Cow<'a, [f32]>> {
    let nelems: usize = info.dims.iter().try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| anyhow::anyhow!("tensor size overflow"))?
        .try_into()
        .map_err(|_| anyhow::anyhow!("tensor too large"))?;
    let offset = (data_base + info.offset) as usize;

    match info.ty {
        0 => { // f32
            let need = nelems * 4;
            if offset + need > mmap.len() { bail!("f32 tensor out of bounds"); }
            // Safety: GGUF stores little-endian f32; copy to owned Vec to be safe/aligned.
            let mut out = vec![0f32; nelems];
            let bytes = &mmap[offset .. offset + need];
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            Ok(Cow::Owned(out))
        }
        1 => { // f16
            let need = nelems * 2;
            if offset + need > mmap.len() { bail!("f16 tensor out of bounds"); }
            let mut out = vec![0f32; nelems];
            let bytes = &mmap[offset .. offset + need];
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out[i] = f16::from_bits(bits).to_f32();
            }
            Ok(Cow::Owned(out))
        }
        36 => { // I2_S - BitNet 2-bit quantization
            let layout = I2SLayout::default();
            let num_blocks = (nelems + layout.block_size - 1) / layout.block_size;
            let need = num_blocks * layout.bytes_per_block;
            
            ensure!(offset + need <= mmap.len(), "I2_S tensor out of bounds");
            
            // Verify shape/blocks consistency
            ensure!(
                num_blocks * layout.block_size >= nelems && 
                num_blocks * layout.block_size - nelems < layout.block_size,
                "I2_S blocks/shape mismatch: nelems={nelems} blocks={num_blocks} block_size={}",
                layout.block_size
            );
            
            // Extract quantized data and scales
            let tensor_bytes = &mmap[offset .. offset + need];
            let mut scales = Vec::with_capacity(num_blocks);
            let mut packed_data = Vec::with_capacity(num_blocks * layout.data_bytes_per_block);
            
            for block_idx in 0..num_blocks {
                let block_offset = block_idx * layout.bytes_per_block;
                // Copy packed data
                packed_data.extend_from_slice(
                    &tensor_bytes[block_offset..block_offset + layout.data_bytes_per_block]
                );
                // Read scale (f16)
                let scale_start = block_offset + layout.data_bytes_per_block;
                let scale_end = scale_start + layout.scale_bytes_per_block;
                let scale_bytes = &tensor_bytes[scale_start..scale_end];
                let scale_bits = u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]);
                scales.push(f16::from_bits(scale_bits).to_f32());
            }
            
            // Create quantized tensor
            let quantized = QuantizedTensor::new_with_params(
                packed_data,
                scales,
                None,
                info.dims.iter().map(|&d| d as usize).collect(),
                QuantizationType::I2S,
                block_size,
            );
            
            // Dequantize using our existing infrastructure
            let quantizer = I2SQuantizer::with_block_size(block_size);
            let tensor = quantizer.dequantize_tensor(&quantized)
                .with_context(|| format!("Failed to dequantize I2_S tensor {}", info.name))?;
            
            // Extract f32 data from BitNetTensor
            let data = tensor.to_vec()
                .with_context(|| "Failed to extract tensor data")?;
            
            Ok(Cow::Owned(data))
        }
        other => bail!("unsupported ggml tensor type {other} (only f32/f16/I2_S supported)"),
    }
}

// ---------- helpers ----------

fn align_up(x: u64, a: u64) -> u64 { ((x + a - 1) / a) * a }

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf).context("string not utf8")?)
}

// Minimal KV skipping (we only read general.alignment if present)
fn read_gguf_value_uint64<R: Read>(r: &mut R, ty: u32) -> Result<u64> {
    match ty {
        12 /* uint64 */ => read_u64(r),
        _ => {
            // Type mismatch — skip value and return default alignment
            skip_gguf_value(r, ty)?;
            Ok(32)
        }
    }
}

// Allocation-free skip helper
#[inline]
fn skip_n<R: Read>(r: &mut R, n: u64) -> Result<()> {
    let copied = io::copy(&mut r.by_ref().take(n), &mut io::sink())?;
    ensure!(copied == n, "unexpected EOF while skipping {n} bytes");
    Ok(())
}

fn skip_gguf_value<R: Read>(r: &mut R, ty: u32) -> Result<()> {
    // GGUF scalar sizes (see llama.cpp)
    match ty {
        0 | 1 => skip_n(r, 1)?,                  // uint8 | int8
        2 | 3 => skip_n(r, 2)?,                  // uint16 | int16
        4 | 5 | 6 => skip_n(r, 4)?,              // uint32 | int32 | float32
        7 => skip_n(r, 1)?,                      // bool
        8 => {                                   // string: u64 len + bytes
            let n = read_u64(r)?;
            skip_n(r, n)?;
        }
        9 => {                                   // array: elem_ty + count + values
            let elem_ty = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_gguf_value(r, elem_ty)?;
            }
        }
        10 | 11 | 12 => skip_n(r, 8)?,           // uint64 | int64 | float64
        _ => bail!("unknown GGUF kv type id {ty}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore] // set BITNET_GGUF to a real path to run
    fn loads_two_tensors() {
        let p = std::env::var_os("BITNET_GGUF").expect("set BITNET_GGUF");
        let two = load_two(p).unwrap();
        assert!(two.vocab > 0 && two.dim > 0);
        assert_eq!(two.tok_embeddings.len(), two.vocab * two.dim);
        assert_eq!(two.lm_head.len(), two.dim * two.vocab);
    }
}