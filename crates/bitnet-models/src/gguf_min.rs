//! Minimal GGUF reader: just enough to fetch tok_embeddings + output/lm_head.
//! - Supports GGUF v2/v3
//! - Tensors: f32 / f16 -> f32
//! - Uses memmap for zero-copy reads where possible
//!
//! This is deliberately small: robust error messages, no full schema.

use anyhow::{bail, Context, Result};
use half::f16;
use memmap2::Mmap;
use std::{borrow::Cow, fs::File, io::{Read, Seek}, path::Path};

#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    n_dims: u32,
    dims: Vec<u64>,
    ty: u32,      // ggml_type (0=f32, 1=f16, others unsupported here)
    offset: u64,  // from start of data section (not file start)
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

    // Common case:
    //   tok:  [vocab, dim]
    //   head: [dim, vocab]
    let (vocab, dim, tok_is_vd) = if ta == hb && tb == ha {
        (ta, tb, true)
    } else if tb == hb && ta == ha {
        // or tok might be [dim, vocab] (rare) and head [dim, vocab]
        // prefer the consistent pairing where one is transposed of the other
        // If both are the same shape, we try to infer by names below.
        bail!("ambiguous/unsupported tensor shapes: tok={:?}, head={:?}", tok_info.dims, head_info.dims);
    } else {
        // handle alternate naming/layout: try to infer
        // If tok dims [vocab, dim] and head [vocab, dim] -> unsupported without transpose
        bail!("mismatched tensor shapes: tok={:?}, head={:?}", tok_info.dims, head_info.dims);
    };

    // Ensure we have row-major contiguous data in the expected layout.
    // tok_embeddings: [vocab, dim]
    let tok_embeddings = if tok_is_vd {
        tok.into_owned()
    } else {
        bail!("unexpected tok layout; add a transpose if you hit this path");
    };

    // lm_head: [dim, vocab] expected — already in that order from check above.
    let lm_head = head.into_owned();

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
        parsed.tensors.iter().find(|t| names.iter().any(|n| t.name == *n))
            .cloned()
    };

    let tok = find(TOK_NAMES)
        .or_else(|| parsed.tensors.iter().find(|t| t.name.contains("emb")) .cloned())
        .ok_or_else(|| anyhow::anyhow!("could not find token embeddings tensor"))?;

    let head = find(HEAD_NAMES)
        .or_else(|| parsed.tensors.iter().find(|t| t.name.contains("lm_head") || t.name.contains("output")) .cloned())
        .ok_or_else(|| anyhow::anyhow!("could not find output/lm_head tensor"))?;

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
        other => bail!("unsupported ggml tensor type {other} (only f32/f16 supported here)"),
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

fn skip_gguf_value<R: Read>(r: &mut R, ty: u32) -> Result<()> {
    // GGUF types are documented in llama.cpp; we only need to skip.
    match ty {
        0  /* uint8  */ => { let mut buf = [0u8; 1]; r.read_exact(&mut buf)?; }
        1  /* int8   */ => { let mut buf = [0u8; 1]; r.read_exact(&mut buf)?; }
        2  /* uint16 */ => { let mut buf = [0u8; 2]; r.read_exact(&mut buf)?; }
        3  /* int16  */ => { let mut buf = [0u8; 2]; r.read_exact(&mut buf)?; }
        4  /* uint32 */ => { let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; }
        5  /* int32  */ => { let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; }
        6  /* float32*/ => { let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; }
        7  /* bool   */ => { let mut buf = [0u8; 1]; r.read_exact(&mut buf)?; }
        8  /* string */ => { 
            let n = read_u64(r)? as usize; 
            let mut buf = vec![0u8; n];
            r.read_exact(&mut buf)?;
        }
        9  /* array  */ => {
            let elem_ty = read_u32(r)?;
            let count = read_u64(r)? as usize;
            for _ in 0..count { skip_gguf_value(r, elem_ty)?; }
        }
        10 /* uint64 */ => { let mut buf = [0u8; 8]; r.read_exact(&mut buf)?; }
        11 /* int64  */ => { let mut buf = [0u8; 8]; r.read_exact(&mut buf)?; }
        12 /* float64*/ => { let mut buf = [0u8; 8]; r.read_exact(&mut buf)?; }
        _ => bail!("unknown GGUF kv type id {}", ty),
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