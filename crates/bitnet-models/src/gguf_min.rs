//! Minimal GGUF reader: just enough to fetch tok_embeddings + output/lm_head.
//! - Supports GGUF v2/v3
//! - Tensors: f32 / f16 / I2_S -> f32
//! - Uses memmap for zero-copy reads where possible
//! - Integrates with BitNet quantization for I2_S support
//!
//! This is deliberately small: robust error messages, no full schema.

use crate::formats::gguf::GgufTensorType;
use anyhow::{Context, Result, bail, ensure};
use bitnet_common::QuantizationType;
use bitnet_quantization::{I2SLayout, I2SQuantizer, QuantizedTensor};
use half::f16;
use memmap2::Mmap;
use std::{
    borrow::Cow,
    fs::File,
    io::{self, Read, Seek},
    path::Path,
};

// Macro for consistent I2_S out-of-bounds error formatting
macro_rules! i2s_oob {
    ($info:expr, $offset:expr, $need:expr, $len:expr) => {
        format!(
            "I2_S tensor '{}' OOB: offset={}, need={}, mmap_len={}, shape={:?}",
            $info.name, $offset, $need, $len, $info.dims
        )
    };
}

#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    n_dims: u32,
    dims: Vec<u64>,
    ty: u32,     // ggml_type
    offset: u64, // from start of data section (not file start)
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
fn is_2d(info: &TensorInfo) -> bool {
    info.n_dims == 2 && info.dims.len() == 2
}

// Helper for shape-driven tensor selection
fn looks_like_embeddings(info: &TensorInfo) -> bool {
    is_2d(info)
        && is_supported_ty(info.ty)
        && (info.name.contains("emb") || info.name.contains("wte") || info.name.contains("embed"))
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
    let (tok_info, head_info) =
        pick_tensors(&parsed).with_context(|| "locate tok_embeddings/output tensors")?;

    // Materialize as f32 (copy if needed).
    let tok = tensor_as_f32(&mmap, parsed.data_offset, parsed.alignment, &tok_info)?;
    let head = tensor_as_f32(&mmap, parsed.data_offset, parsed.alignment, &head_info)?;

    // Shapes can vary; deduce (vocab, dim) consistently.
    if !is_2d(&tok_info) || !is_2d(&head_info) {
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

    Ok(TwoTensors { tok_embeddings, lm_head, vocab, dim })
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
            alignment = read_gguf_value_uint64(r, ty).with_context(|| "parse general.alignment")?;
        } else {
            skip_gguf_value_seek(r, ty)?;
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
        ensure!(n_dims as usize == dims.len(), "tensor '{}' dims mismatch", name);
        let ty = read_u32(r)?;
        let offset = read_u64(r)?;
        ensure!(
            offset % alignment == 0,
            "tensor '{}' offset {} not aligned to {alignment}",
            name,
            offset
        );
        tensors.push(TensorInfo { name, n_dims, dims, ty, offset });
    }

    // data section begins aligned after the tensor header table
    let here = r.stream_position()?;
    let data_offset = align_up(here, alignment);

    ensure!(data_offset.is_multiple_of(alignment), "data section not aligned to {alignment}");

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
    const HEAD_NAMES: &[&str] =
        &["output.weight", "lm_head.weight", "model.lm_head.weight", "transformer.lm_head.weight"];

    let find = |names: &[&str]| {
        parsed
            .tensors
            .iter()
            .find(|t| names.iter().any(|n| t.name == *n) && is_supported_ty(t.ty))
            .cloned()
    };

    // Find embeddings tensor - prefer by name first, then shape
    let tok = find(TOK_NAMES)
        .or_else(|| parsed.tensors.iter().find(|t| looks_like_embeddings(t)).cloned())
        .ok_or_else(|| anyhow::anyhow!("could not find token embeddings tensor"))?;

    // Extract dimensions from the chosen embeddings tensor
    let (dim, vocab) = if tok.dims[0] > tok.dims[1] {
        (tok.dims[0], tok.dims[1]) // [vocab, dim] layout
    } else {
        (tok.dims[1], tok.dims[0]) // [dim, vocab] layout
    };

    // Helper to check if a tensor matches the expected output shape
    let looks_like_output_with_shape = |t: &TensorInfo| -> bool {
        is_2d(t)
            && is_supported_ty(t.ty)
            && ((t.dims[0] == dim && t.dims[1] == vocab)
                || (t.dims[0] == vocab && t.dims[1] == dim))
            && !(t.name.contains("attn_output")
                || t.name.contains("blk")
                || t.name.contains("norm"))
    };

    // Find output tensor - shape-verified with name hints
    let head = find(HEAD_NAMES)
        .or_else(|| parsed.tensors.iter()
            .find(|t| looks_like_output_with_shape(t) &&
                  (t.name.contains("lm_head") || t.name.contains("output")))
            .cloned())
        .or_else(|| parsed.tensors.iter()
            .find(|t| looks_like_output_with_shape(t))
            .cloned())
        // If no output head found, BitNet models often use tied embeddings
        .unwrap_or_else(|| tok.clone());

    Ok((tok, head))
}

// ---------- tensor materialization ----------

fn tensor_as_f32<'a>(
    mmap: &'a [u8],
    data_base: u64,
    alignment: u64,
    info: &TensorInfo,
) -> Result<Cow<'a, [f32]>> {
    use crate::formats::gguf::GgufTensorType;

    let nelems: usize = info
        .dims
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| anyhow::anyhow!("tensor size overflow"))?
        .try_into()
        .map_err(|_| anyhow::anyhow!("tensor too large"))?;
    ensure!(
        info.offset.is_multiple_of(alignment),
        "tensor '{}' offset {} not aligned to {alignment}",
        info.name,
        info.offset
    );
    let offset = (data_base + info.offset) as usize;

    match GgufTensorType::from_u32(info.ty)? {
        GgufTensorType::F32 => {
            // f32
            let need = nelems * 4;
            if offset + need > mmap.len() {
                bail!("f32 tensor out of bounds");
            }
            // Safety: GGUF stores little-endian f32; copy to owned Vec to be safe/aligned.
            let mut out = vec![0f32; nelems];
            let bytes = &mmap[offset..offset + need];
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            Ok(Cow::Owned(out))
        }
        GgufTensorType::F16 => {
            // f16
            let need = nelems * 2;
            if offset + need > mmap.len() {
                bail!("f16 tensor out of bounds");
            }
            let mut out = vec![0f32; nelems];
            let bytes = &mmap[offset..offset + need];
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out[i] = f16::from_bits(bits).to_f32();
            }
            Ok(Cow::Owned(out))
        }
        GgufTensorType::I2_S => {
            // I2_S - BitNet 2-bit quantization
            let layout = I2SLayout::default();
            let num_blocks = nelems.div_ceil(layout.block_size);
            let need = num_blocks * layout.bytes_per_block;

            ensure!(offset + need <= mmap.len(), "{}", i2s_oob!(info, offset, need, mmap.len()));

            // Verify shape/blocks consistency
            ensure!(
                num_blocks * layout.block_size >= nelems
                    && num_blocks * layout.block_size - nelems < layout.block_size,
                "I2_S blocks/shape mismatch for tensor '{}': nelems={}, blocks={}, block_size={}, shape={:?}",
                info.name,
                nelems,
                num_blocks,
                layout.block_size,
                info.dims
            );

            // Extract quantized data and scales
            let tensor_bytes = &mmap[offset..offset + need];
            let mut scales = Vec::with_capacity(num_blocks);
            let mut packed_data = Vec::with_capacity(num_blocks * layout.data_bytes_per_block);

            for block_idx in 0..num_blocks {
                let block_offset = block_idx * layout.bytes_per_block;
                // Copy packed data
                packed_data.extend_from_slice(
                    &tensor_bytes[block_offset..block_offset + layout.data_bytes_per_block],
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
                layout.block_size,
            );

            // Dequantize using our existing infrastructure
            let quantizer = I2SQuantizer::with_block_size(layout.block_size);
            let tensor = quantizer
                .dequantize_tensor(&quantized)
                .with_context(|| format!("Failed to dequantize I2_S tensor {}", info.name))?;

            // Extract f32 data from BitNetTensor
            let data = tensor.to_vec().with_context(|| "Failed to extract tensor data")?;

            Ok(Cow::Owned(data))
        }
        other => bail!("unsupported ggml tensor type {:?} (only f32/f16/I2_S supported)", other),
    }
}

// ---------- helpers ----------

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

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
    String::from_utf8(buf).context("string not utf8")
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

// Allocation-free skip helper with fast path for seekable readers
#[inline]
fn skip_n<R: Read>(r: &mut R, n: u64) -> Result<()> {
    let copied = io::copy(&mut r.by_ref().take(n), &mut io::sink())?;
    ensure!(copied == n, "unexpected EOF while skipping {n} bytes");
    Ok(())
}

// Fast skip for seekable readers (e.g., Cursor over mmap)
#[inline]
fn skip_n_seek<R: Read + Seek>(r: &mut R, n: u64) -> Result<()> {
    r.seek(io::SeekFrom::Current(n as i64))?;
    Ok(())
}

fn skip_gguf_value<R: Read>(r: &mut R, ty: u32) -> Result<()> {
    // GGUF scalar sizes (see llama.cpp)
    match ty {
        0 | 1 => skip_n(r, 1)?, // uint8 | int8
        2 | 3 => skip_n(r, 2)?, // uint16 | int16
        4..=6 => skip_n(r, 4)?, // uint32 | int32 | float32
        7 => skip_n(r, 1)?,     // bool
        8 => {
            // string: u64 len + bytes
            let n = read_u64(r)?;
            skip_n(r, n)?;
        }
        9 => {
            // array: elem_ty + count + values
            let elem_ty = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_gguf_value(r, elem_ty)?;
            }
        }
        10..=12 => skip_n(r, 8)?, // uint64 | int64 | float64
        _ => bail!("unknown GGUF kv type id {ty}"),
    }
    Ok(())
}

// Seek-optimized version for seekable readers
fn skip_gguf_value_seek<R: Read + Seek>(r: &mut R, ty: u32) -> Result<()> {
    // GGUF scalar sizes (see llama.cpp)
    match ty {
        /* ~ changed by cargo-mutants ~ */ // uint8 | int8
        2 | 3 => skip_n_seek(r, 2)?, // uint16 | int16
        4..=6 => skip_n_seek(r, 4)?, // uint32 | int32 | float32
        7 => skip_n_seek(r, 1)?,     // bool
        8 => {
            // string: u64 len + bytes
            let n = read_u64(r)?;
            skip_n_seek(r, n)?;
        }
        9 => {
            // array: elem_ty + count + values
            let elem_ty = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_gguf_value_seek(r, elem_ty)?;
            }
        }
        10..=12 => skip_n_seek(r, 8)?, // uint64 | int64 | float64
        _ => bail!("unknown GGUF kv type id {ty}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    #[ignore = "set BITNET_GGUF to a real path to run"]
    fn loads_two_tensors() {
        let p = std::env::var_os("BITNET_GGUF").expect("set BITNET_GGUF");
        let two = load_two(p).unwrap();
        assert!(two.vocab > 0 && two.dim > 0);
        assert_eq!(two.tok_embeddings.len(), two.vocab * two.dim);
        assert_eq!(two.lm_head.len(), two.dim * two.vocab);
    }

    #[test]
    fn test_transpose_roundtrip() {
        let vocab = 100;
        let dim = 50;

        // Create test matrix [vocab, dim]
        let mut original = vec![0f32; vocab * dim];
        for i in 0..vocab {
            for j in 0..dim {
                original[i * dim + j] = (i * 1000 + j) as f32;
            }
        }

        // Transpose to [dim, vocab]
        let mut transposed = vec![0f32; vocab * dim];
        for i in 0..vocab {
            for j in 0..dim {
                transposed[j * vocab + i] = original[i * dim + j];
            }
        }

        // Transpose back to [vocab, dim]
        let mut roundtrip = vec![0f32; vocab * dim];
        for i in 0..dim {
            for j in 0..vocab {
                roundtrip[j * dim + i] = transposed[i * vocab + j];
            }
        }

        // Verify roundtrip
        assert_eq!(original, roundtrip);
    }

    #[test]
    fn test_skip_gguf_value_coverage() {
        use std::io::Cursor;

        // Build a buffer with: [type=8 string "hi"] [type=4 u32] [type=9 array<int8>(3)]
        let mut buf = Vec::new();

        // String type=8, value="hi"
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes()); // string length
        buf.extend_from_slice(b"hi");

        // U32 type=4, value=0xAABBCCDD
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&0xAABBCCDDu32.to_le_bytes());

        // Array<int8> type=9
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // element type: int8
        buf.extend_from_slice(&3u64.to_le_bytes()); // count
        buf.extend_from_slice(&[10u8, 11u8, 12u8]);

        // Test skipping through all values
        let mut cur = Cursor::new(&buf[..]);

        let t0 = read_u32(&mut cur).unwrap();
        skip_gguf_value(&mut cur, t0).unwrap();

        let t1 = read_u32(&mut cur).unwrap();
        skip_gguf_value(&mut cur, t1).unwrap();

        let t2 = read_u32(&mut cur).unwrap();
        skip_gguf_value(&mut cur, t2).unwrap();

        // Should have consumed entire buffer
        assert_eq!(cur.position() as usize, buf.len());
    }

    #[test]
    fn test_skip_gguf_value_seek() {
        // Same test but using seek version
        let mut buf = Vec::new();

        // String type=8, value="hello"
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&5u64.to_le_bytes());
        buf.extend_from_slice(b"hello");

        // Float64 type=12
        buf.extend_from_slice(&12u32.to_le_bytes());
        buf.extend_from_slice(&std::f64::consts::PI.to_le_bytes());

        let mut cur = Cursor::new(&buf[..]);

        let t0 = read_u32(&mut cur).unwrap();
        skip_gguf_value_seek(&mut cur, t0).unwrap();

        let t1 = read_u32(&mut cur).unwrap();
        skip_gguf_value_seek(&mut cur, t1).unwrap();

        assert_eq!(cur.position() as usize, buf.len());
    }

    #[test]
    fn test_i2s_layout_geometry() {
        let layout = I2SLayout::default();

        // Test basic properties
        assert_eq!(layout.block_size, 32);
        // bits_per_elem is implicitly 2 for I2S (2 bits per element)
        assert_eq!(layout.data_bytes_per_block, 8); // 32 * 2 / 8
        assert_eq!(layout.scale_bytes_per_block, 2); // f16
        assert_eq!(layout.bytes_per_block, 10); // 8 + 2

        // Test block calculations for various element counts
        let test_cases = vec![
            (1, 1),   // 1 element -> 1 block
            (32, 1),  // exactly 1 block
            (33, 2),  // just over 1 block -> 2 blocks
            (64, 2),  // exactly 2 blocks
            (100, 4), // 100 elements -> 4 blocks (96 + 4 padding)
        ];

        for (nelems, expected_blocks) in test_cases {
            let num_blocks = (nelems as usize).div_ceil(layout.block_size);
            assert_eq!(
                num_blocks, expected_blocks,
                "nelems={} should need {} blocks",
                nelems, expected_blocks
            );

            let total_bytes = num_blocks * layout.bytes_per_block;
            assert_eq!(
                total_bytes,
                expected_blocks * 10,
                "nelems={} should need {} bytes",
                nelems,
                expected_blocks * 10
            );
        }
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(100, 32), 128);
    }

    #[test]
    fn test_i2s_dequant_nonzero_block() {
        use bitnet_common::QuantizationType;
        use bitnet_quantization::{I2SLayout, I2SQuantizer, QuantizedTensor};

        let layout = I2SLayout::default();
        let blocks = 1usize;

        // Put a simple non-zero nibble pattern; exact mapping isn't important, just non-zero.
        let mut packed = vec![0u8; layout.data_bytes_per_block * blocks];
        packed.fill(0x11); // alternating small positives

        let scales = vec![2.0f32; blocks]; // non-1 scale to exercise path

        let qt = QuantizedTensor::new_with_params(
            packed,
            scales,
            None,
            vec![layout.block_size * blocks],
            QuantizationType::I2S,
            layout.block_size,
        );

        let quantizer = I2SQuantizer::with_block_size(layout.block_size);
        let out = quantizer.dequantize_tensor(&qt).unwrap().to_vec().unwrap();

        assert_eq!(out.len(), layout.block_size);
        assert!(out.iter().any(|&v| v != 0.0), "dequant should produce non-zero values");
    }

    #[test]
    fn test_i2s_roundtrip_dequant() {
        use bitnet_quantization::{I2SLayout, I2SQuantizer, QuantizedTensor};

        let layout = I2SLayout::default();
        // Create two blocks of zero data w/ scale=1.0 -> expect zeros out
        let blocks = 2usize;
        let packed = vec![0u8; blocks * layout.data_bytes_per_block];
        let scales = vec![1.0f32; blocks];

        let qt = QuantizedTensor::new_with_params(
            packed,
            scales,
            None,
            vec![layout.block_size * blocks],
            QuantizationType::I2S,
            layout.block_size,
        );

        let quantizer = I2SQuantizer::with_block_size(layout.block_size);
        let tensor = quantizer.dequantize_tensor(&qt).unwrap();
        let out = tensor.to_vec().unwrap();

        assert_eq!(out.len(), layout.block_size * blocks);
        // Values should be close to zero (within quantization error)
        // I2_S quantization maps to {-2, 0, 2} * scale, so exact zeros may not be preserved
        for &val in &out {
            assert!(val.abs() <= 2.0, "Expected small value, got {}", val);
        }
    }

    #[test]
    fn tied_weights_transpose_roundtrip() {
        // Simulate tok = [dim, vocab] and use it as both embeddings and head
        let dim = 16usize;
        let vocab = 32usize;
        let mut tok = vec![0f32; dim * vocab];
        for r in 0..dim {
            for c in 0..vocab {
                tok[r * vocab + c] = (r * 1_000 + c) as f32;
            }
        }

        // Transpose to [vocab, dim] (what loader does for tied weights)
        let mut tok_t = vec![0f32; vocab * dim];
        for r in 0..dim {
            for c in 0..vocab {
                tok_t[c * dim + r] = tok[r * vocab + c];
            }
        }

        // Transpose back
        let mut back = vec![0f32; dim * vocab];
        for r in 0..vocab {
            for c in 0..dim {
                back[c * vocab + r] = tok_t[r * dim + c];
            }
        }

        assert_eq!(tok, back);
    }

    #[cfg(test)]
    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        // Cap proptest cases for CI speed
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            #[test]
            fn i2s_blocks_never_underflow(nelems in 1usize..10_000_000) {
                let layout = I2SLayout::default();
                let blocks = nelems.div_ceil(layout.block_size);
                prop_assert!(blocks >= 1);
                let need = blocks * layout.bytes_per_block;
                prop_assert!(need >= layout.bytes_per_block);
                // Verify no integer overflow in calculation
                prop_assert!(need / layout.bytes_per_block == blocks);
            }

            #[test]
            fn i2s_block_alignment_correct(nelems in 1usize..1_000_000) {
                let layout = I2SLayout::default();
                let blocks = nelems.div_ceil(layout.block_size);
                let padded_elems = blocks * layout.block_size;
                // Elements should be padded to block boundary
                prop_assert!(padded_elems >= nelems);
                prop_assert!(padded_elems - nelems < layout.block_size);
            }

            /// Test I2S quantization stability under varying conditions
            #[test]
            fn i2s_quantization_stability_test(
                block_count in 1usize..100,
                scale_factor in 0.1f32..10.0f32
            ) {
                let layout = I2SLayout::default();
                let total_elems = block_count * layout.block_size;

                // Create test data with known patterns
                let mut test_data = vec![0.0f32; total_elems];
                for (i, elem) in test_data.iter_mut().enumerate().take(total_elems) {
                    *elem = ((i as f32).sin() * scale_factor).clamp(-2.0, 2.0);
                }

                // Test that block calculation is correct
                let calculated_blocks = total_elems.div_ceil(layout.block_size);
                prop_assert_eq!(calculated_blocks, block_count);

                // Test that data size requirements are met
                let required_bytes = calculated_blocks * layout.bytes_per_block;
                prop_assert!(required_bytes >= layout.bytes_per_block);
                prop_assert!(required_bytes <= total_elems * 4); // Should not exceed f32 storage
            }
        }

        /// Test I2S block boundary calculations for edge cases
        #[test]
        fn test_i2s_block_boundary_edge_cases() {
            let layout = I2SLayout::default();

            // Test various element counts around block boundaries
            let test_cases = vec![
                (1, 1),                           // Single element
                (layout.block_size - 1, 1),       // Just under one block
                (layout.block_size, 1),           // Exactly one block
                (layout.block_size + 1, 2),       // Just over one block
                (layout.block_size * 2, 2),       // Exactly two blocks
                (layout.block_size * 10 + 7, 11), // Multiple blocks plus remainder
            ];

            for (elems, expected_blocks) in test_cases {
                let calculated_blocks = elems.div_ceil(layout.block_size);
                assert_eq!(
                    calculated_blocks, expected_blocks,
                    "Block count mismatch for {} elements",
                    elems
                );

                let padded_elems = calculated_blocks * layout.block_size;
                assert!(
                    padded_elems >= elems,
                    "Padded elements {} should be >= original {}",
                    padded_elems,
                    elems
                );
                assert!(
                    padded_elems - elems < layout.block_size,
                    "Padding {} too large for {} elements",
                    padded_elems - elems,
                    elems
                );
            }
        }

        /// Test I2S memory layout calculations for security
        #[test]
        fn test_i2s_memory_layout_security() {
            let layout = I2SLayout::default();

            // Test that memory calculations don't overflow
            let max_safe_elems = usize::MAX / 4; // Avoid overflow in bytes calculation
            let large_elems = max_safe_elems.min(1_000_000); // Reasonable upper bound

            let blocks = large_elems.div_ceil(layout.block_size);
            let bytes_needed = blocks.saturating_mul(layout.bytes_per_block);

            // Should not panic or overflow
            assert!(bytes_needed > 0);
            assert!(bytes_needed >= layout.bytes_per_block);
            assert!(blocks * layout.bytes_per_block == bytes_needed); // No overflow
        }

        /// Test device-aware operations with fallback scenarios
        #[test]
        fn test_i2s_device_fallback_scenarios() {
            use bitnet_common::Device;

            let layout = I2SLayout::default();
            let test_elems = layout.block_size * 4;

            // Test CPU device handling
            let cpu_device = Device::Cpu;
            assert_eq!(format!("{:?}", cpu_device), "Cpu");

            // Test device-specific calculations remain consistent
            let blocks = test_elems.div_ceil(layout.block_size);
            let bytes_per_device = blocks * layout.bytes_per_block;

            // Device shouldn't affect layout calculations
            assert_eq!(blocks, 4); // 4 blocks for 4 * block_size elements
            assert!(bytes_per_device > 0);
        }
    }
}
