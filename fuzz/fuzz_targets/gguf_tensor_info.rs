#![no_main]

use arbitrary::Arbitrary;
use bitnet_gguf::{check_magic, parse_header, read_version};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct TensorInfoInput {
    /// Number of tensor entries to encode (capped).
    n_tensors: u8,
    /// Per-tensor fuzz data: name bytes, n_dims, dim values, dtype, offset.
    tensors: Vec<TensorEntry>,
    /// Extra trailing bytes after tensor index.
    trailing: Vec<u8>,
}

#[derive(Arbitrary, Debug)]
struct TensorEntry {
    name: Vec<u8>,
    n_dims: u8,
    dims: Vec<u32>,
    dtype: u32,
    offset: u64,
}

fuzz_target!(|input: TensorInfoInput| {
    if input.trailing.len() > 512 {
        return;
    }

    let n_tensors = (input.n_tensors as u64).min(16);
    let tensors: Vec<&TensorEntry> = input.tensors.iter().take(n_tensors as usize).collect();

    let buf = build_gguf_with_tensors(n_tensors, &tensors, &input.trailing);

    // Low-level primitives must never panic.
    let _ = check_magic(&buf);
    let _ = read_version(&buf);
    let _ = parse_header(&buf);

    // GgufReader must handle arbitrary tensor counts, names, dims, dtypes.
    if buf.len() >= 16 {
        if let Ok(reader) = bitnet_models::GgufReader::new(&buf) {
            let _ = reader.tensor_count();
            let _ = reader.validate();

            // Iterate tensor info by index — must never panic.
            let count = reader.tensor_count();
            for i in 0..count.min(32) {
                let _ = reader.get_tensor_info(i as usize);
            }

            // Lookup by arbitrary name — must never panic.
            for t in &tensors {
                if let Ok(name) = std::str::from_utf8(&t.name) {
                    let _ = reader.get_tensor_info_by_name(name);
                }
            }
        }
    }

    // Also feed sub-slices to catch off-by-one in tensor index parsing.
    for end in [24, 32, 48, buf.len().min(128), buf.len()] {
        if end <= buf.len() {
            let _ = parse_header(&buf[..end]);
        }
    }
});

fn build_gguf_with_tensors(n_tensors: u64, tensors: &[&TensorEntry], trailing: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();

    // GGUF magic + version 3
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    // tensor_count
    buf.extend_from_slice(&n_tensors.to_le_bytes());
    // metadata_kv_count = 0
    buf.extend_from_slice(&0u64.to_le_bytes());

    // Write tensor info entries
    for t in tensors {
        let name_len = t.name.len().min(128);
        // name: u64 length + bytes
        buf.extend_from_slice(&(name_len as u64).to_le_bytes());
        buf.extend_from_slice(&t.name[..name_len]);
        // n_dims (u32)
        let n_dims = (t.n_dims as u32).min(4);
        buf.extend_from_slice(&n_dims.to_le_bytes());
        // dims (u64 each)
        for i in 0..n_dims as usize {
            let dim = t.dims.get(i).copied().unwrap_or(1) as u64;
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        // dtype (u32)
        buf.extend_from_slice(&t.dtype.to_le_bytes());
        // offset (u64)
        buf.extend_from_slice(&t.offset.to_le_bytes());
    }

    buf.extend_from_slice(trailing);
    buf
}
