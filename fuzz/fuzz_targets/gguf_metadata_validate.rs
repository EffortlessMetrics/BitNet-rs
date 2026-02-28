#![no_main]

use arbitrary::Arbitrary;
use bitnet_gguf::{check_magic, parse_header, read_version};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct MetadataInput {
    kv_count: u8,
    entries: Vec<KvEntry>,
    trailing: Vec<u8>,
}

#[derive(Arbitrary, Debug)]
struct KvEntry {
    key: Vec<u8>,
    value_type: u8,
    str_val: Vec<u8>,
    int_val: u64,
    float_val: f32,
    bool_val: bool,
    array_len: u8,
}

fuzz_target!(|input: MetadataInput| {
    if input.trailing.len() > 512 {
        return;
    }

    let kv_count = (input.kv_count as u64).min(16);
    let entries: Vec<&KvEntry> = input.entries.iter().take(kv_count as usize).collect();

    let buf = build_gguf_with_metadata(kv_count, &entries, &input.trailing);

    // Low-level primitives must never panic on crafted metadata.
    let _ = check_magic(&buf);
    let _ = read_version(&buf);
    let _ = parse_header(&buf);

    // GgufReader must handle arbitrary metadata kv pairs.
    if buf.len() >= 16 {
        if let Ok(reader) = bitnet_models::GgufReader::new(&buf) {
            let _ = reader.metadata_kv_count();
            let _ = reader.tensor_count();
            let _ = reader.validate();

            // Iterate metadata by index — must never panic.
            let count = reader.metadata_kv_count();
            let _ = reader.metadata_count();
            let _ = count;

            // Lookup by arbitrary key — must never panic.
            for e in &entries {
                if let Ok(key) = std::str::from_utf8(&e.key) {
                    let _ = reader.get_string_metadata(key);
                    let _ = reader.get_u32_metadata(key);
                    let _ = reader.get_i32_metadata(key);
                    let _ = reader.get_f32_metadata(key);
                    let _ = reader.get_bool_metadata(key);
                }
            }

            // Architecture-specific lookups.
            let _ = reader.get_string_metadata("general.architecture");
            let _ = reader.get_string_metadata("general.name");
            let _ = reader.get_u32_metadata("general.file_type");
            let _ = reader.get_u32_metadata("llama.context_length");
            let _ = reader.get_u32_metadata("llama.embedding_length");
            let _ = reader.get_u32_metadata("llama.block_count");
        }
    }

    // Also test sub-slices.
    for end in [16, 24, 48, buf.len().min(256), buf.len()] {
        if end <= buf.len() {
            let _ = parse_header(&buf[..end]);
        }
    }
});

fn build_gguf_with_metadata(kv_count: u64, entries: &[&KvEntry], trailing: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();

    // GGUF magic + version 3
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    // tensor_count = 0
    buf.extend_from_slice(&0u64.to_le_bytes());
    // metadata_kv_count
    buf.extend_from_slice(&kv_count.to_le_bytes());

    // Write metadata kv entries
    for e in entries {
        let key_len = e.key.len().min(128);
        // key: u64 length + bytes
        buf.extend_from_slice(&(key_len as u64).to_le_bytes());
        buf.extend_from_slice(&e.key[..key_len]);

        // value_type: GGUF types 0-12
        let vtype = e.value_type % 13;
        buf.extend_from_slice(&(vtype as u32).to_le_bytes());

        match vtype {
            0 => buf.push(e.bool_val as u8), // UINT8
            1 => buf.push(e.int_val as u8),  // INT8
            2 => buf.extend_from_slice(&(e.int_val as u16).to_le_bytes()), // UINT16
            3 => buf.extend_from_slice(&(e.int_val as i16).to_le_bytes()), // INT16
            4 => buf.extend_from_slice(&(e.int_val as u32).to_le_bytes()), // UINT32
            5 => buf.extend_from_slice(&(e.int_val as i32).to_le_bytes()), // INT32
            6 => buf.extend_from_slice(&e.float_val.to_le_bytes()), // FLOAT32
            7 => buf.push(e.bool_val as u8), // BOOL
            8 => {
                // STRING
                let s_len = e.str_val.len().min(128);
                buf.extend_from_slice(&(s_len as u64).to_le_bytes());
                buf.extend_from_slice(&e.str_val[..s_len]);
            }
            9 => {
                // ARRAY — write small array of UINT32
                let arr_len = (e.array_len as u64).min(8);
                buf.extend_from_slice(&4u32.to_le_bytes()); // element type UINT32
                buf.extend_from_slice(&arr_len.to_le_bytes());
                for i in 0..arr_len {
                    buf.extend_from_slice(&(i as u32).to_le_bytes());
                }
            }
            10 => buf.extend_from_slice(&e.int_val.to_le_bytes()), // UINT64
            11 => buf.extend_from_slice(&(e.int_val as i64).to_le_bytes()), // INT64
            12 => buf.extend_from_slice(&(e.float_val as f64).to_le_bytes()), // FLOAT64
            _ => {}
        }
    }

    buf.extend_from_slice(trailing);
    buf
}
