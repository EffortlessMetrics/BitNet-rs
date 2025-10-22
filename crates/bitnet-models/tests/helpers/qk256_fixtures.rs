//! QK256 GGUF Fixture Generators for BitNet.rs Testing
//!
//! This module provides deterministic fixture generators for QK256 (GGML I2_S) and
//! BitNet32F16 (I2_S with inline F16 scales) quantization formats. Each generator
//! creates minimal, valid GGUF v3 files with single tensors for focused testing.
//!
//! ## Fixture Types
//!
//! 1. **QK256 4×256** - Single-block QK256 tensor (256 elements per row)
//! 2. **BitNet32 2×64** - Two-block BitNet32F16 tensor (32 elements per block)
//! 3. **QK256 3×300** - Multi-block QK256 with tail (1 full block + partial tail)
//!
//! ## GGUF Structure
//!
//! All fixtures follow GGUF v3 format:
//! - 4-byte magic: "GGUF"
//! - 4-byte version: 3 (little-endian u32)
//! - 8-byte tensor count (little-endian u64)
//! - 8-byte KV count (little-endian u64)
//! - KV pairs: (key_len, key_bytes, type, value)
//! - Tensor info: (name_len, name_bytes, n_dims, dims[], type, offset)
//! - Alignment padding to 32 bytes
//! - Tensor data

/// QK256 block size (256 elements)
const QK256_BLOCK: usize = 256;

/// QK256 packed bytes per block (2 bits × 256 elements / 8 = 64 bytes)
const QK256_PACKED_BYTES: usize = 64;

/// BitNet32F16 block size (32 elements)
const BITNET32_BLOCK: usize = 32;

/// BitNet32F16 bytes per block (8 bytes packed data + 2 bytes F16 scale = 10 bytes)
const BITNET32_BYTES_PER_BLOCK: usize = 10;

/// GGUF alignment (32 bytes for data section)
const GGUF_ALIGNMENT: usize = 32;

/// GGUF v3 version number
const GGUF_VERSION: u32 = 3;

/// GGUF data type for I2_S quantization (GGUF type 26)
const GGUF_TYPE_I2S: u32 = 26;

/// GGUF value type: String (type 8)
const GGUF_VALUE_TYPE_STRING: u32 = 8;

/// Generate QK256 4×256 single-block fixture
///
/// Creates a minimal GGUF v3 file with a single QK256 tensor of shape [4, 256].
/// This tests the single-block edge case where cols exactly equals QK256_BLOCK.
/// Uses canonical tensor name "tok_embeddings.weight" for parser compatibility.
///
/// # Arguments
///
/// * `seed` - Seed for deterministic RNG (affects code pattern)
///
/// # Returns
///
/// Valid GGUF v3 bytes ready for writing to disk or mmap
pub fn generate_qk256_4x256(seed: u64) -> Vec<u8> {
    let rows = 4usize;
    let cols = 256usize;
    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;
    let tensor_bytes = rows * row_stride_bytes;

    let code = ((seed % 4) as u8).clamp(0, 3);
    let packed_byte = code | (code << 2) | (code << 4) | (code << 6);
    let tensor_data = vec![packed_byte; tensor_bytes];

    build_gguf_fixture("tok_embeddings.weight", rows, cols, GGUF_TYPE_I2S, &tensor_data, seed)
}

/// Generate BitNet32F16 2×64 two-block fixture
///
/// Creates a minimal GGUF v3 file with a single BitNet32F16 tensor of shape [2, 64].
/// This tests the two-block case with inline F16 scales (format detection via size).
/// Uses canonical tensor name "tok_embeddings.weight" for parser compatibility.
///
/// # Arguments
///
/// * `seed` - Seed for deterministic RNG
///
/// # Returns
///
/// Valid GGUF v3 bytes with BitNet32F16 format
pub fn generate_bitnet32_2x64(seed: u64) -> Vec<u8> {
    let rows = 2usize;
    let cols = 64usize;
    let blocks_per_row = cols.div_ceil(BITNET32_BLOCK);
    let bytes_per_row = blocks_per_row * BITNET32_BYTES_PER_BLOCK;

    let code = ((seed % 4) as u8).clamp(0, 3);
    let packed_byte = code | (code << 2) | (code << 4) | (code << 6);

    // F16 scale value of 1.0 (0x3C00 in little-endian)
    let scale_f16: [u8; 2] = [0x00, 0x3C];

    let mut tensor_data = Vec::with_capacity(rows * bytes_per_row);
    for _row in 0..rows {
        for _block in 0..blocks_per_row {
            tensor_data.extend_from_slice(&[packed_byte; 8]);
            tensor_data.extend_from_slice(&scale_f16);
        }
    }

    build_gguf_fixture("tok_embeddings.weight", rows, cols, GGUF_TYPE_I2S, &tensor_data, seed)
}

/// Generate QK256 3×300 multi-block with tail fixture
///
/// Creates a minimal GGUF v3 file with a single QK256 tensor of shape [3, 300].
/// This tests multi-block handling with a tail (300 = 256 + 44 tail elements).
/// Uses canonical tensor name "tok_embeddings.weight" for parser compatibility.
///
/// # Arguments
///
/// * `seed` - Seed for deterministic RNG
///
/// # Returns
///
/// Valid GGUF v3 bytes with QK256 multi-block layout
pub fn generate_qk256_3x300(seed: u64) -> Vec<u8> {
    let rows = 3usize;
    let cols = 300usize;
    let blocks_per_row = cols.div_ceil(QK256_BLOCK);
    let row_stride_bytes = blocks_per_row * QK256_PACKED_BYTES;
    let tensor_bytes = rows * row_stride_bytes;

    let code = ((seed % 4) as u8).clamp(0, 3);
    let packed_byte = code | (code << 2) | (code << 4) | (code << 6);
    let tensor_data = vec![packed_byte; tensor_bytes];

    build_gguf_fixture("tok_embeddings.weight", rows, cols, GGUF_TYPE_I2S, &tensor_data, seed)
}

/// Build a complete GGUF v3 fixture from tensor specifications
///
/// Creates a minimal GGUF file with TWO tensors (tok_embeddings.weight and output.weight)
/// to satisfy the minimal parser's requirement for both embedding and output layers.
/// Both tensors use the same quantization data for simplicity.
fn build_gguf_fixture(
    _tensor_name: &str, // Ignored - always uses canonical names
    rows: usize,
    cols: usize,
    data_type: u32,
    tensor_data: &[u8],
    seed: u64,
) -> Vec<u8> {
    let mut buf = Vec::new();

    // Header
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes()); // tensor count: 2 (tok_embeddings + output)
    buf.extend_from_slice(&8u64.to_le_bytes()); // KV count (added num_heads, kv_heads, intermediate_size)

    // KV pairs - add required metadata for GGUF loader
    write_kv_string(&mut buf, "general.name", &format!("fixture_seed_{}", seed));
    write_kv_string(&mut buf, "general.architecture", "bitnet");

    // Add tokenizer.ggml.tokens as a string array (required by gguf_simple.rs line 619)
    // This is the authoritative source for vocab_size
    write_kv_string_array(&mut buf, "tokenizer.ggml.tokens", 1000);

    // Add embedding_length metadata (required by gguf_parity and gguf_simple.rs line 625)
    write_kv_u32(&mut buf, "bitnet-b1.58.embedding_length", 512);

    // Add block_count metadata to avoid layer discovery from tensor names (gguf_simple.rs line 638)
    // Set to 1 since this is a minimal fixture with no layer tensors
    write_kv_u32(&mut buf, "bitnet-b1.58.block_count", 1);

    // Add num_heads metadata (gguf_simple.rs line 650)
    write_kv_u32(&mut buf, "bitnet-b1.58.attention.head_count", 8);

    // Add num_key_value_heads metadata (gguf_simple.rs line 661)
    write_kv_u32(&mut buf, "bitnet-b1.58.attention.head_count_kv", 8);

    // Add intermediate_size metadata (gguf_simple.rs line 678)
    write_kv_u32(&mut buf, "bitnet-b1.58.feed_forward_length", 2048);

    // Tensor 1: tok_embeddings.weight (21 chars)
    let name1 = "tok_embeddings.weight";
    buf.extend_from_slice(&(name1.len() as u64).to_le_bytes()); // name length
    buf.extend_from_slice(name1.as_bytes());
    // Pad name to 8-byte boundary: (8 - (21 % 8)) % 8 = 3 bytes
    let pad1 = (8 - (name1.len() % 8)) % 8;
    buf.extend_from_slice(&vec![0u8; pad1]);
    buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
    buf.extend_from_slice(&(rows as u64).to_le_bytes());
    buf.extend_from_slice(&(cols as u64).to_le_bytes());
    buf.extend_from_slice(&data_type.to_le_bytes());

    let tok_offset_pos = buf.len();
    buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder for offset

    // Tensor 2: output.weight (13 chars)
    let name2 = "output.weight";
    buf.extend_from_slice(&(name2.len() as u64).to_le_bytes()); // name length
    buf.extend_from_slice(name2.as_bytes());
    // Pad name to 8-byte boundary: (8 - (13 % 8)) % 8 = 3 bytes
    let pad2 = (8 - (name2.len() % 8)) % 8;
    buf.extend_from_slice(&vec![0u8; pad2]);
    buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
    buf.extend_from_slice(&(rows as u64).to_le_bytes());
    buf.extend_from_slice(&(cols as u64).to_le_bytes());
    buf.extend_from_slice(&data_type.to_le_bytes());

    let out_offset_pos = buf.len();
    buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder for offset

    // Alignment to 32-byte boundary before data section
    let current_len = buf.len();
    let padding = (GGUF_ALIGNMENT - (current_len % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;
    buf.resize(current_len + padding, 0);

    // Write tok_embeddings data
    let tok_offset = buf.len() as u64;
    buf[tok_offset_pos..tok_offset_pos + 8].copy_from_slice(&tok_offset.to_le_bytes());
    buf.extend_from_slice(tensor_data);

    // Write output.weight data (reuse same quantization data)
    let out_offset = buf.len() as u64;
    buf[out_offset_pos..out_offset_pos + 8].copy_from_slice(&out_offset.to_le_bytes());
    buf.extend_from_slice(tensor_data);

    buf
}

/// Write a KV pair with string value
fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&GGUF_VALUE_TYPE_STRING.to_le_bytes());
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

/// Write a KV pair with u32 value (GGUF value type 4)
fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    const GGUF_VALUE_TYPE_U32: u32 = 4;
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&GGUF_VALUE_TYPE_U32.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Write a KV pair with string array value (GGUF value type 9 = array)
/// Creates an array of empty strings to represent vocabulary tokens
fn write_kv_string_array(buf: &mut Vec<u8>, key: &str, count: usize) {
    const GGUF_VALUE_TYPE_ARRAY: u32 = 9;

    // Write key
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());

    // Write value type: array
    buf.extend_from_slice(&GGUF_VALUE_TYPE_ARRAY.to_le_bytes());

    // Write array element type: string (type 8)
    buf.extend_from_slice(&GGUF_VALUE_TYPE_STRING.to_le_bytes());

    // Write array length
    buf.extend_from_slice(&(count as u64).to_le_bytes());

    // Write array elements (empty strings for minimal fixture)
    for _ in 0..count {
        buf.extend_from_slice(&0u64.to_le_bytes()); // string length 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qk256_4x256_fixture_size() {
        let fixture = generate_qk256_4x256(42);
        assert!(fixture.len() > 8);
        assert_eq!(&fixture[0..4], b"GGUF");
        let version = u32::from_le_bytes([fixture[4], fixture[5], fixture[6], fixture[7]]);
        assert_eq!(version, 3);
        assert!(fixture.len() >= 256);
    }

    #[test]
    fn test_bitnet32_2x64_fixture_size() {
        let fixture = generate_bitnet32_2x64(43);
        assert!(fixture.len() > 8);
        assert_eq!(&fixture[0..4], b"GGUF");
        let version = u32::from_le_bytes([fixture[4], fixture[5], fixture[6], fixture[7]]);
        assert_eq!(version, 3);
        assert!(fixture.len() >= 40);
    }

    #[test]
    fn test_qk256_3x300_fixture_size() {
        let fixture = generate_qk256_3x300(44);
        assert!(fixture.len() > 8);
        assert_eq!(&fixture[0..4], b"GGUF");
        let version = u32::from_le_bytes([fixture[4], fixture[5], fixture[6], fixture[7]]);
        assert_eq!(version, 3);
        assert!(fixture.len() >= 384);
    }

    #[test]
    fn test_deterministic_generation() {
        let fixture1 = generate_qk256_4x256(42);
        let fixture2 = generate_qk256_4x256(42);
        assert_eq!(fixture1, fixture2);

        let fixture3 = generate_qk256_4x256(43);
        assert_ne!(fixture1, fixture3);
    }
}
