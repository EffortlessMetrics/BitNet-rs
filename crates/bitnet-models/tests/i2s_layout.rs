//! Synthetic tests for I2_S layout detection and loading
//! Tests both GGML split (data + separate scales) and inline f16 layouts

use bitnet_models::formats::gguf::{GgufTensorType, I2SLayoutKind, TensorInfo};

/// Helper to create a minimal TensorInfo for testing
fn mk_info(name: &str, shape: &[usize], t: GgufTensorType) -> TensorInfo {
    TensorInfo { name: name.into(), shape: shape.to_vec(), tensor_type: t, offset: 0, size: 0 }
}

/// Test inline f16 layout: 64 elements -> 2 blocks of 32
/// Each block: 8B packed data + 2B f16 scale = 10B per block
/// Total: 2 * 10 = 20 bytes
#[test]
fn i2s_inline_f16_10b_blocks() {
    let _info = mk_info("w.ffn_up", &[64], GgufTensorType::I2_S);
    let mut data = [0u8; 20];

    // Fill fake packed+scale pairs per block
    for b in 0..2 {
        let off = b * 10;
        // 8 bytes of packed data (fake pattern)
        data[off..off + 8].copy_from_slice(&[0xAA; 8]);
        // 2 bytes f16 scale = 1.0
        data[off + 8..off + 10].copy_from_slice(&0x3C00u16.to_le_bytes());
    }

    // Test block size detection
    let blocks = 64_usize.div_ceil(I2SLayoutKind::InlineF16.block_size());
    assert_eq!(blocks, 2);

    let need = blocks * I2SLayoutKind::InlineF16.total_bytes_per_block();
    assert_eq!(need, 20);
    assert_eq!(data.len(), need);
}

/// Test GGML split layout: 64 elements -> 2 blocks of 32
/// Data: 2 * 8 = 16 bytes (packed only)
/// Scales: 2 f32 values = 8 bytes (stored separately)
#[test]
fn i2s_ggml_split_8b_data_separate_scales() {
    let _info = mk_info("blk.0.wv.data", &[64], GgufTensorType::I2_S);

    // Data only: 8 bytes per block
    let data = [0x55u8; 16];

    // Scales stored separately (simulated)
    let scales = [1.0f32, 2.0f32];

    // Test block size detection
    let blocks = 64_usize.div_ceil(I2SLayoutKind::GgmlSplit.block_size());
    assert_eq!(blocks, 2);

    let data_need = blocks * I2SLayoutKind::GgmlSplit.data_bytes_per_block();
    assert_eq!(data_need, 16);
    assert_eq!(data.len(), data_need);
    assert_eq!(scales.len(), blocks);
}

/// Test layout detection constants
#[test]
fn i2s_layout_constants() {
    // Both layouts use 32-element blocks
    assert_eq!(I2SLayoutKind::GgmlSplit.block_size(), 32);
    assert_eq!(I2SLayoutKind::InlineF16.block_size(), 32);

    // Both use 8 bytes of packed data per block
    assert_eq!(I2SLayoutKind::GgmlSplit.data_bytes_per_block(), 8);
    assert_eq!(I2SLayoutKind::InlineF16.data_bytes_per_block(), 8);

    // Total bytes differ due to inline scale
    assert_eq!(I2SLayoutKind::GgmlSplit.total_bytes_per_block(), 8); // data only
    assert_eq!(I2SLayoutKind::InlineF16.total_bytes_per_block(), 10); // data + f16 scale
}

/// Test edge case: single block (32 elements)
#[test]
fn i2s_single_block() {
    // Inline layout: 1 block * 10 bytes = 10 bytes
    let blocks_inline = 32_usize.div_ceil(I2SLayoutKind::InlineF16.block_size());
    assert_eq!(blocks_inline, 1);
    let need_inline = blocks_inline * I2SLayoutKind::InlineF16.total_bytes_per_block();
    assert_eq!(need_inline, 10);

    // GGML split: 1 block * 8 bytes data = 8 bytes
    let blocks_split = 32_usize.div_ceil(I2SLayoutKind::GgmlSplit.block_size());
    assert_eq!(blocks_split, 1);
    let need_split = blocks_split * I2SLayoutKind::GgmlSplit.data_bytes_per_block();
    assert_eq!(need_split, 8);
}

/// Test realistic model size: 2048 hidden dimension
#[test]
fn i2s_realistic_model_size() {
    let hidden_size: usize = 2048;
    let blocks = hidden_size.div_ceil(32);
    assert_eq!(blocks, 64);

    // Inline: 64 blocks * 10 bytes = 640 bytes
    let inline_bytes = blocks * I2SLayoutKind::InlineF16.total_bytes_per_block();
    assert_eq!(inline_bytes, 640);

    // GGML split: 64 blocks * 8 bytes data = 512 bytes
    let split_data_bytes = blocks * I2SLayoutKind::GgmlSplit.data_bytes_per_block();
    assert_eq!(split_data_bytes, 512);

    // Plus 64 f32 scales = 256 bytes (stored separately)
    let split_scale_bytes = blocks * 4; // f32 = 4 bytes
    assert_eq!(split_scale_bytes, 256);
}

/// Test non-aligned element count (not a multiple of 32)
#[test]
fn i2s_non_aligned_elements() {
    // 100 elements -> needs 4 blocks (ceil(100/32) = 4)
    let nelems: usize = 100;
    let blocks = nelems.div_ceil(32);
    assert_eq!(blocks, 4);

    // Inline: 4 blocks * 10 bytes = 40 bytes
    let inline_bytes = blocks * I2SLayoutKind::InlineF16.total_bytes_per_block();
    assert_eq!(inline_bytes, 40);

    // GGML split: 4 blocks * 8 bytes = 32 bytes data
    let split_bytes = blocks * I2SLayoutKind::GgmlSplit.data_bytes_per_block();
    assert_eq!(split_bytes, 32);
}
