//! Tests for I2S flavor detection and sibling scale search
//!
//! Validates the I2SFlavor enum and detect_i2s_flavor function with:
//! - BitNet32F16 (inline f16 scales)
//! - Split32WithSibling (separate scale tensor)
//! - GgmlQk256NoScale (GGML format)
//! - Fail-closed behavior for unsupported variants

use bitnet_models::formats::gguf::{GgufTensorType, I2SFlavor, TensorInfo, detect_i2s_flavor};

/// Helper to create a minimal TensorInfo for testing
fn mk_info(name: &str, shape: &[usize], size: u64) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        shape: shape.to_vec(),
        tensor_type: GgufTensorType::I2_S,
        offset: 0,
        size,
    }
}

#[test]
fn detect_bitnet32f16_inline_scales() {
    // 64 elements → 2 blocks of 32
    // Each block: 8B packed data + 2B f16 scale = 10B
    // Total: 2 * 10 = 20 bytes
    let nelems = 64;
    let info = mk_info("blk.0.attn_q.weight", &[64], 20);

    let flavor = detect_i2s_flavor(&info, false, nelems).expect("should detect BitNet32F16");

    assert_eq!(flavor, I2SFlavor::BitNet32F16);
    assert_eq!(flavor.block_size(), 32);
    assert_eq!(flavor.total_bytes_per_block(), 10);
}

#[test]
fn detect_split32_with_sibling() {
    // 64 elements → 2 blocks of 32
    // Data only: 2 * 8 = 16 bytes
    // Scales stored separately (indicated by has_scale_sibling=true)
    let nelems = 64;
    let info = mk_info("blk.0.attn_k.weight", &[64], 16);

    let flavor = detect_i2s_flavor(&info, true, nelems).expect("should detect Split32WithSibling");

    assert_eq!(flavor, I2SFlavor::Split32WithSibling);
    assert_eq!(flavor.block_size(), 32);
    assert_eq!(flavor.total_bytes_per_block(), 8);
}

#[test]
fn detect_split32_without_sibling_warns() {
    // 128 elements → 4 blocks of 32
    // blocks32 = 4, split_need = 4*8 = 32 bytes, inline_need = 4*10 = 40 bytes
    // blocks256 = 1, qk256_need = 1*64 = 64 bytes
    // Provide exactly 32 bytes (split format) - outside inline tolerance (40-8=32, but at boundary)
    // Actually 40-32=8, so still within tolerance! Let's use smaller element count.
    // Use 160 elements:
    // blocks32 = 5, split_need = 40, inline_need = 50
    // diff(40,50) = 10 > 8, so inline won't match
    // No sibling scale tensor (has_scale_sibling=false) - should warn but still detect Split32
    let nelems = 160;
    let info = mk_info("blk.0.attn_v.weight", &[160], 40);

    let flavor =
        detect_i2s_flavor(&info, false, nelems).expect("should still detect Split32WithSibling");

    assert_eq!(flavor, I2SFlavor::Split32WithSibling);
}

#[test]
fn detect_ggml_qk256_no_scale() {
    // 1024 elements → 4 blocks of 256
    // Each block: 64B packed data (no per-block scales)
    // Total: 4 * 64 = 256 bytes
    // Note: 1024 elements would need ceil(1024/32)=32 blocks for 32-elem format
    // inline_need = 32*10 = 320 bytes (different)
    // split_need = 32*8 = 256 bytes (SAME as qk256!)
    // qk256_need = 4*64 = 256 bytes
    // Priority logic: qk256 preferred when both match due to !matches_split32 in qk256 check
    // WAIT - this will fail because split32 also matches!
    // Let's use 768 elements instead:
    // blocks32 = ceil(768/32) = 24
    // blocks256 = ceil(768/256) = 3
    // inline_need = 24*10 = 240
    // split_need = 24*8 = 192
    // qk256_need = 3*64 = 192 (SAME!)
    // Still conflicts. Need different element count.
    // Use 1536 elements:
    // blocks32 = ceil(1536/32) = 48
    // blocks256 = ceil(1536/256) = 6
    // inline_need = 48*10 = 480
    // split_need = 48*8 = 384
    // qk256_need = 6*64 = 384 (SAME!)
    // This is a fundamental problem - qk256 and split32 sizes can coincide!
    // Solution: Use element counts that don't create coinciding sizes
    // Let's use 512 elements but recognize qk256 has priority:
    // blocks32 = ceil(512/32) = 16
    // blocks256 = ceil(512/256) = 2
    // inline_need = 16*10 = 160
    // split_need = 16*8 = 128
    // qk256_need = 2*64 = 128
    // Both split32 and qk256 match! Priority logic needs fixing.
    let nelems = 512;
    let info = mk_info("blk.0.ffn_gate.weight", &[512], 128);

    let flavor = detect_i2s_flavor(&info, false, nelems).expect("should detect GgmlQk256NoScale");

    assert_eq!(flavor, I2SFlavor::GgmlQk256NoScale);
    assert_eq!(flavor.block_size(), 256);
    assert_eq!(flavor.total_bytes_per_block(), 64);
}

#[test]
fn detect_with_alignment_tolerance() {
    // 64 elements → 2 blocks of 32
    // Expected inline: 2 * 10 = 20 bytes
    // Test with slight padding (within ±8 byte tolerance)
    let nelems = 64;
    let info = mk_info("blk.0.attn_output.weight", &[64], 24); // 20 + 4 padding

    let flavor = detect_i2s_flavor(&info, false, nelems).expect("should detect with tolerance");

    assert_eq!(flavor, I2SFlavor::BitNet32F16);
}

#[test]
fn detect_realistic_model_size() {
    // Realistic hidden dimension: 2048 elements
    // Blocks: 2048 / 32 = 64 blocks
    // Inline format: 64 * 10 = 640 bytes
    let nelems = 2048;
    let info = mk_info("token_embd.weight", &[2048], 640);

    let flavor = detect_i2s_flavor(&info, false, nelems).expect("should detect realistic size");

    assert_eq!(flavor, I2SFlavor::BitNet32F16);
}

#[test]
fn detect_realistic_split_format() {
    // Realistic hidden dimension: 2080 elements (ensures split != inline and != qk256)
    // Blocks32: 2080 / 32 = 65 blocks
    // Blocks256: ceil(2080/256) = 9 blocks
    // Split format: 65 * 8 = 520 bytes (data only)
    // Inline format: 65 * 10 = 650 bytes
    // Qk256 format: 9 * 64 = 576 bytes
    // All different! Has sibling, so should detect Split32WithSibling
    let nelems = 2080;
    let info = mk_info("output.weight", &[2080], 520);

    let flavor = detect_i2s_flavor(&info, true, nelems).expect("should detect realistic split");

    assert_eq!(flavor, I2SFlavor::Split32WithSibling);
}

#[test]
fn detect_non_aligned_elements() {
    // 100 elements → ceil(100/32) = 4 blocks
    // Inline format: 4 * 10 = 40 bytes
    let nelems = 100;
    let info = mk_info("blk.0.ffn_up.weight", &[100], 40);

    let flavor = detect_i2s_flavor(&info, false, nelems).expect("should handle non-aligned");

    assert_eq!(flavor, I2SFlavor::BitNet32F16);
}

#[test]
fn fail_closed_on_invalid_size() {
    // 96 elements → 3 blocks of 32
    // Expected inline: 3 * 10 = 30 bytes
    // Expected split: 3 * 8 = 24 bytes
    // Expected qk256: ceil(96/256)=1 block * 64 = 64 bytes
    // Provide invalid size: 50 bytes (exceeds tolerance of 8 for all)
    let nelems = 96;
    let info = mk_info("blk.0.invalid.weight", &[96], 50);

    let result = detect_i2s_flavor(&info, false, nelems);

    assert!(result.is_err(), "should fail on invalid size");
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("no valid flavor detected"), "error should explain failure");
    assert!(err_msg.contains("available: 50"), "error should show available bytes");
    assert!(err_msg.contains("split_need"), "error should show expected bytes");
    assert!(err_msg.contains("inline_need"), "error should show expected bytes");
}

#[test]
fn priority_qk256_over_split32_coincidence() {
    // Edge case: bytes match multiple flavors due to size coincidence
    // 512 elements:
    // - blocks32 = 16, split_need = 16*8 = 128
    // - blocks256 = 2, qk256_need = 2*64 = 128
    // Both split32 and qk256 need exactly 128 bytes!
    // Priority: qk256 > split32 (qk256 is more specific with larger blocks)
    let nelems = 512;
    let info = mk_info("blk.0.test.weight", &[512], 128);

    let flavor = detect_i2s_flavor(&info, false, nelems).expect("should detect");

    // Should prefer GgmlQk256NoScale over Split32WithSibling when both match
    assert_eq!(flavor, I2SFlavor::GgmlQk256NoScale);
}

#[test]
fn i2s_flavor_block_size_constants() {
    // Verify block size constants for each flavor
    assert_eq!(I2SFlavor::BitNet32F16.block_size(), 32);
    assert_eq!(I2SFlavor::Split32WithSibling.block_size(), 32);
    assert_eq!(I2SFlavor::GgmlQk256NoScale.block_size(), 256);
}

#[test]
fn i2s_flavor_bytes_per_block() {
    // Verify bytes per block for each flavor
    assert_eq!(I2SFlavor::BitNet32F16.data_bytes_per_block(), 8);
    assert_eq!(I2SFlavor::BitNet32F16.total_bytes_per_block(), 10);

    assert_eq!(I2SFlavor::Split32WithSibling.data_bytes_per_block(), 8);
    assert_eq!(I2SFlavor::Split32WithSibling.total_bytes_per_block(), 8);

    assert_eq!(I2SFlavor::GgmlQk256NoScale.data_bytes_per_block(), 64);
    assert_eq!(I2SFlavor::GgmlQk256NoScale.total_bytes_per_block(), 64);
}

#[test]
fn single_block_edge_case() {
    // Single block (32 elements)
    // Inline: 1 * 10 = 10 bytes
    let nelems = 32;
    let info = mk_info("blk.0.single.weight", &[32], 10);

    let flavor = detect_i2s_flavor(&info, false, nelems).expect("should handle single block");

    assert_eq!(flavor, I2SFlavor::BitNet32F16);
}

#[test]
fn large_tensor_256_blocks() {
    // Large tensor with 256-element blocks
    // 1280 elements → 5 blocks of 256
    // blocks32 = ceil(1280/32) = 40
    // blocks256 = ceil(1280/256) = 5
    // split_need = 40*8 = 320 bytes
    // qk256_need = 5*64 = 320 bytes
    // COINCIDENCE AGAIN! But qk256 has priority.
    // Actually, let's use 768 to avoid coincidence:
    // blocks32 = 24, split_need = 24*8 = 192
    // blocks256 = 3, qk256_need = 3*64 = 192
    // STILL coincides! This is a pattern. Use 800:
    // blocks32 = 25, split_need = 25*8 = 200
    // blocks256 = ceil(800/256) = 4, qk256_need = 4*64 = 256 (different!)
    let nelems = 800;
    let info = mk_info("blk.0.large.weight", &[800], 256);

    let flavor = detect_i2s_flavor(&info, false, nelems).expect("should detect GGML format");

    assert_eq!(flavor, I2SFlavor::GgmlQk256NoScale);
}

#[test]
fn tolerance_boundary_positive() {
    // Test tolerance boundary (8 bytes)
    // 64 elements → 2 blocks → inline needs 20 bytes
    // Provide 20 + 8 = 28 bytes (at tolerance limit)
    let nelems = 64;
    let info = mk_info("blk.0.boundary.weight", &[64], 28);

    let flavor =
        detect_i2s_flavor(&info, false, nelems).expect("should accept at tolerance boundary");

    assert_eq!(flavor, I2SFlavor::BitNet32F16);
}

#[test]
fn tolerance_boundary_negative() {
    // Test tolerance boundary (8 bytes)
    // 96 elements → 3 blocks → inline needs 30 bytes
    // Provide 30 + 9 = 39 bytes (exceeds tolerance of 8)
    let nelems = 96;
    let info = mk_info("blk.0.boundary_fail.weight", &[96], 39);

    let result = detect_i2s_flavor(&info, false, nelems);

    assert!(result.is_err(), "should fail beyond tolerance");
}
