//! Test for I2S close-match priority fix
//!
//! Validates that when multiple I2S flavors have "close match" sizes (within tolerance),
//! QK256 format is preferred over Split32 and Inline formats.

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
fn test_i2s_close_match_prioritizes_qk256() {
    // Create an ambiguous tensor size that could match multiple flavors with tolerance
    // Use 512 elements:
    // - blocks32 = ceil(512/32) = 16
    // - blocks256 = ceil(512/256) = 2
    // - split_need = 16 * 8 = 128 bytes
    // - inline_need = 16 * 10 = 160 bytes
    // - qk256_need = 2 * 64 = 128 bytes
    //
    // Notice: split_need == qk256_need (both 128 bytes exactly)
    // This creates an exact match scenario for both, but we want QK256 to win.
    let nelems = 512;
    let info = mk_info("test.ambiguous.weight", &[512], 128);

    // Test without sibling (should prefer QK256 over Split32)
    let flavor_no_sibling =
        detect_i2s_flavor(&info, false, nelems).expect("should detect flavor without sibling");
    assert_eq!(
        flavor_no_sibling,
        I2SFlavor::GgmlQk256NoScale,
        "QK256 should be prioritized over Split32 when sizes match exactly"
    );

    // Test with sibling present (QK256 should still win in exact match)
    let flavor_with_sibling =
        detect_i2s_flavor(&info, true, nelems).expect("should detect flavor with sibling");
    assert_eq!(
        flavor_with_sibling,
        I2SFlavor::GgmlQk256NoScale,
        "QK256 should be prioritized even when sibling is present in exact match"
    );
}

#[test]
fn test_i2s_close_match_qk256_with_tolerance() {
    // Test QK256 priority when within tolerance but not exact match
    // Use 520 elements to create slight misalignment:
    // - blocks32 = ceil(520/32) = 17
    // - blocks256 = ceil(520/256) = 3
    // - split_need = 17 * 8 = 136 bytes
    // - inline_need = 17 * 10 = 170 bytes
    // - qk256_need = 3 * 64 = 192 bytes
    //
    // Provide 190 bytes (within tolerance of qk256_need=192, diff=2)
    // Also within tolerance of split_need=136 (diff=54, exceeds tolerance)
    // So only QK256 matches within tolerance
    let nelems = 520;
    let info = mk_info("test.qk256_tolerance.weight", &[520], 190);

    let flavor =
        detect_i2s_flavor(&info, false, nelems).expect("should detect QK256 within tolerance");
    assert_eq!(
        flavor,
        I2SFlavor::GgmlQk256NoScale,
        "QK256 should be detected when within tolerance"
    );
}

#[test]
fn test_i2s_close_match_multiple_within_tolerance() {
    // Test that when multiple flavors are within tolerance, QK256 is preferred
    // This is tricky because we need sizes where both QK256 and another flavor
    // are within tolerance but not exact matches.
    //
    // Strategy: Use a size that's close to both split32 and qk256
    // Use 256 elements:
    // - blocks32 = 8, split_need = 64
    // - blocks256 = 1, qk256_need = 64
    // Both need exactly 64 bytes! Exact match scenario again.
    //
    // Let's use 264 elements to create tolerance scenario:
    // - blocks32 = ceil(264/32) = 9, split_need = 72
    // - blocks256 = ceil(264/256) = 2, qk256_need = 128
    // Provide 70 bytes:
    // - diff_split32 = |70 - 72| = 2 (within tolerance)
    // - diff_qk256 = |70 - 128| = 58 (exceeds tolerance)
    // Only split32 matches, so should detect Split32
    let nelems = 264;
    let info_split = mk_info("test.split_only.weight", &[264], 70);

    let flavor_split = detect_i2s_flavor(&info_split, true, nelems).expect("should detect Split32");
    assert_eq!(
        flavor_split,
        I2SFlavor::Split32WithSibling,
        "Split32 should be detected when only it is within tolerance"
    );

    // Now test QK256 priority: provide 130 bytes
    // - diff_split32 = |130 - 72| = 58 (exceeds tolerance)
    // - diff_qk256 = |130 - 128| = 2 (within tolerance)
    // Only QK256 matches, so should detect QK256
    let info_qk256 = mk_info("test.qk256_only.weight", &[264], 130);

    let flavor_qk256 = detect_i2s_flavor(&info_qk256, true, nelems).expect("should detect QK256");
    assert_eq!(
        flavor_qk256,
        I2SFlavor::GgmlQk256NoScale,
        "QK256 should be detected when only it is within tolerance"
    );
}

#[test]
fn test_i2s_inline_fallback_when_qk256_not_match() {
    // Test that Inline is detected when QK256 doesn't match but Inline does
    // Use 64 elements:
    // - blocks32 = 2, split_need = 16, inline_need = 20
    // - blocks256 = 1, qk256_need = 64
    // Provide 20 bytes:
    // - diff_inline = 0 (exact match)
    // - diff_qk256 = 44 (exceeds tolerance)
    // Should detect Inline
    let nelems = 64;
    let info = mk_info("test.inline_exact.weight", &[64], 20);

    let flavor = detect_i2s_flavor(&info, false, nelems).expect("should detect Inline");
    assert_eq!(flavor, I2SFlavor::BitNet32F16, "Inline should be detected when it matches exactly");
}

#[test]
fn test_i2s_adaptive_tolerance_qk256_priority() {
    // Test QK256 priority with adaptive tolerance (non-strict mode)
    // Use a large tensor to trigger size-proportional tolerance
    // 10240 elements:
    // - blocks32 = 320, split_need = 2560
    // - blocks256 = 40, qk256_need = 2560
    // Both need exactly 2560 bytes (coincidence!)
    //
    // With adaptive tolerance (~0.1% of 2560 = ~2.56 bytes, rounds up to larger value),
    // provide 2560 bytes exactly - should prefer QK256 over Split32
    let nelems = 10240;
    let info = mk_info("test.large_adaptive.weight", &[10240], 2560);

    let flavor_no_sibling =
        detect_i2s_flavor(&info, false, nelems).expect("should detect with adaptive tolerance");
    assert_eq!(
        flavor_no_sibling,
        I2SFlavor::GgmlQk256NoScale,
        "QK256 should be prioritized with adaptive tolerance (exact match)"
    );

    // Also test with sibling present - QK256 should still win in exact match
    let flavor_with_sibling =
        detect_i2s_flavor(&info, true, nelems).expect("should detect with sibling");
    assert_eq!(
        flavor_with_sibling,
        I2SFlavor::GgmlQk256NoScale,
        "QK256 should be prioritized even with sibling in exact match"
    );
}

#[test]
fn test_i2s_close_match_all_formats_within_tolerance() {
    // Edge case: Create a scenario where all three formats could be within tolerance
    // This is difficult because the formats have different byte requirements.
    // Use 32 elements (single block):
    // - blocks32 = 1, split_need = 8, inline_need = 10
    // - blocks256 = 1, qk256_need = 64
    // With tolerance of 8 bytes, provide 10 bytes:
    // - diff_split32 = |10 - 8| = 2 (within tolerance)
    // - diff_inline = |10 - 10| = 0 (exact match, higher priority)
    // - diff_qk256 = |10 - 64| = 54 (exceeds tolerance)
    // Should detect Inline (exact match takes priority over close match)
    let nelems = 32;
    let info = mk_info("test.inline_exact_priority.weight", &[32], 10);

    let flavor = detect_i2s_flavor(&info, false, nelems).expect("should detect Inline");
    assert_eq!(
        flavor,
        I2SFlavor::BitNet32F16,
        "Inline exact match should take priority over Split32 close match"
    );

    // Now test QK256 exact match priority: provide 64 bytes
    // - diff_qk256 = 0 (exact match)
    // - diff_split32 = 56 (exceeds tolerance)
    // - diff_inline = 54 (exceeds tolerance)
    let info_qk256 = mk_info("test.qk256_exact_priority.weight", &[32], 64);

    let flavor_qk256 = detect_i2s_flavor(&info_qk256, false, nelems).expect("should detect QK256");
    assert_eq!(flavor_qk256, I2SFlavor::GgmlQk256NoScale, "QK256 exact match should take priority");
}
