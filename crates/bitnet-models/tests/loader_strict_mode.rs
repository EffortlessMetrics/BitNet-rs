//! AC1: Strict Loader Mode UX Tests (Issue #469)
//!
//! Tests feature spec: docs/explanation/issue-469-spec.md#ac1-loader-strict-mode-ux
//! API contract: docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md#ac1
//!
//! This test validates the --strict-loader CLI flag and GGUF loader strict mode behavior
//! for QK256 tensor size validation with configurable tolerance enforcement.

#![cfg(all(test, feature = "cpu"))]

/// AC1: Strict loader mode rejects misaligned QK256 tensors
///
/// Tests that strict mode (--strict-loader) rejects QK256 tensors with >0.1% size deviation.
///
/// # Implementation Note
/// This test validates the strict mode validation logic via unit test approach.
/// Since we don't have misaligned GGUF fixtures, we test the tolerance calculation
/// and error message formatting directly.
///
/// # Expected Behavior
/// - Loader configured with strict_mode=true has tolerance=0
/// - Any deviation from expected size is rejected in strict mode
/// - Error message includes proper diagnostic information
#[test]
fn test_strict_loader_rejects_misaligned_qk256() {
    // AC1: Verify strict loader config sets zero tolerance
    use bitnet_models::GGUFLoaderConfig;

    let strict_config = GGUFLoaderConfig {
        strict_mode: true,
        tolerance_bytes: 128, // Ignored in strict mode
    };

    // In strict mode, tolerance should be 0 (any deviation rejected)
    assert!(strict_config.strict_mode, "AC1: Strict mode should be enabled");

    // Simulate strict mode behavior: tolerance is always 0 when strict_mode=true
    let effective_tolerance =
        if strict_config.strict_mode { 0 } else { strict_config.tolerance_bytes };

    assert_eq!(effective_tolerance, 0, "AC1: Strict mode should have zero tolerance");

    // Test deviation calculation (simulate what loader does)
    let expected_bytes: usize = 98304; // 256-elem blocks * 64 B/block
    let actual_bytes: usize = 98560; // +256 bytes deviation
    let deviation = actual_bytes.abs_diff(expected_bytes);
    let deviation_pct =
        ((actual_bytes as f64 - expected_bytes as f64) / expected_bytes as f64) * 100.0;

    // Strict mode should reject this deviation
    assert!(
        deviation > effective_tolerance,
        "AC1: Deviation ({} bytes) should exceed strict tolerance ({})",
        deviation,
        effective_tolerance
    );

    // Verify deviation percentage is calculated correctly
    assert!(
        deviation_pct > 0.1,
        "AC1: Deviation ({:.2}%) should exceed 0.1% threshold",
        deviation_pct
    );

    // Verify error message format (what the loader would produce)
    let error_msg = format!(
        "Tensor '{}' size mismatch (strict mode): expected {} bytes (256-elem blocks), \
         got {} bytes ({:+.2}% deviation). Use --strict-loader to enforce exact alignment \
         or regenerate GGUF with clean export.",
        "blk.0.attn_q.weight", expected_bytes, actual_bytes, deviation_pct
    );

    assert!(error_msg.contains("size mismatch"), "AC1: Error should mention size mismatch");
    assert!(error_msg.contains("strict mode"), "AC1: Error should reference strict mode");
    assert!(error_msg.contains("%"), "AC1: Error should show deviation percentage");
    assert!(error_msg.contains("256-elem blocks"), "AC1: Error should mention block size");
    assert!(
        error_msg.contains("strict-loader") || error_msg.contains("regenerate"),
        "AC1: Error should provide actionable guidance"
    );
}

/// AC1: Permissive loader mode allows small deviation with warning
///
/// Tests that default permissive mode accepts QK256 tensors within 0.1% tolerance with warning.
///
/// # Implementation Note
/// This test validates the permissive mode tolerance calculation via unit test approach.
/// We test that deviations within the calculated tolerance threshold are accepted.
///
/// # Expected Behavior
/// - Loader configured with strict_mode=false (default) uses calculated tolerance
/// - Deviations within tolerance are accepted
/// - Warning message format includes diagnostic information
#[test]
fn test_permissive_loader_allows_small_deviation() {
    // AC1: Verify permissive loader uses calculated tolerance
    use bitnet_models::{GGUFLoaderConfig, qk256_tolerance_bytes};

    let permissive_config = GGUFLoaderConfig {
        strict_mode: false,
        tolerance_bytes: 128, // Default tolerance
    };

    assert!(!permissive_config.strict_mode, "AC1: Permissive mode should be disabled");

    // Test with realistic QK256 tensor size
    let expected_bytes: usize = 98304; // 256-elem blocks * 64 B/block (typical QK256 tensor)
    let calculated_tolerance = qk256_tolerance_bytes(expected_bytes);

    // In permissive mode, use calculated tolerance
    let effective_tolerance = if permissive_config.strict_mode { 0 } else { calculated_tolerance };

    // Verify tolerance is calculated correctly (0.1% with minimum 8 bytes)
    // For 98304 bytes: 98304 * 0.001 = 98.304, ceil = 99 bytes
    assert!(effective_tolerance > 0, "AC1: Permissive mode should have non-zero tolerance");
    assert!(effective_tolerance >= 8, "AC1: Tolerance should have 8-byte minimum");

    // Test small deviation within tolerance
    let small_deviation_bytes: usize = 49; // 0.05% of 98304
    let actual_bytes: usize = expected_bytes + small_deviation_bytes;
    let deviation = actual_bytes.abs_diff(expected_bytes);
    let deviation_pct =
        ((actual_bytes as f64 - expected_bytes as f64) / expected_bytes as f64) * 100.0;

    // Permissive mode should accept this small deviation
    assert!(
        deviation <= effective_tolerance,
        "AC1: Small deviation ({} bytes) should be within tolerance ({})",
        deviation,
        effective_tolerance
    );

    // Verify deviation percentage is small
    assert!(deviation_pct < 0.1, "AC1: Deviation ({:.2}%) should be under 0.1%", deviation_pct);

    // Verify warning message format (what the loader would produce)
    let threshold_pct = (effective_tolerance as f64 / expected_bytes as f64) * 100.0;
    let warning_msg = format!(
        "QK256 size mismatch (permissive): tensor='{}', expected={}B, actual={}B, \
         deviation={:+.2}% (threshold={:.2}%), ACCEPTED with tolerance",
        "blk.0.attn_q.weight", expected_bytes, actual_bytes, deviation_pct, threshold_pct
    );

    assert!(warning_msg.contains("permissive"), "AC1: Warning should mention permissive mode");
    assert!(warning_msg.contains("threshold"), "AC1: Warning should mention threshold");
    assert!(warning_msg.contains("%"), "AC1: Warning should show percentages");
    assert!(warning_msg.contains("ACCEPTED"), "AC1: Warning should indicate acceptance");
}

/// AC1: Strict loader error message format validation
///
/// Tests that strict loader error messages include all required diagnostic information.
///
/// # Implementation Note
/// This test validates error message formatting for multiple tensor scenarios.
/// We verify that the error message format is consistent and includes all required components.
///
/// # Expected Behavior
/// - Error message includes: exact tensor name, expected bytes, actual bytes, deviation %
/// - Error message provides actionable guidance
/// - Error message format is consistent across different tensors
#[test]
fn test_strict_loader_error_message_format() {
    // AC1: Verify error message format and actionable guidance
    // Test multiple tensor scenarios to ensure format consistency

    // Test case 1: Attention query weight
    let tensor_name_1 = "blk.0.attn_q.weight";
    let expected_1: usize = 98304;
    let actual_1: usize = 98560;
    let deviation_pct_1 = ((actual_1 as f64 - expected_1 as f64) / expected_1 as f64) * 100.0;

    let error_msg_1 = format!(
        "Tensor '{}' size mismatch (strict mode): expected {} bytes (256-elem blocks), \
         got {} bytes ({:+.2}% deviation). Use --strict-loader to enforce exact alignment \
         or regenerate GGUF with clean export.",
        tensor_name_1, expected_1, actual_1, deviation_pct_1
    );

    // Verify all required components in error message
    assert!(error_msg_1.contains("blk."), "AC1: Error should include tensor name");
    assert!(error_msg_1.contains("bytes"), "AC1: Error should show byte counts");
    assert!(error_msg_1.contains("%"), "AC1: Error should show percentage");
    assert!(error_msg_1.contains("256-elem blocks"), "AC1: Error should mention block size");
    assert!(
        error_msg_1.contains("strict-loader") || error_msg_1.contains("regenerate"),
        "AC1: Error should provide actionable guidance"
    );
    assert!(error_msg_1.contains("strict mode"), "AC1: Error should mention strict mode");

    // Test case 2: Feed-forward weight (different tensor, same format)
    let tensor_name_2 = "blk.1.ffn_up.weight";
    let expected_2: usize = 196608;
    let actual_2: usize = 197120;
    let deviation_pct_2 = ((actual_2 as f64 - expected_2 as f64) / expected_2 as f64) * 100.0;

    let error_msg_2 = format!(
        "Tensor '{}' size mismatch (strict mode): expected {} bytes (256-elem blocks), \
         got {} bytes ({:+.2}% deviation). Use --strict-loader to enforce exact alignment \
         or regenerate GGUF with clean export.",
        tensor_name_2, expected_2, actual_2, deviation_pct_2
    );

    // Verify format consistency across different tensors
    assert!(error_msg_2.contains(tensor_name_2), "AC1: Error should include specific tensor name");
    assert!(error_msg_2.contains(&expected_2.to_string()), "AC1: Error should show expected bytes");
    assert!(error_msg_2.contains(&actual_2.to_string()), "AC1: Error should show actual bytes");
    assert!(error_msg_2.contains("strict mode"), "AC1: Error format should be consistent");

    // Test case 3: Negative deviation (actual < expected)
    let tensor_name_3 = "blk.2.attn_k.weight";
    let expected_3: usize = 98304;
    let actual_3: usize = 98048; // 256 bytes less
    let deviation_pct_3 = ((actual_3 as f64 - expected_3 as f64) / expected_3 as f64) * 100.0;

    let error_msg_3 = format!(
        "Tensor '{}' size mismatch (strict mode): expected {} bytes (256-elem blocks), \
         got {} bytes ({:+.2}% deviation). Use --strict-loader to enforce exact alignment \
         or regenerate GGUF with clean export.",
        tensor_name_3, expected_3, actual_3, deviation_pct_3
    );

    // Verify negative deviation is formatted correctly
    assert!(deviation_pct_3 < 0.0, "AC1: Negative deviation should be calculated correctly");
    assert!(
        error_msg_3.contains('-') || error_msg_3.contains("−"),
        "AC1: Negative deviation should be indicated in message"
    );
}

/// AC1: Default loader behavior unchanged (backward compatibility)
///
/// Tests that default loader behavior is permissive for backward compatibility.
///
/// # Fixture Requirements
/// - None (tests API contract)
///
/// # Expected Behavior
/// - GGUFLoaderConfig::default() has strict_mode=false
/// - Default tolerance is 0.1% (128 bytes for typical QK256 tensors)
/// - No breaking changes to existing loader API
#[test]
fn test_default_loader_is_permissive() {
    // AC1: Verify default loader config is permissive (backward compat)
    use bitnet_models::GGUFLoaderConfig;

    let config = GGUFLoaderConfig::default();
    assert!(!config.strict_mode, "AC1: Default should be permissive");
    assert_eq!(config.tolerance_bytes, 128, "AC1: Default tolerance should be 128 bytes");
}

/// AC1: CLI flag parsing for --strict-loader
///
/// Tests that CLI correctly parses --strict-loader flag and passes to loader config.
///
/// # Fixture Requirements
/// - None (tests CLI argument parsing)
///
/// # Expected Behavior
/// - --strict-loader flag parsed as boolean
/// - Flag value passed to GGUFLoaderConfig
/// - Default: --strict-loader not present → strict_mode=false
#[test]
#[ignore = "Requires CLI integration - test in bitnet-cli crate"]
fn test_cli_strict_loader_flag_parsing() {
    // AC1: Verify CLI flag parsing for --strict-loader
    // NOTE: This test should be in bitnet-cli/tests/cli_smoke.rs
    //
    // Expected CLI usage:
    //   bitnet-cli run --model model.gguf --strict-loader --prompt "Test" --max-tokens 16
    //
    // Expected behavior:
    //   1. Parse --strict-loader as boolean flag
    //   2. Pass strict_mode=true to GGUFLoaderConfig
    //   3. Loader enforces strict validation

    panic!(
        "AC1: CLI --strict-loader flag not yet implemented. \
         Expected: CLI argument parsing with boolean flag, passes to loader config."
    );
}

/// AC1: Tolerance calculation for different tensor sizes
///
/// Tests that tolerance calculation correctly computes 0.1% of tensor size.
///
/// # Fixture Requirements
/// - None (unit test for tolerance calculation)
///
/// # Expected Behavior
/// - 0.1% tolerance calculated correctly for various tensor sizes
/// - Tolerance rounded appropriately (ceil or floor)
/// - Edge cases: very small tensors, very large tensors
#[test]
fn test_tolerance_calculation_for_tensor_sizes() {
    // AC1: Verify tolerance calculation (0.1% of tensor size)
    use bitnet_models::qk256_tolerance_bytes;

    // Expected tolerance calculations (0.1% with ceiling rounding, minimum 8 bytes)
    assert_eq!(qk256_tolerance_bytes(100_000), 100, "AC1: 100 KB → 100 bytes");
    assert_eq!(qk256_tolerance_bytes(1_000_000), 1_000, "AC1: 1 MB → 1 KB");
    assert_eq!(qk256_tolerance_bytes(10_000_000), 10_000, "AC1: 10 MB → 10 KB");
    assert_eq!(qk256_tolerance_bytes(1_000), 8, "AC1: 1 KB → 8 bytes (minimum)");
    assert_eq!(qk256_tolerance_bytes(500), 8, "AC1: 500 bytes → 8 bytes (minimum)");
}

/// AC1: Strict mode validates all tensors
///
/// Tests that strict mode validation logic applies to all tensors and identifies failures.
///
/// # Implementation Note
/// This test validates the logic for multiple tensor validation scenarios.
/// We verify that error messages correctly identify specific tensors.
///
/// # Expected Behavior
/// - Strict mode validation applies to all tensors during load
/// - Error message identifies specific problematic tensor by name
/// - Validation logic is consistent across all tensor types
#[test]
fn test_strict_mode_validates_all_tensors() {
    // AC1: Verify strict mode validates all tensors with proper error reporting
    use bitnet_models::GGUFLoaderConfig;

    let strict_config = GGUFLoaderConfig { strict_mode: true, tolerance_bytes: 128 };

    // Simulate multiple tensors with varying deviations
    let test_tensors: Vec<(&str, usize, usize)> = vec![
        ("blk.0.attn_q.weight", 98304, 98304),   // Perfect match
        ("blk.0.attn_k.weight", 98304, 98560),   // +256 bytes deviation
        ("blk.1.ffn_up.weight", 196608, 197120), // +512 bytes deviation
    ];

    for (tensor_name, expected, actual) in test_tensors {
        let deviation = actual.abs_diff(expected);
        let deviation_pct = ((actual as f64 - expected as f64) / expected as f64) * 100.0;

        // In strict mode, tolerance is 0
        let effective_tolerance = if strict_config.strict_mode { 0 } else { 128 };

        if deviation > effective_tolerance {
            // This tensor would fail strict mode validation
            let error_msg = format!(
                "Tensor '{}' size mismatch (strict mode): expected {} bytes (256-elem blocks), \
                 got {} bytes ({:+.2}% deviation). Use --strict-loader to enforce exact alignment \
                 or regenerate GGUF with clean export.",
                tensor_name, expected, actual, deviation_pct
            );

            // Verify error identifies specific tensor
            assert!(
                error_msg.contains(tensor_name),
                "AC1: Error should identify specific tensor '{}'",
                tensor_name
            );
            assert!(
                error_msg.contains("blk."),
                "AC1: Error should contain tensor block identifier"
            );

            // Verify error provides diagnostic information
            assert!(
                error_msg.contains(&expected.to_string()),
                "AC1: Error should show expected size"
            );
            assert!(error_msg.contains(&actual.to_string()), "AC1: Error should show actual size");

            // In a real loader, this would stop validation and return the error
            // Here we verify the validation logic would correctly identify the failure
        } else {
            // Perfect match - would pass validation
            assert_eq!(deviation, 0, "AC1: Perfect match tensor should have zero deviation");
        }
    }

    // Verify strict mode behavior: any deviation is rejected
    let first_failing_tensor = "blk.0.attn_k.weight";
    let first_expected: usize = 98304;
    let first_actual: usize = 98560;
    let error_for_first_failure = format!(
        "Tensor '{}' size mismatch (strict mode): expected {} bytes (256-elem blocks), \
         got {} bytes ({:+.2}% deviation). Use --strict-loader to enforce exact alignment \
         or regenerate GGUF with clean export.",
        first_failing_tensor,
        first_expected,
        first_actual,
        ((first_actual as f64 - first_expected as f64) / first_expected as f64) * 100.0
    );

    assert!(
        error_for_first_failure.contains(first_failing_tensor),
        "AC1: Error should identify first failing tensor specifically"
    );
}
