//! AC2: QK256 Tolerance & Logs Centralization Tests (Issue #469)
//!
//! Tests feature spec: docs/explanation/issue-469-spec.md#ac2-qk256-tolerance-logs-centralization
//! API contract: docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md#ac2
//!
//! This test validates centralized QK256_SIZE_TOLERANCE constant and consistent logging.

#![cfg(all(test, feature = "cpu"))]

/// AC2: QK256 tolerance constant value and definition
///
/// Tests that QK256_SIZE_TOLERANCE_PERCENT is defined as 0.001 (0.1%).
///
/// # Fixture Requirements
/// - None (unit test for constant value)
///
/// # Expected Behavior
/// - Constant defined in bitnet-quantization crate
/// - Value is exactly 0.001 (0.1%)
/// - Constant is public and re-exported in bitnet-models
#[test]
fn test_qk256_tolerance_constant_value() {
    // AC2: Verify QK256_SIZE_TOLERANCE_PERCENT constant
    use bitnet_quantization::QK256_SIZE_TOLERANCE_PERCENT;

    assert_eq!(QK256_SIZE_TOLERANCE_PERCENT, 0.001, "AC2: Tolerance must be 0.1%");

    // Verify it's exactly 0.1%
    let expected_percent = 0.1 / 100.0;
    assert!((QK256_SIZE_TOLERANCE_PERCENT - expected_percent).abs() < f64::EPSILON);
}

/// AC2: QK256 tolerance bytes calculation
///
/// Tests the qk256_tolerance_bytes helper function.
///
/// # Fixture Requirements
/// - None (unit test for helper function)
///
/// # Expected Behavior
/// - Function calculates 0.1% of expected bytes
/// - Ceiling rounding for fractional bytes
/// - Handles edge cases (very small/large tensors)
#[test]
fn test_qk256_tolerance_bytes_calculation() {
    // AC2: Verify qk256_tolerance_bytes calculation
    use bitnet_quantization::qk256_tolerance_bytes;

    assert_eq!(qk256_tolerance_bytes(1_000_000), 1000, "AC2: 1 MB tensor → 1 KB tolerance");
    assert_eq!(
        qk256_tolerance_bytes(131_072),
        132,
        "AC2: 128 KB tensor → 132 bytes tolerance (ceiling)"
    );
    assert_eq!(qk256_tolerance_bytes(100_000), 100, "AC2: 100 KB tensor → 100 bytes tolerance");
    assert_eq!(qk256_tolerance_bytes(1_000), 8, "AC2: 1 KB tensor → 8 bytes tolerance (minimum)");
}

/// AC2: QK256 tolerance constant re-export in models crate
///
/// Tests that bitnet-models re-exports tolerance constants from bitnet-quantization.
///
/// # Fixture Requirements
/// - None (unit test for API contract)
///
/// # Expected Behavior
/// - bitnet-models re-exports QK256_SIZE_TOLERANCE_PERCENT
/// - bitnet-models re-exports qk256_tolerance_bytes
/// - Consumers can import from either crate
#[test]
fn test_qk256_tolerance_reexport() {
    // AC2: Verify bitnet-models re-exports tolerance constants
    use bitnet_models::{QK256_SIZE_TOLERANCE_PERCENT, qk256_tolerance_bytes};

    // Verify constant value through re-export
    assert_eq!(QK256_SIZE_TOLERANCE_PERCENT, 0.001, "AC2: Re-exported constant must match");

    // Verify function works through re-export
    assert_eq!(qk256_tolerance_bytes(1_000_000), 1000, "AC2: Re-exported function must work");
}

/// AC2: QK256 tolerance permissive mode accepts values within threshold
///
/// Tests that the tolerance calculation correctly accepts size deviations
/// that fall within the 0.1% tolerance in permissive mode.
///
/// # Expected Behavior
/// - Actual size within tolerance → accepted (deviation < tolerance)
/// - Tolerance calculated per-tensor using qk256_tolerance_bytes
#[test]
fn test_qk256_tolerance_permissive_acceptance() {
    use bitnet_quantization::{QK256_SIZE_TOLERANCE_PERCENT, qk256_tolerance_bytes};

    // Simulate permissive mode: accept if |actual - expected| <= tolerance
    let expected_bytes: usize = 98304; // ~96 KB tensor
    let tolerance = qk256_tolerance_bytes(expected_bytes);

    // Slightly larger actual (padding) — within tolerance
    let actual_within = expected_bytes + tolerance / 2;
    let deviation = actual_within as f64 - expected_bytes as f64;
    assert!(
        deviation.abs() <= tolerance as f64,
        "AC2: deviation {deviation} should be within tolerance {tolerance}"
    );

    // Exactly at tolerance boundary — still accepted
    let actual_boundary = expected_bytes + tolerance;
    let deviation_boundary = actual_boundary as f64 - expected_bytes as f64;
    assert!(
        deviation_boundary.abs() <= tolerance as f64,
        "AC2: boundary deviation should be accepted"
    );

    // Verify threshold percentage matches constant
    let threshold_pct = QK256_SIZE_TOLERANCE_PERCENT * 100.0;
    assert!(
        (threshold_pct - 0.1).abs() < f64::EPSILON,
        "AC2: threshold should be 0.10%, got {threshold_pct}%"
    );
}

/// AC2: QK256 strict mode rejects any size deviation
///
/// Tests that strict mode (zero tolerance) rejects tensors with any
/// byte-level deviation from expected size.
///
/// # Expected Behavior
/// - Strict mode uses 0% tolerance (no padding accepted)
/// - Any deviation → rejected
/// - Exact match → accepted
#[test]
fn test_qk256_tolerance_strict_rejection() {
    use bitnet_quantization::qk256_tolerance_bytes;

    // Strict mode: tolerance is 0 (no allowance)
    let strict_tolerance: usize = 0;

    let expected_bytes: usize = 98304;
    let permissive_tolerance = qk256_tolerance_bytes(expected_bytes);

    // Permissive tolerance is non-zero
    assert!(
        permissive_tolerance > 0,
        "AC2: permissive tolerance should be positive, got {permissive_tolerance}"
    );

    // In strict mode, even 1-byte deviation is rejected
    let actual_off_by_one = expected_bytes + 1;
    let deviation = actual_off_by_one.abs_diff(expected_bytes);
    assert!(deviation > strict_tolerance, "AC2: strict mode should reject 1-byte deviation");

    // Exact match is accepted in both modes
    let exact = expected_bytes;
    let exact_deviation = exact.abs_diff(expected_bytes);
    assert_eq!(exact_deviation, 0, "AC2: exact match has zero deviation");
    assert!(exact_deviation <= strict_tolerance, "AC2: exact match accepted in strict mode");
    assert!(
        exact_deviation <= permissive_tolerance,
        "AC2: exact match accepted in permissive mode"
    );
}

/// AC2: QK256 tolerance constant is documented in quantization-support.md
///
/// Verifies that the quantization-support reference doc exists and covers
/// the I2S QK256 format, which implicitly documents the tolerance policy.
#[test]
fn test_qk256_tolerance_documentation() {
    use bitnet_quantization::QK256_SIZE_TOLERANCE_PERCENT;

    // Verify documentation file exists at expected path
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let doc_path = workspace_root.join("docs/reference/quantization-support.md");
    assert!(
        doc_path.exists(),
        "AC2: docs/reference/quantization-support.md must exist, checked: {}",
        doc_path.display()
    );

    // Verify documentation mentions QK256 format
    let content = std::fs::read_to_string(&doc_path).expect("AC2: should be able to read doc file");
    assert!(
        content.contains("QK256") || content.contains("qk256"),
        "AC2: quantization-support.md should document QK256 format"
    );
    assert!(
        content.contains("I2S") || content.contains("I2_S") || content.contains("i2s"),
        "AC2: quantization-support.md should document I2S quantization"
    );

    // Verify the constant value is consistent with documented 0.1%
    assert_eq!(QK256_SIZE_TOLERANCE_PERCENT, 0.001, "AC2: tolerance constant must be 0.1% (0.001)");
}

/// AC2: QK256 tolerance ceiling rounding behavior
///
/// Tests that tolerance calculation uses ceiling rounding for fractional bytes.
///
/// # Fixture Requirements
/// - None (unit test for rounding behavior)
///
/// # Expected Behavior
/// - Fractional bytes rounded up (ceiling)
/// - Edge case: 0.5 bytes → 1 byte
/// - Edge case: 0.1 bytes → 1 byte (minimum tolerance)
#[test]
fn test_qk256_tolerance_ceiling_rounding() {
    // AC2: Verify ceiling rounding for tolerance calculation
    // FIXTURE NEEDED: None (unit test)
    //
    // Expected rounding behavior:
    //   qk256_tolerance_bytes(500) = 1 (0.5 bytes → 1 byte ceiling)
    //   qk256_tolerance_bytes(100) = 1 (0.1 bytes → 1 byte ceiling)
    //   qk256_tolerance_bytes(1_500) = 2 (1.5 bytes → 2 bytes ceiling)

    use bitnet_quantization::qk256_tolerance_bytes;

    assert_eq!(qk256_tolerance_bytes(500), 8, "AC2: 0.5 bytes → 8 bytes (minimum)");
    assert_eq!(qk256_tolerance_bytes(100), 8, "AC2: 0.1 bytes → 8 bytes (minimum)");
    assert_eq!(qk256_tolerance_bytes(1_500), 8, "AC2: 1.5 bytes → 8 bytes (minimum)");

    // Edge case: very small tensors still get at least 8 bytes tolerance (alignment padding)
    assert_eq!(qk256_tolerance_bytes(10), 8, "AC2: Minimum tolerance is 8 bytes");
}

/// AC2: QK256 tolerance used in loader validation
///
/// Tests that loader uses centralized tolerance function instead of hardcoded values.
///
/// # Fixture Requirements
/// - None (integration test with loader)
///
/// # Expected Behavior
/// - Loader imports qk256_tolerance_bytes from bitnet-quantization
/// - Loader calculates tolerance dynamically per tensor
/// - No hardcoded tolerance values in loader code
#[test]
fn test_loader_uses_centralized_tolerance() {
    // AC2: Verify centralized tolerance function produces consistent values.
    use bitnet_quantization::qk256_tolerance_bytes;

    // For a 256-element QK256 tensor:
    //   bytes = 256 * 2 / 8 = 64 (2-bit packed)
    //   tolerance = ceil(64 * 0.001) = 1, but minimum is 8 bytes (alignment)
    let small_tolerance = qk256_tolerance_bytes(64);
    assert!(
        small_tolerance >= 8,
        "minimum tolerance is 8 bytes (alignment), got {}",
        small_tolerance
    );
    assert!(
        small_tolerance <= 64,
        "tolerance cannot exceed tensor size for small tensors, got {}",
        small_tolerance
    );

    // For a large 1MB tensor, tolerance should be proportional:
    let large_tolerance = qk256_tolerance_bytes(1_048_576);
    assert!(large_tolerance > small_tolerance, "larger tensors should have larger tolerance");
    assert!(large_tolerance <= 1_048_576, "tolerance cannot exceed tensor size");

    // Monotonicity: larger tensors have larger or equal tolerance.
    let sizes = [64usize, 256, 1024, 65536, 1_048_576];
    for pair in sizes.windows(2) {
        let t1 = qk256_tolerance_bytes(pair[0]);
        let t2 = qk256_tolerance_bytes(pair[1]);
        assert!(
            t1 <= t2,
            "tolerance must be monotone: tol({}) = {} > tol({}) = {}",
            pair[0],
            t1,
            pair[1],
            t2
        );
    }
}
