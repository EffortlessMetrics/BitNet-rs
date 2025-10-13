//! Tests for projection weight RMS bounds enforcement

use bitnet_cli::ln_rules::Ruleset;

#[test]
fn proj_upper_bound_only_is_enforced() {
    let rs = Ruleset {
        ln: vec![],
        proj_weight_rms_min: None,
        proj_weight_rms_max: Some(0.1),
        name: "test-upper-only".into(),
    };
    // Within bound
    assert!(rs.check_proj_rms(0.05));
    // Exactly at bound
    assert!(rs.check_proj_rms(0.1));
    // Outside bound
    assert!(!rs.check_proj_rms(0.2));
}

#[test]
fn proj_lower_bound_only_is_enforced() {
    let rs = Ruleset {
        ln: vec![],
        proj_weight_rms_min: Some(0.1),
        proj_weight_rms_max: None,
        name: "test-lower-only".into(),
    };
    // Below bound
    assert!(!rs.check_proj_rms(0.05));
    // Exactly at bound
    assert!(rs.check_proj_rms(0.1));
    // Above bound
    assert!(rs.check_proj_rms(0.2));
}

#[test]
fn proj_both_bounds_are_enforced() {
    let rs = Ruleset {
        ln: vec![],
        proj_weight_rms_min: Some(0.1),
        proj_weight_rms_max: Some(0.5),
        name: "test-both-bounds".into(),
    };
    // Below lower bound
    assert!(!rs.check_proj_rms(0.05));
    // At lower bound
    assert!(rs.check_proj_rms(0.1));
    // In range
    assert!(rs.check_proj_rms(0.3));
    // At upper bound
    assert!(rs.check_proj_rms(0.5));
    // Above upper bound
    assert!(!rs.check_proj_rms(0.6));
}

#[test]
fn proj_no_bounds_always_passes() {
    let rs = Ruleset {
        ln: vec![],
        proj_weight_rms_min: None,
        proj_weight_rms_max: None,
        name: "test-no-bounds".into(),
    };
    assert!(rs.check_proj_rms(0.0));
    assert!(rs.check_proj_rms(0.1));
    assert!(rs.check_proj_rms(1.0));
    assert!(rs.check_proj_rms(100.0));
}
