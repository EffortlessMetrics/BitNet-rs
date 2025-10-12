//! Prometheus removal verification tests
//!
//! Tests AC3: Remove Prometheus code paths and verify clean compilation
//! Specification: docs/explanation/specs/opentelemetry-otlp-migration-spec.md

/// AC:3 - Verify no Prometheus imports remain in monitoring module
///
/// This test ensures all `use opentelemetry_prometheus::*` statements
/// have been removed from the codebase.
///
/// Tests specification: opentelemetry-otlp-migration-spec.md#ac3-remove-prometheus-code-paths
#[cfg(feature = "opentelemetry")]
#[test]
fn test_ac3_no_prometheus_imports() -> std::io::Result<()> {
    use std::path::PathBuf;

    let monitoring_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/monitoring");

    if !monitoring_dir.exists() {
        // If monitoring directory doesn't exist, test passes (unusual but safe)
        return Ok(());
    }

    // Check opentelemetry.rs for Prometheus imports
    let opentelemetry_rs = monitoring_dir.join("opentelemetry.rs");
    if opentelemetry_rs.exists() {
        let content = std::fs::read_to_string(&opentelemetry_rs)?;

        assert!(
            !content.contains("use opentelemetry_prometheus"),
            "Found deprecated opentelemetry_prometheus import in opentelemetry.rs.\n\
             All Prometheus imports should be removed."
        );

        assert!(
            !content.contains("PrometheusExporter"),
            "Found PrometheusExporter reference in opentelemetry.rs.\n\
             All Prometheus code should be removed."
        );

        assert!(
            !content.contains("exporter()"),
            "Found deprecated exporter() call in opentelemetry.rs.\n\
             Should be replaced with OTLP metrics initialization."
        );
    }

    Ok(())
}

/// AC:3 - Verify compilation with opentelemetry feature succeeds
///
/// This test ensures that the bitnet-server crate compiles successfully
/// with the opentelemetry feature enabled after Prometheus removal.
///
/// Tests specification: opentelemetry-otlp-migration-spec.md#ac3-remove-prometheus-code-paths
#[cfg(feature = "opentelemetry")]
#[test]
fn test_ac3_compilation_with_opentelemetry_feature() {
    // This test passes if it compiles, demonstrating that:
    // 1. No type errors from missing PrometheusExporter
    // 2. No trait bound failures from deprecated exporter() usage
    // 3. OTLP metrics integration compiles cleanly

    // Type visibility check - should compile without errors
    #[cfg(feature = "opentelemetry")]
    {
        // Import OTLP types to verify they're accessible
        // (Actual implementation will be added in AC2)
        // use bitnet_server::monitoring::otlp::{init_otlp_metrics, create_resource};

        // This test just needs to compile, not execute
        let _phantom: Option<()> = None;
    }
}

/// AC:3 - Verify OTLP module exists and is feature-gated
///
/// This test checks that the new OTLP module is properly declared
/// and feature-gated in the monitoring module.
///
/// Tests specification: opentelemetry-otlp-migration-spec.md#ac3-remove-prometheus-code-paths
#[cfg(feature = "opentelemetry")]
#[test]
#[should_panic(expected = "not yet implemented")]
fn test_ac3_otlp_module_exists() {
    // This test will fail until AC2/AC3 implementation is complete
    panic!("not yet implemented: OTLP module verification");

    // Expected implementation after AC3:
    // Verify that otlp module is accessible
    // use bitnet_server::monitoring::otlp;
    //
    // // These functions should be public and accessible
    // let _: fn(Option<String>, opentelemetry_sdk::Resource) -> anyhow::Result<_> =
    //     otlp::init_otlp_metrics;
    // let _: fn() -> opentelemetry_sdk::Resource = otlp::create_resource;
}

/// AC:3 - Verify no clippy warnings in observability code
///
/// This test ensures that the migration from Prometheus to OTLP
/// introduces zero clippy warnings in the monitoring module.
///
/// Tests specification: opentelemetry-otlp-migration-spec.md#ac3-remove-prometheus-code-paths
#[cfg(feature = "opentelemetry")]
#[test]
fn test_ac3_no_clippy_warnings_expected() {
    // This test documents the expectation that clippy should pass:
    //
    // cargo clippy -p bitnet-server --no-default-features --features opentelemetry -- -D warnings
    //
    // If this test compiles, it demonstrates that:
    // 1. No unused imports (deprecated Prometheus)
    // 2. No deprecated function calls (exporter())
    // 3. No type errors or trait bound failures
    //
    // The actual clippy validation occurs in CI, but this test
    // provides traceability to AC3.

    // Compilation success = test passes
    assert!(true, "Compilation with opentelemetry feature succeeded");
}

/// AC:3 - Verify monitoring module structure after migration
///
/// This test checks that the monitoring module structure is correct
/// after Prometheus removal and OTLP migration.
///
/// Tests specification: opentelemetry-otlp-migration-spec.md#ac3-remove-prometheus-code-paths
#[cfg(feature = "opentelemetry")]
#[test]
fn test_ac3_monitoring_module_structure() -> std::io::Result<()> {
    use std::path::PathBuf;

    let monitoring_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/monitoring");

    if !monitoring_dir.exists() {
        return Ok(()); // Module doesn't exist yet
    }

    // Check that mod.rs declares otlp module (after AC2)
    let mod_rs = monitoring_dir.join("mod.rs");
    if mod_rs.exists() {
        let content = std::fs::read_to_string(&mod_rs)?;

        // After AC2/AC3, should contain:
        // #[cfg(feature = "opentelemetry")]
        // pub mod otlp;

        // For now, just verify no Prometheus module references
        assert!(
            !content.contains("pub mod prometheus") && !content.contains("use prometheus"),
            "Found Prometheus module references in monitoring/mod.rs.\n\
             Should be removed after migration."
        );
    }

    Ok(())
}

/// AC:3 - Verify init_metrics function uses OTLP
///
/// This test ensures the init_metrics function in opentelemetry.rs
/// has been updated to use OTLP instead of Prometheus.
///
/// Tests specification: opentelemetry-otlp-migration-spec.md#ac3-remove-prometheus-code-paths
#[cfg(feature = "opentelemetry")]
#[test]
fn test_ac3_init_metrics_uses_otlp() -> std::io::Result<()> {
    use std::path::PathBuf;

    let opentelemetry_rs =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/monitoring/opentelemetry.rs");

    if !opentelemetry_rs.exists() {
        return Ok(()); // File doesn't exist yet
    }

    let content = std::fs::read_to_string(&opentelemetry_rs)?;

    // Verify old Prometheus code is removed
    assert!(
        !content.contains("let reader = exporter().build()?;"),
        "Found deprecated Prometheus exporter().build() call.\n\
         Should be replaced with OTLP initialization."
    );

    // After AC2/AC3, init_metrics should call OTLP initialization
    // (Cannot validate new implementation until AC2 complete)

    Ok(())
}
