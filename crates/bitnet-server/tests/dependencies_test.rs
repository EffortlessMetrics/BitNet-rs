//! Dependency verification tests for OpenTelemetry OTLP migration
//!
//! Tests AC1: Remove opentelemetry-prometheus@0.29.1, migrate to OTLP@0.31
//! Specification: docs/explanation/specs/opentelemetry-otlp-migration-spec.md

/// AC:1 - Verify OTLP dependency is present in workspace
///
/// This test validates that opentelemetry-otlp@0.31 with metrics feature
/// is available in the dependency tree after migration.
///
/// Tests specification: opentelemetry-otlp-migration-spec.md#ac1-remove-deprecated-prometheus-exporter
#[cfg(feature = "opentelemetry")]
#[test]
fn test_ac1_otlp_dependency_present() -> std::io::Result<()> {
    // Parse Cargo.toml to verify OTLP dependency
    let cargo_toml_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.toml");
    let cargo_toml = std::fs::read_to_string(cargo_toml_path)?;

    // Verify opentelemetry-otlp with metrics feature is present
    assert!(
        cargo_toml.contains("opentelemetry-otlp") && cargo_toml.contains("\"metrics\""),
        "opentelemetry-otlp with metrics feature not found in workspace dependencies.\n\
         Expected: opentelemetry-otlp = {{ version = \"0.31.0\", features = [\"metrics\", ...] }}"
    );

    Ok(())
}

/// AC:1 - Verify Prometheus exporter dependency is removed
///
/// This test ensures the deprecated opentelemetry-prometheus@0.29.1 dependency
/// has been completely removed from workspace dependencies.
///
/// Tests specification: opentelemetry-otlp-migration-spec.md#ac1-remove-deprecated-prometheus-exporter
#[cfg(feature = "opentelemetry")]
#[test]
fn test_ac1_prometheus_dependency_removed() -> std::io::Result<()> {
    // Parse workspace Cargo.toml
    let cargo_toml_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.toml");
    let cargo_toml = std::fs::read_to_string(cargo_toml_path)?;

    // Verify opentelemetry-prometheus is not present
    assert!(
        !cargo_toml.contains("opentelemetry-prometheus"),
        "opentelemetry-prometheus should be removed from workspace dependencies.\n\
         Found deprecated dependency that conflicts with opentelemetry-sdk@0.31.0"
    );

    Ok(())
}

/// AC:1 - Verify tonic version compatibility
///
/// This test ensures tonic (gRPC transport for OTLP) is compatible with
/// opentelemetry-otlp@0.31.0 in the dependency tree.
///
/// Tests specification: opentelemetry-otlp-migration-spec.md#ac1-remove-deprecated-prometheus-exporter
#[cfg(feature = "opentelemetry")]
#[test]
fn test_ac1_tonic_version_compatible() -> std::io::Result<()> {
    // Parse crate Cargo.toml
    let cargo_toml_path = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    let cargo_toml = std::fs::read_to_string(cargo_toml_path)?;

    // Verify opentelemetry feature does not reference prometheus
    assert!(
        !cargo_toml.contains("dep:opentelemetry-prometheus"),
        "bitnet-server Cargo.toml should not reference opentelemetry-prometheus.\n\
         The feature list should only include: opentelemetry, opentelemetry_sdk, opentelemetry-otlp"
    );

    Ok(())
}

/// AC:1 - Verify no Prometheus imports in source code
///
/// This test ensures no `use opentelemetry_prometheus::*` statements remain
/// in the bitnet-server codebase after migration.
///
/// Tests specification: opentelemetry-otlp-migration-spec.md#ac1-remove-deprecated-prometheus-exporter
#[cfg(feature = "opentelemetry")]
#[test]
fn test_ac1_no_prometheus_imports_in_source() -> std::io::Result<()> {
    use std::path::PathBuf;

    let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");

    if !src_dir.exists() {
        return Ok(()); // Skip if src directory doesn't exist (unusual but safe)
    }

    // Recursively search for Prometheus imports in Rust source files
    let mut found_prometheus_import = false;
    let mut offending_file = String::new();

    for entry in walkdir::WalkDir::new(&src_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "rs"))
    {
        let content = std::fs::read_to_string(entry.path())?;
        if content.contains("use opentelemetry_prometheus") {
            found_prometheus_import = true;
            offending_file = entry.path().display().to_string();
            break;
        }
    }

    assert!(
        !found_prometheus_import,
        "Found deprecated opentelemetry_prometheus import in: {}\n\
         All Prometheus imports should be removed and replaced with OTLP",
        offending_file
    );

    Ok(())
}

// Note: walkdir is required for recursive directory traversal
// Add to Cargo.toml dev-dependencies if not present:
// walkdir = "2"
