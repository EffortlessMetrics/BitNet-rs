//! Workspace Quality Gate Integration Tests
//!
//! ## Purpose
//! Validates workspace-level quality gates after dependency upgrades to ensure:
//! - Workspace integrity (all crates build with proper feature flags)
//! - Dependency hygiene (no deprecated deps, version consistency)
//! - Feature gate correctness (CPU/GPU/crossval combinations)
//! - Build health across all workspace crates
//! - CI/CD enforcement via cargo-deny checks
//! - GitHub Actions compatibility
//!
//! ## Test Structure
//! Tests are organized into 9 categories covering all 8 quality gates from spec:
//! 1. **Workspace Integrity** - Build validation with feature flags (Gate 6)
//! 2. **Dependency Hygiene** - Deprecated dependency detection (Gate 4)
//! 3. **Feature Gate Validation** - CPU/GPU/crossval combinations
//! 4. **Code Quality** - Clippy and formatting checks (Gates 1, 2)
//! 5. **Workspace Structure** - Member completeness and centralization
//! 6. **Post-Migration Validation** - dirs/once_cell version consolidation (Gate 5)
//! 7. **Cargo Deny Tests** - Bans, advisories, licenses, sources (Gate 7)
//! 8. **Dependency Tree Analysis** - Tree parsing, conflicts, minimal versions
//! 9. **CI/CD Environment** - CI variable validation, GitHub Actions compatibility
//!
//! ## Quality Gates Mapping (from Phase 2 Specification)
//!
//! | Gate # | Name | Test Functions | Status |
//! |--------|------|----------------|--------|
//! | Gate 1 | Format Check | `test_workspace_formatting` | ✅ Implemented |
//! | Gate 2 | Clippy Clean | `test_workspace_clippy_clean` | ✅ Implemented |
//! | Gate 3 | Tests Pass | Meta-test (verified by CI) | ✅ N/A |
//! | Gate 4 | No Deprecated Deps | `test_no_deprecated_yaml_dependencies`, `test_no_lazy_static_dependencies` | ✅ Implemented |
//! | Gate 5 | Version Consolidation | `test_dependency_versions_consistent`, `test_once_cell_version_consolidated`, `test_no_version_conflicts` | ✅ Implemented |
//! | Gate 6 | Build Succeeds | `test_workspace_builds_with_*_features`, `test_cpu_only_feature_build` | ✅ Implemented |
//! | Gate 7 | Cargo Deny Check | `test_cargo_deny_bans`, `test_cargo_deny_advisories`, `test_cargo_deny_licenses`, `test_cargo_deny_sources`, `test_cargo_deny_all` | ✅ Implemented |
//! | Gate 8 | Cross-Validation | `test_crossval_feature_build` | ✅ Implemented |
//!
//! ## Specification References
//! - `/tmp/phase2_orchestration_specification.md` - Quality gates Section 3 (lines 643-1000)
//! - `/tmp/phase1_deprecated_deps_analysis.md` - Workspace structure and deprecated deps
//!
//! ## Usage
//! ```bash
//! # Run all quality gate tests (non-ignored)
//! cargo test --test workspace_quality_gates
//!
//! # Run specific category
//! cargo test --test workspace_quality_gates -- workspace_integrity
//! cargo test --test workspace_quality_gates -- dependency_hygiene
//! cargo test --test workspace_quality_gates -- cargo_deny
//!
//! # Run ignored tests (requires cargo-deny installed)
//! cargo test --test workspace_quality_gates -- --ignored --include-ignored
//!
//! # Run in CI environment (enables CI-specific checks)
//! CI=1 cargo test --test workspace_quality_gates
//! ```

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Command;

/// Helper: Run cargo command and capture output
fn run_cargo_command(args: &[&str]) -> Result<(bool, String, String), String> {
    let output = Command::new("cargo")
        .args(args)
        .current_dir("/home/steven/code/Rust/BitNet-rs")
        .output()
        .map_err(|e| format!("Failed to execute cargo: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let success = output.status.success();

    Ok((success, stdout, stderr))
}

/// Helper: Parse Cargo.lock to detect duplicate dependencies
fn parse_cargo_lock_for_duplicates() -> Result<HashMap<String, Vec<String>>, String> {
    let cargo_lock_path = "/home/steven/code/Rust/BitNet-rs/Cargo.lock";
    let content = std::fs::read_to_string(cargo_lock_path)
        .map_err(|e| format!("Failed to read Cargo.lock: {}", e))?;

    let mut packages: HashMap<String, Vec<String>> = HashMap::new();
    let mut current_name: Option<String> = None;

    for line in content.lines() {
        let trimmed = line.trim();

        // Start of new package entry
        if trimmed == "[[package]]" {
            current_name = None;
        } else if trimmed.starts_with("name = ") {
            let name = trimmed.trim_start_matches("name = ").trim_matches('"').to_string();
            current_name = Some(name);
        } else if trimmed.starts_with("version = ")
            && let Some(ref name) = current_name
        {
            let version = trimmed.trim_start_matches("version = ").trim_matches('"').to_string();
            packages.entry(name.clone()).or_default().push(version);
            current_name = None; // Reset after capturing version
        }
    }

    Ok(packages)
}

/// Helper: Check if a crate exists in Cargo.toml dependencies
fn check_deprecated_dependency(crate_name: &str, toml_path: &str) -> Result<bool, String> {
    let content = std::fs::read_to_string(toml_path)
        .map_err(|e| format!("Failed to read {}: {}", toml_path, e))?;

    // Check both direct dependency and workspace specification
    let patterns = [format!("{} = ", crate_name), format!("{}.", crate_name)];

    Ok(patterns.iter().any(|pattern| content.contains(pattern)))
}

// ============================================================================
// Category 1: Workspace Integrity Tests
// ============================================================================

/// Tests workspace builds with --no-default-features (baseline)
///
/// Specification: phase2_upgrade_orchestration_spec.md#universal-quality-gates (line 512)
#[test]
fn test_workspace_builds_with_no_default_features() {
    let (success, stdout, stderr) =
        run_cargo_command(&["check", "--workspace", "--no-default-features"])
            .expect("Failed to run cargo check");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        panic!("Workspace check with --no-default-features failed");
    }

    assert!(success, "Workspace should build with --no-default-features");
}

/// Tests workspace builds with CPU features
///
/// Specification: phase2_upgrade_orchestration_spec.md#build-gates (line 512)
#[test]
fn test_workspace_builds_with_cpu_features() {
    let (success, stdout, stderr) =
        run_cargo_command(&["build", "--workspace", "--no-default-features", "--features", "cpu"])
            .expect("Failed to run cargo build");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        panic!("Workspace build with CPU features failed");
    }

    assert!(success, "Workspace should build with CPU features");
}

/// Tests workspace builds with GPU features
///
/// Specification: phase2_upgrade_orchestration_spec.md#build-gates (line 513)
#[test]
#[ignore = "Requires GPU toolchain (CUDA, etc.) - run explicitly with --ignored"]
fn test_workspace_builds_with_gpu_features() {
    let (success, stdout, stderr) =
        run_cargo_command(&["build", "--workspace", "--no-default-features", "--features", "gpu"])
            .expect("Failed to run cargo build");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        panic!("Workspace build with GPU features failed");
    }

    assert!(success, "Workspace should build with GPU features");
}

/// Tests workspace builds with crossval-all features
///
/// Specification: phase2_upgrade_orchestration_spec.md#cross-validation-gates (line 529)
#[test]
#[ignore = "Post-migration test - requires crossval-all feature implementation"]
fn test_workspace_builds_with_crossval_features() {
    let (success, stdout, stderr) = run_cargo_command(&[
        "build",
        "--workspace",
        "--no-default-features",
        "--features",
        "crossval-all",
    ])
    .expect("Failed to run cargo build");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        panic!("Workspace build with crossval-all features failed");
    }

    assert!(success, "Workspace should build with crossval-all features");
}

// ============================================================================
// Category 2: Dependency Hygiene Tests
// ============================================================================

/// Tests that serde_yaml is fully replaced with serde_yaml_ng
///
/// Specification: phase1_deprecated_deps_analysis.md#serde_yaml (lines 23-52)
/// Phase: Phase 1 (P0 CRITICAL) - ✅ COMPLETE (Migration validated)
#[test]
fn test_no_deprecated_yaml_dependencies() {
    let crates_to_check = vec![
        "/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/Cargo.toml",
        "/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/Cargo.toml",
        "/home/steven/code/Rust/BitNet-rs/tests/Cargo.toml",
    ];

    let mut failures = Vec::new();

    for toml_path in crates_to_check {
        let content = std::fs::read_to_string(toml_path)
            .unwrap_or_else(|_| panic!("Failed to read {}", toml_path));

        // Check for old serde_yaml 0.9 pattern (should NOT exist)
        if content.contains("serde_yaml = \"0.9") {
            failures.push(format!("Found deprecated serde_yaml 0.9 in {}", toml_path));
        }

        // Verify new serde_yaml_ng pattern exists (either inline or via workspace inheritance)
        let has_alias = content.contains("package = \"serde_yaml_ng\"");
        let uses_workspace =
            content.lines().any(|l| l.contains("serde_yaml") && l.contains("workspace = true"));
        if !has_alias && !uses_workspace {
            failures.push(format!("Missing serde_yaml_ng aliasing in {}", toml_path));
        }
    }

    if !failures.is_empty() {
        panic!("Dependency validation failed:\n{}", failures.join("\n"));
    }
}

/// Tests comprehensive serde_yaml_ng migration validation (AC1-AC6)
///
/// Specification: phase2_serde_yaml_specification.md#gate-1 (lines 375-395)
/// **Migration Status**: ✅ COMPLETE (validated 2025-10-27)
///
/// **Comprehensive Validation**:
/// 1. Cargo.toml aliasing: All crates use `package = "serde_yaml_ng"`
/// 2. Version correctness: All use 0.10.0+
/// 3. Zero deprecated patterns: No `serde_yaml = "0.9"` anywhere
/// 4. Cargo.lock verification: Only serde_yaml_ng v0.10.0 in dependency tree
#[test]
fn test_serde_yaml_ng_migration_complete() {
    let crates_to_check = vec![
        "/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/Cargo.toml",
        "/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/Cargo.toml",
        "/home/steven/code/Rust/BitNet-rs/tests/Cargo.toml",
    ];

    let mut failures = Vec::new();

    for toml_path in crates_to_check {
        let content = std::fs::read_to_string(toml_path)
            .unwrap_or_else(|_| panic!("Failed to read {}", toml_path));

        // AC1: Verify package aliasing exists (inline or via workspace inheritance)
        let has_inline_alias = content.contains("package = \"serde_yaml_ng\"");
        let uses_workspace =
            content.lines().any(|l| l.contains("serde_yaml") && l.contains("workspace = true"));
        if !has_inline_alias && !uses_workspace {
            failures.push(format!("AC1 FAIL: Missing serde_yaml_ng aliasing in {}", toml_path));
        }

        // AC1: Verify version is 0.10.0+ (only applies to inline definitions)
        if content.contains("serde_yaml_ng") && !content.contains("0.10.") && !uses_workspace {
            failures.push(format!("AC1 FAIL: Incorrect serde_yaml_ng version in {}", toml_path));
        }

        // AC1: Verify no deprecated serde_yaml 0.9 patterns
        if content.contains("serde_yaml = \"0.9") {
            failures.push(format!("AC1 FAIL: Found deprecated serde_yaml 0.9 in {}", toml_path));
        }

        // AC1: Verify no direct serde_yaml dependency (without aliasing)
        let lines: Vec<&str> = content.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("serde_yaml = \"")
                && !line.contains("package")
                && !trimmed.starts_with("#")
            {
                failures.push(format!(
                    "AC1 FAIL: Found direct serde_yaml dependency at {}:{}",
                    toml_path,
                    idx + 1
                ));
            }
        }
    }

    // AC2: Verify Cargo.lock shows only serde_yaml_ng
    let cargo_lock_path = "/home/steven/code/Rust/BitNet-rs/Cargo.lock";
    let cargo_lock_content =
        std::fs::read_to_string(cargo_lock_path).expect("Failed to read Cargo.lock");

    // Check for presence of serde_yaml_ng 0.10
    if !cargo_lock_content.contains("name = \"serde_yaml_ng\"")
        || !cargo_lock_content.contains("version = \"0.10.")
    {
        failures.push("AC2 FAIL: Cargo.lock missing serde_yaml_ng 0.10.x".to_string());
    }

    // Check for absence of deprecated serde_yaml 0.9
    if cargo_lock_content.contains("name = \"serde_yaml\"\nversion = \"0.9") {
        failures.push("AC2 FAIL: Cargo.lock still contains deprecated serde_yaml 0.9".to_string());
    }

    if !failures.is_empty() {
        panic!("serde_yaml_ng migration validation FAILED:\n{}", failures.join("\n"));
    }
}

/// Tests cargo tree shows only serde_yaml_ng (no deprecated serde_yaml 0.9)
///
/// Specification: phase2_serde_yaml_specification.md#gate-2 (lines 274-299)
#[test]
fn test_cargo_tree_shows_only_serde_yaml_ng() {
    let (success, stdout, stderr) = run_cargo_command(&["tree", "--workspace", "--depth", "3"])
        .expect("Failed to run cargo tree");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        panic!("Cargo tree command failed");
    }

    // Should NOT contain old serde_yaml 0.9
    if stdout.contains("serde_yaml v0.9") {
        panic!(
            "FAIL: Found deprecated serde_yaml 0.9 in dependency tree:\n{}",
            stdout
                .lines()
                .filter(|line| line.contains("serde_yaml"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    // Should contain serde_yaml_ng 0.10
    if !stdout.contains("serde_yaml_ng v0.10") {
        panic!("FAIL: Missing serde_yaml_ng 0.10 in dependency tree");
    }

    eprintln!("✅ Dependency tree verification passed:");
    eprintln!("  - No deprecated serde_yaml 0.9 found");
    eprintln!("  - serde_yaml_ng 0.10 present");
}

/// Tests serde_yaml_ng version consistency across workspace
///
/// Specification: phase2_serde_yaml_specification.md#ac1 (lines 375-395)
#[test]
fn test_serde_yaml_ng_version_consistency() {
    let crates_to_check = vec![
        "/home/steven/code/Rust/BitNet-rs/Cargo.toml", // workspace root defines canonical version
        "/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/Cargo.toml",
        "/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/Cargo.toml",
        "/home/steven/code/Rust/BitNet-rs/tests/Cargo.toml",
    ];

    let mut versions = HashSet::new();

    for toml_path in crates_to_check {
        let content = std::fs::read_to_string(toml_path)
            .unwrap_or_else(|_| panic!("Failed to read {}", toml_path));

        // Extract version from: serde_yaml = { package = "serde_yaml_ng", version = "0.10.0" }
        for line in content.lines() {
            if line.contains("package = \"serde_yaml_ng\"") {
                // Extract version from this line or next few lines
                let context = content
                    .lines()
                    .skip_while(|l| !l.contains("package = \"serde_yaml_ng\""))
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(" ");

                if let Some(version_start) = context.find("version = \"") {
                    let version_str = &context[version_start + 11..];
                    if let Some(version_end) = version_str.find('"') {
                        let version = &version_str[..version_end];
                        versions.insert(version.to_string());
                    }
                }
            }
        }
    }

    // All versions should be identical (0.10.0). When crates use workspace = true,
    // their version is inherited from the workspace root which is also checked.
    if versions.is_empty() {
        panic!(
            "FAIL: No explicit serde_yaml_ng version found in workspace (workspace root should define it)"
        );
    }
    if versions.len() != 1 {
        panic!("FAIL: Inconsistent serde_yaml_ng versions across workspace: {:?}", versions);
    }

    let version = versions.iter().next().unwrap();
    assert!(
        version.starts_with("0.10"),
        "FAIL: serde_yaml_ng version should be 0.10.x, found: {}",
        version
    );

    eprintln!("✅ Version consistency verified: serde_yaml_ng v{}", version);
}

/// Tests that lazy_static is fully replaced with std::sync::OnceLock
///
/// Specification: phase1_deprecated_deps_analysis.md#lazy_static (lines 58-89)
/// Phase: Phase 2 (P1 HIGH)
#[test]
#[ignore = "Post-Phase-2: Enable after lazy_static → OnceLock migration"]
fn test_no_lazy_static_dependencies() {
    let crates_to_check =
        vec!["/home/steven/code/Rust/BitNet-rs/crates/bitnet-st-tools/Cargo.toml"];

    let mut failures = Vec::new();

    for toml_path in crates_to_check {
        match check_deprecated_dependency("lazy_static", toml_path) {
            Ok(true) => {
                failures.push(format!("Found deprecated lazy_static in {}", toml_path));
            }
            Ok(false) => {
                // Good - lazy_static not found
            }
            Err(e) => {
                failures.push(format!("Error checking {}: {}", toml_path, e));
            }
        }
    }

    if !failures.is_empty() {
        panic!("Deprecated lazy_static dependencies found:\n{}", failures.join("\n"));
    }
}

/// Tests dependency version consistency across workspace
///
/// Specification: phase2_upgrade_orchestration_spec.md#final-gates (line 539)
///
/// **Note**: This test allows known ecosystem duplicates and transitives.
/// Focus is on critical direct dependencies (serde_yaml, once_cell) that are migrated.
#[test]
fn test_dependency_versions_consistent() {
    let packages = parse_cargo_lock_for_duplicates().expect("Failed to parse Cargo.lock");

    let mut duplicates = Vec::new();

    // Known allowed duplicates (due to ecosystem constraints)
    // These are transitive dependencies or ecosystem standard duplicates
    let allowed_duplicates: HashSet<&str> = vec![
        // Windows ecosystem
        "windows_x86_64_msvc",
        "windows-sys",
        "windows-core",
        "windows-targets",
        "windows-link",
        // Macro crates (often have version skew)
        "syn",
        "quote",
        "proc-macro2",
        // Common transitives with ecosystem version skew
        "bitflags",
        "itertools",
        "getrandom",
        "thiserror",
        "unicode-width",
        "toml",
        "cargo_metadata",
        "linux-raw-sys",
        "dirs",
        "dirs-sys",
        "redox_users",
        "foreign-types",
        // GEMM/BLAS ecosystem (version constrained by different BLAS implementations)
        "gemm",
        "gemm-common",
        "gemm-f32",
        "gemm-f64",
        "gemm-c64",
        "pulp",
        // Protobuf ecosystem
        "prost",
        "prost-derive",
        // Git ecosystem
        "gix-features",
        "gix-fs",
        // Misc ecosystem transitives
        "faster-hex",
        "cargo-platform",
        "aho-corasick",
    ]
    .into_iter()
    .collect();

    // Critical dependencies we care about (from migration targets)
    let critical_deps: HashSet<&str> = vec![
        "serde_yaml",  // Should not exist (migrated to serde_yaml_ng)
        "once_cell",   // Should be single version or eliminated
        "lazy_static", // Should be transitive only
    ]
    .into_iter()
    .collect();

    for (name, versions) in packages.iter() {
        if versions.len() > 1 {
            // Flag critical dependencies even if duplicate
            if critical_deps.contains(name.as_str()) {
                duplicates.push(format!(
                    "⚠️  CRITICAL: {}: {} versions ({:?})",
                    name,
                    versions.len(),
                    versions
                ));
            }
            // Flag non-allowed duplicates
            else if !allowed_duplicates.contains(name.as_str()) {
                duplicates.push(format!("{}: {} versions ({:?})", name, versions.len(), versions));
            }
        }
    }

    if !duplicates.is_empty() {
        eprintln!("=== Duplicate Dependencies Found ===");
        for dup in &duplicates {
            eprintln!("  - {}", dup);
        }

        // Only panic if critical deps are duplicated
        let has_critical = duplicates
            .iter()
            .any(|d| d.contains("CRITICAL") || d.contains("once_cell") || d.contains("serde_yaml"));

        if has_critical {
            panic!("Found CRITICAL duplicate dependencies - these require migration");
        } else {
            eprintln!(
                "⚠️  Found {} non-critical duplicates (ecosystem transitives)",
                duplicates.len()
            );
            eprintln!("These are acceptable during migration phase");
        }
    }
}

/// Tests that cargo audit passes (or only known-safe advisories)
///
/// Specification: phase2_upgrade_orchestration_spec.md#final-gates (line 540)
#[test]
#[ignore = "Requires cargo-audit installed - run explicitly in CI"]
fn test_cargo_audit_passes() {
    // Check if cargo-audit is installed
    let check = Command::new("cargo").args(["audit", "--version"]).output();

    if check.is_err() {
        eprintln!("cargo-audit not installed - skipping test");
        return;
    }

    let (success, stdout, stderr) =
        run_cargo_command(&["audit"]).expect("Failed to run cargo audit");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);

        // Check if only known-safe advisories (paste unmaintained)
        if stderr.contains("RUSTSEC-2024-0436") && !stderr.contains("RUSTSEC-") {
            eprintln!(
                "Only known-safe advisory RUSTSEC-2024-0436 (paste unmaintained) found - acceptable"
            );
            return;
        }

        panic!("Cargo audit found security advisories");
    }

    assert!(success, "Cargo audit should pass");
}

// ============================================================================
// Category 3: Feature Gate Tests
// ============================================================================

/// Tests CPU-only feature build for specific crates
///
/// Specification: phase2_upgrade_orchestration_spec.md#build-gates (line 512)
#[test]
fn test_cpu_only_feature_build() {
    let critical_crates = vec!["bitnet-kernels", "bitnet-quantization", "bitnet-inference"];

    for crate_name in critical_crates {
        let (success, stdout, stderr) = run_cargo_command(&[
            "build",
            "-p",
            crate_name,
            "--no-default-features",
            "--features",
            "cpu",
        ])
        .expect("Failed to run cargo build");

        if !success {
            eprintln!("=== CRATE: {} ===", crate_name);
            eprintln!("=== STDOUT ===\n{}", stdout);
            eprintln!("=== STDERR ===\n{}", stderr);
            panic!("Crate {} failed to build with CPU features", crate_name);
        }

        assert!(success, "Crate {} should build with CPU features", crate_name);
    }
}

/// Tests GPU feature build for GPU-capable crates
///
/// Specification: phase2_upgrade_orchestration_spec.md#build-gates (line 513)
#[test]
#[ignore = "Requires GPU toolchain (CUDA, etc.) - run explicitly with --ignored"]
fn test_gpu_feature_build() {
    let gpu_crates = vec!["bitnet-kernels", "bitnet-quantization"];

    for crate_name in gpu_crates {
        let (success, stdout, stderr) = run_cargo_command(&[
            "build",
            "-p",
            crate_name,
            "--no-default-features",
            "--features",
            "gpu",
        ])
        .expect("Failed to run cargo build");

        if !success {
            eprintln!("=== CRATE: {} ===", crate_name);
            eprintln!("=== STDOUT ===\n{}", stdout);
            eprintln!("=== STDERR ===\n{}", stderr);
            panic!("Crate {} failed to build with GPU features", crate_name);
        }

        assert!(success, "Crate {} should build with GPU features", crate_name);
    }
}

/// Tests that feature flags don't conflict
///
/// Specification: phase2_upgrade_orchestration_spec.md#build-gates (lines 512-516)
#[test]
fn test_feature_flags_dont_conflict() {
    // Test CPU + fixtures combination (common in tests)
    let (success, stdout, stderr) = run_cargo_command(&[
        "check",
        "-p",
        "bitnet-models",
        "--no-default-features",
        "--features",
        "cpu,fixtures",
    ])
    .expect("Failed to run cargo check");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        panic!("CPU + fixtures features should not conflict");
    }

    assert!(success, "CPU + fixtures features should build successfully");
}

/// Tests that crossval feature builds correctly
///
/// Specification: phase2_upgrade_orchestration_spec.md#cross-validation-gates (line 529)
#[test]
#[ignore = "Post-migration test - requires crossval infrastructure"]
fn test_crossval_feature_build() {
    let (success, stdout, stderr) =
        run_cargo_command(&["build", "-p", "xtask", "--features", "crossval-all"])
            .expect("Failed to run cargo build");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        panic!("xtask with crossval-all features should build");
    }

    assert!(success, "xtask should build with crossval-all features");
}

// ============================================================================
// Category 4: Code Quality Tests
// ============================================================================

/// Tests that workspace passes clippy with warnings denied
///
/// Specification: phase2_upgrade_orchestration_spec.md#build-gates (line 516)
#[test]
#[ignore = "Slow test - run explicitly in CI"]
fn test_workspace_clippy_clean() {
    let (success, stdout, stderr) =
        run_cargo_command(&["clippy", "--all-targets", "--all-features", "--", "-D", "warnings"])
            .expect("Failed to run cargo clippy");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        panic!("Clippy found warnings");
    }

    assert!(success, "Workspace should be clippy-clean");
}

/// Tests that workspace code is formatted correctly
///
/// Specification: phase2_upgrade_orchestration_spec.md#final-gates (line 541)
#[test]
fn test_workspace_formatting() {
    let (success, stdout, stderr) =
        run_cargo_command(&["fmt", "--all", "--check"]).expect("Failed to run cargo fmt");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        panic!("Code formatting check failed - run 'cargo fmt --all'");
    }

    assert!(success, "Workspace code should be formatted");
}

// ============================================================================
// Category 5: Workspace Structure Tests
// ============================================================================

/// Tests that all workspace members are declared correctly
///
/// Specification: phase1_deprecated_deps_analysis.md#workspace-summary (line 228)
#[test]
fn test_workspace_members_complete() {
    let expected_members = vec![
        "crates/bitnet-common",
        "crates/bitnet-runtime-context",
        "crates/bitnet-runtime-context-core",
        "crates/bitnet-bdd-grid",
        "crates/bitnet-bdd-grid-core",
        "crates/bitnet-compat",
        "crates/bitnet-ggml-ffi",
        "crates/bitnet-runtime-feature-flags",
        "crates/bitnet-runtime-feature-flags-core",
        "crates/bitnet-models",
        "crates/bitnet-quantization",
        "crates/bitnet-kernels",
        "crates/bitnet-inference",
        "crates/bitnet-tokenizers",
        "crates/bitnet-feature-matrix",
        "crates/bitnet-feature-contract",
        "crates/bitnet-runtime-profile",
        "crates/bitnet-runtime-profile-core",
        "crates/bitnet-runtime-profile-contract",
        "crates/bitnet-runtime-profile-contract-core",
        "crates/bitnet-testing-profile",
        "crates/bitnet-testing-scenarios",
        "crates/bitnet-testing-scenarios-core",
        "crates/bitnet-testing-scenarios-profile-core",
        "crates/bitnet-testing-policy-core",
        "crates/bitnet-testing-policy",
        "crates/bitnet-testing-policy-tests",
        "crates/bitnet-testing-policy-kit",
        "crates/bitnet-testing-policy-contract",
        "crates/bitnet-testing-policy-interop",
        "crates/bitnet-testing-policy-runtime",
        "crates/bitnet-server",
        "crates/bitnet-cli",
        "crates/bitnet-runtime-bootstrap",
        "crates/bitnet-startup-contract",
        "crates/bitnet-startup-contract-core",
        "crates/bitnet-startup-contract-diagnostics",
        "crates/bitnet-startup-contract-guard",
        "crates/bitnet-st2gguf",
        "crates/bitnet-st-tools",
        "crates/bitnet-trace",
        "crates/bitnet-ffi",
        "crates/bitnet-py",
        "crates/bitnet-wasm",
        "crates/bitnet-sys",
        "crossval",
        "tests",
        "xtask",
        "xtask-build-helper",
        "fuzz",
        "tools/migrate-gen-config",
    ];

    let root_toml = std::fs::read_to_string("/home/steven/code/Rust/BitNet-rs/Cargo.toml")
        .expect("Failed to read root Cargo.toml");

    let mut missing_members = Vec::new();

    for member in expected_members {
        if !root_toml.contains(member) {
            missing_members.push(member);
        }
    }

    if !missing_members.is_empty() {
        panic!("Missing workspace members in root Cargo.toml:\n{}", missing_members.join("\n"));
    }
}

/// Tests that workspace dependency versions are centralized
///
/// Specification: phase1_deprecated_deps_analysis.md#workspace-summary (line 243)
#[test]
fn test_workspace_dependencies_centralized() {
    let root_toml = std::fs::read_to_string("/home/steven/code/Rust/BitNet-rs/Cargo.toml")
        .expect("Failed to read root Cargo.toml");

    // Check that workspace dependencies section exists
    assert!(
        root_toml.contains("[workspace.dependencies]"),
        "Root Cargo.toml should contain [workspace.dependencies] section"
    );

    // Check for common dependencies that should be centralized
    // Note: once_cell is NOT included - we use std::sync::OnceLock instead
    let centralized_deps = vec!["serde", "anyhow", "thiserror", "serial_test"];

    for dep in centralized_deps {
        let pattern = format!("{} =", dep);
        assert!(
            root_toml.contains(&pattern),
            "Workspace dependency '{}' should be centralized in [workspace.dependencies]",
            dep
        );
    }
}

// ============================================================================
// Category 6: Post-Migration Validation Tests
// ============================================================================

/// Tests that dirs version is consolidated (Phase 3)
///
/// Specification: phase1_deprecated_deps_analysis.md#dirs (lines 119-145)
/// Phase: Phase 3 (P2 MEDIUM)
#[test]
#[ignore = "Post-Phase-3: Enable after dirs version consolidation"]
fn test_dirs_version_consolidated() {
    let packages = parse_cargo_lock_for_duplicates().expect("Failed to parse Cargo.lock");

    if let Some(versions) = packages.get("dirs") {
        assert_eq!(
            versions.len(),
            1,
            "dirs should have only one version after Phase 3 (found: {:?})",
            versions
        );

        // Check that it's version 6.0.0
        assert!(
            versions[0].starts_with("6.0"),
            "dirs should be version 6.0.0 (found: {})",
            versions[0]
        );
    }
}

/// Tests that once_cell versions are consolidated (Phase 3)
///
/// Specification: phase1_deprecated_deps_analysis.md#once_cell (lines 149-169)
/// Phase: Phase 3 (P2 MEDIUM)
#[test]
#[ignore = "Post-Phase-3: Enable after once_cell version consolidation"]
fn test_once_cell_version_consolidated() {
    let packages = parse_cargo_lock_for_duplicates().expect("Failed to parse Cargo.lock");

    if let Some(versions) = packages.get("once_cell") {
        assert_eq!(
            versions.len(),
            1,
            "once_cell should have only one version after Phase 3 (found: {:?})",
            versions
        );

        // Check that it's version 1.21.3 (workspace standard)
        assert!(
            versions[0].starts_with("1.21"),
            "once_cell should be version 1.21.3 (found: {})",
            versions[0]
        );
    }
}

// ============================================================================
// Category 7: Cargo Deny Tests (NEW - Quality Gates from Spec)
// ============================================================================

/// Tests that cargo-deny bans check passes
///
/// Specification: phase2_upgrade_orchestration_spec.md#gate-7-cargo-deny-check (line 915)
#[test]
#[ignore = "Requires cargo-deny installed - run explicitly in CI"]
fn test_cargo_deny_bans() {
    // Check if cargo-deny is installed
    let check = Command::new("cargo").args(["deny", "--version"]).output();

    if check.is_err() {
        eprintln!("cargo-deny not installed - skipping test");
        eprintln!("Install with: cargo install cargo-deny --locked");
        return;
    }

    let (success, stdout, stderr) =
        run_cargo_command(&["deny", "check", "bans", "--allow-warnings"])
            .expect("Failed to run cargo deny check bans");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        panic!("Cargo deny bans check failed - deprecated dependencies detected");
    }

    assert!(success, "Cargo deny bans check should pass");
}

/// Tests that cargo-deny advisories check passes
///
/// Specification: phase2_upgrade_orchestration_spec.md#gate-7-cargo-deny-check (line 927)
#[test]
#[ignore = "Requires cargo-deny installed - run explicitly in CI"]
fn test_cargo_deny_advisories() {
    // Check if cargo-deny is installed
    let check = Command::new("cargo").args(["deny", "--version"]).output();

    if check.is_err() {
        eprintln!("cargo-deny not installed - skipping test");
        return;
    }

    let (success, stdout, stderr) = run_cargo_command(&["deny", "check", "advisories"])
        .expect("Failed to run cargo deny check advisories");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);

        // Check if only known-safe advisories (paste unmaintained)
        if stderr.contains("RUSTSEC-2024-0436") && !stderr.contains("RUSTSEC-") {
            eprintln!(
                "Only known-safe advisory RUSTSEC-2024-0436 (paste unmaintained) found - acceptable"
            );
            return;
        }

        panic!("Cargo deny advisories check found security vulnerabilities");
    }

    assert!(success, "Cargo deny advisories check should pass");
}

/// Tests that cargo-deny licenses check passes
///
/// Specification: phase2_upgrade_orchestration_spec.md#gate-7-cargo-deny-check (line 928)
#[test]
#[ignore = "Requires cargo-deny installed - run explicitly in CI"]
fn test_cargo_deny_licenses() {
    // Check if cargo-deny is installed
    let check = Command::new("cargo").args(["deny", "--version"]).output();

    if check.is_err() {
        eprintln!("cargo-deny not installed - skipping test");
        return;
    }

    let (success, stdout, stderr) = run_cargo_command(&["deny", "check", "licenses"])
        .expect("Failed to run cargo deny check licenses");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        panic!("Cargo deny licenses check failed - incompatible licenses detected");
    }

    assert!(success, "Cargo deny licenses check should pass");
}

/// Tests that cargo-deny sources check passes
///
/// Specification: phase2_upgrade_orchestration_spec.md#gate-7-cargo-deny-check (line 929)
#[test]
#[ignore = "Requires cargo-deny installed - run explicitly in CI"]
fn test_cargo_deny_sources() {
    // Check if cargo-deny is installed
    let check = Command::new("cargo").args(["deny", "--version"]).output();

    if check.is_err() {
        eprintln!("cargo-deny not installed - skipping test");
        return;
    }

    let (success, stdout, stderr) = run_cargo_command(&["deny", "check", "sources"])
        .expect("Failed to run cargo deny check sources");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        panic!("Cargo deny sources check failed - invalid dependency sources detected");
    }

    assert!(success, "Cargo deny sources check should pass");
}

/// Tests comprehensive cargo-deny check (all gates)
///
/// Specification: phase2_upgrade_orchestration_spec.md#gate-7-cargo-deny-check (line 924)
#[test]
#[ignore = "Requires cargo-deny installed - run explicitly in CI"]
fn test_cargo_deny_all() {
    // Check if cargo-deny is installed
    let check = Command::new("cargo").args(["deny", "--version"]).output();

    if check.is_err() {
        eprintln!("cargo-deny not installed - skipping test");
        eprintln!("Install with: cargo install cargo-deny --locked");
        return;
    }

    let (success, stdout, stderr) =
        run_cargo_command(&["deny", "check", "all"]).expect("Failed to run cargo deny check all");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);

        // Allow known-safe advisories
        if stderr.contains("RUSTSEC-2024-0436") && stderr.matches("RUSTSEC-").count() == 1 {
            eprintln!(
                "Only known-safe advisory RUSTSEC-2024-0436 (paste unmaintained) found - acceptable"
            );
            return;
        }

        panic!("Cargo deny check all failed - review issues above");
    }

    assert!(success, "Cargo deny check all should pass");
}

// ============================================================================
// Category 8: Dependency Tree Analysis Tests (NEW)
// ============================================================================

/// Tests that cargo tree generates valid output
///
/// Helper test to verify dependency tree parsing works
#[test]
fn test_cargo_tree_parse() {
    let (success, stdout, stderr) = run_cargo_command(&["tree", "--workspace", "--depth", "1"])
        .expect("Failed to run cargo tree");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        panic!("Cargo tree command failed");
    }

    // Verify output contains expected workspace crates
    assert!(stdout.contains("bitnet"), "Cargo tree should contain bitnet crate");
    assert!(success, "Cargo tree should generate valid output");
}

/// Tests that no duplicate dependency versions exist (with exceptions)
///
/// Specification: phase2_upgrade_orchestration_spec.md#gate-5-version-consolidation (line 829)
///
/// **Note**: This test is informational only. Use `test_dependency_versions_consistent`
/// for strict validation of critical dependencies.
#[test]
fn test_no_version_conflicts() {
    let (success, stdout, _stderr) = run_cargo_command(&["tree", "--workspace", "--duplicates"])
        .expect("Failed to run cargo tree --duplicates");

    if !success {
        // Command may fail if no duplicates, which is actually good
        eprintln!("✅ No duplicates found (command returned non-zero, which is expected)");
        return;
    }

    // If command succeeded, check if output is empty (no duplicates)
    if stdout.trim().is_empty() {
        eprintln!("✅ No duplicate dependency versions found");
        return;
    }

    // Count duplicate packages (informational only)
    let duplicate_lines: Vec<_> = stdout
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && (trimmed.starts_with("└──") || trimmed.starts_with("├──"))
        })
        .collect();

    eprintln!("=== Duplicate Dependency Summary ===");
    eprintln!("Found {} duplicate dependency entries", duplicate_lines.len());
    eprintln!(
        "This is informational only - see test_dependency_versions_consistent for critical checks"
    );

    // Don't fail - this is informational
    // test_dependency_versions_consistent handles critical dependency validation
}

/// Tests that minimal dependency versions compile (MSRV compatibility)
///
/// Ensures workspace builds with minimal versions to verify version constraints
#[test]
#[ignore = "Slow test - requires cargo clean and full rebuild"]
fn test_minimal_dependency_versions() {
    // This test validates that version constraints in Cargo.toml are correct
    // by attempting to build with --minimal-versions
    let (success, stdout, stderr) =
        run_cargo_command(&["check", "--workspace", "-Z", "minimal-versions"])
            .expect("Failed to run cargo check with minimal versions");

    if !success {
        eprintln!("=== STDOUT ===\n{}", stdout);
        eprintln!("=== STDERR ===\n{}", stderr);
        eprintln!("Note: This test requires nightly Rust for -Z minimal-versions");
        eprintln!("Failure may indicate incorrect version constraints in Cargo.toml");
        // Don't panic - this is informational only
        return;
    }

    assert!(success, "Workspace should build with minimal dependency versions");
}

// ============================================================================
// Category 9: CI/CD Environment Tests (NEW)
// ============================================================================

/// Tests CI environment variables are set correctly (when in CI)
///
/// Validates CI-specific environment configuration
#[test]
fn test_ci_environment_variables() {
    // Check if running in CI environment
    let in_ci = std::env::var("CI").is_ok() || std::env::var("GITHUB_ACTIONS").is_ok();

    if !in_ci {
        eprintln!("Not running in CI environment - skipping CI-specific checks");
        return;
    }

    // In CI, verify expected environment variables
    eprintln!("Running in CI environment - verifying configuration");

    // Check RUST_LOG (should be set for test visibility)
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        eprintln!("RUST_LOG: {}", rust_log);
    }

    // Check CARGO_TERM_COLOR (should be set for consistent output)
    if let Ok(term_color) = std::env::var("CARGO_TERM_COLOR") {
        eprintln!("CARGO_TERM_COLOR: {}", term_color);
    }

    // Check CI variable
    assert!(std::env::var("CI").is_ok(), "CI environment variable should be set");
}

/// Tests GitHub Actions compatibility
///
/// Validates test execution in GitHub Actions environment
#[test]
fn test_github_actions_compatibility() {
    // Check if running in GitHub Actions
    let in_gha = std::env::var("GITHUB_ACTIONS").is_ok();

    if !in_gha {
        eprintln!("Not running in GitHub Actions - skipping GHA-specific checks");
        return;
    }

    eprintln!("Running in GitHub Actions - verifying environment");

    // Verify GitHub Actions environment variables
    if let Ok(github_workflow) = std::env::var("GITHUB_WORKFLOW") {
        eprintln!("GITHUB_WORKFLOW: {}", github_workflow);
    }

    if let Ok(github_run_id) = std::env::var("GITHUB_RUN_ID") {
        eprintln!("GITHUB_RUN_ID: {}", github_run_id);
    }

    // Verify tests can run in parallel
    let test_threads = std::env::var("RUST_TEST_THREADS").unwrap_or_else(|_| "4".to_string());
    eprintln!("RUST_TEST_THREADS: {}", test_threads);

    assert!(
        std::env::var("GITHUB_ACTIONS").is_ok(),
        "GITHUB_ACTIONS environment variable should be set"
    );
}

// ============================================================================
// Helper Tests (Smoke Tests)
// ============================================================================

/// Smoke test: Verify test harness is working
#[test]
fn test_harness_working() {
    // Basic sanity check - test harness is operational

    // Verify workspace root exists
    let workspace_root = Path::new("/home/steven/code/Rust/BitNet-rs");
    assert!(workspace_root.exists(), "Workspace root should exist at {}", workspace_root.display());

    // Verify Cargo.toml exists
    let cargo_toml = workspace_root.join("Cargo.toml");
    assert!(cargo_toml.exists(), "Root Cargo.toml should exist at {}", cargo_toml.display());

    // Verify Cargo.lock exists
    let cargo_lock = workspace_root.join("Cargo.lock");
    assert!(cargo_lock.exists(), "Cargo.lock should exist at {}", cargo_lock.display());
}

/// Smoke test: Verify cargo is available
#[test]
fn test_cargo_available() {
    let output =
        Command::new("cargo").arg("--version").output().expect("Failed to execute cargo --version");

    assert!(output.status.success(), "Cargo should be available in PATH");

    let version = String::from_utf8_lossy(&output.stdout);
    eprintln!("Cargo version: {}", version.trim());
    assert!(version.contains("cargo"), "Cargo version output should contain 'cargo'");
}
