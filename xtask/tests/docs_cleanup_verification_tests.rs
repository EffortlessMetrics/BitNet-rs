//! Documentation Cleanup Verification Tests
//!
//! Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md
//! Validates comprehensive documentation cleanup and messaging standards compliance
//! across BitNet.rs codebase (AC1-AC10).
//!
//! Test Categories:
//! - AC1: Zero "when available" phrasing verification
//! - AC2: Consistent "detected at build time" terminology
//! - AC3: Runtime fallback with rebuild guidance
//! - AC4: RepairMode help text documentation
//! - AC5: Exit code reference table parsing
//! - AC6: Error message 4-part structure validation
//! - AC7: Help text 8-section template validation
//! - AC8: Verbose output with timestamps
//! - AC9: CLAUDE.md workflow validation
//! - AC10: cpp-setup.md dual-backend verification

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

// ============================================================================
// Test Helpers
// ============================================================================

/// Get repository root directory
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask should have parent directory")
        .to_path_buf()
}

/// Run grep command and return output
fn run_grep(pattern: &str, paths: &[&str], include: &[&str]) -> Result<String> {
    let repo = repo_root();
    let mut cmd = Command::new("grep");
    cmd.current_dir(&repo);
    cmd.arg("-rn");
    cmd.arg(pattern);

    for path in paths {
        cmd.arg(path);
    }

    for inc in include {
        cmd.arg("--include").arg(inc);
    }

    let output = cmd.output().context("Failed to run grep")?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Run cargo command with preflight --help
fn run_preflight_help() -> Result<String> {
    let repo = repo_root();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&repo);
    cmd.args(["run", "-p", "xtask", "--", "preflight", "--help"]);

    let output = cmd.output().context("Failed to run preflight --help")?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Check if file exists in repository
fn file_exists(relative_path: &str) -> bool {
    let repo = repo_root();
    repo.join(relative_path).exists()
}

/// Read file contents from repository
fn read_file(relative_path: &str) -> Result<String> {
    let repo = repo_root();
    let path = repo.join(relative_path);
    std::fs::read_to_string(&path).context(format!("Failed to read file: {}", relative_path))
}

// ============================================================================
// AC1: Zero "When Available" Phrasing Verification
// ============================================================================

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac1
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac1_grep_no_when_available_in_rust_sources() -> Result<()> {
    let output =
        run_grep("when available", &["xtask/src", "bitnet-cli/src", "crates/*/src"], &["*.rs"])?;

    assert!(output.is_empty(), "Found 'when available' in Rust sources:\n{}", output);
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac1
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac1_grep_no_if_available_in_rust_sources() -> Result<()> {
    let output =
        run_grep("if available", &["xtask/src", "bitnet-cli/src", "crates/*/src"], &["*.rs"])?;

    assert!(output.is_empty(), "Found 'if available' in Rust sources:\n{}", output);
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac1
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac1_grep_no_runtime_availability_in_rust_sources() -> Result<()> {
    let output = run_grep(
        "runtime availability",
        &["xtask/src", "bitnet-cli/src", "crates/*/src"],
        &["*.rs"],
    )?;

    assert!(output.is_empty(), "Found 'runtime availability' in Rust sources:\n{}", output);
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac1
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac1_grep_no_as_available_in_rust_sources() -> Result<()> {
    let output =
        run_grep("as available", &["xtask/src", "bitnet-cli/src", "crates/*/src"], &["*.rs"])?;

    assert!(output.is_empty(), "Found 'as available' in Rust sources:\n{}", output);
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac1
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac1_grep_no_when_available_in_documentation() -> Result<()> {
    let output = run_grep("when available", &["docs", "CLAUDE.md"], &["*.md"])?;

    assert!(output.is_empty(), "Found 'when available' in documentation:\n{}", output);
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac1
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac1_grep_no_if_available_in_documentation() -> Result<()> {
    let output = run_grep("if available", &["docs", "CLAUDE.md"], &["*.md"])?;

    assert!(output.is_empty(), "Found 'if available' in documentation:\n{}", output);
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac1
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac1_automated_verification_script_exists() {
    assert!(
        file_exists("scripts/verify_no_ambiguous_phrasing.sh"),
        "Missing verification script: scripts/verify_no_ambiguous_phrasing.sh"
    );
}

// ============================================================================
// AC2: Consistent "Detected at Build Time" Terminology
// ============================================================================

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac2
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac2_preflight_uses_detected_at_build_time() -> Result<()> {
    let content = read_file("xtask/src/crossval/preflight.rs")?;

    assert!(
        content.contains("detected at build time"),
        "preflight.rs should use 'detected at build time' terminology"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac2
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac2_feature_flags_use_compiled_if_terminology() -> Result<()> {
    let content = read_file("docs/explanation/FEATURES.md")?;

    assert!(
        content.contains("compiled if") || content.contains("compiled when"),
        "FEATURES.md should use 'compiled if' terminology for feature flags"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac2
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac2_runtime_library_resolution_terminology() -> Result<()> {
    let content = read_file("xtask/src/crossval/preflight.rs")?;

    assert!(
        content.contains("runtime library resolution")
            || content.contains("resolved via")
            || content.contains("LD_LIBRARY_PATH")
            || content.contains("rpath"),
        "preflight.rs should explain runtime library resolution mechanism"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac2
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac2_backend_libraries_found_terminology() -> Result<()> {
    let content = read_file("xtask/src/crossval/preflight.rs")?;

    assert!(
        content.contains("backend libraries") || content.contains("libraries found"),
        "preflight.rs should use 'backend libraries found' terminology"
    );
    Ok(())
}

// ============================================================================
// AC3: Runtime Fallback with Rebuild Guidance
// ============================================================================

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac3
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac3_rebuild_instructions_in_error_messages() -> Result<()> {
    let content = read_file("xtask/src/crossval/preflight.rs")?;

    assert!(
        content.contains("cargo clean -p xtask")
            && content.contains("cargo build -p xtask --features crossval-all"),
        "preflight.rs should include exact rebuild commands in error messages"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac3
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac3_environment_override_documented() -> Result<()> {
    let content = read_file("xtask/src/crossval/preflight.rs")?;

    assert!(
        content.contains("BITNET_CROSSVAL_LIBDIR") || content.contains("BITNET_CPP_DIR"),
        "preflight.rs should document environment variable overrides"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac3
#[test]
#[ignore] // TODO: Implement after documentation cleanup
fn test_ac3_rebuild_guidance_explains_build_time_constants() -> Result<()> {
    let content = read_file("xtask/src/crossval/preflight.rs")?;

    assert!(
        content.contains("build-time") && content.contains("constant"),
        "preflight.rs should explain why rebuild is needed (build-time constants)"
    );
    Ok(())
}

// ============================================================================
// AC4: RepairMode Help Text Documentation
// ============================================================================

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac4
#[test]
#[ignore] // TODO: Implement after preflight help text implementation
fn test_ac4_help_text_documents_repair_mode_auto() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("auto") && help.contains("repair"),
        "Help text should document RepairMode::Auto"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac4
#[test]
#[ignore] // TODO: Implement after preflight help text implementation
fn test_ac4_help_text_documents_repair_mode_never() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("never") && help.contains("repair"),
        "Help text should document RepairMode::Never"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac4
#[test]
#[ignore] // TODO: Implement after preflight help text implementation
fn test_ac4_help_text_documents_repair_mode_always() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("always") && help.contains("repair"),
        "Help text should document RepairMode::Always"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac4
#[test]
#[ignore] // TODO: Implement after preflight help text implementation
fn test_ac4_help_text_documents_ci_aware_defaults() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("CI") || help.contains("GITHUB_ACTIONS"),
        "Help text should document CI-aware RepairMode defaults"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac4
#[test]
#[ignore] // TODO: Implement after preflight help text implementation
fn test_ac4_help_text_includes_repair_flag_examples() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("--repair") && help.contains("EXAMPLES"),
        "Help text should include --repair flag examples"
    );
    Ok(())
}

// ============================================================================
// AC5: Exit Code Reference Table Parsing
// ============================================================================

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac5
#[test]
#[ignore] // TODO: Implement after exit code documentation created
fn test_ac5_exit_codes_reference_file_exists() {
    assert!(
        file_exists("docs/reference/exit-codes.md"),
        "Missing exit code reference: docs/reference/exit-codes.md"
    );
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac5
#[test]
#[ignore] // TODO: Implement after exit code documentation created
fn test_ac5_exit_codes_reference_includes_taxonomy() -> Result<()> {
    let content = read_file("docs/reference/exit-codes.md")?;

    // Check for key exit codes
    for code in &["0", "1", "2", "3", "4", "5", "6"] {
        assert!(
            content.contains(&format!("| {} ", code)) || content.contains(&format!("{} ", code)),
            "Exit code reference should document exit code {}",
            code
        );
    }
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac5
#[test]
#[ignore] // TODO: Implement after preflight help text implementation
fn test_ac5_help_text_includes_exit_codes_section() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("EXIT CODES") || help.contains("Exit Codes"),
        "Help text should include EXIT CODES section"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac5
#[test]
#[ignore] // TODO: Implement after preflight help text implementation
fn test_ac5_help_text_documents_recovery_by_exit_code() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("RECOVERY") || help.contains("Recovery"),
        "Help text should include RECOVERY BY EXIT CODE section"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac5
#[test]
#[ignore] // TODO: Implement after exit code documentation created
fn test_ac5_exit_codes_reference_links_in_help_text() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("docs/reference/exit-codes.md"),
        "Help text should link to docs/reference/exit-codes.md"
    );
    Ok(())
}

// ============================================================================
// AC6: Error Message 4-Part Structure Validation
// ============================================================================

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac6
#[test]
#[ignore] // TODO: Implement after error message templates implemented
fn test_ac6_error_messages_include_status_icon() -> Result<()> {
    let content = read_file("xtask/src/crossval/preflight.rs")?;

    assert!(
        content.contains("❌") || content.contains("✓") || content.contains("⚠️"),
        "Error messages should include status icons"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac6
#[test]
#[ignore] // TODO: Implement after error message templates implemented
fn test_ac6_error_messages_include_error_detail_section() -> Result<()> {
    let content = read_file("xtask/src/crossval/preflight.rs")?;

    assert!(
        content.contains("Error Detail:") || content.contains("Error:"),
        "Error messages should include Error Detail section"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac6
#[test]
#[ignore] // TODO: Implement after error message templates implemented
fn test_ac6_error_messages_include_recovery_steps() -> Result<()> {
    let content = read_file("xtask/src/crossval/preflight.rs")?;

    assert!(
        content.contains("Recovery Steps:") || content.contains("Recovery:"),
        "Error messages should include Recovery Steps section"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac6
#[test]
#[ignore] // TODO: Implement after error message templates implemented
fn test_ac6_error_messages_include_documentation_links() -> Result<()> {
    let content = read_file("xtask/src/crossval/preflight.rs")?;

    assert!(
        content.contains("Documentation:") || content.contains("See:"),
        "Error messages should include Documentation links section"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac6
#[test]
#[ignore] // TODO: Implement after error message templates implemented
fn test_ac6_error_messages_include_exit_code_description() -> Result<()> {
    let content = read_file("xtask/src/crossval/preflight.rs")?;

    assert!(
        content.contains("Exit code:") || content.contains("exit code"),
        "Error messages should include semantic exit code description"
    );
    Ok(())
}

// ============================================================================
// AC7: Help Text 8-Section Template Validation
// ============================================================================

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac7
#[test]
#[ignore] // TODO: Implement after help text standardization
fn test_ac7_help_text_includes_command_name_section() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("PREFLIGHT") || help.starts_with("preflight"),
        "Help text should include COMMAND NAME section"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac7
#[test]
#[ignore] // TODO: Implement after help text standardization
fn test_ac7_help_text_includes_usage_section() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("USAGE:") || help.contains("Usage:"),
        "Help text should include USAGE section"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac7
#[test]
#[ignore] // TODO: Implement after help text standardization
fn test_ac7_help_text_includes_description_section() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("DESCRIPTION:") || help.contains("Description:"),
        "Help text should include DESCRIPTION section"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac7
#[test]
#[ignore] // TODO: Implement after help text standardization
fn test_ac7_help_text_includes_options_section() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("OPTIONS:") || help.contains("Options:"),
        "Help text should include OPTIONS section"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac7
#[test]
#[ignore] // TODO: Implement after help text standardization
fn test_ac7_help_text_includes_examples_section() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("EXAMPLES:") || help.contains("Examples:"),
        "Help text should include EXAMPLES section"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac7
#[test]
#[ignore] // TODO: Implement after help text standardization
fn test_ac7_help_text_includes_exit_codes_section() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("EXIT CODES:") || help.contains("Exit Codes:"),
        "Help text should include EXIT CODES section"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac7
#[test]
#[ignore] // TODO: Implement after help text standardization
fn test_ac7_help_text_includes_recovery_section() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("RECOVERY BY EXIT CODE:") || help.contains("Recovery:"),
        "Help text should include RECOVERY BY EXIT CODE section"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac7
#[test]
#[ignore] // TODO: Implement after help text standardization
fn test_ac7_help_text_includes_documentation_section() -> Result<()> {
    let help = run_preflight_help()?;

    assert!(
        help.contains("DOCUMENTATION:") || help.contains("Documentation:"),
        "Help text should include DOCUMENTATION section"
    );
    Ok(())
}

// ============================================================================
// AC8: Verbose Output with Timestamps
// ============================================================================

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac8
#[test]
#[ignore] // TODO: Implement after verbose output enhancement
fn test_ac8_verbose_output_includes_timestamps() -> Result<()> {
    let repo = repo_root();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&repo);
    cmd.args(["run", "-p", "xtask", "--", "preflight", "--backend", "bitnet", "--verbose"]);

    let output = cmd.output().context("Failed to run preflight --verbose")?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check for timestamp format: [YYYY-MM-DD HH:MM:SS]
    assert!(
        stdout.contains("[2") && stdout.contains("]"),
        "Verbose output should include timestamps in format [YYYY-MM-DD HH:MM:SS]"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac8
#[test]
#[ignore] // TODO: Implement after verbose output enhancement
fn test_ac8_verbose_output_includes_phase_markers() -> Result<()> {
    let repo = repo_root();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&repo);
    cmd.args(["run", "-p", "xtask", "--", "preflight", "--backend", "bitnet", "--verbose"]);

    let output = cmd.output().context("Failed to run preflight --verbose")?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check for phase markers
    let has_config = stdout.contains("Config:") || stdout.contains("Operation:");
    let has_progress = stdout.contains("Progress:");
    let has_status = stdout.contains("Status:");
    let has_result = stdout.contains("Result:");

    assert!(
        has_config || has_progress || has_status || has_result,
        "Verbose output should include phase markers (Config, Progress, Status, Result)"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac8
#[test]
#[ignore] // TODO: Implement after verbose output enhancement
fn test_ac8_verbose_output_timestamp_format_consistency() -> Result<()> {
    let repo = repo_root();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&repo);
    cmd.args(["run", "-p", "xtask", "--", "preflight", "--backend", "bitnet", "--verbose"]);

    let output = cmd.output().context("Failed to run preflight --verbose")?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Use regex to validate timestamp format consistency
    let timestamp_pattern =
        regex::Regex::new(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]").expect("Valid regex pattern");

    assert!(
        timestamp_pattern.is_match(&stdout),
        "Verbose output should use consistent timestamp format [YYYY-MM-DD HH:MM:SS]"
    );
    Ok(())
}

// ============================================================================
// AC9: CLAUDE.md Workflow Validation
// ============================================================================

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac9
#[test]
#[ignore] // TODO: Implement after CLAUDE.md updates
fn test_ac9_claude_md_documents_preflight_auto_repair() -> Result<()> {
    let content = read_file("CLAUDE.md")?;

    assert!(
        content.contains("preflight") && content.contains("repair"),
        "CLAUDE.md should document preflight auto-repair workflows"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac9
#[test]
#[ignore] // TODO: Implement after CLAUDE.md updates
fn test_ac9_claude_md_documents_repair_mode_variants() -> Result<()> {
    let content = read_file("CLAUDE.md")?;

    assert!(
        content.contains("RepairMode") || (content.contains("auto") && content.contains("never")),
        "CLAUDE.md should document RepairMode variants (Auto, Never, Always)"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac9
#[test]
#[ignore] // TODO: Implement after CLAUDE.md updates
fn test_ac9_claude_md_includes_dual_backend_patterns() -> Result<()> {
    let content = read_file("CLAUDE.md")?;

    assert!(
        content.contains("bitnet.cpp") && content.contains("llama.cpp"),
        "CLAUDE.md should document dual-backend patterns (bitnet.cpp + llama.cpp)"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac9
#[test]
#[ignore] // TODO: Implement after CLAUDE.md updates
fn test_ac9_claude_md_includes_exit_code_handling() -> Result<()> {
    let content = read_file("CLAUDE.md")?;

    assert!(
        content.contains("exit code") || content.contains("Exit") && content.contains("3"),
        "CLAUDE.md should document exit code handling for CI/CD integration"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac9
#[test]
#[ignore] // TODO: Implement after CLAUDE.md updates
fn test_ac9_claude_md_no_ambiguous_when_available() -> Result<()> {
    let content = read_file("CLAUDE.md")?;

    assert!(
        !content.contains("when available"),
        "CLAUDE.md should not contain ambiguous 'when available' phrasing"
    );
    Ok(())
}

// ============================================================================
// AC10: cpp-setup.md Dual-Backend Verification
// ============================================================================

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac10
#[test]
#[ignore] // TODO: Implement after cpp-setup.md updates
fn test_ac10_cpp_setup_includes_quick_start_auto_provisioning() -> Result<()> {
    let content = read_file("docs/howto/cpp-setup.md")?;

    assert!(
        content.contains("Quick Start") && content.contains("Auto-Provisioning"),
        "cpp-setup.md should include Quick Start: Auto-Provisioning section"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac10
#[test]
#[ignore] // TODO: Implement after cpp-setup.md updates
fn test_ac10_cpp_setup_documents_backend_selection() -> Result<()> {
    let content = read_file("docs/howto/cpp-setup.md")?;

    assert!(
        content.contains("Backend Selection")
            || (content.contains("bitnet.cpp") && content.contains("llama.cpp")),
        "cpp-setup.md should document backend selection (bitnet.cpp vs llama.cpp)"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac10
#[test]
#[ignore] // TODO: Implement after cpp-setup.md updates
fn test_ac10_cpp_setup_includes_preflight_examples() -> Result<()> {
    let content = read_file("docs/howto/cpp-setup.md")?;

    assert!(
        content.contains("preflight") && content.contains("--repair"),
        "cpp-setup.md should include preflight auto-repair examples"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac10
#[test]
#[ignore] // TODO: Implement after cpp-setup.md updates
fn test_ac10_cpp_setup_includes_manual_setup_alternatives() -> Result<()> {
    let content = read_file("docs/howto/cpp-setup.md")?;

    assert!(
        content.contains("Manual Setup") || content.contains("manual"),
        "cpp-setup.md should provide manual setup alternatives"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac10
#[test]
#[ignore] // TODO: Implement after cpp-setup.md updates
fn test_ac10_cpp_setup_documents_troubleshooting() -> Result<()> {
    let content = read_file("docs/howto/cpp-setup.md")?;

    assert!(
        content.contains("Troubleshooting")
            || content.contains("Problem:")
            || content.contains("Exit"),
        "cpp-setup.md should include troubleshooting section with exit code recovery"
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac10
#[test]
#[ignore] // TODO: Implement after cpp-setup.md updates
fn test_ac10_cpp_setup_no_ambiguous_when_available() -> Result<()> {
    let content = read_file("docs/howto/cpp-setup.md")?;

    assert!(
        !content.contains("when available"),
        "cpp-setup.md should not contain ambiguous 'when available' phrasing"
    );
    Ok(())
}

// ============================================================================
// Integration Tests: Script Execution
// ============================================================================

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac1
#[test]
#[ignore] // TODO: Implement after verification script created
fn test_integration_verify_no_ambiguous_phrasing_script_succeeds() -> Result<()> {
    let repo = repo_root();
    let script_path = repo.join("scripts/verify_no_ambiguous_phrasing.sh");

    if !script_path.exists() {
        return Err(anyhow::anyhow!(
            "Verification script not found: scripts/verify_no_ambiguous_phrasing.sh"
        ));
    }

    let mut cmd = Command::new("bash");
    cmd.current_dir(&repo);
    cmd.arg(script_path);

    let output = cmd.output().context("Failed to run verify_no_ambiguous_phrasing.sh")?;

    assert!(
        output.status.success(),
        "verify_no_ambiguous_phrasing.sh should exit 0 (no ambiguous phrasing found)\nStdout: {}\nStderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac5
#[test]
#[ignore] // TODO: Implement after exit code verification script created
fn test_integration_verify_exit_codes_script_succeeds() -> Result<()> {
    let repo = repo_root();
    let script_path = repo.join("scripts/verify_exit_codes.sh");

    if !script_path.exists() {
        return Err(anyhow::anyhow!("Verification script not found: scripts/verify_exit_codes.sh"));
    }

    let mut cmd = Command::new("bash");
    cmd.current_dir(&repo);
    cmd.arg(script_path);

    let output = cmd.output().context("Failed to run verify_exit_codes.sh")?;

    assert!(
        output.status.success(),
        "verify_exit_codes.sh should exit 0 (all exit codes consistent)\nStdout: {}\nStderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac1
#[test]
#[ignore] // TODO: Implement after replacement script created
fn test_integration_replace_ambiguous_phrasing_script_exists() {
    assert!(
        file_exists("scripts/replace_ambiguous_phrasing.sh"),
        "Missing replacement script: scripts/replace_ambiguous_phrasing.sh"
    );
}

// ============================================================================
// Property-Based Tests: Error Message Structure
// ============================================================================

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac6
#[test]
#[ignore] // TODO: Implement after error message templates implemented
fn test_property_all_error_messages_have_4_part_structure() -> Result<()> {
    let content = read_file("xtask/src/crossval/preflight.rs")?;

    // Extract all error message patterns
    let error_patterns = vec![
        "Backend 'bitnet.cpp' UNAVAILABLE",
        "Backend 'llama.cpp' UNAVAILABLE",
        "Auto-repair failed",
    ];

    for pattern in error_patterns {
        if content.contains(pattern) {
            // Check that error message context includes 4 parts
            let has_icon = content.contains("❌") || content.contains("⚠️");
            let has_error_detail = content.contains("Error Detail:") || content.contains("Error:");
            let has_recovery = content.contains("Recovery Steps:") || content.contains("Recovery:");
            let has_docs = content.contains("Documentation:") || content.contains("See:");

            assert!(
                has_icon && has_error_detail && has_recovery && has_docs,
                "Error message '{}' should follow 4-part structure (Icon, Error Detail, Recovery Steps, Documentation)",
                pattern
            );
        }
    }
    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac7
#[test]
#[ignore] // TODO: Implement after help text standardization
fn test_property_all_commands_use_8_section_help_template() -> Result<()> {
    let commands = vec!["preflight", "setup-cpp-auto", "crossval-per-token"];

    for command in commands {
        let repo = repo_root();
        let mut cmd = Command::new("cargo");
        cmd.current_dir(&repo);
        cmd.args(["run", "-p", "xtask", "--", command, "--help"]);

        let output = cmd.output().context(format!("Failed to run {} --help", command))?;
        let help = String::from_utf8_lossy(&output.stdout);

        // Check for 8 sections
        let sections = vec![
            "USAGE:",
            "DESCRIPTION:",
            "OPTIONS:",
            "EXAMPLES:",
            "EXIT CODES:",
            "RECOVERY",
            "DOCUMENTATION:",
        ];

        for section in &sections {
            assert!(
                help.contains(section),
                "Command '{}' help text should include {} section",
                command,
                section
            );
        }
    }
    Ok(())
}

// ============================================================================
// Cross-Reference Tests: Documentation Consistency
// ============================================================================

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac9-ac10
#[test]
#[ignore] // TODO: Implement after documentation updates
fn test_cross_reference_claude_md_and_cpp_setup_consistency() -> Result<()> {
    let claude_content = read_file("CLAUDE.md")?;
    let cpp_setup_content = read_file("docs/howto/cpp-setup.md")?;

    // Both should document preflight auto-repair
    assert!(
        claude_content.contains("preflight") && cpp_setup_content.contains("preflight"),
        "Both CLAUDE.md and cpp-setup.md should document preflight command"
    );

    // Both should document dual backends
    assert!(
        claude_content.contains("bitnet.cpp") && cpp_setup_content.contains("bitnet.cpp"),
        "Both CLAUDE.md and cpp-setup.md should document bitnet.cpp backend"
    );

    assert!(
        claude_content.contains("llama.cpp") && cpp_setup_content.contains("llama.cpp"),
        "Both CLAUDE.md and cpp-setup.md should document llama.cpp backend"
    );

    Ok(())
}

/// Tests feature spec: docs/specs/docs-messaging-standards-cleanup.md#ac5-ac6
#[test]
#[ignore] // TODO: Implement after exit code documentation and error templates
fn test_cross_reference_exit_codes_in_error_messages_and_help() -> Result<()> {
    let preflight_content = read_file("xtask/src/crossval/preflight.rs")?;
    let exit_codes_doc = read_file("docs/reference/exit-codes.md")?;

    // Exit codes mentioned in error messages should be documented
    let exit_code_pattern = regex::Regex::new(r"Exit code: (\d+)").expect("Valid regex pattern");

    for cap in exit_code_pattern.captures_iter(&preflight_content) {
        let code = &cap[1];
        assert!(
            exit_codes_doc.contains(&format!("| {} ", code))
                || exit_codes_doc.contains(&format!("{} ", code)),
            "Exit code {} mentioned in preflight.rs should be documented in exit-codes.md",
            code
        );
    }

    Ok(())
}
