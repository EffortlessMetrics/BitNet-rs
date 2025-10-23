//! AC8: Documentation Validation Tests (Issue #469)
//!
//! Tests feature spec: docs/explanation/issue-469-spec.md#ac8-docs-readme-quick-start
//! API contract: docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md#ac8
//!
//! This test validates documentation updates for QK256 quick-start guidance.

#![cfg(test)]

#[allow(unused_imports)]
use std::fs;
#[allow(unused_imports)]
use std::path::Path;

/// AC8: README.md contains QK256 quick-start section
///
/// Tests that README.md includes QK256-specific quick-start examples.
///
/// # Fixture Requirements
/// - README.md file in repository root
///
/// # Expected Behavior
/// - README.md has "QK256 Format" section
/// - Section includes: format explanation, command example, link to docs/howto/use-qk256-models.md
/// - Command example uses --strict-loader flag
#[test]
fn test_readme_qk256_quickstart_section() {
    // AC8: Verify README.md has QK256 quick-start section
    // FIXTURE NEEDED: README.md with QK256 section
    //
    // Expected content:
    //   #### QK256 Format (GGML I2_S, 256-element blocks)
    //   ```bash
    //   cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf
    //   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
    //     --model models/ggml-model-i2_s.gguf \
    //     --strict-loader \
    //     --prompt "Test" \
    //     --max-tokens 16
    //   ```
    //   **Learn More:**
    //   - [QK256 Usage Guide](docs/howto/use-qk256-models.md)

    let readme = fs::read_to_string("../README.md").expect("AC8: README.md should exist");

    assert!(readme.contains("QK256 Format"), "AC8: README should mention QK256 Format");

    assert!(readme.contains("--strict-loader"), "AC8: README should show --strict-loader flag");

    assert!(
        readme.contains("docs/howto/use-qk256-models.md"),
        "AC8: README should link to QK256 guide"
    );

    println!("// AC8: README.md includes QK256 quick-start section");
}

/// AC8: docs/quickstart.md contains QK256 section
///
/// Tests that docs/quickstart.md includes "Using QK256 Models" section.
///
/// # Fixture Requirements
/// - docs/quickstart.md file
///
/// # Expected Behavior
/// - Section titled "Using QK256 Models (GGML I2_S)"
/// - Covers: automatic format detection, strict loader mode, cross-validation
/// - Includes command examples with --strict-loader
#[test]
fn test_quickstart_qk256_section() {
    // AC8: Verify docs/quickstart.md has QK256 section
    // FIXTURE NEEDED: docs/quickstart.md with QK256 section
    //
    // Expected content:
    //   ## Using QK256 Models (GGML I2_S)
    //
    //   ### Automatic Format Detection
    //   The loader automatically detects QK256 format based on tensor size...
    //
    //   ### Strict Loader Mode
    //   Enforce exact QK256 alignment (reject tensors with >0.1% size deviation):
    //   ```bash
    //   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
    //     --model models/ggml-model-i2_s.gguf \
    //     --strict-loader \
    //     --prompt "Test" \
    //     --max-tokens 16
    //   ```

    let quickstart =
        fs::read_to_string("../docs/quickstart.md").expect("AC8: docs/quickstart.md should exist");

    assert!(quickstart.contains("Using QK256 Models"), "AC8: Quickstart should have QK256 section");

    assert!(
        quickstart.contains("Automatic Format Detection"),
        "AC8: Should explain automatic detection"
    );

    assert!(quickstart.contains("Strict Loader Mode"), "AC8: Should explain strict mode");

    println!("// AC8: docs/quickstart.md includes QK256 section");
}

/// AC8: Documentation cross-links are valid
///
/// Tests that all cross-links in README.md and docs/quickstart.md are valid.
///
/// # Fixture Requirements
/// - README.md and docs/quickstart.md with links
///
/// # Expected Behavior
/// - All Markdown links point to existing files
/// - No broken links to docs/howto/use-qk256-models.md
/// - No broken links to docs/explanation/i2s-dual-flavor.md
#[test]
fn test_documentation_cross_links_valid() {
    // AC8: Verify all documentation cross-links are valid
    // This test verifies that key documentation links referenced in the tests exist

    // Check README.md links
    assert!(
        Path::new("../docs/howto/use-qk256-models.md").exists(),
        "AC8: README should link to existing QK256 guide"
    );

    assert!(
        Path::new("../docs/explanation/i2s-dual-flavor.md").exists(),
        "AC8: README should link to existing dual-flavor architecture doc"
    );

    // Check docs/quickstart.md links
    assert!(
        Path::new("../docs/howto/use-qk256-models.md").exists(),
        "AC8: Quickstart should link to existing QK256 guide"
    );

    println!("// AC8: Documentation cross-links are valid");
}

/// AC8: README.md references dual-flavor architecture doc
///
/// Tests that README.md links to docs/explanation/i2s-dual-flavor.md.
///
/// # Fixture Requirements
/// - README.md with architecture doc link
///
/// # Expected Behavior
/// - README.md contains link to docs/explanation/i2s-dual-flavor.md
/// - Link appears in "Learn More" or similar section
/// - Link text mentions "Dual I2_S Flavor Architecture"
#[test]
fn test_readme_dual_flavor_architecture_link() {
    // AC8: Verify README.md links to dual-flavor architecture doc
    // FIXTURE NEEDED: README.md with architecture doc link
    //
    // Expected link:
    //   - [Dual I2_S Flavor Architecture](docs/explanation/i2s-dual-flavor.md)

    let readme = fs::read_to_string("../README.md").expect("AC8: README.md should exist");

    assert!(
        readme.contains("docs/explanation/i2s-dual-flavor.md"),
        "AC8: README should link to dual-flavor architecture doc"
    );

    assert!(
        readme.contains("Dual I2_S Flavor") || readme.contains("I2_S format architecture"),
        "AC8: Link should mention dual-flavor or I2_S architecture"
    );

    println!("// AC8: README.md includes dual-flavor architecture link");
}

/// AC8: docs/quickstart.md cross-validation examples
///
/// Tests that docs/quickstart.md includes cross-validation command examples.
///
/// # Fixture Requirements
/// - docs/quickstart.md with cross-validation section
///
/// # Expected Behavior
/// - Section covers: setting BITNET_CPP_DIR, running crossval, receipt validation
/// - Includes jq command for checking parity metrics
/// - Shows expected receipt output structure
#[test]
fn test_quickstart_crossval_examples() {
    // AC8: Verify docs/quickstart.md has cross-validation examples
    let quickstart =
        fs::read_to_string("../docs/quickstart.md").expect("AC8: docs/quickstart.md should exist");

    assert!(
        quickstart.contains("Cross-Validation") || quickstart.contains("cross-validation"),
        "AC8: Quickstart should have cross-validation section"
    );

    assert!(
        quickstart.contains("BITNET_CPP_DIR"),
        "AC8: Should mention BITNET_CPP_DIR environment variable"
    );

    assert!(quickstart.contains("parity_smoke.sh"), "AC8: Should show parity smoke script usage");

    assert!(
        quickstart.contains("crossval") || quickstart.contains("parity"),
        "AC8: Should include cross-validation or parity commands"
    );

    println!("// AC8: Quickstart includes cross-validation examples");
}

/// AC8: Quick-start examples are executable
///
/// Tests that command examples in README.md and docs/quickstart.md are valid.
///
/// # Fixture Requirements
/// - scripts/validate_quickstart_examples.sh (validation script)
///
/// # Expected Behavior
/// - Validation script extracts code blocks from Markdown
/// - Executes commands in isolated environment
/// - Verifies commands succeed without errors
#[test]
#[ignore = "Integration test - requires model download and execution"]
fn test_quickstart_examples_executable() {
    // AC8: Verify quick-start examples are executable
    // FIXTURE NEEDED: scripts/validate_quickstart_examples.sh
    //
    // Expected validation script:
    //   #!/bin/bash
    //   # Extract code blocks from Markdown
    //   # Execute commands with BITNET_DETERMINISTIC=1
    //   # Verify exit codes and output

    // TODO: Implement once validation script is available
    // use std::process::Command;
    //
    // let status = Command::new("./scripts/validate_quickstart_examples.sh")
    //     .status()?;
    //
    // assert!(status.success(), "AC8: Quick-start examples should be executable");

    panic!(
        "AC8: Quick-start example validation not yet implemented. \
         Expected: scripts/validate_quickstart_examples.sh validates Markdown code blocks."
    );
}

/// AC8: QK256 usage documentation link validation
///
/// Tests that docs/howto/use-qk256-models.md exists and is referenced.
///
/// # Fixture Requirements
/// - docs/howto/use-qk256-models.md file
///
/// # Expected Behavior
/// - File exists at docs/howto/use-qk256-models.md
/// - Referenced from README.md and docs/quickstart.md
/// - Contains comprehensive QK256 usage guide
#[test]
fn test_qk256_usage_doc_exists_and_linked() {
    // AC8: Verify docs/howto/use-qk256-models.md exists and is linked
    assert!(
        Path::new("../docs/howto/use-qk256-models.md").exists(),
        "AC8: QK256 usage guide should exist at docs/howto/use-qk256-models.md"
    );

    let quickstart =
        fs::read_to_string("../docs/quickstart.md").expect("AC8: docs/quickstart.md should exist");
    assert!(
        quickstart.contains("howto/use-qk256-models.md"),
        "AC8: Quickstart should link to QK256 usage guide"
    );

    println!("// AC8: QK256 usage documentation exists and is linked");
}

/// AC8: Strict loader mode usage documentation
///
/// Tests that both README.md and docs/quickstart.md explain --strict-loader usage.
///
/// # Fixture Requirements
/// - README.md and docs/quickstart.md with --strict-loader documentation
///
/// # Expected Behavior
/// - Both docs explain when to use --strict-loader
/// - Examples show --strict-loader flag usage
/// - Documents strict mode behavior (reject >0.1% deviation)
#[test]
fn test_strict_loader_mode_documentation() {
    // AC8: Verify --strict-loader documentation
    // FIXTURE NEEDED: README.md and docs/quickstart.md with strict mode docs
    //
    // Expected content:
    //   **Use strict mode when:**
    //   - Validating model exports for production deployment
    //   - Debugging model loading issues
    //   - Running CI/CD parity tests

    let readme = fs::read_to_string("../README.md").expect("AC8: README.md should exist");

    assert!(readme.contains("--strict-loader"), "AC8: README should mention --strict-loader");

    let quickstart =
        fs::read_to_string("../docs/quickstart.md").expect("AC8: docs/quickstart.md should exist");

    assert!(
        quickstart.contains("Strict Loader Mode"),
        "AC8: Quickstart should explain strict mode"
    );

    assert!(quickstart.contains("Use strict mode when"), "AC8: Should document use cases");

    println!("// AC8: Strict loader mode is documented");
}

/// AC8: Documentation index updated with QK256 links
///
/// Tests that docs/README.md includes QK256 documentation in index.
///
/// # Fixture Requirements
/// - docs/README.md (documentation index)
///
/// # Expected Behavior
/// - docs/README.md lists docs/howto/use-qk256-models.md
/// - docs/README.md lists docs/explanation/i2s-dual-flavor.md
/// - Links categorized appropriately (How-To, Explanation)
#[test]
fn test_documentation_index_qk256_links() {
    // AC8: Verify docs/README.md includes QK256 documentation
    let docs_index =
        fs::read_to_string("../docs/README.md").expect("AC8: docs/README.md should exist");

    assert!(
        docs_index.contains("howto/use-qk256-models.md"),
        "AC8: Documentation index should list QK256 usage guide"
    );

    assert!(
        docs_index.contains("explanation/i2s-dual-flavor.md"),
        "AC8: Documentation index should list dual-flavor architecture doc"
    );

    println!("// AC8: Documentation index includes QK256 links");
}

/// AC8: Quick-start reproducibility test
///
/// Tests that quick-start examples produce reproducible results.
///
/// # Fixture Requirements
/// - Model and tokenizer fixtures
///
/// # Expected Behavior
/// - Running quick-start example with BITNET_DETERMINISTIC=1 produces same output
/// - Output matches documented example
/// - Seed control works as expected (--seed 42)
#[test]
#[ignore = "Integration test - requires model fixtures and execution"]
fn test_quickstart_example_reproducibility() {
    // AC8: Verify quick-start examples are reproducible
    // FIXTURE NEEDED: Model and tokenizer fixtures
    //
    // Expected:
    //   use std::process::Command;
    //
    //   let output1 = Command::new("cargo")
    //       .args(&["run", "-p", "bitnet-cli", "--no-default-features", "--features", "cpu,full-cli", "--",
    //               "run", "--model", "tests/fixtures/model.gguf", "--prompt", "What is 2+2?",
    //               "--max-tokens", "16", "--seed", "42"])
    //       .env("BITNET_DETERMINISTIC", "1")
    //       .output()?;
    //
    //   let output2 = Command::new("cargo")
    //       .args(&["run", "-p", "bitnet-cli", "--no-default-features", "--features", "cpu,full-cli", "--",
    //               "run", "--model", "tests/fixtures/model.gguf", "--prompt", "What is 2+2?",
    //               "--max-tokens", "16", "--seed", "42"])
    //       .env("BITNET_DETERMINISTIC", "1")
    //       .output()?;
    //
    //   assert_eq!(output1.stdout, output2.stdout, "AC8: Quick-start examples should be reproducible");

    panic!(
        "AC8: Quick-start example reproducibility test not yet implemented. \
         Expected: Deterministic inference with BITNET_DETERMINISTIC=1 and --seed."
    );
}
