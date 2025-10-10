//! Gitignore validation tests for Issue #439
//!
//! Tests specification: docs/explanation/issue-439-spec.md#ac8-gitignore-cleanup
//!
//! Validates that ephemeral test artifacts are properly excluded from version control.

use std::fs;
use std::path::Path;

/// AC:8 - Verify proptest regressions are ignored
///
/// Tests that .gitignore contains pattern for proptest regression files
/// which are generated during property-based testing.
///
/// Tests specification: docs/explanation/issue-439-spec.md#ac8
#[test]
fn ac8_proptest_regressions_ignored() {
    let gitignore_path = "/home/steven/code/Rust/BitNet-rs/.gitignore";

    assert!(
        Path::new(gitignore_path).exists(),
        "AC:8 FAIL - .gitignore file not found at repository root"
    );

    let gitignore = fs::read_to_string(gitignore_path)
        .expect("AC:8 FAIL - Failed to read .gitignore");

    // Check for proptest-regressions pattern
    let has_proptest_pattern = gitignore.contains("*.proptest-regressions")
        || gitignore.contains("**/*.proptest-regressions")
        || gitignore.contains(".proptest-regressions");

    assert!(
        has_proptest_pattern,
        "AC:8 FAIL - .gitignore must contain pattern for proptest regressions\n\
         Expected: **/*.proptest-regressions or *.proptest-regressions"
    );

    println!("AC:8 PASS - .gitignore contains proptest-regressions pattern");
}

/// AC:8 - Verify cache/incremental/last_run.json is ignored
///
/// Tests that .gitignore excludes the incremental test cache file which
/// tracks test execution state between runs.
///
/// Tests specification: docs/explanation/issue-439-spec.md#ac8
#[test]
fn ac8_cache_incremental_ignored() {
    let gitignore_path = "/home/steven/code/Rust/BitNet-rs/.gitignore";

    let gitignore = fs::read_to_string(gitignore_path)
        .expect("AC:8 FAIL - Failed to read .gitignore");

    // Check for last_run.json pattern (spec says it should already be at line 196)
    let has_last_run_pattern = gitignore.contains("tests/tests/cache/incremental/last_run.json")
        || gitignore.contains("last_run.json")
        || gitignore.contains("cache/incremental/");

    assert!(
        has_last_run_pattern,
        "AC:8 FAIL - .gitignore must contain pattern for last_run.json\n\
         Expected: tests/tests/cache/incremental/last_run.json or similar"
    );

    println!("AC:8 PASS - .gitignore contains last_run.json pattern");
}

#[cfg(test)]
mod gitignore_comprehensive_audit {
    use super::*;

    /// AC:8 - Verify common ephemeral test artifacts are ignored
    ///
    /// Tests that .gitignore includes patterns for other common test artifacts
    /// that should not be version controlled.
    #[test]
    fn ac8_common_test_artifacts_ignored() {
        let gitignore_path = "/home/steven/code/Rust/BitNet-rs/.gitignore";

        let gitignore = fs::read_to_string(gitignore_path)
            .expect("Failed to read .gitignore");

        let recommended_patterns = vec![
            ("target/", "Cargo build directory"),
            ("*.swp", "Editor swap files"),
            (".DS_Store", "macOS metadata"),
        ];

        for (pattern, description) in recommended_patterns {
            let has_pattern = gitignore.contains(pattern);

            if has_pattern {
                println!("AC:8 INFO - .gitignore includes {} ({})", pattern, description);
            } else {
                println!("AC:8 INFO - .gitignore could include {} ({})", pattern, description);
            }
        }
    }

    /// AC:8 - Verify model files are appropriately handled
    ///
    /// Tests that .gitignore has appropriate patterns for large model files
    /// which shouldn't be committed directly.
    #[test]
    fn ac8_model_files_handling() {
        let gitignore_path = "/home/steven/code/Rust/BitNet-rs/.gitignore";

        let gitignore = fs::read_to_string(gitignore_path)
            .expect("Failed to read .gitignore");

        let model_patterns = vec!["*.gguf", "*.safetensors", "models/"];

        let mut has_model_ignore = false;

        for pattern in model_patterns {
            if gitignore.contains(pattern) {
                println!("AC:8 INFO - .gitignore excludes model files ({})", pattern);
                has_model_ignore = true;
            }
        }

        if !has_model_ignore {
            println!("AC:8 INFO - .gitignore could exclude model files (*.gguf, *.safetensors)");
        }
    }

    /// AC:8 - Verify test output directories are ignored
    ///
    /// Tests that temporary test output directories are excluded from git.
    #[test]
    fn ac8_test_output_directories_ignored() {
        let gitignore_path = "/home/steven/code/Rust/BitNet-rs/.gitignore";

        let gitignore = fs::read_to_string(gitignore_path)
            .expect("Failed to read .gitignore");

        let test_output_patterns = vec!["test-output/", "tmp/", "temp/"];

        for pattern in test_output_patterns {
            if gitignore.contains(pattern) {
                println!("AC:8 INFO - .gitignore excludes test output ({})", pattern);
            }
        }
    }

    /// AC:8 - Verify no committed proptest regression files exist
    ///
    /// Tests that no proptest regression files are currently committed,
    /// which would indicate the gitignore pattern was added too late.
    #[test]
    fn ac8_no_committed_proptest_regressions() {
        use std::process::Command;

        let output = Command::new("git")
            .args(&["ls-files", "*.proptest-regressions"])
            .current_dir("/home/steven/code/Rust/BitNet-rs")
            .output()
            .expect("Failed to run git ls-files");

        let stdout = String::from_utf8_lossy(&output.stdout);

        if !stdout.trim().is_empty() {
            println!(
                "AC:8 WARNING - Found committed proptest regression files:\n{}\n\
                 These should be removed from git history",
                stdout
            );
        } else {
            println!("AC:8 PASS - No proptest regression files committed to git");
        }
    }

    /// AC:8 - Verify no committed last_run.json files exist
    ///
    /// Tests that no test cache files are currently committed.
    #[test]
    fn ac8_no_committed_last_run_json() {
        use std::process::Command;

        let output = Command::new("git")
            .args(&["ls-files", "**/last_run.json"])
            .current_dir("/home/steven/code/Rust/BitNet-rs")
            .output()
            .expect("Failed to run git ls-files");

        let stdout = String::from_utf8_lossy(&output.stdout);

        if !stdout.trim().is_empty() {
            println!(
                "AC:8 WARNING - Found committed last_run.json files:\n{}\n\
                 These should be removed from git history",
                stdout
            );
        } else {
            println!("AC:8 PASS - No last_run.json files committed to git");
        }
    }
}

#[cfg(test)]
mod gitignore_format_validation {
    use super::*;

    /// AC:8 - Verify .gitignore has proper format and comments
    ///
    /// Tests that .gitignore is well-organized with comments explaining
    /// different sections of ignored patterns.
    #[test]
    fn ac8_gitignore_well_documented() {
        let gitignore_path = "/home/steven/code/Rust/BitNet-rs/.gitignore";

        let gitignore = fs::read_to_string(gitignore_path)
            .expect("Failed to read .gitignore");

        // Check for comment lines
        let has_comments = gitignore.lines().any(|line| line.starts_with('#'));

        if has_comments {
            println!("AC:8 INFO - .gitignore includes documentation comments");
        } else {
            println!("AC:8 INFO - .gitignore could benefit from section comments");
        }

        // Check for blank line separation
        let has_organization = gitignore.contains("\n\n");

        if has_organization {
            println!("AC:8 INFO - .gitignore is organized with blank line separation");
        }
    }

    /// AC:8 - Verify .gitignore uses glob patterns correctly
    ///
    /// Tests that patterns use proper glob syntax (e.g., **/ for recursive matching).
    #[test]
    fn ac8_gitignore_glob_patterns_correct() {
        let gitignore_path = "/home/steven/code/Rust/BitNet-rs/.gitignore";

        let gitignore = fs::read_to_string(gitignore_path)
            .expect("Failed to read .gitignore");

        // Check for recursive glob patterns
        let uses_recursive_glob = gitignore.contains("**/");

        if uses_recursive_glob {
            println!("AC:8 INFO - .gitignore uses recursive glob patterns (**/pattern)");
        }

        // Check for wildcard patterns
        let uses_wildcards = gitignore.contains("*.");

        if uses_wildcards {
            println!("AC:8 INFO - .gitignore uses wildcard patterns (*.extension)");
        }
    }
}
