//! Unit tests for shell export parsing (env-export-before-rebuild-deterministic.md)
//!
//! **Specification**: docs/specs/env-export-before-rebuild-deterministic.md (Version 1.0.0)
//!
//! This test suite validates parsing of shell export formats from `setup-cpp-auto` output,
//! ensuring environment variables are correctly extracted for the auto-repair rebuild flow.
//!
//! **Acceptance Criteria Coverage (12 tests)**:
//! - AC1: Parse Shell Exports (12 tests)
//!   - POSIX sh format: `export VAR="value"`
//!   - Fish shell format: `set -gx VAR "value"`
//!   - PowerShell format: `$env:VAR = "value"`
//!   - Edge cases: multi-line scripts, quoted values, variable references
//!
//! **Test Strategy**:
//! - Feature-gated with `#[cfg(feature = "crossval-all")]`
//! - Pure function testing (no environment mutation, no serial needed)
//! - TDD scaffolding: Tests compile but fail with `todo!()` until implementation
//! - Spec references: `/// Tests spec: env-export-before-rebuild-deterministic.md#AC1`
//!
//! **Traceability**: Each test validates specific shell export format parsing from the spec.

#![cfg(feature = "crossval-all")]

#[cfg(test)]
mod parse_sh_exports_tests {
    use xtask::crossval::preflight::parse_sh_exports;

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC1
    /// AC:AC1 - Parse POSIX sh basic unquoted export
    ///
    /// **Given**: Shell output with simple unquoted export
    /// **When**: Calling `parse_sh_exports(output)`
    /// **Then**: Returns HashMap with correct key-value pair
    ///
    /// **Example input**:
    /// ```text
    /// export BITNET_CPP_DIR=/path/to/libs
    /// ```
    ///
    /// **Expected output**:
    /// ```rust
    /// HashMap { "BITNET_CPP_DIR" => "/path/to/libs" }
    /// ```
    #[test]
    fn test_parse_sh_exports_basic_unquoted() {
        let input = "export BITNET_CPP_DIR=/path/to/libs";
        let exports = parse_sh_exports(input);

        assert_eq!(exports.len(), 1);
        assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/path/to/libs".to_string()));
    }

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC1
    /// AC:AC1 - Parse POSIX sh double-quoted export
    ///
    /// **Given**: Shell output with double-quoted values
    /// **When**: Calling `parse_sh_exports(output)`
    /// **Then**: Returns HashMap with quotes stripped
    ///
    /// **Example input**:
    /// ```text
    /// export BITNET_CPP_DIR="/path/to/libs"
    /// ```
    ///
    /// **Expected output**:
    /// ```rust
    /// HashMap { "BITNET_CPP_DIR" => "/path/to/libs" }
    /// ```
    #[test]
    fn test_parse_sh_exports_double_quoted() {
        let input = r#"export BITNET_CPP_DIR="/home/user/.cache/bitnet_cpp""#;
        let exports = parse_sh_exports(input);

        assert_eq!(exports.len(), 1);
        assert_eq!(
            exports.get("BITNET_CPP_DIR"),
            Some(&"/home/user/.cache/bitnet_cpp".to_string())
        );
    }

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC1
    /// AC:AC1 - Parse multi-line shell script with multiple exports
    ///
    /// **Given**: Multi-line shell script with mixed content
    /// **When**: Calling `parse_sh_exports(output)`
    /// **Then**: Returns HashMap with all exports, skipping non-export lines
    ///
    /// **Example input**:
    /// ```text
    /// export BITNET_CPP_DIR="/path1"
    /// export LD_LIBRARY_PATH="/path2:/path3"
    /// export BITNET_CROSSVAL_LIBDIR="/path4"
    /// echo "[bitnet] C++ ready"
    /// ```
    ///
    /// **Expected output**:
    /// ```rust
    /// HashMap {
    ///     "BITNET_CPP_DIR" => "/path1",
    ///     "LD_LIBRARY_PATH" => "/path2:/path3",
    ///     "BITNET_CROSSVAL_LIBDIR" => "/path4"
    /// }
    /// ```
    #[test]
    fn test_parse_sh_exports_multi_line_script() {
        let input = r#"export BITNET_CPP_DIR="/path1"
export LD_LIBRARY_PATH="/path2:/path3"
export BITNET_CROSSVAL_LIBDIR="/path4"
echo "[bitnet] C++ ready""#;
        let exports = parse_sh_exports(input);

        assert_eq!(exports.len(), 3);
        assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/path1".to_string()));
        assert_eq!(exports.get("LD_LIBRARY_PATH"), Some(&"/path2:/path3".to_string()));
        assert_eq!(exports.get("BITNET_CROSSVAL_LIBDIR"), Some(&"/path4".to_string()));
    }

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC1
    /// AC:AC1 - Parse empty script (edge case)
    ///
    /// **Given**: Empty string or whitespace-only input
    /// **When**: Calling `parse_sh_exports(output)`
    /// **Then**: Returns empty HashMap (no errors)
    ///
    /// **Example inputs**:
    /// - `""`
    /// - `"   "`
    /// - `"\n\n\n"`
    ///
    /// **Expected output**:
    /// ```rust
    /// HashMap::new()  // Empty map
    /// ```
    #[test]
    fn test_parse_sh_exports_empty_script() {
        let exports = parse_sh_exports("");
        assert!(exports.is_empty());

        let exports2 = parse_sh_exports("   ");
        assert!(exports2.is_empty());

        let exports3 = parse_sh_exports("\n\n\n");
        assert!(exports3.is_empty());
    }

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC1
    /// AC:AC1 - Parse script with no exports (only comments and echo)
    ///
    /// **Given**: Script with non-export lines only
    /// **When**: Calling `parse_sh_exports(output)`
    /// **Then**: Returns empty HashMap (no errors)
    ///
    /// **Example input**:
    /// ```text
    /// # This is a comment
    /// echo "[bitnet] Setup complete"
    /// echo "No exports here"
    /// ```
    ///
    /// **Expected output**:
    /// ```rust
    /// HashMap::new()  // Empty map
    /// ```
    #[test]
    fn test_parse_sh_exports_no_exports() {
        let input = r#"# This is a comment
echo "[bitnet] Setup complete"
echo "No exports here""#;
        let exports = parse_sh_exports(input);
        assert!(exports.is_empty());
    }

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC1
    /// AC:AC1 - Skip malformed export lines
    ///
    /// **Given**: Script with malformed export syntax
    /// **When**: Calling `parse_sh_exports(output)`
    /// **Then**: Returns HashMap with valid exports only (skips malformed)
    ///
    /// **Example input**:
    /// ```text
    /// export VALID="/path"
    /// export MISSING_VALUE
    /// export =value_without_key
    /// export ANOTHER_VALID="/path2"
    /// ```
    ///
    /// **Expected output**:
    /// ```rust
    /// HashMap {
    ///     "VALID" => "/path",
    ///     "ANOTHER_VALID" => "/path2"
    /// }
    /// ```
    #[test]
    fn test_parse_sh_exports_malformed_lines() {
        let input = r#"export VALID="/path"
export MISSING_VALUE
export =value_without_key
export ANOTHER_VALID="/path2""#;
        let exports = parse_sh_exports(input);

        assert_eq!(exports.len(), 2);
        assert_eq!(exports.get("VALID"), Some(&"/path".to_string()));
        assert_eq!(exports.get("ANOTHER_VALID"), Some(&"/path2".to_string()));
    }

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC1
    /// AC:AC1 - Parse fish shell format
    ///
    /// **Given**: Fish shell export format
    /// **When**: Calling `parse_sh_exports(output)`
    /// **Then**: Returns HashMap with parsed fish exports
    ///
    /// **Example input**:
    /// ```fish
    /// set -gx BITNET_CPP_DIR "/path/to/libs"
    /// set -gx LD_LIBRARY_PATH "/lib:/usr/lib"
    /// ```
    ///
    /// **Expected output**:
    /// ```rust
    /// HashMap {
    ///     "BITNET_CPP_DIR" => "/path/to/libs",
    ///     "LD_LIBRARY_PATH" => "/lib:/usr/lib"
    /// }
    /// ```
    #[test]
    fn test_parse_sh_exports_fish_format() {
        let input = r#"set -gx BITNET_CPP_DIR "/path/to/libs"
set -gx LD_LIBRARY_PATH "/lib:/usr/lib""#;
        let exports = parse_sh_exports(input);

        assert_eq!(exports.len(), 2);
        assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/path/to/libs".to_string()));
        assert_eq!(exports.get("LD_LIBRARY_PATH"), Some(&"/lib:/usr/lib".to_string()));
    }

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC1
    /// AC:AC1 - Parse PowerShell format
    ///
    /// **Given**: PowerShell export format
    /// **When**: Calling `parse_sh_exports(output)`
    /// **Then**: Returns HashMap with parsed PowerShell exports
    ///
    /// **Example input**:
    /// ```powershell
    /// $env:BITNET_CPP_DIR = "/path/to/libs"
    /// $env:LD_LIBRARY_PATH = "/lib:/usr/lib"
    /// ```
    ///
    /// **Expected output**:
    /// ```rust
    /// HashMap {
    ///     "BITNET_CPP_DIR" => "/path/to/libs",
    ///     "LD_LIBRARY_PATH" => "/lib:/usr/lib"
    /// }
    /// ```
    #[test]
    fn test_parse_sh_exports_pwsh_format() {
        let input = r#"$env:BITNET_CPP_DIR = "/path/to/libs"
$env:LD_LIBRARY_PATH = "/lib:/usr/lib""#;
        let exports = parse_sh_exports(input);

        assert_eq!(exports.len(), 2);
        assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/path/to/libs".to_string()));
        assert_eq!(exports.get("LD_LIBRARY_PATH"), Some(&"/lib:/usr/lib".to_string()));
    }

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC1
    /// AC:AC1 - Parse nested quotes edge case
    ///
    /// **Given**: Export with nested/escaped quotes in value
    /// **When**: Calling `parse_sh_exports(output)`
    /// **Then**: Returns HashMap preserving inner quotes
    ///
    /// **Example input**:
    /// ```text
    /// export PROJECT_NAME="BitNet \"C++\" Backend"
    /// ```
    ///
    /// **Expected output**:
    /// ```rust
    /// HashMap { "PROJECT_NAME" => "BitNet \"C++\" Backend" }
    /// ```
    #[test]
    fn test_parse_sh_exports_nested_quotes() {
        let input = r#"export PROJECT_NAME="BitNet \"C++\" Backend""#;
        let exports = parse_sh_exports(input);

        assert_eq!(exports.len(), 1);
        assert_eq!(exports.get("PROJECT_NAME"), Some(&r#"BitNet \"C++\" Backend"#.to_string()));
    }

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC1
    /// AC:AC1 - Parse paths with spaces
    ///
    /// **Given**: Export with space-containing paths
    /// **When**: Calling `parse_sh_exports(output)`
    /// **Then**: Returns HashMap with spaces preserved
    ///
    /// **Example input**:
    /// ```text
    /// export BITNET_CPP_DIR="/home/user/My Documents/bitnet cpp"
    /// ```
    ///
    /// **Expected output**:
    /// ```rust
    /// HashMap { "BITNET_CPP_DIR" => "/home/user/My Documents/bitnet cpp" }
    /// ```
    #[test]
    fn test_parse_sh_exports_paths_with_spaces() {
        let input = r#"export BITNET_CPP_DIR="/home/user/My Documents/bitnet cpp""#;
        let exports = parse_sh_exports(input);

        assert_eq!(exports.len(), 1);
        assert_eq!(
            exports.get("BITNET_CPP_DIR"),
            Some(&"/home/user/My Documents/bitnet cpp".to_string())
        );
    }

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC1
    /// AC:AC1 - Parse LD_LIBRARY_PATH with variable references
    ///
    /// **Given**: Export with shell variable references (${VAR})
    /// **When**: Calling `parse_sh_exports(output)`
    /// **Then**: Returns HashMap with literal variable reference (no expansion)
    ///
    /// **Example input**:
    /// ```text
    /// export LD_LIBRARY_PATH="/new/path:${LD_LIBRARY_PATH:-}"
    /// ```
    ///
    /// **Expected output**:
    /// ```rust
    /// HashMap { "LD_LIBRARY_PATH" => "/new/path:${LD_LIBRARY_PATH:-}" }
    /// ```
    #[test]
    fn test_parse_sh_exports_ld_library_path() {
        let input = r#"export LD_LIBRARY_PATH="/new/path:${LD_LIBRARY_PATH:-}""#;
        let exports = parse_sh_exports(input);

        assert_eq!(exports.len(), 1);
        assert_eq!(
            exports.get("LD_LIBRARY_PATH"),
            Some(&"/new/path:${LD_LIBRARY_PATH:-}".to_string())
        );
    }

    /// Tests spec: env-export-before-rebuild-deterministic.md#AC1
    /// AC:AC1 - Parse Windows PATH with semicolon separator
    ///
    /// **Given**: Windows PATH export with semicolon separators
    /// **When**: Calling `parse_sh_exports(output)`
    /// **Then**: Returns HashMap with semicolons preserved
    ///
    /// **Example input**:
    /// ```text
    /// export PATH="C:\bitnet\bin;C:\tools;%PATH%"
    /// ```
    ///
    /// **Expected output**:
    /// ```rust
    /// HashMap { "PATH" => "C:\\bitnet\\bin;C:\\tools;%PATH%" }
    /// ```
    #[test]
    fn test_parse_sh_exports_windows_path() {
        let input = r#"export PATH="C:\bitnet\bin;C:\tools;%PATH%""#;
        let exports = parse_sh_exports(input);

        assert_eq!(exports.len(), 1);
        assert_eq!(exports.get("PATH"), Some(&r"C:\bitnet\bin;C:\tools;%PATH%".to_string()));
    }
}
