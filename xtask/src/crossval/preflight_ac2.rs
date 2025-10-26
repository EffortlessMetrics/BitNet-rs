//! AC2 implementation: setup-cpp-auto subprocess invocation
//!
//! This module implements the setup-cpp-auto invocation logic for auto-repair.

use crate::crossval::CppBackend;
use super::RepairError;
use std::{env, process::Command};

/// Invoke setup-cpp-auto subprocess to install C++ reference implementation
///
/// Implements AC2: setup-cpp-auto subprocess invocation with:
/// - Command construction with backend parameter
/// - stdout/stderr capture
/// - Shell export parsing from stdout
/// - Environment variable application to current process
/// - Error classification from stderr on failure
///
/// # Arguments
///
/// * `backend` - The C++ backend to install (bitnet or llama)
///
/// # Returns
///
/// * `Ok(())` - Setup succeeded, environment variables applied
/// * `Err(RepairError)` - Setup failed with classified error
///
/// # Examples
///
/// ```ignore
/// use xtask::crossval::CppBackend;
///
/// // Install BitNet.cpp and update environment
/// invoke_setup_cpp_auto(CppBackend::BitNet)?;
/// ```
pub fn invoke_setup_cpp_auto(backend: CppBackend) -> Result<(), RepairError> {
    // Build command: cargo run -p xtask -- setup-cpp-auto --backend {backend}
    let backend_arg = match backend {
        CppBackend::BitNet => "bitnet",
        CppBackend::Llama => "llama",
    };

    let output = Command::new("cargo")
        .args(["run", "-p", "xtask", "--", "setup-cpp-auto", "--backend", backend_arg, "--emit=sh"])
        .env("BITNET_REPAIR_IN_PROGRESS", "1") // Pass recursion guard to child
        .output()
        .map_err(|e| RepairError::SetupFailed(format!("Failed to execute setup-cpp-auto: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(RepairError::classify(&stderr, backend.name()));
    }

    // Parse shell exports from stdout and apply to current process
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_and_apply_shell_exports(&stdout)?;

    Ok(())
}

/// Parse shell exports from stdout and apply to current process environment
///
/// Parses output from setup-cpp-auto --emit=sh and applies environment variable
/// exports to the current process.
///
/// Expected format:
/// ```sh
/// export BITNET_CPP_DIR=/path/to/bitnet.cpp
/// export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH
/// ```
///
/// # Arguments
///
/// * `stdout` - Shell script output from setup-cpp-auto
///
/// # Returns
///
/// * `Ok(())` - Environment variables applied successfully
/// * `Err(RepairError)` - Failed to parse or apply exports
fn parse_and_apply_shell_exports(stdout: &str) -> Result<(), RepairError> {
    for line in stdout.lines() {
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Match: export VAR=value
        if let Some(export_line) = trimmed.strip_prefix("export ") {
            if let Some((var_name, value)) = export_line.split_once('=') {
                let var_name = var_name.trim();
                let mut value = value.trim();

                // Handle quoted values: export VAR="value" or export VAR='value'
                if (value.starts_with('"') && value.ends_with('"'))
                    || (value.starts_with('\'') && value.ends_with('\''))
                {
                    value = &value[1..value.len() - 1];
                }

                // Expand $VAR references in value
                let expanded_value = expand_env_vars(value);

                // Apply to current process environment
                // SAFETY: We're setting environment variables as part of the auto-repair flow.
                // These are controlled by setup-cpp-auto output and expected to be safe.
                unsafe {
                    env::set_var(var_name, expanded_value);
                }
            }
        }
    }

    Ok(())
}

/// Expand environment variable references in a string
///
/// Supports both `$VAR` and `${VAR}` syntax.
///
/// # Arguments
///
/// * `value` - String potentially containing env var references
///
/// # Returns
///
/// String with environment variables expanded
fn expand_env_vars(value: &str) -> String {
    let mut result = value.to_string();

    // Expand ${VAR} first (more specific)
    while let Some(start) = result.find("${") {
        if let Some(end) = result[start..].find('}') {
            let var_name = &result[start + 2..start + end];
            let var_value = env::var(var_name).unwrap_or_default();
            result.replace_range(start..start + end + 1, &var_value);
        } else {
            break; // Malformed ${VAR, skip
        }
    }

    // Expand $VAR (simple form)
    let mut expanded = String::new();
    let mut chars = result.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '$' {
            // Collect variable name (alphanumeric + underscore)
            let mut var_name = String::new();
            while let Some(&next_ch) = chars.peek() {
                if next_ch.is_alphanumeric() || next_ch == '_' {
                    var_name.push(next_ch);
                    chars.next();
                } else {
                    break;
                }
            }

            if !var_name.is_empty() {
                // Expand variable
                expanded.push_str(&env::var(&var_name).unwrap_or_default());
            } else {
                // Literal $
                expanded.push('$');
            }
        } else {
            expanded.push(ch);
        }
    }

    expanded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_env_vars_basic() {
        env::set_var("TEST_VAR", "test_value");
        assert_eq!(expand_env_vars("prefix $TEST_VAR suffix"), "prefix test_value suffix");
        env::remove_var("TEST_VAR");
    }

    #[test]
    fn test_expand_env_vars_braces() {
        env::set_var("TEST_VAR", "test_value");
        assert_eq!(expand_env_vars("prefix ${TEST_VAR} suffix"), "prefix test_value suffix");
        env::remove_var("TEST_VAR");
    }

    #[test]
    fn test_expand_env_vars_missing() {
        assert_eq!(expand_env_vars("prefix $NONEXISTENT suffix"), "prefix  suffix");
    }

    #[test]
    fn test_parse_shell_exports_basic() {
        let script = r#"
export VAR1=value1
export VAR2="value2"
export VAR3='value3'
"#;
        parse_and_apply_shell_exports(script).unwrap();
        assert_eq!(env::var("VAR1").unwrap(), "value1");
        assert_eq!(env::var("VAR2").unwrap(), "value2");
        assert_eq!(env::var("VAR3").unwrap(), "value3");
        env::remove_var("VAR1");
        env::remove_var("VAR2");
        env::remove_var("VAR3");
    }

    #[test]
    fn test_parse_shell_exports_with_expansion() {
        env::set_var("EXISTING", "/existing/path");
        let script = r#"
export PATH=/new/path:$EXISTING
"#;
        parse_and_apply_shell_exports(script).unwrap();
        assert_eq!(env::var("PATH").unwrap(), "/new/path:/existing/path");
        env::remove_var("EXISTING");
    }
}
