# Environment Gap: Code Snippets and Implementation Guide

## Current Broken Implementation

### setup-cpp-auto Output (CORRECT)

**File**: `xtask/src/cpp_setup_auto.rs:792-866`

```rust
fn emit_exports(emit: Emit, repo: &Path, lib_dir: &Path, crossval_libdir: Option<&str>) {
    match emit {
        Emit::Sh => {
            // POSIX sh/bash/zsh
            println!(r#"export BITNET_CPP_DIR="{}""#, repo.display());
            if let Some(libdir) = crossval_libdir {
                println!(r#"export BITNET_CROSSVAL_LIBDIR="{}""#, libdir);
            }
            #[cfg(target_os = "linux")]
            println!(r#"export LD_LIBRARY_PATH="{}:${{LD_LIBRARY_PATH:-}}""#, lib_dir.display());
            println!(r#"echo "[bitnet] C++ ready at $BITNET_CPP_DIR""#);
        }
        Emit::Fish => {
            println!(r#"set -gx BITNET_CPP_DIR "{}""#, repo.display());
            if let Some(libdir) = crossval_libdir {
                println!(r#"set -gx BITNET_CROSSVAL_LIBDIR "{}""#, libdir);
            }
            #[cfg(target_os = "linux")]
            println!(r#"set -gx LD_LIBRARY_PATH "{}" $LD_LIBRARY_PATH"#, lib_dir.display());
        }
        Emit::Pwsh => {
            println!(r#"$env:BITNET_CPP_DIR = "{}""#, repo.display());
            if let Some(libdir) = crossval_libdir {
                println!(r#"$env:BITNET_CROSSVAL_LIBDIR = "{}""#, libdir);
            }
            println!(r#"$env:PATH = "{};" + $env:PATH"#, lib_dir.display());
        }
        // ... cmd format ...
    }
}
```

**Output Example**:
```bash
export BITNET_CPP_DIR="/home/user/.cache/bitnet_cpp"
export BITNET_CROSSVAL_LIBDIR="/home/user/.cache/bitnet_cpp/build/3rdparty/llama.cpp/build/bin"
export LD_LIBRARY_PATH="/home/user/.cache/bitnet_cpp/build/bin:${LD_LIBRARY_PATH:-}"
echo "[bitnet] C++ ready at $BITNET_CPP_DIR"
```

### attempt_repair_once() - Gap 1 (Stdout Discarded)

**File**: `xtask/src/crossval/preflight.rs:1951-1992`

```rust
fn attempt_repair_once(backend: CppBackend, verbose: bool) -> Result<(), RepairError> {
    let progress = RepairProgress::new(verbose);

    if env::var("BITNET_REPAIR_IN_PROGRESS").is_ok() {
        return Err(RepairError::RecursionDetected);
    }

    unsafe {
        env::set_var("BITNET_REPAIR_IN_PROGRESS", "1");
    }

    progress.log("DETECT", &format!("Backend '{}' not found", backend.name()));
    progress.log("REPAIR", "Invoking setup-cpp-auto...");

    // ┌─ SETUP RUNS HERE
    // └─ CAPTURES STDOUT CONTAINING EXPORTS
    let setup_result = Command::new(
        env::current_exe()
            .map_err(|e| RepairError::SetupFailed(format!("Failed to get current exe: {}", e)))?,
    )
    .args(["setup-cpp-auto", "--emit=sh"])
    .output()
    .map_err(|e| RepairError::SetupFailed(format!("Failed to execute setup-cpp-auto: {}", e)))?;

    unsafe {
        env::remove_var("BITNET_REPAIR_IN_PROGRESS");
    }

    if !setup_result.status.success() {
        let stderr = String::from_utf8_lossy(&setup_result.stderr);
        let backend_name = backend.name();

        return Err(RepairError::classify(&stderr, backend_name));
    }

    // ❌ GAP 1: setup_result.stdout NEVER READ
    // The output containing "export BITNET_CPP_DIR=..." is discarded here
    
    progress.log("REPAIR", "C++ libraries installed successfully");
    progress.log("REBUILD", "Next: Rebuild xtask to detect libraries");

    Ok(())
}
```

### rebuild_xtask() - Gap 2 (Env Not Applied)

**File**: `xtask/src/crossval/preflight.rs:1617-1639`

```rust
fn rebuild_xtask(verbose: bool) -> Result<(), RebuildError> {
    if verbose {
        eprintln!("[preflight] Rebuilding xtask...");
    }

    // ❌ GAP 2: No .env() calls on this Command
    // Cargo inherits parent's environment (which doesn't have BITNET_CPP_DIR)
    let build_status = Command::new("cargo")
        .args(["build", "-p", "xtask", "--features", "crossval-all"])
        .status()  // ← Child inherits stale parent env
        .map_err(|e: std::io::Error| RebuildError::BuildFailed(e.to_string()))?;

    if !build_status.success() {
        return Err(RebuildError::BuildFailed(format!(
            "cargo build exited with code {:?}",
            build_status.code()
        )));
    }

    if verbose {
        eprintln!("[preflight] ✓ Rebuild complete");
    }

    Ok(())
}
```

### preflight_with_auto_repair() - Gap Location

**File**: `xtask/src/crossval/preflight.rs:1393-1407`

```rust
    // Step 3: Attempt repair with retry logic
    if verbose {
        eprintln!();
        eprintln!("Backend '{}' not found at build time", backend.name());
        eprintln!();
        eprintln!("Auto-repairing... (this will take 5-10 minutes on first run)");
    }

    if let Err(e) = attempt_repair_with_retry(backend, verbose) {
        // Repair failed - show error with recovery steps
        eprintln!("\n{}", e);
        bail!("Auto-repair failed for backend '{}'", backend.name());
    }

    // ❌ CRITICAL GAP: setup-cpp-auto output is lost, env vars not applied
    // ├─ Gap 1: stdout from setup-cpp-auto was never captured
    // ├─ Gap 2: even if captured, was never parsed
    // └─ Gap 3: even if parsed, was never applied to environment

    // Step 4: Rebuild xtask to pick up new detection (AC3)
    if verbose {
        eprintln!("[repair] Step 2/3: Rebuilding xtask binary...");
    }

    if let Err(e) = rebuild_xtask(verbose) {
        // ← Child cargo inherits STALE environment
        // ← BITNET_CPP_DIR not visible to build.rs detection
        eprintln!("\n{}", e);
        bail!("xtask rebuild failed after successful C++ setup");
    }
```

## Fixed Implementation

### New: parse_sh_exports() Function

**File**: `xtask/src/crossval/preflight.rs` (new function)

```rust
use std::collections::HashMap;

/// Parse shell export format from setup-cpp-auto output
///
/// Handles multiple shell formats:
/// - POSIX sh/bash/zsh: export VAR="value"
/// - fish shell: set -gx VAR "value"
/// - PowerShell: $env:VAR = "value"
/// - cmd.exe: set VAR=value
///
/// Returns HashMap of key=value pairs extracted from output
fn parse_sh_exports(output: &str) -> Result<HashMap<String, String>, String> {
    let mut exports = HashMap::new();

    for line in output.lines() {
        let trimmed = line.trim();

        // Skip comments and empty lines
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Parse POSIX sh/bash/zsh format: export VAR="value"
        if let Some(rest) = trimmed.strip_prefix("export ") {
            if let Some((key, value)) = parse_key_value(rest, '"') {
                exports.insert(key.to_string(), value.to_string());
                continue;
            }
        }

        // Parse fish shell format: set -gx VAR "value"
        if trimmed.starts_with("set ") && trimmed.contains("-gx ") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 4 && parts[1] == "-gx" {
                let key = parts[2].to_string();
                // Extract everything after the key, handling quoted values
                let value_part = trimmed
                    .find(key.as_str())
                    .and_then(|pos| trimmed[pos + key.len()..].trim().strip_prefix('"'))
                    .and_then(|s| s.rfind('"').map(|p| &s[..p]))
                    .unwrap_or("")
                    .to_string();
                if !value_part.is_empty() {
                    exports.insert(key, value_part);
                    continue;
                }
            }
        }

        // Parse PowerShell format: $env:VAR = "value"
        if trimmed.starts_with("$env:") {
            if let Some(rest) = trimmed.strip_prefix("$env:") {
                if let Some((key, value)) = parse_key_value(rest, '"') {
                    exports.insert(key.to_string(), value.to_string());
                    continue;
                }
            }
        }

        // Parse cmd.exe format: set VAR=value
        if let Some(rest) = trimmed.strip_prefix("set ") {
            if let Some((key, value)) = parse_key_value(rest, '\0') {
                exports.insert(key.to_string(), value.to_string());
                continue;
            }
        }
    }

    Ok(exports)
}

/// Helper: Parse key=value from a string
/// Returns (key, value) if found
fn parse_key_value(s: &str, quote: char) -> Option<(&str, &str)> {
    let parts: Vec<&str> = s.split('=').collect();
    if parts.len() != 2 {
        return None;
    }

    let key = parts[0].trim();
    let value_str = parts[1].trim();

    let value = if quote != '\0' && value_str.starts_with(quote) && value_str.ends_with(quote) {
        &value_str[1..value_str.len() - 1]
    } else {
        value_str
    };

    Some((key, value))
}

#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn test_parse_sh_export() {
        let output = r#"export BITNET_CPP_DIR="/home/user/.cache/bitnet_cpp""#;
        let exports = parse_sh_exports(output).unwrap();
        assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/home/user/.cache/bitnet_cpp".to_string()));
    }

    #[test]
    fn test_parse_fish_set() {
        let output = r#"set -gx BITNET_CPP_DIR "/home/user/.cache/bitnet_cpp""#;
        let exports = parse_sh_exports(output).unwrap();
        assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/home/user/.cache/bitnet_cpp".to_string()));
    }

    #[test]
    fn test_parse_pwsh_env() {
        let output = r#"$env:BITNET_CPP_DIR = "/home/user/.cache/bitnet_cpp""#;
        let exports = parse_sh_exports(output).unwrap();
        assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/home/user/.cache/bitnet_cpp".to_string()));
    }

    #[test]
    fn test_parse_multiple_exports() {
        let output = r#"export BITNET_CPP_DIR="/path1"
export LD_LIBRARY_PATH="/path2:/path3"
export BITNET_CROSSVAL_LIBDIR="/path4""#;
        let exports = parse_sh_exports(output).unwrap();
        assert_eq!(exports.get("BITNET_CPP_DIR"), Some(&"/path1".to_string()));
        assert_eq!(exports.get("LD_LIBRARY_PATH"), Some(&"/path2:/path3".to_string()));
        assert_eq!(exports.get("BITNET_CROSSVAL_LIBDIR"), Some(&"/path4".to_string()));
    }
}
```

### New: rebuild_xtask_with_env() Function

**File**: `xtask/src/crossval/preflight.rs` (new function)

```rust
/// Rebuild xtask with environment variables
///
/// Applies parsed environment variables from setup-cpp-auto to the cargo
/// build process, ensuring build.rs detection can find newly-installed libraries.
///
/// # Arguments
///
/// * `verbose` - If true, print progress messages
/// * `exports` - Environment variables to apply to child cargo process
///
/// # Returns
///
/// * `Ok(())` - Rebuild succeeded
/// * `Err(RebuildError)` - Rebuild failed
fn rebuild_xtask_with_env(
    verbose: bool,
    exports: &HashMap<String, String>,
) -> Result<(), RebuildError> {
    if verbose {
        eprintln!("[preflight] Rebuilding xtask with environment variables...");
    }

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "-p", "xtask", "--features", "crossval-all"]);

    // ✅ APPLY ENVIRONMENT VARIABLES TO CHILD PROCESS
    for (key, value) in exports {
        if verbose {
            eprintln!("[preflight]   {} = {}", key, value);
        }
        cmd.env(key, value);
    }

    let build_status = cmd.status().map_err(|e: std::io::Error| {
        RebuildError::BuildFailed(e.to_string())
    })?;

    if !build_status.success() {
        return Err(RebuildError::BuildFailed(format!(
            "cargo build exited with code {:?}",
            build_status.code()
        )));
    }

    if verbose {
        eprintln!("[preflight] ✓ Rebuild complete with environment variables");
    }

    Ok(())
}
```

### Modified: attempt_repair_with_retry() - Return Stdout

**File**: `xtask/src/crossval/preflight.rs` (MODIFIED)

**Change**: Return type changes to include stdout

```rust
/// Attempt to repair a missing backend with retry logic
///
/// Returns the stdout from setup-cpp-auto containing environment exports
pub fn attempt_repair_with_retry(
    backend: CppBackend,
    verbose: bool,
) -> Result<String, RepairError> {  // ← Changed to return String
    let config = RetryConfig::default();
    let mut retries = 0;

    loop {
        match attempt_repair_once(backend, verbose) {
            Ok(setup_output) => return Ok(setup_output),  // ← Return stdout
            Err(e) if is_retryable_error(&e) && retries < config.max_retries => {
                retries += 1;
                let backoff_ms = config.initial_backoff_ms * 2_u64.pow(retries - 1);
                let backoff_ms = backoff_ms.min(config.max_backoff_ms);

                eprintln!(
                    "[repair] Network error, retry {}/{} after {}ms...",
                    retries, config.max_retries, backoff_ms
                );

                std::thread::sleep(Duration::from_millis(backoff_ms));
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}
```

### Modified: attempt_repair_once() - Capture & Return Stdout

**File**: `xtask/src/crossval/preflight.rs` (MODIFIED)

```rust
fn attempt_repair_once(backend: CppBackend, verbose: bool) -> Result<String, RepairError> {  // ← Return String
    let progress = RepairProgress::new(verbose);

    if env::var("BITNET_REPAIR_IN_PROGRESS").is_ok() {
        return Err(RepairError::RecursionDetected);
    }

    unsafe {
        env::set_var("BITNET_REPAIR_IN_PROGRESS", "1");
    }

    progress.log("DETECT", &format!("Backend '{}' not found", backend.name()));
    progress.log("REPAIR", "Invoking setup-cpp-auto... (this will take 5-10 minutes on first run)");

    let setup_result = Command::new(
        env::current_exe()
            .map_err(|e| RepairError::SetupFailed(format!("Failed to get current exe: {}", e)))?,
    )
    .args(["setup-cpp-auto", "--emit=sh"])
    .output()
    .map_err(|e| RepairError::SetupFailed(format!("Failed to execute setup-cpp-auto: {}", e)))?;

    unsafe {
        env::remove_var("BITNET_REPAIR_IN_PROGRESS");
    }

    if !setup_result.status.success() {
        let stderr = String::from_utf8_lossy(&setup_result.stderr);
        let backend_name = backend.name();
        return Err(RepairError::classify(&stderr, backend_name));
    }

    // ✅ CAPTURE AND RETURN STDOUT
    let stdout = String::from_utf8_lossy(&setup_result.stdout).to_string();
    
    progress.log("REPAIR", "C++ libraries installed successfully");
    progress.log("REBUILD", "Next: Rebuild xtask to detect libraries");

    Ok(stdout)  // ← Return the shell exports
}
```

### Modified: preflight_with_auto_repair() - Integrate Fix

**File**: `xtask/src/crossval/preflight.rs:1393-1407` (MODIFIED)

```rust
    // Step 3: Attempt repair with retry logic
    if verbose {
        eprintln!();
        eprintln!("Backend '{}' not found at build time", backend.name());
        eprintln!();
        eprintln!("Auto-repairing... (this will take 5-10 minutes on first run)");
    }

    // ✅ FIX: Capture setup-cpp-auto output
    let setup_output = match attempt_repair_with_retry(backend, verbose) {
        Ok(output) => output,
        Err(e) => {
            eprintln!("\n{}", e);
            bail!("Auto-repair failed for backend '{}'", backend.name());
        }
    };

    // ✅ FIX: Parse environment exports
    let exports = match parse_sh_exports(&setup_output) {
        Ok(map) => map,
        Err(e) => {
            eprintln!("Failed to parse setup-cpp-auto output: {}", e);
            bail!("Environment parsing failed");
        }
    };

    // ✅ FIX: Apply to current process (for diagnostics/logging)
    for (key, value) in &exports {
        unsafe {
            env::set_var(key, value);
        }
    }

    // Step 4: Rebuild xtask to pick up new detection (AC3)
    if verbose {
        eprintln!("[repair] Step 2/3: Rebuilding xtask binary...");
    }

    // ✅ FIX: Pass environment to child cargo process
    if let Err(e) = rebuild_xtask_with_env(verbose, &exports) {
        eprintln!("\n{}", e);
        bail!("xtask rebuild failed after successful C++ setup");
    }
```

## Summary of Changes

| Component | Change | Gap Fixed |
|-----------|--------|-----------|
| `attempt_repair_once()` | Capture & return stdout | Gap 1 |
| `parse_sh_exports()` | NEW: Parse shell exports | Gap 3 |
| `rebuild_xtask_with_env()` | NEW: Apply env to cargo | Gap 2 |
| `preflight_with_auto_repair()` | Integrate parse + apply | All gaps |

**Total Lines Changed**: ~50 lines modified + 150 lines added (parsing + tests)

