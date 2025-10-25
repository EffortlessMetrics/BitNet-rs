//! Trace diff wrapper for cross-validation debugging.
//!
//! This module provides a Rust wrapper for the Python `scripts/trace_diff.py`
//! script, which performs Blake3 hash comparison of trace files captured during
//! Rust vs C++ cross-validation runs.
//!
//! # Usage
//!
//! ```bash
//! # Capture traces (example)
//! BITNET_TRACE_DIR=/tmp/rs cargo run -p bitnet-cli --features cpu,trace -- \
//!   run --model model.gguf --tokenizer tok.json --prompt "Test" --max-tokens 4
//!
//! # (capture C++ trace to /tmp/cpp using C++ inference)
//!
//! # Compare traces
//! cargo run -p xtask -- trace-diff /tmp/rs /tmp/cpp
//! ```
//!
//! # Output
//!
//! The tool prints:
//! - First divergence point: `(seq, layer, stage)` where traces differ
//! - "All tracepoints match" if traces are identical
//! - Error diagnostics if trace files are missing or Python is unavailable

use anyhow::{Context, Result, bail};
use std::{fs, path::Path, process::Command};

/// Find available Python interpreter
///
/// Tries interpreters in order: python3, python, py
fn find_python() -> Result<String> {
    for interp in ["python3", "python", "py"] {
        if Command::new(interp)
            .arg("--version")
            .output()
            .ok()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return Ok(interp.to_string());
        }
    }
    bail!(
        "Python interpreter not found (tried python3, python, py).\n\
         Install Python 3.7+ to use trace comparison."
    )
}

/// Check if a directory has trace files
fn has_trace_files(dir: &Path) -> bool {
    fs::read_dir(dir).ok().map(|entries| entries.count() > 0).unwrap_or(false)
}

/// Compare Rust vs C++ traces and report first divergence
///
/// # Arguments
///
/// - `rs_dir`: Path to directory containing Rust trace files
/// - `cpp_dir`: Path to directory containing C++ trace files
///
/// # Workflow
///
/// 1. Validates both trace directories exist
/// 2. Checks `scripts/trace_diff.py` is available in repo
/// 3. Executes Python script via `python3` subprocess
/// 4. Propagates exit code from Python script
///
/// # Returns
///
/// - `Ok(())` if comparison succeeds (traces match or divergence found)
/// - `Err(_)` if:
///   - Trace directories don't exist
///   - `trace_diff.py` script not found
///   - `python3` not available
///   - Script execution fails
pub fn run(rs_dir: &Path, cpp_dir: &Path) -> Result<()> {
    // 1) Validate trace directories exist
    if !rs_dir.exists() {
        eprintln!("❌ Rust trace directory not found: {}", rs_dir.display());
        eprintln!();
        eprintln!("How to capture Rust traces:");
        eprintln!(
            "  BITNET_TRACE_DIR=/tmp/rs RUST_LOG=warn BITNET_DETERMINISTIC=1 BITNET_SEED=42 \\"
        );
        eprintln!("    cargo run -p bitnet-cli --features cpu,trace -- run \\");
        eprintln!("    --model <model.gguf> --tokenizer <tokenizer.json> \\");
        eprintln!("    --prompt \"What is 2+2?\" --max-tokens 4 --greedy");
        std::process::exit(2); // Usage error
    }

    if !cpp_dir.exists() {
        eprintln!("❌ C++ trace directory not found: {}", cpp_dir.display());
        eprintln!();
        eprintln!("How to capture C++ traces:");
        eprintln!("  See docs/howto/cpp-setup.md for C++ instrumentation and trace capture");
        std::process::exit(2); // Usage error
    }

    // 2) Check if directories are empty
    if !has_trace_files(rs_dir) {
        eprintln!("❌ Rust trace directory is empty: {}", rs_dir.display());
        eprintln!();
        eprintln!("How to capture Rust traces:");
        eprintln!(
            "  BITNET_TRACE_DIR=/tmp/rs RUST_LOG=warn BITNET_DETERMINISTIC=1 BITNET_SEED=42 \\"
        );
        eprintln!("    cargo run -p bitnet-cli --features cpu,trace -- run \\");
        eprintln!("    --model <model.gguf> --tokenizer <tokenizer.json> \\");
        eprintln!("    --prompt \"What is 2+2?\" --max-tokens 4 --greedy");
        std::process::exit(2); // Usage error
    }

    if !has_trace_files(cpp_dir) {
        eprintln!("❌ C++ trace directory is empty: {}", cpp_dir.display());
        eprintln!();
        eprintln!("How to capture C++ traces:");
        eprintln!("  See docs/howto/cpp-setup.md for C++ instrumentation and trace capture");
        std::process::exit(2); // Usage error
    }

    // 3) Verify trace_diff.py script exists
    let script = Path::new("scripts/trace_diff.py");
    if !script.exists() {
        bail!(
            "scripts/trace_diff.py not found at {}\n\
             Are you running from the repository root?",
            script.display()
        );
    }

    // 4) Find Python interpreter
    let python = find_python()?;

    // 5) Execute Python script
    eprintln!("[bitnet] Comparing traces:");
    eprintln!("  Rust:  {}", rs_dir.display());
    eprintln!("  C++:   {}", cpp_dir.display());
    eprintln!();

    let status = Command::new(&python)
        .arg(script)
        .arg(rs_dir)
        .arg(cpp_dir)
        .status()
        .context(format!("failed to spawn {} for trace_diff.py", python))?;

    // 6) Propagate exit code
    if !status.success() {
        bail!("trace_diff.py failed with exit code: {:?}", status.code().unwrap_or(-1));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_run_missing_dirs() {
        let rs_dir = PathBuf::from("/nonexistent/rs");
        let cpp_dir = PathBuf::from("/nonexistent/cpp");

        let result = run(&rs_dir, &cpp_dir);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("trace directory not found"));
    }
}
