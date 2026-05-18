//! Native trace diff support for cross-validation debugging.
//!
//! This module compares trace files captured during Rust vs C++ cross-validation
//! runs.  It intentionally lives in `xtask` instead of a Python helper so the
//! workflow uses the repository's Rust developer-tooling path and works in
//! minimal environments without a Python interpreter.
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
//! - Error diagnostics if trace files are missing or malformed

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct TraceKey {
    seq: i64,
    layer: i64,
    stage: String,
}

type TraceMap = BTreeMap<TraceKey, Value>;

/// Check if a directory has any entries.
fn has_trace_files(dir: &Path) -> bool {
    fs::read_dir(dir).ok().map(|entries| entries.count() > 0).unwrap_or(false)
}

fn is_trace_path(path: &Path) -> bool {
    matches!(path.extension().and_then(|ext| ext.to_str()), Some("trace" | "jsonl"))
}

fn trace_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(dir)
        .with_context(|| format!("failed to read trace directory {}", dir.display()))?
        .filter_map(|entry| match entry {
            Ok(entry) => Some(entry.path()),
            Err(err) => {
                eprintln!("Warning: Failed to read trace directory entry: {err}");
                None
            }
        })
        .filter(|path| path.is_file() && is_trace_path(path))
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

fn integer_field(record: &Value, name: &str) -> Option<i64> {
    let value = record.get(name)?;
    value.as_i64().or_else(|| value.as_u64().and_then(|v| i64::try_from(v).ok()))
}

fn key_from_record(record: &Value) -> Option<TraceKey> {
    Some(TraceKey {
        seq: integer_field(record, "seq")?,
        layer: integer_field(record, "layer")?,
        stage: record.get("stage")?.as_str()?.to_owned(),
    })
}

fn load_traces(directory: &Path) -> Result<TraceMap> {
    let mut traces = TraceMap::new();

    for path in trace_files(directory)? {
        let file =
            File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
        let reader = BufReader::new(file);

        for (line_no, line) in reader.lines().enumerate() {
            let line =
                line.with_context(|| format!("failed to read {}:{}", path.display(), line_no + 1))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let record: Value = match serde_json::from_str(line) {
                Ok(record) => record,
                Err(err) => {
                    eprintln!("Warning: Failed to parse {}:{}: {err}", path.display(), line_no + 1);
                    continue;
                }
            };

            // Skip records without seq/layer/stage for backward compatibility.
            if let Some(key) = key_from_record(&record) {
                traces.insert(key, record);
            }
        }
    }

    Ok(traces)
}

fn shape(record: &Value) -> Option<&Value> {
    record.get("shape")
}

fn dtype(record: &Value) -> Option<&str> {
    record.get("dtype").and_then(Value::as_str)
}

fn blake3(record: &Value) -> &str {
    record.get("blake3").and_then(Value::as_str).unwrap_or("")
}

fn f64_field(record: &Value, name: &str) -> f64 {
    record.get(name).and_then(Value::as_f64).unwrap_or(0.0)
}

fn u64_field(record: &Value, name: &str) -> u64 {
    record.get(name).and_then(Value::as_u64).unwrap_or(0)
}

fn print_key(prefix: &str, key: &TraceKey) {
    println!("{prefix}: seq={}, layer={}, stage={}", key.seq, key.layer, key.stage);
}

fn compare_trace_maps(rust_traces: &TraceMap, cpp_traces: &TraceMap) -> Result<()> {
    let all_keys = rust_traces.keys().chain(cpp_traces.keys()).cloned().collect::<BTreeSet<_>>();

    for key in all_keys {
        let Some(rust_rec) = rust_traces.get(&key) else {
            print_key("✗ Missing in Rust", &key);
            bail!("trace divergence detected");
        };
        let Some(cpp_rec) = cpp_traces.get(&key) else {
            print_key("✗ Missing in C++", &key);
            bail!("trace divergence detected");
        };

        if shape(rust_rec) != shape(cpp_rec) {
            print_key("✗ Shape mismatch at", &key);
            println!("  Rust shape: {}", shape(rust_rec).unwrap_or(&Value::Null));
            println!("  C++ shape:  {}", shape(cpp_rec).unwrap_or(&Value::Null));
            bail!("trace divergence detected");
        }

        if dtype(rust_rec) != dtype(cpp_rec) {
            print_key("✗ Dtype mismatch at", &key);
            println!("  Rust dtype: {}", dtype(rust_rec).unwrap_or(""));
            println!("  C++ dtype:  {}", dtype(cpp_rec).unwrap_or(""));
            bail!("trace divergence detected");
        }

        let rust_hash = blake3(rust_rec);
        let cpp_hash = blake3(cpp_rec);
        if rust_hash != cpp_hash {
            print_key("✗ First divergence at", &key);
            println!("  Rust blake3: {}...", hash_prefix(rust_hash));
            println!("  C++ blake3:  {}...", hash_prefix(cpp_hash));
            println!(
                "  Rust stats:  rms={:.6}, num_elements={}",
                f64_field(rust_rec, "rms"),
                u64_field(rust_rec, "num_elements")
            );
            println!(
                "  C++ stats:   rms={:.6}, num_elements={}",
                f64_field(cpp_rec, "rms"),
                u64_field(cpp_rec, "num_elements")
            );
            bail!("trace divergence detected");
        }
    }

    println!("✓ All tracepoints match");
    Ok(())
}

fn hash_prefix(hash: &str) -> &str {
    hash.get(..16).unwrap_or(hash)
}

/// Compare Rust vs C++ traces and report first divergence.
///
/// # Arguments
///
/// - `rs_dir`: Path to directory containing Rust trace files
/// - `cpp_dir`: Path to directory containing C++ trace files
///
/// # Workflow
///
/// 1. Validates both trace directories exist
/// 2. Reads `.trace` and `.jsonl` newline-delimited JSON trace records
/// 3. Compares sorted `(seq, layer, stage)` records for shape, dtype, and Blake3 hash
/// 4. Returns an error when the first divergence is found
pub fn run(rs_dir: &Path, cpp_dir: &Path) -> Result<()> {
    // 1) Validate trace directories exist
    if !rs_dir.exists() {
        eprintln!("❌ Rust trace directory not found: {}", rs_dir.display());
        eprintln!();
        eprintln!("How to capture Rust traces:");
        eprintln!("  BITNET_TRACE_DIR=/tmp/rs RUST_LOG=warn BITNET_DETERMINISTIC=1 BITNET_SEED=42");
        eprintln!("    cargo run -p bitnet-cli --features cpu,trace -- run \\");
        eprintln!("    --model <model.gguf> --tokenizer <tokenizer.json> \\");
        eprintln!("    --prompt \"What is 2+2?\" --max-tokens 4 --greedy");
        bail!("Rust trace directory not found: {}", rs_dir.display());
    }

    if !cpp_dir.exists() {
        eprintln!("❌ C++ trace directory not found: {}", cpp_dir.display());
        eprintln!();
        eprintln!("How to capture C++ traces:");
        eprintln!("  See docs/howto/cpp-setup.md for C++ instrumentation and trace capture");
        bail!("C++ trace directory not found: {}", cpp_dir.display());
    }

    // 2) Check if directories are empty
    if !has_trace_files(rs_dir) {
        eprintln!("❌ Rust trace directory is empty: {}", rs_dir.display());
        eprintln!();
        eprintln!("How to capture Rust traces:");
        eprintln!("  BITNET_TRACE_DIR=/tmp/rs RUST_LOG=warn BITNET_DETERMINISTIC=1 BITNET_SEED=42");
        eprintln!("    cargo run -p bitnet-cli --features cpu,trace -- run \\");
        eprintln!("    --model <model.gguf> --tokenizer <tokenizer.json> \\");
        eprintln!("    --prompt \"What is 2+2?\" --max-tokens 4 --greedy");
        bail!("Rust trace directory is empty: {}", rs_dir.display());
    }

    if !has_trace_files(cpp_dir) {
        eprintln!("❌ C++ trace directory is empty: {}", cpp_dir.display());
        eprintln!();
        eprintln!("How to capture C++ traces:");
        eprintln!("  See docs/howto/cpp-setup.md for C++ instrumentation and trace capture");
        bail!("C++ trace directory is empty: {}", cpp_dir.display());
    }

    eprintln!("[bitnet] Comparing traces:");
    eprintln!("  Rust:  {}", rs_dir.display());
    eprintln!("  C++:   {}", cpp_dir.display());
    eprintln!();

    println!("Loading Rust traces from {}...", rs_dir.display());
    let rust_traces = load_traces(rs_dir)?;
    println!("  Loaded {} Rust tracepoints", rust_traces.len());

    println!("Loading C++ traces from {}...", cpp_dir.display());
    let cpp_traces = load_traces(cpp_dir)?;
    println!("  Loaded {} C++ tracepoints", cpp_traces.len());
    println!();

    if rust_traces.is_empty() || cpp_traces.is_empty() {
        bail!(
            "no comparable trace records found; expected newline-delimited JSON with seq, layer, and stage fields"
        );
    }

    compare_trace_maps(&rust_traces, &cpp_traces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, path::PathBuf};
    use tempfile::tempdir;

    fn write_trace(dir: &Path, name: &str, hash: &str) -> Result<()> {
        fs::write(
            dir.join(name),
            format!(
                r#"{{"seq":0,"layer":1,"stage":"attn","shape":[1,2],"dtype":"f32","blake3":"{hash}","rms":1.25,"num_elements":2}}
"#
            ),
        )?;
        Ok(())
    }

    #[test]
    fn test_run_missing_dirs() {
        let rs_dir = PathBuf::from("/nonexistent/rs");
        let cpp_dir = PathBuf::from("/nonexistent/cpp");

        let result = run(&rs_dir, &cpp_dir);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("trace directory not found"));
    }

    #[test]
    fn test_load_traces_skips_records_without_keys() -> Result<()> {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("sample.trace"),
            "{\"event\":\"legacy\"}\n{\"seq\":1,\"layer\":2,\"stage\":\"mlp\",\"blake3\":\"abc\"}\n",
        )?;

        let traces = load_traces(dir.path())?;
        assert_eq!(traces.len(), 1);
        assert!(traces.contains_key(&TraceKey { seq: 1, layer: 2, stage: "mlp".to_owned() }));
        Ok(())
    }

    #[test]
    fn test_compare_matching_traces() -> Result<()> {
        let rs_dir = tempdir()?;
        let cpp_dir = tempdir()?;
        write_trace(rs_dir.path(), "a.trace", "abcdef0123456789")?;
        write_trace(cpp_dir.path(), "a.jsonl", "abcdef0123456789")?;

        run(rs_dir.path(), cpp_dir.path())?;
        Ok(())
    }

    #[test]
    fn test_compare_detects_hash_divergence() -> Result<()> {
        let rs_dir = tempdir()?;
        let cpp_dir = tempdir()?;
        write_trace(rs_dir.path(), "a.trace", "abcdef0123456789")?;
        write_trace(cpp_dir.path(), "a.trace", "fedcba9876543210")?;

        match run(rs_dir.path(), cpp_dir.path()) {
            Ok(()) => bail!("expected trace divergence"),
            Err(err) => assert!(err.to_string().contains("trace divergence detected")),
        }
        Ok(())
    }
}
