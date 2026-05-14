//! Rust-native trace diffing for activation diagnostics.
//!
//! Trace files are JSON records produced by `bitnet-trace` when
//! `BITNET_TRACE_DIR` is set. The diff is diagnostic-only: it never updates
//! proof manifests or promotes backend/residency claims.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

#[derive(Debug, Clone, Deserialize)]
struct TraceRecord {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    blake3: String,
    rms: f64,
    num_elements: usize,
    seq: Option<usize>,
    layer: Option<isize>,
    stage: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct TraceDiffReport {
    diagnostic: &'static str,
    diagnostic_only: bool,
    promotion_allowed: bool,
    proof_receipts_written: bool,
    manifest_updated: bool,
    left_dir: String,
    right_dir: String,
    left_record_count: usize,
    right_record_count: usize,
    comparable_record_count: usize,
    matched_record_count: usize,
    divergent_record_count: usize,
    missing_left_count: usize,
    missing_right_count: usize,
    first_divergence: Option<TraceDivergence>,
    not_claims: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct TraceDivergence {
    key: String,
    kind: String,
    left: Option<TraceSummary>,
    right: Option<TraceSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct TraceSummary {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    blake3: String,
    rms: f64,
    num_elements: usize,
    seq: Option<usize>,
    layer: Option<isize>,
    stage: Option<String>,
}

impl From<&TraceRecord> for TraceSummary {
    fn from(record: &TraceRecord) -> Self {
        Self {
            name: record.name.clone(),
            shape: record.shape.clone(),
            dtype: record.dtype.clone(),
            blake3: record.blake3.clone(),
            rms: record.rms,
            num_elements: record.num_elements,
            seq: record.seq,
            layer: record.layer,
            stage: record.stage.clone(),
        }
    }
}

fn critical_not_claims() -> Vec<&'static str> {
    vec![
        "selected_attention_residency",
        "resident_kv_decode",
        "attention_scores_residency",
        "softmax_residency",
        "attention_value_mix_residency",
        "full_support_op_residency",
        "full_device_residency",
        "completion",
    ]
}

fn trace_key(record: &TraceRecord) -> String {
    format!(
        "seq={}|layer={}|stage={}|name={}",
        record.seq.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
        record.layer.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
        record.stage.as_deref().unwrap_or("none"),
        record.name
    )
}

fn read_trace_dir(dir: &Path) -> Result<BTreeMap<String, TraceRecord>> {
    if !dir.exists() {
        bail!("trace directory not found: {}", dir.display());
    }
    if !dir.is_dir() {
        bail!("trace path is not a directory: {}", dir.display());
    }

    let mut records = BTreeMap::new();
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("trace") {
            continue;
        }

        let json = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let record: TraceRecord = serde_json::from_str(&json)
            .with_context(|| format!("failed to parse trace JSON {}", path.display()))?;
        let key = trace_key(&record);
        if records.insert(key.clone(), record).is_some() {
            bail!("duplicate trace key {key} in {}", dir.display());
        }
    }

    if records.is_empty() {
        bail!("trace directory is empty: {}", dir.display());
    }

    Ok(records)
}

fn compare_trace_dirs(left_dir: &Path, right_dir: &Path) -> Result<TraceDiffReport> {
    let left = read_trace_dir(left_dir)?;
    let right = read_trace_dir(right_dir)?;
    let keys = left.keys().chain(right.keys()).cloned().collect::<BTreeSet<_>>();

    let mut comparable_record_count = 0usize;
    let mut matched_record_count = 0usize;
    let mut divergent_record_count = 0usize;
    let mut missing_left_count = 0usize;
    let mut missing_right_count = 0usize;
    let mut first_divergence = None;

    for key in keys {
        match (left.get(&key), right.get(&key)) {
            (Some(left_record), Some(right_record)) => {
                comparable_record_count += 1;
                let matches = left_record.shape == right_record.shape
                    && left_record.dtype == right_record.dtype
                    && left_record.blake3 == right_record.blake3;
                if matches {
                    matched_record_count += 1;
                } else {
                    divergent_record_count += 1;
                    if first_divergence.is_none() {
                        let kind = if left_record.shape != right_record.shape {
                            "shape_mismatch"
                        } else if left_record.dtype != right_record.dtype {
                            "dtype_mismatch"
                        } else {
                            "hash_mismatch"
                        };
                        first_divergence = Some(TraceDivergence {
                            key,
                            kind: kind.to_string(),
                            left: Some(left_record.into()),
                            right: Some(right_record.into()),
                        });
                    }
                }
            }
            (None, Some(right_record)) => {
                missing_left_count += 1;
                if first_divergence.is_none() {
                    first_divergence = Some(TraceDivergence {
                        key,
                        kind: "missing_left".to_string(),
                        left: None,
                        right: Some(right_record.into()),
                    });
                }
            }
            (Some(left_record), None) => {
                missing_right_count += 1;
                if first_divergence.is_none() {
                    first_divergence = Some(TraceDivergence {
                        key,
                        kind: "missing_right".to_string(),
                        left: Some(left_record.into()),
                        right: None,
                    });
                }
            }
            (None, None) => unreachable!("trace key came from at least one input"),
        }
    }

    Ok(TraceDiffReport {
        diagnostic: "bitnet_trace_diff",
        diagnostic_only: true,
        promotion_allowed: false,
        proof_receipts_written: false,
        manifest_updated: false,
        left_dir: left_dir.display().to_string(),
        right_dir: right_dir.display().to_string(),
        left_record_count: left.len(),
        right_record_count: right.len(),
        comparable_record_count,
        matched_record_count,
        divergent_record_count,
        missing_left_count,
        missing_right_count,
        first_divergence,
        not_claims: critical_not_claims(),
    })
}

fn print_human(report: &TraceDiffReport) {
    println!("trace diff: diagnostic_only=true promotion_allowed=false");
    println!("left:  {} records ({})", report.left_record_count, report.left_dir);
    println!("right: {} records ({})", report.right_record_count, report.right_dir);
    println!(
        "matched={} divergent={} missing_left={} missing_right={}",
        report.matched_record_count,
        report.divergent_record_count,
        report.missing_left_count,
        report.missing_right_count
    );

    if let Some(divergence) = &report.first_divergence {
        println!("first_divergence: {} ({})", divergence.key, divergence.kind);
        if let (Some(left), Some(right)) = (&divergence.left, &divergence.right) {
            println!(
                "left:  shape={:?} dtype={} rms={:.8} blake3={}",
                left.shape, left.dtype, left.rms, left.blake3
            );
            println!(
                "right: shape={:?} dtype={} rms={:.8} blake3={}",
                right.shape, right.dtype, right.rms, right.blake3
            );
        }
    } else {
        println!("all tracepoints match");
    }
}

/// Compare two trace directories and report the first divergence.
pub fn run(left_dir: &Path, right_dir: &Path, format: &str) -> Result<()> {
    let report = compare_trace_dirs(left_dir, right_dir)?;
    match format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "human" => print_human(&report),
        other => bail!("unsupported trace-diff format: {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn write_trace(dir: &Path, file: &str, name: &str, hash: &str, rms: f64) {
        let path = dir.join(file);
        let record = serde_json::json!({
            "name": name,
            "shape": [1, 2],
            "dtype": "F32",
            "blake3": hash,
            "rms": rms,
            "num_elements": 2,
            "seq": 0,
            "layer": 0,
            "stage": name
        });
        fs::write(path, serde_json::to_string_pretty(&record).unwrap()).unwrap();
    }

    #[test]
    fn missing_trace_dir_errors() {
        let left = PathBuf::from("/nonexistent/left");
        let right = PathBuf::from("/nonexistent/right");

        let result = compare_trace_dirs(&left, &right);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("trace directory not found"));
    }

    #[test]
    fn matching_trace_dirs_are_diagnostic_not_promoting() {
        let left = tempdir().unwrap();
        let right = tempdir().unwrap();
        write_trace(left.path(), "a.trace", "attention_q", "abc", 1.0);
        write_trace(right.path(), "a.trace", "attention_q", "abc", 1.0);

        let report = compare_trace_dirs(left.path(), right.path()).unwrap();
        assert!(report.diagnostic_only);
        assert!(!report.promotion_allowed);
        assert_eq!(report.matched_record_count, 1);
        assert_eq!(report.divergent_record_count, 0);
        assert!(report.first_divergence.is_none());
        assert!(report.not_claims.contains(&"selected_attention_residency"));
    }

    #[test]
    fn hash_mismatch_reports_first_divergence() {
        let left = tempdir().unwrap();
        let right = tempdir().unwrap();
        write_trace(left.path(), "a.trace", "attention_q", "abc", 1.0);
        write_trace(right.path(), "a.trace", "attention_q", "def", 1.25);

        let report = compare_trace_dirs(left.path(), right.path()).unwrap();
        assert_eq!(report.divergent_record_count, 1);
        let divergence = report.first_divergence.unwrap();
        assert_eq!(divergence.kind, "hash_mismatch");
        assert!(divergence.key.contains("attention_q"));
    }
}
