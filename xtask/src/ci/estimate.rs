//! `xtask ci estimate` — learned LEM estimates from observed actuals.
//!
//! PR 20 of the strict policy / CI economics rollout. Reads a
//! rolling history of `ci-actuals.json` records (one per workflow
//! run) and emits a per-lane learned estimate:
//!
//!   estimate = max(static_floor, p50_recent_actual * 1.15)
//!   warning  = p90_recent_actual
//!   hard     = p95_recent_actual
//!
//! The static floor comes from `policy/ci-lanes.toml`. The history
//! file is a JSON Lines append-only ledger. The model is
//! deliberately simple: PR 20 ships the calculation and the
//! consumer surface so the planner (PR 14) and budget guard
//! (PR 18) can switch from static estimates to learned ones in a
//! follow-up.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneEstimate {
    pub lane: String,
    pub samples: usize,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub static_floor: f64,
    pub estimate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimateReport {
    pub schema_version: u32,
    pub generated_at: String,
    pub window_runs: usize,
    pub lanes: BTreeMap<String, LaneEstimate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistoryEntry {
    #[serde(default)]
    pub lane: String,
    #[serde(default)]
    pub actual_lem: f64,
    #[serde(default)]
    pub conclusion: String,
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (pct / 100.0) * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = rank - rank.floor();
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

/// Compute a learned estimate from a list of recent actual LEMs.
pub fn lane_estimate(lane: &str, samples: &[f64], static_floor: f64) -> LaneEstimate {
    let mut sorted: Vec<f64> = samples.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p50 = percentile(&sorted, 50.0);
    let p90 = percentile(&sorted, 90.0);
    let p95 = percentile(&sorted, 95.0);

    let learned = (p50 * 1.15).max(static_floor);

    LaneEstimate {
        lane: lane.to_string(),
        samples: sorted.len(),
        p50,
        p90,
        p95,
        static_floor,
        estimate: learned,
    }
}

fn read_history(path: &Path) -> Result<Vec<HistoryEntry>> {
    if !path.exists() {
        return Ok(vec![]);
    }
    let body = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in body.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let entry: HistoryEntry = serde_json::from_str(trimmed)
            .with_context(|| format!("parsing line {} of {}", i + 1, path.display()))?;
        out.push(entry);
    }
    Ok(out)
}

fn read_static_floors(path: &Path) -> Result<BTreeMap<String, f64>> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let body = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let value: toml::Value =
        toml::from_str(&body).with_context(|| format!("parsing {}", path.display()))?;

    let mut out = BTreeMap::new();
    if let Some(lane_section) = value.get("lane").and_then(|v| v.as_table()) {
        for (id, table) in lane_section {
            if let Some(base_lem) = table.get("base_lem").and_then(|v| v.as_float()) {
                out.insert(id.clone(), base_lem);
            } else if let Some(base_lem) = table.get("base_lem").and_then(|v| v.as_integer()) {
                out.insert(id.clone(), base_lem as f64);
            }
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    history: PathBuf,
    lanes_toml: PathBuf,
    json_out: PathBuf,
    print_stdout: bool,
    window: usize,
) -> Result<()> {
    let entries = read_history(&history)?;
    let floors = read_static_floors(&lanes_toml)?;

    if window == 0 {
        bail!("--window must be > 0");
    }

    let mut by_lane: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for e in entries.iter().rev().take(window) {
        if e.lane.is_empty() {
            continue;
        }
        by_lane.entry(e.lane.clone()).or_default().push(e.actual_lem);
    }

    let mut lanes = BTreeMap::new();
    for (lane, samples) in by_lane {
        let floor = *floors.get(&lane).unwrap_or(&0.0);
        let est = lane_estimate(&lane, &samples, floor);
        lanes.insert(lane, est);
    }

    let report = EstimateReport {
        schema_version: 1,
        generated_at: chrono::Utc::now().to_rfc3339(),
        window_runs: window,
        lanes,
    };

    let body = serde_json::to_string_pretty(&report)?;
    if print_stdout {
        println!("{body}");
    }
    if let Some(parent) = json_out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(&json_out, &body).with_context(|| format!("writing {}", json_out.display()))?;
    println!(
        "ci-estimate: wrote {} ({} lanes from up to {} samples)",
        json_out.display(),
        report.lanes.len(),
        window
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_midpoints() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&v, 50.0) - 3.0).abs() < 1e-9);
        assert!((percentile(&v, 90.0) - 4.6).abs() < 1e-9);
        assert!((percentile(&v, 100.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn estimate_uses_p50_times_1_15() {
        let samples = vec![10.0, 20.0, 30.0];
        let e = lane_estimate("lane-a", &samples, 0.0);
        assert!((e.p50 - 20.0).abs() < 1e-9);
        assert!((e.estimate - 23.0).abs() < 1e-9);
    }

    #[test]
    fn estimate_respects_static_floor() {
        let samples = vec![1.0, 1.0, 1.0];
        let e = lane_estimate("lane-a", &samples, 50.0);
        assert!((e.p50 - 1.0).abs() < 1e-9);
        assert!((e.estimate - 50.0).abs() < 1e-9);
    }

    #[test]
    fn empty_samples_yields_floor() {
        let e = lane_estimate("lane-a", &[], 12.0);
        assert_eq!(e.samples, 0);
        assert!((e.estimate - 12.0).abs() < 1e-9);
    }

    #[test]
    fn reads_jsonl_history() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("hist.jsonl");
        let body = "\
{\"lane\":\"a\",\"actual_lem\":1.0,\"conclusion\":\"success\"}
{\"lane\":\"b\",\"actual_lem\":2.0,\"conclusion\":\"success\"}
# comment
{\"lane\":\"a\",\"actual_lem\":3.0,\"conclusion\":\"success\"}
";
        std::fs::write(&p, body).unwrap();
        let entries = read_history(&p).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].lane, "a");
        assert_eq!(entries[2].actual_lem, 3.0);
    }

    #[test]
    fn reads_static_floors_from_lanes_toml() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("lanes.toml");
        std::fs::write(
            &p,
            r#"
[lane.alpha]
base_lem = 12
[lane.beta]
base_lem = 1.5
"#,
        )
        .unwrap();
        let m = read_static_floors(&p).unwrap();
        assert_eq!(m.len(), 2);
        assert!((m["alpha"] - 12.0).abs() < 1e-9);
        assert!((m["beta"] - 1.5).abs() < 1e-9);
    }
}
