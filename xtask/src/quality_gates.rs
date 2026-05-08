use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct QualityGateResult {
    pub name: String,
    pub description: String,
    pub passed: bool,
    pub message: String,
    pub details: Value,
}

impl QualityGateResult {
    fn new(name: &str, description: String) -> Self {
        Self {
            name: name.to_string(),
            description,
            passed: false,
            message: String::new(),
            details: json!({}),
        }
    }
}

#[derive(Debug, Serialize)]
struct QualityGateOutput {
    overall_passed: bool,
    gates: Vec<QualityGateResult>,
    summary_report: String,
}

pub fn run(
    coverage_report: &Path,
    performance_report: &Path,
    security_report: &Path,
    cross_platform_results: &str,
    output: Option<&Path>,
) -> Result<()> {
    println!("Evaluating release quality gates...");

    let (overall_passed, gates) = evaluate_all_gates(
        coverage_report,
        performance_report,
        security_report,
        cross_platform_results,
    );
    let summary_report = generate_summary_report(&gates, overall_passed);
    println!("{summary_report}");

    if let Some(output_path) = output {
        let results = QualityGateOutput { overall_passed, gates: gates.clone(), summary_report };
        let json =
            serde_json::to_string_pretty(&results).context("serialize quality gate output")?;
        fs::write(output_path, json)
            .with_context(|| format!("write quality gate output to {}", output_path.display()))?;
        println!("\nDetailed results saved to: {}", output_path.display());
    }

    if overall_passed {
        println!("\n✅ All quality gates passed - release candidate approved!");
        Ok(())
    } else {
        let failed_gates = gates
            .iter()
            .filter(|gate| !gate.passed)
            .map(|gate| gate.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::bail!("\n❌ Quality gates failed: {failed_gates}");
    }
}

fn evaluate_all_gates(
    coverage_file: &Path,
    performance_file: &Path,
    security_file: &Path,
    cross_platform_pattern: &str,
) -> (bool, Vec<QualityGateResult>) {
    let coverage_data = load_json_file(coverage_file);
    let performance_data = load_json_file(performance_file);
    let security_data = load_json_file(security_file);
    let build_results = find_build_results(cross_platform_pattern);

    let gates = vec![
        evaluate_coverage_gate(&coverage_data, 85.0),
        evaluate_performance_gate(&performance_data),
        evaluate_security_gate(&security_data),
        evaluate_cross_platform_gate(&build_results),
    ];
    let all_passed = gates.iter().all(|gate| gate.passed);

    (all_passed, gates)
}

fn load_json_file(file_path: &Path) -> Value {
    if !file_path.exists() {
        eprintln!("Warning: File not found: {}", file_path.display());
        return json!({});
    }

    match fs::read_to_string(file_path)
        .with_context(|| format!("read {}", file_path.display()))
        .and_then(|contents| serde_json::from_str(&contents).context("parse JSON"))
    {
        Ok(value) => value,
        Err(error) => {
            eprintln!("Error loading {}: {error}", file_path.display());
            json!({})
        }
    }
}

fn evaluate_coverage_gate(coverage_data: &Value, threshold: f64) -> QualityGateResult {
    let mut gate =
        QualityGateResult::new("Code Coverage", format!("Code coverage must be >= {threshold}%"));

    let Some(coverage_percentage) = coverage_percentage(coverage_data) else {
        gate.message = "No coverage data found".to_string();
        return gate;
    };

    gate.details = json!({
        "coverage_percentage": coverage_percentage,
        "threshold": threshold,
    });

    if coverage_percentage >= threshold {
        gate.passed = true;
        gate.message = format!("Coverage {coverage_percentage:.2}% meets threshold {threshold}%");
    } else {
        gate.message = format!("Coverage {coverage_percentage:.2}% below threshold {threshold}%");
    }

    gate
}

fn coverage_percentage(coverage_data: &Value) -> Option<f64> {
    if let Some(files) = coverage_data.get("files").and_then(Value::as_object) {
        let mut total_lines = 0_u64;
        let mut covered_lines = 0_u64;

        for file_data in files.values() {
            let Some(coverage) = file_data.get("coverage").and_then(Value::as_array) else {
                continue;
            };

            for line_data in coverage {
                if line_data.is_null() {
                    continue;
                }
                total_lines += 1;
                if line_data.as_f64().is_some_and(|hits| hits > 0.0) {
                    covered_lines += 1;
                }
            }
        }

        return (total_lines != 0).then_some((covered_lines as f64 / total_lines as f64) * 100.0);
    }

    coverage_data.get("coverage").and_then(Value::as_f64)
}

fn evaluate_performance_gate(performance_data: &Value) -> QualityGateResult {
    let mut gate = QualityGateResult::new(
        "Performance Regression",
        "No significant performance regressions allowed".to_string(),
    );
    let regressions =
        performance_data.get("regressions").and_then(Value::as_array).cloned().unwrap_or_default();

    gate.passed = performance_data.get("passed").and_then(Value::as_bool).unwrap_or(false);
    gate.details = json!({
        "regressions_count": regressions.len(),
        "regressions": regressions.into_iter().take(5).collect::<Vec<_>>(),
    });
    gate.message = if gate.passed {
        "No performance regressions detected".to_string()
    } else {
        format!(
            "{} performance regression(s) detected",
            gate.details["regressions_count"].as_u64().unwrap_or(0)
        )
    };

    gate
}

fn evaluate_security_gate(security_data: &Value) -> QualityGateResult {
    let mut gate = QualityGateResult::new(
        "Security Vulnerabilities",
        "No high or critical security vulnerabilities allowed".to_string(),
    );
    let vulnerabilities =
        security_data.get("vulnerabilities").and_then(Value::as_array).cloned().unwrap_or_default();
    let high_critical_vulns = vulnerabilities
        .iter()
        .filter(|vuln| {
            vuln.get("advisory")
                .and_then(|advisory| advisory.get("severity"))
                .and_then(Value::as_str)
                .map(str::to_ascii_lowercase)
                .is_some_and(|severity| matches!(severity.as_str(), "high" | "critical"))
        })
        .cloned()
        .collect::<Vec<_>>();

    gate.passed = high_critical_vulns.is_empty();
    gate.details = json!({
        "total_vulnerabilities": vulnerabilities.len(),
        "high_critical_count": high_critical_vulns.len(),
        "high_critical_vulns": high_critical_vulns.iter().take(3).cloned().collect::<Vec<_>>(),
    });
    gate.message = if gate.passed {
        format!("No high/critical vulnerabilities found ({} total)", vulnerabilities.len())
    } else {
        format!("{} high/critical vulnerabilities found", high_critical_vulns.len())
    };

    gate
}

fn evaluate_cross_platform_gate(build_results: &[PathBuf]) -> QualityGateResult {
    let mut gate = QualityGateResult::new(
        "Cross-Platform Builds",
        "All target platforms must build successfully".to_string(),
    );
    let mut successful_builds = Vec::new();
    let mut failed_builds = Vec::new();

    for result_dir in build_results {
        let name =
            result_dir.file_name().and_then(|name| name.to_str()).unwrap_or_default().to_string();
        if result_dir.is_dir() && dir_has_entries(result_dir) {
            successful_builds.push(name);
        } else {
            failed_builds.push(name);
        }
    }

    gate.passed = failed_builds.is_empty();
    gate.details = json!({
        "successful_builds": successful_builds,
        "failed_builds": failed_builds,
        "total_targets": build_results.len(),
    });
    gate.message = if gate.passed {
        format!(
            "All {} platform builds successful",
            gate.details["successful_builds"].as_array().map_or(0, Vec::len)
        )
    } else {
        format!(
            "{} platform build(s) failed",
            gate.details["failed_builds"].as_array().map_or(0, Vec::len)
        )
    };

    gate
}

fn dir_has_entries(path: &Path) -> bool {
    fs::read_dir(path).map(|mut entries| entries.next().is_some()).unwrap_or(false)
}

fn find_build_results(pattern: &str) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    else {
        return Vec::new();
    };

    entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| matches_pattern(path, pattern))
        .collect()
}

fn matches_pattern(path: &Path, pattern: &str) -> bool {
    let normalized = pattern.trim_end_matches('/');
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };

    if let Some((prefix, suffix)) = normalized.split_once('*') {
        name.starts_with(prefix) && name.ends_with(suffix)
    } else {
        name == normalized
    }
}

fn generate_summary_report(gates: &[QualityGateResult], overall_passed: bool) -> String {
    let mut report = Vec::new();
    report.push("# Release Quality Gates Summary".to_string());
    report.push(String::new());

    if overall_passed {
        report.push("## ✅ Overall Status: PASSED".to_string());
        report.push("All quality gates have been satisfied.".to_string());
    } else {
        report.push("## ❌ Overall Status: FAILED".to_string());
        report.push("One or more quality gates have failed.".to_string());
    }

    report.push(String::new());
    report.push("## Quality Gate Results".to_string());
    report.push(String::new());

    for gate in gates {
        let status_icon = if gate.passed { "✅" } else { "❌" };
        report.push(format!("### {status_icon} {}", gate.name));
        report.push(format!("**Description:** {}", gate.description));
        report.push(format!("**Status:** {}", if gate.passed { "PASSED" } else { "FAILED" }));
        report.push(format!("**Message:** {}", gate.message));

        if let Some(details) = gate.details.as_object().filter(|details| !details.is_empty()) {
            report.push("**Details:**".to_string());
            for (key, value) in details {
                if let Some(items) = value.as_array().filter(|items| !items.is_empty()) {
                    report.push(format!("- {key}: {} items", items.len()));
                } else {
                    report.push(format!("- {key}: {value}"));
                }
            }
        }

        report.push(String::new());
    }

    report.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_tarpaulin_coverage() {
        let gate = evaluate_coverage_gate(
            &json!({
                "files": {
                    "src/lib.rs": { "coverage": [1, 0, null, 3] }
                }
            }),
            60.0,
        );

        assert!(gate.passed);
        assert_eq!(gate.details["coverage_percentage"], json!(66.66666666666666));
    }

    #[test]
    fn flags_high_and_critical_vulnerabilities() {
        let gate = evaluate_security_gate(&json!({
            "vulnerabilities": [
                { "advisory": { "severity": "low" } },
                { "advisory": { "severity": "Critical" } }
            ]
        }));

        assert!(!gate.passed);
        assert_eq!(gate.details["high_critical_count"], json!(1));
    }

    #[test]
    fn wildcard_matching_supports_existing_default_pattern() {
        assert!(matches_pattern(Path::new("build-linux"), "build-*/"));
        assert!(!matches_pattern(Path::new("target"), "build-*/"));
    }
}
