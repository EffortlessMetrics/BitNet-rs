//! Model health checking utilities.
//!
//! Quick validation of model files before full loading.

use std::collections::HashMap;

/// Health check result.
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Warning(String),
    Error(String),
}

impl HealthStatus {
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

/// Individual check result.
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub name: String,
    pub status: HealthStatus,
    pub duration_us: u64,
}

/// Model health report.
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub checks: Vec<CheckResult>,
    pub model_path: String,
    pub total_duration_us: u64,
}

impl HealthReport {
    pub fn is_healthy(&self) -> bool {
        self.checks.iter().all(|c| c.status.is_healthy())
    }

    pub fn error_count(&self) -> usize {
        self.checks.iter().filter(|c| c.status.is_error()).count()
    }

    pub fn warning_count(&self) -> usize {
        self.checks.iter().filter(|c| matches!(c.status, HealthStatus::Warning(_))).count()
    }

    pub fn errors(&self) -> Vec<&str> {
        self.checks
            .iter()
            .filter_map(|c| {
                if let HealthStatus::Error(msg) = &c.status { Some(msg.as_str()) } else { None }
            })
            .collect()
    }
}

/// Validate expected tensor names exist in a model.
pub fn check_expected_tensors(available: &[String], expected_patterns: &[&str]) -> HealthStatus {
    let missing: Vec<&&str> = expected_patterns
        .iter()
        .filter(|pat| !available.iter().any(|a| a.contains(**pat)))
        .collect();
    if missing.is_empty() {
        HealthStatus::Healthy
    } else {
        HealthStatus::Error(format!(
            "missing tensors: {}",
            missing.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", ")
        ))
    }
}

/// Check tensor shapes for consistency.
pub fn check_shape_consistency(shapes: &HashMap<String, Vec<usize>>) -> HealthStatus {
    // Find hidden sizes from known layers
    let hidden_sizes: Vec<usize> = shapes
        .iter()
        .filter(|(k, _)| k.contains("q_proj") || k.contains("self_attn.q_proj"))
        .filter_map(|(_, v)| v.last().copied())
        .collect();

    if hidden_sizes.is_empty() {
        return HealthStatus::Warning("no attention layers found".into());
    }

    let first = hidden_sizes[0];
    if hidden_sizes.iter().all(|&h| h == first) {
        HealthStatus::Healthy
    } else {
        HealthStatus::Error(format!("inconsistent hidden sizes: {hidden_sizes:?}"))
    }
}

/// Check for NaN/Inf in weight samples.
pub fn check_weight_values(samples: &[f32]) -> HealthStatus {
    let nan_count = samples.iter().filter(|v| v.is_nan()).count();
    let inf_count = samples.iter().filter(|v| v.is_infinite()).count();

    if nan_count > 0 {
        HealthStatus::Error(format!("{nan_count} NaN values detected"))
    } else if inf_count > 0 {
        HealthStatus::Warning(format!("{inf_count} Inf values detected"))
    } else {
        HealthStatus::Healthy
    }
}

/// Check vocabulary size is reasonable.
pub fn check_vocab_size(vocab_size: usize, model_family: &str) -> HealthStatus {
    let expected_range = match model_family {
        "llama" | "mistral" => (30_000, 140_000),
        "phi" => (30_000, 110_000),
        "qwen" => (100_000, 160_000),
        "gemma" => (200_000, 270_000),
        _ => (1_000, 300_000),
    };

    if vocab_size >= expected_range.0 && vocab_size <= expected_range.1 {
        HealthStatus::Healthy
    } else {
        HealthStatus::Warning(format!(
            "vocab {vocab_size} outside expected range {}-{} for {model_family}",
            expected_range.0, expected_range.1
        ))
    }
}

/// Check model file size is reasonable.
pub fn check_file_size(bytes: u64, param_count_millions: u64, bits_per_param: u64) -> HealthStatus {
    if param_count_millions == 0 || bits_per_param == 0 {
        return HealthStatus::Warning("cannot estimate expected size".into());
    }
    let expected = param_count_millions * 1_000_000 * bits_per_param / 8;
    let ratio = bytes as f64 / expected as f64;

    if ratio > 0.5 && ratio < 2.0 {
        HealthStatus::Healthy
    } else if ratio > 0.2 && ratio < 5.0 {
        HealthStatus::Warning(format!("file size {bytes} is {ratio:.1}x expected {expected}"))
    } else {
        HealthStatus::Error(format!("file size {bytes} is {ratio:.1}x expected {expected}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_healthy_status() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(!HealthStatus::Healthy.is_error());
    }

    #[test]
    fn test_error_status() {
        let s = HealthStatus::Error("bad".into());
        assert!(!s.is_healthy());
        assert!(s.is_error());
    }

    #[test]
    fn test_report_all_healthy() {
        let r = HealthReport {
            checks: vec![
                CheckResult { name: "a".into(), status: HealthStatus::Healthy, duration_us: 10 },
                CheckResult { name: "b".into(), status: HealthStatus::Healthy, duration_us: 5 },
            ],
            model_path: "m.gguf".into(),
            total_duration_us: 15,
        };
        assert!(r.is_healthy());
        assert_eq!(r.error_count(), 0);
    }

    #[test]
    fn test_report_with_error() {
        let r = HealthReport {
            checks: vec![
                CheckResult { name: "a".into(), status: HealthStatus::Healthy, duration_us: 10 },
                CheckResult {
                    name: "b".into(),
                    status: HealthStatus::Error("fail".into()),
                    duration_us: 5,
                },
            ],
            model_path: "m.gguf".into(),
            total_duration_us: 15,
        };
        assert!(!r.is_healthy());
        assert_eq!(r.error_count(), 1);
        assert_eq!(r.errors(), vec!["fail"]);
    }

    #[test]
    fn test_report_warnings() {
        let r = HealthReport {
            checks: vec![CheckResult {
                name: "a".into(),
                status: HealthStatus::Warning("w".into()),
                duration_us: 1,
            }],
            model_path: "m.gguf".into(),
            total_duration_us: 1,
        };
        assert!(!r.is_healthy());
        assert_eq!(r.warning_count(), 1);
    }

    #[test]
    fn test_check_tensors_all_present() {
        let avail = vec!["model.embed.weight".into(), "model.layers.0.q_proj.weight".into()];
        let s = check_expected_tensors(&avail, &["embed", "q_proj"]);
        assert!(s.is_healthy());
    }

    #[test]
    fn test_check_tensors_missing() {
        let avail = vec!["model.embed.weight".into()];
        let s = check_expected_tensors(&avail, &["embed", "lm_head"]);
        assert!(s.is_error());
    }

    #[test]
    fn test_shape_consistency_ok() {
        let mut shapes = HashMap::new();
        shapes.insert("layer.0.q_proj".into(), vec![4096, 4096]);
        shapes.insert("layer.1.q_proj".into(), vec![4096, 4096]);
        let s = check_shape_consistency(&shapes);
        assert!(s.is_healthy());
    }

    #[test]
    fn test_shape_consistency_bad() {
        let mut shapes = HashMap::new();
        shapes.insert("layer.0.q_proj".into(), vec![4096, 4096]);
        shapes.insert("layer.1.q_proj".into(), vec![4096, 2048]);
        let s = check_shape_consistency(&shapes);
        assert!(s.is_error());
    }

    #[test]
    fn test_weight_values_clean() {
        let s = check_weight_values(&[1.0, -0.5, 0.0, 2.3]);
        assert!(s.is_healthy());
    }

    #[test]
    fn test_weight_values_nan() {
        let s = check_weight_values(&[1.0, f32::NAN, 0.0]);
        assert!(s.is_error());
    }

    #[test]
    fn test_weight_values_inf() {
        let s = check_weight_values(&[1.0, f32::INFINITY]);
        assert!(matches!(s, HealthStatus::Warning(_)));
    }

    #[test]
    fn test_vocab_llama_ok() {
        let s = check_vocab_size(32_000, "llama");
        assert!(s.is_healthy());
    }

    #[test]
    fn test_vocab_too_small() {
        let s = check_vocab_size(100, "llama");
        assert!(matches!(s, HealthStatus::Warning(_)));
    }

    #[test]
    fn test_file_size_ok() {
        // 2B params * 2 bytes = 4GB
        let s = check_file_size(4_000_000_000, 2000, 16);
        assert!(s.is_healthy());
    }

    #[test]
    fn test_file_size_wrong() {
        let s = check_file_size(100, 2000, 16);
        assert!(s.is_error());
    }

    #[test]
    fn test_file_size_zero_params() {
        let s = check_file_size(100, 0, 16);
        assert!(matches!(s, HealthStatus::Warning(_)));
    }
}
