//! AC4 Cross-Validation Implementation
//!
//! Core implementation functions for cross-validation accuracy metrics
use anyhow::{Context, Result};
/// Result from BitNet.rs inference
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub logits: Vec<Vec<f32>>,
    pub probabilities: Vec<Vec<f32>>,
    pub duration_ms: u64,
}
/// Result from C++ reference implementation
#[derive(Debug, Clone)]
pub struct ReferenceResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub logits: Vec<Vec<f32>>,
    pub probabilities: Vec<Vec<f32>>,
    pub duration_ms: u64,
}
/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    pub reference_implementation: ReferenceImplementation,
    pub tolerance: f32,
    pub correlation_threshold: f32,
    pub test_cases: Vec<String>,
    pub deterministic: bool,
    pub seed: u64,
    pub validate_bit_exact: bool,
}
impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            reference_implementation: ReferenceImplementation::CppBitNet,
            tolerance: 1e-6,
            correlation_threshold: 0.999,
            test_cases: vec![],
            deterministic: true,
            seed: 42,
            validate_bit_exact: false,
        }
    }
}
/// Reference implementation type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReferenceImplementation {
    CppBitNet,
    GGML,
}
/// Validation comparison result
#[derive(Debug, Clone)]
pub struct ValidationComparison {
    pub token_accuracy: f32,
    pub logit_correlation: f32,
    pub mse: f32,
    pub perplexity: f32,
    pub reference_perplexity: f32,
    pub exact_token_matches: usize,
    pub total_tokens: usize,
}
/// Aggregated metrics across multiple test cases
#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    pub average_token_accuracy: f32,
    pub average_logit_correlation: f32,
    pub average_mse: f32,
    pub perplexity_degradation: f32,
    pub lookup_performance_metrics: Option<LookupPerformanceMetrics>,
}
/// Lookup performance metrics for table lookup quantization
#[derive(Debug, Clone)]
pub struct LookupPerformanceMetrics {
    pub average_lookup_time_ns: f64,
    pub cache_hit_rate: f32,
}
/// Results from xtask crossval command
#[derive(Debug, Clone)]
pub struct CrossvalResults {
    pub overall_pass_rate: f32,
    pub quantization_accuracy: QuantizationAccuracy,
    pub performance_correlation: f32,
    pub numerical_stability: NumericalStability,
}
/// Quantization accuracy breakdown
#[derive(Debug, Clone)]
pub struct QuantizationAccuracy {
    pub i2s: f32,
    pub tl1: f32,
    pub tl2: f32,
}
/// Numerical stability metrics
#[derive(Debug, Clone)]
pub struct NumericalStability {
    pub nan_count: usize,
    pub inf_count: usize,
}
/// Compare inference outputs using validation metrics
pub fn compare_inference_outputs(
    bitnet_result: &InferenceResult,
    cpp_result: &ReferenceResult,
    _config: &CrossValidationConfig,
) -> Result<ValidationComparison> {
    let total_tokens = bitnet_result.tokens.len().min(cpp_result.tokens.len());
    let exact_token_matches = bitnet_result
        .tokens
        .iter()
        .zip(cpp_result.tokens.iter())
        .take(total_tokens)
        .filter(|(a, b)| a == b)
        .count();
    let token_accuracy =
        if total_tokens > 0 { exact_token_matches as f32 / total_tokens as f32 } else { 0.0 };
    let logit_correlation = calculate_logit_correlation(&bitnet_result.logits, &cpp_result.logits)?;
    let mse = calculate_mse(&bitnet_result.logits, &cpp_result.logits)?;
    let perplexity = calculate_perplexity(&bitnet_result.probabilities)?;
    let reference_perplexity = calculate_perplexity(&cpp_result.probabilities)?;
    Ok(ValidationComparison {
        token_accuracy,
        logit_correlation,
        mse,
        perplexity,
        reference_perplexity,
        exact_token_matches,
        total_tokens,
    })
}
/// Calculate Pearson correlation coefficient for logits
fn calculate_logit_correlation(logits_a: &[Vec<f32>], logits_b: &[Vec<f32>]) -> Result<f32> {
    let flat_a: Vec<f32> = logits_a.iter().flatten().copied().collect();
    let flat_b: Vec<f32> = logits_b.iter().flatten().copied().collect();
    if flat_a.is_empty() || flat_b.is_empty() || flat_a.len() != flat_b.len() {
        anyhow::bail!("Logit size mismatch: {} vs {}", flat_a.len(), flat_b.len());
    }
    let n = flat_a.len() as f32;
    let mean_a: f32 = flat_a.iter().sum::<f32>() / n;
    let mean_b: f32 = flat_b.iter().sum::<f32>() / n;
    let mut numerator = 0.0_f32;
    let mut var_a = 0.0_f32;
    let mut var_b = 0.0_f32;
    for (a, b) in flat_a.iter().zip(flat_b.iter()) {
        let diff_a = a - mean_a;
        let diff_b = b - mean_b;
        numerator += diff_a * diff_b;
        var_a += diff_a * diff_a;
        var_b += diff_b * diff_b;
    }
    let correlation =
        if var_a > 0.0 && var_b > 0.0 { numerator / (var_a.sqrt() * var_b.sqrt()) } else { 0.0 };
    Ok(correlation)
}
/// Calculate mean squared error for logits
fn calculate_mse(logits_a: &[Vec<f32>], logits_b: &[Vec<f32>]) -> Result<f32> {
    let flat_a: Vec<f32> = logits_a.iter().flatten().copied().collect();
    let flat_b: Vec<f32> = logits_b.iter().flatten().copied().collect();
    if flat_a.len() != flat_b.len() {
        anyhow::bail!("Logit size mismatch for MSE: {} vs {}", flat_a.len(), flat_b.len());
    }
    if flat_a.is_empty() {
        return Ok(0.0);
    }
    let squared_diffs: f32 = flat_a.iter().zip(flat_b.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    Ok(squared_diffs / flat_a.len() as f32)
}
/// Calculate perplexity from probabilities
fn calculate_perplexity(probabilities: &[Vec<f32>]) -> Result<f32> {
    if probabilities.is_empty() {
        return Ok(f32::INFINITY);
    }
    let mut log_likelihood = 0.0_f32;
    let mut token_count = 0;
    for step_probs in probabilities {
        if let Some(&max_prob) = step_probs.iter().max_by(|a, b| a.partial_cmp(b).unwrap())
            && max_prob > 0.0
        {
            log_likelihood += max_prob.ln();
            token_count += 1;
        }
    }
    if token_count == 0 {
        return Ok(f32::INFINITY);
    }
    let perplexity = (-log_likelihood / token_count as f32).exp();
    Ok(perplexity)
}
/// Aggregate validation metrics from multiple test cases
pub fn aggregate_validation_metrics(
    results: &[(String, ValidationComparison)],
) -> Result<AggregatedMetrics> {
    if results.is_empty() {
        anyhow::bail!("Cannot aggregate empty validation results");
    }
    let n = results.len() as f32;
    let average_token_accuracy: f32 =
        results.iter().map(|(_, comparison)| comparison.token_accuracy).sum::<f32>() / n;
    let average_logit_correlation: f32 =
        results.iter().map(|(_, comparison)| comparison.logit_correlation).sum::<f32>() / n;
    let average_mse: f32 = results.iter().map(|(_, comparison)| comparison.mse).sum::<f32>() / n;
    let perplexity_degradation: f32 = results
        .iter()
        .map(|(_, comparison)| {
            if comparison.reference_perplexity > 0.0 && comparison.reference_perplexity.is_finite()
            {
                (comparison.reference_perplexity - comparison.perplexity).abs()
                    / comparison.reference_perplexity
            } else {
                0.0
            }
        })
        .sum::<f32>()
        / n;
    Ok(AggregatedMetrics {
        average_token_accuracy,
        average_logit_correlation,
        average_mse,
        perplexity_degradation,
        lookup_performance_metrics: None,
    })
}
/// Parse xtask crossval output
pub fn parse_crossval_output(output: &std::process::Output) -> Result<CrossvalResults> {
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("xtask crossval command failed: {}", stderr);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Ok(json_results) = parse_json_crossval_output(&stdout) {
        return Ok(json_results);
    }
    parse_text_crossval_output(&stdout)
}
/// Parse JSON-formatted crossval output
fn parse_json_crossval_output(stdout: &str) -> Result<CrossvalResults> {
    use serde_json::Value;
    let json_start = stdout.find('{').context("No JSON found in output")?;
    let json_end = stdout.rfind('}').context("No JSON end found in output")? + 1;
    let json_str = &stdout[json_start..json_end];
    let data: Value =
        serde_json::from_str(json_str).context("Failed to parse JSON from xtask output")?;
    let overall_pass_rate = data["overall_pass_rate"].as_f64().unwrap_or(0.0) as f32;
    let quantization_accuracy = QuantizationAccuracy {
        i2s: data["quantization_accuracy"]["i2s"].as_f64().unwrap_or(0.0) as f32,
        tl1: data["quantization_accuracy"]["tl1"].as_f64().unwrap_or(0.0) as f32,
        tl2: data["quantization_accuracy"]["tl2"].as_f64().unwrap_or(0.0) as f32,
    };
    let performance_correlation = data["performance_correlation"].as_f64().unwrap_or(0.0) as f32;
    let numerical_stability = NumericalStability {
        nan_count: data["numerical_stability"]["nan_count"].as_u64().unwrap_or(0) as usize,
        inf_count: data["numerical_stability"]["inf_count"].as_u64().unwrap_or(0) as usize,
    };
    Ok(CrossvalResults {
        overall_pass_rate,
        quantization_accuracy,
        performance_correlation,
        numerical_stability,
    })
}
/// Parse text-based crossval output (fallback)
fn parse_text_crossval_output(stdout: &str) -> Result<CrossvalResults> {
    let text = stdout.to_lowercase();
    let extract_float = |pattern: &str, normalize_percent: bool| -> f32 {
        if let Some(pos) = text.find(pattern) {
            let after = &text[pos + pattern.len()..];
            let trimmed =
                after.trim_start_matches(|c: char| c.is_whitespace() || c == ':' || c == '=');
            let num_str: String =
                trimmed.chars().take_while(|&c| c.is_numeric() || c == '.').collect();
            if let Ok(mut value) = num_str.parse::<f32>() {
                if normalize_percent && value > 1.0 {
                    value /= 100.0;
                }
                return value;
            }
        }
        0.0
    };
    let extract_usize = |pattern: &str| -> usize {
        if let Some(pos) = text.find(pattern) {
            let after = &text[pos + pattern.len()..];
            let trimmed =
                after.trim_start_matches(|c: char| c.is_whitespace() || c == ':' || c == '=');
            let num_str: String = trimmed.chars().take_while(|&c| c.is_numeric()).collect();
            if let Ok(value) = num_str.parse::<usize>() {
                return value;
            }
        }
        0
    };
    let overall_pass_rate =
        extract_float("overall pass rate", true).max(extract_float("overall_pass_rate", true));
    let i2s = extract_float("i2s accuracy", true).max(extract_float("i2s_accuracy", true));
    let tl1 = extract_float("tl1 accuracy", true).max(extract_float("tl1_accuracy", true));
    let tl2 = extract_float("tl2 accuracy", true).max(extract_float("tl2_accuracy", true));
    let performance_correlation = extract_float("performance correlation", true)
        .max(extract_float("performance_correlation", true));
    let nan_count = extract_usize("nan count").max(extract_usize("nan_count"));
    let inf_count = extract_usize("inf count").max(extract_usize("inf_count"));
    Ok(CrossvalResults {
        overall_pass_rate,
        quantization_accuracy: QuantizationAccuracy { i2s, tl1, tl2 },
        performance_correlation,
        numerical_stability: NumericalStability { nan_count, inf_count },
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_calculate_mse_basic() {
        let logits_a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let logits_b = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let mse = calculate_mse(&logits_a, &logits_b).unwrap();
        assert_eq!(mse, 0.0);
    }
    #[test]
    fn test_calculate_mse_different() {
        let logits_a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let logits_b = vec![vec![2.0, 3.0], vec![4.0, 5.0]];
        let mse = calculate_mse(&logits_a, &logits_b).unwrap();
        assert_eq!(mse, 1.0);
    }
    #[test]
    fn test_calculate_logit_correlation_perfect() {
        let logits_a = vec![vec![1.0, 2.0, 3.0]];
        let logits_b = vec![vec![1.0, 2.0, 3.0]];
        let correlation = calculate_logit_correlation(&logits_a, &logits_b).unwrap();
        assert!((correlation - 1.0).abs() < 1e-5);
    }
    #[test]
    fn test_calculate_perplexity_simple() {
        let probs = vec![vec![0.8, 0.1, 0.1], vec![0.7, 0.2, 0.1]];
        let perplexity = calculate_perplexity(&probs).unwrap();
        assert!(perplexity > 0.0 && perplexity.is_finite());
    }
    #[test]
    fn test_aggregate_validation_metrics() {
        let results = vec![
            (
                "test1".to_string(),
                ValidationComparison {
                    token_accuracy: 0.99,
                    logit_correlation: 0.995,
                    mse: 1e-6,
                    perplexity: 10.0,
                    reference_perplexity: 10.5,
                    exact_token_matches: 99,
                    total_tokens: 100,
                },
            ),
            (
                "test2".to_string(),
                ValidationComparison {
                    token_accuracy: 0.98,
                    logit_correlation: 0.990,
                    mse: 2e-6,
                    perplexity: 11.0,
                    reference_perplexity: 11.5,
                    exact_token_matches: 98,
                    total_tokens: 100,
                },
            ),
        ];
        let aggregated = aggregate_validation_metrics(&results).unwrap();
        assert!((aggregated.average_token_accuracy - 0.985).abs() < 0.01);
        assert!(aggregated.average_mse > 0.0);
    }
    #[test]
    fn test_parse_json_crossval_output() {
        let json_output = r#"{
            "overall_pass_rate": 0.99,
            "quantization_accuracy": {
                "i2s": 0.995,
                "tl1": 0.992,
                "tl2": 0.993
            },
            "performance_correlation": 0.98,
            "numerical_stability": {
                "nan_count": 0,
                "inf_count": 0
            }
        }"#;
        let results = parse_json_crossval_output(json_output).unwrap();
        assert_eq!(results.overall_pass_rate, 0.99);
        assert_eq!(results.quantization_accuracy.i2s, 0.995);
        assert_eq!(results.numerical_stability.nan_count, 0);
    }
    #[test]
    fn test_parse_text_crossval_output() {
        let text_output = r"
            Cross-validation results:
            overall pass rate: 99.5%
            i2s accuracy: 99.8%
            tl1 accuracy: 99.2%
            tl2 accuracy: 99.3%
            performance correlation: 97.5%
            nan count: 0
            inf count: 0
        ";
        let results = parse_text_crossval_output(text_output).unwrap();
        assert!((results.overall_pass_rate - 0.995).abs() < 0.01);
        assert!((results.quantization_accuracy.i2s - 0.998).abs() < 0.01);
    }
}
