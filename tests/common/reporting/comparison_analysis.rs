//! Comparison analysis reports for cross-implementation validation
//!
//! This module provides comprehensive reporting capabilities for analyzing
//! cross-implementation comparison results, including accuracy analysis,
//! performance comparisons, regression detection, and executive summaries.

use crate::cross_validation::{
    CrossValidationResult, TokenMismatch,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Configuration for comparison analysis reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonAnalysisConfig {
    pub output_dir: PathBuf,
    pub include_detailed_mismatches: bool,
    pub include_performance_charts: bool,
    pub include_regression_analysis: bool,
    pub max_mismatch_details: usize,
    pub executive_summary: bool,
    pub trend_analysis_days: u32,
}

impl Default for ComparisonAnalysisConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("test-reports/comparison-analysis"),
            include_detailed_mismatches: true,
            include_performance_charts: true,
            include_regression_analysis: true,
            max_mismatch_details: 50,
            executive_summary: true,
            trend_analysis_days: 30,
        }
    }
}

/// Detailed accuracy analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyAnalysisReport {
    pub overall_accuracy: f64,
    pub test_case_accuracies: Vec<TestCaseAccuracy>,
    pub mismatch_analysis: MismatchAnalysis,
    pub accuracy_distribution: AccuracyDistribution,
    pub recommendations: Vec<String>,
}

/// Accuracy information for a single test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCaseAccuracy {
    pub test_name: String,
    pub token_accuracy: f64,
    pub total_tokens: usize,
    pub matching_tokens: usize,
    pub first_mismatch_position: Option<usize>,
    pub severity: AccuracySeverity,
    pub category: String,
}

/// Severity classification for accuracy issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccuracySeverity {
    Critical, // < 50% accuracy
    High,     // 50-80% accuracy
    Medium,   // 80-95% accuracy
    Low,      // 95-99% accuracy
    Minimal,  // > 99% accuracy
}
/// Analysis of token mismatches across all test cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MismatchAnalysis {
    pub total_mismatches: usize,
    pub mismatch_patterns: Vec<MismatchPattern>,
    pub common_mismatch_positions: Vec<usize>,
    pub mismatch_categories: HashMap<String, usize>,
    pub detailed_mismatches: Vec<DetailedMismatch>,
}

/// Pattern analysis for token mismatches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MismatchPattern {
    pub pattern_type: String,
    pub frequency: usize,
    pub examples: Vec<String>,
    pub impact_score: f64,
}

/// Detailed information about a specific mismatch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedMismatch {
    pub test_case: String,
    pub position: usize,
    pub rust_token: u32,
    pub cpp_token: u32,
    pub rust_text: Option<String>,
    pub cpp_text: Option<String>,
    pub context: String,
    pub frequency: usize,
    pub impact: MismatchImpact,
}

/// Impact assessment for a mismatch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MismatchImpact {
    pub semantic_impact: f64,
    pub downstream_effects: usize,
    pub user_visible: bool,
}

/// Distribution analysis of accuracy scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyDistribution {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub percentiles: HashMap<u8, f64>, // 25th, 50th, 75th, 90th, 95th, 99th
    pub histogram: Vec<AccuracyBucket>,
}

/// Histogram bucket for accuracy distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyBucket {
    pub range_start: f64,
    pub range_end: f64,
    pub count: usize,
    pub percentage: f64,
}

/// Performance comparison analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysisReport {
    pub overall_performance: PerformanceOverview,
    pub test_case_performance: Vec<TestCasePerformance>,
    pub performance_trends: PerformanceTrends,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub recommendations: Vec<String>,
}

/// Overall performance overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOverview {
    pub average_throughput_ratio: f64,
    pub average_memory_ratio: f64,
    pub performance_improvement: f64, // Positive means Rust is better
    pub memory_efficiency: f64,       // Positive means Rust uses less memory
    pub regression_count: usize,
    pub improvement_count: usize,
}

/// Performance information for a single test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCasePerformance {
    pub test_name: String,
    pub throughput_ratio: f64,
    pub memory_ratio: f64,
    pub rust_tokens_per_second: f64,
    pub cpp_tokens_per_second: f64,
    pub performance_category: PerformanceCategory,
    pub regression: bool,
}

/// Performance category classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceCategory {
    Excellent,  // > 2x improvement
    Good,       // 1.2-2x improvement
    Acceptable, // 0.8-1.2x (similar performance)
    Concerning, // 0.5-0.8x (some regression)
    Critical,   // < 0.5x (significant regression)
}
/// Performance trends over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub throughput_trend: TrendAnalysis,
    pub memory_trend: TrendAnalysis,
    pub regression_trend: TrendAnalysis,
    pub historical_data: Vec<HistoricalPerformance>,
}

/// Trend analysis for a specific metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub direction: TrendDirection,
    pub slope: f64,
    pub confidence: f64,
    pub significant: bool,
}

/// Direction of a trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

/// Historical performance data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalPerformance {
    pub timestamp: String,
    pub throughput_ratio: f64,
    pub memory_ratio: f64,
    pub test_count: usize,
    pub regression_count: usize,
}

/// Bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub primary_bottlenecks: Vec<Bottleneck>,
    pub performance_hotspots: Vec<PerformanceHotspot>,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Identified performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub component: String,
    pub impact_score: f64,
    pub affected_tests: Vec<String>,
    pub description: String,
}

/// Performance hotspot identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHotspot {
    pub operation: String,
    pub average_overhead: f64,
    pub frequency: usize,
    pub total_impact: f64,
}

/// Optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub area: String,
    pub potential_improvement: f64,
    pub effort_estimate: String,
    pub priority: OptimizationPriority,
}

/// Priority level for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationPriority {
    Critical,
    High,
    Medium,
    Low,
}
/// Regression analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysisReport {
    pub regression_summary: RegressionSummary,
    pub accuracy_regressions: Vec<AccuracyRegression>,
    pub performance_regressions: Vec<PerformanceRegression>,
    pub trend_analysis: RegressionTrendAnalysis,
    pub root_cause_analysis: Vec<RootCauseAnalysis>,
}

/// Summary of regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionSummary {
    pub total_regressions: usize,
    pub accuracy_regressions: usize,
    pub performance_regressions: usize,
    pub critical_regressions: usize,
    pub regression_rate: f64,
}

/// Accuracy regression details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyRegression {
    pub test_case: String,
    pub previous_accuracy: f64,
    pub current_accuracy: f64,
    pub regression_magnitude: f64,
    pub first_detected: String,
    pub severity: RegressionSeverity,
}

/// Performance regression details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    pub test_case: String,
    pub metric: String,
    pub previous_value: f64,
    pub current_value: f64,
    pub regression_percentage: f64,
    pub first_detected: String,
    pub severity: RegressionSeverity,
}

/// Severity of a regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// Regression trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTrendAnalysis {
    pub regression_frequency: TrendAnalysis,
    pub severity_trend: TrendAnalysis,
    pub recovery_time: Duration,
    pub patterns: Vec<RegressionPattern>,
}

/// Pattern in regression occurrences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionPattern {
    pub pattern_type: String,
    pub frequency: usize,
    pub typical_duration: Duration,
    pub common_causes: Vec<String>,
}

/// Root cause analysis for regressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootCauseAnalysis {
    pub regression_id: String,
    pub likely_causes: Vec<String>,
    pub confidence_scores: HashMap<String, f64>,
    pub recommended_actions: Vec<String>,
    pub investigation_notes: String,
}
/// Executive summary report for stakeholders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveSummaryReport {
    pub overall_status: OverallStatus,
    pub key_metrics: KeyMetrics,
    pub critical_issues: Vec<CriticalIssue>,
    pub achievements: Vec<Achievement>,
    pub recommendations: Vec<ExecutiveRecommendation>,
    pub risk_assessment: RiskAssessment,
}

/// Overall status of the comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallStatus {
    pub status: StatusLevel,
    pub confidence: f64,
    pub summary: String,
    pub last_updated: String,
}

/// Status level classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatusLevel {
    Excellent,
    Good,
    Acceptable,
    Concerning,
    Critical,
}

/// Key metrics for executive summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMetrics {
    pub accuracy_score: f64,
    pub performance_score: f64,
    pub reliability_score: f64,
    pub test_coverage: f64,
    pub regression_rate: f64,
}

/// Critical issue for executive attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalIssue {
    pub title: String,
    pub description: String,
    pub impact: ImpactLevel,
    pub urgency: UrgencyLevel,
    pub affected_areas: Vec<String>,
    pub recommended_action: String,
}

/// Impact level classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    High,
    Medium,
    Low,
}

/// Urgency level classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UrgencyLevel {
    Immediate,
    High,
    Medium,
    Low,
}

/// Achievement highlight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achievement {
    pub title: String,
    pub description: String,
    pub metric_improvement: Option<f64>,
    pub business_value: String,
}

/// Executive recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveRecommendation {
    pub title: String,
    pub description: String,
    pub priority: RecommendationPriority,
    pub effort_estimate: String,
    pub expected_benefit: String,
    pub timeline: String,
}

/// Priority level for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}
/// Risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub overall_risk: RiskLevel,
    pub risk_factors: Vec<RiskFactor>,
    pub mitigation_strategies: Vec<String>,
    pub monitoring_recommendations: Vec<String>,
}

/// Risk level classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    High,
    Medium,
    Low,
}

/// Individual risk factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub factor: String,
    pub probability: f64,
    pub impact: f64,
    pub risk_score: f64,
    pub mitigation: String,
}

/// Main comparison analysis reporter
pub struct ComparisonAnalysisReporter {
    config: ComparisonAnalysisConfig,
    historical_data: Vec<CrossValidationResult>,
}

impl ComparisonAnalysisReporter {
    /// Create a new comparison analysis reporter
    pub fn new(config: ComparisonAnalysisConfig) -> Self {
        Self {
            config,
            historical_data: Vec::new(),
        }
    }

    /// Load historical data for trend analysis
    pub fn load_historical_data(&mut self, data: Vec<CrossValidationResult>) {
        self.historical_data = data;
    }

    /// Generate comprehensive comparison analysis report
    pub async fn generate_analysis_report(
        &self,
        result: &CrossValidationResult,
    ) -> Result<AccuracyAnalysisReport, Box<dyn std::error::Error>> {
        // Create output directory
        std::fs::create_dir_all(&self.config.output_dir)?;

        // Generate individual reports
        let accuracy_report = self.generate_accuracy_analysis(result).await?;
        let performance_report = self.generate_performance_analysis(result).await?;
        let regression_report = self.generate_regression_analysis(result).await?;
        let executive_report = self.generate_executive_summary(result).await?;

        // Save individual reports
        self.save_accuracy_report(&accuracy_report).await?;
        self.save_performance_report(&performance_report).await?;
        self.save_regression_report(&regression_report).await?;
        self.save_executive_report(&executive_report).await?;

        Ok(accuracy_report)
    }

    /// Generate detailed accuracy analysis
    async fn generate_accuracy_analysis(
        &self,
        result: &CrossValidationResult,
    ) -> Result<AccuracyAnalysisReport, Box<dyn std::error::Error>> {
        let mut test_case_accuracies = Vec::new();
        let mut all_mismatches = Vec::new();
        let mut accuracy_scores = Vec::new();

        // Analyze each test case
        for test_result in &result.test_results {
            let accuracy = &test_result.accuracy_result;
            accuracy_scores.push(accuracy.token_accuracy);

            let severity = self.classify_accuracy_severity(accuracy.token_accuracy);
            let category = self.categorize_test_case(&test_result.test_case.name);

            test_case_accuracies.push(TestCaseAccuracy {
                test_name: test_result.test_case.name.clone(),
                token_accuracy: accuracy.token_accuracy,
                total_tokens: accuracy.total_tokens,
                matching_tokens: accuracy.matching_tokens,
                first_mismatch_position: accuracy.first_mismatch.as_ref().map(|m| m.position),
                severity,
                category,
            });

            // Collect mismatches for analysis
            if let Some(mismatch) = &accuracy.first_mismatch {
                all_mismatches.push(DetailedMismatch {
                    test_case: test_result.test_case.name.clone(),
                    position: mismatch.position,
                    rust_token: mismatch.rust_token,
                    cpp_token: mismatch.cpp_token,
                    rust_text: mismatch.rust_text.clone(),
                    cpp_text: mismatch.cpp_text.clone(),
                    context: self.format_mismatch_context(mismatch),
                    frequency: 1, // Will be updated in pattern analysis
                    impact: self.assess_mismatch_impact(mismatch),
                });
            }
        }

        let overall_accuracy = if !accuracy_scores.is_empty() {
            accuracy_scores.iter().sum::<f64>() / accuracy_scores.len() as f64
        } else {
            0.0
        };

        let mismatch_analysis = self.analyze_mismatches(&all_mismatches);
        let accuracy_distribution = self.calculate_accuracy_distribution(&accuracy_scores);
        let recommendations =
            self.generate_accuracy_recommendations(&test_case_accuracies, &mismatch_analysis);

        Ok(AccuracyAnalysisReport {
            overall_accuracy,
            test_case_accuracies,
            mismatch_analysis,
            accuracy_distribution,
            recommendations,
        })
    }
    /// Generate performance analysis
    async fn generate_performance_analysis(
        &self,
        result: &CrossValidationResult,
    ) -> Result<PerformanceAnalysisReport, Box<dyn std::error::Error>> {
        let mut test_case_performance = Vec::new();
        let mut throughput_ratios = Vec::new();
        let mut memory_ratios = Vec::new();
        let mut regression_count = 0;
        let mut improvement_count = 0;

        // Analyze each test case performance
        for test_result in &result.test_results {
            let perf = &test_result.performance_comparison;
            throughput_ratios.push(perf.throughput_ratio);
            memory_ratios.push(perf.memory_ratio);

            let category = self.classify_performance_category(perf.throughput_ratio);
            let regression = perf.performance_regression;

            if regression {
                regression_count += 1;
            } else if perf.throughput_ratio < 1.0 {
                improvement_count += 1;
            }

            test_case_performance.push(TestCasePerformance {
                test_name: test_result.test_case.name.clone(),
                throughput_ratio: perf.throughput_ratio,
                memory_ratio: perf.memory_ratio,
                rust_tokens_per_second: perf.rust_tokens_per_second,
                cpp_tokens_per_second: perf.cpp_tokens_per_second,
                performance_category: category,
                regression,
            });
        }

        let overall_performance = PerformanceOverview {
            average_throughput_ratio: throughput_ratios.iter().sum::<f64>()
                / throughput_ratios.len() as f64,
            average_memory_ratio: memory_ratios.iter().sum::<f64>() / memory_ratios.len() as f64,
            performance_improvement: self.calculate_performance_improvement(&throughput_ratios),
            memory_efficiency: self.calculate_memory_efficiency(&memory_ratios),
            regression_count,
            improvement_count,
        };

        let performance_trends = self.analyze_performance_trends(result);
        let bottleneck_analysis = self.analyze_bottlenecks(&test_case_performance);
        let recommendations =
            self.generate_performance_recommendations(&overall_performance, &bottleneck_analysis);

        Ok(PerformanceAnalysisReport {
            overall_performance,
            test_case_performance,
            performance_trends,
            bottleneck_analysis,
            recommendations,
        })
    }

    /// Generate regression analysis
    async fn generate_regression_analysis(
        &self,
        result: &CrossValidationResult,
    ) -> Result<RegressionAnalysisReport, Box<dyn std::error::Error>> {
        let mut accuracy_regressions = Vec::new();
        let mut performance_regressions = Vec::new();

        // Compare with historical data if available
        if let Some(previous_result) = self.get_previous_result(&result.model_name) {
            // Analyze accuracy regressions
            for (current, previous) in result
                .test_results
                .iter()
                .zip(previous_result.test_results.iter())
            {
                if current.accuracy_result.token_accuracy < previous.accuracy_result.token_accuracy
                {
                    let regression_magnitude = previous.accuracy_result.token_accuracy
                        - current.accuracy_result.token_accuracy;
                    let severity = self.classify_regression_severity(regression_magnitude);

                    accuracy_regressions.push(AccuracyRegression {
                        test_case: current.test_case.name.clone(),
                        previous_accuracy: previous.accuracy_result.token_accuracy,
                        current_accuracy: current.accuracy_result.token_accuracy,
                        regression_magnitude,
                        first_detected: result.timestamp.clone(),
                        severity,
                    });
                }

                // Analyze performance regressions
                if current.performance_comparison.performance_regression {
                    let regression_percentage =
                        ((current.performance_comparison.throughput_ratio - 1.0) * 100.0).abs();
                    let severity = self.classify_regression_severity(regression_percentage / 100.0);

                    performance_regressions.push(PerformanceRegression {
                        test_case: current.test_case.name.clone(),
                        metric: "throughput".to_string(),
                        previous_value: previous.performance_comparison.throughput_ratio,
                        current_value: current.performance_comparison.throughput_ratio,
                        regression_percentage,
                        first_detected: result.timestamp.clone(),
                        severity,
                    });
                }
            }
        }

        let regression_summary = RegressionSummary {
            total_regressions: accuracy_regressions.len() + performance_regressions.len(),
            accuracy_regressions: accuracy_regressions.len(),
            performance_regressions: performance_regressions.len(),
            critical_regressions: accuracy_regressions
                .iter()
                .filter(|r| matches!(r.severity, RegressionSeverity::Critical))
                .count()
                + performance_regressions
                    .iter()
                    .filter(|r| matches!(r.severity, RegressionSeverity::Critical))
                    .count(),
            regression_rate: self.calculate_regression_rate(
                &accuracy_regressions,
                &performance_regressions,
                result.test_results.len(),
            ),
        };

        let trend_analysis = self.analyze_regression_trends();
        let root_cause_analysis =
            self.perform_root_cause_analysis(&accuracy_regressions, &performance_regressions);

        Ok(RegressionAnalysisReport {
            regression_summary,
            accuracy_regressions,
            performance_regressions,
            trend_analysis,
            root_cause_analysis,
        })
    }

    /// Generate executive summary
    async fn generate_executive_summary(
        &self,
        result: &CrossValidationResult,
    ) -> Result<ExecutiveSummaryReport, Box<dyn std::error::Error>> {
        let overall_accuracy = result.summary.average_token_accuracy;
        let overall_performance = result.summary.average_throughput_ratio;

        let status = self.determine_overall_status(overall_accuracy, overall_performance);
        let key_metrics = self.calculate_key_metrics(result);
        let critical_issues = self.identify_critical_issues(result);
        let achievements = self.identify_achievements(result);
        let recommendations = self.generate_executive_recommendations(result);
        let risk_assessment = self.assess_risks(result);

        Ok(ExecutiveSummaryReport {
            overall_status: OverallStatus {
                status,
                confidence: self.calculate_confidence_score(result),
                summary: self.generate_status_summary(result),
                last_updated: result.timestamp.clone(),
            },
            key_metrics,
            critical_issues,
            achievements,
            recommendations,
            risk_assessment,
        })
    }
    // Helper methods for analysis
    fn classify_accuracy_severity(&self, accuracy: f64) -> AccuracySeverity {
        match accuracy {
            a if a < 0.5 => AccuracySeverity::Critical,
            a if a < 0.8 => AccuracySeverity::High,
            a if a < 0.95 => AccuracySeverity::Medium,
            a if a < 0.99 => AccuracySeverity::Low,
            _ => AccuracySeverity::Minimal,
        }
    }

    fn categorize_test_case(&self, test_name: &str) -> String {
        // Categorize based on test name patterns
        if test_name.contains("performance") || test_name.contains("long") {
            "Performance".to_string()
        } else if test_name.contains("edge") || test_name.contains("special") {
            "Edge Case".to_string()
        } else if test_name.contains("basic") || test_name.contains("simple") {
            "Basic Functionality".to_string()
        } else {
            "General".to_string()
        }
    }

    fn format_mismatch_context(&self, mismatch: &TokenMismatch) -> String {
        let before = mismatch
            .context_before
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let after = mismatch
            .context_after
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ");

        format!(
            "...{} [{}|{}] {}...",
            before, mismatch.rust_token, mismatch.cpp_token, after
        )
    }

    fn assess_mismatch_impact(&self, _mismatch: &TokenMismatch) -> MismatchImpact {
        // Simplified impact assessment
        MismatchImpact {
            semantic_impact: 0.5, // Would need more sophisticated analysis
            downstream_effects: 1,
            user_visible: true,
        }
    }

    fn analyze_mismatches(&self, mismatches: &[DetailedMismatch]) -> MismatchAnalysis {
        let mut mismatch_categories = HashMap::new();
        let mut patterns = Vec::new();

        // Analyze patterns in mismatches
        for mismatch in mismatches {
            let category = if mismatch.rust_text.is_some() && mismatch.cpp_text.is_some() {
                "Text Difference".to_string()
            } else {
                "Token Difference".to_string()
            };
            *mismatch_categories.entry(category).or_insert(0) += 1;
        }

        // Create pattern analysis
        for (category, count) in &mismatch_categories {
            patterns.push(MismatchPattern {
                pattern_type: category.clone(),
                frequency: *count,
                examples: mismatches
                    .iter()
                    .filter(|m| self.categorize_mismatch(m) == *category)
                    .take(3)
                    .map(|m| m.context.clone())
                    .collect(),
                impact_score: (*count as f64) / (mismatches.len() as f64),
            });
        }

        MismatchAnalysis {
            total_mismatches: mismatches.len(),
            mismatch_patterns: patterns,
            common_mismatch_positions: self.find_common_positions(mismatches),
            mismatch_categories,
            detailed_mismatches: mismatches.to_vec(),
        }
    }

    fn categorize_mismatch(&self, mismatch: &DetailedMismatch) -> String {
        if mismatch.rust_text.is_some() && mismatch.cpp_text.is_some() {
            "Text Difference".to_string()
        } else {
            "Token Difference".to_string()
        }
    }

    fn find_common_positions(&self, mismatches: &[DetailedMismatch]) -> Vec<usize> {
        let mut position_counts = HashMap::new();
        for mismatch in mismatches {
            *position_counts.entry(mismatch.position).or_insert(0) += 1;
        }

        let mut positions: Vec<_> = position_counts.into_iter().collect();
        positions.sort_by(|a, b| b.1.cmp(&a.1));
        positions.into_iter().take(10).map(|(pos, _)| pos).collect()
    }

    fn calculate_accuracy_distribution(&self, scores: &[f64]) -> AccuracyDistribution {
        if scores.is_empty() {
            return AccuracyDistribution {
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                percentiles: HashMap::new(),
                histogram: Vec::new(),
            };
        }

        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let median = sorted_scores[scores.len() / 2];

        let variance = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std_dev = variance.sqrt();

        let mut percentiles = HashMap::new();
        for &p in &[25, 50, 75, 90, 95, 99] {
            let index = ((p as f64 / 100.0) * (scores.len() - 1) as f64) as usize;
            percentiles.insert(p, sorted_scores[index]);
        }

        let histogram = self.create_accuracy_histogram(&sorted_scores);

        AccuracyDistribution {
            mean,
            median,
            std_dev,
            percentiles,
            histogram,
        }
    }

    fn create_accuracy_histogram(&self, scores: &[f64]) -> Vec<AccuracyBucket> {
        let bucket_count = 10;
        let mut buckets = vec![0; bucket_count];

        for &score in scores {
            let bucket_index = ((score * bucket_count as f64) as usize).min(bucket_count - 1);
            buckets[bucket_index] += 1;
        }

        buckets
            .into_iter()
            .enumerate()
            .map(|(i, count)| {
                let range_start = i as f64 / bucket_count as f64;
                let range_end = (i + 1) as f64 / bucket_count as f64;
                let percentage = (count as f64 / scores.len() as f64) * 100.0;

                AccuracyBucket {
                    range_start,
                    range_end,
                    count,
                    percentage,
                }
            })
            .collect()
    }

    fn generate_accuracy_recommendations(
        &self,
        accuracies: &[TestCaseAccuracy],
        analysis: &MismatchAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for low accuracy tests
        let low_accuracy_count = accuracies
            .iter()
            .filter(|a| {
                matches!(
                    a.severity,
                    AccuracySeverity::Critical | AccuracySeverity::High
                )
            })
            .count();

        if low_accuracy_count > 0 {
            recommendations.push(format!(
                "Address {} test cases with critical or high accuracy issues",
                low_accuracy_count
            ));
        }

        // Check for common mismatch patterns
        if analysis.total_mismatches > 0 {
            recommendations.push(
                "Investigate common mismatch patterns to identify systematic issues".to_string(),
            );
        }

        // Check for specific categories with issues
        for accuracy in accuracies {
            if matches!(accuracy.severity, AccuracySeverity::Critical) {
                recommendations.push(format!(
                    "Critical accuracy issue in {} category - immediate attention required",
                    accuracy.category
                ));
            }
        }

        recommendations
    }
    fn classify_performance_category(&self, throughput_ratio: f64) -> PerformanceCategory {
        match throughput_ratio {
            r if r < 0.5 => PerformanceCategory::Excellent,
            r if r < 0.8 => PerformanceCategory::Good,
            r if r < 1.2 => PerformanceCategory::Acceptable,
            r if r < 2.0 => PerformanceCategory::Concerning,
            _ => PerformanceCategory::Critical,
        }
    }

    fn calculate_performance_improvement(&self, ratios: &[f64]) -> f64 {
        let avg_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
        (1.0 - avg_ratio) * 100.0 // Positive means improvement
    }

    fn calculate_memory_efficiency(&self, ratios: &[f64]) -> f64 {
        let avg_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
        (1.0 - avg_ratio) * 100.0 // Positive means less memory usage
    }

    fn analyze_performance_trends(&self, _result: &CrossValidationResult) -> PerformanceTrends {
        // Simplified implementation - would need historical data
        PerformanceTrends {
            throughput_trend: TrendAnalysis {
                direction: TrendDirection::Stable,
                slope: 0.0,
                confidence: 0.5,
                significant: false,
            },
            memory_trend: TrendAnalysis {
                direction: TrendDirection::Stable,
                slope: 0.0,
                confidence: 0.5,
                significant: false,
            },
            regression_trend: TrendAnalysis {
                direction: TrendDirection::Stable,
                slope: 0.0,
                confidence: 0.5,
                significant: false,
            },
            historical_data: Vec::new(),
        }
    }

    fn analyze_bottlenecks(&self, performance: &[TestCasePerformance]) -> BottleneckAnalysis {
        let mut bottlenecks = Vec::new();
        let hotspots = Vec::new();
        let mut opportunities = Vec::new();

        // Identify primary bottlenecks
        let slow_tests: Vec<_> = performance
            .iter()
            .filter(|p| p.throughput_ratio > 1.5)
            .collect();

        if !slow_tests.is_empty() {
            bottlenecks.push(Bottleneck {
                component: "Inference Engine".to_string(),
                impact_score: slow_tests.len() as f64 / performance.len() as f64,
                affected_tests: slow_tests.iter().map(|t| t.test_name.clone()).collect(),
                description: "Multiple test cases showing performance regression".to_string(),
            });
        }

        // Identify optimization opportunities
        if performance.iter().any(|p| p.throughput_ratio > 2.0) {
            opportunities.push(OptimizationOpportunity {
                area: "Core Inference Loop".to_string(),
                potential_improvement: 50.0,
                effort_estimate: "Medium".to_string(),
                priority: OptimizationPriority::High,
            });
        }

        BottleneckAnalysis {
            primary_bottlenecks: bottlenecks,
            performance_hotspots: hotspots,
            optimization_opportunities: opportunities,
        }
    }

    fn generate_performance_recommendations(
        &self,
        overview: &PerformanceOverview,
        bottlenecks: &BottleneckAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if overview.regression_count > 0 {
            recommendations.push(format!(
                "Address {} performance regressions identified in test suite",
                overview.regression_count
            ));
        }

        if overview.average_throughput_ratio > 1.5 {
            recommendations.push(
                "Overall performance is concerning - investigate core bottlenecks".to_string(),
            );
        }

        for bottleneck in &bottlenecks.primary_bottlenecks {
            recommendations.push(format!(
                "Optimize {} component affecting {} tests",
                bottleneck.component,
                bottleneck.affected_tests.len()
            ));
        }

        recommendations
    }

    fn get_previous_result(&self, model_name: &str) -> Option<&CrossValidationResult> {
        self.historical_data
            .iter()
            .filter(|r| r.model_name == model_name)
            .max_by_key(|r| &r.timestamp)
    }

    fn classify_regression_severity(&self, magnitude: f64) -> RegressionSeverity {
        match magnitude {
            m if m > 0.2 => RegressionSeverity::Critical,
            m if m > 0.1 => RegressionSeverity::High,
            m if m > 0.05 => RegressionSeverity::Medium,
            _ => RegressionSeverity::Low,
        }
    }

    fn calculate_regression_rate(
        &self,
        accuracy_regressions: &[AccuracyRegression],
        performance_regressions: &[PerformanceRegression],
        total_tests: usize,
    ) -> f64 {
        let total_regressions = accuracy_regressions.len() + performance_regressions.len();
        (total_regressions as f64 / total_tests as f64) * 100.0
    }

    fn analyze_regression_trends(&self) -> RegressionTrendAnalysis {
        // Simplified implementation
        RegressionTrendAnalysis {
            regression_frequency: TrendAnalysis {
                direction: TrendDirection::Stable,
                slope: 0.0,
                confidence: 0.5,
                significant: false,
            },
            severity_trend: TrendAnalysis {
                direction: TrendDirection::Stable,
                slope: 0.0,
                confidence: 0.5,
                significant: false,
            },
            recovery_time: Duration::from_secs(24 * 3600), // 24 hours
            patterns: Vec::new(),
        }
    }

    fn perform_root_cause_analysis(
        &self,
        _accuracy_regressions: &[AccuracyRegression],
        _performance_regressions: &[PerformanceRegression],
    ) -> Vec<RootCauseAnalysis> {
        // Simplified implementation
        Vec::new()
    }
    fn determine_overall_status(&self, accuracy: f64, performance: f64) -> StatusLevel {
        match (accuracy, performance) {
            (a, p) if a > 0.95 && p < 1.2 => StatusLevel::Excellent,
            (a, p) if a > 0.90 && p < 1.5 => StatusLevel::Good,
            (a, p) if a > 0.80 && p < 2.0 => StatusLevel::Acceptable,
            (a, p) if a > 0.70 || p < 3.0 => StatusLevel::Concerning,
            _ => StatusLevel::Critical,
        }
    }

    fn calculate_key_metrics(&self, result: &CrossValidationResult) -> KeyMetrics {
        KeyMetrics {
            accuracy_score: result.summary.average_token_accuracy * 100.0,
            performance_score: (2.0 - result.summary.average_throughput_ratio.min(2.0)) * 50.0,
            reliability_score: (result.summary.successful_tests as f64
                / result.summary.total_tests as f64)
                * 100.0,
            test_coverage: 100.0, // Assuming full coverage
            regression_rate: ((result.summary.failed_tests as f64
                / result.summary.total_tests as f64)
                * 100.0),
        }
    }

    fn identify_critical_issues(&self, result: &CrossValidationResult) -> Vec<CriticalIssue> {
        let mut issues = Vec::new();

        if result.summary.average_token_accuracy < 0.8 {
            issues.push(CriticalIssue {
                title: "Low Overall Accuracy".to_string(),
                description: format!(
                    "Average token accuracy is {:.2}%, below acceptable threshold",
                    result.summary.average_token_accuracy * 100.0
                ),
                impact: ImpactLevel::High,
                urgency: UrgencyLevel::High,
                affected_areas: vec![
                    "Core Inference".to_string(),
                    "Model Compatibility".to_string(),
                ],
                recommended_action: "Investigate tokenization and inference differences"
                    .to_string(),
            });
        }

        if result.summary.average_throughput_ratio > 2.0 {
            issues.push(CriticalIssue {
                title: "Significant Performance Regression".to_string(),
                description: format!(
                    "Average performance is {:.2}x slower than C++ baseline",
                    result.summary.average_throughput_ratio
                ),
                impact: ImpactLevel::High,
                urgency: UrgencyLevel::Medium,
                affected_areas: vec!["Performance".to_string(), "User Experience".to_string()],
                recommended_action: "Profile and optimize critical performance paths".to_string(),
            });
        }

        issues
    }

    fn identify_achievements(&self, result: &CrossValidationResult) -> Vec<Achievement> {
        let mut achievements = Vec::new();

        if result.summary.average_token_accuracy > 0.95 {
            achievements.push(Achievement {
                title: "High Accuracy Achievement".to_string(),
                description: format!(
                    "Achieved {:.2}% average token accuracy",
                    result.summary.average_token_accuracy * 100.0
                ),
                metric_improvement: Some(result.summary.average_token_accuracy * 100.0),
                business_value: "Ensures reliable model outputs for production use".to_string(),
            });
        }

        if result.summary.average_throughput_ratio < 1.0 {
            achievements.push(Achievement {
                title: "Performance Improvement".to_string(),
                description: format!(
                    "Rust implementation is {:.2}x faster than C++ baseline",
                    1.0 / result.summary.average_throughput_ratio
                ),
                metric_improvement: Some((1.0 - result.summary.average_throughput_ratio) * 100.0),
                business_value:
                    "Improved performance reduces infrastructure costs and improves user experience"
                        .to_string(),
            });
        }

        achievements
    }

    fn generate_executive_recommendations(
        &self,
        result: &CrossValidationResult,
    ) -> Vec<ExecutiveRecommendation> {
        let mut recommendations = Vec::new();

        if result.summary.failed_tests > 0 {
            recommendations.push(ExecutiveRecommendation {
                title: "Address Test Failures".to_string(),
                description: format!(
                    "Resolve {} failing test cases to improve system reliability",
                    result.summary.failed_tests
                ),
                priority: RecommendationPriority::High,
                effort_estimate: "2-3 weeks".to_string(),
                expected_benefit: "Improved system stability and user confidence".to_string(),
                timeline: "Next sprint".to_string(),
            });
        }

        if result.summary.average_throughput_ratio > 1.5 {
            recommendations.push(ExecutiveRecommendation {
                title: "Performance Optimization Initiative".to_string(),
                description: "Launch focused effort to optimize performance-critical components"
                    .to_string(),
                priority: RecommendationPriority::Medium,
                effort_estimate: "1-2 months".to_string(),
                expected_benefit: "Reduced infrastructure costs and improved user experience"
                    .to_string(),
                timeline: "Next quarter".to_string(),
            });
        }

        recommendations
    }

    fn assess_risks(&self, result: &CrossValidationResult) -> RiskAssessment {
        let mut risk_factors = Vec::new();

        if result.summary.average_token_accuracy < 0.9 {
            risk_factors.push(RiskFactor {
                factor: "Accuracy Risk".to_string(),
                probability: 0.8,
                impact: 0.9,
                risk_score: 0.72,
                mitigation: "Implement additional validation and testing".to_string(),
            });
        }

        if result.summary.average_throughput_ratio > 2.0 {
            risk_factors.push(RiskFactor {
                factor: "Performance Risk".to_string(),
                probability: 0.7,
                impact: 0.6,
                risk_score: 0.42,
                mitigation: "Invest in performance optimization".to_string(),
            });
        }

        let overall_risk = if risk_factors.iter().any(|r| r.risk_score > 0.7) {
            RiskLevel::High
        } else if risk_factors.iter().any(|r| r.risk_score > 0.4) {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        RiskAssessment {
            overall_risk,
            risk_factors,
            mitigation_strategies: vec![
                "Implement continuous monitoring of accuracy metrics".to_string(),
                "Establish performance benchmarking pipeline".to_string(),
                "Create automated regression detection".to_string(),
            ],
            monitoring_recommendations: vec![
                "Daily accuracy monitoring".to_string(),
                "Performance trend analysis".to_string(),
                "Automated alerting for regressions".to_string(),
            ],
        }
    }

    fn calculate_confidence_score(&self, result: &CrossValidationResult) -> f64 {
        let test_coverage = 1.0; // Assuming full coverage
        let result_consistency =
            result.summary.successful_tests as f64 / result.summary.total_tests as f64;
        let data_quality = if result.test_results.len() > 10 {
            1.0
        } else {
            0.8
        };

        (test_coverage + result_consistency + data_quality) / 3.0
    }

    fn generate_status_summary(&self, result: &CrossValidationResult) -> String {
        format!(
            "Cross-validation completed with {:.1}% accuracy and {:.2}x performance ratio across {} test cases",
            result.summary.average_token_accuracy * 100.0,
            result.summary.average_throughput_ratio,
            result.summary.total_tests
        )
    }

    // Save methods for individual reports
    async fn save_accuracy_report(
        &self,
        report: &AccuracyAnalysisReport,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = self.config.output_dir.join("accuracy_analysis.json");
        let json = serde_json::to_string_pretty(report)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    async fn save_performance_report(
        &self,
        report: &PerformanceAnalysisReport,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = self.config.output_dir.join("performance_analysis.json");
        let json = serde_json::to_string_pretty(report)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    async fn save_regression_report(
        &self,
        report: &RegressionAnalysisReport,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = self.config.output_dir.join("regression_analysis.json");
        let json = serde_json::to_string_pretty(report)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    async fn save_executive_report(
        &self,
        report: &ExecutiveSummaryReport,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = self.config.output_dir.join("executive_summary.json");
        let json = serde_json::to_string_pretty(report)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_severity_classification() {
        let reporter = ComparisonAnalysisReporter::new(ComparisonAnalysisConfig::default());

        assert!(matches!(
            reporter.classify_accuracy_severity(0.3),
            AccuracySeverity::Critical
        ));
        assert!(matches!(
            reporter.classify_accuracy_severity(0.7),
            AccuracySeverity::High
        ));
        assert!(matches!(
            reporter.classify_accuracy_severity(0.9),
            AccuracySeverity::Medium
        ));
        assert!(matches!(
            reporter.classify_accuracy_severity(0.97),
            AccuracySeverity::Low
        ));
        assert!(matches!(
            reporter.classify_accuracy_severity(0.995),
            AccuracySeverity::Minimal
        ));
    }

    #[test]
    fn test_performance_category_classification() {
        let reporter = ComparisonAnalysisReporter::new(ComparisonAnalysisConfig::default());

        assert!(matches!(
            reporter.classify_performance_category(0.3),
            PerformanceCategory::Excellent
        ));
        assert!(matches!(
            reporter.classify_performance_category(0.7),
            PerformanceCategory::Good
        ));
        assert!(matches!(
            reporter.classify_performance_category(1.0),
            PerformanceCategory::Acceptable
        ));
        assert!(matches!(
            reporter.classify_performance_category(1.5),
            PerformanceCategory::Concerning
        ));
        assert!(matches!(
            reporter.classify_performance_category(3.0),
            PerformanceCategory::Critical
        ));
    }

    #[test]
    fn test_test_case_categorization() {
        let reporter = ComparisonAnalysisReporter::new(ComparisonAnalysisConfig::default());

        assert_eq!(
            reporter.categorize_test_case("performance_test"),
            "Performance"
        );
        assert_eq!(reporter.categorize_test_case("edge_case_test"), "Edge Case");
        assert_eq!(
            reporter.categorize_test_case("basic_functionality"),
            "Basic Functionality"
        );
        assert_eq!(reporter.categorize_test_case("random_test"), "General");
    }

    #[test]
    fn test_regression_severity_classification() {
        let reporter = ComparisonAnalysisReporter::new(ComparisonAnalysisConfig::default());

        assert!(matches!(
            reporter.classify_regression_severity(0.3),
            RegressionSeverity::Critical
        ));
        assert!(matches!(
            reporter.classify_regression_severity(0.15),
            RegressionSeverity::High
        ));
        assert!(matches!(
            reporter.classify_regression_severity(0.08),
            RegressionSeverity::Medium
        ));
        assert!(matches!(
            reporter.classify_regression_severity(0.02),
            RegressionSeverity::Low
        ));
    }

    #[test]
    fn test_overall_status_determination() {
        let reporter = ComparisonAnalysisReporter::new(ComparisonAnalysisConfig::default());

        assert!(matches!(
            reporter.determine_overall_status(0.98, 1.1),
            StatusLevel::Excellent
        ));
        assert!(matches!(
            reporter.determine_overall_status(0.92, 1.3),
            StatusLevel::Good
        ));
        assert!(matches!(
            reporter.determine_overall_status(0.85, 1.8),
            StatusLevel::Acceptable
        ));
        assert!(matches!(
            reporter.determine_overall_status(0.75, 2.5),
            StatusLevel::Concerning
        ));
        assert!(matches!(
            reporter.determine_overall_status(0.60, 4.0),
            StatusLevel::Critical
        ));
    }

    #[test]
    fn test_performance_improvement_calculation() {
        let reporter = ComparisonAnalysisReporter::new(ComparisonAnalysisConfig::default());

        // Rust is 2x faster (ratio = 0.5)
        let improvement = reporter.calculate_performance_improvement(&[0.5]);
        assert_eq!(improvement, 50.0);

        // Rust is 2x slower (ratio = 2.0)
        let regression = reporter.calculate_performance_improvement(&[2.0]);
        assert_eq!(regression, -100.0);

        // Same performance (ratio = 1.0)
        let same = reporter.calculate_performance_improvement(&[1.0]);
        assert_eq!(same, 0.0);
    }

    #[test]
    fn test_accuracy_distribution_calculation() {
        let reporter = ComparisonAnalysisReporter::new(ComparisonAnalysisConfig::default());
        let scores = vec![0.8, 0.9, 0.95, 0.98, 1.0];

        let distribution = reporter.calculate_accuracy_distribution(&scores);

        assert_eq!(distribution.mean, 0.926);
        assert_eq!(distribution.median, 0.95);
        assert!(distribution.std_dev > 0.0);
        assert_eq!(distribution.histogram.len(), 10);
    }

    #[test]
    fn test_empty_accuracy_distribution() {
        let reporter = ComparisonAnalysisReporter::new(ComparisonAnalysisConfig::default());
        let scores = vec![];

        let distribution = reporter.calculate_accuracy_distribution(&scores);

        assert_eq!(distribution.mean, 0.0);
        assert_eq!(distribution.median, 0.0);
        assert_eq!(distribution.std_dev, 0.0);
        assert!(distribution.percentiles.is_empty());
        assert!(distribution.histogram.is_empty());
    }

    #[test]
    fn test_mismatch_context_formatting() {
        let reporter = ComparisonAnalysisReporter::new(ComparisonAnalysisConfig::default());
        let mismatch = TokenMismatch {
            position: 5,
            rust_token: 123,
            cpp_token: 456,
            rust_text: Some("hello".to_string()),
            cpp_text: Some("world".to_string()),
            context_before: vec![1, 2, 3],
            context_after: vec![7, 8, 9],
        };

        let context = reporter.format_mismatch_context(&mismatch);
        assert_eq!(context, "...1 2 3 [123|456] 7 8 9...");
    }

    #[test]
    fn test_config_default() {
        let config = ComparisonAnalysisConfig::default();

        assert_eq!(
            config.output_dir,
            PathBuf::from("test-reports/comparison-analysis")
        );
        assert!(config.include_detailed_mismatches);
        assert!(config.include_performance_charts);
        assert!(config.include_regression_analysis);
        assert_eq!(config.max_mismatch_details, 50);
        assert!(config.executive_summary);
        assert_eq!(config.trend_analysis_days, 30);
    }
}
