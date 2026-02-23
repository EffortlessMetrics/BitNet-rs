//! Core scenario policy contracts and defaults for test configuration orchestration.
//!
//! This crate owns the stable data model used by scenario-policy resolver crates.
//! Behavioral merging and override logic is intentionally left to higher-level crates.

pub use bitnet_testing_profile::{
    ActiveContext, ExecutionEnvironment as EnvironmentType, TestingScenario,
};

/// Canonical alias maintained for readability in test-facing code.
pub type ScenarioType = TestingScenario;

use std::path::PathBuf;
use std::time::Duration;

/// Canonical scenario context coming from active environment detection.
#[derive(Debug, Clone)]
pub struct ConfigurationContext {
    pub scenario: TestingScenario,
    pub environment: EnvironmentType,
    pub resource_constraints: Option<ResourceConstraints>,
    pub time_constraints: Option<TimeConstraints>,
    pub quality_requirements: Option<QualityRequirements>,
    pub platform_settings: Option<PlatformSettings>,
}

impl Default for ConfigurationContext {
    fn default() -> Self {
        Self {
            scenario: TestingScenario::Unit,
            environment: EnvironmentType::Local,
            resource_constraints: None,
            time_constraints: None,
            quality_requirements: None,
            platform_settings: None,
        }
    }
}

/// Resource constraints used by legacy scenario merges.
#[derive(Debug, Clone, Default)]
pub struct ResourceConstraints {
    pub max_memory_mb: Option<usize>,
    pub max_cpu_cores: Option<usize>,
    pub max_disk_gb: Option<usize>,
}

/// Time constraints used by legacy scenario merges.
#[derive(Debug, Clone, Default)]
pub struct TimeConstraints {
    pub max_total_duration: Option<Duration>,
    pub max_test_duration: Option<Duration>,
}

/// Quality requirements used by legacy scenario merges.
#[derive(Debug, Clone, Default)]
pub struct QualityRequirements {
    pub min_coverage: Option<f64>,
    pub max_flakiness: Option<f64>,
    pub required_passes: Option<usize>,
}

/// Platform constraints used by legacy scenario merges.
#[derive(Debug, Clone, Default)]
pub struct PlatformSettings {
    pub os: Option<String>,
    pub arch: Option<String>,
    pub features: Vec<String>,
}

/// Lightweight representation of reporting configuration.
#[derive(Debug, Clone)]
pub struct ReportingProfile {
    pub output_dir: PathBuf,
    pub formats: Vec<ReportFormat>,
    pub include_artifacts: bool,
    pub generate_coverage: bool,
    pub generate_performance: bool,
    pub upload_reports: bool,
}

/// Canonical report format model used by policies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReportFormat {
    Html,
    Json,
    Junit,
    Markdown,
    Csv,
}

#[derive(Debug, Clone)]
pub struct FixtureProfile {
    pub auto_download: bool,
    pub max_cache_size: u64,
    pub cleanup_interval: Duration,
    pub download_timeout: Duration,
    pub base_url: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ComparisonToleranceProfile {
    pub min_token_accuracy: f64,
    pub max_probability_divergence: f64,
    pub max_performance_regression: f64,
    pub numerical_tolerance: f64,
}

#[derive(Debug, Clone)]
pub struct CrossValidationProfile {
    pub enabled: bool,
    pub tolerance: ComparisonToleranceProfile,
    pub cpp_binary_path: Option<PathBuf>,
    pub test_cases: Vec<String>,
    pub performance_comparison: bool,
    pub accuracy_comparison: bool,
}

/// Policy-level configuration that can be converted into test config objects.
#[derive(Debug, Clone)]
pub struct TestConfigProfile {
    pub max_parallel_tests: usize,
    pub test_timeout: Duration,
    pub cache_dir: PathBuf,
    pub log_level: String,
    pub coverage_threshold: f64,
    pub fixtures: FixtureProfile,
    pub crossval: CrossValidationProfile,
    pub reporting: ReportingProfile,
}

impl Default for ReportingProfile {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("test-reports"),
            formats: vec![ReportFormat::Html, ReportFormat::Json],
            include_artifacts: true,
            generate_coverage: true,
            generate_performance: true,
            upload_reports: false,
        }
    }
}

impl Default for FixtureProfile {
    fn default() -> Self {
        Self {
            auto_download: true,
            max_cache_size: 10 * 1024 * 1024 * 1024,
            cleanup_interval: Duration::from_secs(24 * 60 * 60),
            download_timeout: Duration::from_secs(300),
            base_url: None,
        }
    }
}

impl Default for ComparisonToleranceProfile {
    fn default() -> Self {
        Self {
            min_token_accuracy: 0.999999,
            max_probability_divergence: 1e-6,
            max_performance_regression: 0.1,
            numerical_tolerance: 1e-6,
        }
    }
}

impl Default for CrossValidationProfile {
    fn default() -> Self {
        Self {
            enabled: false,
            tolerance: ComparisonToleranceProfile::default(),
            cpp_binary_path: None,
            test_cases: vec![
                "basic_inference".to_string(),
                "tokenization".to_string(),
                "model_loading".to_string(),
            ],
            performance_comparison: true,
            accuracy_comparison: true,
        }
    }
}

impl Default for TestConfigProfile {
    fn default() -> Self {
        Self {
            max_parallel_tests: get_optimal_parallel_tests(),
            test_timeout: Duration::from_secs(300),
            cache_dir: PathBuf::from("tests/cache"),
            log_level: "info".to_string(),
            coverage_threshold: 0.9,
            fixtures: FixtureProfile::default(),
            crossval: CrossValidationProfile::default(),
            reporting: ReportingProfile::default(),
        }
    }
}

fn get_optimal_parallel_tests() -> usize {
    let cores = num_cpus::get();
    if is_ci_context() { (cores / 2).max(1) } else { cores.max(1) }
}

fn is_ci_context() -> bool {
    std::env::var("CI").is_ok()
        || std::env::var("GITHUB_ACTIONS").is_ok()
        || std::env::var("GITLAB_CI").is_ok()
        || std::env::var("JENKINS_URL").is_ok()
}
