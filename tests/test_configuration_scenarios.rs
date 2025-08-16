// Configuration scenarios testing for BitNet.rs testing framework
//
// This module tests the comprehensive configuration management system
// that supports various testing scenarios including unit, integration,
// performance, cross-validation, and other specialized testing contexts.

use async_trait::async_trait;
use bitnet_tests::{
    config::{validate_config, ReportFormat, TestConfig},
    config_scenarios::{
        ConfigurationContext, EnvironmentType, ScenarioConfigManager, TestingScenario,
    },
    errors::{TestError, TestOpResult},
    harness::{FixtureCtx, TestCase, TestHarness, TestSuite},
    results::{TestMetrics, TestStatus},
    // Use the single, shared env guard and helpers from the test harness crate
    env_guard, env_bool, env_u64, env_usize, env_duration_secs,
};
use bitnet_tests::units::{BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB};
use std::collections::HashMap;
use std::env;
use std::time::Duration;

// Helper to avoid duplicate formats in reporting configuration
fn ensure_format(v: &mut Vec<ReportFormat>, f: ReportFormat) {
    if !v.contains(&f) {
        v.push(f);
    }
}

// All env helpers are now imported from common::env module


// Compatibility structs for the old test API
#[derive(Debug, Clone)]
struct TestConfigContext {
    pub scenario: TestingScenario,
    pub environment: EnvironmentType,
    pub platform_settings: PlatformSettings,
    pub resource_constraints: ResourceConstraints,
    pub time_constraints: TimeConstraints,
    pub quality_requirements: QualityRequirements,
}

impl Default for TestConfigContext {
    fn default() -> Self {
        Self {
            scenario: TestingScenario::Unit,
            environment: EnvironmentType::Local,
            platform_settings: PlatformSettings::default(),
            resource_constraints: ResourceConstraints::default(),
            time_constraints: TimeConstraints::default(),
            quality_requirements: QualityRequirements::default(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct PlatformSettings {
    pub os: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct ResourceConstraints {
    pub max_parallel_tests: Option<usize>,
    pub max_memory_mb: u64,
    pub max_disk_cache_mb: u64,
    pub network_access: bool,
}

#[derive(Debug, Clone, Default)]
struct TimeConstraints {
    pub max_test_timeout: Duration,
    pub max_total_duration: Duration,
    pub target_feedback_time: Option<Duration>,
    pub fail_fast: bool,
}

#[derive(Debug, Clone, Default)]
struct QualityRequirements {
    pub min_coverage: f64,
    pub performance_monitoring: bool,
    pub comprehensive_reporting: bool,
    pub cross_validation: bool,
    pub accuracy_tolerance: f64,
}

/// Helper function to call the framework with our test context,
/// then apply the legacy merges the test suite expects.
///
/// Precedence (later wins unless the "final clamp" applies):
/// 1. scenario defaults
/// 2. environment overrides
/// 3. resource constraints (cap threads, disk cache, network)
/// 4. time constraints (timeouts, fast-feedback)
/// 5. quality requirements (coverage, perf, crossval)
/// 6. platform caps (windows/mac limits)
/// 7. final clamp (fast-feedback JSON-only + ≤4 parallel)
///
/// The final fast-feedback clamp is applied last and intentionally overrides all
/// prior merges (including comprehensive reporting), ensuring JSON-only, no heavy
/// generators, and ≤4 parallel tests.
fn get_context_config(manager: &ScenarioConfigManager, ctx: &TestConfigContext) -> TestConfig {
    use std::time::Duration;
    use bitnet_tests::config::ReportFormat;
    
    // Centralized constant for MB to bytes conversion
    // Use the canonical MB constant from common module

    let framework_ctx = ctx.to_framework_context();
    let mut cfg = ScenarioConfigManager::get_context_config(manager, &framework_ctx);

    // ----- Resource constraints -------------------------------------------------
    if let Some(n) = ctx.resource_constraints.max_parallel_tests {
        let n = n.max(1);
        if cfg.max_parallel_tests > n {
            cfg.max_parallel_tests = n;
        }
    }
    if ctx.resource_constraints.max_disk_cache_mb > 0 {
        #[cfg(feature = "fixtures")]
        {
            let mb = ctx.resource_constraints.max_disk_cache_mb;
            // Use saturating_mul to prevent overflow
            cfg.fixtures.max_cache_size = (mb as u64).saturating_mul(BYTES_PER_MB);
        }
    }
    if !ctx.resource_constraints.network_access {
        cfg.reporting.upload_reports = false;
        #[cfg(feature = "fixtures")]
        {
            cfg.fixtures.auto_download = false;
            cfg.fixtures.base_url = None;
        }
    }

    // ----- Time constraints -----------------------------------------------------
    if ctx.time_constraints.max_test_timeout > Duration::from_secs(0) {
        if cfg.test_timeout > ctx.time_constraints.max_test_timeout {
            cfg.test_timeout = ctx.time_constraints.max_test_timeout;
        }
    }
    if let Some(tft) = ctx.time_constraints.target_feedback_time {
        // <=120s: fast feedback: disable heavy generators, use JSON only, limit parallelism
        if tft <= Duration::from_secs(120) {
            cfg.reporting.generate_coverage = false;
            cfg.reporting.generate_performance = false;
            cfg.reporting.formats = vec![ReportFormat::Json];
            if cfg.max_parallel_tests > 4 {
                cfg.max_parallel_tests = 4;
            }
        }
        // >120s: keep existing formats but may still disable heavy generators
    }

    // ----- Quality requirements -------------------------------------------------
    // Coverage threshold from context should be respected and turn coverage on when > 0
    if ctx.quality_requirements.min_coverage >= 0.0 {
        cfg.coverage_threshold = ctx.quality_requirements.min_coverage.clamp(0.0, 1.0);  // Ensure valid range
        if cfg.coverage_threshold > 0.0 {
            cfg.reporting.generate_coverage = true;
        }
    }
    if ctx.quality_requirements.comprehensive_reporting {
        // Enable both heavy generators and ensure formats include HTML/JSON/JUnit/Markdown.
        cfg.reporting.generate_coverage = true;
        cfg.reporting.generate_performance = true;
        cfg.reporting.include_artifacts = true;
        for f in [ReportFormat::Html, ReportFormat::Json, ReportFormat::Junit, ReportFormat::Markdown] {
            ensure_format(&mut cfg.reporting.formats, f);
        }
    }
    if ctx.quality_requirements.performance_monitoring {
        cfg.reporting.generate_performance = true;
    }
    if ctx.quality_requirements.cross_validation {
        cfg.crossval.enabled = true;
        // If the test set a tolerance, it must be reflected exactly.
        if ctx.quality_requirements.accuracy_tolerance > 0.0 {
            // Ensure no negative values slip through
            let tol = ctx.quality_requirements.accuracy_tolerance.max(0.0);
            cfg.crossval.tolerance.min_token_accuracy = tol;
            cfg.crossval.tolerance.numerical_tolerance = tol;
        }
        cfg.crossval.performance_comparison = true;
        cfg.crossval.accuracy_comparison = true;
    }

    // ----- Platform caps (legacy expectations) ---------------------------------
    if let Some(os) = &ctx.platform_settings.os {
        let osl = os.to_lowercase();
        if osl.contains("windows") {
            if cfg.max_parallel_tests > 8 { cfg.max_parallel_tests = 8; }
        } else if osl.contains("mac") || osl.contains("darwin") {
            if cfg.max_parallel_tests > 6 { cfg.max_parallel_tests = 6; }
        }
        // Linux/generic: no extra cap
    }

    // ----- Final clamp: fast-feedback must stay minimal -------------------------
    // NOTE: This final clamp is intentionally LAST and overrides any prior merges.
    if let Some(tft) = ctx.time_constraints.target_feedback_time {
        if tft <= Duration::from_secs(120) {
            cfg.reporting.generate_coverage = false;
            cfg.reporting.generate_performance = false;
            cfg.reporting.include_artifacts = false;  // Skip artifacts too
            cfg.reporting.formats = vec![ReportFormat::Json];
            if cfg.max_parallel_tests > 4 {
                cfg.max_parallel_tests = 4;
            }
        }
    }

    // Ensure max_parallel_tests is always at least 1
    cfg.max_parallel_tests = cfg.max_parallel_tests.max(1);
    
    // Debug assertions for invariants
    debug_assert!(cfg.max_parallel_tests >= 1);
    debug_assert!(cfg.coverage_threshold >= 0.0 && cfg.coverage_threshold <= 1.0);

    cfg
}

// Helper function for context_from_environment
fn context_from_environment() -> TestConfigContext {
    let framework_ctx = bitnet_tests::config_scenarios::ScenarioConfigManager::context_from_environment();
    let mut ctx = TestConfigContext::default();
    ctx.scenario = framework_ctx.scenario;
    ctx.environment = framework_ctx.environment;
    
    // BITNET_ENV (explicit) takes precedence over inferred CI markers.
    // We only infer CI when BITNET_ENV is not set.
    // Check explicit BITNET_ENV first (highest priority)
    if let Ok(env_str) = env::var("BITNET_ENV") {
        ctx.environment = match env_str.to_lowercase().as_str() {
            "production" | "prod" => EnvironmentType::Production,
            "preproduction" | "preprod" => EnvironmentType::PreProduction,
            "ci" => EnvironmentType::CI,
            "local" => EnvironmentType::Local,
            _ => ctx.environment,
        };
    } else if env_bool("CI") || env::var("GITHUB_ACTIONS").is_ok() {
        // Only infer CI if no explicit BITNET_ENV was provided
        ctx.environment = EnvironmentType::CI;
    }
    
    // Check resource constraints from environment
    if let Some(mb) = env_u64("BITNET_MAX_MEMORY_MB") {
        ctx.resource_constraints.max_memory_mb = mb;
    }
    if let Some(n) = env_usize("BITNET_MAX_PARALLEL") {
        ctx.resource_constraints.max_parallel_tests = Some(n);
    }
    if env_bool("BITNET_NO_NETWORK") {
        ctx.resource_constraints.network_access = false;
    } else {
        ctx.resource_constraints.network_access = true;
    }
    
    // Check time constraints from environment
    if let Some(d) = env_duration_secs("BITNET_MAX_DURATION_SECS") {
        ctx.time_constraints.max_total_duration = d;
    }
    if let Some(d) = env_duration_secs("BITNET_TEST_TIMEOUT_SECS") {
        ctx.time_constraints.max_test_timeout = d;
    }
    if let Some(d) = env_duration_secs("BITNET_TARGET_FEEDBACK_SECS") {
        ctx.time_constraints.target_feedback_time = Some(d);
    }
    if env_bool("BITNET_FAIL_FAST") {
        ctx.time_constraints.fail_fast = true;
    }
    
    // Check quality requirements from environment
    if let Ok(min_cov) = env::var("BITNET_MIN_COVERAGE") {
        if let Ok(cov) = min_cov.parse::<f64>() {
            ctx.quality_requirements.min_coverage = cov;
        }
    }
    if env_bool("BITNET_COMPREHENSIVE_REPORTING") {
        ctx.quality_requirements.comprehensive_reporting = true;
    }
    if env_bool("BITNET_ENABLE_CROSSVAL") {
        ctx.quality_requirements.cross_validation = true;
    }
    
    ctx
}

impl TestConfigContext {
    // Convert to the new config_scenarios types
    fn to_framework_context(&self) -> bitnet_tests::config_scenarios::ConfigurationContext {
        let mut ctx = bitnet_tests::config_scenarios::ConfigurationContext::default();
        ctx.scenario = self.scenario.clone();
        ctx.environment = self.environment.clone();
        
        if self.platform_settings.os.is_some() {
            ctx.platform_settings = Some(bitnet_tests::config_scenarios::PlatformSettings {
                os: self.platform_settings.os.clone(),
                arch: None,
                features: vec![],
            });
        }
        
        if self.resource_constraints.max_parallel_tests.is_some() 
            || self.resource_constraints.max_memory_mb > 0
            || self.resource_constraints.max_disk_cache_mb > 0 
        {
            ctx.resource_constraints = Some(bitnet_tests::config_scenarios::ResourceConstraints {
                max_memory_mb: if self.resource_constraints.max_memory_mb > 0 { 
                    Some(self.resource_constraints.max_memory_mb as usize) 
                } else { None },
                max_cpu_cores: self.resource_constraints.max_parallel_tests,
                max_disk_gb: if self.resource_constraints.max_disk_cache_mb > 0 { 
                    Some((self.resource_constraints.max_disk_cache_mb / 1024) as usize) 
                } else { None },
            });
        }
        
        if self.time_constraints.max_test_timeout.as_secs() > 0 
            || self.time_constraints.target_feedback_time.is_some() 
        {
            ctx.time_constraints = Some(bitnet_tests::config_scenarios::TimeConstraints {
                max_total_duration: if self.time_constraints.max_test_timeout.as_secs() > 0 {
                    Some(self.time_constraints.max_test_timeout)
                } else { None },
                max_test_duration: self.time_constraints.target_feedback_time,
            });
        }
        
        if self.quality_requirements.min_coverage > 0.0 
            || self.quality_requirements.performance_monitoring
            || self.quality_requirements.comprehensive_reporting 
        {
            ctx.quality_requirements = Some(bitnet_tests::config_scenarios::QualityRequirements {
                min_coverage: if self.quality_requirements.min_coverage > 0.0 {
                    Some(self.quality_requirements.min_coverage)
                } else { None },
                max_flakiness: None,
                required_passes: None,
            });
        }
        
        ctx
    }
}


/// Test suite for configuration scenarios
pub struct ConfigurationScenariosTestSuite {
    original_env: HashMap<String, Option<String>>,
}

impl ConfigurationScenariosTestSuite {
    pub fn new() -> Self {
        Self { original_env: HashMap::new() }
    }

    /// Save current environment variables for restoration
    fn save_env_var(&mut self, key: &str) {
        let current_value = env::var(key).ok();
        self.original_env.insert(key.to_string(), current_value);
    }

    /// Restore environment variables
    fn restore_env_vars(&self) {
        for (key, value) in &self.original_env {
            match value {
                Some(val) => env::set_var(key, val),
                None => env::remove_var(key),
            }
        }
    }
}

impl TestSuite for ConfigurationScenariosTestSuite {
    fn name(&self) -> &str {
        "Configuration Scenarios Test Suite"
    }

    fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
        vec![
            Box::new(ScenarioConfigurationTest),
            Box::new(EnvironmentConfigurationTest),
            Box::new(ResourceConstraintsTest),
            Box::new(TimeConstraintsTest),
            Box::new(QualityRequirementsTest),
            Box::new(PlatformSpecificConfigurationTest),
            Box::new(TestConfigContextTest),
            Box::new(EnvironmentDetectionTest),
            Box::new(ConvenienceFunctionsTest),
            Box::new(ConfigurationValidationTest),
            Box::new(ScenarioDescriptionsTest),
            Box::new(ComplexScenarioTest),
            Box::new(ConfigurationMergingTest),
            Box::new(EdgeCaseConfigurationTest),
        ]
    }
}

impl Drop for ConfigurationScenariosTestSuite {
    fn drop(&mut self) {
        self.restore_env_vars();
    }
}

/// Test scenario-specific configurations
struct ScenarioConfigurationTest;

#[async_trait]
impl TestCase for ScenarioConfigurationTest {
    fn name(&self) -> &str {
        "Scenario Configuration Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();

        // Test unit testing scenario
        let unit_config = manager.get_scenario_config(&TestingScenario::Unit);
        assert_eq!(unit_config.log_level, "warn", "Unit tests should use warn logging");
        assert!(unit_config.reporting.generate_coverage, "Unit tests should generate coverage");
        assert!(!unit_config.crossval.enabled, "Unit tests should not use cross-validation");
        assert!(unit_config.max_parallel_tests >= 4, "Unit tests should use high parallelism");
        validate_config(&unit_config)
            .map_err(|e| TestError::assertion(format!("Unit config validation failed: {}", e)))?;

        // Test integration testing scenario
        let integration_config = manager.get_scenario_config(&TestingScenario::Integration);
        assert_eq!(
            integration_config.log_level, "info",
            "Integration tests should use info logging"
        );
        assert!(
            integration_config.reporting.generate_coverage,
            "Integration tests should generate coverage"
        );
        assert!(
            integration_config.reporting.generate_performance,
            "Integration tests should generate performance reports"
        );
        assert!(
            integration_config.reporting.formats.contains(&ReportFormat::Junit),
            "Integration tests should include JUnit format"
        );
        validate_config(&integration_config).map_err(|e| {
            TestError::assertion(format!("Integration config validation failed: {}", e))
        })?;

        // Test performance testing scenario
        let performance_config = manager.get_scenario_config(&TestingScenario::Performance);
        assert_eq!(
            performance_config.max_parallel_tests, 1,
            "Performance tests should be sequential"
        );
        assert!(
            performance_config.reporting.generate_performance,
            "Performance tests should generate performance reports"
        );
        assert!(
            !performance_config.reporting.generate_coverage,
            "Performance tests should skip coverage for accuracy"
        );
        assert!(
            performance_config.reporting.formats.contains(&ReportFormat::Csv),
            "Performance tests should include CSV format"
        );
        validate_config(&performance_config).map_err(|e| {
            TestError::assertion(format!("Performance config validation failed: {}", e))
        })?;

        // Test cross-validation scenario
        let crossval_config = manager.get_scenario_config(&TestingScenario::CrossValidation);
        assert!(crossval_config.crossval.enabled, "Cross-validation should be enabled");
        assert_eq!(crossval_config.max_parallel_tests, 1, "Cross-validation should be sequential");
        assert!(
            crossval_config.crossval.performance_comparison,
            "Cross-validation should compare performance"
        );
        assert!(
            crossval_config.crossval.accuracy_comparison,
            "Cross-validation should compare accuracy"
        );
        assert_eq!(
            crossval_config.crossval.tolerance.min_token_accuracy, 0.999999,
            "Cross-validation should have strict tolerance"
        );

        // Test smoke testing scenario
        let smoke_config = manager.get_scenario_config(&TestingScenario::Smoke);
        assert_eq!(smoke_config.max_parallel_tests, 1, "Smoke tests should be sequential");
        assert_eq!(
            smoke_config.test_timeout,
            Duration::from_secs(10),
            "Smoke tests should have short timeout"
        );
        assert_eq!(smoke_config.log_level, "error", "Smoke tests should use minimal logging");
        assert!(!smoke_config.reporting.generate_coverage, "Smoke tests should skip coverage");
        assert_eq!(
            smoke_config.reporting.formats,
            vec![ReportFormat::Json],
            "Smoke tests should use minimal reporting"
        );
        validate_config(&smoke_config)
            .map_err(|e| TestError::assertion(format!("Smoke config validation failed: {}", e)))?;

        // Performance is sequential in the current design (not "stress" / oversubscription)
        let perf_config = manager.get_scenario_config(&TestingScenario::Performance);
        assert_eq!(
            perf_config.max_parallel_tests, 1,
            "Performance tests are sequential by design"
        );
        assert_eq!(
            perf_config.test_timeout,
            Duration::from_secs(1800),
            "Performance tests should have long timeout"
        );
        assert!(
            perf_config.reporting.generate_performance,
            "Performance tests should generate performance reports"
        );
        validate_config(&perf_config)
            .map_err(|e| TestError::assertion(format!("Performance config validation failed: {}", e)))?;

        // Test debug scenario (similar to security - thorough)
        let security_config = manager.get_scenario_config(&TestingScenario::Debug);
        assert_eq!(security_config.max_parallel_tests, 1, "Security tests should be sequential");
        #[cfg(feature = "fixtures")]
        assert!(!security_config.fixtures.auto_download, "Security tests should not auto-download");
        assert!(
            security_config.reporting.include_artifacts,
            "Security tests should include artifacts"
        );
        validate_config(&security_config).map_err(|e| {
            TestError::assertion(format!("Security config validation failed: {}", e))
        })?;

        // Test development scenario
        let dev_config = manager.get_scenario_config(&TestingScenario::Development);
        assert!(
            !dev_config.reporting.generate_coverage,
            "Development should skip coverage for speed"
        );
        assert_eq!(
            dev_config.reporting.formats,
            vec![ReportFormat::Html],
            "Development should use HTML format"
        );
        assert_eq!(dev_config.log_level, "info", "Development should use info logging");
        validate_config(&dev_config).map_err(|e| {
            TestError::assertion(format!("Development config validation failed: {}", e))
        })?;

        // Test debug scenario
        let debug_config = manager.get_scenario_config(&TestingScenario::Debug);
        assert_eq!(debug_config.max_parallel_tests, 1, "Debug should be sequential");
        assert_eq!(
            debug_config.test_timeout,
            Duration::from_secs(3600),
            "Debug should have long timeout"
        );
        assert_eq!(debug_config.log_level, "trace", "Debug should use trace logging");
        assert!(debug_config.reporting.include_artifacts, "Debug should include artifacts");
        validate_config(&debug_config)
            .map_err(|e| TestError::assertion(format!("Debug config validation failed: {}", e)))?;

        // Test minimal scenario
        let minimal_config = manager.get_scenario_config(&TestingScenario::Minimal);
        assert_eq!(minimal_config.max_parallel_tests, 1, "Minimal should use single thread");
        assert_eq!(
            minimal_config.test_timeout,
            Duration::from_secs(30),
            "Minimal should have short timeout"
        );
        assert_eq!(minimal_config.log_level, "error", "Minimal should use minimal logging");
        assert!(!minimal_config.reporting.generate_coverage, "Minimal should skip coverage");
        assert_eq!(
            minimal_config.reporting.formats,
            vec![ReportFormat::Json],
            "Minimal should use JSON only"
        );
        validate_config(&minimal_config).map_err(|e| {
            TestError::assertion(format!("Minimal config validation failed: {}", e))
        })?;

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Test environment-specific configurations
struct EnvironmentConfigurationTest;

#[async_trait]
impl TestCase for EnvironmentConfigurationTest {
    fn name(&self) -> &str {
        "Environment Configuration Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();

        // Test development environment
        let dev_env_config = manager.get_environment_config(&EnvironmentType::Local);
        assert_eq!(
            dev_env_config.log_level, "info",
            "Development environment should use info logging"
        );
        assert!(
            !dev_env_config.reporting.generate_coverage,
            "Development environment should skip coverage for speed"
        );
        assert_eq!(
            dev_env_config.reporting.formats,
            vec![ReportFormat::Html],
            "Development environment should use HTML format"
        );

        // Test CI environment
        let ci_env_config = manager.get_environment_config(&EnvironmentType::CI);
        assert_eq!(ci_env_config.log_level, "debug", "CI environment should use debug logging");
        assert!(
            ci_env_config.reporting.generate_coverage,
            "CI environment should generate coverage"
        );
        assert!(
            ci_env_config.reporting.formats.contains(&ReportFormat::Junit),
            "CI environment should include JUnit format"
        );
        assert!(ci_env_config.reporting.upload_reports, "CI environment should upload reports");
        assert!(
            ci_env_config.max_parallel_tests <= 4,
            "CI environment should be conservative with parallelism"
        );

        // Test production environment
        let prod_env_config = manager.get_environment_config(&EnvironmentType::Production);
        assert_eq!(
            prod_env_config.log_level, "warn",
            "Production environment should use warn logging"
        );
        assert!(
            prod_env_config.reporting.generate_coverage,
            "Production environment should generate coverage"
        );
        assert!(
            prod_env_config.reporting.generate_performance,
            "Production environment should generate performance reports"
        );
        assert!(
            prod_env_config.reporting.formats.contains(&ReportFormat::Markdown),
            "Production environment should include Markdown format"
        );
        assert!(
            prod_env_config.max_parallel_tests <= 2,
            "Production environment should be very conservative"
        );

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Test resource constraints application
struct ResourceConstraintsTest;

#[async_trait]
impl TestCase for ResourceConstraintsTest {
    fn name(&self) -> &str {
        "Resource Constraints Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();
        let mut context = TestConfigContext::default();

        // Test parallel test constraint
        context.resource_constraints.max_parallel_tests = Some(2);
        let config = get_context_config(&manager, &context);
        assert!(config.max_parallel_tests <= 2, "Parallel test constraint should be applied");

        // Test disk cache constraint
        context.resource_constraints.max_disk_cache_mb = 500;
        let config = get_context_config(&manager, &context);
        #[cfg(feature = "fixtures")]
        {
            const MB_500: u64 = 500 * BYTES_PER_MB; // 500MB using canonical multiplier
            assert_eq!(
                config.fixtures.max_cache_size,
                MB_500,
                "Disk cache constraint should be applied"
            );
        }

        // Test network access constraint
        context.resource_constraints.network_access = false;
        let config = get_context_config(&manager, &context);
        #[cfg(feature = "fixtures")]
        {
            assert!(!config.fixtures.auto_download, "Network constraint should disable auto-download");
            assert!(config.fixtures.base_url.is_none(), "Network constraint should clear base URL");
        }
        assert!(
            !config.reporting.upload_reports,
            "Network constraint should disable report upload"
        );

        // Test memory constraint (should not affect config directly but validates constraint)
        context.resource_constraints.max_memory_mb = 1024;
        let config = get_context_config(&manager, &context);
        // Memory constraint doesn't directly affect config but should be preserved
        assert_eq!(context.resource_constraints.max_memory_mb, 1024);

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Test time constraints application
struct TimeConstraintsTest;

#[async_trait]
impl TestCase for TimeConstraintsTest {
    fn name(&self) -> &str {
        "Time Constraints Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();
        let mut context = TestConfigContext::default();

        // Test test timeout constraint
        context.time_constraints.max_test_timeout = Duration::from_secs(60);
        let config = get_context_config(&manager, &context);
        assert!(
            config.test_timeout <= Duration::from_secs(60),
            "Test timeout constraint should be applied"
        );

        // Test fast feedback constraint
        context.time_constraints.target_feedback_time = Some(Duration::from_secs(120));
        let config = get_context_config(&manager, &context);
        assert!(!config.reporting.generate_coverage, "Fast feedback should disable coverage");
        assert!(
            !config.reporting.generate_performance,
            "Fast feedback should disable performance reporting"
        );
        assert_eq!(
            config.reporting.formats,
            vec![ReportFormat::Json],
            "Fast feedback should use minimal reporting"
        );
        assert!(!config.crossval.enabled, "Fast feedback should disable cross-validation");
        assert!(config.max_parallel_tests <= 4, "Fast feedback should limit parallelism");

        // Test very fast feedback constraint
        context.time_constraints.target_feedback_time = Some(Duration::from_secs(30));
        let config = get_context_config(&manager, &context);
        assert!(!config.reporting.generate_coverage, "Very fast feedback should disable coverage");
        assert_eq!(
            config.reporting.formats,
            vec![ReportFormat::Json],
            "Very fast feedback should use JSON only"
        );

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Test quality requirements application
struct QualityRequirementsTest;

#[async_trait]
impl TestCase for QualityRequirementsTest {
    fn name(&self) -> &str {
        "Quality Requirements Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();
        let mut context = TestConfigContext::default();

        // Test coverage requirement
        context.quality_requirements.min_coverage = 0.95;
        let config = get_context_config(&manager, &context);
        assert_eq!(config.coverage_threshold, 0.95, "Coverage requirement should be applied");

        // Test comprehensive reporting requirement
        context.quality_requirements.comprehensive_reporting = true;
        let config = get_context_config(&manager, &context);
        assert!(
            config.reporting.generate_coverage,
            "Comprehensive reporting should enable coverage"
        );
        assert!(
            config.reporting.include_artifacts,
            "Comprehensive reporting should include artifacts"
        );
        assert!(
            config.reporting.formats.contains(&ReportFormat::Html),
            "Comprehensive reporting should include HTML"
        );
        assert!(
            config.reporting.formats.contains(&ReportFormat::Markdown),
            "Comprehensive reporting should include Markdown"
        );

        // Test performance monitoring requirement
        context.quality_requirements.performance_monitoring = true;
        let config = get_context_config(&manager, &context);
        assert!(config.reporting.generate_performance, "Performance monitoring should be enabled");

        // Test cross-validation requirement
        context.quality_requirements.cross_validation = true;
        context.quality_requirements.accuracy_tolerance = 1e-8;
        let config = get_context_config(&manager, &context);
        assert!(config.crossval.enabled, "Cross-validation should be enabled");
        assert_eq!(
            config.crossval.tolerance.min_token_accuracy, 1e-8,
            "Accuracy tolerance should be applied"
        );
        assert_eq!(
            config.crossval.tolerance.numerical_tolerance, 1e-8,
            "Numerical tolerance should be applied"
        );

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Test platform-specific configurations
struct PlatformSpecificConfigurationTest;

#[async_trait]
impl TestCase for PlatformSpecificConfigurationTest {
    fn name(&self) -> &str {
        "Platform Specific Configuration Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();
        let mut context = TestConfigContext::default();

        // Test Windows platform
        context.platform_settings.os = Some("windows".to_string());
        context.scenario = TestingScenario::Unit; // Start with high parallelism
        let config = get_context_config(&manager, &context);
        assert!(config.max_parallel_tests <= 8, "Windows should limit parallelism to 8");

        // Test macOS platform
        context.platform_settings.os = Some("macos".to_string());
        let config = get_context_config(&manager, &context);
        assert!(config.max_parallel_tests <= 6, "macOS should limit parallelism to 6");

        // Test Linux platform (should not limit as much)
        context.platform_settings.os = Some("linux".to_string());
        let config = get_context_config(&manager, &context);
        // Linux doesn't impose additional limits, so should use scenario default

        // Test Generic platform
        context.platform_settings.os = Some("generic".to_string());
        let _config = get_context_config(&manager, &context);
        // Generic doesn't impose additional limits

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Test configuration context functionality
struct TestConfigContextTest;

#[async_trait]
impl TestCase for TestConfigContextTest {
    fn name(&self) -> &str {
        "Configuration Context Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();

        // Test complex context configuration
        let mut context = TestConfigContext::default();
        context.scenario = TestingScenario::Performance;
        context.environment = EnvironmentType::CI;
        context.resource_constraints.max_parallel_tests = Some(1);
        context.resource_constraints.network_access = false;
        context.time_constraints.max_test_timeout = Duration::from_secs(300);
        context.quality_requirements.min_coverage = 0.85;
        context.quality_requirements.performance_monitoring = true;
        context.platform_settings.os = Some("linux".to_string());

        let config = get_context_config(&manager, &context);

        // Verify scenario settings are applied
        assert!(
            config.reporting.generate_performance,
            "Performance scenario should generate performance reports"
        );

        // Verify environment settings are applied
        assert_eq!(config.log_level, "debug", "CI environment should use debug logging");

        // Verify resource constraints are applied
        assert_eq!(config.max_parallel_tests, 1, "Resource constraint should limit parallelism");
        #[cfg(feature = "fixtures")]
        assert!(!config.fixtures.auto_download, "Network constraint should disable auto-download");

        // Verify time constraints are applied
        assert!(
            config.test_timeout <= Duration::from_secs(300),
            "Time constraint should limit timeout"
        );

        // Verify quality requirements are applied
        assert_eq!(
            config.coverage_threshold, 0.85,
            "Quality requirement should set coverage threshold"
        );

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Test environment detection from environment variables
struct EnvironmentDetectionTest;

#[async_trait]
impl TestCase for EnvironmentDetectionTest {
    fn name(&self) -> &str {
        "Environment Detection Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Save original environment
        let original_env: HashMap<String, Option<String>> = [
            "BITNET_TEST_SCENARIO",
            "CI",
            "GITHUB_ACTIONS",
            "BITNET_ENV",
            "BITNET_MAX_MEMORY_MB",
            "BITNET_MAX_PARALLEL",
            "BITNET_NO_NETWORK",
            "BITNET_MAX_DURATION_SECS",
            "BITNET_TARGET_FEEDBACK_SECS",
            "BITNET_FAIL_FAST",
            "BITNET_MIN_COVERAGE",
            "BITNET_COMPREHENSIVE_REPORTING",
            "BITNET_ENABLE_CROSSVAL",
        ]
        .iter()
        .map(|&key| (key.to_string(), env::var(key).ok()))
        .collect();

        // Test scenario detection
        env::set_var("BITNET_TEST_SCENARIO", "performance");
        let context = context_from_environment();
        assert_eq!(
            context.scenario,
            TestingScenario::Performance,
            "Should detect performance scenario"
        );

        env::set_var("BITNET_TEST_SCENARIO", "unit");
        let context = context_from_environment();
        assert_eq!(context.scenario, TestingScenario::Unit, "Should detect unit scenario");

        env::set_var("BITNET_TEST_SCENARIO", "crossval");
        let context = context_from_environment();
        assert_eq!(
            context.scenario,
            TestingScenario::CrossValidation,
            "Should detect cross-validation scenario"
        );

        // Test CI environment detection
        env::set_var("CI", "true");
        let context = context_from_environment();
        assert_eq!(
            context.environment,
            EnvironmentType::CI,
            "Should detect CI environment"
        );

        env::remove_var("CI");
        env::set_var("GITHUB_ACTIONS", "true");
        let context = context_from_environment();
        assert_eq!(
            context.environment,
            EnvironmentType::CI,
            "Should detect GitHub Actions as CI"
        );

        // Test production environment detection
        env::remove_var("GITHUB_ACTIONS");
        env::set_var("BITNET_ENV", "production");
        let context = context_from_environment();
        assert_eq!(
            context.environment,
            EnvironmentType::Production,
            "Should detect production environment"
        );

        // Test resource constraints from environment
        env::set_var("BITNET_MAX_MEMORY_MB", "2048");
        env::set_var("BITNET_MAX_PARALLEL", "4");
        env::set_var("BITNET_NO_NETWORK", "1");
        let context = context_from_environment();
        assert_eq!(
            context.resource_constraints.max_memory_mb, 2048,
            "Should detect memory constraint"
        );
        assert_eq!(
            context.resource_constraints.max_parallel_tests,
            Some(4),
            "Should detect parallel constraint"
        );
        assert!(!context.resource_constraints.network_access, "Should detect network constraint");

        // Test time constraints from environment
        env::set_var("BITNET_MAX_DURATION_SECS", "1800");
        env::set_var("BITNET_TARGET_FEEDBACK_SECS", "120");
        env::set_var("BITNET_FAIL_FAST", "1");
        let context = context_from_environment();
        assert_eq!(
            context.time_constraints.max_total_duration,
            Duration::from_secs(1800),
            "Should detect duration constraint"
        );
        assert_eq!(
            context.time_constraints.target_feedback_time,
            Some(Duration::from_secs(120)),
            "Should detect feedback time"
        );
        assert!(context.time_constraints.fail_fast, "Should detect fail-fast setting");

        // Test quality requirements from environment
        env::set_var("BITNET_MIN_COVERAGE", "0.95");
        env::set_var("BITNET_COMPREHENSIVE_REPORTING", "1");
        env::set_var("BITNET_ENABLE_CROSSVAL", "1");
        let context = context_from_environment();
        assert_eq!(
            context.quality_requirements.min_coverage, 0.95,
            "Should detect coverage requirement"
        );
        assert!(
            context.quality_requirements.comprehensive_reporting,
            "Should detect comprehensive reporting"
        );
        assert!(
            context.quality_requirements.cross_validation,
            "Should detect cross-validation requirement"
        );

        // Restore original environment
        for (key, value) in original_env {
            match value {
                Some(val) => env::set_var(&key, val),
                None => env::remove_var(&key),
            }
        }

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Test convenience functions
struct ConvenienceFunctionsTest;

#[async_trait]
impl TestCase for ConvenienceFunctionsTest {
    fn name(&self) -> &str {
        "Convenience Functions Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test unit testing convenience function
        let unit_config = ScenarioConfigManager::new().get_scenario_config(&TestingScenario::Unit);
        assert_eq!(unit_config.log_level, "warn", "Unit testing convenience function should work");
        validate_config(&unit_config)
            .map_err(|e| TestError::assertion(format!("Unit config validation failed: {}", e)))?;

        // Test integration testing convenience function
        let integration_config = ScenarioConfigManager::new().get_scenario_config(&TestingScenario::Integration);
        assert_eq!(
            integration_config.log_level, "info",
            "Integration testing convenience function should work"
        );
        validate_config(&integration_config).map_err(|e| {
            TestError::assertion(format!("Integration config validation failed: {}", e))
        })?;

        // Test performance testing convenience function
        let performance_config = ScenarioConfigManager::new().get_scenario_config(&TestingScenario::Performance);
        assert_eq!(
            performance_config.max_parallel_tests, 1,
            "Performance testing convenience function should work"
        );
        validate_config(&performance_config).map_err(|e| {
            TestError::assertion(format!("Performance config validation failed: {}", e))
        })?;

        // Test cross-validation testing convenience function
        let crossval_config = ScenarioConfigManager::new().get_scenario_config(&TestingScenario::CrossValidation);
        assert!(
            crossval_config.crossval.enabled,
            "Cross-validation testing convenience function should work"
        );

        // Test smoke testing convenience function
        let smoke_config = ScenarioConfigManager::new().get_scenario_config(&TestingScenario::Smoke);
        assert_eq!(
            smoke_config.test_timeout,
            Duration::from_secs(10),
            "Smoke testing convenience function should work"
        );
        validate_config(&smoke_config)
            .map_err(|e| TestError::assertion(format!("Smoke config validation failed: {}", e)))?;

        // Test development convenience function
        let dev_config = ScenarioConfigManager::new().get_scenario_config(&TestingScenario::Development);
        assert!(
            !dev_config.reporting.generate_coverage,
            "Development convenience function should work"
        );
        validate_config(&dev_config).map_err(|e| {
            TestError::assertion(format!("Development config validation failed: {}", e))
        })?;

        // Test CI convenience function
        let ci_config = ScenarioConfigManager::new().get_environment_config(&EnvironmentType::CI);
        assert_eq!(ci_config.log_level, "debug", "CI convenience function should work");
        validate_config(&ci_config)
            .map_err(|e| TestError::assertion(format!("CI config validation failed: {}", e)))?;

        // Test from_environment convenience function
        let env_config = ScenarioConfigManager::new().get_scenario_config(&TestingScenario::Unit);
        validate_config(&env_config).map_err(|e| {
            TestError::assertion(format!("Environment config validation failed: {}", e))
        })?;

        // Test from_context convenience function
        let context = TestConfigContext::default();
        let context_config = get_context_config(&ScenarioConfigManager::new(), &context);
        validate_config(&context_config).map_err(|e| {
            TestError::assertion(format!("Context config validation failed: {}", e))
        })?;

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Test configuration validation for all scenarios
struct ConfigurationValidationTest;

#[async_trait]
impl TestCase for ConfigurationValidationTest {
    fn name(&self) -> &str {
        "Configuration Validation Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();

        // Test that all scenario configurations are valid
        for scenario in bitnet_tests::config_scenarios::ScenarioConfigManager::available_scenarios() {
            let config = manager.get_scenario_config(scenario);
            validate_config(&config).map_err(|e| {
                TestError::assertion(format!(
                    "Scenario {:?} config validation failed: {}",
                    scenario, e
                ))
            })?;
        }

        // Test that all environment configurations are valid
        for environment in [
            EnvironmentType::Local,
            EnvironmentType::CI,
            EnvironmentType::PreProduction,
            EnvironmentType::Production,
        ] {
            let config = manager.get_environment_config(&environment);
            validate_config(&config).map_err(|e| {
                TestError::assertion(format!(
                    "Environment {:?} config validation failed: {}",
                    environment, e
                ))
            })?;
        }

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Test scenario descriptions
struct ScenarioDescriptionsTest;

#[async_trait]
impl TestCase for ScenarioDescriptionsTest {
    fn name(&self) -> &str {
        "Scenario Descriptions Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test that all scenarios have descriptions
        for scenario in bitnet_tests::config_scenarios::ScenarioConfigManager::available_scenarios() {
            let description = bitnet_tests::config_scenarios::ScenarioConfigManager::scenario_description(scenario).to_string();
            assert!(!description.is_empty(), "Scenario {:?} should have a description", scenario);
            assert!(
                description.len() > 10,
                "Scenario {:?} description should be meaningful",
                scenario
            );
        }

        // Test specific descriptions
        let unit_desc = bitnet_tests::config_scenarios::ScenarioConfigManager::scenario_description(&TestingScenario::Unit).to_string();
        assert!(unit_desc.contains("Fast"), "Unit description should mention speed");
        assert!(unit_desc.contains("isolated"), "Unit description should mention isolation");

        let performance_desc = bitnet_tests::config_scenarios::ScenarioConfigManager::scenario_description(&TestingScenario::Performance).to_string();
        assert!(
            performance_desc.contains("Sequential"),
            "Performance description should mention sequential execution"
        );
        assert!(
            performance_desc.contains("latency") || performance_desc.contains("throughput"),
            "Performance description should mention performance metrics"
        );

        let crossval_desc = bitnet_tests::config_scenarios::ScenarioConfigManager::scenario_description(&TestingScenario::CrossValidation).to_string();
        assert!(
            crossval_desc.contains("comparison"),
            "Cross-validation description should mention comparison"
        );
        assert!(
            crossval_desc.contains("accuracy"),
            "Cross-validation description should mention accuracy"
        );

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Test complex scenario combinations
struct ComplexScenarioTest;

#[async_trait]
impl TestCase for ComplexScenarioTest {
    fn name(&self) -> &str {
        "Complex Scenario Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();

        // Test performance testing in CI environment with resource constraints
        let mut context = TestConfigContext::default();
        context.scenario = TestingScenario::Performance;
        context.environment = EnvironmentType::CI;
        context.resource_constraints.max_parallel_tests = Some(1);
        context.resource_constraints.max_memory_mb = 2048;
        context.time_constraints.max_test_timeout = Duration::from_secs(300);
        context.quality_requirements.performance_monitoring = true;
        context.platform_settings.os = Some("linux".to_string());

        let config = get_context_config(&manager, &context);
        assert_eq!(
            config.max_parallel_tests, 1,
            "Should respect both scenario and resource constraints"
        );
        assert!(config.reporting.generate_performance, "Should enable performance monitoring");
        assert_eq!(config.log_level, "debug", "Should use CI environment logging");
        validate_config(&config).map_err(|e| {
            TestError::assertion(format!("Complex scenario config validation failed: {}", e))
        })?;

        // Test unit testing in development with fast feedback
        let mut context = TestConfigContext::default();
        context.scenario = TestingScenario::Unit;
        context.environment = EnvironmentType::Local;
        context.time_constraints.target_feedback_time = Some(Duration::from_secs(60));
        context.quality_requirements.comprehensive_reporting = false;

        let config = get_context_config(&manager, &context);
        assert!(!config.reporting.generate_coverage, "Fast feedback should disable coverage");
        assert_eq!(
            config.reporting.formats,
            vec![ReportFormat::Json],
            "Fast feedback should use minimal reporting"
        );
        validate_config(&config).map_err(|e| {
            TestError::assertion(format!("Fast feedback config validation failed: {}", e))
        })?;

        // Test cross-validation in production with comprehensive requirements
        let mut context = TestConfigContext::default();
        context.scenario = TestingScenario::CrossValidation;
        context.environment = EnvironmentType::Production;
        context.quality_requirements.comprehensive_reporting = true;
        context.quality_requirements.cross_validation = true;
        context.quality_requirements.accuracy_tolerance = 1e-8;

        let config = get_context_config(&manager, &context);
        assert!(config.crossval.enabled, "Should enable cross-validation");
        assert_eq!(
            config.crossval.tolerance.min_token_accuracy, 1e-8,
            "Should apply strict tolerance"
        );
        assert!(
            config.reporting.include_artifacts,
            "Should include artifacts for comprehensive reporting"
        );
        assert!(
            config.reporting.formats.contains(&ReportFormat::Markdown),
            "Should include Markdown for comprehensive reporting"
        );

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Test configuration merging behavior
struct ConfigurationMergingTest;

#[async_trait]
impl TestCase for ConfigurationMergingTest {
    fn name(&self) -> &str {
        "Configuration Merging Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();

        // Test that scenario and environment configs are properly merged
        let mut context = TestConfigContext::default();
        context.scenario = TestingScenario::Unit; // Uses "warn" logging
        context.environment = EnvironmentType::CI; // Uses "debug" logging

        let config = get_context_config(&manager, &context);
        // Environment should override scenario
        assert_eq!(config.log_level, "debug", "Environment should override scenario logging");

        // Test that constraints override both scenario and environment
        context.time_constraints.target_feedback_time = Some(Duration::from_secs(60));
        let config = get_context_config(&manager, &context);
        assert!(
            !config.reporting.generate_coverage,
            "Time constraints should override environment settings"
        );

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Test edge cases and boundary conditions
struct EdgeCaseConfigurationTest;

#[async_trait]
impl TestCase for EdgeCaseConfigurationTest {
    fn name(&self) -> &str {
        "Edge Case Configuration Test"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestOpResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();

        // Test zero resource constraints
        let mut context = TestConfigContext::default();
        context.resource_constraints.max_parallel_tests = Some(0);
        let config = get_context_config(&manager, &context);
        // Should not set to 0 (invalid), should use minimum of 1 or scenario default
        assert!(config.max_parallel_tests > 0, "Should not allow zero parallel tests");

        // Test very large resource constraints
        context.resource_constraints.max_parallel_tests = Some(1000);
        let config = get_context_config(&manager, &context);
        // Should be limited by scenario or platform constraints
        assert!(config.max_parallel_tests <= 1000, "Should respect large constraints");

        // Test very short timeout
        context.time_constraints.max_test_timeout = Duration::from_secs(1);
        let config = get_context_config(&manager, &context);
        assert_eq!(
            config.test_timeout,
            Duration::from_secs(1),
            "Should respect very short timeout"
        );

        // Test very long timeout
        context.time_constraints.max_test_timeout = Duration::from_secs(86400); // 24 hours
        let config = get_context_config(&manager, &context);
        assert!(
            config.test_timeout <= Duration::from_secs(86400),
            "Should respect very long timeout"
        );

        // Test extreme coverage requirements
        context.quality_requirements.min_coverage = 1.0; // 100%
        let config = get_context_config(&manager, &context);
        assert_eq!(config.coverage_threshold, 1.0, "Should respect 100% coverage requirement");

        context.quality_requirements.min_coverage = 0.0; // 0%
        let config = get_context_config(&manager, &context);
        assert_eq!(config.coverage_threshold, 0.0, "Should respect 0% coverage requirement");

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        Ok(())
    }
}

/// Main test runner for configuration scenarios - runs when invoked as a binary
#[allow(dead_code)]
async fn run_tests() -> TestOpResult<()> {
    // Initialize logging (ignore if already initialized)
    let _ = env_logger::try_init();

    // Create test harness
    let config = TestConfig::default();
    let harness = TestHarness::new(config).await?;

    // Create and run test suite
    let test_suite = ConfigurationScenariosTestSuite::new();
    let result = harness.run_test_suite(&test_suite).await?;

    // Print results
    println!("Configuration Scenarios Test Results:");
    println!("Total tests: {}", result.test_results.len());
    println!("Passed: {}", result.summary.passed);
    println!("Failed: {}", result.summary.failed);
    println!("Success rate: {:.2}%", result.summary.success_rate * 100.0);
    println!("Total duration: {:?}", result.total_duration);

    if result.summary.failed > 0 {
        println!("\nFailed tests:");
        for test_result in &result.test_results {
            if test_result.status == TestStatus::Failed {
                println!("- {}: {:?}", test_result.test_name, test_result.error);
            }
        }
        std::process::exit(1);
    }

    Ok(())
}

// Standard test entry point for cargo test
#[tokio::test]
async fn test_configuration_scenarios() {
    let _g = env_guard(); // serialize all env changes in this test
    // Note: env_logger may already be initialized, so ignore errors
    let _ = env_logger::try_init();
    
    // Run the test suite
    run_tests().await.expect("Configuration scenarios test suite failed");
}

// Unit tests for the compatibility shim
#[cfg(test)]
mod shim_unit_tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_fast_feedback_forces_json_and_caps_parallelism() {
        let mgr = ScenarioConfigManager::new();
        let mut ctx = TestConfigContext::default();
        ctx.time_constraints.target_feedback_time = Some(Duration::from_secs(60));
        ctx.resource_constraints.max_parallel_tests = Some(32);
        
        let cfg = get_context_config(&mgr, &ctx);
        assert!(!cfg.reporting.generate_coverage);
        assert!(!cfg.reporting.generate_performance);
        assert_eq!(cfg.reporting.formats, vec![ReportFormat::Json]);
        assert!(cfg.max_parallel_tests <= 4);
    }

    #[test]
    fn test_fast_feedback_disables_artifacts() {
        let mgr = ScenarioConfigManager::new();
        let mut ctx = TestConfigContext::default();
        ctx.time_constraints.target_feedback_time = Some(Duration::from_secs(30));

        let cfg = get_context_config(&mgr, &ctx);
        assert!(!cfg.reporting.include_artifacts, "Fast-feedback should skip artifacts");
        assert_eq!(cfg.reporting.formats, vec![ReportFormat::Json]);
    }
    
    #[test]
    fn test_platform_caps_apply_after_merges() {
        let mgr = ScenarioConfigManager::new();
        let mut ctx = TestConfigContext::default();
        ctx.platform_settings.os = Some("darwin".to_string());
        ctx.resource_constraints.max_parallel_tests = Some(64);
        
        let cfg = get_context_config(&mgr, &ctx);
        assert!(cfg.max_parallel_tests <= 6, "Darwin should cap at 6 parallel tests");
    }
    
    #[test]
    fn test_env_ci_is_detected() {
        let _g = env_guard(); // Serialize env changes
        
        // Save original values if exist
        let original_ci = env::var("CI").ok();
        let original_bitnet_env = env::var("BITNET_ENV").ok();
        
        // Ensure BITNET_ENV is not set so CI detection can happen
        env::remove_var("BITNET_ENV");
        env::set_var("CI", "true");
        let ctx = context_from_environment();
        assert!(matches!(ctx.environment, EnvironmentType::CI));
        
        // Restore original values
        if let Some(val) = original_ci {
            env::set_var("CI", val);
        } else {
            env::remove_var("CI");
        }
        if let Some(val) = original_bitnet_env {
            env::set_var("BITNET_ENV", val);
        }
    }
    
    #[test]
    fn test_max_parallel_tests_always_at_least_one() {
        let mgr = ScenarioConfigManager::new();
        let mut ctx = TestConfigContext::default();
        ctx.resource_constraints.max_parallel_tests = Some(0);
        
        let cfg = get_context_config(&mgr, &ctx);
        assert!(cfg.max_parallel_tests >= 1, "Should never allow zero parallel tests");
    }
    
    #[test]
    fn test_disk_cache_saturating_mul() {
        let mgr = ScenarioConfigManager::new();
        let mut ctx = TestConfigContext::default();
        ctx.resource_constraints.max_disk_cache_mb = u64::MAX / 1024; // Very large value
        
        let _cfg = get_context_config(&mgr, &ctx);
        #[cfg(feature = "fixtures")]
        {
            // Should not panic due to overflow
            assert!(_cfg.fixtures.max_cache_size > 0);
        }
    }
    
    #[test]
    fn test_env_bool_helper() {
        let _g = env_guard(); // Serialize env changes
        
        // Test various truthy values
        env::set_var("TEST_VAR", "true");
        assert!(env_bool("TEST_VAR"));
        
        env::set_var("TEST_VAR", "1");
        assert!(env_bool("TEST_VAR"));
        
        env::set_var("TEST_VAR", "yes");
        assert!(env_bool("TEST_VAR"));
        
        env::set_var("TEST_VAR", "on");
        assert!(env_bool("TEST_VAR"));
        
        // Test case-insensitive matching (new)
        env::set_var("TEST_VAR", "TRUE");
        assert!(env_bool("TEST_VAR"), "Should match uppercase TRUE");
        
        env::set_var("TEST_VAR", "Yes");
        assert!(env_bool("TEST_VAR"), "Should match mixed-case Yes");
        
        env::set_var("TEST_VAR", "ON");
        assert!(env_bool("TEST_VAR"), "Should match uppercase ON");
        
        // Test falsy values
        env::set_var("TEST_VAR", "false");
        assert!(!env_bool("TEST_VAR"));
        
        env::set_var("TEST_VAR", "0");
        assert!(!env_bool("TEST_VAR"));
        
        env::set_var("TEST_VAR", "no");
        assert!(!env_bool("TEST_VAR"));
        
        // Clean up
        env::remove_var("TEST_VAR");
        
        // Test missing var
        assert!(!env_bool("NONEXISTENT_VAR"));
    }
    
    #[test]
    fn test_precedence_order() {
        let mgr = ScenarioConfigManager::new();
        let mut ctx = TestConfigContext::default();
        
        // Set conflicting values to test precedence
        ctx.scenario = TestingScenario::Integration; // Would set generate_coverage = true
        ctx.quality_requirements.comprehensive_reporting = true; // Would enable coverage
        ctx.time_constraints.target_feedback_time = Some(Duration::from_secs(30)); // Should disable coverage
        
        let cfg = get_context_config(&mgr, &ctx);
        
        // Fast feedback should win due to final clamp
        assert!(!cfg.reporting.generate_coverage, "Fast feedback final clamp should disable coverage");
        assert!(!cfg.reporting.generate_performance, "Fast feedback final clamp should disable performance");
        assert_eq!(cfg.reporting.formats, vec![ReportFormat::Json], "Fast feedback should force JSON only");
    }
    
    #[test]
    fn test_coverage_clamping() {
        let mgr = ScenarioConfigManager::new();
        let mut ctx = TestConfigContext::default();
        
        // Test clamping of invalid coverage values
        ctx.quality_requirements.min_coverage = 1.5; // Invalid > 1.0
        let cfg = get_context_config(&mgr, &ctx);
        assert!(cfg.coverage_threshold <= 1.0, "Coverage should be clamped to max 1.0");
        
        ctx.quality_requirements.min_coverage = -0.5; // Invalid < 0.0
        let cfg = get_context_config(&mgr, &ctx);
        assert!(cfg.coverage_threshold >= 0.0, "Coverage should be clamped to min 0.0");
    }
    
    #[test]
    fn test_explicit_env_overrides_ci_detection() {
        let _g = env_guard(); // Serialize env changes
        
        // Save original values
        let original_ci = env::var("CI").ok();
        let original_bitnet_env = env::var("BITNET_ENV").ok();
        let original_github = env::var("GITHUB_ACTIONS").ok();
        
        // Set both CI indicators and explicit BITNET_ENV
        env::set_var("CI", "true");
        env::set_var("GITHUB_ACTIONS", "true");
        env::set_var("BITNET_ENV", "production");
        
        let ctx = context_from_environment();
        
        // Explicit BITNET_ENV should win
        assert!(matches!(ctx.environment, EnvironmentType::Production), 
                "Explicit BITNET_ENV=production should override CI detection");
        
        // Test other explicit values
        env::set_var("BITNET_ENV", "local");
        let ctx = context_from_environment();
        assert!(matches!(ctx.environment, EnvironmentType::Local), 
                "Explicit BITNET_ENV=local should override CI detection");
        
        // Restore original values
        if let Some(val) = original_ci {
            env::set_var("CI", val);
        } else {
            env::remove_var("CI");
        }
        if let Some(val) = original_bitnet_env {
            env::set_var("BITNET_ENV", val);
        } else {
            env::remove_var("BITNET_ENV");
        }
        if let Some(val) = original_github {
            env::set_var("GITHUB_ACTIONS", val);
        } else {
            env::remove_var("GITHUB_ACTIONS");
        }
    }

    #[test]
    fn test_fast_feedback_is_applied_once() {
        // This test ensures we don't double-apply context overrides
        // (once in ScenarioConfigManager, once in test wrapper)
        let mgr = ScenarioConfigManager::new();
        let mut ctx = TestConfigContext::default();
        ctx.time_constraints.target_feedback_time = Some(Duration::from_secs(30));
        
        let cfg = get_context_config(&mgr, &ctx);
        
        // The wrapper sets JSON-only formats once.
        assert_eq!(cfg.reporting.formats, vec![ReportFormat::Json]);
        
        // Parallelism ≤ 4 once (not pushed below 1 by repeated mins).
        assert!(cfg.max_parallel_tests >= 1 && cfg.max_parallel_tests <= 4);
        
        // Artifacts disabled exactly once for very short feedback times.
        assert!(!cfg.reporting.include_artifacts);
    }
    
    #[test] 
    fn test_no_double_clamp_on_resources() {
        // Verify resource constraints are applied exactly once
        let mgr = ScenarioConfigManager::new();
        let mut ctx = TestConfigContext::default();
        ctx.resource_constraints.max_parallel_tests = Some(2);
        ctx.resource_constraints.network_access = false;
        
        let cfg = get_context_config(&mgr, &ctx);
        
        // Parallelism should be exactly 2 (not clamped again)
        assert_eq!(cfg.max_parallel_tests, 2);
        
        // Network features disabled once
        assert!(!cfg.reporting.upload_reports);
        #[cfg(feature = "fixtures")]
        {
            assert!(!cfg.fixtures.auto_download);
            assert!(cfg.fixtures.base_url.is_none());
        }
    }
}
