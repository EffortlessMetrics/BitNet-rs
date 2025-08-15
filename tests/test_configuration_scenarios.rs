// Configuration scenarios testing for BitNet.rs testing framework
//
// This module tests the comprehensive configuration management system
// that supports various testing scenarios including unit, integration,
// performance, cross-validation, and other specialized testing contexts.

use bitnet_tests::{
    config::{validate_config, ReportFormat, TestConfig},
    config_scenarios::{
        scenarios, ConfigurationContext, EnvironmentType, Platform, PlatformSettings,
        QualityRequirements, ResourceConstraints, ScenarioConfigManager, TestingScenario,
        TimeConstraints,
    },
    errors::{TestError, TestResult},
    harness::{TestCase, TestHarness, TestSuite},
    results::{TestMetrics, TestResult as TestCaseResult, TestStatus},
    FixtureManager,
};
use std::collections::HashMap;
use std::env;
use std::time::Duration;

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
            Box::new(ConfigurationContextTest),
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

#[async_trait::async_trait]
impl TestCase for ScenarioConfigurationTest {
    fn name(&self) -> &str {
        "Scenario Configuration Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
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

        // Test stress testing scenario
        let stress_config = manager.get_scenario_config(&TestingScenario::Stress);
        assert!(
            stress_config.max_parallel_tests > num_cpus::get(),
            "Stress tests should oversubscribe CPU"
        );
        assert_eq!(
            stress_config.test_timeout,
            Duration::from_secs(1800),
            "Stress tests should have long timeout"
        );
        assert!(
            stress_config.reporting.generate_performance,
            "Stress tests should generate performance reports"
        );
        validate_config(&stress_config)
            .map_err(|e| TestError::assertion(format!("Stress config validation failed: {}", e)))?;

        // Test security testing scenario
        let security_config = manager.get_scenario_config(&TestingScenario::Security);
        assert_eq!(security_config.max_parallel_tests, 1, "Security tests should be sequential");
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

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test environment-specific configurations
struct EnvironmentConfigurationTest;

#[async_trait::async_trait]
impl TestCase for EnvironmentConfigurationTest {
    fn name(&self) -> &str {
        "Environment Configuration Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();

        // Test development environment
        let dev_env_config = manager.get_environment_config(&EnvironmentType::Development);
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
        let ci_env_config = manager.get_environment_config(&EnvironmentType::ContinuousIntegration);
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

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test resource constraints application
struct ResourceConstraintsTest;

#[async_trait::async_trait]
impl TestCase for ResourceConstraintsTest {
    fn name(&self) -> &str {
        "Resource Constraints Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();
        let mut context = ConfigurationContext::default();

        // Test parallel test constraint
        context.resource_constraints.max_parallel_tests = Some(2);
        let config = manager.get_context_config(&context);
        assert!(config.max_parallel_tests <= 2, "Parallel test constraint should be applied");

        // Test disk cache constraint
        context.resource_constraints.max_disk_cache_mb = 500;
        let config = manager.get_context_config(&context);
        assert_eq!(
            config.fixtures.max_cache_size,
            500 * 1024 * 1024,
            "Disk cache constraint should be applied"
        );

        // Test network access constraint
        context.resource_constraints.network_access = false;
        let config = manager.get_context_config(&context);
        assert!(!config.fixtures.auto_download, "Network constraint should disable auto-download");
        assert!(config.fixtures.base_url.is_none(), "Network constraint should clear base URL");
        assert!(
            !config.reporting.upload_reports,
            "Network constraint should disable report upload"
        );

        // Test memory constraint (should not affect config directly but validates constraint)
        context.resource_constraints.max_memory_mb = 1024;
        let config = manager.get_context_config(&context);
        // Memory constraint doesn't directly affect config but should be preserved
        assert_eq!(context.resource_constraints.max_memory_mb, 1024);

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test time constraints application
struct TimeConstraintsTest;

#[async_trait::async_trait]
impl TestCase for TimeConstraintsTest {
    fn name(&self) -> &str {
        "Time Constraints Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();
        let mut context = ConfigurationContext::default();

        // Test test timeout constraint
        context.time_constraints.max_test_timeout = Duration::from_secs(60);
        let config = manager.get_context_config(&context);
        assert!(
            config.test_timeout <= Duration::from_secs(60),
            "Test timeout constraint should be applied"
        );

        // Test fast feedback constraint
        context.time_constraints.target_feedback_time = Some(Duration::from_secs(120));
        let config = manager.get_context_config(&context);
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
        let config = manager.get_context_config(&context);
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

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test quality requirements application
struct QualityRequirementsTest;

#[async_trait::async_trait]
impl TestCase for QualityRequirementsTest {
    fn name(&self) -> &str {
        "Quality Requirements Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();
        let mut context = ConfigurationContext::default();

        // Test coverage requirement
        context.quality_requirements.min_coverage = 0.95;
        let config = manager.get_context_config(&context);
        assert_eq!(config.coverage_threshold, 0.95, "Coverage requirement should be applied");

        // Test comprehensive reporting requirement
        context.quality_requirements.comprehensive_reporting = true;
        let config = manager.get_context_config(&context);
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
        let config = manager.get_context_config(&context);
        assert!(config.reporting.generate_performance, "Performance monitoring should be enabled");

        // Test cross-validation requirement
        context.quality_requirements.cross_validation = true;
        context.quality_requirements.accuracy_tolerance = 1e-8;
        let config = manager.get_context_config(&context);
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

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test platform-specific configurations
struct PlatformSpecificConfigurationTest;

#[async_trait::async_trait]
impl TestCase for PlatformSpecificConfigurationTest {
    fn name(&self) -> &str {
        "Platform Specific Configuration Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();
        let mut context = ConfigurationContext::default();

        // Test Windows platform
        context.platform_settings.platform = Platform::Windows;
        context.scenario = TestingScenario::Unit; // Start with high parallelism
        let config = manager.get_context_config(&context);
        assert!(config.max_parallel_tests <= 8, "Windows should limit parallelism to 8");

        // Test macOS platform
        context.platform_settings.platform = Platform::MacOS;
        let config = manager.get_context_config(&context);
        assert!(config.max_parallel_tests <= 6, "macOS should limit parallelism to 6");

        // Test Linux platform (should not limit as much)
        context.platform_settings.platform = Platform::Linux;
        let config = manager.get_context_config(&context);
        // Linux doesn't impose additional limits, so should use scenario default

        // Test Generic platform
        context.platform_settings.platform = Platform::Generic;
        let config = manager.get_context_config(&context);
        // Generic doesn't impose additional limits

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test configuration context functionality
struct ConfigurationContextTest;

#[async_trait::async_trait]
impl TestCase for ConfigurationContextTest {
    fn name(&self) -> &str {
        "Configuration Context Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();

        // Test complex context configuration
        let mut context = ConfigurationContext::default();
        context.scenario = TestingScenario::Performance;
        context.environment = EnvironmentType::ContinuousIntegration;
        context.resource_constraints.max_parallel_tests = Some(1);
        context.resource_constraints.network_access = false;
        context.time_constraints.max_test_timeout = Duration::from_secs(300);
        context.quality_requirements.min_coverage = 0.85;
        context.quality_requirements.performance_monitoring = true;
        context.platform_settings.platform = Platform::Linux;

        let config = manager.get_context_config(&context);

        // Verify scenario settings are applied
        assert!(
            config.reporting.generate_performance,
            "Performance scenario should generate performance reports"
        );

        // Verify environment settings are applied
        assert_eq!(config.log_level, "debug", "CI environment should use debug logging");

        // Verify resource constraints are applied
        assert_eq!(config.max_parallel_tests, 1, "Resource constraint should limit parallelism");
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

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test environment detection from environment variables
struct EnvironmentDetectionTest;

#[async_trait::async_trait]
impl TestCase for EnvironmentDetectionTest {
    fn name(&self) -> &str {
        "Environment Detection Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
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
        let context = ScenarioConfigManager::context_from_environment();
        assert_eq!(
            context.scenario,
            TestingScenario::Performance,
            "Should detect performance scenario"
        );

        env::set_var("BITNET_TEST_SCENARIO", "unit");
        let context = ScenarioConfigManager::context_from_environment();
        assert_eq!(context.scenario, TestingScenario::Unit, "Should detect unit scenario");

        env::set_var("BITNET_TEST_SCENARIO", "crossval");
        let context = ScenarioConfigManager::context_from_environment();
        assert_eq!(
            context.scenario,
            TestingScenario::CrossValidation,
            "Should detect cross-validation scenario"
        );

        // Test CI environment detection
        env::set_var("CI", "true");
        let context = ScenarioConfigManager::context_from_environment();
        assert_eq!(
            context.environment,
            EnvironmentType::ContinuousIntegration,
            "Should detect CI environment"
        );

        env::remove_var("CI");
        env::set_var("GITHUB_ACTIONS", "true");
        let context = ScenarioConfigManager::context_from_environment();
        assert_eq!(
            context.environment,
            EnvironmentType::ContinuousIntegration,
            "Should detect GitHub Actions as CI"
        );

        // Test production environment detection
        env::remove_var("GITHUB_ACTIONS");
        env::set_var("BITNET_ENV", "production");
        let context = ScenarioConfigManager::context_from_environment();
        assert_eq!(
            context.environment,
            EnvironmentType::Production,
            "Should detect production environment"
        );

        // Test resource constraints from environment
        env::set_var("BITNET_MAX_MEMORY_MB", "2048");
        env::set_var("BITNET_MAX_PARALLEL", "4");
        env::set_var("BITNET_NO_NETWORK", "1");
        let context = ScenarioConfigManager::context_from_environment();
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
        let context = ScenarioConfigManager::context_from_environment();
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
        let context = ScenarioConfigManager::context_from_environment();
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

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test convenience functions
struct ConvenienceFunctionsTest;

#[async_trait::async_trait]
impl TestCase for ConvenienceFunctionsTest {
    fn name(&self) -> &str {
        "Convenience Functions Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test unit testing convenience function
        let unit_config = scenarios::unit_testing();
        assert_eq!(unit_config.log_level, "warn", "Unit testing convenience function should work");
        validate_config(&unit_config)
            .map_err(|e| TestError::assertion(format!("Unit config validation failed: {}", e)))?;

        // Test integration testing convenience function
        let integration_config = scenarios::integration_testing();
        assert_eq!(
            integration_config.log_level, "info",
            "Integration testing convenience function should work"
        );
        validate_config(&integration_config).map_err(|e| {
            TestError::assertion(format!("Integration config validation failed: {}", e))
        })?;

        // Test performance testing convenience function
        let performance_config = scenarios::performance_testing();
        assert_eq!(
            performance_config.max_parallel_tests, 1,
            "Performance testing convenience function should work"
        );
        validate_config(&performance_config).map_err(|e| {
            TestError::assertion(format!("Performance config validation failed: {}", e))
        })?;

        // Test cross-validation testing convenience function
        let crossval_config = scenarios::cross_validation_testing();
        assert!(
            crossval_config.crossval.enabled,
            "Cross-validation testing convenience function should work"
        );

        // Test smoke testing convenience function
        let smoke_config = scenarios::smoke_testing();
        assert_eq!(
            smoke_config.test_timeout,
            Duration::from_secs(10),
            "Smoke testing convenience function should work"
        );
        validate_config(&smoke_config)
            .map_err(|e| TestError::assertion(format!("Smoke config validation failed: {}", e)))?;

        // Test development convenience function
        let dev_config = scenarios::development();
        assert!(
            !dev_config.reporting.generate_coverage,
            "Development convenience function should work"
        );
        validate_config(&dev_config).map_err(|e| {
            TestError::assertion(format!("Development config validation failed: {}", e))
        })?;

        // Test CI convenience function
        let ci_config = scenarios::continuous_integration();
        assert_eq!(ci_config.log_level, "debug", "CI convenience function should work");
        validate_config(&ci_config)
            .map_err(|e| TestError::assertion(format!("CI config validation failed: {}", e)))?;

        // Test from_environment convenience function
        let env_config = scenarios::from_environment();
        validate_config(&env_config).map_err(|e| {
            TestError::assertion(format!("Environment config validation failed: {}", e))
        })?;

        // Test from_context convenience function
        let context = ConfigurationContext::default();
        let context_config = scenarios::from_context(&context);
        validate_config(&context_config).map_err(|e| {
            TestError::assertion(format!("Context config validation failed: {}", e))
        })?;

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test configuration validation for all scenarios
struct ConfigurationValidationTest;

#[async_trait::async_trait]
impl TestCase for ConfigurationValidationTest {
    fn name(&self) -> &str {
        "Configuration Validation Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();

        // Test that all scenario configurations are valid
        for scenario in ScenarioConfigManager::available_scenarios() {
            let config = manager.get_scenario_config(&scenario);
            validate_config(&config).map_err(|e| {
                TestError::assertion(format!(
                    "Scenario {:?} config validation failed: {}",
                    scenario, e
                ))
            })?;
        }

        // Test that all environment configurations are valid
        for environment in [
            EnvironmentType::Development,
            EnvironmentType::ContinuousIntegration,
            EnvironmentType::Staging,
            EnvironmentType::Production,
            EnvironmentType::Testing,
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

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test scenario descriptions
struct ScenarioDescriptionsTest;

#[async_trait::async_trait]
impl TestCase for ScenarioDescriptionsTest {
    fn name(&self) -> &str {
        "Scenario Descriptions Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        // Test that all scenarios have descriptions
        for scenario in ScenarioConfigManager::available_scenarios() {
            let description = ScenarioConfigManager::scenario_description(&scenario);
            assert!(!description.is_empty(), "Scenario {:?} should have a description", scenario);
            assert!(
                description.len() > 10,
                "Scenario {:?} description should be meaningful",
                scenario
            );
        }

        // Test specific descriptions
        let unit_desc = ScenarioConfigManager::scenario_description(&TestingScenario::Unit);
        assert!(unit_desc.contains("Fast"), "Unit description should mention speed");
        assert!(unit_desc.contains("isolated"), "Unit description should mention isolation");

        let performance_desc =
            ScenarioConfigManager::scenario_description(&TestingScenario::Performance);
        assert!(
            performance_desc.contains("Sequential"),
            "Performance description should mention sequential execution"
        );
        assert!(
            performance_desc.contains("benchmarking"),
            "Performance description should mention benchmarking"
        );

        let crossval_desc =
            ScenarioConfigManager::scenario_description(&TestingScenario::CrossValidation);
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

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test complex scenario combinations
struct ComplexScenarioTest;

#[async_trait::async_trait]
impl TestCase for ComplexScenarioTest {
    fn name(&self) -> &str {
        "Complex Scenario Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();

        // Test performance testing in CI environment with resource constraints
        let mut context = ConfigurationContext::default();
        context.scenario = TestingScenario::Performance;
        context.environment = EnvironmentType::ContinuousIntegration;
        context.resource_constraints.max_parallel_tests = Some(1);
        context.resource_constraints.max_memory_mb = 2048;
        context.time_constraints.max_test_timeout = Duration::from_secs(300);
        context.quality_requirements.performance_monitoring = true;
        context.platform_settings.platform = Platform::Linux;

        let config = manager.get_context_config(&context);
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
        let mut context = ConfigurationContext::default();
        context.scenario = TestingScenario::Unit;
        context.environment = EnvironmentType::Development;
        context.time_constraints.target_feedback_time = Some(Duration::from_secs(60));
        context.quality_requirements.comprehensive_reporting = false;

        let config = manager.get_context_config(&context);
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
        let mut context = ConfigurationContext::default();
        context.scenario = TestingScenario::CrossValidation;
        context.environment = EnvironmentType::Production;
        context.quality_requirements.comprehensive_reporting = true;
        context.quality_requirements.cross_validation = true;
        context.quality_requirements.accuracy_tolerance = 1e-8;

        let config = manager.get_context_config(&context);
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

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test configuration merging behavior
struct ConfigurationMergingTest;

#[async_trait::async_trait]
impl TestCase for ConfigurationMergingTest {
    fn name(&self) -> &str {
        "Configuration Merging Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();

        // Test that scenario and environment configs are properly merged
        let mut context = ConfigurationContext::default();
        context.scenario = TestingScenario::Unit; // Uses "warn" logging
        context.environment = EnvironmentType::ContinuousIntegration; // Uses "debug" logging

        let config = manager.get_context_config(&context);
        // Environment should override scenario
        assert_eq!(config.log_level, "debug", "Environment should override scenario logging");

        // Test that constraints override both scenario and environment
        context.time_constraints.target_feedback_time = Some(Duration::from_secs(60));
        let config = manager.get_context_config(&context);
        assert!(
            !config.reporting.generate_coverage,
            "Time constraints should override environment settings"
        );

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Test edge cases and boundary conditions
struct EdgeCaseConfigurationTest;

#[async_trait::async_trait]
impl TestCase for EdgeCaseConfigurationTest {
    fn name(&self) -> &str {
        "Edge Case Configuration Test"
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let mut metrics = TestMetrics::default();
        let start_time = std::time::Instant::now();

        let manager = ScenarioConfigManager::new();

        // Test zero resource constraints
        let mut context = ConfigurationContext::default();
        context.resource_constraints.max_parallel_tests = Some(0);
        let config = manager.get_context_config(&context);
        // Should not set to 0 (invalid), should use minimum of 1 or scenario default
        assert!(config.max_parallel_tests > 0, "Should not allow zero parallel tests");

        // Test very large resource constraints
        context.resource_constraints.max_parallel_tests = Some(1000);
        let config = manager.get_context_config(&context);
        // Should be limited by scenario or platform constraints
        assert!(config.max_parallel_tests <= 1000, "Should respect large constraints");

        // Test very short timeout
        context.time_constraints.max_test_timeout = Duration::from_secs(1);
        let config = manager.get_context_config(&context);
        assert_eq!(
            config.test_timeout,
            Duration::from_secs(1),
            "Should respect very short timeout"
        );

        // Test very long timeout
        context.time_constraints.max_test_timeout = Duration::from_secs(86400); // 24 hours
        let config = manager.get_context_config(&context);
        assert!(
            config.test_timeout <= Duration::from_secs(86400),
            "Should respect very long timeout"
        );

        // Test extreme coverage requirements
        context.quality_requirements.min_coverage = 1.0; // 100%
        let config = manager.get_context_config(&context);
        assert_eq!(config.coverage_threshold, 1.0, "Should respect 100% coverage requirement");

        context.quality_requirements.min_coverage = 0.0; // 0%
        let config = manager.get_context_config(&context);
        assert_eq!(config.coverage_threshold, 0.0, "Should respect 0% coverage requirement");

        metrics.wall_time = start_time.elapsed();
        metrics.add_assertion();
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        Ok(())
    }
}

/// Main test runner for configuration scenarios
#[tokio::main]
async fn main() -> TestResult<()> {
    // Initialize logging
    env_logger::init();

    // Create test harness
    let config = TestConfig::default();
    let harness = TestHarness::new(config);

    // Create and run test suite
    let mut test_suite = ConfigurationScenariosTestSuite::new();
    let result = harness.run_test_suite(test_suite).await?;

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
