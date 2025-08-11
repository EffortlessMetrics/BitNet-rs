//! Simple tests for CI reporting functionality

use std::collections::HashMap;
use std::time::Duration;

// Test the basic structures and functionality without external dependencies
#[test]
fn test_notification_config_creation() {
    use bitnet_tests::ci_reporting::NotificationConfig;

    let config = NotificationConfig::default();

    assert!(config.notify_on_failure);
    assert!(!config.notify_on_success);
    assert!(config.check_performance_regression);
    assert_eq!(config.performance_regression_threshold, 1.1);
    assert!(config.create_status_checks);
    assert!(config.create_pr_comments);
}

#[test]
fn test_ci_context_creation() {
    use bitnet_tests::ci_reporting::CIContext;

    let context = CIContext {
        commit_sha: Some("abc123".to_string()),
        pr_number: Some(42),
        branch_name: Some("main".to_string()),
        workflow_run_id: Some("12345".to_string()),
        actor: Some("developer".to_string()),
    };

    assert_eq!(context.commit_sha, Some("abc123".to_string()));
    assert_eq!(context.pr_number, Some(42));
    assert_eq!(context.branch_name, Some("main".to_string()));
}

#[test]
fn test_trend_config_creation() {
    use bitnet_tests::trend_reporting::TrendConfig;

    let config = TrendConfig::default();

    assert_eq!(config.retention_days, 90);
    assert_eq!(config.min_samples_for_baseline, 5);
    assert_eq!(config.regression_threshold, 1.2);
}

#[test]
fn test_trend_config_custom() {
    use bitnet_tests::trend_reporting::TrendConfig;

    let config = TrendConfig {
        retention_days: 30,
        min_samples_for_baseline: 3,
        regression_threshold: 1.5,
    };

    assert_eq!(config.retention_days, 30);
    assert_eq!(config.min_samples_for_baseline, 3);
    assert_eq!(config.regression_threshold, 1.5);
}

#[test]
fn test_test_run_metadata_creation() {
    use bitnet_tests::trend_reporting::TestRunMetadata;

    let mut env = HashMap::new();
    env.insert("os".to_string(), "ubuntu".to_string());

    let mut config = HashMap::new();
    config.insert("features".to_string(), "cpu".to_string());

    let metadata = TestRunMetadata {
        commit_sha: Some("abc123".to_string()),
        branch: Some("main".to_string()),
        pr_number: Some(42),
        environment: env,
        configuration: config,
    };

    assert_eq!(metadata.commit_sha, Some("abc123".to_string()));
    assert_eq!(metadata.branch, Some("main".to_string()));
    assert_eq!(metadata.pr_number, Some(42));
    assert_eq!(metadata.environment.get("os"), Some(&"ubuntu".to_string()));
    assert_eq!(
        metadata.configuration.get("features"),
        Some(&"cpu".to_string())
    );
}
