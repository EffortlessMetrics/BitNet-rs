# BitNet.rs CI Reporting and Notifications Guide

This guide explains how to use the comprehensive CI reporting and notifications system implemented for BitNet.rs.

## Overview

The CI reporting system provides:

- **GitHub Status Checks**: Automatic status updates on commits and pull requests
- **Pull Request Comments**: Detailed test result summaries with failure analysis
- **Performance Regression Detection**: Automated detection of performance degradations
- **Trend Analysis**: Historical tracking of test performance and stability
- **Failure Notifications**: Configurable alerts for test failures
- **Comprehensive Reporting**: HTML, JSON, and Markdown report generation

## Components

### 1. CI Notification Manager (`ci_reporting.rs`)

The main component that orchestrates CI reporting activities.

```rust
use bitnet_tests::ci_reporting::{CINotificationManager, NotificationConfig};

let config = NotificationConfig {
    notify_on_failure: true,
    notify_on_success: false,
    check_performance_regression: true,
    performance_regression_threshold: 1.1, // 10% slower threshold
    create_status_checks: true,
    create_pr_comments: true,
};

let manager = CINotificationManager::new(config)?;
```

### 2. GitHub Reporter (`ci_reporting.rs`)

Handles GitHub API interactions for status checks and PR comments.

```rust
use bitnet_tests::ci_reporting::GitHubReporter;

let reporter = GitHubReporter::new()?;

// Create status check
reporter.create_status_check(
    "abc123", // commit SHA
    "bitnet-rs/tests", // context
    StatusState::Success,
    "All tests passed",
    Some("https://example.com/report")
).await?;
```

### 3. Trend Reporter (`trend_reporting.rs`)

Tracks test results over time and detects performance regressions.

```rust
use bitnet_tests::trend_reporting::{TrendReporter, TrendConfig};

let config = TrendConfig {
    retention_days: 90,
    min_samples_for_baseline: 5,
    regression_threshold: 1.2, // 20% slower threshold
};

let reporter = TrendReporter::new(storage_path, config);

// Record test results
reporter.record_test_results(&test_results, &metadata).await?;

// Generate trend report
let report = reporter.generate_trend_report(30, Some("main")).await?;
```

## GitHub Actions Integration

### Workflow Configuration

The CI reporting system integrates with GitHub Actions through the `ci-reporting.yml` workflow:

```yaml
name: CI Reporting and Notifications

on:
  workflow_run:
    workflows: ["CI", "Testing Framework Unit Tests"]
    types: [completed]
  pull_request:
    types: [opened, synchronize, reopened]
  push:
    branches: [main, develop]

jobs:
  collect-test-results:
    runs-on: ubuntu-latest
    steps:
    - name: Generate CI reports
      run: |
        cd tests
        cargo run --bin generate_ci_report -- \
          --results-dir ../test-results \
          --output-dir ../ci-reports \
          --commit-sha ${{ github.sha }} \
          --pr-number ${{ github.event.pull_request.number }}
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Required Environment Variables

- `GITHUB_TOKEN`: GitHub API token for creating status checks and comments
- `GITHUB_REPOSITORY`: Repository in format "owner/repo"
- `GITHUB_SHA`: Commit SHA for status checks
- `GITHUB_PR_NUMBER`: Pull request number (for PR events)
- `GITHUB_REF_NAME`: Branch name
- `GITHUB_RUN_ID`: Workflow run ID
- `GITHUB_ACTOR`: User who triggered the workflow

## Command Line Tools

### 1. Generate CI Report (`generate_ci_report`)

Processes test results and generates CI reports.

```bash
cargo run --bin generate_ci_report -- \
  --results-dir ./test-results \
  --output-dir ./ci-reports \
  --trend-data-dir ./trend-data \
  --commit-sha abc123 \
  --pr-number 42 \
  --branch main
```

**Outputs:**
- `status-checks.json`: GitHub status check data
- `pr-comment.md`: Pull request comment content
- `test-results.json`: Processed test results
- Trend data files for historical analysis

### 2. Check Performance Regressions (`check_performance_regressions`)

Analyzes current results against historical baselines.

```bash
cargo run --bin check_performance_regressions -- \
  --current-results ./test-results.json \
  --trend-data ./trend-data \
  --output ./regression-report.json \
  --threshold 1.2 \
  --min-samples 5
```

**Exit Codes:**
- `0`: No regressions detected
- `1`: Performance regressions found

### 3. Generate Trend Analysis (`generate_trend_analysis`)

Creates comprehensive trend reports from historical data.

```bash
cargo run --bin generate_trend_analysis -- \
  --trend-data ./trend-data \
  --output-dir ./trend-reports \
  --days-back 30 \
  --branch main
```

**Outputs:**
- `trend-analysis.html`: Interactive HTML report
- `trend-analysis.json`: Machine-readable data
- `performance-trends.json`: Performance trend data
- `trend-summary.md`: Executive summary

## Configuration

### Notification Configuration

```rust
pub struct NotificationConfig {
    pub notify_on_failure: bool,           // Send notifications on test failures
    pub notify_on_success: bool,           // Send notifications on success (main branch)
    pub check_performance_regression: bool, // Enable regression detection
    pub performance_regression_threshold: f64, // Threshold for regression (1.1 = 10% slower)
    pub create_status_checks: bool,        // Create GitHub status checks
    pub create_pr_comments: bool,          // Create PR comments
}
```

### Trend Configuration

```rust
pub struct TrendConfig {
    pub retention_days: u32,               // How long to keep trend data
    pub min_samples_for_baseline: usize,   // Minimum samples for baseline calculation
    pub regression_threshold: f64,         // Performance regression threshold
}
```

## Report Formats

### Status Checks

GitHub status checks are created for:
- Overall test suite status
- Individual test suite status
- Performance regression status

### Pull Request Comments

PR comments include:
- Overall test summary with pass/fail counts
- Test suite breakdown with individual results
- Failed test details with error messages
- Performance regression warnings (if detected)

### Trend Reports

HTML trend reports provide:
- Test stability analysis over time
- Performance trend visualization
- Regression detection results
- Executive summary with recommendations

## Integration Examples

### Basic CI Integration

```rust
use bitnet_tests::ci_reporting::{CINotificationManager, NotificationConfig, CIContext};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load test results
    let test_results = load_test_results().await?;
    
    // Create CI context
    let ci_context = CIContext::from_env();
    
    // Configure notifications
    let config = NotificationConfig::default();
    let manager = CINotificationManager::new(config)?;
    
    // Process results and send notifications
    manager.process_test_results(&test_results, &ci_context).await?;
    
    Ok(())
}
```

### Performance Regression Detection

```rust
use bitnet_tests::trend_reporting::{TrendReporter, TrendConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrendConfig::default();
    let reporter = TrendReporter::new("./trend-data".into(), config);
    
    // Load current test results
    let current_results = load_current_results().await?;
    
    // Detect regressions
    let regressions = reporter.detect_regressions(&current_results, 30).await?;
    
    if !regressions.is_empty() {
        println!("⚠️ Performance regressions detected:");
        for regression in &regressions {
            println!("  - {}: {:.1}% slower", 
                regression.test_name, 
                regression.regression_percent);
        }
        std::process::exit(1);
    }
    
    Ok(())
}
```

## Troubleshooting

### Common Issues

1. **GitHub API Rate Limits**
   - Use GitHub App tokens instead of personal access tokens
   - Implement retry logic with exponential backoff

2. **Missing Environment Variables**
   - Ensure all required GitHub environment variables are set
   - Check workflow permissions for GitHub token

3. **Performance Baseline Issues**
   - Ensure sufficient historical data (min_samples_for_baseline)
   - Check trend data retention settings

4. **Report Generation Failures**
   - Verify output directory permissions
   - Check available disk space for trend data

### Debug Mode

Enable debug logging:

```bash
RUST_LOG=debug cargo run --bin generate_ci_report -- [args]
```

### Validation

Test the CI reporting system:

```bash
# Run standalone tests
rustc tests/ci_reporting_standalone_test.rs
./ci_reporting_standalone_test

# Run example
cargo run --example ci_reporting_example
```

## Best Practices

1. **Configure Appropriate Thresholds**
   - Set regression thresholds based on acceptable performance variance
   - Use different thresholds for different types of tests

2. **Manage Trend Data**
   - Regularly clean up old trend data to manage storage
   - Archive important historical data before cleanup

3. **Monitor Notification Volume**
   - Avoid notification fatigue by configuring appropriate triggers
   - Use different notification channels for different severity levels

4. **Test Report Quality**
   - Ensure test names are descriptive for better reporting
   - Include relevant metadata in test results

5. **Performance Baselines**
   - Establish baselines on stable, representative hardware
   - Update baselines when making intentional performance changes

## Future Enhancements

Planned improvements include:
- Slack/Teams integration for notifications
- Advanced statistical analysis for regression detection
- Interactive performance dashboards
- Integration with external monitoring systems
- Automated performance baseline updates