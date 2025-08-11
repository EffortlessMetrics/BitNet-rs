//! CI reporting and notifications system
//!
//! This module provides comprehensive CI integration for test reporting,
//! including GitHub status checks, pull request comments, and notifications.

use crate::results::{TestResult, TestSuiteResult};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::time::Duration;
use tokio::fs;
use tracing::{debug, error, info, warn};

/// GitHub API client for CI reporting
pub struct GitHubReporter {
    client: reqwest::Client,
    token: Option<String>,
    repo_owner: String,
    repo_name: String,
    base_url: String,
}

impl GitHubReporter {
    pub fn new() -> Result<Self> {
        let token = env::var("GITHUB_TOKEN").ok();
        let repo =
            env::var("GITHUB_REPOSITORY").unwrap_or_else(|_| "bitnet-rs/bitnet.rs".to_string());

        let parts: Vec<&str> = repo.split('/').collect();
        if parts.len() != 2 {
            return Err(anyhow::anyhow!(
                "Invalid GITHUB_REPOSITORY format: {}",
                repo
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("BitNet.rs-CI-Reporter/1.0")
            .build()?;

        Ok(Self {
            client,
            token,
            repo_owner: parts[0].to_string(),
            repo_name: parts[1].to_string(),
            base_url: "https://api.github.com".to_string(),
        })
    }

    /// Create or update a status check for a commit
    pub async fn create_status_check(
        &self,
        sha: &str,
        context: &str,
        state: StatusState,
        description: &str,
        target_url: Option<&str>,
    ) -> Result<()> {
        let token = self
            .token
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GITHUB_TOKEN not available"))?;

        let url = format!(
            "{}/repos/{}/{}/statuses/{}",
            self.base_url, self.repo_owner, self.repo_name, sha
        );

        let payload = StatusCheckPayload {
            state,
            target_url: target_url.map(|s| s.to_string()),
            description: description.to_string(),
            context: context.to_string(),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("token {}", token))
            .header("Accept", "application/vnd.github.v3+json")
            .json(&payload)
            .send()
            .await
            .context("Failed to send status check request")?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "GitHub API error: {} - {}",
                response.status(),
                error_text
            ));
        }

        info!("Created status check: {} - {}", context, state);
        Ok(())
    }

    /// Create or update a pull request comment
    pub async fn create_pr_comment(
        &self,
        pr_number: u64,
        body: &str,
        comment_id: Option<u64>,
    ) -> Result<u64> {
        let token = self
            .token
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GITHUB_TOKEN not available"))?;

        let (url, method) = if let Some(id) = comment_id {
            // Update existing comment
            (
                format!(
                    "{}/repos/{}/{}/issues/comments/{}",
                    self.base_url, self.repo_owner, self.repo_name, id
                ),
                "PATCH",
            )
        } else {
            // Create new comment
            (
                format!(
                    "{}/repos/{}/{}/issues/{}/comments",
                    self.base_url, self.repo_owner, self.repo_name, pr_number
                ),
                "POST",
            )
        };

        let payload = CommentPayload {
            body: body.to_string(),
        };

        let request = match method {
            "POST" => self.client.post(&url),
            "PATCH" => self.client.patch(&url),
            _ => unreachable!(),
        };

        let response = request
            .header("Authorization", format!("token {}", token))
            .header("Accept", "application/vnd.github.v3+json")
            .json(&payload)
            .send()
            .await
            .context("Failed to send comment request")?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "GitHub API error: {} - {}",
                response.status(),
                error_text
            ));
        }

        let comment_response: CommentResponse = response
            .json()
            .await
            .context("Failed to parse comment response")?;

        info!("Created/updated PR comment: {}", comment_response.id);
        Ok(comment_response.id)
    }

    /// Find existing comment by marker
    pub async fn find_comment_by_marker(
        &self,
        pr_number: u64,
        marker: &str,
    ) -> Result<Option<u64>> {
        let token = self
            .token
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GITHUB_TOKEN not available"))?;

        let url = format!(
            "{}/repos/{}/{}/issues/{}/comments",
            self.base_url, self.repo_owner, self.repo_name, pr_number
        );

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("token {}", token))
            .header("Accept", "application/vnd.github.v3+json")
            .send()
            .await
            .context("Failed to fetch comments")?;

        if !response.status().is_success() {
            return Ok(None);
        }

        let comments: Vec<CommentResponse> = response
            .json()
            .await
            .context("Failed to parse comments response")?;

        for comment in comments {
            if comment.body.contains(marker) {
                return Ok(Some(comment.id));
            }
        }

        Ok(None)
    }
}

/// CI notification manager
pub struct CINotificationManager {
    github_reporter: GitHubReporter,
    config: NotificationConfig,
}

impl CINotificationManager {
    pub fn new(config: NotificationConfig) -> Result<Self> {
        let github_reporter = GitHubReporter::new()?;

        Ok(Self {
            github_reporter,
            config,
        })
    }

    /// Process test results and send appropriate notifications
    pub async fn process_test_results(
        &self,
        results: &[TestSuiteResult],
        context: &CIContext,
    ) -> Result<()> {
        info!("Processing test results for CI notifications");

        // Generate summary
        let summary = self.generate_test_summary(results);

        // Create status checks
        if let Some(sha) = &context.commit_sha {
            self.create_test_status_checks(sha, &summary).await?;
        }

        // Create PR comment if this is a PR
        if let Some(pr_number) = context.pr_number {
            self.create_pr_test_comment(pr_number, results, &summary)
                .await?;
        }

        // Send failure notifications if needed
        if summary.has_failures && self.config.notify_on_failure {
            self.send_failure_notifications(results, context).await?;
        }

        // Check for performance regressions
        if self.config.check_performance_regression {
            self.check_performance_regressions(results, context).await?;
        }

        Ok(())
    }

    async fn create_test_status_checks(&self, sha: &str, summary: &TestSummary) -> Result<()> {
        // Overall test status
        let overall_state = if summary.has_failures {
            StatusState::Failure
        } else {
            StatusState::Success
        };

        let description = format!(
            "{}/{} tests passed ({:.1}%)",
            summary.passed_tests, summary.total_tests, summary.success_rate
        );

        self.github_reporter
            .create_status_check(
                sha,
                "bitnet-rs/tests",
                overall_state,
                &description,
                summary.report_url.as_deref(),
            )
            .await?;

        // Individual suite status checks
        for suite_summary in &summary.suite_summaries {
            let state = if suite_summary.failed > 0 {
                StatusState::Failure
            } else {
                StatusState::Success
            };

            let description = format!(
                "{}/{} tests passed",
                suite_summary.passed, suite_summary.total
            );

            let context = format!("bitnet-rs/tests/{}", suite_summary.name);

            self.github_reporter
                .create_status_check(sha, &context, state, &description, None)
                .await?;
        }

        Ok(())
    }

    async fn create_pr_test_comment(
        &self,
        pr_number: u64,
        results: &[TestSuiteResult],
        summary: &TestSummary,
    ) -> Result<()> {
        let marker = "<!-- BitNet.rs Test Results -->";

        // Check if comment already exists
        let existing_comment = self
            .github_reporter
            .find_comment_by_marker(pr_number, marker)
            .await?;

        let comment_body = self.generate_pr_comment_body(results, summary, marker);

        self.github_reporter
            .create_pr_comment(pr_number, &comment_body, existing_comment)
            .await?;

        Ok(())
    }

    async fn send_failure_notifications(
        &self,
        results: &[TestSuiteResult],
        context: &CIContext,
    ) -> Result<()> {
        let failed_tests: Vec<&TestResult> = results
            .iter()
            .flat_map(|suite| &suite.test_results)
            .filter(|test| matches!(test.status, crate::results::TestStatus::Failed))
            .collect();

        if failed_tests.is_empty() {
            return Ok();
        }

        info!(
            "Sending failure notifications for {} failed tests",
            failed_tests.len()
        );

        // For now, we'll log the failures. In a real implementation,
        // you might send to Slack, email, or other notification systems
        for test in &failed_tests {
            error!(
                "Test failed: {} - {}",
                test.test_name,
                test.error.as_deref().unwrap_or("Unknown error")
            );
        }

        Ok(())
    }

    async fn check_performance_regressions(
        &self,
        results: &[TestSuiteResult],
        context: &CIContext,
    ) -> Result<()> {
        // Load baseline performance data
        let baseline_path = PathBuf::from("tests/performance_baselines.json");
        let baseline_data = if baseline_path.exists() {
            let content = fs::read_to_string(&baseline_path).await?;
            serde_json::from_str::<PerformanceBaseline>(&content)?
        } else {
            warn!("No performance baseline found, skipping regression check");
            return Ok(());
        };

        let mut regressions = Vec::new();

        for suite in results {
            for test in &suite.test_results {
                if let Some(baseline_duration) =
                    baseline_data.get_baseline_duration(&test.test_name)
                {
                    let current_duration = test.duration;
                    let regression_threshold =
                        baseline_duration.mul_f64(self.config.performance_regression_threshold);

                    if current_duration > regression_threshold {
                        let regression_percent = (current_duration.as_secs_f64()
                            / baseline_duration.as_secs_f64()
                            - 1.0)
                            * 100.0;

                        regressions.push(PerformanceRegression {
                            test_name: test.test_name.clone(),
                            baseline_duration,
                            current_duration,
                            regression_percent,
                        });
                    }
                }
            }
        }

        if !regressions.is_empty() {
            self.send_performance_regression_notification(&regressions, context)
                .await?;
        }

        Ok(())
    }

    async fn send_performance_regression_notification(
        &self,
        regressions: &[PerformanceRegression],
        context: &CIContext,
    ) -> Result<()> {
        info!("Found {} performance regressions", regressions.len());

        // Create status check for performance regression
        if let Some(sha) = &context.commit_sha {
            let description = format!("{} performance regressions detected", regressions.len());

            self.github_reporter
                .create_status_check(
                    sha,
                    "bitnet-rs/performance",
                    StatusState::Failure,
                    &description,
                    None,
                )
                .await?;
        }

        // Log regressions
        for regression in regressions {
            warn!(
                "Performance regression in {}: {:.1}% slower ({:?} -> {:?})",
                regression.test_name,
                regression.regression_percent,
                regression.baseline_duration,
                regression.current_duration
            );
        }

        Ok(())
    }

    fn generate_test_summary(&self, results: &[TestSuiteResult]) -> TestSummary {
        let mut total_tests = 0;
        let mut passed_tests = 0;
        let mut failed_tests = 0;
        let mut suite_summaries = Vec::new();

        for suite in results {
            total_tests += suite.summary.total_tests;
            passed_tests += suite.summary.passed;
            failed_tests += suite.summary.failed;

            suite_summaries.push(SuiteSummary {
                name: suite.suite_name.clone(),
                total: suite.summary.total_tests,
                passed: suite.summary.passed,
                failed: suite.summary.failed,
                duration: suite.total_duration,
            });
        }

        let success_rate = if total_tests > 0 {
            (passed_tests as f64 / total_tests as f64) * 100.0
        } else {
            100.0
        };

        TestSummary {
            total_tests,
            passed_tests,
            failed_tests,
            success_rate,
            has_failures: failed_tests > 0,
            suite_summaries,
            report_url: None, // Could be set to point to detailed report
        }
    }

    fn generate_pr_comment_body(
        &self,
        results: &[TestSuiteResult],
        summary: &TestSummary,
        marker: &str,
    ) -> String {
        let mut body = String::new();

        body.push_str(marker);
        body.push_str("\n\n");

        // Header with overall status
        if summary.has_failures {
            body.push_str("## ❌ Test Results - Some tests failed\n\n");
        } else {
            body.push_str("## ✅ Test Results - All tests passed\n\n");
        }

        // Summary table
        body.push_str("| Metric | Value |\n");
        body.push_str("|--------|-------|\n");
        body.push_str(&format!("| Total Tests | {} |\n", summary.total_tests));
        body.push_str(&format!("| Passed | {} |\n", summary.passed_tests));
        body.push_str(&format!("| Failed | {} |\n", summary.failed_tests));
        body.push_str(&format!(
            "| Success Rate | {:.1}% |\n",
            summary.success_rate
        ));

        // Suite breakdown
        if summary.suite_summaries.len() > 1 {
            body.push_str("\n### Test Suite Breakdown\n\n");
            body.push_str("| Suite | Status | Tests | Duration |\n");
            body.push_str("|-------|--------|-------|----------|\n");

            for suite in &summary.suite_summaries {
                let status = if suite.failed > 0 { "❌" } else { "✅" };
                body.push_str(&format!(
                    "| {} | {} | {}/{} | {:.2}s |\n",
                    suite.name,
                    status,
                    suite.passed,
                    suite.total,
                    suite.duration.as_secs_f64()
                ));
            }
        }

        // Failed tests details
        if summary.has_failures {
            body.push_str("\n### Failed Tests\n\n");

            for suite in results {
                let failed_tests: Vec<&TestResult> = suite
                    .test_results
                    .iter()
                    .filter(|test| matches!(test.status, crate::results::TestStatus::Failed))
                    .collect();

                if !failed_tests.is_empty() {
                    body.push_str(&format!("**{}**\n", suite.suite_name));
                    for test in failed_tests {
                        body.push_str(&format!(
                            "- `{}`: {}\n",
                            test.test_name,
                            test.error.as_deref().unwrap_or("Unknown error")
                        ));
                    }
                    body.push('\n');
                }
            }
        }

        body.push_str("\n---\n");
        body.push_str(&format!(
            "*Generated at {}*",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        body
    }
}

/// Configuration for CI notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    pub notify_on_failure: bool,
    pub notify_on_success: bool,
    pub check_performance_regression: bool,
    pub performance_regression_threshold: f64, // e.g., 1.1 for 10% slower
    pub create_status_checks: bool,
    pub create_pr_comments: bool,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            notify_on_failure: true,
            notify_on_success: false,
            check_performance_regression: true,
            performance_regression_threshold: 1.1, // 10% regression threshold
            create_status_checks: true,
            create_pr_comments: true,
        }
    }
}

/// CI context information
#[derive(Debug, Clone)]
pub struct CIContext {
    pub commit_sha: Option<String>,
    pub pr_number: Option<u64>,
    pub branch_name: Option<String>,
    pub workflow_run_id: Option<String>,
    pub actor: Option<String>,
}

impl CIContext {
    pub fn from_env() -> Self {
        Self {
            commit_sha: env::var("GITHUB_SHA").ok(),
            pr_number: env::var("GITHUB_PR_NUMBER")
                .ok()
                .and_then(|s| s.parse().ok()),
            branch_name: env::var("GITHUB_REF_NAME").ok(),
            workflow_run_id: env::var("GITHUB_RUN_ID").ok(),
            actor: env::var("GITHUB_ACTOR").ok(),
        }
    }
}

// GitHub API types
#[derive(Debug, Serialize)]
struct StatusCheckPayload {
    state: StatusState,
    target_url: Option<String>,
    description: String,
    context: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
enum StatusState {
    Pending,
    Success,
    Error,
    Failure,
}

impl std::fmt::Display for StatusState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatusState::Pending => write!(f, "pending"),
            StatusState::Success => write!(f, "success"),
            StatusState::Error => write!(f, "error"),
            StatusState::Failure => write!(f, "failure"),
        }
    }
}

#[derive(Debug, Serialize)]
struct CommentPayload {
    body: String,
}

#[derive(Debug, Deserialize)]
struct CommentResponse {
    id: u64,
    body: String,
}

// Summary types
#[derive(Debug)]
struct TestSummary {
    total_tests: usize,
    passed_tests: usize,
    failed_tests: usize,
    success_rate: f64,
    has_failures: bool,
    suite_summaries: Vec<SuiteSummary>,
    report_url: Option<String>,
}

#[derive(Debug)]
struct SuiteSummary {
    name: String,
    total: usize,
    passed: usize,
    failed: usize,
    duration: Duration,
}

// Performance regression types
#[derive(Debug, Serialize, Deserialize)]
struct PerformanceBaseline {
    baselines: HashMap<String, Duration>,
}

impl PerformanceBaseline {
    fn get_baseline_duration(&self, test_name: &str) -> Option<Duration> {
        self.baselines.get(test_name).copied()
    }
}

#[derive(Debug)]
struct PerformanceRegression {
    test_name: String,
    baseline_duration: Duration,
    current_duration: Duration,
    regression_percent: f64,
}
