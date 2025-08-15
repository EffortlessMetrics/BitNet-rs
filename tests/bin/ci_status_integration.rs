#!/usr/bin/env cargo +stable -Zscript
//! CI Status Integration Tool
//!
//! This tool integrates with GitHub Actions to provide unified status reporting
//! for the testing framework workflows.

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio;

#[derive(Parser)]
#[command(name = "ci_status_integration")]
#[command(version = "1.0")]
#[command(about = "Integrate testing framework status with CI systems")]
struct Args {
    /// GitHub repository (owner/repo)
    #[arg(long)]
    repository: String,

    /// GitHub token for API access
    #[arg(long)]
    github_token: String,

    /// Commit SHA to update status for
    #[arg(long)]
    commit_sha: String,

    /// Workflow run ID
    #[arg(long)]
    workflow_run_id: String,

    /// Output directory for status reports
    #[arg(long, default_value = "ci-status")]
    output_dir: PathBuf,

    /// Status context prefix
    #[arg(long, default_value = "bitnet-rs")]
    context_prefix: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct WorkflowStatus {
    name: String,
    status: String,
    conclusion: Option<String>,
    html_url: String,
    created_at: String,
    updated_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct StatusCheck {
    context: String,
    state: String,
    description: String,
    target_url: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CIStatusReport {
    repository: String,
    commit_sha: String,
    workflow_run_id: String,
    timestamp: String,
    overall_status: String,
    workflow_statuses: Vec<WorkflowStatus>,
    status_checks: Vec<StatusCheck>,
    summary: StatusSummary,
}

#[derive(Debug, Serialize, Deserialize)]
struct StatusSummary {
    total_workflows: usize,
    successful_workflows: usize,
    failed_workflows: usize,
    pending_workflows: usize,
    success_rate: f64,
}

struct CIStatusIntegrator {
    args: Args,
    github_client: reqwest::Client,
}

impl CIStatusIntegrator {
    fn new(args: Args) -> Self {
        let github_client = reqwest::Client::builder()
            .user_agent("bitnet-rs-ci-status-integration/1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self { args, github_client }
    }

    async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔄 Starting CI status integration...");

        // Create output directory
        tokio::fs::create_dir_all(&self.args.output_dir).await?;

        // Fetch workflow statuses
        let workflow_statuses = self.fetch_workflow_statuses().await?;

        // Generate status checks
        let status_checks = self.generate_status_checks(&workflow_statuses).await?;

        // Create status report
        let report = self.create_status_report(workflow_statuses, status_checks).await?;

        // Save report
        self.save_status_report(&report).await?;

        // Update GitHub status checks
        self.update_github_status_checks(&report.status_checks).await?;

        println!("✅ CI status integration completed successfully");
        Ok(())
    }

    async fn fetch_workflow_statuses(
        &self,
    ) -> Result<Vec<WorkflowStatus>, Box<dyn std::error::Error>> {
        println!("📡 Fetching workflow statuses from GitHub API...");

        let url = format!(
            "https://api.github.com/repos/{}/actions/runs/{}/jobs",
            self.args.repository, self.args.workflow_run_id
        );

        let response = self
            .github_client
            .get(&url)
            .header("Authorization", format!("token {}", self.args.github_token))
            .header("Accept", "application/vnd.github.v3+json")
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("GitHub API request failed: {}", response.status()).into());
        }

        let jobs_response: serde_json::Value = response.json().await?;
        let jobs = jobs_response["jobs"].as_array().ok_or("Invalid jobs response format")?;

        let mut workflow_statuses = Vec::new();
        for job in jobs {
            let status = WorkflowStatus {
                name: job["name"].as_str().unwrap_or("Unknown").to_string(),
                status: job["status"].as_str().unwrap_or("unknown").to_string(),
                conclusion: job["conclusion"].as_str().map(|s| s.to_string()),
                html_url: job["html_url"].as_str().unwrap_or("").to_string(),
                created_at: job["created_at"].as_str().unwrap_or("").to_string(),
                updated_at: job["updated_at"].as_str().unwrap_or("").to_string(),
            };
            workflow_statuses.push(status);
        }

        println!("📊 Found {} workflow jobs", workflow_statuses.len());
        Ok(workflow_statuses)
    }

    async fn generate_status_checks(
        &self,
        workflow_statuses: &[WorkflowStatus],
    ) -> Result<Vec<StatusCheck>, Box<dyn std::error::Error>> {
        println!("🏷️ Generating status checks...");

        let mut status_checks = Vec::new();

        // Group workflows by category
        let mut categories: HashMap<String, Vec<&WorkflowStatus>> = HashMap::new();

        for status in workflow_statuses {
            let category = self.categorize_workflow(&status.name);
            categories.entry(category).or_default().push(status);
        }

        // Generate status check for each category
        for (category, workflows) in categories {
            let (state, description) = self.calculate_category_status(&workflows);

            let status_check = StatusCheck {
                context: format!("{}/{}", self.args.context_prefix, category),
                state,
                description,
                target_url: format!(
                    "https://github.com/{}/actions/runs/{}",
                    self.args.repository, self.args.workflow_run_id
                ),
            };

            status_checks.push(status_check);
        }

        // Generate overall status check
        let (overall_state, overall_description) = self.calculate_overall_status(workflow_statuses);
        let overall_check = StatusCheck {
            context: format!("{}/overall", self.args.context_prefix),
            state: overall_state,
            description: overall_description,
            target_url: format!(
                "https://github.com/{}/actions/runs/{}",
                self.args.repository, self.args.workflow_run_id
            ),
        };
        status_checks.push(overall_check);

        println!("✅ Generated {} status checks", status_checks.len());
        Ok(status_checks)
    }

    fn categorize_workflow(&self, workflow_name: &str) -> String {
        let name_lower = workflow_name.to_lowercase();

        if name_lower.contains("unit") || name_lower.contains("test") {
            "unit-tests".to_string()
        } else if name_lower.contains("integration") {
            "integration-tests".to_string()
        } else if name_lower.contains("coverage") {
            "coverage".to_string()
        } else if name_lower.contains("crossval") || name_lower.contains("cross-validation") {
            "cross-validation".to_string()
        } else if name_lower.contains("performance") || name_lower.contains("benchmark") {
            "performance".to_string()
        } else if name_lower.contains("cache") || name_lower.contains("optimization") {
            "optimization".to_string()
        } else {
            "other".to_string()
        }
    }

    fn calculate_category_status(&self, workflows: &[&WorkflowStatus]) -> (String, String) {
        let total = workflows.len();
        let completed = workflows.iter().filter(|w| w.status == "completed").count();
        let successful = workflows
            .iter()
            .filter(|w| {
                w.status == "completed" && w.conclusion.as_ref().map_or(false, |c| c == "success")
            })
            .count();
        let failed = workflows
            .iter()
            .filter(|w| {
                w.status == "completed" && w.conclusion.as_ref().map_or(false, |c| c == "failure")
            })
            .count();

        if completed < total {
            ("pending".to_string(), format!("{}/{} workflows running", completed, total))
        } else if failed > 0 {
            ("failure".to_string(), format!("{}/{} workflows failed", failed, total))
        } else if successful == total {
            ("success".to_string(), format!("All {} workflows passed", total))
        } else {
            ("error".to_string(), format!("Unexpected workflow states"))
        }
    }

    fn calculate_overall_status(&self, workflow_statuses: &[WorkflowStatus]) -> (String, String) {
        let total = workflow_statuses.len();
        let completed = workflow_statuses.iter().filter(|w| w.status == "completed").count();
        let successful = workflow_statuses
            .iter()
            .filter(|w| {
                w.status == "completed" && w.conclusion.as_ref().map_or(false, |c| c == "success")
            })
            .count();
        let failed = workflow_statuses
            .iter()
            .filter(|w| {
                w.status == "completed" && w.conclusion.as_ref().map_or(false, |c| c == "failure")
            })
            .count();

        if completed < total {
            ("pending".to_string(), format!("Testing framework running ({}/{})", completed, total))
        } else if failed > 0 {
            ("failure".to_string(), format!("Testing framework failed ({} failures)", failed))
        } else if successful == total {
            ("success".to_string(), "Testing framework passed".to_string())
        } else {
            ("error".to_string(), "Testing framework status unknown".to_string())
        }
    }

    async fn create_status_report(
        &self,
        workflow_statuses: Vec<WorkflowStatus>,
        status_checks: Vec<StatusCheck>,
    ) -> Result<CIStatusReport, Box<dyn std::error::Error>> {
        println!("📋 Creating status report...");

        let successful = workflow_statuses
            .iter()
            .filter(|w| {
                w.status == "completed" && w.conclusion.as_ref().map_or(false, |c| c == "success")
            })
            .count();
        let failed = workflow_statuses
            .iter()
            .filter(|w| {
                w.status == "completed" && w.conclusion.as_ref().map_or(false, |c| c == "failure")
            })
            .count();
        let pending = workflow_statuses.iter().filter(|w| w.status != "completed").count();
        let total = workflow_statuses.len();

        let success_rate = if total > 0 { successful as f64 / total as f64 * 100.0 } else { 0.0 };

        let overall_status = if failed > 0 {
            "failure".to_string()
        } else if pending > 0 {
            "pending".to_string()
        } else {
            "success".to_string()
        };

        let report = CIStatusReport {
            repository: self.args.repository.clone(),
            commit_sha: self.args.commit_sha.clone(),
            workflow_run_id: self.args.workflow_run_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            overall_status,
            workflow_statuses,
            status_checks,
            summary: StatusSummary {
                total_workflows: total,
                successful_workflows: successful,
                failed_workflows: failed,
                pending_workflows: pending,
                success_rate,
            },
        };

        Ok(report)
    }

    async fn save_status_report(
        &self,
        report: &CIStatusReport,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("💾 Saving status report...");

        // Save JSON report
        let json_path = self.args.output_dir.join("ci-status-report.json");
        let json_content = serde_json::to_string_pretty(report)?;
        tokio::fs::write(&json_path, json_content).await?;

        // Save human-readable report
        let md_path = self.args.output_dir.join("ci-status-report.md");
        let md_content = self.generate_markdown_report(report)?;
        tokio::fs::write(&md_path, md_content).await?;

        println!("📄 Reports saved to {}", self.args.output_dir.display());
        Ok(())
    }

    fn generate_markdown_report(
        &self,
        report: &CIStatusReport,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut md = String::new();

        md.push_str("# CI Status Report\n\n");
        md.push_str(&format!("**Repository**: {}\n", report.repository));
        md.push_str(&format!("**Commit**: {}\n", report.commit_sha));
        md.push_str(&format!("**Workflow Run**: {}\n", report.workflow_run_id));
        md.push_str(&format!("**Timestamp**: {}\n\n", report.timestamp));

        // Overall status
        let status_emoji = match report.overall_status.as_str() {
            "success" => "✅",
            "failure" => "❌",
            "pending" => "🔄",
            _ => "⚠️",
        };
        md.push_str(&format!(
            "## {} Overall Status: {}\n\n",
            status_emoji,
            report.overall_status.to_uppercase()
        ));

        // Summary
        md.push_str("## Summary\n\n");
        md.push_str(&format!("- **Total Workflows**: {}\n", report.summary.total_workflows));
        md.push_str(&format!("- **Successful**: {}\n", report.summary.successful_workflows));
        md.push_str(&format!("- **Failed**: {}\n", report.summary.failed_workflows));
        md.push_str(&format!("- **Pending**: {}\n", report.summary.pending_workflows));
        md.push_str(&format!("- **Success Rate**: {:.1}%\n\n", report.summary.success_rate));

        // Status checks
        md.push_str("## Status Checks\n\n");
        md.push_str("| Context | State | Description |\n");
        md.push_str("|---------|-------|-------------|\n");
        for check in &report.status_checks {
            let state_emoji = match check.state.as_str() {
                "success" => "✅",
                "failure" => "❌",
                "pending" => "🔄",
                _ => "⚠️",
            };
            md.push_str(&format!(
                "| {} | {} {} | {} |\n",
                check.context, state_emoji, check.state, check.description
            ));
        }
        md.push_str("\n");

        // Workflow details
        md.push_str("## Workflow Details\n\n");
        md.push_str("| Workflow | Status | Conclusion | Updated |\n");
        md.push_str("|----------|--------|------------|----------|\n");
        for workflow in &report.workflow_statuses {
            let status_emoji = match workflow.status.as_str() {
                "completed" => match workflow.conclusion.as_deref() {
                    Some("success") => "✅",
                    Some("failure") => "❌",
                    _ => "⚠️",
                },
                "in_progress" => "🔄",
                _ => "⏳",
            };
            md.push_str(&format!(
                "| {} | {} {} | {} | {} |\n",
                workflow.name,
                status_emoji,
                workflow.status,
                workflow.conclusion.as_deref().unwrap_or("N/A"),
                workflow.updated_at
            ));
        }

        Ok(md)
    }

    async fn update_github_status_checks(
        &self,
        status_checks: &[StatusCheck],
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔄 Updating GitHub status checks...");

        for check in status_checks {
            let url = format!(
                "https://api.github.com/repos/{}/statuses/{}",
                self.args.repository, self.args.commit_sha
            );

            let payload = serde_json::json!({
                "state": check.state,
                "target_url": check.target_url,
                "description": check.description,
                "context": check.context
            });

            let response = self
                .github_client
                .post(&url)
                .header("Authorization", format!("token {}", self.args.github_token))
                .header("Accept", "application/vnd.github.v3+json")
                .json(&payload)
                .send()
                .await?;

            if response.status().is_success() {
                println!("✅ Updated status check: {}", check.context);
            } else {
                println!(
                    "⚠️ Failed to update status check {}: {}",
                    check.context,
                    response.status()
                );
            }
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let integrator = CIStatusIntegrator::new(args);
    integrator.run().await
}
