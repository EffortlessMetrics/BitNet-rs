use std::collections::HashMap;
use std::fmt::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

use super::errors::{TestError, TestOpResult};
use super::results::TestResult;
use super::units::BYTES_PER_MB;

/// Comprehensive debugging support for the testing framework
pub struct TestDebugger {
    debug_config: DebugConfig,
    session_id: String,
    debug_data: Arc<RwLock<DebugSession>>,
    output_dir: PathBuf,
}

/// Configuration for debugging features
#[derive(Debug, Clone)]
pub struct DebugConfig {
    pub enabled: bool,
    pub capture_stack_traces: bool,
    pub capture_environment: bool,
    pub capture_system_info: bool,
    pub verbose_logging: bool,
    pub save_debug_artifacts: bool,
    pub max_debug_files: usize,
    pub debug_output_dir: PathBuf,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            capture_stack_traces: true,
            capture_environment: true,
            capture_system_info: true,
            verbose_logging: false,
            save_debug_artifacts: true,
            max_debug_files: 100,
            debug_output_dir: PathBuf::from("tests/debug"),
        }
    }
}

/// Debug session data
#[derive(Debug, Default)]
struct DebugSession {
    start_time: Option<SystemTime>,
    test_traces: HashMap<String, TestTrace>,
    error_reports: Vec<ErrorReport>,
    performance_data: Vec<PerformanceSnapshot>,
    system_snapshots: Vec<SystemSnapshot>,
    debug_logs: Vec<DebugLogEntry>,
}

/// Trace information for a single test
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TestTrace {
    pub test_name: String,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub phases: Vec<TestPhase>,
    pub resource_usage: Vec<ResourceSnapshot>,
    pub debug_messages: Vec<String>,
    pub stack_traces: Vec<StackTrace>,
    pub artifacts: Vec<DebugArtifact>,
}

/// Test execution phase
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TestPhase {
    pub name: String,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub status: PhaseStatus,
    pub details: HashMap<String, String>,
}

/// Status of a test phase
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PhaseStatus {
    Started,
    InProgress,
    Completed,
    Failed(String),
    Skipped(String),
}

/// Resource usage snapshot
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourceSnapshot {
    pub timestamp: SystemTime,
    pub memory_usage: u64,
    pub cpu_usage: Option<f64>,
    pub open_files: Option<usize>,
    pub thread_count: Option<usize>,
}

/// Stack trace information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StackTrace {
    pub timestamp: SystemTime,
    pub context: String,
    pub frames: Vec<StackFrame>,
}

/// Single stack frame
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StackFrame {
    pub function: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub column: Option<u32>,
}

/// Debug artifact (file, screenshot, etc.)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DebugArtifact {
    pub name: String,
    pub artifact_type: ArtifactType,
    pub path: PathBuf,
    pub size: u64,
    pub created_at: SystemTime,
    pub description: String,
}

/// Type of debug artifact
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ArtifactType {
    LogFile,
    MemoryDump,
    ConfigSnapshot,
    ErrorReport,
    PerformanceProfile,
    SystemInfo,
    TestOutput,
    Other(String),
}

/// Performance snapshot
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: SystemTime,
    pub test_name: Option<String>,
    pub memory_usage: u64,
    pub cpu_usage: Option<f64>,
    pub duration_since_start: Duration,
    pub active_tests: usize,
}

/// System information snapshot
#[derive(Debug, Clone)]
pub struct SystemSnapshot {
    pub timestamp: SystemTime,
    pub available_memory: u64,
    pub cpu_cores: usize,
    pub load_average: Option<f64>,
    pub disk_space: u64,
    pub network_active: bool,
}

/// Debug log entry
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DebugLogEntry {
    pub timestamp: SystemTime,
    pub level: LogLevel,
    pub component: String,
    pub message: String,
    pub context: HashMap<String, String>,
}

/// Log levels for debugging
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl TestDebugger {
    /// Create a new test debugger
    pub async fn new(config: DebugConfig) -> TestOpResult<Self> {
        let session_id = generate_session_id();
        let output_dir = config.debug_output_dir.join(&session_id);

        // Create debug output directory
        tokio::fs::create_dir_all(&output_dir).await?;

        let debug_session =
            DebugSession { start_time: Some(SystemTime::now()), ..Default::default() };

        let debugger = Self {
            debug_config: config,
            session_id,
            debug_data: Arc::new(RwLock::new(debug_session)),
            output_dir,
        };

        // Initialize system monitoring if enabled
        if debugger.debug_config.capture_system_info {
            debugger.start_system_monitoring().await;
        }

        debugger.log_debug("TestDebugger", "Debug session started", HashMap::new()).await;

        Ok(debugger)
    }

    /// Start debugging a test
    pub async fn start_test_debug(&self, test_name: &str) -> TestOpResult<()> {
        if !self.debug_config.enabled {
            return Ok(());
        }

        let mut debug_data = self.debug_data.write().await;

        let test_trace = TestTrace {
            test_name: test_name.to_string(),
            start_time: SystemTime::now(),
            end_time: None,
            phases: vec![TestPhase {
                name: "initialization".to_string(),
                start_time: SystemTime::now(),
                end_time: None,
                status: PhaseStatus::Started,
                details: HashMap::new(),
            }],
            resource_usage: vec![self.capture_resource_snapshot().await],
            debug_messages: Vec::new(),
            stack_traces: Vec::new(),
            artifacts: Vec::new(),
        };

        debug_data.test_traces.insert(test_name.to_string(), test_trace);

        self.log_debug(
            "TestDebugger",
            &format!("Started debugging test: {}", test_name),
            [("test_name".to_string(), test_name.to_string())].into(),
        )
        .await;

        Ok(())
    }

    /// End debugging a test
    pub async fn end_test_debug(&self, test_name: &str, result: &TestResult) -> TestOpResult<()> {
        if !self.debug_config.enabled {
            return Ok(());
        }

        let mut debug_data = self.debug_data.write().await;

        if let Some(test_trace) = debug_data.test_traces.get_mut(test_name) {
            test_trace.end_time = Some(SystemTime::now());
            test_trace.resource_usage.push(self.capture_resource_snapshot().await);

            // Complete the last phase
            if let Some(last_phase) = test_trace.phases.last_mut() {
                last_phase.end_time = Some(SystemTime::now());
                last_phase.status = if result.is_success() {
                    PhaseStatus::Completed
                } else {
                    PhaseStatus::Failed(
                        result.error.clone().unwrap_or_else(|| "Unknown error".to_string()),
                    )
                };
            }

            // Capture stack trace if test failed
            if !result.is_success() && self.debug_config.capture_stack_traces {
                let stack_trace = self.capture_stack_trace("test_failure").await;
                test_trace.stack_traces.push(stack_trace);
            }

            // Save debug artifacts if enabled
            if self.debug_config.save_debug_artifacts {
                self.save_test_debug_artifacts(test_name, test_trace).await?;
            }
        }

        self.log_debug(
            "TestDebugger",
            &format!(
                "Ended debugging test: {} ({})",
                test_name,
                if result.is_success() { "PASSED" } else { "FAILED" }
            ),
            [
                ("test_name".to_string(), test_name.to_string()),
                (
                    "status".to_string(),
                    if result.is_success() { "passed".to_string() } else { "failed".to_string() },
                ),
                ("duration".to_string(), format!("{:?}", result.duration)),
            ]
            .into(),
        )
        .await;

        Ok(())
    }

    /// Start a test phase
    pub async fn start_phase(&self, test_name: &str, phase_name: &str) -> TestOpResult<()> {
        if !self.debug_config.enabled {
            return Ok(());
        }

        let mut debug_data = self.debug_data.write().await;

        if let Some(test_trace) = debug_data.test_traces.get_mut(test_name) {
            // Complete previous phase
            if let Some(last_phase) = test_trace.phases.last_mut()
                && last_phase.end_time.is_none()
            {
                last_phase.end_time = Some(SystemTime::now());
                last_phase.status = PhaseStatus::Completed;
            }

            // Start new phase
            let phase = TestPhase {
                name: phase_name.to_string(),
                start_time: SystemTime::now(),
                end_time: None,
                status: PhaseStatus::Started,
                details: HashMap::new(),
            };

            test_trace.phases.push(phase);
            test_trace.resource_usage.push(self.capture_resource_snapshot().await);
        }

        self.log_debug(
            "TestDebugger",
            &format!("Started phase '{}' for test '{}'", phase_name, test_name),
            [
                ("test_name".to_string(), test_name.to_string()),
                ("phase".to_string(), phase_name.to_string()),
            ]
            .into(),
        )
        .await;

        Ok(())
    }

    /// End a test phase
    pub async fn end_phase(
        &self,
        test_name: &str,
        phase_name: &str,
        success: bool,
        details: Option<HashMap<String, String>>,
    ) -> TestOpResult<()> {
        if !self.debug_config.enabled {
            return Ok(());
        }

        let mut debug_data = self.debug_data.write().await;

        if let Some(test_trace) = debug_data.test_traces.get_mut(test_name)
            && let Some(phase) = test_trace
                .phases
                .iter_mut()
                .rev()
                .find(|p| p.name == phase_name && p.end_time.is_none())
        {
            phase.end_time = Some(SystemTime::now());
            phase.status = if success {
                PhaseStatus::Completed
            } else {
                PhaseStatus::Failed("Phase failed".to_string())
            };

            if let Some(details) = details {
                phase.details.extend(details);
            }
        }

        Ok(())
    }

    /// Add a debug message for a test
    pub async fn add_debug_message(&self, test_name: &str, message: &str) -> TestOpResult<()> {
        if !self.debug_config.enabled {
            return Ok(());
        }

        let mut debug_data = self.debug_data.write().await;

        if let Some(test_trace) = debug_data.test_traces.get_mut(test_name) {
            test_trace.debug_messages.push(format!(
                "[{}] {}",
                SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs(),
                message
            ));
        }

        self.log_debug(
            "TestDebugger",
            message,
            [("test_name".to_string(), test_name.to_string())].into(),
        )
        .await;

        Ok(())
    }

    /// Capture and record an error for debugging
    pub async fn capture_error(
        &self,
        test_name: Option<&str>,
        error: &TestError,
    ) -> TestOpResult<()> {
        if !self.debug_config.enabled {
            return Ok(());
        }

        let error_report = error.create_error_report();

        // Save error report to file
        let error_file = self.output_dir.join(format!(
            "error_{}_{}.json",
            test_name.unwrap_or("unknown"),
            SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
        ));

        error_report.save_to_file(&error_file).await?;

        // Add to debug session
        let mut debug_data = self.debug_data.write().await;
        debug_data.error_reports.push(error_report.clone());

        // Add to test trace if test name provided
        if let Some(test_name) = test_name
            && let Some(test_trace) = debug_data.test_traces.get_mut(test_name)
        {
            test_trace.artifacts.push(DebugArtifact {
                name: "error_report".to_string(),
                artifact_type: ArtifactType::ErrorReport,
                path: error_file,
                size: 0, // Will be filled when file is written
                created_at: SystemTime::now(),
                description: format!("Error report for {}: {}", test_name, error),
            });

            // Capture stack trace if enabled
            if self.debug_config.capture_stack_traces {
                let stack_trace = self.capture_stack_trace("error_capture").await;
                test_trace.stack_traces.push(stack_trace);
            }
        }

        self.log_debug(
            "TestDebugger",
            &format!("Captured error: {}", error),
            [
                ("test_name".to_string(), test_name.unwrap_or("unknown").to_string()),
                ("error_type".to_string(), error.category().to_string()),
                ("severity".to_string(), error.severity().to_string()),
            ]
            .into(),
        )
        .await;

        Ok(())
    }

    /// Generate a comprehensive debug report
    pub async fn generate_debug_report(&self) -> TestOpResult<DebugReport> {
        let debug_data = self.debug_data.read().await;

        let mut report = DebugReport {
            session_id: self.session_id.clone(),
            start_time: debug_data.start_time.unwrap_or_else(SystemTime::now),
            end_time: SystemTime::now(),
            total_tests: debug_data.test_traces.len(),
            failed_tests: debug_data
                .test_traces
                .values()
                .filter(|t| t.phases.iter().any(|p| matches!(p.status, PhaseStatus::Failed(_))))
                .count(),
            error_count: debug_data.error_reports.len(),
            test_summaries: Vec::new(),
            performance_summary: self.generate_performance_summary(&debug_data).await,
            system_summary: self.generate_system_summary(&debug_data).await,
            recommendations: self.generate_recommendations(&debug_data).await,
            artifacts: self.collect_all_artifacts(&debug_data).await,
        };

        // Generate test summaries
        for (test_name, test_trace) in &debug_data.test_traces {
            let summary = TestDebugSummary {
                test_name: test_name.clone(),
                duration: test_trace
                    .end_time
                    .and_then(|end| end.duration_since(test_trace.start_time).ok())
                    .unwrap_or_default(),
                phases: test_trace.phases.len(),
                failed_phases: test_trace
                    .phases
                    .iter()
                    .filter(|p| matches!(p.status, PhaseStatus::Failed(_)))
                    .count(),
                peak_memory: test_trace
                    .resource_usage
                    .iter()
                    .map(|r| r.memory_usage)
                    .max()
                    .unwrap_or(0),
                debug_messages: test_trace.debug_messages.len(),
                stack_traces: test_trace.stack_traces.len(),
                artifacts: test_trace.artifacts.len(),
                issues: self.analyze_test_issues(test_trace).await,
            };

            report.test_summaries.push(summary);
        }

        Ok(report)
    }

    /// Save debug report to file
    pub async fn save_debug_report(&self, report: &DebugReport) -> TestOpResult<PathBuf> {
        let report_file = self.output_dir.join("debug_report.json");

        let json = serde_json::to_string_pretty(report).map_err(|e| {
            TestError::execution(format!("Failed to serialize debug report: {}", e))
        })?;

        tokio::fs::write(&report_file, json).await?;

        // Also generate human-readable summary
        let summary_file = self.output_dir.join("debug_summary.md");
        let summary = self.generate_human_readable_summary(report).await;
        tokio::fs::write(&summary_file, summary).await?;

        Ok(report_file)
    }

    /// Generate troubleshooting guide based on captured data
    pub async fn generate_troubleshooting_guide(&self) -> TestOpResult<String> {
        let debug_data = self.debug_data.read().await;
        let mut guide = String::new();

        writeln!(guide, "# Troubleshooting Guide")?;
        writeln!(guide, "Generated: {:?}", SystemTime::now())?;
        writeln!(guide, "Session: {}", self.session_id)?;
        writeln!(guide)?;

        // Common issues section
        writeln!(guide, "## Common Issues Detected")?;

        let mut issues_found = false;

        // Check for timeout issues
        let timeout_tests: Vec<_> = debug_data
            .test_traces
            .values()
            .filter(|t| {
                t.phases.iter().any(
                    |p| matches!(p.status, PhaseStatus::Failed(ref msg) if msg.contains("timeout")),
                )
            })
            .collect();

        if !timeout_tests.is_empty() {
            issues_found = true;
            writeln!(guide, "### Timeout Issues ({} tests)", timeout_tests.len())?;
            writeln!(guide, "Tests that experienced timeouts:")?;
            for test in timeout_tests {
                writeln!(guide, "- {}", test.test_name)?;
            }
            writeln!(guide, "\n**Recommendations:**")?;
            writeln!(guide, "1. Increase test timeout values")?;
            writeln!(guide, "2. Check for infinite loops or blocking operations")?;
            writeln!(guide, "3. Review resource contention issues")?;
            writeln!(guide)?;
        }

        // Check for memory issues
        let high_memory_tests: Vec<_> = debug_data.test_traces.values()
            .filter(|t| t.resource_usage.iter().any(|r| r.memory_usage > BYTES_PER_MB * 1024)) // > 1GB
            .collect();

        if !high_memory_tests.is_empty() {
            issues_found = true;
            writeln!(guide, "### High Memory Usage ({} tests)", high_memory_tests.len())?;
            writeln!(guide, "Tests with high memory usage:")?;
            for test in high_memory_tests {
                let peak = test.resource_usage.iter().map(|r| r.memory_usage).max().unwrap_or(0);
                writeln!(guide, "- {}: {} MB", test.test_name, peak / (BYTES_PER_MB))?;
            }
            writeln!(guide, "\n**Recommendations:**")?;
            writeln!(guide, "1. Review memory allocation patterns")?;
            writeln!(guide, "2. Check for memory leaks")?;
            writeln!(guide, "3. Consider reducing test data size")?;
            writeln!(guide)?;
        }

        // Check for frequent errors
        let error_patterns = self.analyze_error_patterns(&debug_data).await;
        if !error_patterns.is_empty() {
            issues_found = true;
            writeln!(guide, "### Frequent Error Patterns")?;
            for (pattern, count) in error_patterns {
                writeln!(guide, "- {}: {} occurrences", pattern, count)?;
            }
            writeln!(guide)?;
        }

        if !issues_found {
            writeln!(guide, "No common issues detected in this session.")?;
        }

        // General debugging tips
        writeln!(guide, "## General Debugging Tips")?;
        writeln!(
            guide,
            "1. **Check logs**: Review test execution logs for detailed error messages"
        )?;
        writeln!(
            guide,
            "2. **Run in isolation**: Execute failing tests individually to isolate issues"
        )?;
        writeln!(
            guide,
            "3. **Check environment**: Verify environment variables and system configuration"
        )?;
        writeln!(
            guide,
            "4. **Resource monitoring**: Monitor CPU, memory, and disk usage during tests"
        )?;
        writeln!(
            guide,
            "5. **Incremental debugging**: Add debug prints to narrow down the issue location"
        )?;
        writeln!(guide)?;

        // Session-specific information
        writeln!(guide, "## Session Information")?;
        writeln!(guide, "- Total tests: {}", debug_data.test_traces.len())?;
        writeln!(
            guide,
            "- Failed tests: {}",
            debug_data
                .test_traces
                .values()
                .filter(|t| { t.phases.iter().any(|p| matches!(p.status, PhaseStatus::Failed(_))) })
                .count()
        )?;
        writeln!(guide, "- Error reports: {}", debug_data.error_reports.len())?;
        writeln!(guide, "- Debug artifacts: {}", self.output_dir.display())?;

        Ok(guide)
    }

    // Private helper methods

    async fn start_system_monitoring(&self) {
        let debug_data = Arc::clone(&self.debug_data);
        let config = self.debug_config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));

            loop {
                interval.tick().await;

                if !config.enabled {
                    break;
                }

                let snapshot = SystemSnapshot {
                    timestamp: SystemTime::now(),
                    available_memory: get_available_memory(),
                    cpu_cores: num_cpus::get(),
                    load_average: get_load_average(),
                    disk_space: get_available_disk_space(),
                    network_active: false, // Simplified
                };

                let mut data = debug_data.write().await;
                data.system_snapshots.push(snapshot);

                // Keep only recent snapshots to avoid memory bloat
                if data.system_snapshots.len() > 1000 {
                    data.system_snapshots.drain(0..500);
                }
            }
        });
    }

    async fn capture_resource_snapshot(&self) -> ResourceSnapshot {
        ResourceSnapshot {
            timestamp: SystemTime::now(),
            memory_usage: super::utils::get_memory_usage(),
            cpu_usage: None,    // Could be implemented with system monitoring
            open_files: None,   // Could be implemented with system monitoring
            thread_count: None, // Could be implemented with system monitoring
        }
    }

    async fn capture_stack_trace(&self, context: &str) -> StackTrace {
        StackTrace {
            timestamp: SystemTime::now(),
            context: context.to_string(),
            frames: vec![], // Simplified - would use backtrace crate in production
        }
    }

    async fn save_test_debug_artifacts(
        &self,
        test_name: &str,
        test_trace: &TestTrace,
    ) -> TestOpResult<()> {
        let test_dir = self.output_dir.join(format!("test_{}", test_name));
        tokio::fs::create_dir_all(&test_dir).await?;

        // Save test trace
        let trace_file = test_dir.join("trace.json");
        let trace_json = serde_json::to_string_pretty(test_trace)
            .map_err(|e| TestError::execution(format!("Failed to serialize test trace: {}", e)))?;
        tokio::fs::write(trace_file, trace_json).await?;

        // Save debug messages
        if !test_trace.debug_messages.is_empty() {
            let messages_file = test_dir.join("debug_messages.txt");
            let messages = test_trace.debug_messages.join("\n");
            tokio::fs::write(messages_file, messages).await?;
        }

        Ok(())
    }

    async fn log_debug(&self, component: &str, message: &str, context: HashMap<String, String>) {
        if !self.debug_config.verbose_logging {
            return;
        }

        let entry = DebugLogEntry {
            timestamp: SystemTime::now(),
            level: LogLevel::Debug,
            component: component.to_string(),
            message: message.to_string(),
            context,
        };

        let mut debug_data = self.debug_data.write().await;
        debug_data.debug_logs.push(entry);

        // Also print to console if verbose
        if self.debug_config.verbose_logging {
            println!("[DEBUG] {}: {}", component, message);
        }
    }

    async fn generate_performance_summary(&self, debug_data: &DebugSession) -> PerformanceSummary {
        let total_duration = debug_data
            .start_time
            .and_then(|start| SystemTime::now().duration_since(start).ok())
            .unwrap_or_default();

        let peak_memory =
            debug_data.performance_data.iter().map(|p| p.memory_usage).max().unwrap_or(0);

        let avg_test_duration = if !debug_data.test_traces.is_empty() {
            let total_test_time: Duration = debug_data
                .test_traces
                .values()
                .filter_map(|t| t.end_time.and_then(|end| end.duration_since(t.start_time).ok()))
                .sum();
            total_test_time / debug_data.test_traces.len() as u32
        } else {
            Duration::ZERO
        };

        PerformanceSummary {
            total_duration,
            peak_memory,
            average_test_duration: avg_test_duration,
            slowest_tests: self.find_slowest_tests(debug_data, 5).await,
        }
    }

    async fn generate_system_summary(&self, debug_data: &DebugSession) -> SystemSummary {
        let latest_snapshot = debug_data.system_snapshots.last();

        SystemSummary {
            cpu_cores: latest_snapshot.map(|s| s.cpu_cores).unwrap_or(0),
            available_memory: latest_snapshot.map(|s| s.available_memory).unwrap_or(0),
            disk_space: latest_snapshot.map(|s| s.disk_space).unwrap_or(0),
            load_average: latest_snapshot.and_then(|s| s.load_average),
        }
    }

    async fn generate_recommendations(&self, debug_data: &DebugSession) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for common issues and generate recommendations
        let failed_tests = debug_data
            .test_traces
            .values()
            .filter(|t| t.phases.iter().any(|p| matches!(p.status, PhaseStatus::Failed(_))))
            .count();

        if failed_tests > 0 {
            recommendations
                .push(format!("Review {} failed tests for common error patterns", failed_tests));
        }

        let high_memory_tests = debug_data
            .test_traces
            .values()
            .filter(|t| t.resource_usage.iter().any(|r| r.memory_usage > 512 * BYTES_PER_MB))
            .count();

        if high_memory_tests > 0 {
            recommendations.push(format!(
                "Investigate {} tests with high memory usage (>512MB)",
                high_memory_tests
            ));
        }

        if debug_data.error_reports.len() > 10 {
            recommendations
                .push("High error count detected - consider reviewing test stability".to_string());
        }

        recommendations
    }

    async fn collect_all_artifacts(&self, debug_data: &DebugSession) -> Vec<DebugArtifact> {
        let mut artifacts = Vec::new();

        for test_trace in debug_data.test_traces.values() {
            artifacts.extend(test_trace.artifacts.clone());
        }

        artifacts
    }

    async fn analyze_test_issues(&self, test_trace: &TestTrace) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for failed phases
        for phase in &test_trace.phases {
            if let PhaseStatus::Failed(msg) = &phase.status {
                issues.push(format!("Phase '{}' failed: {}", phase.name, msg));
            }
        }

        // Check for high memory usage
        if let Some(peak) = test_trace.resource_usage.iter().map(|r| r.memory_usage).max()
            && peak > BYTES_PER_MB * 1024
        {
            // > 1GB
            issues.push(format!("High memory usage: {} MB", peak / (BYTES_PER_MB)));
        }

        // Check for long duration
        if let Some(duration) =
            test_trace.end_time.and_then(|end| end.duration_since(test_trace.start_time).ok())
            && duration > Duration::from_secs(300)
        {
            // > 5 minutes
            issues.push(format!("Long execution time: {:?}", duration));
        }

        issues
    }

    async fn analyze_error_patterns(&self, debug_data: &DebugSession) -> HashMap<String, usize> {
        let mut patterns = HashMap::new();

        for error_report in &debug_data.error_reports {
            let pattern = error_report.error_category.clone();
            *patterns.entry(pattern).or_insert(0) += 1;
        }

        patterns
    }

    async fn find_slowest_tests(
        &self,
        debug_data: &DebugSession,
        count: usize,
    ) -> Vec<(String, Duration)> {
        let mut test_durations: Vec<_> = debug_data
            .test_traces
            .iter()
            .filter_map(|(name, trace)| {
                trace
                    .end_time
                    .and_then(|end| end.duration_since(trace.start_time).ok())
                    .map(|duration| (name.clone(), duration))
            })
            .collect();

        test_durations.sort_by(|a, b| b.1.cmp(&a.1));
        test_durations.truncate(count);
        test_durations
    }

    async fn generate_human_readable_summary(&self, report: &DebugReport) -> String {
        let mut summary = String::new();

        writeln!(summary, "# Debug Session Summary").unwrap();
        writeln!(summary, "Session ID: {}", report.session_id).unwrap();
        writeln!(
            summary,
            "Duration: {:?}",
            report.end_time.duration_since(report.start_time).unwrap_or_default()
        )
        .unwrap();
        writeln!(summary).unwrap();

        writeln!(summary, "## Overview").unwrap();
        writeln!(summary, "- Total tests: {}", report.total_tests).unwrap();
        writeln!(summary, "- Failed tests: {}", report.failed_tests).unwrap();
        writeln!(summary, "- Error reports: {}", report.error_count).unwrap();
        writeln!(
            summary,
            "- Success rate: {:.1}%",
            if report.total_tests > 0 {
                ((report.total_tests - report.failed_tests) as f64 / report.total_tests as f64)
                    * 100.0
            } else {
                0.0
            }
        )
        .unwrap();
        writeln!(summary).unwrap();

        writeln!(summary, "## Performance").unwrap();
        writeln!(
            summary,
            "- Peak memory: {} MB",
            report.performance_summary.peak_memory / (BYTES_PER_MB)
        )
        .unwrap();
        writeln!(
            summary,
            "- Average test duration: {:?}",
            report.performance_summary.average_test_duration
        )
        .unwrap();
        writeln!(summary).unwrap();

        if !report.performance_summary.slowest_tests.is_empty() {
            writeln!(summary, "### Slowest Tests").unwrap();
            for (test_name, duration) in &report.performance_summary.slowest_tests {
                writeln!(summary, "- {}: {:?}", test_name, duration).unwrap();
            }
            writeln!(summary).unwrap();
        }

        if !report.recommendations.is_empty() {
            writeln!(summary, "## Recommendations").unwrap();
            for (i, rec) in report.recommendations.iter().enumerate() {
                writeln!(summary, "{}. {}", i + 1, rec).unwrap();
            }
        }

        summary
    }
}

/// Complete debug report
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DebugReport {
    pub session_id: String,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub total_tests: usize,
    pub failed_tests: usize,
    pub error_count: usize,
    pub test_summaries: Vec<TestDebugSummary>,
    pub performance_summary: PerformanceSummary,
    pub system_summary: SystemSummary,
    pub recommendations: Vec<String>,
    pub artifacts: Vec<DebugArtifact>,
}

/// Summary of a single test's debug information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TestDebugSummary {
    pub test_name: String,
    pub duration: Duration,
    pub phases: usize,
    pub failed_phases: usize,
    pub peak_memory: u64,
    pub debug_messages: usize,
    pub stack_traces: usize,
    pub artifacts: usize,
    pub issues: Vec<String>,
}

/// Performance summary
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceSummary {
    pub total_duration: Duration,
    pub peak_memory: u64,
    pub average_test_duration: Duration,
    pub slowest_tests: Vec<(String, Duration)>,
}

/// System summary
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemSummary {
    pub cpu_cores: usize,
    pub available_memory: u64,
    pub disk_space: u64,
    pub load_average: Option<f64>,
}

// Helper functions

fn generate_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    format!("debug_session_{}", timestamp)
}

fn get_available_memory() -> u64 {
    // Simplified implementation - would use proper system info in production
    0
}

fn get_load_average() -> Option<f64> {
    // Simplified implementation - would use proper system info in production
    None
}

fn get_available_disk_space() -> u64 {
    // Simplified implementation - would use proper system info in production
    0
}

// Re-export for convenience
pub use super::errors::{ErrorDebugInfo, ErrorReport, TroubleshootingStep};
