use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{Level, Subscriber};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer,
};

use crate::{
    config::TestConfig,
    errors::{TestError, TestResult},
};

/// Initialize logging for the test framework
pub fn init_logging(config: &TestConfig) -> TestResult<()> {
    let log_level = parse_log_level(&config.log_level)?;

    // Create environment filter
    let env_filter = EnvFilter::builder()
        .with_default_directive(log_level.into())
        .from_env_lossy()
        .add_directive("bitnet=debug".parse().unwrap()) // Always debug BitNet crates
        .add_directive("bitnet_tests=debug".parse().unwrap()); // Always debug test framework

    // Create console layer
    let console_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_span_events(FmtSpan::CLOSE)
        .with_filter(env_filter.clone());

    // Create file layer if output directory exists
    let file_layer = if config.reporting.output_dir.exists() {
        let log_file = config.reporting.output_dir.join("test-execution.log");
        let file = File::create(&log_file).map_err(|e| {
            TestError::config(format!("Failed to create log file {:?}: {}", log_file, e))
        })?;

        Some(
            fmt::layer()
                .with_writer(Arc::new(file))
                .with_ansi(false) // No ANSI colors in file
                .with_target(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_span_events(FmtSpan::FULL)
                .with_filter(env_filter),
        )
    } else {
        None
    };

    // Initialize subscriber
    let subscriber = tracing_subscriber::registry().with(console_layer);

    if let Some(file_layer) = file_layer {
        subscriber.with(file_layer).init();
    } else {
        subscriber.init();
    }

    tracing::info!("Logging initialized with level: {}", config.log_level);
    Ok(())
}

/// Parse log level string into tracing Level
fn parse_log_level(level_str: &str) -> TestResult<Level> {
    match level_str.to_lowercase().as_str() {
        "trace" => Ok(Level::TRACE),
        "debug" => Ok(Level::DEBUG),
        "info" => Ok(Level::INFO),
        "warn" => Ok(Level::WARN),
        "error" => Ok(Level::ERROR),
        _ => Err(TestError::config(format!(
            "Invalid log level: {}",
            level_str
        ))),
    }
}

/// Test execution tracer for detailed debugging
pub struct TestTracer {
    traces: Arc<RwLock<HashMap<String, Vec<TraceEvent>>>>,
    enabled: bool,
}

impl TestTracer {
    /// Create a new test tracer
    pub fn new(enabled: bool) -> Self {
        Self {
            traces: Arc::new(RwLock::new(HashMap::new())),
            enabled,
        }
    }

    /// Start tracing for a test
    pub async fn start_trace(&self, test_name: &str) {
        if !self.enabled {
            return;
        }

        let mut traces = self.traces.write().await;
        traces.insert(test_name.to_string(), Vec::new());

        tracing::debug!("Started tracing for test: {}", test_name);
    }

    /// Add a trace event
    pub async fn trace_event(&self, test_name: &str, event: TraceEvent) {
        if !self.enabled {
            return;
        }

        let mut traces = self.traces.write().await;
        if let Some(test_traces) = traces.get_mut(test_name) {
            test_traces.push(event);
        }
    }

    /// Get traces for a test
    pub async fn get_traces(&self, test_name: &str) -> Vec<TraceEvent> {
        let traces = self.traces.read().await;
        traces.get(test_name).cloned().unwrap_or_default()
    }

    /// Clear traces for a test
    pub async fn clear_traces(&self, test_name: &str) {
        let mut traces = self.traces.write().await;
        traces.remove(test_name);
    }

    /// Get all traces
    pub async fn get_all_traces(&self) -> HashMap<String, Vec<TraceEvent>> {
        self.traces.read().await.clone()
    }

    /// Clear all traces
    pub async fn clear_all_traces(&self) {
        let mut traces = self.traces.write().await;
        traces.clear();
    }
}

/// A single trace event
#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub timestamp: std::time::SystemTime,
    pub event_type: TraceEventType,
    pub message: String,
    pub metadata: HashMap<String, String>,
}

impl TraceEvent {
    /// Create a new trace event
    pub fn new(event_type: TraceEventType, message: String) -> Self {
        Self {
            timestamp: std::time::SystemTime::now(),
            event_type,
            message,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the trace event
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Types of trace events
#[derive(Debug, Clone)]
pub enum TraceEventType {
    TestStart,
    TestEnd,
    Setup,
    Execution,
    Cleanup,
    Assertion,
    Error,
    Warning,
    Info,
    Debug,
    MemoryUsage,
    Performance,
    Custom(String),
}

impl std::fmt::Display for TraceEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TestStart => write!(f, "TEST_START"),
            Self::TestEnd => write!(f, "TEST_END"),
            Self::Setup => write!(f, "SETUP"),
            Self::Execution => write!(f, "EXECUTION"),
            Self::Cleanup => write!(f, "CLEANUP"),
            Self::Assertion => write!(f, "ASSERTION"),
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
            Self::Info => write!(f, "INFO"),
            Self::Debug => write!(f, "DEBUG"),
            Self::MemoryUsage => write!(f, "MEMORY"),
            Self::Performance => write!(f, "PERFORMANCE"),
            Self::Custom(name) => write!(f, "CUSTOM_{}", name.to_uppercase()),
        }
    }
}

/// Debug context for test execution
pub struct DebugContext {
    test_name: String,
    tracer: Arc<TestTracer>,
    artifacts: Arc<RwLock<Vec<DebugArtifact>>>,
    start_time: std::time::Instant,
}

impl DebugContext {
    /// Create a new debug context
    pub fn new(test_name: String, tracer: Arc<TestTracer>) -> Self {
        Self {
            test_name,
            tracer,
            artifacts: Arc::new(RwLock::new(Vec::new())),
            start_time: std::time::Instant::now(),
        }
    }

    /// Log a trace event
    pub async fn trace(&self, event_type: TraceEventType, message: String) {
        let event = TraceEvent::new(event_type, message);
        self.tracer.trace_event(&self.test_name, event).await;
    }

    /// Log a trace event with metadata
    pub async fn trace_with_metadata(
        &self,
        event_type: TraceEventType,
        message: String,
        metadata: HashMap<String, String>,
    ) {
        let event = TraceEvent {
            timestamp: std::time::SystemTime::now(),
            event_type,
            message,
            metadata,
        };
        self.tracer.trace_event(&self.test_name, event).await;
    }

    /// Add a debug artifact
    pub async fn add_artifact(&self, artifact: DebugArtifact) {
        let mut artifacts = self.artifacts.write().await;
        artifacts.push(artifact);
    }

    /// Get all debug artifacts
    pub async fn get_artifacts(&self) -> Vec<DebugArtifact> {
        self.artifacts.read().await.clone()
    }

    /// Get elapsed time since context creation
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Create a scoped debug context for a sub-operation
    pub fn scope(&self, scope_name: &str) -> ScopedDebugContext {
        ScopedDebugContext::new(
            format!("{}::{}", self.test_name, scope_name),
            Arc::clone(&self.tracer),
        )
    }
}

/// Scoped debug context for sub-operations
pub struct ScopedDebugContext {
    scope_name: String,
    tracer: Arc<TestTracer>,
    start_time: std::time::Instant,
}

impl ScopedDebugContext {
    fn new(scope_name: String, tracer: Arc<TestTracer>) -> Self {
        Self {
            scope_name,
            tracer,
            start_time: std::time::Instant::now(),
        }
    }

    /// Log a trace event in this scope
    pub async fn trace(&self, event_type: TraceEventType, message: String) {
        let scoped_message = format!("[{}] {}", self.scope_name, message);
        let event = TraceEvent::new(event_type, scoped_message);
        self.tracer.trace_event(&self.scope_name, event).await;
    }

    /// Get elapsed time in this scope
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}

impl Drop for ScopedDebugContext {
    fn drop(&mut self) {
        let duration = self.elapsed();
        tracing::debug!("Scope '{}' completed in {:?}", self.scope_name, duration);
    }
}

/// Debug artifact (file, data, or other debugging information)
#[derive(Debug, Clone)]
pub struct DebugArtifact {
    pub name: String,
    pub artifact_type: ArtifactType,
    pub path: Option<PathBuf>,
    pub content: Option<String>,
    pub metadata: HashMap<String, String>,
    pub created_at: std::time::SystemTime,
}

impl DebugArtifact {
    /// Create a file artifact
    pub fn file<S: Into<String>>(name: S, path: PathBuf) -> Self {
        Self {
            name: name.into(),
            artifact_type: ArtifactType::File,
            path: Some(path),
            content: None,
            metadata: HashMap::new(),
            created_at: std::time::SystemTime::now(),
        }
    }

    /// Create a text artifact
    pub fn text<S1: Into<String>, S2: Into<String>>(name: S1, content: S2) -> Self {
        Self {
            name: name.into(),
            artifact_type: ArtifactType::Text,
            path: None,
            content: Some(content.into()),
            metadata: HashMap::new(),
            created_at: std::time::SystemTime::now(),
        }
    }

    /// Create a JSON artifact
    pub fn json<S: Into<String>, T: serde::Serialize>(name: S, data: &T) -> TestResult<Self> {
        let content = serde_json::to_string_pretty(data)?;
        Ok(Self {
            name: name.into(),
            artifact_type: ArtifactType::Json,
            path: None,
            content: Some(content),
            metadata: HashMap::new(),
            created_at: std::time::SystemTime::now(),
        })
    }

    /// Add metadata to the artifact
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Types of debug artifacts
#[derive(Debug, Clone)]
pub enum ArtifactType {
    File,
    Text,
    Json,
    Binary,
    Image,
    Log,
    Trace,
    Performance,
    Memory,
    Custom(String),
}

/// Performance profiler for test operations
pub struct PerformanceProfiler {
    enabled: bool,
    samples: Arc<RwLock<Vec<PerformanceSample>>>,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            samples: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start profiling an operation
    pub async fn start_operation(&self, name: String) -> OperationProfiler {
        OperationProfiler::new(name, Arc::clone(&self.samples), self.enabled)
    }

    /// Get all performance samples
    pub async fn get_samples(&self) -> Vec<PerformanceSample> {
        self.samples.read().await.clone()
    }

    /// Clear all samples
    pub async fn clear_samples(&self) {
        let mut samples = self.samples.write().await;
        samples.clear();
    }

    /// Get performance summary
    pub async fn get_summary(&self) -> PerformanceSummary {
        let samples = self.samples.read().await;
        PerformanceSummary::from_samples(&samples)
    }
}

/// Profiler for a single operation
pub struct OperationProfiler {
    name: String,
    start_time: std::time::Instant,
    samples: Arc<RwLock<Vec<PerformanceSample>>>,
    enabled: bool,
}

impl OperationProfiler {
    fn new(name: String, samples: Arc<RwLock<Vec<PerformanceSample>>>, enabled: bool) -> Self {
        Self {
            name,
            start_time: std::time::Instant::now(),
            samples,
            enabled,
        }
    }

    /// Finish profiling and record the sample
    pub async fn finish(self) {
        if !self.enabled {
            return;
        }

        let duration = self.start_time.elapsed();
        let sample = PerformanceSample {
            operation: self.name,
            duration,
            memory_delta: 0, // TODO: Implement memory tracking
            timestamp: std::time::SystemTime::now(),
        };

        let mut samples = self.samples.write().await;
        samples.push(sample);
    }
}

/// A single performance sample
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    pub operation: String,
    pub duration: std::time::Duration,
    pub memory_delta: i64,
    pub timestamp: std::time::SystemTime,
}

/// Summary of performance data
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub total_operations: usize,
    pub total_duration: std::time::Duration,
    pub average_duration: std::time::Duration,
    pub min_duration: std::time::Duration,
    pub max_duration: std::time::Duration,
    pub operations_by_type: HashMap<String, usize>,
}

impl PerformanceSummary {
    fn from_samples(samples: &[PerformanceSample]) -> Self {
        if samples.is_empty() {
            return Self {
                total_operations: 0,
                total_duration: std::time::Duration::ZERO,
                average_duration: std::time::Duration::ZERO,
                min_duration: std::time::Duration::ZERO,
                max_duration: std::time::Duration::ZERO,
                operations_by_type: HashMap::new(),
            };
        }

        let total_operations = samples.len();
        let total_duration: std::time::Duration = samples.iter().map(|s| s.duration).sum();
        let average_duration = total_duration / total_operations as u32;
        let min_duration = samples.iter().map(|s| s.duration).min().unwrap();
        let max_duration = samples.iter().map(|s| s.duration).max().unwrap();

        let mut operations_by_type = HashMap::new();
        for sample in samples {
            *operations_by_type
                .entry(sample.operation.clone())
                .or_insert(0) += 1;
        }

        Self {
            total_operations,
            total_duration,
            average_duration,
            min_duration,
            max_duration,
            operations_by_type,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TestConfig;

    #[test]
    fn test_parse_log_level() {
        assert!(matches!(parse_log_level("trace").unwrap(), Level::TRACE));
        assert!(matches!(parse_log_level("debug").unwrap(), Level::DEBUG));
        assert!(matches!(parse_log_level("info").unwrap(), Level::INFO));
        assert!(matches!(parse_log_level("warn").unwrap(), Level::WARN));
        assert!(matches!(parse_log_level("error").unwrap(), Level::ERROR));

        assert!(parse_log_level("invalid").is_err());
    }

    #[tokio::test]
    async fn test_test_tracer() {
        let tracer = TestTracer::new(true);
        let test_name = "test_example";

        tracer.start_trace(test_name).await;

        let event = TraceEvent::new(TraceEventType::TestStart, "Test started".to_string());
        tracer.trace_event(test_name, event).await;

        let traces = tracer.get_traces(test_name).await;
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].message, "Test started");

        tracer.clear_traces(test_name).await;
        let traces = tracer.get_traces(test_name).await;
        assert!(traces.is_empty());
    }

    #[tokio::test]
    async fn test_debug_context() {
        let tracer = Arc::new(TestTracer::new(true));
        let context = DebugContext::new("test_debug".to_string(), tracer);

        context
            .trace(TraceEventType::Info, "Debug message".to_string())
            .await;

        let artifact = DebugArtifact::text("test_output", "Some test output");
        context.add_artifact(artifact).await;

        let artifacts = context.get_artifacts().await;
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].name, "test_output");
    }

    #[tokio::test]
    async fn test_performance_profiler() {
        let profiler = PerformanceProfiler::new(true);

        let op_profiler = profiler.start_operation("test_operation".to_string()).await;
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        op_profiler.finish().await;

        let samples = profiler.get_samples().await;
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].operation, "test_operation");
        assert!(samples[0].duration >= std::time::Duration::from_millis(10));

        let summary = profiler.get_summary().await;
        assert_eq!(summary.total_operations, 1);
        assert!(summary.total_duration >= std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_debug_artifact_creation() {
        let file_artifact = DebugArtifact::file("test_file", PathBuf::from("/tmp/test.txt"));
        assert_eq!(file_artifact.name, "test_file");
        assert!(matches!(file_artifact.artifact_type, ArtifactType::File));
        assert!(file_artifact.path.is_some());

        let text_artifact = DebugArtifact::text("test_text", "Hello, world!");
        assert_eq!(text_artifact.name, "test_text");
        assert!(matches!(text_artifact.artifact_type, ArtifactType::Text));
        assert_eq!(text_artifact.content.as_deref(), Some("Hello, world!"));

        let data = serde_json::json!({"key": "value"});
        let json_artifact = DebugArtifact::json("test_json", &data).unwrap();
        assert_eq!(json_artifact.name, "test_json");
        assert!(matches!(json_artifact.artifact_type, ArtifactType::Json));
        assert!(json_artifact.content.is_some());
    }
}
