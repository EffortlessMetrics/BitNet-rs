use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::Level;
use tracing_subscriber::{
    EnvFilter, Layer,
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
};

use super::{
    config::TestConfig,
    errors::{TestError, TestOpResult},
};

/// Initialize logging for the test framework
pub fn init_logging(config: &TestConfig) -> TestOpResult<()> {
    let log_level = parse_log_level(&config.log_level)?;

    // Create environment filter for console
    let console_filter = EnvFilter::builder()
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
        .with_filter(console_filter);

    // Create file layer if output directory exists
    let file_layer = if config.reporting.output_dir.exists() {
        let log_file = config.reporting.output_dir.join("test-execution.log");
        let file = File::create(&log_file).map_err(|e| {
            TestError::config(format!("Failed to create log file {:?}: {}", log_file, e))
        })?;

        // Create separate filter for file layer
        let file_filter = EnvFilter::builder()
            .with_default_directive(log_level.into())
            .from_env_lossy()
            .add_directive("bitnet=debug".parse().unwrap())
            .add_directive("bitnet_tests=debug".parse().unwrap());

        Some(
            fmt::layer()
                .with_writer(Arc::new(file))
                .with_ansi(false) // No ANSI colors in file
                .with_target(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_span_events(FmtSpan::FULL)
                .with_filter(file_filter),
        )
    } else {
        None
    };

    // Initialize subscriber (use try_init to avoid panic if already initialized)
    let subscriber = tracing_subscriber::registry().with(console_layer);

    let result = if let Some(file_layer) = file_layer {
        subscriber.with(file_layer).try_init()
    } else {
        subscriber.try_init()
    };

    match result {
        Ok(_) => {
            tracing::info!("Logging initialized with level: {}", config.log_level);
        }
        Err(_) => {
            // Already initialized, just log a debug message
            tracing::debug!("Logging already initialized, skipping");
        }
    }

    Ok(())
}

/// Parse log level string into tracing Level
fn parse_log_level(level_str: &str) -> TestOpResult<Level> {
    match level_str.to_lowercase().as_str() {
        "trace" => Ok(Level::TRACE),
        "debug" => Ok(Level::DEBUG),
        "info" => Ok(Level::INFO),
        "warn" => Ok(Level::WARN),
        "error" => Ok(Level::ERROR),
        _ => Err(TestError::config(format!("Invalid log level: {}", level_str))),
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
        Self { traces: Arc::new(RwLock::new(HashMap::new())), enabled }
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    error_context: Arc<RwLock<Vec<ErrorContext>>>,
}

impl DebugContext {
    /// Create a new debug context
    pub fn new(test_name: String, tracer: Arc<TestTracer>) -> Self {
        Self {
            test_name,
            tracer,
            artifacts: Arc::new(RwLock::new(Vec::new())),
            start_time: std::time::Instant::now(),
            error_context: Arc::new(RwLock::new(Vec::new())),
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
        let event =
            TraceEvent { timestamp: std::time::SystemTime::now(), event_type, message, metadata };
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

    /// Add error context for debugging
    pub async fn add_error_context(&self, error: &TestError, context: String) {
        let error_ctx = ErrorContext {
            error: error.to_string(),
            error_category: error.category().to_string(),
            context: context.clone(),
            stack_trace: capture_stack_trace(),
            timestamp: std::time::SystemTime::now(),
        };

        let mut error_contexts = self.error_context.write().await;
        error_contexts.push(error_ctx);

        // Also log as trace event
        self.trace(TraceEventType::Error, format!("Error: {} - Context: {}", error, context)).await;
    }

    /// Get all error contexts
    pub async fn get_error_contexts(&self) -> Vec<ErrorContext> {
        self.error_context.read().await.clone()
    }

    /// Get the tracer for this debug context
    pub fn get_tracer(&self) -> &TestTracer {
        &self.tracer
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
        Self { scope_name, tracer, start_time: std::time::Instant::now() }
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
    pub fn json<S: Into<String>, T: serde::Serialize>(name: S, data: &T) -> TestOpResult<Self> {
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

/// Error context for debugging failed tests
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorContext {
    pub error: String,
    pub error_category: String,
    pub context: String,
    pub stack_trace: Option<String>,
    pub timestamp: std::time::SystemTime,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(error: TestError, context: String) -> Self {
        Self {
            error: error.to_string(),
            error_category: error.category().to_string(),
            context,
            stack_trace: capture_stack_trace(),
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Create error context with custom stack trace
    pub fn with_stack_trace(
        error: TestError,
        context: String,
        stack_trace: Option<String>,
    ) -> Self {
        Self {
            error: error.to_string(),
            error_category: error.category().to_string(),
            context,
            stack_trace,
            timestamp: std::time::SystemTime::now(),
        }
    }
}

/// Capture current stack trace for debugging
pub fn capture_stack_trace() -> Option<String> {
    // For now, return a simple backtrace
    // In a real implementation, you might use the `backtrace` crate
    // or other stack trace capture mechanisms
    Some(format!(
        "Stack trace captured at {}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
    ))
}

/// Artifact collector for failed tests
pub struct FailureArtifactCollector {
    enabled: bool,
    output_dir: std::path::PathBuf,
    artifacts: Arc<RwLock<Vec<DebugArtifact>>>,
}

impl FailureArtifactCollector {
    /// Create a new failure artifact collector
    pub fn new(enabled: bool, output_dir: std::path::PathBuf) -> Self {
        Self { enabled, output_dir, artifacts: Arc::new(RwLock::new(Vec::new())) }
    }

    /// Collect artifacts for a failed test
    pub async fn collect_failure_artifacts(
        &self,
        test_name: &str,
        error: &TestError,
        debug_context: &DebugContext,
    ) -> TestOpResult<()> {
        if !self.enabled {
            return Ok(());
        }

        // Create test-specific artifact directory
        let test_dir = self.output_dir.join("failures").join(test_name);
        tokio::fs::create_dir_all(&test_dir).await?;

        // Collect error information
        let error_artifact = DebugArtifact::json(
            "error_info",
            &serde_json::json!({
                "error": error.to_string(),
                "category": error.category(),
                "recoverable": error.is_recoverable(),
                "timestamp": std::time::SystemTime::now(),
            }),
        )?;

        let error_path = test_dir.join("error_info.json");
        if let Some(content) = &error_artifact.content {
            tokio::fs::write(&error_path, content).await?;
        }

        // Collect traces
        let traces = debug_context.tracer.get_traces(test_name).await;
        if !traces.is_empty() {
            let trace_artifact = DebugArtifact::json("execution_trace", &traces)?;
            let trace_path = test_dir.join("execution_trace.json");
            if let Some(content) = &trace_artifact.content {
                tokio::fs::write(&trace_path, content).await?;
            }
        }

        // Collect error contexts
        let error_contexts = debug_context.get_error_contexts().await;
        if !error_contexts.is_empty() {
            let context_artifact = DebugArtifact::json("error_contexts", &error_contexts)?;
            let context_path = test_dir.join("error_contexts.json");
            if let Some(content) = &context_artifact.content {
                tokio::fs::write(&context_path, content).await?;
            }
        }

        // Collect existing artifacts
        let existing_artifacts = debug_context.get_artifacts().await;
        for artifact in existing_artifacts {
            match &artifact.content {
                Some(content) => {
                    let artifact_path = test_dir.join(&artifact.name);
                    tokio::fs::write(&artifact_path, content).await?;
                }
                None => {
                    if let Some(source_path) = &artifact.path {
                        let artifact_path = test_dir.join(&artifact.name);
                        if source_path.exists() {
                            tokio::fs::copy(source_path, &artifact_path).await?;
                        }
                    }
                }
            }
        }

        // Collect system information
        let system_info = collect_system_info().await;
        let system_artifact = DebugArtifact::json("system_info", &system_info)?;
        let system_path = test_dir.join("system_info.json");
        if let Some(content) = &system_artifact.content {
            tokio::fs::write(&system_path, content).await?;
        }

        tracing::info!("Collected failure artifacts for test '{}' in {:?}", test_name, test_dir);
        Ok(())
    }

    /// Get all collected artifacts
    pub async fn get_artifacts(&self) -> Vec<DebugArtifact> {
        self.artifacts.read().await.clone()
    }
}

/// Collect system information for debugging
async fn collect_system_info() -> serde_json::Value {
    use super::utils::{get_cpu_cores, get_memory_usage, is_ci};

    serde_json::json!({
        "timestamp": std::time::SystemTime::now(),
        "platform": std::env::consts::OS,
        "architecture": std::env::consts::ARCH,
        "cpu_cores": get_cpu_cores(),
        "memory_usage": get_memory_usage(),
        "is_ci": is_ci(),
        "rust_version": std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
        "environment_variables": {
            "RUST_LOG": std::env::var("RUST_LOG").unwrap_or_default(),
            "BITNET_TEST_LOG_LEVEL": std::env::var("BITNET_TEST_LOG_LEVEL").unwrap_or_default(),
            "CI": std::env::var("CI").unwrap_or_default(),
        }
    })
}

/// Performance profiler for test operations
pub struct PerformanceProfiler {
    enabled: bool,
    samples: Arc<RwLock<Vec<PerformanceSample>>>,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(enabled: bool) -> Self {
        Self { enabled, samples: Arc::new(RwLock::new(Vec::new())) }
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
    start_memory: u64,
    samples: Arc<RwLock<Vec<PerformanceSample>>>,
    enabled: bool,
    metadata: std::collections::HashMap<String, String>,
}

impl OperationProfiler {
    fn new(name: String, samples: Arc<RwLock<Vec<PerformanceSample>>>, enabled: bool) -> Self {
        use super::utils::get_memory_usage;

        Self {
            name,
            start_time: std::time::Instant::now(),
            start_memory: get_memory_usage(),
            samples,
            enabled,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata to the profiler
    pub fn add_metadata<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Finish profiling and record the sample
    pub async fn finish(self) {
        if !self.enabled {
            return;
        }

        use super::utils::{get_memory_usage, get_peak_memory_usage};

        let duration = self.start_time.elapsed();
        let current_memory = get_memory_usage();
        let peak_memory = get_peak_memory_usage();
        let memory_delta = current_memory as i64 - self.start_memory as i64;

        let mut sample =
            PerformanceSample::new(self.name, duration).with_memory(memory_delta, peak_memory);

        // Add collected metadata
        for (key, value) in self.metadata {
            sample = sample.with_metadata(key, value);
        }

        let mut samples = self.samples.write().await;
        samples.push(sample);
    }
}

/// A single performance sample
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceSample {
    pub operation: String,
    pub duration: std::time::Duration,
    pub memory_delta: i64,
    pub memory_peak: u64,
    pub cpu_usage: Option<f64>,
    pub timestamp: std::time::SystemTime,
    pub metadata: std::collections::HashMap<String, String>,
}

impl PerformanceSample {
    /// Create a new performance sample
    pub fn new(operation: String, duration: std::time::Duration) -> Self {
        Self {
            operation,
            duration,
            memory_delta: 0,
            memory_peak: 0,
            cpu_usage: None,
            timestamp: std::time::SystemTime::now(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata to the sample
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set memory information
    pub fn with_memory(mut self, delta: i64, peak: u64) -> Self {
        self.memory_delta = delta;
        self.memory_peak = peak;
        self
    }

    /// Set CPU usage
    pub fn with_cpu_usage(mut self, cpu_usage: f64) -> Self {
        self.cpu_usage = Some(cpu_usage);
        self
    }
}

/// Summary of performance data
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceSummary {
    pub total_operations: usize,
    pub total_duration: std::time::Duration,
    pub average_duration: std::time::Duration,
    pub min_duration: std::time::Duration,
    pub max_duration: std::time::Duration,
    pub operations_by_type: HashMap<String, usize>,
    pub memory_stats: MemoryStats,
    pub performance_percentiles: PerformancePercentiles,
}

/// Memory usage statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryStats {
    pub total_memory_delta: i64,
    pub average_memory_delta: i64,
    pub peak_memory_usage: u64,
    pub memory_efficiency: f64, // operations per MB
}

/// Performance percentiles for detailed analysis
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformancePercentiles {
    pub p50: std::time::Duration,
    pub p90: std::time::Duration,
    pub p95: std::time::Duration,
    pub p99: std::time::Duration,
}

/// Comprehensive metrics collector
pub struct MetricsCollector {
    enabled: bool,
    metrics: Arc<RwLock<HashMap<String, MetricValue>>>,
    counters: Arc<RwLock<HashMap<String, u64>>>,
    gauges: Arc<RwLock<HashMap<String, f64>>>,
    histograms: Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record a counter metric
    pub async fn increment_counter(&self, name: &str, value: u64) {
        if !self.enabled {
            return;
        }

        let mut counters = self.counters.write().await;
        *counters.entry(name.to_string()).or_insert(0) += value;
    }

    /// Record a gauge metric
    pub async fn set_gauge(&self, name: &str, value: f64) {
        if !self.enabled {
            return;
        }

        let mut gauges = self.gauges.write().await;
        gauges.insert(name.to_string(), value);
    }

    /// Record a histogram value
    pub async fn record_histogram(&self, name: &str, value: f64) {
        if !self.enabled {
            return;
        }

        let mut histograms = self.histograms.write().await;
        histograms.entry(name.to_string()).or_default().push(value);
    }

    /// Record a custom metric
    pub async fn record_metric(&self, name: &str, value: MetricValue) {
        if !self.enabled {
            return;
        }

        let mut metrics = self.metrics.write().await;
        metrics.insert(name.to_string(), value);
    }

    /// Get all collected metrics
    pub async fn get_metrics_summary(&self) -> MetricsSummary {
        let metrics = self.metrics.read().await.clone();
        let counters = self.counters.read().await.clone();
        let gauges = self.gauges.read().await.clone();
        let histograms = self.histograms.read().await;

        let histogram_stats: HashMap<String, HistogramStats> = histograms
            .iter()
            .map(|(name, values)| {
                let stats = calculate_histogram_stats(values);
                (name.clone(), stats)
            })
            .collect();

        MetricsSummary {
            timestamp: std::time::SystemTime::now(),
            counters,
            gauges,
            histogram_stats,
            custom_metrics: metrics,
        }
    }

    /// Clear all metrics
    pub async fn clear_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        let mut counters = self.counters.write().await;
        let mut gauges = self.gauges.write().await;
        let mut histograms = self.histograms.write().await;

        metrics.clear();
        counters.clear();
        gauges.clear();
        histograms.clear();
    }
}

/// Custom metric value
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Duration(std::time::Duration),
    String(String),
    Boolean(bool),
}

/// Summary of all collected metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetricsSummary {
    pub timestamp: std::time::SystemTime,
    pub counters: HashMap<String, u64>,
    pub gauges: HashMap<String, f64>,
    pub histogram_stats: HashMap<String, HistogramStats>,
    pub custom_metrics: HashMap<String, MetricValue>,
}

/// Statistics for histogram data
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistogramStats {
    pub count: usize,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub std_dev: f64,
}

/// Calculate statistics for histogram values
fn calculate_histogram_stats(values: &[f64]) -> HistogramStats {
    if values.is_empty() {
        return HistogramStats {
            count: 0,
            sum: 0.0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            p90: 0.0,
            p95: 0.0,
            p99: 0.0,
            std_dev: 0.0,
        };
    }

    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let count = values.len();
    let sum: f64 = values.iter().sum();
    let min = sorted_values[0];
    let max = sorted_values[count - 1];
    let mean = sum / count as f64;

    let median = if count % 2 == 0 {
        (sorted_values[count / 2 - 1] + sorted_values[count / 2]) / 2.0
    } else {
        sorted_values[count / 2]
    };

    let p90 = sorted_values[(count as f64 * 0.9) as usize];
    let p95 = sorted_values[(count as f64 * 0.95) as usize];
    let p99 = sorted_values[(count as f64 * 0.99) as usize];

    let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
    let std_dev = variance.sqrt();

    HistogramStats { count, sum, min, max, mean, median, p90, p95, p99, std_dev }
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
                memory_stats: MemoryStats {
                    total_memory_delta: 0,
                    average_memory_delta: 0,
                    peak_memory_usage: 0,
                    memory_efficiency: 0.0,
                },
                performance_percentiles: PerformancePercentiles {
                    p50: std::time::Duration::ZERO,
                    p90: std::time::Duration::ZERO,
                    p95: std::time::Duration::ZERO,
                    p99: std::time::Duration::ZERO,
                },
            };
        }

        let total_operations = samples.len();
        let total_duration: std::time::Duration = samples.iter().map(|s| s.duration).sum();
        let average_duration = total_duration / total_operations as u32;
        let min_duration = samples.iter().map(|s| s.duration).min().unwrap();
        let max_duration = samples.iter().map(|s| s.duration).max().unwrap();

        let mut operations_by_type = HashMap::new();
        for sample in samples {
            *operations_by_type.entry(sample.operation.clone()).or_insert(0) += 1;
        }

        // Calculate memory statistics
        let total_memory_delta: i64 = samples.iter().map(|s| s.memory_delta).sum();
        let average_memory_delta = total_memory_delta / total_operations as i64;
        let peak_memory_usage = samples.iter().map(|s| s.memory_peak).max().unwrap_or(0);
        let memory_efficiency = if peak_memory_usage > 0 {
            total_operations as f64 / (peak_memory_usage as f64 / 1024.0 / 1024.0)
        // operations per MB
        } else {
            0.0
        };

        // Calculate performance percentiles
        let mut durations: Vec<std::time::Duration> = samples.iter().map(|s| s.duration).collect();
        durations.sort();

        let p50 = durations[total_operations / 2];
        let p90 = durations[(total_operations as f64 * 0.9) as usize];
        let p95 = durations[(total_operations as f64 * 0.95) as usize];
        let p99 = durations[(total_operations as f64 * 0.99) as usize];

        Self {
            total_operations,
            total_duration,
            average_duration,
            min_duration,
            max_duration,
            operations_by_type,
            memory_stats: MemoryStats {
                total_memory_delta,
                average_memory_delta,
                peak_memory_usage,
                memory_efficiency,
            },
            performance_percentiles: PerformancePercentiles { p50, p90, p95, p99 },
        }
    }
}

/// Comprehensive logging and debugging manager
pub struct LoggingManager {
    config: TestConfig,
    tracer: Arc<TestTracer>,
    performance_profiler: Arc<PerformanceProfiler>,
    metrics_collector: Arc<MetricsCollector>,
    artifact_collector: Arc<FailureArtifactCollector>,
}

impl LoggingManager {
    /// Create a new logging manager
    pub fn new(config: TestConfig) -> TestOpResult<Self> {
        // Initialize logging first
        init_logging(&config)?;

        let tracer = Arc::new(TestTracer::new(true));
        let performance_profiler = Arc::new(PerformanceProfiler::new(true));
        let metrics_collector = Arc::new(MetricsCollector::new(true));
        let artifact_collector = Arc::new(FailureArtifactCollector::new(
            config.reporting.include_artifacts,
            config.reporting.output_dir.clone(),
        ));

        Ok(Self { config, tracer, performance_profiler, metrics_collector, artifact_collector })
    }

    /// Create a debug context for a test
    pub fn create_debug_context(&self, test_name: String) -> DebugContext {
        DebugContext::new(test_name, Arc::clone(&self.tracer))
    }

    /// Start profiling an operation
    pub async fn start_profiling(&self, operation_name: String) -> OperationProfiler {
        self.performance_profiler.start_operation(operation_name).await
    }

    /// Record a metric
    pub async fn record_metric(&self, name: &str, value: MetricValue) {
        self.metrics_collector.record_metric(name, value).await;
    }

    /// Increment a counter
    pub async fn increment_counter(&self, name: &str, value: u64) {
        self.metrics_collector.increment_counter(name, value).await;
    }

    /// Set a gauge value
    pub async fn set_gauge(&self, name: &str, value: f64) {
        self.metrics_collector.set_gauge(name, value).await;
    }

    /// Record a histogram value
    pub async fn record_histogram(&self, name: &str, value: f64) {
        self.metrics_collector.record_histogram(name, value).await;
    }

    /// Handle test failure and collect artifacts
    pub async fn handle_test_failure(
        &self,
        test_name: &str,
        error: &TestError,
        debug_context: &DebugContext,
    ) -> TestOpResult<()> {
        // Log the failure
        tracing::error!("Test '{}' failed: {}", test_name, error);

        // Collect failure artifacts
        self.artifact_collector.collect_failure_artifacts(test_name, error, debug_context).await?;

        // Record failure metrics
        self.increment_counter("test_failures_total", 1).await;
        self.increment_counter(&format!("test_failures_by_category_{}", error.category()), 1).await;

        Ok(())
    }

    /// Get performance summary
    pub async fn get_performance_summary(&self) -> PerformanceSummary {
        self.performance_profiler.get_summary().await
    }

    /// Get metrics summary
    pub async fn get_metrics_summary(&self) -> MetricsSummary {
        self.metrics_collector.get_metrics_summary().await
    }

    /// Generate comprehensive test report
    pub async fn generate_test_report(
        &self,
        test_results: &[super::results::TestResult],
    ) -> TestOpResult<TestReport> {
        let performance_summary = self.get_performance_summary().await;
        let metrics_summary = self.get_metrics_summary().await;
        let all_traces = self.tracer.get_all_traces().await;

        Ok(TestReport {
            timestamp: std::time::SystemTime::now(),
            test_results: test_results.to_vec(),
            performance_summary,
            metrics_summary,
            traces: all_traces,
            system_info: collect_system_info().await,
        })
    }

    /// Clear all collected data
    pub async fn clear_all_data(&self) {
        self.tracer.clear_all_traces().await;
        self.performance_profiler.clear_samples().await;
        self.metrics_collector.clear_metrics().await;
    }
}

/// Comprehensive test report
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TestReport {
    pub timestamp: std::time::SystemTime,
    pub test_results: Vec<super::results::TestResult>,
    pub performance_summary: PerformanceSummary,
    pub metrics_summary: MetricsSummary,
    pub traces: HashMap<String, Vec<TraceEvent>>,
    pub system_info: serde_json::Value,
}

impl TestReport {
    /// Save report to file
    pub async fn save_to_file(&self, path: &std::path::Path) -> TestOpResult<()> {
        let json = serde_json::to_string_pretty(self)?;
        tokio::fs::write(path, json).await?;
        Ok(())
    }

    /// Get summary statistics
    pub fn get_summary_stats(&self) -> ReportSummaryStats {
        let total_tests = self.test_results.len();
        let passed_tests = self.test_results.iter().filter(|r| r.is_success()).count();
        let failed_tests = self.test_results.iter().filter(|r| r.is_failure()).count();

        let total_duration: std::time::Duration =
            self.test_results.iter().map(|r| r.duration).sum();
        let average_duration = if total_tests > 0 {
            total_duration / total_tests as u32
        } else {
            std::time::Duration::ZERO
        };

        ReportSummaryStats {
            total_tests,
            passed_tests,
            failed_tests,
            success_rate: if total_tests > 0 {
                passed_tests as f64 / total_tests as f64
            } else {
                0.0
            },
            total_duration,
            average_duration,
            total_operations: self.performance_summary.total_operations,
            peak_memory: self.performance_summary.memory_stats.peak_memory_usage,
        }
    }
}

/// Summary statistics for a test report
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReportSummaryStats {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub success_rate: f64,
    pub total_duration: std::time::Duration,
    pub average_duration: std::time::Duration,
    pub total_operations: usize,
    pub peak_memory: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

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

        context.trace(TraceEventType::Info, "Debug message".to_string()).await;

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

    #[tokio::test]
    async fn test_metrics_collector() {
        let collector = MetricsCollector::new(true);

        collector.increment_counter("test_counter", 5).await;
        collector.set_gauge("test_gauge", 42.5).await;
        collector.record_histogram("test_histogram", 1.0).await;
        collector.record_histogram("test_histogram", 2.0).await;
        collector.record_histogram("test_histogram", 3.0).await;

        let summary = collector.get_metrics_summary().await;

        assert_eq!(summary.counters.get("test_counter"), Some(&5));
        assert_eq!(summary.gauges.get("test_gauge"), Some(&42.5));

        let histogram_stats = summary.histogram_stats.get("test_histogram").unwrap();
        assert_eq!(histogram_stats.count, 3);
        assert_eq!(histogram_stats.mean, 2.0);
    }

    #[tokio::test]
    async fn test_error_context() {
        let tracer = Arc::new(TestTracer::new(true));
        let context = DebugContext::new("test_error_context".to_string(), tracer);

        let error = TestError::execution("Test error message");
        context.add_error_context(&error, "Additional context".to_string()).await;

        let error_contexts = context.get_error_contexts().await;
        assert_eq!(error_contexts.len(), 1);
        assert_eq!(error_contexts[0].context, "Additional context");
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
