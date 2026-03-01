//! Inference execution tracing with span hierarchy, Chrome trace export,
//! analysis, filtering, and RAII span guards.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// -----------------------------------------------------------------------
// Trace ID generation
// -----------------------------------------------------------------------

static NEXT_TRACE_SPAN_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a trace span.
pub type TraceSpanId = u64;

fn next_trace_span_id() -> TraceSpanId {
    NEXT_TRACE_SPAN_ID.fetch_add(1, Ordering::Relaxed)
}

fn trace_now_us() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| {
        #[allow(clippy::cast_possible_truncation)]
        let us = d.as_micros() as u64;
        us
    })
}

// -----------------------------------------------------------------------
// TraceConfig
// -----------------------------------------------------------------------

/// Verbosity level for trace output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceVerbosity {
    /// Only record top-level spans.
    Minimal,
    /// Record spans and key events.
    Normal,
    /// Record everything including fine-grained details.
    Verbose,
}

/// Output format for exported traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceOutputFormat {
    /// Chrome `chrome://tracing` JSON format.
    ChromeTrace,
    /// Plain JSON array of spans.
    Json,
}

/// Configuration for the inference tracer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceConfig {
    /// Whether tracing is enabled.
    pub enabled: bool,
    /// Verbosity level.
    pub verbosity: TraceVerbosity,
    /// Output format for exports.
    pub output_format: TraceOutputFormat,
    /// Span names to include (empty = all).
    pub enabled_spans: Vec<String>,
    /// Maximum number of spans before dropping new ones.
    pub max_trace_size: usize,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            verbosity: TraceVerbosity::Normal,
            output_format: TraceOutputFormat::ChromeTrace,
            enabled_spans: Vec::new(),
            max_trace_size: 100_000,
        }
    }
}

// -----------------------------------------------------------------------
// TraceEvent
// -----------------------------------------------------------------------

/// Types of events that can occur during inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceEvent {
    /// GPU/CPU kernel launch.
    KernelLaunch { kernel_name: String, grid_size: [u32; 3], block_size: [u32; 3] },
    /// Memory allocation.
    MemoryAlloc { bytes: u64, device: String },
    /// Memory deallocation.
    MemoryFree { bytes: u64, device: String },
    /// Data transfer between devices.
    DataTransfer { bytes: u64, source: String, destination: String },
    /// Device synchronization barrier.
    Synchronize { device: String },
    /// User-defined event.
    Custom { tag: String, payload: String },
}

impl fmt::Display for TraceEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelLaunch { kernel_name, .. } => {
                write!(f, "KernelLaunch({kernel_name})")
            }
            Self::MemoryAlloc { bytes, device } => {
                write!(f, "MemoryAlloc({bytes}B on {device})")
            }
            Self::MemoryFree { bytes, device } => {
                write!(f, "MemoryFree({bytes}B on {device})")
            }
            Self::DataTransfer { bytes, source, destination } => {
                write!(f, "DataTransfer({bytes}B {source}->{destination})")
            }
            Self::Synchronize { device } => {
                write!(f, "Synchronize({device})")
            }
            Self::Custom { tag, .. } => write!(f, "Custom({tag})"),
        }
    }
}

// -----------------------------------------------------------------------
// TraceSpan
// -----------------------------------------------------------------------

/// A named time span with optional parent/child relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    /// Unique span identifier.
    pub id: TraceSpanId,
    /// Human-readable name.
    pub name: String,
    /// Start time in microseconds since epoch.
    pub start_us: u64,
    /// End time in microseconds since epoch (`None` if still open).
    pub end_us: Option<u64>,
    /// Parent span ID (`None` for root spans).
    pub parent_id: Option<TraceSpanId>,
    /// Events that occurred within this span.
    pub events: Vec<TraceEvent>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl TraceSpan {
    /// Duration in microseconds (0 if still open).
    #[must_use]
    pub fn duration_us(&self) -> u64 {
        self.end_us.unwrap_or(self.start_us).saturating_sub(self.start_us)
    }

    /// Whether the span is still open (no `end_us`).
    #[must_use]
    pub const fn is_open(&self) -> bool {
        self.end_us.is_none()
    }

    /// IDs of direct child spans within a given slice.
    #[must_use]
    pub fn children<'a>(&self, all: &'a [Self]) -> Vec<&'a Self> {
        all.iter().filter(|s| s.parent_id == Some(self.id)).collect()
    }
}

// -----------------------------------------------------------------------
// TraceTimeline
// -----------------------------------------------------------------------

/// Ordered collection of spans for visualization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceTimeline {
    spans: Vec<TraceSpan>,
}

impl TraceTimeline {
    /// Create an empty timeline.
    #[must_use]
    pub const fn new() -> Self {
        Self { spans: Vec::new() }
    }

    /// Add a span to the timeline.
    pub fn push(&mut self, span: TraceSpan) {
        self.spans.push(span);
    }

    /// All spans in insertion order.
    #[must_use]
    pub fn spans(&self) -> &[TraceSpan] {
        &self.spans
    }

    /// Number of spans.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether the timeline is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Return root spans (those with no parent).
    #[must_use]
    pub fn root_spans(&self) -> Vec<&TraceSpan> {
        self.spans.iter().filter(|s| s.parent_id.is_none()).collect()
    }

    /// Find a span by ID.
    #[must_use]
    pub fn find(&self, id: TraceSpanId) -> Option<&TraceSpan> {
        self.spans.iter().find(|s| s.id == id)
    }

    /// Return spans sorted by start time.
    #[must_use]
    pub fn sorted_by_start(&self) -> Vec<&TraceSpan> {
        let mut sorted: Vec<&TraceSpan> = self.spans.iter().collect();
        sorted.sort_by_key(|s| s.start_us);
        sorted
    }

    /// Total wall-clock duration (last end − first start).
    #[must_use]
    pub fn total_duration_us(&self) -> u64 {
        if self.spans.is_empty() {
            return 0;
        }
        let first_start = self.spans.iter().map(|s| s.start_us).min().unwrap_or(0);
        let last_end = self.spans.iter().filter_map(|s| s.end_us).max().unwrap_or(first_start);
        last_end.saturating_sub(first_start)
    }
}

// -----------------------------------------------------------------------
// InferenceTracer
// -----------------------------------------------------------------------

/// Records detailed traces of inference execution.
pub struct InferenceTracer {
    config: TraceConfig,
    timeline: TraceTimeline,
    span_stack: Vec<TraceSpanId>,
}

impl InferenceTracer {
    /// Create a new tracer with the given configuration.
    #[must_use]
    pub const fn new(config: TraceConfig) -> Self {
        Self { config, timeline: TraceTimeline::new(), span_stack: Vec::new() }
    }

    /// Create a tracer with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(TraceConfig::default())
    }

    /// Borrow the current configuration.
    #[must_use]
    pub const fn config(&self) -> &TraceConfig {
        &self.config
    }

    /// Borrow the recorded timeline.
    #[must_use]
    pub const fn timeline(&self) -> &TraceTimeline {
        &self.timeline
    }

    /// Number of recorded spans.
    #[must_use]
    pub const fn span_count(&self) -> usize {
        self.timeline.len()
    }

    /// Whether tracing is enabled and we haven't hit the limit.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.config.enabled && self.timeline.len() < self.config.max_trace_size
    }

    /// Check if a span name passes the enabled-spans filter.
    fn passes_filter(&self, name: &str) -> bool {
        self.config.enabled_spans.is_empty() || self.config.enabled_spans.iter().any(|s| s == name)
    }

    /// Begin a new span. Returns the span ID.
    pub fn begin_span(&mut self, name: impl Into<String>) -> TraceSpanId {
        let name = name.into();
        if !self.is_active() || !self.passes_filter(&name) {
            return 0;
        }
        let id = next_trace_span_id();
        let parent_id = self.span_stack.last().copied();
        let span = TraceSpan {
            id,
            name,
            start_us: trace_now_us(),
            end_us: None,
            parent_id,
            events: Vec::new(),
            metadata: HashMap::new(),
        };
        self.timeline.push(span);
        self.span_stack.push(id);
        id
    }

    /// End an open span.
    pub fn end_span(&mut self, id: TraceSpanId) {
        if id == 0 {
            return;
        }
        let now = trace_now_us();
        for span in &mut self.timeline.spans {
            if span.id == id {
                span.end_us = Some(now);
                break;
            }
        }
        self.span_stack.retain(|&s| s != id);
    }

    /// Record an event within the current (innermost) open span.
    pub fn record_event(&mut self, event: TraceEvent) {
        if !self.is_active() {
            return;
        }
        if let Some(&current_id) = self.span_stack.last() {
            for span in &mut self.timeline.spans {
                if span.id == current_id {
                    span.events.push(event);
                    break;
                }
            }
        }
    }

    /// Attach metadata to a span.
    pub fn set_metadata(
        &mut self,
        span_id: TraceSpanId,
        key: impl Into<String>,
        value: impl Into<String>,
    ) {
        for span in &mut self.timeline.spans {
            if span.id == span_id {
                span.metadata.insert(key.into(), value.into());
                break;
            }
        }
    }

    /// Create an RAII span guard that ends the span on drop.
    pub fn span_guard(&mut self, name: impl Into<String>) -> SpanGuard<'_> {
        let id = self.begin_span(name);
        SpanGuard { tracer: self, id }
    }

    /// Record a complete span (already timed) without begin/end.
    pub fn record_complete_span(
        &mut self,
        name: impl Into<String>,
        start_us: u64,
        end_us: u64,
        events: Vec<TraceEvent>,
    ) {
        let name = name.into();
        if !self.is_active() || !self.passes_filter(&name) {
            return;
        }
        let id = next_trace_span_id();
        let parent_id = self.span_stack.last().copied();
        self.timeline.push(TraceSpan {
            id,
            name,
            start_us,
            end_us: Some(end_us),
            parent_id,
            events,
            metadata: HashMap::new(),
        });
    }

    /// Consume the tracer and return the timeline.
    #[must_use]
    pub fn into_timeline(self) -> TraceTimeline {
        self.timeline
    }

    /// Clear all recorded spans.
    pub fn reset(&mut self) {
        self.timeline = TraceTimeline::new();
        self.span_stack.clear();
    }
}

// -----------------------------------------------------------------------
// SpanGuard (RAII)
// -----------------------------------------------------------------------

/// RAII guard that closes a span when dropped.
pub struct SpanGuard<'a> {
    tracer: &'a mut InferenceTracer,
    id: TraceSpanId,
}

impl SpanGuard<'_> {
    /// The span ID managed by this guard.
    #[must_use]
    pub const fn id(&self) -> TraceSpanId {
        self.id
    }

    /// Record an event on the guarded span.
    pub fn record_event(&mut self, event: TraceEvent) {
        self.tracer.record_event(event);
    }
}

impl Drop for SpanGuard<'_> {
    fn drop(&mut self) {
        self.tracer.end_span(self.id);
    }
}

// -----------------------------------------------------------------------
// TraceExporter trait
// -----------------------------------------------------------------------

/// Export a [`TraceTimeline`] to an external format.
pub trait TraceExporter {
    /// Export the timeline and return the serialized bytes.
    ///
    /// # Errors
    /// Returns an error string if serialization fails.
    fn export(&self, timeline: &TraceTimeline) -> Result<String, String>;

    /// Format name for display.
    fn format_name(&self) -> &'static str;
}

// -----------------------------------------------------------------------
// ChromeTraceExporter
// -----------------------------------------------------------------------

/// Chrome trace event entry (catapult / `chrome://tracing`).
#[derive(Debug, Serialize)]
struct ChromeTraceEventEntry {
    name: String,
    cat: String,
    ph: String,
    ts: u64,
    dur: u64,
    pid: u32,
    tid: u32,
    args: HashMap<String, String>,
}

/// Exports traces in Chrome `chrome://tracing` JSON format.
#[derive(Debug, Default)]
pub struct ChromeTraceExporter {
    /// Process ID to embed in the output.
    pub pid: u32,
}

impl ChromeTraceExporter {
    /// Create a new exporter with the given process ID.
    #[must_use]
    pub const fn new(pid: u32) -> Self {
        Self { pid }
    }
}

impl TraceExporter for ChromeTraceExporter {
    fn export(&self, timeline: &TraceTimeline) -> Result<String, String> {
        let entries: Vec<ChromeTraceEventEntry> = timeline
            .spans()
            .iter()
            .filter(|s| s.end_us.is_some())
            .map(|s| {
                let mut args = s.metadata.clone();
                for ev in &s.events {
                    args.insert("event".to_string(), ev.to_string());
                }
                let tid = s.parent_id.unwrap_or(0);
                #[allow(clippy::cast_possible_truncation)]
                let tid32 = tid as u32;
                ChromeTraceEventEntry {
                    name: s.name.clone(),
                    cat: event_category_str(&s.events),
                    ph: "X".to_string(),
                    ts: s.start_us,
                    dur: s.duration_us(),
                    pid: self.pid,
                    tid: tid32,
                    args,
                }
            })
            .collect();

        serde_json::to_string_pretty(&entries).map_err(|e| e.to_string())
    }

    fn format_name(&self) -> &'static str {
        "chrome_trace"
    }
}

fn event_category_str(events: &[TraceEvent]) -> String {
    if events.is_empty() {
        return "general".to_string();
    }
    match &events[0] {
        TraceEvent::KernelLaunch { .. } => "kernel",
        TraceEvent::MemoryAlloc { .. } | TraceEvent::MemoryFree { .. } => "memory",
        TraceEvent::DataTransfer { .. } => "transfer",
        TraceEvent::Synchronize { .. } => "sync",
        TraceEvent::Custom { tag, .. } => return tag.clone(),
    }
    .to_string()
}

// -----------------------------------------------------------------------
// JsonExporter
// -----------------------------------------------------------------------

/// Exports traces as a plain JSON array of spans.
#[derive(Debug, Default)]
pub struct JsonExporter;

impl TraceExporter for JsonExporter {
    fn export(&self, timeline: &TraceTimeline) -> Result<String, String> {
        serde_json::to_string_pretty(timeline.spans()).map_err(|e| e.to_string())
    }

    fn format_name(&self) -> &'static str {
        "json"
    }
}

// -----------------------------------------------------------------------
// TraceAnalyzer
// -----------------------------------------------------------------------

/// Statistics computed from a [`TraceTimeline`].
#[derive(Debug, Clone, Serialize)]
pub struct TraceStatistics {
    pub total_spans: usize,
    pub total_wall_time_us: u64,
    pub total_kernel_time_us: u64,
    pub total_transfer_time_us: u64,
    pub total_memory_allocated: u64,
    pub total_memory_freed: u64,
    pub sync_count: u64,
    pub idle_time_us: u64,
    pub kernel_names: Vec<String>,
}

impl TraceStatistics {
    /// Fraction of wall-clock time spent in kernels.
    #[must_use]
    pub fn kernel_utilization(&self) -> f64 {
        if self.total_wall_time_us == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let util = self.total_kernel_time_us as f64 / self.total_wall_time_us as f64;
        util
    }

    /// Fraction of wall-clock time spent transferring data.
    #[must_use]
    pub fn transfer_overhead(&self) -> f64 {
        if self.total_wall_time_us == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let pct = self.total_transfer_time_us as f64 / self.total_wall_time_us as f64;
        pct
    }

    /// Human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "Spans: {} | Wall: {}us | Kernel: {}us | Transfer: {}us",
            self.total_spans,
            self.total_wall_time_us,
            self.total_kernel_time_us,
            self.total_transfer_time_us,
        );
        let _ = writeln!(
            s,
            "Alloc: {} bytes | Freed: {} bytes | Syncs: {} | Idle: {}us",
            self.total_memory_allocated,
            self.total_memory_freed,
            self.sync_count,
            self.idle_time_us,
        );
        if !self.kernel_names.is_empty() {
            let _ = writeln!(s, "Kernels: {}", self.kernel_names.join(", "));
        }
        s
    }
}

/// Computes statistics from trace data.
pub struct TraceAnalyzer;

impl TraceAnalyzer {
    /// Analyze a timeline and produce [`TraceStatistics`].
    #[must_use]
    pub fn analyze(timeline: &TraceTimeline) -> TraceStatistics {
        let mut kernel_time: u64 = 0;
        let mut transfer_time: u64 = 0;
        let mut alloc_bytes: u64 = 0;
        let mut free_bytes: u64 = 0;
        let mut sync_count: u64 = 0;
        let mut kernel_names: Vec<String> = Vec::new();

        for span in timeline.spans() {
            for event in &span.events {
                match event {
                    TraceEvent::KernelLaunch { kernel_name, .. } => {
                        kernel_time += span.duration_us();
                        if !kernel_names.contains(kernel_name) {
                            kernel_names.push(kernel_name.clone());
                        }
                    }
                    TraceEvent::DataTransfer { .. } => {
                        transfer_time += span.duration_us();
                    }
                    TraceEvent::MemoryAlloc { bytes, .. } => {
                        alloc_bytes += bytes;
                    }
                    TraceEvent::MemoryFree { bytes, .. } => {
                        free_bytes += bytes;
                    }
                    TraceEvent::Synchronize { .. } => {
                        sync_count += 1;
                    }
                    TraceEvent::Custom { .. } => {}
                }
            }
        }

        let wall = timeline.total_duration_us();
        let busy = kernel_time.saturating_add(transfer_time);
        let idle = wall.saturating_sub(busy);

        TraceStatistics {
            total_spans: timeline.len(),
            total_wall_time_us: wall,
            total_kernel_time_us: kernel_time,
            total_transfer_time_us: transfer_time,
            total_memory_allocated: alloc_bytes,
            total_memory_freed: free_bytes,
            sync_count,
            idle_time_us: idle,
            kernel_names,
        }
    }
}

// -----------------------------------------------------------------------
// TraceFilter
// -----------------------------------------------------------------------

/// Criteria for filtering trace spans.
#[derive(Debug, Clone, Default)]
pub struct TraceFilter {
    /// Only include spans whose name contains this substring.
    pub name_contains: Option<String>,
    /// Only include spans lasting at least this many microseconds.
    pub min_duration_us: Option<u64>,
    /// Only include spans lasting at most this many microseconds.
    pub max_duration_us: Option<u64>,
    /// Only include spans that have events of these types.
    pub event_types: Vec<TraceEventKind>,
    /// Only include root spans (no parent).
    pub root_only: bool,
}

/// Discriminant for [`TraceEvent`] used in filter matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceEventKind {
    KernelLaunch,
    MemoryAlloc,
    MemoryFree,
    DataTransfer,
    Synchronize,
    Custom,
}

impl TraceFilter {
    /// Create an empty filter (matches everything).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the name substring filter.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name_contains = Some(name.into());
        self
    }

    /// Set the minimum duration filter.
    #[must_use]
    pub const fn with_min_duration(mut self, us: u64) -> Self {
        self.min_duration_us = Some(us);
        self
    }

    /// Set the maximum duration filter.
    #[must_use]
    pub const fn with_max_duration(mut self, us: u64) -> Self {
        self.max_duration_us = Some(us);
        self
    }

    /// Require spans to have at least one event of the given kind.
    #[must_use]
    pub fn with_event_type(mut self, kind: TraceEventKind) -> Self {
        self.event_types.push(kind);
        self
    }

    /// Only match root spans.
    #[must_use]
    pub const fn root_only(mut self) -> Self {
        self.root_only = true;
        self
    }

    /// Test whether a span matches this filter.
    #[must_use]
    pub fn matches(&self, span: &TraceSpan) -> bool {
        if let Some(ref name) = self.name_contains
            && !span.name.contains(name.as_str())
        {
            return false;
        }
        if let Some(min) = self.min_duration_us
            && span.duration_us() < min
        {
            return false;
        }
        if let Some(max) = self.max_duration_us
            && span.duration_us() > max
        {
            return false;
        }
        if !self.event_types.is_empty() {
            let has_match = span.events.iter().any(|e| self.event_types.contains(&event_kind(e)));
            if !has_match {
                return false;
            }
        }
        if self.root_only && span.parent_id.is_some() {
            return false;
        }
        true
    }

    /// Apply this filter to a timeline, returning matching spans.
    #[must_use]
    pub fn apply<'a>(&self, timeline: &'a TraceTimeline) -> Vec<&'a TraceSpan> {
        timeline.spans().iter().filter(|s| self.matches(s)).collect()
    }
}

const fn event_kind(event: &TraceEvent) -> TraceEventKind {
    match event {
        TraceEvent::KernelLaunch { .. } => TraceEventKind::KernelLaunch,
        TraceEvent::MemoryAlloc { .. } => TraceEventKind::MemoryAlloc,
        TraceEvent::MemoryFree { .. } => TraceEventKind::MemoryFree,
        TraceEvent::DataTransfer { .. } => TraceEventKind::DataTransfer,
        TraceEvent::Synchronize { .. } => TraceEventKind::Synchronize,
        TraceEvent::Custom { .. } => TraceEventKind::Custom,
    }
}

// -----------------------------------------------------------------------
// DistributedTrace
// -----------------------------------------------------------------------

/// Identifier for a device or node in a distributed trace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId {
    /// Node name or IP.
    pub node: String,
    /// Device index on the node.
    pub device_index: u32,
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:gpu{}", self.node, self.device_index)
    }
}

/// A per-device trace for distributed correlation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceTrace {
    /// Which device produced this trace.
    pub device: DeviceId,
    /// The timeline of spans on this device.
    pub timeline: TraceTimeline,
}

/// Correlates traces across multiple devices and/or nodes.
#[derive(Debug, Default)]
pub struct DistributedTrace {
    traces: Vec<DeviceTrace>,
    /// Optional global trace ID.
    pub trace_id: Option<String>,
}

impl DistributedTrace {
    /// Create a new distributed trace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with an explicit global trace ID.
    #[must_use]
    pub fn with_trace_id(id: impl Into<String>) -> Self {
        Self { traces: Vec::new(), trace_id: Some(id.into()) }
    }

    /// Add a device trace.
    pub fn add_device_trace(&mut self, trace: DeviceTrace) {
        self.traces.push(trace);
    }

    /// All device traces.
    #[must_use]
    pub fn device_traces(&self) -> &[DeviceTrace] {
        &self.traces
    }

    /// Number of devices.
    #[must_use]
    pub const fn device_count(&self) -> usize {
        self.traces.len()
    }

    /// Total span count across all devices.
    #[must_use]
    pub fn total_span_count(&self) -> usize {
        self.traces.iter().map(|t| t.timeline.len()).sum()
    }

    /// Merge all device timelines into a single timeline, sorted
    /// by start time. Span IDs remain unique (from the atomic counter).
    #[must_use]
    pub fn merged_timeline(&self) -> TraceTimeline {
        let mut merged = TraceTimeline::new();
        for dt in &self.traces {
            for span in dt.timeline.spans() {
                let mut s = span.clone();
                s.metadata.insert("device".to_string(), dt.device.to_string());
                merged.push(s);
            }
        }
        merged
    }

    /// Compute per-device statistics.
    #[must_use]
    pub fn per_device_stats(&self) -> Vec<(DeviceId, TraceStatistics)> {
        self.traces
            .iter()
            .map(|dt| (dt.device.clone(), TraceAnalyzer::analyze(&dt.timeline)))
            .collect()
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- TraceConfig ---------------------------------------------------

    #[test]
    fn test_config_default() {
        let cfg = TraceConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.verbosity, TraceVerbosity::Normal);
        assert_eq!(cfg.output_format, TraceOutputFormat::ChromeTrace);
        assert!(cfg.enabled_spans.is_empty());
        assert_eq!(cfg.max_trace_size, 100_000);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = TraceConfig {
            enabled: false,
            verbosity: TraceVerbosity::Verbose,
            output_format: TraceOutputFormat::Json,
            enabled_spans: vec!["matmul".into()],
            max_trace_size: 50,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: TraceConfig = serde_json::from_str(&json).unwrap();
        assert!(!cfg2.enabled);
        assert_eq!(cfg2.verbosity, TraceVerbosity::Verbose);
        assert_eq!(cfg2.enabled_spans, vec!["matmul".to_string()]);
    }

    // -- TraceEvent ----------------------------------------------------

    #[test]
    fn test_event_display_kernel_launch() {
        let ev = TraceEvent::KernelLaunch {
            kernel_name: "gemm".into(),
            grid_size: [1, 1, 1],
            block_size: [256, 1, 1],
        };
        assert!(ev.to_string().contains("gemm"));
    }

    #[test]
    fn test_event_display_memory_alloc() {
        let ev = TraceEvent::MemoryAlloc { bytes: 4096, device: "gpu0".into() };
        assert!(ev.to_string().contains("4096"));
    }

    #[test]
    fn test_event_display_data_transfer() {
        let ev = TraceEvent::DataTransfer {
            bytes: 1024,
            source: "cpu".into(),
            destination: "gpu0".into(),
        };
        let s = ev.to_string();
        assert!(s.contains("cpu") && s.contains("gpu0"));
    }

    #[test]
    fn test_event_display_synchronize() {
        let ev = TraceEvent::Synchronize { device: "gpu0".into() };
        assert!(ev.to_string().contains("gpu0"));
    }

    #[test]
    fn test_event_display_custom() {
        let ev = TraceEvent::Custom { tag: "checkpoint".into(), payload: "step=100".into() };
        assert!(ev.to_string().contains("checkpoint"));
    }

    #[test]
    fn test_event_display_memory_free() {
        let ev = TraceEvent::MemoryFree { bytes: 2048, device: "gpu1".into() };
        let s = ev.to_string();
        assert!(s.contains("2048") && s.contains("gpu1"));
    }

    #[test]
    fn test_event_serde_roundtrip() {
        let ev = TraceEvent::KernelLaunch {
            kernel_name: "softmax".into(),
            grid_size: [4, 1, 1],
            block_size: [128, 1, 1],
        };
        let json = serde_json::to_string(&ev).unwrap();
        let ev2: TraceEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(ev, ev2);
    }

    // -- TraceSpan -----------------------------------------------------

    #[test]
    fn test_span_duration_closed() {
        let span = TraceSpan {
            id: 1,
            name: "test".into(),
            start_us: 100,
            end_us: Some(300),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        assert_eq!(span.duration_us(), 200);
        assert!(!span.is_open());
    }

    #[test]
    fn test_span_duration_open() {
        let span = TraceSpan {
            id: 1,
            name: "test".into(),
            start_us: 100,
            end_us: None,
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        assert_eq!(span.duration_us(), 0);
        assert!(span.is_open());
    }

    #[test]
    fn test_span_children() {
        let parent = TraceSpan {
            id: 1,
            name: "parent".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        let child1 = TraceSpan {
            id: 2,
            name: "child1".into(),
            start_us: 10,
            end_us: Some(50),
            parent_id: Some(1),
            events: vec![],
            metadata: HashMap::new(),
        };
        let child2 = TraceSpan {
            id: 3,
            name: "child2".into(),
            start_us: 60,
            end_us: Some(90),
            parent_id: Some(1),
            events: vec![],
            metadata: HashMap::new(),
        };
        let unrelated = TraceSpan {
            id: 4,
            name: "other".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        let all = [parent.clone(), child1, child2, unrelated];
        let kids = parent.children(&all);
        assert_eq!(kids.len(), 2);
    }

    #[test]
    fn test_span_serde_roundtrip() {
        let span = TraceSpan {
            id: 42,
            name: "attn".into(),
            start_us: 1000,
            end_us: Some(2000),
            parent_id: None,
            events: vec![TraceEvent::Synchronize { device: "gpu0".into() }],
            metadata: HashMap::new(),
        };
        let json = serde_json::to_string(&span).unwrap();
        let span2: TraceSpan = serde_json::from_str(&json).unwrap();
        assert_eq!(span2.id, 42);
        assert_eq!(span2.duration_us(), 1000);
    }

    // -- TraceTimeline -------------------------------------------------

    #[test]
    fn test_timeline_empty() {
        let tl = TraceTimeline::new();
        assert!(tl.is_empty());
        assert_eq!(tl.len(), 0);
        assert_eq!(tl.total_duration_us(), 0);
        assert!(tl.root_spans().is_empty());
    }

    #[test]
    fn test_timeline_push_and_find() {
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 10,
            name: "a".into(),
            start_us: 100,
            end_us: Some(200),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        });
        assert_eq!(tl.len(), 1);
        assert!(tl.find(10).is_some());
        assert!(tl.find(99).is_none());
    }

    #[test]
    fn test_timeline_root_spans() {
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 1,
            name: "root".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        });
        tl.push(TraceSpan {
            id: 2,
            name: "child".into(),
            start_us: 10,
            end_us: Some(50),
            parent_id: Some(1),
            events: vec![],
            metadata: HashMap::new(),
        });
        assert_eq!(tl.root_spans().len(), 1);
    }

    #[test]
    fn test_timeline_sorted_by_start() {
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 1,
            name: "b".into(),
            start_us: 200,
            end_us: Some(300),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        });
        tl.push(TraceSpan {
            id: 2,
            name: "a".into(),
            start_us: 100,
            end_us: Some(150),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        });
        let sorted = tl.sorted_by_start();
        assert_eq!(sorted[0].name, "a");
        assert_eq!(sorted[1].name, "b");
    }

    #[test]
    fn test_timeline_total_duration() {
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 1,
            name: "a".into(),
            start_us: 100,
            end_us: Some(300),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        });
        tl.push(TraceSpan {
            id: 2,
            name: "b".into(),
            start_us: 200,
            end_us: Some(500),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        });
        assert_eq!(tl.total_duration_us(), 400); // 500 - 100
    }

    // -- InferenceTracer -----------------------------------------------

    #[test]
    fn test_tracer_new_defaults() {
        let tracer = InferenceTracer::with_defaults();
        assert!(tracer.is_active());
        assert_eq!(tracer.span_count(), 0);
    }

    #[test]
    fn test_tracer_disabled() {
        let mut cfg = TraceConfig::default();
        cfg.enabled = false;
        let mut tracer = InferenceTracer::new(cfg);
        let id = tracer.begin_span("nope");
        assert_eq!(id, 0);
        assert_eq!(tracer.span_count(), 0);
    }

    #[test]
    fn test_tracer_begin_end_span() {
        let mut tracer = InferenceTracer::with_defaults();
        let id = tracer.begin_span("inference_step");
        assert_ne!(id, 0);
        assert_eq!(tracer.span_count(), 1);
        tracer.end_span(id);
        let span = tracer.timeline().find(id).unwrap();
        assert!(span.end_us.is_some());
    }

    #[test]
    fn test_tracer_nested_spans() {
        let mut tracer = InferenceTracer::with_defaults();
        let outer = tracer.begin_span("outer");
        let inner = tracer.begin_span("inner");
        tracer.end_span(inner);
        tracer.end_span(outer);

        let inner_span = tracer.timeline().find(inner).unwrap();
        assert_eq!(inner_span.parent_id, Some(outer));
    }

    #[test]
    fn test_tracer_record_event() {
        let mut tracer = InferenceTracer::with_defaults();
        let id = tracer.begin_span("kernel");
        tracer.record_event(TraceEvent::KernelLaunch {
            kernel_name: "gemm".into(),
            grid_size: [1, 1, 1],
            block_size: [256, 1, 1],
        });
        tracer.end_span(id);
        let span = tracer.timeline().find(id).unwrap();
        assert_eq!(span.events.len(), 1);
    }

    #[test]
    fn test_tracer_set_metadata() {
        let mut tracer = InferenceTracer::with_defaults();
        let id = tracer.begin_span("op");
        tracer.set_metadata(id, "layer", "12");
        tracer.end_span(id);
        let span = tracer.timeline().find(id).unwrap();
        assert_eq!(span.metadata.get("layer").unwrap(), "12");
    }

    #[test]
    fn test_tracer_record_complete_span() {
        let mut tracer = InferenceTracer::with_defaults();
        tracer.record_complete_span("prerecorded", 100, 500, vec![]);
        assert_eq!(tracer.span_count(), 1);
        let span = &tracer.timeline().spans()[0];
        assert_eq!(span.duration_us(), 400);
    }

    #[test]
    fn test_tracer_max_trace_size() {
        let cfg = TraceConfig { max_trace_size: 3, ..TraceConfig::default() };
        let mut tracer = InferenceTracer::new(cfg);
        for i in 0..5 {
            let id = tracer.begin_span(format!("span_{i}"));
            tracer.end_span(id);
        }
        assert_eq!(tracer.span_count(), 3);
    }

    #[test]
    fn test_tracer_enabled_spans_filter() {
        let cfg = TraceConfig { enabled_spans: vec!["matmul".into()], ..TraceConfig::default() };
        let mut tracer = InferenceTracer::new(cfg);
        let id1 = tracer.begin_span("matmul");
        tracer.end_span(id1);
        let id2 = tracer.begin_span("softmax");
        assert_eq!(id2, 0); // filtered out
        assert_eq!(tracer.span_count(), 1);
    }

    #[test]
    fn test_tracer_reset() {
        let mut tracer = InferenceTracer::with_defaults();
        let id = tracer.begin_span("a");
        tracer.end_span(id);
        assert_eq!(tracer.span_count(), 1);
        tracer.reset();
        assert_eq!(tracer.span_count(), 0);
    }

    #[test]
    fn test_tracer_into_timeline() {
        let mut tracer = InferenceTracer::with_defaults();
        let id = tracer.begin_span("x");
        tracer.end_span(id);
        let tl = tracer.into_timeline();
        assert_eq!(tl.len(), 1);
    }

    #[test]
    fn test_tracer_end_span_zero_is_noop() {
        let mut tracer = InferenceTracer::with_defaults();
        tracer.end_span(0);
        assert_eq!(tracer.span_count(), 0);
    }

    // -- SpanGuard (RAII) ----------------------------------------------

    #[test]
    fn test_span_guard_closes_on_drop() {
        let mut tracer = InferenceTracer::with_defaults();
        let id;
        {
            let guard = tracer.span_guard("guarded");
            id = guard.id();
            assert_ne!(id, 0);
        }
        let span = tracer.timeline().find(id).unwrap();
        assert!(!span.is_open());
    }

    #[test]
    fn test_span_guard_record_event() {
        let mut tracer = InferenceTracer::with_defaults();
        let id;
        {
            let mut guard = tracer.span_guard("guarded");
            id = guard.id();
            guard.record_event(TraceEvent::Synchronize { device: "gpu0".into() });
        }
        let span = tracer.timeline().find(id).unwrap();
        assert_eq!(span.events.len(), 1);
    }

    // -- ChromeTraceExporter -------------------------------------------

    #[test]
    fn test_chrome_exporter_empty_timeline() {
        let tl = TraceTimeline::new();
        let exporter = ChromeTraceExporter::new(1);
        let result = exporter.export(&tl).unwrap();
        assert_eq!(result.trim(), "[]");
    }

    #[test]
    fn test_chrome_exporter_format_name() {
        let exporter = ChromeTraceExporter::new(0);
        assert_eq!(exporter.format_name(), "chrome_trace");
    }

    #[test]
    fn test_chrome_exporter_single_span() {
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 1,
            name: "attn".into(),
            start_us: 1000,
            end_us: Some(2000),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        });
        let exporter = ChromeTraceExporter::new(42);
        let json = exporter.export(&tl).unwrap();
        assert!(json.contains("\"name\": \"attn\""));
        assert!(json.contains("\"ph\": \"X\""));
        assert!(json.contains("\"pid\": 42"));
    }

    #[test]
    fn test_chrome_exporter_skips_open_spans() {
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 1,
            name: "open".into(),
            start_us: 0,
            end_us: None,
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        });
        tl.push(TraceSpan {
            id: 2,
            name: "closed".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        });
        let exporter = ChromeTraceExporter::new(0);
        let json = exporter.export(&tl).unwrap();
        assert!(!json.contains("\"name\": \"open\""));
        assert!(json.contains("\"name\": \"closed\""));
    }

    #[test]
    fn test_chrome_exporter_includes_metadata() {
        let mut meta = HashMap::new();
        meta.insert("layer".to_string(), "5".to_string());
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 1,
            name: "op".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: meta,
        });
        let exporter = ChromeTraceExporter::new(0);
        let json = exporter.export(&tl).unwrap();
        assert!(json.contains("\"layer\": \"5\""));
    }

    // -- JsonExporter --------------------------------------------------

    #[test]
    fn test_json_exporter_empty() {
        let tl = TraceTimeline::new();
        let exporter = JsonExporter;
        let result = exporter.export(&tl).unwrap();
        assert_eq!(result.trim(), "[]");
    }

    #[test]
    fn test_json_exporter_format_name() {
        assert_eq!(JsonExporter.format_name(), "json");
    }

    #[test]
    fn test_json_exporter_roundtrip() {
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 7,
            name: "test".into(),
            start_us: 500,
            end_us: Some(600),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        });
        let exporter = JsonExporter;
        let json = exporter.export(&tl).unwrap();
        let spans: Vec<TraceSpan> = serde_json::from_str(&json).unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "test");
    }

    // -- TraceAnalyzer -------------------------------------------------

    #[test]
    fn test_analyzer_empty_timeline() {
        let tl = TraceTimeline::new();
        let stats = TraceAnalyzer::analyze(&tl);
        assert_eq!(stats.total_spans, 0);
        assert_eq!(stats.total_kernel_time_us, 0);
        assert_eq!(stats.kernel_utilization(), 0.0);
    }

    #[test]
    fn test_analyzer_kernel_time() {
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 1,
            name: "k1".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![TraceEvent::KernelLaunch {
                kernel_name: "gemm".into(),
                grid_size: [1, 1, 1],
                block_size: [256, 1, 1],
            }],
            metadata: HashMap::new(),
        });
        let stats = TraceAnalyzer::analyze(&tl);
        assert_eq!(stats.total_kernel_time_us, 100);
        assert_eq!(stats.kernel_names, vec!["gemm".to_string()]);
    }

    #[test]
    fn test_analyzer_transfer_time() {
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 1,
            name: "xfer".into(),
            start_us: 0,
            end_us: Some(50),
            parent_id: None,
            events: vec![TraceEvent::DataTransfer {
                bytes: 4096,
                source: "cpu".into(),
                destination: "gpu0".into(),
            }],
            metadata: HashMap::new(),
        });
        let stats = TraceAnalyzer::analyze(&tl);
        assert_eq!(stats.total_transfer_time_us, 50);
    }

    #[test]
    fn test_analyzer_memory_tracking() {
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 1,
            name: "alloc".into(),
            start_us: 0,
            end_us: Some(1),
            parent_id: None,
            events: vec![TraceEvent::MemoryAlloc { bytes: 8192, device: "gpu0".into() }],
            metadata: HashMap::new(),
        });
        tl.push(TraceSpan {
            id: 2,
            name: "free".into(),
            start_us: 10,
            end_us: Some(11),
            parent_id: None,
            events: vec![TraceEvent::MemoryFree { bytes: 4096, device: "gpu0".into() }],
            metadata: HashMap::new(),
        });
        let stats = TraceAnalyzer::analyze(&tl);
        assert_eq!(stats.total_memory_allocated, 8192);
        assert_eq!(stats.total_memory_freed, 4096);
    }

    #[test]
    fn test_analyzer_sync_count() {
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 1,
            name: "sync".into(),
            start_us: 0,
            end_us: Some(5),
            parent_id: None,
            events: vec![TraceEvent::Synchronize { device: "gpu0".into() }],
            metadata: HashMap::new(),
        });
        let stats = TraceAnalyzer::analyze(&tl);
        assert_eq!(stats.sync_count, 1);
    }

    #[test]
    fn test_analyzer_utilization_ratio() {
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 1,
            name: "k".into(),
            start_us: 0,
            end_us: Some(500),
            parent_id: None,
            events: vec![TraceEvent::KernelLaunch {
                kernel_name: "gemm".into(),
                grid_size: [1, 1, 1],
                block_size: [256, 1, 1],
            }],
            metadata: HashMap::new(),
        });
        tl.push(TraceSpan {
            id: 2,
            name: "t".into(),
            start_us: 500,
            end_us: Some(1000),
            parent_id: None,
            events: vec![TraceEvent::DataTransfer {
                bytes: 100,
                source: "cpu".into(),
                destination: "gpu0".into(),
            }],
            metadata: HashMap::new(),
        });
        let stats = TraceAnalyzer::analyze(&tl);
        assert!((stats.kernel_utilization() - 0.5).abs() < 0.01);
        assert!((stats.transfer_overhead() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_analyzer_summary_format() {
        let tl = TraceTimeline::new();
        let stats = TraceAnalyzer::analyze(&tl);
        let summary = stats.summary();
        assert!(summary.contains("Spans: 0"));
    }

    // -- TraceFilter ---------------------------------------------------

    #[test]
    fn test_filter_empty_matches_all() {
        let filter = TraceFilter::new();
        let span = TraceSpan {
            id: 1,
            name: "anything".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        assert!(filter.matches(&span));
    }

    #[test]
    fn test_filter_by_name() {
        let filter = TraceFilter::new().with_name("matmul");
        let hit = TraceSpan {
            id: 1,
            name: "matmul_fp32".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        let miss = TraceSpan {
            id: 2,
            name: "softmax".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        assert!(filter.matches(&hit));
        assert!(!filter.matches(&miss));
    }

    #[test]
    fn test_filter_by_min_duration() {
        let filter = TraceFilter::new().with_min_duration(50);
        let long_span = TraceSpan {
            id: 1,
            name: "long".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        let short_span = TraceSpan {
            id: 2,
            name: "short".into(),
            start_us: 0,
            end_us: Some(10),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        assert!(filter.matches(&long_span));
        assert!(!filter.matches(&short_span));
    }

    #[test]
    fn test_filter_by_max_duration() {
        let filter = TraceFilter::new().with_max_duration(50);
        let short = TraceSpan {
            id: 1,
            name: "s".into(),
            start_us: 0,
            end_us: Some(30),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        let long = TraceSpan {
            id: 2,
            name: "l".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        assert!(filter.matches(&short));
        assert!(!filter.matches(&long));
    }

    #[test]
    fn test_filter_by_event_type() {
        let filter = TraceFilter::new().with_event_type(TraceEventKind::KernelLaunch);
        let with_kernel = TraceSpan {
            id: 1,
            name: "a".into(),
            start_us: 0,
            end_us: Some(10),
            parent_id: None,
            events: vec![TraceEvent::KernelLaunch {
                kernel_name: "gemm".into(),
                grid_size: [1, 1, 1],
                block_size: [256, 1, 1],
            }],
            metadata: HashMap::new(),
        };
        let without = TraceSpan {
            id: 2,
            name: "b".into(),
            start_us: 0,
            end_us: Some(10),
            parent_id: None,
            events: vec![TraceEvent::Synchronize { device: "gpu0".into() }],
            metadata: HashMap::new(),
        };
        assert!(filter.matches(&with_kernel));
        assert!(!filter.matches(&without));
    }

    #[test]
    fn test_filter_root_only() {
        let filter = TraceFilter::new().root_only();
        let root = TraceSpan {
            id: 1,
            name: "root".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        let child = TraceSpan {
            id: 2,
            name: "child".into(),
            start_us: 0,
            end_us: Some(50),
            parent_id: Some(1),
            events: vec![],
            metadata: HashMap::new(),
        };
        assert!(filter.matches(&root));
        assert!(!filter.matches(&child));
    }

    #[test]
    fn test_filter_apply_to_timeline() {
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 1,
            name: "matmul".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        });
        tl.push(TraceSpan {
            id: 2,
            name: "softmax".into(),
            start_us: 100,
            end_us: Some(200),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        });
        let filter = TraceFilter::new().with_name("matmul");
        let results = filter.apply(&tl);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "matmul");
    }

    #[test]
    fn test_filter_combined() {
        let filter = TraceFilter::new().with_name("kernel").with_min_duration(50);
        let good = TraceSpan {
            id: 1,
            name: "kernel_op".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        let too_short = TraceSpan {
            id: 2,
            name: "kernel_op".into(),
            start_us: 0,
            end_us: Some(10),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        let wrong_name = TraceSpan {
            id: 3,
            name: "softmax".into(),
            start_us: 0,
            end_us: Some(200),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        };
        assert!(filter.matches(&good));
        assert!(!filter.matches(&too_short));
        assert!(!filter.matches(&wrong_name));
    }

    // -- DistributedTrace ----------------------------------------------

    #[test]
    fn test_distributed_trace_empty() {
        let dt = DistributedTrace::new();
        assert_eq!(dt.device_count(), 0);
        assert_eq!(dt.total_span_count(), 0);
    }

    #[test]
    fn test_distributed_trace_with_trace_id() {
        let dt = DistributedTrace::with_trace_id("abc-123");
        assert_eq!(dt.trace_id.as_deref(), Some("abc-123"));
    }

    #[test]
    fn test_distributed_trace_add_device() {
        let mut dt = DistributedTrace::new();
        let dev = DeviceId { node: "node0".into(), device_index: 0 };
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: 1,
            name: "op".into(),
            start_us: 0,
            end_us: Some(100),
            parent_id: None,
            events: vec![],
            metadata: HashMap::new(),
        });
        dt.add_device_trace(DeviceTrace { device: dev, timeline: tl });
        assert_eq!(dt.device_count(), 1);
        assert_eq!(dt.total_span_count(), 1);
    }

    #[test]
    fn test_distributed_trace_merged_timeline() {
        let mut dt = DistributedTrace::new();
        for i in 0..2 {
            let dev = DeviceId { node: "node0".into(), device_index: i };
            let mut tl = TraceTimeline::new();
            tl.push(TraceSpan {
                id: next_trace_span_id(),
                name: format!("op_gpu{i}"),
                start_us: 0,
                end_us: Some(100),
                parent_id: None,
                events: vec![],
                metadata: HashMap::new(),
            });
            dt.add_device_trace(DeviceTrace { device: dev, timeline: tl });
        }
        let merged = dt.merged_timeline();
        assert_eq!(merged.len(), 2);
        // Each span should have a "device" metadata entry.
        for span in merged.spans() {
            assert!(span.metadata.contains_key("device"));
        }
    }

    #[test]
    fn test_distributed_trace_per_device_stats() {
        let mut dt = DistributedTrace::new();
        let dev = DeviceId { node: "host".into(), device_index: 0 };
        let mut tl = TraceTimeline::new();
        tl.push(TraceSpan {
            id: next_trace_span_id(),
            name: "k".into(),
            start_us: 0,
            end_us: Some(200),
            parent_id: None,
            events: vec![TraceEvent::KernelLaunch {
                kernel_name: "gemm".into(),
                grid_size: [1, 1, 1],
                block_size: [256, 1, 1],
            }],
            metadata: HashMap::new(),
        });
        dt.add_device_trace(DeviceTrace { device: dev.clone(), timeline: tl });
        let stats = dt.per_device_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].0, dev);
        assert_eq!(stats[0].1.total_kernel_time_us, 200);
    }

    #[test]
    fn test_device_id_display() {
        let dev = DeviceId { node: "server1".into(), device_index: 3 };
        assert_eq!(dev.to_string(), "server1:gpu3");
    }

    // -- event_kind helper ---------------------------------------------

    #[test]
    fn test_event_kind_mapping() {
        assert_eq!(
            event_kind(&TraceEvent::KernelLaunch {
                kernel_name: "x".into(),
                grid_size: [1, 1, 1],
                block_size: [1, 1, 1],
            }),
            TraceEventKind::KernelLaunch
        );
        assert_eq!(
            event_kind(&TraceEvent::MemoryAlloc { bytes: 1, device: "d".into() }),
            TraceEventKind::MemoryAlloc
        );
        assert_eq!(
            event_kind(&TraceEvent::MemoryFree { bytes: 1, device: "d".into() }),
            TraceEventKind::MemoryFree
        );
        assert_eq!(
            event_kind(&TraceEvent::DataTransfer {
                bytes: 1,
                source: "a".into(),
                destination: "b".into(),
            }),
            TraceEventKind::DataTransfer
        );
        assert_eq!(
            event_kind(&TraceEvent::Synchronize { device: "d".into() }),
            TraceEventKind::Synchronize
        );
        assert_eq!(
            event_kind(&TraceEvent::Custom { tag: "t".into(), payload: "p".into() }),
            TraceEventKind::Custom
        );
    }
}
