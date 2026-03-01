//! Inference profiling framework for GPU HAL.
//!
//! Provides comprehensive profiling of inference execution including layer timing,
//! kernel launch instrumentation, memory tracking, and bottleneck detection.
//! Supports export to Chrome trace format for visualization in `chrome://tracing`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ── Output format ────────────────────────────────────────────────────────────

/// Output format for profiling data export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Chrome,
    Text,
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the inference profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    pub enabled: bool,
    pub trace_layers: bool,
    pub trace_kernels: bool,
    pub trace_memory: bool,
    pub output_format: OutputFormat,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trace_layers: true,
            trace_kernels: true,
            trace_memory: true,
            output_format: OutputFormat::Chrome,
        }
    }
}

// ── Span ─────────────────────────────────────────────────────────────────────

/// A single profiling span representing a timed region of execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerSpan {
    pub name: String,
    pub start_time_ns: u64,
    pub end_time_ns: u64,
    pub metadata: HashMap<String, String>,
    pub parent_span_id: Option<usize>,
    id: usize,
}

impl ProfilerSpan {
    /// Create a new completed span.
    pub fn new(name: impl Into<String>, start_time_ns: u64, end_time_ns: u64) -> Self {
        Self {
            name: name.into(),
            start_time_ns,
            end_time_ns,
            metadata: HashMap::new(),
            parent_span_id: None,
            id: 0,
        }
    }

    /// Duration in nanoseconds.
    pub fn duration_ns(&self) -> u64 {
        self.end_time_ns.saturating_sub(self.start_time_ns)
    }

    /// Unique identifier within the timeline.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Add a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set parent span id.
    pub fn with_parent(mut self, parent_id: usize) -> Self {
        self.parent_span_id = Some(parent_id);
        self
    }
}

// ── Timeline ─────────────────────────────────────────────────────────────────

/// Ordered collection of profiler spans with support for nested spans.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfilerTimeline {
    spans: Vec<ProfilerSpan>,
    #[serde(skip)]
    span_stack: Vec<usize>,
    next_id: usize,
}

impl ProfilerTimeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a new span onto the stack (begins a nested region).
    /// Returns the span id.
    pub fn begin_span(&mut self, name: impl Into<String>, start_time_ns: u64) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let parent_id = self.span_stack.last().copied();
        let span = ProfilerSpan {
            name: name.into(),
            start_time_ns,
            end_time_ns: 0,
            metadata: HashMap::new(),
            parent_span_id: parent_id,
            id,
        };
        self.spans.push(span);
        self.span_stack.push(id);
        id
    }

    /// End the current innermost span.
    pub fn end_span(&mut self, end_time_ns: u64) {
        if let Some(id) = self.span_stack.pop() {
            if let Some(span) = self.spans.iter_mut().find(|s| s.id == id) {
                span.end_time_ns = end_time_ns;
            }
        }
    }

    /// Add a pre-built completed span directly.
    pub fn add_span(&mut self, mut span: ProfilerSpan) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        span.id = id;
        if span.parent_span_id.is_none() {
            span.parent_span_id = self.span_stack.last().copied();
        }
        self.spans.push(span);
        id
    }

    /// All recorded spans, ordered by insertion.
    pub fn spans(&self) -> &[ProfilerSpan] {
        &self.spans
    }

    /// Number of recorded spans.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether the timeline is empty.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Current nesting depth.
    pub fn depth(&self) -> usize {
        self.span_stack.len()
    }

    /// Get root spans (no parent).
    pub fn root_spans(&self) -> Vec<&ProfilerSpan> {
        self.spans.iter().filter(|s| s.parent_span_id.is_none()).collect()
    }

    /// Get children of a given span.
    pub fn children_of(&self, span_id: usize) -> Vec<&ProfilerSpan> {
        self.spans.iter().filter(|s| s.parent_span_id == Some(span_id)).collect()
    }

    /// Total wall-clock time from first span start to last span end.
    pub fn total_time_ns(&self) -> u64 {
        if self.spans.is_empty() {
            return 0;
        }
        let min_start = self.spans.iter().map(|s| s.start_time_ns).min().unwrap_or(0);
        let max_end = self.spans.iter().map(|s| s.end_time_ns).max().unwrap_or(0);
        max_end.saturating_sub(min_start)
    }
}

// ── Layer Profiler ───────────────────────────────────────────────────────────

/// Instruments individual model layers with timing and memory tracking.
pub struct LayerProfiler {
    config: ProfilerConfig,
    timeline: ProfilerTimeline,
    reference: Instant,
}

impl LayerProfiler {
    pub fn new(config: ProfilerConfig) -> Self {
        Self { config, timeline: ProfilerTimeline::new(), reference: Instant::now() }
    }

    fn now_ns(&self) -> u64 {
        self.reference.elapsed().as_nanos() as u64
    }

    /// Call before entering a layer. Returns the span id.
    pub fn pre_layer(&mut self, layer_name: &str) -> Option<usize> {
        if !self.config.enabled || !self.config.trace_layers {
            return None;
        }
        let ts = self.now_ns();
        Some(self.timeline.begin_span(layer_name, ts))
    }

    /// Call after a layer completes. Optionally attach metadata.
    pub fn post_layer(&mut self, metadata: Option<HashMap<String, String>>) {
        if !self.config.enabled || !self.config.trace_layers {
            return;
        }
        let ts = self.now_ns();
        // Attach metadata to the span about to close
        if let Some(meta) = metadata {
            if let Some(&id) = self.timeline.span_stack.last() {
                if let Some(span) = self.timeline.spans.iter_mut().find(|s| s.id == id) {
                    span.metadata.extend(meta);
                }
            }
        }
        self.timeline.end_span(ts);
    }

    /// Consume and return the timeline.
    pub fn into_timeline(self) -> ProfilerTimeline {
        self.timeline
    }

    /// Borrow the timeline.
    pub fn timeline(&self) -> &ProfilerTimeline {
        &self.timeline
    }
}

// ── Kernel Profiler ──────────────────────────────────────────────────────────

/// Instruments GPU kernel launches with timing and grid-size metadata.
pub struct KernelProfiler {
    config: ProfilerConfig,
    timeline: ProfilerTimeline,
    reference: Instant,
}

impl KernelProfiler {
    pub fn new(config: ProfilerConfig) -> Self {
        Self { config, timeline: ProfilerTimeline::new(), reference: Instant::now() }
    }

    fn now_ns(&self) -> u64 {
        self.reference.elapsed().as_nanos() as u64
    }

    /// Call before launching a kernel. Returns the span id.
    pub fn pre_kernel(&mut self, kernel_name: &str, grid_size: [u32; 3]) -> Option<usize> {
        if !self.config.enabled || !self.config.trace_kernels {
            return None;
        }
        let ts = self.now_ns();
        let id = self.timeline.begin_span(kernel_name, ts);
        // Attach grid size as metadata
        if let Some(span) = self.timeline.spans.iter_mut().find(|s| s.id == id) {
            span.metadata.insert(
                "grid_size".to_string(),
                format!("{}x{}x{}", grid_size[0], grid_size[1], grid_size[2]),
            );
        }
        Some(id)
    }

    /// Call after a kernel completes.
    pub fn post_kernel(&mut self) {
        if !self.config.enabled || !self.config.trace_kernels {
            return;
        }
        let ts = self.now_ns();
        self.timeline.end_span(ts);
    }

    /// Consume and return the timeline.
    pub fn into_timeline(self) -> ProfilerTimeline {
        self.timeline
    }

    /// Borrow the timeline.
    pub fn timeline(&self) -> &ProfilerTimeline {
        &self.timeline
    }
}

// ── Memory Profiler ──────────────────────────────────────────────────────────

/// A single memory event (allocation or deallocation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub timestamp_ns: u64,
    pub size_bytes: i64,
    pub tag: String,
}

/// Tracks GPU memory allocations and deallocations over time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryProfiler {
    events: Vec<MemoryEvent>,
    current_bytes: i64,
    peak_bytes: i64,
    allocation_count: u64,
    deallocation_count: u64,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an allocation.
    pub fn record_alloc(&mut self, timestamp_ns: u64, size_bytes: u64, tag: impl Into<String>) {
        self.current_bytes += size_bytes as i64;
        if self.current_bytes > self.peak_bytes {
            self.peak_bytes = self.current_bytes;
        }
        self.allocation_count += 1;
        self.events.push(MemoryEvent {
            timestamp_ns,
            size_bytes: size_bytes as i64,
            tag: tag.into(),
        });
    }

    /// Record a deallocation.
    pub fn record_dealloc(&mut self, timestamp_ns: u64, size_bytes: u64, tag: impl Into<String>) {
        self.current_bytes -= size_bytes as i64;
        self.deallocation_count += 1;
        self.events.push(MemoryEvent {
            timestamp_ns,
            size_bytes: -(size_bytes as i64),
            tag: tag.into(),
        });
    }

    pub fn current_memory(&self) -> i64 {
        self.current_bytes
    }

    pub fn peak_memory(&self) -> i64 {
        self.peak_bytes
    }

    pub fn allocation_count(&self) -> u64 {
        self.allocation_count
    }

    pub fn deallocation_count(&self) -> u64 {
        self.deallocation_count
    }

    pub fn events(&self) -> &[MemoryEvent] {
        &self.events
    }
}

// ── Chrome Trace Exporter ────────────────────────────────────────────────────

/// Exports a [`ProfilerTimeline`] to Chrome trace event format (JSON).
///
/// The output is compatible with `chrome://tracing` and Perfetto UI.
pub struct ChromeTraceExporter;

impl ChromeTraceExporter {
    /// Export a timeline to Chrome trace JSON string.
    pub fn export(timeline: &ProfilerTimeline) -> String {
        let events: Vec<serde_json::Value> = timeline
            .spans()
            .iter()
            .flat_map(|span| {
                let dur_us = span.duration_ns() as f64 / 1000.0;
                let ts_us = span.start_time_ns as f64 / 1000.0;
                let mut args = serde_json::Map::new();
                for (k, v) in &span.metadata {
                    args.insert(k.clone(), serde_json::Value::String(v.clone()));
                }
                if let Some(pid) = span.parent_span_id {
                    args.insert(
                        "parent_span_id".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(pid)),
                    );
                }
                vec![serde_json::json!({
                    "name": span.name,
                    "cat": "inference",
                    "ph": "X",
                    "ts": ts_us,
                    "dur": dur_us,
                    "pid": 1,
                    "tid": 1,
                    "args": args
                })]
            })
            .collect();

        serde_json::to_string_pretty(&events).unwrap_or_else(|_| "[]".to_string())
    }

    /// Export timeline to a JSON value.
    pub fn export_value(timeline: &ProfilerTimeline) -> serde_json::Value {
        let json_str = Self::export(timeline);
        serde_json::from_str(&json_str).unwrap_or(serde_json::Value::Array(vec![]))
    }
}

// ── Profiler Aggregator ──────────────────────────────────────────────────────

/// Aggregated statistics for a set of spans sharing the same name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedStats {
    pub name: String,
    pub count: usize,
    pub total_time_ns: u64,
    pub avg_time_ns: u64,
    pub min_time_ns: u64,
    pub max_time_ns: u64,
    pub percentage: f64,
}

/// Aggregates spans by name and computes statistics.
pub struct ProfilerAggregator;

impl ProfilerAggregator {
    /// Aggregate all spans in the timeline grouped by name.
    pub fn aggregate(timeline: &ProfilerTimeline) -> Vec<AggregatedStats> {
        let mut groups: HashMap<String, Vec<u64>> = HashMap::new();
        for span in timeline.spans() {
            groups.entry(span.name.clone()).or_default().push(span.duration_ns());
        }

        let total_wall = timeline.total_time_ns().max(1);

        let mut stats: Vec<AggregatedStats> = groups
            .into_iter()
            .map(|(name, durations)| {
                let count = durations.len();
                let total: u64 = durations.iter().sum();
                let min = *durations.iter().min().unwrap_or(&0);
                let max = *durations.iter().max().unwrap_or(&0);
                let avg = if count > 0 { total / count as u64 } else { 0 };
                let percentage = (total as f64 / total_wall as f64) * 100.0;
                AggregatedStats {
                    name,
                    count,
                    total_time_ns: total,
                    avg_time_ns: avg,
                    min_time_ns: min,
                    max_time_ns: max,
                    percentage,
                }
            })
            .collect();

        // Sort by total_time descending
        stats.sort_by(|a, b| b.total_time_ns.cmp(&a.total_time_ns));
        stats
    }
}

// ── Bottleneck Detector ──────────────────────────────────────────────────────

/// A detected bottleneck with an optimization suggestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub span_name: String,
    pub total_time_ns: u64,
    pub percentage: f64,
    pub suggestion: String,
}

/// Analyzes a timeline to find the top-N bottlenecks and suggest optimizations.
pub struct BottleneckDetector {
    top_n: usize,
}

impl BottleneckDetector {
    pub fn new(top_n: usize) -> Self {
        Self { top_n }
    }

    /// Detect bottlenecks from the timeline.
    pub fn detect(&self, timeline: &ProfilerTimeline) -> Vec<Bottleneck> {
        let stats = ProfilerAggregator::aggregate(timeline);
        stats
            .into_iter()
            .take(self.top_n)
            .map(|s| {
                let suggestion = Self::suggest(&s);
                Bottleneck {
                    span_name: s.name,
                    total_time_ns: s.total_time_ns,
                    percentage: s.percentage,
                    suggestion,
                }
            })
            .collect()
    }

    fn suggest(stats: &AggregatedStats) -> String {
        if stats.percentage > 50.0 {
            format!(
                "'{}' dominates at {:.1}% of total time — consider kernel fusion or async overlap",
                stats.name, stats.percentage
            )
        } else if stats.count > 10 && stats.avg_time_ns < 1_000 {
            format!(
                "'{}' called {} times with very short duration — consider batching",
                stats.name, stats.count
            )
        } else if stats.max_time_ns > stats.avg_time_ns * 3 && stats.count > 1 {
            format!(
                "'{}' has high variance (max {:.1}x avg) — investigate outlier launches",
                stats.name,
                stats.max_time_ns as f64 / stats.avg_time_ns.max(1) as f64
            )
        } else {
            format!("'{}' accounts for {:.1}% of total time", stats.name, stats.percentage)
        }
    }
}

// ── Inference Profiler (main entry point) ────────────────────────────────────

/// Main profiling entry point that orchestrates layer, kernel, and memory profilers.
pub struct InferenceProfiler {
    config: ProfilerConfig,
    timeline: ProfilerTimeline,
    memory: MemoryProfiler,
    reference: Instant,
    active: bool,
}

impl InferenceProfiler {
    /// Create a new profiler with the given configuration.
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            timeline: ProfilerTimeline::new(),
            memory: MemoryProfiler::new(),
            reference: Instant::now(),
            active: false,
        }
    }

    /// Create with default configuration.
    pub fn default_enabled() -> Self {
        Self::new(ProfilerConfig::default())
    }

    fn now_ns(&self) -> u64 {
        self.reference.elapsed().as_nanos() as u64
    }

    /// Start a profiling session.
    pub fn start(&mut self) {
        self.active = true;
        self.reference = Instant::now();
        self.timeline = ProfilerTimeline::new();
        self.memory = MemoryProfiler::new();
    }

    /// Stop the profiling session.
    pub fn stop(&mut self) {
        // Close any remaining open spans
        let ts = self.now_ns();
        while self.timeline.depth() > 0 {
            self.timeline.end_span(ts);
        }
        self.active = false;
    }

    /// Whether profiling is currently active.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Begin a named span. Returns span id if profiling is active.
    pub fn begin_span(&mut self, name: &str) -> Option<usize> {
        if !self.active || !self.config.enabled {
            return None;
        }
        let ts = self.now_ns();
        Some(self.timeline.begin_span(name, ts))
    }

    /// End the current span.
    pub fn end_span(&mut self) {
        if !self.active || !self.config.enabled {
            return;
        }
        let ts = self.now_ns();
        self.timeline.end_span(ts);
    }

    /// Record a memory allocation.
    pub fn record_alloc(&mut self, size_bytes: u64, tag: &str) {
        if !self.active || !self.config.trace_memory {
            return;
        }
        let ts = self.now_ns();
        self.memory.record_alloc(ts, size_bytes, tag);
    }

    /// Record a memory deallocation.
    pub fn record_dealloc(&mut self, size_bytes: u64, tag: &str) {
        if !self.active || !self.config.trace_memory {
            return;
        }
        let ts = self.now_ns();
        self.memory.record_dealloc(ts, size_bytes, tag);
    }

    /// Get the recorded timeline.
    pub fn get_timeline(&self) -> &ProfilerTimeline {
        &self.timeline
    }

    /// Get the memory profiler state.
    pub fn get_memory(&self) -> &MemoryProfiler {
        &self.memory
    }

    /// Export timeline using the configured format.
    pub fn export(&self) -> String {
        match self.config.output_format {
            OutputFormat::Chrome => ChromeTraceExporter::export(&self.timeline),
            OutputFormat::Json => {
                serde_json::to_string_pretty(&self.timeline).unwrap_or_else(|_| "{}".to_string())
            }
            OutputFormat::Text => self.export_text(),
        }
    }

    fn export_text(&self) -> String {
        let mut out = String::new();
        for span in self.timeline.spans() {
            let indent = if span.parent_span_id.is_some() { "  " } else { "" };
            out.push_str(&format!(
                "{}{}: {:.3}ms\n",
                indent,
                span.name,
                span.duration_ns() as f64 / 1_000_000.0
            ));
        }
        out
    }

    /// Get detected bottlenecks.
    pub fn get_bottlenecks(&self, top_n: usize) -> Vec<Bottleneck> {
        BottleneckDetector::new(top_n).detect(&self.timeline)
    }

    /// Get aggregated statistics.
    pub fn get_aggregated_stats(&self) -> Vec<AggregatedStats> {
        ProfilerAggregator::aggregate(&self.timeline)
    }

    /// Configuration reference.
    pub fn config(&self) -> &ProfilerConfig {
        &self.config
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ProfilerSpan tests ───────────────────────────────────────────────

    #[test]
    fn test_span_creation() {
        let span = ProfilerSpan::new("test_op", 100, 500);
        assert_eq!(span.name, "test_op");
        assert_eq!(span.start_time_ns, 100);
        assert_eq!(span.end_time_ns, 500);
        assert_eq!(span.duration_ns(), 400);
    }

    #[test]
    fn test_span_zero_duration() {
        let span = ProfilerSpan::new("zero", 100, 100);
        assert_eq!(span.duration_ns(), 0);
    }

    #[test]
    fn test_span_metadata() {
        let span = ProfilerSpan::new("op", 0, 100)
            .with_metadata("key1", "val1")
            .with_metadata("key2", "val2");
        assert_eq!(span.metadata.len(), 2);
        assert_eq!(span.metadata["key1"], "val1");
        assert_eq!(span.metadata["key2"], "val2");
    }

    #[test]
    fn test_span_parent() {
        let span = ProfilerSpan::new("child", 0, 100).with_parent(42);
        assert_eq!(span.parent_span_id, Some(42));
    }

    #[test]
    fn test_span_no_parent() {
        let span = ProfilerSpan::new("root", 0, 100);
        assert_eq!(span.parent_span_id, None);
    }

    #[test]
    fn test_span_saturating_duration() {
        // end < start should return 0 via saturating_sub
        let mut span = ProfilerSpan::new("weird", 500, 100);
        span.end_time_ns = 100;
        assert_eq!(span.duration_ns(), 0);
    }

    // ── ProfilerTimeline tests ───────────────────────────────────────────

    #[test]
    fn test_timeline_empty() {
        let tl = ProfilerTimeline::new();
        assert!(tl.is_empty());
        assert_eq!(tl.len(), 0);
        assert_eq!(tl.total_time_ns(), 0);
        assert_eq!(tl.depth(), 0);
    }

    #[test]
    fn test_timeline_single_span() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("op1", 100);
        tl.end_span(500);
        assert_eq!(tl.len(), 1);
        assert_eq!(tl.spans()[0].duration_ns(), 400);
    }

    #[test]
    fn test_timeline_multiple_sequential_spans() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("op1", 100);
        tl.end_span(200);
        tl.begin_span("op2", 300);
        tl.end_span(500);
        assert_eq!(tl.len(), 2);
        assert_eq!(tl.total_time_ns(), 400); // 500 - 100
    }

    #[test]
    fn test_timeline_nested_spans() {
        let mut tl = ProfilerTimeline::new();
        let parent = tl.begin_span("parent", 100);
        let child = tl.begin_span("child", 150);
        tl.end_span(250); // end child
        tl.end_span(300); // end parent
        assert_eq!(tl.len(), 2);
        assert_eq!(tl.spans()[1].parent_span_id, Some(parent));
        assert_eq!(tl.spans()[0].parent_span_id, None);
        assert_eq!(tl.children_of(parent).len(), 1);
        assert_eq!(tl.children_of(parent)[0].id(), child);
    }

    #[test]
    fn test_timeline_deeply_nested() {
        let mut tl = ProfilerTimeline::new();
        let ids: Vec<usize> =
            (0..10).map(|i| tl.begin_span(format!("level_{i}"), i as u64 * 100)).collect();
        for i in (0..10).rev() {
            tl.end_span((i + 1) as u64 * 100 + 50);
        }
        assert_eq!(tl.len(), 10);
        // Each span except the first should have a parent
        for i in 1..10 {
            assert_eq!(tl.spans()[i].parent_span_id, Some(ids[i - 1]));
        }
    }

    #[test]
    fn test_timeline_root_spans() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("root1", 0);
        tl.begin_span("child1", 10);
        tl.end_span(20);
        tl.end_span(30);
        tl.begin_span("root2", 40);
        tl.end_span(50);
        let roots = tl.root_spans();
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0].name, "root1");
        assert_eq!(roots[1].name, "root2");
    }

    #[test]
    fn test_timeline_add_completed_span() {
        let mut tl = ProfilerTimeline::new();
        let span = ProfilerSpan::new("prebuilt", 0, 1000);
        let id = tl.add_span(span);
        assert_eq!(tl.len(), 1);
        assert_eq!(tl.spans()[0].id(), id);
    }

    #[test]
    fn test_timeline_add_span_inherits_parent() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("parent", 0);
        let span = ProfilerSpan::new("child", 10, 20);
        tl.add_span(span);
        tl.end_span(30);
        assert!(tl.spans()[1].parent_span_id.is_some());
    }

    #[test]
    fn test_timeline_depth_tracking() {
        let mut tl = ProfilerTimeline::new();
        assert_eq!(tl.depth(), 0);
        tl.begin_span("a", 0);
        assert_eq!(tl.depth(), 1);
        tl.begin_span("b", 10);
        assert_eq!(tl.depth(), 2);
        tl.end_span(20);
        assert_eq!(tl.depth(), 1);
        tl.end_span(30);
        assert_eq!(tl.depth(), 0);
    }

    #[test]
    fn test_timeline_end_span_on_empty_stack() {
        let mut tl = ProfilerTimeline::new();
        // Should not panic
        tl.end_span(100);
        assert_eq!(tl.len(), 0);
    }

    #[test]
    fn test_timeline_children_of_nonexistent() {
        let tl = ProfilerTimeline::new();
        assert!(tl.children_of(999).is_empty());
    }

    // ── LayerProfiler tests ─────────────────────────────────────────────

    #[test]
    fn test_layer_profiler_basic() {
        let mut lp = LayerProfiler::new(ProfilerConfig::default());
        let id = lp.pre_layer("attention");
        assert!(id.is_some());
        std::thread::sleep(std::time::Duration::from_millis(1));
        lp.post_layer(None);
        let tl = lp.into_timeline();
        assert_eq!(tl.len(), 1);
        assert!(tl.spans()[0].duration_ns() > 0);
    }

    #[test]
    fn test_layer_profiler_with_metadata() {
        let mut lp = LayerProfiler::new(ProfilerConfig::default());
        lp.pre_layer("ffn");
        let mut meta = HashMap::new();
        meta.insert("hidden_size".to_string(), "4096".to_string());
        lp.post_layer(Some(meta));
        let tl = lp.into_timeline();
        assert_eq!(tl.spans()[0].metadata["hidden_size"], "4096");
    }

    #[test]
    fn test_layer_profiler_disabled() {
        let config = ProfilerConfig { enabled: false, ..Default::default() };
        let mut lp = LayerProfiler::new(config);
        let id = lp.pre_layer("attention");
        assert!(id.is_none());
        lp.post_layer(None);
        assert!(lp.into_timeline().is_empty());
    }

    #[test]
    fn test_layer_profiler_layers_disabled() {
        let config = ProfilerConfig { trace_layers: false, ..Default::default() };
        let mut lp = LayerProfiler::new(config);
        let id = lp.pre_layer("attention");
        assert!(id.is_none());
    }

    #[test]
    fn test_layer_profiler_multiple_layers() {
        let mut lp = LayerProfiler::new(ProfilerConfig::default());
        for i in 0..5 {
            lp.pre_layer(&format!("layer_{i}"));
            lp.post_layer(None);
        }
        assert_eq!(lp.timeline().spans().len(), 5);
    }

    #[test]
    fn test_layer_profiler_nested_sublayers() {
        let mut lp = LayerProfiler::new(ProfilerConfig::default());
        lp.pre_layer("transformer_block");
        lp.pre_layer("self_attention");
        lp.post_layer(None);
        lp.pre_layer("feed_forward");
        lp.post_layer(None);
        lp.post_layer(None);
        let tl = lp.into_timeline();
        assert_eq!(tl.len(), 3);
        let roots = tl.root_spans();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].name, "transformer_block");
    }

    // ── KernelProfiler tests ────────────────────────────────────────────

    #[test]
    fn test_kernel_profiler_basic() {
        let mut kp = KernelProfiler::new(ProfilerConfig::default());
        let id = kp.pre_kernel("matmul", [256, 256, 1]);
        assert!(id.is_some());
        std::thread::sleep(std::time::Duration::from_millis(1));
        kp.post_kernel();
        let tl = kp.into_timeline();
        assert_eq!(tl.len(), 1);
        assert_eq!(tl.spans()[0].metadata["grid_size"], "256x256x1");
    }

    #[test]
    fn test_kernel_profiler_disabled() {
        let config = ProfilerConfig { enabled: false, ..Default::default() };
        let mut kp = KernelProfiler::new(config);
        let id = kp.pre_kernel("matmul", [1, 1, 1]);
        assert!(id.is_none());
    }

    #[test]
    fn test_kernel_profiler_kernels_disabled() {
        let config = ProfilerConfig { trace_kernels: false, ..Default::default() };
        let mut kp = KernelProfiler::new(config);
        let id = kp.pre_kernel("matmul", [1, 1, 1]);
        assert!(id.is_none());
    }

    #[test]
    fn test_kernel_profiler_multiple_kernels() {
        let mut kp = KernelProfiler::new(ProfilerConfig::default());
        for _ in 0..3 {
            kp.pre_kernel("softmax", [128, 1, 1]);
            kp.post_kernel();
        }
        assert_eq!(kp.timeline().spans().len(), 3);
    }

    #[test]
    fn test_kernel_profiler_grid_size_format() {
        let mut kp = KernelProfiler::new(ProfilerConfig::default());
        kp.pre_kernel("conv2d", [32, 64, 8]);
        kp.post_kernel();
        let tl = kp.into_timeline();
        assert_eq!(tl.spans()[0].metadata["grid_size"], "32x64x8");
    }

    // ── MemoryProfiler tests ────────────────────────────────────────────

    #[test]
    fn test_memory_profiler_alloc() {
        let mut mp = MemoryProfiler::new();
        mp.record_alloc(0, 1024, "weights");
        assert_eq!(mp.current_memory(), 1024);
        assert_eq!(mp.peak_memory(), 1024);
        assert_eq!(mp.allocation_count(), 1);
    }

    #[test]
    fn test_memory_profiler_alloc_dealloc() {
        let mut mp = MemoryProfiler::new();
        mp.record_alloc(0, 1024, "buf");
        mp.record_dealloc(100, 512, "buf");
        assert_eq!(mp.current_memory(), 512);
        assert_eq!(mp.peak_memory(), 1024);
        assert_eq!(mp.deallocation_count(), 1);
    }

    #[test]
    fn test_memory_profiler_peak_tracking() {
        let mut mp = MemoryProfiler::new();
        mp.record_alloc(0, 1000, "a");
        mp.record_alloc(10, 2000, "b");
        mp.record_dealloc(20, 1000, "a");
        mp.record_alloc(30, 500, "c");
        assert_eq!(mp.peak_memory(), 3000);
        assert_eq!(mp.current_memory(), 2500);
    }

    #[test]
    fn test_memory_profiler_events() {
        let mut mp = MemoryProfiler::new();
        mp.record_alloc(0, 100, "x");
        mp.record_dealloc(50, 50, "x");
        assert_eq!(mp.events().len(), 2);
        assert_eq!(mp.events()[0].size_bytes, 100);
        assert_eq!(mp.events()[1].size_bytes, -50);
    }

    #[test]
    fn test_memory_profiler_empty() {
        let mp = MemoryProfiler::new();
        assert_eq!(mp.current_memory(), 0);
        assert_eq!(mp.peak_memory(), 0);
        assert_eq!(mp.allocation_count(), 0);
        assert_eq!(mp.deallocation_count(), 0);
        assert!(mp.events().is_empty());
    }

    #[test]
    fn test_memory_profiler_full_dealloc() {
        let mut mp = MemoryProfiler::new();
        mp.record_alloc(0, 2048, "temp");
        mp.record_dealloc(100, 2048, "temp");
        assert_eq!(mp.current_memory(), 0);
        assert_eq!(mp.peak_memory(), 2048);
    }

    #[test]
    fn test_memory_profiler_many_small_allocs() {
        let mut mp = MemoryProfiler::new();
        for i in 0..100 {
            mp.record_alloc(i, 16, "small");
        }
        assert_eq!(mp.allocation_count(), 100);
        assert_eq!(mp.current_memory(), 1600);
        assert_eq!(mp.peak_memory(), 1600);
    }

    // ── ChromeTraceExporter tests ───────────────────────────────────────

    #[test]
    fn test_chrome_export_empty() {
        let tl = ProfilerTimeline::new();
        let json = ChromeTraceExporter::export(&tl);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.as_array().unwrap().is_empty());
    }

    #[test]
    fn test_chrome_export_single_span() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("test_kernel", 1_000_000);
        tl.end_span(2_000_000);
        let json = ChromeTraceExporter::export(&tl);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let events = parsed.as_array().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["name"], "test_kernel");
        assert_eq!(events[0]["ph"], "X");
        assert_eq!(events[0]["cat"], "inference");
        // ts should be in microseconds: 1_000_000 ns / 1000 = 1000.0 µs
        assert_eq!(events[0]["ts"], 1000.0);
        assert_eq!(events[0]["dur"], 1000.0);
    }

    #[test]
    fn test_chrome_export_multiple_spans() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("op1", 0);
        tl.end_span(100_000);
        tl.begin_span("op2", 200_000);
        tl.end_span(400_000);
        let parsed = ChromeTraceExporter::export_value(&tl);
        let events = parsed.as_array().unwrap();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_chrome_export_metadata_included() {
        let mut tl = ProfilerTimeline::new();
        let span = ProfilerSpan::new("kernel", 0, 1000).with_metadata("grid", "256x256");
        tl.add_span(span);
        let json = ChromeTraceExporter::export(&tl);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed[0]["args"]["grid"], "256x256");
    }

    #[test]
    fn test_chrome_export_parent_in_args() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("parent", 0);
        tl.begin_span("child", 10);
        tl.end_span(20);
        tl.end_span(30);
        let parsed = ChromeTraceExporter::export_value(&tl);
        let events = parsed.as_array().unwrap();
        // Child should have parent_span_id in args
        let child_event = &events[1];
        assert!(child_event["args"]["parent_span_id"].is_number());
    }

    #[test]
    fn test_chrome_export_pid_tid() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("op", 0);
        tl.end_span(100);
        let parsed = ChromeTraceExporter::export_value(&tl);
        assert_eq!(parsed[0]["pid"], 1);
        assert_eq!(parsed[0]["tid"], 1);
    }

    // ── ProfilerAggregator tests ────────────────────────────────────────

    #[test]
    fn test_aggregator_empty() {
        let tl = ProfilerTimeline::new();
        let stats = ProfilerAggregator::aggregate(&tl);
        assert!(stats.is_empty());
    }

    #[test]
    fn test_aggregator_single_group() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("matmul", 0);
        tl.end_span(1000);
        tl.begin_span("matmul", 2000);
        tl.end_span(4000);
        let stats = ProfilerAggregator::aggregate(&tl);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].name, "matmul");
        assert_eq!(stats[0].count, 2);
        assert_eq!(stats[0].total_time_ns, 3000); // 1000 + 2000
        assert_eq!(stats[0].avg_time_ns, 1500);
        assert_eq!(stats[0].min_time_ns, 1000);
        assert_eq!(stats[0].max_time_ns, 2000);
    }

    #[test]
    fn test_aggregator_multiple_groups() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("matmul", 0);
        tl.end_span(3000);
        tl.begin_span("softmax", 3000);
        tl.end_span(4000);
        let stats = ProfilerAggregator::aggregate(&tl);
        assert_eq!(stats.len(), 2);
        // Sorted by total_time desc, so matmul first
        assert_eq!(stats[0].name, "matmul");
        assert_eq!(stats[1].name, "softmax");
    }

    #[test]
    fn test_aggregator_percentage() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("fast", 0);
        tl.end_span(100);
        tl.begin_span("slow", 100);
        tl.end_span(1000);
        let stats = ProfilerAggregator::aggregate(&tl);
        // total wall = 1000 ns
        let slow = stats.iter().find(|s| s.name == "slow").unwrap();
        assert!((slow.percentage - 90.0).abs() < 1.0);
    }

    #[test]
    fn test_aggregator_sorted_by_total_time() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("short", 0);
        tl.end_span(100);
        tl.begin_span("medium", 200);
        tl.end_span(500);
        tl.begin_span("long", 600);
        tl.end_span(2000);
        let stats = ProfilerAggregator::aggregate(&tl);
        assert_eq!(stats[0].name, "long");
        assert_eq!(stats[1].name, "medium");
        assert_eq!(stats[2].name, "short");
    }

    // ── BottleneckDetector tests ────────────────────────────────────────

    #[test]
    fn test_bottleneck_empty_timeline() {
        let tl = ProfilerTimeline::new();
        let detector = BottleneckDetector::new(5);
        let bottlenecks = detector.detect(&tl);
        assert!(bottlenecks.is_empty());
    }

    #[test]
    fn test_bottleneck_top_n() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("a", 0);
        tl.end_span(100);
        tl.begin_span("b", 100);
        tl.end_span(500);
        tl.begin_span("c", 500);
        tl.end_span(2000);
        let detector = BottleneckDetector::new(2);
        let bottlenecks = detector.detect(&tl);
        assert_eq!(bottlenecks.len(), 2);
        assert_eq!(bottlenecks[0].span_name, "c");
        assert_eq!(bottlenecks[1].span_name, "b");
    }

    #[test]
    fn test_bottleneck_dominant_suggestion() {
        let mut tl = ProfilerTimeline::new();
        // One span dominates > 50%
        tl.begin_span("huge_kernel", 0);
        tl.end_span(9000);
        tl.begin_span("tiny", 9000);
        tl.end_span(10000);
        let bottlenecks = BottleneckDetector::new(5).detect(&tl);
        assert!(bottlenecks[0].suggestion.contains("dominates"));
    }

    #[test]
    fn test_bottleneck_has_percentage() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("op", 0);
        tl.end_span(1000);
        let bottlenecks = BottleneckDetector::new(1).detect(&tl);
        assert!(bottlenecks[0].percentage > 0.0);
    }

    #[test]
    fn test_bottleneck_suggestion_not_empty() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("op", 0);
        tl.end_span(1000);
        let bottlenecks = BottleneckDetector::new(1).detect(&tl);
        assert!(!bottlenecks[0].suggestion.is_empty());
    }

    // ── InferenceProfiler tests ─────────────────────────────────────────

    #[test]
    fn test_profiler_start_stop() {
        let mut p = InferenceProfiler::new(ProfilerConfig::default());
        assert!(!p.is_active());
        p.start();
        assert!(p.is_active());
        p.stop();
        assert!(!p.is_active());
    }

    #[test]
    fn test_profiler_spans() {
        let mut p = InferenceProfiler::default_enabled();
        p.start();
        p.begin_span("layer_0");
        std::thread::sleep(std::time::Duration::from_millis(1));
        p.end_span();
        p.stop();
        assert_eq!(p.get_timeline().len(), 1);
        assert!(p.get_timeline().spans()[0].duration_ns() > 0);
    }

    #[test]
    fn test_profiler_inactive_noop() {
        let mut p = InferenceProfiler::default_enabled();
        // Not started
        let id = p.begin_span("noop");
        assert!(id.is_none());
        p.end_span();
        assert!(p.get_timeline().is_empty());
    }

    #[test]
    fn test_profiler_memory_tracking() {
        let mut p = InferenceProfiler::default_enabled();
        p.start();
        p.record_alloc(4096, "kv_cache");
        p.record_alloc(2048, "weights");
        p.record_dealloc(1024, "temp");
        p.stop();
        assert_eq!(p.get_memory().current_memory(), 5120);
        assert_eq!(p.get_memory().allocation_count(), 2);
        assert_eq!(p.get_memory().deallocation_count(), 1);
    }

    #[test]
    fn test_profiler_memory_disabled() {
        let config = ProfilerConfig { trace_memory: false, ..Default::default() };
        let mut p = InferenceProfiler::new(config);
        p.start();
        p.record_alloc(4096, "kv_cache");
        p.stop();
        assert_eq!(p.get_memory().allocation_count(), 0);
    }

    #[test]
    fn test_profiler_export_chrome() {
        let config = ProfilerConfig { output_format: OutputFormat::Chrome, ..Default::default() };
        let mut p = InferenceProfiler::new(config);
        p.start();
        p.begin_span("test");
        p.end_span();
        p.stop();
        let json = p.export();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_array());
    }

    #[test]
    fn test_profiler_export_json() {
        let config = ProfilerConfig { output_format: OutputFormat::Json, ..Default::default() };
        let mut p = InferenceProfiler::new(config);
        p.start();
        p.begin_span("test");
        p.end_span();
        p.stop();
        let json = p.export();
        // Should be valid JSON
        assert!(serde_json::from_str::<serde_json::Value>(&json).is_ok());
    }

    #[test]
    fn test_profiler_export_text() {
        let config = ProfilerConfig { output_format: OutputFormat::Text, ..Default::default() };
        let mut p = InferenceProfiler::new(config);
        p.start();
        p.begin_span("my_layer");
        p.end_span();
        p.stop();
        let text = p.export();
        assert!(text.contains("my_layer"));
        assert!(text.contains("ms"));
    }

    #[test]
    fn test_profiler_get_bottlenecks() {
        let mut p = InferenceProfiler::default_enabled();
        p.start();
        p.begin_span("slow_op");
        std::thread::sleep(std::time::Duration::from_millis(2));
        p.end_span();
        p.begin_span("fast_op");
        p.end_span();
        p.stop();
        let bottlenecks = p.get_bottlenecks(1);
        assert_eq!(bottlenecks.len(), 1);
        assert_eq!(bottlenecks[0].span_name, "slow_op");
    }

    #[test]
    fn test_profiler_get_aggregated_stats() {
        let mut p = InferenceProfiler::default_enabled();
        p.start();
        p.begin_span("matmul");
        p.end_span();
        p.begin_span("matmul");
        p.end_span();
        p.begin_span("softmax");
        p.end_span();
        p.stop();
        let stats = p.get_aggregated_stats();
        assert_eq!(stats.len(), 2);
    }

    #[test]
    fn test_profiler_multiple_sessions() {
        let mut p = InferenceProfiler::default_enabled();

        // Session 1
        p.start();
        p.begin_span("session1_op");
        p.end_span();
        p.stop();
        assert_eq!(p.get_timeline().len(), 1);

        // Session 2 — resets
        p.start();
        assert!(p.get_timeline().is_empty());
        p.begin_span("session2_op");
        p.end_span();
        p.stop();
        assert_eq!(p.get_timeline().len(), 1);
        assert_eq!(p.get_timeline().spans()[0].name, "session2_op");
    }

    #[test]
    fn test_profiler_stop_closes_open_spans() {
        let mut p = InferenceProfiler::default_enabled();
        p.start();
        p.begin_span("unclosed_parent");
        p.begin_span("unclosed_child");
        // Don't end spans manually
        p.stop();
        // All spans should have end times
        for span in p.get_timeline().spans() {
            assert!(span.end_time_ns > 0);
        }
    }

    #[test]
    fn test_profiler_config_access() {
        let config = ProfilerConfig {
            enabled: true,
            trace_layers: false,
            trace_kernels: true,
            trace_memory: false,
            output_format: OutputFormat::Text,
        };
        let p = InferenceProfiler::new(config);
        assert!(p.config().enabled);
        assert!(!p.config().trace_layers);
        assert!(p.config().trace_kernels);
        assert!(!p.config().trace_memory);
        assert_eq!(p.config().output_format, OutputFormat::Text);
    }

    // ── ProfilerConfig tests ────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let config = ProfilerConfig::default();
        assert!(config.enabled);
        assert!(config.trace_layers);
        assert!(config.trace_kernels);
        assert!(config.trace_memory);
        assert_eq!(config.output_format, OutputFormat::Chrome);
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = ProfilerConfig {
            enabled: false,
            trace_layers: true,
            trace_kernels: false,
            trace_memory: true,
            output_format: OutputFormat::Json,
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ProfilerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.enabled, config.enabled);
        assert_eq!(parsed.trace_layers, config.trace_layers);
        assert_eq!(parsed.trace_kernels, config.trace_kernels);
        assert_eq!(parsed.trace_memory, config.trace_memory);
        assert_eq!(parsed.output_format, config.output_format);
    }

    // ── OutputFormat tests ──────────────────────────────────────────────

    #[test]
    fn test_output_format_equality() {
        assert_eq!(OutputFormat::Json, OutputFormat::Json);
        assert_eq!(OutputFormat::Chrome, OutputFormat::Chrome);
        assert_eq!(OutputFormat::Text, OutputFormat::Text);
        assert_ne!(OutputFormat::Json, OutputFormat::Chrome);
    }

    #[test]
    fn test_output_format_serialization() {
        let json_str = serde_json::to_string(&OutputFormat::Chrome).unwrap();
        let parsed: OutputFormat = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed, OutputFormat::Chrome);
    }

    // ── Integration / edge-case tests ───────────────────────────────────

    #[test]
    fn test_end_to_end_workflow() {
        let mut p = InferenceProfiler::default_enabled();
        p.start();

        // Simulate layer profiling
        p.begin_span("transformer_block_0");
        p.begin_span("self_attention");
        p.record_alloc(8192, "qkv_buffer");
        p.end_span();
        p.begin_span("feed_forward");
        p.end_span();
        p.end_span();

        p.begin_span("transformer_block_1");
        p.begin_span("self_attention");
        p.end_span();
        p.begin_span("feed_forward");
        p.end_span();
        p.end_span();

        p.stop();

        // Verify structure
        let tl = p.get_timeline();
        assert_eq!(tl.len(), 6);
        assert_eq!(tl.root_spans().len(), 2);

        // Verify export
        let export = p.export();
        assert!(!export.is_empty());

        // Verify memory
        assert_eq!(p.get_memory().allocation_count(), 1);
        assert_eq!(p.get_memory().current_memory(), 8192);

        // Verify stats
        let stats = p.get_aggregated_stats();
        assert!(!stats.is_empty());
    }

    #[test]
    fn test_many_spans_performance() {
        let mut tl = ProfilerTimeline::new();
        for i in 0..1000u64 {
            tl.begin_span("op", i * 10);
            tl.end_span(i * 10 + 5);
        }
        assert_eq!(tl.len(), 1000);
        let stats = ProfilerAggregator::aggregate(&tl);
        assert_eq!(stats[0].count, 1000);
    }

    #[test]
    fn test_span_id_uniqueness() {
        let mut tl = ProfilerTimeline::new();
        let mut ids = Vec::new();
        for _ in 0..50 {
            let id = tl.begin_span("span", 0);
            ids.push(id);
            tl.end_span(10);
        }
        // All ids should be unique
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len());
    }

    #[test]
    fn test_memory_event_tags() {
        let mut mp = MemoryProfiler::new();
        mp.record_alloc(0, 100, "weights");
        mp.record_alloc(10, 200, "activations");
        mp.record_dealloc(20, 50, "temp");
        assert_eq!(mp.events()[0].tag, "weights");
        assert_eq!(mp.events()[1].tag, "activations");
        assert_eq!(mp.events()[2].tag, "temp");
    }

    #[test]
    fn test_chrome_export_valid_json_structure() {
        let mut tl = ProfilerTimeline::new();
        for i in 0..5u64 {
            tl.begin_span(format!("op_{i}"), i * 100);
            tl.end_span(i * 100 + 50);
        }
        let parsed = ChromeTraceExporter::export_value(&tl);
        let events = parsed.as_array().unwrap();
        assert_eq!(events.len(), 5);
        for event in events {
            assert!(event["name"].is_string());
            assert!(event["ph"].is_string());
            assert!(event["ts"].is_number());
            assert!(event["dur"].is_number());
            assert!(event["pid"].is_number());
            assert!(event["tid"].is_number());
        }
    }

    #[test]
    fn test_bottleneck_batching_suggestion() {
        let mut tl = ProfilerTimeline::new();
        // Many very short-duration spans
        for i in 0..20u64 {
            tl.begin_span("micro_kernel", i * 2);
            tl.end_span(i * 2 + 1); // 1 ns each < 1000 ns threshold
        }
        // Need total wall time to be non-trivial
        tl.begin_span("filler", 40);
        tl.end_span(10040);
        let bottlenecks = BottleneckDetector::new(5).detect(&tl);
        let micro = bottlenecks.iter().find(|b| b.span_name == "micro_kernel");
        if let Some(b) = micro {
            assert!(b.suggestion.contains("batching"));
        }
    }

    #[test]
    fn test_aggregator_min_max_single() {
        let mut tl = ProfilerTimeline::new();
        tl.begin_span("only", 0);
        tl.end_span(500);
        let stats = ProfilerAggregator::aggregate(&tl);
        assert_eq!(stats[0].min_time_ns, 500);
        assert_eq!(stats[0].max_time_ns, 500);
        assert_eq!(stats[0].avg_time_ns, 500);
    }

    #[test]
    fn test_profiler_disabled_config() {
        let config = ProfilerConfig { enabled: false, ..Default::default() };
        let mut p = InferenceProfiler::new(config);
        p.start();
        let id = p.begin_span("should_not_record");
        assert!(id.is_none());
        p.end_span();
        p.stop();
        assert!(p.get_timeline().is_empty());
    }
}
