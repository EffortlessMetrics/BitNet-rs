//! Comprehensive inference profiling for performance analysis.
//!
//! Provides hierarchical timing spans, per-kernel statistics, memory tracking,
//! bottleneck analysis, Chrome trace export, and live rolling-window profiling.
//! All implementations are CPU reference code — no OpenCL runtime required.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// ProfileSpan — named timing span
// ---------------------------------------------------------------------------

/// A named timing span with start/end/duration.
#[derive(Debug, Clone)]
pub struct ProfileSpan {
    /// Human-readable span name.
    pub name: String,
    /// Offset from the profiling session start (microseconds).
    pub start_us: u64,
    /// Offset from the profiling session start at span end (microseconds).
    pub end_us: u64,
    /// Optional parent span index for tree building.
    pub parent: Option<usize>,
    /// Category tag for grouping (e.g. "kernel", "memory", "sync").
    pub category: String,
}

impl ProfileSpan {
    /// Create a completed span.
    pub fn new(
        name: impl Into<String>,
        start_us: u64,
        end_us: u64,
        category: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            start_us,
            end_us: end_us.max(start_us),
            parent: None,
            category: category.into(),
        }
    }

    /// Create a span with a parent reference.
    pub fn with_parent(mut self, parent: usize) -> Self {
        self.parent = Some(parent);
        self
    }

    /// Duration in microseconds.
    pub fn duration_us(&self) -> u64 {
        self.end_us - self.start_us
    }

    /// Duration as `std::time::Duration`.
    pub fn duration(&self) -> Duration {
        Duration::from_micros(self.duration_us())
    }
}

impl fmt::Display for ProfileSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{:.3}ms] ({})",
            self.name,
            self.duration_us() as f64 / 1000.0,
            self.category,
        )
    }
}

// ---------------------------------------------------------------------------
// ProfileTree — hierarchical span tree
// ---------------------------------------------------------------------------

/// Hierarchical tree of profiling spans (parent-child relationships).
#[derive(Debug, Clone, Default)]
pub struct ProfileTree {
    spans: Vec<ProfileSpan>,
}

impl ProfileTree {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self { spans: Vec::new() }
    }

    /// Add a root span (no parent). Returns the span index.
    pub fn add_root(&mut self, span: ProfileSpan) -> usize {
        let idx = self.spans.len();
        self.spans.push(span);
        idx
    }

    /// Add a child span under `parent`. Returns the span index.
    pub fn add_child(&mut self, parent: usize, mut span: ProfileSpan) -> usize {
        span.parent = Some(parent);
        let idx = self.spans.len();
        self.spans.push(span);
        idx
    }

    /// Total number of spans.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether the tree has no spans.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Get a span by index.
    pub fn get(&self, index: usize) -> Option<&ProfileSpan> {
        self.spans.get(index)
    }

    /// Iterate over all spans.
    pub fn spans(&self) -> &[ProfileSpan] {
        &self.spans
    }

    /// Indices of root spans (those with no parent).
    pub fn roots(&self) -> Vec<usize> {
        self.spans
            .iter()
            .enumerate()
            .filter(|(_, s)| s.parent.is_none())
            .map(|(i, _)| i)
            .collect()
    }

    /// Direct children of a given span index.
    pub fn children(&self, parent: usize) -> Vec<usize> {
        self.spans
            .iter()
            .enumerate()
            .filter(|(_, s)| s.parent == Some(parent))
            .map(|(i, _)| i)
            .collect()
    }

    /// Depth of a span (root = 0).
    pub fn depth(&self, index: usize) -> usize {
        let mut d = 0;
        let mut cur = index;
        while let Some(p) = self.spans.get(cur).and_then(|s| s.parent) {
            d += 1;
            cur = p;
        }
        d
    }

    /// Sum of direct child durations for a given span.
    pub fn child_duration_sum_us(&self, parent: usize) -> u64 {
        self.children(parent).iter().map(|&i| self.spans[i].duration_us()).sum()
    }

    /// Max depth in the tree.
    pub fn max_depth(&self) -> usize {
        (0..self.spans.len()).map(|i| self.depth(i)).max().unwrap_or(0)
    }
}

impl fmt::Display for ProfileTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, span) in self.spans.iter().enumerate() {
            let indent = "  ".repeat(self.depth(i));
            writeln!(f, "{indent}{span}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// KernelProfile — per-kernel statistics
// ---------------------------------------------------------------------------

/// Accumulated statistics for a single named kernel.
#[derive(Debug, Clone)]
pub struct KernelProfile {
    /// Kernel name.
    pub name: String,
    /// Number of dispatches recorded.
    pub dispatch_count: u64,
    /// Total accumulated time (microseconds).
    pub total_us: u64,
    /// Minimum single-dispatch time (microseconds).
    pub min_us: u64,
    /// Maximum single-dispatch time (microseconds).
    pub max_us: u64,
    /// All recorded durations for percentile computation.
    durations: Vec<u64>,
    /// Estimated FLOPs per dispatch (0 if unknown).
    pub flops_per_dispatch: u64,
}

impl KernelProfile {
    /// Create a new empty profile for the given kernel name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dispatch_count: 0,
            total_us: 0,
            min_us: u64::MAX,
            max_us: 0,
            durations: Vec::new(),
            flops_per_dispatch: 0,
        }
    }

    /// Set estimated FLOPs per dispatch.
    pub fn with_flops(mut self, flops: u64) -> Self {
        self.flops_per_dispatch = flops;
        self
    }

    /// Record a single dispatch duration.
    pub fn record(&mut self, duration_us: u64) {
        self.dispatch_count += 1;
        self.total_us += duration_us;
        self.min_us = self.min_us.min(duration_us);
        self.max_us = self.max_us.max(duration_us);
        self.durations.push(duration_us);
    }

    /// Average time per dispatch in microseconds.
    pub fn avg_us(&self) -> f64 {
        if self.dispatch_count == 0 {
            0.0
        } else {
            self.total_us as f64 / self.dispatch_count as f64
        }
    }

    /// P99 latency in microseconds. Returns 0 if no data.
    pub fn p99_us(&self) -> u64 {
        percentile(&self.durations, 99)
    }

    /// P50 (median) latency in microseconds.
    pub fn p50_us(&self) -> u64 {
        percentile(&self.durations, 50)
    }

    /// Estimated GFLOP/s based on average dispatch time.
    pub fn gflops(&self) -> f64 {
        if self.dispatch_count == 0 || self.flops_per_dispatch == 0 {
            return 0.0;
        }
        let avg_seconds = self.avg_us() / 1_000_000.0;
        if avg_seconds == 0.0 {
            return 0.0;
        }
        (self.flops_per_dispatch as f64) / avg_seconds / 1e9
    }
}

impl fmt::Display for KernelProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: dispatches={} avg={:.1}µs min={}µs max={}µs p99={}µs",
            self.name,
            self.dispatch_count,
            self.avg_us(),
            self.min_us,
            self.max_us,
            self.p99_us(),
        )?;
        let gflops = self.gflops();
        if gflops > 0.0 {
            write!(f, " {gflops:.2} GFLOP/s")?;
        }
        Ok(())
    }
}

/// Compute the `p`-th percentile of a duration slice (0–100). Returns 0 if empty.
fn percentile(values: &[u64], p: u32) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted: Vec<u64> = values.to_vec();
    sorted.sort_unstable();
    let idx = ((p as f64 / 100.0) * (sorted.len() as f64 - 1.0)).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ---------------------------------------------------------------------------
// MemoryProfile — allocation timeline and peak tracking
// ---------------------------------------------------------------------------

/// Tracks memory allocations over time for peak and fragmentation analysis.
#[derive(Debug, Clone)]
pub struct MemoryProfile {
    /// Chronological allocation events.
    events: Vec<MemoryEvent>,
    /// Currently live allocations keyed by id.
    live: HashMap<u64, AllocationInfo>,
    /// Running total of currently allocated bytes.
    current_bytes: u64,
    /// High-water mark.
    peak_bytes: u64,
    /// Monotonic allocation id counter.
    next_id: u64,
}

/// A single allocation or deallocation event.
#[derive(Debug, Clone)]
pub struct MemoryEvent {
    /// Allocation id.
    pub id: u64,
    /// Kind of event.
    pub kind: MemoryEventKind,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Timestamp offset in microseconds from profile start.
    pub timestamp_us: u64,
    /// Human-readable label.
    pub label: String,
}

/// Kind of memory event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryEventKind {
    Allocate,
    Deallocate,
}

/// Info about a live allocation.
#[derive(Debug, Clone)]
struct AllocationInfo {
    size_bytes: u64,
    label: String,
}

impl MemoryProfile {
    /// Create an empty profile.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            live: HashMap::new(),
            current_bytes: 0,
            peak_bytes: 0,
            next_id: 0,
        }
    }

    /// Record an allocation. Returns the allocation id.
    pub fn allocate(
        &mut self,
        size_bytes: u64,
        timestamp_us: u64,
        label: impl Into<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let label = label.into();
        self.events.push(MemoryEvent {
            id,
            kind: MemoryEventKind::Allocate,
            size_bytes,
            timestamp_us,
            label: label.clone(),
        });
        self.live.insert(id, AllocationInfo { size_bytes, label });
        self.current_bytes += size_bytes;
        self.peak_bytes = self.peak_bytes.max(self.current_bytes);
        id
    }

    /// Record a deallocation.
    pub fn deallocate(&mut self, id: u64, timestamp_us: u64) {
        if let Some(info) = self.live.remove(&id) {
            self.current_bytes = self.current_bytes.saturating_sub(info.size_bytes);
            self.events.push(MemoryEvent {
                id,
                kind: MemoryEventKind::Deallocate,
                size_bytes: info.size_bytes,
                timestamp_us,
                label: info.label,
            });
        }
    }

    /// Peak memory in bytes.
    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    /// Currently allocated bytes.
    pub fn current_bytes(&self) -> u64 {
        self.current_bytes
    }

    /// Number of currently live allocations.
    pub fn live_count(&self) -> usize {
        self.live.len()
    }

    /// Total number of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// All events in chronological order.
    pub fn events(&self) -> &[MemoryEvent] {
        &self.events
    }

    /// Fragmentation estimate: 1.0 − (largest_free / total_free).
    ///
    /// This CPU reference implementation uses a simplified heuristic:
    /// ratio of live-allocation count to peak allocations ever seen.
    /// Returns 0.0 when nothing is allocated.
    pub fn fragmentation_estimate(&self) -> f64 {
        if self.current_bytes == 0 || self.peak_bytes == 0 {
            return 0.0;
        }
        // Heuristic: more live allocations relative to total bytes = more fragmented
        let live_count = self.live.len() as f64;
        if live_count <= 1.0 {
            return 0.0;
        }
        // Normalise by log to keep in [0, 1)
        let frag = (live_count.ln()) / (live_count.ln() + 1.0);
        frag.clamp(0.0, 1.0)
    }
}

impl Default for MemoryProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MemoryProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryProfile: current={:.2}MB peak={:.2}MB live={} frag={:.2}%",
            self.current_bytes as f64 / (1024.0 * 1024.0),
            self.peak_bytes as f64 / (1024.0 * 1024.0),
            self.live_count(),
            self.fragmentation_estimate() * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// InferenceProfile — full inference pass timing
// ---------------------------------------------------------------------------

/// Timing breakdown of a complete inference pass.
#[derive(Debug, Clone)]
pub struct InferenceProfile {
    /// Time spent in the prefill (prompt encoding) phase.
    pub prefill_us: u64,
    /// Per-token decode times in microseconds.
    pub token_times_us: Vec<u64>,
    /// Total wall-clock time for the entire pass.
    pub total_us: u64,
    /// Number of prompt tokens processed.
    pub prompt_tokens: u32,
    /// Number of tokens generated.
    pub generated_tokens: u32,
}

impl InferenceProfile {
    /// Create a new inference profile.
    pub fn new(
        prefill_us: u64,
        token_times_us: Vec<u64>,
        total_us: u64,
        prompt_tokens: u32,
        generated_tokens: u32,
    ) -> Self {
        Self { prefill_us, token_times_us, total_us, prompt_tokens, generated_tokens }
    }

    /// Average per-token decode time in microseconds.
    pub fn avg_token_us(&self) -> f64 {
        if self.token_times_us.is_empty() {
            return 0.0;
        }
        self.token_times_us.iter().sum::<u64>() as f64 / self.token_times_us.len() as f64
    }

    /// Tokens per second (decode phase only).
    pub fn tokens_per_second(&self) -> f64 {
        let avg = self.avg_token_us();
        if avg == 0.0 {
            return 0.0;
        }
        1_000_000.0 / avg
    }

    /// Time to first token in microseconds (== prefill_us).
    pub fn time_to_first_token_us(&self) -> u64 {
        self.prefill_us
    }

    /// Prefill tokens per second.
    pub fn prefill_tokens_per_second(&self) -> f64 {
        if self.prefill_us == 0 || self.prompt_tokens == 0 {
            return 0.0;
        }
        self.prompt_tokens as f64 / (self.prefill_us as f64 / 1_000_000.0)
    }

    /// P99 per-token latency in microseconds.
    pub fn p99_token_us(&self) -> u64 {
        percentile(&self.token_times_us, 99)
    }

    /// Decode overhead: total − prefill − sum(token_times).
    pub fn overhead_us(&self) -> u64 {
        let sum: u64 = self.token_times_us.iter().sum();
        self.total_us.saturating_sub(self.prefill_us + sum)
    }
}

impl fmt::Display for InferenceProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InferenceProfile: prefill={:.1}ms tokens={} avg={:.1}ms/tok {:.1} tok/s total={:.1}ms",
            self.prefill_us as f64 / 1000.0,
            self.generated_tokens,
            self.avg_token_us() / 1000.0,
            self.tokens_per_second(),
            self.total_us as f64 / 1000.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Bottleneck — identified performance bottleneck
// ---------------------------------------------------------------------------

/// A single identified bottleneck.
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Description of the bottleneck.
    pub description: String,
    /// Severity score (higher = worse). Normalised to [0, 100].
    pub severity: f64,
    /// Category (e.g. "compute", "memory", "sync").
    pub category: String,
    /// Suggested action.
    pub suggestion: String,
}

impl fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.0}] {}: {} → {}",
            self.severity, self.category, self.description, self.suggestion,
        )
    }
}

// ---------------------------------------------------------------------------
// BottleneckAnalyzer
// ---------------------------------------------------------------------------

/// Identifies top-N performance bottlenecks from kernel and memory profiles.
pub struct BottleneckAnalyzer;

impl BottleneckAnalyzer {
    /// Analyse kernel profiles and return the top `n` bottlenecks sorted by severity.
    pub fn analyse_kernels(kernels: &[KernelProfile], n: usize) -> Vec<Bottleneck> {
        let total_us: u64 = kernels.iter().map(|k| k.total_us).sum();
        if total_us == 0 {
            return Vec::new();
        }

        let mut bottlenecks = Vec::new();

        for kernel in kernels {
            // Fraction of total time
            let fraction = kernel.total_us as f64 / total_us as f64;
            let severity = fraction * 100.0;
            if severity < 1.0 {
                continue;
            }
            bottlenecks.push(Bottleneck {
                description: format!(
                    "{} consumes {:.1}% of total kernel time ({} dispatches)",
                    kernel.name,
                    severity,
                    kernel.dispatch_count,
                ),
                severity,
                category: "compute".into(),
                suggestion: format!(
                    "Optimise {} kernel (avg {:.1}µs, p99 {}µs)",
                    kernel.name,
                    kernel.avg_us(),
                    kernel.p99_us(),
                ),
            });

            // High variance detection
            if kernel.dispatch_count >= 2 && kernel.max_us > 0 {
                let ratio = kernel.max_us as f64 / kernel.min_us.max(1) as f64;
                if ratio > 10.0 {
                    bottlenecks.push(Bottleneck {
                        description: format!(
                            "{} has high variance (max/min = {:.1}×)",
                            kernel.name, ratio,
                        ),
                        severity: (ratio.ln() * 10.0).min(100.0),
                        category: "variance".into(),
                        suggestion: format!(
                            "Investigate {} dispatch variability",
                            kernel.name,
                        ),
                    });
                }
            }
        }

        bottlenecks.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
        bottlenecks.truncate(n);
        bottlenecks
    }

    /// Analyse memory profile for bottlenecks.
    pub fn analyse_memory(memory: &MemoryProfile, n: usize) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        let frag = memory.fragmentation_estimate();
        if frag > 0.3 {
            bottlenecks.push(Bottleneck {
                description: format!("High memory fragmentation ({:.0}%)", frag * 100.0),
                severity: frag * 100.0,
                category: "memory".into(),
                suggestion: "Consider memory pool or defragmentation".into(),
            });
        }

        if memory.peak_bytes() > 0 && memory.current_bytes() > 0 {
            let utilisation =
                memory.current_bytes() as f64 / memory.peak_bytes() as f64;
            if utilisation < 0.5 && memory.live_count() > 0 {
                bottlenecks.push(Bottleneck {
                    description: format!(
                        "Low memory utilisation ({:.0}% of peak)",
                        utilisation * 100.0,
                    ),
                    severity: (1.0 - utilisation) * 50.0,
                    category: "memory".into(),
                    suggestion: "Release unused allocations earlier".into(),
                });
            }
        }

        bottlenecks.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
        bottlenecks.truncate(n);
        bottlenecks
    }

    /// Combined analysis across kernel and memory profiles.
    pub fn analyse(
        kernels: &[KernelProfile],
        memory: &MemoryProfile,
        n: usize,
    ) -> Vec<Bottleneck> {
        let mut all = Self::analyse_kernels(kernels, n);
        all.extend(Self::analyse_memory(memory, n));
        all.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(n);
        all
    }
}

// ---------------------------------------------------------------------------
// ProfileExporter — Chrome trace format (JSON)
// ---------------------------------------------------------------------------

/// Exports profiling data to Chrome `chrome://tracing` JSON format.
pub struct ProfileExporter;

impl ProfileExporter {
    /// Export a `ProfileTree` to Chrome trace JSON (array format).
    pub fn to_chrome_trace(tree: &ProfileTree) -> String {
        let mut events = Vec::new();
        for (i, span) in tree.spans().iter().enumerate() {
            let tid = span.parent.unwrap_or(0);
            let depth = tree.depth(i);
            // Duration event (B/E pair collapsed into X for simplicity)
            events.push(format!(
                concat!(
                    "{{",
                    "\"name\":\"{name}\",",
                    "\"cat\":\"{cat}\",",
                    "\"ph\":\"X\",",
                    "\"ts\":{ts},",
                    "\"dur\":{dur},",
                    "\"pid\":0,",
                    "\"tid\":{tid},",
                    "\"args\":{{\"depth\":{depth}}}",
                    "}}"
                ),
                name = escape_json(&span.name),
                cat = escape_json(&span.category),
                ts = span.start_us,
                dur = span.duration_us(),
                tid = tid,
                depth = depth,
            ));
        }
        format!("[{}]", events.join(","))
    }

    /// Export kernel profiles as Chrome trace counter events.
    pub fn kernels_to_chrome_trace(kernels: &[KernelProfile]) -> String {
        let mut events = Vec::new();
        let mut ts: u64 = 0;
        for kernel in kernels {
            for &dur in &kernel.durations {
                events.push(format!(
                    concat!(
                        "{{",
                        "\"name\":\"{name}\",",
                        "\"cat\":\"kernel\",",
                        "\"ph\":\"X\",",
                        "\"ts\":{ts},",
                        "\"dur\":{dur},",
                        "\"pid\":0,",
                        "\"tid\":1",
                        "}}"
                    ),
                    name = escape_json(&kernel.name),
                    ts = ts,
                    dur = dur,
                ));
                ts += dur;
            }
        }
        format!("[{}]", events.join(","))
    }

    /// Export an `InferenceProfile` to Chrome trace JSON.
    pub fn inference_to_chrome_trace(profile: &InferenceProfile) -> String {
        let mut events = Vec::new();
        // Prefill span
        events.push(format!(
            concat!(
                "{{",
                "\"name\":\"prefill\",",
                "\"cat\":\"inference\",",
                "\"ph\":\"X\",",
                "\"ts\":0,",
                "\"dur\":{dur},",
                "\"pid\":0,",
                "\"tid\":0",
                "}}"
            ),
            dur = profile.prefill_us,
        ));
        // Per-token spans
        let mut ts = profile.prefill_us;
        for (i, &dur) in profile.token_times_us.iter().enumerate() {
            events.push(format!(
                concat!(
                    "{{",
                    "\"name\":\"token_{idx}\",",
                    "\"cat\":\"inference\",",
                    "\"ph\":\"X\",",
                    "\"ts\":{ts},",
                    "\"dur\":{dur},",
                    "\"pid\":0,",
                    "\"tid\":0",
                    "}}"
                ),
                idx = i,
                ts = ts,
                dur = dur,
            ));
            ts += dur;
        }
        format!("[{}]", events.join(","))
    }
}

/// Minimal JSON string escaping (quotes, backslashes, control chars).
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// LiveProfiler — real-time profiling with rolling window
// ---------------------------------------------------------------------------

/// Real-time profiler that maintains a rolling window of recent spans.
#[derive(Debug)]
pub struct LiveProfiler {
    /// Fixed-capacity ring buffer of recent spans.
    buffer: Vec<ProfileSpan>,
    /// Maximum number of spans to retain.
    capacity: usize,
    /// Write cursor (wraps around).
    cursor: usize,
    /// Total spans ever recorded.
    total_recorded: u64,
    /// Per-kernel running aggregates.
    kernel_stats: HashMap<String, KernelProfile>,
    /// Session start time.
    start: Instant,
    /// Whether profiling is enabled.
    enabled: bool,
}

impl LiveProfiler {
    /// Create a new live profiler with the given ring-buffer capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity.min(64 * 1024)),
            capacity: capacity.max(1),
            cursor: 0,
            total_recorded: 0,
            kernel_stats: HashMap::new(),
            start: Instant::now(),
            enabled: true,
        }
    }

    /// Enable or disable the profiler.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Whether profiling is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record a span. If the buffer is full, the oldest span is overwritten.
    pub fn record(&mut self, span: ProfileSpan) {
        if !self.enabled {
            return;
        }

        // Update kernel stats
        self.kernel_stats
            .entry(span.name.clone())
            .or_insert_with(|| KernelProfile::new(span.name.clone()))
            .record(span.duration_us());

        // Ring buffer insert
        if self.buffer.len() < self.capacity {
            self.buffer.push(span);
        } else {
            self.buffer[self.cursor % self.capacity] = span;
        }
        self.cursor += 1;
        self.total_recorded += 1;
    }

    /// Record a quick span from name, duration, and category.
    pub fn record_quick(
        &mut self,
        name: impl Into<String>,
        duration_us: u64,
        category: impl Into<String>,
    ) {
        let elapsed = self.start.elapsed().as_micros() as u64;
        let start = elapsed.saturating_sub(duration_us);
        self.record(ProfileSpan::new(name, start, elapsed, category));
    }

    /// Spans currently in the rolling window (newest first).
    pub fn recent_spans(&self) -> Vec<&ProfileSpan> {
        if self.buffer.len() < self.capacity {
            // Not yet wrapped — return in reverse insertion order
            self.buffer.iter().rev().collect()
        } else {
            // Wrapped — newest is at cursor-1, oldest at cursor
            let cap = self.capacity;
            let mut result = Vec::with_capacity(cap);
            for i in 0..cap {
                let idx = (self.cursor + cap - 1 - i) % cap;
                result.push(&self.buffer[idx]);
            }
            result
        }
    }

    /// Number of spans currently in the buffer.
    pub fn window_size(&self) -> usize {
        self.buffer.len()
    }

    /// Total number of spans ever recorded.
    pub fn total_recorded(&self) -> u64 {
        self.total_recorded
    }

    /// Get accumulated kernel statistics.
    pub fn kernel_stats(&self) -> &HashMap<String, KernelProfile> {
        &self.kernel_stats
    }

    /// Summary of top-N kernels by total time.
    pub fn top_kernels(&self, n: usize) -> Vec<&KernelProfile> {
        let mut profiles: Vec<&KernelProfile> = self.kernel_stats.values().collect();
        profiles.sort_by(|a, b| b.total_us.cmp(&a.total_us));
        profiles.truncate(n);
        profiles
    }

    /// Elapsed time since profiler creation.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Reset all statistics and clear the buffer.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.cursor = 0;
        self.total_recorded = 0;
        self.kernel_stats.clear();
        self.start = Instant::now();
    }
}

// ---------------------------------------------------------------------------
// ProfileConfig — configurable profiling
// ---------------------------------------------------------------------------

/// Configuration for the profiling system.
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// Whether profiling is globally enabled.
    pub enabled: bool,
    /// Spans matching these categories are recorded (empty = all).
    pub enabled_categories: Vec<String>,
    /// Sampling rate in [0.0, 1.0] — 1.0 means capture everything.
    pub sampling_rate: f64,
    /// Maximum ring-buffer size for the live profiler.
    pub buffer_size: usize,
    /// Whether to record memory events.
    pub track_memory: bool,
    /// Whether to collect per-kernel statistics.
    pub track_kernels: bool,
}

impl ProfileConfig {
    /// Default configuration: everything enabled, full sampling.
    pub fn default_config() -> Self {
        Self {
            enabled: true,
            enabled_categories: Vec::new(),
            sampling_rate: 1.0,
            buffer_size: 4096,
            track_memory: true,
            track_kernels: true,
        }
    }

    /// Disabled configuration (no-op profiling).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            enabled_categories: Vec::new(),
            sampling_rate: 0.0,
            buffer_size: 0,
            track_memory: false,
            track_kernels: false,
        }
    }

    /// Minimal overhead configuration: low sampling, small buffer.
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            enabled_categories: Vec::new(),
            sampling_rate: 0.01,
            buffer_size: 256,
            track_memory: false,
            track_kernels: true,
        }
    }

    /// Check if a category should be recorded.
    pub fn should_record(&self, category: &str) -> bool {
        if !self.enabled {
            return false;
        }
        if self.enabled_categories.is_empty() {
            return true;
        }
        self.enabled_categories.iter().any(|c| c == category)
    }

    /// Check if a sample should be recorded based on the sampling rate.
    /// Uses a deterministic hash of the `sample_id` for reproducibility.
    pub fn should_sample(&self, sample_id: u64) -> bool {
        if !self.enabled {
            return false;
        }
        if self.sampling_rate >= 1.0 {
            return true;
        }
        if self.sampling_rate <= 0.0 {
            return false;
        }
        // Simple deterministic sampling based on hash
        let hash = sample_id.wrapping_mul(0x517cc1b727220a95);
        let threshold = (self.sampling_rate * u64::MAX as f64) as u64;
        hash < threshold
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.sampling_rate < 0.0 || self.sampling_rate > 1.0 {
            return Err(format!(
                "sampling_rate must be in [0.0, 1.0], got {}",
                self.sampling_rate
            ));
        }
        Ok(())
    }
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

impl fmt::Display for ProfileConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProfileConfig: enabled={} sampling={:.0}% buffer={} mem={} kernels={}",
            self.enabled,
            self.sampling_rate * 100.0,
            self.buffer_size,
            self.track_memory,
            self.track_kernels,
        )
    }
}

// ---------------------------------------------------------------------------
// ScopedTimer — RAII timing helper
// ---------------------------------------------------------------------------

/// RAII timer that records its duration on drop.
#[derive(Debug)]
pub struct ScopedTimer {
    name: String,
    category: String,
    start: Instant,
    /// Collected duration (set on `stop`).
    duration: Option<Duration>,
}

impl ScopedTimer {
    /// Start a new scoped timer.
    pub fn start(name: impl Into<String>, category: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            category: category.into(),
            start: Instant::now(),
            duration: None,
        }
    }

    /// Manually stop and return elapsed duration.
    pub fn stop(&mut self) -> Duration {
        let d = self.start.elapsed();
        self.duration = Some(d);
        d
    }

    /// Elapsed time since start (does not stop the timer).
    pub fn elapsed(&self) -> Duration {
        self.duration.unwrap_or_else(|| self.start.elapsed())
    }

    /// Name of this timer.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Category of this timer.
    pub fn category(&self) -> &str {
        &self.category
    }

    /// Convert into a `ProfileSpan` using the given session-start offset.
    pub fn into_span(mut self, session_start: Instant) -> ProfileSpan {
        if self.duration.is_none() {
            self.stop();
        }
        let start_us = self
            .start
            .checked_duration_since(session_start)
            .unwrap_or_default()
            .as_micros() as u64;
        let end_us = start_us + self.elapsed().as_micros() as u64;
        ProfileSpan::new(self.name, start_us, end_us, self.category)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- ProfileSpan --------------------------------------------------------

    #[test]
    fn span_duration_basic() {
        let span = ProfileSpan::new("test", 100, 500, "kernel");
        assert_eq!(span.duration_us(), 400);
        assert_eq!(span.duration(), Duration::from_micros(400));
    }

    #[test]
    fn span_zero_duration() {
        let span = ProfileSpan::new("zero", 42, 42, "misc");
        assert_eq!(span.duration_us(), 0);
    }

    #[test]
    fn span_end_before_start_clamped() {
        // Constructor clamps end to max(end, start)
        let span = ProfileSpan::new("bad", 100, 50, "misc");
        assert_eq!(span.start_us, 100);
        assert_eq!(span.end_us, 100);
        assert_eq!(span.duration_us(), 0);
    }

    #[test]
    fn span_with_parent() {
        let span = ProfileSpan::new("child", 10, 20, "kernel").with_parent(3);
        assert_eq!(span.parent, Some(3));
    }

    #[test]
    fn span_display() {
        let span = ProfileSpan::new("matmul", 0, 1500, "kernel");
        let s = span.to_string();
        assert!(s.contains("matmul"));
        assert!(s.contains("kernel"));
        assert!(s.contains("1.500ms"));
    }

    #[test]
    fn span_large_duration() {
        let span = ProfileSpan::new("long", 0, 60_000_000, "compute");
        assert_eq!(span.duration_us(), 60_000_000); // 60 seconds
        assert_eq!(span.duration(), Duration::from_secs(60));
    }

    // -- ProfileTree --------------------------------------------------------

    #[test]
    fn tree_empty() {
        let tree = ProfileTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.roots().is_empty());
        assert_eq!(tree.max_depth(), 0);
    }

    #[test]
    fn tree_single_root() {
        let mut tree = ProfileTree::new();
        let root = tree.add_root(ProfileSpan::new("root", 0, 100, "top"));
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.roots(), vec![root]);
        assert_eq!(tree.depth(root), 0);
    }

    #[test]
    fn tree_parent_child() {
        let mut tree = ProfileTree::new();
        let root = tree.add_root(ProfileSpan::new("root", 0, 1000, "top"));
        let child = tree.add_child(root, ProfileSpan::new("child", 10, 500, "kernel"));
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.children(root), vec![child]);
        assert_eq!(tree.depth(child), 1);
        assert_eq!(tree.get(child).unwrap().parent, Some(root));
    }

    #[test]
    fn tree_multiple_children() {
        let mut tree = ProfileTree::new();
        let root = tree.add_root(ProfileSpan::new("root", 0, 1000, "top"));
        let c1 = tree.add_child(root, ProfileSpan::new("a", 10, 200, "k"));
        let c2 = tree.add_child(root, ProfileSpan::new("b", 200, 500, "k"));
        let c3 = tree.add_child(root, ProfileSpan::new("c", 500, 900, "k"));
        assert_eq!(tree.children(root), vec![c1, c2, c3]);
    }

    #[test]
    fn tree_nested_depth() {
        let mut tree = ProfileTree::new();
        let r = tree.add_root(ProfileSpan::new("r", 0, 1000, "t"));
        let c = tree.add_child(r, ProfileSpan::new("c", 10, 900, "t"));
        let gc = tree.add_child(c, ProfileSpan::new("gc", 20, 800, "t"));
        let ggc = tree.add_child(gc, ProfileSpan::new("ggc", 30, 700, "t"));
        assert_eq!(tree.depth(r), 0);
        assert_eq!(tree.depth(c), 1);
        assert_eq!(tree.depth(gc), 2);
        assert_eq!(tree.depth(ggc), 3);
        assert_eq!(tree.max_depth(), 3);
    }

    #[test]
    fn tree_child_duration_sum() {
        let mut tree = ProfileTree::new();
        let root = tree.add_root(ProfileSpan::new("root", 0, 1000, "t"));
        tree.add_child(root, ProfileSpan::new("a", 0, 300, "k"));
        tree.add_child(root, ProfileSpan::new("b", 300, 700, "k"));
        assert_eq!(tree.child_duration_sum_us(root), 700);
    }

    #[test]
    fn tree_child_duration_leq_parent() {
        let mut tree = ProfileTree::new();
        let root = tree.add_root(ProfileSpan::new("root", 0, 1000, "t"));
        tree.add_child(root, ProfileSpan::new("a", 10, 400, "k"));
        tree.add_child(root, ProfileSpan::new("b", 400, 800, "k"));
        let parent_dur = tree.get(root).unwrap().duration_us();
        let child_sum = tree.child_duration_sum_us(root);
        assert!(child_sum <= parent_dur, "child sum {child_sum} > parent {parent_dur}");
    }

    #[test]
    fn tree_multiple_roots() {
        let mut tree = ProfileTree::new();
        let r1 = tree.add_root(ProfileSpan::new("r1", 0, 100, "t"));
        let r2 = tree.add_root(ProfileSpan::new("r2", 100, 200, "t"));
        assert_eq!(tree.roots(), vec![r1, r2]);
    }

    #[test]
    fn tree_display() {
        let mut tree = ProfileTree::new();
        let root = tree.add_root(ProfileSpan::new("root", 0, 100, "top"));
        tree.add_child(root, ProfileSpan::new("child", 10, 50, "sub"));
        let s = tree.to_string();
        assert!(s.contains("root"));
        assert!(s.contains("child"));
    }

    #[test]
    fn tree_get_out_of_bounds() {
        let tree = ProfileTree::new();
        assert!(tree.get(0).is_none());
        assert!(tree.get(999).is_none());
    }

    // -- KernelProfile ------------------------------------------------------

    #[test]
    fn kernel_profile_empty() {
        let kp = KernelProfile::new("matmul");
        assert_eq!(kp.dispatch_count, 0);
        assert_eq!(kp.avg_us(), 0.0);
        assert_eq!(kp.p99_us(), 0);
        assert_eq!(kp.gflops(), 0.0);
    }

    #[test]
    fn kernel_profile_single_dispatch() {
        let mut kp = KernelProfile::new("softmax");
        kp.record(100);
        assert_eq!(kp.dispatch_count, 1);
        assert_eq!(kp.total_us, 100);
        assert_eq!(kp.min_us, 100);
        assert_eq!(kp.max_us, 100);
        assert!((kp.avg_us() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn kernel_profile_multiple_dispatches() {
        let mut kp = KernelProfile::new("norm");
        for d in [10, 20, 30, 40, 50] {
            kp.record(d);
        }
        assert_eq!(kp.dispatch_count, 5);
        assert_eq!(kp.total_us, 150);
        assert_eq!(kp.min_us, 10);
        assert_eq!(kp.max_us, 50);
        assert!((kp.avg_us() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn kernel_profile_p99() {
        let mut kp = KernelProfile::new("attn");
        for i in 1..=100 {
            kp.record(i);
        }
        assert!(kp.p99_us() >= 99);
    }

    #[test]
    fn kernel_profile_p50() {
        let mut kp = KernelProfile::new("ffn");
        for i in 1..=100 {
            kp.record(i);
        }
        let p50 = kp.p50_us();
        assert!(p50 >= 49 && p50 <= 51, "p50={p50}");
    }

    #[test]
    fn kernel_profile_gflops() {
        let mut kp = KernelProfile::new("matmul").with_flops(1_000_000_000); // 1 GFLOP
        kp.record(1_000_000); // 1 second
        let gflops = kp.gflops();
        assert!((gflops - 1.0).abs() < 0.01, "gflops={gflops}");
    }

    #[test]
    fn kernel_profile_gflops_zero_flops() {
        let mut kp = KernelProfile::new("noop");
        kp.record(100);
        assert_eq!(kp.gflops(), 0.0);
    }

    #[test]
    fn kernel_profile_display() {
        let mut kp = KernelProfile::new("test");
        kp.record(100);
        let s = kp.to_string();
        assert!(s.contains("test"));
        assert!(s.contains("dispatches=1"));
    }

    #[test]
    fn kernel_profile_zero_duration_dispatch() {
        let mut kp = KernelProfile::new("instant");
        kp.record(0);
        assert_eq!(kp.dispatch_count, 1);
        assert_eq!(kp.min_us, 0);
        assert_eq!(kp.max_us, 0);
        assert_eq!(kp.avg_us(), 0.0);
    }

    // -- percentile helper --------------------------------------------------

    #[test]
    fn percentile_empty() {
        assert_eq!(percentile(&[], 99), 0);
    }

    #[test]
    fn percentile_single() {
        assert_eq!(percentile(&[42], 50), 42);
        assert_eq!(percentile(&[42], 99), 42);
    }

    #[test]
    fn percentile_sorted_input() {
        let data: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile(&data, 0), 1);
        assert!(percentile(&data, 100) >= 100);
    }

    // -- MemoryProfile ------------------------------------------------------

    #[test]
    fn memory_profile_empty() {
        let mp = MemoryProfile::new();
        assert_eq!(mp.peak_bytes(), 0);
        assert_eq!(mp.current_bytes(), 0);
        assert_eq!(mp.live_count(), 0);
        assert_eq!(mp.event_count(), 0);
        assert_eq!(mp.fragmentation_estimate(), 0.0);
    }

    #[test]
    fn memory_profile_allocate_deallocate() {
        let mut mp = MemoryProfile::new();
        let id = mp.allocate(1024, 0, "buf");
        assert_eq!(mp.current_bytes(), 1024);
        assert_eq!(mp.peak_bytes(), 1024);
        assert_eq!(mp.live_count(), 1);

        mp.deallocate(id, 100);
        assert_eq!(mp.current_bytes(), 0);
        assert_eq!(mp.peak_bytes(), 1024); // peak preserved
        assert_eq!(mp.live_count(), 0);
    }

    #[test]
    fn memory_profile_peak_tracking() {
        let mut mp = MemoryProfile::new();
        let a = mp.allocate(1000, 0, "a");
        let b = mp.allocate(2000, 10, "b");
        assert_eq!(mp.peak_bytes(), 3000);
        mp.deallocate(a, 20);
        assert_eq!(mp.current_bytes(), 2000);
        assert_eq!(mp.peak_bytes(), 3000); // still 3000
        mp.deallocate(b, 30);
        assert_eq!(mp.current_bytes(), 0);
        assert_eq!(mp.peak_bytes(), 3000);
    }

    #[test]
    fn memory_profile_multiple_allocs() {
        let mut mp = MemoryProfile::new();
        let ids: Vec<u64> = (0..10).map(|i| mp.allocate(100, i * 10, format!("buf{i}"))).collect();
        assert_eq!(mp.current_bytes(), 1000);
        assert_eq!(mp.live_count(), 10);
        for id in ids {
            mp.deallocate(id, 200);
        }
        assert_eq!(mp.current_bytes(), 0);
        assert_eq!(mp.live_count(), 0);
    }

    #[test]
    fn memory_profile_deallocate_unknown_id() {
        let mut mp = MemoryProfile::new();
        mp.allocate(100, 0, "x");
        mp.deallocate(999, 10); // no-op
        assert_eq!(mp.current_bytes(), 100);
        assert_eq!(mp.live_count(), 1);
    }

    #[test]
    fn memory_profile_event_count() {
        let mut mp = MemoryProfile::new();
        let id = mp.allocate(100, 0, "x");
        mp.deallocate(id, 10);
        assert_eq!(mp.event_count(), 2);
    }

    #[test]
    fn memory_profile_fragmentation_single_alloc() {
        let mut mp = MemoryProfile::new();
        mp.allocate(1024, 0, "single");
        assert_eq!(mp.fragmentation_estimate(), 0.0);
    }

    #[test]
    fn memory_profile_fragmentation_many_allocs() {
        let mut mp = MemoryProfile::new();
        for i in 0..100 {
            mp.allocate(64, i * 10, format!("buf{i}"));
        }
        let frag = mp.fragmentation_estimate();
        assert!(frag > 0.0, "fragmentation should be > 0 with 100 live allocs");
        assert!(frag <= 1.0);
    }

    #[test]
    fn memory_profile_display() {
        let mut mp = MemoryProfile::new();
        mp.allocate(1_048_576, 0, "1MB");
        let s = mp.to_string();
        assert!(s.contains("peak="));
        assert!(s.contains("MB"));
    }

    #[test]
    fn memory_profile_events_chronological() {
        let mut mp = MemoryProfile::new();
        let a = mp.allocate(100, 10, "a");
        mp.allocate(200, 20, "b");
        mp.deallocate(a, 30);
        let events = mp.events();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].timestamp_us, 10);
        assert_eq!(events[1].timestamp_us, 20);
        assert_eq!(events[2].timestamp_us, 30);
    }

    // -- InferenceProfile ---------------------------------------------------

    #[test]
    fn inference_profile_basic() {
        let ip = InferenceProfile::new(
            5000,                          // 5ms prefill
            vec![1000, 1100, 900, 1050],   // 4 tokens
            10000,                         // 10ms total
            128,                           // prompt tokens
            4,                             // generated tokens
        );
        assert_eq!(ip.time_to_first_token_us(), 5000);
        assert_eq!(ip.generated_tokens, 4);
        assert!(ip.avg_token_us() > 900.0 && ip.avg_token_us() < 1200.0);
        assert!(ip.tokens_per_second() > 800.0);
    }

    #[test]
    fn inference_profile_empty_tokens() {
        let ip = InferenceProfile::new(1000, vec![], 1000, 10, 0);
        assert_eq!(ip.avg_token_us(), 0.0);
        assert_eq!(ip.tokens_per_second(), 0.0);
        assert_eq!(ip.p99_token_us(), 0);
    }

    #[test]
    fn inference_profile_prefill_tps() {
        let ip = InferenceProfile::new(1_000_000, vec![100], 1_100_000, 100, 1);
        let pps = ip.prefill_tokens_per_second();
        assert!((pps - 100.0).abs() < 1.0, "prefill tps={pps}");
    }

    #[test]
    fn inference_profile_overhead() {
        let ip = InferenceProfile::new(
            1000,
            vec![200, 200, 200],
            2000, // total=2000, prefill+tokens=1600, overhead=400
            10,
            3,
        );
        assert_eq!(ip.overhead_us(), 400);
    }

    #[test]
    fn inference_profile_p99_token() {
        let ip = InferenceProfile::new(
            100,
            (1..=100).collect(),
            6000,
            10,
            100,
        );
        assert!(ip.p99_token_us() >= 99);
    }

    #[test]
    fn inference_profile_display() {
        let ip = InferenceProfile::new(1000, vec![500, 600], 3000, 8, 2);
        let s = ip.to_string();
        assert!(s.contains("prefill="));
        assert!(s.contains("tok/s"));
    }

    #[test]
    fn inference_profile_zero_prefill() {
        let ip = InferenceProfile::new(0, vec![100], 100, 0, 1);
        assert_eq!(ip.prefill_tokens_per_second(), 0.0);
    }

    // -- BottleneckAnalyzer -------------------------------------------------

    #[test]
    fn bottleneck_empty_kernels() {
        let b = BottleneckAnalyzer::analyse_kernels(&[], 5);
        assert!(b.is_empty());
    }

    #[test]
    fn bottleneck_single_kernel() {
        let mut kp = KernelProfile::new("matmul");
        kp.record(1000);
        let b = BottleneckAnalyzer::analyse_kernels(&[kp], 5);
        assert_eq!(b.len(), 1);
        assert!((b[0].severity - 100.0).abs() < 0.01);
    }

    #[test]
    fn bottleneck_sorted_by_severity() {
        let mut k1 = KernelProfile::new("big");
        for _ in 0..10 {
            k1.record(1000);
        }
        let mut k2 = KernelProfile::new("medium");
        for _ in 0..5 {
            k2.record(800);
        }
        let b = BottleneckAnalyzer::analyse_kernels(&[k1, k2], 5);
        assert!(b.len() >= 2);
        assert!(b[0].severity >= b[1].severity);
    }

    #[test]
    fn bottleneck_top_n_limit() {
        let mut kernels = Vec::new();
        for i in 0..20 {
            let mut kp = KernelProfile::new(format!("k{i}"));
            kp.record((i + 1) * 100);
            kernels.push(kp);
        }
        let b = BottleneckAnalyzer::analyse_kernels(&kernels, 3);
        assert!(b.len() <= 3);
    }

    #[test]
    fn bottleneck_high_variance_detected() {
        let mut kp = KernelProfile::new("flaky");
        kp.record(1);
        kp.record(10_000);
        let b = BottleneckAnalyzer::analyse_kernels(&[kp], 10);
        let variance_bottleneck = b.iter().any(|b| b.category == "variance");
        assert!(variance_bottleneck, "expected variance bottleneck");
    }

    #[test]
    fn bottleneck_memory_fragmentation() {
        let mut mp = MemoryProfile::new();
        for i in 0..100 {
            mp.allocate(64, i * 10, format!("f{i}"));
        }
        let b = BottleneckAnalyzer::analyse_memory(&mp, 5);
        assert!(!b.is_empty(), "expected fragmentation bottleneck");
    }

    #[test]
    fn bottleneck_combined_analysis() {
        let mut kp = KernelProfile::new("matmul");
        kp.record(5000);

        let mut mp = MemoryProfile::new();
        for i in 0..50 {
            mp.allocate(128, i, format!("b{i}"));
        }

        let b = BottleneckAnalyzer::analyse(&[kp], &mp, 10);
        assert!(!b.is_empty());
        // Should be sorted by severity
        for w in b.windows(2) {
            assert!(w[0].severity >= w[1].severity);
        }
    }

    #[test]
    fn bottleneck_display() {
        let b = Bottleneck {
            description: "slow kernel".into(),
            severity: 80.0,
            category: "compute".into(),
            suggestion: "optimise it".into(),
        };
        let s = b.to_string();
        assert!(s.contains("slow kernel"));
        assert!(s.contains("compute"));
    }

    // -- ProfileExporter (Chrome trace) -------------------------------------

    #[test]
    fn chrome_trace_empty_tree() {
        let tree = ProfileTree::new();
        let json = ProfileExporter::to_chrome_trace(&tree);
        assert_eq!(json, "[]");
    }

    #[test]
    fn chrome_trace_valid_json() {
        let mut tree = ProfileTree::new();
        tree.add_root(ProfileSpan::new("root", 0, 1000, "top"));
        let json = ProfileExporter::to_chrome_trace(&tree);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_array());
    }

    #[test]
    fn chrome_trace_contains_span_names() {
        let mut tree = ProfileTree::new();
        let r = tree.add_root(ProfileSpan::new("inference", 0, 5000, "top"));
        tree.add_child(r, ProfileSpan::new("prefill", 0, 2000, "phase"));
        tree.add_child(r, ProfileSpan::new("decode", 2000, 5000, "phase"));
        let json = ProfileExporter::to_chrome_trace(&tree);
        assert!(json.contains("inference"));
        assert!(json.contains("prefill"));
        assert!(json.contains("decode"));
    }

    #[test]
    fn chrome_trace_has_required_fields() {
        let mut tree = ProfileTree::new();
        tree.add_root(ProfileSpan::new("k", 100, 200, "cat"));
        let json = ProfileExporter::to_chrome_trace(&tree);
        assert!(json.contains("\"ph\":\"X\""));
        assert!(json.contains("\"ts\":100"));
        assert!(json.contains("\"dur\":100"));
        assert!(json.contains("\"pid\":0"));
    }

    #[test]
    fn chrome_trace_kernel_export() {
        let mut kp = KernelProfile::new("matmul");
        kp.record(500);
        kp.record(600);
        let json = ProfileExporter::kernels_to_chrome_trace(&[kp]);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn chrome_trace_inference_export() {
        let ip = InferenceProfile::new(1000, vec![200, 300], 2000, 8, 2);
        let json = ProfileExporter::inference_to_chrome_trace(&ip);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 3); // prefill + 2 tokens
    }

    #[test]
    fn chrome_trace_escape_special_chars() {
        let mut tree = ProfileTree::new();
        tree.add_root(ProfileSpan::new("has\"quote", 0, 100, "cat\\slash"));
        let json = ProfileExporter::to_chrome_trace(&tree);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_array());
    }

    #[test]
    fn chrome_trace_timestamps_monotonic() {
        let mut tree = ProfileTree::new();
        tree.add_root(ProfileSpan::new("a", 0, 100, "t"));
        tree.add_root(ProfileSpan::new("b", 100, 300, "t"));
        tree.add_root(ProfileSpan::new("c", 300, 600, "t"));
        let json = ProfileExporter::to_chrome_trace(&tree);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = parsed.as_array().unwrap();
        let mut prev_ts = 0u64;
        for event in arr {
            let ts = event["ts"].as_u64().unwrap();
            assert!(ts >= prev_ts, "timestamps not monotonic: {ts} < {prev_ts}");
            prev_ts = ts;
        }
    }

    // -- LiveProfiler -------------------------------------------------------

    #[test]
    fn live_profiler_basic() {
        let mut lp = LiveProfiler::new(10);
        lp.record(ProfileSpan::new("a", 0, 100, "k"));
        assert_eq!(lp.window_size(), 1);
        assert_eq!(lp.total_recorded(), 1);
    }

    #[test]
    fn live_profiler_rolling_window() {
        let mut lp = LiveProfiler::new(3);
        for i in 0..5 {
            lp.record(ProfileSpan::new(format!("s{i}"), i * 100, (i + 1) * 100, "k"));
        }
        assert_eq!(lp.window_size(), 3);
        assert_eq!(lp.total_recorded(), 5);
        // Most recent should be s4
        let recent = lp.recent_spans();
        assert_eq!(recent[0].name, "s4");
    }

    #[test]
    fn live_profiler_kernel_stats_accumulated() {
        let mut lp = LiveProfiler::new(100);
        for _ in 0..5 {
            lp.record(ProfileSpan::new("matmul", 0, 100, "k"));
        }
        let stats = lp.kernel_stats();
        let matmul = stats.get("matmul").unwrap();
        assert_eq!(matmul.dispatch_count, 5);
    }

    #[test]
    fn live_profiler_disabled() {
        let mut lp = LiveProfiler::new(10);
        lp.set_enabled(false);
        lp.record(ProfileSpan::new("x", 0, 100, "k"));
        assert_eq!(lp.window_size(), 0);
        assert_eq!(lp.total_recorded(), 0);
    }

    #[test]
    fn live_profiler_record_quick() {
        let mut lp = LiveProfiler::new(10);
        lp.record_quick("matmul", 500, "kernel");
        assert_eq!(lp.window_size(), 1);
        let recent = lp.recent_spans();
        assert_eq!(recent[0].name, "matmul");
        assert_eq!(recent[0].category, "kernel");
    }

    #[test]
    fn live_profiler_top_kernels() {
        let mut lp = LiveProfiler::new(100);
        for _ in 0..10 {
            lp.record(ProfileSpan::new("big", 0, 1000, "k"));
        }
        for _ in 0..5 {
            lp.record(ProfileSpan::new("small", 0, 10, "k"));
        }
        let top = lp.top_kernels(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].name, "big");
    }

    #[test]
    fn live_profiler_reset() {
        let mut lp = LiveProfiler::new(10);
        lp.record(ProfileSpan::new("x", 0, 100, "k"));
        lp.reset();
        assert_eq!(lp.window_size(), 0);
        assert_eq!(lp.total_recorded(), 0);
        assert!(lp.kernel_stats().is_empty());
    }

    #[test]
    fn live_profiler_capacity_one() {
        let mut lp = LiveProfiler::new(1);
        lp.record(ProfileSpan::new("a", 0, 100, "k"));
        lp.record(ProfileSpan::new("b", 0, 200, "k"));
        assert_eq!(lp.window_size(), 1);
        let recent = lp.recent_spans();
        assert_eq!(recent[0].name, "b");
    }

    #[test]
    fn live_profiler_elapsed_nonnegative() {
        let lp = LiveProfiler::new(10);
        let _ = lp.elapsed(); // Should not panic
    }

    // -- ProfileConfig ------------------------------------------------------

    #[test]
    fn config_default() {
        let cfg = ProfileConfig::default_config();
        assert!(cfg.enabled);
        assert_eq!(cfg.sampling_rate, 1.0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_disabled() {
        let cfg = ProfileConfig::disabled();
        assert!(!cfg.enabled);
        assert!(!cfg.should_record("kernel"));
        assert!(!cfg.should_sample(42));
    }

    #[test]
    fn config_minimal() {
        let cfg = ProfileConfig::minimal();
        assert!(cfg.enabled);
        assert!(cfg.sampling_rate < 0.1);
    }

    #[test]
    fn config_category_filter() {
        let mut cfg = ProfileConfig::default_config();
        cfg.enabled_categories = vec!["kernel".into(), "memory".into()];
        assert!(cfg.should_record("kernel"));
        assert!(cfg.should_record("memory"));
        assert!(!cfg.should_record("sync"));
    }

    #[test]
    fn config_category_filter_empty_means_all() {
        let cfg = ProfileConfig::default_config();
        assert!(cfg.should_record("anything"));
    }

    #[test]
    fn config_sampling_full() {
        let cfg = ProfileConfig::default_config();
        // With rate=1.0, everything should be sampled
        for i in 0..100 {
            assert!(cfg.should_sample(i));
        }
    }

    #[test]
    fn config_sampling_zero() {
        let mut cfg = ProfileConfig::default_config();
        cfg.sampling_rate = 0.0;
        for i in 0..100 {
            assert!(!cfg.should_sample(i));
        }
    }

    #[test]
    fn config_sampling_deterministic() {
        let cfg = ProfileConfig { sampling_rate: 0.5, ..ProfileConfig::default_config() };
        let r1: Vec<bool> = (0..100).map(|i| cfg.should_sample(i)).collect();
        let r2: Vec<bool> = (0..100).map(|i| cfg.should_sample(i)).collect();
        assert_eq!(r1, r2, "sampling should be deterministic");
    }

    #[test]
    fn config_validate_bad_rate() {
        let mut cfg = ProfileConfig::default_config();
        cfg.sampling_rate = 1.5;
        assert!(cfg.validate().is_err());

        cfg.sampling_rate = -0.1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_display() {
        let cfg = ProfileConfig::default_config();
        let s = cfg.to_string();
        assert!(s.contains("enabled=true"));
        assert!(s.contains("sampling=100%"));
    }

    #[test]
    fn config_default_trait() {
        let cfg: ProfileConfig = Default::default();
        assert!(cfg.enabled);
    }

    // -- ScopedTimer --------------------------------------------------------

    #[test]
    fn scoped_timer_basic() {
        let mut timer = ScopedTimer::start("test", "kernel");
        std::thread::sleep(Duration::from_millis(5));
        let d = timer.stop();
        assert!(d >= Duration::from_millis(1));
        assert_eq!(timer.name(), "test");
        assert_eq!(timer.category(), "kernel");
    }

    #[test]
    fn scoped_timer_elapsed_before_stop() {
        let timer = ScopedTimer::start("running", "misc");
        let _ = timer.elapsed(); // Should not panic
    }

    #[test]
    fn scoped_timer_into_span() {
        let session_start = Instant::now();
        std::thread::sleep(Duration::from_millis(2));
        let timer = ScopedTimer::start("op", "compute");
        std::thread::sleep(Duration::from_millis(5));
        let span = timer.into_span(session_start);
        assert_eq!(span.name, "op");
        assert!(span.start_us > 0); // started after session_start
        assert!(span.duration_us() > 0);
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn concurrent_spans_in_tree() {
        let mut tree = ProfileTree::new();
        let root = tree.add_root(ProfileSpan::new("root", 0, 1000, "t"));
        // Two overlapping children
        tree.add_child(root, ProfileSpan::new("a", 0, 600, "k"));
        tree.add_child(root, ProfileSpan::new("b", 300, 900, "k"));
        // Overlapping children are valid (concurrent execution)
        assert_eq!(tree.children(root).len(), 2);
    }

    #[test]
    fn deeply_nested_tree() {
        let mut tree = ProfileTree::new();
        let mut parent = tree.add_root(ProfileSpan::new("d0", 0, 10000, "t"));
        for d in 1..20 {
            parent = tree.add_child(
                parent,
                ProfileSpan::new(format!("d{d}"), d as u64, 10000 - d as u64, "t"),
            );
        }
        assert_eq!(tree.max_depth(), 19);
    }

    #[test]
    fn kernel_profile_many_dispatches() {
        let mut kp = KernelProfile::new("stress");
        for i in 0..10_000 {
            kp.record(i % 100);
        }
        assert_eq!(kp.dispatch_count, 10_000);
        assert!(kp.p99_us() > 0);
    }

    #[test]
    fn memory_profile_saturating_dealloc() {
        let mut mp = MemoryProfile::new();
        mp.allocate(100, 0, "x");
        // Force dealloc of more than allocated (shouldn't underflow)
        mp.deallocate(0, 10);
        assert_eq!(mp.current_bytes(), 0); // saturating sub
    }

    #[test]
    fn escape_json_basic() {
        assert_eq!(escape_json("hello"), "hello");
        assert_eq!(escape_json("a\"b"), "a\\\"b");
        assert_eq!(escape_json("a\\b"), "a\\\\b");
        assert_eq!(escape_json("a\nb"), "a\\nb");
    }

    // -- Property-style tests -----------------------------------------------

    #[test]
    fn property_child_duration_leq_parent_random() {
        // For any tree where children are within parent bounds,
        // child sum ≤ parent duration.
        let mut tree = ProfileTree::new();
        let root = tree.add_root(ProfileSpan::new("root", 0, 10000, "t"));
        // Non-overlapping children
        tree.add_child(root, ProfileSpan::new("c1", 0, 2500, "k"));
        tree.add_child(root, ProfileSpan::new("c2", 2500, 5000, "k"));
        tree.add_child(root, ProfileSpan::new("c3", 5000, 7500, "k"));
        tree.add_child(root, ProfileSpan::new("c4", 7500, 10000, "k"));

        let parent_dur = tree.get(root).unwrap().duration_us();
        let child_sum = tree.child_duration_sum_us(root);
        assert!(child_sum <= parent_dur);
    }

    #[test]
    fn property_peak_never_decreases() {
        let mut mp = MemoryProfile::new();
        let mut prev_peak = 0u64;
        for i in 0..50 {
            let id = mp.allocate(100 + i * 10, i, format!("a{i}"));
            assert!(mp.peak_bytes() >= prev_peak);
            prev_peak = mp.peak_bytes();
            if i % 3 == 0 {
                mp.deallocate(id, i + 1);
            }
            assert!(mp.peak_bytes() >= prev_peak);
        }
    }

    #[test]
    fn property_live_count_equals_alloc_minus_dealloc() {
        let mut mp = MemoryProfile::new();
        let mut live_ids = Vec::new();
        for i in 0..20 {
            let id = mp.allocate(64, i, format!("b{i}"));
            live_ids.push(id);
        }
        assert_eq!(mp.live_count(), 20);
        for id in live_ids.iter().take(10) {
            mp.deallocate(*id, 100);
        }
        assert_eq!(mp.live_count(), 10);
    }

    #[test]
    fn property_total_recorded_never_decreases() {
        let mut lp = LiveProfiler::new(5);
        let mut prev = 0;
        for i in 0..20u64 {
            lp.record(ProfileSpan::new(format!("s{i}"), i * 10, (i + 1) * 10, "k"));
            assert!(lp.total_recorded() >= prev);
            prev = lp.total_recorded();
        }
    }

    #[test]
    fn property_sampling_rate_fraction() {
        // With sampling_rate = 0.5, roughly half should be sampled
        let cfg = ProfileConfig { sampling_rate: 0.5, ..ProfileConfig::default_config() };
        let sampled = (0..10_000u64).filter(|&i| cfg.should_sample(i)).count();
        // Allow wide tolerance: between 20% and 80%
        assert!(
            sampled > 2000 && sampled < 8000,
            "expected ~50% sampled, got {sampled}/10000"
        );
    }
}
