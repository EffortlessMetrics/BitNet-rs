use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

// ---------------------------------------------------------------------------
// Span ID generation
// ---------------------------------------------------------------------------

static NEXT_SPAN_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a profiling span.
pub type SpanId = u64;

fn next_span_id() -> SpanId {
    NEXT_SPAN_ID.fetch_add(1, Ordering::Relaxed)
}

fn now_us() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| {
        #[allow(clippy::cast_possible_truncation)]
        let us = d.as_micros() as u64;
        us
    })
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls what the profiler records.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ProfilerConfig {
    pub enabled: bool,
    pub max_events: usize,
    pub track_memory: bool,
    pub track_transfers: bool,
    pub track_kernels: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_events: 100_000,
            track_memory: true,
            track_transfers: true,
            track_kernels: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Event types
// ---------------------------------------------------------------------------

/// Category of a profiling event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum EventCategory {
    KernelLaunch,
    MemoryTransfer,
    MemoryAlloc,
    MemoryFree,
    Synchronization,
    QueueSubmit,
    PipelineCompile,
    Custom(String),
}

/// Metadata attached to a single profiling event.
#[derive(Debug, Clone, Default, Serialize)]
pub struct EventMetadata {
    pub device: Option<String>,
    pub kernel_name: Option<String>,
    pub bytes_transferred: Option<u64>,
    pub workgroup_size: Option<[u32; 3]>,
    pub grid_size: Option<[u32; 3]>,
    pub bandwidth_gbps: Option<f64>,
    pub occupancy: Option<f64>,
}

/// A single profiling event recorded by [`GpuProfiler`].
#[derive(Debug, Clone, Serialize)]
pub struct ProfileEvent {
    pub id: SpanId,
    pub name: String,
    pub category: EventCategory,
    pub start_us: u64,
    pub end_us: Option<u64>,
    pub metadata: EventMetadata,
}

impl ProfileEvent {
    fn duration_us(&self) -> u64 {
        self.end_us.unwrap_or(self.start_us).saturating_sub(self.start_us)
    }
}

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

/// Aggregated statistics for a single kernel name.
#[derive(Debug, Clone, Serialize)]
pub struct KernelStats {
    pub name: String,
    pub call_count: u32,
    pub total_us: u64,
    pub avg_us: f64,
    pub min_us: u64,
    pub max_us: u64,
    pub total_flops: u64,
    pub effective_bandwidth_gbps: f64,
}

/// Summary report produced by [`GpuProfiler::generate_report`].
#[derive(Debug, Clone, Serialize)]
pub struct ProfileReport {
    pub total_time_us: u64,
    pub kernel_time_us: u64,
    pub transfer_time_us: u64,
    pub idle_time_us: u64,
    pub top_kernels: Vec<KernelStats>,
    pub memory_high_water_mark: u64,
    pub transfer_volume: u64,
    pub events_count: usize,
}

impl ProfileReport {
    /// Human-readable summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "Total: {}us | Kernel: {}us | Transfer: {}us | Idle: {}us",
            self.total_time_us, self.kernel_time_us, self.transfer_time_us, self.idle_time_us,
        );
        let _ = writeln!(
            s,
            "Events: {} | Memory HWM: {} bytes | Transfer vol: {} bytes",
            self.events_count, self.memory_high_water_mark, self.transfer_volume,
        );
        if !self.top_kernels.is_empty() {
            s.push_str("Top kernels:\n");
            for k in &self.top_kernels {
                let _ = writeln!(
                    s,
                    "  {} — calls:{} total:{}us avg:{:.1}us",
                    k.name, k.call_count, k.total_us, k.avg_us,
                );
            }
        }
        s
    }
}

// ---------------------------------------------------------------------------
// GpuProfiler
// ---------------------------------------------------------------------------

/// Records GPU profiling events and produces aggregate reports.
pub struct GpuProfiler {
    events: Vec<ProfileEvent>,
    active_spans: Vec<SpanId>,
    config: ProfilerConfig,
    current_memory: u64,
    peak_memory: u64,
}

impl GpuProfiler {
    /// Create a new profiler with the given configuration.
    #[must_use]
    pub const fn new(config: ProfilerConfig) -> Self {
        Self {
            events: Vec::new(),
            active_spans: Vec::new(),
            config,
            current_memory: 0,
            peak_memory: 0,
        }
    }

    /// Begin a profiling span. Returns a [`SpanId`] to pass to
    /// [`end_span`](Self::end_span).
    pub fn begin_span(&mut self, name: impl Into<String>, category: EventCategory) -> SpanId {
        if !self.config.enabled {
            return 0;
        }
        if self.events.len() >= self.config.max_events {
            return 0;
        }
        let id = next_span_id();
        let event = ProfileEvent {
            id,
            name: name.into(),
            category,
            start_us: now_us(),
            end_us: None,
            metadata: EventMetadata::default(),
        };
        self.events.push(event);
        self.active_spans.push(id);
        id
    }

    /// Close a previously opened span.
    pub fn end_span(&mut self, id: SpanId) {
        if !self.config.enabled || id == 0 {
            return;
        }
        let now = now_us();
        if let Some(ev) = self.events.iter_mut().rev().find(|e| e.id == id) {
            ev.end_us = Some(now);
        }
        self.active_spans.retain(|&s| s != id);
    }

    /// Record a completed kernel execution.
    pub fn record_kernel(
        &mut self,
        name: impl Into<String>,
        duration_us: u64,
        metadata: EventMetadata,
    ) {
        if !self.config.enabled || !self.config.track_kernels {
            return;
        }
        if self.events.len() >= self.config.max_events {
            return;
        }
        let start = now_us().saturating_sub(duration_us);
        let id = next_span_id();
        self.events.push(ProfileEvent {
            id,
            name: name.into(),
            category: EventCategory::KernelLaunch,
            start_us: start,
            end_us: Some(start + duration_us),
            metadata,
        });
    }

    /// Record a completed memory transfer.
    pub fn record_transfer(&mut self, bytes: u64, duration_us: u64) {
        if !self.config.enabled || !self.config.track_transfers {
            return;
        }
        if self.events.len() >= self.config.max_events {
            return;
        }
        let start = now_us().saturating_sub(duration_us);
        let id = next_span_id();
        let bandwidth_gbps = if duration_us > 0 {
            #[allow(clippy::cast_precision_loss)]
            let bw = (bytes as f64) / (duration_us as f64 * 1e-6) / 1e9;
            Some(bw)
        } else {
            None
        };
        self.events.push(ProfileEvent {
            id,
            name: "transfer".to_string(),
            category: EventCategory::MemoryTransfer,
            start_us: start,
            end_us: Some(start + duration_us),
            metadata: EventMetadata {
                bytes_transferred: Some(bytes),
                bandwidth_gbps,
                ..EventMetadata::default()
            },
        });
    }

    /// Record a memory allocation (for high-water-mark tracking).
    pub fn record_alloc(&mut self, bytes: u64) {
        if !self.config.enabled || !self.config.track_memory {
            return;
        }
        self.current_memory = self.current_memory.saturating_add(bytes);
        if self.current_memory > self.peak_memory {
            self.peak_memory = self.current_memory;
        }
        if self.events.len() >= self.config.max_events {
            return;
        }
        let id = next_span_id();
        let now = now_us();
        self.events.push(ProfileEvent {
            id,
            name: "alloc".to_string(),
            category: EventCategory::MemoryAlloc,
            start_us: now,
            end_us: Some(now),
            metadata: EventMetadata { bytes_transferred: Some(bytes), ..EventMetadata::default() },
        });
    }

    /// Record a memory free.
    pub fn record_free(&mut self, bytes: u64) {
        if !self.config.enabled || !self.config.track_memory {
            return;
        }
        self.current_memory = self.current_memory.saturating_sub(bytes);
        if self.events.len() >= self.config.max_events {
            return;
        }
        let id = next_span_id();
        let now = now_us();
        self.events.push(ProfileEvent {
            id,
            name: "free".to_string(),
            category: EventCategory::MemoryFree,
            start_us: now,
            end_us: Some(now),
            metadata: EventMetadata { bytes_transferred: Some(bytes), ..EventMetadata::default() },
        });
    }

    /// Return a read-only view of all recorded events.
    #[must_use]
    pub fn events(&self) -> &[ProfileEvent] {
        &self.events
    }

    /// Return currently active (un-ended) spans.
    #[must_use]
    pub fn active_spans(&self) -> &[SpanId] {
        &self.active_spans
    }

    /// Generate an aggregated [`ProfileReport`].
    #[must_use]
    pub fn generate_report(&self) -> ProfileReport {
        let mut kernel_time_us: u64 = 0;
        let mut transfer_time_us: u64 = 0;
        let mut transfer_volume: u64 = 0;

        // Aggregate per-kernel stats
        let mut kernel_map: HashMap<String, Vec<u64>> = HashMap::new();
        let mut kernel_bw: HashMap<String, Vec<f64>> = HashMap::new();

        for ev in &self.events {
            let dur = ev.duration_us();
            match &ev.category {
                EventCategory::KernelLaunch => {
                    kernel_time_us = kernel_time_us.saturating_add(dur);
                    kernel_map.entry(ev.name.clone()).or_default().push(dur);
                    if let Some(bw) = ev.metadata.bandwidth_gbps {
                        kernel_bw.entry(ev.name.clone()).or_default().push(bw);
                    }
                }
                EventCategory::MemoryTransfer => {
                    transfer_time_us = transfer_time_us.saturating_add(dur);
                    if let Some(b) = ev.metadata.bytes_transferred {
                        transfer_volume = transfer_volume.saturating_add(b);
                    }
                }
                _ => {}
            }
        }

        let total_time_us = kernel_time_us.saturating_add(transfer_time_us);
        let idle_time_us = 0; // would need wall-clock span to compute

        let mut top_kernels: Vec<KernelStats> = kernel_map
            .into_iter()
            .map(|(name, durations)| {
                let call_count = u32::try_from(durations.len()).unwrap_or(u32::MAX);
                let total_us: u64 = durations.iter().sum();
                #[allow(clippy::cast_precision_loss)]
                let avg_us = total_us as f64 / durations.len() as f64;
                let min_us = durations.iter().copied().min().unwrap_or(0);
                let max_us = durations.iter().copied().max().unwrap_or(0);
                let bw_samples = kernel_bw.get(&name);
                #[allow(clippy::cast_precision_loss)]
                let effective_bandwidth_gbps =
                    bw_samples.map_or(0.0, |v| v.iter().sum::<f64>() / v.len() as f64);
                KernelStats {
                    name,
                    call_count,
                    total_us,
                    avg_us,
                    min_us,
                    max_us,
                    total_flops: 0,
                    effective_bandwidth_gbps,
                }
            })
            .collect();

        // Sort descending by total time
        top_kernels.sort_by(|a, b| b.total_us.cmp(&a.total_us));

        ProfileReport {
            total_time_us,
            kernel_time_us,
            transfer_time_us,
            idle_time_us,
            top_kernels,
            memory_high_water_mark: self.peak_memory,
            transfer_volume,
            events_count: self.events.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Exporters
// ---------------------------------------------------------------------------

/// Exports profiling events in Chrome `chrome://tracing` JSON format.
pub struct ChromeTraceExporter;

#[derive(Serialize)]
struct ChromeTraceEvent {
    name: String,
    cat: String,
    ph: &'static str,
    ts: u64,
    dur: u64,
    pid: u32,
    tid: u32,
    args: HashMap<String, serde_json::Value>,
}

impl ChromeTraceExporter {
    /// Export events as a JSON array string compatible with
    /// `chrome://tracing`.
    #[must_use]
    pub fn export(events: &[ProfileEvent]) -> String {
        let trace_events: Vec<ChromeTraceEvent> = events
            .iter()
            .filter(|e| e.end_us.is_some())
            .map(|e| {
                let mut args = HashMap::new();
                if let Some(ref dev) = e.metadata.device {
                    args.insert("device".to_string(), serde_json::Value::String(dev.clone()));
                }
                if let Some(bytes) = e.metadata.bytes_transferred {
                    args.insert("bytes".to_string(), serde_json::json!(bytes));
                }
                if let Some(bw) = e.metadata.bandwidth_gbps {
                    args.insert("bandwidth_gbps".to_string(), serde_json::json!(bw));
                }
                if let Some(occ) = e.metadata.occupancy {
                    args.insert("occupancy".to_string(), serde_json::json!(occ));
                }
                if let Some(wg) = e.metadata.workgroup_size {
                    args.insert("workgroup_size".to_string(), serde_json::json!(wg));
                }
                if let Some(gs) = e.metadata.grid_size {
                    args.insert("grid_size".to_string(), serde_json::json!(gs));
                }
                ChromeTraceEvent {
                    name: e.name.clone(),
                    cat: format!("{:?}", e.category),
                    ph: "X",
                    ts: e.start_us,
                    dur: e.duration_us(),
                    pid: 1,
                    tid: 1,
                    args,
                }
            })
            .collect();
        serde_json::to_string_pretty(&trace_events).unwrap_or_else(|_| "[]".to_string())
    }
}

/// Exports a [`ProfileReport`] as CSV.
pub struct CsvExporter;

impl CsvExporter {
    /// Export the kernel statistics from a report as CSV.
    #[must_use]
    pub fn export(report: &ProfileReport) -> String {
        let mut csv = String::from(
            "name,call_count,total_us,avg_us,min_us,max_us,\
             total_flops,effective_bandwidth_gbps\n",
        );
        for k in &report.top_kernels {
            let _ = writeln!(
                csv,
                "{},{},{},{:.1},{},{},{},{:.3}",
                k.name,
                k.call_count,
                k.total_us,
                k.avg_us,
                k.min_us,
                k.max_us,
                k.total_flops,
                k.effective_bandwidth_gbps,
            );
        }
        csv
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_profiler() -> GpuProfiler {
        GpuProfiler::new(ProfilerConfig::default())
    }

    // -- span tracking -----------------------------------------------------

    #[test]
    fn begin_span_returns_nonzero_id() {
        let mut p = default_profiler();
        let id = p.begin_span("k1", EventCategory::KernelLaunch);
        assert_ne!(id, 0);
    }

    #[test]
    fn end_span_closes_event() {
        let mut p = default_profiler();
        let id = p.begin_span("k1", EventCategory::KernelLaunch);
        p.end_span(id);
        let ev = p.events().iter().find(|e| e.id == id).unwrap();
        assert!(ev.end_us.is_some());
    }

    #[test]
    fn active_spans_tracks_open_spans() {
        let mut p = default_profiler();
        let id = p.begin_span("k1", EventCategory::KernelLaunch);
        assert!(p.active_spans().contains(&id));
        p.end_span(id);
        assert!(!p.active_spans().contains(&id));
    }

    #[test]
    fn nested_spans() {
        let mut p = default_profiler();
        let outer = p.begin_span("outer", EventCategory::Custom("test".into()));
        let inner = p.begin_span("inner", EventCategory::KernelLaunch);
        assert_eq!(p.active_spans().len(), 2);
        p.end_span(inner);
        assert_eq!(p.active_spans().len(), 1);
        p.end_span(outer);
        assert!(p.active_spans().is_empty());
    }

    #[test]
    fn span_ids_are_unique() {
        let mut p = default_profiler();
        let id1 = p.begin_span("a", EventCategory::KernelLaunch);
        let id2 = p.begin_span("b", EventCategory::KernelLaunch);
        let id3 = p.begin_span("c", EventCategory::MemoryTransfer);
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn end_span_idempotent() {
        let mut p = default_profiler();
        let id = p.begin_span("k", EventCategory::KernelLaunch);
        p.end_span(id);
        p.end_span(id); // second end is a no-op (already removed)
        assert!(p.active_spans().is_empty());
    }

    #[test]
    fn begin_span_records_event_name_and_category() {
        let mut p = default_profiler();
        let id = p.begin_span("my_kernel", EventCategory::PipelineCompile);
        let ev = p.events().iter().find(|e| e.id == id).unwrap();
        assert_eq!(ev.name, "my_kernel");
        assert_eq!(ev.category, EventCategory::PipelineCompile);
    }

    // -- record_kernel -----------------------------------------------------

    #[test]
    fn record_kernel_creates_event() {
        let mut p = default_profiler();
        p.record_kernel("matmul", 500, EventMetadata::default());
        assert_eq!(p.events().len(), 1);
        assert_eq!(p.events()[0].name, "matmul");
        assert_eq!(p.events()[0].duration_us(), 500);
    }

    #[test]
    fn record_kernel_with_metadata() {
        let mut p = default_profiler();
        let meta = EventMetadata {
            device: Some("GPU:0".into()),
            kernel_name: Some("matmul_f16".into()),
            workgroup_size: Some([256, 1, 1]),
            grid_size: Some([1024, 1, 1]),
            ..EventMetadata::default()
        };
        p.record_kernel("matmul", 100, meta);
        let ev = &p.events()[0];
        assert_eq!(ev.metadata.device.as_deref(), Some("GPU:0"));
        assert_eq!(ev.metadata.workgroup_size, Some([256, 1, 1]));
    }

    #[test]
    fn record_kernel_disabled_tracking() {
        let cfg = ProfilerConfig { track_kernels: false, ..ProfilerConfig::default() };
        let mut p = GpuProfiler::new(cfg);
        p.record_kernel("k", 100, EventMetadata::default());
        assert!(p.events().is_empty());
    }

    // -- record_transfer ---------------------------------------------------

    #[test]
    fn record_transfer_creates_event() {
        let mut p = default_profiler();
        p.record_transfer(1024, 50);
        assert_eq!(p.events().len(), 1);
        assert_eq!(p.events()[0].category, EventCategory::MemoryTransfer);
    }

    #[test]
    fn record_transfer_computes_bandwidth() {
        let mut p = default_profiler();
        let bytes: u64 = 1_000_000_000; // 1 GB
        let duration_us: u64 = 1_000_000; // 1 s
        p.record_transfer(bytes, duration_us);
        let bw = p.events()[0].metadata.bandwidth_gbps.unwrap();
        assert!((bw - 1.0).abs() < 0.01, "expected ~1 GB/s, got {bw}");
    }

    #[test]
    fn record_transfer_disabled_tracking() {
        let cfg = ProfilerConfig { track_transfers: false, ..ProfilerConfig::default() };
        let mut p = GpuProfiler::new(cfg);
        p.record_transfer(1024, 10);
        assert!(p.events().is_empty());
    }

    #[test]
    fn record_transfer_zero_duration() {
        let mut p = default_profiler();
        p.record_transfer(1024, 0);
        assert!(p.events()[0].metadata.bandwidth_gbps.is_none());
    }

    // -- memory tracking ---------------------------------------------------

    #[test]
    fn memory_high_water_mark_single_alloc() {
        let mut p = default_profiler();
        p.record_alloc(1024);
        let report = p.generate_report();
        assert_eq!(report.memory_high_water_mark, 1024);
    }

    #[test]
    fn memory_high_water_mark_alloc_free_alloc() {
        let mut p = default_profiler();
        p.record_alloc(1000);
        p.record_alloc(500);
        // peak = 1500
        p.record_free(1000);
        // current = 500
        p.record_alloc(200);
        // current = 700, peak still 1500
        let report = p.generate_report();
        assert_eq!(report.memory_high_water_mark, 1500);
    }

    #[test]
    fn memory_free_does_not_underflow() {
        let mut p = default_profiler();
        p.record_alloc(100);
        p.record_free(200); // more than allocated
        let report = p.generate_report();
        assert_eq!(report.memory_high_water_mark, 100);
    }

    #[test]
    fn memory_tracking_disabled() {
        let cfg = ProfilerConfig { track_memory: false, ..ProfilerConfig::default() };
        let mut p = GpuProfiler::new(cfg);
        p.record_alloc(4096);
        let report = p.generate_report();
        assert_eq!(report.memory_high_water_mark, 0);
        assert!(p.events().is_empty());
    }

    // -- disabled profiler -------------------------------------------------

    #[test]
    fn disabled_profiler_skips_spans() {
        let cfg = ProfilerConfig { enabled: false, ..ProfilerConfig::default() };
        let mut p = GpuProfiler::new(cfg);
        let id = p.begin_span("k", EventCategory::KernelLaunch);
        assert_eq!(id, 0);
        assert!(p.events().is_empty());
    }

    #[test]
    fn disabled_profiler_skips_kernel() {
        let cfg = ProfilerConfig { enabled: false, ..ProfilerConfig::default() };
        let mut p = GpuProfiler::new(cfg);
        p.record_kernel("k", 100, EventMetadata::default());
        assert!(p.events().is_empty());
    }

    #[test]
    fn disabled_profiler_skips_transfer() {
        let cfg = ProfilerConfig { enabled: false, ..ProfilerConfig::default() };
        let mut p = GpuProfiler::new(cfg);
        p.record_transfer(1024, 10);
        assert!(p.events().is_empty());
    }

    #[test]
    fn disabled_profiler_end_span_noop() {
        let cfg = ProfilerConfig { enabled: false, ..ProfilerConfig::default() };
        let mut p = GpuProfiler::new(cfg);
        p.end_span(0); // should not panic
        p.end_span(42);
    }

    // -- event overflow ----------------------------------------------------

    #[test]
    fn max_events_limits_recording() {
        let cfg = ProfilerConfig { max_events: 3, ..ProfilerConfig::default() };
        let mut p = GpuProfiler::new(cfg);
        p.record_kernel("a", 10, EventMetadata::default());
        p.record_kernel("b", 20, EventMetadata::default());
        p.record_kernel("c", 30, EventMetadata::default());
        p.record_kernel("d", 40, EventMetadata::default()); // dropped
        assert_eq!(p.events().len(), 3);
    }

    #[test]
    fn max_events_limits_spans() {
        let cfg = ProfilerConfig { max_events: 2, ..ProfilerConfig::default() };
        let mut p = GpuProfiler::new(cfg);
        let _id1 = p.begin_span("a", EventCategory::KernelLaunch);
        let _id2 = p.begin_span("b", EventCategory::KernelLaunch);
        let id3 = p.begin_span("c", EventCategory::KernelLaunch);
        assert_eq!(id3, 0);
        assert_eq!(p.events().len(), 2);
    }

    #[test]
    fn max_events_limits_transfers() {
        let cfg = ProfilerConfig { max_events: 1, ..ProfilerConfig::default() };
        let mut p = GpuProfiler::new(cfg);
        p.record_transfer(100, 10);
        p.record_transfer(200, 20); // dropped
        assert_eq!(p.events().len(), 1);
    }

    // -- report generation -------------------------------------------------

    #[test]
    fn empty_profiler_report() {
        let p = default_profiler();
        let report = p.generate_report();
        assert_eq!(report.total_time_us, 0);
        assert_eq!(report.kernel_time_us, 0);
        assert_eq!(report.transfer_time_us, 0);
        assert_eq!(report.events_count, 0);
        assert!(report.top_kernels.is_empty());
    }

    #[test]
    fn report_kernel_time_aggregation() {
        let mut p = default_profiler();
        p.record_kernel("k1", 100, EventMetadata::default());
        p.record_kernel("k2", 200, EventMetadata::default());
        let report = p.generate_report();
        assert_eq!(report.kernel_time_us, 300);
    }

    #[test]
    fn report_transfer_time_aggregation() {
        let mut p = default_profiler();
        p.record_transfer(1024, 50);
        p.record_transfer(2048, 100);
        let report = p.generate_report();
        assert_eq!(report.transfer_time_us, 150);
    }

    #[test]
    fn report_transfer_volume() {
        let mut p = default_profiler();
        p.record_transfer(1024, 10);
        p.record_transfer(2048, 20);
        let report = p.generate_report();
        assert_eq!(report.transfer_volume, 3072);
    }

    #[test]
    fn report_events_count() {
        let mut p = default_profiler();
        p.record_kernel("a", 10, EventMetadata::default());
        p.record_transfer(100, 5);
        p.record_alloc(256);
        let report = p.generate_report();
        assert_eq!(report.events_count, 3);
    }

    #[test]
    fn report_top_kernels_sorted_by_total_time() {
        let mut p = default_profiler();
        p.record_kernel("fast", 10, EventMetadata::default());
        p.record_kernel("slow", 500, EventMetadata::default());
        p.record_kernel("medium", 100, EventMetadata::default());
        let report = p.generate_report();
        assert_eq!(report.top_kernels.len(), 3);
        assert_eq!(report.top_kernels[0].name, "slow");
        assert_eq!(report.top_kernels[1].name, "medium");
        assert_eq!(report.top_kernels[2].name, "fast");
    }

    #[test]
    fn report_same_name_kernels_aggregated() {
        let mut p = default_profiler();
        p.record_kernel("matmul", 100, EventMetadata::default());
        p.record_kernel("matmul", 200, EventMetadata::default());
        p.record_kernel("matmul", 150, EventMetadata::default());
        let report = p.generate_report();
        assert_eq!(report.top_kernels.len(), 1);
        let k = &report.top_kernels[0];
        assert_eq!(k.name, "matmul");
        assert_eq!(k.call_count, 3);
        assert_eq!(k.total_us, 450);
        assert!((k.avg_us - 150.0).abs() < 0.1);
        assert_eq!(k.min_us, 100);
        assert_eq!(k.max_us, 200);
    }

    #[test]
    fn report_kernel_stats_min_max() {
        let mut p = default_profiler();
        p.record_kernel("k", 50, EventMetadata::default());
        p.record_kernel("k", 300, EventMetadata::default());
        p.record_kernel("k", 100, EventMetadata::default());
        let report = p.generate_report();
        let k = &report.top_kernels[0];
        assert_eq!(k.min_us, 50);
        assert_eq!(k.max_us, 300);
    }

    #[test]
    fn report_total_time_is_kernel_plus_transfer() {
        let mut p = default_profiler();
        p.record_kernel("k", 200, EventMetadata::default());
        p.record_transfer(1024, 100);
        let report = p.generate_report();
        assert_eq!(report.total_time_us, 300);
    }

    // -- report summary string ---------------------------------------------

    #[test]
    fn report_summary_contains_totals() {
        let mut p = default_profiler();
        p.record_kernel("k", 100, EventMetadata::default());
        let report = p.generate_report();
        let s = report.summary();
        assert!(s.contains("Kernel: 100us"));
        assert!(s.contains("Events: 1"));
    }

    #[test]
    fn report_summary_contains_kernel_names() {
        let mut p = default_profiler();
        p.record_kernel("my_kernel", 500, EventMetadata::default());
        let report = p.generate_report();
        let s = report.summary();
        assert!(s.contains("my_kernel"));
    }

    #[test]
    fn empty_report_summary_no_panic() {
        let p = default_profiler();
        let report = p.generate_report();
        let s = report.summary();
        assert!(s.contains("Total: 0us"));
    }

    // -- Chrome trace export -----------------------------------------------

    #[test]
    fn chrome_trace_empty_events() {
        let json = ChromeTraceExporter::export(&[]);
        assert_eq!(json.trim(), "[]");
    }

    #[test]
    fn chrome_trace_valid_json() {
        let mut p = default_profiler();
        p.record_kernel("k", 100, EventMetadata::default());
        let json = ChromeTraceExporter::export(p.events());
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_array());
    }

    #[test]
    fn chrome_trace_has_required_fields() {
        let mut p = default_profiler();
        p.record_kernel("test_kernel", 200, EventMetadata::default());
        let json = ChromeTraceExporter::export(p.events());
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
        let ev = &parsed[0];
        assert_eq!(ev["name"], "test_kernel");
        assert_eq!(ev["ph"], "X");
        assert!(ev["ts"].is_number());
        assert_eq!(ev["dur"], 200);
        assert!(ev["pid"].is_number());
        assert!(ev["tid"].is_number());
    }

    #[test]
    fn chrome_trace_includes_metadata_args() {
        let mut p = default_profiler();
        let meta = EventMetadata {
            device: Some("GPU:0".into()),
            bytes_transferred: Some(4096),
            bandwidth_gbps: Some(12.5),
            occupancy: Some(0.75),
            workgroup_size: Some([64, 1, 1]),
            grid_size: Some([128, 1, 1]),
            ..EventMetadata::default()
        };
        p.record_kernel("k", 100, meta);
        let json = ChromeTraceExporter::export(p.events());
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        let args = &parsed[0]["args"];
        assert_eq!(args["device"], "GPU:0");
        assert_eq!(args["bytes"], 4096);
        assert!((args["bandwidth_gbps"].as_f64().unwrap() - 12.5).abs() < 0.01);
        assert!((args["occupancy"].as_f64().unwrap() - 0.75).abs() < 0.01);
    }

    #[test]
    fn chrome_trace_skips_open_spans() {
        let mut p = default_profiler();
        let _id = p.begin_span("open", EventCategory::KernelLaunch);
        // span never ended
        let json = ChromeTraceExporter::export(p.events());
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn chrome_trace_multiple_events() {
        let mut p = default_profiler();
        p.record_kernel("k1", 100, EventMetadata::default());
        p.record_kernel("k2", 200, EventMetadata::default());
        p.record_transfer(1024, 50);
        let json = ChromeTraceExporter::export(p.events());
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 3);
    }

    #[test]
    fn chrome_trace_category_string() {
        let mut p = default_profiler();
        p.record_kernel("k", 50, EventMetadata::default());
        let json = ChromeTraceExporter::export(p.events());
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed[0]["cat"], "KernelLaunch");
    }

    #[test]
    fn chrome_trace_custom_category() {
        let mut p = default_profiler();
        let id = p.begin_span("c", EventCategory::Custom("my_cat".into()));
        p.end_span(id);
        let json = ChromeTraceExporter::export(p.events());
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert!(parsed[0]["cat"].as_str().unwrap().contains("my_cat"));
    }

    // -- CSV export --------------------------------------------------------

    #[test]
    fn csv_export_header() {
        let p = default_profiler();
        let report = p.generate_report();
        let csv = CsvExporter::export(&report);
        assert!(csv.starts_with("name,call_count,total_us,avg_us,"));
    }

    #[test]
    fn csv_export_kernel_row() {
        let mut p = default_profiler();
        p.record_kernel("matmul", 100, EventMetadata::default());
        let report = p.generate_report();
        let csv = CsvExporter::export(&report);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2); // header + 1 row
        assert!(lines[1].starts_with("matmul,1,100,"));
    }

    #[test]
    fn csv_export_multiple_kernels() {
        let mut p = default_profiler();
        p.record_kernel("a", 10, EventMetadata::default());
        p.record_kernel("b", 20, EventMetadata::default());
        let report = p.generate_report();
        let csv = CsvExporter::export(&report);
        assert_eq!(csv.lines().count(), 3); // header + 2 rows
    }

    #[test]
    fn csv_export_empty_report() {
        let p = default_profiler();
        let report = p.generate_report();
        let csv = CsvExporter::export(&report);
        assert_eq!(csv.lines().count(), 1); // header only
    }

    // -- bandwidth calculation ---------------------------------------------

    #[test]
    fn bandwidth_high_throughput() {
        let mut p = default_profiler();
        // 10 GB in 1 second → 10 GB/s
        p.record_transfer(10_000_000_000, 1_000_000);
        let bw = p.events()[0].metadata.bandwidth_gbps.unwrap();
        assert!((bw - 10.0).abs() < 0.1, "expected ~10 GB/s, got {bw}");
    }

    #[test]
    fn bandwidth_small_transfer() {
        let mut p = default_profiler();
        // 1 KB in 1 us → 1e9 B/s = 1 GB/s
        p.record_transfer(1000, 1);
        let bw = p.events()[0].metadata.bandwidth_gbps.unwrap();
        assert!((bw - 1.0).abs() < 0.01, "expected ~1 GB/s, got {bw}");
    }

    // -- concurrent-safe span IDs ------------------------------------------

    #[test]
    fn span_ids_monotonically_increasing() {
        let mut p = default_profiler();
        let id1 = p.begin_span("a", EventCategory::KernelLaunch);
        let id2 = p.begin_span("b", EventCategory::KernelLaunch);
        assert!(id2 > id1);
    }

    #[test]
    fn span_ids_unique_across_profilers() {
        let mut p1 = default_profiler();
        let mut p2 = default_profiler();
        let id1 = p1.begin_span("a", EventCategory::KernelLaunch);
        let id2 = p2.begin_span("b", EventCategory::KernelLaunch);
        assert_ne!(id1, id2);
    }

    // -- mixed workloads ---------------------------------------------------

    #[test]
    fn mixed_kernels_and_transfers_report() {
        let mut p = default_profiler();
        p.record_kernel("matmul", 1000, EventMetadata::default());
        p.record_kernel("softmax", 200, EventMetadata::default());
        p.record_transfer(4096, 50);
        p.record_transfer(8192, 100);
        p.record_alloc(16384);
        let report = p.generate_report();
        assert_eq!(report.kernel_time_us, 1200);
        assert_eq!(report.transfer_time_us, 150);
        assert_eq!(report.transfer_volume, 4096 + 8192);
        assert_eq!(report.memory_high_water_mark, 16384);
        assert_eq!(report.events_count, 5);
        assert_eq!(report.top_kernels.len(), 2);
        assert_eq!(report.top_kernels[0].name, "matmul");
    }

    #[test]
    fn profiler_default_config() {
        let cfg = ProfilerConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.max_events, 100_000);
        assert!(cfg.track_memory);
        assert!(cfg.track_transfers);
        assert!(cfg.track_kernels);
    }

    #[test]
    fn event_duration_no_end() {
        let ev = ProfileEvent {
            id: 1,
            name: "test".into(),
            category: EventCategory::KernelLaunch,
            start_us: 1000,
            end_us: None,
            metadata: EventMetadata::default(),
        };
        assert_eq!(ev.duration_us(), 0);
    }

    #[test]
    fn event_duration_with_end() {
        let ev = ProfileEvent {
            id: 1,
            name: "test".into(),
            category: EventCategory::KernelLaunch,
            start_us: 1000,
            end_us: Some(1500),
            metadata: EventMetadata::default(),
        };
        assert_eq!(ev.duration_us(), 500);
    }

    #[test]
    fn record_alloc_creates_event() {
        let mut p = default_profiler();
        p.record_alloc(2048);
        assert_eq!(p.events().len(), 1);
        assert_eq!(p.events()[0].category, EventCategory::MemoryAlloc);
        assert_eq!(p.events()[0].metadata.bytes_transferred, Some(2048));
    }

    #[test]
    fn record_free_creates_event() {
        let mut p = default_profiler();
        p.record_alloc(1024);
        p.record_free(512);
        assert_eq!(p.events().len(), 2);
        assert_eq!(p.events()[1].category, EventCategory::MemoryFree);
    }

    #[test]
    fn kernel_stats_effective_bandwidth() {
        let mut p = default_profiler();
        let meta = EventMetadata { bandwidth_gbps: Some(25.0), ..EventMetadata::default() };
        p.record_kernel("k", 100, meta.clone());
        p.record_kernel("k", 200, meta);
        let report = p.generate_report();
        let k = &report.top_kernels[0];
        assert!((k.effective_bandwidth_gbps - 25.0).abs() < 0.01);
    }

    #[test]
    fn event_category_equality() {
        assert_eq!(EventCategory::KernelLaunch, EventCategory::KernelLaunch);
        assert_ne!(EventCategory::KernelLaunch, EventCategory::MemoryTransfer);
        assert_eq!(EventCategory::Custom("x".into()), EventCategory::Custom("x".into()));
        assert_ne!(EventCategory::Custom("x".into()), EventCategory::Custom("y".into()));
    }

    #[test]
    fn profiler_report_serializable() {
        let mut p = default_profiler();
        p.record_kernel("k", 100, EventMetadata::default());
        let report = p.generate_report();
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("kernel_time_us"));
    }

    #[test]
    fn chrome_trace_dur_matches_event() {
        let mut p = default_profiler();
        p.record_kernel("k", 777, EventMetadata::default());
        let json = ChromeTraceExporter::export(p.events());
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed[0]["dur"], 777);
    }

    #[test]
    fn csv_export_bandwidth_field() {
        let mut p = default_profiler();
        let meta = EventMetadata { bandwidth_gbps: Some(42.5), ..EventMetadata::default() };
        p.record_kernel("k", 100, meta);
        let report = p.generate_report();
        let csv = CsvExporter::export(&report);
        assert!(csv.contains("42.500"));
    }

    #[test]
    fn max_events_alloc_still_tracks_memory() {
        let cfg = ProfilerConfig { max_events: 0, ..ProfilerConfig::default() };
        let mut p = GpuProfiler::new(cfg);
        p.record_alloc(1024);
        // Event not recorded but memory tracking still works
        assert!(p.events().is_empty());
        let report = p.generate_report();
        assert_eq!(report.memory_high_water_mark, 1024);
    }

    #[test]
    fn synchronization_event() {
        let mut p = default_profiler();
        let id = p.begin_span("sync", EventCategory::Synchronization);
        p.end_span(id);
        let ev = p.events().iter().find(|e| e.id == id).unwrap();
        assert_eq!(ev.category, EventCategory::Synchronization);
    }

    #[test]
    fn queue_submit_event() {
        let mut p = default_profiler();
        let id = p.begin_span("submit", EventCategory::QueueSubmit);
        p.end_span(id);
        let ev = p.events().iter().find(|e| e.id == id).unwrap();
        assert_eq!(ev.category, EventCategory::QueueSubmit);
    }
}
